#include "expr_eval.hpp"

#include "mymyr/successor/conditions.hpp"
#include "mymyr/task/numeric.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mymyr
{
using namespace formalism;

namespace
{
[[noreturn]] void invalid(const std::string& what) { throw std::invalid_argument("mymyr: condition: " + what); }

/// Copies the expression tree rooted at T.exprs[e] into the pools (children first), returning its index in exprs.
u32 copy_expr(const TaskData& T, u32 e, std::vector<Expr>& exprs, std::vector<Term>& expr_terms)
{
    Expr x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Add:
        case ExprOp::Sub:
        case ExprOp::Mul:
        case ExprOp::Div:
            x.a = copy_expr(T, x.a, exprs, expr_terms);
            x.b = copy_expr(T, x.b, exprs, expr_terms);
            break;
        case ExprOp::Neg: x.a = copy_expr(T, x.a, exprs, expr_terms); break;
        case ExprOp::Function:
        {
            const auto terms = TaskData::slice(T.terms, x.terms);
            x.terms = Range{static_cast<u32>(expr_terms.size()), static_cast<u32>(terms.size())};
            expr_terms.insert(expr_terms.end(), terms.begin(), terms.end());
            break;
        }
        case ExprOp::Number: break;
    }
    exprs.push_back(x);
    return static_cast<u32>(exprs.size() - 1);
}

void copy_constraints(const TaskData& T, const Condition& cond, std::vector<NumericConstraint>& constraints,
                      std::vector<Expr>& exprs, std::vector<Term>& expr_terms)
{
    for (const NumericConstraint& k : T.constraints_of(cond))
    {
        NumericConstraint n = k;
        n.lhs = copy_expr(T, k.lhs, exprs, expr_terms);
        n.rhs = copy_expr(T, k.rhs, exprs, expr_terms);
        constraints.push_back(n);
    }
}

/// The literals (each once: normalization lists a nullary literal both as a literal and as a nullary ground literal)
/// and numeric constraints of a normalized condition.
void copy_condition(const TaskData& T, const Condition& cond, ConjunctiveCondition& c)
{
    for (const Literal& l : T.literals_of(cond))
    {
        const auto terms = T.terms_of(l);
        ConjunctiveCondition::Literal lit{l.pred, l.positive, std::vector<Term>(terms.begin(), terms.end())};
        const bool seen = std::ranges::any_of(c.literals, [&](const ConjunctiveCondition::Literal& o)
                                              { return o.predicate == lit.predicate && o.positive == lit.positive && o.terms == lit.terms; });
        if (!seen)
            c.literals.push_back(std::move(lit));
    }
    copy_constraints(T, cond, c.constraints, c.exprs, c.expr_terms);
}

void add_variables(const TaskData& T, Range params, ConjunctiveCondition& c)
{
    for (const Parameter& p : TaskData::slice(T.params, params))
    {
        const auto types = TaskData::slice(T.type_ids, p.types);
        c.variables.push_back({std::string(T.str(p.name)), std::vector<TypeId>(types.begin(), types.end())});
    }
}

const char* cmp_text(Comparator c)
{
    switch (c)
    {
        case Comparator::Eq: return "=";
        case Comparator::Ne: return "!=";
        case Comparator::Lt: return "<";
        case Comparator::Le: return "<=";
        case Comparator::Gt: return ">";
        case Comparator::Ge: return ">=";
    }
    return "?";
}

/// The shortest text that reads back as exactly v.
std::string number_text(f64 v)
{
    std::array<char, 64> buf{};
    const auto r = std::to_chars(buf.data(), buf.data() + buf.size(), v);
    return std::string(buf.data(), r.ptr);
}

bool same_expr(const Expr& a, const Expr& b)
{
    return a.op == b.op && std::bit_cast<u64>(a.value) == std::bit_cast<u64>(b.value) && a.func == b.func &&
           a.terms.begin == b.terms.begin && a.terms.count == b.terms.count && a.a == b.a && a.b == b.b;
}

bool same_constraints(std::span<const NumericConstraint> a, std::span<const NumericConstraint> b)
{
    if (a.size() != b.size())
        return false;
    for (usize i = 0; i < a.size(); ++i)
        if (a[i].cmp != b[i].cmp || a[i].lhs != b[i].lhs || a[i].rhs != b[i].rhs)
            return false;
    return true;
}

bool same_exprs(std::span<const Expr> a, std::span<const Expr> b)
{
    if (a.size() != b.size())
        return false;
    for (usize i = 0; i < a.size(); ++i)
        if (!same_expr(a[i], b[i]))
            return false;
    return true;
}

/// Checks literal predicates, arities and terms, and the expression trees reachable from the constraints (every node
/// once; a node met again on the current path is a cycle).
template<class TermOk>
void check_exprs(const Task& task, std::span<const NumericConstraint> constraints, std::span<const Expr> exprs,
                 std::span<const Term> expr_terms, TermOk&& term_ok)
{
    const TaskData& T = task.data();
    const plan::Numeric& N = task.numeric();
    std::vector<u8> state(exprs.size(), 0);  // 0 unseen, 1 on the path, 2 checked
    auto check = [&](auto&& self, u32 e, const std::string& where) -> void
    {
        if (e >= exprs.size())
            invalid(where + ": expression index " + std::to_string(e) + " out of range");
        if (state[e] == 2)
            return;
        if (state[e] == 1)
            invalid(where + ": the expressions contain a cycle");
        state[e] = 1;
        const Expr& x = exprs[e];
        switch (x.op)
        {
            case ExprOp::Number: break;
            case ExprOp::Function:
            {
                if (x.func.v >= T.functions.size())
                    invalid(where + ": function index " + std::to_string(x.func.v) + " out of range");
                const Function& f = T.functions[x.func.v];
                const std::string name(T.str(f.name));
                if (f.kind == FuncKind::Auxiliary)
                    invalid(where + ": " + name + " is the total-cost function, which no state holds");
                if (x.func.v >= N.tables.size() || N.tables[x.func.v].arity != f.arity)
                    invalid(where + ": the task has no value table for " + name);
                if (x.terms.count != f.arity || x.terms.end() > expr_terms.size() || x.terms.end() < x.terms.begin)
                    invalid(where + ": " + name + " takes " + std::to_string(f.arity) +
                            " arguments (terms out of range or of the wrong count)");
                for (u32 j = x.terms.begin; j < x.terms.end(); ++j)
                    term_ok(expr_terms[j], where);
                break;
            }
            case ExprOp::Add:
            case ExprOp::Sub:
            case ExprOp::Mul:
            case ExprOp::Div:
                self(self, x.a, where);
                self(self, x.b, where);
                break;
            case ExprOp::Neg: self(self, x.a, where); break;
            default: invalid(where + ": unknown expression operator");
        }
        state[e] = 2;
    };
    for (usize i = 0; i < constraints.size(); ++i)
    {
        const std::string where = "numeric constraint " + std::to_string(i);
        if (static_cast<u8>(constraints[i].cmp) > static_cast<u8>(Comparator::Ge))
            invalid(where + ": unknown comparator");
        check(check, constraints[i].lhs, where);
        check(check, constraints[i].rhs, where);
    }
}

void check_literal(const TaskData& T, usize i, PredicateId pred, usize terms)
{
    const std::string where = "literal " + std::to_string(i);
    if (pred.v >= T.predicates.size())
        invalid(where + ": predicate index " + std::to_string(pred.v) + " out of range");
    const Predicate& p = T.predicates[pred.v];
    if (terms != p.arity)
        invalid(where + ": " + std::string(T.str(p.name)) + " takes " + std::to_string(p.arity) + " arguments, got " +
                std::to_string(terms));
}

/// PDDL text of expression e; `term` writes a term.
template<class TermText>
std::string expr_text(const TaskData& T, std::span<const Expr> exprs, std::span<const Term> expr_terms, u32 e, u32 depth,
                      TermText&& term)
{
    if (e >= exprs.size() || depth > exprs.size())
        return "<expression?>";
    const Expr& x = exprs[e];
    auto sub = [&](u32 c) { return expr_text(T, exprs, expr_terms, c, depth + 1, term); };
    switch (x.op)
    {
        case ExprOp::Number: return number_text(x.value);
        case ExprOp::Function:
        {
            std::string s = "(" + (x.func.v < T.functions.size() ? std::string(T.str(T.functions[x.func.v].name)) : "<function?>");
            for (u32 j = x.terms.begin; j < x.terms.end() && j < expr_terms.size(); ++j)
                s += " " + term(expr_terms[j]);
            return s + ")";
        }
        case ExprOp::Add: return "(+ " + sub(x.a) + " " + sub(x.b) + ")";
        case ExprOp::Sub: return "(- " + sub(x.a) + " " + sub(x.b) + ")";
        case ExprOp::Mul: return "(* " + sub(x.a) + " " + sub(x.b) + ")";
        case ExprOp::Div: return "(/ " + sub(x.a) + " " + sub(x.b) + ")";
        case ExprOp::Neg: return "(- " + sub(x.a) + ")";
    }
    return "<expression?>";
}

std::string object_text(const TaskData& T, ObjectId o)
{
    return o.v < T.num_objects() ? std::string(T.str(T.objects[o.v].name)) : "<object?>";
}

std::string predicate_text(const TaskData& T, PredicateId p)
{
    return p.v < T.predicates.size() ? std::string(T.str(T.predicates[p.v].name)) : "<predicate?>";
}

// ------------------------------------------------------------------------------------------------ constraint parser

std::string lower(std::string_view s)
{
    std::string out(s);
    for (char& c : out)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

/// A recursive-descent reader of one PDDL numeric constraint over the task's functions and objects.
class ConstraintParser
{
public:
    using VariableOf = std::function<Term(const std::string&)>;

    ConstraintParser(const Task& task, std::string_view text, std::vector<Expr>& exprs, std::vector<Term>& terms,
                     VariableOf variable)
        : m_t(task.data()), m_text(text), m_exprs(exprs), m_terms(terms), m_variable(std::move(variable))
    {
        std::string cur;
        for (char c : text)
        {
            if (c == '(' || c == ')' || std::isspace(static_cast<unsigned char>(c)))
            {
                if (!cur.empty())
                    m_tok.push_back(lower(cur));
                cur.clear();
                if (c == '(' || c == ')')
                    m_tok.emplace_back(1, c);
            }
            else
                cur.push_back(c);
        }
        if (!cur.empty())
            m_tok.push_back(lower(cur));
    }

    NumericConstraint constraint()
    {
        // roll the pools back if the text is malformed, so that a failed call leaves the condition unchanged
        const usize e0 = m_exprs.size(), t0 = m_terms.size();
        try
        {
            expect("(");
            const std::string op = next("a comparator");
            NumericConstraint k;
            if (op == "=")
                k.cmp = Comparator::Eq;
            else if (op == "!=")
                k.cmp = Comparator::Ne;
            else if (op == "<")
                k.cmp = Comparator::Lt;
            else if (op == "<=")
                k.cmp = Comparator::Le;
            else if (op == ">")
                k.cmp = Comparator::Gt;
            else if (op == ">=")
                k.cmp = Comparator::Ge;
            else
                fail("expected a comparator (=, !=, <, <=, >, >=), got '" + op + "'");
            k.lhs = expr();
            k.rhs = expr();
            expect(")");
            if (m_pos != m_tok.size())
                fail("unexpected '" + m_tok[m_pos] + "' after the constraint");
            return k;
        }
        catch (...)
        {
            m_exprs.resize(e0);
            m_terms.resize(t0);
            throw;
        }
    }

private:
    [[noreturn]] void fail(const std::string& what) const
    {
        throw std::invalid_argument("mymyr: constraint '" + std::string(m_text) + "': " + what);
    }
    std::string next(const char* what)
    {
        if (m_pos >= m_tok.size())
            fail(std::string("unexpected end, expected ") + what);
        return m_tok[m_pos++];
    }
    void expect(const char* tok)
    {
        const std::string t = next(tok);
        if (t != tok)
            fail("expected '" + std::string(tok) + "', got '" + t + "'");
    }
    [[nodiscard]] bool peek(const char* tok) const { return m_pos < m_tok.size() && m_tok[m_pos] == tok; }

    u32 push(const Expr& x)
    {
        m_exprs.push_back(x);
        return static_cast<u32>(m_exprs.size() - 1);
    }

    u32 expr()
    {
        const std::string t = next("an expression");
        if (t == ")")
            fail("expected an expression, got ')'");
        if (t != "(")
        {
            f64 v = 0;
            const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
            if (r.ec != std::errc{} || r.ptr != t.data() + t.size())
                fail("'" + t + "' is not a number (a function term is written \"(" + t + " ...)\")");
            Expr x;
            x.op = ExprOp::Number;
            x.value = v;
            return push(x);
        }
        const std::string head = next("an operator or a function");
        if (head == "+" || head == "-" || head == "*" || head == "/")
        {
            const ExprOp op = head == "+" ? ExprOp::Add : head == "-" ? ExprOp::Sub : head == "*" ? ExprOp::Mul : ExprOp::Div;
            u32 a = expr();
            if (peek(")"))
            {
                ++m_pos;
                if (op != ExprOp::Sub)
                    fail("'" + head + "' needs two operands");
                Expr x;
                x.op = ExprOp::Neg;
                x.a = a;
                return push(x);
            }
            while (!peek(")"))  // n-ary forms fold to the left
            {
                Expr x;
                x.op = op;
                x.a = a;
                x.b = expr();
                a = push(x);
            }
            ++m_pos;
            return a;
        }
        u32 f = 0;
        while (f < m_t.functions.size() && lower(m_t.str(m_t.functions[f].name)) != head)
            ++f;
        if (f == m_t.functions.size())
            fail("no function named '" + head + "'");
        Expr x;
        x.op = ExprOp::Function;
        x.func = FunctionId{f};
        x.terms.begin = static_cast<u32>(m_terms.size());
        while (!peek(")"))
        {
            const std::string a = next("a term or ')'");
            if (a == "(")
                fail("the arguments of '" + head + "' are objects or variables");
            m_terms.push_back(term(a));
        }
        ++m_pos;
        x.terms.count = static_cast<u32>(m_terms.size()) - x.terms.begin;
        if (x.terms.count != m_t.functions[f].arity)
            fail("'" + head + "' takes " + std::to_string(m_t.functions[f].arity) + " arguments, got " +
                 std::to_string(x.terms.count));
        return push(x);
    }

    Term term(const std::string& a)
    {
        if (a[0] == '?')
        {
            if (!m_variable)
                fail("'" + a + "': a ground constraint has no variables");
            try
            {
                return m_variable(a.substr(1));
            }
            catch (const std::invalid_argument& e)
            {
                fail(e.what());
            }
        }
        for (u32 o = 0; o < m_t.num_objects(); ++o)
            if (lower(m_t.str(m_t.objects[o].name)) == a)
                return object_term(ObjectId{o});
        fail("no object named '" + a + "'");
    }

    const TaskData& m_t;
    std::string_view m_text;
    std::vector<Expr>& m_exprs;
    std::vector<Term>& m_terms;
    VariableOf m_variable;
    std::vector<std::string> m_tok;
    usize m_pos = 0;
};

// ------------------------------------------------------------------------------------------------ evaluation

/// Evaluates ground atoms in one state; the axiom closure is computed on the first derived atom.
class AtomTruth
{
public:
    AtomTruth(const Task& task, StateView s) : m_task(task), m_s(s) {}

    bool operator()(PredicateId pred, std::span<const ObjectId> objects)
    {
        const TaskData& T = m_task.data();
        if (pred.v >= T.predicates.size())
            throw std::invalid_argument("mymyr: holds: predicate index " + std::to_string(pred.v) + " out of range");
        const u32 arity = T.predicates[pred.v].arity;
        if (objects.size() != arity)
            throw std::invalid_argument("mymyr: holds: " + std::string(T.str(T.predicates[pred.v].name)) + " takes " +
                                        std::to_string(arity) + " objects, got " + std::to_string(objects.size()));
        m_args.resize(arity);
        for (u32 i = 0; i < arity; ++i)
        {
            if (objects[i].v >= T.num_objects())
                throw std::invalid_argument("mymyr: holds: object index " + std::to_string(objects[i].v) + " out of range");
            m_args[i] = objects[i].v;
        }
        const plan::Compiled& C = m_task.compiled();
        if (C.kinds[pred.v] == PredKind::Static)
        {
            const plan::StaticRelation& R = C.statics[pred.v];
            u64 key = 0;
            for (u32 i = 0; i < arity; ++i)
                key += R.position_table(i, C.num_objects)[m_args[i]];
            return R.contains(key);
        }
        const CanonicalLayout& L = C.layout;
        if (!L.has_predicate(pred.v))
            return false;
        const CanonicalAtom c = L.encode(pred.v, m_args.data());
        if (c >= L.total)
            return false;  // outside the reachable domains
        const u32 slot = m_task.atoms().find(c);
        if (slot == AtomIndex::k_empty)
            return false;  // no state has produced it
        if (C.kinds[pred.v] == PredKind::Fluent)
            return bits::test(m_s.w, m_s.nw, slot);
        if (!m_prepared)
        {
            m_lease = m_task.workspace();
            m_succ = &m_lease->successors();
            m_succ->prepare(m_s);
            m_prepared = true;
        }
        const detail::Engine& e = m_succ->engine();
        return bits::test(e.derived(), e.derived_words(), slot);
    }

private:
    const Task& m_task;
    StateView m_s;
    WorkspaceLease m_lease;  // taken when a derived atom is first asked for
    Successors* m_succ = nullptr;
    bool m_prepared = false;
    std::vector<u32> m_args;
};
}  // namespace

// ================================================================================================ ConjunctiveCondition

ConjunctiveCondition ConjunctiveCondition::precondition(const TaskData& T, SchemaId schema)
{
    if (schema.v >= T.schemas.size())
        throw std::invalid_argument("mymyr: precondition: schema index out of range");
    const Schema& S = T.schemas[schema.v];
    ConjunctiveCondition c;
    add_variables(T, S.params, c);
    copy_condition(T, S.precondition, c);
    return c;
}

ConjunctiveCondition ConjunctiveCondition::effect_condition(const TaskData& T, SchemaId schema, u32 effect)
{
    if (schema.v >= T.schemas.size())
        throw std::invalid_argument("mymyr: effect_condition: schema index out of range");
    const Schema& S = T.schemas[schema.v];
    if (effect < S.effects.begin || effect >= S.effects.end())
        throw std::invalid_argument("mymyr: effect_condition: effect " + std::to_string(effect) + " is not one of " +
                                    std::string(T.str(S.name)) + "'s");
    const ConditionalEffect& ce = T.conditional_effects[effect];
    ConjunctiveCondition c;
    add_variables(T, S.params, c);
    add_variables(T, ce.extra_params, c);
    copy_condition(T, ce.condition, c);
    return c;
}

ConjunctiveCondition ConjunctiveCondition::axiom_body(const TaskData& T, u32 axiom)
{
    if (axiom >= T.axioms.size())
        throw std::invalid_argument("mymyr: axiom_body: axiom index out of range");
    ConjunctiveCondition c;
    add_variables(T, T.axioms[axiom].params, c);
    copy_condition(T, T.axioms[axiom].body, c);
    return c;
}

ConjunctiveCondition ConjunctiveCondition::goal(const TaskData& T)
{
    ConjunctiveCondition c;
    copy_condition(T, T.goal, c);
    return c;
}

void ConjunctiveCondition::add_constraint(const Task& task, std::string_view pddl)
{
    auto variable = [this](const std::string& name) -> Term
    {
        for (u32 v = 0; v < arity(); ++v)
            if (lower(variables[v].name) == name)
                return static_cast<Term>(v);
        throw std::invalid_argument("no variable named '?" + name + "'");
    };
    constraints.push_back(ConstraintParser(task, pddl, exprs, expr_terms, variable).constraint());
}

GroundCondition ConjunctiveCondition::ground(std::span<const ObjectId> binding) const
{
    if (binding.size() != arity())
        throw std::invalid_argument("mymyr: ground: the binding has " + std::to_string(binding.size()) +
                                    " objects, the condition " + std::to_string(arity()) + " variables");
    auto object = [&](Term t) -> ObjectId
    {
        if (is_object(t))
            return term_object(t);
        if (term_parameter(t) >= arity())
            invalid("variable " + std::to_string(t) + " out of range (" + std::to_string(arity()) + " variables)");
        return binding[term_parameter(t)];
    };
    for (const Equality& e : equalities)
        if ((object(e.lhs) == object(e.rhs)) != e.positive)
            throw std::invalid_argument("mymyr: ground: the binding violates an equality of the condition");
    GroundCondition g;
    g.literals.reserve(literals.size());
    for (const Literal& l : literals)
    {
        GroundLiteral x{{l.predicate, {}}, l.positive};
        x.atom.objects.reserve(l.terms.size());
        for (Term t : l.terms)
            x.atom.objects.push_back(object(t));
        g.literals.push_back(std::move(x));
    }
    g.constraints = constraints;
    g.exprs = exprs;
    g.expr_terms.reserve(expr_terms.size());
    for (Term t : expr_terms)
        g.expr_terms.push_back(object_term(object(t)));
    return g;
}

void ConjunctiveCondition::validate(const Task& task) const
{
    const TaskData& T = task.data();
    const u32 n = T.num_objects(), V = arity();
    auto term_ok = [&](Term t, const std::string& where)
    {
        if (is_object(t) ? term_object(t).v >= n : term_parameter(t) >= V)
            invalid(where + ": " + (is_object(t) ? "object index " + std::to_string(term_object(t).v) + " out of range"
                                                 : "variable " + std::to_string(t) + " out of range (" + std::to_string(V) +
                                                       " variables)"));
    };
    for (u32 v = 0; v < V; ++v)
        for (TypeId t : variables[v].types)
            if (t.v >= T.types.size())
                invalid("variable " + std::to_string(v) + ": type index " + std::to_string(t.v) + " out of range");
    for (usize i = 0; i < literals.size(); ++i)
    {
        check_literal(T, i, literals[i].predicate, literals[i].terms.size());
        for (Term t : literals[i].terms)
            term_ok(t, "literal " + std::to_string(i));
    }
    for (usize i = 0; i < equalities.size(); ++i)
    {
        term_ok(equalities[i].lhs, "equality " + std::to_string(i));
        term_ok(equalities[i].rhs, "equality " + std::to_string(i));
    }
    check_exprs(task, constraints, exprs, expr_terms, term_ok);
}

std::string ConjunctiveCondition::str(const TaskData& T) const
{
    auto var = [&](u32 v)
    { return "?" + (v < variables.size() && !variables[v].name.empty() ? variables[v].name : "x" + std::to_string(v)); };
    auto term = [&](Term t) { return is_object(t) ? object_text(T, term_object(t)) : var(term_parameter(t)); };
    std::string out = "(";
    for (u32 v = 0; v < arity(); ++v)
    {
        out += (v ? " " : "") + var(v);
        const auto& ts = variables[v].types;
        auto tname = [&](TypeId t) { return t.v < T.types.size() ? std::string(T.str(T.types[t.v].name)) : "<type?>"; };
        if (ts.size() == 1)
            out += " - " + tname(ts[0]);
        else if (ts.size() > 1)
        {
            out += " - (either";
            for (TypeId t : ts)
                out += " " + tname(t);
            out += ")";
        }
    }
    out += ") (and";
    for (const Literal& l : literals)
    {
        std::string a = "(" + predicate_text(T, l.predicate);
        for (Term t : l.terms)
            a += " " + term(t);
        a += ")";
        out += " " + (l.positive ? a : "(not " + a + ")");
    }
    for (const Equality& e : equalities)
        out += std::string(" (") + (e.positive ? "= " : "!= ") + term(e.lhs) + " " + term(e.rhs) + ")";
    for (const NumericConstraint& k : constraints)
        out += std::string(" (") + cmp_text(k.cmp) + " " + expr_text(T, exprs, expr_terms, k.lhs, 0, term) + " " +
               expr_text(T, exprs, expr_terms, k.rhs, 0, term) + ")";
    return out + ")";
}

bool operator==(const ConjunctiveCondition& a, const ConjunctiveCondition& b)
{
    if (a.variables.size() != b.variables.size() || a.literals.size() != b.literals.size() ||
        a.equalities.size() != b.equalities.size() || a.expr_terms != b.expr_terms ||
        !same_constraints(a.constraints, b.constraints) || !same_exprs(a.exprs, b.exprs))
        return false;
    for (usize i = 0; i < a.variables.size(); ++i)
        if (a.variables[i].name != b.variables[i].name || a.variables[i].types != b.variables[i].types)
            return false;
    for (usize i = 0; i < a.literals.size(); ++i)
        if (a.literals[i].predicate != b.literals[i].predicate || a.literals[i].positive != b.literals[i].positive ||
            a.literals[i].terms != b.literals[i].terms)
            return false;
    for (usize i = 0; i < a.equalities.size(); ++i)
        if (a.equalities[i].lhs != b.equalities[i].lhs || a.equalities[i].rhs != b.equalities[i].rhs ||
            a.equalities[i].positive != b.equalities[i].positive)
            return false;
    return true;
}

ConjunctiveCondition ConjunctiveCondition::precondition(const Task& task, SchemaId schema) { return precondition(task.data(), schema); }
ConjunctiveCondition ConjunctiveCondition::effect_condition(const Task& task, SchemaId schema, u32 effect)
{
    return effect_condition(task.data(), schema, effect);
}
ConjunctiveCondition ConjunctiveCondition::axiom_body(const Task& task, u32 axiom) { return axiom_body(task.data(), axiom); }
ConjunctiveCondition ConjunctiveCondition::goal(const Task& task) { return goal(task.data()); }
std::string ConjunctiveCondition::str(const Task& task) const { return str(task.data()); }

// ================================================================================================ GroundCondition

GroundCondition GroundCondition::goal(const Task& task) { return goal(task.data()); }
std::string GroundCondition::str(const Task& task) const { return str(task.data()); }

GroundCondition GroundCondition::goal(const TaskData& T)
{
    GroundCondition g;
    for (const Literal& l : T.literals_of(T.goal))
    {
        GroundLiteral x{{l.pred, {}}, l.positive};
        for (Term t : T.terms_of(l))
            x.atom.objects.push_back(term_object(t));
        if (std::ranges::find(g.literals, x) == g.literals.end())
            g.literals.push_back(std::move(x));
    }
    copy_constraints(T, T.goal, g.constraints, g.exprs, g.expr_terms);
    return g;
}

void GroundCondition::add_constraint(const Task& task, std::string_view pddl)
{
    constraints.push_back(ConstraintParser(task, pddl, exprs, expr_terms, nullptr).constraint());
}

ConjunctiveCondition GroundCondition::lift(bool add_inequalities) const
{
    ConjunctiveCondition c;
    std::unordered_map<u32, u32> var_of;
    auto variable = [&](ObjectId o) -> Term
    {
        const auto [it, fresh] = var_of.try_emplace(o.v, static_cast<u32>(var_of.size()));
        if (fresh)
            c.variables.push_back({"x" + std::to_string(it->second), {}});
        return static_cast<Term>(it->second);
    };
    c.literals.reserve(literals.size());
    for (const GroundLiteral& l : literals)
    {
        ConjunctiveCondition::Literal x{l.atom.predicate, l.positive, {}};
        x.terms.reserve(l.atom.objects.size());
        for (ObjectId o : l.atom.objects)
            x.terms.push_back(variable(o));
        c.literals.push_back(std::move(x));
    }
    c.constraints = constraints;
    c.exprs = exprs;
    c.expr_terms.reserve(expr_terms.size());
    for (Term t : expr_terms)
        c.expr_terms.push_back(is_object(t) ? variable(term_object(t)) : t);
    if (add_inequalities)
        for (u32 i = 0; i < c.arity(); ++i)
            for (u32 j = i + 1; j < c.arity(); ++j)
                c.equalities.push_back({static_cast<Term>(i), static_cast<Term>(j), false});
    return c;
}

void GroundCondition::validate(const Task& task) const
{
    const TaskData& T = task.data();
    const u32 n = T.num_objects();
    for (usize i = 0; i < literals.size(); ++i)
    {
        check_literal(T, i, literals[i].atom.predicate, literals[i].atom.objects.size());
        for (ObjectId o : literals[i].atom.objects)
            if (o.v >= n)
                invalid("literal " + std::to_string(i) + ": object index " + std::to_string(o.v) + " out of range");
    }
    auto term_ok = [&](Term t, const std::string& where)
    {
        if (!is_object(t))
            invalid(where + ": a ground condition has no variables (term " + std::to_string(t) + ")");
        if (term_object(t).v >= n)
            invalid(where + ": object index " + std::to_string(term_object(t).v) + " out of range");
    };
    check_exprs(task, constraints, exprs, expr_terms, term_ok);
}

std::string GroundCondition::str(const TaskData& T) const
{
    auto term = [&](Term t) { return is_object(t) ? object_text(T, term_object(t)) : "?" + std::to_string(t); };
    std::string out = "(and";
    for (const GroundLiteral& l : literals)
    {
        std::string a = "(" + predicate_text(T, l.atom.predicate);
        for (ObjectId o : l.atom.objects)
            a += " " + object_text(T, o);
        a += ")";
        out += " " + (l.positive ? a : "(not " + a + ")");
    }
    for (const NumericConstraint& k : constraints)
        out += std::string(" (") + cmp_text(k.cmp) + " " + expr_text(T, exprs, expr_terms, k.lhs, 0, term) + " " +
               expr_text(T, exprs, expr_terms, k.rhs, 0, term) + ")";
    return out + ")";
}

bool operator==(const GroundCondition& a, const GroundCondition& b)
{
    return a.literals == b.literals && a.expr_terms == b.expr_terms && same_constraints(a.constraints, b.constraints) &&
           same_exprs(a.exprs, b.exprs);
}

// ================================================================================================ holds

bool holds(const Task& task, StateView s, const mymyr::GroundAtom& atom)
{
    AtomTruth truth(task, s);
    return truth(atom.predicate, atom.objects);
}

bool holds(const Task& task, StateView s, const GroundLiteral& literal)
{
    AtomTruth truth(task, s);
    return truth(literal.atom.predicate, literal.atom.objects) == literal.positive;
}

bool holds(const Task& task, StateView s, const GroundCondition& condition)
{
    condition.validate(task);
    AtomTruth truth(task, s);
    // static and fluent literals first: the derived ones may need the axiom closure
    for (int derived = 0; derived < 2; ++derived)
        for (const GroundLiteral& l : condition.literals)
            if ((task.compiled().kinds[l.atom.predicate.v] == PredKind::Derived) == (derived != 0) &&
                truth(l.atom.predicate, l.atom.objects) != l.positive)
                return false;
    const plan::Numeric& N = task.numeric();
    for (const NumericConstraint& k : condition.constraints)
        if (!plan::compare(N, k.cmp, detail::eval_expr(N, condition.exprs, condition.expr_terms, k.lhs, nullptr, s.num),
                           detail::eval_expr(N, condition.exprs, condition.expr_terms, k.rhs, nullptr, s.num)))
            return false;
    return true;
}
}  // namespace mymyr
