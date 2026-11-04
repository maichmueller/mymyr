#include "mymyr/formalism/task_data.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mymyr::formalism
{
namespace
{
[[noreturn]] void invalid(const std::string& msg) { throw std::invalid_argument("TaskData: " + msg); }

template<class T>
void check_range(const std::vector<T>& pool, Range r, const char* what)
{
    if (static_cast<u64>(r.begin) + r.count > pool.size())
        invalid(std::string(what) + " range out of bounds");
}

void check_term(const TaskData& t, Term x, u32 num_params, const char* where)
{
    if (is_object(x))
    {
        if (term_object(x).v >= t.objects.size())
            invalid(std::string(where) + ": object term out of range");
    }
    else if (term_parameter(x) >= num_params)
        invalid(std::string(where) + ": parameter term " + std::to_string(x) + " >= " + std::to_string(num_params));
}

void check_literal(const TaskData& t, const Literal& l, u32 num_params, const char* where)
{
    if (l.pred.v >= t.predicates.size())
        invalid(std::string(where) + ": predicate out of range");
    check_range(t.terms, l.terms, "literal terms");
    if (l.terms.count != t.predicates[l.pred.v].arity)
        invalid(std::string(where) + ": literal arity mismatch for " + std::string(t.str(t.predicates[l.pred.v].name)));
    for (Term x : t.terms_of(l))
        check_term(t, x, num_params, where);
}

void check_expr(const TaskData& t, u32 e, u32 num_params, const char* where, u32 depth = 0)
{
    if (e >= t.exprs.size())
        invalid(std::string(where) + ": expression index out of range");
    if (depth > 10000)
        invalid(std::string(where) + ": expression too deep (cycle?)");
    const Expr& x = t.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return;
        case ExprOp::Function:
        {
            if (x.func.v >= t.functions.size())
                invalid(std::string(where) + ": function out of range");
            const Function& f = t.functions[x.func.v];
            check_range(t.terms, x.terms, "function terms");
            if (f.kind != FuncKind::Auxiliary && x.terms.count != f.arity)
                invalid(std::string(where) + ": function arity mismatch");
            for (Term tt : TaskData::slice(t.terms, x.terms))
                check_term(t, tt, num_params, where);
            return;
        }
        case ExprOp::Neg: check_expr(t, x.a, num_params, where, depth + 1); return;
        default:
            check_expr(t, x.a, num_params, where, depth + 1);
            check_expr(t, x.b, num_params, where, depth + 1);
    }
}

void check_condition(const TaskData& t, const Condition& c, u32 num_params, const char* where)
{
    check_range(t.literals, c.literals, "condition literals");
    check_range(t.constraints, c.constraints, "condition constraints");
    for (const auto& l : t.literals_of(c))
        check_literal(t, l, num_params, where);
    for (const auto& nc : t.constraints_of(c))
    {
        check_expr(t, nc.lhs, num_params, where);
        check_expr(t, nc.rhs, num_params, where);
    }
}

void check_encoding_goal(const TaskData& t);
}  // namespace

static void check_ground_atom(const TaskData& t, const GroundAtom& a, const char* where)
{
    check_range(t.object_ids, a.objects, where);
    if (a.pred.v >= t.predicates.size())
        invalid(std::string(where) + ": predicate out of range");
    if (a.objects.count != t.predicates[a.pred.v].arity)
        invalid(std::string(where) + ": arity mismatch");
    for (ObjectId o : t.objects_of(a))
        if (o.v >= t.objects.size())
            invalid(std::string(where) + ": object out of range");
}

void validate(const TaskData& t)
{
    for (const auto& p : t.predicates)
        check_range(t.params, p.params, "predicate params");
    for (const auto& a : t.static_init)
    {
        check_ground_atom(t, a, "static initial atom");
        if (t.predicates[a.pred.v].kind != PredKind::Static)
            invalid("static initial atom over a non-static predicate");
    }
    for (const auto& a : t.fluent_init)
    {
        check_ground_atom(t, a, "fluent initial atom");
        if (t.predicates[a.pred.v].kind != PredKind::Fluent)
            invalid("fluent initial atom over a non-fluent predicate");
    }
    check_condition(t, t.goal, 0, "goal");
    check_range(t.literals, t.encoding_goal, "encoding goal");
    for (const auto& l : t.encoding_goal_literals())
        check_literal(t, l, 0, "encoding goal");
    check_encoding_goal(t);
    for (const auto& s : t.schemas)
    {
        check_range(t.params, s.params, "schema params");
        check_range(t.collapsed_params, s.collapsed, "collapsed parameters");
        if (s.original_arity > s.arity())
            invalid("schema original_arity above its arity");
        for (u32 k = 0; k < s.collapsed.count; ++k)
        {
            const CollapsedParameter& c = t.collapsed_params[s.collapsed.begin + k];
            if (c.position >= s.arity() + s.collapsed.count || (k > 0 && t.collapsed_params[s.collapsed.begin + k - 1].position >= c.position))
                invalid("collapsed parameter positions must ascend within the schema's parameters before collapsing");
            if (c.param >= t.params.size())
                invalid("collapsed parameter out of range");
            if (c.representative.valid() && c.representative.v >= t.num_objects())
                invalid("collapsed parameter representative out of range");
        }
        const char* where = "schema";
        check_condition(t, s.precondition, s.arity(), where);
        check_range(t.conditional_effects, s.effects, "schema effects");
        for (const auto& ce : t.effects_of(s))
        {
            const u32 np = s.arity() + ce.extra_params.count;
            check_condition(t, ce.condition, np, "conditional effect");
            check_range(t.literals, ce.effects, "effect literals");
            for (const auto& l : TaskData::slice(t.literals, ce.effects))
            {
                check_literal(t, l, np, "effect");
                if (t.predicates[l.pred.v].kind != PredKind::Fluent)
                    invalid("effect on a non-fluent predicate " + std::string(t.str(t.predicates[l.pred.v].name)));
            }
            check_range(t.numeric_effects, ce.numeric_effects, "numeric effects");
            for (const auto& e : TaskData::slice(t.numeric_effects, ce.numeric_effects))
                check_expr(t, e.expr, np, "numeric effect");
            if (ce.auxiliary)
                check_expr(t, ce.auxiliary->expr, np, "auxiliary effect");
        }
    }
    for (const auto& a : t.axioms)
    {
        check_range(t.params, a.params, "axiom params");
        check_literal(t, a.head, a.params.count, "axiom head");
        if (t.predicates[a.head.pred.v].kind != PredKind::Derived || !a.head.positive)
            invalid("axiom head must be a positive derived literal");
        check_condition(t, a.body, a.params.count, "axiom body");
    }
    if (t.metric)
        check_expr(t, t.metric->expr, 0, "metric");
}

namespace
{
std::vector<u32> key_of(PredicateId pred, std::span<const ObjectId> objects)
{
    std::vector<u32> k{pred.v};
    for (ObjectId o : objects)
        k.push_back(o.v);
    return k;
}

/// Whether goal normalization introduced the predicate: loki names these "axiom_<k>".
bool introduced_by_normalization(std::string_view name)
{
    constexpr std::string_view prefix = "axiom_";
    return name.size() > prefix.size() && name.starts_with(prefix) &&
           std::ranges::all_of(name.substr(prefix.size()), [](char c) { return c >= '0' && c <= '9'; });
}

/// The literals of the planner's goal that the encoding goal keeps by default, and whether they are all of it.
bool default_goal(const TaskData& t, std::vector<GroundLiteralData>& out)
{
    std::vector<u8> normalized(t.predicates.size(), 0);
    for (u32 p = 0; p < t.predicates.size(); ++p)
        normalized[p] = t.predicates[p].kind == PredKind::Derived && introduced_by_normalization(t.str(t.predicates[p].name));
    bool complete = t.goal.constraints.count == 0;
    for (const Literal& l : t.literals_of(t.goal))
    {
        if (normalized[l.pred.v])
        {
            complete = false;
            continue;
        }
        GroundLiteralData g{l.pred, l.positive, {}};
        for (Term x : t.terms_of(l))
            g.objects.push_back(term_object(x));
        out.push_back(std::move(g));
    }
    return complete;
}

void sort_unique(std::vector<GroundLiteralData>& v)
{
    auto less = [](const GroundLiteralData& a, const GroundLiteralData& b)
    {
        if (a.positive != b.positive)
            return a.positive;
        if (a.pred.v != b.pred.v)
            return a.pred.v < b.pred.v;
        return std::ranges::lexicographical_compare(a.objects, b.objects, {}, &ObjectId::v, &ObjectId::v);
    };
    std::ranges::sort(v, less);
    v.erase(std::unique(v.begin(), v.end(), [&](const auto& a, const auto& b) { return !less(a, b) && !less(b, a); }), v.end());
}

/// Whether a literal of the encoding goal over a static predicate fails against the initial atoms.
bool static_goal_fails(const TaskData& t)
{
    std::set<std::vector<u32>> wanted_true, wanted_false;  // static literals by (predicate, objects): positive / negative
    std::vector<u8> is_static(t.predicates.size(), 0);
    for (const Literal& l : t.encoding_goal_literals())
        if (t.predicates[l.pred.v].kind == PredKind::Static)
        {
            is_static[l.pred.v] = 1;
            std::vector<ObjectId> objects;
            for (Term x : t.terms_of(l))
                objects.push_back(term_object(x));
            (l.positive ? wanted_true : wanted_false).insert(key_of(l.pred, objects));
        }
    if (wanted_true.empty() && wanted_false.empty())
        return false;
    // a static literal fails when its atom is absent (positive) or present (negative)
    std::set<std::vector<u32>> present;
    for (const GroundAtom& a : t.static_init)
        if (is_static[a.pred.v])
            if (auto k = key_of(a.pred, t.objects_of(a)); wanted_true.contains(k) || wanted_false.contains(k))
                present.insert(std::move(k));
    return std::ranges::any_of(wanted_true, [&](const auto& k) { return !present.contains(k); })
           || std::ranges::any_of(wanted_false, [&](const auto& k) { return present.contains(k); });
}

/// The encoding goal is sorted (positive first, predicate id, objects) without duplicates, and its unsatisfiable flag
/// matches its static literals.
void check_encoding_goal(const TaskData& t)
{
    if (!t.encoding_goal_recorded && t.encoding_goal.count != 0)
        invalid("encoding goal literals without a recorded encoding goal");
    const auto literals = t.encoding_goal_literals();
    auto key = [&](const Literal& l)
    {
        std::vector<u32> k{l.positive ? 0u : 1u, l.pred.v};
        for (Term x : t.terms_of(l))
            k.push_back(term_object(x).v);
        return k;
    };
    for (size_t i = 1; i < literals.size(); ++i)
        if (!(key(literals[i - 1]) < key(literals[i])))
            invalid("encoding goal is not sorted and unique");
    if (t.encoding_goal_unsatisfiable != static_goal_fails(t))
        invalid("encoding_goal_unsatisfiable does not match the static literals of the encoding goal");
}
}  // namespace

void set_encoding_goal(TaskData& t, std::vector<GroundLiteralData> literals, bool complete)
{
    sort_unique(literals);
    std::vector<Literal> out;
    for (const GroundLiteralData& g : literals)
    {
        Literal l;
        l.pred = g.pred;
        l.positive = g.positive;
        std::vector<Term> terms;
        for (ObjectId o : g.objects)
            terms.push_back(object_term(o));
        l.terms = TaskData::append<Term>(t.terms, terms);
        out.push_back(l);
    }
    t.encoding_goal = TaskData::append<Literal>(t.literals, out);
    t.encoding_goal_recorded = true;
    t.encoding_goal_complete = complete;
    t.encoding_goal_unsatisfiable = static_goal_fails(t);
}

void set_encoding_goal_from_normalized(TaskData& t)
{
    std::vector<GroundLiteralData> lits;
    const bool complete = default_goal(t, lits);
    set_encoding_goal(t, std::move(lits), complete);
}

bool has_default_encoding_goal(const TaskData& t)
{
    if (!t.encoding_goal_recorded)
        return true;
    std::vector<GroundLiteralData> lits;
    const bool complete = default_goal(t, lits);
    if (complete != t.encoding_goal_complete)
        return false;
    sort_unique(lits);
    const auto have = t.encoding_goal_literals();
    if (have.size() != lits.size())
        return false;
    for (size_t i = 0; i < lits.size(); ++i)
    {
        const Literal& l = have[i];
        if (l.pred != lits[i].pred || l.positive != lits[i].positive || l.terms.count != lits[i].objects.size())
            return false;
        const auto terms = t.terms_of(l);
        for (size_t j = 0; j < terms.size(); ++j)
            if (term_object(terms[j]) != lits[i].objects[j])
                return false;
    }
    return true;
}
namespace
{
/// The schema parameters (terms < arity) that a part of a schema mentions.
struct Mentions
{
    const TaskData& t;
    u32 arity;
    std::vector<u8> used;

    void term(Term x)
    {
        if (!is_object(x) && term_parameter(x) < arity)
            used[term_parameter(x)] = 1;
    }
    void terms(Range r)
    {
        for (Term x : TaskData::slice(t.terms, r))
            term(x);
    }
    void expr(u32 e)
    {
        const Expr& x = t.exprs[e];
        switch (x.op)
        {
        case ExprOp::Number: break;
        case ExprOp::Function: terms(x.terms); break;
        case ExprOp::Neg: expr(x.a); break;
        default:
            expr(x.a);
            expr(x.b);
        }
    }
    void condition(const Condition& c)
    {
        for (const Literal& l : t.literals_of(c))
            terms(l.terms);
        for (const NumericConstraint& n : t.constraints_of(c))
        {
            expr(n.lhs);
            expr(n.rhs);
        }
    }
    void numeric_effect(const NumericEffect& e)
    {
        terms(e.terms);
        expr(e.expr);
    }
};

/// Copies the parts of a schema into new pool entries with every parameter term renumbered.
struct Renumber
{
    TaskData& t;
    std::vector<Term> map;  // parameter index -> new index (for every schema and forall parameter)

    Range terms(Range r)
    {
        std::vector<Term> out(TaskData::slice(t.terms, r).begin(), TaskData::slice(t.terms, r).end());
        for (Term& x : out)
            if (!is_object(x))
                x = map[term_parameter(x)];
        return TaskData::append<Term>(t.terms, out);
    }
    u32 expr(u32 e)
    {
        Expr x = t.exprs[e];
        switch (x.op)
        {
        case ExprOp::Number: break;
        case ExprOp::Function: x.terms = terms(x.terms); break;
        case ExprOp::Neg: x.a = expr(x.a); break;
        default:
            x.a = expr(x.a);
            x.b = expr(x.b);
        }
        t.exprs.push_back(x);
        return static_cast<u32>(t.exprs.size() - 1);
    }
    Literal literal(Literal l)
    {
        l.terms = terms(l.terms);
        return l;
    }
    Range literals(Range r, const std::vector<u8>* drop = nullptr)
    {
        std::vector<Literal> out;
        for (u32 i = 0; i < r.count; ++i)
            if (drop == nullptr || !(*drop)[i])
                out.push_back(literal(t.literals[r.begin + i]));
        return TaskData::append<Literal>(t.literals, out);
    }
    Condition condition(const Condition& c, const std::vector<u8>* drop = nullptr)
    {
        Condition out;
        out.literals = literals(c.literals, drop);
        std::vector<NumericConstraint> cs;
        for (u32 i = 0; i < c.constraints.count; ++i)
        {
            NumericConstraint n = t.constraints[c.constraints.begin + i];
            n.lhs = expr(n.lhs);
            n.rhs = expr(n.rhs);
            cs.push_back(n);
        }
        out.constraints = TaskData::append<NumericConstraint>(t.constraints, cs);
        return out;
    }
    NumericEffect numeric_effect(NumericEffect e)
    {
        e.terms = terms(e.terms);
        e.expr = expr(e.expr);
        return e;
    }
};
}  // namespace

void collapse_unused_parameters(TaskData& t, std::span<const u8> type_predicate)
{
    if (t.unused_parameters_collapsed)
        return;
    t.unused_parameters_collapsed = true;
    const u32 n_objects = t.num_objects();
    // members[p]: the objects of static unary predicate p (filled on first use)
    std::vector<std::vector<u8>> members(t.predicates.size());
    auto member_table = [&](u32 p) -> const std::vector<u8>&
    {
        std::vector<u8>& m = members[p];
        if (m.empty())
        {
            m.assign(n_objects, 0);
            for (const GroundAtom& a : t.static_init)
                if (a.pred.v == p && a.objects.count == 1)
                    m[t.object_ids[a.objects.begin].v] = 1;
        }
        return m;
    };
    auto type_literal = [&](const Literal& l)
    {
        return l.positive && l.terms.count == 1 && !is_object(t.terms[l.terms.begin]) && l.pred.v < type_predicate.size() &&
               type_predicate[l.pred.v] && t.predicates[l.pred.v].kind == PredKind::Static;
    };
    for (u32 si = 0; si < t.schemas.size(); ++si)
    {
        const Schema s = t.schemas[si];
        const u32 n = s.arity();
        if (n == 0)
            continue;
        Mentions m{t, n, std::vector<u8>(n, 0)};
        std::vector<u8> is_type(s.precondition.literals.count, 0);
        for (u32 i = 0; i < s.precondition.literals.count; ++i)
        {
            const Literal& l = t.literals[s.precondition.literals.begin + i];
            if (type_literal(l) && term_parameter(t.terms[l.terms.begin]) < n)
                is_type[i] = 1;
            else
                m.terms(l.terms);
        }
        for (const NumericConstraint& c : t.constraints_of(s.precondition))
        {
            m.expr(c.lhs);
            m.expr(c.rhs);
        }
        u32 max_extra = 0;
        for (const ConditionalEffect& ce : t.effects_of(s))
        {
            max_extra = std::max(max_extra, ce.extra_params.count);
            m.condition(ce.condition);
            for (const Literal& l : TaskData::slice(t.literals, ce.effects))
                m.terms(l.terms);
            for (const NumericEffect& e : TaskData::slice(t.numeric_effects, ce.numeric_effects))
                m.numeric_effect(e);
            if (ce.auxiliary)
                m.numeric_effect(*ce.auxiliary);
        }
        const u32 n_collapsed = static_cast<u32>(std::count(m.used.begin(), m.used.end(), u8{0}));
        if (n_collapsed == 0)
            continue;

        Renumber r{t, std::vector<Term>(n + max_extra, 0)};
        std::vector<Parameter> kept;
        std::vector<CollapsedParameter> collapsed;
        u32 original_arity = 0;
        for (u32 i = 0, next = 0; i < n; ++i)
        {
            const Parameter& param = t.params[s.params.begin + i];
            if (m.used[i])
            {
                r.map[i] = static_cast<Term>(next++);
                kept.push_back(param);
                original_arity += i < s.original_arity ? 1 : 0;
                continue;
            }
            // the representative: the first object that every type literal of the parameter holds for
            std::vector<u8> fits(n_objects, 1);
            for (u32 k = 0; k < is_type.size(); ++k)
            {
                const Literal& l = t.literals[s.precondition.literals.begin + k];
                if (!is_type[k] || term_parameter(t.terms[l.terms.begin]) != i)
                    continue;
                const std::vector<u8>& in = member_table(l.pred.v);
                for (u32 o = 0; o < n_objects; ++o)
                    fits[o] &= in[o];
            }
            CollapsedParameter c;
            c.param = s.params.begin + i;
            c.position = i;
            c.written = i < s.original_arity;
            if (auto it = std::find(fits.begin(), fits.end(), u8{1}); it != fits.end())
                c.representative = ObjectId{static_cast<u32>(it - fits.begin())};
            collapsed.push_back(c);
        }
        for (u32 j = 0; j < max_extra; ++j)
            r.map[n + j] = static_cast<Term>(n - n_collapsed + j);
        // the type literals of the collapsed parameters leave the precondition
        std::vector<u8> drop(is_type.size(), 0);
        for (u32 k = 0; k < is_type.size(); ++k)
            drop[k] = is_type[k] && !m.used[term_parameter(t.terms[t.literals[s.precondition.literals.begin + k].terms.begin])];

        Schema out = s;
        out.original_arity = original_arity;
        out.params = TaskData::append<Parameter>(t.params, kept);
        out.precondition = r.condition(s.precondition, &drop);
        std::vector<ConditionalEffect> ces;
        for (u32 e = 0; e < s.effects.count; ++e)
        {
            ConditionalEffect ce = t.conditional_effects[s.effects.begin + e];
            ce.condition = r.condition(ce.condition);
            ce.effects = r.literals(ce.effects);
            std::vector<NumericEffect> nes;
            for (u32 k = 0; k < ce.numeric_effects.count; ++k)
                nes.push_back(r.numeric_effect(t.numeric_effects[ce.numeric_effects.begin + k]));
            ce.numeric_effects = TaskData::append<NumericEffect>(t.numeric_effects, nes);
            if (ce.auxiliary)
                ce.auxiliary = r.numeric_effect(*ce.auxiliary);
            ces.push_back(ce);
        }
        out.effects = TaskData::append<ConditionalEffect>(t.conditional_effects, ces);
        out.collapsed = TaskData::append<CollapsedParameter>(t.collapsed_params, collapsed);
        t.schemas[si] = out;
    }
}

bool has_representatives(const TaskData& t, const Schema& s)
{
    for (const CollapsedParameter& c : t.collapsed_of(s))
        if (!c.representative.valid())
            return false;
    return true;
}

u32 written_arity(const TaskData& t, const Schema& s)
{
    u32 n = s.original_arity;
    for (const CollapsedParameter& c : t.collapsed_of(s))
        n += c.written ? 1 : 0;
    return n;
}

std::vector<ObjectId> full_binding(const TaskData& t, const Schema& s, std::span<const ObjectId> binding)
{
    const auto collapsed = t.collapsed_of(s);
    std::vector<ObjectId> out;
    out.reserve(binding.size() + collapsed.size());
    usize next = 0;
    for (const CollapsedParameter& c : collapsed)
    {
        while (out.size() < c.position && next < binding.size())
            out.push_back(binding[next++]);
        out.push_back(c.representative);
    }
    out.insert(out.end(), binding.begin() + static_cast<std::ptrdiff_t>(next), binding.end());
    return out;
}
bool has_type(const TaskData& t, ObjectId o, Range types)
{
    if (types.count == 0)
        return true;
    std::vector<u8> wanted(t.types.size(), 0);
    for (TypeId ty : TaskData::slice(t.type_ids, types))
    {
        if (t.str(t.types[ty.v].name) == "object")
            return true;
        wanted[ty.v] = 1;
    }
    std::vector<TypeId> todo(TaskData::slice(t.type_ids, t.objects[o.v].types).begin(),
                             TaskData::slice(t.type_ids, t.objects[o.v].types).end());
    std::vector<u8> seen(t.types.size(), 0);
    while (!todo.empty())
    {
        const TypeId ty = todo.back();
        todo.pop_back();
        if (ty.v >= t.types.size() || seen[ty.v])
            continue;
        if (wanted[ty.v])
            return true;
        seen[ty.v] = 1;
        for (TypeId b : TaskData::slice(t.type_ids, t.types[ty.v].bases))
            todo.push_back(b);
    }
    return false;
}
}  // namespace mymyr::formalism
