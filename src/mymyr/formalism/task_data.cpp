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
}  // namespace mymyr::formalism
