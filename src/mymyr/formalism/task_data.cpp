#include "mymyr/formalism/task_data.hpp"

#include <stdexcept>
#include <string>

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
}  // namespace mymyr::formalism
