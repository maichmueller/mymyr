#include "../successor/expr_eval.hpp"

#include "mymyr/search/goal.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <vector>

namespace mymyr::search
{
using formalism::PredKind;

std::optional<GoalSpec::AtomGoal> atom_goal(const Task& task, const GroundCondition& condition)
{
    condition.validate(task);
    const plan::Compiled& C = task.compiled();
    const CanonicalLayout& L = C.layout;
    GoalSpec::AtomGoal g;
    std::vector<u32> args;
    for (const GroundLiteral& l : condition.literals)
    {
        const u32 p = l.atom.predicate.v;
        if (C.kinds[p] == PredKind::Static)
        {
            if (holds(task, StateView{}, l.atom) != l.positive)
                return std::nullopt;
            continue;
        }
        args.clear();
        for (ObjectId o : l.atom.objects)
            args.push_back(o.v);
        const CanonicalAtom c = L.has_predicate(p) ? L.encode(p, args.data()) : L.total;
        if (c >= L.total)
        {
            if (l.positive)
                return std::nullopt;  // never true
            continue;                 // never true: its negation always holds
        }
        if (C.kinds[p] == PredKind::Fluent)
            (l.positive ? g.positive : g.negative).push_back(SlotId{task.atoms().intern(c)});
        else
            (l.positive ? g.derived_positive : g.derived_negative).push_back(c);
    }
    if (!condition.constraints.empty())
    {
        auto numeric = std::make_shared<GroundCondition>();
        numeric->constraints = condition.constraints;
        numeric->exprs = condition.exprs;
        numeric->expr_terms = condition.expr_terms;
        g.numeric = std::move(numeric);
    }
    return g;
}

GoalSpec any_of(const Task& task, std::span<const GroundCondition> conditions)
{
    GoalSpec spec;
    spec.kind = GoalSpec::Kind::AnyOf;
    for (const GroundCondition& c : conditions)
        if (std::optional<GoalSpec::AtomGoal> g = atom_goal(task, c))
            spec.goals.push_back(std::move(*g));
    return spec;
}

bool holds(const GoalSpec::AtomGoal& goal, Successors& succ, StateView s)
{
    for (SlotId x : goal.positive)
        if (!bits::test(s.w, s.nw, x.v))
            return false;
    for (SlotId x : goal.negative)
        if (bits::test(s.w, s.nw, x.v))
            return false;
    const detail::Engine& e = succ.engine();
    if (goal.needs_view())
    {
        auto derived = [&](CanonicalAtom c)
        {
            const u32 slot = e.atoms().find(c);
            return slot != AtomIndex::k_empty && bits::test(e.derived(), e.derived_words(), slot);
        };
        for (CanonicalAtom c : goal.derived_positive)
            if (!derived(c))
                return false;
        for (CanonicalAtom c : goal.derived_negative)
            if (derived(c))
                return false;
    }
    if (goal.numeric)
    {
        const plan::Numeric& N = e.compiled().num;
        const GroundCondition& g = *goal.numeric;
        for (const formalism::NumericConstraint& k : g.constraints)
            if (!plan::compare(N, k.cmp, mymyr::detail::eval_expr(N, g.exprs, g.expr_terms, k.lhs, nullptr, s.num),
                               mymyr::detail::eval_expr(N, g.exprs, g.expr_terms, k.rhs, nullptr, s.num)))
                return false;
    }
    return true;
}
}  // namespace mymyr::search
