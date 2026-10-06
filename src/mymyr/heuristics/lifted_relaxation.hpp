#pragma once
// Internal: the lifted evaluation of h_max / h_add / h_FF (the fallback of heuristics.cpp beyond the grounding budget
// and for states outside the grounded relaxation) over the relaxed reachability matchers (reachability/join.hpp).
//
// The relaxation is relaxed_task.hpp's, read the same way as the grounded evaluation: one operator per (schema,
// conditional effect with effects) and one per axiom; positive fluent and derived preconditions (a condition's literals
// as a set), the action precondition's static literals, and for a quantified effect parameter mimir's single-
// parameter static candidates (the effect condition's static literals are otherwise ignored); negated fluent
// preconditions cost 0 when the atom is false in the evaluated state and otherwise the cost of deleting it; negated
// derived preconditions hold. Actions cost 1 (or their real cost), axioms 0; h_add sums the preconditions of an action
// and takes the maximum over an axiom's.
//
// The evaluation is a cost-bucketed semi-naive fixpoint (generalized Dijkstra, lifted): atoms are settled bucket by
// bucket in increasing cost; within a bucket, in rounds. A round's new atoms are the delta: every operator binding is
// enumerated exactly once, by the plan anchored at the first relation literal whose atom is in the delta (literals
// before it read the atoms settled before the round, the others those settled up to it), when its last positive
// precondition settles, so its positive precondition costs are final. A negated precondition over an atom of the state
// whose deletion cost is not settled yet parks the binding until it is (the re-check list). The fixpoint stops as soon
// as every goal literal is settled. h_FF's best supporters are one word each: an index into the (operator, binding)
// records of the evaluation. Per-evaluation scratch is reset in O(1) by a generation counter.

#include "mymyr/heuristics/heuristic.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mymyr::heuristics::detail
{
struct GoalLit
{
    u32 pred;
    std::vector<u32> args;
    bool positive;
};

class LiftedRelaxation
{
public:
    LiftedRelaxation(const Task& task, Kind kind, bool real);
    ~LiftedRelaxation();
    LiftedRelaxation(const LiftedRelaxation&) = delete;
    LiftedRelaxation& operator=(const LiftedRelaxation&) = delete;

    /// Minimum over the goals (each a conjunction of literals over task atoms).
    Value evaluate(StateView s, const std::vector<std::vector<GoalLit>>& goals);
    /// Polled every 1024 bindings; true throws Interrupted.
    const std::function<bool()>* interrupt = nullptr;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::heuristics::detail
