#pragma once
// Approximate fact landmarks over the grounded delete relaxation, matching mimir 0.16.3's
// ApproximateFactLandmarkGenerator.
//
//   auto g = mymyr::landmarks::approximate_fact_landmarks(*task);            // grounds the task (budget: options)
//   auto g = mymyr::landmarks::approximate_fact_landmarks(*relaxed_task);    // reuses a grounding
//
// The reading is mimir's:
//   1. An h_max-style Dijkstra from the initial state over the relaxed operators of heuristics::RelaxedTask (one per
//      ground action and effect instance, plus one per ground axiom): an operator's cost is the maximum of its
//      precondition costs, firing it costs one more (actions and axioms alike, unlike h_max, where axioms are free). A
//      "false" proposition costs 0 unless its atom holds initially; then it needs an operator deleting the atom. No early
//      exit. The first achievers of an atom are the operators whose firing cost equals the atom's cost.
//   2. Fact landmarks: the positive fluent goal atoms, then back-chaining: the positive fluent preconditions shared by
//      every first achiever of a landmark are landmarks, each ordered greedy-necessarily before it. An atom true
//      initially has no first achiever and ends a chain.
//   3. Disjunctive landmarks (max_disjunctive_landmark_size > 0): for every expanded atom and every predicate for which
//      every first achiever has a positive fluent precondition, the union of those preconditions, if it has at most
//      max_disjunctive_landmark_size members; members are expanded in turn, up to max_disjunctive_landmark_depth layers
//      (0: unbounded). Sets holding a fact landmark are dropped.
//   4. Achiever index: per atom every ground action with an operator adding it (whether or not the operator ever fires),
//      the first achievers' ground actions, and per ground action the landmarks it achieves, first-achieves and
//      uniquely achieves (as the only achiever at all).
//
// The grounding keeps the operators that can never fire (RelaxedTaskOptions::keep_unreachable_operators): mimir's
// achiever index counts them. A RelaxedTask built without that option gives the same landmarks and orderings, but its
// achiever index lacks those actions.

#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"

namespace mymyr::landmarks
{
struct ApproximateFactLandmarkOptions
{
    bool include_positive_goal_facts = true;
    bool compute_greedy_necessary_orderings = true;
    usize max_disjunctive_landmark_size = 0;   // 0: no disjunctive landmarks
    usize max_disjunctive_landmark_depth = 0;  // 0: unbounded
    heuristics::GroundingBudget budget = {};   // the Task overload's grounding
};

/// Grounds `task` (keeping operators that can never fire) and generates. Throws std::runtime_error when the grounding
/// exceeds options.budget.
[[nodiscard]] FactLandmarkGraph approximate_fact_landmarks(const Task& task, const ApproximateFactLandmarkOptions& options = {});

/// Generates over an existing grounding of `relaxed.task()`.
[[nodiscard]] FactLandmarkGraph approximate_fact_landmarks(const heuristics::RelaxedTask& relaxed,
                                                           const ApproximateFactLandmarkOptions& options = {});
}  // namespace mymyr::landmarks
