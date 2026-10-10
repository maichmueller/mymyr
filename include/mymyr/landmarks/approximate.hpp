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
//      initially has no first achiever and ends a chain. The first achievers are mimir's (FirstAchievers::HMax): the
//      operators that reach the atom at its cost in step 1. A plan may reach the atom through a more expensive operator
//      instead, so these landmarks and orderings are not sound in general (an atom every cheapest achiever needs is
//      a landmark although a costlier achiever does without it). FirstAchievers::PossiblyBefore makes them sound, as in
//      the landmark generators of Zhu and Givan (2003) and of Richter, Helmert and Westphal (2008): the first
//      achievers of an atom are all operators adding it that are applicable in the delete relaxation of the initial
//      state without the operators adding it (one exploration per atom). Every plan first makes the atom true by one of
//      them, so their shared preconditions hold right before, and the disjunctive landmarks of step 3 drawn from the
//      fact landmarks (max_disjunctive_landmark_depth = 1) are sound as well.
//   3. Disjunctive landmarks (max_disjunctive_landmark_size > 0): for every expanded atom and every predicate for which
//      every first achiever has a positive fluent precondition, the union of those preconditions, if it has at most
//      max_disjunctive_landmark_size members; members are expanded in turn, up to max_disjunctive_landmark_depth layers
//      (0: unbounded). Sets holding a fact landmark are dropped.
//   4. Achiever index: per atom every ground action with an operator adding it (whether or not the operator ever fires),
//      the first achievers' ground actions (PossiblyBefore: for the atoms step 2 and 3 expanded only), and per ground
//      action the landmarks it achieves, first-achieves and uniquely achieves (as the only achiever at all).
//
// The grounding keeps the operators that can never fire (RelaxedTaskOptions::keep_unreachable_operators): mimir's
// achiever index counts them. A RelaxedTask built without that option gives the same landmarks and orderings, but its
// achiever index lacks those actions.

#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"

namespace mymyr::landmarks
{
enum class FirstAchievers : u8
{
    HMax,            // mimir's: the operators reaching the atom at its h_max-style cost (not sound in general)
    PossiblyBefore,  // the operators applicable before the atom can first hold (sound)
};

struct ApproximateFactLandmarkOptions
{
    bool include_positive_goal_facts = true;
    bool compute_greedy_necessary_orderings = true;
    usize max_disjunctive_landmark_size = 0;   // 0: no disjunctive landmarks
    usize max_disjunctive_landmark_depth = 0;  // 0: unbounded
    heuristics::GroundingBudget budget = {};   // the Task overload's grounding
    FirstAchievers first_achievers = FirstAchievers::HMax;
};

/// Grounds `task` (keeping operators that can never fire) and generates. Throws std::runtime_error when the grounding
/// exceeds options.budget.
[[nodiscard]] FactLandmarkGraph approximate_fact_landmarks(const Task& task, const ApproximateFactLandmarkOptions& options = {});

/// Generates over an existing grounding of `relaxed.task()`.
[[nodiscard]] FactLandmarkGraph approximate_fact_landmarks(const heuristics::RelaxedTask& relaxed,
                                                           const ApproximateFactLandmarkOptions& options = {});
}  // namespace mymyr::landmarks
