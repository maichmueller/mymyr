#pragma once
// The perfect heuristic h*: goal distances looked up in a state space (as mimir's PerfectHeuristic).
//
//   auto r = mymyr::datasets::generate_state_space(task, {.remove_if_unsolvable = false});
//   auto h = mymyr::heuristics::perfect(r.space);             // unit goal distances
//   double v = h->evaluate(state);                             // +inf: no goal is reachable
//
// Costs::Unit gives the number of actions to a nearest goal (StateSpace::unit_goal_distances), Costs::Real the cost of
// a cheapest path to a goal (StateSpace::cost_goal_distances: the transition costs, as in mimir's action goal
// distance). Evaluating a state the space does not contain throws std::invalid_argument; so does a goal other than the
// task's (evaluate(s, goals)): the distances are those to the task's goal. Batched evaluation is a lookup per state.
// The heuristic holds no per-thread scratch beyond its statistics; the space must be one without symmetry pruning
// (a pruned space stores one state per class, so most states would not be found).

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/heuristics/heuristic.hpp"

#include <memory>

namespace mymyr::heuristics
{
/// h* of the states of `space` (shared, kept alive by the heuristic). Throws std::invalid_argument for a null or
/// symmetry-reduced space.
[[nodiscard]] std::unique_ptr<Heuristic> perfect(datasets::StateSpacePtr space, Costs costs = Costs::Unit);
}  // namespace mymyr::heuristics
