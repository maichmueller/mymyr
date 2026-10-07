#pragma once
// Search goals from ground conditions (GoalSpec::AnyOf of search/control.hpp) and their test.
//
//   const GroundCondition g = ...;                               // successor/conditions.hpp
//   IwOptions o;
//   o.control.goal = search::any_of(*task, std::span(&g, 1));   // a state is a goal iff g holds in it
//
// A ground condition becomes one GoalSpec::AtomGoal: its static literals are decided against the task's static facts
// (a false one makes the goal impossible: it is left out), its fluent literals become atom slots (assigned now under
// lazy slots), its derived literals canonical atom ids tested on the state's closure under the axioms, and its numeric
// constraints stay expressions evaluated on the state's values. A fluent or derived atom outside the reachable
// domains of its predicate is false in every state: as a positive literal it makes the goal impossible, as a negative
// one it is dropped.

#include "mymyr/search/control.hpp"
#include "mymyr/successor/conditions.hpp"

#include <optional>
#include <span>

namespace mymyr
{
class Task;
class Successors;
}  // namespace mymyr

namespace mymyr::search
{
/// The goal of one ground condition, or std::nullopt if it can never hold. Throws std::invalid_argument for an invalid
/// condition (GroundCondition::validate).
[[nodiscard]] std::optional<GoalSpec::AtomGoal> atom_goal(const Task& task, const GroundCondition& condition);

/// GoalSpec::AnyOf: a state is a goal iff one of `conditions` holds in it (impossible ones are left out, so a list of
/// impossible conditions gives a goal no state satisfies).
[[nodiscard]] GoalSpec any_of(const Task& task, std::span<const GroundCondition> conditions);

/// Truth of one AnyOf goal in s. When goal.needs_view(), `succ` (a successor generator of the goal's task) must be
/// prepared on s (Successors::prepare); otherwise it is not read.
[[nodiscard]] bool holds(const GoalSpec::AtomGoal& goal, Successors& succ, StateView s);
}  // namespace mymyr::search
