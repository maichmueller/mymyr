#pragma once
// Test support: replay of a plan on a task with numeric effects.

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <cmath>
#include <vector>

namespace mymyr::test
{
/// Replays the plan with numeric effects and total-cost effects: the metric value of the goal state it reaches, or
/// NaN if an action is not applicable or the last state is not a goal state.
inline double replay(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    const heuristics::ActionCosts costs(task);
    State s = task.initial_state();
    double g = costs.initial(s.view());
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s.view(), a.label()))
            return std::nan("");
        StateBuilder b;
        const Delta d = succ.apply_with_delta(s.view(), a.label(), b);
        g = costs.next(g, d);
        s = b.build();
    }
    return task.is_goal(s.view()) ? g : std::nan("");
}
}  // namespace mymyr::test
