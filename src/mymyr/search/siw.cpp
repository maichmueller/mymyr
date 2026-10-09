// Serialized IW (search/siw.hpp): a port of mimir's siw::find_solution (mimir 0.16.3 src/search/algorithms/siw.cpp)
// over the IW ladder of iw.cpp.

#include "mymyr/search/siw.hpp"

#include "iw_detail.hpp"

#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>

namespace mymyr::search
{
SiwResult siw(const Task& task, const SiwOptions& options)
{
    SiwResult r;
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    detail::Context c(task, options, succ);
    State cur = options.start ? *options.start : task.initial_state();
    if (c.observer)
        c.observer->on_start(cur);
    const detail::GoalTest goal = detail::GoalTest::from_spec(task, options.control.goal);
    const detail::GoalTest task_goal = detail::GoalTest::counter(task, 0);
    const detail::PlanCost plan_cost(task);

    auto finish = [&](SearchStatus s)
    {
        r.status = s;
        if (s != SearchStatus::Solved)
            r.cost = 0;
        r.cost_exact = plan_cost.exact();
        r.fluent_slots = task.atoms().fluent_slots();
        if (c.observer)
        {
            if (s == SearchStatus::Solved)
                c.observer->on_solution(r.plan, r.cost);
            c.observer->on_end(s, r.total);
        }
        return r;
    };

    if (goal.statically_false() || task_goal.statically_false())
        return finish(SearchStatus::Unsolvable);

    while (true)
    {
        // the overall goal (mimir's goal_strategy->test_dynamic_goal on the current state)
        bool prepared = false;
        if (goal.needs_view() || task_goal.needs_view())
        {
            succ.prepare(cur.view());
            prepared = true;
        }
        if (goal.test(succ, cur.view(), prepared))
        {
            r.goal_state = cur;
            return finish(SearchStatus::Solved);
        }
        if (c.out_of_time())
            return finish(SearchStatus::OutOfTime);
        if (options.control.cancel.requested())
            return finish(SearchStatus::Cancelled);

        SiwSubproblem sub;
        sub.unsatisfied_at_start = task_goal.unsatisfied(succ, cur.view(), prepared);
        const detail::GoalTest sub_goal = detail::GoalTest::counter(task, sub.unsatisfied_at_start);
        IwResult ir = detail::run_ladder(c, cur.view(), sub_goal, options.max_arity);
        sub.status = ir.status;
        sub.passes = std::move(ir.passes);
        sub.plan_length = static_cast<u32>(ir.plan.size());
        sub.effective_width = ir.effective_width;
        for (const IwPassStatistics& p : sub.passes)
            detail::add_pass(r.total, p);
        r.peak_table_bytes = std::max(r.peak_table_bytes, ir.peak_table_bytes);
        r.peak_node_bytes = std::max(r.peak_node_bytes, ir.peak_node_bytes);
        r.subproblems.push_back(std::move(sub));
        if (ir.status != SearchStatus::Solved)
        {
            r.message = std::move(ir.message);
            return finish(ir.status);
        }
        r.max_effective_width = std::max(r.max_effective_width, ir.effective_width);
        // mimir adds up the subplans' costs, each measured from the start value (siw.cpp out_plan_cost)
        r.cost += plan_cost.apply(succ, cur, ir.plan);
        r.plan.insert(r.plan.end(), std::make_move_iterator(ir.plan.begin()), std::make_move_iterator(ir.plan.end()));
        cur = std::move(*ir.goal_state);
    }
}
}  // namespace mymyr::search
