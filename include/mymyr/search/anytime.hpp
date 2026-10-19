#pragma once
// Anytime search: restarting weighted A* (Richter, Thayer and Ruml 2010; the iterated searches of LAMA).
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::search::AnytimeOptions o;  // weights 5, 3, 2, 1; h_FF
//   o.control.budget.max_seconds = 60;
//   mymyr::search::AnytimeResult r = mymyr::search::anytime_astar(*task, o);  // r.plan: the best plan found
//
// Iteration k runs weighted A* (best_first.hpp: astar_eager, or astar_lazy with `lazy`) from the start state with the
// weight weights[min(k, n - 1)] (the last weight repeats) and the cost bound of the best plan found so far (before the
// first plan: the options' cost_bound), so each plan it finds is cheaper than the one before. Every iteration starts
// afresh (a new store and open list) and shares the heuristic, which is built once (its grounding and its
// statistics). A cost bound prunes states whose g is at least the bound, and with an admissible heuristic
// (Heuristic::admissible) and a goal of the task or AnyOf also those whose g + h is: an iteration that cannot improve
// on the incumbent then ends early, when its open list runs dry.
//
// The search ends when
//   - an iteration exhausts its search space: no plan is cheaper than the incumbent. With reopening (`reopen`, the
//     default) and no depth budget this proves the incumbent optimal (`optimal`), provided h = +inf only for dead ends;
//   - an eager iteration with weight 1 finds a plan with reopening and an admissible heuristic of the task's goal (not
//     GoalSpec::Custom): A*'s plan is optimal (`optimal`);
//   - a budget, the cancellation or an observer stops an iteration, or an iteration fails.
// status is Solved when a plan was found (also when a budget ended the search later; `iterations` tells how each
// iteration ended), otherwise the status of the last iteration. plan, cost and goal_state are those of the best plan.
//
// Options: every BestFirstOptions member applies to each iteration as for A*, except weight (it must stay 1: the
// schedule is `weights`) and cost_bound (the bound of the first iteration). weights: non-empty, finite, at least 1 and
// non-increasing (LAMA's 5, 3, 2, 1 by default). Invalid options throw std::invalid_argument; a heuristic that cannot
// run on the task gives status Failed with a message.
//
// SearchControl: max_seconds and max_expanded bound the whole search, max_states and max_depth each iteration (a depth
// budget voids the proof of optimality by exhaustion); cancel and blocked_states, the goal and progress_interval apply to
// every iteration. Observer events: on_start once; the on_expand, on_generate, on_prune and on_progress events of every
// iteration (ids are the iteration's own state ids); on_solution(plan, cost) for each improved plan as it is found;
// on_pass(iteration, statistics of the iteration) after each iteration; on_end once.
//
// Result: stats, evaluations, dead_ends and reopened add up the iterations (stats.states: the states stored by all
// iterations), stats.memory_bytes is the largest iteration's, initial_h, store and queue are the first iteration's,
// setup_seconds is building the heuristic, stats.seconds the search after it.
//
// Thread-safety: as the best-first searches (one heuristic per search thread; the task may be shared).

#include "mymyr/search/best_first.hpp"

#include <vector>

namespace mymyr::search
{
struct AnytimeOptions : BestFirstOptions
{
    std::vector<double> weights{5, 3, 2, 1};  // the weight of iteration k: weights[min(k, n - 1)]
    bool lazy = false;                        // iterations run lazy weighted A* (astar_lazy)
};

/// An improved plan, in the order found.
struct AnytimeSolution
{
    double cost = 0;
    double seconds = 0;  // since the search started (after building the heuristic)
    double weight = 0;   // the weight of the iteration that found it
    u32 iteration = 0;
    u64 expanded = 0;    // expansions of the whole search until the plan was found
    u64 length = 0;      // actions
};

struct AnytimeIteration
{
    double weight = 0;
    double cost_bound = 0;  // plans of the iteration cost less (+inf: no bound)
    SearchStatus status = SearchStatus::Exhausted;
    SearchStatistics stats;
    u64 evaluations = 0;
    double cost = 0;  // the plan's cost when it found one
};

struct AnytimeResult : BestFirstResult
{
    std::vector<AnytimeSolution> solutions;
    std::vector<AnytimeIteration> iterations;
    bool optimal = false;  // the plan is proven optimal (see above)
};

[[nodiscard]] AnytimeResult anytime_astar(const Task& task, const AnytimeOptions& options = {});
}  // namespace mymyr::search
