#pragma once
// The atomic-goal portfolio, matching mimir 0.16.3's iw::find_solution_atomic_goal_portfolio
// (src/search/algorithms/iw/atomic_goal_portfolio.cpp): one canonical IW(1) certifier plus K goal-guided Rollout IW(1)
// workers over one goal, side by side under a shared SearchCoordination (search/control.hpp).
//
//   mymyr::search::PortfolioOptions o;
//   o.num_rollout_workers = 4;
//   o.num_threads = 1;  // serial and reproducible
//   mymyr::search::PortfolioResult r = mymyr::search::atomic_goal_portfolio(*task, o);
//   if (r.status == mymyr::search::SearchStatus::Solved && r.certified_optimal) ...
//
// Workers: 0 is the certifier, the ladder of search/iw.hpp with max_arity 1 (optimized IW(1)) on the IW family engine,
// publishing completed depths; 1..K run rollout_iw (search/rollout_iw.hpp) with ordering rollout_orderings[(k - 1) %
// size], or, when that is empty, mimir's built-in cycle DirectGoalAchieverFirst, MixedRegressionRandom, Randomized,
// GoalRegressionRelevance with seed base_seed + k - 1. The best plan so far is kept under a mutex (its length is the
// coordination's incumbent, which prunes the rollouts). As in mimir:
//   - a certifier plan is shortest in the width-1 space and cancels everyone; otherwise the certifier cancels everyone
//     once it has certified the incumbent;
//   - a rollout worker that finds a plan publishes it, cancels everyone if it is certified, and otherwise restarts
//     (with its original seed) only if the incumbent is now strictly shorter than when its round began;
//   - certified_optimal: the certifier found the plan, or it completed depth plan_length - 1 and did not exhaust its
//     space (a certifier that searched its whole width-1 space without a plan certifies nothing about a rollout plan);
//   - no plan: Unsolvable (static goal false), Exhausted (the certifier exhausted the width-1 space; mimir reports
//     its certifier's FAILED), or the certifier's status.
// num_threads (0: the hardware's) is capped at K + 1; with one thread every worker runs to completion in order on the
// calling thread (certifier first), which is fully reproducible. With more, workers are handed to threads in index
// order and the calling thread polls every 5 ms, cancelling on the deadline, the total expansion budget and the
// CancelToken; the result then depends on timing, as in mimir.
//
// SearchControl: goal is the atomic goal (Task, AnyOf or Custom; the goal-directed orderings use its positive atoms,
// see rollout_iw.hpp); budget.max_seconds spans the call; budget.max_expanded caps the expansions of all workers
// together (mimir's max_total_expansions; one per popped IW node, one per materialized rollout node);
// budget.max_depth applies to every worker; budget.max_states applies per worker (per IW pass, per rollout round);
// cancel and blocked_states apply to every worker. coordination must be null (the portfolio coordinates its workers
// itself). Observer (tyr's make_worker protocol, search/control.hpp): SearchObserver::make_worker(k) is called for
// k = 0..K on the calling thread, and worker k sends its hot events there (node ids are per worker); the root observer
// gets on_start before, and after every worker has joined on_pass for each certifier pass, on_solution for the
// returned plan and on_end (statistics summed over all workers). A Custom goal needs GoalSpec::make_worker(k) the
// same way. If the observer's or the goal's make_worker returns null/empty for some worker, the portfolio runs on the
// calling thread alone (threads_used = 1, the serial order above) with every event on the root observer and the
// shared goal test. successor_order, if set, orders the successors of every worker (IwOptions::successor_order) and
// must be thread-safe.

#include "mymyr/novelty/novelty_table.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/rollout_iw.hpp"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mymyr::search
{
struct RolloutOrdering
{
    ActionOrdering ordering = ActionOrdering::InOrder;
    u64 seed = 0;
};

struct PortfolioOptions
{
    SearchControl control;
    std::optional<State> start;
    u32 num_rollout_workers = 4;  // 0: the certifier alone
    u32 num_threads = 0;          // 0: std::thread::hardware_concurrency()
    u64 base_seed = 0;
    std::vector<RolloutOrdering> rollout_orderings;  // cycled; empty: the built-in mix
    bool canonical_order = true;
    novelty::TableOptions tables;  // the certifier's novelty table
    std::function<void(StateView state, std::span<const Action> actions, std::vector<u32>& order)> successor_order;
};

struct PortfolioResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;
    double cost = 0;
    bool cost_exact = true;
    u32 plan_length = ~u32{0};
    bool certified_optimal = false;
    u32 iw_lower_bound = 0;                                 // iw_completed_depth + 1, or 0
    u32 iw_completed_depth = SearchCoordination::k_no_depth;  // deepest layer the certifier finished
    u32 winning_worker = ~u32{0};                           // 0: the certifier, 1..K: rollout workers
    std::string message;                                    // mimir's stop_reason

    bool certifier_ran = false;
    SearchStatus certifier_status = SearchStatus::Exhausted;  // valid if certifier_ran
    IwResult certifier;                                       // the certifier's ladder (passes, statistics)
    std::vector<RolloutIwStatistics> rollout_statistics;      // per rollout worker, summed over its rounds
    std::vector<SearchStatus> rollout_statuses;               // of each worker's last round
    std::vector<u32> rollout_rounds;                          // rounds run (0: the worker never started)
    u64 total_expansions = 0;
    u32 threads_used = 0;
    u32 fluent_slots = 0;
};

[[nodiscard]] PortfolioResult atomic_goal_portfolio(const Task& task, const PortfolioOptions& options = {});
}  // namespace mymyr::search
