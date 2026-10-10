// The atomic-goal portfolio (search/portfolio.hpp): a port of mimir's find_solution_atomic_goal_portfolio
// (src/search/algorithms/iw/atomic_goal_portfolio.cpp) on the IW family engine and rollout_iw.

#include "mymyr/search/portfolio.hpp"

#include "novelty_brfs.hpp"
#include "rollout_detail.hpp"

#include "mymyr/core/threads.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mymyr::search
{
namespace
{
/// The best plan so far: length compared and plan stored under one lock (the coordination's incumbent is the
/// lock-free hint the searches read).
class Incumbent
{
public:
    bool publish(const std::vector<Action>& plan, u32 worker, SearchCoordination& coord)
    {
        const u32 len = static_cast<u32>(plan.size());
        std::lock_guard lock(m_mutex);
        if (len >= m_length)
            return false;
        m_plan = plan;
        m_length = len;
        m_worker = worker;
        coord.incumbent_length.store(len, std::memory_order_relaxed);
        return true;
    }
    struct Snapshot
    {
        std::vector<Action> plan;
        u32 length = SearchCoordination::k_no_incumbent;
        u32 worker = ~u32{0};
    };
    [[nodiscard]] Snapshot read() const
    {
        std::lock_guard lock(m_mutex);
        return {m_plan, m_length, m_worker};
    }

private:
    mutable std::mutex m_mutex;
    std::vector<Action> m_plan;
    u32 m_length = SearchCoordination::k_no_incumbent;
    u32 m_worker = ~u32{0};
};

RolloutOrdering default_ordering_for(u32 offset, u64 base_seed)
{
    static constexpr ActionOrdering k_cycle[] = {ActionOrdering::DirectGoalAchieverFirst, ActionOrdering::MixedRegressionRandom,
                                                 ActionOrdering::Randomized, ActionOrdering::GoalRegressionRelevance};
    return {k_cycle[offset % 4], base_seed + offset};
}

void accumulate(RolloutIwStatistics& a, const RolloutIwStatistics& b)
{
    a.rollouts += b.rollouts;
    a.generated += b.generated;
    a.expanded += b.expanded;
    a.feature_depth_improvements += b.feature_depth_improvements;
    a.case1 += b.case1;
    a.case2 += b.case2;
    a.case3 += b.case3;
    a.case4 += b.case4;
    a.solved_propagations += b.solved_propagations;
    a.dead_ends += b.dead_ends;
    a.depth_bound_prunings += b.depth_bound_prunings;
    a.incumbent_bound_prunings += b.incumbent_bound_prunings;
    a.blocked += b.blocked;
    a.max_rollout_depth = std::max(a.max_rollout_depth, b.max_rollout_depth);
    a.tree_nodes += b.tree_nodes;
    a.seconds += b.seconds;
}

void add_stats(SearchStatistics& a, const SearchStatistics& b)
{
    a.expanded += b.expanded;
    a.generated += b.generated;
    a.states += b.states;
    a.pruned += b.pruned;
    a.seconds += b.seconds;
}
}  // namespace

PortfolioResult atomic_goal_portfolio(const Task& task, const PortfolioOptions& o)
{
    // one search per goal atom: numeric goal constraints would be dropped
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: the atomic goal portfolio does not support tasks with numeric fluents");
    PortfolioResult result;
    const SearchControl& control = o.control;
    const u32 K = o.num_rollout_workers;
    const u32 W = K + 1;  // worker 0 is the certifier
    result.rollout_statistics.resize(K);
    result.rollout_statuses.assign(K, SearchStatus::Exhausted);
    result.rollout_rounds.assign(K, 0);
    result.fluent_slots = task.atoms().fluent_slots();
    if (control.coordination)
    {
        result.status = SearchStatus::Failed;
        result.message = "atomic_goal_portfolio: SearchControl::coordination must be null (the portfolio coordinates its workers itself)";
        return result;
    }
    {
        const detail::GoalTest g = detail::GoalTest::from_spec(task, control.goal);
        if (g.statically_false())
        {
            result.status = SearchStatus::Unsolvable;
            result.message = "the atomic goal's static part cannot hold";
            return result;
        }
    }

    // per-worker hooks, created on the calling thread
    SearchObserver* const root = control.observer;
    std::vector<std::shared_ptr<SearchObserver>> worker_obs(W);
    std::vector<GoalSpec> worker_goals;
    bool safe = true;
    if (root)
        for (u32 k = 0; k < W && safe; ++k)
        {
            worker_obs[k] = root->make_worker(k);
            if (!worker_obs[k])
            {
                safe = false;
                result.message = "the observer is not parallel-safe (make_worker returned null): ran on the calling thread";
            }
        }
    if (safe && control.goal.kind == GoalSpec::Kind::Custom)
    {
        worker_goals.resize(W);
        for (u32 k = 0; k < W && safe; ++k)
        {
            std::function<bool(StateView)> f;
            if (control.goal.make_worker)
                f = control.goal.make_worker(k);
            if (!f)
            {
                safe = false;
                result.message = "the custom goal test is not parallel-safe (GoalSpec::make_worker): ran on the calling thread";
                break;
            }
            worker_goals[k].kind = GoalSpec::Kind::Custom;
            worker_goals[k].test = std::move(f);
        }
    }
    const std::string serial_note = safe ? std::string{} : result.message;
    if (!safe)
    {
        worker_obs.assign(W, nullptr);
        worker_goals.clear();
    }
    auto hot_of = [&](u32 k) { return worker_obs[k] ? worker_obs[k].get() : root; };
    auto goal_of = [&](u32 k) -> const GoalSpec& { return worker_goals.empty() ? control.goal : worker_goals[k]; };

    const State start = o.start ? *o.start : task.initial_state();
    const double secs = control.budget.max_seconds;
    const bool timed = secs < 1e15;
    const auto deadline = timed ? detail::Clock::now() + std::chrono::duration_cast<detail::Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)))
                                : detail::Clock::time_point{};
    auto remaining_seconds = [&]() -> double
    {
        if (!timed)
            return std::numeric_limits<double>::infinity();
        return std::max(0.0, std::chrono::duration<double>(deadline - detail::Clock::now()).count());
    };
    auto out_of_time = [&] { return timed && detail::Clock::now() >= deadline; };

    SearchCoordination coord;
    coord.max_total_expansions.store(control.budget.max_expanded, std::memory_order_relaxed);
    Incumbent incumbent;
    if (root)
        root->on_start(start);

    auto run_certifier = [&]
    {
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
        const detail::GoalTest goal = detail::GoalTest::from_spec(task, goal_of(0));
        const detail::BlockedSet blocked(control.blocked_states);
        SearchControl cc;
        cc.budget = control.budget;
        cc.budget.max_expanded = std::numeric_limits<u64>::max();  // the total cap is the coordination's
        cc.budget.max_seconds = std::numeric_limits<double>::infinity();
        cc.cancel = control.cancel;
        cc.progress_interval = control.progress_interval;
        detail::Env env(task, succ, goal, blocked, cc);
        env.obs = hot_of(0);
        env.root = nullptr;
        env.coord = &coord;
        env.canonical = o.canonical_order;
        env.successor_order = &o.successor_order;
        env.timed = timed;
        env.deadline = deadline;
        IwResult r = detail::classic_ladder(env, start, 1, true, WidthZero::ExpandDepthOne, o.tables);
        detail::finish_result(env, start, r);
        result.certifier_status = r.status;
        result.certifier_ran = true;
        if (r.status == SearchStatus::Solved)
        {
            incumbent.publish(r.plan, 0, coord);
            coord.request_cancel();  // shortest in the width-1 space: nothing left to find
        }
        else if (coord.is_incumbent_certified(coord.get_incumbent_length()))
            coord.request_cancel();
        result.certifier = std::move(r);
    };

    auto run_rollout_worker = [&](u32 k)
    {
        const u32 offset = k - 1;
        const RolloutOrdering cfg = o.rollout_orderings.empty() ? default_ordering_for(offset, o.base_seed)
                                                                : o.rollout_orderings[offset % o.rollout_orderings.size()];
        RolloutIwOptions ro;
        ro.control.budget = control.budget;
        ro.control.budget.max_expanded = std::numeric_limits<u64>::max();
        ro.control.cancel = control.cancel;
        ro.control.goal = goal_of(k);
        ro.control.blocked_states = control.blocked_states;
        ro.control.progress_interval = control.progress_interval;
        ro.control.coordination = &coord;
        ro.ordering = cfg.ordering;
        ro.seed = cfg.seed;
        ro.canonical_order = o.canonical_order;
        ro.start = start;
        ro.successor_order = o.successor_order;
        SearchObserver* const hot = hot_of(k);
        for (;;)
        {
            const u32 bound_before = coord.get_incumbent_length();
            ro.control.budget.max_seconds = remaining_seconds();
            const RolloutIwResult r = detail::run_rollout_iw(task, ro, hot, nullptr);
            accumulate(result.rollout_statistics[offset], r.statistics);
            result.rollout_statuses[offset] = r.status;
            ++result.rollout_rounds[offset];
            if (r.status != SearchStatus::Solved)
                return;  // exhausted, cancelled or out of budget
            incumbent.publish(r.plan, k, coord);
            if (coord.is_incumbent_certified(coord.get_incumbent_length()))
            {
                coord.request_cancel();
                return;
            }
            if (coord.is_cancelled() || control.cancel.requested() || out_of_time())
                return;
            // restart only under a strictly tighter bound (a depth-1 goal is immune to the bound)
            if (coord.get_incumbent_length() >= bound_before)
                return;
        }
    };
    auto run_worker = [&](u32 k)
    {
        if (k == 0)
            run_certifier();
        else
            run_rollout_worker(k);
    };
    auto should_stop = [&] { return coord.is_cancelled() || control.cancel.requested() || out_of_time() || coord.get_total_expansions() >= control.budget.max_expanded; };

    u32 T = resolve_threads(o.num_threads, "num_threads");
    T = std::min(T, W);
    if (!safe)
        T = 1;
    result.threads_used = T;
    if (T <= 1)
    {
        // serial: every worker to completion in order, the certifier first
        for (u32 k = 0; k < W; ++k)
        {
            if (should_stop())
            {
                coord.request_cancel();
                break;
            }
            run_worker(k);
        }
    }
    else
    {
        std::atomic<u32> next{0};
        std::vector<std::exception_ptr> errors(W);
        std::mutex done_mutex;
        std::condition_variable done_cv;
        u32 running = T;
        auto thread_main = [&](u32)
        {
            for (;;)
            {
                const u32 k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= W)
                    break;
                try
                {
                    run_worker(k);
                }
                catch (...)
                {
                    coord.request_cancel();  // siblings must not keep the exception pending
                    errors[k] = std::current_exception();
                }
            }
            std::lock_guard lock(done_mutex);
            --running;
            done_cv.notify_all();
        };
        std::vector<std::thread> threads;
        start_threads(threads, T, thread_main, [&] { coord.request_cancel(); });
        {
            // poll for the budgets a worker deep inside one long step cannot see
            std::unique_lock lock(done_mutex);
            while (!done_cv.wait_for(lock, std::chrono::milliseconds(5), [&] { return running == 0; }))
                if (should_stop())
                    coord.request_cancel();
        }
        for (std::thread& t : threads)
            t.join();
        for (const std::exception_ptr& e : errors)
            if (e)
                std::rethrow_exception(e);
    }

    // finalize
    const Incumbent::Snapshot snap = incumbent.read();
    result.iw_completed_depth = coord.get_completed_depth();
    result.iw_lower_bound = coord.get_lower_bound();
    result.total_expansions = coord.get_total_expansions();
    result.fluent_slots = task.atoms().fluent_slots();
    auto append_note = [&]
    {
        if (!serial_note.empty())
            result.message += " (" + serial_note + ")";
    };
    SearchStatistics total = result.certifier.total;
    for (const RolloutIwStatistics& s : result.rollout_statistics)
        add_stats(total, s.statistics());
    if (snap.length == SearchCoordination::k_no_incumbent)
    {
        const SearchStatus cs = result.certifier_ran ? result.certifier_status : (control.cancel.requested() ? SearchStatus::Cancelled
                                                                                  : out_of_time()             ? SearchStatus::OutOfTime
                                                                                                              : SearchStatus::OutOfStates);
        if (cs == SearchStatus::Unsolvable)
        {
            result.status = SearchStatus::Unsolvable;
            result.message = "the certifier proved the goal unreachable";
        }
        else if (cs == SearchStatus::Exhausted)
        {
            result.status = SearchStatus::Exhausted;
            result.message = "the certifier exhausted the width-1 space without finding a plan";
        }
        else
        {
            result.status = cs;
            result.message = std::string("no plan found; certifier stopped: ") + to_string(cs);
        }
        append_note();
    }
    else
    {
        result.status = SearchStatus::Solved;
        result.plan = snap.plan;
        result.plan_length = snap.length;
        result.winning_worker = snap.worker;
        const bool whole_space = result.certifier_ran && (result.certifier_status == SearchStatus::Exhausted || result.certifier_status == SearchStatus::Failed);
        const bool outside = whole_space && snap.worker != 0;
        result.certified_optimal = snap.worker == 0 || (coord.is_incumbent_certified(snap.length) && !outside);
        if (result.certified_optimal)
            result.message = "certified shortest under width-1 assumptions";
        else if (outside)
            result.message = "best plan found by a rollout worker; the certifier found no plan in the width-1 space at all, so its bound does not apply";
        else
            result.message = std::string("best plan found; optimality not certified (certifier stopped: ") +
                             (result.certifier_ran ? to_string(result.certifier_status) : "not started") + ")";
        append_note();
        const detail::PlanCost pc(task);
        const WorkspaceLease lease = task.workspace();
        result.cost = pc.apply(lease->successors(), start, result.plan);
        result.cost_exact = pc.exact();
    }
    if (root)
    {
        for (const IwPassStatistics& p : result.certifier.passes)
            root->on_pass(p.arity, p.statistics());
        if (result.status == SearchStatus::Solved)
            root->on_solution(result.plan, result.cost);
        root->on_end(result.status, total);
    }
    return result;
}
}  // namespace mymyr::search
