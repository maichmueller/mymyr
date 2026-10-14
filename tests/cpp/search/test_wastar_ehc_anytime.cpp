// Weighted A*, the cost bound of A*, enforced hill-climbing and the anytime search (best_first.hpp, ehc.hpp,
// anytime.hpp) on small suite tasks (unit costs: the optimal cost is the BrFS plan length):
//   - weight 1 is A* (the same expansions); with weight w the plan costs at most w times the optimum (admissible
//     heuristics with reopening, consistent ones without); ties and queues are deterministic;
//   - the cost bound prunes plans at or above it (by g, and by g + h with an admissible heuristic);
//   - enforced hill-climbing returns plans that replay, with and without helpful actions;
//   - the anytime search improves its plan until it proves it optimal, reports each plan to the observer and keeps
//     the best one when a budget stops it;
//   - refusals of unsupported options, and the searches on several threads over one task and grounding.

#include "../support/suite.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/anytime.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/ehc.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;
namespace H = mymyr::heuristics;

namespace
{
const char* const k_tasks[] = {"depot__p02", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "pegsol-08-strips__p22"};

std::shared_ptr<const Task> load(const std::string& name) { return Task::from_text_file(task_path(name)); }

size_t optimal_length(const Task& task)
{
    BrfsOptions o;
    o.stop_at_goal = true;
    const BrfsResult r = brfs(task, o);
    EXPECT_TRUE(r.solved);
    return r.plan.size();
}

bool valid_plan(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    return succ.is_goal(s);
}

BestFirstOptions with_h(H::Kind k, double weight = 1)
{
    BestFirstOptions o;
    o.heuristic.kind = k;
    o.weight = weight;
    return o;
}

/// Counts the events of a search.
struct Events : SearchObserver
{
    u64 starts = 0, expands = 0, generates = 0, passes = 0, ends = 0;
    std::vector<double> solutions;
    void on_start(StateView) override { ++starts; }
    void on_expand(u64, StateView) override { ++expands; }
    void on_generate(u64, const Action&, u64, StateView, bool) override { ++generates; }
    void on_pass(u32, const SearchStatistics&) override { ++passes; }
    void on_solution(std::span<const Action>, double cost) override { solutions.push_back(cost); }
    void on_end(SearchStatus, const SearchStatistics&) override { ++ends; }
};
}  // namespace

// ------------------------------------------------------------------------------------------------------ weighted A*

TEST(WeightedAStar, WeightOneIsAStar)
{
    for (const char* name : {"depot__p02", "openstacks-opt08-adl__p03"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        for (const auto k : {H::Kind::Max, H::Kind::LmCount})
            for (const auto fn : {&astar_eager, &astar_lazy})
            {
                SCOPED_TRACE(H::to_string(k));
                BestFirstOptions plain = with_h(k);
                BestFirstOptions weighted = with_h(k, 1.0);
                weighted.queue = BestFirstOptions::Queue::Heap;
                plain.queue = BestFirstOptions::Queue::Heap;
                const BestFirstResult a = fn(*task, plain), b = fn(*task, weighted);
                ASSERT_EQ(a.status, SearchStatus::Solved) << a.message;
                EXPECT_EQ(b.status, a.status);
                EXPECT_EQ(b.cost, a.cost);
                EXPECT_EQ(b.plan, a.plan);
                EXPECT_EQ(b.stats.expanded, a.stats.expanded);
                EXPECT_EQ(b.stats.generated, a.stats.generated);
                EXPECT_EQ(b.evaluations, a.evaluations);
            }
    }
}

TEST(WeightedAStar, CostIsWithinTheWeightTimesTheOptimum)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const double opt = static_cast<double>(optimal_length(*task));
        for (const double w : {1.5, 2.0, 5.0})
        {
            // admissible heuristics with reopening (eager), consistent ones without and in lazy A* with requeueing
            struct Run
            {
                H::Kind kind;
                BestFirstResult (*fn)(const Task&, const BestFirstOptions&);
                bool reopen;
            };
            const Run runs[] = {{H::Kind::Blind, &astar_eager, true}, {H::Kind::Max, &astar_eager, true},
                                {H::Kind::LmCut, &astar_eager, true}, {H::Kind::Max, &astar_eager, false},
                                {H::Kind::Blind, &astar_eager, false}, {H::Kind::Max, &astar_lazy, true}};
            for (const Run& run : runs)
            {
                SCOPED_TRACE(std::to_string(w) + " " + H::to_string(run.kind) + (run.reopen ? "" : " no reopening") +
                             (run.fn == &astar_lazy ? " lazy" : ""));
                BestFirstOptions o = with_h(run.kind, w);
                o.reopen = run.reopen;
                const BestFirstResult r = run.fn(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_LE(r.cost, w * opt);
                EXPECT_GE(r.cost, opt);
                EXPECT_EQ(r.cost, static_cast<double>(r.plan.size()));
                EXPECT_TRUE(valid_plan(*task, r.plan));
                EXPECT_EQ(r.queue, w == std::floor(w) ? "bucket" : "heap");  // Auto: buckets for integral keys
            }
        }
    }
}

TEST(WeightedAStar, StoresAndQueuesSeeTheSameSearch)
{
    const auto task = load("philosophers__p03-phil4");
    for (const double w : {2.0, 1.5})
        for (const auto fn : {&astar_eager, &astar_lazy})
        {
            u64 expanded = 0;
            std::vector<Action> plan;
            for (const auto store : {BestFirstOptions::Store::Flat, BestFirstOptions::Store::Chunked, BestFirstOptions::Store::Compact})
                for (const auto queue : {BestFirstOptions::Queue::Bucket, BestFirstOptions::Queue::Heap})
                {
                    if (queue == BestFirstOptions::Queue::Bucket && w != std::floor(w))
                        continue;
                    BestFirstOptions o = with_h(H::Kind::FF, w);
                    o.store = store;
                    o.queue = queue;
                    const BestFirstResult r = fn(*task, o);
                    SCOPED_TRACE(std::to_string(w) + " " + r.store + "/" + r.queue);
                    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                    if (plan.empty())
                    {
                        expanded = r.stats.expanded;
                        plan = r.plan;
                    }
                    EXPECT_EQ(r.stats.expanded, expanded);
                    EXPECT_EQ(r.plan, plan);
                }
        }
}

TEST(WeightedAStar, LargerWeightsExpandFewerStatesHere)
{
    // not a guarantee of weighted A*, but the reason to use it: on this task h_max with weight 5 expands far less
    const auto task = load("pegsol-08-strips__p22");
    const BestFirstResult a = astar_eager(*task, with_h(H::Kind::Max));
    const BestFirstResult w = astar_eager(*task, with_h(H::Kind::Max, 5));
    ASSERT_EQ(a.status, SearchStatus::Solved);
    ASSERT_EQ(w.status, SearchStatus::Solved);
    EXPECT_LT(w.stats.expanded, a.stats.expanded);
}

TEST(WeightedAStar, PathDependentHeuristic)
{
    for (const char* name : {"depot__p02", "openstacks-opt08-adl__p03"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        for (const double w : {2.0, 5.0})
            for (const auto fn : {&astar_eager, &astar_lazy})
            {
                const BestFirstResult r = fn(*task, with_h(H::Kind::LmCount, w));
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_TRUE(valid_plan(*task, r.plan));
            }
    }
}

TEST(WeightedAStar, Refusals)
{
    const auto task = load("depot__p02");
    for (const double w : {0.5, 0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW((void)astar_eager(*task, with_h(H::Kind::Max, w)), std::invalid_argument) << w;
        EXPECT_THROW((void)astar_lazy(*task, with_h(H::Kind::Max, w)), std::invalid_argument) << w;
    }
    // the weight and the cost bound are A*'s
    for (const auto fn : {&gbfs_eager, &gbfs_lazy, &beam})
    {
        EXPECT_THROW((void)fn(*task, with_h(H::Kind::FF, 2)), std::invalid_argument);
        BestFirstOptions o = with_h(H::Kind::FF);
        o.cost_bound = 10;
        EXPECT_THROW((void)fn(*task, o), std::invalid_argument);
    }
    // bucket queues need integral keys
    BestFirstOptions o = with_h(H::Kind::Max, 1.5);
    o.queue = BestFirstOptions::Queue::Bucket;
    EXPECT_THROW((void)astar_eager(*task, o), std::invalid_argument);
    o.weight = 2;
    EXPECT_EQ(astar_eager(*task, o).status, SearchStatus::Solved);
    BestFirstOptions nan = with_h(H::Kind::Max);
    nan.cost_bound = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)astar_eager(*task, nan), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------------ cost bound

TEST(CostBound, PlansCostLessThanTheBound)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const double opt = static_cast<double>(optimal_length(*task));
        // Max: admissible (prunes by g + h); FF: inadmissible (prunes by g only)
        for (const auto k : {H::Kind::Max, H::Kind::FF})
            for (const auto fn : {&astar_eager, &astar_lazy})
            {
                SCOPED_TRACE(std::string(H::to_string(k)) + (fn == &astar_lazy ? " lazy" : ""));
                BestFirstOptions o = with_h(k);
                o.cost_bound = opt;
                const BestFirstResult none = fn(*task, o);
                EXPECT_EQ(none.status, SearchStatus::Exhausted);
                EXPECT_TRUE(none.plan.empty());
                o.cost_bound = opt + 0.5;
                const BestFirstResult r = fn(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_EQ(r.cost, opt);
                EXPECT_TRUE(valid_plan(*task, r.plan));
            }
    }
    // an admissible heuristic prunes by g + h: h_max expands fewer states than blind (g + 0) to prove that no plan is
    // cheaper than the optimum
    const auto task = load("depot__p02");
    const double opt = static_cast<double>(optimal_length(*task));
    BestFirstOptions max = with_h(H::Kind::Max);
    max.cost_bound = opt;
    BestFirstOptions blind = with_h(H::Kind::Blind);
    blind.cost_bound = opt;
    const BestFirstResult rm = astar_eager(*task, max), rb = astar_eager(*task, blind);
    EXPECT_EQ(rm.status, SearchStatus::Exhausted);
    EXPECT_EQ(rb.status, SearchStatus::Exhausted);
    EXPECT_LT(rm.stats.expanded, rb.stats.expanded);
}

TEST(CostBound, AStartBeyondTheBoundEndsAtOnce)
{
    const auto task = load("depot__p02");
    BestFirstOptions o = with_h(H::Kind::Max);
    o.cost_bound = 0;
    const BestFirstResult r = astar_eager(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Exhausted);
    EXPECT_EQ(r.stats.expanded, 0u);
    o.cost_bound = 1;  // h_max of the start is at least 1
    EXPECT_EQ(astar_eager(*task, o).stats.expanded, 0u);
}

// ------------------------------------------------------------------------------------------------------ EHC

TEST(Ehc, ReturnsPlansThatReplay)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        for (const auto k : {H::Kind::FF, H::Kind::Add, H::Kind::GoalCount, H::Kind::LmCount})
            for (const auto helpful : {EhcOptions::Helpful::Auto, EhcOptions::Helpful::Off, EhcOptions::Helpful::First,
                                       EhcOptions::Helpful::Only})
            {
                const bool preferred = k == H::Kind::FF || k == H::Kind::LmCount;
                if (!preferred && (helpful == EhcOptions::Helpful::First || helpful == EhcOptions::Helpful::Only))
                    continue;
                SCOPED_TRACE(std::string(H::to_string(k)) + " helpful " + std::to_string(static_cast<int>(helpful)));
                EhcOptions o;
                o.heuristic.kind = k;
                o.helpful_actions = helpful;
                const EhcResult r = ehc(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_TRUE(valid_plan(*task, r.plan));
                EXPECT_EQ(r.cost, static_cast<double>(r.plan.size()));
                EXPECT_EQ(r.ehc.fallback, r.ehc.failed);
                EXPECT_GE(r.ehc.lookaheads, r.ehc.improvements);
                if (helpful == EhcOptions::Helpful::Off || helpful == EhcOptions::Helpful::Only || !preferred)
                {
                    EXPECT_EQ(r.ehc.retries, 0u);
                }
                // deterministic
                const EhcResult again = ehc(*task, o);
                EXPECT_EQ(again.plan, r.plan);
                EXPECT_EQ(again.stats.expanded, r.stats.expanded);
            }
    }
}

TEST(Ehc, HelpfulActionsNeedPreferredOperators)
{
    const auto task = load("depot__p02");
    for (const auto helpful : {EhcOptions::Helpful::First, EhcOptions::Helpful::Only})
    {
        EhcOptions o;
        o.heuristic.kind = H::Kind::Max;
        o.helpful_actions = helpful;
        const EhcResult r = ehc(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Failed);
        EXPECT_NE(r.message.find("preferred operators"), std::string::npos) << r.message;
    }
    EhcOptions o;
    o.heuristic.kind = H::Kind::Max;  // Auto: every action
    EXPECT_EQ(ehc(*task, o).status, SearchStatus::Solved);
    o.helpful_actions = static_cast<EhcOptions::Helpful>(9);
    EXPECT_THROW((void)ehc(*task, o), std::invalid_argument);
    o.helpful_actions = EhcOptions::Helpful::Auto;
    o.fallback = static_cast<EhcOptions::Fallback>(9);
    EXPECT_THROW((void)ehc(*task, o), std::invalid_argument);
}

TEST(Ehc, ObserverBudgetsAndStores)
{
    const auto task = load("openstacks-opt08-adl__p03");
    EhcOptions o;
    o.fallback = EhcOptions::Fallback::None;
    Events ev;
    o.control.observer = &ev;
    const EhcResult r = ehc(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_EQ(ev.starts, 1u);
    EXPECT_EQ(ev.ends, 1u);
    EXPECT_EQ(ev.expands, r.stats.expanded);
    EXPECT_EQ(ev.generates, r.stats.generated);
    EXPECT_EQ(ev.passes, r.ehc.lookaheads);
    ASSERT_EQ(ev.solutions.size(), 1u);
    EXPECT_EQ(ev.solutions[0], r.cost);
    EXPECT_EQ(r.ehc.expanded, r.stats.expanded);
    for (const auto store : {BestFirstOptions::Store::Flat, BestFirstOptions::Store::Chunked, BestFirstOptions::Store::Compact})
    {
        EhcOptions s;
        s.store = store;
        const EhcResult x = ehc(*task, s);
        ASSERT_EQ(x.status, SearchStatus::Solved) << x.message;
        EXPECT_EQ(x.plan, r.plan);
        EXPECT_EQ(x.stats.expanded, r.stats.expanded);
    }
    EhcOptions b;
    b.control.budget.max_expanded = 3;
    const EhcResult rb = ehc(*task, b);
    EXPECT_EQ(rb.status, SearchStatus::OutOfStates);
    EXPECT_LE(rb.stats.expanded, 3u);
    EXPECT_TRUE(rb.plan.empty());
    EhcOptions c;
    c.control.cancel.request();
    EXPECT_EQ(ehc(*task, c).status, SearchStatus::Cancelled);
}

// ------------------------------------------------------------------------------------------------------ anytime

TEST(Anytime, ImprovesUntilItProvesTheOptimum)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4", "pegsol-08-strips__p22"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const double opt = static_cast<double>(optimal_length(*task));
        // h_max is admissible: the weight-1 iteration's plan is optimal; FF is not: the last iteration exhausts
        for (const auto k : {H::Kind::Max, H::Kind::FF})
            for (const bool lazy : {false, true})
            {
                SCOPED_TRACE(std::string(H::to_string(k)) + (lazy ? " lazy" : ""));
                AnytimeOptions o;
                o.heuristic.kind = k;
                o.lazy = lazy;
                Events ev;
                o.control.observer = &ev;
                const AnytimeResult r = anytime_astar(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_TRUE(r.optimal);
                EXPECT_EQ(r.cost, opt);
                EXPECT_TRUE(valid_plan(*task, r.plan));
                ASSERT_FALSE(r.solutions.empty());
                for (usize i = 1; i < r.solutions.size(); ++i)
                {
                    EXPECT_LT(r.solutions[i].cost, r.solutions[i - 1].cost);
                    EXPECT_GE(r.solutions[i].seconds, r.solutions[i - 1].seconds);
                    EXPECT_GT(r.solutions[i].iteration, r.solutions[i - 1].iteration);
                }
                EXPECT_EQ(r.solutions.back().cost, r.cost);
                EXPECT_EQ(r.solutions.back().length, r.plan.size());
                // the schedule, the last weight repeated
                for (usize i = 0; i < r.iterations.size(); ++i)
                    EXPECT_EQ(r.iterations[i].weight, o.weights[std::min<usize>(i, o.weights.size() - 1)]);
                // the proof: an iteration below the incumbent exhausts its space, or (eager, admissible) the weight-1
                // iteration finds a plan
                const AnytimeIteration& last = r.iterations.back();
                if (last.status == SearchStatus::Solved)
                {
                    EXPECT_TRUE(k == H::Kind::Max && !lazy);
                    EXPECT_EQ(last.weight, 1.0);
                }
                else
                {
                    EXPECT_EQ(last.status, SearchStatus::Exhausted);
                    EXPECT_EQ(last.cost_bound, opt);
                }
                // each improved plan reaches the observer as it is found
                EXPECT_EQ(ev.solutions.size(), r.solutions.size());
                for (usize i = 0; i < ev.solutions.size(); ++i)
                    EXPECT_EQ(ev.solutions[i], r.solutions[i].cost);
                EXPECT_EQ(ev.starts, 1u);
                EXPECT_EQ(ev.ends, 1u);
                EXPECT_EQ(ev.passes, r.iterations.size());
                EXPECT_EQ(ev.expands, r.stats.expanded);
            }
    }
}

TEST(Anytime, ABudgetKeepsTheBestPlan)
{
    const auto task = load("pegsol-08-strips__p22");
    AnytimeOptions o;
    o.heuristic.kind = H::Kind::Max;
    const AnytimeResult full = anytime_astar(*task, o);
    ASSERT_EQ(full.status, SearchStatus::Solved);
    ASSERT_GE(full.solutions.size(), 1u);
    o.control.budget.max_expanded = full.solutions.front().expanded + 1;
    const AnytimeResult r = anytime_astar(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_FALSE(r.optimal);
    EXPECT_EQ(r.cost, full.solutions.front().cost);
    EXPECT_EQ(r.iterations.back().status, SearchStatus::OutOfStates);
    EXPECT_LE(r.stats.expanded, o.control.budget.max_expanded);
    EXPECT_TRUE(valid_plan(*task, r.plan));
}

TEST(Anytime, AnInitialBoundBelowTheOptimumFindsNothing)
{
    const auto task = load("depot__p02");
    const double opt = static_cast<double>(optimal_length(*task));
    AnytimeOptions o;
    o.heuristic.kind = H::Kind::FF;
    o.cost_bound = opt;
    const AnytimeResult r = anytime_astar(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Exhausted);
    EXPECT_TRUE(r.solutions.empty());
    EXPECT_FALSE(r.optimal);
    EXPECT_EQ(r.iterations.size(), 1u);
}

TEST(Anytime, Refusals)
{
    const auto task = load("depot__p02");
    const std::vector<std::vector<double>> bad = {{}, {2, 3}, {0.5}, {5, std::numeric_limits<double>::quiet_NaN()},
                                                  {std::numeric_limits<double>::infinity()}};
    for (const auto& ws : bad)
    {
        AnytimeOptions o;
        o.weights = ws;
        EXPECT_THROW((void)anytime_astar(*task, o), std::invalid_argument) << ws.size();
    }
    AnytimeOptions w;
    w.weight = 2;  // the schedule is `weights`
    EXPECT_THROW((void)anytime_astar(*task, w), std::invalid_argument);
    AnytimeOptions q;
    q.weights = {1.5, 1};
    q.queue = BestFirstOptions::Queue::Bucket;
    EXPECT_THROW((void)anytime_astar(*task, q), std::invalid_argument);
    AnytimeOptions n;
    n.cost_bound = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)anytime_astar(*task, n), std::invalid_argument);
    AnytimeOptions any;
    any.heuristic.kind = H::Kind::LmCount;
    any.control.goal.kind = GoalSpec::Kind::AnyOf;  // the landmark count estimates the task's goal only
    any.control.goal.goals.emplace_back();
    const AnytimeResult r = anytime_astar(*task, any);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_FALSE(r.message.empty());
}

// ------------------------------------------------------------------------------------------------------ threads

TEST(WeightedEhcAnytime, SearchesOnSeveralThreadsGiveTheSingleThreadedResults)
{
    const auto task = load("openstacks-opt08-adl__p03");
    const auto relaxed = H::ground(*task);
    struct Out
    {
        SearchStatus status;
        double cost;
        u64 expanded;
        std::vector<Action> plan;
    };
    auto run = [&](usize i) -> Out
    {
        switch (i)
        {
            case 0:
            {
                BestFirstOptions o = with_h(H::Kind::LmCount, 2);
                o.heuristic.relaxed = relaxed;
                const BestFirstResult r = astar_eager(*task, o);
                return {r.status, r.cost, r.stats.expanded, r.plan};
            }
            case 1:
            {
                EhcOptions o;
                o.heuristic.relaxed = relaxed;
                const EhcResult r = ehc(*task, o);
                return {r.status, r.cost, r.stats.expanded, r.plan};
            }
            case 2:
            {
                EhcOptions o;
                o.heuristic.kind = H::Kind::LmCount;
                o.heuristic.relaxed = relaxed;
                const EhcResult r = ehc(*task, o);
                return {r.status, r.cost, r.stats.expanded, r.plan};
            }
            default:
            {
                AnytimeOptions o;
                o.heuristic.relaxed = relaxed;
                o.weights = {3, 1};
                const AnytimeResult r = anytime_astar(*task, o);
                return {r.status, r.cost, r.stats.expanded, r.plan};
            }
        }
    };
    constexpr usize k_runs = 4;
    std::vector<Out> expected;
    for (usize i = 0; i < k_runs; ++i)
    {
        expected.push_back(run(i));
        ASSERT_EQ(expected.back().status, SearchStatus::Solved) << i;
    }
    constexpr u32 k_threads = 4;
    std::vector<std::vector<Out>> got(k_threads);
    {
        std::vector<std::jthread> pool;
        for (u32 t = 0; t < k_threads; ++t)
            pool.emplace_back(
                [&, t]
                {
                    for (usize i = 0; i < k_runs; ++i)
                        got[t].push_back(run((i + t) % k_runs));
                });
    }
    for (u32 t = 0; t < k_threads; ++t)
        for (usize i = 0; i < k_runs; ++i)
        {
            const Out& e = expected[(i + t) % k_runs];
            const Out& g = got[t][i];
            EXPECT_EQ(g.status, e.status) << t << " " << i;
            EXPECT_EQ(g.cost, e.cost) << t << " " << i;
            EXPECT_EQ(g.expanded, e.expanded) << t << " " << i;
            EXPECT_EQ(g.plan, e.plan) << t << " " << i;
        }
}
