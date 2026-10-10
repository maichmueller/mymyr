// BrFS: state, generated and goal-state counts equal the probe's on the suite in every mode and with every store
// (the full suite with MYMYR_TEST_FULL_SUITE=1, otherwise the tasks up to 100k states), deterministic ids
// independent of the thread count (gate 4), and plans that reach the goal (shortest ones at every thread count); the
// observer's events equal the statistics, per worker with make_worker; budgets, cancellation and on_progress set the
// status.

#include "../support/suite.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct Config
{
    const char* name;
    TaskOptions::Matching matching;
    TaskOptions::Atoms atoms;
    BrfsOptions::Store store;
    u32 threads;
    bool witness;
    bool canonical;
    bool deterministic;
};

const Config k_configs[] = {
    {"fixed_lazy", TaskOptions::Matching::FixedOrder, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Flat, 1, true, true, true},
    {"fixed_frozen", TaskOptions::Matching::FixedOrder, TaskOptions::Atoms::Frozen, BrfsOptions::Store::Flat, 1, true, false, true},
    {"fc_lazy", TaskOptions::Matching::ForwardChecking, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Flat, 1, true, false, true},
    {"fc_frozen", TaskOptions::Matching::ForwardChecking, TaskOptions::Atoms::Frozen, BrfsOptions::Store::Flat, 1, true, true, true},
    {"fixed_lazy_nowitness", TaskOptions::Matching::FixedOrder, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Flat, 1, false, true, true},
    {"fc_frozen_nowitness", TaskOptions::Matching::ForwardChecking, TaskOptions::Atoms::Frozen, BrfsOptions::Store::Flat, 1, false, false, true},
    {"auto_chunked", TaskOptions::Matching::Auto, TaskOptions::Atoms::Auto, BrfsOptions::Store::Chunked, 1, true, true, true},
    {"auto_chunked_lazy", TaskOptions::Matching::Auto, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Chunked, 1, true, true, true},
    {"auto_compact", TaskOptions::Matching::Auto, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Compact, 1, true, true, true},
    {"concurrent_t1", TaskOptions::Matching::Auto, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Concurrent, 1, true, true, true},
    {"concurrent_t4", TaskOptions::Matching::Auto, TaskOptions::Atoms::Lazy, BrfsOptions::Store::Concurrent, 4, true, true, true},
    {"concurrent_t4_ranges", TaskOptions::Matching::Auto, TaskOptions::Atoms::Frozen, BrfsOptions::Store::Concurrent, 4, false, false, false},
};

struct Param
{
    SuiteTask task;
    Config config;
};
void PrintTo(const Param& p, std::ostream* os) { *os << p.task.name << "/" << p.config.name; }

class SuiteCounts : public ::testing::TestWithParam<Param>
{
};

TEST_P(SuiteCounts, MatchTheProbe)
{
    const Param& p = GetParam();
    if (p.task.states > suite_state_limit())
        GTEST_SKIP() << "above the default state limit (set MYMYR_TEST_FULL_SUITE=1)";
    TaskOptions to;
    to.matching = p.config.matching;
    to.atoms = p.config.atoms;
    const auto task = Task::from_text_file(task_path(p.task.name), to);
    BrfsOptions bo;
    bo.stop_at_goal = false;
    bo.store = p.config.store;
    bo.threads = p.config.threads;
    bo.witness_pruning = p.config.witness;
    bo.canonical_order = p.config.canonical;
    bo.deterministic_ids = p.config.deterministic;
    const BrfsResult r = brfs(*task, bo);
    EXPECT_TRUE(r.exhausted);
    EXPECT_EQ(r.states, p.task.states);
    EXPECT_EQ(r.goal_states, p.task.goal_states);
    EXPECT_EQ(r.expanded, p.task.states);
    if (p.config.witness)
        EXPECT_EQ(r.generated, p.task.generated);
    else
        EXPECT_GE(r.generated, p.task.generated);
}

std::vector<Param> params()
{
    std::vector<Param> out;
    for (const auto& t : suite())
        for (const auto& c : k_configs)
            out.push_back({t, c});
    return out;
}

INSTANTIATE_TEST_SUITE_P(Suite, SuiteCounts, ::testing::ValuesIn(params()),
                         [](const ::testing::TestParamInfo<Param>& i)
                         {
                             std::string n = i.param.task.name + "_" + i.param.config.name;
                             for (char& c : n)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return n;
                         });

TEST(Brfs, DeterministicIdsIndependentOfThreadCount)
{
    for (const auto& t : suite())
    {
        if (t.states > std::min<u64>(suite_state_limit(), 200000))
            continue;
        for (auto atoms : {TaskOptions::Atoms::Lazy, TaskOptions::Atoms::Frozen})
        {
            TaskOptions to;
            to.atoms = atoms;
            const auto task = Task::from_text_file(task_path(t.name), to);
            BrfsOptions bo;
            bo.stop_at_goal = false;
            bo.fingerprint = true;
            bo.store = BrfsOptions::Store::Flat;
            const u64 ref = brfs(*task, bo).fingerprint;
            bo.store = BrfsOptions::Store::Chunked;
            EXPECT_EQ(brfs(*task, bo).fingerprint, ref) << t.name;
            bo.store = BrfsOptions::Store::Concurrent;
            for (u32 T : {1u, 3u, 8u})
            {
                bo.threads = T;
                const BrfsResult r = brfs(*task, bo);
                EXPECT_EQ(r.states, t.states) << t.name << " T=" << T;
                EXPECT_EQ(r.fingerprint, ref) << t.name << " T=" << T;
            }
        }
    }
}

TEST(Brfs, StopAtGoalReturnsAValidPlan)
{
    for (const char* name : {"gripper__prob05", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "depot__p02"})
        for (auto store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked, BrfsOptions::Store::Compact})
        {
            const auto task = Task::from_text_file(task_path(name));
            BrfsOptions bo;
            bo.store = store;
            bo.stop_at_goal = true;
            const BrfsResult r = brfs(*task, bo);
            ASSERT_TRUE(r.solved) << name;
            const WorkspaceLease lease = task->workspace();
            Successors& succ = lease->successors();
            State s = task->initial_state();
            for (const Action& a : r.plan)
            {
                ASSERT_TRUE(succ.is_applicable(s, a.label())) << name << " " << task->format(a.label());
                s = succ.apply(s, a.label());
            }
            EXPECT_TRUE(task->is_goal(s)) << name;
        }
}

TEST(Brfs, MaxStatesStopsEarly)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    for (auto store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked, BrfsOptions::Store::Concurrent})
    {
        BrfsOptions bo;
        bo.stop_at_goal = false;
        bo.store = store;
        bo.max_states = 1000;
        bo.threads = store == BrfsOptions::Store::Concurrent ? 2 : 1;
        const BrfsResult r = brfs(*task, bo);
        EXPECT_FALSE(r.exhausted);
        EXPECT_GE(r.states, 1000u);
        EXPECT_LT(r.states, 100000u);
    }
}

TEST(Brfs, SingleThreadedStoresRejectThreads)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    BrfsOptions bo;
    bo.stop_at_goal = false;
    bo.store = BrfsOptions::Store::Flat;
    bo.threads = 2;
    EXPECT_THROW((void) brfs(*task, bo), std::invalid_argument);
}

bool replays(const Task& task, const std::vector<Action>& plan)
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
    return task.is_goal(s);
}

TEST(Brfs, LayerOrderings)
{
    using K = search::LayerOrdering::Kind;
    for (const char* name : {"gripper__prob05", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "depot__p02"})
        for (auto store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked})
        {
            const auto task = Task::from_text_file(task_path(name));
            BrfsOptions q;
            q.store = store;
            q.witness_pruning = false;
            q.fingerprint = true;
            q.stop_at_goal = false;
            const BrfsResult all = brfs(*task, q);
            q.stop_at_goal = true;
            q.fingerprint = false;
            const BrfsResult first = brfs(*task, q);
            ASSERT_TRUE(first.solved) << name;
            for (K kind : {K::InOrder, K::Reverse, K::Randomized, K::GoalCount})
            {
                BrfsOptions o = q;
                o.layers.kind = kind;
                o.layers.seed = 9;
                o.stop_at_goal = false;
                o.fingerprint = true;
                const BrfsResult r = brfs(*task, o);
                EXPECT_EQ(r.states, all.states) << name;  // the order within layers changes ids, not the states
                EXPECT_EQ(r.expanded, all.expanded) << name;
                EXPECT_EQ(r.generated, all.generated) << name;
                if (kind == K::InOrder)
                {
                    EXPECT_EQ(r.fingerprint, all.fingerprint) << name;  // the queued BrFS, id for id
                }
                o.stop_at_goal = true;
                o.fingerprint = false;
                const BrfsResult g = brfs(*task, o);
                ASSERT_TRUE(g.solved) << name;
                EXPECT_EQ(g.plan.size(), first.plan.size()) << name;  // still layer by layer: a shallowest goal
                EXPECT_TRUE(replays(*task, g.plan)) << name;
            }
        }
}

TEST(Brfs, ReverseAndGoalCountChangeTheExpansionOrder)
{
    using K = search::LayerOrdering::Kind;
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    BrfsOptions q;
    q.stop_at_goal = true;
    const u64 queued = brfs(*task, q).expanded;
    std::vector<u64> expanded;
    for (K kind : {K::Reverse, K::GoalCount})
    {
        BrfsOptions o = q;
        o.layers.kind = kind;
        expanded.push_back(brfs(*task, o).expanded);
    }
    EXPECT_TRUE(expanded[0] != queued || expanded[1] != queued);
    // goal count first: the goal is found earlier within its layer than with fewer satisfied literals first
    BrfsOptions more = q, fewer = q;
    more.layers.kind = fewer.layers.kind = K::GoalCount;
    fewer.layers.prefer_more_satisfied_goals = false;
    EXPECT_LE(brfs(*task, more).expanded, brfs(*task, fewer).expanded);
}

TEST(Brfs, TruncatedLayers)
{
    // goal count first with at most 4 states per layer: a narrow descent on the satisfied goal literals
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    BrfsOptions o;
    o.stop_at_goal = true;
    o.layers.kind = search::LayerOrdering::Kind::GoalCount;
    o.layers.max_next_layer_states = 4;
    const BrfsResult r = brfs(*task, o);
    if (r.solved)
    {
        EXPECT_TRUE(replays(*task, r.plan));
    }
    else
    {
        EXPECT_TRUE(r.exhausted);
    }
    BrfsOptions q;
    q.stop_at_goal = true;
    EXPECT_LT(r.expanded, brfs(*task, q).expanded);
    o.store = BrfsOptions::Store::Compact;
    EXPECT_THROW((void)brfs(*task, o), std::invalid_argument);
    o.store = BrfsOptions::Store::Flat;
    o.layers.max_next_layer_states = 0;
    EXPECT_THROW((void)brfs(*task, o), std::invalid_argument);
    o.layers = {};
    o.layers.max_next_layer_states = 5;  // Queue
    EXPECT_THROW((void)brfs(*task, o), std::invalid_argument);
}

TEST(Brfs, ThreadsReturnAShortestPlan)
{
    for (const char* name : {"gripper__prob05", "depot__p02", "philosophers__p03-phil4"})
    {
        const auto task = Task::from_text_file(task_path(name));
        BrfsOptions bo;
        bo.stop_at_goal = true;
        const BrfsResult one = brfs(*task, bo);
        ASSERT_TRUE(one.solved) << name;
        bo.store = BrfsOptions::Store::Concurrent;
        for (u32 T : {1u, 2u, 4u})
        {
            bo.threads = T;
            const BrfsResult r = brfs(*task, bo);
            EXPECT_EQ(r.status, search::SearchStatus::Solved) << name << " T=" << T;
            EXPECT_EQ(r.plan.size(), one.plan.size()) << name << " T=" << T;
            EXPECT_TRUE(replays(*task, r.plan)) << name << " T=" << T;
        }
    }
}

/// Counts the events it receives; worker observers are made per thread and summed by the root.
struct Counting : search::SearchObserver
{
    u64 starts = 0, expanded = 0, generated = 0, fresh = 0, passes = 0, solutions = 0, ends = 0;
    u64 pass_expanded = 0, pass_generated = 0, pass_states = 0;
    search::SearchStatus status = search::SearchStatus::Failed;
    std::vector<Action> plan;
    bool parallel = false;
    std::vector<std::shared_ptr<Counting>> workers;

    void on_start(StateView) override { ++starts; }
    void on_expand(u64, StateView) override { ++expanded; }
    void on_generate(u64, const Action&, u64, StateView, bool is_new) override
    {
        ++generated;
        fresh += is_new;
    }
    void on_pass(u32, const search::SearchStatistics& s) override
    {
        ++passes;
        pass_expanded += s.expanded;
        pass_generated += s.generated;
        pass_states += s.states;
    }
    void on_solution(std::span<const Action> p, double) override
    {
        ++solutions;
        plan.assign(p.begin(), p.end());
    }
    void on_end(search::SearchStatus s, const search::SearchStatistics&) override
    {
        ++ends;
        status = s;
    }
    std::shared_ptr<SearchObserver> make_worker(u32) override
    {
        if (!parallel)
            return nullptr;
        workers.push_back(std::make_shared<Counting>());
        return workers.back();
    }
    [[nodiscard]] std::array<u64, 3> hot() const
    {
        std::array<u64, 3> n{expanded, generated, fresh};
        for (const auto& w : workers)
            n = {n[0] + w->expanded, n[1] + w->generated, n[2] + w->fresh};
        return n;
    }
};

TEST(Brfs, ObserverEventsEqualTheStatistics)
{
    const auto task = Task::from_text_file(task_path("depot__p02"));
    using S = BrfsOptions::Store;
    const std::tuple<S, u32, bool> cases[] = {{S::Flat, 1, false},       {S::Chunked, 1, false},
                                              {S::Compact, 1, false},    {S::Concurrent, 1, false},
                                              {S::Concurrent, 4, false}, {S::Concurrent, 2, true},
                                              {S::Concurrent, 4, true}};
    for (const auto& [store, threads, parallel] : cases)
        for (bool stop : {false, true})
        {
            Counting o;
            o.parallel = parallel;
            BrfsOptions bo;
            bo.store = store;
            bo.threads = threads;
            bo.stop_at_goal = stop;
            bo.observer = &o;
            const BrfsResult r = brfs(*task, bo);
            const std::string where = std::string(r.store) + " T=" + std::to_string(threads)
                                      + (parallel ? " workers" : "") + (stop ? " stop" : "");
            // without make_worker the search runs on one thread and sends everything to the root
            EXPECT_EQ(r.threads, parallel ? threads : 1u) << where;
            EXPECT_EQ(o.workers.size(), parallel ? threads : 0u) << where;
            if (parallel)
            {
                EXPECT_EQ(o.expanded + o.generated, 0u) << where;
            }
            const auto [expanded, generated, fresh] = o.hot();
            EXPECT_EQ(expanded, r.expanded) << where;
            EXPECT_EQ(generated, r.generated) << where;
            EXPECT_EQ(o.starts, 1u) << where;
            EXPECT_EQ(o.ends, 1u) << where;
            EXPECT_EQ(o.status, r.status) << where;
            EXPECT_EQ(o.passes, r.layers) << where;
            EXPECT_EQ(o.pass_expanded, r.expanded) << where;
            EXPECT_EQ(o.pass_generated, r.generated) << where;
            // a goal state is expanded either way: solved, with the plan to the first one
            EXPECT_EQ(r.status, search::SearchStatus::Solved) << where;
            ASSERT_EQ(o.solutions, 1u) << where;
            EXPECT_EQ(o.plan.size(), r.plan.size()) << where;
            EXPECT_TRUE(replays(*task, o.plan)) << where;
            EXPECT_EQ(r.exhausted, !stop) << where;
            if (!stop)
            {
                EXPECT_EQ(fresh, r.states - 1) << where;
                EXPECT_EQ(o.pass_states, r.states - 1) << where;
            }
        }
}

TEST(Brfs, BudgetsCancelAndProgressSetTheStatus)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    using S = BrfsOptions::Store;
    using search::SearchStatus;
    const std::pair<S, u32> cases[] = {{S::Flat, 1}, {S::Chunked, 1}, {S::Compact, 1}, {S::Concurrent, 3}};
    for (const auto& [store, threads] : cases)
    {
        const std::string where = "store " + std::to_string(static_cast<int>(store));
        BrfsOptions bo;
        bo.stop_at_goal = false;
        bo.store = store;
        bo.threads = threads;
        bo.max_seconds = 0;
        EXPECT_EQ(brfs(*task, bo).status, SearchStatus::OutOfTime) << where;
        bo.max_seconds = std::numeric_limits<double>::infinity();
        bo.cancel.request();
        BrfsResult r = brfs(*task, bo);
        EXPECT_EQ(r.status, SearchStatus::Cancelled) << where;
        EXPECT_FALSE(r.exhausted) << where;
        bo.cancel = {};
        bo.max_states = 1000;
        EXPECT_EQ(brfs(*task, bo).status, SearchStatus::OutOfStates) << where;
        bo.max_states = std::numeric_limits<u64>::max();
        struct Stop : search::SearchObserver
        {
            u64 calls = 0;
            bool on_progress(const search::SearchStatistics& s) override
            {
                ++calls;
                return s.expanded < 500;
            }
            std::shared_ptr<SearchObserver> make_worker(u32) override { return std::make_shared<Stop>(); }
        } stop;
        bo.observer = &stop;
        bo.progress_interval = 100;
        r = brfs(*task, bo);
        EXPECT_EQ(r.status, SearchStatus::Cancelled) << where;
        EXPECT_LT(r.expanded, (500u + 100u + 64u) * threads) << where;  // on_progress sees each worker's counts
        bo.observer = nullptr;
        r = brfs(*task, bo);
        EXPECT_EQ(r.status, SearchStatus::Solved) << where;  // the whole space, goals included
        EXPECT_TRUE(r.exhausted) << where;
    }
}

TEST(Brfs, CancelFromAnotherThread)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    for (u32 T : {1u, 4u})
    {
        BrfsOptions bo;
        bo.stop_at_goal = false;
        bo.store = BrfsOptions::Store::Concurrent;
        bo.threads = T;
        std::atomic<bool> started{false};
        std::thread canceller([&started, token = bo.cancel] {
            while (!started.load())
                std::this_thread::yield();
            token.request();
        });
        struct Start : search::SearchObserver
        {
            std::atomic<bool>* started;
            void on_start(StateView) override { started->store(true); }
        } start;
        start.started = &started;
        bo.observer = &start;
        const BrfsResult r = brfs(*task, bo);
        canceller.join();
        // the search may finish before the request lands; when it does not, it stops as cancelled
        EXPECT_TRUE(r.status == search::SearchStatus::Cancelled || r.status == search::SearchStatus::Solved)
            << search::to_string(r.status);
        EXPECT_EQ(r.exhausted, r.status == search::SearchStatus::Solved);
    }
}
}  // namespace
