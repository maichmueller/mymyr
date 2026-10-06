// BrFS: state, generated and goal-state counts equal the probe's on the suite in every mode and with every store
// (the full suite with MYMYR_TEST_FULL_SUITE=1, otherwise the tasks up to 100k states), deterministic ids
// independent of the thread count (gate 4), and plans that reach the goal.

#include "../support/suite.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

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
            Successors& succ = task->workspace().successors();
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
    bo.store = BrfsOptions::Store::Flat;
    bo.threads = 2;
    EXPECT_THROW((void) brfs(*task, bo), std::invalid_argument);
}

bool replays(const Task& task, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
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
}  // namespace
