// Device IW search tests (run on GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the device RNG (cuda/device_rng.hpp) is mymyr::SplitMix64 bit for bit;
//   - every device search equals search::iw() from the same start (status, per-pass expanded / generated /
//     generated_in_tree / skipped, plan, goal state, cost) on random-walk starts of every suite task, frozen and lazy:
//     optimized IW(1), IW(1) with a RootOnly width-0 pass, IW(2) with a state budget; budgets; per-search goals;
//     exact mode does not depend on the group size, the chunk size or the emit budget, and a device loop hands a
//     layer's last row over the emit budget to the host; a chunk with more distinct rows than the generator was sized
//     for is redone; device rollouts equal
//     search::find_rollouts_parallel for the same seeds (status, plan, passes, reached fluent atoms), with and without
//     truncated layers, and do not depend on the launch configuration or the run; a reused object stops capturing
//     after its first runs;
//   - relaxed novelty returns valid plans; numeric tasks are refused.
// MYMYR_TEST_SANITIZER=1 shrinks the detail tests for compute-sanitizer.
// MYMYR_TEST_FULL_SUITE=1 runs the gate's sizes (IW(1): 310 starts per task and mode; IW(2): 104; 128 rollouts).

#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "../cpp/support/wide_task.hpp"
#include "mymyr/core/random.hpp"
#include "mymyr/cuda/device_rng.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/multi_iw_kernels.hpp"
#include "mymyr/cuda/rollouts.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/search/goal.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/successor/conditions.hpp"
#include "mymyr/successor/successors.hpp"

#ifdef MYMYR_HAS_FRONTEND
#include "mymyr/frontend/domain.hpp"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (cuda::device_count() == 0)                                                                                  \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{4} << 30;  // GPU 0 may be shared: cap this process at 7 GB
    return cuda::DeviceContext::create(0, o);
}

TaskPtr load(const std::string& name, bool frozen)
{
    TaskOptions o;
    o.atoms = frozen ? TaskOptions::Atoms::Dense : TaskOptions::Atoms::Lazy;
    return Task::from_text_file(task_path(name), o);
}

bool full_suite()
{
    const char* e = std::getenv("MYMYR_TEST_FULL_SUITE");
    return e && std::string(e) == "1";
}

std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const SuiteTask& t : suite())
        for (bool frozen : {true, false})
            out.emplace_back(t.name, frozen);
    return out;
}

std::string param_name(const ::testing::TestParamInfo<std::tuple<std::string, bool>>& info)
{
    std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
    std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
    return n;
}

search::IwOptions cpu_options(const cuda::MultiIwOptions& d, const State& start)
{
    search::IwOptions o;
    o.max_width = d.max_width;
    o.optimize_iw1 = d.optimize_iw1;
    o.width_zero = d.width_zero;
    o.witness_pruning = d.witness_pruning;
    o.canonical_order = d.canonical_order;
    o.control.budget = d.budget;
    if (d.max_next_layer_states != ~u32{0})
        o.layers = {.kind = search::LayerOrdering::Kind::InOrder, .max_next_layer_states = d.max_next_layer_states};
    o.start = start;
    return o;
}

/// Everything search::iw() reports that the device reproduces (not the timings, peaks or lazy slot counts).
void expect_same(const search::IwResult& d, const search::IwResult& c, const std::string& what)
{
    SCOPED_TRACE(what);
    EXPECT_EQ(d.status, c.status);
    EXPECT_EQ(d.effective_width, c.effective_width);
    ASSERT_EQ(d.passes.size(), c.passes.size());
    for (usize k = 0; k < c.passes.size(); ++k)
    {
        SCOPED_TRACE("pass " + std::to_string(k));
        EXPECT_EQ(d.passes[k].width, c.passes[k].width);
        EXPECT_EQ(d.passes[k].status, c.passes[k].status);
        EXPECT_EQ(d.passes[k].placeholder, c.passes[k].placeholder);
        EXPECT_EQ(d.passes[k].expanded, c.passes[k].expanded);
        EXPECT_EQ(d.passes[k].generated, c.passes[k].generated);
        EXPECT_EQ(d.passes[k].generated_in_tree, c.passes[k].generated_in_tree);
        EXPECT_EQ(d.passes[k].skipped, c.passes[k].skipped);
    }
    EXPECT_EQ(d.stats.expanded, c.stats.expanded);
    EXPECT_EQ(d.stats.generated, c.stats.generated);
    EXPECT_EQ(d.stats.states, c.stats.states);
    EXPECT_TRUE(d.plan == c.plan) << "plans differ: " << d.plan.size() << " vs " << c.plan.size() << " steps";
    EXPECT_EQ(d.cost, c.cost);
    ASSERT_EQ(d.goal_state.has_value(), c.goal_state.has_value());
    if (c.goal_state)
    {
        EXPECT_TRUE(d.goal_state->view() == c.goal_state->view());
    }
}

/// Device results equal (the same device semantics under two launch configurations).
void expect_same_batch(const cuda::MultiIwBatch& a, const cuda::MultiIwBatch& b, const std::string& what)
{
    SCOPED_TRACE(what);
    ASSERT_EQ(a.n, b.n);
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.plan_length, b.plan_length);
    EXPECT_EQ(a.plan_offsets, b.plan_offsets);
    EXPECT_EQ(a.plan_labels, b.plan_labels);
    EXPECT_EQ(a.effective_width, b.effective_width);
    ASSERT_EQ(a.words, b.words);
    EXPECT_EQ(a.goal_rows, b.goal_rows);
    EXPECT_EQ(a.reached, b.reached);
    for (u32 i = 0; i < a.n; ++i)
    {
        ASSERT_EQ(a.passes(i).size(), b.passes(i).size());
        for (usize k = 0; k < a.passes(i).size(); ++k)
        {
            EXPECT_EQ(a.passes(i)[k].status, b.passes(i)[k].status);
            EXPECT_EQ(a.passes(i)[k].expanded, b.passes(i)[k].expanded);
            EXPECT_EQ(a.passes(i)[k].generated, b.passes(i)[k].generated);
            EXPECT_EQ(a.passes(i)[k].generated_in_tree, b.passes(i)[k].generated_in_tree);
            EXPECT_EQ(a.passes(i)[k].skipped, b.passes(i)[k].skipped);
        }
    }
}

/// Per-start goals from other walk states: up to `atoms` atoms of state (7 i + 3) that start i lacks, and, with
/// `negative`, one atom that state lacks. Random-walk starts rarely solve the task's conjunctive goal at width 1; these
/// goals give the searches plans of every length.
std::vector<search::GoalSpec::AtomGoal> walk_goals(const Task& task, const std::vector<State>& starts, u32 atoms, bool negative)
{
    std::vector<search::GoalSpec::AtomGoal> goals(starts.size());
    for (usize i = 0; i < starts.size(); ++i)
    {
        const State& t = starts[(i * 7 + 3) % starts.size()];
        const State& s = starts[i];
        std::vector<u32> fresh;
        bits::for_each(t.data(), t.size_words(), [&](u64 a)
                       {
                           if (!bits::test(s.data(), s.size_words(), a))
                               fresh.push_back(static_cast<u32>(a));
                       });
        for (usize k = 0; k < fresh.size() && goals[i].positive.size() < atoms; k += 1 + fresh.size() / (atoms + 1))
            goals[i].positive.push_back(SlotId{fresh[k]});
        if (negative)
            for (u32 a = 0; a < std::min<u32>(64 * t.size_words(), task.atoms().fluent_slots()); ++a)
                if (!bits::test(t.data(), t.size_words(), a) && a % 5 == i % 5)
                {
                    goals[i].negative.push_back(SlotId{a});
                    break;
                }
    }
    return goals;
}

/// multi_iw against iw() per start (goals: empty, the task's goal; or one per start).
void check_against_cpu(const TaskPtr& task, const std::vector<State>& starts, const cuda::MultiIwOptions& o, const std::string& what,
                       const std::vector<search::GoalSpec::AtomGoal>& goals = {})
{
    const std::vector<search::IwResult> d = cuda::multi_iw(context(), task, starts, o, goals);
    ASSERT_EQ(d.size(), starts.size());
    u32 solved = 0;
    for (usize i = 0; i < starts.size(); ++i)
    {
        search::IwOptions co = cpu_options(o, starts[i]);
        if (!goals.empty())
        {
            co.control.goal.kind = search::GoalSpec::Kind::AnyOf;
            co.control.goal.goals = {goals[i]};
        }
        const search::IwResult c = search::iw(*task, co);
        if (std::getenv("MYMYR_DEVICE_IW_DEBUG") && i < 3)
            std::printf("start %zu: cpu status %d expanded %llu generated %llu plan %zu msg '%s' | device status %d expanded %llu generated %llu plan %zu msg '%s'\n",
                        i, static_cast<int>(c.status), static_cast<unsigned long long>(c.stats.expanded),
                        static_cast<unsigned long long>(c.stats.generated), c.plan.size(), c.message.c_str(),
                        static_cast<int>(d[i].status), static_cast<unsigned long long>(d[i].stats.expanded),
                        static_cast<unsigned long long>(d[i].stats.generated), d[i].plan.size(), d[i].message.c_str());
        expect_same(d[i], c, what + ", start " + std::to_string(i));
        solved += c.status == search::SearchStatus::Solved;
        if (::testing::Test::HasFailure())
            return;  // one diverging search is enough
    }
    ::testing::Test::RecordProperty(what + "_solved", static_cast<int>(solved));
    ::testing::Test::RecordProperty(what + "_searches", static_cast<int>(starts.size()));
}

std::vector<State> walks(const Task& task, bool gate_size, u64 seed)
{
    return gate_size ? device_ref_walks(task, 10, 30, seed) : device_ref_walks(task, 3, 20, seed);
}

// ------------------------------------------------------------------------------------------------ RNG

TEST(DeviceSplitMix64Rng, IsSplitMix64BitForBit)
{
    for (u64 seed : {u64{0}, u64{1}, u64{42}, ~u64{0}, u64{0x9e3779b97f4a7c15ULL}})
    {
        SplitMix64 ref(seed);
        u64 st = seed;
        for (int i = 0; i < 1000; ++i)
            ASSERT_EQ(cuda::device_rng::next(st), ref.next());
        for (u64 n : {u64{1}, u64{2}, u64{3}, u64{7}, u64{1000}, u64{1} << 33, ~u64{0}})
            for (int i = 0; i < 100; ++i)
                ASSERT_EQ(cuda::device_rng::bounded(st, n), ref.bounded(n));
        for (u32 n : {0u, 1u, 2u, 5u, 63u, 64u, 65u, 1000u})
        {
            std::vector<u32> a(n), b(n);
            std::iota(a.begin(), a.end(), 0u);
            std::iota(b.begin(), b.end(), 0u);
            cuda::device_rng::shuffle(st, a.data(), n);
            ref.shuffle(std::span<u32>(b));
            ASSERT_EQ(a, b);
            ASSERT_EQ(st, ref.state());
        }
    }
}

// ------------------------------------------------------------------------------------------------ gate 1: suite

class CudaMultiIwSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(CudaMultiIwSuite, OptimizedIw1EqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen);
    const std::vector<State> starts = walks(*task, full_suite(), 3);
    cuda::MultiIwOptions o;  // max_arity 1, optimized
    check_against_cpu(task, starts, o, "iw1");
    check_against_cpu(task, starts, o, "iw1_atom_goals", walk_goals(*task, starts, 1, false));
    check_against_cpu(task, starts, o, "iw1_pair_goals", walk_goals(*task, starts, 2, true));
}

TEST_P(CudaMultiIwSuite, Iw1WithRootOnlyWidthZeroEqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen);
    const std::vector<State> starts = walks(*task, false, 11);
    cuda::MultiIwOptions o;
    o.optimize_iw1 = false;
    o.width_zero = search::WidthZero::RootOnly;
    check_against_cpu(task, starts, o, "iw1_root_only");
    check_against_cpu(task, starts, o, "iw1_root_only_goals", walk_goals(*task, starts, 1, false));
    o.width_zero = search::WidthZero::ExpandDepthOne;
    o.witness_pruning = true;
    check_against_cpu(task, starts, o, "iw1_plain_witness");
    check_against_cpu(task, starts, o, "iw1_plain_witness_goals", walk_goals(*task, starts, 2, false));
}

TEST_P(CudaMultiIwSuite, Iw2EqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen);
    const std::vector<State> starts = full_suite() ? device_ref_walks(*task, 4, 25, 7) : device_ref_walks(*task, 2, 11, 7);
    cuda::MultiIwOptions o;
    o.max_width = 2;
    o.budget.max_states = 20000;  // per pass: bounds the CPU reference's time on the large tasks
    check_against_cpu(task, starts, o, "iw2");
    check_against_cpu(task, starts, o, "iw2_goals", walk_goals(*task, starts, 3, true));
}

TEST_P(CudaMultiIwSuite, RolloutsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen);
    const u32 n = full_suite() ? 128 : 24;
    // the task's goal, and a goal of two atoms of a random-walk state (solved at width 1 or 2)
    const std::vector<State> w = device_ref_walks(*task, 1, 12, 31);
    const search::GoalSpec::AtomGoal walk_goal = walk_goals(*task, {w.front(), w.back()}, 2, false)[0];
    for (bool own_goal : {false, true})
        for (u32 next : {~u32{0}, 3u})
        {
            SCOPED_TRACE("max_next_layer_states " + std::to_string(next) + (own_goal ? ", walk goal" : ""));
            std::vector<u64> seeds(n);
            for (u32 i = 0; i < n; ++i)
                seeds[i] = 1000 * i + 17;
            cuda::DeviceRolloutOptions d;
            d.seeds = seeds;
            d.iw.max_next_layer_states = next;
            search::ParallelRolloutOptions c;
            c.iw.max_width = 1;
            c.seeds = seeds;
            c.threads = 4;
            c.max_next_layer_states = next;
            if (own_goal)
            {
                d.goals.assign(n, walk_goal);
                c.iw.control.goal.kind = search::GoalSpec::Kind::AnyOf;
                c.iw.control.goal.goals = {walk_goal};
            }
            const cuda::DeviceRolloutsResult r = cuda::find_rollouts(context(), task, d);
            const search::ParallelRolloutsResult ref = search::find_rollouts_parallel(*task, c);
            ASSERT_EQ(r.rollouts.size(), ref.rollouts.size());
            u32 solved = 0;
            for (u32 i = 0; i < n; ++i)
            {
                expect_same(r.rollouts[i].search, ref.rollouts[i].search, "rollout " + std::to_string(i));
                EXPECT_EQ(r.rollouts[i].reached_fluent_atoms, ref.rollouts[i].reached_fluent_atoms) << "rollout " << i;
                solved += ref.rollouts[i].search.status == search::SearchStatus::Solved;
                if (HasFailure())
                    return;
            }
            RecordProperty(std::string(own_goal ? "walk_goal_" : "") + "solved_" + std::to_string(next), static_cast<int>(solved));
            RecordProperty(std::string(own_goal ? "walk_goal_" : "") + "rollouts_" + std::to_string(next), static_cast<int>(n));
        }
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaMultiIwSuite, ::testing::ValuesIn(params()), param_name);

TEST(CudaGoals, DerivedConditionsEqualCpuSearches)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("philosophers__p03-phil4", false);
    const GroundCondition goal = GroundCondition::goal(*task);  // derived literals
    const std::optional<search::GoalSpec::AtomGoal> g = search::atom_goal(*task, goal);
    ASSERT_TRUE(g.has_value());
    ASSERT_FALSE(g->fluent_only());
    const std::vector<State> starts{task->initial_state()};
    cuda::MultiIwOptions options;
    options.max_width = 2;
    options.budget.max_states = 3000;
    check_against_cpu(task, starts, options, "derived goal", {*g});
    cuda::DeviceRolloutOptions d;
    d.seeds = {1};
    d.iw = options;
    d.goals = {*g};
    search::IwOptions cpu;
    cpu.max_width = 2;
    cpu.control.budget = options.budget;
    cpu.control.goal.kind = search::GoalSpec::Kind::AnyOf;
    cpu.control.goal.goals = {*g};
    cpu.layers = {.kind = search::LayerOrdering::Kind::Randomized, .seed = 1};
    expect_same(cuda::find_rollouts(context(), task, d).rollouts[0].search, search::iw(*task, cpu), "derived rollout");
    // the same goal over fluent literals only runs
    search::GoalSpec::AtomGoal fluent = *g;
    fluent.derived_positive.clear();
    fluent.derived_negative.clear();
    fluent.positive.clear();
    ASSERT_TRUE(fluent.fluent_only());
    EXPECT_NO_THROW((void)cuda::multi_iw(context(), task, starts, cuda::MultiIwOptions{}, std::span(&fluent, 1)));
}

// ------------------------------------------------------------------------------------------------ gate 1: details

/// MYMYR_TEST_SANITIZER=1: smaller detail tests for compute-sanitizer (one-row chunks cost milliseconds per launch).
bool sanitizer_size()
{
    const char* e = std::getenv("MYMYR_TEST_SANITIZER");
    return e && std::string(e) == "1";
}

const std::vector<const char*>& detail_tasks()
{
    // CPU-fallback schemas, axioms, derived goals, conditional effects, wide and lazy states
    static const std::vector<const char*> t = {"depot__p02", "philosophers__p03-phil4", "organic-synthesis-opt18-strips__p20",
                                               "caldera-split-opt18-adl__p04", "miconic-simpleadl__s10-2", "blocks__probBLOCKS-8-0"};
    static const std::vector<const char*> small = {"philosophers__p03-phil4", "miconic-simpleadl__s10-2"};
    return sanitizer_size() ? small : t;
}

/// The rollout tests' tasks.
std::vector<const char*> rollout_tasks(std::vector<const char*> all)
{
    if (sanitizer_size())
        all.resize(std::min<usize>(all.size(), 2));
    return all;
}

TEST(CudaMultiIw, ExactModeDoesNotDependOnTheLaunchConfiguration)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : detail_tasks())
        for (u32 arity : {1u, 2u})
        {
            SCOPED_TRACE(std::string(name) + " arity " + std::to_string(arity));
            const auto task = load(name, true);
            const std::vector<State> starts = sanitizer_size() ? device_ref_walks(*task, 1, 6, 23) : device_ref_walks(*task, 2, 15, 23);
            cuda::MultiIwOptions o;
            o.max_width = arity;
            o.budget.max_states = arity == 1 ? 5000 : 1500;  // one-row chunks at arity 2 are slow (host round trips)
            cuda::DeviceMultiIw ref_run(context(), task, o);
            const cuda::MultiIwBatch ref = ref_run.run(starts);
            struct Config
            {
                u32 max_searches, chunk_states;
                u64 emit_budget, view_bytes;
            };
            for (const Config& c : {Config{1, 1u << 20, 1u << 24, 0}, Config{7, 1u << 20, 1u << 24, 0}, Config{0, arity == 1 ? 1u : 16u, 1u << 24, 0},
                                    Config{0, 5, 64, 0}, Config{3, 1000, 1, 4096}, Config{0, 1u << 20, 4096, 0}})
            {
                SCOPED_TRACE("max_searches " + std::to_string(c.max_searches) + " chunk " + std::to_string(c.chunk_states) +
                             " emit " + std::to_string(c.emit_budget));
                cuda::MultiIwOptions v = o;
                v.max_searches = c.max_searches;
                v.chunk_states = c.chunk_states;
                v.emit_budget = c.emit_budget;
                v.view_bytes = c.view_bytes;
                cuda::DeviceMultiIw run(context(), task, v);
                expect_same_batch(run.run(starts), ref, name);
            }
        }
}

TEST(CudaMultiIw, CaptureDedupAndCapacitiesDoNotChangeTheResults)
{
    // Graphs replayed or launches enqueued, parents deduplicated or not, chunks redone at grown capacities (one
    // candidate per row to start) or split by the chunk budget: the same searches.
    SKIP_WITHOUT_GPU();
    u64 all_replays = 0;  // (tasks with host work in their chunks are not captured)
    for (const char* name : detail_tasks())
        for (u32 arity : {1u, 2u})
        {
            SCOPED_TRACE(std::string(name) + " arity " + std::to_string(arity));
            const auto task = load(name, true);
            const std::vector<State> starts = sanitizer_size() ? device_ref_walks(*task, 1, 6, 29) : device_ref_walks(*task, 2, 15, 29);
            cuda::MultiIwOptions o;
            o.max_width = arity;
            o.budget.max_states = arity == 1 ? 5000 : 1500;
            o.graphs = false;
            cuda::DeviceMultiIw ref_run(context(), task, o);
            const cuda::MultiIwBatch ref = ref_run.run(starts);
            EXPECT_EQ(ref.stats.replays, 0u);

            // the same object three times: chunk shapes recur, are captured and replayed
            cuda::MultiIwOptions g = o;
            g.graphs = true;
            cuda::DeviceMultiIw graph_run(context(), task, g);
            u64 replays = 0;
            for (int rep = 0; rep < 3; ++rep)
            {
                const cuda::MultiIwBatch b = graph_run.run(starts);
                expect_same_batch(b, ref, "graphs, run " + std::to_string(rep));
                replays += b.stats.replays;
            }
            all_replays += replays;

            struct Config
            {
                bool dedup;
                u32 per_row;
                u64 chunk_bytes;
            };
            for (const Config& c : {Config{false, 8, u64{1} << 28}, Config{true, 1, u64{1} << 28}, Config{false, 1, u64{1} << 28},
                                    Config{true, 8, u64{1} << 16}})
            {
                SCOPED_TRACE(std::string("dedup ") + (c.dedup ? "1" : "0") + " per row " + std::to_string(c.per_row) + " chunk bytes " +
                             std::to_string(c.chunk_bytes));
                cuda::MultiIwOptions v = g;
                v.dedup_parents = c.dedup;
                v.capacity_per_row = c.per_row;
                v.chunk_bytes = c.chunk_bytes;
                cuda::DeviceMultiIw run(context(), task, v);
                const cuda::MultiIwBatch b = run.run(starts);
                expect_same_batch(b, ref, name);
                if (c.per_row == 1)
                {
                    EXPECT_GT(b.stats.redone, 0u);
                }
                if (!c.dedup)
                {
                    EXPECT_EQ(b.stats.distinct, 0u);
                }
            }
        }
    if (cuda::GraphExec::enabled())  // MYMYR_CUDA_GRAPHS=0 (racecheck) launches every chunk from the host
    {
        EXPECT_GT(all_replays, 0u);
    }
}

TEST(CudaMultiIw, ChunksOverTheLearntDistinctRowsAreRedone)
{
    // With parent dedup the generator is sized for the most distinct rows of a chunk seen so far (with a margin);
    // a chunk with more aborts (miw::k_abort_distinct) and is redone with room for them. An object that has only met
    // copies of one start meets many distinct starts: the same searches as a fresh object's.
    SKIP_WITHOUT_GPU();
    for (const char* name : {"blocks__probBLOCKS-8-0", "gripper__prob05"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        const std::vector<State> starts = device_ref_walks(*task, 4, 40, 17);
        const std::vector<State> same(starts.size(), starts.front());
        cuda::MultiIwOptions o;
        o.budget.max_states = 3000;
        cuda::DeviceMultiIw fresh(context(), task, o);
        const cuda::MultiIwBatch ref = fresh.run(starts);
        cuda::DeviceMultiIw learnt(context(), task, o);
        (void)learnt.run(same);
        const cuda::MultiIwBatch b = learnt.run(starts);
        expect_same_batch(b, ref, name);
        EXPECT_GT(b.stats.redone, ref.stats.redone);
        EXPECT_GT(b.stats.distinct, 64u);
        RecordProperty(std::string("redone_") + name, static_cast<int>(ref.stats.redone) * 1000 + static_cast<int>(b.stats.redone));
    }
}

TEST(CudaMultiIw, OneRowChunksOverTheEmitBudgetLeaveTheDeviceLoop)
{
    // A device loop captures as many rows as the largest layer it met and caps the emission of a chunk of more than
    // one row at emit_budget. A layer's last entry alone over emit_budget cannot be split: the loop hands it to the
    // host (MultiIwStats::loop_handoffs), which runs it as one row, so the loop does not meet (and abort on) that row
    // again. Two searches from one start, then one from a start with more unseen atoms: at chunk_states 2 the root
    // layer is a chunk of two rows that fits emit_budget, then the third root alone over it.
    // The same searches as chunks launched from the host.
    SKIP_WITHOUT_GPU();
    const auto task = load("gripper__prob05", true);
    const std::vector<State> walk = device_ref_walks(*task, 1, 7, 23);
    ASSERT_EQ(walk.size(), 8u);
    for (u32 arity : {1u, 2u})
    {
        SCOPED_TRACE("arity " + std::to_string(arity));
        const std::vector<State> starts = {walk[3], walk[3], walk[0]};
        cuda::MultiIwOptions o;
        o.max_width = arity;
        o.optimize_iw1 = false;  // (a root layer of the normal rule: a device loop from the first chunk on)
        o.budget.max_states = arity == 1 ? 3000 : 600;
        o.chunk_states = 2;
        cuda::MultiIwOptions h = o;
        h.graphs = false;
        const cuda::MultiIwBatch ref = cuda::DeviceMultiIw(context(), task, h).run(starts);
        u64 handoffs = 0;
        for (u64 emit : {8u, 16u})
        {
            SCOPED_TRACE("emit_budget " + std::to_string(emit));
            cuda::MultiIwOptions v = o;
            v.emit_budget = emit;
            cuda::DeviceMultiIw run(context(), task, v);
            for (int rep = 0; rep < 2; ++rep)
            {
                const cuda::MultiIwBatch b = run.run(starts);
                expect_same_batch(b, ref, "device loops, run " + std::to_string(rep));
                handoffs += b.stats.loop_handoffs;
            }
        }
        if (cuda::GraphExec::loops_enabled())  // (the sanitizers run without device loops)
        {
            EXPECT_GT(handoffs, 0u);
        }
        else
        {
            EXPECT_EQ(handoffs, 0u);
        }
    }
}

TEST(CudaMultiIw, BudgetsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : detail_tasks())
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            const auto task = load(name, frozen);
            const std::vector<State> starts = device_ref_walks(*task, 2, 10, 5);
            for (u32 arity : {1u, 2u})
            {
                for (u64 e : {u64{0}, u64{1}, u64{5}, u64{37}})
                {
                    cuda::MultiIwOptions o;
                    o.max_width = arity;
                    o.budget.max_expanded = e;
                    o.budget.max_states = 3000;
                    check_against_cpu(task, starts, o, "max_expanded " + std::to_string(e));
                }
                for (u64 s : {u64{1}, u64{2}, u64{10}, u64{100}})
                {
                    cuda::MultiIwOptions o;
                    o.max_width = arity;
                    o.budget.max_states = s;
                    check_against_cpu(task, starts, o, "max_states " + std::to_string(s));
                }
                for (u32 d : {0u, 1u, 3u})
                {
                    cuda::MultiIwOptions o;
                    o.max_width = arity;
                    o.budget.max_depth = d;
                    o.budget.max_states = 3000;
                    check_against_cpu(task, starts, o, "max_depth " + std::to_string(d));
                }
            }
        }
}

TEST(CudaMultiIw, PerSearchGoalsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : detail_tasks())
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        const std::vector<State> starts = device_ref_walks(*task, 2, 12, 9);
        for (u32 arity : {1u, 2u})
            for (u32 atoms : {1u, 3u})
            {
                cuda::MultiIwOptions o;
                o.max_width = arity;
                o.budget.max_states = 3000;
                check_against_cpu(task, starts, o, "arity " + std::to_string(arity) + " goal atoms " + std::to_string(atoms),
                                  walk_goals(*task, starts, atoms, true));
                // mixed with the budgets of the details below
                o.budget.max_expanded = 7;
                check_against_cpu(task, starts, o, "arity " + std::to_string(arity) + " max_expanded 7",
                                  walk_goals(*task, starts, atoms, false));
            }
    }
}

TEST(CudaMultiIw, BatchedIw1FromDeviceRows)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : detail_tasks())
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        const std::vector<State> starts = device_ref_walks(*task, 3, 20, 13);
        const auto ctx = context();
        const u32 W = std::max<u32>(1, task->words()) + 1;  // padded rows
        std::vector<u64> rows(starts.size() * W, 0);
        for (usize i = 0; i < starts.size(); ++i)
            std::copy_n(starts[i].data(), starts[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * W));
        cuda::DeviceBuffer dev(ctx, rows.size() * sizeof(u64), ctx->stream());
        cuda::check(cudaMemcpyAsync(dev.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, ctx->stream()), "H2D");
        const cuda::MultiIwBatch b = cuda::batched_iw1(
            ctx, task, cuda::DeviceStarts{static_cast<const u64*>(dev.data()), W, W - 1, static_cast<u32>(starts.size())}, {}, ctx->stream());
        ASSERT_EQ(b.n, starts.size());
        for (u32 i = 0; i < b.n; ++i)
        {
            const search::IwResult c = search::iw(*task, cpu_options(cuda::MultiIwOptions{}, starts[i]));
            expect_same(b.result(i, *task, starts[i], true), c, "start " + std::to_string(i));
            EXPECT_EQ(b.plan_length[i], c.status == search::SearchStatus::Solved ? static_cast<i32>(c.plan.size()) : -1);
            if (HasFailure())
                return;
        }
    }
}

#ifdef MYMYR_HAS_FRONTEND
TEST(CudaMultiIw, WideLayerAndStateLimitsEqualCpu)
{
    SKIP_WITHOUT_GPU();
    const std::string directory = std::string(MYMYR_SOURCE_DIR) + "/tests/data/ipc/blocksworld-ipc/test/";
    const auto domain = frontend::Domain::from_file(directory + "domain.pddl");
    TaskOptions options;
    options.atoms = TaskOptions::Atoms::Dense;
    const auto task = Task::create(*domain->instantiate_file(directory + "p10-medium.pddl"), options);
    ASSERT_EQ(task->words(), 78u);
    const std::vector<State> starts = device_ref_walks(*task, 1, 1, 13);
    ASSERT_EQ(starts.size(), 2u);
    const auto ctx = context();
    const u32 W = task->words();
    std::vector<u64> rows(starts.size() * W, 0);
    for (usize i = 0; i < starts.size(); ++i)
        std::copy_n(starts[i].data(), starts[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * W));
    cuda::DeviceBuffer dev(ctx, rows.size() * sizeof(u64), ctx->stream());
    cuda::check(cudaMemcpyAsync(dev.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, ctx->stream()), "H2D");
    cuda::check(cudaStreamSynchronize(ctx->stream()), "sync");
    const cuda::DeviceStarts ds{static_cast<const u64*>(dev.data()), W, W, static_cast<u32>(starts.size())};
    bool layer_limit_changed_search = false;
    for (u32 next : {1u, 8u, 64u, ~u32{0}})
        for (u64 states : {8u, 96u})
        {
            SCOPED_TRACE("next " + std::to_string(next) + " states " + std::to_string(states));
            cuda::MultiIwOptions o;
            o.max_next_layer_states = next;
            o.budget.max_states = states;
            std::vector<search::IwResult> cpu;
            for (const State& start : starts)
            {
                const search::IwOptions c = cpu_options(o, start);
                cpu.push_back(search::iw(*task, c));
                search::IwOptions unlimited = c;
                unlimited.layers = {};
                const search::IwResult other = search::iw(*task, unlimited);
                layer_limit_changed_search |= other.stats.generated != cpu.back().stats.generated;
            }
            cuda::DeviceMultiIw run(ctx, task, o);
            const cuda::MultiIwBatch batch = run.run(starts);
            o.chunk_states = 1;
            o.max_searches = 1;
            const cuda::MultiIwBatch single = cuda::batched_iw1(ctx, task, ds, o, ctx->stream());
            expect_same_batch(batch, single, "host starts and device rows with separate searches and one-row chunks");
            for (u32 i = 0; i < starts.size(); ++i)
                expect_same(batch.result(i, *task, starts[i], true), cpu[i], "start " + std::to_string(i));
            if (HasFailure())
                return;
        }
    EXPECT_TRUE(layer_limit_changed_search);
}
#endif

TEST(CudaMultiIw, RelaxedNoveltyReturnsValidPlans)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : detail_tasks())
        for (u32 arity : {1u, 2u})
        {
            SCOPED_TRACE(std::string(name) + " arity " + std::to_string(arity));
            const auto task = load(name, true);
            const std::vector<State> starts = device_ref_walks(*task, 2, 15, 29);
            cuda::MultiIwOptions o;
            o.max_width = arity;
            o.exact = false;
            o.budget.max_states = 5000;
            const std::vector<search::IwResult> d = cuda::multi_iw(context(), task, starts, o);
            const WorkspaceLease lease = task->workspace();
            Successors& succ = lease->successors();
            for (usize i = 0; i < starts.size(); ++i)
            {
                if (d[i].status != search::SearchStatus::Solved)
                    continue;
                State s = starts[i];
                for (const Action& a : d[i].plan)
                {
                    succ.prepare(s.view());
                    bool applicable = false;
                    succ.generate<false>(
                        [&](u32 schema, const ObjectId* b, const Delta&)
                        {
                            applicable = applicable || (SchemaId{schema} == a.schema &&
                                                        std::equal(a.binding.begin(), a.binding.end(), b));
                            return true;
                        },
                        false, true);
                    ASSERT_TRUE(applicable) << "start " << i;
                    s = succ.apply(s.view(), a.label());
                }
                EXPECT_TRUE(succ.is_goal(s.view())) << "start " << i;
                EXPECT_TRUE(s.view() == d[i].goal_state->view()) << "start " << i;
            }
        }
}

TEST(CudaMultiIw, ReusedAcrossRunsWhileLazySlotsGrow)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"organic-synthesis-opt18-strips__p20", "depot__p02"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, false);
        cuda::DeviceMultiIw x(context(), task);
        const std::vector<State> first{task->initial_state()};
        (void)x.run(first);
        // the walks intern atoms on the CPU: the next run's rows are wider than the first's
        const u32 before = task->words();
        const std::vector<State> starts = device_ref_walks(*task, 3, 40, 41);
        const std::vector<search::GoalSpec::AtomGoal> goals = walk_goals(*task, starts, 2, false);
        const cuda::MultiIwBatch b = x.run(starts, goals);
        for (usize i = 0; i < starts.size(); ++i)
        {
            search::IwOptions c = cpu_options(x.options(), starts[i]);
            c.control.goal.kind = search::GoalSpec::Kind::AnyOf;
            c.control.goal.goals = {goals[i]};
            expect_same(b.result(static_cast<u32>(i), *task, starts[i], true), search::iw(*task, c), "start " + std::to_string(i));
            if (HasFailure())
                return;
        }
        RecordProperty(std::string("words_") + name, static_cast<int>(before) * 1000 + static_cast<int>(task->words()));
    }
}

TEST(CudaMultiIw, RejectsInvalidArguments)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("depot__p02", true);
    cuda::MultiIwOptions o;
    o.max_width = 3;
    EXPECT_THROW((void)cuda::DeviceMultiIw(context(), task, o), std::invalid_argument);
    cuda::DeviceMultiIw x(context(), task);
    const std::vector<State> starts{task->initial_state(), task->initial_state()};
    const std::vector<u64> one_seed{1};
    EXPECT_THROW((void)x.run(starts, {}, one_seed), std::invalid_argument);
    // no searches: an empty batch
    EXPECT_EQ(x.run(std::span<const State>{}).n, 0u);
}

// ------------------------------------------------------------------------------------------------ rollouts

TEST(CudaRollouts, DoNotDependOnTheLaunchConfigurationOrTheRun)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : rollout_tasks({"depot__p02", "philosophers__p03-phil4", "blocks__probBLOCKS-8-0", "rovers__p02"}))
        for (u32 next : {~u32{0}, 2u})
        {
            SCOPED_TRACE(std::string(name) + " next " + std::to_string(next));
            const auto task = load(name, true);
            cuda::DeviceRolloutOptions o;
            for (u64 s = 0; s < (sanitizer_size() ? 40u : 300u); ++s)
                o.seeds.push_back(s * 0x9e3779b97f4a7c15ULL);
            o.iw.max_next_layer_states = next;
            const cuda::MultiIwBatch ref = cuda::rollouts_batch(context(), task, o);
            expect_same_batch(cuda::rollouts_batch(context(), task, o), ref, "second run");
            struct Config
            {
                u32 max_searches, chunk_states;
            };
            for (const Config& c : {Config{1, 1u << 20}, Config{37, 1u << 20}, Config{0, 3}, Config{64, 100}})
            {
                cuda::DeviceRolloutOptions v = o;
                v.iw.max_searches = c.max_searches;
                v.iw.chunk_states = c.chunk_states;
                expect_same_batch(cuda::rollouts_batch(context(), task, v), ref,
                                  "max_searches " + std::to_string(c.max_searches) + " chunk " + std::to_string(c.chunk_states));
            }
            // a rollout does not depend on the other rollouts: rollout 5 alone
            cuda::DeviceRolloutOptions one = o;
            one.seeds = {o.seeds[5]};
            const cuda::MultiIwBatch b = cuda::rollouts_batch(context(), task, one);
            EXPECT_EQ(b.status[0], ref.status[5]);
            EXPECT_EQ(b.plan_length[0], ref.plan_length[5]);
            EXPECT_TRUE(std::equal(b.reached.begin(), b.reached.end(), ref.reached.begin() + 5 * static_cast<std::ptrdiff_t>(ref.reached_words)));
        }
}

TEST(CudaRollouts, RepeatedRunsReplayTheirCaptures)
{
    // A reused object's chunks capture their shapes in the first runs (up to the fourth when the host drives the
    // chunks, MYMYR_CUDA_DEVICE_LOOPS=0: their capacity estimates still grow) and replay them afterwards: the fifth and
    // sixth runs capture nothing. When the two order lists had different capacities, a captured chunk's key alternated
    // with the lists' swap at layer ends, and every other run captured again (depot: 0 0 1 0 1 1 captures per run).
    SKIP_WITHOUT_GPU();
    if (!cuda::GraphExec::enabled())
        GTEST_SKIP() << "graphs off (MYMYR_CUDA_GRAPHS=0)";
    for (const char* name : {"depot__p02", "blocks__probBLOCKS-8-0"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        std::vector<u64> seeds;
        for (u64 s = 0; s < (sanitizer_size() ? 256u : 4096u); ++s)
            seeds.push_back(1 + s * 0x9e3779b97f4a7c15ULL);
        cuda::DeviceRollouts x(context(), task);
        const cuda::MultiIwBatch ref = x.run(seeds);
        std::string captures = std::to_string(ref.stats.captures);
        for (int run = 1; run < 6; ++run)
        {
            const cuda::MultiIwBatch b = x.run(seeds);
            expect_same_batch(b, ref, "run " + std::to_string(run));
            captures += " " + std::to_string(b.stats.captures);
            if (run >= 4)
            {
                EXPECT_EQ(b.stats.captures, 0u) << "run " << run;
                EXPECT_GT(b.stats.replays, 0u) << "run " << run;
            }
        }
        RecordProperty(std::string("captures_") + name, captures);
    }
}

TEST(CudaRollouts, Iw2AndBudgetsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : rollout_tasks({"depot__p02", "philosophers__p03-phil4", "organic-synthesis-opt18-strips__p20", "miconic-simpleadl__s10-2"}))
        for (bool frozen : {true, false})
            for (u32 next : {~u32{0}, 5u})
            {
                SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy") + " next " + std::to_string(next));
                const auto task = load(name, frozen);
                std::vector<u64> seeds;
                for (u64 s = 0; s < 16; ++s)
                    seeds.push_back(s + 1);
                cuda::DeviceRolloutOptions d;
                d.seeds = seeds;
                d.iw.max_width = 2;
                d.iw.budget.max_states = 4000;
                d.iw.max_next_layer_states = next;
                const cuda::DeviceRolloutsResult r = cuda::find_rollouts(context(), task, d);
                search::ParallelRolloutOptions c;
                c.iw.max_width = 2;
                c.iw.control.budget.max_states = 4000;
                c.seeds = seeds;
                c.threads = 4;
                c.max_next_layer_states = next;
                const search::ParallelRolloutsResult ref = search::find_rollouts_parallel(*task, c);
                for (usize i = 0; i < seeds.size(); ++i)
                {
                    expect_same(r.rollouts[i].search, ref.rollouts[i].search, "rollout " + std::to_string(i));
                    EXPECT_EQ(r.rollouts[i].reached_fluent_atoms, ref.rollouts[i].reached_fluent_atoms) << "rollout " << i;
                    if (HasFailure())
                        return;
                }
            }
}
}  // namespace

TEST(DeviceIw, WideRowsEqualCpu)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    for (u32 objects : {65u, 80u, 129u, 192u})
    {
        SCOPED_TRACE(objects);
        const auto task = wide_task(objects);
        ASSERT_GT(task->words(), 64u);
        const std::vector<State> starts(3, task->initial_state());
        for (u32 width : {1u, 2u})
        {
            cuda::MultiIwOptions o;
            o.max_width = width;
            o.chunk_states = 2;
            o.max_searches = 2;
            cuda::DeviceMultiIw device(ctx, task, o);
            for (u32 repeat = 0; repeat < 3; ++repeat)
            {
                const auto batch = device.run(starts);
                for (u32 i = 0; i < starts.size(); ++i)
                {
                    const auto c = search::iw(*task, cpu_options(o, starts[i]));
                    expect_same(batch.result(i, *task, starts[i], true), c, "wide rows");
                    if (width == 2)
                    {
                        EXPECT_EQ(c.effective_width, 2u);
                    }
                }
            }
        }
        cuda::DeviceRolloutOptions ro;
        ro.iw.max_width = 2;
        ro.seeds = {0, 1, 7, 12345};
        const auto batch = cuda::rollouts_batch(ctx, task, ro);
        search::ParallelRolloutOptions cpu;
        cpu.iw.max_width = 2;
        cpu.seeds = ro.seeds;
        cpu.threads = 1;
        const auto c = search::find_rollouts_parallel(*task, cpu);
        for (u32 i = 0; i < ro.seeds.size(); ++i)
        {
            expect_same(batch.result(i, *task, starts[0], true), c.rollouts[i].search, "wide rollouts");
            EXPECT_EQ(cuda::reached_atoms(batch, i, *task), c.rollouts[i].reached_fluent_atoms);
        }
    }
}

TEST(DeviceIw, WidePairTableFailsTheSearchesThatReachIt)
{
    SKIP_WITHOUT_GPU();
    const auto task = wide_task(80);
    cuda::MultiIwOptions o;
    o.max_width = 2;
    o.table_bytes = 4096;  // the IW(1) table (800 bytes), not the IW(2) one
    cuda::DeviceMultiIw device(context(), task, o);
    const std::vector<State> starts{task->initial_state()};
    const auto batch = device.run(starts);
    ASSERT_EQ(batch.status[0], search::SearchStatus::Failed);
    const auto passes = batch.passes(0);
    ASSERT_FALSE(passes.empty());
    EXPECT_EQ(passes.back().width, 2u);
    EXPECT_EQ(passes.back().status, search::SearchStatus::Failed);
    for (usize k = 0; k + 1 < passes.size(); ++k)
        EXPECT_EQ(passes[k].status, search::SearchStatus::Exhausted);
    EXPECT_NE(batch.message.find("IW(2)"), std::string::npos);
    EXPECT_NE(batch.message.find("2560400 bytes per search"), std::string::npos);
    EXPECT_NE(batch.message.find("table_bytes is 4096"), std::string::npos);
    EXPECT_EQ(batch.result(0, *task, starts[0], true).message, batch.message);
}

TEST(DeviceIw, LazyGrowthShrinksTableGroups)
{
    SKIP_WITHOUT_GPU();
    const auto task = wide_task(80, TaskOptions::Atoms::Lazy, 4093, true);
    ASSERT_EQ(task->words(), 64u);
    const std::vector<State> starts(2, task->initial_state());
    cuda::MultiIwOptions o;
    o.max_width = 1;
    o.table_bytes = 1024;
    o.chunk_states = 1;
    cuda::DeviceMultiIw device(context(), task, o);
    const auto batch = device.run(starts);
    EXPECT_GT(task->words(), 64u);
    EXPECT_EQ(batch.stats.groups, 2u);
    EXPECT_EQ(batch.stats.regroups, 1u);
    for (u32 i = 0; i < starts.size(); ++i)
        expect_same(batch.result(i, *task, starts[i], true), search::iw(*task, cpu_options(o, starts[i])), "lazy table groups");
    // the counts are those of the groups that ran to the end: a run over the widened rows, which never regroups
    cuda::DeviceMultiIw again(context(), task, o);
    const auto wide = again.run(starts);
    EXPECT_EQ(wide.stats.regroups, 0u);
    EXPECT_EQ(wide.stats.groups, 2u);
    EXPECT_EQ(batch.stats.nodes, wide.stats.nodes);
    EXPECT_EQ(batch.stats.candidates, wide.stats.candidates);
    EXPECT_EQ(batch.stats.layers, wide.stats.layers);
}

TEST(DeviceIw, PairTableRelayoutAcrossChunkBoundary)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const auto stream = ctx->stream();
    const std::vector<std::pair<u32, u32>> pairs{{0, 0}, {4095, 4095}, {1, 4094}, {63, 4095}};
    u32 from = 64;
    std::vector<u64> host(cuda::miw::table_words(2, from), 0);
    for (const auto& [a, b] : pairs)
    {
        host[u64{a} * from + (b >> 6)] |= 1ull << (b & 63);
        host[u64{b} * from + (a >> 6)] |= 1ull << (a & 63);
    }
    cuda::DeviceBuffer source(ctx, host.size() * sizeof(u64), stream);
    cuda::check(cudaMemcpyAsync(source.data(), host.data(), source.size(), cudaMemcpyHostToDevice, stream), "pair table upload");
    cuda::check(cudaStreamSynchronize(stream), "pair table upload sync");
    for (u32 to : {65u, 67u})
    {
        cuda::DeviceBuffer target(ctx, cuda::miw::table_words(2, to) * sizeof(u64), stream);
        cuda::check(cuda::miw::launch_table_relayout(static_cast<const u64*>(source.data()), static_cast<u64*>(target.data()),
                                                    2, 1, from, to, stream), "pair table relayout");
        host.resize(cuda::miw::table_words(2, to));
        cuda::check(cudaMemcpyAsync(host.data(), target.data(), target.size(), cudaMemcpyDeviceToHost, stream), "pair table download");
        cuda::check(cudaStreamSynchronize(stream), "pair table sync");
        std::vector<u64> expected(host.size(), 0);
        for (const auto& [a, b] : pairs)
        {
            const u64 lo = std::min(a, b), hi = std::max(a, b), index = hi * (hi + 1) / 2 + lo;
            expected[index >> 6] |= 1ull << (index & 63);
        }
        EXPECT_EQ(host, expected);
        source = std::move(target);
        from = to;
    }
}

TEST(DeviceIw, HighAtomSlotsKeepSearchesSeparate)
{
    SKIP_WITHOUT_GPU();
    const auto task = wide_task(300);
    ASSERT_GT(task->atoms().fluent_slots(), 65536u);
    const std::vector<State> starts(3, task->initial_state());
    cuda::MultiIwOptions o;
    o.max_width = 1;
    cuda::DeviceMultiIw device(context(), task, o);
    const auto batch = device.run(starts);
    const auto cpu = search::iw(*task, cpu_options(o, starts[0]));
    for (u32 i = 0; i < starts.size(); ++i)
        expect_same(batch.result(i, *task, starts[i], true), cpu, "high atom slots");
}

TEST(DeviceIw, VeryWideRowsRespectByteReservations)
{
    SKIP_WITHOUT_GPU();
    const auto task = wide_task(2048);
    ASSERT_EQ(task->words(), 65536u);
    cuda::ContextOptions co;
    co.max_bytes = u64{256} << 20;
    const auto ctx = cuda::DeviceContext::create(0, co);
    const std::vector<State> starts{task->initial_state()};
    cuda::MultiIwOptions o;
    o.max_width = 1;
    cuda::DeviceMultiIw device(ctx, task, o);
    const auto batch = device.run(starts);
    expect_same(batch.result(0, *task, starts[0], true), search::iw(*task, cpu_options(o, starts[0])), "very wide rows");
}

TEST(DeviceLiw, LandmarkGroupsAndWideRowsEqualCpu)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    for (bool lazy : {false, true})
    {
        const auto task = wide_task(80, lazy ? TaskOptions::Atoms::Lazy : TaskOptions::Atoms::Dense,
                                    lazy ? 4093 : 80, lazy);
        std::vector<CanonicalAtom> facts;
        for (u32 slot = 0; slot < 75; ++slot)
            facts.push_back(task->atoms().canonical(AtomKind::Fluent, slot));
        const auto graph = std::make_shared<const landmarks::FactLandmarkGraph>(
            landmarks::FactLandmarkGraph::create(facts, {{facts[0], facts[1]}, {facts[1], facts[2]}}));
        const std::vector<State> starts(3, task->initial_state());
        for (u32 grouping = 0; grouping < 3; ++grouping)
        {
            cuda::MultiIwOptions o;
            o.landmarks.emplace(3);
            (*o.landmarks)[1].graph = graph;
            (*o.landmarks)[1].disjunctive = true;
            (*o.landmarks)[1].all_private = grouping == 1;
            if (grouping == 2)
                (*o.landmarks)[1].unshared_atoms = {facts[1]};
            (*o.landmarks)[2] = (*o.landmarks)[1];
            o.chunk_states = 1;
            o.max_searches = 2;
            cuda::DeviceMultiIw device(ctx, task, o);
            for (u32 repeat = 0; repeat < 2; ++repeat)
            {
                const auto batch = device.run(starts);
                for (u32 i = 0; i < starts.size(); ++i)
                {
                    search::LiwOptions cpu;
                    cpu.max_width = 1;
                    cpu.start = starts[i];
                    cpu.landmarks = (*o.landmarks)[i];
                    expect_same(batch.result(i, *task, starts[i], true), search::liw(*task, cpu), "landmark groups");
                }
            }
        }
    }
}

TEST(DeviceLiw, SeededRolloutsReplayCapturedChunks)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const auto task = load("blocks__probBLOCKS-8-0", true);
    cuda::MultiIwOptions o;
    o.landmarks.emplace(1);
    const auto root = task->initial_state();
    std::vector<CanonicalAtom> facts;
    bits::for_each(root.data(), root.size_words(), [&](u64 slot)
                   { facts.push_back(task->atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot))); });
    o.landmarks->front().graph = std::make_shared<const landmarks::FactLandmarkGraph>(
        landmarks::FactLandmarkGraph::create(facts));
    o.budget.max_states = 100;
    o.max_next_layer_states = 8;
    cuda::DeviceRollouts device(ctx, task, o);
    const std::vector<u64> seeds{0, 1, 7};
    u64 replays = 0;
    for (u32 repeat = 0; repeat < 6; ++repeat)
    {
        const auto batch = device.run(seeds);
        replays += batch.stats.replays;
        for (u32 i = 0; i < seeds.size(); ++i)
        {
            search::LiwOptions cpu;
            cpu.max_width = 1;
            cpu.landmarks = o.landmarks->front();
            cpu.control.budget = o.budget;
            cpu.layers.kind = search::LayerOrdering::Kind::Randomized;
            cpu.layers.seed = seeds[i];
            cpu.layers.max_next_layer_states = o.max_next_layer_states;
            expect_same(batch.result(i, *task, root, true), search::liw(*task, cpu), "seeded landmark rollout");
        }
    }
    if (cuda::GraphExec::enabled())
    {
        EXPECT_GT(replays, 0u);
    }
}

TEST(DeviceLiw, LandmarkResolutionSizesLazyRowsBeforeSearch)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const auto task = wide_task(80, TaskOptions::Atoms::Lazy, 4093);
    const State root = task->initial_state();
    ASSERT_FALSE(cuda::lifted::wide_row(task->words()));
    cuda::MultiIwOptions o;
    o.landmarks.emplace(1);
    o.landmarks->front().graph = std::make_shared<const landmarks::FactLandmarkGraph>(
        landmarks::FactLandmarkGraph::create({6399, 6398, 6319, 6318}));
    cuda::DeviceMultiIw device(ctx, task, o);
    const std::vector<State> starts(2, root);
    const auto batch = device.run(starts);
    EXPECT_TRUE(cuda::lifted::wide_row(task->words()));
    search::LiwOptions cpu;
    cpu.max_width = 1;
    cpu.start = root;
    cpu.landmarks = o.landmarks->front();
    const auto expected = search::liw(*task, cpu);
    for (u32 i = 0; i < starts.size(); ++i)
        expect_same(batch.result(i, *task, root, true), expected, "landmark slot resolution");
}

TEST(DeviceLiw, RankMaskViewsFollowNodeReservations)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const auto task = load("blocks__probBLOCKS-8-0", true);
    const State root = task->initial_state();
    std::vector<CanonicalAtom> facts;
    bits::for_each(root.data(), root.size_words(), [&](u64 slot)
                   { facts.push_back(task->atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot))); });
    cuda::MultiIwOptions o;
    o.landmarks.emplace(1);
    o.landmarks->front().graph = std::make_shared<const landmarks::FactLandmarkGraph>(
        landmarks::FactLandmarkGraph::create(facts));
    o.budget.max_states = 60;
    o.chunk_states = 256;
    o.graphs = false;
    const std::vector<State> starts(64, root);
    cuda::DeviceMultiIw device(ctx, task, o);
    const auto batch = device.run(starts);
    search::LiwOptions cpu;
    cpu.max_width = 1;
    cpu.landmarks = o.landmarks->front();
    cpu.control.budget = o.budget;
    const auto expected = search::liw(*task, cpu);
    for (u32 i = 0; i < starts.size(); ++i)
        expect_same(batch.result(i, *task, root, true), expected, "node reservations");
}
