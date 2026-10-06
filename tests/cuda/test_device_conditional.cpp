// Device tests for conditional effects and axiom strata.
//   - on every task with conditional effects or axioms of the suite (tests/data/tasks) and of the
//     fork's data (MYMYR_FORK_DATA_DIR, PDDL through the front end), frozen and lazy: no schema with conditional effects
//     is left on the CPU fallback and the axioms run on the device; the device BrFS equals the CPU BrFS (counts, layers,
//     deterministic ids), stop_at_goal returns the CPU's plan, and the device expand is byte-equal to rl::expand (goal
//     flags over derived atoms included); the ids do not depend on the chunk size; the device goal test over derived
//     atoms equals Task::is_goal;
//   - the CPU fallback still works next to the device paths (forward-checking conditions stay on the CPU).
// MYMYR_TEST_FULL_SUITE=1 also runs the tasks above 100,000 states; MYMYR_TEST_SANITIZER=1 shrinks the expand batches.

#include "../cpp/frontend/golden.hpp"
#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <tuple>
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
    o.max_bytes = u64{3} << 30;  // GPU 0 may be shared: cap this process at 4 GB
    return cuda::DeviceContext::create(0, o);
}

bool full_suite()
{
    const char* e = std::getenv("MYMYR_TEST_FULL_SUITE");
    return e && std::string(e) == "1";
}

bool sanitizer_size()
{
    const char* e = std::getenv("MYMYR_TEST_SANITIZER");
    return e && std::string(e) == "1";
}

/// Tasks above this many states run only under MYMYR_TEST_FULL_SUITE=1 (sanitizer runs keep to the smallest ones).
u64 suite_state_limit()
{
    if (full_suite())
        return ~u64{0};
    return sanitizer_size() ? 50'000 : 100'000;
}

/// A task with conditional effects or axioms: an exported suite task (text) or a PDDL task of the fork's data.
struct ConditionalTask
{
    std::string name;     // test name
    std::string text;     // suite task file stem, or empty
    std::string dir;      // fork data directory
    std::string domain, problem;
    u64 states = 0;       // exhaustive BrFS states (for the size limit)
};

const std::vector<ConditionalTask>& conditional_tasks()
{
    static const std::vector<ConditionalTask> t = {
        {"caldera_split_p04", "caldera-split-opt18-adl__p04", "", "", "", 468041},
        {"miconic_simpleadl_s10_2", "miconic-simpleadl__s10-2", "", "", "", 316680},
        {"openstacks_adl_p03", "openstacks-opt08-adl__p03", "", "", "", 97953},
        {"philosophers_p03_phil4", "philosophers__p03-phil4", "", "", "", 47938},
        {"fork_philosophers", "", "philosophers", "domain.pddl", "test_problem.pddl", 0},
        {"fork_relaxed_reachability_axiom", "", "relaxed_reachability_axiom", "domain.pddl", "test_problem.pddl", 0},
        {"fork_airport", "", "airport", "domain.pddl", "test_problem.pddl", 0},
        {"fork_assembly", "", "assembly", "domain.pddl", "test_problem.pddl", 0},
        {"fork_landmark_cond_effect_dedup", "", "landmark_cond_effect_dedup", "domain.pddl", "test_problem.pddl", 0},
        {"fork_landmark_lifted_cond_effect", "", "landmark_lifted_cond_effect", "domain.pddl", "test_problem.pddl", 0},
        {"fork_liw_precheck_conditional", "", "liw_precheck", "conditional_domain.pddl", "conditional_problem.pddl", 0},
        {"fork_miconic_fulladl", "", "miconic-fulladl", "domain.pddl", "test_problem.pddl", 0},
        {"fork_miconic_simpleadl", "", "miconic-simpleadl", "domain.pddl", "test_problem.pddl", 0},
        {"fork_pushworld", "", "pushworld", "domain.pddl", "problem.pddl", 3101889},
        {"fork_relaxed_reachability_chain", "", "relaxed_reachability_chain", "domain.pddl", "test_problem.pddl", 0},
        {"fork_schedule", "", "schedule", "domain.pddl", "test_problem.pddl", 15279078},
    };
    return t;
}

const ConditionalTask& conditional_task(const std::string& name)
{
    for (const ConditionalTask& t : conditional_tasks())
        if (t.name == name)
            return t;
    throw std::logic_error("no task " + name);
}

/// The task, or null when its PDDL is not there (the test skips).
TaskPtr load(const ConditionalTask& g, bool frozen, TaskOptions::Matching matching = TaskOptions::Matching::Auto)
{
    TaskOptions o;
    o.atoms = frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
    o.matching = matching;
    if (!g.text.empty())
        return Task::from_text_file(task_path(g.text), o);
    const std::filesystem::path d = fork_data_dir() / g.dir;
    if (!std::filesystem::exists(d / g.domain) || !std::filesystem::exists(d / g.problem))
        return nullptr;
    return Task::create(*frontend::load_task(d / g.domain, d / g.problem), o);
}

std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const ConditionalTask& t : conditional_tasks())
        for (bool frozen : {true, false})
            out.emplace_back(t.name, frozen);
    return out;
}

std::string param_name(const ::testing::TestParamInfo<std::tuple<std::string, bool>>& info)
{
    return std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
}

template<class T>
std::vector<T> to_host(const void* d, u64 n, cudaStream_t s)
{
    std::vector<T> out(n);
    if (n)
    {
        cuda::check(cudaMemcpyAsync(out.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, s), "D2H");
        cuda::check(cudaStreamSynchronize(s), "sync");
    }
    return out;
}

/// Random-walk states of the task at its current width, as rows.
std::vector<u64> walk_rows(const Task& task, u32 walks, u32 steps, u64 seed, u32& W)
{
    const std::vector<State> walk = device_ref_walks(task, walks, steps, seed);
    W = std::max<u32>(task.words(), 1);
    std::vector<u64> rows(walk.size() * W, 0);
    for (usize i = 0; i < walk.size(); ++i)
        std::copy(walk[i].words().begin(), walk[i].words().end(), rows.begin() + static_cast<std::ptrdiff_t>(i * W));
    return rows;
}

// ------------------------------------------------------------------------------------------------ BrFS

class DeviceConditionalEffectsSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(DeviceConditionalEffectsSuite, NothingOfItOnTheHost)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const auto task = load(conditional_task(name), frozen);
    if (!task)
        GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
    ASSERT_TRUE(task->compiled().has_conditional_effects || task->has_axioms());
    const cuda::ChunkGenerator gen(context(), task);
    for (bool witness : {true, false})
    {
        const cuda::SchemaPlacement& pl = gen.placement(witness);
        EXPECT_EQ(pl.host_ce_count, 0u) << "a schema with conditional effects on the CPU fallback";
        for (u32 s = 0; s < task->num_schemas(); ++s)
            if (pl.host[s])
                ADD_FAILURE() << task->schema_name(SchemaId{s}) << " on the CPU: " << cuda::ChunkGenerator::host_reason(*task, s, witness);
    }
    if (task->has_axioms())
    {
        EXPECT_TRUE(gen.device_axioms()) << gen.host_axioms_reason();
    }
    EXPECT_EQ(gen.needs_host(true), false);
}

TEST_P(DeviceConditionalEffectsSuite, BrfsEqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const ConditionalTask& g = conditional_task(name);
    if (g.states > suite_state_limit())
        GTEST_SKIP() << g.states << " states (MYMYR_TEST_FULL_SUITE=1 runs it)";
    const auto task = load(g, frozen);
    if (!task)
        GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
    for (bool witness : {true, false})
    {
        SCOPED_TRACE(witness ? "witness pruning" : "no witness pruning");
        const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.witness_pruning = witness, .fingerprint = true});
        const BrfsResult c = brfs(*task, {.threads = 2, .witness_pruning = witness, .fingerprint = true});
        EXPECT_EQ(d.result.states, c.states);
        EXPECT_EQ(d.result.expanded, c.expanded);
        EXPECT_EQ(d.result.generated, c.generated);
        EXPECT_EQ(d.result.goal_states, c.goal_states);
        EXPECT_EQ(d.result.layers, c.layers);
        EXPECT_TRUE(d.result.exhausted);
        EXPECT_EQ(d.result.fingerprint, c.fingerprint) << "device ids differ from the CPU's deterministic ids";
        EXPECT_EQ(d.stats.host_schemas, 0u);
        EXPECT_EQ(d.stats.host_ce_schemas, 0u);
        EXPECT_EQ(d.stats.device_axioms, task->has_axioms());
        EXPECT_EQ(d.stats.host_axiom_ms, 0.0);
        if (g.states)
        {
            EXPECT_EQ(d.result.states, g.states);
        }
        RecordProperty("states", static_cast<int>(d.result.states));
    }
}

TEST_P(DeviceConditionalEffectsSuite, StopAtGoalReturnsTheCpuPlan)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const ConditionalTask& g = conditional_task(name);
    if (g.states > suite_state_limit())
        GTEST_SKIP() << g.states << " states (MYMYR_TEST_FULL_SUITE=1 runs it)";
    const auto task = load(g, frozen);
    if (!task)
        GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
    const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.stop_at_goal = true, .fingerprint = true});
    const BrfsResult c = brfs(*task, {.threads = 1, .stop_at_goal = true, .fingerprint = true});
    EXPECT_EQ(d.result.solved, c.solved);
    EXPECT_EQ(d.result.states, c.states);
    EXPECT_EQ(d.result.expanded, c.expanded);
    EXPECT_EQ(d.result.generated, c.generated);
    EXPECT_EQ(d.result.goal_states, c.goal_states);
    EXPECT_EQ(d.result.layers, c.layers);
    EXPECT_EQ(d.result.fingerprint, c.fingerprint);
    EXPECT_TRUE(d.result.plan == c.plan);
}

// ------------------------------------------------------------------------------------------------ expand

/// Host copies of every output array of an expansion.
struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    u64 total = 0;
    u32 words_needed = 0;
    bool operator==(const Flat&) const = default;
};

/// Device buffers of one expansion (all arrays, poisoned) and their host copies.
struct DeviceFlat
{
    cuda::DeviceBuffer succ, parent, schema, binding, goal, offsets;
    rl::Expansion x;
    DeviceFlat(const cuda::ContextPtr& ctx, u64 rows, u64 cap, u32 W, u32 L, cudaStream_t s) :
        succ(ctx, std::max<u64>(cap * W * 8, 8), s), parent(ctx, std::max<u64>(cap * 4, 4), s),
        schema(ctx, std::max<u64>(cap * 4, 4), s), binding(ctx, std::max<u64>(cap * L * 4, 4), s),
        goal(ctx, std::max<u64>(cap, 1), s), offsets(ctx, (rows + 1) * 4, s)
    {
        x.capacity = cap;
        x.words = W;
        x.label_width = L;
        x.succ = static_cast<u64*>(succ.data());
        x.parent = static_cast<i32*>(parent.data());
        x.schema = static_cast<i32*>(schema.data());
        x.binding = static_cast<i32*>(binding.data());
        x.goal = static_cast<u8*>(goal.data());
        x.offsets = static_cast<i32*>(offsets.data());
        for (auto* b : {&this->succ, &this->parent, &this->schema, &this->binding, &this->goal})
            cuda::check(cudaMemsetAsync(b->data(), 0x5A, b->size(), s), "memset");
    }
    Flat host(u64 rows, cudaStream_t s) const
    {
        const u64 cap = x.capacity;
        Flat f;
        f.succ = to_host<u64>(x.succ, cap * x.words, s);
        f.parent = to_host<i32>(x.parent, cap, s);
        f.schema = to_host<i32>(x.schema, cap, s);
        f.binding = to_host<i32>(x.binding, cap * x.label_width, s);
        f.goal = to_host<u8>(x.goal, cap, s);
        f.offsets = to_host<i32>(x.offsets, rows + 1, s);
        f.total = x.total;
        f.words_needed = x.words_needed;
        return f;
    }
};

/// rl::expand into host buffers poisoned like DeviceFlat's.
Flat cpu_expand(const TaskPtr& task, const std::vector<u64>& states, u64 rows, u32 in_words, u64 cap, u32 W, u32 L,
                const rl::ExpandOptions& o)
{
    Flat f;
    f.succ.assign(cap * W, 0x5A5A5A5A5A5A5A5Aull);
    f.parent.assign(cap, 0x5A5A5A5A);
    f.schema.assign(cap, 0x5A5A5A5A);
    f.binding.assign(cap * L, 0x5A5A5A5A);
    f.goal.assign(cap, 0x5A);
    f.offsets.assign(rows + 1, 0);
    rl::Expansion x;
    x.capacity = cap;
    x.words = W;
    x.label_width = L;
    x.succ = f.succ.data();
    x.parent = f.parent.data();
    x.schema = f.schema.data();
    x.binding = f.binding.data();
    x.goal = f.goal.data();
    x.offsets = f.offsets.data();
    rl::expand(*rl::TaskTable::single(task), {states.data(), rows, in_words, 0}, nullptr, x, o);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

TEST_P(DeviceConditionalEffectsSuite, ExpandByteEqualToRlExpand)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    for (bool device_first : {true, false})
    {
        SCOPED_TRACE(device_first ? "device first" : "cpu first");
        const auto task = load(conditional_task(name), frozen);
        if (!task)
            GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
        u32 Win = 0;
        const std::vector<u64> states = sanitizer_size() ? walk_rows(*task, 2, 8, 17, Win) : walk_rows(*task, 4, 25, 17, Win);
        const u64 N = states.size() / Win;
        auto ctx = context();
        const cudaStream_t s = ctx->stream();
        cuda::DeviceBuffer d_states(ctx, states.size() * 8, s);
        cuda::check(cudaMemcpyAsync(d_states.data(), states.data(), states.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
        cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task));
        const u32 L = rl::max_label_width(*task) + 1;
        for (bool canonical : {true, false})
            for (bool witness : {false, true})
            {
                SCOPED_TRACE(std::string(canonical ? "canonical" : "matcher order") + (witness ? ", witness" : ""));
                const rl::ExpandOptions o{canonical, witness, true};
                if (!device_first)
                    (void)cpu_expand(task, states, N, Win, 1, Win, L, o);  // interns first under lazy slots
                const u64 total = ex.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o);
                const u32 Wfull = std::max(Win, task->max_words());
                for (auto [cap, W] : {std::pair<u64, u32>{total + 3, Wfull}, {total / 2 + 1, Wfull}, {std::max<u64>(total, 1), 1u}})
                {
                    SCOPED_TRACE("capacity " + std::to_string(cap) + ", words " + std::to_string(W));
                    DeviceFlat df(ctx, N, cap, W, L, s);
                    ex.write(df.x);
                    const Flat dev = df.host(N, s);
                    const Flat cpu = cpu_expand(task, states, N, Win, cap, W, L, o);
                    ASSERT_EQ(dev.total, cpu.total);
                    ASSERT_EQ(dev.words_needed, cpu.words_needed);
                    ASSERT_EQ(dev.offsets, cpu.offsets);
                    ASSERT_EQ(dev.parent, cpu.parent);
                    ASSERT_EQ(dev.schema, cpu.schema);
                    ASSERT_EQ(dev.binding, cpu.binding);
                    ASSERT_EQ(dev.succ, cpu.succ);
                    ASSERT_EQ(dev.goal, cpu.goal);
                }
                // chunks of 7 parents: the same bytes
                cuda::DeviceExpander ex7(ctx, rl::TaskTable::single(task));
                ex7.set_chunk_rows(7);
                ASSERT_EQ(ex7.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o), total);
                DeviceFlat df(ctx, N, total + 1, Wfull, L, s);
                ex7.write(df.x);
                ASSERT_EQ(df.host(N, s), cpu_expand(task, states, N, Win, total + 1, Wfull, L, o));
            }
    }
}

INSTANTIATE_TEST_SUITE_P(DeviceConditionalEffects, DeviceConditionalEffectsSuite, ::testing::ValuesIn(params()), param_name);

// ------------------------------------------------------------------------------------------------ details

TEST(DeviceConditionalEffects, ChunkSizeDoesNotChangeTheIds)
{
    SKIP_WITHOUT_GPU();
    // axioms (non-recursive and recursive strata, derived goals) and conditional effects (forall), in chunks down to 7
    for (const char* name : {"philosophers_p03_phil4", "openstacks_adl_p03", "fork_miconic_fulladl", "fork_relaxed_reachability_axiom",
                             "fork_miconic_simpleadl", "fork_airport"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            if (conditional_task(name).states > suite_state_limit())
                continue;
            u64 ref_fp = 0, ref_states = 0;
            for (u32 chunk : {u32{1} << 20, 1000u, conditional_task(name).states > 5000 ? 97u : 7u})
            {
                const auto task = load(conditional_task(name), frozen);
                if (!task)
                    GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
                if (sanitizer_size() && chunk < 100u)
                    continue;
                const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.fingerprint = true, .chunk_states = chunk});
                if (chunk == u32{1} << 20)
                {
                    ref_fp = d.result.fingerprint;
                    ref_states = d.result.states;
                    continue;
                }
                EXPECT_EQ(d.result.states, ref_states) << "chunk " << chunk;
                EXPECT_EQ(d.result.fingerprint, ref_fp) << "chunk " << chunk;
            }
        }
}

TEST(DeviceConditionalEffects, DerivedAtomsFirstMetOnTheDevice)
{
    SKIP_WITHOUT_GPU();
    // fresh lazy tasks: the device interns the derived atoms it derives (canonical-id order) and reruns the chunk
    for (const char* name : {"philosophers_p03_phil4", "fork_miconic_fulladl", "fork_relaxed_reachability_axiom"})
    {
        SCOPED_TRACE(name);
        const auto task = load(conditional_task(name), false);
        if (!task)
            GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
        const u32 derived0 = task->atoms().derived_slots();
        const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.fingerprint = true});
        EXPECT_GT(task->atoms().derived_slots(), derived0);
        EXPECT_GT(d.stats.axiom_reruns, 0u);
        const BrfsResult c = brfs(*task, {.threads = 2, .fingerprint = true});
        EXPECT_EQ(d.result.states, c.states);
        EXPECT_EQ(d.result.goal_states, c.goal_states);
        EXPECT_EQ(d.result.fingerprint, c.fingerprint);
    }
}

TEST(DeviceConditionalEffects, GoalFlagsEqualTaskIsGoal)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"philosophers_p03_phil4", "fork_miconic_fulladl", "fork_relaxed_reachability_axiom", "fork_philosophers"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            const auto task = load(conditional_task(name), frozen);
            if (!task)
                GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
            ASSERT_TRUE(task->compiled().goal.uses_derived);
            u32 W = 0;
            const std::vector<u64> rows = walk_rows(*task, 6, 30, 5, W);
            const u64 N = rows.size() / W;
            auto ctx = context();
            const cudaStream_t s = ctx->stream();
            cuda::ChunkGenerator gen(ctx, task, s);
            // rows 2 words apart from a strided buffer, read through an order with a hole (0xFFFFFFFF: a zero row)
            const u32 stride = W + 2;
            std::vector<u64> strided(N * stride, 0);
            for (u64 i = 0; i < N; ++i)
                std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(i * W), W, strided.begin() + static_cast<std::ptrdiff_t>(i * stride));
            std::vector<u32> order(N + 1);
            for (u64 i = 0; i < N; ++i)
                order[i] = static_cast<u32>(N - 1 - i);
            order[N] = 0xFFFFFFFFu;
            cuda::DeviceBuffer d_rows(ctx, strided.size() * 8, s), d_order(ctx, order.size() * 4, s), d_out(ctx, N + 1, s);
            cuda::check(cudaMemcpyAsync(d_rows.data(), strided.data(), strided.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
            cuda::check(cudaMemcpyAsync(d_order.data(), order.data(), order.size() * 4, cudaMemcpyHostToDevice, s), "H2D");
            gen.goal_flags(static_cast<const u64*>(d_rows.data()), stride, W, N, static_cast<const u32*>(d_order.data()), N + 1,
                           static_cast<u8*>(d_out.data()));
            const std::vector<u8> got = to_host<u8>(d_out.data(), N + 1, s);
            u64 goals = 0;
            for (u64 i = 0; i < N; ++i)
            {
                const u64* r = rows.data() + (N - 1 - i) * W;
                const bool want = task->is_goal(StateView{r, bits::trimmed_size(r, W), nullptr, 0});
                goals += want ? 1 : 0;
                EXPECT_EQ(got[i] != 0, want) << "row " << i;
            }
            const std::vector<u64> zero(W, 0);
            EXPECT_EQ(got[N] != 0, task->is_goal(StateView{zero.data(), 0, nullptr, 0}));
            RecordProperty("goals", static_cast<int>(goals));
        }
}

TEST(DeviceConditionalEffects, GeneratorOutlivesItsCallersStreams)
{
    // DeviceBuffer lifetime: each call on a caller's stream that is destroyed after the call (mark_last_use at
    // the end of each): the scratch is reallocated on the next stream and freed on live streams only
    SKIP_WITHOUT_GPU();
    const auto task = load(conditional_task("philosophers_p03_phil4"), false);
    if (!task)
        GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
    u32 W = 0;
    const std::vector<u64> rows = walk_rows(*task, 6, 30, 5, W);
    const u64 N = rows.size() / W;
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    std::vector<u32> order(N);
    std::iota(order.begin(), order.end(), 0u);
    cuda::DeviceBuffer d_rows(ctx, rows.size() * 8, s), d_order(ctx, order.size() * 4, s);
    cuda::check(cudaMemcpyAsync(d_rows.data(), rows.data(), rows.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaMemcpyAsync(d_order.data(), order.data(), order.size() * 4, cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
    cuda::ChunkGenerator gen(ctx, task, s);
    std::vector<u8> first;
    for (u32 round = 0; round < 4; ++round)
    {
        cuda::Stream side;
        gen.set_stream(side);
        // a growing row count: the scratch is reallocated on this round's stream
        const u64 n = std::max<u64>(1, N * (round + 1) / 4);
        cuda::DeviceBuffer d_out(ctx, n, side);
        gen.goal_flags(static_cast<const u64*>(d_rows.data()), W, W, N, static_cast<const u32*>(d_order.data()), n,
                       static_cast<u8*>(d_out.data()));
        const std::vector<u8> got = to_host<u8>(d_out.data(), n, side);
        for (u64 i = 0; i < n; ++i)
        {
            const u64* r = rows.data() + i * W;
            ASSERT_EQ(got[i] != 0, task->is_goal(StateView{r, bits::trimmed_size(r, W), nullptr, 0})) << "round " << round << " row " << i;
        }
        gen.mark_last_use();
        d_out.reset(s);
    }  // each side stream is destroyed with the generator's buffers still allocated on it
    cuda::check(cudaDeviceSynchronize(), "sync");
}

TEST(DeviceConditionalEffects, ForcedForwardCheckingEqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    // forward checking in every plan (the CPU's matchers): the device takes its own matcher wherever the order of the
    // bindings does not matter (rl::dev::device_fc: counts, rows in canonical order, axiom bodies, the conditions of
    // conditional effects, so nothing stays on the CPU) and forward checks the schemas whose rows keep the matcher's
    // order; both equal the CPU
    for (const char* name : {"fork_miconic_fulladl", "fork_assembly", "fork_airport", "philosophers_p03_phil4"})
        for (bool frozen : {true, false})
            for (bool canonical : {true, false})
            {
                SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy") + (canonical ? " canonical" : " matcher order"));
                const auto task = load(conditional_task(name), frozen, TaskOptions::Matching::ForwardChecking);
                if (!task)
                    GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
                const cuda::DeviceBrfsResult d =
                    cuda::brfs(context(), task, {.canonical_order = canonical, .stop_at_goal = false, .fingerprint = true});
                const BrfsResult c = brfs(*task, {.threads = 2, .canonical_order = canonical, .fingerprint = true});
                EXPECT_EQ(d.result.states, c.states);
                EXPECT_EQ(d.result.generated, c.generated);
                EXPECT_EQ(d.result.goal_states, c.goal_states);
                EXPECT_EQ(d.result.fingerprint, c.fingerprint);
                EXPECT_EQ(d.stats.host_schemas, 0u);
            }
}

TEST(DeviceConditionalEffects, MultiIwEqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    // IW(1) and IW(2) from random-walk states (the device IW search): conditional effects, axioms and derived goals on the device
    for (const char* name : {"fork_miconic_fulladl", "fork_airport", "fork_assembly", "fork_schedule", "fork_relaxed_reachability_axiom",
                             "fork_philosophers"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            const auto task = load(conditional_task(name), frozen);
            if (!task)
                GTEST_SKIP() << "PDDL not found under " << fork_data_dir();
            const std::vector<State> starts = device_ref_walks(*task, sanitizer_size() ? 1 : 3, 8, 23);
            for (u32 arity : {1u, 2u})
            {
                SCOPED_TRACE("IW(" + std::to_string(arity) + ")");
                cuda::MultiIwOptions o;
                o.max_arity = arity;
                o.budget.max_states = 20000;
                const std::vector<search::IwResult> d = cuda::multi_iw(context(), task, starts, o);
                ASSERT_EQ(d.size(), starts.size());
                for (usize i = 0; i < starts.size(); ++i)
                {
                    SCOPED_TRACE("start " + std::to_string(i));
                    search::IwOptions co;
                    co.max_arity = o.max_arity;
                    co.optimize_iw1 = o.optimize_iw1;
                    co.width_zero = o.width_zero;
                    co.witness_pruning = o.witness_pruning;
                    co.canonical_order = o.canonical_order;
                    co.control.budget = o.budget;
                    co.start = starts[i];
                    const search::IwResult c = search::iw(*task, co);
                    EXPECT_EQ(d[i].status, c.status);
                    EXPECT_EQ(d[i].effective_width, c.effective_width);
                    EXPECT_EQ(d[i].total.expanded, c.total.expanded);
                    EXPECT_EQ(d[i].total.generated, c.total.generated);
                    EXPECT_TRUE(d[i].plan == c.plan) << d[i].plan.size() << " vs " << c.plan.size() << " steps";
                    ASSERT_EQ(d[i].goal_state.has_value(), c.goal_state.has_value());
                    if (c.goal_state)
                    {
                        EXPECT_TRUE(d[i].goal_state->view() == c.goal_state->view());
                    }
                    if (HasFailure())
                        return;
                }
            }
        }
}
}  // namespace
