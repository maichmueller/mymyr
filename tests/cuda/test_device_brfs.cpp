// Device BrFS tests (run on GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the device BrFS reproduces the suite's state, generated, goal and layer counts, frozen and lazy, and the
//     CPU BrFS's deterministic ids (fingerprint); the ids do not depend on the chunk size or on a candidate budget
//     that cuts the device loops' chunks; stop_at_goal returns the CPU's plan; a budget stops at a prefix of the
//     id order;
//   - the device expand is byte-equal to rl::expand (flat and padded; canonical order on and off, witness
//     pruning on and off, clipped capacities, narrow rows, one chunk and chunks of 7 parents) on random-walk states of
//     every suite task, frozen and lazy; launches of at most lifted::k_warp_rows rows search by warps, larger
//     ones a lane per state: both write rl::expand's bytes.
// The full suite (every task in the BrFS test) runs with MYMYR_TEST_FULL_SUITE=1.

#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/search/brfs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
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
    o.max_bytes = u64{6} << 30;  // well below the 16 GB this machine allows
    return cuda::DeviceContext::create(0, o);
}

TaskPtr load(const std::string& name, bool frozen)
{
    TaskOptions o;
    o.atoms = frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
    return Task::from_text_file(task_path(name), o);
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

const SuiteTask& suite_task(const std::string& name)
{
    for (const SuiteTask& t : suite())
        if (t.name == name)
            return t;
    throw std::logic_error("no suite task " + name);
}

/// All state rows of a device BrFS, W words each.
std::vector<u64> device_rows(const cuda::DeviceBrfs& b)
{
    const cuda::DeviceArena& a = b.states();
    return to_host<u64>(a.device_data(), a.device_size() * b.words(), a.stream());
}

// ------------------------------------------------------------------------------------------------ BrFS (gate 1)

class CudaBrfsSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(CudaBrfsSuite, CountsAndIdsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const SuiteTask& st = suite_task(name);
    if (st.states > suite_state_limit())
        GTEST_SKIP() << st.states << " states (MYMYR_TEST_FULL_SUITE=1 runs it)";
    const auto task = load(name, frozen);
    const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.fingerprint = true});
    const BrfsResult& r = d.result;
    EXPECT_EQ(r.states, st.states);
    EXPECT_EQ(r.generated, st.generated);
    EXPECT_EQ(r.goal_states, st.goal_states);
    EXPECT_EQ(r.expanded, st.states);
    EXPECT_TRUE(r.exhausted);
    const BrfsResult c = brfs(*task, {.threads = 2, .fingerprint = true});
    EXPECT_EQ(r.layers, c.layers);
    EXPECT_EQ(r.fingerprint, c.fingerprint) << "device ids differ from the CPU's deterministic ids";
    RecordProperty("host_schemas", static_cast<int>(d.stats.host_schemas));
    RecordProperty("widenings", static_cast<int>(d.stats.widenings));
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaBrfsSuite, ::testing::ValuesIn(params()), param_name);

TEST(CudaBrfs, ChunkSizeDoesNotChangeTheIds)
{
    SKIP_WITHOUT_GPU();
    // fixed-order matchers with sorting, axioms and forward checking, CPU-fallback schemas and wide states
    // (MYMYR_TEST_SANITIZER=1: organic-synthesis without the 7-state chunks, ~10 s of deep matchers a run on their own)
    const bool small = std::getenv("MYMYR_TEST_SANITIZER") != nullptr;
    for (const char* name : {"depot__p02", "philosophers__p03-phil4", "organic-synthesis-opt18-strips__p20"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            std::vector<u64> ref;
            u64 ref_fp = 0;
            for (u32 chunk : {u32{1} << 20, 1000u, 7u})
            {
                if (small && chunk == 7 && std::strcmp(name, "organic-synthesis-opt18-strips__p20") == 0)
                    continue;
                const auto task = load(name, frozen);
                cuda::DeviceBrfs b(context(), task, {.fingerprint = true, .chunk_states = chunk});
                const cuda::DeviceBrfsResult d = b.run();
                std::vector<u64> rows = device_rows(b);
                // compare canonically: the row width follows the lazy slot numbering of each run
                if (chunk == u32{1} << 20)
                {
                    ref = std::move(rows);
                    ref_fp = d.result.fingerprint;
                    EXPECT_EQ(d.result.states, suite_task(name).states);
                    continue;
                }
                EXPECT_EQ(d.result.fingerprint, ref_fp) << "chunk " << chunk;
                if (frozen)
                {
                    EXPECT_EQ(rows, ref) << "chunk " << chunk;
                }
            }
        }
}

// Chunks sized on the device in groups; past their capacity (gripper's candidates per parent alternate between
// layers), past the table's limit (a small expected size), redone from their rows or their views: the same ids
TEST(CudaBrfs, ChunkGroupsRedoneOnTheDeviceKeepTheIds)
{
    SKIP_WITHOUT_GPU();
    // (MYMYR_TEST_SANITIZER=1: gripper alone)
    const bool small = std::getenv("MYMYR_TEST_SANITIZER") != nullptr;
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0"})
        for (bool frozen : {true, false})
        {
            if (small && std::string(name) != "gripper__prob05")
                continue;
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            std::vector<u64> ref;
            u64 ref_fp = 0;
            u64 resumed = 0, redone = 0;
            u32 rehashes = 0;
            for (const cuda::DeviceBrfsOptions& o : {cuda::DeviceBrfsOptions{.fingerprint = true},
                                                     cuda::DeviceBrfsOptions{.fingerprint = true, .chunk_states = 5000},
                                                     cuda::DeviceBrfsOptions{.fingerprint = true, .chunk_states = 5000, .expected_states = 1}})
            {
                const auto task = load(name, frozen);
                cuda::DeviceBrfs b(context(), task, o);
                const cuda::DeviceBrfsResult d = b.run();
                std::vector<u64> rows = device_rows(b);
                const cuda::DeviceBrfsStats& st = d.stats;
                SCOPED_TRACE("chunk " + std::to_string(o.chunk_states) + " expected " + std::to_string(o.expected_states));
                EXPECT_EQ(d.result.states, suite_task(name).states);
                EXPECT_LE(st.groups, st.chunks + st.resumed + st.redone);
                resumed += st.resumed;
                redone += st.redone;
                rehashes += st.rehashes;
                if (ref.empty())
                {
                    ref = std::move(rows);
                    ref_fp = d.result.fingerprint;
                    continue;
                }
                EXPECT_EQ(d.result.fingerprint, ref_fp);
                if (frozen)
                {
                    EXPECT_EQ(rows, ref);
                }
            }
            EXPECT_GT(rehashes, 0u);
            if (std::string(name) == "gripper__prob05")
            {
                EXPECT_GT(resumed, 0u);
                if (frozen)  // (lazy slots: groups of one chunk)
                {
                    EXPECT_GT(redone, 0u);
                }
            }
        }
}

// A candidate budget below a chunk's candidates (DeviceBrfsOptions::candidate_bytes): the device loops cut their
// chunks to the parents whose candidates fit, the host's chunks halt and are redone with their candidates known: the
// same ids, with the goal states of the cut chunks counted once
TEST(CudaBrfs, CandidateBudgetKeepsTheIds)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            std::vector<u64> ref;
            u64 ref_fp = 0, cuts = 0, loops = 0;
            for (const u64 bytes : {u64{32} << 20, u64{1}, u64{20000}})
            {
                SCOPED_TRACE("candidate bytes " + std::to_string(bytes));
                const auto task = load(name, frozen);
                cuda::DeviceBrfs b(context(), task, {.fingerprint = true, .candidate_bytes = bytes});
                const cuda::DeviceBrfsResult d = b.run();
                std::vector<u64> rows = device_rows(b);
                EXPECT_EQ(d.result.states, suite_task(name).states);
                EXPECT_EQ(d.result.goal_states, suite_task(name).goal_states);
                if (ref.empty())
                {
                    ref = std::move(rows);
                    ref_fp = d.result.fingerprint;
                    continue;
                }
                cuts += d.stats.cuts;
                loops += d.stats.loops;
                EXPECT_EQ(d.result.fingerprint, ref_fp);
                if (frozen)
                {
                    EXPECT_EQ(rows, ref);
                }
            }
            if (loops > 0)
            {
                EXPECT_GT(cuts, 0u);
            }
        }
}

TEST(CudaBrfs, StopAtGoalReturnsTheCpuPlan)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"depot__p02", "philosophers__p03-phil4", "caldera-split-opt18-adl__p04", "rovers__p02",
                             "organic-synthesis-opt18-strips__p20", "parcprinter-opt11-strips__p03"})
        for (bool frozen : {true, false})
        {
            SCOPED_TRACE(std::string(name) + (frozen ? " frozen" : " lazy"));
            const auto task = load(name, frozen);
            const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.stop_at_goal = true, .fingerprint = true});
            const BrfsResult c = brfs(*task, {.threads = 1, .stop_at_goal = true, .fingerprint = true});
            ASSERT_TRUE(d.result.solved);
            EXPECT_EQ(d.result.states, c.states);
            EXPECT_EQ(d.result.expanded, c.expanded);
            EXPECT_EQ(d.result.generated, c.generated);
            EXPECT_EQ(d.result.goal_states, c.goal_states);
            EXPECT_EQ(d.result.layers, c.layers);
            EXPECT_EQ(d.result.fingerprint, c.fingerprint);
            EXPECT_EQ(d.result.plan.size(), c.plan.size());
            EXPECT_TRUE(d.result.plan == c.plan);
        }
}

TEST(CudaBrfs, BudgetStopsAtAPrefixOfTheIds)
{
    SKIP_WITHOUT_GPU();
    const auto full_task = load("depot__p02", true);
    cuda::DeviceBrfs full(context(), full_task);
    (void)full.run();
    const std::vector<u64> all = device_rows(full);
    const auto task = load("depot__p02", true);
    cuda::DeviceBrfs b(context(), task, {.max_states = 10000, .chunk_states = 256});
    const cuda::DeviceBrfsResult d = b.run();
    EXPECT_GE(d.result.states, 10000u);
    EXPECT_LT(d.result.states, suite_task("depot__p02").states);
    EXPECT_FALSE(d.result.exhausted);
    const std::vector<u64> rows = device_rows(b);
    ASSERT_EQ(b.words(), full.words());
    ASSERT_LE(rows.size(), all.size());
    EXPECT_TRUE(std::equal(rows.begin(), rows.end(), all.begin()));
}

TEST(CudaBrfs, RejectsUnsupportedOptions)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("depot__p02", true);
    EXPECT_THROW(cuda::DeviceBrfs(context(), task, {.chunk_states = 0}), std::invalid_argument);
    cuda::DeviceBrfs b(context(), task);
    (void)b.run();
    EXPECT_THROW((void)b.run(), std::logic_error);
    EXPECT_THROW((void)b.plan_to(1u << 30), std::out_of_range);
}

// ------------------------------------------------------------------------------------------------ expand (gate 2)

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

/// Device buffers of one expansion (all arrays) and their host copies.
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
        // poison: rows the expansion must not write stay recognizable
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

/// rl::expand into host buffers poisoned like DeviceFlat's (without `binding`: no binding labels, the buffer stays
/// poisoned).
Flat cpu_expand(const TaskPtr& task, const std::vector<u64>& states, u64 rows, u32 in_words, u64 cap, u32 W, u32 L,
                const rl::ExpandOptions& o, bool binding = true)
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
    x.binding = binding ? f.binding.data() : nullptr;
    x.goal = f.goal.data();
    x.offsets = f.offsets.data();
    rl::expand(*rl::TaskTable::single(task), {states.data(), rows, in_words, 0}, nullptr, x, o);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

struct Padded
{
    std::vector<i32> index, count, schema, binding;
    std::vector<u8> mask, goal;
    std::vector<u64> succ;
    bool overflow = false;
    bool operator==(const Padded&) const = default;
};

class CudaExpandSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(CudaExpandSuite, ByteEqualToRlExpand)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    for (bool device_first : {true, false})
    {
        SCOPED_TRACE(device_first ? "device first" : "cpu first");
        const auto task = load(name, frozen);
        // a batch of random-walk states at the task's current width
        const std::vector<State> walk = device_ref_walks(*task, 4, 25, 17);
        const u32 Win = std::max<u32>(task->words(), 1);
        std::vector<u64> states(walk.size() * Win, 0);
        for (usize i = 0; i < walk.size(); ++i)
            std::copy(walk[i].words().begin(), walk[i].words().end(), states.begin() + static_cast<std::ptrdiff_t>(i * Win));
        const u64 N = walk.size();
        auto ctx = context();
        const cudaStream_t s = ctx->stream();
        cuda::DeviceBuffer d_states(ctx, states.size() * 8, s);
        cuda::check(cudaMemcpyAsync(d_states.data(), states.data(), states.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
        cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task));
        const u32 L = rl::max_label_width(*task) + 1;  // one column of -1 padding more than needed
        for (bool canonical : {true, false})
            for (bool witness : {false, true})
            {
                SCOPED_TRACE(std::string(canonical ? "canonical" : "matcher order") + (witness ? ", witness" : ""));
                const rl::ExpandOptions o{canonical, witness, true};
                Flat cpu_full;
                if (!device_first)
                    cpu_full = cpu_expand(task, states, N, Win, 1, Win, L, o);  // interns first under lazy slots
                const u64 total = ex.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o);
                ASSERT_GT(total, 0u);
                // full capacity at the task's width, a clipped capacity, and narrow rows (zeroed, words_needed)
                const u32 Wfull = std::max(Win, task->max_words());
                for (auto [cap, W] : {std::pair<u64, u32>{total + 3, Wfull}, {total / 2 + 1, Wfull}, {total, 1u}})
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
                    // padded: K at the largest count, and a smaller K (overflow)
                    for (u32 K : {0u, 2u})
                    {
                        u32 k = K;
                        if (k == 0)
                            for (u64 i = 0; i < N; ++i)
                                k = std::max<u32>(k, static_cast<u32>(cpu.offsets[i + 1] - cpu.offsets[i]));
                        k = std::max(k, 1u);
                        const u32 PW = W + 1, PL = L + 1;
                        cuda::DeviceBuffer pi(ctx, N * k * 4, s), pm(ctx, N * k, s), pc(ctx, N * 4, s),
                            ps(ctx, N * k * PW * 8, s), psc(ctx, N * k * 4, s), pb(ctx, N * k * PL * 4, s), pg(ctx, N * k, s);
                        rl::PaddedExpansion dp{k, static_cast<i32*>(pi.data()), static_cast<u8*>(pm.data()), static_cast<i32*>(pc.data()),
                                               PW, static_cast<u64*>(ps.data()), static_cast<i32*>(psc.data()), PL,
                                               static_cast<i32*>(pb.data()), static_cast<u8*>(pg.data()), false};
                        ex.pad(df.x, N, dp);
                        Padded pd{to_host<i32>(pi.data(), N * k, s), to_host<i32>(pc.data(), N, s), to_host<i32>(psc.data(), N * k, s),
                                  to_host<i32>(pb.data(), N * k * PL, s), to_host<u8>(pm.data(), N * k, s),
                                  to_host<u8>(pg.data(), N * k, s), to_host<u64>(ps.data(), N * k * PW, s), dp.overflow};
                        Padded pc_{std::vector<i32>(N * k), std::vector<i32>(N), std::vector<i32>(N * k), std::vector<i32>(N * k * PL),
                                   std::vector<u8>(N * k), std::vector<u8>(N * k), std::vector<u64>(N * k * PW), false};
                        rl::Expansion hx;
                        hx.capacity = cap;
                        hx.words = W;
                        hx.label_width = L;
                        hx.succ = const_cast<u64*>(cpu.succ.data());
                        hx.schema = const_cast<i32*>(cpu.schema.data());
                        hx.binding = const_cast<i32*>(cpu.binding.data());
                        hx.goal = const_cast<u8*>(cpu.goal.data());
                        hx.offsets = const_cast<i32*>(cpu.offsets.data());
                        hx.total = cpu.total;
                        rl::PaddedExpansion hp{k, pc_.index.data(), pc_.mask.data(), pc_.count.data(), PW, pc_.succ.data(),
                                               pc_.schema.data(), PL, pc_.binding.data(), pc_.goal.data(), false};
                        rl::pad(hx, N, hp);
                        pc_.overflow = hp.overflow;
                        ASSERT_EQ(pd, pc_) << "K " << k;
                    }
                }
                // chunks of 7 parents (the path of batches beyond the view budget): the same bytes
                cuda::DeviceExpander ex7(ctx, rl::TaskTable::single(task));
                ex7.set_chunk_rows(7);
                ASSERT_EQ(ex7.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o), total);
                for (auto [cap, W] : {std::pair<u64, u32>{total + 3, Wfull}, {total / 2 + 1, Wfull}, {total, 1u}})
                {
                    SCOPED_TRACE("chunks of 7, capacity " + std::to_string(cap) + ", words " + std::to_string(W));
                    DeviceFlat df(ctx, N, cap, W, L, s);
                    ex7.write(df.x);
                    ASSERT_EQ(df.host(N, s), cpu_expand(task, states, N, Win, cap, W, L, o));
                }
                if (!device_first)
                {
                    ASSERT_EQ(cpu_full.total, total);
                }
            }
    }
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaExpandSuite, ::testing::ValuesIn(params()), param_name);

TEST(CudaExpand, WarpAndLaneMatchersWriteTheSameRows)
{
    SKIP_WITHOUT_GPU();
    // A launch of at most k_warp_rows rows searches each segment by a warp (the lanes split it), a larger one by a
    // lane per state: the walk states once, then repeated past k_warp_rows rows, both against rl::expand. The tasks:
    // fixed order with sorted segments (depot), forward checking with axioms (philosophers), rank bitmaps (blocks,
    // gripper), long segments sorted in the label rows (miconic-simpleadl's conditional effects, pathways)
    for (const char* name : {"depot__p02", "philosophers__p03-phil4", "blocks__probBLOCKS-8-0", "gripper__prob05", "pathways__p02",
                             "miconic-simpleadl__s10-2"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        const std::vector<State> walk = device_ref_walks(*task, 4, 25, 29);
        const u32 Win = std::max<u32>(task->words(), 1);
        const u64 n_walk = walk.size();
        auto ctx = context();
        const cudaStream_t s = ctx->stream();
        for (const u64 N : {n_walk, u64{cuda::lifted::k_warp_rows} + 37})
        {
            SCOPED_TRACE(std::to_string(N) + " rows");
            std::vector<u64> states(N * Win, 0);
            for (u64 i = 0; i < N; ++i)
                std::copy(walk[i % n_walk].words().begin(), walk[i % n_walk].words().end(),
                          states.begin() + static_cast<std::ptrdiff_t>(i * Win));
            cuda::DeviceBuffer d_states(ctx, states.size() * 8, s);
            cuda::check(cudaMemcpyAsync(d_states.data(), states.data(), states.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
            cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task));
            const u32 L = rl::max_label_width(*task) + 1;
            for (bool canonical : {true, false})
                for (bool witness : {false, true})
                {
                    SCOPED_TRACE(std::string(canonical ? "canonical" : "matcher order") + (witness ? ", witness" : ""));
                    const rl::ExpandOptions o{canonical, witness, true};
                    const u64 total = ex.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o);
                    ASSERT_GT(total, 0u);
                    const u32 Wfull = std::max(Win, task->max_words());
                    // full capacity, and a clipped one (rows past it only count their widths); with binding labels,
                    // and without (segments sorted in the expander's scratch, or written as found)
                    for (const u64 cap : {total, total / 3 + 1})
                        for (const bool binding : {true, false})
                        {
                            SCOPED_TRACE("capacity " + std::to_string(cap) + (binding ? "" : ", no binding labels"));
                            DeviceFlat df(ctx, N, cap, Wfull, L, s);
                            if (!binding)
                                df.x.binding = nullptr;
                            ex.write(df.x);
                            df.x.binding = static_cast<i32*>(df.binding.data());  // (poisoned: compared as such)
                            ASSERT_EQ(df.host(N, s), cpu_expand(task, states, N, Win, cap, Wfull, L, o, binding));
                        }
                }
        }
    }
}

TEST(CudaExpand, ValidationAndDestinationChecksMatchRlExpand)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("gripper__prob05", true);
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task));
    const u32 W = task->words() + 1;  // a word past the assigned slots
    std::vector<u64> rows(2 * W, 0);
    const State s0 = task->initial_state();
    std::copy(s0.words().begin(), s0.words().end(), rows.begin());
    std::copy(s0.words().begin(), s0.words().end(), rows.begin() + W);
    rows[2 * W - 1] = 1;  // row 1 sets an unassigned slot
    cuda::DeviceBuffer d(ctx, rows.size() * 8, s);
    cuda::check(cudaMemcpyAsync(d.data(), rows.data(), rows.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
    const rl::StateBatchView in{static_cast<const u64*>(d.data()), 2, W, 0};
    try
    {
        (void)ex.count(in, nullptr);
        FAIL() << "expected std::invalid_argument";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("state row 1"), std::string::npos) << e.what();
    }
    EXPECT_GT(ex.count(in, nullptr, {true, false, false}), 0u);  // validate off
    rl::Expansion x;
    x.capacity = 4;
    x.words = W;
    x.label_width = 1;  // below the largest arity
    cuda::DeviceBuffer b(ctx, 64, s);
    x.binding = static_cast<i32*>(b.data());
    EXPECT_THROW(ex.write(x), std::invalid_argument);
    x.binding = nullptr;
    x.goal = static_cast<u8*>(b.data());
    EXPECT_THROW(ex.write(x), std::invalid_argument);  // goal flags need the successor rows
    EXPECT_THROW(cuda::DeviceExpander(ctx, rl::TaskTable::single(task)).write(x), std::logic_error);  // count() first
    // an empty batch: offsets [0]
    cuda::DeviceBuffer off(ctx, 4, s);
    cuda::check(cudaMemsetAsync(off.data(), 0x5A, 4, s), "memset");
    rl::Expansion e;
    e.offsets = static_cast<i32*>(off.data());
    ex.expand({static_cast<const u64*>(d.data()), 0, W, 0}, nullptr, e);
    EXPECT_EQ(e.total, 0u);
    EXPECT_EQ(to_host<i32>(off.data(), 1, s)[0], 0);
}

TEST(CudaExpand, LazySlotsFirstMetOnTheDevice)
{
    SKIP_WITHOUT_GPU();
    // a fresh lazy task: the successors' atoms have no slots until the device meets them (interned in canonical id
    // order); the CPU then expands the same rows on the same task byte for byte
    for (const char* name : {"gripper__prob05", "logistics00__probLOGISTICS-6-1", "sokoban-opt08-strips__p14",
                             "philosophers__p03-phil4", "caldera-split-opt18-adl__p04", "organic-synthesis-opt18-strips__p20"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, false);
        auto ctx = context();
        const cudaStream_t s = ctx->stream();
        cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task));
        const u32 W = task->max_words(), L = rl::max_label_width(*task);
        std::vector<u64> frontier(W, 0);
        const State s0 = task->initial_state();
        std::copy(s0.words().begin(), s0.words().end(), frontier.begin());
        const u32 slots0 = task->atoms().fluent_slots();
        for (int depth = 0; depth < 3 && !frontier.empty(); ++depth)
        {
            const u64 N = frontier.size() / W;
            cuda::DeviceBuffer d(ctx, frontier.size() * 8, s);
            cuda::check(cudaMemcpyAsync(d.data(), frontier.data(), frontier.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
            const rl::StateBatchView in{static_cast<const u64*>(d.data()), N, W, 0};
            const u64 total = ex.count(in, nullptr, {});
            DeviceFlat df(ctx, N, total, W, L, s);
            ex.write(df.x);
            const Flat dev = df.host(N, s);
            const Flat cpu = cpu_expand(task, frontier, N, W, total, W, L, {});
            ASSERT_EQ(dev, cpu) << "depth " << depth;
            // the next frontier: up to 48 distinct successors
            std::vector<u64> next;
            for (u64 j = 0; j < dev.total && next.size() < 48 * W; ++j)
            {
                const auto row = dev.succ.begin() + static_cast<std::ptrdiff_t>(j * W);
                bool seen = std::equal(row, row + W, frontier.begin());
                for (usize k = 0; k < next.size() && !seen; k += W)
                    seen = std::equal(row, row + W, next.begin() + static_cast<std::ptrdiff_t>(k));
                if (!seen)
                    next.insert(next.end(), row, row + W);
            }
            frontier = std::move(next);
        }
        EXPECT_GT(task->atoms().fluent_slots(), slots0) << "no atom was first met on the device";
    }
}

TEST(CudaExpand, ThreadsAndStreamsShareATask)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("logistics00__probLOGISTICS-6-1", false);
    const std::vector<State> walk = device_ref_walks(*task, 8, 30, 5);
    const u32 W = task->max_words();
    std::vector<u64> states(walk.size() * W, 0);
    for (usize i = 0; i < walk.size(); ++i)
        std::copy(walk[i].words().begin(), walk[i].words().end(), states.begin() + static_cast<std::ptrdiff_t>(i * W));
    const u64 N = walk.size();
    const u32 L = rl::max_label_width(*task);
    const Flat ref = cpu_expand(task, states, N, W, 20000, W, L, {});
    auto ctx = context();
    constexpr int T = 4;
    std::vector<std::unique_ptr<cuda::Stream>> streams;
    for (int t = 0; t < T; ++t)
        streams.push_back(std::make_unique<cuda::Stream>());
    std::vector<Flat> got(T);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < T; ++t)
            threads.emplace_back(
                [&, t]
                {
                    const cudaStream_t s = *streams[t];
                    cuda::DeviceBuffer d(ctx, states.size() * 8, s);
                    cuda::check(cudaMemcpyAsync(d.data(), states.data(), states.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
                    cuda::DeviceExpander ex(ctx, rl::TaskTable::single(task), s);
                    for (int rep = 0; rep < 3; ++rep)
                    {
                        DeviceFlat df(ctx, N, 20000, W, L, s);
                        ex.expand({static_cast<const u64*>(d.data()), N, W, 0}, nullptr, df.x);
                        got[t] = df.host(N, s);
                    }
                });
        for (auto& th : threads)
            th.join();
    }
    for (int t = 0; t < T; ++t)
        EXPECT_EQ(got[t], ref) << "thread " << t;
}
}  // namespace
