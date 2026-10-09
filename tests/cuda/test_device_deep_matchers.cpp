// Device deep-matcher tests (run on GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - deep matchers: organic-synthesis' reactions (matchers of 17 to 31 parameters) run in the deep kernels, not on the
//     CPU fallback, and the device expand is byte-equal to rl::expand with them (canonical order and the matcher's,
//     with and without witness pruning), frozen and lazy; exact IW(1) over them gives the same searches in chunks of one
//     state as in one chunk;
//   - the deep kernels' second pass (several warps on a segment's search) gives the rows of the first (one warp),
//     forward checking and fixed order; the device BrFS over them equals the CPU's in the memory it needs, with its
//     loops' chunks cut to a candidate budget;
//   - the launch order of multi-instance batches (lifted::launch_order: the counting sort and CUB's radix sort) equals
//     a stable sort by key on the host, for keys anywhere in their key_bits range.
#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/search/brfs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
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
    o.max_bytes = u64{3} << 30;
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
    std::vector<T> h(n);
    if (n)
        cuda::check(cudaMemcpyAsync(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return h;
}

constexpr const char* k_organic = "organic-synthesis-opt18-strips__p20";

TEST(CudaDeep, OrganicSynthesisReactionsRunOnTheDevice)
{
    SKIP_WITHOUT_GPU();
    for (bool frozen : {true, false})
    {
        const auto task = load(k_organic, frozen);
        const cuda::ChunkGenerator gen(context(), task);
        for (bool witness : {true, false})
        {
            SCOPED_TRACE(std::string(frozen ? "frozen" : "lazy") + (witness ? ", witness" : ""));
            const cuda::SchemaPlacement& pl = gen.placement(witness);
            EXPECT_EQ(pl.host_count, 0u);
            u32 deep = 0, deeper = 0;
            for (u32 s = 0; s < task->num_schemas(); ++s)
            {
                const plan::Matcher& m = task->compiled().schemas[s].pre[witness ? 0 : 1];
                if (m.total > cuda::lifted::k_max_depth)
                {
                    ++deeper;
                    EXPECT_EQ(pl.deep[s], 1) << task->schema_name(SchemaId{s});
                }
                deep += pl.deep[s];
            }
            EXPECT_GE(deeper, 6u);  // 17, 21, 22, 29 and 31 parameters
            EXPECT_EQ(deep, pl.deep_count);
            EXPECT_GE(deep, deeper);
        }
        EXPECT_FALSE(gen.needs_host(true));
    }
}

TEST(CudaDeep, OrganicSynthesisExpandEqualsRlExpand)
{
    SKIP_WITHOUT_GPU();
    for (bool frozen : {true, false})
    {
        const auto task = load(k_organic, frozen);
        const std::vector<State> walk = device_ref_walks(*task, 4, 20, 23);
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
        const u32 L = rl::max_label_width(*task);
        ASSERT_GT(L, cuda::lifted::k_max_label);
        // canonical order: the device's matcher (fixed order); the matcher's order: the plan's (forward checking)
        for (bool canonical : {true, false})
            for (bool witness : {false, true})
            {
                SCOPED_TRACE(std::string(frozen ? "frozen" : "lazy") + (canonical ? ", canonical" : ", matcher order") +
                             (witness ? ", witness" : ""));
                const rl::ExpandOptions o{canonical, witness, true};
                const u64 total = ex.count({static_cast<const u64*>(d_states.data()), N, Win, 0}, nullptr, o);
                ASSERT_GT(total, 0u);
                const u32 W = std::max(Win, task->max_words());
                cuda::DeviceBuffer succ(ctx, total * W * 8, s), parent(ctx, total * 4, s), schema(ctx, total * 4, s),
                    binding(ctx, total * L * 4, s), offsets(ctx, (N + 1) * 4, s);
                rl::Expansion d;
                d.capacity = total;
                d.words = W;
                d.label_width = L;
                d.succ = static_cast<u64*>(succ.data());
                d.parent = static_cast<i32*>(parent.data());
                d.schema = static_cast<i32*>(schema.data());
                d.binding = static_cast<i32*>(binding.data());
                d.offsets = static_cast<i32*>(offsets.data());
                ex.write(d);
                std::vector<u64> c_succ(total * W);
                std::vector<i32> c_parent(total), c_schema(total), c_binding(total * L), c_offsets(N + 1);
                rl::Expansion c;
                c.capacity = total;
                c.words = W;
                c.label_width = L;
                c.succ = c_succ.data();
                c.parent = c_parent.data();
                c.schema = c_schema.data();
                c.binding = c_binding.data();
                c.offsets = c_offsets.data();
                rl::expand(*rl::TaskTable::single(task), {states.data(), N, Win, 0}, nullptr, c, o);
                ASSERT_EQ(c.total, total);
                EXPECT_EQ(to_host<i32>(d.offsets, N + 1, s), c_offsets);
                EXPECT_EQ(to_host<i32>(d.parent, total, s), c_parent);
                EXPECT_EQ(to_host<i32>(d.schema, total, s), c_schema);
                EXPECT_EQ(to_host<i32>(d.binding, total * L, s), c_binding);
                EXPECT_EQ(to_host<u64>(d.succ, total * W, s), c_succ);
            }
    }
}

TEST(CudaDeep, OrganicSynthesisIwInSmallChunksEqualsOneChunk)
{
    // CudaMultiIw.ExactModeDoesNotDependOnTheLaunchConfiguration's configurations at small budgets (for the sanitizers)
    SKIP_WITHOUT_GPU();
    const auto task = load("organic-synthesis-opt18-strips__p20", true);
    const std::vector<State> starts = device_ref_walks(*task, 1, 6, 23);
    struct Config
    {
        u32 max_searches, chunk_states;
        u64 emit_budget, view_bytes;
    };
    for (const u32 arity : {1u, 2u})
    {
        SCOPED_TRACE("arity " + std::to_string(arity));
        cuda::MultiIwOptions o;
        o.max_arity = arity;
        o.budget.max_states = arity == 1 ? 200 : 60;
        cuda::DeviceMultiIw ref_run(context(), task, o);
        const cuda::MultiIwBatch ref = ref_run.run(starts);
        for (const Config& c : {Config{1, 1u << 20, 1u << 24, 0}, Config{0, arity == 1 ? 1u : 16u, 1u << 24, 0}, Config{0, 5, 64, 0},
                                Config{3, 1000, 1, 4096}, Config{0, 1u << 20, 4096, 0}})
        {
            SCOPED_TRACE("max_searches " + std::to_string(c.max_searches) + " chunk " + std::to_string(c.chunk_states) +
                         " emit " + std::to_string(c.emit_budget));
            cuda::MultiIwOptions v = o;
            v.max_searches = c.max_searches;
            v.chunk_states = c.chunk_states;
            v.emit_budget = c.emit_budget;
            v.view_bytes = c.view_bytes;
            cuda::DeviceMultiIw run(context(), task, v);
            const cuda::MultiIwBatch b = run.run(starts);
            ASSERT_EQ(b.n, ref.n);
            EXPECT_EQ(b.status, ref.status);
            EXPECT_EQ(b.plan_length, ref.plan_length);
            EXPECT_EQ(b.plan_labels, ref.plan_labels);
            EXPECT_EQ(b.words, ref.words);
            EXPECT_EQ(b.reached, ref.reached);
        }
    }
}

// ------------------------------------------------------------------------------------------------ the two deep-kernel passes

/// The rows of a chunk by a generator: segment offsets, labels and successor words.
struct ChunkRows
{
    std::vector<u32> offsets, binding, schema, parent;
    std::vector<u64> words;
};

ChunkRows chunk_rows(const cuda::ContextPtr& ctx, const TaskPtr& task, const u64* d_states, u32 words, u32 rows, bool witness,
                     bool canonical, u32 deep_budget)
{
    const cudaStream_t s = ctx->stream();
    cuda::ChunkGenerator gen(ctx, task, s);
    gen.set_deep_budget(deep_budget);
    gen.begin(cuda::ChunkInput{d_states, words, words, rows, nullptr, words, 0}, witness, canonical);
    gen.views();
    gen.count();
    ChunkRows r;
    r.offsets = to_host<u32>(gen.seg_offsets(), u64{rows} * gen.num_schemas() + 1, s);
    const u64 total = r.offsets.back();
    const u32 L = gen.label_width(), W = std::max(words, task->max_words());
    cuda::DeviceBuffer binding(ctx, std::max<u64>(total * L, 1) * 4, s), schema(ctx, std::max<u64>(total, 1) * 4, s),
        parent(ctx, std::max<u64>(total, 1) * 4, s), succ(ctx, std::max<u64>(total * W, 1) * 8, s);
    cuda::lifted::Labels labels;
    labels.binding = static_cast<u32*>(binding.data());
    labels.schema = static_cast<u32*>(schema.data());
    labels.parent = static_cast<u32*>(parent.data());
    labels.capacity = total;
    labels.label_width = L;
    gen.write(labels, static_cast<u64*>(succ.data()), W);
    r.binding = to_host<u32>(binding.data(), total * L, s);
    r.schema = to_host<u32>(schema.data(), total, s);
    r.parent = to_host<u32>(parent.data(), total, s);
    r.words = to_host<u64>(succ.data(), total * W, s);
    return r;
}

/// The deep kernels' rows do not depend on their first pass's budget: with one search node every segment goes to the
/// second pass, where several warps share its search (dealing a level's candidates as they finish): its count is their
/// sum, its rows in canonical order are gathered at slots and sorted by its last warp, its rows in the matcher's order
/// come from one warp. Forward checking (the plan's) and fixed order (TaskOptions::Matching::FixedOrder).
TEST(CudaDeep, SecondPassEqualsTheFirst)
{
    SKIP_WITHOUT_GPU();
    for (const TaskOptions::Matching matching : {TaskOptions::Matching::Auto, TaskOptions::Matching::FixedOrder})
    {
        const bool fixed = matching == TaskOptions::Matching::FixedOrder;
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        to.matching = matching;
        const auto task = Task::from_text_file(task_path(k_organic), to);
        // the deep matchers keep the matching's kind
        u32 deep = 0;
        for (const plan::Schema& ps : task->compiled().schemas)
            if (ps.pre[0].total > cuda::lifted::k_max_depth && !ps.pre[0].never)
            {
                ++deep;
                EXPECT_EQ(ps.pre[0].use_fc, !fixed);
            }
        ASSERT_GE(deep, 2u);
        const std::vector<State> walk = device_ref_walks(*task, 4, 20, 23);
        const u32 W = std::max<u32>(task->words(), 1), N = static_cast<u32>(walk.size());
        std::vector<u64> states(u64{N} * W, 0);
        for (u32 i = 0; i < N; ++i)
            std::copy(walk[i].words().begin(), walk[i].words().end(), states.begin() + static_cast<std::ptrdiff_t>(u64{i} * W));
        auto ctx = context();
        cuda::DeviceBuffer d_states(ctx, states.size() * 8, ctx->stream());
        cuda::check(cudaMemcpyAsync(d_states.data(), states.data(), states.size() * 8, cudaMemcpyHostToDevice, ctx->stream()), "H2D");
        for (bool canonical : {true, false})
            for (bool witness : {true, false})
            {
                SCOPED_TRACE(std::string(fixed ? "fixed order" : "forward checking") + (canonical ? ", canonical" : ", matcher order") +
                             (witness ? ", witness" : ""));
                const auto* d = static_cast<const u64*>(d_states.data());
                const ChunkRows ref = chunk_rows(ctx, task, d, W, N, witness, canonical, 0);
                ASSERT_GT(ref.offsets.back(), 0u);
                const ChunkRows split = chunk_rows(ctx, task, d, W, N, witness, canonical, 1);
                EXPECT_EQ(split.offsets, ref.offsets);
                EXPECT_EQ(split.binding, ref.binding);
                EXPECT_EQ(split.schema, ref.schema);
                EXPECT_EQ(split.parent, ref.parent);
                EXPECT_EQ(split.words, ref.words);
            }
    }
}

/// The device BrFS over organic-synthesis' reactions equals the CPU's, in the device memory organic-synthesis needs
/// (its capacity estimates had sized 1.8 GB for 41311 states), and with a candidate budget that cuts its loops'
/// chunks (DeviceBrfsOptions::candidate_bytes).
TEST(CudaDeep, OrganicSynthesisBrfsEqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto task = load(k_organic, true);
    const BrfsResult c = brfs(*task, {.threads = 2, .stop_at_goal = false, .fingerprint = true});
    for (const u64 candidate_bytes : {u64{32} << 20, u64{1}})
    {
        SCOPED_TRACE("candidate bytes " + std::to_string(candidate_bytes));
        const cuda::DeviceBrfsResult d = cuda::brfs(context(), task, {.stop_at_goal = false, .fingerprint = true, .candidate_bytes = candidate_bytes});
        EXPECT_EQ(d.result.states, c.states);
        EXPECT_EQ(d.result.expanded, c.expanded);
        EXPECT_EQ(d.result.generated, c.generated);
        EXPECT_EQ(d.result.goal_states, c.goal_states);
        EXPECT_EQ(d.result.fingerprint, c.fingerprint);
        EXPECT_EQ(d.stats.host_schemas, 0u);
        if (candidate_bytes > 1)
        {
            EXPECT_LT(d.stats.device_bytes, u64{256} << 20);
        }
        else if (d.stats.loops > 0)
        {
            EXPECT_GT(d.stats.cuts, 0u);
        }
    }
}

// ------------------------------------------------------------------------------------------------ launch order

/// lifted::launch_order equals a stable sort of the rows on the host by their instances' keys, clamped to the key
/// count (ids outside [0, instances): key 0), in both of its sorts (the counting sort up to 1024 keys and 2^16
/// (key, tile) cells, CUB's radix sort above), at an order stride: dense ranks (a table's launch groups), keys sparse
/// in their range (the bits of a domain above its table's ranks; most keys at or above the instance count), few keys
/// over many instances (a suite's domains), keys at or past the count.
TEST(CudaLaunchOrder, EqualsAStableSortByKey)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cuda::Stream s;
    struct Case
    {
        std::string name;
        u32 instances, keys;  // the key count
        u32 (*key)(u32 k);
    };
    const std::vector<Case> cases{
        {"dense ranks", 10, 10, [](u32 k) { return (k * 7) % 10; }},
        {"a domain above the ranks", 10, 13, [](u32 k) { return k < 5 ? 4 - k : (1u << 3) | (k - 5); }},
        {"four domains above the ranks", 20, 29, [](u32 k) { return ((k % 4) << 3) | (k / 4); }},
        {"few keys", 20, 4, [](u32 k) { return k / 5; }},
        {"keys past the count", 12, 3, [](u32 k) { return (k % 5) | ((k % 2) << 9); }},
        {"1024 keys", 1024, 1024, [](u32 k) { return 1023 - k; }},
        {"radix sort: 3000 keys", 3000, 3000, [](u32 k) { return (k * 11) % 3000; }},
    };
    // one tile, tile edges, several tiles; with 16 or more keys the largest n leaves the counting sort's cells
    std::vector<u64> sizes{1, 67, 511, 512, 513, 1500, 70000};
    const char* san = std::getenv("MYMYR_TEST_SANITIZER");
    if (!(san && std::string(san) == "1"))
        sizes.push_back(u64{1} << 22);
    for (const Case& c : cases)
    {
        std::vector<u32> key(c.instances);
        for (u32 k = 0; k < c.instances; ++k)
            key[k] = c.key(k);
        cuda::DeviceBuffer dkey(ctx, key.size() * 4, s);
        cuda::check(cudaMemcpyAsync(dkey.data(), key.data(), key.size() * 4, cudaMemcpyHostToDevice, s), "H2D");
        for (const u64 n : sizes)
            for (const u64 stride : {u64{1}, u64{3}})
            {
                SCOPED_TRACE(c.name + ", n " + std::to_string(n) + ", stride " + std::to_string(stride));
                std::vector<u32> inst(n);
                for (u64 i = 0; i < n; ++i)  // instances interleaved, every 97th row outside [0, instances)
                    inst[i] = i % 97 == 50 ? c.instances + static_cast<u32>(i % 5)
                                           : static_cast<u32>((i * 5 + i / 7) % c.instances);
                std::vector<u32> want(n);
                std::iota(want.begin(), want.end(), 0u);
                const auto key_of = [&](u32 r) { return inst[r] < c.instances ? std::min(key[inst[r]], c.keys - 1) : 0u; };
                std::stable_sort(want.begin(), want.end(), [&](u32 a, u32 b) { return key_of(a) < key_of(b); });
                const u64 tb = cuda::lifted::order_temp_bytes(n);
                cuda::DeviceBuffer dinst(ctx, n * 4, s), temp(ctx, tb, s), out(ctx, n * stride * 2 * 4, s);
                cuda::check(cudaMemcpyAsync(dinst.data(), inst.data(), n * 4, cudaMemcpyHostToDevice, s), "H2D");
                cuda::check(cudaMemsetAsync(out.data(), 0xFF, n * stride * 2 * 4, s), "memset");
                auto* o = static_cast<u32*>(out.data());
                ASSERT_EQ(cuda::lifted::launch_order(static_cast<const u32*>(dinst.data()), c.instances,
                                                     {static_cast<const u32*>(dkey.data()), c.keys}, n, temp.data(), tb,
                                                     o, o + 1, stride * 2, s),
                          cudaSuccess);
                const std::vector<u32> got = to_host<u32>(out.data(), n * stride * 2, s);
                for (u64 j = 0; j < n; ++j)
                {
                    ASSERT_EQ(got[j * stride * 2], want[j]) << "position " << j;
                    ASSERT_EQ(got[u64{want[j]} * stride * 2 + 1], j) << "row " << want[j];
                }
            }
    }
}
}  // namespace
