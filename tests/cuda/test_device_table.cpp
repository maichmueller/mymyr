// Device task-table tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device), on the instance sets of
// tests/cpp/rl/table_instance_sets.hpp (the fork's PDDL through the front end, plus tests/data/table_instances), frozen and lazy:
//   - the device expand of a mixed batch over a table (cuda::DeviceExpander: one pass of the multi-instance kernels,
//     or each instance's rows apart) equals rl::expand byte for byte, flat (with capacity clipping) and padded, for
//     any chunk size and bucket launch (rl::expand's mixed batch equals the single-instance expansions:
//     tests/cpp/rl/test_table.cpp);
//   - the device env over a table (cuda::DeviceEnv, fast and general path) equals rl::HostEnv step by step on a mixed
//     batch: random and given actions (invalid ones too), truncation, autoreset into the rows' instances and into
//     next_task_ids, per-env goals, both dead-end modes; with the bucket launch changing between steps, tiny chunks, a
//     stream switch, and the batch's rows rearranged in the middle (a stale launch order: not a permutation of the
//     rows, or one no longer sorted by key, so the range launches take every position);
//   - task ids outside the table are reported.

#include "env_harness.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/rl/expand.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
// ------------------------------------------------------------------------------------------------ expand

/// A flat expansion on the host.
struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    u64 total = 0;
    u32 words_needed = 0;
};

Flat host_expand(const rl::TaskTable& T, const std::vector<u64>& rows, const std::vector<i32>& ids, u64 cap, u32 L)
{
    const u32 W = T.words();
    const u64 N = ids.size();
    Flat f;
    f.succ.assign(std::max<u64>(cap, 1) * W, 0x5A5A5A5A5A5A5A5Aull);
    f.parent.assign(std::max<u64>(cap, 1), 0x5A5A5A5A);
    f.schema.assign(std::max<u64>(cap, 1), 0x5A5A5A5A);
    f.binding.assign(std::max<u64>(cap, 1) * L, 0x5A5A5A5A);
    f.goal.assign(std::max<u64>(cap, 1), 0x5A);
    f.offsets.assign(N + 1, 0);
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
    rl::expand(T, rl::StateBatchView{rows.data(), N, W, 0}, ids.data(), x);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

/// write() of the counted batch into buffers poisoned like host_expand's.
Flat device_write(const cuda::ContextPtr& ctx, cuda::DeviceExpander& ex, u64 N, u32 W, u64 cap, u32 L)
{
    const cudaStream_t s = ex.stream();
    const u64 c = std::max<u64>(cap, 1);
    cuda::DeviceBuffer succ = to_device(ctx, std::vector<u64>(c * W, 0x5A5A5A5A5A5A5A5Aull), s);
    cuda::DeviceBuffer parent = to_device(ctx, std::vector<i32>(c, 0x5A5A5A5A), s);
    cuda::DeviceBuffer schema = to_device(ctx, std::vector<i32>(c, 0x5A5A5A5A), s);
    cuda::DeviceBuffer binding = to_device(ctx, std::vector<i32>(c * L, 0x5A5A5A5A), s);
    cuda::DeviceBuffer goal = to_device(ctx, std::vector<u8>(c, 0x5A), s);
    cuda::DeviceBuffer offsets(ctx, (N + 1) * sizeof(i32), s);
    rl::Expansion x;
    x.capacity = cap;
    x.words = W;
    x.label_width = L;
    x.succ = static_cast<u64*>(succ.data());
    x.parent = static_cast<i32*>(parent.data());
    x.schema = static_cast<i32*>(schema.data());
    x.binding = static_cast<i32*>(binding.data());
    x.goal = static_cast<u8*>(goal.data());
    x.offsets = static_cast<i32*>(offsets.data());
    ex.write(x);
    Flat f;
    f.succ = to_host<u64>(succ.data(), c * W, s);
    f.parent = to_host<i32>(parent.data(), c, s);
    f.schema = to_host<i32>(schema.data(), c, s);
    f.binding = to_host<i32>(binding.data(), c * L, s);
    f.goal = to_host<u8>(goal.data(), c, s);
    f.offsets = to_host<i32>(offsets.data(), N + 1, s);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

/// The same expansion on the device (count(), then write()).
Flat device_expand(const cuda::ContextPtr& ctx, cuda::DeviceExpander& ex, const cuda::DeviceBuffer& d_rows,
                   const cuda::DeviceBuffer& d_ids, u64 N, u32 W, u64 cap, u32 L, u64* total)
{
    *total = ex.count(rl::StateBatchView{static_cast<const u64*>(d_rows.data()), N, W, 0},
                      static_cast<const i32*>(d_ids.data()));
    return device_write(ctx, ex, N, W, cap, L);
}

void expect_flat_equal(const Flat& d, const Flat& h)
{
    ASSERT_EQ(d.total, h.total);
    ASSERT_EQ(d.words_needed, h.words_needed);
    ASSERT_EQ(d.offsets, h.offsets);
    ASSERT_EQ(d.parent, h.parent);
    ASSERT_EQ(d.schema, h.schema);
    ASSERT_EQ(d.binding, h.binding);
    ASSERT_EQ(d.succ, h.succ);
    ASSERT_EQ(d.goal, h.goal);
}

// ------------------------------------------------------------------------------------------------ envs

/// Rows of `width` elements of `v`, in the order `perm` (row k of the result is row perm[k]).
template<class T>
std::vector<T> permuted(const std::vector<T>& v, const std::vector<u64>& perm, u64 width)
{
    std::vector<T> out(v.size());
    for (u64 k = 0; k < perm.size(); ++k)
        std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(perm[k] * width), width,
                    out.begin() + static_cast<std::ptrdiff_t>(k * width));
    return out;
}

/// Rearranges rows of a host vector in place (its storage stays: the batch points into it).
template<class T>
void permute_host(std::vector<T>& v, const std::vector<u64>& perm, u64 width)
{
    const std::vector<T> p = permuted(v, perm, width);
    std::copy(p.begin(), p.end(), v.begin());
}

/// Rearranges a device array of N rows of `width` Ts in place: the first `cols` of each row (the others stay).
template<class T>
void permute_device(cuda::DeviceBuffer& d, u64 N, u64 width, const std::vector<u64>& perm, cudaStream_t s, u64 cols)
{
    if (width == 0)
        return;
    std::vector<T> v = to_host<T>(d.data(), N * width, s);
    const std::vector<T> p = permuted(v, perm, width);
    for (u64 k = 0; k < N; ++k)
        std::copy_n(p.begin() + static_cast<std::ptrdiff_t>(k * width), cols,
                    v.begin() + static_cast<std::ptrdiff_t>(k * width));
    upload(d.data(), v, s);
}

/// Rearranges a device array of N rows of `width` Ts in place.
template<class T>
void permute_device(cuda::DeviceBuffer& d, u64 N, u64 width, const std::vector<u64>& perm, cudaStream_t s)
{
    permute_device<T>(d, N, width, perm, s, width);
}

struct RunOptions
{
    bool goals = false;         // per-env goal masks (custom masks for some rows)
    bool rearrange = false;     // rotate the batch's rows in the middle (both sides; the device's launch order is stale)
    bool keep_order = false;    // and leave the counts' launch-order columns in place: still a permutation of the
                                // rows, no longer sorted by key (several instances)
    bool switch_stream = false;  // move the device env to another stream in the middle
    u64 chunk_rows = 0;
    int steps = 36;
};

/// Steps a mixed batch on the host and on the device side by side and compares every step.
void run_env(const cuda::ContextPtr& ctx, const rl::TaskTablePtr& table, const rl::EnvConfig& cfg,
             cuda::DeviceEnv::Path path, const RunOptions& r)
{
    const rl::TaskTable& T = *table;
    const u32 I = T.size(), W = T.words();
    std::vector<i32> ids;
    for (u32 k = 0; k < 6 * I + 3; ++k)
        ids.push_back(static_cast<i32>((k * 5 + k / I) % I));
    const u64 N = ids.size();
    const cuda::Stream other;  // outlives the env (its scratch may live on this stream)
    cuda::DeviceEnv env(ctx, table, cfg, path);
    rl::HostEnv host(table, cfg);
    for (u32 i = 0; i < I; ++i)
        ASSERT_EQ(env.initial_count(i), host.initial_count(i));
    if (r.chunk_rows)
        env.set_chunk_rows(r.chunk_rows);
    HostEnvs h(T, ids, r.goals);
    DeviceEnvs d(ctx, env, ids, r.goals);
    host.reset(h.b, nullptr, h.out.count);
    env.reset(d.b, nullptr, d.out.count);
    if (r.goals)
    {
        custom_goals(h.v, W);
        upload(d.gpos.data(), h.v.gpos, d.s);
        upload(d.gneg.data(), h.v.gneg, d.s);
    }
    for (int t = 0; t < r.steps; ++t)
    {
        env.set_launch(t % 2 ? cuda::BucketLaunch::PerBucket : cuda::BucketLaunch::Widest);
        if (r.switch_stream && t == r.steps / 3)
        {
            env.set_stream(other);
            d.s = other;
        }
        if (r.rearrange && t == r.steps / 2)
        {
            // rotate the rows by one on both sides (the device's cache rows move along: its launch order columns no
            // longer form a permutation, the launches take the rows in batch order)
            std::vector<u64> perm(N);
            for (u64 k = 0; k < N; ++k)
                perm[k] = (k + 1) % N;
            permute_host(h.v.states, perm, W);
            permute_host(h.v.task_ids, perm, 1);
            permute_host(h.v.steps, perm, 1);
            permute_host(h.v.draws, perm, 1);
            permute_host(h.v.count, perm, 1);
            if (r.goals)
            {
                permute_host(h.v.gpos, perm, W);
                permute_host(h.v.gneg, perm, W);
            }
            permute_device<u64>(d.states, N, W, perm, d.s);
            permute_device<i32>(d.task_ids, N, 1, perm, d.s);
            permute_device<i32>(d.steps, N, 1, perm, d.s);
            permute_device<u64>(d.draws, N, 1, perm, d.s);
            permute_device<i32>(d.count, N, 1, perm, d.s);
            permute_device<u32>(d.counts, N, d.C, perm, d.s, r.keep_order && I > 1 ? d.C - 2 : d.C);
            permute_device<u64>(d.views, N, d.V, perm, d.s);
            if (r.goals)
            {
                permute_device<u64>(d.gpos, N, W, perm, d.s);
                permute_device<u64>(d.gneg, N, W, perm, d.s);
            }
        }
        // every third step takes given actions (from the counts of the previous step), every fourth restarts finished
        // rows in other instances
        const bool given = t % 3 == 2, curriculum = t % 4 == 1;
        const std::vector<i64> act = actions(h.v.count, t);
        std::vector<i32> next(N);
        for (u64 i = 0; i < N; ++i)
            next[i] = static_cast<i32>((i * 7 + static_cast<u64>(t)) % I);
        if (given)
            upload(d.action.data(), act, d.s);
        if (curriculum)
            upload(d.next.data(), next, d.s);
        env.step(d.b, d.out, rl::Actions{given ? static_cast<const i64*>(d.action.data()) : nullptr},
                 curriculum ? static_cast<const i32*>(d.next.data()) : nullptr);
        host.step(h.b, h.out, rl::Actions{given ? act.data() : nullptr}, curriculum ? next.data() : nullptr);
        env.check_errors();
        expect_equal(d.snapshot(), h.v, t);
        if (::testing::Test::HasFatalFailure())
            break;
    }
    other.synchronize();
}
}  // namespace

// ------------------------------------------------------------------------------------------------ expand

TEST(DeviceTableExpand, MixedBatchEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const Set& set : sets(atoms))
        {
            SCOPED_TRACE(set.name + atoms_name(atoms));
            const rl::TaskTable& T = *set.table;
            std::vector<u64> rows;
            std::vector<i32> ids;
            table_rows(T, 12, rows, ids);
            const u32 W = T.words(), L = std::max<u32>(1, T.label_width()) + 1;  // one column of -1 padding more
            const u64 N = ids.size();
            const Flat full = host_expand(T, rows, ids, 1, L);  // counts (and interns first under lazy slots)
            cuda::Stream other;  // (a stream the expander ran on must outlive its scratch)
            cuda::DeviceExpander ex(ctx, set.table);
            const bool one_pass = atoms == TaskOptions::Atoms::Frozen && set.name != "miconic-simpleadl";
            EXPECT_EQ(ex.mode(), one_pass ? cuda::DeviceExpander::Mode::Multi : cuda::DeviceExpander::Mode::PerInstance);
            const cudaStream_t s = ex.stream();
            const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s);
            const cuda::DeviceBuffer d_ids = to_device(ctx, ids, s);
            for (const u64 chunk : {u64{0}, u64{7}})
                for (const auto launch : {cuda::BucketLaunch::Widest, cuda::BucketLaunch::PerBucket})
                    for (const u64 cap : {full.total, full.total / 2})
                    {
                        SCOPED_TRACE("chunk " + std::to_string(chunk) + (launch == cuda::BucketLaunch::Widest ? " widest" : " per bucket") +
                                     " capacity " + std::to_string(cap));
                        ex.set_chunk_rows(chunk);
                        ex.set_launch(launch);
                        const Flat h = host_expand(T, rows, ids, cap, L);
                        u64 total = 0;
                        const Flat d = device_expand(ctx, ex, d_rows, d_ids, N, W, cap, L, &total);
                        ASSERT_EQ(total, h.total);
                        expect_flat_equal(d, h);
                        if (::testing::Test::HasFatalFailure())
                            return;
                    }
            // several chunks: a second write() of the counted batch, and (one pass) one after a stream switch, equal the
            // first (the per-instance path's parts are scratch: count() again)
            {
                SCOPED_TRACE("rewrite");
                ex.set_chunk_rows(5);
                const Flat h = host_expand(T, rows, ids, full.total, L);
                u64 total = 0;
                expect_flat_equal(device_expand(ctx, ex, d_rows, d_ids, N, W, full.total, L, &total), h);
                expect_flat_equal(device_write(ctx, ex, N, W, full.total, L), h);
                ex.set_stream(other.get());
                if (one_pass)
                    expect_flat_equal(device_write(ctx, ex, N, W, full.total, L), h);
                else
                    EXPECT_THROW((void)device_write(ctx, ex, N, W, full.total, L), std::logic_error);
                ex.set_stream(nullptr);
                ex.set_chunk_rows(0);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
            // the padded view on the device equals rl::pad's
            {
                const Flat h = host_expand(T, rows, ids, full.total, L);
                i32 maxc = 0;
                for (u64 r = 0; r < N; ++r)
                    maxc = std::max(maxc, h.offsets[r + 1] - h.offsets[r]);
                const u32 K = static_cast<u32>(std::max(1, maxc - 1));  // one short: overflow
                rl::Expansion hx;
                hx.capacity = h.total;
                hx.words = W;
                hx.label_width = L;
                hx.succ = const_cast<u64*>(h.succ.data());
                hx.schema = const_cast<i32*>(h.schema.data());
                hx.binding = const_cast<i32*>(h.binding.data());
                hx.goal = const_cast<u8*>(h.goal.data());
                hx.offsets = const_cast<i32*>(h.offsets.data());
                std::vector<i32> index(N * K), count(N), schema(N * K), binding(N * K * L);
                std::vector<u8> mask(N * K), goal(N * K);
                std::vector<u64> succ(N * K * W);
                rl::PaddedExpansion hp{K, index.data(), mask.data(), count.data(), W, succ.data(), schema.data(), L,
                                       binding.data(), goal.data(), false, 0};
                rl::pad(hx, N, hp);
                const cuda::DeviceBuffer ds = to_device(ctx, h.succ, s), dsc = to_device(ctx, h.schema, s),
                                         db = to_device(ctx, h.binding, s), dg = to_device(ctx, h.goal, s),
                                         doff = to_device(ctx, h.offsets, s);
                cuda::DeviceBuffer pi(ctx, N * K * 4, s), pm(ctx, N * K, s), pc(ctx, N * 4, s), psc(ctx, N * K * 4, s),
                    pb(ctx, N * K * L * 4, s), pg(ctx, N * K, s), psu(ctx, N * K * W * 8, s);
                rl::Expansion dx = hx;
                dx.succ = static_cast<u64*>(ds.data());
                dx.schema = static_cast<i32*>(dsc.data());
                dx.binding = static_cast<i32*>(db.data());
                dx.goal = static_cast<u8*>(dg.data());
                dx.offsets = static_cast<i32*>(doff.data());
                rl::PaddedExpansion dp{K, static_cast<i32*>(pi.data()), static_cast<u8*>(pm.data()),
                                       static_cast<i32*>(pc.data()), W, static_cast<u64*>(psu.data()),
                                       static_cast<i32*>(psc.data()), L, static_cast<i32*>(pb.data()),
                                       static_cast<u8*>(pg.data()), false, 0};
                ex.pad(dx, N, dp);
                EXPECT_EQ(dp.overflow, hp.overflow);
                EXPECT_EQ(to_host<i32>(pi.data(), N * K, s), index);
                EXPECT_EQ(to_host<u8>(pm.data(), N * K, s), mask);
                EXPECT_EQ(to_host<i32>(pc.data(), N, s), count);
                EXPECT_EQ(to_host<i32>(psc.data(), N * K, s), schema);
                EXPECT_EQ(to_host<i32>(pb.data(), N * K * L, s), binding);
                EXPECT_EQ(to_host<u8>(pg.data(), N * K, s), goal);
                EXPECT_EQ(to_host<u64>(psu.data(), N * K * W, s), succ);
            }
        }
}

TEST(DeviceTableExpand, TaskIdsAreChecked)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto all = sets(atoms);
        ASSERT_FALSE(all.empty());
        const rl::TaskTable& T = *all[0].table;
        std::vector<u64> rows;
        std::vector<i32> ids;
        table_rows(T, 2, rows, ids);
        cuda::DeviceExpander ex(ctx, all[0].table);
        const cudaStream_t s = ex.stream();
        const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s);
        const rl::StateBatchView in{static_cast<const u64*>(d_rows.data()), ids.size(), T.words(), 0};
        EXPECT_THROW((void)ex.count(in, nullptr), std::invalid_argument);  // several instances need task ids
        for (const i32 bad : {static_cast<i32>(T.size()), -1})
        {
            std::vector<i32> b = ids;
            b[3] = bad;
            const cuda::DeviceBuffer d_ids = to_device(ctx, b, s);
            try
            {
                (void)ex.count(in, static_cast<const i32*>(d_ids.data()));
                ADD_FAILURE() << "task id " << bad << " was accepted";
            }
            catch (const std::invalid_argument& e)
            {
                EXPECT_NE(std::string(e.what()).find("task id " + std::to_string(bad) + " of row 3"), std::string::npos)
                    << e.what();
            }
        }
        // a row of another instance's width: bits past its instance's slots
        const cuda::DeviceBuffer d_ids = to_device(ctx, ids, s);
        std::vector<u64> wide = rows;
        u32 small = 0;
        for (u32 i = 1; i < T.size(); ++i)
            if (T.task(i)->atoms().fluent_slots() < T.task(small)->atoms().fluent_slots())
                small = i;
        const u64 r = static_cast<u64>(std::find(ids.begin(), ids.end(), static_cast<i32>(small)) - ids.begin());
        if (T.task(small)->atoms().fluent_slots() < 64 * T.words())
        {
            wide[r * T.words() + T.words() - 1] |= u64{1} << 63;
            const cuda::DeviceBuffer d_wide = to_device(ctx, wide, s);
            EXPECT_THROW((void)ex.count(rl::StateBatchView{static_cast<const u64*>(d_wide.data()), ids.size(), T.words(), 0},
                                        static_cast<const i32*>(d_ids.data())),
                         std::invalid_argument);
        }
    }
}

// ------------------------------------------------------------------------------------------------ envs

TEST(DeviceTableEnv, MixedBatchEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const Set& set : sets(atoms))
            for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
            {
                SCOPED_TRACE(set.name + atoms_name(atoms) + (path == cuda::DeviceEnv::Path::Auto ? " auto" : " general"));
                if (path == cuda::DeviceEnv::Path::Auto)
                {
                    const std::string why = cuda::DeviceEnv::fast_unsupported(*set.table, config(1));
                    const bool fast = atoms == TaskOptions::Atoms::Frozen && set.name != "miconic-simpleadl";
                    EXPECT_EQ(why.empty(), fast) << why;
                }
                RunOptions r;
                r.rearrange = true;
                run_env(ctx, set.table, config(0x5eed + set.table->size()), path, r);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
}

TEST(DeviceTableEnv, OrderNotSortedByKeyIsStale)
{
    // the rows rotated with the launch order left in place: a permutation of the rows whose positions no longer group
    // the instances, so the step's select reports it stale and the range launches take every position
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const Set& set : sets(TaskOptions::Atoms::Frozen))
    {
        if (!cuda::DeviceEnv::fast_unsupported(*set.table, config(1)).empty())
            continue;
        SCOPED_TRACE(set.name);
        RunOptions r;
        r.rearrange = true;
        r.keep_order = true;
        r.steps = 20;
        run_env(ctx, set.table, config(0x0de4 + set.table->size()), cuda::DeviceEnv::Path::Auto, r);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST(DeviceTableEnv, GoalsDeadEndsChunksAndStreams)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const Set& set : sets(TaskOptions::Atoms::Frozen))
    {
        if (set.name == "miconic-simpleadl")
            continue;  // goal axioms: per-env goal masks cannot express them
        for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
        {
            SCOPED_TRACE(set.name + (path == cuda::DeviceEnv::Path::Auto ? " auto" : " general"));
            rl::EnvConfig cfg = config(77);
            RunOptions r;
            r.goals = true;
            r.switch_stream = true;
            run_env(ctx, set.table, cfg, path, r);
            if (::testing::Test::HasFatalFailure())
                return;
            cfg.dead_end = rl::DeadEnd::None;
            cfg.autoreset = false;
            r.goals = false;
            r.switch_stream = false;
            r.chunk_rows = 5;
            run_env(ctx, set.table, cfg, path, r);
            if (::testing::Test::HasFatalFailure())
                return;
            cfg.dead_end = rl::DeadEnd::NoSuccessors;
            cfg.dead_end_terminal = false;
            cfg.autoreset = true;
            run_env(ctx, set.table, cfg, path, r);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

TEST(DeviceTableEnv, RankBitmapsOfLongSegmentsEqualHost)
{
    // gripper's pick and drop: sorted segments of up to 2 x 80 bindings, recorded in the count cache by the bitmaps of
    // their canonical ranks, so their picks read the cache: alone (gripper-80) and in the mixed table
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const Set& set : sets(TaskOptions::Atoms::Frozen))
    {
        if (set.name != "gripper")
            continue;
        const rl::TaskTablePtr widest = rl::TaskTable::single(set.tasks.back());
        // a rank space of one bitmap word (gripper with 4 balls: 16 ranks) is not recorded: its picks search
        EXPECT_EQ(cuda::DeviceEnv(ctx, rl::TaskTable::single(set.tasks.front()), config(5)).cache_regions(), 0u);
        for (const rl::TaskTablePtr& table : {widest, set.table})
        {
            SCOPED_TRACE(table->size() == 1 ? "gripper-80" : "gripper set");
            {
                const cuda::DeviceEnv env(ctx, table, config(5));
                ASSERT_TRUE(env.fast());
                EXPECT_EQ(env.cache_regions(), 2u);  // pick and drop
            }
            RunOptions r;
            r.steps = 40;
            run_env(ctx, table, config(0xb17 + table->size()), cuda::DeviceEnv::Path::Auto, r);
            if (::testing::Test::HasFatalFailure())
                return;
        }
        // states written without refresh(): the bitmaps belong to other states, which the picks detect
        rl::EnvConfig cfg = config(9);
        cfg.max_steps = 0;
        cuda::DeviceEnv env(ctx, widest, cfg);
        const std::vector<i32> ids(64, 0);
        DeviceEnvs d(ctx, env, ids, false);
        env.reset(d.b, nullptr, static_cast<i32*>(d.count.data()));
        for (int t = 0; t < 6; ++t)
            env.step(d.b, d.out);
        EXPECT_NO_THROW(env.check_errors());
        std::vector<u64> other = to_host<u64>(d.states.data(), d.N * d.W, d.s);
        std::rotate(other.begin(), other.begin() + d.W, other.end());
        upload(d.states.data(), other, d.s);
        for (int t = 0; t < 4; ++t)
            env.step(d.b, d.out);
        EXPECT_THROW(env.check_errors(), std::logic_error);
        env.refresh(d.b, static_cast<i32*>(d.count.data()));
        env.step(d.b, d.out);
        EXPECT_NO_THROW(env.check_errors());
    }
}

TEST(DeviceTableEnv, TaskIdsAreChecked)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = sets(TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(all.empty());
    const rl::TaskTablePtr& table = all[0].table;
    const u32 I = table->size();
    for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
    {
        cuda::DeviceEnv env(ctx, table, config(3), path);
        std::vector<i32> ids(16);
        for (u64 i = 0; i < ids.size(); ++i)
            ids[i] = static_cast<i32>(i % I);
        DeviceEnvs d(ctx, env, ids, false);
        rl::EnvBatch no_ids = d.b;
        no_ids.task_ids = nullptr;
        EXPECT_THROW(env.reset(no_ids), std::invalid_argument);
        env.reset(d.b);
        env.check_errors();
        // a next task id outside the table: reported, the row keeps its instance
        std::vector<i32> next(ids.size(), static_cast<i32>(I));
        upload(d.next.data(), next, d.s);
        for (int t = 0; t < 12; ++t)
            env.step(d.b, d.out, {}, static_cast<const i32*>(d.next.data()));
        EXPECT_THROW(env.check_errors(), std::invalid_argument);
        env.check_errors();  // reported once
        // a task id outside the table in reset
        ids[5] = -2;
        upload(d.task_ids.data(), ids, d.s);
        env.reset(d.b);
        EXPECT_THROW(env.check_errors(), std::invalid_argument);
    }
}
