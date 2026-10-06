// Device task-suite tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device): instances of several
// domains in one batch (rl/task_suite.hpp), on device, against the host over the suite (which equals each domain's
// table apart: tests/cpp/rl/test_suite.cpp), on suites of the table instance sets (tests/cpp/rl/table_instance_sets.hpp), frozen
// and lazy:
//   - the device expand over a suite (cuda::SuiteExpander: one DeviceExpander per domain, in place over one run per
//     domain, or gathered and written at the batch rows: DeviceExpander::write with a RowMap, in each mode) equals
//     rl::expand byte for byte, flat (capacity clipping) and padded, for interleaved batches, batches grouped by
//     domain in either order, batches of one domain, single-instance domains, chunk sizes and bucket launches; task
//     ids outside the suite are reported;
//   - the debug check of launch orders (lifted::launch_check_order_keys) finds orders that are not stable sorts;
//   - the device env over a suite (cuda::DeviceEnv; the fast path: each domain's multi-instance kernels on its rows,
//     on a stream of its own; the general path: SuiteExpander) equals rl::HostEnv step by step: random and given
//     actions, truncation, autoresets into the rows' instances and into next_task_ids of other domains, per-env goals,
//     both dead-end modes, the batch's rows rearranged in the middle (a stale launch order), tiny chunks, a stream
//     switch; interleaved and grouped batches;
//   - the fast path over a suite captured into CUDA graphs and replayed equals the eager calls and the host;
//   - which suites the fast path runs, and why not ("domain d (name): ...").

#include "env_harness.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_suite.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct NamedSuite
{
    std::string name;
    rl::TaskSuitePtr suite;
    bool fast = false;  // the fast path runs it (frozen slots, no domain with axioms or conditional effects)
};

/// Suites of two and four domains, and one with conditional effects and goal axioms (the general path).
std::vector<NamedSuite> suites(TaskOptions::Atoms atoms)
{
    std::map<std::string, rl::TaskTablePtr> by;
    for (Set& s : sets(atoms))
        by[s.name] = s.table;
    std::vector<NamedSuite> out;
    for (const std::vector<std::string>& names : std::vector<std::vector<std::string>>{
             {"gripper", "blocks"}, {"miconic", "blocks", "gripper", "logistics"}, {"miconic-simpleadl", "gripper", "blocks"}})
    {
        std::vector<rl::TaskTablePtr> tables;
        std::string name;
        bool fast = atoms == TaskOptions::Atoms::Frozen;
        for (const auto& n : names)
        {
            if (!by.count(n))
                return {};
            tables.push_back(by.at(n));
            name += (name.empty() ? "" : "+") + n;
            fast = fast && n != "miconic-simpleadl";
        }
        out.push_back({name + atoms_name(atoms), rl::TaskSuite::create(tables), fast});
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ expand

enum class Order
{
    Interleaved,  // round robin over the domains
    Grouped,      // domain by domain
    Reversed,     // domain by domain, the last first
    LastOnly,     // the last domain's rows alone
};

/// A batch over a suite: `per` rows of every instance of the domains (table_rows of each domain's table, at the suite's
/// width), in the order `o`; global ids.
void suite_rows(const rl::TaskSuite& S, u32 per, Order o, std::vector<u64>& out, std::vector<i32>& ids_out)
{
    const u32 D = S.num_domains(), W = S.words();
    std::vector<std::vector<u64>> rows(D);
    std::vector<std::vector<i32>> ids(D);
    for (u32 d = 0; d < D; ++d)
        table_rows(*S.table(d), per, rows[d], ids[d]);
    out.clear();
    ids_out.clear();
    auto put = [&](u32 d, u64 k)
    {
        const u32 Wd = S.table(d)->words();
        std::vector<u64> row(W, 0);
        std::copy_n(rows[d].begin() + static_cast<std::ptrdiff_t>(k * Wd), Wd, row.begin());
        out.insert(out.end(), row.begin(), row.end());
        ids_out.push_back(static_cast<i32>(S.global_id(d, static_cast<u32>(ids[d][k]))));
    };
    switch (o)
    {
        case Order::Interleaved:
        {
            std::vector<u64> next(D, 0);
            for (bool any = true; any;)
            {
                any = false;
                for (u32 d = 0; d < D; ++d)
                    if (next[d] < ids[d].size())
                    {
                        put(d, next[d]++);
                        any = true;
                    }
            }
            break;
        }
        case Order::Grouped:
            for (u32 d = 0; d < D; ++d)
                for (u64 k = 0; k < ids[d].size(); ++k)
                    put(d, k);
            break;
        case Order::Reversed:
            for (u32 d = D; d-- > 0;)
                for (u64 k = 0; k < ids[d].size(); ++k)
                    put(d, k);
            break;
        case Order::LastOnly:
            for (u64 k = 0; k < ids[D - 1].size(); ++k)
                put(D - 1, k);
            break;
    }
}

const char* order_name(Order o)
{
    switch (o)
    {
        case Order::Interleaved: return " interleaved";
        case Order::Grouped: return " grouped";
        case Order::Reversed: return " reversed";
        case Order::LastOnly: return " last domain only";
    }
    return "";
}

struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    u64 total = 0;
    u32 words_needed = 0;
};

Flat host_expand(const rl::TaskSuite& S, const std::vector<u64>& rows, const std::vector<i32>& ids, u64 cap, u32 L)
{
    const u32 W = S.words();
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
    rl::expand(S, rl::StateBatchView{rows.data(), N, W, 0}, ids.data(), x);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

/// write() of the counted batch into buffers poisoned like host_expand's.
Flat device_write(const cuda::ContextPtr& ctx, cuda::SuiteExpander& ex, u64 N, u32 W, u64 cap, u32 L)
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

template<class T>
std::vector<T> permuted(const std::vector<T>& v, const std::vector<u64>& perm, u64 width)
{
    std::vector<T> out(v.size());
    for (u64 k = 0; k < perm.size(); ++k)
        std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(perm[k] * width), width,
                    out.begin() + static_cast<std::ptrdiff_t>(k * width));
    return out;
}

template<class T>
void permute_host(std::vector<T>& v, const std::vector<u64>& perm, u64 width)
{
    const std::vector<T> p = permuted(v, perm, width);
    std::copy(p.begin(), p.end(), v.begin());
}

template<class T>
void permute_device(cuda::DeviceBuffer& d, u64 N, u64 width, const std::vector<u64>& perm, cudaStream_t s)
{
    if (width == 0)
        return;
    upload(d.data(), permuted(to_host<T>(d.data(), N * width, s), perm, width), s);
}

/// Global ids of a batch of N rows: the domains interleaved (every instance in turn), or grouped by domain.
std::vector<i32> batch_ids(const rl::TaskSuite& S, u64 N, bool grouped)
{
    const u32 I = S.size();
    std::vector<i32> ids(N);
    for (u64 k = 0; k < N; ++k)
        ids[k] = static_cast<i32>((k * 5 + k / I) % I);
    if (grouped)
        std::stable_sort(ids.begin(), ids.end(), [&](i32 a, i32 b)
                         { return S.domain_of(static_cast<u32>(a)) < S.domain_of(static_cast<u32>(b)); });
    return ids;
}

struct RunOptions
{
    bool goals = false;
    bool rearrange = false;
    bool switch_stream = false;
    bool grouped = false;
    u64 chunk_rows = 0;
    int steps = 36;
};

/// Steps a batch over a suite on the host and on the device side by side and compares every step. Every fourth step
/// restarts finished rows in next_task_ids that cross the domains.
void run_env(const cuda::ContextPtr& ctx, const rl::TaskSuitePtr& suite, const rl::EnvConfig& cfg,
             cuda::DeviceEnv::Path path, const RunOptions& r)
{
    const rl::TaskSuite& S = *suite;
    const u32 I = S.size(), W = S.words();
    const std::vector<i32> ids = batch_ids(S, 4 * I + 5, r.grouped);
    const u64 N = ids.size();
    const cuda::Stream other;
    cuda::DeviceEnv env(ctx, suite, cfg, path);
    rl::HostEnv host(suite, cfg);
    for (u32 i = 0; i < I; ++i)
        ASSERT_EQ(env.initial_count(i), host.initial_count(i));
    if (r.chunk_rows)
        env.set_chunk_rows(r.chunk_rows);
    HostEnvs h(S, ids, r.goals);
    DeviceEnvs d(ctx, env, ids, r.goals);
    host.reset(h.b, nullptr, h.out.count);
    env.reset(d.b, nullptr, d.out.count);
    if (r.goals)
    {
        custom_goals(h.v, W);
        upload(d.gpos.data(), h.v.gpos, d.s);
        upload(d.gneg.data(), h.v.gneg, d.s);
    }
    u64 crossed = 0;
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
            std::vector<u64> perm(N);
            for (u64 k = 0; k < N; ++k)
                perm[k] = (k + 3) % N;
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
            permute_device<u32>(d.counts, N, d.C, perm, d.s);
            permute_device<u64>(d.views, N, d.V, perm, d.s);
            if (r.goals)
            {
                permute_device<u64>(d.gpos, N, W, perm, d.s);
                permute_device<u64>(d.gneg, N, W, perm, d.s);
            }
        }
        const bool given = t % 3 == 2, curriculum = t % 4 == 1;
        const std::vector<i64> act = actions(h.v.count, t);
        std::vector<i32> next(N);
        for (u64 i = 0; i < N; ++i)
            next[i] = static_cast<i32>((i * 7 + static_cast<u64>(t)) % I);
        if (given)
            upload(d.action.data(), act, d.s);
        if (curriculum)
            upload(d.next.data(), next, d.s);
        const std::vector<i32> before = h.v.task_ids;
        env.step(d.b, d.out, rl::Actions{given ? static_cast<const i64*>(d.action.data()) : nullptr},
                 curriculum ? static_cast<const i32*>(d.next.data()) : nullptr);
        host.step(h.b, h.out, rl::Actions{given ? act.data() : nullptr}, curriculum ? next.data() : nullptr);
        env.check_errors();
        expect_equal(d.snapshot(), h.v, t);
        if (::testing::Test::HasFatalFailure())
            break;
        for (u64 i = 0; i < N; ++i)
            crossed += S.domain_of(static_cast<u32>(before[i])) != S.domain_of(static_cast<u32>(h.v.task_ids[i])) ? 1 : 0;
    }
    EXPECT_GT(crossed, 0u);  // some rows restarted in another domain
    other.synchronize();
}

/// A CUDA graph of the work that `record` enqueues on stream s, captured in global mode.
class Graph
{
public:
    Graph(cudaStream_t s, const std::function<void()>& record)
    {
        cuda::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
        try
        {
            record();
        }
        catch (...)
        {
            cudaGraph_t g = nullptr;
            (void)cudaStreamEndCapture(s, &g);
            if (g)
                (void)cudaGraphDestroy(g);
            throw;
        }
        cuda::check(cudaStreamEndCapture(s, &m_graph), "cudaStreamEndCapture");
        cuda::check(cudaGraphInstantiate(&m_exec, m_graph, 0), "cudaGraphInstantiate");
    }
    ~Graph()
    {
        if (m_exec)
            (void)cudaGraphExecDestroy(m_exec);
        if (m_graph)
            (void)cudaGraphDestroy(m_graph);
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    void launch(cudaStream_t s) const { cuda::check(cudaGraphLaunch(m_exec, s), "cudaGraphLaunch"); }

private:
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_exec = nullptr;
};

/// Steps a batch over a suite three ways, K steps at a time: a captured graph of K steps replayed (random steps, given
/// actions, autoresets into next_task_ids across the domains at the last step), the same steps on another env eagerly,
/// and rl::HostEnv; compares all three after every replay. Every fifth replay is preceded by a graph of reset and
/// refresh.
void run_graph(const cuda::ContextPtr& ctx, const rl::TaskSuitePtr& suite, const rl::EnvConfig& cfg, bool goals, u32 K,
               bool grouped)
{
    const rl::TaskSuite& S = *suite;
    const u32 I = S.size(), W = S.words();
    const std::vector<i32> ids = batch_ids(S, std::max<u64>(4 * I + 3, 67), grouped);
    const u64 N = ids.size();
    const int replays = 12;
    const cuda::Stream cs, other;
    cuda::DeviceEnv ge(ctx, suite, cfg, cuda::DeviceEnv::Path::Fast), ee(ctx, suite, cfg, cuda::DeviceEnv::Path::Fast);
    rl::HostEnv host(suite, cfg);
    EXPECT_TRUE(ge.capture_unsupported().empty()) << ge.capture_unsupported();
    ge.set_stream(cs);
    HostEnvs h(S, ids, goals);
    DeviceEnvs g(ctx, ge, ids, goals), e(ctx, ee, ids, goals);
    cuda::DeviceBuffer g_act(ctx, K * N * 8, g.s), g_next(ctx, K * N * 4, g.s), e_act(ctx, K * N * 8, e.s),
        e_next(ctx, K * N * 4, e.s), g_mask(ctx, N, g.s), e_mask(ctx, N, e.s);
    host.reset(h.b, nullptr, h.out.count);
    ge.reset(g.b, nullptr, g.out.count);
    ee.reset(e.b, nullptr, e.out.count);
    if (goals)
    {
        custom_goals(h.v, W);
        upload(g.gpos.data(), h.v.gpos, g.s);
        upload(g.gneg.data(), h.v.gneg, g.s);
        upload(e.gpos.data(), h.v.gpos, e.s);
        upload(e.gneg.data(), h.v.gneg, e.s);
    }
    ge.reserve(N);
    const auto steps = [&](cuda::DeviceEnv& env, DeviceEnvs& d, const cuda::DeviceBuffer& act,
                           const cuda::DeviceBuffer& next) {
        for (u32 k = 0; k < K; ++k)
            env.step(d.b, d.out,
                     rl::Actions{k % 3 == 1 ? static_cast<const i64*>(act.data()) + u64{k} * N : nullptr},
                     k + 1 == K ? static_cast<const i32*>(next.data()) + u64{k} * N : nullptr);
    };
    const Graph step_graph(cs, [&] { steps(ge, g, g_act, g_next); });
    const Graph reset_graph(cs, [&] {
        ge.reset(g.b, static_cast<const u8*>(g_mask.data()), g.out.count, goals);
        ge.refresh(g.b, g.out.count);
    });
    for (int r = 0; r < replays; ++r)
    {
        SCOPED_TRACE("replay " + std::to_string(r));
        if (r % 5 == 4)
        {
            std::vector<u8> mask(N);
            for (u64 i = 0; i < N; ++i)
                mask[i] = (i + static_cast<u64>(r)) % 2;
            upload(g_mask.data(), mask, g.s);
            upload(e_mask.data(), mask, e.s);
            host.reset(h.b, mask.data(), h.out.count, goals);
            reset_graph.launch(g.s);
            ee.reset(e.b, static_cast<const u8*>(e_mask.data()), e.out.count, goals);
            ee.refresh(e.b, e.out.count);
        }
        std::vector<i64> act(K * N, 0);
        std::vector<i32> next(K * N, 0);
        for (u32 k = 0; k < K; ++k)
        {
            const int t = r * static_cast<int>(K) + static_cast<int>(k);
            const std::vector<i64> a = actions(h.v.count, t);
            std::copy(a.begin(), a.end(), act.begin() + static_cast<std::ptrdiff_t>(u64{k} * N));
            for (u64 i = 0; i < N; ++i)
                next[u64{k} * N + i] = static_cast<i32>((i * 7 + static_cast<u64>(t)) % I);
            host.step(h.b, h.out, rl::Actions{k % 3 == 1 ? act.data() + u64{k} * N : nullptr},
                      k + 1 == K ? next.data() + u64{k} * N : nullptr);
        }
        upload(g_act.data(), act, g.s);
        upload(g_next.data(), next, g.s);
        upload(e_act.data(), act, e.s);
        upload(e_next.data(), next, e.s);
        const cudaStream_t rs = r % 3 == 2 ? other.get() : cs.get();
        if (rs != cs.get())
            cuda::stream_wait(rs, cs);
        step_graph.launch(rs);
        if (rs != cs.get())
            cuda::stream_wait(cs, rs);
        ee.set_launch(r % 2 ? cuda::BucketLaunch::PerBucket : cuda::BucketLaunch::Widest);
        ee.set_chunk_rows(r % 2 ? 7 : 0);
        steps(ee, e, e_act, e_next);
        ge.check_errors();
        ee.check_errors();
        expect_equal(g.snapshot(), h.v, r);
        expect_equal(e.snapshot(), h.v, r);
        if (::testing::Test::HasFatalFailure())
            break;
    }
    cuda::check(cudaStreamSynchronize(cs), "sync");
    cuda::check(cudaStreamSynchronize(other), "sync");
}
}  // namespace

// ------------------------------------------------------------------------------------------------ expand

TEST(DeviceTaskSuiteExpand, SuiteEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto all = suites(atoms);
        ASSERT_EQ(all.size(), 3u);
        for (const NamedSuite& ns : all)
        {
            const rl::TaskSuite& S = *ns.suite;
            cuda::Stream other;  // (a stream the expander ran on must outlive its scratch)
            cuda::SuiteExpander ex(ctx, ns.suite);
            for (const Order o : {Order::Interleaved, Order::Grouped, Order::Reversed, Order::LastOnly})
            {
                SCOPED_TRACE(ns.name + order_name(o));
                std::vector<u64> rows;
                std::vector<i32> ids;
                suite_rows(S, 6, o, rows, ids);
                const u32 W = S.words(), L = std::max<u32>(1, S.label_width()) + 1;  // a column of -1 padding more
                const u64 N = ids.size();
                const Flat full = host_expand(S, rows, ids, 1, L);  // counts (and interns first under lazy slots)
                const cudaStream_t s = ex.stream();
                const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s);
                const cuda::DeviceBuffer d_ids = to_device(ctx, ids, s);
                const rl::StateBatchView in{static_cast<const u64*>(d_rows.data()), N, W, 0};
                for (const u64 chunk : {u64{0}, u64{7}})
                    for (const auto launch : {cuda::BucketLaunch::Widest, cuda::BucketLaunch::PerBucket})
                        for (const u64 cap : {full.total, full.total / 2, u64{0}})
                        {
                            SCOPED_TRACE("chunk " + std::to_string(chunk) +
                                         (launch == cuda::BucketLaunch::Widest ? " widest" : " per bucket") +
                                         " capacity " + std::to_string(cap));
                            ex.set_chunk_rows(chunk);
                            ex.set_launch(launch);
                            const Flat h = host_expand(S, rows, ids, cap, L);
                            ASSERT_EQ(ex.count(in, static_cast<const i32*>(d_ids.data())), h.total);
                            expect_flat_equal(device_write(ctx, ex, N, W, cap, L), h);
                            if (::testing::Test::HasFatalFailure())
                                return;
                        }
                // a second write() of the counted batch equals the first; after a stream switch the batch is counted
                // again
                {
                    SCOPED_TRACE("rewrite");
                    const Flat h = host_expand(S, rows, ids, full.total, L);
                    expect_flat_equal(device_write(ctx, ex, N, W, full.total, L), h);
                    ex.set_stream(other.get());
                    EXPECT_THROW((void)device_write(ctx, ex, N, W, full.total, L), std::logic_error);
                    (void)ex.count(in, static_cast<const i32*>(d_ids.data()));
                    expect_flat_equal(device_write(ctx, ex, N, W, full.total, L), h);
                    ex.set_stream(nullptr);
                    ex.set_chunk_rows(0);
                    if (::testing::Test::HasFatalFailure())
                        return;
                }
                // the padded view on the device equals rl::pad's
                {
                    const Flat h = host_expand(S, rows, ids, full.total, L);
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
            other.synchronize();
        }
    }
}

/// Gathered domains write at their batch rows (DeviceExpander::write with a RowMap) in every mode: a Multi table's
/// successor kernel puts the rows there, single-instance domains (Single) and lazy tables (PerInstance) expand into
/// scratch and scatter; each domain's offsets come first (DeviceExpander::offsets). Suites with single-instance
/// domains, interleaved and reversed, against the host.
TEST(DeviceTaskSuiteExpand, MappedWritesOfEveryModeEqualHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        std::map<std::string, Set> by;
        for (Set& s : sets(atoms))
            by[s.name] = s;
        ASSERT_TRUE(by.count("gripper") && by.count("blocks") && by.count("miconic"));
        const rl::TaskSuitePtr suite = rl::TaskSuite::create({rl::TaskTable::single(by.at("gripper").tasks[3]),
                                                              by.at("blocks").table,
                                                              rl::TaskTable::single(by.at("miconic").tasks[4])});
        const rl::TaskSuite& S = *suite;
        cuda::SuiteExpander ex(ctx, suite);
        ASSERT_EQ(ex.domain(0).mode(), cuda::DeviceExpander::Mode::Single);
        ASSERT_EQ(ex.domain(1).mode(), atoms == TaskOptions::Atoms::Frozen ? cuda::DeviceExpander::Mode::Multi
                                                                             : cuda::DeviceExpander::Mode::PerInstance);
        for (const Order o : {Order::Interleaved, Order::Reversed})
        {
            SCOPED_TRACE(std::string(atoms_name(atoms)) + order_name(o));
            std::vector<u64> rows;
            std::vector<i32> ids;
            suite_rows(S, 7, o, rows, ids);
            const u32 W = S.words(), L = std::max<u32>(1, S.label_width()) + 2;
            const u64 N = ids.size();
            const Flat full = host_expand(S, rows, ids, 1, L);
            const cudaStream_t s = ex.stream();
            const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s);
            const cuda::DeviceBuffer d_ids = to_device(ctx, ids, s);
            const rl::StateBatchView in{static_cast<const u64*>(d_rows.data()), N, W, 0};
            for (const u64 chunk : {u64{0}, u64{5}})
                for (const u64 cap : {full.total, full.total / 3, u64{0}})
                {
                    SCOPED_TRACE("chunk " + std::to_string(chunk) + " capacity " + std::to_string(cap));
                    ex.set_chunk_rows(chunk);
                    const Flat h = host_expand(S, rows, ids, cap, L);
                    ASSERT_EQ(ex.count(in, static_cast<const i32*>(d_ids.data())), h.total);
                    expect_flat_equal(device_write(ctx, ex, N, W, cap, L), h);
                    if (::testing::Test::HasFatalFailure())
                        return;
                }
            ex.set_chunk_rows(0);
        }
    }
}

TEST(DeviceTaskSuiteExpand, TaskIdsAndRowsAreChecked)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = suites(TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(all.empty());
    const rl::TaskSuite& S = *all[0].suite;
    std::vector<u64> rows;
    std::vector<i32> ids;
    suite_rows(S, 2, Order::Interleaved, rows, ids);
    cuda::SuiteExpander ex(ctx, all[0].suite);
    const cudaStream_t s = ex.stream();
    const cuda::DeviceBuffer d_rows = to_device(ctx, rows, s);
    const rl::StateBatchView in{static_cast<const u64*>(d_rows.data()), ids.size(), S.words(), 0};
    EXPECT_THROW((void)ex.count(in, nullptr), std::invalid_argument);
    for (const i32 bad : {static_cast<i32>(S.size()), -1})
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
            EXPECT_NE(std::string(e.what()).find("outside the suite's"), std::string::npos) << e.what();
        }
    }
    // a row with bits past its instance's slots (a state of another domain's width)
    const cuda::DeviceBuffer d_ids = to_device(ctx, ids, s);
    std::vector<u64> wide = rows;
    u64 r = 0;
    while (S.task(static_cast<u32>(ids[r]))->atoms().fluent_slots() >= 64 * S.words())
        ++r;
    wide[r * S.words() + S.words() - 1] |= u64{1} << 63;
    const cuda::DeviceBuffer d_wide = to_device(ctx, wide, s);
    EXPECT_THROW((void)ex.count(rl::StateBatchView{static_cast<const u64*>(d_wide.data()), ids.size(), S.words(), 0},
                                static_cast<const i32*>(d_ids.data())),
                 std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ envs

TEST(DeviceTaskSuiteEnv, FastPathPerDomain)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const NamedSuite& ns : suites(atoms))
        {
            SCOPED_TRACE(ns.name);
            const std::string why = cuda::DeviceEnv::fast_unsupported(*ns.suite, config(1));
            EXPECT_EQ(why.empty(), ns.fast) << why;
            if (!ns.fast)
            {
                EXPECT_EQ(why.rfind("domain ", 0), 0u) << why;
                EXPECT_THROW(cuda::DeviceEnv(ctx, ns.suite, config(1), cuda::DeviceEnv::Path::Fast), std::invalid_argument);
                continue;
            }
            cuda::DeviceEnv env(ctx, ns.suite, config(1));
            EXPECT_TRUE(env.fast());
            EXPECT_EQ(env.suite(), ns.suite);
            EXPECT_EQ(env.words(), ns.suite->words());
            EXPECT_EQ(env.label_width(), ns.suite->label_width());
            EXPECT_GE(env.cache_schemas(), ns.suite->max_schemas() + 2);  // the counts, then the launch order
            EXPECT_TRUE(env.capture_unsupported().empty());
        }
}

TEST(DeviceTaskSuiteEnv, SuiteEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const NamedSuite& ns : suites(atoms))
            for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
                for (const bool grouped : {false, true})
                {
                    SCOPED_TRACE(ns.name + (path == cuda::DeviceEnv::Path::Auto ? " auto" : " general") +
                                 (grouped ? " grouped" : " interleaved"));
                    RunOptions r;
                    r.rearrange = true;
                    r.grouped = grouped;
                    run_env(ctx, ns.suite, config(0x5eed + ns.suite->size()), path, r);
                    if (::testing::Test::HasFatalFailure())
                        return;
                }
}

TEST(DeviceTaskSuiteEnv, GoalsDeadEndsChunksAndStreams)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const NamedSuite& ns : suites(TaskOptions::Atoms::Frozen))
    {
        if (!ns.fast)
            continue;  // goal axioms: per-env goal masks cannot express them
        for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
        {
            SCOPED_TRACE(ns.name + (path == cuda::DeviceEnv::Path::Auto ? " auto" : " general"));
            rl::EnvConfig cfg = config(77);
            RunOptions r;
            r.goals = true;
            r.switch_stream = true;
            run_env(ctx, ns.suite, cfg, path, r);
            if (::testing::Test::HasFatalFailure())
                return;
            cfg.dead_end = rl::DeadEnd::None;
            cfg.autoreset = false;
            r.goals = false;
            r.switch_stream = false;
            r.chunk_rows = 5;
            r.steps = 20;
            {
                // without autoresets no row changes its domain
                const rl::TaskSuite& S = *ns.suite;
                const std::vector<i32> ids = batch_ids(S, 4 * S.size() + 5, false);
                cuda::DeviceEnv env(ctx, ns.suite, cfg, path);
                rl::HostEnv host(ns.suite, cfg);
                env.set_chunk_rows(5);
                HostEnvs h(S, ids, false);
                DeviceEnvs d(ctx, env, ids, false);
                host.reset(h.b, nullptr, h.out.count);
                env.reset(d.b, nullptr, d.out.count);
                for (int t = 0; t < 20; ++t)
                {
                    host.step(h.b, h.out);
                    env.step(d.b, d.out);
                    env.check_errors();
                    expect_equal(d.snapshot(), h.v, t);
                    if (::testing::Test::HasFatalFailure())
                        return;
                }
            }
            cfg.dead_end = rl::DeadEnd::NoSuccessors;
            cfg.dead_end_terminal = false;
            cfg.autoreset = true;
            run_env(ctx, ns.suite, cfg, path, r);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}

namespace
{
/// The fast path's cached launch order of a batch over a suite (the count cache's last two columns: the row at each
/// position, the position of each row) is a permutation of the rows with its inverse, grouped by domain, each
/// instance's rows one run in batch order.
void expect_launch_order(const rl::TaskSuite& S, const DeviceEnvs& d)
{
    const u64 N = d.N, C = d.C, col = C - 2;
    const std::vector<u32> counts = to_host<u32>(d.counts.data(), N * C, d.s);
    const std::vector<i32> ids = to_host<i32>(d.task_ids.data(), N, d.s);
    std::vector<u8> placed(N, 0), closed(S.size(), 0);
    u32 domain = 0, inst = ~u32{0}, prev = 0;
    for (u64 j = 0; j < N; ++j)
    {
        const u32 r = counts[j * C + col];
        ASSERT_LT(r, N) << "position " << j;
        ASSERT_FALSE(placed[r]) << "row " << r << " at two positions";
        placed[r] = 1;
        ASSERT_EQ(counts[u64{r} * C + col + 1], j) << "row " << r;
        const auto g = static_cast<u32>(ids[r]);
        ASSERT_GE(S.domain_of(g), domain) << "position " << j << ": the rows are not grouped by domain";
        domain = S.domain_of(g);
        if (g != inst)
        {
            ASSERT_FALSE(closed[g]) << "position " << j << ": instance " << g << "'s rows are not one run";
            if (inst != ~u32{0})
                closed[inst] = 1;
            inst = g;
        }
        else
            ASSERT_GT(r, prev) << "position " << j << ": instance " << g << "'s rows out of batch order";
        prev = r;
    }
}
}  // namespace

/// lifted::launch_check_order_keys (the debug builds' check after every launch order) passes launch_order's orders
/// and finds orders that are not a stable sort by key: two rows of one key swapped, rows of two keys swapped, a stale
/// inverse.
TEST(DeviceTaskSuiteLaunchOrder, CheckFindsOrdersThatAreNotStableSorts)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cuda::Stream st;
    const cudaStream_t s = st.get();
    namespace lifted = cuda::lifted;
    const u32 I = 7, K = 4;
    const std::vector<u32> key{3, 0, 2, 3, 1, 0, 9};  // instance 6's key past the count: sorts as K - 1
    for (const u64 n : {u64{300}, u64{5000}})
    {
        SCOPED_TRACE("n " + std::to_string(n));
        std::vector<u32> inst(n);
        for (u64 i = 0; i < n; ++i)  // every 11th row's task id outside [0, I): key 0
            inst[i] = i % 11 == 4 ? I + 2 : static_cast<u32>((i * 3 + i / 5) % I);
        const auto key_of = [&](u32 r) { return inst[r] < I ? std::min(key[inst[r]], K - 1) : 0u; };
        const cuda::DeviceBuffer dinst = to_device(ctx, inst, s), dkey = to_device(ctx, key, s);
        const lifted::OrderKeys keys{static_cast<const u32*>(dkey.data()), K};
        const u64 tb = lifted::order_temp_bytes(n);
        cuda::DeviceBuffer temp(ctx, tb, s), order(ctx, n * 4, s), pos(ctx, n * 4, s), bad(ctx, 4, s);
        const auto* di = static_cast<const u32*>(dinst.data());
        ASSERT_EQ(lifted::launch_order(di, I, keys, n, temp.data(), tb, static_cast<u32*>(order.data()),
                                       static_cast<u32*>(pos.data()), 1, s),
                  cudaSuccess);
        std::vector<u32> o = to_host<u32>(order.data(), n, s), p = to_host<u32>(pos.data(), n, s);
        std::vector<u32> want(n);
        std::iota(want.begin(), want.end(), 0u);
        std::stable_sort(want.begin(), want.end(), [&](u32 a, u32 b) { return key_of(a) < key_of(b); });
        ASSERT_EQ(o, want);
        const auto checked = [&](const std::vector<u32>& ord, const std::vector<u32>& ps)
        {
            upload(order.data(), ord, s);
            upload(pos.data(), ps, s);
            cuda::check(cudaMemsetAsync(bad.data(), 0, 4, s), "memset");
            cuda::check(lifted::launch_check_order_keys(di, I, keys, n, static_cast<const u32*>(order.data()),
                                                        static_cast<const u32*>(pos.data()), 1,
                                                        static_cast<u32*>(bad.data()), s),
                        "launch_check_order_keys");
            return to_host<u32>(bad.data(), 1, s)[0];
        };
        EXPECT_EQ(checked(o, p), 0u);
        // two positions swapped (and the inverse with them): of one key (not stable), of two keys (not sorted)
        for (const bool same : {true, false})
        {
            u64 j = 0;
            while (j + 1 < n && (key_of(o[j]) == key_of(o[j + 1])) != same)
                ++j;
            ASSERT_LT(j + 1, n);
            std::vector<u32> o2 = o, p2 = p;
            std::swap(o2[j], o2[j + 1]);
            p2[o2[j]] = static_cast<u32>(j);
            p2[o2[j + 1]] = static_cast<u32>(j + 1);
            EXPECT_EQ(checked(o2, p2), 1u) << (same ? "one key" : "two keys");
        }
        std::vector<u32> p3 = p;  // a stale inverse
        p3[o[n / 2]] = static_cast<u32>(n / 2 + 1);
        EXPECT_EQ(checked(o, p3), 1u);
    }
}

TEST(DeviceTaskSuiteEnv, LaunchOrderGroupsTheRowsByDomain)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const NamedSuite& ns : suites(TaskOptions::Atoms::Frozen))
    {
        if (!ns.fast)
            continue;
        const rl::TaskSuite& S = *ns.suite;
        const u32 I = S.size();
        // one tile of the counting sort, and several (lifted::launch_order)
        for (const u64 n : {u64{67}, u64{1500}})
            for (const bool grouped : {false, true})
            {
                SCOPED_TRACE(ns.name + " " + std::to_string(n) + (grouped ? " grouped" : " interleaved"));
                const std::vector<i32> ids = batch_ids(S, n, grouped);
                cuda::DeviceEnv env(ctx, ns.suite, config(5), cuda::DeviceEnv::Path::Fast);
                DeviceEnvs d(ctx, env, ids, false);
                ASSERT_GE(d.C, S.max_schemas() + 2);  // the counts (and recorded keys), then the launch order's two
                env.reset(d.b, nullptr, d.out.count);
                {
                    SCOPED_TRACE("reset");
                    expect_launch_order(S, d);
                }
                env.refresh(d.b, d.out.count);
                {
                    SCOPED_TRACE("refresh");
                    expect_launch_order(S, d);
                }
                // autoresets (truncation after 9 steps) into next task ids of other domains regroup the rows
                std::vector<i32> next(n);
                for (u64 i = 0; i < n; ++i)
                    next[i] = static_cast<i32>((i * 7 + 3) % I);
                upload(d.next.data(), next, d.s);
                for (int t = 0; t < 10; ++t)
                    env.step(d.b, d.out, {}, static_cast<const i32*>(d.next.data()));
                env.check_errors();
                {
                    SCOPED_TRACE("steps with next_task_ids");
                    expect_launch_order(S, d);
                }
                if (::testing::Test::HasFatalFailure())
                    return;
            }
    }
}

TEST(DeviceTaskSuiteEnv, TaskIdsAreChecked)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = suites(TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(all.empty());
    const rl::TaskSuitePtr& suite = all[0].suite;
    const u32 I = suite->size();
    for (const auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
    {
        cuda::DeviceEnv env(ctx, suite, config(3), path);
        std::vector<i32> ids(16);
        for (u64 i = 0; i < ids.size(); ++i)
            ids[i] = static_cast<i32>(i % I);
        DeviceEnvs d(ctx, env, ids, false);
        env.reset(d.b);
        env.check_errors();
        std::vector<i32> next(ids.size(), static_cast<i32>(I));
        upload(d.next.data(), next, d.s);
        for (int t = 0; t < 12; ++t)
            env.step(d.b, d.out, {}, static_cast<const i32*>(d.next.data()));
        try
        {
            env.check_errors();
            ADD_FAILURE() << "a next task id outside the suite was not reported";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("outside the suite's"), std::string::npos) << e.what();
        }
        env.check_errors();
        ids[5] = -2;
        upload(d.task_ids.data(), ids, d.s);
        env.reset(d.b);
        EXPECT_THROW(env.check_errors(), std::invalid_argument);
    }
}

TEST(DeviceTaskSuiteEnv, CapturedStepsEqualEagerAndHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const NamedSuite& ns : suites(TaskOptions::Atoms::Frozen))
    {
        if (!ns.fast)
            continue;
        for (const bool grouped : {false, true})
        {
            SCOPED_TRACE(ns.name + (grouped ? " grouped" : " interleaved"));
            run_graph(ctx, ns.suite, config(0x9a + ns.suite->size()), false, 3, grouped);
            if (::testing::Test::HasFatalFailure())
                return;
            rl::EnvConfig cfg = config(31);
            cfg.dead_end = rl::DeadEnd::NoSuccessors;
            cfg.dead_end_terminal = false;
            run_graph(ctx, ns.suite, cfg, true, 4, grouped);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}
