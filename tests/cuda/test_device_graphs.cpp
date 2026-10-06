// Device env CUDA-graph tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device): the device env's calls
// captured into CUDA graphs (stream capture in global mode, so a call that synchronizes, queries or allocates fails
// the capture) and replayed equal the same calls run eagerly and rl::HostEnv, byte for byte, on the table instance
// sets (frozen: the fast
// path over several instances, the launch groups forked onto the env's auxiliary streams inside the capture) and on
// tables of one of their instances:
//   - graphs of K steps (random steps, given actions from a device buffer the test rewrites between replays,
//     autoresets into next_task_ids) replayed many times, on the stream they were captured on and on another one;
//     per-env goals; both dead-end modes; tiny chunks; both bucket launches; the eager env on another launch and chunk;
//   - reset (a mask from device memory) and refresh in a graph;
//   - set_seed() after the capture reaches the replays (the key lives in device memory);
//   - a captured call whose scratch the env has not sized throws (reserve(), or an eager call of that size first), a
//     captured call of the general path throws, and so does check_errors() under capture; the env works on after.

#include "env_harness.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
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
    [[nodiscard]] usize nodes() const
    {
        usize n = 0;
        cuda::check(cudaGraphGetNodes(m_graph, nullptr, &n), "cudaGraphGetNodes");
        return n;
    }

private:
    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_exec = nullptr;
};

/// Captures `record` on s and expects it to throw std::logic_error naming `what`.
void expect_capture_throws(cudaStream_t s, const std::function<void()>& record, const std::string& what)
{
    try
    {
        const Graph g(s, record);
        ADD_FAILURE() << "the capture did not throw";
    }
    catch (const std::logic_error& e)
    {
        EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    }
}

std::vector<i32> mixed_ids(u32 I, u64 N)
{
    std::vector<i32> ids(N);
    for (u64 k = 0; k < N; ++k)
        ids[k] = static_cast<i32>((k * 5 + k / I) % I);
    return ids;
}

struct GraphOptions
{
    bool goals = false;
    u64 chunk_rows = 0;  // of the graph's env (the eager env runs automatic chunks)
    cuda::BucketLaunch launch = cuda::BucketLaunch::Widest;
    u32 K = 3;        // steps per graph
    int replays = 12;
};

/// Steps a batch three ways, K steps at a time: a captured graph of K steps replayed, the same steps on another env
/// eagerly, and rl::HostEnv; compares all three after every replay.
void run_graph(const cuda::ContextPtr& ctx, const rl::TaskTablePtr& table, const rl::EnvConfig& cfg,
               const GraphOptions& o)
{
    const rl::TaskTable& T = *table;
    const u32 I = T.size(), W = T.words(), K = o.K;
    const std::vector<i32> ids = mixed_ids(I, std::max<u64>(6 * I + 3, 67));
    const u64 N = ids.size();
    const cuda::Stream cs, other;  // the capture stream, and a second replay stream (they outlive the envs)
    cuda::DeviceEnv ge(ctx, table, cfg, cuda::DeviceEnv::Path::Fast), ee(ctx, table, cfg, cuda::DeviceEnv::Path::Fast);
    rl::HostEnv host(table, cfg);
    EXPECT_TRUE(ge.capture_unsupported().empty()) << ge.capture_unsupported();
    ge.set_stream(cs);
    ge.set_chunk_rows(o.chunk_rows);
    ge.set_launch(o.launch);
    HostEnvs h(T, ids, o.goals);
    DeviceEnvs g(ctx, ge, ids, o.goals), e(ctx, ee, ids, o.goals);
    // the per-step actions and next task ids of a graph: [K, N] each
    cuda::DeviceBuffer g_act(ctx, K * N * 8, g.s), g_next(ctx, K * N * 4, g.s), e_act(ctx, K * N * 8, e.s),
        e_next(ctx, K * N * 4, e.s), g_mask(ctx, N, g.s), e_mask(ctx, N, e.s);
    host.reset(h.b, nullptr, h.out.count);
    ge.reset(g.b, nullptr, g.out.count);
    ee.reset(e.b, nullptr, e.out.count);
    if (o.goals)
    {
        custom_goals(h.v, W);
        upload(g.gpos.data(), h.v.gpos, g.s);
        upload(g.gneg.data(), h.v.gneg, g.s);
        upload(e.gpos.data(), h.v.gpos, e.s);
        upload(e.gneg.data(), h.v.gneg, e.s);
    }
    ge.reserve(N);
    // step k of a graph: given actions when k % 3 == 1, else the random policy; autoresets into next_task_ids at the
    // last step
    const auto steps = [&](cuda::DeviceEnv& env, DeviceEnvs& d, const cuda::DeviceBuffer& act,
                           const cuda::DeviceBuffer& next) {
        for (u32 k = 0; k < K; ++k)
            env.step(d.b, d.out,
                     rl::Actions{k % 3 == 1 ? static_cast<const i64*>(act.data()) + u64{k} * N : nullptr},
                     k + 1 == K ? static_cast<const i32*>(next.data()) + u64{k} * N : nullptr);
    };
    const Graph step_graph(cs, [&] { steps(ge, g, g_act, g_next); });
    const Graph reset_graph(cs, [&] {
        ge.reset(g.b, static_cast<const u8*>(g_mask.data()), g.out.count, o.goals);
        ge.refresh(g.b, g.out.count);
    });
    EXPECT_GT(step_graph.nodes(), K);
    for (int r = 0; r < o.replays; ++r)
    {
        SCOPED_TRACE("replay " + std::to_string(r));
        if (r == o.replays / 2)
        {
            // a new key: the graph reads it from device memory
            ge.set_seed(cfg.seed + 101);
            ee.set_seed(cfg.seed + 101);
            host.set_seed(cfg.seed + 101);
        }
        if (r % 5 == 4)
        {
            // reset every other row (the rows' own instances; goals kept): a graph of reset and refresh
            std::vector<u8> mask(N);
            for (u64 i = 0; i < N; ++i)
                mask[i] = (i + static_cast<u64>(r)) % 2;
            upload(g_mask.data(), mask, g.s);
            upload(e_mask.data(), mask, e.s);
            host.reset(h.b, mask.data(), h.out.count, o.goals);
            reset_graph.launch(g.s);
            ee.reset(e.b, static_cast<const u8*>(e_mask.data()), e.out.count, o.goals);
            ee.refresh(e.b, e.out.count);
        }
        // the host's K steps first: they give the actions (from the counts before each step) and next task ids
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
        // replays on the capture stream, and every third one on another stream
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
        const Snapshot gs = g.snapshot(), es = e.snapshot();
        expect_equal(gs, h.v, r);
        expect_equal(es, h.v, r);
        if (::testing::Test::HasFatalFailure())
            break;
    }
    cuda::check(cudaStreamSynchronize(cs), "sync");
    cuda::check(cudaStreamSynchronize(other), "sync");
}

/// The sets as tables (frozen slots: the fast path), and tables of one of their smallest and largest instances.
std::vector<Set> graph_sets()
{
    std::vector<Set> out;
    for (Set& s : sets(TaskOptions::Atoms::Frozen))
    {
        if (!cuda::DeviceEnv::fast_unsupported(*s.table, config(0)).empty())
            continue;  // miconic-simpleadl: axioms
        for (const usize i : {usize{0}, s.tasks.size() - 1})
            out.push_back({s.name + " #" + std::to_string(i), {s.tasks[i]}, rl::TaskTable::create({s.tasks[i]})});
        out.push_back(std::move(s));
    }
    return out;
}
}  // namespace

TEST(DeviceGraphEnv, CapturedStepsEqualEagerAndHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = graph_sets();
    EXPECT_EQ(all.size(), 12u);  // blocks, gripper, miconic, logistics: the table and two of its instances
    for (const Set& set : all)
    {
        SCOPED_TRACE(set.name);
        GraphOptions o;
        run_graph(ctx, set.table, config(0x9a + set.table->size()), o);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST(DeviceGraphEnv, CapturedGoalsDeadEndsChunksAndLaunches)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const Set& set : graph_sets())
    {
        SCOPED_TRACE(set.name);
        rl::EnvConfig cfg = config(31);
        GraphOptions o;
        o.goals = true;
        o.chunk_rows = 5;
        o.launch = cuda::BucketLaunch::PerBucket;
        o.K = 4;
        run_graph(ctx, set.table, cfg, o);
        if (::testing::Test::HasFatalFailure())
            return;
        cfg.dead_end = rl::DeadEnd::None;
        cfg.autoreset = false;
        o.goals = false;
        o.K = 1;
        run_graph(ctx, set.table, cfg, o);
        if (::testing::Test::HasFatalFailure())
            return;
        cfg.dead_end = rl::DeadEnd::NoSuccessors;
        cfg.dead_end_terminal = false;
        cfg.autoreset = true;
        o.K = 5;
        o.launch = cuda::BucketLaunch::Widest;
        run_graph(ctx, set.table, cfg, o);
        if (::testing::Test::HasFatalFailure())
            return;
    }
}

TEST(DeviceGraphEnv, CaptureNeedsSizedScratch)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = graph_sets();
    ASSERT_FALSE(all.empty());
    for (const Set& set : all)
    {
        SCOPED_TRACE(set.name);
        const u32 I = set.table->size();
        const cuda::Stream cs;
        cuda::DeviceEnv env(ctx, set.table, config(5), cuda::DeviceEnv::Path::Fast);
        env.set_stream(cs);
        const std::vector<i32> ids = mixed_ids(I, 40);
        DeviceEnvs d(ctx, env, ids, false);
        env.reserve(20);
        // the reset of a batch of I rows or more needs the launch order's scratch (several instances)
        env.reset(d.b, nullptr, d.out.count);
        expect_capture_throws(cs, [&] { env.step(d.b, d.out); }, "reserve(rows)");
        expect_capture_throws(cs, [&] { env.check_errors(); }, "cannot be captured");
        // the env works on: an eager step of 40 rows sizes the scratch, a captured one then runs
        env.step(d.b, d.out);
        env.check_errors();
        const Graph g(cs, [&] { env.step(d.b, d.out); });
        g.launch(cs);
        env.check_errors();
        // growth after a capture keeps the captured buffers alive (a replay after a larger eager call)
        const std::vector<i32> more = mixed_ids(I, 90);
        DeviceEnvs d2(ctx, env, more, false);
        env.reset(d2.b, nullptr, d2.out.count);
        env.step(d2.b, d2.out);
        g.launch(cs);
        env.check_errors();
        cuda::check(cudaStreamSynchronize(cs), "sync");
    }
}

TEST(DeviceGraphEnv, GeneralPathRefusesCapture)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto all = sets(TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(all.empty());
    const cuda::Stream cs;
    cuda::DeviceEnv env(ctx, all[0].table, config(5), cuda::DeviceEnv::Path::General);
    EXPECT_NE(env.capture_unsupported().find("general path"), std::string::npos) << env.capture_unsupported();
    env.set_stream(cs);
    const std::vector<i32> ids = mixed_ids(all[0].table->size(), 24);
    DeviceEnvs d(ctx, env, ids, false);
    env.reset(d.b, nullptr, d.out.count);
    env.step(d.b, d.out);
    expect_capture_throws(cs, [&] { env.step(d.b, d.out); }, "cannot be captured");
    expect_capture_throws(cs, [&] { env.refresh(d.b, d.out.count); }, "cannot be captured");
    env.step(d.b, d.out);
    env.check_errors();
    cuda::check(cudaStreamSynchronize(cs), "sync");
}
