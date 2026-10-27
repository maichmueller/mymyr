// Device RL-helper-kernel tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the device kernels of the RL helpers (cuda/rl_ops.hpp) equal the host loops (rl/novelty.hpp, rl/prefix.hpp,
//     rl/her.hpp) byte for byte on suite windows: novelty rewards and tables, prefix and schema masks, hindsight
//     relabels (every strategy, goal subsets, eligible-atom masks, env ids, the states the steps started from);
//   - the random policy's actions without a step (envk::launch_random_actions) equal rl::rng::successor_index;
//   - the count cache's recorded keys (lifted::CountCache regions): sokoban p14 records its deep sorted schemas, its
//     device steps still equal the host's, and a state written without refresh() is reported by check_errors().

#include "../cpp/support/suite.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/cuda/rl_ops.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/her.hpp"
#include "mymyr/rl/goal_templates.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"
#include "mymyr/successor/bindings.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
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
    o.max_bytes = u64{1} << 30;  // the GPU is shared
    return cuda::DeviceContext::create(0, o);
}

TaskPtr load(const std::string& name)
{
    TaskOptions o;
    o.atoms = TaskOptions::Atoms::Dense;
    return Task::from_text_file(task_path(name), o);
}

/// A device copy of a host vector (at least one element, so empty inputs still have an address).
template<class T>
cuda::DeviceBuffer upload(const cuda::ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    cuda::DeviceBuffer d(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return d;
}

template<class T>
std::vector<T> download(const cuda::DeviceBuffer& d, u64 n, cudaStream_t s)
{
    std::vector<T> out(n);
    if (n)
        cuda::check(cudaMemcpyAsync(out.data(), d.data(), n * sizeof(T), cudaMemcpyDeviceToHost, s), "D2H");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return out;
}

template<class T>
T* ptr(cuda::DeviceBuffer& d)
{
    return static_cast<T*>(d.data());
}

/// T steps of N random-policy environments (max_steps 6, autoreset): reached states [T, N, RW] and done [T, N].
struct Window
{
    u32 RW = 0, W = 0;
    std::vector<u64> states;
    std::vector<u64> prev;  // [T, N, RW]: the states the steps started from
    std::vector<u8> done;
};

Window rollout(const TaskPtr& task, u64 N, u64 T, u64 seed)
{
    rl::EnvConfig cfg;
    cfg.seed = seed;
    cfg.max_steps = 6;
    rl::HostEnv env(rl::TaskTable::single(task), cfg);
    Window w;
    w.W = env.words();
    w.RW = env.words() + task->numeric_words();
    std::vector<u64> states(N * w.RW), finals(N * w.RW), draws(N, 0);
    std::vector<i32> steps(N, 0);
    std::vector<u8> term(N), trunc(N);
    rl::EnvBatch b;
    b.states = states.data();
    b.rows = N;
    b.words = env.words();
    b.numeric_words = task->numeric_words();
    b.steps = steps.data();
    b.draws = draws.data();
    env.reset(b);
    rl::StepOutputs o;
    o.final_states = finals.data();
    o.terminated = term.data();
    o.truncated = trunc.data();
    for (u64 t = 0; t < T; ++t)
    {
        w.prev.insert(w.prev.end(), states.begin(), states.end());
        env.step(b, o);
        w.states.insert(w.states.end(), finals.begin(), finals.end());
        for (u64 i = 0; i < N; ++i)
            w.done.push_back(term[i] || trunc[i] ? 1 : 0);
    }
    return w;
}

const std::vector<std::string>& op_tasks()
{
    static const std::vector<std::string> t = {"blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1",
                                               "sokoban-opt08-strips__p14", "freecell__p02", "openstacks-opt08-adl__p03"};
    return t;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ RL helpers

TEST(DeviceRlOps, NoveltyDeviceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    for (const std::string& name : op_tasks())
    {
        const TaskPtr task = load(name);
        const u64 N = 96, T = 30;
        const Window w = rollout(task, N, T, 4);
        std::vector<u64> seen(N * w.W, 0);
        cuda::DeviceBuffer dseen = upload(ctx, seen, s);
        cuda::DeviceBuffer dstates = upload(ctx, w.states, s);
        cuda::DeviceBuffer dr(ctx, N * sizeof(i32), s);
        std::vector<i32> r(N);
        for (u64 t = 0; t < T; ++t)
        {
            const u64* rows = w.states.data() + t * N * w.RW;
            rl::novelty_update(seen.data(), rows, N, w.W, w.RW, r.data());
            cuda::check(cuda::launch_novelty_update(ptr<u64>(dseen), ptr<const u64>(dstates) + t * N * w.RW, N, w.W, w.RW,
                                                    ptr<i32>(dr), s),
                        "novelty");
            ASSERT_EQ(download<i32>(dr, N, s), r) << name << " " << t;
            ASSERT_EQ(download<u64>(dseen, N * w.W, s), seen) << name << " " << t;
        }
    }
}

TEST(DeviceRlOps, PrefixAndSchemaMasksDeviceEqualHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    for (const std::string& name : op_tasks())
    {
        const TaskPtr task = load(name);
        const u64 N = 64, T = 7;
        const Window w = rollout(task, N, T, 9);
        const u64* rows = w.states.data() + (T - 1) * N * w.RW;
        const u32 L = std::max<u32>(1, rl::max_label_width(*task));
        rl::Expansion e;
        e.words = w.W;
        e.numeric_words = task->numeric_words();
        e.label_width = L;
        std::vector<i32> offsets(N + 1);
        e.offsets = offsets.data();
        rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{rows, N, w.W, 0, e.numeric_words}, nullptr, e);
        const u64 M = e.total;
        std::vector<u64> succ(std::max<u64>(M, 1) * w.RW);
        std::vector<i32> parent(M), schema(M), binding(M * L);
        e.capacity = M;
        e.succ = succ.data();
        e.parent = parent.data();
        e.schema = schema.data();
        e.binding = binding.data();
        rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{rows, N, w.W, 0, e.numeric_words}, nullptr, e);
        // padding rows (parent -1) are skipped by both
        parent.push_back(-1);
        schema.push_back(0);
        binding.insert(binding.end(), L, 0);
        const rl::LabelRows host{M + 1, parent.data(), schema.data(), binding.data(), L};
        cuda::DeviceBuffer dparent = upload(ctx, parent, s), dschema = upload(ctx, schema, s), dbinding = upload(ctx, binding, s);
        const rl::LabelRows dev{M + 1, ptr<const i32>(dparent), ptr<const i32>(dschema), ptr<const i32>(dbinding), L};
        const u32 n = task->num_objects(), S = task->num_schemas();
        // schema masks
        std::vector<u8> hm(N * S);
        rl::schema_masks(host, N, S, hm.data());
        cuda::DeviceBuffer dm(ctx, std::max<u64>(N * std::max(n, S), 1), s);
        cuda::check(cuda::launch_schema_masks(dev, N, S, ptr<u8>(dm), s), "schema masks");
        ASSERT_EQ(download<u8>(dm, N * S, s), hm) << name;
        // prefix masks of each state's (i mod count)-th label, some spoiled
        std::vector<i32> qs(N, 0), prefix(N * L, -1);
        for (u64 i = 0; i < N; ++i)
            if (const i32 c = offsets[i + 1] - offsets[i]; c > 0)
            {
                const u64 j = static_cast<u64>(offsets[i] + static_cast<i32>(i) % c);
                qs[i] = schema[j];
                std::copy_n(binding.begin() + static_cast<std::ptrdiff_t>(j * L), L, prefix.begin() + static_cast<std::ptrdiff_t>(i * L));
                if (i % 7 == 3)
                    prefix[i * L] = (prefix[i * L] + 1) % static_cast<i32>(n);
            }
        cuda::DeviceBuffer dqs = upload(ctx, qs, s), dprefix = upload(ctx, prefix, s);
        for (u32 depth = 0; depth < L; ++depth)
        {
            std::vector<u8> m(N * n);
            rl::prefix_masks(host, rl::PrefixQuery{N, qs.data(), prefix.data(), L, depth, n}, m.data());
            cuda::check(cuda::launch_prefix_masks(dev, rl::PrefixQuery{N, ptr<const i32>(dqs), ptr<const i32>(dprefix), L, depth, n},
                                                  ptr<u8>(dm), s),
                        "prefix masks");
            ASSERT_EQ(download<u8>(dm, N * n, s), m) << name << " depth " << depth;
        }
    }
}

TEST(DeviceRlOps, HerDeviceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    for (const std::string& name : {std::string("gripper__prob05"), std::string("sokoban-opt08-strips__p14")})
    {
        const TaskPtr task = load(name);
        const u64 N = 40, T = 32;
        const Window w = rollout(task, N, T, 6);
        std::vector<u64> atoms(w.W);
        for (u32 k = 0; k < w.W; ++k)
            atoms[k] = hash::mix64(k + 17);
        std::vector<u32> ids(N);
        for (u64 i = 0; i < N; ++i)
            ids[i] = static_cast<u32>(1000 + 3 * i);
        cuda::DeviceBuffer dstates = upload(ctx, w.states, s), ddone = upload(ctx, w.done, s), datoms = upload(ctx, atoms, s),
                           dids = upload(ctx, ids, s), dprev = upload(ctx, w.prev, s);
        for (const rl::HerStrategy st : {rl::HerStrategy::Future, rl::HerStrategy::Final, rl::HerStrategy::Episode})
            for (const u32 subset : {0u, 1u, 4u})
                for (const bool extras : {false, true})
                {
                    rl::HerConfig c;
                    c.strategy = st;
                    c.k = 3;
                    c.subset = subset;
                    c.seed = 0x1234'5678'9ull + subset;
                    c.first_env = 5;
                    c.goal_reward = 2.0f;
                    const u64 R = T * N * c.k;
                    std::vector<u64> goal(R * w.W);
                    std::vector<i32> source(R);
                    std::vector<u8> achieved(R);
                    std::vector<f32> reward(R);
                    std::vector<u8> valid(R);
                    rl::HerBatch hb{T, N, w.W, w.RW, w.states.data(), w.done.data(), extras ? w.prev.data() : nullptr,
                                    extras ? atoms.data() : nullptr, 0u, extras ? ids.data() : nullptr, goal.data(),
                                    source.data(), achieved.data(), reward.data(), valid.data()};
                    rl::her_relabel(hb, c);
                    cuda::DeviceBuffer dgoal(ctx, R * w.W * 8, s), dsource(ctx, R * 4, s), dach(ctx, R, s), drew(ctx, R * 4, s),
                        dvalid(ctx, R, s);
                    rl::HerBatch db{T, N, w.W, w.RW, ptr<const u64>(dstates), ptr<const u8>(ddone),
                                    extras ? ptr<const u64>(dprev) : nullptr, extras ? ptr<const u64>(datoms) : nullptr, 0u,
                                    extras ? ptr<const u32>(dids) : nullptr, ptr<u64>(dgoal), ptr<i32>(dsource),
                                    ptr<u8>(dach), ptr<f32>(drew), ptr<u8>(dvalid)};
                    cuda::check(cuda::launch_her_relabel(db, c, s), "her");
                    ASSERT_EQ(download<u64>(dgoal, R * w.W, s), goal) << name;
                    ASSERT_EQ(download<i32>(dsource, R, s), source) << name;
                    ASSERT_EQ(download<u8>(dach, R, s), achieved) << name;
                    ASSERT_EQ(download<u8>(dvalid, R, s), valid) << name;
                    const std::vector<f32> dr = download<f32>(drew, R, s);
                    ASSERT_EQ(std::memcmp(dr.data(), reward.data(), R * sizeof(f32)), 0) << name;
                }
    }
}

TEST(DeviceRlOps, RandomActionsEqualTheRandomPolicy)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    const u64 N = 5000, seed = 0xFEEDu, first_env = 77;
    std::vector<u64> draws(N);
    std::vector<i32> count(N);
    for (u64 i = 0; i < N; ++i)
    {
        draws[i] = hash::mix64(i) % 1000;
        count[i] = static_cast<i32>(hash::mix64(i + N) % 40) - 3;  // some without successors
    }
    cuda::DeviceBuffer dd = upload(ctx, draws, s), dc = upload(ctx, count, s), dout(ctx, N * sizeof(i64), s);
    for (const u32 limit : {~u32{0}, 1u, 7u})  // ~u32{0}: no limit
    {
        cuda::check(cuda::envk::launch_random_actions(ptr<const u64>(dd), ptr<const i32>(dc), N, seed, nullptr,
                                                      first_env, limit, ptr<i64>(dout), s),
                    "random actions");
        const std::vector<i64> got = download<i64>(dout, N, s);
        for (u64 i = 0; i < N; ++i)
        {
            u32 c = count[i] > 0 ? static_cast<u32>(count[i]) : 0u;
            c = std::min(c, limit);
            const i64 want = c ? i64{rl::rng::successor_index(seed, first_env + i, draws[i], c)} : i64{0};
            ASSERT_EQ(got[i], want) << i << " limit " << limit;
        }
    }
}

// ------------------------------------------------------------------------------------------------ count cache keys

TEST(DeviceCountCache, SokobanRecordsKeysAndStaysEqualToTheHost)
{
    SKIP_WITHOUT_GPU();
    const TaskPtr task = load("sokoban-opt08-strips__p14");
    auto ctx = context();
    rl::EnvConfig cfg;
    cfg.seed = 21;
    cfg.max_steps = 50;
    cfg.goal_reward = 1.0f;
    cuda::DeviceEnv env(ctx, rl::TaskTable::single(task), cfg);
    ASSERT_TRUE(env.fast());
    EXPECT_GT(env.cache_regions(), 0u);  // push and move: sorted segments of deep matchers
    const cudaStream_t s = env.stream();
    rl::HostEnv host(rl::TaskTable::single(task), cfg);
    const u64 N = 512;
    const u32 W = env.words(), L = env.label_width(), C = env.cache_schemas();
    const u64 V = env.cache_view_words();
    std::vector<u64> hs(N * W), hd(N, 0), hf(N * W);
    std::vector<i32> hsteps(N, 0), hcount(N), hschema(N), hbinding(N * L);
    rl::EnvBatch hb;
    hb.states = hs.data();
    hb.rows = N;
    hb.words = W;
    hb.steps = hsteps.data();
    hb.draws = hd.data();
    rl::StepOutputs ho{nullptr, nullptr, nullptr, hcount.data(), hf.data(), hschema.data(), hbinding.data(), L, nullptr, nullptr};
    host.reset(hb);
    cuda::DeviceBuffer ds(ctx, N * W * 8, s), dd(ctx, N * 8, s), dsteps(ctx, N * 4, s), dcounts(ctx, N * C * 4, s),
        dviews(ctx, std::max<u64>(N * V, 1) * 8, s), dcount(ctx, N * 4, s), df(ctx, N * W * 8, s), dschema(ctx, N * 4, s),
        dbinding(ctx, N * L * 4, s), daction(ctx, N * 8, s), dat_goal(ctx, N, s);
    cuda::check(cudaMemsetAsync(dd.data(), 0, N * 8, s), "memset");
    rl::EnvBatch db;
    db.states = ptr<u64>(ds);
    db.rows = N;
    db.words = W;
    db.steps = ptr<i32>(dsteps);
    db.draws = ptr<u64>(dd);
    db.counts = ptr<u32>(dcounts);
    db.views = ptr<u64>(dviews);
    db.at_goal = ptr<u8>(dat_goal);
    rl::StepOutputs dout{nullptr, nullptr, nullptr, ptr<i32>(dcount), ptr<u64>(df), ptr<i32>(dschema), ptr<i32>(dbinding), L,
                         nullptr, nullptr};
    env.reset(db, nullptr, ptr<i32>(dcount));
    for (int t = 0; t < 80; ++t)
    {
        if (t % 4 == 3)
        {
            // given actions: the last successor of each state (the end of the canonical order)
            std::vector<i64> a(N);
            for (u64 i = 0; i < N; ++i)
                a[i] = std::max(hcount[i], 1) - 1;
            cuda::DeviceBuffer da = upload(ctx, a, s);
            host.step(hb, ho, rl::Actions{a.data()});
            env.step(db, dout, rl::Actions{ptr<const i64>(da)});
            cuda::check(cudaStreamSynchronize(s), "sync");
        }
        else
        {
            host.step(hb, ho);
            env.step(db, dout);
        }
        ASSERT_EQ(download<u64>(ds, N * W, s), hs) << t;
        ASSERT_EQ(download<i32>(dschema, N, s), hschema) << t;
        ASSERT_EQ(download<i32>(dbinding, N * L, s), hbinding) << t;
        ASSERT_EQ(download<i32>(dcount, N, s), hcount) << t;
    }
    EXPECT_NO_THROW(env.check_errors());
    // states written without refresh(): their cached keys belong to other states, which the picks detect
    std::vector<u64> other = hs;
    std::rotate(other.begin(), other.begin() + W, other.end());
    cuda::check(cudaMemcpyAsync(ds.data(), other.data(), other.size() * 8, cudaMemcpyHostToDevice, s), "H2D");
    for (int t = 0; t < 4; ++t)
        env.step(db, dout);
    EXPECT_THROW(env.check_errors(), std::logic_error);
    // refresh() makes the batch consistent again
    env.refresh(db, ptr<i32>(dcount));
    env.step(db, dout);
    EXPECT_NO_THROW(env.check_errors());
}

TEST(DeviceRlOps, MatchFirstDeviceEqualsHostAndCapturedReplay)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto stream = ctx->stream();
    for (const auto& name : std::vector<std::string>{"blocks__probBLOCKS-8-0", "gripper__prob05"})
    {
        const auto task = load(name);
        const auto suite = rl::TaskSuite::of(rl::TaskTable::single(task));
        const auto templates = rl::lifted_goal_templates(suite);
        const auto window = rollout(task, 5, 9, 13);
        const u64 rows = 45;
        std::vector<std::byte> packed(templates.arrays().bytes());
        std::memcpy(packed.data(), templates.arrays().block(), packed.size());
        auto block = upload(ctx, packed, stream);
        auto states = upload(ctx, window.states, stream);
        auto ids = upload(ctx, std::vector<i32>(rows, 0), stream);
        for (bool first : {false, true})
        {
            const u32 capacity = first ? 1 : templates.capacity();
            std::vector<u8> found(rows * capacity);
            std::vector<i32> binding(rows * capacity * templates.max_vars()), which(rows * capacity), count(rows);
            u64 status = 99;
            rl::MatchBatch host;
            host.states = window.states.data();
            host.rows = rows;
            host.row_words = window.RW;
            host.capacity = capacity;
            host.first = first;
            host.found = found.data();
            host.binding = binding.data();
            host.template_id = which.data();
            host.count = count.data();
            host.status = &status;
            rl::match_first(templates.view(), host);
            ASSERT_EQ(status, 0);
            auto df = upload(ctx, found, stream), db = upload(ctx, binding, stream);
            auto dt = upload(ctx, which, stream), dc = upload(ctx, count, stream);
            auto ds = upload(ctx, std::vector<u64>{99}, stream);
            auto device = host;
            device.states = ptr<const u64>(states);
            device.task_ids = ptr<const i32>(ids);
            device.found = ptr<u8>(df);
            device.binding = ptr<i32>(db);
            device.template_id = ptr<i32>(dt);
            device.count = ptr<i32>(dc);
            device.status = ptr<u64>(ds);
            const auto view = templates.view(static_cast<const std::byte*>(block.data()));
            cuda::check(cuda::launch_match_first(view, device, stream), "match_first warm-up");
            cuda::check(cudaStreamSynchronize(stream), "sync");
            cudaGraph_t graph{};
            cudaGraphExec_t executable{};
            if (cuda::GraphExec::enabled())
            {
                cuda::check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
                cuda::check(cuda::launch_match_first(view, device, stream), "match_first capture");
                cuda::check(cudaStreamEndCapture(stream, &graph), "end capture");
                cuda::check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
                cuda::check(cudaGraphLaunch(executable, stream), "replay");
            }
            EXPECT_EQ(download<u8>(df, found.size(), stream), found);
            EXPECT_EQ(download<i32>(db, binding.size(), stream), binding);
            EXPECT_EQ(download<i32>(dt, which.size(), stream), which);
            EXPECT_EQ(download<i32>(dc, count.size(), stream), count);
            EXPECT_EQ(download<u64>(ds, 1, stream)[0], 0);
            cuda::check(cudaMemsetAsync(ids.data(), 0xff, rows * sizeof(i32), stream), "invalid ids");
            if (executable)
                cuda::check(cudaGraphLaunch(executable, stream), "replay invalid ids");
            else
                cuda::check(cuda::launch_match_first(view, device, stream), "match_first invalid ids");
            EXPECT_EQ(download<u64>(ds, 1, stream)[0], 4);
            EXPECT_EQ(download<i32>(dc, count.size(), stream), std::vector<i32>(rows, 0));
            if (executable)
                cuda::check(cudaGraphExecDestroy(executable), "destroy exec");
            if (graph)
                cuda::check(cudaGraphDestroy(graph), "destroy graph");
            cuda::check(cudaMemsetAsync(ids.data(), 0, rows * sizeof(i32), stream), "restore ids");
            cuda::check(cudaStreamSynchronize(stream), "sync");
        }
    }
}

TEST(DeviceRlOps, MatchFirstReportsCapacityVariableLimitAndInvalidInstances)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto stream = ctx->stream();
    const auto task = load("gripper__prob05");
    const auto templates = rl::lifted_goal_templates(rl::TaskSuite::of(rl::TaskTable::single(task)));
    const auto window = rollout(task, 2, 2, 3);
    std::vector<std::byte> packed(templates.arrays().bytes());
    std::memcpy(packed.data(), templates.arrays().block(), packed.size());
    auto block = upload(ctx, packed, stream);
    auto states = upload(ctx, window.states, stream);
    auto ids = upload(ctx, std::vector<i32>{0, -1, 0, 0}, stream);
    auto count = upload(ctx, std::vector<i32>(4), stream);
    auto status = upload(ctx, std::vector<u64>{99}, stream);
    auto oversized = upload(ctx, std::vector<i32>(templates.num_templates(), 65), stream);
    auto view = templates.view(static_cast<const std::byte*>(block.data()));
    rl::MatchBatch b;
    b.rows = 4;
    b.states = ptr<const u64>(states);
    b.task_ids = ptr<const i32>(ids);
    b.row_words = window.RW;
    b.capacity = 0;
    b.count = ptr<i32>(count);
    b.status = ptr<u64>(status);
    cuda::check(cuda::launch_match_first(view, b, stream), "truncated matches");
    EXPECT_EQ(download<u64>(status, 1, stream)[0], 5);
    const auto counts = download<i32>(count, 4, stream);
    EXPECT_GT(counts[0], 0);
    EXPECT_EQ(counts[1], 0);
    view.num_vars = ptr<const i32>(oversized);
    cuda::check(cuda::launch_match_first(view, b, stream), "oversized templates");
    EXPECT_EQ(download<u64>(status, 1, stream)[0], 6);
    EXPECT_EQ(download<i32>(count, 4, stream), std::vector<i32>(4, 0));
    b.rows = 0;
    cuda::check(cuda::launch_match_first(view, b, stream), "empty matches");
    EXPECT_EQ(download<u64>(status, 1, stream)[0], 0);
}

TEST(DeviceRlOps, MatchFirstGroundOnlyNeedsNoBindingStorage)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto stream = ctx->stream();
    const auto task = load("gripper__prob05");
    const auto suite = rl::TaskSuite::of(rl::TaskTable::single(task));
    const auto lifted = rl::lifted_goal_templates(suite);
    auto condition = lifted.condition(0);
    const auto state = task->initial_state();
    std::vector<ObjectId> binding;
    {
        const auto workspace = task->workspace();
        for_each_binding(*task, *workspace, condition, state.view(), {}, [&](std::span<const ObjectId> values)
        {
            binding.assign(values.begin(), values.end());
            return false;
        });
    }
    ASSERT_EQ(binding.size(), condition.variables.size());
    for (auto& literal : condition.literals)
        for (auto& term : literal.terms)
            if (term >= 0)
                term = formalism::object_term(binding[static_cast<u32>(term)]);
    condition.variables.clear();
    condition.equalities.clear();
    const auto templates = rl::GoalTemplates::from_conditions(suite, {{condition}});
    ASSERT_EQ(templates.max_vars(), 0);
    std::vector<u64> words(suite->words(), 0);
    std::copy(state.words().begin(), state.words().end(), words.begin());
    u8 found = 0;
    i32 which = -1, count = 0;
    u64 status = 99;
    rl::MatchBatch batch;
    batch.rows = 1;
    batch.row_words = suite->words();
    batch.states = words.data();
    batch.capacity = 1;
    batch.first = true;
    batch.found = &found;
    batch.template_id = &which;
    batch.count = &count;
    batch.status = &status;
    rl::match_first(templates.view(), batch);
    ASSERT_EQ(found, 1);
    ASSERT_EQ(which, 0);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(status, 0);
    std::vector<std::byte> packed(templates.arrays().bytes());
    std::memcpy(packed.data(), templates.arrays().block(), packed.size());
    auto block = upload(ctx, packed, stream), states = upload(ctx, words, stream);
    auto df = upload(ctx, std::vector<u8>{0}, stream);
    auto dt = upload(ctx, std::vector<i32>{-1}, stream), dc = upload(ctx, std::vector<i32>{0}, stream);
    auto ds = upload(ctx, std::vector<u64>{99}, stream);
    batch.states = ptr<const u64>(states);
    batch.found = ptr<u8>(df);
    batch.template_id = ptr<i32>(dt);
    batch.count = ptr<i32>(dc);
    batch.status = ptr<u64>(ds);
    cuda::check(cuda::launch_match_first(templates.view(static_cast<const std::byte*>(block.data())), batch, stream),
                "ground-only match");
    EXPECT_EQ(download<u8>(df, 1, stream)[0], found);
    EXPECT_EQ(download<i32>(dt, 1, stream)[0], which);
    EXPECT_EQ(download<i32>(dc, 1, stream)[0], count);
    EXPECT_EQ(download<u64>(ds, 1, stream)[0], status);
}
