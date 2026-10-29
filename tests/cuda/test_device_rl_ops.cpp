// Device RL-helper-kernel tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the device kernels of the RL helpers (cuda/rl_ops.hpp) equal the host loops (rl/novelty.hpp, rl/prefix.hpp,
//     rl/her.hpp) byte for byte on suite windows: novelty rewards and tables, prefix and schema masks, hindsight
//     relabels (every strategy, goal subsets, eligible-atom masks, env ids, the states the steps started from);
//   - the random policy's actions without a step (envk::launch_random_actions) equal rl::rng::successor_index;
//   - visited-table lookups (cuda::launch_visited_mask) equal rl::visited_mask, and the visited flags of action
//     targets (cuda::launch_target_visited) equal rl::target_visited and a direct search of the batch;
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

TEST(DeviceRlOps, MatchFirstPigeonholeAndSearchBudgetEqualHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto stream = ctx->stream();
    namespace ed = encoding::detail;
    for (u32 mode = 0; mode < 5; ++mode)
    {
        SCOPED_TRACE(mode);
        const bool typed = mode == 1;
        const u32 vars = mode < 2 || mode == 4 ? 64 : 4;
        std::vector<u64> inst(ed::k_instance_columns), states(1);
        inst[ed::k_num_objects] = typed ? 70 : mode < 2 || mode == 4 ? 63 : 4;
        std::vector<u32> ranges(ed::k_literal_columns);
        ranges[ed::k_literal_predicate_kind] = ed::k_literal_static_predicate;
        ranges[ed::k_literal_static_end] = typed ? 63 : 1;
        std::vector<i32> instance_offsets{0, 1}, num_vars{static_cast<i32>(vars)};
        std::vector<i32> literal_offsets{0, typed ? 64 : 1}, order(64), terms(64), args(63);
        for (i32 v = 0; v < 64; ++v)
            order[v] = terms[v] = v;
        for (i32 v = 0; v < 63; ++v)
            args[v] = v;
        std::vector<i32> predicate(typed ? 64 : 1), arity(predicate.size(), typed ? 1 : 0);
        std::vector<u8> polarity(predicate.size(), 1), distinct{static_cast<u8>(mode != 4)};
        rl::MatchTemplates t;
        t.num_instances = t.capacity = t.max_arity = 1;
        t.max_vars = 64;
        t.instance_offsets = instance_offsets.data();
        t.num_vars = num_vars.data();
        t.literal_offsets = literal_offsets.data();
        t.order = order.data();
        t.terms = terms.data();
        t.predicate = predicate.data();
        t.arity = arity.data();
        t.polarity = polarity.data();
        t.distinct = distinct.data();
        t.literals.inst = inst.data();
        t.literals.num_instances = t.literals.num_predicates = t.literals.max_arity = 1;
        t.literals.literal_ranges = ranges.data();
        t.literals.static_args = args.data();
        std::vector<u8> found(1, 99);
        std::vector<i32> binding(64, 99), which(1, 99), count(1, 99);
        u64 status = 99;
        rl::MatchBatch b;
        b.states = states.data();
        b.rows = b.row_words = b.capacity = 1;
        b.max_steps = mode == 0 ? 0 : typed ? 3 : mode == 2 ? 7 : mode == 3 ? 8 : 128;
        b.found = found.data();
        b.binding = binding.data();
        b.template_id = which.data();
        b.count = count.data();
        b.status = &status;
        rl::match_first(t, b);
        ASSERT_EQ(status, mode == 2 ? 8 : 0);
        ASSERT_EQ(found[0], mode >= 3 ? 1 : 0);
        ASSERT_EQ(count[0], mode >= 3 ? 1 : 0);
        ASSERT_EQ(which[0], mode >= 3 ? 0 : -1);
        for (u32 v = 0; v < 64; ++v)
            ASSERT_EQ(binding[v], mode == 4 ? 0 : mode == 3 && v < vars ? static_cast<i32>(v) : -1);
        auto di = upload(ctx, inst, stream), dr = upload(ctx, ranges, stream);
        auto dio = upload(ctx, instance_offsets, stream), dnv = upload(ctx, num_vars, stream);
        auto dlo = upload(ctx, literal_offsets, stream), dor = upload(ctx, order, stream);
        auto dte = upload(ctx, terms, stream), da = upload(ctx, args, stream);
        auto dp = upload(ctx, predicate, stream), dar = upload(ctx, arity, stream);
        auto dpo = upload(ctx, polarity, stream), dd = upload(ctx, distinct, stream);
        t.instance_offsets = ptr<const i32>(dio);
        t.num_vars = ptr<const i32>(dnv);
        t.literal_offsets = ptr<const i32>(dlo);
        t.order = ptr<const i32>(dor);
        t.terms = ptr<const i32>(dte);
        t.predicate = ptr<const i32>(dp);
        t.arity = ptr<const i32>(dar);
        t.polarity = ptr<const u8>(dpo);
        t.distinct = ptr<const u8>(dd);
        t.literals.inst = ptr<const u64>(di);
        t.literals.literal_ranges = ptr<const u32>(dr);
        t.literals.static_args = ptr<const i32>(da);
        auto ds = upload(ctx, states, stream), df = upload(ctx, found, stream), db = upload(ctx, binding, stream);
        auto dt = upload(ctx, which, stream), dc = upload(ctx, count, stream);
        auto dst = upload(ctx, std::vector<u64>{99}, stream);
        b.states = ptr<const u64>(ds);
        b.found = ptr<u8>(df);
        b.binding = ptr<i32>(db);
        b.template_id = ptr<i32>(dt);
        b.count = ptr<i32>(dc);
        b.status = ptr<u64>(dst);
        cuda::check(cuda::launch_match_first(t, b, stream), "bounded matches");
        EXPECT_EQ(download<u8>(df, 1, stream), found);
        EXPECT_EQ(download<i32>(db, 64, stream), binding);
        EXPECT_EQ(download<i32>(dt, 1, stream), which);
        EXPECT_EQ(download<i32>(dc, 1, stream), count);
        EXPECT_EQ(download<u64>(dst, 1, stream)[0], status);
    }
}

TEST(DeviceRlOps, MatchFirstRequiresStatus)
{
    rl::MatchTemplates t;
    rl::MatchBatch b;
    EXPECT_THROW(rl::match_first(t, b), std::invalid_argument);
    EXPECT_EQ(cuda::launch_match_first(t, b, nullptr), cudaErrorInvalidValue);
}

// rl::visited_mask on the device equals the host loop: tables with zero and the reserved key, wrap-around probing and
// saturated rows; parents with and without a row map, padding, and entries outside the table (not visited).
TEST(DeviceRlOps, VisitedMaskDeviceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    for (u32 H : {2u, 8u, 64u})
    {
        SCOPED_TRACE("H " + std::to_string(H));
        const u64 N = 37;
        std::vector<u64> table(N * H, 0);
        std::vector<i32> count(N, 0);
        std::vector<u64> keys;
        for (u64 i = 0; i < N; ++i)
        {
            // row i: i % 5 inserts, more than the capacity holds in some rows; keys that start probing at the last slot
            const u64 inserts = (i % 5) * H / 3 + (i % 7 == 0 ? H : 0);
            for (u64 k = 0; k < inserts; ++k)
            {
                u64 key = hash::mix64(i * 1000 + k);
                if (k == 0 && i % 3 == 0)
                    key = 0;
                else if (k == 1 && i % 4 == 0)
                    key = rl::k_visited_zero;
                else if (k % 3 == 2)
                    key = (key & ~u64{H - 1}) | (H - 1);
                (void)rl::visited_insert(table.data() + i * H, H, count[i], key);
                keys.push_back(key);
            }
        }
        // queries: inserted keys, new keys, zero and the reserved key, in every row
        std::vector<u64> hashes;
        std::vector<i32> parent;
        for (u64 j = 0; j < 4 * keys.size() + 64; ++j)
        {
            const u64 pick = hash::mix64(j ^ 0xBEEF);
            hashes.push_back(j % 4 == 0 ? hash::mix64(j) : j % 9 == 1 ? 0 : j % 9 == 2 ? rl::k_visited_zero
                                                                         : keys[pick % keys.size()]);
            parent.push_back(j % 11 == 3 ? -1 : static_cast<i32>(pick % N));
        }
        const u64 M = hashes.size();
        std::vector<u8> host(M);
        rl::visited_mask(table.data(), count.data(), H, hashes.data(), parent.data(), nullptr, M, host.data());
        u64 hits = 0;
        for (u8 h : host)
            hits += h;
        EXPECT_GT(hits, 0u);
        EXPECT_LT(hits, M);
        cuda::DeviceBuffer dt = upload(ctx, table, s), dc = upload(ctx, count, s), dh = upload(ctx, hashes, s),
                           dp = upload(ctx, parent, s), dout(ctx, M, s);
        cuda::check(cuda::launch_visited_mask(ptr<const u64>(dt), ptr<const i32>(dc), N, H, ptr<const u64>(dh),
                                              ptr<const i32>(dp), nullptr, N, M, ptr<u8>(dout), s),
                    "visited_mask");
        ASSERT_EQ(download<u8>(dout, M, s), host);
        // a row map: parent p is row map[p] (a permutation with -1 entries)
        std::vector<i32> map(N + 4);
        for (u64 p = 0; p < map.size(); ++p)
            map[p] = p % 6 == 5 ? -1 : static_cast<i32>((p * 7 + 3) % N);
        std::vector<i32> mapped(M);
        for (u64 j = 0; j < M; ++j)
            mapped[j] = parent[j] < 0 ? -1 : static_cast<i32>((static_cast<u64>(parent[j]) * 5 + j) % map.size());
        rl::visited_mask(table.data(), count.data(), H, hashes.data(), mapped.data(), map.data(), M, host.data());
        cuda::DeviceBuffer dm = upload(ctx, map, s), dq = upload(ctx, mapped, s);
        cuda::check(cuda::launch_visited_mask(ptr<const u64>(dt), ptr<const i32>(dc), N, H, ptr<const u64>(dh),
                                              ptr<const i32>(dq), ptr<const i32>(dm), map.size(), M, ptr<u8>(dout), s),
                    "visited_mask (rows)");
        ASSERT_EQ(download<u8>(dout, M, s), host);
        // entries outside the table read as not visited: parents past the map or the rows, mapped rows past the table
        std::vector<i32> outside{static_cast<i32>(N), -2, 0, 1};
        std::vector<i32> far_map{static_cast<i32>(N), -5};
        std::vector<u64> any(outside.size(), keys.front());
        cuda::DeviceBuffer d_out = upload(ctx, outside, s), d_far = upload(ctx, far_map, s),
                           d_any = upload(ctx, any, s);
        cuda::check(cuda::launch_visited_mask(ptr<const u64>(dt), ptr<const i32>(dc), N, H, ptr<const u64>(d_any),
                                              ptr<const i32>(d_out), ptr<const i32>(d_far), far_map.size(), outside.size(),
                                              ptr<u8>(dout), s),
                    "visited_mask (outside)");
        ASSERT_EQ(download<u8>(dout, outside.size(), s), std::vector<u8>(outside.size(), 0));
    }
}

/// A synthetic encoded batch over G graphs with its expansion rows (some graphs without rows or entities, rows
/// dropped with row_index), its targets, and visited tables: target_visited of the host, of the device and of a
/// direct search agree for index type I.
template<class I>
void check_target_visited(const cuda::ContextPtr& ctx, bool drop, bool mapped, i64 live, i32 status)
{
    const cudaStream_t s = ctx->stream();
    const u64 G = 23, H = 16, N = G + 3;
    std::vector<I> entity_offsets{0};
    std::vector<u8> role;
    std::vector<I> ref;
    std::vector<i32> row_offsets{0}, row_index;
    std::vector<u64> succ_hash;
    for (u64 g = 0; g < G; ++g)
    {
        const u64 successors = g % 5 == 4 ? 0 : 3 + g % 4;  // before the drop
        for (u64 k = 0; k < successors; ++k)
            if (!drop || k % 3 != 1)
            {
                row_index.push_back(static_cast<i32>(k));
                succ_hash.push_back(hash::mix64(g * 100 + k));
            }
        row_offsets.push_back(static_cast<i32>(succ_hash.size()));
        const u64 entities = g % 7 == 6 ? 0 : 4 + successors;
        for (u64 e = 0; e < entities; ++e)
        {
            // objects, then one action entity per successor before the drop (a dropped one too: it has no row)
            const bool action = e >= 4;
            role.push_back(action ? 4 : static_cast<u8>(e % 3));
            ref.push_back(static_cast<I>(action ? static_cast<i64>(e - 4) : static_cast<i64>(e)));
        }
        entity_offsets.push_back(static_cast<I>(role.size()));
    }
    for (u64 k = 0; k < 5; ++k)
    {
        row_offsets.push_back(row_offsets.back());  // a suite's rows past the batch's graphs
        entity_offsets.push_back(0);                // a capacity batch's graphs past its live ones
    }
    const u64 M = succ_hash.size();
    std::vector<i32> env_rows(G);
    for (u64 g = 0; g < G; ++g)
        env_rows[g] = g % 9 == 8 ? -1 : static_cast<i32>((g * 5 + 2) % N);
    // tables: every other successor of a graph visited, in its table row
    std::vector<u64> table(N * H, 0);
    std::vector<i32> count(N, 0);
    for (u64 g = 0; g < G; ++g)
    {
        const i64 row = mapped ? env_rows[g] : static_cast<i64>(g);
        if (row < 0)
            continue;
        for (i32 r = row_offsets[g]; r < row_offsets[g + 1]; ++r)
            if (r % 2 == 0)
                (void)rl::visited_insert(table.data() + row * H, H, count[row], succ_hash[r]);
    }
    // targets: every entity once, then out-of-range ones
    std::vector<I> target;
    for (u64 e = 0; e < role.size(); ++e)
        target.push_back(static_cast<I>((e * 7) % role.size()));
    target.push_back(static_cast<I>(-1));
    target.push_back(static_cast<I>(role.size()));
    const u64 T = target.size();
    std::vector<u8> want(T, 0);
    for (u64 t = 0; t < T; ++t)
    {
        const i64 e = static_cast<i64>(target[t]);
        if ((live >= 0 && static_cast<i64>(t) >= live) || status || e < 0 || e >= static_cast<i64>(role.size()) ||
            role[e] != 4)
            continue;
        u64 g = 0;
        while (!(static_cast<i64>(entity_offsets[g]) <= e && e < static_cast<i64>(entity_offsets[g + 1])))
            ++g;
        i64 at = -1;
        for (i32 r = row_offsets[g]; r < row_offsets[g + 1]; ++r)
            if ((drop ? row_index[r] : r - row_offsets[g]) == static_cast<i64>(ref[e]))
                at = r;
        const i64 row = mapped ? env_rows[g] : static_cast<i64>(g);
        if (at < 0 || row < 0)
            continue;
        want[t] = rl::visited_contains(table.data() + row * H, H, count[row], succ_hash[at]) ? 1 : 0;
    }
    u64 hits = 0;
    for (const u8 w : want)
        hits += w;
    if (live < 0 && !status)
    {
        EXPECT_GT(hits, 3u);
    }
    const std::vector<i64> live_v{live};
    const std::vector<i32> status_v{status};
    rl::TargetRows<I> x;
    x.target = target.data();
    x.live = live >= 0 ? live_v.data() : nullptr;
    x.targets = T;
    x.role = role.data();
    x.ref = ref.data();
    x.entities = role.size();
    x.action_role = 4;
    x.entity_offsets = entity_offsets.data();
    x.graphs = G + 5;
    x.row_offsets = row_offsets.data();
    x.row_index = drop ? row_index.data() : nullptr;
    x.succ_hash = succ_hash.data();
    x.rows = M;
    x.status = status_v.data();
    x.env_rows = mapped ? env_rows.data() : nullptr;
    std::vector<u8> host(T);
    rl::target_visited(x, table.data(), count.data(), N, H, host.data());
    EXPECT_EQ(host, want);
    cuda::DeviceBuffer dt = upload(ctx, target, s), dl = upload(ctx, live_v, s), dr = upload(ctx, role, s),
                       df = upload(ctx, ref, s), de = upload(ctx, entity_offsets, s), dro = upload(ctx, row_offsets, s),
                       dri = upload(ctx, row_index, s), dh = upload(ctx, succ_hash, s), ds = upload(ctx, status_v, s),
                       dm = upload(ctx, env_rows, s), dtab = upload(ctx, table, s), dc = upload(ctx, count, s),
                       dout(ctx, T, s);
    rl::TargetRows<I> d = x;
    d.target = ptr<const I>(dt);
    d.live = live >= 0 ? ptr<const i64>(dl) : nullptr;
    d.role = ptr<const u8>(dr);
    d.ref = ptr<const I>(df);
    d.entity_offsets = ptr<const I>(de);
    d.row_offsets = ptr<const i32>(dro);
    d.row_index = drop ? ptr<const i32>(dri) : nullptr;
    d.succ_hash = ptr<const u64>(dh);
    d.status = ptr<const i32>(ds);
    d.env_rows = mapped ? ptr<const i32>(dm) : nullptr;
    cuda::check(cuda::launch_target_visited(d, ptr<const u64>(dtab), ptr<const i32>(dc), N, H, ptr<u8>(dout), s),
                "target_visited");
    EXPECT_EQ(download<u8>(dout, T, s), want);
}

TEST(DeviceRlOps, TargetVisitedDeviceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    for (const bool drop : {false, true})
        for (const bool mapped : {false, true})
        {
            SCOPED_TRACE(std::string(drop ? "drop" : "all rows") + (mapped ? ", mapped rows" : ""));
            check_target_visited<i32>(ctx, drop, mapped, -1, 0);
            check_target_visited<i64>(ctx, drop, mapped, -1, 0);
            check_target_visited<i32>(ctx, drop, mapped, 40, 0);
            check_target_visited<i64>(ctx, drop, mapped, -1, 1);
        }
}
