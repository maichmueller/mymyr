// Device RL environment tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - the device RNG: rl::rng's Philox-4x32-10 on the device equals the host and cuRAND's curand_Philox4x32_10;
//   - the device step (cuda::DeviceEnv, the fast and the general path) equals the host step (rl::HostEnv) bit
//     for bit on every suite task, frozen and lazy: states, step and draw counters, rewards, terminated, truncated, goal
//     and invalid flags, counts, final states and labels, over steps of the random policy and of given actions
//     (including invalid ones), with truncation and autoreset; the TorchRL convention (no autoreset, masked resets);
//   - the results do not depend on the chunk sizes, the path, or the batch an environment runs in (first_env).
// By default 256 environments step 60 times per task; MYMYR_TEST_FULL_SUITE=1 runs the gate's 1024 x 300.

#include "../cpp/support/suite.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/rng.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
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
    o.max_bytes = u64{3} << 30;  // the GPU may be shared: well below a full device
    return cuda::DeviceContext::create(0, o);
}

TaskPtr load(const std::string& name, bool frozen)
{
    TaskOptions o;
    o.atoms = frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
    return Task::from_text_file(task_path(name), o);
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

template<class T>
std::vector<T> to_host(const cuda::DeviceBuffer& d, u64 n, cudaStream_t s)
{
    std::vector<T> out(n);
    if (n)
    {
        cuda::check(cudaMemcpyAsync(out.data(), d.data(), n * sizeof(T), cudaMemcpyDeviceToHost, s), "D2H");
        cuda::check(cudaStreamSynchronize(s), "sync");
    }
    return out;
}

template<class T>
void to_device(cuda::DeviceBuffer& d, const std::vector<T>& v, cudaStream_t s)
{
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
}

/// The outputs and state of one step, on the host (for comparisons).
struct Snapshot
{
    std::vector<u64> states, final_states, draws;
    std::vector<i32> steps, count, schema, binding;
    std::vector<f32> reward;
    std::vector<u8> terminated, truncated, invalid, goal;
    bool operator==(const Snapshot& o) const
    {
        // rewards bitwise
        return states == o.states && final_states == o.final_states && draws == o.draws && steps == o.steps &&
               count == o.count && schema == o.schema && binding == o.binding && terminated == o.terminated &&
               truncated == o.truncated && invalid == o.invalid && goal == o.goal && reward.size() == o.reward.size() &&
               std::memcmp(reward.data(), o.reward.data(), reward.size() * sizeof(f32)) == 0;
    }
};

/// A batch on the host (rl::HostEnv).
struct HostBatch
{
    u32 W, L;
    Snapshot v;
    rl::EnvBatch b;
    rl::StepOutputs out;

    HostBatch(u32 words, u32 label_width, u64 rows, u64 first_env) : W(words), L(label_width)
    {
        v.states.assign(rows * W, 0);
        v.final_states.assign(rows * W, 0);
        v.draws.assign(rows, 0);
        v.steps.assign(rows, 0);
        v.count.assign(rows, 0);
        v.schema.assign(rows, 0);
        v.binding.assign(rows * L, 0);
        v.reward.assign(rows, 0);
        v.terminated.assign(rows, 0);
        v.truncated.assign(rows, 0);
        v.invalid.assign(rows, 0);
        v.goal.assign(rows, 0);
        b = rl::EnvBatch{};
        b.states = v.states.data();
        b.rows = rows;
        b.words = W;
        b.steps = v.steps.data();
        b.draws = v.draws.data();
        b.first_env = first_env;
        out = rl::StepOutputs{v.reward.data(), v.terminated.data(), v.truncated.data(), v.count.data(),
                              v.final_states.data(), v.schema.data(), v.binding.data(), L, v.invalid.data(), v.goal.data()};
    }
};

/// A batch in device memory (cuda::DeviceEnv).
struct DeviceBatch
{
    u64 N;
    u32 W, L, S;
    u64 V;
    cudaStream_t s;
    cuda::DeviceBuffer states, final_states, draws, steps, count, schema, binding, reward, terminated, truncated, invalid,
        goal, counts, views, action, mask;
    rl::EnvBatch b;
    rl::StepOutputs out;

    DeviceBatch(const cuda::ContextPtr& ctx, const cuda::DeviceEnv& env, u64 rows, u64 first_env)
        : N(rows), W(env.words()), L(env.label_width()), S(env.cache_schemas()), V(env.cache_view_words()), s(env.stream()),
          states(ctx, rows * W * 8, s), final_states(ctx, rows * W * 8, s), draws(ctx, rows * 8, s), steps(ctx, rows * 4, s),
          count(ctx, rows * 4, s), schema(ctx, rows * 4, s), binding(ctx, rows * L * 4, s), reward(ctx, rows * 4, s),
          terminated(ctx, rows, s), truncated(ctx, rows, s), invalid(ctx, rows, s), goal(ctx, rows, s),
          counts(ctx, std::max<u64>(rows * S, 1) * 4, s), views(ctx, std::max<u64>(rows * V, 1) * 8, s),
          action(ctx, rows * 8, s), mask(ctx, rows, s)
    {
        cuda::check(cudaMemsetAsync(draws.data(), 0, rows * 8, s), "memset");
        b = rl::EnvBatch{};
        b.states = static_cast<u64*>(states.data());
        b.rows = rows;
        b.words = W;
        b.steps = static_cast<i32*>(steps.data());
        b.draws = static_cast<u64*>(draws.data());
        b.first_env = first_env;
        b.counts = S ? static_cast<u32*>(counts.data()) : nullptr;
        b.views = V ? static_cast<u64*>(views.data()) : nullptr;
        out = rl::StepOutputs{static_cast<f32*>(reward.data()),   static_cast<u8*>(terminated.data()),
                              static_cast<u8*>(truncated.data()), static_cast<i32*>(count.data()),
                              static_cast<u64*>(final_states.data()), static_cast<i32*>(schema.data()),
                              static_cast<i32*>(binding.data()),  L,
                              static_cast<u8*>(invalid.data()),   static_cast<u8*>(goal.data())};
    }

    [[nodiscard]] Snapshot snapshot() const
    {
        Snapshot v;
        v.states = to_host<u64>(states, N * W, s);
        v.final_states = to_host<u64>(final_states, N * W, s);
        v.draws = to_host<u64>(draws, N, s);
        v.steps = to_host<i32>(steps, N, s);
        v.count = to_host<i32>(count, N, s);
        v.schema = to_host<i32>(schema, N, s);
        v.binding = to_host<i32>(binding, N * L, s);
        v.reward = to_host<f32>(reward, N, s);
        v.terminated = to_host<u8>(terminated, N, s);
        v.truncated = to_host<u8>(truncated, N, s);
        v.invalid = to_host<u8>(invalid, N, s);
        v.goal = to_host<u8>(goal, N, s);
        return v;
    }
};

/// Actions of step t: an index in [0, count] from a hash (count itself is invalid; so is anything for count 0).
std::vector<i64> actions(const std::vector<i32>& count, int t)
{
    std::vector<i64> a(count.size());
    for (usize i = 0; i < count.size(); ++i)
        a[i] = static_cast<i64>(hash::mix64((static_cast<u64>(t) << 32) ^ i) % (static_cast<u64>(std::max(count[i], 0)) + 1));
    return a;
}

void expect_equal(const Snapshot& d, const Snapshot& h, const char* what, int t)
{
    ASSERT_EQ(d.states, h.states) << what << " step " << t;
    ASSERT_EQ(d.final_states, h.final_states) << what << " step " << t;
    ASSERT_EQ(d.steps, h.steps) << what << " step " << t;
    ASSERT_EQ(d.draws, h.draws) << what << " step " << t;
    ASSERT_EQ(d.count, h.count) << what << " step " << t;
    ASSERT_EQ(d.schema, h.schema) << what << " step " << t;
    ASSERT_EQ(d.binding, h.binding) << what << " step " << t;
    ASSERT_EQ(d.terminated, h.terminated) << what << " step " << t;
    ASSERT_EQ(d.truncated, h.truncated) << what << " step " << t;
    ASSERT_EQ(d.goal, h.goal) << what << " step " << t;
    ASSERT_EQ(d.invalid, h.invalid) << what << " step " << t;
    ASSERT_TRUE(d == h) << what << " step " << t << ": rewards differ";
}

rl::EnvConfig config(u64 seed, bool autoreset)
{
    rl::EnvConfig c;
    c.seed = seed;
    c.max_steps = 40;
    c.step_reward = -1.0f;
    c.goal_reward = 3.5f;
    c.dead_end_reward = -0.25f;
    c.autoreset = autoreset;
    return c;
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
}  // namespace

// ------------------------------------------------------------------------------------------------ RNG

TEST(DevicePhiloxRng, DeviceEqualsHostAndCurand)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const cudaStream_t s = ctx->stream();
    const u64 n = 4096;
    std::vector<u32> ctr(n * 4), key(n * 2);
    for (u64 i = 0; i < n; ++i)
    {
        for (u32 j = 0; j < 4; ++j)
            ctr[i * 4 + j] = static_cast<u32>(hash::mix64(i * 8 + j));
        for (u32 j = 0; j < 2; ++j)
            key[i * 2 + j] = static_cast<u32>(hash::mix64(i * 8 + 4 + j) >> 7);
    }
    cuda::DeviceBuffer dc(ctx, n * 16, s), dk(ctx, n * 8, s), d0(ctx, n * 16, s), d1(ctx, n * 16, s);
    to_device(dc, ctr, s);
    to_device(dk, key, s);
    cuda::check(cuda::envk::launch_philox(static_cast<u32*>(dc.data()), static_cast<u32*>(dk.data()), n, static_cast<u32*>(d0.data()), 0, s), "philox");
    cuda::check(cuda::envk::launch_philox(static_cast<u32*>(dc.data()), static_cast<u32*>(dk.data()), n, static_cast<u32*>(d1.data()), 1, s), "curand");
    const std::vector<u32> ours = to_host<u32>(d0, n * 4, s), cur = to_host<u32>(d1, n * 4, s);
    EXPECT_EQ(ours, cur) << "rl::rng differs from curand_Philox4x32_10";
    for (u64 i = 0; i < n; ++i)
    {
        const rl::rng::Block4 h =
            rl::rng::philox4x32_10(rl::rng::Block4{{ctr[i * 4], ctr[i * 4 + 1], ctr[i * 4 + 2], ctr[i * 4 + 3]}}, key[i * 2], key[i * 2 + 1]);
        for (u32 j = 0; j < 4; ++j)
            ASSERT_EQ(ours[i * 4 + j], h.v[j]) << "block " << i;
    }
}

// ------------------------------------------------------------------------------------------------ gate 1

class DeviceEnvStepSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(DeviceEnvStepSuite, DeviceStepEqualsHostStep)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const u64 N = sanitizer_size() ? 64 : full_suite() ? 1024 : 256;
    const int T = sanitizer_size() ? 12 : full_suite() ? 300 : 60;
    const auto task = load(name, frozen);
    auto ctx = context();
    rl::EnvConfig cfg = config(0x5eed0000 + N, true);
    if (sanitizer_size())
        cfg.max_steps = 5;
    // the device envs: the fast path where the task allows it, and the general path
    std::vector<std::unique_ptr<cuda::DeviceEnv>> envs;
    const std::string why = cuda::DeviceEnv::fast_unsupported(*rl::TaskTable::single(task), cfg);
    if (why.empty())
        envs.push_back(std::make_unique<cuda::DeviceEnv>(ctx, rl::TaskTable::single(task), cfg, cuda::DeviceEnv::Path::Fast));
    envs.push_back(std::make_unique<cuda::DeviceEnv>(ctx, rl::TaskTable::single(task), cfg, cuda::DeviceEnv::Path::General));
    RecordProperty("fast", why.empty() ? "yes" : why);
    rl::HostEnv host(rl::TaskTable::single(task), cfg);
    ASSERT_EQ(host.words(), envs[0]->words());
    ASSERT_EQ(host.initial_count(0), envs[0]->initial_count(0));
    HostBatch h(host.words(), envs[0]->label_width(), N, 0);
    std::vector<std::unique_ptr<DeviceBatch>> dev;
    for (auto& e : envs)
        dev.push_back(std::make_unique<DeviceBatch>(ctx, *e, N, 0));
    host.reset(h.b, nullptr, h.out.count);
    for (usize k = 0; k < envs.size(); ++k)
        envs[k]->reset(dev[k]->b, nullptr, dev[k]->out.count);
    u64 terminated = 0, truncated = 0, goals = 0, invalid = 0;
    for (int t = 0; t < T; ++t)
    {
        // every third step takes given actions (from the counts of the previous step), the others the random policy
        const bool given = t % 3 == 2;
        const std::vector<i64> act = actions(h.v.count, t);
        for (usize k = 0; k < envs.size(); ++k)
        {
            // the device steps first: under lazy slots it interns the atoms it meets, the host then finds them
            if (given)
                to_device(dev[k]->action, act, dev[k]->s);
            envs[k]->step(dev[k]->b, dev[k]->out,
                          rl::Actions{given ? static_cast<const i64*>(dev[k]->action.data()) : nullptr});
        }
        host.step(h.b, h.out, rl::Actions{given ? act.data() : nullptr});
        for (usize k = 0; k < envs.size(); ++k)
        {
            const Snapshot d = dev[k]->snapshot();
            expect_equal(d, h.v, envs[k]->fast() ? "fast path" : "general path", t);
            if (::testing::Test::HasFatalFailure())
                return;
        }
        for (u64 i = 0; i < N; ++i)
        {
            terminated += h.v.terminated[i];
            truncated += h.v.truncated[i];
            goals += h.v.goal[i];
            invalid += h.v.invalid[i];
        }
    }
    for (auto& e : envs)
        EXPECT_NO_THROW(e->check_errors());
    // episodes end (some tasks end all of theirs in dead ends before max_steps: pegsol, openstacks, organic-synthesis)
    EXPECT_GT(truncated + terminated, 0u) << "no episode ended";
    EXPECT_GT(invalid, 0u) << "no invalid action was tested";
    RecordProperty("terminated", static_cast<int>(terminated));
    RecordProperty("truncated", static_cast<int>(truncated));
    RecordProperty("goals", static_cast<int>(goals));
}

INSTANTIATE_TEST_SUITE_P(Suite, DeviceEnvStepSuite, ::testing::ValuesIn(params()), param_name);

// ------------------------------------------------------------------------------------------------ TorchRL convention

TEST(DeviceEnvStep, NoAutoresetWithMaskedResets)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"pegsol-08-strips__p22", "gripper__prob05", "philosophers__p03-phil4"})
        for (auto path : {cuda::DeviceEnv::Path::Auto, cuda::DeviceEnv::Path::General})
        {
            SCOPED_TRACE(std::string(name) + (path == cuda::DeviceEnv::Path::Auto ? " auto" : " general"));
            const auto task = load(name, true);
            auto ctx = context();
            const rl::EnvConfig cfg = config(7, false);
            cuda::DeviceEnv env(ctx, rl::TaskTable::single(task), cfg, path);
            rl::HostEnv host(rl::TaskTable::single(task), cfg);
            const u64 N = 300;
            HostBatch h(host.words(), env.label_width(), N, 0);
            DeviceBatch d(ctx, env, N, 0);
            host.reset(h.b, nullptr, h.out.count);
            env.reset(d.b, nullptr, d.out.count);
            for (int t = 0; t < 80; ++t)
            {
                env.step(d.b, d.out);
                host.step(h.b, h.out);
                expect_equal(d.snapshot(), h.v, "step", t);
                if (HasFatalFailure())
                    return;
                // reset the finished environments, as TorchRL's step_and_maybe_reset does, every other step
                if (t % 2 == 1)
                {
                    std::vector<u8> mask(N);
                    for (u64 i = 0; i < N; ++i)
                        mask[i] = h.v.terminated[i] | h.v.truncated[i];
                    to_device(d.mask, mask, d.s);
                    host.reset(h.b, mask.data(), h.out.count);
                    env.reset(d.b, static_cast<const u8*>(d.mask.data()), d.out.count);
                    ASSERT_EQ(to_host<u64>(d.states, N * d.W, d.s), h.v.states);
                    ASSERT_EQ(to_host<i32>(d.count, N, d.s), h.v.count);
                }
            }
            EXPECT_NO_THROW(env.check_errors());
        }
}

// ------------------------------------------------------------------------------------------------ determinism

TEST(DeviceEnvStep, ChunksPathsAndBatchesDoNotChangeTrajectories)
{
    SKIP_WITHOUT_GPU();
    for (const char* name : {"sokoban-opt08-strips__p14", "depot__p02", "freecell__p02", "rovers__p02"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name, true);
        auto ctx = context();
        const rl::EnvConfig cfg = config(11, true);
        const u64 N = 500;
        // reference: the fast path, automatic chunks, one batch
        cuda::DeviceEnv ref(ctx, rl::TaskTable::single(task), cfg);
        ASSERT_TRUE(ref.fast()) << cuda::DeviceEnv::fast_unsupported(*rl::TaskTable::single(task), cfg);
        DeviceBatch r(ctx, ref, N, 0);
        ref.reset(r.b);
        // the fast path in chunks of 7 rows, the general path in chunks of 13, and two batches [0, 180) + [180, 500)
        const rl::TaskTablePtr table = rl::TaskTable::single(task);
        cuda::DeviceEnv c7(ctx, table, cfg), gen(ctx, table, cfg, cuda::DeviceEnv::Path::General), lo(ctx, table, cfg),
            hi(ctx, table, cfg);
        c7.set_chunk_rows(7);
        gen.set_chunk_rows(13);
        DeviceBatch a(ctx, c7, N, 0), g(ctx, gen, N, 0), l(ctx, lo, 180, 0), u(ctx, hi, N - 180, 180);
        c7.reset(a.b);
        gen.reset(g.b);
        lo.reset(l.b);
        hi.reset(u.b);
        for (int t = 0; t < 50; ++t)
        {
            for (auto [env, batch] : {std::pair{&ref, &r}, {&c7, &a}, {&gen, &g}, {&lo, &l}, {&hi, &u}})
                env->step(batch->b, batch->out);
            const Snapshot sr = r.snapshot();
            ASSERT_TRUE(a.snapshot() == sr) << "chunks of 7, step " << t;
            ASSERT_TRUE(g.snapshot() == sr) << "general path, step " << t;
            const Snapshot sl = l.snapshot(), su = u.snapshot();
            std::vector<u64> states(sl.states);
            states.insert(states.end(), su.states.begin(), su.states.end());
            ASSERT_EQ(states, sr.states) << "split batch, step " << t;
            std::vector<f32> reward(sl.reward);
            reward.insert(reward.end(), su.reward.begin(), su.reward.end());
            ASSERT_EQ(0, std::memcmp(reward.data(), sr.reward.data(), N * sizeof(f32))) << "split batch, step " << t;
        }
        // another stream: the same trajectory from the same start
        cuda::Stream other;
        cuda::DeviceEnv moved(ctx, rl::TaskTable::single(task), cfg);
        DeviceBatch m(ctx, moved, N, 0);
        cuda::DeviceEnv again(ctx, rl::TaskTable::single(task), cfg);
        DeviceBatch q(ctx, again, N, 0);
        moved.reset(m.b);
        again.reset(q.b);
        cuda::check(cudaStreamSynchronize(m.s), "sync");
        moved.set_stream(other);
        m.s = other;
        for (int t = 0; t < 10; ++t)
        {
            moved.step(m.b, m.out);
            again.step(q.b, q.out);
        }
        ASSERT_TRUE(m.snapshot() == q.snapshot());
    }
}

TEST(DeviceEnvStep, RefreshAfterWritingStates)
{
    SKIP_WITHOUT_GPU();
    const auto task = load("blocks__probBLOCKS-8-0", true);
    auto ctx = context();
    const rl::EnvConfig cfg = config(3, true);
    cuda::DeviceEnv env(ctx, rl::TaskTable::single(task), cfg);
    rl::HostEnv host(rl::TaskTable::single(task), cfg);
    const u64 N = 64;
    DeviceBatch d(ctx, env, N, 0);
    HostBatch h(host.words(), env.label_width(), N, 0);
    host.reset(h.b);
    for (int t = 0; t < 9; ++t)
        host.step(h.b, h.out);
    // the host's states (and counters) written into the device batch, then refresh() rebuilds the cache
    to_device(d.states, h.v.states, d.s);
    to_device(d.steps, h.v.steps, d.s);
    to_device(d.draws, h.v.draws, d.s);
    env.refresh(d.b, d.out.count);
    host.count(h.b, h.v.count.data());
    ASSERT_EQ(to_host<i32>(d.count, N, d.s), h.v.count);
    for (int t = 0; t < 20; ++t)
    {
        env.step(d.b, d.out);
        host.step(h.b, h.out);
        expect_equal(d.snapshot(), h.v, "after refresh", t);
        if (HasFatalFailure())
            return;
    }
    EXPECT_NO_THROW(env.check_errors());
}

TEST(DeviceEnvStep, Refusals)
{
    SKIP_WITHOUT_GPU();
    auto ctx = context();
    const auto numeric = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-counters.txt");
    EXPECT_NO_THROW(cuda::DeviceEnv(ctx, rl::TaskTable::single(numeric), {}));
    const auto lazy = load("gripper__prob05", false);
    EXPECT_FALSE(cuda::DeviceEnv::fast_unsupported(*rl::TaskTable::single(lazy), {}).empty());
    EXPECT_THROW(cuda::DeviceEnv(ctx, rl::TaskTable::single(lazy), {}, cuda::DeviceEnv::Path::Fast), std::invalid_argument);
    cuda::DeviceEnv env(ctx, rl::TaskTable::single(lazy), {});
    EXPECT_FALSE(env.fast());
    const auto axioms = load("philosophers__p03-phil4", true);
    EXPECT_NE(cuda::DeviceEnv::fast_unsupported(*rl::TaskTable::single(axioms), {}).find("axioms"), std::string::npos);
    const auto frozen = load("gripper__prob05", true);
    cuda::DeviceEnv fast(ctx, rl::TaskTable::single(frozen), {});
    ASSERT_TRUE(fast.fast());
    DeviceBatch d(ctx, fast, 4, 0);
    rl::EnvBatch no_cache = d.b;
    no_cache.counts = nullptr;
    EXPECT_THROW(fast.step(no_cache, d.out), std::invalid_argument);
    rl::EnvBatch wide = d.b;
    wide.words += 1;
    EXPECT_THROW(fast.reset(wide), std::invalid_argument);
}
