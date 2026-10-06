#pragma once
// The device env harness of the CUDA env tests (tests/cuda/test_device_table.cpp, test_device_graphs.cpp,
// test_device_task_suite.cpp): the table instance sets as tables, batches on the host (rl::HostEnv) and in device
// memory (cuda::DeviceEnv, with the fast path's cache arrays), their snapshots and comparison, and the actions and
// configuration of the tests. Needs GPU 0 (CUDA_VISIBLE_DEVICES=0); the tests skip without a device.

#include "../cpp/rl/table_instance_sets.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/env.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (mymyr::cuda::device_count() == 0)                                                                           \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

namespace mymyr::test
{
inline cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{2} << 30;  // the GPU is shared
    return cuda::DeviceContext::create(0, o);
}

struct Set
{
    std::string name;
    std::vector<TaskPtr> tasks;
    rl::TaskTablePtr table;
};

inline std::vector<Set> sets(TaskOptions::Atoms atoms)
{
    std::vector<Set> out;
    for (const InstanceSet& s : table_instance_sets())
    {
        std::vector<TaskPtr> tasks = load_set(s, atoms);
        if (tasks.empty())
        {
            ADD_FAILURE() << "missing PDDL of set " << s.name << " (MYMYR_FORK_DATA_DIR)";
            continue;
        }
        auto table = rl::TaskTable::create(tasks);
        out.push_back({s.name, std::move(tasks), std::move(table)});
    }
    return out;
}

inline const char* atoms_name(TaskOptions::Atoms a) { return a == TaskOptions::Atoms::Frozen ? " frozen" : " lazy"; }

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

template<class T>
cuda::DeviceBuffer to_device(const cuda::ContextPtr& ctx, const std::vector<T>& v, cudaStream_t s)
{
    cuda::DeviceBuffer d(ctx, std::max<u64>(v.size(), 1) * sizeof(T), s);
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(d.data(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
    return d;
}

template<class T>
void upload(void* d, const std::vector<T>& v, cudaStream_t s)
{
    if (!v.empty())
        cuda::check(cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    cuda::check(cudaStreamSynchronize(s), "sync");
}

/// One step's state and outputs of a batch, on the host (for comparisons).
struct Snapshot
{
    std::vector<u64> states, final_states, draws, gpos, gneg;
    std::vector<i32> task_ids, steps, count, schema, binding;
    std::vector<f32> reward;
    std::vector<u8> terminated, truncated, invalid, goal;
};

inline void expect_equal(const Snapshot& d, const Snapshot& h, int t)
{
    ASSERT_EQ(d.task_ids, h.task_ids) << "step " << t;
    ASSERT_EQ(d.states, h.states) << "step " << t;
    ASSERT_EQ(d.final_states, h.final_states) << "step " << t;
    ASSERT_EQ(d.steps, h.steps) << "step " << t;
    ASSERT_EQ(d.draws, h.draws) << "step " << t;
    ASSERT_EQ(d.count, h.count) << "step " << t;
    ASSERT_EQ(d.schema, h.schema) << "step " << t;
    ASSERT_EQ(d.binding, h.binding) << "step " << t;
    ASSERT_EQ(d.terminated, h.terminated) << "step " << t;
    ASSERT_EQ(d.truncated, h.truncated) << "step " << t;
    ASSERT_EQ(d.goal, h.goal) << "step " << t;
    ASSERT_EQ(d.invalid, h.invalid) << "step " << t;
    ASSERT_EQ(d.gpos, h.gpos) << "step " << t;
    ASSERT_EQ(d.gneg, h.gneg) << "step " << t;
    ASSERT_EQ(d.reward.size(), h.reward.size());
    ASSERT_EQ(std::memcmp(d.reward.data(), h.reward.data(), d.reward.size() * sizeof(f32)), 0) << "step " << t;
}

/// A batch on the host (rl::HostEnv).
struct HostEnvs
{
    Snapshot v;
    rl::EnvBatch b;
    rl::StepOutputs out;

    /// Over a table or a suite (rl::TaskTable, rl::TaskSuite: rows of its words(), labels of its label_width()).
    template<class Table>
    HostEnvs(const Table& T, const std::vector<i32>& ids, bool goals)
    {
        const u64 n = ids.size();
        const u32 W = T.words(), L = std::max<u32>(1, T.label_width());
        v.states.assign(n * W, 0);
        v.final_states.assign(n * W, 0);
        v.draws.assign(n, 0);
        v.task_ids = ids;
        v.steps.assign(n, 0);
        v.count.assign(n, 0);
        v.schema.assign(n, 0);
        v.binding.assign(n * L, 0);
        v.reward.assign(n, 0);
        v.terminated.assign(n, 0);
        v.truncated.assign(n, 0);
        v.invalid.assign(n, 0);
        v.goal.assign(n, 0);
        if (goals)
        {
            v.gpos.assign(n * W, 0);
            v.gneg.assign(n * W, 0);
        }
        bind(W);
        out = rl::StepOutputs{v.reward.data(), v.terminated.data(), v.truncated.data(), v.count.data(), v.final_states.data(),
                              v.schema.data(), v.binding.data(), L, v.invalid.data(), v.goal.data()};
    }

    void bind(u32 W)
    {
        b = rl::EnvBatch{};
        b.states = v.states.data();
        b.rows = v.task_ids.size();
        b.words = W;
        b.task_ids = v.task_ids.data();
        b.goal_pos = v.gpos.empty() ? nullptr : v.gpos.data();
        b.goal_neg = v.gneg.empty() ? nullptr : v.gneg.data();
        b.steps = v.steps.data();
        b.draws = v.draws.data();
    }
};

/// A batch in device memory (cuda::DeviceEnv), with the fast path's cache arrays.
struct DeviceEnvs
{
    u64 N;
    u32 W, L, C;
    u64 V;
    cudaStream_t s;
    cuda::DeviceBuffer states, final_states, draws, task_ids, steps, count, schema, binding, reward, terminated, truncated,
        invalid, goal, counts, views, gpos, gneg, action, next;
    bool goals;
    rl::EnvBatch b;
    rl::StepOutputs out;

    DeviceEnvs(const cuda::ContextPtr& ctx, const cuda::DeviceEnv& env, const std::vector<i32>& ids, bool with_goals)
        : N(ids.size()), W(env.words()), L(env.label_width()), C(env.cache_schemas()), V(env.cache_view_words()),
          s(env.stream()), states(ctx, N * W * 8, s), final_states(ctx, N * W * 8, s), draws(ctx, N * 8, s),
          task_ids(to_device(ctx, ids, s)), steps(ctx, N * 4, s), count(ctx, N * 4, s), schema(ctx, N * 4, s),
          binding(ctx, N * L * 4, s), reward(ctx, N * 4, s), terminated(ctx, N, s), truncated(ctx, N, s),
          invalid(ctx, N, s), goal(ctx, N, s), counts(ctx, std::max<u64>(N * C, 1) * 4, s),
          views(ctx, std::max<u64>(N * V, 1) * 8, s), gpos(ctx, N * W * 8, s), gneg(ctx, N * W * 8, s),
          action(ctx, N * 8, s), next(ctx, N * 4, s), goals(with_goals)
    {
        cuda::check(cudaMemsetAsync(draws.data(), 0, N * 8, s), "memset");
        cuda::check(cudaMemsetAsync(steps.data(), 0, N * 4, s), "memset");
        // the views cache as a caller allocates it (torch.zeros): the env writes a row's own instance's V_i words of
        // the widest V, and the row permutation below copies whole rows
        cuda::check(cudaMemsetAsync(views.data(), 0, std::max<u64>(N * V, 1) * 8, s), "memset");
        bind();
        out = rl::StepOutputs{static_cast<f32*>(reward.data()),   static_cast<u8*>(terminated.data()),
                              static_cast<u8*>(truncated.data()), static_cast<i32*>(count.data()),
                              static_cast<u64*>(final_states.data()), static_cast<i32*>(schema.data()),
                              static_cast<i32*>(binding.data()),  L,
                              static_cast<u8*>(invalid.data()),   static_cast<u8*>(goal.data())};
    }

    void bind()
    {
        b = rl::EnvBatch{};
        b.states = static_cast<u64*>(states.data());
        b.rows = N;
        b.words = W;
        b.task_ids = static_cast<i32*>(task_ids.data());
        b.goal_pos = goals ? static_cast<u64*>(gpos.data()) : nullptr;
        b.goal_neg = goals ? static_cast<u64*>(gneg.data()) : nullptr;
        b.steps = static_cast<i32*>(steps.data());
        b.draws = static_cast<u64*>(draws.data());
        b.counts = C ? static_cast<u32*>(counts.data()) : nullptr;
        b.views = V ? static_cast<u64*>(views.data()) : nullptr;
    }

    [[nodiscard]] Snapshot snapshot() const
    {
        Snapshot v;
        v.task_ids = to_host<i32>(task_ids.data(), N, s);
        v.states = to_host<u64>(states.data(), N * W, s);
        v.final_states = to_host<u64>(final_states.data(), N * W, s);
        v.draws = to_host<u64>(draws.data(), N, s);
        v.steps = to_host<i32>(steps.data(), N, s);
        v.count = to_host<i32>(count.data(), N, s);
        v.schema = to_host<i32>(schema.data(), N, s);
        v.binding = to_host<i32>(binding.data(), N * L, s);
        v.reward = to_host<f32>(reward.data(), N, s);
        v.terminated = to_host<u8>(terminated.data(), N, s);
        v.truncated = to_host<u8>(truncated.data(), N, s);
        v.invalid = to_host<u8>(invalid.data(), N, s);
        v.goal = to_host<u8>(goal.data(), N, s);
        if (goals)
        {
            v.gpos = to_host<u64>(gpos.data(), N * W, s);
            v.gneg = to_host<u64>(gneg.data(), N * W, s);
        }
        return v;
    }
};

/// Custom goals for every third row of a reset batch: the instance's goal without its first literal, plus a negative
/// literal of the initial state.
inline void custom_goals(Snapshot& v, u32 W)
{
    const u64 N = v.task_ids.size();
    for (u64 i = 0; i < N; i += 3)
    {
        u64* gp = v.gpos.data() + i * W;
        for (u32 w = 0; w < W; ++w)
            if (gp[w])
            {
                gp[w] &= gp[w] - 1;
                break;
            }
        const u64* s0 = v.states.data() + i * W;
        for (u32 w = 0; w < W; ++w)
            if (s0[w])
            {
                v.gneg[i * W + w] |= s0[w] & ~(s0[w] - 1);  // a literal of the initial state must go
                v.gneg[i * W + w] &= ~gp[w];
                break;
            }
    }
}

/// Actions of step t: an index in [0, count] from a hash (count itself is invalid; so is anything for count 0).
inline std::vector<i64> actions(const std::vector<i32>& count, int t)
{
    std::vector<i64> a(count.size());
    for (usize i = 0; i < count.size(); ++i)
        a[i] = static_cast<i64>(hash::mix64((static_cast<u64>(t) << 32) ^ i) % (static_cast<u64>(std::max(count[i], 0)) + 1));
    return a;
}

/// The tests' configuration: truncation after 9 steps, goal and dead-end rewards.
inline rl::EnvConfig config(u64 seed)
{
    rl::EnvConfig cfg;
    cfg.seed = seed;
    cfg.max_steps = 9;
    cfg.goal_reward = 5.0f;
    cfg.dead_end_reward = -3.0f;
    return cfg;
}
}  // namespace mymyr::test
