// Host environments (rl/env.hpp) and the counter-based RNG (rl/rng.hpp):
//   - Philox-4x32-10 equals Random123's known-answer vectors; successor_index stays in range and is close to uniform;
//   - the random policy moves to successor rng::successor_index(seed, env, draw, count) of the canonical order
//     (rl::expand's rows and labels), with goal flags, rewards and counts as documented;
//   - truncation, autoreset (final states), goal termination along a BrFS plan given as actions, dead ends and stuck
//     environments (pegsol), invalid actions;
//   - a ThreadPool gives the same results, and an environment's trajectory does not depend on its batch.

#include "../support/suite.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/rng.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
TaskPtr load(const std::string& name, TaskOptions::Atoms atoms = TaskOptions::Atoms::Frozen)
{
    TaskOptions o;
    o.atoms = atoms;
    return Task::from_text_file(task_path(name), o);
}

/// Host arrays of a batch and of one step's outputs.
struct Batch
{
    u32 RW = 0, L = 0;
    std::vector<u64> states, final_states;
    std::vector<i32> steps, count, schema, binding;
    std::vector<u64> draws;
    std::vector<f32> reward;
    std::vector<u8> terminated, truncated, invalid, goal;
    rl::EnvBatch b;
    rl::StepOutputs out;

    Batch(const rl::HostEnv& env, u64 rows, u64 first_env = 0)
    {
        RW = env.words() + env.numeric_words();
        L = std::max<u32>(1, env.suite()->label_width());
        states.assign(rows * RW, 0);
        final_states.assign(rows * RW, 0);
        steps.assign(rows, 0);
        count.assign(rows, 0);
        schema.assign(rows, 0);
        binding.assign(rows * L, 0);
        draws.assign(rows, 0);
        reward.assign(rows, 0);
        terminated.assign(rows, 0);
        truncated.assign(rows, 0);
        invalid.assign(rows, 0);
        goal.assign(rows, 0);
        b.states = states.data();
        b.rows = rows;
        b.words = env.words();
        b.numeric_words = env.numeric_words();
        b.steps = steps.data();
        b.draws = draws.data();
        b.first_env = first_env;
        out.reward = reward.data();
        out.terminated = terminated.data();
        out.truncated = truncated.data();
        out.count = count.data();
        out.final_states = final_states.data();
        out.schema = schema.data();
        out.binding = binding.data();
        out.label_width = L;
        out.invalid = invalid.data();
        out.goal = goal.data();
    }
    [[nodiscard]] const u64* row(u64 i) const { return states.data() + i * RW; }
};

/// The flat expansion of rows (canonical order, goal flags, labels).
struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> schema, binding, offsets;
    std::vector<u8> goal;
};

/// rows are RW = W + NN words wide (NN: the table's numeric words), of instances task_ids (null: instance 0).
Flat expand_rows(const rl::TaskSuite& table, const u64* rows, u64 n, u32 RW, u32 L, const i32* task_ids = nullptr)
{
    const u32 NN = table.numeric_words(), W = RW - NN;
    Flat f;
    rl::Expansion x;
    x.words = W;
    x.numeric_words = NN;
    x.label_width = L;
    f.offsets.resize(n + 1);
    x.offsets = f.offsets.data();
    rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, task_ids, x);
    f.succ.resize(std::max<u64>(x.total, 1) * RW);
    f.schema.resize(std::max<u64>(x.total, 1));
    f.binding.resize(std::max<u64>(x.total, 1) * L);
    f.goal.resize(std::max<u64>(x.total, 1));
    x.capacity = x.total;
    x.succ = f.succ.data();
    x.schema = f.schema.data();
    x.binding = f.binding.data();
    x.goal = f.goal.data();
    rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, task_ids, x);
    return f;
}

/// Checks one random-policy step of rows [0, n) against rl::expand of the rows before it.
void check_random_step(const rl::HostEnv& env, const Batch& x, const std::vector<u64>& before,
                       const std::vector<u64>& draws, u64 first_env, u64 n)
{
    const rl::EnvConfig& cfg = env.config();
    const Flat f = expand_rows(*env.suite(), before.data(), n, x.RW, x.L);
    for (u64 i = 0; i < n; ++i)
    {
        const i32 c = f.offsets[i + 1] - f.offsets[i];
        ASSERT_GT(c, 0);
        const u64 j = static_cast<u64>(f.offsets[i]) + rl::rng::successor_index(cfg.seed, first_env + i, draws[i], static_cast<u32>(c));
        EXPECT_EQ(x.draws[i], draws[i] + 1);
        EXPECT_TRUE(std::equal(f.succ.begin() + static_cast<std::ptrdiff_t>(j * x.RW),
                               f.succ.begin() + static_cast<std::ptrdiff_t>((j + 1) * x.RW),
                               x.final_states.begin() + static_cast<std::ptrdiff_t>(i * x.RW)));
        EXPECT_EQ(x.schema[i], f.schema[j]);
        for (u32 k = 0; k < x.L; ++k)
            EXPECT_EQ(x.binding[i * x.L + k], f.binding[j * x.L + k]);
        EXPECT_EQ(x.goal[i], f.goal[j]);
        EXPECT_EQ(x.invalid[i], 0);
        const Flat g = expand_rows(*env.suite(), x.final_states.data() + i * x.RW, 1, x.RW, x.L);
        const i32 c2 = g.offsets[1];
        const bool dead = !f.goal[j] && c2 == 0;
        EXPECT_EQ(x.terminated[i], f.goal[j] || dead ? 1 : 0);
        EXPECT_EQ(x.reward[i], cfg.step_reward + (f.goal[j] ? cfg.goal_reward : 0.0f) + (dead ? cfg.dead_end_reward : 0.0f));
        const bool trunc = !x.terminated[i] && cfg.max_steps && x.truncated[i];
        if (!x.terminated[i] && !trunc)
        {
            EXPECT_EQ(x.count[i], c2);
            EXPECT_TRUE(std::equal(x.row(i), x.row(i) + x.RW, x.final_states.begin() + static_cast<std::ptrdiff_t>(i * x.RW)));
        }
        else
            EXPECT_EQ(x.count[i], static_cast<i32>(env.initial_count(0)));
    }
}
}  // namespace

TEST(Rng, PhiloxKnownAnswers)
{
    // Random123 kat_vectors: philox4x32 10
    struct Kat
    {
        std::array<u32, 4> ctr;
        std::array<u32, 2> key;
        std::array<u32, 4> out;
    };
    const Kat kats[] = {
        {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    for (const Kat& k : kats)
    {
        const rl::rng::Block4 r = rl::rng::philox4x32_10(rl::rng::Block4{{k.ctr[0], k.ctr[1], k.ctr[2], k.ctr[3]}}, k.key[0], k.key[1]);
        for (u32 j = 0; j < 4; ++j)
            EXPECT_EQ(r.v[j], k.out[j]) << "word " << j;
    }
    // the draw layout: counter (draw lo, draw hi, env, purpose), key (seed lo, seed hi)
    const rl::rng::Block4 a = rl::rng::block(0x0123456789abcdefull, 7, 0x100000002ull, rl::rng::k_successor);
    const rl::rng::Block4 b = rl::rng::philox4x32_10(rl::rng::Block4{{2, 1, 7, 0}}, 0x89abcdefu, 0x01234567u);
    for (u32 j = 0; j < 4; ++j)
        EXPECT_EQ(a.v[j], b.v[j]);
}

TEST(Rng, SuccessorIndexIsInRangeAndUniform)
{
    for (u32 n : {1u, 2u, 3u, 7u, 1000u, 0xFFFFFFFFu})
        for (u64 d = 0; d < 200; ++d)
            EXPECT_LT(rl::rng::successor_index(42, d % 13, d, n), n);
    EXPECT_EQ(rl::rng::below(~u64{0}, 10), 9u);
    EXPECT_EQ(rl::rng::below(0, 10), 0u);
    // 7 buckets over 70000 draws of one stream and over 70000 streams: chi-square (6 dof) well below 30
    for (int mode = 0; mode < 2; ++mode)
    {
        std::array<u64, 7> h{};
        for (u64 d = 0; d < 70000; ++d)
            ++h[mode ? rl::rng::successor_index(3, d, 0, 7) : rl::rng::successor_index(3, 0, d, 7)];
        double chi = 0;
        for (u64 c : h)
            chi += (static_cast<double>(c) - 10000.0) * (static_cast<double>(c) - 10000.0) / 10000.0;
        EXPECT_LT(chi, 30.0) << "mode " << mode;
    }
}

TEST(HostEnv, RandomPolicyFollowsTheCanonicalOrder)
{
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0", "depot__p02", "miconic-simpleadl__s10-2"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        rl::EnvConfig cfg;
        cfg.seed = 0x5eed;
        cfg.goal_reward = 10.0f;
        cfg.dead_end_reward = -7.0f;
        rl::HostEnv env(rl::TaskTable::single(task), cfg);
        Batch x(env, 24, 100);
        env.reset(x.b, nullptr, x.count.data());
        for (i32 c : x.count)
            ASSERT_EQ(c, static_cast<i32>(env.initial_count(0)));
        for (int t = 0; t < 30; ++t)
        {
            const std::vector<u64> before = x.states;
            const std::vector<u64> draws = x.draws;
            env.step(x.b, x.out);
            check_random_step(env, x, before, draws, 100, 24);
            for (u64 i = 0; i < 24; ++i)
                EXPECT_EQ(x.truncated[i], 0);
        }
    }
}

TEST(HostEnv, TruncationAndAutoreset)
{
    const auto task = load("blocks__probBLOCKS-8-0");
    rl::EnvConfig cfg;
    cfg.max_steps = 5;
    rl::HostEnv env(rl::TaskTable::single(task), cfg);
    Batch x(env, 16);
    env.reset(x.b);
    for (int t = 1; t <= 12; ++t)
    {
        env.step(x.b, x.out);
        for (u64 i = 0; i < 16; ++i)
        {
            EXPECT_EQ(x.terminated[i], 0);  // blocks: no dead ends, and the goal is 20+ steps away
            EXPECT_EQ(x.truncated[i], t % 5 == 0 ? 1 : 0);
            EXPECT_EQ(x.steps[i], t % 5);
            if (t % 5 == 0)
            {
                EXPECT_TRUE(std::equal(x.row(i), x.row(i) + x.RW, env.suite()->instance(0).init.begin()));
            }
        }
    }
    // without autoreset the caller resets (TorchRL's convention): states stay at the reached state
    cfg.autoreset = false;
    rl::HostEnv manual(rl::TaskTable::single(task), cfg);
    Batch y(manual, 4);
    manual.reset(y.b);
    for (int t = 0; t < 5; ++t)
        manual.step(y.b, y.out);
    std::vector<u8> mask{1, 0, 1, 0};
    for (u64 i = 0; i < 4; ++i)
    {
        EXPECT_EQ(y.truncated[i], 1);
        EXPECT_EQ(y.steps[i], 5);
        EXPECT_TRUE(std::equal(y.row(i), y.row(i) + y.RW, y.final_states.begin() + static_cast<std::ptrdiff_t>(i * y.RW)));
    }
    manual.reset(y.b, mask.data(), y.count.data());
    EXPECT_EQ(y.steps[0], 0);
    EXPECT_EQ(y.steps[1], 5);
    EXPECT_TRUE(std::equal(y.row(0), y.row(0) + y.RW, manual.suite()->instance(0).init.begin()));
}

TEST(HostEnv, PlanActionsReachTheGoal)
{
    for (const char* name : {"gripper__prob05", "depot__p02", "philosophers__p03-phil4"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const BrfsResult r = brfs(*task, {.stop_at_goal = true});
        ASSERT_TRUE(r.solved);
        rl::EnvConfig cfg;
        cfg.goal_reward = 100.0f;
        rl::HostEnv env(rl::TaskTable::single(task), cfg);
        Batch x(env, 3);
        env.reset(x.b);
        for (usize k = 0; k < r.plan.size(); ++k)
        {
            // the plan action's index in the canonical order of the current state
            const Flat f = expand_rows(*env.suite(), x.states.data(), 3, x.RW, x.L);
            std::vector<i64> act(3, -1);
            for (u64 i = 0; i < 3; ++i)
                for (i32 j = f.offsets[i]; j < f.offsets[i + 1]; ++j)
                {
                    bool same = f.schema[static_cast<usize>(j)] == static_cast<i32>(r.plan[k].schema.v);
                    for (usize a = 0; a < r.plan[k].binding.size() && same; ++a)
                        same = f.binding[static_cast<usize>(j) * x.L + a] == static_cast<i32>(r.plan[k].binding[a].v);
                    if (same)
                        act[i] = j - f.offsets[i];
                }
            ASSERT_GE(act[0], 0) << "plan step " << k;
            env.step(x.b, x.out, rl::Actions{act.data()});
            const bool last = k + 1 == r.plan.size();
            for (u64 i = 0; i < 3; ++i)
            {
                EXPECT_EQ(x.goal[i], last ? 1 : 0);
                EXPECT_EQ(x.terminated[i], last ? 1 : 0);
                EXPECT_EQ(x.reward[i], last ? 99.0f : -1.0f);
                EXPECT_EQ(x.schema[i], static_cast<i32>(r.plan[k].schema.v));
            }
        }
        for (u64 i = 0; i < 3; ++i)
            EXPECT_EQ(x.steps[i], 0);  // reset after the goal
    }
}

TEST(HostEnv, DeadEndsStuckAndInvalidActions)
{
    const auto task = load("pegsol-08-strips__p22");
    rl::EnvConfig cfg;
    cfg.autoreset = false;
    cfg.dead_end_reward = -50.0f;
    rl::HostEnv env(rl::TaskTable::single(task), cfg);
    Batch x(env, 64);
    env.reset(x.b, nullptr, x.count.data());
    std::vector<u8> done(64, 0);
    u64 dead_ends = 0;
    for (int t = 0; t < 40; ++t)
    {
        const std::vector<u64> before = x.states;
        const std::vector<u8> was_done = done;
        env.step(x.b, x.out);
        for (u64 i = 0; i < 64; ++i)
        {
            if (was_done[i])
            {
                // stuck: no move, terminal, the dead-end reward again
                EXPECT_TRUE(std::equal(x.row(i), x.row(i) + x.RW, before.begin() + static_cast<std::ptrdiff_t>(i * x.RW)));
                EXPECT_EQ(x.terminated[i], 1);
                EXPECT_EQ(x.schema[i], -1);
                EXPECT_EQ(x.reward[i], -51.0f);
                EXPECT_EQ(x.count[i], 0);
                continue;
            }
            if (x.terminated[i] && !x.goal[i])
            {
                ++dead_ends;
                EXPECT_EQ(x.count[i], 0);
                EXPECT_EQ(x.reward[i], -51.0f);
                done[i] = 1;
            }
            else if (x.terminated[i])
                done[i] = 1;
        }
    }
    EXPECT_GT(dead_ends, 0u);
    // invalid actions: no move, not terminal, flagged
    Batch y(env, 4);
    env.reset(y.b, nullptr, y.count.data());
    const std::vector<i64> act{-1, y.count[1], 1000000, 0};
    env.step(y.b, y.out, rl::Actions{act.data()});
    for (u64 i = 0; i < 3; ++i)
    {
        EXPECT_EQ(y.invalid[i], 1);
        EXPECT_EQ(y.terminated[i], 0);
        EXPECT_EQ(y.reward[i], -1.0f);
        EXPECT_EQ(y.schema[i], -1);
        EXPECT_EQ(y.steps[i], 1);
        EXPECT_TRUE(std::equal(y.row(i), y.row(i) + y.RW, env.suite()->instance(0).init.begin()));
    }
    EXPECT_EQ(y.invalid[3], 0);
    EXPECT_EQ(y.draws[0], 1u);  // the draws advance with actions too
}

TEST(HostEnv, DeadEndModes)
{
    // pegsol reaches dead ends quickly: compare the three dead-end settings on the same random trajectories
    const auto task = load("pegsol-08-strips__p22");
    const auto table = rl::TaskTable::single(task);
    rl::EnvConfig cfg;
    cfg.autoreset = false;
    cfg.dead_end_reward = -50.0f;
    rl::EnvConfig soft = cfg, none = cfg;
    soft.dead_end_terminal = false;
    none.dead_end = rl::DeadEnd::None;
    rl::HostEnv a(table, cfg), b(table, soft), c(table, none);
    Batch x(a, 32), y(b, 32), z(c, 32);
    a.reset(x.b);
    b.reset(y.b);
    c.reset(z.b);
    u64 dead = 0;
    for (int t = 0; t < 40; ++t)
    {
        a.step(x.b, x.out);
        b.step(y.b, y.out);
        c.step(z.b, z.out);
        ASSERT_EQ(x.states, y.states);  // the same moves: only termination and rewards differ
        ASSERT_EQ(x.states, z.states);
        for (u64 i = 0; i < 32; ++i)
        {
            const bool is_dead = x.terminated[i] && !x.goal[i];
            dead += is_dead ? 1 : 0;
            EXPECT_EQ(y.terminated[i], x.goal[i]);  // not terminal
            EXPECT_EQ(y.reward[i], x.reward[i]);    // the dead-end reward still applies
            EXPECT_EQ(z.terminated[i], x.goal[i]);
            EXPECT_EQ(z.reward[i], x.reward[i] - (is_dead ? cfg.dead_end_reward : 0.0f));  // no dead-end reward
            if (is_dead && x.count[i] == 0 && x.schema[i] == -1)
            {
                EXPECT_EQ(z.schema[i], -1);  // stuck: no move in any mode
            }
        }
    }
    EXPECT_GT(dead, 0u);
}

TEST(HostEnv, PoolAndBatchCompositionDoNotChangeTrajectories)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto task = load("logistics00__probLOGISTICS-6-1", atoms);
        rl::EnvConfig cfg;
        cfg.seed = 99;
        cfg.max_steps = 17;
        const auto table = rl::TaskTable::single(task);
        rl::HostEnv a(table, cfg), b(table, cfg), c(table, cfg);
        Batch x(a, 48), y(b, 20, 0), z(c, 28, 20);
        a.reset(x.b);
        b.reset(y.b);
        c.reset(z.b);
        ThreadPool pool(4);
        for (int t = 0; t < 40; ++t)
        {
            a.step(x.b, x.out, {}, nullptr, &pool);
            b.step(y.b, y.out);
            c.step(z.b, z.out);
            ASSERT_TRUE(std::equal(y.states.begin(), y.states.end(), x.states.begin()));
            ASSERT_TRUE(std::equal(z.states.begin(), z.states.end(), x.states.begin() + 20 * x.RW));
            ASSERT_TRUE(std::equal(y.reward.begin(), y.reward.end(), x.reward.begin()));
            ASSERT_TRUE(std::equal(z.truncated.begin(), z.truncated.end(), x.truncated.begin() + 20));
            ASSERT_TRUE(std::equal(z.binding.begin(), z.binding.end(), x.binding.begin() + 20 * x.L));
        }
    }
}

TEST(HostEnv, NumericTasks)
{
    const auto task = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-counters.txt");
    ASSERT_GT(task->numeric_words(), 0u);
    rl::EnvConfig cfg;
    cfg.seed = 5;
    rl::HostEnv env(rl::TaskTable::single(task), cfg);
    Batch x(env, 8);
    EXPECT_EQ(x.RW, env.words() + task->numeric_words());
    env.reset(x.b);
    for (int t = 0; t < 10; ++t)
    {
        const std::vector<u64> before = x.states;
        const std::vector<u64> draws = x.draws;
        env.step(x.b, x.out);
        check_random_step(env, x, before, draws, 0, 8);  // rows [bits | numeric slots]
    }
}
