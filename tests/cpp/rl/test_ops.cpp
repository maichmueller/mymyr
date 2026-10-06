// Host RL helpers (rl/novelty.hpp, rl/prefix.hpp, rl/her.hpp):
//   - novelty_update: r_int > 0 is the verdict of a width-1 NoveltyTable fed each environment's states in order, and
//     r_int counts the atoms the table had not seen;
//   - prefix_masks / schema_masks equal the masks computed from the expanded labels one by one, for every prefix of a
//     taken action and for prefixes no action extends;
//   - her_relabel: sources inside the episode window of the strategy, goals are (subsets of) the source's atoms,
//     achieved / reward are the goal test, and a relabel depends on (seed, env id, t, j) only (not on the batch).

#include "../support/suite.hpp"
#include "mymyr/novelty/novelty_table.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/her.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <set>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
TaskPtr load(const std::string& name)
{
    TaskOptions o;
    o.atoms = TaskOptions::Atoms::Frozen;
    return Task::from_text_file(task_path(name), o);
}

/// T steps of N random-policy environments (max_steps 6, autoreset): the reached states [T, N, RW] and done [T, N].
struct Window
{
    u32 RW = 0, W = 0;
    std::vector<u64> states;
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
        env.step(b, o);
        w.states.insert(w.states.end(), finals.begin(), finals.end());
        for (u64 i = 0; i < N; ++i)
            w.done.push_back(term[i] || trunc[i] ? 1 : 0);
    }
    return w;
}

// ------------------------------------------------------------------------------------------------ novelty

class NoveltyOps : public ::testing::TestWithParam<std::string>
{
};

TEST_P(NoveltyOps, VerdictsEqualAWidthOneNoveltyTable)
{
    const TaskPtr task = load(GetParam());
    const u64 N = 16, T = 40;
    const Window w = rollout(task, N, T, 3);
    std::vector<u64> seen(N * w.W, 0);
    std::vector<novelty::NoveltyTable> tables;
    for (u64 i = 0; i < N; ++i)
        tables.emplace_back(1, w.W * 64);
    std::vector<i32> r(N);
    u64 novel = 0;
    for (u64 t = 0; t < T; ++t)
    {
        const u64* rows = w.states.data() + t * N * w.RW;
        std::vector<u64> before = seen;
        rl::novelty_update(seen.data(), rows, N, w.W, w.RW, r.data());
        for (u64 i = 0; i < N; ++i)
        {
            const u64* s = rows + i * w.RW;
            const bool verdict = tables[i].mark_state(s, w.W);
            EXPECT_EQ(r[i] > 0, verdict) << t << " " << i;
            i32 fresh = 0;
            for (u32 k = 0; k < w.W; ++k)
            {
                fresh += std::popcount(s[k] & ~before[i * w.W + k]);
                EXPECT_EQ(seen[i * w.W + k], before[i * w.W + k] | s[k]);
            }
            EXPECT_EQ(r[i], fresh);
            novel += verdict ? 1 : 0;
        }
    }
    EXPECT_GT(novel, N);  // the first states are novel, and some later ones
}

INSTANTIATE_TEST_SUITE_P(Suite, NoveltyOps,
                         ::testing::Values("blocks__probBLOCKS-8-0", "gripper__prob05", "sokoban-opt08-strips__p14",
                                           "openstacks-opt08-adl__p03"));

// ------------------------------------------------------------------------------------------------ prefix masks

struct Labels
{
    std::vector<i32> parent, schema, binding, offsets;
    u32 L = 0;
    u64 total = 0;
};

Labels expand_labels(const rl::TaskTable& table, const u64* rows, u64 n, u32 W, u32 NN)
{
    Labels x;
    x.L = std::max<u32>(1, table.label_width());
    rl::Expansion e;
    e.words = W;
    e.numeric_words = NN;
    e.label_width = x.L;
    x.offsets.resize(n + 1);
    e.offsets = x.offsets.data();
    rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, nullptr, e);
    x.total = e.total;
    std::vector<u64> succ(std::max<u64>(e.total, 1) * (W + NN));
    x.parent.resize(std::max<u64>(e.total, 1));
    x.schema.resize(std::max<u64>(e.total, 1));
    x.binding.resize(std::max<u64>(e.total, 1) * x.L);
    e.capacity = e.total;
    e.succ = succ.data();
    e.parent = x.parent.data();
    e.schema = x.schema.data();
    e.binding = x.binding.data();
    rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, nullptr, e);
    return x;
}

class PrefixOps : public ::testing::TestWithParam<std::string>
{
};

TEST_P(PrefixOps, MasksEqualTheExpandedLabels)
{
    const TaskPtr task = load(GetParam());
    const u64 N = 24, T = 6;
    const Window w = rollout(task, N, T, 5);
    const u64* rows = w.states.data() + (T - 1) * N * w.RW;
    const Labels x = expand_labels(*rl::TaskTable::single(task), rows, N, w.W, task->numeric_words());
    const u32 n = task->num_objects();
    const u32 S = task->num_schemas();
    const rl::LabelRows lr{x.total, x.parent.data(), x.schema.data(), x.binding.data(), x.L};
    // schema masks
    std::vector<u8> sm(N * S);
    rl::schema_masks(lr, N, S, sm.data());
    for (u64 i = 0; i < N; ++i)
    {
        std::set<i32> want;
        for (i32 j = x.offsets[i]; j < x.offsets[i + 1]; ++j)
            want.insert(x.schema[static_cast<u64>(j)]);
        for (u32 s = 0; s < S; ++s)
            EXPECT_EQ(sm[i * S + s], want.contains(static_cast<i32>(s)) ? 1 : 0);
    }
    // the prefixes of each state's k-th successor label (k = i mod count), and of a label no successor extends
    std::vector<i32> schema(N, 0), prefix(N * x.L, -1);
    u32 arity = 0;
    for (u64 i = 0; i < N; ++i)
    {
        const i32 c = x.offsets[i + 1] - x.offsets[i];
        if (c == 0)
            continue;
        const u64 j = static_cast<u64>(x.offsets[i] + static_cast<i32>(i) % c);
        schema[i] = x.schema[j];
        std::copy_n(x.binding.begin() + static_cast<std::ptrdiff_t>(j * x.L), x.L, prefix.begin() + static_cast<std::ptrdiff_t>(i * x.L));
        if (i % 5 == 4)
            prefix[i * x.L] = static_cast<i32>(n);  // no object has this id: the masks past depth 0 are empty
        arity = std::max<u32>(arity, static_cast<u32>(task->data().schemas[static_cast<u32>(x.schema[j])].arity()));
    }
    for (u32 depth = 0; depth <= arity && depth < x.L; ++depth)
    {
        const rl::PrefixQuery q{N, schema.data(), prefix.data(), x.L, depth, n};
        std::vector<u8> m(N * n);
        rl::prefix_masks(lr, q, m.data());
        for (u64 i = 0; i < N; ++i)
        {
            std::set<i32> want;
            for (i32 jj = x.offsets[i]; jj < x.offsets[i + 1]; ++jj)
            {
                const u64 j = static_cast<u64>(jj);
                if (x.schema[j] != schema[i])
                    continue;
                bool ok = true;
                for (u32 d = 0; d < depth; ++d)
                    ok = ok && x.binding[j * x.L + d] == prefix[i * x.L + d];
                if (ok && x.binding[j * x.L + depth] >= 0)
                    want.insert(x.binding[j * x.L + depth]);
            }
            for (u32 o = 0; o < n; ++o)
                EXPECT_EQ(m[i * n + o], want.contains(static_cast<i32>(o)) ? 1 : 0) << i << " " << depth << " " << o;
            // the taken action's own object is always in its mask (unless the prefix was spoiled)
            const i32 own = prefix[i * x.L + depth];
            if (x.offsets[i + 1] > x.offsets[i] && i % 5 != 4 && own >= 0)
            {
                EXPECT_EQ(m[i * n + static_cast<u32>(own)], 1);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Suite, PrefixOps,
                         ::testing::Values("blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "depot__p02",
                                           "sokoban-opt08-strips__p14", "miconic-simpleadl__s10-2", "freecell__p02"));

// ------------------------------------------------------------------------------------------------ HER

struct Relabels
{
    std::vector<u64> goal;
    std::vector<i32> source;
    std::vector<u8> achieved;
    std::vector<f32> reward;
};

Relabels relabel(const Window& w, u64 T, u64 N, const rl::HerConfig& c, const u64* atoms = nullptr,
                 const u32* env_ids = nullptr, u64 column = 0, u64 columns = 0)
{
    // columns [column, column + columns) of the window (all of them when columns == 0)
    const u64 n = columns ? columns : N;
    std::vector<u64> states(T * n * w.RW);
    std::vector<u8> done(T * n);
    for (u64 t = 0; t < T; ++t)
        for (u64 i = 0; i < n; ++i)
        {
            std::copy_n(w.states.begin() + static_cast<std::ptrdiff_t>((t * N + column + i) * w.RW), w.RW,
                        states.begin() + static_cast<std::ptrdiff_t>((t * n + i) * w.RW));
            done[t * n + i] = w.done[t * N + column + i];
        }
    Relabels r;
    r.goal.assign(T * n * c.k * w.W, 0);
    r.source.assign(T * n * c.k, 0);
    r.achieved.assign(T * n * c.k, 0);
    r.reward.assign(T * n * c.k, 0);
    rl::HerBatch b;
    b.steps = T;
    b.envs = n;
    b.words = w.W;
    b.row_words = w.RW;
    b.states = states.data();
    b.done = done.data();
    b.goal_atoms = atoms;
    b.env_ids = env_ids;
    b.goal = r.goal.data();
    b.source = r.source.data();
    b.achieved = r.achieved.data();
    b.reward = r.reward.data();
    rl::her_relabel(b, c);
    return r;
}

TEST(HerOps, RelabelsFollowTheStrategyAndTheGoalTest)
{
    const TaskPtr task = load("gripper__prob05");
    const u64 N = 12, T = 30;
    const Window w = rollout(task, N, T, 7);
    std::vector<u64> atoms(w.W, 0);
    for (u32 k = 0; k < w.W; ++k)
        atoms[k] = 0x5555'5555'5555'5555ull << (k % 2);
    for (const rl::HerStrategy st : {rl::HerStrategy::Future, rl::HerStrategy::Final, rl::HerStrategy::Episode})
        for (const u32 subset : {0u, 1u, 3u})
            for (const bool restrict : {false, true})
            {
                rl::HerConfig c;
                c.strategy = st;
                c.k = 4;
                c.subset = subset;
                c.seed = 99;
                c.step_reward = -1.0f;
                c.goal_reward = 10.0f;
                const Relabels r = relabel(w, T, N, c, restrict ? atoms.data() : nullptr);
                for (u64 t = 0; t < T; ++t)
                    for (u64 i = 0; i < N; ++i)
                    {
                        u64 start = t, end = t;
                        while (start > 0 && !w.done[(start - 1) * N + i])
                            --start;
                        while (end + 1 < T && !w.done[end * N + i])
                            ++end;
                        for (u32 j = 0; j < c.k; ++j)
                        {
                            const u64 o = (t * N + i) * c.k + j;
                            const u64 src = static_cast<u64>(r.source[o]);
                            const u64 lo = st == rl::HerStrategy::Future ? t : st == rl::HerStrategy::Final ? end : start;
                            EXPECT_TRUE(src >= lo && src <= end) << t << " " << i;
                            const u64* s = w.states.data() + (src * N + i) * w.RW;
                            const u64* g = r.goal.data() + o * w.W;
                            const u64* reached = w.states.data() + (t * N + i) * w.RW;
                            u32 size = 0, cand = 0;
                            bool ok = true;
                            for (u32 k = 0; k < w.W; ++k)
                            {
                                const u64 a = restrict ? atoms[k] : ~u64{0};
                                EXPECT_EQ(g[k] & ~(s[k] & a), 0u);  // a subset of the source's (eligible) atoms
                                size += static_cast<u32>(std::popcount(g[k]));
                                cand += static_cast<u32>(std::popcount(s[k] & a));
                                ok = ok && (reached[k] & g[k]) == g[k];
                            }
                            EXPECT_EQ(size, subset ? std::min(subset, cand) : cand);
                            EXPECT_EQ(r.achieved[o], ok ? 1 : 0);
                            EXPECT_EQ(r.reward[o], ok ? 9.0f : -1.0f);
                            if (src == t)
                            {
                                EXPECT_TRUE(ok);  // the transition's own reached state
                            }
                        }
                    }
            }
}

TEST(HerOps, RelabelsDoNotDependOnTheBatch)
{
    const TaskPtr task = load("blocks__probBLOCKS-8-0");
    const u64 N = 20, T = 25;
    const Window w = rollout(task, N, T, 2);
    rl::HerConfig c;
    c.k = 3;
    c.subset = 2;
    c.seed = 5;
    const Relabels whole = relabel(w, T, N, c);
    const Relabels again = relabel(w, T, N, c);
    EXPECT_EQ(whole.goal, again.goal);
    // columns 7..14 as their own batch, with their env ids
    std::vector<u32> ids(8);
    for (u32 i = 0; i < 8; ++i)
        ids[i] = 7 + i;
    const Relabels part = relabel(w, T, N, c, nullptr, ids.data(), 7, 8);
    rl::HerConfig c2 = c;
    c2.first_env = 7;
    const Relabels part2 = relabel(w, T, N, c2, nullptr, nullptr, 7, 8);
    EXPECT_EQ(part.goal, part2.goal);
    for (u64 t = 0; t < T; ++t)
        for (u64 i = 0; i < 8; ++i)
            for (u32 j = 0; j < c.k; ++j)
            {
                const u64 a = (t * N + 7 + i) * c.k + j, b = (t * 8 + i) * c.k + j;
                EXPECT_EQ(whole.source[a], part.source[b]);
                for (u32 k = 0; k < w.W; ++k)
                    EXPECT_EQ(whole.goal[a * w.W + k], part.goal[b * w.W + k]);
            }
    // another seed draws other relabels
    c.seed = 6;
    EXPECT_NE(relabel(w, T, N, c).goal, whole.goal);
}
}  // namespace
