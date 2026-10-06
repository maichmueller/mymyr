// Batched expand (rl/expand.hpp) and the task-level arrays (rl/task_arrays.hpp), against the per-state successor API
// that is validated against the exhaustive oracle. Plus a concurrency check: 64 threads calling expand / apply /
// is_goal on one shared task (run under TSan in build/tsan).

#include "../support/suite.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <barrier>
#include <random>
#include <thread>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
const char* const k_tasks[] = {
    "blocks__probBLOCKS-8-0",
    "gripper__prob05",
    "logistics00__probLOGISTICS-6-1",
    "depot__p02",
    "miconic-simpleadl__s10-2",       // conditional effects
    "openstacks-opt08-adl__p03",      // axioms
    "philosophers__p03-phil4",        // axioms, derived goal
    "organic-synthesis-opt18-strips__p20",
};

TaskPtr load(const std::string& name, TaskOptions::Atoms atoms)
{
    TaskOptions o;
    o.atoms = atoms;
    return Task::from_text_file(task_path(name), o);
}

/// States of seeded random walks (canonical order, witness pruning off), the initial state first.
std::vector<State> walk_states(const Task& task, u32 walks, u32 steps, u64 seed)
{
    std::mt19937_64 rng(seed);
    Successors& succ = task.workspace().successors();
    std::vector<State> out;
    for (u32 w = 0; w < walks; ++w)
    {
        State s = task.initial_state();
        out.push_back(s);
        for (u32 k = 0; k < steps; ++k)
        {
            std::vector<State> kids;
            std::vector<u64> tmp;
            succ.prepare(s);
            succ.generate<false>(
                [&](u32, const ObjectId*, const Delta& d)
                {
                    const u32 n = apply_delta(s.data(), s.size_words(), d, tmp);
                    kids.emplace_back(tmp.data(), n);
                    return true;
                },
                false, true);
            if (kids.empty())
                break;
            s = kids[rng() % kids.size()];
            out.push_back(s);
        }
    }
    return out;
}

/// Packs states into a [N, W] row-major batch.
std::vector<u64> pack(const std::vector<State>& states, u32 W)
{
    std::vector<u64> rows(states.size() * W, 0);
    for (usize i = 0; i < states.size(); ++i)
        std::copy_n(states[i].data(), states[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * W));
    return rows;
}

struct Buffers
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    rl::Expansion x;

    Buffers(u64 rows, u64 cap, u32 W, u32 L)
        : succ(cap * W), parent(cap), schema(cap), binding(cap * L), offsets(rows + 1), goal(cap)
    {
        x.capacity = cap;
        x.words = W;
        x.label_width = L;
        x.succ = succ.data();
        x.parent = parent.data();
        x.schema = schema.data();
        x.binding = binding.data();
        x.offsets = offsets.data();
        x.goal = goal.data();
    }
};

/// Reference: per state, the applicable actions in canonical order and their successors (Successors::apply).
struct Reference
{
    std::vector<Action> actions;
    std::vector<State> succ;
    std::vector<i32> parent;
    std::vector<i32> offsets{0};
};

Reference reference(const Task& task, const std::vector<State>& states)
{
    Reference r;
    Successors& succ = task.workspace().successors();
    for (usize i = 0; i < states.size(); ++i)
    {
        std::vector<Action> acts;
        succ.prepare(states[i]);
        succ.generate<false>([&](u32 s, const ObjectId* b, const Delta&)
                             { acts.emplace_back(ActionLabel{SchemaId{s}, {b, succ.arity(s)}}); return true; },
                             false, true);
        for (const Action& a : acts)
        {
            r.actions.push_back(a);
            r.succ.push_back(succ.apply(states[i], a.label()));
            r.parent.push_back(static_cast<i32>(i));
        }
        r.offsets.push_back(static_cast<i32>(r.actions.size()));
    }
    return r;
}

void expect_equal(const Task& task, const Reference& ref, const Buffers& b, u64 rows)
{
    const u32 W = b.x.words, L = b.x.label_width;
    ASSERT_EQ(b.x.total, ref.actions.size());
    ASSERT_FALSE(b.x.overflow());
    for (u64 i = 0; i <= rows; ++i)
        ASSERT_EQ(b.offsets[i], ref.offsets[i]) << "offset " << i;
    for (usize j = 0; j < ref.actions.size(); ++j)
    {
        const Action& a = ref.actions[j];
        ASSERT_EQ(b.parent[j], ref.parent[j]);
        ASSERT_EQ(b.schema[j], static_cast<i32>(a.schema.v)) << task.format(a.label());
        for (u32 k = 0; k < L; ++k)
            ASSERT_EQ(b.binding[j * L + k], k < a.binding.size() ? static_cast<i32>(a.binding[k].v) : -1);
        const StateView got{b.succ.data() + j * W, W, nullptr, 0};
        ASSERT_TRUE(got == ref.succ[j].view()) << "successor " << j << " of " << task.format(a.label());
        ASSERT_EQ(b.goal[j], task.is_goal(ref.succ[j]) ? 1 : 0);
    }
}

class RlExpand : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(RlExpand, EqualsPerStateSuccessorsInCanonicalOrder)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    const std::vector<State> states = walk_states(*task, 3, 12, 7);
    const Reference ref = reference(*task, states);  // interns every successor atom (lazy slots) first
    const u32 W = task->words(), L = rl::max_label_width(*task);
    const std::vector<u64> rows = pack(states, W);
    const rl::StateBatchView in{rows.data(), states.size(), W, 0};

    Buffers one(states.size(), ref.actions.size() + 5, W, L);
    rl::expand(*rl::TaskTable::single(task), in, nullptr, one.x);
    expect_equal(*task, ref, one, states.size());

    for (u32 T : {2u, 3u, 8u})
    {
        ThreadPool pool(T);
        rl::ExpandScratch scratch;
        Buffers many(states.size(), ref.actions.size() + 5, W, L);
        rl::expand(*rl::TaskTable::single(task), in, nullptr, many.x, {}, pool, scratch);
        expect_equal(*task, ref, many, states.size());
        rl::expand(*rl::TaskTable::single(task), in, nullptr, many.x, {}, pool, scratch);  // scratch reuse
        expect_equal(*task, ref, many, states.size());
    }
}

TEST_P(RlExpand, CapacityOverflowCountsAndClips)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    const std::vector<State> states = walk_states(*task, 2, 8, 3);
    const Reference ref = reference(*task, states);
    const u32 W = task->words(), L = rl::max_label_width(*task);
    const std::vector<u64> rows = pack(states, W);
    const rl::StateBatchView in{rows.data(), states.size(), W, 0};
    const u64 cap = ref.actions.size() / 2;
    ThreadPool pool(4);
    rl::ExpandScratch scratch;
    for (int mt = 0; mt < 2; ++mt)
    {
        Buffers b(states.size(), std::max<u64>(cap, 1), W, L);
        b.x.capacity = cap;
        if (mt)
            rl::expand(*rl::TaskTable::single(task), in, nullptr, b.x, {}, pool, scratch);
        else
            rl::expand(*rl::TaskTable::single(task), in, nullptr, b.x);
        EXPECT_EQ(b.x.total, ref.actions.size());
        EXPECT_EQ(b.x.overflow(), cap < ref.actions.size());
        for (u64 i = 0; i <= states.size(); ++i)
            ASSERT_EQ(b.offsets[i], ref.offsets[i]);
        for (u64 j = 0; j < cap; ++j)
        {
            ASSERT_EQ(b.parent[j], ref.parent[j]);
            ASSERT_TRUE((StateView{b.succ.data() + j * W, W, nullptr, 0}) == ref.succ[j].view());
        }
    }
}

TEST_P(RlExpand, NarrowRowsReportTheWidthNeeded)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    const std::vector<State> states = walk_states(*task, 2, 8, 5);
    const Reference ref = reference(*task, states);
    u32 need = 0;
    for (const State& s : ref.succ)
        need = std::max(need, s.size_words());
    if (need < 2)
        GTEST_SKIP() << "successors fit in one word";
    const u32 W = task->words(), L = rl::max_label_width(*task);
    const std::vector<u64> rows = pack(states, W);
    Buffers b(states.size(), ref.actions.size(), need - 1, L);
    rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{rows.data(), states.size(), W, 0}, nullptr, b.x);
    EXPECT_TRUE(b.x.overflow());
    EXPECT_EQ(b.x.words_needed, need);
    EXPECT_EQ(b.x.total, ref.actions.size());
}

TEST_P(RlExpand, PaddedViewScattersTheFlatRows)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    const std::vector<State> states = walk_states(*task, 2, 10, 11);
    const Reference ref = reference(*task, states);
    const u32 W = task->words(), L = rl::max_label_width(*task);
    const std::vector<u64> rows = pack(states, W);
    Buffers b(states.size(), ref.actions.size(), W, L);
    rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{rows.data(), states.size(), W, 0}, nullptr, b.x);
    i32 maxc = 0;
    for (usize i = 0; i < states.size(); ++i)
        maxc = std::max(maxc, ref.offsets[i + 1] - ref.offsets[i]);
    for (u32 K : {static_cast<u32>(std::max(1, maxc)), static_cast<u32>(std::max(1, maxc / 2))})
    {
        const u64 N = states.size();
        std::vector<i32> index(N * K), count(N), schema(N * K), binding(N * K * L);
        std::vector<u8> mask(N * K), goal(N * K);
        std::vector<u64> succ(N * K * W);
        rl::PaddedExpansion p{K, index.data(), mask.data(), count.data(), W, succ.data(), schema.data(), L, binding.data(), goal.data()};
        rl::pad(b.x, N, p);
        EXPECT_EQ(p.overflow, static_cast<i32>(K) < maxc);
        for (u64 i = 0; i < N; ++i)
        {
            const i32 c = ref.offsets[i + 1] - ref.offsets[i];
            ASSERT_EQ(count[i], c);
            for (u32 k = 0; k < K; ++k)
            {
                const bool valid = static_cast<i32>(k) < c;
                const u64 at = i * K + k;
                ASSERT_EQ(mask[at], valid ? 1 : 0);
                ASSERT_EQ(index[at], valid ? ref.offsets[i] + static_cast<i32>(k) : -1);
                if (valid)
                {
                    const u64 j = static_cast<u64>(index[at]);
                    ASSERT_EQ(schema[at], b.schema[j]);
                    ASSERT_TRUE((StateView{succ.data() + at * W, W, nullptr, 0}) == ref.succ[j].view());
                    ASSERT_EQ(goal[at], b.goal[j]);
                }
                else
                {
                    ASSERT_EQ(schema[at], -1);
                    ASSERT_EQ(binding[at * L], L ? -1 : binding[at * L]);
                }
            }
        }
    }
}

std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const char* t : k_tasks)
        for (bool frozen : {false, true})
            out.emplace_back(t, frozen);
    return out;
}

INSTANTIATE_TEST_SUITE_P(Suite, RlExpand, ::testing::ValuesIn(params()),
                         [](const auto& info)
                         {
                             std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
                             std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                             return n;
                         });

TEST(RlExpandChecks, RejectsBitsBeyondTheAssignedSlots)
{
    const auto task = load("blocks__probBLOCKS-8-0", TaskOptions::Atoms::Lazy);
    const u32 W = task->words() + 1;
    std::vector<u64> row(W, 0);
    row[W - 1] = 1;  // a slot no atom has
    std::vector<u64> succ(64 * W);
    rl::Expansion x;
    x.capacity = 64;
    x.words = W;
    x.succ = succ.data();
    EXPECT_THROW(rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{row.data(), 1, W, 0}, nullptr, x), std::invalid_argument);
}

TEST(RlExpandChecks, StridedRowsAndRandomWalks)
{
    const auto task = load("gripper__prob05", TaskOptions::Atoms::Auto);
    const std::vector<State> states = walk_states(*task, 1, 10, 1);
    const u32 W = task->words();
    std::vector<u64> rows = pack(states, 2 * W);  // stride 2W, width W
    const Reference ref = reference(*task, states);
    Buffers b(states.size(), ref.actions.size(), W, rl::max_label_width(*task));
    rl::expand(*rl::TaskTable::single(task), rl::StateBatchView{rows.data(), states.size(), W, 2 * W}, nullptr, b.x);
    expect_equal(*task, ref, b, states.size());

    const rl::WalkStats a = rl::random_walks(*task, 500, 20, 42);
    const rl::WalkStats c = rl::random_walks(*task, 500, 20, 42);
    EXPECT_EQ(a.steps, 500u);
    EXPECT_EQ(a.successors, c.successors);
    EXPECT_GT(a.successors, a.steps);
}

// ------------------------------------------------------------------------------------------------ task arrays

class RlTaskArrays : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(RlTaskArrays, AtomMetadataMatchesTheIndex)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    (void)walk_states(*task, 2, 10, 3);  // assign some lazy slots
    const rl::ArrayBundle m = rl::atom_metadata(*task);
    const u32 F = static_cast<u32>(m.scalar("num_atoms"));
    EXPECT_EQ(F, task->atoms().fluent_slots());
    const auto* pred = static_cast<const i32*>(m.data(*m.find("atom_pred")));
    const auto* off = static_cast<const i32*>(m.data(*m.find("atom_args_offsets")));
    const auto* args = static_cast<const i32*>(m.data(*m.find("atom_args")));
    const auto* cid = static_cast<const i32*>(m.data(*m.find("atom_cid")));
    const rl::ArrayInfo& padded_info = *m.find("atom_args_padded");
    const auto* padded = static_cast<const i32*>(m.data(padded_info));
    const i64 A = padded_info.shape[1];
    for (u32 s = 0; s < F; ++s)
    {
        ASSERT_EQ(pred[s], static_cast<i32>(task->atoms().predicate(SlotId{s}).v));
        const auto a = task->atoms().arguments(SlotId{s});
        ASSERT_EQ(off[s + 1] - off[s], static_cast<i32>(a.size()));
        for (usize k = 0; k < a.size(); ++k)
        {
            ASSERT_EQ(args[off[s] + static_cast<i32>(k)], static_cast<i32>(a[k]));
            ASSERT_EQ(padded[s * A + static_cast<i64>(k)], static_cast<i32>(a[k]));
        }
        ASSERT_EQ(static_cast<u64>(cid[s]), task->atoms().canonical(AtomKind::Fluent, s));
    }
    // predicate groups partition the slots
    const auto* gs = static_cast<const i32*>(m.data(*m.find("pred_slots")));
    const auto* go = static_cast<const i32*>(m.data(*m.find("pred_slot_offsets")));
    const i64 P = m.scalar("num_predicates");
    EXPECT_EQ(go[P], static_cast<i32>(F));
    for (i64 p = 0; p < P; ++p)
        for (i32 j = go[p]; j < go[p + 1]; ++j)
            ASSERT_EQ(pred[gs[j]], p);
    EXPECT_EQ(m.scalar("num_static_atoms"), static_cast<i64>(task->data().static_init.size()));
}

TEST_P(RlTaskArrays, GoalMasksAgreeWithIsGoalAndLayoutEncodesCids)
{
    const auto& [name, frozen] = GetParam();
    const auto task = load(name, frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy);
    const std::vector<State> states = walk_states(*task, 3, 15, 9);
    const rl::ArrayBundle d = rl::device_arrays(*task, 1);  // the version-1 layout stays available (version 2 adds to it)
    EXPECT_EQ(d.scalar("version"), 1);
    const rl::ArrayInfo& gp = *d.find("goal_pos");
    ASSERT_TRUE(gp.words);
    const u32 GW = static_cast<u32>(gp.shape[0]);
    const auto* pos = static_cast<const u64*>(d.data(gp));
    const auto* neg = static_cast<const u64*>(d.data(*d.find("goal_neg")));
    if (d.scalar("goal_uses_derived") == 0)
    {
        const u32 W = task->words();
        const std::vector<u64> rows = pack(states, W);
        std::vector<u8> flags(states.size());
        std::vector<i32> counts(states.size());
        rl::goal_test(rl::StateBatchView{rows.data(), states.size(), W, 0}, rl::StateBatchView{pos, 1, GW, 0},
                      rl::StateBatchView{neg, 1, GW, 0}, flags.data());
        rl::goal_count(rl::StateBatchView{rows.data(), states.size(), W, 0}, rl::StateBatchView{pos, 1, GW, 0},
                       rl::StateBatchView{neg, 1, GW, 0}, counts.data());
        for (usize i = 0; i < states.size(); ++i)
        {
            ASSERT_EQ(flags[i], task->is_goal(states[i]) ? 1 : 0);
            ASSERT_EQ(counts[i] == 0, flags[i] == 1);
        }
    }
    const auto* init = static_cast<const u64*>(d.data(*d.find("init")));
    EXPECT_TRUE((StateView{init, static_cast<u32>(d.find("init")->shape[0]), nullptr, 0}) == task->initial_state().view());

    // canonical ids from the exported layout
    const auto* poff = static_cast<const i32*>(d.data(*d.find("pred_offset")));
    const auto* pbeg = static_cast<const i32*>(d.data(*d.find("pos_begin")));
    const auto* pstr = static_cast<const i32*>(d.data(*d.find("pos_stride")));
    const auto* prank = static_cast<const i32*>(d.data(*d.find("pos_rank")));
    const u32 n = task->num_objects();
    const u32 F = static_cast<u32>(d.scalar("num_atoms"));
    for (u32 s = 0; s < F; ++s)
    {
        const u32 p = task->atoms().predicate(SlotId{s}).v;
        const auto a = task->atoms().arguments(SlotId{s});
        i64 c = poff[p];
        for (usize k = 0; k < a.size(); ++k)
        {
            const i32 r = prank[(static_cast<i64>(pbeg[p]) + static_cast<i64>(k)) * n + a[k]];
            ASSERT_GE(r, 0);
            c += static_cast<i64>(r) * pstr[pbeg[p] + static_cast<i32>(k)];
        }
        ASSERT_EQ(static_cast<u64>(c), task->atoms().canonical(AtomKind::Fluent, s));
    }
    EXPECT_THROW((void)rl::device_arrays(*task, 99), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(Suite, RlTaskArrays, ::testing::ValuesIn(params()),
                         [](const auto& info)
                         {
                             std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
                             std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                             return n;
                         });

// ------------------------------------------------------------------------------------------------ concurrency gate

/// 64 threads share one task (lazy slots, so slot interning races too) and call expand, apply and is_goal on
/// overlapping batches. Every result must equal the single-threaded reference of a second instance, compared through
/// layout-independent keys (labels, canonical hashes), since lazy slot numbers depend on the order of first touch.
TEST(RlConcurrency, SixtyFourThreadsShareOneTask)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "openstacks-opt08-adl__p03"})
    {
        const auto ref_task = load(name, TaskOptions::Atoms::Lazy);
        const std::vector<State> ref_states = walk_states(*ref_task, 4, 10, 17);
        const Reference ref = reference(*ref_task, ref_states);
        // the same states as canonical-id sets, to rebuild them in the shared task
        std::vector<std::vector<u64>> cids(ref_states.size());
        for (usize i = 0; i < ref_states.size(); ++i)
            bits::for_each(ref_states[i].data(), ref_states[i].size_words(),
                           [&](u64 s) { cids[i].push_back(ref_task->atoms().canonical(AtomKind::Fluent, static_cast<u32>(s))); });
        std::vector<u64> ref_hash;
        for (const State& s : ref.succ)
            ref_hash.push_back(ref_task->canonical_hash(s));

        const auto task = load(name, TaskOptions::Atoms::Lazy);
        const auto table = rl::TaskTable::single(task);  // shared by the 64 threads
        constexpr u32 T = 64;
        std::barrier sync(T);
        std::vector<int> failures(T, 0);
        ThreadPool pool(4);
        std::vector<std::thread> threads;
        for (u32 t = 0; t < T; ++t)
            threads.emplace_back(
                [&, t]
                {
                    // rebuild the states in the shared task (all 64 threads intern at once)
                    sync.arrive_and_wait();
                    std::vector<State> states;
                    for (const auto& cs : cids)
                    {
                        StateBuilder b;
                        for (u64 c : cs)
                            b.set(SlotId{task->atoms().intern(c)});
                        states.push_back(b.build());
                    }
                    sync.arrive_and_wait();
                    for (int rep = 0; rep < 3; ++rep)
                    {
                        const u32 W = task->max_words();
                        const std::vector<u64> rows = pack(states, W);
                        Buffers b(states.size(), ref.actions.size(), W, rl::max_label_width(*task));
                        rl::ExpandScratch scratch;
                        if (t % 16 == 0)
                            rl::expand(*table, rl::StateBatchView{rows.data(), states.size(), W, 0}, nullptr, b.x, {}, pool, scratch);
                        else
                            rl::expand(*table, rl::StateBatchView{rows.data(), states.size(), W, 0}, nullptr, b.x);
                        if (b.x.total != ref.actions.size() || b.x.overflow())
                            ++failures[t];
                        for (usize j = 0; j < ref.actions.size() && j < b.x.total; ++j)
                        {
                            const StateView sv{b.succ.data() + j * W, W, nullptr, 0};
                            if (task->canonical_hash(sv) != ref_hash[j] || b.schema[j] != static_cast<i32>(ref.actions[j].schema.v))
                                ++failures[t];
                            if (b.goal[j] != (ref_task->is_goal(ref.succ[j]) ? 1 : 0))
                                ++failures[t];
                        }
                        // apply and is_goal through the per-thread workspace
                        Successors& succ = task->workspace().successors();
                        for (usize j = t % 7; j < ref.actions.size(); j += 7)
                        {
                            const State s = succ.apply(states[static_cast<usize>(ref.parent[j])], ref.actions[j].label());
                            if (task->canonical_hash(s) != ref_hash[j] || task->is_goal(s) != ref_task->is_goal(ref.succ[j]))
                                ++failures[t];
                        }
                    }
                });
        for (auto& th : threads)
            th.join();
        for (u32 t = 0; t < T; ++t)
            EXPECT_EQ(failures[t], 0) << name << " thread " << t;
    }
}
}  // namespace
