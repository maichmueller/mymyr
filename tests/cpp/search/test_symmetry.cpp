// Symmetry pruning (successor/symmetry.hpp) on the lifted suite (tests/data/tasks):
//   - along seeded random walks, the colour classes equal those of the reference colour refinement
//     (datasets::color_refinement_certificate on the state's object graph), and generate() with SymmetryPruning::Wl1
//     emits exactly the applicable actions that the representative rule keeps, as a subsequence of the unpruned
//     canonical order (and the same set without canonical order); with Off it emits the unpruned sequence;
//   - every search with pruning on returns plans that replay to a goal, also through the IPC plan text;
//   - brfs on 1, 4 and 8 threads with pruning on finds the same states with the same ids (fingerprint) as the
//     single-threaded stores, and the same shortest plan length (the TSan case of the threaded brfs).
// Fork parity of the pruned actions and searches: successor/test_symmetry_fork.cpp.

#include "../support/suite.hpp"

#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
std::vector<u32> canonical(std::span<const u32> c)
{
    std::map<u32, u32> id;
    std::vector<u32> out;
    for (u32 x : c)
        out.push_back(id.emplace(x, static_cast<u32>(id.size())).first->second);
    return out;
}

using Label = std::pair<u32, std::vector<u32>>;  // schema, binding

/// The representative rule of successor/symmetry.hpp on the applicable `actions` with the pruner's classes and static
/// domains: the actions whose parameters are all kept, in the order given.
std::vector<Label> select(detail::SymmetryPruner& pruner, const std::vector<Label>& actions)
{
    const std::span<const u32> cls = pruner.object_classes();
    std::map<u32, std::vector<std::set<u32>>> kept;
    std::vector<Label> out;
    for (const Label& a : actions)
    {
        const u32 schema = a.first, arity = static_cast<u32>(a.second.size());
        auto [it, fresh] = kept.try_emplace(schema);
        if (fresh)
        {
            std::map<u32, u32> n;
            for (u32 i = 0; i < arity; ++i)
            {
                std::set<u32> met;
                for (u32 o : pruner.domain(schema, i))
                    met.insert(cls[o]);
                for (u32 c : met)
                    ++n[c];
            }
            for (u32 i = 0; i < arity; ++i)
            {
                std::map<u32, u32> used;
                std::set<u32> k;
                for (u32 o : pruner.domain(schema, i))
                    if (used[cls[o]]++ < n[cls[o]])
                        k.insert(o);
                it->second.push_back(std::move(k));
            }
        }
        bool all = true;
        for (u32 i = 0; i < arity; ++i)
            all = all && it->second[i].contains(a.second[i]);
        if (all)
            out.push_back(a);
    }
    return out;
}

std::vector<Label> collect(Successors& succ, bool canonical_order, SymmetryPruning symmetry)
{
    std::vector<Label> out;
    succ.generate<false>(
        [&](u32 schema, const ObjectId* b, const Delta&)
        {
            Label l{schema, {}};
            for (u32 i = 0; i < succ.arity(schema); ++i)
                l.second.push_back(b[i].v);
            out.push_back(std::move(l));
            return true;
        },
        false, canonical_order, symmetry);
    return out;
}

/// Replays `plan`: every action applicable and the last state a goal state; the same for the plan read back from its
/// IPC text (format_plan, parse_plan).
bool valid_plan(const Task& task, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    auto replay = [&](const std::vector<Action>& p)
    {
        State s = task.initial_state();
        for (const Action& a : p)
        {
            if (!succ.is_applicable(s, a.label()))
                return false;
            s = succ.apply(s, a.label());
        }
        return succ.is_goal(s);
    };
    if (!replay(plan))
        return false;
    const std::vector<Action> parsed = parse_plan(task, format_plan(task, plan));
    return parsed.size() == plan.size() && replay(parsed);
}

u64 state_budget()
{
#if defined(MYMYR_SANITIZED)
    return 5'000;
#else
    return 50'000;
#endif
}
}  // namespace

TEST(SymmetryPruning, KeptActionsFollowTheRuleAlongWalks)
{
    u64 states = 0, all = 0, kept = 0;
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        Successors& succ = task->workspace().successors();
        detail::SymmetryPruner& pruner = succ.symmetry_pruner();
        datasets::ObjectGraphBuilder graphs(*task);
        const u32 n = task->data().num_objects();
        std::mt19937_64 rng(4242);
        for (u32 walk = 0; walk < 2; ++walk)
        {
            State s = task->initial_state();
            for (u32 step = 0; step < 25; ++step)
            {
                SCOPED_TRACE(t.name + " walk " + std::to_string(walk) + " step " + std::to_string(step));
                succ.prepare(s);
                const std::vector<Label> plain = collect(succ, true, SymmetryPruning::Off);
                const std::vector<Label> pruned = collect(succ, true, SymmetryPruning::Wl1);
                std::vector<Label> unordered = collect(succ, false, SymmetryPruning::Wl1);
                (void) pruner.compute(succ.engine());
                const datasets::ObjectGraph g = graphs.build(s.view());
                std::vector<u32> stable;
                (void) datasets::color_refinement_certificate(g, &stable);
                ASSERT_GE(stable.size(), n);
                EXPECT_EQ(canonical(pruner.object_classes()), canonical(std::span<const u32>(stable.data(), n)));
                EXPECT_EQ(pruned, select(pruner, plain));  // the rule, in canonical order
                std::sort(unordered.begin(), unordered.end());
                std::vector<Label> sorted = pruned;
                std::sort(sorted.begin(), sorted.end());
                EXPECT_EQ(unordered, sorted);
                std::vector<Label> direct;
                succ.prepare(s);
                succ.generate<false>(
                    [&](u32 schema, const ObjectId* b, const Delta&)
                    {
                        Label l{schema, {}};
                        for (u32 i = 0; i < succ.arity(schema); ++i)
                            l.second.push_back(b[i].v);
                        direct.push_back(std::move(l));
                        return true;
                    },
                    false, true);
                EXPECT_EQ(direct, plain);
                ++states;
                all += plain.size();
                kept += pruned.size();
                if (plain.empty())
                    break;
                const Label& a = plain[std::uniform_int_distribution<usize>(0, plain.size() - 1)(rng)];
                std::vector<ObjectId> b;
                for (u32 o : a.second)
                    b.push_back(ObjectId{o});
                s = succ.apply(s, ActionLabel{SchemaId{a.first}, b});
            }
        }
    }
    std::printf("symmetry pruning along walks: %llu states, %llu applicable actions, %llu kept\n",
                static_cast<unsigned long long>(states), static_cast<unsigned long long>(all),
                static_cast<unsigned long long>(kept));
}

TEST(SymmetryPruning, PlansAreValid)
{
    u32 solved = 0, runs = 0;
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        auto check = [&](const char* what, bool ok, const std::vector<Action>& plan)
        {
            ++runs;
            if (!ok)
                return;
            ++solved;
            EXPECT_TRUE(valid_plan(*task, plan)) << t.name << " " << what;
        };
        BrfsOptions bo;
        bo.symmetry_pruning = SymmetryPruning::Wl1;
        bo.stop_at_goal = true;
        bo.max_states = state_budget();
        const BrfsResult br = brfs(*task, bo);
        check("brfs", br.solved, br.plan);
        for (const heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::FF})
        {
            BestFirstOptions ao;
            ao.heuristic.kind = k;
            ao.symmetry_pruning = SymmetryPruning::Wl1;
            ao.control.budget.max_states = state_budget();
            const BestFirstResult a = astar_eager(*task, ao);
            check("astar_eager", a.status == SearchStatus::Solved, a.plan);
            if (k == heuristics::Kind::FF)
            {
                const BestFirstResult g = gbfs_lazy(*task, ao);
                check("gbfs_lazy", g.status == SearchStatus::Solved, g.plan);
                ao.beam_width = 64;
                const BestFirstResult bm = beam(*task, ao);
                check("beam", bm.status == SearchStatus::Solved, bm.plan);
            }
        }
        if (task->numeric_slots() == 0)
        {
            IwOptions io;
            io.symmetry_pruning = SymmetryPruning::Wl1;
            io.max_arity = 1;
            io.control.budget.max_states = state_budget();
            const IwResult r = iw(*task, io);
            check("iw", r.status == SearchStatus::Solved, r.plan);
            const SiwResult sr = siw(*task, io);
            check("siw", sr.status == SearchStatus::Solved, sr.plan);
            RolloutIwOptions ro;
            ro.symmetry_pruning = SymmetryPruning::Wl1;
            ro.control.budget.max_states = state_budget();
            const RolloutIwResult rr = rollout_iw(*task, ro);
            check("rollout_iw", rr.status == SearchStatus::Solved, rr.plan);
            AStarIwOptions ai;
            ai.symmetry_pruning = SymmetryPruning::Wl1;
            ai.control.budget.max_states = state_budget();
            const AStarIwResult air = astar_iw(*task, ai);
            check("astar_iw", air.status == SearchStatus::Solved, air.plan);
        }
    }
    std::printf("symmetry pruning: %u of %u searches solved, every plan valid\n", solved, runs);
    EXPECT_GT(solved, runs / 2);
}

TEST(SymmetryPruning, ThreadedBrfsIsDeterministic)
{
    for (const SuiteTask& t : suite())
    {
        if (t.states > std::min<u64>(suite_state_limit(), 50'000))
            continue;
        const auto task = Task::from_text_file(task_path(t.name));
        BrfsOptions bo;
        bo.symmetry_pruning = SymmetryPruning::Wl1;
        bo.fingerprint = true;
        bo.store = BrfsOptions::Store::Flat;
        const BrfsResult ref = brfs(*task, bo);
        EXPECT_LE(ref.states, t.states) << t.name;
        bo.store = BrfsOptions::Store::Chunked;
        EXPECT_EQ(brfs(*task, bo).fingerprint, ref.fingerprint) << t.name;
        BrfsOptions go = bo;
        go.store = BrfsOptions::Store::Flat;
        go.fingerprint = false;
        go.stop_at_goal = true;
        const BrfsResult goal_ref = brfs(*task, go);
        bo.store = BrfsOptions::Store::Concurrent;
        go.store = BrfsOptions::Store::Concurrent;
        for (u32 T : {1u, 4u, 8u})
        {
            bo.threads = T;
            const BrfsResult r = brfs(*task, bo);
            EXPECT_EQ(r.states, ref.states) << t.name << " T=" << T;
            EXPECT_EQ(r.fingerprint, ref.fingerprint) << t.name << " T=" << T;
            go.threads = T;
            const BrfsResult g = brfs(*task, go);
            EXPECT_EQ(g.solved, goal_ref.solved) << t.name << " T=" << T;
            if (g.solved)
            {
                EXPECT_EQ(g.plan.size(), goal_ref.plan.size()) << t.name << " T=" << T;
                EXPECT_TRUE(valid_plan(*task, g.plan)) << t.name << " T=" << T;
            }
        }
    }
}
