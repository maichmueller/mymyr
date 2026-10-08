// The beam's layer step on several threads (IwOptions::threads, AbstractedIwOptions::threads, BrfsOptions::threads;
// search/layer_ordering.hpp, "Threads") and the relaxed SurvivorsOnly beam (BeamNovelty::RelaxedSurvivorsOnly):
//   - on the beam suite (the tasks of tests/data/beam/fork_beam.json), IW(2) and BrFS with widths 1, 4 and 32, both
//     novelty modes, both goal-count directions, ties in generation order and random with three seeds: 2, 4 and 8
//     threads give the plan, the counts of every pass (BrFS: the counts and the fingerprint of the store) of one
//     thread;
//   - the IW family variants (abstracted, projective, LIW) alike;
//   - the relaxed beam equals a literal re-implementation of its definition (per-part best candidates by the read-only
//     test, merged, the states seen before skipped, then the SurvivorsOnly replay) for several part counts, is
//     deterministic for a thread count, finds valid plans, and with one part does not depend on the thread count;
//   - the refusals, and concurrent multi-threaded beam searches over one task.

#include "../support/suite.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
using Novelty = LayerOrdering::BeamNovelty;

bool sanitized()
{
#if defined(MYMYR_SANITIZED)
    return true;
#else
    return false;
#endif
}

/// The tasks of the beam suite: the lifted suite without philosophers, pegsol and organic-synthesis (run_beam.py).
std::vector<std::string> beam_suite()
{
    std::vector<std::string> out;
    for (const SuiteTask& t : suite())
        if (!t.name.starts_with("philosophers") && !t.name.starts_with("pegsol") && !t.name.starts_with("organic-synthesis"))
            out.push_back(t.name);
    return out;
}

LayerOrdering beam(u32 width, Novelty mode, bool prefer_more = true, int tie_seed = -1)
{
    LayerOrdering lo;
    lo.kind = LayerOrdering::Kind::GoalCount;
    lo.prefer_more_satisfied_goals = prefer_more;
    lo.beam_width = width;
    lo.beam_novelty = mode;
    if (tie_seed >= 0)
    {
        lo.randomize_ties = true;
        lo.seed = static_cast<u64>(tie_seed);
    }
    return lo;
}

std::string name_of(Novelty m)
{
    return m == Novelty::AllTested ? "all_tested" : m == Novelty::SurvivorsOnly ? "survivors_only" : "relaxed";
}

void expect_same(const IwResult& a, const IwResult& b, const std::string& what)
{
    EXPECT_EQ(a.status, b.status) << what;
    EXPECT_EQ(a.plan, b.plan) << what;
    ASSERT_EQ(a.passes.size(), b.passes.size()) << what;
    for (usize i = 0; i < a.passes.size(); ++i)
    {
        EXPECT_EQ(a.passes[i].status, b.passes[i].status) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].expanded, b.passes[i].expanded) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].generated, b.passes[i].generated) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].generated_in_tree, b.passes[i].generated_in_tree) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].skipped, b.passes[i].skipped) << what << " pass " << i;
    }
}

void expect_same(const SiwResult& a, const SiwResult& b, const std::string& what)
{
    EXPECT_EQ(a.status, b.status) << what;
    EXPECT_EQ(a.plan, b.plan) << what;
    EXPECT_EQ(a.total.expanded, b.total.expanded) << what;
    EXPECT_EQ(a.total.generated, b.total.generated) << what;
    ASSERT_EQ(a.subproblems.size(), b.subproblems.size()) << what;
    for (usize i = 0; i < a.subproblems.size(); ++i)
    {
        EXPECT_EQ(a.subproblems[i].status, b.subproblems[i].status) << what << " subproblem " << i;
        EXPECT_EQ(a.subproblems[i].passes.size(), b.subproblems[i].passes.size()) << what << " subproblem " << i;
    }
}

void expect_same(const BrfsResult& a,const BrfsResult& b, const std::string& what)
{
    EXPECT_EQ(a.status, b.status) << what;
    EXPECT_EQ(a.plan, b.plan) << what;
    EXPECT_EQ(a.expanded, b.expanded) << what;
    EXPECT_EQ(a.generated, b.generated) << what;
    EXPECT_EQ(a.states, b.states) << what;
    EXPECT_EQ(a.goal_states, b.goal_states) << what;
    EXPECT_EQ(a.layers, b.layers) << what;
    EXPECT_EQ(a.fingerprint, b.fingerprint) << what;
}

IwResult run_iw(const Task& task, const LayerOrdering& lo, u32 threads, u64 max_expanded = ~u64{0})
{
    IwOptions o;
    o.max_arity = 2;
    o.control.budget.max_expanded = max_expanded;
    o.layers = lo;
    o.threads = threads;
    return iw(task, o);
}

BrfsResult run_brfs(const Task& task, const LayerOrdering& lo, u32 threads, u64 max_states = ~u64{0})
{
    BrfsOptions o;
    o.max_states = max_states;
    o.witness_pruning = false;
    o.stop_at_goal = true;
    o.fingerprint = true;
    o.layers = lo;
    o.threads = threads;
    return brfs(task, o);
}

/// Whether `plan` leads from the initial state to a goal state.
bool valid_plan(const Task& task, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    return succ.is_goal(s);
}
}  // namespace

// ------------------------------------------------------------------------------------------------- exact
TEST(BeamParallel, EqualsTheSerialBeamOnTheBeamSuite)
{
    // the whole matrix with MYMYR_TEST_FULL_SUITE=1; by default (and sanitized) the searches stop at a budget, which
    // the parallel search must meet at the same expansion
    const bool full = !sanitized() && suite_state_limit() == ~u64{0};
    const u64 budget = full ? ~u64{0} : sanitized() ? 1500 : 6000;
    const std::vector<u32> threads = sanitized() ? std::vector<u32>{2, 4} : std::vector<u32>{2, 4, 8};
    const std::vector<int> seeds = full ? std::vector<int>{-1, 1, 2, 3} : std::vector<int>{-1, 2};
    u32 compared = 0;
    for (const std::string& name : beam_suite())
    {
        const auto task = Task::from_text_file(task_path(name));
        for (u32 w : {1u, 4u, 32u})
            for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
                for (bool more : {true, false})
                    for (int seed : seeds)
                    {
                        const LayerOrdering lo = beam(w, mode, more, seed);
                        const std::string what = name + " width " + std::to_string(w) + " " + name_of(mode) + (more ? " more" : " fewer") +
                                                 " seed " + std::to_string(seed);
                        const IwResult si = run_iw(*task, lo, 1, budget);
                        const BrfsResult sb = run_brfs(*task, lo, 1, budget * 8);
                        for (u32 T : threads)
                        {
                            expect_same(run_iw(*task, lo, T, budget), si, what + " iw threads " + std::to_string(T));
                            const BrfsResult pb = run_brfs(*task, lo, T, budget * 8);
                            EXPECT_EQ(pb.threads, T) << what;
                            expect_same(pb, sb, what + " brfs threads " + std::to_string(T));
                            compared += 2;
                        }
                    }
    }
    EXPECT_GE(compared, full ? 5472u : sanitized() ? 1824u : 2736u);
}

TEST(BeamParallel, OptimizedIw1SiwAndIwPassEqualTheSerialBeam)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "gripper__prob05", "miconic__s7-4", "visitall__visitall_x-6_y-3_r-100"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
            for (u32 w : {2u, 16u})
            {
                const std::string what = std::string(name) + " " + name_of(mode) + " width " + std::to_string(w);
                IwOptions o;
                o.layers = beam(w, mode, true, 7);
                o.max_arity = 1;  // optimized IW(1)
                const IwResult a = iw(*task, o);
                const SiwResult sa = siw(*task, o);
                const IwResult pa = iw_pass(*task, 2, o);
                o.threads = 4;
                expect_same(iw(*task, o), a, what + " iw(1)");
                expect_same(siw(*task, o), sa, what + " siw");
                expect_same(iw_pass(*task, 2, o), pa, what + " iw_pass(2)");
            }
    }
}

TEST(BeamParallel, ChunkedAndFlatStoresEqualTheSerialBeam)
{
    for (const char* name : {"gripper__prob05", "sokoban-opt08-strips__p14", "transport-opt08-strips__p23"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (BrfsOptions::Store store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked})
            for (Novelty mode : {Novelty::AllTested, Novelty::RelaxedSurvivorsOnly})
            {
                BrfsOptions o;
                o.store = store;
                o.stop_at_goal = true;
                o.fingerprint = true;
                o.layer_stats = true;
                o.layers = beam(8, mode, true, 3);
                const BrfsResult a = brfs(*task, o);
                o.threads = mode == Novelty::RelaxedSurvivorsOnly ? 1 : 4;
                o.layers.beam_chunk = mode == Novelty::RelaxedSurvivorsOnly ? 7 : 1024;
                // relaxed: one thread, so one part whatever the chunk
                const BrfsResult b = brfs(*task, o);
                expect_same(b, a, std::string(name) + (store == BrfsOptions::Store::Flat ? " flat " : " chunked ") + name_of(mode));
                EXPECT_EQ(b.layer_counts, a.layer_counts) << name;
            }
    }
}

TEST(BeamParallel, IwFamilyVariantsEqualTheSerialBeam)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "gripper__prob05", "logistics00__probLOGISTICS-6-1", "rovers__p02"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (u32 w : {1u, 8u})
            for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
            {
                const std::string what = std::string(name) + " " + name_of(mode) + " width " + std::to_string(w);
                AbstractedIwOptions a;
                a.layers = beam(w, mode, true, 5);
                for (u32 width : {1u, 2u})
                {
                    a.width = width;
                    a.threads = 1;
                    const IwResult s = abstracted_iw(*task, a);
                    a.threads = 4;
                    expect_same(abstracted_iw(*task, a), s, what + " abstracted width " + std::to_string(width));
                }
                ProjectiveIwOptions p;
                p.layers = beam(w, mode, false);
                const IwResult ps = projective_iw(*task, p);
                p.threads = 3;
                expect_same(projective_iw(*task, p), ps, what + " projective");
                if (mode == Novelty::AllTested)
                {
                    LiwOptions l;
                    l.max_arity = 2;
                    l.layers = beam(w, mode);
                    const IwResult ls = liw(*task, l);
                    l.threads = 4;
                    expect_same(liw(*task, l), ls, what + " liw");
                }
            }
    }
}

// ------------------------------------------------------------------------------------------------- relaxed
namespace
{
std::vector<u32> atoms_of(const State& s)
{
    std::vector<u32> out;
    for (SlotId x : s.slots())
        out.push_back(x.v);
    return out;
}

u64 pack(std::vector<u32> t)
{
    std::sort(t.begin(), t.end());
    u64 key = 0;
    for (u32 x : t)
        key = (key << 21) | (u64{x} + 1);
    return key;
}

template<class F>
void for_each_subset(const std::vector<u32>& atoms, u32 size, F f)
{
    std::vector<u32> t;
    auto rec = [&](auto& self, usize from) -> void
    {
        f(t);
        if (t.size() == size)
            return;
        for (usize i = from; i < atoms.size(); ++i)
        {
            t.push_back(atoms[i]);
            self(self, i + 1);
            t.pop_back();
        }
    };
    rec(rec, 0);
}

/// Unsatisfied goal literals of the task (fluent goals only; static literals are folded), each literal once.
u32 unsatisfied_of(const Task& task, const State& s)
{
    const formalism::TaskData& t = task.data();
    std::set<std::pair<std::vector<u32>, bool>> lits;
    for (const formalism::Literal& l : t.literals_of(t.goal))
    {
        if (t.predicates[l.pred.v].kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> key{l.pred.v};
        for (formalism::Term x : t.terms_of(l))
            key.push_back(formalism::term_object(x).v);
        lits.insert({key, l.positive});
    }
    u32 c = 0;
    for (const auto& [key, pos] : lits)
    {
        std::vector<ObjectId> args;
        for (usize i = 1; i < key.size(); ++i)
            args.push_back(ObjectId{key[i]});
        const SlotId slot = task.find_atom(PredicateId{key[0]}, args);
        c += s.contains(slot) != pos;
    }
    return c;
}

struct RelaxedSpec
{
    int k = 1;  // novelty arity; -1: BrFS
    u32 width = 4;
    u32 threads = 1;
    u32 chunk = 1024;
    bool continuation = false;
    u64 max_expanded = ~u64{0};
};

struct RelaxedRef
{
    SearchStatus status = SearchStatus::Exhausted;
    u64 expanded = 0, generated = 0, in_tree = 0, skipped = 0;
    u32 plan_length = 0;
};

/// RelaxedSurvivorsOnly by its definition (search/layer_ordering.hpp), with ties in generation order: the
/// transitions of a layer are numbered in generation order and split into min(threads, ceil(n / chunk)) parts of
/// sizes base + 1 (the first n mod parts) and base; a transition is a candidate if its successor passes the read-only
/// novelty test (no test for states seen before); each part keeps its best `width` candidates by (unsatisfied goal
/// literals, number); the kept ones of all parts, in that order, enter unless their state was seen before, until
/// `width` entered; then the SurvivorsOnly replay.
RelaxedRef ref_relaxed(const Task& task, const RelaxedSpec& sp)
{
    Successors& succ = task.workspace().successors();
    const bool w0 = succ.witness_pruning(), c0 = succ.canonical_order();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);
    RelaxedRef r;
    const State root = task.initial_state();
    std::unordered_set<u64> table;
    if (sp.k >= 1)
        for_each_subset(atoms_of(root), static_cast<u32>(sp.k),
                        [&](const std::vector<u32>& t)
                        {
                            if (!t.empty())
                                table.insert(pack(t));
                        });
    auto novel_tuples = [&](const State& p, const State& t)
    {
        std::vector<u64> out;
        const std::vector<u32> pa = atoms_of(p), ta = atoms_of(t);
        for (u32 a : ta)
        {
            if (std::binary_search(pa.begin(), pa.end(), a))
                continue;
            std::vector<u32> others;
            for (u32 x : ta)
                if (x != a)
                    others.push_back(x);
            for_each_subset(others, static_cast<u32>(sp.k) - 1,
                            [&](std::vector<u32> tu)
                            {
                                tu.push_back(a);
                                out.push_back(pack(std::move(tu)));
                            });
        }
        return out;
    };
    struct Node
    {
        State s, parent;
        u32 depth = 0;
        bool skip = false;
    };
    struct Cand
    {
        u32 key;
        u64 pos;
        Node node;
    };
    std::unordered_set<State> seen{root}, closed;
    std::vector<Node> layer{{root, root, 0, false}};
    auto done = [&](SearchStatus s, u32 len)
    {
        r.status = s;
        r.plan_length = len;
        succ.set_witness_pruning(w0);
        succ.set_canonical_order(c0);
        return r;
    };
    while (!layer.empty())
    {
        std::vector<Cand> cand;
        u64 n = 0;
        for (const Node& nd : layer)
        {
            if (!closed.insert(nd.s).second)
                continue;
            if (task.is_goal(nd.s))
                return done(SearchStatus::Solved, nd.depth);
            if (r.expanded >= sp.max_expanded)
                return done(SearchStatus::OutOfStates, 0);
            ++r.expanded;
            if (nd.skip)
            {
                ++r.skipped;
                continue;
            }
            const bool at_root = nd.depth == 0;
            for (const Action& a : succ.applicable_actions(nd.s))
            {
                ++r.generated;
                const u64 pos = n++;
                const State t = succ.apply(nd.s, a.label());
                bool keep = false, skip = false;
                if (sp.k < 0)
                    keep = true;
                else if (sp.k == 0)
                    keep = at_root && t != nd.s;
                else if (t != nd.s)
                {
                    bool novel = false;
                    for (u64 x : novel_tuples(nd.s, t))
                        novel |= !table.contains(x);
                    if (at_root && sp.continuation)
                    {
                        keep = true;
                        skip = !novel;
                    }
                    else
                        keep = novel;
                }
                if (keep)
                    cand.push_back({unsatisfied_of(task, t), pos, Node{t, nd.s, nd.depth + 1, skip}});
            }
        }
        const u64 parts = std::min<u64>(sp.threads, std::max<u64>(1, (n + sp.chunk - 1) / sp.chunk));
        const u64 base = n / parts, longer = n % parts;
        std::vector<Cand> merged;
        u64 begin = 0;
        for (u64 p = 0; p < parts; ++p)
        {
            const u64 end = begin + base + (p < longer ? 1 : 0);
            std::vector<Cand> part;
            for (const Cand& c : cand)
                if (c.pos >= begin && c.pos < end)
                    part.push_back(c);
            std::stable_sort(part.begin(), part.end(), [](const Cand& a, const Cand& b) { return a.key < b.key; });
            for (usize i = 0; i < part.size() && i < sp.width; ++i)
                merged.push_back(part[i]);
            begin = end;
        }
        std::sort(merged.begin(), merged.end(), [](const Cand& a, const Cand& b) { return a.key != b.key ? a.key < b.key : a.pos < b.pos; });
        std::vector<Node> kept;
        for (const Cand& c : merged)
        {
            if (kept.size() >= sp.width)
                break;
            if (seen.insert(c.node.s).second)
                kept.push_back(c.node);
        }
        std::vector<Node> next;
        if (sp.k >= 1)
        {
            std::unordered_set<u64> delta;
            for (Node& nd : kept)
            {
                bool novel = false;
                for (u64 x : novel_tuples(nd.parent, nd.s))
                    if (!table.contains(x) && delta.insert(x).second)
                        novel = true;
                if (nd.depth == 1 && sp.continuation)
                {
                    nd.skip = !novel;
                    next.push_back(nd);
                }
                else if (novel)
                    next.push_back(nd);
            }
            table.insert(delta.begin(), delta.end());
        }
        else
            next = kept;
        r.in_tree += next.size();
        layer = std::move(next);
    }
    return done(SearchStatus::Exhausted, 0);
}
}  // namespace

TEST(RelaxedBeam, EqualsItsDefinition)
{
    u32 compared = 0;
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0", "miconic__s7-4", "visitall__visitall_x-6_y-3_r-100"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (u32 w : {1u, 3u, 8u})
            for (u32 T : {1u, 2u, 3u, 5u})
                for (u32 chunk : {1u, 6u, 1024u})
                {
                    if (T == 1 && chunk != 1024)
                        continue;  // one part either way
                    LayerOrdering lo = beam(w, Novelty::RelaxedSurvivorsOnly);
                    lo.beam_chunk = chunk;
                    const std::string what = std::string(name) + " width " + std::to_string(w) + " threads " + std::to_string(T) +
                                             " chunk " + std::to_string(chunk);
                    for (int k : {0, 1, 2})
                    {
                        IwOptions o;
                        o.layers = lo;
                        o.threads = T;
                        o.control.budget.max_expanded = 400;
                        const IwResult m = iw_pass(*task, static_cast<u32>(k), o);
                        const RelaxedRef ref = ref_relaxed(*task, {.k = k, .width = w, .threads = T, .chunk = chunk, .max_expanded = 400});
                        ASSERT_EQ(m.passes.size(), 1u) << what;
                        EXPECT_EQ(m.status, ref.status) << what << " k " << k;
                        EXPECT_EQ(m.plan.size(), ref.plan_length) << what << " k " << k;
                        EXPECT_EQ(m.passes[0].expanded, ref.expanded) << what << " k " << k;
                        EXPECT_EQ(m.passes[0].generated, ref.generated) << what << " k " << k;
                        EXPECT_EQ(m.passes[0].generated_in_tree, ref.in_tree) << what << " k " << k;
                        ++compared;
                    }
                    {
                        // optimized IW(1): the root's successors enter as candidates, the non-novel ones not expanded
                        IwOptions o;
                        o.layers = lo;
                        o.threads = T;
                        o.max_arity = 1;
                        o.control.budget.max_expanded = 400;
                        const IwResult m = iw(*task, o);
                        const RelaxedRef ref =
                            ref_relaxed(*task, {.k = 1, .width = w, .threads = T, .chunk = chunk, .continuation = true, .max_expanded = 400});
                        ASSERT_EQ(m.passes.size(), 2u) << what;
                        EXPECT_EQ(m.status, ref.status) << what << " optimized";
                        EXPECT_EQ(m.passes[1].expanded, ref.expanded) << what << " optimized";
                        EXPECT_EQ(m.passes[1].skipped, ref.skipped) << what << " optimized";
                        EXPECT_EQ(m.passes[1].generated_in_tree, ref.in_tree) << what << " optimized";
                        ++compared;
                    }
                    {
                        BrfsOptions o;
                        o.witness_pruning = false;
                        o.stop_at_goal = true;
                        o.layers = lo;
                        o.threads = T;
                        const BrfsResult m = brfs(*task, o);
                        const RelaxedRef ref = ref_relaxed(*task, {.k = -1, .width = w, .threads = T, .chunk = chunk});
                        EXPECT_EQ(m.solved, ref.status == SearchStatus::Solved) << what << " brfs";
                        EXPECT_EQ(m.plan.size(), ref.plan_length) << what << " brfs";
                        EXPECT_EQ(m.expanded, ref.expanded) << what << " brfs";
                        EXPECT_EQ(m.generated, ref.generated) << what << " brfs";
                        EXPECT_EQ(m.states, ref.in_tree + 1) << what << " brfs";
                        ++compared;
                    }
                }
    }
    EXPECT_GE(compared, 300u);
}

TEST(RelaxedBeam, DeterministicForAThreadCountWithValidPlans)
{
    std::vector<std::string> names = beam_suite();
    if (sanitized())
        names.resize(6);
    for (const std::string& name : names)
    {
        const auto task = Task::from_text_file(task_path(name));
        for (u32 w : {1u, 4u, 32u})
            for (int seed : {-1, 1, 2, 3})
                for (u32 T : {1u, 2u, 4u, 8u})
                {
                    if (sanitized() && (w == 32 || T == 8))
                        continue;
                    LayerOrdering lo = beam(w, Novelty::RelaxedSurvivorsOnly, true, seed);
                    lo.beam_chunk = 64;  // several parts on these tasks
                    const std::string what = name + " width " + std::to_string(w) + " seed " + std::to_string(seed) + " threads " +
                                             std::to_string(T);
                    const IwResult a = run_iw(*task, lo, T);
                    expect_same(run_iw(*task, lo, T), a, what + " iw");
                    EXPECT_TRUE(a.status != SearchStatus::Solved || valid_plan(*task, a.plan)) << what;
                    const BrfsResult b = run_brfs(*task, lo, T);
                    expect_same(run_brfs(*task, lo, T), b, what + " brfs");
                    EXPECT_TRUE(!b.solved || valid_plan(*task, b.plan)) << what;
                }
    }
}

TEST(RelaxedBeam, OnePartDoesNotDependOnTheThreadCount)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "rovers__p02", "transport-opt08-strips__p23"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (int seed : {-1, 4})
        {
            LayerOrdering lo = beam(4, Novelty::RelaxedSurvivorsOnly, true, seed);
            lo.beam_chunk = ~u32{0};
            const IwResult a = run_iw(*task, lo, 1);
            const BrfsResult b = run_brfs(*task, lo, 1);
            for (u32 T : {2u, 8u})
            {
                expect_same(run_iw(*task, lo, T), a, std::string(name) + " iw");
                BrfsResult c = run_brfs(*task, lo, T);
                c.threads = b.threads;
                expect_same(c, b, std::string(name) + " brfs");
            }
            // the abstracted IW engine alike
            AbstractedIwOptions o;
            o.layers = lo;
            const IwResult s = abstracted_iw(*task, o);
            o.threads = 4;
            expect_same(abstracted_iw(*task, o), s, std::string(name) + " abstracted");
        }
    }
}

TEST(RelaxedBeam, RandomTiesFollowTheSeed)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    std::set<std::vector<Action>> plans;
    for (u64 seed = 0; seed < 6; ++seed)
    {
        LayerOrdering lo = beam(2, Novelty::RelaxedSurvivorsOnly, true, static_cast<int>(seed));
        const IwResult a = run_iw(*task, lo, 2);
        expect_same(run_iw(*task, lo, 2), a, "seed " + std::to_string(seed));
        plans.insert(a.plan);
    }
    EXPECT_GT(plans.size(), 1u);  // the seed changes the ties
}

// ------------------------------------------------------------------------------------------------- refusals
TEST(BeamParallel, Refusals)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    // threads > 1 without a beam
    IwOptions o;
    o.threads = 2;
    IwResult r = iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("requires a beam"), std::string::npos) << r.message;
    o.layers.kind = LayerOrdering::Kind::GoalCount;
    EXPECT_EQ(siw(*task, o).status, SearchStatus::Failed);
    AbstractedIwOptions a;
    a.threads = 2;
    EXPECT_EQ(abstracted_iw(*task, a).status, SearchStatus::Failed);
    LiwOptions l;
    l.threads = 2;
    EXPECT_EQ(liw(*task, l).status, SearchStatus::Failed);
    // relaxed: needs GoalCount and a positive chunk; LIW has no read-only test
    LayerOrdering lo = beam(4, Novelty::RelaxedSurvivorsOnly);
    lo.kind = LayerOrdering::Kind::InOrder;
    o = IwOptions{};
    o.layers = lo;
    r = iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("RelaxedSurvivorsOnly requires Kind::GoalCount"), std::string::npos) << r.message;
    BrfsOptions b;
    b.layers = lo;
    EXPECT_THROW((void)brfs(*task, b), std::invalid_argument);
    o.layers = beam(4, Novelty::RelaxedSurvivorsOnly);
    o.layers.beam_chunk = 0;
    r = iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("beam_chunk must be positive"), std::string::npos) << r.message;
    l = LiwOptions{};
    l.layers = beam(4, Novelty::RelaxedSurvivorsOnly);
    r = liw(*task, l);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("AllTested"), std::string::npos) << r.message;
    // relaxed with an observer, blocked states or a successor order
    struct Quiet : SearchObserver
    {
    };
    Quiet q;
    o = IwOptions{};
    o.layers = beam(4, Novelty::RelaxedSurvivorsOnly);
    o.control.observer = &q;
    r = iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("observer"), std::string::npos) << r.message;
    a = AbstractedIwOptions{};
    a.layers = beam(4, Novelty::RelaxedSurvivorsOnly);
    a.control.observer = &q;
    EXPECT_EQ(abstracted_iw(*task, a).status, SearchStatus::Failed);
    b = BrfsOptions{};
    b.layers = beam(4, Novelty::RelaxedSurvivorsOnly);
    b.observer = &q;
    EXPECT_THROW((void)brfs(*task, b), std::invalid_argument);
    // an observer runs an exact beam on the calling thread: the same result
    b.layers = beam(4, Novelty::AllTested);
    b.threads = 4;
    b.stop_at_goal = true;
    const BrfsResult with = brfs(*task, b);
    EXPECT_EQ(with.threads, 1u);
    b.observer = nullptr;
    const BrfsResult without = brfs(*task, b);
    EXPECT_EQ(without.threads, 4u);
    EXPECT_EQ(with.plan, without.plan);
    EXPECT_EQ(with.expanded, without.expanded);
    o = IwOptions{};
    o.layers = beam(4, Novelty::SurvivorsOnly);
    o.threads = 4;
    o.control.observer = &q;
    const IwResult io = iw(*task, o);
    o.control.observer = nullptr;
    expect_same(iw(*task, o), io, "observer");
}

// ------------------------------------------------------------------------------------------------- concurrency
TEST(BeamParallel, MultiThreadedBeamSearchesPerThreadOverOneTask)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    auto ordering = [](u32 i)
    {
        const Novelty modes[] = {Novelty::AllTested, Novelty::SurvivorsOnly, Novelty::RelaxedSurvivorsOnly};
        LayerOrdering lo = beam(i % 2 ? 4 : 16, modes[i % 3], true, i % 4 == 0 ? static_cast<int>(i) : -1);
        lo.beam_chunk = 32;
        return lo;
    };
    constexpr u32 n = 6;
    std::vector<IwResult> ref, out(n);
    std::vector<BrfsResult> bref, bout(n);
    for (u32 i = 0; i < n; ++i)
    {
        ref.push_back(run_iw(*task, ordering(i), 2));
        bref.push_back(run_brfs(*task, ordering(i), 2));
    }
    std::vector<std::thread> ts;
    for (u32 i = 0; i < n; ++i)
        ts.emplace_back(
            [&, i]
            {
                out[i] = run_iw(*task, ordering(i), 2);
                bout[i] = run_brfs(*task, ordering(i), 2);
            });
    for (std::thread& t : ts)
        t.join();
    for (u32 i = 0; i < n; ++i)
    {
        expect_same(out[i], ref[i], "iw " + std::to_string(i));
        expect_same(bout[i], bref[i], "brfs " + std::to_string(i));
    }
}
