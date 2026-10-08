// The beam over the layers of BrFS and the IW family (LayerOrdering::beam_width, search/layer_ordering.hpp):
//   - an oracle: a literal re-implementation of the fork's find_solution_with_beam (brfs/beam.cpp) with its
//     DuplicatePruning, ArityZero and ArityK beam hooks (iw/pruning_strategy.cpp), over State objects and a set of
//     tuples; search::iw_pass, search::iw (optimized IW(1)), search::brfs, the IW family engine, LIW without landmarks
//     and abstracted IW on a task with unary fluents (its features are then the atoms) equal it pass for pass,
//     expansion for expansion;
//   - the fork's beam semantics tests on small hand-made tasks: AllTested marks the tuples of the successors the beam
//     drops, SurvivorsOnly does not, and its replay can leave fewer than beam_width states;
//   - random ties (deterministic per seed), a beam wider than every layer, every refusal, and concurrent beam
//     searches over one task.

#include "../../../src/mymyr/search/novelty_brfs.hpp"
#include "../support/suite.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/search/transition_ordering.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;
namespace sd = mymyr::search::detail;

namespace
{
using Novelty = LayerOrdering::BeamNovelty;

LayerOrdering beam(u32 width, Novelty mode, bool prefer_more = true)
{
    LayerOrdering lo;
    lo.kind = LayerOrdering::Kind::GoalCount;
    lo.prefer_more_satisfied_goals = prefer_more;
    lo.beam_width = width;
    lo.beam_novelty = mode;
    return lo;
}

/// Unsatisfied goal literals of the task (fluent goals only; static literals are folded), each literal once.
u32 unsatisfied(const Task& task, const State& s)
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

// ------------------------------------------------------------------------------------------------- the oracle
std::vector<u32> atoms_of(const State& s)
{
    std::vector<u32> out;
    for (SlotId x : s.slots())
        out.push_back(x.v);
    return out;
}

/// A tuple of at most 3 atoms (each below 2^21 - 1) as one key: the sorted atoms + 1, 21 bits each.
u64 pack(std::vector<u32> t)
{
    std::sort(t.begin(), t.end());
    u64 key = 0;
    for (u32 x : t)
        key = (key << 21) | (u64{x} + 1);
    return key;
}

/// Calls f(tuple) for every set of at most `size` atoms of `atoms` (the empty set included).
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

struct RefSpec
{
    int k = 1;  // the novelty arity; -1: BrFS (mimir's DuplicatePruningStrategy)
    u32 width = 1;
    Novelty mode = Novelty::AllTested;
    bool prefer_more = true;
    bool continuation = false;  // the optimized IW(1) root rule (every new root successor enters)
    u64 max_expanded = ~u64{0};
};

struct RefPass
{
    SearchStatus status = SearchStatus::Exhausted;
    u64 expanded = 0, generated = 0, in_tree = 0, skipped = 0;
    u32 plan_length = 0;
    std::vector<State> order;  // the expanded states (goal-tested and counted), in order
};

/// The fork's beam pass, literally: candidates are the successors its pruning strategy admits for beam selection
/// (is_new over every generated state), ranked by (unsatisfied goal literals, generation order) and cut to the
/// width; SurvivorsOnly replays the kept ones against the table and the layer's delta, then commits the delta.
RefPass ref_beam(const Task& task, const RefSpec& sp)
{
    Successors& succ = task.workspace().successors();
    const bool w0 = succ.witness_pruning(), c0 = succ.canonical_order();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);
    const bool survivors = sp.mode == Novelty::SurvivorsOnly;
    RefPass r;
    const State root = task.initial_state();
    std::unordered_set<u64> seen;
    if (sp.k >= 1)
        for_each_subset(atoms_of(root), static_cast<u32>(sp.k),
                        [&](const std::vector<u32>& t)
                        {
                            if (!t.empty())
                                seen.insert(pack(t));
                        });
    // the tuples of t that contain an atom its parent p lacks
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
    std::unordered_set<State> generated{root}, closed;
    std::vector<Node> layer{{root, root, 0, false}};
    while (!layer.empty())
    {
        std::vector<std::pair<u32, Node>> cand;
        for (const Node& nd : layer)
        {
            if (!closed.insert(nd.s).second)
                continue;  // a width-0 duplicate entry
            if (task.is_goal(nd.s))
            {
                r.status = SearchStatus::Solved;
                r.plan_length = nd.depth;
                succ.set_witness_pruning(w0);
                succ.set_canonical_order(c0);
                return r;
            }
            if (r.expanded >= sp.max_expanded)
            {
                r.status = SearchStatus::OutOfStates;
                succ.set_witness_pruning(w0);
                succ.set_canonical_order(c0);
                return r;
            }
            ++r.expanded;
            r.order.push_back(nd.s);
            if (nd.skip)
            {
                ++r.skipped;
                continue;
            }
            const bool at_root = nd.depth == 0;
            for (const Action& a : succ.applicable_actions(nd.s))
            {
                ++r.generated;
                const State t = succ.apply(nd.s, a.label());
                const bool is_new = generated.insert(t).second;
                bool keep = false, skip = false;
                if (sp.k < 0)
                    keep = is_new;
                else if (sp.k == 0)
                    keep = at_root && (survivors ? is_new : t != nd.s);
                else if (t != nd.s && is_new)
                {
                    const std::vector<u64> tu = novel_tuples(nd.s, t);
                    bool novel = false;
                    for (u64 x : tu)
                        novel |= !seen.contains(x);
                    if (!survivors)
                        seen.insert(tu.begin(), tu.end());
                    if (at_root && sp.continuation)
                    {
                        keep = true;
                        skip = !novel;
                    }
                    else
                        keep = novel;
                }
                if (keep)
                {
                    const u32 u = unsatisfied(task, t);
                    cand.push_back({sp.prefer_more ? u : ~u, Node{t, nd.s, nd.depth + 1, skip}});
                }
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (cand.size() > sp.width)
            cand.resize(sp.width);
        std::vector<Node> next;
        if (survivors && sp.k >= 1)
        {
            std::unordered_set<u64> delta;
            for (auto& [key, nd] : cand)
            {
                bool novel = false;
                for (u64 x : novel_tuples(nd.parent, nd.s))
                    if (!seen.contains(x) && delta.insert(x).second)
                        novel = true;
                if (nd.depth == 1 && sp.continuation)
                {
                    nd.skip = !novel;
                    next.push_back(nd);
                }
                else if (novel)
                    next.push_back(nd);
            }
            seen.insert(delta.begin(), delta.end());
        }
        else
            for (auto& [key, nd] : cand)
                next.push_back(nd);
        r.in_tree += next.size();
        layer = std::move(next);
    }
    succ.set_witness_pruning(w0);
    succ.set_canonical_order(c0);
    return r;
}

/// Records the expanded states.
struct Expansions : SearchObserver
{
    std::vector<State> order;
    void on_expand(u64, StateView s) override { order.emplace_back(s); }
};

std::string describe(const std::string& task, const RefSpec& sp)
{
    return task + " k " + std::to_string(sp.k) + " width " + std::to_string(sp.width) +
           (sp.mode == Novelty::SurvivorsOnly ? " survivors_only" : " all_tested") + (sp.prefer_more ? " more" : " fewer") +
           (sp.continuation ? " continuation" : "");
}

void expect_pass(const IwPassStatistics& p, const RefPass& ref, const std::string& what)
{
    EXPECT_EQ(p.status, ref.status) << what;
    EXPECT_EQ(p.expanded, ref.expanded) << what;
    EXPECT_EQ(p.generated, ref.generated) << what;
    EXPECT_EQ(p.generated_in_tree, ref.in_tree) << what;
    EXPECT_EQ(p.skipped, ref.skipped) << what;
}

/// One pass of the IW family engine with the classic pruner (or the width-0 rule).
IwResult engine_pass(const Task& task, u32 k, bool continuation, const LayerOrdering& lo, SearchObserver* obs = nullptr)
{
    Successors& succ = task.workspace().successors();
    SearchControl control;
    control.observer = obs;
    const sd::GoalTest goal = sd::GoalTest::from_spec(task, control.goal);
    const sd::BlockedSet blocked(control.blocked_states);
    sd::Env env(task, succ, goal, blocked, control);
    sd::LayerOrderer layers(task, lo);
    env.layers = &layers;
    const State root = task.initial_state();
    return sd::novelty_ladder(env, k, k, false,
                              [&](u32 a, sd::PassOut& po)
                              {
                                  if (a == 0)
                                  {
                                      sd::NullPruner p;
                                      sd::novelty_pass(env, root, p, sd::PassConfig{.arity = 0, .root = sd::RootRule::ArityZero}, po);
                                      return;
                                  }
                                  sd::ClassicPruner p(task, a, {});
                                  sd::novelty_pass(env, root, p,
                                                   sd::PassConfig{.arity = a,
                                                                  .root = continuation ? sd::RootRule::Continuation : sd::RootRule::Normal},
                                                   po);
                              });
}

std::vector<RefSpec> oracle_specs(int k, bool continuation)
{
    std::vector<RefSpec> out;
    for (u32 width : {1u, 3u, 16u})
        for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
            for (bool more : {true, false})
                out.push_back({.k = k, .width = width, .mode = mode, .prefer_more = more, .continuation = continuation,
                               .max_expanded = 20000});
    return out;
}

const char* const k_oracle_tasks[] = {"gripper__prob05", "blocks__probBLOCKS-8-0", "miconic__s7-4", "rovers__p02",
                                      "visitall__visitall_x-6_y-3_r-100"};

// ------------------------------------------------------------------------------------------------- hand-made tasks
// A graph walk: move(x, y): at(x), adj(x, y) -> at(y), not at(x), visited(y); with Second::Jump, a second schema
// jump(x, y) that also forgets visited(x); with Second::Twin, a second schema with the effects of move. Objects
// o0..o4, start at o0 with visited(o0); `edges` lists adj(x, y).
enum class Second
{
    None,
    Jump,
    Twin,
};
std::shared_ptr<const Task> walk_task(const std::vector<std::pair<u32, u32>>& edges, const std::string& goal, Second second)
{
    std::string text = "O 5\nP 3\nS 2 adj\nF 1 at\nF 1 visited\nSI " + std::to_string(edges.size()) + "\n";
    for (const auto& [x, y] : edges)
        text += "0 " + std::to_string(x) + " " + std::to_string(y) + "\n";
    text += "FI 2\n1 0\n2 0\n" + goal + "A " + std::string(second == Second::None ? "1" : "2") + "\n";
    text += "move 2\nL 2\n1 1 0\n0 1 0 1\nE 1\n0\nL 0\nF 3\n1 1 1\n1 0 0\n2 1 1\n";
    if (second == Second::Jump)
        text += "jump 2\nL 2\n1 1 0\n0 1 0 1\nE 1\n0\nL 0\nF 4\n1 1 1\n1 0 0\n2 1 1\n2 0 0\n";
    if (second == Second::Twin)
        text += "twin 2\nL 2\n1 1 0\n0 1 0 1\nE 1\n0\nL 0\nF 3\n1 1 1\n1 0 0\n2 1 1\n";
    std::istringstream in(text);
    return Task::create(formalism::read_task_text(in));
}

/// Counts the successors reported as entering the tree (the beam's candidates).
struct Candidates : SearchObserver
{
    u64 entered = 0;
    void on_generate(u64, const Action&, u64, StateView, bool is_new) override { entered += is_new; }
};
}  // namespace

// ------------------------------------------------------------------------------------------------- oracle tests
TEST(BeamOracle, IwPassesEqualTheReference)
{
    u32 compared = 0;
    for (const char* name : k_oracle_tasks)
    {
        const auto task = Task::from_text_file(task_path(name));
        for (int k : {0, 1, 2})
            for (const RefSpec& sp : oracle_specs(k, false))
            {
                const RefPass ref = ref_beam(*task, sp);
                if (ref.status == SearchStatus::OutOfStates)
                    continue;  // above the budget of this build
                ++compared;
                const std::string what = describe(name, sp);
                IwOptions o;
                o.layers = beam(sp.width, sp.mode, sp.prefer_more);
                const IwResult fast = iw_pass(*task, static_cast<u32>(k), o);
                ASSERT_EQ(fast.passes.size(), 1u) << what;
                expect_pass(fast.passes[0], ref, what);
                EXPECT_EQ(fast.plan.size(), ref.plan_length) << what;
                Expansions obs;  // the observed (slow) path
                o.control.observer = &obs;
                const IwResult slow = iw_pass(*task, static_cast<u32>(k), o);
                expect_pass(slow.passes[0], ref, what + " observed");
                EXPECT_TRUE(obs.order == ref.order) << what << ": the expansion order differs";
                // the IW family engine (classic pruner) runs the same pass
                Expansions eobs;
                const IwResult eng = engine_pass(*task, static_cast<u32>(k), false, o.layers, &eobs);
                ASSERT_EQ(eng.passes.size(), 1u) << what;
                expect_pass(eng.passes[0], ref, what + " engine");
                EXPECT_TRUE(eobs.order == ref.order) << what << ": the engine's expansion order differs";
            }
    }
    EXPECT_GE(compared, 180u);
}

TEST(BeamOracle, OptimizedIw1EqualsTheReference)
{
    u32 compared = 0;
    for (const char* name : k_oracle_tasks)
    {
        const auto task = Task::from_text_file(task_path(name));
        for (const RefSpec& sp : oracle_specs(1, true))
        {
            const RefPass ref = ref_beam(*task, sp);
            if (ref.status == SearchStatus::OutOfStates)
                continue;
            ++compared;
            const std::string what = describe(name, sp);
            IwOptions o;
            o.max_arity = 1;
            o.layers = beam(sp.width, sp.mode, sp.prefer_more);
            const IwResult r = iw(*task, o);
            ASSERT_EQ(r.passes.size(), 2u) << what;
            EXPECT_TRUE(r.passes[0].placeholder) << what;
            expect_pass(r.passes[1], ref, what);
            const IwResult eng = engine_pass(*task, 1, true, o.layers);
            expect_pass(eng.passes[0], ref, what + " engine");
        }
    }
    EXPECT_GE(compared, 60u);
}

TEST(BeamOracle, BrfsEqualsTheReference)
{
    u32 compared = 0;
    for (const char* name : k_oracle_tasks)
    {
        const auto task = Task::from_text_file(task_path(name));
        for (const RefSpec& sp : oracle_specs(-1, false))
        {
            const RefPass ref = ref_beam(*task, sp);
            if (ref.status == SearchStatus::OutOfStates)
                continue;
            ++compared;
            for (BrfsOptions::Store store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked})
            {
                const std::string what = describe(name, sp) + (store == BrfsOptions::Store::Flat ? " flat" : " chunked");
                BrfsOptions o;
                o.store = store;
                o.witness_pruning = false;
                o.stop_at_goal = true;
                o.layer_stats = true;
                o.layers = beam(sp.width, sp.mode, sp.prefer_more);
                const BrfsResult r = brfs(*task, o);
                EXPECT_EQ(r.solved, ref.status == SearchStatus::Solved) << what;
                EXPECT_EQ(r.exhausted, ref.status == SearchStatus::Exhausted) << what;
                EXPECT_EQ(r.expanded, ref.expanded) << what;
                EXPECT_EQ(r.generated, ref.generated) << what;
                EXPECT_EQ(r.plan.size(), ref.plan_length) << what;
                // every expanded layer below the root holds at most beam_width states
                for (usize d = 1; d < r.layer_counts.size(); ++d)
                    EXPECT_LE(r.layer_counts[d][0], sp.width) << what << " layer " << d;
            }
        }
    }
    EXPECT_GE(compared, 56u);
}

TEST(BeamOracle, LiwWithoutLandmarksEqualsTheReference)
{
    u32 compared = 0;
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0", "visitall__visitall_x-6_y-3_r-100"})
    {
        const auto task = Task::from_text_file(task_path(name));
        for (int k : {1, 2})
            for (const RefSpec& sp : oracle_specs(k, false))
            {
                if (sp.mode != Novelty::AllTested)
                    continue;  // LIW refuses SurvivorsOnly
                const RefPass ref = ref_beam(*task, sp);
                if (ref.status == SearchStatus::OutOfStates)
                    continue;
                ++compared;
                LiwOptions o;
                o.max_arity = static_cast<u32>(k);
                o.layers = beam(sp.width, sp.mode, sp.prefer_more);
                const IwResult r = liw(*task, o);
                if (r.status == SearchStatus::Solved && static_cast<int>(r.effective_width) < k)
                    continue;  // an earlier pass solved it
                ASSERT_EQ(r.passes.size(), static_cast<usize>(k) + 1) << describe(name, sp);
                expect_pass(r.passes[static_cast<usize>(k)], ref, describe(name, sp) + " liw");
            }
    }
    EXPECT_GE(compared, 36u);
}

TEST(BeamOracle, AbstractedIwOnUnaryFluentsEqualsTheReference)
{
    u32 compared = 0;
    // visitall's fluents are unary: each atom has exactly one abstracted feature, so abstracted IW(w) is the
    // continuation pass of IW(w) (its read-only test included)
    const auto task = Task::from_text_file(task_path("visitall__visitall_x-6_y-3_r-100"));
    for (u32 w : {1u, 2u})
        for (const RefSpec& sp : oracle_specs(static_cast<int>(w), true))
        {
            const RefPass ref = ref_beam(*task, sp);
            if (ref.status == SearchStatus::OutOfStates)
                continue;
            ++compared;
            AbstractedIwOptions o;
            o.width = w;
            o.layers = beam(sp.width, sp.mode, sp.prefer_more);
            const IwResult r = abstracted_iw(*task, o);
            ASSERT_EQ(r.passes.size(), 1u) << describe("visitall", sp);
            expect_pass(r.passes[0], ref, describe("visitall", sp) + " abstracted");
            ProjectiveIwOptions p;
            p.layers = o.layers;
            if (w == 1)
                expect_pass(projective_iw(*task, p).passes[0], ref, describe("visitall", sp) + " projective");
        }
    EXPECT_GE(compared, 24u);
}

// ------------------------------------------------------------------------------------------------- semantics
TEST(BeamSemantics, KeepsTheTopEntriesWithTiesInGenerationOrder)
{
    // o0 -> o1, o2, o3, then o1, o2, o3 -> o4. The objects where an expanded state is.
    auto where = [](const Task& task, const std::vector<State>& order)
    {
        std::vector<u32> out;
        for (const State& s : order)
            for (u32 x = 0; x < 5; ++x)
                if (s.contains(task.find_atom(PredicateId{1}, std::vector<ObjectId>{ObjectId{x}})))
                    out.push_back(x);
        return out;
    };
    auto via = [](const std::vector<Action>& plan) { return plan.empty() ? ~u32{0} : plan[0].binding[1].v; };
    const std::vector<std::pair<u32, u32>> edges{{0, 1}, {0, 2}, {0, 3}, {1, 4}, {2, 4}, {3, 4}};
    // visited(o3) and visited(o4): o3 scores best, the beam of width 1 keeps it
    const auto top = walk_task(edges, "G 2\n2 1 3\n2 1 4\n", Second::None);
    // visited(o4): every depth-1 state ties, the first ones generated are kept
    const auto tie = walk_task(edges, "G 1\n2 1 4\n", Second::None);
    for (const auto& [task, width, expected] :
         {std::tuple{top, 1u, std::vector<u32>{0, 3}}, std::tuple{tie, 2u, std::vector<u32>{0, 1, 2}}})
    {
        Expansions obs;
        IwOptions o;
        o.layers = beam(width, Novelty::AllTested);
        o.control.observer = &obs;
        const IwResult r = iw_pass(*task, 1, o);
        EXPECT_EQ(where(*task, obs.order), expected) << "width " << width;
        EXPECT_EQ(via(r.plan), expected[1]) << "width " << width;
        BrfsOptions b;
        b.stop_at_goal = true;
        b.layers = o.layers;
        const BrfsResult br = brfs(*task, b);
        EXPECT_EQ(via(br.plan), expected[1]) << "brfs width " << width;
        EXPECT_EQ(br.expanded, expected.size()) << "brfs width " << width;
    }
    // fewer satisfied goals first: o3 is ranked last and dropped
    IwOptions o;
    o.layers = beam(2, Novelty::AllTested, false);
    Expansions obs;
    o.control.observer = &obs;
    (void)iw_pass(*top, 1, o);
    EXPECT_EQ(where(*top, obs.order), (std::vector<u32>{0, 1, 2, 4}));  // the goal needs o3: never reached
}

TEST(BeamSemantics, AllTestedMarksDroppedSuccessorsSurvivorsOnlyDoesNot)
{
    // o0 -> o1 -> o3 -> o2 and o0 -> o2; the goal visited(o1), visited(o4) prefers the o1 branch. Width 1 drops the
    // o0 -> o2 successor. AllTested marked at(o2) and visited(o2) when it tested that successor, so reaching o2 again
    // via o3 is not novel; SurvivorsOnly marked nothing for it, so that state enters.
    const auto task = walk_task({{0, 1}, {0, 2}, {1, 3}, {3, 2}}, "G 2\n2 1 1\n2 1 4\n", Second::None);
    IwOptions o;
    o.layers = beam(1, Novelty::AllTested);
    const IwResult all = iw_pass(*task, 1, o);
    EXPECT_EQ(all.passes[0].expanded, 3u);  // o0, o1, o3
    EXPECT_EQ(all.passes[0].generated, 4u);
    EXPECT_EQ(all.passes[0].generated_in_tree, 2u);
    o.layers = beam(1, Novelty::SurvivorsOnly);
    const IwResult surv = iw_pass(*task, 1, o);
    EXPECT_EQ(surv.passes[0].expanded, 4u);  // o0, o1, o3, o2
    EXPECT_EQ(surv.passes[0].generated, 4u);
    EXPECT_EQ(surv.passes[0].generated_in_tree, 3u);
    // without a beam both keep both depth-1 states
    o.layers = beam(~u32{0} - 1, Novelty::AllTested);
    EXPECT_EQ(iw_pass(*task, 1, o).passes[0].generated_in_tree, 3u);
}

TEST(BeamSemantics, SurvivorsOnlyReplayCanShrinkTheLayer)
{
    // move(o0, o1) and jump(o0, o1) reach two states that add the same atoms at(o1), visited(o1). AllTested admits
    // only the first; SurvivorsOnly admits both as candidates (read-only test), the beam of width 2 keeps both, and
    // the replay drops the second: the next layer holds one state, not beam_width.
    const auto task = walk_task({{0, 1}}, "G 1\n2 1 4\n", Second::Jump);
    for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
    {
        const bool surv = mode == Novelty::SurvivorsOnly;
        Candidates obs;
        IwOptions o;
        o.layers = beam(2, mode);
        o.control.observer = &obs;
        const IwResult r = iw_pass(*task, 1, o);
        EXPECT_EQ(obs.entered, surv ? 2u : 1u);
        EXPECT_EQ(r.passes[0].generated_in_tree, 1u);
        EXPECT_EQ(r.passes[0].expanded, 2u);
        o.control.observer = nullptr;
        EXPECT_EQ(iw_pass(*task, 1, o).passes[0].generated_in_tree, 1u);
        // the engine
        Candidates eobs;
        const IwResult e = engine_pass(*task, 1, false, o.layers, &eobs);
        EXPECT_EQ(eobs.entered, surv ? 2u : 1u);
        EXPECT_EQ(e.passes[0].generated_in_tree, 1u);
    }
}

TEST(BeamSemantics, WidthZeroSurvivorsOnlyPrunesDuplicateEntries)
{
    // mimir's ArityZero beam selection: AllTested keeps a duplicate root successor as a second entry (it takes a beam
    // slot), SurvivorsOnly prunes it as not new
    // move and twin to o1 and o2: 4 successors, 2 states
    const auto task = walk_task({{0, 1}, {0, 2}}, "G 1\n2 1 4\n", Second::Twin);
    IwOptions o;
    o.layers = beam(8, Novelty::AllTested);
    IwResult r = iw_pass(*task, 0, o);
    EXPECT_EQ(r.passes[0].generated, 4u);
    EXPECT_EQ(r.passes[0].generated_in_tree, 4u);
    EXPECT_EQ(r.passes[0].expanded, 3u);  // a duplicate entry is popped but not expanded again
    o.layers = beam(3, Novelty::AllTested);
    EXPECT_EQ(iw_pass(*task, 0, o).passes[0].generated_in_tree, 3u);  // the duplicate takes a slot
    o.layers = beam(8, Novelty::SurvivorsOnly);
    r = iw_pass(*task, 0, o);
    EXPECT_EQ(r.passes[0].generated, 4u);
    EXPECT_EQ(r.passes[0].generated_in_tree, 2u);
    EXPECT_EQ(r.passes[0].expanded, 3u);
    // jump forgets visited(o0): the 4 successors are distinct states, both modes keep them
    const auto jump = walk_task({{0, 1}, {0, 2}}, "G 1\n2 1 4\n", Second::Jump);
    EXPECT_EQ(iw_pass(*jump, 0, o).passes[0].generated_in_tree, 4u);
}

TEST(BeamSemantics, RandomTiesAreDeterministicPerSeed)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    auto run = [&](bool random, u64 seed)
    {
        IwOptions o;
        o.layers = beam(3, Novelty::AllTested);
        o.layers.randomize_ties = random;
        o.layers.seed = seed;
        Expansions obs;
        o.control.observer = &obs;
        const IwResult r = iw(*task, o);
        BrfsOptions b;
        b.layers = o.layers;
        b.stop_at_goal = true;
        const BrfsResult br = brfs(*task, b);
        return std::make_tuple(obs.order, r.total.expanded, br.expanded, br.plan.size());
    };
    const auto plain = run(false, 0);
    EXPECT_TRUE(run(false, 7) == plain);  // without randomize_ties the seed changes nothing
    const auto a = run(true, 7);
    EXPECT_TRUE(run(true, 7) == a);
    u32 differ = 0;
    for (u64 seed : {1u, 2u, 3u, 4u})
        differ += run(true, seed) != plain;
    EXPECT_GE(differ, 3u);  // gripper's layers are full of equal scores
}

TEST(BeamSemantics, ABeamWiderThanEveryLayerKeepsThemWhole)
{
    // with AllTested and a width above every layer's size the beam changes nothing (generated_in_tree included)
    for (const char* name : {"blocks__probBLOCKS-8-0", "miconic__s7-4"})
    {
        const auto task = Task::from_text_file(task_path(name));
        LayerOrdering plain;
        plain.kind = LayerOrdering::Kind::GoalCount;
        const LayerOrdering wide = beam(1u << 30, Novelty::AllTested);
        auto same = [&](const IwResult& a, const IwResult& b)
        {
            ASSERT_EQ(a.passes.size(), b.passes.size()) << name;
            for (usize i = 0; i < a.passes.size(); ++i)
            {
                EXPECT_EQ(a.passes[i].expanded, b.passes[i].expanded) << name << " pass " << i;
                EXPECT_EQ(a.passes[i].generated, b.passes[i].generated) << name << " pass " << i;
                EXPECT_EQ(a.passes[i].generated_in_tree, b.passes[i].generated_in_tree) << name << " pass " << i;
            }
            EXPECT_EQ(a.plan, b.plan) << name;
        };
        IwOptions o;
        o.layers = plain;
        const IwResult ip = iw(*task, o);
        o.layers = wide;
        same(ip, iw(*task, o));
        LiwOptions l;
        l.layers = plain;
        const IwResult lp = liw(*task, l);
        l.layers = wide;
        same(lp, liw(*task, l));
        AbstractedIwOptions ab;
        ab.layers = plain;
        const IwResult ap = abstracted_iw(*task, ab);
        ab.layers = wide;
        same(ap, abstracted_iw(*task, ab));
        for (BrfsOptions::Store store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked})
        {
            BrfsOptions b;
            b.store = store;
            b.stop_at_goal = true;
            b.layers = plain;
            const BrfsResult bp = brfs(*task, b);
            b.layers = wide;
            const BrfsResult bw = brfs(*task, b);
            EXPECT_EQ(bp.expanded, bw.expanded) << name;
            EXPECT_EQ(bp.generated, bw.generated) << name;
            EXPECT_EQ(bp.states, bw.states) << name;
            EXPECT_EQ(bp.plan, bw.plan) << name;
        }
    }
}

TEST(BeamSemantics, BrfsModesAreTheSameAndSiwRuns)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    BrfsOptions b;
    b.stop_at_goal = true;
    b.layers = beam(4, Novelty::AllTested);
    const BrfsResult all = brfs(*task, b);
    b.layers = beam(4, Novelty::SurvivorsOnly);
    const BrfsResult surv = brfs(*task, b);
    EXPECT_EQ(all.expanded, surv.expanded);
    EXPECT_EQ(all.plan, surv.plan);
    EXPECT_TRUE(all.solved);
    // SIW forwards the layers to the IW ladder of every subproblem
    const auto gripper = Task::from_text_file(task_path("gripper__prob05"));
    SiwOptions plain;
    plain.max_arity = 2;
    const SiwResult p = siw(*gripper, plain);
    ASSERT_EQ(p.status, SearchStatus::Solved);
    for (u32 width : {1u, 4u, 64u})
        for (Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
        {
            const std::string what = std::to_string(width) + (mode == Novelty::AllTested ? " all_tested" : " survivors_only");
            SiwOptions s = plain;
            s.layers = beam(width, mode);
            const SiwResult r = siw(*gripper, s);
            ASSERT_FALSE(r.subproblems.empty()) << what;
            EXPECT_TRUE(r.message.empty()) << r.message;
            if (r.status == SearchStatus::Solved)
            {
                EXPECT_TRUE(gripper->is_goal(*r.goal_state)) << what;
            }
            else
            {
                EXPECT_EQ(r.status, SearchStatus::Exhausted) << what;  // a subproblem the beam cannot solve
            }
            if (width == 1)
            {
                EXPECT_NE(r.total.expanded, p.total.expanded) << what;
            }
        }
}

// ------------------------------------------------------------------------------------------------- refusals
TEST(BeamRefusals, InvalidOptionsAreRefused)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    std::vector<std::pair<LayerOrdering, std::string>> bad;
    {
        LayerOrdering lo = beam(0, Novelty::AllTested);
        bad.push_back({lo, "beam_width must be positive"});
        lo = beam(4, Novelty::AllTested);
        lo.kind = LayerOrdering::Kind::Queue;
        bad.push_back({lo, "beam_width requires an ordered layer kind"});
        lo = beam(4, Novelty::AllTested);
        lo.max_next_layer_states = 10;
        bad.push_back({lo, "mutually exclusive"});
        lo = LayerOrdering{};
        lo.kind = LayerOrdering::Kind::GoalCount;
        lo.beam_novelty = Novelty::SurvivorsOnly;
        bad.push_back({lo, "requires a beam"});
        lo = beam(4, Novelty::AllTested);
        lo.kind = LayerOrdering::Kind::InOrder;
        lo.randomize_ties = true;
        bad.push_back({lo, "randomize_ties requires Kind::GoalCount"});
    }
    for (const auto& [lo, msg] : bad)
    {
        IwOptions o;
        o.layers = lo;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Failed) << msg;
        EXPECT_NE(r.message.find(msg), std::string::npos) << r.message;
        EXPECT_EQ(siw(*task, o).status, SearchStatus::Failed) << msg;
        LiwOptions l;
        l.layers = lo;
        EXPECT_EQ(liw(*task, l).status, SearchStatus::Failed) << msg;
        AbstractedIwOptions a;
        a.layers = lo;
        EXPECT_EQ(abstracted_iw(*task, a).status, SearchStatus::Failed) << msg;
        BrfsOptions b;
        b.layers = lo;
        EXPECT_THROW((void)brfs(*task, b), std::invalid_argument) << msg;
    }
    // every ordered kind takes a beam
    for (auto kind : {LayerOrdering::Kind::InOrder, LayerOrdering::Kind::Reverse, LayerOrdering::Kind::Randomized})
    {
        IwOptions o;
        o.layers = beam(2, Novelty::SurvivorsOnly);
        o.layers.kind = kind;
        EXPECT_NE(iw(*task, o).status, SearchStatus::Failed);
    }
    // LIW: no read-only test
    LiwOptions l;
    l.layers = beam(4, Novelty::SurvivorsOnly);
    const IwResult lr = liw(*task, l);
    EXPECT_EQ(lr.status, SearchStatus::Failed);
    EXPECT_NE(lr.message.find("AllTested"), std::string::npos) << lr.message;
    // coordination: only the queued BrFS
    SearchCoordination coord;
    AbstractedIwOptions a;
    a.layers = beam(4, Novelty::AllTested);
    a.control.coordination = &coord;
    EXPECT_EQ(abstracted_iw(*task, a).status, SearchStatus::Failed);
    l.layers = beam(4, Novelty::AllTested);
    l.control.coordination = &coord;
    EXPECT_EQ(liw(*task, l).status, SearchStatus::Failed);
    // the multi-threaded BrFS orders layers only with a beam; the concurrent and compact stores order none
    BrfsOptions b;
    b.layers = beam(4, Novelty::AllTested);
    b.layers.beam_width = ~u32{0};
    b.threads = 2;
    EXPECT_THROW((void)brfs(*task, b), std::invalid_argument);
    b.layers = beam(4, Novelty::AllTested);
    b.store = BrfsOptions::Store::Concurrent;
    EXPECT_THROW((void)brfs(*task, b), std::invalid_argument);
    b.threads = 1;
    b.store = BrfsOptions::Store::Compact;
    EXPECT_THROW((void)brfs(*task, b), std::invalid_argument);
}

TEST(BeamRefusals, TransitionOrderingIsRefused)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    struct Keep : TransitionOrdering
    {
        void order(const Task&, std::span<const LayerTransition>, std::vector<u32>&) const override {}
    };
    IwOptions o;
    o.layers = beam(4, Novelty::AllTested);
    o.transition_ordering = std::make_shared<Keep>();
    const IwResult r = iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_NE(r.message.find("transition ordering"), std::string::npos) << r.message;
}

// ------------------------------------------------------------------------------------------------- threads
TEST(BeamConcurrency, BeamSearchesPerThreadOverOneTask)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    auto options = [](u32 i)
    {
        IwOptions o;
        o.layers = beam(i % 2 ? 4 : 16, i % 3 ? Novelty::AllTested : Novelty::SurvivorsOnly);
        o.layers.randomize_ties = i % 4 == 0;
        o.layers.seed = i;
        return o;
    };
    constexpr u32 n = 8;
    std::vector<IwResult> ref, out(n);
    std::vector<BrfsResult> bref, bout(n);
    for (u32 i = 0; i < n; ++i)
    {
        ref.push_back(iw(*task, options(i)));
        BrfsOptions b;
        b.stop_at_goal = true;
        b.layers = options(i).layers;
        bref.push_back(brfs(*task, b));
    }
    std::vector<std::thread> ts;
    for (u32 i = 0; i < n; ++i)
        ts.emplace_back(
            [&, i]
            {
                out[i] = iw(*task, options(i));
                BrfsOptions b;
                b.stop_at_goal = true;
                b.layers = options(i).layers;
                bout[i] = brfs(*task, b);
            });
    for (auto& t : ts)
        t.join();
    for (u32 i = 0; i < n; ++i)
    {
        EXPECT_EQ(out[i].status, ref[i].status) << i;
        EXPECT_EQ(out[i].plan, ref[i].plan) << i;
        EXPECT_EQ(out[i].total.expanded, ref[i].total.expanded) << i;
        EXPECT_EQ(out[i].total.generated, ref[i].total.generated) << i;
        EXPECT_EQ(bout[i].expanded, bref[i].expanded) << i;
        EXPECT_EQ(bout[i].plan, bref[i].plan) << i;
    }
}
