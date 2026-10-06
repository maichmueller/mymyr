// The IW family variants (search/aiw.hpp, liw.hpp, rollout_iw.hpp, parallel_rollouts.hpp, portfolio.hpp) and
// their shared BrFS engine (src/mymyr/search/novelty_brfs.hpp):
//   - the engine's classic IW ladder equals search::iw() pass for pass on the suite, on the fast path, on the slow
//     (tracked) path and with in-order layers;
//   - further sections test each variant (fork parity of the counts is in test_iw_variants_golden.cpp, against
//     tests/data/iw_variants/fork_golden.json).

#include "../../../src/mymyr/search/novelty_brfs.hpp"
#include "../support/suite.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/novelty/landmark_table.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;
namespace sd = mymyr::search::detail;

namespace
{
std::string param_name(std::string n)
{
    for (char& c : n)
        if (!std::isalnum(static_cast<unsigned char>(c)))
            c = '_';
    return n;
}

struct EngineRun
{
    bool tracked = false;
    LayerOrdering layers;
};

IwResult run_classic(const Task& task, u32 k, EngineRun how, u32* num_states = nullptr, const SearchControl& control = {})
{
    Successors& succ = task.workspace().successors();
    const sd::GoalTest goal = sd::GoalTest::from_spec(task, control.goal);
    const sd::BlockedSet blocked(control.blocked_states);
    sd::Env env(task, succ, goal, blocked, control);
    sd::StateTracker tracker(task, true, true);
    if (how.tracked)
        env.tracker = &tracker;
    sd::LayerOrderer layers(task, how.layers);
    if (how.layers.kind != LayerOrdering::Kind::Queue)
        env.layers = &layers;
    const State root = task.initial_state();
    IwResult r = sd::classic_ladder(env, root, k, true, WidthZero::ExpandDepthOne, {});
    sd::finish_result(env, root, r);
    if (num_states)
        *num_states = tracker.num_states();
    return r;
}

void expect_same_passes(const IwResult& a, const IwResult& b, const std::string& what)
{
    EXPECT_EQ(a.status, b.status) << what;
    EXPECT_EQ(a.plan, b.plan) << what;
    EXPECT_EQ(a.cost, b.cost) << what;
    ASSERT_EQ(a.passes.size(), b.passes.size()) << what;
    for (usize i = 0; i < a.passes.size(); ++i)
    {
        EXPECT_EQ(a.passes[i].arity, b.passes[i].arity) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].status, b.passes[i].status) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].expanded, b.passes[i].expanded) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].generated, b.passes[i].generated) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].generated_in_tree, b.passes[i].generated_in_tree) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].skipped, b.passes[i].skipped) << what << " pass " << i;
        EXPECT_EQ(a.passes[i].placeholder, b.passes[i].placeholder) << what << " pass " << i;
    }
}

class EngineOnSuite : public ::testing::TestWithParam<SuiteTask>
{
};

TEST_P(EngineOnSuite, ClassicLadderEqualsIw)
{
    const SuiteTask& t = GetParam();
    const auto task = Task::from_text_file(task_path(t.name));
    for (u32 k : {0u, 1u, 2u})
    {
        IwOptions o;
        o.max_arity = k;
        const IwResult ref = iw(*task, o);
        expect_same_passes(run_classic(*task, k, {}), ref, t.name + " fast k=" + std::to_string(k));
        u32 states = 0;
        expect_same_passes(run_classic(*task, k, {.tracked = true, .layers = {}}, &states), ref, t.name + " tracked k=" + std::to_string(k));
        for (const IwPassStatistics& p : ref.passes)
            EXPECT_GE(states, p.expanded) << "every expanded state is a distinct created state";
        expect_same_passes(run_classic(*task, k, {.layers = {.kind = LayerOrdering::Kind::InOrder}}), ref,
                           t.name + " in order k=" + std::to_string(k));
    }
}

TEST_P(EngineOnSuite, LayerOrderingsOfIwEqualTheClassicLadders)
{
    // search::iw and the engine of the IW family variants implement the orderings independently
    const SuiteTask& t = GetParam();
    const auto task = Task::from_text_file(task_path(t.name));
    using K = LayerOrdering::Kind;
    const std::vector<LayerOrdering> orderings{
        {.kind = K::InOrder},
        {.kind = K::Reverse},
        {.kind = K::Randomized, .seed = 3},
        {.kind = K::GoalCount},
        {.kind = K::GoalCount, .prefer_more_satisfied_goals = false},
        {.kind = K::Reverse, .max_next_layer_states = 7},
        {.kind = K::Randomized, .seed = 5, .max_next_layer_states = 3},
        {.kind = K::GoalCount, .max_next_layer_states = 2},
    };
    for (u32 k : {0u, 1u, 2u})
        for (usize i = 0; i < orderings.size(); ++i)
        {
            IwOptions o;
            o.max_arity = k;
            o.layers = orderings[i];
            o.control.budget.max_expanded = 20000;
            const IwResult ref = iw(*task, o);
            ASSERT_NE(ref.status, SearchStatus::Failed) << ref.message;
            SearchControl control;
            control.budget.max_expanded = 20000;
            expect_same_passes(run_classic(*task, k, {.layers = orderings[i]}, nullptr, control), ref,
                               t.name + " ordering " + std::to_string(i) + " k=" + std::to_string(k));
            if (i == 0)
            {
                IwOptions q;
                q.max_arity = k;
                q.control.budget.max_expanded = 20000;
                expect_same_passes(iw(*task, q), ref, t.name + " in order = queue k=" + std::to_string(k));
            }
        }
}

INSTANTIATE_TEST_SUITE_P(Suite, EngineOnSuite, ::testing::ValuesIn(suite()), [](const auto& i) { return param_name(i.param.name); });

// ================================================================================================ oracle
// A literal re-implementation of the documented pass semantics (the fork's brfs.cpp with its pruning strategies):
// every successor materialized with Successors::apply, a duplicate check on admitted states (the fork's "is new"),
// novelty over std::set features computed from scratch. Shares no code with the engine or the pruners.
using Atoms = std::vector<u32>;  // ascending slots

Atoms atoms_of(const State& s)
{
    Atoms out;
    for (SlotId x : s.slots())
        out.push_back(x.v);
    return out;
}

bool has(const Atoms& a, u32 x) { return std::binary_search(a.begin(), a.end(), x); }

/// Calls f(subset) for every subset of `a` of size lo..hi (ascending elements).
template<class F>
void subsets(const Atoms& a, u32 lo, u32 hi, F&& f)
{
    std::vector<u32> t;
    auto rec = [&](auto& self, usize from) -> void
    {
        if (t.size() >= lo)
            f(t);
        if (t.size() == hi)
            return;
        for (usize i = from; i < a.size(); ++i)
        {
            t.push_back(a[i]);
            self(self, i + 1);
            t.pop_back();
        }
    };
    rec(rec, 0);
}

struct OracleNovelty
{
    virtual ~OracleNovelty() = default;
    virtual bool init(const Atoms& s) = 0;
    virtual bool test(const Atoms& parent, const Atoms& child) = 0;
};

enum class ORoot
{
    Normal,
    ArityZero,
    Continuation,
};

struct OPass
{
    SearchStatus status = SearchStatus::Exhausted;
    u64 expanded = 0, generated = 0, in_tree = 0, skipped = 0;
    u32 plan_length = 0;
};

OPass oracle_pass(const Task& task, const State& root, OracleNovelty* nov, ORoot rule, bool keep_depth_one)
{
    Successors& succ = task.workspace().successors();
    const bool w0 = succ.witness_pruning(), c0 = succ.canonical_order();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);
    OPass r;
    struct Node
    {
        State s;
        u32 depth;
        bool skip;
    };
    std::vector<Node> q;
    std::set<Atoms> in_tree;
    if (!nov || nov->init(atoms_of(root)))
    {
        q.push_back({root, 0, false});
        in_tree.insert(atoms_of(root));
    }
    for (usize i = 0; i < q.size(); ++i)
    {
        const Node nd = q[i];
        if (task.is_goal(nd.s))
        {
            r.status = SearchStatus::Solved;
            r.plan_length = nd.depth;
            break;
        }
        ++r.expanded;
        if (nd.skip)
        {
            ++r.skipped;
            continue;
        }
        const bool is_root = i == 0;
        const Atoms pa = atoms_of(nd.s);
        for (const Action& a : succ.applicable_actions(nd.s))
        {
            ++r.generated;
            const State t = succ.apply(nd.s, a.label());
            const Atoms ta = atoms_of(t);
            if (ta == pa)
                continue;  // self loop
            if (rule == ORoot::ArityZero)
            {
                if (!is_root)
                    continue;
                ++r.in_tree;
                if (in_tree.insert(ta).second)
                    q.push_back({t, 1, false});
                continue;
            }
            if (in_tree.count(ta))
                continue;  // not new
            const bool novel = nov->test(pa, ta);
            if (rule == ORoot::Continuation && is_root)
            {
                ++r.in_tree;
                in_tree.insert(ta);
                q.push_back({t, 1, !novel && !keep_depth_one});
                continue;
            }
            if (!novel)
                continue;
            ++r.in_tree;
            in_tree.insert(ta);
            q.push_back({t, nd.depth + 1, false});
        }
    }
    succ.set_witness_pruning(w0);
    succ.set_canonical_order(c0);
    return r;
}

/// Coordinates: the ranks of the groups with a true member, or BOT (= groups.size()).
struct OracleRanks
{
    std::vector<Atoms> groups;
    [[nodiscard]] std::set<u32> of(const Atoms& s) const
    {
        std::set<u32> r;
        for (u32 g = 0; g < groups.size(); ++g)
            for (u32 x : groups[g])
                if (has(s, x))
                    r.insert(g);
        if (r.empty())
            r.insert(static_cast<u32>(groups.size()));
        return r;
    }
    void split(const Atoms& p, const Atoms& c, std::vector<u32>& flipped, std::vector<u32>& kept) const
    {
        const std::set<u32> rp = of(p), rc = of(c);
        flipped.clear();
        kept.clear();
        for (u32 r : rc)
            (rp.count(r) ? kept : flipped).push_back(r);
    }
};

struct LiwOracle : OracleNovelty
{
    OracleRanks ranks;
    u32 k = 1;
    std::set<std::pair<u32, std::vector<u32>>> seen;

    bool mark(u32 r, const Atoms& s, const Atoms* parent)
    {
        bool novel = false;
        if (s.size() + 1 < k)
            return false;
        subsets(s, parent ? 1 : 0, k,
                [&](const std::vector<u32>& t)
                {
                    if (parent && std::all_of(t.begin(), t.end(), [&](u32 x) { return has(*parent, x); }))
                        return;  // no added atom
                    novel = seen.insert({r, t}).second || novel;
                });
        return novel;
    }
    bool init(const Atoms& s) override
    {
        bool novel = false;
        for (u32 r : ranks.of(s))
            novel = mark(r, s, nullptr) || novel;
        return novel;
    }
    bool test(const Atoms& p, const Atoms& c) override
    {
        std::vector<u32> flipped, kept;
        ranks.split(p, c, flipped, kept);
        bool novel = false;
        for (u32 r : flipped)
            novel = mark(r, c, nullptr) || novel;
        for (u32 r : kept)
            novel = mark(r, c, &p) || novel;
        return novel;
    }
};

std::vector<CanonicalAtom> goal_atoms(const Task& task)
{
    const formalism::TaskData& t = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    std::vector<CanonicalAtom> out;
    for (const formalism::Literal& l : t.literals_of(t.goal))
    {
        if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> args;
        for (formalism::Term x : t.terms_of(l))
            args.push_back(formalism::term_object(x).v);
        const CanonicalAtom c = L.encode(l.pred.v, args.data());
        if (c < L.fluent_count)
            out.push_back(c);
    }
    return out;
}

Atoms slots_of(const Task& task, std::span<const CanonicalAtom> atoms)
{
    Atoms out;
    for (CanonicalAtom c : atoms)
        out.push_back(task.atoms().intern(c));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// The oracle's grouping, written from the fork's rules independently of LandmarkCoordinates::make.
OracleRanks oracle_ranks(const Task& task, const LandmarkNovelty& ln)
{
    OracleRanks r;
    if (!ln.graph)
        return r;
    const Atoms facts = slots_of(task, ln.graph->landmarks());
    if (!ln.disjunctive || ln.graph->disjunctive().empty())
    {
        for (u32 x : facts)
            r.groups.push_back({x});
        return r;
    }
    if (ln.all_private)
    {
        std::set<u32> all(facts.begin(), facts.end());
        for (const auto& d : ln.graph->disjunctive())
            for (u32 x : slots_of(task, d))
                all.insert(x);
        for (u32 x : all)
            r.groups.push_back({x});
        return r;
    }
    const Atoms un = slots_of(task, ln.unshared_atoms);
    std::set<u32> singles(facts.begin(), facts.end());
    for (const auto& d : ln.graph->disjunctive())
    {
        Atoms kept;
        for (u32 x : slots_of(task, d))
        {
            if (has(un, x))
                singles.insert(x);
            else
                kept.push_back(x);
        }
        if (!kept.empty())
            r.groups.push_back(kept);
    }
    for (u32 x : singles)
        r.groups.push_back({x});
    return r;
}

/// Landmark settings for the tests: the goal atoms, split into fact landmarks (the first half) and disjunctive pairs
/// of the rest (fact and disjunctive atoms are disjoint, as the graph requires).
struct LmCase
{
    std::string name;
    bool disjunctive = false, all_private = false, unshared = false;
};

LandmarkNovelty make_landmarks(const Task& task, const LmCase& c)
{
    const std::vector<CanonicalAtom> goals = goal_atoms(task);
    std::vector<CanonicalAtom> facts(goals.begin(), goals.begin() + static_cast<std::ptrdiff_t>(goals.size() / 2));
    std::vector<std::vector<CanonicalAtom>> disj;
    for (usize i = goals.size() / 2; i + 1 < goals.size(); i += 2)
        disj.push_back({goals[i], goals[i + 1]});
    LandmarkNovelty ln;
    ln.graph = std::make_shared<const landmarks::FactLandmarkGraph>(landmarks::FactLandmarkGraph::create(facts, disj));
    ln.disjunctive = c.disjunctive;
    ln.all_private = c.all_private;
    if (c.unshared && !disj.empty())
        ln.unshared_atoms = {disj.front().front()};
    return ln;
}

const std::vector<const char*>& oracle_tasks()
{
    static const std::vector<const char*> t = {"gripper__prob05", "blocks__probBLOCKS-8-0", "depot__p02", "freecell__p02",
                                               "philosophers__p03-phil4", "pegsol-08-strips__p22", "visitall__visitall_x-6_y-3_r-100",
                                               "miconic__s7-4", "parcprinter-opt11-strips__p03", "driverlog__p03"};
    return t;
}

void expect_pass(const IwPassStatistics& p, const OPass& o, const std::string& what)
{
    EXPECT_EQ(p.status, o.status) << what;
    EXPECT_EQ(p.expanded, o.expanded) << what;
    EXPECT_EQ(p.generated, o.generated) << what;
    EXPECT_EQ(p.generated_in_tree, o.in_tree) << what;
    EXPECT_EQ(p.skipped, o.skipped) << what;
}
}  // namespace

// ================================================================================================ LIW
TEST(Liw, NoLandmarksEqualsPlainIw)
{
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        for (u32 k : {1u, 2u})
        {
            IwOptions io;
            io.max_arity = k;
            io.optimize_iw1 = false;
            const IwResult ref = iw(*task, io);
            LiwOptions lo;
            lo.max_arity = k;
            expect_same_passes(liw(*task, lo), ref, t.name + " k=" + std::to_string(k));
        }
    }
}

TEST(Liw, EqualsTheOracle)
{
    const std::vector<LmCase> cases = {{"facts"}, {"shared", true}, {"private", true, true}, {"unshared", true, false, true}};
    u32 differs = 0;  // cases whose LIW passes differ from plain IW (the landmark path is exercised)
    for (const char* name : oracle_tasks())
    {
        const auto task = Task::from_text_file(task_path(name));
        for (const LmCase& c : cases)
            for (u32 k : {1u, 2u})
            {
                const std::string what = std::string(name) + " " + c.name + " k=" + std::to_string(k);
                LiwOptions lo;
                lo.max_arity = k;
                lo.landmarks = make_landmarks(*task, c);
                const IwResult r = liw(*task, lo);
                IwOptions io;
                io.max_arity = k;
                io.optimize_iw1 = false;
                const IwResult plain = iw(*task, io);
                differs += plain.total.expanded != r.total.expanded || plain.passes.size() != r.passes.size() ? 1 : 0;
                ASSERT_NE(r.status, SearchStatus::Failed) << what << ": " << r.message;
                const State root = task->initial_state();
                for (usize i = 0; i < r.passes.size(); ++i)
                {
                    OPass o;
                    if (i == 0)
                        o = oracle_pass(*task, root, nullptr, ORoot::ArityZero, false);
                    else
                    {
                        LiwOracle nov;
                        nov.k = static_cast<u32>(i);
                        nov.ranks = oracle_ranks(*task, lo.landmarks);
                        o = oracle_pass(*task, root, &nov, ORoot::Normal, false);
                    }
                    expect_pass(r.passes[i], o, what + " pass " + std::to_string(i));
                    if (o.status == SearchStatus::Solved) {
                        EXPECT_EQ(r.plan.size(), o.plan_length) << what;
                    }
                }
            }
    }
    EXPECT_GT(differs, 4u) << "LIW should prune differently from IW on some cases";
    std::printf("LIW cases differing from IW: %u\n", differs);
}

TEST(Liw, AllPrivateWithUnsharedFails)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    LiwOptions lo;
    lo.max_arity = 1;
    lo.landmarks = make_landmarks(*task, {"bad", true, true, true});
    const IwResult r = liw(*task, lo);
    EXPECT_EQ(r.status, SearchStatus::Failed);
    EXPECT_FALSE(r.message.empty());
}

// ================================================================================================ abstracted IW
namespace
{
/// The abstracted features and tuples, written from the fork's compute_features_for_atom / generate_tuples.
struct AiwOracle : OracleNovelty
{
    const Task* task = nullptr;
    u32 width = 1;
    bool base = false;
    Atoms preserved;
    std::optional<OracleRanks> ranks;
    std::map<std::vector<u32>, u32> ids;
    std::set<std::pair<u32, std::vector<u32>>> seen;

    u32 id(const std::vector<u32>& key) { return ids.emplace(key, static_cast<u32>(ids.size())).first->second; }
    std::vector<u32> features(u32 slot)
    {
        const formalism::TaskData& t = task->data();
        const u32 pred = task->atoms().predicate(SlotId{slot}).v;
        const std::vector<u32> args(task->atoms().arguments(SlotId{slot}).begin(), task->atoms().arguments(SlotId{slot}).end());
        std::vector<u32> full{1, pred};
        full.insert(full.end(), args.begin(), args.end());
        std::vector<u32> out;
        if (has(preserved, slot))
        {
            out.push_back(id(full));
            if (args.size() <= 1)
                return out;
        }
        if (args.empty())
            return {id(full)};
        for (u32 i = 0; i < args.size(); ++i)
        {
            std::vector<u32> key{0, pred, i, args[i]};
            if (!base)
                for (u32 j = 0; j < args.size(); ++j)
                {
                    if (j == i)
                        continue;
                    const formalism::Range r = t.objects[args[j]].types;
                    if (r.count == 0)
                    {
                        key.push_back(1);
                        key.push_back(~u32{0});
                        continue;
                    }
                    key.push_back(r.count);
                    for (u32 x = 0; x < r.count; ++x)
                        key.push_back(t.type_ids[r.begin + x].v);
                }
            out.push_back(id(key));
        }
        return out;
    }
    /// Marks the tuples of s (atoms of `parent` are not added; null: all added) under rank r.
    bool mark(u32 r, const Atoms& s, const Atoms* parent)
    {
        std::vector<std::vector<u32>> f;
        std::vector<bool> added;
        for (u32 a : s)
        {
            f.push_back(features(a));
            added.push_back(!parent || !has(*parent, a));
        }
        bool novel = false;
        auto put = [&](std::vector<u32> t)
        {
            std::sort(t.begin(), t.end());
            if (std::adjacent_find(t.begin(), t.end()) != t.end())
                return;  // repeated feature
            novel = seen.insert({r, t}).second || novel;
        };
        const usize G = s.size();
        for (usize i = 0; i < G; ++i)
            if (added[i])
                for (u32 x : f[i])
                    put({x});
        if (width >= 2)
            for (usize i = 0; i < G; ++i)
                for (usize j = i + 1; j < G; ++j)
                    if (added[i] || added[j])
                        for (u32 x : f[i])
                            for (u32 y : f[j])
                                put({x, y});
        if (width >= 3)
            for (usize i = 0; i < G; ++i)
                for (usize j = i + 1; j < G; ++j)
                    for (usize l = j + 1; l < G; ++l)
                        if (added[i] || added[j] || added[l])
                            for (u32 x : f[i])
                                for (u32 y : f[j])
                                    for (u32 z : f[l])
                                        put({x, y, z});
        return novel;
    }
    bool init(const Atoms& s) override
    {
        if (ranks)
            for (u32 r : ranks->of(s))
                mark(r, s, nullptr);
        else
            mark(0, s, nullptr);
        return true;
    }
    bool test(const Atoms& p, const Atoms& c) override
    {
        if (!ranks)
            return mark(0, c, &p);
        std::vector<u32> flipped, kept;
        ranks->split(p, c, flipped, kept);
        bool novel = false;
        for (u32 r : flipped)
            novel = mark(r, c, nullptr) || novel;
        for (u32 r : kept)
            novel = mark(r, c, &p) || novel;
        return novel;
    }
};

struct AiwCase
{
    u32 width;
    bool base, preserve_goals, keep_depth_one, landmarks;
};
}  // namespace

TEST(AbstractedIw, EqualsTheOracle)
{
    const std::vector<AiwCase> cases = {{1, false, true, false, false}, {1, true, false, false, false}, {1, false, false, true, false},
                                        {2, false, true, false, false}, {2, true, true, false, false},  {3, false, true, false, false},
                                        {1, false, true, false, true},  {2, false, true, false, true}};
    u32 differs = 0;
    for (const char* name : oracle_tasks())
    {
        const auto task = Task::from_text_file(task_path(name));
        for (const AiwCase& c : cases)
        {
            if (c.width == 3 && std::string(name) != "gripper__prob05" && std::string(name) != "pegsol-08-strips__p22")
                continue;  // the oracle's width-3 sets are slow
            const std::string what = std::string(name) + " w=" + std::to_string(c.width) + (c.base ? " base" : "") +
                                     (c.preserve_goals ? " goals" : "") + (c.keep_depth_one ? " keep1" : "") + (c.landmarks ? " lm" : "");
            AbstractedIwOptions o;
            o.width = c.width;
            o.base_abstracted = c.base;
            o.preserve_goal_atoms = c.preserve_goals;
            o.keep_depth_one_novel = c.keep_depth_one;
            if (c.landmarks)
                o.landmarks = make_landmarks(*task, {"shared", true});
            const IwResult r = abstracted_iw(*task, o);
            differs += iw_pass(*task, c.width).passes[0].expanded != r.passes[0].expanded ? 1 : 0;
            ASSERT_EQ(r.passes.size(), 1u) << what << ": " << r.message;
            AiwOracle nov;
            nov.task = task.get();
            nov.width = c.width;
            nov.base = c.base;
            if (c.preserve_goals)
                nov.preserved = slots_of(*task, goal_atoms(*task));
            if (c.landmarks)
            {
                nov.ranks = oracle_ranks(*task, o.landmarks);
                for (const Atoms& g : nov.ranks->groups)
                    nov.preserved.insert(nov.preserved.end(), g.begin(), g.end());
                std::sort(nov.preserved.begin(), nov.preserved.end());
                nov.preserved.erase(std::unique(nov.preserved.begin(), nov.preserved.end()), nov.preserved.end());
            }
            const OPass op = oracle_pass(*task, task->initial_state(), &nov, ORoot::Continuation, c.keep_depth_one);
            expect_pass(r.passes[0], op, what);
            EXPECT_EQ(r.status, op.status) << what;
            if (op.status == SearchStatus::Solved) {
                EXPECT_EQ(r.plan.size(), op.plan_length) << what;
            }
        }
    }
    EXPECT_GT(differs, 4u) << "abstracted novelty should prune differently from IW(k) on some cases";
    std::printf("AIW cases differing from IW: %u\n", differs);
}

TEST(AbstractedIw, ProjectiveAliasAndErrors)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    ProjectiveIwOptions p;
    AbstractedIwOptions a;
    a.width = 1;
    a.base_abstracted = true;
    a.preserve_goal_atoms = false;
    expect_same_passes(projective_iw(*task, p), abstracted_iw(*task, a), "projective = BAIW(1) without goal atoms");
    p.typed_projection = true;
    p.keep_goal_nonunary_atoms = true;
    a.base_abstracted = false;
    a.preserve_goal_atoms = true;
    expect_same_passes(projective_iw(*task, p), abstracted_iw(*task, a), "typed projective = AIW(1) with goal atoms");
    a.width = 4;
    EXPECT_EQ(abstracted_iw(*task, a).status, SearchStatus::Failed);
}

// ================================================================================================ landmark table layouts
namespace
{
std::vector<u64> words_of(const std::vector<u32>& atoms)
{
    std::vector<u64> w;
    for (u32 a : atoms)
    {
        if (bits::word_of(a) >= w.size())
            w.resize(bits::word_of(a) + 1, 0);
        bits::set(w.data(), a);
    }
    return w;
}

std::vector<u32> random_atoms(SplitMix64& rng, const std::vector<u32>& pool, u32 max_size)
{
    std::vector<u32> out;
    const u64 n = rng.bounded(max_size + 1);
    for (u64 i = 0; i < n; ++i)
        out.push_back(pool[rng.bounded(pool.size())]);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
}  // namespace

TEST(LandmarkTable, ChildMaskEqualsMask)
{
    SplitMix64 rng(7);
    const std::vector<std::vector<std::vector<u32>>> groupings = {{{3}, {5}, {9}, {70}}, {{3}, {5, 7}, {7, 9}, {}, {11}, {70, 3}}};
    std::vector<u32> pool;
    for (u32 a = 0; a < 16; ++a)
        pool.push_back(a);
    pool.push_back(70);
    pool.push_back(71);
    for (const auto& groups : groupings)
    {
        const novelty::LandmarkCoordinates coords(groups);
        const u32 mw = coords.mask_words();
        std::vector<u64> pm(mw), cm(mw), inc(mw), f1(mw), k1(mw);
        std::vector<u32> flipped, kept;
        for (int it = 0; it < 4000; ++it)
        {
            const std::vector<u32> p = random_atoms(rng, pool, 8), add = random_atoms(rng, pool, 3), del = random_atoms(rng, pool, 3);
            std::vector<u32> c;
            for (u32 a : p)
                if (!std::binary_search(del.begin(), del.end(), a))
                    c.push_back(a);
            c.insert(c.end(), add.begin(), add.end());
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            std::vector<u32> added;  // the engine's add list: added atoms false in the parent
            for (u32 a : add)
                if (!std::binary_search(p.begin(), p.end(), a))
                    added.push_back(a);
            std::vector<SlotId> dels;
            for (u32 a : del)
                dels.push_back(SlotId{a});
            const std::vector<u64> pw = words_of(p), cw = words_of(c);
            const bool pany = coords.mask(pw.data(), static_cast<u32>(pw.size()), pm.data());
            const bool cany = coords.mask(cw.data(), static_cast<u32>(cw.size()), cm.data());
            const bool iany = coords.child_mask(pm.data(), cw.data(), static_cast<u32>(cw.size()), added, dels, inc.data());
            ASSERT_EQ(cany, iany);
            ASSERT_EQ(cm, inc);
            coords.split(pm.data(), pany, cm.data(), cany, flipped, kept);
            coords.split_masks(pm.data(), pany, cm.data(), cany, f1.data(), k1.data());
            std::vector<u32> f2, k2;
            bits::for_each(f1.data(), mw, [&](u64 r) { f2.push_back(static_cast<u32>(r)); });
            bits::for_each(k1.data(), mw, [&](u64 r) { k2.push_back(static_cast<u32>(r)); });
            ASSERT_EQ(flipped, f2);
            ASSERT_EQ(kept, k2);
        }
    }
}

TEST(LandmarkTable, LayoutsAgree)
{
    // rank-major (default budget), sparse (budget 0) and a table that leaves the rank-major layout for tuple-major rows
    // when the atoms grow past 64, over random operations: every call must agree
    constexpr u32 groups = 100;  // num_ranks 101 (BOT = 100)
    const std::vector<u32> rank_pool = {0, 1, 2, 50, 99, 100};
    for (u32 k : {1u, 2u})
    {
        novelty::LandmarkTableOptions fast, sparse, switching;
        sparse.max_dense_bytes = 0;
        switching.max_dense_bytes = k == 1 ? 3000 : 150000;
        novelty::LandmarkNoveltyTable a(k, groups + 1, 200, fast), b(k, groups + 1, 200, sparse), c(k, groups + 1, 200, switching);
        novelty::LandmarkNoveltyTable* tables[3] = {&a, &b, &c};
        for (auto* t : tables)
            t->reserve(20);
        EXPECT_TRUE(a.dense());
        EXPECT_FALSE(b.dense());
        EXPECT_TRUE(c.dense());
        const u64 c_bytes = c.bytes();
        SplitMix64 rng(11 + k);
        std::vector<u32> pool = {0, 1, 2, 3, 5, 8, 13, 21, 34, 40, 55, 63};
        u32 novel = 0, calls = 0;
        const u32 mw = bits::words_for(groups + 1);
        for (int it = 0; it < 6000; ++it)
        {
            if (it == 3000)
                for (u32 x : {64u, 65u, 90u, 100u, 127u})
                    pool.push_back(x);
            const std::vector<u32> p = random_atoms(rng, pool, 5), w = random_atoms(rng, pool, 5);
            const std::vector<u32> ranks = random_atoms(rng, rank_pool, 3);
            const std::vector<u64> pw = words_of(p), ww = words_of(w);
            const u32 pn = static_cast<u32>(pw.size()), wn = static_cast<u32>(ww.size());
            std::vector<u32> add;
            for (u32 x : w)
                if (!std::binary_search(p.begin(), p.end(), x))
                    add.push_back(x);
            const u64 op = rng.bounded(3);
            bool r[3] = {false, false, false};
            if (op == 0)
                for (int i = 0; i < 3; ++i)
                    r[i] = tables[i]->mark_state(ww.data(), wn, ranks);
            else if (op == 1)
                for (int i = 0; i < 3; ++i)
                    r[i] = tables[i]->mark_transition(pw.data(), pn, ww.data(), wn, ranks);
            else
            {
                std::vector<u64> fm(mw, 0), km(mw, 0);
                for (u32 x : ranks)
                    bits::set(rng.bounded(2) ? fm.data() : km.data(), x);
                for (int i = 0; i < 3; ++i)
                    r[i] = tables[i]->mark_successor(pw.data(), pn, ww.data(), wn, add, fm.data(), km.data());
            }
            ASSERT_EQ(r[0], r[1]) << "k=" << k << " op " << op << " at " << it;
            ASSERT_EQ(r[0], r[2]) << "k=" << k << " op " << op << " at " << it;
            novel += r[0] ? 1 : 0;
            ++calls;
        }
        EXPECT_TRUE(a.dense());
        EXPECT_FALSE(b.dense());
        EXPECT_TRUE(c.dense()) << "the switching table should end tuple-major dense";
        EXPECT_NE(c.bytes(), c_bytes);
        EXPECT_GT(novel, 30u) << "k=" << k;  // the small universe saturates (k = 1: 48 novel calls)
        EXPECT_LT(novel + 500, calls) << "k=" << k;
        std::printf("k=%u: %u of %u calls novel\n", k, novel, calls);
    }
}

TEST(Liw, TableLayoutsAgree)
{
    const std::vector<LmCase> cases = {{"facts"}, {"shared", true}, {"private", true, true}};
    for (const char* name : oracle_tasks())
    {
        const auto task = Task::from_text_file(task_path(name));
        for (const LmCase& c : cases)
            for (u32 k : {1u, 2u})
            {
                LiwOptions lo;
                lo.max_arity = k;
                lo.landmarks = make_landmarks(*task, c);
                const IwResult fast = liw(*task, lo);
                lo.landmarks.tables.max_dense_bytes = 0;
                expect_same_passes(fast, liw(*task, lo), std::string(name) + " " + c.name + " k=" + std::to_string(k) + " sparse");
            }
    }
}

// ================================================================================================ state tracker
TEST(StateTracker, SuccessorRecordingEqualsFullRecording)
{
    // record_successor (the parent's rows and reached atoms reused) equals record over a breadth-first exploration
    // that creates every successor of every expanded state, as the IW family's tracked searches do
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        Successors& succ = task->workspace().successors();
        sd::StateTracker full(*task, true, true), incremental(*task, true, true);
        const State root = task->initial_state();
        full.record(root.data(), root.size_words());
        incremental.record(root.data(), root.size_words());
        std::vector<State> queue{root};
        std::unordered_set<State> seen{root};
        for (usize q = 0; q < queue.size() && q < 300; ++q)
        {
            const State parent = queue[q];
            for (const Action& a : succ.applicable_actions(parent))
            {
                const State child = succ.apply(parent, a.label());
                std::vector<u32> add;
                bits::for_each(child.data(), child.size_words(),
                               [&](u64 x)
                               {
                                   if (!bits::test(parent.data(), parent.size_words(), static_cast<u32>(x)))
                                       add.push_back(static_cast<u32>(x));
                               });
                const bool fresh = full.record(child.data(), child.size_words());
                ASSERT_EQ(incremental.record_successor(child.data(), child.size_words(), add), fresh) << t.name;
                if (seen.insert(child).second)
                    queue.push_back(child);
            }
        }
        auto trimmed = [](std::span<const u64> w) { return std::vector<u64>(w.begin(), w.begin() + bits::trimmed_size(w.data(), static_cast<u32>(w.size()))); };
        EXPECT_EQ(incremental.num_states(), full.num_states()) << t.name;
        EXPECT_EQ(trimmed(incremental.reached()), trimmed(full.reached())) << t.name;
        EXPECT_TRUE(std::ranges::equal(incremental.first_achievers(), full.first_achievers())) << t.name;
        const auto& a = incremental.co_occurrence();
        const auto& b = full.co_occurrence();
        ASSERT_EQ(a.size(), b.size()) << t.name;
        for (usize i = 0; i < a.size(); ++i)
            ASSERT_EQ(trimmed(a[i]), trimmed(b[i])) << t.name << " row " << i;
    }
}

TEST(AbstractedIw, PairLayoutsAgree)
{
    // width 2: the dense pair tables, the pair set (max_dense_bytes 0) and a switch from one to the other mid-search
    // give the same passes, with and without landmark ranks
    u32 switched = 0;
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        for (bool lm : {false, true})
        {
            AbstractedIwOptions o;
            o.width = 2;
            o.control.budget.max_expanded = 3000;
            if (lm)
                o.landmarks = make_landmarks(*task, {"shared", true});
            const IwResult dense = abstracted_iw(*task, o);
            o.landmarks.tables.max_dense_bytes = 0;
            const IwResult set = abstracted_iw(*task, o);
            o.landmarks.tables.max_dense_bytes = 4096;
            const IwResult mixed = abstracted_iw(*task, o);
            const std::string what = t.name + (lm ? " lm" : "");
            expect_same_passes(dense, set, what + " dense = set");
            expect_same_passes(mixed, set, what + " switch = set");
            switched += dense.passes[0].expanded > 50 ? 1 : 0;
        }
    }
    EXPECT_GT(switched, 10u) << "most cases should grow past the small dense budget";
}
