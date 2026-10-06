// Relaxed reachability (reachability/relaxed_reachability.hpp), the approximate and lifted fact landmark generators
// (landmarks/approximate.hpp, landmarks/lifted.hpp), the landmark transition ordering of IW's width-1 pass
// (landmarks/transition_ordering.hpp, search/transition_ordering.hpp) and the lifted heuristic fallback, on the text
// tasks of the lifted suite (suite.hpp). The fork comparison is against the fork's own landmark generator; these tests
// check what holds without the fork: the reachable set against an independent relaxed exploration through the
// successor generator, restricted fixpoints, witnesses and conjunctive queries against brute force, the landmark
// graphs' invariants and the Pi+ property, the admission order of an ordered layer, and the lifted h_max / h_add
// against the grounded evaluation for atom goals.

#include "../support/suite.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/landmarks/transition_ordering.hpp"
#include "mymyr/reachability/relaxed_reachability.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;
using mymyr::landmarks::format_atom;

namespace
{
// Small enough for the sanitizer builds; together they cover STRIPS, negative preconditions, conditional effects and
// derived predicates (philosophers, openstacks-adl, miconic-simpleadl).
const char* const k_tasks[] = {"gripper__prob05", "blocks__probBLOCKS-8-0", "depot__p02", "philosophers__p03-phil4",
                               "openstacks-opt08-adl__p03", "pegsol-08-strips__p22", "miconic-simpleadl__s10-2", "driverlog__p03"};

std::shared_ptr<const Task> load(const std::string& name) { return Task::from_text_file(task_path(name)); }

/// Deterministic 64-bit LCG (no std distributions: the draws must be identical on every platform).
struct Lcg
{
    u64 x;
    u32 next(u32 n)
    {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<u32>((x >> 33) % n);
    }
};

/// Whether some action or axiom condition has a negated fluent or derived literal (the relaxed exploration through
/// the successor generator then under-approximates the delete relaxation).
bool has_negative_conditions(const Task& task)
{
    const auto& T = task.data();
    auto neg = [&](const formalism::Condition& c)
    {
        for (const auto& l : T.literals_of(c))
            if (!l.positive && T.predicates[l.pred.v].kind != formalism::PredKind::Static)
                return true;
        return false;
    };
    for (const auto& s : T.schemas)
    {
        if (neg(s.precondition))
            return true;
        for (const auto& ce : T.effects_of(s))
            if (neg(ce.condition))
                return true;
    }
    for (const auto& x : T.axioms)
        if (neg(x.body))
            return true;
    return false;
}

/// The delete-relaxed fixpoint through the successor generator: S grows by the adds of every action applicable in S.
std::set<CanonicalAtom> relaxed_by_successors(const Task& task)
{
    Successors& succ = task.workspace().successors();
    const State s0 = task.initial_state();
    std::vector<u64> w(s0.data(), s0.data() + s0.size_words());
    for (bool changed = true; changed;)
    {
        changed = false;
        std::vector<u32> adds;
        succ.for_each_applicable(StateView{w.data(), static_cast<u32>(w.size()), nullptr, 0},
                                 [&](const ActionLabel&, const Delta& d)
                                 {
                                     for (SlotId x : d.add)
                                         adds.push_back(x.v);
                                     return true;
                                 });
        for (u32 x : adds)
        {
            if (bits::word_of(x) >= w.size())
                w.resize(bits::word_of(x) + 1, 0);
            if (!bits::test(w.data(), static_cast<u32>(w.size()), x))
            {
                bits::set(w.data(), x);
                changed = true;
            }
        }
    }
    std::set<CanonicalAtom> out;
    bits::for_each(w.data(), static_cast<u32>(w.size()), [&](u64 slot) { out.insert(task.atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot))); });
    return out;
}

std::set<CanonicalAtom> reachable_fluent(const Task& task, const reachability::Table& t)
{
    std::set<CanonicalAtom> out;
    const auto& T = task.data();
    for (u32 p = 0; p < T.predicates.size(); ++p)
        if (T.predicates[p].kind == formalism::PredKind::Fluent)
            for (CanonicalAtom c : t.atoms(PredicateId{p}))
                out.insert(c);
    return out;
}

std::vector<CanonicalAtom> goal_atoms(const Task& task)
{
    std::vector<CanonicalAtom> out;
    const auto& T = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    for (const auto& l : T.literals_of(T.goal))
    {
        if (!l.positive || T.predicates[l.pred.v].kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> args;
        for (auto t : T.terms_of(l))
            args.push_back(formalism::term_object(t).v);
        out.push_back(L.encode(l.pred.v, args.data()));
    }
    return out;
}

std::set<CanonicalAtom> initial_atoms(const Task& task)
{
    std::set<CanonicalAtom> out;
    const State s0 = task.initial_state();
    bits::for_each(s0.data(), s0.size_words(), [&](u64 slot) { out.insert(task.atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot))); });
    return out;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ relaxed reachability
TEST(RelaxedReachability, EqualsTheRelaxedExplorationOfTheSuccessorGenerator)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto rr = reachability::RelaxedReachability::create(*task);
        const auto mine = reachable_fluent(*task, rr->table());
        const auto oracle = relaxed_by_successors(*task);
        // the successor generator's exploration is exact without negated conditions and a subset otherwise
        for (CanonicalAtom c : oracle)
            EXPECT_TRUE(mine.count(c)) << format_atom(*task, c) << " reached by the successor generator only";
        if (!has_negative_conditions(*task))
        {
            EXPECT_EQ(mine.size(), oracle.size());
        }
        EXPECT_EQ(rr->table().num_fluent_atoms(), mine.size());
        for (CanonicalAtom c : initial_atoms(*task))
            EXPECT_TRUE(rr->is_reachable(c));
        bool goal = true;
        for (CanonicalAtom g : goal_atoms(*task))
            goal = goal && rr->is_reachable(g);
        EXPECT_EQ(rr->goal_reachable(), goal);
        // tuples and atoms agree, and is_reachable by objects agrees with is_reachable by id
        const auto& T = task->data();
        for (u32 p = 0; p < T.predicates.size(); ++p)
        {
            const auto atoms = rr->table().atoms(PredicateId{p});
            const auto tuples = rr->table().tuples(PredicateId{p});
            const u32 ar = T.predicates[p].arity;
            ASSERT_EQ(tuples.size(), atoms.size() * ar);
            for (usize i = 0; i < atoms.size(); ++i)
            {
                std::vector<ObjectId> objs;
                for (u32 k = 0; k < ar; ++k)
                    objs.push_back(ObjectId{tuples[i * ar + k]});
                EXPECT_TRUE(rr->table().is_reachable(PredicateId{p}, objs));
                EXPECT_EQ(task->atoms().layout().encode(p, tuples.data() + i * ar), atoms[i]);
            }
        }
    }
}

TEST(RelaxedReachability, EveryReachableStateStaysInside)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto rr = reachability::RelaxedReachability::create(*task);
        Successors& succ = task->workspace().successors();
        Lcg rng{7};
        for (int walk = 0; walk < 5; ++walk)
        {
            State s = task->initial_state();
            for (int step = 0; step < 30; ++step)
            {
                bits::for_each(s.data(), s.size_words(),
                               [&](u64 slot) { EXPECT_TRUE(rr->is_reachable(task->atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot)))); });
                const auto acts = succ.applicable_actions(s);
                if (acts.empty())
                    break;
                s = succ.apply(s, acts[rng.next(static_cast<u32>(acts.size()))].label());
            }
        }
    }
}

TEST(RelaxedReachability, RestrictedFixpointsAndWitnesses)
{
    for (const char* name : {"gripper__prob05", "depot__p02", "philosophers__p03-phil4", "miconic-simpleadl__s10-2"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto rr = reachability::RelaxedReachability::create(*task);
        ASSERT_TRUE(rr->table().has_witnesses());
        const auto all = reachable_fluent(*task, rr->table());
        const auto init = initial_atoms(*task);
        std::vector<CanonicalAtom> cands;
        for (CanonicalAtom c : all)
            if (!init.count(c))
                cands.push_back(c);
        for (usize i = 0; i < cands.size() && i < 25; ++i)
        {
            const CanonicalAtom a = cands[i * std::max<usize>(1, cands.size() / 25) % cands.size()];
            const std::vector<CanonicalAtom> forbid{a};
            const reachability::Table r = rr->restricted(forbid);
            EXPECT_FALSE(r.is_reachable(a));
            const auto sub = reachable_fluent(*task, r);
            for (CanonicalAtom c : sub)
                EXPECT_TRUE(all.count(c));
            EXPECT_EQ(rr->goal_reachable_without(forbid), r.goal_reachable());
            const reachability::WitnessQuery q = rr->table().witness_query(forbid);
            for (CanonicalAtom c : all)
            {
                if (q.avoids(c) == reachability::WitnessVerdict::ReachableWithout)
                {
                    EXPECT_TRUE(r.is_reachable(c)) << format_atom(*task, c) << " without " << format_atom(*task, a);
                }
            }
        }
    }
}

TEST(RelaxedReachability, ConjunctiveQueriesEqualBruteForce)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto rr = reachability::RelaxedReachability::create(*task);
        const auto& T = task->data();
        for (u32 p = 0; p < T.predicates.size(); ++p)
        {
            if (T.predicates[p].kind == formalism::PredKind::Static || T.predicates[p].arity != 2)
                continue;
            const auto tuples = rr->table().tuples(PredicateId{p});
            // p(x, y) and p(y, z), x != z
            reachability::ConjunctiveQuery q;
            q.num_variables = 3;
            q.literals.push_back({PredicateId{p}, {reachability::query_variable(0), reachability::query_variable(1)}, true});
            q.literals.push_back({PredicateId{p}, {reachability::query_variable(1), reachability::query_variable(2)}, true});
            q.disequalities.emplace_back(reachability::query_variable(0), reachability::query_variable(2));
            const auto proj = rr->table().project(q);
            std::vector<std::set<u32>> brute(3);
            for (usize i = 0; i < tuples.size(); i += 2)
                for (usize j = 0; j < tuples.size(); j += 2)
                    if (tuples[i + 1] == tuples[j] && tuples[i] != tuples[j + 1])
                    {
                        brute[0].insert(tuples[i]);
                        brute[1].insert(tuples[i + 1]);
                        brute[2].insert(tuples[j + 1]);
                    }
            ASSERT_EQ(proj.size(), 3u);
            for (u32 v = 0; v < 3; ++v)
            {
                std::set<u32> got;
                for (ObjectId o : proj[v])
                    got.insert(o.v);
                EXPECT_EQ(got, brute[v]) << "predicate " << T.str(T.predicates[p].name) << " variable " << v;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ landmark graphs
TEST(ApproximateLandmarks, GraphInvariants)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        landmarks::ApproximateFactLandmarkOptions o;
        o.max_disjunctive_landmark_size = 4;
        const auto g = landmarks::approximate_fact_landmarks(*task, o);
        std::set<CanonicalAtom> lms(g.landmarks().begin(), g.landmarks().end());
        EXPECT_EQ(lms.size(), g.landmarks().size());
        for (CanonicalAtom c : goal_atoms(*task))
            EXPECT_TRUE(lms.count(c)) << "goal atom " << format_atom(*task, c) << " is not a landmark";
        for (const auto& [a, b] : g.orderings())
        {
            EXPECT_TRUE(lms.count(a));
            EXPECT_TRUE(lms.count(b));
        }
        for (const auto& d : g.disjunctive())
        {
            EXPECT_GE(d.size(), 1u);  // the fork keeps singleton sets that hold no fact landmark
            EXPECT_LE(d.size(), 4u);
            for (CanonicalAtom c : d)
                EXPECT_FALSE(lms.count(c)) << "a disjunctive landmark holds the fact landmark " << format_atom(*task, c);
        }
        ASSERT_TRUE(g.has_achiever_index());
        const auto& ix = g.achievers();
        for (CanonicalAtom c : g.landmarks())
        {
            const auto ach = ix.achievers_of(c);
            const std::set<u32> as(ach.begin(), ach.end());
            for (u32 f : ix.first_achievers_of(c))
                EXPECT_TRUE(as.count(f)) << "a first achiever of " << format_atom(*task, c) << " is not an achiever";
        }
        for (u32 a = 0; a < ix.actions.size(); ++a)
            for (CanonicalAtom c : ix.uniquely_achieved_by(a))
                EXPECT_EQ(g.unique_achiever(c), a);
        // the same graph from a shared grounding
        const auto relaxed = heuristics::RelaxedTask::build(*task, {}, nullptr, heuristics::RelaxedTaskOptions{.keep_unreachable_operators = true});
        ASSERT_TRUE(relaxed);
        const auto g2 = landmarks::approximate_fact_landmarks(*relaxed, o);
        EXPECT_TRUE(std::equal(g.landmarks().begin(), g.landmarks().end(), g2.landmarks().begin(), g2.landmarks().end()));
        EXPECT_EQ(g.disjunctive(), g2.disjunctive());
    }
}

TEST(LiftedLandmarks, FactLandmarksBlockTheRelaxedGoal)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto rr = reachability::RelaxedReachability::create(*task);
        for (auto dis : {landmarks::ReachabilityDisambiguation::Joint, landmarks::ReachabilityDisambiguation::PerLiteral,
                         landmarks::ReachabilityDisambiguation::Off})
        {
            landmarks::LiftedFactLandmarkOptions o;
            o.reachability_disambiguation = dis;
            landmarks::FactLandmarkGraph g;
            ASSERT_NO_THROW(g = landmarks::lifted_fact_landmarks(*rr, o));  // verify_pi_plus runs inside
            EXPECT_FALSE(g.has_achiever_index());
            const auto init = initial_atoms(*task);
            const auto goals = goal_atoms(*task);
            std::set<CanonicalAtom> lms(g.landmarks().begin(), g.landmarks().end());
            for (CanonicalAtom c : goals)
                EXPECT_TRUE(lms.count(c));
            if (!rr->goal_reachable())
                continue;
            for (CanonicalAtom c : g.landmarks())
            {
                if (!init.count(c) && std::find(goals.begin(), goals.end(), c) == goals.end())
                {
                    EXPECT_FALSE(rr->goal_reachable_without(std::vector<CanonicalAtom>{c})) << format_atom(*task, c);
                }
            }
            // complete_fact_landmarks = Members: no member of a partial record is a landmark it missed
            for (const auto& l : g.lifted())
                if (!l.is_fact())
                    for (CanonicalAtom c : l.members)
                    {
                        if (!lms.count(c) && !init.count(c))
                        {
                            EXPECT_TRUE(rr->goal_reachable_without(std::vector<CanonicalAtom>{c})) << format_atom(*task, c);
                        }
                    }
            // orderings: between fact landmarks
            for (const auto& [a, b] : g.orderings())
                EXPECT_TRUE(lms.count(a) && lms.count(b));
        }
        // verify_pi_plus on a wrong graph: the initial state's atoms are skipped, a non-landmark throws
        std::vector<CanonicalAtom> fake;
        for (CanonicalAtom c : reachable_fluent(*task, rr->table()))
            if (!initial_atoms(*task).count(c) && rr->goal_reachable_without(std::vector<CanonicalAtom>{c}))
            {
                fake.push_back(c);
                break;
            }
        if (!fake.empty() && rr->goal_reachable())
        {
            EXPECT_THROW(landmarks::verify_pi_plus_fact_landmarks(*rr, landmarks::FactLandmarkGraph::create(fake)), std::logic_error);
        }
    }
}

// ------------------------------------------------------------------------------------------------ transition ordering
namespace
{
/// Records per layer (parent depth) the admission-order events of the width-1 pass with their scores.
class LayerRecorder final : public search::SearchObserver
{
public:
    explicit LayerRecorder(const landmarks::LandmarkTransitionOrdering& o) : m_o(o) {}
    void on_expand(u64 id, StateView s) override
    {
        auto& n = m_nodes[id];
        n.words.assign(s.w, s.w + s.nw);
    }
    void on_generate(u64 parent, const Action& a, u64 child, StateView cs, bool is_new) override
    {
        const auto& p = m_nodes.at(parent);
        const auto sc = m_o.score(StateView{p.words.data(), static_cast<u32>(p.words.size()), nullptr, 0}, a.label(), cs);
        if (layers.size() <= p.depth)
            layers.resize(p.depth + 1);
        layers[p.depth].push_back(sc);
        if (is_new)
            m_nodes[child].depth = p.depth + 1;
    }
    std::vector<std::vector<landmarks::LandmarkTransitionScore>> layers;

private:
    struct Node
    {
        std::vector<u64> words;
        u32 depth = 0;
    };
    const landmarks::LandmarkTransitionOrdering& m_o;
    std::map<u64, Node> m_nodes;
};

/// Reverses every layer (a TransitionOrdering that is not the landmark one).
class Reverse final : public search::TransitionOrdering
{
public:
    void order(const Task&, std::span<const search::LayerTransition> layer, std::vector<u32>& order) const override
    {
        for (u32 i = static_cast<u32>(layer.size()); i-- > 0;)
            order.push_back(i);
    }
};

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

TEST(TransitionOrdering, LayersAreAdmittedInPreferenceOrder)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto graph = landmarks::approximate_fact_landmarks(*task);
        const auto ordering = std::make_shared<const landmarks::LandmarkTransitionOrdering>(*task, graph);
        LayerRecorder rec(*ordering);
        search::IwOptions o;
        o.max_arity = 1;
        o.transition_ordering = ordering;
        o.control.observer = &rec;
        o.control.budget.max_expanded = 20000;
        const auto r = search::iw(*task, o);
        if (r.status == search::SearchStatus::Solved)
        {
            EXPECT_TRUE(valid_plan(*task, r.plan));
        }
        u64 transitions = 0;
        for (const auto& layer : rec.layers)
        {
            transitions += layer.size();
            for (usize i = 1; i < layer.size(); ++i)
                EXPECT_FALSE(ordering->prefer(layer[i], layer[i - 1])) << "layer transition " << i << " should come earlier";
        }
        EXPECT_GT(transitions, 0u);
        // any ordering keeps the pass sound
        search::IwOptions ro;
        ro.max_arity = 1;
        ro.transition_ordering = std::make_shared<const Reverse>();
        ro.control.budget.max_expanded = 20000;
        const auto rr = search::iw(*task, ro);
        if (rr.status == search::SearchStatus::Solved)
        {
            EXPECT_TRUE(valid_plan(*task, rr.plan));
        }
    }
}

TEST(TransitionOrdering, NeedsAnAchieverIndex)
{
    const auto task = load("gripper__prob05");
    const auto lifted = landmarks::lifted_fact_landmarks(*task);
    EXPECT_THROW(landmarks::LandmarkTransitionOrdering(*task, lifted), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ lifted heuristic fallback
TEST(LiftedHeuristics, EqualTheGroundedEvaluationForAtomGoals)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        Successors& succ = task->workspace().successors();
        for (auto kind : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
        {
            // grounded where the grounding covers the state and the goals (Auto; a negated goal atom without a "false"
            // proposition in the grounding falls back to the lifted evaluation, which is then not compared)
            heuristics::Options g;
            g.kind = kind;
            auto hg = heuristics::make_heuristic(*task, g);
            heuristics::Options l = g;
            l.evaluation = heuristics::Evaluation::Lifted;
            u64 compared = 0;
            auto hl = heuristics::make_heuristic(*task, l);
            Lcg rng{11};
            State s = task->initial_state();
            std::vector<State> seen{s};
            for (int step = 0; step < 25; ++step)
            {
                // task goal
                u64 before = hg->stats().grounded;
                const double vg = hg->evaluate(s), vl = hl->evaluate(s);
                if (hg->stats().grounded > before)
                {
                    ++compared;
                    if (kind == heuristics::Kind::FF)
                        EXPECT_EQ(std::isinf(vg), std::isinf(vl));
                    else
                        EXPECT_EQ(vg, vl) << "task goal, step " << step;
                }
                // atom goals: two goals drawn from the atoms of earlier states, one of them negated
                std::vector<search::GoalSpec::AtomGoal> goals(2);
                for (auto& goal : goals)
                {
                    const State& t = seen[rng.next(static_cast<u32>(seen.size()))];
                    std::vector<u32> slots;
                    bits::for_each(t.data(), t.size_words(), [&](u64 x) { slots.push_back(static_cast<u32>(x)); });
                    if (slots.empty())
                        continue;
                    goal.positive.push_back(SlotId{slots[rng.next(static_cast<u32>(slots.size()))]});
                    goal.positive.push_back(SlotId{slots[rng.next(static_cast<u32>(slots.size()))]});
                    if (step % 2)  // negated goal atoms on odd steps only: the grounding often lacks their propositions
                        goal.negative.push_back(SlotId{slots[rng.next(static_cast<u32>(slots.size()))]});
                }
                before = hg->stats().grounded;
                const double ag = hg->evaluate(s, goals), al = hl->evaluate(s, goals);
                if (hg->stats().grounded > before)
                {
                    ++compared;
                    if (kind == heuristics::Kind::FF)
                        EXPECT_EQ(std::isinf(ag), std::isinf(al));
                    else
                        EXPECT_EQ(ag, al) << "atom goals, step " << step;
                }
                const auto acts = succ.applicable_actions(s);
                if (acts.empty())
                    break;
                s = succ.apply(s, acts[rng.next(static_cast<u32>(acts.size()))].label());
                seen.push_back(s);
            }
            EXPECT_GT(hl->stats().lifted, 0u);
            EXPECT_GT(compared, 30u);
        }
    }
}
