// LM-cut and landmark count on small PDDL tasks with hand-evaluated values: conditional effects, derived predicates,
// zero and fractional costs, states outside the grounding, sound first achievers, the path-dependent landmark status
// (reached, required again, two paths into a state) and its handling in the best-first searches.

#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace mymyr;
namespace H = mymyr::heuristics;

namespace
{
std::shared_ptr<const Task> make(const std::string& domain, const std::string& problem)
{
    const auto d = frontend::Domain::from_string(domain, "d.pddl");
    return Task::create(*d->instantiate_string(problem, "p.pddl"));
}

std::unique_ptr<H::Heuristic> heuristic(const Task& t, H::Kind k, H::Costs costs = H::Costs::Auto)
{
    H::Options o;
    o.kind = k;
    o.costs = costs;
    return H::make_heuristic(t, o);
}

f64 h(const Task& t, H::Kind k, StateView s, H::Costs costs = H::Costs::Auto) { return heuristic(t, k, costs)->evaluate(s); }

Action find(const Task& t, StateView s, const std::string& action)
{
    for (const Action& a : t.workspace().successors().applicable_actions(s))
        if (t.format(a.label()) == action)
            return a;
    ADD_FAILURE() << action << " is not applicable";
    return Action{};
}

/// The states and statuses of a path of named actions from the initial state, progressed by `hc`.
struct Path
{
    std::vector<State> states;
    std::vector<std::vector<u64>> status;
};

Path walk(const Task& t, H::Heuristic& hc, const std::vector<std::string>& actions)
{
    Path p;
    p.states.push_back(t.initial_state());
    p.status.emplace_back(hc.status_words());
    hc.initial_status(p.states[0].view(), p.status[0]);
    for (const std::string& name : actions)
    {
        const StateView s = p.states.back().view();
        const Action a = find(t, s, name);
        State next = t.workspace().successors().apply(s, a.label());
        std::vector<u64> st(hc.status_words());
        hc.progress(p.status.back(), s, a.label(), next.view(), st);
        p.states.push_back(std::move(next));
        p.status.push_back(std::move(st));
    }
    return p;
}

std::set<std::string> landmark_names(const Task& t, const landmarks::FactLandmarkGraph& g)
{
    std::set<std::string> out;
    for (CanonicalAtom c : g.landmarks())
        out.insert(landmarks::format_atom(t, c));
    return out;
}

/// get-y, get-x (needs y) and make-g (needs x) reach the goal atom g, get-h the goal atom h; every atom can be lost
/// again. Landmarks: y, x, g, h with y before x before g (greedy-necessary).
const char* const k_chain_domain = R"((define (domain chain) (:requirements :strips)
 (:predicates (y) (x) (g) (h))
 (:action get-y :parameters () :precondition (and) :effect (y))
 (:action lose-y :parameters () :precondition (y) :effect (not (y)))
 (:action get-x :parameters () :precondition (y) :effect (x))
 (:action lose-x :parameters () :precondition (x) :effect (not (x)))
 (:action make-g :parameters () :precondition (x) :effect (g))
 (:action lose-g :parameters () :precondition (g) :effect (not (g)))
 (:action get-h :parameters () :precondition (and) :effect (h))))";
const char* const k_chain_problem = "(define (problem p) (:domain chain) (:init) (:goal (and (g) (h))))";
}  // namespace

TEST(LandmarkHeuristicsPddl, LmCutChargesAnActionOnceForAllOfItsConditionalEffects)
{
    // one action achieves both goal atoms by two conditional effects: h* = 1, and a cut per effect would sum to 2
    const auto t = make(R"((define (domain d) (:requirements :strips :conditional-effects)
 (:predicates (c1) (c2) (g1) (g2))
 (:action both :parameters () :precondition (and) :effect (and (when (c1) (g1)) (when (c2) (g2))))
 (:action set2 :parameters () :precondition (and) :effect (c2))))",
                        "(define (problem p) (:domain d) (:init (c1) (c2)) (:goal (and (g1) (g2))))");
    const State s0 = t->initial_state();
    EXPECT_EQ(h(*t, H::Kind::LmCut, s0.view()), 1);
    EXPECT_EQ(h(*t, H::Kind::Max, s0.view()), 1);
    // without c2: both, then set2 and both again, or set2 first: h* = 2
    const auto t2 = make(R"((define (domain d) (:requirements :strips :conditional-effects)
 (:predicates (c1) (c2) (g1) (g2))
 (:action both :parameters () :precondition (and) :effect (and (when (c1) (g1)) (when (c2) (g2))))
 (:action set2 :parameters () :precondition (and) :effect (c2))))",
                         "(define (problem p) (:domain d) (:init (c1)) (:goal (and (g1) (g2))))");
    EXPECT_EQ(h(*t2, H::Kind::LmCut, t2->initial_state().view()), 2);
    search::BestFirstOptions o;
    o.heuristic.kind = H::Kind::LmCut;
    const auto r = search::astar_eager(*t2, o);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.cost, 2);
}

TEST(LandmarkHeuristicsPddl, DerivedPredicatesAreFreeOperatorsOutsideEveryCut)
{
    const auto t = make(R"((define (domain d) (:requirements :strips :derived-predicates)
 (:predicates (g1) (g2) (done))
 (:derived (done) (and (g1) (g2)))
 (:action get1 :parameters () :precondition (and) :effect (g1))
 (:action get2 :parameters () :precondition (and) :effect (g2))))",
                        "(define (problem p) (:domain d) (:init) (:goal (done)))");
    const State s0 = t->initial_state();
    EXPECT_EQ(h(*t, H::Kind::Max, s0.view()), 1);
    EXPECT_EQ(h(*t, H::Kind::LmCut, s0.view()), 2);
    EXPECT_EQ(h(*t, H::Kind::LmCount, s0.view()), 3);  // g1, g2 and the derived goal atom
    for (H::Kind k : {H::Kind::LmCut, H::Kind::LmCount})
    {
        search::BestFirstOptions o;
        o.heuristic.kind = k;
        const auto r = search::astar_eager(*t, o);
        ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
        EXPECT_EQ(r.cost, 2);
    }
}

TEST(LandmarkHeuristicsPddl, ZeroAndFractionalCosts)
{
    // s1 (0), s2 (0.25) and s3 (0.5) reach the goal for 0.75, direct for 1
    const auto t = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (a) (b) (goal)) (:functions (total-cost) (price))
 (:action direct :parameters () :precondition (and) :effect (and (goal) (increase (total-cost) 1)))
 (:action s1 :parameters () :precondition (and) :effect (and (a) (increase (total-cost) 0)))
 (:action s2 :parameters () :precondition (a) :effect (and (b) (increase (total-cost) 0.25)))
 (:action s3 :parameters () :precondition (b) :effect (and (goal) (increase (total-cost) (price))))))",
                        R"((define (problem p) (:domain d) (:init (= (total-cost) 0) (= (price) 0.5)) (:goal (goal))
 (:metric minimize (total-cost))))");
    const State s0 = t->initial_state();
    EXPECT_EQ(h(*t, H::Kind::LmCut, s0.view()), 0.75);
    EXPECT_EQ(h(*t, H::Kind::LmCut, s0.view(), H::Costs::Unit), 1);
    // the goal is the only landmark; its cheapest first achiever is s3
    EXPECT_EQ(h(*t, H::Kind::LmCount, s0.view()), 0.5);
    EXPECT_EQ(h(*t, H::Kind::LmCount, s0.view(), H::Costs::Unit), 1);
    search::BestFirstOptions o;
    o.heuristic.kind = H::Kind::LmCut;
    const auto r = search::astar_eager(*t, o);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.cost, 0.75);
    EXPECT_EQ(r.plan.size(), 3u);

    // a chain of landmarks: real costs add their cheapest first achievers
    const auto chain = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (a) (b) (goal) (noise)) (:functions (total-cost))
 (:action s1 :parameters () :precondition (and) :effect (and (a) (increase (total-cost) 0)))
 (:action s2 :parameters () :precondition (a) :effect (and (b) (increase (total-cost) 0.25)))
 (:action s3 :parameters () :precondition (b) :effect (and (goal) (increase (total-cost) 0.5)))
 (:action other :parameters () :precondition (and) :effect (and (noise) (increase (total-cost) 1)))))",
                            R"((define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal))
 (:metric minimize (total-cost))))");
    const State c0 = chain->initial_state();
    auto lmcount = heuristic(*chain, H::Kind::LmCount);
    EXPECT_EQ(lmcount->evaluate(c0.view()), 0.75);
    EXPECT_EQ(h(*chain, H::Kind::LmCount, c0.view(), H::Costs::Unit), 3);
    EXPECT_EQ(h(*chain, H::Kind::LmCut, c0.view()), 0.75);
    // preferred: the applicable action that adds a needed landmark
    ASSERT_TRUE(lmcount->provides_preferred());
    EXPECT_TRUE(lmcount->preferred(find(*chain, c0.view(), "(s1)").label()));
    EXPECT_FALSE(lmcount->preferred(find(*chain, c0.view(), "(other)").label()));
}

TEST(LandmarkHeuristicsPddl, StatesOutsideTheGroundingAreRefused)
{
    // (u a a) lies in the domains of u's arguments but is never reachable: the grounding has no proposition for it
    const auto t = make(R"((define (domain d) (:requirements :strips)
 (:predicates (ok ?x ?y) (u ?x ?y) (g))
 (:action mark :parameters (?x ?y) :precondition (ok ?x ?y) :effect (u ?x ?y))
 (:action use :parameters (?x ?y) :precondition (u ?x ?y) :effect (g))))",
                        "(define (problem p) (:domain d) (:objects a b) (:init (ok a b) (ok b a)) (:goal (g)))");
    const State s0 = t->initial_state();
    const formalism::TaskData& D = t->data();
    auto index = [&](const auto& v, std::string_view name)
    {
        for (u32 i = 0; i < v.size(); ++i)
            if (D.str(v[i].name) == name)
                return i;
        ADD_FAILURE() << name;
        return u32{0};
    };
    const CanonicalLayout& L = t->atoms().layout();
    const u32 args[] = {index(D.objects, "a"), index(D.objects, "a")};
    const CanonicalAtom c = L.encode(index(D.predicates, "u"), args);
    ASSERT_LT(c, L.fluent_count);
    StateBuilder b(s0.view());
    b.set(SlotId{t->atoms().intern(c)});
    for (H::Kind k : {H::Kind::LmCut, H::Kind::LmCount})
    {
        auto hk = heuristic(*t, k);
        EXPECT_EQ(hk->evaluate(s0.view()), 2) << H::to_string(k);
        EXPECT_THROW((void)hk->evaluate(b.view()), std::runtime_error) << H::to_string(k);
    }
}

TEST(LandmarkHeuristicsPddl, PossiblyBeforeFirstAchieversAreSound)
{
    // x is reached first (h_max 1) and its achiever of g first (h_max 2), but the plan get-y1, get-y, from-y never
    // makes x true: h_max first achievers make x a landmark, possibly-before first achievers do not
    const auto t = make(R"((define (domain d) (:requirements :strips)
 (:predicates (x) (y1) (y) (g))
 (:action get-x :parameters () :precondition (and) :effect (x))
 (:action get-y1 :parameters () :precondition (and) :effect (y1))
 (:action get-y :parameters () :precondition (y1) :effect (y))
 (:action from-x :parameters () :precondition (x) :effect (g))
 (:action from-y :parameters () :precondition (y) :effect (g))))",
                        "(define (problem p) (:domain d) (:init) (:goal (g)))");
    landmarks::ApproximateFactLandmarkOptions o;
    o.first_achievers = landmarks::FirstAchievers::PossiblyBefore;
    EXPECT_EQ(landmark_names(*t, landmarks::approximate_fact_landmarks(*t, o)), (std::set<std::string>{"g()"}));
    o.first_achievers = landmarks::FirstAchievers::HMax;
    EXPECT_EQ(landmark_names(*t, landmarks::approximate_fact_landmarks(*t, o)), (std::set<std::string>{"g()", "x()"}));
    EXPECT_EQ(h(*t, H::Kind::LmCount, t->initial_state().view()), 1);
}

TEST(LandmarkHeuristicsPddl, LandmarkCountFollowsThePath)
{
    const auto t = make(k_chain_domain, k_chain_problem);
    auto hc = heuristic(*t, H::Kind::LmCount);
    ASSERT_TRUE(hc->path_dependent());
    ASSERT_GE(hc->status_words(), 1u);
    auto value = [&](const Path& p) { return hc->evaluate(p.states.back().view(), p.status.back()); };
    auto state_only = [&](const Path& p) { return hc->evaluate(p.states.back().view()); };

    EXPECT_EQ(value(walk(*t, *hc, {})), 4);  // y, x, g, h
    EXPECT_EQ(value(walk(*t, *hc, {"(get-y)"})), 3);
    EXPECT_EQ(value(walk(*t, *hc, {"(get-y)", "(get-x)"})), 2);
    // y is lost after x was reached: not needed again; the state alone shows x, so every path to it reached y
    const Path lost_y = walk(*t, *hc, {"(get-y)", "(get-x)", "(lose-y)"});
    EXPECT_EQ(value(lost_y), 2);
    EXPECT_EQ(state_only(lost_y), 2);
    // x is lost before g was reached: needed again
    EXPECT_EQ(value(walk(*t, *hc, {"(get-y)", "(get-x)", "(lose-x)"})), 3);
    // a goal atom that was reached and lost is needed again
    EXPECT_EQ(value(walk(*t, *hc, {"(get-y)", "(get-x)", "(make-g)", "(lose-g)"})), 2);
    const Path lost_xg = walk(*t, *hc, {"(get-y)", "(get-x)", "(make-g)", "(lose-x)", "(lose-g)"});
    EXPECT_EQ(value(lost_xg), 2);  // g and h; x is not needed again once g was reached
    EXPECT_EQ(state_only(lost_xg), 3);  // the state alone shows y only
    EXPECT_EQ(value(walk(*t, *hc, {"(get-y)", "(get-x)", "(make-g)", "(get-h)"})), 0);

    // the initial state again after reaching y and x: x is needed again (g is not reached), y is not
    const Path back = walk(*t, *hc, {"(get-y)", "(get-x)", "(lose-y)", "(lose-x)"});
    EXPECT_EQ(value(back), 3);
    // reached on both paths: the landmarks both reached (none)
    std::vector<u64> merged = back.status.back();
    const Path start = walk(*t, *hc, {});
    EXPECT_TRUE(hc->merge(merged, start.status[0]));
    EXPECT_EQ(merged, start.status[0]);
    EXPECT_FALSE(hc->merge(merged, back.status.back()));
    EXPECT_EQ(hc->evaluate(back.states.back().view(), merged), 4);

    // the searches solve it with the landmark count
    for (auto search : {search::gbfs_eager, search::gbfs_lazy, search::astar_eager, search::astar_lazy, search::beam})
    {
        search::BestFirstOptions o;
        o.heuristic.kind = H::Kind::LmCount;
        const auto r = search(*t, o);
        ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
        EXPECT_EQ(r.plan.size(), 4u);
        EXPECT_EQ(r.initial_h, 4);
    }
}

TEST(LandmarkHeuristicsPddl, LandmarkCountInfersTheStatusOfAnotherStartState)
{
    // y holds initially and has no achiever; use-y spends it on x, make-g turns x into the goal g and leaves had-g,
    // from which restore-g gets g back. Landmarks: y, x, g with y before x before g (greedy-necessary).
    const auto t = make(R"((define (domain d) (:requirements :strips)
 (:predicates (y) (x) (g) (had-g))
 (:action use-y :parameters () :precondition (y) :effect (and (x) (not (y))))
 (:action make-g :parameters () :precondition (x) :effect (and (g) (had-g)))
 (:action lose-x :parameters () :precondition (x) :effect (not (x)))
 (:action lose-g :parameters () :precondition (g) :effect (not (g)))
 (:action restore-g :parameters () :precondition (had-g) :effect (g))))",
                        "(define (problem p) (:domain d) (:init (y)) (:goal (g)))");
    auto hc = heuristic(*t, H::Kind::LmCount);
    const Path p = walk(*t, *hc, {"(use-y)", "(make-g)", "(lose-x)", "(lose-g)"});
    const StateView s = p.states.back().view();  // had-g only
    // on the path, g is needed again (restore-g)
    EXPECT_EQ(hc->evaluate(s, p.status.back()), 1);
    // from s alone: every path to s reached y; x and g count as unreached, but y, which has no achiever, is not
    // needed again, as a path to s may well have reached x
    EXPECT_EQ(hc->evaluate(s), 2);
    std::vector<u64> inferred(hc->status_words());
    hc->initial_status(s, inferred);
    EXPECT_EQ(hc->evaluate(s, inferred), 2);
    // without the inference, a path that reached y only makes s a dead end (y is needed again for x)
    EXPECT_EQ(hc->evaluate(s, p.status[0]), H::k_dead_end);
    // a status merged with an inferred one is inferred
    std::vector<u64> merged = p.status.back();
    EXPECT_TRUE(hc->merge(merged, inferred));
    EXPECT_EQ(merged, inferred);
    EXPECT_FALSE(hc->merge(merged, p.status.back()));
    EXPECT_EQ(hc->evaluate(s, merged), 2);

    // a search started in s finds restore-g
    for (auto search : {search::gbfs_eager, search::gbfs_lazy, search::astar_eager, search::astar_lazy})
    {
        search::BestFirstOptions o;
        o.heuristic.kind = H::Kind::LmCount;
        o.start = p.states.back();
        const auto res = search(*t, o);
        ASSERT_EQ(res.status, search::SearchStatus::Solved) << res.message;
        EXPECT_EQ(res.plan.size(), 1u);
        EXPECT_EQ(res.initial_h, 2);
    }
}

namespace
{
/// A path-dependent heuristic that records the statuses it evaluates: bit 0 marks a path from the start, bit 1 + i
/// the action schema i on the path. h is 0.
struct Recorder final : H::Heuristic
{
    const Task& task;
    bool batch = false;
    std::vector<std::pair<std::string, u64>> seen;  // (true atoms, status) per evaluation

    explicit Recorder(const Task& t) : task(t) {}
    [[nodiscard]] H::Kind kind() const noexcept override { return H::Kind::Custom; }
    [[nodiscard]] bool path_dependent() const noexcept override { return true; }
    [[nodiscard]] bool batched() const noexcept override { return batch; }
    [[nodiscard]] u32 status_words() const noexcept override { return 1; }
    void initial_status(StateView, std::span<u64> out) override { out[0] = 1; }
    void progress(std::span<const u64> parent, StateView, const ActionLabel& a, StateView, std::span<u64> out) override
    {
        out[0] = parent[0] | u64{1} << (1 + a.schema.v);
    }
    H::Value evaluate(StateView s) override
    {
        const u64 start = 1;
        return evaluate(s, std::span<const u64>(&start, 1));
    }
    H::Value evaluate(StateView s, std::span<const u64> status) override
    {
        std::string atoms;
        for (const std::string& a : task.format_atoms(s))
            atoms += a;
        seen.emplace_back(atoms, status[0]);
        return 0;
    }
    H::Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal>) override { return evaluate(s); }
};

u64 status_of(const Recorder& r, const std::string& atoms)
{
    u64 out = ~u64{0};
    for (const auto& [a, s] : r.seen)
        if (a == atoms)
            out = s;  // the last evaluation
    return out;
}
}  // namespace

TEST(LandmarkHeuristicsPddl, SearchesMergeTheStatusesOfEveryPathIntoAState)
{
    // a1 and b1 both reach (p); fin reaches the goal from (p)
    const auto t = make(R"((define (domain d) (:requirements :strips)
 (:predicates (p) (goal))
 (:action a1 :parameters () :precondition (and) :effect (p))
 (:action b1 :parameters () :precondition (and) :effect (p))
 (:action fin :parameters () :precondition (p) :effect (goal))))",
                        "(define (problem p) (:domain d) (:init) (:goal (goal)))");
    for (auto search : {search::gbfs_eager, search::gbfs_lazy, search::astar_eager, search::astar_lazy, search::beam})
    {
        Recorder rec(*t);
        search::BestFirstOptions o;
        o.evaluator = &rec;
        const auto r = search(*t, o);
        ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
        EXPECT_EQ(status_of(rec, ""), 1u);
        EXPECT_EQ(status_of(rec, "(p)"), 1u);  // both paths: the actions of neither
    }
    Recorder rec(*t);
    search::AStarIwOptions o;
    o.evaluator = &rec;
    const auto r = search::astar_iw(*t, o);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_EQ(status_of(rec, "(p)"), 1u);

    // A*: (q) is reached for 3 by expensive, then for 2 through cheap1, cheap2; the cheaper path re-evaluates it
    const auto w = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (m) (q) (goal)) (:functions (total-cost))
 (:action expensive :parameters () :precondition (and) :effect (and (q) (increase (total-cost) 3)))
 (:action cheap1 :parameters () :precondition (and) :effect (and (m) (increase (total-cost) 1)))
 (:action cheap2 :parameters () :precondition (m) :effect (and (q) (not (m)) (increase (total-cost) 1)))
 (:action fin :parameters () :precondition (q) :effect (and (goal) (increase (total-cost) 1)))))",
                        R"((define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal))
 (:metric minimize (total-cost))))");
    for (auto search : {search::astar_eager, search::astar_lazy})
    {
        Recorder wr(*w);
        search::BestFirstOptions wo;
        wo.evaluator = &wr;
        const auto res = search(*w, wo);
        ASSERT_EQ(res.status, search::SearchStatus::Solved) << res.message;
        EXPECT_EQ(res.cost, 3);
        EXPECT_EQ(status_of(wr, "(q)"), 1u) << "the paths through expensive and through cheap1, cheap2 share bit 0 only";
    }
}

TEST(LandmarkHeuristicsPddl, Refusals)
{
    const auto t = make(k_chain_domain, k_chain_problem);
    // an AnyOf goal: landmark count estimates the task's goal
    search::BestFirstOptions o;
    o.heuristic.kind = H::Kind::LmCount;
    o.control.goal.kind = search::GoalSpec::Kind::AnyOf;
    search::GoalSpec::AtomGoal g;
    for (u32 i = 0; i < t->atoms().fluent_slots(); ++i)
        if (t->format(SlotId{i}) == "(h)")
            g.positive.push_back(SlotId{i});
    ASSERT_EQ(g.positive.size(), 1u);
    o.control.goal.goals = {g};
    const auto r = search::gbfs_eager(*t, o);
    EXPECT_EQ(r.status, search::SearchStatus::Failed);
    EXPECT_NE(r.message.find("GoalSpec::AnyOf"), std::string::npos) << r.message;
    EXPECT_THROW((void)heuristic(*t, H::Kind::LmCount)->evaluate(t->initial_state().view(), o.control.goal.goals),
                 std::invalid_argument);
    // LM-cut estimates the nearest of several goals
    o.heuristic.kind = H::Kind::LmCut;
    const auto rc = search::astar_eager(*t, o);
    ASSERT_EQ(rc.status, search::SearchStatus::Solved) << rc.message;
    EXPECT_EQ(rc.cost, 1);
    EXPECT_EQ(heuristic(*t, H::Kind::LmCut)->evaluate(t->initial_state().view(), o.control.goal.goals), 1);

    // a path-dependent heuristic cannot evaluate in batches
    Recorder rec(*t);
    rec.batch = true;
    search::BestFirstOptions b;
    b.evaluator = &rec;
    const auto rb = search::gbfs_eager(*t, b);
    EXPECT_EQ(rb.status, search::SearchStatus::Failed);
    EXPECT_NE(rb.message.find("batch"), std::string::npos) << rb.message;
}
