// PDDL semantics against hand-evaluated values: union types, numeric expressions in every position, numeric values
// without an initial value, conditional numeric effects, metrics, and the heuristics' default costs.

#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
std::shared_ptr<const Task> make(const std::string& domain, const std::string& problem, const TaskOptions& options = {})
{
    const auto d = frontend::Domain::from_string(domain, "d.pddl");
    return Task::create(*d->instantiate_string(problem, "p.pddl"), options);
}

std::set<std::string> applicable(const Task& t, StateView s)
{
    std::set<std::string> out;
    const WorkspaceLease ws = t.workspace();
    for (const Action& a : ws->successors().applicable_actions(s))
        out.insert(t.format(a.label()));
    return out;
}

State apply(const Task& t, StateView s, const std::string& action)
{
    const WorkspaceLease ws = t.workspace();
    for (const Action& a : ws->successors().applicable_actions(s))
        if (t.format(a.label()) == action)
            return ws->successors().apply(s, a.label());
    ADD_FAILURE() << action << " is not applicable";
    return State{};
}

f64 value(const Task& t, StateView s, const std::string& name)
{
    for (u32 i = 0; i < t.numeric_slots(); ++i)
        if (t.numeric_name(i) == name)
            return t.numeric_value(s, i);
    ADD_FAILURE() << "no slot " << name;
    return 0;
}
}  // namespace

TEST(PddlSemantics, UnionTypesRangeOverEveryListedType)
{
    const auto t = make(R"((define (domain d) (:requirements :strips :typing :conditional-effects :universal-preconditions
                                :existential-preconditions)
 (:types sub - good good bad other)
 (:constants k - (either bad other))
 (:predicates (marked ?x - object) (done))
 (:action pick :parameters (?x - (either good bad)) :precondition (and) :effect (marked ?x))
 (:action pick-other :parameters (?x - (either other sub)) :precondition (and) :effect (marked ?x))
 (:action only-good :parameters (?x - good) :precondition (and) :effect (marked ?x))
 (:action only-other :parameters (?x - other) :precondition (and) :effect (marked ?x))
 (:action some :parameters () :precondition (exists (?y - (either bad other)) (marked ?y)) :effect (done))
 (:action all :parameters () :precondition (forall (?y - (either good bad)) (marked ?y)) :effect (done))
 (:action mark :parameters () :precondition (and) :effect (forall (?y - (either bad sub)) (marked ?y)))))",
                        R"((define (problem p) (:domain d)
 (:objects a - good s - sub b - bad o - other e - (either good other))
 (:init (marked o)) (:goal (done))))");
    const State s0 = t->initial_state();
    const std::set<std::string> want = {"(pick a)",       "(pick b)",       "(pick e)",       "(pick k)",
                                        "(pick s)",       "(pick-other e)", "(pick-other k)", "(pick-other o)",
                                        "(pick-other s)", "(only-good a)",  "(only-good e)",  "(only-good s)",
                                        "(only-other e)", "(only-other k)", "(only-other o)", "(some o)",
                                        "(mark)"};
    EXPECT_EQ(applicable(*t, s0.view()), want);
    // the forall effect marks b, k and s; then every good or bad object is marked once a and e are
    State s = apply(*t, s0.view(), "(mark)");
    const auto atoms = t->format_atoms(s.view());
    for (const char* x : {"(marked b)", "(marked k)", "(marked s)", "(marked o)"})
        EXPECT_NE(std::ranges::find(atoms, std::string(x)), atoms.end()) << x;
    EXPECT_EQ(atoms.size(), 4u);
    EXPECT_FALSE(applicable(*t, s.view()).contains("(all)"));
    s = apply(*t, s.view(), "(pick a)");
    s = apply(*t, s.view(), "(pick e)");
    EXPECT_TRUE(applicable(*t, s.view()).contains("(all)"));
}

TEST(PddlSemantics, NumericExpressionsInEveryPosition)
{
    // x = 2 and y = 0.5 are fluents, z = 3 a static function, (v o) = 1.5 a fluent and (w o) = 4 a static function
    struct Case
    {
        const char* expr;
        f64 value;  // NaN: undefined
    };
    const f64 undefined = std::nan("");
    const std::vector<Case> cases = {
        {"2.5", 2.5},
        {"(x)", 2},
        {"(z)", 3},
        {"(- (x))", -2},
        {"(- (z))", -3},
        {"(+ (x) (y))", 2.5},
        {"(- (x) (y))", 1.5},
        {"(* (x) (y))", 1},
        {"(/ (x) (y))", 4},
        {"(- (+ (x) (z)))", -5},
        {"(- (* (x) (z)) (y))", 5.5},
        {"(/ (z) (- (x) 2))", undefined},
        {"(/ (z) (- (z) 3))", undefined},
        {"(+ (v ?p) (w ?p))", 5.5},
        {"(- (v ?p))", -1.5},
        {"(/ (w ?p) (x))", 2},
        {"(* (- (w ?p)) (y))", -2},
        {"(/ (v ?p) (- (w ?p) 4))", undefined},
    };
    auto num = [](f64 v) { return v < 0 ? "(- " + std::to_string(-v) + ")" : std::to_string(v); };
    // e equals v: (>= e v) and (<= e v); an undefined value satisfies neither (>= e -100) nor (< e -100)
    auto equals = [&](const std::string& e, f64 v)
    {
        return std::isnan(v) ? "(or (>= " + e + " (- 100)) (< " + e + " (- 100)))"
                             : "(and (>= " + e + " " + num(v) + ") (<= " + e + " " + num(v) + "))";
    };
    std::string actions, axioms, preds;
    for (usize i = 0; i < cases.size(); ++i)
    {
        const std::string n = std::to_string(i), e = cases[i].expr;
        const std::string eq = equals(e, cases[i].value);
        preds += " (ok" + n + " ?p - obj) (hit" + n + ")";
        axioms += " (:derived (ok" + n + " ?p - obj) " + eq + ")";
        actions += " (:action pre" + n + " :parameters (?p - obj) :precondition " + eq + " :effect (done))";
        actions += " (:action ax" + n + " :parameters (?p - obj) :precondition (ok" + n + " ?p) :effect (done))";
        actions += " (:action ce" + n + " :parameters (?p - obj) :precondition (and) :effect (when " + eq + " (hit" + n + ")))";
        actions += " (:action eff" + n + " :parameters (?p - obj) :precondition (and) :effect (assign (r) " + e + "))";
    }
    const std::string domain = "(define (domain d) (:requirements :strips :typing :numeric-fluents :conditional-effects "
                               ":derived-predicates :disjunctive-preconditions) (:types obj) (:constants o - obj) "
                               "(:predicates (done)" +
                               preds + ") (:functions (x) (y) (z) (r) (v ?p - obj) (w ?p - obj))" + axioms + actions +
                               " (:action touch :parameters () :precondition (and) :effect (and (increase (x) 0) "
                               "(increase (y) 0) (increase (v o) 0))))";
    const std::string init = "(:init (= (x) 2) (= (y) 0.5) (= (z) 3) (= (r) 0) (= (v o) 1.5) (= (w o) 4))";
    const auto t = make(domain, "(define (problem p) (:domain d) " + init + " (:goal (done)))");
    const State s0 = t->initial_state();
    const std::set<std::string> app = applicable(*t, s0.view());
    for (usize i = 0; i < cases.size(); ++i)
    {
        const std::string n = std::to_string(i);
        const bool defined = !std::isnan(cases[i].value);
        SCOPED_TRACE(cases[i].expr);
        EXPECT_EQ(app.contains("(pre" + n + " o)"), defined);  // precondition
        EXPECT_EQ(app.contains("(ax" + n + " o)"), defined);   // axiom body
        EXPECT_EQ(app.contains("(eff" + n + " o)"), defined);  // effect: undefined values make the action inapplicable
        if (defined)
        {
            EXPECT_EQ(value(*t, apply(*t, s0.view(), "(eff" + n + " o)").view(), "(r)"), cases[i].value);
        }
        const State c = apply(*t, s0.view(), "(ce" + n + " o)");  // effect condition
        const auto atoms = t->format_atoms(c.view());
        EXPECT_EQ(std::ranges::find(atoms, "(hit" + n + ")") != atoms.end(), defined);
        // goal (the parameter fixed to o)
        std::string goal = cases[i].expr;
        for (usize at; (at = goal.find("?p")) != std::string::npos;)
            goal.replace(at, 2, "o");
        const std::string eq = defined ? equals(goal, cases[i].value) : "(>= " + goal + " (- 100))";
        const auto g = make(domain, "(define (problem p) (:domain d) " + init + " (:goal " + eq + "))");
        EXPECT_EQ(g->is_goal(g->initial_state().view()), defined);
    }
}

TEST(PddlSemantics, UnaryMinusInANullaryConstraint)
{
    // (< (- (x)) 0) with x = 2 holds: -2 < 0
    const auto t = make(R"((define (domain d) (:requirements :strips :numeric-fluents) (:predicates (done)) (:functions (x))
 (:action finish :parameters () :precondition (< (- (x)) 0) :effect (and (done) (increase (x) 1)))))",
                        "(define (problem p) (:domain d) (:init (= (x) 2)) (:goal (and (done) (< (- (x)) (- 2)))))");
    EXPECT_EQ(applicable(*t, t->initial_state().view()), std::set<std::string>{"(finish)"});
    EXPECT_FALSE(t->is_goal(t->initial_state().view()));
    EXPECT_TRUE(t->is_goal(apply(*t, t->initial_state().view(), "(finish)").view()));
}

TEST(PddlSemantics, AssignGivesAnUndefinedValueAValue)
{
    const std::string domain = R"((define (domain d) (:requirements :strips :typing :numeric-fluents)
 (:types truck place) (:predicates (moved ?t - truck))
 (:functions (fuel ?t - truck) (cap ?t - truck))
 (:action refuel :parameters (?t - truck) :precondition (and) :effect (assign (fuel ?t) (cap ?t)))
 (:action pump :parameters (?t - truck) :precondition (and) :effect (increase (fuel ?t) 1))
 (:action drive :parameters (?t - truck) :precondition (>= (fuel ?t) 1) :effect (and (moved ?t) (decrease (fuel ?t) 1)))))";
    const std::string problem = R"((define (problem p) (:domain d) (:objects t1 t2 - truck p1 - place)
 (:init (= (fuel t1) 2) (= (cap t1) 3) (= (cap t2) 5)) (:goal (and (moved t2) (>= (fuel t2) 0)))))";
    const auto t = make(domain, problem);
    // (fuel t2) can be assigned: it has a slot, undefined at first; a place has no fuel
    ASSERT_EQ(t->numeric_slots(), 2u);
    EXPECT_EQ(t->numeric_name(0), "(fuel t1)");
    EXPECT_EQ(t->numeric_name(1), "(fuel t2)");
    EXPECT_EQ(t->numeric_storage(), NumericStorage::F64);  // I32 cannot hold an undefined value
    const State s0 = t->initial_state();
    EXPECT_TRUE(std::isnan(value(*t, s0.view(), "(fuel t2)")));
    // reading or increasing an undefined value: inapplicable; assigning it: applicable
    const std::set<std::string> want0 = {"(refuel t1)", "(refuel t2)", "(pump t1)", "(drive t1)"};
    EXPECT_EQ(applicable(*t, s0.view()), want0);
    const State s1 = apply(*t, s0.view(), "(refuel t2)");
    EXPECT_EQ(value(*t, s1.view(), "(fuel t2)"), 5);
    EXPECT_TRUE(applicable(*t, s1.view()).contains("(drive t2)"));
    EXPECT_TRUE(applicable(*t, s1.view()).contains("(pump t2)"));
    const auto r = search::astar_eager(*t);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.plan.size(), 2u);
    TaskOptions i32;
    i32.numeric_storage = TaskOptions::NumericStorageMode::I32;
    try
    {
        (void) make(domain, problem, i32);
        ADD_FAILURE() << "I32 storage of an undefined value";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("undefined"), std::string::npos) << e.what();
    }
}

TEST(PddlSemantics, OnlyEffectsThatFireConflict)
{
    // ce: (when (q) (assign (y) 5)) (when (p) (increase (y) 1)) conflicts only when p and q both hold; def-u defines u
    const auto t = make(R"((define (domain d) (:requirements :strips :negative-preconditions :conditional-effects
                                :numeric-fluents)
 (:predicates (p) (q) (done)) (:functions (y) (u))
 (:action set-p :parameters () :precondition (not (p)) :effect (p))
 (:action ce :parameters () :precondition (not (done)) :effect (and (done) (when (q) (assign (y) 5)) (when (p) (increase (y) 1))))
 (:action def-u :parameters () :precondition (and) :effect (assign (u) 1))))",
                        "(define (problem p) (:domain d) (:init (q) (= (y) 2)) (:goal (and (done) (>= (u) 0))))");
    const BrfsResult r = brfs(*t, {.stop_at_goal = false});
    EXPECT_EQ(r.states, 8u);
    EXPECT_EQ(r.generated, 14u);
    EXPECT_EQ(r.goal_states, 2u);
}

TEST(PddlSemantics, MetricsAreMinimizedOrRefused)
{
    const std::string domain = R"((define (domain d) (:requirements :strips :numeric-fluents :action-costs)
 (:predicates (done)) (:functions (x) (y) (total-cost))
 (:action cheap :parameters () :precondition (and) :effect (and (done) (increase (total-cost) 1) (increase (x) 10)))
 (:action dear :parameters () :precondition (and) :effect (and (done) (increase (total-cost) 10) (increase (x) 1)))
 (:action zero :parameters () :precondition (and) :effect (and (assign (y) 0) (increase (x) 0.5)))))";
    auto problem = [](const std::string& metric)
    { return "(define (problem p) (:domain d) (:init (= (x) 0) (= (y) 1) (= (total-cost) 0)) (:goal (done)) " + metric + ")"; };
    // a metric over the fluents is the state metric, also when the domain has total-cost effects
    const auto fluent = make(domain, problem("(:metric minimize (x))"));
    EXPECT_EQ(heuristics::ActionCosts(*fluent).kind(), heuristics::ActionCosts::Kind::StateMetric);
    auto r = search::astar_eager(*fluent);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.cost, 1);
    EXPECT_EQ(fluent->schema_name(r.plan.at(0).schema), "dear");
    // total-cost alone is the total-cost metric
    const auto total = make(domain, problem("(:metric minimize (total-cost))"));
    r = search::astar_eager(*total);
    EXPECT_EQ(r.cost, 1);
    EXPECT_EQ(total->schema_name(r.plan.at(0).schema), "cheap");
    // refused, naming the metric: maximize, and total-cost with other terms
    for (const auto& [metric, why] : {std::pair{"(:metric maximize (x))", "(:metric maximize (x)) is not supported"},
                                      std::pair{"(:metric minimize (+ (total-cost) (x)))", "total-cost can only be the whole metric"}})
    {
        const auto t = make(domain, problem(metric));
        EXPECT_THROW((void) heuristics::ActionCosts(*t), std::invalid_argument) << metric;
        r = search::astar_eager(*t);
        EXPECT_EQ(r.status, search::SearchStatus::Failed) << metric;
        EXPECT_NE(r.message.find(why), std::string::npos) << r.message;
    }
    // an undefined metric value: in the start state, and in a reached state (division by zero)
    const auto start = make(domain, "(define (problem p) (:domain d) (:init (= (x) 0)) (:goal (done)) (:metric minimize (y)))");
    r = search::astar_eager(*start);
    EXPECT_EQ(r.status, search::SearchStatus::Failed);
    EXPECT_NE(r.message.find("(:metric minimize (y)) is undefined in the start state"), std::string::npos) << r.message;
    const auto reached = make(domain, problem("(:metric minimize (/ (x) (y)))"));
    const heuristics::ActionCosts costs(*reached);
    const State s0 = reached->initial_state();
    EXPECT_EQ(costs.initial(s0.view()), 0);
    const WorkspaceLease ws = reached->workspace();
    ws->successors().for_each_applicable(
        s0.view(),
        [&](const ActionLabel& a, const Delta& d)
        {
            if (reached->schema_name(a.schema) == "zero")
            {
                EXPECT_THROW((void) costs.next(0, d), std::domain_error);
            }
        });
}

TEST(PddlSemantics, HeuristicCostsFollowTheObjective)
{
    // three actions of 0, 0.25 and 0.5 reach the goal for 0.75; one action reaches it for 1
    const auto t = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (a) (b) (goal)) (:functions (total-cost) (price))
 (:action direct :parameters () :precondition (and) :effect (and (goal) (increase (total-cost) 1)))
 (:action s1 :parameters () :precondition (and) :effect (and (a) (increase (total-cost) 0)))
 (:action s2 :parameters () :precondition (a) :effect (and (b) (increase (total-cost) 0.25)))
 (:action s3 :parameters () :precondition (b) :effect (and (goal) (increase (total-cost) (price))))))",
                        R"((define (problem p) (:domain d) (:init (= (total-cost) 0) (= (price) 0.5)) (:goal (goal))
 (:metric minimize (total-cost))))");
    const State s0 = t->initial_state();
    EXPECT_EQ(heuristics::resolve_costs(*t, heuristics::Costs::Auto), heuristics::Costs::Real);
    for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::H2})
        for (heuristics::Evaluation e : {heuristics::Evaluation::Grounded, heuristics::Evaluation::Lifted})
        {
            if (k == heuristics::Kind::H2 && e == heuristics::Evaluation::Lifted)
                continue;
            heuristics::Options o;
            o.kind = k;
            o.evaluation = e;
            EXPECT_EQ(heuristics::make_heuristic(*t, o)->evaluate(s0.view()), 0.75) << heuristics::to_string(k);
            o.costs = heuristics::Costs::Unit;
            EXPECT_EQ(heuristics::make_heuristic(*t, o)->evaluate(s0.view()), 1) << heuristics::to_string(k);
        }
    for (heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::Max, heuristics::Kind::H2})
    {
        search::BestFirstOptions o;
        o.heuristic.kind = k;
        const auto r = search::astar_eager(*t, o);
        ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
        EXPECT_EQ(r.cost, 0.75) << heuristics::to_string(k);
        EXPECT_EQ(r.plan.size(), 3u) << heuristics::to_string(k);
    }
    // a task without costs: unit costs
    const auto unit = make("(define (domain d) (:requirements :strips) (:predicates (a) (goal)) "
                           "(:action s1 :parameters () :precondition (and) :effect (a)) "
                           "(:action s2 :parameters () :precondition (a) :effect (goal)))",
                           "(define (problem p) (:domain d) (:init) (:goal (goal)))");
    EXPECT_EQ(heuristics::resolve_costs(*unit, heuristics::Costs::Auto), heuristics::Costs::Unit);
    heuristics::Options o;
    o.kind = heuristics::Kind::Max;
    EXPECT_EQ(heuristics::make_heuristic(*unit, o)->evaluate(unit->initial_state().view()), 2);
}

TEST(PddlSemantics, RelaxedCostsStayLowerBounds)
{
    // a cost with more than 6 decimal places is rounded down; an undefined cost leaves its action out
    const auto t = make(R"((define (domain d) (:requirements :strips :typing :action-costs)
 (:types item) (:predicates (got ?i - item) (goal))
 (:functions (total-cost) (price ?i - item))
 (:action get :parameters (?i - item) :precondition (and) :effect (and (got ?i) (increase (total-cost) (price ?i))))
 (:action finish :parameters (?i - item) :precondition (got ?i) :effect (and (goal) (increase (total-cost) 0.1234567)))))",
                        R"((define (problem p) (:domain d) (:objects cheap fine - item)
 (:init (= (total-cost) 0) (= (price fine) 0.5)) (:goal (goal)) (:metric minimize (total-cost))))");
    const heuristics::ActionCosts costs(*t);
    EXPECT_EQ(costs.relaxed_scale(), 1e6);
    EXPECT_FALSE(costs.relaxed_exact());
    heuristics::Options o;
    o.kind = heuristics::Kind::Max;
    const f64 h = heuristics::make_heuristic(*t, o)->evaluate(t->initial_state().view());
    EXPECT_EQ(h, 0.623456);  // (get fine) and (finish fine); (get cheap) has no price and is never applicable
    const auto r = search::astar_eager(*t);
    ASSERT_EQ(r.status, search::SearchStatus::Solved) << r.message;
    EXPECT_DOUBLE_EQ(r.cost, 0.6234567);
    EXPECT_LE(h, r.cost);
}

/// A goal literal over an atom no state holds (no effect adds it, the initial state lacks it): negative, it always
/// holds; positive, the goal is unreachable. The goal masks of the RL and device paths say the same.
TEST(PddlSemantics, GoalAtomsNoStateHolds)
{
    const std::string domain = R"((define (domain d) (:requirements :strips :negative-preconditions)
 (:predicates (f ?x) (g ?x))
 (:action set :parameters (?x) :precondition (g ?x) :effect (f ?x))))";
    auto problem = [](const std::string& goal)
    { return "(define (problem p) (:domain d) (:objects a b) (:init (g a)) (:goal " + goal + "))"; };
    const auto holds = make(domain, problem("(and (f a) (not (f b)))"));
    BrfsResult r = brfs(*holds, {.stop_at_goal = false});
    EXPECT_TRUE(r.exhausted);
    EXPECT_EQ(r.states, 2u);
    EXPECT_EQ(r.goal_states, 1u);
    rl::GoalMasks g = rl::goal_masks(*holds);
    EXPECT_FALSE(g.unsatisfiable);
    EXPECT_TRUE(std::ranges::all_of(g.neg, [](u64 w) { return w == 0; }));
    const auto never = make(domain, problem("(and (f a) (f b))"));
    r = brfs(*never, {.stop_at_goal = false});
    EXPECT_TRUE(r.exhausted);
    EXPECT_EQ(r.goal_states, 0u);
    g = rl::goal_masks(*never);
    EXPECT_TRUE(g.unsatisfiable);
}
