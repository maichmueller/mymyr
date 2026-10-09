// Numeric fluents and action costs from PDDL: the fork's applicability rules for numeric and total-cost effects,
// canonical values, metric kinds, and plan costs of tasks with numeric functions (transport-opt08, folding).

#include "golden.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>

using namespace mymyr;

namespace
{
const char* kDomain = R"(
(define (domain nd)
 (:requirements :strips :numeric-fluents :conditional-effects :action-costs)
 (:predicates (p) (q))
 (:functions (x) (y) (z) (u) (total-cost))
 (:action inc :parameters () :precondition (and) :effect (and (increase (x) 1)))
 (:action make-p :parameters () :precondition (and) :effect (and (p) (not (q))))
 (:action inc-undefined :parameters () :precondition (and) :effect (and (increase (u) 1)))
 (:action assign-undefined :parameters () :precondition (and) :effect (and (assign (u) 1)))
 (:action assign-and-increase :parameters () :precondition (and) :effect (and (increase (x) 1) (assign (x) 2)))
 (:action two-increases :parameters () :precondition (and) :effect (and (increase (x) 1) (increase (x) 2)))
 (:action increase-decrease :parameters () :precondition (and) :effect (and (increase (x) 1) (decrease (x) 3)))
 (:action two-scales :parameters () :precondition (and) :effect (and (scale-up (y) 2) (scale-up (y) 3)))
 (:action increase-scale :parameters () :precondition (and) :effect (and (increase (y) 1) (scale-up (y) 2)))
 (:action divide-by-zero :parameters () :precondition (and) :effect (and (assign (y) (/ (x) (z)))))
 (:action ce-not-firing :parameters () :precondition (and) :effect (and (increase (x) 1) (when (p) (assign (x) 5))))
 (:action ce-firing :parameters () :precondition (and) :effect (and (increase (x) 1) (when (q) (assign (x) 5))))
 (:action ce-families :parameters () :precondition (and) :effect (and (when (p) (assign (y) 5)) (when (q) (increase (y) 1))))
 (:action ce-families-conflict :parameters () :precondition (and) :effect (and (when (q) (assign (y) 5)) (when (p) (increase (y) 1))))
 (:action fluent-cost :parameters () :precondition (and) :effect (and (increase (y) 1) (increase (total-cost) (x))))
 (:action undefined-cost :parameters () :precondition (and) :effect (and (increase (y) 1) (increase (total-cost) (u))))
 (:action conditional-cost :parameters () :precondition (and)
   :effect (and (increase (y) 1) (increase (total-cost) 1) (when (q) (increase (total-cost) 10)) (when (p) (increase (total-cost) 100))))
 (:action numeric-pre :parameters () :precondition (and (>= (x) 1)) :effect (and (increase (y) 1)))
 (:action undefined-pre :parameters () :precondition (and (>= (u) 0)) :effect (and (increase (y) 1)))
 (:action negative-zero :parameters () :precondition (and) :effect (and (assign (y) (* -1 (z)))))
 (:action zero :parameters () :precondition (and) :effect (and (assign (y) 0)))
)
)";

const char* kProblem = R"(
(define (problem np) (:domain nd)
 (:init (q) (= (x) 1) (= (y) 2) (= (z) 0) (= (total-cost) 0))
 (:goal (and (>= (x) 3)))
 (:metric minimize (total-cost))
)
)";

std::shared_ptr<const Task> rules_task()
{
    const auto domain = frontend::Domain::from_string(kDomain, "nd.pddl");
    return Task::create(*domain->instantiate_string(kProblem, "np.pddl"));
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

TEST(NumericPddl, EffectApplicabilityFollowsTheFork)
{
    const auto task = rules_task();
    EXPECT_EQ(task->numeric_slots(), 2u);  // x, y; z is static, u has no value, total-cost is auxiliary
    EXPECT_EQ(task->numeric_storage(), NumericStorage::F64);  // a division
    Successors& succ = task->workspace().successors();
    const State s0 = task->initial_state();
    std::map<std::string, State> next;
    const heuristics::ActionCosts costs(*task);
    std::map<std::string, f64> cost;
    std::vector<Action> actions;
    succ.for_each_applicable(s0.view(),
                             [&](const ActionLabel& a, const Delta& d)
                             {
                                 actions.emplace_back(a);
                                 cost[task->schema_name(a.schema)] = costs.next(0, d);
                             });
    for (const Action& a : actions)  // after the enumeration: Successors is not reentrant
        next.emplace(task->schema_name(a.schema), succ.apply(s0.view(), a.label()));
    std::set<std::string> applicable;
    for (const auto& [name, _] : next)
        applicable.insert(name);
    const std::set<std::string> want = {"inc",           "make-p",        "two-increases",        "increase-decrease",
                                        "two-scales",    "ce-not-firing", "ce-families",          "ce-families-conflict",
                                        "fluent-cost",   "conditional-cost", "numeric-pre",       "negative-zero",
                                        "zero"};
    // not applicable: an effect on a function without a value (increase and, unlike the fork, assign), an assignment
    // with another effect on the same target, additive with multiplicative effects, a NaN value (division by zero),
    // a total-cost over an undefined value, a constraint over one. Unlike the fork, a conditional effect that does
    // not fire conflicts with nothing (ce-families-conflict).
    EXPECT_EQ(applicable, want);
    EXPECT_EQ(value(*task, next.at("inc").view(), "(x)"), 2);
    EXPECT_EQ(value(*task, next.at("two-increases").view(), "(x)"), 4);
    EXPECT_EQ(value(*task, next.at("increase-decrease").view(), "(x)"), -1);
    // loki merges the effects of one operator on one target by adding their expressions, for scale-up too (as in the
    // fork): (scale-up (y) 2) (scale-up (y) 3) becomes (scale-up (y) (+ 2 3))
    EXPECT_EQ(value(*task, next.at("two-scales").view(), "(y)"), 10);
    EXPECT_EQ(value(*task, next.at("ce-not-firing").view(), "(x)"), 2);
    // -0 is stored as +0: the two successors are one state
    EXPECT_EQ(next.at("negative-zero"), next.at("zero"));
    EXPECT_EQ(next.at("negative-zero").hash(), next.at("zero").hash());
    // costs: the total-cost effects that fire, over fluent values of the parent
    EXPECT_EQ(costs.kind(), heuristics::ActionCosts::Kind::TotalCost);
    EXPECT_FALSE(costs.state_independent());
    EXPECT_EQ(cost.at("fluent-cost"), 1);        // x = 1
    EXPECT_EQ(cost.at("conditional-cost"), 11);  // q holds, p does not
    EXPECT_EQ(cost.at("inc"), 0);                // no total-cost effect
}

TEST(NumericPddl, OnlyConditionalEffectsThatFireRecordTheirFamilies)
{
    // The fork records the effect families of every conditional effect in turn, also when its condition is false, so
    // that the order loki gives the conditional effects decides applicability: (when (p) (increase (y) 1)) with p
    // false, then (when (q) (assign (y) 5)), which fires, made ce-families-conflict inapplicable there. An effect
    // that does not fire does not execute: both actions are applicable, and an assignment and an increase of one
    // target that both fire are not (in either order).
    const auto task = rules_task();
    Successors& succ = task->workspace().successors();
    std::set<std::string> seen;
    succ.for_each_applicable(task->initial_state().view(),
                             [&](const ActionLabel& a, const Delta&) { seen.insert(task->schema_name(a.schema)); });
    EXPECT_TRUE(seen.contains("ce-families"));
    EXPECT_TRUE(seen.contains("ce-families-conflict"));

    const char* domain = R"((define (domain ce) (:requirements :strips :numeric-fluents :conditional-effects)
 (:predicates (a) (z) (done)) (:functions (x))
 (:action finish :parameters () :precondition (and)
  :effect (and (when (a) (assign (x) 7)) (when (z) (increase (x) 1)) (done)))
 (:action finish2 :parameters () :precondition (and)
  :effect (and (when (z) (increase (x) 1)) (when (a) (assign (x) 7)) (done)))))";
    const auto d = frontend::Domain::from_string(domain, "ce.pddl");
    for (const char* init : {"", "(a)", "(z)", "(a) (z)"})
    {
        const std::string problem = std::string("(define (problem p) (:domain ce) (:init ") + init + " (= (x) 2)) (:goal (done)))";
        const auto t = Task::create(*d->instantiate_string(problem, "p.pddl"));
        const bool a = std::string(init).find("(a)") != std::string::npos, z = std::string(init).find("(z)") != std::string::npos;
        std::map<std::string, State> next;
        std::vector<Action> actions;
        Successors& s = t->workspace().successors();
        s.for_each_applicable(t->initial_state().view(), [&](const ActionLabel& l, const Delta&) { actions.emplace_back(l); });
        for (const Action& x : actions)
            next.emplace(t->schema_name(x.schema), s.apply(t->initial_state().view(), x.label()));
        SCOPED_TRACE(init);
        EXPECT_EQ(next.size(), a && z ? 0u : 2u);
        for (const auto& [name, state] : next)
            EXPECT_EQ(value(*t, state.view(), "(x)"), a ? 7 : z ? 3 : 2) << name;
    }
}

TEST(NumericPddl, MetricWithoutTotalCostIsTheStateMetric)
{
    const char* problem = R"((define (problem np2) (:domain nd)
 (:init (= (x) 1) (= (y) 2) (= (z) 0))
 (:goal (and (>= (x) 3)))
 (:metric minimize (+ (x) (* 2 (y))))))";
    const char* domain = R"((define (domain nd)
 (:requirements :strips :numeric-fluents)
 (:functions (x) (y) (z))
 (:action inc :parameters () :precondition (and) :effect (and (increase (x) 1)))
 (:action dbl :parameters () :precondition (and) :effect (and (scale-up (y) 2)))))";
    const auto d = frontend::Domain::from_string(domain, "nd.pddl");
    const auto task = Task::create(*d->instantiate_string(problem, "np2.pddl"));
    const heuristics::ActionCosts costs(*task);
    EXPECT_EQ(costs.kind(), heuristics::ActionCosts::Kind::StateMetric);
    const State s0 = task->initial_state();
    EXPECT_EQ(costs.initial(s0.view()), 5);
    Successors& succ = task->workspace().successors();
    std::map<std::string, f64> g;
    succ.for_each_applicable(s0.view(), [&](const ActionLabel& a, const Delta& delta)
                             { g[task->schema_name(a.schema)] = costs.next(5, delta); });
    EXPECT_EQ(g.at("inc"), 6);
    EXPECT_EQ(g.at("dbl"), 9);
    // BrFS: x grows without bound; depth-capped
    BrfsOptions o;
    o.max_depth = 3;
    o.layer_stats = true;
    const BrfsResult r = brfs(*task, o);
    const std::vector<std::array<u64, 3>> want = {{1, 2, 2}, {2, 4, 3}, {3, 6, 4}};
    EXPECT_EQ(r.layer_counts, want);
    EXPECT_EQ(r.goal_states, 1u);  // x = 3 at depth 2 (y doubled 0 times); depth-3 states are not expanded
}

TEST(NumericPddl, TransportOpt08SiwPlanCostIsTheForks)
{
    const auto dir = test::work_dir() / "mimir-cs/Benchmark/strips/transport-opt08-strips";
    if (!std::filesystem::exists(dir / "p23.pddl"))
        GTEST_SKIP() << "no " << dir;
    const auto task = Task::create(*frontend::load_task(dir / "domain.pddl", dir / "p23.pddl"));
    search::SiwOptions o;
    o.max_arity = 2;
    const search::SiwResult r = search::siw(*task, o);
    ASSERT_EQ(r.status, search::SearchStatus::Solved);
    EXPECT_EQ(r.plan.size(), 21u);
    EXPECT_EQ(r.cost, 1102);  // the fork's SIW(2); 21 is the plan length, with cost_exact = false
    EXPECT_TRUE(r.cost_exact);
}

TEST(NumericPddl, TasksWithNumericFunctionsKeepTheirStateSpace)
{
    // rotate-cost / road-length only feed the metric: the BrFS counts are the text exports' (tests/cpp/support/suite.hpp)
    struct Case
    {
        const char* dir;
        const char* problem;
        u64 states, generated, goals;
    };
    for (const Case& c : {Case{"adl/folding-opt23-adl", "p01.pddl", 148260, 170776, 1}})
    {
        const auto dir = test::work_dir() / "mimir-cs/Benchmark" / c.dir;
        if (!std::filesystem::exists(dir / c.problem))
            GTEST_SKIP() << "no " << dir;
        const auto task = Task::create(*frontend::load_task(dir / "domain.pddl", dir / c.problem));
        EXPECT_EQ(task->numeric_slots(), 0u);
        const BrfsResult r = brfs(*task);
        EXPECT_EQ(r.states, c.states) << c.dir;
        EXPECT_EQ(r.generated, c.generated) << c.dir;
        EXPECT_EQ(r.goal_states, c.goals) << c.dir;
    }
}

namespace
{
/// The initial value of the numeric fluent (x) of a problem that writes `token` as its value, or nullopt if the
/// front end rejects the problem.
std::optional<f64> init_value(const std::string& token)
{
    static const char* domain_text = R"(
(define (domain nv)
 (:requirements :strips :numeric-fluents)
 (:predicates (p))
 (:functions (x))
 (:action inc :parameters () :precondition (and) :effect (and (increase (x) 1)))
))";
    const auto domain = frontend::Domain::from_string(domain_text, "nv.pddl");
    const std::string problem = "(define (problem nv1) (:domain nv) (:init (= (x) " + token + ")) (:goal (p)))";
    try
    {
        const auto task = Task::create(*domain->instantiate_string(problem, "nv1.pddl"));
        return value(*task, task->initial_state().view(), "(x)");
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}
}  // namespace

TEST(NumericPddl, InitialValuesAreParsedExactly)
{
    EXPECT_EQ(init_value("0"), 0.0);
    EXPECT_EQ(init_value("7"), 7.0);
    EXPECT_EQ(init_value("-3"), -3.0);
    EXPECT_EQ(init_value("2.5"), 2.5);
    EXPECT_EQ(init_value("-0.125"), -0.125);
    EXPECT_EQ(init_value("3."), 3.0);
    EXPECT_EQ(init_value(".5"), 0.5);
    EXPECT_EQ(init_value("1e3"), 1000.0);
    EXPECT_EQ(init_value("1.5e-2"), 0.015);
    EXPECT_EQ(init_value("-2.5E+1"), -25.0);
    EXPECT_EQ(init_value("0.1"), 0.1);  // correctly rounded
    EXPECT_EQ(init_value("123456789012345678"), 123456789012345678.0);
    EXPECT_EQ(init_value("4.9406564584124654e-324"), 4.9406564584124654e-324);  // the smallest subnormal
    for (const char* bad : {"", "abc", "1.2.3", "1e", "--1", "+1", "0x10", "1 2", "1e999", "5x", "-"})
        EXPECT_FALSE(init_value(bad).has_value()) << "'" << bad << "' was accepted";
}
