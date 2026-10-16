// Enforced hill-climbing, weighted A* and the anytime search on small PDDL tasks built for them: plateaus that need
// deeper lookaheads, helpful actions that lead into a dead end (the retry with every action, the fallback), the
// status of a path-dependent heuristic along the committed path, real and zero action costs, and a cost bound on a
// task whose costs can decrease.

#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/anytime.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/ehc.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
namespace H = mymyr::heuristics;

namespace
{
std::shared_ptr<const Task> make(const std::string& domain, const std::string& problem)
{
    const auto d = frontend::Domain::from_string(domain, "d.pddl");
    return Task::create(*d->instantiate_string(problem, "p.pddl"));
}

std::shared_ptr<const Task> load(const std::string& dir, const std::string& problem)
{
    const std::filesystem::path root = std::filesystem::path(MYMYR_SOURCE_DIR) / "tests/data/benchmarks" / dir;
    const auto d = frontend::Domain::from_file(root / "domain.pddl");
    return Task::create(*d->instantiate_file(root / problem));
}

/// The plan's states from the initial state (empty when an action is not applicable), the last one a goal state.
std::vector<State> replay(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    std::vector<State> states{task.initial_state()};
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(states.back(), a.label()))
            return {};
        states.push_back(succ.apply(states.back(), a.label()));
    }
    if (!succ.is_goal(states.back()))
        return {};
    return states;
}

std::unique_ptr<H::Heuristic> heuristic(const Task& task, H::Kind k)
{
    H::Options o;
    o.kind = k;
    return H::make_heuristic(task, o);
}

/// The metric value the plan reaches (the cost line of its IPC plan text).
double metric(const Task& task, const std::vector<Action>& plan)
{
    const std::string text = format_plan(task, plan);
    return std::stod(text.substr(text.rfind("; cost = ") + 9));
}

std::vector<std::string> names(const Task& task, const std::vector<Action>& plan)
{
    std::vector<std::string> out;
    for (const Action& a : plan)
        out.push_back(task.format(a.label()));
    return out;
}

const char* const k_corridor_domain = R"((define (domain corridor) (:requirements :strips)
 (:predicates (at ?c) (link ?a ?b))
 (:action move :parameters (?a ?b) :precondition (and (at ?a) (link ?a ?b)) :effect (and (at ?b) (not (at ?a))))))";

/// c0 - c1 - ... - c5, the goal at c5.
std::shared_ptr<const Task> corridor()
{
    return make(k_corridor_domain, R"((define (problem p) (:domain corridor) (:objects c0 c1 c2 c3 c4 c5)
 (:init (at c0) (link c0 c1) (link c1 c0) (link c1 c2) (link c2 c1) (link c2 c3) (link c3 c2) (link c3 c4) (link c4 c3)
  (link c4 c5) (link c5 c4))
 (:goal (at c5))))");
}

/// c0 to m through a or b, then g: two paths into m.
std::shared_ptr<const Task> diamond()
{
    return make(k_corridor_domain, R"((define (problem p) (:domain corridor) (:objects c0 a b m g)
 (:init (at c0) (link c0 a) (link a c0) (link c0 b) (link b c0) (link a m) (link m a) (link b m) (link m b) (link m g)
  (link g m))
 (:goal (at g))))");
}

/// The relaxed plan takes `spend`, which consumes the key the goal needs too: a dead end. `prepare` and `finish` (not
/// helpful in the start state) reach the goal.
std::shared_ptr<const Task> misleading()
{
    return make(R"((define (domain key) (:requirements :strips)
 (:predicates (key) (q) (r))
 (:action spend :parameters () :precondition (key) :effect (and (q) (not (key))))
 (:action prepare :parameters () :precondition (key) :effect (r))
 (:action finish :parameters () :precondition (r) :effect (q))))",
                R"((define (problem p) (:domain key) (:init (key)) (:goal (and (q) (key)))))");
}

/// A path-dependent heuristic whose status is the length of the path: it records the status of every evaluation and
/// counts merges, and takes its values from another heuristic.
class PathLength final : public H::Heuristic
{
public:
    explicit PathLength(std::unique_ptr<H::Heuristic> inner) : m_inner(std::move(inner)) {}
    [[nodiscard]] H::Kind kind() const noexcept override { return H::Kind::Custom; }
    H::Value evaluate(StateView s) override { return m_inner->evaluate(s); }
    H::Value evaluate(StateView s, std::span<const GoalSpec::AtomGoal> goals) override { return m_inner->evaluate(s, goals); }
    [[nodiscard]] bool path_dependent() const noexcept override { return true; }
    [[nodiscard]] u32 status_words() const noexcept override { return 1; }
    void initial_status(StateView, std::span<u64> out) override { out[0] = 0; }
    void progress(std::span<const u64> parent, StateView, const ActionLabel&, StateView, std::span<u64> out) override
    {
        out[0] = parent[0] + 1;
    }
    bool merge(std::span<u64> into, std::span<const u64> other) const override
    {
        ++merges;
        const bool changed = other[0] < into[0];
        into[0] = std::min(into[0], other[0]);
        return changed;
    }
    H::Value evaluate(StateView s, std::span<const u64> status) override
    {
        seen.emplace_back(State(s), status[0]);
        return m_inner->evaluate(s);
    }

    mutable u64 merges = 0;
    std::vector<std::pair<State, u64>> seen;

private:
    std::unique_ptr<H::Heuristic> m_inner;
};

/// Whether `h` evaluated every state of the plan but the first and the last with the length of the plan up to it.
void expect_plan_statuses(const PathLength& h, const std::vector<State>& states)
{
    for (usize k = 1; k + 1 < states.size(); ++k)
    {
        bool found = false;
        for (const auto& [s, status] : h.seen)
            found |= s == states[k] && status == k;
        EXPECT_TRUE(found) << "state " << k << " of the plan";
    }
}
}  // namespace

// ------------------------------------------------------------------------------------------------------ EHC

TEST(EhcPddl, APlateauNeedsADeeperLookahead)
{
    const auto task = corridor();
    // goal count: 1 everywhere but at the goal, one lookahead to depth 5
    EhcOptions o;
    o.heuristic.kind = H::Kind::GoalCount;
    const EhcResult r = ehc(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.plan.size(), 5u);
    EXPECT_EQ(r.ehc.lookaheads, 1u);
    EXPECT_EQ(r.ehc.improvements, 0u);
    EXPECT_EQ(r.ehc.max_depth, 5u);
    EXPECT_FALSE(replay(*task, r.plan).empty());
    // h_FF: the distance; every lookahead commits to the next cell (helpful actions: only the move ahead)
    o.heuristic.kind = H::Kind::FF;
    const EhcResult f = ehc(*task, o);
    ASSERT_EQ(f.status, SearchStatus::Solved) << f.message;
    EXPECT_EQ(f.plan.size(), 5u);
    EXPECT_EQ(f.ehc.lookaheads, 5u);
    EXPECT_EQ(f.ehc.improvements, 4u);
    EXPECT_EQ(f.ehc.max_depth, 1u);
    EXPECT_EQ(f.ehc.retries, 0u);
    EXPECT_EQ(f.stats.expanded, 5u);
    EXPECT_EQ(f.initial_h, 5);
}

TEST(EhcPddl, HelpfulActionsIntoADeadEndAreRetriedWithEveryAction)
{
    const auto task = misleading();
    // First (Auto for h_FF): the helpful lookahead only reaches the dead end; the retry finds prepare, finish
    EhcOptions o;
    const EhcResult r = ehc(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_EQ(names(*task, r.plan), (std::vector<std::string>{"(prepare)", "(finish)"}));
    EXPECT_EQ(r.ehc.retries, 1u);
    EXPECT_EQ(r.ehc.lookaheads, 2u);
    EXPECT_FALSE(r.ehc.failed);
    EXPECT_FALSE(r.ehc.fallback);
    EXPECT_GE(r.dead_ends, 1u);
    // Only: hill-climbing fails; the fallback (greedy best-first search) solves the task
    o.helpful_actions = EhcOptions::Helpful::Only;
    const EhcResult only = ehc(*task, o);
    ASSERT_EQ(only.status, SearchStatus::Solved) << only.message;
    EXPECT_TRUE(only.ehc.failed);
    EXPECT_TRUE(only.ehc.fallback);
    EXPECT_EQ(only.ehc.retries, 0u);
    EXPECT_FALSE(replay(*task, only.plan).empty());
    EXPECT_GT(only.stats.expanded, only.ehc.expanded);
    // without a fallback: Exhausted (hill-climbing failed)
    o.fallback = EhcOptions::Fallback::None;
    const EhcResult none = ehc(*task, o);
    EXPECT_EQ(none.status, SearchStatus::Exhausted);
    EXPECT_TRUE(none.ehc.failed);
    EXPECT_FALSE(none.ehc.fallback);
    EXPECT_TRUE(none.plan.empty());
    // Off: every action at once, the dead end is dropped
    o.helpful_actions = EhcOptions::Helpful::Off;
    const EhcResult off = ehc(*task, o);
    ASSERT_EQ(off.status, SearchStatus::Solved) << off.message;
    EXPECT_EQ(off.plan, r.plan);
    EXPECT_EQ(off.ehc.retries, 0u);
    EXPECT_EQ(off.ehc.lookaheads, 1u);
}

TEST(EhcPddl, ADeadEndStartIsUnsolvable)
{
    const auto task = make(R"((define (domain key) (:requirements :strips)
 (:predicates (key) (q))
 (:action spend :parameters () :precondition (key) :effect (and (q) (not (key))))))",
                           R"((define (problem p) (:domain key) (:init) (:goal (and (q) (key)))))");
    const EhcResult r = ehc(*task);
    EXPECT_EQ(r.status, SearchStatus::Unsolvable);
    EXPECT_EQ(r.stats.expanded, 0u);
}

TEST(EhcPddl, PathDependentStatusFollowsTheCommittedPath)
{
    // diamond: m is reached through a and through b in one lookahead; it keeps the status of the first path (no merge)
    {
        const auto task = diamond();
        PathLength h(heuristic(*task, H::Kind::GoalCount));
        EhcOptions o;
        o.evaluator = &h;
        const EhcResult r = ehc(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        const std::vector<State> states = replay(*task, r.plan);
        ASSERT_EQ(states.size(), 4u);
        expect_plan_statuses(h, states);
        EXPECT_EQ(h.merges, 0u);
        // the best-first searches merge the statuses of the two paths into m
        PathLength g(heuristic(*task, H::Kind::GoalCount));
        BestFirstOptions bo;
        bo.evaluator = &g;
        ASSERT_EQ(gbfs_eager(*task, bo).status, SearchStatus::Solved);
        EXPECT_GT(g.merges, 0u);
    }
    // corridor with h_FF: four commitments; the status of each committed state is the plan's length up to it
    {
        const auto task = corridor();
        PathLength h(heuristic(*task, H::Kind::FF));
        EhcOptions o;
        o.evaluator = &h;
        const EhcResult r = ehc(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        EXPECT_EQ(r.ehc.improvements, 4u);
        const std::vector<State> states = replay(*task, r.plan);
        ASSERT_EQ(states.size(), 6u);
        expect_plan_statuses(h, states);
        EXPECT_EQ(h.merges, 0u);
    }
    // the landmark count
    for (const auto helpful : {EhcOptions::Helpful::First, EhcOptions::Helpful::Off})
    {
        const auto task = load("small/transport", "test_problem.pddl");
        EhcOptions o;
        o.heuristic.kind = H::Kind::LmCount;
        o.helpful_actions = helpful;
        const EhcResult r = ehc(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        EXPECT_FALSE(replay(*task, r.plan).empty());
    }
}

TEST(EhcPddl, RealCostsAreTheMetricOfThePlan)
{
    for (const char* dir : {"small/transport", "small/woodworking"})
    {
        SCOPED_TRACE(dir);
        const auto task = load(dir, "test_problem.pddl");
        const EhcResult r = ehc(*task);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        ASSERT_FALSE(replay(*task, r.plan).empty());
        EXPECT_NEAR(r.cost, metric(*task, r.plan), 1e-9);
    }
}

// ------------------------------------------------------------------------------------------------------ weighted A*

TEST(WeightedAStarPddl, RealAndZeroCostsStayWithinTheBound)
{
    // zero and fractional costs: s1 (0), s2 (0.25), s3 (0.5) reach the goal for 0.75, direct for 1
    const auto zero = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (a) (b) (goal)) (:functions (total-cost))
 (:action direct :parameters () :precondition (and) :effect (and (goal) (increase (total-cost) 1)))
 (:action s1 :parameters () :precondition (and) :effect (and (a) (increase (total-cost) 0)))
 (:action s2 :parameters () :precondition (a) :effect (and (b) (increase (total-cost) 0.25)))
 (:action s3 :parameters () :precondition (b) :effect (and (goal) (increase (total-cost) 0.5)))))",
                           R"((define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal))
 (:metric minimize (total-cost))))");
    std::vector<std::shared_ptr<const Task>> tasks{zero, load("small/transport", "test_problem.pddl"),
                                                   load("small/woodworking", "test_problem.pddl")};
    for (const auto& task : tasks)
    {
        BestFirstOptions o;
        o.heuristic.kind = H::Kind::LmCut;
        const BestFirstResult opt = astar_eager(*task, o);
        ASSERT_EQ(opt.status, SearchStatus::Solved) << opt.message;
        for (const double w : {1.5, 2.0, 5.0})
            for (const auto k : {H::Kind::LmCut, H::Kind::Max})
            {
                SCOPED_TRACE(std::to_string(w) + " " + H::to_string(k));
                o.heuristic.kind = k;
                o.weight = w;
                const BestFirstResult r = astar_eager(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_LE(r.cost, w * opt.cost + 1e-9);
                EXPECT_GE(r.cost, opt.cost - 1e-9);
                ASSERT_FALSE(replay(*task, r.plan).empty());
                EXPECT_NEAR(r.cost, metric(*task, r.plan), 1e-9);
            }
    }
}

TEST(WeightedAStarPddl, AStarRefusesATransitionThatLowersTheMetric)
{
    const auto task = make(R"((define (domain d) (:requirements :strips :action-costs)
 (:predicates (a) (b) (goal)) (:functions (total-cost))
 (:action pay :parameters () :precondition (and) :effect (and (a) (increase (total-cost) 2)))
 (:action refund :parameters () :precondition (a) :effect (and (b) (not (a)) (decrease (total-cost) 1)))
 (:action finish :parameters () :precondition (b) :effect (and (goal) (increase (total-cost) 1)))))",
                           R"((define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal))
 (:metric minimize (total-cost))))");
    BestFirstOptions o;
    o.heuristic.kind = H::Kind::Blind;
    for (const double bound : {std::numeric_limits<double>::infinity(), 10.0})
    {
        o.cost_bound = bound;
        for (const auto fn : {&astar_eager, &astar_lazy})
        {
            const BestFirstResult r = fn(*task, o);
            EXPECT_EQ(r.status, SearchStatus::Failed);
            EXPECT_TRUE(r.plan.empty());
            EXPECT_NE(r.message.find("non-negative"), std::string::npos) << r.message;
            EXPECT_NE(r.message.find("from 2 to 1"), std::string::npos) << r.message;
        }
    }
    o.cost_bound = std::numeric_limits<double>::infinity();
    o.weight = 2.0;
    EXPECT_EQ(astar_eager(*task, o).status, SearchStatus::Failed);
    o.weight = 1.0;
    AnytimeOptions any;
    any.heuristic.kind = H::Kind::Blind;
    const AnytimeResult a = anytime_astar(*task, any);
    EXPECT_EQ(a.status, SearchStatus::Failed);
    EXPECT_TRUE(a.plan.empty());
    EXPECT_NE(a.message.find("non-negative"), std::string::npos) << a.message;
    // Greedy searches close each state once, so they terminate and report the plan with its metric value.
    const BestFirstResult g = gbfs_eager(*task, o);
    ASSERT_EQ(g.status, SearchStatus::Solved) << g.message;
    EXPECT_EQ(g.cost, 2);
    EhcOptions e;
    e.heuristic.kind = H::Kind::GoalCount;
    const EhcResult h = ehc(*task, e);
    ASSERT_EQ(h.status, SearchStatus::Solved) << h.message;
    EXPECT_EQ(h.cost, 2);
}

TEST(WeightedAStarPddl, AStarStopsOnANegativeCostCycle)
{
    // Without the check, reopening would lower g around the on/off cycle forever: every lap is cheaper than finishing.
    const auto task = make(R"((define (domain d) (:requirements :strips :negative-preconditions :action-costs)
 (:predicates (a) (goal)) (:functions (total-cost))
 (:action on :parameters () :precondition (not (a)) :effect (and (a) (decrease (total-cost) 1)))
 (:action off :parameters () :precondition (a) :effect (and (not (a)) (decrease (total-cost) 1)))
 (:action finish :parameters () :precondition (a) :effect (and (goal) (increase (total-cost) 100)))))",
                           R"((define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal))
 (:metric minimize (total-cost))))");
    BestFirstOptions o;
    o.heuristic.kind = H::Kind::Blind;
    for (const auto fn : {&astar_eager, &astar_lazy})
    {
        const BestFirstResult r = fn(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Failed);
        EXPECT_NE(r.message.find("non-negative"), std::string::npos) << r.message;
        EXPECT_LE(r.stats.expanded, 2u);
    }
    // A greedy search closes each state once.
    const BestFirstResult g = gbfs_eager(*task, o);
    ASSERT_EQ(g.status, SearchStatus::Solved) << g.message;
    EXPECT_EQ(g.cost, 99);
}

// ------------------------------------------------------------------------------------------------------ anytime

TEST(AnytimePddl, RealCostsReachTheOptimum)
{
    for (const char* dir : {"small/transport", "small/woodworking"})
    {
        SCOPED_TRACE(dir);
        const auto task = load(dir, "test_problem.pddl");
        BestFirstOptions a;
        a.heuristic.kind = H::Kind::LmCut;
        const BestFirstResult opt = astar_eager(*task, a);
        ASSERT_EQ(opt.status, SearchStatus::Solved) << opt.message;
        for (const auto k : {H::Kind::LmCut, H::Kind::FF})
        {
            AnytimeOptions o;
            o.heuristic.kind = k;
            const AnytimeResult r = anytime_astar(*task, o);
            ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
            EXPECT_TRUE(r.optimal);
            EXPECT_NEAR(r.cost, opt.cost, 1e-9);
            for (usize i = 1; i < r.solutions.size(); ++i)
                EXPECT_LT(r.solutions[i].cost, r.solutions[i - 1].cost);
            ASSERT_FALSE(replay(*task, r.plan).empty());
        }
    }
}
