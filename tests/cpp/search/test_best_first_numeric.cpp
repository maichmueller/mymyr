// Best-first search on numeric tasks against the fork (tests/data/numeric_tasks/fork_best_first.json, written by
// tests/data/fork_golden/search_fork/run_numeric.py): A* eager and lazy with blind and h_max, GBFS eager and lazy
// with h_FF and eager with h_add, all with the fork's unit-cost heuristics, on the text exports of the same PDDL tasks.
//
// g is mimir's metric value (total-cost effects or the metric over the numeric values) and the relaxation heuristics
// ignore numeric conditions and effects, as the fork's delete relaxation does. Expectations: wherever the fork
// solves the task, mymyr solves it too; A* finds the fork's (optimal) plan cost; every plan replays to a goal state
// (numeric goal constraints included) at the reported cost. GBFS plans depend on the order of equally good states,
// and, for lazy GBFS, on h_FF's preferred operators (mymyr breaks ties among equally cheap supporters by its own
// operator order, heuristics/heuristic.hpp), so GBFS costs are reported but not required to be equal.

#include "../support/json.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
std::string numeric_dir() { return std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks"; }

struct ForkRun
{
    std::string task, algo, h, status;
    double cost = 0;
    u64 expanded = 0;
};

std::vector<ForkRun> fork_runs()
{
    const test::json::Value doc = test::json::parse_file(numeric_dir() + "/fork_best_first.json");
    std::vector<ForkRun> out;
    for (const test::json::Value& r : doc["runs"].arr)
    {
        if (r.has("killed"))
            continue;
        ForkRun f{r["task"].str, r["algo"].str, r["h"].str, r["status"].str};
        if (!r["plan_cost"].is_null())
            f.cost = r["plan_cost"].num;
        f.expanded = static_cast<u64>(r["expanded"].num);
        out.push_back(std::move(f));
    }
    return out;
}

BestFirstResult run(const Task& task, const ForkRun& f)
{
    BestFirstOptions o;
    o.heuristic.kind = heuristics::parse_kind(f.h);
    o.control.budget.max_states = 2'000'000;  // the fork's budget
    o.control.budget.max_seconds = 120;
    if (f.algo == "astar_eager")
        return astar_eager(task, o);
    if (f.algo == "astar_lazy")
        return astar_lazy(task, o);
    if (f.algo == "gbfs_eager")
        return gbfs_eager(task, o);
    return gbfs_lazy(task, o);
}

/// Replays the plan with numeric effects and total-cost effects: the metric value of the goal state it reaches, or
/// NaN if an action is not applicable or the last state is not a goal state.
double replay(const Task& task, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    const heuristics::ActionCosts costs(task);
    State s = task.initial_state();
    double g = costs.initial(s.view());
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s.view(), a.label()))
            return std::nan("");
        StateBuilder b;
        const Delta d = succ.apply_with_delta(s.view(), a.label(), b);
        g = costs.next(g, d);
        s = b.build();
    }
    return task.is_goal(s.view()) ? g : std::nan("");
}

bool sanitized()
{
#if defined(MYMYR_SANITIZED)
    return true;
#else
    return false;
#endif
}
}  // namespace

TEST(BestFirstNumeric, AStarCostsEqualTheForks)
{
    u32 compared = 0;
    for (const ForkRun& f : fork_runs())
    {
        if (f.algo.rfind("astar", 0) != 0 || f.status != "solved" || (sanitized() && f.expanded > 20000))
            continue;
        const auto task = Task::from_text_file(numeric_dir() + "/" + f.task + ".txt");
        const BestFirstResult r = run(*task, f);
        ASSERT_EQ(r.status, SearchStatus::Solved) << f.task << " " << f.algo << " " << f.h << ": " << r.message;
        EXPECT_EQ(r.cost, f.cost) << f.task << " " << f.algo << " " << f.h;
        EXPECT_EQ(replay(*task, r.plan), r.cost) << f.task << " " << f.algo << " " << f.h;
        ++compared;
    }
    EXPECT_GE(compared, sanitized() ? 20u : 60u);
}

TEST(BestFirstNumeric, GbfsSolvesWhatTheForkSolves)
{
    u32 compared = 0, equal = 0;
    for (const ForkRun& f : fork_runs())
    {
        if (f.algo.rfind("gbfs", 0) != 0 || f.status != "solved" || (sanitized() && f.expanded > 20000))
            continue;
        const auto task = Task::from_text_file(numeric_dir() + "/" + f.task + ".txt");
        const BestFirstResult r = run(*task, f);
        ASSERT_EQ(r.status, SearchStatus::Solved) << f.task << " " << f.algo << " " << f.h << ": " << r.message;
        EXPECT_EQ(replay(*task, r.plan), r.cost) << f.task << " " << f.algo << " " << f.h;
        ++compared;
        equal += r.cost == f.cost;
        if (r.cost != f.cost)
            std::printf("  %s %s %s: cost %g, the fork's %g\n", f.task.c_str(), f.algo.c_str(), f.h.c_str(), r.cost, f.cost);
    }
    std::printf("  GBFS: %u of %u plan costs equal the fork's\n", equal, compared);
    EXPECT_GE(compared, sanitized() ? 15u : 45u);
}

TEST(BestFirstNumeric, BlindAndGoalCountRunOnEveryNumericTask)
{
    for (const char* name : {"cs-counters", "cs-tpp", "m-transport", "cs-expedition", "m-woodworking", "m-zenotravel-numeric"})
    {
        const auto task = Task::from_text_file(numeric_dir() + "/" + name + ".txt");
        for (heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::GoalCount})
        {
            BestFirstOptions o;
            o.heuristic.kind = k;
            o.control.budget.max_states = 200'000;
            const BestFirstResult a = astar_eager(*task, o);
            const BestFirstResult g = gbfs_eager(*task, o);
            ASSERT_EQ(a.status, SearchStatus::Solved) << name << " " << heuristics::to_string(k) << ": " << a.message;
            ASSERT_EQ(g.status, SearchStatus::Solved) << name << " " << heuristics::to_string(k) << ": " << g.message;
            EXPECT_EQ(replay(*task, a.plan), a.cost) << name;
            EXPECT_EQ(replay(*task, g.plan), g.cost) << name;
            EXPECT_LE(a.cost, g.cost) << name;  // blind A* is optimal
        }
    }
}

TEST(BestFirstNumeric, StatesDifferingOnlyInTheirValuesAreDistinctInEveryStore)
{
    // cs-counters: increments change values only, so a store that ignored the numeric words would merge states
    const auto task = Task::from_text_file(numeric_dir() + "/cs-counters.txt");
    double cost = -1;
    for (BestFirstOptions::Store st : {BestFirstOptions::Store::Flat, BestFirstOptions::Store::Chunked, BestFirstOptions::Store::Compact})
    {
        BestFirstOptions o;
        o.heuristic.kind = heuristics::Kind::Blind;
        o.store = st;
        const BestFirstResult r = astar_eager(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.store << ": " << r.message;
        EXPECT_EQ(replay(*task, r.plan), r.cost) << r.store;
        if (cost < 0)
            cost = r.cost;
        EXPECT_EQ(r.cost, cost) << r.store;
        ASSERT_TRUE(r.goal_state.has_value());
        EXPECT_EQ(r.goal_state->numeric_words(), task->numeric_words());
    }
}

TEST(BestFirstNumeric, BlockedStatesCompareTheirValues)
{
    const auto task = Task::from_text_file(numeric_dir() + "/cs-counters.txt");
    Successors& succ = task->workspace().successors();
    const State s0 = task->initial_state();
    std::vector<State> next;
    for (const Action& a : succ.applicable_actions(s0.view()))
        next.push_back(succ.apply(s0.view(), a.label()));
    ASSERT_FALSE(next.empty());
    BestFirstOptions o;
    o.heuristic.kind = heuristics::Kind::Blind;
    o.control.blocked_states = next;  // every successor of the start state
    const BestFirstResult r = astar_eager(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Exhausted);
    EXPECT_EQ(r.stats.pruned, next.size());
}
