// Best-first search on numeric tasks: blind and goal-count searches on the numeric suite, and states that differ only in
// their numeric values.

#include "../support/replay.hpp"

#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using test::replay;

namespace
{
std::string numeric_dir() { return std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks"; }
}  // namespace

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
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
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
