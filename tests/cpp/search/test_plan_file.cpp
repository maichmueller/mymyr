// IPC plan text (search/plan_file.hpp): writing, reading back, costs and errors.

#include "mymyr/search/best_first.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
std::shared_ptr<const Task> load(const std::string& name, const char* dir = "tasks")
{
    return Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/" + dir + "/" + name + ".txt");
}

BestFirstResult solve(const Task& task)
{
    BestFirstOptions o;
    o.heuristic.kind = heuristics::Kind::Max;
    return astar_eager(task, o);
}
}  // namespace

TEST(PlanFile, WritesOneLinePerActionAndTheCost)
{
    const auto task = load("gripper__prob05");
    const BestFirstResult r = solve(*task);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    const std::string text = format_plan(*task, r.plan);
    EXPECT_EQ(std::ranges::count(text, '\n'), static_cast<long>(r.plan.size()) + 1);
    EXPECT_TRUE(text.starts_with("("));
    EXPECT_TRUE(text.ends_with("; cost = " + std::to_string(static_cast<int>(r.cost)) + " (unit cost)\n")) << text;
    EXPECT_EQ(parse_plan(*task, text), r.plan);
}

TEST(PlanFile, ReadsIgnoringCaseCommentsAndBlankLines)
{
    const auto task = load("gripper__prob05");
    const BestFirstResult r = solve(*task);
    std::string text = "; a plan\n\n" + format_plan(*task, r.plan) + "\n\n";
    std::ranges::transform(text, text.begin(), [](char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
    EXPECT_EQ(parse_plan(*task, text), r.plan);
    EXPECT_TRUE(parse_plan(*task, "; cost = 0 (unit cost)\n").empty());
}

TEST(PlanFile, GeneralCosts)
{
    // cs-counters: a metric over numeric values
    const auto task = load("cs-counters", "numeric_tasks");
    const BestFirstResult r = solve(*task);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    const std::string text = format_plan(*task, r.plan);
    EXPECT_EQ(parse_plan(*task, text), r.plan);
    EXPECT_NE(text.find("; cost = "), std::string::npos);
}

TEST(PlanFile, StartState)
{
    const auto task = load("gripper__prob05");
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s0 = task->initial_state();
    const std::vector<Action> first = succ.applicable_actions(s0.view());
    ASSERT_FALSE(first.empty());
    const State s1 = succ.apply(s0.view(), first.front().label());
    BestFirstOptions o;
    o.heuristic.kind = heuristics::Kind::Max;
    o.start = s1;
    const BestFirstResult r = astar_eager(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    const std::string text = format_plan(*task, r.plan, s1);
    EXPECT_EQ(parse_plan(*task, text, s1), r.plan);
}

TEST(PlanFile, Errors)
{
    const auto task = load("gripper__prob05");
    const BestFirstResult r = solve(*task);
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s0 = task->initial_state();
    const auto later = std::ranges::find_if(r.plan, [&](const Action& a) { return !succ.is_applicable(s0.view(), a.label()); });
    ASSERT_NE(later, r.plan.end());
    const std::vector<Action> alone{*later};  // not applicable in the initial state
    EXPECT_THROW((void)format_plan(*task, alone), std::invalid_argument);
    const std::string text = format_plan(*task, r.plan);
    std::string line = text;
    for (auto i = later - r.plan.begin(); i > 0; --i)
        line = line.substr(line.find('\n') + 1);
    line = "\n" + line.substr(0, line.find('\n') + 1);
    try
    {
        (void)parse_plan(*task, line);
        FAIL() << "no error";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
    }
    EXPECT_THROW((void)parse_plan(*task, "(no-such-action a b)\n"), std::invalid_argument);
    EXPECT_THROW((void)parse_plan(*task, "pick ball1\n"), std::invalid_argument);
}
