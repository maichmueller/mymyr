// AStarIW against the fork's generated goldens: status, cost, length, full plan, expansions and generations.
// The beam comparison's known successor-order differences apply here too: tuple ownership and state ids depend
// on generation order. Those instances remain in the golden data but are excluded from equality assertions.

#include "../support/json.hpp"
#include "../support/suite.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
namespace
{
bool action_order_differs(const std::string& name)
{
    for (const char* prefix : {"depot", "driverlog", "gripper", "logistics00", "zenotravel", "freecell", "folding", "organic-synthesis"})
        if (name.starts_with(prefix))
            return true;
    return false;
}
}  // namespace

TEST(AStarIwFork, StatusPlanAndCountsEqualTheFork)
{
    const auto doc = test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/astar_iw/fork_astar_iw.json");
    std::map<std::string, std::shared_ptr<const Task>> tasks;
    std::map<std::string, std::vector<std::string>> objects;
    u32 compared = 0, order = 0, unsupported = 0, budget = 0;
    for (const auto& run : doc["runs"].arr)
    {
        if (run.has("killed"))
        {
            ++unsupported;
            continue;
        }
        if (action_order_differs(run["task"].str))
        {
            ++order;
            continue;
        }
        const std::string status = run["status"].str;
        if (status != "solved" && status != "exhausted" && status != "unsolvable")
        {
            ++budget;
            continue;
        }
        const std::string& name = run["task"].str;
        if (!tasks.contains(name))
        {
            tasks[name] = Task::from_text_file(test::task_path(name));
            const auto names = test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/expected/" + name + ".json");
            for (const auto& o : names["names"]["objects"].arr)
                objects[name].push_back(o.str);
        }
        const Task& task = *tasks[name];
        SCOPED_TRACE(name + " " + run["h"].str + " " + run["features"].str + " " + std::to_string(run["width"].num));
        AStarIwOptions options;
        options.width = static_cast<u32>(run["width"].num);
        options.features = run["features"].str == "classical" ? AStarIwFeatures::Classical : AStarIwFeatures::Abstracted;
        options.heuristic.kind = run["h"].str == "blind" ? heuristics::Kind::Blind : heuristics::Kind::Max;
        const auto r = astar_iw(task, options);
        ASSERT_NE(r.status, SearchStatus::Failed) << r.message;
        EXPECT_EQ(to_string(r.status), status);
        EXPECT_EQ(r.stats.expanded, static_cast<u64>(run["expanded"].num));
        EXPECT_EQ(r.stats.generated, static_cast<u64>(run["generated"].num));
        if (status == "solved")
        {
            EXPECT_EQ(r.cost, run["plan_cost"].num);
            EXPECT_EQ(r.plan.size(), static_cast<usize>(run["plan_length"].num));
            std::vector<std::string> plan, expected;
            for (const Action& a : r.plan)
            {
                std::string text = "(" + std::string(task.schema_name(a.schema));
                for (ObjectId o : a.binding)
                    text += " " + objects[name].at(o.v);
                plan.push_back(text + ")");
            }
            for (const auto& a : run["plan"].arr)
                expected.push_back(a.str);
            EXPECT_EQ(plan, expected);
        }
        else
            EXPECT_TRUE(r.plan.empty());
        ++compared;
    }
    std::cout << "AStarIW fork compared=" << compared << " action_order=" << order << " unsupported=" << unsupported
              << " budget=" << budget << '\n';
    EXPECT_EQ(compared, 48u);
}
