// Layer orderings against the fork (tests/data/layer_orders/fork_layer_orders.json, written by
// tests/data/fork_golden/search_fork/run_layer_orders.py): IW(2) and BrFS with stop_if_goal under the in-order,
// reverse and goal-count orderings (more and fewer satisfied goal literals first), with and without truncated next
// layers. Expectations: the status, the plan (action for action) and the counts of every IW pass and of BrFS equal
// the fork's.
//
// The fork enumerates the actions of a state in the order of its k-partite clique search, which branches on the
// parameter with the fewest candidates first; mymyr enumerates them in (schema, binding) order. Most suite tasks
// agree on every state, but on depot, driverlog, gripper, logistics00 and zenotravel some states differ. A
// reversed or truncated layer depends on that order, so on these tasks the untruncated runs compare the status and
// the plan length only, and the truncated runs are skipped.

#include "../support/json.hpp"
#include "../support/suite.hpp"

#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
LayerOrdering ordering(const test::json::Value& r)
{
    LayerOrdering lo;
    const std::string& o = r["order"].str;
    if (o == "in_order")
        lo.kind = LayerOrdering::Kind::InOrder;
    else if (o == "reverse")
        lo.kind = LayerOrdering::Kind::Reverse;
    else
    {
        lo.kind = LayerOrdering::Kind::GoalCount;
        lo.prefer_more_satisfied_goals = o == "goal_count";
    }
    if (!r["limit"].is_null())
        lo.max_next_layer_states = static_cast<u32>(r["limit"].num);
    return lo;
}

/// The plan in the fork's spelling: the text exports number the objects as the fork does, and
/// tests/data/expected/<task>.json lists their names in that order.
std::vector<std::string> plan_strings(const Task& task, const std::vector<std::string>& objects,
                                      const std::vector<Action>& plan)
{
    std::vector<std::string> out;
    for (const Action& a : plan)
    {
        std::string s = "(" + std::string(task.schema_name(a.schema));
        for (ObjectId o : a.binding)
            s += " " + objects.at(o.v);
        out.push_back(s + ")");
    }
    return out;
}

std::vector<std::string> fork_plan(const test::json::Value& r)
{
    std::vector<std::string> out;
    if (!r["plan"].is_null())
        for (const test::json::Value& a : r["plan"].arr)
            out.push_back(a.str);
    return out;
}

bool action_order_differs(const std::string& task)
{
    for (const char* prefix : {"depot", "driverlog", "gripper", "logistics00", "zenotravel"})
        if (task.starts_with(prefix))
            return true;
    return false;
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

TEST(LayerOrdersFork, IwAndBrfsEqualTheForks)
{
    const test::json::Value doc =
        test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/layer_orders/fork_layer_orders.json");
    std::map<std::string, std::shared_ptr<const Task>> tasks;
    std::map<std::string, std::vector<std::string>> objects;
    u32 compared = 0;
    for (const test::json::Value& r : doc["runs"].arr)
    {
        if (r.has("killed"))
            continue;
        const std::string& status = r["status"].str;
        if (status != "solved" && status != "failed" && status != "exhausted")
            continue;  // a budget stopped the fork
        if (sanitized() && r["expanded"].num > 20000)
            continue;
        const std::string& name = r["task"].str;
        const bool exact = !action_order_differs(name);
        if (!exact && !r["limit"].is_null())
            continue;
        if (!tasks.contains(name))
        {
            tasks[name] = Task::from_text_file(test::task_path(name));
            const test::json::Value golden =
                test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/expected/" + name + ".json");
            for (const test::json::Value& o : golden["names"]["objects"].arr)
                objects[name].push_back(o.str);
        }
        const Task& task = *tasks[name];
        const std::vector<std::string>& names = objects[name];
        const std::string what = name + " " + r["algo"].str + " " + r["order"].str + " limit " +
                                 (r["limit"].is_null() ? std::string("none") : std::to_string(static_cast<u32>(r["limit"].num)));
        if (r["algo"].str == "iw")
        {
            IwOptions o;
            o.max_arity = static_cast<u32>(r["k"].num);
            o.layers = ordering(r);
            const IwResult m = iw(task, o);
            EXPECT_STREQ(mimir_status_name(m.status), status.c_str()) << what;
            if (!exact)
            {
                EXPECT_EQ(m.plan.size(), fork_plan(r).size()) << what;
                ++compared;
                continue;
            }
            EXPECT_EQ(plan_strings(task, names, m.plan), fork_plan(r)) << what;
            const auto& passes = r["passes"].arr;
            ASSERT_EQ(m.passes.size(), passes.size()) << what;
            for (usize i = 0; i < passes.size(); ++i)
            {
                EXPECT_EQ(m.passes[i].expanded, static_cast<u64>(passes[i]["expanded"].num)) << what << " pass " << i;
                EXPECT_EQ(m.passes[i].generated, static_cast<u64>(passes[i]["generated"].num)) << what << " pass " << i;
                EXPECT_EQ(m.passes[i].generated_in_tree, static_cast<u64>(passes[i]["generated_in_tree"].num))
                    << what << " pass " << i;
            }
        }
        else
        {
            BrfsOptions o;
            o.witness_pruning = false;  // every applicable action is a transition (the fork's counts)
            o.stop_at_goal = true;
            o.layers = ordering(r);
            const BrfsResult m = brfs(task, o);
            EXPECT_EQ(m.solved, status == "solved") << what;
            EXPECT_EQ(m.exhausted, status == "exhausted") << what;
            if (!exact)
            {
                EXPECT_EQ(m.plan.size(), fork_plan(r).size()) << what;
                ++compared;
                continue;
            }
            EXPECT_EQ(plan_strings(task, names, m.plan), fork_plan(r)) << what;
            EXPECT_EQ(m.expanded, static_cast<u64>(r["expanded"].num)) << what;
            EXPECT_EQ(m.generated, static_cast<u64>(r["generated"].num)) << what;
        }
        ++compared;
    }
    EXPECT_GE(compared, sanitized() ? 40u : 100u);
}
