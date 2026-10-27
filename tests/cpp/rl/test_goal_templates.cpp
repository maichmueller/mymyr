// Lifted goal templates (rl/goal_templates.hpp), on the text exports of tests/data/tasks:
//   - the subsets of a component are the first connected subsets of each size in lexicographic order, as a brute-force
//     enumeration over all index tuples finds them;
//   - generated templates: sizes bounded and largest first, at most the quota per size, each the lift of its goal
//     literals (with or without disequalities), the binding order of its condition, the packed arrays consistent;
//   - an instance's templates are those of its task at its global id, whatever the other instances;
//   - from_conditions and from_arrays rebuild the same templates; invalid options and conditions throw.

#include "../support/suite.hpp"
#include "mymyr/rl/goal_templates.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/successor/bindings.hpp"
#include "mymyr/successor/conditions.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;
using namespace mymyr::test;

namespace
{
std::vector<std::string> task_names()
{
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(fs::path(MYMYR_TEST_DATA_DIR) / "tasks"))
        if (e.path().extension() == ".txt")
            out.push_back(e.path().stem().string());
    std::ranges::sort(out);
    return out;
}

TaskPtr load(const std::string& name) { return Task::from_text_file(task_path(name)); }

/// Per goal literal, its connected component (literals joined when their atoms share an object), numbered by the
/// smallest literal index.
std::vector<u32> components(const GroundCondition& goal)
{
    const usize n = goal.literals.size();
    std::vector<u32> parent(n);
    for (u32 i = 0; i < n; ++i)
        parent[i] = i;
    const auto find = [&](u32 i)
    {
        while (parent[i] != i)
            i = parent[i];
        return i;
    };
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < i; ++j)
            for (ObjectId o : goal.literals[i].atom.objects)
                if (std::ranges::find(goal.literals[j].atom.objects, o) != goal.literals[j].atom.objects.end())
                {
                    const u32 a = find(i), b = find(j);
                    parent[std::max(a, b)] = std::min(a, b);
                }
    std::vector<u32> out(n);
    for (u32 i = 0; i < n; ++i)
        out[i] = find(i);
    return out;
}

bool shares_object(const GroundLiteral& a, const GroundLiteral& b)
{
    for (ObjectId o : a.atom.objects)
        if (std::ranges::find(b.atom.objects, o) != b.atom.objects.end())
            return true;
    return false;
}

bool connected(const GroundCondition& goal, const std::vector<u32>& subset)
{
    std::vector<u8> seen(subset.size(), 0);
    std::vector<usize> stack{0};
    seen[0] = 1;
    usize count = 1;
    while (!stack.empty())
    {
        const usize v = stack.back();
        stack.pop_back();
        for (usize u = 0; u < subset.size(); ++u)
            if (!seen[u] && shares_object(goal.literals[subset[v]], goal.literals[subset[u]]))
            {
                seen[u] = 1;
                ++count;
                stack.push_back(u);
            }
    }
    return count == subset.size();
}

/// The first `limit` connected k-subsets of `vertices` (ascending literal indices) in lexicographic order, by trying
/// every k-subset in that order.
std::vector<std::vector<u32>> brute_force_subsets(const GroundCondition& goal, const std::vector<u32>& vertices, u32 k, u32 limit)
{
    std::vector<std::vector<u32>> out;
    const u32 m = static_cast<u32>(vertices.size());
    std::vector<u32> pick(k);
    for (u32 i = 0; i < k; ++i)
        pick[i] = i;
    while (out.size() < limit)
    {
        std::vector<u32> subset(k);
        for (u32 i = 0; i < k; ++i)
            subset[i] = vertices[pick[i]];
        if (connected(goal, subset))
            out.push_back(subset);
        i64 i = static_cast<i64>(k) - 1;
        while (i >= 0 && pick[static_cast<usize>(i)] == m - k + static_cast<u32>(i))
            --i;
        if (i < 0)
            break;
        ++pick[static_cast<usize>(i)];
        for (usize j = static_cast<usize>(i) + 1; j < k; ++j)
            pick[j] = pick[j - 1] + 1;
    }
    return out;
}

GroundCondition sub_condition(const GroundCondition& goal, const std::vector<u32>& literals)
{
    GroundCondition g;
    for (u32 i : literals)
        g.literals.push_back(goal.literals[i]);
    return g;
}

template<class T>
std::vector<T> array_of(const rl::ArrayBundle& b, std::string_view name)
{
    const rl::ArrayInfo* a = b.find(name);
    EXPECT_NE(a, nullptr) << name;
    if (!a)
        return {};
    std::vector<T> out(a->elements());
    std::memcpy(out.data(), b.data(*a), out.size() * sizeof(T));
    return out;
}

rl::GoalTemplates::Arrays arrays_of(const rl::GoalTemplates& t)
{
    const rl::ArrayBundle& b = t.arrays();
    rl::GoalTemplates::Arrays a;
    a.instance_offsets = array_of<i32>(b, "instance_offsets");
    a.template_size = array_of<i32>(b, "template_size");
    a.num_vars = array_of<i32>(b, "num_vars");
    a.distinct = array_of<u8>(b, "distinct");
    a.literal_offsets = array_of<i32>(b, "literal_offsets");
    a.predicate = array_of<i32>(b, "predicate");
    a.polarity = array_of<u8>(b, "polarity");
    a.kind = array_of<i32>(b, "kind");
    a.arity = array_of<i32>(b, "arity");
    a.terms = array_of<i32>(b, "terms");
    a.goal_literal = array_of<i32>(b, "goal_literal");
    a.max_arity = t.max_arity();
    return a;
}
}  // namespace

TEST(GoalTemplates, ComponentSubsetsAreTheFirstConnectedSubsetsInLexicographicOrder)
{
    for (const std::string& name : task_names())
    {
        const TaskPtr task = load(name);
        const GroundCondition goal = GroundCondition::goal(*task);
        const std::vector<u32> comp = components(goal);
        for (u32 limit : {3u, 100u})
        {
            rl::TemplateOptions o;
            o.max_size = 5;
            o.max_subsets_per_size = limit;
            o.max_subgoals_per_size = 1u << 30;  // every subgoal
            o.max_component_samples = 64;        // as many single-component draws as there are components
            o.max_products = 20000;              // more than a component's subsets
            // the subgoals within one component, by (component, size), in order: the subset lists of the components
            // the uniform single-component draws hit (duplicate draws are dropped, so not necessarily all)
            std::map<std::pair<u32, u32>, std::vector<std::vector<u32>>> got;
            for (const std::vector<u32>& g : rl::goal_subgoals(*task, 0, o))
                if (std::ranges::all_of(g, [&](u32 i) { return comp[i] == comp[g[0]]; }))
                    got[{comp[g[0]], static_cast<u32>(g.size())}].push_back(g);
            std::map<u32, std::vector<u32>> vertices;
            for (u32 i = 0; i < comp.size(); ++i)
                vertices[comp[i]].push_back(i);
            u32 drawn = 0;
            for (const auto& [c, vs] : vertices)
            {
                if (!got.contains(std::pair{c, 1u}))
                    continue;
                ++drawn;
                for (u32 k = 1; k <= std::min<u32>(o.max_size, static_cast<u32>(vs.size())); ++k)
                {
                    const std::vector<std::vector<u32>>& listed = got[std::pair{c, k}];
                    EXPECT_EQ(listed, brute_force_subsets(goal, vs, k, limit))
                        << name << ": component " << c << ", size " << k << ", limit " << limit;
                }
            }
            EXPECT_GT(drawn, 0u) << name;
        }
    }
}

TEST(GoalTemplates, GeneratedTemplatesAreLiftedSubgoals)
{
    for (const std::string& name : task_names())
    {
        const TaskPtr task = load(name);
        const GroundCondition goal = GroundCondition::goal(*task);
        const rl::TaskSuitePtr suite = rl::TaskSuite::group({task});
        const WorkspaceLease ws = task->workspace();
        rl::TemplateOptions small;
        small.max_subgoals_per_size = 3;
        small.max_size = 4;
        small.distinct = rl::Distinct::None;
        small.seed = 9;
        for (const rl::TemplateOptions& o : {rl::TemplateOptions{}, small})
        {
            const rl::GoalTemplates t = rl::lifted_goal_templates(suite, o);
            ASSERT_EQ(t.num_instances(), 1u);
            ASSERT_NE(t.options(), nullptr);
            EXPECT_EQ(t.options()->seed, o.seed);
            const std::vector<std::vector<u32>> subgoals = rl::goal_subgoals(*task, 0, o);
            ASSERT_EQ(t.num_templates(), subgoals.size()) << name;
            EXPECT_FALSE(subgoals.empty()) << name;
            const std::vector<i32> order = array_of<i32>(t.arrays(), "order");
            const std::vector<u8> distinct = array_of<u8>(t.arrays(), "distinct");
            std::map<usize, u32> per_size;
            for (u32 k = 0; k < t.num_templates(); ++k)
            {
                const std::vector<u32>& g = subgoals[k];
                EXPECT_EQ(t.goal_literals(k), g);
                EXPECT_LE(g.size(), o.max_size);
                if (k > 0)
                {
                    EXPECT_LE(g.size(), subgoals[k - 1].size()) << name << ": sizes largest first";
                }
                EXPECT_LE(++per_size[g.size()], o.max_subgoals_per_size);
                EXPECT_EQ(std::set<u32>(g.begin(), g.end()).size(), g.size());
                const ConjunctiveCondition c = t.condition(k);
                EXPECT_EQ(c, sub_condition(goal, g).lift(o.distinct == rl::Distinct::All)) << name << ": template " << k;
                EXPECT_EQ(distinct[k], o.distinct == rl::Distinct::All ? 1 : 0);
                const std::vector<u32> want = binding_order(*task, *ws, c);
                for (u32 v = 0; v < t.max_vars(); ++v)
                    EXPECT_EQ(order[static_cast<usize>(k) * t.max_vars() + v], v < want.size() ? static_cast<i32>(want[v]) : -1);
            }
            EXPECT_EQ(t.conditions(0).size(), t.num_templates());
        }
    }
}

TEST(GoalTemplates, InstancesDependOnTheirTaskAndGlobalIdOnly)
{
    const std::vector<TaskPtr> tasks = {load("gripper__prob05"), load("logistics00__probLOGISTICS-6-1"),
                                        load("miconic-simpleadl__s10-2"), load("pegsol-08-strips__p22")};
    const rl::TaskSuitePtr suite = rl::TaskSuite::group(tasks);
    rl::TemplateOptions o;
    o.seed = 123;
    const rl::GoalTemplates t = rl::lifted_goal_templates(suite, o);
    const rl::GoalTemplates again = rl::lifted_goal_templates(suite, o);
    ASSERT_EQ(t.num_instances(), suite->size());
    for (u32 g = 0; g < suite->size(); ++g)
    {
        const std::vector<std::vector<u32>> want = rl::goal_subgoals(*suite->task(g), g, o);
        ASSERT_EQ(t.first(g + 1) - t.first(g), want.size());
        for (u32 k = t.first(g); k < t.first(g + 1); ++k)
        {
            EXPECT_EQ(t.goal_literals(k), want[k - t.first(g)]);
            EXPECT_EQ(again.condition(k), t.condition(k));
        }
        // the first instance alone in a suite (again at id 0) has the same templates
        if (g == 0)
        {
            EXPECT_EQ(rl::lifted_goal_templates(rl::TaskSuite::group({suite->task(0)}), o).conditions(0), t.conditions(0));
        }
    }
}

TEST(GoalTemplates, FromConditionsAndArraysRebuildTheTemplates)
{
    const std::vector<TaskPtr> tasks = {load("logistics00__probLOGISTICS-6-1"), load("depot__p02")};
    const rl::TaskSuitePtr suite = rl::TaskSuite::group(tasks);
    rl::TemplateOptions o;
    o.distinct = rl::Distinct::Equality;
    const rl::GoalTemplates t = rl::lifted_goal_templates(suite, o);
    std::vector<std::vector<ConjunctiveCondition>> lists;
    for (u32 g = 0; g < suite->size(); ++g)
        lists.push_back(t.conditions(g));
    const rl::GoalTemplates given = rl::GoalTemplates::from_conditions(suite, lists);
    EXPECT_EQ(given.options(), nullptr);
    const rl::GoalTemplates::Arrays a = arrays_of(t), b = arrays_of(given);
    EXPECT_EQ(a.instance_offsets, b.instance_offsets);
    EXPECT_EQ(a.terms, b.terms);
    EXPECT_EQ(a.predicate, b.predicate);
    EXPECT_EQ(a.distinct, b.distinct);
    EXPECT_EQ(array_of<i32>(t.arrays(), "order"), array_of<i32>(given.arrays(), "order"));
    EXPECT_TRUE(std::ranges::all_of(b.goal_literal, [](i32 x) { return x == -1; }));

    const rl::GoalTemplates restored = rl::GoalTemplates::from_arrays(suite, a, t.options());
    ASSERT_NE(restored.options(), nullptr);
    EXPECT_EQ(restored.options()->distinct, rl::Distinct::Equality);
    for (u32 g = 0; g < suite->size(); ++g)
        EXPECT_EQ(restored.conditions(g), t.conditions(g));
    for (u32 k = 0; k < t.num_templates(); ++k)
        EXPECT_EQ(restored.goal_literals(k), t.goal_literals(k));

    // a corrupt array is refused
    rl::GoalTemplates::Arrays bad = a;
    bad.terms[0] = 1000;
    EXPECT_THROW((void)rl::GoalTemplates::from_arrays(suite, bad), std::invalid_argument);
    bad = a;
    bad.instance_offsets.pop_back();
    EXPECT_THROW((void)rl::GoalTemplates::from_arrays(suite, bad), std::invalid_argument);
}

TEST(GoalTemplates, InvalidOptionsAndConditionsThrow)
{
    const TaskPtr task = load("gripper__prob05");
    const rl::TaskSuitePtr suite = rl::TaskSuite::group({task});
    for (u32 field = 0; field < 5; ++field)
    {
        rl::TemplateOptions o;
        u32* counts[] = {&o.max_subgoals_per_size, &o.max_size, &o.max_subsets_per_size, &o.max_component_samples,
                         &o.max_products};
        *counts[field] = 0;
        EXPECT_THROW((void)rl::lifted_goal_templates(suite, o), std::invalid_argument) << field;
    }
    const GroundCondition goal = GroundCondition::goal(*task);
    const ConjunctiveCondition lifted = sub_condition(goal, {0, 1}).lift(true);
    EXPECT_THROW((void)rl::GoalTemplates::from_conditions(suite, {}), std::invalid_argument);
    ConjunctiveCondition empty;
    EXPECT_THROW((void)rl::GoalTemplates::from_conditions(suite, {{empty}}), std::invalid_argument);
    ConjunctiveCondition partial = lifted;  // one disequality short of every pair
    partial.equalities.pop_back();
    EXPECT_THROW((void)rl::GoalTemplates::from_conditions(suite, {{partial}}), std::invalid_argument);
    ConjunctiveCondition equal = lifted;  // an equality, not a disequality
    equal.equalities[0].positive = true;
    EXPECT_THROW((void)rl::GoalTemplates::from_conditions(suite, {{equal}}), std::invalid_argument);
    ConjunctiveCondition constant = sub_condition(goal, {0, 1}).lift(false);
    constant.literals[0].terms[0] = formalism::object_term(goal.literals[0].atom.objects[0]);
    const rl::GoalTemplates t = rl::GoalTemplates::from_conditions(suite, {{constant, lifted}});
    EXPECT_EQ(t.conditions(0), (std::vector<ConjunctiveCondition>{constant, lifted}));
}
