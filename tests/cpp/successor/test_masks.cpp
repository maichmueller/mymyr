#include "../support/suite.hpp"
#include "mymyr/rl/masks.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
struct MaskCase
{
    std::string name;
    TaskOptions::Matching matching;
};
class ActionMasks : public testing::TestWithParam<MaskCase> {};

TEST_P(ActionMasks, EqualEnumeratedLabels)
{
    const auto& p = GetParam();
    TaskOptions options;
    options.matching = p.matching;
    const auto task = Task::from_text_file(test::task_path(p.name), options);
    const auto lease = task->workspace();
    auto& succ = lease->successors();
    succ.set_witness_pruning(false);
    std::mt19937 rng(37);
    State state = task->initial_state();
    const u32 n = task->compiled().num_objects;
    std::vector<u64> mask(bits::words_for(n)), expected(mask.size());
    for (u32 step = 0; step < 4; ++step)
    {
        succ.set_witness_pruning(false);
        const auto labels = succ.applicable_actions(state);
        succ.set_witness_pruning(step % 2 == 0);
        succ.prepare(state);
        for (u32 schema = 0; schema < succ.num_schemas(); ++schema)
        {
            const bool exists = std::ranges::any_of(labels, [&](const Action& a) { return a.schema.v == schema; });
            EXPECT_EQ(succ.schema_applicable(schema), exists);
            for (u32 depth = 0; depth < succ.arity(schema); ++depth)
                for (u32 query = 0; query < 5; ++query)
                {
                    std::vector<ObjectId> prefix(depth);
                    const auto it = std::ranges::find_if(labels, [&](const Action& a) { return a.schema.v == schema; });
                    for (u32 d = 0; d < depth; ++d)
                        prefix[d] = query == 0 && it != labels.end() ? it->binding[d] : ObjectId{static_cast<u32>(rng() % (n + 1))};
                    std::fill(expected.begin(), expected.end(), 0);
                    for (const auto& a : labels)
                        if (a.schema.v == schema && std::equal(prefix.begin(), prefix.end(), a.binding.begin()))
                            bits::set(expected.data(), a.binding[depth].v);
                    succ.parameter_mask(schema, prefix, depth, mask.data());
                    EXPECT_EQ(mask, expected) << schema << '/' << depth << '/' << query;
                }
        }
        EXPECT_FALSE(succ.schema_applicable(succ.num_schemas()));
        succ.parameter_mask(succ.num_schemas(), {}, 0, mask.data());
        EXPECT_FALSE(bits::any(mask.data(), static_cast<u32>(mask.size())));
        if (labels.empty())
            break;
        state = succ.apply(state, labels[rng() % labels.size()].label());
    }
}

std::vector<MaskCase> cases()
{
    std::vector<MaskCase> out;
    for (const auto& task : test::suite())
        for (const auto mode : {TaskOptions::Matching::FixedOrder, TaskOptions::Matching::ForwardChecking})
            out.push_back({task.name, mode});
    for (const char* name : {"cs-counters", "m-refuel-adl", "cs-hydropower", "m-woodworking"})
        for (const auto mode : {TaskOptions::Matching::FixedOrder, TaskOptions::Matching::ForwardChecking})
            out.push_back({std::string("../numeric_tasks/") + name, mode});
    return out;
}
INSTANTIATE_TEST_SUITE_P(Bundled, ActionMasks, testing::ValuesIn(cases()), [](const auto& info)
{
    std::string name = info.param.name;
    std::ranges::replace_if(name, [](char c) { return !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')); }, '_');
    return name + (info.param.matching == TaskOptions::Matching::FixedOrder ? "_fixed" : "_fc");
});

TEST(ActionMaskRows, PoolAndMutableRows)
{
    const auto task = Task::from_text_file(test::task_path("gripper__prob05"));
    const auto suite = rl::TaskSuite::group({task});
    const u32 W = suite->words(), S = suite->max_schemas(), O = suite->max_objects();
    std::vector<u64> rows(3 * W);
    for (u32 i = 0; i < 3; ++i)
        suite->initial_row(0, rows.data() + i * W, W, 0);
    const rl::StateBatchView batch{rows.data(), 3, W};
    std::vector<u8> serial(3 * S), parallel(serial.size()), objects(3 * O), app(3);
    ThreadPool pool(3);
    rl::schema_masks(*suite, batch, nullptr, serial.data());
    rl::schema_masks(*suite, batch, nullptr, parallel.data(), &pool);
    EXPECT_EQ(serial, parallel);
    const i32 schema[] = {-1, 0, 999};
    rl::parameter_masks(*suite, batch, nullptr, schema, nullptr, 0, 0, objects.data(), O, &pool);
    EXPECT_TRUE(std::all_of(objects.begin(), objects.begin() + O, [](u8 v) { return !v; }));
    EXPECT_THROW(rl::is_applicable(*suite, batch, nullptr, schema, nullptr, 0, app.data()), std::invalid_argument);
    EXPECT_THROW(rl::is_applicable(*suite, batch, nullptr, schema, nullptr, 0, app.data(), &pool), std::invalid_argument);
    std::vector<i32> binding(3 * suite->label_width());
    rl::is_applicable(*suite, batch, nullptr, schema, binding.data(), suite->label_width(), app.data(), &pool);
    EXPECT_EQ(app[0], 0);
    EXPECT_EQ(app[2], 0);
    std::fill(rows.begin(), rows.end(), 0);
    rl::schema_masks(*suite, batch, nullptr, serial.data());
    const auto lease = task->workspace();
    for (u32 s = 0; s < S; ++s)
    {
        lease->successors().prepare({rows.data(), W});
        EXPECT_EQ(serial[s], lease->successors().schema_applicable(s));
    }
}
}  // namespace
