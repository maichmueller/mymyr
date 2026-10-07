#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_arrays_view.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstring>
#include <vector>

using namespace mymyr;

namespace
{
TaskPtr numeric_task(const char* name)
{
    return Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/" + name + ".txt");
}
}

TEST(NumericDeviceArrays, OptionalSectionAndDeterministicEncoding)
{
    const auto classical = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/tasks/gripper__prob05.txt");
    EXPECT_EQ(rl::device_arrays(*classical).find("num_code"), nullptr);
    EXPECT_EQ(rl::device_arrays(*classical).scalar("section_numeric", 0), 0);
    for (const char* name : {"cs-counters", "cs-hydropower", "cs-farmland"})
    {
        const auto task = numeric_task(name);
        const auto a = rl::device_arrays(*task), b = rl::device_arrays(*task);
        ASSERT_EQ(a.bytes(), b.bytes());
        EXPECT_EQ(std::memcmp(a.block(), b.block(), a.bytes()), 0);
        EXPECT_EQ(a.scalar("section_numeric"), 1);
        EXPECT_EQ(a.scalar("numeric_slots"), task->numeric_slots());
        const auto view = rl::task_view(a);
        EXPECT_EQ(view.numeric.slots, task->numeric_slots());
        for (u32 slot = 0; slot < task->numeric_slots(); ++slot)
            EXPECT_EQ(std::bit_cast<u64>(view.numeric.initial[slot]),
                      std::bit_cast<u64>(plan::load(task->compiled().num, task->initial_state().numeric().data(), slot)));
    }
}

TEST(NumericDeviceArrays, PointersRebaseWithTheImmutableBlock)
{
    const auto task = numeric_task("cs-farmland");
    const auto arrays = rl::device_arrays(*task);
    std::vector<u64> copy((arrays.bytes() + 7) / 8);
    std::memcpy(copy.data(), arrays.block(), arrays.bytes());
    const auto a = rl::task_view(arrays), b = rl::task_view(arrays, copy.data());
    const auto offset = [&](const void* p, const void* base) {
        return static_cast<const std::byte*>(p) - static_cast<const std::byte*>(base);
    };
    EXPECT_EQ(offset(a.numeric.code, arrays.block()), offset(b.numeric.code, copy.data()));
    EXPECT_EQ(offset(a.numeric.table, arrays.block()), offset(b.numeric.table, copy.data()));
    EXPECT_EQ(offset(a.numeric.effects, arrays.block()), offset(b.numeric.effects, copy.data()));
    EXPECT_EQ(a.numeric.quantum, b.numeric.quantum);
    EXPECT_EQ(b.numeric.goal_count, task->compiled().num.goal.size());
}

TEST(NumericDeviceArrays, ProgramsMatchCpuOnWalkStates)
{
    for (const char* name : {"cs-counters", "cs-hydropower", "cs-farmland", "cs-sailing"})
    {
        const auto task = numeric_task(name);
        const auto arrays = rl::device_arrays(*task);
        const auto v = rl::task_view(arrays);
        const auto& n = task->compiled().num;
        State state = task->initial_state();
        for (u32 step = 0; step < 8; ++step)
        {
            std::vector<u64> values(n.slots);
            for (u32 i = 0; i < n.slots; ++i)
                values[i] = std::bit_cast<u64>(plan::load(n, state.numeric().data(), i));
            const auto actions = task->workspace().successors().applicable_actions(state.view());
            for (const auto& action : actions)
            {
                const auto label = action.label();
                const auto& schema = task->compiled().schemas[label.schema.v];
                const auto compare = [&](plan::NumProg p) {
                    const f64 cpu = plan::eval(n, p, state.numeric().data(), label.binding.data());
                    const f64 device = rl::dev::numeric_eval(v.numeric, p.begin, p.end, values.data(),
                        reinterpret_cast<const u32*>(label.binding.data()), v.num_objects);
                    EXPECT_EQ(std::bit_cast<u64>(cpu), std::bit_cast<u64>(device)) << name;
                };
                for (const auto& m : schema.pre)
                {
                    for (const auto& c : m.npre) { compare(c.lhs); compare(c.rhs); }
                    for (const auto& c : m.nchecks) { compare(c.lhs); compare(c.rhs); }
                }
                for (const auto& group : schema.uncond_num)
                {
                    for (const auto& effect : group.neffs) compare(effect.expr);
                    if (group.has_aux) compare(group.aux.expr);
                }
            }
            if (actions.empty()) break;
            state = task->workspace().successors().apply(state.view(), actions[step % actions.size()].label());
        }
    }
}

TEST(NumericDeviceArrays, MetricProgramMetadata)
{
    for (const char* name : {"cs-counters", "cs-farmland", "cs-delivery", "cs-delivery-nometric"})
    {
        const auto task = numeric_task(name);
        const auto arrays = rl::device_arrays(*task);
        const auto& num = task->compiled().num;
        EXPECT_EQ(arrays.scalar("num_has_aux"), num.has_aux ? 1u : 0u);
        EXPECT_EQ(arrays.scalar("num_has_metric"), num.has_metric ? 1u : 0u);
        EXPECT_EQ(arrays.scalar("num_metric_begin"), num.metric.begin);
        EXPECT_EQ(arrays.scalar("num_metric_end"), num.metric.end);
    }
}
