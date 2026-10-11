#include "../support/suite.hpp"
#include "mymyr/encoding/flat.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>

using namespace mymyr;

namespace
{
TaskPtr task(TaskOptions::Atoms mode = TaskOptions::Atoms::Frozen)
{
    TaskOptions options;
    options.atoms = mode;
    return Task::from_text_file(test::task_path("gripper__prob05"), options);
}

std::span<const i32> array(const encoding::FlatBatch& batch, const char* name)
{
    const rl::ArrayInfo& info = *batch.arrays->find(name);
    return {static_cast<const i32*>(batch.arrays->data(info)), info.elements()};
}

TEST(FlatEncoding, CountsArgumentsAndNullaryOffsetsAreConsistent)
{
    const auto source = task();
    const State state = source->initial_state();
    const encoding::FlatRelationEncoder encoder(source);
    const auto batch = encoder.encode({state.data(), 1, state.size_words()});
    const auto counts = array(batch, "relation_counts");
    const auto offsets = array(batch, "relation_offsets");
    const auto tuple_offsets = array(batch, "relation_tuple_offsets");
    for (u32 relation = 0; relation < batch.schema->names.size(); ++relation)
    {
        EXPECT_EQ(offsets[relation + 1] - offsets[relation], counts[relation] * batch.schema->arities[relation]);
        EXPECT_EQ(tuple_offsets[relation + 1] - tuple_offsets[relation], counts[relation]);
    }
    EXPECT_EQ(batch.graphs, 1u);
    EXPECT_EQ(batch.task_fingerprints[0], source->fingerprint());
    EXPECT_EQ(batch.object_names.size(), source->num_objects());
}

TEST(FlatEncoding, MultiInstanceObjectOffsetsAndIds)
{
    const auto first = task();
    const auto second = task();
    const auto table = rl::TaskTable::create({first, second});
    const encoding::FlatRelationEncoder encoder(table);
    std::vector<u64> states;
    for (const auto& source : {second, first})
    {
        const State state = source->initial_state();
        const usize begin = states.size();
        states.resize(begin + table->words(), 0);
        std::copy(state.words().begin(), state.words().end(), states.begin() + begin);
    }
    const i32 ids[]{1, 0};
    const auto batch = encoder.encode({states.data(), 2, table->words()}, ids);
    const auto objects = array(batch, "object_offsets");
    EXPECT_EQ(objects[0], 0);
    EXPECT_EQ(objects[1], second->num_objects());
    EXPECT_EQ(objects[2], first->num_objects() + second->num_objects());
    EXPECT_THROW(static_cast<void>(encoder.encode({states.data(), 2, table->words()})), std::invalid_argument);
    const i32 invalid_ids[]{1, 2};
    EXPECT_THROW(static_cast<void>(encoder.encode({states.data(), 2, table->words()}, invalid_ids)), std::invalid_argument);
}

TEST(FlatEncoding, ActionTargetsRetainCallerOrder)
{
    const auto source = task();
    const State state = source->initial_state();
    const WorkspaceLease workspace = source->workspace();
    auto& successors = workspace->successors();
    std::vector<i32> schemas, bindings;
    u32 width = 0;
    for (const auto& schema : source->data().schemas)
        width = std::max(width, schema.arity());
    successors.prepare(state.view());
    successors.generate<false>([&](u32 schema, const ObjectId* binding, const Delta&)
    {
        schemas.push_back(static_cast<i32>(schema));
        for (u32 position = 0; position < width; ++position)
            bindings.push_back(position < source->data().schemas[schema].arity() ? static_cast<i32>(binding[position].v) : -1);
        return true;
    }, false, true);
    const i32 offsets[]{0, static_cast<i32>(schemas.size())};
    const encoding::ActionBatchView actions{offsets, schemas.data(), bindings.data(), schemas.size(), width};
    const encoding::FlatRelationEncoder encoder(source);
    const auto batch = encoder.encode({state.data(), 1, state.size_words()}, nullptr, actions);
    EXPECT_EQ(std::vector<i32>(array(batch, "action_schema").begin(), array(batch, "action_schema").end()), schemas);
    EXPECT_EQ(std::vector<i32>(array(batch, "action_binding").begin(), array(batch, "action_binding").end()), bindings);
    const auto tuple_ids = array(batch, "action_tuple");
    EXPECT_EQ(tuple_ids.size(), schemas.size());
    EXPECT_TRUE(std::ranges::all_of(tuple_ids, [](i32 value) { return value >= 0; }));
    auto invalid = actions;
    const i32 bad_offsets[]{1, static_cast<i32>(schemas.size())};
    invalid.offsets = bad_offsets;
    EXPECT_THROW(static_cast<void>(encoder.encode({state.data(), 1, state.size_words()}, nullptr, invalid)), std::invalid_argument);
}

TEST(FlatEncoding, EmptyBatchAndZeroWidthState)
{
    const auto source = task(TaskOptions::Atoms::Lazy);
    const encoding::FlatRelationEncoder encoder(source);
    const auto empty = encoder.encode({nullptr, 0, 0});
    EXPECT_EQ(empty.graphs, 0u);
    EXPECT_EQ(array(empty, "relation_offsets")[0], 0);
    const auto zero = encoder.encode({nullptr, 1, 0});
    EXPECT_EQ(zero.graphs, 1u);
    EXPECT_THROW(static_cast<void>(encoder.encode({nullptr, 1, 1})), std::invalid_argument);
    std::vector<u64> invalid(source->max_words() + 1, 0);
    invalid.back() = 1;
    EXPECT_THROW(static_cast<void>(encoder.encode({invalid.data(), 1, static_cast<u32>(invalid.size())})), std::invalid_argument);
}

TEST(FlatEncoding, BatchOwnsMetadataAndArrayStorage)
{
    encoding::FlatBatch batch;
    {
        const auto source = task();
        const encoding::FlatRelationEncoder encoder(source);
        const State state = source->initial_state();
        batch = encoder.encode({state.data(), 1, state.size_words()});
    }
    EXPECT_FALSE(batch.schema->names.empty());
    EXPECT_FALSE(batch.object_names.empty());
    EXPECT_FALSE(array(batch, "relation_args").empty());
}

TEST(FlatEncoding, DuplicateInitialFactsDoNotChangeTheObservation)
{
    const auto original = task();
    auto data = formalism::read_task_text_file(test::task_path("gripper__prob05"));
    ASSERT_FALSE(data.static_init.empty());
    data.static_init.push_back(data.static_init.front());
    const auto repeated = Task::create(std::move(data));
    const State first = original->initial_state(), second = repeated->initial_state();
    const auto expected = encoding::FlatRelationEncoder(original).encode({first.data(), 1, first.size_words()});
    const auto actual = encoding::FlatRelationEncoder(repeated).encode({second.data(), 1, second.size_words()});
    EXPECT_TRUE(std::ranges::equal(array(actual, "relation_counts"), array(expected, "relation_counts")));
    EXPECT_TRUE(std::ranges::equal(array(actual, "relation_args"), array(expected, "relation_args")));
}

TEST(FlatEncoding, DefaultStaticGoalLiteralsRetainBothTruthPolarities)
{
    auto data = formalism::read_task_text_file(test::task_path("gripper__prob05"));
    ASSERT_FALSE(data.static_init.empty());
    const auto atom = data.static_init.front();
    std::vector<formalism::Term> terms;
    for (ObjectId object : data.objects_of(atom))
        terms.push_back(formalism::object_term(object));
    const auto arguments = formalism::TaskData::append(data.terms, std::span<const formalism::Term>(terms));
    const formalism::Literal goals[]{{atom.pred, true, arguments}, {atom.pred, false, arguments}};
    data.goal = {formalism::TaskData::append(data.literals, std::span<const formalism::Literal>(goals)), {}};
    const auto source = Task::create(std::move(data));
    const State state = source->initial_state();
    const auto batch = encoding::FlatRelationEncoder(source).encode({state.data(), 1, state.size_words()});
    EXPECT_EQ(array(batch, "relation_counts")[atom.pred.v * 5 + 1], 1);
    EXPECT_EQ(array(batch, "relation_counts")[atom.pred.v * 5 + 4], 1);
}

TEST(FlatEncoding, InvalidStrideAndOversizedAddressSpanAreRejected)
{
    const auto source = task();
    const encoding::FlatRelationEncoder encoder(source);
    const u64 words[]{0, 0};
    EXPECT_THROW(static_cast<void>(encoder.encode({words, 1, 2, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(encoder.encode({words, 2, 1, std::numeric_limits<u64>::max()})), std::length_error);
}

TEST(FlatEncoding, SharedEncoderIsReentrant)
{
    const auto source = task(TaskOptions::Atoms::Lazy);
    const encoding::FlatRelationEncoder encoder(source);
    const State state = source->initial_state();
    const auto expected = encoder.encode({state.data(), 1, state.size_words()});
    std::atomic<bool> equal{true};
    std::vector<std::thread> threads;
    for (u32 thread = 0; thread < 8; ++thread)
        threads.emplace_back([&]
        {
            for (u32 repeat = 0; repeat < 20; ++repeat)
            {
                const auto batch = encoder.encode({state.data(), 1, state.size_words()});
                if (!std::ranges::equal(array(batch, "relation_args"), array(expected, "relation_args")))
                    equal.store(false);
            }
        });
    for (auto& thread : threads)
        thread.join();
    EXPECT_TRUE(equal.load());
}
}
