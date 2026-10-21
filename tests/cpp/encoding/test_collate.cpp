#include "matrix.hpp"
#include "mymyr/encoding/collate.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstring>
#include <initializer_list>
#include <vector>

using namespace mymyr;
using namespace mymyr::test::encoding_matrix;

namespace
{
void same(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    EXPECT_EQ(a.scalars(), b.scalars());
    ASSERT_EQ(a.arrays().size(), b.arrays().size());
    for (const auto& info : a.arrays())
    {
        const auto* other = b.find(info.name.view());
        ASSERT_NE(other, nullptr);
        ASSERT_EQ(info.shape, other->shape) << info.name;
        EXPECT_EQ(std::memcmp(a.data(info), b.data(*other), info.elements() * rl::dtype_bytes(info.dtype)), 0) << info.name;
    }
}

/// The consecutive slices of `input` with `counts` rows each.
std::vector<encoding::EncodeInput> parts(const encoding::EncodeInput& input, u32 words, std::initializer_list<u64> counts)
{
    std::vector<encoding::EncodeInput> out;
    u64 start = 0;
    for (const u64 count : counts)
    {
        auto part = input;
        part.states.data += start * words;
        part.states.rows = count;
        part.task_ids += start;
        if (part.derived) part.derived += start * part.derived_stride;
        if (part.goal_pos)
        {
            part.goal_pos += start * part.goal_stride;
            part.goal_neg += start * part.goal_stride;
        }
        if (part.goal_derived_pos)
        {
            part.goal_derived_pos += start * part.goal_derived_stride;
            part.goal_derived_neg += start * part.goal_derived_stride;
        }
        out.push_back(part);
        start += count;
    }
    return out;
}

TEST(EncodingCollate, SlicesEqualOneEncodeInBothLayoutsAndIndexTypes)
{
    for (const auto& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const auto rows = make_batch(c.table, 33, c.walk, 137);
        for (auto config : configs())
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    config.options.layout = layout;
                    config.options.index = index;
                    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                    const auto input = host_input(rows, 33, config.inputs);
                    const auto expected = encoder->encode(input);
                    std::vector<std::shared_ptr<const rl::ArrayBundle>> batches;
                    for (const auto& part : parts(input, rows.words, {3u, 0u, 11u, 19u}))
                        batches.push_back(encoder->encode(part));
                    same(*expected, *encoding::collate(batches));
                }
    }
}

TEST(EncodingCollate, SharedStaticsCollateToTheBlocksOfOneEncode)
{
    for (const auto& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const auto rows = make_batch(c.table, 33, c.walk, 141);
        for (auto config : configs())
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    SCOPED_TRACE(config.name);
                    config.options.layout = layout;
                    config.options.index = index;
                    config.options.statics = encoding::Statics::Shared;
                    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                    const auto input = host_input(rows, 33, config.inputs);
                    const auto expected = encoder->encode(input);
                    std::vector<std::shared_ptr<const rl::ArrayBundle>> batches;
                    for (const auto& part : parts(input, rows.words, {3u, 0u, 11u, 19u}))
                        batches.push_back(encoder->encode(part));
                    same(*expected, *encoding::collate(batches));
                }
    }
}

TEST(EncodingCollate, SharedStaticsIgnoreTheCapacityTailsOfTheBlocks)
{
    const auto table = table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto rows = make_batch(table, 24, true, 143);
    auto options = test::encoder_options();
    options.statics = encoding::Statics::Shared;
    options.predicate_virtual_nodes = true;
    const auto encoder = encoding::FlatEncoder::create(table, options);
    const auto input = host_input(rows, 24, {});
    ThreadPool pool(1);
    encoding::EncodeScratch scratch;
    std::vector<std::shared_ptr<const rl::ArrayBundle>> batches;
    for (const auto& part : parts(input, rows.words, {5u, 0u, 19u}))
    {
        encoding::StaticSizes statics;
        const encoding::FlatSizes exact = encoder->count(part, pool, scratch, &statics);
        const encoding::BatchShape caps{part.states.rows, exact.entities + 9, exact.args + 7, 0};
        const encoding::StaticShape static_caps{statics.instances + 2, statics.args + 11};
        encoding::FlatSizes sizes;
        batches.push_back(encoder->encode_capacity(part, caps, pool, scratch, sizes, &static_caps, &statics));
        ASSERT_FALSE(sizes.overflowed());
    }
    same(*encoder->encode(input), *encoding::collate(batches));
}

TEST(EncodingCollate, RefusesMixedStatics)
{
    const auto c = corpus().front();
    const auto rows = make_batch(c.table, 3, false, 147);
    auto options = test::encoder_options();
    const auto repeated = encoding::FlatEncoder::create(c.table, options)->encode(host_input(rows, 3, {}));
    options.statics = encoding::Statics::Shared;
    const auto shared = encoding::FlatEncoder::create(c.table, options)->encode(host_input(rows, 3, {}));
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{shared, repeated};
    EXPECT_THROW((void)encoding::collate(batches), std::invalid_argument);
}

TEST(EncodingCollate, EmptySlicesRetainTheSchema)
{
    const auto c = corpus().front();
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    const auto rows = make_batch(c.table, 0, false, 139);
    const auto empty = encoder->encode(host_input(rows, 0, {}));
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{empty, empty};
    same(*empty, *encoding::collate(batches));
    EXPECT_THROW((void)encoding::collate({}), std::invalid_argument);
}

TEST(EncodingCollate, RefusesBatchesOfDifferentTablesOverOneSchema)
{
    // Task ids name instances of their table: batches of two tables share the schema but not the ids' meaning.
    const auto one = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto two = table_of("blocks", {"probBLOCKS-8-0", "smedium"});
    const auto first = encoding::FlatEncoder::create(one, test::encoder_options());
    const auto second = encoding::FlatEncoder::create(two, test::encoder_options());
    ASSERT_EQ(first->schema().fingerprint(), second->schema().fingerprint());
    ASSERT_NE(first->table_fingerprint(), second->table_fingerprint());
    const auto a = first->encode(host_input(make_batch(one, 3, false, 151), 3, {}));
    const auto b = second->encode(host_input(make_batch(two, 3, false, 157), 3, {}));
    EXPECT_EQ(a->scalar("table_fingerprint"), std::bit_cast<i64>(first->table_fingerprint()));
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> mixed{a, b};
    EXPECT_THROW((void)encoding::collate(mixed), std::invalid_argument);
    const auto again = encoding::FlatEncoder::create(two, test::encoder_options());
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> same_table{b, again->encode(host_input(make_batch(two, 2, false, 163), 2, {}))};
    EXPECT_EQ(encoding::collate(same_table)->scalar("num_graphs"), 5);
}

TEST(EncodingCollate, RefusesMixedLayoutsIndexTypesAndSchemas)
{
    const auto c = corpus().front();
    const auto rows = make_batch(c.table, 3, false, 149);
    auto options = test::encoder_options();
    const auto first = encoding::FlatEncoder::create(c.table, options)->encode(host_input(rows, 3, {}));
    for (int change = 0; change < 3; ++change)
    {
        auto other = options;
        if (change == 0) other.layout = encoding::Layout::Relation;
        if (change == 1) other.index = encoding::IndexType::I64;
        if (change == 2) other.include_static = !other.include_static;
        const auto second = encoding::FlatEncoder::create(c.table, other)->encode(host_input(rows, 3, {}));
        const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{first, second};
        EXPECT_THROW((void)encoding::collate(batches), std::invalid_argument);
    }
}
}

TEST(EncodingCollate, ChecksCombinedWideTotalsBeforeAllocatingIndexArrays)
{
    rl::ArrayBundle::Builder builder;
    builder.scalar("version", 1); builder.scalar("num_graphs", 1); builder.scalar("num_relations", 0);
    builder.scalar("layout", 0); builder.scalar("index_dtype", 0); builder.scalar("statics", 0);
    builder.scalar("schema_fingerprint", 0);
    builder.add("task_ids", rl::DType::I32, {1}, std::vector<i32>{0});
    builder.add("num_objects", rl::DType::I32, {1}, std::vector<i32>{0x7fffffff});
    builder.add("entity_offsets", rl::DType::I32, {2}, std::vector<i32>{0, 0x7fffffff});
    builder.add("relation_offsets", rl::DType::I32, {1}, std::vector<i32>{0});
    builder.add("relation_arg_offsets", rl::DType::I32, {1}, std::vector<i32>{0});
    builder.add_uninitialized("relation_args", rl::DType::I32, {0});
    const auto batch = std::make_shared<const rl::ArrayBundle>(builder.build());
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{batch, batch};
    EXPECT_THROW((void)encoding::collate(batches), std::overflow_error);
}
