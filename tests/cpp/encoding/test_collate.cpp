#include "matrix.hpp"
#include "mymyr/encoding/collate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
        encoding::ExtraSizes extra;
        const encoding::FlatSizes exact = encoder->count(part, pool, scratch, &extra);
        const encoding::BatchShape caps{part.states.rows, exact.entities + 9, exact.args + 7, 0};
        const encoding::StaticShape static_caps{extra.statics.instances + 2, extra.statics.args + 11};
        encoding::FlatSizes sizes;
        batches.push_back(encoder->encode_capacity(part, caps, pool, scratch, sizes, &static_caps, &extra));
        ASSERT_FALSE(sizes.overflowed());
    }
    same(*encoder->encode(input), *encoding::collate(batches));
}

namespace
{
/// A copy of `batch` whose array `name` `change` edited (as elements of type I).
template<class I, class Change>
std::shared_ptr<const rl::ArrayBundle> edited(const rl::ArrayBundle& batch, std::string_view name, Change change)
{
    rl::ArrayBundle::Builder builder;
    for (const auto& [key, value] : batch.scalars())
        builder.scalar(key, value);
    for (const auto& info : batch.arrays())
        builder.add_uninitialized(info.name, info.dtype, info.shape);
    auto out = std::make_shared<rl::ArrayBundle>(builder.build());
    for (const auto& info : batch.arrays())
        std::memcpy(out->mutable_data(info.name.view()), batch.data(info), info.elements() * rl::dtype_bytes(info.dtype));
    const rl::ArrayInfo* info = out->find(name);
    EXPECT_NE(info, nullptr) << name;
    if (info)
        change(std::span(static_cast<I*>(out->mutable_data(*info)), static_cast<usize>(info->elements())));
    return out;
}

void expect_refused(const std::vector<std::shared_ptr<const rl::ArrayBundle>>& batches, std::string_view message)
{
    try
    {
        (void)encoding::collate(batches);
        ADD_FAILURE() << "collated: " << message;
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string_view(e.what()).find(message), std::string_view::npos) << e.what();
    }
}
}

TEST(EncodingCollate, SharedStaticsRefuseSlicesWhoseStaticRelationsOrBlocksDiffer)
{
    const auto table = table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto rows = make_batch(table, 24, true, 149);
    auto options = test::encoder_options();
    options.statics = encoding::Statics::Shared;
    const auto encoder = encoding::FlatEncoder::create(table, options);
    const auto input = host_input(rows, 24, {});
    ThreadPool pool(1);
    encoding::EncodeScratch scratch;
    // the second slice with capacity tails behind its blocks
    const auto slices = parts(input, rows.words, {12u, 12u});
    const auto first = encoder->encode(slices[0]);
    encoding::ExtraSizes extra;
    const encoding::FlatSizes exact = encoder->count(slices[1], pool, scratch, &extra);
    const encoding::BatchShape caps{12, exact.entities, exact.args, 0};
    const encoding::StaticShape static_caps{extra.statics.instances + 1, extra.statics.args + 5};
    encoding::FlatSizes sizes;
    const auto second = encoder->encode_capacity(slices[1], caps, pool, scratch, sizes, &static_caps, &extra);
    ASSERT_FALSE(sizes.overflowed());
    // both slices hold the block of the second slice's first instance
    const auto ids = [](const rl::ArrayBundle& b)
    {
        const auto* info = b.find("static_task_ids");
        const auto* p = static_cast<const i32*>(b.data(*info));
        return std::vector<i32>(p, p + info->elements());
    };
    const i32 shared_id = ids(*second).front();
    const std::vector<i32> first_ids = ids(*first);
    ASSERT_NE(std::ranges::find(first_ids, shared_id), first_ids.end());
    const auto expected = encoder->encode(input);
    same(*expected, *encoding::collate(std::vector{first, second}));

    // the capacity tails of the blocks are not compared
    const u64 live_args = static_cast<u64>(extra.statics.args);
    same(*expected, *encoding::collate(std::vector{first, edited<i32>(*second, "static_args", [&](std::span<i32> args)
    {
        ASSERT_GT(args.size(), live_args);
        for (u64 j = live_args; j < args.size(); ++j)
            args[j] = 7;
    })}));
    // another catalog of static relations
    expect_refused({first, edited<i32>(*second, "static_relations", [](std::span<i32> catalog)
    {
        ASSERT_GE(catalog.size(), 2u);
        std::swap(catalog[0], catalog[1]);
    })}, "the static relations of collated batches differ");
    // the block of the shared instance with another tuple count (its argument offsets stay) or other arguments
    const u64 relations = first->find("static_relations")->elements();
    const auto* offsets = static_cast<const i32*>(second->data(*second->find("static_offsets")));
    u64 filled = 0;
    while (filled < relations && offsets[filled + 1] == offsets[filled])
        ++filled;
    ASSERT_LT(filled, relations);
    const std::string differ = "the static blocks of instance " + std::to_string(shared_id) + " differ";
    expect_refused({first, edited<i32>(*second, "static_offsets", [&](std::span<i32> o) { o[filled + 1] -= 1; })}, differ);
    expect_refused({first, edited<i32>(*second, "static_args", [](std::span<i32> args)
    {
        ASSERT_FALSE(args.empty());
        args[0] += 1;
    })}, differ);
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

namespace
{
/// The encodes of the row slices {3, 0, 11, the rest} of `input`, whose rows are `width` words wide.
std::vector<std::shared_ptr<const rl::ArrayBundle>> encoded_parts(const encoding::FlatEncoder& encoder,
                                                                  const encoding::EncodeInput& input, u32 width)
{
    std::vector<std::shared_ptr<const rl::ArrayBundle>> out;
    for (const auto& part : parts(input, width, {3u, 0u, 11u, input.states.rows - 14}))
        out.push_back(encoder.encode(part));
    return out;
}

const encoding::Outputs k_relations{encoding::Output::Relations};
const encoding::Outputs k_both{encoding::Output::Relations, encoding::Output::Simple};
const encoding::Outputs k_simple{encoding::Output::Simple};
}

TEST(EncodingCollate, NumericValuesFollowTheirTuplesInBothLayoutsIndexTypesAndDtypes)
{
    for (const auto& c : numeric_corpus())
    {
        SCOPED_TRACE(c.name);
        const test::Rows rows = walk_rows(*c.table, 20, 167);
        const auto input = input_of(rows);
        for (const auto& outputs : {k_relations, k_both})
            for (const auto layout : {encoding::Layout::Graph, encoding::Layout::Relation})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                    for (const auto dtype : {encoding::NumericDtype::F32, encoding::NumericDtype::F64})
                    {
                        if (outputs.contains(encoding::Output::Simple) && layout == encoding::Layout::Relation)
                            continue;
                        auto options = with(test::encoder_options(), layout, index);
                        options.numeric_dtype = dtype;
                        options.outputs = outputs;
                        const auto encoder = encoding::FlatEncoder::create(c.table, options);
                        ASSERT_TRUE(encoder->schema().stores_numeric_values());
                        const auto expected = encoder->encode(input);
                        EXPECT_EQ(expected->scalar("numeric_relations"),
                                  encoder->schema().num_relations() - encoder->schema().first_numeric_relation());
                        same(*expected, *encoding::collate(encoded_parts(*encoder, input, c.table->row_words())));
                    }
    }
}

TEST(EncodingCollate, SimpleGraphsShiftTheirNodeIds)
{
    for (const auto& c : corpus())
    {
        SCOPED_TRACE(c.name);
        const auto rows = make_batch(c.table, 33, c.walk, 173);
        for (auto config : configs())
        {
            if (config.options.predicate_virtual_nodes || config.options.entity_arrays != encoding::EntityArrays::Always)
                continue;
            SCOPED_TRACE(config.name);
            for (const auto& outputs : {k_both, k_simple})
                for (const auto index : {encoding::IndexType::I32, encoding::IndexType::I64})
                {
                    config.options.index = index;
                    config.options.outputs = outputs;
                    const auto encoder = encoding::FlatEncoder::create(c.table, config.options);
                    const auto input = host_input(rows, 33, config.inputs);
                    same(*encoder->encode(input), *encoding::collate(encoded_parts(*encoder, input, rows.words)));
                }
        }
    }
    // a numeric table adds node_value; the Simple graph alone has no numeric_values
    for (const auto& c : numeric_corpus())
    {
        SCOPED_TRACE(c.name);
        const test::Rows rows = walk_rows(*c.table, 20, 179);
        const auto input = input_of(rows);
        auto options = test::encoder_options();
        options.outputs = k_simple;
        const auto encoder = encoding::FlatEncoder::create(c.table, options);
        ASSERT_TRUE(encoder->schema().stores_node_value());
        const auto expected = encoder->encode(input);
        EXPECT_EQ(expected->find("numeric_values"), nullptr);
        same(*expected, *encoding::collate(encoded_parts(*encoder, input, c.table->row_words())));
    }
}

TEST(EncodingCollate, NumericValuesAndSimpleGraphsIgnoreCapacityTails)
{
    for (const auto& c : numeric_corpus())
    {
        SCOPED_TRACE(c.name);
        const test::Rows rows = walk_rows(*c.table, 8, 181);
        const auto input = input_of(rows);
        for (const auto& outputs : {k_relations, k_both, k_simple})
        {
            auto options = test::encoder_options();
            options.outputs = outputs;
            const auto encoder = encoding::FlatEncoder::create(c.table, options);
            ThreadPool pool(1);
            encoding::EncodeScratch scratch;
            std::vector<std::shared_ptr<const rl::ArrayBundle>> batches;
            for (const auto& part : parts(input, c.table->row_words(), {5u, 0u, input.states.rows - 5}))
            {
                encoding::ExtraSizes extra;
                const encoding::FlatSizes exact = encoder->count(part, pool, scratch, &extra);
                encoding::BatchShape caps{part.states.rows, exact.entities + 9, exact.args + 7, 0};
                caps.numeric = extra.numeric + 5;
                encoding::FlatSizes sizes;
                batches.push_back(encoder->encode_capacity(part, caps, pool, scratch, sizes));
                ASSERT_FALSE(sizes.overflowed());
            }
            same(*encoder->encode(input), *encoding::collate(batches));
        }
    }
}

TEST(EncodingCollate, RefusesNumericValuesWithoutTheirRelationCount)
{
    const auto c = numeric_corpus().front();
    const test::Rows rows = walk_rows(*c.table, 2, 191);
    const auto batch = encoding::FlatEncoder::create(c.table, test::encoder_options())->encode(input_of(rows));
    rl::ArrayBundle::Builder builder;
    for (const auto& [key, value] : batch->scalars())
        if (key.view() != "numeric_relations")
            builder.scalar(key, value);
    for (const auto& info : batch->arrays())
        builder.add_uninitialized(info.name, info.dtype, info.shape);
    auto stripped = std::make_shared<rl::ArrayBundle>(builder.build());
    for (const auto& info : batch->arrays())
        std::memcpy(stripped->mutable_data(info.name.view()), batch->data(info),
                    info.elements() * rl::dtype_bytes(info.dtype));
    const std::vector<std::shared_ptr<const rl::ArrayBundle>> batches{stripped};
    EXPECT_THROW((void)encoding::collate(batches), std::invalid_argument);
}

namespace
{
/// A copy of `batch` with its scalar `key` set to `value`.
std::shared_ptr<const rl::ArrayBundle> with_scalar(const rl::ArrayBundle& batch, std::string_view key, i64 value)
{
    rl::ArrayBundle::Builder builder;
    for (const auto& [k, v] : batch.scalars())
        builder.scalar(k, k.view() == key ? value : v);
    for (const auto& info : batch.arrays())
        builder.add_uninitialized(info.name, info.dtype, info.shape);
    auto out = std::make_shared<rl::ArrayBundle>(builder.build());
    for (const auto& info : batch.arrays())
        std::memcpy(out->mutable_data(info.name.view()), batch.data(info), info.elements() * rl::dtype_bytes(info.dtype));
    return out;
}
}

TEST(EncodingCollate, RefusesNumericScalarsThatDifferBetweenSlicesOrDoNotDescribeTheValues)
{
    const auto c = numeric_corpus().front();
    const test::Rows rows = walk_rows(*c.table, 6, 193);
    const auto input = input_of(rows);
    const auto encoder = encoding::FlatEncoder::create(c.table, test::encoder_options());
    const auto slices = parts(input, c.table->row_words(), {3u, input.states.rows - 3});
    const auto first = encoder->encode(slices[0]), second = encoder->encode(slices[1]);
    const i64 numeric_relations = first->scalar("numeric_relations");
    ASSERT_GT(numeric_relations, 0);
    ASSERT_GT(first->find("numeric_values")->elements(), 0);
    ASSERT_EQ(first->scalar("numeric_dtype"), 0);
    same(*encoder->encode(input), *encoding::collate(std::vector{first, second}));
    // a later slice with other numeric scalars
    expect_refused({first, with_scalar(*second, "numeric_relations", numeric_relations - 1)},
                   "collate received mixed numeric_relations");
    expect_refused({first, with_scalar(*second, "numeric_dtype", 1)}, "collate received mixed numeric_dtype");
    // no numeric relation for the stored values, and a numeric_dtype that is not the values' dtype
    for (const auto& [key, value, message] :
         {std::tuple{"numeric_relations", i64{0}, "has no valid numeric_relations"},
          std::tuple{"numeric_dtype", i64{1}, "is not the dtype of its numeric values"}})
    {
        SCOPED_TRACE(key);
        const auto a = with_scalar(*first, key, value), b = with_scalar(*second, key, value);
        expect_refused({a}, message);
        expect_refused({a, b}, message);
    }
}
