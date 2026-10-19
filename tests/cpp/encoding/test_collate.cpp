#include "matrix.hpp"
#include "mymyr/encoding/collate.hpp"

#include <gtest/gtest.h>

#include <cstring>

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
                    u64 start = 0;
                    for (const u64 count : {3u, 0u, 11u, 19u})
                    {
                        auto part = input;
                        part.states.data += start * rows.words;
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
                        batches.push_back(encoder->encode(part));
                        start += count;
                    }
                    same(*expected, *encoding::collate(batches));
                }
    }
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
