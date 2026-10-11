#include "support.hpp"

#include "mymyr/encoding/context.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <set>

using namespace mymyr;
using namespace mymyr::encoding;
namespace lanes = mymyr::encoding::detail;
using Cell = lanes::Cell;
using CellOp = lanes::CellOp;

namespace
{
std::shared_ptr<const EncodingContext> context_of(const rl::TaskTablePtr& table, const FlatOptions& options)
{
    return EncodingContext::create(table, EncodingSchema::create(*table, options));
}

std::vector<Cell> cells_of(const EncodingContext& context, u32 instance)
{
    const lanes::InstanceView v = lanes::instance_view(context.view(), instance);
    return {v.cells, v.cells + v.num_cells};
}

std::set<std::string> cell_names(const EncodingContext& context, u32 instance)
{
    std::set<std::string> out;
    for (const Cell& cell : cells_of(context, instance))
        out.insert(context.schema().relation(cell.relation).name);
    return out;
}

u32 count_op(const std::vector<Cell>& cells, CellOp op)
{
    return static_cast<u32>(std::ranges::count(cells, static_cast<u32>(op), &Cell::op));
}

TEST(EncodingContext, BlocksCellsAreTheStateRelationsWithTheirSlotRanges)
{
    const auto task = test::pddl_task("blocks", "smedium");
    const auto table = rl::TaskTable::single(task);
    const auto context = context_of(table, test::encoder_options());
    EXPECT_EQ(cell_names(*context, 0), (std::set<std::string>{"clear", "handempty", "holding", "object", "on", "ontable"}));
    const EncodingSchema& schema = context->schema();
    const auto cells = cells_of(*context, 0);
    EXPECT_TRUE(std::ranges::is_sorted(cells, {}, &Cell::relation));
    const rl::ArrayBundle meta = rl::atom_metadata(*task);
    const auto* offsets = static_cast<const i32*>(meta.data(*meta.find("pred_slot_offsets")));
    const u32 predicates = static_cast<u32>(meta.find("pred_arity")->elements());
    const auto& instance = table->instance(0);
    u32 checked = 0, statics = 0;
    for (u32 ip = 0; ip < predicates; ++ip)
    {
        const std::string& name = table->predicate_names()[instance.pred_map[ip]];
        const auto relation = schema.find(name);
        if (!relation)
            continue;
        const auto it = std::ranges::find(cells, *relation, &Cell::relation);
        if (it == cells.end() && schema.relation(*relation).lane == Lane::Static)
            continue;  // a type with no object of it in this instance has no static tuple, so no cell
        ASSERT_NE(it, cells.end()) << name;
        if (it->op == static_cast<u32>(CellOp::Static))
        {
            ++statics;  // the type predicate `object`
            continue;
        }
        EXPECT_EQ(it->op, static_cast<u32>(CellOp::Fluent)) << name;
        EXPECT_EQ(it->a, static_cast<u32>(offsets[ip])) << name;
        EXPECT_EQ(it->b, static_cast<u32>(offsets[ip + 1])) << name;
        EXPECT_EQ(it->logical_arity, schema.relation(*relation).logical_arity);
        EXPECT_EQ(it->arity, schema.relation(*relation).arity);
        ++checked;
    }
    EXPECT_EQ(checked, 5u);
    EXPECT_EQ(statics, 1u);
    const Cell& handempty = *std::ranges::find(cells, *schema.find("handempty"), &Cell::relation);
    EXPECT_EQ(handempty.logical_arity, 0u);
    EXPECT_EQ(handempty.arity, 1u);
    EXPECT_EQ(handempty.b - handempty.a, 1u);
    const lanes::InstanceView v = lanes::instance_view(context->view(), 0);
    EXPECT_EQ(v.num_objects, task->num_objects());
    EXPECT_EQ(v.fluent_slots, task->atoms().fluent_slots());
    EXPECT_EQ(v.words, instance.words);
    EXPECT_FALSE(context->needs_derived(0));
    EXPECT_EQ(context->max_derived_words(), 0u);
    EXPECT_EQ(context->max_arity(), 2u);
}

TEST(EncodingContext, NullaryOffRemovesTheNullaryCells)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    FlatOptions options = test::encoder_options();
    options.nullary = false;
    const auto context = context_of(table, options);
    EXPECT_EQ(cell_names(*context, 0), (std::set<std::string>{"clear", "holding", "object", "on", "ontable"}));
    for (const Cell& cell : cells_of(*context, 0))
        EXPECT_GT(cell.logical_arity, 0u);
}

TEST(EncodingContext, StaticTuplesAreSortedAndUnique)
{
    const auto table = rl::TaskTable::single(test::pddl_task("gripper", "p01"));
    const auto context = context_of(table, test::encoder_options());
    const auto cells = cells_of(*context, 0);
    ASSERT_GT(count_op(cells, CellOp::Static), 0u);
    const lanes::InstanceView v = lanes::instance_view(context->view(), 0);
    const u32 A = context->max_arity();
    for (const Cell& cell : cells)
    {
        if (cell.op != static_cast<u32>(CellOp::Static))
            continue;
        for (u32 s = cell.a + 1; s < cell.b; ++s)
            EXPECT_TRUE(std::lexicographical_compare(v.static_args + (s - 1) * A, v.static_args + (s - 1) * A + cell.logical_arity,
                                                     v.static_args + s * A, v.static_args + s * A + cell.logical_arity))
                << context->schema().relation(cell.relation).name << " row " << s;
    }
    FlatOptions options = test::encoder_options();
    options.include_static = false;
    EXPECT_EQ(count_op(cells_of(*context_of(table, options), 0), CellOp::Static), 0u);
}

TEST(EncodingContext, PhilosophersHaveDerivedCellsAndEqualityOnlyOnRequest)
{
    const auto task = test::pddl_task("philosophers", "p03-phil4");
    const auto table = rl::TaskTable::single(task);
    const auto plain = context_of(table, test::encoder_options());
    const auto cells = cells_of(*plain, 0);
    EXPECT_GT(count_op(cells, CellOp::Derived), 0u);
    EXPECT_TRUE(plain->needs_derived(0));
    EXPECT_GT(plain->max_derived_words(), 0u);
    EXPECT_FALSE(plain->schema().find("="));
    EXPECT_EQ(cell_names(*plain, 0).count("="), 0u);

    FlatOptions equality = test::encoder_options();
    equality.equality = true;
    const auto with_equality = context_of(table, equality);
    const auto id = with_equality->schema().find("=");
    ASSERT_TRUE(id);
    const auto equality_cells = cells_of(*with_equality, 0);
    const auto cell = std::ranges::find(equality_cells, *id, &Cell::relation);
    ASSERT_NE(cell, equality_cells.end());
    EXPECT_EQ(cell->op, static_cast<u32>(CellOp::Static));
    EXPECT_EQ(cell->b - cell->a, task->num_objects());

    FlatOptions no_derived = test::encoder_options();
    no_derived.include_derived = false;
    const auto without = context_of(table, no_derived);
    EXPECT_EQ(count_op(cells_of(*without, 0), CellOp::Derived), 0u);
    EXPECT_FALSE(without->needs_derived(0));
    EXPECT_EQ(without->max_derived_words(), 0u);
}

TEST(EncodingContext, AMixedTableHasConsistentOffsetsAndBuildsDeterministically)
{
    const auto small = test::pddl_task("blocks", "smedium");
    const auto large = test::pddl_task("blocks", "probBLOCKS-8-0");
    const auto table = rl::TaskTable::create({small, large, small});
    const FlatOptions options = test::encoder_options();
    const auto context = context_of(table, options);
    const lanes::ContextView view = context->view();
    u64 cells = 0, slots = 0;
    for (u32 i = 0; i < table->size(); ++i)
    {
        const u64* row = view.inst + i * lanes::k_instance_columns;
        EXPECT_EQ(row[lanes::k_cell_offset], cells);
        EXPECT_EQ(row[lanes::k_slot_args_offset], slots);
        EXPECT_EQ(row[lanes::k_num_objects], table->instance(i).num_objects);
        EXPECT_EQ(row[lanes::k_encoding_fingerprint], table->task(i)->encoding_fingerprint());
        cells += row[lanes::k_num_cells];
        slots += row[lanes::k_fluent_slots];
    }
    EXPECT_EQ(cells, context->arrays().find("cell")->shape[0]);
    EXPECT_EQ(slots, static_cast<u64>(context->arrays().find("slot_args")->shape[0]));
    // the first and third instance are the same task
    EXPECT_EQ(cells_of(*context, 0).size(), cells_of(*context, 2).size());

    const auto again = context_of(table, options);
    ASSERT_EQ(again->arrays().bytes(), context->arrays().bytes());
    EXPECT_EQ(std::memcmp(again->arrays().block(), context->arrays().block(), context->arrays().bytes()), 0);

    // a view on a copy of the block reads the same values
    std::vector<std::byte> copy(context->arrays().block(), context->arrays().block() + context->arrays().bytes());
    const lanes::ContextView moved = context->view(copy.data());
    EXPECT_NE(moved.inst, view.inst);
    EXPECT_EQ(moved.num_instances, view.num_instances);
    for (u32 i = 0; i < table->size(); ++i)
    {
        const lanes::InstanceView a = lanes::instance_view(view, i), b = lanes::instance_view(moved, i);
        EXPECT_EQ(a.num_cells, b.num_cells);
        EXPECT_EQ(a.fluent_slots, b.fluent_slots);
        for (u32 c = 0; c < a.num_cells; ++c)
            EXPECT_EQ(std::memcmp(a.cells + c, b.cells + c, sizeof(Cell)), 0);
        for (u64 k = 0; k < static_cast<u64>(a.fluent_slots) * a.max_arity; ++k)
            ASSERT_EQ(a.slot_args[k], b.slot_args[k]);
    }
}

TEST(EncodingContext, ASchemaOfAnotherDomainIsRefused)
{
    const auto blocks = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto gripper = rl::TaskTable::single(test::pddl_task("gripper", "p01"));
    EXPECT_THROW((void)EncodingContext::create(blocks, EncodingSchema::create(*gripper, test::encoder_options())), std::invalid_argument);
    EXPECT_THROW((void)EncodingContext::create(nullptr, EncodingSchema::create(*blocks, test::encoder_options())), std::invalid_argument);
}
}  // namespace
