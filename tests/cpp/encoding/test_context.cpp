#include "support.hpp"

#include "mymyr/encoding/context.hpp"
#include "mymyr/encoding/detail/literals.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <sstream>

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

/// The cells of the state and goal relations of an instance (the action cells come with `with_actions`).
std::vector<Cell> cells_of(const EncodingContext& context, u32 instance, bool with_actions = false)
{
    const lanes::InstanceView v = lanes::instance_view(context.view(), instance);
    std::vector<Cell> cells;
    for (u32 i = 0; i < v.num_cells; ++i)
        if (with_actions || !lanes::is_action_op(v.cells[i].op))
            cells.push_back(v.cells[i]);
    return cells;
}

std::set<std::string> cell_names(const EncodingContext& context, u32 instance)
{
    std::set<std::string> out;
    for (const Cell& cell : cells_of(context, instance))
        out.insert(context.schema().relation(cell.relation).name);
    return out;
}

std::string file_text(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

/// The bytes of one named array of the context.
std::vector<std::byte> bytes_of(const EncodingContext& context, const char* name)
{
    const rl::ArrayInfo& info = *context.arrays().find(name);
    const std::byte* at = static_cast<const std::byte*>(context.arrays().data(info));
    return {at, at + info.elements() * rl::dtype_bytes(info.dtype)};
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
        // the lists of an instance are stored once each, one after the other; a list equal to an earlier one shares it
        u64 end = cells;
        for (const auto& [count, offset] : {std::pair{lanes::k_num_cells, lanes::k_cell_offset},
                                            std::pair{lanes::k_num_row_cells, lanes::k_row_cell_offset},
                                            std::pair{lanes::k_num_plain_cells, lanes::k_plain_cell_offset},
                                            std::pair{lanes::k_num_plain_row_cells, lanes::k_plain_row_cell_offset}})
        {
            EXPECT_GE(row[offset], cells);
            end = std::max(end, row[offset] + row[count]);
        }
        EXPECT_EQ(row[lanes::k_num_objects], table->instance(i).num_objects);
        EXPECT_EQ(row[lanes::k_encoding_fingerprint], table->task(i)->encoding_fingerprint());
        cells = end;
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

TEST(EncodingContext, PredicatesAreMappedThroughTheTablesOwnSchemaWhateverTheDeclarationOrder)
{
    // The same blocks domain with its predicates declared in the opposite order: equal schemas (the fingerprint and
    // the sorted relation names do not depend on the order), different predicate ids.
    const std::string directory = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/blocks/";
    std::string domain = file_text(directory + "domain.pddl");
    const std::string declared = "(:predicates (on ?x ?y)\n\t       (ontable ?x)\n\t       (clear ?x)\n\t       (handempty)\n\t       (holding ?x)\n\t       )";
    const std::size_t at = domain.find(declared);
    ASSERT_NE(at, std::string::npos);
    domain.replace(at, declared.size(), "(:predicates (holding ?x)\n\t       (handempty)\n\t       (clear ?x)\n\t       (ontable ?x)\n\t       (on ?x ?y))");
    const std::filesystem::path reordered = std::filesystem::path(::testing::TempDir()) / "reordered_blocks_domain.pddl";
    std::ofstream(reordered) << domain;

    const auto a = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto b = rl::TaskTable::single(Task::create(*frontend::load_task(reordered, directory + "smedium.pddl")));
    std::filesystem::remove(reordered);
    ASSERT_NE(a->predicate_names(), b->predicate_names());
    const auto schema_a = EncodingSchema::create(*a, test::encoder_options());
    const auto schema_b = EncodingSchema::create(*b, test::encoder_options());
    ASSERT_EQ(schema_a->fingerprint(), schema_b->fingerprint());

    const auto own = EncodingContext::create(a, schema_a);
    const auto foreign = EncodingContext::create(a, schema_b);
    for (const char* name : {"inst", "cell", "slot_args", "derived_args", "static_args"})
        EXPECT_EQ(bytes_of(*own, name), bytes_of(*foreign, name)) << name;
    // every cell is the relation named like its predicate
    const auto& names = a->predicate_names();
    const auto& arities = a->predicate_arities();
    for (const Cell& cell : cells_of(*foreign, 0))
    {
        const std::string& relation = foreign->schema().relation(cell.relation).name;
        const auto predicate = std::ranges::find(names, relation);
        ASSERT_NE(predicate, names.end()) << relation;
        EXPECT_EQ(cell.logical_arity, static_cast<u32>(arities[static_cast<std::size_t>(predicate - names.begin())])) << relation;
    }
}

TEST(EncodingContext, SlotRangesAreCheckedToBeOneAscendingRangePerPredicate)
{
    // slots 0..3: predicate 1 with binary atoms (0,1) (0,2) (1,0), then predicate 2
    const std::vector<i32> pred = {1, 1, 1, 2};
    const std::vector<i32> padded = {0, 1, 0, 2, 1, 0, 5, -1};
    using lanes::check_slot_range;
    EXPECT_NO_THROW(check_slot_range(pred, padded, 2, 0, 3, 1, 2, "p"));
    EXPECT_NO_THROW(check_slot_range(pred, padded, 2, 3, 4, 2, 1, "q"));
    EXPECT_NO_THROW(check_slot_range(pred, padded, 2, 2, 2, 7, 2, "empty"));
    EXPECT_THROW(check_slot_range(pred, padded, 2, 0, 4, 1, 2, "a range that spills into the next predicate"), std::logic_error);
    EXPECT_THROW(check_slot_range(pred, padded, 2, 0, 3, 2, 2, "the wrong predicate"), std::logic_error);
    const std::vector<i32> descending = {0, 2, 0, 1, 1, 0, 5, -1};
    EXPECT_THROW(check_slot_range(pred, descending, 2, 0, 3, 1, 2, "descending atoms"), std::logic_error);
    const std::vector<i32> repeated = {0, 1, 0, 1, 1, 0, 5, -1};
    EXPECT_THROW(check_slot_range(pred, repeated, 2, 0, 3, 1, 2, "a repeated atom"), std::logic_error);
}

TEST(EncodingContext, EveryFixedSlotLayoutGivesContiguousSlotRanges)
{
    for (const auto atoms : {TaskOptions::Atoms::Reachable, TaskOptions::Atoms::Dense})
    {
        TaskOptions options;
        options.atoms = atoms;
        const auto table = rl::TaskTable::single(Task::create(test::pddl_task("blocks", "smedium")->data(), options));
        const auto context = context_of(table, test::encoder_options());
        EXPECT_GT(count_op(cells_of(*context, 0), CellOp::Fluent), 0u);
    }
}

TEST(EncodingContext, DuplicateStaticAtomsGiveTheTuplesOfTheTaskWithoutThem)
{
    const TaskPtr original = test::pddl_task("gripper", "p01");
    formalism::TaskData data = original->data();
    ASSERT_FALSE(data.static_init.empty());
    data.static_init.push_back(data.static_init.front());
    data.static_init.push_back(data.static_init.back());
    const auto repeated = rl::TaskTable::single(Task::create(std::move(data)));
    const auto expected = context_of(rl::TaskTable::single(original), test::encoder_options());
    const auto actual = context_of(repeated, test::encoder_options());
    for (const char* name : {"cell", "static_args"})
        EXPECT_EQ(bytes_of(*actual, name), bytes_of(*expected, name)) << name;
}
TEST(EncodingContext, TheCellsOfTheInstanceGoalHoldOnlyTheSlotsItsMasksCanSetAndRowGoalsKeepEveryPredicateWhole)
{
    // the goal: seven positive literals over on/2 of the eight blocks
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "probBLOCKS-8-0"));
    FlatOptions options = test::encoder_options();
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    const auto context = context_of(table, options);
    const EncodingSchema& schema = context->schema();
    const lanes::InstanceView v = lanes::instance_view(context->view(), 0);
    auto goal_cells = [](const Cell* cells, u32 count)
    {
        std::vector<Cell> out;
        for (u32 i = 0; i < count; ++i)
            if (lanes::is_goal_op(cells[i].op))
                out.push_back(cells[i]);
        return out;
    };

    // the instance's goal: only the goal relations of on/2 with a positive literal, over the slots from its first to
    // its last
    const std::vector<Cell> own = goal_cells(v.cells, v.num_cells);
    ASSERT_EQ(own.size(), 3u);
    const std::vector<Cell> state_cells_of_instance = cells_of(*context, 0);
    const Cell& state_on = *std::ranges::find(state_cells_of_instance, *schema.find("on"), &Cell::relation);
    for (const Cell& cell : own)
    {
        EXPECT_EQ(schema.relation(cell.relation).sign, 1) << schema.relation(cell.relation).name;
        EXPECT_EQ(schema.relation(cell.relation).name.rfind("[+]on[g]", 0), 0u);
        ASSERT_LT(cell.a, cell.b);
        EXPECT_EQ(lanes::popcount_range(v.goal_pos, cell.a, cell.b), 7u);
        EXPECT_EQ(lanes::popcount_range(v.goal_pos, cell.a, cell.a + 1), 1u);
        EXPECT_EQ(lanes::popcount_range(v.goal_pos, cell.b - 1, cell.b), 1u);
        EXPECT_GE(cell.a, state_on.a);
        EXPECT_LE(cell.b, state_on.b);
        EXPECT_LT(cell.b - cell.a, state_on.b - state_on.a);
    }

    // the goals of a row: every goal relation of every predicate with slots, over the predicate's whole range
    const std::vector<Cell> whole = goal_cells(v.row_cells, v.num_row_cells);
    EXPECT_EQ(whole.size(), 5u * 2u * 3u);  // clear, handempty, holding, on, ontable; both signs; three variants
    for (const Cell& cell : whole)
    {
        const Relation& relation = schema.relation(cell.relation);
        const u32 state_relation = *schema.find(table->predicate_names()[relation.predicate]);
        const Cell& state = *std::ranges::find(state_cells_of_instance, state_relation, &Cell::relation);
        EXPECT_EQ(cell.a, state.a) << relation.name;
        EXPECT_EQ(cell.b, state.b) << relation.name;
        EXPECT_EQ(cell.c, cell.d) << relation.name;
    }
    // the other cells are the same in both lists
    std::vector<Cell> state_cells, row_state_cells;
    for (u32 i = 0; i < v.num_cells; ++i)
        if (!lanes::is_goal_op(v.cells[i].op))
            state_cells.push_back(v.cells[i]);
    for (u32 i = 0; i < v.num_row_cells; ++i)
        if (!lanes::is_goal_op(v.row_cells[i].op))
            row_state_cells.push_back(v.row_cells[i]);
    ASSERT_EQ(state_cells.size(), row_state_cells.size());
    EXPECT_EQ(std::memcmp(state_cells.data(), row_state_cells.data(), state_cells.size() * sizeof(Cell)), 0);
    EXPECT_TRUE(std::ranges::is_sorted(std::span(v.cells, v.num_cells), {}, &Cell::relation));
    EXPECT_TRUE(std::ranges::is_sorted(std::span(v.row_cells, v.num_row_cells), {}, &Cell::relation));
}

TEST(EncodingContext, ACellWithOnlyConstantGoalTuplesHasNoSlots)
{
    // (road a b) is static: its goal cells have constant tuples and no slot range, and row goals have no such cell
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "static_false"));
    FlatOptions options = test::encoder_options();
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied};
    const auto context = context_of(table, options);
    const lanes::InstanceView v = lanes::instance_view(context->view(), 0);
    u32 constant_cells = 0;
    for (u32 i = 0; i < v.num_cells; ++i)
    {
        if (lanes::is_goal_op(v.cells[i].op) && v.cells[i].a == v.cells[i].b)
        {
            EXPECT_LT(v.cells[i].c, v.cells[i].d) << context->schema().relation(v.cells[i].relation).name;
            ++constant_cells;
        }
    }
    EXPECT_GT(constant_cells, 0u);
    for (u32 i = 0; i < v.num_row_cells; ++i)
    {
        if (lanes::is_goal_op(v.row_cells[i].op))
        {
            EXPECT_LT(v.row_cells[i].a, v.row_cells[i].b);
        }
    }
}

/// The kind of the atom `objects` of instance predicate `ip` from the task's atom metadata, by a linear scan.
lanes::LiteralSlot expected_literal(const rl::ArrayBundle& meta, u32 ip, const std::vector<i32>& objects)
{
    auto column = [&](const char* name)
    {
        const rl::ArrayInfo& info = *meta.find(name);
        return std::span(static_cast<const i32*>(meta.data(info)), info.elements());
    };
    auto scan = [&](const char* pred_name, const char* padded_name, u32& at)
    {
        const auto pred = column(pred_name), padded = column(padded_name);
        const u64 width = pred.empty() ? 0 : padded.size() / pred.size();
        for (u64 k = 0; k < pred.size(); ++k)
            if (static_cast<u32>(pred[k]) == ip &&
                std::equal(objects.begin(), objects.end(), padded.begin() + static_cast<std::ptrdiff_t>(k * width)))
            {
                at = static_cast<u32>(k);
                return true;
            }
        return false;
    };
    u32 at = 0;
    switch (column("pred_kind")[ip])
    {
    case rl::k_pred_static:
        return {scan("static_pred", "static_args_padded", at) ? lanes::LiteralKind::StaticTrue
                                                              : lanes::LiteralKind::StaticFalse,
                0};
    case rl::k_pred_derived:
        return scan("derived_pred", "derived_args_padded", at) ? lanes::LiteralSlot{lanes::LiteralKind::Derived, at}
                                                               : lanes::LiteralSlot{lanes::LiteralKind::Unreachable, 0};
    default:
        return scan("atom_pred", "atom_args_padded", at) ? lanes::LiteralSlot{lanes::LiteralKind::Fluent, at}
                                                         : lanes::LiteralSlot{lanes::LiteralKind::Unreachable, 0};
    }
}

TEST(EncodingContext, TheLiteralSectionResolvesEveryAtomUnderEveryFixedLayoutAndAnyOptions)
{
    // every tuple of objects of every domain predicate: fluent, derived and static predicates, nullary ones and `=`
    std::vector<FlatOptions> variants(6, test::encoder_options());
    variants[1].include_static = false;
    variants[2].include_derived = false;
    variants[3].nullary = false;
    variants[4].statics = Statics::Shared;
    variants[5].horizon.mode = HorizonMode::Delta;
    for (const auto atoms : {TaskOptions::Atoms::Reachable, TaskOptions::Atoms::Dense})
    {
        TaskOptions task_options;
        task_options.atoms = atoms;
        const TaskPtr task = Task::create(test::pddl_task("goals", "static_true")->data(), task_options);
        const auto table = rl::TaskTable::single(task);
        const rl::ArrayBundle meta = rl::atom_metadata(*task);
        const auto& instance = table->instance(0);
        const u32 objects = task->num_objects();
        for (const FlatOptions& options : variants)
        {
            const auto context = context_of(table, options);
            const lanes::ContextView view = context->view();
            ASSERT_EQ(view.num_predicates, table->num_domain_predicates());
            u32 kinds[6] = {};
            for (u32 ip = 0; ip < instance.pred_map.size(); ++ip)
            {
                const u32 tp = instance.pred_map[ip];
                if (tp >= table->num_domain_predicates())
                    continue;
                const u32 arity = table->predicate_arities()[tp];
                ASSERT_LE(arity, context->max_arity());
                std::vector<i32> tuple(arity, 0);
                for (bool more = true; more;)
                {
                    std::vector<i32> padded = tuple;
                    padded.resize(context->max_arity() + 1, -1);
                    const lanes::LiteralSlot got =
                        lanes::resolve_literal(view, 0, tp, padded.data(), static_cast<u32>(padded.size()));
                    const lanes::LiteralSlot want = expected_literal(meta, ip, tuple);
                    EXPECT_EQ(got.kind, want.kind) << table->predicate_names()[tp];
                    EXPECT_EQ(got.slot, want.slot) << table->predicate_names()[tp];
                    ++kinds[static_cast<u32>(got.kind)];
                    more = false;
                    for (u32 k = arity; k-- > 0 && !more;)
                    {
                        more = ++tuple[k] < static_cast<i32>(objects);
                        if (!more)
                            tuple[k] = 0;
                    }
                }
            }
            for (const lanes::LiteralKind kind : {lanes::LiteralKind::Fluent, lanes::LiteralKind::Derived,
                                                  lanes::LiteralKind::StaticTrue, lanes::LiteralKind::StaticFalse})
                EXPECT_GT(kinds[static_cast<u32>(kind)], 0u) << static_cast<u32>(kind);
            if (atoms == TaskOptions::Atoms::Reachable)
            {
                EXPECT_GT(kinds[static_cast<u32>(lanes::LiteralKind::Unreachable)], 0u);
            }
            EXPECT_EQ(kinds[static_cast<u32>(lanes::LiteralKind::Invalid)], 0u);

            // out of range: the instance, the predicate, an object, the padding, a row too narrow for the arity
            const auto& names = table->predicate_names();
            const i64 at = std::ranges::find(names, std::string("at")) - names.begin();
            const i64 past = table->num_domain_predicates();
            const i32 good[3] = {0, 1, -1}, negative[3] = {-1, 1, -1}, beyond[3] = {0, static_cast<i32>(objects), -1},
                      unpadded[3] = {0, 1, 0};
            EXPECT_NE(lanes::resolve_literal(view, 0, at, good, 3).kind, lanes::LiteralKind::Invalid);
            for (const auto& [instance_id, predicate, row, width] :
                 {std::tuple{i64{1}, at, good, 3u}, std::tuple{i64{-1}, at, good, 3u},
                  std::tuple{i64{0}, i64{-1}, good, 3u}, std::tuple{i64{0}, past, good, 3u},
                  std::tuple{i64{0}, at, negative, 3u}, std::tuple{i64{0}, at, beyond, 3u},
                  std::tuple{i64{0}, at, unpadded, 3u}, std::tuple{i64{0}, at, good, 1u}})
                EXPECT_EQ(lanes::resolve_literal(view, instance_id, predicate, row, width).kind,
                          lanes::LiteralKind::Invalid);
        }
    }
}
}  // namespace
