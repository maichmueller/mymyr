#pragma once

// The table-wide arrays an encoder reads: for every instance of a task table, which relations of an EncodingSchema
// have tuples in its states and where those tuples come from (a slot range of the state words, a slot range of the
// derived words, the instance's static tuples, or its goal), plus the objects of every atom slot and static tuple. The
// goal of an instance is stored as masks over its fluent and derived slots plus a constant part: the goal literals over
// static predicates and over atoms without a slot, whose truth no state changes. They are fixed
// width arrays in one rl::ArrayBundle with per-instance offsets, so a copy of the block (for example on a device) is
// read through the same POD view, detail::ContextView (detail/lanes.hpp).

#include "mymyr/encoding/detail/lanes.hpp"
#include "mymyr/encoding/schema.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr::encoding
{
static_assert(detail::k_role_object == static_cast<u8>(EntityRole::Object));
static_assert(detail::k_role_auxiliary == static_cast<u8>(EntityRole::Auxiliary));

namespace detail
{
/// Throws std::logic_error naming `name` unless the slots [begin, end) of a task's atom table are the atoms of
/// `predicate` and their first `arity` objects ascend strictly (lexicographically) from slot to slot. `pred` holds the
/// predicate of each slot and `padded` the objects of each slot in rows of `width` entries.
void check_slot_range(std::span<const i32> pred, std::span<const i32> padded, u32 width, u32 begin, u32 end, u32 predicate,
                      u32 arity, const std::string& name);
}  // namespace detail

class EncodingContext
{
public:
    /// Builds the arrays of every instance of `table` for `schema`. Deterministic: equal inputs give equal bytes. The
    /// schema must be the schema of the table's domain (EncodingSchema::create(table, options)); throws
    /// std::invalid_argument otherwise, and std::logic_error if an instance's atom slots are not one ascending range per
    /// predicate.
    ///
    /// An instance has a cell for each domain predicate that has a state relation in the schema (the options can leave
    /// out statics, derived predicates, `=` and nullary predicates) and, in that instance, at least one atom slot or
    /// static tuple.
    [[nodiscard]] static std::shared_ptr<const EncodingContext> create(rl::TaskTablePtr table,
                                                                       std::shared_ptr<const EncodingSchema> schema);

    /// One block: inst (u64 [instances, detail::k_instance_columns]), cell (u32 [cells, 8]), slot_args, derived_args
    /// and static_args (i32 [rows, max_arity]; the objects of each fluent slot, derived slot and sorted, deduplicated
    /// static tuple of an instance, -1 padded), goal_words (u64: per instance the positive and negative goal masks over
    /// its fluent words, then over its derived words) and goal_args (i32 [rows, max_arity]: the constant goal tuples of
    /// each goal cell, sorted and without duplicates, -1 padded), action_schemas (u32: the schemas of each action
    /// relation, ascending), schema_arity (u32 [schemas]: the objects of each action schema's ground actions),
    /// relation_virtual (u32 [relations]: the virtual index of each relation's predicate, k_no_virtual for none) and
    /// virtual_relation (u32: the key relation of each virtual index), both empty without predicate virtual nodes, and
    /// relation_arity (u32 [relations]: the encoded arity of each relation). An instance has two lists of cells, each sorted by relation id with at most one cell per relation: the
    /// cells of a graph with the instance's goal, and the cells of a graph whose row has goal masks of its own (the two
    /// lists share their state and action cells). Per-graph cell counts (detail::count_graph) are per cell, in the
    /// order of the list the graph uses, not per relation. Every instance has a cell for each action relation of the
    /// schema, whose tuples come from the expansion rows of a graph and not from its state.
    ///
    /// An instance's goal is its encoding goal (Task::encoding_goal()): a literal whose predicate has no goal relation
    /// in the schema (the equality and nullary predicates can be left out) is dropped, the others become mask bits or
    /// constant tuples. The goal is complete (the `goal_complete` column) unless a literal was dropped whose truth is
    /// not fixed (one over a fluent or derived predicate). With the instance's goal a goal cell exists for each goal
    /// relation of a predicate that has a goal literal over atom slots or a constant literal, over the slots from the
    /// first to the last of those literals; with goals of the row, for each goal relation of a predicate that has atom
    /// slots, over all of its slots.
    [[nodiscard]] const rl::ArrayBundle& arrays() const noexcept { return m_arrays; }
    /// The arrays as a POD view on the copy of the block that starts at `block` (the host block by default; a device
    /// copy of the same bytes gives a device view).
    [[nodiscard]] detail::ContextView view(const std::byte* block = nullptr) const noexcept;

    [[nodiscard]] const rl::TaskTable& table() const noexcept { return *m_table; }
    [[nodiscard]] const rl::TaskTablePtr& table_ptr() const noexcept { return m_table; }
    [[nodiscard]] const EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] u32 max_arity() const noexcept { return m_max_arity; }
    /// The most derived words any instance that needs them has (0 if none does).
    [[nodiscard]] u32 max_derived_words() const noexcept { return m_max_derived_words; }
    /// Whether the graphs of the instance read the derived atoms of the state (detail::reads_derived).
    [[nodiscard]] bool needs_derived(u32 instance, bool row_goals = false, bool row_derived_goals = false) const noexcept
    {
        return detail::reads_derived(detail::instance_view(view(), instance), row_goals, row_derived_goals);
    }

private:
    EncodingContext() = default;

    rl::TaskTablePtr m_table;
    std::shared_ptr<const EncodingSchema> m_schema;
    rl::ArrayBundle m_arrays;
    u32 m_max_arity = 1;
    u32 m_max_derived_words = 0;
    u32 m_num_virtual = 0;
};
}
