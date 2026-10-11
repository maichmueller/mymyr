#pragma once

// The table-wide arrays an encoder reads: for every instance of a task table, which relations of an EncodingSchema
// have tuples in its states and where those tuples come from (a slot range of the state words, a slot range of the
// derived words, or the instance's static tuples), plus the objects of every atom slot and static tuple. They are fixed
// width arrays in one rl::ArrayBundle with per-instance offsets, so a copy of the block (for example on a device) is
// read through the same POD view, detail::ContextView (detail/lanes.hpp).

#include "mymyr/encoding/detail/lanes.hpp"
#include "mymyr/encoding/schema.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <vector>

namespace mymyr::encoding
{
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

    /// One block: inst (u64 [instances, detail::k_instance_columns]), cell (u32 [cells, 6]), slot_args, derived_args
    /// and static_args (i32 [rows, max_arity]; the objects of each fluent slot, derived slot and sorted, deduplicated
    /// static tuple of an instance, -1 padded).
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
    /// Whether the instance has a derived cell, i.e. its graphs read the derived atoms of the state.
    [[nodiscard]] bool needs_derived(u32 instance) const noexcept { return m_needs_derived[instance] != 0; }

private:
    EncodingContext() = default;

    rl::TaskTablePtr m_table;
    std::shared_ptr<const EncodingSchema> m_schema;
    rl::ArrayBundle m_arrays;
    u32 m_max_arity = 1;
    u32 m_max_derived_words = 0;
    std::vector<u8> m_needs_derived;
};
}
