#pragma once

// The relational encoder: state rows of a task table to one array bundle of graphs, one graph per row. A graph has the
// instance's objects as entities (plus one auxiliary entity iff some nullary tuple exists) and, for every relation of
// the schema, the tuples of the row's atoms: fluent atoms from the state, derived atoms, and the instance's static
// atoms. Every per-graph decision lives in detail/lanes.hpp; this file only loops, scans, allocates and checks.
//
// The bundle holds (G graphs, R = schema().num_relations(), I the index type of the options: int32 or int64; every
// entity id is batch-global, the graph's first entity is entity_offsets[g]):
//   task_ids             i32 [G]       the instance of each row
//   num_objects          I   [G]       the instance's objects (the auxiliary entity is not counted)
//   entity_offsets       I   [G + 1]   graph g's entities are [entity_offsets[g], entity_offsets[g + 1])
//   entity_role          u8  [E]       the EntityRole: 0 object, 8 auxiliary
//   entity_ref           I   [E]       object: its instance-local id; auxiliary: -1
//   relation_counts      I   [G, R]    the tuples of cell (g, r)
//   relation_offsets     I   [G * R + 1]      cell (g, r) is segment g * R + r: its tuples are [offsets[s], offsets[s + 1])
//   relation_arg_offsets I   [G * R + 1]      the same for the argument array
//   relation_args        I   [A]       the entities of the tuples, tuple after tuple, Relation::arity per tuple
// and the scalars version, num_graphs, num_relations, layout (0: graph), index_dtype (0: int32, 1: int64),
// schema_fingerprint (the schema's fingerprint as an i64 bit pattern) and statics (0: the static tuples are repeated in
// every graph).
//
// The tuples of a cell are the atoms of its predicate in the order of the instance's atom slots, which is the
// lexicographic order of their objects; a nullary atom is the unary tuple [auxiliary entity]. A graph contains nothing
// that its row and instance do not have. Encoding a slice of a batch gives the slice of the batch's encoding, with the
// offsets rebased.

#include "mymyr/encoding/context.hpp"
#include "mymyr/encoding/options.hpp"
#include "mymyr/rl/expand.hpp"

#include <memory>

namespace mymyr::encoding
{
/// The inputs of one batch.
struct EncodeInput
{
    rl::StateBatchView states;      // N rows (numeric_words must be 0); the stride is honoured
    const i32* task_ids = nullptr;  // [N] the instance of each row; null: instance 0
    const u64* derived = nullptr;   // [N, derived_stride] the derived words of each row; null: evaluated here
    u64 derived_stride = 0;         // words from one row of `derived` to the next (at least max_derived_words())
};

class FlatEncoder
{
public:
    /// Throws std::invalid_argument ("mymyr: encoding: ...") for invalid options and for options this encoder does not
    /// encode: goal derivations, subgoal levels, targets, predicate virtual nodes, horizons and the relation layout.
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(rl::TaskTablePtr table, const FlatOptions& options);
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(TaskPtr task, const FlatOptions& options);

    [[nodiscard]] const rl::TaskTable& table() const noexcept { return m_context->table(); }
    [[nodiscard]] const rl::TaskTablePtr& table_ptr() const noexcept { return m_context->table_ptr(); }
    [[nodiscard]] const EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const std::shared_ptr<const EncodingSchema>& schema_ptr() const noexcept { return m_schema; }
    [[nodiscard]] const EncodingContext& context() const noexcept { return *m_context; }

    /// Encodes the rows. Const and reentrant. Throws std::invalid_argument for invalid input (a task id out of range,
    /// a row narrower than its instance, a state bit its instance does not have, numeric words) and std::overflow_error
    /// when a count exceeds the index type.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in) const;

private:
    FlatEncoder() = default;

    std::shared_ptr<const EncodingSchema> m_schema;
    std::shared_ptr<const EncodingContext> m_context;
};

namespace detail
{
/// Throws std::overflow_error unless `value` fits the index type; `what` names the counted thing.
void check_index_range(u64 value, IndexType index, const char* what);
}
}
