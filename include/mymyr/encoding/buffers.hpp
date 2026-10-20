#pragma once

// The sizes and destination arrays of the capacity form of the relational encoder (encoder.hpp, cuda/encoding.hpp):
// plain structs, so that device code can use them.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// Capacities of the arrays of a batch: graphs bound the per-graph and per-cell arrays, entities the entity arrays,
/// args the entries of relation_args, targets the target_entity and target_group arrays, history the history_entity
/// and history_dt arrays, and numeric the entries of numeric_values (only a schema that stores numeric values has a
/// numeric capacity; the true count is FlatEncoder::numeric_tuples). Tuple counts have no separate capacity because every
/// tuple has at least one argument and no array is indexed by tuple alone.
struct BatchShape
{
    u64 graphs = 0;
    u64 entities = 0;
    u64 args = 0;
    u64 targets = 0;
    u64 history = 0;
    u64 numeric = 0;
    [[nodiscard]] friend bool operator==(const BatchShape&, const BatchShape&) noexcept = default;
};

/// Flags of FlatSizes::overflow.
inline constexpr u32 k_overflow_capacity = 1;  // some size exceeds its capacity
inline constexpr u32 k_overflow_index = 2;     // some size or count does not fit the index type
inline constexpr u32 k_overflow_input = 4;     // a row is invalid: a task id outside the table, a state row or goal mask
                                               // narrower than its instance, or a bit its instance does not have. Device
                                               // calls report it here, host calls throw

/// The true sizes of a batch (not clipped to any capacity) and whether the batch can be written. A POD, so that a
/// device call can return it in the same form; its six 64-bit words are entities, tuples, args, targets, history
/// and the overflow flags, and no byte of it is padding.
struct FlatSizes
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
    u64 history = 0;
    u32 overflow = 0;  // k_overflow_* flags; 0: the batch fits
    u32 reserved = 0;  // always 0: the flags fill a 64-bit word
    [[nodiscard]] bool overflowed() const noexcept { return overflow != 0; }
};
static_assert(sizeof(FlatSizes) == 6 * sizeof(u64));

/// The destination arrays of a batch of G graphs and R relations, typed by the index type of the options (int32 or
/// int64). The caller owns them. Arrays of size G, G + 1, G * R or G * R + 1 must have that size; relation_args,
/// entity_role, entity_ref, target_entity, target_group, history_entity and history_dt must hold at least the sizes of
/// the batch (FlatSizes). entity_role and entity_ref must be null iff the schema has implicit entity arrays,
/// relation_counts iff it does not store counts, goal_complete iff it has no goal relations, the three target arrays
/// iff it has no targets, the three history arrays iff it has no history. The relation arrays (relation_counts,
/// relation_offsets, relation_arg_offsets, relation_args) are null iff the schema does not store them
/// (EncodingSchema::emits_relations), the Simple arrays (node_offsets, node_label, edge_offsets, edge_index,
/// edge_label) are null iff it does not store the Simple graph. numeric_values is null iff the schema does not store
/// numeric values (EncodingSchema::stores_numeric_values) and node_value iff it does not store node values
/// (stores_node_value); numeric_values holds at least the batch's numeric tuples, node_value at least its nodes.
/// Nothing is written past the sizes: the entries of the variable-size arrays beyond them keep their old values.
///
/// The Simple arrays hold the nodes (entities + tuples of the batch, at most entities + args) and edges (twice the
/// args): node_label holds at least the nodes and edge_label at least the edges, and edge_index is two rows of
/// `edge_stride` entries each (at least the edges), row 0 the sources and row 1 the targets.
struct FlatBuffers
{
    i32* task_ids = nullptr;
    void* num_objects = nullptr;
    void* entity_offsets = nullptr;
    u8* entity_role = nullptr;
    void* entity_ref = nullptr;
    void* relation_counts = nullptr;
    void* relation_offsets = nullptr;
    void* relation_arg_offsets = nullptr;
    void* relation_args = nullptr;
    u8* goal_complete = nullptr;
    void* target_offsets = nullptr;
    void* target_entity = nullptr;
    u8* target_group = nullptr;
    void* history_offsets = nullptr;
    void* history_entity = nullptr;
    i32* history_dt = nullptr;
    void* node_offsets = nullptr;   // [G + 1]
    i32* node_label = nullptr;
    void* edge_offsets = nullptr;   // [G + 1]
    void* edge_index = nullptr;     // [2, edge_stride]
    u64 edge_stride = 0;
    i32* edge_label = nullptr;
    void* numeric_values = nullptr;  // [T_num] float or double (FlatOptions::numeric_dtype)
    f32* node_value = nullptr;       // [V]
};
}
