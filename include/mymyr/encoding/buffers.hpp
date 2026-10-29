#pragma once

// The sizes and destination arrays of the capacity form of the relational encoder (encoder.hpp, cuda/encoding.hpp):
// plain structs, so that device code can use them.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// Capacities of the arrays of a batch: graphs bound the per-graph and per-cell arrays, entities the entity arrays,
/// args the entries of relation_args, targets the target_entity and target_group arrays, history the history_entity
/// and history_dt arrays, and numeric the entries of numeric_values (only a schema that stores numeric values has a
/// numeric capacity; the true count is FlatEncoder::numeric_tuples). Tuple counts have no separate capacity: the only
/// arrays indexed by tuple are the Simple node arrays, which simple_node_capacity bounds.
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

/// The Simple nodes (entities and tuples) of a batch within `capacities`, at most: a tuple with arguments takes at
/// least one of the args, and a cell of a relation without argument slots holds at most one tuple, the empty one.
/// `argumentless_relations` is the schema's number of such relations (EncodingSchema::argumentless_relations).
MYMYR_HD u64 simple_node_capacity(const BatchShape& capacities, u64 argumentless_relations)
{
    return capacities.entities + capacities.args + capacities.graphs * argumentless_relations;
}

/// Capacities of the static blocks of a batch with shared statics: the instances that have a block and the entries of
/// static_args. Tuples have no separate capacity because no static array is indexed by tuple. Kept apart from
/// BatchShape because only encoders that store the static tuples once per instance have these arrays.
struct StaticShape
{
    u64 instances = 0;
    u64 args = 0;
    [[nodiscard]] friend bool operator==(const StaticShape&, const StaticShape&) noexcept = default;
};

/// The true sizes of the static blocks: the distinct instances of the batch's graphs, and the tuples and arguments of
/// their blocks.
struct StaticSizes
{
    u64 instances = 0;
    u64 tuples = 0;
    u64 args = 0;
};

/// The true sizes of a batch that its FlatSizes do not hold (the FlatSizes keep the six words of the device's sizes):
/// its numeric values and its static blocks.
struct ExtraSizes
{
    u64 numeric = 0;        // the numeric tuples, the length of numeric_values (0 if the schema stores none)
    StaticSizes statics{};  // the static blocks (all 0 unless the schema shares its statics)
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
/// The Simple arrays hold the nodes (entities + tuples of the batch, at most simple_node_capacity) and edges (twice the
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
    // The static blocks, present iff the schema shares its statics (EncodingSchema::shared_statics): the arrays
    // static_task_ids [I_b], static_offsets and static_arg_offsets [I_b * R_s + 1], static_args [A_s] and
    // static_relations [R_s]. static_offsets and static_arg_offsets must hold I_b * R_s + 1 entries (the entries of the
    // blocks), static_task_ids I_b and static_args the arguments of the blocks (ExtraSizes::statics); static_relations is always
    // R_s entries. The destination arrays of a capacity form may be longer; the encoder writes only the blocks' entries.
    i32* static_task_ids = nullptr;
    void* static_offsets = nullptr;
    void* static_arg_offsets = nullptr;
    void* static_args = nullptr;
    void* static_relations = nullptr;
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
