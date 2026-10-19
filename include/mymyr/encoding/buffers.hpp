#pragma once

// The sizes and destination arrays of the capacity form of the relational encoder (encoder.hpp, cuda/encoding.hpp):
// plain structs, so that device code can use them.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// Capacities of the arrays of a batch: graphs bound the per-graph and per-cell arrays, entities the entity arrays,
/// args the entries of relation_args, and targets the target_entity and target_group arrays. Tuple counts have no
/// separate capacity because every tuple has at least one argument and no array is indexed by tuple alone.
struct BatchShape
{
    u64 graphs = 0;
    u64 entities = 0;
    u64 args = 0;
    u64 targets = 0;
    [[nodiscard]] friend bool operator==(const BatchShape&, const BatchShape&) noexcept = default;
};

/// Capacities of the static blocks of a batch with shared statics: the instances that have a block and the entries of
/// static_args. Tuples have no separate capacity because every tuple has at least one argument. Kept apart from
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

/// Flags of FlatSizes::overflow.
inline constexpr u32 k_overflow_capacity = 1;  // some size exceeds its capacity
inline constexpr u32 k_overflow_index = 2;     // some size or count does not fit the index type
inline constexpr u32 k_overflow_input = 4;     // a row is invalid: a task id outside the table, a state row or goal mask
                                               // narrower than its instance, or a bit its instance does not have. Device
                                               // calls report it here, host calls throw

/// The true sizes of a batch (not clipped to any capacity) and whether the batch can be written. A POD, so that a
/// device call can return it in the same form; its five 64-bit words are entities, tuples, args, targets and the
/// overflow flags, and no byte of it is padding.
struct FlatSizes
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
    u32 overflow = 0;  // k_overflow_* flags; 0: the batch fits
    u32 reserved = 0;  // always 0: the flags fill a 64-bit word
    [[nodiscard]] bool overflowed() const noexcept { return overflow != 0; }
};
static_assert(sizeof(FlatSizes) == 5 * sizeof(u64));

/// The destination arrays of a batch of G graphs and R relations, typed by the index type of the options (int32 or
/// int64). The caller owns them. Arrays of size G, G + 1, G * R or G * R + 1 must have that size; relation_args,
/// entity_role, entity_ref, target_entity and target_group must hold at least the sizes of the batch (FlatSizes).
/// entity_role and entity_ref must be null iff the schema has implicit entity arrays, relation_counts iff it does not
/// store counts, goal_complete iff it has no goal relations, the three target arrays iff it has no targets. Nothing is
/// written past the sizes: the entries of the variable-size arrays beyond them keep their old values.
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
    // blocks), static_task_ids I_b and static_args the arguments of the blocks (StaticSizes); static_relations is always
    // R_s entries. The destination arrays of a capacity form may be longer; the encoder writes only the blocks' entries.
    i32* static_task_ids = nullptr;
    void* static_offsets = nullptr;
    void* static_arg_offsets = nullptr;
    void* static_args = nullptr;
    void* static_relations = nullptr;
};
}
