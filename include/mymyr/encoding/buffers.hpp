#pragma once

// The sizes and destination arrays of the capacity form of the relational encoder (encoder.hpp, cuda/encoding.hpp):
// plain structs, so that device code can use them.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// The capacities of the variable-size arrays of a batch, for the capacity form: how many entities (the entries of the
/// entity arrays), tuples, tuple arguments (the entries of relation_args), targets (the entries of target_entity and
/// target_group) and history steps (the entries of history_entity and history_dt) the caller's buffers hold. The arrays
/// whose size follows from the number of graphs and relations are always large enough by contract (see FlatBuffers).
struct FlatCapacities
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
    u64 history = 0;
};

/// Flags of FlatSizes::overflow.
inline constexpr u32 k_overflow_capacity = 1;  // some size exceeds its capacity
inline constexpr u32 k_overflow_index = 2;     // some size or count does not fit the index type
inline constexpr u32 k_overflow_input = 4;     // a row is invalid: a task id outside the table, a state row or goal mask
                                               // narrower than its instance, or a bit its instance does not have. Device
                                               // calls report it here, host calls throw

/// The true sizes of a batch (not clipped to any capacity) and whether the batch can be written. A POD, so that a
/// device call can return it in the same form.
struct FlatSizes
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
    u64 history = 0;
    u32 overflow = 0;  // k_overflow_* flags; 0: the batch fits
    [[nodiscard]] bool overflowed() const noexcept { return overflow != 0; }
};

/// The destination arrays of a batch of G graphs and R relations, typed by the index type of the options (int32 or
/// int64). The caller owns them. Arrays of size G, G + 1, G * R or G * R + 1 must have that size; relation_args,
/// entity_role, entity_ref, target_entity, target_group, history_entity and history_dt must hold at least the sizes of
/// the batch (FlatSizes). entity_role and entity_ref must be null iff the schema has implicit entity arrays,
/// relation_counts iff it does not store counts, goal_complete iff it has no goal relations, the three target arrays
/// iff it has no targets, the three history arrays iff it has no history. Nothing is written past the sizes: the
/// entries of the variable-size arrays beyond them keep their old values.
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
};
}
