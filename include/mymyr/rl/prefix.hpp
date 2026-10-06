#pragma once
// Factored action masks from the labels of an expansion: a lifted action is decided schema first, then one object per
// parameter, and the mask of a decision is exact: object o is valid for parameter `depth` after the prefix
// (schema, o_0 .. o_{depth-1}) iff some successor of the state is labelled (schema, o_0 .. o_{depth-1}, o, ...).
// The labels are those of rl::expand (schema, then the schema's
// full parameter list bound to objects, -1 past the arity), so the masks equal the ones computed from the expanded
// actions one by one.
//
// A mask is a scatter of the matching rows: mask[parent[j], binding[j, depth]] = 1 for every successor row j whose label
// extends the prefix of its state. Writes of the same value in any order give the same bytes, so the host loop and the
// device kernel (cuda/rl_ops.hpp) agree bit for bit without ordering. A padded expansion is a flat one whose row j of
// state i has parent i (and parent -1 for padding).

#include "mymyr/core/types.hpp"

namespace mymyr::rl
{
/// The labels of an expansion: `size` successor rows.
struct LabelRows
{
    u64 size = 0;
    const i32* parent = nullptr;   // [size]: the state of the row (-1: padding, skipped)
    const i32* schema = nullptr;   // [size]
    const i32* binding = nullptr;  // [size, label_width]
    u32 label_width = 0;
};

/// The decision of every state: its chosen schema and the objects of its first `depth` parameters.
struct PrefixQuery
{
    u64 rows = 0;                 // the expanded states (N)
    const i32* schema = nullptr;  // [rows]
    const i32* prefix = nullptr;  // [rows, prefix_stride] (read up to depth; may be null for depth 0)
    u32 prefix_stride = 0;
    u32 depth = 0;                // the parameter whose mask is computed (0 = the first)
    u32 num_objects = 0;          // mask columns
};

/// The (state, object) row j sets in the mask, or false when j does not extend its state's prefix (or the object is
/// outside [0, num_objects)).
MYMYR_HD bool prefix_hit(const LabelRows& x, const PrefixQuery& q, u64 j, u64& row, u32& object)
{
    const i32 p = x.parent[j];
    if (p < 0 || static_cast<u64>(p) >= q.rows || q.depth >= x.label_width)
        return false;
    if (x.schema[j] != q.schema[p])
        return false;
    const i32* b = x.binding + j * x.label_width;
    const i32* pre = q.prefix + static_cast<u64>(p) * q.prefix_stride;
    for (u32 d = 0; d < q.depth; ++d)
        if (b[d] != pre[d])
            return false;
    const i32 o = b[q.depth];
    if (o < 0 || static_cast<u32>(o) >= q.num_objects)
        return false;
    row = static_cast<u64>(p);
    object = static_cast<u32>(o);
    return true;
}

/// The schema a row sets in the schema mask (the first decision), or false for padding.
MYMYR_HD bool schema_hit(const LabelRows& x, u64 rows, u32 num_schemas, u64 j, u64& row, u32& schema)
{
    const i32 p = x.parent[j];
    const i32 s = x.schema[j];
    if (p < 0 || static_cast<u64>(p) >= rows || s < 0 || static_cast<u32>(s) >= num_schemas)
        return false;
    row = static_cast<u64>(p);
    schema = static_cast<u32>(s);
    return true;
}

/// mask [q.rows, q.num_objects] (bytes 0 / 1): the objects valid for parameter q.depth after each state's prefix.
void prefix_masks(const LabelRows& x, const PrefixQuery& q, u8* mask);
/// mask [rows, num_schemas]: the schemas with at least one successor of the state.
void schema_masks(const LabelRows& x, u64 rows, u32 num_schemas, u8* mask);
}  // namespace mymyr::rl
