#pragma once

// The cells of a horizon batch. A horizon graph is a root state and the nodes below it, all of one instance; its
// relations hold facts of the nodes (anchored by a state entity), goal literals of the root, the satisfaction of the
// goal at the nodes, what a node adds to or deletes from its root, and the edges between the nodes. The cells of a
// horizon instance are `Cell`s (lanes.hpp) whose op is a horizon op: an op of at least k_horizon_op_base, which no
// flat code reads, built from
//   - a kind: what the tuples are (HorizonKind);
//   - a scope: the nodes whose tuples the cell lists (HorizonScope);
//   - lead: whether each tuple starts with the entity of its state node;
//   - minus: the deleting variant of a delta cell, the falsifying variant of an added-satisfaction cell;
//   - low: the flat op that says where the tuple's atoms come from: a CellOp for the cells of facts and delta, the goal
//     op (goal_op, lanes.hpp) for the cells of goal literals.
// Cell::a and b are the slot range (or static rows) as in the flat cells, c and d the rows of the constant goal tuples.

#include "mymyr/encoding/detail/lanes.hpp"

namespace mymyr::encoding::detail
{
/// The role of a state entity and the group of the targets that are state entities (EntityRole::State and
/// TargetGroup::State in schema.hpp).
inline constexpr u8 k_role_state = 7;
inline constexpr u8 k_target_group_state = 4;

/// A graph whose edges are not valid: an endpoint outside the graph's nodes, a schema outside the table's, an object
/// outside the instance's.
inline constexpr u32 k_graph_bad_edge = 128u;

/// What the tuples of a horizon cell are.
enum class HorizonKind : u32
{
    Facts = 1,   // the atoms of a predicate at the nodes of the scope (low: Fluent, Derived or Static)
    Delta = 2,   // the atoms a node adds (deletes with `minus`) relative to the root (low: Fluent or Derived)
    Goal = 3,    // the plain goal literals of the root (low: a goal op of mode Plain)
    Sat = 4,     // the goal literals that hold / do not hold at the nodes of the scope (low: goal op, Satisfied / Unsatisfied)
    Added = 5,   // the goal literals a node newly satisfies (falsifies with `minus`): the literals whose atom the node adds or
                 // deletes (low: a goal op of mode Plain)
    Action = 6,  // one tuple per edge whose schema belongs to the relation: (destination, objects)
    Parent = 7,  // one tuple per edge: (source, destination)
    Sibling = 8, // the pairs of nodes that share a parent, both orders, ascending (kin_row)
    Cousin = 9,  // the pairs of nodes whose parents are siblings, both orders, ascending (kin_row)
};

/// Which nodes of a graph a cell lists: the root alone, the nodes below it, or all of them.
enum class HorizonScope : u32
{
    Root = 0,
    Nodes = 1,
    All = 2,
};

inline constexpr u32 k_horizon_op_base = 1u << 8;

MYMYR_HD constexpr u32 horizon_op(HorizonKind kind, HorizonScope scope, bool lead, bool minus, u32 low)
{
    return (static_cast<u32>(kind) << 8) | (static_cast<u32>(scope) << 12) | (lead ? 1u << 14 : 0u) |
           (minus ? 1u << 15 : 0u) | low;
}
MYMYR_HD constexpr bool is_horizon_op(u32 op) { return op >= k_horizon_op_base; }
MYMYR_HD constexpr HorizonKind horizon_kind(u32 op) { return static_cast<HorizonKind>((op >> 8) & 15u); }
MYMYR_HD constexpr HorizonScope horizon_scope(u32 op) { return static_cast<HorizonScope>((op >> 12) & 3u); }
MYMYR_HD constexpr bool horizon_lead(u32 op) { return ((op >> 14) & 1u) != 0; }
MYMYR_HD constexpr bool horizon_minus(u32 op) { return ((op >> 15) & 1u) != 0; }
MYMYR_HD constexpr u32 horizon_low(u32 op) { return op & 255u; }
}  // namespace mymyr::encoding::detail

namespace mymyr::encoding::detail
{
/// The kinship of the nodes of one horizon graph, over its distinct edges (parallel edges count once; an edge from a
/// node to itself makes the node its own parent). Node q is a child of p, and p a parent of q, iff there is an edge
/// p -> q. Two nodes are siblings iff they differ and have a common parent; the lists below are the data the relatives
/// of a node are read from:
///   - children(p) and parents(q): ascending, distinct;
///   - far(p): the children of the siblings of p, ascending, distinct. Node q is a cousin of node n iff q != n and q is in
///     far(p) for a parent p of n; the far lists exist only for cousin relations.
struct KinView
{
    u32 nodes = 0;
    const u32* child_offsets = nullptr;   // [nodes + 1]
    const u32* children = nullptr;
    const u32* parent_offsets = nullptr;  // [nodes + 1]
    const u32* parents = nullptr;
    const u32* far_offsets = nullptr;     // [nodes + 1], null without cousins
    const u32* far = nullptr;
};

/// The ascending list a relative of node `n` comes from through its parent `p`: the children of p for a sibling, far(p)
/// for a cousin.
MYMYR_HD const u32* kin_source(const KinView& k, bool cousin, u32 p, u32& size)
{
    const u32* offsets = cousin ? k.far_offsets : k.child_offsets;
    size = offsets[p + 1] - offsets[p];
    return (cousin ? k.far : k.children) + offsets[p];
}

/// The entries of the lists of node `n`'s parents: the room kin_row needs in `out` and in `work`.
MYMYR_HD u32 kin_total(const KinView& k, bool cousin, u32 n)
{
    u32 total = 0;
    for (u32 i = k.parent_offsets[n]; i < k.parent_offsets[n + 1]; ++i)
    {
        u32 size;
        kin_source(k, cousin, k.parents[i], size);
        total += size;
    }
    return total;
}

MYMYR_HD bool kin_contains(const u32* list, u32 size, u32 value)
{
    u32 lo = 0, hi = size;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2u;
        if (list[mid] < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < size && list[lo] == value;
}

/// Sorts list[0, n) ascending with the same room in `work`.
MYMYR_HD void kin_sort(u32* list, u32 n, u32* work)
{
    u32* from = list;
    u32* to = work;
    for (u32 width = 1; width < n; width *= 2)
    {
        for (u32 lo = 0; lo < n; lo += 2 * width)
        {
            const u32 mid = lo + width < n ? lo + width : n, hi = lo + 2 * width < n ? lo + 2 * width : n;
            u32 a = lo, b = mid, o = lo;
            while (a < mid && b < hi)
                to[o++] = from[b] < from[a] ? from[b++] : from[a++];
            while (a < mid)
                to[o++] = from[a++];
            while (b < hi)
                to[o++] = from[b++];
        }
        u32* swap = from;
        from = to;
        to = swap;
    }
    if (from != list)
        for (u32 i = 0; i < n; ++i)
            list[i] = from[i];
}

/// The relatives of node `n`: its siblings (or cousins), ascending, distinct, without `n` itself and, with `skip_zero`,
/// without node 0. They are written to `out` and counted; `out` and `work` hold kin_total entries each. The tuples of
/// a node in the relation are these relatives in order, so a node's tuples start after the tuples of the nodes
/// before it.
MYMYR_HD u32 kin_row(const KinView& k, bool cousin, u32 n, bool skip_zero, u32* out, u32* work)
{
    const u32 first = k.parent_offsets[n], last = k.parent_offsets[n + 1];
    u32 count = 0;
    if (last == first)
        return 0;
    if (last - first == 1)
    {
        u32 size;
        const u32* list = kin_source(k, cousin, k.parents[first], size);
        for (u32 i = 0; i < size; ++i)
            if (list[i] != n && !(skip_zero && list[i] == 0))
                out[count++] = list[i];
        return count;
    }
    u32 total = 0;
    for (u32 i = first; i < last; ++i)
    {
        u32 size;
        const u32* list = kin_source(k, cousin, k.parents[i], size);
        for (u32 j = 0; j < size; ++j)
            out[total++] = list[j];
    }
    kin_sort(out, total, work);
    for (u32 i = 0; i < total; ++i)
        if (out[i] != n && !(skip_zero && out[i] == 0) && (count == 0 || out[count - 1] != out[i]))
            out[count++] = out[i];
    return count;
}

/// The number of relatives of node `n` (kin_row's count) without writing them where one parent makes that possible.
MYMYR_HD u32 kin_count(const KinView& k, bool cousin, u32 n, bool skip_zero, u32* out, u32* work)
{
    const u32 first = k.parent_offsets[n], last = k.parent_offsets[n + 1];
    if (last - first != 1)
        return kin_row(k, cousin, n, skip_zero, out, work);
    u32 size;
    const u32* list = kin_source(k, cousin, k.parents[first], size);
    u32 count = size;
    if (kin_contains(list, size, n))
        --count;
    if (skip_zero && n != 0 && size != 0 && list[0] == 0)
        --count;
    return count;
}
}  // namespace mymyr::encoding::detail
