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
