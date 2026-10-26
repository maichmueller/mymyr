#pragma once

#include "mymyr/core/types.hpp"

#include <vector>

namespace mymyr::search
{
/// The amount of one IW pass retained in its result.
enum class TreeRecord : u8
{
    None,
    Nodes,
    Transitions,
};

/// Flags in SearchTree::node_flags. Expanded means successors were enumerated; Skipped means a counted
/// expansion did not enumerate successors. DuplicateEntry marks a node admitted through multiple queue entries.
enum class TreeNodeFlag : u8
{
    Expanded = 1,
    Goal = 2,
    Skipped = 4,
    Dropped = 8,
    DuplicateEntry = 16,
};

/// One IW pass as flat, owning arrays. Node ids are the observer's ids, root 0. Node i is row i; the remaining
/// distinct rows are non-node successors in first-generation order. Rows have stride words + numeric_words,
/// with numeric values stored as their state-row bits. Labels use task schema/object ids and -1 binding padding.
/// Transitions form CSR over nodes, in generation order within each node. A rejected transition has t_child = -1;
/// outcomes are TransitionOutcome values. Nodes-only recording leaves all transition arrays empty.
/// row_goal always uses the pass's own goal, including custom and serialized subproblem goals.
/// The flat layout allows host and device searches to populate the same result structure.
/// Shapes below: N = num_nodes(), T = num_transitions(), R = row_goal.size() distinct rows, L = label_width.
struct SearchTree
{
    u32 width = 0;          // the novelty width of the pass
    u32 words = 0;          // fluent words per row (the task's word width when the search ended)
    u32 numeric_words = 0;  // numeric words per row, after the fluent words
    u32 label_width = 0;    // L: the largest action arity of the task
    std::vector<u64> rows;          // [R, words + numeric_words]: nodes first, then other successors
    std::vector<i32> parent;        // [N]: parent node; -1 for the root
    std::vector<i32> depth;         // [N]: depth; 0 for the root
    std::vector<i32> schema;        // [N]: schema id of the incoming action; -1 for the root
    std::vector<i32> binding;       // [N, L]: object ids of the incoming action, -1 padded (all -1 for the root)
    std::vector<u8> node_flags;     // [N]: TreeNodeFlag bits
    std::vector<i64> offsets;       // [N + 1]: node i's transitions are [offsets[i], offsets[i + 1]); empty for Nodes
    std::vector<i32> t_schema;      // [T]: schema id of the transition's action
    std::vector<i32> t_binding;     // [T, L]: object ids of the transition's action, -1 padded
    std::vector<i32> t_row;         // [T]: row of the successor
    std::vector<i32> t_child;       // [T]: node the transition entered; -1 when the successor got no node
    std::vector<u8> t_outcome;      // [T]: TransitionOutcome
    std::vector<u8> row_goal;       // [R]: 1 when the row satisfies the pass's goal

    [[nodiscard]] usize num_nodes() const noexcept { return parent.size(); }
    [[nodiscard]] usize num_transitions() const noexcept { return t_row.size(); }
};
}  // namespace mymyr::search
