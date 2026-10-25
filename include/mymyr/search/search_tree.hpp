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
struct SearchTree
{
    u32 width = 0, words = 0, numeric_words = 0, label_width = 0;
    std::vector<u64> rows;
    std::vector<i32> parent, depth, schema, binding;
    std::vector<u8> node_flags;
    std::vector<i64> offsets;
    std::vector<i32> t_schema, t_binding, t_row, t_child;
    std::vector<u8> t_outcome, row_goal;

    [[nodiscard]] usize num_nodes() const noexcept { return parent.size(); }
    [[nodiscard]] usize num_transitions() const noexcept { return t_row.size(); }
};
}  // namespace mymyr::search
