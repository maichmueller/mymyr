#pragma once

// Horizon graphs: the nodes and edges that FlatEncoder encodes as horizons (HorizonInput), made from the successors of
// states.
//
//   - build_horizon_graphs expands each root breadth first to a depth. A node is a state; a state reached twice is one
//     node (merging by the words of its row, within a graph) unless merging is off, in which case every successor is
//     a node of its own. Nodes are numbered in the order they are first reached: the root is node 0, then, level by
//     level, the successors of the nodes of the level before in node order, each node's successors in the canonical
//     order of rl::expand (by schema, then by binding). An edge goes from a node to each successor of it; the edges of a
//     graph are ordered by destination node, then by schema, then by binding, and edges that agree keep the order of
//     their source nodes and of rl::expand. An edge to a state that is already a node leads to that node (a cycle, or
//     a node with several parents). The states of the deepest level are
//     not expanded.
//   - transition_graphs makes the depth-one graphs of states whose expansion exists already: node 0 is the state, node
//     k its k-th successor row and edge k-1 leads to it (one node per successor row, nothing merged: a successor that
//     equals its state is a node equal to node 0).
//   - the overload over a SuccessorGraph walks the transitions of a state space in place of expanding.
//
// The result does not depend on the pool or its size: expansion is deterministic, and merging and numbering are done
// in a fixed order.
//
// Memory. All the graphs of a call are built at once and held in memory. While they are built every node takes
// 8 * words + 8 bytes (its row, its graph and number) and, when merging, 8 more for its hash and 8 to 16 for the hash
// table; every edge takes 4 * (4 + label_width) bytes. The result (8 * words bytes per node, 4 * (3 + label_width) per
// edge, plus 4 bytes per distinct parent pair and 4 * (graphs + 1) for its offsets) is written from them, so the peak is
// about the sum of the two. HorizonBuild::depth bounds the nodes of a graph by
// the number of states within that many actions of the root, HorizonBuild::max_nodes by a number of your choice (the
// call refuses a graph above it, or stops adding nodes to it with `truncate`). Expansion runs in chunks of at most
// k_horizon_chunk states, so the working memory beyond the above is bounded by the successors of a chunk.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/rl/expand.hpp"

#include <limits>
#include <span>
#include <vector>

namespace mymyr::encoding
{
/// States expanded by one call of rl::expand.
inline constexpr u64 k_horizon_chunk = 4096;

struct HorizonBuild
{
    /// Levels below the root; 0 gives root-only graphs.
    u32 depth = 1;
    /// The most nodes of a graph, the root included.
    u64 max_nodes = std::numeric_limits<u64>::max();
    /// A graph that would exceed max_nodes stops growing (HorizonGraphs::truncated says which) instead of the call
    /// failing: a successor that is not yet a node is left out together with its edge.
    bool truncate = false;
    /// One node per distinct state of a graph (otherwise one per successor row, so the graph is a tree).
    bool merge = true;
};

/// The arrays of G horizon graphs. input() is the HorizonInput over them.
struct HorizonGraphs
{
    u32 words = 0;                   // atom words per node row
    u32 label_width = 0;             // columns of edge_binding
    std::vector<i32> task_ids;       // [G] the instance of each graph
    std::vector<u64> nodes;          // [M, words] graph after graph, the root first
    std::vector<i32> node_offsets{0};  // [G + 1]
    std::vector<i32> edge_offsets{0};  // [G + 1]
    std::vector<i32> edge_src, edge_dst, edge_schema;  // [E] graph-local nodes
    std::vector<i32> edge_binding;   // [E, label_width], -1 past a schema's arity
    std::vector<u8> truncated;       // [G] whether the graph was cut at max_nodes
    /// The derived words of the node rows [M, derived_words]; empty when not known (transition_graphs with a
    /// derived-word input).
    std::vector<u64> derived;
    u32 derived_words = 0;
    std::vector<i32> parent_offsets{0};  // [G + 1] canonical parent pairs of each graph
    std::vector<i32> parent_edges;       // graph-local edge indices, ascending by (source, destination), distinct pairs

    [[nodiscard]] u64 graphs() const noexcept { return task_ids.size(); }
    [[nodiscard]] u64 num_nodes() const noexcept { return words ? nodes.size() / words : 0; }
    [[nodiscard]] u64 num_edges() const noexcept { return edge_src.size(); }
    [[nodiscard]] HorizonInput input() const noexcept
    {
        HorizonInput h;
        h.nodes = {nodes.data(), num_nodes(), words, 0, 0};
        h.graphs = graphs();
        h.node_offsets = node_offsets.data();
        h.edge_offsets = edge_offsets.data();
        h.edge_src = edge_src.data();
        h.edge_dst = edge_dst.data();
        h.edge_schema = edge_schema.data();
        h.edge_binding = edge_binding.data();
        h.label_width = label_width;
        if (parent_offsets.size() == graphs() + 1 && static_cast<u64>(parent_offsets.back()) == parent_edges.size())
        {
            h.parent_offsets = parent_offsets.data();
            h.parent_edges = parent_edges.data();
            h.parent_pairs = parent_edges.size();
        }
        return h;
    }
};

/// The horizon graphs of `roots` (row g of instance task_ids[g]; null: instance 0). Throws std::invalid_argument
/// ("mymyr: encoding: ...") for invalid rows or task ids, a graph above max_nodes (without truncate), and
/// std::length_error when the graphs would hold more than 2^31 - 1 nodes or edges.
[[nodiscard]] HorizonGraphs build_horizon_graphs(const rl::TaskTable& table, rl::StateBatchView roots,
                                                 const i32* task_ids, const HorizonBuild& options);
[[nodiscard]] HorizonGraphs build_horizon_graphs(const rl::TaskTable& table, rl::StateBatchView roots,
                                                 const i32* task_ids, const HorizonBuild& options, ThreadPool& pool,
                                                 rl::ExpandScratch& scratch);

/// The depth-one graphs of `parents` from their expansion (rl::expand over the same rows, with schema, binding and
/// offsets; it must not have overflowed). With `parent_derived` ([rows, parent_derived_stride] words) and an expansion
/// that holds derived words, the result's `derived` holds the derived words of every node row.
[[nodiscard]] HorizonGraphs transition_graphs(const rl::TaskTable& table, rl::StateBatchView parents,
                                              const i32* task_ids, const rl::Expansion& expansion,
                                              const u64* parent_derived = nullptr, u64 parent_derived_stride = 0);

/// The transitions of a state space of one instance: state s has the successors targets[offsets[s]] ... in order, edge e
/// by schema schemas[e] and the objects bindings[e * label_width ...] (an unused position holds ~0u).
struct SuccessorGraph
{
    u64 states = 0;
    u32 words = 0;                     // atom words of a state row
    const u64* state_words = nullptr;  // [states, row_stride]
    u64 row_stride = 0;
    const u64* offsets = nullptr;      // [states + 1]
    const u32* targets = nullptr;
    const u32* schemas = nullptr;
    const u32* bindings = nullptr;
    u32 label_width = 0;
};

/// The horizon graphs of the states `roots` (ids) of `space`, all of instance `task_id`, by the same rules as
/// build_horizon_graphs.
[[nodiscard]] HorizonGraphs build_horizon_graphs(const SuccessorGraph& space, std::span<const u32> roots, i32 task_id,
                                                 const HorizonBuild& options);
}  // namespace mymyr::encoding
