#pragma once
// Tuple graphs (Lipovetzky and Geffner, "Width and Serialization of Classical Planning Problems", ECAI 2012), as
// mimir's TupleGraphImpl (mimir 0.16.3) defines them: the tuple graph of a state-space vertex r describes which atom
// tuples of size at most `width` are first reached at each breadth-first distance from r, and by which shortest paths.
//
//   auto graphs = mymyr::datasets::tuple_graphs(space, {.width = 2, .threads = 8});   // one per vertex
//   const TupleGraph& g = graphs[r];
//   for (u32 d = 0; d < g.num_distances(); ++d)
//       for (u32 v : g.vertices_at(d)) { g.tuple(v); g.problem_vertices(v); g.successors(v); }
//
// Definition (width w >= 1). The problem vertices at distance d are the state-space vertices at breadth-first distance
// d from r. A tuple (a set of 0..w fluent atoms) is novel at distance d in a state at distance d if no state at a
// smaller distance contains it; its problem vertices at d are the states at distance d in which it is novel (a tuple
// is novel at one distance only). The vertices at distance 0 are the root's tuples (every tuple of size <= w of r's
// state, the empty one included); at distance d >= 1, a novel tuple t becomes a vertex iff some vertex u at d - 1
// "extends" to it: every problem vertex of u has a successor among the problem vertices of t (every optimal plan for
// u's tuple extends by one action to an optimal plan for t). There is an edge u -> t for every such u.
// Construction stops at the first distance with no state, no novel tuple or no extended tuple.
//
// Dominance pruning (default on): of the extended tuples of a distance, only those whose problem vertices are
// minimal under strict inclusion are kept, and of tuples with equal problem vertices only one. At distance 0 a single
// vertex is kept (every root tuple has the problem vertices {r}). The tuple kept for a set of problem vertices is the
// smallest in the canonical tuple order: fewer atoms first, then the lexicographic order of the sorted atom names (so
// the empty tuple at distance 0). mimir keeps the first in an unspecified hash-set order instead; the vertices, edges
// and problem vertices are the same, and its tuple has the same problem vertices as the one kept here.
//
// Width 0: the root (empty tuple, problem vertices {r}) at distance 0, and at distance 1 one vertex per successor
// state of r other than r (empty tuple, that state as its only problem vertex), each with an edge from the root.
// Deviations from mimir, which are bugs there: mimir adds one distance-1 vertex per transition (parallel transitions
// to one state give equal vertices) and leaves r out of the problem vertices at distance 0.
//
// Vertex order: by distance, and within a distance by the canonical order of their tuples. Tuples are fluent atom
// slots in ascending slot order; problem vertices and edges are ascending. The result is a function of the state
// space (and, for the tuple kept by dominance pruning, of the atom names): it does not depend on the thread count.
//
// Cost: one breadth-first search over (at most) the whole space per vertex, enumerating the C(atoms, <= w) tuples of
// every reached state. tuple_graphs() runs the vertices in parallel, one search per thread at a time.

#include "mymyr/core/types.hpp"
#include "mymyr/datasets/state_space.hpp"

#include <ranges>
#include <span>
#include <vector>

namespace mymyr::datasets
{
struct TupleGraphOptions
{
    u32 width = 0;                  // the largest tuple size (0..novelty::k_max_arity)
    bool dominance_pruning = true;  // keep the minimal problem-vertex sets only (mimir's enable_dominance_pruning)
    u32 threads = 1;                // tuple_graphs(): worker threads (0: std::thread::hardware_concurrency())
};

class TupleGraph
{
public:
    /// The state space and its vertex this graph is of, and the options it was built with.
    [[nodiscard]] const StateSpacePtr& space() const noexcept { return m_space; }
    [[nodiscard]] u32 root() const noexcept { return m_root; }
    [[nodiscard]] u32 width() const noexcept { return m_width; }
    [[nodiscard]] bool dominance_pruning() const noexcept { return m_pruning; }

    [[nodiscard]] u32 num_vertices() const noexcept { return static_cast<u32>(m_tuple_offsets.size() - 1); }
    [[nodiscard]] u64 num_edges() const noexcept { return m_succ.size(); }
    /// Distances 0 .. num_distances() - 1 have vertices (at least distance 0, the root's).
    [[nodiscard]] u32 num_distances() const noexcept { return static_cast<u32>(m_distance_offsets.size() - 1); }
    /// [num_distances() + 1]: the vertices at distance d are distance_offsets()[d] .. distance_offsets()[d + 1] - 1.
    [[nodiscard]] std::span<const u32> distance_offsets() const noexcept { return m_distance_offsets; }
    [[nodiscard]] auto vertices_at(u32 d) const { return std::views::iota(m_distance_offsets.at(d), m_distance_offsets.at(d + 1)); }
    [[nodiscard]] u32 distance(u32 v) const;

    /// The tuple of vertex v: fluent atom slots of the space's task, ascending (empty for the empty tuple).
    [[nodiscard]] std::span<const u32> tuple(u32 v) const;
    /// The problem vertices of vertex v: the state-space vertices at its distance in which its tuple is novel,
    /// ascending.
    [[nodiscard]] std::span<const u32> problem_vertices(u32 v) const;
    /// The edges out of / into vertex v (to distance(v) + 1 / from distance(v) - 1), ascending.
    [[nodiscard]] std::span<const u32> successors(u32 v) const;
    [[nodiscard]] std::span<const u32> predecessors(u32 v) const;
    /// The state-space vertices at breadth-first distance d from the root (ascending), for d < num_distances().
    [[nodiscard]] std::span<const u32> problem_vertices_at(u32 d) const;

    /// The flat arrays behind the accessors: per vertex v, tuple(v) = tuple_atoms()[tuple_offsets()[v] ..
    /// tuple_offsets()[v + 1]), and likewise problem_vertices (problem_offsets(), problem_vertex_ids()), successors
    /// (successor_offsets(), successor_ids()) and predecessors; problem_vertices_at(d) is
    /// layer_vertex_ids()[layer_offsets()[d] .. layer_offsets()[d + 1]).
    [[nodiscard]] std::span<const u32> tuple_offsets() const noexcept { return m_tuple_offsets; }
    [[nodiscard]] std::span<const u32> tuple_atoms() const noexcept { return m_tuple_atoms; }
    [[nodiscard]] std::span<const u32> problem_offsets() const noexcept { return m_problem_offsets; }
    [[nodiscard]] std::span<const u32> problem_vertex_ids() const noexcept { return m_problem_vertices; }
    [[nodiscard]] std::span<const u32> successor_offsets() const noexcept { return m_succ_offsets; }
    [[nodiscard]] std::span<const u32> successor_ids() const noexcept { return m_succ; }
    [[nodiscard]] std::span<const u32> predecessor_offsets() const noexcept { return m_pred_offsets; }
    [[nodiscard]] std::span<const u32> predecessor_ids() const noexcept { return m_pred; }
    [[nodiscard]] std::span<const u32> layer_offsets() const noexcept { return m_layer_offsets; }
    [[nodiscard]] std::span<const u32> layer_vertex_ids() const noexcept { return m_layer_vertices; }

    /// Heap bytes of the arrays.
    [[nodiscard]] u64 bytes() const noexcept;

    friend bool operator==(const TupleGraph& a, const TupleGraph& b) noexcept;

private:
    friend class TupleGraphBuilder;

    StateSpacePtr m_space;
    u32 m_root = 0, m_width = 0;
    bool m_pruning = true;
    std::vector<u32> m_distance_offsets{0};
    std::vector<u32> m_tuple_offsets{0}, m_tuple_atoms;
    std::vector<u32> m_problem_offsets{0}, m_problem_vertices;
    std::vector<u32> m_succ_offsets{0}, m_succ, m_pred_offsets{0}, m_pred;
    std::vector<u32> m_layer_offsets{0}, m_layer_vertices;
};

/// The tuple graph of every vertex of `space` (index = vertex), built on options.threads threads. Throws
/// std::invalid_argument for a width above novelty::k_max_arity.
[[nodiscard]] std::vector<TupleGraph> tuple_graphs(const StateSpacePtr& space, const TupleGraphOptions& options = {});
/// The tuple graph of one vertex (options.threads is not used). Throws std::invalid_argument for a width above
/// novelty::k_max_arity or a vertex outside the space.
[[nodiscard]] TupleGraph tuple_graph(const StateSpacePtr& space, u32 vertex, const TupleGraphOptions& options = {});
}  // namespace mymyr::datasets
