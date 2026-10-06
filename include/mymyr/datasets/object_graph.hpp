#pragma once
// Object graphs, matching mimir's `create_object_graph`: a vertex-coloured undirected graph of a state, the input of
// the certificates (datasets/certificates.hpp) that symmetry pruning uses. It is a structure for isomorphism tests,
// not an observation encoder; mymyr ships no encoders.
//
// Vertices, in this order (as in mimir):
//   1. one vertex per object (vertex i = object i). Its colour is the sorted list of the predicates of the true unary
//      atoms of the object (static atoms of the problem, fluent and derived atoms of the state), plus the sorted list
//      of (predicate, polarity) of the unary goal literals over it;
//   2. per true atom of arity 0 (static, then fluent, then derived): one vertex coloured (predicate); per true atom of
//      arity k >= 2: k vertices coloured (predicate, position), vertex j joined to object j of the atom and to vertex
//      j - 1 of the atom. Unary atoms have no vertex (they colour their object);
//   3. per goal literal (static, then fluent, then derived; numeric goal constraints are ignored): arity 0, one vertex
//      coloured (predicate, polarity); arity k >= 2, k vertices coloured (predicate, position, polarity), joined as
//      above.
// Colours: mimir compares them as C++ tuples of different types (ordered across types by typeid, which is not
// portable). mymyr encodes each colour as an integer sequence, tagged by its kind:
//   object [0, n, p_1..p_n, m, q_1, s_1, .., q_m, s_m]   atom [1, p]   atom position [2, p, j]
//   literal [3, p, s]   literal position [4, p, j, s]     (p, q: predicate ids, s: 1 positive / 0 negative)
// Predicate ids are the task's (TaskData order: the domain's predicates first, so they agree across the problems of
// one domain). The palette holds the graph's distinct colours in lexicographic order; color[v] indexes it. Equal
// colours are equal sequences, so colours compare across graphs through their palette entries (or color keys).

#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <span>
#include <vector>

namespace mymyr::datasets
{
struct ObjectGraph
{
    u32 num_objects = 0;
    /// Per vertex: its colour's index in the palette.
    std::vector<u32> color;
    /// The distinct colours as integer sequences (see above), sorted: colour c is palette_values[palette_offsets[c]
    /// .. palette_offsets[c + 1]).
    std::vector<u32> palette_offsets{0};
    std::vector<u32> palette_values;
    /// Undirected adjacency as CSR: the neighbours of v are neighbors[offsets[v] .. offsets[v + 1]), ascending. Every
    /// edge appears in both directions; these graphs have no parallel edges and no loops.
    std::vector<u64> offsets{0};
    std::vector<u32> neighbors;

    [[nodiscard]] u32 num_vertices() const noexcept { return static_cast<u32>(color.size()); }
    [[nodiscard]] u64 num_edges() const noexcept { return neighbors.size() / 2; }  // undirected edges
    [[nodiscard]] u32 num_colors() const noexcept { return static_cast<u32>(palette_offsets.size() - 1); }
    [[nodiscard]] std::span<const u32> palette(u32 c) const noexcept
    {
        return {palette_values.data() + palette_offsets[c], palette_offsets[c + 1] - palette_offsets[c]};
    }
    [[nodiscard]] std::span<const u32> adjacent(u32 v) const noexcept
    {
        return {neighbors.data() + offsets[v], static_cast<usize>(offsets[v + 1] - offsets[v])};
    }
    /// A 64-bit key of colour c's sequence (equal sequences give equal keys in every graph and process).
    [[nodiscard]] u64 color_key(u32 c) const noexcept;
};

/// Builds object graphs of one task's states. Holds per-task tables and scratch: one builder per thread.
class ObjectGraphBuilder
{
public:
    explicit ObjectGraphBuilder(const Task& task);
    ~ObjectGraphBuilder();
    ObjectGraphBuilder(const ObjectGraphBuilder&) = delete;
    ObjectGraphBuilder& operator=(const ObjectGraphBuilder&) = delete;

    /// The object graph of state s (derived atoms are evaluated when the task has axioms). Uses this thread's
    /// workspace of the task.
    [[nodiscard]] ObjectGraph build(StateView s);
    void build(StateView s, ObjectGraph& out);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

/// Convenience: one object graph.
[[nodiscard]] ObjectGraph object_graph(const Task& task, StateView s);
}  // namespace mymyr::datasets
