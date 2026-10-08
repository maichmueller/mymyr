#pragma once
// The per-thread state of symmetry pruning (successor/symmetry.hpp): the object graph of the prepared state, its
// colour refinement classes, and per schema the masks of the kept objects of every parameter. All scratch is sized
// once (graphs grow it to the largest state seen); computing the classes of a state allocates nothing after that.

#include "mymyr/core/types.hpp"

#include <span>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::detail
{
class Engine;

class SymmetryPruner
{
public:
    /// Per-task tables of `task` for an engine with `ow` object words per mask.
    SymmetryPruner(const Task& task, u32 ow);

    /// Computes the colour classes of the objects in the engine's current state. The state must be prepared
    /// (Successors::prepare: view built, derived atoms evaluated). Returns false when every object is alone in its class
    /// (pruning then keeps every action; refinement stops there, as it cannot split the objects further).
    [[nodiscard]] bool compute(const Engine& e);

    /// The colour class of every object (ids in [0, number of vertices)) of the state of the last compute(). Class ids
    /// name classes within one state only.
    [[nodiscard]] std::span<const u32> object_classes() const noexcept { return {m_color.data(), m_n}; }

    /// The kept objects of every parameter of `schema` in the state of the last compute(): arity * ow words, parameter
    /// i at [i * ow, (i + 1) * ow). Valid until the next call.
    [[nodiscard]] const u64* masks(u32 schema);

    /// The static domain of parameter `param` of `schema` (ascending object ids): the objects consistent with the
    /// schema's static precondition literals, as mimir's static consistency graph.
    [[nodiscard]] std::span<const u32> domain(u32 schema, u32 param) const noexcept;

private:
    void build_graph(const Engine& e);
    bool refine();

    u32 m_n = 0;   // objects
    u32 m_ow = 0;  // words per object mask
    u32 m_V = 0;   // vertices of the current graph
    bool m_axioms = false;
    // per schema and parameter: static domain (flat), offsets per (schema, parameter)
    std::vector<u32> m_dom, m_dom_begin, m_param_begin;
    // object colours: the class of the object's static unary atoms and unary goal literals (ranked)
    std::vector<u32> m_const_class;
    // vertex colour of an atom or goal literal vertex, before the offset by the number of object colours
    std::vector<u32> m_pred_code;  // per predicate: first code (positions follow)
    u32 m_num_codes = 0;
    std::vector<u32> m_arity;  // per predicate
    // static part of the graph: vertices n .. n + static_vertices - 1 and their edges
    std::vector<u32> m_static_code;
    std::vector<std::pair<u32, u32>> m_static_edges;
    // per-state scratch
    std::vector<std::pair<u32, u32>> m_unary;  // (object, predicate) of the true fluent / derived unary atoms
    std::vector<u32> m_unary_begin;            // per object + 1
    std::vector<u32> m_code;                   // per vertex (initial colour key of structure vertices)
    std::vector<std::pair<u32, u32>> m_edges;
    std::vector<u32> m_offsets, m_adj, m_fill;
    std::vector<u32> m_color, m_next, m_order, m_sig;
    std::vector<u32> m_work, m_round;  // classes to re-examine in this and the next round
    std::vector<u8> m_dirty;
    // per-schema scratch
    std::vector<u32> m_count, m_used, m_stamp;
    u32 m_tick = 0;
    std::vector<u64> m_masks;
};
}  // namespace mymyr::detail
