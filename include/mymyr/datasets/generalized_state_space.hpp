#pragma once
// Generalized state spaces, as mimir's `GeneralizedStateSpaceImpl`: one graph over the state spaces of several problems
// of one domain, their disjoint union in input order. Vertex v of problem p is vertex vertex_offsets()[p] + v, and edge
// e of problem p is edge edge_offsets()[p] + e, so the edges out of a vertex are its state's transitions in their
// forward order. The initial, goal and unsolvable vertices are the problems' ones.
// mimir's StateSpaceImpl::create(GeneralizedSearchContext, options) sorts the spaces by their number of states
// (its option sort_ascending_by_num_states) with std::sort, whose order of ties is unspecified; ordered_spaces() sorts
// stably (ties keep the input order), which is what mimir's libstdc++ std::sort does for up to 16 spaces.

#include "mymyr/core/types.hpp"
#include "mymyr/datasets/state_space.hpp"

#include <memory>
#include <span>
#include <vector>

namespace mymyr::datasets
{
/// The successful spaces of `results` (the failed ones are skipped, as mimir skips them), sorted ascending by their
/// number of states if `sort_ascending` (stable).
[[nodiscard]] std::vector<StateSpacePtr> ordered_spaces(std::span<const StateSpaceResult> results, bool sort_ascending = true);

class GeneralizedStateSpace
{
public:
    /// The disjoint union of `spaces`, in their order. Throws std::invalid_argument for a null space or spaces of
    /// different domains (by domain name), std::length_error for 2^32 vertices or more.
    [[nodiscard]] static std::shared_ptr<const GeneralizedStateSpace> create(std::vector<StateSpacePtr> spaces);

    /// The problems' state spaces (problem p = spaces()[p]).
    [[nodiscard]] const std::vector<StateSpacePtr>& spaces() const noexcept { return m_spaces; }
    [[nodiscard]] u32 num_vertices() const noexcept { return m_voffsets.back(); }
    [[nodiscard]] u64 num_edges() const noexcept { return m_eoffsets.back(); }

    /// [spaces().size() + 1]: the vertices of problem p are vertex_offsets()[p] .. vertex_offsets()[p + 1] - 1, its
    /// edges edge_offsets()[p] .. edge_offsets()[p + 1] - 1.
    [[nodiscard]] std::span<const u32> vertex_offsets() const noexcept { return m_voffsets; }
    [[nodiscard]] std::span<const u64> edge_offsets() const noexcept { return m_eoffsets; }
    /// The vertex of state `state` of problem `problem`, and the edge of its transition `edge`. Throw std::out_of_range
    /// for an index outside its problem or space.
    [[nodiscard]] u32 vertex(u32 problem, u32 state) const;
    [[nodiscard]] u64 edge(u32 problem, u64 edge) const;
    /// The problem of a vertex (binary search over vertex_offsets()); its state is vertex - vertex_offsets()[problem].
    /// Throws std::out_of_range for a vertex outside the graph.
    [[nodiscard]] u32 problem_of(u32 vertex) const;

    /// Forward CSR: the edges out of vertex v are forward_offsets()[v] .. forward_offsets()[v + 1] - 1, edge e goes to
    /// forward_targets()[e].
    [[nodiscard]] std::span<const u64> forward_offsets() const noexcept { return m_foffsets; }
    [[nodiscard]] std::span<const u32> forward_targets() const noexcept { return m_targets; }
    /// The vertex an edge leaves (binary search over forward_offsets()). Throws std::out_of_range for an edge outside
    /// the graph.
    [[nodiscard]] u32 source(u64 edge) const;

    /// Flags per vertex (0 or 1), and the flagged vertices in ascending order.
    [[nodiscard]] std::span<const u8> initial_flags() const noexcept { return m_initial; }
    [[nodiscard]] std::span<const u8> goal_flags() const noexcept { return m_goal; }
    [[nodiscard]] std::span<const u8> unsolvable_flags() const noexcept { return m_unsolvable; }
    [[nodiscard]] std::vector<u32> initial_vertices() const;
    [[nodiscard]] std::vector<u32> goal_vertices() const;
    [[nodiscard]] std::vector<u32> unsolvable_vertices() const;

    GeneralizedStateSpace() = default;

private:
    std::vector<StateSpacePtr> m_spaces;
    std::vector<u32> m_voffsets{0};
    std::vector<u64> m_eoffsets{0};
    std::vector<u64> m_foffsets{0};
    std::vector<u32> m_targets;
    std::vector<u8> m_initial, m_goal, m_unsolvable;
};
using GeneralizedStateSpacePtr = std::shared_ptr<const GeneralizedStateSpace>;
}  // namespace mymyr::datasets
