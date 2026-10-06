#pragma once
// Generalized state spaces, matching mimir's `GeneralizedStateSpaceImpl`: a class graph over the state spaces of
// several problems of one domain.
//   - Without symmetry reduction (the input spaces were generated without symmetry pruning): the class graph is the
//     disjoint union of the problem graphs, in input order: class vertex (problem p, vertex v) = offset[p] + v, and
//     class edge (p, e) = edge offset[p] + e. Initial, goal and unsolvable class vertices are the problems' ones.
//   - With symmetry reduction (every input space was generated with symmetry pruning): the certificate of each
//     problem vertex's state (the object graph certificate its space was pruned with) names its class. A problem whose
//     initial state's class exists already is dropped (isomorphic to an earlier one); otherwise each of its vertices
//     maps to its certificate's class (a new class vertex with this problem vertex as representative, flagged
//     initial / goal / unsolvable from it, if the certificate is new), and each edge to the class edge between the
//     classes of its ends (no parallel class edges). mimir stores the problem vertex index instead of the class
//     vertex index for new certificates (a bug that maps later problems' vertices to wrong classes); mymyr stores the
//     class vertex.
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
    /// Symmetry reduction is applied iff every space is symmetry reduced (as in mimir). The spaces must belong to
    /// one domain (equal domain names).
    [[nodiscard]] static std::shared_ptr<const GeneralizedStateSpace> create(std::vector<StateSpacePtr> spaces);

    /// The problems' spaces kept (with symmetry reduction, problems isomorphic to earlier ones are dropped).
    [[nodiscard]] const std::vector<StateSpacePtr>& spaces() const noexcept { return m_spaces; }
    [[nodiscard]] bool symmetry_reduced() const noexcept { return m_symmetric; }

    // class graph
    [[nodiscard]] u32 num_vertices() const noexcept { return static_cast<u32>(m_vproblem.size()); }
    [[nodiscard]] u64 num_edges() const noexcept { return m_esource.size(); }
    /// Per class vertex: the problem (index into spaces()) and the problem vertex of its representative.
    [[nodiscard]] std::span<const u32> vertex_problems() const noexcept { return m_vproblem; }
    [[nodiscard]] std::span<const u32> vertex_problem_vertices() const noexcept { return m_vvertex; }
    /// Per class edge, in insertion order: its ends, and the problem and problem edge of its representative.
    [[nodiscard]] std::span<const u32> edge_sources() const noexcept { return m_esource; }
    [[nodiscard]] std::span<const u32> edge_targets() const noexcept { return m_etarget; }
    [[nodiscard]] std::span<const u32> edge_problems() const noexcept { return m_eproblem; }
    [[nodiscard]] std::span<const u32> edge_problem_edges() const noexcept { return m_eedge; }
    /// Forward CSR over class vertices: the class edges out of v are forward_edges()[offsets[v] .. offsets[v + 1])
    /// (ascending edge indices).
    [[nodiscard]] std::span<const u64> forward_offsets() const noexcept { return m_foffsets; }
    [[nodiscard]] std::span<const u32> forward_edges() const noexcept { return m_fedges; }
    /// Flags per class vertex (0 or 1), and the flagged vertices in ascending order.
    [[nodiscard]] std::span<const u8> initial_flags() const noexcept { return m_initial; }
    [[nodiscard]] std::span<const u8> goal_flags() const noexcept { return m_goal; }
    [[nodiscard]] std::span<const u8> unsolvable_flags() const noexcept { return m_unsolvable; }
    [[nodiscard]] std::vector<u32> initial_vertices() const;
    [[nodiscard]] std::vector<u32> goal_vertices() const;
    [[nodiscard]] std::vector<u32> unsolvable_vertices() const;

    /// Per problem p (index into spaces()): the class vertex of each problem vertex and the class edge of each edge.
    [[nodiscard]] std::span<const u32> vertex_mapping(u32 p) const { return m_vmap.at(p); }
    [[nodiscard]] std::span<const u32> edge_mapping(u32 p) const { return m_emap.at(p); }

    GeneralizedStateSpace() = default;

private:
    std::vector<StateSpacePtr> m_spaces;
    bool m_symmetric = false;
    std::vector<u32> m_vproblem, m_vvertex;
    std::vector<u32> m_esource, m_etarget, m_eproblem, m_eedge;
    std::vector<u64> m_foffsets;
    std::vector<u32> m_fedges;
    std::vector<u8> m_initial, m_goal, m_unsolvable;
    std::vector<std::vector<u32>> m_vmap, m_emap;
};
using GeneralizedStateSpacePtr = std::shared_ptr<const GeneralizedStateSpace>;
}  // namespace mymyr::datasets
