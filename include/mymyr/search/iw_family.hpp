#pragma once
// Option types shared by the IW family variants (search/aiw.hpp, liw.hpp, parallel_rollouts.hpp).

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/novelty/landmark_table.hpp"
#include "mymyr/search/layer_ordering.hpp"

#include <memory>
#include <vector>

namespace mymyr::search
{
/// Landmark novelty (LIW(k), abstracted LIW; novelty/landmark_table.hpp).
struct LandmarkNovelty
{
    /// The landmarks: fact landmarks give one rank each. Null: no landmarks (every state's coordinate is BOT).
    std::shared_ptr<const landmarks::FactLandmarkGraph> graph;
    /// Also rank the graph's disjunctive landmarks (mimir's landmark_novelty_disjunctive): each set shares a rank.
    bool disjunctive = false;
    /// With `disjunctive`: every disjunctive member gets its own rank (mimir's landmark_novelty_all_private).
    bool all_private = false;
    /// With `disjunctive`: atoms taken out of every shared set and given their own rank (mimir's
    /// landmark_novelty_unshared_atoms). Cannot be combined with all_private.
    std::vector<CanonicalAtom> unshared_atoms;
    novelty::LandmarkTableOptions tables;
};
}  // namespace mymyr::search
