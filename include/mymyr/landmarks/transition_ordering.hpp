#pragma once
// The landmark transition ordering, matching mimir 0.16.3's LandmarkTransitionOrderingStrategy: a
// search::TransitionOrdering for IW's width-1 pass.
//
//   auto graph = mymyr::landmarks::approximate_fact_landmarks(*task);
//   mymyr::search::IwOptions o;
//   o.max_arity = 1;
//   o.transition_ordering = std::make_shared<mymyr::landmarks::LandmarkTransitionOrdering>(*task, graph);
//   auto r = mymyr::search::iw(*task, o);
//
// Every transition parent --a--> child of a layer is scored against the fact landmarks: the landmarks the child holds
// and the parent does not (new), those among them whose unique achiever is a (new unique), whether a uniquely achieves
// some landmark at all (an action-level property), and the landmarks the parent holds and the child does not
// (deleted). The layer is then stable-sorted by the lexicographic preference: more new, more new unique, a unique
// achiever action first, fewer deleted; generation order breaks the remaining ties. Each option flag skips its key.
// "Unique achiever" is the achiever index's: the only ground action adding the atom, so the graph needs an achiever
// index (the approximate generator's; the lifted generator has none).

#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/search/transition_ordering.hpp"

namespace mymyr::landmarks
{
struct LandmarkTransitionOrderingOptions
{
    bool prefer_new_landmarks = true;
    bool prefer_unique_achievers = true;
    bool prefer_landmark_actions_when_deleting = true;
    bool prefer_fewer_deleted_landmarks = true;
};

struct LandmarkTransitionScore
{
    u32 num_new_landmarks = 0;
    u32 num_new_unique_landmarks = 0;
    bool unique_achiever_action = false;
    bool deletes_achieved_landmark = false;
    u32 num_deleted_achieved_landmarks = 0;

    friend bool operator==(const LandmarkTransitionScore&, const LandmarkTransitionScore&) = default;
};

class LandmarkTransitionOrdering final : public search::TransitionOrdering
{
public:
    /// Interns the landmark atoms of `task` (their state slots). Throws std::invalid_argument when the graph has no
    /// achiever index.
    LandmarkTransitionOrdering(const Task& task, FactLandmarkGraph graph, LandmarkTransitionOrderingOptions options = {});

    [[nodiscard]] LandmarkTransitionScore score(StateView parent, const ActionLabel& action, StateView child) const;
    /// Whether a sorts strictly before b.
    [[nodiscard]] bool prefer(const LandmarkTransitionScore& a, const LandmarkTransitionScore& b) const;

    void order(const Task& task, std::span<const search::LayerTransition> layer, std::vector<u32>& order) const override;

    [[nodiscard]] const FactLandmarkGraph& graph() const noexcept { return m_graph; }
    [[nodiscard]] const LandmarkTransitionOrderingOptions& options() const noexcept { return m_options; }

private:
    FactLandmarkGraph m_graph;
    LandmarkTransitionOrderingOptions m_options;
    std::vector<SlotId> m_slots;  // per fact landmark
    std::vector<u32> m_unique;    // per fact landmark: its unique achiever (action id) or k_no_action
};
}  // namespace mymyr::landmarks
