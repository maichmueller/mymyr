// The landmark transition ordering (landmarks/transition_ordering.hpp).

#include "mymyr/landmarks/transition_ordering.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mymyr::landmarks
{
LandmarkTransitionOrdering::LandmarkTransitionOrdering(const Task& task, FactLandmarkGraph graph, LandmarkTransitionOrderingOptions options)
    : m_graph(std::move(graph)), m_options(options)
{
    if (!m_graph.has_achiever_index())
        throw std::invalid_argument("mymyr: the landmark transition ordering needs a landmark graph with an achiever index "
                                    "(the approximate generator's)");
    m_slots = FactLandmarkGraph::slots(task, m_graph.landmarks());
    for (CanonicalAtom c : m_graph.landmarks())
        m_unique.push_back(m_graph.unique_achiever(c));
}

LandmarkTransitionScore LandmarkTransitionOrdering::score(StateView parent, const ActionLabel& action, StateView child) const
{
    LandmarkTransitionScore s;
    const u32 id = m_graph.achievers().find(action);
    for (usize i = 0; i < m_slots.size(); ++i)
    {
        const bool in_parent = parent.contains(m_slots[i]), in_child = child.contains(m_slots[i]);
        if (in_child && !in_parent)
        {
            ++s.num_new_landmarks;
            if (id != k_no_action && m_unique[i] == id)
                ++s.num_new_unique_landmarks;
        }
        else if (in_parent && !in_child)
            ++s.num_deleted_achieved_landmarks;
    }
    s.unique_achiever_action = id != k_no_action && m_graph.is_unique_landmark_achiever(id);
    s.deletes_achieved_landmark = s.num_deleted_achieved_landmarks > 0;
    return s;
}

bool LandmarkTransitionOrdering::prefer(const LandmarkTransitionScore& a, const LandmarkTransitionScore& b) const
{
    if (m_options.prefer_new_landmarks && a.num_new_landmarks != b.num_new_landmarks)
        return a.num_new_landmarks > b.num_new_landmarks;
    if (m_options.prefer_unique_achievers && a.num_new_unique_landmarks != b.num_new_unique_landmarks)
        return a.num_new_unique_landmarks > b.num_new_unique_landmarks;
    if (m_options.prefer_landmark_actions_when_deleting && a.unique_achiever_action != b.unique_achiever_action)
        return a.unique_achiever_action;
    if (m_options.prefer_fewer_deleted_landmarks && a.num_deleted_achieved_landmarks != b.num_deleted_achieved_landmarks)
        return a.num_deleted_achieved_landmarks < b.num_deleted_achieved_landmarks;
    return false;
}

void LandmarkTransitionOrdering::order(const Task&, std::span<const search::LayerTransition> layer, std::vector<u32>& order) const
{
    std::vector<LandmarkTransitionScore> scores;
    scores.reserve(layer.size());
    for (const search::LayerTransition& t : layer)
        scores.push_back(score(t.parent, t.action, t.child));
    order.resize(layer.size());
    std::iota(order.begin(), order.end(), 0u);
    // generation order breaks ties (stable)
    std::stable_sort(order.begin(), order.end(), [&](u32 x, u32 y) { return prefer(scores[x], scores[y]); });
}
}  // namespace mymyr::landmarks
