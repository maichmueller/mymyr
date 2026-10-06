#pragma once
// Reordering the next layer of a breadth-first pass by a LayerOrdering (search/layer_ordering.hpp): shared by
// search::iw (iw.cpp), search::brfs (brfs.cpp) and the engine of the IW family variants (novelty_brfs.hpp).

#include "iw_detail.hpp"

#include "mymyr/core/random.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/successor/successors.hpp"

#include <algorithm>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::search::detail
{
/// The ordering state of one search: its kind, the generator (Randomized; it persists across layers and passes)
/// and the goal literals (GoalCount).
class LayerOrderer
{
public:
    LayerOrderer() = default;
    LayerOrderer(const Task& task, const LayerOrdering& lo)
        : m_kind(lo.kind), m_prefer_more(lo.prefer_more_satisfied_goals), m_limit(lo.max_next_layer_states), m_rng(lo.seed)
    {
        if (lo.kind == LayerOrdering::Kind::GoalCount)
            m_goal = GoalTest::counter(task, 0);
    }

    /// Layer by layer (every kind but Queue).
    [[nodiscard]] bool ordered() const noexcept { return m_kind != LayerOrdering::Kind::Queue; }
    /// max_next_layer_states (~0: no limit).
    [[nodiscard]] u32 limit() const noexcept { return m_limit; }
    [[nodiscard]] bool limited() const noexcept { return ordered() && m_limit != ~u32{0}; }

    /// Reorders a next layer of entries (node or state ids). state_of(entry) -> StateView gives an entry's state
    /// (GoalCount scores it; the engine of `succ` is pointed at it, so call this between expansions).
    template<class StateOf>
    void order(std::span<u32> layer, Successors& succ, StateOf&& state_of)
    {
        switch (m_kind)
        {
            case LayerOrdering::Kind::Queue:
            case LayerOrdering::Kind::InOrder: return;
            case LayerOrdering::Kind::Reverse: std::reverse(layer.begin(), layer.end()); return;
            case LayerOrdering::Kind::Randomized: m_rng.shuffle(layer); return;
            case LayerOrdering::Kind::GoalCount:
            {
                // the literal count is fixed: more satisfied literals = fewer unsatisfied ones
                m_scored.clear();
                for (u32 e : layer)
                    m_scored.emplace_back(m_goal.unsatisfied(succ, state_of(e), false), e);
                if (m_prefer_more)
                    std::ranges::stable_sort(m_scored, [](const auto& a, const auto& b) { return a.first < b.first; });
                else
                    std::ranges::stable_sort(m_scored, [](const auto& a, const auto& b) { return a.first > b.first; });
                for (usize i = 0; i < layer.size(); ++i)
                    layer[i] = m_scored[i].second;
                return;
            }
        }
    }

private:
    LayerOrdering::Kind m_kind = LayerOrdering::Kind::Queue;
    bool m_prefer_more = true;
    u32 m_limit = ~u32{0};
    SplitMix64 m_rng{0};
    GoalTest m_goal;
    std::vector<std::pair<u32, u32>> m_scored;
};

/// Checks a LayerOrdering: an error message, or an empty string.
inline std::string check_layers(const LayerOrdering& lo)
{
    if (lo.kind == LayerOrdering::Kind::Queue && lo.max_next_layer_states != ~u32{0})
        return "LayerOrdering::max_next_layer_states requires an ordered layer kind (every kind but Queue)";
    if (lo.max_next_layer_states == 0)
        return "LayerOrdering::max_next_layer_states must be positive";
    return {};
}
}  // namespace mymyr::search::detail
