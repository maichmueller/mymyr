#pragma once
// Reordering the next layer of a breadth-first pass by a LayerOrdering (search/layer_ordering.hpp) and cutting it to
// the beam: shared by search::iw (iw.cpp), search::brfs (brfs.cpp) and the engine of the IW family variants
// (novelty_brfs.hpp).

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
/// The ordering state of one search: its kind, the beam, the generator (Randomized and random tie tokens; it
/// persists across layers and passes) and the goal literals (GoalCount).
class LayerOrderer
{
public:
    LayerOrderer() = default;
    LayerOrderer(const Task& task, const LayerOrdering& lo)
        : m_kind(lo.kind), m_prefer_more(lo.prefer_more_satisfied_goals), m_random_ties(lo.randomize_ties),
          m_survivors_only(lo.beam_novelty == LayerOrdering::BeamNovelty::SurvivorsOnly), m_limit(lo.max_next_layer_states),
          m_beam(lo.beam_width), m_rng(lo.seed)
    {
        if (lo.kind == LayerOrdering::Kind::GoalCount)
            m_goal = GoalTest::counter(task, 0);
    }

    /// Layer by layer (every kind but Queue).
    [[nodiscard]] bool ordered() const noexcept { return m_kind != LayerOrdering::Kind::Queue; }
    /// max_next_layer_states (~0: no limit).
    [[nodiscard]] u32 limit() const noexcept { return m_limit; }
    [[nodiscard]] bool limited() const noexcept { return ordered() && m_limit != ~u32{0}; }
    /// beam_width (~0: no beam).
    [[nodiscard]] bool beam() const noexcept { return ordered() && m_beam != ~u32{0}; }
    [[nodiscard]] u32 beam_width() const noexcept { return m_beam; }
    /// A beam whose kept entries are replayed through the novelty test (BeamNovelty::SurvivorsOnly).
    [[nodiscard]] bool survivors_only() const noexcept { return beam() && m_survivors_only; }

    /// Reorders a next layer of entries (node or state ids) and, with a beam, keeps its first beam_width entries.
    /// state_of(entry) -> StateView gives an entry's state (GoalCount scores it; the engine of `succ` is pointed at
    /// it, so call this between expansions). Returns the number of entries the beam dropped.
    template<class StateOf>
    usize select(std::vector<u32>& layer, Successors& succ, StateOf&& state_of)
    {
        const usize keep = beam() ? std::min<usize>(m_beam, layer.size()) : layer.size();
        switch (m_kind)
        {
            case LayerOrdering::Kind::Queue:
            case LayerOrdering::Kind::InOrder: break;
            case LayerOrdering::Kind::Reverse: std::reverse(layer.begin(), layer.end()); break;
            case LayerOrdering::Kind::Randomized: m_rng.shuffle(std::span<u32>(layer)); break;
            case LayerOrdering::Kind::GoalCount:
            {
                // ranked by (key, tie token, generation position): a strict total order, so sorting equals mimir's
                // stable sort (and its beam ranking), and a partial sort gives the same top entries. The literal
                // count is fixed: more satisfied literals = fewer unsatisfied ones.
                m_ranked.clear();
                for (u32 i = 0; i < layer.size(); ++i)
                {
                    const u32 unsat = m_goal.unsatisfied(succ, state_of(layer[i]), false);
                    m_ranked.push_back({m_prefer_more ? unsat : ~unsat, m_random_ties ? m_rng.next() : 0, i, layer[i]});
                }
                const auto better = [](const Ranked& a, const Ranked& b)
                {
                    if (a.key != b.key)
                        return a.key < b.key;
                    if (a.tie != b.tie)
                        return a.tie < b.tie;
                    return a.pos < b.pos;
                };
                if (keep < m_ranked.size())
                    std::partial_sort(m_ranked.begin(), m_ranked.begin() + static_cast<std::ptrdiff_t>(keep), m_ranked.end(), better);
                else
                    std::sort(m_ranked.begin(), m_ranked.end(), better);
                for (usize i = 0; i < keep; ++i)
                    layer[i] = m_ranked[i].entry;
                break;
            }
        }
        const usize dropped = layer.size() - keep;
        layer.resize(keep);
        return dropped;
    }

private:
    struct Ranked
    {
        u32 key;   // smaller first
        u64 tie;   // random tie token (0 without randomize_ties)
        u32 pos;   // generation position
        u32 entry;
    };

    LayerOrdering::Kind m_kind = LayerOrdering::Kind::Queue;
    bool m_prefer_more = true;
    bool m_random_ties = false;
    bool m_survivors_only = false;
    u32 m_limit = ~u32{0};
    u32 m_beam = ~u32{0};
    SplitMix64 m_rng{0};
    GoalTest m_goal;
    std::vector<Ranked> m_ranked;
};

/// Checks a LayerOrdering: an error message, or an empty string.
inline std::string check_layers(const LayerOrdering& lo)
{
    if (lo.kind == LayerOrdering::Kind::Queue && lo.max_next_layer_states != ~u32{0})
        return "LayerOrdering::max_next_layer_states requires an ordered layer kind (every kind but Queue)";
    if (lo.max_next_layer_states == 0)
        return "LayerOrdering::max_next_layer_states must be positive";
    if (lo.beam_width == 0)
        return "LayerOrdering::beam_width must be positive";
    if (lo.beam() && lo.kind == LayerOrdering::Kind::Queue)
        return "LayerOrdering::beam_width requires an ordered layer kind (every kind but Queue)";
    if (lo.beam() && lo.max_next_layer_states != ~u32{0})
        return "LayerOrdering::beam_width and LayerOrdering::max_next_layer_states are mutually exclusive";
    if (!lo.beam() && lo.beam_novelty != LayerOrdering::BeamNovelty::AllTested)
        return "LayerOrdering::beam_novelty SurvivorsOnly requires a beam (beam_width)";
    if (lo.randomize_ties && lo.kind != LayerOrdering::Kind::GoalCount)
        return "LayerOrdering::randomize_ties requires Kind::GoalCount (the only kind with scores)";
    return {};
}
}  // namespace mymyr::search::detail
