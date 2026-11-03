#pragma once
// Reordering the next layer of a breadth-first pass by a LayerOrdering (search/layer_ordering.hpp) and cutting it to
// the beam: shared by search::iw (iw.cpp), search::brfs (brfs.cpp) and the engine of the IW family variants
// (novelty_brfs.hpp).

#include "iw_detail.hpp"

#include "mymyr/core/random.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/successor/successors.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::search::detail
{
/// The ordering state of one search: its kind, the beam, the generator (Randomized and random tie tokens; it
/// persists across layers and passes), the goal literals (GoalCount) and the scorer (Scored).
class LayerOrderer
{
public:
    /// A ranked entry: key (smaller first), tie token (0 without randomize_ties), generation position, and the
    /// caller's reference. better() is a strict total order when the positions differ.
    struct Ranked
    {
        u64 key;
        u64 tie;
        u64 pos;
        u32 entry;
    };
    static bool better(const Ranked& a, const Ranked& b) noexcept
    {
        if (a.key != b.key)
            return a.key < b.key;
        if (a.tie != b.tie)
            return a.tie < b.tie;
        return a.pos < b.pos;
    }

    LayerOrderer() = default;
    LayerOrderer(const Task& task, const LayerOrdering& lo)
        : m_kind(lo.kind), m_prefer_more(lo.prefer_more_satisfied_goals), m_random_ties(lo.randomize_ties),
          m_novelty(lo.beam_novelty), m_chunk(lo.beam_chunk), m_limit(lo.max_next_layer_states), m_beam(lo.beam_width),
          m_rng(lo.seed), m_scorer(lo.scorer)
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
    /// A beam whose kept entries are replayed through the novelty test (SurvivorsOnly and RelaxedSurvivorsOnly).
    [[nodiscard]] bool survivors_only() const noexcept { return beam() && m_novelty != LayerOrdering::BeamNovelty::AllTested; }
    /// BeamNovelty::RelaxedSurvivorsOnly.
    [[nodiscard]] bool relaxed() const noexcept { return beam() && m_novelty == LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly; }
    [[nodiscard]] u32 beam_chunk() const noexcept { return m_chunk; }
    /// GoalCount: entries are ranked by key(), a function of the state alone (computable on any thread). Scored ranks
    /// too, but by one call of the scorer per layer in select().
    [[nodiscard]] bool scored() const noexcept { return m_kind == LayerOrdering::Kind::GoalCount; }
    [[nodiscard]] bool random_ties() const noexcept { return m_random_ties; }

    /// GoalCount: the rank key of state s (smaller first). Points the engine of `succ` at s; callable from several
    /// threads at once with one Successors each.
    [[nodiscard]] u32 key(Successors& succ, StateView s) const
    {
        // the literal count is fixed: more satisfied literals = fewer unsatisfied ones
        const u32 unsat = m_goal.unsatisfied(succ, s, false);
        return m_prefer_more ? unsat : ~unsat;
    }

    /// Reorders a next layer of entries (node or state ids) and, with a beam, keeps its first beam_width entries.
    /// state_of(entry) -> StateView gives an entry's state (GoalCount scores it; the engine of `succ` is pointed at
    /// it, so call this between expansions; Scored copies the states and scores them in one call of the scorer).
    /// Returns the number of entries the beam dropped.
    template<class StateOf>
    usize select(std::vector<u32>& layer, Successors& succ, StateOf&& state_of)
    {
        if (m_kind == LayerOrdering::Kind::Scored)
        {
            score(layer, state_of);
            return rank(layer, m_wide_keys);
        }
        if (scored())
        {
            m_keys.resize(layer.size());
            for (usize i = 0; i < layer.size(); ++i)
                m_keys[i] = key(succ, state_of(layer[i]));
        }
        return select_keyed(layer, m_keys);
    }

    /// select() with the GoalCount keys given: keys[i] = key() of layer[i] (ignored by the other kinds but Scored,
    /// which select() ranks).
    usize select_keyed(std::vector<u32>& layer, std::span<const u32> keys)
    {
        switch (m_kind)
        {
            case LayerOrdering::Kind::Queue:
            case LayerOrdering::Kind::InOrder:
            case LayerOrdering::Kind::Scored: break;
            case LayerOrdering::Kind::Reverse: std::reverse(layer.begin(), layer.end()); break;
            case LayerOrdering::Kind::Randomized: m_rng.shuffle(std::span<u32>(layer)); break;
            case LayerOrdering::Kind::GoalCount: return rank(layer, keys);
        }
        const usize keep = beam() ? std::min<usize>(m_beam, layer.size()) : layer.size();
        const usize dropped = layer.size() - keep;
        layer.resize(keep);
        return dropped;
    }

    /// The rank key of a score: the order of the doubles (-inf first, +inf last) as unsigned integers.
    [[nodiscard]] static u64 score_key(f64 v) noexcept
    {
        const u64 bits = std::bit_cast<u64>(v == 0 ? 0.0 : v);  // -0 ranks as 0
        return (bits >> 63) ? ~bits : bits | (u64{1} << 63);
    }

    /// RelaxedSurvivorsOnly: the tie tokens of a layer of n transitions, as n draws of the generator in generation
    /// order (with randomize_ties; else nothing is drawn). Returns the generator before them: the token of transition
    /// i is output_at(i).
    [[nodiscard]] SplitMix64 draw_ties(u64 n) noexcept
    {
        const SplitMix64 at = m_rng;
        if (m_random_ties)
            m_rng.discard(n);
        return at;
    }

private:
    /// Ranks the layer by (key, tie token, generation position), a strict total order, so sorting equals a stable
    /// sort (and the beam ranking), and a partial sort gives the same top entries; keeps the beam.
    template<class Key>
    usize rank(std::vector<u32>& layer, std::span<const Key> keys)
    {
        const usize keep = beam() ? std::min<usize>(m_beam, layer.size()) : layer.size();
        m_ranked.clear();
        for (u32 i = 0; i < layer.size(); ++i)
            m_ranked.push_back({static_cast<u64>(keys[i]), m_random_ties ? m_rng.next() : 0, i, layer[i]});
        if (keep < m_ranked.size())
            std::partial_sort(m_ranked.begin(), m_ranked.begin() + static_cast<std::ptrdiff_t>(keep), m_ranked.end(), better);
        else
            std::sort(m_ranked.begin(), m_ranked.end(), better);
        for (usize i = 0; i < keep; ++i)
            layer[i] = m_ranked[i].entry;
        const usize dropped = layer.size() - keep;
        layer.resize(keep);
        return dropped;
    }
    template<class Key>
    usize rank(std::vector<u32>& layer, const std::vector<Key>& keys)
    {
        return rank(layer, std::span<const Key>(keys));
    }

    /// Scored: the keys of the layer's states (copied, as state_of may reuse its buffer) from one scorer call.
    template<class StateOf>
    void score(const std::vector<u32>& layer, StateOf& state_of)
    {
        const usize n = layer.size();
        m_offsets.assign(n + 1, 0);
        m_rows.clear();
        m_counts.resize(n);
        for (usize i = 0; i < n; ++i)
        {
            const StateView v = state_of(layer[i]);
            m_rows.insert(m_rows.end(), v.w, v.w + v.nw);
            if (v.nnum)
                m_rows.insert(m_rows.end(), v.num, v.num + v.nnum);
            m_counts[i] = {v.nw, v.nnum};
            m_offsets[i + 1] = m_rows.size();
        }
        m_views.resize(n);
        for (usize i = 0; i < n; ++i)
        {
            const u64* w = m_rows.data() + m_offsets[i];
            m_views[i] = {w, m_counts[i].first, m_counts[i].second ? w + m_counts[i].first : nullptr, m_counts[i].second};
        }
        m_scores.assign(n, 0);
        if (n)
            m_scorer(m_views, m_scores);
        m_wide_keys.resize(n);
        for (usize i = 0; i < n; ++i)
        {
            if (std::isnan(m_scores[i]))
                throw std::invalid_argument("mymyr: a layer score is NaN (LayerOrdering::scorer)");
            m_wide_keys[i] = score_key(m_scores[i]);
        }
    }

    LayerOrdering::Kind m_kind = LayerOrdering::Kind::Queue;
    bool m_prefer_more = true;
    bool m_random_ties = false;
    LayerOrdering::BeamNovelty m_novelty = LayerOrdering::BeamNovelty::AllTested;
    u32 m_chunk = 1024;
    u32 m_limit = ~u32{0};
    u32 m_beam = ~u32{0};
    SplitMix64 m_rng{0};
    GoalTest m_goal;
    std::vector<Ranked> m_ranked;
    std::vector<u32> m_keys;
    std::function<void(std::span<const StateView>, std::span<f64>)> m_scorer;
    std::vector<u64> m_rows, m_wide_keys;
    std::vector<usize> m_offsets;
    std::vector<std::pair<u32, u32>> m_counts;
    std::vector<StateView> m_views;
    std::vector<f64> m_scores;
};

/// The parts of a relaxed selection over n transitions (LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly):
/// min(threads, ceil(n / chunk)), at least 1.
[[nodiscard]] inline u32 relaxed_parts(u64 n, u32 threads, u32 chunk) noexcept
{
    const u64 by_chunk = std::max<u64>(1, (n + chunk - 1) / chunk);
    return static_cast<u32>(std::min<u64>(std::max<u32>(threads, 1), by_chunk));
}

/// The first transition of part p of n transitions split into `parts` (sizes differ by at most one, the longer ones
/// first).
[[nodiscard]] inline u64 relaxed_part_begin(u64 n, u32 parts, u32 p) noexcept
{
    const u64 base = n / parts, longer = n % parts;
    return p * base + std::min<u64>(p, longer);
}

/// The relaxed selection's ranking: `cands` (in generation order, pos = the transition's index in the layer; key and
/// tie filled) is split by pos into the parts of relaxed_parts(n, threads, chunk), the best k of each part are kept,
/// and the survivors of all parts are left in `cands` sorted by LayerOrderer::better.
inline void relaxed_rank(std::vector<LayerOrderer::Ranked>& cands, u64 n, u32 threads, u32 chunk, u32 k)
{
    const u32 parts = relaxed_parts(n, threads, chunk);
    usize write = 0, at = 0;
    for (u32 p = 0; p < parts; ++p)
    {
        const u64 end = relaxed_part_begin(n, parts, p + 1);
        const usize first = at;
        while (at < cands.size() && cands[at].pos < end)
            ++at;
        const usize m = at - first;
        const usize keep = std::min<usize>(m, k);
        auto b = cands.begin() + static_cast<std::ptrdiff_t>(first);
        std::partial_sort(b, b + static_cast<std::ptrdiff_t>(keep), b + static_cast<std::ptrdiff_t>(m), LayerOrderer::better);
        for (usize i = 0; i < keep; ++i)
            cands[write++] = cands[first + i];
    }
    cands.resize(write);
    std::sort(cands.begin(), cands.end(), LayerOrderer::better);
}
}  // namespace mymyr::search::detail
