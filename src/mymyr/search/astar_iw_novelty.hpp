#pragma once

#include "abstracted_features.hpp"
#include "iw_family_detail.hpp"

#include "mymyr/novelty/minimum_g_table.hpp"
#include "mymyr/search/astar_iw.hpp"

#include <array>
#include <optional>
#include <unordered_map>

namespace mymyr::search::detail
{
class MinimumGBackend
{
public:
    MinimumGBackend(const Task& task, const AStarIwOptions& o)
        : m_width(o.width), m_coords(make_coordinates(task, o.landmarks)), m_classical(o.width),
          m_pmask(m_coords.mask_words()), m_smask(m_coords.mask_words())
    {
        if (o.features != AStarIwFeatures::Classical)
            m_features.emplace(task, o.features == AStarIwFeatures::BaseAbstracted, o.preserve_goal_atoms,
                               o.landmarks.graph ? &m_coords : nullptr, o.preserve_landmark_atoms);
    }
    bool initialize(StateView s, f64 g)
    {
        const bool any = m_coords.mask(s.w, s.nw, m_smask.data());
        m_coords.collect(m_smask.data(), any, m_kept);
        bool improved = false;
        for (u32 rank : m_kept)
            improved = query(s, nullptr, g, rank, Mode::Update) || improved;
        return improved;
    }
    bool test_and_update(StateView parent, StateView s, f64 g) { return transition(parent, s, g, Mode::Update); }
    bool would_improve(StateView parent, StateView s, f64 g) { return transition(parent, s, g, Mode::Probe); }
    bool test_at_g(StateView s, f64 g)
    {
        const bool any = m_coords.mask(s.w, s.nw, m_smask.data());
        m_coords.collect(m_smask.data(), any, m_kept);
        for (u32 rank : m_kept)
            if (query(s, nullptr, g, rank, Mode::AtG))
                return true;
        return false;
    }
    [[nodiscard]] bool may_have_stale_novelty() const noexcept
    {
        return m_lowered || m_classical.has_lowered_existing_label();
    }
    [[nodiscard]] u64 bytes() const noexcept
    {
        return m_classical.bytes() + m_labels.bucket_count() * sizeof(void*) +
               m_labels.size() * (sizeof(Labels::value_type) + sizeof(void*)) + (m_features ? m_features->bytes() : 0);
    }

private:
    enum class Mode { Update, Probe, AtG };
    struct Key
    {
        u32 coordinate;
        u32 size;
        std::array<u32, 3> features{};
        friend bool operator==(const Key&, const Key&) = default;
    };
    struct KeyHash
    {
        usize operator()(const Key& k) const noexcept
        {
            u64 h = hash::combine(k.coordinate, k.size);
            for (u32 f : k.features)
                h = hash::combine(h, f);
            return static_cast<usize>(h);
        }
    };
    using Labels = std::unordered_map<Key, f64, KeyHash>;
    bool transition(StateView p, StateView s, f64 g, Mode mode)
    {
        const bool pany = m_coords.mask(p.w, p.nw, m_pmask.data());
        const bool sany = m_coords.mask(s.w, s.nw, m_smask.data());
        m_coords.split(m_pmask.data(), pany, m_smask.data(), sany, m_flipped, m_kept);
        bool improved = false;
        for (u32 rank : m_flipped)
        {
            improved = query(s, nullptr, g, rank, mode) || improved;
            if (improved && mode != Mode::Update)
                return true;
        }
        for (u32 rank : m_kept)
        {
            improved = query(s, &p, g, rank, mode) || improved;
            if (improved && mode != Mode::Update)
                return true;
        }
        return improved;
    }
    bool touch(Key key, f64 g, Mode mode)
    {
        if (key.size >= 2 && key.features[0] > key.features[1])
            std::swap(key.features[0], key.features[1]);
        if (key.size == 3)
        {
            if (key.features[1] > key.features[2])
                std::swap(key.features[1], key.features[2]);
            if (key.features[0] > key.features[1])
                std::swap(key.features[0], key.features[1]);
        }
        const auto it = m_labels.find(key);
        if (mode == Mode::AtG)
            return it != m_labels.end() && it->second == g;
        if (mode == Mode::Probe)
            return it == m_labels.end() || g < it->second;
        if (it == m_labels.end())
        {
            m_labels.emplace(key, g);
            return true;
        }
        if (g < it->second)
        {
            it->second = g;
            m_lowered = true;
            return true;
        }
        return false;
    }
    bool query(StateView s, const StateView* p, f64 g, u32 rank, Mode mode)
    {
        if (!m_features)
        {
            if (mode == Mode::AtG)
                return m_classical.test_at_g(s, g, rank);
            if (mode == Mode::Probe)
                return p ? m_classical.would_improve(*p, s, g, rank) : m_classical.would_improve(s, g, rank);
            return p ? m_classical.test_and_update(*p, s, g, rank) : m_classical.test_and_update(s, g, rank);
        }
        std::vector<u32> atoms;
        bits::for_each(s.w, s.nw, [&](u64 a) { atoms.push_back(static_cast<u32>(a)); });
        if (!atoms.empty())
            m_features->features(atoms.back());  // intern before taking spans into the slot cache
        std::vector<std::span<const u32>> groups;
        for (u32 a : atoms)
            groups.push_back(m_features->features(a));
        Key key{rank, 0, {}};
        bool improved = false;
        // Select distinct atom groups, then one distinct feature from each. A tuple must contain an added group.
        auto walk = [&](auto& self, usize from, bool added) -> bool
        {
            if (key.size && (!p || added))
            {
                const bool hit = touch(key, g, mode);
                improved = hit || improved;
                if (hit && mode != Mode::Update)
                    return true;
            }
            if (key.size >= m_width || key.size >= 3)
                return false;
            for (usize i = from; i < groups.size(); ++i)
                for (u32 feature : groups[i])
                {
                    if (std::find(key.features.begin(), key.features.begin() + key.size, feature) != key.features.begin() + key.size)
                        continue;
                    key.features[key.size++] = feature;
                    const bool stop = self(self, i + 1, added || (p && !p->contains(SlotId{atoms[i]})));
                    key.features[--key.size] = 0;
                    if (stop)
                        return true;
                }
            return false;
        };
        walk(walk, 0, false);
        return improved;
    }

    u32 m_width;
    novelty::LandmarkCoordinates m_coords;
    novelty::MinimumGNoveltyTable m_classical;
    std::optional<AbstractedFeatures> m_features;
    Labels m_labels;
    bool m_lowered = false;
    std::vector<u64> m_pmask, m_smask;
    std::vector<u32> m_flipped, m_kept;
};
}  // namespace mymyr::search::detail
