#pragma once
// Atom feature interning shared by abstracted IW and minimum-g width search.

#include "mymyr/core/hash.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/novelty/landmark_table.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <span>
#include <unordered_map>
#include <vector>

namespace mymyr::search::detail
{
struct KeyHash
{
    usize operator()(const std::vector<u32>& k) const noexcept
    {
        u64 h = k.size();
        for (u32 x : k)
            h = hash::combine(h, x);
        return static_cast<usize>(h);
    }
};

class AbstractedFeatures
{
public:
    AbstractedFeatures(const Task& task, bool base, bool preserve_goal_atoms,
                       const novelty::LandmarkCoordinates* coords, bool preserve_landmark_atoms)
        : m_task(task), m_base(base)
    {
        if (preserve_goal_atoms)
        {
            const formalism::TaskData& t = task.data();
            const CanonicalLayout& L = task.atoms().layout();
            std::vector<u32> args;
            for (const formalism::Literal& l : t.literals_of(t.goal))
            {
                if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
                    continue;
                args.clear();
                for (formalism::Term x : t.terms_of(l))
                    args.push_back(formalism::term_object(x).v);
                const CanonicalAtom c = L.encode(l.pred.v, args.data());
                if (c < L.fluent_count)
                    m_preserved.push_back(task.atoms().intern(c));
            }
        }
        if (coords && preserve_landmark_atoms)
            m_preserved.insert(m_preserved.end(), coords->atoms().begin(), coords->atoms().end());
        std::sort(m_preserved.begin(), m_preserved.end());
        m_preserved.erase(std::unique(m_preserved.begin(), m_preserved.end()), m_preserved.end());
    }
    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_ids.size()); }
    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 bytes = m_ids.bucket_count() * sizeof(void*);
        for (const auto& [key, id] : m_ids)
            bytes += sizeof(key) + sizeof(id) + key.capacity() * sizeof(u32);
        for (const auto& f : m_features)
            bytes += sizeof(f) + f.capacity() * sizeof(u32);
        return bytes;
    }
private:
    u32 intern(const std::vector<u32>& key)
    {
        const auto [it, inserted] = m_ids.emplace(key, static_cast<u32>(m_ids.size()));
        return it->second;
    }
    void append_signature(u32 object, std::vector<u32>& out) const
    {
        const formalism::TaskData& t = m_task.data();
        const formalism::Range r = t.objects[object].types;
        if (r.count == 0)
        {
            out.push_back(1);
            out.push_back(~u32{0});
            return;
        }
        out.push_back(r.count);
        for (u32 i = 0; i < r.count; ++i)
            out.push_back(t.type_ids[r.begin + i].v);
    }
    std::vector<u32> compute(u32 slot)
    {
        const AtomIndex& ix = m_task.atoms();
        const u32 pred = ix.predicate(SlotId{slot}).v;
        const std::span<const u32> args = ix.arguments(SlotId{slot});
        std::vector<u32> out;
        auto full = [&]()
        {
            m_key.assign({1u, pred});
            m_key.insert(m_key.end(), args.begin(), args.end());
            return intern(m_key);
        };
        if (std::binary_search(m_preserved.begin(), m_preserved.end(), slot))
        {
            out.push_back(full());
            if (args.size() <= 1)
                return out;
        }
        if (args.empty())
            return {full()};
        for (u32 i = 0; i < args.size(); ++i)
        {
            m_key.assign({0u, pred, i, args[i]});
            if (!m_base)
                for (u32 j = 0; j < args.size(); ++j)
                    if (j != i)
                        append_signature(args[j], m_key);
            out.push_back(intern(m_key));
        }
        return out;
    }
public:
    std::span<const u32> features(u32 slot)
    {
        if (slot >= m_features.size())
            m_features.resize(slot + 1);
        if (m_features[slot].empty())
            m_features[slot] = compute(slot);
        return m_features[slot];
    }

private:
    const Task& m_task;
    bool m_base;
    std::vector<u32> m_preserved;
    std::unordered_map<std::vector<u32>, u32, KeyHash> m_ids;
    std::vector<u32> m_key;
    std::vector<std::vector<u32>> m_features;
};
}  // namespace mymyr::search::detail
