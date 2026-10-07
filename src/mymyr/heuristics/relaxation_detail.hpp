#pragma once
// Internal: pieces shared by the grounded relaxation heuristics (heuristics.cpp, h2.cpp): saturating costs, the
// monotone priority queue of the explorations and the mapping of a state to the propositions of a RelaxedTask.

#include "mymyr/core/bitset.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace mymyr::heuristics::detail
{
inline constexpr u32 k_inf = ~u32{0};
inline constexpr u32 k_none = ~u32{0};

[[nodiscard]] inline u32 sat_add(u32 a, u32 b) noexcept
{
    const u64 s = static_cast<u64>(a) + b;
    return s >= k_inf ? k_inf - 1 : static_cast<u32>(s);
}
[[nodiscard]] inline Value to_value(u64 h) noexcept { return h >= k_inf ? k_dead_end : static_cast<Value>(h); }

/// Monotone priority queue of the relaxed exploration: buckets for small keys, a binary heap beyond.
class RelaxedQueue
{
public:
    static constexpr u32 k_buckets = 1024;

    RelaxedQueue() : m_b(k_buckets) {}
    void clear()
    {
        if (m_size)
            for (u32 i = m_cur; i <= m_top; ++i)
                m_b[i].clear();
        m_cur = 0;
        m_top = 0;
        m_size = 0;
        m_heap.clear();
    }
    void push(u32 key, u32 x)
    {
        if (key < k_buckets)
        {
            m_b[key].push_back(x);
            m_top = std::max(m_top, key);
            m_cur = std::min(m_cur, key);
            ++m_size;
        }
        else
        {
            m_heap.push_back((static_cast<u64>(key) << 32) | x);
            std::push_heap(m_heap.begin(), m_heap.end(), std::greater<u64>());
        }
    }
    bool pop(u32& key, u32& x)
    {
        if (m_size)
        {
            while (m_b[m_cur].empty())
                ++m_cur;
            x = m_b[m_cur].back();
            m_b[m_cur].pop_back();
            key = m_cur;
            --m_size;
            return true;
        }
        if (m_heap.empty())
            return false;
        std::pop_heap(m_heap.begin(), m_heap.end(), std::greater<u64>());
        key = static_cast<u32>(m_heap.back() >> 32);
        x = static_cast<u32>(m_heap.back());
        m_heap.pop_back();
        return true;
    }

private:
    std::vector<std::vector<u32>> m_b;
    u32 m_cur = 0, m_top = 0;
    u64 m_size = 0;
    std::vector<u64> m_heap;
};

/// The propositions of a state: its fluent atoms' "true" propositions (pos) and the "false" propositions of those of
/// its atoms that have one (neg: these do not hold). Caches the slot -> proposition mapping.
class StateProps
{
public:
    std::vector<u32> pos, neg;

    /// False if an atom of `s` lies outside the grounded relaxation.
    bool convert(const Task& task, const RelaxedTask& R, StateView s)
    {
        const AtomIndex& A = task.atoms();
        pos.clear();
        neg.clear();
        bool ok = true;
        bits::for_each(s.w, s.nw,
                       [&](u64 slot)
                       {
                           if (!ok)
                               return;
                           if (slot >= m_slot_pos.size())
                           {
                               m_slot_pos.resize(slot + 1, k_unknown);
                               m_slot_neg.resize(slot + 1, k_unknown);
                           }
                           if (m_slot_pos[slot] == k_unknown)
                           {
                               const auto [pp, np] = R.fluent_props(A.canonical(AtomKind::Fluent, static_cast<u32>(slot)));
                               m_slot_pos[slot] = pp;
                               m_slot_neg[slot] = np;
                           }
                           const u32 p = m_slot_pos[slot];
                           if (p == k_none)
                           {
                               ok = false;
                               return;
                           }
                           pos.push_back(p);
                           if (m_slot_neg[slot] != k_none)
                               neg.push_back(m_slot_neg[slot]);
                       });
        return ok;
    }

private:
    static constexpr u32 k_unknown = ~u32{0} - 1;
    std::vector<u32> m_slot_pos, m_slot_neg;
};
}  // namespace mymyr::heuristics::detail
