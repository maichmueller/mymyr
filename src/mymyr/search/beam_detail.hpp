#pragma once
// The parallel layer step of a beam search (search/layer_ordering.hpp, "Threads"), shared by search::iw (iw.cpp),
// search::brfs (brfs.cpp) and the engine of the IW family variants (novelty_brfs.hpp).
//
// A layer's states are expanded in two phases: the members of a Team generate the successors of the states, each
// state by one member, into that member's Candidates (with whatever the engine computes from a successor alone: the
// read-only novelty test, the score); then the calling thread walks the states in layer order and replays the
// engine's serial step over the recorded transitions. Which member expands which state depends on the timing, but
// every state's record is the same, so the result does not.

#include "mymyr/core/team.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <span>
#include <thread>
#include <vector>

namespace mymyr::search::detail
{
/// threads, with 0 meaning std::thread::hardware_concurrency().
[[nodiscard]] inline u32 resolve_threads(u32 threads) noexcept
{
    return threads == 0 ? std::max<u32>(1, std::thread::hardware_concurrency()) : threads;
}

/// The transitions one member recorded, as structure of arrays: per transition its index among its state's
/// transitions (seq), schema, binding, added atoms (true in the successor, false in the state; each once), the
/// delta's add and delete slots when the engine needs them, the successor's words (trimmed) and numeric words, and
/// two engine-defined values.
class Candidates
{
public:
    void clear()
    {
        m_seq.clear();
        m_schema.clear();
        m_value.clear();
        m_flag.clear();
        m_boff.clear();
        m_binding.clear();
        m_aoff.clear();
        m_add.clear();
        m_doff.clear();
        m_dadd.clear();
        m_ddel.clear();
        m_woff.clear();
        m_words.clear();
        m_num.clear();
        m_boff.push_back(0);
        m_aoff.push_back(0);
        m_doff.push_back(0);
        m_doff.push_back(0);
        m_woff.push_back(0);
    }
    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_seq.size()); }

    /// Records a transition; returns its index. `words` may be null (nn = 0) when the engine needs no successor.
    u32 push(u32 seq, u32 schema, const ObjectId* binding, u32 arity, std::span<const u32> add, const Delta* delta,
             const u64* words, u32 nn, const u64* num, u32 nnum, u32 value = 0, u8 flag = 0)
    {
        m_seq.push_back(seq);
        m_schema.push_back(schema);
        m_value.push_back(value);
        m_flag.push_back(flag);
        m_binding.insert(m_binding.end(), binding, binding + arity);
        m_boff.push_back(m_binding.size());
        m_add.insert(m_add.end(), add.begin(), add.end());
        m_aoff.push_back(m_add.size());
        if (delta)
        {
            m_dadd.insert(m_dadd.end(), delta->add.begin(), delta->add.end());
            m_ddel.insert(m_ddel.end(), delta->del.begin(), delta->del.end());
        }
        m_doff.push_back(m_dadd.size());
        m_doff.push_back(m_ddel.size());
        m_words.insert(m_words.end(), words, words + nn);
        m_woff.push_back(m_words.size());
        if (nnum)
            m_num.insert(m_num.end(), num, num + nnum);
        return size() - 1;
    }

    [[nodiscard]] u32 seq(u32 i) const noexcept { return m_seq[i]; }
    [[nodiscard]] u32 schema(u32 i) const noexcept { return m_schema[i]; }
    [[nodiscard]] u32 value(u32 i) const noexcept { return m_value[i]; }
    void set_value(u32 i, u32 v) noexcept { m_value[i] = v; }
    [[nodiscard]] u8 flag(u32 i) const noexcept { return m_flag[i]; }
    [[nodiscard]] const ObjectId* binding(u32 i) const noexcept { return m_binding.data() + m_boff[i]; }
    [[nodiscard]] std::span<const u32> add(u32 i) const noexcept { return {m_add.data() + m_aoff[i], m_aoff[i + 1] - m_aoff[i]}; }
    /// The delta of transition i (its add and delete slots; numeric words as num()), when it was recorded.
    [[nodiscard]] Delta delta(u32 i, u32 nnum) const noexcept
    {
        const u64 a0 = m_doff[2 * i], d0 = m_doff[2 * i + 1], a1 = m_doff[2 * i + 2], d1 = m_doff[2 * i + 3];
        return {{m_dadd.data() + a0, a1 - a0}, {m_ddel.data() + d0, d1 - d0}, num(i, nnum), nnum, {}, {}};
    }
    [[nodiscard]] const u64* words(u32 i) const noexcept { return m_words.data() + m_woff[i]; }
    [[nodiscard]] u32 nwords(u32 i) const noexcept { return static_cast<u32>(m_woff[i + 1] - m_woff[i]); }
    /// The numeric words of transition i (nnum per transition, in push order).
    [[nodiscard]] const u64* num(u32 i, u32 nnum) const noexcept { return nnum ? m_num.data() + static_cast<usize>(i) * nnum : nullptr; }

private:
    std::vector<u32> m_seq, m_schema, m_value;
    std::vector<u8> m_flag;
    std::vector<u64> m_boff, m_aoff, m_doff, m_woff;
    std::vector<ObjectId> m_binding;
    std::vector<u32> m_add;
    std::vector<SlotId> m_dadd, m_ddel;  // m_doff: (add, delete) offsets per transition
    std::vector<u64> m_words, m_num;
};

/// What one member recorded for one state of the layer.
struct Expansion
{
    u32 member = 0;       // whose Candidates hold its transitions
    u32 first = 0;        // its transitions: [first, first + count) of that member's Candidates
    u32 count = 0;
    u32 transitions = 0;  // all transitions generated (recorded or not)
    u8 goal = 0;          // the goal test, when the engine runs it in parallel
    u8 expanded = 0;      // 1: the successors were generated
};

/// The members of a parallel layer step: a Team and each member's successor generator (its own workspace; member 0
/// is the calling thread) and Candidates.
class BeamTeam
{
public:
    BeamTeam(const Task& task, u32 threads) : m_team(threads), m_succ(m_team.size()), m_cands(m_team.size())
    {
        m_team.run([&](u32 t) { m_succ[t] = &task.workspace().successors(); });
    }

    [[nodiscard]] u32 size() const noexcept { return m_team.size(); }
    [[nodiscard]] Successors& succ(u32 t) noexcept { return *m_succ[t]; }
    [[nodiscard]] Candidates& candidates(u32 t) noexcept { return m_cands[t]; }
    [[nodiscard]] const Candidates& candidates(u32 t) const noexcept { return m_cands[t]; }

    /// Clears every member's Candidates, then calls f(t, i) for every i in [0, n) on the members (t: the member),
    /// handing out `grain` consecutive indices at a time. f may write only to what belongs to i or to member t.
    template<class F>
    void for_each(usize n, usize grain, F&& f)
    {
        for (Candidates& c : m_cands)
            c.clear();
        std::atomic<usize> next{0};
        m_team.run(
            [&](u32 t)
            {
                for (;;)
                {
                    const usize a = next.fetch_add(grain, std::memory_order_relaxed);
                    if (a >= n)
                        break;
                    const usize b = std::min(n, a + grain);
                    for (usize i = a; i < b; ++i)
                        f(t, i);
                }
            });
    }

    /// Calls f(t) on every member at once.
    template<class F>
    void run(F&& f)
    {
        m_team.run([&](u32 t) { f(t); });
    }

private:
    Team m_team;
    std::vector<Successors*> m_succ;
    std::vector<Candidates> m_cands;
};
}  // namespace mymyr::search::detail
