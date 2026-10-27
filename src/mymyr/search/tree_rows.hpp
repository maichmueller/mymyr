#pragma once

#include "mymyr/core/hash.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/state/state.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr::search::detail
{
/// Interned successor rows. Ids are dense and assigned in insertion order. Equality always compares the full state,
/// including numeric words. Hashes ignore trailing zero fluent words, so lazy-slot widening preserves the index.
///
/// Rows are interned in batches (stage, flush): staging hashes a row and prefetches its index slot, and a flush
/// resolves the batch in staging order, so index probes of different rows overlap their cache misses instead of
/// stalling the search on each one. Index entries hold a 32-bit hash tag next to the id, so a probe reads a stored
/// row only on a tag match. The index grows fourfold and rehashes from the stored rows. Row storage and the index
/// grow into buffers advised for huge pages (reserve_huge).
class TreeRows
{
public:
    static constexpr u32 k_batch = 32;

    TreeRows(u32 words, u32 numeric_words)
        : m_W(std::max<u32>(1, words)), m_NN(numeric_words), m_table(1024, 0)
    {
        m_pending.reserve(k_batch);
    }
    TreeRows(const TreeRows&) = delete;
    TreeRows& operator=(const TreeRows&) = delete;

    [[nodiscard]] u32 size() const noexcept { return m_size; }
    [[nodiscard]] StateView operator[](u32 id) const noexcept
    {
        const u64* row = m_rows.data() + static_cast<usize>(id) * (m_W + m_NN);
        return {row, m_W, row + m_W, m_NN};
    }

    /// Queues `state` (copied); the flush that resolves it writes its row id to ids[target]. Flushes a full batch.
    void stage(StateView state, usize target, std::vector<i32>& ids)
    {
        const u32 n = bits::trimmed_size(state.w, state.nw);
        const u64 h = hash::state(state.w, n, state.num, m_NN);
        __builtin_prefetch(m_table.data() + (h & (m_table.size() - 1)));
        m_pending.push_back({h, target, m_staged.size(), n});
        m_staged.insert(m_staged.end(), state.w, state.w + n);
        if (m_NN)
            m_staged.insert(m_staged.end(), state.num, state.num + m_NN);
        if (m_pending.size() == k_batch)
            flush(ids);
    }

    /// Resolves every staged row, in staging order.
    void flush(std::vector<i32>& ids)
    {
        for (const Pending& p : m_pending)
        {
            const u64* w = m_staged.data() + p.offset;
            ids[p.target] = static_cast<i32>(insert(w, p.n, w + p.n, p.hash).first);
        }
        m_pending.clear();
        m_staged.clear();
    }

    /// Interns `state` immediately. Staged rows are flushed first by the caller, so that ids keep the insertion order.
    std::pair<u32, bool> insert(StateView state)
    {
        const u32 n = bits::trimmed_size(state.w, state.nw);
        return insert(state.w, n, state.num, hash::state(state.w, n, state.num, m_NN));
    }

private:
    struct Pending
    {
        u64 hash;
        usize target, offset;
        u32 n;
    };

    /// `w` holds n trimmed words, `num` m_NN numeric words, `h` their hash::state.
    std::pair<u32, bool> insert(const u64* w, u32 n, const u64* num, u64 h)
    {
        if (n > m_W)
            widen(std::max(m_W * 2, n));
        if ((usize{m_size} + 1) * 10 > m_table.size() * 7)
            grow();
        const StateView state{w, n, num, m_NN};
        const u64 mask = m_table.size() - 1;
        for (u64 at = h & mask;; at = (at + 1) & mask)
        {
            const u64 entry = m_table[at];
            if (!entry)
            {
                if (m_size >= static_cast<u32>(std::numeric_limits<i32>::max()))
                    throw std::length_error("SearchTree: more than 2^31 - 1 rows");
                const u32 id = m_size;
                const usize start = m_rows.size();
                if (start + m_W + m_NN > m_rows.capacity())
                    reserve_huge(m_rows, start + m_W + m_NN);
                m_rows.resize(start + m_W + m_NN, 0);
                std::copy_n(w, n, m_rows.data() + start);
                if (m_NN)
                    std::copy_n(num, m_NN, m_rows.data() + start + m_W);
                ++m_size;
                m_table[at] = (h & 0xffffffff00000000ULL) | (static_cast<u64>(id) + 1);
                return {id, true};
            }
            const u32 id = static_cast<u32>(entry) - 1;
            if ((entry >> 32) == (h >> 32) && (*this)[id] == state)
                return {id, false};
        }
    }

    void grow()
    {
        std::vector<u64> table;
        reserve_huge(table, m_table.size() * 4);
        table.resize(m_table.size() * 4, 0);
        const u64 mask = table.size() - 1;
        // hashes run k_ahead rows ahead of the insertions, with their slots prefetched
        constexpr u32 k_ahead = 16;
        u64 ahead[k_ahead];
        auto hash_of = [&](u32 id)
        {
            const StateView row = (*this)[id];
            const u64 h = hash::state(row.w, row.nw, row.num, m_NN);
            __builtin_prefetch(table.data() + (h & mask));
            return h;
        };
        for (u32 id = 0; id < std::min(m_size, k_ahead); ++id)
            ahead[id] = hash_of(id);
        for (u32 id = 0; id < m_size; ++id)
        {
            const u64 h = ahead[id % k_ahead];
            if (id + k_ahead < m_size)
                ahead[id % k_ahead] = hash_of(id + k_ahead);
            u64 at = h & mask;
            while (table[at])
                at = (at + 1) & mask;
            table[at] = (h & 0xffffffff00000000ULL) | (static_cast<u64>(id) + 1);
        }
        m_table.swap(table);
    }
    void widen(u32 words)
    {
        const usize stride = usize{words} + m_NN;
        std::vector<u64> rows;
        reserve_huge(rows, m_size * stride);
        rows.resize(m_size * stride, 0);
        for (u32 id = 0; id < m_size; ++id)
        {
            const StateView source = (*this)[id];
            u64* row = rows.data() + static_cast<usize>(id) * stride;
            std::copy_n(source.w, m_W, row);
            if (m_NN)
                std::copy_n(source.num, m_NN, row + words);
        }
        m_rows.swap(rows);
        m_W = words;
    }

    u32 m_W, m_NN;
    u32 m_size = 0;
    std::vector<u64> m_rows;  // m_size rows of m_W fluent words (zero-padded) and m_NN numeric words
    std::vector<u64> m_table;
    std::vector<Pending> m_pending;
    std::vector<u64> m_staged;
};
}  // namespace mymyr::search::detail
