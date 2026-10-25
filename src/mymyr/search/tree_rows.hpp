#pragma once

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
/// Interned successor rows with cached hashes. Growing the index reads hashes instead of state rows; fourfold
/// growth amortizes index rebuilds during bulk recording. Equality always compares the full state, including
/// numeric words. Hashes ignore trailing zero fluent words, so lazy-slot widening preserves the index.
class TreeRows
{
public:
    TreeRows(u32 words, u32 numeric_words)
        : m_W(std::max<u32>(1, words)), m_NN(numeric_words), m_table(1024, 0)
    {
    }
    TreeRows(const TreeRows&) = delete;
    TreeRows& operator=(const TreeRows&) = delete;
    ~TreeRows() { std::free(m_rows); }

    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_hashes.size()); }
    [[nodiscard]] StateView operator[](u32 id) const noexcept
    {
        const u64* row = m_rows + static_cast<usize>(id) * (m_W + m_NN);
        return {row, m_W, row + m_W, m_NN};
    }

    std::pair<u32, bool> insert(StateView state)
    {
        const u32 n = bits::trimmed_size(state.w, state.nw);
        if (n > m_W)
            widen(std::max(m_W * 2, n));
        if ((m_hashes.size() + 1) * 10 > m_table.size() * 7)
            grow();
        const u64 h = hash::state(state.w, n, state.num, m_NN);
        const u64 mask = m_table.size() - 1;
        for (u64 at = h & mask;; at = (at + 1) & mask)
        {
            const u64 entry = m_table[at];
            if (!entry)
            {
                if (m_hashes.size() >= static_cast<usize>(std::numeric_limits<i32>::max()))
                    throw std::length_error("SearchTree: more than 2^31 - 1 rows");
                const u32 id = size();
                const usize start = static_cast<usize>(id) * (m_W + m_NN);
                const usize end = start + m_W + m_NN;
                if (end > m_capacity)
                {
                    const usize capacity = std::max<usize>({end, m_capacity * 2, 1024});
                    auto* rows = static_cast<u64*>(std::realloc(m_rows, capacity * sizeof(u64)));
                    if (!rows)
                        throw std::bad_alloc();
                    m_rows = rows;
                    m_capacity = capacity;
                }
                u64* row = m_rows + start;
                std::copy_n(state.w, n, row);
                std::fill(row + n, row + m_W, 0);
                if (m_NN)
                    std::copy_n(state.num, m_NN, row + m_W);
                m_hashes.push_back(h);
                m_table[at] = (h & 0xffffffff00000000ULL) | (static_cast<u64>(id) + 1);
                return {id, true};
            }
            const u32 id = static_cast<u32>(entry) - 1;
            if ((entry >> 32) == (h >> 32) && (*this)[id] == state)
                return {id, false};
        }
    }

private:
    void grow()
    {
        std::vector<u64> table(m_table.size() * 4, 0);
        const u64 mask = table.size() - 1;
        for (u32 id = 0; id < size(); ++id)
        {
            const u64 h = m_hashes[id];
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
        const usize capacity = std::max<usize>(size() * stride, 1);
        auto* rows = static_cast<u64*>(std::calloc(capacity, sizeof(u64)));
        if (!rows)
            throw std::bad_alloc();
        for (u32 id = 0; id < size(); ++id)
        {
            const StateView source = (*this)[id];
            u64* row = rows + static_cast<usize>(id) * stride;
            std::copy_n(source.w, m_W, row);
            if (m_NN)
                std::copy_n(source.num, m_NN, row + words);
        }
        std::free(m_rows);
        m_rows = rows;
        m_capacity = capacity;
        m_W = words;
    }

    u32 m_W, m_NN;
    u64* m_rows = nullptr;
    usize m_capacity = 0;
    std::vector<u64> m_table, m_hashes;
};
}  // namespace mymyr::search::detail
