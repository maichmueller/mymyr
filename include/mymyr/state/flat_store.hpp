#pragma once
// FlatStateStore: the default store for W <= 8 words.
//   - one arena of fixed-stride records, `u64 arena[id * W + w]`, u32 ids in insertion order;
//   - open addressing over 64-bit entries (tag32 << 32 | id + 1), grown at 70% load;
//   - records are zero-padded to the stride W. A state wider than W (lazy slots grow the width) re-lays out every
//     record with a wider stride; ids stay, and the table stays too because the hash covers the *trimmed* words
//     (hash::state_words), so equal states of any stored width hash equal;
//   - numeric tasks: a record is [W bits | NW numeric words] (slots inline); the hash is hash::state and equality
//     compares the numeric words bitwise (canonical values). Without numeric words nothing changes.
// Single-threaded; the concurrent store is ConcurrentStateStore.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr
{
class FlatStateStore
{
public:
    explicit FlatStateStore(u32 words = 1, u32 table_log = 16, u32 numeric_words = 0)
        : m_w(std::max<u32>(1, words)),
          m_nn(numeric_words),
          m_rs(m_w + numeric_words),
          m_table(u64{1} << table_log, 0),
          m_mask((u64{1} << table_log) - 1)
    {
    }
    FlatStateStore(const FlatStateStore&) = delete;
    FlatStateStore& operator=(const FlatStateStore&) = delete;
    ~FlatStateStore() { std::free(m_arena); }

    /// Inserts the state with words w[0, n) (missing words are zero) and, numeric tasks, numeric words num[0, NW).
    /// Returns its id and whether it is new.
    std::pair<StateId, bool> insert(const u64* w, u32 n, const u64* num = nullptr)
    {
        if (m_nn) [[unlikely]]
            return insert_impl<true>(w, n, num);
        return insert_impl<false>(w, n, nullptr);
    }
    std::pair<StateId, bool> insert(StateView s) { return insert(s.w, s.nw, s.num); }

    /// Id of s, or an invalid id.
    [[nodiscard]] StateId find(StateView s) const
    {
        const u32 n = bits::trimmed_size(s.w, s.nw);
        if (n > m_w)
            return StateId{};
        const u64 h = hash::state(s.w, n, s.num, m_nn);
        for (u64 j = h & m_mask;; j = (j + 1) & m_mask)
        {
            const u64 e = m_table[j];
            if (e == 0)
                return StateId{};
            const u32 id = static_cast<u32>(e) - 1;
            if ((e >> 32) == (h >> 32) && equal_record<true>(m_arena + static_cast<usize>(id) * m_rs, s.w, n, s.num))
                return StateId{id};
        }
    }

private:
    template<bool Numeric>
    std::pair<StateId, bool> insert_impl(const u64* w, u32 n, const u64* num)
    {
        n = bits::trimmed_size(w, n);
        if (n > m_w)
            widen(std::max(m_w * 2, n));
        if ((static_cast<u64>(m_count) + 1) * 10 > m_table.size() * 7)
            grow();
        // hash::state of the trimmed words
        const u64 h = Numeric ? hash::words(w, n, hash::words(num, m_nn)) : hash::words(w, n);
        const u64 tag = h >> 32;
        u64* table = m_table.data();
        for (u64 j = h & m_mask;; j = (j + 1) & m_mask)
        {
            const u64 e = table[j];
            if (e == 0)
            {
                if (m_count == ~u32{0} - 1)
                    throw std::length_error("FlatStateStore: more than 2^32 - 2 states");
                table[j] = (tag << 32) | (static_cast<u64>(m_count) + 1);
                u64* r = append_record();
                u32 i = 0;
                MYMYR_NOVECTOR
                for (; i < n; ++i)
                    r[i] = w[i];
                MYMYR_NOVECTOR
                for (; i < m_w; ++i)
                    r[i] = 0;
                if constexpr (Numeric)
                {
                    MYMYR_NOVECTOR
                    for (u32 j = 0; j < m_nn; ++j)
                        r[m_w + j] = num[j];
                }
                return {StateId{m_count++}, true};
            }
            if ((e >> 32) == tag)
            {
                const u32 id = static_cast<u32>(e) - 1;
                if (equal_record<Numeric>(m_arena + static_cast<usize>(id) * m_rs, w, n, num))
                    return {StateId{id}, false};
            }
        }
    }

public:
    /// The fluent words of a state: `stride()` words, zero-padded, then its numeric words. Invalidated by the next
    /// insert.
    [[nodiscard]] const u64* words(StateId id) const noexcept { return m_arena + static_cast<usize>(id.v) * m_rs; }
    [[nodiscard]] const u64* numeric(StateId id) const noexcept { return words(id) + m_w; }
    [[nodiscard]] StateView operator[](StateId id) const noexcept
    {
        return {words(id), m_w, m_nn ? numeric(id) : nullptr, m_nn};
    }
    [[nodiscard]] State state(StateId id) const { return State((*this)[id]); }

    [[nodiscard]] u32 size() const noexcept { return m_count; }
    /// Fluent words per record.
    [[nodiscard]] u32 stride() const noexcept { return m_w; }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_nn; }
    /// Words per record: stride() + numeric_words().
    [[nodiscard]] u32 record_words() const noexcept { return m_rs; }
    [[nodiscard]] u64 bytes() const noexcept { return m_cap * sizeof(u64) + m_table.size() * sizeof(u64); }

private:
    template<bool Numeric>
    [[nodiscard]] bool equal_record(const u64* r, const u64* w, u32 n, const u64* num) const noexcept
    {
        MYMYR_NOVECTOR
        for (u32 i = 0; i < n; ++i)
            if (r[i] != w[i])
                return false;
        MYMYR_NOVECTOR
        for (u32 i = n; i < m_w; ++i)
            if (r[i])
                return false;
        if constexpr (Numeric)
        {
            MYMYR_NOVECTOR
            for (u32 j = 0; j < m_nn; ++j)
                if (r[m_w + j] != num[j])
                    return false;
        }
        return true;
    }
    u64* append_record()
    {
        const usize off = static_cast<usize>(m_count) * m_rs;
        if (off + m_rs > m_cap)
            reserve_words(std::max<usize>(off + m_rs, m_cap * 2));
        return m_arena + off;
    }
    void reserve_words(usize cap)
    {
        cap = std::max<usize>(cap, 1024);
        auto* p = static_cast<u64*>(std::realloc(m_arena, cap * sizeof(u64)));
        if (!p)
            throw std::bad_alloc();
        m_arena = p;
        m_cap = cap;
    }
    void grow()
    {
        std::vector<u64> t(m_table.size() * 2, 0);
        const u64 mask = t.size() - 1;
        for (const u64 e : m_table)
        {
            if (!e)
                continue;
            const u32 id = static_cast<u32>(e) - 1;
            const u64* r = m_arena + static_cast<usize>(id) * m_rs;
            u64 j = hash::state(r, m_w, r + m_w, m_nn) & mask;
            while (t[j])
                j = (j + 1) & mask;
            t[j] = e;
        }
        m_table.swap(t);
        m_mask = mask;
    }
    void widen(u32 nw)
    {
        const u32 nrs = nw + m_nn;
        auto* a = static_cast<u64*>(std::calloc(std::max<usize>(1, static_cast<usize>(m_count) * nrs), sizeof(u64)));
        if (!a)
            throw std::bad_alloc();
        for (u32 i = 0; i < m_count; ++i)
        {
            const u64* src = m_arena + static_cast<usize>(i) * m_rs;
            u64* dst = a + static_cast<usize>(i) * nrs;
            std::memcpy(dst, src, m_w * sizeof(u64));
            if (m_nn)
                std::memcpy(dst + nw, src + m_w, m_nn * sizeof(u64));
        }
        std::free(m_arena);
        m_arena = a;
        m_cap = std::max<usize>(1, static_cast<usize>(m_count) * nrs);
        m_w = nw;
        m_rs = nrs;
    }

    u32 m_w;
    u32 m_nn = 0;  // numeric words per record
    u32 m_rs = 1;  // record stride: m_w + m_nn
    u64* m_arena = nullptr;
    usize m_cap = 0;  // words
    std::vector<u64> m_table;
    u64 m_mask;
    u32 m_count = 0;
};
}  // namespace mymyr
