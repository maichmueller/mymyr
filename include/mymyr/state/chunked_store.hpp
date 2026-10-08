#pragma once
// ChunkedStateStore: the default store above W = 8 words.
//   - a chunk is CW = 8 words (512 atoms, 64 B), interned by content in its own open-addressing table;
//   - a state is its vector of NC = W / CW chunk ids, deduplicated by a second table over the id vectors;
//   - a successor re-interns only the chunks its delta touches (insert_successor) and copies the other ids from
//     its parent, so dedup hashes NC ids instead of W words;
//   - wider states (lazy slots) extend every stored state with the id of the all-zero chunk (widen);
//   - numeric tasks: the numeric words are cut into chunks of their own (never mixed with bit words), which
//     follow the bit chunks in a state's id vector. A transition changes 1 to 3 slots, so a successor re-interns only
//     the numeric chunks whose content changed.
// Single-threaded, but for the read-only find_successor.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr
{
class ChunkedStateStore
{
public:
    static constexpr u32 k_chunk_words = 8;
    static constexpr u32 k_none = ~u32{0};

    explicit ChunkedStateStore(u32 words = k_chunk_words, u32 numeric_words = 0)
        : m_nc(std::max<u32>(1, (words + k_chunk_words - 1) / k_chunk_words)),
          m_nn(numeric_words),
          m_nnc((numeric_words + k_chunk_words - 1) / k_chunk_words),
          m_cslot(1u << 16, 0),
          m_cmask((1u << 16) - 1),
          m_sslot(1u << 16, 0),
          m_stag(1u << 16, 0),
          m_smask((1u << 16) - 1)
    {
    }

    /// Words per decoded state (a multiple of the chunk size).
    [[nodiscard]] u32 words() const noexcept { return m_nc * k_chunk_words; }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_nn; }
    /// Chunk ids per state: bit chunks, then numeric chunks.
    [[nodiscard]] u32 chunks_per_state() const noexcept { return m_nc + m_nnc; }
    [[nodiscard]] u32 size() const noexcept { return m_count; }
    [[nodiscard]] u32 num_chunks() const noexcept { return m_nchunks; }
    [[nodiscard]] u64 bytes() const noexcept
    {
        return m_chunks.capacity() * sizeof(u64) + m_cslot.size() * sizeof(u32) + m_ids.capacity() * sizeof(u32) +
               m_sslot.size() * 2 * sizeof(u32);
    }

    [[nodiscard]] const u64* chunk(u32 c) const noexcept { return m_chunks.data() + static_cast<usize>(c) * k_chunk_words; }
    [[nodiscard]] const u32* ids(StateId id) const noexcept
    {
        return m_ids.data() + static_cast<usize>(id.v) * (m_nc + m_nnc);
    }

    /// Interns one chunk (k_chunk_words words).
    u32 intern_chunk(const u64* w)
    {
        if ((static_cast<u64>(m_nchunks) + 1) * 10 > m_cslot.size() * 7)
        {
            std::vector<u32> ns(m_cslot.size() * 2, 0);
            const u64 m = ns.size() - 1;
            for (u32 c = 0; c < m_nchunks; ++c)
            {
                u64 j = hash::words(chunk(c), k_chunk_words) & m;
                while (ns[j])
                    j = (j + 1) & m;
                ns[j] = c + 1;
            }
            m_cslot.swap(ns);
            m_cmask = m;
        }
        for (u64 j = hash::words(w, k_chunk_words) & m_cmask;; j = (j + 1) & m_cmask)
        {
            const u32 v = m_cslot[j];
            if (!v)
            {
                m_cslot[j] = m_nchunks + 1;
                m_chunks.insert(m_chunks.end(), w, w + k_chunk_words);
                return m_nchunks++;
            }
            if (std::memcmp(chunk(v - 1), w, k_chunk_words * sizeof(u64)) == 0)
                return v - 1;
        }
    }

    /// Inserts a state given as chunks_per_state() chunk ids.
    std::pair<StateId, bool> insert_ids(const u32* x)
    {
        const u32 nc = chunks_per_state();  // every id of the state
        if ((static_cast<u64>(m_count) + 1) * 10 > m_sslot.size() * 7)
        {
            m_sslot.assign(m_sslot.size() * 2, 0);
            m_stag.assign(m_sslot.size(), 0);
            rehash_states();
        }
        const u64 h = hash_ids(x, nc);
        const u32 tag = static_cast<u32>(h >> 32);
        for (u64 j = h & m_smask;; j = (j + 1) & m_smask)
        {
            const u32 v = m_sslot[j];
            if (!v)
            {
                if (m_count == ~u32{0} - 1)
                    throw std::length_error("ChunkedStateStore: more than 2^32 - 2 states");
                m_sslot[j] = m_count + 1;
                m_stag[j] = tag;
                m_ids.insert(m_ids.end(), x, x + nc);
                return {StateId{m_count++}, true};
            }
            if (m_stag[j] == tag && std::memcmp(ids(StateId{v - 1}), x, nc * sizeof(u32)) == 0)
                return {StateId{v - 1}, false};
        }
    }

    /// Inserts a state given as words (any length; missing words are zero) and, numeric tasks, its numeric words,
    /// interning every chunk.
    std::pair<StateId, bool> insert(const u64* w, u32 n, const u64* num = nullptr)
    {
        n = bits::trimmed_size(w, n);
        if (n > words())
            widen(n);
        m_scratch_w.assign(words() + m_nnc * k_chunk_words, 0);
        if (n)
            std::memcpy(m_scratch_w.data(), w, n * sizeof(u64));
        if (m_nn && num)
            std::memcpy(m_scratch_w.data() + words(), num, m_nn * sizeof(u64));
        m_scratch_ids.resize(m_nc + m_nnc);
        for (u32 c = 0; c < m_nc + m_nnc; ++c)
            m_scratch_ids[c] = intern_chunk(m_scratch_w.data() + static_cast<usize>(c) * k_chunk_words);
        return insert_ids(m_scratch_ids.data());
    }
    std::pair<StateId, bool> insert(StateView s) { return insert(s.w, s.nw, s.num); }

    /// Inserts `next` (words() words), the successor of stored state `parent` whose decoded words are `cur`, under
    /// delta d: only the chunks the delta touches are compared and re-interned. The caller widens first if an added
    /// slot lies beyond words().
    std::pair<StateId, bool> insert_successor(StateId parent, const u64* cur, const u64* next, const Delta& d)
    {
        m_scratch_ids.assign(ids(parent), ids(parent) + m_nc + m_nnc);
        auto touch = [&](u32 slot)
        {
            const u32 ci = slot / (64 * k_chunk_words);
            if (m_scratch_ids[ci] != k_none &&
                std::memcmp(cur + static_cast<usize>(ci) * k_chunk_words, next + static_cast<usize>(ci) * k_chunk_words,
                            k_chunk_words * sizeof(u64)) != 0)
                m_scratch_ids[ci] = k_none;  // re-interned below
        };
        for (SlotId x : d.del)
            if (x.v < words() * 64)
                touch(x.v);
        for (SlotId x : d.add)
            touch(x.v);
        for (u32 ci = 0; ci < m_nc; ++ci)
            if (m_scratch_ids[ci] == k_none)
                m_scratch_ids[ci] = intern_chunk(next + static_cast<usize>(ci) * k_chunk_words);
        if (m_nnc && d.num)
        {
            // numeric chunks: re-intern the ones whose words changed (the parent's are decoded from its ids)
            u64 buf[k_chunk_words];
            for (u32 c = 0; c < m_nnc; ++c)
            {
                const u32 lo = c * k_chunk_words, len = std::min(k_chunk_words, m_nn - lo);
                const u64* pc = chunk(m_scratch_ids[m_nc + c]);
                if (std::memcmp(pc, d.num + lo, len * sizeof(u64)) == 0)
                    continue;
                std::memset(buf, 0, sizeof(buf));
                std::memcpy(buf, d.num + lo, len * sizeof(u64));
                m_scratch_ids[m_nc + c] = intern_chunk(buf);
            }
        }
        return insert_ids(m_scratch_ids.data());
    }

    /// The id of `next` (its first n words; the rest are zero), the successor of stored state `parent` whose decoded
    /// words are `cur` (words() words), under delta d, or an invalid id if it is not stored. Read-only: calls from
    /// several threads are safe while nothing is inserted. `x` is scratch of chunks_per_state() ids.
    [[nodiscard]] StateId find_successor(StateId parent, const u64* cur, const u64* next, u32 n, const Delta& d, u32* x) const
    {
        if (bits::trimmed_size(next, n) > words())
            return StateId{};
        const u32 nc = m_nc + m_nnc;
        std::memcpy(x, ids(parent), nc * sizeof(u32));
        u64 buf[k_chunk_words];
        auto touch = [&](u32 slot)
        {
            const u32 ci = slot / (64 * k_chunk_words);
            if (ci >= m_nc || x[ci] == k_none)
                return;
            const u32 lo = ci * k_chunk_words;
            std::memset(buf, 0, sizeof(buf));
            if (lo < n)
                std::memcpy(buf, next + lo, std::min(k_chunk_words, n - lo) * sizeof(u64));
            if (std::memcmp(cur + lo, buf, sizeof(buf)) != 0)
                x[ci] = k_none;  // looked up below
        };
        for (SlotId s : d.del)
            touch(s.v);
        for (SlotId s : d.add)
            touch(s.v);
        for (u32 ci = 0; ci < m_nc; ++ci)
            if (x[ci] == k_none)
            {
                const u32 lo = ci * k_chunk_words;
                std::memset(buf, 0, sizeof(buf));
                if (lo < n)
                    std::memcpy(buf, next + lo, std::min(k_chunk_words, n - lo) * sizeof(u64));
                x[ci] = find_chunk(buf);
                if (x[ci] == k_none)
                    return StateId{};  // a new chunk: a new state
            }
        if (m_nnc && d.num)
            for (u32 c = 0; c < m_nnc; ++c)
            {
                const u32 lo = c * k_chunk_words, len = std::min(k_chunk_words, m_nn - lo);
                if (std::memcmp(chunk(x[m_nc + c]), d.num + lo, len * sizeof(u64)) == 0)
                    continue;
                std::memset(buf, 0, sizeof(buf));
                std::memcpy(buf, d.num + lo, len * sizeof(u64));
                x[m_nc + c] = find_chunk(buf);
                if (x[m_nc + c] == k_none)
                    return StateId{};
            }
        const u64 h = hash_ids(x, nc);
        const u32 tag = static_cast<u32>(h >> 32);
        for (u64 j = h & m_smask;; j = (j + 1) & m_smask)
        {
            const u32 v = m_sslot[j];
            if (!v)
                return StateId{};
            if (m_stag[j] == tag && std::memcmp(ids(StateId{v - 1}), x, nc * sizeof(u32)) == 0)
                return StateId{v - 1};
        }
    }

    /// Writes the words() words of a state and, numeric tasks, its numeric_words() numeric words into `num`.
    void decode(StateId id, u64* out, u64* num = nullptr) const
    {
        const u32* x = ids(id);
        for (u32 c = 0; c < m_nc; ++c)
            std::memcpy(out + static_cast<usize>(c) * k_chunk_words, chunk(x[c]), k_chunk_words * sizeof(u64));
        if (num)
            for (u32 c = 0; c < m_nnc; ++c)
            {
                const u32 lo = c * k_chunk_words;
                std::memcpy(num + lo, chunk(x[m_nc + c]), std::min(k_chunk_words, m_nn - lo) * sizeof(u64));
            }
    }
    [[nodiscard]] State state(StateId id) const
    {
        std::vector<u64> w(words()), num(m_nn);
        decode(id, w.data(), num.data());
        return State(w.data(), words(), num.data(), m_nn);
    }

    /// Makes room for states of at least `nw` words: every stored state gets zero chunks appended.
    void widen(u32 nw)
    {
        const u32 nnc = (nw + k_chunk_words - 1) / k_chunk_words;
        if (nnc <= m_nc)
            return;
        const u64 zero[k_chunk_words] = {};
        const u32 z = intern_chunk(zero);
        const u32 per = nnc + m_nnc;
        std::vector<u32> nids(static_cast<usize>(m_count) * per, z);
        for (u32 i = 0; i < m_count; ++i)
        {
            std::memcpy(nids.data() + static_cast<usize>(i) * per, ids(StateId{i}), m_nc * sizeof(u32));
            if (m_nnc)
                std::memcpy(nids.data() + static_cast<usize>(i) * per + nnc, ids(StateId{i}) + m_nc, m_nnc * sizeof(u32));
        }
        m_ids.swap(nids);
        m_nc = nnc;
        rehash_states();
    }

private:
    [[nodiscard]] u32 find_chunk(const u64* w) const noexcept
    {
        for (u64 j = hash::words(w, k_chunk_words) & m_cmask;; j = (j + 1) & m_cmask)
        {
            const u32 v = m_cslot[j];
            if (!v)
                return k_none;
            if (std::memcmp(chunk(v - 1), w, k_chunk_words * sizeof(u64)) == 0)
                return v - 1;
        }
    }
    static u64 hash_ids(const u32* x, u32 n)
    {
        u64 h = 0x9E3779B97F4A7C15ULL ^ n;
        for (u32 i = 0; i < n; ++i)
        {
            h ^= x[i];
            h *= 0xBF58476D1CE4E5B9ULL;
            h ^= h >> 31;
        }
        return h ^ (h >> 29);
    }
    void rehash_states()
    {
        std::fill(m_sslot.begin(), m_sslot.end(), 0);
        const u64 m = m_sslot.size() - 1;
        for (u32 i = 0; i < m_count; ++i)
        {
            const u64 h = hash_ids(ids(StateId{i}), m_nc + m_nnc);
            u64 j = h & m;
            while (m_sslot[j])
                j = (j + 1) & m;
            m_sslot[j] = i + 1;
            m_stag[j] = static_cast<u32>(h >> 32);
        }
        m_smask = m;
    }

    u32 m_nc;
    u32 m_nn = 0;   // numeric words
    u32 m_nnc = 0;  // numeric chunks
    std::vector<u64> m_chunks;  // chunk arena
    std::vector<u32> m_cslot;   // open addressing: chunk id + 1
    u64 m_cmask;
    u32 m_nchunks = 0;
    std::vector<u32> m_ids;  // state arena: m_nc chunk ids per state
    std::vector<u32> m_sslot, m_stag;
    u64 m_smask;
    u32 m_count = 0;
    std::vector<u64> m_scratch_w;
    std::vector<u32> m_scratch_ids;
};
}  // namespace mymyr
