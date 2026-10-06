#pragma once
// ConcurrentStateStore: the one concurrent store code path (dedup "cas", variable-width records):
//   - per-thread arenas with stable addresses. A state is a record [len, words...] with trailing zero words trimmed
//     (lazy slots let W grow; missing words compare as zero), addressed by a 32-bit handle (thread, local index);
//   - one lock-free open-addressing table of 64-bit entries (tag32 << 32 | handle). A thread probes read-only, writes
//     its candidate record at its own arena's end only when it meets an empty entry, publishes it with a CAS, and
//     simply reuses the record space when it loses the race to an equal state;
//   - growth by a stop-the-world parallel rehash (rehash(team)) that the search runs between frontier chunks once
//     wants_rehash() reports a load factor above 60%;
//   - a per-record discoverer key for deterministic ids: new records carry the inserting thread's key and a
//     duplicate found in the current layer lowers it by an atomic min. The search sorts each layer by key, so the
//     resulting ids do not depend on thread scheduling.
// Numeric tasks: a record is [len, bits..., numeric words...]; the numeric words (a fixed count per task) follow
// the trimmed bits, the hash is hash::state and equality compares them bitwise.
// Every thread t in [0, threads) may call insert(t, ...) concurrently with the others; nothing else is concurrent.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/team.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr
{
class ConcurrentStateStore
{
public:
    using Handle = u32;

    explicit ConcurrentStateStore(u32 threads, u32 numeric_words = 0) : m_T(threads == 0 ? 1 : threads), m_nn(numeric_words)
    {
        u32 tb = 0;
        while ((u32{1} << tb) < m_T)
            ++tb;
        m_lb = 32 - tb;
        m_limask = m_lb == 32 ? ~u64{0} >> 32 : (u64{1} << m_lb) - 1;
        m_arenas.reserve(m_T);
        for (u32 t = 0; t < m_T; ++t)
            m_arenas.push_back(std::make_unique<Arena>(m_limask + 1));
        u64 cap = u64{1} << 18;
        while (cap < (static_cast<u64>(m_T) << 15))
            cap *= 2;
        alloc_table(cap);
    }
    ConcurrentStateStore(const ConcurrentStateStore&) = delete;
    ConcurrentStateStore& operator=(const ConcurrentStateStore&) = delete;
    ~ConcurrentStateStore() { std::free(m_tab); }

    [[nodiscard]] u32 threads() const noexcept { return m_T; }

    /// Thread t inserts the state w[0, n). `key` is the discoverer key of a new state; a duplicate found among the
    /// records of the current layer lowers the stored key to `key` if smaller. Returns (handle, inserted).
    std::pair<Handle, bool> insert(u32 t, const u64* w, u32 n, u64 key, const u64* num = nullptr)
    {
        n = bits::trimmed_size(w, n);
        const u64 h = hash::state(w, n, num, m_nn);
        const u32 tag = static_cast<u32>(h >> 32) | 1u;
        Arena& A = *m_arenas[t];
        bool written = false;
        std::atomic<u64>* S = m_tab;
        const u64 mask = m_cap - 1;
        u64 probes = 0;
        for (u64 j = h & mask;; j = (j + 1) & mask)
        {
            u64 v = S[j].load(std::memory_order_acquire);
            if (v == 0)
            {
                if (!written)
                {
                    A.write(w, n, num, m_nn, key);
                    written = true;
                }
                const Handle nh = make_handle(t, A.n);
                const u64 nv = (static_cast<u64>(tag) << 32) | nh;
                if (S[j].compare_exchange_strong(v, nv, std::memory_order_acq_rel, std::memory_order_acquire))
                {
                    A.commit();
                    if (++A.local_new == 64)
                    {
                        m_approx.fetch_add(64, std::memory_order_relaxed);
                        A.local_new = 0;
                    }
                    return {nh, true};
                }
            }
            if (static_cast<u32>(v >> 32) == tag && equal(static_cast<Handle>(v), w, n, num))
            {
                on_duplicate(static_cast<Handle>(v), key);
                return {static_cast<Handle>(v), false};
            }
            if (++probes > mask)
                throw std::length_error("ConcurrentStateStore: table full");
        }
    }

    /// The record of a handle (trimmed words, then the numeric words).
    [[nodiscard]] StateView record(Handle h) const noexcept
    {
        const u64* r = m_arenas[thread_of(h)]->record(local_of(h));
        return {r + 1, static_cast<u32>(r[0]), m_nn ? r + 1 + r[0] : nullptr, m_nn};
    }
    [[nodiscard]] u64 key(Handle h) const noexcept
    {
        return m_arenas[thread_of(h)]->key(local_of(h)).load(std::memory_order_relaxed);
    }

    [[nodiscard]] Handle make_handle(u32 t, u32 li) const noexcept
    {
        return static_cast<Handle>((static_cast<u64>(t) << m_lb) | li);
    }
    [[nodiscard]] u32 thread_of(Handle h) const noexcept { return m_lb == 32 ? 0 : static_cast<u32>(h >> m_lb); }
    [[nodiscard]] u32 local_of(Handle h) const noexcept { return static_cast<u32>(h & m_limask); }

    /// Records committed by thread t so far, and the first one of the current layer.
    [[nodiscard]] u32 local_size(u32 t) const noexcept { return m_arenas[t]->n; }
    [[nodiscard]] u32 layer_start(u32 t) const noexcept { return m_arenas[t]->layer_start; }
    /// Starts a new layer: duplicates of records inserted before no longer update keys. Not concurrent.
    void begin_layer() noexcept
    {
        for (auto& a : m_arenas)
            a->layer_start = a->n;
    }

    /// Approximate number of states (lags by up to 64 per thread).
    [[nodiscard]] u64 approx_size() const noexcept { return m_approx.load(std::memory_order_relaxed); }
    [[nodiscard]] bool wants_rehash() const noexcept { return approx_size() * 10 > m_cap * 6; }
    [[nodiscard]] u64 capacity() const noexcept { return m_cap; }

    /// Doubles the table (stop the world: no insert may run). Every team member rehashes a slice.
    void rehash(Team& team)
    {
        const u64 nc = m_cap * 2;
        std::atomic<u64>* nt = alloc(nc);
        std::atomic<u64>* old = m_tab;
        const u64 ocap = m_cap;
        team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(ocap, t, team.size());
                for (u64 i = a; i < b; ++i)
                    if (const u64 v = old[i].load(std::memory_order_relaxed))
                    {
                        const StateView r = record(static_cast<Handle>(v));
                        for (u64 j = hash::state(r.w, r.nw, r.num, r.nnum) & (nc - 1);; j = (j + 1) & (nc - 1))
                        {
                            u64 e = 0;
                            if (nt[j].compare_exchange_strong(e, v, std::memory_order_relaxed))
                                break;
                        }
                    }
            });
        std::free(old);
        m_tab = nt;
        m_cap = nc;
    }

    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 b = m_cap * sizeof(u64);
        for (const auto& a : m_arenas)
            b += a->bytes;
        return b;
    }

private:
    // Per-thread arena of variable-width records [len, words...].
    struct alignas(64) Arena
    {
        static constexpr u32 SB = 14, BS = 1u << SB, MASK = BS - 1;
        static constexpr u32 WB = 1u << 18;  // words per word-stream block

        std::vector<u64**> iblk;               // per record: pointer to [len, words...]
        std::vector<std::atomic<u64>*> kblk;   // per record: discoverer key
        std::vector<u64*> wblk;                // word-stream blocks
        u64* wcur = nullptr;
        u32 wfill = WB, pend = 0;
        u32 n = 0;            // committed records
        u32 layer_start = 0;  // first record of the current layer
        u32 local_new = 0;
        u64 bytes = 0;

        explicit Arena(u64 max_local)
        {
            const u64 nb = max_local / BS + 2;
            iblk.assign(nb, nullptr);
            kblk.assign(nb, nullptr);
        }
        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;
        ~Arena()
        {
            for (auto* p : iblk)
                delete[] p;
            for (auto* p : kblk)
                delete[] p;
            for (auto* p : wblk)
                delete[] p;
        }
        [[nodiscard]] const u64* record(u32 li) const noexcept { return iblk[li >> SB][li & MASK]; }
        [[nodiscard]] std::atomic<u64>& key(u32 li) const noexcept { return kblk[li >> SB][li & MASK]; }
        // Writes a candidate record at local index n (not yet counted; commit() makes it count).
        void write(const u64* s, u32 len, const u64* num, u32 nn, u64 k)
        {
            const u32 li = n;
            const u32 b = li >> SB;
            if (b >= kblk.size())
                throw std::length_error("ConcurrentStateStore: thread arena full");
            if (!kblk[b])
            {
                kblk[b] = new std::atomic<u64>[BS];
                iblk[b] = new u64*[BS];
                bytes += BS * (sizeof(u64) + sizeof(u64*));
            }
            if (wfill + len + nn + 1 > WB)
            {
                wcur = new u64[WB];
                wblk.push_back(wcur);
                bytes += WB * sizeof(u64);
                wfill = 0;
            }
            u64* h = wcur + wfill;
            h[0] = len;
            if (len)
                std::memcpy(h + 1, s, len * sizeof(u64));
            if (nn && num)
                std::memcpy(h + 1 + len, num, nn * sizeof(u64));
            iblk[b][li & MASK] = h;
            pend = len + nn + 1;
            kblk[b][li & MASK].store(k, std::memory_order_relaxed);
        }
        void commit() noexcept
        {
            wfill += pend;
            ++n;
        }
    };

    [[nodiscard]] bool equal(Handle h, const u64* w, u32 n, const u64* num) const noexcept
    {
        const StateView r = record(h);
        return r.nw == n && (n == 0 || std::memcmp(r.w, w, n * sizeof(u64)) == 0) &&
               (m_nn == 0 || !num || std::memcmp(r.num, num, m_nn * sizeof(u64)) == 0);
    }
    void on_duplicate(Handle h, u64 k) const noexcept
    {
        const Arena& a = *m_arenas[thread_of(h)];
        const u32 li = local_of(h);
        if (li < a.layer_start)
            return;  // found in an earlier layer
        std::atomic<u64>& x = a.key(li);
        u64 cur = x.load(std::memory_order_relaxed);
        while (k < cur && !x.compare_exchange_weak(cur, k, std::memory_order_relaxed))
        {
        }
    }
    static std::atomic<u64>* alloc(u64 cap)
    {
        auto* p = static_cast<std::atomic<u64>*>(std::calloc(cap, sizeof(u64)));
        if (!p)
            throw std::bad_alloc();
        return p;
    }
    void alloc_table(u64 cap)
    {
        m_tab = alloc(cap);
        m_cap = cap;
    }

    u32 m_T;
    u32 m_nn = 0;  // numeric words per record
    u32 m_lb = 32;
    u64 m_limask = 0;
    std::vector<std::unique_ptr<Arena>> m_arenas;
    std::atomic<u64>* m_tab = nullptr;
    u64 m_cap = 0;
    std::atomic<u64> m_approx{0};
};
}  // namespace mymyr
