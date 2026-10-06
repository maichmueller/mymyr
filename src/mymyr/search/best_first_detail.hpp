#pragma once
// Internals of the best-first searches (astar.cpp, gbfs.cpp, beam.cpp): open lists, store adapters, SoA search nodes
// and the context every search shares (heuristic, action costs, goal test, blocked states, budgets, observer, plans).

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/state/chunked_store.hpp"
#include "mymyr/state/compact_store.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mymyr::search::bf
{
using Clock = std::chrono::steady_clock;
inline constexpr u32 k_none = ~u32{0};

[[nodiscard]] inline double seconds_since(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ================================================================================================ open lists
/// Binary min-heap over (p, s, insertion order). The order is total, so pops do not depend on the standard library.
class HeapQueue
{
public:
    static constexpr const char* k_name = "heap";
    /// The key of a value: its IEEE-754 bit pattern mapped so that unsigned order is the order of the values (g can
    /// decrease on numeric tasks whose total-cost effects decrease it); -0 and +0 share a key.
    [[nodiscard]] static u64 key(f64 v) noexcept
    {
        const u64 b = std::bit_cast<u64>(v + 0.0);
        return (b >> 63) ? ~b : (b | (u64{1} << 63));
    }

    void push(u64 p, u64 s, u32 ref)
    {
        m_h.push_back({p, s, m_seq++, ref});
        usize i = m_h.size() - 1;
        const Entry e = m_h[i];
        while (i > 0)
        {
            const usize q = (i - 1) / 2;
            if (!less(e, m_h[q]))
                break;
            m_h[i] = m_h[q];
            i = q;
        }
        m_h[i] = e;
    }
    [[nodiscard]] bool empty() const noexcept { return m_h.empty(); }
    [[nodiscard]] usize size() const noexcept { return m_h.size(); }
    [[nodiscard]] u64 top_key() noexcept { return m_h.front().p; }
    u32 pop(u64& p)
    {
        const Entry top = m_h.front();
        const Entry e = m_h.back();
        m_h.pop_back();
        const usize n = m_h.size();
        if (n > 0)
        {
            usize i = 0;
            for (;;)
            {
                usize c = 2 * i + 1;
                if (c >= n)
                    break;
                if (c + 1 < n && less(m_h[c + 1], m_h[c]))
                    ++c;
                if (!less(m_h[c], e))
                    break;
                m_h[i] = m_h[c];
                i = c;
            }
            m_h[i] = e;
        }
        p = top.p;
        return top.ref;
    }
    [[nodiscard]] u64 bytes() const noexcept { return m_h.capacity() * sizeof(Entry); }

private:
    struct Entry
    {
        u64 p, s, seq;
        u32 ref;
    };
    [[nodiscard]] static bool less(const Entry& a, const Entry& b) noexcept
    {
        if (a.p != b.p)
            return a.p < b.p;
        if (a.s != b.s)
            return a.s < b.s;
        return a.seq < b.seq;
    }
    std::vector<Entry> m_h;
    u64 m_seq = 0;
};

/// Two-level bucket queue over small integer keys (p, s), first in first out within a bucket: the same order as
/// HeapQueue for integral keys.
class BucketQueue
{
public:
    static constexpr const char* k_name = "bucket";
    static constexpr u64 k_max_key = u64{1} << 22;
    [[nodiscard]] static u64 key(f64 v)
    {
        const f64 r = v >= 0 ? std::floor(v) : -1;
        if (r != v || r >= static_cast<f64>(k_max_key))
            throw std::domain_error("mymyr best-first search: a bucket queue key is not a small non-negative integer "
                                    "(use BestFirstOptions::Queue::Heap)");
        return static_cast<u64>(r);
    }

    void push(u64 p, u64 s, u32 ref)
    {
        if (p >= m_outer.size())
            m_outer.resize(p + 1);
        Inner& in = m_outer[p];
        if (s >= in.fifo.size())
            in.fifo.resize(s + 1);
        in.fifo[s].v.push_back(ref);
        if (in.count++ == 0 || s < in.cur)
            in.cur = s;
        if (m_count++ == 0 || p < m_cur)
            m_cur = p;
    }
    [[nodiscard]] bool empty() const noexcept { return m_count == 0; }
    [[nodiscard]] usize size() const noexcept { return m_count; }
    [[nodiscard]] u64 top_key() noexcept
    {
        while (m_outer[m_cur].count == 0)
            ++m_cur;
        return m_cur;
    }
    u32 pop(u64& p)
    {
        p = top_key();
        Inner& in = m_outer[m_cur];
        while (in.fifo[in.cur].head == in.fifo[in.cur].v.size())
            ++in.cur;
        Fifo& f = in.fifo[in.cur];
        const u32 r = f.v[f.head++];
        if (f.head == f.v.size())
        {
            f.v.clear();
            f.head = 0;
        }
        else if (f.head >= 4096 && 2 * static_cast<usize>(f.head) >= f.v.size())
        {
            f.v.erase(f.v.begin(), f.v.begin() + f.head);
            f.head = 0;
        }
        --in.count;
        --m_count;
        return r;
    }
    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 b = m_outer.capacity() * sizeof(Inner);
        for (const Inner& in : m_outer)
        {
            b += in.fifo.capacity() * sizeof(Fifo);
            for (const Fifo& f : in.fifo)
                b += f.v.capacity() * sizeof(u32);
        }
        return b;
    }

private:
    struct Fifo
    {
        std::vector<u32> v;
        u32 head = 0;
    };
    struct Inner
    {
        std::vector<Fifo> fifo;
        u64 cur = 0;
        u64 count = 0;
    };
    std::vector<Inner> m_outer;
    u64 m_cur = 0;
    u64 m_count = 0;
};

/// As in mimir's AlternatingOpenList, over a preferred (0) and a standard (1) list: up to `weight` pops from the current
/// list, then the next non-empty one.
template<class Q>
class Alternating
{
public:
    Alternating(u32 preferred_weight, u32 standard_weight) : m_w{preferred_weight, standard_weight} {}
    [[nodiscard]] Q& list(u32 i) noexcept { return m_q[i]; }
    [[nodiscard]] bool empty() const noexcept { return m_q[0].empty() && m_q[1].empty(); }
    [[nodiscard]] usize size() const noexcept { return m_q[0].size() + m_q[1].size(); }
    /// Pops the next entry; `from` is the list it came from.
    u32 pop(u64& p, u32& from)
    {
        if (m_q[m_pos].empty() || m_count >= m_w[m_pos])
        {
            do
                m_pos ^= 1;
            while (m_q[m_pos].empty());
            m_count = 0;
        }
        ++m_count;
        from = m_pos;
        return m_q[m_pos].pop(p);
    }
    /// Smallest key over both lists (not empty).
    [[nodiscard]] u64 min_key()
    {
        if (m_q[0].empty())
            return m_q[1].top_key();
        if (m_q[1].empty())
            return m_q[0].top_key();
        return std::min(m_q[0].top_key(), m_q[1].top_key());
    }
    [[nodiscard]] u64 bytes() const noexcept { return m_q[0].bytes() + m_q[1].bytes(); }

private:
    Q m_q[2];
    u32 m_w[2];
    u32 m_pos = 0;
    u32 m_count = 0;
};

// ================================================================================================ store adapters
/// The state being expanded: its fluent words (trimmed length n) and, numeric tasks, its numeric words.
struct Cur
{
    std::vector<u64> w, num;
    u32 n = 0;
    [[nodiscard]] StateView view() const noexcept
    {
        return {w.data(), n, num.empty() ? nullptr : num.data(), static_cast<u32>(num.size())};
    }
};

// A search refers to open states by a u32 `ref`: the state id for Flat and Chunked, an open-list record (the state's
// words) for Compact. Ids are dense and assigned in insertion order by every adapter (the start state is 0). Numeric
// tasks: every state also carries the task's numeric words (states that differ only in their values are distinct).
//
//   insert(s)                             -> {id, new}      a state given by its view (the start state)
//   load(ref, cur)                        -> id             the words of an open state into cur
//   release(ref)                                            the open entry is done (Compact frees the record)
//   make_ref(id, s)                       -> ref            an open entry for state id with view s
//   successor(cur, d, next, succ)         -> nn             the successor's fluent words under delta d into next
//   insert_successor(parent, cur, next, nn, d) -> {id, new} (the successor's numeric words are d.num)

class FlatAdapter
{
public:
    static constexpr const char* k_name = "flat";
    static constexpr bool k_needs_words = false;
    explicit FlatAdapter(const Task& task) : m_store(std::max<u32>(1, task.words()), 16, task.numeric_words()) {}

    std::pair<u32, bool> insert(StateView s)
    {
        const auto [id, fresh] = m_store.insert(s.w, s.nw, s.num);
        return {id.v, fresh};
    }
    [[nodiscard]] u32 id_of(u32 ref) const noexcept { return ref; }
    u32 load(u32 ref, Cur& cur)
    {
        const u32 W = m_store.stride();
        const u64* rec = m_store.words(StateId{ref});
        cur.w.assign(rec, rec + W);  // the arena may move while the state is expanded
        cur.num.assign(rec + W, rec + W + m_store.numeric_words());
        cur.n = bits::trimmed_size(cur.w.data(), W);
        return ref;
    }
    void release(u32) noexcept {}
    u32 make_ref(u32 id, StateView) noexcept { return id; }
    u32 successor(Cur& cur, const Delta& d, std::vector<u64>& next, Successors&)
    {
        return apply_delta(cur.w.data(), cur.n, d, next);
    }
    std::pair<u32, bool> insert_successor(u32, const Cur&, const u64* next, u32 nn, const Delta& d)
    {
        const auto [id, fresh] = m_store.insert(next, nn, d.num);
        return {id.v, fresh};
    }
    [[nodiscard]] u32 size() const noexcept { return m_store.size(); }
    [[nodiscard]] u64 bytes() const noexcept { return m_store.bytes(); }

private:
    FlatStateStore m_store;
};

class ChunkedAdapter
{
public:
    static constexpr const char* k_name = "chunked";
    static constexpr bool k_needs_words = false;
    explicit ChunkedAdapter(const Task& task) : m_store(std::max<u32>(1, task.words()), task.numeric_words()) {}

    std::pair<u32, bool> insert(StateView s)
    {
        const auto [id, fresh] = m_store.insert(s.w, s.nw, s.num);
        return {id.v, fresh};
    }
    [[nodiscard]] u32 id_of(u32 ref) const noexcept { return ref; }
    u32 load(u32 ref, Cur& cur)
    {
        const u32 W = m_store.words();
        cur.w.assign(W, 0);
        cur.num.assign(m_store.numeric_words(), 0);
        m_store.decode(StateId{ref}, cur.w.data(), cur.num.data());
        cur.n = bits::trimmed_size(cur.w.data(), W);
        return ref;
    }
    void release(u32) noexcept {}
    u32 make_ref(u32 id, StateView) noexcept { return id; }
    u32 successor(Cur& cur, const Delta& d, std::vector<u64>& next, Successors& succ)
    {
        u32 W = m_store.words();
        u32 need = 0;
        MYMYR_NOVECTOR
        for (SlotId a : d.add)
            need = std::max<u32>(need, bits::word_of(a.v) + 1);
        if (need > W || cur.w.size() < W)
        {
            // more atoms were interned (lazy slots): extend every state with zero chunks
            if (need > W)
                m_store.widen(std::max(W * 2, need));
            W = m_store.words();
            cur.w.resize(W, 0);
            succ.engine().set_state(cur.w.data(), cur.n);  // cur may have moved; the view is unchanged
        }
        next.resize(W);
        std::memcpy(next.data(), cur.w.data(), W * sizeof(u64));
        for (SlotId x : d.del)
            if (bits::word_of(x.v) < W)
                bits::reset(next.data(), x.v);
        for (SlotId x : d.add)
            bits::set(next.data(), x.v);
        return W;
    }
    std::pair<u32, bool> insert_successor(u32 parent, const Cur& cur, const u64* next, u32, const Delta& d)
    {
        const auto [id, fresh] = m_store.insert_successor(StateId{parent}, cur.w.data(), next, d);
        return {id.v, fresh};
    }
    [[nodiscard]] u32 size() const noexcept { return m_store.size(); }
    [[nodiscard]] u64 bytes() const noexcept { return m_store.bytes(); }

private:
    ChunkedStateStore m_store;
};

/// Fingerprint -> dense id (open addressing over (a, b, id)); the fingerprint (0, 0) is kept aside.
class FingerprintMap
{
public:
    FingerprintMap() : m_a(1u << 16, 0), m_b(1u << 16, 0), m_id(1u << 16, 0), m_mask((1u << 16) - 1) {}

    std::pair<u32, bool> insert(Fingerprint128 f)
    {
        if (f.a == 0 && f.b == 0)
        {
            if (m_zero != k_none)
                return {m_zero, false};
            m_zero = m_count++;
            return {m_zero, true};
        }
        if ((m_used + 1) * 10 > m_a.size() * 7)
            grow();
        for (u64 j = hash::mix64(f.a) & m_mask;; j = (j + 1) & m_mask)
        {
            if (!(m_a[j] | m_b[j]))
            {
                if (m_count == k_none - 1)
                    throw std::length_error("mymyr best-first search: more than 2^32 - 2 states");
                m_a[j] = f.a;
                m_b[j] = f.b;
                m_id[j] = m_count;
                ++m_used;
                return {m_count++, true};
            }
            if (m_a[j] == f.a && m_b[j] == f.b)
                return {m_id[j], false};
        }
    }
    [[nodiscard]] u32 size() const noexcept { return m_count; }
    [[nodiscard]] u64 bytes() const noexcept { return m_a.capacity() * 16 + m_id.capacity() * 4; }

private:
    void grow()
    {
        const usize n = m_a.size() * 2;
        std::vector<u64> na(n, 0), nb(n, 0);
        std::vector<u32> ni(n, 0);
        const u64 mask = n - 1;
        for (usize i = 0; i < m_a.size(); ++i)
            if (m_a[i] | m_b[i])
            {
                u64 j = hash::mix64(m_a[i]) & mask;
                while (na[j] | nb[j])
                    j = (j + 1) & mask;
                na[j] = m_a[i];
                nb[j] = m_b[i];
                ni[j] = m_id[i];
            }
        m_a.swap(na);
        m_b.swap(nb);
        m_id.swap(ni);
        m_mask = mask;
    }
    std::vector<u64> m_a, m_b;
    std::vector<u32> m_id;
    u64 m_mask;
    u64 m_used = 0;
    u32 m_count = 0;
    u32 m_zero = k_none;
};

/// Closed states as 128-bit Zobrist fingerprints (keys from canonical atom ids, as the compact BrFS; numeric tasks also
/// mix in a hash of the numeric words); every open-list entry owns a record with its state's words.
class CompactAdapter
{
public:
    static constexpr const char* k_name = "compact";
    static constexpr bool k_needs_words = true;
    explicit CompactAdapter(const Task& task)
        : m_atoms(task.atoms()), m_w(std::max<u32>(1, task.words())), m_nn(task.numeric_words())
    {
        ensure_keys();
    }

    std::pair<u32, bool> insert(StateView s) { return m_map.insert(with_numeric(fingerprint(s.w, s.nw), s.num)); }
    [[nodiscard]] u32 id_of(u32 ref) const noexcept { return static_cast<u32>(m_rec[static_cast<usize>(ref) * stride()]); }
    u32 load(u32 ref, Cur& cur)
    {
        const u64* r = m_rec.data() + static_cast<usize>(ref) * stride();
        cur.w.assign(r + 1, r + 1 + m_w);
        cur.num.assign(r + 1 + m_w, r + 1 + m_w + m_nn);
        cur.n = bits::trimmed_size(cur.w.data(), m_w);
        m_cur_fp = fingerprint(cur.w.data(), cur.n);
        return static_cast<u32>(r[0]);
    }
    void release(u32 ref) { m_free.push_back(ref); }
    u32 make_ref(u32 id, StateView s)
    {
        const u32 n = bits::trimmed_size(s.w, s.nw);
        if (n > m_w)
            widen(std::max(m_w * 2, n));
        u32 ref;
        if (!m_free.empty())
        {
            ref = m_free.back();
            m_free.pop_back();
        }
        else
        {
            ref = m_records++;
            m_rec.resize(static_cast<usize>(m_records) * stride(), 0);
        }
        u64* r = m_rec.data() + static_cast<usize>(ref) * stride();
        r[0] = id;
        std::memcpy(r + 1, s.w, n * sizeof(u64));
        std::fill(r + 1 + n, r + 1 + m_w, u64{0});
        if (m_nn)
            std::memcpy(r + 1 + m_w, s.num, m_nn * sizeof(u64));
        return ref;
    }
    u32 successor(Cur& cur, const Delta& d, std::vector<u64>& next, Successors&)
    {
        const u32 n = cur.n;
        const u32 nn = apply_delta(cur.w.data(), n, d, next);
        ensure_keys();
        Fingerprint128 f = m_cur_fp;
        const u32 tw = std::max(n, nn);
        if (m_tog.size() < tw)
            m_tog.resize(tw, 0);
        m_changed.clear();
        auto toggle = [&](u32 slot)
        {
            // each atom whose truth value changes, once (the lists may repeat slots)
            if (bits::test(cur.w.data(), n, slot) != bits::test(next.data(), nn, slot) && !bits::test(m_tog.data(), tw, slot))
            {
                bits::set(m_tog.data(), slot);
                f.toggle(m_keys[slot]);
                m_changed.push_back(slot);
            }
        };
        for (SlotId x : d.del)
            toggle(x.v);
        for (SlotId x : d.add)
            toggle(x.v);
        for (u32 slot : m_changed)
            bits::reset(m_tog.data(), slot);
        m_next_fp = with_numeric(f, d.num);
        return nn;
    }
    std::pair<u32, bool> insert_successor(u32, const Cur&, const u64*, u32, const Delta&) { return m_map.insert(m_next_fp); }
    [[nodiscard]] u32 size() const noexcept { return m_map.size(); }
    [[nodiscard]] u64 bytes() const noexcept { return m_map.bytes() + m_rec.capacity() * sizeof(u64); }

private:
    [[nodiscard]] usize stride() const noexcept { return static_cast<usize>(m_w) + 1 + m_nn; }
    void ensure_keys()
    {
        const u32 slots = m_atoms.fluent_slots();
        while (m_keys.size() < slots)
        {
            const u64 c = m_atoms.canonical(AtomKind::Fluent, static_cast<u32>(m_keys.size()));
            m_keys.push_back({hash::mix64(2 * c + 1 + 0x9E3779B97F4A7C15ULL), hash::mix64(2 * c + 2 + 0x9E3779B97F4A7C15ULL)});
        }
    }
    Fingerprint128 fingerprint(const u64* w, u32 n)
    {
        ensure_keys();
        Fingerprint128 f;
        bits::for_each(w, n, [&](u64 slot) { f.toggle(m_keys[slot]); });
        return f;
    }
    /// The fingerprint of a state with atom fingerprint f and numeric words num (as the compact BrFS).
    [[nodiscard]] Fingerprint128 with_numeric(Fingerprint128 f, const u64* num) const noexcept
    {
        if (m_nn)
        {
            f.a ^= hash::words(num, m_nn, 0x3c6ef372fe94f82bULL);
            f.b ^= hash::words(num, m_nn, 0xa54ff53a5f1d36f1ULL);
        }
        return f;
    }
    void widen(u32 nw)
    {
        const usize ns = static_cast<usize>(nw) + 1 + m_nn;
        std::vector<u64> nr(static_cast<usize>(m_records) * ns, 0);
        for (usize i = 0; i < m_records; ++i)
        {
            const u64* r = m_rec.data() + i * stride();
            u64* o = nr.data() + i * ns;
            std::memcpy(o, r, (static_cast<usize>(m_w) + 1) * sizeof(u64));
            if (m_nn)
                std::memcpy(o + 1 + nw, r + 1 + m_w, m_nn * sizeof(u64));
        }
        m_rec.swap(nr);
        m_w = nw;
    }

    const AtomIndex& m_atoms;
    u32 m_w;
    u32 m_nn;  // numeric words per state
    std::vector<Fingerprint128> m_keys;
    FingerprintMap m_map;
    std::vector<u64> m_rec;  // records: [id, words (m_w), numeric words (m_nn)]
    u32 m_records = 0;
    std::vector<u32> m_free;
    Fingerprint128 m_cur_fp, m_next_fp;  // m_cur_fp covers the atoms only (updated per transition)
    std::vector<u64> m_tog;
    std::vector<u32> m_changed;
};

// ================================================================================================ search nodes
/// SoA search nodes keyed by state id.
struct Nodes
{
    enum : u8
    {
        New = 0,       // generated, never opened (beam: not retained)
        Open = 1,
        Closed = 2,
        DeadEnd = 3,
        StatusMask = 3,
        Goal = 4,       // a goal state (tested once, when first generated)
        Evaluated = 8,  // h holds the state's value
        Candidate = 16, // beam: in the current candidate list
    };

    std::vector<u32> parent, schema;
    std::vector<u64> offset;
    std::vector<ObjectId> binding;
    std::vector<f64> g, h;
    std::vector<u8> flags;
    std::vector<u32> depth;  // only with a depth budget
    bool track_depth = false;

    void add(u32 p, u32 s, const ObjectId* b, u32 arity, f64 gv)
    {
        parent.push_back(p);
        schema.push_back(s);
        offset.push_back(binding.size());
        binding.insert(binding.end(), b, b + arity);
        g.push_back(gv);
        h.push_back(0);
        flags.push_back(New);
        if (track_depth)
            depth.push_back(p == k_none ? 0 : depth[p] + 1);
    }
    void relink(u32 id, u32 p, u32 s, const ObjectId* b, u32 arity, f64 gv)
    {
        parent[id] = p;
        schema[id] = s;
        offset[id] = binding.size();
        binding.insert(binding.end(), b, b + arity);
        g[id] = gv;
        if (track_depth)
            depth[id] = depth[p] + 1;
    }
    [[nodiscard]] u8 status(u32 id) const noexcept { return flags[id] & StatusMask; }
    void set_status(u32 id, u8 st) noexcept { flags[id] = static_cast<u8>((flags[id] & ~StatusMask) | st); }
    [[nodiscard]] bool has(u32 id, u8 f) const noexcept { return (flags[id] & f) != 0; }
    void set(u32 id, u8 f) noexcept { flags[id] = static_cast<u8>(flags[id] | f); }
    void unset(u32 id, u8 f) noexcept { flags[id] = static_cast<u8>(flags[id] & ~f); }

    [[nodiscard]] std::vector<Action> plan(u32 id, Successors& succ) const
    {
        std::vector<Action> out;
        for (u32 v = id; parent[v] != k_none; v = parent[v])
        {
            const u32 s = schema[v];
            const ObjectId* b = binding.data() + offset[v];
            out.emplace_back(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s)));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
    [[nodiscard]] u64 bytes() const noexcept
    {
        return parent.capacity() * 4 + schema.capacity() * 4 + offset.capacity() * 8 + binding.capacity() * 4 +
               g.capacity() * 8 + h.capacity() * 8 + flags.capacity() + depth.capacity() * 4;
    }
};

/// The transitions of one expansion, buffered while the successor generator runs (goal tests and evaluations need
/// the engine afterwards). A kept state is its trimmed fluent words followed by `nn` numeric words.
struct Transitions
{
    struct T
    {
        u32 schema;
        u32 child;
        u64 boff;  // binding in `bind`
        u64 woff;  // words in `words` (when kept)
        u32 nw;
        f64 g;  // g of the child through this transition (mimir's metric value of the successor)
        bool fresh;
        bool preferred;
    };
    std::vector<T> t;
    std::vector<ObjectId> bind;
    std::vector<u64> words;
    u32 nn = 0;  // numeric words per kept state

    void clear()
    {
        t.clear();
        bind.clear();
        words.clear();
    }
    [[nodiscard]] usize index(const T& x) const noexcept { return static_cast<usize>(&x - t.data()); }
    [[nodiscard]] const ObjectId* binding(const T& x) const noexcept { return bind.data() + x.boff; }
    [[nodiscard]] const u64* state(const T& x) const noexcept { return words.data() + x.woff; }
    [[nodiscard]] StateView view(const T& x) const noexcept
    {
        return {state(x), x.nw, nn ? state(x) + x.nw : nullptr, nn};
    }
};

// ================================================================================================ context
/// Everything a best-first search shares: heuristic, costs, goal test, blocked states, budgets, observer, result.
class Context
{
public:
    Context(const Task& task, const BestFirstOptions& o, BestFirstResult& r, const char* algorithm);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    /// False when the search cannot run (r.status and r.message are set).
    [[nodiscard]] bool ok() const noexcept { return m_ok; }

    const Task& task;
    const BestFirstOptions& o;
    BestFirstResult& r;
    Successors& succ;
    heuristics::Heuristic* h = nullptr;
    const heuristics::ActionCosts* costs = nullptr;
    State start;
    f64 g0 = 0;
    SearchObserver* obs = nullptr;
    bool witness = false, canonical = true;
    bool stop_on_states = false;  // max_states is finite: stop generating once it is exceeded
    bool batched = false;         // the heuristic evaluates batches (Heuristic::batched)
    u32 nn = 0;                   // numeric words per state
    u64 max_states = ~u64{0};

    /// g of the successor under delta d of a state whose g is gp: mimir's metric value of the successor (total-cost
    /// effects applied to gp, or the metric of the successor's numeric values).
    [[nodiscard]] f64 g_next(f64 gp, const Delta& d) const { return costs->next(gp, d); }
    f64 evaluate(StateView s);
    /// Batched evaluators: h of each state of `states` into `out`, in one call of the heuristic (one per state for
    /// GoalSpec::AnyOf).
    void evaluate(std::span<const StateView> states, std::span<f64> out);
    /// Batched evaluators: goal-tests the fresh transitions of tr in order (stopping after the first goal state when
    /// `stop_at_goal`, where GBFS ends the search) and evaluates in one call the fresh ones up to that point
    /// (`goals_too`: goal states as well, as A* does), into fresh_goal and fresh_h (indexed like tr.t).
    void evaluate_fresh(const Transitions& tr, bool stop_at_goal, bool goals_too);
    std::vector<u8> fresh_goal;
    std::vector<f64> fresh_h;
    bool is_goal(StateView s);
    [[nodiscard]] bool blocked(const u64* w, u32 n, const u64* num) const
    {
        return m_has_blocked && m_blocked.find({w, n, num, nn}).valid();
    }
    /// Budget, time, cancellation and progress, once per expansion. False (and r.status set) to stop.
    bool keep_going();
    /// The task's goal has a false static literal.
    [[nodiscard]] bool unsolvable() const;
    void begin_search();
    /// Status Solved with the plan to `id` (nodes) and the goal state. `cheapest`: rewrite each step to the
    /// cheapest action between its two states (as in mimir's plan extraction; GBFS and beam).
    void solved(const Nodes& nodes, u32 id, StateView goal, bool cheapest);
    /// Final bookkeeping: statistics, heuristic statistics, observer on_end.
    void finish(u64 states, u64 bytes);
    /// The status of a search whose heuristic was interrupted (deadline or cancellation).
    [[nodiscard]] SearchStatus interrupted_status() const
    {
        return o.control.cancel.requested() ? SearchStatus::Cancelled : SearchStatus::OutOfTime;
    }
    [[nodiscard]] Action action(u32 schema, const ObjectId* b) const
    {
        return Action(SchemaId{schema}, std::vector<ObjectId>(b, b + succ.arity(schema)));
    }

    /// Runs the successor generator on the prepared state; emit returns false to stop (only honoured with a finite
    /// max_states).
    template<class Emit>
    void generate(Emit&& emit)
    {
        if (stop_on_states)
            succ.generate<true>(emit, witness, canonical);
        else
            succ.generate<false>(emit, witness, canonical);
    }

private:
    std::unique_ptr<heuristics::Heuristic> m_owned;
    std::unique_ptr<heuristics::ActionCosts> m_costs;
    FlatStateStore m_blocked;
    bool m_has_blocked = false;
    bool m_ok = true;
    Clock::time_point m_t0, m_search_t0, m_deadline;
    bool m_timed = false;
    u64 m_tick = 0;
    u64 m_next_progress = 0;
    std::vector<StateView> m_batch;
    std::vector<u32> m_batch_index;
    std::vector<f64> m_batch_h;
};

/// Generates the transitions of state `id` (cur) into tr, in generation order: blocked successors are dropped
/// (pruned, on_prune), the others are inserted into the store; a new state gets its search node (parent id, g through
/// this transition). Words are kept for new states, and for all when the store needs them (Compact) or an observer is
/// set. `preferred`: record whether each action is a preferred operator of the last evaluation. False when max_states
/// was exceeded (the expansion stops there).
template<class S>
bool expand(Context& c, S& store, Nodes& nodes, u32 id, Cur& cur, std::vector<u64>& next, Transitions& tr, bool preferred)
{
    tr.clear();
    tr.nn = c.nn;
    c.succ.prepare(cur.view());
    const f64 gp = nodes.g[id];
    const bool keep_all = S::k_needs_words || c.obs != nullptr;
    bool full = false;
    c.generate(
        [&](u32 s, const ObjectId* b, const Delta& d) -> bool
        {
            ++c.r.stats.generated;
            const u32 nn = store.successor(cur, d, next, c.succ);
            if (c.blocked(next.data(), nn, d.num))
            {
                ++c.r.stats.pruned;
                if (c.obs)
                    c.obs->on_prune(id, c.action(s, b), {next.data(), nn, d.num, c.nn});
                return true;
            }
            const auto [cid, fresh] = store.insert_successor(id, cur, next.data(), nn, d);
            const u32 arity = c.succ.arity(s);
            const f64 g = c.g_next(gp, d);
            if (fresh)
                nodes.add(id, s, b, arity, g);
            Transitions::T t{s, cid, tr.bind.size(), tr.words.size(), 0, g, fresh, false};
            tr.bind.insert(tr.bind.end(), b, b + arity);
            if (fresh || keep_all)
            {
                t.nw = bits::trimmed_size(next.data(), nn);
                tr.words.insert(tr.words.end(), next.data(), next.data() + t.nw);
                if (c.nn)
                    tr.words.insert(tr.words.end(), d.num, d.num + c.nn);
            }
            if (preferred)
                t.preferred = c.h->preferred(ActionLabel{SchemaId{s}, std::span<const ObjectId>(b, arity)});
            tr.t.push_back(t);
            if (fresh && store.size() > c.max_states)
            {
                full = true;
                return false;
            }
            return true;
        });
    return !full;
}

/// The queue kind for the options (Auto: buckets for small integral costs).
[[nodiscard]] bool use_buckets(const Context& c);
/// The store kind for the options (Auto: Flat for W <= 8 words, Chunked above).
[[nodiscard]] BestFirstOptions::Store store_kind(const Context& c);

/// Instantiates `Run<Store, Queue>{}(c)` for the store and queue the options select.
template<template<class, class> class Run>
void dispatch(Context& c)
{
    const bool buckets = use_buckets(c);
    switch (store_kind(c))
    {
        case BestFirstOptions::Store::Chunked:
            buckets ? Run<ChunkedAdapter, BucketQueue>{}(c) : Run<ChunkedAdapter, HeapQueue>{}(c);
            break;
        case BestFirstOptions::Store::Compact:
            buckets ? Run<CompactAdapter, BucketQueue>{}(c) : Run<CompactAdapter, HeapQueue>{}(c);
            break;
        default: buckets ? Run<FlatAdapter, BucketQueue>{}(c) : Run<FlatAdapter, HeapQueue>{}(c); break;
    }
}

/// Runs a search: builds the context, checks for an unsolvable goal, dispatches, catches errors (status Failed).
template<template<class, class> class Run>
BestFirstResult run(const Task& task, const BestFirstOptions& o, const char* algorithm)
{
    BestFirstResult r;
    std::unique_ptr<Context> c;
    try
    {
        c = std::make_unique<Context>(task, o, r, algorithm);
        if (!c->ok())
            return r;
        dispatch<Run>(*c);
    }
    catch (const heuristics::Interrupted&)
    {
        // a long evaluation (the lifted fallback) ran past the deadline or was cancelled
        r.status = c->interrupted_status();
        c->finish(r.stats.states, r.store_bytes);
    }
    catch (const std::exception& e)
    {
        r.status = SearchStatus::Failed;
        r.message = e.what();
        r.plan.clear();
        r.goal_state.reset();
    }
    r.fluent_slots = task.atoms().fluent_slots();
    return r;
}
}  // namespace mymyr::search::bf
