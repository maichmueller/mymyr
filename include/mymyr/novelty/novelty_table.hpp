#pragma once
// Novelty tables for IW(k).
//
// A table remembers which atom tuples of size <= k have been seen. Atoms are fluent atom slots (dense and monotone:
// lazy slots are assigned in order of first touch), so no extra index is needed. Semantics are mimir's
// `DynamicNoveltyTable` (fluent atoms only; tuples of size 1..k):
//   - a state is novel iff one of its tuples is unseen; mark_state() marks all of them (the root of an IW pass);
//   - a transition s -> t is novel iff a tuple of t containing at least one atom *added* by it (an atom of t that is
//     not in s) is unseen; test_and_mark() marks all such tuples (mimir's StatePairTupleIndexGenerator).
// Every state admitted to an IW tree has all of its tuples marked (the root by mark_state, a successor because its
// other tuples lie in its parent), so a novel successor is never a duplicate: IW needs no duplicate-detection table.
//
// Layout (per level j = tuple size):
//   - j = 1: a dense bitset over the slots (always dense: 2^28 slots at most, 32 MB);
//   - j = 2: dense, a symmetric square bit matrix (row a holds every b with {a, b} seen; the diagonal bit a holds
//     "singleton a seen"), so a transition test is one word loop per added atom: next & ~row[a] over the successor's
//     words; sparse, an open-addressing set of 64-bit keys (min << 32 | max);
//   - j >= 3: tuples are numbered by their combinadic rank: a sorted tuple a_1 < ... < a_j has rank
//     C(a_1, 1) + C(a_2, 2) + ... + C(a_j, j), a bijection onto 0 .. C(n, j) - 1 for atoms below n that does not
//     depend on n. Dense, a bit table of C(capacity, j) bits indexed by the rank (j! times smaller than a mixed-radix
//     table, and growing the capacity only appends zero bits); sparse, an open-addressing set of the 64-bit ranks, or
//     of 128-bit packed tuples when C(max_atoms, j) does not fit in 64 bits (the idea is from tyr's novelty table).
// A level is dense while its bit table fits in TableOptions::max_dense_bytes. Tables grow with the slot count (lazy
// slots, 1.5x; level 2 copies its rows); a level whose dense table would exceed the budget switches to sparse, one
// way (mimir's LIW table does the same with its 256 MiB default).
//
// Not thread-safe: one table per search (per thread).

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/core/types.hpp"

#include <array>
#include <span>
#include <vector>

namespace mymyr::novelty
{
/// Largest supported k: mimir's MAX_ARITY is 6, so its iw::find_solution accepts max_arity <= 5.
inline constexpr u32 k_max_arity = 5;

struct TableOptions
{
    /// Byte budget of each dense level j >= 2. A level above it is sparse (hashed packed tuples).
    u64 max_dense_bytes = u64{256} << 20;
};

/// Open-addressing set of 128-bit keys (packed sorted tuples). The all-ones key is reserved as "empty".
class TupleSet
{
public:
    struct Key
    {
        u64 lo = 0, hi = 0;
        friend bool operator==(const Key&, const Key&) = default;
    };
    static constexpr Key k_empty{~u64{0}, ~u64{0}};

    TupleSet() { rehash(64); }
    /// Returns true if k was not in the set.
    bool insert(Key k)
    {
        if ((m_size + 1) * 2 > m_slots.size()) [[unlikely]]
            rehash(m_slots.size() * 2);
        for (usize i = hash(k) & m_mask;; i = (i + 1) & m_mask)
        {
            Key& s = m_slots[i];
            if (s == k)
                return false;
            if (s == k_empty)
            {
                s = k;
                ++m_size;
                return true;
            }
        }
    }
    [[nodiscard]] bool contains(Key k) const
    {
        for (usize i = hash(k) & m_mask;; i = (i + 1) & m_mask)
        {
            const Key& s = m_slots[i];
            if (s == k)
                return true;
            if (s == k_empty)
                return false;
        }
    }
    [[nodiscard]] u64 size() const noexcept { return m_size; }
    [[nodiscard]] u64 bytes() const noexcept { return m_slots.capacity() * sizeof(Key); }
    void clear();

private:
    [[nodiscard]] static u64 hash(Key k) noexcept { return hash::mix64(k.lo ^ hash::mix64(k.hi + 0x9E3779B97F4A7C15ULL)); }
    void rehash(usize n);

    std::vector<Key> m_slots;
    usize m_mask = 0;
    u64 m_size = 0;
};

/// Open-addressing set of 64-bit keys (ranks, packed pairs). The all-ones key is reserved as "empty".
class RankSet
{
public:
    static constexpr u64 k_empty = ~u64{0};

    RankSet() { rehash(64); }
    /// Returns true if k was not in the set.
    bool insert(u64 k)
    {
        if ((m_size + 1) * 2 > m_slots.size()) [[unlikely]]
            rehash(m_slots.size() * 2);
        for (usize i = hash::mix64(k) & m_mask;; i = (i + 1) & m_mask)
        {
            u64& s = m_slots[i];
            if (s == k)
                return false;
            if (s == k_empty)
            {
                s = k;
                ++m_size;
                return true;
            }
        }
    }
    [[nodiscard]] bool contains(u64 k) const
    {
        for (usize i = hash::mix64(k) & m_mask;; i = (i + 1) & m_mask)
        {
            const u64 s = m_slots[i];
            if (s == k)
                return true;
            if (s == k_empty)
                return false;
        }
    }
    [[nodiscard]] u64 size() const noexcept { return m_size; }
    [[nodiscard]] u64 bytes() const noexcept { return m_slots.capacity() * sizeof(u64); }
    void clear();

private:
    void rehash(usize n);

    std::vector<u64> m_slots;
    usize m_mask = 0;
    u64 m_size = 0;
};

class NoveltyTable
{
public:
    /// arity k in 1..k_max_arity; max_atoms: an upper bound of the slot count (Task::atoms().max_fluent_slots()).
    /// Throws std::invalid_argument for k outside 1..k_max_arity and std::length_error if a sparse level j >= 3 can
    /// neither rank its tuples in 64 bits nor pack them into 128 (k = 5 with more than 2^25 atoms).
    NoveltyTable(u32 arity, u32 max_atoms, const TableOptions& options = {});

    [[nodiscard]] u32 arity() const noexcept { return m_k; }
    /// Slots below capacity() are addressable. reserve(n) makes slots < n addressable.
    [[nodiscard]] u32 capacity() const noexcept { return m_cap; }
    void reserve(u32 atoms)
    {
        if (atoms > m_cap) [[unlikely]]
            grow(atoms);
    }
    /// Whether level j (2..k) is dense.
    [[nodiscard]] bool dense(u32 j) const noexcept { return j < 2 || (j == 2 ? m_dense2 : m_levels[j].dense); }
    [[nodiscard]] u64 bytes() const noexcept;

    // --------------------------------------------------------------------------------------------- level 1
    [[nodiscard]] bool seen1(u32 a) const noexcept { return (m_t1[a >> 6] >> (a & 63)) & 1; }
    /// Read-only IW(1) precheck: whether some atom of `add` (slots < capacity()) is unseen.
    [[nodiscard]] bool any_new1(std::span<const u32> add) const noexcept
    {
        for (u32 a : add)
            if (!seen1(a))
                return true;
        return false;
    }
    /// Marks the singletons of `add` (IW(1) only: for k >= 2 use test_and_mark, which keeps the pair diagonal).
    void mark1(std::span<const u32> add) noexcept
    {
        for (u32 a : add)
            m_t1[a >> 6] |= u64{1} << (a & 63);
    }

    // --------------------------------------------------------------------------------------------- states
    /// Marks every tuple of the state (words w, n words; slots < capacity()). Returns whether one was unseen.
    bool mark_state(const u64* w, u32 n);
    /// Whether a tuple has been marked: `sorted` holds 1..arity() distinct slots in ascending order, each below
    /// capacity(). Read-only.
    [[nodiscard]] bool seen(std::span<const u32> sorted) const noexcept;
    /// Unmarks every tuple; the capacity and the layout of every level (dense or sparse) stay.
    void clear();

    /// Transition novelty. parent: the expanded state (np words); succ: the successor (ns words); add: the atoms of
    /// succ that are not in parent (each once; all slots < capacity()). Returns whether a tuple of succ containing an
    /// added atom is unseen; with Mark, marks all of them. For k == 1 only `add` is read.
    template<bool Mark>
    bool test(const u64* parent, u32 np, const u64* succ, u32 ns, std::span<const u32> add)
    {
        if (m_k == 1)
        {
            const bool novel = any_new1(add);
            if constexpr (Mark)
                if (novel)
                    mark1(add);
            return novel;
        }
        if (m_k == 2 && m_dense2)
            return test_dense2<Mark>(succ, ns, add);
        return test_generic<Mark>(*this, parent, np, succ, ns, add, m_scratch);
    }

    /// Scratch of a read-only test (peek); one per thread.
    struct Scratch
    {
        std::vector<u32> atoms;
        std::vector<u8> fresh;    // per atom of `atoms`: 1 if it is added
        std::vector<u32> suffix;  // suffix[i]: added atoms among atoms[i..]
    };
    /// test<false> that touches nothing but `scratch`, so several threads can peek at once while nobody marks. The
    /// slots of `add` may exceed capacity(): such an atom was never marked, so the transition is novel.
    [[nodiscard]] bool peek(const u64* parent, u32 np, const u64* succ, u32 ns, std::span<const u32> add, Scratch& scratch) const
    {
        for (u32 a : add)
            if (a >= m_cap || !seen1(a))
                return true;
        if (m_k == 1)
            return false;
        if (m_k == 2 && m_dense2)
            return peek_dense2(succ, ns, add);
        return test_generic<false>(*this, parent, np, succ, ns, add, scratch);
    }

private:
    struct Level  // j >= 3
    {
        bool dense = true;
        bool ranked = true;     // sparse keys are ranks (C(max_atoms, j) < 2^64 - 1), else packed tuples
        std::vector<u64> bits;  // dense: C(cap, j) bits, indexed by rank
        RankSet ranks;          // sparse, ranked
        TupleSet packed;        // sparse, not ranked
    };

    // IW(2), dense: novel iff some added a is unseen or some b of succ has {a, b} unseen (bit b of row a clear).
    template<bool Mark>
    bool test_dense2(const u64* succ, u32 ns, std::span<const u32> add)
    {
        const usize rw = m_row_words;
        bool novel = false;
        for (u32 a : add)
        {
            if (!seen1(a))
            {
                novel = true;
                break;
            }
            const u64* row = m_t2.data() + static_cast<usize>(a) * rw;
            MYMYR_NOVECTOR
            for (u32 i = 0; i < ns; ++i)
                if (succ[i] & ~row[i])
                {
                    novel = true;
                    break;
                }
            if (novel)
                break;
        }
        if constexpr (Mark)
            if (novel)
                mark_dense2(succ, ns, add);
        return novel;
    }
    void mark_dense2(const u64* succ, u32 ns, std::span<const u32> add);

    // read-only test_dense2 after the singletons were found seen (every atom of succ is then below capacity)
    [[nodiscard]] bool peek_dense2(const u64* succ, u32 ns, std::span<const u32> add) const noexcept
    {
        const usize rw = m_row_words;
        for (u32 a : add)
        {
            const u64* row = m_t2.data() + static_cast<usize>(a) * rw;
            for (u32 i = 0; i < ns; ++i)
                if (succ[i] & ~row[i])
                    return true;
        }
        return false;
    }

    // Self: NoveltyTable, or const NoveltyTable without Mark (then only `s` is written).
    template<bool Mark, class Self>
    static bool test_generic(Self& self, const u64* parent, u32 np, const u64* succ, u32 ns, std::span<const u32> add, Scratch& s);

    void grow(u32 atoms);
    [[nodiscard]] TupleSet::Key pack(const u32* sorted, u32 j) const noexcept;
    [[nodiscard]] u64 binom(u32 a, u32 i) const noexcept { return m_binom[static_cast<usize>(i) * m_cap + a]; }
    [[nodiscard]] u64 rank(const u32* sorted, u32 j) const noexcept
    {
        u64 r = sorted[0];
        for (u32 i = 1; i < j; ++i)
            r += binom(sorted[i], i + 1);
        return r;
    }
    void unrank(u64 r, u32 j, u32* sorted) const noexcept;
    bool visit_pair(u32 a, u32 b, bool mark);  // sparse level 2
    bool visit_tuple(const u32* sorted, u32 j, bool mark);

    u32 m_k;
    u32 m_max_atoms;
    u32 m_cap = 0;  // multiple of 64
    u32 m_pack_bits = 0;
    TableOptions m_options;
    std::vector<u64> m_t1;  // cap bits
    bool m_dense2 = true;
    usize m_row_words = 0;   // cap / 64
    std::vector<u64> m_t2;   // cap x cap bits (dense level 2)
    RankSet m_s2;            // sparse level 2: (min << 32 | max)
    std::array<Level, k_max_arity + 1> m_levels;
    std::vector<u64> m_binom;  // C(a, i) at [i * cap + a] for a < cap, i <= the largest dense or ranked level >= 3
    Scratch m_scratch;  // of test_generic
};

// ----------------------------------------------------------------------------------------------------- inline
template<bool Mark, class Self>
bool NoveltyTable::test_generic(Self& self, const u64* parent, u32 np, const u64* succ, u32 ns, std::span<const u32> add, Scratch& s)
{
    // level 1
    bool novel = self.any_new1(add);
    if (novel && !Mark)
        return true;
    if constexpr (Mark)
        self.mark1(add);
    // the successor's atoms, in increasing order, with a flag for the added ones
    s.atoms.clear();
    s.fresh.clear();
    bits::for_each(succ, ns,
                   [&](u64 b)
                   {
                       s.atoms.push_back(static_cast<u32>(b));
                       s.fresh.push_back(bits::test(parent, np, b) ? 0 : 1);
                   });
    const u32 n = static_cast<u32>(s.atoms.size());
    s.suffix.assign(n + 1, 0);
    for (u32 i = n; i-- > 0;)
        s.suffix[i] = s.suffix[i + 1] + s.fresh[i];
    // levels 2..k: every sorted j-subset of the successor's atoms with at least one added atom
    u32 tuple[k_max_arity];
    u32 idx[k_max_arity];
    for (u32 j = 2; j <= self.m_k && j <= n; ++j)
    {
        // iterative enumeration of index combinations idx[0] < ... < idx[j-1], pruned when no added atom can occur
        u32 depth = 0;
        u32 news[k_max_arity + 1] = {0};  // added atoms among idx[0..depth)
        idx[0] = 0;
        while (true)
        {
            if (idx[depth] > n - (j - depth))
            {
                // exhausted this depth: backtrack
                if (depth == 0)
                    break;
                --depth;
                ++idx[depth];
                continue;
            }
            // prune: no added atom chosen so far and none left from idx[depth] on
            if (news[depth] == 0 && s.suffix[idx[depth]] == 0)
            {
                if (depth == 0)
                    break;
                --depth;
                ++idx[depth];
                continue;
            }
            news[depth + 1] = news[depth] + s.fresh[idx[depth]];
            if (depth + 1 == j)
            {
                if (news[j] > 0)
                {
                    for (u32 i = 0; i < j; ++i)
                        tuple[i] = s.atoms[idx[i]];
                    bool unseen;
                    if constexpr (Mark)
                        unseen = j == 2 ? self.visit_pair(tuple[0], tuple[1], true) : self.visit_tuple(tuple, j, true);
                    else
                        unseen = !self.seen(std::span<const u32>(tuple, j));
                    if (unseen)
                    {
                        novel = true;
                        if constexpr (!Mark)
                            return true;
                    }
                }
                ++idx[depth];
                continue;
            }
            idx[depth + 1] = idx[depth] + 1;
            ++depth;
        }
    }
    return novel;
}
}  // namespace mymyr::novelty
