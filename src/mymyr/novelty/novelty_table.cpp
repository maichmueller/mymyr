// Novelty tables: growth, remapping, the sparse levels and state marking (see novelty/novelty_table.hpp).

#include "mymyr/novelty/novelty_table.hpp"

#include <algorithm>
#include <bit>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mymyr::novelty
{
namespace
{
u32 round64(u64 n) { return static_cast<u32>(std::min<u64>((n + 63) / 64 * 64, u64{1} << 31)); }

/// Bytes of a dense level j over cap atoms (level 2: the cap x cap matrix; j >= 3: C(cap, j) bits), saturating.
u64 dense_bytes(u32 cap, u32 j)
{
    long double bits = 1;
    if (j == 2)
        bits = static_cast<long double>(cap) * cap;
    else
        for (u32 i = 0; i < j; ++i)
            bits = bits * (static_cast<long double>(cap) - i) / (i + 1);
    const long double bytes = bits / 8;
    return bytes > 1e18L ? ~u64{0} : static_cast<u64>(bytes);
}

/// Whether every rank of a j-subset of n atoms (0 .. C(n, j) - 1) is below 2^64 - 1 (the empty key excluded).
bool ranks_fit(u64 n, u32 j)
{
    u64 c = 1;  // C(n, i)
    for (u32 i = 0; i < j; ++i)
    {
        if (n < i + 1)
            return true;  // C(n, j) = 0
        // C(n, i + 1) = C(n, i) * (n - i) / (i + 1) without overflow: after dividing c by g = gcd(c, i + 1), the rest
        // of i + 1 divides n - i
        const u64 g = std::gcd(c, u64{i} + 1);
        const u64 x = c / g, y = (n - i) / ((u64{i} + 1) / g);
        if (y != 0 && x > (~u64{0} - 1) / y)
            return false;
        c = x * y;
    }
    return c < ~u64{0};
}
}  // namespace

// ----------------------------------------------------------------------------------------------------- RankSet
void RankSet::rehash(usize n)
{
    std::vector<u64> old;
    old.swap(m_slots);
    m_slots.assign(n, k_empty);
    m_mask = n - 1;
    m_size = 0;
    for (u64 k : old)
        if (k != k_empty)
            insert(k);
}

void RankSet::clear()
{
    m_slots.assign(64, k_empty);
    m_mask = 63;
    m_size = 0;
}

// ----------------------------------------------------------------------------------------------------- TupleSet
void TupleSet::rehash(usize n)
{
    std::vector<Key> old;
    old.swap(m_slots);
    m_slots.assign(n, k_empty);
    m_mask = n - 1;
    m_size = 0;
    for (const Key& k : old)
        if (!(k == k_empty))
            insert(k);
}

void TupleSet::clear()
{
    m_slots.assign(64, k_empty);
    m_mask = 63;
    m_size = 0;
}

// ----------------------------------------------------------------------------------------------------- NoveltyTable
NoveltyTable::NoveltyTable(u32 arity, u32 max_atoms, const TableOptions& options)
    : m_k(arity), m_max_atoms(std::max<u32>(max_atoms, 1)), m_options(options)
{
    if (arity < 1 || arity > k_max_arity)
        throw std::invalid_argument("novelty table arity must be in 1.." + std::to_string(k_max_arity) + ", got " + std::to_string(arity));
    m_pack_bits = static_cast<u32>(std::bit_width(m_max_atoms));
    for (u32 j = 3; j <= m_k; ++j)
    {
        m_levels[j].ranked = ranks_fit(round64(m_max_atoms), j);
        if (!m_levels[j].ranked && static_cast<u64>(j) * m_pack_bits > 128)
            throw std::length_error("novelty table: " + std::to_string(j) + "-tuples over " + std::to_string(m_max_atoms) +
                                    " atoms neither rank into 64 bits nor pack into 128");
    }
    grow(64);
}

u64 NoveltyTable::bytes() const noexcept
{
    u64 b = m_t1.capacity() * 8 + m_t2.capacity() * 8 + m_s2.bytes() + m_binom.capacity() * 8;
    for (const Level& l : m_levels)
        b += l.bits.capacity() * 8 + l.ranks.bytes() + l.packed.bytes();
    b += m_atoms.capacity() * 4 + m_new.capacity() + m_suffix_new.capacity() * 4;
    return b;
}

void NoveltyTable::grow(u32 atoms)
{
    const u32 limit = round64(m_max_atoms);
    u32 cap = m_cap == 0 ? round64(std::max<u32>(atoms, 64))
                         : round64(std::max<u64>(atoms, static_cast<u64>(m_cap) + m_cap / 2));
    cap = std::min(cap, std::max(limit, round64(atoms)));
    if (cap <= m_cap)
        return;
    const u32 old_cap = m_cap;

    // level 1
    m_t1.resize(cap / 64, 0);

    // level 2
    if (m_k >= 2)
    {
        if (m_dense2 && dense_bytes(cap, 2) > m_options.max_dense_bytes)
        {
            // switch to sparse (one way): move every off-diagonal pair
            for (u32 a = 0; a < old_cap; ++a)
            {
                const u64* row = m_t2.data() + static_cast<usize>(a) * m_row_words;
                bits::for_each(row, static_cast<u32>(m_row_words),
                               [&](u64 b)
                               {
                                   if (b > a)
                                       m_s2.insert((static_cast<u64>(a) << 32) | b);
                               });
            }
            m_dense2 = false;
            std::vector<u64>().swap(m_t2);
            m_row_words = 0;
        }
        else if (m_dense2)
        {
            const usize rw = cap / 64;
            std::vector<u64> t2(static_cast<usize>(cap) * rw, 0);
            for (u32 a = 0; a < old_cap; ++a)
                std::copy_n(m_t2.data() + static_cast<usize>(a) * m_row_words, m_row_words, t2.data() + static_cast<usize>(a) * rw);
            m_t2.swap(t2);
            m_row_words = rw;
        }
    }

    m_cap = cap;
    if (m_k < 3)
        return;

    // binomial coefficients C(a, i), i <= the largest level that is dense or ranked, a < cap (Pascal's rule,
    // saturating: only ranks of fitting levels are ever formed, and those stay below 2^64 - 1)
    u32 rows = 0;
    for (u32 j = 3; j <= m_k; ++j)
        if (m_levels[j].dense || m_levels[j].ranked)
            rows = j;
    m_binom.assign(rows == 0 ? 0 : static_cast<usize>(rows + 1) * cap, 0);
    for (u32 a = 0; a < cap && rows > 0; ++a)
    {
        m_binom[a] = 1;
        for (u32 i = 1; i <= rows && a > 0; ++i)
        {
            const u64 x = m_binom[static_cast<usize>(i - 1) * cap + a - 1], y = m_binom[static_cast<usize>(i) * cap + a - 1];
            m_binom[static_cast<usize>(i) * cap + a] = x + y < x ? ~u64{0} : x + y;
        }
    }

    // levels 3..k: ranks do not depend on the capacity, so a dense level only appends zero bits
    for (u32 j = 3; j <= m_k; ++j)
    {
        Level& l = m_levels[j];
        if (!l.dense)
            continue;
        const u64 bytes = dense_bytes(cap, j);
        if (bytes <= m_options.max_dense_bytes)
        {
            l.bits.resize((bytes + 7) / 8 + 1, 0);
            continue;
        }
        // switch to sparse (one way): move every set rank
        l.dense = false;
        std::vector<u64> old;
        old.swap(l.bits);
        u32 t[k_max_arity];
        bits::for_each(old.data(), static_cast<u32>(old.size()),
                       [&](u64 r)
                       {
                           if (l.ranked)
                               l.ranks.insert(r);
                           else
                           {
                               unrank(r, j, t);
                               l.packed.insert(pack(t, j));
                           }
                       });
    }
}

TupleSet::Key NoveltyTable::pack(const u32* sorted, u32 j) const noexcept
{
    u64 lo = 0, hi = 0;
    u32 used = 0;
    for (u32 i = 0; i < j; ++i)
    {
        const u64 v = sorted[i];
        if (used + m_pack_bits <= 64)
            lo |= v << used;
        else if (used >= 64)
            hi |= v << (used - 64);
        else
        {
            lo |= v << used;
            hi |= v >> (64 - used);
        }
        used += m_pack_bits;
    }
    return {lo, hi};
}

void NoveltyTable::unrank(u64 r, u32 j, u32* sorted) const noexcept
{
    // greedy combinadic decoding: the largest a with C(a, i) <= r, for i = j .. 1
    for (u32 i = j; i >= 1; --i)
    {
        const u64* row = m_binom.data() + static_cast<usize>(i) * m_cap;
        const u32 a = static_cast<u32>(std::upper_bound(row, row + m_cap, r) - row) - 1;
        sorted[i - 1] = a;
        r -= row[a];
    }
}

bool NoveltyTable::visit_pair(u32 a, u32 b, bool mark)
{
    // a < b
    if (m_dense2)
    {
        u64* ra = m_t2.data() + static_cast<usize>(a) * m_row_words;
        if ((ra[b >> 6] >> (b & 63)) & 1)
            return false;
        if (mark)
        {
            ra[b >> 6] |= u64{1} << (b & 63);
            u64* rb = m_t2.data() + static_cast<usize>(b) * m_row_words;
            rb[a >> 6] |= u64{1} << (a & 63);
        }
        return true;
    }
    const u64 key = (static_cast<u64>(a) << 32) | b;
    return mark ? m_s2.insert(key) : !m_s2.contains(key);
}

bool NoveltyTable::visit_tuple(const u32* sorted, u32 j, bool mark)
{
    Level& l = m_levels[j];
    if (l.dense)
    {
        const u64 x = rank(sorted, j);
        u64& w = l.bits[x >> 6];
        const u64 bit = u64{1} << (x & 63);
        if (w & bit)
            return false;
        if (mark)
            w |= bit;
        return true;
    }
    if (l.ranked)
    {
        const u64 r = rank(sorted, j);
        return mark ? l.ranks.insert(r) : !l.ranks.contains(r);
    }
    const TupleSet::Key key = pack(sorted, j);
    return mark ? l.packed.insert(key) : !l.packed.contains(key);
}

void NoveltyTable::mark_dense2(const u64* succ, u32 ns, std::span<const u32> add)
{
    const usize rw = m_row_words;
    for (u32 a : add)
    {
        m_t1[a >> 6] |= u64{1} << (a & 63);
        u64* row = m_t2.data() + static_cast<usize>(a) * rw;
        MYMYR_NOVECTOR
        for (u32 i = 0; i < ns; ++i)
            row[i] |= succ[i];  // every pair {a, b} of the successor, and the diagonal (singleton a)
    }
    // symmetric half: bit a in the row of every atom b of the successor
    bits::for_each(succ, ns,
                   [&](u64 b)
                   {
                       u64* row = m_t2.data() + static_cast<usize>(b) * rw;
                       for (u32 a : add)
                           row[a >> 6] |= u64{1} << (a & 63);
                   });
}

bool NoveltyTable::seen(std::span<const u32> sorted) const noexcept
{
    const u32 j = static_cast<u32>(sorted.size());
    if (j == 1)
        return seen1(sorted[0]);
    if (j == 2)
    {
        const u32 a = sorted[0], b = sorted[1];
        if (m_dense2)
            return (m_t2[static_cast<usize>(a) * m_row_words + (b >> 6)] >> (b & 63)) & 1;
        return m_s2.contains((static_cast<u64>(a) << 32) | b);
    }
    const Level& l = m_levels[j];
    if (l.dense)
    {
        const u64 x = rank(sorted.data(), j);
        return (l.bits[x >> 6] >> (x & 63)) & 1;
    }
    if (l.ranked)
        return l.ranks.contains(rank(sorted.data(), j));
    return l.packed.contains(pack(sorted.data(), j));
}

void NoveltyTable::clear()
{
    std::fill(m_t1.begin(), m_t1.end(), 0);
    std::fill(m_t2.begin(), m_t2.end(), 0);
    m_s2.clear();
    for (Level& l : m_levels)
    {
        std::fill(l.bits.begin(), l.bits.end(), 0);
        l.ranks.clear();
        l.packed.clear();
    }
}

bool NoveltyTable::mark_state(const u64* w, u32 n)
{
    std::vector<u32> atoms;
    bits::for_each(w, n, [&](u64 b) { atoms.push_back(static_cast<u32>(b)); });
    if (!atoms.empty())
        reserve(atoms.back() + 1);
    bool novel = false;
    if (m_k == 1)
    {
        novel = any_new1(atoms);
        mark1(atoms);
    }
    else if (m_k == 2 && m_dense2)
    {
        // every atom is "added" relative to the empty state
        novel = test_dense2<false>(w, n, atoms);
        mark_dense2(w, n, atoms);
    }
    else
        novel = test_generic<true>(nullptr, 0, w, n, atoms);
    return novel;
}
}  // namespace mymyr::novelty
