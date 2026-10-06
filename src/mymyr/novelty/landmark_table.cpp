// Landmark-restricted novelty (see novelty/landmark_table.hpp).

#include "mymyr/novelty/landmark_table.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mymyr::novelty
{
namespace
{
void normalize(std::vector<u32>& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// C(n + k, k) as a long double (saturating well above any budget).
long double combinations(u64 n, u32 k)
{
    long double c = 1;
    for (u32 i = 1; i <= k; ++i)
        c = c * static_cast<long double>(n + i) / i;
    return c;
}

u32 round64(u64 n) { return static_cast<u32>(std::min<u64>((n + 63) / 64 * 64, u64{1} << 31)); }
}  // namespace

// ------------------------------------------------------------------------------------------ LandmarkCoordinates
LandmarkCoordinates::LandmarkCoordinates(std::vector<std::vector<u32>> groups)
{
    m_groups = static_cast<u32>(groups.size());
    std::vector<std::pair<u32, u32>> pairs;  // (atom, rank)
    for (u32 r = 0; r < groups.size(); ++r)
    {
        normalize(groups[r]);
        m_bijective = m_bijective && groups[r].size() == 1;
        for (u32 a : groups[r])
            pairs.emplace_back(a, r);
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    m_offsets.push_back(0);
    for (usize i = 0; i < pairs.size(); ++i)
    {
        if (m_atoms.empty() || m_atoms.back() != pairs[i].first)
        {
            if (!m_atoms.empty())
                m_offsets.push_back(static_cast<u32>(m_flat.size()));
            m_atoms.push_back(pairs[i].first);
        }
        m_flat.push_back(pairs[i].second);
    }
    if (!m_atoms.empty())
        m_offsets.push_back(static_cast<u32>(m_flat.size()));
    for (usize i = 0; i + 1 < m_offsets.size(); ++i)
        m_bijective = m_bijective && m_offsets[i + 1] - m_offsets[i] == 1;
    if (!m_atoms.empty())
    {
        m_lookup.assign(static_cast<usize>(m_atoms.back()) + 1, 0);
        for (u32 i = 0; i < m_atoms.size(); ++i)
            m_lookup[m_atoms[i]] = i + 1;
    }
    m_member_offsets.assign(static_cast<usize>(m_groups) + 1, 0);
    for (const auto& pr : pairs)
        ++m_member_offsets[pr.second + 1];
    for (u32 r = 0; r < m_groups; ++r)
        m_member_offsets[r + 1] += m_member_offsets[r];
    m_members.resize(pairs.size());
    std::vector<u32> fill(m_member_offsets.begin(), m_member_offsets.end() - 1);
    for (const auto& pr : pairs)  // ascending atoms per rank
        m_members[fill[pr.second]++] = pr.first;
}

LandmarkCoordinates LandmarkCoordinates::make(std::vector<u32> fact_landmarks, const std::vector<std::vector<u32>>& disjunctive, bool use_disjunctive,
                                              bool all_private, const std::vector<u32>& unshared)
{
    std::vector<std::vector<u32>> groups;
    if (!use_disjunctive || disjunctive.empty())
    {
        normalize(fact_landmarks);
        for (u32 a : fact_landmarks)
            groups.push_back({a});
        return LandmarkCoordinates(std::move(groups));
    }
    if (all_private)
    {
        if (!unshared.empty())
            throw std::invalid_argument("LIW: all_private cannot be combined with unshared atoms");
        std::vector<u32> atoms = std::move(fact_landmarks);
        for (const auto& members : disjunctive)
            atoms.insert(atoms.end(), members.begin(), members.end());
        normalize(atoms);
        for (u32 a : atoms)
            groups.push_back({a});
        return LandmarkCoordinates(std::move(groups));
    }
    std::vector<u32> un = unshared;
    normalize(un);
    std::vector<u32> singletons = std::move(fact_landmarks);
    for (const auto& members : disjunctive)
    {
        std::vector<u32> kept;
        for (u32 a : members)
        {
            if (std::binary_search(un.begin(), un.end(), a))
                singletons.push_back(a);
            else
                kept.push_back(a);
        }
        if (!kept.empty())
        {
            normalize(kept);
            groups.push_back(std::move(kept));
        }
    }
    normalize(singletons);
    for (u32 a : singletons)
        groups.push_back({a});
    return LandmarkCoordinates(std::move(groups));
}

bool LandmarkCoordinates::mask(const u64* w, u32 n, u64* out) const
{
    const u32 mw = mask_words();
    std::fill(out, out + mw, u64{0});
    bool any = false;
    for (u32 i = 0; i < m_atoms.size(); ++i)
        if (bits::test(w, n, m_atoms[i]))
        {
            any = true;
            for (u32 j = m_offsets[i]; j < m_offsets[i + 1]; ++j)
                bits::set(out, m_flat[j]);
        }
    return any;
}

void LandmarkCoordinates::collect(const u64* mask, bool any, std::vector<u32>& ranks) const
{
    ranks.clear();
    if (!any)
    {
        ranks.push_back(bot());
        return;
    }
    bits::for_each(mask, mask_words(), [&](u64 r) { ranks.push_back(static_cast<u32>(r)); });
}

void LandmarkCoordinates::split(const u64* pmask, bool pany, const u64* smask, bool sany, std::vector<u32>& flipped, std::vector<u32>& kept) const
{
    flipped.clear();
    kept.clear();
    if (!sany)
    {
        (pany ? flipped : kept).push_back(bot());
        return;
    }
    const u32 mw = mask_words();
    for (u32 i = 0; i < mw; ++i)
    {
        u64 x = smask[i];
        while (x)
        {
            const u32 b = static_cast<u32>(bits::ctz64(x));
            x &= x - 1;
            ((pmask[i] >> b) & 1 ? kept : flipped).push_back(i * 64 + b);
        }
    }
}

bool LandmarkCoordinates::child_mask(const u64* pmask, const u64* w, u32 n, std::span<const u32> add, std::span<const SlotId> del, u64* out) const
{
    const u32 mw = mask_words();
    std::copy(pmask, pmask + mw, out);
    for (SlotId x : del)
    {
        const u32 pos = position_of(x.v);
        if (pos == 0 || bits::test(w, n, x.v))
            continue;
        for (u32 j = m_offsets[pos - 1]; j < m_offsets[pos]; ++j)
        {
            const u32 r = m_flat[j];
            bool held = false;
            if (!m_bijective)
                for (u32 i = m_member_offsets[r]; i < m_member_offsets[r + 1] && !held; ++i)
                    held = bits::test(w, n, m_members[i]);
            if (!held)
                bits::reset(out, r);
        }
    }
    for (u32 a : add)
        if (const u32 pos = position_of(a))
            for (u32 j = m_offsets[pos - 1]; j < m_offsets[pos]; ++j)
                bits::set(out, m_flat[j]);
    return bits::any(out, mw);
}

void LandmarkCoordinates::split_masks(const u64* pmask, bool pany, const u64* smask, bool sany, u64* flipped, u64* kept) const
{
    const u32 mw = mask_words();
    for (u32 i = 0; i < mw; ++i)
    {
        flipped[i] = smask[i] & ~pmask[i];
        kept[i] = smask[i] & pmask[i];
    }
    if (!sany)
        bits::set(pany ? flipped : kept, bot());
}

// ------------------------------------------------------------------------------------------ LandmarkNoveltyTable
LandmarkNoveltyTable::LandmarkNoveltyTable(u32 arity, u32 num_ranks, u32 max_atoms, const LandmarkTableOptions& options)
    : m_k(arity), m_ranks(std::max<u32>(num_ranks, 1)), m_rw(bits::words_for(std::max<u32>(num_ranks, 1))), m_max_atoms(std::max<u32>(max_atoms, 1)),
      m_options(options)
{
    if (m_k < 1 || m_k > k_max_arity)
        throw std::invalid_argument("LIW arity " + std::to_string(m_k) + " outside 1.." + std::to_string(k_max_arity));
    m_packed = m_k <= 3;
    if (!m_packed && combinations(m_max_atoms, m_k) >= 1.8e19L)
        throw std::length_error("LIW(" + std::to_string(m_k) + "): tuples over " + std::to_string(m_max_atoms) + " atoms cannot be ranked in 64 bits");
    m_dense = options.max_dense_bytes > 0;
    m_fast = m_dense && m_k <= 2 ? m_k : 0;
    if (m_fast == 2)
        m_dense = false;  // the rank-major sets hold the marks
    m_mask.assign(m_rw, 0);
    m_binom.resize(m_k + 1);
}

u64 LandmarkNoveltyTable::rows_for(u32 cap) const
{
    const long double r = combinations(cap, m_k);
    return r > 1e18L ? ~u64{0} : static_cast<u64>(r);
}

void LandmarkNoveltyTable::reserve(u32 n)
{
    if (n <= m_cap)
        return;
    const u32 cap = std::max<u32>(round64(std::max<u64>(n, u64{m_cap} + m_cap / 2)), n);
    if (m_fast != 0)
    {
        if (fast_bytes(cap) <= m_options.max_dense_bytes)
        {
            grow_fast(cap);
            m_cap = cap;
        }
        else
            leave_fast(cap);
        return;
    }
    generic_reserve(cap);
}

u64 LandmarkNoveltyTable::fast_bytes(u32 cap) const
{
    const long double aw = bits::words_for(cap);
    long double b = m_ranks;
    if (m_fast == 1)
        b += (static_cast<long double>(rows_for(cap)) * m_rw + static_cast<long double>(m_ranks) * aw) * 8;
    else
        b += static_cast<long double>(m_ranks) * cap * aw * 8;
    return b > 1e18L ? ~u64{0} : static_cast<u64>(b);
}

void LandmarkNoveltyTable::grow_fast(u32 cap)
{
    const u32 aw = bits::words_for(cap);
    if (m_fast == 1)
    {
        m_rows.resize(static_cast<usize>(rows_for(cap)) * m_rw, 0);
        std::vector<u64> nb(static_cast<usize>(m_ranks) * aw, 0);
        for (u32 r = 0; r < m_ranks && m_aw > 0; ++r)
            std::copy_n(m_bits.data() + static_cast<usize>(r) * m_aw, m_aw, nb.data() + static_cast<usize>(r) * aw);
        m_bits.swap(nb);
    }
    else
    {
        std::vector<u64> nb(static_cast<usize>(m_ranks) * cap * aw, 0);
        for (u32 r = 0; r < m_ranks && m_aw > 0; ++r)
            for (u32 a = 0; a < m_cap; ++a)
                std::copy_n(m_bits.data() + (static_cast<usize>(r) * m_cap + a) * m_aw, m_aw, nb.data() + (static_cast<usize>(r) * cap + a) * aw);
        m_bits.swap(nb);
    }
    m_empty.resize(m_ranks, 0);
    m_aw = aw;
}

void LandmarkNoveltyTable::leave_fast(u32 cap)
{
    std::vector<u64> old;
    old.swap(m_bits);
    std::vector<u8> empty;
    empty.swap(m_empty);
    const u32 fast = m_fast, old_cap = m_cap, old_aw = m_aw;
    m_fast = 0;
    m_aw = 0;
    if (fast == 1)
    {
        generic_reserve(cap);  // the tuple-major rows hold every mark
        return;
    }
    m_dense = true;
    generic_reserve(cap);
    u32 t[2];
    for (u32 r = 0; r < m_ranks && old_aw > 0; ++r)
    {
        if (empty[r])
            insert_generic(t, 0, r);
        for (u32 a = 0; a < old_cap; ++a)
            bits::for_each(old.data() + (static_cast<usize>(r) * old_cap + a) * old_aw, old_aw,
                           [&](u64 b)
                           {
                               if (b < a)
                                   return;
                               t[0] = a;
                               t[1] = static_cast<u32>(b);
                               insert_generic(t, b == a ? 1 : 2, r);
                           });
    }
}

void LandmarkNoveltyTable::insert_generic(const u32* sorted, u32 j, u32 rank)
{
    if (m_dense)
        bits::set(m_rows.data() + code(sorted, j) * m_rw, rank);
    else
        sparse_insert(sorted, j, m_packed ? 0 : code(sorted, j), rank);
}

void LandmarkNoveltyTable::generic_reserve(u32 cap)
{
    if (m_dense)
    {
        const u64 rows = rows_for(cap);
        const bool fits = rows != ~u64{0} && static_cast<long double>(rows) * m_rw * 8 <= static_cast<long double>(m_options.max_dense_bytes);
        if (!fits)
            to_sparse();
    }
    if (m_dense || !m_packed)
    {
        // C(x, i) for i >= 2 and x < cap + k (C(x, 1) = x is computed inline)
        const usize X = static_cast<usize>(cap) + m_k;
        for (u32 i = 2; i <= m_k; ++i)
        {
            std::vector<u64>& b = m_binom[i];
            const usize from = b.size();
            b.resize(X);
            for (usize x = from; x < X; ++x)
            {
                // C(x, i) = C(x - 1, i) + C(x - 1, i - 1)
                const u64 prev = x == 0 ? 0 : b[x - 1];
                const u64 lower = x == 0 ? 0 : (i == 2 ? static_cast<u64>(x - 1) : m_binom[i - 1][x - 1]);
                b[x] = x < i ? 0 : prev + lower;
            }
        }
    }
    if (m_dense)
        m_rows.resize(static_cast<usize>(rows_for(cap)) * m_rw, 0);
    m_cap = cap;
}

u64 LandmarkNoveltyTable::code(const u32* sorted, u32 j) const noexcept
{
    const u32 pad = m_k - j;
    u64 r = 0;
    for (u32 i = 0; i < m_k; ++i)
    {
        const u64 v = i < pad ? i : static_cast<u64>(sorted[i - pad]) + m_k;
        r += i == 0 ? v : m_binom[i + 1][v];
    }
    return r;
}

void LandmarkNoveltyTable::unrank(u64 r, u32* sorted, u32& j) const
{
    u64 v[k_max_arity];
    for (u32 i = m_k; i >= 1; --i)
    {
        u64 x;
        if (i == 1)
            x = r;
        else
        {
            const std::vector<u64>& b = m_binom[i];
            // the largest x with C(x, i) <= r
            x = static_cast<u64>(std::upper_bound(b.begin(), b.end(), r) - b.begin()) - 1;
            r -= b[x];
        }
        v[i - 1] = x;
    }
    j = 0;
    for (u32 i = 0; i < m_k; ++i)
        if (v[i] >= m_k)
            sorted[j++] = static_cast<u32>(v[i] - m_k);
}

bool LandmarkNoveltyTable::sparse_insert(const u32* sorted, u32 j, u64 c, u32 rank)
{
    TupleSet::Key key;
    if (m_packed)
    {
        u64 t[3] = {0, 0, 0};
        for (u32 i = 0; i < j; ++i)
            t[i] = u64{sorted[i]} + 1;
        key.lo = t[0] | (t[1] << 32);
        key.hi = t[2] | (u64{rank} << 32);
    }
    else
    {
        key.lo = c;
        key.hi = rank;
    }
    return m_sparse.insert(key);
}

void LandmarkNoveltyTable::to_sparse()
{
    if (!m_dense)
        return;
    u32 t[k_max_arity];
    const u64 rows = m_rw ? m_rows.size() / m_rw : 0;
    for (u64 r = 0; r < rows; ++r)
    {
        const u64* row = m_rows.data() + r * m_rw;
        bool any = false;
        for (u32 w = 0; w < m_rw; ++w)
            any = any || row[w] != 0;
        if (!any)
            continue;
        u32 j = 0;
        unrank(r, t, j);
        bits::for_each(row, m_rw, [&](u64 rank) { sparse_insert(t, j, r, static_cast<u32>(rank)); });
    }
    std::vector<u64>().swap(m_rows);
    m_dense = false;
}

void LandmarkNoveltyTable::load_atoms(const u64* w, u32 n)
{
    m_atoms.clear();
    bits::for_each(w, n, [&](u64 a) { m_atoms.push_back(static_cast<u32>(a)); });
    if (!m_atoms.empty() && m_atoms.back() >= m_cap)
        reserve(m_atoms.back() + 1);
    else if (m_cap == 0)
        reserve(1);
}

void LandmarkNoveltyTable::set_ranks(std::span<const u32> ranks)
{
    m_rank_list.assign(ranks.begin(), ranks.end());
    std::fill(m_mask.begin(), m_mask.end(), u64{0});
    for (u32 r : ranks)
        bits::set(m_mask.data(), r);
}

bool LandmarkNoveltyTable::visit(const u32* sorted, u32 j)
{
    if (m_dense)
    {
        u64* row = m_rows.data() + code(sorted, j) * m_rw;
        for (u32 w = 0; w < m_rw; ++w)
        {
            m_novel = m_novel || (m_mask[w] & ~row[w]) != 0;
            row[w] |= m_mask[w];
        }
        return true;
    }
    const u64 c = m_packed ? 0 : code(sorted, j);
    for (u32 r : m_rank_list)
        m_novel = sparse_insert(sorted, j, c, r) || m_novel;
    return true;
}

namespace
{
/// Calls f(out) for every r-combination of items[0, n), in lexicographic order; out holds the chosen items.
template<class F>
void for_each_combination(const u32* items, u32 n, u32 r, F&& f)
{
    if (r > k_max_arity || r > n)
        return;
    u32 idx[k_max_arity];
    u32 out[k_max_arity];
    for (u32 i = 0; i < r && i < k_max_arity; ++i)
        idx[i] = i;
    while (true)
    {
        for (u32 i = 0; i < r && i < k_max_arity; ++i)
            out[i] = items[idx[i]];
        f(static_cast<const u32*>(out));
        // advance the rightmost index that can still move
        u32 i = r;
        while (i > 0 && idx[i - 1] == n - r + (i - 1))
            --i;
        if (i == 0)
            return;
        ++idx[i - 1];
        for (u32 j = i; j < r && j < k_max_arity; ++j)
            idx[j] = idx[j - 1] + 1;
    }
}
}  // namespace

template<class Visit>
void LandmarkNoveltyTable::state_tuples(Visit&& visit_fn)
{
    const u32 m = static_cast<u32>(m_atoms.size());
    if (m + 1 < m_k)
        return;
    for (u32 j = 0; j <= m_k; ++j)
        for_each_combination(m_atoms.data(), m, j, [&](const u32* t) { visit_fn(t, j); });
}

template<class Visit>
void LandmarkNoveltyTable::transition_tuples(const u64* p, u32 np, Visit&& visit_fn)
{
    const u32 m = static_cast<u32>(m_atoms.size());
    if (m + 1 < m_k)
        return;
    u32 buf[k_max_arity];
    m_low.clear();  // the atoms below the current one that are not added
    for (u32 ia = 0; ia < m; ++ia)
    {
        const u32 a = m_atoms[ia];
        if (bits::test(p, np, a))
        {
            m_low.push_back(a);
            continue;  // not added
        }
        // the tuples whose smallest added atom is a: non-added atoms below it, a, then any atoms above it
        for (u32 l = 0; l < m_k && l < k_max_arity; ++l)
            for_each_combination(m_low.data(), static_cast<u32>(m_low.size()), l,
                                 [&](const u32* lower)
                                 {
                                     for (u32 i = 0; i < l; ++i)
                                         buf[i] = lower[i];
                                     buf[l] = a;
                                     for (u32 u = 0; l + 1 + u <= m_k; ++u)
                                         for_each_combination(m_atoms.data() + ia + 1, m - ia - 1, u,
                                                              [&](const u32* upper)
                                                              {
                                                                  for (u32 i = 0; i < u && l + 1 + i < k_max_arity; ++i)
                                                                      buf[l + 1 + i] = upper[i];
                                                                  visit_fn(static_cast<const u32*>(buf), l + 1 + u);
                                                              });
                                 });
    }
}

// ------------------------------------------------------------------------------------------ rank-major tests
bool LandmarkNoveltyTable::fast_state_novel(const u64* w, u32 n, u32 rank) const
{
    if (!m_empty[rank])
        return true;
    if (m_k == 1)
    {
        const u64* A = m_bits.data() + static_cast<usize>(rank) * m_aw;
        for (u32 i = 0; i < n; ++i)
            if (w[i] & ~A[i])
                return true;
        return false;
    }
    const u64* base = m_bits.data() + static_cast<usize>(rank) * m_cap * m_aw;
    for (u32 i = 0; i < n; ++i)
        for (u64 x = w[i]; x; x &= x - 1)
        {
            const u64* P = base + (static_cast<usize>(i) * 64 + static_cast<usize>(bits::ctz64(x))) * m_aw;
            for (u32 j = 0; j < n; ++j)
                if (w[j] & ~P[j])
                    return true;
        }
    return false;
}

void LandmarkNoveltyTable::fast_state_mark(const u64* w, u32 n, u32 rank)
{
    m_empty[rank] = 1;
    if (m_k == 1)
    {
        u64* A = m_bits.data() + static_cast<usize>(rank) * m_aw;
        for (u32 i = 0; i < n; ++i)
            A[i] |= w[i];
        bits::set(m_rows.data(), rank);  // row 0: the empty tuple
        bits::for_each(w, n, [&](u64 a) { bits::set(m_rows.data() + (a + 1) * m_rw, rank); });
        return;
    }
    u64* base = m_bits.data() + static_cast<usize>(rank) * m_cap * m_aw;
    bits::for_each(w, n,
                   [&](u64 a)
                   {
                       u64* P = base + a * m_aw;
                       for (u32 j = 0; j < n; ++j)
                           P[j] |= w[j];
                   });
}

bool LandmarkNoveltyTable::fast_transition_novel(const u64* w, u32 n, std::span<const u32> add, const u64* kept) const
{
    if (m_k == 1)
    {
        for (u32 a : add)
        {
            const u64* row = m_rows.data() + (static_cast<usize>(a) + 1) * m_rw;
            for (u32 i = 0; i < m_rw; ++i)
                if (kept[i] & ~row[i])
                    return true;
        }
        return false;
    }
    for (u32 i = 0; i < m_rw; ++i)
        for (u64 x = kept[i]; x; x &= x - 1)
        {
            const u64* base = m_bits.data() + (static_cast<usize>(i) * 64 + static_cast<usize>(bits::ctz64(x))) * m_cap * m_aw;
            for (u32 a : add)
            {
                const u64* P = base + static_cast<usize>(a) * m_aw;
                for (u32 j = 0; j < n; ++j)
                    if (w[j] & ~P[j])
                        return true;
            }
        }
    return false;
}

void LandmarkNoveltyTable::fast_transition_mark(const u64* w, u32 n, std::span<const u32> add, const u64* kept)
{
    if (m_k == 1)
    {
        for (u32 a : add)
        {
            u64* row = m_rows.data() + (static_cast<usize>(a) + 1) * m_rw;
            for (u32 i = 0; i < m_rw; ++i)
                row[i] |= kept[i];
        }
        bits::for_each(kept, m_rw,
                       [&](u64 r)
                       {
                           u64* A = m_bits.data() + r * m_aw;
                           for (u32 a : add)
                               bits::set(A, a);
                       });
        return;
    }
    bits::for_each(kept, m_rw,
                   [&](u64 r)
                   {
                       u64* base = m_bits.data() + r * m_cap * m_aw;
                       for (u32 a : add)
                       {
                           u64* P = base + static_cast<usize>(a) * m_aw;
                           for (u32 j = 0; j < n; ++j)
                               P[j] |= w[j];
                           bits::for_each(w, n, [&](u64 b) { bits::set(base + b * m_aw, a); });
                       }
                   });
}

bool LandmarkNoveltyTable::fast_successor(const u64* w, u32 n, std::span<const u32> add, const u64* flipped, const u64* kept)
{
    if (m_k == 2 && !bits::any(w, n))
        return false;  // no tuples (fewer than k - 1 atoms)
    bool novel = false;
    for (u32 i = 0; i < m_rw && !novel; ++i)
        for (u64 x = flipped[i]; x && !novel; x &= x - 1)
            novel = fast_state_novel(w, n, i * 64 + static_cast<u32>(bits::ctz64(x)));
    if (!novel)
        novel = fast_transition_novel(w, n, add, kept);
    if (!novel)
        return false;  // every pair is marked
    bits::for_each(flipped, m_rw, [&](u64 r) { fast_state_mark(w, n, static_cast<u32>(r)); });
    fast_transition_mark(w, n, add, kept);
    return true;
}

bool LandmarkNoveltyTable::mark_successor(const u64* p, u32 np, const u64* w, u32 n, std::span<const u32> add, const u64* flipped, const u64* kept)
{
    if (m_fast != 0)
    {
        if (m_cap == 0 || u64{n} * 64 > m_cap)
            reserve(std::max<u32>(n * 64, 1));
        if (m_fast != 0)
            return fast_successor(w, n, add, flipped, kept);
    }
    std::vector<u32> f, k;
    bits::for_each(flipped, m_rw, [&](u64 r) { f.push_back(static_cast<u32>(r)); });
    bits::for_each(kept, m_rw, [&](u64 r) { k.push_back(static_cast<u32>(r)); });
    bool novel = false;
    if (!f.empty())
        novel = mark_state(w, n, f) || novel;
    if (!k.empty())
        novel = mark_transition(p, np, w, n, k) || novel;
    return novel;
}

bool LandmarkNoveltyTable::mark_state(const u64* w, u32 n, std::span<const u32> ranks)
{
    if (m_fast != 0)
    {
        if (m_cap == 0 || u64{n} * 64 > m_cap)
            reserve(std::max<u32>(n * 64, 1));
        if (m_fast != 0)
        {
            if (m_k == 2 && !bits::any(w, n))
                return false;
            bool novel = false;
            for (u32 r : ranks)
                novel = novel || fast_state_novel(w, n, r);
            if (novel)
                for (u32 r : ranks)
                    fast_state_mark(w, n, r);
            return novel;
        }
    }
    load_atoms(w, n);
    set_ranks(ranks);
    m_novel = false;
    state_tuples([&](const u32* t, u32 j) { visit(t, j); });
    return m_novel;
}

bool LandmarkNoveltyTable::mark_transition(const u64* p, u32 np, const u64* w, u32 n, std::span<const u32> ranks)
{
    if (m_fast != 0)
    {
        if (m_cap == 0 || u64{n} * 64 > m_cap)
            reserve(std::max<u32>(n * 64, 1));
        if (m_fast != 0)
        {
            if (m_k == 2 && !bits::any(w, n))
                return false;
            m_low.clear();  // the added atoms
            bits::for_each(w, n,
                           [&](u64 a)
                           {
                               if (!bits::test(p, np, a))
                                   m_low.push_back(static_cast<u32>(a));
                           });
            set_ranks(ranks);
            if (!fast_transition_novel(w, n, m_low, m_mask.data()))
                return false;
            fast_transition_mark(w, n, m_low, m_mask.data());
            return true;
        }
    }
    load_atoms(w, n);
    set_ranks(ranks);
    m_novel = false;
    transition_tuples(p, np, [&](const u32* t, u32 j) { visit(t, j); });
    return m_novel;
}

u64 LandmarkNoveltyTable::bytes() const noexcept
{
    u64 b = m_rows.capacity() * 8 + m_sparse.bytes() + m_bits.capacity() * 8 + m_empty.capacity();
    for (const auto& v : m_binom)
        b += v.capacity() * 8;
    return b;
}
}  // namespace mymyr::novelty
