// Colour refinement and k-FWL certificates: ports of mimir's src/graphs/algorithms/color_refinement.hpp and
// folklore_weisfeiler_leman.hpp (see datasets/certificates.hpp for the deviations).

#include "mymyr/datasets/certificates.hpp"

#include "mymyr/core/hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mymyr::datasets
{
namespace
{
/// Two independent 64-bit lanes over a u64 stream.
struct Digest
{
    u64 a = 0x6a09e667f3bcc908ULL, b = 0xbb67ae8584caa73bULL;
    void add(u64 x)
    {
        a = hash::combine(a, x);
        b = hash::combine(b ^ 0x3c6ef372fe94f82bULL, x + 0xa54ff53a5f1d36f1ULL);
    }
    [[nodiscard]] Certificate done() const { return {hash::mix64(a), hash::mix64(b ^ a)}; }
};

/// One refinement entry: (old colour, sorted signature, element); the signature is a run of `width` u32 per item.
struct Entry
{
    u32 old = 0;
    u32 begin = 0, count = 0;  // signature items in the pool
    u32 element = 0;
};

/// The work-list refinement shared by colour refinement and k-FWL (as in mimir's replace_tuples/split_color_classes).
/// `colors` holds the current colour of every element; `signature(e, out)` appends the sorted signature items of e
/// (each `width` u32). Refines until no class splits; returns the decoding table in canonical order.
template<class Signature>
std::vector<std::tuple<u32, std::vector<u32>, u32>> refine(std::vector<u32>& colors, u32 width, Signature&& signature)
{
    u32 max_color = 0;
    for (u32 c : colors)
        max_color = std::max(max_color, c);
    std::map<std::pair<u32, std::vector<u32>>, u32> f;
    std::vector<u32> pool;
    std::vector<Entry> M;
    auto sig_less = [&](const Entry& x, const Entry& y)
    {
        if (x.old != y.old)
            return x.old < y.old;
        const bool lex = std::lexicographical_compare(pool.begin() + x.begin, pool.begin() + x.begin + x.count, pool.begin() + y.begin,
                                                      pool.begin() + y.begin + y.count);
        if (lex)
            return true;
        const bool rlex = std::lexicographical_compare(pool.begin() + y.begin, pool.begin() + y.begin + y.count, pool.begin() + x.begin,
                                                       pool.begin() + x.begin + x.count);
        if (rlex)
            return false;
        return x.element < y.element;
    };
    auto same_sig = [&](const Entry& x, const Entry& y)
    {
        return x.count == y.count && std::equal(pool.begin() + x.begin, pool.begin() + x.begin + x.count, pool.begin() + y.begin);
    };
    for (bool changed = true; changed;)
    {
        changed = false;
        pool.clear();
        M.clear();
        for (u32 e = 0; e < colors.size(); ++e)
        {
            const u32 begin = static_cast<u32>(pool.size());
            signature(e, pool);
            M.push_back({colors[e], begin, static_cast<u32>(pool.size()) - begin, e});
        }
        std::sort(M.begin(), M.end(), sig_less);
        // split every class whose members differ in their signatures: each distinct signature gets a new colour, in
        // the order of the signatures (canonical)
        for (usize i = 0; i < M.size();)
        {
            usize j = i;
            bool split = false;
            while (j < M.size() && M[j].old == M[i].old)
            {
                split |= !same_sig(M[i], M[j]);
                ++j;
            }
            if (split)
            {
                changed = true;
                for (usize k = i; k < j;)
                {
                    const u32 nc = ++max_color;
                    f.emplace(std::make_pair(M[k].old, std::vector<u32>(pool.begin() + M[k].begin, pool.begin() + M[k].begin + M[k].count)), nc);
                    usize l = k;
                    while (l < j && same_sig(M[k], M[l]))
                        colors[M[l++].element] = nc;
                    k = l;
                }
            }
            i = j;
        }
    }
    // stable classes: report their neighbourhood structure too
    for (usize i = 0; i < M.size(); ++i)
        if (i == 0 || M[i].old != M[i - 1].old || !same_sig(M[i - 1], M[i]))
            f.emplace(std::make_pair(M[i].old, std::vector<u32>(pool.begin() + M[i].begin, pool.begin() + M[i].begin + M[i].count)), M[i].old);
    std::vector<std::tuple<u32, std::vector<u32>, u32>> out;
    out.reserve(f.size());
    for (auto& [k, v] : f)
        out.emplace_back(k.first, k.second, v);
    (void) width;
    return out;
}

void digest_palette(Digest& d, const ObjectGraph& g)
{
    d.add(g.num_colors());
    for (u32 c = 0; c < g.num_colors(); ++c)
    {
        const auto s = g.palette(c);
        d.add(s.size());
        for (u32 x : s)
            d.add(x);
    }
}

void digest_result(Digest& d, const std::vector<std::tuple<u32, std::vector<u32>, u32>>& f, const std::vector<u32>& colors)
{
    d.add(f.size());
    for (const auto& [old, sig, nc] : f)
    {
        d.add(old);
        d.add(sig.size());
        for (u32 x : sig)
            d.add(x);
        d.add(nc);
    }
    // histogram of the stable colours
    std::vector<u32> sorted = colors;
    std::sort(sorted.begin(), sorted.end());
    d.add(sorted.size());
    for (usize i = 0; i < sorted.size();)
    {
        usize j = i;
        while (j < sorted.size() && sorted[j] == sorted[i])
            ++j;
        d.add(sorted[i]);
        d.add(j - i);
        i = j;
    }
}
}  // namespace

Certificate color_refinement_certificate(const ObjectGraph& g, std::vector<u32>* stable_colors)
{
    std::vector<u32> colors = g.color;  // the palette is sorted: colour ids are the canonical initial colours
    std::vector<u32> nb;
    auto f = refine(colors, 1,
                    [&](u32 v, std::vector<u32>& out)
                    {
                        nb.clear();
                        for (u32 u : g.adjacent(v))
                            nb.push_back(colors[u]);
                        std::sort(nb.begin(), nb.end());
                        out.insert(out.end(), nb.begin(), nb.end());
                    });
    Digest d;
    d.add(1);
    digest_palette(d, g);
    digest_result(d, f, colors);
    if (stable_colors)
        *stable_colors = colors;
    return d.done();
}

namespace
{
/// One k-tuple in a k-FWL round: its signature (a, b), its colour before the round and its index.
struct KfwlEntry
{
    u64 a = 0, b = 0;
    u32 old = 0, tuple = 0;
};
static_assert(sizeof(KfwlEntry) + sizeof(u32) == kfwl_bytes_per_tuple);

/// n^e, saturated at 2^64 - 1.
u64 saturated_power(u64 n, u32 e)
{
    u64 r = 1;
    for (u32 i = 0; i < e; ++i)
    {
        if (n != 0 && r > ~u64{0} / n)
            return ~u64{0};
        r *= n;
    }
    return r;
}

/// Sorts the entries by (old colour, signature), gives every distinct key a dense new colour in that order and
/// digests the classes as (old colour, a, b, size). Returns the number of classes.
u32 assign_classes(std::vector<KfwlEntry>& entries, std::vector<u32>& colors, Digest& d)
{
    std::sort(entries.begin(), entries.end(), [](const KfwlEntry& x, const KfwlEntry& y)
              { return x.old != y.old ? x.old < y.old : x.a != y.a ? x.a < y.a : x.b < y.b; });
    u32 classes = 0;
    for (usize i = 0; i < entries.size();)
    {
        usize j = i;
        while (j < entries.size() && entries[j].old == entries[i].old && entries[j].a == entries[i].a && entries[j].b == entries[i].b)
            colors[entries[j++].tuple] = classes;
        d.add(entries[i].old);
        d.add(entries[i].a);
        d.add(entries[i].b);
        d.add(j - i);
        ++classes;
        i = j;
    }
    d.add(classes);
    return classes;
}

/// Adds the two 64-bit hashes of a colour k-tuple, packed as x = (c_0, c_1) and y = (c_2, c_3) (zero beyond k), to
/// the signature lanes. The lanes use different seeds and orders, so a signature collision needs both 64-bit sums to
/// collide.
template<u32 K>
inline void hash_item(u64 x, u64 y, u64& a, u64& b)
{
    if constexpr (K == 2)
    {
        (void)y;
        a += hash::mix64(x ^ 0x243f6a8885a308d3ULL);
        b += hash::mix64(x ^ 0x13198a2e03707344ULL);
    }
    else
    {
        a += hash::mix64(hash::mix64(x ^ 0xa4093822299f31d0ULL) ^ y);
        b += hash::mix64(hash::mix64(y ^ 0x082efa98ec4e6c89ULL) ^ x ^ 0x452821e638d01377ULL);
    }
}

template<u32 K>
Certificate kfwl(const ObjectGraph& g)
{
    const u32 n = g.num_vertices();
    const u64 T = saturated_power(n, K);
    // tuple t <-> (v_0, ..., v_{K-1}) with v_0 least significant (as in mimir's tuple_to_hash)
    std::array<u64, K> weight{};
    weight[0] = 1;
    for (u32 i = 1; i < K; ++i)
        weight[i] = weight[i - 1] * n;
    Digest d;
    d.add(100 + K);
    digest_palette(d, g);
    d.add(n);
    std::vector<u32> colors(T);
    std::vector<KfwlEntry> entries(T);
    // initial colours: the ordered isomorphism type, packed as (colour of v_0, ..., colour of v_{K-1}, then per pair
    // i < j: 1 if v_i = v_j, 2 if adjacent) with v_0's colour most significant, so that the packed keys sort as these
    // sequences do
    const u32 color_bits = static_cast<u32>(std::bit_width(std::max<u32>(g.num_colors(), 1) - 1));
    if (color_bits * K + K * (K - 1) > 64)
        throw std::length_error("mymyr: " + std::to_string(K) + "-FWL certificate of a graph with " + std::to_string(g.num_colors()) +
                                " colours: the ordered isomorphism types do not fit 64 bits");
    std::vector<u64> adjacency((static_cast<u64>(n) * n + 63) / 64, 0);  // n x n bit matrix
    for (u32 x = 0; x < n; ++x)
        for (u32 y : g.adjacent(x))
        {
            const u64 bit = static_cast<u64>(x) * n + y;
            adjacency[bit / 64] |= u64{1} << (bit % 64);
        }
    auto adjacent = [&](u32 x, u32 y)
    {
        const u64 bit = static_cast<u64>(x) * n + y;
        return ((adjacency[bit / 64] >> (bit % 64)) & 1) != 0;
    };
    std::array<u32, K> v{};
    for (u64 t = 0; t < T; ++t)
    {
        u64 key = 0;
        for (u32 i = 0; i < K; ++i)
            key = (key << color_bits) | g.color[v[i]];
        for (u32 i = 0; i < K; ++i)
            for (u32 j = i + 1; j < K; ++j)
                key = (key << 2) | (v[i] == v[j] ? 1u : 0u) | (adjacent(v[i], v[j]) ? 2u : 0u);
        entries[t] = {key, 0, 0, static_cast<u32>(t)};
        for (u32 i = 0; i < K && ++v[i] == n; ++i)
            v[i] = 0;
    }
    u32 classes = assign_classes(entries, colors, d);
    // refinement: the signature of t is the multiset over u of (C(t[0 <- u]), ..., C(t[K-1 <- u]))
    for (;;)
    {
        v.fill(0);
        for (u64 t = 0; t < T; ++t)
        {
            std::array<const u32*, K> col{};  // col[i][u * weight[i]] = C(t[i <- u])
            for (u32 i = 0; i < K; ++i)
                col[i] = colors.data() + (t - v[i] * weight[i]);
            u64 a = 0, b = 0;
            for (u32 u = 0; u < n; ++u)
            {
                const u64 x = col[0][u] | (u64{col[1][u * weight[1]]} << 32);
                u64 y = 0;
                if constexpr (K >= 3)
                    y = col[2][u * weight[2]];
                if constexpr (K >= 4)
                    y |= u64{col[3][u * weight[3]]} << 32;
                hash_item<K>(x, y, a, b);
            }
            entries[t] = {a, b, colors[t], static_cast<u32>(t)};
            for (u32 i = 0; i < K && ++v[i] == n; ++i)
                v[i] = 0;
        }
        const u32 next = assign_classes(entries, colors, d);
        if (next == classes)
            break;  // no class split: the colouring is stable (and the colours are unchanged)
        classes = next;
    }
    return d.done();
}
}  // namespace

Certificate kfwl_certificate(const ObjectGraph& g, u32 k, const KfwlLimits& limits)
{
    if (k < 2 || k > 4)
        throw std::invalid_argument("mymyr: k-FWL certificates support k = 2, 3 and 4");
    const u64 n = g.num_vertices();
    const u64 tuples = saturated_power(n, k), work = saturated_power(n, k + 1);
    const std::string what = "mymyr: " + std::to_string(k) + "-FWL certificate of a graph with n = " + std::to_string(n) + " vertices: ";
    // tuple indices are u32
    const u64 max_tuples = std::min<u64>(limits.max_tuples, u64{0xffffffff});
    if (tuples > max_tuples)
        throw std::length_error(what + "n^" + std::to_string(k) + " vertex tuples exceed the limit of " + std::to_string(max_tuples) +
                                " (KfwlLimits::max_tuples; " + std::to_string(kfwl_bytes_per_tuple) + " bytes per tuple)");
    if (work > limits.max_round_work)
        throw std::length_error(what + "n^" + std::to_string(k + 1) + " colour tuples per round exceed the limit of " +
                                std::to_string(limits.max_round_work) + " (KfwlLimits::max_round_work)");
    switch (k)
    {
    case 2: return kfwl<2>(g);
    case 3: return kfwl<3>(g);
    default: return kfwl<4>(g);
    }
}
}  // namespace mymyr::datasets
