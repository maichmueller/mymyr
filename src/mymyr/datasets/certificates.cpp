// Colour refinement and k-FWL certificates: ports of mimir's src/graphs/algorithms/color_refinement.hpp and
// folklore_weisfeiler_leman.hpp (see datasets/certificates.hpp for the deviations).

#include "mymyr/datasets/certificates.hpp"

#include "mymyr/core/hash.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
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

Certificate kfwl_certificate(const ObjectGraph& g, u32 k)
{
    if (k != 2 && k != 3)
        throw std::invalid_argument("mymyr: k-FWL certificates support k = 2 and k = 3");
    const u64 n = g.num_vertices();
    u64 tuples = 1;
    for (u32 i = 0; i < k; ++i)
        tuples *= n;
    if (tuples > (u64{1} << 26))
        throw std::length_error("mymyr: k-FWL certificate of a graph with too many vertex tuples (n^k > 2^26)");
    auto adjacent = [&](u32 a, u32 b)
    {
        const auto adj = g.adjacent(a);
        return std::binary_search(adj.begin(), adj.end(), b);
    };
    // tuple t <-> (v_0, ..., v_{k-1}) with v_0 least significant (as in mimir's tuple_to_hash)
    auto decode = [&](u64 t, u32* v)
    {
        for (u32 i = 0; i < k; ++i)
        {
            v[i] = static_cast<u32>(t % n);
            t /= n;
        }
    };
    std::vector<u64> weight(k, 1);
    for (u32 i = 1; i < k; ++i)
        weight[i] = weight[i - 1] * n;
    // initial colours: the ordered isomorphism type (colours of the positions, equalities, adjacencies)
    std::map<std::vector<u32>, u32> types;
    std::vector<std::vector<u32>> type_of(tuples);
    std::vector<u32> key;
    for (u64 t = 0; t < tuples; ++t)
    {
        u32 v[3] = {0, 0, 0};
        decode(t, v);
        key.clear();
        for (u32 i = 0; i < k; ++i)
            key.push_back(g.color[v[i]]);
        for (u32 i = 0; i < k; ++i)
            for (u32 j = i + 1; j < k; ++j)
                key.push_back((v[i] == v[j] ? 1u : 0u) | (adjacent(v[i], v[j]) ? 2u : 0u));
        types.emplace(key, 0);
        type_of[t] = key;
    }
    u32 next = 0;
    for (auto& [kk, id] : types)
        id = next++;
    std::vector<u32> colors(tuples);
    for (u64 t = 0; t < tuples; ++t)
        colors[t] = types.at(type_of[t]);
    std::vector<std::vector<u32>>().swap(type_of);
    std::vector<u32> items;
    std::vector<std::array<u32, 3>> runs(n);
    auto f = refine(colors, k,
                    [&](u32 t, std::vector<u32>& out)
                    {
                        u32 v[3] = {0, 0, 0};
                        decode(t, v);
                        items.clear();
                        for (u32 u = 0; u < n; ++u)
                            for (u32 i = 0; i < k; ++i)
                            {
                                // the tuple with position i replaced by u
                                const u64 x = t + (static_cast<u64>(u) - v[i]) * weight[i];
                                items.push_back(colors[x]);
                            }
                        // sort the k-tuples of colours (runs of k) lexicographically
                        for (u32 u = 0; u < n; ++u)
                            for (u32 i = 0; i < 3; ++i)
                                runs[u][i] = i < k ? items[u * k + i] : 0;
                        std::sort(runs.begin(), runs.end());
                        for (const auto& r : runs)
                            out.insert(out.end(), r.begin(), r.begin() + k);
                    });
    Digest d;
    d.add(100 + k);
    digest_palette(d, g);
    d.add(types.size());
    for (const auto& [kk, id] : types)
    {
        d.add(kk.size());
        for (u32 x : kk)
            d.add(x);
    }
    digest_result(d, f, colors);
    return d.done();
}
}  // namespace mymyr::datasets
