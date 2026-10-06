#pragma once
// The view, count-cache and pick kernels of the lifted successor generator (include/mymyr/cuda/lifted.hpp: launch_view,
// launch_count_keys, launch_pick and their multi-instance forms), shared by lifted.cu (one instance: the task view is a
// kernel parameter) and lifted_multi*.cu (several instances: each thread resolves its row's instance view). The
// template flag M selects the form; everything else is the same code. Internal: included by .cu files only (nvcc,
// C++20; the device-code subset).

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted
{
namespace
{
/// multi_row of launch position j; in a range launch (Multi::starts) false at once for a position outside the launch's
/// keys (before reading its row). The positions keep their threads (a range does not compact them): the blocks of
/// concurrent launches over few rows then spread over the SMs as without ranges.
template<u32 OW>
__device__ __forceinline__ bool env_row(const Multi& m, u64 j, u32& li, u32& k)
{
    if (m.starts)
    {
        // the three loads issued together (the starts do not wait for the flag)
        const u64 s0 = m.starts[m.key_lo], s1 = m.starts[m.key_hi], at = m.first + j;
        if ((at < s0 || at >= s1) && !(m.order_bad && *m.order_bad))
            return false;
    }
    return multi_row<OW>(m, j, li, k);
}

/// A multi launch's range (Multi::starts) is valid: an order to range over, and keys in order.
inline bool range_valid(const Multi& m) { return !m.starts || (m.order && m.key_lo <= m.key_hi); }

// ------------------------------------------------------------------------------------------------ views

/// Warps (states) per block of k_view, and the largest view it builds in shared memory (larger ones are built with
/// global atomics in place).
constexpr u32 k_view_warps = 8;
constexpr u64 k_view_shared_bytes = 48 * 1024;

/// One warp per state: set the view bits of the state's fluent (and derived) atoms in `view` (zeroed).
__device__ __forceinline__ void view_bits(const TaskView& t, const Parents& p, u64 row, u64* view, u32 lane)
{
    const u32 ow = t.ow;
    // an op's rows are set independently: the compact view may keep a predicate's backward rows and not its forward
    // rows (rl/task_arrays_view.hpp)
    auto set = [&](const u32* op)
    {
        if (op[0] != k_none)
            atomicOr(reinterpret_cast<unsigned long long*>(view + u64{op[0]} * ow + (op[1] >> 6)), 1ull << (op[1] & 63));
        if (op[2] != k_none)
            atomicOr(reinterpret_cast<unsigned long long*>(view + u64{op[2]} * ow + (op[3] >> 6)), 1ull << (op[3] & 63));
    };
    const u64* st = p.data + row * p.stride;
    const u32 nf = p.words * 64u < t.fluent_slots ? p.words * 64u : t.fluent_slots;  // bits past the export are unset
    for (u32 i = lane; i < nf; i += 32)
        if ((st[i >> 6] >> (i & 63)) & 1)
            set(t.fluent_view_op + u64{i} * 4);
    if (p.derived)
    {
        const u64* d = p.derived + row * p.derived_words;
        const u32 nd = p.derived_words * 64u < t.derived_slots ? p.derived_words * 64u : t.derived_slots;
        for (u32 i = lane; i < nd; i += 32)
            if ((d[i >> 6] >> (i & 63)) & 1)
                set(t.derived_view_op + u64{i} * 4);
    }
}

/// The view of row `row` of instance t (its view_rows * ow words; `words` of them zeroed first).
template<bool Shared>
__device__ __forceinline__ void view_row(const TaskView& t, const Parents& p, const Views& v, u64 row, u64 words, u64* sm,
                                         u32 lane)
{
    u64* view = v.data + row * v.words;
    u64* build = Shared ? sm + u64{threadIdx.x >> 5} * v.words : view;
    for (u64 i = lane; i < words; i += 32)
        build[i] = 0;
    __syncwarp();
    view_bits(t, p, row, build, lane);
    if constexpr (Shared)
    {
        __syncwarp();
        for (u64 i = lane; i < words; i += 32)
            view[i] = build[i];
    }
}

template<bool Shared, bool M>
__global__ void __launch_bounds__(k_view_warps * 32) k_view(TaskView t, Multi m, Parents p, Views v)
{
    extern __shared__ u64 sm[];
    const u64 warp = u64{blockIdx.x} * k_view_warps + (threadIdx.x >> 5);
    const u32 lane = threadIdx.x & 31;
    if (warp >= live_rows(p))
        return;
    if constexpr (M)
    {
        u32 li, k;
        if (!env_row<0>(m, warp, li, k))
            return;
        const TaskView& tk = m.views[k];
        if (m.ow && tk.ow != m.ow)
            return;
        view_row<Shared>(tk, p, v, li, u64{tk.view_rows} * tk.ow, sm, lane);
    }
    else
        view_row<Shared>(t, p, v, warp, v.words, sm, lane);
}

template<bool M>
cudaError_t launch_view_any(const TaskView& t, const Multi& m, Parents p, Views v, cudaStream_t s)
{
    if (p.rows == 0 || v.words == 0)
        return cudaSuccess;
    const u64 blocks = (u64{p.rows} + k_view_warps - 1) / k_view_warps;
    // several instances: built in place (global atomics). Shared memory sized for the widest instance's view would
    // limit the occupancy of every row, and building in place is as fast as the single-instance launches' shared
    // build
    if constexpr (!M)
    {
        const u64 smem = k_view_warps * v.words * sizeof(u64);
        if (smem <= k_view_shared_bytes)
        {
            k_view<true, M><<<static_cast<unsigned>(blocks), k_view_warps * 32, smem, s>>>(t, m, p, v);
            return cudaGetLastError();
        }
    }
    k_view<false, M><<<static_cast<unsigned>(blocks), k_view_warps * 32, 0, s>>>(t, m, p, v);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------ count cache

/// The region of `schema` in instance k's rows of the count cache (k_none: not recorded).
__device__ __forceinline__ u32 cache_region(const CountCache& c, u32 k, u32 schema)
{
    return c.regions ? c.region[u64{k} * c.region_stride + schema] : k_none;
}

/// The offset of region r in a cache row (2 * seg_cap u32 words: seg_cap packed keys or 64 * seg_cap rank bits, each
/// 64-bit word as its low and high u32 word).
__device__ __forceinline__ u64 region_at(const CountCache& c, u32 r)
{
    return c.num_schemas + 2 + 2 * u64{r} * c.seg_cap;
}

/// Whether a segment of n bindings of a schema with a rank table of `size` ranks has its bitmap in the region
/// (CountCache: long and dense segments).
__device__ __forceinline__ bool rank_written(const CountCache& c, u32 n, u32 size)
{
    return n > c.seg_cap && n >= k_rank_density * ((size + 63) >> 6);
}

/// The rank table of `schema` in instance k (CountCache::rank_table: its region holds the segment's rank bitmap), or
/// null (its region holds keys, or it is not recorded).
__device__ __forceinline__ const u32* cache_ranks(const CountCache& c, u32 k, u32 schema)
{
    if (!c.rank)
        return nullptr;
    const u32 off = c.rank[u64{k} * c.region_stride + schema];
    return off == k_none ? nullptr : c.rank_table + off;
}

/// Counts one (state, schema) segment into its cache row and, for a recorded schema, records the segment in the
/// schema's region: with a rank table (R: the launch has rank tables), the bitmap of its bindings' canonical ranks for
/// long dense segments, else, for at most seg_cap bindings, their packed keys in canonical order.
template<u32 OW, bool FC, bool R>
__device__ __forceinline__ void count_keys_one(const GenArgs& a, const TaskView& t, const CountCache& c, u32 k, u32 si,
                                               const u64* view, const u64* row, const u64* der, u32* crow)
{
    const u32 schema = a.set.schemas[si];
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[a.set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const StateRef s{row, a.p.words, der, a.p.derived_words};
    u32 bind[k_max_depth];
    auto run = [&](auto& emit)
    {
        if constexpr (FC)
            run_fc<OW>(t, mx, s, view, bind, emit);
        else
            run_fixed<OW>(t, mx, s, view, bind, emit);
    };
    u32 n = 0;
    const u32 r = cache_region(c, k, schema);
    if (r == k_none)
    {
        auto emit = [&](const u32*) { ++n; };
        run(emit);
        crow[schema] = n;
        return;
    }
    const u32 arity = sc[k_sc_arity];
    if (const u32* rt = R ? cache_ranks(c, k, schema) : nullptr)
    {
        // the ranks' bitmap, in registers, a binding's rank the sum of its objects' shares; a binding outside the rank
        // space (none: the rank space spans the matcher's static domains) is left out, and the pick, seeing fewer bits
        // than the count, searches
        const u32 no = t.num_objects;
        const u32* share = rt + 1;
        u64 bm[k_max_rank_words] = {};
        auto emit = [&](const u32* b)
        {
            u32 x = 0;
            for (u32 i = 0; i < arity; ++i)
                x += __ldg(share + u64{i} * no + b[i]);
            const u32 xw = x >> 6;  // past every word for a share k_rank_outside
            const u64 bit = 1ull << (x & 63);
#pragma unroll
            for (u32 i = 0; i < k_max_rank_words; ++i)
                bm[i] |= i == xw ? bit : 0;
            ++n;
        };
        run(emit);
        crow[schema] = n;
        if (!rank_written(c, n, rt[0]))
            return;  // a short or sparse segment: its picks search
        const u32 nw = (rt[0] + 63) >> 6;
        u32* dst = crow + region_at(c, r);
#pragma unroll
        for (u32 i = 0; i < k_max_rank_words; ++i)
            if (i < nw)
            {
                dst[2 * i] = static_cast<u32>(bm[i]);
                dst[2 * i + 1] = static_cast<u32>(bm[i] >> 32);
            }
        return;
    }
    u64 keys[k_max_seg_cap];
    const u32 bits = key_bits(t.num_objects);
    auto emit = [&](const u32* b)
    {
        if (n < c.seg_cap)
        {
            u64 key = 0;
            for (u32 i = 0; i < arity; ++i)
                key = (key << bits) | b[i];
            keys[n] = key;
        }
        ++n;
    };
    run(emit);
    crow[schema] = n;
    if (n > c.seg_cap)
        return;  // a long segment: its picks search
    for (u32 i = 1; i < n; ++i)
    {
        const u64 key = keys[i];
        u32 j = i;
        for (; j > 0 && keys[j - 1] > key; --j)
            keys[j] = keys[j - 1];
        keys[j] = key;
    }
    u32* dst = crow + region_at(c, r);  // after the segment (its address held through the search cost registers)
    for (u32 i = 0; i < n; ++i)
    {
        dst[2 * i] = static_cast<u32>(keys[i]);
        dst[2 * i + 1] = static_cast<u32>(keys[i] >> 32);
    }
}

template<u32 OW, bool FC, bool M, bool R>
__device__ __forceinline__ void count_keys_row(const GenArgs& a, const TaskView& t, const CountCache& c, u32 first,
                                               u32 li, u32 k, u32 warp, u32 nwarps)
{
    const Parents& p = a.p;
    u32* crow = c.data + u64{li} * c.stride;
    const u64* row = p.data + u64{li} * p.stride;
    if (first && c.regions && warp == 0)
    {
        const u64 f = row_fingerprint(row, p.words);
        crow[c.num_schemas] = static_cast<u32>(f);
        crow[c.num_schemas + 1] = static_cast<u32>(f >> 32);
    }
    const u64* view = a.v.data + u64{li} * a.v.words;
    const u64* der = p.derived ? p.derived + u64{li} * p.derived_words : nullptr;
    for (u32 si = warp; si < a.set.count; si += nwarps)
    {
        if constexpr (M)
        {
            const u32 schema = a.set.schemas[si];
            if (!FC && a.m.fc_of[u64{k} * a.m.num_schemas + schema] == 2)
                crow[schema] = 0;  // never applicable in this instance (the row may have held another instance's count)
            if (!multi_kind<FC>(a.m, k, schema))
                continue;
        }
        count_keys_one<OW, FC, R>(a, t, c, k, si, view, row, der, crow);
    }
}

/// k_gen's mapping: k_tile states per block, one lane per state, the warps taking the schemas in turn. R: the cache
/// has rank tables (their bitmaps' registers are the other launches' occupancy).
template<u32 OW, bool FC, bool M, bool R>
__global__ void __launch_bounds__(k_gen_block) k_count_keys(GenArgs a, CountCache c, u32 first)
{
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const u32 j = blockIdx.x * k_tile + lane;
    if (j >= a.p.rows)
        return;
    if constexpr (M)
    {
        u32 li, k;
        if (!env_row<OW>(a.m, j, li, k))
            return;
        count_keys_row<OW, FC, true, R>(a, a.m.views[k], c, first, li, k, warp, nwarps);
    }
    else
        count_keys_row<OW, FC, false, R>(a, a.t, c, first, j, 0, warp, nwarps);
}

template<u32 OW, bool FC, bool M>
cudaError_t launch_count_keys_ow(const GenArgs& a, const CountCache& c, bool first, cudaStream_t s)
{
    const u64 blocks = (u64{a.p.rows} + k_tile - 1) / k_tile;
    if (blocks == 0)
        return cudaSuccess;
    if (blocks > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    const u32 n = a.set.count ? a.set.count : 1;
    const u32 warps = n < k_gen_block / 32 ? n : k_gen_block / 32;
    if (c.rank)
        k_count_keys<OW, FC, M, true><<<static_cast<unsigned>(blocks), warps * 32, 0, s>>>(a, c, first ? 1u : 0u);
    else
        k_count_keys<OW, FC, M, false><<<static_cast<unsigned>(blocks), warps * 32, 0, s>>>(a, c, first ? 1u : 0u);
    return cudaGetLastError();
}

template<bool M>
cudaError_t dispatch_count_keys(const GenArgs& a, const CountCache& c, bool first, cudaStream_t s)
{
    if (!c.data || c.num_schemas != a.set.num_schemas || c.seg_cap > k_max_seg_cap ||
        c.stride < cache_stride(c.num_schemas, c.regions, c.seg_cap) || (c.regions && !c.region) ||
        (c.rank && (!c.regions || !c.rank_table)) || a.set.deep)
        return cudaErrorInvalidValue;
    const u32 ow = M ? a.m.ow : a.t.ow;
    if (a.set.fc)
    {
        switch (ow)
        {
            case 1: return launch_count_keys_ow<1, true, M>(a, c, first, s);
            case 2: return launch_count_keys_ow<2, true, M>(a, c, first, s);
            case 3: return launch_count_keys_ow<3, true, M>(a, c, first, s);
            case 4: return launch_count_keys_ow<4, true, M>(a, c, first, s);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (ow)
    {
        case 1: return launch_count_keys_ow<1, false, M>(a, c, first, s);
        case 2: return launch_count_keys_ow<2, false, M>(a, c, first, s);
        case 3: return launch_count_keys_ow<3, false, M>(a, c, first, s);
        case 4: return launch_count_keys_ow<4, false, M>(a, c, first, s);
        case 5: return launch_count_keys_ow<5, false, M>(a, c, first, s);
        case 6: return launch_count_keys_ow<6, false, M>(a, c, first, s);
        case 7: return launch_count_keys_ow<7, false, M>(a, c, first, s);
        case 8: return launch_count_keys_ow<8, false, M>(a, c, first, s);
        default: return cudaErrorInvalidValue;
    }
}

// ------------------------------------------------------------------------------------------------ picks

/// Parent li's pick from its cache row: true if the row's region holds the pick (chosen = its binding), false if the
/// pick must search. A row that does not belong to the parent (states written without a refresh) sets `bad`. R: the
/// cache has rank tables.
template<u32 OW, bool R>
__device__ __forceinline__ bool pick_cached(const TaskView& t, const u32* mx, const Picks& pk, const Parents& p, u32 k,
                                            u32 schema, u32 arity, u32 li, u32* chosen, bool& bad)
{
    const CountCache& c = pk.cache;
    const u32 r = cache_region(c, k, schema);
    if (r == k_none)
        return false;
    const u32 S = c.num_schemas;
    const u32* crow = c.data + u64{li} * c.stride;
    const u32 n = crow[schema], rank = pk.rank[li];
    const u32* rt = R ? cache_ranks(c, k, schema) : nullptr;
    if (rt ? !rank_written(c, n, rt[0]) : n > c.seg_cap)
        return false;
    const u64 f = row_fingerprint(p.data + u64{li} * p.stride, p.words);
    if (crow[S] != static_cast<u32>(f) || crow[S + 1] != static_cast<u32>(f >> 32) || rank >= n)
    {
        bad = true;
        return false;
    }
    const u32* e = crow + region_at(c, r);
    if (R && rt)
    {
        // the rank-th set bit (its binding: RankSpace's ranks are the table's); fewer bits than bindings (one outside
        // the rank space): search
        const RankSpace<OW> ranks(t, mx, arity);
        const u32 nw = (rt[0] + 63) >> 6;
        u32 left = rank, total = 0, hit = k_none;
        for (u32 i = 0; i < nw; ++i)
        {
            const u64 w = static_cast<u64>(e[2 * i]) | (static_cast<u64>(e[2 * i + 1]) << 32);
            const auto cnt = static_cast<u32>(__popcll(w));
            if (hit == k_none)
            {
                if (left < cnt)
                    hit = i * 64 + nth_bit(w, left);
                else
                    left -= cnt;
            }
            total += cnt;
        }
        if (total != n || hit == k_none)
            return false;
        ranks.unrank(hit, chosen);
        return true;
    }
    e += 2 * rank;
    u64 key = static_cast<u64>(e[0]) | (static_cast<u64>(e[1]) << 32);
    const u32 bits = key_bits(t.num_objects);
    const u64 mask = (1ull << bits) - 1;
    for (u32 j = arity; j-- > 0;)
    {
        chosen[j] = static_cast<u32>(key & mask);
        key >>= bits;
    }
    return true;
}

/// Parent li's pick: the binding of rank pk.rank[li] in its (li, schema) segment, written at row li.
template<u32 OW, bool FC, u32 WB, bool R>
__device__ __forceinline__ void pick_one(const GenArgs& a, const TaskView& t, const Picks& pk, u32 k, u32 schema, u32 li)
{
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[a.set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const Parents& p = a.p;
    const u64* row = p.data + u64{li} * p.stride;
    const u64* der = p.derived ? p.derived + u64{li} * p.derived_words : nullptr;
    const u64* view = a.v.data + u64{li} * a.v.words;
    const StateRef s{row, p.words, der, p.derived_words};
    const u32 arity = sc[k_sc_arity];
    const u32 rank = pk.rank[li];
    u32 bind[k_max_depth];
    u32 chosen[k_max_label];
    bool bad = false;
    bool found = pick_cached<OW, R>(t, mx, pk, p, k, schema, arity, li, chosen, bad);
    if (bad)
    {
        if (pk.error)
            atomicOr(pk.error, 2u);
        return;
    }
    auto run = [&](auto& emit)
    {
        if constexpr (FC)
            run_fc<OW>(t, mx, s, view, bind, emit);
        else
            run_fixed<OW>(t, mx, s, view, bind, emit);
    };
    const bool sort = a.set.canonical && arity > 1 && device_sorts(mx[k_mc_flags], FC);
    if (!found && !sort)
    {
        // the matcher's order is the segment's order: the rank-th binding found ends the search
        u32 n = 0;
        auto emit = [&](const u32* b) -> bool
        {
            if (n++ != rank)
                return true;
            for (u32 i = 0; i < arity; ++i)
                chosen[i] = b[i];
            found = true;
            return false;
        };
        run(emit);
    }
    else if (!found)
    {
        // the rank-th binding in canonical order: one search into a bitmap of the bindings' canonical ranks where the
        // static domains allow (RankSpace), else passes over packed keys
        const RankSpace<OW> ranks(t, mx, arity);
        bool searched = false;
        if (ranks.size)
        {
            u64 bm[k_rank_words];
            const u32 nw = (ranks.size + 63) >> 6;
            for (u32 i = 0; i < nw; ++i)
                bm[i] = 0;
            bool outside = false;
            auto emit = [&](const u32* b)
            {
                const u32 r = ranks.rank(b);
                if (r == k_none)
                    outside = true;
                else
                    bm[r >> 6] |= 1ull << (r & 63);
            };
            run(emit);
            if (!outside)
            {
                searched = true;
                u32 left = rank;
                for (u32 i = 0; i < nw; ++i)
                {
                    const auto c = static_cast<u32>(__popcll(bm[i]));
                    if (left < c)
                    {
                        ranks.unrank(i * 64 + nth_bit(bm[i], left), chosen);
                        found = true;
                        break;
                    }
                    left -= c;
                }
            }
        }
        if (!searched && pick_keys_fit(arity, t.num_objects))
        {
            // the rank-th smallest packed key (lexicographic order = numeric order; the keys of a segment are
            // unique): each pass keeps the k_local_keys smallest keys above the previous pass's largest
            const u32 bits = key_bits(t.num_objects);
            u64 win[k_local_keys];
            u32 base = 0;
            u64 prev = 0;
            bool first = true;
            for (;;)
            {
                u32 m = 0;
                auto emit = [&](const u32* b)
                {
                    u64 key = 0;
                    for (u32 i = 0; i < arity; ++i)
                        key = (key << bits) | b[i];
                    if (!first && key <= prev)
                        return;
                    u32 j;
                    if (m < k_local_keys)
                        j = m++;
                    else if (key < win[k_local_keys - 1])
                        j = k_local_keys - 1;
                    else
                        return;
                    for (; j > 0 && win[j - 1] > key; --j)
                        win[j] = win[j - 1];
                    win[j] = key;
                };
                run(emit);
                if (m == 0)
                    break;
                if (rank < base + m)
                {
                    u64 key = win[rank - base];
                    const u64 mask = (1ull << bits) - 1;
                    for (u32 j = arity; j-- > 0;)
                    {
                        chosen[j] = static_cast<u32>(key & mask);
                        key >>= bits;
                    }
                    found = true;
                    break;
                }
                if (m < k_local_keys)
                    break;
                base += m;
                prev = win[m - 1];
                first = false;
            }
        }
    }
    if (!found)
    {
        if (pk.error)
            atomicOr(pk.error, 2u);
        return;
    }
    const RowWriter<WB> w(a, t, sc, schema, li, row);
    w.put(li, chosen);
}

/// One thread per parent. R: the cache has rank tables (as k_count_keys: their code's registers are the other launches'
/// occupancy).
template<u32 OW, bool FC, u32 WB, bool M, bool R>
__global__ void __launch_bounds__(k_gen_block) k_pick(GenArgs a, Picks pk)
{
    const u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    if (j >= a.p.rows)
        return;
    if constexpr (M)
    {
        u32 li, k;
        if (!env_row<OW>(a.m, j, li, k))
            return;
        const u32 schema = pk.schema[li];
        if (schema == k_none || !multi_kind<FC>(a.m, k, schema))
            return;
        pick_one<OW, FC, WB, R>(a, a.m.views[k], pk, k, schema, li);
    }
    else
    {
        const u32 schema = pk.schema[j];
        if (schema == k_none || pk.fc_of[schema] != a.set.fc)
            return;
        pick_one<OW, FC, WB, R>(a, a.t, pk, 0, schema, static_cast<u32>(j));
    }
}

template<u32 OW, bool FC, u32 WB, bool M>
cudaError_t launch_pick_ow(const GenArgs& a, const Picks& pk, cudaStream_t s)
{
    const u64 blocks = (u64{a.p.rows} + k_gen_block - 1) / k_gen_block;
    if (blocks == 0)
        return cudaSuccess;
    if (blocks > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    if (pk.cache.rank)
        k_pick<OW, FC, WB, M, true><<<static_cast<unsigned>(blocks), k_gen_block, 0, s>>>(a, pk);
    else
        k_pick<OW, FC, WB, M, false><<<static_cast<unsigned>(blocks), k_gen_block, 0, s>>>(a, pk);
    return cudaGetLastError();
}

template<u32 WB, bool M>
cudaError_t dispatch_pick(const GenArgs& a, const Picks& pk, cudaStream_t s)
{
    if (a.set.deep)
        return cudaErrorInvalidValue;  // deep matchers: count and write only
    const u32 ow = M ? a.m.ow : a.t.ow;
    if (a.set.fc)
    {
        switch (ow)
        {
            case 1: return launch_pick_ow<1, true, WB, M>(a, pk, s);
            case 2: return launch_pick_ow<2, true, WB, M>(a, pk, s);
            case 3: return launch_pick_ow<3, true, WB, M>(a, pk, s);
            case 4: return launch_pick_ow<4, true, WB, M>(a, pk, s);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (ow)
    {
        case 1: return launch_pick_ow<1, false, WB, M>(a, pk, s);
        case 2: return launch_pick_ow<2, false, WB, M>(a, pk, s);
        case 3: return launch_pick_ow<3, false, WB, M>(a, pk, s);
        case 4: return launch_pick_ow<4, false, WB, M>(a, pk, s);
        case 5: return launch_pick_ow<5, false, WB, M>(a, pk, s);
        case 6: return launch_pick_ow<6, false, WB, M>(a, pk, s);
        case 7: return launch_pick_ow<7, false, WB, M>(a, pk, s);
        case 8: return launch_pick_ow<8, false, WB, M>(a, pk, s);
        default: return cudaErrorInvalidValue;
    }
}

/// launch_pick's argument checks.
inline bool picks_valid(const Parents& p, const SchemaSet& set, const Picks& picks, const Labels& labels, bool multi)
{
    if (labels.capacity < p.rows || !picks.schema || !picks.rank || (!multi && !picks.fc_of))
        return false;
    const CountCache& c = picks.cache;
    return !(c.regions && (!c.data || !c.region || c.num_schemas != set.num_schemas || c.seg_cap > k_max_seg_cap ||
                           c.stride < cache_stride(c.num_schemas, c.regions, c.seg_cap)));
}
}  // namespace
}  // namespace mymyr::cuda::lifted
