// The kernels of the device state spaces (state_space_kernels.hpp). Compiled by nvcc as C++20; device-code subset.

#include "state_space_kernels.hpp"

#include "mymyr/cuda/state_set.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace mymyr::cuda::ssk
{
namespace
{
constexpr unsigned k_block = 256;
constexpr u64 k_tag = 0xFFFFFFFF00000000ull;
constexpr u32 k_pending = state_set::k_pending;
constexpr u64 k_inf_bits = 0x7FF0000000000000ull;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

#define MYMYR_GRID_LOOP(i, n) for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < (n); i += u64{gridDim.x} * blockDim.x)

/// The hash of a key (instance, row); the upper 32 bits are the slot tag.
__device__ __forceinline__ u64 key_hash(const u64* x, u32 nw, u32 inst)
{
    u64 h = state_set::row_hash(x, nw) ^ ((u64{inst} + 1) * 0x9E3779B97F4A7C15ull);
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 31;
    return h;
}

template<u32 W>
__device__ __forceinline__ bool equal(const u64* a, const u64* b, u32 n)
{
    if constexpr (W > 0)
    {
#pragma unroll
        for (u32 i = 0; i < W; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
    else
    {
        for (u32 i = 0; i < n; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }
}

__device__ __forceinline__ u64 load_slot(const u64* p) { return __ldcg(reinterpret_cast<const unsigned long long*>(p)); }

template<u32 W>
__global__ void k_insert(Table t, Store st, const u64* cand, const u32* parent, u64 n, u32* result)
{
    const u32 nw = W > 0 ? W : st.words;
    MYMYR_GRID_LOOP(c, n)
    {
        const u64* x = cand + c * nw;
        const u32 inst = st.inst[parent[c]];
        const u64 h = key_hash(x, nw, inst);
        const u64 tag = h & k_tag;
        const u64 mine = tag | k_pending | (c + 1);
        for (u64 j = h & t.mask;; j = (j + 1) & t.mask)
        {
            u64 s = load_slot(t.slots + j);
            if (s == 0)
            {
                const u64 old = atomicCAS(reinterpret_cast<unsigned long long*>(t.slots + j), 0ull, mine);
                if (old == 0)
                {
                    result[c] = static_cast<u32>(j);
                    break;
                }
                s = old;
            }
            if ((s & k_tag) != tag)
                continue;
            const u32 ref = static_cast<u32>(s);
            if (ref & k_pending)
            {
                // a pending slot only ever changes to a smaller candidate of the same key
                const u64 other = u64{(ref & ~k_pending) - 1};
                if (st.inst[parent[other]] == inst && equal<W>(x, cand + other * nw, nw))
                {
                    atomicMin(reinterpret_cast<unsigned long long*>(t.slots + j), mine);
                    result[c] = static_cast<u32>(j);
                    break;
                }
            }
            else if (st.inst[ref - 1] == inst && equal<W>(x, st.rows + u64{ref - 1} * nw, nw))
            {
                result[c] = static_cast<u32>(j);
                break;
            }
        }
    }
}

/// The owner flag of candidate c (0 past the batch: the scan's last entry is the total).
struct OwnerFlag
{
    Table t;
    const u32* result;
    u64 n;
    __device__ u32 operator()(u64 c) const
    {
        if (c >= n)
            return 0;
        return static_cast<u32>(load_slot(t.slots + result[c])) == (k_pending | static_cast<u32>(c + 1)) ? 1u : 0u;
    }
};
using OwnerIt = thrust::transform_iterator<OwnerFlag, thrust::counting_iterator<u64>>;

template<u32 W>
__global__ void k_compact(Table t, Compact m, u64 n)
{
    const u32 nw = W > 0 ? W : m.words;
    MYMYR_GRID_LOOP(c, n)
    {
        const u32 r = m.rank[c];
        if (m.rank[c + 1] == r)
            continue;
        const u64 id = m.base + r;
        const u64* x = m.cand + c * nw;
        u64* y = m.rows + id * nw;
        for (u32 i = 0; i < nw; ++i)
            y[i] = x[i];
        const u32 p = m.parent[c];
        const u32 inst = m.inst[p];
        m.inst[id] = inst;
        m.depth[id] = m.depth[p] + 1;
        u64* slot = t.slots + m.result[c];
        *slot = (*slot & k_tag) | (id + 1);
    }
}

/// Warp-aggregated atomics over keys (lanes of a warp mostly share a key: states come grouped by instance): the lanes
/// of one key reduce, the lowest one adds. Lanes with key k_no_instance contribute nothing.
__device__ __forceinline__ void add_by_key(u32* base, u32 key, u32 v)
{
    const unsigned act = __activemask();
    const unsigned grp = __match_any_sync(act, key);
    const u32 sum = __reduce_add_sync(grp, v);
    if (key != k_no_instance && sum && (threadIdx.x & 31u) == static_cast<unsigned>(__ffs(grp) - 1))
        atomicAdd(base + key, sum);
}
__device__ __forceinline__ void max_by_key(u32* base, u32 key, u32 v)
{
    const unsigned act = __activemask();
    const unsigned grp = __match_any_sync(act, key);
    const u32 m = __reduce_max_sync(grp, v);
    if (key != k_no_instance && (threadIdx.x & 31u) == static_cast<unsigned>(__ffs(grp) - 1))
        atomicMax(base + key, m);
}

__global__ void k_owner_counts(Table t, const u32* result, const u32* parent, const u32* inst, u64 n, u32* counts)
{
    MYMYR_GRID_LOOP(c, n)
    {
        const bool owner = static_cast<u32>(load_slot(t.slots + result[c])) == (k_pending | static_cast<u32>(c + 1));
        add_by_key(counts, owner ? inst[parent[c]] : k_no_instance, 1u);
    }
}

__global__ void k_pick_u64(const u64* src, const u64* P, u32 k, u64* out)
{
    MYMYR_GRID_LOOP(i, u64{k} + 1)
    out[i] = src[P[i]];
}

__global__ void k_pick_roots(const i32* dist, const u64* P, u32 k, i32* out)
{
    MYMYR_GRID_LOOP(i, k)
    out[i] = P[i] < P[i + 1] ? dist[P[i]] : -1;
}

__global__ void k_resolve(Table t, u32* ids, u64 n)
{
    MYMYR_GRID_LOOP(c, n)
    ids[c] = static_cast<u32>(load_slot(t.slots + ids[c])) - 1;
}

__global__ void k_rehash(Table t, Store st, u64 first, u64 count)
{
    MYMYR_GRID_LOOP(k, count - first)
    {
        const u64 i = first + k;
        const u64 h = key_hash(st.rows + i * st.words, st.words, st.inst[i]);
        const u64 v = (h & k_tag) | (i + 1);
        for (u64 j = h & t.mask;; j = (j + 1) & t.mask)
            if (atomicCAS(reinterpret_cast<unsigned long long*>(t.slots + j), 0ull, v) == 0)
                break;
    }
}

__global__ void k_forward_offsets(const u32* seg, u32 S, u64 rows, u64 base, u64* out)
{
    MYMYR_GRID_LOOP(i, rows + 1)
    out[i] = base + seg[i * S];
}

__global__ void k_chunk_ids(const u32* inst, const u32* live, u64 n, u32* ids)
{
    MYMYR_GRID_LOOP(j, n)
    ids[j] = live[inst[j]];
}

__global__ void k_costs(costs::Program pr, u32 inst_base, const u32* schema, const u32* binding, u32 L, const u32* parent,
                        const u32* inst, u64 n, f64* out, u32* error)
{
    MYMYR_GRID_LOOP(c, n)
    {
        const f64 v = costs::evaluate(pr, inst_base + inst[parent[c]], schema[c], binding + c * L);
        if (isnan(v))
            atomicOr(error, 1u);
        out[c] = v;
    }
}

/// The largest i < I with Q[i] <= e (the instance of edge e).
__device__ __forceinline__ u32 instance_of_edge(const u64* Q, u32 I, u64 e)
{
    u32 lo = 0, hi = I;
    while (hi - lo > 1)
    {
        const u32 mid = (lo + hi) / 2;
        if (Q[mid] <= e)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

struct WidenU32
{
    const u32* in;
    u64 n;
    __device__ u64 operator()(u64 i) const { return i < n ? u64{in[i]} : 0; }
};
struct WidenU64
{
    const u64* in;
    u64 n;
    __device__ u64 operator()(u64 i) const { return i < n ? in[i] : 0; }
};

__global__ void k_iota(u32* out, u64 n)
{
    MYMYR_GRID_LOOP(i, n)
    out[i] = static_cast<u32>(i);
}

__global__ void k_histogram(const u32* key, u64 n, u32* counts)
{
    MYMYR_GRID_LOOP(i, n)
    atomicAdd(counts + key[i], 1u);
}

__global__ void k_histogram_grouped(const u32* key, u64 n, u32* counts)
{
    MYMYR_GRID_LOOP(i, n)
    add_by_key(counts, key[i], 1u);
}

__global__ void k_invert(const u32* order, u64 n, u32* dst)
{
    MYMYR_GRID_LOOP(i, n)
    dst[order[i]] = static_cast<u32>(i);
}

__global__ void k_degrees(const u64* off, const u32* perm, u64 n, u64* deg)
{
    MYMYR_GRID_LOOP(p, n)
    {
        const u32 g = perm[p];
        deg[p] = off[g + 1] - off[g];
    }
}

__global__ void k_edge_map(const u64* off, const u64* noff, const u32* pos, u64 n, u32* emap)
{
    MYMYR_GRID_LOOP(g, n)
    {
        const u64 b = off[g], e = off[g + 1], nb = noff[pos[g]];
        for (u64 k = b; k < e; ++k)
            emap[k] = static_cast<u32>(nb + (k - b));
    }
}

__global__ void k_scatter_u32(const u32* src, const u32* emap, const u32* map, u64 n, u32 width, u32* dst)
{
    MYMYR_GRID_LOOP(e, n)
    {
        const u64 d = emap[e];
        for (u32 k = 0; k < width; ++k)
        {
            const u32 v = src[e * width + k];
            dst[d * width + k] = map ? map[v] : v;
        }
    }
}

__global__ void k_scatter_f64(const f64* src, const u32* emap, u64 n, f64* dst)
{
    MYMYR_GRID_LOOP(e, n)
    dst[emap[e]] = src[e];
}

__global__ void k_gather_u8(const u8* src, const u32* perm, u64 n, u8* dst)
{
    MYMYR_GRID_LOOP(i, n)
    dst[i] = src[perm[i]];
}

__global__ void k_edge_sources(const u64* off, u64 n, const u32* edges, u64 m, u32* bsrc)
{
    MYMYR_GRID_LOOP(r, m)
    {
        const u64 e = edges[r];
        // the largest g with off[g] <= e (off[n] > e): the state whose range holds e
        u64 lo = 0, hi = n;
        while (hi - lo > 1)
        {
            const u64 mid = (lo + hi) / 2;
            if (off[mid] <= e)
                lo = mid;
            else
                hi = mid;
        }
        bsrc[r] = static_cast<u32>(lo);
    }
}

__global__ void k_bfs_init(const u8* goal, u64 n, i32* dist, u32* frontier, u32* count)
{
    MYMYR_GRID_LOOP(i, n)
    {
        if (goal[i])
        {
            dist[i] = 0;
            frontier[atomicAdd(count, 1u)] = static_cast<u32>(i);
        }
        else
            dist[i] = -1;
    }
}

__global__ void k_bfs_step(const u64* boff, const u32* bsrc, const u32* frontier, u64 nf, i32 d, i32* dist, u32* next,
                           u32* next_count)
{
    MYMYR_GRID_LOOP(k, nf)
    {
        const u32 v = frontier[k];
        for (u64 r = boff[v]; r < boff[v + 1]; ++r)
        {
            const u32 u = bsrc[r];
            if (__ldcg(dist + u) == -1 && atomicCAS(dist + u, -1, d) == -1)
                next[atomicAdd(next_count, 1u)] = u;
        }
    }
}

__global__ void k_sssp_init(const u8* goal, u64 n, u64* dist, u32* frontier, u32* count)
{
    MYMYR_GRID_LOOP(i, n)
    {
        if (goal[i])
        {
            dist[i] = 0;  // the bits of +0.0
            frontier[atomicAdd(count, 1u)] = static_cast<u32>(i);
        }
        else
            dist[i] = k_inf_bits;
    }
}

__global__ void k_sssp_step(const u64* boff, const u32* bsrc, const u32* bedge, const f64* cost, const u32* frontier, u64 nf,
                            u64* dist, u32* queued, u32* next, u32* next_count)
{
    MYMYR_GRID_LOOP(k, nf)
    {
        const u32 v = frontier[k];
        const f64 dv = __longlong_as_double(static_cast<long long>(load_slot(dist + v)));
        for (u64 r = boff[v]; r < boff[v + 1]; ++r)
        {
            const u32 u = bsrc[r];
            const f64 nd = __dadd_rn(dv, cost[bedge[r]]);
            const auto nb = static_cast<unsigned long long>(__double_as_longlong(nd));
            if (nb >= load_slot(dist + u))
                continue;
            const unsigned long long old = atomicMin(reinterpret_cast<unsigned long long*>(dist + u), nb);
            if (nb < old && atomicExch(queued + u, 1u) == 0)
                next[atomicAdd(next_count, 1u)] = u;
        }
    }
}

__global__ void k_sssp_clear(const u32* frontier, u64 nf, u32* queued)
{
    MYMYR_GRID_LOOP(k, nf)
    queued[frontier[k]] = 0;
}

__global__ void k_check_costs(const f64* cost, const u64* Q, u32 I, u64 n, u32* flags)
{
    MYMYR_GRID_LOOP(e, n)
    if (!(cost[e] >= 0))
        flags[instance_of_edge(Q, I, e)] = 1;
}

__global__ void k_unit_costs(const f64* cost, const u64* Q, u32 I, u64 n, u32* flags)
{
    MYMYR_GRID_LOOP(e, n)
    if (cost[e] != 1.0)
        flags[instance_of_edge(Q, I, e)] = 1;
}

__global__ void k_flags(const u8* goal, const i32* dist, const u32* inst, u64 n, bool unit_cost, u8* unsolvable, u8* alive,
                        f64* cost, u32* stats)
{
    MYMYR_GRID_LOOP(i, n)
    {
        const i32 d = dist[i];
        const bool g = goal[i] != 0, u = d < 0;
        unsolvable[i] = u ? 1 : 0;
        alive[i] = !g && !u ? 1 : 0;
        if (unit_cost)
            cost[i] = u ? __longlong_as_double(static_cast<long long>(k_inf_bits)) : static_cast<f64>(d);
        const u32 key = inst ? inst[i] : 0;
        add_by_key(stats, key == k_no_instance ? key : key * 3, g ? 1u : 0u);
        add_by_key(stats + 1, key == k_no_instance ? key : key * 3, u ? 1u : 0u);
        max_by_key(stats + 2, key == k_no_instance ? key : key * 3, u ? 0u : static_cast<u32>(d) + 1);
    }
}

__global__ void k_max_depth(const u32* depth, const u32* inst, u64 n, u32* maxdepth)
{
    MYMYR_GRID_LOOP(i, n)
    max_by_key(maxdepth, inst ? inst[i] : 0, depth[i]);
}

__global__ void k_local_rows(const u64* rows, u32 W, const u32* perm, Layout l, u64 n, u64* out)
{
    MYMYR_GRID_LOOP(p, n)
    {
        const u32 i = l.inst ? l.inst[p] : 0;
        const u32 w = l.words[i];
        const u64 src = perm ? perm[p] : p;
        u64* y = out + l.row_base[i] + (p - l.P[i]) * w;
        for (u32 k = 0; k < w; ++k)
            y[k] = rows[src * W + k];
    }
}

__global__ void k_local_offsets(const u64* off, Layout l, u64 n, u64* out)
{
    MYMYR_GRID_LOOP(p, n)
    {
        const u32 i = l.inst ? l.inst[p] : 0;
        out[p + i] = off[p] - l.Q[i];
    }
}

__global__ void k_local_ends(const u64* off, Layout l, u32 I, u64* out)
{
    MYMYR_GRID_LOOP(i, I)
    out[l.P[i + 1] + i] = off[l.P[i + 1]] - l.Q[i];
}

__global__ void k_local_states(u32* ids, Layout l, u64 m)
{
    MYMYR_GRID_LOOP(r, m)
    {
        const u32 v = ids[r];
        ids[r] = static_cast<u32>(v - l.P[l.inst[v]]);
    }
}

__global__ void k_local_edges(u32* edges, const u32* state, Layout l, u64 m)
{
    MYMYR_GRID_LOOP(r, m)
    edges[r] = static_cast<u32>(edges[r] - l.Q[l.inst[state[r]]]);
}
}  // namespace

cudaError_t launch_insert(Table t, Store st, const u64* cand, const u32* parent, u64 n, u32* result, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    const unsigned g = grid_for(n);
    switch (st.words)
    {
        case 1: k_insert<1><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
        case 2: k_insert<2><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
        case 3: k_insert<3><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
        case 4: k_insert<4><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
        case 8: k_insert<8><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
        default: k_insert<0><<<g, k_block, 0, s>>>(t, st, cand, parent, n, result); break;
    }
    return cudaGetLastError();
}

u64 rank_temp_bytes(u64 n)
{
    size_t bytes = 0;
    const OwnerIt it(thrust::counting_iterator<u64>(0), OwnerFlag{{}, nullptr, 0});
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, it, static_cast<u32*>(nullptr), n + 1);
    return bytes;
}

cudaError_t launch_rank(Table t, const u32* result, u64 n, u32* rank, void* temp, u64 temp_bytes, cudaStream_t s)
{
    size_t bytes = temp_bytes;
    const OwnerIt it(thrust::counting_iterator<u64>(0), OwnerFlag{t, result, n});
    return cub::DeviceScan::ExclusiveSum(temp, bytes, it, rank, n + 1, s);
}

cudaError_t launch_compact(Table t, Compact c, u64 n, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    const unsigned g = grid_for(n);
    switch (c.words)
    {
        case 1: k_compact<1><<<g, k_block, 0, s>>>(t, c, n); break;
        case 2: k_compact<2><<<g, k_block, 0, s>>>(t, c, n); break;
        case 4: k_compact<4><<<g, k_block, 0, s>>>(t, c, n); break;
        default: k_compact<0><<<g, k_block, 0, s>>>(t, c, n); break;
    }
    return cudaGetLastError();
}

cudaError_t launch_owner_counts(Table t, const u32* result, const u32* parent, const u32* inst, u64 n, u32* counts,
                                cudaStream_t s)
{
    if (n)
        k_owner_counts<<<grid_for(n), k_block, 0, s>>>(t, result, parent, inst, n, counts);
    return cudaGetLastError();
}

cudaError_t launch_pick_u64(const u64* src, const u64* P, u32 k, u64* out, cudaStream_t s)
{
    k_pick_u64<<<grid_for(u64{k} + 1), k_block, 0, s>>>(src, P, k, out);
    return cudaGetLastError();
}

cudaError_t launch_pick_roots(const i32* dist, const u64* P, u32 k, i32* out, cudaStream_t s)
{
    if (k)
        k_pick_roots<<<grid_for(k), k_block, 0, s>>>(dist, P, k, out);
    return cudaGetLastError();
}

cudaError_t launch_resolve(Table t, u32* ids, u64 n, cudaStream_t s)
{
    if (n)
        k_resolve<<<grid_for(n), k_block, 0, s>>>(t, ids, n);
    return cudaGetLastError();
}

cudaError_t launch_rehash(Table t, Store st, u64 first, u64 count, cudaStream_t s)
{
    if (count > first)
        k_rehash<<<grid_for(count - first), k_block, 0, s>>>(t, st, first, count);
    return cudaGetLastError();
}

cudaError_t launch_forward_offsets(const u32* seg, u32 num_schemas, u64 rows, u64 base, u64* out, cudaStream_t s)
{
    k_forward_offsets<<<grid_for(rows + 1), k_block, 0, s>>>(seg, num_schemas, rows, base, out);
    return cudaGetLastError();
}

cudaError_t launch_chunk_ids(const u32* inst, const u32* live, u64 n, u32* ids, cudaStream_t s)
{
    if (n)
        k_chunk_ids<<<grid_for(n), k_block, 0, s>>>(inst, live, n, ids);
    return cudaGetLastError();
}

cudaError_t launch_costs(const costs::Program& programs, u32 inst_base, const u32* schema, const u32* binding, u32 label_width,
                         const u32* parent, const u32* inst, u64 n, f64* out, u32* error, cudaStream_t s)
{
    if (n)
        k_costs<<<grid_for(n), k_block, 0, s>>>(programs, inst_base, schema, binding, label_width, parent, inst, n, out, error);
    return cudaGetLastError();
}

u64 scan_temp_bytes(u64 n)
{
    size_t bytes = 0;
    const thrust::transform_iterator<WidenU64, thrust::counting_iterator<u64>> it(thrust::counting_iterator<u64>(0),
                                                                                  WidenU64{nullptr, 0});
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, it, static_cast<u64*>(nullptr), n + 1);
    return bytes;
}

cudaError_t launch_scan_u32(const u32* in, u64* out, u64 n, void* temp, u64 temp_bytes, cudaStream_t s)
{
    size_t bytes = temp_bytes;
    const thrust::transform_iterator<WidenU32, thrust::counting_iterator<u64>> it(thrust::counting_iterator<u64>(0),
                                                                                  WidenU32{in, n});
    return cub::DeviceScan::ExclusiveSum(temp, bytes, it, out, n + 1, s);
}

cudaError_t launch_scan_u64(const u64* in, u64* out, u64 n, void* temp, u64 temp_bytes, cudaStream_t s)
{
    size_t bytes = temp_bytes;
    const thrust::transform_iterator<WidenU64, thrust::counting_iterator<u64>> it(thrust::counting_iterator<u64>(0),
                                                                                  WidenU64{in, n});
    return cub::DeviceScan::ExclusiveSum(temp, bytes, it, out, n + 1, s);
}

u64 sort_temp_bytes(u64 n)
{
    size_t bytes = 0;
    cub::DoubleBuffer<u32> k(nullptr, nullptr), v(nullptr, nullptr);
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, v, n, 0, 32);
    return bytes;
}

cudaError_t launch_sort_pairs(u32* keys0, u32* keys1, u32* values0, u32* values1, u64 n, u32 bits, void* temp,
                              u64 temp_bytes, int* selector, cudaStream_t s)
{
    size_t bytes = temp_bytes;
    cub::DoubleBuffer<u32> k(keys0, keys1), v(values0, values1);
    const cudaError_t e = cub::DeviceRadixSort::SortPairs(temp, bytes, k, v, n, 0, static_cast<int>(bits ? bits : 1), s);
    *selector = k.selector;
    return e;
}

cudaError_t launch_iota(u32* out, u64 n, cudaStream_t s)
{
    if (n)
        k_iota<<<grid_for(n), k_block, 0, s>>>(out, n);
    return cudaGetLastError();
}

cudaError_t launch_histogram(const u32* key, u64 n, u32* counts, cudaStream_t s)
{
    if (n)
        k_histogram<<<grid_for(n), k_block, 0, s>>>(key, n, counts);
    return cudaGetLastError();
}

cudaError_t launch_histogram_grouped(const u32* key, u64 n, u32* counts, cudaStream_t s)
{
    if (n)
        k_histogram_grouped<<<grid_for(n), k_block, 0, s>>>(key, n, counts);
    return cudaGetLastError();
}

cudaError_t launch_invert(const u32* order, u64 n, u32* dst, cudaStream_t s)
{
    if (n)
        k_invert<<<grid_for(n), k_block, 0, s>>>(order, n, dst);
    return cudaGetLastError();
}

cudaError_t launch_degrees(const u64* off, const u32* perm, u64 n, u64* deg, cudaStream_t s)
{
    if (n)
        k_degrees<<<grid_for(n), k_block, 0, s>>>(off, perm, n, deg);
    return cudaGetLastError();
}

cudaError_t launch_edge_map(const u64* off, const u64* noff, const u32* pos, u64 n, u32* emap, cudaStream_t s)
{
    if (n)
        k_edge_map<<<grid_for(n), k_block, 0, s>>>(off, noff, pos, n, emap);
    return cudaGetLastError();
}

cudaError_t launch_scatter_u32(const u32* src, const u32* emap, const u32* map, u64 n, u32 width, u32* dst, cudaStream_t s)
{
    if (n && width)
        k_scatter_u32<<<grid_for(n), k_block, 0, s>>>(src, emap, map, n, width, dst);
    return cudaGetLastError();
}

cudaError_t launch_scatter_f64(const f64* src, const u32* emap, u64 n, f64* dst, cudaStream_t s)
{
    if (n)
        k_scatter_f64<<<grid_for(n), k_block, 0, s>>>(src, emap, n, dst);
    return cudaGetLastError();
}

cudaError_t launch_gather_u8(const u8* src, const u32* perm, u64 n, u8* dst, cudaStream_t s)
{
    if (n)
        k_gather_u8<<<grid_for(n), k_block, 0, s>>>(src, perm, n, dst);
    return cudaGetLastError();
}

cudaError_t launch_edge_sources(const u64* off, u64 n, const u32* edges, u64 m, u32* bsrc, cudaStream_t s)
{
    if (m)
        k_edge_sources<<<grid_for(m), k_block, 0, s>>>(off, n, edges, m, bsrc);
    return cudaGetLastError();
}

cudaError_t launch_bfs_init(const u8* goal, u64 n, i32* dist, u32* frontier, u32* count, cudaStream_t s)
{
    if (n)
        k_bfs_init<<<grid_for(n), k_block, 0, s>>>(goal, n, dist, frontier, count);
    return cudaGetLastError();
}

cudaError_t launch_bfs_step(const u64* boff, const u32* bsrc, const u32* frontier, u64 nf, i32 d, i32* dist, u32* next,
                            u32* next_count, cudaStream_t s)
{
    if (nf)
        k_bfs_step<<<grid_for(nf), k_block, 0, s>>>(boff, bsrc, frontier, nf, d, dist, next, next_count);
    return cudaGetLastError();
}

cudaError_t launch_sssp_init(const u8* goal, u64 n, u64* dist, u32* frontier, u32* count, cudaStream_t s)
{
    if (n)
        k_sssp_init<<<grid_for(n), k_block, 0, s>>>(goal, n, dist, frontier, count);
    return cudaGetLastError();
}

cudaError_t launch_sssp_step(const u64* boff, const u32* bsrc, const u32* bedge, const f64* cost, const u32* frontier,
                             u64 nf, u64* dist, u32* queued, u32* next, u32* next_count, cudaStream_t s)
{
    if (nf)
        k_sssp_step<<<grid_for(nf), k_block, 0, s>>>(boff, bsrc, bedge, cost, frontier, nf, dist, queued, next, next_count);
    return cudaGetLastError();
}

cudaError_t launch_sssp_clear(const u32* frontier, u64 nf, u32* queued, cudaStream_t s)
{
    if (nf)
        k_sssp_clear<<<grid_for(nf), k_block, 0, s>>>(frontier, nf, queued);
    return cudaGetLastError();
}

cudaError_t launch_check_costs(const f64* cost, const u64* Q, u32 instances, u64 n, u32* flags, cudaStream_t s)
{
    if (n)
        k_check_costs<<<grid_for(n), k_block, 0, s>>>(cost, Q, instances, n, flags);
    return cudaGetLastError();
}

cudaError_t launch_unit_costs(const f64* cost, const u64* Q, u32 instances, u64 n, u32* flags, cudaStream_t s)
{
    if (n)
        k_unit_costs<<<grid_for(n), k_block, 0, s>>>(cost, Q, instances, n, flags);
    return cudaGetLastError();
}

cudaError_t launch_flags(const u8* goal, const i32* dist, const u32* inst, u64 n, bool unit_cost, u8* unsolvable,
                         u8* alive, f64* cost, u32* stats, cudaStream_t s)
{
    if (n)
        k_flags<<<grid_for(n), k_block, 0, s>>>(goal, dist, inst, n, unit_cost, unsolvable, alive, cost, stats);
    return cudaGetLastError();
}

cudaError_t launch_max_depth(const u32* depth, const u32* inst, u64 n, u32* maxdepth, cudaStream_t s)
{
    if (n)
        k_max_depth<<<grid_for(n), k_block, 0, s>>>(depth, inst, n, maxdepth);
    return cudaGetLastError();
}

cudaError_t launch_local_rows(const u64* rows, u32 W, const u32* perm, Layout l, u64 n, u64* out, cudaStream_t s)
{
    if (n)
        k_local_rows<<<grid_for(n), k_block, 0, s>>>(rows, W, perm, l, n, out);
    return cudaGetLastError();
}

cudaError_t launch_local_offsets(const u64* off, Layout l, u64 n, u64* out, cudaStream_t s)
{
    if (n)
        k_local_offsets<<<grid_for(n), k_block, 0, s>>>(off, l, n, out);
    return cudaGetLastError();
}

cudaError_t launch_local_ends(const u64* off, Layout l, u32 instances, u64* out, cudaStream_t s)
{
    if (instances)
        k_local_ends<<<grid_for(instances), k_block, 0, s>>>(off, l, instances, out);
    return cudaGetLastError();
}

cudaError_t launch_local_states(u32* ids, Layout l, u64 m, cudaStream_t s)
{
    if (m)
        k_local_states<<<grid_for(m), k_block, 0, s>>>(ids, l, m);
    return cudaGetLastError();
}

cudaError_t launch_local_edges(u32* edges, const u32* state, Layout l, u64 m, cudaStream_t s)
{
    if (m)
        k_local_edges<<<grid_for(m), k_block, 0, s>>>(edges, state, l, m);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::ssk
