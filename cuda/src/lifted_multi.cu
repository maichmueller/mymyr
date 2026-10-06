// Multi-instance launches of the lifted successor kernels (include/mymyr/cuda/lifted.hpp, "several instances"):
// the views, counts, count caches and goal tests over rows of several instances of a task table, and the row order that
// groups them by instance. The kernels are those of lifted.cu with the instance resolved per thread (lifted_env.cuh,
// lifted_device.cuh: template flag M); the write and pick kernels are instantiated in lifted_multi_write.cu and
// lifted_multi_pick.cu (parallel builds).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_env.cuh"

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>

namespace mymyr::cuda::lifted
{
namespace
{
__global__ void k_goal_rows_multi(const TaskView* views, u32 instances, const u32* inst, const u64* states, u32 words, u64 n,
                                  u8* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        out[i] = inst[i] < instances && goal_holds(views[inst[i]], states + i * words, words, nullptr, 0) ? 1 : 0;
}

__global__ void k_check_multi(const u32* limits, u32 instances, const i32* inst, const u64* data, u64 stride, u32 words,
                              u64 rows, u32* bad_id, u32* bad_row)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const i32 k = inst[i];
        if (k < 0 || static_cast<u32>(k) >= instances)
        {
            atomicMin(bad_id, static_cast<u32>(i));
            continue;
        }
        if (!data)
            continue;
        // launch_check_rows with the row's instance's limit
        const u32 limit = limits[k], lw = limit >> 6;
        const u64 first = limit & 63 ? ~((1ull << (limit & 63)) - 1) : ~0ull;
        const u64* r = data + i * stride;
        bool b = false;
        for (u32 w = lw; w < words && !b; ++w)
            b = (r[w] & (w == lw ? first : ~0ull)) != 0;
        if (b)
            atomicMin(bad_row, static_cast<u32>(i));
    }
}

__global__ void k_check_ids(const i32* inst, u32 instances, u64 rows, u32* bad_id)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
        if (inst[i] < 0 || static_cast<u32>(inst[i]) >= instances)
            atomicMin(bad_id, static_cast<u32>(i));
}

__global__ void k_part_counts(const u32* rows, const i32* offsets, u64 n, u32* counts)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
        counts[rows[j]] = static_cast<u32>(offsets[j + 1] - offsets[j]);
}

/// 2^shift lanes per part successor j (launch_part_scatter): every lane of the group finds j's batch position (the
/// group's loads hit the same words), the lanes copy its words and labels, the group's first lane writes its schema,
/// parent and goal flag. The grid's thread count is a multiple of the group size, so a thread keeps its lane.
__global__ void k_part_scatter(PartScatter p, u32 shift)
{
    const u32 lanes = 1u << shift, lane = threadIdx.x & (lanes - 1);
    const u64 threads = u64{gridDim.x} * blockDim.x;
    for (u64 t = u64{blockIdx.x} * blockDim.x + threadIdx.x; (t >> shift) < p.total; t += threads)
    {
        const u64 j = t >> shift;
        const u32 q = static_cast<u32>(p.parent[j]);
        const u32 r = p.outer ? p.outer[p.rows[q]] : p.rows[q];
        const u64 at = static_cast<u64>(p.batch_offsets[r]) + (j - static_cast<u64>(p.offsets[q]));
        if (at >= p.capacity)
            continue;
        if (p.out_succ)
        {
            const u64* src = p.succ + j * p.words;
            u64* dst = p.out_succ + at * p.out_words;
            for (u32 w = lane; w < p.out_words; w += lanes)
                dst[w] = w < p.words ? src[w] : 0;
        }
        if (p.out_binding)
        {
            const i32* src = p.binding + j * p.label_width;
            i32* dst = p.out_binding + at * p.out_label_width;
            for (u32 k = lane; k < p.out_label_width; k += lanes)
                dst[k] = k < p.label_width ? src[k] : -1;
        }
        if (lane == 0)
        {
            if (p.out_schema)
                p.out_schema[at] = p.schema[j];
            if (p.out_parent)
                p.out_parent[at] = static_cast<i32>(r);
            if (p.out_goal)
                p.out_goal[at] = p.goal[j];
        }
    }
}

__global__ void k_order_scatter(const u32* sorted, u64 n, u32* order, u32* pos, u64 stride)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
    {
        const u32 r = sorted[j];
        order[j * stride] = r;
        pos[u64{r} * stride] = static_cast<u32>(j);
    }
}

__global__ void k_check_order(const u32* order, const u32* pos, u64 stride, u64 n, u32* bad)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
    {
        const u32 r = order[j * stride];
        if (r >= n || pos[u64{r} * stride] != j)
            *bad = 1;
    }
}

/// CUB's radix sort scratch for n pairs, 64-byte aligned, after the four u32 arrays of launch_order.
u64 sort_bytes(u64 n)
{
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                    static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                    static_cast<int>(n > 0x7FFFFFFFull ? 0x7FFFFFFF : n), 0, 32);
    return bytes;
}

u64 align64(u64 b) { return (b + 63) & ~u64{63}; }

// launch_order's counting sort: two launches instead of the keys, CUB's radix sort (three) and the scatter, for
// up to k_order_max_keys keys and k_order_cells (key, tile) counts; more keys and larger batches sort by CUB. Both read
// a row's key clamped to the key count (order_key).
constexpr u32 k_order_tile = 512;  // rows per block
constexpr u32 k_order_block = 256;
constexpr u32 k_order_warps = k_order_block / 32;
constexpr u32 k_order_max_keys = 1024;
constexpr u64 k_order_cells = u64{1} << 16;

u64 order_tiles(u64 n) { return (n + k_order_tile - 1) / k_order_tile; }
u64 order_cells(u64 n) { return std::min<u64>(k_order_cells, u64{k_order_max_keys} * order_tiles(n)); }

/// Row i's key: its instance's, at most keys - 1; 0 for an id outside [0, instances).
__device__ __forceinline__ u32 order_key(const u32* inst, const u32* key, u32 instances, u32 keys, u64 i)
{
    const u32 k = inst[i];
    if (k >= instances)
        return 0;
    const u32 x = key[k];
    return x < keys ? x : keys - 1;
}

__global__ void k_order_keys(const u32* inst, OrderKeys ok, u32 instances, u64 n, u32* keys, u32* rows)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
    {
        keys[i] = order_key(inst, ok.key, instances, ok.count, i);
        rows[i] = static_cast<u32>(i);
    }
}

__global__ void k_check_order_keys(const u32* inst, u32 instances, OrderKeys ok, u64 n, const u32* order, const u32* pos,
                                   u64 stride, u32* bad)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
    {
        const u32 r = order[j * stride];
        if (r >= n || pos[u64{r} * stride] != j)
        {
            *bad = 1;
            continue;
        }
        if (j == 0)
            continue;
        const u32 q = order[(j - 1) * stride];  // checked by its own thread
        if (q >= n)
            continue;
        const u32 kq = order_key(inst, ok.key, instances, ok.count, q), kr = order_key(inst, ok.key, instances, ok.count, r);
        if (kq > kr || (kq == kr && q >= r))
            *bad = 1;
    }
}

/// hist[key * tiles + b]: the rows of tile b (k_order_tile rows) with that key, of `keys` keys.
__global__ void __launch_bounds__(k_order_block) k_order_hist(const u32* inst, const u32* key, u32 instances, u64 n, u32 keys,
                                                              u32 tiles, u32* hist)
{
    extern __shared__ u32 order_h[];  // [keys]
    for (u32 k = threadIdx.x; k < keys; k += k_order_block)
        order_h[k] = 0;
    __syncthreads();
    const u64 lo = u64{blockIdx.x} * k_order_tile, hi = lo + k_order_tile < n ? lo + k_order_tile : n;
    for (u64 i = lo + threadIdx.x; i < hi; i += k_order_block)
        atomicAdd(order_h + order_key(inst, key, instances, keys, i), 1u);
    __syncthreads();
    for (u32 k = threadIdx.x; k < keys; k += k_order_block)
        hist[u64{k} * tiles + blockIdx.x] = order_h[k];
}

/// Tile b's rows to their positions: the rows of smaller keys, then of the same key in earlier tiles, then in this
/// tile in row order (rounds of k_order_block rows; within a round, warps in order and lanes in order).
__global__ void __launch_bounds__(k_order_block) k_order_place(const u32* inst, const u32* key, u32 instances, u64 n,
                                                               u32 keys, u32 tiles, const u32* hist, u32* order, u32* pos,
                                                               u64 stride)
{
    using Scan = cub::BlockScan<u32, k_order_block>;
    constexpr u32 per = k_order_max_keys / k_order_block;  // keys per thread in the scan
    __shared__ typename Scan::TempStorage scan;
    extern __shared__ u32 order_sm[];
    u32* next = order_sm;       // [keys]: the next position of a key's rows in this tile
    u32* wc = order_sm + keys;  // [warps, keys]: a round's rows per warp and key (zero between rounds)
    u32 total[per], before[per];
    for (u32 q = 0; q < per; ++q)
    {
        const u32 k = threadIdx.x * per + q;
        u32 t = 0, b = 0;
        if (k < keys)
            for (u32 x = 0; x < tiles; ++x)
            {
                const u32 c = hist[u64{k} * tiles + x];
                t += c;
                b += x < blockIdx.x ? c : 0;
            }
        total[q] = t;
        before[q] = b;
    }
    Scan(scan).ExclusiveSum(total, total);
    for (u32 q = 0; q < per; ++q)
        if (const u32 k = threadIdx.x * per + q; k < keys)
            next[k] = total[q] + before[q];
    for (u32 j = threadIdx.x; j < k_order_warps * keys; j += k_order_block)
        wc[j] = 0;
    __syncthreads();
    const u32 warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const u64 lo = u64{blockIdx.x} * k_order_tile;
    for (u32 r = 0; r < k_order_tile && lo + r < n; r += k_order_block)
    {
        const u64 i = lo + r + threadIdx.x;
        const bool live = i < n;
        const u32 k = live ? order_key(inst, key, instances, keys, i) : keys;
        const u32 peers = __match_any_sync(0xFFFFFFFFu, k);
        const u32 below = static_cast<u32>(__popc(peers & ((1u << lane) - 1)));
        const bool lead = live && below == 0;
        if (lead)
            wc[warp * keys + k] = static_cast<u32>(__popc(peers));
        __syncthreads();
        if (live)
        {
            u32 at = next[k] + below;
            for (u32 w = 0; w < warp; ++w)
                at += wc[w * keys + k];
            order[u64{at} * stride] = static_cast<u32>(i);
            pos[i * stride] = at;
        }
        __syncthreads();
        if (lead)
        {
            atomicAdd(next + k, static_cast<u32>(__popc(peers)));
            wc[warp * keys + k] = 0;
        }
        __syncthreads();
    }
}
}  // namespace

cudaError_t launch_view_multi(const Multi& m, Parents p, Views v, cudaStream_t s)
{
    if (!m.views || !m.inst || !m.instances || !range_valid(m))
        return cudaErrorInvalidValue;
    return launch_view_any<true>(TaskView{}, m, p, v, s);
}

cudaError_t launch_count_multi(const Multi& m, Parents p, Views v, SchemaSet set, u32* counts, cudaStream_t s)
{
    if (!m.views || !m.inst || !m.instances || !m.fc_of || m.num_schemas != set.num_schemas || m.starts)
        return cudaErrorInvalidValue;
    GenArgs a{TaskView{}, p, v, set, nullptr, counts, Labels{}, SuccessorRows{}};
    a.m = m;
    return dispatch_ow<false, 1, false, true>(a, s);
}

cudaError_t launch_count_keys_multi(const Multi& m, Parents p, Views v, SchemaSet set, CountCache c, bool first,
                                    cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!m.views || !m.inst || !m.instances || !m.fc_of || m.num_schemas != set.num_schemas || !range_valid(m) ||
        (c.regions && c.region_stride < c.num_schemas))
        return cudaErrorInvalidValue;
    GenArgs a{TaskView{}, p, v, set, nullptr, nullptr, Labels{}, SuccessorRows{}};
    a.m = m;
    return dispatch_count_keys<true>(a, c, first, s);
}

cudaError_t launch_goal_rows_multi(const rl::dev::TaskView* views, u32 instances, const u32* inst, const u64* states,
                                   u32 words, u64 n, u8* out, cudaStream_t s)
{
    if (n)
        k_goal_rows_multi<<<grid_for(n), k_block, 0, s>>>(views, instances, inst, states, words, n, out);
    return cudaGetLastError();
}

cudaError_t launch_check_multi(const u32* limits, u32 instances, const i32* inst, const u64* data, u64 stride, u32 words,
                               u64 rows, u32* bad_id, u32* bad_row, cudaStream_t s)
{
    if (rows && (!inst || (data && !limits)))
        return cudaErrorInvalidValue;
    if (rows)
        k_check_multi<<<grid_for(rows), k_block, 0, s>>>(limits, instances, inst, data, stride, words, rows, bad_id, bad_row);
    return cudaGetLastError();
}

cudaError_t launch_check_ids(const i32* inst, u32 instances, u64 rows, u32* bad_id, cudaStream_t s)
{
    if (rows)
        k_check_ids<<<grid_for(rows), k_block, 0, s>>>(inst, instances, rows, bad_id);
    return cudaGetLastError();
}

cudaError_t launch_part_counts(const u32* rows, const i32* offsets, u64 n, u32* counts, cudaStream_t s)
{
    if (n)
        k_part_counts<<<grid_for(n), k_block, 0, s>>>(rows, offsets, n, counts);
    return cudaGetLastError();
}

cudaError_t launch_part_scatter(const PartScatter& p, cudaStream_t s)
{
    if (p.total == 0)
        return cudaSuccess;
    if (!p.rows || !p.offsets || !p.batch_offsets || !p.parent || (p.out_succ && (!p.succ || p.words > p.out_words)) ||
        (p.out_schema && !p.schema) || (p.out_binding && (!p.binding || p.label_width > p.out_label_width)) ||
        (p.out_goal && !p.goal))
        return cudaErrorInvalidValue;
    if (!p.out_succ && !p.out_schema && !p.out_binding && !p.out_parent && !p.out_goal)
        return cudaSuccess;
    const u32 rw = p.out_succ ? p.out_words : 1u, lw = p.out_binding ? p.out_label_width : 1u;
    const u32 width = rw > lw ? rw : lw;
    u32 shift = 0;
    while (shift < 5 && (1u << shift) < width)
        ++shift;
    k_part_scatter<<<grid_for(p.total << shift), k_block, 0, s>>>(p, shift);
    return cudaGetLastError();
}

u64 order_temp_bytes(u64 n)
{
    return std::max(4 * align64((n ? n : 1) * sizeof(u32)) + align64(sort_bytes(n)), align64(order_cells(n) * sizeof(u32)));
}

cudaError_t launch_order(const u32* inst, u32 instances, OrderKeys keys, u64 n, void* temp, u64 temp_bytes, u32* order,
                         u32* pos, u64 stride, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    if (n > 0x7FFFFFFFull || temp_bytes < order_temp_bytes(n) || keys.count == 0 || (instances && !keys.key))
        return cudaErrorInvalidValue;
    auto* base = static_cast<unsigned char*>(temp);
    const u64 tiles = order_tiles(n), nk = keys.count;
    if (nk <= k_order_max_keys && nk * tiles <= order_cells(n))
    {
        // the counting sort: tile counts per key, then each tile places its rows
        auto* hist = reinterpret_cast<u32*>(base);
        const auto k = static_cast<u32>(nk), b = static_cast<u32>(tiles);
        k_order_hist<<<b, k_order_block, k * sizeof(u32), s>>>(inst, keys.key, instances, n, k, b, hist);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            return e;
        k_order_place<<<b, k_order_block, (1 + k_order_warps) * k * sizeof(u32), s>>>(inst, keys.key, instances, n, k, b,
                                                                                        hist, order, pos, stride);
        return cudaGetLastError();
    }
    const u64 a = align64(n * sizeof(u32));
    auto* kv = reinterpret_cast<u32*>(base);
    auto* kv_out = reinterpret_cast<u32*>(base + a);
    auto* rows = reinterpret_cast<u32*>(base + 2 * a);
    auto* rows_out = reinterpret_cast<u32*>(base + 3 * a);
    k_order_keys<<<grid_for(n), k_block, 0, s>>>(inst, keys, instances, n, kv, rows);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return e;
    // radix sort over the keys' bits is stable: rows of one key keep their order
    int bits = 1;
    while (bits < 32 && (u64{1} << bits) < nk)
        ++bits;
    size_t bytes = temp_bytes - 4 * a;
    if (const cudaError_t e = cub::DeviceRadixSort::SortPairs(base + 4 * a, bytes, kv, kv_out, rows, rows_out,
                                                              static_cast<int>(n), 0, bits, s);
        e != cudaSuccess)
        return e;
    k_order_scatter<<<grid_for(n), k_block, 0, s>>>(rows_out, n, order, pos, stride);
    return cudaGetLastError();
}

cudaError_t launch_check_order(const u32* order, const u32* pos, u64 stride, u64 n, u32* bad, cudaStream_t s)
{
    if (n)
        k_check_order<<<grid_for(n), k_block, 0, s>>>(order, pos, stride, n, bad);
    return cudaGetLastError();
}

cudaError_t launch_check_order_keys(const u32* inst, u32 instances, OrderKeys keys, u64 n, const u32* order,
                                    const u32* pos, u64 stride, u32* bad, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    if (keys.count == 0 || (instances && !keys.key) || !inst || !order || !pos || !bad)
        return cudaErrorInvalidValue;
    k_check_order_keys<<<grid_for(n), k_block, 0, s>>>(inst, instances, keys, n, order, pos, stride, bad);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::lifted
