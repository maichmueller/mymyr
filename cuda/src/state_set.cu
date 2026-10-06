// The device state set of the layer BrFS (include/mymyr/cuda/state_set.hpp): a tag|ref table with a deterministic
// winner among equal candidates (atomicMin of the pending reference) and scan-assigned ids (the owner flags are
// computed inside the scan). Compiled by nvcc as C++20; includes only the device-code subset.

#include "mymyr/cuda/state_set.hpp"

#include "grid_stride.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <algorithm>

namespace mymyr::cuda::state_set
{
namespace
{
constexpr unsigned k_block = 256;
constexpr u64 k_tag = 0xFFFFFFFF00000000ull;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

/// Blocks at most for the loop's parent gather (a copy of at most a chunk of rows).
constexpr unsigned k_gather_blocks = 256;

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

/// The live candidates of [0, n).
__device__ __forceinline__ u64 live_count(Live l, u64 n)
{
    if (!l.live)
        return n;
    const u64 m = *l.live;
    return m < n ? m : n;
}

template<u32 W>
__global__ void k_insert(Table t, Rows arena, Rows cand, u64 n, u32* result, Live live)
{
    const u32 nw = W > 0 ? W : cand.words;
    n = live_count(live, n);
    for (u64 c = live_first(n); c < n; c += u64{gridDim.x} * blockDim.x)
    {
        const u64* x = cand.data + c * nw;
        const u64 h = row_hash(x, nw);
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
                // a slot holding a pending reference only ever changes to a smaller candidate of the same content
                if (equal<W>(x, cand.data + u64{(ref & ~k_pending) - 1} * nw, nw))
                {
                    atomicMin(reinterpret_cast<unsigned long long*>(t.slots + j), mine);
                    result[c] = static_cast<u32>(j);
                    break;
                }
            }
            else if (equal<W>(x, arena.data + u64{ref - 1} * nw, nw))
            {
                result[c] = k_dup;
                break;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ the rank scan
// Tiles of k_tile candidates, k_per (8) per thread: k_rank_bits writes each thread's owner bits (a byte) and the tile's
// owners, k_rank_offsets scans the live tiles' counts (one block), k_rank_write the ranks. Tiles past the live
// candidates exit at once.
constexpr u32 k_per = 8;
constexpr u32 k_tile = k_block * k_per;
constexpr u32 k_scan_block = 1024;

__host__ __device__ __forceinline__ u64 tiles_for(u64 n) { return (n + k_tile - 1) / k_tile; }

/// The scratch: [tiles] u32 counts (then offsets), [tiles * k_block] owner bytes.
struct RankTemp
{
    u32* tile = nullptr;
    u8* bits = nullptr;
};
RankTemp rank_temp(void* temp, u64 n)
{
    const u64 tiles = tiles_for(n + 1);
    auto* t = static_cast<u32*>(temp);
    return {t, reinterpret_cast<u8*>(t + ((tiles + 3) & ~u64{3}))};
}

/// Whether candidate c, whose insert result is r, owns its slot.
__device__ __forceinline__ bool owner(const Table& t, u32 r, u64 c)
{
    return r != k_dup && static_cast<u32>(load_slot(t.slots + r)) == (k_pending | static_cast<u32>(c + 1));
}

/// Whether p is 16-byte aligned (a thread's k_per entries then move as two uint4).
__device__ __forceinline__ bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

__global__ void k_rank_bits(Table t, const u32* result, u64 n, RankTemp tmp, Live live)
{
    const u64 m = live_count(live, n);
    using Reduce = cub::BlockReduce<u32, k_block>;
    __shared__ typename Reduce::TempStorage red;
    for (u64 tile = blockIdx.x; tile * k_tile < m; tile += gridDim.x)  // (block-uniform)
    {
        const u64 first = tile * k_tile + u64{threadIdx.x} * k_per;
        u32 bits = 0;
        if (first + k_per <= m && aligned16(result))
        {
            const uint4 a = reinterpret_cast<const uint4*>(result + first)[0];
            const uint4 b = reinterpret_cast<const uint4*>(result + first)[1];
            const u32 r[k_per] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            for (u32 j = 0; j < k_per; ++j)
                if (owner(t, r[j], first + j))
                    bits |= 1u << j;
        }
        else
            for (u32 j = 0; j < k_per; ++j)
                if (first + j < m && owner(t, result[first + j], first + j))
                    bits |= 1u << j;
        tmp.bits[tile * k_block + threadIdx.x] = static_cast<u8>(bits);
        const u32 total = Reduce(red).Sum(static_cast<u32>(__popc(bits)));
        if (threadIdx.x == 0)
            tmp.tile[tile] = total;
        __syncthreads();  // (red is reused)
    }
}

__global__ void k_rank_offsets(u32* rank, u64 n, RankTemp tmp, Live live)
{
    const u64 m = live_count(live, n);
    const u64 tiles = tiles_for(m);
    using Scan = cub::BlockScan<u32, k_scan_block>;
    __shared__ typename Scan::TempStorage sc;
    u32 carry = 0;
    for (u64 b = 0; b < tiles; b += k_scan_block)
    {
        const u64 i = b + threadIdx.x;
        const u32 x = i < tiles ? tmp.tile[i] : 0u;
        u32 y = 0, sum = 0;
        Scan(sc).ExclusiveSum(x, y, sum);
        if (i < tiles)
            tmp.tile[i] = carry + y;
        carry += sum;
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        rank[m] = carry;
        rank[n] = carry;
    }
}

__global__ void k_rank_write(u32* rank, u64 n, RankTemp tmp, Live live)
{
    const u64 m = live_count(live, n);
    using Scan = cub::BlockScan<u32, k_block>;
    __shared__ typename Scan::TempStorage sc;
    for (u64 tile = blockIdx.x; tile * k_tile < m; tile += gridDim.x)  // (block-uniform)
    {
        const u32 bits = tmp.bits[tile * k_block + threadIdx.x];
        u32 before = 0;
        Scan(sc).ExclusiveSum(static_cast<u32>(__popc(bits)), before);
        const u32 o = tmp.tile[tile] + before;
        const u64 first = tile * k_tile + u64{threadIdx.x} * k_per;
        if (first + k_per <= m && aligned16(rank))
        {
            u32 v[k_per];
            for (u32 j = 0; j < k_per; ++j)
                v[j] = o + static_cast<u32>(__popc(bits & ((1u << j) - 1)));
            reinterpret_cast<uint4*>(rank + first)[0] = make_uint4(v[0], v[1], v[2], v[3]);
            reinterpret_cast<uint4*>(rank + first)[1] = make_uint4(v[4], v[5], v[6], v[7]);
        }
        else
            for (u32 j = 0; j < k_per && first + j < m; ++j)
                rank[first + j] = o + static_cast<u32>(__popc(bits & ((1u << j) - 1)));
        __syncthreads();  // (sc is reused)
    }
}

template<u32 W>
__global__ void k_compact(Table t, Compact m, u64 n, Live live)
{
    const u32 nw = W > 0 ? W : m.words;
    n = live_count(live, n);
    const u64 o = m.offset ? *m.offset : 0;
    for (u64 c = live_first(n); c < n; c += u64{gridDim.x} * blockDim.x)
    {
        const u32 r0 = m.rank[c];
        if (m.rank[c + 1] == r0)
            continue;
        const u64 r = o + r0;
        const u64 id = m.base + r;
        const u64* x = m.cand + c * nw;
        u64* y = m.arena_tail + r * nw;
        for (u32 i = 0; i < nw; ++i)
            y[i] = x[i];
        if (m.nodes_tail)
        {
            const u32 parent = m.parent[c];
            const u32 first = m.seg_offsets[u64{parent - m.parent_base} * m.num_schemas];
            m.nodes_tail[r * 2] = parent + (m.parent_offset ? *m.parent_offset : 0u);
            m.nodes_tail[r * 2 + 1] = static_cast<u32>(c) - first;
        }
        u64* slot = t.slots + m.result[c];
        *slot = (*slot & k_tag) | (id + 1);
    }
}

template<u32 W>
__global__ void k_rehash(Table t, Rows arena, u64 count)
{
    const u32 nw = W > 0 ? W : arena.words;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < count; i += u64{gridDim.x} * blockDim.x)
    {
        const u64 h = row_hash(arena.data + i * nw, nw);
        const u64 v = (h & k_tag) | (i + 1);
        for (u64 j = h & t.mask;; j = (j + 1) & t.mask)
            if (atomicCAS(reinterpret_cast<unsigned long long*>(t.slots + j), 0ull, v) == 0)
                break;
    }
}

__global__ void k_advance(const u32* rank, u64 n, u32* count, u32* fresh)
{
    const u32 f = rank[n];
    *count += f;
    if (fresh)
        *fresh = f;
}

// a chunk's halt from its candidates m (ctl->halt clear)
__device__ void size_chunk(ChunkCtl& ctl, ChunkRecord& rec, u32 m, const ChunkLimits& lim)
{
    rec.candidates = m;
    u32 h = 0;
    if (ctl.count >= lim.budget)
        h = k_halt_budget;
    else if (m > lim.capacity || u64{ctl.count} + m > lim.table)
        h = k_halt_room;
    else if (lim.stop_at_goal && rec.goals > 0)
        h = k_halt_goal;
    ctl.halt = h;
    rec.halt = h;
    if (!h)
        ctl.live = m;
}

__global__ void k_chunk_size(ChunkCtl* ctl, ChunkRecord* rec, const u32* total, ChunkLimits lim)
{
    ctl->live = 0;
    if (ctl->halt)
    {
        rec->halt = ctl->halt;
        return;
    }
    size_chunk(*ctl, *rec, *total, lim);
}

__global__ void k_loop_plan(LoopCtl* ctl, LoopLimits lim)
{
    LoopCtl& c = *ctl;
    ChunkCtl& k = c.chunk;
    k.halt = 0;
    k.live = 0;
    c.rec = ChunkRecord{};
    c.ns = 0;
    if (c.end != k_loop_running)
        return;
    if (c.b == c.le)
    {
        // the next layer
        c.lb = c.le;
        c.le = k.count;
        c.started = 0;
        if (c.lb == c.le)
        {
            c.end = k_loop_done;
            return;
        }
        if (k.count >= lim.budget)
        {
            c.end = k_loop_budget;
            return;
        }
        ++c.layers;
        c.started = 1;
    }
    const u32 left = c.le - c.b;
    c.ns = left < lim.chunk ? left : lim.chunk;
}

__global__ void k_gather_parents(Rows arena, LoopCtl* ctl, u64* out)
{
    const u32 W = arena.words;
    const u64 n = u64{ctl->ns} * W;
    const u64* src = arena.data + u64{ctl->b} * W;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        out[i] = src[i];
}

__global__ void k_loop_size(LoopCtl* ctl, const u32* offsets, u32 S, ChunkLimits lim)
{
    LoopCtl& c = *ctl;
    ChunkCtl& k = c.chunk;
    k.live = 0;
    if (k.halt)
    {
        c.rec.halt = k.halt;
        return;
    }
    u32 m = offsets[u64{c.ns} * S];
    if (m > lim.capacity && c.ns > 1 && c.rec.goals == 0)
    {
        // the most parents p < ns whose candidates offsets[p * S] fit (the offsets do not decrease)
        u32 lo = 0, hi = c.ns;
        while (hi - lo > 1)
        {
            const u32 mid = lo + (hi - lo) / 2;
            if (offsets[u64{mid} * S] <= lim.capacity)
                lo = mid;
            else
                hi = mid;
        }
        if (lo > 0)
        {
            c.ns = lo;
            m = offsets[u64{lo} * S];
            ++c.cuts;
        }
    }
    size_chunk(k, c.rec, m, lim);
}

__global__ void k_loop_next(LoopCtl* ctl, LoopLimits lim, cudaGraphConditionalHandle handle, u32 looped)
{
    LoopCtl& c = *ctl;
    const ChunkCtl& k = c.chunk;
    if (c.end == k_loop_running)
    {
        if (k.halt || c.rec.error)
            c.end = k_loop_halt;  // (the cursor stays at the chunk: the host redoes it)
        else
        {
            c.expanded += c.ns;
            c.generated += c.rec.candidates;
            c.goal_states += c.rec.goals;
            ++c.chunks;
            if (c.ns)
            {
                const u64 r = (u64{c.rec.candidates} * 256 + c.ns - 1) / c.ns;
                const u32 q = static_cast<u32>(r < 0xFFFFFFFFull ? r : 0xFFFFFFFFull);
                if (q > c.max_ratio)
                    c.max_ratio = q;
            }
            c.b += c.ns;
            if (c.b == c.le && k.count == c.le)
                c.end = k_loop_done;  // (the next layer is empty)
            else if (c.b == c.le && k.count >= lim.budget)
                c.end = k_loop_budget;
            else if (c.b == c.le && u64{k.count} - c.le > lim.max_layer)
                c.end = k_loop_grow;
            else if (u64{k.count} + lim.capacity > lim.table || u64{k.count} + lim.capacity > lim.arena)
                c.end = k_loop_room;
            else if (c.steps_left <= 1)
                c.end = k_loop_steps;
            if (c.steps_left > 0)
                --c.steps_left;
        }
    }
    if (looped && c.end != k_loop_running)
        cudaGraphSetConditional(handle, 0);
}

__global__ void k_relayout(const u64* src, u32 sw, u64* dst, u32 dw, u64 rows)
{
    const u64 n = rows * dw;
    for (u64 k = u64{blockIdx.x} * blockDim.x + threadIdx.x; k < n; k += u64{gridDim.x} * blockDim.x)
    {
        const u64 i = k / dw;
        const u32 w = static_cast<u32>(k % dw);
        dst[k] = w < sw ? src[i * sw + w] : 0;
    }
}
}  // namespace

cudaError_t launch_insert(Table t, Rows arena, Rows cand, u64 n, u32* result, cudaStream_t s, Live live)
{
    if (n == 0)
        return cudaSuccess;
    const unsigned g = grid_for(n);
    switch (cand.words)
    {
        case 1: k_insert<1><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        case 2: k_insert<2><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        case 3: k_insert<3><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        case 4: k_insert<4><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        case 8: k_insert<8><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        case 16: k_insert<16><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
        default: k_insert<0><<<g, k_block, 0, s>>>(t, arena, cand, n, result, live); break;
    }
    return cudaGetLastError();
}

u64 rank_temp_bytes(u64 n)
{
    const u64 tiles = tiles_for(n + 1);
    return ((tiles + 3) & ~u64{3}) * sizeof(u32) + tiles * k_block;
}

cudaError_t launch_rank(Table t, const u32* result, u64 n, u32* rank, void* temp, u64 temp_bytes, cudaStream_t s, Live live)
{
    if (n + 1 > 0x7FFFFFFFull || temp_bytes < rank_temp_bytes(n))
        return cudaErrorInvalidValue;
    const RankTemp tmp = rank_temp(temp, n);
    const u64 tiles = tiles_for(n);
    const auto g = static_cast<unsigned>(tiles);
    if (tiles)
        k_rank_bits<<<g, k_block, 0, s>>>(t, result, n, tmp, live);
    k_rank_offsets<<<1, k_scan_block, 0, s>>>(rank, n, tmp, live);
    if (tiles)
        k_rank_write<<<g, k_block, 0, s>>>(rank, n, tmp, live);
    return cudaGetLastError();
}

cudaError_t launch_compact(Table t, Compact c, u64 n, cudaStream_t s, Live live)
{
    if (n == 0)
        return cudaSuccess;
    const unsigned g = grid_for(n);
    switch (c.words)
    {
        case 1: k_compact<1><<<g, k_block, 0, s>>>(t, c, n, live); break;
        case 2: k_compact<2><<<g, k_block, 0, s>>>(t, c, n, live); break;
        case 4: k_compact<4><<<g, k_block, 0, s>>>(t, c, n, live); break;
        default: k_compact<0><<<g, k_block, 0, s>>>(t, c, n, live); break;
    }
    return cudaGetLastError();
}

cudaError_t launch_advance(const u32* rank, u64 n, u32* count, u32* fresh, cudaStream_t s)
{
    k_advance<<<1, 1, 0, s>>>(rank, n, count, fresh);
    return cudaGetLastError();
}

cudaError_t launch_chunk_size(ChunkCtl* ctl, ChunkRecord* rec, const u32* total, const ChunkLimits& limits, cudaStream_t s)
{
    k_chunk_size<<<1, 1, 0, s>>>(ctl, rec, total, limits);
    return cudaGetLastError();
}

cudaError_t launch_loop_plan(LoopCtl* ctl, LoopLimits lim, cudaStream_t s)
{
    k_loop_plan<<<1, 1, 0, s>>>(ctl, lim);
    return cudaGetLastError();
}

cudaError_t launch_loop_size(LoopCtl* ctl, const u32* offsets, u32 S, const ChunkLimits& limits, cudaStream_t s)
{
    k_loop_size<<<1, 1, 0, s>>>(ctl, offsets, S, limits);
    return cudaGetLastError();
}

cudaError_t launch_gather_parents(Rows arena, LoopCtl* ctl, u64* out, u32 chunk, cudaStream_t s)
{
    if (chunk && arena.words)
        k_gather_parents<<<std::min(grid_for(u64{chunk} * arena.words), k_gather_blocks), k_block, 0, s>>>(arena, ctl, out);
    return cudaGetLastError();
}

cudaError_t launch_loop_next(LoopCtl* ctl, LoopLimits lim, unsigned long long handle, cudaStream_t s)
{
    k_loop_next<<<1, 1, 0, s>>>(ctl, lim, static_cast<cudaGraphConditionalHandle>(handle), handle ? 1u : 0u);
    return cudaGetLastError();
}

cudaError_t launch_rehash(Table t, Rows arena, u64 count, cudaStream_t s)
{
    if (count)
        k_rehash<0><<<grid_for(count), k_block, 0, s>>>(t, arena, count);
    return cudaGetLastError();
}

cudaError_t launch_relayout(const u64* src, u32 src_words, u64* dst, u32 dst_words, u64 rows, cudaStream_t s)
{
    if (rows && dst_words)
        k_relayout<<<grid_for(rows * dst_words), k_block, 0, s>>>(src, src_words, dst, dst_words, rows);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::state_set
