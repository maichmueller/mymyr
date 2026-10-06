#pragma once
// A single-pass exclusive scan over a count read on the device (cuda/src only, compiled by nvcc as C++20).
//
// One block per tile of k_tile items; the tiles chain by decoupled look-back: a tile publishes its aggregate, sums its
// predecessors' (a warp reads 32 at a time) until it meets an inclusive prefix, then publishes its own prefix. Tiles
// past the live count exit at once, so a launch sized for a capacity (a captured chunk) costs what its live items cost,
// in one launch. The tile states of a scan are a State whose flags are zero before the scan; the scan leaves them set,
// so a caller with several scans gives each its own State (a site of one buffer) and zeroes the flags of all of them
// at once before they run again (a kernel it launches anyway). Like CUB's single-pass scan, the chain relies on blocks
// starting in index order.
//
// Determinism: integer sums in tile order.

#include "mymyr/core/types.hpp"

#include <cub/block/block_scan.cuh>
#include <cuda/atomic>

namespace mymyr::cuda::live_scan
{
inline constexpr unsigned k_threads = 256;
inline constexpr unsigned k_items = 2;
inline constexpr u64 k_tile = u64{k_threads} * k_items;

/// The tiles of a scan with outputs [0, n].
__host__ __device__ __forceinline__ u64 tiles_for(u64 n) { return n / k_tile + 1; }

/// Tile states: per tile a flag (0 none, 1 aggregate, 2 inclusive prefix), the aggregate and the prefix. The flags
/// are zero before a scan.
struct State
{
    u32* flag = nullptr;
    unsigned long long* agg = nullptr;
    unsigned long long* inc = nullptr;
    u64 tiles = 0;  // the tiles it holds
};

/// A buffer of `sites` States of `tiles` tiles each: the flags of all sites first (zero them at once), then the
/// aggregates and the prefixes.
__host__ __device__ __forceinline__ u64 flag_words(u64 tiles, u32 sites) { return (tiles * sites + 1) / 2; }
__host__ __device__ __forceinline__ u64 buffer_words(u64 tiles, u32 sites) { return flag_words(tiles, sites) + 2 * tiles * sites; }
__host__ __device__ __forceinline__ State state_at(u64* w, u64 tiles, u32 sites, u32 site)
{
    State s;
    s.tiles = tiles;
    s.flag = reinterpret_cast<u32*>(w) + tiles * site;
    unsigned long long* v = reinterpret_cast<unsigned long long*>(w + flag_words(tiles, sites));
    s.agg = v + tiles * site;
    s.inc = v + tiles * (sites + site);
    return s;
}

/// The count of a scan's live items: min(*n_dev, n) (null: n), 0 once *abort (may be null) is set.
struct Live
{
    const u32* n_dev = nullptr;
    const u32* abort = nullptr;
    __device__ __forceinline__ u64 count(u64 n) const
    {
        if (abort && *abort)
            return 0;
        if (!n_dev)
            return n;
        const u64 m = *n_dev;
        return m < n ? m : n;
    }
};

namespace detail
{
using Flag = ::cuda::atomic_ref<u32, ::cuda::thread_scope_device>;

/// Warp 0 of tile `tile` > 0: publishes its aggregate, returns the sum of the tiles before it.
__device__ __forceinline__ unsigned long long look_back(const State& st, u64 tile, unsigned long long aggregate)
{
    const unsigned lane = threadIdx.x & 31;
    if (lane == 0)
    {
        st.agg[tile] = aggregate;
        Flag(st.flag[tile]).store(1u, ::cuda::memory_order_release);
    }
    unsigned long long prefix = 0;
    long long top = static_cast<long long>(tile) - 1;
    while (true)
    {
        const long long j = top - static_cast<long long>(lane);
        u32 f = 2;  // (before tile 0: an empty prefix)
        if (j >= 0)
            while ((f = Flag(st.flag[j]).load(::cuda::memory_order_acquire)) == 0)
            {
            }
        const unsigned prefixes = __ballot_sync(0xFFFFFFFFu, f == 2);
        const unsigned stop = prefixes ? static_cast<unsigned>(__ffs(static_cast<int>(prefixes)) - 1) : 32u;
        unsigned long long v = 0;
        if (j >= 0 && lane < stop)
            v = st.agg[j];
        else if (j >= 0 && lane == stop)
            v = st.inc[j];
        for (unsigned o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xFFFFFFFFu, v, o);
        prefix += __shfl_sync(0xFFFFFFFFu, v, 0);
        if (stop < 32)
            return prefix;
        top -= 32;
    }
}
}  // namespace detail

/// out[i] = in(0) + .. + in(i - 1) for i <= live = in.live(n) (<= n), and out[n] = the total; the outputs between live
/// and n are not written. in(i) is a u64 (sums of packed u32 pairs stay exact below 2^32 per half), called once per
/// item. Launch with grid tiles_for(n) and k_threads threads, over a State of at least tiles_for(n) tiles whose flags
/// are zero.
template<class In, class Out>
__global__ void __launch_bounds__(k_threads) k_scan(In in, u64 n, Out* out, State st)
{
    using BlockScan = cub::BlockScan<unsigned long long, k_threads>;
    __shared__ typename BlockScan::TempStorage temp;
    __shared__ unsigned long long s_prefix;
    const u64 live = in.live(n);
    const u64 tile = blockIdx.x;
    const u64 base = tile * k_tile;
    if (base > live)
        return;  // (block-uniform)
    const u64 first = base + u64{threadIdx.x} * k_items;
    unsigned long long v[k_items];
    unsigned long long sum = 0;
#pragma unroll
    for (unsigned k = 0; k < k_items; ++k)
    {
        v[k] = first + k < live ? static_cast<unsigned long long>(in(first + k)) : 0ull;
        sum += v[k];
    }
    unsigned long long before = 0, aggregate = 0;
    BlockScan(temp).ExclusiveSum(sum, before, aggregate);
    if (threadIdx.x < 32)
    {
        unsigned long long prefix = 0;
        if (tile > 0)
            prefix = detail::look_back(st, tile, aggregate);
        if (threadIdx.x == 0)
        {
            st.inc[tile] = prefix + aggregate;
            detail::Flag(st.flag[tile]).store(2u, ::cuda::memory_order_release);
            s_prefix = prefix;
        }
    }
    __syncthreads();
    unsigned long long x = s_prefix + before;
#pragma unroll
    for (unsigned k = 0; k < k_items; ++k)
    {
        const u64 i = first + k;
        if (i < live)
            out[i] = static_cast<Out>(x);
        else if (i == live)
        {
            out[i] = static_cast<Out>(x);
            if (n != live)
                out[n] = static_cast<Out>(x);
        }
        x += v[k];
    }
}

/// Launches k_scan over n items (outputs [0, n]); st must hold tiles_for(n) tiles, its flags zero.
template<class In, class Out>
cudaError_t launch(In in, u64 n, Out* out, State st, cudaStream_t s)
{
    const u64 tiles = tiles_for(n);
    if (tiles > st.tiles || tiles > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    k_scan<<<static_cast<unsigned>(tiles), k_threads, 0, s>>>(in, n, out, st);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::live_scan
