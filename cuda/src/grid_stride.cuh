#pragma once
// Grid-stride loops for launches sized for a capacity over a count read on the device (cuda/src only, compiled by
// nvcc).
//
// The block scheduler fills an SM with consecutive blocks, so under a grid sized for a capacity the few live items of
// a plain grid-stride loop (item i to block i / blockDim) end up on the first blocks and few SMs, leaving most of the
// device idle. Spread loops hand the warps of 32 consecutive items to the blocks in turn (item warp w to block
// w % gridDim), so that the live items reach every block and the accesses of a warp stay consecutive.

#include "mymyr/core/types.hpp"

namespace mymyr::cuda
{
/// This thread's first item in a spread grid-stride loop (step gridDim.x * blockDim.x; blockDim.x a multiple of 32).
__device__ __forceinline__ u64 spread_first()
{
    return ((u64{threadIdx.x >> 5} * gridDim.x + blockIdx.x) << 5) + (threadIdx.x & 31);
}

/// This warp's first index in a spread warp-stride loop (step gridDim.x * blockDim.x / 32).
__device__ __forceinline__ u64 spread_warp() { return u64{threadIdx.x >> 5} * gridDim.x + blockIdx.x; }

/// Live items from which live_first maps them in block order: about a wave of resident threads (L4 89K, H100 270K).
inline constexpr u64 k_spread_below = u64{1} << 17;

/// This thread's first item in a grid-stride loop over `m` live items (the same m in every thread) of a grid sized for a
/// capacity of about one item per thread: spread while they are few (they reach every SM), in block order from
/// k_spread_below on. Block order keeps the items in flight consecutive, and a hash insert's probes of duplicate
/// candidates, which come close together, then hit L2: the state set's insert of 9M candidates (miconic f28) took 1.22
/// ms as a spread loop over 256 blocks, 0.96 ms in block order over a full grid (L4).
__device__ __forceinline__ u64 live_first(u64 m)
{
    return m < k_spread_below ? spread_first() : u64{blockIdx.x} * blockDim.x + threadIdx.x;
}
}  // namespace mymyr::cuda
