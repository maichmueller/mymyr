#include "encoding_rows.hpp"

#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
constexpr unsigned k_append_threads = 256;
constexpr unsigned k_append_warps = k_append_threads / 32;
/// A grid large enough to keep the copy at memory speed; its warps stride over the rows.
constexpr u64 k_append_blocks = 2048;

__device__ bool fits(const u64* status, u64 rows, u64 capacity)
{
    const u64 tail = status[1];
    return tail <= capacity && rows <= capacity - tail;
}

__device__ void copy_column(const u64* src, u64 stride, u32 width, const u64* dst, u64 dst_stride, u64 row, u64 at,
                            u32 lane)
{
    auto* output = const_cast<u64*>(dst);
    for (u32 word = lane; word < width; word += 32)
        output[at * dst_stride + word] = src[row * stride + word];
}

/// The copy of an append that fits: a warp per row, its lanes over the row's words, the warps striding over the rows.
/// The tail stays as it is (append_commit_kernel advances it after the copy).
__global__ void __launch_bounds__(k_append_threads) append_copy_kernel(DeviceEncodeInput in, DeviceEncodeInput ring,
                                                                       const u64* status)
{
    if (!fits(status, in.graphs, ring.graphs))
        return;
    u64 first = status[0] % ring.graphs + status[1] % ring.graphs;
    if (first >= ring.graphs)
        first -= ring.graphs;
    const u32 lane = threadIdx.x & 31u;
    const u64 warps = u64{gridDim.x} * k_append_warps;
    for (u64 row = u64{blockIdx.x} * k_append_warps + (threadIdx.x >> 5); row < in.graphs; row += warps)
    {
        u64 at = first + row;
        if (at >= ring.graphs)
            at -= ring.graphs;
        copy_column(in.states, in.state_stride, ring.state_words, ring.states, ring.state_stride, row, at, lane);
        if (ring.derived)
            copy_column(in.derived, in.derived_stride, static_cast<u32>(ring.derived_stride), ring.derived,
                        ring.derived_stride, row, at, lane);
        if (ring.goal_pos)
        {
            copy_column(in.goal_pos, in.goal_stride, ring.goal_words, ring.goal_pos, ring.goal_stride, row, at, lane);
            copy_column(in.goal_neg, in.goal_stride, ring.goal_words, ring.goal_neg, ring.goal_stride, row, at, lane);
        }
        if (ring.goal_derived_pos)
        {
            copy_column(in.goal_derived_pos, in.goal_derived_stride, ring.goal_derived_words, ring.goal_derived_pos,
                        ring.goal_derived_stride, row, at, lane);
            copy_column(in.goal_derived_neg, in.goal_derived_stride, ring.goal_derived_words, ring.goal_derived_neg,
                        ring.goal_derived_stride, row, at, lane);
        }
        if (lane == 0)
            const_cast<i32*>(ring.task_ids)[at] = in.task_ids ? in.task_ids[row] : 0;
    }
}

/// One thread after the copy: the new tail, or the sticky overflow flag for an append that did not fit.
__global__ void append_commit_kernel(u64 rows, u64 capacity, u64* status)
{
    if (fits(status, rows, capacity))
        status[1] += rows;
    else
        status[2] = 1;
}

__global__ void clear_kernel(u64* status, u64 capacity, bool overflow)
{
    status[0] = (status[0] + status[1]) % capacity;
    status[1] = 0;
    if (overflow) status[2] = 0;
}
}

cudaError_t launch_append_rows(const DeviceEncodeInput& in, const DeviceEncodeInput& ring, u64* status, cudaStream_t stream)
{
    const u64 wanted = (in.graphs + k_append_warps - 1) / k_append_warps;
    const unsigned blocks = static_cast<unsigned>(wanted < k_append_blocks ? wanted : k_append_blocks);
    append_copy_kernel<<<blocks, k_append_threads, 0, stream>>>(in, ring, status);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return e;
    append_commit_kernel<<<1, 1, 0, stream>>>(in.graphs, ring.graphs, status);
    return cudaGetLastError();
}
cudaError_t launch_clear_rows(u64* status, u64 capacity, bool overflow, cudaStream_t stream)
{
    clear_kernel<<<1, 1, 0, stream>>>(status, capacity, overflow);
    return cudaGetLastError();
}
}
