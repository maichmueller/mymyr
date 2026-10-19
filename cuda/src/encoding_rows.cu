#include "encoding_rows.hpp"

#include <stdexcept>

namespace mymyr::cuda
{
namespace
{
__device__ void copy_column(const u64* src, u64 stride, u32 width, const u64* dst, u64 dst_stride,
                            u64 rows, u64 first, u64 capacity)
{
    auto* output = const_cast<u64*>(dst);
    for (u64 j = threadIdx.x; j < rows * width; j += blockDim.x)
    {
        const u64 row = j / width, word = j % width;
        output[((first + row) % capacity) * dst_stride + word] = src[row * stride + word];
    }
}

__global__ void append_kernel(DeviceEncodeInput in, DeviceEncodeInput ring, u64* status)
{
    const u64 head = status[0], tail = status[1];
    if (tail > ring.graphs || in.graphs > ring.graphs - tail)
    {
        if (threadIdx.x == 0) status[2] = 1;
        return;
    }
    const u64 first = (head + tail) % ring.graphs;
    copy_column(in.states, in.state_stride, ring.state_words, ring.states, ring.state_stride, in.graphs, first, ring.graphs);
    if (ring.derived)
        copy_column(in.derived, in.derived_stride, static_cast<u32>(ring.derived_stride), ring.derived, ring.derived_stride,
                    in.graphs, first, ring.graphs);
    if (ring.goal_pos)
    {
        copy_column(in.goal_pos, in.goal_stride, ring.goal_words, ring.goal_pos, ring.goal_stride, in.graphs, first, ring.graphs);
        copy_column(in.goal_neg, in.goal_stride, ring.goal_words, ring.goal_neg, ring.goal_stride, in.graphs, first, ring.graphs);
    }
    if (ring.goal_derived_pos)
    {
        copy_column(in.goal_derived_pos, in.goal_derived_stride, ring.goal_derived_words, ring.goal_derived_pos,
                    ring.goal_derived_stride, in.graphs, first, ring.graphs);
        copy_column(in.goal_derived_neg, in.goal_derived_stride, ring.goal_derived_words, ring.goal_derived_neg,
                    ring.goal_derived_stride, in.graphs, first, ring.graphs);
    }
    for (u64 row = threadIdx.x; row < in.graphs; row += blockDim.x)
        const_cast<i32*>(ring.task_ids)[(first + row) % ring.graphs] = in.task_ids ? in.task_ids[row] : 0;
    __syncthreads();
    if (threadIdx.x == 0) status[1] = tail + in.graphs;
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
    append_kernel<<<1, 256, 0, stream>>>(in, ring, status);
    return cudaGetLastError();
}
cudaError_t launch_clear_rows(u64* status, u64 capacity, bool overflow, cudaStream_t stream)
{
    clear_kernel<<<1, 1, 0, stream>>>(status, capacity, overflow);
    return cudaGetLastError();
}
}
