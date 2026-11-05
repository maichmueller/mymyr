// The successors' axioms of a multi-instance expansion (expand_axioms.hpp).
// Compiled by nvcc as C++23; includes only the device-code subset.

#include "expand_axioms.hpp"

#include "mymyr/rl/task_arrays_view.hpp"

namespace mymyr::cuda::detail
{
namespace
{
constexpr unsigned k_threads = 256;

unsigned blocks_for(u64 n)
{
    const u64 b = (n + k_threads - 1) / k_threads;
    return static_cast<unsigned>(b < 65535 ? (b ? b : 1) : 65535);
}

__global__ void k_successor_ids(const u32* offsets, u32 num_schemas, u64 rows, u64 base, const i32* inst,
                                const u32* map_rows, const i32* batch_offsets, u64 capacity, i32* out)
{
    for (u64 q = u64{blockIdx.x} * blockDim.x + threadIdx.x; q < rows; q += u64{gridDim.x} * blockDim.x)
    {
        const u64 first = offsets[q * num_schemas], end = offsets[(q + 1) * num_schemas];
        const u64 at = map_rows ? static_cast<u64>(batch_offsets[map_rows[q]]) : base + first;
        const i32 id = inst[q];
        for (u64 k = 0; k < end - first && at + k < capacity; ++k)
            out[at + k] = id;
    }
}

__global__ void k_goal_rows_derived_multi(const rl::dev::TaskView* views, u32 instances, const i32* inst,
                                          const u64* states, u64 stride, u32 words, u64 n, const u64* derived,
                                          u32 derived_words, u8* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
    {
        const i32 k = inst[i];
        if (k < 0 || static_cast<u32>(k) >= instances)
            continue;
        out[i] = rl::dev::goal_holds(views[k], states + i * stride, words,
                                       derived ? derived + i * derived_words : nullptr, derived_words)
                     ? 1
                     : 0;
    }
}

__global__ void k_widen_count(const u32* src, u64* dst) { *dst = *src; }
}  // namespace

cudaError_t launch_successor_ids(const u32* offsets, u32 num_schemas, u64 rows, u64 base, const i32* inst,
                                 const u32* map_rows, const i32* batch_offsets, u64 capacity, i32* out, cudaStream_t s)
{
    if (rows == 0)
        return cudaSuccess;
    if (!offsets || !num_schemas || !inst || !out || (map_rows && !batch_offsets))
        return cudaErrorInvalidValue;
    k_successor_ids<<<blocks_for(rows), k_threads, 0, s>>>(offsets, num_schemas, rows, base, inst, map_rows,
                                                           batch_offsets, capacity, out);
    return cudaGetLastError();
}

cudaError_t launch_goal_rows_derived_multi(const rl::dev::TaskView* views, u32 instances, const i32* inst,
                                           const u64* states, u64 stride, u32 words, u64 n, const u64* derived,
                                           u32 derived_words, u8* out, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    if (!views || !inst || !states || !out || (derived_words && !derived))
        return cudaErrorInvalidValue;
    k_goal_rows_derived_multi<<<blocks_for(n), k_threads, 0, s>>>(views, instances, inst, states, stride, words, n,
                                                                  derived, derived_words, out);
    return cudaGetLastError();
}

cudaError_t launch_widen_count(const u32* src, u64* dst, cudaStream_t s)
{
    k_widen_count<<<1, 1, 0, s>>>(src, dst);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::detail
