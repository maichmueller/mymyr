// The device kernels of task suites (include/mymyr/cuda/suite_kernels.hpp). Compiled by nvcc as C++20; includes only
// the device-code subset.

#include "mymyr/cuda/suite_kernels.hpp"

namespace mymyr::cuda::suitek
{
namespace
{
constexpr unsigned k_block = 256;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

__global__ void k_domain_ids(const i32* task_ids, u64 rows, const u32* domain_of, const u32* local_of, u32 instances,
                             u32 domains, u32* out, u64 stride)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const i32 t = task_ids[i];
        const bool valid = t >= 0 && static_cast<u32>(t) < instances;
        const u32 dom = valid ? domain_of[t] : domains;
        const u32 loc = valid ? local_of[t] : 0xFFFFFFFFu;
        for (u32 d = 0; d < domains; ++d)
            out[d * stride + i] = d == dom ? loc : 0xFFFFFFFFu;
    }
}

__global__ void k_add(i32* a, u64 n, i32 delta)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        a[i] += delta;
}

/// Domains whose counters a block keeps in shared memory (more: global atomics).
constexpr u32 k_shared_domains = 1024;

__global__ void k_classify_init(u32* ctl, u32 domains)
{
    const u32 n = k_classify_words + 2 * domains;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x)
        ctl[i] = i == k_changes || (i >= k_classify_words && i < k_classify_words + domains) ? 0u : 0xFFFFFFFFu;
}

__global__ void k_classify(const i32* ids, u64 rows, const u32* dom_of, const u32* loc_of, u32 instances, u32 domains,
                           i32* local, u32* ctl)
{
    extern __shared__ u32 sh[];  // [domains] rows, [domains] first rows, changes
    const bool shared = domains <= k_shared_domains;
    u32* per = shared ? sh : ctl + k_classify_words;
    u32* first = shared ? sh + domains : ctl + k_classify_words + domains;
    u32* changes = shared ? sh + 2 * domains : ctl + k_changes;
    if (shared)
    {
        for (u32 k = threadIdx.x; k < 2 * domains + 1; k += blockDim.x)
            sh[k] = k >= domains && k < 2 * domains ? 0xFFFFFFFFu : 0u;
        __syncthreads();
    }
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const i32 t = ids[i];
        if (t < 0 || static_cast<u32>(t) >= instances)
        {
            atomicMin(ctl + k_bad_id, static_cast<u32>(i));
            local[i] = 0;
            continue;
        }
        const u32 d = dom_of[t];
        local[i] = static_cast<i32>(loc_of[t]);
        atomicAdd(per + d, 1u);
        atomicMin(first + d, static_cast<u32>(i));
        if (i > 0)
        {
            const i32 tp = ids[i - 1];
            const u32 dp = tp >= 0 && static_cast<u32>(tp) < instances ? dom_of[tp] : domains;
            if (dp != d)
                atomicAdd(changes, 1u);
        }
    }
    if (shared)
    {
        __syncthreads();
        for (u32 k = threadIdx.x; k < domains; k += blockDim.x)
        {
            if (sh[k])
                atomicAdd(ctl + k_classify_words + k, sh[k]);
            if (sh[domains + k] != 0xFFFFFFFFu)
                atomicMin(ctl + k_classify_words + domains + k, sh[domains + k]);
        }
        if (threadIdx.x == 0 && sh[2 * domains])
            atomicAdd(ctl + k_changes, sh[2 * domains]);
    }
}

__global__ void k_local_ids(const i32* ids, const u32* index, u64 n, const u32* loc_of, i32* local)
{
    for (u64 p = u64{blockIdx.x} * blockDim.x + threadIdx.x; p < n; p += u64{gridDim.x} * blockDim.x)
        local[p] = static_cast<i32>(loc_of[ids[index[p]]]);
}

}  // namespace

cudaError_t launch_domain_ids(const i32* task_ids, u64 rows, const u32* domain_of, const u32* local_of, u32 instances,
                              u32 domains, u32* out, u64 stride, cudaStream_t s)
{
    if (rows == 0)
        return cudaSuccess;
    if (!task_ids || !domain_of || !local_of || !out || stride < rows)
        return cudaErrorInvalidValue;
    k_domain_ids<<<grid_for(rows), k_block, 0, s>>>(task_ids, rows, domain_of, local_of, instances, domains, out, stride);
    return cudaGetLastError();
}

cudaError_t launch_add(i32* a, u64 n, i32 delta, cudaStream_t s)
{
    if (n == 0 || delta == 0)
        return cudaSuccess;
    if (!a)
        return cudaErrorInvalidValue;
    k_add<<<grid_for(n), k_block, 0, s>>>(a, n, delta);
    return cudaGetLastError();
}

cudaError_t launch_classify(const i32* task_ids, u64 rows, const u32* domain_of, const u32* local_of, u32 instances,
                            u32 domains, i32* local, u32* ctl, cudaStream_t s)
{
    if (!ctl || domains == 0 || (rows && (!task_ids || !domain_of || !local_of || !local)))
        return cudaErrorInvalidValue;
    k_classify_init<<<1, k_block, 0, s>>>(ctl, domains);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess || rows == 0)
        return e;
    const u64 shared = domains <= k_shared_domains ? (2 * u64{domains} + 1) * sizeof(u32) : 0;
    const u64 blocks = (rows + k_block - 1) / k_block;
    k_classify<<<static_cast<unsigned>(blocks < 1024 ? blocks : 1024), k_block, shared, s>>>(
        task_ids, rows, domain_of, local_of, instances, domains, local, ctl);
    return cudaGetLastError();
}

cudaError_t launch_local_ids(const i32* task_ids, const u32* index, u64 n, const u32* local_of, i32* local, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    if (!task_ids || !index || !local_of || !local)
        return cudaErrorInvalidValue;
    k_local_ids<<<grid_for(n), k_block, 0, s>>>(task_ids, index, n, local_of, local);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::suitek
