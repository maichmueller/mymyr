// Device kernels of the RL helpers (include/mymyr/cuda/rl_ops.hpp): novelty rewards, prefix / schema masks and
// hindsight relabels. Each thread runs the element function of the host loop (src/mymyr/rl/ops.cpp), so both give the
// same bytes. Compiled by nvcc as C++20; includes only the device-code subset.

#include "mymyr/cuda/rl_ops.hpp"

namespace mymyr::cuda
{
namespace
{
constexpr unsigned k_block = 256;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

__global__ void k_novelty(u64* seen, const u64* states, u64 rows, u32 words, u32 stride, i32* reward)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const i32 r = rl::novelty_update_row(seen + i * words, states + i * stride, words);
        if (reward)
            reward[i] = r;
    }
}

__global__ void k_prefix(rl::LabelRows x, rl::PrefixQuery q, u8* mask)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < x.size; j += u64{gridDim.x} * blockDim.x)
    {
        u64 row = 0;
        u32 o = 0;
        if (rl::prefix_hit(x, q, j, row, o))
            mask[row * q.num_objects + o] = 1;  // every writer writes 1
    }
}

__global__ void k_schema(rl::LabelRows x, u64 rows, u32 num_schemas, u8* mask)
{
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < x.size; j += u64{gridDim.x} * blockDim.x)
    {
        u64 row = 0;
        u32 s = 0;
        if (rl::schema_hit(x, rows, num_schemas, j, row, s))
            mask[row * num_schemas + s] = 1;
    }
}

__global__ void k_her(rl::HerBatch b, rl::HerConfig c)
{
    const u64 n = b.steps * b.envs * c.k;
    for (u64 x = u64{blockIdx.x} * blockDim.x + threadIdx.x; x < n; x += u64{gridDim.x} * blockDim.x)
    {
        const u32 j = static_cast<u32>(x % c.k);
        const u64 ti = x / c.k;
        rl::her_one(b, c, ti / b.envs, ti % b.envs, j);
    }
}
}  // namespace

cudaError_t launch_novelty_update(u64* seen, const u64* states, u64 rows, u32 words, u32 state_stride, i32* reward,
                                  cudaStream_t s)
{
    if (rows == 0)
        return cudaSuccess;
    if (words == 0)
        return reward ? cudaMemsetAsync(reward, 0, rows * sizeof(i32), s) : cudaSuccess;
    if (!seen || !states || state_stride < words)
        return cudaErrorInvalidValue;
    k_novelty<<<grid_for(rows), k_block, 0, s>>>(seen, states, rows, words, state_stride, reward);
    return cudaGetLastError();
}

cudaError_t launch_prefix_masks(const rl::LabelRows& x, const rl::PrefixQuery& q, u8* mask, cudaStream_t s)
{
    if (q.rows == 0 || q.num_objects == 0)
        return cudaSuccess;
    if (!mask || (q.depth > 0 && (!q.prefix || q.prefix_stride < q.depth)) || (x.size && (!x.parent || !x.schema)))
        return cudaErrorInvalidValue;
    if (const cudaError_t e = cudaMemsetAsync(mask, 0, q.rows * q.num_objects, s); e != cudaSuccess)
        return e;
    if (x.size == 0 || q.depth >= x.label_width)
        return cudaSuccess;
    k_prefix<<<grid_for(x.size), k_block, 0, s>>>(x, q, mask);
    return cudaGetLastError();
}

cudaError_t launch_schema_masks(const rl::LabelRows& x, u64 rows, u32 num_schemas, u8* mask, cudaStream_t s)
{
    if (rows == 0 || num_schemas == 0)
        return cudaSuccess;
    if (!mask || (x.size && (!x.parent || !x.schema)))
        return cudaErrorInvalidValue;
    if (const cudaError_t e = cudaMemsetAsync(mask, 0, rows * num_schemas, s); e != cudaSuccess)
        return e;
    if (x.size == 0)
        return cudaSuccess;
    k_schema<<<grid_for(x.size), k_block, 0, s>>>(x, rows, num_schemas, mask);
    return cudaGetLastError();
}

cudaError_t launch_her_relabel(const rl::HerBatch& b, const rl::HerConfig& c, cudaStream_t s)
{
    if (c.k == 0)
        return cudaErrorInvalidValue;
    if (b.steps == 0 || b.envs == 0)
        return cudaSuccess;
    if (!b.states || !b.done || !b.goal || b.row_words < b.words)
        return cudaErrorInvalidValue;
    k_her<<<grid_for(b.steps * b.envs * c.k), k_block, 0, s>>>(b, c);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda
