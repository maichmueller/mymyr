// Device kernels of the RL helpers (include/mymyr/cuda/rl_ops.hpp): novelty rewards, prefix / schema masks and
// hindsight relabels. Each thread runs the element function of the host loop (src/mymyr/rl/ops.cpp), so both give the
// same bytes. Compiled by nvcc as C++23; includes only the device-code subset.

#include "mymyr/cuda/rl_ops.hpp"
#include "mymyr/cuda/lifted.hpp"

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
__global__ void k_generated(u32 value, u32* out) { *out = value; }

__global__ void k_hash_rows(const u64* states, u64 rows, u64 stride, u32 words, u32 nn, u64* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const u64* row = states + i * stride;
        out[i] = hash::state(row, words, row + words, nn);
    }
}

__global__ void k_hash_wide(const u64* states, u64 rows, u64 stride, u32 words, u32 nn, u64* out)
{
    const u32 lane = threadIdx.x & 31;
    for (u64 i = (u64{blockIdx.x} * blockDim.x + threadIdx.x) / 32; i < rows; i += u64{gridDim.x} * blockDim.x / 32)
    {
        const u64* row = states + i * stride;
        u32 n = 0;
        for (u32 w = lane; w < words; w += 32)
            if (row[w]) n = w + 1;
        for (u32 d = 16; d; d >>= 1)
        {
            const u32 other = __shfl_down_sync(0xFFFFFFFFu, n, d);
            n = n > other ? n : other;
        }
        n = __shfl_sync(0xFFFFFFFFu, n, 0);
        u64 h = (nn ? hash::words(row + words, nn) : hash::k_seed) ^ (u64{n} * hash::k_p1);
        for (u32 base = 0; base < n; base += 32)
        {
            const u64 word = base + lane < n ? row[base + lane] : 0;
            for (u32 q = 0; q < 32 && base + q < n; q += 2)
            {
                const u64 a = __shfl_sync(0xFFFFFFFFu, word, q);
                const u64 b = __shfl_sync(0xFFFFFFFFu, word, q + 1);
                if (!lane) h = hash::mulfold(h ^ a ^ hash::k_p1, b ^ hash::k_p2);
            }
        }
        if (!lane) out[i] = hash::mix64(h);
    }
}

__global__ void k_noop_counts(ExpansionRows src, u64 rows, u32* counts, const u32* status)
{
    const u32 lane = threadIdx.x & 31;
    for (u64 p = (u64{blockIdx.x} * blockDim.x + threadIdx.x) / 32; p < rows; p += u64{gridDim.x} * blockDim.x / 32)
    {
        const u32 b = static_cast<u32>(src.offsets[p]), e = static_cast<u32>(src.offsets[p + 1]);
        u32 n = 0;
        const bool fits = static_cast<u32>(src.offsets[rows]) <= src.capacity && (!status || *status == 0);
        if (fits)
            for (u64 j = u64{b} + lane; j < e; j += 32)
                n += !src.noop[j];
        for (u32 d = 16; d; d >>= 1) n += __shfl_down_sync(0xFFFFFFFFu, n, d);
        // an overflowing batch keeps every row: its offsets are those before dropping, so offsets[rows] = generated
        if (!lane) counts[p] = fits && n ? n : e - b;
    }
    if (!blockIdx.x && !threadIdx.x) counts[rows] = 0;
}

__device__ void clear_row(ExpansionRows o, u64 j)
{
    if (o.succ) for (u32 w = 0; w < o.words; ++w) o.succ[j * o.words + w] = 0;
    if (o.parent) o.parent[j] = -1;
    if (o.schema) o.schema[j] = -1;
    if (o.binding) for (u32 k = 0; k < o.label_width; ++k) o.binding[j * o.label_width + k] = -1;
    if (o.goal) o.goal[j] = 0;
    if (o.derived) for (u32 w = 0; w < o.derived_words; ++w) o.derived[j * o.derived_words + w] = 0;
    if (o.hash) o.hash[j] = 0;
    if (o.noop) o.noop[j] = 0;
    if (o.index) o.index[j] = -1;
}

__global__ void k_noop_scatter(ExpansionRows src, ExpansionRows dst, u64 rows)
{
    const u32 lane = threadIdx.x & 31;
    const u64 threads = u64{gridDim.x} * blockDim.x;
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < dst.capacity; j += threads)
        if (j >= static_cast<u32>(dst.offsets[rows])) clear_row(dst, j);
    for (u64 p = (u64{blockIdx.x} * blockDim.x + threadIdx.x) / 32; p < rows; p += threads / 32)
    {
        const u32 b = static_cast<u32>(src.offsets[p]), e = static_cast<u32>(src.offsets[p + 1]);
        u32 at = static_cast<u32>(dst.offsets[p]);
        const u32 count = static_cast<u32>(dst.offsets[p + 1]) - at;
        if (!count) continue;
        const bool all = count == e - b;
        for (u64 base = b; base < e && base < src.capacity; base += 32)
        {
            const u64 j = base + lane;
            const bool keep = j < e && j < src.capacity && (all || !src.noop[j]);
            const u32 mask = __ballot_sync(0xFFFFFFFFu, keep);
            const u64 k = u64{at} + __popc(mask & ((1u << lane) - 1));
            if (keep && k < dst.capacity)
            {
                if (dst.succ) for (u32 w = 0; w < dst.words; ++w) dst.succ[k * dst.words + w] = src.succ[j * src.words + w];
                if (dst.parent) dst.parent[k] = src.parent[j];
                if (dst.schema) dst.schema[k] = src.schema[j];
                if (dst.binding) for (u32 w = 0; w < dst.label_width; ++w) dst.binding[k * dst.label_width + w] = src.binding[j * src.label_width + w];
                if (dst.goal) dst.goal[k] = src.goal[j];
                if (dst.derived) for (u32 w = 0; w < dst.derived_words; ++w) dst.derived[k * dst.derived_words + w] = src.derived[j * src.derived_words + w];
                if (dst.hash) dst.hash[k] = src.hash[j];
                if (dst.noop) dst.noop[k] = src.noop[j];
                if (dst.index) dst.index[k] = static_cast<i32>(j - b);
            }
            at += __popc(mask);
        }
    }
}

__global__ void k_expansion_indices(ExpansionRows out, u64 rows)
{
    const u32 end = static_cast<u32>(out.offsets[rows]);
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < out.capacity; j += u64{gridDim.x} * blockDim.x)
    {
        if (j >= end)
        {
            if (out.hash) out.hash[j] = 0;
            if (out.noop) out.noop[j] = 0;
            if (out.index) out.index[j] = -1;
        }
        else if (out.index)
        {
            u64 lo = 0, hi = rows;
            while (lo < hi)
            {
                const u64 mid = lo + (hi - lo) / 2;
                if (static_cast<u32>(out.offsets[mid + 1]) <= j) lo = mid + 1;
                else hi = mid;
            }
            out.index[j] = static_cast<i32>(j - static_cast<u32>(out.offsets[lo]));
        }
    }
}
__global__ void k_segment_counts(const i32* offsets, u64 rows, i32* counts)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
        counts[i] = offsets[i + 1] - offsets[i];
}
}  // namespace

cudaError_t launch_segment_counts(const i32* offsets, u64 rows, i32* counts, cudaStream_t s)
{
    if (rows) k_segment_counts<<<grid_for(rows), k_block, 0, s>>>(offsets, rows, counts);
    return cudaGetLastError();
}

cudaError_t launch_expansion_generated(u32 value, u32* out, cudaStream_t s)
{
    if (out) k_generated<<<1, 1, 0, s>>>(value, out);
    return cudaGetLastError();
}

cudaError_t launch_hash_rows(const u64* states, u64 rows, u64 stride, u32 words, u32 nn, u64* out, cudaStream_t s)
{
    if (!rows) return cudaSuccess;
    if (lifted::wide_row(words)) k_hash_wide<<<grid_for(rows * 32), k_block, 0, s>>>(states, rows, stride, words, nn, out);
    else k_hash_rows<<<grid_for(rows), k_block, 0, s>>>(states, rows, stride, words, nn, out);
    return cudaGetLastError();
}

cudaError_t launch_noop_counts(ExpansionRows src, u64 rows, u32* counts, cudaStream_t s, const u32* status)
{
    k_noop_counts<<<grid_for(rows * 32), k_block, 0, s>>>(src, rows, counts, status);
    return cudaGetLastError();
}

cudaError_t launch_noop_scatter(ExpansionRows src, ExpansionRows dst, u64 rows, cudaStream_t s)
{
    const u64 n = rows * 32 > dst.capacity ? rows * 32 : dst.capacity;
    k_noop_scatter<<<grid_for(n), k_block, 0, s>>>(src, dst, rows);
    return cudaGetLastError();
}

cudaError_t launch_expansion_indices(ExpansionRows out, u64 rows, cudaStream_t s)
{
    if (out.capacity) k_expansion_indices<<<grid_for(out.capacity), k_block, 0, s>>>(out, rows);
    return cudaGetLastError();
}

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
