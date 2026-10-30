#include "mymyr/cuda/rl_ops.hpp"

namespace mymyr::cuda
{
namespace
{
struct MatchWarp
{
    __device__ u32 lane() const { return threadIdx.x % 32; }
    __device__ u32 size() const { return 32; }
    __device__ u32 bit_or(u32 value) const
    {
        for (int offset = 16; offset; offset /= 2)
            value |= __shfl_xor_sync(0xffffffffu, value, offset);
        return value;
    }
    __device__ u32 byte_sum(u32 value) const
    {
        for (int offset = 16; offset; offset /= 2)
            value = __vaddus4(value, __shfl_xor_sync(0xffffffffu, value, offset));
        return value;
    }
};
__global__ void initialize(rl::MatchTemplates t, rl::MatchBatch b)
{
    for (u64 row = u64{blockIdx.x} * blockDim.x + threadIdx.x; row < b.rows; row += u64{gridDim.x} * blockDim.x)
    {
        b.count[row] = 0;
        for (u32 j = 0; j < b.capacity; ++j)
        {
            const u64 at = row * b.capacity + j;
            b.found[at] = 0;
            b.template_id[at] = -1;
            for (u32 v = 0; v < t.max_vars; ++v)
                b.binding[at * t.max_vars + v] = -1;
        }
    }
}
template<u32 MaxVars, bool First>
__global__ void match(rl::MatchTemplates t, rl::MatchBatch b, u64 jobs)
{
    const MatchWarp grp;
    for (u64 job = (u64{blockIdx.x} * blockDim.x + threadIdx.x) / 32; job < jobs;
         job += u64{gridDim.x} * blockDim.x / 32)
    {
        const u32 width = t.capacity ? t.capacity : 1;
        const u64 row = First ? job : job / width;
        const u32 local = First ? 0 : static_cast<u32>(job % width);
        const i32 g = b.task_ids ? b.task_ids[row] : 0;
        u64 flags = 0;
        if (g < 0 || static_cast<u32>(g) >= t.num_instances)
            flags = 4;
        else
        {
            const i32 begin = t.instance_offsets[g], end = t.instance_offsets[g + 1];
            if (!First && local >= static_cast<u32>(end - begin))
                continue;
            const i32 first = begin + static_cast<i32>(local);
            const i32 last = First ? end : first + 1;
            for (i32 k = first; k < last; ++k)
            {
                if (t.num_vars[k] > static_cast<i32>(MaxVars))
                {
                    flags |= 2;
                    continue;
                }
                i32 binding[MaxVars];
                u32 base[MaxVars], masks[MaxVars], need[MaxVars];
                rl::match_detail::Budget budget{b.max_steps};
                const bool found = rl::match_detail::search<MatchWarp, (MaxVars > 2), (MaxVars > 2)>(grp, t, static_cast<u32>(g), static_cast<u32>(k),
                        b.states + row * b.row_words, b.derived ? b.derived + row * b.derived_words : nullptr,
                        binding, base, masks, need, budget);
                if (budget.exhausted)
                    flags |= 8;
                if (!found)
                    continue;
                const u32 j = First ? 0 : local;
                if (j >= b.capacity)
                    flags |= 1;
                if (grp.lane() == 0)
                {
                    // A first-result row has one writer.
                    if constexpr (First)
                        b.count[row] = 1;
                    else
                        atomicAdd(b.count + row, 1);
                    if (j < b.capacity)
                    {
                        const u64 at = row * b.capacity + j;
                        b.found[at] = 1;
                        b.template_id[at] = k;
                        for (u32 v = 0; v < static_cast<u32>(t.num_vars[k]); ++v)
                            b.binding[at * t.max_vars + v] = binding[v];
                    }
                }
                if (First)
                    break;
            }
        }
        if (flags && b.status && grp.lane() == 0)
            atomicOr(reinterpret_cast<unsigned long long*>(b.status), static_cast<unsigned long long>(flags));
    }
}
}

cudaError_t launch_match_first(const rl::MatchTemplates& t, const rl::MatchBatch& b, cudaStream_t stream)
{
    if (!b.status)
        return cudaErrorInvalidValue;
    auto err = cudaMemsetAsync(b.status, 0, sizeof(u64), stream);
    if (err != cudaSuccess)
        return err;
    if (!b.rows)
        return cudaSuccess;
    const u32 blocks = static_cast<u32>((b.rows + 127) / 128 > 65535 ? 65535 : (b.rows + 127) / 128);
    initialize<<<blocks, 128, 0, stream>>>(t, b);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;
    const u64 jobs = b.rows * (b.first || !t.capacity ? 1 : t.capacity);
    if (jobs)
    {
        const u32 search_blocks = static_cast<u32>((jobs + 3) / 4 > 65535 ? 65535 : (jobs + 3) / 4);
        if (t.max_vars <= 2)
        {
            if (b.first)
                match<2, true><<<search_blocks, 128, 0, stream>>>(t, b, jobs);
            else
                match<2, false><<<search_blocks, 128, 0, stream>>>(t, b, jobs);
        }
        else
        {
            if (b.first)
                match<64, true><<<search_blocks, 128, 0, stream>>>(t, b, jobs);
            else
                match<64, false><<<search_blocks, 128, 0, stream>>>(t, b, jobs);
        }
    }
    return cudaGetLastError();
}
}  // namespace mymyr::cuda
