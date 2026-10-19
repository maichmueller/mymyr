#include "encoding_collate.hpp"

#include <cub/device/device_scan.cuh>

#include <limits>
#include <stdexcept>

namespace mymyr::cuda::detail
{
namespace
{
constexpr u32 k_block = 256;
struct AddTotals
{
    __host__ __device__ static u64 add(u64 a, u64 b) { return b > ~u64{0} - a ? ~u64{0} : a + b; }
    __host__ __device__ CollationTotals operator()(const CollationTotals& a, const CollationTotals& b) const
    {
        return {add(a.entities, b.entities), add(a.tuples, b.tuples), add(a.args, b.args), add(a.targets, b.targets)};
    }
};
u32 blocks(u64 n) { return static_cast<u32>(n > u64{65535} * k_block ? 65535 : (n + k_block - 1) / k_block); }
i64 length(u64 n)
{
    if (n > static_cast<u64>(std::numeric_limits<i64>::max()))
        throw std::overflow_error("mymyr: encoding: collation scan length overflow");
    return static_cast<i64>(n);
}
template<class I> __device__ const I* as(const void* p) { return static_cast<const I*>(p); }
template<class I> __device__ I* as(void* p) { return static_cast<I*>(p); }

__device__ const CollationSlice& slice_of(const CollationParams& p, u64 graph)
{
    u64 lo = 0, hi = p.batches;
    while (lo + 1 < hi)
    {
        const u64 mid = (lo + hi) / 2;
        if (p.slices[mid].first <= graph) lo = mid;
        else hi = mid;
    }
    return p.slices[lo];
}
__device__ bool bad_source(const CollationParams& p)
{
    for (u64 b = 0; b < p.batches; ++b)
        if (p.slices[b].batch.sizes && p.slices[b].batch.sizes->overflow)
            return true;
    return false;
}
template<class I> __global__ void count_kernel(CollationParams p)
{
    const bool bad = bad_source(p);
    for (u64 at = u64{blockIdx.x} * blockDim.x + threadIdx.x; at < p.entries; at += u64{gridDim.x} * blockDim.x)
    {
        CollationTotals counts;
        if (at && !bad)
        {
            if (at <= p.graphs)
            {
                const u64 graph = at - 1;
                const auto& s = slice_of(p, graph);
                const u64 g = graph - s.first;
                const auto& a = s.batch.arrays;
                counts.entities = static_cast<u64>(as<I>(a.entity_offsets)[g + 1] - as<I>(a.entity_offsets)[g]);
                if (a.target_offsets)
                    counts.targets = static_cast<u64>(as<I>(a.target_offsets)[g + 1] - as<I>(a.target_offsets)[g]);
            }
            else
            {
                const u64 cell = at - p.graphs - 1;
                const u64 graph = p.relation_major ? cell % p.graphs : cell / p.relations;
                const u64 relation = p.relation_major ? cell / p.graphs : cell % p.relations;
                const auto& s = slice_of(p, graph);
                const u64 src = p.relation_major ? relation * s.batch.graphs + graph - s.first
                                                 : (graph - s.first) * p.relations + relation;
                const auto& a = s.batch.arrays;
                counts.tuples = static_cast<u64>(as<I>(a.relation_offsets)[src + 1] - as<I>(a.relation_offsets)[src]);
                counts.args = static_cast<u64>(as<I>(a.relation_arg_offsets)[src + 1] - as<I>(a.relation_arg_offsets)[src]);
            }
        }
        p.totals[at] = counts;
    }
}
__device__ encoding::FlatSizes sizes_of(const CollationParams& p)
{
    const auto t = p.totals[p.entries - 1];
    encoding::FlatSizes sizes{t.entities, t.tuples, t.args, t.targets};
    const u64 limit = p.index64 ? 0x7fffffffffffffffull : 0x7fffffffull;
    if (sizes.entities > limit || sizes.tuples > limit || sizes.args > limit || sizes.targets > limit ||
        p.graphs >= limit || p.graphs * p.relations >= limit)
        sizes.overflow |= encoding::k_overflow_index;
    if (sizes.entities > p.output.capacities.entities || sizes.args > p.output.capacities.args ||
        sizes.targets > p.output.capacities.targets)
        sizes.overflow |= encoding::k_overflow_capacity;
    if (bad_source(p)) sizes.overflow |= encoding::k_overflow_input;
    return sizes;
}
__global__ void sizes_kernel(CollationParams p) { *p.output.sizes = sizes_of(p); }

template<class T> __device__ void zero(T* dst, u64 begin, u64 end)
{
    if (dst)
        for (u64 j = begin + u64{blockIdx.x} * blockDim.x + threadIdx.x; j < end; j += u64{gridDim.x} * blockDim.x)
            dst[j] = T{0};
}
template<class I> __global__ void fill_kernel(CollationParams p)
{
    const auto sizes = sizes_of(p);
    const auto& out = p.output.arrays;
    const u64 thread = u64{blockIdx.x} * blockDim.x + threadIdx.x, step = u64{gridDim.x} * blockDim.x;
    const u64 cells = p.graphs * p.relations;
    if (thread == 0)
    {
        *p.output.sizes = sizes;
        if (p.output.graphs) *p.output.graphs = p.graphs;
    }
    if (p.output.zero_unused)
    {
        const bool bad = sizes.overflow;
        zero(out.entity_role, bad ? 0 : sizes.entities, p.output.capacities.entities);
        zero(as<I>(out.entity_ref), bad ? 0 : sizes.entities, p.output.capacities.entities);
        zero(as<I>(out.relation_args), bad ? 0 : sizes.args, p.output.capacities.args);
        zero(as<I>(out.target_entity), bad ? 0 : sizes.targets, p.output.capacities.targets);
        zero(out.target_group, bad ? 0 : sizes.targets, p.output.capacities.targets);
        if (bad)
        {
            const u64 graphs = p.output.capacities.graphs, capacity_cells = graphs * p.relations;
            zero(out.task_ids, 0, graphs); zero(as<I>(out.num_objects), 0, graphs);
            zero(as<I>(out.entity_offsets), 0, graphs + 1); zero(as<I>(out.relation_counts), 0, capacity_cells);
            zero(as<I>(out.relation_offsets), 0, capacity_cells + 1); zero(as<I>(out.relation_arg_offsets), 0, capacity_cells + 1);
            zero(out.goal_complete, 0, graphs); zero(as<I>(out.target_offsets), 0, graphs + 1);
        }
    }
    if (sizes.overflow) return;
    for (u64 g = thread; g <= p.graphs; g += step)
    {
        const auto before = p.totals[g];
        as<I>(out.entity_offsets)[g] = static_cast<I>(before.entities);
        if (out.target_offsets) as<I>(out.target_offsets)[g] = static_cast<I>(before.targets);
        if (g == p.graphs) continue;
        const auto& s = slice_of(p, g);
        const u64 local = g - s.first;
        const auto& a = s.batch.arrays;
        out.task_ids[g] = a.task_ids[local];
        as<I>(out.num_objects)[g] = as<I>(a.num_objects)[local];
        if (out.goal_complete) out.goal_complete[g] = a.goal_complete[local];
        const u64 src_entity = static_cast<u64>(as<I>(a.entity_offsets)[local]);
        const u64 shift = before.entities - src_entity;
        const u64 n = p.totals[g + 1].entities - before.entities;
        if (out.entity_role)
            for (u64 j = 0; j < n; ++j)
            {
                out.entity_role[before.entities + j] = a.entity_role[src_entity + j];
                as<I>(out.entity_ref)[before.entities + j] = as<I>(a.entity_ref)[src_entity + j];
            }
        if (out.target_offsets)
        {
            const u64 first = static_cast<u64>(as<I>(a.target_offsets)[local]);
            const u64 count = p.totals[g + 1].targets - before.targets;
            for (u64 j = 0; j < count; ++j)
            {
                as<I>(out.target_entity)[before.targets + j] = static_cast<I>(as<I>(a.target_entity)[first + j] + shift);
                out.target_group[before.targets + j] = a.target_group[first + j];
            }
        }
    }
    for (u64 c = thread; c <= cells; c += step)
    {
        const auto before = p.totals[p.graphs + c];
        as<I>(out.relation_offsets)[c] = static_cast<I>(before.tuples);
        as<I>(out.relation_arg_offsets)[c] = static_cast<I>(before.args);
        if (c == cells) continue;
        const u64 g = p.relation_major ? c % p.graphs : c / p.relations;
        const u64 r = p.relation_major ? c / p.graphs : c % p.relations;
        const auto& s = slice_of(p, g);
        const auto& a = s.batch.arrays;
        const u64 local = g - s.first;
        const u64 src = p.relation_major ? r * s.batch.graphs + local : local * p.relations + r;
        const auto after = p.totals[p.graphs + c + 1];
        if (out.relation_counts) as<I>(out.relation_counts)[g * p.relations + r] = static_cast<I>(after.tuples - before.tuples);
        const u64 shift = p.totals[g].entities - static_cast<u64>(as<I>(a.entity_offsets)[local]);
        const u64 start = static_cast<u64>(as<I>(a.relation_arg_offsets)[src]);
        for (u64 j = 0; j < after.args - before.args; ++j)
            as<I>(out.relation_args)[before.args + j] = static_cast<I>(as<I>(a.relation_args)[start + j] + shift);
    }
}
}

u64 collation_scan_bytes(u64 entries)
{
    size_t bytes = 0;
    const auto error = cub::DeviceScan::InclusiveScan(nullptr, bytes, static_cast<CollationTotals*>(nullptr), AddTotals{}, length(entries));
    if (error != cudaSuccess)
        throw std::runtime_error("mymyr: encoding: cannot query collation scan scratch");
    return bytes;
}
cudaError_t launch_collation(const CollationParams& p, int phase, cudaStream_t stream)
{
    const u32 grid = blocks(p.entries);
    if (phase != 2)
    {
        if (p.index64) count_kernel<i64><<<grid, k_block, 0, stream>>>(p);
        else count_kernel<i32><<<grid, k_block, 0, stream>>>(p);
        if (const auto e = cudaGetLastError(); e != cudaSuccess) return e;
        size_t bytes = p.temp_bytes;
        if (const auto e = cub::DeviceScan::InclusiveScan(p.temp, bytes, p.totals, AddTotals{}, length(p.entries), stream);
            e != cudaSuccess) return e;
    }
    if (phase == 1) sizes_kernel<<<1, 1, 0, stream>>>(p);
    else if (p.index64) fill_kernel<i64><<<grid, k_block, 0, stream>>>(p);
    else fill_kernel<i32><<<grid, k_block, 0, stream>>>(p);
    return cudaGetLastError();
}
}
