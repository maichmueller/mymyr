#include "encoding_collate.hpp"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mymyr::cuda::detail
{
namespace
{
constexpr u32 k_block = 256;
__host__ __device__ u64 saturating(u64 a, u64 b) { return b > ~u64{0} - a ? ~u64{0} : a + b; }
struct AddGraphs
{
    __host__ __device__ GraphCounts operator()(const GraphCounts& a, const GraphCounts& b) const
    {
        return {saturating(a.entities, b.entities), saturating(a.targets, b.targets), saturating(a.numeric, b.numeric),
                saturating(a.nodes, b.nodes), saturating(a.edges, b.edges)};
    }
};
struct AddCells
{
    __host__ __device__ CellCounts operator()(const CellCounts& a, const CellCounts& b) const
    {
        return {saturating(a.tuples, b.tuples), saturating(a.args, b.args), saturating(a.numeric, b.numeric)};
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
/// The graph and relation of output cell c.
struct Cell
{
    u64 graph, relation;
};
__device__ Cell cell_of(const CollationParams& p, u64 c)
{
    return p.relation_major ? Cell{c % p.graphs, c / p.graphs} : Cell{c / p.relations, c % p.relations};
}
/// The cell of the slice's own batch that output cell `c` copies.
__device__ u64 source_cell(const CollationParams& p, const CollationSlice& s, Cell c)
{
    const u64 local = c.graph - s.first;
    return p.relation_major ? c.relation * s.batch.graphs + local : local * p.relations + c.relation;
}

template<class I> __global__ void __launch_bounds__(k_block) count_kernel(CollationParams p)
{
    const u64 thread = u64{blockIdx.x} * blockDim.x + threadIdx.x, step = u64{gridDim.x} * blockDim.x;
    if (thread == 0)
    {
        u64 bad = 0;
        for (u64 b = 0; b < p.batches; ++b)
            if (p.slices[b].batch.sizes && p.slices[b].batch.sizes->overflow)
                bad = 1;
        *p.bad = bad;
    }
    const u64 entries = (p.graphs > p.cells ? p.graphs : p.cells) + 1;
    for (u64 at = thread; at < entries; at += step)
    {
        if (at <= p.graphs)
        {
            GraphCounts counts;
            if (at)
            {
                const u64 graph = at - 1;
                const auto& s = slice_of(p, graph);
                const u64 g = graph - s.first;
                const auto& a = s.batch.arrays;
                for (u64 r = p.relations - p.numeric_relations; r < p.relations; ++r)
                {
                    const u64 c = p.relation_major ? r * s.batch.graphs + g : g * p.relations + r;
                    counts.numeric += static_cast<u64>(as<I>(a.relation_offsets)[c + 1] - as<I>(a.relation_offsets)[c]);
                }
                counts.entities = static_cast<u64>(as<I>(a.entity_offsets)[g + 1] - as<I>(a.entity_offsets)[g]);
                if (p.simple)
                {
                    counts.nodes = static_cast<u64>(as<I>(a.node_offsets)[g + 1] - as<I>(a.node_offsets)[g]);
                    counts.edges = static_cast<u64>(as<I>(a.edge_offsets)[g + 1] - as<I>(a.edge_offsets)[g]);
                }
                if (a.target_offsets)
                    counts.targets = static_cast<u64>(as<I>(a.target_offsets)[g + 1] - as<I>(a.target_offsets)[g]);
            }
            p.graph_counts[at] = counts;
        }
        if (at <= p.cells)
        {
            CellCounts counts;
            if (at && p.relations_stored)
            {
                const Cell cell = cell_of(p, at - 1);
                const auto& s = slice_of(p, cell.graph);
                const u64 src = source_cell(p, s, cell);
                const auto& a = s.batch.arrays;
                counts.tuples = static_cast<u64>(as<I>(a.relation_offsets)[src + 1] - as<I>(a.relation_offsets)[src]);
                if (cell.relation >= p.relations - p.numeric_relations) counts.numeric = counts.tuples;
                counts.args =
                    static_cast<u64>(as<I>(a.relation_arg_offsets)[src + 1] - as<I>(a.relation_arg_offsets)[src]);
            }
            p.cell_counts[at] = counts;
        }
    }
}
__device__ encoding::FlatSizes sizes_of(const CollationParams& p)
{
    const GraphCounts graphs = p.graph_counts[p.graphs];
    const CellCounts cells = p.cell_counts[p.cells];
    encoding::FlatSizes sizes;
    sizes.entities = graphs.entities;
    sizes.tuples = p.simple ? graphs.nodes - graphs.entities : cells.tuples;
    sizes.args = p.simple ? graphs.edges / 2 : cells.args;
    sizes.targets = graphs.targets;
    const u64 limit = p.index64 ? 0x7fffffffffffffffull : 0x7fffffffull;
    if ((p.simple && (graphs.nodes > limit || graphs.edges > limit)) ||
        sizes.entities > limit || sizes.tuples > limit || sizes.args > limit || sizes.targets > limit ||
        p.graphs >= limit || p.cells >= limit)
        sizes.overflow |= encoding::k_overflow_index;
    if (sizes.entities > p.output.capacities.entities || sizes.args > p.output.capacities.args ||
        sizes.targets > p.output.capacities.targets || cells.numeric > p.output.capacities.numeric)
        sizes.overflow |= encoding::k_overflow_capacity;
    if (*p.bad) sizes.overflow |= encoding::k_overflow_input;
    return sizes;
}
__global__ void sizes_kernel(CollationParams p)
{
    *p.output.sizes = sizes_of(p);
    if (p.output.numeric_size) *p.output.numeric_size = p.cell_counts[p.cells].numeric;
}

template<class T> __device__ void zero(T* dst, u64 begin, u64 end)
{
    if (dst)
        for (u64 j = begin + u64{blockIdx.x} * blockDim.x + threadIdx.x; j < end; j += u64{gridDim.x} * blockDim.x)
            dst[j] = T{0};
}
/// The last index i in [lo, hi] with key(i) <= value: the graph (cell) that holds element `value` of a scanned array.
template<class Key> __device__ u64 holder(const Key& key, u64 lo, u64 hi, u64 value)
{
    while (lo < hi)
    {
        const u64 mid = lo + (hi - lo + 1) / 2;
        if (key(mid) <= value) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}
/// Calls body(element, holder) for every element in [0, total), the elements of a block side by side; the holders of a
/// block's first and last element bracket the search of every element of the block. Uniform over the block.
template<class Key, class Body> __device__ void for_elements(u64 total, u64 last, const Key& key, const Body& body)
{
    __shared__ u64 bracket[2];
    const u64 step = u64{gridDim.x} * blockDim.x;
    for (u64 base = u64{blockIdx.x} * blockDim.x; base < total; base += step)
    {
        const u64 end = total - base < blockDim.x ? total : base + blockDim.x;
        if (threadIdx.x == 0)
        {
            bracket[0] = holder(key, 0, last, base);
            bracket[1] = holder(key, bracket[0], last, end - 1);
        }
        __syncthreads();
        const u64 at = base + threadIdx.x;
        if (at < end)
            body(at, holder(key, bracket[0], bracket[1], at));
        __syncthreads();
    }
}
template<class I> __global__ void __launch_bounds__(k_block) fill_kernel(CollationParams p)
{
    const auto sizes = sizes_of(p);
    const auto& out = p.output.arrays;
    const u64 thread = u64{blockIdx.x} * blockDim.x + threadIdx.x, step = u64{gridDim.x} * blockDim.x;
    if (thread == 0)
    {
        *p.output.sizes = sizes;
        if (p.output.numeric_size) *p.output.numeric_size = p.cell_counts[p.cells].numeric;
        if (p.output.graphs) *p.output.graphs = p.graphs;
    }
    if (p.output.zero_unused)
    {
        // The variable arrays past their sizes and the per-graph and per-cell arrays past the graphs, up to the
        // capacities; everything on overflow.
        const bool bad = sizes.overflow;
        const auto& capacities = p.output.capacities;
        zero(out.entity_role, bad ? 0 : sizes.entities, capacities.entities);
        zero(as<I>(out.entity_ref), bad ? 0 : sizes.entities, capacities.entities);
        zero(as<I>(out.relation_args), bad ? 0 : sizes.args, capacities.args);
        zero(as<I>(out.target_entity), bad ? 0 : sizes.targets, capacities.targets);
        zero(out.target_group, bad ? 0 : sizes.targets, capacities.targets);
        const u64 numeric = p.cell_counts[p.cells].numeric;
        if (p.numeric_f64) zero(as<f64>(out.numeric_values), bad ? 0 : numeric, capacities.numeric);
        else zero(as<f32>(out.numeric_values), bad ? 0 : numeric, capacities.numeric);
        zero(out.node_label, bad ? 0 : sizes.entities + sizes.tuples, capacities.entities + capacities.args);
        zero(out.edge_label, bad ? 0 : 2 * sizes.args, 2 * capacities.args);
        zero(as<I>(out.edge_index), bad ? 0 : 2 * sizes.args, 2 * capacities.args);
        if (out.edge_index)
            zero(as<I>(out.edge_index) + out.edge_stride, bad ? 0 : 2 * sizes.args, 2 * capacities.args);
        const u64 graphs = bad ? 0 : p.graphs, cells = bad ? 0 : p.cells, closing = bad ? 0 : 1;
        const u64 capacity_cells = capacities.graphs * p.relations;
        zero(out.task_ids, graphs, capacities.graphs);
        zero(as<I>(out.num_objects), graphs, capacities.graphs);
        zero(out.goal_complete, graphs, capacities.graphs);
        zero(as<I>(out.entity_offsets), graphs + closing, capacities.graphs + 1);
        zero(as<I>(out.target_offsets), graphs + closing, capacities.graphs + 1);
        zero(as<I>(out.node_offsets), graphs + closing, capacities.graphs + 1);
        zero(as<I>(out.edge_offsets), graphs + closing, capacities.graphs + 1);
        zero(as<I>(out.relation_counts), cells, capacity_cells);
        zero(as<I>(out.relation_offsets), cells + closing, capacity_cells + 1);
        zero(as<I>(out.relation_arg_offsets), cells + closing, capacity_cells + 1);
    }
    if (sizes.overflow) return;
    for (u64 g = thread; g <= p.graphs; g += step)
    {
        const GraphCounts before = p.graph_counts[g];
        as<I>(out.entity_offsets)[g] = static_cast<I>(before.entities);
        if (out.target_offsets) as<I>(out.target_offsets)[g] = static_cast<I>(before.targets);
        if (p.simple)
        {
            as<I>(out.node_offsets)[g] = static_cast<I>(before.nodes);
            as<I>(out.edge_offsets)[g] = static_cast<I>(before.edges);
        }
        if (g == p.graphs) continue;
        const auto& s = slice_of(p, g);
        const u64 local = g - s.first;
        const auto& a = s.batch.arrays;
        out.task_ids[g] = a.task_ids[local];
        as<I>(out.num_objects)[g] = as<I>(a.num_objects)[local];
        if (out.goal_complete) out.goal_complete[g] = a.goal_complete[local];
    }
    for (u64 c = thread; p.relations_stored && c <= p.cells; c += step)
    {
        const CellCounts before = p.cell_counts[c];
        as<I>(out.relation_offsets)[c] = static_cast<I>(before.tuples);
        as<I>(out.relation_arg_offsets)[c] = static_cast<I>(before.args);
        if (c == p.cells || !out.relation_counts) continue;
        const Cell cell = cell_of(p, c);
        as<I>(out.relation_counts)[cell.graph * p.relations + cell.relation] =
            static_cast<I>(p.cell_counts[c + 1].tuples - before.tuples);
    }
    const auto entities_before = [&](u64 g) { return p.graph_counts[g].entities; };
    // The entity id shift of graph g: its first output entity minus its first entity in its own batch.
    const auto shift_of = [&](const CollationSlice& s, u64 g)
    { return p.graph_counts[g].entities - static_cast<u64>(as<I>(s.batch.arrays.entity_offsets)[g - s.first]); };
    if (out.entity_role)
        for_elements(sizes.entities, p.graphs - 1, entities_before,
                     [&](u64 e, u64 g)
                     {
                         const auto& s = slice_of(p, g);
                         const auto& a = s.batch.arrays;
                         const u64 src = static_cast<u64>(as<I>(a.entity_offsets)[g - s.first]) + e - entities_before(g);
                         out.entity_role[e] = a.entity_role[src];
                         as<I>(out.entity_ref)[e] = as<I>(a.entity_ref)[src];
                     });
    if (out.target_offsets)
        for_elements(sizes.targets, p.graphs - 1, [&](u64 g) { return p.graph_counts[g].targets; },
                     [&](u64 t, u64 g)
                     {
                         const auto& s = slice_of(p, g);
                         const auto& a = s.batch.arrays;
                         const u64 src =
                             static_cast<u64>(as<I>(a.target_offsets)[g - s.first]) + t - p.graph_counts[g].targets;
                         as<I>(out.target_entity)[t] = static_cast<I>(as<I>(a.target_entity)[src] + shift_of(s, g));
                         out.target_group[t] = a.target_group[src];
                     });
    if (p.relations_stored)
        for_elements(p.cell_counts[p.cells].numeric, p.cells - 1, [&](u64 c) { return p.cell_counts[c].numeric; },
                     [&](u64 j, u64 c)
                     {
                         const Cell cell = cell_of(p, c);
                         const auto& s = slice_of(p, cell.graph);
                         const auto& a = s.batch.arrays;
                         const u64 local = cell.graph - s.first, first = p.relations - p.numeric_relations;
                         const u64 src_cell = source_cell(p, s, cell);
                         const u64 base = p.relation_major ? first * s.batch.graphs : local * p.relations + first;
                         const u64 src = static_cast<u64>(as<I>(a.relation_offsets)[src_cell] -
                                                          as<I>(a.relation_offsets)[base]) +
                             (p.relation_major ? 0 : p.graph_counts[cell.graph].numeric - p.graph_counts[s.first].numeric) +
                             j - p.cell_counts[c].numeric;
                         if (p.numeric_f64) as<f64>(out.numeric_values)[j] = as<f64>(a.numeric_values)[src];
                         else as<f32>(out.numeric_values)[j] = as<f32>(a.numeric_values)[src];
                     });
    if (p.relations_stored)
        for_elements(sizes.args, p.cells - 1, [&](u64 c) { return p.cell_counts[c].args; },
                 [&](u64 j, u64 c)
                 {
                     const Cell cell = cell_of(p, c);
                     const auto& s = slice_of(p, cell.graph);
                     const auto& a = s.batch.arrays;
                     const u64 src = static_cast<u64>(as<I>(a.relation_arg_offsets)[source_cell(p, s, cell)]) + j -
                                     p.cell_counts[c].args;
                     as<I>(out.relation_args)[j] = static_cast<I>(as<I>(a.relation_args)[src] + shift_of(s, cell.graph));
                 });
    if (p.simple)
    {
        for_elements(sizes.entities + sizes.tuples, p.graphs - 1,
                     [&](u64 g) { return p.graph_counts[g].nodes; },
                     [&](u64 node, u64 g)
                     {
                         const auto& s = slice_of(p, g);
                         const auto& a = s.batch.arrays;
                         const u64 src = static_cast<u64>(as<I>(a.node_offsets)[g - s.first]) + node -
                                         p.graph_counts[g].nodes;
                         out.node_label[node] = a.node_label[src];
                     });
        for_elements(2 * sizes.args, p.graphs - 1,
                     [&](u64 g) { return p.graph_counts[g].edges; },
                     [&](u64 edge, u64 g)
                     {
                         const auto& s = slice_of(p, g);
                         const auto& a = s.batch.arrays;
                         const u64 local = g - s.first;
                         const u64 src = static_cast<u64>(as<I>(a.edge_offsets)[local]) + edge -
                                         p.graph_counts[g].edges;
                         const I shift = static_cast<I>(p.graph_counts[g].nodes) - as<I>(a.node_offsets)[local];
                         as<I>(out.edge_index)[edge] = as<I>(a.edge_index)[src] + shift;
                         as<I>(out.edge_index)[out.edge_stride + edge] = as<I>(a.edge_index)[a.edge_stride + src] + shift;
                         out.edge_label[edge] = a.edge_label[src];
                     });
    }
}
}

u64 collation_scan_bytes(u64 graphs, u64 cells)
{
    size_t graph_bytes = 0, cell_bytes = 0;
    if (cub::DeviceScan::InclusiveScan(nullptr, graph_bytes, static_cast<GraphCounts*>(nullptr), AddGraphs{},
                                       length(graphs + 1)) != cudaSuccess ||
        cub::DeviceScan::InclusiveScan(nullptr, cell_bytes, static_cast<CellCounts*>(nullptr), AddCells{},
                                       length(cells + 1)) != cudaSuccess)
        throw std::runtime_error("mymyr: encoding: cannot query collation scan scratch");
    return graph_bytes > cell_bytes ? graph_bytes : cell_bytes;
}
cudaError_t launch_collation(const CollationParams& p, int phase, cudaStream_t stream)
{
    if (phase != 2)
    {
        const u32 grid = std::min(p.grid, blocks((p.graphs > p.cells ? p.graphs : p.cells) + 1));
        if (p.index64) count_kernel<i64><<<grid, k_block, 0, stream>>>(p);
        else count_kernel<i32><<<grid, k_block, 0, stream>>>(p);
        if (const auto e = cudaGetLastError(); e != cudaSuccess) return e;
        size_t bytes = p.temp_bytes;
        if (const auto e = cub::DeviceScan::InclusiveScan(p.temp, bytes, p.graph_counts, AddGraphs{},
                                                          length(p.graphs + 1), stream);
            e != cudaSuccess)
            return e;
        bytes = p.temp_bytes;
        if (const auto e =
                cub::DeviceScan::InclusiveScan(p.temp, bytes, p.cell_counts, AddCells{}, length(p.cells + 1), stream);
            e != cudaSuccess)
            return e;
    }
    if (phase == 1) sizes_kernel<<<1, 1, 0, stream>>>(p);
    else if (p.index64) fill_kernel<i64><<<p.grid, k_block, 0, stream>>>(p);
    else fill_kernel<i32><<<p.grid, k_block, 0, stream>>>(p);
    return cudaGetLastError();
}
}
