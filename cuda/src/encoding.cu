// The kernels of the relational encoder (include/mymyr/cuda/encoding.hpp, encoding_kernels.hpp). They loop over graphs
// and call the per-graph functions of encoding/detail/lanes.hpp, which hold every rule of the encoding. Compiled by nvcc
// as C++23.

#include "encoding_kernels.hpp"

#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

namespace mymyr::cuda
{
void check(cudaError_t error, const char* what);
}

namespace mymyr::cuda::detail
{
namespace ed = mymyr::encoding::detail;
namespace enc = mymyr::encoding;

namespace
{
constexpr u32 k_block = k_encode_block;
constexpr u32 k_warp = 32;

i64 scan_size(u64 n)
{
    if (n > static_cast<u64>(std::numeric_limits<i64>::max()))
        throw std::overflow_error("mymyr: encoding: scan length overflow");
    return static_cast<i64>(n);
}

/// A group of `Lanes` consecutive lanes of a warp, as encoding::detail's group functions use it.
template<u32 Lanes>
struct LaneGroup
{
    static_assert(Lanes == 4 || Lanes == 8 || Lanes == 32);
    u32 mask;

    __device__ LaneGroup()
        : mask(Lanes == 32 ? 0xffffffffu : (((1u << Lanes) - 1u) << (threadIdx.x & 31u & ~(Lanes - 1u))))
    {
    }
    __device__ u32 lane() const { return threadIdx.x & (Lanes - 1u); }
    __device__ u32 size() const { return Lanes; }
    __device__ void sync() const { __syncwarp(mask); }
    template<class X>
    __device__ X sum_of(X x) const
    {
        for (u32 offset = Lanes / 2; offset > 0; offset >>= 1)
            x += __shfl_xor_sync(mask, x, offset, Lanes);
        return x;
    }
    __device__ u32 sum(u32 x) const { return sum_of(x); }
    __device__ u64 sum(u64 x) const { return sum_of(x); }
    __device__ u32 bit_or(u32 x) const
    {
        for (u32 offset = Lanes / 2; offset > 0; offset >>= 1)
            x |= __shfl_xor_sync(mask, x, offset, Lanes);
        return x;
    }
    __device__ u32 exclusive_scan(u32 x, u32& total) const
    {
        const u32 mine = lane();
        u32 inclusive = x;
        for (u32 offset = 1; offset < Lanes; offset <<= 1)
        {
            const u32 below = __shfl_up_sync(mask, inclusive, offset, Lanes);
            if (mine >= offset)
                inclusive += below;
        }
        total = __shfl_sync(mask, inclusive, Lanes - 1, Lanes);
        return inclusive - x;
    }
};

/// The scanned array: u32 entries, u64 when wide.
struct Totals
{
    void* data;
    bool wide;

    __device__ u64 get(u64 i) const
    {
        return wide ? static_cast<const u64*>(data)[i] : static_cast<u64>(static_cast<const u32*>(data)[i]);
    }
    __device__ void set(u64 i, u64 value) const
    {
        if (wide)
            static_cast<u64*>(data)[i] = value;
        else
            static_cast<u32*>(data)[i] = static_cast<u32>(value);
    }
};

/// The instance and row of graph g, or false if the row is invalid for its instance (a task id outside the table, a state
/// row or goal masks narrower than the instance). Nothing of an invalid row is read.
struct RowView
{
    ed::InstanceView v;
    ed::RowInput in;
    u32 instance;
};

__device__ bool row_of(const EncodeParams& p, u64 g, RowView& r)
{
    const i32 id = p.task_ids ? p.task_ids[g] : 0;
    if (ed::row_validity(p.context, id, p.state_words, p.goal_pos != nullptr, p.goal_words,
                         p.goal_derived_pos != nullptr, p.goal_derived_words) != ed::RowValidity::Valid)
        return false;
    r.instance = static_cast<u32>(id);
    r.v = ed::instance_view(p.context, r.instance);
    r.in = ed::RowInput{p.states + g * p.state_stride, nullptr};
    if (p.derived && ed::reads_derived(r.v, p.goal_pos != nullptr, p.goal_derived_pos != nullptr))
        r.in.derived = p.derived + g * p.derived_stride;
    if (p.goal_pos)
    {
        r.in.goal_pos = p.goal_pos + g * p.goal_stride;
        r.in.goal_neg = p.goal_neg + g * p.goal_stride;
    }
    if (p.goal_derived_pos)
    {
        r.in.goal_derived_pos = p.goal_derived_pos + g * p.goal_derived_stride;
        r.in.goal_derived_neg = p.goal_derived_neg + g * p.goal_derived_stride;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ count

template<u32 Lanes>
__global__ void __launch_bounds__(k_block) count_kernel(const EncodeParams p)
{
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 gi = thread / Lanes;
    const Totals totals{p.totals, p.wide};
    const u64 row = p.graphs + 1;
    // every row of the scanned array opens with a zero (the cell row of a relation-major batch is cleared before)
    if (thread == 0)
    {
        totals.set(0, 0);
        totals.set(row, 0);
        if (!p.relation_major)
        {
            totals.set(2 * row, 0);
            totals.set(3 * row, 0);
        }
    }
    if (gi >= p.graphs)
        return;
    const LaneGroup<Lanes> grp;
    const u64 g = p.order ? p.order[gi] : gi;
    if (g >= p.graphs)
    {
        atomicOr(p.flags, 1u);
        return;
    }
    const u32 virtual_nodes = p.context.num_virtual;
    u32* virtual_scratch =
        virtual_nodes ? reinterpret_cast<u32*>(shared_bytes) + (threadIdx.x / Lanes) * virtual_nodes : nullptr;
    RowView r;
    ed::GraphTotals t = ed::empty_totals();
    bool ok = row_of(p, g, r);
    if (p.classify && (ok ? warp_fill_for(r.v, p.relation_major) != p.warp_class : !p.first_class))
        return;
    if (ok)
    {
        if (p.relation_major)
        {
            const u64 cells = p.segments + 1;
            t = ed::count_graph_group_into(
                grp, r.v, r.in,
                [&](u32, const ed::Cell& c, u32 n) { totals.set(cells + ed::cell_segment(true, g, c.relation, p.graphs, r.v.num_relations), n); },
                virtual_scratch);
        }
        else
        {
            u32* counts = p.cell_counts + g * p.cell_stride;
            t = ed::count_graph_group(grp, r.v, r.in, counts, virtual_scratch);
        }
        ok = (t.flags & (ed::k_graph_bad_row | ed::k_graph_bad_goal)) == 0;
    }
    if (grp.lane() != 0)
        return;
    if (!ok)
    {
        atomicOr(p.flags, 1u);
        t = ed::empty_totals();
    }
    totals.set(1 + g, t.entities);
    totals.set(row + 1 + g, t.targets);
    if (!p.relation_major)
    {
        totals.set(2 * row + 1 + g, t.tuples);
        totals.set(3 * row + 1 + g, t.args);
    }
}

// ------------------------------------------------------------------------------------------------ fill

template<class I>
__device__ ed::GraphOut<I> graph_out(const EncodeParams& p)
{
    const enc::FlatBuffers& o = p.out;
    return {p.graphs,
            static_cast<I*>(o.num_objects),
            static_cast<I*>(o.relation_counts),
            static_cast<I*>(o.relation_offsets),
            static_cast<I*>(o.relation_arg_offsets),
            static_cast<I*>(o.relation_args),
            o.entity_role,
            static_cast<I*>(o.entity_ref),
            o.goal_complete,
            static_cast<I*>(o.target_offsets),
            static_cast<I*>(o.target_entity),
            o.target_group};
}

/// The positions of the cells of a relation-major batch from the scanned cell row. The row holds, at entry s, the
/// tuples of the cells before segment s (plus the base of the rows before it). The arguments of a segment are placed by
/// its relation: they start at `first[relation]` plus the relation's arity per tuple before the segment.
struct ScanPositions
{
    Totals totals;
    u64 segments;      // the index of the first entry of the cell row
    u64 base;          // its first entry
    const u64* first;  // [relations] (shared memory): the arguments before the relation, less arity times its tuples before it
    const u32* arity;  // [relations]

    __device__ u64 tuples_before(u64 segment) const { return totals.get(segments + segment) - base; }
    __device__ u32 count(u64 segment) const
    {
        return static_cast<u32>(totals.get(segments + segment + 1) - totals.get(segments + segment));
    }
    __device__ u64 args(u64 segment, u32 relation) const
    {
        return first[relation] + u64{arity[relation]} * tuples_before(segment);
    }
};

/// The argument position table of a relation-major batch, into the shared memory `first` (one entry per relation), and
/// the batch's argument total. The arguments of relation r start at the sum of arity * tuples over the relations before it;
/// the tuples of relation r are the cells r * G .. (r + 1) * G of the cell row. Every thread of the block takes part.
__device__ u64 build_argument_table(const EncodeParams& p, const Totals& totals, u64* first)
{
    using Scan = cub::BlockScan<u64, k_block>;
    __shared__ typename Scan::TempStorage scan_storage;
    const u64 G = p.graphs;
    const u32 R = p.context.num_relations;
    const u64 segments = 2 * (G + 1);
    const u64 base = totals.get(segments);
    u64 carry = 0;
    for (u32 r0 = 0; r0 < R; r0 += k_block)
    {
        const u32 r = r0 + threadIdx.x;
        u64 tuples_before = 0, count = 0, arity = 0;
        if (r < R)
        {
            tuples_before = totals.get(segments + u64{r} * G) - base;
            count = totals.get(segments + u64{r + 1} * G) - base - tuples_before;
            arity = p.context.relation_arity[r];
        }
        u64 before = 0, aggregate = 0;
        Scan(scan_storage).ExclusiveSum(arity * count, before, aggregate);
        if (r < R)
            first[r] = carry + before - arity * tuples_before;
        carry += aggregate;
        __syncthreads();
    }
    return carry;
}

__host__ __device__ constexpr u64 align8(u64 n) { return (n + 7) & ~u64{7}; }

/// The true sizes of the batch from the scanned totals and the flags, with the overflow flags of the capacities, the index
/// type and the rows. A relation-major batch builds its argument table into `first` (shared memory, every thread of the
/// block takes part) and returns the base of its cell row in `base`.
template<bool RelationMajor>
__device__ enc::FlatSizes batch_sizes(const EncodeParams& p, const Totals& totals, u64* first, u64& base)
{
    const u64 G = p.graphs;
    const u32 R = p.context.num_relations;
    const u64 row = G + 1;
    const u64 segments = 2 * row;
    enc::FlatSizes sizes;
    sizes.entities = totals.get(row - 1) - totals.get(0);
    sizes.targets = totals.get(row + G) - totals.get(row);
    base = 0;
    if constexpr (RelationMajor)
    {
        sizes.args = build_argument_table(p, totals, first);
        base = totals.get(segments);
        sizes.tuples = totals.get(segments + G * R) - base;
    }
    else
    {
        sizes.tuples = totals.get(2 * row + G) - totals.get(2 * row);
        sizes.args = totals.get(3 * row + G) - totals.get(3 * row);
    }
    const u64 limit = p.index64 ? u64{0x7fffffffffffffffull} : u64{0x7fffffffu};
    sizes.overflow = p.static_overflow;
    if (sizes.entities > p.capacities.entities || sizes.args > p.capacities.args ||
        sizes.targets > p.capacities.targets)
        sizes.overflow |= enc::k_overflow_capacity;
    if (sizes.entities > limit || sizes.tuples > limit || sizes.args > limit || sizes.targets > limit)
        sizes.overflow |= enc::k_overflow_index;
    if (p.flags[0] != 0)
        sizes.overflow |= enc::k_overflow_input;
    return sizes;
}

/// One block writes the sizes of a counted batch: the second half of the counting, for a caller that sizes its outputs
/// by them.
template<bool RelationMajor>
__global__ void __launch_bounds__(k_block) sizes_kernel(const EncodeParams p)
{
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const Totals totals{p.totals, p.wide};
    u64 base = 0;
    const enc::FlatSizes sizes = batch_sizes<RelationMajor>(p, totals, reinterpret_cast<u64*>(shared_bytes), base);
    if (threadIdx.x == 0)
        *p.sizes = sizes;
}

template<class I, bool Warp, bool RelationMajor>
__global__ void __launch_bounds__(k_block) fill_kernel(const EncodeParams p)
{
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const Totals totals{p.totals, p.wide};
    const u64 G = p.graphs;
    const u32 R = p.context.num_relations;
    const u64 row = G + 1;
    const u64 segments = 2 * row;
    u64* first = reinterpret_cast<u64*>(shared_bytes);
    u32* virtual_base = reinterpret_cast<u32*>(shared_bytes + (RelationMajor ? align8(u64{R} * 8) : 0));

    // the sizes, and whether the batch can be written at all
    u64 base = 0;
    const enc::FlatSizes sizes = batch_sizes<RelationMajor>(p, totals, first, base);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *p.sizes = sizes;
    if (sizes.overflow != 0)
        return;

    const ed::GraphOut<I> out = graph_out<I>(p);
    I* entity_offsets = static_cast<I*>(p.out.entity_offsets);
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        entity_offsets[G] = static_cast<I>(sizes.entities);
        if (out.target_offsets)
            out.target_offsets[G] = static_cast<I>(sizes.targets);
        if constexpr (!RelationMajor)
        {
            out.relation_offsets[G * R] = static_cast<I>(sizes.tuples);
            out.relation_arg_offsets[G * R] = static_cast<I>(sizes.args);
        }
    }
    ScanPositions positions{totals, segments, base, first, p.context.relation_arity};
    if constexpr (RelationMajor)
    {
        // the scanned cell row becomes the two offset arrays, one entry per thread and round
        const u64 cells = G * R;
        for (u64 s = p.first_class ? static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x : cells + 1; s <= cells;
             s += static_cast<u64>(gridDim.x) * blockDim.x)
        {
            const u64 tuples = positions.tuples_before(s);
            out.relation_offsets[s] = static_cast<I>(tuples);
            out.relation_arg_offsets[s] = static_cast<I>(s == cells ? sizes.args : positions.args(s, static_cast<u32>(s / G)));
        }
    }

    // the graphs
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 gi = Warp ? thread / k_warp : thread;
    if (gi >= G)
        return;
    const u64 g = p.order ? p.order[gi] : gi;
    if (g >= p.graphs)
    {
        atomicOr(p.flags, 1u);
        return;
    }
    RowView r;
    if (!row_of(p, g, r))
        return;  // not reached: an invalid row has set the input flag
    if (p.classify && warp_fill_for(r.v, p.relation_major) != p.warp_class)
        return;
    const u32 virtual_nodes = p.context.num_virtual;
    u32* virtual_scratch =
        virtual_nodes ? virtual_base + (Warp ? threadIdx.x / k_warp : threadIdx.x) * virtual_nodes : nullptr;
    const u64 entity_base = totals.get(g);
    const u64 target_base = totals.get(row + g) - totals.get(row);
    const u32 instance = r.instance;
    auto close_graph = [&]
    {
        p.out.task_ids[g] = static_cast<i32>(instance);
        entity_offsets[g] = static_cast<I>(entity_base);
    };
    if constexpr (RelationMajor)
    {
        const ed::GraphTotals graph = ed::totals_from_counts(
            r.v, r.in, [&](u32, const ed::Cell& c) { return positions.count(ed::cell_segment(true, g, c.relation, G, R)); });
        if constexpr (Warp)
        {
            const LaneGroup<k_warp> grp;
            if (grp.lane() == 0)
                close_graph();
            ed::fill_graph_positioned_group<I>(grp, r.v, r.in, g, entity_base, graph, out, positions, target_base,
                                               virtual_scratch, nullptr);
        }
        else
        {
            close_graph();
            ed::fill_graph_positioned<I>(r.v, r.in, g, entity_base, graph, out, positions, target_base, virtual_scratch,
                                         nullptr);
        }
    }
    else
    {
        const u32* counts = p.cell_counts + g * p.cell_stride;
        const ed::GraphTotals graph =
            ed::totals_from_counts(r.v, r.in, [&](u32 i, const ed::Cell&) { return counts[i]; });
        const u64 tuple_base = totals.get(2 * row + g) - totals.get(2 * row);
        const u64 arg_base = totals.get(3 * row + g) - totals.get(3 * row);
        if constexpr (Warp)
        {
            const LaneGroup<k_warp> grp;
            if (grp.lane() == 0)
                close_graph();
            ed::fill_graph_group<I>(grp, r.v, r.in, g, tuple_base, arg_base, entity_base, graph, out, target_base,
                                    virtual_scratch, counts);
        }
        else
        {
            close_graph();
            ed::fill_graph<I>(r.v, r.in, g, tuple_base, arg_base, entity_base, graph, out, target_base, virtual_scratch,
                              virtual_nodes ? counts : nullptr);
        }
    }
}

// ------------------------------------------------------------------------------------------------ launches

u32 blocks_for(u64 threads)
{
    const u64 blocks = threads ? (threads + k_block - 1) / k_block : 1;
    if (blocks > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::overflow_error("mymyr: encoding: the batch has too many graphs to launch");
    return static_cast<u32>(blocks);
}

template<u32 Lanes>
cudaError_t launch_count(const EncodeParams& p, cudaStream_t stream)
{
    count_kernel<Lanes><<<blocks_for(p.graphs * Lanes), k_block, count_shared_bytes(p.context.num_virtual, Lanes), stream>>>(p);
    return cudaPeekAtLastError();
}

template<class I, bool Warp, bool RelationMajor>
cudaError_t launch_fill_of(const EncodeParams& p, cudaStream_t stream)
{
    const u64 threads = Warp ? p.graphs * k_warp : p.graphs;
    // the argument table of a relation-major batch is built by every block, and its offsets copy needs at least one
    // thread per cell round: a grid of the graphs' blocks is enough, the loop strides over the cells
    fill_kernel<I, Warp, RelationMajor><<<blocks_for(threads), k_block,
        fill_shared_bytes(p.context.num_virtual, p.context.num_relations, Warp, RelationMajor), stream>>>(p);
    return cudaPeekAtLastError();
}

template<class I>
cudaError_t launch_fill_index(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    if (shape.warp_fill)
        return p.relation_major ? launch_fill_of<I, true, true>(p, stream) : launch_fill_of<I, true, false>(p, stream);
    return p.relation_major ? launch_fill_of<I, false, true>(p, stream) : launch_fill_of<I, false, false>(p, stream);
}
}  // namespace

u64 totals_scan_temp_bytes(u64 length, bool wide)
{
    size_t bytes = 0;
    const cudaError_t error = wide ? cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u64*>(nullptr), scan_size(length))
                                   : cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u32*>(nullptr), scan_size(length));
    cuda::check(error, "cub::DeviceScan::InclusiveSum (encoding scratch)");
    return bytes;
}

u64 count_shared_bytes(u32 virtual_nodes, u32 lanes)
{
    return u64{k_block / lanes} * virtual_nodes * sizeof(u32);
}

u64 fill_shared_bytes(u32 virtual_nodes, u32 relations, bool warp_fill, bool relation_major)
{
    const u64 graphs_in_flight = warp_fill ? k_block / k_warp : k_block;
    return (relation_major ? align8(u64{relations} * 8) : 0) + graphs_in_flight * virtual_nodes * sizeof(u32);
}

namespace
{
template<class Kernel> u64 static_shared(Kernel kernel)
{
    cudaFuncAttributes attributes{};
    cuda::check(cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes (encoding)");
    return attributes.sharedSizeBytes;
}
template<class I, bool Warp, bool Relation> u64 fill_static()
{
    static const u64 bytes = static_shared(fill_kernel<I, Warp, Relation>);
    return bytes;
}
}
u64 count_shared_required(u32 virtual_nodes, u32 lanes)
{
    static const u64 a = static_shared(count_kernel<4>), b = static_shared(count_kernel<8>),
                     c = static_shared(count_kernel<32>);
    return count_shared_bytes(virtual_nodes, lanes) + (lanes == 4 ? a : lanes == 8 ? b : c);
}
u64 fill_shared_required(u32 virtual_nodes, u32 relations, bool warp, bool relation, bool index64)
{
    const u64 bytes = index64 ? (warp ? (relation ? fill_static<i64, true, true>() : fill_static<i64, true, false>())
                                     : (relation ? fill_static<i64, false, true>() : fill_static<i64, false, false>()))
                             : (warp ? (relation ? fill_static<i32, true, true>() : fill_static<i32, true, false>())
                                     : (relation ? fill_static<i32, false, true>() : fill_static<i32, false, false>()));
    return fill_shared_bytes(virtual_nodes, relations, warp, relation) + bytes;
}

namespace
{
/// The clear, the count kernel and the scan: what both the whole encoding and the counting alone start with.
cudaError_t launch_counts(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    cudaError_t error = cudaMemsetAsync(shape.zero_begin, 0, shape.zero_bytes, stream);
    if (error != cudaSuccess)
        return error;
    bool first = true;
    for (u32 kind = 0; kind < 2; ++kind)
    {
        if (!(shape.fill_mask & (1u << kind)))
            continue;
        EncodeParams row = p;
        row.classify = shape.fill_mask == 3;
        row.warp_class = kind != 0;
        row.first_class = first;
        first = false;
        switch (shape.lanes)
        {
        case 4: error = launch_count<4>(row, stream); break;
        case 8: error = launch_count<8>(row, stream); break;
        case 32: error = launch_count<32>(row, stream); break;
        default: return cudaErrorInvalidValue;
        }
        if (error != cudaSuccess)
            return error;
    }
    size_t temp_bytes = shape.scan_temp_bytes;
    return p.wide ? cub::DeviceScan::InclusiveSum(shape.scan_temp, temp_bytes, static_cast<u64*>(p.totals),
                                                  scan_size(shape.scan_length), stream)
                  : cub::DeviceScan::InclusiveSum(shape.scan_temp, temp_bytes, static_cast<u32*>(p.totals),
                                                  scan_size(shape.scan_length), stream);
}
}  // namespace

cudaError_t launch_encode(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    const cudaError_t error = launch_counts(p, shape, stream);
    if (error != cudaSuccess)
        return error;
    return launch_fill(p, shape, stream);
}

cudaError_t launch_count_sizes(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    const cudaError_t error = launch_counts(p, shape, stream);
    if (error != cudaSuccess)
        return error;
    if (p.relation_major)
        sizes_kernel<true><<<1, k_block, align8(u64{p.context.num_relations} * 8), stream>>>(p);
    else
        sizes_kernel<false><<<1, k_block, 0, stream>>>(p);
    return cudaPeekAtLastError();
}

cudaError_t launch_fill(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    bool first = true;
    for (u32 kind = 0; kind < 2; ++kind)
    {
        if (!(shape.fill_mask & (1u << kind)))
            continue;
        EncodeParams row = p;
        row.classify = shape.fill_mask == 3;
        row.warp_class = kind != 0;
        row.first_class = first;
        first = false;
        LaunchShape launch = shape;
        launch.warp_fill = kind != 0;
        const cudaError_t error = p.index64 ? launch_fill_index<i64>(row, launch, stream)
                                          : launch_fill_index<i32>(row, launch, stream);
        if (error != cudaSuccess)
            return error;
    }
    return cudaSuccess;
}

}  // namespace mymyr::cuda::detail
