// The kernels of the relational encoder (include/mymyr/cuda/encoding.hpp, encoding_kernels.hpp). They loop over graphs
// and call the per-graph functions of encoding/detail/lanes.hpp, which hold every rule of the encoding. Compiled by nvcc
// as C++23.

#include "encoding_kernels.hpp"
#include "mymyr/encoding/detail/horizon.hpp"
#include "live_scan.cuh"

#include <algorithm>
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
/// The blocks of a live fill launch that stay to zero the tails when the live graphs need fewer: enough to keep the
/// memory system busy, few enough that each one's sizes computation costs little.
constexpr u64 k_zero_blocks = 256;

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

/// The parameters of a live launch with the live graph count: min(*graphs_device, graphs).
__device__ EncodeParams live_params(const EncodeParams& params)
{
    EncodeParams p = params;
    if (p.graphs_device)
    {
        const u64 live = *p.graphs_device;
        p.graphs = live < p.graphs ? live : p.graphs;
    }
    p.segments = 2 * (p.graphs + 1);
    const u64 extra = p.relation_major ? p.segments + p.graphs * p.context.num_relations + 1 : 4 * (p.graphs + 1);
    if (p.history_at)
        p.history_at = extra;
    if (p.numeric_at)
        p.numeric_at = extra + (p.history_at ? p.graphs + 1 : 0);
    return p;
}

/// Whether an instance occurs in the logical rows, including a circular live view.
__device__ bool static_instance_present(const EncodeParams& p, u32 instance)
{
    for (u64 g = 0; g < p.graphs; ++g)
    {
        const u64 row = p.row_start ? (*p.row_start + g) % p.row_capacity : g;
        const i32 id = p.task_ids ? p.task_ids[row] : 0;
        if (id == static_cast<i32>(instance))
            return true;
    }
    return false;
}

/// One ordered pass selects distinct instances without reading task ids on the host. Its work is O(I * G).
__global__ void count_statics_kernel(EncodeParams p)
{
    p = live_params(p);
    enc::StaticSizes sizes{};
    for (u32 i = 0; i < p.context.num_instances; ++i)
        if (static_instance_present(p, i))
        {
            const auto t = ed::static_block_totals(ed::instance_view(p.context, i));
            ++sizes.instances;
            sizes.tuples += t.tuples;
            sizes.args += t.args;
        }
    *p.static_totals = sizes;
    *p.static_sizes = sizes;
}

template<class I>
__global__ void fill_statics_kernel(EncodeParams p)
{
    p = live_params(p);
    const auto sizes = *p.static_totals;
    *p.static_sizes = sizes;
    const bool overflow = p.sizes->overflow != 0;
    const auto& a = p.out;
    auto* offsets = static_cast<I*>(a.static_offsets);
    auto* arg_offsets = static_cast<I*>(a.static_arg_offsets);
    auto* args = static_cast<I*>(a.static_args);
    const u64 relations = p.context.num_static;
    if (overflow)
    {
        if (p.zero_unused)
        {
            for (u64 b = 0; b < p.static_capacities.instances; ++b) a.static_task_ids[b] = 0;
            for (u64 s = 0; s <= p.static_capacities.instances * relations; ++s)
                offsets[s] = arg_offsets[s] = 0;
            for (u64 j = 0; j < p.static_capacities.args; ++j) args[j] = 0;
            for (u64 s = 0; s < relations; ++s) static_cast<I*>(a.static_relations)[s] = 0;
        }
        return;
    }
    u64 block = 0, tuples = 0, positions = 0;
    for (u32 i = 0; i < p.context.num_instances; ++i)
        if (static_instance_present(p, i))
        {
            const auto v = ed::instance_view(p.context, i);
            a.static_task_ids[block] = static_cast<i32>(i);
            ed::fill_static_block<I>(v, p.context.static_relations, p.context.num_static, block * relations,
                                     tuples, positions, offsets, arg_offsets, args);
            const auto t = ed::static_block_totals(v);
            tuples += t.tuples;
            positions += t.args;
            ++block;
        }
    offsets[block * relations] = static_cast<I>(tuples);
    arg_offsets[block * relations] = static_cast<I>(positions);
    for (u64 s = 0; s < relations; ++s)
        static_cast<I*>(a.static_relations)[s] = static_cast<I>(p.context.static_relations[s]);
    if (p.zero_unused)
    {
        for (u64 b = block; b < p.static_capacities.instances; ++b) a.static_task_ids[b] = 0;
        for (u64 s = block * relations + 1; s <= p.static_capacities.instances * relations; ++s)
        {
            offsets[s] = static_cast<I>(tuples);
            arg_offsets[s] = static_cast<I>(positions);
        }
        for (u64 j = sizes.args; j < p.static_capacities.args; ++j) args[j] = 0;
    }
}

/// The entries of the scanned array of a batch of `graphs` graphs (detail::ScratchLayout::scanned).
__device__ u64 scanned_entries(const EncodeParams& p, u64 graphs)
{
    return (p.relation_major ? 2 * (graphs + 1) + graphs * p.context.num_relations + 1 : 4 * (graphs + 1)) +
           (p.numeric_at ? graphs + 1 : 0);
}

/// The live scan's input: entry i of the totals, in place. The count kernel of a live launch writes entry i + 1 of the
/// scanned array at i, so the exclusive scan of the first scanned_entries - 1 entries leaves the inclusive scan of the
/// array at the positions the fill kernel reads.
template<class T>
struct LiveTotals
{
    const T* data;
    const u64* graphs_device;
    u64 graphs;
    u32 relations;
    bool relation_major;
    bool numeric;
    __device__ u64 live(u64 n) const
    {
        u64 g = graphs;
        if (graphs_device)
            g = *graphs_device < g ? *graphs_device : g;
        const u64 items = (relation_major ? 2 * (g + 1) + g * relations + 1 : 4 * (g + 1)) +
                          (numeric ? g + 1 : 0) - 1;
        return items < n ? items : n;
    }
    __device__ u64 operator()(u64 i) const { return static_cast<u64>(data[i]); }
};

/// The instance and row of graph g, or false if the row is invalid for its instance (a task id outside the table, a state
/// row, goal masks, subgoal layers or history steps narrower than the instance, a history count outside [0, steps]).
/// Nothing of an invalid row is read. `layers` holds the row's subgoal layers and history (in.layers refers to it iff the
/// row has any); the row view must not be copied once it is made.
struct RowView
{
    ed::InstanceView v;
    ed::RowInput in;
    ed::RowLayers layers;
    u32 instance;
};

template<bool Live, bool Layered>
__device__ bool row_of(const EncodeParams& p, u64 g, RowView& r)
{
    u64 source = g;
    if constexpr (Live)
        if (p.row_start)
        {
            source = *p.row_start % p.row_capacity + g;
            if (source >= p.row_capacity)
                source -= p.row_capacity;
        }
    const i32 id = p.task_ids ? p.task_ids[source] : 0;
    if (ed::row_validity(p.context, id, p.state_words, p.goal_pos != nullptr, p.goal_words,
                         p.goal_derived_pos != nullptr, p.goal_derived_words) != ed::RowValidity::Valid)
        return false;
    r.instance = static_cast<u32>(id);
    r.v = ed::instance_view(p.context, r.instance);
    r.in = ed::RowInput{p.states + source * p.state_stride, nullptr};
    if (p.numeric_words)
        r.in.numeric = r.in.atoms + p.state_words;
    const bool layer_derived = Layered && p.layers.subgoals.layers != 0 && p.layers.subgoals.derived_pos != nullptr;
    if (p.derived && ed::reads_derived(r.v, p.goal_pos != nullptr, p.goal_derived_pos != nullptr, layer_derived))
        r.in.derived = p.derived + (Live && p.derived_logical ? g : source) * p.derived_stride;
    if (p.goal_pos)
    {
        r.in.goal_pos = p.goal_pos + source * p.goal_stride;
        r.in.goal_neg = p.goal_neg + source * p.goal_stride;
    }
    if (p.goal_derived_pos)
    {
        r.in.goal_derived_pos = p.goal_derived_pos + source * p.goal_derived_stride;
        r.in.goal_derived_neg = p.goal_derived_neg + source * p.goal_derived_stride;
    }
    if constexpr (Layered)
    {
        if (ed::layer_validity(r.v, p.layers, source) != ed::LayerValidity::Valid)
            return false;
        if (ed::bind_layers(p.layers, source, r.layers))
            r.in.layers = &r.layers;
    }
    return true;
}

/// The subgoal levels of graph g in the scratch: the count binds the row's levels (bind_levels) into the graph's
/// level_stride words and keeps their LevelPlanes::present in the last of them; the fill binds them from there.
template<class Group>
__device__ __forceinline__ void count_levels(const Group& grp, const EncodeParams& p, RowView& r, u64 g)
{
    u64* storage = p.level_storage + g * p.level_stride;
    ed::bind_levels(grp, r.v, r.in, r.layers, storage);
    if (p.level_stride && grp.lane() == 0)
        storage[p.level_stride - 1] = r.layers.levels.present;
}

__device__ __forceinline__ void fill_levels(const EncodeParams& p, RowView& r, u64 g)
{
    const u64* storage = p.level_storage + g * p.level_stride;
    ed::bind_stored_levels(r.v, r.in, r.layers, storage, p.level_stride ? storage[p.level_stride - 1] : 0);
}

// ------------------------------------------------------------------------------------------------ count

/// The count of the graphs. A live launch writes entry i + 1 of the scanned array at i (LiveTotals), every one of them
/// (it has no clear before it), and clears the tile states of the scan.
template<u32 Lanes, bool Live, bool Layered>
__device__ __forceinline__ void count_graphs(const EncodeParams& p)
{
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 gi = thread / Lanes;
    const Totals totals{p.totals, p.wide};
    const u64 row = p.graphs + 1;
    constexpr u64 shift = Live ? 1 : 0;
    if constexpr (Live)
    {
        if (thread == 0)
        {
            totals.set(row - 1, 0);
            totals.set(2 * row - 1, 0);
            if (!p.relation_major)
                totals.set(3 * row - 1, 0);
            if (p.numeric_at) totals.set(p.numeric_at - 1, 0);
        }
        if (p.first_class)
        {
            const live_scan::State st = live_scan::state_at(static_cast<u64*>(p.scan_state), p.scan_tiles, 1, 0);
            const u64 tiles = live_scan::tiles_for(scanned_entries(p, p.graphs) - 1);
            for (u64 i = thread; i < tiles && i < st.tiles; i += static_cast<u64>(gridDim.x) * blockDim.x)
                st.flag[i] = 0;
        }
    }
    else
    {
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
        if (p.history_at)
            totals.set(p.history_at, 0);
        if (p.numeric_at) totals.set(p.numeric_at, 0);
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
    bool ok = row_of<Live, Layered>(p, g, r);
    if (p.classify && (ok ? fill_lanes_for(p, r.v) != p.fill_class : !p.first_class))
        return;
    if constexpr (Live)
        if (p.relation_major)
        {
            const u32 R = p.context.num_relations;
            for (u32 relation = grp.lane(); relation < R; relation += Lanes)
                totals.set(p.segments + ed::cell_segment(true, g, relation, p.graphs, R), 0);
            grp.sync();
        }
    if (ok)
    {
        if constexpr (Layered)
            count_levels(grp, p, r, g);
        if (p.relation_major)
        {
            const u64 cells = p.segments + 1 - shift;
            t = ed::count_graph_group_into<Layered>(
                grp, r.v, r.in,
                [&](u32, const ed::Cell& c, u32 n) { totals.set(cells + ed::cell_segment(true, g, c.relation, p.graphs, r.v.num_relations), n); },
                virtual_scratch);
        }
        else
        {
            u32* counts = p.cell_counts + g * p.cell_stride;
            t = ed::count_graph_group<Layered>(grp, r.v, r.in, counts, virtual_scratch);
        }
        ok = (t.flags & (ed::k_graph_bad_row | ed::k_graph_bad_goal | ed::k_graph_bad_layers | ed::k_graph_bad_history)) == 0;
    }
    if (grp.lane() != 0)
        return;
    if (!ok)
    {
        atomicOr(p.flags, 1u);
        t = ed::empty_totals();
    }
    totals.set(1 + g - shift, t.entities);
    totals.set(row + 1 + g - shift, t.targets);
    if (!p.relation_major)
    {
        totals.set(2 * row + 1 + g - shift, t.tuples);
        totals.set(3 * row + 1 + g - shift, t.args);
    }
    if (p.history_at)
        totals.set(p.history_at + 1 + g, t.history_steps);
    if (p.numeric_at)
        totals.set(p.numeric_at + 1 + g - shift, ok ? ed::numeric_slot_count(r.v) : 0);
}

/// A live launch carries no subgoal layers or history (the rings refuse them).
template<u32 Lanes, bool Live, bool Layered>
__global__ void __launch_bounds__(k_block) count_kernel(const EncodeParams p)
{
    static_assert(!(Live && Layered));
    if constexpr (Live)
        count_graphs<Lanes, true, false>(live_params(p));
    else
        count_graphs<Lanes, false, Layered>(p);
}

// ------------------------------------------------------------------------------------------------ fill

template<class I>
__device__ ed::GraphOut<I> graph_out(const EncodeParams& p)
{
    const enc::FlatBuffers& o = p.core;
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
            o.target_group,
            static_cast<I*>(o.history_offsets),
            static_cast<I*>(o.history_entity),
            o.history_dt,
            p.numeric_f64 ? nullptr : static_cast<f32*>(o.numeric_values),
            p.numeric_f64 ? static_cast<f64*>(o.numeric_values) : nullptr,
            p.relation_major && p.first_numeric < p.context.num_relations
                ? Totals{p.totals, p.wide}.get(p.segments + u64{p.first_numeric} * p.graphs) -
                      Totals{p.totals, p.wide}.get(p.segments) : 0};
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
    __device__ u64 numeric_before(u64 segment, u64 block) const { return tuples_before(segment) - block; }
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
/// type and the rows (and, live, of rows dropped before the encode). A relation-major batch builds its argument table into
/// `first` (shared memory, every thread of the block takes part) and returns the base of its cell row in `base`.
template<bool RelationMajor, bool Live>
__device__ enc::FlatSizes batch_sizes(const EncodeParams& p, const Totals& totals, u64* first, u64& base)
{
    const u64 G = p.graphs;
    const u32 R = p.context.num_relations;
    const u64 row = G + 1;
    const u64 segments = 2 * row;
    enc::FlatSizes sizes;
    sizes.entities = totals.get(row - 1) - totals.get(0);
    sizes.targets = totals.get(row + G) - totals.get(row);
    sizes.history = p.history_at ? totals.get(p.history_at + G) - totals.get(p.history_at) : 0;
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
    if (p.simple && (sizes.entities > limit - sizes.tuples || sizes.args > limit / 2))
        sizes.overflow |= enc::k_overflow_index;
    if (sizes.entities > p.capacities.entities || sizes.args > p.capacities.args ||
        sizes.targets > p.capacities.targets || sizes.history > p.capacities.history ||
        (p.numeric_at && totals.get(p.numeric_at + G) - totals.get(p.numeric_at) > p.capacities.numeric))
        sizes.overflow |= enc::k_overflow_capacity;
    if (sizes.entities > limit || sizes.tuples > limit || sizes.args > limit || sizes.targets > limit ||
        sizes.history > limit)
        sizes.overflow |= enc::k_overflow_index;
    if (p.shared_statics)
    {
        const auto s = *p.static_totals;
        if (s.instances > p.static_capacities.instances || s.args > p.static_capacities.args)
            sizes.overflow |= enc::k_overflow_capacity;
        if (s.instances * p.context.num_static >= limit || s.tuples > limit || s.args > limit)
            sizes.overflow |= enc::k_overflow_index;
    }
    if (p.flags[0] != 0)
        sizes.overflow |= enc::k_overflow_input;
    if constexpr (Live)
        if (p.rows_overflow && *p.rows_overflow != 0)
            sizes.overflow |= enc::k_overflow_capacity;
    // a horizon built from an expansion that overflowed its capacity has invalid node offsets: the capacity is the cause
    if (p.horizon_mode && p.horizon.expansion_status)
    {
        const u32 status = *p.horizon.expansion_status;
        if (status & 1u)
            sizes.overflow = (sizes.overflow & ~enc::k_overflow_input) | enc::k_overflow_capacity;
        if (status & 2u)
            sizes.overflow |= enc::k_overflow_input;
    }
    if (p.actions.overflow)
    {
        const u64 status = *p.actions.overflow;
        if (status & 1u)
            sizes.overflow = (sizes.overflow & ~enc::k_overflow_input) | enc::k_overflow_capacity;
        if (status & 2u)
            sizes.overflow |= enc::k_overflow_input;
    }
    return sizes;
}

/// One block writes the sizes of a counted batch: the second half of the counting, for a caller that sizes its outputs
/// by them.
template<bool RelationMajor, bool Live>
__device__ __forceinline__ void write_sizes(const EncodeParams& p)
{
    extern __shared__ __align__(8) unsigned char shared_bytes[];
    const Totals totals{p.totals, p.wide};
    u64 base = 0;
    const enc::FlatSizes sizes = batch_sizes<RelationMajor, Live>(p, totals, reinterpret_cast<u64*>(shared_bytes), base);
    if (threadIdx.x == 0)
    {
        *p.sizes = sizes;
        if (p.numeric_size)
            *p.numeric_size = p.numeric_at ? totals.get(p.numeric_at + p.graphs) - totals.get(p.numeric_at) : 0;
        if constexpr (Live)
            if (p.output_graphs)
                *p.output_graphs = p.graphs;
    }
}

template<bool RelationMajor, bool Live>
__global__ void __launch_bounds__(k_block) sizes_kernel(const EncodeParams p)
{
    if constexpr (Live)
        write_sizes<RelationMajor, true>(live_params(p));
    else
        write_sizes<RelationMajor, false>(p);
}

template<class T>
__device__ void zero_tail(T* output, u64 begin, u64 end)
{
    if (!output)
        return;
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (u64 i = begin + thread; i < end; i += static_cast<u64>(gridDim.x) * blockDim.x)
        output[i] = T{0};
}

template<class I>
__device__ void zero_unused(const EncodeParams& p, const enc::FlatSizes& sizes)
{
    const auto& a = p.out;
    const bool overflow = sizes.overflow != 0;
    zero_tail(a.entity_role, overflow ? 0 : sizes.entities, p.capacities.entities);
    zero_tail(static_cast<I*>(a.entity_ref), overflow ? 0 : sizes.entities, p.capacities.entities);
    zero_tail(static_cast<I*>(a.relation_args), overflow ? 0 : sizes.args, p.capacities.args);
    zero_tail(static_cast<I*>(a.target_entity), overflow ? 0 : sizes.targets, p.capacities.targets);
    zero_tail(a.target_group, overflow ? 0 : sizes.targets, p.capacities.targets);
    const Totals totals{p.totals, p.wide};
    const u64 numeric = p.numeric_at ? totals.get(p.numeric_at + p.graphs) - totals.get(p.numeric_at) : 0;
    if (p.numeric_f64) zero_tail(static_cast<f64*>(a.numeric_values), overflow ? 0 : numeric, p.capacities.numeric);
    else zero_tail(static_cast<f32*>(a.numeric_values), overflow ? 0 : numeric, p.capacities.numeric);
    zero_tail(a.node_label, overflow ? 0 : sizes.entities + sizes.tuples, p.capacities.entities + p.capacities.args);
    zero_tail(a.edge_label, overflow ? 0 : 2 * sizes.args, 2 * p.capacities.args);
    zero_tail(static_cast<I*>(a.edge_index), overflow ? 0 : 2 * sizes.args, 2 * p.capacities.args);
    if (a.edge_index)
        zero_tail(static_cast<I*>(a.edge_index) + a.edge_stride, overflow ? 0 : 2 * sizes.args, 2 * p.capacities.args);
    if (overflow)
    {
        const u64 graphs = p.capacities.graphs;
        const u64 cells = graphs * p.context.num_relations;
        zero_tail(a.task_ids, 0, graphs);
        zero_tail(static_cast<I*>(a.num_objects), 0, graphs);
        zero_tail(static_cast<I*>(a.entity_offsets), 0, graphs + 1);
        zero_tail(static_cast<I*>(a.relation_counts), 0, cells);
        zero_tail(static_cast<I*>(a.relation_offsets), 0, cells + 1);
        zero_tail(static_cast<I*>(a.relation_arg_offsets), 0, cells + 1);
        zero_tail(a.goal_complete, 0, graphs);
        zero_tail(static_cast<I*>(a.target_offsets), 0, graphs + 1);
        zero_tail(static_cast<I*>(a.node_offsets), 0, graphs + 1);
        zero_tail(static_cast<I*>(a.edge_offsets), 0, graphs + 1);
    }
}

/// zero_unused of the history arrays, after the fill has written the sizes: a launch of its own, so that the fills of
/// batches without history arrays carry no code for them.
template<class I>
__global__ void __launch_bounds__(k_block) zero_history_kernel(const EncodeParams p)
{
    const enc::FlatSizes sizes = *p.sizes;
    const bool overflow = sizes.overflow != 0;
    zero_tail(static_cast<I*>(p.out.history_entity), overflow ? 0 : sizes.history, p.capacities.history);
    zero_tail(p.out.history_dt, overflow ? 0 : sizes.history, p.capacities.history);
    if (overflow)
        zero_tail(static_cast<I*>(p.out.history_offsets), 0, p.capacities.graphs + 1);
}

/// Zeroes output[begin, end) by `stride` threads, in 16-byte stores between the unaligned ends.
template<class T>
__device__ void zero_range(T* output, u64 begin, u64 end, u64 stride)
{
    static_assert(16 % sizeof(T) == 0);
    if (!output || begin >= end)
        return;
    constexpr u64 per = 16 / sizeof(T);
    T* first = output + begin;
    const u64 n = end - begin;
    const u64 skew = (reinterpret_cast<std::uintptr_t>(first) % 16) / sizeof(T);
    const u64 head = skew ? (per - skew < n ? per - skew : n) : 0;
    const u64 vectors = (n - head) / per;
    const u64 tail = head + vectors * per;
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread < head)
        first[thread] = T{0};
    if (tail + thread < n)
        first[tail + thread] = T{0};
    uint4* words = reinterpret_cast<uint4*>(first + head);
    for (u64 i = thread; i < vectors; i += stride)
        words[i] = make_uint4(0, 0, 0, 0);
}

/// The tails of a live batch, whose graph count is below the graph capacity, by `stride` threads: every per-graph and
/// per-cell array past the live graphs, and with zero_unused the variable arrays past their sizes as zero_unused zeroes
/// them.
template<class I>
__device__ void zero_unused_live(const EncodeParams& p, const enc::FlatSizes& sizes, u64 stride)
{
    const auto& a = p.out;
    const bool overflow = sizes.overflow != 0;
    if (p.zero_unused)
    {
        zero_range(a.entity_role, overflow ? 0 : sizes.entities, p.capacities.entities, stride);
        zero_range(static_cast<I*>(a.entity_ref), overflow ? 0 : sizes.entities, p.capacities.entities, stride);
        zero_range(static_cast<I*>(a.relation_args), overflow ? 0 : sizes.args, p.capacities.args, stride);
        zero_range(static_cast<I*>(a.target_entity), overflow ? 0 : sizes.targets, p.capacities.targets, stride);
        zero_range(a.target_group, overflow ? 0 : sizes.targets, p.capacities.targets, stride);
        const Totals totals{p.totals, p.wide};
        const u64 numeric = p.numeric_at ? totals.get(p.numeric_at + p.graphs) - totals.get(p.numeric_at) : 0;
        if (p.numeric_f64) zero_range(static_cast<f64*>(a.numeric_values), overflow ? 0 : numeric, p.capacities.numeric, stride);
        else zero_range(static_cast<f32*>(a.numeric_values), overflow ? 0 : numeric, p.capacities.numeric, stride);
        zero_range(static_cast<I*>(a.history_entity), overflow ? 0 : sizes.history, p.capacities.history, stride);
        zero_range(a.history_dt, overflow ? 0 : sizes.history, p.capacities.history, stride);
        zero_range(a.node_label, overflow ? 0 : sizes.entities + sizes.tuples,
                   p.capacities.entities + p.capacities.args, stride);
        zero_range(a.edge_label, overflow ? 0 : 2 * sizes.args, 2 * p.capacities.args, stride);
        zero_range(static_cast<I*>(a.edge_index), overflow ? 0 : 2 * sizes.args, 2 * p.capacities.args, stride);
        if (a.edge_index)
            zero_range(static_cast<I*>(a.edge_index) + a.edge_stride, overflow ? 0 : 2 * sizes.args,
                       2 * p.capacities.args, stride);
    }
    const u64 G = overflow ? 0 : p.graphs, R = p.context.num_relations;
    // the offsets of a batch that fits keep their closing entry at G (or G * R)
    const u64 closed = overflow ? 0 : 1;
    const u64 cap = p.capacities.graphs, cells = cap * R;
    zero_range(a.task_ids, G, cap, stride);
    zero_range(static_cast<I*>(a.num_objects), G, cap, stride);
    zero_range(static_cast<I*>(a.entity_offsets), G + closed, cap + 1, stride);
    zero_range(static_cast<I*>(a.relation_counts), G * R, cells, stride);
    zero_range(static_cast<I*>(a.relation_offsets), G * R + closed, cells + 1, stride);
    zero_range(static_cast<I*>(a.relation_arg_offsets), G * R + closed, cells + 1, stride);
    zero_range(a.goal_complete, G, cap, stride);
    zero_range(static_cast<I*>(a.target_offsets), G + closed, cap + 1, stride);
    zero_range(static_cast<I*>(a.history_offsets), G + closed, cap + 1, stride);
    zero_range(static_cast<I*>(a.node_offsets), G + closed, cap + 1, stride);
    zero_range(static_cast<I*>(a.edge_offsets), G + closed, cap + 1, stride);
}

/// The incidence graph is emitted by the group that filled its relation arguments. Prefix differences of the
/// relation-major scan give the graph-major tuple and argument bases without another scan or inter-block dependency.
template<class I, bool RelationMajor, bool Layered, class Group>
__device__ void fill_simple(const Group& grp, const EncodeParams& p, const RowView& r, u64 g,
                            const ed::GraphTotals& totals, const ed::GraphOut<I>& core,
                            const ScanPositions& positions, u64 entity_base, u64 tuple_base, u64 arg_base,
                            const u32* counts)
{
    if (!p.simple)
        return;
    grp.sync();
    if constexpr (RelationMajor)
    {
        tuple_base = 0;
        arg_base = 0;
        for (u32 relation = 0; relation < p.context.num_relations; ++relation)
        {
            const u64 start = u64{relation} * p.graphs;
            const u64 before = positions.tuples_before(start + g) - positions.tuples_before(start);
            tuple_base += before;
            arg_base += before * p.context.relation_arity[relation];
        }
    }
    const u64 entities = Totals{p.totals, p.wide}.get(g + 1) - entity_base;
    const ed::SimpleGraph graph{entity_base, entities, tuple_base, totals.tuples, arg_base, totals.args};
    I* const edge = static_cast<I*>(p.out.edge_index);
    const ed::SimpleOut<I> simple{static_cast<I*>(p.out.node_offsets), static_cast<I*>(p.out.edge_offsets),
                                 p.out.node_label, edge, edge ? edge + p.out.edge_stride : nullptr, p.out.edge_label};
    if (grp.lane() == 0)
    {
        simple.node_offsets[g] = static_cast<I>(entity_base + tuple_base);
        simple.edge_offsets[g] = static_cast<I>(2 * arg_base);
    }
    ed::simple_fill_entities_group<I>(grp, p.simple_params, simple, core.entity_role, graph);
    const auto cells = ed::cell_list<Layered>(r.v, r.in);
    u64 graph_tuple = 0, graph_arg = 0;
    for (u32 i = 0; i < cells.count; ++i)
    {
        const auto& cell = cells.cells[i];
        const u64 segment = ed::cell_segment(RelationMajor, g, cell.relation, p.graphs, p.context.num_relations);
        const u32 n = RelationMajor ? positions.count(segment) : counts[i];
        const u64 at = RelationMajor ? positions.args(segment, cell.relation) : arg_base + graph_arg;
        if (n)
            ed::simple_emit_cell_group<I>(grp, p.simple_params, simple, graph, cell.relation, cell.arity, n,
                                           core.relation_args + at, graph_tuple, graph_arg);
        graph_tuple += n;
        graph_arg += u64{n} * cell.arity;
    }
}

/// The counts and offsets of the R segments of graph g of a graph-major batch from the counts of its cells, by the lanes
/// of a warp together; the values are those fill_graph writes, and the stores of a round fall into the graph's
/// segments, which are contiguous. With `table` (R words of shared memory for the warp) the cells' counts are scattered
/// to their relations and a lane takes a relation: a scan over the relations gives every segment's offsets, three
/// coalesced stores per 32 relations. Without it a lane takes a cell, a scan over the cells places it, and the lane
/// writes the cell's segment and the empty segments up to the next cell.
template<class I>
__device__ void fill_cell_arrays(const EncodeParams& p, const ed::GraphOut<I>& out, u64 g, const ed::CellList& list,
                                 u64 tuple_base, u64 arg_base, u32* table)
{
    const LaneGroup<k_warp> grp;
    const u32 lane = grp.lane();
    const u32 R = p.context.num_relations;
    const u32* counts = p.cell_counts + g * p.cell_stride;
    I* const offsets = out.relation_offsets + g * R;
    I* const arg_offsets = out.relation_arg_offsets + g * R;
    I* const relation_counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    u64 tuples = tuple_base, args = arg_base;
    if (table)
    {
        for (u32 r = lane; r < R; r += k_warp)
            table[r] = 0;
        grp.sync();
        for (u32 i = lane; i < list.count; i += k_warp)
            table[list.cells[i].relation] = counts[i];
        grp.sync();
        for (u32 round = 0; round < R; round += k_warp)
        {
            const u32 r = round + lane;
            const u32 n = r < R ? table[r] : 0u;
            const u32 a = r < R ? n * p.context.relation_arity[r] : 0u;
            u32 round_tuples = 0, round_args = 0;
            const u32 tuples_before = grp.exclusive_scan(n, round_tuples);
            const u32 args_before = grp.exclusive_scan(a, round_args);
            if (r < R)
            {
                offsets[r] = static_cast<I>(tuples + tuples_before);
                arg_offsets[r] = static_cast<I>(args + args_before);
                if (relation_counts)
                    relation_counts[r] = static_cast<I>(n);
            }
            tuples += round_tuples;
            args += round_args;
        }
        grp.sync();  // the table is the next graph's
        return;
    }
    auto empty = [&](u32 from, u32 to, u64 t, u64 a)
    {
        for (u32 r = from; r < to; ++r)
        {
            offsets[r] = static_cast<I>(t);
            arg_offsets[r] = static_cast<I>(a);
            if (relation_counts)
                relation_counts[r] = static_cast<I>(0);
        }
    };
    const u32 first = list.count ? list.cells[0].relation : R;
    for (u32 r = lane; r < first; r += k_warp)
        empty(r, r + 1, tuple_base, arg_base);
    for (u32 round = 0; round < list.count; round += k_warp)
    {
        const u32 i = round + lane;
        const bool cell = i < list.count;
        const u32 n = cell ? counts[i] : 0u;
        const u32 arity = cell ? list.cells[i].arity : 0u;
        u32 round_tuples = 0, round_args = 0;
        const u32 tuples_before = grp.exclusive_scan(n, round_tuples);
        const u32 args_before = grp.exclusive_scan(n * arity, round_args);
        if (cell)
        {
            const u32 r = list.cells[i].relation;
            const u32 next = i + 1 < list.count ? list.cells[i + 1].relation : R;
            const u64 t = tuples + tuples_before, a = args + args_before;
            offsets[r] = static_cast<I>(t);
            arg_offsets[r] = static_cast<I>(a);
            if (relation_counts)
                relation_counts[r] = static_cast<I>(n);
            empty(r + 1, next, t + n, a + u64{n} * arity);
        }
        tuples += round_tuples;
        args += round_args;
    }
}

/// The thread fill of a graph-major batch without subgoal layers or history: a thread writes its graph's entities and
/// arguments, then the warp writes the segment counts and offsets of its graphs (fill_cell_arrays), one graph after the
/// other. Every lane of a warp with a graph of the batch takes part.
template<class I, bool Live>
__device__ void fill_thread_graphs(const EncodeParams& p, const Totals& totals, const ed::GraphOut<I>& out,
                                   I* entity_offsets, u32* virtual_base, const ScanPositions& positions)
{
    u32* table = p.relation_table
                     ? virtual_base + k_block * p.context.num_virtual + (threadIdx.x / k_warp) * p.context.num_relations
                     : nullptr;
    const u64 G = p.graphs;
    const u64 row = G + 1;
    const u64 gi = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u32 lane = threadIdx.x % k_warp;
    if (gi - lane >= G)
        return;  // the whole warp
    u64 g = 0;
    bool mine = false;
    RowView r;
    if (gi < G)
    {
        g = p.order ? p.order[gi] : gi;
        if (g >= G)
            atomicOr(p.flags, 1u);
        // an invalid row is not reached: it has set the input flag
        else if (row_of<Live, false>(p, g, r))
            mine = !p.classify || fill_lanes_for(p, r.v) == p.fill_class;
    }
    const u64 tuple_base = mine ? totals.get(2 * row + g) - totals.get(2 * row) : 0;
    const u64 arg_base = mine ? totals.get(3 * row + g) - totals.get(3 * row) : 0;
    ed::CellList list{nullptr, 0};
    if (mine)
    {
        const u32 virtual_nodes = p.context.num_virtual;
        u32* virtual_scratch = virtual_nodes ? virtual_base + threadIdx.x * virtual_nodes : nullptr;
        const u64 entity_base = totals.get(g);
        const u64 target_base = totals.get(row + g) - totals.get(row);
        const u64 history_base = p.history_at ? totals.get(p.history_at + g) - totals.get(p.history_at) : 0;
        const u64 numeric_base = p.numeric_at ? totals.get(p.numeric_at + g) - totals.get(p.numeric_at) : 0;
        const u32* counts = p.cell_counts + g * p.cell_stride;
        const ed::GraphTotals graph =
            ed::totals_from_counts<false>(r.v, r.in, [&](u32 i, const ed::Cell&) { return counts[i]; });
        p.out.task_ids[g] = static_cast<i32>(r.instance);
        entity_offsets[g] = static_cast<I>(entity_base);
        ed::fill_graph<I, false, false>(r.v, r.in, g, tuple_base, arg_base, entity_base, graph, out, target_base,
                                        virtual_scratch, virtual_nodes ? counts : nullptr, history_base, numeric_base);
        fill_simple<I, false, false>(ed::SoloGroup{}, p, r, g, graph, out, positions, entity_base, tuple_base, arg_base,
                                     counts);
        list = ed::cell_list<false>(r.v, r.in);
    }
    for (u32 filled = __ballot_sync(~0u, mine); filled; filled &= filled - 1)
    {
        const u32 j = __ffs(filled) - 1;
        const ed::CellList cells{reinterpret_cast<const ed::Cell*>(__shfl_sync(
                                     ~0u, reinterpret_cast<unsigned long long>(list.cells), j)),
                                 __shfl_sync(~0u, list.count, j)};
        fill_cell_arrays<I>(p, out, __shfl_sync(~0u, g, j), cells, __shfl_sync(~0u, tuple_base, j),
                            __shfl_sync(~0u, arg_base, j), table);
    }
}

/// The fill of the graphs. `blocks` is the grid the loops over the cells and the tails stride with: the whole grid, or a
/// live launch's active blocks.
template<class I, u32 Lanes, bool RelationMajor, bool Live, bool Layered>
__device__ __forceinline__ void fill_graphs(const EncodeParams& p, u64 blocks)
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
    const enc::FlatSizes sizes = batch_sizes<RelationMajor, Live>(p, totals, first, base);
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *p.sizes = sizes;
        if (p.numeric_size)
            *p.numeric_size = p.numeric_at ? totals.get(p.numeric_at + p.graphs) - totals.get(p.numeric_at) : 0;
        if constexpr (Live)
            if (p.output_graphs)
                *p.output_graphs = p.graphs;
    }
    if constexpr (Live)
    {
        if (p.first_class)
            zero_unused_live<I>(p, sizes, blocks * blockDim.x);
    }
    else if (p.zero_unused && p.first_class)
        zero_unused<I>(p, sizes);
    if (sizes.overflow != 0)
        return;

    const ed::GraphOut<I> out = graph_out<I>(p);
    I* entity_offsets = static_cast<I*>(p.out.entity_offsets);
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        entity_offsets[G] = static_cast<I>(sizes.entities);
        if (p.simple)
        {
            static_cast<I*>(p.out.node_offsets)[G] = static_cast<I>(sizes.entities + sizes.tuples);
            static_cast<I*>(p.out.edge_offsets)[G] = static_cast<I>(2 * sizes.args);
        }
        if (out.target_offsets)
            out.target_offsets[G] = static_cast<I>(sizes.targets);
        if (out.history_offsets)
            out.history_offsets[G] = static_cast<I>(sizes.history);
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
             s += (Live ? blocks : static_cast<u64>(gridDim.x)) * blockDim.x)
        {
            const u64 tuples = positions.tuples_before(s);
            out.relation_offsets[s] = static_cast<I>(tuples);
            out.relation_arg_offsets[s] = static_cast<I>(s == cells ? sizes.args : positions.args(s, static_cast<u32>(s / G)));
        }
    }

    // the graphs
    if constexpr (Lanes == 1 && !RelationMajor && !Layered)
    {
        fill_thread_graphs<I, Live>(p, totals, out, entity_offsets, virtual_base, positions);
        return;
    }
    const u64 thread = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 gi = thread / Lanes;
    if (gi >= G)
        return;
    const u64 g = p.order ? p.order[gi] : gi;
    if (g >= p.graphs)
    {
        atomicOr(p.flags, 1u);
        return;
    }
    RowView r;
    if (!row_of<Live, Layered>(p, g, r))
        return;  // not reached: an invalid row has set the input flag
    if (p.classify && fill_lanes_for(p, r.v) != p.fill_class)
        return;
    const u32 virtual_nodes = p.context.num_virtual;
    u32* virtual_scratch =
        virtual_nodes ? virtual_base + (threadIdx.x / Lanes) * virtual_nodes : nullptr;
    const u64 entity_base = totals.get(g);
    const u64 target_base = totals.get(row + g) - totals.get(row);
    const u64 history_base = p.history_at ? totals.get(p.history_at + g) - totals.get(p.history_at) : 0;
    const u64 numeric_base = p.numeric_at ? totals.get(p.numeric_at + g) - totals.get(p.numeric_at) : 0;
    const u32 instance = r.instance;
    auto close_graph = [&]
    {
        p.out.task_ids[g] = static_cast<i32>(instance);
        entity_offsets[g] = static_cast<I>(entity_base);
    };
    if constexpr (RelationMajor)
    {
        if constexpr (Lanes > 1)
        {
            const LaneGroup<Lanes> grp;
            if constexpr (Layered)
                fill_levels(p, r, g);
            const ed::GraphTotals graph = ed::totals_from_counts<Layered>(
                r.v, r.in, [&](u32, const ed::Cell& c) { return positions.count(ed::cell_segment(true, g, c.relation, G, R)); });
            if (grp.lane() == 0)
                close_graph();
            ed::fill_graph_positioned_group<I, decltype(positions), Layered>(
                grp, r.v, r.in, g, entity_base, graph, out, positions, target_base, virtual_scratch, nullptr, history_base);
            fill_simple<I, true, Layered>(grp, p, r, g, graph, out, positions, entity_base, 0, 0, nullptr);
        }
        else
        {
            if constexpr (Layered)
                fill_levels(p, r, g);
            const ed::GraphTotals graph = ed::totals_from_counts<Layered>(
                r.v, r.in, [&](u32, const ed::Cell& c) { return positions.count(ed::cell_segment(true, g, c.relation, G, R)); });
            close_graph();
            ed::fill_graph_positioned<I, decltype(positions), Layered>(r.v, r.in, g, entity_base, graph, out, positions,
                                                                       target_base, virtual_scratch, nullptr, history_base);
            fill_simple<I, true, Layered>(ed::SoloGroup{}, p, r, g, graph, out, positions, entity_base, 0, 0, nullptr);
        }
    }
    else
    {
        const u32* counts = p.cell_counts + g * p.cell_stride;
        const u64 tuple_base = totals.get(2 * row + g) - totals.get(2 * row);
        const u64 arg_base = totals.get(3 * row + g) - totals.get(3 * row);
        if constexpr (Lanes > 1)
        {
            const LaneGroup<Lanes> grp;
            if constexpr (Layered)
                fill_levels(p, r, g);
            const ed::GraphTotals graph =
                ed::totals_from_counts<Layered>(r.v, r.in, [&](u32 i, const ed::Cell&) { return counts[i]; });
            if (grp.lane() == 0)
                close_graph();
            ed::fill_graph_group<I, Layered>(grp, r.v, r.in, g, tuple_base, arg_base, entity_base, graph, out,
                                             target_base, virtual_scratch, counts, history_base, numeric_base);
            fill_simple<I, false, Layered>(grp, p, r, g, graph, out, positions, entity_base, tuple_base, arg_base, counts);
        }
        else
        {
            if constexpr (Layered)
                fill_levels(p, r, g);
            const ed::GraphTotals graph =
                ed::totals_from_counts<Layered>(r.v, r.in, [&](u32 i, const ed::Cell&) { return counts[i]; });
            close_graph();
            ed::fill_graph<I, Layered>(r.v, r.in, g, tuple_base, arg_base, entity_base, graph, out, target_base,
                                       virtual_scratch, (virtual_nodes || Layered) ? counts : nullptr, history_base, numeric_base);
            fill_simple<I, false, Layered>(ed::SoloGroup{}, p, r, g, graph, out, positions, entity_base, tuple_base, arg_base, counts);
        }
    }
}

/// A live launch is sized for the capacity: the blocks past those of the live graphs (and past the k_zero_blocks that
/// zero the tails) exit at once.
template<class I, u32 Lanes, bool RelationMajor, bool Live, bool Layered>
__global__ void __launch_bounds__(k_block) fill_kernel(const EncodeParams p)
{
    static_assert(!(Live && Layered));
    if constexpr (Live)
    {
        const EncodeParams q = live_params(p);
        const u64 threads = q.graphs * Lanes;
        u64 active = (threads + k_block - 1) / k_block;
        if (active < k_zero_blocks)
            active = k_zero_blocks < gridDim.x ? k_zero_blocks : gridDim.x;
        if (active == 0)
            active = 1;
        if (blockIdx.x >= active)
            return;
        fill_graphs<I, Lanes, RelationMajor, true, false>(q, active);
    }
    else
        fill_graphs<I, Lanes, RelationMajor, false, Layered>(p, gridDim.x);
}

// ------------------------------------------------------------------------------------------------ launches

u32 blocks_for(u64 threads)
{
    const u64 blocks = threads ? (threads + k_block - 1) / k_block : 1;
    if (blocks > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::overflow_error("mymyr: encoding: the batch has too many graphs to launch");
    return static_cast<u32>(blocks);
}

template<u32 Lanes, bool Live>
cudaError_t launch_count(const EncodeParams& p, cudaStream_t stream)
{
    const u64 shared = count_shared_bytes(p.context.num_virtual, Lanes);
    if constexpr (Live)
        count_kernel<Lanes, true, false><<<blocks_for(p.graphs * Lanes), k_block, shared, stream>>>(p);
    else if (p.layered)
        count_kernel<Lanes, false, true><<<blocks_for(p.graphs * Lanes), k_block, shared, stream>>>(p);
    else
        count_kernel<Lanes, false, false><<<blocks_for(p.graphs * Lanes), k_block, shared, stream>>>(p);
    return cudaPeekAtLastError();
}

template<class I, u32 Lanes, bool RelationMajor, bool Live>
cudaError_t launch_fill_of(const EncodeParams& params, cudaStream_t stream)
{
    EncodeParams p = params;
    const u64 threads = p.graphs * Lanes;
    u64 shared = fill_shared_bytes(p.context.num_virtual, p.context.num_relations, Lanes, RelationMajor);
    if constexpr (Lanes == 1 && !RelationMajor)
    {
        // the relation table of fill_cell_arrays when it fits, else its per-cell form
        const u64 table = u64{k_block / k_warp} * p.context.num_relations * sizeof(u32);
        p.relation_table = fill_shared_required(p.context.num_virtual, p.context.num_relations, 1, false,
                                                p.index64) + table <= k_shared_limit;
        if (p.relation_table)
            shared += table;
    }
    // the argument table of a relation-major batch is built by every block, and its offsets copy needs at least one
    // thread per cell round: a grid of the graphs' blocks is enough, the loop strides over the cells
    if constexpr (Live)
        fill_kernel<I, Lanes, RelationMajor, true, false><<<blocks_for(threads), k_block, shared, stream>>>(p);
    else if (p.layered)
        fill_kernel<I, Lanes, RelationMajor, false, true><<<blocks_for(threads), k_block, shared, stream>>>(p);
    else
        fill_kernel<I, Lanes, RelationMajor, false, false><<<blocks_for(threads), k_block, shared, stream>>>(p);
    return cudaPeekAtLastError();
}

template<class I, bool Live>
cudaError_t launch_fill_index(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    switch (shape.fill_lanes)
    {
    case 1: return p.relation_major ? launch_fill_of<I, 1, true, Live>(p, stream) : launch_fill_of<I, 1, false, Live>(p, stream);
    case 8: return p.relation_major ? launch_fill_of<I, 8, true, Live>(p, stream) : launch_fill_of<I, 8, false, Live>(p, stream);
    case 32: return p.relation_major ? launch_fill_of<I, 32, true, Live>(p, stream) : launch_fill_of<I, 32, false, Live>(p, stream);
    default: return cudaErrorInvalidValue;
    }
}

u64 live_scan_bytes(u64 length) { return live_scan::buffer_words(live_scan::tiles_for(length - 1), 1) * sizeof(u64); }
}  // namespace

u64 live_scan_tiles(u64 length) { return live_scan::tiles_for(length ? length - 1 : 0); }

u64 totals_scan_temp_bytes(u64 length, bool wide)
{
    size_t bytes = 0;
    const cudaError_t error = wide ? cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u64*>(nullptr), scan_size(length))
                                   : cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u32*>(nullptr), scan_size(length));
    cuda::check(error, "cub::DeviceScan::InclusiveSum (encoding scratch)");
    return std::max<u64>(bytes, length ? live_scan_bytes(length) : 0);
}

u64 count_shared_bytes(u32 virtual_nodes, u32 lanes)
{
    return u64{k_block / lanes} * virtual_nodes * sizeof(u32);
}

u64 fill_shared_bytes(u32 virtual_nodes, u32 relations, u32 lanes, bool relation_major)
{
    const u64 graphs_in_flight = k_block / lanes;
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
#include "encoding_structured.cuh"

/// The lanes per graph of the action and horizon kernels: one, else the fewest of 4, 8 and 32 for which a block's
/// predicate scratch (one graph per that many lanes), with `fixed` bytes besides, fits the shared memory of a block.
u32 structured_lanes_for(const EncodeParams& p, u64 fixed)
{
    for (const u32 lanes : {1u, 4u, 8u, 32u})
        if (fixed + count_shared_bytes(p.context.num_virtual, lanes) <= k_shared_limit)
            return lanes;
    return k_warp;
}

template<class I, u32 Lanes, bool Relation> u64 fill_static()
{
    static const u64 bytes = std::max({static_shared(fill_kernel<I, Lanes, Relation, false, false>),
                                       static_shared(fill_kernel<I, Lanes, Relation, false, true>),
                                       static_shared(fill_kernel<I, Lanes, Relation, true, false>)});
    return bytes;
}
template<class I, bool Relation> u64 fill_static_of(u32 lanes)
{
    return lanes == 1 ? fill_static<I, 1, Relation>() : lanes == 8 ? fill_static<I, 8, Relation>() : fill_static<I, 32, Relation>();
}
}
u64 count_shared_required(u32 virtual_nodes, u32 lanes)
{
    static const u64 a = std::max({static_shared(count_kernel<4, false, false>), static_shared(count_kernel<4, false, true>),
                                   static_shared(count_kernel<4, true, false>)}),
                     b = std::max({static_shared(count_kernel<8, false, false>), static_shared(count_kernel<8, false, true>),
                                   static_shared(count_kernel<8, true, false>)}),
                     c = std::max({static_shared(count_kernel<32, false, false>), static_shared(count_kernel<32, false, true>),
                                   static_shared(count_kernel<32, true, false>)});
    return count_shared_bytes(virtual_nodes, lanes) + (lanes == 4 ? a : lanes == 8 ? b : c);
}
u64 fill_shared_required(u32 virtual_nodes, u32 relations, u32 lanes, bool relation, bool index64)
{
    const u64 bytes = index64 ? (relation ? fill_static_of<i64, true>(lanes) : fill_static_of<i64, false>(lanes))
                              : (relation ? fill_static_of<i32, true>(lanes) : fill_static_of<i32, false>(lanes));
    return fill_shared_bytes(virtual_nodes, relations, lanes, relation) + bytes;
}

cudaError_t launch_horizon_instances(const EncodeParams& p, i32* instances, cudaStream_t stream)
{
    cudaError_t error = cudaMemsetAsync(instances, 0xff, p.horizon.nodes * sizeof(i32), stream);
    if (error != cudaSuccess)
        return error;
    horizon_instances_kernel<<<std::max<u32>(1, blocks_for(p.graphs)), k_block, 0, stream>>>(p, instances);
    return cudaPeekAtLastError();
}

cudaError_t launch_clear(const LaunchShape& shape, cudaStream_t stream)
{
    return cudaMemsetAsync(shape.zero_begin, 0, shape.zero_bytes, stream);
}

namespace
{
/// The clear, the count kernel and the scan: what both the whole encoding and the counting alone start with.
template<bool Live>
cudaError_t launch_counts(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    cudaError_t error = shape.cleared ? cudaSuccess : launch_clear(shape, stream);
    if (error != cudaSuccess)
        return error;
    const bool structured = p.actions.offsets || p.horizon_mode || (Live && (p.layered || p.history_at));
    if (structured)
    {
        if constexpr (Live)
        {
            // Structured rows use an inclusive scan with zeros outside the live totals.
            error = cudaMemsetAsync(p.totals, 0, shape.scan_length * (p.wide ? 8 : 4), stream);
            if (error != cudaSuccess)
                return error;
        }
        EncodeParams q = p;
        q.structured_lanes = structured_lanes_for(p, 0);
        structured_count_kernel<<<std::max<u32>(1, blocks_for(p.graphs * q.structured_lanes)), k_block,
                                  count_shared_bytes(p.context.num_virtual, q.structured_lanes), stream>>>(q);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return error;
    }
    bool first = true;
    for (u32 kind = 0; !structured && kind < 3; ++kind)
    {
        if (!(shape.fill_mask & (1u << kind)))
            continue;
        EncodeParams row = p;
        row.classify = (shape.fill_mask & (shape.fill_mask - 1)) != 0;
        row.fill_class = k_fill_classes[kind];
        row.first_class = first;
        first = false;
        switch (shape.lanes)
        {
        case 4: error = launch_count<4, Live>(row, stream); break;
        case 8: error = launch_count<8, Live>(row, stream); break;
        case 32: error = launch_count<32, Live>(row, stream); break;
        default: return cudaErrorInvalidValue;
        }
        if (error != cudaSuccess)
            return error;
    }
    if (Live && !structured)
    {
        const live_scan::State st = live_scan::state_at(static_cast<u64*>(p.scan_state), p.scan_tiles, 1, 0);
        if (p.wide)
            return live_scan::launch(LiveTotals<u64>{static_cast<u64*>(p.totals), p.graphs_device, p.graphs,
                                                     p.context.num_relations, p.relation_major, p.numeric_at != 0},
                                     shape.scan_length - 1, static_cast<u64*>(p.totals), st, stream);
        return live_scan::launch(LiveTotals<u32>{static_cast<u32*>(p.totals), p.graphs_device, p.graphs,
                                                 p.context.num_relations, p.relation_major, p.numeric_at != 0},
                                 shape.scan_length - 1, static_cast<u32*>(p.totals), st, stream);
    }
    else
    {
        size_t temp_bytes = shape.scan_temp_bytes;
        return p.wide ? cub::DeviceScan::InclusiveSum(shape.scan_temp, temp_bytes, static_cast<u64*>(p.totals),
                                                      scan_size(shape.scan_length), stream)
                      : cub::DeviceScan::InclusiveSum(shape.scan_temp, temp_bytes, static_cast<u32*>(p.totals),
                                                      scan_size(shape.scan_length), stream);
    }
}

template<bool Live>
cudaError_t launch_fills(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    if (p.actions.offsets || p.horizon_mode || (Live && (p.layered || p.history_at)))
    {
        EncodeParams q = p;
        const u64 table = p.relation_major ? align8(u64{p.context.num_relations} * 8) : 0;
        q.structured_lanes = structured_lanes_for(p, table + (p.index64 ? static_shared(structured_fill_kernel<i64>)
                                                                       : static_shared(structured_fill_kernel<i32>)));
        const u32 blocks = std::max<u32>(1, blocks_for(p.graphs * q.structured_lanes));
        const u64 shared = fill_shared_bytes(p.context.num_virtual, p.context.num_relations, q.structured_lanes,
                                             p.relation_major);
        p.index64 ? structured_fill_kernel<i64><<<blocks, k_block, shared, stream>>>(q)
                  : structured_fill_kernel<i32><<<blocks, k_block, shared, stream>>>(q);
        const cudaError_t error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return error;
        if (p.zero_unused && (p.out.history_offsets || p.out.history_entity || p.out.history_dt))
        {
            const u64 entries = std::max(p.capacities.history, p.capacities.graphs + 1);
            const u32 tail_blocks = std::min<u32>(blocks_for(entries), 4096);
            p.index64 ? zero_history_kernel<i64><<<tail_blocks, k_block, 0, stream>>>(p)
                      : zero_history_kernel<i32><<<tail_blocks, k_block, 0, stream>>>(p);
        }
        return cudaPeekAtLastError();
    }
    bool first = true;
    for (u32 kind = 0; kind < 3; ++kind)
    {
        if (!(shape.fill_mask & (1u << kind)))
            continue;
        EncodeParams row = p;
        row.classify = (shape.fill_mask & (shape.fill_mask - 1)) != 0;
        row.fill_class = k_fill_classes[kind];
        row.first_class = first;
        first = false;
        LaunchShape launch = shape;
        launch.fill_lanes = k_fill_classes[kind];
        const cudaError_t error = p.index64 ? launch_fill_index<i64, Live>(row, launch, stream)
                                          : launch_fill_index<i32, Live>(row, launch, stream);
        if (error != cudaSuccess)
            return error;
    }
    if constexpr (!Live)
        if (p.zero_unused && (p.out.history_offsets || p.out.history_entity || p.out.history_dt))
        {
            const u64 entries = std::max(p.capacities.history, p.capacities.graphs + 1);
            const u32 blocks = std::min<u32>(blocks_for(entries), 4096);
            p.index64 ? zero_history_kernel<i64><<<blocks, k_block, 0, stream>>>(p)
                      : zero_history_kernel<i32><<<blocks, k_block, 0, stream>>>(p);
            return cudaPeekAtLastError();
        }
    return cudaSuccess;
}
}  // namespace

cudaError_t launch_encode(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    const cudaError_t error = shape.live ? launch_counts<true>(p, shape, stream) : launch_counts<false>(p, shape, stream);
    if (error != cudaSuccess)
        return error;
    if (p.shared_statics)
    {
        count_statics_kernel<<<1, 1, 0, stream>>>(p);
        if (const auto e = cudaPeekAtLastError(); e != cudaSuccess) return e;
    }
    return launch_fill(p, shape, stream);
}

cudaError_t launch_count_sizes(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    const cudaError_t error = shape.live ? launch_counts<true>(p, shape, stream) : launch_counts<false>(p, shape, stream);
    if (error != cudaSuccess)
        return error;
    if (p.shared_statics)
    {
        count_statics_kernel<<<1, 1, 0, stream>>>(p);
        if (const auto e = cudaPeekAtLastError(); e != cudaSuccess) return e;
    }
    const u64 shared = p.relation_major ? align8(u64{p.context.num_relations} * 8) : 0;
    if (shape.live)
        p.relation_major ? sizes_kernel<true, true><<<1, k_block, shared, stream>>>(p)
                         : sizes_kernel<false, true><<<1, k_block, shared, stream>>>(p);
    else
        p.relation_major ? sizes_kernel<true, false><<<1, k_block, shared, stream>>>(p)
                         : sizes_kernel<false, false><<<1, k_block, shared, stream>>>(p);
    return cudaPeekAtLastError();
}

cudaError_t launch_fill(const EncodeParams& p, const LaunchShape& shape, cudaStream_t stream)
{
    const auto error = shape.live ? launch_fills<true>(p, shape, stream) : launch_fills<false>(p, shape, stream);
    if (error != cudaSuccess || !p.shared_statics) return error;
    if (p.index64) fill_statics_kernel<i64><<<1, 1, 0, stream>>>(p);
    else fill_statics_kernel<i32><<<1, 1, 0, stream>>>(p);
    return cudaPeekAtLastError();
}

}  // namespace mymyr::cuda::detail
