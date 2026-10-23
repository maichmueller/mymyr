#pragma once
// The launches of the relational encoder on the device (include/mymyr/cuda/encoding.hpp builds the parameters). The
// active width classes count and fill around one shared scan, with no allocation or synchronization, so a CUDA graph
// can capture the launches:
//   count  one group of 4, 8 or 32 lanes per graph counts its cells (encoding::detail::count_graph_group), writes the
//          graph's totals and cell counts into the scratch, and flags invalid rows. The levels of a row's subgoal
//          layers are computed by the group first, into the scratch;
//   scan   one inclusive scan over the scratch's totals, two kernels (cub::DeviceScan): the entities, the targets and,
//          graph-major, the tuples and arguments of every graph, relation-major the tuples of every (relation, graph)
//          cell, and with history the history steps of every graph, in this order, each row opened by a zero so that
//          its entries are the exclusive prefix sums plus the sums of the rows before it;
//   fill   a thread per graph (narrow rows) or a warp per graph checks the sizes against the capacities and the index
//          type, and then writes the graph from the positions of the scan (encoding::detail::fill_graph and its group
//          and relation-major forms).
// The launch order of the graphs only decides which thread or warp takes which graph: every position is a prefix sum
// in batch order.

#include "mymyr/core/types.hpp"
#include "mymyr/encoding/buffers.hpp"
#include "mymyr/cuda/encoding_types.hpp"
#include "mymyr/encoding/detail/lanes.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::detail
{
u64 space_horizon_builder_scratch_bytes(u32 states, u64 edges, u32 graphs);
cudaError_t launch_space_horizons(const DeviceHorizonSpace& space, const DeviceSpaceHorizonBuild& output,
                                cudaStream_t stream);

/// Everything the kernels read: device pointers and sizes.
struct EncodeParams
{
    encoding::detail::ContextView context;
    u64 graphs = 0;
    // the rows (device memory)
    const u64* states = nullptr;
    u64 state_stride = 0;  // words from one row to the next
    u32 state_words = 0;
    u32 numeric_words = 0;
    const i32* task_ids = nullptr;
    const u64* derived = nullptr;
    u64 derived_stride = 0;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    u64 goal_stride = 0;
    u32 goal_words = 0;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    u64 goal_derived_stride = 0;
    u32 goal_derived_words = 0;
    // the subgoal layers and history of the rows, and the scratch their levels are computed into
    encoding::detail::LayerSource layers;
    u64* level_storage = nullptr;  // [graphs, level_stride] words
    u64 level_stride = 0;
    DeviceActionInput actions;
    DeviceHorizonInput horizon;
    const i32* schema_relations = nullptr;
    u32* action_storage = nullptr;
    u64 action_stride = 0;
    bool horizon_mode = false, horizon_actions = false, horizon_parents = false, horizon_kin = false;
    bool horizon_cousins = false, horizon_exclude = false, horizon_root_target = false;
    bool layered = false;          // some row may have subgoal layers or history steps (LayerSource::layered)
    bool classify = false, first_class = true;
    u32 fill_class = 1;  // with `classify`: the lanes per graph (fill_lanes_for) of the graphs this launch takes
    const u32* order = nullptr;  // [graphs] the graph each thread or warp takes (null: in batch order)
    // the graph-major thread fill: the dynamic shared memory holds a table of R words per warp after the virtual-node
    // scratch (fill_shared_bytes)
    bool relation_table = false;
    // the batch
    encoding::FlatBuffers out;
    encoding::FlatBuffers core;  // relation arrays may be temporary storage for a Simple-only output
    encoding::detail::SimpleParams simple_params{};
    bool simple = false;
    encoding::FlatSizes* sizes = nullptr;
    encoding::BatchShape capacities;
    encoding::StaticShape static_capacities{};
    encoding::StaticSizes* static_sizes = nullptr;
    encoding::StaticSizes* static_totals = nullptr;
    bool shared_statics = false;
    u32 static_overflow = 0;  // the k_overflow_ flags that follow from the shape alone
    bool index64 = false;
    bool zero_unused = false;
    bool relation_major = false;
    // the scratch
    void* totals = nullptr;  // the scanned array: u32, u64 when wide
    bool wide = false;
    u64 segments = 0;        // relation-major: the index of the first entry of the cell row in `totals`
    u64 numeric_at = 0;  // the optional prefix row of numeric counts
    u32 first_numeric = 0;
    bool numeric_f64 = false;
    u64* numeric_size = nullptr;
    u64 history_at = 0;      // the index of the first entry of the history row in `totals` (0: the schema has no history)
    u32* flags = nullptr;    // flags[0]: a row is invalid
    u32* cell_counts = nullptr;  // graph-major: [graphs, cell_stride] the counts of each graph's cells
    u64 cell_stride = 0;
    // The live launches (LaunchShape::live): the batch is the first min(*graphs_device, graphs) rows, read on the
    // device, of a circular row view (logical row g is source row (*row_start + g) mod row_capacity).
    const u64* graphs_device = nullptr;
    const u64* row_start = nullptr;
    u64 row_capacity = 0;
    bool derived_logical = false;      // `derived` is indexed by the logical row (evaluated words), not the source row
    const u64* rows_overflow = nullptr;  // nonzero: rows were dropped before the encode; the batch reports k_overflow_capacity
    u64* output_graphs = nullptr;      // receives the live graph count
    void* scan_state = nullptr;        // the tile states of the scan over the live totals (live_scan::State, one site)
    u64 scan_tiles = 0;
};

/// How a batch is launched.
struct LaunchShape
{
    u32 lanes = 4;            // lanes per graph of the count kernel: 4, 8 or 32
    u32 fill_mask = 1;         // the fill classes: bit 0 a thread per graph, bit 1 eight lanes, bit 2 a warp
    u32 fill_lanes = 1;        // the lanes per graph of one fill launch: 1, 8 or 32
    u64 scan_length = 0;      // entries of the scanned array
    void* scan_temp = nullptr;
    u64 scan_temp_bytes = 0;
    void* zero_begin = nullptr;  // cleared before the count: the flags and scanned totals
    u64 zero_bytes = 0;
    /// The live launches: the graph count and the rows are read on the device (EncodeParams::graphs_device). Only the
    /// flags are cleared; the count kernel writes every entry of the live totals and clears the scan's tile states, the
    /// scan is a single-pass scan over the live entries, and the fill kernel's blocks past the live graphs exit at once
    /// beyond a fixed number that zero the unused tails. Their cost follows the live rows, not the capacity.
    bool live = false;
    bool cleared = false;  // launch_clear ran already on the stream
};

/// The threads of the launches' blocks.
inline constexpr u32 k_encode_block = 128;
/// The dynamic shared memory a launch may use without opting in.
inline constexpr u64 k_shared_limit = 48 * 1024;

/// The fill classes: the lanes that fill one graph of an instance. One lane while the graph's words, cells and arguments
/// are few; a warp when every graph writes many static arguments, whose rows the lanes share; else eight lanes, four
/// graphs per warp, which keeps the lanes of sparse fluent words and small cells busy.
inline constexpr u32 k_fill_classes[3] = {1, 8, 32};
__host__ __device__ inline u32 fill_lanes_for(const encoding::detail::GraphWidth& width)
{
    if (width.words <= 8u && width.cells <= 128u && width.args <= 128u)
        return 1;
    return width.static_args >= 512u ? 32u : 8u;
}
__host__ __device__ inline u32 fill_lanes_for(const encoding::detail::InstanceView& v)
{
    return fill_lanes_for(encoding::detail::graph_width(v));
}
/// The fill class (an index into k_fill_classes) of a fill with `lanes` lanes per graph.
__host__ __device__ inline u32 fill_class_of(u32 lanes) { return lanes == 1 ? 0u : lanes == 8 ? 1u : 2u; }

/// The temporary storage of the scan of `length` entries, u64 entries when `wide`: CUB's storage, or the tile states of
/// the live scan if they are larger.
[[nodiscard]] u64 totals_scan_temp_bytes(u64 length, bool wide);
/// The tiles of the live scan over `length` entries.
[[nodiscard]] u64 live_scan_tiles(u64 length);

/// The dynamic shared memory of the count and fill kernels: the virtual-node scratch of every graph in flight, and, for
/// the relation-major fill, the argument position of every relation.
[[nodiscard]] u64 count_shared_bytes(u32 virtual_nodes, u32 lanes);
[[nodiscard]] u64 fill_shared_bytes(u32 virtual_nodes, u32 relations, u32 lanes, bool relation_major);
[[nodiscard]] u64 count_shared_required(u32 virtual_nodes, u32 lanes);
[[nodiscard]] u64 fill_shared_required(u32 virtual_nodes, u32 relations, u32 lanes, bool relation_major, bool index64);

/// The depth-one horizon builder (encoding_horizon.cu); its scratch holds horizon_builder_scratch_bytes() bytes.
cudaError_t launch_horizon_builder(const DeviceEncodeInput& parents, const u64* children, u64 child_stride,
                                   const DeviceHorizonBuild& output, cudaStream_t stream);
[[nodiscard]] u64 horizon_builder_scratch_bytes(u64 capacity, bool merge);

/// Maps valid horizon node slices to graph task ids before evaluating the nodes' derived words.
cudaError_t launch_horizon_instances(const EncodeParams& params, i32* instances, cudaStream_t stream);

/// The clear the counting starts with: the flags, and the scanned totals unless the launch is live. A caller that sets
/// the input flag before the count (the axiom evaluation) enqueues it first and sets LaunchShape::cleared.
cudaError_t launch_clear(const LaunchShape& shape, cudaStream_t stream);
/// Enqueues the encoding on `stream`: a clear of the flags (and cell row), the count kernel, the scan and the fill kernel.
cudaError_t launch_encode(const EncodeParams& params, const LaunchShape& shape, cudaStream_t stream);
/// The first half of the encoding: the clear, the count kernel and the scan, then a one-block kernel that writes the true
/// sizes of the batch to `params.sizes` (the capacity flag against `params.capacities`).
cudaError_t launch_count_sizes(const EncodeParams& params, const LaunchShape& shape, cudaStream_t stream);
/// The second half: the fill kernel, over the scratch that launch_count_sizes left for the same rows.
cudaError_t launch_fill(const EncodeParams& params, const LaunchShape& shape, cudaStream_t stream);
}  // namespace mymyr::cuda::detail
