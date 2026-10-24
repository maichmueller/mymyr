#pragma once
// Pointer-and-size views shared by the host API and device kernels.
#include "mymyr/encoding/buffers.hpp"
#include "mymyr/encoding/layers.hpp"
#include "mymyr/encoding/options.hpp"

namespace mymyr::cuda
{
/// Which arrays a batch stores (see encoding/encoder.hpp). entity_role and entity_ref come together, as do the three
/// target arrays.
struct OutputArrays
{
    bool entity_arrays = true;
    bool counts = true;
    bool goal_complete = false;
    bool targets = false;
    bool history = false;  // history_offsets, history_entity and history_dt
    bool numeric = false;
    encoding::NumericDtype numeric_dtype = encoding::NumericDtype::F32;
    bool relations = true;
    bool simple = false;
    bool shared_statics = false;
    u32 static_relations = 0;
    [[nodiscard]] friend bool operator==(const OutputArrays&, const OutputArrays&) noexcept = default;
};
/// Where an encoding writes: the arrays of a batch (the pointers of arrays it does not store are null), the capacities
/// of the variable-size arrays, and the device address of the FlatSizes that receives the true sizes. The arrays hold at
/// least the capacities' entries, the others the entries their sizes follow from (encoding::FlatBuffers).
struct DeviceOutput
{
    encoding::FlatBuffers arrays;
    encoding::BatchShape capacities;
    encoding::FlatSizes* sizes = nullptr;
    /// Temporary relation storage for Simple-only filling: [graphs * relations + 1] offsets and argument offsets,
    /// and [capacities.args] arguments, all of the schema's index type. Null when relation arrays are stored.
    void* scratch_relation_offsets = nullptr;
    void* scratch_relation_arg_offsets = nullptr;
    void* scratch_relation_args = nullptr;
    /// Zero the variable arrays' unused tails, and all output arrays on overflow, in the fill kernel.
    bool zero_unused = false;
    /// Optional device scalar receiving the number of graphs (the live ones of a dynamically sized input). The per-graph
    /// and per-cell arrays past the live graphs of such an input are zero up to the graph capacity.
    u64* graphs = nullptr;
    /// Optional device scalar receiving the number of numeric values, including on count-only calls.
    u64* numeric_size = nullptr;
    /// Shared-static output capacities and device counts. Required exactly for shared-static schemas; the counts
    /// remain on the device. FlatSizes::overflow also reports a static capacity or index overflow.
    encoding::StaticShape static_capacities{};
    encoding::StaticSizes* static_sizes = nullptr;
};
/// Device expansion labels. The offsets delimit one canonical (schema, binding) run per parent row; capacity bounds
/// every label array, including unused capacity tails. Overflow is a device uint64 status word: bit 0 reports
/// expansion capacity overflow, bit 1 reports invalid input.
struct DeviceActionInput
{
    const i32* offsets = nullptr;  // [graphs + 1]
    const i32* schema = nullptr;   // [capacity]
    const i32* binding = nullptr;  // [capacity, label_width]
    const i32* parent = nullptr;   // optional [capacity]
    u64 capacity = 0;
    u32 label_width = 0;
    const u64* overflow = nullptr;
};

/// A horizon in device memory. Edge endpoints are graph-local node indices. Node and edge offsets start at zero
/// and delimit disjoint graph slices. Scratch holds horizon_scratch_words() uint32 words,
/// remains alive and unchanged between count and fill, and belongs to this call (including captured replays).
struct DeviceHorizonInput
{
    const i32* node_offsets = nullptr;  // [graphs + 1], strictly ascending
    const i32* edge_offsets = nullptr;  // [graphs + 1], nondecreasing
    const i32* src = nullptr;
    const i32* dst = nullptr;
    const i32* schema = nullptr;
    const i32* binding = nullptr;
    u64 nodes = 0, edges = 0;           // capacities of the node rows and edge arrays
    u32 label_width = 0;
    u32* scratch = nullptr;
    u64 scratch_words = 0;
    /// Optional device status of the expansion that produced the horizon (DeviceExpander::expand_async): bit 0 says
    /// its successors exceed its capacity, and the batch reports k_overflow_capacity instead of the invalid node
    /// offsets that follow from it; bit 1 says a parent row is invalid, and the batch reports k_overflow_input.
    const u32* expansion_status = nullptr;
};

/// Canonical edges and kinship storage. Cousin rows are gathered on demand so scratch is linear in the input size.
MYMYR_HD constexpr u64 horizon_scratch_words(u64 graphs, u64 nodes, u64 edges)
{
    return 6 * graphs + 8 * nodes + 5 * edges;
}

/// Recorded transitions of a deduplicated state space. The CSR and labels are immutable device arrays.
struct DeviceHorizonSpace
{
    const u64* rows = nullptr;
    const u64* offsets = nullptr;
    const u32* targets = nullptr;
    const u32* schemas = nullptr;
    const u32* bindings = nullptr;
    u32 states = 0, words = 0, label_width = 0;
    u64 edges = 0;
};

/// Output of a breadth-first traversal of recorded state ids, with compact graph slices and graph-local endpoints.
/// Storage holds graphs * min(limit or states, states) node rows and graphs * edges edge rows.
/// Scratch holds space_horizon_scratch_bytes(states, edges, graphs) bytes at a 256-byte aligned address.
/// The builder reads roots[graphs], writes offsets[graphs + 1], and keeps at most limit nodes per graph (0: states).
/// A limit reached with truncate=false or an invalid root writes negative offsets; truncated[graphs] records limits.
struct DeviceSpaceHorizonBuild
{
    const i32* roots = nullptr;
    u32 graphs = 0, depth = 1, limit = 0;
    bool truncate = false;
    u64* nodes = nullptr;
    i32* node_offsets = nullptr;
    i32* edge_offsets = nullptr;
    i32* src = nullptr;
    i32* dst = nullptr;
    i32* schema = nullptr;
    i32* binding = nullptr;
    void* scratch = nullptr;
    u64 scratch_bytes = 0;
    u8* truncated = nullptr;
};

/// The rows of a batch, in device memory, as encoding::EncodeInput gives them to the host encoder.
struct DeviceEncodeInput
{
    u64 graphs = 0;
    const u64* states = nullptr;    // [graphs, state_stride] the atom words of each row
    u64 state_stride = 0;           // words from one row to the next (at least state_words)
    u32 state_words = 0;            // the words of a row, at least those of every instance in the batch
    const i32* task_ids = nullptr;  // [graphs] the instance of each row; null: instance 0
    // The derived words of each row. Null: evaluate the instances' axioms on this call's stream into scratch.
    // Supplied words take precedence over evaluation.
    const u64* derived = nullptr;
    u64 derived_stride = 0;
    // The goal of each row, replacing its instance's goal, with the layout and rules of encoding::EncodeInput.
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    u64 goal_stride = 0;
    u32 goal_words = 0;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    u64 goal_derived_stride = 0;
    u32 goal_derived_words = 0;
    // The subgoal layers and the history steps of each row, with the layouts and rules of encoding::EncodeInput (device
    // pointers). The schema needs subgoal levels (at least as many as layers) or history; `history.dt` [graphs, layers]
    // is required with the masks. A row whose history count lies outside [0, layers], or whose layers or steps are
    // narrower than its instance or set a bit it does not have, is invalid.
    encoding::LayerInput subgoals{};
    encoding::HistoryInput history{};
    DeviceActionInput actions{};
    DeviceHorizonInput horizon{};
    /// Optional device count, bounded by `graphs`. A circular row view starts at *row_start modulo row_capacity.
    /// Counts and starts are read on the encoder's stream; derived inputs and goal masks use the same row order.
    /// An encode with any of them (or with DeviceOutput::graphs) is sized for `graphs` rows but costs what the live rows
    /// cost; it takes no launch order. Circular views carry no subgoal layers and their schema has no history steps;
    /// linear live views support both.
    const u64* graphs_device = nullptr;
    const u64* row_start = nullptr;
    u64 row_capacity = 0;
    /// Optional device flag: nonzero says rows were dropped before this encode (an append to a full ring), and the batch
    /// reports k_overflow_capacity.
    const u64* dropped = nullptr;
    /// Numeric words following the atom words in each state row, at the table-wide atom width.
    u32 numeric_words = 0;
};

/// One valid batch to concatenate (cuda::collation_batch fills it from its encoder). Pointers name arrays in device
/// memory and remain live through all uses of a collator, including graph replays. Capacity tails are ignored; sizes,
/// when supplied, reports input overflow. table_fingerprint is rl::TaskTable::fingerprint of the encoder's table: the
/// task ids of batches with different ones name different instances. max_goal_level and history are the schema's
/// options: device collation carries no subgoal layers or history steps.
struct DeviceCollationBatch
{
    encoding::FlatBuffers arrays;
    const encoding::FlatSizes* sizes = nullptr;
    u64 graphs = 0, relations = 0, schema_fingerprint = 0, table_fingerprint = 0;
    encoding::Layout layout = encoding::Layout::Graph;
    encoding::IndexType index = encoding::IndexType::I32;
    bool shared_statics = false;
    u32 static_relations = 0, instances = 0;
    encoding::StaticShape static_capacities{};
    const encoding::StaticSizes* static_sizes = nullptr;
    OutputArrays stored;
    u32 max_goal_level = 0;
    bool history = false;
    u32 numeric_relations = 0;
};

}

namespace mymyr::cuda
{
/// Caller-owned output of a depth-one builder. Nodes hold graphs + action capacity rows, node_offsets graphs + 1,
/// and source/destination/schema action capacity entries, binding capacity * label_width entries. Edges are ordered
/// by destination and then by their canonical expansion label; edge offsets remain those of the expansion. Scratch
/// holds depth_one_horizon_scratch_bytes(capacity, merge) bytes, 256-byte aligned, and belongs to the call until its
/// work on the stream is done.
struct DeviceHorizonBuild
{
    u64* nodes = nullptr;
    i32* node_offsets = nullptr;
    i32* src = nullptr;
    i32* dst = nullptr;
    bool merge = true;
    i32* schema = nullptr;
    i32* binding = nullptr;
    void* scratch = nullptr;
    u64 scratch_bytes = 0;
};
}
