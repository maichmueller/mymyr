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
    /// Zero the variable arrays' unused tails, and all output arrays on overflow, in the fill kernel.
    bool zero_unused = false;
    /// Optional device scalar receiving the number of graphs (the live ones of a dynamically sized input). The per-graph
    /// and per-cell arrays past the live graphs of such an input are zero up to the graph capacity.
    u64* graphs = nullptr;
};
/// Device expansion labels. The offsets delimit one canonical (schema, binding) run per parent row; capacity bounds
/// every label array, including unused capacity tails. Overflow is a device scalar (nonzero: expansion overflowed).
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

/// A depth-one horizon in device memory. Every edge starts at node zero; deeper views are invalid input. Node and
/// edge offsets start at zero and delimit disjoint graph slices. Scratch holds horizon_scratch_words() uint32 words,
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
};

/// Canonical edges and kinship storage for device depth-one horizons. Cousin lists are empty at this depth.
MYMYR_HD constexpr u64 horizon_scratch_words(u64 graphs, u64 nodes, u64 edges)
{
    return 6 * graphs + 8 * nodes + 5 * edges;
}

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
    /// cost; it takes no launch order, and its rows carry no subgoal layers and its schema no history steps.
    const u64* graphs_device = nullptr;
    const u64* row_start = nullptr;
    u64 row_capacity = 0;
    /// Optional device flag: nonzero says rows were dropped before this encode (an append to a full ring), and the batch
    /// reports k_overflow_capacity.
    const u64* dropped = nullptr;
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
    OutputArrays stored;
    u32 max_goal_level = 0;
    bool history = false;
};

}

namespace mymyr::cuda
{
/// Caller-owned output of a depth-one builder. Nodes hold graphs + action capacity rows, node_offsets graphs + 1,
/// and source/destination action capacity entries. The edge labels and offsets remain those of the expansion.
struct DeviceHorizonBuild
{
    u64* nodes = nullptr;
    i32* node_offsets = nullptr;
    i32* src = nullptr;
    i32* dst = nullptr;
    bool merge = true;
};
}
