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
    /// Zero the variable arrays' unused tails, and all output arrays on overflow, in the fill kernel.
    bool zero_unused = false;
    /// Optional device scalar receiving the number of graphs (the live ones of a dynamically sized input). The per-graph
    /// and per-cell arrays past the live graphs of such an input are zero up to the graph capacity.
    u64* graphs = nullptr;
    encoding::StaticShape static_capacities{};
    encoding::StaticSizes* static_sizes = nullptr;
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
    u32 static_relations = 0, instances = 0;
    encoding::StaticShape static_capacities{};
    const encoding::StaticSizes* static_sizes = nullptr;
    OutputArrays stored;
    u32 max_goal_level = 0;
    bool history = false;
};

}
