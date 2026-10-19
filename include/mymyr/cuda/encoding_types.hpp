#pragma once
// Pointer-and-size views shared by the host API and device kernels.
#include "mymyr/encoding/buffers.hpp"
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
    /// Optional device scalar receiving the number of live graphs, including a dynamically sized row stream.
    u64* graphs = nullptr;
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
    /// Optional device count, bounded by `graphs`. A circular row view starts at *row_start modulo row_capacity.
    /// Counts and starts are read on the encoder's stream; derived inputs and goal masks use the same row order.
    const u64* graphs_device = nullptr;
    const u64* row_start = nullptr;
    u64 row_capacity = 0;
};

/// One valid batch to concatenate. Pointers name arrays in device memory and remain live through all uses of a
/// collator, including graph replays. Capacity tails are ignored; sizes, when supplied, reports input overflow.
struct DeviceCollationBatch
{
    encoding::FlatBuffers arrays;
    const encoding::FlatSizes* sizes = nullptr;
    u64 graphs = 0, relations = 0, schema_fingerprint = 0;
    encoding::Layout layout = encoding::Layout::Graph;
    encoding::IndexType index = encoding::IndexType::I32;
    bool shared_statics = false;
    OutputArrays stored;
};

}
