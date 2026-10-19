#pragma once
#include "mymyr/cuda/encoding.hpp"

namespace mymyr::cuda::detail
{
/// The scratch of a batch in one block: the flags and totals, the cell counts, the subgoal levels of the graphs and the
/// temporary storage of one inclusive scan over the totals. The totals are four rows of G + 1 entries (the entities,
/// targets, tuples and arguments of the graphs, each row opened by a zero) for the graph-major layout, and for the
/// relation-major layout the entity and target rows and a row of G * R + 1 entries, the tuples of the cells in segment
/// order; with history, a last row of G + 1 entries holds the history steps of the graphs. The entries are u32 when the
/// conservative sum of the bounds fits u32, u64 otherwise (always with history, whose steps per row the bounds do not
/// limit). Cell counts are [G, cells] u32 (graph-major only). The subgoal levels of a graph (the plain, satisfied and
/// unsatisfied masks of every level, sign and slot kind) take `level_stride` u64 words each. The scan is two kernels.
/// Offsets are bytes into the scratch. Scratch bytes: 16 per graph and 4 per cell (graph-major, narrow), plus the
/// levels and the scan's storage.
struct ScratchLayout
{
    u64 flags = 0, totals = 0, cell_counts = 0, level_storage = 0, scan_temp = 0;
    u64 cell_stride = 0;  // entries from the counts of one graph to the next
    u64 level_stride = 0;  // words from the levels of one graph to the next
    u64 cell_entries = 0, scan_length = 0;
    u64 scan_temp_bytes = 0, bytes = 0;
    u64 relations = 0;
    bool wide = false;
    bool relation_major = false;
    bool history = false;  // the totals have the history row
    [[nodiscard]] friend bool operator==(const ScratchLayout&, const ScratchLayout&) noexcept = default;
    /// The entries of the scanned array for a batch of `graphs` graphs.
    [[nodiscard]] u64 scanned(u64 graphs) const;
    /// The index of the first entry of the history row in the scanned array (0: the totals have none).
    [[nodiscard]] u64 history_at(u64 graphs) const;
};
/// Uses the current CUDA device for the CUB query; throws CudaError or std::overflow_error. The width of the totals
/// follows the bounds only, never the capacities of the outputs.
[[nodiscard]] ScratchLayout scratch_layout(const BatchShape& shape, const GraphBounds& bounds, encoding::Layout layout,
                                           u64 relations = 0);

/// The scratch of a held workspace as device pointers.
struct ScratchView
{
    u32* flags = nullptr;
    void* totals = nullptr;      // u32 (u64 when wide)
    u32* cell_counts = nullptr;  // [G, cell_stride] counts
    u64 cell_stride = 0;
    u64* level_storage = nullptr;  // [G, level_stride] words
    u64 level_stride = 0;
    void* scan_temp = nullptr;
    u64 scan_temp_bytes = 0;
    bool wide = false;
};


[[nodiscard]] ScratchView scratch_view(const ScratchLayout& layout, void* block) noexcept;

struct EncodingFacts
{
    GraphBounds bounds;
    std::vector<encoding::detail::GraphWidth> widths;
    std::array<bool, 3> reads_derived{};
    bool reads_derived_layers = false;  // some instance reads derived words for subgoal layers over derived slots
    u32 derived_words = 0;
};

struct EncodingExecution
{
    const EncodingAcquisition& context;
    void* block;
    u64 bytes;
    const BatchShape& shape;
    const ScratchLayout& layout;
};

struct EncodingWorkspaceState
{
    // Completion is recorded before releasing the last scratch owner.
    std::shared_ptr<const DeviceBuffer> scratch;
    EncodingAcquisition context;
    BatchShape shape;
    ScratchLayout layout;
};
}
