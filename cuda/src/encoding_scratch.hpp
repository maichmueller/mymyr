#pragma once
#include "mymyr/cuda/encoding.hpp"

namespace mymyr::cuda::detail
{
/// The scratch of a batch in one block: the flags and totals, the cell counts and the temporary storage of one inclusive
/// scan over the totals. The totals are four rows of G + 1 entries (the entities, targets, tuples and arguments of the
/// graphs, each row opened by a zero) for the graph-major layout, and for the relation-major layout the entity and
/// target rows and a row of G * R + 1 entries, the tuples of the cells in segment order. The entries are u32 when the
/// conservative sum of the bounds fits u32, u64 otherwise. Cell counts are [G, cells] u32 (graph-major only). The scan
/// is two kernels. Offsets are bytes into the scratch. Scratch bytes: 16 per graph and 4 per cell (graph-major,
/// narrow), plus the scan's storage.
struct ScratchLayout
{
    u64 flags = 0, totals = 0, cell_counts = 0, scan_temp = 0;
    u64 cell_stride = 0;  // entries from the counts of one graph to the next
    u64 cell_entries = 0, scan_length = 0;
    u64 scan_temp_bytes = 0, bytes = 0;
    u64 relations = 0;
    bool wide = false;
    bool relation_major = false;
    [[nodiscard]] friend bool operator==(const ScratchLayout&, const ScratchLayout&) noexcept = default;
    /// The entries of the scanned array for a batch of `graphs` graphs.
    [[nodiscard]] u64 scanned(u64 graphs) const;
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
    EncodingAcquisition context;
    std::shared_ptr<const DeviceBuffer> scratch;
    BatchShape shape;
    ScratchLayout layout;
};
}
