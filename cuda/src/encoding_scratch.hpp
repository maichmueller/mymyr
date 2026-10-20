#pragma once
#include "mymyr/cuda/encoding.hpp"
#include "mymyr/cuda/device_task.hpp"

#include <string>

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
    u64 derived = 0, axiom_views = 0;
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

/// What the device needs to know about a context: the bounds and widths of its graphs, whether they read derived atoms
/// under each goal mode (0: the instance's goal, 1: row goals over the fluent slots, 2: also over the derived slots), and
/// the axiom evaluation of the instances that do. Immutable once built; one per context generation.
struct EncodingFacts
{
    /// One instance's axiom evaluation (none for an instance whose graphs read no derived atoms).
    struct Axioms
    {
        u32 modes = 0;          // bit m: its graphs read derived atoms under goal mode m
        std::string refusal;    // nonempty: the device cannot evaluate its axioms
        u32 ow = 0;             // its object-bitset words
        u64 view_words = 0;     // the words of its per-row view
        bool fc = false;        // some axiom body forward checks on the device
        std::vector<u64> reads; // lifted::Derived::reads
        DeviceTaskPtr task;     // uploaded with the generation
    };
    GraphBounds bounds;
    std::vector<encoding::detail::GraphWidth> widths;
    std::array<bool, 3> reads_derived{};
    u32 derived_words = 0;
    u64 axiom_view_words = 0;  // the widest view of an evaluated instance
    std::vector<Axioms> axioms;  // [instances], or empty when no graph reads derived atoms
    /// Uploaded with the generation: the instances' lifted::EncodingAxiomInstance records and their reads, and per goal
    /// mode the evaluator launches (object-bitset width classes, with forward checking where a class has it).
    std::shared_ptr<DeviceBuffer> axiom_block;
    std::array<std::vector<u32>, 3> class_ow;
    std::array<std::vector<u8>, 3> class_fc;
};

/// The scratch of a batch: scratch_layout(shape, facts.bounds, ...), followed, when an instance's graphs read derived
/// atoms, by the derived rows and views of an evaluation of its axioms (ScratchLayout::derived, ::axiom_views). An
/// encode that passes its derived words needs only the part before ScratchLayout::derived.
[[nodiscard]] ScratchLayout encoding_scratch_layout(const BatchShape& shape, const EncodingFacts& facts,
                                                    encoding::Layout layout, u64 relations);
/// The bytes an encode needs: the whole layout, or with supplied derived words only the part before the derived rows.
[[nodiscard]] u64 required_bytes(const ScratchLayout& layout, bool evaluate) noexcept;

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
