#pragma once
// Relational encoding contexts are immutable device generations. An acquisition holds its generation while work
// is enqueued; releasing it records completion on its stream. Buffers allocate and free on the context's compute
// stream, so a caller stream may be destroyed after its acquisitions have been released.
//
// Acquisitions and CUDA graph capture:
//   - acquire and release outside a capture. Acquiring on a capturing stream throws std::invalid_argument before the
//     stream is touched; a release inside a capture could not order the free after the captured work.
//   - hold the acquisition through the last replay of every graph that captured work with it.
//   - the work and its replays run on the acquisition's stream. Work on another stream is ordered into the
//     acquisition's stream (an event wait there) before the release.
// A held acquisition only exposes pointers, sizes and offsets: using it enqueues no stream waits and records no
// events, so kernels can be launched with it inside a capture.

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/encoding/buffers.hpp"
#include "mymyr/encoding/encoder.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace mymyr::cuda
{
/// A device view and its owning generation, acquired together. Keep the stream alive until this object is released.
/// Acquisitions on the same stream may overlap; callers order work on that stream. Move-only.
class EncodingAcquisition
{
public:
    EncodingAcquisition(encoding::detail::ContextView view, std::shared_ptr<const DeviceBuffer> generation,
                        u64 number, std::shared_ptr<LastUse> last, cudaStream_t stream);
    ~EncodingAcquisition();
    EncodingAcquisition(EncodingAcquisition&& other) noexcept;
    EncodingAcquisition& operator=(EncodingAcquisition&&) = delete;
    EncodingAcquisition(const EncodingAcquisition&) = delete;
    EncodingAcquisition& operator=(const EncodingAcquisition&) = delete;

    /// The stream the acquisition was made on and records its release on.
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }

    encoding::detail::ContextView view;
    std::shared_ptr<const DeviceBuffer> generation;
    u64 number;

private:
    std::shared_ptr<LastUse> m_last;
    cudaStream_t m_stream;
};

/// The host encoding context copied byte for byte onto the device. Refresh and all accessors are thread-safe.
class DeviceEncodingContext
{
public:
    /// Copies on `stream` (nullptr: the compute stream), after allocation on the compute stream.
    /// Returns before the copy completes. The caller may destroy its stream after this call.
    static std::shared_ptr<DeviceEncodingContext> upload(ContextPtr ctx,
        std::shared_ptr<const encoding::EncodingContext> host, cudaStream_t stream = nullptr);
    DeviceEncodingContext(const DeviceEncodingContext&) = delete;
    DeviceEncodingContext& operator=(const DeviceEncodingContext&) = delete;
    ~DeviceEncodingContext();

    /// Uploads an extension as a new generation. Every old array must be a prefix of its new array.
    /// Existing acquisitions and captured graphs keep their old generation. False if no instances were added.
    bool refresh(std::shared_ptr<const encoding::EncodingContext> grown, cudaStream_t stream = nullptr);
    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] std::shared_ptr<const encoding::EncodingContext> host() const;
    [[nodiscard]] u32 num_instances() const;
    [[nodiscard]] u64 bytes() const;
    /// The number of uploads so far: 1 after upload, one more per refresh that copied.
    [[nodiscard]] u64 generation() const;
    /// Atomically obtains the view and generation, waits for the upload, and tracks completion when released.
    /// `use` is the LastUse of the stream that the release records on (nullptr: one shared by the context's
    /// acquisitions on this stream). Throws std::invalid_argument on a capturing stream.
    [[nodiscard]] EncodingAcquisition acquire(cudaStream_t stream = nullptr, std::shared_ptr<LastUse> use = nullptr) const;
    /// Drops the bookkeeping of `stream`, which may be destroyed once its acquisitions are released. The generations'
    /// frees still wait for the work of those acquisitions.
    void release(cudaStream_t stream = nullptr) const;
    [[nodiscard]] std::vector<std::byte> download(cudaStream_t stream = nullptr) const;

private:
    DeviceEncodingContext(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host);
    void copy(const std::shared_ptr<const encoding::EncodingContext>& host, cudaStream_t stream);

    ContextPtr m_ctx;
    std::shared_ptr<const encoding::EncodingContext> m_host;
    std::shared_ptr<DeviceBuffer> m_block;
    encoding::detail::ContextView m_view;
    PinnedLease m_staging;
    Event m_ready;
    u64 m_generation = 0;
    mutable std::unordered_map<cudaStream_t, std::shared_ptr<LastUse>> m_uses;
    mutable std::mutex m_mutex;
};

struct BatchShape
{
    u64 graphs = 0;
    u64 entities = 0;
    u64 args = 0;
    u64 targets = 0;
    [[nodiscard]] friend bool operator==(const BatchShape&, const BatchShape&) noexcept = default;
};

/// Conservative maxima per graph over the instances of the context, covering every lane the device encodes: the state,
/// static and derived cells, the nullary (auxiliary) entity, the goal cells of the instance's goal and of a row's own
/// goal, the goal targets and one entity per predicate virtual node. `words` is the widest row of an instance (state
/// or derived words), `cells` the longest cell list of a graph.
struct GraphBounds
{
    u64 entities = 0, tuples = 0, args = 0, cells = 0, targets = 0, words = 0;
};

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
[[nodiscard]] OutputArrays output_arrays(const encoding::EncodingSchema& schema);

/// Byte offsets in a caller-owned output block; each array begins at a multiple of 64 bytes. An array that the batch
/// does not store has no bytes. The last array is the FlatSizes the encoding writes.
struct OutputLayout
{
    u64 task_ids = 0, num_objects = 0, entity_offsets = 0, entity_role = 0, entity_ref = 0, relation_counts = 0,
        relation_offsets = 0, relation_arg_offsets = 0, relation_args = 0, goal_complete = 0, target_offsets = 0,
        target_entity = 0, target_group = 0, sizes = 0;
    u64 bytes = 0;
    OutputArrays arrays;
};
/// Throws std::overflow_error if a size product, sum or alignment exceeds u64.
[[nodiscard]] OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations,
                                         const OutputArrays& arrays = {});

/// Where an encoding writes: the arrays of a batch (the pointers of arrays it does not store are null), the capacities
/// of the variable-size arrays, and the device address of the FlatSizes that receives the true sizes. The arrays hold at
/// least the capacities' entries, the others the entries their sizes follow from (encoding::FlatBuffers).
struct DeviceOutput
{
    encoding::FlatBuffers arrays;
    encoding::FlatCapacities capacities;
    encoding::FlatSizes* sizes = nullptr;
};
/// The destinations in the device block `block` of output_layout(shape, ...), with the capacities of `shape`
/// (every tuple has at least one argument, so the tuple capacity is the argument capacity).
[[nodiscard]] DeviceOutput output_of(const OutputLayout& layout, const BatchShape& shape, void* block);

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

/// The rows of a batch, in device memory, as encoding::EncodeInput gives them to the host encoder.
struct DeviceEncodeInput
{
    u64 graphs = 0;
    const u64* states = nullptr;    // [graphs, state_stride] the atom words of each row
    u64 state_stride = 0;           // words from one row to the next (at least state_words)
    u32 state_words = 0;            // the words of a row, at least those of every instance in the batch
    const i32* task_ids = nullptr;  // [graphs] the instance of each row; null: instance 0
    // The derived words of each row. Required when a graph reads them (the schema has derived relations, or goal
    // relations that list derived literals); the device does not evaluate axioms.
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
};

enum class FillMode : u8
{
    Auto,  // a thread per graph for rows of at most 8 words, 128 cells and 1024 argument entries, else a warp per graph
    Thread,
    Warp,
};

/// How a batch is launched. None of it changes the output: the launch order only decides which thread or warp takes
/// which graph, every position being a prefix sum in batch order.
struct LaunchConfig
{
    u32 count_lanes = 0;  // lanes per graph of the count kernel: 4, 8 or 32; 0: by the width of the rows
    FillMode fill = FillMode::Auto;
    const u32* order = nullptr;  // device [graphs]: a permutation of the graphs, the graph each thread or warp takes
};

/// A held acquisition: the context generation and the scratch of one stream; outputs belong to the caller. It is the
/// object a capture-time encode takes: encode() on it enqueues kernels on stream() and nothing else. It waits on no
/// stream and records no event.
class EncodingWorkspace
{
public:
    EncodingWorkspace(EncodingAcquisition context, std::shared_ptr<const DeviceBuffer> scratch, BatchShape shape,
                      ScratchLayout layout);
    EncodingWorkspace(EncodingWorkspace&&) noexcept = default;
    EncodingWorkspace& operator=(EncodingWorkspace&&) = delete;
    EncodingWorkspace(const EncodingWorkspace&) = delete;
    EncodingWorkspace& operator=(const EncodingWorkspace&) = delete;

    /// The acquisition's stream: work and replays run on it.
    [[nodiscard]] cudaStream_t stream() const noexcept { return context.stream(); }
    [[nodiscard]] const encoding::detail::ContextView& view() const noexcept { return context.view; }
    [[nodiscard]] ScratchView scratch_view() const noexcept;

    EncodingAcquisition context;
    std::shared_ptr<const DeviceBuffer> scratch;
    BatchShape shape;
    ScratchLayout layout;
};

class DeviceEncoder
{
public:
    /// Throws std::invalid_argument for a schema the device does not encode: action targets.
    /// Each stream has its own scratch.
    DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder);
    DeviceEncoder(const DeviceEncoder&) = delete;
    DeviceEncoder& operator=(const DeviceEncoder&) = delete;

    [[nodiscard]] std::shared_ptr<const encoding::FlatEncoder> encoder() const;
    [[nodiscard]] const encoding::EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const encoding::FlatOptions& options() const noexcept { return m_schema->options(); }
    [[nodiscard]] const DeviceEncodingContext& context() const noexcept { return *m_context; }
    [[nodiscard]] GraphBounds bounds() const;
    [[nodiscard]] BatchShape worst_case(u64 graphs) const;
    /// The arrays a batch of this schema stores, and their layout in a block.
    [[nodiscard]] OutputLayout output_layout(const BatchShape& shape) const;
    /// Scratch plus caller-owned output bytes. Conservative bounds can be large; reserve by measured capacities.
    [[nodiscard]] u64 bytes_needed(const BatchShape& shape) const;
    [[nodiscard]] u64 bytes_needed(u64 graphs) const;
    [[nodiscard]] ScratchLayout scratch_layout(const BatchShape& shape) const;

    /// Allocates only this stream's scratch, on the compute stream, and makes the stream wait for allocation.
    /// Smaller reservations retain the allocation. Refresh resizes existing workspaces when their cell bounds grow.
    /// A failed first reservation leaves the stream unreserved. The batch shape's graphs size the scratch; the other
    /// fields size the output block of output_layout().
    void reserve(const BatchShape& shape, cudaStream_t stream = nullptr);
    void reserve(u64 max_graphs, cudaStream_t stream = nullptr);
    [[nodiscard]] BatchShape reserved(cudaStream_t stream = nullptr) const;
    /// Scratch bytes owned on this stream (outputs are owned by the caller).
    [[nodiscard]] u64 reserved_bytes(cudaStream_t stream = nullptr) const;
    [[nodiscard]] OutputLayout reserved_layout(cudaStream_t stream = nullptr) const;
    /// Leases context and scratch together; throws unless reserve was called for this stream, and on a capturing
    /// stream. The generation and the scratch both free after the work of the acquisition's stream, which one
    /// recorded event per release marks.
    [[nodiscard]] EncodingWorkspace acquire(cudaStream_t stream = nullptr) const;
    /// Drops the scratch and the bookkeeping of `stream`, which may be destroyed once its acquisitions are released.
    /// The scratch frees after their work. The stream reserves again before its next acquire.
    void release(cudaStream_t stream = nullptr);
    bool refresh(std::shared_ptr<const encoding::FlatEncoder> grown, cudaStream_t stream = nullptr);

    /// Encodes the rows into `out` on the workspace's stream: a clear, the count kernel, a scan (two kernels) and the
    /// fill kernel, with no allocation, no synchronization and no stream or event operation, so it may be captured in a
    /// CUDA graph (acquire the workspace before the capture and hold it through the last replay). The batch writes
    /// `out.sizes` with its true sizes; if they exceed the capacities or the index type, or a row is invalid (a task id
    /// outside the table, a row or goal mask narrower than its instance, a bit its instance does not have), nothing of
    /// the batch is written and sizes.overflow says why. The rows, outputs and `launch.order` are device memory the
    /// stream may read and write. Throws std::invalid_argument for input that can be checked without reading device
    /// memory (more graphs than the workspace's shape, arrays that do not match the schema, derived words that
    /// are needed and missing, a launch that needs more shared memory than a block has).
    void encode(const EncodingWorkspace& workspace, const DeviceEncodeInput& in, const DeviceOutput& out,
                const LaunchConfig& launch = {}) const;
    /// The same on `stream` with a workspace acquired for this one call: requires a reservation for the stream.
    void encode(const DeviceEncodeInput& in, const DeviceOutput& out, cudaStream_t stream = nullptr,
                const LaunchConfig& launch = {}) const;

private:
    struct Workspace
    {
        BatchShape shape;
        ScratchLayout layout;
        std::shared_ptr<DeviceBuffer> scratch;
        std::shared_ptr<LastUse> last;  // the use of the scratch and of the generation, recorded at each release
        Event ready;
        std::mutex mutex;
    };
    [[nodiscard]] std::shared_ptr<Workspace> workspace(cudaStream_t stream) const;
    void reserve_workspace(Workspace& workspace, const BatchShape& shape);
    void reserve_locked(const BatchShape& shape, cudaStream_t stream);

    ContextPtr m_ctx;
    std::shared_ptr<const encoding::FlatEncoder> m_encoder;
    std::shared_ptr<const encoding::EncodingSchema> m_schema;
    std::shared_ptr<DeviceEncodingContext> m_context;
    GraphBounds m_bounds;
    std::array<bool, 3> m_reads_derived{};  // by the goals of the rows: none, over the fluent slots, also derived
    mutable std::shared_mutex m_state_mutex;
    mutable std::mutex m_workspace_mutex;
    mutable std::unordered_map<cudaStream_t, std::shared_ptr<Workspace>> m_workspaces;
};
}  // namespace mymyr::cuda
