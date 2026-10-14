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

#include "mymyr/cuda/encoding_scan.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/encoding/encoder.hpp"

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
    [[nodiscard]] friend bool operator==(const BatchShape&, const BatchShape&) noexcept = default;
};

/// Conservative maxima per graph, including repeated statics and the auxiliary entity. They cover the state, static
/// and nullary lanes; the goal, target and virtual-node lanes must add their bounds when they arrive.
struct GraphBounds
{
    u64 entities = 0, tuples = 0, args = 0, cells = 0;
};

/// Byte offsets in a caller-owned graph-major output block; each array begins at a multiple of 64 bytes.
struct OutputLayout
{
    u64 task_ids = 0, num_objects = 0, entity_offsets = 0, entity_role = 0, entity_ref = 0, relation_counts = 0,
        relation_offsets = 0, relation_arg_offsets = 0, relation_args = 0;
    u64 bytes = 0;
};
/// Throws std::overflow_error if a size product, sum or alignment exceeds u64.
[[nodiscard]] OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations);

/// One in-place inclusive scan over the three rows of totals as a single array [3, G+1]: entities, tuples and
/// arguments. A later row is thereby offset by the sums of the earlier rows; the fill kernel subtracts those bases.
/// The scan is two kernels. Totals are u32 when the conservative sum of the three fits u32, u64 otherwise. Cell
/// counts [G, cells] and the CUB temporary storage are separate aligned spans. Relation-major scratch is refused: it
/// needs [G,R] counts instead of [G,C] counts. Offsets are bytes into the scratch.
struct ScratchLayout
{
    u64 entities = 0, tuples = 0, args = 0;
    u64 cell_counts = 0, scan_temp = 0;
    u64 cell_entries = 0, scan_length = 0;
    u64 scan_temp_bytes = 0, bytes = 0;
    bool wide = false;
};
/// Uses the current CUDA device for the CUB query; throws CudaError or std::overflow_error. The width of the totals
/// follows the bounds only, never the capacities of the outputs.
[[nodiscard]] ScratchLayout scratch_layout(const BatchShape& shape, const GraphBounds& bounds, encoding::Layout layout);

/// The scratch of a held workspace as device pointers.
struct ScratchView
{
    void* totals = nullptr;      // [3, G+1] u32 (u64 when wide): entities, tuples, arguments
    u64 totals_stride = 0;       // entries per row, G + 1
    u64 scan_length = 0;         // 3 (G + 1)
    u32* cell_counts = nullptr;  // [G, cells] counts
    u64 cell_entries = 0;
    void* scan_temp = nullptr;
    u64 scan_temp_bytes = 0;
    bool wide = false;
};

/// A held acquisition: the context generation and the scratch of one stream; outputs belong to the caller. It is the
/// object a capture-time encode takes: view(), scratch_view(), shape and layout give the pointers and sizes, and
/// scan_totals() enqueues the scan on stream(). None of them waits on a stream or records an event.
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
    /// Enqueues the in-place scan of the totals on stream().
    cudaError_t scan_totals() const;

    EncodingAcquisition context;
    std::shared_ptr<const DeviceBuffer> scratch;
    BatchShape shape;
    ScratchLayout layout;
};

class DeviceEncoder
{
public:
    /// Derived state relations are refused per schema when include_derived is true. Each stream has its own scratch.
    DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder);
    DeviceEncoder(const DeviceEncoder&) = delete;
    DeviceEncoder& operator=(const DeviceEncoder&) = delete;

    [[nodiscard]] std::shared_ptr<const encoding::FlatEncoder> encoder() const;
    [[nodiscard]] const encoding::EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const encoding::FlatOptions& options() const noexcept { return m_schema->options(); }
    [[nodiscard]] const DeviceEncodingContext& context() const noexcept { return *m_context; }
    [[nodiscard]] GraphBounds bounds() const;
    [[nodiscard]] BatchShape worst_case(u64 graphs) const;
    /// Scratch plus caller-owned output bytes. Conservative bounds can be large; reserve by measured capacities.
    [[nodiscard]] u64 bytes_needed(const BatchShape& shape) const;
    [[nodiscard]] u64 bytes_needed(u64 graphs) const;
    [[nodiscard]] ScratchLayout scratch_layout(const BatchShape& shape) const;

    /// Allocates only this stream's scratch, on the compute stream, and makes the stream wait for allocation.
    /// Smaller reservations retain the allocation. Refresh resizes existing workspaces when their cell bounds grow.
    /// A failed first reservation leaves the stream unreserved.
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
    mutable std::shared_mutex m_state_mutex;
    mutable std::mutex m_workspace_mutex;
    mutable std::unordered_map<cudaStream_t, std::shared_ptr<Workspace>> m_workspaces;
};
}  // namespace mymyr::cuda
