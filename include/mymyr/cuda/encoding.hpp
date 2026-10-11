#pragma once
// The relational encoder on the device (encoding/encoder.hpp):
//   - DeviceEncodingContext: the host EncodingContext block copied byte for byte into device memory, read by kernels
//     through the POD view detail::ContextView (encoding/detail/lanes.hpp) rebased onto the copy;
//   - DeviceEncoder: the context copy of an encoder's table plus the scratch and output memory of its batches, sized
//     once by reserve().
//
// Shareable across streams like DeviceTaskTable: the upload runs on the stream it is given and records an event;
// acquire(s) makes stream s wait for it and records s as a user of the block: the device memory is freed after the
// work enqueued on s, so s must still exist when the object is destroyed. The copy is immutable after the upload.

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/encoding/encoder.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace mymyr::cuda
{
/// The host context of a table on the device.
class DeviceEncodingContext
{
public:
    /// Copies the context's block to the device on `stream` (nullptr: the device context's compute stream) and
    /// records an event after the copy. Returns before the copy has run. Throws CudaError.
    static std::shared_ptr<DeviceEncodingContext> upload(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host,
                                                         cudaStream_t stream = nullptr);

    DeviceEncodingContext(const DeviceEncodingContext&) = delete;
    DeviceEncodingContext& operator=(const DeviceEncodingContext&) = delete;
    /// Waits for the upload; the device memory is freed after the work enqueued on the streams that acquired it.
    ~DeviceEncodingContext();

    /// Replaces the copy by the copy of `grown`, a context of the same schema over a table whose first instances are
    /// the instances of the current one, with more instances after them. The instances keep their rows in every array
    /// (so their offsets and bytes), and their slot, tuple and cell rows are the same bytes in the new block. One
    /// upload on `stream` (nullptr: the compute stream) with a new event. Thread-safe. Returns false and uploads nothing if
    /// `grown` has the instances of the current context only. Views taken before stay valid for the work enqueued
    /// before the call; take a new view for later work. Throws std::invalid_argument ("mymyr: encoding: ...") if
    /// `grown` is not an extension, and CudaError.
    bool refresh(std::shared_ptr<const encoding::EncodingContext> grown, cudaStream_t stream = nullptr);

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] const std::shared_ptr<const encoding::EncodingContext>& host() const noexcept { return m_host; }
    [[nodiscard]] u32 num_instances() const noexcept { return m_host->table().size(); }
    /// The view of the context on the device: the same arrays at the same offsets, as pointers into the copy.
    [[nodiscard]] encoding::detail::ContextView view() const;
    /// The copy: arrays().bytes() bytes, equal to host()->arrays().block().
    [[nodiscard]] const void* data() const noexcept { return m_block.data(); }
    [[nodiscard]] u64 bytes() const noexcept { return m_block.size(); }
    /// Makes `s` wait for the latest upload and records `s` as a user of the block. Thread-safe.
    void acquire(cudaStream_t s) const;
    /// Copies the block back to the host, after the upload and the work enqueued on `s` (nullptr: the compute stream).
    [[nodiscard]] std::vector<std::byte> download(cudaStream_t s = nullptr) const;
    /// The uploads so far: 1 after upload(), plus 1 for each refresh() that copied.
    [[nodiscard]] u32 uploads() const noexcept { return m_uploads; }

private:
    DeviceEncodingContext(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host);
    void copy(cudaStream_t stream);

    ContextPtr m_ctx;
    std::shared_ptr<const encoding::EncodingContext> m_host;
    mutable DeviceBuffer m_block;
    mutable PinnedBuffer m_staging;  // the source of the copy in flight; freed once its event has passed
    Event m_ready;
    u32 m_uploads = 0;
    mutable std::mutex m_mutex;
};

/// What a batch needs, as totals over its graphs.
struct BatchShape
{
    u64 graphs = 0;
    u64 entities = 0;  // entities of all graphs (objects and auxiliary)
    u64 args = 0;      // entries of relation_args of all graphs (the sum over tuples of their arity)

    [[nodiscard]] friend bool operator==(const BatchShape&, const BatchShape&) noexcept = default;
};

/// The most one graph of any instance of a table can have.
struct GraphBounds
{
    u64 entities = 0;  // num_objects, plus 1 if the instance has a nullary relation
    u64 tuples = 0;    // the slots and static rows of all its cells
    u64 args = 0;      // the same weighted by the arity of the cell
    u64 cells = 0;     // num_cells
};

/// Byte offsets, in the output block, of the arrays of a graph-major batch of `shape` (encoder.hpp lists them): each
/// array starts at a multiple of 64 bytes.
struct OutputLayout
{
    u64 task_ids = 0, num_objects = 0, entity_offsets = 0, entity_role = 0, entity_ref = 0, relation_counts = 0,
        relation_offsets = 0, relation_arg_offsets = 0, relation_args = 0;
    u64 bytes = 0;  // the block: the end of relation_args
};
/// For `relations` relations and index type `index` (4 or 8 bytes per index entry, I below), with G = shape.graphs,
/// R = relations, E = shape.entities, A = shape.args, the arrays take
///   task_ids 4 G; num_objects I G; entity_offsets I (G + 1); entity_role E; entity_ref I E;
///   relation_counts I G R; relation_offsets I (G R + 1); relation_arg_offsets I (G R + 1); relation_args I A
/// bytes, each rounded up to a multiple of 64.
[[nodiscard]] OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations) noexcept;

/// The scratch of a batch, in bytes, as the sum of
///   per-graph totals (encoding::detail::GraphTotals): 24 G,
///   per-graph bases (entity, tuple and argument offsets, three u64): 24 G,
///   per-cell counts of the graphs (u32), for the most cells an instance has: 4 G C,
///   the temporary storage of an inclusive prefix sum of G + 1 u64 values (cub::DeviceScan::InclusiveSum),
/// each rounded up to a multiple of 64 bytes (C: GraphBounds::cells maximum).
[[nodiscard]] u64 scratch_bytes(u64 graphs, u64 max_cells);

class DeviceEncoder
{
public:
    /// The encoder's context copied to the device (on the compute stream of `ctx`). Throws std::invalid_argument
    /// ("mymyr: encoding: ...") for a null argument and for options the device does not encode: derived relations of a
    /// table that has derived atoms (set `include_derived = false`).
    DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder);
    DeviceEncoder(const DeviceEncoder&) = delete;
    DeviceEncoder& operator=(const DeviceEncoder&) = delete;

    [[nodiscard]] const std::shared_ptr<const encoding::FlatEncoder>& encoder() const noexcept { return m_encoder; }
    [[nodiscard]] const encoding::EncodingSchema& schema() const noexcept { return m_encoder->schema(); }
    [[nodiscard]] const encoding::FlatOptions& options() const noexcept { return m_encoder->schema().options(); }
    [[nodiscard]] const DeviceEncodingContext& context() const noexcept { return *m_context; }
    [[nodiscard]] DeviceEncodingContext& context() noexcept { return *m_context; }
    /// Per-graph maxima over the instances of the table.
    [[nodiscard]] const GraphBounds& bounds() const noexcept { return m_bounds; }

    /// The shape of `graphs` graphs that each have the most any instance can have.
    [[nodiscard]] BatchShape worst_case(u64 graphs) const noexcept;
    /// The bytes of the scratch and the output block for a batch of `shape`: scratch_bytes(shape.graphs, bounds().cells)
    /// + output_layout(shape, options().index, schema().num_relations()).bytes. reserve(shape) allocates exactly this.
    [[nodiscard]] u64 bytes_needed(const BatchShape& shape) const;
    /// bytes_needed(worst_case(graphs)): any batch of at most `graphs` graphs fits.
    [[nodiscard]] u64 bytes_needed(u64 graphs) const;

    /// Allocates scratch and output memory for batches up to `shape` (on `stream`, nullptr: the compute stream) and
    /// returns when it is allocated. A shape inside the reserved one allocates nothing; a larger one replaces the
    /// memory by one of the elementwise maximum of the two shapes. Nothing else allocates device memory.
    void reserve(const BatchShape& shape, cudaStream_t stream = nullptr);
    /// reserve(worst_case(max_graphs)).
    void reserve(u64 max_graphs, cudaStream_t stream = nullptr);
    /// The shape reserve() has allocated for (zero before the first call).
    [[nodiscard]] const BatchShape& reserved() const noexcept { return m_reserved; }
    [[nodiscard]] u64 reserved_bytes() const noexcept { return m_scratch.size() + m_output.size(); }
    /// The reserved scratch and output memory (null before reserve). One batch at a time uses them: callers order
    /// their calls on one stream, or with events.
    [[nodiscard]] void* scratch() const noexcept { return m_scratch.data(); }
    [[nodiscard]] void* output() const noexcept { return m_output.data(); }
    /// The layout of the output block for the reserved shape.
    [[nodiscard]] OutputLayout reserved_layout() const noexcept;

    /// Makes `s` wait for the context upload and records `s` as a user of the context copy and of the reserved
    /// memory. Thread-safe.
    void acquire(cudaStream_t s) const;

    /// Replaces the table by a grown one (DeviceEncodingContext::refresh): `grown` has the schema and options of this
    /// encoder and a table that extends the current one. The reserved memory stays; bytes_needed changes with the
    /// bounds. Returns whether the context was copied.
    bool refresh(std::shared_ptr<const encoding::FlatEncoder> grown, cudaStream_t stream = nullptr);

private:
    void set_bounds();

    ContextPtr m_ctx;
    std::shared_ptr<const encoding::FlatEncoder> m_encoder;
    std::shared_ptr<DeviceEncodingContext> m_context;
    GraphBounds m_bounds;
    BatchShape m_reserved;
    mutable DeviceBuffer m_scratch, m_output;
    mutable std::mutex m_mutex;
};
}  // namespace mymyr::cuda
