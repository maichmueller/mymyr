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
#include "mymyr/cuda/encoding_types.hpp"
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
namespace detail { struct EncodingExecution; struct EncodingFacts; struct EncodingWorkspaceState; }
class DeviceEncodingContext;
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
    friend class DeviceEncodingContext;
    friend class DeviceEncoder;
    std::shared_ptr<const detail::EncodingFacts> m_facts;
    const DeviceEncodingContext* m_owner = nullptr;
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
    friend class DeviceEncoder;
    DeviceEncodingContext(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host);
    /// An upload that also builds the encoder's facts of every generation (bounds, widths and the axiom evaluation).
    static std::shared_ptr<DeviceEncodingContext> create(ContextPtr ctx, std::shared_ptr<const encoding::EncodingContext> host,
                                                         bool facts, cudaStream_t stream);
    void copy(const std::shared_ptr<const encoding::EncodingContext>& host, cudaStream_t stream, bool facts);
    [[nodiscard]] std::shared_ptr<const detail::EncodingFacts> facts() const
    {
        std::lock_guard lock(m_mutex);
        return m_facts;
    }

    ContextPtr m_ctx;
    std::shared_ptr<const encoding::EncodingContext> m_host;
    std::shared_ptr<const detail::EncodingFacts> m_facts;
    std::shared_ptr<DeviceBuffer> m_block;
    encoding::detail::ContextView m_view;
    PinnedLease m_staging;
    Event m_ready;
    u64 m_generation = 0;
    mutable std::unordered_map<cudaStream_t, std::shared_ptr<LastUse>> m_uses;
    mutable std::mutex m_mutex;
};

using encoding::BatchShape;

/// How the derived words of an encode's rows come, for the size of its scratch: evaluated on the device when a graph
/// reads derived atoms and DeviceEncodeInput::derived is null, or always supplied (no room for an evaluation, which
/// such an encode then refuses).
enum class DerivedWords : u8
{
    Evaluated,
    Supplied,
};

/// Conservative maxima per graph over the instances of the context, covering every lane the device encodes: the state,
/// static and derived cells, the nullary (auxiliary) entity, the goal cells of the instance's goal and of a row's own
/// goal, the subgoal levels, the goal and subgoal targets and one entity per predicate virtual node. `words` is the
/// widest row of an instance (state or derived words), `cells` the longest cell list of a graph (with its subgoal and
/// history cells). The history steps of a row add per step: one entity, the tuples of the history cells and their
/// arguments, and one entity and target per history-target tuple (the `step_` bounds, which are zero without history).
/// `state_words` and `derived_words` are the widest state and derived rows, `levels` the subgoal levels of the schema.
struct GraphBounds
{
    u64 entities = 0, tuples = 0, args = 0, cells = 0, targets = 0, words = 0;
    u64 state_words = 0, derived_words = 0;
    u64 step_entities = 0, step_tuples = 0, step_args = 0, step_targets = 0;
    u32 levels = 0;
    bool history = false;
};

[[nodiscard]] OutputArrays output_arrays(const encoding::EncodingSchema& schema);
/// The collation view of a device batch of `graphs` graphs that `encoder` (or a device encoder over it) wrote.
[[nodiscard]] DeviceCollationBatch collation_batch(const encoding::FlatEncoder& encoder,
                                                   const encoding::FlatBuffers& arrays,
                                                   const encoding::FlatSizes* sizes, u64 graphs);

/// Byte offsets in a caller-owned output block; each array begins at a multiple of 64 bytes. An array that the batch
/// does not store has no bytes. The last array is the FlatSizes the encoding writes.
struct OutputLayout
{
    u64 task_ids = 0, num_objects = 0, entity_offsets = 0, entity_role = 0, entity_ref = 0, relation_counts = 0,
        relation_offsets = 0, relation_arg_offsets = 0, relation_args = 0, goal_complete = 0, target_offsets = 0,
        target_entity = 0, target_group = 0, history_offsets = 0, history_entity = 0, history_dt = 0, sizes = 0;
    u64 bytes = 0;
    OutputArrays arrays;
};
/// Throws std::overflow_error if a size product, sum or alignment exceeds u64.
[[nodiscard]] OutputLayout output_layout(const BatchShape& shape, encoding::IndexType index, u64 relations,
                                         const OutputArrays& arrays = {});

/// The destinations in the device block `block` of output_layout(shape, ...), with the capacities of `shape`
/// (the graph capacity bounds every per-graph and per-cell array).
[[nodiscard]] DeviceOutput output_of(const OutputLayout& layout, const BatchShape& shape, void* block);

/// Append all rows or set status[2] without writing when they do not fit. Status is device uint64
/// [head, live rows, overflow]; a grid copies the rows, then one thread advances the tail. One stream orders all uses.
/// The ring's columns have table-wide widths, fixed presence, and room for ring.graphs rows. Its status is zeroed
/// before first use. The input's widths equal the columns' widths; row strides may differ or broadcast goals.
void append_rows(const DeviceEncodeInput& input, const DeviceEncodeInput& ring, u64* status, cudaStream_t stream);
/// Advances the head by the live rows modulo capacity, empties the ring, and clears the overflow flag.
void clear_rows(u64* status, u64 capacity, cudaStream_t stream);
/// Empties the ring while preserving its sticky overflow flag.
void finish_rows(u64* status, u64 capacity, cudaStream_t stream);
/// Throws std::invalid_argument for a schema whose rows a device ring does not carry: subgoal layers and history steps.
void check_stream_schema(const encoding::EncodingSchema& schema);

enum class FillMode : u8
{
    Auto,  // per-instance work and layout select a thread or warp; shared-memory limits may require a warp
    Thread,
    Warp,
};

/// How a batch is launched. None of it changes the output: the launch order only decides which thread or warp takes
/// which graph, every position being a prefix sum in batch order.
struct LaunchConfig
{
    u32 count_lanes = 0;  // lanes per graph of the count kernel: 4, 8 or 32; 0: four lanes, then eight or 32 if needed for shared memory
    FillMode fill = FillMode::Auto;
    const u32* order = nullptr;  // device [graphs]: a permutation; every graph must occur exactly once (caller contract).
                                 // An entry >= graphs sets the input flag without reading that graph.
};

/// A held acquisition: the context generation and the scratch of one stream; outputs belong to the caller. It is the
/// object a capture-time encode takes: encode() on it enqueues kernels on stream() and nothing else. It waits on no
/// stream and records no event.
class EncodingWorkspace
{
public:
    EncodingWorkspace(EncodingWorkspace&&) noexcept = default;
    EncodingWorkspace& operator=(EncodingWorkspace&&) = delete;
    EncodingWorkspace(const EncodingWorkspace&) = delete;
    EncodingWorkspace& operator=(const EncodingWorkspace&) = delete;

    [[nodiscard]] cudaStream_t stream() const noexcept;
    [[nodiscard]] const encoding::detail::ContextView& view() const noexcept;
    [[nodiscard]] BatchShape shape() const noexcept;
    [[nodiscard]] u64 generation() const noexcept;
    [[nodiscard]] u64 scratch_bytes() const noexcept;

private:
    friend class DeviceEncoder;
    explicit EncodingWorkspace(std::shared_ptr<const detail::EncodingWorkspaceState> state) : m_state(std::move(state)) {}
    std::shared_ptr<const detail::EncodingWorkspaceState> m_state;
};

class DeviceEncoder
{
public:
    /// Throws std::invalid_argument for a schema the device does not encode: action targets and horizons (with the
    /// goal derivations that only a delta horizon has).
    static void check_schema(const encoding::EncodingSchema& schema);
    /// Throws std::invalid_argument for an encoder the device cannot encode: a schema it does not encode, or one whose
    /// predicate virtual nodes or relations need more shared memory than a block has for the default launch. Computed
    /// on the host, without uploading a context (without a CUDA device only the schema is checked).
    static void check_encoder(const encoding::FlatEncoder& encoder);

    /// Each stream has its own scratch.
    DeviceEncoder(ContextPtr ctx, std::shared_ptr<const encoding::FlatEncoder> encoder);
    DeviceEncoder(const DeviceEncoder&) = delete;
    DeviceEncoder& operator=(const DeviceEncoder&) = delete;

    [[nodiscard]] std::shared_ptr<const encoding::FlatEncoder> encoder() const;
    [[nodiscard]] const encoding::EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const encoding::FlatOptions& options() const noexcept { return m_schema->options(); }
    [[nodiscard]] const DeviceEncodingContext& context() const noexcept { return *m_context; }
    [[nodiscard]] GraphBounds bounds() const;
    /// The capacities that hold any batch of `graphs` rows. With history, `history_steps` is the most steps one row's
    /// history has (the width of the history input); at most history_max_steps are encoded, which is also the number of
    /// steps when `history_steps` is 0. A schema that encodes every step needs `history_steps`
    /// (std::invalid_argument without it).
    [[nodiscard]] BatchShape worst_case(u64 graphs, u64 history_steps = 0) const;
    /// The arrays a batch of this schema stores, and their layout in a block.
    [[nodiscard]] OutputLayout output_layout(const BatchShape& shape) const;
    /// Scratch plus caller-owned output bytes. Conservative bounds can be large; reserve by measured capacities.
    [[nodiscard]] u64 bytes_needed(const BatchShape& shape) const;
    [[nodiscard]] u64 bytes_needed(u64 graphs, u64 history_steps = 0) const;
    /// Bytes of device scratch, independent of the output capacities. The address must be aligned to 256 bytes. With
    /// DerivedWords::Supplied, the scratch of encodes that pass their derived words: smaller where a graph reads derived
    /// atoms (no room for their evaluation), equal otherwise.
    [[nodiscard]] u64 scratch_bytes(const BatchShape& shape, DerivedWords derived = DerivedWords::Evaluated) const;
    /// Sizes scratch without uploading a context or exporting a task. Uses the current CUDA device for the scan storage
    /// query.
    [[nodiscard]] static u64 scratch_bytes(const encoding::FlatEncoder& encoder, const BatchShape& shape,
                                           DerivedWords derived = DerivedWords::Evaluated);
    /// The exact byte count for a held generation, including one held across refresh().
    [[nodiscard]] u64 scratch_bytes(const EncodingAcquisition& context, const BatchShape& shape,
                                    DerivedWords derived = DerivedWords::Evaluated) const;
    /// Whether the rows of some instance of `encoder` need their derived words supplied (DeviceEncodeInput::derived): their
    /// graphs read derived atoms (for rows without goal masks, with goal masks over the fluent slots, or with derived
    /// goal masks too) and the device cannot evaluate the axioms (they exceed the matcher limits or the object-bitset
    /// widths), so that such rows without supplied words are flagged invalid (a batch without task ids is refused).
    /// Computed on the host, without uploading a context.
    [[nodiscard]] static bool needs_derived(const encoding::FlatEncoder& encoder, bool goal_masks,
                                            bool derived_goal_masks);

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

    /// Encodes the rows into `out` on the workspace's stream: a clear, a count and fill kernel per active width class,
    /// and a shared scan (two kernels), with no allocation, synchronization or stream or event operation. It may be
    /// captured in a CUDA graph (acquire the workspace before capture and hold it through the last replay). The batch writes
    /// `out.sizes` with its true sizes; if they exceed the capacities or the index type, or a row is invalid (a task id
    /// outside the table, a row or goal mask narrower than its instance, a bit its instance does not have), nothing of
    /// the batch is written and sizes.overflow says why. With `out.zero_unused`, the unused tails are zero and
    /// overflow zeros all output arrays. The rows, outputs and `launch.order` are device memory the
    /// stream may read and write. Throws std::invalid_argument for input that can be checked without reading device
    /// memory (more graphs than the workspace or output shape, arrays that do not match the schema, rows without task ids
    /// and without supplied derived words whose instance 0 has axioms beyond the device limits, a launch that needs more
    /// shared memory than a block has). Rows of another instance beyond those limits, without supplied derived words,
    /// set the input flag of the batch.
    void encode(const EncodingWorkspace& workspace, const DeviceEncodeInput& in, const DeviceOutput& out,
                const LaunchConfig& launch = {}) const;
    /// The same on `stream` with a workspace acquired for this one call: requires a reservation for the stream.
    void encode(const DeviceEncodeInput& in, const DeviceOutput& out, cudaStream_t stream = nullptr,
                const LaunchConfig& launch = {}) const;

    /// encode() in two steps, for a caller that sizes its outputs by the batch: count() enqueues the clear, the count
    /// kernel, the scan and a kernel that writes the true sizes of the rows to `out.sizes` (device memory; the capacity flag
    /// is never set, the index and input flags are), and nothing else. fill() then enqueues the fill kernel with the
    /// capacities of `out`, which writes `out.sizes` again and writes the batch unless a flag is set, as encode() does.
    /// fill() reads the counts that count() left in the workspace's scratch: it must follow a count() of the same
    /// workspace, rows and launch configuration on the same stream with no other count() or encode() of that workspace in
    /// between. Both have the checks of encode(); neither synchronizes.
    void count(const EncodingWorkspace& workspace, const DeviceEncodeInput& in, const DeviceOutput& out,
               const LaunchConfig& launch = {}) const;
    void fill(const EncodingWorkspace& workspace, const DeviceEncodeInput& in, const DeviceOutput& out,
              const LaunchConfig& launch = {}) const;

    /// The caller-owned scratch forms. `shape.graphs` bounds the batch, and `bytes` is at least scratch_bytes(shape), or
    /// scratch_bytes(shape, DerivedWords::Supplied) for rows with supplied derived words (an encode that would evaluate
    /// them in less than scratch_bytes(shape) throws std::invalid_argument).
    /// Obtain the acquisition from context().acquire(stream) before capture. It holds the context generation; keep it,
    /// the scratch, inputs and outputs alive through the last replay. These forms allocate and synchronize nothing.
    /// count() and fill() use the same acquisition, scratch address, shape, rows and launch configuration on one stream,
    /// with no other use of that scratch between them. Refresh preserves the held generation's scratch requirements.
    void encode(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
                const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch = {}) const;
    /// encode() on `stream` instead of the acquisition's stream, for a caller that holds one acquisition for every
    /// stream it encodes on (an XLA custom call, which cannot acquire under capture). The caller orders the upload that
    /// preceded the acquisition before the first launch on `stream` and every launch before the acquisition's release.
    void encode(const EncodingAcquisition& context, cudaStream_t stream, void* scratch, u64 bytes, const BatchShape& shape,
                const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch = {}) const;
    void count(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
               const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch = {}) const;
    void fill(const EncodingAcquisition& context, void* scratch, u64 bytes, const BatchShape& shape,
              const DeviceEncodeInput& in, const DeviceOutput& out, const LaunchConfig& launch = {}) const;

private:
    enum class Phase : u8
    {
        Whole,
        Count,
        Fill,
    };
    void run(Phase phase, const detail::EncodingExecution& execution, const DeviceEncodeInput& in,
             const DeviceOutput& out, const LaunchConfig& launch) const;
    struct Workspace;
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
/// A device ring bound to one stream. Append copies device rows and inputs; flush encodes every live row once,
/// advances the head and empties the ring on that stream. The overflow flag is sticky until clear(), and every flush
/// while it is set reports k_overflow_capacity (nothing of the batch is written): a batch never silently lacks the rows
/// of an append that did not fit. The rows carry no subgoal layers or history steps (check_stream_schema).
/// Construct outside capture and keep the ring, encoder and stream alive through the final replay. All operations
/// on one ring are ordered on its stream; independent rings may run concurrently. The mutable stream is host-only.
class DeviceRowStream
{
public:
    DeviceRowStream(ContextPtr context, std::shared_ptr<const encoding::FlatEncoder> encoder, u64 capacity,
                    cudaStream_t stream = nullptr, bool goals = false, bool derived = false, bool derived_goals = false);
    ~DeviceRowStream();
    DeviceRowStream(const DeviceRowStream&) = delete;
    DeviceRowStream& operator=(const DeviceRowStream&) = delete;
    [[nodiscard]] u64 capacity() const noexcept;
    [[nodiscard]] cudaStream_t stream() const noexcept;
    /// Device uint64 [head, live rows, overflow]. Reading it on the host requires synchronization.
    [[nodiscard]] const u64* status() const noexcept;
    void append(const DeviceEncodeInput& input);
    /// Capacity form. out.graphs, when nonnull, receives the number of encoded rows on the device. Per-graph arrays
    /// hold capacity() graphs, zero past the live ones; the variable arrays past the sizes are zero with
    /// out.zero_unused and left unwritten without it, so that the work of a flush follows its live rows. Capturable.
    void flush(const DeviceOutput& out);
    void clear();
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/// An uploaded list of batch views, bound to one stream. Construction validates common schema, table, layout, index
/// type, statics and stored arrays, and refuses subgoal layers and history steps; empty slices are accepted. Construct outside capture. encode/count/fill only launch
/// count, scan and fill kernels, with caller-owned 256-byte-aligned scratch, and may be captured. Keep the collator,
/// source arrays, scratch, output and stream alive through the final replay. No host loop visits individual graphs.
class DeviceCollator
{
public:
    DeviceCollator(ContextPtr context, std::vector<DeviceCollationBatch> batches, cudaStream_t stream = nullptr);
    ~DeviceCollator();
    DeviceCollator(const DeviceCollator&) = delete;
    DeviceCollator& operator=(const DeviceCollator&) = delete;
    [[nodiscard]] u64 graphs() const noexcept;
    [[nodiscard]] u64 scratch_bytes() const noexcept;
    void encode(void* scratch, u64 bytes, const DeviceOutput& out) const;
    /// count writes sizes with capacity unbounded; fill reuses its scratch and unchanged source arrays,
    /// with no intervening use of that scratch.
    void count(void* scratch, u64 bytes, encoding::FlatSizes* sizes) const;
    void fill(void* scratch, u64 bytes, const DeviceOutput& out) const;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace mymyr::cuda
