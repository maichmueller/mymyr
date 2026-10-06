#pragma once
// The CUDA backend's runtime layer:
//   - DeviceContext: a user-owned device context (no global state): the device ordinal, its own stream-ordered
//     memory pool (cudaMemPoolCreate, not the device's default pool), a compute stream and a copy stream. Everything
//     that allocates device memory holds a shared_ptr to its context, so the pool outlives every buffer;
//   - memory resources over the context (core/memory.hpp's MemoryResource: Device, Pinned, Managed);
//   - DeviceBuffer: RAII stream-ordered device memory (cudaMallocFromPoolAsync / cudaFreeAsync) that may be read on
//     other streams: record_stream(s) makes the free wait for the work enqueued on s (torch's record_stream rule);
//   - PinnedBuffer, Event, Stream: small RAII wrappers; PinnedLease: pinned staging buffers a context caches.
// Host code only: compiled by the host compiler (C++26); device code includes rl/task_arrays_view.hpp and
// cuda/kernels.hpp only. Errors throw CudaError (destructors never throw).

#include "mymyr/core/memory.hpp"
#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::cuda
{
class CudaError : public std::runtime_error
{
public:
    CudaError(cudaError_t code, const std::string& what);
    [[nodiscard]] cudaError_t code() const noexcept { return m_code; }

private:
    cudaError_t m_code;
};

/// Throws CudaError if e is not cudaSuccess.
void check(cudaError_t e, const char* what);

/// Number of visible CUDA devices (0 without a driver or device; never throws).
[[nodiscard]] int device_count() noexcept;

/// Sets the calling thread's current device for a scope and restores the previous one (the runtime's current device
/// is per-thread state; mymyr never leaves it changed).
class DeviceGuard
{
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int m_prev = -1;
    bool m_changed = false;
};

class Event
{
public:
    explicit Event(bool timing = false);
    ~Event();
    Event(Event&& o) noexcept : m_e(o.m_e) { o.m_e = nullptr; }
    Event& operator=(Event&& o) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t s);
    /// Makes stream s wait for this event (enqueued; does not block the host).
    void wait_on(cudaStream_t s) const;
    void synchronize() const;
    [[nodiscard]] bool done() const;
    /// Milliseconds from `start` to this event (both recorded with timing).
    [[nodiscard]] float elapsed_ms(const Event& start) const;
    [[nodiscard]] cudaEvent_t get() const noexcept { return m_e; }

private:
    cudaEvent_t m_e = nullptr;
};

/// Makes `consumer` wait for the work enqueued so far on `producer` (an event handoff; nothing if they are equal).
void stream_wait(cudaStream_t consumer, cudaStream_t producer);

/// An owned non-blocking stream of the current device; the destructor synchronizes it before destroying it.
class Stream
{
public:
    Stream();
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return m_s; }
    operator cudaStream_t() const noexcept { return m_s; }  // NOLINT: a handle
    void synchronize() const;

private:
    cudaStream_t m_s = nullptr;
};

class PinnedBuffer;
class PinnedLease;

struct ContextOptions
{
    /// Upper bound of the pool (cudaMemPoolProps::maxSize); 0: the device's limit.
    u64 max_bytes = 0;
    /// Bytes the pool keeps reserved across synchronizations instead of returning them to the driver. All by
    /// default (trim() returns them): otherwise a run's transient buffers above the threshold would be unmapped at
    /// each synchronization and mapped again by the next allocation.
    u64 release_threshold = ~u64{0};
};

struct PoolUsage
{
    u64 reserved = 0;       // bytes the pool holds from the driver
    u64 used = 0;           // bytes handed out
    u64 reserved_high = 0;  // high-water marks
    u64 used_high = 0;
};

class DeviceContext : public std::enable_shared_from_this<DeviceContext>
{
    struct Private
    {
    };

public:
    static std::shared_ptr<DeviceContext> create(int device = 0, const ContextOptions& options = {});
    DeviceContext(Private, int device, const ContextOptions& options);
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    [[nodiscard]] int device() const noexcept { return m_device; }
    [[nodiscard]] const std::string& name() const noexcept { return m_name; }
    [[nodiscard]] int compute_capability() const noexcept { return m_cc; }
    /// Non-blocking compute stream (kernels, uploads) and copy stream (tail syncs).
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }
    [[nodiscard]] cudaStream_t copy_stream() const noexcept { return m_copy; }
    [[nodiscard]] cudaMemPool_t pool() const noexcept { return m_pool; }
    [[nodiscard]] const ContextOptions& options() const noexcept { return m_options; }

    /// Stream-ordered allocation from the context's pool: usable on `s` from this point of its order (and on other
    /// streams after an event handoff). Throws CudaError (cudaErrorMemoryAllocation beyond max_bytes).
    [[nodiscard]] void* allocate(u64 bytes, cudaStream_t s);
    /// Stream-ordered free on `s`: the memory is reused only after the work enqueued on s before this call.
    void deallocate(void* p, cudaStream_t s) noexcept;

    /// Memory resources (core/memory.hpp) of this context. Device: stream-ordered on stream(), and allocate()
    /// synchronizes that stream so the memory is usable anywhere; Pinned: cudaMallocHost; Managed: cudaMallocManaged.
    [[nodiscard]] MemoryResource& device_memory() noexcept { return *m_device_mr; }
    [[nodiscard]] MemoryResource& pinned_memory() noexcept { return *m_pinned_mr; }
    [[nodiscard]] MemoryResource& managed_memory() noexcept { return *m_managed_mr; }

    /// `n` pinned host buffers of at least `bytes` each (staging for downloads into pageable memory), from a
    /// cache the context keeps across calls (pinning is costly per call); the lease gives them back. Thread-safe.
    [[nodiscard]] PinnedLease lease_pinned(u32 n, u64 bytes);

    [[nodiscard]] PoolUsage usage() const;
    /// Returns reserved pool memory beyond `keep` bytes to the driver (after synchronizing the context's streams), and
    /// frees the cached pinned buffers (lease_pinned).
    void trim(u64 keep = 0);
    /// Blocks until the compute and copy streams are idle.
    void synchronize() const;

private:
    int m_device = 0;
    int m_cc = 0;
    std::string m_name;
    ContextOptions m_options;
    cudaMemPool_t m_pool = nullptr;
    cudaStream_t m_stream = nullptr;
    cudaStream_t m_copy = nullptr;
    std::unique_ptr<MemoryResource> m_device_mr, m_pinned_mr, m_managed_mr;
    std::mutex m_pinned_mutex;
    std::vector<PinnedBuffer> m_pinned_cache;  // lease_pinned's buffers between leases

    friend class PinnedLease;
};

using ContextPtr = std::shared_ptr<DeviceContext>;

/// Stream-ordered device memory from a context's pool. Move-only; the context stays alive while the buffer does.
/// The end of an owner's last call on its buffers: an event the owner records on each call's stream after the
/// call's work (record()). A buffer that tracks it (DeviceBuffer::track) frees after it on a live stream (the context's,
/// or the releasing call's), never on its allocation stream, so a caller may destroy its stream after a call while
/// the owner's buffers live on. Shared by the owner's buffers; one record per call.
class LastUse
{
public:
    explicit LastUse(int device);
    ~LastUse();
    LastUse(const LastUse&) = delete;
    LastUse& operator=(const LastUse&) = delete;
    /// The calls so far end with the work enqueued on s up to here (s must exist now, not later). Thread-safe.
    void record(cudaStream_t s);
    /// Makes stream s wait for the last record (nothing before the first one).
    void wait_on(cudaStream_t s) const noexcept;
    /// Whether record() was called (the owner marks its calls' ends).
    [[nodiscard]] bool recorded() const noexcept;

private:
    int m_device = 0;
    cudaEvent_t m_event = nullptr;
    bool m_recorded = false;
    mutable std::mutex m_mutex;
};

class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    /// Allocates `bytes` on stream `s` (nullptr: the context's compute stream). The buffer's stream is `s`.
    DeviceBuffer(ContextPtr ctx, u64 bytes, cudaStream_t s = nullptr);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return m_ptr; }
    [[nodiscard]] u64 size() const noexcept { return m_bytes; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_stream; }
    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    /// Declares that work enqueued on `s` reads or writes the buffer: the free is ordered after the work enqueued on
    /// s up to the free (s must still exist then). Thread-safe.
    void record_stream(cudaStream_t s);
    /// The buffer's uses end with `last` (its owner's LastUse): it frees after `last` on the context's stream (or
    /// the stream reset() is given), and its allocation stream may be gone by then. Thread-safe.
    void track(std::shared_ptr<const LastUse> last);
    /// The uses on `s` (record_stream, or the allocation stream) end with `last`, just recorded on s: the free
    /// waits for `last` instead of touching s, which may be gone by then. Thread-safe.
    void retire_stream(cudaStream_t s, std::shared_ptr<const LastUse> last);
    /// Frees now (stream-ordered), as the destructor does: on the allocation stream, or on the context's stream when
    /// the buffer tracks a LastUse.
    void reset() noexcept;
    /// Frees now on `s` (a live stream, e.g. the call that releases the buffer: ordered after its work so far),
    /// after the recorded users, the tracked LastUse, and (when it tracks none) the allocation stream's work so far.
    void reset(cudaStream_t s) noexcept;

private:
    ContextPtr m_ctx;
    void* m_ptr = nullptr;
    u64 m_bytes = 0;
    cudaStream_t m_stream = nullptr;
    std::unique_ptr<std::mutex> m_mutex = std::make_unique<std::mutex>();
    std::vector<cudaStream_t> m_users;
    std::vector<std::shared_ptr<const LastUse>> m_lasts;  // the free waits for each
    bool m_free_live = false;                             // the allocation stream may be gone: free on a live one
};

/// Page-locked host memory (cudaMallocHost), for tail syncs and staging. Move-only.
class PinnedBuffer
{
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(u64 bytes);
    ~PinnedBuffer();
    PinnedBuffer(PinnedBuffer&& o) noexcept : m_ptr(o.m_ptr), m_bytes(o.m_bytes)
    {
        o.m_ptr = nullptr;
        o.m_bytes = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return m_ptr; }
    [[nodiscard]] u64 size() const noexcept { return m_bytes; }

private:
    void* m_ptr = nullptr;
    u64 m_bytes = 0;
};

/// Pinned buffers lent by a context (DeviceContext::lease_pinned); the destructor returns them to its cache.
class PinnedLease
{
public:
    PinnedLease() = default;
    PinnedLease(std::shared_ptr<DeviceContext> ctx, std::vector<PinnedBuffer> buffers) noexcept
        : m_ctx(std::move(ctx)), m_buffers(std::move(buffers))
    {
    }
    ~PinnedLease();
    PinnedLease(PinnedLease&&) noexcept = default;
    PinnedLease& operator=(PinnedLease&& o) noexcept;
    PinnedLease(const PinnedLease&) = delete;
    PinnedLease& operator=(const PinnedLease&) = delete;

    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_buffers.size()); }
    [[nodiscard]] void* data(u32 i) const noexcept { return m_buffers[i].data(); }

private:
    void give_back() noexcept;

    std::shared_ptr<DeviceContext> m_ctx;
    std::vector<PinnedBuffer> m_buffers;
};

/// An instantiated CUDA graph, captured from a stream and replayed with launch() (a loop body's launch sequence
/// captured once and replayed per iteration, without its launches' host cost). Move-only; the destructor frees it.
class GraphExec
{
public:
    GraphExec() = default;
    ~GraphExec();
    GraphExec(GraphExec&& o) noexcept : m_exec(o.m_exec) { o.m_exec = nullptr; }
    GraphExec& operator=(GraphExec&& o) noexcept;
    GraphExec(const GraphExec&) = delete;
    GraphExec& operator=(const GraphExec&) = delete;

    /// Captures what `enqueue` puts on stream `s` (relaxed mode: recorded, not run) and instantiates it. The work must
    /// not allocate, synchronize or copy from pageable memory. Throws, with the stream out of capture again.
    template<class F>
    static GraphExec capture(cudaStream_t s, F&& enqueue)
    {
        begin_capture(s);
        try
        {
            enqueue();
        }
        catch (...)
        {
            abandon_capture(s);
            throw;
        }
        return end_capture(s);
    }
    /// Captures a device loop: a graph of one conditional WHILE node whose body is what `body(handle)` puts on `s`
    /// (relaxed mode, recorded into the body). The body runs at least once per launch and again while the handle is
    /// nonzero: a kernel of the body ends the loop with cudaGraphSetConditional(handle, 0). Throws, with the stream out
    /// of capture again.
    template<class F>
    static GraphExec capture_while(cudaStream_t s, F&& body)
    {
        cudaGraph_t g = nullptr;
        const unsigned long long handle = begin_while(s, g);
        try
        {
            body(handle);
        }
        catch (...)
        {
            abandon_while(s, g);
            throw;
        }
        return end_while(s, g);
    }
    /// Whether callers capture graphs or launch their work from the host: true unless the environment sets
    /// MYMYR_CUDA_GRAPHS=0 (read once). compute-sanitizer's racecheck (2025.3) breaks down over many captured graphs
    /// (its host library crashed after thousands of captures and replays); the kernels are the same either way.
    [[nodiscard]] static bool enabled() noexcept;
    /// Whether callers capture device loops (capture_while) or fall back to replays launched by the host: enabled()
    /// unless the environment sets MYMYR_CUDA_DEVICE_LOOPS=0 (read once). Every compute-sanitizer tool (2025.3) runs
    /// without device loops; the kernels are the same either way.
    /// - racecheck and synccheck do not separate the kernels of a conditional node's body. They report races and
    ///   divergent barriers across kernels that a captured graph without the WHILE node runs clean.
    /// - memcheck faulted the GPU inside BrFS WHILE bodies, at an address outside the application's allocations.
    ///   Inside them it also reported accesses within live allocations as out of bounds.
    [[nodiscard]] static bool loops_enabled() noexcept;
    /// Enqueues one replay on `s`.
    void launch(cudaStream_t s) const;
    [[nodiscard]] explicit operator bool() const noexcept { return m_exec != nullptr; }

private:
    static void begin_capture(cudaStream_t s);
    static void abandon_capture(cudaStream_t s) noexcept;
    static GraphExec end_capture(cudaStream_t s);
    static unsigned long long begin_while(cudaStream_t s, cudaGraph_t& g);
    static void abandon_while(cudaStream_t s, cudaGraph_t g) noexcept;
    static GraphExec end_while(cudaStream_t s, cudaGraph_t g);

    cudaGraphExec_t m_exec = nullptr;
};
}  // namespace mymyr::cuda
