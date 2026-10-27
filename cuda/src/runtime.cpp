// The CUDA backend's runtime layer: contexts, pools, buffers, events (include/mymyr/cuda/runtime.hpp).

#include "completed_release.hpp"
#include "deferred_release.hpp"

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/cuda/suite_expand.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string_view>
#include <utility>

namespace mymyr::cuda
{
namespace
{
thread_local cudaStream_t completed_release_stream = nullptr;
}

CudaError::CudaError(cudaError_t code, const std::string& what)
    : std::runtime_error("mymyr: CUDA: " + what + ": " + cudaGetErrorName(code) + " (" + cudaGetErrorString(code) + ")"),
      m_code(code)
{
}

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        (void)cudaGetLastError();  // clear a sticky non-fatal error so the next call does not report it again
        throw CudaError(e, what);
    }
}

int device_count() noexcept
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        (void)cudaGetLastError();
        return 0;
    }
    return n;
}

std::string_view build_info() noexcept { return "CUDA " MYMYR_CUDA_TOOLKIT_VERSION ", " MYMYR_CUDA_CODE; }

DeviceGuard::DeviceGuard(int device)
{
    if (cudaGetDevice(&m_prev) != cudaSuccess)
    {
        (void)cudaGetLastError();
        m_prev = -1;
    }
    if (m_prev != device)
    {
        check(cudaSetDevice(device), "cudaSetDevice");
        m_changed = true;
    }
}

DeviceGuard::~DeviceGuard()
{
    if (m_changed && m_prev >= 0)
        (void)cudaSetDevice(m_prev);
}

// ------------------------------------------------------------------------------------------------ events

Event::Event(bool timing)
{
    check(cudaEventCreateWithFlags(&m_e, timing ? cudaEventDefault : cudaEventDisableTiming), "cudaEventCreate");
}

Event::~Event()
{
    if (m_e)
        (void)cudaEventDestroy(m_e);
}

Event& Event::operator=(Event&& o) noexcept
{
    if (this != &o)
    {
        if (m_e)
            (void)cudaEventDestroy(m_e);
        m_e = std::exchange(o.m_e, nullptr);
    }
    return *this;
}

void Event::record(cudaStream_t s) { check(cudaEventRecord(m_e, s), "cudaEventRecord"); }
void Event::wait_on(cudaStream_t s) const { check(cudaStreamWaitEvent(s, m_e, 0), "cudaStreamWaitEvent"); }
void Event::synchronize() const
{
    RelaxedCaptureScope capture_mode;
    check(cudaEventSynchronize(m_e), "cudaEventSynchronize");
}

bool Event::done() const
{
    const cudaError_t e = cudaEventQuery(m_e);
    if (e == cudaErrorNotReady)
        return false;
    check(e, "cudaEventQuery");
    return true;
}

float Event::elapsed_ms(const Event& start) const
{
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start.m_e, m_e), "cudaEventElapsedTime");
    return ms;
}

void stream_wait(cudaStream_t consumer, cudaStream_t producer)
{
    if (consumer == producer)
        return;
    Event e;
    e.record(producer);
    e.wait_on(consumer);
}

Stream::Stream() { check(cudaStreamCreateWithFlags(&m_s, cudaStreamNonBlocking), "cudaStreamCreate"); }

Stream::~Stream()
{
    RelaxedCaptureScope capture_mode;
    if (m_s)
    {
        (void)cudaStreamSynchronize(m_s);
        (void)cudaStreamDestroy(m_s);
    }
}

void Stream::synchronize() const
{
    RelaxedCaptureScope capture_mode;
    check(cudaStreamSynchronize(m_s), "cudaStreamSynchronize");
}

// ------------------------------------------------------------------------------------------------ resources

namespace
{
class DeviceResource final : public MemoryResource
{
public:
    explicit DeviceResource(DeviceContext& ctx) : m_ctx(ctx) {}
    void* allocate(usize bytes, usize align) override
    {
        if (align > 256)
            throw std::bad_alloc();
        void* p = m_ctx.allocate(bytes, m_ctx.stream());
        check(cudaStreamSynchronize(m_ctx.stream()), "cudaStreamSynchronize");  // usable on every stream
        return p;
    }
    void deallocate(void* p, usize, usize) noexcept override { m_ctx.deallocate(p, m_ctx.stream()); }
    [[nodiscard]] MemKind kind() const noexcept override { return MemKind::Device; }

private:
    DeviceContext& m_ctx;
};

class PinnedResource final : public MemoryResource
{
public:
    explicit PinnedResource(int device) : m_device(device) {}
    void* allocate(usize bytes, usize align) override
    {
        if (align > 4096)
            throw std::bad_alloc();
        DeviceGuard g(m_device);
        void* p = nullptr;
        check(cudaMallocHost(&p, std::max<usize>(bytes, 1)), "cudaMallocHost");
        return p;
    }
    void deallocate(void* p, usize, usize) noexcept override
    {
        if (p)
        {
            RelaxedCaptureScope capture_mode;
            (void)cudaFreeHost(p);
        }
    }
    [[nodiscard]] MemKind kind() const noexcept override { return MemKind::Pinned; }

private:
    int m_device;
};

class ManagedResource final : public MemoryResource
{
public:
    explicit ManagedResource(int device) : m_device(device) {}
    void* allocate(usize bytes, usize align) override
    {
        if (align > 256)
            throw std::bad_alloc();
        DeviceGuard g(m_device);
        void* p = nullptr;
        check(cudaMallocManaged(&p, std::max<usize>(bytes, 1), cudaMemAttachGlobal), "cudaMallocManaged");
        return p;
    }
    void deallocate(void* p, usize, usize) noexcept override
    {
        if (p)
        {
            RelaxedCaptureScope capture_mode;
            (void)cudaFree(p);
        }
    }
    [[nodiscard]] MemKind kind() const noexcept override { return MemKind::Managed; }

private:
    int m_device;
};
}  // namespace

// ------------------------------------------------------------------------------------------------ context

std::shared_ptr<DeviceContext> DeviceContext::create(int device, const ContextOptions& options)
{
    return std::make_shared<DeviceContext>(Private{}, device, options);
}

DeviceContext::DeviceContext(Private, int device, const ContextOptions& options) : m_device(device), m_options(options)
{
    const int n = device_count();
    if (device < 0 || device >= n)
        throw CudaError(cudaErrorInvalidDevice,
                        "no CUDA device " + std::to_string(device) + " (" + std::to_string(n) + " visible)");
    DeviceGuard g(device);
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    m_name = prop.name;
    m_cc = prop.major * 10 + prop.minor;
    int pools = 0;
    check(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device), "cudaDeviceGetAttribute");
    if (!pools)
        throw CudaError(cudaErrorNotSupported, "device " + std::to_string(device) + " has no stream-ordered memory pools");
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    props.maxSize = options.max_bytes;
    check(cudaMemPoolCreate(&m_pool, &props), "cudaMemPoolCreate");
    std::uint64_t threshold = options.release_threshold;
    check(cudaMemPoolSetAttribute(m_pool, cudaMemPoolAttrReleaseThreshold, &threshold), "cudaMemPoolSetAttribute");
    check(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check(cudaStreamCreateWithFlags(&m_copy, cudaStreamNonBlocking), "cudaStreamCreate");
    m_device_mr = std::make_unique<DeviceResource>(*this);
    m_pinned_mr = std::make_unique<PinnedResource>(device);
    m_managed_mr = std::make_unique<ManagedResource>(device);
}

DeviceContext::~DeviceContext()
{
    RelaxedCaptureScope capture_mode;
    try
    {
        DeviceGuard g(m_device);
        if (m_stream)
        {
            (void)cudaStreamSynchronize(m_stream);
            (void)cudaStreamDestroy(m_stream);
        }
        if (m_copy)
        {
            (void)cudaStreamSynchronize(m_copy);
            (void)cudaStreamDestroy(m_copy);
        }
        if (m_pool)
            (void)cudaMemPoolDestroy(m_pool);  // released once outstanding frees complete
    }
    catch (...)
    {
        // runtime shutting down (interpreter exit): nothing left to release
    }
}

void* DeviceContext::allocate(u64 bytes, cudaStream_t s)
{
    if (bytes == 0)
        return nullptr;
    detail::drain_releases(*this);
    DeviceGuard g(m_device);
    void* p = nullptr;
    const cudaError_t e = cudaMallocFromPoolAsync(&p, bytes, m_pool, s ? s : m_stream);
    if (e != cudaSuccess)
        check(e, ("cudaMallocFromPoolAsync (" + std::to_string(bytes) + " bytes)").c_str());
    return p;
}

void DeviceContext::deallocate(void* p, cudaStream_t s) noexcept
{
    if (!p)
        return;
    RelaxedCaptureScope capture_mode;
    try
    {
        DeviceGuard guard(m_device);
        s = s ? s : m_stream;
        if (auto capture = detail::capture_lifetime(s))
            detail::release_deferred(shared_from_this(), p, {}, {std::move(capture)});
        else
            check(cudaFreeAsync(p, s), "cudaFreeAsync");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "mymyr: CUDA: free retained allocation: %s\n", e.what());
    }
}

void DeviceContext::release_after(std::shared_ptr<void> owner, void (*on_failure)() noexcept)
{
    DeviceGuard guard(m_device);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(m_stream, &status), "cudaStreamIsCapturing");
    if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("mymyr: CUDA: release_after needs a non-capturing context stream");
    Event ready;
    ready.record(m_stream);
    detail::release_deferred(shared_from_this(), std::move(owner), std::move(ready), on_failure);
}

PoolUsage DeviceContext::usage() const
{
    detail::drain_releases(*this);
    PoolUsage u;
    std::uint64_t v = 0;
    check(cudaMemPoolGetAttribute(m_pool, cudaMemPoolAttrReservedMemCurrent, &v), "cudaMemPoolGetAttribute");
    u.reserved = v;
    check(cudaMemPoolGetAttribute(m_pool, cudaMemPoolAttrUsedMemCurrent, &v), "cudaMemPoolGetAttribute");
    u.used = v;
    check(cudaMemPoolGetAttribute(m_pool, cudaMemPoolAttrReservedMemHigh, &v), "cudaMemPoolGetAttribute");
    u.reserved_high = v;
    check(cudaMemPoolGetAttribute(m_pool, cudaMemPoolAttrUsedMemHigh, &v), "cudaMemPoolGetAttribute");
    u.used_high = v;
    return u;
}

void DeviceContext::trim(u64 keep)
{
    RelaxedCaptureScope capture_mode;
    synchronize();
    check(cudaMemPoolTrimTo(m_pool, keep), "cudaMemPoolTrimTo");
    std::vector<PinnedBuffer> cached;
    {
        std::lock_guard lock(m_pinned_mutex);
        cached.swap(m_pinned_cache);
    }
}

PinnedLease DeviceContext::lease_pinned(u32 n, u64 bytes)
{
    std::vector<PinnedBuffer> out;
    out.reserve(n);
    {
        // the smallest cached buffers that fit (a small lease does not take a large staging buffer)
        std::lock_guard lock(m_pinned_mutex);
        while (out.size() < n)
        {
            usize best = m_pinned_cache.size();
            for (usize i = 0; i < m_pinned_cache.size(); ++i)
                if (m_pinned_cache[i].size() >= bytes && (best == m_pinned_cache.size() || m_pinned_cache[i].size() < m_pinned_cache[best].size()))
                    best = i;
            if (best == m_pinned_cache.size())
                break;
            out.push_back(std::move(m_pinned_cache[best]));
            m_pinned_cache.erase(m_pinned_cache.begin() + static_cast<std::ptrdiff_t>(best));
        }
    }
    DeviceGuard g(m_device);
    while (out.size() < n)
        out.emplace_back(bytes);
    return {shared_from_this(), std::move(out)};
}

void DeviceContext::synchronize() const
{
    RelaxedCaptureScope capture_mode;
    DeviceGuard g(m_device);
    check(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
    check(cudaStreamSynchronize(m_copy), "cudaStreamSynchronize");
    detail::drain_releases(*this);
}

// ------------------------------------------------------------------------------------------------ buffers

DeviceBuffer::DeviceBuffer(ContextPtr ctx, u64 bytes, cudaStream_t s)
    : m_ctx(std::move(ctx)), m_bytes(bytes), m_stream(s ? s : m_ctx->stream())
{
    m_ptr = m_ctx->allocate(bytes, m_stream);
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept
    : m_ctx(std::move(o.m_ctx)), m_ptr(std::exchange(o.m_ptr, nullptr)), m_bytes(std::exchange(o.m_bytes, 0)),
      m_stream(o.m_stream), m_mutex(std::move(o.m_mutex)), m_users(std::move(o.m_users)), m_lasts(std::move(o.m_lasts)),
      m_free_live(o.m_free_live)
{
    o.m_mutex = std::make_unique<std::mutex>();
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept
{
    if (this != &o)
    {
        reset();
        m_ctx = std::move(o.m_ctx);
        m_ptr = std::exchange(o.m_ptr, nullptr);
        m_bytes = std::exchange(o.m_bytes, 0);
        m_stream = o.m_stream;
        m_users = std::move(o.m_users);
        m_lasts = std::move(o.m_lasts);
        m_free_live = o.m_free_live;
    }
    return *this;
}

void DeviceBuffer::record_stream(cudaStream_t s)
{
    if (s == m_stream)
        return;
    std::lock_guard lock(*m_mutex);
    if (std::find(m_users.begin(), m_users.end(), s) == m_users.end())
        m_users.push_back(s);
}

void DeviceBuffer::track(std::shared_ptr<const LastUse> last)
{
    std::lock_guard lock(*m_mutex);
    if (std::find(m_lasts.begin(), m_lasts.end(), last) == m_lasts.end())
        m_lasts.push_back(std::move(last));
    m_free_live = true;
}

void DeviceBuffer::retire_stream(cudaStream_t s, std::shared_ptr<const LastUse> last)
{
    std::lock_guard lock(*m_mutex);
    std::erase(m_users, s);
    if (s == m_stream)
        m_free_live = true;
    if (std::find(m_lasts.begin(), m_lasts.end(), last) == m_lasts.end())
        m_lasts.push_back(std::move(last));
}

void DeviceBuffer::reset() noexcept { reset(nullptr); }

namespace
{
/// Makes `waiter` wait for the work enqueued on `s` so far (errors are dropped: a best-effort ordering).
void order_after(cudaStream_t waiter, cudaStream_t s) noexcept
{
    cudaEvent_t e = nullptr;
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess)
    {
        if (cudaEventRecord(e, s) == cudaSuccess)
            (void)cudaStreamWaitEvent(waiter, e, 0);
        (void)cudaEventDestroy(e);
    }
    (void)cudaGetLastError();
}
}  // namespace

void DeviceBuffer::reset(cudaStream_t release) noexcept
{
    if (m_ptr && completed_release_stream)
    {
        m_ctx->deallocate(m_ptr, completed_release_stream);
        m_ptr = nullptr;
        m_bytes = 0;
        m_free_live = false;
        m_users.clear();
        m_lasts.clear();
        return;
    }
    if (!m_ptr)
        return;
    RelaxedCaptureScope capture_mode;
    try
    {
        std::lock_guard lock(*m_mutex);
        DeviceGuard guard(m_ctx->device());
        // the stream that frees: the releasing call's, the context's (a tracked LastUse: the allocation stream may be
        // gone), or the allocation stream
        const cudaStream_t fs = release ? release : m_free_live ? m_ctx->stream() : m_stream;
        std::vector<cudaStream_t> streams = m_users;
        streams.push_back(fs);
        if (!m_free_live)
            streams.push_back(m_stream);
        std::vector<cudaStream_t> uncaptured;
        std::vector<detail::CapturePtr> captures;
        for (usize i = 0; i < streams.size(); ++i)
        {
            const auto s = streams[i];
            if (std::find(streams.begin(), streams.begin() + static_cast<std::ptrdiff_t>(i), s) !=
                streams.begin() + static_cast<std::ptrdiff_t>(i))
                continue;
            if (auto capture = detail::capture_lifetime(s))
                captures.push_back(std::move(capture));
            else
                uncaptured.push_back(s);
        }
        for (const auto& last : m_lasts)
            last->snapshot(fs, captures);
        if (!captures.empty())
        {
            std::vector<Event> events;
            for (auto s : uncaptured)
            {
                events.emplace_back();
                events.back().record(s);
            }
            detail::release_deferred(m_ctx, m_ptr, std::move(events), std::move(captures));
        }
        else
        {
            for (cudaStream_t s : m_users)
                if (s != fs)
                    order_after(fs, s);  // (the users must exist now)
            if (!m_free_live && fs != m_stream)
                order_after(fs, m_stream);
            m_ctx->deallocate(m_ptr, fs);
        }
        m_ptr = nullptr;
        m_bytes = 0;
        m_free_live = false;
        m_users.clear();
        m_lasts.clear();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "mymyr: CUDA: buffer release retained allocation: %s\n", e.what());
    }
}

namespace
{
class CompletedRelease
{
public:
    explicit CompletedRelease(ContextPtr ctx);
    ~CompletedRelease();
    CompletedRelease(const CompletedRelease&) = delete;
    CompletedRelease& operator=(const CompletedRelease&) = delete;

private:
    ContextPtr m_ctx;
    DeviceGuard m_guard;
    cudaStreamCaptureMode m_mode = cudaStreamCaptureModeRelaxed;
    cudaStream_t m_stream = nullptr;
    cudaStream_t m_previous = nullptr;
};

CompletedRelease::CompletedRelease(ContextPtr ctx) : m_ctx(std::move(ctx)), m_guard(m_ctx->device())
{
    check(cudaThreadExchangeStreamCaptureMode(&m_mode), "cudaThreadExchangeStreamCaptureMode");
    const cudaError_t result = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    if (result != cudaSuccess)
    {
        (void)cudaThreadExchangeStreamCaptureMode(&m_mode);
        check(result, "cudaStreamCreateWithFlags");
    }
    m_previous = std::exchange(completed_release_stream, m_stream);
}

CompletedRelease::~CompletedRelease()
{
    (void)cudaStreamSynchronize(m_stream);
    completed_release_stream = m_previous;
    // All frees finish before the last context reference can destroy its pool.
    m_ctx.reset();
    (void)cudaStreamDestroy(m_stream);
    (void)cudaThreadExchangeStreamCaptureMode(&m_mode);
}
}  // namespace

namespace detail
{
bool completed_release_active() noexcept { return completed_release_stream != nullptr; }

void release_completed(std::unique_ptr<SuiteExpander>& expander, const Event& ready)
{
    if (!expander)
        return;
    // Only the workspace worker enters this scope. The expander exclusively owns every device object it destroys,
    // all graph launches have completed, and ready covers eager work, so frees need no other stream ordering.
    // Relaxed capture mode permits cleanup while unrelated resources are being captured on another thread.
    CompletedRelease release(expander->context());
    ready.synchronize();
    expander.reset();
}
}  // namespace detail

// ------------------------------------------------------------------------------------------------ LastUse

LastUse::LastUse(int device) : m_device(device)
{
    DeviceGuard g(device);
    check(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
}

LastUse::~LastUse()
{
    if (m_event)
        (void)cudaEventDestroy(m_event);
}

void LastUse::record(cudaStream_t s)
{
    std::lock_guard lock(m_mutex);
    DeviceGuard g(m_device);
    std::erase_if(m_captures, [](const auto& c) { return c->complete.load(std::memory_order_acquire); });
    auto capture = detail::capture_lifetime(s);
    if (capture)
        m_captures.push_back(capture);
    check(cudaEventRecord(m_event, s), "cudaEventRecord (last use)");
    m_record_capture = std::move(capture);
    m_recorded = true;
}

void LastUse::snapshot(cudaStream_t stream, std::vector<std::shared_ptr<detail::CaptureLifetime>>& captures) const
{
    std::lock_guard lock(m_mutex);
    for (const auto& c : m_captures)
        if (!c->complete.load(std::memory_order_acquire))
            captures.push_back(c);
    // A captured record completes with its graph's lifetime. Ordinary records are copied into stream order while
    // locked, so a subsequent record cannot change the retirement dependency.
    if (m_recorded && !m_record_capture)
        check(cudaStreamWaitEvent(stream, m_event, 0), "cudaStreamWaitEvent (last use snapshot)");
}

bool LastUse::recorded() const noexcept
{
    std::lock_guard lock(m_mutex);
    return m_recorded;
}

void LastUse::wait_on(cudaStream_t s) const noexcept
{
    std::lock_guard lock(m_mutex);
    if (m_recorded)
        (void)cudaStreamWaitEvent(s, m_event, 0);
    (void)cudaGetLastError();
}

PinnedBuffer::PinnedBuffer(u64 bytes) : m_bytes(bytes)
{
    if (bytes)
        check(cudaMallocHost(&m_ptr, bytes), "cudaMallocHost");
}

PinnedBuffer::~PinnedBuffer()
{
    if (m_ptr)
    {
        RelaxedCaptureScope capture_mode;
        (void)cudaFreeHost(m_ptr);
    }
}

PinnedLease::~PinnedLease() { give_back(); }

PinnedLease& PinnedLease::operator=(PinnedLease&& o) noexcept
{
    if (this != &o)
    {
        give_back();
        m_ctx = std::move(o.m_ctx);
        m_buffers = std::move(o.m_buffers);
    }
    return *this;
}

void PinnedLease::give_back() noexcept
{
    if (!m_ctx)
        return;
    constexpr usize keep = 64;  // buffers the context caches at most
    try
    {
        std::lock_guard lock(m_ctx->m_pinned_mutex);
        for (PinnedBuffer& b : m_buffers)
            if (b.data() && m_ctx->m_pinned_cache.size() < keep)
                m_ctx->m_pinned_cache.push_back(std::move(b));
    }
    catch (...)
    {
        // (allocation failure: the buffers are freed instead)
    }
    m_buffers.clear();
    m_ctx.reset();
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& o) noexcept
{
    if (this != &o)
    {
        if (m_ptr)
        {
            RelaxedCaptureScope capture_mode;
            (void)cudaFreeHost(m_ptr);
        }
        m_ptr = std::exchange(o.m_ptr, nullptr);
        m_bytes = std::exchange(o.m_bytes, 0);
    }
    return *this;
}

// ------------------------------------------------------------------------------------------------ GraphExec

GraphExec::~GraphExec()
{
    RelaxedCaptureScope capture_mode;
    if (m_exec)
        (void)cudaGraphExecDestroy(m_exec);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept
{
    RelaxedCaptureScope capture_mode;
    if (this != &o)
    {
        if (m_exec)
            (void)cudaGraphExecDestroy(m_exec);
        m_exec = std::exchange(o.m_exec, nullptr);
    }
    return *this;
}

void GraphExec::begin_capture(cudaStream_t s) { check(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture"); }

void GraphExec::abandon_capture(cudaStream_t s) noexcept
{
    cudaGraph_t g = nullptr;
    (void)cudaStreamEndCapture(s, &g);
    if (g)
        (void)cudaGraphDestroy(g);
    (void)cudaGetLastError();  // (the capture's error, if any, is the caller's exception)
}

GraphExec GraphExec::end_capture(cudaStream_t s)
{
    cudaGraph_t g = nullptr;
    check(cudaStreamEndCapture(s, &g), "cudaStreamEndCapture");
    GraphExec out;
    const cudaError_t e = cudaGraphInstantiate(&out.m_exec, g, 0);
    (void)cudaGraphDestroy(g);
    check(e, "cudaGraphInstantiate");
    return out;
}

unsigned long long GraphExec::begin_while(cudaStream_t s, cudaGraph_t& g)
{
    check(cudaGraphCreate(&g, 0), "cudaGraphCreate");
    try
    {
        cudaGraphConditionalHandle handle{};
        check(cudaGraphConditionalHandleCreate(&handle, g, 1, cudaGraphCondAssignDefault), "cudaGraphConditionalHandleCreate");
        cudaGraphNodeParams p{};
        p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = handle;
        p.conditional.type = cudaGraphCondTypeWhile;
        p.conditional.size = 1;
        cudaGraphNode_t node = nullptr;
        check(cudaGraphAddNode(&node, g, nullptr, nullptr, 0, &p), "cudaGraphAddNode (while)");
        check(cudaStreamBeginCaptureToGraph(s, p.conditional.phGraph_out[0], nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed),
              "cudaStreamBeginCaptureToGraph");
        return static_cast<unsigned long long>(handle);
    }
    catch (...)
    {
        (void)cudaGraphDestroy(g);
        g = nullptr;
        throw;
    }
}

void GraphExec::abandon_while(cudaStream_t s, cudaGraph_t g) noexcept
{
    cudaGraph_t body = nullptr;
    (void)cudaStreamEndCapture(s, &body);  // (the body belongs to g)
    if (g)
        (void)cudaGraphDestroy(g);
    (void)cudaGetLastError();
}

GraphExec GraphExec::end_while(cudaStream_t s, cudaGraph_t g)
{
    cudaGraph_t body = nullptr;
    const cudaError_t e = cudaStreamEndCapture(s, &body);
    GraphExec out;
    const cudaError_t i = e == cudaSuccess ? cudaGraphInstantiate(&out.m_exec, g, 0) : e;
    (void)cudaGraphDestroy(g);
    check(e, "cudaStreamEndCapture (while)");
    check(i, "cudaGraphInstantiate (while)");
    return out;
}

namespace
{
/// Whether the environment variable `name` is set to "0".
bool env_off(const char* name)
{
    const char* e = std::getenv(name);
    return e && std::string_view(e) == "0";
}
}  // namespace

bool GraphExec::enabled() noexcept
{
    static const bool on = !env_off("MYMYR_CUDA_GRAPHS");
    return on;
}

bool GraphExec::loops_enabled() noexcept
{
    static const bool on = enabled() && !env_off("MYMYR_CUDA_DEVICE_LOOPS");
    return on;
}

void GraphExec::launch(cudaStream_t s) const
{
    if (!m_exec)
        throw std::logic_error("mymyr: GraphExec::launch: no graph");
    check(cudaGraphLaunch(m_exec, s), "cudaGraphLaunch");
}
}  // namespace mymyr::cuda
