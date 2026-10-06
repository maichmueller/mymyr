// The CUDA backend's runtime layer: contexts, pools, buffers, events (include/mymyr/cuda/runtime.hpp).

#include "mymyr/cuda/runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace mymyr::cuda
{
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
void Event::synchronize() const { check(cudaEventSynchronize(m_e), "cudaEventSynchronize"); }

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
    if (m_s)
    {
        (void)cudaStreamSynchronize(m_s);
        (void)cudaStreamDestroy(m_s);
    }
}

void Stream::synchronize() const { check(cudaStreamSynchronize(m_s), "cudaStreamSynchronize"); }

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
            (void)cudaFreeHost(p);
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
            (void)cudaFree(p);
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
    int prev = -1;
    const bool got = cudaGetDevice(&prev) == cudaSuccess;
    if (got && prev != m_device)
        (void)cudaSetDevice(m_device);
    if (cudaFreeAsync(p, s ? s : m_stream) != cudaSuccess)
        (void)cudaGetLastError();  // runtime unloading at exit: the driver reclaims everything
    if (got && prev != m_device)
        (void)cudaSetDevice(prev);
}

PoolUsage DeviceContext::usage() const
{
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
    DeviceGuard g(m_device);
    check(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
    check(cudaStreamSynchronize(m_copy), "cudaStreamSynchronize");
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
    if (!m_ptr)
        return;
    cudaStream_t fs = m_stream;
    {
        std::lock_guard lock(*m_mutex);
        int prev = -1;
        const bool got = cudaGetDevice(&prev) == cudaSuccess;
        if (got && prev != m_ctx->device())
            (void)cudaSetDevice(m_ctx->device());
        // the stream that frees: the releasing call's, the context's (a tracked LastUse: the allocation stream may be
        // gone), or the allocation stream
        fs = release ? release : m_free_live ? m_ctx->stream() : m_stream;
        for (cudaStream_t s : m_users)
            if (s != fs)
                order_after(fs, s);  // (the users must exist now)
        m_users.clear();
        for (const auto& l : m_lasts)
            l->wait_on(fs);
        m_lasts.clear();
        if (!m_free_live && fs != m_stream)
            order_after(fs, m_stream);
        if (got && prev != m_ctx->device())
            (void)cudaSetDevice(prev);
    }
    m_ctx->deallocate(m_ptr, fs);
    m_ptr = nullptr;
    m_bytes = 0;
    m_free_live = false;
}

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
    check(cudaEventRecord(m_event, s), "cudaEventRecord (last use)");
    m_recorded = true;
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
        (void)cudaFreeHost(m_ptr);
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
            (void)cudaFreeHost(m_ptr);
        m_ptr = std::exchange(o.m_ptr, nullptr);
        m_bytes = std::exchange(o.m_bytes, 0);
    }
    return *this;
}

// ------------------------------------------------------------------------------------------------ GraphExec

GraphExec::~GraphExec()
{
    if (m_exec)
        (void)cudaGraphExecDestroy(m_exec);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept
{
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
