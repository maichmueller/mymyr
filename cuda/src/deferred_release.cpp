#include "capture_scope.hpp"
#include "deferred_release.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <thread>
#include <utility>

namespace mymyr::cuda::detail
{
namespace
{
struct Release
{
    ContextPtr ctx;
    void* pointer = nullptr;
    std::vector<Event> events;
    std::vector<CapturePtr> captures;
    std::shared_ptr<void> owner;
    void (*on_failure)() noexcept = nullptr;

    bool ready() const noexcept
    {
        return std::ranges::all_of(captures, [](const auto& c) { return c->complete.load(std::memory_order_acquire); });
    }
};

class Reclaimer
{
public:
    static Reclaimer& get()
    {
        // CUDA may release graph references after C++ static destructors have run.
        static auto* instance = new Reclaimer;
        return *instance;
    }

    void submit(std::unique_ptr<Release> release)
    {
        {
            std::lock_guard lock(m_mutex);
            m_pending.push_back(std::move(release));
        }
        m_changed.notify_all();
    }

    void complete(CaptureLifetime& lifetime) noexcept
    {
        {
            std::lock_guard lock(m_mutex);
            lifetime.complete.store(true, std::memory_order_release);
        }
        m_changed.notify_all();
    }

    void drain(const DeviceContext& ctx)
    {
        std::unique_lock lock(m_mutex);
        const auto ready = [&] {
            return std::ranges::find_if(m_pending, [&](const auto& r) { return r->ctx.get() == &ctx && r->ready(); });
        };
        if (s_current == this)
        {
            // Destructors may drain their context. The worker completes ready entries inline, excluding the
            // destruction already on its stack, because those entries cannot be serviced by another thread.
            for (auto it = ready(); it != m_pending.end(); it = ready())
            {
                auto release = take(it);
                lock.unlock();
                process(std::move(release));
                lock.lock();
            }
            return;
        }
        m_changed.wait(lock, [&] {
            return std::ranges::find(m_active, &ctx) == m_active.end() && ready() == m_pending.end();
        });
    }

private:
    std::unique_ptr<Release> take(std::vector<std::unique_ptr<Release>>::iterator it)
    {
        m_active.push_back((*it)->ctx.get());
        auto release = std::move(*it);
        m_pending.erase(it);
        return release;
    }

    void process(std::unique_ptr<Release> release)
    {
        // Keep the context alive through owner destruction and completion bookkeeping, including nested drains.
        auto ctx = release->ctx;
        try
        {
            RelaxedCaptureScope capture_mode;
            DeviceGuard guard(ctx->device());
            Stream stream;
            for (const auto& e : release->events)
                e.wait_on(stream);
            if (release->pointer)
                check(cudaFreeAsync(release->pointer, stream), "cudaFreeAsync (deferred release)");
            stream.synchronize();
            release.reset();
        }
        catch (const std::exception& e)
        {
            // A failed device may still read these resources; retain them for driver teardown.
            std::fprintf(stderr, "mymyr: CUDA: deferred release retained resources: %s\n", e.what());
            if (release->on_failure)
                release->on_failure();
            (void)release.release();
        }
        {
            std::lock_guard lock(m_mutex);
            m_active.pop_back();
        }
        m_changed.notify_all();
    }

    Reclaimer()
    {
        std::thread([this] {
            s_current = this;
            for (;;)
            {
                std::unique_ptr<Release> release;
                {
                    std::unique_lock lock(m_mutex);
                    auto ready = [&] { return std::ranges::find_if(m_pending, [](const auto& r) { return r->ready(); }); };
                    m_changed.wait(lock, [&] { return ready() != m_pending.end(); });
                    release = take(ready());
                }
                process(std::move(release));
            }
        }).detach();
    }

    std::mutex m_mutex;
    std::condition_variable m_changed;
    std::vector<std::unique_ptr<Release>> m_pending;
    std::vector<const DeviceContext*> m_active;
    inline static thread_local Reclaimer* s_current = nullptr;
};

std::atomic<Reclaimer*> reclaimer{nullptr};

Reclaimer& start_reclaimer()
{
    auto& worker = Reclaimer::get();
    reclaimer.store(&worker, std::memory_order_release);
    return worker;
}
}  // namespace

CapturePtr capture_lifetime(cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    cudaGraph_t graph = nullptr;
    check(cudaStreamGetCaptureInfo(stream, &status, &id, &graph), "cudaStreamGetCaptureInfo (release)");
    if (status == cudaStreamCaptureStatusNone)
        return {};
    if (status != cudaStreamCaptureStatusActive)
        check(cudaErrorStreamCaptureInvalidated, "capture lifetime");
    start_reclaimer();
    auto lifetime = std::make_shared<CaptureLifetime>();
    lifetime->id = id;
    auto holder = std::make_unique<CapturePtr>(lifetime);
    cudaUserObject_t object = nullptr;
    check(cudaUserObjectCreate(&object, holder.get(), [](void* p) {
        std::unique_ptr<CapturePtr> held(static_cast<CapturePtr*>(p));
        reclaimer.load(std::memory_order_acquire)->complete(**held);
    }, 1, cudaUserObjectNoDestructorSync), "cudaUserObjectCreate (release)");
    (void)holder.release();
    const auto retained = cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove);
    if (retained != cudaSuccess)
        (void)cudaUserObjectRelease(object, 1);
    check(retained, "cudaGraphRetainUserObject (release)");
    return lifetime;
}

void release_deferred(ContextPtr ctx, void* pointer, std::vector<Event> events, std::vector<CapturePtr> captures)
{
    auto release = std::make_unique<Release>();
    release->ctx = std::move(ctx);
    release->pointer = pointer;
    release->events = std::move(events);
    release->captures = std::move(captures);
    start_reclaimer().submit(std::move(release));
}

void release_after(ContextPtr ctx, std::shared_ptr<void> owner, void (*on_failure)() noexcept)
{
    DeviceGuard guard(ctx->device());
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(ctx->stream(), &status), "cudaStreamIsCapturing");
    if (status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("mymyr: CUDA: release_after needs a non-capturing context stream");
    Event ready;
    ready.record(ctx->stream());
    auto release = std::make_unique<Release>();
    release->ctx = std::move(ctx);
    release->owner = std::move(owner);
    release->on_failure = on_failure;
    release->events.push_back(std::move(ready));
    start_reclaimer().submit(std::move(release));
}

void drain_releases(const DeviceContext& ctx)
{
    if (auto* worker = reclaimer.load(std::memory_order_acquire))
        worker->drain(ctx);
}
}  // namespace mymyr::cuda::detail
