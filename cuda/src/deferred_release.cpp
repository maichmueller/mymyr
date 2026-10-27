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
    std::vector<std::shared_ptr<const LastUse>> lasts;
    std::vector<CapturePtr> captures;
    std::shared_ptr<void> owner;

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
        m_changed.wait(lock, [&] {
            return m_active != &ctx && std::ranges::none_of(m_pending, [&](const auto& r) {
                return r->ctx.get() == &ctx && r->ready();
            });
        });
    }

private:
    Reclaimer()
    {
        std::thread([this] {
            for (;;)
            {
                std::unique_ptr<Release> release;
                {
                    std::unique_lock lock(m_mutex);
                    auto ready = [&] { return std::ranges::find_if(m_pending, [](const auto& r) { return r->ready(); }); };
                    m_changed.wait(lock, [&] { return ready() != m_pending.end(); });
                    const auto it = ready();
                    release = std::move(*it);
                    m_pending.erase(it);
                    m_active = release->ctx.get();
                }
                try
                {
                    RelaxedCaptureScope capture_mode;
                    DeviceGuard guard(release->ctx->device());
                    Stream stream;
                    for (const auto& e : release->events)
                        e.wait_on(stream);
                    for (const auto& last : release->lasts)
                        last->wait_on(stream);
                    if (release->pointer)
                        check(cudaFreeAsync(release->pointer, stream), "cudaFreeAsync (deferred release)");
                    stream.synchronize();
                    release.reset();
                }
                catch (const std::exception& e)
                {
                    // A failed device may still read these resources; retain them for driver teardown.
                    std::fprintf(stderr, "mymyr: CUDA: deferred release retained resources: %s\n", e.what());
                    (void)release.release();
                }
                {
                    std::lock_guard lock(m_mutex);
                    m_active = nullptr;
                }
                m_changed.notify_all();
            }
        }).detach();
    }

    std::mutex m_mutex;
    std::condition_variable m_changed;
    std::vector<std::unique_ptr<Release>> m_pending;
    const DeviceContext* m_active = nullptr;
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

void release_deferred(ContextPtr ctx, void* pointer, std::vector<Event> events,
                      std::vector<std::shared_ptr<const LastUse>> lasts, std::vector<CapturePtr> captures)
{
    auto release = std::make_unique<Release>();
    release->ctx = std::move(ctx);
    release->pointer = pointer;
    release->events = std::move(events);
    release->lasts = std::move(lasts);
    release->captures = std::move(captures);
    start_reclaimer().submit(std::move(release));
}

void release_deferred(ContextPtr ctx, std::shared_ptr<void> owner, Event ready)
{
    auto release = std::make_unique<Release>();
    release->ctx = std::move(ctx);
    release->owner = std::move(owner);
    release->events.push_back(std::move(ready));
    start_reclaimer().submit(std::move(release));
}

void drain_releases(const DeviceContext& ctx)
{
    if (auto* worker = reclaimer.load(std::memory_order_acquire))
        worker->drain(ctx);
}
}  // namespace mymyr::cuda::detail
