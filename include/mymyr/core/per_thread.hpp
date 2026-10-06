#pragma once
// PerThread<W>: lazily created per-thread mutable scratch owned by an immutable object. A Task owns one
// PerThread<Workspace>; every thread that touches the task gets its own workspace on first use and then reaches it
// without locks.
//
// Lifetime rules, so thousands of short-lived tasks (multi-instance RL) and churning thread pools do not leak:
//   - the owner holds every workspace it created and frees them all when it dies;
//   - each thread keeps a small thread_local cache of (uid, weak anchor, W*). Uids come from next_uid() and are never
//     reused, so a cache hit on a live owner is always correct; entries of dead owners are pruned on the next miss;
//   - when a thread exits, its cached workspaces are returned to their (still alive) owners' free lists and reused
//     by the next new thread.
// local() must not race with the owner's destructor (the owner is alive while it is being used).

#include "mymyr/core/memory.hpp"
#include "mymyr/core/types.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace mymyr
{
namespace detail
{
struct PerThreadAnchor
{
    std::mutex mutex;
    std::vector<void*> free;  // workspaces released by exited threads, ready for reuse
};

struct PerThreadCacheEntry
{
    u64 uid;
    std::weak_ptr<PerThreadAnchor> anchor;
    void* workspace;
};

struct PerThreadCache
{
    std::vector<PerThreadCacheEntry> entries;

    ~PerThreadCache()
    {
        for (auto& e : entries)
            if (auto a = e.anchor.lock())
            {
                std::lock_guard lock(a->mutex);
                a->free.push_back(e.workspace);
            }
    }

    static PerThreadCache& get()
    {
        thread_local PerThreadCache cache;
        return cache;
    }
};
}  // namespace detail

template<class W>
class PerThread
{
public:
    PerThread() : m_uid(next_uid()), m_anchor(std::make_shared<detail::PerThreadAnchor>()) {}
    PerThread(const PerThread&) = delete;
    PerThread& operator=(const PerThread&) = delete;

    ~PerThread()
    {
        // Expire the anchor first: exiting threads then stop returning pointers into m_all.
        m_anchor.reset();
    }

    /// The calling thread's instance, created by make() (returning std::unique_ptr<W>) on first use.
    template<class Make>
    W& local(Make&& make)
    {
        auto& cache = detail::PerThreadCache::get();
        for (const auto& e : cache.entries)
            if (e.uid == m_uid)
                return *static_cast<W*>(e.workspace);
        return insert_slow(cache, static_cast<Make&&>(make));
    }

    [[nodiscard]] u64 uid() const noexcept { return m_uid; }
    /// Number of instances ever created (not the number of live threads).
    [[nodiscard]] usize created() const
    {
        std::lock_guard lock(m_anchor->mutex);
        return m_all.size();
    }

private:
    template<class Make>
    W& insert_slow(detail::PerThreadCache& cache, Make&& make)
    {
        std::erase_if(cache.entries, [](const detail::PerThreadCacheEntry& e) { return e.anchor.expired(); });
        W* w = nullptr;
        {
            std::lock_guard lock(m_anchor->mutex);
            if (!m_anchor->free.empty())
            {
                w = static_cast<W*>(m_anchor->free.back());
                m_anchor->free.pop_back();
            }
        }
        if (!w)
        {
            std::unique_ptr<W> fresh = make();
            w = fresh.get();
            std::lock_guard lock(m_anchor->mutex);
            m_all.push_back(std::move(fresh));
        }
        cache.entries.push_back({m_uid, m_anchor, w});
        return *w;
    }

    u64 m_uid;
    std::shared_ptr<detail::PerThreadAnchor> m_anchor;
    std::vector<std::unique_ptr<W>> m_all;  // guarded by m_anchor->mutex
};
}  // namespace mymyr
