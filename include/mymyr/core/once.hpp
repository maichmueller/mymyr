#pragma once
// Once: a one-time initialization whose initializer may throw, with the semantics of std::call_once but without
// pthread_once. libstdc++ runs std::call_once's callable inside pthread_once, and an exception that unwinds through
// pthread_once's C frames aborts the process where those frames carry no unwind information (as in some Python
// processes). Use Once for every lazy initialization that can throw (std::bad_alloc included).

#include <atomic>
#include <mutex>
#include <utility>

namespace mymyr
{
class Once
{
public:
    Once() = default;
    Once(const Once&) = delete;
    Once& operator=(const Once&) = delete;

    /// Runs `init` unless a call has completed it. Concurrent callers wait for the running call. As with
    /// std::call_once, an `init` that throws leaves the initialization undone: the exception reaches this caller and
    /// the next call runs `init` again. Calling it again from inside `init` deadlocks.
    template<class F>
    void call(F&& init)
    {
        if (m_done.load(std::memory_order_acquire))
            return;
        std::lock_guard lock(m_mutex);
        if (m_done.load(std::memory_order_relaxed))
            return;
        std::forward<F>(init)();
        m_done.store(true, std::memory_order_release);
    }

    /// Whether a call has completed the initialization.
    [[nodiscard]] bool done() const noexcept { return m_done.load(std::memory_order_acquire); }

private:
    std::atomic<bool> m_done{false};
    std::mutex m_mutex;
};
}  // namespace mymyr
