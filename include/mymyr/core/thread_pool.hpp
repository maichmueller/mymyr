#pragma once
// ThreadPool: persistent worker threads that sleep between jobs (for batched calls from user code, e.g. rl::expand).
//
// Team (core/team.hpp) spins between the many fork/joins of one layer-synchronous search and is created per search. A
// pool that user code holds across calls must not burn CPU while idle, so its workers block on a condition variable.
// run(f) calls f(t) for every member t in [0, size()), the caller being member 0, and returns when all are done.
// Concurrent run() calls from different threads are serialized (one job at a time); run() is not reentrant.
// The pool is an object the caller owns; there is no global pool.

#include "mymyr/core/types.hpp"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace mymyr
{
class ThreadPool
{
public:
    explicit ThreadPool(u32 threads) : m_T(threads == 0 ? 1 : threads)
    {
        m_threads.reserve(m_T - 1);
        for (u32 i = 1; i < m_T; ++i)
            m_threads.emplace_back([this, i] { loop(i); });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool()
    {
        {
            std::lock_guard lock(m_mutex);
            m_quit = true;
            ++m_gen;
        }
        m_wake.notify_all();
        for (auto& t : m_threads)
            t.join();
    }

    [[nodiscard]] u32 size() const noexcept { return m_T; }

    /// Runs f(t) on every member; rethrows the first exception a member threw. Serialized across callers.
    void run(const std::function<void(u32)>& f)
    {
        std::lock_guard serial(m_run_mutex);
        if (m_T == 1)
        {
            f(0);
            return;
        }
        {
            std::lock_guard lock(m_mutex);
            m_job = &f;
            m_error = nullptr;
            m_pending = m_T - 1;
            ++m_gen;
        }
        m_wake.notify_all();
        call(f, 0);
        std::unique_lock lock(m_mutex);
        m_done.wait(lock, [this] { return m_pending == 0; });
        m_job = nullptr;
        if (m_error)
            std::rethrow_exception(std::exchange(m_error, nullptr));
    }

    /// Static partition of [0, n) for member t of T.
    [[nodiscard]] static std::pair<u64, u64> slice(u64 n, u32 t, u32 T) { return {n * t / T, n * (t + 1) / T}; }

private:
    void call(const std::function<void(u32)>& f, u32 t)
    {
        try
        {
            f(t);
        }
        catch (...)
        {
            std::lock_guard lock(m_mutex);
            if (!m_error)
                m_error = std::current_exception();
        }
    }
    void loop(u32 t)
    {
        u64 seen = 0;
        for (;;)
        {
            const std::function<void(u32)>* job = nullptr;
            {
                std::unique_lock lock(m_mutex);
                m_wake.wait(lock, [&] { return m_gen != seen; });
                seen = m_gen;
                if (m_quit)
                    return;
                job = m_job;
            }
            call(*job, t);
            bool last = false;
            {
                std::lock_guard lock(m_mutex);
                last = --m_pending == 0;
            }
            if (last)
                m_done.notify_one();
        }
    }

    u32 m_T;
    std::vector<std::thread> m_threads;
    std::mutex m_run_mutex;  // one job at a time
    std::mutex m_mutex;      // guards everything below
    std::condition_variable m_wake, m_done;
    u64 m_gen = 0;
    u32 m_pending = 0;
    bool m_quit = false;
    const std::function<void(u32)>* m_job = nullptr;
    std::exception_ptr m_error;
};
}  // namespace mymyr
