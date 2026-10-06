#pragma once
// Team: a fixed group of persistent threads with a spinning fork/join. run(f) calls f(t) on every member t in
// [0, T), with the caller as member 0, and returns when all are done. Layer-synchronous
// searches fork/join several times per layer, so waking sleeping threads would dominate; members spin briefly and
// then yield. Threads exit in the destructor.

#include "mymyr/core/types.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace mymyr
{
inline void cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

class Team
{
public:
    explicit Team(u32 threads) : m_T(threads == 0 ? 1 : threads)
    {
        for (u32 i = 1; i < m_T; ++i)
            m_threads.emplace_back([this, i] { loop(i); });
    }
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    ~Team()
    {
        m_quit.store(true, std::memory_order_release);
        m_gen.fetch_add(1, std::memory_order_release);
        for (auto& t : m_threads)
            t.join();
    }

    [[nodiscard]] u32 size() const noexcept { return m_T; }

    /// Runs f(t) on every member; rethrows the first exception a member threw.
    void run(const std::function<void(u32)>& f)
    {
        if (m_T == 1)
        {
            f(0);
            return;
        }
        m_job = &f;
        m_error = nullptr;
        m_done.store(0, std::memory_order_relaxed);
        m_gen.fetch_add(1, std::memory_order_release);
        call(f, 0);
        u32 spins = 0;
        while (m_done.load(std::memory_order_acquire) != m_T - 1)
            if (++spins > 2000)
                std::this_thread::yield();
            else
                cpu_relax();
        if (m_error)
            std::rethrow_exception(m_error);
    }

    /// Static partition of [0, n) for member t.
    [[nodiscard]] static std::pair<u64, u64> slice(u64 n, u32 t, u32 T)
    {
        return {n * t / T, n * (t + 1) / T};
    }

private:
    void call(const std::function<void(u32)>& f, u32 t)
    {
        try
        {
            f(t);
        }
        catch (...)
        {
            std::lock_guard lock(m_error_mutex);
            if (!m_error)
                m_error = std::current_exception();
        }
    }
    void loop(u32 t)
    {
        u64 seen = 0;
        for (;;)
        {
            u64 g;
            u32 spins = 0;
            while ((g = m_gen.load(std::memory_order_acquire)) == seen)
            {
                if (++spins > 2000)
                    std::this_thread::yield();
                else
                    cpu_relax();
            }
            seen = g;
            if (m_quit.load(std::memory_order_acquire))
                return;
            call(*m_job, t);
            m_done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    u32 m_T;
    std::vector<std::thread> m_threads;
    std::atomic<u64> m_gen{0};
    std::atomic<u32> m_done{0};
    std::atomic<bool> m_quit{false};
    const std::function<void(u32)>* m_job = nullptr;
    std::mutex m_error_mutex;
    std::exception_ptr m_error;
};
}  // namespace mymyr
