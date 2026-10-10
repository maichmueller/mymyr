#pragma once
// Thread counts. Every parallel call takes a thread count from its caller (`threads`, `num_threads`): 0 means one
// thread per hardware thread, and a count above max_threads() is refused, so that a typo or a negative number cast to
// unsigned cannot start millions of threads. Oversubscription up to the bound stays possible (tests and small machines
// run more threads than cores).

#include "mymyr/core/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mymyr
{
/// std::thread::hardware_concurrency(), at least 1.
[[nodiscard]] inline u32 hardware_threads() noexcept { return std::max<u32>(1, std::thread::hardware_concurrency()); }

/// The largest thread count a parallel call accepts: max(64, 4 * hardware_threads()).
[[nodiscard]] inline u32 max_threads() noexcept { return std::max<u32>(64, 4 * hardware_threads()); }

/// `threads` worker threads, 0 meaning hardware_threads(). Throws std::invalid_argument naming `name` above
/// max_threads().
[[nodiscard]] inline u32 resolve_threads(u64 threads, std::string_view name = "threads")
{
    if (threads > max_threads())
        throw std::invalid_argument("mymyr: " + std::string(name) + "=" + std::to_string(threads) +
                                    " exceeds the limit of " + std::to_string(max_threads()) +
                                    " threads (max(64, 4 x the " + std::to_string(hardware_threads()) +
                                    " hardware threads)); pass 0 for one per hardware thread");
    return threads == 0 ? hardware_threads() : static_cast<u32>(threads);
}

/// Thrown when the operating system refuses to start a worker thread (out of memory or address space, a thread limit).
/// The threads already started have been stopped and joined.
struct ThreadStartError : std::runtime_error
{
    ThreadStartError(u32 started, u32 wanted, const std::exception& e)
        : std::runtime_error("mymyr: could not start worker thread " + std::to_string(started + 1) + " of " +
                             std::to_string(wanted) + " (" + e.what() + "); use fewer threads")
    {
    }
};

/// Appends n threads running f(t), t in [0, n), to `threads`; f must outlive them. If the system refuses one, calls
/// stop() (which must make the running ones return soon), joins and removes every thread of `threads`, and throws
/// ThreadStartError.
template<class F, class Stop>
void start_threads(std::vector<std::thread>& threads, u32 n, F& f, Stop&& stop)
{
    threads.reserve(threads.size() + n);
    for (u32 t = 0; t < n; ++t)
    {
        try
        {
            threads.emplace_back([&f, t] { f(t); });
        }
        catch (const std::exception& e)
        {
            stop();
            for (std::thread& th : threads)
                th.join();
            threads.clear();
            throw ThreadStartError(t, n, e);
        }
    }
}
}  // namespace mymyr
