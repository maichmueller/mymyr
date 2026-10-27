#pragma once
// mymyr.rl.ThreadPool: the pool of worker threads that user code holds across calls (rl.expand and the relational
// encoder take it as `pool=`).

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/expand.hpp"
#include "typing.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <mutex>

namespace mymyr::python
{
/// Validates a visited-table capacity; True chooses room for every state of a bounded episode.
inline u32 visited_capacity_arg(nanobind::handle value, u32 max_steps)
{
    namespace nb = nanobind;
    if (nb::isinstance<nb::bool_>(value))
    {
        if (!nb::cast<bool>(value))
            return 0;
        if (!max_steps || max_steps > (1u << 19) - 1)
            throw nb::value_error("mymyr: track_visited=True needs max_steps in [1, 524287]; pass an explicit capacity otherwise");
        u32 H = 2;
        while (H < 2 * (max_steps + 1))
            H *= 2;
        return H;
    }
    const u32 H = int_arg<u32>(value, "track_visited", 2, 1u << 20);
    if (H & (H - 1))
        throw nb::value_error("mymyr: track_visited must be a power of two in [2, 2^20]");
    return H;
}

struct PyPool
{
    explicit PyPool(u32 threads) : pool(std::make_unique<ThreadPool>(threads)) {}
    std::unique_ptr<ThreadPool> pool;
    rl::ExpandScratch scratch;
    std::mutex mutex;  // one batch at a time (the scratch is shared by the pool's members)
};

/// The pool behind a `pool=` argument, or null for None; TypeError for anything else.
inline PyPool* pool_arg(nanobind::handle pool)
{
    if (pool.is_none())
        return nullptr;
    if (!nanobind::isinstance<PyPool>(pool))
        throw nanobind::type_error("mymyr: pool must be a mymyr.rl.ThreadPool");
    return nanobind::inst_ptr<PyPool>(pool);
}
}  // namespace mymyr::python
