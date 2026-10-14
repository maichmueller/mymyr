#pragma once
// mymyr.rl.ThreadPool: the pool of worker threads that user code holds across calls (rl.expand and the relational
// encoder take it as `pool=`).

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/expand.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <mutex>

namespace mymyr::python
{
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
