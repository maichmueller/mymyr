#pragma once

#include "mymyr/cuda/runtime.hpp"

#include <atomic>

namespace mymyr::cuda::detail
{
struct CaptureLifetime
{
    unsigned long long id = 0;
    std::atomic<bool> complete{false};
};
using CapturePtr = std::shared_ptr<CaptureLifetime>;

/// A fence released by the graph, its executable clones and their outstanding launches; null outside capture.
CapturePtr capture_lifetime(cudaStream_t stream);

/// Takes ownership of a free after every capture and event, on a private non-capturing stream.
void release_deferred(ContextPtr ctx, void* pointer, std::vector<Event> events, std::vector<CapturePtr> captures);
/// Releases a host owner on a worker once ready completes. Its destructor may call CUDA.
void release_deferred(ContextPtr ctx, std::shared_ptr<void> owner, Event ready, void (*on_failure)() noexcept);
/// Waits for this context's ready releases, without waiting for a live graph's lifetime fence.
void drain_releases(const DeviceContext& ctx);
}  // namespace mymyr::cuda::detail
