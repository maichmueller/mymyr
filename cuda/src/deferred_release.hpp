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
/// Destroys an owner on a host worker after the context stream's preceding work. The stream must not be capturing.
/// The owner and context stay alive through completion; the owner's destructor may call CUDA and drain releases.
/// Failed completion retains the owner and calls on_failure from the worker.
void release_after(ContextPtr ctx, std::shared_ptr<void> owner, void (*on_failure)() noexcept = nullptr);
/// Waits for this context's ready releases, without waiting for a live graph's lifetime fence.
void drain_releases(const DeviceContext& ctx);
}  // namespace mymyr::cuda::detail
