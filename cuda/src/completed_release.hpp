#pragma once

#include "mymyr/cuda/runtime.hpp"

namespace mymyr::cuda
{
class SuiteExpander;

namespace detail
{
/// Destroys an exclusive expander after its captured launches have completed, waiting for its uncaptured warm-up
/// event. On failure the expander remains owned by the caller.
void release_completed(std::unique_ptr<SuiteExpander>& expander, const Event& ready);

/// Whether buffer destruction on this thread can bypass the original allocation streams.
[[nodiscard]] bool completed_release_active() noexcept;
}  // namespace detail
}  // namespace mymyr::cuda
