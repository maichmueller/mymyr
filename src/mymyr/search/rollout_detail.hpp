#pragma once
// Internals of Rollout IW(1) shared with the atomic-goal portfolio.

#include "mymyr/search/rollout_iw.hpp"

namespace mymyr::search::detail
{
/// rollout_iw() with its observer split: `hot` receives on_expand / on_generate / on_prune / on_transition /
/// on_progress, `root` the lifecycle events (on_start, on_solution, on_end); either may be null. The options'
/// control.observer is ignored.
RolloutIwResult run_rollout_iw(const Task& task, const RolloutIwOptions& options, SearchObserver* hot, SearchObserver* root);
}  // namespace mymyr::search::detail
