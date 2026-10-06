#pragma once
// Device greedy best-first search with batched expansion:
//
//   mymyr::cuda::DeviceBestFirstOptions o;
//   o.search.heuristic.kind = mymyr::heuristics::Kind::FF;
//   o.batch = 1000;
//   mymyr::cuda::DeviceBestFirstResult r = mymyr::cuda::gbfs(ctx, task, o);
//
// Every step pops the `batch` open nodes with the smallest keys (h, g, first in first out: search::gbfs_eager's order)
// and expands them together, as device A* does (cuda/astar.hpp): no reopening, a state keeps the g of its first
// generation (the first candidate in pop and canonical order), and the search stops at the first new goal state of a
// chunk (in candidate order). The plan's cost is mimir's plan extraction (cheapest action per step), as the CPU's.
// A batch expands states the sequential search never reaches; stats.expanded reports the expansion overhead against
// search::gbfs_eager.
//
// The same options, statistics and refusals as cuda/astar.hpp.

#include "mymyr/cuda/astar.hpp"

namespace mymyr::cuda
{
/// GBFS on the device (see above). Throws std::invalid_argument for what it refuses.
[[nodiscard]] DeviceBestFirstResult gbfs(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options = {});
}  // namespace mymyr::cuda
