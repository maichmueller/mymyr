#pragma once
// Eager A* and eager GBFS with BestFirstOptions::expansion_batch > 1 (best_first.hpp); astar_eager and gbfs_eager
// run these for such options.

#include "mymyr/search/best_first.hpp"

namespace mymyr::search::bf
{
[[nodiscard]] BestFirstResult astar_eager_batched(const Task& task, const BestFirstOptions& options);
[[nodiscard]] BestFirstResult gbfs_eager_batched(const Task& task, const BestFirstOptions& options);
}  // namespace mymyr::search::bf
