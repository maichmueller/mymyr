// Device GBFS (include/mymyr/cuda/gbfs.hpp): the batched best-first driver of cuda/src/astar.cpp in greedy mode.

#include "mymyr/cuda/gbfs.hpp"

namespace mymyr::cuda
{
DeviceBestFirstResult gbfs(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options)
{
    return detail::best_first(std::move(ctx), std::move(task), options, true);
}
}  // namespace mymyr::cuda
