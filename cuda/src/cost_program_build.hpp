#pragma once
// The host side of the device action costs (mymyr/cuda/cost_program.hpp): builds the programs of instances'
// state-independent costs and uploads them.

#include "mymyr/cuda/cost_program.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <span>
#include <string>

namespace mymyr
{
class Task;
namespace heuristics
{
class ActionCosts;
}
}  // namespace mymyr

namespace mymyr::cuda::costs
{
/// The programs of a set of instances of one domain, or why the device cannot evaluate their costs.
struct Programs
{
    Program view;          // the device arrays (valid while `buf` lives)
    DeviceBuffer buf;      // every array in one allocation
    std::string why;       // non-empty: no programs (costs that depend on the state)
    bool integral = true;  // ActionCosts::integral of every instance
};

/// The programs of instances `tasks` with their costs (one per task; Unit, or TotalCost with state-independent costs,
/// else `why`; the instances have the same schemas and functions), uploaded on stream s (the call waits for it).
[[nodiscard]] Programs build(const ContextPtr& ctx, std::span<const Task* const> tasks,
                             std::span<const heuristics::ActionCosts* const> costs, cudaStream_t s);
}  // namespace mymyr::cuda::costs
