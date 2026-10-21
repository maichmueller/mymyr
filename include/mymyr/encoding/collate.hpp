#pragma once

#include "mymyr/rl/task_arrays.hpp"

#include <memory>
#include <span>

namespace mymyr::encoding
{
/// Concatenates graphs in batch order, rebuilding offsets in the batches' layout and shifting entity ids.
/// Entity references remain graph-local. Inputs must be valid batches of one schema, layout, index type and
/// repeated-static representation. Empty batches are allowed; an empty list is refused because it has no schema.
/// Capacity tails are ignored. Throws std::invalid_argument for incompatible metadata and std::overflow_error
/// before writing indices when a combined total does not fit their type.
[[nodiscard]] std::shared_ptr<const rl::ArrayBundle> collate(
    std::span<const std::shared_ptr<const rl::ArrayBundle>> batches);
}
