#pragma once
// Internal: TaskData -> plan::Compiled (static tables, matchers, canonical layout, axioms, goal).

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/task/plan.hpp"
#include "mymyr/task/task.hpp"

namespace mymyr::detail
{
/// Throws std::invalid_argument for a predicate or function of arity above 64.
void check_supported(const formalism::TaskData& t);

void compile_task(const formalism::TaskData& t, const TaskOptions& options, plan::Compiled& out);

/// Specializes a schema's resolved constraints for a prefix and witness search. depth == arity asks for existence.
[[nodiscard]] plan::Matcher compile_mask(const plan::Compiled& compiled, u32 schema, u32 depth);
/// The schema's mask matcher of `entry` (0: existence, d + 1: parameter d), compiled on first use and cached in the
/// schema's plan (thread-safe).
[[nodiscard]] const plan::Matcher& mask_matcher(const plan::Compiled& compiled, u32 schema, u32 entry);
}  // namespace mymyr::detail
