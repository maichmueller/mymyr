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
}  // namespace mymyr::detail
