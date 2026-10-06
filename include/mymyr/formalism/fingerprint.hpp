#pragma once
// Content fingerprint of a normalized task: a 64-bit hash of everything that determines the
// compiled task and its atom layout (the text format of text_format.hpp, which covers predicates, initial atoms, goal,
// schemas, axioms and numerics) plus the names the text format leaves out (domain, problem, objects, types, parameter
// counts written in the PDDL). Two TaskData with equal fingerprints build identical Tasks; a pickled State carries its
// task's fingerprint and is checked against the task it is loaded into.

#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"

namespace mymyr::formalism
{
[[nodiscard]] u64 fingerprint(const TaskData& t);
}  // namespace mymyr::formalism
