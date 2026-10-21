#pragma once
// The semi-naive skipping data of the device's axiom evaluation (lifted::Derived::reads), from a compiled task.

#include "mymyr/core/types.hpp"

#include <vector>

namespace mymyr::plan
{
struct Compiled;
}

namespace mymyr::cuda::detail
{
/// Per axiom in plan_axiom order (strata, then their axioms): {the predicates of its stratum its body reads, as bits of
/// their index in the stratum's head list; the bit of its head} (lifted::Derived::reads). All bits when a stratum has
/// more than 64 head predicates.
[[nodiscard]] std::vector<u64> axiom_reads(const plan::Compiled& compiled);
}
