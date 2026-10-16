#pragma once
// The host arrays of a mymyr.rl.Expansion, for entry points that read an expansion's labels (the encoder's `actions`).

#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_table.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <vector>

namespace mymyr::python
{
struct ExpansionBuffers
{
    rl::Expansion view;               // the arrays of the expansion (read-only)
    u64 states = 0;                   // the expanded states
    std::vector<i32> task_ids;        // the instance of each expanded state
    rl::TaskTablePtr table;           // the table the expansion was made over (one domain)
    std::shared_ptr<void> keep;       // keeps the arrays alive
};

/// The buffers of `obj`, a host mymyr.rl.Expansion. Throws TypeError for any other object (a device expansion included)
/// and ValueError for an expansion over several domains.
ExpansionBuffers expansion_buffers(nb::handle obj);
}  // namespace mymyr::python
