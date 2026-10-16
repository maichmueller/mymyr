#pragma once
// The mymyr.datasets wrappers that the CUDA bindings share (datasets_bindings.cpp; cuda_datasets_bindings.cpp makes
// device state spaces and hands out their host copies as mymyr.datasets.StateSpace).

#include "py_task.hpp"

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/search/control.hpp"

#include <nanobind/nanobind.h>

#include <vector>

namespace mymyr::python
{
/// A mymyr.datasets.StateSpace.
struct PyStateSpace
{
    datasets::StateSpacePtr space;
    Owner owner;  // the Task (or handle) the space was generated from: States and Actions are owned by it
};

/// The unit goal distances of a mymyr.cuda.DeviceStateSpace, downloaded: true (and `out` filled) if `obj` is one. The
/// CUDA bindings install it (null in CPU builds); StateSpaceSampler takes device spaces through it.
using DeviceDistancesHook = bool (*)(nb::handle obj, std::vector<i32>& out);
void set_device_distances_hook(DeviceDistancesHook hook) noexcept;

/// A generation's outcome as a search Status (mymyr.search.Status): a complete space is EXHAUSTED (every reachable
/// state was expanded), the budgets give OUT_OF_STATES and OUT_OF_TIME.
[[nodiscard]] inline search::SearchStatus search_status(datasets::StateSpaceStatus s) noexcept
{
    switch (s)
    {
        case datasets::StateSpaceStatus::Ok: return search::SearchStatus::Exhausted;
        case datasets::StateSpaceStatus::OutOfStates: return search::SearchStatus::OutOfStates;
        case datasets::StateSpaceStatus::Timeout: return search::SearchStatus::OutOfTime;
        case datasets::StateSpaceStatus::Unsolvable: return search::SearchStatus::Unsolvable;
    }
    return search::SearchStatus::Failed;
}

/// The Statistics of a generation: the states it stored and its wall time (it counts nothing else).
[[nodiscard]] inline search::SearchStatistics generation_stats(u64 states, f64 seconds) noexcept
{
    search::SearchStatistics s;
    s.states = states;
    s.seconds = seconds;
    return s;
}
}  // namespace mymyr::python
