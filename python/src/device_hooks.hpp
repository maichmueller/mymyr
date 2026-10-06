#pragma once
// The device path of rl.expand / rl.expand_into: rl_bindings.cpp is built in every configuration, the device
// expand only in CUDA builds (cuda_bindings.cpp), which installs these hooks when the module is bound. rl.expand calls
// them for CUDA device arrays (__dlpack_device__ reports kDLCUDA); everything else takes the unchanged CPU path.

#include <nanobind/nanobind.h>

namespace mymyr::python
{
namespace nb = nanobind;

/// The arguments of rl.expand.
struct ExpandArgs
{
    nb::handle table, states, task_ids, capacity, words, K;
    bool goal = false, canonical = true, witness = false;
    nb::handle framework;
    bool validate = true;
    nb::handle stream, ctx;
};

/// The arguments of rl.expand_into.
struct ExpandIntoArgs
{
    nb::handle table, states, task_ids, succ, parent, schema, binding, goal, offsets;
    bool canonical = true, witness = false, validate = true;
    nb::handle stream, ctx;
};

struct DeviceHooks
{
    nb::object (*expand)(const ExpandArgs&) = nullptr;
    nb::dict (*expand_into)(const ExpandIntoArgs&) = nullptr;
};

/// The process-wide hooks (null in CPU builds).
[[nodiscard]] DeviceHooks& device_hooks();

/// Whether obj is a CUDA device array (not pinned or managed memory, which host operations read in place).
[[nodiscard]] bool is_cuda_array(nb::handle obj);
}  // namespace mymyr::python
