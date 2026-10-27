#pragma once
// The device path of rl.expand / rl.expand_into and of the action masks (rl.applicable_schemas, ...): rl_bindings.cpp
// is built in every configuration, the device functions only in CUDA builds (cuda_bindings.cpp), which installs these
// hooks when the module is bound. The functions call them for CUDA device arrays (__dlpack_device__ reports kDLCUDA);
// everything else takes the unchanged CPU path.

#include <nanobind/nanobind.h>

namespace mymyr::python
{
namespace nb = nanobind;

/// The arguments of rl.expand.
struct ExpandArgs
{
    nb::handle table, states, task_ids, capacity, words, k;
    bool goal = false, derived = false, canonical = true, witness = false;
    nb::handle framework;
    bool validate = true;
    nb::handle stream, ctx;
};

/// The arguments of rl.expand_into.
struct ExpandIntoArgs
{
    nb::handle table, states, task_ids, succ, parent, schema, binding, goal, derived, offsets;
    bool canonical = true, witness = false, validate = true;
    nb::handle stream, ctx, status, row_count;
};

/// The function of an action-mask call.
enum class MaskKind
{
    Schemas,     // rl.applicable_schemas
    Objects,     // rl.applicable_objects
    Applicable,  // rl.is_applicable
};

/// The arguments of rl.applicable_schemas / rl.applicable_objects / rl.is_applicable (those a function does not take
/// are None): `labels` is the prefix (objects) or the binding (applicable).
struct MaskArgs
{
    MaskKind kind = MaskKind::Schemas;
    nb::handle table, states, task_ids, schema, labels, depth, num_objects;
};

struct DeviceHooks
{
    nb::object (*expand)(const ExpandArgs&) = nullptr;
    nb::dict (*expand_into)(const ExpandIntoArgs&) = nullptr;
    nb::object (*masks)(const MaskArgs&) = nullptr;
};

/// The process-wide hooks (null in CPU builds).
[[nodiscard]] DeviceHooks& device_hooks();

/// Whether obj is a CUDA device array (not pinned or managed memory, which host operations read in place).
[[nodiscard]] bool is_cuda_array(nb::handle obj);
}  // namespace mymyr::python
