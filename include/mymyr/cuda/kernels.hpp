#pragma once
// Destination-passing launchers of the CUDA backend: every launcher takes its inputs,
// caller-owned outputs and a stream, allocates nothing, never synchronizes, and returns the launch's error code. The
// same launchers back mymyr's drivers, the Python bindings and, later, XLA FFI handlers and torch custom ops.
//
// Device-code subset: this header is included by .cu translation units compiled by nvcc as C++20, so it
// includes only the C runtime API header and task_arrays_view.hpp.
//
// Smoke kernels over an uploaded task (cuda/device_task.hpp): they run the MYMYR_HD readers of
// rl/task_arrays_view.hpp on the device, one thread per item, and must agree with the CPU engine. Lifted successor
// generation and BrFS are implemented separately.

#include "mymyr/rl/task_arrays_view.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::kernels
{
/// A batch of state words on the device: rows [N, words] with a row stride in words.
struct DeviceStates
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 words = 0;
    u64 stride = 0;  // words between rows (>= words)
};

/// Ground actions: label i is (schema[i], binding[i * width .. i * width + arity)) applied in state state_index[i].
struct DeviceLabels
{
    const u32* state_index = nullptr;
    const u32* schema = nullptr;
    const u32* binding = nullptr;
    u64 count = 0;
    u32 width = 0;  // binding columns (at least the largest arity among the labels' schemas)
};

/// Structural validation of every table (rl::dev::validate_item over all items). out[0] = number of failed items,
/// out[1] = the smallest failed rule number (0xFFFFFFFF if none). `out` must be zeroed except out[1] = 0xFFFFFFFF.
cudaError_t launch_validate(const rl::dev::TaskView& v, u32* out, cudaStream_t s);

/// out[i] = 1 if label i is applicable in its state (0 for malformed labels: schema or object out of range).
/// `derived` [N, derived_words] holds the states' derived bitsets (nullptr when the task has no axioms).
cudaError_t launch_applicable(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words,
                              DeviceLabels labels, u8* out, cudaStream_t s);

/// Successors of applicable labels (schemas without conditional effects): succ [count, out_words] and status[i] =
/// rl::dev::k_apply_* (k_apply_ok with a written row), or 0xFFFFFFFF where the label is not applicable (row zeroed).
cudaError_t launch_apply(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words,
                         DeviceLabels labels, u64* succ, u32 out_words, u32* status, cudaStream_t s);

/// out[i] = 1 if state i is a goal state.
cudaError_t launch_goal(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words, u8* out,
                        cudaStream_t s);

// ------------------------------------------------------------------------------------------------ test and bench helpers
/// dst[i] = value + i for i < n.
cudaError_t launch_iota(u64* dst, u64 n, u64 value, cudaStream_t s);
/// Spins for about `cycles` clock cycles in one thread, then writes dst[i] = value + i (stream-semantics tests: a
/// consumer that does not wait for this kernel reads the old contents).
cudaError_t launch_delayed_iota(u64* dst, u64 n, u64 value, u64 cycles, cudaStream_t s);
/// sync_cost.cu's k_busy: a DRAM-heavy kernel over n words, `iters` multiply-adds per word.
cudaError_t launch_busy(u64* x, u64 n, int iters, cudaStream_t s);
/// sync_cost.cu's k_touch: atomicAdd(c, 1) in one thread.
cudaError_t launch_touch(u32* c, cudaStream_t s);
}  // namespace mymyr::cuda::kernels
