#pragma once
// The device kernels of the lifted action masks (cuda/masks.hpp drives them; rl/masks.hpp is the host reference):
// destination-passing launchers over POD parameters (caller-owned buffers, one stream, no allocation, no
// synchronization). They run the schemas' mask matchers of the device task (plan_schema k_sc_masks: existence, then one
// matcher per parameter with the parameters before it prebound) over the per-row views of lifted::launch_view, with
// the semantics of the host functions bit for bit:
//   - schema masks: one thread per (row, schema) runs the existence matcher and stops at its first binding;
//   - object masks: one warp per row; its lanes split the candidates of the enumerated parameter (the matcher binds it
//     first), each candidate searching one witness for the parameters after it;
//   - label applicability: one thread per row checks the precondition literals of the bound label (no view);
//   - existence over every schema (the label-mode environment's 0/1 count): one thread per (row, schema).
// Rows of instances whose object bitsets are not the launch's width class (`ow`) are skipped: a batch over a table or
// suite takes one launch per width class present. Rows with a task id outside [0, instances) are skipped as well
// (launch_check reports them).
//
// Device-code subset: included by .cu files compiled by nvcc as C++23; POD, raw pointers and cudaStream_t.

#include "mymyr/rl/task_arrays_view.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::maskk
{
/// Status bits of the launches (OR-ed into *status; the host functions raise for the same conditions).
inline constexpr u32 k_bad_task_id = 1;      // a task id outside [0, instances)
inline constexpr u32 k_bad_row = 2;          // a row sets atom slots its instance does not have
inline constexpr u32 k_narrow_rows = 4;      // the rows have fewer words than some row's instance
inline constexpr u32 k_bad_depth = 8;        // the depth is not below a selected schema's arity (object masks)
inline constexpr u32 k_narrow_binding = 16;  // the binding columns are fewer than a selected schema's arity

/// The rows of a launch: state words [rows, stride] (atom words only), row i of instance task_ids[i] (null: 0), whose
/// device view is views[k] (device memory).
struct Rows
{
    const u64* data = nullptr;
    u64 stride = 0;
    u32 words = 0;
    u64 rows = 0;
    const i32* task_ids = nullptr;
    const rl::dev::TaskView* views = nullptr;
    u32 instances = 1;
};

/// Matcher views of the rows (lifted::launch_view's), `words` u64 words per row.
struct ViewRows
{
    const u64* data = nullptr;
    u64 words = 0;
};

/// A schema per row: schema[i], or `value` for every row when schema is null.
struct SchemaArg
{
    const i32* schema = nullptr;
    i32 value = -1;
};

/// The checks of the rows (rl::schema_masks' validation): task ids (k_bad_task_id), rows narrower than their
/// instance's atom words (k_narrow_rows; words [instances]) and, in the word of the instance's last atom slot, bits past
/// it (k_bad_row; slots [instances], the instances' fluent slots).
cudaError_t launch_check(const Rows& r, const u32* words, const u32* slots, u32* status, cudaStream_t s);

/// out[i * num_schemas + k] = 1 iff some full binding of schema k is applicable in row i (rows of width class `ow`;
/// out zeroed by the caller; schemas past the row's domain's stay 0).
cudaError_t launch_schema_masks(const Rows& r, ViewRows v, u32 ow, u32 num_schemas, u8* out, cudaStream_t s);

/// out[i * num_objects + o] = 1 iff the prefix of row i (prefix[i * stride + 0 .. depth)) extended by object o at
/// parameter `depth` extends to an applicable full binding of the row's schema (rows of width class `ow`; out zeroed by
/// the caller). Invalid schemas, prefix objects outside the parameters' static domains and depths at or past the
/// schema's arity (k_bad_depth) leave the row zero.
struct ObjectMasks
{
    SchemaArg schema;
    const i32* prefix = nullptr;
    u32 stride = 0;
    u32 depth = 0;
    u8* out = nullptr;
    u32 num_objects = 0;
};
cudaError_t launch_object_masks(const Rows& r, ViewRows v, u32 ow, const ObjectMasks& m, u32* status, cudaStream_t s);

/// out[i] = 1 iff the label (schema of row i, binding[i * width ..]) is applicable in row i: a valid schema and objects
/// and every precondition literal (Successors::is_applicable). Binding entries past the arity are ignored; fewer
/// columns than a valid schema's arity give 0 and k_narrow_binding.
cudaError_t launch_applicable(const Rows& r, SchemaArg schema, const i32* binding, u32 width, u8* out, u32* status,
                              cudaStream_t s);

/// out[i * stride] = 1 iff some action is applicable in row i (rows of width class `ow` with skip[i] zero, or every
/// row of the class when skip is null): the existence matchers of the row's schemas (num_schemas: the most of any
/// domain). launch_any_clear first writes 0 to the same rows (of every class).
cudaError_t launch_any(const Rows& r, ViewRows v, u32 ow, u32 num_schemas, const u8* skip, u32* out, u64 stride,
                       cudaStream_t s);
cudaError_t launch_any_clear(u64 rows, const u8* skip, u32* out, u64 stride, cudaStream_t s);
}  // namespace mymyr::cuda::maskk
