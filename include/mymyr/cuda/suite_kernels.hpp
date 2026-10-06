#pragma once
// The device kernels of task suites (cuda/suite_expand.hpp and env.hpp drive them): the rows of a
// batch over several domains split into each domain's instance ids, and the shifts that put a run of a domain's rows at
// its batch positions (a gathered domain writes at its batch positions itself: cuda::DeviceExpander::RowMap).
// Destination-passing launchers over POD parameters.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20; POD, raw pointers and cudaStream_t.

#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::suitek
{
/// The instance ids of each domain's launches over a batch of `rows` rows with global task ids `task_ids`:
/// out[d * stride + i] = local_of[t] if domain_of[t] == d, else 0xFFFFFFFF (the multi-instance kernels skip such rows),
/// for t = task_ids[i]; a task id outside [0, instances) is 0xFFFFFFFF in every domain (the env reports it).
cudaError_t launch_domain_ids(const i32* task_ids, u64 rows, const u32* domain_of, const u32* local_of, u32 instances,
                              u32 domains, u32* out, u64 stride, cudaStream_t s);

/// a[i] += delta for i < n.
cudaError_t launch_add(i32* a, u64 n, i32 delta, cudaStream_t s);

/// The domain split of a batch over a suite (SuiteExpander::count), from the rows' global task ids: local[i] =
/// local_of[task_ids[i]], and the control words ctl [k_classify_words + 2 * domains], initialized here:
/// ctl[k_bad_id] = the first row whose task id is outside [0, instances) (0xFFFFFFFF: none), ctl[k_changes] = the
/// rows whose domain differs from the previous row's, ctl[k_bad_row] = 0xFFFFFFFF (left to lifted::launch_check_multi),
/// ctl[k_classify_words + d] = domain d's rows, ctl[k_classify_words + domains + d] = its first row (0xFFFFFFFF: none).
inline constexpr u32 k_bad_id = 0, k_changes = 1, k_bad_row = 2, k_classify_words = 3;
cudaError_t launch_classify(const i32* task_ids, u64 rows, const u32* domain_of, const u32* local_of, u32 instances,
                            u32 domains, i32* local, u32* ctl, cudaStream_t s);
/// local[p] = local_of[task_ids[index[p]]] for p < n (the local ids of gathered rows).
cudaError_t launch_local_ids(const i32* task_ids, const u32* index, u64 n, const u32* local_of, i32* local, cudaStream_t s);
}  // namespace mymyr::cuda::suitek
