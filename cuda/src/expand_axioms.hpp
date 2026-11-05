#pragma once
// The successors' axioms of a multi-instance expansion (DeviceExpander, Multi mode): the instance of each written
// successor row, and the goal test of rows of several instances over their derived words.

#include "mymyr/cuda/lifted.hpp"

namespace mymyr::cuda::detail
{
/// out[r] := inst[q] for successor k of each counted row q in [0, rows), whose successors are the num_schemas
/// segments of `offsets` from q * num_schemas on: at r = base + offsets[q * num_schemas] + k, or (map_rows not null) at
/// r = batch_offsets[map_rows[q]] + k. Only rows r below `capacity` are written; the others keep their entries.
cudaError_t launch_successor_ids(const u32* offsets, u32 num_schemas, u64 rows, u64 base, const i32* inst,
                                 const u32* map_rows, const i32* batch_offsets, u64 capacity, i32* out, cudaStream_t s);

/// out[i] := whether row i of `states` (`words` atom words, rows `stride` apart) satisfies the goal of instance inst[i]
/// (views[inst[i]]) with its derived atoms derived [i * derived_words, ...), for i < n. Rows whose instance is outside
/// [0, instances) keep their flags.
cudaError_t launch_goal_rows_derived_multi(const rl::dev::TaskView* views, u32 instances, const i32* inst,
                                           const u64* states, u64 stride, u32 words, u64 n, const u64* derived,
                                           u32 derived_words, u8* out, cudaStream_t s);

/// *dst := *src (a device row count, widened).
cudaError_t launch_widen_count(const u32* src, u64* dst, cudaStream_t s);
}  // namespace mymyr::cuda::detail
