#pragma once
// Device kernels of the RL helpers (rl/novelty.hpp, rl/prefix.hpp, rl/her.hpp): destination-passing launchers
// (caller-owned device buffers, one stream, no allocation, no synchronization) running the element functions of
// the host loops, so both give the same bytes. Masks are zeroed on the stream first, then scattered (writes of 1 in any
// order); novelty rows and relabels are independent threads.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20; POD, raw pointers and cudaStream_t.

#include "mymyr/rl/her.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda
{
/// rl::novelty_update on device arrays: seen [rows, words] (in place), states [rows, state_stride], reward [rows].
cudaError_t launch_novelty_update(u64* seen, const u64* states, u64 rows, u32 words, u32 state_stride, i32* reward,
                                  cudaStream_t s);
/// rl::prefix_masks on device arrays (x, q point to device memory; the structs themselves are passed by value).
cudaError_t launch_prefix_masks(const rl::LabelRows& x, const rl::PrefixQuery& q, u8* mask, cudaStream_t s);
/// rl::schema_masks on device arrays.
cudaError_t launch_schema_masks(const rl::LabelRows& x, u64 rows, u32 num_schemas, u8* mask, cudaStream_t s);
/// rl::her_relabel on device arrays (one thread per relabel).
cudaError_t launch_her_relabel(const rl::HerBatch& b, const rl::HerConfig& c, cudaStream_t s);
}  // namespace mymyr::cuda
