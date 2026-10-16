#pragma once
// Launcher of the cuRAND reference kernel (tests/cuda/curand_philox.cu).

#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::test
{
/// out[4 i .. 4 i + 4) = curand_Philox4x32_10(counters[4 i ..], keys[2 i ..]) for i < n, stream-ordered.
cudaError_t launch_curand_philox(const u32* counters, const u32* keys, u64 n, u32* out, cudaStream_t s);
}  // namespace mymyr::test
