#pragma once
// The scan of the relational encoder's totals, small enough for nvcc to compile as C++20.

#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda
{
/// The temporary storage of the in-place scan of `length` totals, u64 when `wide`.
[[nodiscard]] u64 totals_scan_temp_bytes(u64 length, bool wide);
/// Enqueues the in-place inclusive scan of `length` u32 (u64 when `wide`) totals on `stream`: 2 kernels, no
/// allocation. `temp` holds `temp_bytes` bytes, at least totals_scan_temp_bytes(length, wide).
cudaError_t launch_totals_scan(void* totals, u64 length, bool wide, void* temp, u64 temp_bytes, cudaStream_t stream);
}  // namespace mymyr::cuda
