#pragma once
#include "mymyr/cuda/encoding_types.hpp"
#include <cuda_runtime_api.h>
namespace mymyr::cuda
{
cudaError_t launch_append_rows(const DeviceEncodeInput& input, const DeviceEncodeInput& ring, u64* status, cudaStream_t stream);
cudaError_t launch_clear_rows(u64* status, u64 capacity, bool overflow, cudaStream_t stream);
}
