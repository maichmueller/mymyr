// The device-side scan of the relational encoder's totals (include/mymyr/cuda/encoding.hpp). Compiled by nvcc as C++23.

#include "mymyr/cuda/encoding_scan.hpp"
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>

#include <cub/device/device_scan.cuh>

namespace mymyr::cuda
{
void check(cudaError_t error, const char* what);

namespace
{
i64 scan_size(u64 n)
{
    if (n > static_cast<u64>(std::numeric_limits<i64>::max()))
        throw std::overflow_error("mymyr: encoding: scan length overflow");
    return static_cast<i64>(n);
}
}  // namespace

u64 totals_scan_temp_bytes(u64 length, bool wide)
{
    size_t bytes = 0;
    const cudaError_t error = wide ? cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u64*>(nullptr), scan_size(length))
                                   : cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<u32*>(nullptr), scan_size(length));
    check(error, "cub::DeviceScan::InclusiveSum (encoding scratch)");
    return bytes;
}

cudaError_t launch_totals_scan(void* totals, u64 length, bool wide, void* temp, u64 temp_bytes, cudaStream_t stream)
{
    size_t bytes = temp_bytes;
    return wide ? cub::DeviceScan::InclusiveSum(temp, bytes, static_cast<u64*>(totals), scan_size(length), stream)
                : cub::DeviceScan::InclusiveSum(temp, bytes, static_cast<u32*>(totals), scan_size(length), stream);
}
}  // namespace mymyr::cuda
