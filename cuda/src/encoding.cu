// The device-side sizing of the relational encoder (include/mymyr/cuda/encoding.hpp). Compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <limits>

#include <cub/device/device_scan.cuh>

namespace mymyr::cuda
{
void check(cudaError_t error, const char* what);

namespace detail
{
/// The temporary storage of one in-place scalar scan, shared by the three sequential scans.
u64 scan_temp_bytes(u64 n, bool wide)
{
    if (n > static_cast<u64>(std::numeric_limits<i64>::max()))
        throw std::overflow_error("mymyr: encoding: scan length overflow");
    size_t bytes = 0;
    const cudaError_t error = wide
        ? cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<const u64*>(nullptr), static_cast<u64*>(nullptr),
                                       static_cast<i64>(n))
        : cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                       static_cast<i64>(n));
    cuda::check(error, "cub::DeviceScan::InclusiveSum (encoding scratch)");
    return bytes;
}
}  // namespace detail
}  // namespace mymyr::cuda
