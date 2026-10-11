// The device-side sizing of the relational encoder (include/mymyr/cuda/encoding.hpp). Compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"

#include <cub/device/device_scan.cuh>

namespace mymyr::cuda::detail
{
/// The temporary storage cub::DeviceScan::InclusiveSum needs for `n` u64 values.
u64 scan_temp_bytes(u64 n)
{
    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<const u64*>(nullptr), static_cast<u64*>(nullptr),
                                  static_cast<i64>(n));
    return bytes;
}
}  // namespace mymyr::cuda::detail
