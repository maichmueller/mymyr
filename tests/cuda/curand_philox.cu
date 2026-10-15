// cuRAND's Philox-4x32-10 as a device kernel, the reference of test_curand_philox.cpp. Compiled by nvcc as C++23 and
// built only when the CUDA toolkit provides cuRAND's headers.

#include "curand_philox.hpp"

#include <curand_philox4x32_x.h>

namespace
{
__global__ void k_curand_philox(const mymyr::u32* counters, const mymyr::u32* keys, mymyr::u64 n, mymyr::u32* out)
{
    for (mymyr::u64 i = mymyr::u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n;
         i += mymyr::u64{gridDim.x} * blockDim.x)
    {
        const mymyr::u32* c = counters + i * 4;
        const mymyr::u32* k = keys + i * 2;
        const uint4 r = curand_Philox4x32_10(make_uint4(c[0], c[1], c[2], c[3]), make_uint2(k[0], k[1]));
        out[i * 4 + 0] = r.x;
        out[i * 4 + 1] = r.y;
        out[i * 4 + 2] = r.z;
        out[i * 4 + 3] = r.w;
    }
}
}  // namespace

namespace mymyr::test
{
cudaError_t launch_curand_philox(const u32* counters, const u32* keys, u64 n, u32* out, cudaStream_t s)
{
    if (n)
        k_curand_philox<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(counters, keys, n, out);
    return cudaGetLastError();
}
}  // namespace mymyr::test
