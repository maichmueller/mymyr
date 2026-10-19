#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace mymyr::python
{
/// CUDA stream handles: torch uses 0 for the legacy stream; the array protocols use 1 and 2 for the default streams.
inline cudaStream_t cuda_stream(std::intptr_t handle)
{
    if (handle == 0 || handle == 1)
        return cudaStreamLegacy;
    if (handle == 2)
        return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(handle);
}
}
