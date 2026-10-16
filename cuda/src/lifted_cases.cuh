#pragma once
// Definitions of the per-case launches the lifted dispatchers call (lifted_device.cuh: gen_warp_case, gen_block_case, deep_case;
// lifted_env.cuh: count_keys_case, pick_case). Included only by the generated translation units of the kernel
// instantiations (lifted_units.cmake): a unit that includes this file and explicitly instantiates a case compiles that
// case's kernels, and no other translation unit does. Internal: nvcc, C++23.

#include "lifted_env.cuh"

namespace mymyr::cuda::lifted
{
template<u32 OW, bool FC, bool Write, u32 WB, bool CE>
cudaError_t gen_warp_case(const GenArgs& a, unsigned blocks, cudaStream_t s)
{
    k_gen_warp<OW, FC, Write, WB, CE><<<blocks, k_gen_warp_block, 0, s>>>(a);
    return cudaGetLastError();
}

template<u32 OW, bool FC, bool Write, u32 WB, bool CE, bool M>
cudaError_t gen_block_case(const GenArgs& a, unsigned blocks, unsigned threads, cudaStream_t s)
{
    k_gen<OW, FC, Write, WB, CE, M><<<blocks, threads, 0, s>>>(a);
    return cudaGetLastError();
}

template<u32 OW, bool FC, bool Write, u32 WB>
cudaError_t deep_case(const GenArgs& a, cudaStream_t s)
{
    return launch_deep<OW, FC, Write, WB>(a, s);
}

template<u32 OW, bool FC, bool M>
cudaError_t count_keys_case(const GenArgs& a, const CountCache& c, bool first, cudaStream_t s)
{
    return launch_count_keys_ow<OW, FC, M>(a, c, first, s);
}

template<u32 OW, bool FC, u32 WB, bool M>
cudaError_t pick_case(const GenArgs& a, const Picks& pk, cudaStream_t s)
{
    return launch_pick_ow<OW, FC, WB, M>(a, pk, s);
}
}  // namespace mymyr::cuda::lifted
