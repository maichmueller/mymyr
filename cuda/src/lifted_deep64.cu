// The deep write kernels for successor rows of 5..k_max_words words (see lifted_deep.cu).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::deep_units
{
cudaError_t write_deep_64(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                          SuccessorRows out, cudaStream_t s)
{
    const GenArgs a{t, p, v, set, offsets, nullptr, labels, out};
    const u32 need = p.words > out.out_words ? p.words : out.out_words;
    if (need <= 16)
        return dispatch_deep<true, 16>(a, s);
    if (need <= k_max_words)
        return dispatch_deep<true, k_max_words>(a, s);
    return cudaErrorInvalidValue;
}
}  // namespace mymyr::cuda::lifted::deep_units
