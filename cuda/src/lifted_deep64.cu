// The deep write kernels for successor rows of 5..k_max_words words (see lifted_deep.cu; wider rows: lifted_wide_ce.cu).
// Compiled by nvcc as C++23; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted
{
namespace wide_units
{
cudaError_t write_deep(const GenArgs& a, cudaStream_t s);
}  // namespace wide_units

namespace deep_units
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
    return wide_units::write_deep(a, s);
}
}  // namespace deep_units
}  // namespace mymyr::cuda::lifted
