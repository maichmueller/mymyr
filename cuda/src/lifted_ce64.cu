// Rows of schemas with conditional effects, successor rows of 17..64 words (see lifted_ce.cu).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::ce_units
{
cudaError_t write_ce_64(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                        SuccessorRows out, cudaStream_t s)
{
    return dispatch_ow<true, k_max_words, true>(GenArgs{t, p, v, set, offsets, nullptr, labels, out}, s);
}
}  // namespace mymyr::cuda::lifted::ce_units
