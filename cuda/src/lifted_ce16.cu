// Rows of schemas with conditional effects, successor rows of 5..16 words (see lifted_ce.cu).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::ce_units
{
cudaError_t write_ce_16(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                        SuccessorRows out, cudaStream_t s)
{
    return dispatch_ow<true, 16, true>(GenArgs{t, p, v, set, offsets, nullptr, labels, out}, s);
}
}  // namespace mymyr::cuda::lifted::ce_units
