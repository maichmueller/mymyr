// Rows of schemas with conditional effects on the device (include/mymyr/cuda/lifted.hpp, launch_write_ce): the
// write kernels of lifted.cu instantiated with the conditional-effect sub-plans at the leaf (RowWriter<WB, OW>). The
// instantiations are spread over units of their own by row width (this one: WB = 4; lifted_ce16.cu, lifted_ce64.cu),
// so that nvcc builds them in parallel with each other and with lifted.cu.
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted
{
namespace ce_units
{
cudaError_t write_ce_16(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                        SuccessorRows out, cudaStream_t s);
cudaError_t write_ce_64(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                        SuccessorRows out, cudaStream_t s);
}  // namespace ce_units

cudaError_t launch_write_ce(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                            SuccessorRows out, cudaStream_t s)
{
    if (out.deferred)
        return cudaErrorInvalidValue;  // launch_put has no conditional effects
    // the successor row and the adds live in u64 [WB] arrays: registers for WB <= 4, local memory beyond
    const u32 need = p.words > out.out_words ? p.words : out.out_words;
    if (need <= 4)
        return dispatch_ow<true, 4, true>(GenArgs{t, p, v, set, offsets, nullptr, labels, out}, s);
    if (need <= 16)
        return ce_units::write_ce_16(t, p, v, set, offsets, labels, out, s);
    if (need <= k_max_words)
        return ce_units::write_ce_64(t, p, v, set, offsets, labels, out, s);
    return cudaErrorInvalidValue;
}
}  // namespace mymyr::cuda::lifted
