// The deep kernels (include/mymyr/cuda/lifted.hpp, SchemaSet::deep): the count and write kernels for matchers
// of up to k_deep_depth parameters, a warp per search (lifted_device.cuh, warp_fixed / warp_fc). launch_count and
// launch_write route a deep set here; this unit has the counts and the rows of up to 4 words, lifted_deep64.cu the
// wider rows (parallel builds).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::deep_units
{
cudaError_t write_deep_64(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                          SuccessorRows out, cudaStream_t s);

cudaError_t count_deep(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, u32* counts, cudaStream_t s)
{
    return dispatch_deep<false, 1>(GenArgs{t, p, v, set, nullptr, counts, Labels{}, SuccessorRows{}}, s);
}

cudaError_t write_deep(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                       SuccessorRows out, cudaStream_t s)
{
    const GenArgs a{t, p, v, set, offsets, nullptr, labels, out};
    const u32 need = p.words > out.out_words ? p.words : out.out_words;
    if (need <= 2)
        return dispatch_deep<true, 2>(a, s);
    if (need <= 4)
        return dispatch_deep<true, 4>(a, s);
    return write_deep_64(t, p, v, set, offsets, labels, out, s);
}
}  // namespace mymyr::cuda::lifted::deep_units
