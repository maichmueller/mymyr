// The write kernels of schemas with conditional effects and of deep matchers for successor rows wider than k_max_words
// words (include/mymyr/cuda/lifted.hpp, WB = k_wide; see lifted_ce.cu and lifted_deep.cu).
// Compiled by nvcc as C++23; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::wide_units
{
cudaError_t write_ce(const GenArgs& a, cudaStream_t s) { return dispatch_ow<true, k_wide, true>(a, s); }

cudaError_t write_deep(const GenArgs& a, cudaStream_t s) { return dispatch_deep<true, k_wide>(a, s); }
}  // namespace mymyr::cuda::lifted::wide_units
