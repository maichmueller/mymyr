// The pick kernels for successor rows wider than k_max_words words (include/mymyr/cuda/lifted.hpp, WB = k_wide), one
// task or several instances (see lifted_multi_pick.cu).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_env.cuh"

namespace mymyr::cuda::lifted::wide_units
{
cudaError_t pick(const GenArgs& a, const Picks& picks, cudaStream_t s) { return dispatch_pick<k_wide, false>(a, picks, s); }

cudaError_t pick_multi(const GenArgs& a, const Picks& picks, cudaStream_t s) { return dispatch_pick<k_wide, true>(a, picks, s); }
}  // namespace mymyr::cuda::lifted::wide_units
