// The write kernels for successor rows wider than k_max_words words (include/mymyr/cuda/lifted.hpp, WB = k_wide): the
// successor is built k_max_words words at a time (RowWriter::successor_chunks), one task or several instances.
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_device.cuh"

namespace mymyr::cuda::lifted::wide_units
{
cudaError_t write(const GenArgs& a, cudaStream_t s) { return dispatch_ow<true, k_wide>(a, s); }

cudaError_t write_multi(const GenArgs& a, cudaStream_t s) { return dispatch_ow<true, k_wide, false, true>(a, s); }
}  // namespace mymyr::cuda::lifted::wide_units
