#pragma once

#include "mymyr/cuda/lifted.hpp"

namespace mymyr::cuda::numeric
{
cudaError_t launch_pad(lifted::Flat flat, u64 rows, lifted::Padded out, u32 numeric_words, cudaStream_t stream);
}
