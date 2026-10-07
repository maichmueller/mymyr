#pragma once
// Numeric lifted successors. Parents expose atom words followed by one canonical double per fluent slot; successor
// rows contain out_words - numeric.slots atom words followed by the same numeric block.

#include "mymyr/cuda/lifted.hpp"

namespace mymyr::cuda::numeric
{
cudaError_t launch_count(rl::dev::TaskView task, lifted::Parents parents, lifted::Views views,
                         lifted::SchemaSet schemas, u32* counts, u32* error, cudaStream_t stream);
cudaError_t launch_write(rl::dev::TaskView task, lifted::Parents parents, lifted::Views views,
                         lifted::SchemaSet schemas, const u32* offsets, lifted::Labels labels,
                         lifted::SuccessorRows out, u32* error, cudaStream_t stream);
cudaError_t launch_goals(rl::dev::TaskView task, lifted::Parents parents, u32* count,
                         const u32* order, u64 n, u8* flags, cudaStream_t stream);
/// Converts rows between the CPU encoding and the device's doubles. Atom widths may differ; missing atom words are zero.
cudaError_t launch_convert(rl::dev::TaskView task, const u64* src, u64 src_stride, u32 src_words,
                           u64* dst, u64 dst_stride, u32 dst_words, u64 rows, bool to_device, cudaStream_t stream);
}  // namespace mymyr::cuda::numeric
