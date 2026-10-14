// The device side of context_probe.hpp.

#include "context_probe.hpp"

namespace mymyr::test
{
namespace
{
__global__ void probe_kernel(encoding::detail::ContextView context, u32 instances, u64* out)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < instances)
        probe_instance(context, i, out + u64{i} * k_probe_width);
}
}  // namespace

void launch_context_probe(const encoding::detail::ContextView& context, u32 instances, u64* out, cudaStream_t stream)
{
    if (instances)
        probe_kernel<<<(instances + 127) / 128, 128, 0, stream>>>(context, instances, out);
}
}  // namespace mymyr::test
