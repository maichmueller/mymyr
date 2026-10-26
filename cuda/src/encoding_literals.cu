#include "encoding_kernels.hpp"
#include "mymyr/encoding/detail/literals.hpp"

namespace mymyr::cuda::detail
{
namespace
{
constexpr unsigned k_threads = 256;

__device__ void invalid(u64* status)
{
    atomicOr(reinterpret_cast<unsigned long long*>(status), 4ull);
}

__global__ void slots_kernel(encoding::detail::ContextView context, DeviceLiteralInput input,
                             u8* kinds, i32* slots, u64* status)
{
    using encoding::detail::LiteralKind;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < input.count;
         i += u64{gridDim.x} * blockDim.x)
    {
        const auto slot = encoding::detail::resolve_literal(
            context, input.task_ids ? input.task_ids[i] : 0, input.predicates[i],
            input.width ? input.objects + i * input.width : nullptr, input.width);
        kinds[i] = static_cast<u8>(slot.kind);
        slots[i] = slot.kind == LiteralKind::Fluent || slot.kind == LiteralKind::Derived
                       ? static_cast<i32>(slot.slot) : -1;
        if (slot.kind == LiteralKind::Invalid)
            invalid(status);
    }
}

__global__ void masks_kernel(encoding::detail::ContextView context, DeviceLiteralInput input,
                             DeviceLiteralPlacement placement, DeviceLiteralMasks output)
{
    using encoding::detail::LiteralKind;
    const u64 stride = u64{gridDim.x} * blockDim.x;
    const u64 first = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    for (u64 row = first; row < placement.num_rows; row += stride)
        if (input.task_ids && (input.task_ids[row] < 0 || static_cast<u32>(input.task_ids[row]) >= context.num_instances))
            invalid(output.status);
    for (u64 i = first; i < input.count; i += stride)
    {
        const i32 row = placement.rows[i];
        const i32 layer = placement.layers ? placement.layers[i] : 0;
        if (row < 0 || static_cast<u32>(row) >= placement.num_rows ||
            layer < 0 || static_cast<u32>(layer) >= placement.num_layers)
        {
            invalid(output.status);
            continue;
        }
        const auto slot = encoding::detail::resolve_literal(
            context, input.task_ids ? input.task_ids[row] : 0, input.predicates[i],
            input.width ? input.objects + i * input.width : nullptr, input.width);
        if (slot.kind == LiteralKind::Invalid)
        {
            invalid(output.status);
            continue;
        }
        const bool positive = placement.positive[i] != 0;
        const u64 cell = u64{static_cast<u32>(row)} * placement.num_layers + static_cast<u32>(layer);
        if (slot.kind == LiteralKind::Fluent || slot.kind == LiteralKind::Derived)
        {
            const bool derived = slot.kind == LiteralKind::Derived;
            u64* mask = derived ? (positive ? output.derived_pos : output.derived_neg)
                                : (positive ? output.pos : output.neg);
            const u32 words = derived ? output.derived_words : output.words;
            atomicOr(reinterpret_cast<unsigned long long*>(mask + cell * words + (slot.slot >> 6)),
                     1ull << (slot.slot & 63u));
        }
        else if (!encoding::detail::fixed_literal_holds(slot.kind, positive))
            atomicOr(output.unsatisfiable + cell, 1u);
    }
}

unsigned blocks(u64 count)
{
    const u64 n = count / k_threads + (count % k_threads != 0);
    return static_cast<unsigned>(n < 65535 ? n : 65535);
}
}  // namespace

cudaError_t launch_literal_slots(const encoding::detail::ContextView& context, const DeviceLiteralInput& input,
                                u8* kinds, i32* slots, u64* status, cudaStream_t stream)
{
    if (input.count)
        slots_kernel<<<blocks(input.count), k_threads, 0, stream>>>(context, input, kinds, slots, status);
    return cudaGetLastError();
}

cudaError_t launch_literal_masks(const encoding::detail::ContextView& context, const DeviceLiteralInput& input,
                                const DeviceLiteralPlacement& placement, const DeviceLiteralMasks& output,
                                cudaStream_t stream)
{
    const u64 count = input.count > placement.num_rows ? input.count : placement.num_rows;
    if (count)
        masks_kernel<<<blocks(count), k_threads, 0, stream>>>(context, input, placement, output);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::detail
