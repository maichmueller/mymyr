#include "numeric_pad.hpp"

namespace mymyr::cuda::numeric
{
namespace
{
__global__ void pad(lifted::Flat f, u64 rows, lifted::Padded o, u32 nn)
{
    const u64 at = u64{blockIdx.x} * blockDim.x + threadIdx.x;
    if (at >= rows * o.K) return;
    const u64 i = at / o.K;
    const u32 k = static_cast<u32>(at % o.K);
    const u64 b = static_cast<u64>(f.offsets[i]), e = static_cast<u64>(f.offsets[i + 1]), c = e - b;
    if (!k)
    {
        if (o.count) o.count[i] = static_cast<i32>(c);
        if (c > o.K && o.overflow) atomicOr(o.overflow, 1u);
    }
    const u64 j = b + k;
    const bool valid = k < c && j < f.capacity;
    if (k < c && j >= f.capacity && o.overflow) atomicOr(o.overflow, 1u);
    if (o.index) o.index[at] = valid ? static_cast<i32>(j) : -1;
    if (o.mask) o.mask[at] = valid;
    if (o.succ)
    {
        u64* dst = o.succ + at * o.words;
        const u64* src = valid && f.succ ? f.succ + j * f.words : nullptr;
        for (u32 w = 0; w < o.words - nn; ++w)
            dst[w] = src && w < f.words - nn ? src[w] : 0;
        for (u32 w = 0; w < nn; ++w)
            dst[o.words - nn + w] = src ? src[f.words - nn + w] : 0;
    }
    if (o.schema) o.schema[at] = valid && f.schema ? f.schema[j] : -1;
    if (o.binding)
        for (u32 w = 0; w < o.label_width; ++w)
            o.binding[at * o.label_width + w] = valid && f.binding && w < f.label_width ? f.binding[j * f.label_width + w] : -1;
    if (o.goal) o.goal[at] = valid && f.goal ? f.goal[j] : 0;
}
}  // namespace

cudaError_t launch_pad(lifted::Flat flat, u64 rows, lifted::Padded out, u32 numeric_words, cudaStream_t stream)
{
    const u64 n = rows * out.K;
    if (numeric_words > flat.words || numeric_words > out.words) return cudaErrorInvalidValue;
    if (n) pad<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(flat, rows, out, numeric_words);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::numeric
