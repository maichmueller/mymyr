// Lifted successor generation on the device (include/mymyr/cuda/lifted.hpp), over the "plan" section of the device
// task. The matcher executors (lifted_device.cuh) are line-by-line ports of successor/detail/engine.hpp (Engine::run,
// search, search_fc) turned into iterative loops, so a
// thread emits the bindings of one (state, schema) in exactly the CPU engine's order, witnesses included. This unit holds
// the views, the counts, the rows of schemas without conditional effects (their rows: lifted_ce.cu), the goal tests,
// the host-row placement, the scans, the padding and the picks.
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "lifted_env.cuh"

#include <cub/device/device_scan.cuh>

namespace mymyr::cuda::lifted
{
namespace
{
// ------------------------------------------------------------------------------------------------ goals, checks

__global__ void k_goal_count(TaskView t, Parents p, u32* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x, n = live_rows(p); i < n; i += u64{gridDim.x} * blockDim.x)
    {
        const u64* d = p.derived ? p.derived + i * p.derived_words : nullptr;
        if (goal_holds(t, p.data + i * p.stride, p.words, d, p.derived_words))
        {
            atomicAdd(out, 1u);
            atomicMin(out + 1, static_cast<u32>(i));
        }
    }
}

__global__ void k_goal_rows(TaskView t, const u64* states, u32 words, u64 n, const u32* n_dev, u8* out)
{
    const u64 m = n_dev && *n_dev < n ? *n_dev : n;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < m; i += u64{gridDim.x} * blockDim.x)
        out[i] = goal_holds(t, states + i * words, words, nullptr, 0) ? 1 : 0;
}

__global__ void k_check_rows(const u64* data, u64 stride, u32 words, u64 rows, u32 limit, u32* bad)
{
    const u32 lw = limit >> 6;
    const u64 first = limit & 63 ? ~((1ull << (limit & 63)) - 1) : ~0ull;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        const u64* r = data + i * stride;
        bool b = false;
        for (u32 w = lw; w < words && !b; ++w)
            b = (r[w] & (w == lw ? first : ~0ull)) != 0;
        if (b)
            atomicMin(bad, static_cast<u32>(i < 0xFFFFFFFEull ? i : 0xFFFFFFFEull));
    }
}

// ------------------------------------------------------------------------------------------------ host rows, offsets

__global__ void k_place_host(HostRows h, const u32* offsets, u32 S, Labels lab, u64* out_words, u32 out_width)
{
    if (lab.live && *lab.live == 0)
        return;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < h.n; i += u64{gridDim.x} * blockDim.x)
    {
        const u32 seg = h.segment[i];
        const u64 pos = u64{offsets[seg]} + h.rank[i];
        if (pos >= lab.capacity)
            continue;
        const u32 L = lab.label_width;
        if (lab.binding)
            for (u32 k = 0; k < L; ++k)
                lab.binding[pos * L + k] = h.binding[i * L + k];
        if (lab.schema)
            lab.schema[pos] = seg % S;
        if (lab.parent)
            lab.parent[pos] = lab.parent_base + seg / S;
        if (h.words && out_words)
            for (u32 k = 0; k < out_width; ++k)
                out_words[pos * out_width + k] = h.words[i * out_width + k];
    }
}

__global__ void k_scatter_counts(const u32* segment, const u32* value, u64 n, u32* counts)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        counts[segment[i]] = value[i];
}

__global__ void k_state_offsets(const u32* seg, u32 S, u64 rows, u64 base, i32* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i <= rows; i += u64{gridDim.x} * blockDim.x)
        out[i] = static_cast<i32>(base + seg[i * S]);
}

__global__ void k_pad(Flat f, u64 rows, Padded o)
{
    const u64 n = rows * o.K;
    for (u64 at = u64{blockIdx.x} * blockDim.x + threadIdx.x; at < n; at += u64{gridDim.x} * blockDim.x)
    {
        const u64 i = at / o.K;
        const u32 k = static_cast<u32>(at % o.K);
        const u64 b = static_cast<u64>(f.offsets[i]), e = static_cast<u64>(f.offsets[i + 1]);
        const u64 c = e - b;
        if (k == 0)
        {
            if (o.count)
                o.count[i] = static_cast<i32>(c);
            if (c > o.K && o.overflow)
                *o.overflow = 1;
        }
        const u64 j = b + k;
        const bool valid = k < c && j < f.capacity;
        if (k < c && j >= f.capacity && o.overflow)
            *o.overflow = 1;
        if (o.index)
            o.index[at] = valid ? static_cast<i32>(j) : -1;
        if (o.mask)
            o.mask[at] = valid ? 1 : 0;
        if (o.succ)
        {
            u64* dst = o.succ + at * o.words;
            u32 w = 0;
            if (valid && f.succ)
                for (; w < f.words; ++w)
                    dst[w] = f.succ[j * f.words + w];
            for (; w < o.words; ++w)
                dst[w] = 0;
        }
        if (o.schema)
            o.schema[at] = valid && f.schema ? f.schema[j] : -1;
        if (o.binding)
        {
            i32* dst = o.binding + at * o.label_width;
            u32 w = 0;
            if (valid && f.binding)
                for (; w < f.label_width; ++w)
                    dst[w] = f.binding[j * f.label_width + w];
            for (; w < o.label_width; ++w)
                dst[w] = -1;
        }
        if (o.goal)
            o.goal[at] = valid && f.goal ? f.goal[j] : 0;
    }
}
}  // namespace

// ------------------------------------------------------------------------------------------------ launchers

cudaError_t launch_view(const rl::dev::TaskView& t, Parents p, Views v, cudaStream_t s)
{
    return launch_view_any<false>(t, Multi{}, p, v, s);
}

cudaError_t launch_goal_count(const rl::dev::TaskView& t, Parents p, u32* out, cudaStream_t s)
{
    if (p.rows)
        k_goal_count<<<grid_for(p.rows), k_block, 0, s>>>(t, p, out);
    return cudaGetLastError();
}

cudaError_t launch_goal_rows(const rl::dev::TaskView& t, const u64* states, u32 words, u64 n, const u32* n_dev, u8* out,
                             cudaStream_t s)
{
    if (n)
        k_goal_rows<<<grid_for(n), k_block, 0, s>>>(t, states, words, n, n_dev, out);
    return cudaGetLastError();
}

cudaError_t launch_check_rows(const u64* data, u64 stride, u32 words, u64 rows, u32 limit, u32* bad_row, cudaStream_t s)
{
    if (rows && u64{words} * 64 > limit)
        k_check_rows<<<grid_for(rows), k_block, 0, s>>>(data, stride, words, rows, limit, bad_row);
    return cudaGetLastError();
}

namespace deep_units
{
cudaError_t count_deep(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, u32* counts, cudaStream_t s);
cudaError_t write_deep(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                       SuccessorRows out, cudaStream_t s);
}  // namespace deep_units

cudaError_t launch_count(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, u32* counts, cudaStream_t s)
{
    if (set.deep)
        return deep_units::count_deep(t, p, v, set, counts, s);
    return dispatch_ow<false, 1>(GenArgs{t, p, v, set, nullptr, counts, Labels{}, SuccessorRows{}}, s);
}

cudaError_t launch_write(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, const u32* offsets, Labels labels,
                         SuccessorRows out, cudaStream_t s)
{
    if (out.deferred && !labels.binding)
        return cudaErrorInvalidValue;
    if (set.deep)
        return deep_units::write_deep(t, p, v, set, offsets, labels, out, s);
    const GenArgs a{t, p, v, set, offsets, nullptr, labels, out};
    // the successor row lives in u64 y[WB]: registers for WB <= 4, a WB-word local array up to 16, 64 words beyond
    const u32 need = p.words > out.out_words ? p.words : out.out_words;
    if (need <= 2)
        return dispatch_ow<true, 2>(a, s);
    if (need <= 4)
        return dispatch_ow<true, 4>(a, s);
    if (need <= 16)
        return dispatch_ow<true, 16>(a, s);
    if (need <= k_max_words)
        return dispatch_ow<true, k_max_words>(a, s);
    return cudaErrorInvalidValue;
}

cudaError_t launch_put(const rl::dev::TaskView& t, Parents p, u32 num_schemas, const u8* deferred, const u32* offsets,
                       Labels labels, SuccessorRows out, u64 rows_hint, cudaStream_t s)
{
    if (!labels.binding || !num_schemas)
        return cudaErrorInvalidValue;
    SchemaSet set;
    set.num_schemas = num_schemas;
    out.deferred = 0;
    const GenArgs a{t, p, Views{}, set, offsets, nullptr, labels, out};
    return launch_put_rows<false>(a, PutRows{}, deferred, rows_hint, s);
}

cudaError_t launch_place_host(HostRows h, const u32* offsets, u32 num_schemas, Labels labels, u64* out_words, u32 out_width,
                              cudaStream_t s)
{
    if (h.n)
        k_place_host<<<grid_for(h.n), k_block, 0, s>>>(h, offsets, num_schemas, labels, out_words, out_width);
    return cudaGetLastError();
}

cudaError_t launch_scatter_counts(const u32* segment, const u32* value, u64 n, u32* counts, cudaStream_t s)
{
    if (n)
        k_scatter_counts<<<grid_for(n), k_block, 0, s>>>(segment, value, n, counts);
    return cudaGetLastError();
}

u64 scan_temp_bytes(u64 n)
{
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                  static_cast<int>(n > 0x7FFFFFFFull ? 0x7FFFFFFF : n));
    return bytes;
}

cudaError_t launch_scan(const u32* in, u32* out, u64 n, void* temp, u64 temp_bytes, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    if (n > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    size_t bytes = temp_bytes;
    return cub::DeviceScan::ExclusiveSum(temp, bytes, in, out, static_cast<int>(n), s);
}

cudaError_t launch_state_offsets(const u32* seg_offsets, u32 num_schemas, u64 rows, u64 base, i32* offsets_out, cudaStream_t s)
{
    k_state_offsets<<<grid_for(rows + 1), k_block, 0, s>>>(seg_offsets, num_schemas, rows, base, offsets_out);
    return cudaGetLastError();
}

cudaError_t launch_pad(Flat flat, u64 rows, Padded out, cudaStream_t s)
{
    if (rows && out.K)
        k_pad<<<grid_for(rows * out.K), k_block, 0, s>>>(flat, rows, out);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::lifted

// ------------------------------------------------------------------------------------------------ picks
// The environment step's kernels (include/mymyr/cuda/env.hpp): the count cache and the picks (lifted_env.cuh), one
// instance per launch here; the multi-instance forms are in lifted_multi*.cu.

namespace mymyr::cuda::lifted
{
cudaError_t launch_count_keys(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, CountCache c, bool first,
                              cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    return dispatch_count_keys<false>(GenArgs{t, p, v, set, nullptr, nullptr, Labels{}, SuccessorRows{}}, c, first, s);
}

cudaError_t launch_pick(const rl::dev::TaskView& t, Parents p, Views v, SchemaSet set, Picks picks, Labels labels,
                        SuccessorRows out, cudaStream_t s)
{
    if (p.rows == 0 || set.count == 0)
        return cudaSuccess;
    if (!picks_valid(p, set, picks, labels, false))
        return cudaErrorInvalidValue;
    const GenArgs a{t, p, v, set, nullptr, nullptr, labels, out};
    const u32 need = p.words > out.out_words ? p.words : out.out_words;
    if (need <= 2)
        return dispatch_pick<2, false>(a, picks, s);
    if (need <= 4)
        return dispatch_pick<4, false>(a, picks, s);
    if (need <= 16)
        return dispatch_pick<16, false>(a, picks, s);
    if (need <= k_max_words)
        return dispatch_pick<k_max_words, false>(a, picks, s);
    return cudaErrorInvalidValue;
}
}  // namespace mymyr::cuda::lifted
