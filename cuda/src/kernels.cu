// Smoke kernels (include/mymyr/cuda/kernels.hpp) over an uploaded task, and the helpers of the tests and the cost
// measurements. Compiled by nvcc as C++20; includes only the device-code subset.

#include "mymyr/cuda/kernels.hpp"

namespace mymyr::cuda::kernels
{
namespace
{
constexpr unsigned k_block = 256;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 16u ? (g ? g : 1) : 65535u * 16u);
}

__global__ void k_validate(rl::dev::TaskView v, u64 items, u32* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < items; i += u64{gridDim.x} * blockDim.x)
    {
        const u32 rule = rl::dev::validate_item(v, i);
        if (rule)
        {
            atomicAdd(out, 1u);
            atomicMin(out + 1, rule);
        }
    }
}

/// Copies label i's binding into `bind` (max_bind entries) and checks it; false for a malformed label.
__device__ bool load_label(const rl::dev::TaskView& v, const DeviceLabels& l, u64 i, u64 rows, u32* bind, u32& schema,
                           u64& state)
{
    schema = l.schema[i];
    state = l.state_index[i];
    if (schema >= v.num_schemas || state >= rows)
        return false;
    const u32 arity = v.schema[static_cast<u64>(schema) * rl::dev::k_sc_count + rl::dev::k_sc_arity];
    if (arity > l.width || arity > v.max_bind)
        return false;
    for (u32 j = 0; j < arity; ++j)
    {
        bind[j] = l.binding[i * l.width + j];
        if (bind[j] >= v.num_objects)
            return false;
    }
    return true;
}

// Binding arrays live in registers/local memory; schemas of the device path have small arities (plan: <= 16).
constexpr u32 k_max_bind = 64;

__global__ void k_applicable(rl::dev::TaskView v, DeviceStates st, const u64* derived, u32 dw, DeviceLabels l, u8* out)
{
    u32 bind[k_max_bind];
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < l.count; i += u64{gridDim.x} * blockDim.x)
    {
        u32 schema = 0;
        u64 si = 0;
        if (v.max_bind > k_max_bind || !load_label(v, l, i, st.rows, bind, schema, si))
        {
            out[i] = 0;
            continue;
        }
        const u64* s = st.data + si * st.stride;
        const u64* d = derived ? derived + si * dw : nullptr;
        out[i] = rl::dev::is_applicable(v, schema, bind, s, st.words, d, dw) ? 1 : 0;
    }
}

__global__ void k_apply(rl::dev::TaskView v, DeviceStates st, const u64* derived, u32 dw, DeviceLabels l, u64* succ,
                        u32 out_words, u32* status)
{
    u32 bind[k_max_bind];
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < l.count; i += u64{gridDim.x} * blockDim.x)
    {
        u64* row = succ + i * out_words;
        u32 schema = 0;
        u64 si = 0;
        bool ok = v.max_bind <= k_max_bind && load_label(v, l, i, st.rows, bind, schema, si);
        const u64* s = ok ? st.data + si * st.stride : nullptr;
        if (ok)
            ok = rl::dev::is_applicable(v, schema, bind, s, st.words, derived ? derived + si * dw : nullptr, dw);
        if (!ok)
        {
            for (u32 w = 0; w < out_words; ++w)
                row[w] = 0;
            status[i] = rl::dev::k_none;
            continue;
        }
        const u32 r = rl::dev::apply_action(v, schema, bind, s, st.words, row, out_words);
        if (r != rl::dev::k_apply_ok)
            for (u32 w = 0; w < out_words; ++w)
                row[w] = 0;
        status[i] = r;
    }
}

__global__ void k_goal(rl::dev::TaskView v, DeviceStates st, const u64* derived, u32 dw, u8* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < st.rows; i += u64{gridDim.x} * blockDim.x)
        out[i] = rl::dev::goal_holds(v, st.data + i * st.stride, st.words, derived ? derived + i * dw : nullptr, dw) ? 1 : 0;
}

__global__ void k_iota(u64* dst, u64 n, u64 value)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        dst[i] = value + i;
}

__global__ void k_spin(u64 cycles)
{
    const long long start = clock64();
    while (static_cast<u64>(clock64() - start) < cycles)
    {
    }
}

__global__ void k_busy(u64* x, u64 n, int iters)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
    {
        u64 y = x[i];
        for (int k = 0; k < iters; ++k)
            y = y * 0x9E3779B97F4A7C15ull + 1;
        x[i] = y;
    }
}

__global__ void k_touch(u32* c) { atomicAdd(c, 1u); }
}  // namespace

cudaError_t launch_validate(const rl::dev::TaskView& v, u32* out, cudaStream_t s)
{
    const u64 items = rl::dev::validate_items(v);
    if (items)
        k_validate<<<grid_for(items), k_block, 0, s>>>(v, items, out);
    return cudaGetLastError();
}

cudaError_t launch_applicable(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words,
                              DeviceLabels labels, u8* out, cudaStream_t s)
{
    if (labels.count)
        k_applicable<<<grid_for(labels.count), k_block, 0, s>>>(v, states, derived, derived_words, labels, out);
    return cudaGetLastError();
}

cudaError_t launch_apply(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words,
                         DeviceLabels labels, u64* succ, u32 out_words, u32* status, cudaStream_t s)
{
    if (labels.count)
        k_apply<<<grid_for(labels.count), k_block, 0, s>>>(v, states, derived, derived_words, labels, succ, out_words, status);
    return cudaGetLastError();
}

cudaError_t launch_goal(const rl::dev::TaskView& v, DeviceStates states, const u64* derived, u32 derived_words, u8* out,
                        cudaStream_t s)
{
    if (states.rows)
        k_goal<<<grid_for(states.rows), k_block, 0, s>>>(v, states, derived, derived_words, out);
    return cudaGetLastError();
}

cudaError_t launch_iota(u64* dst, u64 n, u64 value, cudaStream_t s)
{
    if (n)
        k_iota<<<grid_for(n), k_block, 0, s>>>(dst, n, value);
    return cudaGetLastError();
}

cudaError_t launch_delayed_iota(u64* dst, u64 n, u64 value, u64 cycles, cudaStream_t s)
{
    k_spin<<<1, 1, 0, s>>>(cycles);
    if (n)
        k_iota<<<grid_for(n), k_block, 0, s>>>(dst, n, value);
    return cudaGetLastError();
}

cudaError_t launch_busy(u64* x, u64 n, int iters, cudaStream_t s)
{
    if (n)
        k_busy<<<1024, 256, 0, s>>>(x, n, iters);
    return cudaGetLastError();
}

cudaError_t launch_touch(u32* c, cudaStream_t s)
{
    k_touch<<<1, 1, 0, s>>>(c);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::kernels
