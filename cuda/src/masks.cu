// The lifted action mask kernels (include/mymyr/cuda/mask_kernels.hpp): the schemas' mask matchers run by the device
// matcher loops of lifted_device.cuh. Compiled by nvcc as C++23; includes only the device-code subset.

#include "mymyr/cuda/mask_kernels.hpp"

#include "lifted_device.cuh"

namespace mymyr::cuda::maskk
{
namespace
{
using namespace rl::dev;

constexpr unsigned k_block = 256;
constexpr u32 k_lanes = 32;

unsigned grid(u64 threads)
{
    const u64 g = (threads + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

__device__ __forceinline__ u32 instance_of(const Rows& r, u64 i)
{
    return r.task_ids ? static_cast<u32>(r.task_ids[i]) : 0u;
}

__device__ __forceinline__ i32 schema_of(const SchemaArg& a, u64 i) { return a.schema ? a.schema[i] : a.value; }

/// The mask matcher `entry` of schema k (0: existence, d + 1: parameter d).
__device__ __forceinline__ const u32* mask_matcher(const TaskView& t, u32 k, u32 entry)
{
    return t.matcher + (u64{t.schema[u64{k} * k_sc_count + k_sc_masks]} + entry) * k_mc_count;
}

/// Runs a mask matcher with the device's preferred loop (forward checking where the export marks it and the object
/// bitsets allow it). Split: k_lanes lanes deal the candidates of the first level, which the matcher enters before
/// any witness search (the enumerated parameter), so that every candidate searches its witnesses whole.
template<u32 OW, bool Split, class Emit>
__device__ __forceinline__ void run(const TaskView& t, const u32* mx, const lifted::StateRef& s, const u64* view,
                                    u32* bind, Emit& emit, u32 lane)
{
    if constexpr (OW <= k_max_fc_ow)
    {
        if (device_fc(mx[k_mc_flags], MatchOrder::Free))
        {
            lifted::run_fc<OW, Split, Split>(t, mx, s, view, bind, emit, Split ? k_lanes : 1, lane);
            return;
        }
    }
    lifted::run_fixed<OW, lifted::k_max_depth, Split, Split>(t, mx, s, view, bind, emit, Split ? k_lanes : 1, lane);
}

__global__ void k_check(Rows r, const u32* words, const u32* slots, u32* status)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < r.rows; i += u64{gridDim.x} * blockDim.x)
    {
        const u32 k = instance_of(r, i);
        if (k >= r.instances)
        {
            atomicOr(status, k_bad_task_id);
            continue;
        }
        if (r.words < words[k])
        {
            atomicOr(status, k_narrow_rows);
            continue;
        }
        const u32 limit = slots[k];
        if ((limit & 63) && (r.data[i * r.stride + (limit >> 6)] >> (limit & 63)))
            atomicOr(status, k_bad_row);
    }
}

template<u32 OW>
__global__ void k_schema_masks(Rows r, ViewRows v, u32 S, u8* out)
{
    u32 bind[lifted::k_max_depth];
    const u64 n = r.rows * S;
    for (u64 x = u64{blockIdx.x} * blockDim.x + threadIdx.x; x < n; x += u64{gridDim.x} * blockDim.x)
    {
        const u64 i = x / S;
        const u32 k = static_cast<u32>(x % S), inst = instance_of(r, i);
        if (inst >= r.instances)
            continue;
        const TaskView& t = r.views[inst];
        if (t.ow != OW || k >= t.num_schemas)
            continue;
        const lifted::StateRef st{r.data + i * r.stride, r.words, nullptr, 0};
        bool found = false;
        auto emit = [&](const u32*)
        {
            found = true;
            return false;
        };
        run<OW, false>(t, mask_matcher(t, k, 0), st, v.data + i * v.words, bind, emit, 0);
        if (found)
            out[x] = 1;
    }
}

template<u32 OW>
__global__ void k_object_masks(Rows r, ViewRows v, ObjectMasks m, u32* status)
{
    u32 bind[lifted::k_max_depth];
    const u32 lane = threadIdx.x % k_lanes;
    const u64 warps = u64{gridDim.x} * blockDim.x / k_lanes;
    for (u64 i = (u64{blockIdx.x} * blockDim.x + threadIdx.x) / k_lanes; i < r.rows; i += warps)
    {
        // every lane of the warp takes the same branches up to the search
        const u32 inst = instance_of(r, i);
        if (inst >= r.instances)
            continue;
        const TaskView& t = r.views[inst];
        const i32 sc = schema_of(m.schema, i);
        if (t.ow != OW || sc < 0 || static_cast<u32>(sc) >= t.num_schemas)
            continue;
        const u32 k = static_cast<u32>(sc), arity = t.schema[u64{k} * k_sc_count + k_sc_arity];
        if (m.depth >= arity)
        {
            if (lane == 0)
                atomicOr(status, k_bad_depth);
            continue;
        }
        const u32* mx = mask_matcher(t, k, m.depth + 1);
        // a prefix object outside its parameter's static domain: no binding
        const u64* dom = t.dom0 + mx[k_mc_dom0];
        bool ok = true;
        for (u32 d = 0; d < m.depth && ok; ++d)
        {
            const u32 o = static_cast<u32>(m.prefix[i * m.stride + d]);
            ok = o < t.num_objects && ((dom[u64{d} * OW + o / 64] >> (o % 64)) & 1) != 0;
            bind[d] = o;
        }
        if (!ok)
            continue;
        u8* row = m.out + i * m.num_objects;
        const u32 depth = m.depth;
        auto emit = [&](const u32* b)
        {
            row[b[depth]] = 1;
            return true;
        };
        const lifted::StateRef st{r.data + i * r.stride, r.words, nullptr, 0};
        run<OW, true>(t, mx, st, v.data + i * v.words, bind, emit, lane);
    }
}

__global__ void k_applicable(Rows r, SchemaArg a, const i32* binding, u32 width, u8* out, u32* status)
{
    u32 bind[lifted::k_max_depth];
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < r.rows; i += u64{gridDim.x} * blockDim.x)
    {
        const u32 inst = instance_of(r, i);
        const i32 sc = schema_of(a, i);
        bool ok = inst < r.instances && sc >= 0;
        const TaskView* t = ok ? r.views + inst : nullptr;
        ok = ok && static_cast<u32>(sc) < t->num_schemas;
        if (ok)
        {
            const u32 arity = t->schema[u64{static_cast<u32>(sc)} * k_sc_count + k_sc_arity];
            if (arity > width)
            {
                atomicOr(status, k_narrow_binding);
                ok = false;
            }
            for (u32 j = 0; j < arity && ok; ++j)
            {
                bind[j] = static_cast<u32>(binding[i * width + j]);
                ok = bind[j] < t->num_objects;
            }
            ok = ok && is_applicable(*t, static_cast<u32>(sc), bind, r.data + i * r.stride, r.words, nullptr, 0);
        }
        out[i] = ok ? 1 : 0;
    }
}

template<u32 OW>
__global__ void k_any(Rows r, ViewRows v, const u8* skip, u32* out, u64 stride, u32 S)
{
    u32 bind[lifted::k_max_depth];
    const u64 n = r.rows * S;
    for (u64 x = u64{blockIdx.x} * blockDim.x + threadIdx.x; x < n; x += u64{gridDim.x} * blockDim.x)
    {
        const u64 i = x / S;
        const u32 k = static_cast<u32>(x % S), inst = instance_of(r, i);
        if ((skip && skip[i]) || inst >= r.instances)
            continue;
        const TaskView& t = r.views[inst];
        if (t.ow != OW || k >= t.num_schemas)
            continue;
        const lifted::StateRef st{r.data + i * r.stride, r.words, nullptr, 0};
        bool found = false;
        auto emit = [&](const u32*)
        {
            found = true;
            return false;
        };
        run<OW, false>(t, mask_matcher(t, k, 0), st, v.data + i * v.words, bind, emit, 0);
        if (found)
            out[i * stride] = 1;
    }
}

__global__ void k_any_clear(u64 rows, const u8* skip, u32* out, u64 stride)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
        if (!skip || !skip[i])
            out[i * stride] = 0;
}

/// Calls f.template operator()<OW>() for the width class ow (cudaErrorInvalidValue for another width).
template<class F>
cudaError_t by_width(u32 ow, F&& f)
{
    switch (ow)
    {
        case 1: return f.template operator()<1>();
        case 2: return f.template operator()<2>();
        case 3: return f.template operator()<3>();
        case 4: return f.template operator()<4>();
        case 5: return f.template operator()<5>();
        case 6: return f.template operator()<6>();
        case 7: return f.template operator()<7>();
        case 8: return f.template operator()<8>();
        case 12: return f.template operator()<12>();
        case 16: return f.template operator()<16>();
        case 24: return f.template operator()<24>();
        case 32: return f.template operator()<32>();
        default: return cudaErrorInvalidValue;
    }
}
}  // namespace

cudaError_t launch_check(const Rows& r, const u32* words, const u32* slots, u32* status, cudaStream_t s)
{
    if (r.rows)
        k_check<<<grid(r.rows), k_block, 0, s>>>(r, words, slots, status);
    return cudaGetLastError();
}

cudaError_t launch_schema_masks(const Rows& r, ViewRows v, u32 ow, u32 num_schemas, u8* out, cudaStream_t s)
{
    if (r.rows == 0 || num_schemas == 0)
        return cudaSuccess;
    return by_width(ow, [&]<u32 OW>()
    {
        k_schema_masks<OW><<<grid(r.rows * num_schemas), k_block, 0, s>>>(r, v, num_schemas, out);
        return cudaGetLastError();
    });
}

cudaError_t launch_object_masks(const Rows& r, ViewRows v, u32 ow, const ObjectMasks& m, u32* status, cudaStream_t s)
{
    if (r.rows == 0)
        return cudaSuccess;
    return by_width(ow, [&]<u32 OW>()
    {
        k_object_masks<OW><<<grid(r.rows * k_lanes), k_block, 0, s>>>(r, v, m, status);
        return cudaGetLastError();
    });
}

cudaError_t launch_applicable(const Rows& r, SchemaArg schema, const i32* binding, u32 width, u8* out, u32* status,
                              cudaStream_t s)
{
    if (r.rows)
        k_applicable<<<grid(r.rows), k_block, 0, s>>>(r, schema, binding, width, out, status);
    return cudaGetLastError();
}

cudaError_t launch_any(const Rows& r, ViewRows v, u32 ow, u32 num_schemas, const u8* skip, u32* out, u64 stride,
                       cudaStream_t s)
{
    if (r.rows == 0 || num_schemas == 0)
        return cudaSuccess;
    return by_width(ow, [&]<u32 OW>()
    {
        k_any<OW><<<grid(r.rows * num_schemas), k_block, 0, s>>>(r, v, skip, out, stride, num_schemas);
        return cudaGetLastError();
    });
}

cudaError_t launch_any_clear(u64 rows, const u8* skip, u32* out, u64 stride, cudaStream_t s)
{
    if (rows)
        k_any_clear<<<grid(rows), k_block, 0, s>>>(rows, skip, out, stride);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::maskk
