// Each root traverses the space's recorded unique ids in breadth-first order. Independent graph scratch preserves
// first-seen numbering; scans compact the slices before parallel copies write the node rows and canonical edges.
#include "encoding_kernels.hpp"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::cuda::detail
{
namespace
{
constexpr u32 k_space_block = 128;

struct SpaceScratch
{
    i32* map;
    i32* queue;
    i32* src;
    i32* dst;
    i32* edge;
    i32* order;
    i32* nodes;
    i32* edges;
    u32* invalid;
    void* temp;
    u64 temp_bytes, bytes;
};

SpaceScratch space_carve(void* base, u32 states, u64 edges, u32 graphs)
{
    auto* p = static_cast<std::byte*>(base);
    u64 at = 0;
    auto take = [&](u64 count)
    {
        if (at > std::numeric_limits<u64>::max() - 255 ||
            count > (std::numeric_limits<u64>::max() - at - 255) / sizeof(i32))
            throw std::length_error("mymyr: encoding: state-space horizon scratch exceeds the address space");
        auto* q = p ? reinterpret_cast<i32*>(p + at) : nullptr;
        at += (count * sizeof(i32) + 255) & ~u64{255};
        return q;
    };
    SpaceScratch s;
    s.map = take(u64(graphs) * states);
    s.queue = take(u64(graphs) * states);
    s.src = take(u64(graphs) * edges);
    s.dst = take(u64(graphs) * edges);
    s.edge = take(u64(graphs) * edges);
    s.order = take(u64(graphs) * edges);
    s.nodes = take(u64(graphs) + 1);
    s.edges = take(u64(graphs) + 1);
    s.invalid = reinterpret_cast<u32*>(take(1));
    s.temp_bytes = 0;
    const auto error = cub::DeviceScan::ExclusiveSum(nullptr, s.temp_bytes, static_cast<const i32*>(nullptr),
                                                    static_cast<i32*>(nullptr), i64(graphs) + 1);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("mymyr: encoding: state-space horizon scan: ") + cudaGetErrorString(error));
    if (s.temp_bytes > std::numeric_limits<u64>::max() - at - 255)
        throw std::length_error("mymyr: encoding: state-space horizon scratch exceeds the address space");
    s.temp = p ? p + at : nullptr;
    s.bytes = at + ((std::max<u64>(1, s.temp_bytes) + 255) & ~u64{255});
    return s;
}

__device__ bool space_edge_before(const DeviceHorizonSpace& space, const SpaceScratch& s, u64 base, u32 a, u32 b)
{
    if (s.dst[base + a] != s.dst[base + b])
        return s.dst[base + a] < s.dst[base + b];
    const u32 ea = u32(s.edge[base + a]), eb = u32(s.edge[base + b]);
    if (space.schemas[ea] != space.schemas[eb])
        return space.schemas[ea] < space.schemas[eb];
    for (u32 k = 0; k < space.label_width; ++k)
        if (space.bindings[u64(ea) * space.label_width + k] != space.bindings[u64(eb) * space.label_width + k])
            return static_cast<i32>(space.bindings[u64(ea) * space.label_width + k]) <
                   static_cast<i32>(space.bindings[u64(eb) * space.label_width + k]);
    return a < b;  // Equal labels keep their traversal order.
}

__device__ void space_edge_sift(const DeviceHorizonSpace& space, const SpaceScratch& s, u64 base,
                               u32 count, u32 root)
{
    i32* const order = s.order + base;
    while (u64(root) * 2 + 1 < count)
    {
        u32 child = 2 * root + 1;
        if (child + 1 < count && space_edge_before(space, s, base, u32(order[child]), u32(order[child + 1])))
            ++child;
        if (!space_edge_before(space, s, base, u32(order[root]), u32(order[child])))
            break;
        const i32 t = order[root]; order[root] = order[child]; order[child] = t;
        root = child;
    }
}

__global__ void space_traverse_kernel(const DeviceHorizonSpace space, const DeviceSpaceHorizonBuild out,
                                      const SpaceScratch s)
{
    const u32 g = blockIdx.x;
    i32* const map = s.map + u64(g) * space.states;
    i32* const queue = s.queue + u64(g) * space.states;
    for (u64 n = threadIdx.x; n < space.states; n += blockDim.x)
        map[n] = -1;
    __syncthreads();
    if (threadIdx.x)
        return;
    if (g == 0)
        s.nodes[out.graphs] = s.edges[out.graphs] = 0;
    bool valid = u32(out.roots[g]) < space.states;
    bool truncated = false;
    const u32 limit = out.limit ? min(out.limit, space.states) : space.states;
    u32 count = valid ? 1 : 0, first = 0, last = count, edges = 0;
    const u64 edge_base = u64(g) * space.edges;
    if (valid)
    {
        queue[0] = out.roots[g];
        map[u32(queue[0])] = 0;
    }
    for (u32 level = 0; valid && level < out.depth && first < last; ++level)
    {
        for (u32 src = first; valid && src < last; ++src)
        {
            const u32 state = u32(queue[src]);
            const u64 begin = space.offsets[state], end = space.offsets[state + 1];
            if (begin > end || end > space.edges)
            {
                valid = false;
                break;
            }
            for (u64 e = begin; e < end; ++e)
            {
                const u32 target = space.targets[e];
                if (target >= space.states || edges == space.edges)
                {
                    valid = false;
                    break;
                }
                i32 dst = map[target];
                if (dst < 0)
                {
                    if (count == limit)
                    {
                        truncated = true;
                        if (!out.truncate)
                            valid = false;
                        if (!valid)
                            break;
                        continue;
                    }
                    dst = static_cast<i32>(count);
                    map[target] = dst;
                    queue[count++] = static_cast<i32>(target);
                }
                const u64 at = edge_base + edges;
                s.src[at] = static_cast<i32>(src);
                s.dst[at] = dst;
                s.edge[at] = static_cast<i32>(e);
                s.order[at] = static_cast<i32>(edges++);
            }
        }
        first = last;
        last = count;
    }
    s.nodes[g] = valid ? static_cast<i32>(count) : 0;
    s.edges[g] = valid ? static_cast<i32>(edges) : 0;
    if (out.truncated)
        out.truncated[g] = truncated;
    if (!valid)
    {
        atomicMin(s.invalid, g);
        return;
    }
    for (u32 i = edges / 2; i > 0; --i)
        space_edge_sift(space, s, edge_base, edges, i - 1);
    for (u32 end = edges; end > 1; --end)
    {
        i32* const order = s.order + edge_base;
        const i32 t = order[0]; order[0] = order[end - 1]; order[end - 1] = t;
        space_edge_sift(space, s, edge_base, end - 1, 0);
    }
}

__global__ void space_copy_kernel(const DeviceHorizonSpace space, const DeviceSpaceHorizonBuild out,
                                  const SpaceScratch s)
{
    const u32 g = blockIdx.x;
    if (g >= *s.invalid)
        return;
    const u64 node_base = u32(out.node_offsets[g]), edge_base = u32(out.edge_offsets[g]);
    const i32* const queue = s.queue + u64(g) * space.states;
    for (u64 i = threadIdx.x; i < u64(s.nodes[g]) * space.words; i += blockDim.x)
        out.nodes[node_base * space.words + i] = space.rows[u64(queue[i / space.words]) * space.words + i % space.words];
    const u64 base = u64(g) * space.edges;
    for (u64 i = threadIdx.x; i < u32(s.edges[g]); i += blockDim.x)
    {
        const u64 at = base + u32(s.order[base + i]);
        const u64 e = u32(s.edge[at]), dst = edge_base + i;
        out.src[dst] = s.src[at];
        out.dst[dst] = s.dst[at];
        out.schema[dst] = static_cast<i32>(space.schemas[e]);
        for (u32 k = 0; k < space.label_width; ++k)
            out.binding[dst * space.label_width + k] = static_cast<i32>(space.bindings[e * space.label_width + k]);
    }
}

__global__ void space_invalid_offsets_kernel(const DeviceSpaceHorizonBuild out, const u32* invalid)
{
    const u64 g = u64(blockIdx.x) * k_space_block + threadIdx.x;
    if (g < out.graphs && g >= *invalid)
        out.node_offsets[g + 1] = out.edge_offsets[g + 1] = -1;
}
}  // namespace

u64 space_horizon_builder_scratch_bytes(u32 states, u64 edges, u32 graphs)
{
    return space_carve(nullptr, states, edges, graphs).bytes;
}

cudaError_t launch_space_horizons(const DeviceHorizonSpace& space, const DeviceSpaceHorizonBuild& out,
                                cudaStream_t stream)
{
    if (!out.graphs)
    {
        auto error = cudaMemsetAsync(out.node_offsets, 0, sizeof(i32), stream);
        return error == cudaSuccess ? cudaMemsetAsync(out.edge_offsets, 0, sizeof(i32), stream) : error;
    }
    auto s = space_carve(out.scratch, space.states, space.edges, out.graphs);
    auto error = cudaMemsetAsync(s.invalid, 0xff, sizeof(u32), stream);
    if (error != cudaSuccess)
        return error;
    space_traverse_kernel<<<out.graphs, k_space_block, 0, stream>>>(space, out, s);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        return error;
    error = cub::DeviceScan::ExclusiveSum(s.temp, s.temp_bytes, s.nodes, out.node_offsets, i64(out.graphs) + 1, stream);
    if (error != cudaSuccess)
        return error;
    error = cub::DeviceScan::ExclusiveSum(s.temp, s.temp_bytes, s.edges, out.edge_offsets, i64(out.graphs) + 1, stream);
    if (error != cudaSuccess)
        return error;
    space_copy_kernel<<<out.graphs, k_space_block, 0, stream>>>(space, out, s);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        return error;
    space_invalid_offsets_kernel<<<(u64(out.graphs) + k_space_block - 1) / k_space_block, k_space_block, 0, stream>>>(out, s.invalid);
    return cudaPeekAtLastError();
}
}  // namespace mymyr::cuda::detail
