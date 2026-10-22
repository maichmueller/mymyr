// The depth-one horizon builder on the device (cuda::build_depth_one_horizon): every graph and every successor in
// parallel, with the host builder's result.
//   check   a thread per graph flags offsets that do not delimit its slice of the expansion, or a slice with rows of
//           another parent (the first such graph and every graph after it get node offset -1);
//   edges   a thread per row of the valid graphs (those before the first invalid one) finds its graph (a binary
//           search of their offsets) and, merging, marks the rows equal to their root and inserts the others into an
//           open-addressing table over (graph, row) whose slots keep the smallest row of each class (atomicMin): the
//           first-seen successor;
//   reps    merging, each row looks its class up: the class's first row is a new node, every row of the class goes
//           to it;
//   scan    an exclusive scan of the new-node flags numbers the new nodes in batch order, so graph g's nodes start at
//           g + scan[offsets[g]] (its root first) and a new node of row e sits at g + 1 + scan[e];
//   nodes   the roots and the new nodes are copied to their rows, and the node offsets written;
//   order   unmerged, every row is its own node in expansion order, and the edges stay where they are; merged, a
//           stable radix sort of the rows by their destination's node gives the edges of each graph by destination,
//           the expansion's canonical label order within one.
// The table's slots depend on timing, the classes and their smallest rows do not: the result is deterministic.

#include "encoding_kernels.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <bit>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::cuda::detail
{
namespace
{
constexpr u32 k_build_block = 256;
constexpr u32 k_none = 0xffffffffu;       // an empty table slot; a row past the expansion
constexpr u32 k_root_node = 0xfffffffeu;  // a row equal to its root
constexpr u64 k_align = 256;

[[nodiscard]] constexpr u64 aligned(u64 bytes) { return (bytes + k_align - 1) & ~(k_align - 1); }

/// The scratch of one build, carved from one block (base null: sizes only).
struct BuildScratch
{
    u32* graph = nullptr;     // [capacity] the graph of each row
    u32* rep = nullptr;       // [capacity] the row's node class: its first row, or k_root_node
    u32* fresh = nullptr;     // [capacity + 1] 1 for a row that adds a node
    u32* rank = nullptr;      // [capacity + 1] the exclusive scan of fresh
    u32* invalid = nullptr;   // [1] the first invalid graph
    u32* table = nullptr;     // [table_size] merging: the classes' first rows
    u32* keys = nullptr;      // merging: [capacity] each row's destination node, sorted into keys_out
    u32* keys_out = nullptr;
    u32* rows = nullptr;      // merging: [capacity] the rows, sorted by key into rows_out
    u32* rows_out = nullptr;
    void* temp = nullptr;
    u64 temp_bytes = 0;
    u64 table_size = 0;
    u64 bytes = 0;
};

void sized(cudaError_t error, const char* what)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("mymyr: encoding: ") + what + ": " + cudaGetErrorString(error));
}

[[nodiscard]] u64 table_size_of(u64 capacity) { return std::bit_ceil(std::max<u64>(16, 2 * capacity)); }

/// The node keys of the sort: below graphs + capacity, which marks a row without an edge.
[[nodiscard]] u32 key_bits(u64 graphs, u64 capacity) { return std::max<u32>(1, std::bit_width(graphs + capacity)); }

BuildScratch carve(void* base, u64 capacity, bool merge)
{
    BuildScratch s;
    auto* p = static_cast<std::byte*>(base);
    u64 at = 0;
    auto take = [&](u64 count) -> u32*
    {
        u32* q = p ? reinterpret_cast<u32*>(p + at) : nullptr;
        at += aligned(count * sizeof(u32));
        return q;
    };
    s.graph = take(capacity);
    s.rep = take(capacity);
    s.fresh = take(capacity + 1);
    s.rank = take(capacity + 1);
    s.invalid = take(1);
    u64 temp = 0;
    sized(cub::DeviceScan::ExclusiveSum(nullptr, temp, static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                        static_cast<i64>(capacity + 1)),
          "cub::DeviceScan::ExclusiveSum (horizon builder)");
    if (merge)
    {
        s.table_size = table_size_of(capacity);
        s.table = take(s.table_size);
        s.keys = take(capacity);
        s.keys_out = take(capacity);
        s.rows = take(capacity);
        s.rows_out = take(capacity);
        u64 sort = 0;
        sized(cub::DeviceRadixSort::SortPairs(nullptr, sort, static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                              static_cast<const u32*>(nullptr), static_cast<u32*>(nullptr),
                                              static_cast<i64>(capacity)),
              "cub::DeviceRadixSort::SortPairs (horizon builder)");
        temp = std::max(temp, sort);
    }
    s.temp = p ? p + at : nullptr;
    s.temp_bytes = std::max<u64>(temp, 1);
    at += aligned(s.temp_bytes);
    s.bytes = at;
    return s;
}

u32 blocks_of(u64 threads)
{
    const u64 blocks = std::max<u64>(1, (threads + k_build_block - 1) / k_build_block);
    if (blocks > static_cast<u64>(std::numeric_limits<i32>::max()))
        throw std::overflow_error("mymyr: encoding: the horizon has too many rows to launch");
    return static_cast<u32>(blocks);
}

/// The graph of row e < offsets[graphs] of graphs whose offsets are valid: the last graph whose first offset is at
/// most e.
__device__ __forceinline__ u32 graph_of(const i32* offsets, u64 graphs, u64 e)
{
    u64 lo = 0, hi = graphs;
    while (hi - lo > 1)
    {
        const u64 mid = lo + (hi - lo) / 2;
        if (i64(offsets[mid]) <= i64(e))
            lo = mid;
        else
            hi = mid;
    }
    return u32(lo);
}

__device__ __forceinline__ bool same_row(const u64* a, const u64* b, u32 words)
{
    for (u32 w = 0; w < words; ++w)
        if (a[w] != b[w])
            return false;
    return true;
}

__device__ __forceinline__ u64 mix(u64 x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ u64 row_hash(const u64* row, u32 words, u32 graph)
{
    u64 h = mix(0x9e3779b97f4a7c15ull + graph);
    for (u32 w = 0; w < words; ++w)
        h = mix(h ^ row[w]);
    return h;
}

struct BuildArgs
{
    DeviceEncodeInput in;
    const u64* children;
    u64 child_stride;
    DeviceHorizonBuild out;
    BuildScratch s;
};

/// The graphs before the first invalid one, whose offsets build_check_kernel validated.
__device__ __forceinline__ u64 valid_graphs(const BuildArgs& b)
{
    const u64 invalid = *b.s.invalid;
    return b.in.graphs < invalid ? b.in.graphs : invalid;
}

/// The rows of the valid graphs: [0, offsets[valid_graphs()]), within the capacity. Rows of an invalid graph and
/// past it are not read: offsets past the first invalid graph may hold anything.
__device__ __forceinline__ u64 valid_rows(const BuildArgs& b)
{
    const u64 g = valid_graphs(b);
    return g ? u64(b.in.actions.offsets[g]) : 0;
}

__global__ void __launch_bounds__(k_build_block) build_check_kernel(const BuildArgs b)
{
    const u64 g = u64(blockIdx.x) * k_build_block + threadIdx.x;
    if (g >= b.in.graphs)
        return;
    const auto& a = b.in.actions;
    const i32 first = a.offsets[g], last = a.offsets[g + 1];
    bool bad = a.offsets[0] != 0 || first < 0 || last < first || u64(last) > a.capacity;
    if (a.parent)
        for (i32 e = first; !bad && e < last; ++e)
            bad = a.parent[e] != i32(g);
    if (bad)
        atomicMin(b.s.invalid, u32(g));
}

__global__ void __launch_bounds__(k_build_block) build_edges_kernel(const BuildArgs b)
{
    const u64 e = u64(blockIdx.x) * k_build_block + threadIdx.x;
    const auto& a = b.in.actions;
    if (e > a.capacity)
        return;
    const u64 G = valid_graphs(b);
    if (e >= valid_rows(b))
    {
        b.s.fresh[e] = 0;
        if (e < a.capacity)
            b.s.rep[e] = k_none;
        return;
    }
    const u32 g = graph_of(a.offsets, G, e);
    b.s.graph[e] = g;
    const u32 W = b.in.state_words;
    const u64* child = b.children + e * b.child_stride;
    if (!b.out.merge)
    {
        b.s.rep[e] = u32(e);
        b.s.fresh[e] = 1;
        return;
    }
    if (same_row(b.in.states + u64(g) * b.in.state_stride, child, W))
    {
        b.s.rep[e] = k_root_node;
        b.s.fresh[e] = 0;
        return;
    }
    b.s.rep[e] = k_none;  // build_reps_kernel finds the class
    const u64 mask = b.s.table_size - 1;
    for (u64 at = row_hash(child, W, g) & mask;; at = (at + 1) & mask)
    {
        u32 v = b.s.table[at];
        if (v == k_none)
        {
            v = atomicCAS(b.s.table + at, k_none, u32(e));
            if (v == k_none)
                break;
        }
        if (graph_of(a.offsets, G, v) == g && same_row(b.children + u64(v) * b.child_stride, child, W))
        {
            atomicMin(b.s.table + at, u32(e));
            break;
        }
    }
}

__global__ void __launch_bounds__(k_build_block) build_reps_kernel(const BuildArgs b)
{
    const u64 e = u64(blockIdx.x) * k_build_block + threadIdx.x;
    const auto& a = b.in.actions;
    if (e >= a.capacity || e >= valid_rows(b) || b.s.rep[e] == k_root_node)
        return;
    const u32 g = b.s.graph[e], W = b.in.state_words;
    const u64* child = b.children + e * b.child_stride;
    const u64 mask = b.s.table_size - 1;
    for (u64 at = row_hash(child, W, g) & mask;; at = (at + 1) & mask)
    {
        const u32 v = b.s.table[at];
        if (v != k_none && b.s.graph[v] == g && same_row(b.children + u64(v) * b.child_stride, child, W))
        {
            b.s.rep[e] = v;
            b.s.fresh[e] = v == u32(e) ? 1u : 0u;
            return;
        }
    }
}

/// The first node of graph g (its root), from the scan.
__device__ __forceinline__ u64 first_node(const BuildArgs& b, u64 g) { return g + b.s.rank[b.in.actions.offsets[g]]; }

__global__ void __launch_bounds__(k_build_block) build_nodes_kernel(const BuildArgs b)
{
    const u64 g = u64(blockIdx.x) * k_build_block + threadIdx.x;
    if (g == 0)
        b.out.node_offsets[0] = 0;
    if (g >= b.in.graphs)
        return;
    const bool valid = g < *b.s.invalid;
    const auto& a = b.in.actions;
    b.out.node_offsets[g + 1] = valid ? i32(g + 1 + b.s.rank[a.offsets[g + 1]]) : -1;
    if (!valid)
        return;
    const u32 W = b.in.state_words;
    u64* row = b.out.nodes + first_node(b, g) * W;
    const u64* root = b.in.states + g * b.in.state_stride;
    for (u32 w = 0; w < W; ++w)
        row[w] = root[w];
}

__device__ __forceinline__ void copy_label(const DeviceActionInput& a, const DeviceHorizonBuild& out, u64 from, u64 to)
{
    out.src[to] = 0;
    out.schema[to] = a.schema[from];
    for (u32 k = 0; k < a.label_width; ++k)
        out.binding[to * a.label_width + k] = a.binding[from * a.label_width + k];
}

__global__ void __launch_bounds__(k_build_block) build_successors_kernel(const BuildArgs b)
{
    const u64 e = u64(blockIdx.x) * k_build_block + threadIdx.x;
    const auto& a = b.in.actions;
    if (e >= a.capacity)
        return;
    const u32 pad = u32(b.in.graphs + a.capacity);
    if (e >= valid_rows(b))
    {
        if (b.out.merge)
        {
            b.s.keys[e] = pad;
            b.s.rows[e] = u32(e);
        }
        return;
    }
    const u32 g = b.s.graph[e], rep = b.s.rep[e], W = b.in.state_words;
    const u64 base = first_node(b, g);
    if (b.s.fresh[e])
    {
        u64* row = b.out.nodes + (g + 1 + u64(b.s.rank[e])) * W;
        const u64* child = b.children + e * b.child_stride;
        for (u32 w = 0; w < W; ++w)
            row[w] = child[w];
    }
    const u64 node = rep == k_root_node ? base : g + 1 + u64(b.s.rank[rep]);
    if (!b.out.merge)
    {
        b.out.dst[e] = i32(node - base);
        copy_label(a, b.out, e, e);
        return;
    }
    b.s.keys[e] = u32(node);
    b.s.rows[e] = u32(e);
}

__global__ void __launch_bounds__(k_build_block) build_edges_out_kernel(const BuildArgs b)
{
    const u64 i = u64(blockIdx.x) * k_build_block + threadIdx.x;
    const auto& a = b.in.actions;
    if (i >= a.capacity || b.s.keys_out[i] == u32(b.in.graphs + a.capacity))
        return;
    const u64 e = b.s.rows_out[i];
    b.out.dst[i] = i32(b.s.keys_out[i] - first_node(b, b.s.graph[e]));
    copy_label(a, b.out, e, i);
}
}  // namespace

u64 horizon_builder_scratch_bytes(u64 capacity, bool merge) { return carve(nullptr, capacity, merge).bytes; }

cudaError_t launch_horizon_builder(const DeviceEncodeInput& parents, const u64* children, u64 child_stride,
                                   const DeviceHorizonBuild& output, cudaStream_t stream)
{
    const u64 M = parents.actions.capacity, G = parents.graphs;
    BuildArgs b{parents, children, child_stride, output, carve(output.scratch, M, output.merge)};
    cudaError_t error = cudaMemsetAsync(b.s.invalid, 0xff, sizeof(u32), stream);
    if (error == cudaSuccess && output.merge)
        error = cudaMemsetAsync(b.s.table, 0xff, b.s.table_size * sizeof(u32), stream);
    if (error != cudaSuccess)
        return error;
    build_check_kernel<<<blocks_of(G), k_build_block, 0, stream>>>(b);
    build_edges_kernel<<<blocks_of(M + 1), k_build_block, 0, stream>>>(b);
    if (output.merge)
        build_reps_kernel<<<blocks_of(M), k_build_block, 0, stream>>>(b);
    if ((error = cudaPeekAtLastError()) != cudaSuccess)
        return error;
    u64 temp = b.s.temp_bytes;
    error = cub::DeviceScan::ExclusiveSum(b.s.temp, temp, b.s.fresh, b.s.rank, static_cast<i64>(M + 1), stream);
    if (error != cudaSuccess)
        return error;
    build_nodes_kernel<<<blocks_of(G), k_build_block, 0, stream>>>(b);
    if (M)
        build_successors_kernel<<<blocks_of(M), k_build_block, 0, stream>>>(b);
    if (!output.merge || !M)
        return cudaPeekAtLastError();
    temp = b.s.temp_bytes;
    error = cub::DeviceRadixSort::SortPairs(b.s.temp, temp, b.s.keys, b.s.keys_out, b.s.rows, b.s.rows_out,
                                            static_cast<i64>(M), 0, static_cast<int>(key_bits(G, M)), stream);
    if (error != cudaSuccess)
        return error;
    build_edges_out_kernel<<<blocks_of(M), k_build_block, 0, stream>>>(b);
    return cudaPeekAtLastError();
}
}  // namespace mymyr::cuda::detail
