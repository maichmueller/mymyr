#pragma once
// The launches of the relational encoder on the device (include/mymyr/cuda/encoding.hpp builds the parameters). The
// encoding of a batch is four kernels on one stream, with no allocation and no synchronization, which a CUDA graph can
// capture:
//   count  one group of 4, 8 or 32 lanes per graph counts its cells (encoding::detail::count_graph_group), writes the
//          graph's totals and cell counts into the scratch, and flags invalid rows;
//   scan   one inclusive scan over the scratch's totals, two kernels (cub::DeviceScan): the entities, the targets and,
//          graph-major, the tuples and arguments of every graph, relation-major the tuples of every (relation, graph)
//          cell, in this order, each row opened by a zero so that its entries are the exclusive prefix sums plus the
//          sums of the rows before it;
//   fill   a thread per graph (narrow rows) or a warp per graph checks the sizes against the capacities and the index
//          type, and then writes the graph from the positions of the scan (encoding::detail::fill_graph and its group
//          and relation-major forms).
// The launch order of the graphs only decides which thread or warp takes which graph: every position is a prefix sum
// in batch order.

#include "mymyr/core/types.hpp"
#include "mymyr/encoding/buffers.hpp"
#include "mymyr/encoding/detail/lanes.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::detail
{
/// Everything the kernels read: device pointers and sizes.
struct EncodeParams
{
    encoding::detail::ContextView context;
    u64 graphs = 0;
    // the rows (device memory)
    const u64* states = nullptr;
    u64 state_stride = 0;  // words from one row to the next
    u32 state_words = 0;
    const i32* task_ids = nullptr;
    const u64* derived = nullptr;
    u64 derived_stride = 0;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    u64 goal_stride = 0;
    u32 goal_words = 0;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    u64 goal_derived_stride = 0;
    u32 goal_derived_words = 0;
    const u32* order = nullptr;  // [graphs] the graph each thread or warp takes (null: in batch order)
    // the batch
    encoding::FlatBuffers out;
    encoding::FlatSizes* sizes = nullptr;
    encoding::FlatCapacities capacities;
    u32 static_overflow = 0;  // the k_overflow_ flags that follow from the shape alone
    bool index64 = false;
    bool relation_major = false;
    // the scratch
    void* totals = nullptr;  // the scanned array: u32, u64 when wide
    bool wide = false;
    u64 segments = 0;        // relation-major: the index of the first entry of the cell row in `totals`
    u32* flags = nullptr;    // flags[0]: a row is invalid
    u32* cell_counts = nullptr;  // graph-major: [graphs, cell_stride] the counts of each graph's cells
    u64 cell_stride = 0;
};

/// How a batch is launched.
struct LaunchShape
{
    u32 lanes = 4;            // lanes per graph of the count kernel: 4, 8 or 32
    bool warp_fill = false;   // a warp per graph in the fill kernel, else a thread
    u64 scan_length = 0;      // entries of the scanned array
    void* scan_temp = nullptr;
    u64 scan_temp_bytes = 0;
    void* zero_begin = nullptr;  // cleared before the count: the flags and, relation-major, the cell row
    u64 zero_bytes = 0;
};

/// The threads of the launches' blocks.
inline constexpr u32 k_encode_block = 128;
/// The dynamic shared memory a launch may use without opting in.
inline constexpr u64 k_shared_limit = 48 * 1024;

/// The temporary storage of the scan of `length` entries, u64 entries when `wide`.
[[nodiscard]] u64 totals_scan_temp_bytes(u64 length, bool wide);

/// The dynamic shared memory of the count and fill kernels: the virtual-node scratch of every graph in flight, and, for
/// the relation-major fill, the argument position of every relation.
[[nodiscard]] u64 count_shared_bytes(u32 virtual_nodes, u32 lanes);
[[nodiscard]] u64 fill_shared_bytes(u32 virtual_nodes, u32 relations, bool warp_fill, bool relation_major);

/// Enqueues the encoding on `stream`: a clear of the flags (and cell row), the count kernel, the scan and the fill kernel.
cudaError_t launch_encode(const EncodeParams& params, const LaunchShape& shape, cudaStream_t stream);
}  // namespace mymyr::cuda::detail
