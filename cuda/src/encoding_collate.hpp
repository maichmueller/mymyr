#pragma once

#include "mymyr/cuda/encoding_types.hpp"
#include <cuda_runtime_api.h>

namespace mymyr::cuda::detail
{
struct CollationSlice
{
    DeviceCollationBatch batch;
    u64 first = 0;
};
/// Scanned in place to the entities, targets, Simple nodes and edges before each graph (entry 0 stays zero).
struct GraphCounts
{
    u64 entities = 0, targets = 0, nodes = 0, edges = 0;
};
/// Scanned in place to the tuples and arguments before each cell (entry 0 stays zero).
struct CellCounts
{
    u64 tuples = 0, args = 0;
};
/// Scratch: graph counts [graphs + 1], cell counts [cells + 1], the source overflow word, then the scan's temporary
/// storage, each part 256-byte aligned.
struct CollationParams
{
    const CollationSlice* slices = nullptr;
    u64 batches = 0, graphs = 0, relations = 0, cells = 0;
    bool index64 = false, relation_major = false, relations_stored = true, simple = false;
    u32 grid = 1;  // blocks of the grid-stride count and fill
    GraphCounts* graph_counts = nullptr;
    CellCounts* cell_counts = nullptr;
    u64* bad = nullptr;  // nonzero: a source batch reported overflow
    void* temp = nullptr;
    u64 temp_bytes = 0;
    DeviceOutput output;
};
/// The temporary storage of both scans.
u64 collation_scan_bytes(u64 graphs, u64 cells);
/// phase 0: count, scan and fill; 1: count, scan and the sizes; 2: fill after a phase 1 on the same scratch.
cudaError_t launch_collation(const CollationParams& params, int phase, cudaStream_t stream);
}
