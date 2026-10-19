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
struct CollationTotals
{
    u64 entities = 0, tuples = 0, args = 0, targets = 0;
};
struct CollationParams
{
    const CollationSlice* slices = nullptr;
    u64 batches = 0, graphs = 0, relations = 0, entries = 0;
    bool index64 = false, relation_major = false;
    CollationTotals* totals = nullptr;
    void* temp = nullptr;
    u64 temp_bytes = 0;
    DeviceOutput output;
};
u64 collation_scan_bytes(u64 entries);
cudaError_t launch_collation(const CollationParams& params, int phase, cudaStream_t stream);
}
