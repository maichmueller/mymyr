#pragma once
// Per-search ground conjunctions: fluent and derived masks plus numeric comparison programs. Numeric rows use the
// internal double encoding. All pointers address device memory; the goal index of a row selects its conjunction.

#include "mymyr/cuda/lifted.hpp"

namespace mymyr::cuda::goal
{
struct View
{
    u32 count = 0, words = 0, derived_words = 0;
    const u64* positive = nullptr;
    const u64* negative = nullptr;
    const u64* derived_positive = nullptr;
    const u64* derived_negative = nullptr;
    const u32* offsets = nullptr;  // [count + 1]: numeric comparisons of each conjunction
    rl::dev::NumericView numeric;
};

/// Parents are gathered rows; order maps them to the original rows whose goal ids are in row_goals. An order index
/// outside row_count is absent and receives flag zero. Null order selects row i.
cudaError_t launch_flags(View goals, lifted::Parents parents, const u32* row_goals, const u32* order,
                          u64 row_count, u64 n, u8* flags, cudaStream_t stream);
}  // namespace mymyr::cuda::goal
