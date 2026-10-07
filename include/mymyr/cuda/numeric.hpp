#pragma once
// Conversions between CPU state encodings and device rows [atom words | double fluent values]. The atom width is
// total_words - task.numeric_slots(); any padding belongs to the atom block, so the numeric block has a fixed tail.

#include "mymyr/task/task.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <vector>

namespace mymyr::cuda::numeric
{
inline void encode(const Task& task, StateView state, u64* row, u32 total_words)
{
    const u32 slots = task.numeric_slots();
    if (total_words < slots || state.nnum != task.numeric_words())
        throw std::invalid_argument("mymyr: device numeric state has the wrong width");
    const u32 words = total_words - slots;
    std::fill_n(row, total_words, 0);
    std::copy_n(state.w, std::min(state.nw, words), row);
    for (u32 i = 0; i < slots; ++i)
        row[words + i] = std::bit_cast<u64>(plan::load(task.compiled().num, state.num, i));
}

[[nodiscard]] inline State decode(const Task& task, const u64* row, u32 total_words)
{
    const u32 slots = task.numeric_slots();
    if (total_words < slots)
        throw std::invalid_argument("mymyr: device numeric state has the wrong width");
    const u32 words = total_words - slots;
    std::vector<u64> values(task.numeric_words(), 0);
    for (u32 i = 0; i < slots; ++i)
        plan::store(task.compiled().num, values.data(), i, std::bit_cast<f64>(row[words + i]));
    return State(row, words, values.data(), static_cast<u32>(values.size()));
}
}  // namespace mymyr::cuda::numeric
