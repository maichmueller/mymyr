#pragma once

#include "mymyr/core/types.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace mymyr::cuda
{
/// Small rows keep fixed-width kernel specializations. Wider classical rows are stored in 64-word chunks and
/// processed by runtime-width hashes, comparisons, goals and copies; only successor construction holds a chunk.
inline u32 search_row_words(u32 words)
{
    if (words <= 64)
        return std::bit_ceil(std::max<u32>(words, 1));
    // Atom slots and the successor kernels' word cursors are 32-bit.
    if (words > std::numeric_limits<u32>::max() / 64)
        throw std::length_error("mymyr: device search: row exceeds the 32-bit atom slot range");
    return (words + 63) / 64 * 64;
}
/// Default reservations hold at most 16 MiB of wide rows; narrow tasks retain their row-count tuning.
inline u64 row_reserve_floor(u32 words, u64 rows)
{
    return words <= 64 ? rows : std::min(rows, std::max<u64>(1, (u64{16} << 20) / (u64{words} * 8)));
}
}  // namespace mymyr::cuda
