#pragma once

#include "mymyr/core/types.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace mymyr::cuda
{
/// Small rows keep fixed-width kernel specializations. Wider classical rows are stored in 64-word chunks and
/// processed by runtime-width hashes, comparisons, goals and copies; only successor construction holds a chunk.
inline u32 search_row_words(u32 words)
{
    if (!lifted::wide_row(words))
        return std::bit_ceil(std::max<u32>(words, 1));
    // Atom slots and the successor kernels' word cursors are 32-bit.
    if (words > std::numeric_limits<u32>::max() / 64)
        throw std::length_error("mymyr: device search: row exceeds the 32-bit atom slot range");
    return (words + 63) / 64 * 64;
}
/// search_row_words of rows of `atom_words` atom words followed by `numeric_slots` numeric values (a word each). The
/// numeric kernels hold whole rows: numeric rows past lifted::k_max_words words are refused, also when a lazy task's
/// rows grow there during a search.
/// Refuses numeric rows of `words` words (atoms and numeric values) past the numeric kernels' lifted::k_max_words.
inline void check_numeric_row(u32 words)
{
    if (lifted::wide_row(words))
        throw std::invalid_argument("mymyr: device search: numeric states of " + std::to_string(words) +
                                    " words exceed the numeric kernels' " + std::to_string(lifted::k_max_words) +
                                    " words; use the CPU search");
}
inline u32 search_row_words(u32 atom_words, u32 numeric_slots)
{
    const u32 words = atom_words + numeric_slots;
    if (numeric_slots)
        check_numeric_row(words);
    return search_row_words(words);
}
/// Default reservations hold at most 16 MiB of wide rows; narrow tasks retain their row-count tuning.
inline u64 row_reserve_floor(u32 words, u64 rows)
{
    return !lifted::wide_row(words) ? rows : std::min(rows, std::max<u64>(1, (u64{16} << 20) / (u64{words} * 8)));
}

/// Bytes the context's pool can still hand out: its limit (ContextOptions::max_bytes, and the device's free memory
/// plus what the pool holds unused) less what is handed out.
inline u64 device_room(const DeviceContext& ctx)
{
    const PoolUsage u = ctx.usage();
    std::size_t free = 0, total = 0;
    {
        DeviceGuard g(ctx.device());
        check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    }
    u64 limit = u.reserved + free;
    if (ctx.options().max_bytes)
        limit = std::min<u64>(limit, ctx.options().max_bytes);
    return limit > u.used ? limit - u.used : 0;
}

/// Refuses a state (or a chunk of states) whose successors need more device memory than the context has left: the
/// count, the bytes (0: more than `room`, an allocation for them failed) and the options that change it.
[[noreturn]] inline void refuse_successors(const DeviceContext& ctx, const std::string& search, const std::string& what,
                                           u64 successors, u64 bytes, u64 room, const std::string& advice)
{
    const u64 max_bytes = ctx.options().max_bytes;
    const std::string limit = max_bytes ? " (max_bytes=" + std::to_string(max_bytes) + ")" : std::string();
    const std::string need = bytes ? std::to_string(bytes) + " bytes of device memory; the CUDA context can still allocate " +
                                         std::to_string(room) + " bytes" + limit
                                   : "more device memory than the CUDA context can still allocate (" + std::to_string(room) +
                                         " bytes" + limit + ")";
    throw std::length_error("mymyr: " + search + ": " + what + " has " + std::to_string(successors) + " successors, which need " +
                            need + ". " + advice);
}

/// A window [first, first + rows) of one state's successor rows (lifted::Labels::first) and the rows of the indexed
/// sort scratch that covers every schema segment it touches (Labels::scratch_first, scratch_rows).
struct SuccessorWindow
{
    u64 first = 0, rows = 0, scratch_first = 0, scratch_rows = 0;
};
/// The window from row `first` (below offsets.back()) of at most `rows` rows of a state whose schema segments start
/// at `offsets` [S + 1] (offsets[S]: its successors).
inline SuccessorWindow successor_window(std::span<const u32> offsets, u64 first, u64 rows)
{
    SuccessorWindow w;
    w.first = first;
    w.rows = std::min<u64>(std::max<u64>(rows, 1), offsets.back() - first);
    // the segment of a row r starts at the last offset <= r and ends at the first offset > r (empty segments share it)
    const u64 last = first + w.rows - 1;
    w.scratch_first = *(std::upper_bound(offsets.begin(), offsets.end(), first) - 1);
    w.scratch_rows = *std::upper_bound(offsets.begin(), offsets.end(), last) - w.scratch_first;
    return w;
}
}  // namespace mymyr::cuda
