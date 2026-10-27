#pragma once

#include "mymyr/core/types.hpp"
#include "mymyr/cuda/lifted.hpp"
#include "mymyr/cuda/runtime.hpp"

#include <algorithm>
#include <bit>
#include <initializer_list>
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

/// True when `e` is the pool's allocation failure (the error is then cleared).
inline bool out_of_memory(const CudaError& e)
{
    if (e.code() != cudaErrorMemoryAllocation)
        return false;
    (void)cudaGetLastError();
    return true;
}

/// Runs `allocate(attempt)`, which must leave what it changes consistent when an allocation fails. After a failure
/// the pool's unused memory is returned to the driver (free blocks of other sizes need not fit the request) and it
/// runs again with attempt 1 (at exact sizes, say). False when that attempt fails too.
template<class F>
bool try_allocate(DeviceContext& ctx, F&& allocate)
{
    for (int attempt = 0;; ++attempt)
    {
        try
        {
            allocate(attempt);
            return true;
        }
        catch (const CudaError& e)
        {
            if (!out_of_memory(e))
                throw;
            if (attempt == 1)
                return false;
        }
        ctx.trim(0);
    }
}

/// A device buffer a search holds (refusals name what took the context's memory).
struct HeldBytes
{
    const char* what;
    u64 bytes;
};
/// "N bytes are in use: <what> <bytes>, ..., and R of the task's device tables, the generator and the context's other
/// buffers".
inline std::string held_text(const DeviceContext& ctx, std::initializer_list<HeldBytes> held)
{
    const u64 used = ctx.usage().used;
    u64 named = 0;
    std::string s;
    for (const HeldBytes& h : held)
    {
        s += std::string(h.what) + " " + std::to_string(h.bytes) + ", ";
        named += h.bytes;
    }
    return std::to_string(used) + " bytes are in use: " + s + "and " + std::to_string(used > named ? used - named : 0) +
           " of the task's device tables, the generator and the context's other buffers";
}

/// " (max_bytes=N)" when the context has a limit.
inline std::string max_bytes_text(const DeviceContext& ctx)
{
    const u64 max_bytes = ctx.options().max_bytes;
    return max_bytes ? " (max_bytes=" + std::to_string(max_bytes) + ")" : std::string();
}

/// Refuses a search whose device memory (`held`: held_text) leaves the context too little for `need` (what it was
/// about to allocate, a phrase).
[[noreturn]] inline void refuse_memory(const DeviceContext& ctx, const std::string& search, const std::string& need,
                                       const std::string& held, const std::string& advice)
{
    throw std::length_error("mymyr: " + search + ": the CUDA context has too little device memory left for " + need +
                            ": it can still allocate " + std::to_string(device_room(ctx)) + " bytes" + max_bytes_text(ctx) +
                            ", and " + held + ". " + advice);
}

/// Refuses a search after a device allocation it did not size against the context's room failed (`phase`: what it was
/// allocating; `held`: held_text). The message says that an allocation failed, unlike the sized refusals'.
[[noreturn]] inline void refuse_allocation_failure(const DeviceContext& ctx, const std::string& search, const std::string& phase,
                                                   const std::string& held, const std::string& advice)
{
    throw std::length_error("mymyr: " + search + ": a device memory allocation failed during " + phase +
                            ": the CUDA context can still allocate " + std::to_string(device_room(ctx)) + " bytes" +
                            max_bytes_text(ctx) + ", and " + held + ". " + advice);
}

/// Refuses a state (or a chunk of states) whose successors need more device memory than the context has left: the
/// count, the bytes (0: more than `room`, an allocation for them failed) and the options that change it. When `room`
/// is below `row_bytes` (what one successor takes), the search's own buffers took the memory: the refusal names them
/// (`held`: held_text) instead.
[[noreturn]] inline void refuse_successors(const DeviceContext& ctx, const std::string& search, const std::string& what,
                                           u64 successors, u64 bytes, u64 room, const std::string& advice, u64 row_bytes = 0,
                                           const std::string& held = {})
{
    const std::string limit = max_bytes_text(ctx);
    if (!held.empty() && room < row_bytes)
        throw std::length_error("mymyr: " + search + ": the search's own device memory leaves the CUDA context " +
                                std::to_string(room) + " bytes" + limit + ", less than one successor needs (" +
                                std::to_string(row_bytes) + " bytes; " + what + " has " + std::to_string(successors) +
                                " successors); " + held + ". " + advice);
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
