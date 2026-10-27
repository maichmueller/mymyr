#pragma once

#include "mymyr/core/types.hpp"

namespace mymyr::rl
{
/// Zero denotes an empty slot. The two hashes 0 and this reserved key share one entry.
inline constexpr u64 k_visited_zero = 0x9E3779B97F4A7C15ULL;

[[nodiscard]] MYMYR_HD constexpr u64 visited_key(u64 hash) { return hash ? hash : k_visited_zero; }

/// H must be a power of two in [2, 2^20]. A row has one writer; lookups must not overlap its writes.
[[nodiscard]] MYMYR_HD bool visited_contains(const u64* table, u32 H, i32 count, u64 hash)
{
    if (!count)
        return false;
    const u64 key = visited_key(hash);
    u32 at = static_cast<u32>(key) & (H - 1);
    for (u32 probe = 0; probe < H; ++probe, at = (at + 1) & (H - 1))
    {
        if (table[at] == key)
            return true;
        if (!table[at])
            return false;
    }
    return false;
}

/// Returns whether the hash was already present. New entries above floor(3H/4) are skipped and count saturates
/// at H to report overflow. Stored entries remain searchable; overflow introduces false negatives only.
MYMYR_HD bool visited_insert(u64* table, u32 H, i32& count, u64 hash)
{
    const u64 key = visited_key(hash);
    u32 at = static_cast<u32>(key) & (H - 1);
    for (u32 probe = 0; probe < H; ++probe, at = (at + 1) & (H - 1))
    {
        if (table[at] == key)
            return true;
        if (!table[at])
        {
            if (count >= static_cast<i32>((u64{H} * 3) / 4))
                count = static_cast<i32>(H);
            else
            {
                table[at] = key;
                ++count;
            }
            return false;
        }
    }
    count = static_cast<i32>(H);
    return false;
}

/// Clears a row in O(H); the caller inserts its current or reset state afterwards.
MYMYR_HD void visited_clear(u64* table, u32 H, i32& count)
{
    for (u32 i = 0; i < H; ++i)
        table[i] = 0;
    count = 0;
}

/// One lookup per successor. parent contains env indices, or indices into rows when supplied; -1 is padding.
/// count has one entry per table row; every nonnegative parent and mapped row must be in range.
inline void visited_mask(const u64* table, const i32* count, u32 H, const u64* succ_hash, const i32* parent,
                         const i32* rows, u64 size, u8* out)
{
    for (u64 j = 0; j < size; ++j)
    {
        const i32 p = parent[j];
        const i32 row = p < 0 ? -1 : rows ? rows[p] : p;
        out[j] = row >= 0 && visited_contains(table + u64{static_cast<u32>(row)} * H, H, count[row], succ_hash[j]);
    }
}
}  // namespace mymyr::rl
