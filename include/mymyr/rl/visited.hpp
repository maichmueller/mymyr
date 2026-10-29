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

/// The targets of an encoded batch (entity indices of type I, int32 or int64) and the expansion rows that its action
/// entities stand for, for target_visited. The batch has `graphs` graphs (a capacity batch's entity offsets past its
/// live graphs are zero: see live_graphs); the rows of graph g are [row_offsets[g], row_offsets[g + 1]) (row_offsets
/// has at least graphs + 1 entries, nondecreasing; rows past its last entry are padding). An action entity's entity_ref is its successor's position in its state's canonical successor order: the
/// row offsets[g] + ref, or with row_index (the rows' positions, ascending per graph: some rows dropped) the row of
/// that position.
template<class I>
struct TargetRows
{
    const I* target = nullptr;   // target_entity [targets]
    const i64* live = nullptr;   // the number of live targets (null: all of them)
    u64 targets = 0;
    const u8* role = nullptr;    // entity_role [entities]
    const I* ref = nullptr;      // entity_ref [entities]
    u64 entities = 0;
    u8 action_role = 0;          // the role of an action entity
    const I* entity_offsets = nullptr;  // [graphs + 1]
    u64 graphs = 0;
    const i32* row_offsets = nullptr;
    const i32* row_index = nullptr;  // [rows] or null
    const u64* succ_hash = nullptr;  // [rows]
    u64 rows = 0;
    const i32* status = nullptr;     // [1] or null: nonzero when the expansion overflowed (no row is usable)
    const i32* env_rows = nullptr;   // [graphs]: graph g's table row, -1 none (null: row g)
};

/// The live graphs of a batch: those before the first decrease of its entity offsets (the offsets past a capacity
/// batch's live graphs are zero), all of them without one.
template<class I>
[[nodiscard]] MYMYR_HD u64 live_graphs(const I* entity_offsets, u64 graphs)
{
    for (u64 g = 1; g <= graphs; ++g)
        if (entity_offsets[g] < entity_offsets[g - 1])
            return g - 1;
    return graphs;
}

/// Whether target t is an action entity whose successor is in its graph's visited table (table [table_rows, H],
/// count [table_rows]); `graphs`: live_graphs of the batch. False for a target past the live ones, a target that is
/// not an action entity, an action whose row does not exist, and every target of an overflowed expansion. Reads only
/// in-range entries: every index read from the arrays is checked against their extents.
template<class I>
[[nodiscard]] MYMYR_HD bool target_visited(const TargetRows<I>& x, const u64* table, const i32* count, u64 table_rows,
                                           u32 H, u64 graphs, u64 t)
{
    if ((x.live && static_cast<i64>(t) >= *x.live) || (x.status && *x.status) || graphs == 0 || graphs > x.graphs)
        return false;
    const i64 e = static_cast<i64>(x.target[t]);
    if (e < 0 || static_cast<u64>(e) >= x.entities || x.role[e] != x.action_role)
        return false;
    // the graph of entity e: the last g with entity_offsets[g] <= e
    u64 lo = 0, hi = graphs - 1;
    while (lo < hi)
    {
        const u64 mid = lo + (hi - lo + 1) / 2;
        if (static_cast<i64>(x.entity_offsets[mid]) <= e)
            lo = mid;
        else
            hi = mid - 1;
    }
    const u64 g = lo;
    if (static_cast<i64>(x.entity_offsets[g + 1]) <= e)
        return false;
    const i64 pos = static_cast<i64>(x.ref[e]);
    const i64 begin = x.row_offsets[g], end = x.row_offsets[g + 1];
    if (pos < 0 || begin < 0 || end < begin || static_cast<u64>(end) > x.rows)
        return false;
    i64 at = -1;
    if (!x.row_index)
        at = begin + pos < end ? begin + pos : -1;
    else
    {
        i64 a = begin, b = end;  // the first row of [a, b) with index >= pos
        while (a < b)
        {
            const i64 mid = a + (b - a) / 2;
            if (x.row_index[mid] < pos)
                a = mid + 1;
            else
                b = mid;
        }
        at = a < end && x.row_index[a] == pos ? a : -1;
    }
    if (at < 0)
        return false;
    const i64 row = x.env_rows ? x.env_rows[g] : static_cast<i64>(g);
    if (row < 0 || static_cast<u64>(row) >= table_rows)
        return false;
    return visited_contains(table + static_cast<u64>(row) * H, H, count[row], x.succ_hash[at]);
}

/// target_visited of every target into out [targets].
template<class I>
void target_visited(const TargetRows<I>& x, const u64* table, const i32* count, u64 table_rows, u32 H, u8* out)
{
    const u64 graphs = live_graphs(x.entity_offsets, x.graphs);
    for (u64 t = 0; t < x.targets; ++t)
        out[t] = target_visited(x, table, count, table_rows, H, graphs, t) ? 1 : 0;
}
}  // namespace mymyr::rl
