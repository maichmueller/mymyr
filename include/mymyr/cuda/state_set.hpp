#pragma once
// The device state set of the layer BrFS: open addressing over u64 slots tag32 | ref32 with compare-through-arena,
// sized ahead (pre-sized from an estimate, rehashed on growth only), and deterministic ids:
//   1. launch_insert: every candidate claims a slot for its content. A slot first holds a pending reference into the
//      candidate batch (PENDING | (candidate + 1)); among equal candidates, atomicMin keeps the smallest candidate
//      index, and a candidate equal to a stored state is a duplicate. result[c] = its slot, or k_dup;
//   2. launch_rank: an exclusive scan of the owner flags (candidate c owns its content iff the slot still refers to c,
//      i.e. c is the smallest candidate of a new content) gives the new ids base + rank in candidate order;
//   3. launch_compact writes the new states, their node records (parent, index among the parent's successors) and the
//      arena references.
// Since the candidates of a chunk come in canonical order (lifted.hpp), the ids equal those of the CPU BrFS with
// deterministic ids: a state's id is the rank of its first discovery by (parent id, successor index). Nothing
// here depends on the launch configuration or on timing: atomics decide slots, never ids.
// Device-sized chunks. A chunk's launches may be sized for a capacity n while its live candidates are the first
// *live of them (a device count the host does not read: Live), and the first new id may be a device count (Compact::
// offset) that launch_advance moves past the chunk's new states, so that several chunks run per host read.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::state_set
{
inline constexpr u32 k_pending = 0x80000000u;  // the reference points into the candidate batch
inline constexpr u32 k_dup = 0xFFFFFFFFu;      // result of a candidate equal to a stored state
/// Largest state count (references are 31-bit, one value is reserved).
inline constexpr u64 k_max_states = (u64{1} << 31) - 2;

struct Table
{
    u64* slots = nullptr;
    u64 mask = 0;  // slots - 1 (a power of two minus one)
};

/// Rows of `words` words, contiguous.
struct Rows
{
    const u64* data = nullptr;
    u32 words = 0;
};

/// The live candidates of a chunk sized for n: the first min(n, *live) (live null: all n). Candidates past them are
/// absent: not inserted, never owners (their rows and results may be stale).
struct Live
{
    const u32* live = nullptr;
};

/// result[c] for the live candidates c of [0, n) (rows of cand, same width as the arena rows).
cudaError_t launch_insert(Table t, Rows arena, Rows cand, u64 n, u32* result, cudaStream_t s, Live live = {});

/// Resolves existing rows to arena ids after compact (k_dup when absent).
cudaError_t launch_lookup(Table t, Rows arena, Rows rows, u64 n, u32* ids, cudaStream_t s);

/// rank[c] for the live candidates c, and rank[live] = rank[n] = the number of new states: the owners among candidates
/// [0, c) (candidate c is an owner iff rank[c + 1] > rank[c]). rank_temp_bytes(n) is its scratch. The scan
/// reads the live count on the device (three launches sized for n whose tiles past the live candidates exit at once:
/// owner bits and tile counts, the tiles' offsets, the ranks), so a chunk sized for a capacity scans what it holds.
[[nodiscard]] u64 rank_temp_bytes(u64 n);
cudaError_t launch_rank(Table t, const u32* result, u64 n, u32* rank, void* temp, u64 temp_bytes, cudaStream_t s,
                        Live live = {});

/// New states: owner c gets id base + o + rank[c] with o = *offset (0 when offset is null); its row goes to
/// arena_tail[o + rank[c]], its node record (parent[c] + p, c - seg_offsets[(parent[c] - parent_base) * num_schemas])
/// with p = *parent_offset (0 when null) to nodes_tail[o + rank[c]] (two u32), and the slot refers to it. (With offset,
/// arena_tail and nodes_tail are the arrays' first rows and *offset the states stored so far; with parent_offset, the
/// labels' parents are chunk rows and *parent_offset the id of the chunk's first parent.)
struct Compact
{
    const u32* result = nullptr;
    const u32* rank = nullptr;          // [n + 1] (launch_rank)
    const u64* cand = nullptr;
    const u32* parent = nullptr;        // per candidate: the parent's id
    const u32* seg_offsets = nullptr;   // the chunk's segment offsets; the parent's first row is at [local * S]
    u32 num_schemas = 0;
    u32 parent_base = 0;                // id of the chunk's first parent
    u64* arena_tail = nullptr;
    u32* nodes_tail = nullptr;          // [new, 2] (null: no node records)
    u64 base = 0;                       // id of the first new state (plus *offset)
    u32 words = 0;
    const u32* offset = nullptr;        // a device count of stored states (null: 0)
    const u32* parent_offset = nullptr; // a device id of the chunk's first parent (null: 0)
};
cudaError_t launch_compact(Table t, Compact c, u64 n, cudaStream_t s, Live live = {});

/// After a chunk's compact: *count += rank[n] (its new states; rank has n + 1 entries), and *fresh = rank[n] when
/// fresh is not null.
cudaError_t launch_advance(const u32* rank, u64 n, u32* count, u32* fresh, cudaStream_t s);

/// The control of chunks that run several per host read (a group; the device BrFS): the group's state and one
/// record per chunk, which the host reads after the group. A chunk commits nothing once the group halts: it halts when
/// the states reached the budget before a chunk, when a chunk's candidates exceed its capacity or could take the table
/// past its limit (the host redoes it with room), or at a chunk whose parents hold a goal state when the search stops
/// at goals (the host redoes the parents before the first one).
inline constexpr u32 k_halt_budget = 1, k_halt_room = 2, k_halt_goal = 4;
struct ChunkCtl
{
    u32 count = 0;  // states stored (Compact::offset, launch_advance)
    u32 halt = 0;   // k_halt_* bits
    u32 live = 0;   // the current chunk's live candidates (Live): its candidates, 0 when it commits nothing
    u32 pad = 0;
};
/// One per chunk, uploaded with the control block before the group as ChunkRecord{} (first = k_no_goal).
inline constexpr u32 k_no_goal = 0xFFFFFFFFu;
struct ChunkRecord
{
    u32 goals = 0;          // goal states among the parents (with first: lifted::launch_goal_count's out)
    u32 first = k_no_goal;  // row of the first goal state
    u32 candidates = 0;     // the chunk's successors (its count's total; 0 when the group halted before it)
    u32 fresh = 0;          // new states (launch_advance)
    u32 halt = 0;           // why the chunk committed nothing (k_halt_* bits; 0: it committed)
    u32 error = 0;          // lifted::Labels::error
    u32 pad[2] = {0, 0};
};
/// A chunk's limits: its candidates' capacity (the rows' scratch), stored states plus candidates the table takes, the
/// states at which the search stops, and whether it stops at a goal state.
struct ChunkLimits
{
    u64 capacity = 0;
    u64 table = 0;
    u64 budget = 0;
    bool stop_at_goal = false;
};
/// After the chunk's count (*total: its candidates), before its rows: rec->candidates = *total; halts the group when
/// the states reached the budget before the chunk (k_halt_budget), when its candidates exceed the capacity or the
/// stored states plus them the table limit (k_halt_room), or when stop_at_goal and rec->goals > 0 (k_halt_goal);
/// ctl->live = the candidates, or 0 when the chunk commits nothing (rec->halt: the group's halt bits).
cudaError_t launch_chunk_size(ChunkCtl* ctl, ChunkRecord* rec, const u32* total, const ChunkLimits& limits, cudaStream_t s);

/// The device BrFS's loop over chunks and layers (a WHILE graph over a captured chunk): the cursor (the layer
/// [lb, le) and the chunk's parents [b, b + ns)), the chunk's control block and record, and the loop's counts, which
/// the host reads once per loop. A chunk that halts (ChunkCtl::halt) commits nothing and ends the loop at its cursor:
/// the host redoes it as it redoes a halted chunk of a group.
inline constexpr u32 k_loop_running = 0, k_loop_halt = 1, k_loop_done = 2, k_loop_budget = 3, k_loop_room = 4, k_loop_steps = 5,
                     k_loop_grow = 6;
struct LoopCtl
{
    ChunkCtl chunk;
    ChunkRecord rec;
    u32 lb = 0, le = 0, b = 0, ns = 0;
    u32 layers = 0;      // layers the loop started
    u32 chunks = 0;      // chunks it committed
    u32 steps_left = 0;  // chunks it may still run
    u32 end = 0;         // why it ended (k_loop_*)
    u32 max_ratio = 0;   // the most candidates per parent of its chunks, x 256
    u32 started = 0;     // the layer [lb, le) was counted (the states were below the budget at its start)
    u32 cuts = 0;        // chunks launch_loop_size cut
    u64 expanded = 0, generated = 0, goal_states = 0;
    u64 first_goal = ~u64{0};  // the id of the first goal state among its committed chunks' parents
};
/// The loop's bounds (capture constants): parents per chunk, a chunk's candidates (its scratch), stored states plus a
/// chunk's candidates the table and the arenas take, the state budget, whether it stops at goals, and the largest
/// layer it starts (a larger one ends it: k_loop_grow).
struct LoopLimits
{
    u32 chunk = 0;
    u64 capacity = 0;
    u64 table = 0;
    u64 arena = 0;
    u64 budget = 0;
    bool stop_at_goal = false;
    u64 max_layer = 0;
};
/// The chunk's start (one thread): the next layer when the last ended (the loop ends when it is empty: k_loop_done, or
/// when the states reached the budget: k_loop_budget), the chunk's parents and a fresh record.
cudaError_t launch_loop_plan(LoopCtl* ctl, LoopLimits lim, cudaStream_t s);
/// The chunk's parent rows [b, b + ns) of the arena into out (rows of the arena's width, at most `chunk` of them).
cudaError_t launch_gather_parents(Rows arena, LoopCtl* ctl, u64* out, u32 chunk, cudaStream_t s);
/// launch_chunk_size in a loop, after the chunk's count and goal count (one thread): the chunk keeps its first
/// parents whose candidates fit limits.capacity (ns shrinks; the rest are the next chunk's), so the capacity need not
/// bound the candidates of a chunk. A chunk whose first parent alone exceeds the capacity, or one with goal states
/// among its parents (their count is the whole chunk's), is not cut: it halts for room, and the host redoes it.
/// offsets: the exclusive scan of the chunk's segment counts, S segments per parent.
cudaError_t launch_loop_size(LoopCtl* ctl, const u32* offsets, u32 S, const ChunkLimits& limits, cudaStream_t s);
/// The chunk's end (one thread): its counts and the cursor, then the loop's condition (handle: the WHILE node's; 0
/// outside a loop).
cudaError_t launch_loop_next(LoopCtl* ctl, LoopLimits lim, unsigned long long handle, cudaStream_t s);

/// Inserts stored states [0, count) into an empty table (growth, width change).
cudaError_t launch_rehash(Table t, Rows arena, u64 count, cudaStream_t s);
/// dst[i] = src[i] widened (zero words) or cut to dst_words, rows [0, rows).
cudaError_t launch_relayout(const u64* src, u32 src_words, u64* dst, u32 dst_words, u64 rows, cudaStream_t s);

/// The hash of a state row (shared by insert and rehash; the upper 32 bits are the slot tag).
[[nodiscard]] MYMYR_HD u64 row_hash(const u64* w, u32 n)
{
    u64 h = 0x9E3779B97F4A7C15ull ^ n;
    for (u32 i = 0; i < n; ++i)
    {
        h ^= w[i];
        h *= 0xBF58476D1CE4E5B9ull;
        h ^= h >> 31;
    }
    h *= 0x94D049BB133111EBull;
    h ^= h >> 29;
    return h;
}
}  // namespace mymyr::cuda::state_set
