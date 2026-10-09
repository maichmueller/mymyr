#pragma once
// The kernels of the device state spaces (include/mymyr/cuda/state_space.hpp): the state set with the
// instance in the key, the per-chunk bookkeeping of the transitions, and the post-processing of a wave's union graph
// (instance-major order, reverse CSR, goal distances, flags, per-instance local ids). Internal to cuda/src.
//
// The state set is the one of the device BrFS (cuda/state_set.hpp: tag32 | ref32 slots, the smallest candidate of a
// content wins by atomicMin of its pending reference, ids by a scan of the owner flags in candidate order), with two
// changes: a stored state's key is (instance, row), so the states of many instances share one table, and every
// candidate's result is its slot (a duplicate's too), so that after the compaction the slot names the target of every
// transition. Atomics decide slots, never ids.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20; POD, raw pointers and cudaStream_t.

#include "mymyr/core/types.hpp"
#include "mymyr/cuda/cost_program.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::ssk
{
/// No instance (dead or finished instances' rows in a chunk: the multi-instance kernels skip them).
inline constexpr u32 k_no_instance = 0xFFFFFFFFu;

struct Table
{
    u64* slots = nullptr;
    u64 mask = 0;  // slots - 1 (a power of two minus one)
};

/// The stored states: rows [count, words] and the instance of each.
struct Store
{
    const u64* rows = nullptr;
    u32 words = 0;
    const u32* inst = nullptr;
};

/// result[c] = the slot of candidate c's content (instance of parent[c], row c of cand) for c < n.
cudaError_t launch_insert(Table t, Store st, const u64* cand, const u32* parent, u64 n, u32* result, cudaStream_t s);
/// rank[c] for c <= n: the owners (smallest candidates of new contents) among candidates [0, c); rank[n] = new states.
[[nodiscard]] u64 rank_temp_bytes(u64 n);
cudaError_t launch_rank(Table t, const u32* result, u64 n, u32* rank, void* temp, u64 temp_bytes, cudaStream_t s);

/// The new states of a chunk: owner c gets id base + rank[c], its row, its instance (the parent's) and its depth (the
/// parent's + 1); the slot then refers to it.
struct Compact
{
    const u32* result = nullptr;
    const u32* rank = nullptr;
    const u64* cand = nullptr;
    const u32* parent = nullptr;
    u32 words = 0;
    u64 base = 0;
    u64* rows = nullptr;   // the store's rows (all of them: the new ones go to base + rank)
    u32* inst = nullptr;   // [count] per state
    u32* depth = nullptr;  // [count] per state
};
cudaError_t launch_compact(Table t, Compact c, u64 n, cudaStream_t s);
/// counts[instance of parent[c]] += the owners among candidates [0, n) (after launch_rank, before the compaction).
cudaError_t launch_owner_counts(Table t, const u32* result, const u32* parent, const u32* inst, u64 n, u32* counts,
                                cudaStream_t s);
/// After the compaction: ids[c] = the state its slot ids[c] refers to (in place), c < n.
cudaError_t launch_resolve(Table t, u32* ids, u64 n, cudaStream_t s);
/// Inserts stored states [first, count) (distinct contents, not in the table yet).
cudaError_t launch_rehash(Table t, Store st, u64 first, u64 count, cudaStream_t s);

/// out[i] = base + seg[i * S] for i <= rows (the forward offsets of the chunk's parents; the last is the next's).
cudaError_t launch_forward_offsets(const u32* seg, u32 num_schemas, u64 rows, u64 base, u64* out, cudaStream_t s);
/// ids[j] = live[inst[j]] for j < n (live: k_no_instance for instances no longer expanded).
cudaError_t launch_chunk_ids(const u32* inst, const u32* live, u64 n, u32* ids, cudaStream_t s);

/// State-independent action costs (heuristics::ActionCosts::cost) of the transitions of a chunk: the program of
/// (instance inst_base + inst[parent], schema) at the transition's binding (mymyr/cuda/cost_program.hpp), the exact
/// transition cost the CPU generator records (heuristics::ActionCosts::transition).
/// An undefined (NaN) cost sets *error.
cudaError_t launch_costs(const costs::Program& programs, u32 inst_base, const u32* schema, const u32* binding, u32 label_width,
                         const u32* parent, const u32* inst, u64 n, f64* out, u32* error, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ post-processing

/// Exclusive prefix sums of n + 1 values into u64 (in[n] is taken as 0: out[n] is the total).
[[nodiscard]] u64 scan_temp_bytes(u64 n);
cudaError_t launch_scan_u32(const u32* in, u64* out, u64 n, void* temp, u64 temp_bytes, cudaStream_t s);
cudaError_t launch_scan_u64(const u64* in, u64* out, u64 n, void* temp, u64 temp_bytes, cudaStream_t s);

/// Stable radix sort of (key, value) pairs by the low `bits` bits of the keys (CUB, double buffers: the sorted pairs
/// end in keys[sel] / values[sel], sel returned in *selector).
[[nodiscard]] u64 sort_temp_bytes(u64 n);
cudaError_t launch_sort_pairs(u32* keys0, u32* keys1, u32* values0, u32* values1, u64 n, u32 bits, void* temp,
                              u64 temp_bytes, int* selector, cudaStream_t s);

cudaError_t launch_iota(u32* out, u64 n, cudaStream_t s);
/// counts[key[i]] += 1 for i < n.
cudaError_t launch_histogram(const u32* key, u64 n, u32* counts, cudaStream_t s);
/// The same with warp-aggregated atomics, for keys that consecutive elements mostly share (instance ids).
cudaError_t launch_histogram_grouped(const u32* key, u64 n, u32* counts, cudaStream_t s);
/// dst[order[i]] = i.
cudaError_t launch_invert(const u32* order, u64 n, u32* dst, cudaStream_t s);
/// deg[p] = off[perm[p] + 1] - off[perm[p]] for p < n.
cudaError_t launch_degrees(const u64* off, const u32* perm, u64 n, u64* deg, cudaStream_t s);
/// The new position of every edge: for state g with edges [off[g], off[g + 1]): emap[e] = noff[pos[g]] + e - off[g].
cudaError_t launch_edge_map(const u64* off, const u64* noff, const u32* pos, u64 n, u32* emap, cudaStream_t s);
/// dst[emap[e]] = map ? map[src[e]] : src[e] (u32 rows of `width`), or f64 / u8 values.
cudaError_t launch_scatter_u32(const u32* src, const u32* emap, const u32* map, u64 n, u32 width, u32* dst, cudaStream_t s);
cudaError_t launch_scatter_f64(const f64* src, const u32* emap, u64 n, f64* dst, cudaStream_t s);
/// dst[i] = src[perm[i]] (u8).
cudaError_t launch_gather_u8(const u8* src, const u32* perm, u64 n, u8* dst, cudaStream_t s);

/// out[i] = src[P[i]] for i <= k (the first edge of every instance: P [k + 1] into the offsets).
cudaError_t launch_pick_u64(const u64* src, const u64* P, u32 k, u64* out, cudaStream_t s);
/// out[i] = P[i] < P[i + 1] ? dist[P[i]] : -1 for i < k (the unit goal distance of every instance's initial state).
cudaError_t launch_pick_roots(const i32* dist, const u64* P, u32 k, i32* out, cudaStream_t s);

/// bsrc[r] = the source of forward edge edges[r] (binary search over the forward offsets [n + 1]), r < m.
cudaError_t launch_edge_sources(const u64* off, u64 n, const u32* edges, u64 m, u32* bsrc, cudaStream_t s);

/// Unit goal distances by level-synchronous backward BFS: init (dist = 0 at goals, -1 elsewhere; the goals into
/// frontier, *count = their number), then a step per level d = 1, 2, ... from the frontier into next (*next_count).
cudaError_t launch_bfs_init(const u8* goal, u64 n, i32* dist, u32* frontier, u32* count, cudaStream_t s);
cudaError_t launch_bfs_step(const u64* boff, const u32* bsrc, const u32* frontier, u64 nf, i32 d, i32* dist, u32* next,
                            u32* next_count, cudaStream_t s);
/// Cost goal distances: a label-correcting backward search (worklist Bellman-Ford over the reverse CSR; with
/// non-negative costs its fixpoint is Dijkstra's result, see state_space.cpp). dist holds f64 bit patterns (atomicMin
/// on u64 orders non-negative doubles). init: dist = 0 at goals, +inf elsewhere, the goals into frontier. step: relaxes
/// the frontier's incoming edges; an improved state not yet queued goes into next (queued[u] set). clear: queued = 0
/// for the next frontier's states.
cudaError_t launch_sssp_init(const u8* goal, u64 n, u64* dist, u32* frontier, u32* count, cudaStream_t s);
cudaError_t launch_sssp_step(const u64* boff, const u32* bsrc, const u32* bedge, const f64* cost, const u32* frontier,
                             u64 nf, u64* dist, u32* queued, u32* next, u32* next_count, cudaStream_t s);
cudaError_t launch_sssp_clear(const u32* frontier, u64 nf, u32* queued, cudaStream_t s);
/// flags[i] = 1 if some cost of instance i's edges [Q[i], Q[i + 1]) is negative or NaN.
cudaError_t launch_check_costs(const f64* cost, const u64* Q, u32 instances, u64 n, u32* flags, cudaStream_t s);
/// flags[i] = 1 if some cost of instance i's edges [Q[i], Q[i + 1]) differs from 1 (n: the edges, Q [instances + 1]).
cudaError_t launch_unit_costs(const f64* cost, const u64* Q, u32 instances, u64 n, u32* flags, cudaStream_t s);

/// Flags and per-instance statistics: unsolvable = dist < 0, alive = !goal && !unsolvable; cost = dist (as f64, +inf
/// for -1) when unit_cost; stats[inst * 3 + {0, 1, 2}] += goal, += unsolvable, max= dist (i32 as u32 bits, -1 -> 0
/// means none: stored as dist + 1). inst may be null (instance 0).
cudaError_t launch_flags(const u8* goal, const i32* dist, const u32* inst, u64 n, bool unit_cost, u8* unsolvable,
                         u8* alive, f64* cost, u32* stats, cudaStream_t s);
/// maxdepth[inst[i]] = max over the states of their depth (inst may be null: instance 0).
cudaError_t launch_max_depth(const u32* depth, const u32* inst, u64 n, u32* maxdepth, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ localization
/// Per-instance layout of a wave's results, instance-major: instance i has states [P[i], P[i + 1]) and edges
/// [Q[i], Q[i + 1]) (union positions), rows of words[i] words at row_base[i] (u64 words).
struct Layout
{
    const u64* P = nullptr;  // [I + 1]
    const u64* Q = nullptr;  // [I + 1]
    const u32* words = nullptr;
    const u64* row_base = nullptr;
    const u32* inst = nullptr;  // [N] the instance of every union position (null: one instance)
};
/// out rows: instance i's state k = union row perm[P[i] + k] (perm may be null), cut to words[i] words.
cudaError_t launch_local_rows(const u64* rows, u32 W, const u32* perm, Layout l, u64 n, u64* out, cudaStream_t s);
/// Per-instance offsets [n_i + 1] at P[i] + i: off[P[i] + k] - Q[i] (the last entry per instance by launch_local_ends).
cudaError_t launch_local_offsets(const u64* off, Layout l, u64 n, u64* out, cudaStream_t s);
cudaError_t launch_local_ends(const u64* off, Layout l, u32 instances, u64* out, cudaStream_t s);
/// ids[r] -= P[inst[ids[r]]] (state ids), in place.
cudaError_t launch_local_states(u32* ids, Layout l, u64 m, cudaStream_t s);
/// edges[r] -= Q[inst[state[r]]] (edge ids of the edges whose state is state[r]), in place.
cudaError_t launch_local_edges(u32* edges, const u32* state, Layout l, u64 m, cudaStream_t s);
}  // namespace mymyr::cuda::ssk
