#pragma once
// The kernels of the device best-first searches (cuda/astar.hpp, cuda/gbfs.hpp): the
// open list as a bucket queue on the device, popping a batch of it, relaxing the successors of a chunk against the
// search nodes, and pushing the new entries sorted by key. Destination-passing launchers over arrays the host
// driver owns; every count a step produces stays on the device (Ctl), so that a step needs no host read and can be
// captured into a CUDA graph and looped on the device.
//
// Search nodes live in per-id arrays next to the device state set (cuda/state_set.hpp: ids in candidate order).
// The open list is an append-only arena of entries (g << 32 | id) and a list of runs (key, arena begin, count) sorted
// by key and, within a key, by arena position (the push order): the runs of one key are its bucket, first in first out.
// A chunk's entries are sorted stably by key and appended to the arena; its runs are merged into the list after the
// runs of equal keys already there. A pop takes the entries of the lowest key(s) from the head of the list. An entry is
// valid while its node is open with that g (older entries of a node whose g decreased, and entries of closed nodes, are
// skipped).
//
// Determinism: the relaxation picks, per successor state, the candidate with the smallest (g, candidate index)
// (A*) or the smallest candidate index (GBFS: the first generation) by atomicMin over a key that includes the
// candidate index, so the winner never depends on timing; batches and pushed entries are compacted by scans in
// candidate order; the entries are sorted by key with a stable radix sort; the merge places runs by binary search on
// (key, list position). Atomics decide slots, minima and counters only.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"
#include "mymyr/cuda/cost_program.hpp"
#include "mymyr/cuda/state_set.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::bfk
{
inline constexpr u32 k_none = 0xFFFFFFFFu;
inline constexpr u64 k_no_key = ~u64{0};
inline constexpr u32 k_inf = 0xFFFFFFFFu;  // h of a dead end (heuristics_kernels.hpp)

/// Node status (flags & 3) and the goal bit.
enum : u8
{
    k_new = 0,     // generated in the current chunk, not evaluated yet
    k_open = 1,
    k_closed = 2,
    k_dead = 3,    // h = +inf
    k_goal = 4,    // flag bit: the state is a goal state
};

/// Search modes.
enum : u32
{
    k_astar = 0,
    k_gbfs = 1,
};

/// Error bits (Ctl::error).
enum : u32
{
    k_err_cost = 1,      // an action whose cost is undefined
    k_err_overflow = 2,  // g or f does not fit 32 bits
    k_err_scratch = 4,   // a sorted segment found no scratch (lifted::Labels::error; internal)
};

/// Ctl::status: why the search stopped (k_running: it did not).
enum : u32
{
    k_running = 0,
    k_solved = 1,           // A*: a goal entry was popped (the solution is batch_id[0])
    k_goal_found = 2,       // the chunk Ctl::stop_chunk met goal candidate Ctl::goal_c (the host finishes it)
    k_exhausted = 3,        // the open list is empty
    k_out_of_states = 4,    // more states than the budget (after a chunk)
    k_out_of_expanded = 5,  // the expansions reached the budget (before a pop)
};

/// Ctl::abort: the step stopped before committing the chunk Ctl::stop_chunk (or before its expansion: k_abort_pop); the
/// host takes over there.
enum : u32
{
    k_abort_pop = 1,      // the pop window ended before the batch was full and more eligible entries remain
    k_abort_room = 2,     // the chunk's candidates exceed a capacity (scratch, table, nodes, arena, runs)
    k_abort_missing = 4,  // lazy slots: the chunk met atoms without a slot
};

/// The search's control block in device memory: the queue's sizes, the current step and chunk, the statistics. Every
/// kernel below reads its counts here, so a step's launches do not depend on host values that change between steps.
struct Ctl
{
    // the queue and the states (persistent)
    u64 open_size = 0;  // entries in the arena
    u32 runs = 0;       // the run list holds [head, runs)
    u32 head = 0;
    u32 count = 0;      // stored states
    u32 base = 0;       // the current chunk's first new id (count before it)
    // the step
    u32 status = k_running;
    u32 abort = 0;
    u32 stop_chunk = 0;  // the chunk of the abort / of goal_c
    u32 n = 0;           // the batch: entries popped for expansion
    u32 limit = 0;       // at most this many
    u32 goal_pop = 0;    // A*: the step pops a goal key
    u32 round = 0;       // pop rounds of the step (0: the next pop starts a step)
    u32 window = 0;      // the round's window: entries, runs
    u32 wruns = 0;
    u32 more = 0;        // eligible entries past the window
    u32 cut = 0;         // entries the round consumes (k_none: all of its window)
    u32 pruned_round = 0;
    u64 key0 = 0;        // the step's first key (A*: its f layer is key0 >> 32)
    // the chunk
    u32 rows = 0;     // live parents
    u32 M = 0;        // candidates
    u32 live = 0;     // candidates that take part (0 when the chunk commits nothing)
    u32 fresh = 0;    // new states
    u32 heur_n = 0;   // new states to evaluate
    u32 new_runs = 0; // runs of the chunk's sorted entries (the k_no_key run included)
    u32 push_n = 0;   // entries the chunk pushes
    u32 goal_c = k_none;
    u32 dead = 0;     // dead ends among the new states
    u32 reopened_c = 0;
    u32 merge = 0;    // the chunk merges its runs into the list (it ran and the search runs)
    u32 error = 0;    // k_err_* bits
    u32 label_error = 0;         // lifted::Labels::error
    u32 eval_flags[2] = {0, 0};  // the heuristic's: states outside its grounding, rows with unassigned slots
    u32 step_max_M = 0;          // the step's largest chunk
    u32 loop_max_M = 0;          // the largest chunk since the host reset it
    u32 next_M = 0;              // the next step's chunks predicted (launch_step_end): the host sizes the next loop
    // the goal candidate's chunk (launch_goal_info): the goal's id, its parent (batch index), the end of the parent's
    // successors (a candidate index), the new states before that end, the chunk's first new id and new states
    u32 goal_id = 0, goal_parent = 0, goal_end = 0, goal_fresh_before = 0, goal_base = 0, goal_fresh = 0;
    // statistics (as search::BestFirstResult and DeviceBestFirstStats count them)
    u64 expanded = 0, generated = 0, evaluations = 0, dead_ends = 0, pruned = 0, reopened = 0;
    u64 popped = 0, stale = 0, steps = 0, chunks = 0, max_batch = 0, open_entries = 0;
    // a device loop
    u32 steps_left = 0;
    u32 loop_end = 0;  // why the loop ended (k_loop_*)
};
static_assert(sizeof(Ctl) % 8 == 0);

/// Ctl::loop_end.
enum : u32
{
    k_loop_running = 0,
    k_loop_stopped = 1,  // status or abort
    k_loop_steps = 2,    // its steps
    k_loop_room = 3,     // the next step might not fit the capacities
};

/// Search nodes by state id.
struct Nodes
{
    u32* g = nullptr;       // g - g0
    u32* h = nullptr;
    u32* parent = nullptr;  // k_none for the root
    u32* sidx = nullptr;    // index of the state among its parent's successors (canonical order)
    u32* depth = nullptr;
    u8* flags = nullptr;
    u64* best = nullptr;    // relaxation key of the current chunk (k_no_key outside of it)
};

/// The run list (sorted by key, then list position) and its merge buffers.
struct Runs
{
    u64* key = nullptr;
    u64* begin = nullptr;
    u32* count = nullptr;
    u64* key2 = nullptr;  // the merge's output, copied back
    u64* begin2 = nullptr;
    u32* count2 = nullptr;
    u32 capacity = 0;
};

/// A pop round over a window of the eligible entries: A* the entries of the lowest key (single_bucket) or of the
/// lowest f layer's non-goal keys, a goal key's entries alone; GBFS all, in key order.
struct Pop
{
    Runs runs;
    const u64* entries = nullptr;  // the arena
    u32* wprefix = nullptr;        // [window_cap + 1] the window's runs: exclusive prefix of their entries
    u64* wbegin = nullptr;         // [window_cap] their first entry in the arena
    u32 window_cap = 0;
    u32 batch = 0;                 // B
    u64 max_expanded = ~u64{0};
    u32 max_depth = 0xFFFFFFFFu;   // valid entries at this depth or deeper are pruned (not in a goal pop)
    u32 mode = k_astar;
    u32 single_bucket = 1;
};

/// Starts a pop round (one block): a new step (Ctl::round == 0: the status checks, key0, the limit) or the next round
/// of the current one; the window's runs.
cudaError_t launch_pop_plan(Pop p, Ctl* ctl, cudaStream_t s);
/// flags[i] for i <= window_cap: 0 skip (stale, or past the window), 1 expand, 2 prune (depth).
cudaError_t launch_pop_flags(Pop p, Nodes n, const Ctl* ctl, u8* flags, cudaStream_t s);
/// pos (u32, [window_cap + 1]): the exclusive scan of flags == 1; temp: pop_temp_bytes.
[[nodiscard]] u64 pop_temp_bytes(u32 window_cap);
cudaError_t launch_pop_scan(const u8* flags, u32 window_cap, u32* pos, void* temp, u64 temp_bytes, cudaStream_t s);
/// The round consumes the window's entries before the (limit - n)-th expanded one (all of them in a goal pop): closes
/// their valid nodes, appends the expanded ones to the batch at n (batch_id / batch_g / batch_depth).
cudaError_t launch_pop_take(Pop p, Nodes n, Ctl* ctl, const u8* flags, const u32* pos, u32* batch_id, u32* batch_g,
                            u32* batch_depth, cudaStream_t s);
/// Ends the round (one thread): consumes its entries from the run list, n, the statistics; k_abort_pop when the batch
/// is not full and eligible entries remain past the window; a popped goal: k_solved.
cudaError_t launch_pop_finish(Pop p, Ctl* ctl, const u32* pos, cudaStream_t s);

/// out[i] = arena row ids[i] (rows of `words` words), i < min(n, *count).
cudaError_t launch_gather_rows(const u64* arena, u32 words, const u32* ids, u32 n, const u32* count, u64* out, cudaStream_t s);

/// Action costs on the device: the task's cost programs (mymyr/cuda/cost_program.hpp, instance 0) at the
/// candidate's (schema, binding), any number of cost parameters and any static function key space. A cost that is
/// undefined, negative or not below 2^31 sets k_err_cost (the searches take integral costs as u32; state-dependent and
/// non-integral costs are refused).
struct Costs
{
    costs::Program program;
    u32 unit = 1;  // every action costs 1 (`program` is unused)
};

/// The relaxation of one chunk's candidates (after state_set insert / rank / compact). Counts from the Ctl: the live
/// candidates (live), the chunk's first new id (base).
struct Relax
{
    state_set::Table table;
    state_set::Rows arena;        // the stored states (row id at data + id * words)
    const u64* cand = nullptr;    // candidate rows (arena.words wide)
    const u32* result = nullptr;  // launch_insert's result
    const u32* parent = nullptr;  // label parent: batch index (the chunk's parent_base + row)
    const u32* schema = nullptr;  // label schema (costs only)
    const u32* binding = nullptr; // label binding rows (costs only)
    u32 label_width = 0;
    const u32* seg_offsets = nullptr;  // the chunk's segment offsets
    u32 num_schemas = 0;
    u32 parent_base = 0;               // batch index of the chunk's first parent
    const u32* batch_id = nullptr;     // by batch index
    const u32* batch_g = nullptr;
    const u32* batch_depth = nullptr;
    Costs costs;
    u32 mode = k_astar;
    u32 reopen = 1;
    // outputs per candidate
    u32* id = nullptr;
    u32* gc = nullptr;
    u8* part = nullptr;   // takes part in the relaxation
    u8* push = nullptr;   // won it: the node was set from this candidate
};

/// Starts chunk k of the step (one thread): its live parents (Ctl::rows: none once the step stopped), base = count.
cudaError_t launch_chunk_begin(Ctl* ctl, u32 k, u32 chunk_rows, cudaStream_t s);
/// The chunk's limits: a candidate scratch of `capacity`, stored states plus candidates at most `states` (the table's
/// load and the node arrays), arena entries at most `entries`, live runs plus candidates at most `runs`.
struct ChunkLimits
{
    u64 capacity = 0;
    u64 states = 0;
    u64 entries = 0;
    u64 runs = 0;
};
/// After the chunk's count (*total: its candidates; one thread): M, live; k_abort_room past a limit.
cudaError_t launch_chunk_check(Ctl* ctl, const u32* total, u32 k, ChunkLimits limits, cudaStream_t s);
/// After write (lazy slots, one thread): k_abort_missing when *missing is set.
cudaError_t launch_chunk_missing(Ctl* ctl, const u32* missing, u32 k, cudaStream_t s);
/// The new states [base, base + fresh): status k_new, and their rows copied to fresh_rows (for the goal test and the
/// heuristic: addresses that do not depend on the count). Grid over `capacity` new states.
cudaError_t launch_fresh(Nodes n, const Ctl* ctl, state_set::Rows arena, u64* fresh_rows, u64 capacity, cudaStream_t s);

/// Pass 1: ids (new or stored), g of the candidate, and the atomicMin of the key over the candidates that improve their
/// state (new states; A*: a smaller g of an open node, or of a closed one with reopening). Grid over `cands`.
cudaError_t launch_relax(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s);
/// Pass 2: the winners set their node (g, parent, successor index, depth; a stored node becomes open again).
cudaError_t launch_win(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s);
/// Pass 3: best back to k_no_key.
cudaError_t launch_reset(Relax r, Nodes n, const Ctl* ctl, u64 cands, cudaStream_t s);

/// Goal bits of the new states from the goal test's u8 flags (goal[i] for state base + i, i < fresh).
cudaError_t launch_set_goals(Nodes n, const Ctl* ctl, const u8* goal, u64 capacity, cudaStream_t s);
/// Ctl::goal_c = the smallest winning candidate whose state is a goal state: GBFS a new one (before its evaluation),
/// A* one of f at most the step's layer (after the evaluation).
cudaError_t launch_first_goal(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s);
/// After launch_first_goal (one thread): a goal ends the search at chunk k (k_goal_found); with set_heur, heur_n = the
/// new states to evaluate (none after a goal: GBFS's test comes before the evaluation).
cudaError_t launch_goal_gate(Ctl* ctl, u32 k, bool set_heur, cudaStream_t s);
/// After a goal gate (one thread): when chunk k met the goal, the facts the host's stop needs (Ctl::goal_*), read
/// before a later chunk of the step reuses the scratch.
cudaError_t launch_goal_info(Relax r, const u32* rank, Ctl* ctl, u32 k, cudaStream_t s);
/// The new states after their evaluation (h_fresh[i], i < heur_n; null: blind, 0): h, open or dead (Ctl::dead).
cudaError_t launch_settle(Nodes n, Ctl* ctl, const u32* h_fresh, u64 capacity, cudaStream_t s);
/// *out = the dead ends among the states [base + from, base + fresh) (the host's stop at a goal).
cudaError_t launch_count_dead(Nodes n, u64 first, u64 count, u32* out, cudaStream_t s);

/// The entries the chunk pushes: winners whose state is not a dead end, while the search runs. Keys: A* (f << 32) |
/// (goal ? 0 : h + 1) (goals first among equal f, then lower h); GBFS (h << 32) | g; k_no_key for the candidates that
/// push nothing, and past the live ones up to `cands` (they sort last and form the last run).
struct Push
{
    const u32* id = nullptr;
    const u32* gc = nullptr;
    const u8* push = nullptr;
    u32 mode = k_astar;
    u64* keys = nullptr;     // [cands] out
    u64* entries = nullptr;  // [cands] out: g << 32 | id
};
cudaError_t launch_push_keys(Push p, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s);

/// Sorts n (key, entry) pairs by key (stable) into keys_out / entries_out and run-length encodes the keys: runs_key,
/// runs_count and *runs (device u32). sort_temp_bytes(n) is the scratch.
[[nodiscard]] u64 sort_temp_bytes(u64 n);
cudaError_t launch_sort_runs(const u64* keys, const u64* entries, u64 n, u64* keys_out, u64* entries_out, u64* runs_key,
                             u32* runs_count, u32* runs, void* temp, u64 temp_bytes, cudaStream_t s);

/// The chunk's push (when the search still runs): its runs (runs_key / runs_count, *Ctl::new_runs of them, the
/// k_no_key run excluded) get their arena begins (new_begin, one block) and are merged into the run list after the
/// runs of equal keys (two passes over Runs::key2...), the sorted entries are appended to the arena; then the chunk's
/// statistics (one thread). `cands` bounds the entries, `max_new_runs` the runs.
struct Append
{
    Runs runs;
    const u64* runs_key = nullptr;
    const u32* runs_count = nullptr;
    u64* new_begin = nullptr;           // [max_new_runs]
    const u64* sorted_entries = nullptr;
    u64* arena = nullptr;
    u64 cands = 0;
    u64 max_states = ~u64{0};
    u32 mode = k_astar;
};
cudaError_t launch_append(Append a, Ctl* ctl, cudaStream_t s);
/// The chunk's end (after launch_append): its statistics, the run list's new size, the state budget.
cudaError_t launch_chunk_end(Append a, Ctl* ctl, u32 k, cudaStream_t s);

/// The step's end (one thread): its statistics and the next step's start. In a device loop (handle != 0) also its
/// condition: the loop runs on while the search runs and nothing aborted, it has steps left, and the next step fits
/// `next` (a step's candidates at most next.capacity in all of its chunks). The next step's chunks are predicted
/// (Ctl::next_M) from this step's candidates per parent and the next batch: the head bucket of the run list (the
/// single-bucket A* pop; at most min(p.batch, open entries) otherwise) in chunks of `chunk` parents.
cudaError_t launch_step_end(Ctl* ctl, Pop p, ChunkLimits next, u32 chunks, u32 chunk, unsigned long long handle, cudaStream_t s);

/// out[i] = v for i < n.
cudaError_t launch_fill_u64(u64* out, u64 n, u64 v, cudaStream_t s);
}  // namespace mymyr::cuda::bfk
