#pragma once
// The kernels of the multi-search IW driver (cuda/multi_iw.hpp):
// destination-passing launchers over the arrays the host driver owns. B searches run the same IW pass in
// lock step, layer by layer; every key carries the search id (novelty tuples, the root-successor set, per-search
// counters), so the searches never see each other.
//
// The layer being expanded is a list of entries (node indices), grouped by search and, within a search, in pop order
// (admission order, or the rollout's shuffled order). A chunk is a range of entries. Per chunk:
//   1. launch_rows_live     which entries are popped at all (the search runs and its layer was not truncated), their
//                           goal flags, and per search the first live row and the first goal row (atomicMin: the
//                           minimum is independent of timing);
//   2. launch_rows_select   (after an exclusive scan of the live flags) which live rows are expanded: those before
//                           the first goal row and before the max_expanded budget; which of them generate successors
//                           (not skipped, above max_depth);
//   3. launch_gather        (after a scan of the generate flags) the generating rows into a contiguous parent buffer
//                           for the lifted kernels (cuda/generator.hpp, cuda/lifted.hpp), which write the candidates
//                           in canonical order: (search, pop position, schema, binding);
//   4. novelty, exact (the default): the table after candidates c_1..c_{j-1} is the union of their
//      tuples (a pruned candidate adds nothing new), so c_j is novel iff it holds a tuple, unseen before the batch,
//      whose smallest holder is c_j. launch_emit_count counts each candidate's unseen tuples, launch_emit_insert
//      resolves every unseen tuple's owner (the smallest candidate: atomicMin in a (search, tuple) map),
//      launch_novel_flags marks the owners novel, launch_compact sets the bits of the tuples whose owner was kept.
//      Relaxed (opt-in): launch_relaxed sets the bits directly; a candidate is novel iff it flipped one (the kept set
//      then depends on timing);
//   5. launch_root_insert   the root layer of the width-0 and optimized-IW(1) passes: distinct root successors per
//                           search (the smallest candidate of each content) and self loops;
//   6. launch_admit / launch_cut / launch_keep: admission flags; per search, the cut where the tree reaches
//      max_states or the next layer reaches max_next_layer_states (the candidate whose admission reaches the limit,
//      found from a scan of the admission flags: no atomics decide it); the candidates after a cut are dropped. Without
//      a root rule the novelty kernels admit, and without a cut every candidate is kept (Admission);
//   7. launch_compact       the kept admitted candidates become the next layer's nodes, in candidate order (and the
//                           reached atoms, the novelty commit);
//   8. launch_search_update per-search counters, statuses and skipped counts, the search's entries of the next layer;
//                           after a layer's last chunk the layer end (exhausted searches, the rollouts' per-search
//                           shuffle of the next layer with its own SplitMix64 stream, dead duplicate entries of width
//                           0); in a device loop, the step's advance (ChunkEnd).
//
// A chunk is a fixed launch sequence sized by capacities, not by counts the host reads, so that it can be captured
// into a CUDA graph and replayed. Its scalars (first entry, rows, node and entry bases, depth) are a Step in device
// memory; the gathered rows, the candidates and the emitted tuples are device counts (Gathered::n_dev,
// Candidates::n_dev, Emission::total) against capacities the host chose from its estimates. A count over its capacity
// (launch_check, or the kernel that first consumes it) sets a bit of the chunk's abort word, after which every
// committing kernel of the chunk (Candidates::abort, Chunk::abort) does nothing, so the host redoes the chunk with
// larger capacities after its one control read (the control block, k_ctl_*). The layer lists are named by the step
// too (Step::cur, Step::next), and the layer's end is a step flag the end-of-layer work tests, so that one capture of
// a chunk serves every chunk of every layer: the last block of launch_search_update advances the step on the device
// and ends a device loop (a CUDA graph WHILE node over the chunk: no host round trip per chunk) when the pass ends, a
// chunk aborts or the next chunk would not fit the capture's capacities (Loop). The scans read their live
// counts on the device (one launch each, work in the live items: cuda/src/live_scan.cuh), and the chunk's small
// per-candidate and per-search steps are fused into the kernels next to them (Admission, Compact, ChunkEnd). Rows of
// equal content are
// expanded once (launch_dedup_*: the gathered parents of a chunk collapse to their distinct rows, the lifted kernels
// run on those, and launch_broadcast gives every gathered row its parent's candidates in canonical order): rollouts
// from one start state pop the same states in many searches.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20.

#include "mymyr/rl/task_arrays_view.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::miw
{
inline constexpr u32 k_none = 0xFFFFFFFFu;
inline constexpr u32 k_dead = 0xFFFFFFFEu;  // an entry of a layer that is never popped (a width-0 duplicate)
inline constexpr u64 k_empty_key = ~0ull;
/// Largest atom slot a tuple key holds (states are at most lifted::k_max_words = 64 words).
inline constexpr u32 k_key_atom_bits = 12;

/// Search statuses on the device (the host maps them to search::SearchStatus).
enum : u32
{
    k_running = 0,
    k_solved = 1,
    k_out_of_states = 2,
    k_exhausted = 3,
    k_idle = 4,  // not part of this pass
    k_out_of_time = 5,
};

/// Node flags.
enum : u8
{
    k_flag_skip = 1,    // popped and counted as expanded, never expanded (optimized IW(1), RootOnly)
    k_flag_closed = 2,  // popped already in this layer (width-0 duplicate entries)
};

/// The control block of a chunk: device u32s its kernels write, read by the host once per chunk.
enum : u32
{
    k_ctl_abort = 0,     // abort bits (below): the chunk is redone
    k_ctl_distinct = 2,  // distinct gathered rows (dedup)
    k_ctl_ucand = 3,     // candidates of the distinct rows
    k_ctl_cand = 4,      // candidates
    k_ctl_emit = 5,      // unseen tuples (exact novelty)
    k_ctl_adm = 6,       // kept admitted candidates: new nodes
    k_ctl_ent = 7,       // kept entries of the next layer
    k_ctl_err = 8,       // a sorted segment found no scratch (the lifted kernels' error; internal)
    k_ctl_running = 9,   // searches still running after the layer (its last chunk)
    k_ctl_missing = 10,  // lazy slots: atoms without a slot were met
    k_ctl_done = 11,     // search_update's finished blocks (its last block advances a device loop's step)
    k_ctl_gen_rows = 12, // the rows the generator expands (dedup: the distinct rows, 0 past Dedup::capacity)
    k_ctl_words = 16,
};
/// Abort bits (ctl[k_ctl_abort]).
enum : u32
{
    k_abort_ucand = 1,
    k_abort_cand = 2,
    k_abort_emit = 4,
    k_abort_missing = 8,
    k_abort_space = 16,  // the chunk's nodes or entries exceed node_cap or next_cap (compact, relaxed check)
    k_abort_distinct = 32,  // the distinct rows exceed Dedup::capacity (dedup_compact)
};

/// The scalars of a chunk, in device memory (a captured chunk replays with the next chunk's values).
struct Step
{
    u32 begin = 0;       // first entry of the chunk in the layer's list
    u32 rows = 0;        // entries of the chunk (at most Chunk::rows, the capacity)
    u32 node_base = 0;   // nodes before the chunk: the first node compact writes
    u32 entry_base = 0;  // entries of the next layer before the chunk
    u32 layer = 0;       // depth of the layer being expanded
    u32 entries = 0;     // entries of the layer
    u32 last = 0;        // the chunk ends the layer (layer end)
    u32 next_cap = 0;    // capacity of `next`
    u32 node_cap = 0;    // nodes allocated
    u32* cur = nullptr;   // the layer's entries
    u32* next = nullptr;  // the next layer's entries
};

/// A device loop over chunks (ChunkEnd; the counters of its chunks, for the host's stats and estimates).
struct Loop
{
    u32 rows = 0;        // entries per chunk (at most the capture's row capacity)
    u32 grow_above = 0;  // a new layer of more entries stops the loop (a capture of more rows pays)
    u32 stop = 0;        // why it stopped (k_loop_*)
    u32 running = 0;     // searches running after the last layer end
    u32 max_emit = 0;    // the most unseen tuples of a chunk
    u32 emission = 0;    // the chunks run exact novelty (their unseen tuples count)
    u32 max_distinct = 0;  // the most distinct rows of a chunk (dedup)
    double ucand = 0, cand_rows = 0, emit = 0;  // the largest counts per row of a chunk
    double distinct_rows = 0;  // the most distinct rows per row of a chunk of k_ratio_rows rows or more
    unsigned long long chunks = 0, candidates = 0, distinct = 0, nodes = 0, layers = 0;
};
/// The least rows of a chunk whose distinct rows per row count toward the estimate (a small layer's rows are
/// mostly distinct whatever the large ones are).
inline constexpr u32 k_ratio_rows = 1024;

/// Loop::stop.
enum : u32
{
    k_loop_running = 0,
    k_loop_done = 1,   // a layer ended with no search running or no entries
    k_loop_abort = 2,  // a chunk aborted (the step names it: the host redoes it; k_abort_space: with more room)
    k_loop_grow = 3,   // the new layer exceeds grow_above
    k_loop_err = 4,    // the lifted kernels' error (ctl[k_ctl_err])
};

/// Per-row flags of a chunk.
enum : u8
{
    k_row_live = 1,
    k_row_goal = 2,
    k_row_expanded = 4,  // counted as expanded (popped before the goal and the budget)
    k_row_gen = 8,       // generates its successors
    k_row_skip = 16,     // expanded without generating (skip flag or max_depth)
};

/// Per-candidate flags.
enum : u8
{
    k_cand_novel = 1,
    k_cand_adm = 2,    // becomes a node
    k_cand_ent = 4,    // becomes an entry of the next layer (a node, or a width-0 duplicate entry)
    k_cand_skip = 8,   // its node is skipped
    k_cand_kept = 16,  // generated at all (at or before its search's cut)
    k_cand_self = 32,  // a self loop (root rules)
    k_cand_dup = 64,   // equal to an earlier successor of the same root (root rules)
};

/// The nodes of the current pass (all layers; indices are global node indices).
struct Nodes
{
    u64* rows = nullptr;  // [cap, words]
    u32* parent = nullptr;
    u32* search = nullptr;
    u32* label = nullptr;  // [cap, label_words]: schema, then the binding (0xFFFFFFFF past the arity)
    u8* flags = nullptr;
    u32 words = 0;
    u32 label_words = 0;
};

/// Per-search state, indexed by the group-local search id.
struct Searches
{
    u32 count = 0;
    u32* status = nullptr;
    unsigned long long* expanded = nullptr;
    unsigned long long* generated = nullptr;
    unsigned long long* in_tree = nullptr;
    unsigned long long* skipped = nullptr;
    u32* tree = nullptr;       // nodes in the tree (the root included)
    u32* next = nullptr;       // entries of the next layer so far
    u32* layer_cut = nullptr;  // 1: the layer was truncated (the rest of it is not popped)
    u32* goal_node = nullptr;
    u32* goal_depth = nullptr;
    u32* seg_begin = nullptr;  // the search's entries in the next layer so far (empty at a layer's first chunk)
    u32* seg_end = nullptr;
    u64* rng = nullptr;        // SplitMix64 states (randomized layers)
    // per chunk
    u32* first_live = nullptr;  // chunk rows
    u32* last_live = nullptr;
    u32* first_goal = nullptr;
    u32* first_gen = nullptr;   // gathered rows
    u32* last_gen = nullptr;
    u32* cut = nullptr;      // 2 * candidate + (0: max_states, 1: max_next_layer_states) of the cut, or k_none
    u32* cut_row = nullptr;  // chunk row of the cut candidate's parent
    // goals: per search [count, goal_words] positive / negative masks over fluent slots, or null (the task's goal)
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    u32 goal_words = 0;
    u64* reached = nullptr;  // [count, reached_words] or null: fluent atoms of every created state
    u32 reached_words = 0;
};

/// Budgets of the pass (per search).
struct Limits
{
    unsigned long long max_expanded = ~0ull;
    u32 max_states = k_none;  // tree nodes
    u32 max_next = k_none;    // entries of the next layer (max_next_layer_states)
    u32 max_depth = k_none;
};

/// A chunk: entries [step->begin, step->begin + step->rows) of the current layer's list step->cur.
struct Chunk
{
    const Step* step = nullptr;
    u32 rows = 0;                // the capacity: rows at or past step->rows are absent
    u8* row_flags = nullptr;        // [rows]
    u32* live_scan = nullptr;       // [rows + 1] exclusive scan of the live flags; null when a search's rows
                                    // are all live from its first live row to its last (no dead entries: dense)
    u32* gen_scan = nullptr;        // [rows + 1] exclusive scan of the generate flags
    const u8* host_goal = nullptr;  // [rows] goal flags of goals with derived literals (CPU, or ChunkGenerator::goal_flags), or null
    const u32* abort = nullptr;     // nonzero: the chunk is redone (the committing kernels do nothing)
    u32* ctl = nullptr;             // the control block (search_update writes the kept counts)
};

/// The gathered parents of a chunk and their candidates (written by cuda/generator.hpp).
struct Gathered
{
    u64* rows = nullptr;  // [n, words]
    u32* node = nullptr;  // [n] node index
    u32* crow = nullptr;  // [n] chunk row
    u32* search = nullptr;
    u32 n = 0;
    u32 words = 0;
    const u32* n_dev = nullptr;  // the gathered rows as a device count (n is then the capacity)
};

struct Candidates
{
    const u64* rows = nullptr;  // [n, words], or the distinct rows' candidates (src)
    const u32* parent = nullptr;  // gathered row
    const u32* schema = nullptr;
    const u32* binding = nullptr;  // [n, label_width], or the distinct rows' (src)
    // With parent dedup, candidate i is its distinct row's candidate src[i]: its row, schema and binding are
    // rows[src[i]], schema[src[i]] and binding[src[i]] (the broadcast writes the parent and src of each, not copies
    // of them); null: candidate i is at i
    const u32* src = nullptr;
    u32 label_width = 0;
    u64 n = 0;
    const u32* seg_offsets = nullptr;  // the chunk's (gathered row, schema) segment offsets
    u32 num_schemas = 0;
    u8* flags = nullptr;  // [n]
    // The candidates as a device count (n is then the capacity), and the chunk's abort word (nonzero: every
    // kernel over the candidates does nothing)
    const u32* n_dev = nullptr;
    const u32* abort = nullptr;
};

/// The novelty table of the pass: arity 1: [searches, row_words] bits; arity 2: [searches, 64 * row_words, row_words]
/// (a symmetric bit matrix; the diagonal is the singletons).
struct Table
{
    u64* bits = nullptr;
    u32 arity = 1;
    u32 row_words = 0;  // atoms < 64 * row_words
};

/// The owner map of exact novelty: open addressing over (search, tuple) keys.
struct OwnerMap
{
    u64* keys = nullptr;
    u32* vals = nullptr;
    u64 mask = 0;
};

/// Emission bookkeeping: per candidate the offset of its tuples (an exclusive scan of the counts) and the slot of
/// each emitted tuple.
struct Emission
{
    u32* count = nullptr;   // [n + 1]
    u32* offset = nullptr;  // [n + 1]
    u32* slot = nullptr;    // [capacity]
    const u32* total = nullptr;  // the emitted tuples (device: offset[n]); over `capacity`, the chunk aborts
    u64 capacity = 0;
    u32* ctl = nullptr;          // emit_insert writes the total and the abort bit
};

/// The root-successor set of the root rules: open addressing over tag32 | (candidate + 1).
struct RootSet
{
    u64* slots = nullptr;
    u64 mask = 0;
    u32* owner = nullptr;  // [n] the smallest candidate of the same search and content
};

// ------------------------------------------------------------------------------------------------ pass setup

/// Roots: node i = the start row of search search_of[i] (row search_of[i] of `starts`: rows of `start_words` words,
/// `stride` apart); tree = 1, counters zeroed, status running for those searches (the others idle), order[i] = i.
struct Roots
{
    const u64* starts = nullptr;
    u64 stride = 0;
    u32 start_words = 0;
    const u32* search_of = nullptr;  // [n] (ascending)
    u32 n = 0;
};
cudaError_t launch_init_pass(Nodes nodes, Searches s, Roots r, u32* order, cudaStream_t st);

/// The table rows of the searches of the roots (nodes [0, roots)): arity 1, the root's atoms; arity 2, every pair of them.
cudaError_t launch_table_init(Table t, Nodes nodes, u32 roots, cudaStream_t st);

/// Every running search stops with `status` (a time budget).
cudaError_t launch_stop_running(Searches s, u32 status, cudaStream_t st);
/// The tables of `searches` searches re-laid out from `from` to `to` row words (arity 2: 64 * from rows of each search
/// become 64 * to rows; new rows and words are zero).
cudaError_t launch_table_relayout(const u64* src, u64* dst, u32 arity, u32 searches, u32 from, u32 to, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ chunk phases

/// The tile states of a chunk's scans (cuda/src/live_scan.cuh): a buffer of scan_buffer_words(tiles) u64
/// words holds one site of `tiles` tiles per scan of a chunk (k_scan_*). A site's flags must be zero before its scan,
/// which leaves them set: launch_chunk_begin zeroes the flags of every site (the scan_flag_words(tiles) words at the
/// front of the buffer).
enum : u32
{
    k_scan_live = 0,
    k_scan_gen = 1,
    k_scan_rank = 2,
    k_scan_offsets = 3,
    k_scan_emit = 4,
    k_scan_cut = 5,
    k_scan_kept = 6,
    k_scan_sites = 7,
};
struct ScanState
{
    u64* words = nullptr;
    u64 tiles = 0;  // per site
    u32 site = 0;
};
/// Tiles per site for scans of up to n items; the buffer's words; the words of its flags.
[[nodiscard]] u64 scan_tiles(u64 n);
[[nodiscard]] u64 scan_buffer_words(u64 tiles);
[[nodiscard]] u64 scan_flag_words(u64 tiles);

cudaError_t launch_rows_live(const rl::dev::TaskView& task, Nodes nodes, Searches s, Chunk c, cudaStream_t st);
cudaError_t launch_rows_select(Nodes nodes, Searches s, Limits lim, Chunk c, cudaStream_t st);
cudaError_t launch_gather(Nodes nodes, Searches s, Chunk c, Gathered g, cudaStream_t st);

/// Admission rules.
enum : u32
{
    k_rule_normal = 0,        // admitted iff novel
    k_rule_continuation = 1,  // the root of optimized IW(1): every distinct root successor, skipped unless novel
    k_rule_zero_root = 2,     // the root of width 0: every distinct root successor; duplicates are entries
    k_rule_zero_below = 3,    // below the root of width 0: nothing is admitted
};
/// The admission flags (k_cand_adm, k_cand_ent, k_cand_skip) of a chunk's candidates under `rule`; with
/// `keep` (no cut can drop a candidate) also k_cand_kept on every candidate. The novelty kernels apply it themselves
/// when `admit` is set (no root-rule flags come between novelty and admission); launch_admit always does.
struct Admission
{
    u32 rule = k_rule_normal;
    bool root_only = false;  // search::WidthZero::RootOnly: the zero-root rule skips its admitted nodes
    bool admit = false;
    bool keep = false;
};

cudaError_t launch_emit_count(Gathered g, Candidates c, Table t, Emission e, cudaStream_t st);
cudaError_t launch_emit_insert(Gathered g, Candidates c, Table t, Emission e, OwnerMap m, cudaStream_t st);
cudaError_t launch_novel_flags(Candidates c, Emission e, OwnerMap m, Admission a, cudaStream_t st);
/// launch_novel_flags (with a.admit and a.keep) and the kept scan (launch_scan_flag_pair's of the kept
/// admitted and the kept entry flags into kept [c.n + 1]) in one launch: the scan computes the flags it reads.
cudaError_t launch_novel_kept(Candidates c, Emission e, OwnerMap m, Admission a, u64* kept, ScanState st, cudaStream_t s);
/// Relaxed novelty writes the table before compact's check: it checks first that the chunk's live candidates (a
/// bound of its nodes and entries) fit the step's capacities (else k_abort_space into *abort, nothing written).
cudaError_t launch_relaxed(Gathered g, Candidates c, Table t, const Step* step, u32* abort, Admission a, cudaStream_t st);

cudaError_t launch_root_insert(Gathered g, Candidates c, RootSet r, cudaStream_t st);
cudaError_t launch_root_flags(Gathered g, Candidates c, RootSet r, cudaStream_t st);

cudaError_t launch_admit(Candidates c, Admission a, cudaStream_t st);
/// scan: exclusive scan [n + 1] of the admission (low 32 bits) and entry (high 32 bits) flags (launch_scan_flag_pair).
cudaError_t launch_cut(Gathered g, Candidates c, Searches s, Limits lim, const u64* scan, cudaStream_t st);
/// Kept flags (at or before the search's cut) into c.flags.
cudaError_t launch_keep(Gathered g, Candidates c, Searches s, cudaStream_t st);
/// reached |= the atoms of every kept candidate (the chunks do it in launch_compact; this is for other candidates).
cudaError_t launch_reached(Gathered g, Candidates c, Searches s, cudaStream_t st);

/// The kept admitted candidates as nodes [step->node_base + adm_kept[c]) and the kept entries at
/// step->next[step->entry_base + ent_kept[c]] (a width-0 duplicate entry names its original's node). With
/// `reached`, the kept candidates' atoms go into the searches' reached sets; with exact novelty (`emission.slot`), the
/// tuples whose owner was kept go into the table and every owner empties its slots of the map (the map is empty
/// between chunks; a chunk that does not fit, k_abort_space, empties them without committing).
struct Compact
{
    const u64* kept = nullptr;        // adm_kept (low 32 bits), ent_kept (high 32 bits)
    const u32* root_owner = nullptr;  // null outside the root rules
    const Step* step = nullptr;
    u32* abort = nullptr;             // ctl[k_ctl_abort] (k_abort_space)
    u64* reached = nullptr;           // Searches::reached, or null
    u32 reached_words = 0;
    Table table{};
    OwnerMap map{};
    Emission emission{};
};
cudaError_t launch_compact(Nodes nodes, Gathered g, Candidates c, Compact m, cudaStream_t st);

/// The end of a chunk in launch_search_update. After a layer's last chunk (step->last), per running
/// search: exhausted if its new layer (seg_begin, seg_end of step->next) is empty; else, with `shuffle`, its entries
/// are shuffled with its SplitMix64 stream; with `close_duplicates`, entries of a node popped earlier in the list
/// become k_dead; next and layer_cut are reset; ctl[k_ctl_running] counts the searches still running. With `loop`
/// (a device loop), the kernel's last block then records the chunk's counts in it; unless the chunk aborted, the step
/// advances past it (node and entry bases, and at a layer end the next layer: entries, lists swapped, depth) and the
/// next chunk's rows are set; the loop ends (cudaGraphSetConditional(handle, 0), Loop::stop) when the pass ends, the
/// chunk aborted, or the next layer should be captured larger (grow_above).
struct ChunkEnd
{
    Nodes nodes{};
    bool shuffle = false;
    bool close_duplicates = false;
    Step* step = nullptr;  // (the chunk's, writable: the loop advances it)
    Loop* loop = nullptr;
    unsigned long long handle = 0;
};
/// kept: exclusive scan [n + 1] of the kept admitted (low 32 bits) and kept entry (high 32 bits) flags. Per search:
/// the counters, the status, the skipped rows, cut_row, the search's entries of the next layer (seg_begin, seg_end);
/// ctl[k_ctl_adm], ctl[k_ctl_ent] (the chunk's totals); then the chunk's end.
cudaError_t launch_search_update(Gathered g, Candidates c, Searches s, Limits lim, Chunk ch, const u64* kept, ChunkEnd e,
                                 cudaStream_t st);

// ------------------------------------------------------------------------------------------------ scans

/// out[i] = the number of j < i with (flags[j] & mask) == mask, for i <= live, and out[n] = the total, where live =
/// min(*n_dev, n) (null: n), 0 once *abort (may be null) is set: the flags past it do not count, and the outputs
/// between it and n are not written. One launch; its work is in the live items.
cudaError_t launch_scan_flags(const u8* flags, u8 mask, u64 n, const u32* n_dev, const u32* abort, u32* out, ScanState st,
                              cudaStream_t s);
/// out = the exclusive scan of in [live] as launch_scan_flags's.
cudaError_t launch_scan_u32(const u32* in, u64 n, const u32* n_dev, const u32* abort, u32* out, ScanState st, cudaStream_t s);
/// Two flag scans at once: out[i] = lo + (hi << 32), lo / hi the number of j < i with the mask_lo / mask_hi bits
/// set in flags[j], as launch_scan_flags's.
cudaError_t launch_scan_flag_pair(const u8* flags, u8 mask_lo, u8 mask_hi, u64 n, const u32* n_dev, const u32* abort, u64* out,
                                  ScanState st, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ control

/// ctl[slot] = *count; if *count > capacity, ctl[k_ctl_abort] |= bit (the chunk is redone).
struct Check
{
    u32* ctl = nullptr;  // null: no check
    u32 slot = 0;
    const u32* count = nullptr;
    u32 capacity = 0;
    u32 bit = 0;
};
cudaError_t launch_check(Check c, cudaStream_t st);
/// The start of a chunk: the control block zeroed, the per-chunk search temporaries reset (first_live .. cut_row to
/// k_none, last_live and last_gen to 0), the scans' flags zeroed (every site of `scans`) and, at a layer's first chunk
/// (step->begin 0), seg_begin and seg_end zeroed.
cudaError_t launch_chunk_begin(Searches s, u32* ctl, const Step* step, ScanState scans, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ dedup

/// Distinct rows of the gathered parents: an open-addressing set over row indices (`table`, mask + 1 slots, all
/// k_none before; dedup_compact empties the slots it used, so the table stays empty between chunks). The
/// owner of a content is its smallest row (atomicMin), so the distinct rows and their order do not depend on timing.
struct Dedup
{
    u32* table = nullptr;
    u64 mask = 0;
    u32* owner = nullptr;   // [capacity] the smallest row of the same content
    u8* flags = nullptr;    // [capacity] 1: the row owns its content
    u32* rank = nullptr;    // [capacity + 1] exclusive scan of the flags: the distinct row of an owner
    u32* map = nullptr;     // [capacity] the distinct row of every gathered row (an owner's table slot before compact)
    u64* rows = nullptr;    // [rows_capacity, words] the distinct rows, in the order of their owners
    u32* ctl = nullptr;     // dedup_compact writes ctl[k_ctl_distinct] and ctl[k_ctl_gen_rows]
    // The distinct rows' capacity (the generator's rows, sized from the driver's estimate: the rollouts expand a
    // few distinct states per thousands of gathered rows); past it the chunk aborts (k_abort_distinct)
    u64 rows_capacity = 0;
};
cudaError_t launch_dedup_insert(Gathered g, Dedup d, cudaStream_t st);
/// Owners and flags (rows past *g.n_dev: no flag).
cudaError_t launch_dedup_owner(Gathered g, Dedup d, cudaStream_t st);
/// After the scan of the flags: the distinct rows and the map; the table's slots emptied. More distinct rows than
/// d.rows_capacity: none is written, the chunk aborts (k_abort_distinct) and the generator expands none.
cudaError_t launch_dedup_compact(Gathered g, Dedup d, cudaStream_t st);
/// offsets = the exclusive scan [g.n + 1] of the gathered rows' candidate counts (distinct row map[r]'s, from its
/// segment offsets u_seg over num_schemas schemas), as launch_scan_u32's over the live rows *g.n_dev.
cudaError_t launch_dedup_offsets(Gathered g, Dedup d, const u32* u_seg, u32 num_schemas, u32* offsets, ScanState st,
                                 cudaStream_t s);
/// Candidates in canonical order per gathered row: candidate j of row r (row_offsets: the exclusive scan of the counts)
/// is candidate j - row_offsets[r] of its distinct row, copied from the distinct rows' candidates; its parent is r.
/// One warp per gathered row; with `ucand`, the distinct rows' candidates are checked against their
/// capacity first (k_ctl_ucand; past it they were not all written: the chunk aborts); with `emit_count`, each
/// candidate's unseen tuples over `table` are counted as launch_emit_count counts them.
struct Broadcast
{
    const u64* rows = nullptr;      // the distinct rows' candidates [.., g.words]
    const u32* schema = nullptr;
    const u32* binding = nullptr;   // [.., label_width]
    const u32* seg = nullptr;       // the distinct rows' segment offsets (num_schemas per row)
    u32 num_schemas = 0;
    const u32* row_offsets = nullptr;  // [g.n + 1]: the total is the candidates' count
    u32* out_parent = nullptr;      // [capacity]
    u32* out_src = nullptr;         // [capacity]: the distinct rows' candidate (Candidates::src)
    u64 capacity = 0;  // over it, the chunk aborts (ctl[k_ctl_cand] is the total)
    const u32* abort = nullptr;
    u32* ctl = nullptr;
    const u32* ucand = nullptr;  // the distinct rows' candidates (a device count), or null
    u64 ucand_capacity = 0;
    Table table{};
    u32* emit_count = nullptr;  // [capacity], or null
};
cudaError_t launch_broadcast(Gathered g, Dedup d, Broadcast b, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ results

/// Plans of the searches solved in this pass: search i of `which` writes the labels of the path to its goal node at
/// plan_offset[i] (label_words u32 per step) and the goal row into goal_rows[i] (goal_words words).
struct PlanOut
{
    const u32* which = nullptr;
    const u32* plan_offset = nullptr;
    u32 n = 0;
    u32* labels = nullptr;
    u64* goal_rows = nullptr;
    u32 goal_words = 0;
};
cudaError_t launch_plans(Nodes nodes, Searches s, PlanOut p, cudaStream_t st);
/// The non-goal states of the same plans (the ancestors of the goal nodes, the root first) as gathered rows at
/// plan_offset[i] with their search ids (g.rows, g.search; g.words wide): mimir's plan extraction enumerates their
/// successors, which the rollouts' reached atoms include (cuda/rollouts.hpp).
cudaError_t launch_plan_states(Nodes nodes, Searches s, PlanOut p, Gathered g, cudaStream_t st);
}  // namespace mymyr::cuda::miw
