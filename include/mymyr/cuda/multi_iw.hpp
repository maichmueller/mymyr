#pragma once
// Many IW searches at once on the device:
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::MultiIwOptions o;               // max_arity = 1: optimized IW(1)
//   std::vector<mymyr::search::IwResult> r = mymyr::cuda::multi_iw(ctx, task, starts, o);
//
// B searches, each with its own start state (and optionally its own goal), share one uploaded task and run the IW
// ladder of search::iw() in lock step: every search expands its layer d in the same launches, the search id is part
// of every key (novelty tuples, the root-successor set, per-search counters and budgets), and each search owns its
// novelty table (arity 1: one bit per fluent slot; arity 2: a symmetric bit matrix of the slots, "where the table
// fits": groups of searches are sized to MultiIwOptions::table_bytes and run one after the other).
//
// Semantics: per search, exactly search::iw(task, {start, max_arity, optimize_iw1, width_zero, witness_pruning,
// canonical_order, budgets}) (mimir's conventions, search/iw.hpp): the width-0 pass, optimized IW(1), goal test on
// pop, expanded / generated / generated_in_tree / skipped per pass, the plan and the goal state. The candidates of a
// chunk come in canonical order (search, pop position, schema, binding; the lifted kernels, with their per-schema
// CPU fallback and CPU axioms), and exact batch novelty (the default) resolves every unseen tuple's
// owner as its smallest candidate, so the tree of every search equals the sequential one for every number of searches,
// group size and chunk size. The budgets are exact too: max_expanded and a goal stop at their row, max_states (tree
// nodes, per pass) at the admission that reaches it. MultiIwOptions::exact = false is relaxed batch novelty (opt-in:
// one atomicOr pass; IW(k)'s guarantees hold but the kept set depends on timing).
//
// Rollouts (cuda/rollouts.hpp; search::find_rollouts_parallel): with per-search seeds, every next layer is shuffled by
// the search's own SplitMix64 stream (cuda/device_rng.hpp) before it is popped, and max_next_layer_states truncates the
// next layer, as the CPU's randomized layer ordering does.
//
// Refused with std::invalid_argument ("mymyr: ..."): numeric tasks (ChunkGenerator::unsupported) and arities above 2.
// Not offered: custom goals, blocked states, observers and successor-order hooks. Goals are the task's or, per search,
// one conjunction of fluent atoms (search::GoalSpec::AtomGoal: the CPU's AnyOf with a single goal).
// The driver is host-driven and synchronous (run() returns when the searches are done). A chunk is one launch
// sequence sized by capacities the driver estimates from earlier chunks; the kernels count on the device, an overflow
// marks the chunk aborted (its committing kernels do nothing) and the driver redoes it at the reported sizes; the host
// reads a few bytes of control per chunk, once (the last chunk of a layer ends the layer in the same sequence). Parents
// of equal content in a chunk are expanded once (MultiIwOptions::dedup_parents): rollouts from one start state pop the
// same states in many searches.

#include "mymyr/task/atom_index.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr::cuda
{
struct MultiIwOptions
{
    u32 max_arity = 1;  // the ladder 0..max_arity (the device runs arities 0, 1 and 2)
    bool optimize_iw1 = true;
    search::WidthZero width_zero = search::WidthZero::ExpandDepthOne;
    bool witness_pruning = false;  // search::iw()'s default: every applicable action is a transition
    bool canonical_order = true;
    bool exact = true;             // exact batch novelty; false: relaxed (nondeterministic kept sets)
    search::Budget budget;         // per search and pass as in search::iw(); max_seconds spans the whole call
    u32 max_next_layer_states = ~u32{0};  // truncated next layers (with seeds, as the CPU's ordered layers)
    bool track_reached = false;    // the fluent atoms of every state each search created (the rollouts' report)
    bool plans = true;             // plans and goal states of the solved searches
    bool costs = true;             // plan costs (heuristics::plan_metric; may rewrite actions, as search::iw())
    // launch configuration: in exact mode no result depends on it
    u32 max_searches = 0;          // searches per group (0: as many as fit table_bytes)
    u64 table_bytes = u64{1} << 30;  // novelty tables of a group
    u32 chunk_states = u32{1} << 20;  // layer entries per chunk at most
    u64 view_bytes = 0;            // per-chunk budget of the lifted views (0: half the device's L2 cache)
    u64 emit_budget = u64{1} << 24;  // unseen tuples per chunk (exact mode); larger chunks are split
    bool dedup_parents = true;     // parents of equal content in a chunk are expanded once
    double capacity_per_row = 8;   // first capacity estimates per chunk row (candidates, unseen tuples; grown from
                                   // the counts: a chunk over them is redone)
    u64 chunk_bytes = u64{1} << 28;  // a chunk's candidates and unseen tuples at the estimated capacities; chunks
                                     // are split below it
    bool graphs = true;            // a chunk shape met again replays as a captured CUDA graph (not with CPU work
                                   // inside a chunk: the CPU fallback, host axioms, goals over derived atoms); the
                                   // chunks of a layer chain in a device loop. The environment overrides it for
                                   // compute-sanitizer: MYMYR_CUDA_GRAPHS=0 (no graphs), MYMYR_CUDA_DEVICE_LOOPS=0 (no
                                   // device loops; GraphExec::enabled, loops_enabled)
};

struct MultiIwStats
{
    double seconds = 0;  // wall time of the call
    double host_ms = 0;  // CPU work: fallback schemas, axioms, lazy interning, uploads, host goal tests
    u64 groups = 0;
    u64 passes = 0;      // group passes
    u64 layers = 0;
    u64 chunks = 0;
    u64 splits = 0;      // chunks redone smaller (emit_budget)
    u64 redone = 0;      // chunks redone at larger capacities, or after atoms were interned (lazy slots)
    u64 distinct = 0;    // parents expanded (distinct rows per chunk with dedup_parents; 0 without)
    u64 replays = 0;     // chunks replayed from captured CUDA graphs
    u32 captures = 0;    // chunk shapes captured
    u64 device_loops = 0;  // device loops run (graphs of one WHILE node over a chunk: their chunks replay too)
    u64 loop_handoffs = 0;  // one-row chunks over emit_budget that stopped a device loop (the host ran them)
    u64 nodes = 0;       // tree nodes of all searches and passes
    u64 candidates = 0;  // generated transitions
    u32 uploads = 0;     // device task uploads
    u32 widenings = 0;   // state width growths (lazy slots)
    u32 host_schemas = 0;
    u64 device_bytes = 0;  // high-water mark of the context's pool
};

/// Flat per-search results (host arrays; index = search).
struct MultiIwBatch
{
    u32 n = 0;
    u32 words = 0;         // goal rows
    u32 label_width = 0;   // plan labels hold 1 + label_width u32: schema, binding (0xFFFFFFFF past the arity)
    std::vector<search::SearchStatus> status;
    std::vector<u32> effective_width;
    std::vector<i32> plan_length;   // -1 unless solved
    std::vector<u64> plan_offsets;  // [n + 1] steps
    std::vector<u32> plan_labels;   // [steps, 1 + label_width]
    std::vector<u64> goal_rows;     // [n, words] (zero unless solved)
    u32 pass_slots = 0;                               // passes per search at most (the stride of pass_stats)
    std::vector<search::IwPassStatistics> pass_stats;  // [n, pass_slots]: search i's passes first
    std::vector<u8> num_passes;                        // [n]
    u32 reached_words = 0;
    std::vector<u64> reached;  // [n, reached_words] fluent slots (track_reached)
    std::string message;       // why the searches could not run (status Failed), if so
    MultiIwStats stats;

    /// The IW passes of search i (as search::IwResult::passes).
    [[nodiscard]] std::span<const search::IwPassStatistics> passes(u32 i) const;
    [[nodiscard]] u64 expanded(u32 i) const;
    [[nodiscard]] u64 generated(u32 i) const;
    /// The plan of search i as actions.
    [[nodiscard]] std::vector<Action> plan(u32 i, const Task& task) const;
    /// search::iw()'s result for search i (plan cost when `costs`; start: the search's start state).
    [[nodiscard]] search::IwResult result(u32 i, const Task& task, const State& start, bool costs) const;
};

/// Start states on the device: rows of `words` words, `stride` words apart (stride 0: every search starts from the one
/// row, as rollouts do).
struct DeviceStarts
{
    const u64* data = nullptr;
    u64 stride = 0;
    u32 words = 0;
    u32 rows = 0;
};

class DeviceMultiIw
{
public:
    /// Throws std::invalid_argument for tasks or options the device does not run (see above).
    DeviceMultiIw(ContextPtr ctx, TaskPtr task, const MultiIwOptions& options = {});
    ~DeviceMultiIw();
    DeviceMultiIw(const DeviceMultiIw&) = delete;
    DeviceMultiIw& operator=(const DeviceMultiIw&) = delete;

    /// One search per start row (read on `stream`, null: the context's stream; the call waits for the work enqueued
    /// on it). goals: empty (the task's goal) or one per search. seeds: empty (layers in order) or one per search.
    MultiIwBatch run(DeviceStarts starts, std::span<const search::GoalSpec::AtomGoal> goals = {},
                     std::span<const u64> seeds = {}, cudaStream_t stream = nullptr);
    /// The same from host states.
    MultiIwBatch run(std::span<const State> starts, std::span<const search::GoalSpec::AtomGoal> goals = {},
                     std::span<const u64> seeds = {});

    [[nodiscard]] const MultiIwOptions& options() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};

/// Why the device cannot run these searches (empty: it can).
[[nodiscard]] std::string multi_iw_unsupported(const Task& task, const MultiIwOptions& options);

/// search::iw() from every start state, on the device: results[i] equals iw(task, {.start = starts[i], ...}).
[[nodiscard]] std::vector<search::IwResult> multi_iw(ContextPtr ctx, TaskPtr task, std::span<const State> starts,
                                                     const MultiIwOptions& options = {},
                                                     std::span<const search::GoalSpec::AtomGoal> goals = {});

/// Exact batched IW(1) (an RL building block): optimized IW(1) from each of the device start rows; per search
/// the status, plan length, goal row and counts (MultiIwBatch). options.max_arity is forced to 1.
[[nodiscard]] MultiIwBatch batched_iw1(ContextPtr ctx, TaskPtr task, DeviceStarts starts, MultiIwOptions options = {},
                                       cudaStream_t stream = nullptr);

// ------------------------------------------------------------------------------------------------ over task tables
// IW over a task table: search i runs on instance task_ids[i] of the table (its start row in the table's row
// layout, rl/task_table.hpp; its goal the instance's, or goals[i]), with that instance's results: search i of a table
// run equals search 0 of DeviceMultiIw(ctx, table->task(task_ids[i])) from the same start, and so search::iw(). The
// searches are grouped by instance (stable: ascending search index within an instance) and each group runs as the
// DeviceMultiIw of its instance, the groups one after the other (a DeviceTableIw keeps the instances' DeviceMultiIw
// between runs); the batch lists the searches in input order.

/// The per-instance DeviceMultiIw of a table's searches.
class DeviceTableIw
{
public:
    /// Throws std::invalid_argument when an instance cannot run on the device (multi_iw_unsupported).
    DeviceTableIw(ContextPtr ctx, rl::TaskTablePtr table, const MultiIwOptions& options = {});
    ~DeviceTableIw();
    DeviceTableIw(const DeviceTableIw&) = delete;
    DeviceTableIw& operator=(const DeviceTableIw&) = delete;

    /// One search per start row (table rows of at most the table's words; read on `stream`, null: the context's
    /// stream), on instance task_ids[i]. The batch's goal rows and reached atoms are table rows (words = the table's
    /// words, reached_words the widest instance's); plans are labels of the search's instance.
    MultiIwBatch run(DeviceStarts starts, std::span<const u32> task_ids, std::span<const search::GoalSpec::AtomGoal> goals = {},
                     std::span<const u64> seeds = {}, cudaStream_t stream = nullptr);
    /// The same from host states (each a state of its instance's task).
    MultiIwBatch run(std::span<const State> starts, std::span<const u32> task_ids,
                     std::span<const search::GoalSpec::AtomGoal> goals = {}, std::span<const u64> seeds = {});

    [[nodiscard]] const rl::TaskTablePtr& table() const noexcept;
    [[nodiscard]] const MultiIwOptions& options() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};

/// search::iw() from every start state, each on its instance: results[i] equals iw(table->task(task_ids[i]), {.start =
/// starts[i], ...}).
[[nodiscard]] std::vector<search::IwResult> multi_iw(ContextPtr ctx, rl::TaskTablePtr table, std::span<const u32> task_ids,
                                                     std::span<const State> starts, const MultiIwOptions& options = {},
                                                     std::span<const search::GoalSpec::AtomGoal> goals = {});

/// batched_iw1 over a table: optimized IW(1) from each device start row on its instance task_ids[i].
[[nodiscard]] MultiIwBatch batched_iw1(ContextPtr ctx, rl::TaskTablePtr table, DeviceStarts starts, std::span<const u32> task_ids,
                                       MultiIwOptions options = {}, cudaStream_t stream = nullptr);
}  // namespace mymyr::cuda
