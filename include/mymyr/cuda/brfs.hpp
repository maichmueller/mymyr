#pragma once
// The device layer BrFS:
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::DeviceBrfsResult r = mymyr::cuda::brfs(ctx, task, {.fingerprint = true});
//
// Layer-synchronous and host-driven: each layer's frontier is the id range [begin, end) of the state arena, processed
// in chunks of parents. Per chunk: views, goal test, counts and scan, label rows, successor rows (cuda/generator.hpp,
// cuda/lifted.hpp), then the state set (cuda/state_set.hpp) assigns the new ids by a scan in candidate order. The
// chunks are sized on the device (a candidate capacity from the candidates per parent of the last chunks, the table's
// limit; a chunk past either writes and commits nothing and is redone with room, from its rows when the generator
// still holds its count), so a layer's chunks run in groups with one host read per group (a layer of one chunk: one
// read, where there were two). The candidates come in canonical order (parent, schema, binding)
// and the smallest candidate of a content wins, so the ids equal those of the CPU BrFS with deterministic ids
// (brfs(task, {.threads = N})) for every chunk size and launch configuration: brfs_fingerprint_term over (id,
// canonical state hash) gives the same fingerprint.
//
// States live in a DeviceArena (one row of `words()` words per id, with a pinned mirror kept in sync when the CPU
// needs the parents), the node records (parent id, index among the parent's successors) in a second one: a plan is
// the chain of node records, replayed on the CPU (the index names the action in canonical order).
//
// Semantics follow the CPU BrFS: every expanded state is goal-tested (goal_states); stop_at_goal expands the parents
// before the first goal state of its layer (in id order) and stops, with the plan. The max_states budget is checked
// before each chunk (like the multi-threaded CPU BrFS checks per layer), so the store may exceed it by one chunk's new
// states.

#include "mymyr/cuda/arena.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"

#include <memory>

namespace mymyr::cuda
{
struct DeviceBrfsOptions
{
    bool witness_pruning = true;
    bool canonical_order = true;  // without it a (state, schema) keeps the matcher's order (still deterministic)
    u64 max_states = ~u64{0};
    bool stop_at_goal = false;
    bool fingerprint = false;          // BrfsResult::fingerprint (downloads the states once at the end)
    u32 chunk_states = u32{1} << 20;   // parents per chunk at most (tests use tiny chunks: the ids must not change)
    u64 view_bytes = 0;                // per-chunk view budget (lowers the chunk size); 0: half the device's L2 cache
                                       // (the views and candidates of a chunk then stay in L2), but at least 65536
                                       // states while their views fit 256 MB (and at least 8192)
    u64 expected_states = 0;           // pre-sizes the arenas and the state table (0: from max_states if finite)
    bool timings = false;              // per-phase device times (events per chunk: the host drives every chunk)
    u64 candidate_bytes = u64{32} << 20;  // a chunk's candidate capacity estimate goes no higher than this many
                                          // bytes of candidate rows (at least 256 rows): a chunk past it halts and is
                                          // redone with its candidates known, a device loop's chunk is cut to its first
                                          // parents whose candidates fit (DeviceBrfsStats::cuts)
};

struct DeviceBrfsStats
{
    double view_ms = 0;   // per-state views, with the axioms (axiom_ms) when they run on the device
    double gen_ms = 0;    // goal test, counts, scan, label rows, successor rows
    double dedup_ms = 0;  // table growth, insert, owner, scan, compaction
    double host_ms = 0;   // CPU work: fallback schemas, axioms on the CPU, lazy interning, uploads (wall time)
    u64 chunks = 0;
    u64 groups = 0;         // host reads of the chunks' records (groups of chunks run without one in between)
    u64 resumed = 0;        // chunks redone from their rows (past their capacity as the last of their group)
    u64 redone = 0;         // chunks redone from their views (past their capacity inside a group; goal halts)
    u64 loops = 0;          // device loops over small layers (each one host read; counted in groups)
    u64 captures = 0;       // their captured graphs
    u64 cuts = 0;           // loop chunks cut to the parents whose candidates fit the capacity
    u32 rehashes = 0;       // table growths (each rehashes the stored states)
    u32 uploads = 0;        // device task uploads (1 + refreshes under lazy slots)
    u32 widenings = 0;      // state width growths (lazy slots)
    u32 host_schemas = 0;   // schemas run by the CPU fallback
    u64 table_slots = 0;
    u64 device_bytes = 0;   // high-water mark of the context's pool during the search
    // conditional effects and axioms
    u32 ce_schemas = 0;        // schemas with conditional effects on the device
    u32 host_ce_schemas = 0;   // schemas with conditional effects on the CPU fallback (in host_schemas)
    bool device_axioms = false;  // the task has axioms and the device evaluates them
    double axiom_ms = 0;       // device time of the axiom kernels (in view_ms)
    double host_axiom_ms = 0;  // CPU time on axioms (in host_ms): the host evaluator when the axioms cannot run on the
                               // device, the CPU engine's axioms for the parents of CPU-fallback schemas
    u64 axiom_reruns = 0;      // lazy slots: chunks whose axioms were redone after derived atoms were interned
};

struct DeviceBrfsResult
{
    BrfsResult result;  // store = "device"; threads = 0
    DeviceBrfsStats stats;
};

/// A device BrFS over one task. The state space stays on the device after run() (states(), nodes()).
class DeviceBrfs
{
public:
    DeviceBrfs(ContextPtr ctx, TaskPtr task, const DeviceBrfsOptions& options = {});
    ~DeviceBrfs();
    DeviceBrfs(const DeviceBrfs&) = delete;
    DeviceBrfs& operator=(const DeviceBrfs&) = delete;

    /// Runs the search (once). Throws std::invalid_argument for tasks the device cannot run (ChunkGenerator::
    /// unsupported), std::length_error beyond 2^31 - 2 states.
    DeviceBrfsResult run();

    /// State rows [0, size) of words() words (ids in BrFS order), and node records {parent id, successor index}
    /// (the root's parent is 0xFFFFFFFF).
    [[nodiscard]] const DeviceArena& states() const;
    [[nodiscard]] const DeviceArena& nodes() const;
    [[nodiscard]] u32 words() const noexcept;
    /// The plan to state `id`: the node chain replayed on the CPU.
    [[nodiscard]] std::vector<Action> plan_to(u64 id) const;

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};

/// DeviceBrfs(ctx, task, options).run().
DeviceBrfsResult brfs(ContextPtr ctx, TaskPtr task, const DeviceBrfsOptions& options = {});
}  // namespace mymyr::cuda
