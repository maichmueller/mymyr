#pragma once
// Device A* with batched expansion:
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::DeviceBestFirstOptions o;
//   o.search.heuristic.kind = mymyr::heuristics::Kind::Max;
//   mymyr::cuda::DeviceBestFirstResult r = mymyr::cuda::astar(ctx, task, o);  // r.result: a search::BestFirstResult
//
// Every step pops up to `batch` open nodes of the lowest bucket (f, then h; goal states first; first in first out
// within a bucket: the CPU astar_eager's order; DeviceBestFirstOptions::single_bucket = false pops the whole lowest f
// layer, h ascending), expands them together with the lifted successor kernels
// (cuda/generator.hpp), deduplicates the successors with the device state set (cuda/state_set.hpp), relaxes their g
// (a state reached more cheaply is updated, reopened when closed, as search::astar_eager does), evaluates the new
// states' heuristic in one batch (cuda/heuristics.hpp; blind: 0) and pushes them. The open list is a bucket queue on
// the device (astar_kernels.hpp): an arena of entries and a run list sorted by key (f, goal / h), the
// runs of a key in push order; the device sorts each chunk's entries by key and merges its runs into the list. A step
// keeps its counts on the device: steps are captured into a CUDA graph and looped on the device (one host read per
// loop of up to DeviceBestFirstOptions::loop_steps steps); the host takes over a step only where it must (CPU-fallback
// schemas, derived goals, an explicit start state, lazy slots that met new atoms, a chunk past the capacities: the
// same kernels, launched with reads between them). A goal state is recognized when generated and returned when it is
// popped (its f is minimal), so the plan is optimal with an admissible heuristic (Max, H2, Blind); h_add, h_FF and
// set-additive run as inadmissible options. The search stops after the expansion of the parent of the first goal of
// the current f layer (in candidate order), as
// the CPU's, which pops that goal next. Expanding a bucket at once costs expansions that the sequential search does
// not make (the bucket's states after the goal's parent, in other chunks; the order in which the CPU interleaves
// its pushes): stats.expanded reports them.
//
// Ties are broken deterministically: the relaxation keeps the cheapest candidate and among equally cheap ones the
// first (parent in pop order, successor in canonical order), entries are sorted stably, and ids come from the state
// set in candidate order. Nothing depends on the launch configuration or on timing.
//
// Supported: the task's goal (GoalSpec::Kind::Task), budgets (max_expanded exactly; max_states, max_seconds and
// cancellation between chunks; max_depth), unit and state-independent integral action costs (total-cost; evaluated on
// the device by cost programs, mymyr/cuda/cost_program.hpp: any number of cost parameters, any static function key
// space; an undefined or negative cost, or one of 2^31 or more, throws std::domain_error), witness pruning and
// canonical order as the CPU. Numeric tasks carry canonical double tails and evaluate fluent costs, ordered total-cost
// effects and state metrics in flat device programs. Their F64 priorities use a host heap, one parent per step, with
// eager CPU tie order; numeric searches do not capture graphs. Refused with std::invalid_argument ("mymyr: ..."): tasks the successor
// kernels cannot run (ChunkGenerator::unsupported), heuristics other than blind, max, add, ff, h2 and set_additive (and
// groundings beyond the budget), other goals, blocked states, observers and caller-owned evaluators.

#include "mymyr/cuda/heuristics.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/task/task.hpp"

#include <string>

namespace mymyr::cuda
{
struct DeviceBestFirstOptions
{
    /// The search: control.budget and control.cancel, heuristic (kind, costs, grounding budget, a shared grounding),
    /// start, reopen, witness_pruning and canonical_order are honored; store and queue are the device's.
    search::BestFirstOptions search;
    u32 batch = 10000;  // B: open nodes expanded per step
    /// A*: a step pops up to B entries of the lowest (f, h) bucket only (the default); false: of the whole lowest f
    /// layer (bigger steps, but the states of higher h in the goal's layer that the sequential search never reaches
    /// are expanded).
    bool single_bucket = true;
    /// The device heuristic's launch configuration (variant, threads, ...); kind, costs, budget and the grounding come
    /// from search.heuristic.
    DeviceHeuristicOptions heuristic;
    u64 chunk_states = 0;     // parents per generator chunk (0: as many as fit half of the L2 cache)
    u64 expected_states = 0;  // pre-sizes the state set (0: grows)
    /// Steps replayed from captured CUDA graphs, looped on the device without a time budget (false, or
    /// MYMYR_CUDA_GRAPHS=0: every step is launched from the host; MYMYR_CUDA_DEVICE_LOOPS=0: graphs, no loops).
    /// Nothing else depends on it.
    bool graphs = true;
    u32 loop_steps = 64;  // steps per device loop (cancellation is checked between loops)
};

struct DeviceBestFirstStats
{
    u64 steps = 0;       // batches expanded
    u64 chunks = 0;      // generator chunks
    u64 popped = 0;      // open entries popped (stale ones included)
    u64 stale = 0;       // popped entries of nodes that were closed or reached more cheaply since
    u64 max_batch = 0;   // largest batch
    u64 open_entries = 0;  // entries pushed in total
    // how the steps ran
    u64 graph_steps = 0;  // steps replayed from a captured graph (in device loops or one per launch)
    u64 host_steps = 0;   // steps the host drove (or finished after an abort)
    u64 loops = 0;        // device loop launches
    u64 captures = 0;     // graphs captured
    u64 aborts = 0;       // steps the host finished: a pop window too small, a chunk past the capacities, new atoms
    u32 widenings = 0, rehashes = 0, uploads = 0;
    u64 table_slots = 0;
    u64 device_bytes = 0;  // peak device memory of the context
    double host_ms = 0;    // host work of the generator (CPU-fallback schemas, axioms, lazy slots)
    double heuristic_ms = 0;
    DeviceHeuristicStats heuristic;
};

struct DeviceBestFirstResult
{
    search::BestFirstResult result;  // status, plan, cost, goal state, statistics as the CPU search reports them
    DeviceBestFirstStats device;
};

/// Why the device cannot run this search on this task (empty: it can). Does not ground the task.
[[nodiscard]] std::string best_first_unsupported(const Task& task, const DeviceBestFirstOptions& options);

/// A* on the device (see above). Throws std::invalid_argument for what it refuses.
[[nodiscard]] DeviceBestFirstResult astar(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options = {});

namespace detail
{
/// The driver shared by astar() and gbfs() (cuda/gbfs.hpp).
[[nodiscard]] DeviceBestFirstResult best_first(ContextPtr ctx, TaskPtr task, const DeviceBestFirstOptions& options, bool greedy);
}  // namespace detail
}  // namespace mymyr::cuda
