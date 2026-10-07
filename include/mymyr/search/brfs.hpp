#pragma once
// Breadth-first search.
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::BrfsResult r = mymyr::brfs(*task, {.threads = 8});
//
// Single-threaded (threads == 1 and a Flat, Chunked or Compact store): a BrFS loop over the
// chosen store, with the stored state list as the queue and SoA parent / schema / binding records.
// Multi-threaded (threads > 1, or store == Concurrent): a layer-synchronous search over the
// concurrent store. Ids are deterministic by default: an atomic-min discoverer key (parent id,
// successor index) plus a per-layer counting sort reproduces the single-threaded ids at every thread count.
//
// Every expanded state is goal-tested (goal_states counts them); stop_at_goal ends the search at the first goal
// state expanded and, single-threaded, returns the plan. max_depth caps the expanded layers: with max_depth = D the
// states of depth < D are expanded and those of depth D stored (exhausted then says whether depth D was empty).
//
// Numeric tasks: states are [bits | slots] rows; every store keeps the numeric words next to the atom part and
// deduplicates on both.

#include "mymyr/core/types.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/successor/action.hpp"

#include <array>
#include <string>
#include <vector>

namespace mymyr
{
class Task;

struct BrfsOptions
{
    enum class Store : u8
    {
        Auto,        // Flat when the estimated width is at most 8 words, Chunked above; Concurrent if threads > 1
        Flat,        // fixed-stride bitset arena (single-threaded)
        Chunked,     // hash-consed 64 B chunks (single-threaded)
        Compact,     // 128-bit fingerprints of closed states plus two full layers (single-threaded; opt-in)
        Concurrent,  // per-thread arenas plus one lock-free table (any thread count)
    };

    u32 threads = 1;  // 0: std::thread::hardware_concurrency()
    Store store = Store::Auto;
    bool witness_pruning = true;
    bool canonical_order = true;     // per state, successors in (schema, binding) order
    bool deterministic_ids = true;   // multi-threaded: ids independent of the thread count
    u64 max_states = ~u64{0};        // stop expanding once this many states are stored
    u32 max_depth = ~u32{0};         // expand only states of depth < max_depth (depth-max_depth states are stored)
    bool layer_stats = false;        // fill BrfsResult::layer_counts
    bool stop_at_goal = false;
    /// What counts as a goal state (stop_at_goal, BrfsResult::goal_states): the task's goal by default. A custom test
    /// (GoalSpec::Kind::Custom) needs threads == 1.
    search::GoalSpec goal{};
    bool fingerprint = false;        // hash over (id, canonical state) of the whole store (tests, determinism gates)
    /// The order in which a layer is expanded (search/layer_ordering.hpp; default: the queued BrFS). An ordered kind
    /// needs the flat or chunked store (single-threaded); state ids then follow the expansion order. With a beam the
    /// states it drops stay stored (counted in BrfsResult::states, never expanded, never entered again: mimir's
    /// duplicate pruning), and both novelty modes behave alike (there is no novelty table).
    search::LayerOrdering layers{};
};

struct BrfsResult
{
    u64 states = 0;     // stored states
    u64 expanded = 0;
    u64 generated = 0;  // successors generated (with duplicates)
    u64 goal_states = 0;
    u32 layers = 0;          // layers expanded (partly, if the search stopped inside one)
    /// With layer_stats: per expanded layer {expanded, generated, new states} (mimir's "layers").
    std::vector<std::array<u64, 3>> layer_counts;
    bool exhausted = false;  // the whole reachable space was expanded
    bool solved = false;     // stop_at_goal found a goal state
    std::vector<Action> plan;
    double search_s = 0;
    u32 words = 0;         // fluent width at the end
    u32 fluent_slots = 0;  // assigned fluent atom slots at the end
    u64 store_bytes = 0;
    u64 fingerprint = 0;
    u32 threads = 1;
    std::string store;  // "flat", "chunked", "compact" or "concurrent"
};

BrfsResult brfs(const Task& task, const BrfsOptions& options = {});

/// The fingerprint term of state `id`: brfs(...).fingerprint is the xor of these over all stored states.
[[nodiscard]] u64 brfs_fingerprint_term(u64 id, u64 canonical_state_hash);
}  // namespace mymyr
