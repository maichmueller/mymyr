#pragma once
// Device rollouts: search::find_rollouts_parallel on the device, one rollout per seed,
// all of them in the same launches (cuda/multi_iw.hpp):
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::DeviceRolloutOptions o;
//   o.seeds = {1, 2, 3, 4};                      // one rollout per seed (thousands per launch)
//   o.iw.max_next_layer_states = 64;
//   mymyr::cuda::DeviceRolloutsResult r = mymyr::cuda::find_rollouts(ctx, task, o);
//
// Semantics: rollout k equals the CPU rollout of the same seed, bit for bit: the IW ladder from the start state with
// randomized layer orders (every next layer is shuffled by the rollout's own SplitMix64 stream, cuda/device_rng.hpp, in
// the CPU's order: one Fisher-Yates shuffle per completed non-empty layer, across the passes), truncated layers
// (max_next_layer_states: the admission that fills the next layer stops the layer), status, plan, per-pass counts and
// the reached fluent atoms (every state the rollout created: the start state, every generated successor, and the
// successors mimir's plan extraction creates along a solved rollout's plan). The streams are counter-based, so a
// rollout never depends on the launch configuration (group size, chunk size) or on the other rollouts.
//
// Differences from search::find_rollouts_parallel (documented, tested in tests/cuda/test_device_iw.cpp):
//   - reached derived atoms, num_states, landing states and co-occurrence rows are not reported (fluent atoms only);
//   - numeric tasks are refused (std::invalid_argument); custom goals, blocked states, observers and successor-order
//     hooks are not offered; goals are the task's or one conjunction of fluent atoms per rollout;
//   - max_seconds is checked between chunks, not between pops.

#include "mymyr/cuda/multi_iw.hpp"

#include <optional>

namespace mymyr::cuda
{
struct DeviceRolloutOptions
{
    MultiIwOptions iw;         // the ladder of every rollout (iw.track_reached is forced on)
    std::vector<u64> seeds;    // one rollout per seed
    std::optional<State> start;  // default: the task's initial state
    /// Empty (the task's goal) or one per rollout.
    std::vector<search::GoalSpec::AtomGoal> goals;
};

struct DeviceRollout
{
    search::IwResult search;                          // status, plan, passes of the rollout's ladder
    std::vector<CanonicalAtom> reached_fluent_atoms;  // ascending
};

struct DeviceRolloutsResult
{
    std::vector<DeviceRollout> rollouts;  // in seed order
    MultiIwStats stats;
    std::string message;
};

/// Reusable device rollouts: the task upload and the device buffers persist across run() calls (the steady state of a
/// training loop; rollouts_batch() builds one per call).
class DeviceRollouts
{
public:
    /// iw.track_reached is forced on. Throws std::invalid_argument for tasks the device does not run.
    DeviceRollouts(ContextPtr ctx, TaskPtr task, const MultiIwOptions& iw = {});
    /// One rollout per seed from `start` (null: the task's initial state); goals: empty or one per seed.
    MultiIwBatch run(std::span<const u64> seeds, const State* start = nullptr,
                     std::span<const search::GoalSpec::AtomGoal> goals = {});

private:
    ContextPtr m_ctx;
    TaskPtr m_task;
    DeviceMultiIw m_iw;
    DeviceBuffer m_start;
    u64 m_start_words = 0;
};

/// Device rollouts from options.start (flat results: MultiIwBatch with reached bitsets over fluent slots).
[[nodiscard]] MultiIwBatch rollouts_batch(ContextPtr ctx, TaskPtr task, const DeviceRolloutOptions& options);
/// Device rollouts with per-rollout results as search::find_rollouts_parallel reports them.
[[nodiscard]] DeviceRolloutsResult find_rollouts(ContextPtr ctx, TaskPtr task, const DeviceRolloutOptions& options);
/// The reached fluent atoms of rollout i of a batch as ascending canonical ids.
[[nodiscard]] std::vector<CanonicalAtom> reached_atoms(const MultiIwBatch& batch, u32 i, const Task& task);
}  // namespace mymyr::cuda
