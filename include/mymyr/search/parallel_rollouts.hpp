#pragma once
// Parallel IW rollouts, matching mimir 0.16.3's iw::find_rollouts_parallel: K independent IW ladders, each with a
// randomized layer order seeded by its own seed, run on a thread pool over one read-only Task (each thread uses its
// own Workspace).
//
//   mymyr::search::ParallelRolloutOptions o;
//   o.iw.max_arity = 1;
//   o.seeds = {1, 2, 3, 4};
//   o.report_landing_states = true;
//   auto r = mymyr::search::find_rollouts_parallel(*task, o);
//
// Each rollout is the IW ladder of search/iw.hpp (the same passes, optimized IW(1) included) on the IW family engine
// with LayerOrdering::Kind::Randomized: the states of every next layer are shuffled by a SplitMix64 seeded with the
// rollout's seed (core/random.hpp; portable, unlike mimir's std::mt19937_64 with std::shuffle), the generator
// persisting across layers and passes; max_next_layer_states truncates layers as in mimir. Per rollout the result
// reports what mimir's private state repository records, over every state the rollout created (every successor
// of every expanded state, pruned or not, across all passes, the start state included, and for a solved rollout
// every successor of every non-goal plan state, which mimir's plan extraction creates; that adds states only where
// a truncated layer cut a plan state's enumeration short):
//   - reached fluent and derived atoms, num_states (distinct states);
//   - landing states (report_landing_states): per fluent atom, the first created state holding it; distinct states,
//     each with its atoms and whether it has no applicable action (a direct dead end);
//   - co-occurrence (report_co_occurrence): per fluent atom, the union of the atoms of the created states holding it.
// Atoms are canonical ids (task/atom_index.hpp), so results do not depend on the order in which threads assign lazy
// slots: for a given seed a rollout's result is identical at every thread count. Landing states are ordinary States
// of the task (mimir's migrate_landing_states is not needed; merge_landing_states deduplicates across rollouts).
//
// Threads: num_threads (0: the hardware's) capped at the number of rollouts; rollouts are handed out dynamically and
// results are returned in seed order. Per-worker hooks (search/control.hpp, tyr's make_worker protocol): a worker is
// a rollout (index into seeds). With an observer, SearchObserver::make_worker(k) is called for every rollout on the
// calling thread; rollout k sends its hot events (on_expand, on_generate, on_prune, on_transition, on_progress) to the
// returned observer. The root observer gets on_start before and, after every rollout has joined, in rollout order
// on_pass for each pass of each rollout and on_solution for each solved rollout, then on_end once (status Solved if
// any rollout solved, else the first non-exhausted status, else Exhausted). A Custom goal needs
// GoalSpec::make_worker(k) the same way. If the observer's make_worker or the goal's returns null/empty for some
// rollout, the batch runs on the calling thread alone (threads_used = 1) with every event on the root observer and
// the shared goal test. IwOptions::successor_order is called from the workers and must be thread-safe.
// Budgets: max_seconds spans the whole call; the other budgets apply per rollout as in iw(). A coordination
// (SearchControl::coordination) is rejected, as mimir rejects it with a layer order.

#include "mymyr/search/iw.hpp"

#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::search
{
struct ParallelRolloutOptions
{
    IwOptions iw;                        // the ladder of every rollout
    std::vector<u64> seeds;              // one rollout per seed
    u32 num_threads = 0;                 // 0: std::thread::hardware_concurrency()
    u32 max_next_layer_states = ~u32{0};
    bool report_landing_states = false;
    bool report_co_occurrence = false;
};

struct LandingState
{
    State state;
    std::vector<CanonicalAtom> atoms;  // ascending
    bool direct_dead_end = false;      // no applicable action
};

struct RolloutResult
{
    IwResult search;  // status, plan, passes of the rollout's ladder
    std::vector<CanonicalAtom> reached_fluent_atoms;   // ascending
    std::vector<CanonicalAtom> reached_derived_atoms;  // ascending
    u64 num_states = 0;
    std::vector<LandingState> landing_states;                          // in order of their smallest first-achieved atom
    std::vector<std::pair<CanonicalAtom, u32>> landing_state_by_atom;  // (atom, index into landing_states), ascending
    std::vector<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>> co_occurrence;  // (atom, row), ascending
};

struct ParallelRolloutsResult
{
    std::vector<RolloutResult> rollouts;  // in seed order
    u32 threads_used = 0;
    std::string message;  // why the batch could not run (every rollout then has status Failed), or why it ran serially
};

[[nodiscard]] ParallelRolloutsResult find_rollouts_parallel(const Task& task, const ParallelRolloutOptions& options);

/// mimir's intersect_co_occurrence: per atom of the first rollout, the intersection of its rows over all rollouts
/// (a rollout that never reached the atom clears it; rows that end up empty are kept).
[[nodiscard]] std::vector<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>> intersect_co_occurrence(std::span<const RolloutResult> results);

/// Distinct landing states over a batch, and per rollout the indices of its landing states among them.
struct MergedLandingStates
{
    std::vector<State> states;
    std::vector<std::vector<u32>> by_rollout;
};
[[nodiscard]] MergedLandingStates merge_landing_states(std::span<const RolloutResult> results);
}  // namespace mymyr::search
