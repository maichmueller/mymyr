#pragma once
// State spaces: the full transition model of a task as flat arrays, matching the semantics of mimir's
// `StateSpaceImpl::create` (mimir 0.16.3).
//
//   auto r = mymyr::datasets::generate_state_space(task, {.threads = 8, .remove_if_unsolvable = false});
//   if (r.space) { r.space->forward_targets(); r.space->unit_goal_distances(); ... }
//
// Content:
//   - every reachable state, with ids in breadth-first discovery order (parent id, then the successor's index in the
//     parent's canonical successor order). The ids are deterministic: the same at every thread count;
//   - every applicable action of every state is a transition (witness pruning off, so every binding counts, parallel
//     edges and self-loops included, as in mimir's state space graph); the forward CSR lists them per source in
//     canonical order, with the label (schema, binding) and the transition cost;
//   - a reverse CSR (per target, its incoming transitions in forward edge order);
//   - goal distances (V*) by backward search from the goal states: breadth-first for the unit distance, Dijkstra over
//     the transition costs for the cost distance (for unit-cost tasks the two are equal);
//   - flags: goal, unsolvable (no path to a goal) and alive (neither), as in mimir. The initial state is id 0.
//
// Transition costs (heuristics::ActionCosts::transition): with total-cost, the sum of the action's total-cost
// increases, added up from 0 (other total-cost effects: applied to the parent's depth, minus that depth, as in mimir's
// StateRepositoryImpl::get_or_create_successor_state called by its BrFS with the search depth as the metric value);
// with a metric over the fluents, the metric difference between successor and parent; otherwise 1. Two deliberate
// deviations from mimir: mimir computes (depth + c) - depth also for increases, which rounds fractional costs (so
// that goal distances would not equal plan costs), and with a metric it subtracts the parent's depth from the
// successor's metric value, which makes the costs depend on the search order (and negative, which its Dijkstra
// rejects).
//
// Generators:
//   - threads == 1: a layer-synchronous breadth-first generator run on one thread;
//   - threads > 1: the same generator on a Team, over the concurrent store, with ids deterministic regardless of
//     thread count;
//   - generate_state_spaces: an instance pool, one task per worker thread (many small instances).
//
// State words: row i of state_words() holds state i in the task's slot layout (the fluent/derived words, then
// numeric_words() words for numeric tasks), zero-padded to row_words(). With frozen atoms
// (TaskOptions::Atoms::Frozen, the default for datasets) the layout is fixed; with lazy slots it is the task's
// current one, which depends on the order in which atoms were first reached and hence on the thread count. Every
// other array is independent of the thread count.
//
// Options that mirror mimir's `StateSpaceOptions`: max_states (mimir's max_num_states: the generation fails when
// the space has max_states states or more), remove_if_unsolvable (no state space when the initial state cannot
// reach a goal: the name notwithstanding, mimir removes the whole space, never single states) and max_seconds (mimir
// declares timeout_ms but does not use it). As in mimir, a task whose goal is statically false has no state space (its
// BrFS ends as unsolvable before expanding anything), whatever remove_if_unsolvable.

#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/task.hpp"

#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace mymyr::datasets
{
struct StateSpaceOptions
{
    u32 threads = 1;                                          // 0: std::thread::hardware_concurrency()
    u64 max_states = std::numeric_limits<u64>::max();         // fail (OutOfStates) when the space reaches this size
    f64 max_seconds = std::numeric_limits<f64>::infinity();   // fail (Timeout) after this wall-clock time
    bool remove_if_unsolvable = true;                         // fail (Unsolvable) if the initial state is unsolvable
    bool labels = true;                                       // keep (schema, binding) per transition
};

enum class StateSpaceStatus : u8
{
    Ok,
    OutOfStates,  // max_states reached
    Timeout,      // max_seconds reached
    Unsolvable,   // the goal is statically false, or remove_if_unsolvable and the initial state is unsolvable
};
[[nodiscard]] const char* to_string(StateSpaceStatus s) noexcept;

/// Unit goal distance of a state that cannot reach a goal (mimir uses INT32_MAX).
inline constexpr i32 k_unsolvable_distance = -1;

/// The arrays of a state space generated elsewhere (the device state spaces, cuda/state_space.hpp), for
/// StateSpace::create; each has the size and meaning of the StateSpace accessor of the same name.
struct StateSpaceArrays
{
    TaskPtr task;
    u32 num_states = 0;
    u32 words = 1, numeric_words = 0;
    bool labels = false;
    u32 label_width = 0;
    std::vector<u64> state_words;
    std::vector<u64> forward_offsets;
    std::vector<u32> forward_targets, label_schemas, label_bindings;
    std::vector<f64> costs;  // empty: every transition costs 1
    std::vector<u64> backward_offsets;
    std::vector<u32> backward_sources, backward_edges;
    std::vector<i32> unit_goal_distances;
    std::vector<f64> cost_goal_distances;
    std::vector<u8> goal_flags, unsolvable_flags, alive_flags;
    u32 threads = 1, layers = 0;
    f64 search_seconds = 0, post_seconds = 0;
};

class StateSpace
{
public:
    /// A state space from its arrays (moved in). The goal and unsolvable counts and the largest goal distance are
    /// computed from them. Throws std::invalid_argument when a size does not match or an offset array is not a CSR over
    /// num_states states.
    [[nodiscard]] static std::shared_ptr<const StateSpace> create(StateSpaceArrays&& arrays);

    [[nodiscard]] const TaskPtr& task() const noexcept { return m_task; }
    [[nodiscard]] u32 num_states() const noexcept { return m_n; }
    [[nodiscard]] u64 num_transitions() const noexcept { return m_targets.size(); }
    [[nodiscard]] u32 initial_state() const noexcept { return 0; }

    // ------------------------------------------------------------------------------------------ states
    /// Fluent words per state, numeric words per state, and the row width (their sum).
    [[nodiscard]] u32 words() const noexcept { return m_words; }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_numeric_words; }
    [[nodiscard]] u32 row_words() const noexcept { return m_words + m_numeric_words; }
    /// [num_states, row_words] state words in id order.
    [[nodiscard]] std::span<const u64> state_words() const noexcept { return m_states; }
    [[nodiscard]] StateView state(u32 id) const noexcept
    {
        const u64* r = m_states.data() + static_cast<u64>(id) * row_words();
        return {r, m_words, m_numeric_words ? r + m_words : nullptr, m_numeric_words};
    }
    /// Id of a state of this space, or -1. Builds a hash index on first use (thread-safe).
    [[nodiscard]] i64 find(StateView s) const;

    // ------------------------------------------------------------------------------------------ forward CSR
    /// [num_states + 1]: the transitions of state s are [offsets[s], offsets[s + 1]).
    [[nodiscard]] std::span<const u64> forward_offsets() const noexcept { return m_offsets; }
    [[nodiscard]] std::span<const u32> forward_targets() const noexcept { return m_targets; }
    [[nodiscard]] u32 source(u64 edge) const;  // binary search over forward_offsets
    /// Labels (options.labels): the schema per transition, and label_width() binding objects per transition (the
    /// largest schema arity; unused positions hold ~0u).
    [[nodiscard]] bool has_labels() const noexcept { return m_has_labels; }
    [[nodiscard]] std::span<const u32> label_schemas() const noexcept { return m_schemas; }
    [[nodiscard]] std::span<const u32> label_bindings() const noexcept { return m_bindings; }
    [[nodiscard]] u32 label_width() const noexcept { return m_label_width; }
    [[nodiscard]] Action label(u64 edge) const;
    /// Transition costs; empty when every transition costs 1 (unit_costs()).
    [[nodiscard]] bool unit_costs() const noexcept { return m_costs.empty(); }
    [[nodiscard]] std::span<const f64> costs() const noexcept { return m_costs; }
    [[nodiscard]] f64 cost(u64 edge) const noexcept { return m_costs.empty() ? 1.0 : m_costs[edge]; }

    // ------------------------------------------------------------------------------------------ backward CSR
    /// [num_states + 1]: the transitions into state t are backward_edges()[offsets[t] .. offsets[t + 1]) (forward edge
    /// indices, ascending), from the states backward_sources()[...].
    [[nodiscard]] std::span<const u64> backward_offsets() const noexcept { return m_boffsets; }
    [[nodiscard]] std::span<const u32> backward_sources() const noexcept { return m_bsources; }
    [[nodiscard]] std::span<const u32> backward_edges() const noexcept { return m_bedges; }

    // ------------------------------------------------------------------------------------------ labels of states
    /// Unit goal distance per state (k_unsolvable_distance if no goal is reachable).
    [[nodiscard]] std::span<const i32> unit_goal_distances() const noexcept { return m_unit; }
    /// Cost goal distance per state (+inf if no goal is reachable).
    [[nodiscard]] std::span<const f64> cost_goal_distances() const noexcept { return m_cost; }
    /// Flags per state (0 or 1).
    [[nodiscard]] std::span<const u8> goal_flags() const noexcept { return m_goal; }
    [[nodiscard]] std::span<const u8> unsolvable_flags() const noexcept { return m_unsolvable; }
    [[nodiscard]] std::span<const u8> alive_flags() const noexcept { return m_alive; }
    [[nodiscard]] bool is_goal(u32 s) const noexcept { return m_goal[s] != 0; }
    [[nodiscard]] bool is_unsolvable(u32 s) const noexcept { return m_unsolvable[s] != 0; }
    [[nodiscard]] bool is_alive(u32 s) const noexcept { return m_alive[s] != 0; }
    /// Goal and unsolvable states in increasing id order.
    [[nodiscard]] std::vector<u32> goal_states() const;
    [[nodiscard]] std::vector<u32> unsolvable_states() const;
    [[nodiscard]] u32 num_goal_states() const noexcept { return m_num_goal; }
    [[nodiscard]] u32 num_unsolvable_states() const noexcept { return m_num_unsolvable; }
    /// Largest finite unit goal distance (-1 if no state reaches a goal).
    [[nodiscard]] i32 max_goal_distance() const noexcept { return m_max_unit; }

    /// Generation statistics.
    [[nodiscard]] u32 threads() const noexcept { return m_threads; }
    [[nodiscard]] u32 layers() const noexcept { return m_layers; }
    [[nodiscard]] f64 search_seconds() const noexcept { return m_search_s; }
    [[nodiscard]] f64 post_seconds() const noexcept { return m_post_s; }
    [[nodiscard]] u64 bytes() const noexcept;

    StateSpace() = default;
    StateSpace(const StateSpace&) = delete;
    StateSpace& operator=(const StateSpace&) = delete;
    ~StateSpace();

private:
    friend class StateSpaceBuilder;

    TaskPtr m_task;
    u32 m_n = 0;
    u32 m_words = 0, m_numeric_words = 0;
    bool m_has_labels = false;
    u32 m_label_width = 0;
    std::vector<u64> m_states;
    std::vector<u64> m_offsets;
    std::vector<u32> m_targets;
    std::vector<u32> m_schemas, m_bindings;
    std::vector<f64> m_costs;
    std::vector<u64> m_boffsets;
    std::vector<u32> m_bsources, m_bedges;
    std::vector<i32> m_unit;
    std::vector<f64> m_cost;
    std::vector<u8> m_goal, m_unsolvable, m_alive;
    u32 m_num_goal = 0, m_num_unsolvable = 0;
    i32 m_max_unit = -1;
    u32 m_threads = 1, m_layers = 0;
    f64 m_search_s = 0, m_post_s = 0;
    struct Index;
    mutable std::shared_ptr<Index> m_index;  // find(): built on first use
    mutable std::once_flag m_index_once;
};
using StateSpacePtr = std::shared_ptr<const StateSpace>;

struct StateSpaceResult
{
    StateSpaceStatus status = StateSpaceStatus::Ok;
    StateSpacePtr space;  // set iff status == Ok
    u64 states = 0;       // states stored when the generation ended (also when it failed)
    f64 seconds = 0;      // wall-clock time of the generation
};

/// Generates the state space of `task`.
[[nodiscard]] StateSpaceResult generate_state_space(TaskPtr task, const StateSpaceOptions& options = {});

/// The instance pool: `threads` workers (0: hardware concurrency), each generating one task's state space at a time
/// with options.threads forced to 1. make_task(i) builds task i (called on the worker thread; it may parse PDDL);
/// consume(i, result) receives each result on the worker thread that made it (concurrently; it may drop the space).
/// The first exception of make_task, the generator or consume stops the pool and is rethrown.
void for_each_state_space(u64 count, const std::function<TaskPtr(u64)>& make_task,
                          const std::function<void(u64, StateSpaceResult&&)>& consume, const StateSpaceOptions& options,
                          u32 threads);

/// The pool over a list of tasks: results in input order.
[[nodiscard]] std::vector<StateSpaceResult> generate_state_spaces(std::span<const TaskPtr> tasks, const StateSpaceOptions& options,
                                                                  u32 threads);
}  // namespace mymyr::datasets
