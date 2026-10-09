#pragma once
// Device state spaces: the content of
// datasets::StateSpace generated on the device, for one task or for many instances of a task table at once.
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::DeviceStateSpaceResult r = mymyr::cuda::state_space(ctx, task);
//   if (r.space) { r.space->forward_targets(); r.space->unit_goal_distances(); ... }   // device pointers
//   auto spaces = mymyr::cuda::state_spaces(ctx, table, {.output = StateSpaceOutput::Host});  // datasets::StateSpace each
//
// Content: the same arrays as datasets/state_space.hpp, array by array: the ids in breadth-first discovery order
// (parent id, then the successor's index in the parent's canonical successor order: the ids of
// datasets::generate_state_space at every thread count); every applicable action a transition (witness pruning off,
// parallel edges and self-loops kept); the forward CSR with labels (schema, binding padded with ~0u to
// label_width()) and costs; the reverse CSR (per target, its incoming transitions in ascending forward edge order);
// goal, unsolvable and alive flags; unit and cost goal distances; the initial state is id 0. The options are the
// host's (datasets::StateSpaceOptions: max_states, max_seconds, remove_if_unsolvable, labels; symmetry pruning is
// refused), with the host's statuses per instance. The `states` of an instance that runs out of states is the
// host's count at the failure (its one-thread generator's, after the parent whose successors reached
// max(max_states, 2), whatever the chunk); max_seconds is checked between chunks.
//
// Generation. One pipeline expands the states of every instance of a wave (below) in first-in first-out chunks of
// stored parents (a chunk may span layers and instances): the successor rows of a chunk come in (parent, canonical
// index) order and the state set (cuda/src/state_space_kernels.hpp: the device BrFS's table with the instance in the
// key) gives the new states ids by a scan in candidate order, so each instance's states are numbered in its own BrFS
// order within the shared (interleaved) numbering, whatever the chunk size, the wave or the stream. After the wave a
// stable partition by instance gives every instance its local ids, equal to a run of its own.
//   - A single task, and tables that the multi-instance kernels cannot run (numeric fluents, conditional effects, axioms, lazy slots,
//     CPU-fallback schemas: rl's multi_unsupported), use cuda::ChunkGenerator per instance (conditional effects and
//     axioms on the device, lazy interning in canonical-id order with widening, the CPU fallback), one instance after
//     the other;
//   - other tables use the multi-instance kernels (cuda/lifted.hpp "several instances"): the rows of a chunk carry
//     their instance and the kernels read views[instance].
//
// Waves bound the device memory of a table: instances are admitted (their initial state appended) between chunks while
// the wave holds fewer than wave_states states and wave_instances instances; when the wave's states are all expanded
// and no instance can be admitted, the wave is post-processed and its results output. A wave's device memory is about
// states x (row bytes + 64) + transitions x (32 + 4 x label_width) bytes (plus the table and a chunk's scratch), for
// its states: at most wave_states plus those of the instances admitted last (an instance is never split). With host
// output (StateSpaceOutput::Host) each wave's device memory is freed before the next wave starts, so the peak is one
// wave; with device output the results of every wave stay on the device (their arrays, not the generation scratch).
//
// Post-processing on the device, over the wave's union graph (the instances are disjoint components):
//   - reverse CSR: a stable radix sort of (target, edge) pairs, sources by binary search over the forward offsets;
//   - unit goal distances: level-synchronous backward BFS from the goal states (frontier queues, one synchronization
//     per level);
//   - cost goal distances (non-unit costs): a label-correcting backward search (worklist Bellman-Ford with atomicMin on
//     the bit patterns of non-negative doubles). Its result is exact for real costs as well as integral ones: every
//     value it stores is the value of a path summed from the goal backward, as Dijkstra sums them, and with c >= 0 the
//     rounded sum fl(d + c) is monotone in d and at least d, so both compute the minimum over paths of those sums;
//     the order of relaxations does not matter. Negative or NaN costs throw std::domain_error (as the host's Dijkstra);
//   - transition costs: Unit tasks have none; state-independent total-cost tasks get the host's cost (the action's
//     cost from its instance's cost program, any number of cost parameters and any static function key space:
//     mymyr/cuda/cost_program.hpp) on the device, at every transition of the chunk; other costs (conditional or
//     several total-cost effects, metrics) regenerate the transitions of each state on the CPU (space.threads threads)
//     with the host's formula (heuristics::ActionCosts::transition). A transition whose cost is
//     undefined (the CPU drops such actions; the device kernels do not) throws std::domain_error.
//
// Device arrays live in buffers shared by the instances of a wave (DeviceStateSpace::storage() keeps them alive) and
// are written on the result's stream; to_host() downloads one space into a datasets::StateSpace. Numeric tasks are
// supported with the CPU numeric encoding; tasks beyond ChunkGenerator::unsupported limits are refused.
//
// Host output overlaps the post-processing: the members' host arrays are sized (first touch, threads) in the
// background once their sizes are known, each array is downloaded as soon as it is final (the rows, offsets, labels,
// costs and goal flags first, the targets after the reverse sort; through pinned staging buffers the context keeps,
// DeviceContext::lease_pinned), and the flags, unit-cost goal distances and (for spaces small next to the wave) the
// reverse CSR are derived on the host from the downloaded arrays instead of crossing the bus.

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <span>
#include <vector>

namespace mymyr::cuda
{
enum class StateSpaceOutput : u8
{
    Device,  // DeviceStateSpaceResult::space: the arrays on the device
    Host,    // DeviceStateSpaceResult::host: datasets::StateSpace (the device memory of a wave is freed after it)
    Both,
};

struct DeviceStateSpaceOptions
{
    /// The host generator's options (threads: host threads for the host-side work, 0: hardware concurrency;
    /// symmetry_pruning must be off).
    datasets::StateSpaceOptions space{};
    StateSpaceOutput output = StateSpaceOutput::Device;
    u32 chunk_states = u32{1} << 20;  // parents per chunk at most (tests use tiny chunks: the results must not change)
    u64 view_bytes = 0;               // per-chunk view budget (lowers the chunk size); 0: as cuda::DeviceBrfs
    u64 expected_states = 0;          // pre-sizes a wave's arrays and state table (0: grown as needed)
    u64 wave_states = u64{1} << 24;   // a wave admits instances while it holds fewer states than this
    u32 wave_instances = 0;           // instances per wave at most (0: no limit)
    cudaStream_t stream = nullptr;    // the stream of the work and the results (null: the context's stream)
};

/// Counters of a run (state_space() or state_spaces()).
struct DeviceStateSpaceStats
{
    u32 waves = 0;
    u64 chunks = 0;
    bool multi = false;          // the multi-instance kernels ran the table (else ChunkGenerator per instance)
    u64 states = 0, transitions = 0;
    double generate_ms = 0;      // wall time of the chunk loops
    double post_ms = 0;          // wall time of the post-processing (reverse CSR, distances, flags, local ids)
    double output_ms = 0;        // wall time of the host output (downloads and datasets::StateSpace)
    double host_cost_ms = 0;     // wall time of the CPU cost regeneration (state-dependent costs)
    u32 rehashes = 0;
    u32 widenings = 0;           // lazy slots: state width growths
    u64 device_bytes = 0;        // high-water mark of the context's pool
};

/// One instance's state space on the device. Arrays are device pointers into the wave's shared buffers, ready after
/// the work enqueued on stream() so far; ids are local (0 is the initial state). Immutable.
class DeviceStateSpace
{
public:
    struct Storage;  // the wave's buffers (state_space.cpp)
    struct Part      // where this instance's arrays start in the storage (elements)
    {
        u64 states = 0, edges = 0, rows = 0, offsets = 0;
    };

    DeviceStateSpace(std::shared_ptr<const Storage> storage, Part part, TaskPtr task, u32 num_states, u64 num_transitions,
                     u32 words, u32 label_width, bool labels, bool unit_costs, u32 num_goal, u32 num_unsolvable,
                     i32 max_goal_distance, u32 layers);

    [[nodiscard]] const TaskPtr& task() const noexcept { return m_task; }
    [[nodiscard]] u32 num_states() const noexcept { return m_n; }
    [[nodiscard]] u64 num_transitions() const noexcept { return m_e; }
    [[nodiscard]] u32 initial_state() const noexcept { return 0; }
    /// Words per state row (max(1, task words)).
    [[nodiscard]] u32 words() const noexcept { return m_words; }
    [[nodiscard]] bool has_labels() const noexcept { return m_labels; }
    [[nodiscard]] u32 label_width() const noexcept { return m_label_width; }
    [[nodiscard]] bool unit_costs() const noexcept { return m_unit_costs; }
    [[nodiscard]] u32 num_goal_states() const noexcept { return m_num_goal; }
    [[nodiscard]] u32 num_unsolvable_states() const noexcept { return m_num_unsolvable; }
    [[nodiscard]] i32 max_goal_distance() const noexcept { return m_max_unit; }
    [[nodiscard]] u32 layers() const noexcept { return m_layers; }

    [[nodiscard]] u32 numeric_words() const noexcept { return m_task->numeric_words(); }
    [[nodiscard]] u32 row_words() const noexcept { return m_words + numeric_words(); }

    /// [num_states, row_words] state words in id order, with the CPU numeric encoding.
    [[nodiscard]] const u64* state_words() const noexcept;
    /// [num_states + 1] and [num_transitions]: the forward CSR.
    [[nodiscard]] const u64* forward_offsets() const noexcept;
    [[nodiscard]] const u32* forward_targets() const noexcept;
    /// [num_transitions] and [num_transitions, label_width] (null without labels).
    [[nodiscard]] const u32* label_schemas() const noexcept;
    [[nodiscard]] const u32* label_bindings() const noexcept;
    /// [num_transitions] (null when every transition costs 1).
    [[nodiscard]] const f64* costs() const noexcept;
    /// [num_states + 1], [num_transitions], [num_transitions]: the reverse CSR.
    [[nodiscard]] const u64* backward_offsets() const noexcept;
    [[nodiscard]] const u32* backward_sources() const noexcept;
    [[nodiscard]] const u32* backward_edges() const noexcept;
    /// [num_states] each.
    [[nodiscard]] const i32* unit_goal_distances() const noexcept;
    [[nodiscard]] const f64* cost_goal_distances() const noexcept;
    [[nodiscard]] const u8* goal_flags() const noexcept;
    [[nodiscard]] const u8* unsolvable_flags() const noexcept;
    [[nodiscard]] const u8* alive_flags() const noexcept;

    [[nodiscard]] const ContextPtr& context() const noexcept;
    [[nodiscard]] cudaStream_t stream() const noexcept;
    /// Makes `consumer` wait for the work that writes the arrays, and keeps their memory until the work enqueued on
    /// `consumer` so far has completed (DeviceBuffer::record_stream): the handoff of a zero-copy export to another
    /// stream.
    void use_on(cudaStream_t consumer) const;
    /// Keeps the arrays alive (exports hold it).
    [[nodiscard]] std::shared_ptr<const void> storage() const noexcept { return m_storage; }
    /// The space as a datasets::StateSpace (downloads; synchronizes the stream).
    [[nodiscard]] datasets::StateSpacePtr to_host() const;

private:
    std::shared_ptr<const Storage> m_storage;
    Part m_part;
    TaskPtr m_task;
    u32 m_n = 0;
    u64 m_e = 0;
    u32 m_words = 1, m_label_width = 0;
    bool m_labels = false, m_unit_costs = true;
    u32 m_num_goal = 0, m_num_unsolvable = 0;
    i32 m_max_unit = -1;
    u32 m_layers = 0;
};
using DeviceStateSpacePtr = std::shared_ptr<const DeviceStateSpace>;

struct DeviceStateSpaceResult
{
    datasets::StateSpaceStatus status = datasets::StateSpaceStatus::Ok;
    DeviceStateSpacePtr space;     // status Ok and device output
    datasets::StateSpacePtr host;  // status Ok and host output
    u64 states = 0;                // states stored when the instance's generation ended (also when it failed)
    f64 seconds = 0;               // wall time from the instance's admission to its output
};

struct DeviceStateSpaces
{
    std::vector<DeviceStateSpaceResult> results;  // per instance of the table, in table order
    DeviceStateSpaceStats stats;
};

/// Why the device cannot generate the state space of `task` (empty: it can): ChunkGenerator::unsupported.
[[nodiscard]] std::string state_space_unsupported(const Task& task);

/// The state space of one task. Throws std::invalid_argument for tasks the device cannot run and for symmetry pruning,
/// std::length_error beyond 2^31 - 2 states or 2^32 - 1 transitions, std::domain_error for negative, NaN or undefined
/// transition costs.
[[nodiscard]] DeviceStateSpaceResult state_space(ContextPtr ctx, TaskPtr task, const DeviceStateSpaceOptions& options = {},
                                                 DeviceStateSpaceStats* stats = nullptr);

/// The state spaces of every instance of `table` (results in table order), in waves.
[[nodiscard]] DeviceStateSpaces state_spaces(ContextPtr ctx, rl::TaskTablePtr table, const DeviceStateSpaceOptions& options = {});

/// The host's generalized state space (datasets::GeneralizedStateSpace::create over datasets::ordered_spaces) of device
/// results: their host spaces, downloaded where only the device ones exist.
[[nodiscard]] std::shared_ptr<const datasets::GeneralizedStateSpace> generalized_state_space(
    std::span<const DeviceStateSpaceResult> results, bool sort_ascending = true);
}  // namespace mymyr::cuda
