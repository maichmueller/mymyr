#pragma once
// Batched grounded heuristics on the device:
//
//   auto ctx = mymyr::cuda::DeviceContext::create(0);
//   mymyr::cuda::DeviceHeuristic h(ctx, task, {.kind = mymyr::heuristics::Kind::Max});
//   h.evaluate(rows, stride, words, n, out);  // device rows in, u32 values out (k_dead_end: +inf)
//
// The relaxed grounding of heuristics/relaxed_task.hpp (propositions, one operator per ground action and conditional
// effect, preconditions, effects, action costs) is uploaded once; every launch evaluates a batch of states, a group of
// threads per state (cuda/heuristics_kernels.hpp). Values:
//   - h_max, h_add and h² equal heuristics::make_heuristic's (grounded evaluation) exactly, dead ends (+inf) included;
//   - h_FF extracts the relaxed plan over h_max best supporters as the CPU does (a derived atom passes on its axiom's
//     supporter, every ground action counts once), but it breaks ties among equally cheap supporters by the lowest BFS
//     level and then the smallest operator id, where the CPU takes the first supporter its queue reaches. The two agree
//     where no tie decides; reference() is the CPU implementation of the device's rule (tests compare against it).
//   - set-additive uses the same BFS-level and operator-id ties. A proposition keeps its chosen operator, including
//     an axiom; its achiever set is the union of the preconditions' sets, plus (operator, proposition) for an action.
//     Axioms add no member. The goal union counts one member per supported proposition, weighted by its action cost.
//     reference() implements this rule on the CPU.
// h² sweeps over two global triangular pair tables. A block evaluates a state and reuses its slice for later rows;
// the number of resident blocks is reduced to fit max_scratch_bytes (default 512 MiB). More than 8191 propositions,
// or a single state's scratch larger than the budget, raises std::invalid_argument naming the size and limit.
// Its resolved variant is Sweep, with a block per state and global scratch for every launch configuration.
// Two variants of the h_max / h_add / supporter cost computation: synchronous Jacobi sweeps over all operators, or
// the bucketed frontier (a generalized Dijkstra over precondition counters that evaluates every operator once and stops when the goals
// are settled). Both compute the least fixpoint, so the values never depend on the variant, the group size, the grid
// or where the scratch lives.
// States with an atom outside the grounding (not reachable in the delete relaxation from the initial state) are
// evaluated by the CPU heuristic (its lifted fallback), as heuristics::make_heuristic does. h² and set-additive
// have no lifted fallback and throw std::runtime_error for such states, as their CPU counterparts do.
//
// Numeric values and constraints are ignored as in the CPU relaxation.
// Refused with std::invalid_argument ("mymyr: ..."): groundings beyond the budget
// (heuristics::GroundingBudget, as the CPU's Evaluation::Grounded), kinds other than Max, Add, FF, H2 and SetAdditive,
// and real costs the CPU refuses. Costs (heuristics::Costs, Auto by default) are those of the CPU heuristics: real costs
// are integers in units of 1 / cost_scale() (heuristics::ActionCosts::relaxed_cost). One DeviceHeuristic serves one stream at a time (its scratch is reused across launches).

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mymyr::cuda
{
enum class HeuristicVariant : u8
{
    Auto,      // the frontier for large groundings (O >= 2048) and for P >= 128 with O <= 2P, the sweeps otherwise
    Sweep,     // every round evaluates every operator
    Frontier,  // the bucketed frontier: every operator once, in cost order
};

[[nodiscard]] const char* to_string(HeuristicVariant v) noexcept;

struct DeviceHeuristicOptions
{
    heuristics::Kind kind = heuristics::Kind::FF;  // Max, Add, FF, H2 or SetAdditive
    heuristics::Costs costs = heuristics::Costs::Auto;
    heuristics::GroundingBudget budget;
    /// A grounding to use instead of building one (share it with CPU heuristics).
    std::shared_ptr<const heuristics::RelaxedTask> relaxed;
    HeuristicVariant variant = HeuristicVariant::Auto;
    // launch configuration: no value depends on it
    u32 threads = 0;          // threads per block (0: by the grounding's size)
    int warp_groups = -1;     // 1: a warp per state, 0: a block per state, -1: by the grounding's size
    u32 max_blocks = 0;       // 0: as many as are resident on the device
    u64 shared_bytes = 0;     // scratch per block kept in shared memory up to this size (0: 48 KB); larger: global
    bool force_global = false;  // scratch in global memory even when it fits shared memory (tests)
    /// Global scratch budget in bytes (fewer resident groups beyond it). h² refuses when even one group exceeds it.
    u64 max_scratch_bytes = u64{512} << 20;
};

struct DeviceHeuristicStats
{
    u64 evaluations = 0;  // states evaluated (device and CPU fallback)
    u64 launches = 0;
    u64 fallbacks = 0;    // states outside the grounding, evaluated by the CPU heuristic
    u32 propositions = 0, operators = 0, ground_actions = 0;
    double grounding_seconds = 0;
    // the resolved launch configuration
    HeuristicVariant variant = HeuristicVariant::Auto;
    u32 threads = 0, blocks = 0;
    bool warp_groups = false, shared = false;
    u64 group_bytes = 0;
    bool supporter_levels = false;  // h_FF and set-additive run the levels phase (axioms or different operator costs;
                                    // otherwise the supporters are the smallest tight achievers: heuristics_kernels.hpp)
};

class DeviceHeuristic
{
public:
    /// The most propositions h² accepts (the same limit as the CPU).
    static constexpr u32 k_h2_max_props = 8191;

    /// Dead end (+inf) among the u32 values.
    static constexpr u32 k_dead_end = 0xFFFFFFFFu;

    /// Throws std::invalid_argument for tasks or options the device does not run (see above).
    DeviceHeuristic(ContextPtr ctx, TaskPtr task, const DeviceHeuristicOptions& options = {});
    ~DeviceHeuristic();
    DeviceHeuristic(const DeviceHeuristic&) = delete;
    DeviceHeuristic& operator=(const DeviceHeuristic&) = delete;

    /// Why the device cannot evaluate this heuristic on this task (empty: it can). Does not ground the task.
    [[nodiscard]] static std::string unsupported(const Task& task, const DeviceHeuristicOptions& options);

    /// h of the n device rows rows + i * stride (`words` words each) into out[0, n), on `stream` (null: the context's
    /// stream) after the work enqueued on it. Synchronizes the stream once: states outside the grounding are then
    /// evaluated on the CPU and their values written back. Throws std::invalid_argument for a row that sets an atom
    /// slot the task has not assigned.
    /// The values are integers in units of 1 / cost_scale() (k_inf: a dead end).
    void evaluate(const u64* rows, u64 stride, u32 words, u64 n, u32* out, cudaStream_t stream = nullptr);
    /// The same as doubles: heuristics::Heuristic's values (+inf for dead ends).
    void evaluate(const u64* rows, u64 stride, u32 words, u64 n, f64* out, cudaStream_t stream = nullptr);
    /// Host states (uploaded, evaluated, downloaded).
    [[nodiscard]] std::vector<f64> evaluate(std::span<const State> states);
    /// h of the first min(n, *count) rows (count: a device u32) into out, without synchronizing (for work
    /// that is captured into CUDA graphs or looped on the device): no CPU fallback; the states outside the grounding
    /// and the rows that set unassigned slots are added to flags[0] and flags[1] (device u32s; their values are
    /// undefined). The states reachable from the task's initial state are inside its grounding. Call prepare(n,
    /// stream) first (it allocates and refreshes the lazy slots' tables; this call does not).
    void evaluate_async(const u64* rows, u64 stride, u32 words, u64 n, const u32* count, u32* out, u32* flags, cudaStream_t stream);
    /// Sizes the scratch for evaluate_async of up to n rows on `stream` and refreshes the lazy slots' tables.
    void prepare(u64 n, cudaStream_t stream);

    /// The CPU reference of the device's value of s: heuristics::make_heuristic's for h_max, h_add and h², the device's
    /// supporter rule for h_FF and set-additive (see above); +inf for dead ends.
    [[nodiscard]] f64 reference(StateView s);

    [[nodiscard]] heuristics::Kind kind() const noexcept;
    /// The integer values' unit is 1 / cost_scale(): 1 for unit costs, a power of ten for real costs
    /// (heuristics::ActionCosts::relaxed_scale).
    [[nodiscard]] f64 cost_scale() const noexcept;
    [[nodiscard]] const heuristics::RelaxedTask& relaxed() const noexcept;
    [[nodiscard]] const std::shared_ptr<const heuristics::RelaxedTask>& grounding() const noexcept;
    [[nodiscard]] const DeviceHeuristicStats& stats() const noexcept;
    [[nodiscard]] const ContextPtr& context() const noexcept;
    [[nodiscard]] const TaskPtr& task() const noexcept;

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::cuda
