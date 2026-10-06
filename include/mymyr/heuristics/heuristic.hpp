#pragma once
// Heuristics.
//
//   auto h = mymyr::heuristics::make_heuristic(*task, {.kind = mymyr::heuristics::Kind::FF});
//   double v = h->evaluate(state);  // +inf: dead end of the relaxation
//
// Kinds. Values equal mimir 0.16.3's except where noted:
//   Blind      0 in every state (mimir's BlindHeuristic).
//   GoalCount  the number of fluent and derived goal literals that do not hold. Not one of mimir's heuristics: the
//              complement of mimir's GoalCountLayerOrderingStrategy score.
//   Max, Add   h_max and h_add over mimir's relaxation (relaxed_task.hpp): generalized Dijkstra with per-operator
//              counters of unsatisfied preconditions; an axiom combines its preconditions by max in both (as
//              mimir's RelaxedPlanningGraph does); unit action costs by default.
//   FF         the relaxed plan mimir extracts: best supporters by h_max cost (the first operator reaching an atom's
//              final cost supports it), a derived atom takes over the supporter of its axiom's last settled
//              precondition, every ground action counts once. Among equally cheap supporters mimir picks by the
//              order of a binary heap over its atom numbering; mymyr picks by its own operator order, so h_FF can
//              differ from mimir's where such ties exist.
//   Custom     a Heuristic subclass of the caller (BestFirstOptions::evaluator), e.g. a learned value function.
// Numeric tasks. As mimir's delete relaxation (DeleteRelaxTranslator), h_max, h_add and h_FF ignore numeric conditions
// (of actions, effects and the goal) and numeric effects: they see the task's atoms only. Goal count counts the
// fluent and derived goal literals; numeric goal constraints are not counted.
// Costs. Unit: every action costs 1 and every axiom 0 (mimir's heuristics). Real: the task's action costs
// (heuristics/action_costs.hpp); they must be non-negative integers.
// Evaluation. Grounded (the relaxed task is built on first use, within the budget) or lifted (a cost-bucketed semi-naive
// fixpoint over lifted matchers; the fallback beyond the budget and for states outside the grounded relaxation).
//
// A Heuristic holds per-thread scratch: create one per search thread. Heuristics made from the same Options share the
// grounding when Options::relaxed is set (use ground() to build it once).

#include "mymyr/core/types.hpp"
#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::heuristics
{
using Value = f64;
inline constexpr Value k_dead_end = std::numeric_limits<f64>::infinity();

enum class Kind : u8
{
    Blind,
    GoalCount,
    Max,
    Add,
    FF,
    Custom,  // a Heuristic the caller implements (a learned value function, a Python callable); make_heuristic refuses it
};

enum class Costs : u8
{
    Unit,  // every action 1, axioms 0 (mimir)
    Real,  // the task's action costs (total-cost)
};

enum class Evaluation : u8
{
    Auto,      // grounded within the budget, lifted beyond it and for states the grounding does not cover
    Grounded,  // grounded; throws std::runtime_error if the grounding exceeds the budget
    Lifted,    // lifted only
};

[[nodiscard]] const char* to_string(Kind k) noexcept;
/// "blind", "goal_count" (or "gc"), "max" (or "hmax"), "add" (or "hadd"), "ff" (or "hff"). Throws std::invalid_argument.
[[nodiscard]] Kind parse_kind(std::string_view name);

struct Options
{
    Kind kind = Kind::FF;
    Costs costs = Costs::Unit;
    Evaluation evaluation = Evaluation::Auto;
    GroundingBudget budget;
    /// A grounding to use instead of building one (share it between heuristics and threads).
    std::shared_ptr<const RelaxedTask> relaxed;
};

/// Builds the grounding for `task` (nullptr beyond the budget), to share through Options::relaxed.
[[nodiscard]] std::shared_ptr<const RelaxedTask> ground(const Task& task, const GroundingBudget& budget = {}, GroundingStats* stats = nullptr);

/// Thrown by an evaluation that the interrupt hook (Heuristic::set_interrupt) stopped.
struct Interrupted : std::runtime_error
{
    Interrupted() : std::runtime_error("mymyr: heuristic evaluation interrupted") {}
};

struct HeuristicStats
{
    u64 evaluations = 0;
    u64 grounded = 0;  // evaluations by the grounded relaxation
    u64 lifted = 0;    // evaluations by the lifted fallback
    u64 dead_ends = 0;
    double grounding_seconds = 0;
};

class Heuristic
{
public:
    virtual ~Heuristic() = default;

    [[nodiscard]] virtual Kind kind() const noexcept = 0;
    /// h of s for the task's goal. +inf marks a dead end of the relaxation.
    virtual Value evaluate(StateView s) = 0;
    /// h of s for "any of these goals": the minimum over the goals (search GoalSpec::AnyOf).
    virtual Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> goals) = 0;
    /// h of every state of `states` into `out` (the same length), for the task's goal. The default evaluates them one
    /// by one. An evaluator whose cost is dominated by the call itself (a learned model on an accelerator) overrides
    /// it together with batched().
    virtual void evaluate_batch(std::span<const StateView> states, std::span<Value> out)
    {
        for (usize i = 0; i < states.size(); ++i)
            out[i] = evaluate(states[i]);
    }
    /// Whether searches should hand states over in batches: eager A* and eager GBFS the new successors of each
    /// expansion, beam search those of each layer. False for the library's heuristics (one call per state).
    [[nodiscard]] virtual bool batched() const noexcept { return false; }

    /// FF: the ground actions of the relaxed plan of the last evaluation. An applicable action is a preferred operator
    /// when it is one of them (mimir's preferred actions).
    [[nodiscard]] virtual bool provides_preferred() const noexcept { return false; }
    [[nodiscard]] virtual bool preferred(const ActionLabel& /*action*/) const { return false; }
    [[nodiscard]] virtual std::vector<Action> relaxed_plan() const { return {}; }

    [[nodiscard]] const HeuristicStats& stats() const noexcept { return m_stats; }
    /// Long evaluations (the lifted fallback's enumeration) poll `stop` and throw Interrupted once it returns true.
    /// Searches set it to their deadline and cancellation token. Empty: never interrupted.
    void set_interrupt(std::function<bool()> stop) { m_interrupt = std::move(stop); }
    /// The grounding in use (nullptr for blind, goal count, lifted evaluation or a grounding beyond the budget).
    [[nodiscard]] virtual std::shared_ptr<const RelaxedTask> relaxed() const { return nullptr; }

protected:
    HeuristicStats m_stats;
    std::function<bool()> m_interrupt;
};

/// Throws std::invalid_argument for unsupported combinations (real costs on a task whose costs are not integral).
[[nodiscard]] std::unique_ptr<Heuristic> make_heuristic(const Task& task, const Options& options = {});
}  // namespace mymyr::heuristics
