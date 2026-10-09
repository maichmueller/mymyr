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
//   SetAdditive  mimir's set-additive heuristic (after Keyder and Geffner 2008): every proposition gets the achiever
//              set of its h_max best supporter (the first operator reaching its h_max cost, as for h_FF) plus the
//              member (supporter, proposition), mimir's unary action; an operator's set is the union of its
//              preconditions' sets, and an axiom adds no member of its own. h is the number of members of the union of
//              the goal's sets (unit costs) or the sum of the costs of their actions (real costs). An action that
//              supports two needed propositions is two members, so h >= h_FF for the same supporters; the relaxed plan
//              (preferred operators) is the set of the members' ground actions. Ties among equally cheap supporters
//              are broken as for h_FF.
//   H2         h² (Haslum and Geffner 2000): the cost of the most expensive pair of goal literals, from a fixpoint over the
//              costs of all pairs of propositions in which a pair is reached by an operator adding both, or adding one
//              while the other held before and is not deleted (mutexes count). Admissible, and h_max <= h² <= h*. Over the
//              same grounding as h_max: negative literals are the "false" propositions, axioms are 0-cost operators,
//              and a conditional effect is an operator of its own whose pairs also combine with the effects of the same
//              action (a pair added by two effects needs both conditions; only effects without a condition count as
//              deleting for sure). mimir's H2Heuristic differs where its encoding errs: it ignores axioms and effect
//              conditions, never reaches a negative literal the state does not satisfy, and its delete check excludes
//              the negation of a deleted atom instead of the atom. Grounded only: memory and time grow with the
//              square of the number of propositions (make_heuristic refuses more than 8191).
//   Perfect    h* from a state space (heuristics/perfect.hpp: heuristics::perfect); make_heuristic refuses it.
//   Custom     a Heuristic subclass of the caller (BestFirstOptions::evaluator), e.g. a learned value function.
// Numeric tasks. As mimir's delete relaxation (DeleteRelaxTranslator), the relaxation heuristics ignore numeric conditions
// (of actions, effects and the goal) and numeric effects: they see the task's atoms only. Goal count counts the
// fluent and derived goal literals; numeric goal constraints are not counted.
// Costs. Auto (the default): the task's objective, i.e. Unit for a task without action costs and metric, Real
// otherwise. Unit: every action costs 1 and every axiom 0 (mimir's heuristics). Real: the task's action costs as the
// relaxation sees them (ActionCosts::relaxed_cost, heuristics/action_costs.hpp): exact for integral and decimal costs
// (zero and fractional costs included), rounded down for others, and 0 for an action whose cost depends on the state
// (and for every action under a state metric), so h_max and h² stay admissible for non-negative costs. A heuristic
// with unit costs on a task whose actions cost less than 1 overestimates: A* is then not optimal.
// Evaluation. Grounded (the relaxed task is built on first use, within the budget) or lifted (a cost-bucketed semi-naive
// fixpoint over lifted matchers; the fallback beyond the budget and for states outside the grounded relaxation). h_max,
// h_add and h_FF have both; set-additive and h² are grounded only: make_heuristic throws std::invalid_argument for
// Evaluation::Lifted and std::runtime_error when the grounding exceeds the budget, and an evaluation throws
// std::runtime_error for a state outside the grounding (which only a state not reachable from the initial state is)
// and for a goal of evaluate(s, goals) with a negative literal over an atom that holds in s and that no operator or goal
// of the task uses negatively (the grounding has no proposition for its negation).
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
    SetAdditive,
    H2,
    Perfect,  // h* from a state space: heuristics::perfect (heuristics/perfect.hpp); make_heuristic refuses it
};

enum class Costs : u8
{
    Unit,  // every action 1, axioms 0 (mimir)
    Real,  // the task's action costs (ActionCosts::relaxed_cost)
    Auto,  // the task's objective: Unit for a task without action costs and metric, else Real
};

/// Costs::Auto resolved for `task` (Unit or Real); Unit and Real as they are. Throws std::invalid_argument for a
/// metric mymyr refuses (ActionCosts).
[[nodiscard]] Costs resolve_costs(const Task& task, Costs costs);

enum class Evaluation : u8
{
    Auto,      // grounded within the budget, lifted beyond it and for states the grounding does not cover
    Grounded,  // grounded; throws std::runtime_error if the grounding exceeds the budget
    Lifted,    // lifted only
};

[[nodiscard]] const char* to_string(Kind k) noexcept;
/// "blind", "goal_count" (or "gc"), "max" (or "hmax"), "add" (or "hadd"), "ff" (or "hff"), "set_additive" (or "hsa",
/// "setadd"), "h2". Throws std::invalid_argument for other names, including "perfect" (made by heuristics::perfect).
[[nodiscard]] Kind parse_kind(std::string_view name);

struct Options
{
    Kind kind = Kind::FF;
    Costs costs = Costs::Auto;
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

    /// FF and set-additive: the ground actions of the relaxed plan of the last evaluation (set-additive: those of the
    /// members of the union of the goal's achiever sets). An applicable action is a preferred operator when it is one
    /// of them (mimir's preferred actions).
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

/// Throws std::invalid_argument for unsupported combinations (real costs of 2^31 units or more, h² or set-additive
/// without a grounding) and for Kind::Custom and Kind::Perfect.
[[nodiscard]] std::unique_ptr<Heuristic> make_heuristic(const Task& task, const Options& options = {});
}  // namespace mymyr::heuristics
