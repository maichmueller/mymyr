#pragma once
// Rollout IW(1) (Bandres, Bonet and Geffner, "Planning With Pixels in (Almost) Real Time", AAAI 2018), matching
// mimir 0.16.3's rollout_iw::find_solution.
//
//   mymyr::search::RolloutIwOptions o;
//   o.ordering = mymyr::search::ActionOrdering::DirectGoalAchieverFirst;
//   mymyr::search::RolloutIwResult r = mymyr::search::rollout_iw(*task, o);
//
// A tree over action sequences (two nodes may hold the same state), with the minimum depth at which each fluent atom
// has been seen (the root's atoms at depth 0). The root is goal-tested first. Each rollout descends from the root:
//   - a node's applicable actions are materialized once, all of them (statistics.expanded); none: a dead end, SOLVED;
//   - the first action in ranked order (ActionOrdering) whose child is not SOLVED is taken; none: the node is SOLVED;
//   - an existing child still at the best depth of one of its atoms is walked into (case 4), otherwise it is SOLVED
//     (case 3);
//   - a new child (statistics.generated) is goal-tested first (a goal ends the search), then counted against
//     Budget::max_states, cut at Budget::max_depth (SOLVED), cut when depth + 1 reaches the incumbent bound (SOLVED),
//     then registered: if it lowers the best depth of an atom (all of them are updated) the rollout continues (case 1),
//     else it is SOLVED (case 2);
//   - SOLVED propagates to a parent whose actions are materialized and whose children all exist and are SOLVED.
// Rollouts run until a goal is found, the root is SOLVED (Exhausted: the width-1 space is exhausted, relative to the
// depth and incumbent bounds when those cut anything) or a budget stops the search: max_seconds (OutOfTime), cancel or
// SearchCoordination::cancel (Cancelled), max_states (OutOfStates), max_rollouts (Failed, mimir's FAILED "rollout
// budget expired"). With SearchControl::coordination the incumbent bound is also the published incumbent length and
// every materialization adds one expansion.
//
// Orderings (guidance, never pruning; stable sorts keep the generation order among equals):
//   - InOrder: generation order (canonical; or SuccessorOrder);
//   - Randomized: a SplitMix64 shuffle (core/random.hpp; mimir uses std::shuffle with std::mt19937_64, which is
//     not portable), drawn anew at every visit of a node as mimir does, the generator persisting across the
//     search (so a visit costs draws for all of the node's actions);
//   - DirectGoalAchieverFirst: actions adding a positive fluent goal atom first. mimir inspects every conditional
//     effect of the ground action whether or not its condition holds; here the adds of effects that fire, and the adds
//     of conditional effects without forall parameters whatever their condition, are inspected (a forall effect whose
//     condition does not hold is not);
//   - GoalRegressionRelevance: by the schema's rank in a regression over predicates (goal predicates 0; a predicate in
//     the fluent precondition of a schema adding a rank-r predicate gets r + 1; a schema ranks as its best add);
//   - MixedRegressionRandom: a shuffle, then a stable sort by the regression rank.
// The goal atoms of the goal-directed orderings: the task's goal (GoalSpec::Kind::Task and Custom) or the union of the
// positive atoms of GoalSpec::goals (AnyOf).
//
// blocked_states: a blocked successor becomes a SOLVED child, never goal-tested nor registered. Observer events:
// on_start; on_expand per materialized node; per new child on_generate (is_new = true), on_transition (Goal, Opened for
// case 1, Pruned otherwise) and on_prune when it is not walked into; on_progress every progress_interval
// materializations; on_solution and on_end. Node ids are tree node indices (the root is 0).
//
// The plan is normalized as in the other IW searches (search/iw.hpp): between two consecutive plan states the cheapest
// action is taken, and the cost is mimir's metric along it (it can differ from mimir's rollout plan only when
// two actions with different costs have the same successor).

#include "mymyr/search/control.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/successor/symmetry.hpp"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::search
{
enum class ActionOrdering : u8
{
    InOrder,
    Randomized,
    DirectGoalAchieverFirst,
    GoalRegressionRelevance,
    MixedRegressionRandom,
};

[[nodiscard]] const char* to_string(ActionOrdering o) noexcept;

struct RolloutIwOptions
{
    /// budget.max_states: generated states (mimir's max_num_states); max_depth; max_seconds; max_expanded is not
    /// used. cancel, goal, blocked_states, observer, coordination as documented above.
    SearchControl control;
    ActionOrdering ordering = ActionOrdering::InOrder;
    u64 seed = 0;                  // Randomized and MixedRegressionRandom
    u64 max_rollouts = ~u64{0};
    u32 incumbent_bound = ~u32{0};  // a plan of this length is known elsewhere
    bool canonical_order = true;
    SymmetryPruning symmetry_pruning = SymmetryPruning::Off;  // as IwOptions::symmetry_pruning
    std::optional<State> start;
    /// As IwOptions::successor_order, applied when a node's actions are materialized: its result is the node's
    /// generation order (the parity gate replays mimir's generator order with it).
    std::function<void(StateView state, std::span<const Action> actions, std::vector<u32>& order)> successor_order;
};

struct RolloutIwStatistics
{
    u64 rollouts = 0;
    u64 generated = 0;  // new tree nodes
    u64 expanded = 0;   // nodes whose actions were materialized
    u64 feature_depth_improvements = 0;
    u64 case1 = 0, case2 = 0, case3 = 0, case4 = 0;
    u64 solved_propagations = 0;
    u64 dead_ends = 0;
    u64 depth_bound_prunings = 0;
    u64 incumbent_bound_prunings = 0;
    u64 blocked = 0;
    u32 max_rollout_depth = 0;
    u64 tree_nodes = 0;
    double seconds = 0;

    [[nodiscard]] SearchStatistics statistics() const
    {
        return {.expanded = expanded, .generated = generated, .states = tree_nodes, .pruned = case2 + depth_bound_prunings + incumbent_bound_prunings + blocked, .seconds = seconds};
    }
};

struct RolloutIwResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;
    double cost = 0;
    bool cost_exact = true;
    std::optional<State> goal_state;
    RolloutIwStatistics statistics;
    bool root_solved = false;  // the reachable width-1 space was exhausted (relative to the bounds, see the counters)
    std::string message;       // why the search stopped when not solved (mimir's stop_reason)
    u32 fluent_slots = 0;
};

[[nodiscard]] RolloutIwResult rollout_iw(const Task& task, const RolloutIwOptions& options = {});
}  // namespace mymyr::search
