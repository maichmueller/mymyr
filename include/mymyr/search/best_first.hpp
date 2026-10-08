#pragma once
// Heuristic best-first searches: A* and GBFS, each eager and lazy, and beam search.
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::search::BestFirstOptions o;
//   o.heuristic.kind = mymyr::heuristics::Kind::Max;
//   mymyr::search::BestFirstResult r = mymyr::search::astar_eager(*task, o);
//
// Semantics follow mimir 0.16.3 (src/search/algorithms/{astar,gbfs}_{eager,lazy}.cpp) except where noted:
//   - astar_eager: pops the minimum f = g + h; a goal state is recognized when generated (new states only) and
//     returned when it is the minimum (goal states first among equal f, mimir's (f, GOAL) key); successors reached
//     more cheaply are reopened; h = +inf marks a dead end. Ties among equal f: lower h first, then first in, first
//     out (mimir leaves them to std::priority_queue). Unlike mimir, a goal state that is reopened stays a goal
//     state, and h is cached per state instead of being recomputed on reopening.
//   - astar_lazy: children enter the open list unevaluated; a state is evaluated when popped. With `lazy_requeue`
//     (default) an unevaluated child is keyed by the lower bound g(child) + max(0, h(parent) - cost) and, once
//     evaluated, re-queued under its true f when that is larger; a goal is accepted only when no open entry has a
//     smaller key. This keeps the plan optimal for consistent heuristics (blind, h_max) even with the preferred and
//     standard lists alternating. `lazy_requeue = false` is mimir's search: children keyed g(child) + h(parent),
//     goal returned when popped, which can return a suboptimal plan.
//   - gbfs_eager: key (h, g, first in first out), no reopening, goal test when a state is generated.
//   - gbfs_lazy: children keyed (h(parent), g(child), FIFO) in a preferred list (the action is in the parent's FF
//     relaxed plan) and a standard list, popped alternately with weights preferred_weight : standard_weight (mimir's
//     {64, 1}); the popped state is evaluated, dead ends dropped; goal test when generated.
//   - beam: layer-synchronous. Every state of the current layer is expanded; its new successors are goal-tested when
//     generated, evaluated, and the best `beam_width` of them by (h, g, generation order) form the next layer. Beam
//     search is incomplete: Exhausted means the beam ran dry, not that the task is unsolvable. (mimir's beam is a
//     BrFS/IW with layer orderings and novelty modes; that variant belongs to the IW family.)
// All searches: successors in canonical order, every applicable action a transition (witness pruning off, mimir's
// counts); g is mimir's metric value of a state (heuristics::ActionCosts): the problem's initial total-cost with the
// total-cost effects of each action applied in turn, the metric over the state's numeric values for a metric without
// total-cost, otherwise the number of actions; the plan cost is g of the goal state, and for GBFS and beam each step
// is rewritten to the cheapest action between its two states (mimir's plan extraction). A goal with a false static
// literal gives Unsolvable before the search starts; a dead-end start state gives Unsolvable too (mimir's statuses).
// Numeric tasks: states carry their numeric values (two states with the same atoms and different values are
// distinct), numeric preconditions and goal constraints are checked, and the relaxation heuristics ignore numerics
// (heuristics/heuristic.hpp), all as in mimir.
// Batched evaluators (Heuristic::batched): eager A* and eager GBFS goal-test the new successors of an expansion and
// evaluate them in one call (GBFS up to the first goal state); beam search evaluates the new successors of a whole
// layer in one call. The searches are otherwise unchanged.
//
// SearchControl: budget.max_states bounds the stored states, max_expanded the expansions (both OutOfStates),
// max_depth the depth (in actions) of expanded states (deeper states are not expanded), max_seconds the wall time
// (OutOfTime); cancel and on_progress returning false give Cancelled. blocked_states are never entered (the start
// state is exempt, as in mimir); they count as pruned. The goal: GoalSpec::Task, AnyOf (the heuristic estimates
// the cheapest of the goals) or Custom (the heuristic estimates the task's goal). Observer events: on_start,
// on_expand(id), on_generate(parent, action, child id, child, is_new) for every transition that is not blocked,
// on_prune for blocked successors, on_solution and on_end. Ids are the store's state ids (the start state is 0).
//
// Storage: search nodes are SoA arrays keyed by state id (parent, action, g, h, status); states live in the same
// stores as BrFS (Auto: Flat up to 8 words, Chunked above, as BrFS; Compact on request: closed states as 128-bit fingerprints and
// open states' words in the open list, for searches with small open lists). Queues: bucket queues when costs and
// heuristic values are small integers (Auto: integral constant costs of at most 64), binary heaps otherwise.

#include "mymyr/core/types.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/successor/symmetry.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::search
{
struct BestFirstOptions
{
    enum class Store : u8
    {
        Auto,     // Flat when the estimated width is at most 8 words, Chunked above
        Flat,
        Chunked,
        Compact,  // closed states as fingerprints; open states keep their words in the open list
    };
    enum class Queue : u8
    {
        Auto,    // Bucket for small integral costs (integral, constant per schema, at most 64), else Heap
        Bucket,  // requires integral g and h values
        Heap,
    };

    SearchControl control;
    heuristics::Options heuristic;                 // the heuristic to build (kind, costs, evaluation, grounding)
    heuristics::Heuristic* evaluator = nullptr;    // not owned: use this heuristic (e.g. Kind::Custom) instead of building one
    Store store = Store::Auto;
    Queue queue = Queue::Auto;
    bool witness_pruning = false;  // off: every applicable action is a transition (mimir's counts)
    bool canonical_order = true;   // successors per state in (schema, binding) order
    /// Wl1: successors only by actions over representatives of the objects' colour classes (successor/symmetry.hpp);
    /// the search may then miss every plan, and A* may return a costlier one.
    SymmetryPruning symmetry_pruning = SymmetryPruning::Off;
    std::optional<State> start;    // default: the task's initial state

    // lazy searches
    bool preferred_operators = true;  // alternate a preferred list when the heuristic provides preferred operators
    u32 preferred_weight = 0;         // pops from the preferred list per turn; 0: mimir's (A* lazy 1, GBFS lazy 64)
    u32 standard_weight = 1;          // pops from the standard list per turn
    // A*
    bool reopen = true;               // reopen states reached with a smaller g (mimir does)
    bool lazy_requeue = true;         // astar_lazy: lower-bound keys and re-queueing (optimal); false: mimir's keys
    // beam
    u32 beam_width = 1000;
};

struct BestFirstResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;
    double cost = 0;                     // g of the goal state (mimir's plan cost); 0 unless solved
    std::optional<State> goal_state;
    SearchStatistics stats;              // expanded, generated (transitions), states (stored), pruned, seconds
    u64 evaluations = 0;                 // heuristic evaluations by the search
    u64 dead_ends = 0;                   // states with h = +inf
    u64 reopened = 0;                    // A*: states whose g decreased after they were opened
    u64 layers = 0;                      // beam: layers expanded
    double initial_h = 0;                // h of the start state
    heuristics::HeuristicStats heuristic;  // the evaluator's statistics at the end
    double setup_seconds = 0;            // building the heuristic (grounding) and the action costs
    std::string algorithm, store, queue;
    std::string message;                 // why the search could not run (status Failed)
    u64 store_bytes = 0;                 // states, search nodes and open lists at the end
    u32 fluent_slots = 0;                // assigned fluent atom slots at the end
};

[[nodiscard]] BestFirstResult astar_eager(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult astar_lazy(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult gbfs_eager(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult gbfs_lazy(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult beam(const Task& task, const BestFirstOptions& options = {});
}  // namespace mymyr::search
