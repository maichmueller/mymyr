#pragma once
// Heuristic best-first searches: A* (weighted A* with a weight) and GBFS, each eager and lazy, and beam search.
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::search::BestFirstOptions o;
//   o.heuristic.kind = mymyr::heuristics::Kind::Max;
//   mymyr::search::BestFirstResult r = mymyr::search::astar_eager(*task, o);
//
// Semantics:
//   - astar_eager: pops the minimum f = g + h; a goal state is recognized when generated (new states only) and
//     returned when it is the minimum (goal states first among equal f); successors reached
//     more cheaply are reopened; h = +inf marks a dead end. Ties among equal f: lower h first, then first in, first
//     out. A goal state that is reopened stays a goal
//     state, and h is cached per state instead of being recomputed on reopening.
//   - weighted A* (BestFirstOptions::weight = w > 1, eager and lazy): the same searches with f = g + w * h in place of
//     g + h (w = 1 is A*, with the same expansions). Ties as A*: lower h first, then first in, first out; goal states
//     first among equal f. Suboptimality bound, for non-negative action costs and plan costs counted from the start
//     state's metric value: eager weighted A* returns a plan of cost <= w * C* (C*: the optimal cost) with an
//     admissible heuristic when it reopens states (`reopen`, the default), and with a consistent heuristic (blind,
//     h_max) also without reopening; lazy weighted A* with `lazy_requeue` for consistent heuristics. The bound needs
//     a heuristic of the task's goal (not GoalSpec::Custom) and h = +inf only for dead ends.
//   - astar_lazy: children enter the open list unevaluated; a state is evaluated when popped. With `lazy_requeue`
//     (default) an unevaluated child is keyed by the lower bound g(child) + max(0, h(parent) - cost) and, once
//     evaluated, re-queued under its true f when that is larger; a goal is accepted only when no open entry has a
//     smaller key. This keeps the plan optimal for consistent heuristics (blind, h_max) even with the preferred and
//     standard lists alternating. `lazy_requeue = false` is the plain lazy search: children keyed g(child) + h(parent),
//     goal returned when popped, which can return a suboptimal plan.
//   - gbfs_eager: key (h, g, first in first out), no reopening, goal test when a state is generated.
//   - gbfs_lazy: children keyed (h(parent), g(child), FIFO) in a preferred list (the action is in the parent's FF
//     relaxed plan) and a standard list, popped alternately with weights preferred_weight : standard_weight (64 : 1
//     by default); the popped state is evaluated, dead ends dropped; goal test when generated.
//   - beam: layer-synchronous. Every state of the current layer is expanded; its new successors are goal-tested when
//     generated, evaluated, and the best `beam_width` of them by (h, g, generation order) form the next layer. Beam
//     search is incomplete: Exhausted means the beam ran dry, not that the task is unsolvable. (A beam over a BrFS/IW with layer
//     orderings and novelty modes belongs to the IW family.)
// All searches: successors in canonical order, every applicable action a transition (witness pruning off);
// g is the metric value of a state (heuristics::ActionCosts): the problem's initial total-cost with the
// total-cost effects of each action applied in turn, the metric over the state's numeric values for a metric without
// total-cost, otherwise the number of actions; the plan cost is g of the goal state, and for GBFS and beam each step
// is rewritten to the cheapest action between its two states. A goal with a false static
// literal gives Unsolvable before the search starts; a dead-end start state gives Unsolvable too.
// Numeric tasks: states carry their numeric values (two states with the same atoms and different values are
// distinct), numeric preconditions and goal constraints are checked, and the relaxation heuristics ignore numerics
// (heuristics/heuristic.hpp).
// Batched evaluators (Heuristic::batched): eager A* and eager GBFS goal-test the new successors of an expansion and
// evaluate them in one call (GBFS up to the first goal state); beam search evaluates the new successors of a whole
// layer in one call. The searches are otherwise unchanged.
//
// Batched expansions (BestFirstOptions::expansion_batch = k > 1, eager A* and eager GBFS): every round pops up to k
// open states in key order and expands them, then goal-tests the new successors of all k expansions and evaluates
// them in one call of the heuristic (Heuristic::evaluate_batch; GBFS up to the first goal state), then handles the
// transitions expansion by expansion as the one-expansion search does (pushes, reopening, dead ends, observer
// on_generate). A* accepts a goal state only before a round's first pop (the round's successors are pushed first). The
// states of a round are popped before their predecessors' successors in the round are known, so the search expands
// up to k - 1 states per round that the one-expansion search would not have expanded (yet), and node order, plans of
// equal cost, `expanded`, `generated` and `evaluations` differ from k = 1. Guarantees with k > 1: A* returns an
// optimal plan with an admissible heuristic when it reopens states (`reopen`, the default; a state expanded in a round
// before its cheapest path was known is reopened when that path is found), but no longer with a consistent heuristic
// without reopening; weighted A*'s bound w * C* holds under the same condition; GBFS stays complete on finite spaces.
// Ties are broken as for k = 1 among the states in the open list when a round starts. Budgets are checked before
// every pop (max_expanded is exact); a stop inside a round drops the round's unevaluated successors (the observer still
// gets on_generate for them). Dead ends, the goal kinds and the queues are those of k = 1. Path-dependent heuristics
// are refused (std::invalid_argument). k = 1 is the one-expansion search.
//
// Cost bound (BestFirstOptions::cost_bound, A* only): a plan costs less than the bound. A state whose g is at least
// the bound is not opened, and with an admissible heuristic (Heuristic::admissible) and a goal of the task or AnyOf, a
// state whose g + h is at least the bound neither (the lazy searches test g + h when they evaluate a state); a start
// state beyond the bound ends the search at once. Exhausted then means that the search space holds no plan cheaper
// than the bound. Sound for non-negative action costs: a transition that lowers g makes the search fail.
//
// Negative costs: A* (eager, lazy, weighted, and the anytime search built on it) needs non-negative action costs, as
// its termination with reopening and its optimality rest on them (a cycle of negative cost would reopen states
// forever). A constant negative cost fails every best-first search before it starts (heuristics::ActionCosts); a
// cost that depends on the binding or the state is known only on a transition, and A* fails (status Failed, the
// message names the two metric values) at the first transition that lowers g. GBFS, beam search and enforced
// hill-climbing order by h and close every state once, so they terminate with such costs and report the plan's
// metric value as its cost.
//
// SearchControl: budget.max_states bounds the stored states, max_expanded the expansions (both OutOfStates),
// max_depth the depth (in actions) of expanded states (deeper states are not expanded), max_seconds the wall time
// (OutOfTime); cancel and on_progress returning false give Cancelled. blocked_states are never entered (the start
// state is exempt); they count as pruned. The goal: GoalSpec::Task, AnyOf (the heuristic estimates
// the cheapest of the goals) or Custom (the heuristic estimates the task's goal, which the search does not test: its
// values only order the states; h = +inf prunes nothing and never gives Unsolvable, the states with it come last,
// and A* gives no optimality guarantee). Under Task and AnyOf h = +inf proves a dead end. Observer events: on_start,
// on_expand(id), on_generate(parent, action, child id, child, is_new) for every generated transition (a blocked
// successor: child ~0, is_new false, then on_prune), on_solution and on_end. Ids are the store's state ids (the start
// state is 0).
//
// Storage: search nodes are SoA arrays keyed by state id (parent, action, g, h, status); states live in the same
// stores as BrFS (Auto: Flat up to 8 words, Chunked above, as BrFS; Compact on request: closed states as 128-bit fingerprints and
// open states' words in the open list, for searches with small open lists). Queues: bucket queues when costs and
// heuristic values are small integers (Auto: integral constant costs of at most 64), binary heaps otherwise. Both pop
// in the same order (smallest key, then the tie-breaking key, then first in first out), so the choice never changes a
// result; an Auto bucket queue that meets a key it cannot bucket (a negative cost, a fractional heuristic value, a
// depth beyond 2^22) moves into a heap, and result.queue then says "heap".

#include "mymyr/core/types.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <limits>
#include <optional>
#include <stdexcept>
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
        Auto,    // Bucket for small integral costs (integral, constant per schema, at most 64), else Heap; a bucket
                 // queue that meets a key it cannot bucket moves its entries into a heap (the order of pops is the
                 // heap's throughout)
        Bucket,  // requires small non-negative integral keys (g, h, f); BucketKeyError otherwise
        Heap,
    };

    SearchControl control;
    heuristics::Options heuristic;                 // the heuristic to build (kind, costs, evaluation, grounding)
    heuristics::Heuristic* evaluator = nullptr;    // not owned: use this heuristic (e.g. Kind::Custom) instead of building one
    Store store = Store::Auto;
    Queue queue = Queue::Auto;
    bool witness_pruning = false;  // off: every applicable action is a transition
    u32 expansion_batch = 1;       // eager A* and GBFS: expansions per round of batched evaluation (see above)
    bool canonical_order = true;   // successors per state in (schema, binding) order
    std::optional<State> start;    // default: the task's initial state

    // lazy searches
    bool preferred_operators = true;  // alternate a preferred list when the heuristic provides preferred operators
    u32 preferred_weight = 0;         // pops from the preferred list per turn; 0: the default (A* lazy 1, GBFS lazy 64)
    u32 standard_weight = 1;          // pops from the standard list per turn
    // A*
    bool reopen = true;               // reopen states reached with a smaller g
    bool lazy_requeue = true;         // astar_lazy: lower-bound keys and re-queueing (optimal); false: parent-keyed
    double weight = 1;                // f = g + weight * h: finite and at least 1 (1: A*); the other searches take 1 only
    double cost_bound = std::numeric_limits<double>::infinity();  // plans cost less (see above); not NaN; A* only
    // beam
    u32 beam_width = 1000;  // positive (std::invalid_argument otherwise)
};
/// Thrown by a search with Queue::Bucket when a key of its open list (g, h, f, or GBFS's tie-breaking g) is not an
/// integer in [0, 2^22): a negative action cost, a fractional or infinite heuristic value, or a very deep search.
class BucketKeyError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

// Option values a search does not take (a weight below 1 or not finite, a weight other than 1 or a finite cost bound
// in GBFS or beam search, a NaN cost bound, a bucket queue with a weight that is not integral, an expansion_batch of 0
// or one above 1 outside eager A* and eager GBFS) throw std::invalid_argument before the search starts.

struct BestFirstResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;
    double cost = 0;                     // g of the goal state; 0 unless solved
    std::optional<State> goal_state;
    SearchStatistics stats;  // expanded, generated (transitions), states (stored), pruned, seconds, memory_bytes (states,
                             // search nodes and open lists at the end)
    u64 evaluations = 0;                 // heuristic evaluations by the search
    u64 dead_ends = 0;                   // states with h = +inf
    u64 reopened = 0;                    // A*: states whose g decreased after they were opened
    u64 expanded_until_last_jump = 0;    // A*: expansions before the f-value of the last node popped was first reached
                                         // (the expansions below the final f-layer; Fast Downward's statistic)
    u64 layers = 0;                      // beam: layers expanded
    double initial_h = 0;                // h of the start state
    heuristics::HeuristicStats heuristic;  // the evaluator's statistics at the end
    double setup_seconds = 0;            // building the heuristic (grounding) and the action costs
    std::string algorithm, store, queue;
    std::string message;                 // why the search could not run (status Failed)
    u32 fluent_slots = 0;                // assigned fluent atom slots at the end
};

[[nodiscard]] BestFirstResult astar_eager(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult astar_lazy(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult gbfs_eager(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult gbfs_lazy(const Task& task, const BestFirstOptions& options = {});
[[nodiscard]] BestFirstResult beam(const Task& task, const BestFirstOptions& options = {});
}  // namespace mymyr::search
