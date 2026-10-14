#pragma once
// Enforced hill-climbing (Hoffmann and Nebel 2001).
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::search::EhcOptions o;  // h_FF, helpful actions first, greedy best-first search when it fails
//   mymyr::search::EhcResult r = mymyr::search::ehc(*task, o);
//
// From the current state (the start state first) a breadth-first lookahead searches for a state with a strictly
// smaller h; the search commits to the first one it finds, appends the path to it to the plan and repeats from there.
// The lookahead pops states first in, first out and evaluates a state when it pops it (the heuristic's preferred
// operators are then those of that state): a state with h = +inf is a dead end and is dropped, a state with a smaller
// h than the current one ends the lookahead, any other state is expanded. A goal state ends the search when it is
// generated. Within one lookahead a state is entered once (by the first path that reaches it, in generation order); a
// later lookahead may enter it again. Each commitment lowers h, so the search ends.
//
// Helpful actions (helpful_actions): the lookahead expands a state with the applicable actions that are preferred
// operators of its evaluation (h_FF and set-additive: the actions of the relaxed plan; landmark count: the actions
// that achieve a needed landmark), and skips the others (they are not generated):
//   - Auto (the default): First when the heuristic gives preferred operators, Off otherwise;
//   - First: helpful actions only; a lookahead that finds no better state is repeated with every action;
//   - Only: helpful actions only; a lookahead that finds no better state ends hill-climbing;
//   - Off: every applicable action.
// First and Only with a heuristic without preferred operators fail (status Failed with a message).
//
// When hill-climbing fails (a lookahead finds no better state: a dead end, or a plateau it cannot leave), `fallback`
// decides: Gbfs (the default, as FF does) runs eager greedy best-first search (best_first.hpp: gbfs_eager) from the
// start state with the same heuristic, control and store, and its result is the search's (plan, status); None returns
// status Exhausted, which means that hill-climbing failed, not that the task is unsolvable (as for beam search).
//
// Costs: the lookahead counts actions, not costs (breadth-first); g and the plan cost are the task's metric, and each
// plan step is the cheapest action between its two states (as GBFS).
//
// Path-dependent heuristics (landmark count): the status of a state is progressed along the path by which the
// lookahead reached it, from the status of the current state, without merging other paths: the committed state's
// status is the status of the plan up to it. A state reached again in a later lookahead gets the status of that path.
//
// SearchControl: budget.max_states bounds the stored states (every state of every lookahead: they share one store;
// the fallback search has its own), max_expanded the expansions (hill-climbing and fallback together), max_depth the
// plan length of an expanded state, max_seconds the whole call (fallback included); cancel and on_progress returning
// false give Cancelled. blocked_states are never entered. The goal: GoalSpec::Task, AnyOf or Custom, as for the
// best-first searches (best_first.hpp), with the same requirements on the heuristic. Observer events: on_start,
// on_expand(id), on_generate(parent, action, child id, child, is_new) for every transition the lookahead generates
// (is_new: not entered by this lookahead before), on_prune for blocked successors, on_pass(lookahead, statistics of
// that lookahead) after each lookahead, then the fallback's on_expand, on_generate, on_prune and on_progress, and
// on_solution and on_end once. Ids are the store's state ids (the start state is 0; the fallback's ids are its own).
//
// Thread-safety: as the best-first searches (one heuristic per search thread; the task may be shared).

#include "mymyr/search/best_first.hpp"

#include <optional>

namespace mymyr::search
{
struct EhcOptions
{
    enum class Helpful : u8
    {
        Auto,   // First when the heuristic gives preferred operators, else Off
        Off,    // every applicable action
        Only,   // preferred operators only
        First,  // preferred operators only; a failed lookahead is repeated with every action
    };
    enum class Fallback : u8
    {
        Gbfs,  // eager greedy best-first search from the start state when hill-climbing fails
        None,  // status Exhausted when hill-climbing fails
    };

    SearchControl control;
    heuristics::Options heuristic;               // the heuristic to build (default: h_FF)
    heuristics::Heuristic* evaluator = nullptr;  // not owned: use this heuristic instead of building one
    BestFirstOptions::Store store = BestFirstOptions::Store::Auto;
    std::optional<State> start;    // default: the task's initial state
    bool witness_pruning = false;  // off: every applicable action is a transition
    bool canonical_order = true;   // successors per state in (schema, binding) order
    Helpful helpful_actions = Helpful::Auto;
    Fallback fallback = Fallback::Gbfs;
};

struct EhcStatistics
{
    u64 lookaheads = 0;    // breadth-first lookaheads (retries included)
    u64 improvements = 0;  // lookaheads that reached a state with a smaller h
    u64 retries = 0;       // lookaheads repeated with every action after the helpful actions found no better state
    u64 max_depth = 0;     // the longest path from a lookahead's start to its better state or the goal, in actions
    u64 expanded = 0;      // expansions of hill-climbing (the result's stats add the fallback's)
    u64 evaluations = 0;   // heuristic evaluations of hill-climbing
    bool failed = false;   // hill-climbing found no better state from some state
    bool fallback = false;  // the fallback search ran
};

struct EhcResult : BestFirstResult
{
    EhcStatistics ehc;
};

/// Enforced hill-climbing (see above). stats, evaluations and dead_ends cover hill-climbing and the fallback; store
/// and queue name the fallback's when it ran. Helpful actions (First, Only) with a heuristic without preferred
/// operators, and a heuristic that cannot run on the task, give status Failed with a message.
[[nodiscard]] EhcResult ehc(const Task& task, const EhcOptions& options = {});
}  // namespace mymyr::search
