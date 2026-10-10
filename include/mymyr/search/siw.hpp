#pragma once
// Serialized iterated width, matching mimir's siw::find_solution.
//
//   mymyr::search::SiwOptions o;
//   o.max_arity = 2;
//   mymyr::search::SiwResult r = mymyr::search::siw(*task, o);
//
// SIW runs the IW ladder (search/iw.hpp, with all of its conventions and options) repeatedly: each subproblem starts
// at the state the previous one reached and ends at the first state in which fewer of the task's goal literals are
// unsatisfied than at its start (mimir's ProblemGoalStrategyImplCounter: fluent and derived, positive and
// negative goal literals, each counted once; goals achieved earlier may be undone). It stops when the search goal
// (SearchControl::goal, the task goal by default) holds, or when a subproblem's ladder exhausts every pass
// (status Exhausted, mimir's FAILED) or stops on a budget. A solved SIW's plan is the concatenation of the subplans;
// an unsolved one has an empty plan, and partial_plan holds the subplans of the subproblems solved before the one
// that failed (the path to the state where that one started).
//
// Budgets: max_seconds spans the whole SIW; max_states, max_expanded and max_depth apply per IW pass, as in iw().
// blocked_states and the observer are forwarded to every pass of every subproblem (mimir's SIW has neither).
// Observer: on_start and on_end once, on_pass per IW pass, on_solution once with the whole plan.

#include "mymyr/search/iw.hpp"

namespace mymyr::search
{
/// SIW takes the IW options: max_arity is the width limit of every subproblem's ladder.
using SiwOptions = IwOptions;

struct SiwSubproblem
{
    SearchStatus status = SearchStatus::Exhausted;
    u32 unsatisfied_at_start = 0;  // goal literals unsatisfied in the subproblem's start state
    u32 plan_length = 0;
    u32 effective_width = 0;       // arity of the pass that solved it
    std::vector<IwPassStatistics> passes;
};

struct SiwResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;          // empty unless solved
    std::vector<Action> partial_plan;  // unless solved: the subplans of the solved subproblems (empty when solved)
    double cost = 0;  // the sum of the subplans' costs (search/iw.hpp), each from its start state (mimir's cost); 0 unless solved
    bool cost_exact = true;  // as in IwResult
    std::optional<State> goal_state;
    std::vector<SiwSubproblem> subproblems;  // including a last one that failed
    SearchStatistics total;                  // over every pass of every subproblem
    u32 max_effective_width = 0;             // over the solved subproblems
    std::string message;
    u32 fluent_slots = 0;
    u64 peak_table_bytes = 0;
    u64 peak_node_bytes = 0;
};

[[nodiscard]] SiwResult siw(const Task& task, const SiwOptions& options = {});
}  // namespace mymyr::search
