#pragma once
// Internals shared by iw.cpp and siw.cpp: goal tests, the blocked-state set and the IW ladder.

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/plan.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::search::detail
{
using Clock = std::chrono::steady_clock;

/// When a popped state counts as a goal: the search's GoalSpec, or an SIW subproblem's goal counter.
class GoalTest
{
public:
    enum class Kind : u8
    {
        Task,
        AnyOf,
        Custom,
        Counter,  // fewer unsatisfied task goal literals than `threshold`
    };

    static GoalTest from_spec(const Task& task, const GoalSpec& spec);
    /// The SIW subproblem goal: count(state) < threshold, over the task's goal literals (each once).
    static GoalTest counter(const Task& task, u32 threshold);

    [[nodiscard]] bool task_goal() const noexcept { return m_kind == Kind::Task; }

    /// Whether the test reads derived atoms: the caller prepares the state (axioms) first.
    [[nodiscard]] bool needs_view() const noexcept { return m_view; }
    /// Whether test() may run on several threads at once (each with its own Successors): every kind but Custom,
    /// whose user function runs on the calling thread.
    [[nodiscard]] bool concurrent() const noexcept { return m_kind != Kind::Custom; }
    /// The task's goal contains a false static literal. Custom and AnyOf
    /// goals are never statically false.
    [[nodiscard]] bool statically_false() const noexcept { return m_static_false; }

    /// Goal test of state s. `prepared`: succ.prepare() already ran on it.
    [[nodiscard]] bool test(Successors& succ, StateView s, bool prepared) const;
    /// Unsatisfied task goal literals (each distinct literal once; numeric goal constraints are not counted, ).
    [[nodiscard]] u32 unsatisfied(Successors& succ, StateView s, bool prepared) const;

private:
    void point(Successors& succ, StateView s, bool prepared) const;

    Kind m_kind = Kind::Task;
    const GoalSpec* m_spec = nullptr;
    std::vector<plan::Check> m_lits;  // Task and Counter: the task's goal literals, deduplicated
    u32 m_threshold = 0;
    bool m_view = false;
    bool m_static_false = false;
};

/// The plan cost: the metric value at the end of the plan, from the start state's metric value, through the shared cost evaluator
/// (heuristics::ActionCosts, heuristics::plan_metric): total-cost effects of any operator and expression, conditional
/// ones when they fire, numeric metrics; without total-cost and metric every step costs 1. Between two consecutive
/// states of the plan the cheapest action is taken, so apply() replaces an action by a strictly cheaper one with
/// the same successor.
class PlanCost
{
public:
    explicit PlanCost(const Task& task) : m_costs(task) {}
    [[nodiscard]] bool has_action_costs() const noexcept { return !m_costs.unit(); }
    /// The metric value of `start`: the cost of the empty plan from it.
    [[nodiscard]] double initial(const State& start) const { return m_costs.initial(start.view()); }
    /// The cost of `plan` from `start`; may rewrite actions (see above).
    double apply(Successors& succ, const State& start, std::vector<Action>& plan) const
    {
        return apply(succ, start, initial(start), plan);
    }
    /// The metric value after `plan` from `start`, whose metric value is g0 (a plan continued from a state reached
    /// by an earlier one); may rewrite actions (see above).
    double apply(Successors& succ, const State& start, double g0, std::vector<Action>& plan) const
    {
        return heuristics::plan_metric(succ, m_costs, start, g0, plan);
    }

private:
    heuristics::ActionCosts m_costs;
};

/// blocked_states, compared by content.
class BlockedSet
{
public:
    explicit BlockedSet(const std::vector<State>& states);
    [[nodiscard]] bool empty() const noexcept { return m_index.empty(); }
    [[nodiscard]] bool contains(StateView s) const;

private:
    const std::vector<State>* m_states = nullptr;
    std::vector<std::pair<u64, u32>> m_index;  // (hash, index), sorted
};

/// Everything a ladder needs besides the start state and the goal.
class LayerOrderer;  // layer_order_detail.hpp
class BeamTeam;      // beam_detail.hpp

struct Context
{
    const Task& task;
    const IwOptions& options;
    Successors& succ;
    BlockedSet blocked;
    Clock::time_point deadline;
    bool timed = false;
    SearchObserver* observer = nullptr;
    std::unique_ptr<LayerOrderer> layers;  // IwOptions::layers when ordered, else null
    /// The members of the beam's layer step (beam_detail.hpp): with threads > 1 or a relaxed beam, on the fast path.
    std::unique_ptr<BeamTeam> team;

    Context(const Task& t, const IwOptions& o, Successors& s);
    ~Context();
    [[nodiscard]] bool out_of_time() const { return timed && Clock::now() >= deadline; }
};

/// The ladder: arities 0..max_width from `root` (optimized IW(1) when max_width == 1 and options.optimize_iw1),
/// or the single pass `only_arity` when set. Fills status, plan, goal_state, passes, total, effective_width,
/// message, peak bytes; not cost, fluent_slots, nor the start/solution/end events.
IwResult run_ladder(Context& c, StateView root, const GoalTest& goal, u32 max_width, int only_arity = -1);

void add_pass(SearchStatistics& total, const IwPassStatistics& p);

/// Zero-extend completed pass rows to the task's final lazy-slot width.
void normalize_trees(std::span<IwPassStatistics> passes, u32 words);

/// The width-1 pass under IwOptions::transition_ordering, layer by layer (iw_ordered.cpp; search/transition_ordering.hpp).
struct OrderedPassOut
{
    IwPassStatistics st;
    std::vector<Action> plan;
    std::optional<State> goal_state;
    u64 table_bytes = 0;
    u64 node_bytes = 0;
};
void run_ordered_pass(Context& c, StateView root, const GoalTest& goal, bool root_continuation, OrderedPassOut& out);
}  // namespace mymyr::search::detail
