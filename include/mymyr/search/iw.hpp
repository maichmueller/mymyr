#pragma once
// Iterated width: the IW(k) ladder, optimized IW(1) and single IW passes. SIW is in search/siw.hpp.
//
//   auto task = mymyr::Task::from_text_file("p.txt");
//   mymyr::search::IwOptions o;
//   o.max_arity = 2;
//   mymyr::search::IwResult r = mymyr::search::iw(*task, o);
//
// Follows mimir 0.16.3's `iw::find_solution` conventions, which runs BrFS with a novelty pruning strategy per arity:
//   - the ladder runs one pass per arity 0, 1, ..., max_arity, each from the start state with fresh novelty tables,
//     until a pass solves the task or stops on a budget; if every pass exhausts its tree, the status is Exhausted
//     (mimir calls this FAILED);
//   - within a pass, states are expanded breadth first in the order they were admitted, successors in the successor
//     generator's order (canonical order by default); the goal is tested when a state is popped, so a goal
//     state is not counted as expanded;
//   - a successor enters the tree iff it is novel (novelty/novelty_table.hpp): some tuple of size <= k containing an
//     atom the transition added is unseen. There is no duplicate-detection table: a novel successor is
//     necessarily a new state. Non-novel successors are never materialized in IW(1) (add-effect precheck);
//   - width 0 (WidthZero::ExpandDepthOne, mimir's default): every successor of the start state other than the
//     start state itself enters the tree (duplicates count once per action in generated_in_tree, as in mimir,
//     but are expanded once), and every such depth-1 state is expanded with all of its successors pruned. The
//     alternative WidthZero::RootOnly (C#) goal-tests the depth-1 states without expanding them;
//   - optimized IW(1) (`optimize_iw1`, mimir's default, used when max_arity == 1): the width-0 pass is replaced
//     by an empty placeholder entry (arity 0, all zeros), and the width-1 pass admits every distinct successor of the
//     start state; those that are not novel are goal-tested and counted as expanded but not expanded
//     (IwPassStatistics::skipped counts them; expanded - skipped is the classical count);
//   - budgets: max_seconds spans the whole ladder, max_states (states in the tree, the root included) and
//     max_expanded apply per pass, max_depth bounds the depth of expanded states (deeper ones are popped,
//     goal-tested and counted, not expanded: mimir's max_depth). blocked_states are never entered: they are
//     checked before the novelty test and mark no tuple (mimir's semantics). The goal, the blocked states and
//     the observer are forwarded to every pass;
//   - statistics per pass follow mimir's brfs::Statistics: expanded, generated (every applicable action of every
//     expanded state; with witness pruning off, which is the default here, this equals mimir's count),
//     generated_in_tree (successors admitted).
// Numeric fluents: states carry their numeric values and the goal's numeric constraints are tested; novelty
// looks at the atoms only. Plan cost is mimir's (heuristics/action_costs.hpp): the metric value of the plan's
// last state, where between two consecutive plan states the action giving the lowest metric value is taken (the plan
// is rewritten to it). With total-cost: the problem's initial total-cost plus every fired total-cost effect (any
// operator, conditional ones, expressions over static and fluent functions); else with a metric: the metric on the
// last state; else the plan length.
//
// Observer events (search/control.hpp): on_start once; per pass on_expand for every counted expansion, on_generate
// for every transition (child id = the new node id if admitted, else ~0; the child state is materialized only when
// an observer is set) and on_prune for every transition not admitted, then on_pass(arity, pass statistics);
// on_solution and on_end once. Ids are node indices of the pass (the root is 0).

#include "mymyr/novelty/novelty_table.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/successor/symmetry.hpp"

#include <functional>
#include <memory>
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
class TransitionOrdering;  // search/transition_ordering.hpp

/// How the width-0 pass of the IW ladder treats the successors of the start state.
enum class WidthZero : u8
{
    ExpandDepthOne,  // mimir (ArityZeroNoveltyPruningStrategy): they enter the tree and are expanded
    RootOnly,        // C# (IwSearch.SearchK0): they are goal-tested but not expanded
};

struct IwOptions
{
    SearchControl control;
    u32 max_arity = 2;  // the ladder runs arities 0..max_arity (at most novelty::k_max_arity)
    WidthZero width_zero = WidthZero::ExpandDepthOne;
    bool optimize_iw1 = true;       // mimir's optimized IW(1) when max_arity == 1 (see above)
    bool witness_pruning = false;   // off: every applicable action is a transition (mimir's generated counts)
    bool canonical_order = true;    // successors per state in (schema, binding) order
    /// Wl1: successors only by actions over representatives of the objects' colour classes (successor/symmetry.hpp);
    /// the search may then miss every plan.
    SymmetryPruning symmetry_pruning = SymmetryPruning::Off;
    novelty::TableOptions tables;   // dense/sparse budget of the novelty tables
    std::optional<State> start;     // default: the task's initial state
    /// Optional successor order. Called once per expanded state with its actions in generation order; writes into
    /// `order` the indices in the order the transitions are to be tested (invalid and repeated indices are dropped,
    /// missing ones follow in generation order). Setting it runs the observed (slow) path, which buffers every
    /// transition. The parity gate uses it to replay mimir's successor order.
    std::function<void(StateView state, std::span<const Action> actions, std::vector<u32>& order)> successor_order;
    /// Optional layer transition ordering of the width-1 pass (search/transition_ordering.hpp; e.g.
    /// landmarks::LandmarkTransitionOrdering): the pass then generates a whole layer, orders it and decides novelty
    /// in that order. Other passes are unaffected.
    std::shared_ptr<const TransitionOrdering> transition_ordering;
    /// The order in which every pass expands a layer (search/layer_ordering.hpp; default: the queued BrFS). An
    /// ordered kind cannot be combined with transition_ordering.
    LayerOrdering layers{};
};

struct IwPassStatistics
{
    u32 arity = 0;
    SearchStatus status = SearchStatus::Exhausted;
    u64 expanded = 0;           // popped non-goal states (mimir's count; includes skipped states)
    u64 generated = 0;          // transitions generated from expanded states
    u64 generated_in_tree = 0;  // successors admitted to the tree
    u64 skipped = 0;            // counted as expanded, successors not generated (optimized IW(1), RootOnly, max_depth)
    u64 blocked = 0;            // successors rejected because they are blocked
    double seconds = 0;
    bool placeholder = false;   // the empty arity-0 entry of an optimized IW(1) ladder

    [[nodiscard]] SearchStatistics statistics() const
    {
        return {.expanded = expanded,
                .generated = generated,
                .states = generated_in_tree + (placeholder ? 0 : 1),
                .pruned = generated - generated_in_tree,
                .seconds = seconds};
    }
};

struct IwResult
{
    SearchStatus status = SearchStatus::Exhausted;
    std::vector<Action> plan;
    double cost = 0;                 // mimir's plan cost (see above); 0 unless solved
    bool cost_exact = true;          // always true (every total-cost effect is evaluated)
    std::optional<State> goal_state;
    std::vector<IwPassStatistics> passes;  // one per arity run, index = arity
    SearchStatistics total;                // sums over the passes
    u32 effective_width = 0;               // arity of the pass that solved the task
    std::string message;                   // why the search could not run (status Failed)
    u32 fluent_slots = 0;                  // assigned fluent atom slots at the end
    u64 peak_table_bytes = 0;              // largest novelty table of a pass
    u64 peak_node_bytes = 0;               // largest tree (state words and plan records) of a pass
};

/// The IW ladder: arities 0..max_arity (optimized IW(1) when max_arity == 1 and optimize_iw1).
[[nodiscard]] IwResult iw(const Task& task, const IwOptions& options = {});

/// One IW(k) pass alone (arity 0..k_max_arity), without the ladder and without the optimized-IW(1) root
/// continuation: the width-k pass of a ladder, in isolation.
[[nodiscard]] IwResult iw_pass(const Task& task, u32 arity, const IwOptions& options = {});

/// mimir's name of an IW status ("solved", "failed" for an exhausted ladder, "out_of_time", ...).
[[nodiscard]] const char* mimir_status_name(SearchStatus s) noexcept;
}  // namespace mymyr::search
