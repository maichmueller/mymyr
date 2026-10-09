#pragma once
// Search control shared by every search of the family: budgets, event handlers, goal strategies, blocked_states and
// cancellation apply uniformly across the family. Each search's options embed a SearchControl and its result
// carries a SearchStatus plus SearchStatistics. Extend by appending members; do not rename.
//
// mimir's convention: in an IW ladder max_seconds spans the whole ladder while max_states applies
// per pass; blocked_states, goal and the observer are forwarded to every pass.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace mymyr
{
struct GroundCondition;
}

namespace mymyr::search
{
enum class SearchStatus : u8
{
    Solved,       // a goal state was reached; the result carries the plan
    Exhausted,    // every reachable (non-blocked, non-pruned) state was expanded without reaching the goal
    OutOfStates,  // Budget::max_states or max_expanded hit
    OutOfTime,    // Budget::max_seconds hit
    Cancelled,    // CancelToken requested, or an observer asked to stop
    Failed,       // the search cannot run on this task (e.g. an unsupported feature); see the result's message
    Unsolvable,   // the goal can never hold: it contains a static literal that is false (mimir's UNSOLVABLE)
};

[[nodiscard]] constexpr const char* to_string(SearchStatus s) noexcept
{
    switch (s)
    {
        case SearchStatus::Solved: return "solved";
        case SearchStatus::Exhausted: return "exhausted";
        case SearchStatus::OutOfStates: return "out_of_states";
        case SearchStatus::OutOfTime: return "out_of_time";
        case SearchStatus::Cancelled: return "cancelled";
        case SearchStatus::Failed: return "failed";
        case SearchStatus::Unsolvable: return "unsolvable";
    }
    return "?";
}

/// Transitions between two checks of the deadline and the cancellation inside one expansion: a state with many
/// successors does not delay a stop by more than this many transitions.
inline constexpr u32 k_check_transitions = 1024;

/// Limits. Unset members are unlimited.
///
/// max_states bounds the states a search stores, the start state included (IW: the nodes of a pass's tree; rollout
/// IW: the new nodes its rollouts register, mimir's generated states). The search stops with OutOfStates as soon as
/// it holds max_states states: the count is checked as each new state is stored, inside the expansion that finds it,
/// so a search never stores more than max(max_states, 1) states (the GPU searches check between chunks; see
/// mymyr/cuda). max_expanded is checked before each expansion; max_seconds and the cancellation before each
/// expansion and every k_check_transitions transitions inside one.
struct Budget
{
    u64 max_states = std::numeric_limits<u64>::max();    // stored states (per pass in IW ladders)
    u64 max_expanded = std::numeric_limits<u64>::max();  // expanded states
    u32 max_depth = std::numeric_limits<u32>::max();     // g-depth in actions (BrFS/IW layers)
    double max_seconds = std::numeric_limits<double>::infinity();  // wall time of the whole call
};

/// Cooperative cancellation, cheap to copy and safe to request from any thread (the Python binding hands one to
/// GIL-free searches).
class CancelToken
{
public:
    CancelToken() : m_flag(std::make_shared<std::atomic<bool>>(false)) {}
    void request() const noexcept { m_flag->store(true, std::memory_order_relaxed); }
    [[nodiscard]] bool requested() const noexcept { return m_flag->load(std::memory_order_relaxed); }

private:
    std::shared_ptr<std::atomic<bool>> m_flag;
};

/// When a state counts as a goal. The default is the task's goal (mimir's ProblemGoalStrategy).
struct GoalSpec
{
    enum class Kind : u8
    {
        Task,    // the task's goal condition
        AnyOf,   // any of several goals (mimir's ProblemMultiGoalStrategy); `goals` holds them
        Custom,  // `test` decides
    };
    /// One goal of AnyOf: a ground conjunctive condition resolved against the task (search/goal.hpp makes one from a
    /// GroundCondition, deciding its static literals there). Heuristics estimate the fluent literals only.
    struct AtomGoal
    {
        std::vector<SlotId> positive{}, negative{};  // fluent atom slots that must hold / must not hold
        /// Derived atoms (canonical ids) that must hold / must not hold in the state's closure under the axioms.
        std::vector<CanonicalAtom> derived_positive{}, derived_negative{};
        /// Numeric constraints over ground expressions: those of this condition (its literals are not read), on the
        /// state's numeric values. Null: none.
        std::shared_ptr<const GroundCondition> numeric{};

        /// Whether the goal reads derived atoms: the state's axiom closure must be prepared to test it.
        [[nodiscard]] bool needs_view() const noexcept { return !derived_positive.empty() || !derived_negative.empty(); }
        /// Whether the goal consists of fluent literals only, so it can be tested with atom masks alone.
        [[nodiscard]] bool fluent_only() const noexcept { return !needs_view() && !numeric; }
    };

    Kind kind = Kind::Task;
    std::vector<AtomGoal> goals;
    std::function<bool(StateView)> test;  // Custom only; called on the searching thread(s)
    /// Custom only: per-worker goal tests for parallel searches (tyr's make_worker protocol). A parallel search calls
    /// make_worker(w) once per worker w on the calling thread, before any worker runs, and worker w then calls only
    /// the function it got. Empty (the default), or returning an empty function, means "not parallel-safe": the
    /// parallel searches then run on the calling thread alone (search/parallel_rollouts.hpp, portfolio.hpp).
    std::function<std::function<bool(StateView)>(u32 worker)> make_worker;
};

struct SearchStatistics
{
    u64 expanded = 0;
    u64 generated = 0;
    u64 states = 0;  // stored (BrFS, A*, GBFS) or novel (IW) states
    u64 pruned = 0;  // blocked, not novel, or dead ends
    double seconds = 0;
};

/// What became of the successor of a transition (tyr's TransitionOutcome), reported by SearchObserver::on_transition.
enum class TransitionOutcome : u8
{
    Opened,     // a new state entered the search (open list, IW tree, rollout tree)
    Reopened,   // a known state was reached on a cheaper path and re-entered the open list (tyr's RELAXED)
    Duplicate,  // a known state; nothing changed (a width-0 duplicate entry, an existing rollout child)
    Pruned,     // not entered: not novel, blocked, a self loop, beyond a depth bound, or cut by a budget
    DeadEnd,    // not entered: recognized as a dead end (infinite heuristic value, no applicable action)
    Goal,       // a goal state, found by a search that tests the goal on generation (Rollout IW)
};

/// Event hooks: the public extension point (matching mimir's event handlers; `hierarchical` implements a native tree capture on
/// them). Every method has an empty default; searches call them only when an observer is set. Ids are the search's
/// own state ids. Returning false from on_progress stops the search with SearchStatus::Cancelled.
class SearchObserver
{
public:
    virtual ~SearchObserver() = default;
    virtual void on_start(StateView /*initial*/) {}
    virtual void on_expand(u64 /*id*/, StateView /*state*/) {}
    virtual void on_generate(u64 /*parent*/, const Action& /*action*/, u64 /*child*/, StateView /*child_state*/, bool /*is_new*/) {}
    virtual void on_prune(u64 /*parent*/, const Action& /*action*/, StateView /*child_state*/) {}
    virtual void on_pass(u32 /*arity_or_layer*/, const SearchStatistics& /*pass*/) {}  // end of an IW pass / BrFS layer
    virtual void on_solution(std::span<const Action> /*plan*/, double /*cost*/) {}
    virtual bool on_progress(const SearchStatistics& /*so_far*/) { return true; }
    virtual void on_end(SearchStatus /*status*/, const SearchStatistics& /*total*/) {}
    /// Every transition of an expanded state with what became of its successor (TransitionOutcome, tyr's
    /// TransitionOutcome). Called after on_generate / on_prune, which are kept. `child` is the successor's id, or ~0
    /// when it got none (pruned). Emitted by the searches that document it (the IW family variants).
    virtual void on_transition(u64 /*parent*/, const Action& /*action*/, u64 /*child*/, StateView /*child_state*/, TransitionOutcome /*outcome*/) {}
    /// Per-worker observers for parallel searches (tyr's make_worker protocol). A parallel search calls make_worker(w)
    /// once per worker w on the calling thread before any worker runs. Worker w then sends its hot events (on_expand,
    /// on_generate, on_prune, on_transition, on_progress) to the returned observer, from one thread at a time, while
    /// the lifecycle events (on_start, on_pass, on_solution, on_end) stay on this, the root observer, and are called
    /// from the calling thread only. The default, nullptr, means "not parallel-safe": the parallel searches then run
    /// on the calling thread alone and send every event to this observer. What a worker is (a rollout, a portfolio
    /// worker) is documented by each parallel search. The search keeps the returned observers alive until it returns.
    virtual std::shared_ptr<SearchObserver> make_worker(u32 /*worker*/) { return nullptr; }
};

/// The control block every search's options embed.
struct SearchControl
{
    Budget budget;
    CancelToken cancel;
    GoalSpec goal;
    std::vector<State> blocked_states;    // never entered (mimir's `blocked_states`); compared by content
    SearchObserver* observer = nullptr;   // not owned; must outlive the search
    u64 progress_interval = u64{1} << 16; // expansions between on_progress calls and budget checks
    /// Optional coordination with searches running side by side (SearchCoordination below; not owned). Honored by
    /// the IW family variants (search/aiw.hpp, liw.hpp, rollout_iw.hpp); the portfolio (portfolio.hpp) coordinates
    /// its workers with one of its own and, like find_rollouts_parallel, rejects one; iw(), siw() and the best-first
    /// searches ignore it.
    struct SearchCoordination* coordination = nullptr;
};

/// The shared state of searches that run side by side, matching mimir 0.16.3's SearchControl
/// (search/algorithms/search_control.hpp): a stop flag, the best plan length found so far, the deepest layer a
/// breadth-first certifier has completely processed, and a shared expansion budget. Every field is relaxed-atomic;
/// a reader that is one update stale merely does a little more work.
///
/// A breadth-first search that is given one (SearchControl::coordination) and runs the plain queued path:
///   - stops with SearchStatus::Cancelled when `cancel` is set (checked once per popped state);
///   - on popping the first state of layer g + 1, publishes completed depth g (the goal is tested on pop, so no plan
///     of length <= g exists in its search space) and stops with Cancelled once the incumbent is certified;
///   - adds one expansion per counted expansion (reaching max_total_expansions requests cancel);
///   - on exhausting its space, invalidates the lower bound (its space holds no plan at all, so its depths certify
///     nothing about plans found in differently pruned spaces).
/// A search with a randomized or truncated layer order cannot publish sound depths and rejects a coordination.
struct SearchCoordination
{
    static constexpr u32 k_no_depth = ~u32{0};
    static constexpr u32 k_no_incumbent = ~u32{0};

    std::atomic<bool> cancel{false};
    std::atomic<u32> incumbent_length{k_no_incumbent};  // monotonically non-increasing
    std::atomic<u32> completed_depth{k_no_depth};       // monotonically non-decreasing
    std::atomic<u64> total_expansions{0};
    std::atomic<u64> max_total_expansions{~u64{0}};
    std::atomic<bool> lower_bound_invalidated{false};

    SearchCoordination() = default;
    SearchCoordination(const SearchCoordination&) = delete;
    SearchCoordination& operator=(const SearchCoordination&) = delete;

    [[nodiscard]] bool is_cancelled() const noexcept { return cancel.load(std::memory_order_relaxed); }
    void request_cancel() noexcept { cancel.store(true, std::memory_order_relaxed); }

    [[nodiscard]] u32 get_incumbent_length() const noexcept { return incumbent_length.load(std::memory_order_relaxed); }
    /// Installs `length` if it beats the incumbent; true if this call installed it.
    bool improve_incumbent_length(u32 length) noexcept
    {
        u32 cur = incumbent_length.load(std::memory_order_relaxed);
        while (length < cur)
            if (incumbent_length.compare_exchange_weak(cur, length, std::memory_order_relaxed, std::memory_order_relaxed))
                return true;
        return false;
    }

    [[nodiscard]] u32 get_completed_depth() const noexcept { return completed_depth.load(std::memory_order_relaxed); }
    void publish_completed_depth(u32 depth) noexcept
    {
        u32 cur = completed_depth.load(std::memory_order_relaxed);
        while (cur == k_no_depth || depth > cur)
            if (completed_depth.compare_exchange_weak(cur, depth, std::memory_order_relaxed, std::memory_order_relaxed))
                return;
    }

    void invalidate_lower_bound() noexcept { lower_bound_invalidated.store(true, std::memory_order_relaxed); }
    [[nodiscard]] bool is_lower_bound_invalidated() const noexcept { return lower_bound_invalidated.load(std::memory_order_relaxed); }

    /// 0 while no layer is complete, completed_depth + 1 afterwards.
    [[nodiscard]] u32 get_lower_bound() const noexcept
    {
        const u32 d = get_completed_depth();
        return d == k_no_depth ? 0u : d + 1u;
    }
    /// Whether a plan of `length` is proven shortest by the published depths.
    [[nodiscard]] bool is_incumbent_certified(u32 length) const noexcept
    {
        return length != k_no_incumbent && !is_lower_bound_invalidated() && get_lower_bound() >= length;
    }

    void add_expansions(u64 count) noexcept
    {
        const u64 total = total_expansions.fetch_add(count, std::memory_order_relaxed) + count;
        if (total >= max_total_expansions.load(std::memory_order_relaxed))
            request_cancel();
    }
    [[nodiscard]] u64 get_total_expansions() const noexcept { return total_expansions.load(std::memory_order_relaxed); }
};
}  // namespace mymyr::search
