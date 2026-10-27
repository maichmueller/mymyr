// mymyr._core._search: the search family and the heuristics (mymyr.search), plus the landmark and relaxed
// reachability API (landmarks_bindings.cpp).
//
// Every search releases the thread state while it runs, so other Python threads (free-threaded 3.14t) keep running and
// any number of searches may run at once on one task. Python code runs only in the optional callbacks: an observer
// (any object with some of the on_* methods of search/control.hpp), a custom goal test, and nothing else. A callback
// attaches its thread state for the call. The first exception a callback raises requests the search's CancelToken
// (the caller's, if one was given), and is re-raised when the search returns.
//
// The parallel searches (find_rollouts_parallel, atomic_goal_portfolio) follow the make_worker protocol
// (search/control.hpp): an observer with a make_worker(k) method, or a goal callable with one, gives every worker its
// own object, called from that worker's thread; without it the batch runs on the calling thread alone.

#include "py_datasets.hpp"
#include "py_formula.hpp"
#include "py_landmarks.hpp"
#include "py_task.hpp"
#include "py_search_tree.hpp"

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/perfect.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/anytime.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/ehc.hpp"
#include "mymyr/search/goal.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/iw_family.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/search/validate.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/task/task.hpp"

#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mymyr::python
{
namespace
{
using namespace nb::literals;
using search::SearchStatus;

// ------------------------------------------------------------------------------------------------ small helpers

nb::typed<nb::list, PyAction> plan_list(const Owner& o, const std::vector<Action>& plan)
{
    nb::typed<nb::list, PyAction> out{nb::list()};
    for (const Action& a : plan)
        out.append(action_object(o, a));
    return out;
}

nb::typed<nb::object, std::optional<PyState>> state_or_none(const Owner& o, const std::optional<State>& s)
{
    if (!s)
        return nb::none();
    return make_state(o, State(*s));
}

std::string lower(std::string s)
{
    for (char& c : s)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return s;
}

std::string str_arg(nb::handle h, const char* what)
{
    if (!nb::isinstance<nb::str>(h))
        throw nb::type_error((std::string("mymyr: ") + what + " must be a string").c_str());
    return lower(nb::cast<std::string>(h));
}

State state_for(const Owner& o, nb::handle h, const char* what)
{
    if (!is_state(h))
        throw nb::type_error((std::string("mymyr: ") + what + " must be a mymyr.State").c_str());
    const PyState& s = state_of(h);
    if (s.core->task->uid() != o.core->task->uid())
        throw nb::value_error((std::string("mymyr: ") + what + " belongs to another task").c_str());
    return s.s;
}

heuristics::Kind parse_kind(nb::handle h)
{
    try
    {
        return heuristics::parse_kind(str_arg(h, "heuristic"));
    }
    catch (const std::invalid_argument& e)
    {
        throw nb::value_error(e.what());
    }
}

heuristics::Costs parse_costs(nb::handle h)
{
    const std::string s = str_arg(h, "costs");
    if (s == "auto")
        return heuristics::Costs::Auto;
    if (s == "unit")
        return heuristics::Costs::Unit;
    if (s == "real")
        return heuristics::Costs::Real;
    throw nb::value_error("mymyr: costs must be 'auto', 'unit' or 'real'");
}

const char* costs_name(heuristics::Costs c)
{
    return c == heuristics::Costs::Unit ? "unit" : c == heuristics::Costs::Real ? "real" : "auto";
}

const char* evaluation_name(heuristics::Evaluation e)
{
    return e == heuristics::Evaluation::Grounded ? "grounded" : e == heuristics::Evaluation::Lifted ? "lifted" : "auto";
}

heuristics::Evaluation parse_evaluation(nb::handle h)
{
    const std::string s = str_arg(h, "evaluation");
    if (s == "auto")
        return heuristics::Evaluation::Auto;
    if (s == "grounded")
        return heuristics::Evaluation::Grounded;
    if (s == "lifted")
        return heuristics::Evaluation::Lifted;
    throw nb::value_error("mymyr: evaluation must be 'auto', 'grounded' or 'lifted'");
}

// ------------------------------------------------------------------------------------------------ cancellation

struct PyCancelToken
{
    search::CancelToken token;
};

// ------------------------------------------------------------------------------------------------ callbacks

/// Native search workers are joined before the binding returns. Keep their Python thread state until thread exit,
/// detached between callbacks so that the GIL and garbage collection can proceed while native work runs. Creating
/// and clearing a state per event would repeatedly abandon its allocator heaps into the shared segment pool.
nb::gil_scoped_acquire acquire_callback()
{
    if (!PyGILState_GetThisThreadState())
    {
        struct ThreadState
        {
            PyGILState_STATE token = PyGILState_Ensure();
            PyThreadState* state = PyEval_SaveThread();

            ~ThreadState()
            {
                PyEval_RestoreThread(state);
                PyGILState_Release(token);
            }
        };
        thread_local ThreadState retained;
    }
    return {};
}

/// Collects the first exception of any callback, and cancels the search when one occurs.
struct CallbackErrors
{
    search::CancelToken cancel;
    std::mutex m;
    std::unique_ptr<nb::python_error> first;  // touched with an attached thread state only
    std::atomic<bool> failed{false};

    /// Call with an attached thread state, inside the catch of a python_error.
    void record(nb::python_error&& e)
    {
        {
            std::lock_guard lock(m);
            if (!first)
                first = std::make_unique<nb::python_error>(std::move(e));
        }
        failed.store(true, std::memory_order_relaxed);
        cancel.request();
    }

    /// Call with an attached thread state after the search: re-raises the first error.
    void rethrow()
    {
        if (first)
        {
            std::unique_ptr<nb::python_error> e = std::move(first);
            throw nb::python_error(std::move(*e));
        }
    }
};

/// A SearchObserver forwarding to the on_* methods a Python object defines (the others are skipped for free), and to
/// its make_worker(k) for the parallel searches. The methods of mymyr.search.Observer that a subclass does not
/// override carry the attribute `_mymyr_default_event` and count as not defined.
class PyObserver final : public search::SearchObserver
{
public:
    PyObserver(nb::handle obj, Owner owner, CallbackErrors& errors) : m_owner(std::move(owner)), m_errors(errors)
    {
        auto get = [&](const char* name) -> nb::object {
            if (!nb::hasattr(obj, name))
                return {};
            nb::object fn = nb::getattr(obj, name);
            if (nb::hasattr(fn, "_mymyr_default_event"))
                return {};
            return fn;
        };
        m_start = get("on_start");
        m_expand = get("on_expand");
        m_generate = get("on_generate");
        m_prune = get("on_prune");
        m_pass = get("on_pass");
        m_solution = get("on_solution");
        m_progress = get("on_progress");
        m_end = get("on_end");
        m_transition = get("on_transition");
        m_make_worker = get("make_worker");
    }

    ~PyObserver() override
    {
        nb::gil_scoped_acquire acquire;  // the callables are released with an attached thread state
        m_start.reset();
        m_expand.reset();
        m_generate.reset();
        m_prune.reset();
        m_pass.reset();
        m_solution.reset();
        m_progress.reset();
        m_end.reset();
        m_transition.reset();
        m_make_worker.reset();
        m_owner.obj.reset();
    }

    /// Whether the object defines any event or make_worker (an observer that defines none is not installed).
    [[nodiscard]] bool defines_any() const
    {
        for (const nb::object* f : {&m_start, &m_expand, &m_generate, &m_prune, &m_pass, &m_solution, &m_progress,
                                    &m_end, &m_transition, &m_make_worker})
            if (f->is_valid())
                return true;
        return false;
    }

    void on_transition(u64 parent, const Action& action, u64 child, StateView child_state,
                       search::TransitionOutcome outcome) override
    {
        call(m_transition, [&] {
            nb::object c = child == ~u64{0} ? nb::none() : nb::cast(child);
            return nb::make_tuple(parent, action_object(m_owner, action), c, make_state(m_owner, State(child_state)),
                                  outcome);
        });
    }

    /// Called on the calling thread before the workers run: the Python object's make_worker(k) returns worker k's
    /// observer (None, or no make_worker: not parallel-safe, the search then runs on the calling thread alone).
    std::shared_ptr<search::SearchObserver> make_worker(u32 worker) override
    {
        if (!m_make_worker.is_valid() || m_errors.failed.load(std::memory_order_relaxed))
            return nullptr;
        nb::gil_scoped_acquire acquire;
        try
        {
            nb::object w = m_make_worker(worker);
            if (w.is_none())
                return nullptr;
            return std::make_shared<PyObserver>(w, m_owner, m_errors);
        }
        catch (nb::python_error& e)
        {
            m_errors.record(std::move(e));
            return nullptr;
        }
    }

    void on_start(StateView initial) override
    {
        call(m_start, [&] { return nb::make_tuple(make_state(m_owner, State(initial))); });
    }
    void on_expand(u64 id, StateView state) override
    {
        call(m_expand, [&] { return nb::make_tuple(id, make_state(m_owner, State(state))); });
    }
    void on_generate(u64 parent, const Action& action, u64 child, StateView child_state, bool is_new) override
    {
        call(m_generate, [&] {
            nb::object c = child == ~u64{0} ? nb::none() : nb::cast(child);
            return nb::make_tuple(parent, action_object(m_owner, action), c, make_state(m_owner, State(child_state)),
                                  is_new);
        });
    }
    void on_prune(u64 parent, const Action& action, StateView child_state) override
    {
        call(m_prune, [&] {
            return nb::make_tuple(parent, action_object(m_owner, action), make_state(m_owner, State(child_state)));
        });
    }
    void on_pass(u32 arity_or_layer, const search::SearchStatistics& pass) override
    {
        call(m_pass, [&] { return nb::make_tuple(arity_or_layer, pass); });
    }
    void on_solution(std::span<const Action> plan, double cost) override
    {
        call(m_solution, [&] {
            nb::list p;
            for (const Action& a : plan)
                p.append(action_object(m_owner, a));
            return nb::make_tuple(p, cost);
        });
    }
    bool on_progress(const search::SearchStatistics& so_far) override
    {
        if (m_errors.failed.load(std::memory_order_relaxed))
            return false;
        if (!m_progress.is_valid())
            return true;
        auto acquire = acquire_callback();
        try
        {
            nb::object r = m_progress(so_far);
            if (r.is_none())
                return true;
            const int t = PyObject_IsTrue(r.ptr());
            if (t < 0)
                throw nb::python_error();
            return t != 0;
        }
        catch (nb::python_error& e)
        {
            m_errors.record(std::move(e));
            return false;
        }
    }
    void on_end(SearchStatus status, const search::SearchStatistics& total) override
    {
        call(m_end, [&] { return nb::make_tuple(status, total); });
    }

private:
    template<class MakeArgs>
    void call(const nb::object& fn, MakeArgs&& make_args)
    {
        if (!fn.is_valid() || m_errors.failed.load(std::memory_order_relaxed))
            return;
        auto acquire = acquire_callback();
        try
        {
            nb::tuple args = make_args();
            PyObject* r = PyObject_Call(fn.ptr(), args.ptr(), nullptr);
            if (!r)
                throw nb::python_error();
            Py_DECREF(r);
        }
        catch (nb::python_error& e)
        {
            m_errors.record(std::move(e));
        }
    }

    Owner m_owner;
    CallbackErrors& m_errors;
    nb::object m_start, m_expand, m_generate, m_prune, m_pass, m_solution, m_progress, m_end, m_transition, m_make_worker;
};

/// Owns a Python goal-test callable: the std::function in GoalSpec copies this wrapper (shared_ptr), and the last copy
/// releases the callable with an attached thread state.
struct PyGoalTest
{
    nb::object fn;
    Owner owner;
    CallbackErrors* errors = nullptr;

    ~PyGoalTest()
    {
        nb::gil_scoped_acquire acquire;
        fn.reset();
        owner.obj.reset();
    }
};

// ------------------------------------------------------------------------------------------------ control

/// Everything a search call needs besides its algorithm options: the control block, the errors and the observer.
struct ControlScope
{
    search::SearchControl control;
    std::unique_ptr<CallbackErrors> errors = std::make_unique<CallbackErrors>();
    std::unique_ptr<PyObserver> observer;
};

/// A goal of goal=: a GroundCondition, or a sequence of ground literals and atoms as Task.ground_condition takes them.
GroundCondition goal_condition(const Owner& o, nb::handle g)
{
    if (!nb::isinstance<PyGroundCondition>(g))
    {
        if (nb::isinstance<nb::str>(g) || nb::isinstance<PyGroundAtom>(g) || nb::isinstance<PyGroundLiteral>(g) ||
            !nb::isinstance<nb::sequence>(g))
            throw nb::type_error("mymyr: goal= takes a GroundCondition or a sequence of goals, each a GroundCondition or "
                                 "a sequence of ground literals and atoms (any goal reached is a goal)");
        return *nb::cast<const PyGroundCondition&>(task_object(o.obj).attr("ground_condition")(g)).c;
    }
    const PyGroundCondition& c = *nb::inst_ptr<PyGroundCondition>(g);
    if (c.o.data.get() != o.core->data.get())
        throw nb::value_error("mymyr: the goal belongs to another task");
    return *c.c;
}

/// The goal test calling a Python callable (the thread state attached for the call).
std::function<bool(StateView)> goal_test(std::shared_ptr<PyGoalTest> w)
{
    return [w = std::move(w)](StateView s) -> bool {
        if (w->errors->failed.load(std::memory_order_relaxed))
            return false;
        auto acquire = acquire_callback();
        try
        {
            nb::object r = w->fn(make_state(w->owner, State(s)));
            const int t = PyObject_IsTrue(r.ptr());
            if (t < 0)
                throw nb::python_error();
            return t != 0;
        }
        catch (nb::python_error& e)
        {
            w->errors->record(std::move(e));
            return false;
        }
    };
}

search::GoalSpec parse_goal(const Owner& o, nb::handle goal, CallbackErrors& errors)
{
    search::GoalSpec spec;
    if (goal.is_none())
        return spec;
    if (PyCallable_Check(goal.ptr()) && !nb::isinstance<nb::str>(goal))
    {
        spec.kind = search::GoalSpec::Kind::Custom;
        auto w = std::make_shared<PyGoalTest>();
        w->fn = nb::borrow(goal);
        w->owner = o;
        w->errors = &errors;
        spec.test = goal_test(w);
        if (nb::hasattr(goal, "make_worker"))
        {
            // the make_worker protocol for parallel searches: goal.make_worker(k) is worker k's own callable
            auto mk = std::make_shared<PyGoalTest>();
            mk->fn = nb::getattr(goal, "make_worker");
            mk->owner = o;
            mk->errors = &errors;
            spec.make_worker = [mk](u32 worker) -> std::function<bool(StateView)> {
                if (mk->errors->failed.load(std::memory_order_relaxed))
                    return {};
                nb::gil_scoped_acquire acquire;
                try
                {
                    nb::object f = mk->fn(worker);
                    if (f.is_none())
                        return {};
                    if (!PyCallable_Check(f.ptr()))
                    {
                        PyErr_SetString(PyExc_TypeError, "mymyr: goal.make_worker(k) must return a callable or None");
                        throw nb::python_error();
                    }
                    auto w = std::make_shared<PyGoalTest>();
                    w->fn = std::move(f);
                    w->owner = mk->owner;
                    w->errors = mk->errors;
                    return goal_test(std::move(w));
                }
                catch (nb::python_error& e)
                {
                    mk->errors->record(std::move(e));
                    return {};
                }
            };
        }
        return spec;
    }
    std::vector<GroundCondition> goals;
    if (nb::isinstance<PyGroundCondition>(goal))
        goals.push_back(goal_condition(o, goal));
    else
    {
        if (nb::isinstance<nb::str>(goal) || !nb::isinstance<nb::sequence>(goal))
            throw nb::type_error("mymyr: goal= takes a GroundCondition, a sequence of goals or a callable");
        for (nb::handle g : goal)
            goals.push_back(goal_condition(o, g));
    }
    try
    {
        return search::any_of(*o.core->task, goals);
    }
    catch (const std::invalid_argument& e)
    {
        throw nb::value_error(e.what());
    }
}

/// Fills the control block from the common keyword arguments.
void fill_control(ControlScope& cs, const Owner& o, nb::handle max_states, nb::handle max_expanded, nb::handle max_depth,
                  nb::handle max_seconds, nb::handle cancel, nb::handle goal, nb::handle blocked_states,
                  nb::handle observer, nb::handle progress_interval)
{
    search::SearchControl& c = cs.control;
    if (auto v = opt_int_arg<u64>(max_states, "max_states", 0))
        c.budget.max_states = *v;
    if (auto v = opt_int_arg<u64>(max_expanded, "max_expanded", 0))
        c.budget.max_expanded = *v;
    if (auto v = opt_int_arg<u32>(max_depth, "max_depth", 0))
        c.budget.max_depth = *v;
    if (auto v = opt_float_arg(max_seconds, "max_seconds", 0))
        c.budget.max_seconds = *v;
    if (!cancel.is_none())
    {
        if (!nb::isinstance<PyCancelToken>(cancel))
            throw nb::type_error("mymyr: cancel must be a mymyr.search.CancelToken");
        c.cancel = nb::inst_ptr<PyCancelToken>(cancel)->token;
    }
    cs.errors->cancel = c.cancel;
    c.goal = parse_goal(o, goal, *cs.errors);
    if (!blocked_states.is_none())
        for (nb::handle s : blocked_states)
            c.blocked_states.push_back(state_for(o, s, "a blocked state"));
    if (!observer.is_none())
    {
        cs.observer = std::make_unique<PyObserver>(observer, o, *cs.errors);
        if (cs.observer->defines_any())
            c.observer = cs.observer.get();
        else
            cs.observer.reset();  // nothing to call: the search runs as without an observer
    }
    if (auto v = opt_int_arg<u64>(progress_interval, "progress_interval", 0))
        c.progress_interval = *v == 0 ? 1 : *v;
}

/// Runs `fn` with the thread state detached, then re-raises a callback's exception. Library exceptions propagate
/// (nanobind maps std::invalid_argument to ValueError, std::runtime_error to RuntimeError).
template<class F>
auto run_detached(ControlScope& cs, F&& fn)
{
    std::optional<decltype(fn())> r;
    {
        nb::gil_scoped_release release;
        r.emplace(fn());
    }
    cs.observer.reset();  // releases the observer's callables (attaches itself)
    cs.control.goal = {};
    cs.errors->rethrow();
    return std::move(*r);
}

// ------------------------------------------------------------------------------------------------ results

struct PySiwSubproblem
{
    search::SiwSubproblem p;
    Owner o;
};

template<class T>
nb::ndarray<nb::numpy, const T> tree_array(const PySearchTree& x, const std::vector<T>& values,
                                         std::initializer_list<usize> shape)
{
    auto* hold = new std::shared_ptr<const search::SearchTree>(x.tree);
    nb::capsule owner(hold, [](void* p) noexcept { delete static_cast<std::shared_ptr<const search::SearchTree>*>(p); });
    // Empty arrays still need a non-null data address for NumPy to retain the capsule without allocating storage.
    static const T empty{};
    return nb::ndarray<nb::numpy, const T>(values.empty() ? &empty : values.data(), shape, owner);
}

struct PyIwResult
{
    search::IwResult r;
    Owner o;
};
struct PySiwResult
{
    search::SiwResult r;
    Owner o;
};
struct PyBestFirstResult
{
    search::BestFirstResult r;
    Owner o;
};
struct PyAStarIwResult : PyBestFirstResult
{
    search::AStarIwNoveltyStatistics novelty;
};
struct PyEhcResult : PyBestFirstResult
{
    search::EhcStatistics ehc;
};
struct PyAnytimeResult : PyBestFirstResult
{
    std::vector<search::AnytimeSolution> solutions;
    std::vector<search::AnytimeIteration> iterations;
    bool optimal = false;
};
struct PyBrfsResult
{
    BrfsResult r;
    Owner o;
};

std::string stats_repr(const search::SearchStatistics& s)
{
    return "expanded=" + std::to_string(s.expanded) + ", generated=" + std::to_string(s.generated) +
           ", states=" + std::to_string(s.states) + ", pruned=" + std::to_string(s.pruned);
}

std::string cost_repr(double c)
{
    if (std::floor(c) == c && std::fabs(c) < 1e15)
        return std::to_string(static_cast<long long>(c));
    return std::to_string(c);
}

// The fields every search result has, under the same names and docstrings.
constexpr const char* k_status_doc = "How the search ended (Status).";
constexpr const char* k_solved_doc = "True when status is Status.SOLVED (there is a plan).";
constexpr const char* k_plan_doc =
    "The plan: a list of Actions from the start state to a goal state (empty without a plan).";
constexpr const char* k_cost_doc = "The plan's cost under the task's metric (0.0 without a plan).";
constexpr const char* k_goal_state_doc = "The State the plan reaches (None without a plan).";
constexpr const char* k_stats_doc = "The search's counts and time (Statistics).";
constexpr const char* k_message_doc =
    "Why the search stopped early or could not run (empty when it ran to its end).";

/// Binds status, solved, plan, cost, goal_state, stats and message on `c`: `get(x)` gives the C++ result (its status,
/// plan, cost, goal_state and message), `stats(x)` its SearchStatistics.
template <class Py, class Get, class Stats>
void bind_result_fields(nb::class_<Py>& c, Get get, Stats stats)
{
    c.def_prop_ro("status", [get](const Py& x) { return get(x).status; }, k_status_doc)
        .def_prop_ro("solved", [get](const Py& x) { return get(x).status == SearchStatus::Solved; }, k_solved_doc)
        .def_prop_ro("plan", [get](const Py& x) { return plan_list(x.o, get(x).plan); }, k_plan_doc)
        .def_prop_ro("cost", [get](const Py& x) { return get(x).cost; }, k_cost_doc)
        .def_prop_ro("goal_state", [get](const Py& x) { return state_or_none(x.o, get(x).goal_state); },
                     k_goal_state_doc)
        .def_prop_ro("stats", [stats](const Py& x) { return stats(x); }, k_stats_doc)
        .def_prop_ro("message", [get](const Py& x) { return get(x).message; }, k_message_doc);
}

/// A BrFS result's counts as SearchStatistics.
search::SearchStatistics brfs_stats(const BrfsResult& r)
{
    return {.expanded = r.expanded,
            .generated = r.generated,
            .states = r.states,
            .pruned = r.pruned,
            .seconds = r.search_s,
            .memory_bytes = r.memory_bytes};
}

/// A best-first result's stats.
search::SearchStatistics best_first_stats(const search::BestFirstResult& r)
{
    return r.stats;
}

// ------------------------------------------------------------------------------------------------ parallel rollouts and portfolio: results

struct PyRolloutIwResult
{
    search::RolloutIwResult r;
    Owner o;
};

/// One find_rollouts_parallel batch; its rollouts are views into it.
struct PyParallelRollouts
{
    std::shared_ptr<const search::ParallelRolloutsResult> r;
    std::vector<u64> seeds;
    Owner o;
};

struct PyRolloutResult
{
    std::shared_ptr<const search::ParallelRolloutsResult> batch;
    u32 index = 0;
    u64 seed = 0;
    Owner o;
    [[nodiscard]] const search::RolloutResult& get() const { return batch->rollouts[index]; }
};

struct PyLandingState
{
    search::LandingState l;
    Owner o;
};

struct PyMergedLandingStates
{
    search::MergedLandingStates m;
    Owner o;
};

struct PyPortfolioResult
{
    search::PortfolioResult r;
    Owner o;
};

struct PyPlanValidation
{
    search::PlanValidation v;
    Owner o;
};

/// Canonical ids as Atoms, one Python object per distinct atom (co-occurrence rows repeat atoms many times).
class AtomCache
{
public:
    explicit AtomCache(const Owner& o) : m_o(o) {}

    nb::object get(CanonicalAtom c)
    {
        auto [it, fresh] = m_map.try_emplace(c);
        if (fresh)
            it->second = atom_object(m_o, c);
        return it->second;
    }
    nb::typed<nb::list, PyGroundAtom> list(std::span<const CanonicalAtom> atoms)
    {
        nb::typed<nb::list, PyGroundAtom> out{nb::list()};
        for (CanonicalAtom c : atoms)
            out.append(get(c));
        return out;
    }

private:
    const Owner& m_o;
    std::unordered_map<CanonicalAtom, nb::object> m_map;
};

using CoOccurrence = std::vector<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>>;
using CoOccurrenceDict = nb::typed<nb::dict, PyGroundAtom, nb::typed<nb::list, PyGroundAtom>>;

// ------------------------------------------------------------------------------------------------ pickling results
// A result pickles as (kind, its Task, the task's fingerprint, its fields as a tuple): plans as (schema, objects)
// labels, states as States (which pickle with their task), statistics as tuples, statuses and atoms as ints.
// Unpickling checks the fingerprint, the labels (schemas, arities and objects of the task), the canonical atom ids
// and the statuses.

/// Reads the fields of a pickled tuple in order; ValueError ("corrupt ... pickle") for a missing or mistyped field.
class Fields
{
public:
    Fields(nb::handle t, const char* what) : m_what(what)
    {
        if (!nb::isinstance<nb::tuple>(t))
            corrupt();
        m_t = nb::borrow<nb::tuple>(t);
    }
    nb::handle next()
    {
        if (m_i >= m_t.size())
            corrupt();
        return m_t[m_i++];
    }
    template<class T>
    T get()
    {
        nb::handle h = next();
        T v{};
        if (!nb::try_cast(h, v))
            corrupt();
        return v;
    }
    SearchStatus status()
    {
        const int v = get<int>();
        if (v < 0 || v > static_cast<int>(SearchStatus::Unsolvable))
            corrupt();
        return static_cast<SearchStatus>(v);
    }
    void done() const
    {
        if (m_i != m_t.size())
            corrupt();
    }
    [[noreturn]] void corrupt() const { throw nb::value_error((std::string("mymyr: corrupt ") + m_what + " pickle").c_str()); }

private:
    nb::tuple m_t;
    usize m_i = 0;
    const char* m_what;
};

int status_int(SearchStatus s) { return static_cast<int>(s); }

nb::list enc_plan(const std::vector<Action>& plan)
{
    nb::list out;
    for (const Action& a : plan)
    {
        nb::list b;
        for (const ObjectId o : a.binding)
            b.append(o.v);
        out.append(nb::make_tuple(a.schema.v, nb::tuple(b)));
    }
    return out;
}

/// A plan of labels of `task`: schemas of the task, bindings of their arity over its objects.
std::vector<Action> dec_plan(const Task& task, nb::handle h, const char* what)
{
    if (!nb::isinstance<nb::list>(h))
        Fields(nb::none(), what).corrupt();
    std::vector<Action> plan;
    for (nb::handle step : nb::borrow<nb::list>(h))
    {
        Fields f(step, what);
        const u32 schema = f.get<u32>();
        nb::handle binding = f.next();
        f.done();
        if (schema >= task.num_schemas() || !nb::isinstance<nb::tuple>(binding) ||
            nb::len(binding) != task.data().schemas[schema].arity())
            throw nb::value_error(
                (std::string("mymyr: corrupt ") + what + " pickle (an action is not one of the task's)").c_str());
        std::vector<ObjectId> objects;
        for (nb::handle o : nb::borrow<nb::tuple>(binding))
        {
            u32 v = 0;
            if (!nb::try_cast(o, v) || v >= task.num_objects())
                throw nb::value_error(
                (std::string("mymyr: corrupt ") + what + " pickle (an action is not one of the task's)").c_str());
            objects.push_back(ObjectId{v});
        }
        plan.emplace_back(SchemaId{schema}, std::move(objects));
    }
    return plan;
}

std::optional<State> dec_state(const Owner& o, nb::handle h)
{
    if (h.is_none())
        return std::nullopt;
    return state_for(o, h, "a pickled result's state");
}

nb::list enc_atoms(std::span<const CanonicalAtom> atoms)
{
    nb::list out;
    for (const CanonicalAtom c : atoms)
        out.append(c);
    return out;
}

std::vector<CanonicalAtom> dec_atoms(const Task& task, nb::handle h, const char* what)
{
    std::vector<CanonicalAtom> out;
    if (!nb::isinstance<nb::list>(h))
        Fields(nb::none(), what).corrupt();
    for (nb::handle x : nb::borrow<nb::list>(h))
    {
        u64 c = 0;
        if (!nb::try_cast(x, c) || c >= task.compiled().layout.total)
            throw nb::value_error((std::string("mymyr: corrupt ") + what + " pickle (an atom is not one of the task's)").c_str());
        out.push_back(c);
    }
    return out;
}

template<class T, class F>
nb::list enc_list(const std::vector<T>& v, F&& f)
{
    nb::list out;
    for (const T& x : v)
        out.append(f(x));
    return out;
}

template<class T, class F>
std::vector<T> dec_list(nb::handle h, const char* what, F&& f)
{
    if (!nb::isinstance<nb::list>(h))
        Fields(nb::none(), what).corrupt();
    std::vector<T> out;
    for (nb::handle x : nb::borrow<nb::list>(h))
        out.push_back(f(x));
    return out;
}

// statistics (no task)
nb::tuple enc(const search::SearchStatistics& s)
{
    return nb::make_tuple(s.expanded, s.generated, s.states, s.pruned, s.seconds, s.memory_bytes);
}
search::SearchStatistics dec_stats(nb::handle h)
{
    Fields f(h, "statistics");
    search::SearchStatistics s;
    s.expanded = f.get<u64>();
    s.generated = f.get<u64>();
    s.states = f.get<u64>();
    s.pruned = f.get<u64>();
    s.seconds = f.get<double>();
    s.memory_bytes = f.get<u64>();
    f.done();
    return s;
}

nb::tuple enc_tree(const Owner& o, const std::shared_ptr<const search::SearchTree>& tree)
{
    const PySearchTree x{tree, o};
    const auto& t = *tree;
    const usize N = t.num_nodes(), T = t.num_transitions(), L = t.label_width, R = t.words + t.numeric_words;
    return nb::make_tuple(t.width, t.words, t.numeric_words, t.label_width,
                          tree_array(x, t.rows, {t.row_goal.size(), R}), tree_array(x, t.parent, {N}),
                          tree_array(x, t.depth, {N}), tree_array(x, t.schema, {N}), tree_array(x, t.binding, {N, L}),
                          tree_array(x, t.node_flags, {N}), tree_array(x, t.offsets, {t.offsets.size()}),
                          tree_array(x, t.t_schema, {T}), tree_array(x, t.t_binding, {T, L}),
                          tree_array(x, t.t_row, {T}), tree_array(x, t.t_child, {T}),
                          tree_array(x, t.t_outcome, {T}), tree_array(x, t.row_goal, {t.row_goal.size()}));
}

template<class T>
usize dec_tree_array(Fields& f, std::vector<T>& values, std::initializer_list<usize> shape)
{
    nb::ndarray<nb::numpy, const T, nb::c_contig, nb::device::cpu> a;
    if (!nb::try_cast(f.next(), a, false) || a.ndim() != shape.size())
        f.corrupt();
    usize d = 0;
    for (usize n : shape)
    {
        if (n != ~usize{0} && a.shape(d) != n)
            f.corrupt();
        ++d;
    }
    if (a.size())
        values.assign(a.data(), a.data() + a.size());
    return a.shape(0);
}

std::shared_ptr<const search::SearchTree> dec_tree(const Owner& o, nb::handle h)
{
    Fields f(h, "search tree");
    auto t = std::make_shared<search::SearchTree>();
    t->width = f.get<u32>();
    t->words = f.get<u32>();
    t->numeric_words = f.get<u32>();
    t->label_width = f.get<u32>();
    const Task& task = *o.core->task;
    const u32 max_words = task.numeric_words() ? std::max<u32>(1, task.words()) : task.words();
    if (t->numeric_words != task.numeric_words() || t->words > max_words ||
        t->label_width != o.core->label_width || t->width > novelty::k_max_arity)
        f.corrupt();
    const usize R = dec_tree_array(f, t->rows, {~usize{0}, usize{t->words} + t->numeric_words});
    dec_tree_array(f, t->parent, {~usize{0}});
    const usize stride = usize{t->words} + t->numeric_words;
    const usize N = t->num_nodes(), L = t->label_width;
    if (N == 0 || N > static_cast<usize>(std::numeric_limits<i32>::max()) || R < N || t->parent[0] != -1)
        f.corrupt();
    dec_tree_array(f, t->depth, {N});
    dec_tree_array(f, t->schema, {N});
    dec_tree_array(f, t->binding, {N, L});
    dec_tree_array(f, t->node_flags, {N});
    dec_tree_array(f, t->offsets, {~usize{0}});
    dec_tree_array(f, t->t_schema, {~usize{0}});
    const usize T = t->t_schema.size();
    dec_tree_array(f, t->t_binding, {T, L});
    dec_tree_array(f, t->t_row, {T});
    dec_tree_array(f, t->t_child, {T});
    dec_tree_array(f, t->t_outcome, {T});
    dec_tree_array(f, t->row_goal, {R});
    f.done();
    auto label = [&](i32 schema, const std::vector<i32>& bindings, usize i, bool root)
    {
        const u32 arity = root ? 0 : schema >= 0 && static_cast<u32>(schema) < task.num_schemas()
                                                  ? task.data().schemas[schema].arity() : ~u32{0};
        if (arity > L || (root && schema != -1))
            f.corrupt();
        for (usize j = 0; j < L; ++j)
        {
            const i32 b = bindings[i * L + j];
            if (j < arity ? b < 0 || static_cast<usize>(b) >= task.data().objects.size() : b != -1)
                f.corrupt();
        }
    };
    for (usize i = 0; i < N; ++i)
    {
        if (i ? t->parent[i] < 0 || static_cast<usize>(t->parent[i]) >= i ||
                t->depth[i] != t->depth[t->parent[i]] + 1 : t->depth[i] != 0)
            f.corrupt();
        if ((t->node_flags[i] & ~u8{31}) || !!(t->node_flags[i] & 2) != !!t->row_goal[i])
            f.corrupt();
        label(t->schema[i], t->binding, i, i == 0);
    }
    if (t->offsets.empty() ? T != 0 || R != N : t->offsets.size() != N + 1 || t->offsets[0] != 0 ||
        t->offsets.back() != static_cast<i64>(T) || !std::is_sorted(t->offsets.begin(), t->offsets.end()))
        f.corrupt();
    for (usize i = 0; i < T; ++i)
    {
        label(t->t_schema[i], t->t_binding, i, false);
        if (t->t_row[i] < 0 || static_cast<usize>(t->t_row[i]) >= R || t->t_child[i] < -1 ||
            (t->t_child[i] >= 0 && (static_cast<usize>(t->t_child[i]) >= N || t->t_child[i] != t->t_row[i])) ||
            t->t_outcome[i] > static_cast<u8>(search::TransitionOutcome::SelfLoop))
            f.corrupt();
        const auto outcome = static_cast<search::TransitionOutcome>(t->t_outcome[i]);
        if (outcome != search::TransitionOutcome::Opened && outcome != search::TransitionOutcome::Duplicate &&
            outcome != search::TransitionOutcome::Pruned && outcome != search::TransitionOutcome::Blocked &&
            outcome != search::TransitionOutcome::SelfLoop)
            f.corrupt();
        if ((outcome == search::TransitionOutcome::Opened && t->t_child[i] < 0) ||
            (t->t_child[i] >= 0 && outcome != search::TransitionOutcome::Opened && outcome != search::TransitionOutcome::Duplicate))
            f.corrupt();
    }
    const u32 slots = task.atoms().fluent_slots();
    for (usize i = 0; i < R; ++i)
    {
        const u64* row = stride ? t->rows.data() + i * stride : nullptr;
        for (u32 j = 0; j < t->words; ++j)
            if (usize{j} * 64 >= slots ? row[j] != 0 : slots / 64 == j && slots % 64 && (row[j] >> (slots % 64)) != 0)
                f.corrupt();
    }
    for (u8 g : t->row_goal)
        if (g > 1)
            f.corrupt();
    return t;
}

nb::tuple enc(const Owner& o, const search::IwPassStatistics& p)
{
    return nb::make_tuple(p.width, status_int(p.status), p.expanded, p.generated, p.generated_in_tree, p.skipped, p.blocked,
                          p.seconds, p.placeholder, p.tree ? nb::object(enc_tree(o, p.tree)) : nb::object(nb::none()));
}
search::IwPassStatistics dec_pass(const Owner& o, nb::handle h)
{
    Fields f(h, "IW pass statistics");
    search::IwPassStatistics p;
    p.width = f.get<u32>();
    p.status = f.status();
    p.expanded = f.get<u64>();
    p.generated = f.get<u64>();
    p.generated_in_tree = f.get<u64>();
    p.skipped = f.get<u64>();
    p.blocked = f.get<u64>();
    p.seconds = f.get<double>();
    p.placeholder = f.get<bool>();
    const nb::handle tree = f.next();
    if (!tree.is_none())
        p.tree = dec_tree(o, tree);
    f.done();
    return p;
}

nb::tuple enc(const Owner& o, const search::SiwSubproblem& p)
{
    return nb::make_tuple(status_int(p.status), p.unsatisfied_at_start, p.plan_length, p.effective_width,
                          enc_list(p.passes, [&](const auto& x) { return enc(o, x); }));
}
search::SiwSubproblem dec_subproblem(const Owner& o, nb::handle h)
{
    Fields f(h, "SIW subproblem");
    search::SiwSubproblem p;
    p.status = f.status();
    p.unsatisfied_at_start = f.get<u32>();
    p.plan_length = f.get<u32>();
    p.effective_width = f.get<u32>();
    p.passes = dec_list<search::IwPassStatistics>(f.next(), "SIW subproblem", [&](nb::handle x) { return dec_pass(o, x); });
    f.done();
    return p;
}

nb::tuple enc(const heuristics::HeuristicStats& s)
{
    return nb::make_tuple(s.evaluations, s.grounded, s.lifted, s.dead_ends, s.grounding_seconds);
}
heuristics::HeuristicStats dec_heuristic_stats(nb::handle h)
{
    Fields f(h, "heuristic statistics");
    heuristics::HeuristicStats s;
    s.evaluations = f.get<u64>();
    s.grounded = f.get<u64>();
    s.lifted = f.get<u64>();
    s.dead_ends = f.get<u64>();
    s.grounding_seconds = f.get<double>();
    f.done();
    return s;
}

nb::tuple enc(const search::AStarIwNoveltyStatistics& s)
{
    return nb::make_tuple(s.rejected, s.stale, s.stale_g, s.probes, s.updates, s.pop_tests, s.table_bytes);
}
search::AStarIwNoveltyStatistics dec_novelty(nb::handle h)
{
    Fields f(h, "A*-IW novelty statistics");
    search::AStarIwNoveltyStatistics s;
    s.rejected = f.get<u64>();
    s.stale = f.get<u64>();
    s.stale_g = f.get<u64>();
    s.probes = f.get<u64>();
    s.updates = f.get<u64>();
    s.pop_tests = f.get<u64>();
    s.table_bytes = f.get<u64>();
    f.done();
    return s;
}

nb::tuple enc(const search::EhcStatistics& s)
{
    return nb::make_tuple(s.lookaheads, s.improvements, s.retries, s.max_depth, s.expanded, s.evaluations, s.failed,
                          s.fallback);
}
search::EhcStatistics dec_ehc(nb::handle h)
{
    Fields f(h, "EHC statistics");
    search::EhcStatistics s;
    s.lookaheads = f.get<u64>();
    s.improvements = f.get<u64>();
    s.retries = f.get<u64>();
    s.max_depth = f.get<u64>();
    s.expanded = f.get<u64>();
    s.evaluations = f.get<u64>();
    s.failed = f.get<bool>();
    s.fallback = f.get<bool>();
    f.done();
    return s;
}

nb::tuple enc(const search::AnytimeSolution& s)
{
    return nb::make_tuple(s.cost, s.seconds, s.weight, s.iteration, s.expanded, s.length);
}
search::AnytimeSolution dec_anytime_solution(nb::handle h)
{
    Fields f(h, "anytime solution");
    search::AnytimeSolution s;
    s.cost = f.get<double>();
    s.seconds = f.get<double>();
    s.weight = f.get<double>();
    s.iteration = f.get<u32>();
    s.expanded = f.get<u64>();
    s.length = f.get<u64>();
    f.done();
    return s;
}

nb::tuple enc(const search::AnytimeIteration& s)
{
    return nb::make_tuple(s.weight, s.cost_bound, status_int(s.status), enc(s.stats), s.evaluations, s.cost);
}
search::AnytimeIteration dec_anytime_iteration(nb::handle h)
{
    Fields f(h, "anytime iteration");
    search::AnytimeIteration s;
    s.weight = f.get<double>();
    s.cost_bound = f.get<double>();
    s.status = f.status();
    s.stats = dec_stats(f.next());
    s.evaluations = f.get<u64>();
    s.cost = f.get<double>();
    f.done();
    return s;
}

nb::tuple enc(const search::RolloutIwStatistics& s)
{
    return nb::make_tuple(s.rollouts, s.generated, s.expanded, s.feature_depth_improvements, s.case1, s.case2, s.case3, s.case4,
                          s.solved_propagations, s.dead_ends, s.depth_bound_prunings, s.incumbent_bound_prunings, s.blocked,
                          s.max_rollout_depth, s.tree_nodes, s.seconds);
}
search::RolloutIwStatistics dec_rollout_stats(nb::handle h)
{
    Fields f(h, "Rollout IW statistics");
    search::RolloutIwStatistics s;
    for (u64* x : {&s.rollouts, &s.generated, &s.expanded, &s.feature_depth_improvements, &s.case1, &s.case2, &s.case3, &s.case4,
                   &s.solved_propagations, &s.dead_ends, &s.depth_bound_prunings, &s.incumbent_bound_prunings, &s.blocked})
        *x = f.get<u64>();
    s.max_rollout_depth = f.get<u32>();
    s.tree_nodes = f.get<u64>();
    s.seconds = f.get<double>();
    f.done();
    return s;
}

// results (of a task)
nb::tuple enc(const Owner& o, const search::IwResult& r)
{
    return nb::make_tuple(status_int(r.status), enc_plan(r.plan), r.cost, state_or_none(o, r.goal_state),
                          enc_list(r.passes, [&](const auto& x) { return enc(o, x); }), enc(r.stats), r.effective_width, r.message,
                          r.fluent_slots);
}
search::IwResult dec_iw(const Owner& o, nb::handle h)
{
    const char* what = "IW result";
    Fields f(h, what);
    search::IwResult r;
    r.status = f.status();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.goal_state = dec_state(o, f.next());
    r.passes = dec_list<search::IwPassStatistics>(f.next(), what, [&](nb::handle x) { return dec_pass(o, x); });
    r.stats = dec_stats(f.next());
    r.effective_width = f.get<u32>();
    r.message = f.get<std::string>();
    r.fluent_slots = f.get<u32>();
    f.done();
    return r;
}

nb::tuple enc(const Owner& o, const search::SiwResult& r)
{
    return nb::make_tuple(status_int(r.status), enc_plan(r.plan), enc_plan(r.partial_plan), r.cost,
                          state_or_none(o, r.goal_state), enc_list(r.subproblems, [&](const auto& x) { return enc(o, x); }),
                          enc(r.stats), r.max_effective_width, r.message, r.fluent_slots);
}
search::SiwResult dec_siw(const Owner& o, nb::handle h)
{
    const char* what = "SIW result";
    Fields f(h, what);
    search::SiwResult r;
    r.status = f.status();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.partial_plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.goal_state = dec_state(o, f.next());
    r.subproblems = dec_list<search::SiwSubproblem>(f.next(), what, [&](nb::handle x) { return dec_subproblem(o, x); });
    r.stats = dec_stats(f.next());
    r.max_effective_width = f.get<u32>();
    r.message = f.get<std::string>();
    r.fluent_slots = f.get<u32>();
    f.done();
    return r;
}

nb::tuple enc(const Owner& o, const search::BestFirstResult& r)
{
    return nb::make_tuple(status_int(r.status), enc_plan(r.plan), r.cost, state_or_none(o, r.goal_state), enc(r.stats),
                          r.evaluations, r.dead_ends, r.reopened, r.expanded_until_last_jump, r.layers, r.initial_h,
                          enc(r.heuristic), r.setup_seconds, r.algorithm, r.store, r.queue, r.message, r.fluent_slots);
}
void dec_best_first(const Owner& o, Fields& f, search::BestFirstResult& r, const char* what)
{
    r.status = f.status();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.goal_state = dec_state(o, f.next());
    r.stats = dec_stats(f.next());
    r.evaluations = f.get<u64>();
    r.dead_ends = f.get<u64>();
    r.reopened = f.get<u64>();
    r.expanded_until_last_jump = f.get<u64>();
    r.layers = f.get<u64>();
    r.initial_h = f.get<double>();
    r.heuristic = dec_heuristic_stats(f.next());
    r.setup_seconds = f.get<double>();
    r.algorithm = f.get<std::string>();
    r.store = f.get<std::string>();
    r.queue = f.get<std::string>();
    r.message = f.get<std::string>();
    r.fluent_slots = f.get<u32>();
}

nb::tuple enc(const Owner& o, const BrfsResult& r)
{
    nb::list layers;
    for (const auto& l : r.layer_counts)
        layers.append(nb::make_tuple(l[0], l[1], l[2]));
    return nb::make_tuple(status_int(r.status), r.states, r.expanded, r.generated, r.pruned, r.num_goal_states, r.layers, layers,
                          r.exhausted, r.solved, enc_plan(r.plan), r.cost, state_or_none(o, r.goal_state), r.search_s,
                          r.words, r.fluent_slots, r.memory_bytes, r.fingerprint, r.threads_used, r.store, r.message);
}
BrfsResult dec_brfs(const Owner& o, nb::handle h)
{
    const char* what = "BrFS result";
    Fields f(h, what);
    BrfsResult r;
    r.status = f.status();
    r.states = f.get<u64>();
    r.expanded = f.get<u64>();
    r.generated = f.get<u64>();
    r.pruned = f.get<u64>();
    r.num_goal_states = f.get<u64>();
    r.layers = f.get<u32>();
    r.layer_counts = dec_list<std::array<u64, 3>>(f.next(), what, [&](nb::handle x) {
        Fields l(x, what);
        std::array<u64, 3> c{l.get<u64>(), l.get<u64>(), l.get<u64>()};
        l.done();
        return c;
    });
    r.exhausted = f.get<bool>();
    r.solved = f.get<bool>();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.goal_state = dec_state(o, f.next());
    r.search_s = f.get<double>();
    r.words = f.get<u32>();
    r.fluent_slots = f.get<u32>();
    r.memory_bytes = f.get<u64>();
    r.fingerprint = f.get<u64>();
    r.threads_used = f.get<u32>();
    r.store = f.get<std::string>();
    r.message = f.get<std::string>();
    f.done();
    return r;
}

nb::tuple enc(const Owner& o, const search::RolloutIwResult& r)
{
    return nb::make_tuple(status_int(r.status), enc_plan(r.plan), r.cost, state_or_none(o, r.goal_state),
                          enc(r.rollout), r.root_solved, r.message, r.fluent_slots);
}
search::RolloutIwResult dec_rollout_iw(const Owner& o, nb::handle h)
{
    const char* what = "Rollout IW result";
    Fields f(h, what);
    search::RolloutIwResult r;
    r.status = f.status();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.goal_state = dec_state(o, f.next());
    r.rollout = dec_rollout_stats(f.next());
    r.root_solved = f.get<bool>();
    r.message = f.get<std::string>();
    r.fluent_slots = f.get<u32>();
    f.done();
    return r;
}

nb::tuple enc(const Owner& o, const search::LandingState& l)
{
    return nb::make_tuple(make_state(o, State(l.state)), enc_atoms(l.atoms), l.direct_dead_end);
}
search::LandingState dec_landing(const Owner& o, nb::handle h)
{
    const char* what = "landing state";
    Fields f(h, what);
    search::LandingState l;
    const std::optional<State> s = dec_state(o, f.next());
    if (!s)
        f.corrupt();
    l.state = *s;
    l.atoms = dec_atoms(*o.core->task, f.next(), what);
    l.direct_dead_end = f.get<bool>();
    f.done();
    return l;
}

nb::tuple enc(const Owner& o, const search::RolloutResult& r)
{
    nb::list by_atom, rows;
    for (const auto& [c, i] : r.landing_state_by_atom)
        by_atom.append(nb::make_tuple(c, i));
    for (const auto& [c, row] : r.co_occurrence)
        rows.append(nb::make_tuple(c, enc_atoms(row)));
    return nb::make_tuple(enc(o, r.search), enc_atoms(r.reached_fluent_atoms), enc_atoms(r.reached_derived_atoms), r.num_created_states,
                          enc_list(r.landing_states, [&](const auto& x) { return enc(o, x); }), by_atom, rows);
}
search::RolloutResult dec_rollout(const Owner& o, nb::handle h)
{
    const char* what = "rollout result";
    const Task& T = *o.core->task;
    Fields f(h, what);
    search::RolloutResult r;
    r.search = dec_iw(o, f.next());
    r.reached_fluent_atoms = dec_atoms(T, f.next(), what);
    r.reached_derived_atoms = dec_atoms(T, f.next(), what);
    r.num_created_states = f.get<u64>();
    r.landing_states = dec_list<search::LandingState>(f.next(), what, [&](nb::handle x) { return dec_landing(o, x); });
    auto atom = [&](Fields& p)
    {
        const u64 c = p.get<u64>();
        if (c >= T.compiled().layout.total)
            p.corrupt();
        return CanonicalAtom{c};
    };
    r.landing_state_by_atom = dec_list<std::pair<CanonicalAtom, u32>>(f.next(), what, [&](nb::handle x) {
        Fields p(x, what);
        const CanonicalAtom c = atom(p);
        const u32 i = p.get<u32>();
        p.done();
        if (i >= r.landing_states.size())
            p.corrupt();
        return std::pair<CanonicalAtom, u32>{c, i};
    });
    r.co_occurrence = dec_list<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>>(f.next(), what, [&](nb::handle x) {
        Fields p(x, what);
        const CanonicalAtom c = atom(p);
        std::vector<CanonicalAtom> row = dec_atoms(T, p.next(), what);
        p.done();
        return std::pair<CanonicalAtom, std::vector<CanonicalAtom>>{c, std::move(row)};
    });
    f.done();
    return r;
}


CoOccurrenceDict co_occurrence_dict(const Owner& o, const CoOccurrence& rows)
{
    AtomCache atoms(o);
    CoOccurrenceDict out{nb::dict()};
    for (const auto& [c, row] : rows)
        out[atoms.get(c)] = atoms.list(row);
    return out;
}

std::string status_or_none(bool valid, SearchStatus s) { return valid ? std::string(search::to_string(s)) : "None"; }

nb::tuple enc(const Owner& o, const search::PortfolioResult& r)
{
    nb::list statuses;
    for (const SearchStatus x : r.rollout_statuses)
        statuses.append(status_int(x));
    return nb::make_tuple(status_int(r.status), enc_plan(r.plan), r.cost, r.plan_length,
                          state_or_none(o, r.goal_state), r.certified_optimal,
                          r.iw_lower_bound, r.iw_completed_depth, r.winning_worker, r.message, r.certifier_ran,
                          status_int(r.certifier_status), enc(o, r.certifier),
                          enc_list(r.rollout_statistics, [](const auto& x) { return enc(x); }), statuses,
                          enc_list(r.rollout_rounds, [](u32 x) { return nb::int_(x); }), enc(r.stats), r.threads_used,
                          r.fluent_slots);
}
search::PortfolioResult dec_portfolio(const Owner& o, nb::handle h)
{
    const char* what = "portfolio result";
    Fields f(h, what);
    search::PortfolioResult r;
    r.status = f.status();
    r.plan = dec_plan(*o.core->task, f.next(), what);
    r.cost = f.get<double>();
    r.plan_length = f.get<u32>();
    r.goal_state = dec_state(o, f.next());
    r.certified_optimal = f.get<bool>();
    r.iw_lower_bound = f.get<u32>();
    r.iw_completed_depth = f.get<u32>();
    r.winning_worker = f.get<u32>();
    r.message = f.get<std::string>();
    r.certifier_ran = f.get<bool>();
    r.certifier_status = f.status();
    r.certifier = dec_iw(o, f.next());
    r.rollout_statistics = dec_list<search::RolloutIwStatistics>(f.next(), what, dec_rollout_stats);
    r.rollout_statuses = dec_list<SearchStatus>(f.next(), what, [&](nb::handle x) {
        int v = -1;
        if (!nb::try_cast(x, v) || v < 0 || v > static_cast<int>(SearchStatus::Unsolvable))
            f.corrupt();
        return static_cast<SearchStatus>(v);
    });
    r.rollout_rounds = dec_list<u32>(f.next(), what, [&](nb::handle x) {
        u32 v = 0;
        if (!nb::try_cast(x, v))
            f.corrupt();
        return v;
    });
    r.stats = dec_stats(f.next());
    r.threads_used = f.get<u32>();
    r.fluent_slots = f.get<u32>();
    f.done();
    if (r.rollout_rounds.size() != r.rollout_statuses.size())
        f.corrupt();
    return r;
}

/// (restore, (kind, task, fingerprint, fields)): the pickle of a result of `o`'s task.
nb::tuple result_reduce(const char* kind, const Owner& o, nb::object fields)
{
    return nb::make_tuple(nb::module_::import_("mymyr._pickle").attr("_restore_search_result"),
                          nb::make_tuple(kind, task_object(o.obj), o.core->task->fingerprint(), fields));
}

/// The result a result_reduce pickle describes.
nb::object restore_result(std::string_view kind, nb::handle task, u64 fingerprint, nb::handle fields)
{
    if (!nb::isinstance<PyTask>(task) && !nb::isinstance<PyHandle>(task))
        throw nb::type_error("mymyr: a pickled search result's task must be a Task");
    const Owner o = owner_of(task);
    if (o.core->task->fingerprint() != fingerprint)
        throw nb::value_error("mymyr: the search result was pickled from a different task (content fingerprint mismatch)");
    if (kind == "search_tree")
        return nb::cast(PySearchTree{dec_tree(o, fields), o});
    if (kind == "iw_pass")
        return nb::cast(PyIwPass{dec_pass(o, fields), o});
    if (kind == "siw_subproblem")
        return nb::cast(PySiwSubproblem{dec_subproblem(o, fields), o});
    if (kind == "iw")
        return nb::cast(PyIwResult{dec_iw(o, fields), o});
    if (kind == "siw")
        return nb::cast(PySiwResult{dec_siw(o, fields), o});
    if (kind == "best_first" || kind == "astar_iw" || kind == "ehc" || kind == "anytime")
    {
        const char* what = kind == "best_first" ? "best-first result"
                           : kind == "astar_iw" ? "A*-IW result"
                           : kind == "ehc"      ? "EHC result"
                                                : "anytime result";
        Fields f(fields, what);
        search::BestFirstResult r;
        dec_best_first(o, f, r, what);
        if (kind == "best_first")
        {
            f.done();
            return nb::cast(PyBestFirstResult{std::move(r), o});
        }
        if (kind == "ehc")
        {
            PyEhcResult e;
            e.r = std::move(r);
            e.o = o;
            e.ehc = dec_ehc(f.next());
            f.done();
            return nb::cast(std::move(e));
        }
        if (kind == "anytime")
        {
            PyAnytimeResult a;
            a.r = std::move(r);
            a.o = o;
            a.solutions = dec_list<search::AnytimeSolution>(f.next(), what, dec_anytime_solution);
            a.iterations = dec_list<search::AnytimeIteration>(f.next(), what, dec_anytime_iteration);
            a.optimal = f.get<bool>();
            f.done();
            return nb::cast(std::move(a));
        }
        PyAStarIwResult a;
        a.r = std::move(r);
        a.o = o;
        a.novelty = dec_novelty(f.next());
        f.done();
        return nb::cast(std::move(a));
    }
    if (kind == "brfs")
        return nb::cast(PyBrfsResult{dec_brfs(o, fields), o});
    if (kind == "rollout_iw")
        return nb::cast(PyRolloutIwResult{dec_rollout_iw(o, fields), o});
    if (kind == "landing_state")
        return nb::cast(PyLandingState{dec_landing(o, fields), o});
    if (kind == "merged_landing_states")
    {
        const char* what = "merged landing states";
        Fields f(fields, what);
        search::MergedLandingStates m;
        m.states = dec_list<State>(f.next(), what, [&](nb::handle x) {
            const std::optional<State> st = dec_state(o, x);
            if (!st)
                f.corrupt();
            return *st;
        });
        m.by_rollout = dec_list<std::vector<u32>>(f.next(), what, [&](nb::handle x) {
            return dec_list<u32>(x, what, [&](nb::handle i) {
                u32 v = 0;
                if (!nb::try_cast(i, v) || v >= m.states.size())
                    f.corrupt();
                return v;
            });
        });
        f.done();
        return nb::cast(PyMergedLandingStates{std::move(m), o});
    }
    if (kind == "rollouts" || kind == "rollout")
    {
        // a batch, or one rollout (a batch of one)
        const char* what = kind == "rollouts" ? "parallel rollouts" : "rollout result";
        Fields f(fields, what);
        auto batch = std::make_shared<search::ParallelRolloutsResult>();
        std::vector<u64> seeds;
        if (kind == "rollout")
        {
            seeds.push_back(f.get<u64>());
            batch->rollouts.push_back(dec_rollout(o, f.next()));
            f.done();
            return nb::cast(PyRolloutResult{std::move(batch), 0, seeds[0], o});
        }
        batch->rollouts = dec_list<search::RolloutResult>(f.next(), what, [&](nb::handle x) { return dec_rollout(o, x); });
        seeds = dec_list<u64>(f.next(), what, [&](nb::handle x) {
            u64 v = 0;
            if (!nb::try_cast(x, v))
                f.corrupt();
            return v;
        });
        batch->threads_used = f.get<u32>();
        batch->message = f.get<std::string>();
        f.done();
        if (seeds.size() != batch->rollouts.size())
            f.corrupt();
        return nb::cast(PyParallelRollouts{std::move(batch), std::move(seeds), o});
    }
    if (kind == "portfolio")
        return nb::cast(PyPortfolioResult{dec_portfolio(o, fields), o});
    throw nb::value_error("mymyr: unknown search result pickle kind");
}

/// __getstate__ / __setstate__ of a statistics class (plain values, no task).
template<class T, class Enc, class Dec>
void bind_stats_pickle(nb::class_<T>& c, Enc enc_fn, Dec dec_fn)
{
    c.def("__getstate__", [enc_fn](const T& x) { return enc_fn(x); });
    c.def("__setstate__", [dec_fn](T* self, nb::tuple t) { new (self) T(dec_fn(t)); });
}

// ------------------------------------------------------------------------------------------------ parallel rollouts and portfolio: options

search::ActionOrdering parse_ordering(nb::handle h)
{
    if (nb::isinstance<search::ActionOrdering>(h))
        return nb::cast<search::ActionOrdering>(h);
    const std::string s = str_arg(h, "ordering");
    if (s == "in_order")
        return search::ActionOrdering::InOrder;
    if (s == "randomized")
        return search::ActionOrdering::Randomized;
    if (s == "direct_goal_achiever_first")
        return search::ActionOrdering::DirectGoalAchieverFirst;
    if (s == "goal_regression_relevance")
        return search::ActionOrdering::GoalRegressionRelevance;
    if (s == "mixed_regression_random")
        return search::ActionOrdering::MixedRegressionRandom;
    throw nb::value_error("mymyr: ordering must be an ActionOrdering or one of 'in_order', 'randomized', "
                          "'direct_goal_achiever_first', 'goal_regression_relevance', 'mixed_regression_random'");
}

search::LayerOrdering parse_layers(nb::handle kind, nb::handle seed, nb::handle max_next_layer_states,
                                   bool prefer_more_satisfied_goals, nb::handle beam_width, nb::handle beam_novelty,
                                   bool randomize_ties, nb::handle beam_chunk)
{
    search::LayerOrdering l;
    const std::string k = str_arg(kind, "layer_order");
    if (k == "queue")
        l.kind = search::LayerOrdering::Kind::Queue;
    else if (k == "in_order")
        l.kind = search::LayerOrdering::Kind::InOrder;
    else if (k == "reverse")
        l.kind = search::LayerOrdering::Kind::Reverse;
    else if (k == "randomized")
        l.kind = search::LayerOrdering::Kind::Randomized;
    else if (k == "goal_count")
        l.kind = search::LayerOrdering::Kind::GoalCount;
    else
        throw nb::value_error(
            "mymyr: layer_order must be 'queue', 'in_order', 'reverse', 'randomized' or 'goal_count'");
    l.seed = int_arg<u64>(seed, "seed");
    l.prefer_more_satisfied_goals = prefer_more_satisfied_goals;
    if (auto v = opt_int_arg<u32>(max_next_layer_states, "max_next_layer_states", 0))
        l.max_next_layer_states = *v;
    if (auto v = opt_int_arg<u32>(beam_width, "beam_width", 0))
        l.beam_width = *v;
    const std::string bn = str_arg(beam_novelty, "beam_novelty");
    if (bn == "all_tested")
        l.beam_novelty = search::LayerOrdering::BeamNovelty::AllTested;
    else if (bn == "survivors_only")
        l.beam_novelty = search::LayerOrdering::BeamNovelty::SurvivorsOnly;
    else if (bn == "relaxed_survivors_only")
        l.beam_novelty = search::LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly;
    else
        throw nb::value_error("mymyr: beam_novelty must be 'all_tested', 'survivors_only' or 'relaxed_survivors_only'");
    l.randomize_ties = randomize_ties;
    l.beam_chunk = int_arg<u32>(beam_chunk, "beam_chunk");
    return l;
}

/// Landmark novelty options: `graph` is None, a FactLandmarkGraph of the task, or 'approximate' / 'lifted' (computed
/// here with default options, with the thread state detached).
search::LandmarkNovelty parse_landmarks(const Owner& o, nb::handle graph, bool disjunctive, bool all_private,
                                        nb::handle unshared_atoms, nb::handle max_dense_bytes)
{
    search::LandmarkNovelty lm;
    lm.disjunctive = disjunctive;
    lm.all_private = all_private;
    if (!unshared_atoms.is_none())
        for (nb::handle a : unshared_atoms)
            if (const auto c = canonical_atom(o, a, false))
                lm.unshared_atoms.push_back(*c);
    if (auto v = opt_int_arg<u64>(max_dense_bytes, "max_dense_bytes", 0))
        lm.tables.max_dense_bytes = *v;
    if (graph.is_none())
        return lm;
    if (nb::isinstance<PyFactLandmarkGraph>(graph))
    {
        const PyFactLandmarkGraph& g = *nb::inst_ptr<PyFactLandmarkGraph>(graph);
        same_task(o, g.o, "the landmark graph");
        lm.graph = g.g;
        return lm;
    }
    const std::string kind = str_arg(graph, "landmarks");
    if (kind != "approximate" && kind != "lifted")
        throw nb::value_error("mymyr: landmarks must be a FactLandmarkGraph, 'approximate', 'lifted' or None");
    const Task& t = *o.core->task;
    if (t.numeric_slots() > 0)
        return lm;  // the search refuses the task itself (ValueError)
    nb::gil_scoped_release release;
    lm.graph = std::make_shared<const landmarks::FactLandmarkGraph>(kind == "approximate" ? landmarks::approximate_fact_landmarks(t)
                                                                                          : landmarks::lifted_fact_landmarks(t));
    return lm;
}

std::vector<search::RolloutOrdering> parse_orderings(nb::handle seq)
{
    std::vector<search::RolloutOrdering> out;
    if (seq.is_none())
        return out;
    for (nb::handle x : seq)
    {
        if (nb::isinstance<nb::tuple>(x))
        {
            nb::tuple t = nb::borrow<nb::tuple>(x);
            if (nb::len(t) != 2)
                throw nb::value_error("mymyr: a rollout ordering is an ordering or a pair (ordering, seed)");
            out.push_back({parse_ordering(t[0]), int_arg<u64>(t[1], "a rollout ordering's seed")});
        }
        else
            out.push_back({parse_ordering(x), 0});
    }
    return out;
}

/// The rollouts of a batch, or of any rollout results of one task (copied).
std::vector<search::RolloutResult> collect_rollouts(nb::handle results, Owner& owner,
                                                    std::shared_ptr<const search::ParallelRolloutsResult>& batch)
{
    std::vector<search::RolloutResult> out;
    if (nb::isinstance<PyParallelRollouts>(results))
    {
        const PyParallelRollouts& p = *nb::inst_ptr<PyParallelRollouts>(results);
        owner = p.o;
        batch = p.r;
        return out;
    }
    for (nb::handle x : results)
    {
        if (!nb::isinstance<PyRolloutResult>(x))
            throw nb::type_error("mymyr: expected a ParallelRolloutsResult or a sequence of RolloutResult");
        const PyRolloutResult& r = *nb::inst_ptr<PyRolloutResult>(x);
        if (!owner.core)
            owner = r.o;
        else
            same_task(owner, r.o, "a rollout result");
        out.push_back(r.get());
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ heuristic objects

/// A heuristic usable from Python. Calls on the object are serialized by a mutex (a Heuristic holds scratch); for
/// parallel evaluation give each thread its own object (they can share the grounding: Heuristic(task, share=other)).
/// A search given the object evaluates with an evaluator of its own (make_evaluator), so it never holds the mutex: the
/// object stays callable while the search runs, from the search's callbacks and from other threads.
struct PyHeuristic
{
    std::unique_ptr<heuristics::Heuristic> h;  // evaluates the calls on the object
    Owner o;
    heuristics::Options options;  // h's options, with h's grounding (options.relaxed)
    std::mutex m;
    datasets::StateSpacePtr space{};  // Heuristic.perfect: the state space
    nb::object space_object{};        // Heuristic.perfect: the StateSpace it was given (its pickle)

    /// A new evaluator of h's values with scratch of its own, sharing h's grounding or state space.
    [[nodiscard]] std::unique_ptr<heuristics::Heuristic> make_evaluator() const
    {
        if (space)
            return heuristics::perfect(space, options.costs);
        return heuristics::make_heuristic(*o.core->task, options);
    }
};

/// A heuristic written in Python (heuristics::Kind::Custom): a callable h(state) -> float, or an object with
/// evaluate(state) -> float and/or evaluate_batch(states) -> sequence of floats, and optionally preferred_actions()
/// -> iterable of actions (the preferred operators of the state of the last single evaluation). +inf marks a dead
/// end. Every call attaches the thread state. The first exception is recorded in `errors`, which cancels the search
/// and re-raises it afterwards; evaluations after it return +inf without calling Python.
class PyCallbackHeuristic final : public heuristics::Heuristic
{
public:
    PyCallbackHeuristic(nb::handle obj, Owner owner, CallbackErrors& errors) : m_owner(std::move(owner)), m_errors(errors)
    {
        const bool one = nb::hasattr(obj, "evaluate"), batch = nb::hasattr(obj, "evaluate_batch");
        if (one || batch)
        {
            if (one)
                m_one = nb::getattr(obj, "evaluate");
            if (batch)
                m_batch = nb::getattr(obj, "evaluate_batch");
            if (nb::hasattr(obj, "preferred_actions"))
                m_preferred = nb::getattr(obj, "preferred_actions");
        }
        else if (PyCallable_Check(obj.ptr()))
            m_one = nb::borrow(obj);
        else
            throw nb::type_error("mymyr: heuristic must be a kind name, a mymyr.search.Heuristic, a callable "
                                 "state -> float or an object with evaluate(state) or evaluate_batch(states)");
    }
    PyCallbackHeuristic(const PyCallbackHeuristic&) = delete;
    PyCallbackHeuristic& operator=(const PyCallbackHeuristic&) = delete;
    ~PyCallbackHeuristic() override
    {
        nb::gil_scoped_acquire acquire;
        m_one.reset();
        m_batch.reset();
        m_preferred.reset();
        m_owner.obj.reset();
    }

    [[nodiscard]] heuristics::Kind kind() const noexcept override { return heuristics::Kind::Custom; }

    heuristics::Value evaluate(StateView s) override
    {
        if (m_errors.failed.load(std::memory_order_relaxed))
            return inf();
        nb::gil_scoped_acquire acquire;
        try
        {
            heuristics::Value v = 0;
            if (m_one.is_valid())
                v = to_value(m_one(make_state(m_owner, State(s))).ptr());
            else
            {
                nb::list states;
                states.append(make_state(m_owner, State(s)));
                values(m_batch(states), std::span<heuristics::Value>(&v, 1));
            }
            if (m_preferred.is_valid())
                load_preferred(m_preferred());
            count(v);
            return v;
        }
        catch (nb::python_error& e)
        {
            m_errors.record(std::move(e));
            m_pref.clear();
            return inf();
        }
    }

    /// The goals of GoalSpec::AnyOf are not passed on: a Python heuristic estimates whatever it estimates.
    heuristics::Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> /*goals*/) override { return evaluate(s); }

    void evaluate_batch(std::span<const StateView> states, std::span<heuristics::Value> out) override
    {
        if (!m_batch.is_valid())
        {
            for (usize i = 0; i < states.size(); ++i)
                out[i] = evaluate(states[i]);
            return;
        }
        if (m_errors.failed.load(std::memory_order_relaxed))
        {
            std::ranges::fill(out, inf());
            return;
        }
        nb::gil_scoped_acquire acquire;
        try
        {
            nb::list list;
            for (StateView s : states)
                list.append(make_state(m_owner, State(s)));
            values(m_batch(list), out);
            for (heuristics::Value v : out)
                count(v);
        }
        catch (nb::python_error& e)
        {
            m_errors.record(std::move(e));
            std::ranges::fill(out, inf());
        }
    }

    [[nodiscard]] bool batched() const noexcept override { return m_batch.is_valid(); }
    [[nodiscard]] bool provides_preferred() const noexcept override { return m_preferred.is_valid(); }
    [[nodiscard]] bool preferred(const ActionLabel& a) const override
    {
        const auto less = [](const Action& x, const ActionLabel& y) {
            if (x.schema != y.schema)
                return x.schema < y.schema;
            return std::lexicographical_compare(x.binding.begin(), x.binding.end(), y.binding.begin(), y.binding.end());
        };
        const auto it = std::lower_bound(m_pref.begin(), m_pref.end(), a, less);
        return it != m_pref.end() && it->schema == a.schema && std::ranges::equal(it->binding, a.binding);
    }

private:
    static heuristics::Value inf() { return std::numeric_limits<heuristics::Value>::infinity(); }

    void count(heuristics::Value v)
    {
        ++m_stats.evaluations;
        if (std::isinf(v) && v > 0)
            ++m_stats.dead_ends;
    }

    /// The float of a Python number (anything with __float__); raises (as a python_error) for non-numbers and for
    /// values that are no heuristic values: NaN and negative numbers (inf marks a dead end).
    static heuristics::Value to_value(PyObject* x)
    {
        const double v = PyFloat_AsDouble(x);
        if (v == -1.0 && PyErr_Occurred())
        {
            if (!PyErr_ExceptionMatches(PyExc_TypeError))
                throw nb::python_error();
            PyErr_Clear();
            const std::string text = "mymyr: a Python heuristic returned a " +
                                     nb::cast<std::string>(nb::str(nb::handle(x).type().attr("__name__"))) +
                                     "; heuristic values are numbers >= 0 (inf: a dead end)";
            PyErr_SetString(PyExc_TypeError, text.c_str());
            throw nb::python_error();
        }
        if (std::isnan(v) || v < 0)
        {
            const std::string text = "mymyr: a Python heuristic returned " + nb::cast<std::string>(nb::repr(nb::float_(v))) +
                                     "; heuristic values are >= 0 (inf: a dead end)";
            PyErr_SetString(PyExc_ValueError, text.c_str());
            throw nb::python_error();
        }
        return v;
    }

    static void values(nb::handle result, std::span<heuristics::Value> out)
    {
        nb::object seq = nb::steal(PySequence_Fast(result.ptr(), "mymyr: evaluate_batch must return a sequence of floats"));
        if (!seq.is_valid())
            throw nb::python_error();
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.ptr());
        if (static_cast<usize>(n) != out.size())
        {
            PyErr_Format(PyExc_ValueError, "mymyr: evaluate_batch returned %zd values for %zu states", n, out.size());
            throw nb::python_error();
        }
        PyObject** items = PySequence_Fast_ITEMS(seq.ptr());
        for (Py_ssize_t i = 0; i < n; ++i)
            out[static_cast<usize>(i)] = to_value(items[i]);
    }

    void load_preferred(nb::handle actions)
    {
        m_pref.clear();
        for (nb::handle a : actions)
        {
            if (!nb::isinstance<PyAction>(a))
            {
                PyErr_SetString(PyExc_TypeError, "mymyr: preferred_actions must return mymyr Actions");
                throw nb::python_error();
            }
            const PyAction& p = *nb::inst_ptr<PyAction>(a);
            if (p.core != m_owner.core)
            {
                PyErr_SetString(PyExc_ValueError, "mymyr: preferred_actions returned an action of another task");
                throw nb::python_error();
            }
            std::vector<ObjectId> binding(p.binding.size());
            for (usize i = 0; i < binding.size(); ++i)
                binding[i] = ObjectId{p.binding[i]};
            m_pref.emplace_back(SchemaId{p.schema}, std::move(binding));
        }
        std::ranges::sort(m_pref);
    }

    Owner m_owner;
    CallbackErrors& m_errors;
    nb::object m_one, m_batch, m_preferred;
    std::vector<Action> m_pref;  // sorted
};

// ------------------------------------------------------------------------------------------------ parameter types

// Parameter types for signatures and the stubs (Arg: typing.hpp).
using TaskArg = Arg<std::variant<PyTask, PyHandle>>;
using StateArg = Arg<PyState>;
using StrArg = Arg<std::string>;
using WidthZeroArg = Arg<ann::WidthZero>;
using SeedsArg = Arg<std::vector<u64>>;
using CancelArg = Arg<PyCancelToken>;
using AtomsArg = nb::typed<nb::sequence, AtomLike>;
/// One goal: a GroundCondition, or the ground literals and atoms of one (Task.ground_condition).
using OneGoal = std::variant<PyGroundCondition, nb::typed<nb::sequence, GroundLiteralLike>>;
using GoalArg = Arg<std::variant<nb::typed<nb::callable, bool(PyState)>, PyGroundCondition, nb::typed<nb::sequence, OneGoal>>>;
using StatesArg = Arg<nb::typed<nb::iterable, PyState>>;
using ObserverArg = Arg<ann::Observer>;  // duck-typed: every on_* method is optional
using HeuristicArg =
    Arg<std::variant<std::string, PyHeuristic, nb::typed<nb::callable, double(PyState)>, ann::HeuristicObject>>;
using ActionList = nb::typed<nb::list, PyAction>;
using TransitionArg = Arg<PyLandmarkOrdering>;
using LandmarksArg = Arg<std::variant<PyFactLandmarkGraph, ann::LandmarkSource>>;
using FluentAtomsArg = Arg<nb::typed<nb::iterable, AtomLike>>;
using LayerArg = Arg<ann::LayerOrder>;
using WeightsArg = Arg<nb::typed<nb::sequence, double>>;
using HelpfulArg = Arg<ann::HelpfulActions>;
using FallbackArg = Arg<ann::EhcFallback>;
using BeamNoveltyArg = Arg<ann::BeamNovelty>;
using OrderingArg = Arg<std::variant<search::ActionOrdering, std::string>>;
using OrderingsArg =
    Arg<nb::typed<nb::sequence, std::variant<search::ActionOrdering, std::string,
                                             nb::typed<nb::tuple, std::variant<search::ActionOrdering, std::string>, int>>>>;
using RolloutsArg = Arg<std::variant<PyParallelRollouts, nb::typed<nb::sequence, PyRolloutResult>>>;
using AtomList = nb::typed<nb::list, PyGroundAtom>;

/// The words of a status of a path-dependent heuristic, from its bytes (Heuristic.initial_status / progress).
std::vector<u64> status_of(const PyHeuristic& self, nb::handle status, const char* what)
{
    if (!nb::isinstance<nb::bytes>(status))
        throw nb::type_error((std::string("mymyr: ") + what + " must be bytes (a status from initial_status or progress)").c_str());
    const auto b = nb::borrow<nb::bytes>(status);
    const usize n = self.h->status_words();
    if (b.size() != n * sizeof(u64))
        throw nb::value_error((std::string("mymyr: ") + what + " has " + std::to_string(b.size()) +
                               " bytes; the statuses of this heuristic have " + std::to_string(n * sizeof(u64)))
                                  .c_str());
    std::vector<u64> w(n);
    if (n)
        std::memcpy(w.data(), b.c_str(), n * sizeof(u64));
    return w;
}

nb::bytes status_bytes(const std::vector<u64>& w)
{
    return nb::bytes(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(u64));
}

double py_evaluate(PyHeuristic& self, StateArg state, Arg<nb::bytes> status)
{
    const State s = state_for(self.o, state, "state");
    std::optional<std::vector<u64>> w;
    if (!status.is_none())
        w = status_of(self, status, "status");
    nb::gil_scoped_release release;
    std::lock_guard lock(self.m);
    return w ? self.h->evaluate(s.view(), *w) : self.h->evaluate(s.view());
}

/// heuristic='perfect': h* from the task's state space, generated within the search's state and time budgets.
std::unique_ptr<heuristics::Heuristic> perfect_of_task(const Owner& o, heuristics::Costs costs, const search::Budget& budget)
{
    datasets::StateSpaceOptions so;
    so.remove_if_unsolvable = false;
    so.labels = false;
    so.max_states = budget.max_states;
    so.max_seconds = budget.max_seconds;
    datasets::StateSpaceResult r;
    {
        nb::gil_scoped_release release;
        r = datasets::generate_state_space(o.core->task, so);
    }
    if (!r.space)
        throw nb::value_error((std::string("mymyr: heuristic='perfect' needs the task's whole state space, and its generation stopped (") +
                               search::to_string(r.status) + " after " + std::to_string(r.states) +
                               " states); raise max_states / max_seconds or pass Heuristic.perfect(space)")
                                  .c_str());
    return heuristics::perfect(std::move(r.space), costs);
}

// ------------------------------------------------------------------------------------------------ best-first

template<class Search>
auto best_first(Search search_fn, nb::handle task, nb::handle heuristic, nb::handle costs,
                             nb::handle evaluation, nb::handle store, nb::handle queue, nb::handle start, bool reopen,
                             bool lazy_requeue, bool preferred_operators, u32 preferred_weight, u32 standard_weight,
                             u32 beam_width, bool witness_pruning, nb::handle max_states,
                             nb::handle max_expanded,
                             nb::handle max_depth, nb::handle max_seconds, nb::handle cancel, nb::handle goal,
                             nb::handle blocked_states, nb::handle observer, nb::handle progress_interval)
{
    const Owner o = owner_of(task);
    ControlScope cs;
    fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states, observer,
                 progress_interval);
    search::BestFirstOptions opts;
    opts.control = cs.control;
    std::unique_ptr<heuristics::Heuristic> own;  // the search's evaluator of a Heuristic object
    std::unique_ptr<PyCallbackHeuristic> callback;
    std::unique_ptr<heuristics::Heuristic> perfect;
    if (nb::isinstance<nb::str>(heuristic) && nb::cast<std::string>(heuristic) == "perfect")
    {
        perfect = perfect_of_task(o, parse_costs(costs), cs.control.budget);
        opts.evaluator = perfect.get();
    }
    else if (nb::isinstance<PyHeuristic>(heuristic))
    {
        const PyHeuristic& shared = *nb::inst_ptr<PyHeuristic>(heuristic);
        if (shared.o.core->task->uid() != o.core->task->uid())
            throw nb::value_error("mymyr: the heuristic belongs to another task");
        {
            nb::gil_scoped_release release;
            own = shared.make_evaluator();
        }
        opts.evaluator = own.get();
    }
    else if (nb::isinstance<nb::str>(heuristic))
        opts.heuristic.kind = parse_kind(heuristic);
    else
    {
        callback = std::make_unique<PyCallbackHeuristic>(heuristic, o, *cs.errors);
        opts.evaluator = callback.get();
    }
    opts.heuristic.costs = parse_costs(costs);
    opts.heuristic.evaluation = parse_evaluation(evaluation);
    const std::string st = str_arg(store, "store");
    if (st == "auto")
        opts.store = search::BestFirstOptions::Store::Auto;
    else if (st == "flat")
        opts.store = search::BestFirstOptions::Store::Flat;
    else if (st == "chunked")
        opts.store = search::BestFirstOptions::Store::Chunked;
    else if (st == "compact")
        opts.store = search::BestFirstOptions::Store::Compact;
    else
        throw nb::value_error("mymyr: store must be 'auto', 'flat', 'chunked' or 'compact'");
    const std::string q = str_arg(queue, "queue");
    if (q == "auto")
        opts.queue = search::BestFirstOptions::Queue::Auto;
    else if (q == "bucket")
        opts.queue = search::BestFirstOptions::Queue::Bucket;
    else if (q == "heap")
        opts.queue = search::BestFirstOptions::Queue::Heap;
    else
        throw nb::value_error("mymyr: queue must be 'auto', 'bucket' or 'heap'");
    if (!start.is_none())
        opts.start = state_for(o, start, "start");
    opts.reopen = reopen;
    opts.lazy_requeue = lazy_requeue;
    opts.preferred_operators = preferred_operators;
    opts.preferred_weight = preferred_weight;
    opts.standard_weight = standard_weight;
    opts.beam_width = beam_width;
    opts.witness_pruning = witness_pruning;
    const Task& t = *o.core->task;
    auto r = run_detached(cs, [&] { return search_fn(t, opts); });
    if constexpr (std::is_same_v<decltype(r), search::AStarIwResult>)
    {
        const auto novelty = r.novelty;
        return PyAStarIwResult{{std::move(r), o}, novelty};
    }
    else if constexpr (std::is_same_v<decltype(r), search::EhcResult>)
    {
        const auto ehc = r.ehc;
        return PyEhcResult{{std::move(r), o}, ehc};
    }
    else if constexpr (std::is_same_v<decltype(r), search::AnytimeResult>)
    {
        auto solutions = std::move(r.solutions);
        auto iterations = std::move(r.iterations);
        const bool optimal = r.optimal;
        return PyAnytimeResult{{std::move(r), o}, std::move(solutions), std::move(iterations), optimal};
    }
    else
        return PyBestFirstResult{std::move(r), o};
}

using BfFn = search::BestFirstResult (*)(const Task&, const search::BestFirstOptions&);

PyBestFirstResult bf_call(BfFn fn, nb::handle task, nb::handle heuristic, nb::handle costs, nb::handle evaluation,
                          nb::handle store, nb::handle queue, nb::handle start, bool reopen, bool lazy_requeue,
                          bool preferred_operators, u32 preferred_weight, u32 standard_weight, u32 beam_width,
                          bool witness_pruning, nb::handle max_states, nb::handle max_expanded, nb::handle max_depth,
                          nb::handle max_seconds, nb::handle cancel, nb::handle goal, nb::handle blocked_states,
                          nb::handle observer, nb::handle progress_interval)
{
    return best_first(fn, task, heuristic, costs, evaluation, store, queue, start, reopen, lazy_requeue,
                      preferred_operators, preferred_weight, standard_weight, beam_width, witness_pruning, max_states,
                      max_expanded, max_depth, max_seconds, cancel, goal, blocked_states, observer, progress_interval);
}

// ------------------------------------------------------------------------------------------------ IW / SIW

search::TreeRecord tree_record(nb::handle h)
{
    if (h.is_none())
        return search::TreeRecord::None;
    if (!nb::isinstance<nb::str>(h))
        throw nb::type_error("mymyr: tree must be None, 'nodes' or 'transitions'");
    const std::string value = nb::cast<std::string>(h);
    if (value == "nodes")
        return search::TreeRecord::Nodes;
    if (value == "transitions")
        return search::TreeRecord::Transitions;
    throw nb::value_error("mymyr: tree must be None, 'nodes' or 'transitions'");
}

search::IwOptions iw_options(ControlScope& cs, const Owner& o, u32 max_width, nb::handle width_zero, bool optimize_iw1,
                             bool witness_pruning, bool canonical_order, nb::handle start,
                             nb::handle transition_ordering = nb::none())
{
    search::IwOptions opts;
    opts.control = cs.control;
    opts.max_width = max_width;
    const std::string wz = str_arg(width_zero, "width_zero");
    if (wz == "expand_depth_one")
        opts.width_zero = search::WidthZero::ExpandDepthOne;
    else if (wz == "root_only")
        opts.width_zero = search::WidthZero::RootOnly;
    else
        throw nb::value_error("mymyr: width_zero must be 'expand_depth_one' or 'root_only'");
    opts.optimize_iw1 = optimize_iw1;
    opts.witness_pruning = witness_pruning;
    opts.canonical_order = canonical_order;
    if (!start.is_none())
        opts.start = state_for(o, start, "start");
    if (!transition_ordering.is_none())
    {
        if (!nb::isinstance<PyLandmarkOrdering>(transition_ordering))
            throw nb::type_error("mymyr: transition_ordering must be a mymyr.search.LandmarkTransitionOrdering");
        const PyLandmarkOrdering& t = *nb::inst_ptr<PyLandmarkOrdering>(transition_ordering);
        same_task(o, t.graph.o, "the transition ordering");
        opts.transition_ordering = t.t;
    }
    return opts;
}

/// Abstracted IW options without width, base_abstracted and preserve_goal_atoms (abstracted_iw and projective_iw set
/// them).
search::AbstractedIwOptions aiw_options(ControlScope& cs, const Owner& o, bool keep_depth_one_novel, nb::handle landmarks,
                                        bool disjunctive, bool all_private, nb::handle unshared_atoms,
                                        nb::handle max_dense_bytes, bool preserve_landmark_atoms, nb::handle layer_order,
                                        nb::handle seed, nb::handle max_next_layer_states, bool prefer_more_satisfied_goals,
                                        nb::handle beam_width, nb::handle beam_novelty, bool randomize_ties,
                                        nb::handle beam_chunk, nb::handle threads, bool witness_pruning, bool canonical_order,
                                        nb::handle start)
{
    search::AbstractedIwOptions opts;
    opts.control = cs.control;
    opts.keep_depth_one_novel = keep_depth_one_novel;
    opts.landmarks = parse_landmarks(o, landmarks, disjunctive, all_private, unshared_atoms, max_dense_bytes);
    opts.preserve_landmark_atoms = preserve_landmark_atoms;
    opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
    opts.threads = threads_arg(threads);
    opts.witness_pruning = witness_pruning;
    opts.canonical_order = canonical_order;
    if (!start.is_none())
        opts.start = state_for(o, start, "start");
    return opts;
}

#define MYMYR_CONTROL_ARGS                                                                                             \
    "max_states"_a = nb::none(), "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(),                            \
        "max_seconds"_a = nb::none(), "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), \
        "observer"_a = nb::none(), "progress_interval"_a = nb::none()

#define MYMYR_IW_ARGS                                                                                                  \
    "task"_a, nb::kw_only(), "max_width"_a = 2, "width_zero"_a = "expand_depth_one", "optimize_iw1"_a = true,          \
        "witness_pruning"_a = false, "canonical_order"_a = true, "start"_a = nb::none(), MYMYR_CONTROL_ARGS,          \
        "transition_ordering"_a = nb::none(), MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG, "tree"_a = nb::none()

#define MYMYR_LANDMARK_ARGS                                                                                            \
    "landmarks"_a = nb::none(), "disjunctive"_a = false, "all_private"_a = false, "unshared_atoms"_a = nb::none(),     \
        "max_dense_bytes"_a = nb::none()

#define MYMYR_LAYER_ARGS                                                                                               \
    "layer_order"_a = "queue", "seed"_a = 0, "max_next_layer_states"_a = nb::none(),                                  \
        "prefer_more_satisfied_goals"_a = true, "beam_width"_a = nb::none(), "beam_novelty"_a = "all_tested",         \
        "randomize_ties"_a = false, "beam_chunk"_a = 1024

/// The IW family's thread count (IwOptions::threads, AbstractedIwOptions::threads), after the layer arguments.
#define MYMYR_THREADS_ARG "threads"_a = 1

/// A docstring of paragraphs: each part trimmed of surrounding spaces, the parts joined by blank lines.
std::string paragraphs(std::initializer_list<std::string_view> parts)
{
    std::string out;
    for (std::string_view p : parts)
    {
        while (!p.empty() && p.front() == ' ')
            p.remove_prefix(1);
        while (!p.empty() && p.back() == ' ')
            p.remove_suffix(1);
        if (p.empty())
            continue;
        if (!out.empty())
            out += "\n\n";
        out += p;
    }
    return out;
}

const char* k_control_doc =
    "Common keyword arguments:\n"
    "    max_states, max_expanded, max_depth, max_seconds: budgets; None is unlimited. max_states counts the "
    "stored states, the start state included; the search stops with OUT_OF_STATES at the new state that fills "
    "them, inside its parent's expansion.\n"
    "    cancel: a CancelToken, requestable from any thread.\n"
    "    goal: None (the task's goal); a callable state -> bool; a GroundCondition; or a sequence of goals, "
    "any of which counts, each a GroundCondition or a sequence of ground literals and atoms as "
    "Task.ground_condition takes them (static, fluent and derived literals of either polarity and numeric "
    "constraints).\n"
    "    blocked_states: States never entered.\n"
    "    observer: an object with any of on_start(state), on_expand(id, state), on_generate(parent, action, "
    "child, state, is_new), on_prune(parent, action, state), on_pass(width, stats), on_solution(plan, cost), "
    "on_progress(stats) -> bool, on_end(status, stats).\n"
    "    progress_interval: expansions between on_progress calls.\n"
    "\n"
    "The search runs with the thread state detached; callbacks attach it. An exception in a callback cancels "
    "the search and is re-raised.";

const char* k_heuristic_doc =
    "Python heuristics: a callable state -> float or an object with evaluate(state) -> float "
    "(mymyr._typing.HeuristicObject); math.inf marks a dead end. Optional methods: evaluate_batch(states) -> "
    "one float per state (eager A* and eager GBFS then evaluate the new successors of an expansion in one "
    "call, beam search those of a layer; an object may have evaluate_batch alone) and preferred_actions() -> "
    "the preferred actions of the state of the last evaluate call (the lazy searches' preferred list). It is "
    "called on the thread running the search, with the thread state attached; an exception cancels the search "
    "and is re-raised.";

const char* k_tree_doc =
    "tree: None (default), 'nodes' or 'transitions'. Retains each pass as a SearchTree of read-only NumPy arrays. "
    "Node recording keeps the fast novelty path; transition recording materializes and deduplicates every successor. "
    "Every row's goal flag uses this search's goal; a custom goal is also called for rows the search did not test. "
    "Partial trees survive budget and cancellation stops. Recorded beams use the serial step.";

const char* k_transition_doc =
    "The observer's on_transition(parent, action, child, state, outcome) is called for every transition, with the "
    "most specific TransitionOutcome (BLOCKED and SELF_LOOP before the other reasons; child None when the successor "
    "got no id).";

const char* k_iw_family_doc = "Numeric tasks raise ValueError.";

const char* k_parallel_doc =
    "Parallel hooks (the make_worker protocol): an observer with make_worker(k) -> observer | None, or a goal "
    "callable with make_worker(k) -> callable | None, gives worker k its own object, called from that worker's "
    "thread (the lifecycle events on_start, on_pass, on_solution, on_end stay on the root observer, on the "
    "calling thread). Without make_worker (or when it returns None) the search runs on the calling thread "
    "alone.";

const char* k_landmark_doc =
    "Landmark arguments:\n"
    "    landmarks: a FactLandmarkGraph of the task, 'approximate' or 'lifted' (approximate_fact_landmarks / "
    "lifted_fact_landmarks with default options), or None (no landmarks: every state's coordinate is BOT).\n"
    "    disjunctive: a rank per disjunctive landmark.\n"
    "    all_private: with disjunctive, a rank per member.\n"
    "    unshared_atoms: with disjunctive, members given their own rank.\n"
    "    max_dense_bytes: the budget of the dense table, default 256 MiB.";

const char* k_layer_doc =
    "Layer arguments:\n"
    "    layer_order: 'queue' (plain BrFS, the default), or layer by layer with each next layer reordered "
    "before it is expanded: 'in_order' (as generated), 'reverse', 'randomized' (seed: the SplitMix64 seed) or "
    "'goal_count' (stable, by the number of satisfied goal literals, most first; "
    "prefer_more_satisfied_goals=False: fewest first).\n"
    "    max_next_layer_states (ordered kinds only): once the next layer holds that many states, the rest of "
    "the current layer is not expanded.\n"
    "    beam_width (ordered kinds only, not with max_next_layer_states): the whole next layer is generated, "
    "ordered, and only its first beam_width states are kept (goal_count: the best scores, ties in generation "
    "order or, with randomize_ties, by random tokens drawn from seed); the others are never expanded.\n"
    "    beam_novelty (with beam_width): 'all_tested' (the novelty test marks every successor it admits, kept "
    "or not), 'survivors_only' (successors are tested read-only; the kept ones are replayed in rank order and "
    "only those that still add a tuple stay and mark it, so a layer can end up smaller than beam_width; not "
    "supported by liw; in brfs the two modes are the same) or 'relaxed_survivors_only' (see beam_chunk).\n"
    "    beam_chunk (beam_novelty='relaxed_survivors_only', goal_count only, not supported by liw): the "
    "layer's transitions, in generation order, are split into min(threads, ceil(n / beam_chunk)) contiguous "
    "parts; each part keeps its best beam_width successors by the read-only test, and the kept ones of all "
    "parts enter in rank order, the states seen before skipped, until beam_width entered; then the "
    "survivors_only replay. The parts make the result depend on the thread count; it is deterministic for a "
    "thread count and seed.\n"
    "    threads (brfs and the IW family; > 1 needs a beam): the beam's layer step on that many threads (0: "
    "one per hardware thread), with the result of one thread for 'all_tested' and 'survivors_only'; an "
    "observer, blocked states or a successor order keep it on the calling thread.";
}  // namespace

void bind_search(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_search", "The search family and heuristics (mymyr.search)");

    nb::enum_<SearchStatus>(m, "Status", "Outcome of a search.")
        .value("SOLVED", SearchStatus::Solved)
        .value("EXHAUSTED", SearchStatus::Exhausted)
        .value("OUT_OF_STATES", SearchStatus::OutOfStates)
        .value("OUT_OF_TIME", SearchStatus::OutOfTime)
        .value("CANCELLED", SearchStatus::Cancelled)
        .value("FAILED", SearchStatus::Failed)
        .value("UNSOLVABLE", SearchStatus::Unsolvable)
        .def("__str__", [](SearchStatus s) { return std::string(search::to_string(s)); });

    nb::enum_<search::ActionOrdering>(m, "ActionOrdering",
                                      "Rollout IW's action ordering: guidance, never pruning.")
        .value("IN_ORDER", search::ActionOrdering::InOrder)
        .value("RANDOMIZED", search::ActionOrdering::Randomized)
        .value("DIRECT_GOAL_ACHIEVER_FIRST", search::ActionOrdering::DirectGoalAchieverFirst)
        .value("GOAL_REGRESSION_RELEVANCE", search::ActionOrdering::GoalRegressionRelevance)
        .value("MIXED_REGRESSION_RANDOM", search::ActionOrdering::MixedRegressionRandom)
        .def("__str__", [](search::ActionOrdering o) { return std::string(search::to_string(o)); });

    nb::enum_<search::TransitionOutcome>(m, "TransitionOutcome",
                                         "What became of a transition's successor (observer.on_transition).")
        .value("OPENED", search::TransitionOutcome::Opened, "Entered the search as a new node or open state.")
        .value("REOPENED", search::TransitionOutcome::Reopened,
               "Reached on a cheaper path and re-entered the open list (astar_iw).")
        .value("DUPLICATE", search::TransitionOutcome::Duplicate,
               "Already known, nothing changed: a width-0 duplicate queue entry, a state already in the tree or among "
               "the next layer's candidates, a stored state not reached on a cheaper path.")
        .value("PRUNED", search::TransitionOutcome::Pruned,
               "Rejected for another reason: not novel, beyond a depth or incumbent bound, or cut by a budget.")
        .value("DEAD_END", search::TransitionOutcome::DeadEnd, "An infinite heuristic value (astar_iw).")
        .value("GOAL", search::TransitionOutcome::Goal, "A goal found on generation (rollout_iw).")
        .value("BLOCKED", search::TransitionOutcome::Blocked, "One of blocked_states; checked before anything else.")
        .value("SELF_LOOP", search::TransitionOutcome::SelfLoop,
               "Equal to the state it was generated from; reported before the other reasons except BLOCKED.");

    bind_landmarks(m);

    nb::class_<PyCancelToken>(m, "CancelToken",
                              "Cooperative cancellation: request() from any thread stops every search given this token.")
        .def(nb::init<>())
        .def("request", [](const PyCancelToken& t) { t.token.request(); },
             "Asks every search holding this token to stop; they end with Status.CANCELLED at their next check. "
             "Safe from any thread.")
        .def_prop_ro("requested", [](const PyCancelToken& t) { return t.token.requested(); },
                     "Whether request() was called.")
        .def("__repr__", [](const PyCancelToken& t) {
            return std::string("CancelToken(requested=") + (t.token.requested() ? "True" : "False") + ")";
        });

    auto statistics = nb::class_<search::SearchStatistics>(
        m, "Statistics",
        "The counts and time every search result reports under the same names (its `stats`). Searches that do not "
        "measure a field report 0 for it.");
    bind_stats_pickle(statistics, [](const search::SearchStatistics& x) { return enc(x); }, dec_stats);
    statistics.def_ro("expanded", &search::SearchStatistics::expanded, "States whose successors were generated.")
        .def_ro("generated", &search::SearchStatistics::generated,
                "Successors generated (transitions, duplicates included).")
        .def_ro("states", &search::SearchStatistics::states,
                "States stored (BrFS, A*, GBFS, beam, EHC) or tree nodes (IW, SIW, Rollout IW).")
        .def_ro("pruned", &search::SearchStatistics::pruned,
                "Successors not entered: blocked, not novel, or dead ends.")
        .def_ro("seconds", &search::SearchStatistics::seconds, "Wall time of the search, in seconds.")
        .def_ro("memory_bytes", &search::SearchStatistics::memory_bytes,
                "Bytes of the search's states, nodes, open lists and novelty tables when it ended (IW and SIW: "
                "their largest pass).")
        .def("__repr__", [](const search::SearchStatistics& s) {
            return "Statistics(" + stats_repr(s) + ", seconds=" + std::to_string(s.seconds) +
                   ", memory_bytes=" + std::to_string(s.memory_bytes) + ")";
        });

    auto tree = nb::class_<PySearchTree>(
        m, "SearchTree",
        "One IW pass as flat arrays: N nodes (the root is node 0, node i is row i), R distinct state rows and, "
        "recorded with tree='transitions', T transitions grouped by their parent node (offsets). Every array is a "
        "read-only, zero-copy NumPy view that keeps the tree's storage alive.");
    tree.def_prop_ro("task", [](const PySearchTree& x) { return task_object(x.o.obj); },
                     "The Task whose slot numbering and ids interpret the rows and labels.")
        .def_prop_ro("num_nodes", [](const PySearchTree& x) { return x.tree->num_nodes(); },
                     "N: the nodes of the tree, the root included.")
        .def_prop_ro("num_transitions", [](const PySearchTree& x) { return x.tree->num_transitions(); },
                     "T: the recorded transitions (0 when recorded with tree='nodes').")
        .def("states", [](const PySearchTree& x) {
            nb::typed<nb::list, PyState> out{nb::list()};
            const auto& t = *x.tree;
            const usize R = t.words + t.numeric_words;
            for (usize i = 0; i < t.row_goal.size(); ++i)
            {
                const u64* r = R ? t.rows.data() + i * R : nullptr;
                out.append(make_state(x.o, State(StateView{r, t.words, t.numeric_words ? r + t.words : nullptr, t.numeric_words})));
            }
            return out;
        }, "All recorded rows as States, nodes first.")
        .def("labels", [](const PySearchTree& x) {
            nb::typed<nb::list, std::optional<PyAction>> out{nb::list()};
            const auto& t = *x.tree;
            out.append(nb::none());
            for (usize i = 1; i < t.num_nodes(); ++i)
            {
                const u32 schema = static_cast<u32>(t.schema[i]);
                std::vector<ObjectId> b;
                for (u32 j = 0; j < x.o.core->task->data().schemas[schema].arity(); ++j)
                    b.push_back(ObjectId{static_cast<u32>(t.binding[i * t.label_width + j])});
                out.append(action_object(x.o, Action(SchemaId{schema}, std::move(b))));
            }
            return out;
        }, "Incoming node Actions; the root label is None.")
        .def("__reduce__", [](const PySearchTree& x) { return result_reduce("search_tree", x.o, enc_tree(x.o, x.tree)); });
    tree.def_prop_ro("width", [](const PySearchTree& x) { return x.tree->width; }, "The novelty width of the pass.");
    tree.def_prop_ro("words", [](const PySearchTree& x) { return x.tree->words; },
                     "The fluent words of a row (the task's word width when the search ended).");
    tree.def_prop_ro("numeric_words", [](const PySearchTree& x) { return x.tree->numeric_words; },
                     "The numeric words of a row, after its fluent words.");
    tree.def_prop_ro("label_width", [](const PySearchTree& x) { return x.tree->label_width; },
                     "The largest action arity of the task: the columns of binding and t_binding.");
    tree.def_prop_ro("rows", [](const PySearchTree& x) {
        return tree_array(x, x.tree->rows, {x.tree->row_goal.size(), usize{x.tree->words} + x.tree->numeric_words});
    }, nb::rv_policy::reference,
       "uint64 [R, words + numeric_words]: the distinct states, nodes first (row i is node i), then "
       "the other successors in the order they were first generated; numeric words last.");
    tree.def_prop_ro("parent", [](const PySearchTree& x) {
        return tree_array(x, x.tree->parent, {x.tree->num_nodes()});
    }, nb::rv_policy::reference,
       "int32 [N]: the parent node of each node; -1 for the root.");
    tree.def_prop_ro("depth", [](const PySearchTree& x) {
        return tree_array(x, x.tree->depth, {x.tree->num_nodes()});
    }, nb::rv_policy::reference,
       "int32 [N]: the depth of each node; 0 for the root.");
    tree.def_prop_ro("schema", [](const PySearchTree& x) {
        return tree_array(x, x.tree->schema, {x.tree->num_nodes()});
    }, nb::rv_policy::reference,
       "int32 [N]: the action schema id of each node's incoming action; -1 for the root.");
    tree.def_prop_ro("binding", [](const PySearchTree& x) {
        return tree_array(x, x.tree->binding, {x.tree->num_nodes(), x.tree->label_width});
    }, nb::rv_policy::reference,
       "int32 [N, label_width]: the object ids of each node's incoming action, padded with -1; all "
       "-1 for the root.");
    tree.def_prop_ro("node_flags", [](const PySearchTree& x) {
        return tree_array(x, x.tree->node_flags, {x.tree->num_nodes()});
    }, nb::rv_policy::reference,
       "uint8 [N]: bit flags of each node: 1 successors generated, 2 goal, 4 counted as expanded "
       "without generating successors, 8 dropped by a beam's layer selection, 16 entered the queue "
       "more than once.");
    tree.def_prop_ro("offsets", [](const PySearchTree& x) {
        return tree_array(x, x.tree->offsets, {x.tree->offsets.size()});
    }, nb::rv_policy::reference,
       "int64 [N + 1]: the transitions of node i are offsets[i]:offsets[i + 1]; empty when recorded "
       "with tree='nodes'.");
    tree.def_prop_ro("t_schema", [](const PySearchTree& x) {
        return tree_array(x, x.tree->t_schema, {x.tree->num_transitions()});
    }, nb::rv_policy::reference,
       "int32 [T]: the action schema id of each transition.");
    tree.def_prop_ro("t_binding", [](const PySearchTree& x) {
        return tree_array(x, x.tree->t_binding, {x.tree->num_transitions(), x.tree->label_width});
    }, nb::rv_policy::reference,
       "int32 [T, label_width]: the object ids of each transition's action, padded with -1.");
    tree.def_prop_ro("t_row", [](const PySearchTree& x) {
        return tree_array(x, x.tree->t_row, {x.tree->num_transitions()});
    }, nb::rv_policy::reference,
       "int32 [T]: the row of each transition's successor.");
    tree.def_prop_ro("t_child", [](const PySearchTree& x) {
        return tree_array(x, x.tree->t_child, {x.tree->num_transitions()});
    }, nb::rv_policy::reference,
       "int32 [T]: the node each transition entered (an existing one for a duplicate queue entry); "
       "-1 when its successor got no node.");
    tree.def_prop_ro("t_outcome", [](const PySearchTree& x) {
        return tree_array(x, x.tree->t_outcome, {x.tree->num_transitions()});
    }, nb::rv_policy::reference,
       "uint8 [T]: the TransitionOutcome value of each transition, as an observer gets it.");
    tree.def_prop_ro("row_goal", [](const PySearchTree& x) {
        return tree_array(x, x.tree->row_goal, {x.tree->row_goal.size()});
    }, nb::rv_policy::reference,
       "uint8 [R]: 1 for each row that satisfies the pass's goal, else 0.");
    auto iw_pass = nb::class_<PyIwPass>(m, "IwPass", "The counts and optional tree of one IW pass.");
    iw_pass.def("__reduce__", [](const PyIwPass& x) { return result_reduce("iw_pass", x.o, enc(x.o, x.p)); });
    iw_pass.def_prop_ro("width", [](const PyIwPass& x) { return x.p.width; },
                        "The pass's novelty width k.");
    iw_pass.def_prop_ro("status", [](const PyIwPass& x) { return x.p.status; },
                        "How the pass ended (Status).");
    iw_pass.def_prop_ro("expanded", [](const PyIwPass& x) { return x.p.expanded; },
                        "Non-goal states popped and counted as expanded.");
    iw_pass.def_prop_ro("generated", [](const PyIwPass& x) { return x.p.generated; },
                        "Transitions generated from expanded states.");
    iw_pass.def_prop_ro("generated_in_tree", [](const PyIwPass& x) { return x.p.generated_in_tree; },
                        "Successors admitted, including width-0 duplicate queue entries and excluding beam drops.");
    iw_pass.def_prop_ro("skipped", [](const PyIwPass& x) { return x.p.skipped; },
                        "Counted expansions whose successors were not generated.");
    iw_pass.def_prop_ro("blocked", [](const PyIwPass& x) { return x.p.blocked; },
                        "Successors rejected because they are blocked.");
    iw_pass.def_prop_ro("seconds", [](const PyIwPass& x) { return x.p.seconds; },
                        "Wall time of the pass, in seconds.");
    iw_pass.def_prop_ro("placeholder", [](const PyIwPass& x) { return x.p.placeholder; },
                        "The empty width-0 entry of an optimized IW(1) ladder.");
    iw_pass.def_prop_ro("stats", [](const PyIwPass& x) { return x.p.stats(); }, "The pass's counts as Statistics.")
        .def_prop_ro("tree", [](const PyIwPass& x) -> nb::typed<nb::object, std::optional<PySearchTree>> {
            if (!x.p.tree)
                return nb::none();
            return nb::cast(PySearchTree{x.p.tree, x.o});
        }, "The recorded tree, or None when recording was disabled or the pass is a placeholder.")
        .def("__repr__", [](const PyIwPass& x) {
            return "IwPass(width=" + std::to_string(x.p.width) + ", status=" + search::to_string(x.p.status) +
                   ", expanded=" + std::to_string(x.p.expanded) + ", generated=" + std::to_string(x.p.generated) +
                   ", generated_in_tree=" + std::to_string(x.p.generated_in_tree) + ")";
        });
    auto subproblem = nb::class_<PySiwSubproblem>(
        m, "SiwSubproblem", "One subproblem of SIW: reach a state with fewer unsatisfied goal literals.");
    subproblem.def("__reduce__", [](const PySiwSubproblem& x) { return result_reduce("siw_subproblem", x.o, enc(x.o, x.p)); });
    subproblem.def_prop_ro("status", [](const PySiwSubproblem& x) { return x.p.status; },
                           "How its IW ladder ended (Status).");
    subproblem.def_prop_ro("unsatisfied_at_start", [](const PySiwSubproblem& x) { return x.p.unsatisfied_at_start; },
                           "Goal literals unsatisfied in the subproblem's start state.");
    subproblem.def_prop_ro("plan_length", [](const PySiwSubproblem& x) { return x.p.plan_length; },
                           "Actions of its subplan.");
    subproblem.def_prop_ro("effective_width", [](const PySiwSubproblem& x) { return x.p.effective_width; },
                           "The width of the pass that solved it.");
    subproblem.def_prop_ro("passes", [](const PySiwSubproblem& x) { return pass_list(x.o, x.p.passes); },
                           "Its IW passes (a list of IwPass), one per width.")
        .def("__repr__", [](const PySiwSubproblem& x) {
            return std::string("SiwSubproblem(status=") + search::to_string(x.p.status) +
                   ", unsatisfied_at_start=" + std::to_string(x.p.unsatisfied_at_start) +
                   ", plan_length=" + std::to_string(x.p.plan_length) +
                   ", effective_width=" + std::to_string(x.p.effective_width) + ")";
        });

    auto heuristic_stats = nb::class_<heuristics::HeuristicStats>(
        m, "HeuristicStatistics", "The counts of a heuristic's evaluator (Heuristic.counters, BestFirstResult.heuristic).");
    bind_stats_pickle(heuristic_stats, [](const heuristics::HeuristicStats& x) { return enc(x); }, dec_heuristic_stats);
    heuristic_stats.def_ro("evaluations", &heuristics::HeuristicStats::evaluations, "States evaluated.")
        .def_ro("grounded", &heuristics::HeuristicStats::grounded, "Evaluations by the grounded relaxation.")
        .def_ro("lifted", &heuristics::HeuristicStats::lifted, "Evaluations by the lifted fallback.")
        .def_ro("dead_ends", &heuristics::HeuristicStats::dead_ends, "Evaluations that gave +inf.")
        .def_ro("grounding_seconds", &heuristics::HeuristicStats::grounding_seconds,
                "Seconds spent grounding the relaxation.")
        .def("__repr__", [](const heuristics::HeuristicStats& s) {
            return "HeuristicStatistics(evaluations=" + std::to_string(s.evaluations) +
                   ", grounded=" + std::to_string(s.grounded) + ", lifted=" + std::to_string(s.lifted) +
                   ", dead_ends=" + std::to_string(s.dead_ends) + ")";
        });

    // results ---------------------------------------------------------------------------------------------------------
    auto iw_result = nb::class_<PyIwResult>(
        m, "IwResult",
        "The result of iw and iw_pass: the common result fields, one IwPass per width run in `passes`, and the width "
        "of the pass that solved the task.");
    bind_result_fields(iw_result, [](const PyIwResult& x) -> const search::IwResult& { return x.r; },
                       [](const PyIwResult& x) { return x.r.stats; });
    iw_result
        .def_prop_ro("passes", [](const PyIwResult& x) { return pass_list(x.o, x.r.passes); },
                     "The passes run (a list of IwPass), index = width.")
        .def_prop_ro("effective_width", [](const PyIwResult& x) { return x.r.effective_width; },
                     "The width of the pass that solved the task (0 unless solved).")
        .def_prop_ro("fluent_slots", [](const PyIwResult& x) { return x.r.fluent_slots; },
                     "Fluent atom slots the task had assigned when the search ended.")
        .def("__reduce__", [](const PyIwResult& x) { return result_reduce("iw", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](const PyIwResult& x) {
            return std::string("IwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) +
                   ", effective_width=" + std::to_string(x.r.effective_width) + ", " + stats_repr(x.r.stats) + ")";
        });

    auto siw_result = nb::class_<PySiwResult>(
        m, "SiwResult",
        "The result of siw: the common result fields (plan: the whole plan when solved, else empty) and one "
        "SiwSubproblem per subproblem.");
    bind_result_fields(siw_result, [](const PySiwResult& x) -> const search::SiwResult& { return x.r; },
                       [](const PySiwResult& x) { return x.r.stats; });
    siw_result
        .def_prop_ro("partial_plan", [](const PySiwResult& x) { return plan_list(x.o, x.r.partial_plan); },
                     "Unless solved: the subplans of the subproblems solved before the one that failed, from the "
                     "start state to where that one started (empty when solved).")
        .def_prop_ro("subproblems", [](const PySiwResult& x) {
            nb::typed<nb::list, PySiwSubproblem> out{nb::list()};
            for (const auto& p : x.r.subproblems)
                out.append(PySiwSubproblem{p, x.o});
            return out;
        },
                     "The subproblems (a list of SiwSubproblem), in order.")
        .def_prop_ro("max_effective_width", [](const PySiwResult& x) { return x.r.max_effective_width; },
                     "The largest width a subproblem needed.")
        .def("__reduce__", [](const PySiwResult& x) { return result_reduce("siw", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](const PySiwResult& x) {
            return std::string("SiwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) +
                   (x.r.partial_plan.empty() ? std::string{}
                                             : ", partial_plan_length=" + std::to_string(x.r.partial_plan.size())) +
                   ", cost=" + cost_repr(x.r.cost) + ", subproblems=" + std::to_string(x.r.subproblems.size()) + ", " +
                   stats_repr(x.r.stats) + ")";
        });

    auto novelty_stats = nb::class_<search::AStarIwNoveltyStatistics>(m, "AStarIwNoveltyStatistics",
                                                                      "The counters of AStarIW's novelty pruning.");
    bind_stats_pickle(novelty_stats, [](const search::AStarIwNoveltyStatistics& x) { return enc(x); }, dec_novelty);
    novelty_stats.def_ro("rejected", &search::AStarIwNoveltyStatistics::rejected,
                         "Generated successors that were not novel (pruned).")
        .def_ro("stale", &search::AStarIwNoveltyStatistics::stale,
                "Queued states that lost every novel feature at their g before expansion.")
        .def_ro("stale_g", &search::AStarIwNoveltyStatistics::stale_g,
                "Queue entries superseded by a cheaper path or already closed.")
        .def_ro("probes", &search::AStarIwNoveltyStatistics::probes,
                "Read-only novelty tests of transitions before their heuristic evaluation.")
        .def_ro("updates", &search::AStarIwNoveltyStatistics::updates,
                "Updates of the novelty table by expanded transitions (dead ends never update it).")
        .def_ro("pop_tests", &search::AStarIwNoveltyStatistics::pop_tests,
                "Novelty tests of states taken from the queue.")
        .def_ro("table_bytes", &search::AStarIwNoveltyStatistics::table_bytes,
                "The approximate memory of the novelty table, in bytes.");

    auto best_first_result = nb::class_<PyBestFirstResult>(
        m, "BestFirstResult",
        "The result of astar_eager, astar_lazy, gbfs_eager, gbfs_lazy, beam and the greedy and weighted variants: "
        "the common result fields and the heuristic's counts.");
    bind_result_fields(best_first_result,
                       [](const PyBestFirstResult& x) -> const search::BestFirstResult& { return x.r; },
                       [](const PyBestFirstResult& x) { return best_first_stats(x.r); });
    best_first_result
        .def_prop_ro("evaluations", [](const PyBestFirstResult& x) { return x.r.evaluations; },
                     "Heuristic evaluations by the search.")
        .def_prop_ro("dead_ends", [](const PyBestFirstResult& x) { return x.r.dead_ends; },
                     "States the heuristic rated +inf.")
        .def_prop_ro("reopened", [](const PyBestFirstResult& x) { return x.r.reopened; },
                     "A*: states whose g decreased after they were opened.")
        .def_prop_ro("expanded_until_last_jump", [](const PyBestFirstResult& x) { return x.r.expanded_until_last_jump; },
                     "A*: the expansions before the f-value of the last node popped was first reached (those below "
                     "the final f-layer; Fast Downward's statistic). 0 for the other searches.")
        .def_prop_ro("layers", [](const PyBestFirstResult& x) { return x.r.layers; }, "Beam search: layers expanded.")
        .def_prop_ro("initial_h", [](const PyBestFirstResult& x) { return x.r.initial_h; },
                     "The heuristic value of the start state.")
        .def_prop_ro("heuristic", [](const PyBestFirstResult& x) { return x.r.heuristic; },
                     "The evaluator's counts when the search ended (HeuristicStatistics).")
        .def_prop_ro("setup_seconds", [](const PyBestFirstResult& x) { return x.r.setup_seconds; },
                     "Seconds spent building the heuristic (grounding) and the action costs, before the search.")
        .def_prop_ro("algorithm", [](const PyBestFirstResult& x) { return x.r.algorithm; },
                     "The search that ran, e.g. 'astar_eager' or 'beam'.")
        .def_prop_ro("store", [](const PyBestFirstResult& x) { return x.r.store; }, "The state store used.")
        .def_prop_ro("queue", [](const PyBestFirstResult& x) { return x.r.queue; }, "The open list used.")
        .def("__reduce__",
             [](const PyBestFirstResult& x) { return result_reduce("best_first", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](nb::handle self) {
            const PyBestFirstResult& x = nb::cast<const PyBestFirstResult&>(self);
            return nb::cast<std::string>(self.type().attr("__name__")) + "(algorithm=" + x.r.algorithm +
                   ", status=" + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) + ", " +
                   stats_repr(x.r.stats) + ")";
        });

    auto brfs_result = nb::class_<PyBrfsResult>(
        m, "BrfsResult",
        "The result of brfs: the common result fields (the plan reaches the first goal state expanded) and the "
        "layer counts.");
    bind_result_fields(brfs_result, [](const PyBrfsResult& x) -> const BrfsResult& { return x.r; },
                       [](const PyBrfsResult& x) { return brfs_stats(x.r); });
    brfs_result
        .def_prop_ro("exhausted", [](const PyBrfsResult& x) { return x.r.exhausted; },
                     "The whole reachable space was expanded (with stop_at_goal=False also when solved).")
        .def_prop_ro("num_goal_states", [](const PyBrfsResult& x) { return x.r.num_goal_states; },
                     "Goal states expanded.")
        .def_prop_ro("layers", [](const PyBrfsResult& x) { return x.r.layers; },
                     "Layers expanded (the last one partly when the search stopped inside it).")
        .def_prop_ro("layer_counts",
                     [](const PyBrfsResult& x) {
                         std::vector<std::tuple<u64, u64, u64>> out;
                         for (const auto& l : x.r.layer_counts)
                             out.emplace_back(l[0], l[1], l[2]);
                         return out;
                     },
                     "With layer_stats=True: per expanded layer (expanded, generated, new states); else empty.")
        .def_prop_ro("store", [](const PyBrfsResult& x) { return x.r.store; },
                     "The state store used: 'flat', 'chunked', 'compact' or 'concurrent'.")
        .def_prop_ro("threads_used", [](const PyBrfsResult& x) { return x.r.threads_used; }, "Threads the search ran on.")
        .def_prop_ro("fingerprint", [](const PyBrfsResult& x) { return x.r.fingerprint; },
                     "With fingerprint=True: a hash over (id, state) of every stored state; else 0.")
        .def("__reduce__", [](const PyBrfsResult& x) { return result_reduce("brfs", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](const PyBrfsResult& x) {
            return std::string("BrfsResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) +
                   ", layers=" + std::to_string(x.r.layers) + ", " + stats_repr(brfs_stats(x.r)) + ")";
        });

    // parallel rollouts and portfolio: results ----------------------------------------------------------------------
    using RS = search::RolloutIwStatistics;
    auto rollout_stats = nb::class_<RS>(m, "RolloutIwStatistics", "Counters of a Rollout IW run.");
    bind_stats_pickle(rollout_stats, [](const RS& x) { return enc(x); }, dec_rollout_stats);
    rollout_stats.def_ro("rollouts", &RS::rollouts,
                         "Rollouts from the root.")
        .def_ro("generated", &RS::generated, "New tree nodes.")
        .def_ro("expanded", &RS::expanded, "Nodes whose actions were materialized.")
        .def_ro("feature_depth_improvements", &RS::feature_depth_improvements,
                "New nodes that lowered the best depth of some atom.")
        .def_ro("case1", &RS::case1,
                "New nodes that lowered an atom's best depth (the rollout went on).")
        .def_ro("case2", &RS::case2,
                "New nodes that lowered no atom's best depth (solved at once, pruned).")
        .def_ro("case3", &RS::case3,
                "Visits of an existing node no longer at the best depth of any of its atoms (solved).")
        .def_ro("case4", &RS::case4,
                "Visits of an existing node still at the best depth of one of its atoms (walked into).")
        .def_ro("solved_propagations", &RS::solved_propagations,
                "Parents marked solved because all their children were.")
        .def_ro("dead_ends", &RS::dead_ends,
                "Nodes without applicable actions.")
        .def_ro("depth_bound_prunings", &RS::depth_bound_prunings,
                "New nodes cut at max_depth.")
        .def_ro("incumbent_bound_prunings", &RS::incumbent_bound_prunings,
                "New nodes cut because they could not beat the incumbent plan's length.")
        .def_ro("blocked", &RS::blocked,
                "Successors that were blocked states.")
        .def_ro("max_rollout_depth", &RS::max_rollout_depth,
                "The depth of the deepest rollout.")
        .def_ro("tree_nodes", &RS::tree_nodes,
                "The nodes of the tree (the stored states).")
        .def_ro("seconds", &RS::seconds,
                "Wall time in seconds.")
        .def("__repr__", [](const RS& s) {
            return "RolloutIwStatistics(rollouts=" + std::to_string(s.rollouts) + ", generated=" + std::to_string(s.generated) +
                   ", expanded=" + std::to_string(s.expanded) + ", tree_nodes=" + std::to_string(s.tree_nodes) + ")";
        });

    auto rollout_iw_result = nb::class_<PyRolloutIwResult>(
        m, "RolloutIwResult",
        "The result of rollout_iw: the common result fields (stats: expanded, generated, states = tree nodes, pruned "
        "= successors not novel, cut by a bound or blocked) and the Rollout IW counters in `rollout`.");
    bind_result_fields(rollout_iw_result,
                       [](const PyRolloutIwResult& x) -> const search::RolloutIwResult& { return x.r; },
                       [](const PyRolloutIwResult& x) { return x.r.rollout.stats(); });
    rollout_iw_result
        .def_prop_ro("rollout", [](const PyRolloutIwResult& x) { return x.r.rollout; },
                     "The Rollout IW counters (RolloutIwStatistics).")
        .def_prop_ro("root_solved", [](const PyRolloutIwResult& x) { return x.r.root_solved; },
                     "The reachable width-1 space was exhausted (relative to the depth and incumbent bounds).")
        .def_prop_ro("fluent_slots", [](const PyRolloutIwResult& x) { return x.r.fluent_slots; },
                     "Fluent atom slots the task had assigned when the search ended.")
        .def("__reduce__",
             [](const PyRolloutIwResult& x) { return result_reduce("rollout_iw", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](const PyRolloutIwResult& x) {
            return std::string("RolloutIwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) +
                   ", rollouts=" + std::to_string(x.r.rollout.rollouts) + ", root_solved=" +
                   (x.r.root_solved ? "True" : "False") + ")";
        });

    nb::class_<PyLandingState>(m, "LandingState",
                               "A state a rollout created that first made some fluent atom true.")
        .def_prop_ro("state", [](const PyLandingState& x) { return make_state(x.o, State(x.l.state)); },
                     "The State.")
        .def_prop_ro("atoms", [](const PyLandingState& x) { return atom_list(x.o, x.l.atoms); },
                     "Its fluent atoms (ascending by canonical id).")
        .def_prop_ro("direct_dead_end", [](const PyLandingState& x) { return x.l.direct_dead_end; },
                     "No action is applicable in it.")
        .def("__reduce__",
             [](const PyLandingState& x) { return result_reduce("landing_state", x.o, enc(x.o, x.l)); })
        .def("__repr__", [](const PyLandingState& x) {
            return "LandingState(atoms=" + std::to_string(x.l.atoms.size()) + ", direct_dead_end=" +
                   (x.l.direct_dead_end ? "True" : "False") + ")";
        });

    auto rollout_result = nb::class_<PyRolloutResult>(
        m, "RolloutResult",
        "One rollout of find_rollouts_parallel: the common result fields of its IW ladder, its passes, and what its "
        "private state repository recorded. Atoms are GroundAtoms, equal across rollouts and thread counts.");
    bind_result_fields(rollout_result, [](const PyRolloutResult& x) -> const search::IwResult& { return x.get().search; },
                       [](const PyRolloutResult& x) { return x.get().search.stats; });
    rollout_result
        .def_prop_ro("seed", [](const PyRolloutResult& x) { return x.seed; }, "The rollout's seed.")
        .def_prop_ro("passes", [](const PyRolloutResult& x) { return pass_list(x.o, x.get().search.passes); },
                     "The passes of its IW ladder (a list of IwPass), index = width.")
        .def_prop_ro("effective_width", [](const PyRolloutResult& x) { return x.get().search.effective_width; },
                     "The width of the pass that solved it (0 unless solved).")
        .def_prop_ro("reached_fluent_atoms",
                     [](const PyRolloutResult& x) { return atom_list(x.o, x.get().reached_fluent_atoms); },
                     "The fluent atoms true in some state the rollout created (ascending by canonical id).")
        .def_prop_ro("reached_derived_atoms",
                     [](const PyRolloutResult& x) { return atom_list(x.o, x.get().reached_derived_atoms); },
                     "The derived atoms true in some state the rollout created (ascending by canonical id).")
        .def_prop_ro("num_created_states", [](const PyRolloutResult& x) { return x.get().num_created_states; },
                     "Distinct states the rollout created: every successor of every expanded state, pruned or not, "
                     "across all passes, the start state included (stats.states counts the states the passes stored).")
        .def_prop_ro("landing_states",
                     [](const PyRolloutResult& x) {
                         nb::typed<nb::list, PyLandingState> out{nb::list()};
                         for (const search::LandingState& l : x.get().landing_states)
                             out.append(nb::cast(PyLandingState{l, x.o}, nb::rv_policy::move));
                         return out;
                     },
                     "With report_landing_states: the distinct landing states, in order of their smallest "
                     "first-achieved atom.")
        .def_prop_ro("landing_state_by_atom",
                     [](const PyRolloutResult& x) {
                         nb::typed<nb::dict, PyGroundAtom, int> out{nb::dict()};
                         for (const auto& [c, i] : x.get().landing_state_by_atom)
                             out[atom_object(x.o, c)] = nb::int_(i);
                         return out;
                     },
                     "Per fluent atom the index of its landing state in landing_states.")
        .def_prop_ro("co_occurrence", [](const PyRolloutResult& x) { return co_occurrence_dict(x.o, x.get().co_occurrence); },
                     "With report_co_occurrence: per fluent atom the atoms of the created states holding it.")
        .def("__reduce__",
             [](const PyRolloutResult& x) { return result_reduce("rollout", x.o, nb::make_tuple(x.seed, enc(x.o, x.get()))); })
        .def("__repr__", [](const PyRolloutResult& x) {
            const search::RolloutResult& r = x.get();
            return "RolloutResult(seed=" + std::to_string(x.seed) + ", status=" + search::to_string(r.search.status) +
                   ", num_created_states=" + std::to_string(r.num_created_states) + ", reached_fluent_atoms=" +
                   std::to_string(r.reached_fluent_atoms.size()) + ", landing_states=" +
                   std::to_string(r.landing_states.size()) + ")";
        });

    nb::class_<PyParallelRollouts>(m, "ParallelRolloutsResult",
                                   "The rollouts of a find_rollouts_parallel batch, in seed order (a sequence).")
        .def_prop_ro("rollouts",
                     [](const PyParallelRollouts& x) {
                         nb::typed<nb::list, PyRolloutResult> out{nb::list()};
                         for (u32 i = 0; i < x.r->rollouts.size(); ++i)
                             out.append(nb::cast(PyRolloutResult{x.r, i, x.seeds[i], x.o}, nb::rv_policy::move));
                         return out;
                     },
                     "The RolloutResults, one per seed in seed order.")
        .def("__len__", [](const PyParallelRollouts& x) { return x.r->rollouts.size(); })
        .def(
            "__getitem__",
            [](const PyParallelRollouts& x, i64 i) {
                const i64 n = static_cast<i64>(x.r->rollouts.size());
                if (i < 0)
                    i += n;
                if (i < 0 || i >= n)
                    throw nb::index_error("mymyr: rollout index out of range");
                const u32 k = static_cast<u32>(i);
                return PyRolloutResult{x.r, k, x.seeds[k], x.o};
            },
            "index"_a)
        .def_prop_ro("threads_used", [](const PyParallelRollouts& x) { return x.r->threads_used; },
                     "The threads the batch ran on.")
        .def_prop_ro("message", [](const PyParallelRollouts& x) { return x.r->message; },
                     "Why the batch could not run (every rollout then FAILED), or why it ran on one thread.")
        .def("__reduce__",
             [](const PyParallelRollouts& x) {
                 return result_reduce("rollouts", x.o,
                                      nb::make_tuple(enc_list(x.r->rollouts, [&](const auto& r) { return enc(x.o, r); }),
                                                     enc_list(x.seeds, [](u64 v) { return nb::int_(v); }), x.r->threads_used,
                                                     x.r->message));
             })
        .def("__repr__", [](const PyParallelRollouts& x) {
            usize solved = 0;
            for (const search::RolloutResult& r : x.r->rollouts)
                solved += r.search.status == SearchStatus::Solved;
            return "ParallelRolloutsResult(rollouts=" + std::to_string(x.r->rollouts.size()) + ", solved=" +
                   std::to_string(solved) + ", threads_used=" + std::to_string(x.r->threads_used) + ")";
        });

    nb::class_<PyMergedLandingStates>(m, "MergedLandingStates", "merge_landing_states: distinct landing states of a batch.")
        .def_prop_ro("states",
                     [](const PyMergedLandingStates& x) {
                         nb::typed<nb::list, PyState> out{nb::list()};
                         for (const State& s : x.m.states)
                             out.append(make_state(x.o, State(s)));
                         return out;
                     },
                     "The distinct landing States, in the order first seen.")
        .def_prop_ro("by_rollout", [](const PyMergedLandingStates& x) { return x.m.by_rollout; },
                     "Per rollout the indices of its landing states in `states`.")
        .def("__reduce__", [](const PyMergedLandingStates& x) {
            nb::list states, by_rollout;
            for (const State& st : x.m.states)
                states.append(make_state(x.o, State(st)));
            for (const auto& r : x.m.by_rollout)
                by_rollout.append(enc_list(r, [](u32 v) { return nb::int_(v); }));
            return result_reduce("merged_landing_states", x.o, nb::make_tuple(states, by_rollout));
        });

    auto portfolio_result = nb::class_<PyPortfolioResult>(
        m, "PortfolioResult",
        "The result of atomic_goal_portfolio: the common result fields (stats: expanded counts the expansions of the "
        "certifier and the rollout workers, the other counts add up theirs, seconds is the portfolio's wall time), "
        "the certifier's IW(1) ladder and each rollout worker's counters.");
    bind_result_fields(portfolio_result,
                       [](const PyPortfolioResult& x) -> const search::PortfolioResult& { return x.r; },
                       [](const PyPortfolioResult& x) { return x.r.stats; });
    portfolio_result
        .def_prop_ro("plan_length",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.plan_length == ~u32{0})
                             return std::nullopt;
                         return x.r.plan_length;
                     },
                     "Actions of the plan (None without a plan).")
        .def_prop_ro("certified_optimal", [](const PyPortfolioResult& x) { return x.r.certified_optimal; },
                     "The plan is shortest in the width-1 space: the certifier found it, or completed depth "
                     "plan_length - 1.")
        .def_prop_ro("iw_lower_bound", [](const PyPortfolioResult& x) { return x.r.iw_lower_bound; },
                     "iw_completed_depth + 1 (0 when the certifier completed no layer): no plan in the width-1 space "
                     "is shorter.")
        .def_prop_ro("iw_completed_depth",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.iw_completed_depth == search::SearchCoordination::k_no_depth)
                             return std::nullopt;
                         return x.r.iw_completed_depth;
                     },
                     "The deepest layer the certifier finished (None: none).")
        .def_prop_ro("winning_worker",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.winning_worker == ~u32{0})
                             return std::nullopt;
                         return x.r.winning_worker;
                     },
                     "0: the certifier, 1..K: the rollout workers; None without a plan.")
        .def_prop_ro("certifier_ran", [](const PyPortfolioResult& x) { return x.r.certifier_ran; },
                     "The certifier started.")
        .def_prop_ro("certifier_status",
                     [](const PyPortfolioResult& x) -> std::optional<SearchStatus> {
                         if (!x.r.certifier_ran)
                             return std::nullopt;
                         return x.r.certifier_status;
                     },
                     "How the certifier ended (None when it never started).")
        .def_prop_ro("certifier", [](const PyPortfolioResult& x) { return PyIwResult{x.r.certifier, x.o}; },
                     "The certifier's IW(1) ladder (IwResult).")
        .def_prop_ro("rollout_statistics", [](const PyPortfolioResult& x) { return x.r.rollout_statistics; },
                     "Per rollout worker its RolloutIwStatistics, summed over its rounds.")
        .def_prop_ro("rollout_statuses",
                     [](const PyPortfolioResult& x) {
                         std::vector<std::optional<SearchStatus>> out;
                         for (usize i = 0; i < x.r.rollout_statuses.size(); ++i)
                             out.push_back(x.r.rollout_rounds[i] ? std::optional<SearchStatus>(x.r.rollout_statuses[i])
                                                                 : std::nullopt);
                         return out;
                     },
                     "Per rollout worker the status of its last round; None for a worker that never started.")
        .def_prop_ro("rollout_rounds", [](const PyPortfolioResult& x) { return x.r.rollout_rounds; },
                     "Per rollout worker the rounds it ran (0: it never started).")
        .def_prop_ro("threads_used", [](const PyPortfolioResult& x) { return x.r.threads_used; },
                     "Threads the portfolio ran on (the certifier and the rollout workers).")
        .def_prop_ro("fluent_slots", [](const PyPortfolioResult& x) { return x.r.fluent_slots; },
                     "Fluent atom slots the task had assigned when the search ended.")
        .def("__reduce__",
             [](const PyPortfolioResult& x) { return result_reduce("portfolio", x.o, enc(x.o, x.r)); })
        .def("__repr__", [](const PyPortfolioResult& x) {
            return std::string("PortfolioResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", certified_optimal=" +
                   (x.r.certified_optimal ? "True" : "False") + ", winning_worker=" +
                   (x.r.winning_worker == ~u32{0} ? std::string("None") : std::to_string(x.r.winning_worker)) +
                   ", certifier_status=" + status_or_none(x.r.certifier_ran, x.r.certifier_status) + ")";
        });

    // heuristics ------------------------------------------------------------------------------------------------------
    nb::class_<PyHeuristic>(m, "Heuristic",
                            "A heuristic of a task: h(state) -> float (+inf: dead end). kind: 'blind', 'goal_count', "
                            "'max', 'add', 'ff', 'set_additive', 'h2', 'lmcut' (LM-cut, admissible), 'lmcount' "
                            "(LAMA's landmark count, path-dependent); the perfect heuristic h* comes from "
                            "a state space: Heuristic.perfect(space).\n\ncosts: 'auto' (the default: the task's objective, i.e. 'unit' "
                            "for a task without action costs and metric, else 'real'), 'unit' (every action 1) or "
                            "'real' (the task's action costs: zero and fractional costs exact up to 6 decimal places, "
                            "rounded down beyond, 0 for a cost that depends on the state; set-additive then sums the "
                            "costs of its achiever set, one per supported proposition).\n\nUnit costs on a task whose "
                            "actions cost less than 1 overestimate; "
                            "evaluation: 'auto', 'grounded', 'lifted' (set_additive, h2, lmcut and lmcount are "
                            "grounded only: a state outside the grounding raises RuntimeError). "
                            "share=another Heuristic of the task reuses its grounding.\n\nPath-dependent heuristics "
                            "(path_dependent: lmcount) value a state by the path to it, through a status (bytes): "
                            "initial_status(start), progress(status, state, action, successor) along a transition, "
                            "merge(status, other) for a state reached on two paths, and h(state, status); h(state) "
                            "takes the state as the start of the path. The searches keep the statuses themselves.\n\n"
                            "Calls on one object are serialized; use one object per thread for parallel evaluation. "
                            "A search given the object evaluates with an evaluator of its own that shares the "
                            "grounding, so the object stays callable while the search runs (from its observer, from "
                            "other threads); stats counts the calls on the object, the search's evaluations are in its "
                            "result.")
        .def(
            "__init__",
            [](PyHeuristic* self, TaskArg task, StrArg kind, StrArg costs, StrArg evaluation,
               Arg<PyHeuristic> share) {
                Owner o = owner_of(task);
                heuristics::Options opts;
                opts.kind = parse_kind(kind);
                opts.costs = parse_costs(costs);
                opts.evaluation = parse_evaluation(evaluation);
                if (!share.is_none())
                {
                    const PyHeuristic& other = nb::cast<const PyHeuristic&>(share);
                    if (other.o.core->task->uid() != o.core->task->uid())
                        throw nb::value_error("mymyr: share= must be a heuristic of the same task");
                    opts.relaxed = other.options.relaxed;
                }
                std::unique_ptr<heuristics::Heuristic> h;
                {
                    nb::gil_scoped_release release;  // grounding can take a while
                    h = heuristics::make_heuristic(*o.core->task, opts);
                }
                opts.relaxed = h->relaxed();
                if (!opts.relaxed && opts.evaluation == heuristics::Evaluation::Auto)
                    opts.evaluation = heuristics::Evaluation::Lifted;  // the grounding exceeded the budget: never retry it
                new (self) PyHeuristic{std::move(h), std::move(o), opts, {}};
            },
            "task"_a, "kind"_a = "ff", nb::kw_only(), "costs"_a = "auto", "evaluation"_a = "auto",
            "share"_a = nb::none())
        .def_static(
            "perfect",
            [](Arg<PyStateSpace> space_obj, StrArg costs) {
                if (!nb::isinstance<PyStateSpace>(space_obj))
                    throw nb::type_error("mymyr: space must be a mymyr.datasets.StateSpace");
                const PyStateSpace& space = *nb::inst_ptr<PyStateSpace>(space_obj);
                heuristics::Options opts;
                opts.kind = heuristics::Kind::Perfect;
                opts.costs = parse_costs(costs);
                std::unique_ptr<heuristics::Heuristic> h;
                try
                {
                    h = heuristics::perfect(space.space, opts.costs);
                }
                catch (const std::invalid_argument& e)
                {
                    throw nb::value_error(e.what());
                }
                return std::unique_ptr<PyHeuristic>(new PyHeuristic{std::move(h), space.owner, opts, {}, space.space, space_obj});
            },
            "space"_a, nb::kw_only(), "costs"_a = "auto",
            "The perfect heuristic h* of the task of a state space (mymyr.datasets.state_space(task, "
            "remove_if_unsolvable=False)): the goal distance of a state, +inf where no goal is reachable. costs: 'auto' (the default: 'real' for a task with action costs or a metric, else 'unit'), "
            "'unit' (the number of actions) or 'real' (the transition costs). A state outside "
            "the space raises ValueError, as does a search goal other than the task's.")
        .def("__call__", &py_evaluate, "state"_a, "status"_a = nb::none(),
             "h(state): the heuristic value (+inf for a dead end). status: a status of the path to the state "
             "(path-dependent heuristics); None takes the state as the start of the path.")
        .def("evaluate", &py_evaluate, "state"_a, "status"_a = nb::none(), "As h(state, status).")
        .def_prop_ro("path_dependent", [](const PyHeuristic& self) { return self.h->path_dependent(); },
                     "Whether the value of a state depends on the path to it (lmcount: the landmarks reached on it).")
        .def(
            "initial_status",
            [](PyHeuristic& self, StateArg state) {
                const State s = state_for(self.o, state, "state");
                std::vector<u64> w(self.h->status_words());
                {
                    nb::gil_scoped_release release;
                    std::lock_guard lock(self.m);
                    self.h->initial_status(s.view(), w);
                }
                return status_bytes(w);
            },
            "state"_a,
            "The status of a path that starts in `state` (bytes; empty for a heuristic that is not path-dependent). "
            "lmcount: the landmarks true in the initial state; for another state, inferred from it (those true in it "
            "or initially, and those ordered greedy-necessarily before one true in it but not initially).")
        .def(
            "progress",
            [](PyHeuristic& self, Arg<nb::bytes> status, StateArg state, Arg<PyAction> action, StateArg successor) {
                const std::vector<u64> parent = status_of(self, status, "status");
                const State s = state_for(self.o, state, "state");
                const State t = state_for(self.o, successor, "successor");
                if (!nb::isinstance<PyAction>(action))
                    throw nb::type_error("mymyr: action must be a mymyr.Action");
                const PyAction& a = *nb::inst_ptr<PyAction>(action);
                if (a.core != self.o.core)
                    throw nb::value_error("mymyr: action belongs to another task");
                std::vector<u64> w(parent.size());
                {
                    nb::gil_scoped_release release;
                    std::lock_guard lock(self.m);
                    self.h->progress(parent, s.view(), a.label(), t.view(), w);
                }
                return status_bytes(w);
            },
            "status"_a, "state"_a, "action"_a, "successor"_a,
            "The status of `successor`, reached from `state` (whose status is `status`) by `action`. lmcount: the "
            "landmarks of `status` and those true in `successor`.")
        .def(
            "merge",
            [](PyHeuristic& self, Arg<nb::bytes> status, Arg<nb::bytes> other) {
                std::vector<u64> w = status_of(self, status, "status");
                const std::vector<u64> x = status_of(self, other, "other");
                self.h->merge(w, x);
                return status_bytes(w);
            },
            "status"_a, "other"_a,
            "The status of a state reached on two paths with statuses `status` and `other`. lmcount: the landmarks "
            "both reached (Fast Downward's), inferred if either status is.")
        .def(
            "relaxed_plan",
            [](PyHeuristic& self, StateArg state) {
                const State s = state_for(self.o, state, "state");
                std::vector<Action> plan;
                {
                    nb::gil_scoped_release release;
                    std::lock_guard lock(self.m);
                    self.h->evaluate(s.view());
                    plan = self.h->relaxed_plan();
                }
                return plan_list(self.o, plan);
            },
            "state"_a,
            "FF and set-additive: the ground actions of the relaxed plan of `state` (set-additive: those of its "
            "achiever set); empty for the other kinds.")
        .def(
            "preferred_actions",
            [](PyHeuristic& self, StateArg state, Arg<nb::bytes> status) {
                const State s = state_for(self.o, state, "state");
                std::optional<std::vector<u64>> w;
                if (!status.is_none())
                    w = status_of(self, status, "status");
                nb::list applicable = nb::borrow<nb::list>(task_object(self.o.obj).attr("applicable_actions")(state));
                std::vector<Action> labels;
                for (nb::handle a : applicable)
                    labels.emplace_back(nb::cast<const PyAction&>(a).label());
                std::vector<char> keep(labels.size(), 0);
                {
                    nb::gil_scoped_release release;  // never block on the mutex while attached
                    std::lock_guard lock(self.m);
                    if (w)
                        self.h->evaluate(s.view(), *w);
                    else
                        self.h->evaluate(s.view());
                    if (self.h->provides_preferred())
                        for (usize i = 0; i < labels.size(); ++i)
                            keep[i] = self.h->preferred(labels[i].label()) ? 1 : 0;
                }
                ActionList out{nb::list()};
                for (usize i = 0; i < labels.size(); ++i)
                    if (keep[i])
                        out.append(applicable[i]);
                return out;
            },
            "state"_a, "status"_a = nb::none(),
            "The preferred operators of `state` (status: as for h(state, status)). FF and set-additive: the "
            "applicable actions in its relaxed plan; lmcount: the applicable actions with "
            "an effect that fires in `state` and adds a needed landmark false there; empty for the other kinds.")
        .def_prop_ro("kind", [](const PyHeuristic& self) { return std::string(heuristics::to_string(self.h->kind())); },
                     "The heuristic's name: 'blind', 'goal_count', 'max', 'add', 'ff', 'h2', 'set_additive', "
                     "'lmcut', 'lmcount', 'perfect' or 'custom'.")
        .def(
            "__reduce__",
            [](const PyHeuristic& self) {
                // its specification: (task, kind, costs, evaluation), or a perfect heuristic's (space, costs)
                const nb::module_ pk = nb::module_::import_("mymyr._pickle");
                if (self.space)
                    return nb::make_tuple(pk.attr("_restore_perfect_heuristic"),
                                          nb::make_tuple(self.space_object, costs_name(self.options.costs)));
                return nb::make_tuple(pk.attr("_restore_heuristic"),
                                      nb::make_tuple(task_object(self.o.obj), heuristics::to_string(self.options.kind),
                                                     costs_name(self.options.costs), evaluation_name(self.options.evaluation)));
            },
            "A heuristic pickles as its specification and is built again when unpickled (its grounding included); "
            "a perfect heuristic as its state space. counters start from zero.")
        .def_prop_ro("counters", [](PyHeuristic& self) {
            heuristics::HeuristicStats st;
            {
                nb::gil_scoped_release release;  // never block on the mutex while attached
                std::lock_guard lock(self.m);
                st = self.h->stats();
            }
            return st;
        },
                     "Its evaluation counters: evaluations, those by the grounded relaxation and by the lifted "
                     "evaluation, dead ends, and the grounding time in seconds.")
        .def_prop_ro("grounded", [](PyHeuristic& self) {
            bool g = false;
            {
                nb::gil_scoped_release release;
                std::lock_guard lock(self.m);
                g = self.h->relaxed() != nullptr;
            }
            return g;
        },
                     "Whether it evaluates on a grounded relaxation (False for blind, goal count, the lifted "
                     "evaluation, or a grounding beyond its budget).")
        .def_prop_ro(
            "admissible",
            [](PyHeuristic& self) {
                bool a = false;
                {
                    nb::gil_scoped_release release;
                    std::lock_guard lock(self.m);
                    a = self.h->admissible();
                }
                return a;
            },
            "Whether h never exceeds the optimal cost of the task's objective: 'blind', 'max', 'h2', 'lmcut' and "
            "'perfect' with the costs of the objective (costs 'auto' or 'real', or 'unit' on a unit-cost task). "
            "A cost bound (astar, anytime_astar) prunes by g + h only with an admissible heuristic.");

    nb::class_<PyAStarIwResult, PyBestFirstResult>(m, "AStarIwResult",
                                                   "The result of astar_iw: the common result fields and the "
                                                   "novelty counters in `novelty`.")
        .def_prop_ro("novelty", [](const PyAStarIwResult& r) { return r.novelty; },
                     "The novelty pruning counters (an AStarIwNoveltyStatistics).")
        .def("__reduce__", [](const PyAStarIwResult& x) {
            nb::tuple base = enc(x.o, x.r);
            nb::list fields;
            for (nb::handle h : base)
                fields.append(h);
            fields.append(enc(x.novelty));
            return result_reduce("astar_iw", x.o, nb::tuple(fields));
        });

    auto ehc_stats = nb::class_<search::EhcStatistics>(m, "EhcStatistics",
                                                       "Counters of enforced hill-climbing.");
    bind_stats_pickle(ehc_stats, [](const search::EhcStatistics& x) { return enc(x); }, dec_ehc);
    ehc_stats.def_ro("lookaheads", &search::EhcStatistics::lookaheads, "Breadth-first lookaheads, retries included.")
        .def_ro("improvements", &search::EhcStatistics::improvements, "Lookaheads that reached a state with a smaller h.")
        .def_ro("retries", &search::EhcStatistics::retries,
                "Lookaheads repeated with every action after the helpful actions found no better state.")
        .def_ro("max_depth", &search::EhcStatistics::max_depth,
                "The longest path from a lookahead's start to its better state or the goal, in actions.")
        .def_ro("expanded", &search::EhcStatistics::expanded,
                "Expansions of hill-climbing (the result's stats add the fallback's).")
        .def_ro("evaluations", &search::EhcStatistics::evaluations, "Heuristic evaluations of hill-climbing.")
        .def_ro("failed", &search::EhcStatistics::failed, "Hill-climbing found no better state from some state.")
        .def_ro("fallback", &search::EhcStatistics::fallback, "The fallback search ran.")
        .def("__repr__", [](const search::EhcStatistics& s) {
            return "EhcStatistics(lookaheads=" + std::to_string(s.lookaheads) + ", improvements=" +
                   std::to_string(s.improvements) + ", retries=" + std::to_string(s.retries) +
                   ", max_depth=" + std::to_string(s.max_depth) + ", failed=" + (s.failed ? "True" : "False") +
                   ", fallback=" + (s.fallback ? "True" : "False") + ")";
        });

    nb::class_<PyEhcResult, PyBestFirstResult>(
        m, "EhcResult",
        "The result of ehc: the best-first result fields (stats, evaluations and dead_ends count hill-climbing and the "
        "fallback; store and queue are the fallback's when it ran) and the hill-climbing counters in `ehc`.")
        .def_prop_ro("ehc", [](const PyEhcResult& x) { return x.ehc; },
                     "The hill-climbing counters (lookaheads, improvements, retries, whether it failed and the "
                     "greedy best-first search ran).")
        .def("__reduce__", [](const PyEhcResult& x) {
            nb::list fields;
            for (nb::handle h : enc(x.o, x.r))
                fields.append(h);
            fields.append(enc(x.ehc));
            return result_reduce("ehc", x.o, nb::tuple(fields));
        });

    auto anytime_solution = nb::class_<search::AnytimeSolution>(
        m, "AnytimeSolution", "An improved plan of anytime_astar, in the order found (its cost, not the plan).");
    bind_stats_pickle(anytime_solution, [](const search::AnytimeSolution& x) { return enc(x); }, dec_anytime_solution);
    anytime_solution.def_ro("cost", &search::AnytimeSolution::cost,
                            "The plan's cost under the task's metric.")
        .def_ro("seconds", &search::AnytimeSolution::seconds,
                "Seconds since the search started (after building the heuristic).")
        .def_ro("weight", &search::AnytimeSolution::weight, "The weight of the iteration that found it.")
        .def_ro("iteration", &search::AnytimeSolution::iteration,
                "The index of the iteration that found it (into AnytimeResult.iterations).")
        .def_ro("expanded", &search::AnytimeSolution::expanded, "Expansions of the whole search until it was found.")
        .def_ro("length", &search::AnytimeSolution::length, "Actions.")
        .def("__repr__", [](const search::AnytimeSolution& s) {
            return "AnytimeSolution(cost=" + cost_repr(s.cost) + ", seconds=" + std::to_string(s.seconds) +
                   ", weight=" + cost_repr(s.weight) + ", iteration=" + std::to_string(s.iteration) +
                   ", length=" + std::to_string(s.length) + ")";
        });

    auto anytime_iteration =
        nb::class_<search::AnytimeIteration>(m, "AnytimeIteration", "One weighted A* iteration of anytime_astar.");
    bind_stats_pickle(anytime_iteration, [](const search::AnytimeIteration& x) { return enc(x); }, dec_anytime_iteration);
    anytime_iteration.def_ro("weight", &search::AnytimeIteration::weight,
                             "The weight w of this iteration (f = g + w * h).")
        .def_ro("cost_bound", &search::AnytimeIteration::cost_bound,
                "Plans of the iteration cost less (math.inf: no bound).")
        .def_ro("status", &search::AnytimeIteration::status,
                "How the iteration ended (a Status; SOLVED when it found a cheaper plan).")
        .def_ro("stats", &search::AnytimeIteration::stats,
                "The iteration's counts and time (a Statistics).")
        .def_ro("evaluations", &search::AnytimeIteration::evaluations,
                "The heuristic evaluations of the iteration.")
        .def_ro("cost", &search::AnytimeIteration::cost, "The cost of the iteration's plan when it found one.")
        .def("__repr__", [](const search::AnytimeIteration& s) {
            return "AnytimeIteration(weight=" + cost_repr(s.weight) + ", cost_bound=" + cost_repr(s.cost_bound) +
                   ", status=" + search::to_string(s.status) + ", " + stats_repr(s.stats) + ")";
        });

    nb::class_<PyAnytimeResult, PyBestFirstResult>(
        m, "AnytimeResult",
        "The result of anytime_astar: the best plan in the best-first result fields (status SOLVED when a plan was "
        "found; stats, evaluations, dead_ends and reopened add up the iterations), every improved plan in "
        "`solutions`, every iteration in `iterations`, and `optimal` when the search proved the plan optimal.")
        .def_prop_ro("solutions", [](const PyAnytimeResult& x) { return x.solutions; },
                     "The plans found, each cheaper than the one before (AnytimeSolutions in the order found; the "
                     "last is the result's plan).")
        .def_prop_ro("iterations", [](const PyAnytimeResult& x) { return x.iterations; },
                     "The weighted A* iterations in the order run (AnytimeIterations).")
        .def_prop_ro("optimal", [](const PyAnytimeResult& x) { return x.optimal; },
                     "Whether the plan is proven optimal: an iteration with reopening exhausted the states cheaper "
                     "than the plan, or one with weight 1 and an admissible heuristic found it.")
        .def("__reduce__", [](const PyAnytimeResult& x) {
            nb::list fields;
            for (nb::handle h : enc(x.o, x.r))
                fields.append(h);
            fields.append(enc_list(x.solutions, [](const auto& v) { return enc(v); }));
            fields.append(enc_list(x.iterations, [](const auto& v) { return enc(v); }));
            fields.append(x.optimal);
            return result_reduce("anytime", x.o, nb::tuple(fields));
        });

    using RestoredResult =
        nb::typed<nb::object, std::variant<PyIwResult, PySiwResult, PyBestFirstResult, PyBrfsResult, PyRolloutIwResult,
                                           PyLandingState, PyMergedLandingStates, PyParallelRollouts, PyRolloutResult,
                                           PyPortfolioResult>>;
    m.def(
        "_restore_search_result",
        [](std::string_view kind, TaskArg task, u64 fingerprint, nb::typed<nb::tuple, ann::Any, nb::ellipsis> fields) {
            return RestoredResult(restore_result(kind, task, fingerprint, fields));
        },
        "kind"_a, "task"_a, "fingerprint"_a, "fields"_a,
          "The search result of a result pickle (mymyr._pickle).");

    // searches --------------------------------------------------------------------------------------------------------
    m.def(
        "iw",
        [](TaskArg task, IntArg max_width, WidthZeroArg width_zero, bool optimize_iw1, bool witness_pruning,
           bool canonical_order, StateArg start, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::IwOptions opts = iw_options(cs, o, int_arg<u32>(max_width, "max_width"), width_zero, optimize_iw1,
                                                witness_pruning, canonical_order, start, transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        MYMYR_IW_ARGS,
        paragraphs({"The IW(k) ladder: passes of width 0..max_width (optimized "
                     "IW(1) when max_width == 1). width_zero: 'expand_depth_one' (the depth-1 states enter the tree and are expanded) or 'root_only' (they are goal-tested but not expanded). "
                     "transition_ordering: a LandmarkTransitionOrdering that orders each layer of the width-1 pass "
                     "(a transition ordering strategy). ",
                    k_layer_doc,
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc})
            .c_str());

    m.def(
        "iw_pass",
        [](TaskArg task, IntArg width, WidthZeroArg width_zero, bool witness_pruning, bool canonical_order,
           StateArg start, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            const u32 k = int_arg<u32>(width, "width");
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::IwOptions opts =
                iw_options(cs, o, k, width_zero, false, witness_pruning, canonical_order, start,
                transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::iw_pass(t, k, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, "width"_a, nb::kw_only(), "width_zero"_a = "expand_depth_one", "witness_pruning"_a = false,
        "canonical_order"_a = true, "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        "transition_ordering"_a = nb::none(),
        MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG, "tree"_a = nb::none(), "One IW(width) pass alone, without the ladder. See iw() for the keyword arguments.");

    m.def(
        "siw",
        [](TaskArg task, IntArg max_width, WidthZeroArg width_zero, bool optimize_iw1, bool witness_pruning,
           bool canonical_order, StateArg start, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::SiwOptions opts = iw_options(cs, o, int_arg<u32>(max_width, "max_width"), width_zero, optimize_iw1,
                                                 witness_pruning, canonical_order, start, transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::SiwResult r = run_detached(cs, [&] { return search::siw(t, opts); });
            return PySiwResult{std::move(r), o};
        },
        MYMYR_IW_ARGS,
        paragraphs({"Serialized IW: IW ladders from subgoal to subgoal, each subproblem ending where "
                     "fewer goal literals are unsatisfied. Unsolved, the plan is empty and partial_plan holds the "
                     "subplans of the subproblems solved before the one that failed. ",
                    k_layer_doc,
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc})
            .c_str());

    m.def(
        "brfs",
        [](TaskArg task, IntArg threads, StrArg store, bool witness_pruning, bool canonical_order,
           bool deterministic_ids, StateArg start, bool stop_at_goal,
           bool fingerprint, LayerArg layer_order,
           IntArg seed, IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg max_states, IntArg max_expanded,
           IntArg max_depth, FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states,
           ObserverArg observer, IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            BrfsOptions opts;
            opts.goal = std::move(cs.control.goal);
            opts.max_states = cs.control.budget.max_states;
            opts.max_expanded = cs.control.budget.max_expanded;
            opts.max_depth = cs.control.budget.max_depth;
            opts.blocked_states = std::move(cs.control.blocked_states);
            if (!start.is_none())
                opts.start = state_for(o, start, "start");
            opts.max_seconds = cs.control.budget.max_seconds;
            opts.cancel = cs.control.cancel;
            opts.observer = cs.control.observer;
            opts.progress_interval = cs.control.progress_interval;
            opts.threads = threads_arg(threads);
            const std::string st = str_arg(store, "store");
            if (st == "auto")
                opts.store = BrfsOptions::Store::Auto;
            else if (st == "flat")
                opts.store = BrfsOptions::Store::Flat;
            else if (st == "chunked")
                opts.store = BrfsOptions::Store::Chunked;
            else if (st == "compact")
                opts.store = BrfsOptions::Store::Compact;
            else if (st == "concurrent")
                opts.store = BrfsOptions::Store::Concurrent;
            else
                throw nb::value_error("mymyr: store must be 'auto', 'flat', 'chunked', 'compact' or 'concurrent'");
            opts.witness_pruning = witness_pruning;
            opts.canonical_order = canonical_order;
            opts.deterministic_ids = deterministic_ids;
            opts.stop_at_goal = stop_at_goal;
            opts.fingerprint = fingerprint;
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            const Task& t = *o.core->task;
            BrfsResult r = run_detached(cs, [&] { return brfs(t, opts); });
            return PyBrfsResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "threads"_a = 1, "store"_a = "auto", "witness_pruning"_a = true,
        "canonical_order"_a = true, "deterministic_ids"_a = true, "start"_a = nb::none(), "stop_at_goal"_a = true,
        "fingerprint"_a = false, MYMYR_LAYER_ARGS, "max_states"_a = nb::none(), "max_expanded"_a = nb::none(),
        "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(), "cancel"_a = nb::none(), "goal"_a = nb::none(),
        "blocked_states"_a = nb::none(), "observer"_a = nb::none(), "progress_interval"_a = nb::none(),
        paragraphs({"Breadth-first search over the reachable states: single-threaded, or "
                     "layer-synchronous on `threads` threads with ids independent of the thread count. A search that "
                     "expands a goal state is SOLVED, with the plan to the first one (a shortest plan); stop_at_goal "
                     "(the default) stops there, stop_at_goal=False goes on through the whole reachable space "
                     "(`exhausted` tells whether it got through, `num_goal_states` counts the goal states). A callable goal "
                     "needs threads=1.\n\nfingerprint: the result's fingerprint "
                     "hashes (id, canonical state) over the whole store (determinism checks; 0 when off). An ordered "
                     "layer_order needs the 'flat' or 'chunked' store ('auto' picks one of them) and, without a beam, "
                     "threads=1.\n\n"
                     "The observer gets "
                     "on_start, on_expand, on_generate, on_pass(depth, layer stats) per layer, on_progress, "
                     "on_solution and on_end. With threads > 1 it follows the make_worker protocol: make_worker(k) "
                     "gives thread k its own observer for on_expand, on_generate (child None: ids are assigned when "
                     "the layer ends) and on_progress, called from that thread; without make_worker (or when it "
                     "returns None) the search runs on one thread. ",
                    k_layer_doc,
                    k_control_doc})
            .c_str());

    m.def(
        "astar",
        [](TaskArg task, bool lazy, HeuristicArg heuristic, FloatArg weight, FloatArg cost_bound, StrArg costs,
           StrArg evaluation, StrArg store, StrArg queue, StateArg start, bool reopen, bool lazy_requeue,
           bool preferred_operators, IntArg preferred_weight, IntArg standard_weight, bool witness_pruning,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            const double w = float_arg(weight, "weight", 1.0);
            const double bound =
                opt_float_arg(cost_bound, "cost_bound").value_or(std::numeric_limits<double>::infinity());
            const BfFn fn = lazy ? &search::astar_lazy : &search::astar_eager;
            return best_first([&](const Task& t, const search::BestFirstOptions& base) {
                search::BestFirstOptions options = base;
                options.weight = w;
                options.cost_bound = bound;
                return fn(t, options);
            }, task, heuristic, costs, evaluation, store, queue, start, reopen, lazy_requeue, preferred_operators,
                int_arg<u32>(preferred_weight, "preferred_weight"), int_arg<u32>(standard_weight, "standard_weight", 1),
                1000, witness_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "lazy"_a = false, "heuristic"_a = "max", "weight"_a = 1.0, "cost_bound"_a = nb::none(),
        "costs"_a = "auto", "evaluation"_a = "auto",
        "store"_a = "auto", "queue"_a = "auto", "start"_a = nb::none(), "reopen"_a = true, "lazy_requeue"_a = true,
        "preferred_operators"_a = true, "preferred_weight"_a = 0, "standard_weight"_a = 1, "witness_pruning"_a = false,
        "max_states"_a = nb::none(), "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(),
        "max_seconds"_a = nb::none(), "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(),
        "observer"_a = nb::none(), "progress_interval"_a = nb::none(),
        paragraphs({"A*, eager or lazy. heuristic: a kind ('blind', 'goal_count', 'max', "
                     "'add', 'ff', 'set_additive', 'h2', 'lmcut', 'lmcount', as for Heuristic), 'perfect' (h* from the "
                     "task's state space, "
                     "generated first within max_states and max_seconds; ValueError when they stop it), a Heuristic "
                     "of the task, or a heuristic written in Python; costs: 'auto' (the default), 'unit' or 'real', as for "
                     "Heuristic.\n\nWith costs='auto' and an admissible heuristic ('blind', 'max', 'h2', 'lmcut', "
                     "'perfect'), the plan is optimal for the task's objective (its total-cost or metric). A "
                     "path-dependent heuristic ('lmcount') gets each node's status: a state reached on several paths "
                     "keeps the merge of their statuses, and eager A* evaluates a state again when a cheaper path "
                     "reaches it; it cannot evaluate in batches, nor estimate a goal of several alternatives.\n\n"
                     "weight: weighted A*, f = g + weight * h (finite, >= 1; 1 is A* with the same expansions); ties "
                     "by lower h, then first in, first out. With an admissible heuristic of the task's goal (not a "
                     "callable goal) the plan costs at most weight times the optimum when the search reopens states "
                     "(reopen=True), and with a consistent heuristic ('blind', 'max') also without reopening; lazy "
                     "A* keeps the bound with lazy_requeue for consistent heuristics. A non-integral weight needs "
                     "queue 'auto' or 'heap'.\n\ncost_bound: plans cost less than it; states with g >= cost_bound, and "
                     "with an admissible heuristic of the task's goal those with g + h >= cost_bound, are not "
                     "opened, so EXHAUSTED means that no plan is cheaper than the bound (a transition that lowers g "
                     "then gives FAILED). Invalid values raise ValueError. ",
                    k_heuristic_doc,
                    k_control_doc})
            .c_str());

    m.def(
        "astar_iw",
        [](TaskArg task, HeuristicArg heuristic, IntArg width, StrArg features, LandmarksArg landmarks,
           double weight, bool preserve_goal_atoms, bool preserve_landmark_atoms,
           bool allow_non_novel_root_goal, bool probe_novelty_before_heuristic, StrArg costs,
           StrArg evaluation, StrArg store, StateArg start, bool witness_pruning, bool canonical_order,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            const Owner owner = owner_of(task);
            search::AStarIwOptions options;
            options.width = int_arg<u32>(width, "width");
            const std::string mode = str_arg(features, "features");
            if (mode == "classical")
                options.features = search::AStarIwFeatures::Classical;
            else if (mode == "abstracted")
                options.features = search::AStarIwFeatures::Abstracted;
            else if (mode == "base_abstracted")
                options.features = search::AStarIwFeatures::BaseAbstracted;
            else
                throw nb::value_error("mymyr: features must be 'classical', 'abstracted' or 'base_abstracted'");
            options.landmarks = parse_landmarks(owner, landmarks, false, false, nb::none(), nb::none());
            options.weight = weight;
            options.preserve_goal_atoms = preserve_goal_atoms;
            options.preserve_landmark_atoms = preserve_landmark_atoms;
            options.allow_non_novel_root_goal = allow_non_novel_root_goal;
            options.probe_novelty_before_heuristic = probe_novelty_before_heuristic;
            options.canonical_order = canonical_order;
            return best_first([&](const Task& t, const search::BestFirstOptions& base) {
                options.control = base.control;
                options.heuristic = base.heuristic;
                options.evaluator = base.evaluator;
                options.store = base.store;
                options.start = base.start;
                options.witness_pruning = base.witness_pruning;
                return search::astar_iw(t, options);
            }, task, heuristic, costs, evaluation, store, nb::str("heap"), start, true, true, false, 0, 1,
                1000, witness_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "heuristic"_a = "max", "width"_a = 1, "features"_a = "classical",
        "landmarks"_a = nb::none(), "weight"_a = 1.0, "preserve_goal_atoms"_a = true,
        "preserve_landmark_atoms"_a = true, "allow_non_novel_root_goal"_a = true,
        "probe_novelty_before_heuristic"_a = true, "costs"_a = "auto", "evaluation"_a = "auto",
        "store"_a = "auto", "start"_a = nb::none(), "witness_pruning"_a = false, "canonical_order"_a = true,
        MYMYR_CONTROL_ARGS,
        paragraphs({"Weighted A* with minimum-g novelty pruning. Priority is "
                     "(g + weight*h, h, g, state id). Unit-cost actions and no numeric fluents are required.\n\n"
                     "features: 'classical' (width 1..5), 'abstracted' or 'base_abstracted' (width 1..3). "
                     "landmarks: a fact landmark graph, 'approximate', 'lifted', or None. Abstracted modes "
                     "preserve positive goal atoms and landmark atoms by default. Non-novel root successors "
                     "are admitted with allow_non_novel_root_goal; only goals bypass the pop novelty test. "
                     "The read-only probe skips heuristic work without changing the search. Invalid widths "
                     "and weights return FAILED with a message. Novelty statistics are in result.novelty. "
                     "heuristic and costs: as for astar; novelty pruning does not guarantee optimality. ",
                    k_heuristic_doc,
                    k_control_doc,
                    k_transition_doc}).c_str());

    m.def(
        "gbfs",
        [](TaskArg task, bool lazy, HeuristicArg heuristic, StrArg costs, StrArg evaluation, StrArg store,
           StrArg queue, StateArg start, bool preferred_operators, IntArg preferred_weight, IntArg standard_weight,
           bool witness_pruning, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            return bf_call(lazy ? &search::gbfs_lazy : &search::gbfs_eager, task, heuristic, costs, evaluation, store,
                           queue, start, true, true, preferred_operators,
                           int_arg<u32>(preferred_weight, "preferred_weight"),
                           int_arg<u32>(standard_weight, "standard_weight", 1), 1000,
                           witness_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                           blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "lazy"_a = false, "heuristic"_a = "ff", "costs"_a = "auto", "evaluation"_a = "auto",
        "store"_a = "auto", "queue"_a = "auto", "start"_a = nb::none(), "preferred_operators"_a = true,
        "preferred_weight"_a = 0, "standard_weight"_a = 1, "witness_pruning"_a = false, "max_states"_a = nb::none(),
        "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        paragraphs({"Greedy best-first search, eager or lazy (preferred and standard lists "
                     "alternating, 64:1 by default). heuristic: as for astar. ",
                    k_heuristic_doc,
                    k_control_doc})
            .c_str());

    m.def(
        "ehc",
        [](TaskArg task, HeuristicArg heuristic, HelpfulArg helpful_actions, FallbackArg fallback, StrArg costs,
           StrArg evaluation, StrArg store, StateArg start, bool witness_pruning, bool canonical_order,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            search::EhcOptions options;
            const std::string helpful = str_arg(helpful_actions, "helpful_actions");
            if (helpful == "auto")
                options.helpful_actions = search::EhcOptions::Helpful::Auto;
            else if (helpful == "off")
                options.helpful_actions = search::EhcOptions::Helpful::Off;
            else if (helpful == "only")
                options.helpful_actions = search::EhcOptions::Helpful::Only;
            else if (helpful == "first")
                options.helpful_actions = search::EhcOptions::Helpful::First;
            else
                throw nb::value_error("mymyr: helpful_actions must be 'auto', 'off', 'only' or 'first'");
            const std::string fb = str_arg(fallback, "fallback");
            if (fb == "gbfs")
                options.fallback = search::EhcOptions::Fallback::Gbfs;
            else if (fb == "none")
                options.fallback = search::EhcOptions::Fallback::None;
            else
                throw nb::value_error("mymyr: fallback must be 'gbfs' or 'none'");
            options.canonical_order = canonical_order;
            return best_first([&](const Task& t, const search::BestFirstOptions& base) {
                options.control = base.control;
                options.heuristic = base.heuristic;
                options.evaluator = base.evaluator;
                options.store = base.store;
                options.start = base.start;
                options.witness_pruning = base.witness_pruning;
                return search::ehc(t, options);
            }, task, heuristic, costs, evaluation, store, nb::str("auto"), start, true, true, true, 0, 1, 1000,
                witness_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "heuristic"_a = "ff", "helpful_actions"_a = "auto", "fallback"_a = "gbfs",
        "costs"_a = "auto", "evaluation"_a = "auto", "store"_a = "auto", "start"_a = nb::none(),
        "witness_pruning"_a = false, "canonical_order"_a = true, MYMYR_CONTROL_ARGS,
        paragraphs({"Enforced hill-climbing (Hoffmann and Nebel 2001). From the current state a "
                     "breadth-first lookahead looks for a state with a strictly smaller h, commits to the first one "
                     "and repeats from it. A state is evaluated when the lookahead pops it (h = math.inf: a dead "
                     "end, dropped); a goal state ends the search when generated; within a lookahead each state is "
                     "entered once.\n\nhelpful_actions: 'first' (expand with the preferred operators of the state's "
                     "evaluation only, and repeat a lookahead that finds no better state with every action), 'only' "
                     "(preferred operators only, no repetition), 'off' (every action), 'auto' (the default: 'first' "
                     "when the heuristic gives preferred operators: 'ff' and 'set_additive' evaluated grounded, "
                     "'lmcount', or a heuristic with preferred_actions(); 'off' otherwise). 'first' and 'only' with a "
                     "heuristic without preferred operators give FAILED with a message.\n\nfallback: 'gbfs' (the "
                     "default) runs eager greedy best-first search from the start state with the same heuristic and "
                     "the rest of the budget when hill-climbing fails; 'none' returns EXHAUSTED then (hill-climbing "
                     "failed, which does not mean that the task is unsolvable).\n\nThe lookahead counts actions; g and "
                     "the plan cost are the task's metric.\n\nA path-dependent heuristic ('lmcount') gets the status of "
                     "the path the lookahead took, from the committed state, without merging. Observer: on_pass("
                     "lookahead, stats) after each lookahead; the fallback's on_expand, on_generate, on_prune and "
                     "on_progress follow. heuristic: as for astar. ",
                    k_heuristic_doc,
                    k_control_doc})
            .c_str());

    m.def(
        "anytime_astar",
        [](TaskArg task, WeightsArg weights, bool lazy, HeuristicArg heuristic, FloatArg cost_bound, StrArg costs,
           StrArg evaluation, StrArg store, StrArg queue, StateArg start, bool reopen, bool lazy_requeue,
           bool preferred_operators, IntArg preferred_weight, IntArg standard_weight, bool witness_pruning,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            search::AnytimeOptions options;
            options.weights.clear();
            for (nb::handle x : nb::iter(weights))
                options.weights.push_back(float_arg(x, "weights", 1.0));
            options.lazy = lazy;
            const double bound =
                opt_float_arg(cost_bound, "cost_bound").value_or(std::numeric_limits<double>::infinity());
            return best_first([&](const Task& t, const search::BestFirstOptions& base) {
                static_cast<search::BestFirstOptions&>(options) = base;
                options.cost_bound = bound;
                return search::anytime_astar(t, options);
            }, task, heuristic, costs, evaluation, store, queue, start, reopen, lazy_requeue, preferred_operators,
                int_arg<u32>(preferred_weight, "preferred_weight"), int_arg<u32>(standard_weight, "standard_weight", 1),
                1000, witness_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "weights"_a = nb::make_tuple(5, 3, 2, 1), "lazy"_a = false, "heuristic"_a = "ff",
        "cost_bound"_a = nb::none(), "costs"_a = "auto", "evaluation"_a = "auto", "store"_a = "auto",
        "queue"_a = "auto", "start"_a = nb::none(), "reopen"_a = true, "lazy_requeue"_a = true,
        "preferred_operators"_a = true, "preferred_weight"_a = 0, "standard_weight"_a = 1, "witness_pruning"_a = false,
        MYMYR_CONTROL_ARGS,
        paragraphs({"Anytime search: restarting weighted A* (Richter, Thayer and Ruml 2010, the "
                     "iterated searches of LAMA). Iteration k runs weighted A* (as astar with weight; lazy=True: "
                     "lazy A*) from the start state with the weight weights[min(k, len(weights) - 1)] and the cost "
                     "bound of the best plan so far (before the first plan: cost_bound), so each plan it finds is "
                     "cheaper than the one before.\n\nweights: non-empty, finite, >= 1 and non-increasing (5, 3, 2, 1 "
                     "by default). Every iteration starts afresh and shares the heuristic, which is built once. The "
                     "search ends when an iteration exhausts its space (no cheaper plan; with reopen=True and no "
                     "max_depth the plan is then proven optimal: `optimal`), when an eager iteration with weight 1 "
                     "and an admissible heuristic of the task's goal finds a plan with reopening (A*: `optimal`), or "
                     "when a budget, cancel or an observer stops an iteration.\n\nstatus is SOLVED when a plan was "
                     "found (also when a budget ended the search later). max_seconds and max_expanded bound the "
                     "whole search, max_states and max_depth each iteration. Observer: on_start once; the events of "
                     "every iteration (ids are the iteration's own); on_solution(plan, cost) for each improved plan "
                     "as it is found; on_pass(iteration, stats) after each iteration; on_end once. Invalid weights "
                     "raise ValueError. heuristic and the other options: as for astar. ",
                    k_heuristic_doc,
                    k_control_doc})
            .c_str());

    m.def(
        "beam",
        [](TaskArg task, IntArg beam_width, HeuristicArg heuristic, StrArg costs, StrArg evaluation, StrArg store,
           StateArg start, bool witness_pruning, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            return bf_call(&search::beam, task, heuristic, costs, evaluation, store, nb::str("auto"), start, true, true,
                           true, 0, 1, int_arg<u32>(beam_width, "beam_width", 1), witness_pruning, max_states, max_expanded,
                           max_depth, max_seconds, cancel, goal, blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "beam_width"_a = 1000, "heuristic"_a = "ff", "costs"_a = "auto", "evaluation"_a = "auto",
        "store"_a = "auto", "start"_a = nb::none(), "witness_pruning"_a = false, "max_states"_a = nb::none(),
        "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        paragraphs({"Heuristic beam search: the best `width` new successors of each layer by "
                     "(h, g). Incomplete: EXHAUSTED means the beam ran dry. heuristic: as for astar. ",
                    k_heuristic_doc,
                    k_control_doc})
            .c_str());

    // IPC plan text -----------------------------------------------------------------------------------------------
    m.def(
        "format_plan",
        [](TaskArg task, nb::typed<nb::iterable, PyAction> plan, StateArg start) {
            const Owner o = owner_of(task);
            std::vector<Action> actions;
            for (nb::handle a : plan)
            {
                if (!nb::isinstance<PyAction>(a))
                    throw nb::type_error("mymyr: a plan is a sequence of mymyr Actions");
                const PyAction& p = *nb::inst_ptr<PyAction>(a);
                if (p.core->task->uid() != o.core->task->uid())
                    throw nb::value_error("mymyr: the plan has an action of another task");
                actions.emplace_back(p.label());
            }
            std::optional<State> s;
            if (!start.is_none())
                s = state_for(o, start, "start");
            nb::gil_scoped_release release;
            return search::format_plan(*o.core->task, actions, s);
        },
        "task"_a, "plan"_a, nb::kw_only(), "start"_a = nb::none(),
        "The plan as IPC plan text: one '(name o1 ... ok)' line per action with the PDDL action's "
        "own parameters, then '; cost = N (unit cost)' or '; cost = N (general cost)'. N is the "
        "metric value of the state the plan reaches from `start` (default: the initial state), as a result's cost. "
        "Raises ValueError when an action is not applicable in its state.");

    m.def(
        "parse_plan",
        [](TaskArg task, const std::string& text, StateArg start) {
            const Owner o = owner_of(task);
            std::optional<State> s;
            if (!start.is_none())
                s = state_for(o, start, "start");
            std::vector<Action> actions;
            {
                nb::gil_scoped_release release;
                actions = search::parse_plan(*o.core->task, text, s);
            }
            ActionList out{nb::list()};
            for (const Action& a : actions)
                out.append(action_object(o, a));
            return out;
        },
        "task"_a, "text"_a, nb::kw_only(), "start"_a = nb::none(),
        "The actions of IPC plan text, replayed from `start` (default: the initial state). "
        "Blank lines and ';' comments are skipped and names compared without regard to case. A line names the PDDL "
        "action's own parameters or all of the schema's parameters; among the applicable actions that match, the first "
        "in canonical order is taken. Raises ValueError, naming the line, when no applicable action matches.");

    // plan validation ---------------------------------------------------------------------------------------------
    nb::enum_<search::PlanError>(m, "PlanError", "Why validate_plan rejected a plan.")
        .value("NONE", search::PlanError::None, "The plan is valid.")
        .value("SYNTAX", search::PlanError::Syntax, "Plan text: a line is not '(name o1 ... ok)'.")
        .value("UNKNOWN_ACTION", search::PlanError::UnknownAction, "No action of the task has the step's name.")
        .value("WRONG_ARITY", search::PlanError::WrongArity,
               "The step has another number of arguments than the action's parameters.")
        .value("UNKNOWN_OBJECT", search::PlanError::UnknownObject, "An argument names no object of the task.")
        .value("WRONG_TYPE", search::PlanError::WrongType, "An argument is not of its parameter's declared type.")
        .value("NOT_APPLICABLE", search::PlanError::NotApplicable,
               "The step's precondition does not hold in its state, or its effects are undefined there.")
        .value("GOAL_NOT_REACHED", search::PlanError::GoalNotReached,
               "Every step applies, but the goal does not hold after the last one.")
        .def("__str__", [](search::PlanError e) { return std::string(search::to_string(e)); });

    nb::class_<PyPlanValidation>(m, "PlanValidation",
                                 "The outcome of validate_plan: whether the plan is valid, else its first failing step "
                                 "and the reason; and the cost and state the steps that applied reached.")
        .def_prop_ro("valid", [](const PyPlanValidation& x) { return x.v.valid; }, "The plan is valid.")
        .def("__bool__", [](const PyPlanValidation& x) { return x.v.valid; }, "valid.")
        .def_prop_ro("error", [](const PyPlanValidation& x) { return x.v.error; },
                     "Why the plan is invalid (PlanError.NONE when valid).")
        .def_prop_ro("step", [](const PyPlanValidation& x) { return x.v.step; },
                     "The failing step's index in the plan (0-based); the plan's length when the goal is not reached "
                     "or the plan is valid.")
        .def_prop_ro("line", [](const PyPlanValidation& x) { return x.v.line; },
                     "Plan text: the failing step's line (1-based); 0 when the goal is not reached, the plan is valid "
                     "or the plan is a sequence of Actions.")
        .def_prop_ro("action", [](const PyPlanValidation& x) { return x.v.action; },
                     "The failing step as text, e.g. '(stack a b)'; empty when the goal is not reached or the plan is "
                     "valid.")
        .def_prop_ro("condition", [](const PyPlanValidation& x) { return x.v.condition; },
                     "What does not hold, as PDDL text: the precondition literal or numeric constraint, the "
                     "argument's type ('c - block'), or the goal literal; empty when there is none to name.")
        .def_prop_ro("message", [](const PyPlanValidation& x) { return x.v.message; },
                     "The whole reason in one sentence; empty when valid.")
        .def_prop_ro("cost", [](const PyPlanValidation& x) { return x.v.cost; },
                     "The metric value of `state`, counted from the start state's, as the searches report a plan's "
                     "cost (the plan's cost when valid).")
        .def_prop_ro("plan", [](const PyPlanValidation& x) { return plan_list(x.o, x.v.plan); },
                     "The steps that applied, as Actions (the whole plan when valid).")
        .def_prop_ro("state", [](const PyPlanValidation& x) { return make_state(x.o, State(x.v.state)); },
                     "The State the steps that applied reached (the goal state when valid).")
        .def("__repr__", [](const PyPlanValidation& x) {
            if (x.v.valid)
                return "PlanValidation(valid=True, steps=" + std::to_string(x.v.plan.size()) +
                       ", cost=" + cost_repr(x.v.cost) + ")";
            return std::string("PlanValidation(valid=False, error=") + search::to_string(x.v.error) +
                   ", step=" + std::to_string(x.v.step) + ", message=" + nb::cast<std::string>(nb::repr(nb::str(x.v.message.c_str()))) + ")";
        });

    m.def(
        "validate_plan",
        [](TaskArg task, Arg<ann::PlanSource> plan, StateArg start) {
            const Owner o = owner_of(task);
            std::optional<State> s;
            if (!start.is_none())
                s = state_for(o, start, "start");
            const Task& t = *o.core->task;
            auto text_of = [&]() -> std::optional<std::string> {
                if (nb::isinstance<nb::str>(plan))
                    return nb::cast<std::string>(plan);
                if (!nb::hasattr(plan, "__fspath__"))
                    return std::nullopt;
                // read through pathlib so that a missing or unreadable file raises the OSError subclass of its errno
                return nb::cast<std::string>(
                    nb::module_::import_("pathlib").attr("Path")(plan).attr("read_text")("encoding"_a = "utf-8"));
            };
            if (std::optional<std::string> text = text_of())
            {
                nb::gil_scoped_release release;
                return PyPlanValidation{search::validate_plan_text(t, *text, s), o};
            }
            std::vector<Action> actions;
            for (nb::handle a : plan)
            {
                if (!nb::isinstance<PyAction>(a))
                    throw nb::type_error("mymyr: plan= takes plan text, a plan file path or a sequence of Actions");
                const PyAction& p = *nb::inst_ptr<PyAction>(a);
                if (p.core->task->uid() != t.uid())
                    throw nb::value_error("mymyr: the plan has an action of another task");
                actions.emplace_back(p.label());
            }
            nb::gil_scoped_release release;
            return PyPlanValidation{search::validate_plan(t, actions, s), o};
        },
        "task"_a, "plan"_a, nb::kw_only(), "start"_a = nb::none(),
        "Validates a plan: replays it step by step and checks the goal at the end.\n\n"
        "Each step must name an action of the task with as many arguments as its parameters, objects of the task of "
        "the parameters' declared types, and be applicable in the state the steps before it reached (its "
        "precondition holds, derived atoms through the axioms, and its effects are defined). Its effects are then "
        "applied, conditional and numeric ones included. The first failing step ends the validation.\n\n"
        "Args:\n"
        "    task: The Task (or a TaskTable handle).\n"
        "    plan: IPC plan text (a str: one '(name o1 ... ok)' per line, blank lines and ';' comments skipped, "
        "names compared without regard to case), the path of a plan file (an os.PathLike, e.g. a pathlib.Path), or "
        "a sequence of Actions as the searches return them.\n"
        "    start: The State to start from (default: the task's initial state).\n\n"
        "Returns:\n"
        "    A PlanValidation: valid, or the failing step, its line in plan text and the reason; and the plan's cost "
        "under the task's metric, as the searches report it.\n\n"
        "Raises:\n"
        "    TypeError: plan is none of the three.\n"
        "    OSError: a plan file cannot be read (FileNotFoundError when it does not exist).\n"
        "    ValueError: an Action or the start state belongs to another task.");

    // the IW family variants --------------------------------------------------------------------------------------
    m.def(
        "liw",
        [](TaskArg task, IntArg max_width, LandmarksArg landmarks, bool disjunctive, bool all_private,
           FluentAtomsArg unshared_atoms, IntArg max_dense_bytes, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, WidthZeroArg width_zero,
           bool witness_pruning, bool canonical_order,
           StateArg start, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::LiwOptions opts;
            static_cast<search::IwOptions&>(opts) =
                iw_options(cs, o, int_arg<u32>(max_width, "max_width"), width_zero, true, witness_pruning,
                           canonical_order, start);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            opts.landmarks = parse_landmarks(o, landmarks, disjunctive, all_private, unshared_atoms, max_dense_bytes);
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::liw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "max_width"_a = 2, MYMYR_LANDMARK_ARGS, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "width_zero"_a = "expand_depth_one", "witness_pruning"_a = false, "canonical_order"_a = true,
        "start"_a = nb::none(), MYMYR_CONTROL_ARGS, "tree"_a = nb::none(),
        paragraphs({"LIW(k): the width-0 pass, then "
                     "LIW(1..max_width) over (landmark coordinate, tuple of size <= k) novelty, each pass from the start "
                     "with a fresh table, until one solves the task (an exhausted ladder is EXHAUSTED). No optimized IW(1). The result is an IwResult with the per-pass statistics.",
                    k_landmark_doc,
                    k_layer_doc,
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc})
            .c_str());

    m.def(
        "abstracted_iw",
        [](TaskArg task, IntArg width, bool base_abstracted, bool preserve_goal_atoms, bool keep_depth_one_novel,
           LandmarksArg landmarks, bool disjunctive, bool all_private, FluentAtomsArg unshared_atoms,
           IntArg max_dense_bytes, bool preserve_landmark_atoms, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, bool witness_pruning,
           bool canonical_order, StateArg start,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::AbstractedIwOptions opts =
                aiw_options(cs, o, keep_depth_one_novel, landmarks, disjunctive, all_private, unshared_atoms,
                            max_dense_bytes, preserve_landmark_atoms, layer_order, seed, max_next_layer_states,
                            prefer_more_satisfied_goals, beam_width, beam_novelty, randomize_ties,
                            beam_chunk, threads, witness_pruning, canonical_order, start);
            opts.width = int_arg<u32>(width, "width");
            opts.base_abstracted = base_abstracted;
            opts.preserve_goal_atoms = preserve_goal_atoms;
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::abstracted_iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "width"_a = 1, "base_abstracted"_a = false, "preserve_goal_atoms"_a = true,
        "keep_depth_one_novel"_a = false, MYMYR_LANDMARK_ARGS, "preserve_landmark_atoms"_a = true, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "witness_pruning"_a = false, "canonical_order"_a = true, "start"_a = nb::none(),
        MYMYR_CONTROL_ARGS, "tree"_a = nb::none(),
        paragraphs({"Abstracted IW(width): one breadth-first pass whose "
                     "novelty features are abstracted atoms, one per object position with the other objects' type "
                     "signature (base_abstracted: BAIW, no signatures); preserve_goal_atoms keeps positive goal atoms "
                     "whole; keep_depth_one_novel expands the non-novel successors of the start. width 1..3. With "
                     "landmarks: abstracted LIW (one table per landmark rank; preserve_landmark_atoms keeps landmark "
                     "atoms whole). The IwResult holds the one pass (IwPass.width = width); an exhausted pass is EXHAUSTED.",
                    k_landmark_doc,
                    k_layer_doc,
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc})
            .c_str());

    m.def(
        "projective_iw",
        [](TaskArg task, bool typed_projection, bool keep_goal_nonunary_atoms, bool keep_depth_one_novel,
           LandmarksArg landmarks, bool disjunctive, bool all_private, FluentAtomsArg unshared_atoms,
           IntArg max_dense_bytes, bool preserve_landmark_atoms, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, bool witness_pruning,
           bool canonical_order, StateArg start,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::ProjectiveIwOptions opts;
            static_cast<search::AbstractedIwOptions&>(opts) =
                aiw_options(cs, o, keep_depth_one_novel, landmarks, disjunctive, all_private, unshared_atoms,
                            max_dense_bytes, preserve_landmark_atoms, layer_order, seed, max_next_layer_states,
                            prefer_more_satisfied_goals, beam_width, beam_novelty, randomize_ties,
                            beam_chunk, threads, witness_pruning, canonical_order, start);
            opts.typed_projection = typed_projection;
            opts.keep_goal_nonunary_atoms = keep_goal_nonunary_atoms;
            opts.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::projective_iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "typed_projection"_a = false, "keep_goal_nonunary_atoms"_a = false,
        "keep_depth_one_novel"_a = false, MYMYR_LANDMARK_ARGS, "preserve_landmark_atoms"_a = true, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "witness_pruning"_a = false, "canonical_order"_a = true, "start"_a = nb::none(),
        MYMYR_CONTROL_ARGS, "tree"_a = nb::none(),
        paragraphs({"The projective_iw alias: abstracted IW(1) with base_abstracted = not typed_projection "
                     "and preserve_goal_atoms = keep_goal_nonunary_atoms. See abstracted_iw.",
                    k_landmark_doc,
                    k_layer_doc,
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc})
            .c_str());

    m.def(
        "rollout_iw",
        [](TaskArg task, OrderingArg ordering, IntArg seed, IntArg max_rollouts, IntArg incumbent_bound,
           bool canonical_order, StateArg start, IntArg max_states, IntArg max_depth,
           FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, nb::none(), max_depth, max_seconds, cancel, goal, blocked_states, observer,
                         progress_interval);
            search::RolloutIwOptions opts;
            opts.control = cs.control;
            opts.ordering = parse_ordering(ordering);
            opts.seed = int_arg<u64>(seed, "seed");
            if (auto v = opt_int_arg<u64>(max_rollouts, "max_rollouts", 0))
                opts.max_rollouts = *v;
            if (auto v = opt_int_arg<u32>(incumbent_bound, "incumbent_bound", 0))
                opts.incumbent_bound = *v;
            opts.canonical_order = canonical_order;
            if (!start.is_none())
                opts.start = state_for(o, start, "start");
            const Task& t = *o.core->task;
            search::RolloutIwResult r = run_detached(cs, [&] { return search::rollout_iw(t, opts); });
            return PyRolloutIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "ordering"_a = search::ActionOrdering::InOrder, "seed"_a = 0,
        "max_rollouts"_a = nb::none(), "incumbent_bound"_a = nb::none(), "canonical_order"_a = true,
        "start"_a = nb::none(), "max_states"_a = nb::none(), "max_depth"_a = nb::none(),
        "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        paragraphs({"Rollout IW(1) (Bandres et al. 2018): "
                     "rollouts from the root over a tree of action sequences, pruned by the minimum depth of each "
                     "fluent atom, until a goal is found (the goal is tested on generation), the root is solved "
                     "(EXHAUSTED; root_solved) or a budget stops it.\n\nordering: an ActionOrdering or its name "
                     "('in_order', 'randomized', 'direct_goal_achiever_first', 'goal_regression_relevance', "
                     "'mixed_regression_random'); seed: of the randomized orderings (a "
                     "SplitMix64; the same seed gives the same run on every platform); max_rollouts (OUT_OF_STATES when hit); "
                     "incumbent_bound: a plan of this length is known elsewhere. max_states counts generated tree "
                     "nodes; there is no max_expanded. ",
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc})
            .c_str());

    m.def(
        "find_rollouts_parallel",
        [](TaskArg task, SeedsArg seeds, IntArg max_width, IntArg threads, IntArg max_next_layer_states,
           bool report_landing_states, bool report_co_occurrence, WidthZeroArg width_zero, bool optimize_iw1,
           bool witness_pruning, bool canonical_order, StateArg start, IntArg max_states,
           IntArg max_expanded,
           IntArg max_depth, FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states,
           ObserverArg observer, IntArg progress_interval, Arg<ann::TreeRecord> tree) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::ParallelRolloutOptions opts;
            opts.iw = iw_options(cs, o, int_arg<u32>(max_width, "max_width"), width_zero, optimize_iw1, witness_pruning,
                                 canonical_order, start);
            opts.seeds = int_list_arg<u64>(seeds, "seeds");
            opts.threads = threads_arg(threads, "threads");
            if (auto v = opt_int_arg<u32>(max_next_layer_states, "max_next_layer_states", 0))
                opts.max_next_layer_states = *v;
            opts.report_landing_states = report_landing_states;
            opts.report_co_occurrence = report_co_occurrence;
            opts.iw.tree = tree_record(tree);
            const Task& t = *o.core->task;
            search::ParallelRolloutsResult r = run_detached(cs, [&] { return search::find_rollouts_parallel(t, opts); });
            return PyParallelRollouts{std::make_shared<const search::ParallelRolloutsResult>(std::move(r)),
                                      opts.seeds, o};
        },
        "task"_a, "seeds"_a, nb::kw_only(), "max_width"_a = 2, "threads"_a = 1,
        "max_next_layer_states"_a = nb::none(), "report_landing_states"_a = false, "report_co_occurrence"_a = false,
        "width_zero"_a = "expand_depth_one", "optimize_iw1"_a = true, "witness_pruning"_a = false,
        "canonical_order"_a = true, "start"_a = nb::none(), MYMYR_CONTROL_ARGS, "tree"_a = nb::none(),
        paragraphs({"Parallel IW rollouts: one IW(max_width) ladder per seed with a randomized layer order seeded by it, on "
                     "`threads` threads (0: one per hardware thread, capped at len(seeds)) over the one task. For a seed a "
                     "rollout's result is the same at every thread count. Per rollout: the ladder, the reached atoms, "
                     "the number of distinct states created and, when reported, the landing states (per fluent atom "
                     "the first created state holding it) and co-occurrence (per fluent atom the atoms of the created "
                     "states holding it). max_seconds spans the whole batch, the other budgets apply per rollout.",
                    k_tree_doc,
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc,
                    k_parallel_doc,
                    " A worker is a rollout (an index into seeds)."})
            .c_str());

    m.def(
        "intersect_co_occurrence",
        [](RolloutsArg results) {
            Owner o;
            std::shared_ptr<const search::ParallelRolloutsResult> batch;
            std::vector<search::RolloutResult> copies = collect_rollouts(results, o, batch);
            if (!o.core)
                return CoOccurrenceDict{nb::dict()};
            CoOccurrence rows;
            {
                nb::gil_scoped_release release;
                rows = batch ? search::intersect_co_occurrence(batch->rollouts) : search::intersect_co_occurrence(copies);
            }
            return co_occurrence_dict(o, rows);
        },
        "results"_a,
        "Per atom of the first rollout, the intersection of its co-occurrence rows "
        "over all rollouts (run with report_co_occurrence).");

    m.def(
        "merge_landing_states",
        [](RolloutsArg results) {
            Owner o;
            std::shared_ptr<const search::ParallelRolloutsResult> batch;
            std::vector<search::RolloutResult> copies = collect_rollouts(results, o, batch);
            search::MergedLandingStates merged;
            {
                nb::gil_scoped_release release;
                merged = batch ? search::merge_landing_states(batch->rollouts) : search::merge_landing_states(copies);
            }
            if (!o.core)
                throw nb::value_error("mymyr: merge_landing_states needs at least one rollout");
            return PyMergedLandingStates{std::move(merged), o};
        },
        "results"_a,
        "The distinct landing states over a batch (run with report_landing_states), and per rollout the indices of its "
        "landing states among them.");

    m.def(
        "atomic_goal_portfolio",
        [](TaskArg task, IntArg num_rollout_workers, IntArg threads, IntArg seed, OrderingsArg rollout_orderings,
           bool canonical_order, StateArg start, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::PortfolioOptions opts;
            opts.control = cs.control;
            opts.num_rollout_workers = int_arg<u32>(num_rollout_workers, "num_rollout_workers");
            opts.threads = threads_arg(threads, "threads");
            opts.seed = int_arg<u64>(seed, "seed");
            opts.rollout_orderings = parse_orderings(rollout_orderings);
            opts.canonical_order = canonical_order;
            if (!start.is_none())
                opts.start = state_for(o, start, "start");
            const Task& t = *o.core->task;
            search::PortfolioResult r = run_detached(cs, [&] { return search::atomic_goal_portfolio(t, opts); });
            return PyPortfolioResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "num_rollout_workers"_a = 4, "threads"_a = 1, "seed"_a = 0,
        "rollout_orderings"_a = nb::none(), "canonical_order"_a = true, "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        paragraphs({"The atomic-goal portfolio: one IW(1) certifier (worker 0) and num_rollout_workers "
                     "goal-guided Rollout IW workers (1..K) under one coordination: the best plan's length prunes the "
                     "rollouts, and the certifier's completed depths certify a plan as shortest in the width-1 space.\n\n"
                     "rollout_orderings: cycled over the workers, each an ordering or (ordering, seed); None: "
                     "the cycle DIRECT_GOAL_ACHIEVER_FIRST, MIXED_REGRESSION_RANDOM, RANDOMIZED, "
                     "GOAL_REGRESSION_RELEVANCE with seeds seed + k - 1. threads=1 runs the workers one after "
                     "the other on the calling thread (reproducible); more threads (0: one per hardware thread) run them side by side "
                     "(the result then depends on timing).\n\ngoal: the atomic goal (e.g. "
                     "goal=[['(on a b)']]); max_expanded caps the expansions of all workers together, max_states "
                     "applies per worker, max_seconds to the call.",
                    k_control_doc,
                    k_transition_doc,
                    k_iw_family_doc,
                    k_parallel_doc,
                    " A worker is 0 (the certifier) or 1..K."})
            .c_str());
}
}  // namespace mymyr::python
