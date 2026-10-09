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

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/perfect.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/search/goal.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/iw_family.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/landmarks/lifted.hpp"
#include "mymyr/task/task.hpp"

#include <type_traits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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
    if (s == "unit")
        return heuristics::Costs::Unit;
    if (s == "real")
        return heuristics::Costs::Real;
    throw nb::value_error("mymyr: costs must be 'unit' or 'real'");
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
        nb::gil_scoped_acquire acquire;
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
        nb::gil_scoped_acquire acquire;
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
        nb::gil_scoped_acquire acquire;
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

CoOccurrenceDict co_occurrence_dict(const Owner& o, const CoOccurrence& rows)
{
    AtomCache atoms(o);
    CoOccurrenceDict out{nb::dict()};
    for (const auto& [c, row] : rows)
        out[atoms.get(c)] = atoms.list(row);
    return out;
}

std::string status_or_none(bool valid, SearchStatus s) { return valid ? std::string(search::to_string(s)) : "None"; }

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
    if (s == "direct_goal_achiever_first" || s == "dgaf")
        return search::ActionOrdering::DirectGoalAchieverFirst;
    if (s == "goal_regression_relevance" || s == "regression")
        return search::ActionOrdering::GoalRegressionRelevance;
    if (s == "mixed_regression_random" || s == "mixed")
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
        l.max_next_layer_states = *v;  // checked by the search (0, or a limit with 'queue', is an error)
    if (auto v = opt_int_arg<u32>(beam_width, "beam_width", 0))
        l.beam_width = *v;  // checked by the search, as above
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
    l.beam_chunk = int_arg<u32>(beam_chunk, "beam_chunk");  // checked by the search (0 is an error)
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
using BeamNoveltyArg = Arg<ann::BeamNovelty>;
using OrderingArg = Arg<std::variant<search::ActionOrdering, std::string>>;
using OrderingsArg =
    Arg<nb::typed<nb::sequence, std::variant<search::ActionOrdering, std::string,
                                             nb::typed<nb::tuple, std::variant<search::ActionOrdering, std::string>, int>>>>;
using RolloutsArg = Arg<std::variant<PyParallelRollouts, nb::typed<nb::sequence, PyRolloutResult>>>;
using AtomList = nb::typed<nb::list, PyGroundAtom>;

double py_evaluate(PyHeuristic& self, StateArg state)
{
    const State s = state_for(self.o, state, "state");
    nb::gil_scoped_release release;
    std::lock_guard lock(self.m);
    return self.h->evaluate(s.view());
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
                               datasets::to_string(r.status) + " after " + std::to_string(r.states) +
                               " states); raise max_states / max_seconds or pass Heuristic.perfect(space)")
                                  .c_str());
    return heuristics::perfect(std::move(r.space), costs);
}

// ------------------------------------------------------------------------------------------------ best-first

template<class Search>
auto best_first(Search search_fn, nb::handle task, nb::handle heuristic, nb::handle costs,
                             nb::handle evaluation, nb::handle store, nb::handle queue, nb::handle start, bool reopen,
                             bool lazy_requeue, bool preferred_operators, u32 preferred_weight, u32 standard_weight,
                             u32 beam_width, bool witness_pruning, nb::handle symmetry_pruning, nb::handle max_states,
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
    opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
    const Task& t = *o.core->task;
    auto r = run_detached(cs, [&] { return search_fn(t, opts); });
    if constexpr (std::is_same_v<decltype(r), search::AStarIwResult>)
    {
        const auto novelty = r.novelty;
        return PyAStarIwResult{{std::move(r), o}, novelty};
    }
    else
        return PyBestFirstResult{std::move(r), o};
}

using BfFn = search::BestFirstResult (*)(const Task&, const search::BestFirstOptions&);

PyBestFirstResult bf_call(BfFn fn, nb::handle task, nb::handle heuristic, nb::handle costs, nb::handle evaluation,
                          nb::handle store, nb::handle queue, nb::handle start, bool reopen, bool lazy_requeue,
                          bool preferred_operators, u32 preferred_weight, u32 standard_weight, u32 beam_width,
                          bool witness_pruning, nb::handle symmetry_pruning,
                          nb::handle max_states, nb::handle max_expanded, nb::handle max_depth,
                          nb::handle max_seconds, nb::handle cancel, nb::handle goal, nb::handle blocked_states,
                          nb::handle observer, nb::handle progress_interval)
{
    return best_first(fn, task, heuristic, costs, evaluation, store, queue, start, reopen, lazy_requeue,
                      preferred_operators, preferred_weight, standard_weight, beam_width, witness_pruning,
                      symmetry_pruning, max_states,
                      max_expanded, max_depth, max_seconds, cancel, goal, blocked_states, observer, progress_interval);
}

// ------------------------------------------------------------------------------------------------ IW / SIW

search::IwOptions iw_options(ControlScope& cs, const Owner& o, u32 max_arity, nb::handle width_zero, bool optimize_iw1,
                             bool witness_pruning, bool canonical_order, nb::handle symmetry_pruning,
                             nb::handle start,
                             nb::handle transition_ordering = nb::none())
{
    search::IwOptions opts;
    opts.control = cs.control;
    opts.max_arity = max_arity;
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
    opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
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
                                        nb::handle beam_chunk, nb::handle threads, bool witness_pruning, bool canonical_order, nb::handle symmetry_pruning,
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
    opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
    if (!start.is_none())
        opts.start = state_for(o, start, "start");
    return opts;
}

#define MYMYR_CONTROL_ARGS                                                                                             \
    "max_states"_a = nb::none(), "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(),                            \
        "max_seconds"_a = nb::none(), "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), \
        "observer"_a = nb::none(), "progress_interval"_a = nb::none()

#define MYMYR_IW_ARGS                                                                                                  \
    "task"_a, nb::kw_only(), "max_arity"_a = 2, "width_zero"_a = "expand_depth_one", "optimize_iw1"_a = true,          \
        "witness_pruning"_a = false, "canonical_order"_a = true, "symmetry_pruning"_a = "off", "start"_a = nb::none(), \
        MYMYR_CONTROL_ARGS, "transition_ordering"_a = nb::none(), MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG

#define MYMYR_LANDMARK_ARGS                                                                                            \
    "landmarks"_a = nb::none(), "disjunctive"_a = false, "all_private"_a = false, "unshared_atoms"_a = nb::none(),     \
        "max_dense_bytes"_a = nb::none()

#define MYMYR_LAYER_ARGS                                                                                               \
    "layer_order"_a = "queue", "seed"_a = 0, "max_next_layer_states"_a = nb::none(),                                  \
        "prefer_more_satisfied_goals"_a = true, "beam_width"_a = nb::none(), "beam_novelty"_a = "all_tested",         \
        "randomize_ties"_a = false, "beam_chunk"_a = 1024

/// The IW family's thread count (IwOptions::threads, AbstractedIwOptions::threads), after the layer arguments.
#define MYMYR_THREADS_ARG "threads"_a = 1

#define MYMYR_SYMMETRY_DOC                                                                                             \
    "symmetry_pruning ('off' or 'wl1'; successor/symmetry.hpp): 'wl1' expands only the actions whose parameters are "  \
    "representatives of the objects' colour refinement classes in the state (mimir's WL1 pruning). It gives up "       \
    "completeness and optimality: the search may miss every plan (report a solvable task unsolvable) and A* may "      \
    "return a costlier plan; a plan it finds is valid."

const char* k_control_doc =
    "Common keyword arguments (search/control.hpp): max_states, max_expanded, max_depth, max_seconds (budgets; unset = "
    "unlimited; max_states counts the stored states, the start state included, and the search stops with "
    "OUT_OF_STATES at the new state that fills them, inside its parent's expansion), cancel (a CancelToken, requestable from any thread), goal (None: the task's goal; a callable "
    "state -> bool; a GroundCondition; or a sequence of goals, each a GroundCondition or a sequence of ground literals "
    "and atoms as Task.ground_condition takes them, any of which counts: static, fluent and derived literals of either "
    "polarity and numeric constraints), blocked_states (States never entered), observer (an object with any of on_start(state), "
    "on_expand(id, state), on_generate(parent, action, child, state, is_new), on_prune(parent, action, state), "
    "on_pass(arity, stats), on_solution(plan, cost), on_progress(stats) -> bool, on_end(status, stats)), "
    "progress_interval (expansions between on_progress calls). The search runs with the thread state detached; "
    "callbacks attach it. An exception in a callback cancels the search and is re-raised. " MYMYR_SYMMETRY_DOC;

const char* k_heuristic_doc =
    "A heuristic written in Python is a callable state -> float or an object with evaluate(state) -> float "
    "(mymyr._typing.HeuristicObject); math.inf marks a dead end. Optional methods: evaluate_batch(states) -> one float "
    "per state (eager A* and eager GBFS then evaluate the new successors of an expansion in one call, beam search those "
    "of a layer; an object may have evaluate_batch alone) and preferred_actions() -> the preferred actions of the state "
    "of the last evaluate call (the lazy searches' preferred list). It is called on the thread running the search, "
    "with the thread state attached; an exception cancels the search and is re-raised. ";

const char* k_iw_family_doc =
    " The IW family variants also call the observer's on_transition(parent, action, child, state, outcome) for every "
    "transition (outcome: a TransitionOutcome; child None when pruned). Numeric tasks raise ValueError.";

const char* k_parallel_doc =
    " Parallel hooks (the make_worker protocol): an observer with make_worker(k) -> observer | None, or a goal callable "
    "with make_worker(k) -> callable | None, gives worker k its own object, called from that worker's thread (the "
    "lifecycle events on_start, on_pass, on_solution, on_end stay on the root observer, on the calling thread). "
    "Without make_worker (or when it returns None) the search runs on the calling thread alone.";

const char* k_landmark_doc =
    " landmarks: a FactLandmarkGraph of the task, 'approximate' or 'lifted' (approximate_fact_landmarks / "
    "lifted_fact_landmarks with default options), or None (no landmarks: every state's coordinate is BOT); "
    "disjunctive (a rank per disjunctive landmark), all_private (with disjunctive: a rank per member), unshared_atoms "
    "(with disjunctive: members given their own rank), max_dense_bytes (dense table budget, default 256 MiB). ";

const char* k_layer_doc =
    "layer_order (search/layer_ordering.hpp; mimir's layer ordering strategies): 'queue' (plain BrFS, the default) or "
    "layer by layer, each next layer reordered before it is expanded: 'in_order' (as generated), 'reverse', "
    "'randomized' (seed: the SplitMix64 seed) or 'goal_count' (stable, by the number of satisfied goal literals, most "
    "first; prefer_more_satisfied_goals=False: fewest first). max_next_layer_states (ordered kinds only): once the "
    "next layer holds that many states, the rest of the current layer is not expanded. beam_width (ordered kinds "
    "only, not with max_next_layer_states): the whole next layer is generated, ordered, and only its first "
    "beam_width states are kept (goal_count: the best scores, ties in generation order or, with randomize_ties, by "
    "random tokens drawn from seed); the others are never expanded. beam_novelty (with beam_width): 'all_tested' "
    "(the novelty test marks every successor it admits, kept or not) or 'survivors_only' (successors are tested "
    "read-only; the kept ones are replayed in rank order and only those that still add a tuple stay and mark it, so "
    "a layer can end up smaller than beam_width; not supported by liw; in brfs the "
    "two modes are the same) or 'relaxed_survivors_only' (goal_count only; not supported by liw: the layer's "
    "transitions, in generation order, are split into min(threads, ceil(n / beam_chunk)) contiguous parts, each part "
    "keeps its best beam_width successors by the read-only test, and the kept ones of all parts enter in rank order, "
    "the states seen before skipped, until beam_width entered; then the survivors_only replay. The parts make the "
    "result depend on the thread count; it is deterministic for a thread count and seed). threads (brfs and the IW "
    "family; > 1 needs a beam): the beam's layer step on that many threads (0: one per hardware thread), with the "
    "result of one thread for 'all_tested' and 'survivors_only'; an observer, blocked states or a successor order "
    "keep it on the calling thread.";
}  // namespace

void bind_search(nb::module_& parent)
{
    nb::module_ m = parent.def_submodule("_search", "The search family and heuristics (mymyr.search)");

    nb::enum_<SearchStatus>(m, "Status", "Outcome of a search (search/control.hpp).")
        .value("SOLVED", SearchStatus::Solved)
        .value("EXHAUSTED", SearchStatus::Exhausted)
        .value("OUT_OF_STATES", SearchStatus::OutOfStates)
        .value("OUT_OF_TIME", SearchStatus::OutOfTime)
        .value("CANCELLED", SearchStatus::Cancelled)
        .value("FAILED", SearchStatus::Failed)
        .value("UNSOLVABLE", SearchStatus::Unsolvable)
        .def("__str__", [](SearchStatus s) { return std::string(search::to_string(s)); });

    nb::enum_<search::ActionOrdering>(m, "ActionOrdering",
                                      "Rollout IW's action ordering (search/rollout_iw.hpp; mimir's "
                                      "RolloutIWActionOrderingKind): guidance, never pruning.")
        .value("IN_ORDER", search::ActionOrdering::InOrder)
        .value("RANDOMIZED", search::ActionOrdering::Randomized)
        .value("DIRECT_GOAL_ACHIEVER_FIRST", search::ActionOrdering::DirectGoalAchieverFirst)
        .value("GOAL_REGRESSION_RELEVANCE", search::ActionOrdering::GoalRegressionRelevance)
        .value("MIXED_REGRESSION_RANDOM", search::ActionOrdering::MixedRegressionRandom)
        .def("__str__", [](search::ActionOrdering o) { return std::string(search::to_string(o)); });

    nb::enum_<search::TransitionOutcome>(m, "TransitionOutcome",
                                         "What became of a transition's successor (observer.on_transition; "
                                         "search/control.hpp).")
        .value("OPENED", search::TransitionOutcome::Opened)
        .value("REOPENED", search::TransitionOutcome::Reopened)
        .value("DUPLICATE", search::TransitionOutcome::Duplicate)
        .value("PRUNED", search::TransitionOutcome::Pruned)
        .value("DEAD_END", search::TransitionOutcome::DeadEnd)
        .value("GOAL", search::TransitionOutcome::Goal);

    bind_landmarks(m);

    nb::class_<PyCancelToken>(m, "CancelToken",
                              "Cooperative cancellation: request() from any thread stops every search given this token.")
        .def(nb::init<>())
        .def("request", [](const PyCancelToken& t) { t.token.request(); })
        .def_prop_ro("requested", [](const PyCancelToken& t) { return t.token.requested(); })
        .def("__repr__", [](const PyCancelToken& t) {
            return std::string("CancelToken(requested=") + (t.token.requested() ? "True" : "False") + ")";
        });

    nb::class_<search::SearchStatistics>(m, "Statistics")
        .def_ro("expanded", &search::SearchStatistics::expanded)
        .def_ro("generated", &search::SearchStatistics::generated)
        .def_ro("states", &search::SearchStatistics::states)
        .def_ro("pruned", &search::SearchStatistics::pruned)
        .def_ro("seconds", &search::SearchStatistics::seconds)
        .def("__repr__", [](const search::SearchStatistics& s) { return "Statistics(" + stats_repr(s) + ")"; });

    nb::class_<search::IwPassStatistics>(m, "IwPass", "Statistics of one IW(k) pass (mimir's per-pass counts).")
        .def_ro("arity", &search::IwPassStatistics::arity)
        .def_ro("status", &search::IwPassStatistics::status)
        .def_ro("expanded", &search::IwPassStatistics::expanded)
        .def_ro("generated", &search::IwPassStatistics::generated)
        .def_ro("generated_in_tree", &search::IwPassStatistics::generated_in_tree)
        .def_ro("skipped", &search::IwPassStatistics::skipped)
        .def_ro("blocked", &search::IwPassStatistics::blocked)
        .def_ro("seconds", &search::IwPassStatistics::seconds)
        .def_ro("placeholder", &search::IwPassStatistics::placeholder)
        .def("__repr__", [](const search::IwPassStatistics& p) {
            return "IwPass(arity=" + std::to_string(p.arity) + ", status=" + search::to_string(p.status) +
                   ", expanded=" + std::to_string(p.expanded) + ", generated=" + std::to_string(p.generated) +
                   ", generated_in_tree=" + std::to_string(p.generated_in_tree) + ")";
        });

    nb::class_<search::SiwSubproblem>(m, "SiwSubproblem")
        .def_ro("status", &search::SiwSubproblem::status)
        .def_ro("unsatisfied_at_start", &search::SiwSubproblem::unsatisfied_at_start)
        .def_ro("plan_length", &search::SiwSubproblem::plan_length)
        .def_ro("effective_width", &search::SiwSubproblem::effective_width)
        .def_ro("passes", &search::SiwSubproblem::passes);

    nb::class_<heuristics::HeuristicStats>(m, "HeuristicStatistics")
        .def_ro("evaluations", &heuristics::HeuristicStats::evaluations)
        .def_ro("grounded", &heuristics::HeuristicStats::grounded)
        .def_ro("lifted", &heuristics::HeuristicStats::lifted)
        .def_ro("dead_ends", &heuristics::HeuristicStats::dead_ends)
        .def_ro("grounding_seconds", &heuristics::HeuristicStats::grounding_seconds);

    // results ---------------------------------------------------------------------------------------------------------
    nb::class_<PyIwResult>(m, "IwResult")
        .def_prop_ro("status", [](const PyIwResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PyIwResult& x) { return x.r.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyIwResult& x) { return plan_list(x.o, x.r.plan); })
        .def_prop_ro("cost", [](const PyIwResult& x) { return x.r.cost; })
        .def_prop_ro("cost_exact", [](const PyIwResult& x) { return x.r.cost_exact; })
        .def_prop_ro("goal_state", [](const PyIwResult& x) { return state_or_none(x.o, x.r.goal_state); })
        .def_prop_ro("passes", [](const PyIwResult& x) { return x.r.passes; })
        .def_prop_ro("total", [](const PyIwResult& x) { return x.r.total; })
        .def_prop_ro("effective_width", [](const PyIwResult& x) { return x.r.effective_width; })
        .def_prop_ro("message", [](const PyIwResult& x) { return x.r.message; })
        .def_prop_ro("fluent_slots", [](const PyIwResult& x) { return x.r.fluent_slots; })
        .def_prop_ro("peak_table_bytes", [](const PyIwResult& x) { return x.r.peak_table_bytes; })
        .def_prop_ro("peak_node_bytes", [](const PyIwResult& x) { return x.r.peak_node_bytes; })
        .def("__repr__", [](const PyIwResult& x) {
            return std::string("IwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) +
                   ", effective_width=" + std::to_string(x.r.effective_width) + ", " + stats_repr(x.r.total) + ")";
        });

    nb::class_<PySiwResult>(m, "SiwResult")
        .def_prop_ro("status", [](const PySiwResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PySiwResult& x) { return x.r.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PySiwResult& x) { return plan_list(x.o, x.r.plan); },
                     "The whole plan when solved, else empty.")
        .def_prop_ro("partial_plan", [](const PySiwResult& x) { return plan_list(x.o, x.r.partial_plan); },
                     "Unless solved: the subplans of the subproblems solved before the one that failed, from the "
                     "start state to where that one started (empty when solved).")
        .def_prop_ro("cost", [](const PySiwResult& x) { return x.r.cost; })
        .def_prop_ro("cost_exact", [](const PySiwResult& x) { return x.r.cost_exact; })
        .def_prop_ro("goal_state", [](const PySiwResult& x) { return state_or_none(x.o, x.r.goal_state); })
        .def_prop_ro("subproblems", [](const PySiwResult& x) { return x.r.subproblems; })
        .def_prop_ro("total", [](const PySiwResult& x) { return x.r.total; })
        .def_prop_ro("max_effective_width", [](const PySiwResult& x) { return x.r.max_effective_width; })
        .def_prop_ro("message", [](const PySiwResult& x) { return x.r.message; })
        .def("__repr__", [](const PySiwResult& x) {
            return std::string("SiwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) +
                   (x.r.partial_plan.empty() ? std::string{}
                                             : ", partial_plan_length=" + std::to_string(x.r.partial_plan.size())) +
                   ", cost=" + cost_repr(x.r.cost) + ", subproblems=" + std::to_string(x.r.subproblems.size()) + ", " +
                   stats_repr(x.r.total) + ")";
        });

    nb::class_<search::AStarIwNoveltyStatistics>(m, "AStarIwNoveltyStatistics")
        .def_ro("rejected", &search::AStarIwNoveltyStatistics::rejected)
        .def_ro("stale", &search::AStarIwNoveltyStatistics::stale)
        .def_ro("stale_g", &search::AStarIwNoveltyStatistics::stale_g)
        .def_ro("probes", &search::AStarIwNoveltyStatistics::probes)
        .def_ro("updates", &search::AStarIwNoveltyStatistics::updates)
        .def_ro("pop_tests", &search::AStarIwNoveltyStatistics::pop_tests)
        .def_ro("table_bytes", &search::AStarIwNoveltyStatistics::table_bytes);

    nb::class_<PyBestFirstResult>(m, "BestFirstResult")
        .def_prop_ro("status", [](const PyBestFirstResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PyBestFirstResult& x) { return x.r.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyBestFirstResult& x) { return plan_list(x.o, x.r.plan); })
        .def_prop_ro("cost", [](const PyBestFirstResult& x) { return x.r.cost; })
        .def_prop_ro("goal_state", [](const PyBestFirstResult& x) { return state_or_none(x.o, x.r.goal_state); })
        .def_prop_ro("stats", [](const PyBestFirstResult& x) { return x.r.stats; })
        .def_prop_ro("evaluations", [](const PyBestFirstResult& x) { return x.r.evaluations; })
        .def_prop_ro("dead_ends", [](const PyBestFirstResult& x) { return x.r.dead_ends; })
        .def_prop_ro("reopened", [](const PyBestFirstResult& x) { return x.r.reopened; })
        .def_prop_ro("layers", [](const PyBestFirstResult& x) { return x.r.layers; })
        .def_prop_ro("initial_h", [](const PyBestFirstResult& x) { return x.r.initial_h; })
        .def_prop_ro("heuristic", [](const PyBestFirstResult& x) { return x.r.heuristic; })
        .def_prop_ro("setup_seconds", [](const PyBestFirstResult& x) { return x.r.setup_seconds; })
        .def_prop_ro("algorithm", [](const PyBestFirstResult& x) { return x.r.algorithm; })
        .def_prop_ro("store", [](const PyBestFirstResult& x) { return x.r.store; })
        .def_prop_ro("queue", [](const PyBestFirstResult& x) { return x.r.queue; })
        .def_prop_ro("message", [](const PyBestFirstResult& x) { return x.r.message; })
        .def_prop_ro("store_bytes", [](const PyBestFirstResult& x) { return x.r.store_bytes; })
        .def("__repr__", [](const PyBestFirstResult& x) {
            return "BestFirstResult(algorithm=" + x.r.algorithm + ", status=" + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) + ", " +
                   stats_repr(x.r.stats) + ")";
        });

    nb::class_<PyBrfsResult>(m, "BrfsResult")
        .def_prop_ro("status", [](const PyBrfsResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PyBrfsResult& x) { return x.r.solved; })
        .def_prop_ro("exhausted", [](const PyBrfsResult& x) { return x.r.exhausted; })
        .def_prop_ro("plan", [](const PyBrfsResult& x) { return plan_list(x.o, x.r.plan); })
        .def_prop_ro("states", [](const PyBrfsResult& x) { return x.r.states; })
        .def_prop_ro("expanded", [](const PyBrfsResult& x) { return x.r.expanded; })
        .def_prop_ro("generated", [](const PyBrfsResult& x) { return x.r.generated; })
        .def_prop_ro("goal_states", [](const PyBrfsResult& x) { return x.r.goal_states; })
        .def_prop_ro("layers", [](const PyBrfsResult& x) { return x.r.layers; })
        .def_prop_ro("seconds", [](const PyBrfsResult& x) { return x.r.search_s; })
        .def_prop_ro("store", [](const PyBrfsResult& x) { return x.r.store; })
        .def_prop_ro("store_bytes", [](const PyBrfsResult& x) { return x.r.store_bytes; })
        .def_prop_ro("threads", [](const PyBrfsResult& x) { return x.r.threads; })
        .def_prop_ro("fingerprint", [](const PyBrfsResult& x) { return x.r.fingerprint; })
        .def("__repr__", [](const PyBrfsResult& x) {
            return "BrfsResult(states=" + std::to_string(x.r.states) + ", expanded=" + std::to_string(x.r.expanded) +
                   ", generated=" + std::to_string(x.r.generated) + ", layers=" + std::to_string(x.r.layers) +
                   ", solved=" + (x.r.solved ? "True" : "False") + ", exhausted=" + (x.r.exhausted ? "True" : "False") +
                   ")";
        });

    // parallel rollouts and portfolio: results ----------------------------------------------------------------------
    using RS = search::RolloutIwStatistics;
    nb::class_<RS>(m, "RolloutIwStatistics", "Counters of a Rollout IW run (mimir's rollout_iw::Statistics).")
        .def_ro("rollouts", &RS::rollouts)
        .def_ro("generated", &RS::generated, "New tree nodes.")
        .def_ro("expanded", &RS::expanded, "Nodes whose actions were materialized.")
        .def_ro("feature_depth_improvements", &RS::feature_depth_improvements)
        .def_ro("case1", &RS::case1)
        .def_ro("case2", &RS::case2)
        .def_ro("case3", &RS::case3)
        .def_ro("case4", &RS::case4)
        .def_ro("solved_propagations", &RS::solved_propagations)
        .def_ro("dead_ends", &RS::dead_ends)
        .def_ro("depth_bound_prunings", &RS::depth_bound_prunings)
        .def_ro("incumbent_bound_prunings", &RS::incumbent_bound_prunings)
        .def_ro("blocked", &RS::blocked)
        .def_ro("max_rollout_depth", &RS::max_rollout_depth)
        .def_ro("tree_nodes", &RS::tree_nodes)
        .def_ro("seconds", &RS::seconds)
        .def("__repr__", [](const RS& s) {
            return "RolloutIwStatistics(rollouts=" + std::to_string(s.rollouts) + ", generated=" + std::to_string(s.generated) +
                   ", expanded=" + std::to_string(s.expanded) + ", tree_nodes=" + std::to_string(s.tree_nodes) + ")";
        });

    nb::class_<PyRolloutIwResult>(m, "RolloutIwResult")
        .def_prop_ro("status", [](const PyRolloutIwResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PyRolloutIwResult& x) { return x.r.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyRolloutIwResult& x) { return plan_list(x.o, x.r.plan); })
        .def_prop_ro("cost", [](const PyRolloutIwResult& x) { return x.r.cost; })
        .def_prop_ro("cost_exact", [](const PyRolloutIwResult& x) { return x.r.cost_exact; })
        .def_prop_ro("goal_state", [](const PyRolloutIwResult& x) { return state_or_none(x.o, x.r.goal_state); })
        .def_prop_ro("statistics", [](const PyRolloutIwResult& x) { return x.r.statistics; })
        .def_prop_ro("total", [](const PyRolloutIwResult& x) { return x.r.statistics.statistics(); },
                     "expanded, generated, states (tree nodes) and pruned as Statistics.")
        .def_prop_ro("root_solved", [](const PyRolloutIwResult& x) { return x.r.root_solved; },
                     "The reachable width-1 space was exhausted (relative to the depth and incumbent bounds).")
        .def_prop_ro("message", [](const PyRolloutIwResult& x) { return x.r.message; })
        .def_prop_ro("fluent_slots", [](const PyRolloutIwResult& x) { return x.r.fluent_slots; })
        .def("__repr__", [](const PyRolloutIwResult& x) {
            return std::string("RolloutIwResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", cost=" + cost_repr(x.r.cost) +
                   ", rollouts=" + std::to_string(x.r.statistics.rollouts) + ", root_solved=" +
                   (x.r.root_solved ? "True" : "False") + ")";
        });

    nb::class_<PyLandingState>(m, "LandingState",
                               "A state a rollout created that first made some fluent atom true (mimir's "
                               "IWLandingState).")
        .def_prop_ro("state", [](const PyLandingState& x) { return make_state(x.o, State(x.l.state)); })
        .def_prop_ro("atoms", [](const PyLandingState& x) { return atom_list(x.o, x.l.atoms); },
                     "Its fluent atoms (ascending by canonical id).")
        .def_prop_ro("direct_dead_end", [](const PyLandingState& x) { return x.l.direct_dead_end; },
                     "No action is applicable in it.")
        .def("__repr__", [](const PyLandingState& x) {
            return "LandingState(atoms=" + std::to_string(x.l.atoms.size()) + ", direct_dead_end=" +
                   (x.l.direct_dead_end ? "True" : "False") + ")";
        });

    nb::class_<PyRolloutResult>(m, "RolloutResult",
                                "One rollout of find_rollouts_parallel: its IW ladder and what its private state "
                                "repository recorded (mimir's IWRolloutResult). Atoms are GroundAtoms, equal "
                                "across rollouts and thread counts.")
        .def_prop_ro("seed", [](const PyRolloutResult& x) { return x.seed; })
        .def_prop_ro("search", [](const PyRolloutResult& x) { return PyIwResult{x.get().search, x.o}; },
                     "The rollout's IW ladder (status, plan, passes, ...).")
        .def_prop_ro("status", [](const PyRolloutResult& x) { return x.get().search.status; })
        .def_prop_ro("solved", [](const PyRolloutResult& x) { return x.get().search.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyRolloutResult& x) { return plan_list(x.o, x.get().search.plan); })
        .def_prop_ro("passes", [](const PyRolloutResult& x) { return x.get().search.passes; })
        .def_prop_ro("reached_fluent_atoms", [](const PyRolloutResult& x) { return atom_list(x.o, x.get().reached_fluent_atoms); })
        .def_prop_ro("reached_derived_atoms", [](const PyRolloutResult& x) { return atom_list(x.o, x.get().reached_derived_atoms); })
        .def_prop_ro("num_states", [](const PyRolloutResult& x) { return x.get().num_states; },
                     "Distinct states the rollout created.")
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
        .def("__repr__", [](const PyRolloutResult& x) {
            const search::RolloutResult& r = x.get();
            return "RolloutResult(seed=" + std::to_string(x.seed) + ", status=" + search::to_string(r.search.status) +
                   ", num_states=" + std::to_string(r.num_states) + ", reached_fluent_atoms=" +
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
                     })
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
        .def_prop_ro("threads_used", [](const PyParallelRollouts& x) { return x.r->threads_used; })
        .def_prop_ro("message", [](const PyParallelRollouts& x) { return x.r->message; },
                     "Why the batch could not run (every rollout then FAILED), or why it ran on one thread.")
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
                     })
        .def_prop_ro("by_rollout", [](const PyMergedLandingStates& x) { return x.m.by_rollout; },
                     "Per rollout the indices of its landing states in `states`.");

    nb::class_<PyPortfolioResult>(m, "PortfolioResult",
                                  "The atomic-goal portfolio's result (mimir's AtomicGoalIWPortfolioResult).")
        .def_prop_ro("status", [](const PyPortfolioResult& x) { return x.r.status; })
        .def_prop_ro("solved", [](const PyPortfolioResult& x) { return x.r.status == SearchStatus::Solved; })
        .def_prop_ro("plan", [](const PyPortfolioResult& x) { return plan_list(x.o, x.r.plan); })
        .def_prop_ro("cost", [](const PyPortfolioResult& x) { return x.r.cost; })
        .def_prop_ro("cost_exact", [](const PyPortfolioResult& x) { return x.r.cost_exact; })
        .def_prop_ro("plan_length",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.plan_length == ~u32{0})
                             return std::nullopt;
                         return x.r.plan_length;
                     })
        .def_prop_ro("certified_optimal", [](const PyPortfolioResult& x) { return x.r.certified_optimal; },
                     "The plan is shortest in the width-1 space: the certifier found it, or completed depth "
                     "plan_length - 1.")
        .def_prop_ro("iw_lower_bound", [](const PyPortfolioResult& x) { return x.r.iw_lower_bound; })
        .def_prop_ro("iw_completed_depth",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.iw_completed_depth == search::SearchCoordination::k_no_depth)
                             return std::nullopt;
                         return x.r.iw_completed_depth;
                     })
        .def_prop_ro("winning_worker",
                     [](const PyPortfolioResult& x) -> std::optional<u32> {
                         if (x.r.winning_worker == ~u32{0})
                             return std::nullopt;
                         return x.r.winning_worker;
                     },
                     "0: the certifier, 1..K: the rollout workers; None without a plan.")
        .def_prop_ro("message", [](const PyPortfolioResult& x) { return x.r.message; })
        .def_prop_ro("certifier_ran", [](const PyPortfolioResult& x) { return x.r.certifier_ran; })
        .def_prop_ro("certifier_status",
                     [](const PyPortfolioResult& x) -> std::optional<SearchStatus> {
                         if (!x.r.certifier_ran)
                             return std::nullopt;
                         return x.r.certifier_status;
                     })
        .def_prop_ro("certifier", [](const PyPortfolioResult& x) { return PyIwResult{x.r.certifier, x.o}; },
                     "The certifier's IW(1) ladder.")
        .def_prop_ro("rollout_statistics", [](const PyPortfolioResult& x) { return x.r.rollout_statistics; },
                     "Per rollout worker, summed over its rounds.")
        .def_prop_ro("rollout_statuses",
                     [](const PyPortfolioResult& x) {
                         std::vector<std::optional<SearchStatus>> out;
                         for (usize i = 0; i < x.r.rollout_statuses.size(); ++i)
                             out.push_back(x.r.rollout_rounds[i] ? std::optional<SearchStatus>(x.r.rollout_statuses[i])
                                                                 : std::nullopt);
                         return out;
                     },
                     "Of each worker's last round; None for a worker that never started.")
        .def_prop_ro("rollout_rounds", [](const PyPortfolioResult& x) { return x.r.rollout_rounds; })
        .def_prop_ro("total_expansions", [](const PyPortfolioResult& x) { return x.r.total_expansions; })
        .def_prop_ro("threads_used", [](const PyPortfolioResult& x) { return x.r.threads_used; })
        .def_prop_ro("fluent_slots", [](const PyPortfolioResult& x) { return x.r.fluent_slots; })
        .def("__repr__", [](const PyPortfolioResult& x) {
            return std::string("PortfolioResult(status=") + search::to_string(x.r.status) +
                   ", plan_length=" + std::to_string(x.r.plan.size()) + ", certified_optimal=" +
                   (x.r.certified_optimal ? "True" : "False") + ", winning_worker=" +
                   (x.r.winning_worker == ~u32{0} ? std::string("None") : std::to_string(x.r.winning_worker)) +
                   ", certifier_status=" + status_or_none(x.r.certifier_ran, x.r.certifier_status) + ")";
        });

    // heuristics ------------------------------------------------------------------------------------------------------
    nb::class_<PyHeuristic>(m, "Heuristic",
                            "A heuristic of a task: h(state) -> float (+inf: dead end). kind: 'blind', 'goal_count' "
                            "('gc'), 'max' ('hmax'), 'add' ('hadd'), 'ff' ('hff'), 'set_additive' ('hsa', 'setadd'), "
                            "'h2' (heuristics/heuristic.hpp has their definitions); the perfect heuristic h* comes from "
                            "a state space: Heuristic.perfect(space). costs: 'unit' or 'real' (the task's action costs; "
                            "set-additive then sums the costs of its achiever set, one per supported proposition); "
                            "evaluation: 'auto', 'grounded', 'lifted' (set_additive and h2 are grounded only). "
                            "share=another Heuristic of the task reuses its grounding. Calls on one object are "
                            "serialized; use one object per thread for parallel evaluation. A search given the object "
                            "evaluates with an evaluator of its own that shares the grounding, so the object stays "
                            "callable while the search runs (from its observer, from other threads); stats counts the "
                            "calls on the object, the search's evaluations are in its result.")
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
            "task"_a, "kind"_a = "ff", nb::kw_only(), "costs"_a = "unit", "evaluation"_a = "auto",
            "share"_a = nb::none())
        .def_static(
            "perfect",
            [](const PyStateSpace& space, StrArg costs) {
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
                return std::unique_ptr<PyHeuristic>(new PyHeuristic{std::move(h), space.owner, opts, {}, space.space});
            },
            "space"_a, nb::kw_only(), "costs"_a = "unit",
            "The perfect heuristic h* of the task of a state space (mymyr.datasets.state_space(task, "
            "remove_if_unsolvable=False), without symmetry pruning): the goal distance of a state, +inf where no goal "
            "is reachable. costs: 'unit' (the number of actions) or 'real' (the transition costs). A state outside "
            "the space raises ValueError, as does a search goal other than the task's.")
        .def("__call__", &py_evaluate, "state"_a, "h(state): the heuristic value (+inf for a dead end).")
        .def("evaluate", &py_evaluate, "state"_a)
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
            [](PyHeuristic& self, StateArg state) {
                const State s = state_for(self.o, state, "state");
                nb::list applicable = nb::borrow<nb::list>(task_object(self.o.obj).attr("applicable_actions")(state));
                std::vector<Action> labels;
                for (nb::handle a : applicable)
                    labels.emplace_back(nb::cast<const PyAction&>(a).label());
                std::vector<char> keep(labels.size(), 0);
                {
                    nb::gil_scoped_release release;  // never block on the mutex while attached
                    std::lock_guard lock(self.m);
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
            "state"_a,
            "FF and set-additive: the applicable actions of `state` that are in its relaxed plan (mimir's preferred "
            "actions).")
        .def_prop_ro("kind", [](const PyHeuristic& self) { return std::string(heuristics::to_string(self.h->kind())); })
        .def_prop_ro("stats", [](PyHeuristic& self) {
            heuristics::HeuristicStats st;
            {
                nb::gil_scoped_release release;  // never block on the mutex while attached
                std::lock_guard lock(self.m);
                st = self.h->stats();
            }
            return st;
        })
        .def_prop_ro("grounded", [](PyHeuristic& self) {
            bool g = false;
            {
                nb::gil_scoped_release release;
                std::lock_guard lock(self.m);
                g = self.h->relaxed() != nullptr;
            }
            return g;
        });

    nb::class_<PyAStarIwResult, PyBestFirstResult>(m, "AStarIwResult")
        .def_prop_ro("novelty", [](const PyAStarIwResult& r) { return r.novelty; });

    // searches --------------------------------------------------------------------------------------------------------
    m.def(
        "iw",
        [](TaskArg task, IntArg max_arity, WidthZeroArg width_zero, bool optimize_iw1, bool witness_pruning,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::IwOptions opts = iw_options(cs, o, int_arg<u32>(max_arity, "max_arity"), width_zero, optimize_iw1,
                                                witness_pruning,
                                                canonical_order, symmetry_pruning, start, transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        MYMYR_IW_ARGS,
        (std::string("The IW(k) ladder, arities 0..max_arity, with mimir's conventions (search/iw.hpp; optimized "
                     "IW(1) when max_arity == 1). width_zero: 'expand_depth_one' (mimir) or 'root_only' (C#). "
                     "transition_ordering: a LandmarkTransitionOrdering that orders each layer of the width-1 pass "
                     "(mimir's find_solution_iw with a transition ordering strategy). ") +
         k_layer_doc + " " + k_control_doc)
            .c_str());

    m.def(
        "iw_pass",
        [](TaskArg task, IntArg arity, WidthZeroArg width_zero, bool witness_pruning, bool canonical_order,
           SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads) {
            const Owner o = owner_of(task);
            const u32 k = int_arg<u32>(arity, "arity");
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::IwOptions opts =
                iw_options(cs, o, k, width_zero, false, witness_pruning, canonical_order, symmetry_pruning, start,
                transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::iw_pass(t, k, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, "arity"_a, nb::kw_only(), "width_zero"_a = "expand_depth_one", "witness_pruning"_a = false,
        "canonical_order"_a = true, "symmetry_pruning"_a = "off", "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        "transition_ordering"_a = nb::none(),
        MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG, "One IW(arity) pass alone, without the ladder. See iw() for the keyword arguments.");

    m.def(
        "siw",
        [](TaskArg task, IntArg max_arity, WidthZeroArg width_zero, bool optimize_iw1, bool witness_pruning,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval, TransitionArg transition_ordering, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::SiwOptions opts = iw_options(cs, o, int_arg<u32>(max_arity, "max_arity"), width_zero, optimize_iw1,
                                                 witness_pruning,
                                                 canonical_order, symmetry_pruning, start, transition_ordering);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            const Task& t = *o.core->task;
            search::SiwResult r = run_detached(cs, [&] { return search::siw(t, opts); });
            return PySiwResult{std::move(r), o};
        },
        MYMYR_IW_ARGS,
        (std::string("Serialized IW (search/siw.hpp): IW ladders from subgoal to subgoal, each subproblem ending where "
                     "fewer goal literals are unsatisfied. Unsolved, the plan is empty and partial_plan holds the "
                     "subplans of the subproblems solved before the one that failed. ") +
         k_layer_doc + " " + k_control_doc)
            .c_str());

    m.def(
        "brfs",
        [](TaskArg task, IntArg threads, StrArg store, bool witness_pruning, bool canonical_order,
           SymmetryArg symmetry_pruning, bool deterministic_ids, IntArg max_states, bool stop_at_goal,
           bool fingerprint, LayerArg layer_order,
           IntArg seed, IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, GoalArg goal, FloatArg max_seconds, CancelArg cancel,
           ObserverArg observer, IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, nb::none(), nb::none(), nb::none(), max_seconds, cancel, goal, nb::none(), observer,
                         progress_interval);
            BrfsOptions opts;
            opts.goal = std::move(cs.control.goal);
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
            opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
            if (auto v = opt_int_arg<u64>(max_states, "max_states", 0))
                opts.max_states = *v;
            opts.stop_at_goal = stop_at_goal;
            opts.fingerprint = fingerprint;
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            const Task& t = *o.core->task;
            BrfsResult r = run_detached(cs, [&] { return brfs(t, opts); });
            return PyBrfsResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "threads"_a = 1, "store"_a = "auto", "witness_pruning"_a = true,
        "canonical_order"_a = true, "symmetry_pruning"_a = "off", "deterministic_ids"_a = true,
        "max_states"_a = nb::none(), "stop_at_goal"_a = true,
        "fingerprint"_a = false, MYMYR_LAYER_ARGS, "goal"_a = nb::none(), "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "observer"_a = nb::none(), "progress_interval"_a = nb::none(),
        (std::string("Breadth-first search over the reachable states (search/brfs.hpp): single-threaded, or "
                     "layer-synchronous on `threads` threads with ids independent of the thread count. A search that "
                     "expands a goal state is SOLVED, with the plan to the first one (a shortest plan); stop_at_goal "
                     "(the default) stops there, stop_at_goal=False goes on through the whole reachable space "
                     "(`exhausted` tells whether it got through, `goal_states` counts the goal states). goal: the goal states, as the common "
                     "keyword argument of the other searches (a callable needs threads=1). fingerprint: the result's fingerprint "
                     "hashes (id, canonical state) over the whole store (determinism checks; 0 when off). An ordered "
                     "layer_order needs the 'flat' or 'chunked' store ('auto' picks one of them) and, without a beam, "
                     "threads=1. "
                     "max_seconds, cancel, observer and progress_interval as for the other searches: the observer gets "
                     "on_start, on_expand, on_generate, on_pass(depth, layer stats) per layer, on_progress, "
                     "on_solution and on_end. With threads > 1 it follows the make_worker protocol: make_worker(k) "
                     "gives thread k its own observer for on_expand, on_generate (child None: ids are assigned when "
                     "the layer ends) and on_progress, called from that thread; without make_worker (or when it "
                     "returns None) the search runs on one thread. " MYMYR_SYMMETRY_DOC " ") +
         k_layer_doc)
            .c_str());

    m.def(
        "astar",
        [](TaskArg task, bool lazy, HeuristicArg heuristic, StrArg costs, StrArg evaluation, StrArg store,
           StrArg queue, StateArg start, bool reopen, bool lazy_requeue, bool preferred_operators,
           IntArg preferred_weight, IntArg standard_weight, bool witness_pruning, SymmetryArg symmetry_pruning, IntArg max_states,
           IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel, GoalArg goal,
           StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            return bf_call(lazy ? &search::astar_lazy : &search::astar_eager, task, heuristic, costs, evaluation, store,
                           queue, start, reopen, lazy_requeue, preferred_operators,
                           int_arg<u32>(preferred_weight, "preferred_weight"),
                           int_arg<u32>(standard_weight, "standard_weight", 1),
                           1000, witness_pruning, symmetry_pruning, max_states, max_expanded, max_depth, max_seconds,
                           cancel, goal,
                           blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "lazy"_a = false, "heuristic"_a = "max", "costs"_a = "unit", "evaluation"_a = "auto",
        "store"_a = "auto", "queue"_a = "auto", "start"_a = nb::none(), "reopen"_a = true, "lazy_requeue"_a = true,
        "preferred_operators"_a = true, "preferred_weight"_a = 0, "standard_weight"_a = 1, "witness_pruning"_a = false,
        "symmetry_pruning"_a = "off",
        "max_states"_a = nb::none(), "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(),
        "max_seconds"_a = nb::none(), "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(),
        "observer"_a = nb::none(), "progress_interval"_a = nb::none(),
        (std::string("A* (search/best_first.hpp), eager or lazy. heuristic: a kind ('blind', 'goal_count', 'max', "
                     "'add', 'ff', 'set_additive', 'h2', as for Heuristic), 'perfect' (h* from the task's state space, "
                     "generated first within max_states and max_seconds; ValueError when they stop it), a Heuristic "
                     "of the task, or a heuristic written in Python; costs: 'unit' or 'real' (the task's action "
                     "costs). ") +
         k_heuristic_doc + k_control_doc)
            .c_str());

    m.def(
        "astar_iw",
        [](TaskArg task, HeuristicArg heuristic, IntArg width, StrArg features, LandmarksArg landmarks,
           double weight, bool preserve_goal_atoms, bool preserve_landmark_atoms,
           bool allow_non_novel_root_goal, bool probe_novelty_before_heuristic, StrArg costs,
           StrArg evaluation, StrArg store, StateArg start, bool witness_pruning, bool canonical_order,
           SymmetryArg symmetry_pruning, IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds,
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
                options.symmetry_pruning = base.symmetry_pruning;
                return search::astar_iw(t, options);
            }, task, heuristic, costs, evaluation, store, nb::str("heap"), start, true, true, false, 0, 1,
                1000, witness_pruning, symmetry_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "heuristic"_a = "max", "width"_a = 1, "features"_a = "classical",
        "landmarks"_a = nb::none(), "weight"_a = 1.0, "preserve_goal_atoms"_a = true,
        "preserve_landmark_atoms"_a = true, "allow_non_novel_root_goal"_a = true,
        "probe_novelty_before_heuristic"_a = true, "costs"_a = "unit", "evaluation"_a = "auto",
        "store"_a = "auto", "start"_a = nb::none(), "witness_pruning"_a = false, "canonical_order"_a = true,
        "symmetry_pruning"_a = "off",
        MYMYR_CONTROL_ARGS,
        (std::string("Weighted A* with minimum-g novelty pruning (search/astar_iw.hpp). Priority is "
                     "(g + weight*h, h, g, state id). Unit-cost actions and no numeric fluents are required. "
                     "features: 'classical' (width 1..5), 'abstracted' or 'base_abstracted' (width 1..3). "
                     "landmarks: a fact landmark graph, 'approximate', 'lifted', or None. Abstracted modes "
                     "preserve positive goal atoms and landmark atoms by default. Non-novel root successors "
                     "are admitted with allow_non_novel_root_goal; only goals bypass the pop novelty test. "
                     "The read-only probe skips heuristic work without changing the search. Invalid widths "
                     "and weights return FAILED with a message. Novelty statistics are in result.novelty. "
                     "heuristic and costs: as for astar; novelty pruning does not guarantee optimality. ") +
         k_heuristic_doc + k_control_doc).c_str());

    m.def(
        "gbfs",
        [](TaskArg task, bool lazy, HeuristicArg heuristic, StrArg costs, StrArg evaluation, StrArg store,
           StrArg queue, StateArg start, bool preferred_operators, IntArg preferred_weight, IntArg standard_weight,
           bool witness_pruning, SymmetryArg symmetry_pruning, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            return bf_call(lazy ? &search::gbfs_lazy : &search::gbfs_eager, task, heuristic, costs, evaluation, store,
                           queue, start, true, true, preferred_operators,
                           int_arg<u32>(preferred_weight, "preferred_weight"),
                           int_arg<u32>(standard_weight, "standard_weight", 1), 1000,
                           witness_pruning, symmetry_pruning, max_states, max_expanded, max_depth, max_seconds, cancel, goal,
                           blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "lazy"_a = false, "heuristic"_a = "ff", "costs"_a = "unit", "evaluation"_a = "auto",
        "store"_a = "auto", "queue"_a = "auto", "start"_a = nb::none(), "preferred_operators"_a = true,
        "preferred_weight"_a = 0, "standard_weight"_a = 1, "witness_pruning"_a = false, "symmetry_pruning"_a = "off",
        "max_states"_a = nb::none(),
        "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        (std::string("Greedy best-first search (search/best_first.hpp), eager or lazy (preferred and standard lists "
                     "alternating, mimir's 64:1 by default). heuristic: as for astar. ") +
         k_heuristic_doc + k_control_doc)
            .c_str());

    m.def(
        "beam",
        [](TaskArg task, IntArg width, HeuristicArg heuristic, StrArg costs, StrArg evaluation, StrArg store,
           StateArg start, bool witness_pruning, SymmetryArg symmetry_pruning, IntArg max_states, IntArg max_expanded,
           IntArg max_depth,
           FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            return bf_call(&search::beam, task, heuristic, costs, evaluation, store, nb::str("auto"), start, true, true,
                           true, 0, 1, int_arg<u32>(width, "width", 1), witness_pruning, symmetry_pruning, max_states,
                           max_expanded, max_depth,
                           max_seconds, cancel,
                           goal, blocked_states, observer, progress_interval);
        },
        "task"_a, nb::kw_only(), "width"_a = 1000, "heuristic"_a = "ff", "costs"_a = "unit", "evaluation"_a = "auto",
        "store"_a = "auto", "start"_a = nb::none(), "witness_pruning"_a = false, "symmetry_pruning"_a = "off",
        "max_states"_a = nb::none(),
        "max_expanded"_a = nb::none(), "max_depth"_a = nb::none(), "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        (std::string("Heuristic beam search (search/best_first.hpp): the best `width` new successors of each layer by "
                     "(h, g). Incomplete: EXHAUSTED means the beam ran dry. heuristic: as for astar. ") +
         k_heuristic_doc + k_control_doc)
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
        "The plan as IPC plan text (search/plan_file.hpp): one '(name o1 ... ok)' line per action with the PDDL action's "
        "own parameters (mimir's plan format), then '; cost = N (unit cost)' or '; cost = N (general cost)'. N is the "
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
        "The actions of IPC plan text, replayed from `start` (default: the initial state) (search/plan_file.hpp). "
        "Blank lines and ';' comments are skipped and names compared without regard to case. A line names the PDDL "
        "action's own parameters or all of the schema's parameters; among the applicable actions that match, the first "
        "in canonical order is taken. Raises ValueError, naming the line, when no applicable action matches.");

    // the IW family variants --------------------------------------------------------------------------------------
    m.def(
        "liw",
        [](TaskArg task, IntArg max_arity, LandmarksArg landmarks, bool disjunctive, bool all_private,
           FluentAtomsArg unshared_atoms, IntArg max_dense_bytes, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, WidthZeroArg width_zero,
           bool witness_pruning, bool canonical_order,
           SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_expanded, IntArg max_depth,
           FloatArg max_seconds,
           CancelArg cancel, GoalArg goal, StatesArg blocked_states, ObserverArg observer,
           IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::LiwOptions opts;
            static_cast<search::IwOptions&>(opts) =
                iw_options(cs, o, int_arg<u32>(max_arity, "max_arity"), width_zero, true, witness_pruning, canonical_order,
                           symmetry_pruning, start);
            opts.layers = parse_layers(layer_order, seed, max_next_layer_states, prefer_more_satisfied_goals, beam_width,
                                       beam_novelty, randomize_ties, beam_chunk);
            opts.threads = threads_arg(threads);
            opts.landmarks = parse_landmarks(o, landmarks, disjunctive, all_private, unshared_atoms, max_dense_bytes);
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::liw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "max_arity"_a = 2, MYMYR_LANDMARK_ARGS, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "width_zero"_a = "expand_depth_one", "witness_pruning"_a = false, "canonical_order"_a = true,
        "symmetry_pruning"_a = "off", "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        (std::string("LIW(k) (search/liw.hpp; mimir's iw with landmark_novelty_graph): the width-0 pass, then "
                     "LIW(1..max_arity) over (landmark coordinate, tuple of size <= k) novelty, each pass from the start "
                     "with a fresh table, until one solves the task (an exhausted ladder is EXHAUSTED, mimir's "
                     "FAILED). No optimized IW(1). The result is an IwResult with the per-pass statistics.") +
         k_landmark_doc + k_layer_doc + " " + k_control_doc + k_iw_family_doc)
            .c_str());

    m.def(
        "abstracted_iw",
        [](TaskArg task, IntArg width, bool base_abstracted, bool preserve_goal_atoms, bool keep_depth_one_novel,
           LandmarksArg landmarks, bool disjunctive, bool all_private, FluentAtomsArg unshared_atoms,
           IntArg max_dense_bytes, bool preserve_landmark_atoms, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, bool witness_pruning,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::AbstractedIwOptions opts =
                aiw_options(cs, o, keep_depth_one_novel, landmarks, disjunctive, all_private, unshared_atoms,
                            max_dense_bytes, preserve_landmark_atoms, layer_order, seed, max_next_layer_states,
                            prefer_more_satisfied_goals, beam_width, beam_novelty, randomize_ties,
                            beam_chunk, threads, witness_pruning, canonical_order, symmetry_pruning, start);
            opts.width = int_arg<u32>(width, "width");
            opts.base_abstracted = base_abstracted;
            opts.preserve_goal_atoms = preserve_goal_atoms;
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::abstracted_iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "width"_a = 1, "base_abstracted"_a = false, "preserve_goal_atoms"_a = true,
        "keep_depth_one_novel"_a = false, MYMYR_LANDMARK_ARGS, "preserve_landmark_atoms"_a = true, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "witness_pruning"_a = false, "canonical_order"_a = true, "symmetry_pruning"_a = "off", "start"_a = nb::none(),
        MYMYR_CONTROL_ARGS,
        (std::string("Abstracted IW(width) (search/aiw.hpp; mimir's abstracted_iw): one breadth-first pass whose "
                     "novelty features are abstracted atoms, one per object position with the other objects' type "
                     "signature (base_abstracted: BAIW, no signatures); preserve_goal_atoms keeps positive goal atoms "
                     "whole; keep_depth_one_novel expands the non-novel successors of the start. width 1..3. With "
                     "landmarks: abstracted LIW (one table per landmark rank; preserve_landmark_atoms keeps landmark "
                     "atoms whole). The IwResult holds the one pass (arity = width); an exhausted pass is EXHAUSTED.") +
         k_landmark_doc + k_layer_doc + " " + k_control_doc + k_iw_family_doc)
            .c_str());

    m.def(
        "projective_iw",
        [](TaskArg task, bool typed_projection, bool keep_goal_nonunary_atoms, bool keep_depth_one_novel,
           LandmarksArg landmarks, bool disjunctive, bool all_private, FluentAtomsArg unshared_atoms,
           IntArg max_dense_bytes, bool preserve_landmark_atoms, LayerArg layer_order, IntArg seed,
           IntArg max_next_layer_states, bool prefer_more_satisfied_goals, IntArg beam_width,
           BeamNoveltyArg beam_novelty, bool randomize_ties, IntArg beam_chunk, IntArg threads, bool witness_pruning,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start,
           IntArg max_states, IntArg max_expanded, IntArg max_depth, FloatArg max_seconds, CancelArg cancel,
           GoalArg goal, StatesArg blocked_states, ObserverArg observer, IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::ProjectiveIwOptions opts;
            static_cast<search::AbstractedIwOptions&>(opts) =
                aiw_options(cs, o, keep_depth_one_novel, landmarks, disjunctive, all_private, unshared_atoms,
                            max_dense_bytes, preserve_landmark_atoms, layer_order, seed, max_next_layer_states,
                            prefer_more_satisfied_goals, beam_width, beam_novelty, randomize_ties,
                            beam_chunk, threads, witness_pruning, canonical_order, symmetry_pruning, start);
            opts.typed_projection = typed_projection;
            opts.keep_goal_nonunary_atoms = keep_goal_nonunary_atoms;
            const Task& t = *o.core->task;
            search::IwResult r = run_detached(cs, [&] { return search::projective_iw(t, opts); });
            return PyIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "typed_projection"_a = false, "keep_goal_nonunary_atoms"_a = false,
        "keep_depth_one_novel"_a = false, MYMYR_LANDMARK_ARGS, "preserve_landmark_atoms"_a = true, MYMYR_LAYER_ARGS, MYMYR_THREADS_ARG,
        "witness_pruning"_a = false, "canonical_order"_a = true, "symmetry_pruning"_a = "off", "start"_a = nb::none(),
        MYMYR_CONTROL_ARGS,
        (std::string("Mimir's projective_iw alias: abstracted IW(1) with base_abstracted = not typed_projection "
                     "and preserve_goal_atoms = keep_goal_nonunary_atoms. See abstracted_iw.") +
         k_landmark_doc + k_layer_doc + " " + k_control_doc + k_iw_family_doc)
            .c_str());

    m.def(
        "rollout_iw",
        [](TaskArg task, OrderingArg ordering, IntArg seed, IntArg max_rollouts, IntArg incumbent_bound,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_depth,
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
            opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
            if (!start.is_none())
                opts.start = state_for(o, start, "start");
            const Task& t = *o.core->task;
            search::RolloutIwResult r = run_detached(cs, [&] { return search::rollout_iw(t, opts); });
            return PyRolloutIwResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "ordering"_a = search::ActionOrdering::InOrder, "seed"_a = 0,
        "max_rollouts"_a = nb::none(), "incumbent_bound"_a = nb::none(), "canonical_order"_a = true,
        "symmetry_pruning"_a = "off", "start"_a = nb::none(), "max_states"_a = nb::none(), "max_depth"_a = nb::none(),
        "max_seconds"_a = nb::none(),
        "cancel"_a = nb::none(), "goal"_a = nb::none(), "blocked_states"_a = nb::none(), "observer"_a = nb::none(),
        "progress_interval"_a = nb::none(),
        (std::string("Rollout IW(1) (search/rollout_iw.hpp; Bandres et al. 2018, mimir's find_solution_rollout_iw): "
                     "rollouts from the root over a tree of action sequences, pruned by the minimum depth of each "
                     "fluent atom, until a goal is found (the goal is tested on generation), the root is solved "
                     "(EXHAUSTED; root_solved) or a budget stops it. ordering: an ActionOrdering or its name "
                     "('in_order', 'randomized', 'direct_goal_achiever_first' / 'dgaf', 'goal_regression_relevance' / "
                     "'regression', 'mixed_regression_random' / 'mixed'); seed: of the randomized orderings (a "
                     "SplitMix64; the same seed gives the same run on every platform); max_rollouts (FAILED when hit); "
                     "incumbent_bound: a plan of this length is known elsewhere. max_states counts generated tree "
                     "nodes; there is no max_expanded. ") +
         k_control_doc + k_iw_family_doc)
            .c_str());

    m.def(
        "find_rollouts_parallel",
        [](TaskArg task, SeedsArg seeds, IntArg max_arity, IntArg num_threads, IntArg max_next_layer_states,
           bool report_landing_states, bool report_co_occurrence, WidthZeroArg width_zero, bool optimize_iw1,
           bool witness_pruning, bool canonical_order, SymmetryArg symmetry_pruning, StateArg start, IntArg max_states,
           IntArg max_expanded,
           IntArg max_depth, FloatArg max_seconds, CancelArg cancel, GoalArg goal, StatesArg blocked_states,
           ObserverArg observer, IntArg progress_interval) {
            const Owner o = owner_of(task);
            ControlScope cs;
            fill_control(cs, o, max_states, max_expanded, max_depth, max_seconds, cancel, goal, blocked_states,
                         observer, progress_interval);
            search::ParallelRolloutOptions opts;
            opts.iw = iw_options(cs, o, int_arg<u32>(max_arity, "max_arity"), width_zero, optimize_iw1, witness_pruning,
                                 canonical_order, symmetry_pruning, start);
            opts.seeds = int_list_arg<u64>(seeds, "seeds");
            opts.num_threads = threads_arg(num_threads, "num_threads");
            if (auto v = opt_int_arg<u32>(max_next_layer_states, "max_next_layer_states", 0))
                opts.max_next_layer_states = *v;
            opts.report_landing_states = report_landing_states;
            opts.report_co_occurrence = report_co_occurrence;
            const Task& t = *o.core->task;
            search::ParallelRolloutsResult r = run_detached(cs, [&] { return search::find_rollouts_parallel(t, opts); });
            return PyParallelRollouts{std::make_shared<const search::ParallelRolloutsResult>(std::move(r)),
                                      opts.seeds, o};
        },
        "task"_a, "seeds"_a, nb::kw_only(), "max_arity"_a = 2, "num_threads"_a = 0,
        "max_next_layer_states"_a = nb::none(), "report_landing_states"_a = false, "report_co_occurrence"_a = false,
        "width_zero"_a = "expand_depth_one", "optimize_iw1"_a = true, "witness_pruning"_a = false,
        "canonical_order"_a = true, "symmetry_pruning"_a = "off", "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        (std::string("Parallel IW rollouts (search/parallel_rollouts.hpp; mimir's find_rollouts_iw_parallel / "
                     "iw_parallel): one IW(max_arity) ladder per seed with a randomized layer order seeded by it, on "
                     "num_threads threads (0: all cores, capped at len(seeds)) over the one task. For a seed a "
                     "rollout's result is the same at every thread count. Per rollout: the ladder, the reached atoms, "
                     "the number of distinct states created and, when reported, the landing states (per fluent atom "
                     "the first created state holding it) and co-occurrence (per fluent atom the atoms of the created "
                     "states holding it). max_seconds spans the whole batch, the other budgets apply per rollout.") +
         " " + k_control_doc + k_iw_family_doc + k_parallel_doc + " A worker is a rollout (an index into seeds).")
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
        "Mimir's intersect_co_occurrence: per atom of the first rollout, the intersection of its co-occurrence rows "
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
        [](TaskArg task, IntArg num_rollout_workers, IntArg num_threads, IntArg base_seed, OrderingsArg rollout_orderings,
           bool canonical_order, SymmetryArg symmetry_pruning, StateArg start, IntArg max_states, IntArg max_expanded,
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
            opts.num_threads = threads_arg(num_threads, "num_threads");
            opts.base_seed = int_arg<u64>(base_seed, "base_seed");
            opts.rollout_orderings = parse_orderings(rollout_orderings);
            opts.canonical_order = canonical_order;
            opts.symmetry_pruning = parse_symmetry_pruning(symmetry_pruning);
            if (!start.is_none())
                opts.start = state_for(o, start, "start");
            const Task& t = *o.core->task;
            search::PortfolioResult r = run_detached(cs, [&] { return search::atomic_goal_portfolio(t, opts); });
            return PyPortfolioResult{std::move(r), o};
        },
        "task"_a, nb::kw_only(), "num_rollout_workers"_a = 4, "num_threads"_a = 0, "base_seed"_a = 0,
        "rollout_orderings"_a = nb::none(), "canonical_order"_a = true, "symmetry_pruning"_a = "off",
        "start"_a = nb::none(), MYMYR_CONTROL_ARGS,
        (std::string("The atomic-goal portfolio (search/portfolio.hpp; mimir's "
                     "find_solution_atomic_goal_iw_portfolio): one IW(1) certifier (worker 0) and num_rollout_workers "
                     "goal-guided Rollout IW workers (1..K) under one coordination: the best plan's length prunes the "
                     "rollouts, and the certifier's completed depths certify a plan as shortest in the width-1 space. "
                     "rollout_orderings: cycled over the workers, each an ordering or (ordering, seed); None: "
                     "mimir's cycle DIRECT_GOAL_ACHIEVER_FIRST, MIXED_REGRESSION_RANDOM, RANDOMIZED, "
                     "GOAL_REGRESSION_RELEVANCE with seeds base_seed + k - 1. num_threads 1 runs the workers one after "
                     "the other on the calling thread (reproducible); more threads (0: all cores) run them side by side "
                     "(the result then depends on timing, as in mimir). goal: the atomic goal (e.g. "
                     "goal=[['(on a b)']]); max_expanded caps the expansions of all workers together, max_states "
                     "applies per worker, max_seconds to the call.") +
         " " + k_control_doc + k_iw_family_doc + k_parallel_doc + " A worker is 0 (the certifier) or 1..K.")
            .c_str());
}
}  // namespace mymyr::python
