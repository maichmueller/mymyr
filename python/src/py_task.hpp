#pragma once
// Python-side objects of the task API:
//   - PyTaskCore: the shared state behind one mymyr.Task: the immutable C++ Task, the normalized TaskData (formalism
//     views), the source for pickling, and caches (names, array snapshots). Shared by the Task object and every handle.
//   - Task (PyTask) and TaskHandle (PyHandle, from task.local()): distinct Python objects over one core, so threads
//     do not contend on one object's reference count.
//   - State (PyState): a value (the C++ State) plus its owner (the Task or handle it came from) for names and
//     convenience methods. Equality and hashing use the task uid plus the content, never the owner.
//   - Action (PyAction): the label (schema, binding) plus its owner, for printing.
//   - The formulas (atoms, literals, conditions) are in py_formula.hpp.

#include "formalism_task.hpp"
#include "typing.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/task.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mymyr::python
{
namespace nb = nanobind;

struct NameIndex
{
    std::unordered_map<std::string, u32> objects, predicates, schemas;
};

class PyTaskCore
{
public:
    PyTaskCore(TaskPtr task, std::shared_ptr<const formalism::TaskData> data, std::shared_ptr<const TaskSource> source);

    TaskPtr task;
    std::shared_ptr<const formalism::TaskData> data;
    std::shared_ptr<const TaskSource> source;
    u32 label_width = 0;
    /// Expected successors per state (rounded up, with headroom), for sizing rl.expand outputs (see expand()).
    std::atomic<u32> branching{8};

    [[nodiscard]] const NameIndex& names();
    /// Cached snapshots (rebuilt when lazy slots were added since).
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> atom_metadata();
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> device_arrays(u32 version);
    [[nodiscard]] std::shared_ptr<const rl::GoalMasks> goal_masks();
    /// The task's table of one (rl::TaskTable::single; the RL entry points take tables), made on first use.
    [[nodiscard]] const rl::TaskTablePtr& table()
    {
        std::call_once(m_table_once, [this] { m_table = rl::TaskTable::single(task); });
        return m_table;
    }
    /// The suite of that table (the RL entry points run suites; rl::TaskSuite::of), made on first use.
    [[nodiscard]] const rl::TaskSuitePtr& suite()
    {
        std::call_once(m_suite_once, [this] { m_suite = rl::TaskSuite::of(table()); });
        return m_suite;
    }

    /// The task's action costs (heuristics::ActionCosts), made on first use. Throws std::invalid_argument (on every
    /// call) for a metric mymyr refuses. The error is captured inside call_once and rethrown outside it, so that no
    /// exception unwinds through the C frames of pthread_once.
    [[nodiscard]] const heuristics::ActionCosts& costs()
    {
        std::call_once(m_costs_once,
                       [this]
                       {
                           try
                           {
                               m_costs = std::make_unique<heuristics::ActionCosts>(*task);
                           }
                           catch (...)
                           {
                               m_costs_error = std::current_exception();
                           }
                       });
        if (m_costs_error)
            std::rethrow_exception(m_costs_error);
        return *m_costs;
    }
    /// Numeric slot of a name "(function o1 ... ok)" (Task::numeric_name), or ~0 if no slot has it.
    [[nodiscard]] u32 numeric_slot(const std::string& name)
    {
        std::call_once(m_slots_once,
                       [this]
                       {
                           for (u32 i = 0; i < task->numeric_slots(); ++i)
                               m_slots.emplace(task->numeric_name(i), i);
                       });
        const auto it = m_slots.find(name);
        return it == m_slots.end() ? ~u32{0} : it->second;
    }

    /// State of the CUDA backend for this task (cuda_bindings.cpp: default contexts, device expanders), created on
    /// first use under cuda_mutex; released with the task.
    std::mutex cuda_mutex;
    std::shared_ptr<void> cuda;

private:
    std::once_flag m_names_once, m_table_once, m_suite_once, m_costs_once, m_slots_once;
    std::unique_ptr<heuristics::ActionCosts> m_costs;
    std::exception_ptr m_costs_error;
    std::unordered_map<std::string, u32> m_slots;
    NameIndex m_names;
    rl::TaskTablePtr m_table;
    rl::TaskSuitePtr m_suite;
    std::mutex m_mutex;
    std::shared_ptr<const rl::ArrayBundle> m_meta, m_device;
    u64 m_meta_key = ~u64{0}, m_device_key = ~u64{0};
    std::shared_ptr<const rl::GoalMasks> m_goal;
};
using CorePtr = std::shared_ptr<PyTaskCore>;

struct PyTask
{
    CorePtr core;
};

struct PyHandle
{
    CorePtr core;
    nb::object task;  // the mymyr.Task it came from
};

struct PyState
{
    State s;
    nb::object owner;  // a Task or TaskHandle object: keeps `core` alive
    PyTaskCore* core = nullptr;
};

struct PyAction
{
    u32 schema = 0;
    std::vector<u32> binding;
    nb::object owner;
    PyTaskCore* core = nullptr;
    [[nodiscard]] ActionLabel label() const
    {
        return {SchemaId{schema}, {reinterpret_cast<const ObjectId*>(binding.data()), binding.size()}};
    }
};

// ------------------------------------------------------------------------------------------------ argument types
// What the task API accepts (typing.hpp: rendered in the stubs, checked by the bindings).

/// An object name or index.
using ObjectKey = std::variant<std::string, int>;
using ObjectKeys = nb::typed<nb::iterable, ObjectKey>;
/// One state: a State, or its words (one row; numeric tasks: the atom words, then the numeric words).
using StateLike = Arg<std::variant<PyState, ann::ArrayLike>>;
/// A batch of states: a State, a sequence of States, or a word array [N, W] (numeric tasks: [N, W + NN]).
using StatesLike = Arg<std::variant<PyState, nb::typed<nb::sequence, PyState>, ann::ArrayLike>>;
/// An action: an Action, '(stack a b)', or (schema, objects) with a schema name or index.
using ActionLike = Arg<std::variant<PyAction, std::string, nb::typed<nb::tuple, ObjectKey, ObjectKeys>>>;
using FrameworkArg = Arg<ann::Framework>;

/// The core and owner object of a Task, TaskHandle, State or Action. Throws TypeError otherwise.
struct Owner
{
    PyTaskCore* core = nullptr;
    nb::object obj;  // a Task or TaskHandle
};
[[nodiscard]] Owner owner_of(nb::handle obj);
/// The mymyr.Task object behind an owner (a Task or TaskHandle object).
[[nodiscard]] Arg<PyTask> task_object(nb::handle owner);

[[nodiscard]] Arg<PyState> make_state(const Owner& o, State&& s);
/// An Action (label) owned by `o`; `binding` holds `arity` object ids.
[[nodiscard]] Arg<PyAction> make_label(const Owner& o, u32 schema, const i32* binding, u32 arity);
[[nodiscard]] bool is_state(nb::handle h);
[[nodiscard]] const PyState& state_of(nb::handle h);

struct StateBatch;  // arrays.hpp
/// A batch of states of `task`: import_states, except that the rows of a numeric task are [W + NN] (the atom
/// words, then task.numeric_words() numeric words), for arrays and for packed States alike.
[[nodiscard]] StateBatch import_task_states(nb::handle obj, const Task& task);
/// import_task_states for rows of at least `words` atom words (packed States) and NN numeric words (a task table's
/// rows: words = TaskTable::words(), NN = TaskTable::numeric_words(); a State may carry fewer numeric words).
[[nodiscard]] StateBatch import_rows(nb::handle obj, u32 words, u32 NN);

/// The TaskOptions of the Python keyword arguments (MYMYR_TASK_OPTION_ARGS); raises ValueError for unknown names.
[[nodiscard]] TaskOptions make_options(std::string_view atoms, std::string_view matching, u32 fc_free_params,
                                       u32 frozen_max_words, u32 pilot_expansions);
/// The keyword arguments of make_options (after the positional ones), with their defaults.
#define MYMYR_TASK_OPTION_ARGS                                                                                         \
    nb::kw_only(), "atoms"_a = "auto", "matching"_a = "auto", "fc_free_params"_a = 4, "frozen_max_words"_a = 8,         \
        "pilot_expansions"_a = 1024

void bind_task(nb::module_& m);
void bind_rl(nb::module_& m);
}  // namespace mymyr::python
