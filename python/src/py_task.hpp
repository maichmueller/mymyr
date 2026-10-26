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

#include "mymyr/core/once.hpp"
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
#include <memory>
#include <mutex>
#include <span>
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
    nb::ft_mutex frozen_mutex;
    struct FrozenCore
    {
        std::shared_ptr<PyTaskCore> core;
        nb::object wrapper;  // weak reference: the twin's wrapper owns its source
    };
    std::unordered_map<u64, FrozenCore> frozen_cores;

    [[nodiscard]] const NameIndex& names();
    /// Cached snapshots (rebuilt when lazy slots were added since).
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> atom_metadata();
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> device_arrays(u32 version);
    [[nodiscard]] std::shared_ptr<const rl::GoalMasks> goal_masks();
    /// The task's table of one (rl::TaskTable::single; the RL entry points take tables), made on first use.
    [[nodiscard]] const rl::TaskTablePtr& table()
    {
        m_table_once.call([this] { m_table = rl::TaskTable::single(task); });
        return m_table;
    }
    /// The suite of that table (the RL entry points run suites; rl::TaskSuite::of), made on first use.
    [[nodiscard]] const rl::TaskSuitePtr& suite()
    {
        m_suite_once.call([this] { m_suite = rl::TaskSuite::of(table()); });
        return m_suite;
    }

    /// The task's action costs (heuristics::ActionCosts), made on first use. Throws std::invalid_argument (on every
    /// call: a failed initialization runs again) for a metric mymyr refuses.
    [[nodiscard]] const heuristics::ActionCosts& costs()
    {
        m_costs_once.call([this] { m_costs = std::make_unique<heuristics::ActionCosts>(*task); });
        return *m_costs;
    }
    /// Numeric slot of a name "(function o1 ... ok)" (Task::numeric_name), or ~0 if no slot has it.
    [[nodiscard]] u32 numeric_slot(const std::string& name)
    {
        m_slots_once.call([this] {
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
    // lazy initializations that may throw (Once: not std::call_once, see mymyr/core/once.hpp)
    Once m_names_once, m_table_once, m_suite_once, m_costs_once, m_slots_once;
    std::unique_ptr<heuristics::ActionCosts> m_costs;
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
    nb::object frozen_source = {};
};

/// Cached frozen twin of a Task or TaskHandle, preserving its source for serialization.
[[nodiscard]] nb::object frozen_task(nb::handle source, std::string_view atoms = "auto");

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
/// words, then task.numeric_words() numeric words), for arrays and for packed States alike. Checked with
/// check_task_rows.
[[nodiscard]] StateBatch import_task_states(nb::handle obj, const Task& task);
/// ValueError unless the batch holds states of `task`: its States belong to the task; array rows are at least
/// task.words() wide and set no atom slot the task does not have (the words past task.words() are padding, not read);
/// a task with lazy atom slots takes no array rows.
void check_task_rows(const StateBatch& b, const Task& task);
/// import_task_states for rows of at least `words` atom words (packed States) and NN numeric words (a task table's
/// rows: NN = TaskTable::numeric_words(); a State may carry fewer numeric words).
[[nodiscard]] StateBatch import_rows(nb::handle obj, u32 words, u32 NN);

/// The lazy atom slots of a task: u64 [2 + F + D], the numbers F and D of assigned fluent and derived slots, then
/// their canonical ids in slot order; empty for frozen slots.
[[nodiscard]] std::vector<u64> lazy_slots(const Task& task);
/// Gives `task` (lazy slots) the numbering `slots` (lazy_slots of a task with the same content): the slots it has
/// assigned must agree with it, and it assigns the others in its order. Raises ValueError otherwise, and for a task
/// with frozen slots or a malformed table; `what` names the object being restored in the message.
void adopt_lazy_slots(const Task& task, std::span<const u64> slots, const char* what);

/// The TaskOptions of the Python keyword arguments (MYMYR_TASK_OPTION_ARGS); raises ValueError for unknown names and
/// values out of range.
[[nodiscard]] TaskOptions make_options(std::string_view atoms, std::string_view matching, IntArg fc_free_params,
                                       std::string_view numeric_storage, FloatArg numeric_quantum, bool numeric_tolerant,
                                       IntArg reachability_budget);
/// The task options as binding parameters, their names as arguments of make_options, and their keyword arguments
/// (after the positional ones) with their defaults.
#define MYMYR_TASK_OPTION_PARAMS                                                                                       \
    std::string_view atoms, std::string_view matching, IntArg fc_free_params, std::string_view numeric_storage,         \
        FloatArg numeric_quantum, bool numeric_tolerant, IntArg reachability_budget
#define MYMYR_TASK_OPTION_VALUES                                                                                       \
    atoms, matching, fc_free_params, numeric_storage, numeric_quantum, numeric_tolerant, reachability_budget
#define MYMYR_TASK_OPTION_ARGS                                                                                         \
    nb::kw_only(), "atoms"_a = "auto", "matching"_a = "auto", "fc_free_params"_a = 4, "numeric_storage"_a = "auto",     \
        "numeric_quantum"_a = 0.0, "numeric_tolerant"_a = false,                                                        \
        "reachability_budget"_a.none() = ::mymyr::TaskOptions::k_reachability_budget

void bind_task(nb::module_& m);
void bind_rl(nb::module_& m);
}  // namespace mymyr::python
