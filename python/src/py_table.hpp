#pragma once
// mymyr.rl.TaskTable: many instances of one domain (rl::TaskTable) with their Python task
// cores; mymyr.rl.TaskSuite: instances of several domains, one TaskTable per
// domain (rl::TaskSuite). The argument of the RL entry points is a suite (SuiteRef): a TaskSuite, a TaskTable (the
// suite of its domain) or a Task / TaskHandle (its table of one, cached on the task core), so that every entry point
// takes one path. One-domain entry points (the device searches and state spaces) take a table (TableRef).

#include "py_task.hpp"
#include "typing.hpp"

#include "mymyr/rl/env.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"

#include <nanobind/nanobind.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace mymyr::python
{
struct PyTable
{
    rl::TaskTablePtr table;
    rl::TaskSuitePtr suite;         // rl::TaskSuite::of(table): what the RL entry points run
    std::vector<CorePtr> cores;     // instance i's task core
    std::vector<nb::object> tasks;  // instance i's Task (or TaskHandle) object: the owner of its labels and states
    std::atomic<u32> branching{8};  // expected successors per state (sizing rl.expand outputs)
    std::mutex cuda_mutex;          // device state (cuda_bindings.cpp), as PyTaskCore's
    std::shared_ptr<void> cuda;
    std::mutex meta_mutex;
    std::shared_ptr<const rl::ArrayBundle> meta;  // table_atom_metadata, rebuilt when lazy slots grew
    u64 meta_key = ~u64{0};
};

struct PySuite
{
    rl::TaskSuitePtr suite;
    std::vector<nb::object> tables;  // domain d's mymyr.rl.TaskTable (owns the instances' task cores and objects)
    std::atomic<u32> branching{8};
    std::mutex cuda_mutex;
    std::shared_ptr<void> cuda;
    std::mutex meta_mutex;
    std::shared_ptr<const rl::ArrayBundle> meta;  // suite_atom_metadata, rebuilt when lazy slots grew
    u64 meta_key = ~u64{0};

    [[nodiscard]] const PyTable& table(u32 domain) const { return *nb::inst_ptr<PyTable>(tables.at(domain)); }
};

/// A table argument, resolved (one domain).
struct TableRef
{
    rl::TaskTablePtr table;
    PyTable* py = nullptr;       // the mymyr.rl.TaskTable (null: a task's table of one)
    PyTaskCore* core = nullptr;  // the task core (a task's table of one)
    nb::object obj;              // the TaskTable, Task or TaskHandle object (keeps the rest alive)

    /// The owner of instance i's labels and states (its task core and Task object).
    [[nodiscard]] Owner owner(u32 instance) const
    {
        if (py)
            return {py->cores.at(instance).get(), py->tasks.at(instance)};
        return {core, obj};
    }
    [[nodiscard]] PyTaskCore& core_of(u32 instance) const { return py ? *py->cores.at(instance) : *core; }
    [[nodiscard]] std::atomic<u32>& branching() const { return py ? py->branching : core->branching; }
    [[nodiscard]] std::mutex& cuda_mutex() const { return py ? py->cuda_mutex : core->cuda_mutex; }
    [[nodiscard]] std::shared_ptr<void>& cuda() const { return py ? py->cuda : core->cuda; }
};

/// A suite argument, resolved: global instance ids (rl::TaskSuite).
struct SuiteRef
{
    rl::TaskSuitePtr suite;
    PySuite* ps = nullptr;       // the mymyr.rl.TaskSuite
    PyTable* py = nullptr;       // the mymyr.rl.TaskTable (the suite of its domain)
    PyTaskCore* core = nullptr;  // the task core (a task's table of one)
    nb::object obj;              // the TaskSuite, TaskTable, Task or TaskHandle object (keeps the rest alive)

    /// The owner of global instance g's labels and states (its task core and Task object).
    [[nodiscard]] Owner owner(u32 g) const
    {
        if (ps)
        {
            const rl::TaskSuite::Ref r = suite->ref(g);
            const PyTable& t = ps->table(r.domain);
            return {t.cores.at(r.local).get(), t.tasks.at(r.local)};
        }
        if (py)
            return {py->cores.at(g).get(), py->tasks.at(g)};
        return {core, obj};
    }
    [[nodiscard]] PyTaskCore& core_of(u32 g) const { return *owner(g).core; }
    [[nodiscard]] std::atomic<u32>& branching() const
    {
        return ps ? ps->branching : py ? py->branching : core->branching;
    }
    [[nodiscard]] std::mutex& cuda_mutex() const { return ps ? ps->cuda_mutex : py ? py->cuda_mutex : core->cuda_mutex; }
    [[nodiscard]] std::shared_ptr<void>& cuda() const { return ps ? ps->cuda : py ? py->cuda : core->cuda; }
};

/// A TaskTable, Task or TaskHandle. Raises TypeError otherwise.
[[nodiscard]] TableRef table_of(nb::handle obj);
/// A TaskSuite, TaskTable, Task or TaskHandle. Raises TypeError otherwise.
[[nodiscard]] SuiteRef suite_of(nb::handle obj);

using TableArg = Arg<std::variant<PyTable, PyTask, PyHandle>>;
using SuiteArg = Arg<std::variant<PySuite, PyTable, PyTask, PyHandle>>;
/// Task ids of a batch: an int32 array [N] (host arrays may be int64; device arrays must be int32), or a sequence of
/// ints (host).
using TaskIdsArg = Arg<std::variant<ann::ArrayLike, nb::typed<nb::sequence, int>>>;

/// The current atom words of a table's (suite's) rows: the widest instance's Task::words() (lazy slots grow it up to
/// words()).
[[nodiscard]] u32 current_words(const rl::TaskTable& table) noexcept;
[[nodiscard]] u32 current_words(const rl::TaskSuite& suite) noexcept;

/// The host task ids of a batch of `rows` rows: an int32 array (read in place), an int64 array or a sequence of ints
/// (converted into `copy`, range-checked against `instances`), or None (null, which the entry points accept for a table
/// of one only). `keep` holds what the pointer points into; `noun` names the instances' owner in messages.
[[nodiscard]] const i32* import_task_ids(nb::handle obj, u64 rows, u32 instances, std::vector<i32>& copy,
                                         std::shared_ptr<void>& keep, const char* name = "task_ids",
                                         const char* noun = "table");

/// The dead-end mode of an environment configuration: "no_successors" or "none".
[[nodiscard]] rl::DeadEnd parse_dead_end(const std::string& s);
[[nodiscard]] const char* dead_end_name(rl::DeadEnd d) noexcept;
}  // namespace mymyr::python
