#pragma once
// TaskTable: many instances of one domain, the unit every RL entry point works on. Batches over a table carry a
// per-row instance index (`task_ids`, int32 [N]); a single Task
// is a table of one instance (TaskTable::single), so there is one code path.
//
//   auto table = rl::TaskTable::create({task_a, task_b, task_c});   // std::shared_ptr<const TaskTable>, immutable
//   rl::expand(*table, batch, task_ids, out);                       // rows of any instance, in one call
//
// One domain: every instance has the same action schemas (count, names and arities, in the same order), so labels
// (schema, binding) mean the same everywhere, and the same domain predicates (names, arities, kinds, in the same order;
// the front end builds them from the domain file). Table predicate ids [0, num_domain_predicates()) are the domain
// predicates' ids in that order, a function of the domain alone: a training table and an evaluation table over other
// instances give the same ids to the same domain predicates. Predicates a problem introduces (problem-local: derived
// predicates that only problem axioms define, e.g. goal normalization's "axiom_0") mean something else in every
// problem, so each gets its own table id after the domain's, in (instance, predicate) order, and
// predicate_instances() names its instance. Instance::pred_map maps instance predicate ids to table ids. Objects stay
// instance-local: a label's objects are instance i's; object offsets (the exclusive cumsum of the instances' object
// counts over a batch) globalize them.
//
// Rows. A batch over a table has rows of words() atom words, the widest instance's max_words() (unused words are zero),
// followed by numeric_words() numeric words (the most any instance has; instance i uses the first numeric_words of its
// own and leaves the rest zero). Numeric tables run on the CPU only.
//
// A table holds one domain. Batches that mix domains run over a TaskSuite (rl/task_suite.hpp): one table per
// domain, every RL entry point takes either.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mymyr::rl
{
class TaskTable;
using TaskTablePtr = std::shared_ptr<const TaskTable>;

class TaskTable : public std::enable_shared_from_this<TaskTable>
{
    struct Private
    {
    };

public:
    /// What the table keeps of one instance.
    struct Instance
    {
        TaskPtr task;
        u32 num_objects = 0;
        u32 words = 1;          // Task::max_words(), at least 1
        u32 numeric_words = 0;  // Task::numeric_words()
        u32 ow = 1;             // object-bitset words (the device kernels' OW)
        std::vector<u64> init;  // the initial state as a table row [table words + table numeric words]
        std::vector<u64> goal_pos, goal_neg;  // the fluent goal literals as masks [table words]
        bool goal_derived = false;            // the goal has derived literals (a mask test cannot decide it)
        bool goal_unsatisfiable = false;      // the goal holds a statically false literal
        std::vector<u32> pred_map;            // instance predicate id -> table predicate id
    };

    /// Checks that the tasks are instances of one domain and builds the table. Throws std::invalid_argument (with a
    /// `mymyr: TaskTable:` message) for an empty list, a null task, or instances of different domains.
    static TaskTablePtr create(std::vector<TaskPtr> tasks);
    /// The table of one instance.
    static TaskTablePtr single(TaskPtr task);
    /// Empty if `task` is of the domain of `ref` (the same action schemas and domain predicates, in the same order),
    /// else why not ("action schema 2 is ..., instance 0 has ..."; `ref` plays instance 0).
    [[nodiscard]] static std::string domain_mismatch(const Task& ref, const Task& task);

    TaskTable(Private, std::vector<TaskPtr> tasks);
    TaskTable(const TaskTable&) = delete;
    TaskTable& operator=(const TaskTable&) = delete;

    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_inst.size()); }
    [[nodiscard]] const Instance& instance(u32 i) const { return m_inst.at(i); }
    [[nodiscard]] const TaskPtr& task(u32 i) const { return m_inst.at(i).task; }
    [[nodiscard]] const std::vector<Instance>& instances() const noexcept { return m_inst; }

    /// Atom words of a row: the widest instance's Task::max_words() (at least 1).
    [[nodiscard]] u32 words() const noexcept { return m_words; }
    /// Numeric words of a row (the most any instance has; 0 for classical tables).
    [[nodiscard]] u32 numeric_words() const noexcept { return m_numeric_words; }
    [[nodiscard]] u32 row_words() const noexcept { return m_words + m_numeric_words; }
    /// Schemas (every instance's) and the largest schema arity (the label width).
    [[nodiscard]] u32 num_schemas() const noexcept { return static_cast<u32>(m_schema_names.size()); }
    [[nodiscard]] u32 label_width() const noexcept { return m_label_width; }
    [[nodiscard]] u32 max_objects() const noexcept { return m_max_objects; }
    /// Whether some instance has numeric fluents (the device refuses such tables).
    [[nodiscard]] bool numeric() const noexcept { return m_numeric; }

    [[nodiscard]] const std::vector<std::string>& schema_names() const noexcept { return m_schema_names; }
    [[nodiscard]] const std::vector<u32>& schema_arities() const noexcept { return m_schema_arity; }
    /// Table predicates (the domain's, then the problem-local ones), their arities and kinds (k_pred_* codes), and
    /// the instance a problem-local predicate belongs to (-1: a domain predicate).
    [[nodiscard]] const std::vector<std::string>& predicate_names() const noexcept { return m_pred_names; }
    [[nodiscard]] const std::vector<u32>& predicate_arities() const noexcept { return m_pred_arity; }
    [[nodiscard]] const std::vector<i32>& predicate_kinds() const noexcept { return m_pred_kind; }
    [[nodiscard]] const std::vector<i32>& predicate_instances() const noexcept { return m_pred_instance; }
    /// The domain predicates: table predicate ids [0, num_domain_predicates()).
    [[nodiscard]] u32 num_domain_predicates() const noexcept { return m_domain_preds; }

    /// A hash of the instances' fingerprints in table order (Task::fingerprint).
    [[nodiscard]] u64 fingerprint() const;

    /// Throws std::invalid_argument unless every id of `task_ids` [n] names an instance (null: instance 0, valid).
    void check_task_ids(const i32* task_ids, u64 n) const;
    /// The task of a row (task_ids null: instance 0). No bounds check (check_task_ids first).
    [[nodiscard]] const Task& task_of(const i32* task_ids, u64 row) const noexcept
    {
        return *m_inst[task_ids ? static_cast<u32>(task_ids[row]) : 0].task;
    }

private:
    std::vector<Instance> m_inst;
    u32 m_words = 1, m_numeric_words = 0, m_label_width = 0, m_max_objects = 0;
    bool m_numeric = false;
    std::vector<std::string> m_schema_names;
    std::vector<u32> m_schema_arity;
    std::vector<std::string> m_pred_names;
    std::vector<u32> m_pred_arity;
    std::vector<i32> m_pred_kind, m_pred_instance;
    u32 m_domain_preds = 0;
};

/// The atom metadata of a table (what an encoder needs): the instances' atom_metadata
/// (task_arrays.hpp) concatenated, with per-instance offsets and table predicate ids.
///   atom_offsets [I + 1]    fluent slot j of instance i is row atom_offsets[i] + j of the atom_* arrays
///   atom_pred [F], atom_args [F, A] (instance-local objects, -1 padding), atom_cid [F]
///   derived_offsets [I + 1], derived_pred, derived_args [., A];  static_offsets [I + 1], static_pred, static_args
///   num_objects [I], object_offsets [I + 1] (their exclusive cumsum), num_atoms [I], domain [I] (all 0)
///   pred_arity [P], pred_kind [P], pred_instance [P] (table predicates; -1: a domain predicate, else the instance of
///   a problem-local one); scalars num_instances, num_predicates, num_domain_predicates, max_arity, label_width, words
/// A snapshot: under lazy slots it covers the slots published when it was taken.
[[nodiscard]] ArrayBundle table_atom_metadata(const TaskTable& table);

}  // namespace mymyr::rl
