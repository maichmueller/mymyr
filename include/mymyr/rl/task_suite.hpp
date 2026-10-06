#pragma once
// TaskSuite: instances of several domains in one batch. A suite is a list of
// one-domain TaskTables (rl/task_table.hpp), one per domain; every RL entry point takes a suite or a table (a table
// is the suite of its one domain, TaskSuite::of, which costs nothing), so one code path serves both, and each domain's
// rows run on its table's own path.
//
//   auto suite = rl::TaskSuite::create({blocks_table, gripper_table});   // std::shared_ptr<const TaskSuite>, immutable
//   auto suite = rl::TaskSuite::group({blocks_a, gripper_a, blocks_b}); // one table per domain, first-seen order
//   rl::expand(*suite, batch, task_ids, out);                          // rows of any instance of any domain
//
// Ids. A batch's task ids are global: int32 instance indices into the suite, [0, size()). Global instance g is instance
// local_id(g) of domain domain_of(g)'s table. create(tables) numbers table 0's instances first, then table 1's, ...;
// group(tasks) keeps the tasks' order (global id i is tasks[i]) and puts each task into the table of the first domain
// it belongs to (TaskTable::domain_mismatch), in the order the domains are first seen.
//
// Everything a domain defines stays the domain's: labels (schema, binding) carry the domain's own schema ids (its
// table's) and instance-local object ids, and the atom metadata's predicate ids are the domain table's (its domain
// predicates first: a function of the domain alone, so per-domain encoder weights carry over between suites). A
// consumer tells the domains apart by the row's task id (domain_of; Python `rl.task_domains(suite, task_ids)`), and
// may flatten schema ids with schema_offsets() (schema_offset(d) + schema).
//
// Rows. A batch over a suite has rows of words() atom words (the widest domain's table width; a narrower instance
// leaves its extra words zero) followed by numeric_words() numeric words (the most any domain has). The count cache of
// the device fast path has S = max_schemas() segments per state (a domain's schemas past its own count are never
// applicable: count 0).
//
// Domains are distinct: two tables of one domain are refused (build one table of their instances).

#include "mymyr/core/types.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mymyr::rl
{
class TaskSuite;
using TaskSuitePtr = std::shared_ptr<const TaskSuite>;

class TaskSuite
{
    struct Private
    {
    };

public:
    /// Where a global instance lives: instance `local` of domain `domain`'s table.
    struct Ref
    {
        u32 domain = 0;
        u32 local = 0;
    };

    /// The suite of these domains (one table each, in this order; global ids: table 0's instances, then table 1's,
    /// ...). Throws std::invalid_argument (with a `mymyr: TaskSuite:` message) for an empty list, a null table, two
    /// tables of one domain, or more than 2^31 - 1 instances.
    static TaskSuitePtr create(std::vector<TaskTablePtr> tables);
    /// The suite of these tasks, grouped into one table per domain (first-seen order; a domain's instances in task
    /// order); global id i is tasks[i]. Throws std::invalid_argument for an empty list or a null task.
    static TaskSuitePtr group(std::vector<TaskPtr> tasks);
    /// The suite of one table's domain (global ids are the table's).
    static TaskSuitePtr of(TaskTablePtr table);
    static TaskSuitePtr of(const TaskTable& table) { return of(table.shared_from_this()); }

    TaskSuite(Private, std::vector<TaskTablePtr> tables, std::vector<Ref> refs);
    TaskSuite(const TaskSuite&) = delete;
    TaskSuite& operator=(const TaskSuite&) = delete;

    /// Instances (global ids [0, size())).
    [[nodiscard]] u32 size() const noexcept { return m_size; }
    [[nodiscard]] u32 num_domains() const noexcept { return static_cast<u32>(m_tables.size()); }
    /// One domain: the suite is its table (TaskSuite::of), every op runs the table's path.
    [[nodiscard]] bool single_domain() const noexcept { return m_tables.size() == 1; }
    [[nodiscard]] const TaskTablePtr& table(u32 domain) const { return m_tables.at(domain); }
    [[nodiscard]] const std::vector<TaskTablePtr>& tables() const noexcept { return m_tables; }

    /// The domain and local id of global instance g (no bounds check: check_task_ids first).
    [[nodiscard]] Ref ref(u32 g) const noexcept { return m_refs.empty() ? Ref{0, g} : m_refs[g]; }
    [[nodiscard]] u32 domain_of(u32 g) const noexcept { return ref(g).domain; }
    [[nodiscard]] u32 local_id(u32 g) const noexcept { return ref(g).local; }
    /// The global id of instance `local` of domain `domain`.
    [[nodiscard]] u32 global_id(u32 domain, u32 local) const
    {
        return m_global.empty() ? local : m_global.at(domain).at(local);
    }
    /// The table and instance of global instance g.
    [[nodiscard]] const TaskTable& table_of(u32 g) const { return *m_tables[domain_of(g)]; }
    [[nodiscard]] const TaskTable::Instance& instance(u32 g) const
    {
        const Ref r = ref(g);
        return m_tables[r.domain]->instance(r.local);
    }
    [[nodiscard]] const TaskPtr& task(u32 g) const
    {
        const Ref r = ref(g);
        return m_tables[r.domain]->task(r.local);
    }

    /// Atom words of a row: the widest domain's TaskTable::words().
    [[nodiscard]] u32 words() const noexcept { return m_words; }
    /// Numeric words of a row (the most any domain has).
    [[nodiscard]] u32 numeric_words() const noexcept { return m_numeric_words; }
    [[nodiscard]] u32 row_words() const noexcept { return m_words + m_numeric_words; }
    /// The largest schema arity of any domain (the label width).
    [[nodiscard]] u32 label_width() const noexcept { return m_label_width; }
    [[nodiscard]] u32 max_objects() const noexcept { return m_max_objects; }
    /// The most schemas of any domain (segments per state of the device's count cache).
    [[nodiscard]] u32 max_schemas() const noexcept { return m_max_schemas; }
    /// Whether some domain has numeric fluents.
    [[nodiscard]] bool numeric() const noexcept { return m_numeric; }
    /// [D + 1]: domain d's schemas are flat schema ids [schema_offsets[d], schema_offsets[d + 1]).
    [[nodiscard]] const std::vector<u32>& schema_offsets() const noexcept { return m_schema_offsets; }
    [[nodiscard]] u32 schema_offset(u32 domain) const { return m_schema_offsets.at(domain); }
    /// The domain's name (its first instance's domain_name).
    [[nodiscard]] std::string domain_name(u32 domain) const;
    /// Instance g's initial state as a row of `words` atom words (>= its table's words()), then `numeric_words` numeric
    /// words (>= its table's numeric_words()): the table row's atom words, zeros, its numeric words, zeros.
    void initial_row(u32 g, u64* row, u32 words, u32 numeric_words) const;

    /// The table's fingerprint for one domain, else a hash of the domains' fingerprints and the id mapping.
    [[nodiscard]] u64 fingerprint() const;

    /// Throws std::invalid_argument unless every id of `task_ids` [n] names an instance (null: instance 0, valid).
    void check_task_ids(const i32* task_ids, u64 n) const;
    /// The task of a row (task_ids null: instance 0). No bounds check (check_task_ids first).
    [[nodiscard]] const Task& task_of(const i32* task_ids, u64 row) const noexcept
    {
        return *task(task_ids ? static_cast<u32>(task_ids[row]) : 0);
    }
    /// "the table's" or "the suite's" (error messages).
    [[nodiscard]] const char* noun() const noexcept { return single_domain() ? "table" : "suite"; }

private:
    std::vector<TaskTablePtr> m_tables;
    std::vector<Ref> m_refs;                 // [I] (empty: one domain, identity)
    std::vector<std::vector<u32>> m_global;  // [D][I_d] local -> global (empty: identity)
    std::vector<u32> m_schema_offsets;
    u32 m_size = 0, m_words = 1, m_numeric_words = 0, m_label_width = 0, m_max_objects = 0, m_max_schemas = 0;
    bool m_numeric = false;
};

/// The atom metadata of a suite (what an encoder needs): the
/// instances' atom_metadata (task_arrays.hpp) concatenated in global instance order, with per-instance offsets, and
/// each domain's predicate id space. A one-domain suite's is its table's (table_atom_metadata).
///   atom_offsets [I + 1]    fluent slot j of instance g is row atom_offsets[g] + j of the atom_* arrays
///   atom_pred [F]           the domain table's predicate id (row pred_offsets[domain[g]] + atom_pred of pred_*)
///   atom_args [F, A] (instance-local objects, -1 padding), atom_cid [F]
///   derived_offsets [I + 1], derived_pred, derived_args, derived_cid; static_offsets [I + 1], static_pred, static_args
///   num_objects [I], object_offsets [I + 1] (their exclusive cumsum), num_atoms [I], words [I]
///   domain [I], local_id [I]                       the instance's domain and its id in the domain's table
///   pred_offsets [D + 1]             domain d's predicates are rows [pred_offsets[d], pred_offsets[d + 1])
///   pred_arity [P], pred_kind [P], pred_instance [P]  per domain's predicate (-1: a domain predicate, else the
///                                    global instance of a problem-local one)
///   domain_predicates [D]            each domain's num_domain_predicates (its ids [0, n) are the domain's)
///   schema_offsets [D + 1], schema_arity [S_total]  each domain's schemas (flat id schema_offsets[d] + schema)
///   scalars num_instances, num_domains, num_predicates (P), num_domain_predicates (domain 0's), num_schemas (domain
///   0's), max_arity, label_width, words
/// A snapshot: under lazy slots it covers the slots published when it was taken.
[[nodiscard]] ArrayBundle suite_atom_metadata(const TaskSuite& suite);
}  // namespace mymyr::rl
