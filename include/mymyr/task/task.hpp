#pragma once
// Task: the immutable, shared, compiled planning task.
//
//   auto task = mymyr::Task::create(read_task_text_file("p.txt"));   // std::shared_ptr<const Task>
//   WorkspaceLease ws = task->workspace();                           // scratch of its own, while ws lives
//   Successors& succ = ws->successors();
//
// A task holds the normalized TaskData, the static relations (unary bitsets, binary row tables, k-ary sets and
// projections), the two-level atom index, the compiled schemas, axioms and goal, and the initial state. The only
// mutation after construction is the CAS-guarded append of lazily assigned slots (atom_index.hpp).
//
// Mutable scratch (the matching engine, the axiom evaluator, the successor generator) lives in Workspaces, which the
// task lends out: every search, enumeration or evaluation holds a WorkspaceLease while it runs, and every live lease
// has a workspace of its own. Code that runs inside an enumeration (an observer, a goal test, a heuristic written in
// Python) may therefore enumerate actions, test applicability, evaluate axioms or start another search on the same
// task and thread without disturbing the enumeration around it. Released workspaces are kept per thread for the next
// lease (core/per_thread.hpp), so a thread uses as many as its deepest nesting, and are freed with the task.
//
// Numeric fluents, constraints and effects (task/numeric.hpp): a state carries one value per numeric slot after
// its atom bits ([bits | slots]); numeric_slots() == 0 for classical tasks, whose states are unchanged. Action
// costs (total-cost effects, the metric) are evaluated by the successor generator (Delta::aux) and turned into
// mimir's metric values by heuristics::ActionCosts.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/once.hpp"
#include "mymyr/core/per_thread.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/atom_index.hpp"
#include "mymyr/task/plan.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace mymyr
{
class Task;
class Workspace;
namespace detail
{
struct WorkspacePool;
}

/// A workspace of a task, lent to its holder until the lease is destroyed (Task::workspace()). Move-only. The lease
/// must not outlive the task. Its accessors exist for named leases only, so a workspace is never used after its
/// lease ended: `Successors& s = task.workspace()->successors();` does not compile. A lease may be released on
/// another thread than the one that took it.
class WorkspaceLease
{
public:
    WorkspaceLease() noexcept = default;  // empty
    WorkspaceLease(WorkspaceLease&& other) noexcept;
    WorkspaceLease& operator=(WorkspaceLease&& other) noexcept;
    WorkspaceLease(const WorkspaceLease&) = delete;
    WorkspaceLease& operator=(const WorkspaceLease&) = delete;
    ~WorkspaceLease();

    [[nodiscard]] Workspace& operator*() const& noexcept { return *m_ws; }
    [[nodiscard]] Workspace* operator->() const& noexcept { return m_ws.get(); }
    Workspace& operator*() const&& = delete;
    Workspace* operator->() const&& = delete;
    [[nodiscard]] explicit operator bool() const noexcept { return m_ws != nullptr; }
    /// Returns the workspace to the task (no-op for an empty lease).
    void release() noexcept;

private:
    friend class Task;
    WorkspaceLease(const Task& task, detail::WorkspacePool* home, std::unique_ptr<Workspace> ws) noexcept;

    const Task* m_task = nullptr;
    detail::WorkspacePool* m_home = nullptr;  // the pool of the thread that took it
    std::unique_ptr<Workspace> m_ws;
};

struct TaskOptions
{
    enum class Atoms : u8
    {
        Auto,    // frozen when W_dense <= frozen_max_words or W_dense <= 2 * W_lazy (pilot estimate)
        Lazy,    // slots assigned on first touch
        Frozen,  // slot = canonical id (GPU, datasets)
    };
    enum class Matching : u8
    {
        Auto,             // forward checking for matchers with more than fc_auto_free_params free parameters
        FixedOrder,       // fixed order with backward row checks
        ForwardChecking,  // fail-first forward checking everywhere
    };

    enum class NumericStorageMode : u8
    {
        Auto,  // I32 when the task is statically integral and no quantization is set, else F64
        F64,
        I32,
    };

    Atoms atoms = Atoms::Auto;
    Matching matching = Matching::Auto;
    NumericStorageMode numeric_storage = NumericStorageMode::Auto;
    /// > 0: snap every stored numeric value to the grid q * round(v / q). Changes the state space (never implicit).
    f64 numeric_quantum = 0;
    /// Mimir-C# numeric semantics: values below 1e6 snap to a 1e-9 grid, non-finite results are undefined,
    /// comparisons use a 1e-9 tolerance. Changes the state space (hydropower: 7.6x fewer states at depth 45).
    bool numeric_tolerant = false;
    u32 fc_auto_free_params = 4;
    u32 frozen_max_words = 8;
    /// Expansions of the bounded lazy BrFS that estimates W_lazy for the Auto rule (only run when W_dense exceeds
    /// frozen_max_words). 0 skips the pilot and chooses lazy slots.
    u32 pilot_expansions = 1024;
};

/// Construction facts, for diagnostics and reports.
struct TaskInfo
{
    AtomMode atom_mode = AtomMode::Lazy;
    u64 dense_fluent = 0;   // F_fluent of the typed-dense space
    u64 dense_derived = 0;  // F_derived
    u32 pilot_words = 0;    // W_lazy estimate (0 if no pilot ran)
    double build_s = 0;
};

/// A ground atom given by its predicate and objects (Task::make_state).
struct AtomArgs
{
    PredicateId predicate;
    std::span<const ObjectId> objects;
};

class Task
{
    struct Private
    {
    };

public:
    /// Compiles `data` (validated first). Throws std::invalid_argument for unsupported input.
    static std::shared_ptr<const Task> create(formalism::TaskData data, const TaskOptions& options = {});
    static std::shared_ptr<const Task> from_text_file(const std::string& path, const TaskOptions& options = {});

    Task(Private, formalism::TaskData data, const TaskOptions& options);
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task();

    [[nodiscard]] u64 uid() const noexcept { return m_workspaces.uid(); }
    /// Content hash of the normalized task (formalism/fingerprint.hpp), computed on first use: equal for tasks built
    /// from the same normalized task in any process (pickling checks it; the uid is per instance).
    [[nodiscard]] u64 fingerprint() const;
    [[nodiscard]] const formalism::TaskData& data() const noexcept { return m_data; }
    [[nodiscard]] const TaskOptions& options() const noexcept { return m_options; }
    [[nodiscard]] const TaskInfo& info() const noexcept { return m_info; }
    [[nodiscard]] const plan::Compiled& compiled() const noexcept { return m_compiled; }
    [[nodiscard]] const AtomIndex& atoms() const noexcept { return m_atoms; }

    [[nodiscard]] u32 num_objects() const noexcept { return m_compiled.num_objects; }
    [[nodiscard]] u32 num_schemas() const noexcept { return static_cast<u32>(m_compiled.schemas.size()); }
    [[nodiscard]] bool has_axioms() const noexcept { return !m_compiled.strata.empty(); }
    /// Current fluent width W: words_for(assigned fluent slots). Grows only under lazy slots.
    [[nodiscard]] u32 words() const noexcept { return bits::words_for(m_atoms.fluent_slots()); }
    /// Largest width any state of this task can have.
    [[nodiscard]] u32 max_words() const noexcept { return bits::words_for(m_atoms.max_fluent_slots()); }

    // numerics (task/numeric.hpp)
    /// Numeric slots (ground fluent functions with an initial value); 0 for classical tasks.
    [[nodiscard]] u32 numeric_slots() const noexcept { return m_compiled.num.slots; }
    /// Numeric words per state: slots (F64) or ceil(slots / 2) (I32).
    [[nodiscard]] u32 numeric_words() const noexcept { return m_compiled.num.words; }
    [[nodiscard]] NumericStorage numeric_storage() const noexcept { return m_compiled.num.storage; }
    [[nodiscard]] const plan::Numeric& numeric() const noexcept { return m_compiled.num; }
    /// Value of numeric slot `slot` in s.
    [[nodiscard]] f64 numeric_value(StateView s, u32 slot) const;
    /// The values of every numeric slot of s.
    [[nodiscard]] std::vector<f64> numeric_values(StateView s) const;
    /// "(function o1 ... ok)" of a numeric slot.
    [[nodiscard]] std::string numeric_name(u32 slot) const;

    [[nodiscard]] State initial_state() const { return m_initial; }
    /// The state whose true fluent atoms are `atoms` (in any order, repeats allowed) and whose numeric slots have
    /// `values` (numeric_slots() of them, in slot order; none for a classical task). Static atoms are not part of a
    /// state, and derived atoms follow from the axioms whenever a state is evaluated, so both are rejected here.
    /// Throws std::invalid_argument for a static or derived atom, a wrong number of objects or of values, an atom
    /// outside the reachable domains of its predicate, or a NaN value; std::overflow_error for a value an I32 slot
    /// cannot hold (TaskOptions::numeric_storage).
    [[nodiscard]] State make_state(std::span<const AtomArgs> atoms, std::span<const f64> values = {}) const;
    /// Goal test. Evaluates the axioms (in a workspace of its own) when the goal mentions derived predicates.
    [[nodiscard]] bool is_goal(StateView s) const;
    /// Order-independent hash of a state over the canonical ids of its atoms: equal for equal states under any slot
    /// numbering (lazy slots depend on the order of first touch), so it identifies states across runs and threads.
    /// Numeric tasks: it also covers the numeric values, as canonical doubles (the same under I32 and F64 slots).
    [[nodiscard]] u64 canonical_hash(StateView s) const;

    /// Lends a workspace that no other live lease holds: one this thread released before, or a new one. Its successor
    /// generator has the default settings (witness pruning and canonical order on, no schema filter).
    [[nodiscard]] WorkspaceLease workspace() const;
    /// Number of workspaces this task has made (diagnostics: a thread makes as many as its deepest nesting of leases).
    [[nodiscard]] usize workspaces_made() const noexcept { return m_workspaces_made.load(std::memory_order_relaxed); }

    // names (API, debugging)
    [[nodiscard]] std::string schema_name(SchemaId s) const;
    [[nodiscard]] std::string object_name(ObjectId o) const;
    [[nodiscard]] std::string format(const ActionLabel& a) const;
    [[nodiscard]] std::string format(SlotId fluent_slot) const;
    /// The true fluent atoms of s, formatted, in slot order.
    [[nodiscard]] std::vector<std::string> format_atoms(StateView s) const;
    /// Slot of the fluent atom pred(args) if it has one (lazy mode: if it was ever produced).
    [[nodiscard]] SlotId find_atom(PredicateId pred, std::span<const ObjectId> args) const;

private:
    void choose_atom_mode();
    u32 pilot_words();
    void build_initial_state();

    formalism::TaskData m_data;
    TaskOptions m_options;
    TaskInfo m_info;
    plan::Compiled m_compiled;
    AtomIndex m_atoms;
    State m_initial;
    friend class WorkspaceLease;
    void give_back(detail::WorkspacePool* home, std::unique_ptr<Workspace> ws) const noexcept;

    mutable PerThread<detail::WorkspacePool> m_workspaces;  // each thread's released workspaces
    /// Workspaces released on another thread than the one that took them, for the next lease of any thread.
    mutable std::mutex m_spare_mutex;
    mutable std::vector<std::unique_ptr<Workspace>> m_spare;
    mutable std::atomic<usize> m_spare_count{0};
    mutable std::atomic<usize> m_workspaces_made{0};
    mutable Once m_fingerprint_once;
    mutable u64 m_fingerprint = 0;
};

using TaskPtr = std::shared_ptr<const Task>;
}  // namespace mymyr
