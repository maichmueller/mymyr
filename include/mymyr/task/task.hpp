#pragma once
// Task: the immutable, shared, compiled planning task.
//
//   auto task = mymyr::Task::create(read_task_text_file("p.txt"));   // std::shared_ptr<const Task>
//   WorkspaceLease ws = task->workspace();                           // scratch of its own, while ws lives
//   Successors& succ = ws->successors();
//
// A task holds the normalized TaskData, the static relations (unary bitsets, binary row tables, k-ary sets and
// projections), the two-level atom index, the compiled schemas, axioms and goal, and the initial state. The only
// mutations after construction are guarded caches and the CAS-guarded append of lazily assigned slots
// (atom_index.hpp; TaskOptions::Atoms::Lazy only).
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
// metric values by heuristics::ActionCosts.

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
#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>
#include <unordered_map>

namespace mymyr
{
class Task;
class Workspace;
class SlotMap;
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

/// Construction options. Together with the normalized task they determine everything a task computes, its state
/// encoding included: the same task data and options give the same atom slots, state words and rows in every run,
/// thread and process (except under Atoms::Lazy).
struct TaskOptions
{
    /// How fluent and derived atoms get their slots (bit positions in a state; task/atom_index.hpp).
    enum class Atoms : u8
    {
        /// Reachable slots, or dense ones where the state is not wider that way (W_dense <= W_reachable, fluent and
        /// derived); dense when the reachability analysis passes reachability_budget.
        Auto,
        /// One slot per relaxed-reachable atom, in canonical order. Throws std::length_error when the analysis passes
        /// reachability_budget.
        Reachable,
        /// slot = canonical id: one slot per atom of the typed-dense space. No analysis at construction.
        Dense,
        /// Slots assigned on first touch: the numbering depends on the order in which searches meet the atoms (threads,
        /// earlier searches), so rows of such a task mean nothing outside the task object. For searches over tasks
        /// whose reachable space is far larger than what they explore.
        Lazy,
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

    /// The default bound of the reachability analysis, in steps (reachability::reachable_atoms): a few seconds and
    /// about a gigabyte at most, several times what the largest IPC 2023 learning-track test tasks need.
    static constexpr u64 k_reachability_budget = u64{1} << 25;

    Atoms atoms = Atoms::Auto;
    Matching matching = Matching::Auto;
    NumericStorageMode numeric_storage = NumericStorageMode::Auto;
    /// > 0: snap every stored numeric value to the grid q * round(v / q). Changes the state space (never implicit).
    f64 numeric_quantum = 0;
    /// Tolerant numeric semantics: values below 1e6 snap to a 1e-9 grid, non-finite results are undefined,
    /// comparisons use a 1e-9 tolerance. Changes the state space (hydropower: 7.6x fewer states at depth 45).
    bool numeric_tolerant = false;
    u32 fc_auto_free_params = 4;
    /// Work bound of the relaxed-reachability analysis behind Atoms::Auto and Atoms::Reachable, in steps of its joins:
    /// a count, not a time, so whether it suffices is a property of the task. ~u64{0}: unbounded.
    u64 reachability_budget = k_reachability_budget;
};

/// Construction facts, for diagnostics and reports.
struct TaskInfo
{
    AtomMode atom_mode = AtomMode::Dense;
    u64 dense_fluent = 0;        // F_fluent of the typed-dense space
    u64 dense_derived = 0;       // F_derived
    u64 reachable_fluent = 0;    // relaxed-reachable fluent atoms (0 when the analysis did not run or did not finish)
    u64 reachable_derived = 0;   // relaxed-reachable derived atoms
    u64 reachability_steps = 0;  // work of the analysis (0 when it did not run)
    bool reachability_complete = false;  // the analysis ran and finished within the budget
    double build_s = 0;
};

/// A ground atom given by its predicate and objects (Task::make_state).
struct AtomArgs
{
    PredicateId predicate;
    std::span<const ObjectId> objects;
};

class Task : public std::enable_shared_from_this<Task>
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
    /// Cached task over the same normalized problem and options with fixed atom slots. Lazy is not a frozen mode.
    [[nodiscard]] std::shared_ptr<const Task> frozen(TaskOptions::Atoms atoms = TaskOptions::Atoms::Auto) const;
    /// Cached map to a fixed-slot task with the same problem fingerprint and numeric layout, or null if incompatible.
    /// The target must outlive every use of the returned map.
    [[nodiscard]] std::shared_ptr<SlotMap> slot_map_to(const Task& target) const;
    /// Name-based map, cached while the target lives. The target must outlive every use of the returned map.
    [[nodiscard]] std::shared_ptr<SlotMap> translation_map_to(const Task& target) const;
    /// Translate a state into target's slot and numeric layout. Missing true atoms/values raise unless drop is true.
    [[nodiscard]] State translate(StateView state, const Task& target, bool drop = false, bool* lossless = nullptr) const;

    [[nodiscard]] u32 num_objects() const noexcept { return m_compiled.num_objects; }
    [[nodiscard]] u32 num_schemas() const noexcept { return static_cast<u32>(m_compiled.schemas.size()); }
    [[nodiscard]] bool has_axioms() const noexcept { return !m_compiled.strata.empty(); }
    /// Fluent width W in u64 words: words_for(fluent slots). Fixed at construction for frozen slots; grows under lazy
    /// slots as atoms get theirs.
    [[nodiscard]] u32 words() const noexcept { return bits::words_for(m_atoms.fluent_slots()); }
    /// Largest width any state of this task can have (words() for frozen slots).
    [[nodiscard]] u32 max_words() const noexcept { return bits::words_for(m_atoms.max_fluent_slots()); }
    /// Hash of the state encoding: fluent and derived atoms' predicate and ordered object names in slot order,
    /// numeric function and argument names, and numeric storage. Equal hashes identify equal row layouts across
    /// parses (subject to hash collisions). Lazy slots have a distinct tag and hash
    /// the slots assigned so far.
    [[nodiscard]] u64 encoding_fingerprint() const;

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
    /// The goal as written, expanded into signed ground literals (TaskData::encoding_goal), for encodings; sorted by
    /// (positive first, predicate id, objects) without duplicates. The planner's goal test is unaffected.
    [[nodiscard]] std::span<const formalism::Literal> encoding_goal() const noexcept { return m_data.encoding_goal_literals(); }
    /// Whether encoding_goal() is the whole goal: false when a conjunct could not be expanded (a disjunction with two
    /// or more disjuncts that static atoms do not decide, numeric constraints, too many quantifier bindings), or when
    /// the task came from a text without the goal as written and its planner goal mentions predicates that goal
    /// normalization introduced.
    [[nodiscard]] bool encoding_goal_complete() const noexcept { return m_data.encoding_goal_complete; }
    /// Whether a literal of encoding_goal() over a static predicate fails in the initial atoms: no state is a goal.
    [[nodiscard]] bool encoding_goal_unsatisfiable() const noexcept { return m_data.encoding_goal_unsatisfiable; }
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
    /// Slot of the fluent atom pred(args) if it has one (reachable mode: if it is relaxed-reachable; lazy mode: if it was
    /// ever produced).
    [[nodiscard]] SlotId find_atom(PredicateId pred, std::span<const ObjectId> args) const;

private:
    void choose_atom_mode();
    void build_initial_state();
    [[nodiscard]] u64 compute_encoding_fingerprint() const;

    formalism::TaskData m_data;
    TaskOptions m_options;
    TaskInfo m_info;
    plan::Compiled m_compiled;
    AtomIndex m_atoms;
    State m_initial;
    mutable std::mutex m_frozen_mutex;
    mutable std::array<std::shared_ptr<const Task>, 3> m_frozen;
    mutable std::array<std::shared_ptr<SlotMap>, 3> m_slot_maps;
    /// Name-based maps by target uid. Entries hold their targets weakly; each insertion drops the expired ones.
    struct Translation
    {
        std::weak_ptr<const Task> target;
        std::shared_ptr<SlotMap> map;
    };
    mutable std::unordered_map<u64, Translation> m_translations;
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
    mutable Once m_encoding_once;
    mutable u64 m_encoding = 0;
};

using TaskPtr = std::shared_ptr<const Task>;
}  // namespace mymyr
