#pragma once
// Batched planning environments: N environments over a TaskSuite (rl/task_suite.hpp; a TaskTable is the suite of
// its one domain), stepped in lockstep. This is the host implementation, the reference of the device step
// (cuda/env.hpp), the CPU backend of mymyr.rl.torch and the step of the CPU pool (rl/pool.hpp).
//
// An environment's state is a row of state words, its instance (task_ids: a global index into the suite; a Task
// is a table of one), its episode step counter, its RNG draw counter and, optionally, its goal as masks (goal_pos /
// goal_neg); every array is caller-owned (EnvBatch, StepOutputs). One step of environment i of instance t:
//   1. choose: a caller-given index into the canonical successor order of s (schema, then binding; the action
//      representation is "successor slots"), or, without actions, a uniform index from the counter-
//      based RNG (rl/rng.hpp: seed, env id = first_env + i, the draw counter, purpose k_successor; per-row seeds and
//      env ids replace the config's seed and first_env + i where the batch has them, e.g. the JAX env's keys). The
//      draw counter advances by one in every step, used or not. In Label mode the caller supplies a schema and
//      binding directly; the random policy is unavailable;
//   2. move to that successor s' (rows byte-equal to rl::expand's; the label (schema, binding) is reported, with
//      the schema ids of t's domain and instance t's object ids);
//   3. goal: s' is a goal state by instance t's goal test (derived literals included) or, when the batch carries goal
//      masks, by the mask test (s' & goal_pos) == goal_pos and (s' & goal_neg) == 0;
//   4. dead end (EnvConfig::dead_end): under DeadEnd::NoSuccessors a non-goal s' without successors is a dead end, and
//      so is a state without successors that cannot move (stuck: s' = s); under DeadEnd::None nothing is;
//   5. rewards and termination: reward = step_reward, plus goal_reward at a goal, plus dead_end_reward
//      at a dead end; terminated = goal, or dead end when dead_end_terminal; truncated = not terminated and
//      max_steps > 0 and the episode's step count reaches max_steps;
//   6. count: the successors of s' in Index mode; in Label mode 0/1 (whether an applicable action exists);
//   7. autoreset (SAME_STEP): a terminated or truncated environment restarts in the same
//      step from the initial state of its instance, or of next_task_ids[i] when the step is given next_task_ids (the
//      caller's curriculum; over a suite the next instance may be of another domain); the row's task id and goal
//      masks follow the new instance. final_states keeps s' (the
//      bootstrap state), count is the initial state's. Without autoreset the caller resets finished environments with
//      reset(mask) (TorchRL's convention).
// Special cases: goals and terminal dead ends are absorbing until reset: no move, reward zero and no step-counter
// advance; draw counters still advance. Whether the current state is a goal (the start of an absorbing step) is the
// batch's at_goal flag where it has one: reset, step and refresh keep it (a reset into a goal state, a relabelled goal
// mask that already holds), so a step reads it instead of testing the state again. With goal masks the goal test is
// the masks' test throughout (resets included). An index outside [0, count) or an inapplicable label in a
// nonterminal state with applicable actions does not move
// (s' = s, reward step_reward, not a goal, not a dead end; flagged in `invalid`); a stuck environment does not move
// either.
//
// Rows are `words` atom words wide (at least the suite's words(), so a reset into any instance fits), plus the suite's
// numeric words ([bits | slots]; the numeric encoding follows each instance's storage mode). A step reads and writes
// the row's own instance's words only (TaskSuite::instance(t).words): the words past them are padding, never read,
// and zero once a reset has written the row; goal masks are read over the same words. Every step of a row is a
// function of that row's state, action and RNG stream alone, so results do not depend on the batch composition, the
// batch split or the thread count.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <vector>

namespace mymyr::rl
{
/// What makes a state a dead end.
enum class DeadEnd : u8
{
    NoSuccessors,  // a non-goal state without successors (and a stuck state)
    None,          // nothing: such states are not terminal and get no dead-end reward; episodes end by truncation
};

/// How a caller identifies the action to apply.
enum class ActionSpace : u8
{
    Index,  // an index into the successor order; count is the number of successors
    Label,  // a schema and binding; count is 0/1 (whether an applicable action exists)
};

struct EnvConfig
{
    ActionSpace actions = ActionSpace::Index;
    u64 seed = 0;              // key of the successor draws (rl/rng.hpp)
    u32 max_steps = 0;         // truncation after this many steps of an episode (0: never)
    f32 step_reward = 0.0f;
    f32 goal_reward = 1.0f;
    f32 dead_end_reward = 0.0f;  // added when the step reaches (or is stuck in) a dead end
    DeadEnd dead_end = DeadEnd::NoSuccessors;
    bool dead_end_terminal = true;  // a dead end terminates the episode
    bool autoreset = true;     // SAME_STEP autoreset; false: the caller resets finished environments
    bool canonical_order = true;   // the order successor indices refer to
    bool witness_pruning = false;  // off: every applicable ground action is a successor
};

/// The per-environment state: caller-owned arrays of `rows` environments. A null array is not read or written, except
/// `states`, which is required.
struct EnvBatch
{
    u64* states = nullptr;  // [rows, words + numeric_words], contiguous
    u64 rows = 0;
    u32 words = 0;          // atom words per row (>= TaskSuite::words())
    u32 numeric_words = 0;  // TaskSuite::numeric_words()
    i32* task_ids = nullptr;  // [rows] the (global) instance of each row (required over several instances)
    u64* goal_pos = nullptr;  // [rows, words] per-env goal masks (null: the instances' goal tests); both or neither
    u64* goal_neg = nullptr;  // [rows, words]
    i32* steps = nullptr;   // [rows] steps of the current episode
    u64* draws = nullptr;   // [rows] RNG draw counters (required without actions)
    u64 first_env = 0;      // the RNG stream (env id) of row 0
    // the device fast path's per-environment cache (cuda/env.hpp): the count cache (successor counts per schema,
    // recorded canonical keys and, for tables of several instances, the instance-grouped row order) and the matcher
    // views of the current states; unused by the host
    u32* counts = nullptr;  // [rows, cache_schemas]
    u64* views = nullptr;   // [rows, cache_view_words]
    // per-row RNG streams (e.g. the JAX env's keys and env ids)
    const u64* seeds = nullptr;    // [rows] per-row RNG keys (null: EnvConfig::seed)
    const u32* env_ids = nullptr;  // [rows] per-row env ids (null: first_env + row)
    /// [rows] whether each row's state is a goal (by its goal masks, else its instance's goal test): written by reset,
    /// step and refresh, read by step instead of testing the state (null: the host env tests it; the device env
    /// requires it). Writing states or goal masks directly needs refresh().
    u8* at_goal = nullptr;

    [[nodiscard]] u32 row_words() const noexcept { return words + numeric_words; }
    [[nodiscard]] bool goals() const noexcept { return goal_pos != nullptr; }
    [[nodiscard]] u32 instance(u64 row) const noexcept { return task_ids ? static_cast<u32>(task_ids[row]) : 0; }
};

/// Outputs of step(): caller-owned arrays of `rows` entries; null arrays are not written.
struct StepOutputs
{
    f32* reward = nullptr;      // [rows]
    u8* terminated = nullptr;   // [rows]: goal reached, or a dead end (dead_end_terminal)
    u8* truncated = nullptr;    // [rows]: max_steps reached (not terminated)
    i32* count = nullptr;       // [rows]: successor count (Label: 0/1) after the step and autoreset
    u64* final_states = nullptr;  // [rows, words + numeric_words]: the state the step reached (before the autoreset)
    i32* schema = nullptr;      // [rows]: the label of the step's action (-1: no move)
    i32* binding = nullptr;     // [rows, label_width]: its objects, -1 past the arity (and for no move)
    u32 label_width = 0;        // columns of `binding` (>= the suite's label_width() when set)
    u8* invalid = nullptr;      // [rows]: an out-of-range index or inapplicable label in a nonterminal, non-stuck row
    u8* goal = nullptr;         // [rows]: the reached state is a goal state
    u32 words = 0;  // atom words of a final_states row (0: the batch's; at least each stepped row's instance width)
};

/// The given actions of a step: [rows] indices into the canonical successor order, as int64 or int32 (at most one of
/// them; neither: the random policy).
struct Actions
{
    const i64* v64 = nullptr;
    const i32* v32 = nullptr;

    [[nodiscard]] bool given() const noexcept { return v64 || v32; }
    [[nodiscard]] i64 at(u64 row) const noexcept { return v64 ? v64[row] : v32[row]; }
};

/// Explicit action labels: schema [rows], binding [rows, stride], in each row's domain and instance.
/// stride is at least the suite's largest schema arity; padding past an action's arity is ignored.
struct Labels
{
    const i32* schema;
    const i32* binding;
    u32 stride;

    Labels(const i32* schema, const i32* binding, u32 stride) : schema(schema), binding(binding), stride(stride) {}
};

/// Throws std::invalid_argument when the configuration or the arrays' shapes do not fit the suite. Reads no array (the
/// arrays may be device memory): HostEnv checks the task ids itself (TaskSuite::check_task_ids).
void check_env(const TaskSuite& suite, const EnvBatch& b, const StepOutputs* out);

/// The host environment step (single-threaded, or split over a ThreadPool: the same results bit for bit).
class HostEnv
{
public:
    HostEnv(TaskSuitePtr suite, const EnvConfig& config);
    HostEnv(const TaskTablePtr& table, const EnvConfig& config) : HostEnv(TaskSuite::of(table), config) {}
    ~HostEnv();
    HostEnv(const HostEnv&) = delete;
    HostEnv& operator=(const HostEnv&) = delete;

    [[nodiscard]] const TaskSuitePtr& suite() const noexcept { return m_suite; }
    [[nodiscard]] const EnvConfig& config() const noexcept { return m_config; }
    void set_seed(u64 seed) noexcept { m_config.seed = seed; }
    /// Row width in atom words (TaskSuite::words()).
    [[nodiscard]] u32 words() const noexcept { return m_suite->words(); }
    [[nodiscard]] u32 numeric_words() const noexcept { return m_suite->numeric_words(); }
    /// Successors of an instance's initial state (Label mode: 0/1, whether an applicable action exists).
    [[nodiscard]] u32 initial_count(u32 instance) const { return m_init_count.at(instance); }
    /// Whether an instance's initial state passes its goal test.
    [[nodiscard]] bool initial_goal(u32 instance) const { return m_init_goal.at(instance) != 0; }

    /// Resets the rows with mask[i] != 0 (every row if mask is null) to the initial states of their instances
    /// (task_ids, which the caller sets first): steps 0, goal masks (if the batch has them and keep_goals is false)
    /// the instances' goals, count (if set) the initial states' successor counts (Label: 0/1), at_goal (if set) their goal tests (kept
    /// goal masks included). Draw counters are kept (set_seed and the caller decide when streams restart).
    void reset(EnvBatch& b, const u8* mask = nullptr, i32* count = nullptr, bool keep_goals = false) const;
    /// One step of every row (see the file comment); `action` [rows] indexes the canonical successor order (none: the
    /// random policy); `next_task_ids` [rows] (optional) names the instance an autoresetting row restarts in.
    void step(EnvBatch& b, const StepOutputs& out, Actions action = {}, const i32* next_task_ids = nullptr,
              ThreadPool* pool = nullptr);
    /// Label-mode step: applies each applicable label directly; an inapplicable label does not move and sets invalid.
    /// A goal or terminal dead end absorbs the step regardless of the label. No random policy is available.
    void step(EnvBatch& b, const StepOutputs& out, Labels labels, const i32* next_task_ids = nullptr,
              ThreadPool* pool = nullptr);

    /// Successors of rows (Label mode: 0/1, whether an applicable action exists).
    void count(const EnvBatch& b, i32* count, ThreadPool* pool = nullptr);
    /// The goal tests of the rows' states (goal[i] for i < b.rows); at_goal is not read.
    void goal(const EnvBatch& b, u8* goal) const;
    /// After the caller wrote states, task ids or goal masks: count (if set) and at_goal (if the batch has it).
    void refresh(EnvBatch& b, i32* count, ThreadPool* pool = nullptr);

    /// Scratch of one thread's row steps (grown as needed).
    struct Scratch
    {
        std::vector<u64> succ;
        std::vector<i32> schema, binding;
        std::vector<ObjectId> label;
        StateBuilder next;
    };
    /// One step of row `row` of b, its outputs at row `out_row` of `out` (the unit step() and the CPU pool run;
    /// the batch must have passed check_env and step()'s argument checks). `random` draws from the row's RNG stream,
    /// otherwise `action` is the given one (outside [0, count): invalid); an autoreset restarts in instance
    /// `next_instance` (< 0: the row's own). Thread-safe for distinct rows (each thread with its own scratch).
    void step_row(EnvBatch& b, u64 row, const StepOutputs& out, u64 out_row, i64 action, bool random,
                  i32 next_instance, Scratch& scratch) const;
    /// Resets row `row` into instance `instance` (steps 0, goal masks unless keep_goals, the task id); returns the
    /// initial state's successor count (Label: 0/1). `dirty`: how many of the row's atom words (and goal mask words) may be nonzero,
    /// the words of the instance it held (the rest is zero padding); the reset writes max(dirty, the instance's words).
    u32 reset_row(EnvBatch& b, u64 row, u32 instance, bool keep_goals, u32 dirty) const;
    /// Successors of one row (Label: 0/1).
    [[nodiscard]] u32 count_row(const EnvBatch& b, u64 row) const;
    [[nodiscard]] bool goal_row(const EnvBatch& b, u64 row) const;
    /// goal_row of a row that reset_row just put into `instance`: its goal masks' test (with masks), else the
    /// instance's initial_goal.
    [[nodiscard]] bool reset_goal(const EnvBatch& b, u64 row, u32 instance) const;

private:
    void check_step(const EnvBatch& b, const StepOutputs& out, Actions action, const i32* next_task_ids) const;
    void step_impl(EnvBatch& b, const StepOutputs& out, Actions action, const Labels* labels,
                   const i32* next_task_ids, ThreadPool* pool);
    void step_row_impl(EnvBatch& b, u64 row, const StepOutputs& out, u64 out_row, i64 action, bool random,
                       i32 next_instance, Scratch& scratch, const Labels* labels) const;

    TaskSuitePtr m_suite;
    EnvConfig m_config;
    std::vector<u32> m_init_count;       // per instance
    std::vector<u8> m_init_goal;         // per instance
    std::vector<const Task*> m_tasks;    // per instance: its task (the rows' lookup without the suite's indirections)
    u32 m_label_width = 1;               // max(1, the suite's label width)
    std::vector<Scratch> m_scratch;      // per pool member
};
}  // namespace mymyr::rl
