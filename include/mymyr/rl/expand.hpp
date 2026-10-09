#pragma once
// Batched expand: the RL primitive.
//
//   states[N, W] (+ task_ids[N]) -> flat CSR: successor words [M, W'], parent [M], labels (schema [M],
//                                   binding [M, L]), offsets [N+1]
//
//   - A batch is over a TaskTable (rl/task_table.hpp) or a TaskSuite of several domains (rl/task_suite.hpp):
//     row i is a state of instance task_ids[i] (task_ids may be null only over a table of one instance, the table a
//     single Task makes; over a suite they are global ids). Rows are the table's or suite's width (unused words zero);
//     the successors of row i are states of the same instance, labels carry its domain's schema ids and its
//     instance-local object ids.
//   - Destination-passing: the caller owns every output buffer and passes their capacity. expand() allocates
//     nothing (the multi-threaded variant grows its caller-owned scratch, amortized) and synchronizes with nothing
//     but its own pool. Rows beyond the capacity are counted, not written: `total` > `capacity` is the overflow.
//   - Canonical order (on by default): per state, successors by schema, then lexicographically by binding (the
//     schema's full parameter list). An action is its label (schema, binding), never a slot index.
//   - Witness pruning is off by default: every applicable ground action is a transition.
//   - State words: little-endian u64, fluent slot i at word i >> 6, bit i & 63; missing words are zero.
//     Under lazy slots a successor may need more words than the output rows have (new atoms): such rows are not
//     written, and `words_needed` reports the width that fits all of them (words_needed > words is an overflow too).
//   - The padded [N, K] view (pad()) is built from the flat CSR by a scatter, with a validity mask and true counts.
//   - Numeric tasks: a row is [bits | slots], the `words` atom words followed by the table's numeric words
//     (`numeric_words` = table.numeric_words(); instance i uses the first numeric_words of its own, as in
//     task/numeric.hpp: two int32 per word or one double per word, and the rest stay zero). Inputs, flat and padded
//     outputs carry them; `words` and `words_needed` count atom words only. Classical tables have numeric_words == 0
//     and rows are unchanged. The mask goal tests below see the atom words only (numeric goal constraints need
//     is_goal()).
//
// Thread safety: expand() uses the calling thread's workspaces of the instances, so any number of threads may expand
// batches of one shared table at once. The pool variant splits one batch over a ThreadPool.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <memory>

namespace mymyr::rl
{
/// N rows of state words, `stride` words apart (0: rows are contiguous, stride = words + numeric_words).
struct StateBatchView
{
    const u64* data = nullptr;
    u64 rows = 0;
    u32 words = 0;
    u64 stride = 0;
    u32 numeric_words = 0;  // numeric tables: the numeric words after the atom words of each row

    [[nodiscard]] const u64* row(u64 i) const noexcept { return data + i * (stride ? stride : words + numeric_words); }
};

struct ExpandOptions
{
    bool canonical_order = true;
    bool witness_pruning = false;  // on: one witness per effect-relevant binding (fewer, unlabelled-equivalent rows)
    // reject rows with bits beyond their instance's assigned atom slots (rows narrower than a frozen instance's
    // width are rejected either way)
    bool validate = true;
};

/// States of one batch, and successor rows (capacity, total) of one expansion, at most: parents and offsets are i32.
inline constexpr u64 k_max_rows = (u64{1} << 31) - 1;

/// Destination buffers and results of a flat expansion. A null array is not written.
struct Expansion
{
    u64 capacity = 0;        // rows of succ / parent / schema / binding / goal (at most k_max_rows)
    u32 words = 0;           // atom words per successor row
    u32 label_width = 0;     // columns of `binding` (at least the table's label_width() when binding is set)
    u64* succ = nullptr;     // [capacity, words + numeric_words]
    i32* parent = nullptr;   // [capacity]: index of the expanded state in the batch (its instance: task_ids[parent])
    i32* schema = nullptr;   // [capacity]
    i32* binding = nullptr;  // [capacity, label_width]: objects, then -1 past the schema's arity
    u8* goal = nullptr;      // [capacity]: 1 iff the successor is a goal state
    i32* offsets = nullptr;  // [rows + 1]: successors of state i are [offsets[i], offsets[i+1]) (true counts)

    u64 total = 0;         // successors of the whole batch (may exceed capacity)
    u32 words_needed = 0;  // widest successor (trimmed); > words means some rows did not fit
    u32 numeric_words = 0;  // numeric tables: table.numeric_words() (a row is words + numeric_words wide)
    [[nodiscard]] bool overflow() const noexcept { return total > capacity || words_needed > words; }
};

/// Scratch of the pool variant: per-member row buffers, reused across calls (grown as needed).
class ExpandScratch
{
public:
    ExpandScratch();
    ~ExpandScratch();
    ExpandScratch(const ExpandScratch&) = delete;
    ExpandScratch& operator=(const ExpandScratch&) = delete;

    struct Impl;
    [[nodiscard]] Impl& impl() noexcept { return *m_impl; }

private:
    std::unique_ptr<Impl> m_impl;
};

/// Largest schema arity of a task: the minimum label width.
[[nodiscard]] u32 max_label_width(const Task& task) noexcept;

/// Expands every state of `in` on the calling thread; row i is a state of instance task_ids[i] (task_ids null:
/// instance 0). Throws std::invalid_argument for malformed buffers, states or task ids.
void expand(const TaskSuite& suite, StateBatchView in, const i32* task_ids, Expansion& out,
            const ExpandOptions& options = {});
/// The same on every member of `pool`. Same results as the single-threaded call, bit for bit.
void expand(const TaskSuite& suite, StateBatchView in, const i32* task_ids, Expansion& out,
            const ExpandOptions& options, ThreadPool& pool, ExpandScratch& scratch);
/// Over a table: its one-domain suite.
inline void expand(const TaskTable& table, StateBatchView in, const i32* task_ids, Expansion& out,
                   const ExpandOptions& options = {})
{
    expand(*TaskSuite::of(table), in, task_ids, out, options);
}
inline void expand(const TaskTable& table, StateBatchView in, const i32* task_ids, Expansion& out,
                   const ExpandOptions& options, ThreadPool& pool, ExpandScratch& scratch)
{
    expand(*TaskSuite::of(table), in, task_ids, out, options, pool, scratch);
}

/// The padded [rows, K] view of a flat expansion.
struct PaddedExpansion
{
    u32 K = 0;
    i32* index = nullptr;    // [rows, K]: flat row of the k-th successor, -1 = none
    u8* mask = nullptr;      // [rows, K]: 1 = valid
    i32* count = nullptr;    // [rows]: true successor counts (may exceed K)
    u32 words = 0;           // atom words per successor row of `succ`
    u64* succ = nullptr;     // [rows, K, words + numeric_words], zero padding
    i32* schema = nullptr;   // [rows, K], -1 padding
    u32 label_width = 0;     // columns of `binding`
    i32* binding = nullptr;  // [rows, K, label_width], -1 padding
    u8* goal = nullptr;      // [rows, K], 0 padding
    bool overflow = false;   // some count > K, or some successor was beyond the flat capacity
    u32 numeric_words = 0;   // numeric tasks: must equal the flat expansion's
};

/// Scatters a flat expansion of `rows` states (it must carry offsets) into `out`. Arrays of `out` that are null, or
/// whose flat source is null, are not written.
void pad(const Expansion& flat, u64 rows, PaddedExpansion& out);

/// Goal flags of a batch (evaluates axioms when the goal mentions derived predicates): out[i] = the goal test of
/// instance task_ids[i] (null: a table of one instance) on row i. Throws std::invalid_argument for malformed buffers,
/// states or task ids (as expand with validate on).
void is_goal(const TaskSuite& suite, StateBatchView in, const i32* task_ids, u8* out);
inline void is_goal(const TaskTable& table, StateBatchView in, const i32* task_ids, u8* out)
{
    is_goal(*TaskSuite::of(table), in, task_ids, out);
}

/// Pure mask goal test: out[i] = (s & gpos) == gpos && (s & gneg) == 0 over the words of row i.
/// gpos / gneg hold one row (shared by every state) or in.rows rows (one goal per env); rows may differ in width.
void goal_test(StateBatchView in, StateBatchView gpos, StateBatchView gneg, u8* out);
/// Unsatisfied goal literals per row: popcount(gpos & ~s) + popcount(gneg & s) (goal-count shaping).
void goal_count(StateBatchView in, StateBatchView gpos, StateBatchView gneg, i32* out);

/// Native random walks (the python_ft `native_walks` reference loop): from the initial state, expand, move to a
/// uniformly random successor; restart after `episode` steps or at a dead end.
struct WalkStats
{
    u64 steps = 0;       // expansions
    u64 successors = 0;  // successors generated
    u64 dead_ends = 0;
    u64 goals = 0;       // goal states visited
};
WalkStats random_walks(const Task& task, u64 steps, u64 episode, u64 seed, const ExpandOptions& options = {});
}  // namespace mymyr::rl
