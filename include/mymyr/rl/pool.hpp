#pragma once
// CpuEnvPool: EnvPool-style asynchronous environments on the CPU.
//
//   rl::CpuEnvPool pool(table, config, /*num_envs=*/4096, {.threads = 32});
//   pool.reset(nullptr, 0, task_ids, nullptr, nullptr);         // every env into its instance
//   const u64 t = pool.send(env_ids, n, rl::Actions{actions});  // enqueue steps, return at once
//   rl::PoolBatch out = pool.recv();                            // the oldest send's results (blocks until done)
//
// The pool owns N environments over a TaskSuite or a TaskTable (states, task ids, step and draw counters, optional
// per-env goal masks) and worker threads that step them. A send enqueues a step of some environments (env ids, each at
// most once, none of them in flight); the workers step the queued sends' rows in send order; recv returns the results
// of whole sends in send order (or of one given ticket). An environment's step is rl::HostEnv::step_row, the host env's
// own row step: the same semantics, the same RNG streams (env id = the env's index, draws from the config's seed), so a
// result depends only on the env's actions, never on the thread count, the batch composition or the order of sends.
//
// Thread safety: every member function may be called from any number of threads at once (free-threaded Python calls
// them without holding a lock of its own). Sends from several threads are ordered by the pool's lock; each thread's
// recv(ticket) gets its own sends' results. A send's rows are cut into home ranges, one per worker (at least 128 rows
// each); a worker claims pieces of its own range, then of the others, through the ranges' atomic cursors (the lock is
// taken per send, never per piece), and a receiver waits on the send's own completion counter (std::atomic::wait).
// Batch arrays are not zeroed (PoolArray) and recycle() hands them back for later sends.

#include "mymyr/core/types.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace mymyr::rl
{
/// std::allocator, except that a value-initializing construct() (resize) default-initializes: the pool writes every
/// element of its results itself, so a batch's arrays are not zeroed first.
template<class T>
struct NoInitAllocator : std::allocator<T>
{
    using value_type = T;
    NoInitAllocator() noexcept = default;
    template<class U>
    NoInitAllocator(const NoInitAllocator<U>&) noexcept  // NOLINT: allocator conversion
    {
    }
    template<class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>)
    {
        ::new (static_cast<void*>(p)) U;
    }
    template<class U, class... A>
    void construct(U* p, A&&... a)
    {
        ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
    }
    template<class U>
    friend bool operator==(const NoInitAllocator&, const NoInitAllocator<U>&) noexcept
    {
        return true;
    }
};

/// The arrays of a PoolBatch.
template<class T>
using PoolArray = std::vector<T, NoInitAllocator<T>>;

struct PoolOptions
{
    u32 threads = 0;     // worker threads (0: the hardware's)
    bool goals = false;  // per-env goal masks
};

/// The results of one or more sends (recv), or of a reset: arrays of B rows in the order of their env ids.
struct PoolBatch
{
    u64 rows = 0;
    u32 words = 0, numeric_words = 0, label_width = 0;
    bool goals = false;  // per-env goal masks (goal_pos, goal_neg) are set
    bool step = false;   // step results (reward ... binding) are set: a send's, not a reset's
    PoolArray<u32> env_ids;       // [B]
    PoolArray<u64> states;        // [B, words + numeric_words]: the states after the step (and the autoreset)
    PoolArray<i32> task_ids;      // [B]: their instances
    PoolArray<i32> count;         // [B]: their successor counts
    PoolArray<i32> steps;         // [B]: the episodes' step counts
    PoolArray<u64> goal_pos, goal_neg;  // [B, words] with per-env goals, else empty
    // step results (empty after a reset)
    PoolArray<f32> reward;         // [B]
    PoolArray<u8> terminated, truncated, invalid, goal;  // [B]
    PoolArray<u64> final_states;   // [B, words + numeric_words]: the reached states (before the autoreset)
    PoolArray<i32> schema;         // [B]: the actions' labels (-1: no move)
    PoolArray<i32> binding;        // [B, label_width]

    [[nodiscard]] u32 row_words() const noexcept { return words + numeric_words; }
};

class CpuEnvPool
{
public:
    CpuEnvPool(TaskSuitePtr suite, const EnvConfig& config, u32 num_envs, const PoolOptions& options = {});
    CpuEnvPool(const TaskTablePtr& table, const EnvConfig& config, u32 num_envs, const PoolOptions& options = {})
        : CpuEnvPool(TaskSuite::of(table), config, num_envs, options)
    {
    }
    ~CpuEnvPool();
    CpuEnvPool(const CpuEnvPool&) = delete;
    CpuEnvPool& operator=(const CpuEnvPool&) = delete;

    [[nodiscard]] const TaskSuitePtr& suite() const noexcept;
    [[nodiscard]] const EnvConfig& config() const noexcept;
    [[nodiscard]] u32 num_envs() const noexcept;
    [[nodiscard]] u32 threads() const noexcept;
    [[nodiscard]] bool goals() const noexcept;
    [[nodiscard]] u32 words() const noexcept;
    [[nodiscard]] u32 numeric_words() const noexcept;
    [[nodiscard]] u32 label_width() const noexcept;

    /// Resets envs `env_ids` [n] (null: every env, n ignored) into instances `task_ids` [n] (null: their current
    /// ones) with goal masks `goal_pos` / `goal_neg` [n, words] (null: their instances' goals; per-env goals only);
    /// returns their observations. Draw counters are kept. Throws if one of them has a step in flight.
    PoolBatch reset(const u32* env_ids, u64 n, const i32* task_ids, const u64* goal_pos, const u64* goal_neg);
    /// Enqueues one step of envs `env_ids` [n] (null: every env): `actions` [n] (none: the random policy),
    /// `next_task_ids` [n] (optional: the instances autoresetting envs restart in). Returns the send's ticket.
    /// Throws for an unknown or repeated env id, or one with a step in flight.
    u64 send(const u32* env_ids, u64 n, Actions actions, const i32* next_task_ids);
    /// The results of the oldest pending sends, in send order: at least `min_rows` rows (0: exactly one send),
    /// whole sends only. Blocks until they are done. Throws if fewer rows are pending, or rethrows a step's error.
    PoolBatch recv(u64 min_rows = 0);
    /// The results of the send with this ticket (blocks until done).
    PoolBatch recv_ticket(u64 ticket);
    /// send + recv_ticket.
    PoolBatch step(const u32* env_ids, u64 n, Actions actions, const i32* next_task_ids);
    /// Hands a batch back once its arrays are no longer needed: later calls fill their results into its arrays instead
    /// of allocating fresh ones (optional; it saves the allocations and the page faults of new arrays). Thread-safe.
    void recycle(PoolBatch&& batch);
    /// Sends not yet received.
    [[nodiscard]] u64 pending() const;

    struct Impl;

private:
    std::unique_ptr<Impl> m_impl;
};
}  // namespace mymyr::rl
