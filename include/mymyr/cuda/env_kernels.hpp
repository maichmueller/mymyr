#pragma once
// The device kernels of the environment step (cuda/env.hpp drives them): destination-passing
// launchers over POD parameters (caller-owned buffers, one stream, no allocation, no synchronization). The step's
// successor generation is lifted.hpp's (launch_view, launch_count, launch_pick, or the flat expand of DeviceExpander);
// these kernels choose, move, and finish a step (rewards, termination, truncation, autoreset), with the semantics of
// the host reference rl::HostEnv (rl/env.hpp), bit for bit.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20; POD, raw pointers and cudaStream_t.

#include "mymyr/rl/task_arrays_view.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda::envk
{
/// Status of an environment's move (the host reference's statuses).
enum Status : u8
{
    k_ok = 0,
    k_stuck = 1,    // the state has no successors
    k_invalid = 2,  // the given action is outside [0, count)
};

/// Choose every environment's successor: an index into its canonical successor order (the action, or the counter-based
/// draw of rl/rng.hpp), and advance the draw counters.
struct Select
{
    u64 rows = 0;
    // the successor counts: per (row, schema) from the fast path's cache, or CSR offsets of a flat expansion
    const u32* counts = nullptr;  // [rows, count_stride]: the first num_schemas columns (lifted::CountCache)
    u32 num_schemas = 0;
    u64 count_stride = 0;          // 0: num_schemas
    const i32* offsets = nullptr;  // [rows + 1] (when counts is null)
    const i64* action = nullptr;   // [rows] or null (random policy)
    const i32* action32 = nullptr;  // [rows] int32 actions (instead of `action`)
    u64* draws = nullptr;          // [rows] (advanced by one; may be null with actions)
    u64 seed = 0;
    const u64* seed_dev = nullptr;  // the key in device memory instead of `seed` (captured steps follow set_seed)
    u64 first_env = 0;
    const u64* seeds = nullptr;    // [rows] per-row seeds (null: seed)
    const u32* env_ids = nullptr;  // [rows] per-row env ids (null: first_env + row)
    // outputs
    u32* pick_schema = nullptr;  // [rows] (cache): the schema of the pick, k_none for no move
    u32* pick_rank = nullptr;    // [rows] (cache): the index in the (row, schema) segment
    i64* pick_row = nullptr;     // [rows] (flat): the flat row of the pick, -1 for no move
    u8* status = nullptr;        // [rows]
    // multi-instance tables: the cached row order (columns order_col, order_col + 1 of the count cache: the row at
    // position i, the position of row i); *order_bad is set when it is not a permutation of the rows (a batch
    // rearranged since the order was computed: the launches then take the rows in batch order)
    u32 order_col = 0;           // 0: no order
    u32* order_bad = nullptr;
    // and (starts not null) the order's key starts for range launches (lifted::Multi::starts): starts[x] = the first
    // position of key x's rows, starts[keys] = rows, with row r's key key[inst[r]] as lifted::launch_order's (0 for a
    // task id outside [0, instances), at most keys - 1); *order_bad is also set when the order is not sorted by key
    const u32* inst = nullptr;  // [rows] the rows' task ids
    const u32* key = nullptr;   // [instances]
    u32 instances = 0, keys = 0;
    u32* starts = nullptr;      // [keys + 1]
};
cudaError_t launch_select(const Select& p, cudaStream_t s);

/// The flat path's move: states[i] = succ[pick_row[i]] (in place), the goal flag and the label of the successor row.
struct Move
{
    u64 rows = 0;
    u32 row_words = 0;
    u64* states = nullptr;         // [rows, row_words]
    const i64* pick_row = nullptr;  // [rows]
    const u64* succ = nullptr;     // [M, row_words]
    const u8* succ_goal = nullptr;  // [M]
    const i32* succ_schema = nullptr;   // [M] (may be null without label outputs)
    const i32* succ_binding = nullptr;  // [M, flat_width]
    u32 flat_width = 0;
    u8* goal_flag = nullptr;  // [rows] out
    i32* schema = nullptr;    // [rows] out (null: not written)
    i32* binding = nullptr;   // [rows, label_width] out
    u32 label_width = 0;
};
cudaError_t launch_move(const Move& p, cudaStream_t s);

/// The instances of a batch over a task table and their initial rows, counts and caches, as the environment
/// kernels read them: row i is of instance inst[i] (inst null: instance 0). Per-instance arrays have one row per
/// instance, `init_row` [I, row_words], `init_count` [I], `init_cache` [I, cache_words], `init_view` [I, view_words],
/// `goal_pos` / `goal_neg` [I, words] (the instances' goals as masks, for per-env goals).
struct Instances
{
    u32 count = 1;                 // I
    i32* inst = nullptr;           // [rows] the batch's task ids (written by autoresets into next_task_ids)
    const rl::dev::TaskView* views = nullptr;  // [I] device views (goal tests of several instances; null: the kernel's t)
    const u64* init_row = nullptr;
    const u32* init_count = nullptr;
    const u32* init_cache = nullptr;
    u64 cache_words = 0;           // the count cache's words that an autoreset copies (not the row-order columns)
    const u64* init_view = nullptr;
    u64 view_words = 0;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    // the fast path: per instance its view words (an autoreset copies the instance's own view) and, without derived
    // goal literals, its goal as the words of its masks that are not zero (CSR: instance k's are [goal_off[k],
    // goal_off[k + 1]); word goal_word[j] must hold goal_wpos[j] and none of goal_wneg[j]) and whether it is statically
    // false; the goal test is then this mask test (goal_off null: goal_holds of the instance)
    const u64* inst_view_words = nullptr;  // [I]
    const u32* goal_off = nullptr;         // [I + 1]
    const u32* goal_word = nullptr;
    const u64* goal_wpos = nullptr;
    const u64* goal_wneg = nullptr;
    const u8* goal_never = nullptr;  // [I]
};

/// The end of a step: goal test (or given flags, or per-env goal masks), dead ends, rewards, termination, truncation,
/// step counters, final states, the labels of non-moves, counts, and the autoreset (the initial row of the row's
/// instance, or of next_task_ids[i], and, on the fast path, its cache).
struct Finish
{
    u64 rows = 0;
    u32 words = 0;      // atom words per row
    u32 row_words = 0;  // words + numeric words
    u64* states = nullptr;
    i32* steps = nullptr;
    const u8* status = nullptr;
    // successor counts of the reached states: the fast path's cache (sums over the schemas) or CSR offsets
    const u32* counts = nullptr;  // [rows, count_stride]
    u32 num_schemas = 0;
    u64 count_stride = 0;  // 0: num_schemas
    const i32* offsets = nullptr;
    // goal flags of the reached states: per-env goal masks (goal_pos / goal_neg, rl::EnvBatch), else given (flat
    // path), else tested with goal_holds of the row's instance (no derived goal literals)
    const u8* goal_flag = nullptr;
    u64* goal_pos = nullptr;  // [rows, words]
    u64* goal_neg = nullptr;
    f32 step_reward = 0, goal_reward = 0, dead_end_reward = 0;
    u32 max_steps = 0;
    u32 autoreset = 0;
    u32 dead_end = 1;           // 1: a non-goal state without successors (and a stuck one) is a dead end; 0: none
    u32 dead_end_terminal = 1;  // a dead end terminates
    Instances in;
    const i32* next_task_ids = nullptr;  // [rows]: the instance an autoresetting row restarts in (null: its own)
    u32* error = nullptr;                // value 4: a task id outside the table (a next id: the row kept its
                                         // instance; the row's own: tested as instance 0)
    u32* order_bad = nullptr;            // reset to 0 for the next step (Select sets it)
    // outputs (null: not written)
    f32* reward = nullptr;
    u8* terminated = nullptr;
    u8* truncated = nullptr;
    u8* goal = nullptr;
    u8* invalid = nullptr;
    i32* count = nullptr;
    u64* final_states = nullptr;
    i32* schema = nullptr;   // written for non-moves only (-1)
    i32* binding = nullptr;  // idem
    u32 label_width = 0;
    // on the fast path, the rows' cache (an autoreset copies its instance's initial cache)
    u32* cache_counts = nullptr;
    u64* views = nullptr;
};
cudaError_t launch_finish(const rl::dev::TaskView& t, const Finish& p, cudaStream_t s);

/// Resets rows with mask[i] != 0 (all without a mask) to the initial state of their instance: the initial row, step 0,
/// the initial count and cache, and (unless keep_goals) the instance's goal masks.
struct Reset
{
    u64 rows = 0;
    u32 words = 0;
    u32 row_words = 0;
    u64* states = nullptr;
    i32* steps = nullptr;
    const u8* mask = nullptr;
    i32* count = nullptr;
    Instances in;
    u32* cache_counts = nullptr;
    u64 count_stride = 0;
    u64* views = nullptr;
    u64* goal_pos = nullptr;  // [rows, words] (null: no per-env goals)
    u64* goal_neg = nullptr;
    u32 keep_goals = 0;
    u32* error = nullptr;  // value 4: a task id outside the table (the row is reset into instance 0)
};
cudaError_t launch_reset(const Reset& p, cudaStream_t s);

/// count[i] = the sum of counts[i, :num_schemas] (the fast path's cache, rows `stride` words apart; 0: num_schemas), or
/// offsets[i + 1] - offsets[i] when counts is null.
cudaError_t launch_row_counts(const u32* counts, u32 num_schemas, u64 stride, const i32* offsets, u64 rows, i32* count,
                              cudaStream_t s);

/// *dst = value, stream-ordered (the value is the launch's parameter: a later host change does not reach it).
cudaError_t launch_store(u64* dst, u64 value, cudaStream_t s);

/// The Philox-4x32-10 block of each counter (tests: cuRAND and the host agree with rl/rng.hpp).
cudaError_t launch_philox(const u32* counters, const u32* keys, u64 n, u32* out, u32 curand, cudaStream_t s);

/// The random policy's next choices without stepping (TorchRL's rand_action): out[i] = rl::rng::successor_index(
/// seed, first_env + i, draws[i], c) with c = count[i], limited to max_actions (0: no limit); 0 where c is 0 (the key
/// is *seed_dev where given). The environment's own random step at the same draw counter takes the same successor
/// (when count <= max_actions).
cudaError_t launch_random_actions(const u64* draws, const i32* count, u64 rows, u64 seed, const u64* seed_dev,
                                  u64 first_env, u32 max_actions, i64* out, cudaStream_t s);
}  // namespace mymyr::cuda::envk
