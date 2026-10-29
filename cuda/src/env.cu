// The device kernels of the environment step (include/mymyr/cuda/env_kernels.hpp). Every kernel is one thread per
// environment and mirrors the host reference rl::HostEnv::step (src/mymyr/rl/env.cpp) statement by statement, so the
// results agree bit for bit (the rewards are the same float sums in the same order).
// Compiled by nvcc as C++23; includes only the device-code subset.

#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/rl/rng.hpp"
#include "mymyr/rl/visited.hpp"


namespace mymyr::cuda::envk
{
namespace
{
using rl::dev::k_none;

constexpr unsigned k_block = 256;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

__device__ __forceinline__ u64 stride_of(u64 stride, u32 S) { return stride ? stride : S; }

__device__ __forceinline__ u32 row_count(const u32* counts, u32 S, u64 stride, const i32* offsets, u64 i)
{
    if (counts)
    {
        const u32* r = counts + i * stride_of(stride, S);
        u32 c = 0;
        for (u32 k = 0; k < S; ++k)
            c += r[k];
        return c;
    }
    return static_cast<u32>(offsets[i + 1] - offsets[i]);
}

/// Row r's launch-order key (lifted::launch_order's order_key).
__device__ __forceinline__ u32 order_key(const Select& p, u32 r)
{
    const u32 t = p.inst[r];
    if (t >= p.instances)
        return 0;
    const u32 x = p.key[t];
    return x < p.keys ? x : p.keys - 1;
}

__global__ void k_select(Select p)
{
    // the key of the random policy, loaded before (not after) the rows' counts: it does not depend on them
    const u64 key = p.seed_dev ? __ldg(p.seed_dev) : p.seed;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < p.rows; i += u64{gridDim.x} * blockDim.x)
    {
        const u32 c = row_count(p.counts, p.num_schemas, p.count_stride, p.offsets, i);
        u64 a = 0;
        u8 st = k_ok;
        if (p.goal && p.goal[i])
            st = k_goal;
        else if (c == 0)
            st = k_stuck;
        else if (p.action || p.action32)
        {
            const i64 x = p.action ? p.action[i] : p.action32[i];
            if (x < 0 || static_cast<u64>(x) >= c)
                st = k_invalid;
            else
                a = static_cast<u64>(x);
        }
        else
            a = rl::rng::successor_index(p.seeds ? p.seeds[i] : key, p.env_ids ? u64{p.env_ids[i]} : p.first_env + i,
                                         p.draws[i], c);
        if (p.draws)
            ++p.draws[i];
        p.status[i] = st;
        if (p.order_col)
        {
            // the cached row order: a permutation of the rows (else the launches take the rows in batch order)
            const u64 stride = stride_of(p.count_stride, p.num_schemas);
            const u32 r = p.counts[i * stride + p.order_col];
            if (r >= p.rows || p.counts[u64{r} * stride + p.order_col + 1] != i)
                *p.order_bad = 1;
            else if (p.starts)
            {
                // the key starts: position i starts the keys after its predecessor's up to its own (a predecessor
                // outside the rows is reported by its own thread)
                const u32 k = order_key(p, r);
                u32 from = 0;
                if (i > 0)
                {
                    const u32 q = p.counts[(i - 1) * stride + p.order_col];
                    const u32 kq = q < p.rows ? order_key(p, q) : k;
                    if (kq > k)
                        *p.order_bad = 1;
                    from = kq + 1;
                }
                for (u32 x = from; x <= k; ++x)
                    p.starts[x] = static_cast<u32>(i);
                if (i + 1 == p.rows)
                    for (u32 x = k + 1; x <= p.keys; ++x)
                        p.starts[x] = static_cast<u32>(p.rows);
            }
        }
        if (p.counts)
        {
            u32 schema = k_none, rank = 0;
            if (st == k_ok)
            {
                // the segment of index a: segments are in schema order
                u64 base = 0;
                const u32* r = p.counts + i * stride_of(p.count_stride, p.num_schemas);
                for (u32 k = 0; k < p.num_schemas; ++k)
                {
                    const u32 n = r[k];
                    if (a < base + n)
                    {
                        schema = k;
                        rank = static_cast<u32>(a - base);
                        break;
                    }
                    base += n;
                }
            }
            p.pick_schema[i] = schema;
            p.pick_rank[i] = rank;
        }
        else
            p.pick_row[i] = st == k_ok ? static_cast<i64>(p.offsets[i]) + static_cast<i64>(a) : -1;
    }
}

__global__ void k_move(Move p)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < p.rows; i += u64{gridDim.x} * blockDim.x)
    {
        const i64 r = p.pick_row[i];
        if (r < 0)
        {
            p.goal_flag[i] = 0;
            continue;
        }
        const u64 j = static_cast<u64>(r);
        u64* dst = p.states + i * p.row_words;
        const u64* src = p.succ + j * p.row_words;
        for (u32 w = 0; w < p.row_words; ++w)
            dst[w] = src[w];
        p.goal_flag[i] = p.succ_goal[j];
        if (p.schema)
            p.schema[i] = p.succ_schema[j];
        if (p.binding)
            for (u32 k = 0; k < p.label_width; ++k)
                p.binding[i * p.label_width + k] = k < p.flat_width ? p.succ_binding[j * p.flat_width + k] : -1;
    }
}

/// Whether `row` holds goal_pos and none of goal_neg ([words] each).
__device__ __forceinline__ bool masks_hold(const u64* row, const u64* goal_pos, const u64* goal_neg, u32 words)
{
    for (u32 word = 0; word < words; ++word)
        if ((row[word] & goal_pos[word]) != goal_pos[word] || (row[word] & goal_neg[word]) != 0)
            return false;
    return true;
}

/// The instance of row i (inst null: 0).
__device__ __forceinline__ u32 instance_of(const Instances& in, u64 i) { return in.inst ? static_cast<u32>(in.inst[i]) : 0u; }

/// The visited-table key of a row of instance k: State::hash of its atom words and the instance's numeric words.
__device__ __forceinline__ u64 row_key(const u64* row, u32 words, const u32* numeric_words, u32 k)
{
    const u32 nn = numeric_words ? numeric_words[k] : 0u;
    return hash::state(row, words, row + words, nn);
}

/// Inserts `key` into row i's visited table; returns whether it was there. The insert that saturates the count
/// (overflow) sets bit 0 of the row's status word; a saturated row writes no status again, so that the rows of an
/// overflowed batch do not contend for one word on every step.
__device__ __forceinline__ bool visited_step(const VisitedRows& v, u64 i, u64 key)
{
    const i32 before = v.count[i];
    const bool seen = rl::visited_insert(v.table + i * v.capacity, v.capacity, v.count[i], key);
    if (before != static_cast<i32>(v.capacity) && v.count[i] == static_cast<i32>(v.capacity))
        atomicOr(v.status + (v.status_per_row ? i : 0), 1);
    return seen;
}

/// k_finish's marker of a row that does not autoreset.
constexpr u32 k_keep = 0xFFFFFFFFu;

/// The visited tables of the autoresetting rows of a block's round (`next`: the thread's row's next instance, or
/// k_keep) hold their initial keys only: the whole block zeroes the tables, consecutive words to consecutive threads
/// (coalesced: a table is H words), then each row's thread inserts its key. The rows are found from one ballot per
/// warp, so a round with few restarts visits only those. With `restarts` (wide tables) the rows and keys are listed
/// there instead, for k_restart_tables. Called by every thread of the block (k_block threads).
__device__ __forceinline__ void restart_tables(const VisitedRows& v, u64 b, u64 rows, u32 next, const u64* init_hash,
                                               u64* restarts)
{
    if (restarts)
    {
        if (next != k_keep)
        {
            // the table after k_restart_tables: the key alone, at its first probe position
            const u64 slot = atomicAdd(reinterpret_cast<unsigned long long*>(restarts), 1ull);
            restarts[2 + slot] = b + threadIdx.x;
            restarts[2 + rows + slot] = init_hash[next];
            v.count[b + threadIdx.x] = 1;
        }
        return;
    }
    __shared__ u32 restart[k_block / 32];
    const bool mine = next != k_keep;
    const u32 warp = __ballot_sync(0xFFFFFFFFu, mine);
    if ((threadIdx.x & 31u) == 0)
        restart[threadIdx.x >> 5] = warp;
    if (!__syncthreads_or(mine))
        return;
    for (u32 k = 0; k < blockDim.x / 32; ++k)
        for (u32 bits = restart[k]; bits; bits &= bits - 1)
        {
            u64* table = v.table + (b + k * 32 + static_cast<u32>(__ffs(static_cast<int>(bits)) - 1)) * v.capacity;
            for (u32 w = threadIdx.x; w < v.capacity; w += blockDim.x)
                table[w] = 0;
        }
    __syncthreads();
    if (next != k_keep)
    {
        const u64 i = b + threadIdx.x;
        v.count[i] = 0;
        (void)rl::visited_insert(v.table + i * v.capacity, v.capacity, v.count[i], init_hash[next]);
    }
}

__device__ __forceinline__ bool goal_row(rl::dev::TaskView task, const Finish& p, u64 row_id, u32 instance)
{
    const u64* row = p.states + row_id * p.row_words;
    if (p.goal_pos)
        return masks_hold(row, p.goal_pos + row_id * p.words, p.goal_neg + row_id * p.words, p.words);
    if (p.goal_flag)
        return p.goal_flag[row_id] != 0;
    if (p.in.goal_off)
    {
        if (p.in.goal_never[instance])
            return false;
        for (u32 goal_id = p.in.goal_off[instance]; goal_id < p.in.goal_off[instance + 1]; ++goal_id)
        {
            const u64 word = row[p.in.goal_word[goal_id]];
            if ((word & p.in.goal_wpos[goal_id]) != p.in.goal_wpos[goal_id] || (word & p.in.goal_wneg[goal_id]) != 0)
                return false;
        }
        return true;
    }
    return rl::dev::goal_holds(p.in.views ? p.in.views[instance] : task, row, p.words, nullptr, 0);
}

__global__ void k_goal_rows(rl::dev::TaskView task, Finish p, u8* goal)
{
    for (u64 row = u64{blockIdx.x} * blockDim.x + threadIdx.x; row < p.rows; row += u64{gridDim.x} * blockDim.x)
    {
        const u32 instance = instance_of(p.in, row);
        if (instance >= p.in.count && p.error)
            atomicOr(p.error, 4u);
        goal[row] = instance < p.in.count && goal_row(task, p, row, instance) ? 1 : 0;
    }
}

/// Row i := the initial state of instance k (and, on the fast path, its cache: the instance's cache words, of rows
/// `stride` apart, and view words), its goal masks unless keep_goals; returns the initial state's successor count.
__device__ __forceinline__ u32 reset_row(const Instances& in, u32 k, u64 i, u64* states, u32 words, u32 row_words,
                                         u32* cache_counts, u64 stride, u64* views, u64* goal_pos, u64* goal_neg,
                                         bool keep_goals)
{
    u64* row = states + i * row_words;
    const u64* init = in.init_row + u64{k} * row_words;
    for (u32 w = 0; w < row_words; ++w)
        row[w] = init[w];
    if (cache_counts)
    {
        const u32* c = in.init_cache + u64{k} * in.cache_words;
        for (u64 j = 0; j < in.cache_words; ++j)
            cache_counts[i * stride + j] = c[j];
    }
    if (views)
    {
        // the instance's own view words (its kernels read no others)
        const u64* v = in.init_view + u64{k} * in.view_words;
        const u64 vw = in.inst_view_words ? in.inst_view_words[k] : in.view_words;
        for (u64 w = 0; w < vw; ++w)
            views[i * in.view_words + w] = v[w];
    }
    if (goal_pos && !keep_goals)
        for (u32 w = 0; w < words; ++w)
        {
            goal_pos[i * words + w] = in.goal_pos[u64{k} * words + w];
            goal_neg[i * words + w] = in.goal_neg[u64{k} * words + w];
        }
    if (in.inst)
        in.inst[i] = static_cast<i32>(k);
    return in.init_count[k];
}

/// Row i of k_finish; returns the instance that the row's visited table restarts with (an autoreset with visited
/// tables), else k_keep.
__device__ __forceinline__ u32 finish_row(rl::dev::TaskView t, const Finish& p, u64 i)
{
    if (i == 0 && p.order_bad)
        *p.order_bad = 0;  // Select of the next step sets it again if the order is stale
    const u8 st = p.status[i];
    const i32 c2 = static_cast<i32>(row_count(p.counts, p.num_schemas, p.count_stride, p.offsets, i));
    u32 k = instance_of(p.in, i);
    if (k >= p.in.count)
    {
        // a task id outside the table (written by the caller): instance 0's tests, reported
        if (p.error)
            atomicOr(p.error, 4u);
        k = 0;
    }
    // every row's next task id is checked, autoresetting or not (as the host checks them)
    if (p.next_task_ids && p.error &&
        (p.next_task_ids[i] < 0 || static_cast<u32>(p.next_task_ids[i]) >= p.in.count))
        atomicOr(p.error, 4u);
    bool g = st == k_goal;
    if (st == k_ok)
        g = goal_row(t, p, i, k);
    const bool dead = p.dead_end && !g && ((st == k_ok && c2 == 0) || st == k_stuck);
    const bool term = g || (dead && p.dead_end_terminal);
    const bool absorbed = st == k_goal || (st == k_stuck && dead && p.dead_end_terminal);
    const i32 t2 = p.steps ? p.steps[i] + (absorbed ? 0 : 1) : 0;
    const bool trunc = !term && p.max_steps && t2 >= static_cast<i32>(p.max_steps);
    if (p.reward)
        p.reward[i] = absorbed ? 0.0f : p.step_reward + (g ? p.goal_reward : 0.0f) + (dead ? p.dead_end_reward : 0.0f);
    if (p.terminated)
        p.terminated[i] = term ? 1 : 0;
    if (p.truncated)
        p.truncated[i] = trunc ? 1 : 0;
    if (p.goal)
        p.goal[i] = g ? 1 : 0;
    if (p.invalid)
        p.invalid[i] = st == k_invalid ? 1 : 0;
    if (st != k_ok)
    {
        if (p.schema)
            p.schema[i] = -1;
        if (p.binding)
            for (u32 j = 0; j < p.label_width; ++j)
                p.binding[i * p.label_width + j] = -1;
    }
    bool revisited = false;
    // an episode over without autoreset (absorbed, or truncated before) keeps its table
    const bool over =
        !p.autoreset && (absorbed || (p.max_steps && p.steps && p.steps[i] >= static_cast<i32>(p.max_steps)));
    if (p.visited.capacity && !over)
        revisited = visited_step(p.visited, i, row_key(p.states + i * p.row_words, p.words, p.in.numeric_words, k));
    if (p.revisited)
        p.revisited[i] = revisited ? 1 : 0;
    if (p.autoreset && (term || trunc))
    {
        u32 next = k;
        if (p.next_task_ids)
        {
            const i32 x = p.next_task_ids[i];
            if (x >= 0 && static_cast<u32>(x) < p.in.count)
                next = static_cast<u32>(x);
        }
        const u32 n0 = reset_row(p.in, next, i, p.states, p.words, p.row_words, p.cache_counts,
                                 stride_of(p.count_stride, p.num_schemas), p.views, p.goal_pos, p.goal_neg, false);
        if (p.steps)
            p.steps[i] = 0;
        if (p.count)
            p.count[i] = static_cast<i32>(n0);
        if (p.at_goal)
            p.at_goal[i] = p.goal_pos ? (masks_hold(p.states + i * p.row_words, p.goal_pos + i * p.words,
                                                    p.goal_neg + i * p.words, p.words)
                                             ? 1
                                             : 0)
                                      : p.in.init_goal[next];
        return p.visited.capacity ? next : k_keep;
    }
    if (p.steps)
        p.steps[i] = t2;
    if (p.count)
        p.count[i] = c2;
    if (p.at_goal)
        p.at_goal[i] = g ? 1 : 0;
    return k_keep;
}

/// A block's rows per round (a thread each); with final_states, the round's rows are first copied there by the whole
/// block, consecutive words to consecutive threads (coalesced, before any of them autoresets). Launched with k_block
/// threads per block.
__global__ void k_finish(rl::dev::TaskView t, Finish p)
{
    for (u64 b = u64{blockIdx.x} * blockDim.x; b < p.rows; b += u64{gridDim.x} * blockDim.x)
    {
        if (p.final_states)
        {
            const u64 end = (b + blockDim.x < p.rows ? b + blockDim.x : p.rows) * p.row_words;
            for (u64 w = b * p.row_words + threadIdx.x; w < end; w += blockDim.x)
                p.final_states[w] = p.states[w];
            __syncthreads();
        }
        const u64 i = b + threadIdx.x;
        const u32 next = i < p.rows ? finish_row(t, p, i) : k_keep;
        if (p.visited.capacity && p.autoreset)
            restart_tables(p.visited, b, p.rows, next, p.in.init_hash, p.restarts);
    }
}

/// The tables of the rows k_finish listed in `restarts` ([0]: the count, [1]: the blocks done, then `rows` row ids,
/// then their keys) hold their keys only (as rl::visited_clear, then rl::visited_insert): a thread per table word over
/// the whole grid. The last block to finish zeroes both counters for the next step.
__global__ void k_restart_tables(VisitedRows v, u64* restarts, u64 rows)
{
    const u32 shift = static_cast<u32>(__popc(v.capacity - 1));  // capacity is a power of two
    const u64 n = restarts[0] << shift;
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
    {
        const u64 k = j >> shift;
        const u32 w = static_cast<u32>(j) & (v.capacity - 1);
        const u64 key = rl::visited_key(restarts[2 + rows + k]);
        v.table[restarts[2 + k] * v.capacity + w] = w == (static_cast<u32>(key) & (v.capacity - 1)) ? key : 0;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        auto* done = reinterpret_cast<unsigned long long*>(restarts + 1);
        if (atomicAdd(done, 1ull) == gridDim.x - 1)
        {
            restarts[0] = 0;
            restarts[1] = 0;
        }
    }
}

__global__ void k_reset(Reset p)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < p.rows; i += u64{gridDim.x} * blockDim.x)
    {
        if (p.mask && !p.mask[i])
            continue;
        u32 k = instance_of(p.in, i);
        if (k >= p.in.count)
        {
            if (p.error)
                atomicOr(p.error, 4u);
            k = 0;
        }
        const u32 n0 = reset_row(p.in, k, i, p.states, p.words, p.row_words, p.cache_counts, p.count_stride, p.views,
                                 p.goal_pos, p.goal_neg, p.keep_goals != 0);
        if (p.visited.capacity)
        {
            // the table was zeroed by k_visited_zero
            p.visited.count[i] = 0;
            (void)rl::visited_insert(p.visited.table + i * p.visited.capacity, p.visited.capacity, p.visited.count[i],
                                     p.in.init_hash[k]);
        }
        if (p.steps)
            p.steps[i] = 0;
        if (p.count)
            p.count[i] = static_cast<i32>(n0);
        if (p.at_goal)
            p.at_goal[i] = p.goal_pos ? (masks_hold(p.states + i * p.row_words, p.goal_pos + i * p.words,
                                                    p.goal_neg + i * p.words, p.words)
                                             ? 1
                                             : 0)
                                      : p.in.init_goal[k];
    }
}

/// Zeroes the visited tables of the rows with mask[i] != 0 (all without a mask): a thread per table word, so that the
/// writes coalesce.
__global__ void k_visited_zero(VisitedRows v, u64 rows, const u8* mask)
{
    const u32 shift = static_cast<u32>(__popc(v.capacity - 1));  // capacity is a power of two
    const u64 n = rows << shift;
    for (u64 j = u64{blockIdx.x} * blockDim.x + threadIdx.x; j < n; j += u64{gridDim.x} * blockDim.x)
        if (!mask || mask[j >> shift])
            v.table[j] = 0;
}

__global__ void k_clear_visited(ClearVisited p)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < p.rows; i += u64{gridDim.x} * blockDim.x)
    {
        if (p.mask && !p.mask[i])
            continue;
        u32 k = p.inst ? static_cast<u32>(p.inst[i]) : 0u;
        if (k >= p.instances)
        {
            if (p.error)
                atomicOr(p.error, 4u);
            k = 0;
        }
        p.visited.count[i] = 0;
        (void)rl::visited_insert(p.visited.table + i * p.visited.capacity, p.visited.capacity, p.visited.count[i],
                                 row_key(p.states + i * p.row_words, p.words, p.numeric_words, k));
    }
}

__global__ void k_row_counts(const u32* counts, u32 S, u64 stride, const i32* offsets, u64 rows, i32* count)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
        count[i] = static_cast<i32>(row_count(counts, S, stride, offsets, i));
}

__global__ void k_philox(const u32* counters, const u32* keys, u64 n, u32* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
    {
        const u32* c = counters + i * 4;
        const u32* k = keys + i * 2;
        const rl::rng::Block4 r = rl::rng::philox4x32_10(rl::rng::Block4{{c[0], c[1], c[2], c[3]}}, k[0], k[1]);
        for (u32 j = 0; j < 4; ++j)
            out[i * 4 + j] = r.v[j];
    }
}
__global__ void k_random_actions(const u64* draws, const i32* count, u64 rows, u64 seed, const u64* seed_dev,
                                 u64 first_env, u32 max_actions, i64* out)
{
    const u64 key = seed_dev ? __ldg(seed_dev) : seed;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        u32 c = count[i] > 0 ? static_cast<u32>(count[i]) : 0u;
        if (c > max_actions)
            c = max_actions;
        out[i] = c ? i64{rl::rng::successor_index(key, first_env + i, draws[i], c)} : i64{0};
    }
}

__global__ void k_store(u64* dst, u64 value) { *dst = value; }
__global__ void k_label_step(LabelStep p)
{
    u32 bind[rl::dev::k_max_matcher_depth];
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < p.rows; i += u64{gridDim.x} * blockDim.x)
    {
        u32 k = p.inst ? static_cast<u32>(p.inst[i]) : 0u;
        if (k >= p.instances)
            k = 0;  // launch_finish reports the task id
        u8 st = k_goal;
        if (!p.at_goal[i])
        {
            st = p.any[i * p.any_stride] ? k_invalid : k_stuck;
            const i32 sc = p.schema[i];
            const rl::dev::TaskView& t = p.views[k];
            if (st == k_invalid && sc >= 0 && static_cast<u32>(sc) < t.num_schemas)
            {
                const u32 arity = t.schema[u64{static_cast<u32>(sc)} * rl::dev::k_sc_count + rl::dev::k_sc_arity];
                bool ok = arity <= rl::dev::k_max_matcher_depth;
                for (u32 j = 0; j < arity && ok; ++j)
                {
                    bind[j] = static_cast<u32>(p.binding[i * p.binding_stride + j]);
                    ok = bind[j] < t.num_objects;
                }
                u64* row = p.states + i * p.words;
                if (ok && rl::dev::is_applicable(t, static_cast<u32>(sc), bind, row, p.words, nullptr, 0))
                {
                    st = k_ok;
                    if (rl::dev::apply_action(t, static_cast<u32>(sc), bind, row, p.words, row, p.words) !=
                            rl::dev::k_apply_ok &&
                        p.error)
                        atomicOr(p.error, 8u);
                    if (p.out_schema)
                        p.out_schema[i] = sc;
                    if (p.out_binding)
                        for (u32 j = 0; j < p.out_width; ++j)
                            p.out_binding[i * p.out_width + j] =
                                j < p.label_width && j < arity ? static_cast<i32>(bind[j]) : -1;
                }
            }
        }
        if (p.draws)
            ++p.draws[i];
        p.status[i] = st;
    }
}

bool visited_ok(const VisitedRows& v)
{
    return !v.capacity || (v.table && v.count && v.status && v.capacity >= 2 && !(v.capacity & (v.capacity - 1)));
}

}  // namespace

cudaError_t launch_label_step(const LabelStep& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!p.states || !p.views || !p.at_goal || !p.any || !p.schema || (p.binding_stride && !p.binding) || !p.status ||
        (p.out_binding && p.out_width < p.label_width))
        return cudaErrorInvalidValue;
    k_label_step<<<grid_for(p.rows), k_block, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t launch_select(const Select& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!p.status || (!p.counts && !p.offsets) || (!p.action && !p.action32 && !p.draws) || (p.action && p.action32) ||
        (p.counts ? !p.pick_schema || !p.pick_rank : !p.pick_row) || (p.order_col && (!p.counts || !p.order_bad)) ||
        (p.starts && (!p.order_col || !p.inst || !p.keys || (p.instances && !p.key))))
        return cudaErrorInvalidValue;
    k_select<<<grid_for(p.rows), k_block, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t launch_move(const Move& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    k_move<<<grid_for(p.rows), k_block, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t launch_finish(const rl::dev::TaskView& t, const Finish& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!p.states || !p.status || (!p.counts && !p.offsets) || !p.in.init_row || !p.in.init_count ||
        (p.goal_pos && !p.goal_neg) || (p.goal_pos && !p.in.goal_pos) || (p.cache_counts && !p.in.init_cache) ||
        (p.views && !p.in.init_view) || (p.at_goal && !p.in.init_goal) || !visited_ok(p.visited) ||
        (p.visited.capacity && p.autoreset && !p.in.init_hash))
        return cudaErrorInvalidValue;
    const bool wide = p.restarts && p.visited.capacity && p.autoreset;
    Finish f = p;
    if (!wide)
        f.restarts = nullptr;
    k_finish<<<grid_for(p.rows), k_block, 0, s>>>(t, f);
    if (wide)
    {
        // a fixed grid (the count is on the device): enough blocks to fill the memory bandwidth
        constexpr unsigned k_restart_blocks = 64;
        k_restart_tables<<<k_restart_blocks, k_block, 0, s>>>(p.visited, p.restarts, p.rows);
    }
    return cudaGetLastError();
}

cudaError_t launch_goal(const rl::dev::TaskView& task, const Finish& p, u8* goal, cudaStream_t stream)
{
    if (p.rows)
        k_goal_rows<<<grid_for(p.rows), k_block, 0, stream>>>(task, p, goal);
    return cudaGetLastError();
}

cudaError_t launch_reset(const Reset& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!p.states || !p.in.init_row || !p.in.init_count || (p.goal_pos && (!p.goal_neg || !p.in.goal_pos)) ||
        (p.cache_counts && !p.in.init_cache) || (p.views && !p.in.init_view) || (p.at_goal && !p.in.init_goal) ||
        !visited_ok(p.visited) || (p.visited.capacity && !p.in.init_hash))
        return cudaErrorInvalidValue;
    if (p.visited.capacity)
    {
        k_visited_zero<<<grid_for(p.rows * p.visited.capacity), k_block, 0, s>>>(p.visited, p.rows, p.mask);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            return e;
    }
    k_reset<<<grid_for(p.rows), k_block, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t launch_clear_visited(const ClearVisited& p, cudaStream_t s)
{
    if (p.rows == 0 || !p.visited.capacity)
        return cudaSuccess;
    if (!p.states || !visited_ok(p.visited))
        return cudaErrorInvalidValue;
    k_visited_zero<<<grid_for(p.rows * p.visited.capacity), k_block, 0, s>>>(p.visited, p.rows, p.mask);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return e;
    k_clear_visited<<<grid_for(p.rows), k_block, 0, s>>>(p);
    return cudaGetLastError();
}

cudaError_t launch_row_counts(const u32* counts, u32 num_schemas, u64 stride, const i32* offsets, u64 rows, i32* count,
                              cudaStream_t s)
{
    if (rows && !counts && !offsets)
        return cudaErrorInvalidValue;
    if (rows)
        k_row_counts<<<grid_for(rows), k_block, 0, s>>>(counts, num_schemas, stride, offsets, rows, count);
    return cudaGetLastError();
}

cudaError_t launch_philox(const u32* counters, const u32* keys, u64 n, u32* out, cudaStream_t s)
{
    if (n)
        k_philox<<<grid_for(n), k_block, 0, s>>>(counters, keys, n, out);
    return cudaGetLastError();
}

cudaError_t launch_random_actions(const u64* draws, const i32* count, u64 rows, u64 seed, const u64* seed_dev,
                                  u64 first_env, u32 max_actions, i64* out, cudaStream_t s)
{
    if (rows == 0)
        return cudaSuccess;
    if (!draws || !count || !out)
        return cudaErrorInvalidValue;
    k_random_actions<<<grid_for(rows), k_block, 0, s>>>(draws, count, rows, seed, seed_dev, first_env, max_actions,
                                                         out);
    return cudaGetLastError();
}

cudaError_t launch_store(u64* dst, u64 value, cudaStream_t s)
{
    if (!dst)
        return cudaErrorInvalidValue;
    k_store<<<1, 1, 0, s>>>(dst, value);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::envk
