// The device kernels of the environment step (include/mymyr/cuda/env_kernels.hpp). Every kernel is one thread per
// environment and mirrors the host reference rl::HostEnv::step (src/mymyr/rl/env.cpp) statement by statement, so the
// results agree bit for bit (the rewards are the same float sums in the same order).
// Compiled by nvcc as C++20; includes only the device-code subset.

#include "mymyr/cuda/env_kernels.hpp"
#include "mymyr/rl/rng.hpp"

#include <curand_philox4x32_x.h>

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
        if (c == 0)
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

/// The instance of row i (inst null: 0).
__device__ __forceinline__ u32 instance_of(const Instances& in, u64 i) { return in.inst ? static_cast<u32>(in.inst[i]) : 0u; }

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

/// A block's rows per round (a thread each); with final_states, the round's rows are first copied there by the whole
/// block, consecutive words to consecutive threads (coalesced, before any of them autoresets).
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
        if (i >= p.rows)
            continue;
        if (i == 0 && p.order_bad)
            *p.order_bad = 0;  // Select of the next step sets it again if the order is stale
        const u8 st = p.status[i];
        const i32 c2 = static_cast<i32>(row_count(p.counts, p.num_schemas, p.count_stride, p.offsets, i));
        u64* row = p.states + i * p.row_words;
        u32 k = instance_of(p.in, i);
        if (k >= p.in.count)
        {
            // a task id outside the table (written by the caller): instance 0's tests, reported
            if (p.error)
                atomicOr(p.error, 4u);
            k = 0;
        }
        bool g = false;
        if (st == k_ok)
        {
            if (p.goal_pos)
            {
                g = true;
                for (u32 w = 0; w < p.words && g; ++w)
                {
                    const u64 gp = p.goal_pos[i * p.words + w], gn = p.goal_neg[i * p.words + w];
                    g = (row[w] & gp) == gp && (row[w] & gn) == 0;
                }
            }
            else if (p.goal_flag)
                g = p.goal_flag[i] != 0;
            else if (p.in.goal_off)
            {
                // the instance's goal masks, over their words that are not zero
                g = p.in.goal_never[k] == 0;
                for (u32 j = p.in.goal_off[k]; j < p.in.goal_off[k + 1] && g; ++j)
                {
                    const u64 x = row[p.in.goal_word[j]];
                    g = (x & p.in.goal_wpos[j]) == p.in.goal_wpos[j] && (x & p.in.goal_wneg[j]) == 0;
                }
            }
            else
                g = rl::dev::goal_holds(p.in.views ? p.in.views[k] : t, row, p.words, nullptr, 0);
        }
        const bool dead = p.dead_end && ((st == k_ok && !g && c2 == 0) || st == k_stuck);
        const bool term = g || (dead && p.dead_end_terminal);
        const i32 t2 = p.steps ? p.steps[i] + 1 : 0;
        const bool trunc = !term && p.max_steps && t2 >= static_cast<i32>(p.max_steps);
        if (p.reward)
            p.reward[i] = p.step_reward + (g ? p.goal_reward : 0.0f) + (dead ? p.dead_end_reward : 0.0f);
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
        if (p.autoreset && (term || trunc))
        {
            u32 next = k;
            if (p.next_task_ids)
            {
                const i32 x = p.next_task_ids[i];
                if (x >= 0 && static_cast<u32>(x) < p.in.count)
                    next = static_cast<u32>(x);
                else if (p.error)
                    atomicOr(p.error, 4u);
            }
            const u32 n0 = reset_row(p.in, next, i, p.states, p.words, p.row_words, p.cache_counts,
                                     stride_of(p.count_stride, p.num_schemas), p.views, p.goal_pos, p.goal_neg, false);
            if (p.steps)
                p.steps[i] = 0;
            if (p.count)
                p.count[i] = static_cast<i32>(n0);
        }
        else
        {
            if (p.steps)
                p.steps[i] = t2;
            if (p.count)
                p.count[i] = c2;
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
        if (p.steps)
            p.steps[i] = 0;
        if (p.count)
            p.count[i] = static_cast<i32>(n0);
    }
}

__global__ void k_row_counts(const u32* counts, u32 S, u64 stride, const i32* offsets, u64 rows, i32* count)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
        count[i] = static_cast<i32>(row_count(counts, S, stride, offsets, i));
}

__global__ void k_philox(const u32* counters, const u32* keys, u64 n, u32* out, u32 curand)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
    {
        const u32* c = counters + i * 4;
        const u32* k = keys + i * 2;
        if (curand)
        {
            const uint4 r = curand_Philox4x32_10(make_uint4(c[0], c[1], c[2], c[3]), make_uint2(k[0], k[1]));
            out[i * 4 + 0] = r.x;
            out[i * 4 + 1] = r.y;
            out[i * 4 + 2] = r.z;
            out[i * 4 + 3] = r.w;
        }
        else
        {
            const rl::rng::Block4 r = rl::rng::philox4x32_10(rl::rng::Block4{{c[0], c[1], c[2], c[3]}}, k[0], k[1]);
            for (u32 j = 0; j < 4; ++j)
                out[i * 4 + j] = r.v[j];
        }
    }
}
__global__ void k_random_actions(const u64* draws, const i32* count, u64 rows, u64 seed, const u64* seed_dev,
                                 u64 first_env, u32 max_actions, i64* out)
{
    const u64 key = seed_dev ? __ldg(seed_dev) : seed;
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < rows; i += u64{gridDim.x} * blockDim.x)
    {
        u32 c = count[i] > 0 ? static_cast<u32>(count[i]) : 0u;
        if (max_actions && c > max_actions)
            c = max_actions;
        out[i] = c ? i64{rl::rng::successor_index(key, first_env + i, draws[i], c)} : i64{0};
    }
}

__global__ void k_store(u64* dst, u64 value) { *dst = value; }
}  // namespace

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
        (p.views && !p.in.init_view))
        return cudaErrorInvalidValue;
    k_finish<<<grid_for(p.rows), k_block, 0, s>>>(t, p);
    return cudaGetLastError();
}

cudaError_t launch_reset(const Reset& p, cudaStream_t s)
{
    if (p.rows == 0)
        return cudaSuccess;
    if (!p.states || !p.in.init_row || !p.in.init_count || (p.goal_pos && (!p.goal_neg || !p.in.goal_pos)) ||
        (p.cache_counts && !p.in.init_cache) || (p.views && !p.in.init_view))
        return cudaErrorInvalidValue;
    k_reset<<<grid_for(p.rows), k_block, 0, s>>>(p);
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

cudaError_t launch_philox(const u32* counters, const u32* keys, u64 n, u32* out, u32 curand, cudaStream_t s)
{
    if (n)
        k_philox<<<grid_for(n), k_block, 0, s>>>(counters, keys, n, out, curand);
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
