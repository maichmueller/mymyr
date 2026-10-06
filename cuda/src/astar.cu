// The kernels of the device best-first searches (include/mymyr/cuda/astar_kernels.hpp). Compiled by nvcc as C++20;
// includes only the device-code subset (and CUB / Thrust).

#include "mymyr/cuda/astar_kernels.hpp"

#include "grid_stride.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace mymyr::cuda::bfk
{
namespace
{
constexpr unsigned k_block = 256;
constexpr unsigned k_max_blocks = 256;  // grid-stride loops beyond, spread (the kernels of a captured step are sized by capacity)
constexpr u64 k_tag = 0xFFFFFFFF00000000ull;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < k_max_blocks ? (g ? g : 1) : k_max_blocks);
}

#define MYMYR_GRID_LOOP(i, n) for (u64 i = spread_first(); i < (n); i += u64{gridDim.x} * blockDim.x)  // (grid_stride.cuh)

__device__ __forceinline__ u8 status_of(u8 f) { return f & 3u; }

struct MinU32
{
    __device__ u32 operator()(u32 a, u32 b) const { return a < b ? a : b; }
};
__device__ __forceinline__ u64 min_u64(u64 a, u64 b) { return a < b ? a : b; }

/// Whether run key `key` belongs to the step's pop (the runs of the step's keys form a prefix of the list from head).
__device__ __forceinline__ bool eligible(const Pop& p, const Ctl& c, u64 key)
{
    if (p.mode == k_gbfs)
        return true;
    if (c.goal_pop || p.single_bucket)
        return key == c.key0;
    return (key >> 32) == (c.key0 >> 32) && (key & 0xFFFFFFFFu) != 0;
}

/// Arena index of entry i of the round's window (binary search over the window's run prefix).
__device__ __forceinline__ u64 entry_index(const Pop& p, u32 wruns, u32 i)
{
    u32 lo = 0, hi = wruns;  // wprefix[lo] <= i < wprefix[hi]
    while (hi - lo > 1)
    {
        const u32 mid = (lo + hi) / 2;
        if (p.wprefix[mid] <= i)
            lo = mid;
        else
            hi = mid;
    }
    return p.wbegin[lo] + (i - p.wprefix[lo]);
}

// ------------------------------------------------------------------------------------------------------------- pop

__global__ void k_pop_plan(Pop p, Ctl* ctl)
{
    using Scan = cub::BlockScan<u32, k_block>;
    using Reduce = cub::BlockReduce<u32, k_block>;
    __shared__ typename Scan::TempStorage scan_tmp;
    __shared__ typename Reduce::TempStorage red_tmp;
    __shared__ u32 s_end, s_acc, s_wruns, s_more, s_stop;
    Ctl& c = *ctl;
    if (threadIdx.x == 0)
    {
        s_stop = 0;
        c.window = 0;
        c.wruns = 0;
        c.more = 0;
        c.cut = k_none;
        c.pruned_round = 0;
        if (c.status != k_running || c.abort)
            s_stop = 1;
        else if (c.round == 0)
        {
            if (c.head == c.runs)
            {
                c.status = k_exhausted;
                s_stop = 1;
            }
            else
            {
                c.key0 = p.runs.key[c.head];
                c.goal_pop = p.mode == k_astar && (c.key0 & 0xFFFFFFFFu) == 0 ? 1u : 0u;
                c.n = 0;
                if (!c.goal_pop && c.expanded >= p.max_expanded)
                {
                    c.status = k_out_of_expanded;
                    s_stop = 1;
                }
                else
                    c.limit = c.goal_pop ? k_none : static_cast<u32>(min_u64(p.batch, p.max_expanded - c.expanded));
            }
        }
        if (!s_stop)
            ++c.round;
        s_acc = 0;
        s_wruns = 0;
        s_more = 0;
    }
    __syncthreads();
    if (s_stop)
        return;
    // the window: the eligible runs from head (a prefix of the list), up to window_cap entries
    const u32 head = c.head, runs = c.runs;
    for (u32 t0 = head;; t0 += k_block)
    {
        const u32 r = t0 + threadIdx.x;
        const bool in = r < runs && eligible(p, c, p.runs.key[r]);
        const u32 first_out = Reduce(red_tmp).Reduce(in ? 0xFFFFFFFFu : r, MinU32{});
        if (threadIdx.x == 0)
            s_end = first_out;  // the first run past the eligible ones in this tile (or none: 0xFFFFFFFF)
        __syncthreads();
        const u32 end = s_end;
        const u32 acc = s_acc;
        const u32 cnt = r < end && r < runs ? p.runs.count[r] : 0;
        u32 excl = 0, total = 0;
        Scan(scan_tmp).ExclusiveSum(cnt, excl, total);
        const u64 at = u64{acc} + excl;  // entries of the eligible runs before run r
        if (cnt && at < p.window_cap)
        {
            p.wprefix[r - head] = static_cast<u32>(at);
            p.wbegin[r - head] = p.runs.begin[r];
            atomicAdd(&s_wruns, 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const u64 next = u64{acc} + total;
            if (next > p.window_cap)
                s_more = 1;
            s_acc = static_cast<u32>(min_u64(next, p.window_cap));
        }
        __syncthreads();
        const bool tile_ended = s_end != 0xFFFFFFFFu || t0 + k_block >= runs;
        if (s_acc >= p.window_cap)
        {
            if (threadIdx.x == 0 && !tile_ended)
                s_more = 1;  // (the next tile may hold eligible runs: counted as more)
            break;
        }
        if (tile_ended)
            break;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        c.window = s_acc;
        c.wruns = s_wruns;
        c.more = s_more;
        p.wprefix[s_wruns] = s_acc;
    }
}

__global__ void k_pop_flags(Pop p, Nodes n, const Ctl* ctl, u8* flags)
{
    const u32 window = ctl->window, wruns = ctl->wruns;
    const u32 max_depth = ctl->goal_pop ? 0xFFFFFFFFu : p.max_depth;
    MYMYR_GRID_LOOP(i, u64{p.window_cap} + 1)
    {
        if (i >= window)
        {
            flags[i] = 0;
            continue;
        }
        const u64 e = p.entries[entry_index(p, wruns, static_cast<u32>(i))];
        const u32 id = static_cast<u32>(e), g = static_cast<u32>(e >> 32);
        const bool valid = status_of(n.flags[id]) == k_open && n.g[id] == g;
        flags[i] = !valid ? 0 : n.depth[id] >= max_depth ? 2 : 1;
    }
}

struct IsExpand
{
    const u8* flags;
    __device__ u32 operator()(u64 i) const { return flags[i] == 1 ? 1u : 0u; }
};
using ExpandIt = thrust::transform_iterator<IsExpand, thrust::counting_iterator<u64>>;

__global__ void k_pop_take(Pop p, Nodes n, Ctl* ctl, const u8* flags, const u32* pos, u32* batch_id, u32* batch_g, u32* batch_depth)
{
    Ctl& c = *ctl;
    const u32 window = c.window, wruns = c.wruns, at = c.n;
    const u32 need = c.goal_pop ? k_none : c.limit - c.n;
    MYMYR_GRID_LOOP(i, window)
    {
        const u32 k = pos[i];
        if (k >= need)
            continue;  // past the batch: stays in the queue
        const u8 f = flags[i];
        if (f == 0)
            continue;
        if (f == 1 && k == need - 1)
            c.cut = static_cast<u32>(i) + 1;  // (one writer: the need-th expanded entry)
        const u64 e = p.entries[entry_index(p, wruns, static_cast<u32>(i))];
        const u32 id = static_cast<u32>(e);
        n.flags[id] = static_cast<u8>((n.flags[id] & ~3u) | k_closed);
        if (f == 2)
        {
            atomicAdd(&c.pruned_round, 1u);
            continue;
        }
        batch_id[at + k] = id;
        batch_g[at + k] = static_cast<u32>(e >> 32);
        batch_depth[at + k] = n.depth[id];
    }
}

__global__ void k_pop_finish(Pop p, Ctl* ctl, const u32* pos)
{
    Ctl& c = *ctl;
    if (c.status != k_running || c.abort || c.round == 0)
        return;
    const u32 window = c.window;
    const u32 need = c.goal_pop ? k_none : c.limit - c.n;
    const u32 expand = pos[window];
    const u32 taken = expand < need ? expand : need;
    const u32 cut = need != k_none && expand >= need ? c.cut : window;
    if (cut > 0)
    {
        // the run holding entry cut - 1
        u32 lo = 0, hi = c.wruns;
        while (hi - lo > 1)
        {
            const u32 mid = (lo + hi) / 2;
            if (p.wprefix[mid] <= cut - 1)
                lo = mid;
            else
                hi = mid;
        }
        const u32 r = c.head + lo;
        const u32 x = cut - p.wprefix[lo];
        if (x == p.runs.count[r])
            c.head = r + 1;
        else
        {
            p.runs.begin[r] += x;
            p.runs.count[r] -= x;
            c.head = r;
        }
    }
    c.n += taken;
    c.popped += cut;
    c.pruned += c.pruned_round;
    c.stale += cut - taken - c.pruned_round;
    if (c.goal_pop)
    {
        if (c.n > 0)
            c.status = k_solved;
        else if (c.more)
            c.abort |= k_abort_pop;
    }
    else if (c.n < c.limit && c.more)
        c.abort |= k_abort_pop;
}

__global__ void k_gather_rows(const u64* arena, u32 words, const u32* ids, u32 n, const u32* count, u64* out)
{
    const u64 m = u64{count ? (*count < n ? *count : n) : n} * words;
    MYMYR_GRID_LOOP(k, m)
    {
        const u64 i = k / words, w = k % words;
        out[k] = arena[u64{ids[i]} * words + w];
    }
}

// ---------------------------------------------------------------------------------------------------------- chunk

__global__ void k_chunk_begin(Ctl* ctl, u32 k, u32 chunk_rows)
{
    Ctl& c = *ctl;
    const u64 first = u64{k} * chunk_rows;
    const bool on = c.status == k_running && c.abort == 0 && !c.goal_pop && c.n > first;
    c.rows = on ? static_cast<u32>(min_u64(c.n - first, chunk_rows)) : 0;
    c.base = c.count;
    c.M = 0;
    c.live = 0;
    c.fresh = 0;
    c.heur_n = 0;
    c.new_runs = 0;
    c.push_n = 0;
    c.goal_c = k_none;
    c.dead = 0;
    c.reopened_c = 0;
}

__global__ void k_chunk_check(Ctl* ctl, const u32* total, u32 k, ChunkLimits lim)
{
    Ctl& c = *ctl;
    c.live = 0;
    if (c.rows == 0)
        return;
    const u32 m = *total;
    c.M = m;
    if (m > c.step_max_M)
        c.step_max_M = m;
    if (m > c.loop_max_M)
        c.loop_max_M = m;
    if (m > lim.capacity || u64{c.count} + m > lim.states || c.open_size + m > lim.entries || u64{c.runs - c.head} + m > lim.runs)
    {
        c.abort |= k_abort_room;
        c.stop_chunk = k;
        return;
    }
    c.live = m;
}

__global__ void k_chunk_missing(Ctl* ctl, const u32* missing, u32 k)
{
    Ctl& c = *ctl;
    if (c.live && *missing)
    {
        c.abort |= k_abort_missing;
        c.stop_chunk = k;
        c.live = 0;
    }
}

__global__ void k_fresh(Nodes n, const Ctl* ctl, state_set::Rows arena, u64* fresh_rows)
{
    const u32 fresh = ctl->fresh;
    const u64 base = ctl->base;
    const u32 w = arena.words;
    MYMYR_GRID_LOOP(k, u64{fresh} * w)
    {
        const u64 i = k / w;
        fresh_rows[k] = arena.data[(base + i) * w + (k % w)];
        if (k % w == 0)
            n.flags[base + i] = k_new;
    }
}

/// The id of candidate c after the chunk's compaction: its slot's reference, or (a duplicate of a stored state) the
/// stored state with its content, found by probing like state_set's insert.
__device__ u32 candidate_id(const Relax& r, u64 c)
{
    const u32 res = r.result[c];
    if (res != state_set::k_dup)
        return static_cast<u32>(__ldcg(reinterpret_cast<const unsigned long long*>(r.table.slots + res))) - 1u;
    const u32 nw = r.arena.words;
    const u64* x = r.cand + c * nw;
    const u64 h = state_set::row_hash(x, nw);
    const u64 tag = h & k_tag;
    for (u64 j = h & r.table.mask;; j = (j + 1) & r.table.mask)
    {
        const u64 s = __ldcg(reinterpret_cast<const unsigned long long*>(r.table.slots + j));
        if (s == 0)
            return k_none;  // not stored (cannot happen for a duplicate)
        if ((s & k_tag) != tag)
            continue;
        const u32 ref = static_cast<u32>(s);
        if (ref & state_set::k_pending)
            continue;
        const u64* y = r.arena.data + u64{ref - 1} * nw;
        bool eq = true;
        for (u32 i = 0; i < nw && eq; ++i)
            eq = x[i] == y[i];
        if (eq)
            return ref - 1;
    }
}

__device__ u32 action_cost(const Relax& r, u64 c, u32* error)
{
    if (r.costs.unit)
        return 1;
    const f64 v = costs::evaluate(r.costs.program, 0, r.schema[c], r.binding + c * r.label_width);
    if (!(v >= 0 && v < 2147483648.0))  // undefined (NaN), negative, or beyond the u32 g values
    {
        atomicOr(error, k_err_cost);
        return 0;
    }
    return static_cast<u32>(v);
}

__device__ __forceinline__ u64 relax_key(const Relax& r, u64 c, u32 gc)
{
    return r.mode == k_astar ? (u64{gc} << 32) | c : c;
}

__global__ void k_relax(Relax r, Nodes n, Ctl* ctl)
{
    const u32 live = ctl->live, fresh_base = ctl->base;
    MYMYR_GRID_LOOP(c, live)
    {
        const u32 id = candidate_id(r, c);
        r.id[c] = id;
        const u32 pl = r.parent[c];
        const u64 g = u64{r.batch_g[pl]} + action_cost(r, c, &ctl->error);
        if (g >= k_inf)
            atomicOr(&ctl->error, k_err_overflow);
        const u32 gc = g >= k_inf ? k_inf - 1 : static_cast<u32>(g);
        r.gc[c] = gc;
        bool part = id != k_none && id >= fresh_base;
        if (!part && id != k_none && r.mode == k_astar)
        {
            const u8 st = status_of(n.flags[id]);
            part = gc < n.g[id] && (st == k_open || (st == k_closed && r.reopen));
        }
        r.part[c] = part ? 1 : 0;
        if (part)
            atomicMin(reinterpret_cast<unsigned long long*>(n.best + id), relax_key(r, c, gc));
    }
}

__global__ void k_win(Relax r, Nodes n, Ctl* ctl)
{
    const u32 live = ctl->live, fresh_base = ctl->base;
    MYMYR_GRID_LOOP(c, live)
    {
        u8 won = 0;
        if (r.part[c])
        {
            const u32 id = r.id[c], gc = r.gc[c];
            if (n.best[id] == relax_key(r, c, gc))
            {
                won = 1;
                const u32 pl = r.parent[c];
                n.g[id] = gc;
                n.parent[id] = r.batch_id[pl];
                n.sidx[id] = static_cast<u32>(c) - r.seg_offsets[u64{pl - r.parent_base} * r.num_schemas];
                n.depth[id] = r.batch_depth[pl] + 1;
                if (id < fresh_base)
                {
                    const u8 f = n.flags[id];
                    if (status_of(f) == k_closed)
                        atomicAdd(&ctl->reopened_c, 1u);
                    n.flags[id] = static_cast<u8>((f & ~3u) | k_open);
                }
            }
        }
        r.push[c] = won;
    }
}

__global__ void k_reset(Relax r, Nodes n, const Ctl* ctl)
{
    const u32 live = ctl->live;
    MYMYR_GRID_LOOP(c, live)
    if (r.part[c])
        n.best[r.id[c]] = k_no_key;
}

__global__ void k_set_goals(Nodes n, const Ctl* ctl, const u8* goal)
{
    const u32 fresh = ctl->fresh;
    const u64 base = ctl->base;
    MYMYR_GRID_LOOP(i, fresh)
    if (goal[i])
        n.flags[base + i] |= k_goal;
}

__global__ void k_settle(Nodes n, Ctl* ctl, const u32* h_fresh)
{
    const u32 cnt = ctl->heur_n;
    const u64 base = ctl->base;
    MYMYR_GRID_LOOP(i, cnt)
    {
        const u64 id = base + i;
        const u32 h = h_fresh ? h_fresh[i] : 0;
        n.h[id] = h;
        const bool dead = h == k_inf;
        n.flags[id] = static_cast<u8>((n.flags[id] & ~3u) | (dead ? k_dead : k_open));
        if (dead)
            atomicAdd(&ctl->dead, 1u);
    }
}

__global__ void k_count_dead(Nodes n, u64 first, u64 count, u32* out)
{
    MYMYR_GRID_LOOP(i, count)
    if (status_of(n.flags[first + i]) == k_dead)
        atomicAdd(out, 1u);
}

__global__ void k_first_goal(Relax r, Nodes n, Ctl* ctl)
{
    const u32 live = ctl->status == k_running ? ctl->live : 0;
    const u32 fresh_base = ctl->base;
    const u64 max_f = ctl->key0 >> 32;
    MYMYR_GRID_LOOP(c, live)
    {
        if (!r.push[c])
            continue;
        const u32 id = r.id[c];
        const u8 f = n.flags[id];
        if (!(f & k_goal))
            continue;
        if (r.mode == k_gbfs ? id < fresh_base : (status_of(f) == k_dead || u64{r.gc[c]} + n.h[id] > max_f))
            continue;
        atomicMin(&ctl->goal_c, static_cast<u32>(c));
    }
}

__global__ void k_goal_gate(Ctl* ctl, u32 k, u32 set_heur)
{
    Ctl& c = *ctl;
    if (c.status == k_running && c.goal_c != k_none)
    {
        c.status = k_goal_found;
        c.stop_chunk = k;
    }
    if (set_heur)
        c.heur_n = c.status == k_running ? c.fresh : 0;
}

__global__ void k_goal_info(Relax r, const u32* rank, Ctl* ctl, u32 k)
{
    Ctl& c = *ctl;
    if (c.status != k_goal_found || c.stop_chunk != k)
        return;
    const u32 cg = c.goal_c;
    const u32 pl = r.parent[cg];
    c.goal_id = r.id[cg];
    c.goal_parent = pl;
    c.goal_end = r.seg_offsets[u64{pl - r.parent_base + 1} * r.num_schemas];
    c.goal_fresh_before = rank[c.goal_end];
    c.goal_base = c.base;
    c.goal_fresh = c.fresh;
}

__global__ void k_push_keys(Push p, Nodes n, Ctl* ctl, u64 cands)
{
    const u32 live = ctl->status == k_running ? ctl->live : 0;
    MYMYR_GRID_LOOP(c, cands)
    {
        u64 key = k_no_key, entry = 0;
        if (c < live)
        {
            const u32 id = p.id[c], g = p.gc[c];
            entry = (u64{g} << 32) | id;
            if (p.push[c] && status_of(n.flags[id]) != k_dead)
            {
                const u32 h = n.h[id];
                if (p.mode == k_astar)
                {
                    const u64 f = u64{g} + h;
                    if (f >= k_inf)
                        atomicOr(&ctl->error, k_err_overflow);
                    key = (f << 32) | ((n.flags[id] & k_goal) ? 0u : h + 1u);
                }
                else
                    key = (u64{h} << 32) | g;
            }
        }
        p.keys[c] = key;
        p.entries[c] = entry;
    }
}

// ---------------------------------------------------------------------------------------------------------- append

/// The chunk's runs' arena begins (one block): new_runs excludes the k_no_key run, push_n = their entries.
__global__ void k_append_runs(Append a, Ctl* ctl)
{
    using Scan = cub::BlockScan<u64, k_block>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ u64 s_acc;
    __shared__ u32 s_nr;
    Ctl& c = *ctl;
    if (threadIdx.x == 0)
    {
        s_acc = 0;
        c.merge = c.status == k_running && c.abort == 0 && c.rows > 0 ? 1u : 0u;
        u32 nr = c.merge ? c.new_runs : 0;
        if (nr > 0 && a.runs_key[nr - 1] == k_no_key)
            --nr;
        s_nr = nr;
    }
    __syncthreads();
    const u32 nr = s_nr;
    for (u32 t0 = 0; t0 < nr; t0 += k_block)
    {
        const u32 j = t0 + threadIdx.x;
        const u64 cnt = j < nr ? a.runs_count[j] : 0;
        u64 excl = 0, total = 0;
        Scan(tmp).ExclusiveSum(cnt, excl, total);
        if (j < nr)
            a.new_begin[j] = c.open_size + s_acc + excl;
        __syncthreads();
        if (threadIdx.x == 0)
            s_acc += total;
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        c.new_runs = nr;
        c.push_n = static_cast<u32>(s_acc);
    }
}

/// Merge: old run i (of [head, runs)) goes after the new runs of smaller keys, new run j after the old runs of keys at
/// most its own; the sorted entries go to the arena.
__global__ void k_merge_runs(Append a, const Ctl* ctl)
{
    const Ctl& c = *ctl;
    if (!c.merge)
        return;
    const u32 head = c.head, old_n = c.runs - c.head, nr = c.new_runs;
    const u64 open = c.open_size;
    const u32 push_n = c.push_n;
    const u64 total = u64{old_n} + nr;
    const u64 n = total > push_n ? total : push_n;
    MYMYR_GRID_LOOP(i, n)
    {
        if (i < push_n)
            a.arena[open + i] = a.sorted_entries[i];
        if (i >= total)
            continue;
        if (i < old_n)
        {
            const u64 key = a.runs.key[head + i];
            u32 lo = 0, hi = nr;  // new runs with keys < key
            while (lo < hi)
            {
                const u32 mid = (lo + hi) / 2;
                if (a.runs_key[mid] < key)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const u64 at = i + lo;
            a.runs.key2[at] = key;
            a.runs.begin2[at] = a.runs.begin[head + i];
            a.runs.count2[at] = a.runs.count[head + i];
        }
        else
        {
            const u32 j = static_cast<u32>(i - old_n);
            const u64 key = a.runs_key[j];
            u32 lo = 0, hi = old_n;  // old runs with keys <= key
            while (lo < hi)
            {
                const u32 mid = (lo + hi) / 2;
                if (a.runs.key[head + mid] <= key)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const u64 at = j + lo;
            a.runs.key2[at] = key;
            a.runs.begin2[at] = a.new_begin[j];
            a.runs.count2[at] = a.runs_count[j];
        }
    }
}

__global__ void k_copy_runs(Append a, const Ctl* ctl)
{
    const Ctl& c = *ctl;
    if (!c.merge)
        return;
    const u64 total = u64{c.runs - c.head} + c.new_runs;
    MYMYR_GRID_LOOP(i, total)
    {
        a.runs.key[i] = a.runs.key2[i];
        a.runs.begin[i] = a.runs.begin2[i];
        a.runs.count[i] = a.runs.count2[i];
    }
}

__global__ void k_chunk_end(Append a, Ctl* ctl, u32 k)
{
    Ctl& c = *ctl;
    if (c.rows == 0)
        return;  // (no parents, or the step stopped before the chunk)
    if (c.abort && c.stop_chunk == k)
        return;  // the host redoes the chunk
    ++c.chunks;
    c.reopened += c.reopened_c;
    const bool goal = c.status == k_goal_found && c.stop_chunk == k;
    if (goal && a.mode == k_gbfs)
        return;  // the host's stop at the goal counts the rest
    c.evaluations += c.heur_n;
    c.dead_ends += c.dead;
    c.pruned += c.dead;
    if (!c.merge)
        return;  // (a goal, or the search stopped)
    c.generated += c.M;
    c.expanded += c.rows;
    c.runs = c.runs - c.head + c.new_runs;
    c.head = 0;
    c.open_size += c.push_n;
    c.open_entries += c.push_n;
    if (c.count > a.max_states)
        c.status = k_out_of_states;
}

/// Runs past which the head bucket counts as at least a batch (the prediction's loop bound).
constexpr u32 k_bucket_scan = 256;

__global__ void k_step_end(Ctl* ctl, Pop p, ChunkLimits next, u32 chunks, u32 chunk, cudaGraphConditionalHandle handle, u32 looped)
{
    Ctl& c = *ctl;
    u32 end = k_loop_running;
    if (c.abort)
        end = k_loop_stopped;  // the host finishes the step
    else
    {
        const bool expanded = c.n > 0 && !c.goal_pop;
        if (expanded)
        {
            ++c.steps;
            if (c.n > c.max_batch)
                c.max_batch = c.n;
        }
        c.round = 0;
        // the next step's chunks: this step's candidates per parent of its fullest chunk, times the next batch's
        u64 next_n = c.open_size < p.batch ? c.open_size : p.batch;
        if (p.mode == k_astar && p.single_bucket && c.head < c.runs)
        {
            const u64 key = p.runs.key[c.head];
            u64 bucket = 0;
            u32 i = c.head;
            for (; i < c.runs && i < c.head + k_bucket_scan && p.runs.key[i] == key && bucket < next_n; ++i)
                bucket += p.runs.count[i];
            if (bucket < next_n && (i == c.runs || p.runs.key[i] != key))
                next_n = bucket;
        }
        const u64 rows = c.n < chunk ? c.n : chunk, next_rows = next_n < chunk ? next_n : chunk;
        u64 pred = c.step_max_M;
        if (expanded && rows > 0)
            pred = (u64{c.step_max_M} * next_rows + rows - 1) / rows;
        c.next_M = static_cast<u32>(pred < 0xFFFFFFFFull ? pred : 0xFFFFFFFFull);
        const u64 more = next.capacity * chunks;
        if (c.status != k_running)
            end = k_loop_stopped;
        else if (c.steps_left <= 1)
            end = k_loop_steps;
        else if (u64{c.count} + more > next.states || c.open_size + more > next.entries || u64{c.runs - c.head} + more > next.runs ||
                 u64{c.step_max_M} * 4 > next.capacity * 3 || pred * 4 > next.capacity * 3)
            end = k_loop_room;  // (the next step's chunks may come near the capacity)
        c.step_max_M = 0;
        if (c.steps_left > 0)
            --c.steps_left;
    }
    c.loop_end = end;
    if (looped && end != k_loop_running)
        cudaGraphSetConditional(handle, 0);
}

__global__ void k_fill_u64(u64* out, u64 n, u64 v)
{
    MYMYR_GRID_LOOP(i, n)
    out[i] = v;
}
}  // namespace

// --------------------------------------------------------------------------------------------------------- launchers

cudaError_t launch_pop_plan(Pop p, Ctl* ctl, cudaStream_t s)
{
    k_pop_plan<<<1, k_block, 0, s>>>(p, ctl);
    return cudaGetLastError();
}

cudaError_t launch_pop_flags(Pop p, Nodes n, const Ctl* ctl, u8* flags, cudaStream_t s)
{
    k_pop_flags<<<grid_for(u64{p.window_cap} + 1), k_block, 0, s>>>(p, n, ctl, flags);
    return cudaGetLastError();
}

u64 pop_temp_bytes(u32 window_cap)
{
    size_t bytes = 0;
    ExpandIt in(thrust::counting_iterator<u64>(0), IsExpand{nullptr});
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, static_cast<u32*>(nullptr), static_cast<u64>(window_cap) + 1);
    return bytes;
}

cudaError_t launch_pop_scan(const u8* flags, u32 window_cap, u32* pos, void* temp, u64 temp_bytes, cudaStream_t s)
{
    ExpandIt in(thrust::counting_iterator<u64>(0), IsExpand{flags});
    size_t bytes = temp_bytes;
    return cub::DeviceScan::ExclusiveSum(temp, bytes, in, pos, static_cast<u64>(window_cap) + 1, s);
}

cudaError_t launch_pop_take(Pop p, Nodes n, Ctl* ctl, const u8* flags, const u32* pos, u32* batch_id, u32* batch_g,
                            u32* batch_depth, cudaStream_t s)
{
    k_pop_take<<<grid_for(p.window_cap), k_block, 0, s>>>(p, n, ctl, flags, pos, batch_id, batch_g, batch_depth);
    return cudaGetLastError();
}

cudaError_t launch_pop_finish(Pop p, Ctl* ctl, const u32* pos, cudaStream_t s)
{
    k_pop_finish<<<1, 1, 0, s>>>(p, ctl, pos);
    return cudaGetLastError();
}

cudaError_t launch_gather_rows(const u64* arena, u32 words, const u32* ids, u32 n, const u32* count, u64* out, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    k_gather_rows<<<grid_for(u64{n} * words), k_block, 0, s>>>(arena, words, ids, n, count, out);
    return cudaGetLastError();
}

cudaError_t launch_chunk_begin(Ctl* ctl, u32 k, u32 chunk_rows, cudaStream_t s)
{
    k_chunk_begin<<<1, 1, 0, s>>>(ctl, k, chunk_rows);
    return cudaGetLastError();
}

cudaError_t launch_chunk_check(Ctl* ctl, const u32* total, u32 k, ChunkLimits limits, cudaStream_t s)
{
    k_chunk_check<<<1, 1, 0, s>>>(ctl, total, k, limits);
    return cudaGetLastError();
}

cudaError_t launch_chunk_missing(Ctl* ctl, const u32* missing, u32 k, cudaStream_t s)
{
    k_chunk_missing<<<1, 1, 0, s>>>(ctl, missing, k);
    return cudaGetLastError();
}

cudaError_t launch_fresh(Nodes n, const Ctl* ctl, state_set::Rows arena, u64* fresh_rows, u64 capacity, cudaStream_t s)
{
    if (capacity == 0)
        return cudaSuccess;
    k_fresh<<<grid_for(capacity * arena.words), k_block, 0, s>>>(n, ctl, arena, fresh_rows);
    return cudaGetLastError();
}

cudaError_t launch_relax(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s)
{
    if (cands == 0)
        return cudaSuccess;
    k_relax<<<grid_for(cands), k_block, 0, s>>>(r, n, ctl);
    return cudaGetLastError();
}

cudaError_t launch_win(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s)
{
    if (cands == 0)
        return cudaSuccess;
    k_win<<<grid_for(cands), k_block, 0, s>>>(r, n, ctl);
    return cudaGetLastError();
}

cudaError_t launch_reset(Relax r, Nodes n, const Ctl* ctl, u64 cands, cudaStream_t s)
{
    if (cands == 0)
        return cudaSuccess;
    k_reset<<<grid_for(cands), k_block, 0, s>>>(r, n, ctl);
    return cudaGetLastError();
}

cudaError_t launch_set_goals(Nodes n, const Ctl* ctl, const u8* goal, u64 capacity, cudaStream_t s)
{
    if (capacity == 0)
        return cudaSuccess;
    k_set_goals<<<grid_for(capacity), k_block, 0, s>>>(n, ctl, goal);
    return cudaGetLastError();
}

cudaError_t launch_first_goal(Relax r, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s)
{
    if (cands == 0)
        return cudaSuccess;
    k_first_goal<<<grid_for(cands), k_block, 0, s>>>(r, n, ctl);
    return cudaGetLastError();
}

cudaError_t launch_goal_gate(Ctl* ctl, u32 k, bool set_heur, cudaStream_t s)
{
    k_goal_gate<<<1, 1, 0, s>>>(ctl, k, set_heur ? 1u : 0u);
    return cudaGetLastError();
}

cudaError_t launch_goal_info(Relax r, const u32* rank, Ctl* ctl, u32 k, cudaStream_t s)
{
    k_goal_info<<<1, 1, 0, s>>>(r, rank, ctl, k);
    return cudaGetLastError();
}

cudaError_t launch_settle(Nodes n, Ctl* ctl, const u32* h_fresh, u64 capacity, cudaStream_t s)
{
    if (capacity == 0)
        return cudaSuccess;
    k_settle<<<grid_for(capacity), k_block, 0, s>>>(n, ctl, h_fresh);
    return cudaGetLastError();
}

cudaError_t launch_count_dead(Nodes n, u64 first, u64 count, u32* out, cudaStream_t s)
{
    if (count == 0)
        return cudaSuccess;
    k_count_dead<<<grid_for(count), k_block, 0, s>>>(n, first, count, out);
    return cudaGetLastError();
}

cudaError_t launch_push_keys(Push p, Nodes n, Ctl* ctl, u64 cands, cudaStream_t s)
{
    if (cands == 0)
        return cudaSuccess;
    k_push_keys<<<grid_for(cands), k_block, 0, s>>>(p, n, ctl, cands);
    return cudaGetLastError();
}

u64 sort_temp_bytes(u64 n)
{
    size_t a = 0, b = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, a, static_cast<const u64*>(nullptr), static_cast<u64*>(nullptr),
                                    static_cast<const u64*>(nullptr), static_cast<u64*>(nullptr), static_cast<int>(n));
    cub::DeviceRunLengthEncode::Encode(nullptr, b, static_cast<const u64*>(nullptr), static_cast<u64*>(nullptr),
                                       static_cast<u32*>(nullptr), static_cast<u32*>(nullptr), static_cast<int>(n));
    return (a > b ? a : b);
}

cudaError_t launch_sort_runs(const u64* keys, const u64* entries, u64 n, u64* keys_out, u64* entries_out, u64* runs_key,
                             u32* runs_count, u32* runs, void* temp, u64 temp_bytes, cudaStream_t s)
{
    if (n == 0)
        return cudaMemsetAsync(runs, 0, sizeof(u32), s);
    size_t bytes = temp_bytes;
    cudaError_t e = cub::DeviceRadixSort::SortPairs(temp, bytes, keys, keys_out, entries, entries_out, static_cast<int>(n), 0, 64, s);
    if (e != cudaSuccess)
        return e;
    bytes = temp_bytes;
    return cub::DeviceRunLengthEncode::Encode(temp, bytes, keys_out, runs_key, runs_count, runs, static_cast<int>(n), s);
}

cudaError_t launch_append(Append a, Ctl* ctl, cudaStream_t s)
{
    k_append_runs<<<1, k_block, 0, s>>>(a, ctl);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;
    const u64 n = a.runs.capacity > a.cands ? a.runs.capacity : a.cands;
    k_merge_runs<<<grid_for(n), k_block, 0, s>>>(a, ctl);
    e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;
    k_copy_runs<<<grid_for(a.runs.capacity), k_block, 0, s>>>(a, ctl);
    return cudaGetLastError();
}

cudaError_t launch_chunk_end(Append a, Ctl* ctl, u32 k, cudaStream_t s)
{
    k_chunk_end<<<1, 1, 0, s>>>(a, ctl, k);
    return cudaGetLastError();
}

cudaError_t launch_step_end(Ctl* ctl, Pop p, ChunkLimits next, u32 chunks, u32 chunk, unsigned long long handle, cudaStream_t s)
{
    k_step_end<<<1, 1, 0, s>>>(ctl, p, next, chunks, chunk, static_cast<cudaGraphConditionalHandle>(handle), handle ? 1u : 0u);
    return cudaGetLastError();
}

cudaError_t launch_fill_u64(u64* out, u64 n, u64 v, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    k_fill_u64<<<grid_for(n), k_block, 0, s>>>(out, n, v);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::bfk
