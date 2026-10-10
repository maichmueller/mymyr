// The kernels of the multi-search IW driver (include/mymyr/cuda/multi_iw_kernels.hpp). Compiled by nvcc as C++20;
// includes only the device-code subset.
//
// Determinism: every decision is a function of the candidate order. Atomics pick hash slots and compute minima and
// counts (atomicMin of candidate indices, atomicOr of table bits, atomicAdd of counters), never an order or a winner:
// owners are minima, positions come from exclusive scans, and the cut of a search is the candidate at which the
// scanned admission count reaches its limit.

#include "grid_stride.cuh"
#include "live_scan.cuh"

#include "mymyr/cuda/device_rng.hpp"
#include "mymyr/cuda/multi_iw_kernels.hpp"

namespace mymyr::cuda::miw
{
namespace
{
using rl::dev::TaskView;

constexpr unsigned k_block = 256;
constexpr unsigned k_full_warp = 0xFFFFFFFFu;

/// Blocks of a grid-stride launch over n items. At most k_max_blocks (one wave on current GPUs): the chunk kernels
/// launch over capacities, and their loops spread the live items over the blocks (MIW_FOR, grid_stride.cuh).
constexpr u64 k_max_blocks = 256;

unsigned grid_for(u64 n)
{
    const u64 g = (n + k_block - 1) / k_block;
    return static_cast<unsigned>(g < k_max_blocks ? (g ? g : 1) : k_max_blocks);
}

#define MIW_FOR(i, n) for (u64 i = spread_first(); i < (n); i += u64{gridDim.x} * blockDim.x)  // (grid_stride.cuh)

__device__ __forceinline__ u32 lowest(u64 x) { return static_cast<u32>(__ffsll(static_cast<long long>(x)) - 1); }

__device__ __forceinline__ bool aborted(const u32* abort) { return abort && *abort; }

/// The candidates present (a device count under the capacity c.n; none once the chunk is aborted).
__device__ __forceinline__ u64 live_candidates(const Candidates& c)
{
    if (aborted(c.abort))
        return 0;
    if (!c.n_dev)
        return c.n;
    const u64 n = *c.n_dev;
    return n < c.n ? n : c.n;
}
__device__ __forceinline__ u64 live(const Candidates& c) { return live_candidates(c); }

/// Where candidate i's row and labels are (Candidates::src).
__device__ __forceinline__ u64 cand_at(const Candidates& c, u64 i) { return c.src ? u64{c.src[i]} : i; }
__device__ __forceinline__ const u64* cand_row(const Candidates& c, u64 i, u32 words) { return c.rows + cand_at(c, i) * words; }

/// The gathered rows present (a device count under the capacity g.n).
__device__ __forceinline__ u64 live(const Gathered& g)
{
    if (!g.n_dev)
        return g.n;
    const u64 n = *g.n_dev;
    return n < g.n ? n : g.n;
}

// ------------------------------------------------------------------------------------------------ keys and tables

__device__ __forceinline__ u64 tuple_key(u32 s, u32 a, u32 b)
{
    const u32 lo = a < b ? a : b, hi = a < b ? b : a;
    return (u64{s} << (2 * k_key_atom_bits)) | (u64{lo} << k_key_atom_bits) | hi;
}

/// Word of tuple (lo, hi) in the table of search s, and its bit.
__device__ __forceinline__ unsigned long long* table_word(const Table& t, u32 s, u32 a, u32 b)
{
    if (t.arity == 1)
        return reinterpret_cast<unsigned long long*>(t.bits + u64{s} * t.row_words + (a >> 6));
    return reinterpret_cast<unsigned long long*>(t.bits + (u64{s} * 64 * t.row_words + a) * t.row_words + (b >> 6));
}

/// The pair-table row of atom a of search s (arity 2).
__device__ __forceinline__ const u64* pair_row(const Table& t, u32 s, u32 a)
{
    return t.bits + (u64{s} * 64 * t.row_words + a) * t.row_words;
}

/// Calls f(a, b) for every tuple of the transition parent -> child that is unseen in the table: arity 1, the added
/// atoms a (a == b); arity 2, {a, b} with a added and b in the child (b >= a when both are added: each pair once).
template<class F>
__device__ __forceinline__ void for_each_unseen(const Table& t, u32 s, const u64* pw, const u64* cw, F&& f)
{
    const u32 rw = t.row_words;
    if (t.arity == 1)
    {
        const u64* row = t.bits + u64{s} * rw;
        for (u32 w = 0; w < rw; ++w)
            for (u64 x = cw[w] & ~pw[w] & ~row[w]; x; x &= x - 1)
            {
                const u32 a = w * 64 + lowest(x);
                f(a, a);
            }
        return;
    }
    for (u32 aw = 0; aw < rw; ++aw)
        for (u64 ax = cw[aw] & ~pw[aw]; ax; ax &= ax - 1)
        {
            const u32 a = aw * 64 + lowest(ax);
            const u64* row = pair_row(t, s, a);
            for (u32 w = 0; w < rw; ++w)
            {
                u64 x = cw[w] & ~row[w];
                // added partners below a are enumerated from their own row
                const u64 add = cw[w] & ~pw[w];
                const u64 below = w < aw ? ~0ull : (w == aw ? ((1ull << (a & 63)) - 1) : 0ull);
                x &= ~(add & below);
                for (; x; x &= x - 1)
                    f(a, w * 64 + lowest(x));
            }
        }
}

// ------------------------------------------------------------------------------------------------ pass setup

__global__ void k_reset_searches(Searches s)
{
    MIW_FOR(i, s.count)
    {
        s.status[i] = k_idle;
        s.expanded[i] = 0;
        s.generated[i] = 0;
        s.in_tree[i] = 0;
        s.skipped[i] = 0;
        s.tree[i] = 0;
        s.next[i] = 0;
        s.layer_cut[i] = 0;
        s.goal_node[i] = k_none;
        s.goal_depth[i] = 0;
    }
}

__global__ void k_init_roots(Nodes nodes, Searches s, Roots r, u32* order)
{
    const u32 W = nodes.words;
    MIW_FOR(i, r.n)
    {
        const u32 id = r.search_of[i];
        const u64* src = r.starts + u64{id} * r.stride;
        u64* dst = nodes.rows + i * W;
        for (u32 w = 0; w < W; ++w)
            dst[w] = w < r.start_words ? src[w] : 0;
        nodes.parent[i] = k_none;
        nodes.search[i] = id;
        for (u32 k = 0; k < nodes.label_words; ++k)
            nodes.label[i * nodes.label_words + k] = k_none;
        nodes.flags[i] = 0;
        order[i] = static_cast<u32>(i);
        s.status[id] = k_running;
        s.tree[id] = 1;
    }
}

__global__ void k_table_init(Table t, Nodes nodes, u32 roots)
{
    const u32 rw = t.row_words;
    if (t.arity == 1)
    {
        MIW_FOR(k, u64{roots} * rw)
        {
            const u64 i = k / rw;
            const u32 w = static_cast<u32>(k % rw);
            t.bits[u64{nodes.search[i]} * rw + w] = w < nodes.words ? nodes.rows[i * nodes.words + w] : 0;
        }
        return;
    }
    const u64 atoms = u64{64} * rw;
    MIW_FOR(k, u64{roots} * atoms)
    {
        const u64 i = k / atoms;
        const u32 a = static_cast<u32>(k % atoms);
        const u64* root = nodes.rows + i * nodes.words;
        const bool has = (a >> 6) < nodes.words && ((root[a >> 6] >> (a & 63)) & 1);
        u64* row = t.bits + (u64{nodes.search[i]} * atoms + a) * rw;
        for (u32 w = 0; w < rw; ++w)
            row[w] = has && w < nodes.words ? root[w] : 0;
    }
}

__global__ void k_stop_running(Searches s, u32 status)
{
    MIW_FOR(i, s.count)
    {
        if (s.status[i] == k_running)
            s.status[i] = status;
    }
}

__global__ void k_table_relayout(const u64* src, u64* dst, u32 arity, u32 searches, u32 from, u32 to)
{
    const u64 rows_to = arity == 1 ? 1 : u64{64} * to, rows_from = arity == 1 ? 1 : u64{64} * from;
    MIW_FOR(k, u64{searches} * rows_to * to)
    {
        const u32 w = static_cast<u32>(k % to);
        const u64 row = (k / to) % rows_to;
        const u64 id = k / to / rows_to;
        dst[k] = row < rows_from && w < from ? src[(id * rows_from + row) * from + w] : 0;
    }
}

// ------------------------------------------------------------------------------------------------ chunk rows

/// The rows of a chunk present (its step's, under the capacity).
__device__ __forceinline__ u32 step_rows(const Chunk& c) { return c.step->rows < c.rows ? c.step->rows : c.rows; }

__global__ void k_rows_live(TaskView task, Nodes nodes, Searches s, Chunk c)
{
    const u32 W = nodes.words;
    const u32 begin = c.step->begin, n = step_rows(c);
    MIW_FOR(r, c.rows)
    {
        const u32 node = r < n ? c.step->cur[begin + r] : k_dead;  // (rows past the step's: no flags)
        u8 f = 0;
        if (node != k_dead)
        {
            const u32 id = nodes.search[node];
            if (s.status[id] == k_running && !s.layer_cut[id])
            {
                f = k_row_live;
                atomicMin(s.first_live + id, static_cast<u32>(r));
                atomicMax(s.last_live + id, static_cast<u32>(r));
                const u64* row = nodes.rows + u64{node} * W;
                bool goal;
                if (c.host_goal)
                    goal = c.host_goal[r] != 0;
                else if (s.goal_pos)
                {
                    goal = true;
                    const u64* pos = s.goal_pos + u64{id} * s.goal_words;
                    const u64* neg = s.goal_neg + u64{id} * s.goal_words;
                    for (u32 w = 0; w < s.goal_words; ++w)
                    {
                        const u64 x = w < W ? row[w] : 0;
                        goal = goal && (x & pos[w]) == pos[w] && (x & neg[w]) == 0;
                    }
                }
                else
                    goal = rl::dev::goal_holds(task, row, W, nullptr, 0);
                if (goal)
                {
                    f |= k_row_goal;
                    atomicMin(s.first_goal + id, static_cast<u32>(r));
                }
            }
        }
        c.row_flags[r] = f;
    }
}

/// The live rows in [fl, r) of a search whose first live row is fl (a search's rows are contiguous; dense chunks have no
/// dead entries among them).
__device__ __forceinline__ u64 live_rank(const Chunk& c, u32 r, u32 fl)
{
    return c.live_scan ? u64{c.live_scan[r] - c.live_scan[fl]} : u64{r - fl};
}

__global__ void k_rows_select(Nodes nodes, Searches s, Limits lim, Chunk c)
{
    MIW_FOR(r, step_rows(c))
    {
        u8 f = c.row_flags[r];
        if (!(f & k_row_live))
            continue;
        const u32 node = c.step->cur[c.step->begin + r];
        const u32 id = nodes.search[node];
        const u32 fl = s.first_live[id];
        const u64 li = live_rank(c, static_cast<u32>(r), fl);
        const u32 fg = s.first_goal[id];
        const u64 gi = fg == k_none ? ~0ull : live_rank(c, fg, fl);
        const unsigned long long e = s.expanded[id];
        const u64 bi = lim.max_expanded > e ? lim.max_expanded - e : 0;
        const u64 stop = gi < bi ? gi : bi;
        if (li < stop)
        {
            f |= k_row_expanded;
            if ((nodes.flags[node] & k_flag_skip) || c.step->layer >= lim.max_depth)
                f |= k_row_skip;
            else
                f |= k_row_gen;
        }
        c.row_flags[r] = f;
    }
}

__global__ void k_gather(Nodes nodes, Searches s, Chunk c, Gathered g)
{
    const u32 W = nodes.words;
    MIW_FOR(r, c.rows)
    {
        if (!(c.row_flags[r] & k_row_gen))
            continue;
        const u32 pos = c.gen_scan[r];
        const u32 node = c.step->cur[c.step->begin + r];
        const u32 id = nodes.search[node];
        const u64* src = nodes.rows + u64{node} * W;
        u64* dst = g.rows + u64{pos} * g.words;
        for (u32 w = 0; w < g.words; ++w)
            dst[w] = w < W ? src[w] : 0;
        g.node[pos] = node;
        g.crow[pos] = static_cast<u32>(r);
        g.search[pos] = id;
        atomicMin(s.first_gen + id, pos);
        atomicMax(s.last_gen + id, pos);
    }
}

// ------------------------------------------------------------------------------------------------ novelty

__global__ void k_emit_count(Gathered g, Candidates c, Table t, Emission e)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)  // (the scan reads no count past the live candidates)
    {
        const u32 p = c.parent[i];
        const u32 id = g.search[p];
        u32 n = 0;
        for_each_unseen(t, id, g.rows + u64{p} * g.words, cand_row(c, i, g.words), [&](u32, u32) { ++n; });
        e.count[i] = n;
    }
}

__device__ __forceinline__ u32 map_insert(const OwnerMap& m, u64 mask, u64 key, u32 cand)
{
    for (u64 j = device_rng::mix64(key) & mask;; j = (j + 1) & mask)
    {
        unsigned long long* k = reinterpret_cast<unsigned long long*>(m.keys + j);
        u64 cur = __ldcg(k);
        if (cur == k_empty_key)
        {
            const u64 old = atomicCAS(k, k_empty_key, key);
            cur = old == k_empty_key ? key : old;
        }
        if (cur == key)
        {
            atomicMin(m.vals + j, cand);
            return static_cast<u32>(j);
        }
    }
}

__global__ void k_emit_insert(Gathered g, Candidates c, Table t, Emission e, OwnerMap m)
{
    // the chunk's check of its unseen tuples: over the capacity, it aborts (every block sees the total)
    const u32 total = *e.total;
    if (blockIdx.x == 0 && threadIdx.x == 0 && e.ctl)
    {
        e.ctl[k_ctl_emit] = total;
        if (total > e.capacity)
            atomicOr(e.ctl + k_ctl_abort, k_abort_emit);
    }
    if (total > e.capacity)
        return;
    // the map's slots in use: twice the chunk's tuples (its allocation is for the capacity; a compact map stays in L2)
    u64 used = 1024;
    while (used < 2 * u64{total})
        used <<= 1;
    const u64 mask = used - 1 < m.mask ? used - 1 : m.mask;
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        const u32 p = c.parent[i];
        const u32 id = g.search[p];
        u32 at = e.offset[i];
        for_each_unseen(t, id, g.rows + u64{p} * g.words, cand_row(c, i, g.words),
                        [&](u32 a, u32 b) { e.slot[at++] = map_insert(m, mask, tuple_key(id, a, b), static_cast<u32>(i)); });
    }
}

/// The flags f of a candidate with its admission under a.rule (and k_cand_kept with a.keep).
__device__ __forceinline__ u8 admitted(u8 f, const Admission& a)
{
    const bool novel = f & k_cand_novel, self = f & k_cand_self, dup = f & k_cand_dup;
    switch (a.rule)
    {
        case k_rule_normal:
            if (novel)
                f |= k_cand_adm | k_cand_ent;
            break;
        case k_rule_continuation:
            if (!self && !dup)
                f |= k_cand_adm | k_cand_ent | (novel ? 0 : k_cand_skip);
            break;
        case k_rule_zero_root:
            if (!self)
                f |= k_cand_ent | (dup ? 0 : k_cand_adm | (a.root_only ? k_cand_skip : 0));
            break;
        default: break;
    }
    return a.keep ? static_cast<u8>(f | k_cand_kept) : f;
}

__global__ void k_novel_flags(Candidates c, Emission e, OwnerMap m, Admission a)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        bool novel = false;
        for (u32 k = e.offset[i]; k < e.offset[i + 1] && !novel; ++k)
            novel = m.vals[e.slot[k]] == static_cast<u32>(i);
        const u8 f = novel ? k_cand_novel : 0;
        c.flags[i] = a.admit ? admitted(f, a) : f;
    }
}

__global__ void k_relaxed(Gathered g, Candidates c, Table t, const Step* step, u32* abort, Admission a)
{
    if (!aborted(c.abort) && step)
    {
        const u64 n = live(c);
        if (u64{step->node_base} + n > step->node_cap || u64{step->entry_base} + n > step->next_cap)
        {
            if (blockIdx.x == 0 && threadIdx.x == 0 && abort)
                atomicOr(abort, k_abort_space);
            return;
        }
    }
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        const u32 p = c.parent[i];
        const u32 id = g.search[p];
        bool novel = false;
        for_each_unseen(t, id, g.rows + u64{p} * g.words, cand_row(c, i, g.words),
                        [&](u32 x, u32 y)
                        {
                            const u64 bit = 1ull << (y & 63);
                            if (!(atomicOr(table_word(t, id, x, y), bit) & bit))
                                novel = true;
                            if (t.arity == 2 && x != y)
                                atomicOr(table_word(t, id, y, x), 1ull << (x & 63));
                        });
        const u8 f = novel ? k_cand_novel : 0;
        c.flags[i] = a.admit ? admitted(f, a) : f;
    }
}

// ------------------------------------------------------------------------------------------------ root successors

__device__ __forceinline__ bool rows_equal(const u64* a, const u64* b, u32 n)
{
    for (u32 w = 0; w < n; ++w)
        if (a[w] != b[w])
            return false;
    return true;
}

__device__ __forceinline__ u64 root_hash(u32 id, const u64* w, u32 n)
{
    u64 h = device_rng::mix64(0x9E3779B97F4A7C15ull ^ id);
    for (u32 i = 0; i < n; ++i)
        h = device_rng::mix64(h ^ w[i]) + i;
    return h | 1;  // tags are never 0 (0 marks an empty slot)
}

__global__ void k_root_insert(Gathered g, Candidates c, RootSet r)
{
    const u32 W = g.words;
    constexpr u64 k_tag = 0xFFFFFFFF00000000ull;
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        const u32 p = c.parent[i];
        const u32 id = g.search[p];
        const u64* x = cand_row(c, i, W);
        if (rows_equal(x, g.rows + u64{p} * W, W))
        {
            c.flags[i] |= k_cand_self;
            r.owner[i] = k_none;
            continue;
        }
        const u64 h = root_hash(id, x, W);
        const u64 mine = (h & k_tag) | (i + 1);
        for (u64 j = h & r.mask;; j = (j + 1) & r.mask)
        {
            unsigned long long* sp = reinterpret_cast<unsigned long long*>(r.slots + j);
            u64 v = __ldcg(sp);
            if (v == 0)
            {
                const u64 old = atomicCAS(sp, 0ull, mine);
                if (old == 0)
                {
                    r.owner[i] = static_cast<u32>(j);
                    break;
                }
                v = old;
            }
            if ((v & k_tag) != (mine & k_tag))
                continue;
            const u64 other = (v & 0xFFFFFFFFull) - 1;
            if (g.search[c.parent[other]] == id && rows_equal(x, cand_row(c, other, W), W))
            {
                atomicMin(sp, mine);
                r.owner[i] = static_cast<u32>(j);
                break;
            }
        }
    }
}

__global__ void k_root_flags(Candidates c, RootSet r)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        if (c.flags[i] & k_cand_self)
            continue;
        const u32 owner = static_cast<u32>((r.slots[r.owner[i]] & 0xFFFFFFFFull) - 1);
        r.owner[i] = owner;
        if (owner != static_cast<u32>(i))
            c.flags[i] |= k_cand_dup;
    }
}

// ------------------------------------------------------------------------------------------------ admission and cuts

__global__ void k_admit(Candidates c, Admission a)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    c.flags[i] = admitted(c.flags[i], a);
}

__device__ __forceinline__ u32 first_candidate(const Candidates& c, const Searches& s, u32 id)
{
    return c.seg_offsets[u64{s.first_gen[id]} * c.num_schemas];
}

__device__ __forceinline__ u32 lo32(u64 x) { return static_cast<u32>(x); }
__device__ __forceinline__ u32 hi32(u64 x) { return static_cast<u32>(x >> 32); }

__global__ void k_cut(Gathered g, Candidates c, Searches s, Limits lim, const u64* scan)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        const u8 f = c.flags[i];
        if (!(f & (k_cand_adm | k_cand_ent)))
            continue;
        const u32 id = g.search[c.parent[i]];
        const u32 fc = first_candidate(c, s, id);
        const u32 tree = s.tree[id], next = s.next[id];
        // a tree the root alone fills: the first admitted candidate cuts, and k_keep drops it too
        const u32 lim_states = lim.max_states > tree ? lim.max_states - tree : 1;
        const u32 lim_next = lim.max_next > next ? lim.max_next - next : 1;
        if ((f & k_cand_adm) && lim.max_states != k_none && lo32(scan[i + 1]) - lo32(scan[fc]) == lim_states)
            atomicMin(s.cut + id, static_cast<u32>(2 * i));
        if ((f & k_cand_ent) && lim.max_next != k_none && hi32(scan[i + 1]) - hi32(scan[fc]) == lim_next)
            atomicMin(s.cut + id, static_cast<u32>(2 * i + 1));
    }
}

__global__ void k_keep(Gathered g, Candidates c, Searches s, Limits lim)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        const u32 id = g.search[c.parent[i]];
        const u32 cut = s.cut[id];
        // a state cut of a tree the root alone fills keeps candidates before its own: the search stops before storing it
        const bool full = !(cut & 1) && lim.max_states != k_none && lim.max_states <= s.tree[id];
        if (cut == k_none || i < (cut >> 1) || (i == (cut >> 1) && !full))
            c.flags[i] |= k_cand_kept;
    }
}

/// reached[id] |= the atoms of row x (n words).
__device__ __forceinline__ void add_reached(u64* reached, u32 reached_words, u32 id, const u64* x, u32 n)
{
    u64* r = reached + u64{id} * reached_words;
    const u32 m = n < reached_words ? n : reached_words;
    for (u32 w = 0; w < m; ++w)
    {
        const u64 add = x[w] & ~__ldcg(reinterpret_cast<const unsigned long long*>(r + w));
        if (add)
            atomicOr(reinterpret_cast<unsigned long long*>(r + w), add);
    }
}

/// The emitted tuples of candidate i that it owns go into the table (when `commit`), and their slots are emptied: only
/// a slot's owner touches it (the others only compare its value with their own index), so the map is empty after the
/// kernel without a pass over the map.
__device__ __forceinline__ void commit_owned(const Compact& m, u32 i, bool commit)
{
    const Emission& e = m.emission;
    const Table& t = m.table;
    for (u32 k = e.offset[i]; k < e.offset[i + 1]; ++k)
    {
        const u32 slot = e.slot[k];
        if (__ldcg(m.map.vals + slot) != i)
            continue;
        if (commit)
        {
            const u64 key = m.map.keys[slot];
            const u32 id = static_cast<u32>(key >> (2 * k_key_atom_bits));
            const u32 lo = static_cast<u32>(key >> k_key_atom_bits) & ((1u << k_key_atom_bits) - 1);
            const u32 hi = static_cast<u32>(key) & ((1u << k_key_atom_bits) - 1);
            atomicOr(table_word(t, id, lo, hi), 1ull << (hi & 63));
            if (t.arity == 2 && lo != hi)
                atomicOr(table_word(t, id, hi, lo), 1ull << (lo & 63));
        }
        m.map.keys[slot] = k_empty_key;
        m.map.vals[slot] = k_none;
    }
}

__global__ void k_reached(Gathered g, Candidates c, Searches s)
{
    const u64 n_live = live(c);
    MIW_FOR(i, n_live)
    {
        if (!(c.flags[i] & k_cand_kept))
            continue;
        add_reached(s.reached, s.reached_words, g.search[c.parent[i]], cand_row(c, i, g.words), g.words);
    }
}

__global__ void k_compact(Nodes nodes, Gathered g, Candidates c, Compact m)
{
    const u32 W = nodes.words, LW = nodes.label_words;
    const bool exact = m.emission.slot != nullptr;
    // an earlier abort: nothing to do (no tuple was emitted). Exact novelty's chunks set no k_abort_space before this
    // kernel, so its own bit (which another block may have set already) does not count.
    const u32 ab = c.abort ? *c.abort : 0;
    if (exact ? (ab & ~u32{k_abort_space}) != 0 : ab != 0)
        return;
    // the chunk's nodes and entries must fit (a device loop's room is its expectation, not the capacity's bound):
    // else every block sees the same totals and the chunk aborts (the host grows and redoes it), its owners only
    // emptying their slots of the map
    const u64 k = m.kept[c.n];
    const bool space =
        u64{m.step->node_base} + lo32(k) > m.step->node_cap || u64{m.step->entry_base} + hi32(k) > m.step->next_cap;
    if (space && blockIdx.x == 0 && threadIdx.x == 0 && m.abort)
        atomicOr(m.abort, k_abort_space);
    if (space && !exact)
        return;
    const u64 n_live = c.n_dev ? (*c.n_dev < c.n ? *c.n_dev : c.n) : c.n;  // (not live(c): the abort may be this kernel's)
    MIW_FOR(i, n_live)
    {
        const u8 f = c.flags[i];
        const bool kept = f & k_cand_kept;
        if (exact)
            commit_owned(m, static_cast<u32>(i), kept && !space);
        if (space || !kept)
            continue;
        const u32 p = c.parent[i];
        const u64 at = cand_at(c, i);
        if (m.reached)
            add_reached(m.reached, m.reached_words, g.search[p], c.rows + at * g.words, g.words);
        if (!(f & (k_cand_adm | k_cand_ent)))
            continue;
        u32 nid = k_none;
        if (f & k_cand_adm)
        {
            nid = m.step->node_base + lo32(m.kept[i]);
            const u64* x = c.rows + at * g.words;
            u64* y = nodes.rows + u64{nid} * W;
            for (u32 w = 0; w < W; ++w)
                y[w] = w < g.words ? x[w] : 0;
            nodes.parent[nid] = g.node[p];
            nodes.search[nid] = g.search[p];
            u32* l = nodes.label + u64{nid} * LW;
            l[0] = c.schema[at];
            for (u32 j = 1; j < LW; ++j)
                l[j] = j - 1 < c.label_width ? c.binding[at * c.label_width + (j - 1)] : k_none;
            nodes.flags[nid] = (f & k_cand_skip) ? k_flag_skip : 0;
        }
        if (f & k_cand_ent)
            m.step->next[m.step->entry_base + hi32(m.kept[i])] =
                nid != k_none ? nid : m.step->node_base + lo32(m.kept[m.root_owner[i]]);
    }
}

// ------------------------------------------------------------------------------------------------ the chunk's end

/// A control word other blocks of the kernel wrote (read through L2).
__device__ __forceinline__ u32 ctl_word(const u32* ctl, u32 k) { return __ldcg(ctl + k); }

/// The end of a chunk in a device loop (the last block of k_search_update, one thread): see ChunkEnd.
__device__ void advance(Step& st, Loop& l, const u32* ctl, unsigned long long handle)
{
    const double rows = st.rows ? static_cast<double>(st.rows) : 1.0;
    const u32 ab = ctl_word(ctl, k_ctl_abort);
    const u32 n_ucand = ctl_word(ctl, k_ctl_ucand), n_cand = ctl_word(ctl, k_ctl_cand), n_emit = ctl_word(ctl, k_ctl_emit);
    // the counts of the stages that ran (an aborted chunk's are exact up to its abort: the host learns from them)
    l.ucand = fmax(l.ucand, n_ucand / rows);
    const u32 n_distinct = ctl_word(ctl, k_ctl_distinct);
    l.max_distinct = n_distinct > l.max_distinct ? n_distinct : l.max_distinct;
    if (rows >= k_ratio_rows)
        l.distinct_rows = fmax(l.distinct_rows, n_distinct / rows);
    l.cand_rows = fmax(l.cand_rows, n_cand / rows);
    if (l.emission && (ab & ~u32{k_abort_emit | k_abort_space}) == 0)
    {
        l.emit = fmax(l.emit, n_emit / rows);
        l.max_emit = n_emit > l.max_emit ? n_emit : l.max_emit;
    }
    u32 stop = k_loop_running;
    if (ab)
        stop = k_loop_abort;  // (the step stays: the host redoes the chunk)
    else
    {
        const u32 n_adm = ctl_word(ctl, k_ctl_adm), n_ent = ctl_word(ctl, k_ctl_ent);
        ++l.chunks;
        l.candidates += n_cand;
        l.distinct += ctl_word(ctl, k_ctl_distinct);
        l.nodes += n_adm;
        st.node_base += n_adm;
        st.entry_base += n_ent;
        st.begin += st.rows;
        if (ctl_word(ctl, k_ctl_err))
            stop = k_loop_err;
        else if (st.last)
        {
            ++l.layers;
            l.running = ctl_word(ctl, k_ctl_running);
            st.entries = st.entry_base;
            st.entry_base = 0;
            st.begin = 0;
            ++st.layer;
            u32* t = st.cur;
            st.cur = st.next;
            st.next = t;
            if (l.running == 0 || st.entries == 0)
                stop = k_loop_done;
            else if (st.entries > l.grow_above)
                stop = k_loop_grow;
        }
        const u32 left = st.entries - st.begin;
        st.rows = left < l.rows ? left : l.rows;
        st.last = st.begin + st.rows >= st.entries ? 1u : 0u;
    }
    l.stop = stop;
    if (stop != k_loop_running)
        cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(handle), 0);
}

__global__ void k_search_update(Gathered g, Candidates c, Searches s, Limits lim, Chunk ch, const u64* kept, ChunkEnd end)
{
    const Step& step = *ch.step;
    if (blockIdx.x == 0 && threadIdx.x == 0 && ch.ctl)
    {
        const u64 k = kept[c.n];  // (the scans' totals)
        ch.ctl[k_ctl_adm] = lo32(k);
        ch.ctl[k_ctl_ent] = hi32(k);
    }
    if (!aborted(ch.abort))  // (no kernel of the chunk aborts it from here on: block-uniform)
    {
        const bool layer_end = step.last != 0;
        u32* order = step.next;
        MIW_FOR(id, s.count)
        {
            const u32 fl = s.first_live[id];
            if (fl != k_none)
            {
                const u64 nl = live_rank(ch, s.last_live[id] + 1, fl);
                const u32 fgoal = s.first_goal[id];
                const u64 gi = fgoal == k_none ? ~0ull : live_rank(ch, fgoal, fl);
                const unsigned long long e = s.expanded[id];
                const u64 bi = lim.max_expanded > e ? lim.max_expanded - e : 0;
                const u64 stop = gi < bi ? gi : bi;
                const u32 fg = s.first_gen[id];
                const u32 cut = s.cut[id];
                u64 gen = 0, adm = 0, ent = 0;
                if (fg != k_none)
                {
                    const u32 fc = c.seg_offsets[u64{fg} * c.num_schemas];
                    const u32 ec = c.seg_offsets[(u64{s.last_gen[id]} + 1) * c.num_schemas];
                    const u32 end_c = cut == k_none ? ec : (cut >> 1) + 1;
                    gen = end_c - fc;
                    adm = lo32(kept[ec]) - lo32(kept[fc]);
                    ent = hi32(kept[ec]) - hi32(kept[fc]);
                    if (ent)
                    {
                        // the search's entries of the next layer: contiguous across the layer's chunks
                        const u32 at = step.entry_base + hi32(kept[fc]);
                        if (s.seg_begin[id] == s.seg_end[id])
                            s.seg_begin[id] = at;
                        s.seg_end[id] = at + static_cast<u32>(ent);
                    }
                }
                s.generated[id] += gen;
                s.in_tree[id] += ent;  // width 0 counts duplicate entries too (as in mimir's generated_in_tree)
                s.tree[id] += static_cast<u32>(adm);
                s.next[id] += static_cast<u32>(ent);
                // expanded rows: a prefix of the search's live rows (to its cut's row); skipped: those of them that
                // did not generate (the generating rows are the search's gathered rows, contiguous from first_gen)
                u64 expanded = 0, generating = 0;
                if (cut != k_none)
                {
                    const u32 p = c.parent[cut >> 1];
                    const u32 row = g.crow[p];
                    s.cut_row[id] = row;
                    expanded = live_rank(ch, row, fl) + 1;
                    generating = p - fg + 1;
                    if (cut & 1)
                        s.layer_cut[id] = 1;
                    else
                        s.status[id] = k_out_of_states;
                }
                else
                {
                    expanded = stop < nl ? stop : nl;
                    generating = fg == k_none ? 0 : s.last_gen[id] - fg + 1;
                    if (gi != ~0ull && gi <= bi)
                    {
                        s.status[id] = k_solved;
                        s.goal_node[id] = step.cur[step.begin + fgoal];
                        s.goal_depth[id] = step.layer;
                    }
                    else if (bi < nl)
                        s.status[id] = k_out_of_states;
                }
                s.expanded[id] += expanded;
                s.skipped[id] += expanded - generating;
            }
            if (layer_end)
            {
                s.next[id] = 0;
                s.layer_cut[id] = 0;
                if (s.status[id] != k_running)
                    continue;
                const u32 b = s.seg_begin[id], e = s.seg_end[id];
                if (b == e)
                {
                    s.status[id] = k_exhausted;
                    continue;
                }
                if (end.shuffle)
                    device_rng::shuffle(s.rng[id], order + b, e - b);
                if (end.close_duplicates)
                    for (u32 i = b; i < e; ++i)
                    {
                        const u32 node = order[i];
                        if (end.nodes.flags[node] & k_flag_closed)
                            order[i] = k_dead;
                        else
                            end.nodes.flags[node] |= k_flag_closed;
                    }
                atomicAdd(ch.ctl + k_ctl_running, 1u);
            }
        }
    }
    if (!end.loop)
        return;
    // a device loop: the last block to finish advances the step (every block's counts are in the control block)
    __shared__ bool s_last;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        s_last = atomicAdd(ch.ctl + k_ctl_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (s_last && threadIdx.x == 0)
    {
        __threadfence();
        advance(*end.step, *end.loop, ch.ctl, end.handle);
    }
}

// ------------------------------------------------------------------------------------------------ results

__global__ void k_plans(Nodes nodes, Searches s, PlanOut p)
{
    const u32 LW = nodes.label_words;
    MIW_FOR(i, p.n)
    {
        const u32 id = p.which[i];
        u32 v = s.goal_node[id];
        const u64* src = nodes.rows + u64{v} * nodes.words;
        u64* dst = p.goal_rows + i * p.goal_words;
        for (u32 w = 0; w < p.goal_words; ++w)
            dst[w] = w < nodes.words ? src[w] : 0;
        const u64 off = p.plan_offset[i];
        for (u32 k = s.goal_depth[id]; k-- > 0;)
        {
            for (u32 j = 0; j < LW; ++j)
                p.labels[(off + k) * LW + j] = nodes.label[u64{v} * LW + j];
            v = nodes.parent[v];
        }
    }
}

__global__ void k_plan_states(Nodes nodes, Searches s, PlanOut p, Gathered g)
{
    MIW_FOR(i, p.n)
    {
        const u32 id = p.which[i];
        u32 v = s.goal_node[id];
        const u64 off = p.plan_offset[i];
        for (u32 k = s.goal_depth[id]; k-- > 0;)
        {
            v = nodes.parent[v];
            const u64* src = nodes.rows + u64{v} * nodes.words;
            u64* dst = g.rows + (off + k) * g.words;
            for (u32 w = 0; w < g.words; ++w)
                dst[w] = w < nodes.words ? src[w] : 0;
            g.search[off + k] = id;
        }
    }
}

// ------------------------------------------------------------------------------------------------ scans

/// Scan inputs (live_scan::k_scan): the live count and the value of item i.
struct FlagBit
{
    const u8* flags;
    u8 mask;
    live_scan::Live l;
    __device__ __forceinline__ u64 live(u64 n) const { return l.count(n); }
    __device__ __forceinline__ u64 operator()(u64 i) const { return (flags[i] & mask) == mask ? 1u : 0u; }
};

struct FlagPair
{
    const u8* flags;
    u8 lo, hi;
    live_scan::Live l;
    __device__ __forceinline__ u64 live(u64 n) const { return l.count(n); }
    __device__ __forceinline__ u64 operator()(u64 i) const
    {
        const u8 f = flags[i];
        return ((f & lo) == lo ? 1ull : 0ull) | ((f & hi) == hi ? 1ull << 32 : 0ull);
    }
};

struct ValueU32
{
    const u32* in;
    live_scan::Live l;
    __device__ __forceinline__ u64 live(u64 n) const { return l.count(n); }
    __device__ __forceinline__ u64 operator()(u64 i) const { return in[i]; }
};

/// The candidates of gathered row i: its distinct row's.
struct DedupCount
{
    const u32* map;
    const u32* seg;
    u32 S;
    live_scan::Live l;
    __device__ __forceinline__ u64 live(u64 n) const { return l.count(n); }
    __device__ __forceinline__ u64 operator()(u64 i) const
    {
        const u64 u = map[i];
        return seg[(u + 1) * S] - seg[u * S];
    }
};

/// Novelty and admission of candidate i (k_novel_flags's flags, stored), as kept-scan pairs.
struct NovelKept
{
    Candidates c;
    Emission e;
    OwnerMap m;
    Admission a;
    __device__ __forceinline__ u64 live(u64) const { return live_candidates(c); }
    __device__ __forceinline__ u64 operator()(u64 i) const
    {
        bool novel = false;
        for (u32 k = e.offset[i]; k < e.offset[i + 1] && !novel; ++k)
            novel = m.vals[e.slot[k]] == static_cast<u32>(i);
        const u8 f = admitted(novel ? k_cand_novel : 0, a);
        c.flags[i] = f;
        constexpr u8 lo = k_cand_kept | k_cand_adm, hi = k_cand_kept | k_cand_ent;
        return ((f & lo) == lo ? 1ull : 0ull) | ((f & hi) == hi ? 1ull << 32 : 0ull);
    }
};

live_scan::State scan_state(ScanState st) { return live_scan::state_at(st.words, st.tiles, k_scan_sites, st.site); }
}  // namespace

// ------------------------------------------------------------------------------------------------ launchers

cudaError_t launch_init_pass(Nodes nodes, Searches s, Roots r, u32* order, cudaStream_t st)
{
    if (s.count)
        k_reset_searches<<<grid_for(s.count), k_block, 0, st>>>(s);
    if (r.n)
        k_init_roots<<<grid_for(r.n), k_block, 0, st>>>(nodes, s, r, order);
    return cudaGetLastError();
}

cudaError_t launch_table_init(Table t, Nodes nodes, u32 roots, cudaStream_t st)
{
    if (roots && t.row_words)
    {
        const u64 n = t.arity == 1 ? u64{roots} * t.row_words : u64{roots} * 64 * t.row_words;
        k_table_init<<<grid_for(n), k_block, 0, st>>>(t, nodes, roots);
    }
    return cudaGetLastError();
}

cudaError_t launch_stop_running(Searches s, u32 status, cudaStream_t st)
{
    if (s.count)
        k_stop_running<<<grid_for(s.count), k_block, 0, st>>>(s, status);
    return cudaGetLastError();
}

cudaError_t launch_table_relayout(const u64* src, u64* dst, u32 arity, u32 searches, u32 from, u32 to, cudaStream_t st)
{
    const u64 n = u64{searches} * (arity == 1 ? 1 : u64{64} * to) * to;
    if (n)
        k_table_relayout<<<grid_for(n), k_block, 0, st>>>(src, dst, arity, searches, from, to);
    return cudaGetLastError();
}

cudaError_t launch_rows_live(const rl::dev::TaskView& task, Nodes nodes, Searches s, Chunk c, cudaStream_t st)
{
    if (c.rows)
        k_rows_live<<<grid_for(c.rows), k_block, 0, st>>>(task, nodes, s, c);
    return cudaGetLastError();
}

cudaError_t launch_rows_select(Nodes nodes, Searches s, Limits lim, Chunk c, cudaStream_t st)
{
    if (c.rows)
        k_rows_select<<<grid_for(c.rows), k_block, 0, st>>>(nodes, s, lim, c);
    return cudaGetLastError();
}

cudaError_t launch_gather(Nodes nodes, Searches s, Chunk c, Gathered g, cudaStream_t st)
{
    if (c.rows)
        k_gather<<<grid_for(c.rows), k_block, 0, st>>>(nodes, s, c, g);
    return cudaGetLastError();
}

cudaError_t launch_emit_count(Gathered g, Candidates c, Table t, Emission e, cudaStream_t st)
{
    if (c.n)
        k_emit_count<<<grid_for(c.n), k_block, 0, st>>>(g, c, t, e);
    return cudaGetLastError();
}

cudaError_t launch_emit_insert(Gathered g, Candidates c, Table t, Emission e, OwnerMap m, cudaStream_t st)
{
    if (c.n)
        k_emit_insert<<<grid_for(c.n), k_block, 0, st>>>(g, c, t, e, m);
    return cudaGetLastError();
}

cudaError_t launch_novel_flags(Candidates c, Emission e, OwnerMap m, Admission a, cudaStream_t st)
{
    if (c.n)
        k_novel_flags<<<grid_for(c.n), k_block, 0, st>>>(c, e, m, a);
    return cudaGetLastError();
}

cudaError_t launch_relaxed(Gathered g, Candidates c, Table t, const Step* step, u32* abort, Admission a, cudaStream_t st)
{
    if (c.n)
        k_relaxed<<<grid_for(c.n), k_block, 0, st>>>(g, c, t, step, abort, a);
    return cudaGetLastError();
}

cudaError_t launch_root_insert(Gathered g, Candidates c, RootSet r, cudaStream_t st)
{
    if (c.n)
        k_root_insert<<<grid_for(c.n), k_block, 0, st>>>(g, c, r);
    return cudaGetLastError();
}

cudaError_t launch_root_flags(Gathered, Candidates c, RootSet r, cudaStream_t st)
{
    if (c.n)
        k_root_flags<<<grid_for(c.n), k_block, 0, st>>>(c, r);
    return cudaGetLastError();
}

cudaError_t launch_admit(Candidates c, Admission a, cudaStream_t st)
{
    if (c.n)
        k_admit<<<grid_for(c.n), k_block, 0, st>>>(c, a);
    return cudaGetLastError();
}

cudaError_t launch_cut(Gathered g, Candidates c, Searches s, Limits lim, const u64* scan, cudaStream_t st)
{
    if (c.n && (lim.max_states != k_none || lim.max_next != k_none))
        k_cut<<<grid_for(c.n), k_block, 0, st>>>(g, c, s, lim, scan);
    return cudaGetLastError();
}

cudaError_t launch_keep(Gathered g, Candidates c, Searches s, Limits lim, cudaStream_t st)
{
    if (c.n)
        k_keep<<<grid_for(c.n), k_block, 0, st>>>(g, c, s, lim);
    return cudaGetLastError();
}

cudaError_t launch_search_update(Gathered g, Candidates c, Searches s, Limits lim, Chunk ch, const u64* kept, ChunkEnd e,
                                 cudaStream_t st)
{
    if (s.count)
        k_search_update<<<grid_for(s.count), k_block, 0, st>>>(g, c, s, lim, ch, kept, e);
    return cudaGetLastError();
}

cudaError_t launch_reached(Gathered g, Candidates c, Searches s, cudaStream_t st)
{
    if (c.n && s.reached)
        k_reached<<<grid_for(c.n), k_block, 0, st>>>(g, c, s);
    return cudaGetLastError();
}

cudaError_t launch_compact(Nodes nodes, Gathered g, Candidates c, Compact m, cudaStream_t st)
{
    if (c.n)
        k_compact<<<grid_for(c.n), k_block, 0, st>>>(nodes, g, c, m);
    return cudaGetLastError();
}

u64 scan_tiles(u64 n) { return live_scan::tiles_for(n); }
u64 scan_buffer_words(u64 tiles) { return live_scan::buffer_words(tiles, k_scan_sites); }
u64 scan_flag_words(u64 tiles) { return live_scan::flag_words(tiles, k_scan_sites); }

cudaError_t launch_scan_flags(const u8* flags, u8 mask, u64 n, const u32* n_dev, const u32* abort, u32* out, ScanState st,
                              cudaStream_t s)
{
    return live_scan::launch(FlagBit{flags, mask, {n_dev, abort}}, n, out, scan_state(st), s);
}

cudaError_t launch_scan_u32(const u32* in, u64 n, const u32* n_dev, const u32* abort, u32* out, ScanState st, cudaStream_t s)
{
    return live_scan::launch(ValueU32{in, {n_dev, abort}}, n, out, scan_state(st), s);
}

cudaError_t launch_scan_flag_pair(const u8* flags, u8 mask_lo, u8 mask_hi, u64 n, const u32* n_dev, const u32* abort, u64* out,
                                  ScanState st, cudaStream_t s)
{
    return live_scan::launch(FlagPair{flags, mask_lo, mask_hi, {n_dev, abort}}, n, out, scan_state(st), s);
}

cudaError_t launch_novel_kept(Candidates c, Emission e, OwnerMap m, Admission a, u64* kept, ScanState st, cudaStream_t s)
{
    if (!a.admit || !a.keep)
        return cudaErrorInvalidValue;
    return live_scan::launch(NovelKept{c, e, m, a}, c.n, kept, scan_state(st), s);
}

cudaError_t launch_dedup_offsets(Gathered g, Dedup d, const u32* u_seg, u32 num_schemas, u32* offsets, ScanState st,
                                 cudaStream_t s)
{
    // (an aborted chunk's map may point past the generator's rows: no item then)
    return live_scan::launch(DedupCount{d.map, u_seg, num_schemas, {g.n_dev, d.ctl ? d.ctl + k_ctl_abort : nullptr}}, g.n, offsets,
                             scan_state(st), s);
}

cudaError_t launch_plans(Nodes nodes, Searches s, PlanOut p, cudaStream_t st)
{
    if (p.n)
        k_plans<<<grid_for(p.n), k_block, 0, st>>>(nodes, s, p);
    return cudaGetLastError();
}

cudaError_t launch_plan_states(Nodes nodes, Searches s, PlanOut p, Gathered g, cudaStream_t st)
{
    if (p.n)
        k_plan_states<<<grid_for(p.n), k_block, 0, st>>>(nodes, s, p, g);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------ control

namespace
{
/// The check of a count against its capacity (one thread).
__device__ __forceinline__ void run_check(const Check& c)
{
    const u32 n = *c.count;
    c.ctl[c.slot] = n;
    if (n > c.capacity)
        atomicOr(c.ctl + k_ctl_abort, c.bit);
}

__global__ void k_check(Check c) { run_check(c); }

__global__ void k_chunk_begin(Searches s, u32* ctl, const Step* step, u64* scan_flags, u64 scan_words)
{
    const bool segments = step->begin == 0;  // (the layer's first chunk: its entries of the next layer start)
    const u64 n = s.count > scan_words ? s.count : scan_words;
    MIW_FOR(i, n > k_ctl_words ? n : k_ctl_words)
    {
        if (i < k_ctl_words)
            ctl[i] = 0;
        if (i < scan_words)
            scan_flags[i] = 0;
        if (i >= s.count)
            continue;
        s.first_live[i] = s.first_goal[i] = s.first_gen[i] = s.cut[i] = s.cut_row[i] = k_none;
        s.last_live[i] = s.last_gen[i] = 0;
        if (segments)
            s.seg_begin[i] = s.seg_end[i] = 0;
    }
}

// ------------------------------------------------------------------------------------------------ dedup

__device__ __forceinline__ u64 row_hash(const u64* w, u32 n)
{
    u64 h = 0x243F6A8885A308D3ull;
    for (u32 i = 0; i < n; ++i)
        h = device_rng::mix64(h ^ w[i]) + i;
    return h;
}

/// The slot of row r's content (inserting r when the content is new); the slot keeps the smallest row (atomicMin).
__device__ __forceinline__ u64 dedup_slot(const Gathered& g, const Dedup& d, u32 r, bool insert)
{
    const u32 W = g.words;
    const u64* x = g.rows + u64{r} * W;
    for (u64 j = row_hash(x, W) & d.mask;; j = (j + 1) & d.mask)
    {
        u32 v = __ldcg(d.table + j);
        if (v == k_none)
        {
            if (!insert)
                continue;  // (cannot happen: every row was inserted)
            const u32 old = atomicCAS(d.table + j, k_none, r);
            if (old == k_none)
                return j;
            v = old;
        }
        if (rows_equal(x, g.rows + u64{v} * W, W))
        {
            // (the slot's row only decreases: a row above the one seen leaves it; this keeps the rollouts' thousands
            // of copies of a state from otherwise serializing on its slot)
            if (insert && r < v)
                atomicMin(d.table + j, r);
            return j;
        }
    }
}

__global__ void k_dedup_insert(Gathered g, Dedup d)
{
    const u64 n = live(g);
    MIW_FOR(r, n)
    (void)dedup_slot(g, d, static_cast<u32>(r), true);
}

__global__ void k_dedup_owner(Gathered g, Dedup d)
{
    const u64 n = live(g);
    MIW_FOR(r, n)
    {
        const u64 j = dedup_slot(g, d, static_cast<u32>(r), false);
        const u32 owner = d.table[j];
        d.owner[r] = owner;
        d.flags[r] = owner == static_cast<u32>(r) ? 1 : 0;
        if (owner == static_cast<u32>(r))
            d.map[r] = static_cast<u32>(j);  // (its slot, for dedup_compact to empty)
    }
}

__global__ void k_dedup_compact(Gathered g, Dedup d)
{
    const u32 distinct = d.rank[g.n];
    const bool over = distinct > d.rows_capacity;
    if (blockIdx.x == 0 && threadIdx.x == 0 && d.ctl)
    {
        d.ctl[k_ctl_distinct] = distinct;
        d.ctl[k_ctl_gen_rows] = over ? 0u : distinct;
        if (over)
            atomicOr(d.ctl + k_ctl_abort, k_abort_distinct);
    }
    const u64 n = live(g);
    const u32 W = g.words;
    MIW_FOR(r, n)
    {
        const u32 owner = d.owner[r];
        const u32 u = d.rank[owner];
        if (owner == static_cast<u32>(r))
        {
            d.table[d.map[r]] = k_none;  // (the set is empty again after the chunk)
            if (!over)
                for (u32 w = 0; w < W; ++w)
                    d.rows[u64{u} * W + w] = g.rows[r * W + w];
        }
        d.map[r] = u;
    }
}

__global__ void k_broadcast(Gathered g, Dedup d, Broadcast b)
{
    // an earlier abort (more distinct rows than the generator's, k_abort_distinct): the scan of the offsets wrote
    // no total, nothing is counted (only block 0 of this kernel aborts it, after this read)
    if (aborted(b.abort))
        return;
    // the chunk's checks of the distinct rows' candidates and of the candidates: over a capacity, it aborts (every
    // block sees the totals)
    const u64 rows = live(g);
    const u64 total = b.row_offsets[rows];
    const u32 nu = b.ucand ? *b.ucand : 0u;
    const bool over = (b.ucand && nu > b.ucand_capacity) || total > b.capacity;
    if (blockIdx.x == 0 && threadIdx.x == 0 && b.ctl)
    {
        if (b.ucand)
        {
            b.ctl[k_ctl_ucand] = nu;
            if (nu > b.ucand_capacity)
                atomicOr(b.ctl + k_ctl_abort, k_abort_ucand);
        }
        b.ctl[k_ctl_cand] = static_cast<u32>(total);
        if (total > b.capacity)
            atomicOr(b.ctl + k_ctl_abort, k_abort_cand);
    }
    if (over || aborted(b.abort))
        return;
    // a warp per 32 gathered rows, its lanes over their candidates in order (a warp per row leaves most lanes idle
    // at rollouts' typical few candidates per row): candidate j is in the last of the 32 rows that starts at
    // or before it (a row without candidates starts where the next one does), found by a binary search over the
    // lanes' starts; the candidates of a row are its distinct row's, in order, and stay there (Candidates::src:
    // copying rows and labels per candidate would be most of the chunk's memory traffic)
    const u32 W = g.words;
    const u32 lane = threadIdx.x & 31;
    const u64 warps = (u64{gridDim.x} * blockDim.x) >> 5;
    for (u64 r0 = spread_warp() << 5; r0 < rows; r0 += warps << 5)
    {
        const u64 r = r0 + lane;
        const u64 r_end = r0 + 32 < rows ? r0 + 32 : rows;
        const u32 start = b.row_offsets[r < rows ? r : rows];  // (past the rows: the total, after every candidate)
        const u32 first = __shfl_sync(k_full_warp, start, 0), end = b.row_offsets[r_end];
        u64 src0 = 0;
        u32 id = 0;
        if (r < rows)
        {
            src0 = b.seg[u64{d.map[r]} * b.num_schemas];
            id = g.search[r];
        }
        for (u32 base = first; base < end; base += 32)
        {
            const u32 j = base + lane;
            u32 k = 0;
            for (u32 step = 16; step > 0; step >>= 1)
                if (__shfl_sync(k_full_warp, start, k + step) <= j)
                    k += step;
            const u64 row_src0 = __shfl_sync(k_full_warp, src0, k);
            const u32 row_id = __shfl_sync(k_full_warp, id, k), row_start = __shfl_sync(k_full_warp, start, k);
            if (j >= end)
                continue;
            const u64 row = r0 + k, src = row_src0 + (j - row_start);
            b.out_parent[j] = static_cast<u32>(row);
            b.out_src[j] = static_cast<u32>(src);
            if (b.emit_count)
            {
                u32 n = 0;
                for_each_unseen(b.table, row_id, g.rows + row * W, b.rows + src * W, [&](u32, u32) { ++n; });
                b.emit_count[j] = n;
            }
        }
    }
}
}  // namespace

cudaError_t launch_check(Check c, cudaStream_t st)
{
    k_check<<<1, 1, 0, st>>>(c);
    return cudaGetLastError();
}

cudaError_t launch_chunk_begin(Searches s, u32* ctl, const Step* step, ScanState scans, cudaStream_t st)
{
    const u64 words = scans.words ? live_scan::flag_words(scans.tiles, k_scan_sites) : 0;
    u64 n = s.count > words ? s.count : words;
    n = n > k_ctl_words ? n : u64{k_ctl_words};
    k_chunk_begin<<<grid_for(n), k_block, 0, st>>>(s, ctl, step, scans.words, words);
    return cudaGetLastError();
}

cudaError_t launch_dedup_insert(Gathered g, Dedup d, cudaStream_t st)
{
    if (g.n)
        k_dedup_insert<<<grid_for(g.n), k_block, 0, st>>>(g, d);
    return cudaGetLastError();
}

cudaError_t launch_dedup_owner(Gathered g, Dedup d, cudaStream_t st)
{
    if (g.n)
        k_dedup_owner<<<grid_for(g.n), k_block, 0, st>>>(g, d);
    return cudaGetLastError();
}

cudaError_t launch_dedup_compact(Gathered g, Dedup d, cudaStream_t st)
{
    if (g.n)
        k_dedup_compact<<<grid_for(g.n), k_block, 0, st>>>(g, d);
    return cudaGetLastError();
}

cudaError_t launch_broadcast(Gathered g, Dedup d, Broadcast b, cudaStream_t st)
{
    if (b.capacity && g.n)
        k_broadcast<<<grid_for(u64{g.n} * 32), k_block, 0, st>>>(g, d, b);
    return cudaGetLastError();
}
}  // namespace mymyr::cuda::miw
