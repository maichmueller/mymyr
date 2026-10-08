// The batched grounded heuristics (include/mymyr/cuda/heuristics_kernels.hpp): one group of threads (a block or a
// warp) per state; the costs by Jacobi sweeps of the relaxed Bellman-Ford update or by the bucketed frontier (a
// generalized Dijkstra: precondition counters, the cheapest unsettled propositions settled per round), then h_FF's
// supporter levels (none for uniform costs: the smallest tight achievers) and the relaxed plan. Compiled by nvcc as
// C++20; includes only the device-code subset.

#include "mymyr/cuda/heuristics_kernels.hpp"

namespace mymyr::cuda::hk
{
namespace
{
constexpr u32 k_max_groups_per_block = 32;

/// Counters of a group (static shared memory). Lists and minima are double-buffered by round parity (the extraction
/// frontier triple-buffered): a phase reads one counter and appends to another, and thread 0 of the group clears one
/// that nobody uses in that phase, so no counter is read and written in the same phase.
struct GroupState
{
    u32 nT[2];  // propositions changed / settled / newly leveled in a round
    u32 nF[3];  // extraction frontier
    u32 changed[2];
    u32 bmin[2];  // the frontier's bucket: the cheapest unsettled cost of a round
    u32 flag;     // k_outside / k_invalid bits of the state
    u32 dead;
    u32 hmax;
    u32 left;       // goal propositions without a level (h_FF), or not settled (the frontier's costs)
    u32 goal_cost;  // the frontier's costs: the bucket in which the last goal settled
    unsigned long long hsum;
};

/// The threads that evaluate one state together: the block, or one warp.
template<bool WARP>
struct Group
{
    u32 tid, size;
    __device__ Group()
    {
        if constexpr (WARP)
        {
            tid = threadIdx.x & 31u;
            size = 32;
        }
        else
        {
            tid = threadIdx.x;
            size = blockDim.x;
        }
    }
    __device__ __forceinline__ void sync() const
    {
        if constexpr (WARP)
            __syncwarp();
        else
            __syncthreads();
    }
};

/// One group's scratch (u32 arrays; the *bits arrays are bitsets).
struct Arrays
{
    u32* cost;
    u32* nxt;   // sweeps: the next round's costs; frontier: the settled flags
    u32* T;
    u32* T2;    // frontier: the second level list
    u32* left;  // frontier: unsettled (unleveled) precondition entries per operator
    u32* lvl;
    u32* cand;
    u32* supp;
    u32* pbits;
    u32* gbits;
};

__device__ __forceinline__ u32 words32(u32 n) { return (n + 31) / 32; }

/// Sets bit x of `bits`; true if it was clear (the caller is the one that set it).
__device__ __forceinline__ bool test_and_set(u32* bits, u32 x)
{
    const u32 b = 1u << (x & 31u);
    return (atomicOr(bits + (x >> 5), b) & b) == 0;
}

template<u32 KIND, u32 VARIANT>
__device__ Arrays carve(u32* base, const Relaxed& r)
{
    Arrays a{};
    u32* p = base;
    a.cost = p;
    p += r.P;
    a.nxt = p;
    p += r.P;
    a.T = p;
    p += r.P;
    if constexpr (VARIANT == k_frontier)
    {
        a.T2 = p;
        p += r.P;
        a.left = p;
        p += r.O;
    }
    if constexpr (KIND == k_ff || KIND == k_set_additive)
    {
        if (!r.uniform_cost)
        {
            a.lvl = p;
            p += r.P;
            a.cand = p;
            p += r.P;
            a.supp = p;
            p += r.P;
        }
        a.pbits = p;
        p += words32(r.P);
        a.gbits = p;
    }
    return a;
}

__device__ __forceinline__ u32 sat(u64 v) { return v >= k_inf ? k_inf - 1 : static_cast<u32>(v); }

/// The operator's value on the current costs (k_inf if a precondition is infinite).
template<u32 KIND>
__device__ __forceinline__ u32 op_value(const Relaxed& r, const u32* cost, u32 o)
{
    const u32 b = r.pre_begin[o], e = r.pre_begin[o + 1];
    if (KIND == k_add && !r.axiom[o])
    {
        u64 s = r.opcost[o];
        for (u32 i = b; i < e; ++i)
        {
            const u32 c = cost[r.pre[i]];
            if (c == k_inf)
                return k_inf;
            s += c;
        }
        return sat(s);
    }
    u32 m = 0;
    for (u32 i = b; i < e; ++i)
    {
        const u32 c = cost[r.pre[i]];
        if (c == k_inf)
            return k_inf;
        m = c > m ? c : m;
    }
    return sat(u64{m} + r.opcost[o]);
}

/// The initial costs (and levels for h_FF): "p false" costs 0, everything else infinity, then the state's fluent bits
/// ("p true" costs 0, "p false" infinity). Sets gs.flag for states outside the grounding.
template<u32 KIND, bool WARP>
__device__ void init_state(const Relaxed& r, const u64* row, u32 words, const Arrays& a, GroupState& gs,
                           const Group<WARP>& g)
{
    for (u32 p = g.tid; p < r.P; p += g.size)
    {
        const u32 c0 = r.negative[p] ? 0 : k_inf;
        a.cost[p] = c0;
        a.nxt[p] = c0;
        if ((KIND == k_ff || KIND == k_set_additive) && !r.uniform_cost)
        {
            a.lvl[p] = c0;
            a.cand[p] = k_none;
            a.supp[p] = k_none;
        }
    }
    g.sync();
    for (u32 w = g.tid; w < words; w += g.size)
    {
        u64 x = row[w];
        while (x)
        {
            const u32 slot = w * 64 + static_cast<u32>(__ffsll(static_cast<long long>(x)) - 1);
            x &= x - 1;
            if (slot >= r.slots)
            {
                atomicOr(&gs.flag, u32{k_invalid});
                continue;
            }
            const u32 pos = r.slot_pos[slot];
            if (pos == k_none)
            {
                atomicOr(&gs.flag, u32{k_outside});
                continue;
            }
            const u32 neg = r.slot_neg[slot];
            a.cost[pos] = 0;
            a.nxt[pos] = 0;
            if (neg != k_none)
            {
                a.cost[neg] = k_inf;
                a.nxt[neg] = k_inf;
            }
            if ((KIND == k_ff || KIND == k_set_additive) && !r.uniform_cost)
            {
                a.lvl[pos] = 0;
                if (neg != k_none)
                    a.lvl[neg] = k_inf;
            }
        }
    }
}

/// Lowers *slot to the minimum of v over the calling warp (every lane of the warp calls it).
__device__ __forceinline__ void warp_min_into(u32 v, u32* slot)
{
#pragma unroll
    for (u32 off = 16; off > 0; off >>= 1)
    {
        const u32 x = __shfl_xor_sync(0xFFFFFFFFu, v, off);
        v = x < v ? x : v;
    }
    if ((threadIdx.x & 31u) == 0 && v != k_inf)
        atomicMin(slot, v);
}

/// Jacobi sweeps: every round evaluates every operator on the previous round's costs and lowers the next round's with
/// atomicMin, until no cost changes.
template<u32 KIND, bool WARP>
__device__ void costs_sweep(const Relaxed& r, const Arrays& a, GroupState& gs, const Group<WARP>& g)
{
    const u32 tid = g.tid, nt = g.size;
    for (u32 k = 0;; ++k)
    {
        const u32 cur = k & 1, nx = cur ^ 1;
        if (tid == 0)
            gs.changed[nx] = 0;
        for (u32 o = tid; o < r.O; o += nt)
        {
            const u32 v = op_value<KIND>(r, a.cost, o);
            if (v == k_inf)
                continue;
            for (u32 j = r.eff_begin[o]; j < r.eff_begin[o + 1]; ++j)
                if (v < atomicMin(a.nxt + r.eff[j], v))
                    atomicOr(&gs.changed[cur], 1u);
        }
        g.sync();
        if (!gs.changed[cur])
            break;
        for (u32 p = tid; p < r.P; p += nt)
            if (a.nxt[p] != a.cost[p])
                a.cost[p] = a.nxt[p];
        g.sync();
    }
    g.sync();  // every thread has read the last round's flag (the break) before thread 0 resets counters after this
}

/// The bucketed frontier (P4's work-efficient variant; a generalized Dijkstra): each operator keeps a counter of its
/// unsettled precondition entries; a round settles every unsettled proposition of the cheapest cost B, and each
/// settled proposition decrements the counters of the operators it is a precondition of; an operator whose counter
/// reaches 0 has all its preconditions final, fires once with its value (at least B: the costs are non-negative, max
/// and sums are monotone) and lowers its effects with atomicMin. So every operator is evaluated once and the costs are
/// the least fixpoint, as the sweeps', whatever the thread order. The next round's B is the minimum of the costs the
/// settling scan leaves above B and of the values the round's operators fire with (one scan and two group syncs per
/// round). h_max and h_add stop once every goal is settled; h_FF (whose supporters need the costs up to the dearest
/// goal) once the next bucket is dearer than every goal. Unsettled propositions keep a cost that is an upper bound
/// only: a[nxt] marks the settled ones.
template<u32 KIND, bool WARP>
__device__ void costs_frontier(const Relaxed& r, const Arrays& a, GroupState& gs, const Group<WARP>& g)
{
    const u32 tid = g.tid, nt = g.size;
    u32* const done = a.nxt;
    for (u32 p = tid; p < r.P; p += nt)
        done[p] = 0;
    for (u32 o = tid; o < r.O; o += nt)
        a.left[o] = r.pre_begin[o + 1] - r.pre_begin[o];
    if (tid == 0)
    {
        gs.nT[0] = 0;
        gs.nT[1] = 0;
        gs.bmin[0] = k_inf;
        gs.bmin[1] = k_inf;
        gs.left = r.G;
        gs.goal_cost = k_inf;
    }
    g.sync();
    // fire lowers the costs of the unsettled effects and the next bucket (bmin[nx] of the round)
    auto fire = [&](u32 o, u32* next)
    {
        const u32 v = op_value<KIND>(r, a.cost, o);
        if (v == k_inf)
            return;
        bool any = false;
        for (u32 j = r.eff_begin[o]; j < r.eff_begin[o + 1]; ++j)
        {
            const u32 e = r.eff[j];
            if (!done[e] && v < a.cost[e])
            {
                atomicMin(a.cost + e, v);
                any = true;
            }
        }
        if (any)
            atomicMin(next, v);
    };
    // operators without preconditions fire at once (the others among zero_ops have "false" preconditions: counted);
    // the first bucket is the cheapest initial cost
    for (u32 i = tid; i < r.num_zero; i += nt)
    {
        const u32 o = r.zero_ops[i];
        if (r.pre_begin[o] == r.pre_begin[o + 1])
            fire(o, &gs.bmin[0]);
    }
    {
        u32 m = k_inf;
        for (u32 p = tid; p < r.P; p += nt)
            m = a.cost[p] < m ? a.cost[p] : m;
        warp_min_into(m, &gs.bmin[0]);
    }
    g.sync();
    for (u32 k = 0;; ++k)
    {
        const u32 cur = k & 1, nx = cur ^ 1;
        const u32 B = gs.bmin[cur];
        if (B == k_inf)
            break;  // the rest is unreachable
        if ((KIND == k_ff || KIND == k_set_additive) && gs.left == 0 && B > gs.goal_cost)
            break;
        u32 m = k_inf;  // the cheapest cost left above B
        for (u32 p = tid; p < r.P; p += nt)
            if (!done[p])
            {
                const u32 c = a.cost[p];
                if (c == B)
                {
                    done[p] = 1;
                    a.T[atomicAdd(&gs.nT[cur], 1u)] = p;
                    if (r.is_goal[p])
                        atomicSub(&gs.left, 1u);
                }
                else if (c < m)
                    m = c;
            }
        warp_min_into(m, &gs.bmin[nx]);
        g.sync();
        if (KIND != k_ff && KIND != k_set_additive && gs.left == 0)
            break;
        if (tid == 0)
        {
            gs.nT[nx] = 0;
            gs.bmin[cur] = k_inf;  // every thread read it before the sync above; the next round lowers it again
            if (gs.left == 0 && gs.goal_cost == k_inf)
                gs.goal_cost = B;
        }
        // r.pre_of_lanes lanes per settled proposition over the operators that read it (a proposition may be read by
        // thousands: folding's)
        const u32 n = gs.nT[cur], L = r.pre_of_lanes;
        for (u32 i = tid / L; i < n; i += nt / L)
        {
            const u32 p = a.T[i];
            for (u32 j = r.pre_of_begin[p] + tid % L; j < r.pre_of_begin[p + 1]; j += L)
            {
                const u32 o = r.pre_of[j];
                if (atomicSub(a.left + o, 1u) == 1)
                    fire(o, &gs.bmin[nx]);
            }
        }
        g.sync();
    }
    g.sync();  // every thread has read the last round's counters (the breaks) before thread 0 resets them
}

/// Uniform costs: the supporter of p, its smallest tight achiever (k_none for a proposition of cost 0), searched by
/// the L lanes of an aligned group of a warp in chunks of L achievers (ascending ids: the first tight one of the first
/// chunk that has one; every lane gets it). p is settled and costs at most as much as the dearest goal: so do the
/// preconditions of its tight achievers, and an achiever with an unsettled precondition (an upper bound above the
/// goals') is not tight.
__device__ __forceinline__ u32 tight_achiever(const Relaxed& r, const u32* cost, u32 p, u32 L)
{
    const u32 c = cost[p];
    if (c == 0)
        return k_none;
    const u32 lane = threadIdx.x & 31u, first = lane & ~(L - 1);
    const u32 mask = (L == 32 ? 0xFFFFFFFFu : (1u << L) - 1) << first;
    const u32 b = r.ach_begin[p], e = r.ach_begin[p + 1];
    for (u32 base = b; base < e; base += L)
    {
        const u32 j = base + (lane - first);
        const bool tight = j < e && op_value<k_ff>(r, cost, r.ach[j]) == c;
        const u32 bits = L == 1 ? (tight ? 1u : 0u) : (__ballot_sync(mask, tight) & mask) >> first;
        if (bits)
            return r.ach[base + static_cast<u32>(__ffs(static_cast<int>(bits))) - 1];
    }
    return k_none;
}

/// Level lv for the n propositions of `list` (their candidate supporters are final: the smallest tight achiever of the
/// round), with their supporters: for FF an axiom passes on the supporter of its precondition of level lv - 1 with the
/// smallest id; set-additive keeps the axiom itself. Candidate preconditions have levels below lv already, so no thread
/// reads a level this round sets.
template<u32 KIND, bool WARP>
__device__ __forceinline__ void assign_levels(const Relaxed& r, const Arrays& a, GroupState& gs, const u32* list, u32 n,
                                              u32 lv, const Group<WARP>& g)
{
    for (u32 i = g.tid; i < n; i += g.size)
    {
        const u32 e = list[i];
        const u32 o = a.cand[e];
        a.lvl[e] = lv;
        u32 s = o;
        if (KIND == k_ff && r.axiom[o])
        {
            u32 best = k_none;
            for (u32 j = r.pre_begin[o]; j < r.pre_begin[o + 1]; ++j)
            {
                const u32 q = r.pre[j];
                if (a.lvl[q] + 1 == lv && q < best)
                    best = q;
            }
            s = best == k_none ? k_none : a.supp[best];
        }
        a.supp[e] = s;
        if (r.is_goal[e])
            atomicSub(&gs.left, 1u);
    }
}

/// h_FF's supporter levels by rounds over all operators (sweeps): round lv evaluates the operators whose preconditions
/// all have levels, the largest lv - 1.
template<u32 KIND, bool WARP>
__device__ void levels_sweep(const Relaxed& r, const Arrays& a, GroupState& gs, const Group<WARP>& g)
{
    const u32 tid = g.tid, nt = g.size;
    for (u32 lv = 1;; ++lv)
    {
        const u32 cur = lv & 1, nx = cur ^ 1;
        if (tid == 0)
            gs.nT[nx] = 0;
        for (u32 o = tid; o < r.O; o += nt)
        {
            const u32 pb = r.pre_begin[o], pe = r.pre_begin[o + 1];
            u32 m = 0, cm = 0;
            bool ok = true;
            for (u32 j = pb; j < pe; ++j)
            {
                const u32 q = r.pre[j];
                const u32 l = a.lvl[q];
                if (l == k_inf)
                {
                    ok = false;
                    break;
                }
                m = l > m ? l : m;
                const u32 c = a.cost[q];
                cm = c > cm ? c : cm;
            }
            if (!ok || m + 1 != lv)
                continue;
            const u32 v = sat(u64{cm} + r.opcost[o]);
            for (u32 j = r.eff_begin[o]; j < r.eff_begin[o + 1]; ++j)
            {
                const u32 e = r.eff[j];
                if (a.lvl[e] == k_inf && a.cost[e] == v && atomicMin(a.cand + e, o) == k_none)
                    a.T[atomicAdd(&gs.nT[cur], 1u)] = e;
            }
        }
        g.sync();
        const u32 n = gs.nT[cur];
        if (n == 0)
            break;
        assign_levels<KIND>(r, a, gs, a.T, n, lv, g);
        g.sync();
        if (gs.left == 0)
            break;
    }
}

/// h_FF's supporter levels by the frontier: the operators' counters count the preconditions without a level; the
/// propositions leveled lv - 1 decrement them, and an operator whose counter reaches 0 (its preconditions leveled, the
/// largest lv - 1) is evaluated once, in round lv, as the sweeps evaluate it there. Only settled propositions get
/// levels (the others cost more than every goal: no supporter of the relaxed plan reads them).
template<u32 KIND, bool WARP>
__device__ void levels_frontier(const Relaxed& r, const Arrays& a, GroupState& gs, const Group<WARP>& g)
{
    const u32 tid = g.tid, nt = g.size;
    const u32* done = a.nxt;
    for (u32 o = tid; o < r.O; o += nt)
        a.left[o] = r.pre_begin[o + 1] - r.pre_begin[o];
    if (tid == 0)
    {
        gs.nT[0] = 0;
        gs.nT[1] = 0;
    }
    g.sync();
    for (u32 p = tid; p < r.P; p += nt)
        if (a.lvl[p] == 0)
            a.T2[atomicAdd(&gs.nT[0], 1u)] = p;
    g.sync();
    auto ready = [&](u32 o, u32* to, u32* count)
    {
        u32 cm = 0;
        for (u32 j = r.pre_begin[o]; j < r.pre_begin[o + 1]; ++j)
        {
            const u32 c = a.cost[r.pre[j]];
            cm = c > cm ? c : cm;
        }
        const u32 v = sat(u64{cm} + r.opcost[o]);
        for (u32 j = r.eff_begin[o]; j < r.eff_begin[o + 1]; ++j)
        {
            const u32 e = r.eff[j];
            if (a.lvl[e] == k_inf && done[e] && a.cost[e] == v && atomicMin(a.cand + e, o) == k_none)
                to[atomicAdd(count, 1u)] = e;
        }
    };
    for (u32 lv = 1;; ++lv)
    {
        // the lists alternate: round lv reads the propositions of level lv - 1 from `from` (count nT[prev]) and
        // appends the candidates of level lv to `to` (count nT[cur])
        const u32 cur = lv & 1, prev = cur ^ 1;
        const u32* from = cur ? a.T2 : a.T;
        u32* to = cur ? a.T : a.T2;
        if (lv == 1)
            for (u32 i = tid; i < r.num_zero; i += nt)
            {
                const u32 o = r.zero_ops[i];
                if (r.pre_begin[o] == r.pre_begin[o + 1])
                    ready(o, to, &gs.nT[cur]);
            }
        const u32 nf = gs.nT[prev], L = r.pre_of_lanes;
        for (u32 i = tid / L; i < nf; i += nt / L)
        {
            const u32 q = from[i];
            for (u32 j = r.pre_of_begin[q] + tid % L; j < r.pre_of_begin[q + 1]; j += L)
            {
                const u32 o = r.pre_of[j];
                if (atomicSub(a.left + o, 1u) == 1)
                    ready(o, to, &gs.nT[cur]);
            }
        }
        g.sync();
        const u32 n = gs.nT[cur];
        if (n == 0)
            break;
        if (tid == 0)
            gs.nT[prev] = 0;  // read above by every thread before the sync
        assign_levels<KIND>(r, a, gs, to, n, lv, g);
        g.sync();
        if (gs.left == 0)
            break;
    }
}

/// h of one state (every thread of the group calls it and gets the same value: k_inf for dead ends and states that
/// are not evaluated; *status their status).
template<u32 KIND, u32 VARIANT, bool WARP>
__device__ u32 evaluate_one(const Relaxed& r, const u64* row, u32 words, const Arrays& a, GroupState& gs, u8* status,
                            const Group<WARP>& g)
{
    const u32 tid = g.tid, nt = g.size;
    if (tid == 0)
    {
        gs.flag = 0;
        gs.dead = 0;
        gs.hmax = 0;
        gs.hsum = 0;
        gs.nT[0] = 0;
        gs.nT[1] = 0;
        gs.changed[0] = 0;
    }
    init_state<KIND>(r, row, words, a, gs, g);
    g.sync();
    if (gs.flag)
    {
        *status = (gs.flag & k_invalid) ? k_invalid : k_outside;
        return k_inf;
    }
    *status = k_ok;
    if (r.goal_unreachable)
        return k_inf;

    if constexpr (VARIANT == k_frontier)
        costs_frontier<KIND>(r, a, gs, g);
    else
        costs_sweep<KIND>(r, a, gs, g);

    // ------------------------------------------------------------------------------------------ h_max / h_add, dead ends
    // (the frontier stops early: its goals are settled unless it ran out of reachable propositions, a dead end)
    if (tid == 0)
        gs.left = 0;
    g.sync();
    for (u32 i = tid; i < r.G; i += nt)
    {
        const u32 c = a.cost[r.goal[i]];
        if (c == k_inf || (VARIANT == k_frontier && !a.nxt[r.goal[i]]))
            atomicOr(&gs.dead, 1u);
        else if (KIND == k_add)
            atomicAdd(&gs.hsum, static_cast<unsigned long long>(c));
        else if (KIND == k_max)
            atomicMax(&gs.hmax, c);
        if ((KIND == k_ff || KIND == k_set_additive) && !r.uniform_cost && a.lvl[r.goal[i]] == k_inf)
            atomicAdd(&gs.left, 1u);
    }
    if (tid == 0)
    {
        gs.nT[0] = 0;
        gs.nT[1] = 0;
    }
    g.sync();
    if (gs.dead)
        return k_inf;
    if constexpr (KIND == k_max)
        return gs.hmax;
    if constexpr (KIND == k_add)
        return gs.hsum >= k_inf ? k_inf : static_cast<u32>(gs.hsum);

    if constexpr (KIND == k_ff || KIND == k_set_additive)
    {
        // -------------------------------------------------------------------------------------- supporter levels
        // (uniform costs: none, the relaxed plan looks the supporters up)
        if (!r.uniform_cost && gs.left > 0)
        {
            if constexpr (VARIANT == k_frontier)
                levels_frontier<KIND>(r, a, gs, g);
            else
                levels_sweep<KIND>(r, a, gs, g);
        }

        // -------------------------------------------------------------------------------------- relaxed plan
        if (tid == 0)
        {
            gs.nF[0] = 0;
            gs.nF[1] = 0;
            gs.nF[2] = 0;
            gs.hsum = 0;
        }
        for (u32 i = tid; i < words32(r.P); i += nt)
            a.pbits[i] = 0;
        for (u32 i = tid; i < words32(r.GA); i += nt)
            a.gbits[i] = 0;
        g.sync();
        for (u32 i = tid; i < r.G; i += nt)
        {
            const u32 x = r.goal[i];
            if (test_and_set(a.pbits, x))
                a.T[atomicAdd(&gs.nF[0], 1u)] = x;
        }
        g.sync();
        // the frontier alternates between T and the second list (the sweeps' nxt, the frontier's T2: nxt holds its
        // settled flags)
        u32* const other = VARIANT == k_frontier ? a.T2 : a.nxt;
        for (u32 k = 0;; ++k)
        {
            const u32 cur = k % 3, nx = (k + 1) % 3;
            const u32* from = (k & 1) ? other : a.T;
            u32* to = (k & 1) ? a.T : other;
            const u32 n = gs.nF[cur];
            if (n == 0)
                break;
            if (tid == 0)
                gs.nF[(k + 2) % 3] = 0;
            // uniform costs: r.ach_lanes lanes per proposition look up its supporter and share its preconditions, fewer
            // when the round has more propositions than the group has lanes for (every lane busy either way)
            u32 L = r.uniform_cost ? r.ach_lanes : 1;
            while (L > 1 && n * L > nt)
                L >>= 1;
            for (u32 i = tid / L; i < n; i += nt / L)
            {
                const u32 o = r.uniform_cost ? tight_achiever(r, a.cost, from[i], L) : a.supp[from[i]];
                if (o == k_none)
                    continue;
                const u32 ga = r.op_ga[o];
                if (tid % L == 0 && (KIND == k_ff ? test_and_set(a.gbits, ga) : !r.axiom[o]))
                    atomicAdd(&gs.hsum, static_cast<unsigned long long>(r.ga_cost[ga]));
                for (u32 j = r.pre_begin[o] + tid % L; j < r.pre_begin[o + 1]; j += L)
                {
                    const u32 q = r.pre[j];
                    if (test_and_set(a.pbits, q))
                        to[atomicAdd(&gs.nF[nx], 1u)] = q;
                }
            }
            g.sync();
        }
        return gs.hsum >= k_inf ? k_inf : static_cast<u32>(gs.hsum);
    }
    return k_inf;
}

template<u32 KIND, u32 VARIANT, bool SHARED, bool WARP>
__global__ void k_evaluate(Relaxed r, Rows rows, u64 group_bytes, void* gscratch, Out out)
{
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ GroupState states[WARP ? k_max_groups_per_block : 1];
    const Group<WARP> g;
    const u32 local = WARP ? threadIdx.x / 32 : 0;             // group within the block
    const u32 per_block = WARP ? blockDim.x / 32 : 1;
    const u64 group = u64{blockIdx.x} * per_block + local;       // global group index
    const u64 groups = u64{gridDim.x} * per_block;
    u32* base = SHARED ? reinterpret_cast<u32*>(smem + local * group_bytes)
                       : reinterpret_cast<u32*>(static_cast<unsigned char*>(gscratch) + group * group_bytes);
    const Arrays a = carve<KIND, VARIANT>(base, r);
    GroupState& gs = states[local];
    const u64 nrows = rows.count && *rows.count < rows.n ? *rows.count : rows.n;
    for (u64 i = group; i < nrows; i += groups)
    {
        u8 st = k_ok;
        const u32 h = evaluate_one<KIND, VARIANT>(r, rows.data + i * rows.stride, rows.words, a, gs, &st, g);
        if (g.tid == 0)
        {
            out.h[i] = h;
            if (out.status)
                out.status[i] = st;
            if (st == k_outside)
                atomicAdd(out.counters, 1u);
            else if (st == k_invalid)
                atomicAdd(out.counters + 1, 1u);
        }
        g.sync();
    }
}

__global__ void k_to_f64(const u32* h, u64 n, f64* out)
{
    for (u64 i = u64{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += u64{gridDim.x} * blockDim.x)
        out[i] = h[i] == k_inf ? __longlong_as_double(0x7FF0000000000000ll) : static_cast<f64>(h[i]);
}

using Kernel = void (*)(Relaxed, Rows, u64, void*, Out);

template<u32 KIND, u32 VARIANT>
Kernel pick(bool shared, bool warp)
{
    if (shared)
        return warp ? k_evaluate<KIND, VARIANT, true, true> : k_evaluate<KIND, VARIANT, true, false>;
    return warp ? k_evaluate<KIND, VARIANT, false, true> : k_evaluate<KIND, VARIANT, false, false>;
}

Kernel kernel_of(const Launch& l)
{
    const bool sh = l.shared != 0, wp = l.warp != 0;
    switch (l.kind * 2 + l.variant)
    {
        case k_max * 2 + k_sweep: return pick<k_max, k_sweep>(sh, wp);
        case k_max * 2 + k_frontier: return pick<k_max, k_frontier>(sh, wp);
        case k_add * 2 + k_sweep: return pick<k_add, k_sweep>(sh, wp);
        case k_add * 2 + k_frontier: return pick<k_add, k_frontier>(sh, wp);
        case k_ff * 2 + k_sweep: return pick<k_ff, k_sweep>(sh, wp);
        case k_ff * 2 + k_frontier: return pick<k_ff, k_frontier>(sh, wp);
        case k_set_additive * 2 + k_sweep: return pick<k_set_additive, k_sweep>(sh, wp);
        case k_set_additive * 2 + k_frontier: return pick<k_set_additive, k_frontier>(sh, wp);
        default: return nullptr;
    }
}

u64 dynamic_bytes(const Launch& l)
{
    if (!l.shared)
        return 0;
    return l.group_bytes * (l.warp ? l.threads / 32 : 1);
}

bool valid(const Launch& l)
{
    return l.variant <= k_frontier && l.kind <= k_set_additive && l.threads >= 32 && l.threads % 32 == 0 && l.threads <= 1024 &&
           l.blocks > 0 && l.group_bytes % 16 == 0 && (l.shared || l.scratch) && (!l.warp || l.threads / 32 <= k_max_groups_per_block);
}
}  // namespace

cudaError_t launch_evaluate(const Relaxed& r, Rows rows, Launch l, Out out, cudaStream_t s)
{
    if (rows.n == 0)
        return cudaSuccess;
    if (l.kind == k_h2)
        return launch_h2(r, rows, l, out, s);
    const Kernel k = kernel_of(l);
    if (!k || !valid(l) || !out.h || !out.counters)
        return cudaErrorInvalidValue;
    const u64 dyn = dynamic_bytes(l);
    if (dyn > 0)
    {
        const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dyn));
        if (e != cudaSuccess)
            return e;
    }
    k<<<l.blocks, l.threads, dyn, s>>>(r, rows, l.group_bytes, l.scratch, out);
    return cudaGetLastError();
}

cudaError_t launch_to_f64(const u32* h, u64 n, f64* out, cudaStream_t s)
{
    if (n == 0)
        return cudaSuccess;
    const u64 g = (n + 255) / 256;
    k_to_f64<<<static_cast<unsigned>(g < 65535 ? g : 65535), 256, 0, s>>>(h, n, out);
    return cudaGetLastError();
}

cudaError_t occupancy(const Launch& l, int* blocks_per_sm)
{
    *blocks_per_sm = 0;
    if (l.kind == k_h2)
        return occupancy_h2(l, blocks_per_sm);
    const Kernel k = kernel_of(l);
    Launch v = l;
    v.blocks = 1;
    if (!v.shared)
        v.scratch = reinterpret_cast<void*>(16);  // not dereferenced
    if (!k || !valid(v))
        return cudaErrorInvalidValue;
    const u64 dyn = dynamic_bytes(l);
    if (dyn > 0)
    {
        const cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dyn));
        if (e != cudaSuccess)
            return e;
    }
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, k, static_cast<int>(l.threads), dyn);
}
}  // namespace mymyr::cuda::hk
