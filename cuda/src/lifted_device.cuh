#pragma once
// The device side of the lifted successor kernels (include/mymyr/cuda/lifted.hpp), shared by the translation units that
// instantiate them: lifted.cu (views, counts, rows without conditional effects, picks), lifted_ce.cu (rows of schemas
// with conditional effects) and axioms.cu (axiom strata). Internal: included by .cu files only (nvcc, C++20;
// the device-code subset). Everything lives in an unnamed namespace, so each translation unit instantiates its own
// kernels.

#include "mymyr/cuda/lifted.hpp"

#include <atomic>
#include <concepts>

namespace mymyr::cuda::lifted
{
namespace
{
using namespace rl::dev;

constexpr unsigned k_block = 256;
constexpr unsigned k_gen_block = 128;
constexpr u32 k_full_mask = 0xFFFFFFFFu;

inline unsigned grid_for(u64 n, unsigned block = k_block)
{
    const u64 g = (n + block - 1) / block;
    return static_cast<unsigned>(g < 65535u * 64u ? (g ? g : 1) : 65535u * 64u);
}

/// The rows of `p` that are present: p.rows, or fewer when a device count limits them (Parents::rows_dev).
__device__ __forceinline__ u32 live_rows(const Parents& p)
{
    if (!p.rows_dev)
        return p.rows;
    const u32 n = *p.rows_dev;
    return n < p.rows ? n : p.rows;
}

/// Row o of a table reference: a ViewLayout row (bit 63), whose row o is the view's row view_map[ref + o] (the compact
/// view, task_arrays_view.hpp), or a word offset into the static words (rows of OW words).
template<u32 OW>
__device__ __forceinline__ const u64* table_row(const TaskView& t, const u64* view, u64 ref, u32 o)
{
    if (ref & k_table_view)
        return view + u64{__ldg(t.view_map + (ref & ~k_table_view) + o)} * OW;
    return t.static_words + ref + u64{o} * OW;
}

__device__ __forceinline__ u32 lowest_bit(u64 w) { return static_cast<u32>(__ffsll(static_cast<long long>(w)) - 1); }

// ------------------------------------------------------------------------------------------------ matchers

/// State access of one thread.
struct StateRef
{
    const u64* words;
    u32 nw;
    const u64* derived;
    u32 dw;
    const u64* numeric = nullptr;
};

__device__ __forceinline__ bool check(const TaskView& t, u32 c, const u32* bind, const StateRef& s)
{
    return holds(t, c, bind, s.words, s.nw, s.derived, s.dw);
}

/// Engine::run's prologue: never, the literals without free parameters. False: no binding.
__device__ __forceinline__ bool prologue(const TaskView& t, const u32* mx, const u32* bind, const StateRef& s)
{
    if (mx[k_mc_flags] & k_mc_never)
        return false;
    const u32 pc = mx[k_mc_pre_checks], pn = mx[k_mc_pre_checks_n];
    for (u32 i = 0; i < pn; ++i)
        if (!check(t, pc + i, bind, s))
            return false;
    return true;
}

/// dom1 of parameter `param` (the static domain and the per-state unary constraints); false if it is empty.
template<u32 OW>
__device__ __forceinline__ bool domain(const TaskView& t, const u32* mx, const u64* view, u32 param, u64* out)
{
    const u64* d0 = t.dom0 + mx[k_mc_dom0] + u64{param} * OW;
    const u32* ub = t.index + mx[k_mc_unary_begin];
    const u64* un = t.unary + u64{mx[k_mc_unary]} * 2;
    const u32 b = ub[param], e = ub[param + 1];
    u64 any = 0;
#pragma unroll
    for (u32 w = 0; w < OW; ++w)
    {
        u64 a = d0[w];
        for (u32 u = b; u < e; ++u)
        {
            const u64* tb = table_row<OW>(t, view, un[u * 2], 0);
            a &= un[u * 2 + 1] ? ~tb[w] : tb[w];
        }
        out[w] = a;
        any |= a;
    }
    return any != 0;
}

/// Calls emit(b); false if the emitter returns false (it wants no more bindings: the search stops). Emitters that return
/// nothing see every binding.
template<class Emit>
__device__ __forceinline__ bool emit_more(Emit& emit, const u32* b)
{
    if constexpr (requires { { emit(b) } -> std::same_as<bool>; })
        return emit(b);
    else
    {
        emit(b);
        return true;
    }
}

/// Engine::run + Engine::search (fixed order with backward rows, witness pruning) as a loop: emit(bind) per binding.
/// MaxD bounds the matcher's steps (its stacks are MaxD x OW words; the conditional-effect sub-plans use a small one).
/// Split: split_n lanes share the search, lane split_i taking every split_n-th candidate of the first level with more
/// than one (above it the path is unique). Together they emit every binding, some bindings more than once when witness
/// pruning is on (callers of Split must not mind). Exact: the split level lies above the witness levels (none:
/// lane 0 searches alone), so that the lanes emit the bindings of one search exactly once (a lane's candidates of the
/// split level each search their whole subtree, witness cut-offs included).
template<u32 OW, u32 MaxD = k_max_depth, bool Split = false, bool Exact = false, bool Numeric = false, class Emit>
__device__ __forceinline__ void run_fixed(const TaskView& t, const u32* mx, const StateRef& s, const u64* view, u32* bind,
                                          Emit& emit, u32 split_n = 1, u32 split_i = 0)
{
    if (!prologue(t, mx, bind, s))
        return;
    const u32* nm = nullptr;
    if constexpr (Numeric)
    {
        nm = t.numeric.matcher + ((mx - t.matcher) / k_mc_count) * 8;
        for (u32 i = 0; i < nm[1]; ++i)
            if (!numeric_holds(t.numeric, nm[0] + i, s.numeric, bind, t.num_objects))
                return;
    }
    const u32 nd = mx[k_mc_steps_n];
    if (nd == 0)
    {
        if (!Split || split_i == 0)
            emit(bind);
        return;
    }
    const u32* steps = t.step + u64{mx[k_mc_steps]} * 5;
    u64 dom1[MaxD][OW];
    for (u32 d = 0; d < nd; ++d)
        if (!domain<OW>(t, mx, view, steps[d * 5], dom1[d]))
            return;
    const u64* rows = t.row + u64{mx[k_mc_rows]} * 3;
    const u32* step_checks = t.index + mx[k_mc_step_checks];
    const u32 checks0 = mx[k_mc_checks];
    const int fe = static_cast<int>(mx[k_mc_first_exist]);
    u64 cd[MaxD][OW];
    auto fill = [&](int d)
    {
        const u32* sp = steps + d * 5;
        u64 a[OW];
#pragma unroll
        for (u32 w = 0; w < OW; ++w)
            a[w] = dom1[d][w];
        for (u32 r = sp[1]; r < sp[2]; ++r)
        {
            const u64* rw = rows + u64{r} * 3;
            const u64* x = table_row<OW>(t, view, rw[0], bind[rw[1]]);
            const bool neg = rw[2] != 0;
#pragma unroll
            for (u32 w = 0; w < OW; ++w)
                a[w] &= neg ? ~x[w] : x[w];
        }
#pragma unroll
        for (u32 w = 0; w < OW; ++w)
            cd[d][w] = a[w];
    };
    int d = 0;
    bool wit = fe == 0;
    int sl = -1;   // Split: the level whose candidates the lanes deal
    u32 seen = 0;  // candidates of level sl so far
    auto split_at = [&](int l)
    {
        if constexpr (Split)
        {
            if (sl < 0 && (!Exact || l < fe))
            {
                u32 n = 0;
#pragma unroll
                for (u32 x = 0; x < OW; ++x)
                    n += static_cast<u32>(__popcll(cd[l][x]));
                if (n > 1)
                    sl = l;
            }
        }
    };
    fill(0);
    split_at(0);
    const int last = static_cast<int>(nd) - 1;
    while (d >= 0)
    {
        u32 w = 0;
        while (w < OW && cd[d][w] == 0)
            ++w;
        if (w == OW)
        {
            --d;
            if (d < fe)
                wit = false;
            continue;
        }
        const u32 o = w * 64 + lowest_bit(cd[d][w]);
        cd[d][w] &= cd[d][w] - 1;
        if constexpr (Split)
        {
            if (d == sl && (seen++ % split_n) != split_i)
                continue;
        }
        const u32* sp = steps + d * 5;
        bind[sp[0]] = o;
        bool ok = true;
        for (u32 c = sp[3]; c < sp[4] && ok; ++c)
            ok = check(t, checks0 + step_checks[c], bind, s);
        if constexpr (Numeric)
        {
            const u32* ns = t.numeric.steps + u64{nm[4] + static_cast<u32>(d)} * 2;
            for (u32 j = ns[0]; j < ns[1] && ok; ++j)
                ok = numeric_holds(t.numeric, t.numeric.index[j], s.numeric, bind, t.num_objects);
        }
        if (!ok)
            continue;
        if (d == last)
        {
            if constexpr (Split)
            {
                if (sl < 0 && split_i != 0)
                    return;  // a unique path: lane 0 emits it
            }
            if (!emit_more(emit, bind))
                return;
            if (wit)
            {
                d = fe - 1;  // one witness suffices: back to the last effect-relevant step
                wit = false;
            }
            continue;
        }
        ++d;
        if (d == fe)
            wit = true;
        fill(d);
        split_at(d);
    }
}

/// Engine::run + Engine::search_fc (fail-first forward checking, witness pruning) as a loop over levels. The domains of
/// level l are D[l][param], `total` parameters per level (the touched part of the stack stays small); a level holds the
/// domains of the parameters not bound above it (the others are never read again). The candidates still to try at
/// level l are rem[l]. Split, Exact: as in run_fixed (Exact: the split level is entered before the witness search).
template<u32 OW, bool Split = false, bool Exact = false, bool Numeric = false, class Emit>
__device__ __forceinline__ void run_fc(const TaskView& t, const u32* mx, const StateRef& s, const u64* view, u32* bind,
                                       Emit& emit, u32 split_n = 1, u32 split_i = 0)
{
    if (!prologue(t, mx, bind, s))
        return;
    const u32* nm = nullptr;
    if constexpr (Numeric)
    {
        nm = t.numeric.matcher + ((mx - t.matcher) / k_mc_count) * 8;
        for (u32 i = 0; i < nm[1]; ++i)
            if (!numeric_holds(t.numeric, nm[0] + i, s.numeric, bind, t.num_objects))
                return;
    }
    const u32 nd = mx[k_mc_steps_n];
    if (nd == 0)
    {
        if (!Split || split_i == 0)
            emit(bind);
        return;
    }
    const u32 total = mx[k_mc_total];
    const u32* steps = t.step + u64{mx[k_mc_steps]} * 5;
    u64 Dm[(k_max_depth + 1) * k_max_depth * OW];
    auto D = [&](u32 l, u32 v) { return Dm + (l * total + v) * OW; };
    for (u32 v = 0; v < total; ++v)
#pragma unroll
        for (u32 w = 0; w < OW; ++w)
            D(0, v)[w] = 0;
    for (u32 d = 0; d < nd; ++d)
        if (!domain<OW>(t, mx, view, steps[d * 5], D(0, steps[d * 5])))
            return;
    // prebound -> free rows applied to the initial domains
    {
        const u64* pre = t.row + u64{mx[k_mc_fc_pre]} * 3;
        const u32* to = t.index + mx[k_mc_fc_pre_to];
        for (u32 i = 0; i < mx[k_mc_fc_pre_n]; ++i)
        {
            const u64* rw = pre + u64{i} * 3;
            const u64* row = table_row<OW>(t, view, rw[0], bind[rw[1]]);
            u64 any = 0;
#pragma unroll
            for (u32 w = 0; w < OW; ++w)
                any |= (D(0, to[i])[w] &= rw[2] ? ~row[w] : row[w]);
            if (!any)
                return;
        }
    }
    const u32 nfree = mx[k_mc_free_params_n];
    const u32* fp = t.index + mx[k_mc_free_params];
    const u32* rel = t.index + mx[k_mc_relevant];
    const u32* fob = t.index + mx[k_mc_fc_out_begin];
    const u64* edges = t.edge + u64{mx[k_mc_fc_out]} * 3;
    const u32* fcb = t.index + mx[k_mc_fc_checks_begin];
    const u32* fck = t.index + mx[k_mc_fc_checks];
    const u32* cvb = t.index + mx[k_mc_check_vars_begin];
    const u32* cv = t.index + mx[k_mc_check_vars];
    const u32 checks0 = mx[k_mc_checks];

    u32 bound = 0;  // bit v: parameter v is bound (total <= k_max_depth)
    u32 pick[k_max_depth];
    u64 rem[k_max_depth][OW];
    bool witness = false;
    int wlev = -1;  // the level where the witness search started
    int sl = -1;    // Split: the level whose candidates the lanes deal (the first with more than one)
    u32 seen = 0;
    // search_fc's entry at level l (unbound > 0): witness switch, fail-first choice (effect-relevant parameters first)
    auto enter = [&](int l)
    {
        bool any_relevant = false;
        for (u32 i = 0; i < nfree && !any_relevant; ++i)
            any_relevant = !((bound >> fp[i]) & 1) && rel[fp[i]];
        if (!any_relevant && !witness)
        {
            witness = true;
            wlev = l;
        }
        u32 best = 0, best_cnt = ~0u;
        for (u32 i = 0; i < nfree; ++i)
        {
            const u32 v = fp[i];
            if (((bound >> v) & 1) || (any_relevant && !rel[v]))
                continue;
            u32 cnt = 0;
#pragma unroll
            for (u32 w = 0; w < OW; ++w)
                cnt += static_cast<u32>(__popcll(D(l, v)[w]));
            if (cnt < best_cnt)
            {
                best = v;
                best_cnt = cnt;
            }
        }
        pick[l] = best;
#pragma unroll
        for (u32 w = 0; w < OW; ++w)
            rem[l][w] = D(l, best)[w];
        bound |= 1u << best;
        if constexpr (Split)
        {
            if (sl < 0 && best_cnt > 1 && (!Exact || !witness))
                sl = l;
        }
    };
    int level = 0;
    enter(0);
    while (level >= 0)
    {
        const u32 p = pick[level];
        u32 w = 0;
        while (w < OW && rem[level][w] == 0)
            ++w;
        if (w == OW)
        {
            bound &= ~(1u << p);
            --level;
            if (level < wlev)
            {
                witness = false;
                wlev = -1;
            }
            continue;
        }
        const u32 o = w * 64 + lowest_bit(rem[level][w]);
        rem[level][w] &= rem[level][w] - 1;
        if constexpr (Split)
        {
            if (level == sl && (seen++ % split_n) != split_i)
                continue;
        }
        bind[p] = o;
        bool ok = true;
        for (u32 k = fcb[p]; k < fcb[p + 1] && ok; ++k)
        {
            const u32 c = fck[k];
            bool all = true;
            for (u32 j = cvb[c]; j < cvb[c + 1]; ++j)
                all &= ((bound >> cv[j]) & 1) != 0;
            if (all)
                ok = check(t, checks0 + c, bind, s);
        }
        if constexpr (Numeric)
            for (u32 j = 0; j < nm[3] && ok; ++j)
                if ((t.numeric.masks[nm[5] + j] & bound) == t.numeric.masks[nm[5] + j])
                    ok = numeric_holds(t.numeric, nm[2] + j, s.numeric, bind, t.num_objects);
        if (!ok)
            continue;
        for (u32 v = 0; v < total; ++v)
        {
            if ((bound >> v) & 1)
                continue;
#pragma unroll
            for (u32 x = 0; x < OW; ++x)
                D(level + 1, v)[x] = D(level, v)[x];
        }
        for (u32 e = fob[p]; e < fob[p + 1] && ok; ++e)
        {
            const u64* ed = edges + u64{e} * 3;
            const u32 to = static_cast<u32>(ed[0]);
            if ((bound >> to) & 1)
                continue;
            const u64* row = table_row<OW>(t, view, ed[1], o);
            u64 any = 0;
#pragma unroll
            for (u32 x = 0; x < OW; ++x)
                any |= (D(level + 1, to)[x] &= ed[2] ? ~row[x] : row[x]);
            ok = any != 0;
        }
        if (!ok)
            continue;
        if (static_cast<u32>(level) + 1 == nfree)
        {
            if constexpr (Split)
            {
                if (sl < 0 && split_i != 0)
                    return;  // a unique path: lane 0 emits it
            }
            if (!emit_more(emit, bind))
                return;
            if (witness)
            {
                // the witness search found its binding: return to the level before the witness entry
                for (int l = level; l >= wlev; --l)
                    bound &= ~(1u << pick[l]);
                level = wlev - 1;
                witness = false;
                wlev = -1;
            }
            continue;
        }
        ++level;
        enter(level);
    }
}

// ------------------------------------------------------------------------------------------------ count / write

__device__ __forceinline__ bool row_less(const u32* a, const u32* b, u32 n)
{
    for (u32 i = 0; i < n; ++i)
        if (a[i] != b[i])
            return a[i] < b[i];
    return false;
}

__device__ __forceinline__ void row_swap(u32* a, u32* b, u32 n)
{
    for (u32 i = 0; i < n; ++i)
    {
        const u32 x = a[i];
        a[i] = b[i];
        b[i] = x;
    }
}

/// Sorts `cnt` rows (`stride` u32 apart) lexicographically over their first `n` entries (unique keys, so any correct
/// sort gives the CPU's order): insertion sort for short segments, heapsort otherwise (in place, O(cnt log cnt)).
__device__ inline void sort_rows(u32* base, u64 cnt, u32 stride, u32 n)
{
    auto row = [&](u64 i) { return base + i * stride; };
    if (cnt <= 16)
    {
        for (u64 i = 1; i < cnt; ++i)
            for (u64 j = i; j > 0 && row_less(row(j), row(j - 1), n); --j)
                row_swap(row(j), row(j - 1), n);
        return;
    }
    auto sift = [&](u64 root, u64 end)
    {
        for (;;)
        {
            u64 child = 2 * root + 1;
            if (child >= end)
                return;
            if (child + 1 < end && row_less(row(child), row(child + 1), n))
                ++child;
            if (!row_less(row(root), row(child), n))
                return;
            row_swap(row(root), row(child), n);
            root = child;
        }
    };
    for (u64 start = cnt / 2; start-- > 0;)
        sift(start, cnt);
    for (u64 end = cnt; end > 1; --end)
    {
        row_swap(row(0), row(end - 1), n);
        sift(0, end - 1);
    }
}

/// Everything a count / write launch reads.
struct GenArgs
{
    TaskView t;
    Parents p;
    Views v;
    SchemaSet set;
    const u32* offsets;
    u32* counts;
    Labels lab;
    SuccessorRows out;
    Multi m{};  // multi-instance launches: the instances and the row order
};

/// Bindings of a segment that needs sorting and fits k_local_keys packed keys stay in registers / local memory.
constexpr u32 k_local_keys = 32;

__device__ __forceinline__ u32 key_bits(u32 num_objects)
{
    return num_objects <= 1 ? 1u : 32u - static_cast<u32>(__clz(num_objects - 1));
}

/// Position of set bit k (from 0, lowest first) of x; x has more than k set bits.
__device__ __forceinline__ u32 nth_bit(u64 x, u32 k)
{
    const auto lo = static_cast<u32>(x);
    const auto c = static_cast<u32>(__popc(lo));
    if (k < c)
        return __fns(lo, 0, static_cast<int>(k + 1));
    return 32 + __fns(static_cast<u32>(x >> 32), 0, static_cast<int>(k - c + 1));
}

/// Words of a local rank bitmap (RankSpace).
constexpr u32 k_rank_words = 16;

/// Canonical ranks of a schema's bindings: parameter i's rank among the objects of its static domain (dom0 of
/// the matcher: every binding the matcher emits lies in it), mixed radix with parameter 0 the most significant, so the
/// ranks order bindings as the canonical (lexicographic) order does. When the domains span at most
/// 64 * k_rank_words bindings (`size` > 0), a segment sorts by one search that sets its ranks in a local bitmap, read
/// back in order: gripper's pick and drop over 80 balls, blocks' stack and unstack.
template<u32 OW>
struct RankSpace
{
    const u64* dom;  // dom0 of parameter 0 (parameter i at dom + i * OW)
    u32 arity;
    u32 size = 0;  // the product of the domain sizes, 0 when above 64 * k_rank_words

    __device__ RankSpace(const TaskView& t, const u32* mx, u32 n) : dom(t.dom0 + mx[k_mc_dom0]), arity(n)
    {
        u32 p = 1;
        for (u32 i = 0; i < n && p <= 64 * k_rank_words; ++i)
            p *= count(i);
        size = p <= 64 * k_rank_words ? p : 0;
    }
    [[nodiscard]] __device__ u32 count(u32 i) const
    {
        u32 c = 0;
#pragma unroll
        for (u32 w = 0; w < OW; ++w)
            c += static_cast<u32>(__popcll(dom[u64{i} * OW + w]));
        return c;
    }
    /// The rank of binding b, or k_none if a value lies outside its domain.
    [[nodiscard]] __device__ u32 rank(const u32* b) const
    {
        u32 r = 0;
        for (u32 i = 0; i < arity; ++i)
        {
            const u64* d = dom + u64{i} * OW;
            const u32 o = b[i], w = o >> 6;
            if (w >= OW || !((d[w] >> (o & 63)) & 1))
                return k_none;
            u32 below = static_cast<u32>(__popcll(d[w] & ((1ull << (o & 63)) - 1)));
            for (u32 x = 0; x < w; ++x)
                below += static_cast<u32>(__popcll(d[x]));
            r = r * count(i) + below;
        }
        return r;
    }
    /// The binding of rank r (r < size).
    __device__ void unrank(u32 r, u32* b) const
    {
        for (u32 i = arity; i-- > 0;)
        {
            const u32 c = count(i);
            u32 k = r % c;
            r /= c;
            const u64* d = dom + u64{i} * OW;
            for (u32 w = 0; w < OW; ++w)
            {
                const auto n = static_cast<u32>(__popcll(d[w]));
                if (k < n)
                {
                    b[i] = w * 64 + nth_bit(d[w], k);
                    break;
                }
                k -= n;
            }
        }
    }
};

/// y[slot] = on (word `slot >> 6` < WB; small WB keeps y in registers).
template<u32 WB>
__device__ __forceinline__ void put_bit(u64 (&y)[WB], u32 slot, bool on)
{
    const u32 w = slot >> 6;
    const u64 m = 1ull << (slot & 63);
    if constexpr (WB <= 4)
    {
#pragma unroll
        for (u32 i = 0; i < WB; ++i)
            if (i == w)
                y[i] = on ? (y[i] | m) : (y[i] & ~m);
    }
    else if (w < WB)
        y[w] = on ? (y[w] | m) : (y[w] & ~m);
}

/// The rows of one (state, schema): label columns, and the successor (delete, then add) with rl::expand's width rules.
/// WB >= max(parent words, output words).
/// CEOW > 0 (the task's OW): the schema's conditional effects are evaluated at each row, as sub-plans at the DFS
/// leaf: every conditional effect's condition matcher (fixed order, at most k_ce_depth free
/// parameters: the forall parameters) runs with the schema's binding prebound over the parent's view and derived bitset,
/// and each firing binding contributes its deletes and adds. The successor is (parent minus every delete) plus every add,
/// unconditional or conditional, as Successors::collect_literals and apply_delta build it.
template<u32 WB, u32 CEOW = 0>
struct RowWriter
{
    const GenArgs& a;
    const TaskView& t;  // the row's instance
    const u32* sc;
    u32 schema, parent, arity;
    const u64* st;
    u32 pw;  // the parent's words: the row's; over several instances, its instance's (the words past them are zero)
    u32 ptrim;
    u32 local;

    __device__ RowWriter(const GenArgs& args, const TaskView& view, const u32* schema_row, u32 schema_id, u32 li,
                         const u64* row) :
        a(args), t(view), sc(schema_row), schema(schema_id), parent(args.lab.parent_base + li), arity(schema_row[k_sc_arity]),
        st(row), pw(args.p.words), ptrim(0), local(li)
    {
        if (args.m.inst)
        {
            const u32 iw = view.state_words ? view.state_words : 1;  // frozen slots: every successor fits
            pw = iw < pw ? iw : pw;
        }
        for (u32 i = pw; i-- > 0;)
            if (st[i])
            {
                ptrim = i + 1;
                break;
            }
    }

    /// Row `pos`: written below the capacity; past it only its width and missing atoms count (rl::expand's rules).
    __device__ void put(u64 pos, const u32* b) const
    {
        const Labels& lab = a.lab;
        const bool inside = pos < lab.capacity;
        if (inside)
        {
            if (lab.binding)
            {
                u32* r = lab.binding + pos * lab.label_width;
                for (u32 i = 0; i < arity; ++i)
                    r[i] = b[i];
                for (u32 i = arity; i < lab.label_width; ++i)
                    r[i] = 0xFFFFFFFFu;
            }
            if (lab.schema)
                lab.schema[pos] = schema;
            if (lab.parent)
                lab.parent[pos] = parent;
            if (a.out.deferred)
                return;  // the successor is launch_put's
        }
        successor(pos, b);
    }

    /// Row `pos`'s successor words (below the capacity), its width and missing atoms.
    __device__ void successor(u64 pos, const u32* b) const
    {
        const bool inside = pos < a.lab.capacity;
        const SuccessorRows& out = a.out;
        if (!(out.words && inside) && !out.words_needed && !out.missing_flag)
            return;
        u64 y[WB];
#pragma unroll
        for (u32 i = 0; i < WB; ++i)
            y[i] = i < pw ? st[i] : 0;
        auto del = [&](u64 key)
        {
            const u32 slot = slot_of(t, key);
            if (slot != k_none && (slot >> 6) < WB)
                put_bit<WB>(y, slot, false);
        };
        for (u32 i = 0; i < sc[k_sc_dels_n]; ++i)
            del(pattern_key(t, sc[k_sc_dels] + i, b));
        u32 beyond = 0;   // words needed by adds past WB
        u32 max_add = 0;  // highest added word + 1
        bool missing = false;
        auto add = [&](u64 key, u64 (&to)[WB])
        {
            const u32 slot = slot_of(t, key);
            if (slot == k_none)
            {
                missing = true;
                if (out.missing_bits && key < t.atom_total)
                    atomicOr(out.missing_bits + (key >> 5), 1u << (key & 31));
                return;
            }
            const u32 w = slot >> 6;
            max_add = max_add > w + 1 ? max_add : w + 1;
            if (w >= WB)
            {
                beyond = beyond > w + 1 ? beyond : w + 1;
                return;
            }
            put_bit<WB>(to, slot, true);
        };
        if constexpr (CEOW == 0)
        {
            for (u32 i = 0; i < sc[k_sc_adds_n]; ++i)
                add(pattern_key(t, sc[k_sc_adds] + i, b), y);
        }
        else
        {
            // the adds are collected apart and applied last: a conditional delete must not remove an add
            u64 adds[WB];
#pragma unroll
            for (u32 i = 0; i < WB; ++i)
                adds[i] = 0;
            for (u32 i = 0; i < sc[k_sc_adds_n]; ++i)
                add(pattern_key(t, sc[k_sc_adds] + i, b), adds);
            const u32 nce = sc[k_sc_ces_n];
            if (nce)
            {
                u32 cb[k_max_depth];  // the schema's binding, then the forall parameters
                for (u32 i = 0; i < arity; ++i)
                    cb[i] = b[i];
                const u64* view = a.v.data + u64{local} * a.v.words;
                const u64* der = a.p.derived ? a.p.derived + u64{local} * a.p.derived_words : nullptr;
                const StateRef s{st, pw, der, a.p.derived_words};
                for (u32 c = 0; c < nce; ++c)
                {
                    const u32* ce = t.cond_effect + (u64{sc[k_sc_ces]} + c) * 5;
                    const u32* mx = t.matcher + u64{ce[0]} * k_mc_count;
                    auto fire = [&](const u32* fb)
                    {
                        for (u32 i = 0; i < ce[4]; ++i)
                            del(pattern_key(t, ce[3] + i, fb));
                        for (u32 i = 0; i < ce[2]; ++i)
                            add(pattern_key(t, ce[1] + i, fb), adds);
                    };
                    run_fixed<CEOW, k_ce_depth>(t, mx, s, view, cb, fire);
                }
            }
#pragma unroll
            for (u32 i = 0; i < WB; ++i)
                y[i] |= adds[i];
        }
        if (missing && out.missing_flag)
            *out.missing_flag = 1;
        const u32 ow = out.out_words;
        const bool fits = !missing && ptrim <= ow && max_add <= ow;
        if (!fits && !missing && out.words_needed)
        {
            // the exact trimmed width of the successor (rl::expand reports it for rows that do not fit)
            u32 ex = beyond;
#pragma unroll
            for (u32 i = 0; i < WB; ++i)
                if (y[i])
                    ex = ex > i + 1 ? ex : i + 1;
            atomicMax(out.words_needed, ex);
        }
        if (out.words && inside)
        {
            u64* r = out.words + pos * ow;
            if (fits)
            {
                // over several instances, a row written over its parent keeps the parent's zero words past its
                // instance's (the env's pick)
                const u32 lim = a.m.inst && out.words == a.p.data && ow == a.p.stride && pos == local && pw < ow ? pw : ow;
#pragma unroll
                for (u32 i = 0; i < WB; ++i)
                    if (i < lim)
                        r[i] = y[i];
                for (u32 i = WB; i < lim; ++i)
                    r[i] = 0;
            }
            else
                for (u32 i = 0; i < ow; ++i)
                    r[i] = 0;
        }
    }
};

/// One (state, schema): count its bindings, or write its rows at offsets[seg] in canonical order. CE: the
/// schema's conditional effects are evaluated at each row (RowWriter).
template<u32 OW, bool FC, bool Write, u32 WB, bool CE = false>
__device__ __forceinline__ void gen_one(const GenArgs& a, const TaskView& t, u32 si, u32 li, const u64* view,
                                        const u64* row, const u64* der)
{
    const u32 schema = a.set.schemas[si];
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[a.set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const Parents& p = a.p;
    const StateRef s{row, p.words, der, p.derived_words};
    const u64 seg = u64{li} * a.set.num_schemas + schema;
    u32 bind[k_max_depth];
    auto run = [&](auto& emit)
    {
        if constexpr (FC)
            run_fc<OW>(t, mx, s, view, bind, emit);
        else
            run_fixed<OW>(t, mx, s, view, bind, emit);
    };
    if constexpr (!Write)
    {
        u32 n = 0;
        auto emit = [&](const u32*) { ++n; };
        run(emit);
        a.counts[seg] = n;
    }
    else
    {
        const u64 off = a.offsets[seg], cnt = a.offsets[seg + 1] - off;
        if (cnt == 0)
            return;
        const RowWriter<WB, CE ? OW : 0> w(a, t, sc, schema, li, row);
        const u32 arity = w.arity;
        if (off >= a.lab.capacity)
        {
            // past the capacity: nothing is written, but widths and missing atoms count (in any order)
            if (!a.out.words_needed && !a.out.missing_flag)
                return;
            auto emit = [&](const u32* b) { w.put(a.lab.capacity, b); };
            run(emit);
            return;
        }
        // forward checking and most fixed orders bind out of parameter order (rl::dev::device_sorts)
        const bool sort = a.set.canonical && arity > 1 && device_sorts(mx[k_mc_flags], FC);
        if (!sort)
        {
            // the matcher's order is the canonical one: rows go out as they are found
            u64 n = 0;
            auto emit = [&](const u32* b)
            {
                if (n < cnt)
                    w.put(off + n, b);
                ++n;
            };
            run(emit);
            return;
        }
        const u32 bits = key_bits(t.num_objects);
        if (cnt <= k_local_keys && arity * bits <= 64)
        {
            // packed keys (lexicographic order = numeric order), sorted by insertion
            u64 keys[k_local_keys];
            u32 n = 0;
            auto emit = [&](const u32* b)
            {
                if (n < k_local_keys)
                {
                    u64 k = 0;
                    for (u32 i = 0; i < arity; ++i)
                        k = (k << bits) | b[i];
                    keys[n] = k;
                }
                ++n;
            };
            run(emit);
            const u32 m = n < cnt ? n : static_cast<u32>(cnt);
            for (u32 i = 1; i < m; ++i)
            {
                const u64 k = keys[i];
                u32 j = i;
                for (; j > 0 && keys[j - 1] > k; --j)
                    keys[j] = keys[j - 1];
                keys[j] = k;
            }
            const u64 mask = (1ull << bits) - 1;
            u32 b[k_max_label];
            for (u32 i = 0; i < m; ++i)
            {
                u64 k = keys[i];
                for (u32 j = arity; j-- > 0;)
                {
                    b[j] = static_cast<u32>(k & mask);
                    k >>= bits;
                }
                w.put(off + i, b);
            }
            return;
        }
        // long segments: one search into a bitmap of the bindings' canonical ranks where the static domains allow
        // (RankSpace), read back in order
        if (const RankSpace<OW> ranks(t, mx, arity); ranks.size)
        {
            u64 bm[k_rank_words];
            const u32 nw = (ranks.size + 63) >> 6;
            for (u32 i = 0; i < nw; ++i)
                bm[i] = 0;
            bool outside = false;
            auto emit = [&](const u32* b)
            {
                const u32 r = ranks.rank(b);
                if (r == k_none)
                    outside = true;
                else
                    bm[r >> 6] |= 1ull << (r & 63);
            };
            run(emit);
            if (!outside)
            {
                u32 b[k_max_label];
                u64 n = 0;
                for (u32 i = 0; i < nw && n < cnt; ++i)
                    for (u64 x = bm[i]; x && n < cnt; x &= x - 1)
                    {
                        ranks.unrank(i * 64 + lowest_bit(x), b);
                        w.put(off + n++, b);
                    }
                return;
            }
        }
        // otherwise the bindings are sorted in global memory (the label rows, or the scratch)
        const Labels& lab = a.lab;
        const u32 L = lab.label_width;
        const bool straddle = off + cnt > lab.capacity;
        u32* dst = nullptr;
        if (lab.binding && !straddle)
            dst = lab.binding + off * L;
        else if (lab.scratch && lab.scratch_indexed && off + cnt <= lab.scratch_rows)
            dst = lab.scratch + off * L;
        else if (lab.scratch && !lab.scratch_indexed && cnt <= lab.scratch_rows)
            dst = lab.scratch;  // the one segment that straddles the capacity
        if (!dst)
        {
            if (lab.error)
                atomicOr(lab.error, 1u);
            return;
        }
        u64 n = 0;
        auto emit = [&](const u32* b)
        {
            if (n < cnt)
            {
                u32* r = dst + n * L;
                for (u32 i = 0; i < arity; ++i)
                    r[i] = b[i];
            }
            ++n;
        };
        run(emit);
        sort_rows(dst, cnt, L, arity);
        for (u64 j = 0; j < cnt; ++j)
            w.put(off + j, dst + j * L);
    }
}

/// k_tile states per block, one lane per state; the block's warps take the launch's schemas in turn (a warp shares
/// one schema, as in the probe's schema-major mapping, and the warps of a block read the same states' views).
constexpr u32 k_tile = 32;

/// A multi-instance launch: the row of launch position j and its instance k; false when the row belongs to
/// another launch (another OW group or row-width bucket).
template<u32 OW>
__device__ __forceinline__ bool multi_row(const Multi& m, u64 j, u32& li, u32& k)
{
    j += m.first;
    li = static_cast<u32>(j);
    if (m.order && !(m.order_bad && *m.order_bad))
        li = m.order[j * m.order_stride];
    k = m.inst[li];
    if (k >= m.instances)
        return false;  // an invalid task id (the caller's check reports it)
    const TaskView& t = m.views[k];
    if (OW && t.ow != OW)
        return false;
    const u32 w = t.state_words ? t.state_words : 1;  // an instance without fluent atoms is in the first bucket
    return w > m.words_lo && w <= m.words_hi;
}

/// Whether instance k runs `schema` with the launch's kind of matcher (multi-instance launches; Multi::fc_of: 0 fixed
/// order, 1 forward checking, 2 never applicable: no launch runs it, its count is 0).
template<bool FC>
__device__ __forceinline__ bool multi_kind(const Multi& m, u32 k, u32 schema)
{
    return m.fc_of[u64{k} * m.num_schemas + schema] == (FC ? 1 : 0);
}

template<u32 OW, bool FC, bool Write, u32 WB, bool CE, bool M>
__device__ __forceinline__ void gen_row(const GenArgs& a, const TaskView& t, u32 li, u32 k, u32 warp, u32 nwarps)
{
    const Parents& p = a.p;
    const u64* view = a.v.data + u64{li} * a.v.words;
    const u64* row = p.data + u64{li} * p.stride;
    const u64* der = p.derived ? p.derived + u64{li} * p.derived_words : nullptr;
    for (u32 si = warp; si < a.set.count; si += nwarps)
    {
        if constexpr (M)
        {
            if (!multi_kind<FC>(a.m, k, a.set.schemas[si]))
                continue;
        }
        gen_one<OW, FC, Write, WB, CE>(a, t, si, li, view, row, der);
    }
}

template<u32 OW, bool FC, bool Write, u32 WB, bool CE, bool M>
__global__ void __launch_bounds__(k_gen_block) k_gen(GenArgs a)
{
    const Parents& p = a.p;
    const u32 base = blockIdx.x * k_tile;
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    if (base + lane >= live_rows(p))
        return;
    if constexpr (Write)
        if (a.lab.live && *a.lab.live == 0)
            return;
    if constexpr (M)
    {
        u32 li, k;
        if (!multi_row<OW>(a.m, base + lane, li, k))
            return;
        gen_row<OW, FC, Write, WB, CE, true>(a, a.m.views[k], li, k, warp, nwarps);
    }
    else
        gen_row<OW, FC, Write, WB, CE, false>(a, a.t, base + lane, 0, warp, nwarps);
}

// ------------------------------------------------------------------------------------------------ few rows

/// Launches of at most k_warp_rows rows (their capacity) search each (row, schema) by a warp (k_gen_warp): with a lane
/// per row most of the device idles while a few threads search alone (rollouts' chunks generate from a few dozen
/// distinct states).
constexpr u32 k_gen_warp_block = 128;

/// Ascending sort of the warp's 32 values, one per lane (bitonic, over shuffles).
__device__ __forceinline__ u64 warp_sort(u64 v, u32 lane)
{
#pragma unroll
    for (u32 k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (u32 j = k >> 1; j > 0; j >>= 1)
        {
            const u64 o = __shfl_xor_sync(k_full_mask, v, static_cast<int>(j));
            const bool keep_min = ((lane & j) == 0) == ((lane & k) == 0);
            v = keep_min ? (o < v ? o : v) : (o > v ? o : v);
        }
    return v;
}

/// gen_one by a warp: the lanes split the search (run_fixed / run_fc, Exact: each binding once) and the rows go out in
/// the canonical order of gen_one: counts summed over the lanes; packed keys of segments of up to 32 rows gathered at
/// the lanes' prefix positions and sorted across the warp; a rank bitmap OR-ed over the warp, its bits dealt in order;
/// otherwise the bindings land in the segment's rows (a per-warp cursor: unique rows, sorted before the write, so the
/// slots they land in do not matter). Lane 0 runs gen_one where the lanes cannot keep its order: a non-canonical launch
/// of a schema with more than one parameter (the matcher's own order), and a long segment without room to sort that
/// gen_one writes as found (a matcher that binds in parameter order needs none).
template<u32 OW, bool FC, bool Write, u32 WB, bool CE>
__device__ __forceinline__ void gen_one_warp(const GenArgs& a, const TaskView& t, u32 si, u32 li, const u64* view,
                                             const u64* row, const u64* der, u32 lane, u64* sh_keys, u32* sh_cursor)
{
    const u32 schema = a.set.schemas[si];
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[a.set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const Parents& p = a.p;
    const StateRef s{row, p.words, der, p.derived_words};
    const u64 seg = u64{li} * a.set.num_schemas + schema;
    u32 bind[k_max_depth];
    auto run = [&](auto& emit)
    {
        if constexpr (FC)
            run_fc<OW, true, true>(t, mx, s, view, bind, emit, 32, lane);
        else
            run_fixed<OW, k_max_depth, true, true>(t, mx, s, view, bind, emit, 32, lane);
    };
    if constexpr (!Write)
    {
        u32 n = 0;
        auto emit = [&](const u32*) { ++n; };
        run(emit);
        n = __reduce_add_sync(k_full_mask, n);
        if (lane == 0)
            a.counts[seg] = n;
    }
    else
    {
        const u64 off = a.offsets[seg], cnt = a.offsets[seg + 1] - off;
        if (cnt == 0)
            return;
        const RowWriter<WB, CE ? OW : 0> w(a, t, sc, schema, li, row);
        const u32 arity = w.arity;
        if (off >= a.lab.capacity)
        {
            if (!a.out.words_needed && !a.out.missing_flag)
                return;
            auto emit = [&](const u32* b) { w.put(a.lab.capacity, b); };
            run(emit);
            return;
        }
        if (!a.set.canonical && arity > 1)
        {
            if (lane == 0)
                gen_one<OW, FC, Write, WB, CE>(a, t, si, li, view, row, der);
            return;
        }
        // every other order is the bindings' lexicographic one (canonical, or the matcher's own: one parameter)
        const u32 bits = key_bits(t.num_objects);
        if (cnt <= 32 && arity * bits <= 64)
        {
            u64 keys[32];
            u32 n = 0;
            auto emit = [&](const u32* b)
            {
                if (n < 32)
                {
                    u64 k = 0;
                    for (u32 i = 0; i < arity; ++i)
                        k = (k << bits) | b[i];
                    keys[n] = k;
                }
                ++n;
            };
            run(emit);
            u32 at = n;  // the lanes' inclusive prefix of their counts
#pragma unroll
            for (u32 d = 1; d < 32; d <<= 1)
            {
                const u32 y = __shfl_up_sync(k_full_mask, at, d);
                if (lane >= d)
                    at += y;
            }
            at -= n;
            for (u32 i = 0; i < n && at + i < 32; ++i)
                sh_keys[at + i] = keys[i];
            __syncwarp();
            const u64 k = warp_sort(lane < cnt ? sh_keys[lane] : ~0ull, lane);
            if (lane < cnt)
            {
                const u64 mask = (1ull << bits) - 1;
                u32 b[k_max_label];
                u64 x = k;
                for (u32 j = arity; j-- > 0;)
                {
                    b[j] = static_cast<u32>(x & mask);
                    x >>= bits;
                }
                w.put(off + lane, b);
            }
            return;
        }
        if (const RankSpace<OW> ranks(t, mx, arity); ranks.size)
        {
            u64 bm[k_rank_words];
            const u32 nw = (ranks.size + 63) >> 6;
            for (u32 i = 0; i < nw; ++i)
                bm[i] = 0;
            bool outside = false;
            auto emit = [&](const u32* b)
            {
                const u32 r = ranks.rank(b);
                if (r == k_none)
                    outside = true;
                else
                    bm[r >> 6] |= 1ull << (r & 63);
            };
            run(emit);
            if (!__any_sync(k_full_mask, outside))
            {
                u32 b[k_max_label];
                u64 n = 0;
                for (u32 i = 0; i < nw && n < cnt; ++i)
                {
                    const u64 lo = __reduce_or_sync(k_full_mask, static_cast<u32>(bm[i]));
                    const u64 hi = __reduce_or_sync(k_full_mask, static_cast<u32>(bm[i] >> 32));
                    for (u64 x = hi << 32 | lo; x && n < cnt; x &= x - 1, ++n)
                        if ((n & 31) == lane)
                        {
                            ranks.unrank(i * 64 + lowest_bit(x), b);
                            w.put(off + n, b);
                        }
                }
                return;
            }
        }
        const Labels& lab = a.lab;
        const u32 L = lab.label_width;
        const bool straddle = off + cnt > lab.capacity;
        u32* dst = nullptr;
        if (lab.binding && !straddle)
            dst = lab.binding + off * L;
        else if (lab.scratch && lab.scratch_indexed && off + cnt <= lab.scratch_rows)
            dst = lab.scratch + off * L;
        else if (lab.scratch && !lab.scratch_indexed && cnt <= lab.scratch_rows)
            dst = lab.scratch;
        if (!dst)
        {
            if (lane == 0)
            {
                if (!(a.set.canonical && arity > 1 && device_sorts(mx[k_mc_flags], FC)))
                    gen_one<OW, FC, Write, WB, CE>(a, t, si, li, view, row, der);
                else if (lab.error)
                    atomicOr(lab.error, 1u);
            }
            return;
        }
        if (lane == 0)
            *sh_cursor = 0;
        __syncwarp();
        auto emit = [&](const u32* b)
        {
            const u32 at = atomicAdd(sh_cursor, 1u);
            if (at < cnt)
            {
                u32* r = dst + u64{at} * L;
                for (u32 i = 0; i < arity; ++i)
                    r[i] = b[i];
            }
        };
        run(emit);
        __syncwarp();
        if (lane == 0)
            sort_rows(dst, cnt, L, arity);
        __syncwarp();
        for (u64 j = lane; j < cnt; j += 32)
            w.put(off + j, dst + j * L);
    }
}

/// A warp per (row, schema), rows ascending (the warps of a block mostly share a row's view).
template<u32 OW, bool FC, bool Write, u32 WB, bool CE>
__global__ void __launch_bounds__(k_gen_warp_block) k_gen_warp(GenArgs a)
{
    constexpr u32 k_warps = k_gen_warp_block / 32;
    __shared__ u64 s_keys[k_warps][32];
    __shared__ u32 s_cursor[k_warps];
    const u32 lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    const u64 item = u64{blockIdx.x} * k_warps + wib;
    const u32 S = a.set.count;
    const u64 li = item / S;
    if (li >= live_rows(a.p))
        return;
    if constexpr (Write)
        if (a.lab.live && *a.lab.live == 0)
            return;
    const Parents& p = a.p;
    const u64* view = a.v.data + li * a.v.words;
    const u64* row = p.data + li * p.stride;
    const u64* der = p.derived ? p.derived + li * p.derived_words : nullptr;
    gen_one_warp<OW, FC, Write, WB, CE>(a, a.t, static_cast<u32>(item % S), static_cast<u32>(li), view, row, der, lane,
                                        s_keys[wib], s_cursor + wib);
}

template<u32 OW, bool FC, bool Write, u32 WB, bool CE, bool M = false>
cudaError_t launch_gen(const GenArgs& a, cudaStream_t s)
{
    const u64 blocks = (u64{a.p.rows} + k_tile - 1) / k_tile;
    if (blocks == 0 || a.set.count == 0)
        return cudaSuccess;
    if (blocks > 0x7FFFFFFFull)
        return cudaErrorInvalidValue;
    if constexpr (!M)
    {
        if (a.p.rows <= k_warp_rows)
        {
            const u64 warps = u64{a.p.rows} * a.set.count;
            const u64 wblocks = (warps + k_gen_warp_block / 32 - 1) / (k_gen_warp_block / 32);
            k_gen_warp<OW, FC, Write, WB, CE><<<static_cast<unsigned>(wblocks), k_gen_warp_block, 0, s>>>(a);
            return cudaGetLastError();
        }
    }
    const u32 warps = a.set.count < k_gen_block / 32 ? a.set.count : k_gen_block / 32;
    k_gen<OW, FC, Write, WB, CE, M><<<static_cast<unsigned>(blocks), warps * 32, 0, s>>>(a);
    return cudaGetLastError();
}

/// The kernels' OW is the task's (single instance: a.t.ow) or the launch's OW group (multi-instance: a.m.ow): the tables
/// are OW-strided (view rows, domains, static rows), so the instantiation must equal the rows' OW.
template<bool Write, u32 WB, bool CE = false, bool M = false>
cudaError_t dispatch_ow(const GenArgs& a, cudaStream_t s)
{
    if (a.set.deep)
        return cudaErrorInvalidValue;  // dispatch_deep
    const u32 ow = M ? a.m.ow : a.t.ow;
    if (a.set.fc)
    {
        switch (ow)
        {
            case 1: return launch_gen<1, true, Write, WB, CE, M>(a, s);
            case 2: return launch_gen<2, true, Write, WB, CE, M>(a, s);
            case 3: return launch_gen<3, true, Write, WB, CE, M>(a, s);
            case 4: return launch_gen<4, true, Write, WB, CE, M>(a, s);
            default: return cudaErrorInvalidValue;
        }
    }
    switch (ow)
    {
        case 1: return launch_gen<1, false, Write, WB, CE, M>(a, s);
        case 2: return launch_gen<2, false, Write, WB, CE, M>(a, s);
        case 3: return launch_gen<3, false, Write, WB, CE, M>(a, s);
        case 4: return launch_gen<4, false, Write, WB, CE, M>(a, s);
        case 5: return launch_gen<5, false, Write, WB, CE, M>(a, s);
        case 6: return launch_gen<6, false, Write, WB, CE, M>(a, s);
        case 7: return launch_gen<7, false, Write, WB, CE, M>(a, s);
        case 8: return launch_gen<8, false, Write, WB, CE, M>(a, s);
        default: return cudaErrorInvalidValue;
    }
}

// ------------------------------------------------------------------------------------------------ deferred successors

/// The segment holding row j of a write: the last of the nseg ascending offsets (rows * S + 1, the last the total)
/// that is at most j (j below the total).
__device__ __forceinline__ u64 segment_of(const u32* offsets, u64 nseg, u64 j)
{
    u64 lo = 0, hi = nseg - 1;  // offsets[lo] <= j < offsets[hi]
    while (hi - lo > 1)
    {
        const u64 mid = lo + (hi - lo) / 2;
        if (offsets[mid] <= j)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

constexpr unsigned k_put_block = 128;
constexpr u64 k_put_max_blocks = 2048;

/// The lanes of launch_put's group of a row: G lanes (a power of two up to a warp), this thread's `sub`, the group's
/// lanes of the warp `gm`.
struct PutGroup
{
    u32 G, sub, gm;

    /// The maximum of x over the group.
    [[nodiscard]] __device__ __forceinline__ u32 max(u32 x) const
    {
        for (u32 o = G >> 1; o; o >>= 1)
        {
            const u32 y = __shfl_xor_sync(gm, x, o, G);
            x = x > y ? x : y;
        }
        return x;
    }
};

/// Row j's successor (launch_put) by one group of lanes, rl::expand's rules as RowWriter::successor: lane `sub` holds
/// words sub and sub + G of the row (rows of at most 2G words): the parent's words are read and the row is written
/// coalesced, without a copy of the row per thread; the deletes and adds are resolved one per lane and passed round the
/// group (the deletes first). A row that does not fit out.out_words (an added atom without a slot, a parent or an add
/// past it) is zeroed, and (no atom missing) the exact trimmed width of the successor goes to words_needed. A row whose
/// label holds no binding of the schema is skipped: a write leaves the rows of a segment it cannot sort unlabeled (no
/// scratch for a segment that straddles the capacity: Labels::error, and the caller redoes the chunk).
template<bool M>
__device__ __forceinline__ void put_row(const GenArgs& a, const PutRows& to, const PutGroup& g, const TaskView& t,
                                        u32 schema, u32 li, u64 j)
{
    const Labels& lab = a.lab;
    const SuccessorRows& out = a.out;
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* b = lab.binding + j * lab.label_width;
    const u32 arity = sc[k_sc_arity];
    for (u32 i = 0; i < arity; ++i)
        if (b[i] >= t.num_objects)
            return;
    // the destination: row j, or (mapped) the state's successor j - offsets[its first segment] at its batch row's
    const u32 at = lab.parent_base + li;
    const u64 pos = to.rows ? static_cast<u64>(to.batch_offsets[to.rows[at]]) + (j - a.offsets[u64{li} * a.set.num_schemas])
                            : j;
    const bool inside = !to.rows || pos < to.capacity;
    // the parent's words (over several instances, its instance's: the words past them are zero)
    const u64* st = a.p.data + u64{li} * a.p.stride;
    u32 pw = a.p.words;
    if constexpr (M)
    {
        const u32 iw = t.state_words ? t.state_words : 1;  // frozen slots: every successor fits
        pw = iw < pw ? iw : pw;
    }
    const u32 i0 = g.sub, i1 = g.sub + g.G;
    u64 y0 = i0 < pw ? st[i0] : 0, y1 = i1 < pw ? st[i1] : 0;
    const u32 ptrim = g.max(y1 ? i1 + 1 : y0 ? i0 + 1 : 0);
    // the deletes, then the adds: item k's slot is resolved by lane k mod G and passed round the group
    const u32 nd = sc[k_sc_dels_n], items = nd + sc[k_sc_adds_n];
    u32 max_add = 0;  // highest added word + 1
    bool missing = false;
    for (u32 r = 0; r < items; r += g.G)
    {
        u32 mine = k_none;
        if (const u32 k = r + g.sub; k < items)
        {
            const u64 key = k < nd ? pattern_key(t, sc[k_sc_dels] + k, b) : pattern_key(t, sc[k_sc_adds] + (k - nd), b);
            mine = slot_of(t, key);
            if (mine == k_none && k >= nd && out.missing_bits && key < t.atom_total)
                atomicOr(out.missing_bits + (key >> 5), 1u << (key & 31));
        }
        const u32 n = items - r < g.G ? items - r : g.G;
        for (u32 q = 0; q < n; ++q)
        {
            const u32 slot = __shfl_sync(g.gm, mine, q, g.G);
            const bool add = r + q >= nd;
            if (slot == k_none)
            {
                missing = missing || add;
                continue;
            }
            const u32 w = slot >> 6;
            const u64 m = 1ull << (slot & 63);
            if (add)
                max_add = max_add > w + 1 ? max_add : w + 1;
            if (w == i0)
                y0 = add ? (y0 | m) : (y0 & ~m);
            else if (w == i1)
                y1 = add ? (y1 | m) : (y1 & ~m);
        }
    }
    if (missing && out.missing_flag && g.sub == 0)
        *out.missing_flag = 1;
    const u32 ow = out.out_words;
    const bool fits = !missing && ptrim <= ow && max_add <= ow;
    if (!fits && !missing && out.words_needed)
    {
        // the exact trimmed width of the successor (rl::expand reports it for rows that do not fit): its highest word
        // held here, or an add past them
        const u32 hi = g.max(y1 ? i1 + 1 : y0 ? i0 + 1 : 0);
        if (g.sub == 0)
            atomicMax(out.words_needed, hi > max_add ? hi : max_add);
    }
    if (!inside)
        return;
    if (out.words)
    {
        u64* row = out.words + pos * ow;
        // over several instances, a row written over its parent keeps the parent's zero words past its instance's
        // (the env's pick)
        const u32 lim = !fits ? ow : M && out.words == a.p.data && ow == a.p.stride && pos == li && pw < ow ? pw : ow;
        if (i0 < lim)
            row[i0] = fits ? y0 : 0;
        if (i1 < lim)
            row[i1] = fits ? y1 : 0;
    }
    if (to.rows)
    {
        // the labels at the batch row, widened (-1 past the part's label columns)
        if (to.binding)
            for (u32 k = g.sub; k < to.label_width; k += g.G)
                to.binding[pos * to.label_width + k] = k < lab.label_width ? b[k] : 0xFFFFFFFFu;
        if (g.sub == 0)
        {
            if (to.schema)
                to.schema[pos] = schema;
            if (to.parent)
                to.parent[pos] = to.rows[at];
        }
    }
    if (to.goal && out.words)
    {
        // the goal test of the written row (goals without derived literals), the goal checks split over the group
        __syncwarp(g.gm);
        const u64* row = out.words + pos * ow;
        bool ok = !t.goal_unsatisfiable;
        for (u32 i = g.sub; i < t.goal_check_n && ok; i += g.G)
            ok = holds(t, t.goal_check + i, nullptr, row, ow, nullptr, 0);
        ok = __all_sync(g.gm, ok);
        if (g.sub == 0)
            to.goal[pos] = ok ? 1 : 0;
    }
}

/// launch_put: the successor of every labeled row below min(total, capacity) whose schema is deferred (`deferred`
/// [S], null: every schema), a group of 2^shift lanes per row (put_row); over several instances, the rows of every
/// instance (task ids outside [0, m.instances) have no rows). The groups stride over the rows; a row's segment comes
/// from its parent and schema labels or the offsets.
template<bool M>
__global__ void __launch_bounds__(k_put_block) k_put(GenArgs a, PutRows to, const u8* deferred, u32 shift)
{
    if ((a.lab.live && *a.lab.live == 0) || (a.lab.error && *a.lab.error))
        return;  // no rows, or a chunk its caller redoes
    const u32 S = a.set.num_schemas, live = live_rows(a.p);
    const u64 nseg = u64{live} * S + 1, total = a.offsets[nseg - 1];
    const u64 end = total < a.lab.capacity ? total : a.lab.capacity;
    const u32 G = 1u << shift, lane = threadIdx.x & 31;
    const PutGroup g{G, lane & (G - 1), (G == 32 ? 0xFFFFFFFFu : (1u << G) - 1) << (lane & ~(G - 1))};
    const u64 step = (u64{gridDim.x} * blockDim.x) >> shift;
    for (u64 j = (u64{blockIdx.x} * blockDim.x + threadIdx.x) >> shift; j < end; j += step)
    {
        u32 li, schema;
        if (a.lab.parent && a.lab.schema)
        {
            li = a.lab.parent[j] - a.lab.parent_base;
            schema = a.lab.schema[j];
            // only labels that name row j's own segment: a row no device list wrote (a CPU-fallback row) may hold
            // another chunk's labels
            if (li >= live || schema >= S)
                continue;
            const u64 seg = u64{li} * S + schema;
            if (j < a.offsets[seg] || j >= a.offsets[seg + 1])
                continue;
        }
        else
        {
            const u64 seg = segment_of(a.offsets, nseg, j);
            li = static_cast<u32>(seg / S);
            schema = static_cast<u32>(seg % S);
        }
        if (deferred && !deferred[schema])
            continue;
        if constexpr (M)
        {
            const u32 k = a.m.inst[li];
            if (k >= a.m.instances)
                continue;
            put_row<true>(a, to, g, a.m.views[k], schema, li, j);
        }
        else
            put_row<false>(a, to, g, a.t, schema, li, j);
    }
}

/// k_put over about rows_hint rows: two words of the row per lane (a group as wide as half the wider of the parent and
/// output rows, at most a warp).
template<bool M>
cudaError_t launch_put_rows(const GenArgs& a, const PutRows& to, const u8* deferred, u64 rows_hint, cudaStream_t s)
{
    const u32 ow = a.out.out_words, nw = a.p.words > ow ? a.p.words : ow;
    if (nw > k_max_words || (to.rows && !to.batch_offsets))
        return cudaErrorInvalidValue;
    if (a.p.rows == 0 || rows_hint == 0)
        return cudaSuccess;
    u32 shift = 0;
    while (shift < 5 && (2u << shift) < nw)
        ++shift;
    const u64 want = ((rows_hint << shift) + k_put_block - 1) / k_put_block;
    const auto blocks = static_cast<unsigned>(want < k_put_max_blocks ? want : k_put_max_blocks);
    k_put<M><<<blocks, k_put_block, 0, s>>>(a, to, deferred, shift);
    return cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------ deep matchers

/// The deep kernels (SchemaSet::deep: matchers of up to k_deep_depth = 32 parameters) search a (state, schema) segment
/// with a warp, one lane per parameter. Every search node is a chain of dependent loads (checks, edges, table rows,
/// the trail in local memory); with one thread per segment, the 32 lanes of a warp walked 32 different trees and that
/// latency was never hidden. A warp walks one tree in lock step instead, and the
/// work of a node is spread over its lanes: the checks of a binding, its forward-checking edges (shared-memory atomics
/// on the domains), the fail-first choice (a warp minimum) and the domains' snapshots (lane v keeps parameter v's).
/// A launch runs two passes (SchemaSet::work). The first gives every segment a warp (warps take segments from a queue as
/// they finish) and a budget of search nodes; a search that runs out of it leaves its segment to the second pass. The
/// second splits each such segment over up to k_deep_split warps: they walk the same tree down to a level with enough
/// candidates and claim that level's candidates one at a time as they finish (the subtrees below a level differ widely
/// in size). organic-synthesis' heavy segments are fruitless searches of thousands of nodes among segments of a dozen.
/// The kernels use no named barriers: a kernel with a barrier id in a register reserves all 16 and gets one block per SM.
constexpr u32 k_deep_block = 256;
constexpr u32 k_deep_block_warps = k_deep_block / 32;
/// The first pass's budget in search nodes (a node: a candidate tried or a level left), and its floor for launches of
/// fewer segments than the device holds warps (the budget shrinks with them: the second pass has the room to split).
constexpr u32 k_deep_budget = 1024;
constexpr u32 k_deep_budget_min = 64;
/// The second pass: at most this many warps per segment (a power of two), and the candidates per warp of the level
/// they deal (the first level whose paths reach warps x k_deep_deal; it must be the level of an effect-relevant
/// parameter, before the witness search, so that each binding is found by one warp).
constexpr u32 k_deep_split = 32;
constexpr u32 k_deep_deal = 8;
static_assert(k_deep_split * k_deep_deal <= 256, "DeepSplit::enter: paths up to 256 x 256");

/// A warp's shared memory: the domains of forward checking (lane v: parameter v), the binding and a scratch row.
template<u32 OW>
struct DeepWarp
{
    u64 dom[k_deep_depth * OW];
    u32 bind[k_deep_depth];
    u32 tmp[32];
};

/// SchemaSet::work of a launch over `segs` segments (its capacity: rows x count; lifted::deep_work_words).
struct DeepWork
{
    u32* ctl;     // [0] the first pass's queue, [1] segments left to the second pass, [2] the second pass's queue
    u32* list;    // the segments left to the second pass (launch indices)
    u32* ticket;  // per listed segment: its dealt candidates claimed
    u32* slots;   // per listed segment: its rows gathered (write)
    u32* done;    // per listed segment: its warps finished (write)
    __device__ DeepWork(u32* w, u64 segs)
        : ctl(w), list(w + k_deep_ctl), ticket(list + segs), slots(ticket + segs), done(slots + segs)
    {
    }
};

/// Warps per segment of the second pass: a power of two up to k_deep_split, `heavy` segments x the split at most twice
/// the launch's warps.
__device__ __forceinline__ u32 deep_split(u32 heavy, u32 warps)
{
    u32 S = 1;
    while (S < k_deep_split && u64{heavy} * S * 2 <= 2 * u64{warps})
        S *= 2;
    return S;
}

/// prologue() by a warp: the literals without free parameters, one per lane.
__device__ __forceinline__ bool warp_prologue(const TaskView& t, const u32* mx, const u32* bind, const StateRef& s, u32 lane)
{
    if (mx[k_mc_flags] & k_mc_never)
        return false;
    const u32 pc = mx[k_mc_pre_checks], pn = mx[k_mc_pre_checks_n];
    bool ok = true;
    for (u32 i = lane; i < pn && ok; i += 32)
        ok = check(t, pc + i, bind, s);
    return __all_sync(k_full_mask, ok);
}

/// A warp's share of a search: `n` warps on it (this one `i`), its budget (the first pass), the dealt level, its
/// candidates reached so far and the one this warp claimed (from `ticket`), the paths down to level l (lane l, capped
/// at n x k_deep_deal). No level is dealt after the first leaf (a search too small to deal stays with warp 0, the others
/// stop at the leaf).
struct DeepSplit
{
    u32 n = 1, i = 0;
    u32* ticket = nullptr;
    u32 budget = 0xFFFFFFFFu;
    int level = -1;
    u32 seen = 0, mine = 0;
    bool claimed = false, leafed = false, over = false;
    u32 path = 0;

    /// Level l (with `cnt` candidates) is entered; `open`: it may be dealt (effect-relevant, not the witness search).
    __device__ __forceinline__ void enter(int l, u32 cnt, bool open, u32 lane)
    {
        if (n == 1 || level >= 0)
            return;
        const u32 deal = n * k_deep_deal;
        const u32 above = l > 0 ? __shfl_sync(k_full_mask, path, l - 1) : 1;
        const u32 p = cnt >= deal || above * cnt >= deal ? deal : above * cnt;
        if (lane == static_cast<u32>(l))
            path = p;
        if (!leafed && open && cnt > 1 && p >= deal)
            level = l;
    }
    /// A search node: false (the search stops, `over`) when the budget has run out.
    __device__ __forceinline__ bool node()
    {
        if (budget == 0)
        {
            over = true;
            return false;
        }
        --budget;
        return true;
    }
    /// Whether this warp skips the next candidate of level l: the dealt level's candidates are numbered in the walk's
    /// order (the same for every warp); a warp past its claimed one claims the next unclaimed number (it lies ahead).
    __device__ __forceinline__ bool skip(int l, u32 lane)
    {
        if (n == 1 || l != level)
            return false;
        const u32 k = seen++;
        if (!claimed || k > mine)
        {
            u32 c = 0;
            if (lane == 0)
                c = atomicAdd(ticket, 1u);
            mine = __shfl_sync(k_full_mask, c, 0);
            claimed = true;
        }
        return k != mine;
    }
    /// At a leaf: false if this warp leaves it to warp 0 (nothing dealt).
    __device__ __forceinline__ bool leaf()
    {
        if (n > 1 && level < 0 && i != 0)
            return false;
        leafed = true;
        return true;
    }
};

/// The candidates of a level: lane `d` keeps them (OW words); w the first nonzero word or OW.
template<u32 OW>
__device__ __forceinline__ u32 warp_next(u64 (&mine)[OW], u32 d, u32 lane, u32& o)
{
    u64 c[OW];
#pragma unroll
    for (u32 x = 0; x < OW; ++x)
        c[x] = __shfl_sync(k_full_mask, mine[x], d);
    u32 w = 0;
    while (w < OW && c[w] == 0)
        ++w;
    if (w == OW)
        return OW;
    o = w * 64 + lowest_bit(c[w]);
    if (lane == d)
    {
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
            if (x == w)
                mine[x] &= mine[x] - 1;
    }
    return w;
}

/// run_fixed by a warp (the same search, so the same bindings in the same order): lane d holds level d's static domain
/// and its candidates still to try; a level's rows (bindings above it) and a binding's checks are spread over the lanes.
/// emit() is called by the whole warp at each binding (in `bind`).
template<u32 OW, class Emit>
__device__ void warp_fixed(const TaskView& t, const u32* mx, const StateRef& s, const u64* view, u32* bind, Emit& emit,
                           DeepSplit& sp)
{
    const u32 lane = threadIdx.x & 31;
    if (!warp_prologue(t, mx, bind, s, lane))
        return;
    const u32 nd = mx[k_mc_steps_n];
    if (nd == 0)
    {
        if (sp.i == 0)
            emit();
        return;
    }
    const u32* steps = t.step + u64{mx[k_mc_steps]} * 5;
    u64 dom1[OW], cd[OW];
#pragma unroll
    for (u32 x = 0; x < OW; ++x)
        dom1[x] = cd[x] = 0;
    const bool empty = lane < nd && !domain<OW>(t, mx, view, steps[lane * 5], dom1);
    if (__any_sync(k_full_mask, empty))
        return;
    const u64* rows = t.row + u64{mx[k_mc_rows]} * 3;
    const u32* step_checks = t.index + mx[k_mc_step_checks];
    const u32 checks0 = mx[k_mc_checks];
    const int fe = static_cast<int>(mx[k_mc_first_exist]);
    // level d's candidates: its domain and the rows of its step (over the bindings above it); the count
    auto fill = [&](int d)
    {
        const u32* st = steps + d * 5;
        u64 a[OW];
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
            a[x] = ~u64{0};
        for (u32 r = st[1] + lane; r < st[2]; r += 32)
        {
            const u64* rw = rows + u64{r} * 3;
            const u64* x = table_row<OW>(t, view, rw[0], bind[rw[1]]);
            const bool neg = rw[2] != 0;
#pragma unroll
            for (u32 w = 0; w < OW; ++w)
                a[w] &= neg ? ~x[w] : x[w];
        }
        u32 n = 0;
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
        {
            const u64 lo = __reduce_and_sync(k_full_mask, static_cast<u32>(a[x]));
            const u64 hi = __reduce_and_sync(k_full_mask, static_cast<u32>(a[x] >> 32));
            const u64 m = (hi << 32 | lo) & __shfl_sync(k_full_mask, dom1[x], d);
            if (lane == static_cast<u32>(d))
                cd[x] = m;
            n += static_cast<u32>(__popcll(m));
        }
        sp.enter(d, n, d < fe, lane);
    };
    int d = 0;
    bool wit = fe == 0;
    fill(0);
    const int last = static_cast<int>(nd) - 1;
    while (d >= 0)
    {
        if (!sp.node())
            return;
        u32 o = 0;
        if (warp_next<OW>(cd, static_cast<u32>(d), lane, o) == OW)
        {
            --d;
            if (d < fe)
                wit = false;
            continue;
        }
        if (sp.skip(d, lane))
            continue;
        const u32* st = steps + d * 5;
        // the lanes' reads of `bind` since the last sync (fill()'s rows, the checks, the emit) come first
        __syncwarp();
        if (lane == 0)
            bind[st[0]] = o;
        __syncwarp();
        bool ok = true;
        for (u32 c = st[3] + lane; c < st[4] && ok; c += 32)
            ok = check(t, checks0 + step_checks[c], bind, s);
        if (!__all_sync(k_full_mask, ok))
            continue;
        if (d == last)
        {
            if (!sp.leaf())
                return;
            emit();
            if (wit)
            {
                d = fe - 1;  // one witness suffices: back to the last effect-relevant step
                wit = false;
            }
            continue;
        }
        ++d;
        if (d == fe)
            wit = true;
        fill(d);
    }
}

/// run_fc by a warp (the same search: fail-first forward checking with witness pruning, the same bindings in the same
/// order). Lane v keeps parameter v: its domain in shared memory (D, OW words per parameter) and the domain's snapshot at
/// each level's entry (a candidate restores them, as run_fc's stack of levels); lane l keeps level l's parameter and its
/// candidates still to try. A binding's checks and edges are spread over the lanes (the edges narrow their targets'
/// domains by shared-memory atomics), the next parameter is a warp minimum over (domain size, place among the free
/// parameters). emit() is called by the whole warp at each binding (in `bind`). `tmp`: 32 words of the warp's shared
/// memory.
template<u32 OW, class Emit>
__device__ void warp_fc(const TaskView& t, const u32* mx, const StateRef& s, const u64* view, u64* D, u32* bind, u32* tmp,
                        Emit& emit, DeepSplit& sp)
{
    const u32 lane = threadIdx.x & 31;
    if (!warp_prologue(t, mx, bind, s, lane))
        return;
    const u32 nd = mx[k_mc_steps_n];
    if (nd == 0)
    {
        if (sp.i == 0)
            emit();
        return;
    }
    const u32 total = mx[k_mc_total];
    const u32* steps = t.step + u64{mx[k_mc_steps]} * 5;
    u64* mine = D + lane * OW;
    auto dom_and = [&](u32 v, const u64* row, bool neg)
    {
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
            atomicAnd(reinterpret_cast<unsigned long long*>(D + v * OW + x), neg ? ~row[x] : row[x]);
    };
    auto dom_empty = [&]
    {
        u64 any = 0;
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
            any |= mine[x];
        return any == 0;
    };
#pragma unroll
    for (u32 x = 0; x < OW; ++x)
        mine[x] = 0;
    const u32 nfree = mx[k_mc_free_params_n];
    const u32* fp = t.index + mx[k_mc_free_params];
    // lane v: v's place among the free parameters (the fail-first choice takes the first of the smallest)
    tmp[lane] = 32;
    __syncwarp();
    if (lane < nfree)
        tmp[fp[lane]] = lane;
    __syncwarp();
    const u32 rank = tmp[lane];
    {
        u64 a[OW];
        const bool empty = lane < nd && !domain<OW>(t, mx, view, steps[lane * 5], a);
        if (lane < nd)
#pragma unroll
            for (u32 x = 0; x < OW; ++x)
                D[steps[lane * 5] * OW + x] = a[x];
        if (__any_sync(k_full_mask, empty))
            return;
    }
    __syncwarp();
    // prebound -> free rows applied to the initial domains
    {
        const u32 n = mx[k_mc_fc_pre_n];
        const u64* pre = t.row + u64{mx[k_mc_fc_pre]} * 3;
        const u32* to = t.index + mx[k_mc_fc_pre_to];
        u32 touched = 0;
        for (u32 i = lane; i < n; i += 32)
        {
            const u64* rw = pre + u64{i} * 3;
            dom_and(to[i], table_row<OW>(t, view, rw[0], bind[rw[1]]), rw[2] != 0);
            touched |= 1u << to[i];
        }
        touched = __reduce_or_sync(k_full_mask, touched);
        __syncwarp();
        if (__any_sync(k_full_mask, ((touched >> lane) & 1) && dom_empty()))
            return;
    }
    const u64* edges = t.edge + u64{mx[k_mc_fc_out]} * 3;
    const u32* fck = t.index + mx[k_mc_fc_checks];
    const u32* cvb = t.index + mx[k_mc_check_vars_begin];
    const u32* cv = t.index + mx[k_mc_check_vars];
    const u32 checks0 = mx[k_mc_checks];
    // lane v: whether parameter v is effect-relevant, its edges and its checks
    const bool param = lane < total;
    const bool rel = param && t.index[mx[k_mc_relevant] + lane] != 0;
    const bool has_edges = mx[k_mc_fc_out_begin_n] == total + 1, has_checks = mx[k_mc_fc_checks_begin_n] == total + 1;
    const u32* fob = t.index + mx[k_mc_fc_out_begin];
    const u32* fcb = t.index + mx[k_mc_fc_checks_begin];
    const u32 eb = param && has_edges ? fob[lane] : 0, ee = param && has_edges ? fob[lane + 1] : 0;
    const u32 kb = param && has_checks ? fcb[lane] : 0, ke = param && has_checks ? fcb[lane + 1] : 0;
    u32 bound = 0;  // bit v: parameter v is bound
    bool witness = false;
    int wlev = -1;    // the level where the witness search started
    u32 pick = 0;     // lane l: level l's parameter
    u64 rem[OW];      // lane l: level l's candidates still to try
    u64 saved[k_deep_depth][OW];  // lane v: v's domain when each level was entered
    bool dirty = false;           // D differs from the current level's snapshot
    // search_fc's entry at level l: witness switch, fail-first choice (effect-relevant parameters first), snapshot
    auto enter = [&](int l)
    {
        const bool free = rank < nfree && !((bound >> lane) & 1);
        const bool any_relevant = __any_sync(k_full_mask, free && rel);
        if (!any_relevant && !witness)
        {
            witness = true;
            wlev = l;
        }
        u32 cnt = 0;
#pragma unroll
        for (u32 x = 0; x < OW; ++x)
        {
            saved[l][x] = mine[x];
            cnt += static_cast<u32>(__popcll(mine[x]));
        }
        const u32 key = __reduce_min_sync(k_full_mask, free && (!any_relevant || rel) ? cnt << 5 | rank : ~0u);
        const u32 best = static_cast<u32>(__ffs(static_cast<int>(__ballot_sync(k_full_mask, rank == (key & 31))))) - 1;
        if (lane == static_cast<u32>(l))
        {
            pick = best;
#pragma unroll
            for (u32 x = 0; x < OW; ++x)
                rem[x] = D[best * OW + x];
        }
        bound |= 1u << best;
        dirty = false;
        sp.enter(l, key >> 5, !witness, lane);
    };
    int level = 0;
    enter(0);
    while (level >= 0)
    {
        if (!sp.node())
            return;
        u32 o = 0;
        if (warp_next<OW>(rem, static_cast<u32>(level), lane, o) == OW)
        {
            bound &= ~(1u << __shfl_sync(k_full_mask, pick, level));
            --level;
            dirty = true;
            if (level < wlev)
            {
                witness = false;
                wlev = -1;
            }
            continue;
        }
        if (sp.skip(level, lane))
            continue;
        const u32 p = __shfl_sync(k_full_mask, pick, level);
        // the lanes' reads since the last sync (the checks' and edges' of `bind`, the emit's, enter()'s of D) come first
        __syncwarp();
        if (lane == 0)
            bind[p] = o;
        if (dirty)
        {
            // the domains the level was entered with
#pragma unroll
            for (u32 x = 0; x < OW; ++x)
                mine[x] = saved[level][x];
            dirty = false;
        }
        __syncwarp();
        bool ok = true;
        for (u32 k = __shfl_sync(k_full_mask, kb, p) + lane, ke_p = __shfl_sync(k_full_mask, ke, p); k < ke_p && ok; k += 32)
        {
            const u32 c = fck[k];
            bool all = true;
            for (u32 j = cvb[c]; j < cvb[c + 1]; ++j)
                all &= ((bound >> cv[j]) & 1) != 0;
            if (all)
                ok = check(t, checks0 + c, bind, s);
        }
        if (!__all_sync(k_full_mask, ok))
            continue;
        u32 touched = 0;
        for (u32 e = __shfl_sync(k_full_mask, eb, p) + lane, ee_p = __shfl_sync(k_full_mask, ee, p); e < ee_p; e += 32)
        {
            const u64* ed = edges + u64{e} * 3;
            const u32 to = static_cast<u32>(ed[0]);
            if ((bound >> to) & 1)
                continue;
            dom_and(to, table_row<OW>(t, view, ed[1], o), ed[2] != 0);
            touched |= 1u << to;
        }
        touched = __reduce_or_sync(k_full_mask, touched);
        if (touched)
        {
            dirty = true;
            __syncwarp();
            if (__any_sync(k_full_mask, ((touched >> lane) & 1) && dom_empty()))
                continue;
        }
        if (static_cast<u32>(level) + 1 == nfree)
        {
            if (!sp.leaf())
                return;
            emit();
            if (witness)
            {
                // the witness search found its binding: return to the level before the witness entry
                bound &= ~__reduce_or_sync(k_full_mask, static_cast<int>(lane) >= wlev && static_cast<int>(lane) <= level ? 1u << pick : 0u);
                level = wlev - 1;
                witness = false;
                wlev = -1;
                dirty = true;
            }
            continue;
        }
        ++level;
        enter(level);
    }
}

/// One (state, schema) segment of a deep matcher by a warp: alone with a budget (the first pass, h == k_none), or as
/// warp sp.i of sp.n (the second pass; h: the segment's place in the list). False when the budget ran out (the segment
/// goes to the second pass; a count is stored as 0, rows written so far are written again). Counts of the second pass
/// are the warps' sum. Rows in canonical order are gathered in the segment's rows (the label rows, or Labels::scratch:
/// ChunkGenerator::sorts() holds for deep schemas): at their places in the order found by one warp, at slots from the
/// list's counter by several (the last warp to finish sorts and writes them; unique keys, so the slots do not show).
/// Rows in the matcher's order (canonical order off) come from one warp (the second pass does not split them).
template<u32 OW, bool FC, bool Write, u32 WB>
__device__ __forceinline__ bool deep_segment(const GenArgs& a, u32 si, u32 li, DeepSplit& sp, DeepWarp<OW>& sw,
                                             const DeepWork& wk, u32 h)
{
    const TaskView& t = a.t;
    const u32 lane = threadIdx.x & 31;
    const u32 schema = a.set.schemas[si];
    const u32* sc = t.schema + u64{schema} * k_sc_count;
    const u32* mx = t.matcher + u64{sc[a.set.witness ? k_sc_pre0 : k_sc_pre1]} * k_mc_count;
    const Parents& p = a.p;
    const u64* row = p.data + u64{li} * p.stride;
    const u64* der = p.derived ? p.derived + u64{li} * p.derived_words : nullptr;
    const u64* view = a.v.data + u64{li} * a.v.words;
    const StateRef s{row, p.words, der, p.derived_words};
    const u64 seg = u64{li} * a.set.num_schemas + schema;
    auto run = [&](auto& emit)
    {
        // the warp's shared memory (sw) was read by all its lanes in its last segment
        __syncwarp();
        if constexpr (FC)
            warp_fc<OW>(t, mx, s, view, sw.dom, sw.bind, sw.tmp, emit, sp);
        else
            warp_fixed<OW>(t, mx, s, view, sw.bind, emit, sp);
    };
    if constexpr (!Write)
    {
        u32 n = 0;
        auto emit = [&] { ++n; };
        run(emit);
        if (lane == 0)
        {
            if (h == k_none)
                a.counts[seg] = sp.over ? 0 : n;
            else if (n)
                atomicAdd(a.counts + seg, n);
        }
        return !sp.over;
    }
    else
    {
        const u64 off = a.offsets[seg], cnt = a.offsets[seg + 1] - off;
        if (cnt == 0)
            return true;
        const RowWriter<WB> w(a, t, sc, schema, li, row);
        const u32 arity = w.arity;
        if (off >= a.lab.capacity)
        {
            // past the capacity: nothing is written, but widths and missing atoms count (in any order)
            if (!a.out.words_needed && !a.out.missing_flag)
                return true;
            auto emit = [&]
            {
                if (lane == 0)
                    w.put(a.lab.capacity, sw.bind);
            };
            run(emit);
            return !sp.over;
        }
        if (!a.set.canonical)
        {
            // the matcher's order
            u64 n = 0;
            auto emit = [&]
            {
                if (lane == 0 && n < cnt)
                    w.put(off + n, sw.bind);
                ++n;
            };
            run(emit);
            return !sp.over;
        }
        const Labels& lab = a.lab;
        const u32 LW = lab.label_width;
        const bool straddle = off + cnt > lab.capacity;
        u32* dst = nullptr;
        if (lab.binding && !straddle)
            dst = lab.binding + off * LW;
        else if (lab.scratch && lab.scratch_indexed && off + cnt <= lab.scratch_rows)
            dst = lab.scratch + off * LW;
        else if (lab.scratch && !lab.scratch_indexed && cnt <= lab.scratch_rows)
            dst = lab.scratch;  // the one segment that straddles the capacity
        if (!dst)
        {
            if (sp.i == 0 && lane == 0 && lab.error)
                atomicOr(lab.error, 1u);
            return true;
        }
        u32 found = 0;
        if (h == k_none)
        {
            auto emit = [&]
            {
                if (found < cnt && lane < arity)
                    dst[u64{found} * LW + lane] = sw.bind[lane];
                ++found;
            };
            run(emit);
            if (sp.over)
                return false;
        }
        else
        {
            auto emit = [&]
            {
                u32 at = 0;
                if (lane == 0)
                    at = atomicAdd(wk.slots + h, 1u);
                at = __shfl_sync(k_full_mask, at, 0);
                if (at < cnt && lane < arity)
                    dst[u64{at} * LW + lane] = sw.bind[lane];
            };
            run(emit);
            // the last warp of the segment to finish sorts and writes its rows (fences: the others' rows before)
            __threadfence();
            u32 last = 0;
            if (lane == 0)
                last = atomicAdd(wk.done + h, 1u) + 1 == sp.n ? 1u : 0u;
            if (!__shfl_sync(k_full_mask, last, 0))
                return true;
            __threadfence();
            found = *reinterpret_cast<volatile u32*>(wk.slots + h);
        }
        const u64 n = found < cnt ? found : cnt;
        __syncwarp();
        if (lane == 0)
            sort_rows(dst, n, LW, arity);
        __syncwarp();
        for (u64 j = lane; j < n; j += 32)
            w.put(off + j, dst + j * LW);
        return true;
    }
}

/// The first pass (Second = false): warps take segments (state g % rows, schema g / rows of the set: neighbouring warps
/// search the same matcher) from the queue with `budget` nodes each, and list the segments that ran out of it. The
/// second: the listed segments, each by deep_split() warps (an item per warp from the second queue). Both run over the
/// live rows (a device count: a chunk of a captured loop is sized for its capacity).
template<u32 OW, bool FC, bool Write, u32 WB, bool Second>
__global__ void __launch_bounds__(k_deep_block) k_gen_deep(GenArgs a, u32 budget)
{
    __shared__ DeepWarp<OW> sw[k_deep_block_warps];
    if constexpr (Write)
        if (a.lab.live && *a.lab.live == 0)
            return;
    const u32 rows = live_rows(a.p);
    const u64 segs = u64{rows} * a.set.count;
    if (segs == 0)
        return;
    const DeepWork wk(a.set.work, u64{a.p.rows} * a.set.count);
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if constexpr (!Second)
    {
        for (;;)
        {
            u32 g = 0;
            if (lane == 0)
                g = atomicAdd(wk.ctl, 1u);
            g = __shfl_sync(k_full_mask, g, 0);
            if (g >= segs)
                return;
            DeepSplit sp;
            sp.budget = budget;
            if (!deep_segment<OW, FC, Write, WB>(a, g / rows, g % rows, sp, sw[warp], wk, k_none) && lane == 0)
            {
                const u32 h = atomicAdd(wk.ctl + 1, 1u);
                wk.list[h] = g;
                wk.ticket[h] = 0;
                wk.slots[h] = 0;
                wk.done[h] = 0;
            }
        }
    }
    else
    {
        const u32 heavy = wk.ctl[1];  // (the first pass's launch)
        if (heavy == 0)
            return;
        const u32 S = Write && !a.set.canonical ? 1u : deep_split(heavy, gridDim.x * k_deep_block_warps);
        for (;;)
        {
            u32 x = 0;
            if (lane == 0)
                x = atomicAdd(wk.ctl + 2, 1u);
            x = __shfl_sync(k_full_mask, x, 0);
            if (x >= heavy * S)
                return;
            const u32 h = x / S, g = wk.list[h];
            DeepSplit sp;
            sp.n = S;
            sp.i = x % S;
            sp.ticket = wk.ticket + h;
            (void)deep_segment<OW, FC, Write, WB>(a, g / rows, g % rows, sp, sw[warp], wk, h);
        }
    }
}

/// Blocks of the first pass's kernel the device holds at once (cached per device; the second pass's is alike).
template<u32 OW, bool FC, bool Write, u32 WB>
u32 deep_resident_blocks()
{
    static std::atomic<int> cache[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64)
        return 1;
    int n = cache[dev].load(std::memory_order_relaxed);
    if (n == 0)
    {
        int per_sm = 0, sms = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_gen_deep<OW, FC, Write, WB, false>, k_deep_block, 0) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            return 1;
        n = per_sm > 0 && sms > 0 ? per_sm * sms : 1;
        cache[dev].store(n, std::memory_order_relaxed);
    }
    return static_cast<u32>(n);
}

template<u32 OW, bool FC, bool Write, u32 WB>
cudaError_t launch_deep(const GenArgs& a, cudaStream_t s)
{
    const u64 segs = u64{a.p.rows} * a.set.count;  // (the capacity: the passes follow the live rows)
    if (segs == 0)
        return cudaSuccess;
    if (!a.set.work || segs * k_deep_split > 0xFFFFFFFFull)
        return cudaErrorInvalidValue;
    if (const cudaError_t e = cudaMemsetAsync(a.set.work, 0, k_deep_ctl * sizeof(u32), s); e != cudaSuccess)
        return e;
    const u64 resident = deep_resident_blocks<OW, FC, Write, WB>();
    const u64 warps = resident * k_deep_block_warps;
    const u64 scaled = segs >= warps ? k_deep_budget : k_deep_budget * segs / warps;
    const u32 budget = a.set.budget ? a.set.budget : static_cast<u32>(scaled > k_deep_budget_min ? scaled : k_deep_budget_min);
    const u64 first = (segs + k_deep_block_warps - 1) / k_deep_block_warps;
    k_gen_deep<OW, FC, Write, WB, false><<<static_cast<unsigned>(first < resident ? first : resident), k_deep_block, 0, s>>>(a, budget);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return e;
    const u64 second = (segs * k_deep_split + k_deep_block_warps - 1) / k_deep_block_warps;
    k_gen_deep<OW, FC, Write, WB, true><<<static_cast<unsigned>(second < resident ? second : resident), k_deep_block, 0, s>>>(a, 0);
    return cudaGetLastError();
}

/// The deep kernels (SchemaSet::deep): matchers of up to k_deep_depth parameters, single instance, no conditional
/// effects, object bitsets of at most k_deep_ow words (lifted_deep*.cu).
template<bool Write, u32 WB>
cudaError_t dispatch_deep(const GenArgs& a, cudaStream_t s)
{
    if (!a.set.deep)
        return cudaErrorInvalidValue;
    switch ((a.t.ow - 1) * 2 + (a.set.fc ? 1 : 0))
    {
        case 0: return launch_deep<1, false, Write, WB>(a, s);
        case 1: return launch_deep<1, true, Write, WB>(a, s);
        case 2: return launch_deep<2, false, Write, WB>(a, s);
        case 3: return launch_deep<2, true, Write, WB>(a, s);
        case 4: return launch_deep<3, false, Write, WB>(a, s);
        case 5: return launch_deep<3, true, Write, WB>(a, s);
        case 6: return launch_deep<4, false, Write, WB>(a, s);
        case 7: return launch_deep<4, true, Write, WB>(a, s);
        default: return cudaErrorInvalidValue;
    }
}
static_assert(k_deep_ow == 4, "dispatch_deep: one case pair per OW");
static_assert(k_deep_depth == 32, "the deep kernels: one lane per parameter");

}  // namespace
}  // namespace mymyr::cuda::lifted
