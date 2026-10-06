#pragma once
// Internal: compiled conjunctive matchers over relation views, shared by the relaxed reachability fixpoint
// (reachability/relaxed_reachability.hpp), its conjunctive queries, and the lifted relaxation of the heuristics.
//
// The model is the successor engine's matcher (successor/detail/engine.hpp), generalized:
//   - a conjunction of literals over variables 0..V-1: relation literals (fluent or derived predicates, read from a
//     View), static literals (both polarities), equalities and disequalities between terms, and custom checks;
//   - per variable a static domain (types, static unary literals, projections of static k-ary literals, constants);
//   - binding steps in a greedy connected order (most constraints to bound variables first, relevant variables before
//     witness-only ones among equals), each ANDing the object bitsets of unary rows and of binary rows selected by an
//     earlier variable's value; literals of arity >= 3, with repeated variables or with every variable bound earlier
//     are checked by key when their last variable is bound;
//   - prebound variables (the anchor of a semi-naive join: the variables of a literal unified with a new atom) are
//     tested against their constraints once, before the first step;
//   - relevance: once only witness-only variables remain, one completion suffices (witness pruning), and a prefix check
//     (the head is already derived) can cut the search where every head variable is bound;
//   - every relation literal names a slot, and the plan says per slot which of two views it reads (Old or New): the
//     duplicate-free delta join (anchor on the first new literal: literals before it read Old, after it New).
//
// Views: per fluent/derived predicate of arity 1 or 2 the rows of the task's ViewLayout (a unary row; forward and
// backward rows over objects), stored sparsely (untouched rows share one zero row), plus a bitset over canonical ids
// for membership tests of any arity.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/task/atom_index.hpp"
#include "mymyr/task/plan.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::reach
{
inline constexpr u32 k_none = ~u32{0};

enum class LitType : u8
{
    Rel,     // a fluent or derived predicate, read from a View
    Static,  // a static predicate
    Eq,      // terms[0] == terms[1] (positive) or != (negative)
    Custom,  // Ctx::custom(id, bind) once every variable of `terms` is bound
};

struct Lit
{
    LitType type = LitType::Rel;
    bool positive = true;
    u32 pred = 0;             // Rel, Static
    u32 slot = 0;             // Rel: relation-literal slot (selects the view); Custom: id
    std::vector<i32> terms;   // >= 0: variable, < 0: object -(o + 1)
};

[[nodiscard]] inline constexpr bool is_var(i32 t) { return t >= 0; }
[[nodiscard]] inline constexpr u32 term_obj(i32 t) { return static_cast<u32>(-(t + 1)); }
[[nodiscard]] inline constexpr i32 obj_term(u32 o) { return -static_cast<i32>(o) - 1; }

// Word ranges: the nonzero words of an object bitset lie in [lo, hi), packed as lo | hi << 32 (0: empty). The executor
// intersects the ranges of a step's rows before touching any word, so sparse rows (a location's few neighbours among
// thousands of objects) cost a few words instead of OW.
[[nodiscard]] inline constexpr u64 pack_range(u32 lo, u32 hi) { return static_cast<u64>(lo) | (static_cast<u64>(hi) << 32); }
[[nodiscard]] inline constexpr u32 range_lo(u64 g) { return static_cast<u32>(g); }
[[nodiscard]] inline constexpr u32 range_hi(u64 g) { return static_cast<u32>(g >> 32); }
/// Up to this many words per object set (256 objects) the executor does not intersect row ranges: looking them up costs
/// more than the few words they would save.
inline constexpr u32 k_range_min_ow = 4;
/// The word range of a bitset of `ow` words (0 if it is empty).
[[nodiscard]] inline u64 word_range(const u64* s, u32 ow)
{
    u32 lo = 0, hi = ow;
    while (lo < hi && !s[lo])
        ++lo;
    while (hi > lo && !s[hi - 1])
        --hi;
    return lo < hi ? pack_range(lo, hi) : 0;
}

/// Row tables of one view, sparse: untouched rows share a zero row. Each row keeps its word range (a superset after
/// reset()).
class RowStore
{
public:
    void init(u32 rows, u32 ow)
    {
        m_ow = ow;
        m_zero.assign(ow, 0);
        m_ptr.assign(rows, m_zero.data());
        m_rng.assign(rows, 0);
        m_touched.clear();
        m_block = 0;
        m_fill = 0;
    }
    [[nodiscard]] const u64* row(u32 r) const noexcept { return m_ptr[r]; }
    [[nodiscard]] u64 range(u32 r) const noexcept { return m_rng[r]; }
    void set(u32 r, u32 bit)
    {
        u64* p = m_ptr[r];
        if (p == m_zero.data())
            p = alloc(r);
        const u32 w = bit >> 6;
        p[w] |= u64{1} << (bit & 63);
        const u64 g = m_rng[r];
        m_rng[r] = g == 0 ? pack_range(w, w + 1) : pack_range(std::min(range_lo(g), w), std::max(range_hi(g), w + 1));
    }
    void reset(u32 r, u32 bit)
    {
        u64* p = m_ptr[r];
        if (p != m_zero.data())
            p[bit >> 6] &= ~(u64{1} << (bit & 63));
    }
    /// Every row back to zero (touched rows only).
    void clear()
    {
        for (u32 r : m_touched)
        {
            m_ptr[r] = m_zero.data();
            m_rng[r] = 0;
        }
        m_touched.clear();
        m_block = 0;
        m_fill = 0;
    }
    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 b = m_ptr.capacity() * sizeof(u64*) + m_rng.capacity() * sizeof(u64);
        for (usize n : m_block_size)
            b += n * 8;
        return b;
    }

private:
    u64* alloc(u32 r)
    {
        // blocks grow geometrically (256 words up to 64K words) and are not zeroed (every row is, when handed out):
        // small tasks and short fixpoints touch little memory
        while (m_block < m_blocks.size() && m_fill + m_ow > m_block_size[m_block])
            ++m_block, m_fill = 0;
        if (m_block == m_blocks.size())
        {
            const usize n = std::max<usize>(m_ow, std::min<usize>(usize{1} << 16, usize{256} << std::min<usize>(m_block, 8)));
            m_blocks.emplace_back(new u64[n]);
            m_block_size.push_back(n);
        }
        u64* p = m_blocks[m_block].get() + m_fill;
        std::memset(p, 0, m_ow * sizeof(u64));
        m_fill += m_ow;
        m_ptr[r] = p;
        m_touched.push_back(r);
        return p;
    }

    u32 m_ow = 1;
    std::vector<u64> m_zero;
    std::vector<u64*> m_ptr;
    std::vector<u64> m_rng;  // per row: its word range
    std::vector<u32> m_touched;
    std::vector<std::unique_ptr<u64[]>> m_blocks;
    std::vector<usize> m_block_size;
    usize m_block = 0, m_fill = 0;
};

/// A set of fluent/derived atoms: view rows plus a bitset over canonical ids.
struct View
{
    RowStore rows;
    std::vector<u64> bits;  // canonical ids
    std::vector<u64> touched;  // canonical ids set since the last clear (for clear())

    void init(const plan::Compiled& c, u64 total)
    {
        rows.init(c.view.rows, c.ow);
        bits.assign(bits::words_for(std::max<u64>(total, 1)), 0);
        touched.clear();
    }
    [[nodiscard]] bool test(u64 key) const noexcept { return (key >> 6) < bits.size() && ((bits[key >> 6] >> (key & 63)) & 1); }
    /// Adds an atom (pred, args) with canonical id `key`.
    void add(const plan::Compiled& c, u64 key, u32 pred, u32 arity, const u32* args)
    {
        bits[key >> 6] |= u64{1} << (key & 63);
        touched.push_back(key);
        if (arity == 1)
            rows.set(c.view.unary_row[pred], args[0]);
        else if (arity == 2)
        {
            rows.set(c.view.fwd_row[pred] + args[0], args[1]);
            rows.set(c.view.bwd_row[pred] + args[1], args[0]);
        }
    }
    void clear()
    {
        for (u64 k : touched)
            bits[k >> 6] = 0;
        touched.clear();
        rows.clear();
    }
};

/// Static tables of a task shared by all plans: unary sets, row tables and projections, built on first use
/// (thread-safe; the returned pointers stay valid for the lifetime of the object).
class StaticTables
{
public:
    explicit StaticTables(const Task& task);

    [[nodiscard]] const Task& task() const noexcept { return *m_task; }
    [[nodiscard]] const plan::Compiled& compiled() const noexcept { return *m_c; }
    [[nodiscard]] u32 n() const noexcept { return m_n; }
    [[nodiscard]] u32 ow() const noexcept { return m_ow; }
    /// Membership of a static atom.
    [[nodiscard]] bool contains(u32 pred, const u32* args) const;
    /// Objects occurring at position i of static pred's atoms that agree with the constants of `terms` (OW words).
    const u64* projection(u32 pred, u32 i, const std::vector<i32>& terms);
    /// n * OW row table of static pred: row(a[from]) holds a[to] over the atoms agreeing with the constants of
    /// `terms` (the positions other than from and to). The table is followed by one zero word and the word ranges of
    /// its n rows (row_ranges()).
    const u64* rows(u32 pred, u32 from, u32 to, const std::vector<i32>& terms);
    /// The word ranges of a table returned by rows(), per source object.
    [[nodiscard]] const u64* row_ranges(const u64* table) const noexcept { return table + static_cast<usize>(m_n) * m_ow + 1; }
    /// Objects compatible with a declared type list (every object for an empty list).
    const u64* objects_of_types(const std::vector<u32>& types);
    /// Whether the predicate is PDDL equality (evaluated on object identity where asked).
    [[nodiscard]] bool is_equality(u32 pred) const noexcept { return pred == m_eq; }

private:
    [[nodiscard]] bool agrees(const u32* atom, const std::vector<i32>& terms, u32 skip1, u32 skip2) const;

    const Task* m_task;
    const plan::Compiled* m_c;
    u32 m_n = 0, m_ow = 1, m_eq = k_none;
    std::vector<std::vector<u32>> m_atoms;  // per static predicate: flat args
    std::mutex m_mutex;
    std::map<std::vector<i64>, std::unique_ptr<std::vector<u64>>> m_cache;
};

/// A row constraint: the bits of a row are ANDed into a variable's candidates.
struct RowRef
{
    enum class Kind : u8
    {
        Static,  // base + value(src) * OW (src == k_none: base)
        Rel,     // view row first_row + value(src) (src == k_none: first_row)
        Same,    // the singleton {value(src)}
    };
    Kind kind = Kind::Static;
    bool neg = false;
    u32 slot = 0;       // Rel: relation-literal slot
    u32 first_row = 0;  // Rel
    u32 src = k_none;
    const u64* base = nullptr;  // Static
    u32 var = 0;                // the constrained variable
    const u64* rng = nullptr;   // Static with a source: per source value the word range of its row
};

struct KeyVar
{
    u32 var;
    const u64* rs;
};

/// A literal checked by key once its variables are bound.
struct CheckLit
{
    LitType type = LitType::Rel;
    bool positive = true;
    u32 slot = 0;   // Rel: slot; Custom: id
    u32 pred = 0;
    u64 base = 0;   // key contribution of the constants (Rel: canonical offset included)
    bool never = false;  // a constant lies outside the key space: the atom cannot exist
    u32 var_begin = 0, var_count = 0;  // into Plan::key_vars
    i32 eq_a = 0, eq_b = 0;  // Eq terms
};

struct Step
{
    u32 var = 0;
    u32 row_begin = 0, row_end = 0;
    u32 check_begin = 0, check_end = 0;
};

/// Plans with at least this many steps run arc consistency before the search (Plan::ac), as do plans that enumerate a
/// relevant variable unconnected to the variables before it but constrained by a witness variable after it.
inline constexpr u32 k_ac_min_steps = 6;
/// Arc consistency revises an arc only while its source domain has at most this many values.
inline constexpr u32 k_ac_max_support = 64;

struct Plan
{
    const plan::StaticRelation* statics = nullptr;  // the task's static relations (by predicate)
    u32 num_vars = 0;
    u32 ow = 1;
    bool never = false;
    std::vector<u64> dom0;                 // num_vars * OW
    std::vector<u64> dom_rng;              // per variable: the word range of dom0 (still valid when dom0 is narrowed)
    std::vector<u32> prebound;             // variables bound before step 0
    std::vector<RowRef> pre_rows;          // tests of prebound variables (var = the tested variable)
    std::vector<u32> pre_checks;           // into checks: literals whose variables are all prebound
    std::vector<RowRef> rows;              // per step
    std::vector<u32> step_checks;          // into checks
    std::vector<CheckLit> checks;
    std::vector<KeyVar> key_vars;
    std::vector<Step> steps;
    u32 first_exist = 0;    // steps from here on are witness-only
    u32 head_depth = k_none;  // Ctx::head_ok runs when this many steps are bound (k_none: never)
    std::vector<u8> slot_mode;  // per relation-literal slot: 0 = Old, 1 = New
    // Arc consistency before the search (plans with many variables, where a failing search backtracks through
    // exponentially many partial bindings): the domains of all variables are narrowed by the unary rows and by every
    // binary row constraint in both directions until nothing changes; an empty domain ends the run.
    bool ac = false;
    std::vector<RowRef> ac_unary;  // rows without a source variable (var = the constrained variable)
    std::vector<RowRef> ac_arcs;   // binary rows: src selects the row, var is constrained
    // Forward check at the start of a run: the variables of later steps whose rows depend on prebound variables only;
    // an empty candidate set ends the run before any enumeration.
    struct FcGroup
    {
        u32 var = 0, begin = 0, end = 0;  // into fc_rows
    };
    std::vector<FcGroup> fc;
    std::vector<RowRef> fc_rows;
};

struct CompileSpec
{
    u32 num_vars = 0;
    std::vector<u8> prebound;  // per variable
    std::vector<u8> relevant;  // per variable (enumerate exhaustively); empty: all relevant
    std::vector<u8> head;      // per variable: head_ok needs it bound; empty: no head check
    std::vector<u8> slot_mode; // per slot
    /// Declared types per variable (restricts a variable that no positive literal binds); may be empty.
    std::vector<std::vector<u32>> var_types;
    bool equality_by_identity = true;  // PDDL "=" on object identity, as in mimir's relaxed reachability
};

/// Compiles `lits` (every Rel literal's slot < spec.slot_mode.size()).
[[nodiscard]] Plan compile(StaticTables& st, const std::vector<Lit>& lits, const CompileSpec& spec);

/// The executor. Ctx provides:
///   const RowStore& rows(u32 mode) const;  const View& view(u32 mode) const;
///   bool custom(u32 id, const u32* bind);  bool head_ok(const u32* bind);  bool emit(const u32* bind);  // false stops
class Executor
{
public:
    void reserve(u32 vars, u32 steps, u32 ow)
    {
        if (m_cand.size() < static_cast<usize>(steps + 1) * ow)
            m_cand.resize(static_cast<usize>(steps + 1) * ow);
        if (m_bind.size() < vars)
            m_bind.resize(vars);
    }
    [[nodiscard]] u32* bind() noexcept { return m_bind.data(); }

    /// Runs the plan with the prebound variables already in bind(). Returns false iff emit() asked to stop.
    template<class Ctx>
    bool run(const Plan& p, Ctx& ctx)
    {
        if (p.never)
            return true;
        reserve(p.num_vars, static_cast<u32>(p.steps.size()), p.ow);
        m_plan = &p;
        m_stop = false;
        u32* b = m_bind.data();
        const u32 OW = p.ow;
        for (u32 v : p.prebound)
            if (!bits::test(p.dom0.data() + static_cast<usize>(v) * OW, OW, b[v]))
                return true;
        for (const RowRef& r : p.pre_rows)
        {
            bool t;
            if (r.kind == RowRef::Kind::Same)
                t = b[r.src] == b[r.var];
            else
                t = bits::test(resolve(r, ctx), OW, b[r.var]);
            if (t == r.neg)
                return true;
        }
        for (u32 c : p.pre_checks)
            if (!holds(p.checks[c], ctx))
                return true;
        for (const Plan::FcGroup& g : p.ac ? std::span<const Plan::FcGroup>{} : std::span<const Plan::FcGroup>(p.fc))
        {
            const u64* d0 = p.dom0.data() + static_cast<usize>(g.var) * OW;
            u32 lo = range_lo(p.dom_rng[g.var]), hi = range_hi(p.dom_rng[g.var]);
            for (u32 i = g.begin; OW > k_range_min_ow && i < g.end && lo < hi; ++i)
                if (!p.fc_rows[i].neg)
                {
                    const u64 rg = range_of(p.fc_rows[i], ctx);
                    lo = std::max(lo, range_lo(rg));
                    hi = std::min(hi, range_hi(rg));
                }
            u64 any = 0;
            for (u32 w = lo; w < hi && !any; ++w)
            {
                u64 acc = d0[w];
                for (u32 i = g.begin; i < g.end && acc; ++i)
                {
                    const RowRef& r = p.fc_rows[i];
                    if (r.kind == RowRef::Kind::Same)
                    {
                        const u32 o = b[r.src];
                        const u64 m = (o >> 6) == w ? u64{1} << (o & 63) : 0;
                        acc &= r.neg ? ~m : m;
                    }
                    else
                        acc &= resolve(r, ctx)[w] ^ (r.neg ? ~u64{0} : 0);
                }
                any |= acc;
            }
            if (!any)
                return true;
        }
        m_dom = p.dom0.data();
        m_rng = p.dom_rng.data();
        if (p.ac && !arc_consistency(ctx))
            return true;
        search(0, false, ctx);
        return !m_stop;
    }

    /// Key of a Rel/Static check literal under the current binding.
    [[nodiscard]] u64 key(const CheckLit& c) const noexcept
    {
        u64 k = c.base;
        const KeyVar* kv = m_plan->key_vars.data() + c.var_begin;
        for (u32 i = 0; i < c.var_count; ++i)
            k += kv[i].rs[m_bind[kv[i].var]];
        return k;
    }

private:
    template<class Ctx>
    const u64* resolve(const RowRef& r, const Ctx& ctx) const
    {
        switch (r.kind)
        {
            case RowRef::Kind::Static:
                return r.src == k_none ? r.base : r.base + static_cast<usize>(m_bind[r.src]) * m_plan->ow;
            case RowRef::Kind::Rel:
                return ctx.rows(m_plan->slot_mode[r.slot]).row(r.first_row + (r.src == k_none ? 0 : m_bind[r.src]));
            case RowRef::Kind::Same: return nullptr;
        }
        return nullptr;
    }

    /// The word range of a positive row under the current binding (Same: the word of the source value).
    template<class Ctx>
    u64 range_of(const RowRef& r, const Ctx& ctx) const
    {
        switch (r.kind)
        {
            case RowRef::Kind::Static:
                return r.rng ? r.rng[m_bind[r.src]] : pack_range(0, m_plan->ow);
            case RowRef::Kind::Rel:
                return ctx.rows(m_plan->slot_mode[r.slot]).range(r.first_row + (r.src == k_none ? 0 : m_bind[r.src]));
            case RowRef::Kind::Same:
            {
                const u32 w = m_bind[r.src] >> 6;
                return pack_range(w, w + 1);
            }
        }
        return pack_range(0, m_plan->ow);
    }

    template<class Ctx>
    bool holds(const CheckLit& c, Ctx& ctx)
    {
        switch (c.type)
        {
            case LitType::Rel:
            {
                if (c.never)
                    return !c.positive;
                return ctx.view(m_plan->slot_mode[c.slot]).test(key(c)) == c.positive;
            }
            case LitType::Static:
            {
                if (c.never)
                    return !c.positive;
                return m_plan->statics[c.pred].contains(key(c)) == c.positive;
            }
            case LitType::Eq:
            {
                const u32 a = is_var(c.eq_a) ? m_bind[c.eq_a] : term_obj(c.eq_a);
                const u32 b = is_var(c.eq_b) ? m_bind[c.eq_b] : term_obj(c.eq_b);
                return (a == b) == c.positive;
            }
            case LitType::Custom: return ctx.custom(c.slot, m_bind.data());
        }
        return false;
    }

    /// Narrows the domains of the unbound variables (into m_ac) towards arc consistency. False if one becomes empty.
    /// An arc is revised only while its source domain is small (k_ac_max_support values): the support of a large domain
    /// is nearly everything, and computing it costs a row per value.
    /// Each domain is kept within its word range (m_acr): words outside it are stale and never read.
    template<class Ctx>
    bool arc_consistency(const Ctx& ctx)
    {
        const Plan& p = *m_plan;
        const u32 OW = p.ow, V = p.num_vars;
        m_ac.resize(p.dom0.size());
        m_acr.assign(p.dom_rng.begin(), p.dom_rng.end());
        m_count.assign(V, 0);
        auto D = [&](u32 v) { return m_ac.data() + static_cast<usize>(v) * OW; };
        auto count = [&](u32 v)
        {
            const u64* d = D(v);
            u32 c = 0;
            for (u32 w = range_lo(m_acr[v]); w < range_hi(m_acr[v]); ++w)
                c += static_cast<u32>(std::popcount(d[w]));
            return c;
        };
        auto tighten = [&](u32 v)
        {
            const u64* d = D(v);
            u32 lo = range_lo(m_acr[v]), hi = range_hi(m_acr[v]);
            while (lo < hi && !d[lo])
                ++lo;
            while (hi > lo && !d[hi - 1])
                --hi;
            m_acr[v] = lo < hi ? pack_range(lo, hi) : 0;
        };
        for (u32 v = 0; v < V; ++v)
        {
            const u64* d0 = p.dom0.data() + static_cast<usize>(v) * OW;
            std::copy(d0 + range_lo(m_acr[v]), d0 + range_hi(m_acr[v]), D(v) + range_lo(m_acr[v]));
        }
        for (u32 v : p.prebound)
        {
            const u32 w = m_bind[v] >> 6;
            D(v)[w] = u64{1} << (m_bind[v] & 63);
            m_acr[v] = pack_range(w, w + 1);
        }
        for (const RowRef& r : p.ac_unary)
        {
            const u64* row = resolve(r, ctx);
            const u64 flip = r.neg ? ~u64{0} : 0;
            u64* d = D(r.var);
            u32 lo = range_lo(m_acr[r.var]), hi = range_hi(m_acr[r.var]);
            if (!r.neg)
            {
                const u64 rg = range_of(r, ctx);
                lo = std::max(lo, range_lo(rg));
                hi = std::min(hi, range_hi(rg));
            }
            u64 any = 0;
            for (u32 w = lo; w < hi; ++w)
                any |= (d[w] &= row[w] ^ flip);
            if (!any)
                return false;
            m_acr[r.var] = pack_range(lo, hi);
            tighten(r.var);
        }
        for (u32 v = 0; v < V; ++v)
            m_count[v] = count(v);
        m_support.resize(OW);
        for (u32 pass = 0; pass < 64; ++pass)
        {
            bool changed = false;
            for (const RowRef& r : p.ac_arcs)
            {
                const u32 cs = m_count[r.src];
                if (m_count[r.var] <= 1 && cs <= 1)
                    continue;
                const u64* ds = D(r.src);
                u64* dv = D(r.var);
                u64* sup = m_support.data();
                const u32 slo = range_lo(m_acr[r.src]), shi = range_hi(m_acr[r.src]);
                u32 lo = range_lo(m_acr[r.var]), hi = range_hi(m_acr[r.var]);
                if (r.kind == RowRef::Kind::Same)
                {
                    if (!r.neg)
                    {
                        lo = std::max(lo, slo);
                        hi = std::min(hi, shi);
                        std::copy(ds + std::min(lo, hi), ds + hi, sup + std::min(lo, hi));
                    }
                    else if (cs == 1)
                        for (u32 w = lo; w < hi; ++w)
                            sup[w] = w >= slo && w < shi ? ~ds[w] : ~u64{0};
                    else
                        continue;
                }
                else
                {
                    if (cs > k_ac_max_support)
                        continue;
                    const u64 flip = r.neg ? ~u64{0} : 0;
                    if (!r.neg)
                    {
                        // the support lies within the union of the rows' ranges
                        u32 ulo = hi, uhi = lo;
                        for (u32 w = slo; w < shi; ++w)
                            for (u64 word = ds[w]; word; word &= word - 1)
                            {
                                const u64 rg = row_range_of(r, w * 64 + static_cast<u32>(bits::ctz64(word)), ctx);
                                if (rg)
                                {
                                    ulo = std::min(ulo, range_lo(rg));
                                    uhi = std::max(uhi, range_hi(rg));
                                }
                            }
                        lo = std::max(lo, ulo);
                        hi = std::min(hi, uhi);
                    }
                    if (lo < hi)
                        std::fill(sup + lo, sup + hi, u64{0});
                    bool full = false;  // a negative row's support fills up quickly: stop there
                    for (u32 w = slo; w < shi && !full && lo < hi; ++w)
                    {
                        u64 word = ds[w];
                        while (word)
                        {
                            const u32 o = w * 64 + static_cast<u32>(bits::ctz64(word));
                            word &= word - 1;
                            const u64* row = row_of(r, o, ctx);
                            u64 all = ~u64{0};
                            for (u32 x = lo; x < hi; ++x)
                                all &= (sup[x] |= row[x] ^ flip);
                            if (r.neg && all == ~u64{0})
                            {
                                full = true;
                                break;
                            }
                        }
                    }
                    if (full)
                        continue;
                }
                for (u32 w = lo; w < hi; ++w)
                    dv[w] &= sup[w];
                m_acr[r.var] = lo < hi ? pack_range(lo, hi) : 0;
                tighten(r.var);
                const u32 c = count(r.var);
                if (c != m_count[r.var])
                {
                    changed = true;
                    m_count[r.var] = c;
                    if (c == 0)
                        return false;
                }
            }
            if (!changed)
                break;
        }
        m_dom = m_ac.data();
        m_rng = m_acr.data();
        return true;
    }

    /// The row of a binary RowRef for source value o (Static or Rel).
    template<class Ctx>
    const u64* row_of(const RowRef& r, u32 o, const Ctx& ctx) const
    {
        if (r.kind == RowRef::Kind::Static)
            return r.base + static_cast<usize>(o) * m_plan->ow;
        return ctx.rows(m_plan->slot_mode[r.slot]).row(r.first_row + o);
    }
    /// The word range of that row.
    template<class Ctx>
    u64 row_range_of(const RowRef& r, u32 o, const Ctx& ctx) const
    {
        if (r.kind == RowRef::Kind::Static)
            return r.rng ? r.rng[o] : pack_range(0, m_plan->ow);
        return ctx.rows(m_plan->slot_mode[r.slot]).range(r.first_row + o);
    }

    // Returns true if a full binding was found below step d (used in witness mode).
    template<class Ctx>
    bool search(u32 d, bool witness, Ctx& ctx)
    {
        const Plan& p = *m_plan;
        if (d == p.head_depth && !witness && !ctx.head_ok(m_bind.data()))
            return false;
        if (d == p.steps.size())
        {
            if (!witness && !ctx.emit(m_bind.data()))
                m_stop = true;
            return true;
        }
        if (d == p.first_exist && !witness)
        {
            const bool found = search(d, true, ctx);
            if (found && !ctx.emit(m_bind.data()))
                m_stop = true;
            return found;
        }
        const u32 OW = p.ow;
        const Step& st = p.steps[d];
        u64* cd = m_cand.data() + static_cast<usize>(d) * OW;
        const u64* d0 = m_dom + static_cast<usize>(st.var) * OW;
        // the candidates lie within the intersection of the domain's and the positive rows' word ranges; cd is only
        // defined there
        u32 lo = range_lo(m_rng[st.var]), hi = range_hi(m_rng[st.var]);
        for (u32 ri = st.row_begin; OW > k_range_min_ow && ri < st.row_end && lo < hi; ++ri)
            if (!p.rows[ri].neg)
            {
                const u64 rg = range_of(p.rows[ri], ctx);
                lo = std::max(lo, range_lo(rg));
                hi = std::min(hi, range_hi(rg));
            }
        if (lo >= hi)
            return false;
        MYMYR_NOVECTOR
        for (u32 w = lo; w < hi; ++w)
            cd[w] = d0[w];
        for (u32 ri = st.row_begin; ri < st.row_end; ++ri)
        {
            const RowRef& r = p.rows[ri];
            u64 any = 0;
            if (r.kind == RowRef::Kind::Same)
            {
                const u32 o = m_bind[r.src];
                if (r.neg)
                {
                    if ((o >> 6) >= lo && (o >> 6) < hi)
                        cd[o >> 6] &= ~(u64{1} << (o & 63));
                    MYMYR_NOVECTOR
                    for (u32 w = lo; w < hi; ++w)
                        any |= cd[w];
                }
                else
                {
                    const u32 w0 = o >> 6;
                    const u64 keep = w0 >= lo && w0 < hi ? cd[w0] & (u64{1} << (o & 63)) : 0;
                    MYMYR_NOVECTOR
                    for (u32 w = lo; w < hi; ++w)
                        cd[w] = 0;
                    if (keep)
                        cd[w0] = keep;
                    any = keep;
                }
            }
            else
            {
                const u64* row = resolve(r, ctx);
                const u64 flip = r.neg ? ~u64{0} : 0;
                MYMYR_NOVECTOR
                for (u32 w = lo; w < hi; ++w)
                    any |= (cd[w] &= row[w] ^ flip);
            }
            if (!any)
                return false;
        }
        u32* b = m_bind.data();
        const u32 var = st.var;
        MYMYR_NOVECTOR
        for (u32 w = lo; w < hi; ++w)
        {
            u64 word = cd[w];
            while (word)
            {
                const u32 o = w * 64 + static_cast<u32>(bits::ctz64(word));
                word &= word - 1;
                b[var] = o;
                bool ok = true;
                for (u32 ci = st.check_begin; ci < st.check_end; ++ci)
                    if (!holds(p.checks[p.step_checks[ci]], ctx))
                    {
                        ok = false;
                        break;
                    }
                if (!ok)
                    continue;
                if (search(d + 1, witness, ctx) && witness)
                    return true;
                if (m_stop)
                    return false;
            }
        }
        return false;
    }

    const Plan* m_plan = nullptr;
    const u64* m_dom = nullptr;  // the domains the search starts from: dom0, or m_ac after arc consistency
    const u64* m_rng = nullptr;  // their word ranges: dom_rng, or m_acr
    std::vector<u64> m_ac, m_acr, m_support;
    std::vector<u32> m_count;
    std::vector<u64> m_cand;
    std::vector<u32> m_bind;
    bool m_stop = false;
};
}  // namespace mymyr::reach
