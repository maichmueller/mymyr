#pragma once
// The per-thread matching engine: the per-state fluent view, the binding array, and the executor
// of compiled matchers (fixed order with backward row checks, or fail-first forward checking; witness pruning),
// built over shared immutable plans:
//   - plans (task/plan.hpp) are shared by all threads; an ExecMatcher holds one thread's resolved table pointers and
//     scratch, so the hot loops read raw pointers only;
//   - the view is built from the state's set bits and only the rows touched by the previous state are cleared;
//   - templates, so the emit callback is inlined: no virtual or indirect calls on the hot path.
// Emit callbacks return bool (false = stop). With Stop = false the result is ignored and the checks compile away.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/task/atom_index.hpp"
#include "mymyr/task/plan.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <span>
#include <vector>

namespace mymyr
{
class Task;
}

namespace mymyr::detail
{
struct ExecUnary
{
    const u64* ptr;
    u64 flip;  // ~0 for a negated literal
};
struct ExecRow
{
    const u64* base;
    u32 src;
    u64 flip;
};
struct ExecEdge
{
    u32 to;
    const u64* base;
    u64 flip;
};

/// One binding step of the fixed-order search with every pointer resolved (the hot loops read no vectors).
struct ExecStep
{
    u32 param;
    const ExecRow* rb;  // row constraints to earlier or prebound parameters
    const ExecRow* re;
    const u32* cb;  // literal checks (indices into plan checks)
    const u32* ce;
    const u32* ncb;  // numeric checks (indices into plan nchecks)
    const u32* nce;
    const ExecUnary* ub;  // per-state unary constraints of param
    const ExecUnary* ue;
    const u64* dom0;  // static domain of param
    u64* dom1;        // domain of param in the current state
    u64* cand;        // candidates at this step
};

struct ExecMatcher
{
    const plan::Matcher* plan = nullptr;
    std::vector<ExecUnary> unary;
    std::vector<ExecRow> rows;
    std::vector<ExecEdge> fc_out;
    std::vector<ExecRow> fc_pre;
    std::vector<ExecStep> steps;
    u32 nsteps = 0;
    u32 first_exist = 0;
    const plan::Check* checks = nullptr;
    const plan::NumCheck* nchecks = nullptr;
    // scratch (written per state: whole cache lines, see LineAllocator)
    LineVector<u64> cand;     // steps * OW
    LineVector<u64> dom1;     // total * OW
    LineVector<u64> fc_doms;  // (steps + 1) * total * OW
    LineVector<u8> bound;     // total
};

class Engine
{
public:
    explicit Engine(const Task& task);
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void instantiate(ExecMatcher& x, const plan::Matcher& m);

    [[nodiscard]] const Task& task() const noexcept { return *m_task; }
    [[nodiscard]] const plan::Compiled& compiled() const noexcept { return *m_c; }
    [[nodiscard]] const AtomIndex& atoms() const noexcept { return *m_atoms; }
    [[nodiscard]] u32 ow() const noexcept { return m_ow; }

    // ---------------------------------------------------------------------------------------------- state and view
    /// Points the engine at a state (not copied: it must stay valid and unchanged while the engine uses it).
    void set_state(const u64* w, u32 nw) noexcept
    {
        m_state = w;
        m_nw = nw;
    }
    [[nodiscard]] const u64* state() const noexcept { return m_state; }
    /// Points the engine at the numeric words of the current state (numeric tasks).
    void set_numeric(const u64* num) noexcept { m_num = num; }
    [[nodiscard]] const u64* numeric() const noexcept { return m_num; }
    [[nodiscard]] const plan::Numeric& numerics() const noexcept { return *m_numc; }
    /// Value of a numeric program under the current binding and state.
    [[nodiscard]] f64 eval(plan::NumProg p) const noexcept { return plan::eval(*m_numc, p, m_num, m_bind.data()); }
    /// Truth of a numeric constraint under the current binding and state.
    [[nodiscard]] bool holds(const plan::NumCheck& c) const noexcept { return plan::holds(*m_numc, c, m_num, m_bind.data()); }
    [[nodiscard]] u32 state_words() const noexcept { return m_nw; }
    /// Row `row` of the view tables (plan::ViewLayout; ow() words), as built for the current state.
    [[nodiscard]] const u64* view_row(u32 row) const noexcept { return m_view.data() + static_cast<usize>(row) * m_ow; }

    /// Slots of the true atoms of driver predicate `d` (plan::Compiled::drivers) in the current state, in slot order.
    [[nodiscard]] std::span<const u32> true_atoms(u32 d) const noexcept { return {m_true[d].data(), m_true[d].size()}; }

    /// Rebuilds the view tables (and the driver predicates' atom lists) from the current state's set bits.
    void build_view()
    {
        if (m_drivers) [[unlikely]]
            build_view_impl<true>();
        else
            build_view_impl<false>();
    }

private:
    template<bool Drivers>
    void build_view_impl()
    {
        const u32 OW = m_ow;
        u64* view = m_view.data();
        if constexpr (Drivers)
            for (LineVector<u32>& t : m_true)
                t.clear();
        for (u32 i = 0; i < m_ntouched; ++i)
        {
            const u32 r = m_touched[i];
            u64* row = view + static_cast<usize>(r) * OW;
            MYMYR_NOVECTOR
            for (u32 w = 0; w < OW; ++w)
                row[w] = 0;
            m_mark[r] = 0;
        }
        m_ntouched = 0;
        MYMYR_NOVECTOR
        for (u32 wi = 0; wi < m_nw; ++wi)
        {
            u64 word = m_state[wi];
            while (word)
            {
                const u32 s = wi * 64 + static_cast<u32>(bits::ctz64(word));
                word &= word - 1;
                apply_op(m_atoms->view_op(AtomKind::Fluent, s));
                if constexpr (Drivers)
                    if (const u32 d = m_driver_of[m_atoms->record(AtomKind::Fluent, s)[0]]; d != ~u32{0})
                        m_true[d].push_back(s);
            }
        }
    }

public:

    /// Derived bitset of the current state (axioms module). Every set bit is an assigned derived slot, so clearing
    /// the words of the assigned slots clears everything.
    void clear_derived()
    {
        const usize n = std::min<usize>(m_derived.size(), bits::words_for(m_atoms->derived_slots()));
        std::memset(m_derived.data(), 0, n * sizeof(u64));
    }
    /// Sets derived slot d; returns false if it was already set. Adds the atom to the view.
    bool add_derived(u32 d)
    {
        u64& w = m_derived[d >> 6];
        const u64 bit = u64{1} << (d & 63);
        if (w & bit)
            return false;
        w |= bit;
        apply_op(m_atoms->view_op(AtomKind::Derived, d));
        return true;
    }
    [[nodiscard]] const u64* derived() const noexcept { return m_derived.data(); }
    [[nodiscard]] u32 derived_words() const noexcept { return m_dnw; }

    // ---------------------------------------------------------------------------------------------- atoms
    [[nodiscard]] ObjectId* bind() noexcept { return m_bind.data(); }
    [[nodiscard]] const ObjectId* bind() const noexcept { return m_bind.data(); }

    [[nodiscard]] u64 key(const plan::Pattern& p) const noexcept
    {
        u64 k = p.base;
        const plan::PatternVar* v = m_vars + p.var_begin;
        MYMYR_NOVECTOR
        for (u32 i = 0; i < p.var_count; ++i)
            k += v[i].rs[m_bind[v[i].param].v];
        return k;
    }
    /// Slot of canonical id k: a slot, or a sentinel (k_empty / k_pending) whose bit test is always false.
    [[nodiscard]] u32 slot(u64 k) const noexcept
    {
        return m_slot_of[k < m_total ? k : m_total].load(std::memory_order_relaxed);
    }
    /// Truth of the literal under the current binding, compared with its polarity.
    [[nodiscard]] bool holds(const plan::Check& c) const noexcept
    {
        const u64 k = key(c.pat);
        bool t;
        switch (c.pat.kind)
        {
            case plan::LitKind::Fluent: t = bits::test(m_state, m_nw, slot(k)); break;
            case plan::LitKind::Derived: t = bits::test(m_derived.data(), m_dnw, slot(k)); break;
            default: t = m_c->statics[c.pat.pred].contains(k); break;
        }
        return t == c.pos;
    }
    /// Existing slot of the pattern's atom or AtomIndex::k_empty.
    [[nodiscard]] u32 find(const plan::Pattern& p) const noexcept
    {
        const u32 s = slot(key(p));
        return s < AtomIndex::k_pending ? s : AtomIndex::k_empty;
    }
    /// Slot of the pattern's atom, assigning one on first touch.
    [[nodiscard]] u32 intern(const plan::Pattern& p) const { return m_atoms->intern(key(p)); }

    // ---------------------------------------------------------------------------------------------- matching
    [[nodiscard]] bool stopped() const noexcept { return m_stop; }
    void clear_stop() noexcept { m_stop = false; }

    /// Enumerates the bindings of x's free parameters (prebound parameters already in bind()). Calls emit() once per
    /// binding (once per effect-relevant binding under witness pruning).
    template<bool Stop, class Emit>
    void run(ExecMatcher& x, Emit& emit)
    {
        run_impl<Stop, false>(x, emit, nullptr);
    }
    /// run() with every free parameter v restricted to the objects of mask[v * ow() .. (v + 1) * ow()) (symmetry
    /// pruning: successor/symmetry.hpp).
    template<bool Stop, class Emit>
    void run_masked(ExecMatcher& x, Emit& emit, const u64* mask)
    {
        run_impl<Stop, true>(x, emit, mask);
    }

private:
    template<bool Stop, bool Masked, class Emit>
    void run_impl(ExecMatcher& x, Emit& emit, [[maybe_unused]] const u64* mask)
    {
        const plan::Matcher& m = *x.plan;
        if (m.never)
            return;
        for (const plan::Check& c : m.pre_checks)
            if (!holds(c))
                return;
        for (const plan::NumCheck& c : m.npre)
            if (!holds(c))
                return;
        if (x.nsteps == 0)
        {
            if (!emit())
                m_stop = true;
            return;
        }
        const u32 OW = m_ow;
        for (const ExecStep* st = x.steps.data(), *se = st + x.nsteps; st != se; ++st)
        {
            u64* d = st->dom1;
            const u64* d0 = st->dom0;
            const ExecUnary* ub = st->ub;
            const ExecUnary* ue = st->ue;
            u64 any = 0;
            [[maybe_unused]] const u64* mk = nullptr;
            if constexpr (Masked)
                mk = mask + static_cast<usize>(st->param) * OW;
            MYMYR_NOVECTOR
            for (u32 w = 0; w < OW; ++w)
            {
                u64 a = d0[w];
                if constexpr (Masked)
                    a &= mk[w];
                for (const ExecUnary* u = ub; u != ue; ++u)
                    a &= u->ptr[w] ^ u->flip;
                d[w] = a;
                any |= a;
            }
            if (!any)
                return;
        }
        if (m.use_fc)
        {
            u64* lvl0 = x.fc_doms.data();
            for (u32 v : m.free_params)
                std::memcpy(lvl0 + static_cast<usize>(v) * OW, x.dom1.data() + static_cast<usize>(v) * OW, OW * sizeof(u64));
            for (usize i = 0; i < x.fc_pre.size(); ++i)
            {
                const ExecRow& r = x.fc_pre[i];
                u64* d = lvl0 + static_cast<usize>(m.fc_pre_to[i]) * OW;
                const u64* row = r.base + static_cast<usize>(m_bind[r.src].v) * OW;
                u64 any = 0;
                MYMYR_NOVECTOR
                for (u32 w = 0; w < OW; ++w)
                    any |= (d[w] &= row[w] ^ r.flip);
                if (!any)
                    return;
            }
            if (m.nchecks.empty()) [[likely]]
                search_fc<Stop, false>(x, 0, static_cast<u32>(m.free_params.size()), false, emit);
            else
                search_fc<Stop, true>(x, 0, static_cast<u32>(m.free_params.size()), false, emit);
            return;
        }
        // numeric constraints (numeric tasks) in their own instantiation: the classical search checks none
        if (m.nchecks.empty()) [[likely]]
            search<Stop, false>(x, 0, false, emit);
        else
            search<Stop, true>(x, 0, false, emit);
    }

    void apply_op(const ViewOp& op) noexcept
    {
        if (op.row1 != ViewOp::k_none)
        {
            view_set(op.row1, op.bit1);
            if (op.row2 != ViewOp::k_none)
                view_set(op.row2, op.bit2);
        }
    }
    void view_set(u32 row, u32 bit) noexcept
    {
        m_view[static_cast<usize>(row) * m_ow + (bit >> 6)] |= u64{1} << (bit & 63);
        if (!m_mark[row])
        {
            m_mark[row] = 1;
            m_touched[m_ntouched++] = row;
        }
    }

    /// Whether the numeric constraints nchecks[i], i in [b, e), all hold.
    [[gnu::noinline]] bool holds_all(const plan::NumCheck* nchecks, const u32* b, const u32* e) const noexcept
    {
        for (const u32* c = b; c != e; ++c)
            if (!holds(nchecks[*c]))
                return false;
        return true;
    }

    // Returns true if a full binding was found below step d (used in witness mode).
    template<bool Stop, bool Num, class Emit>
    bool search(ExecMatcher& x, u32 d, bool witness, Emit& emit)
    {
        if (d == x.nsteps)
        {
            if (!witness && !emit())
                m_stop = true;
            return true;
        }
        if (d == x.first_exist && !witness)
        {
            // The remaining parameters occur in no effect: one witness binding suffices.
            const bool found = search<Stop, Num>(x, d, true, emit);
            if (found && !emit())
                m_stop = true;
            return found;
        }
        const u32 OW = m_ow;
        const ExecStep& st = x.steps[d];
        u64* cd = st.cand;
        const u64* d1 = st.dom1;
        const ExecRow* rb = st.rb;
        const ExecRow* re = st.re;
        ObjectId* bind = m_bind.data();
        u64 any = 0;
        MYMYR_NOVECTOR
        for (u32 w = 0; w < OW; ++w)
        {
            u64 a = d1[w];
            for (const ExecRow* r = rb; r != re; ++r)
                a &= r->base[static_cast<usize>(bind[r->src].v) * OW + w] ^ r->flip;
            cd[w] = a;
            any |= a;
        }
        if (!any)
            return false;
        const u32* cb = st.cb;
        const u32* ce = st.ce;
        const plan::Check* checks = x.checks;
        const u32 param = st.param;
        MYMYR_NOVECTOR
        for (u32 w = 0; w < OW; ++w)
        {
            u64 word = cd[w];
            while (word)
            {
                const u32 o = w * 64 + static_cast<u32>(bits::ctz64(word));
                word &= word - 1;
                bind[param] = ObjectId{o};
                bool ok = true;
                for (const u32* c = cb; c != ce; ++c)
                    if (!holds(checks[*c]))
                    {
                        ok = false;
                        break;
                    }
                if (!ok)
                    continue;
                if constexpr (Num)
                    if (st.ncb != st.nce && !holds_all(x.nchecks, st.ncb, st.nce))
                        continue;
                if (search<Stop, Num>(x, d + 1, witness, emit) && witness)
                    return true;
                if constexpr (Stop)
                    if (m_stop)
                        return false;
            }
        }
        return false;
    }

    // Fail-first forward checking: branch on the unbound parameter with the fewest candidates (effect-relevant
    // parameters first); after binding, prune every neighbour's domain and fail as soon as one becomes empty.
    template<bool Stop, bool Num, class Emit>
    bool search_fc(ExecMatcher& x, u32 level, u32 unbound, bool witness, Emit& emit)
    {
        const plan::Matcher& m = *x.plan;
        if (unbound == 0)
        {
            if (!witness && !emit())
                m_stop = true;
            return true;
        }
        const u32 OW = m_ow;
        const usize T = static_cast<usize>(m.total) * OW;
        u64* cur = x.fc_doms.data() + level * T;
        u8* bound = x.bound.data();
        bool any_relevant = false;
        for (u32 v : m.free_params)
            if (!bound[v] && m.relevant[v])
            {
                any_relevant = true;
                break;
            }
        if (!any_relevant && !witness)
        {
            const bool found = search_fc<Stop, Num>(x, level, unbound, true, emit);
            if (found && !emit())
                m_stop = true;
            return found;
        }
        u32 best = 0, best_cnt = ~u32{0};
        for (u32 v : m.free_params)
        {
            if (bound[v] || (any_relevant && !m.relevant[v]))
                continue;
            u32 cnt = 0;
            const u64* dv = cur + static_cast<usize>(v) * OW;
            MYMYR_NOVECTOR
            for (u32 w = 0; w < OW; ++w)
                cnt += static_cast<u32>(bits::popcount64(dv[w]));
            if (cnt < best_cnt)
                best = v, best_cnt = cnt;
        }
        const u32 p = best;
        u64* nxt = cur + T;
        const u64* dp = cur + static_cast<usize>(p) * OW;
        const ExecEdge* eb = x.fc_out.data() + m.fc_out_begin[p];
        const ExecEdge* ee = x.fc_out.data() + m.fc_out_begin[p + 1];
        const u32* kb = m.fc_checks.data() + m.fc_checks_begin[p];
        const u32* ke = m.fc_checks.data() + m.fc_checks_begin[p + 1];
        bound[p] = 1;
        MYMYR_NOVECTOR
        for (u32 w = 0; w < OW; ++w)
        {
            u64 word = dp[w];
            while (word)
            {
                const u32 o = w * 64 + static_cast<u32>(bits::ctz64(word));
                word &= word - 1;
                m_bind[p] = ObjectId{o};
                bool ok = true;
                for (const u32* k = kb; k != ke; ++k)
                {
                    bool all = true;
                    MYMYR_NOVECTOR
                    for (u32 j = m.check_vars_begin[*k]; j < m.check_vars_begin[*k + 1]; ++j)
                        all &= bound[m.check_vars[j]] != 0;
                    if (!all)
                        continue;
                    if (!holds(m.checks[*k]))
                    {
                        ok = false;
                        break;
                    }
                }
                if constexpr (Num)
                    for (u32 j = m.fc_nchecks_begin[p]; ok && j < m.fc_nchecks_begin[p + 1]; ++j)
                    {
                        const u32 k = m.fc_nchecks[j];
                        bool all = true;
                        for (u32 i = m.ncheck_vars_begin[k]; i < m.ncheck_vars_begin[k + 1]; ++i)
                            all &= bound[m.ncheck_vars[i]] != 0;
                        if (all && !holds(m.nchecks[k]))
                            ok = false;
                    }
                if (!ok)
                    continue;
                std::memcpy(nxt, cur, T * sizeof(u64));
                for (const ExecEdge* e = eb; e != ee; ++e)
                {
                    if (bound[e->to])
                        continue;
                    u64* dt = nxt + static_cast<usize>(e->to) * OW;
                    const u64* row = e->base + static_cast<usize>(o) * OW;
                    u64 any = 0;
                    MYMYR_NOVECTOR
                    for (u32 i = 0; i < OW; ++i)
                        any |= (dt[i] &= row[i] ^ e->flip);
                    if (!any)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    continue;
                if (search_fc<Stop, Num>(x, level + 1, unbound - 1, witness, emit) && witness)
                {
                    bound[p] = 0;
                    return true;
                }
                if constexpr (Stop)
                    if (m_stop)
                    {
                        bound[p] = 0;
                        return false;
                    }
            }
        }
        bound[p] = 0;
        return false;
    }

    const Task* m_task;
    const plan::Compiled* m_c;
    const AtomIndex* m_atoms;
    const std::atomic<u32>* m_slot_of;
    u64 m_total;
    const plan::PatternVar* m_vars;
    const plan::Numeric* m_numc;
    const u64* m_num = nullptr;
    u32 m_ow;
    LineVector<u64> m_view;
    LineVector<u8> m_mark;
    LineVector<u32> m_touched;
    u32 m_ntouched = 0;
    LineVector<ObjectId> m_bind;
    const u64* m_state = nullptr;
    u32 m_nw = 0;
    LineVector<u64> m_derived;
    u32 m_dnw = 0;
    bool m_stop = false;
    // atom-driven conditional effects (plan::CondEffect::driver)
    bool m_drivers = false;
    std::vector<u32> m_driver_of;           // per predicate: driver index or ~0
    std::vector<LineVector<u32>> m_true;    // per driver: true atoms of the current state
};
}  // namespace mymyr::detail
