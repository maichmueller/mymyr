#pragma once
// Successor generation: the public API over the per-thread engine.
//   - for_each_applicable(state, f): f(const ActionLabel&, const Delta&) for every applicable action. The Delta holds
//     the raw effect slots (conditional effects evaluated in the state), to be applied as "delete, then add";
//   - any_applicable(state), is_applicable(state, action), apply(state, action);
//   - witness pruning (on by default): parameters that occur in no effect are bound last and only need one witness.
//     It must be off for labelled state spaces, where every binding is a distinct transition;
//   - canonical order (on by default): per state, actions come ordered by schema, then lexicographically by
//     binding (the schema's parameter order). Schemas whose matcher already binds its parameters in index order with
//     the fixed-order search emit in that order for free; the others are buffered per schema and sorted.
//   - symmetry pruning (off by default; successor/symmetry.hpp): generate(..., SymmetryPruning::Wl1) emits only
//     the actions whose parameters are bound to representatives of the objects' colour classes in the state.
// Numeric tasks (task/numeric.hpp): numeric preconditions are checked inside the matchers; a binding whose numeric
// or total-cost effects break mimir's applicability rules is not applicable (never emitted); the Delta carries the
// successor's numeric words and the total-cost effects of the action (the shared cost evaluation: every search takes
// action costs from it through heuristics::ActionCosts).
// A Successors object lives in a Workspace (per thread, per task): obtain it with task.workspace().successors().
// Calls on one object are not reentrant: do not call it from inside its own callback.

#include "mymyr/axioms/evaluator.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/successor/detail/engine.hpp"
#include "mymyr/successor/detail/symmetry.hpp"
#include "mymyr/successor/symmetry.hpp"
#include "mymyr/task/plan.hpp"

#include <algorithm>
#include <bit>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mymyr
{
class Successors
{
public:
    Successors(detail::Engine& engine, AxiomEvaluator& axioms);
    Successors(const Successors&) = delete;
    Successors& operator=(const Successors&) = delete;

    void set_witness_pruning(bool on) noexcept { m_witness = on; }
    void set_canonical_order(bool on) noexcept { m_canonical = on; }
    /// Restricts generate() to the schemas s with only[s] != 0 (nullptr: every schema; the array must outlive its
    /// use). The CUDA backend runs the schemas its kernels do not support on the CPU this way (per-schema fallback).
    void set_schema_filter(const u8* only) noexcept { m_only = only; }
    [[nodiscard]] bool witness_pruning() const noexcept { return m_witness; }
    [[nodiscard]] bool canonical_order() const noexcept { return m_canonical; }

    [[nodiscard]] detail::Engine& engine() noexcept { return m_e; }
    [[nodiscard]] u32 arity(u32 schema) const noexcept { return m_arity[schema]; }
    [[nodiscard]] u32 num_schemas() const noexcept { return static_cast<u32>(m_arity.size()); }
    [[nodiscard]] AxiomEvaluator& axioms() noexcept { return m_ax; }

    // ------------------------------------------------------------------------------------------ high-level API
    /// Calls f(const ActionLabel&, const Delta&) for every applicable action of s. If f returns bool, false stops
    /// the enumeration. The label and delta are valid during the call only.
    template<class F>
    void for_each_applicable(StateView s, F&& f)
    {
        prepare(s);
        generate<std::is_same_v<std::invoke_result_t<F&, const ActionLabel&, const Delta&>, bool>>(
            [&](u32 schema, const ObjectId* binding, const Delta& d) -> bool
            {
                const ActionLabel label{SchemaId{schema}, {binding, m_arity[schema]}};
                if constexpr (std::is_same_v<std::invoke_result_t<F&, const ActionLabel&, const Delta&>, bool>)
                    return f(label, d);
                else
                {
                    f(label, d);
                    return true;
                }
            });
    }

    /// The applicable actions of s (in canonical order when canonical order is on).
    [[nodiscard]] std::vector<Action> applicable_actions(StateView s)
    {
        std::vector<Action> out;
        for_each_applicable(s, [&](const ActionLabel& a, const Delta&) { out.emplace_back(a); });
        return out;
    }

    [[nodiscard]] bool any_applicable(StateView s)
    {
        prepare(s);
        return any();
    }

    /// Whether `a` (schema and full binding) is applicable in s.
    [[nodiscard]] bool is_applicable(StateView s, const ActionLabel& a);

    /// Writes the successor of s under `a` into out. Throws std::invalid_argument if `a` is not applicable in s.
    void apply(StateView s, const ActionLabel& a, StateBuilder& out);
    /// apply(s, a, out) that also returns the action's Delta (valid until the next call on this object).
    Delta apply_with_delta(StateView s, const ActionLabel& a, StateBuilder& out);
    [[nodiscard]] State apply(StateView s, const ActionLabel& a)
    {
        StateBuilder b;
        apply(s, a, b);
        return b.build();
    }

    /// Goal test of s (evaluates the axioms when the task has any).
    [[nodiscard]] bool is_goal(StateView s)
    {
        if (needs_view_for_goal())
            prepare(s);
        else
        {
            m_e.set_state(s.w, s.nw);
            m_e.set_numeric(s.num);
        }
        return goal_holds();
    }

    // ------------------------------------------------------------------------------------------ low-level API
    // Search loops: prepare(s) once per expanded state, then goal_holds() and generate(emit). The state words must
    // stay valid and unchanged until the next prepare().

    /// Points the engine at s, builds its view and closes it under the axioms.
    void prepare(StateView s)
    {
        m_e.set_state(s.w, s.nw);
        m_e.set_numeric(s.num);
        m_e.build_view();
        if (!m_ax.empty())
            m_ax.evaluate();
    }
    /// Whether the goal holds in the prepared state.
    [[nodiscard]] bool goal_holds() const noexcept
    {
        const plan::Goal& g = m_e.compiled().goal;
        if (g.unsatisfiable)
            return false;
        for (const plan::Check& c : g.lits)
            if (!m_e.holds(c))
                return false;
        for (const plan::NumCheck& c : m_e.compiled().num.goal)
            if (!m_e.holds(c))
                return false;
        return true;
    }
    [[nodiscard]] bool needs_view_for_goal() const noexcept { return m_e.compiled().goal.uses_derived; }

    /// Calls emit(u32 schema, const ObjectId* binding, const Delta& delta) -> bool for every applicable action of the
    /// prepared state. With Stop, emit returning false ends the enumeration and generate returns false.
    template<bool Stop, class Emit>
    bool generate(Emit&& emit)
    {
        return generate<Stop>(static_cast<Emit&&>(emit), m_witness, m_canonical);
    }
    /// generate() with explicit settings (searches pass their own options instead of changing this object's), over
    /// the schemas [first_schema, end_schema) only.
    template<bool Stop, class Emit>
    bool generate(Emit&& emit, bool witness_pruning, bool canonical_order, u32 first_schema = 0,
                  u32 end_schema = ~u32{0})
    {
        return generate_impl<Stop, false>(emit, witness_pruning, canonical_order, first_schema, end_schema);
    }
    /// generate() with symmetry pruning (successor/symmetry.hpp): with SymmetryPruning::Wl1 only the actions whose
    /// parameters are all bound to representatives of their colour classes in the prepared state are emitted. The
    /// classes are computed once per call; with Off this is the plain generate().
    template<bool Stop, class Emit>
    bool generate(Emit&& emit, bool witness_pruning, bool canonical_order, SymmetryPruning symmetry, u32 first_schema = 0,
                  u32 end_schema = ~u32{0})
    {
        if (symmetry == SymmetryPruning::Off)
            return generate_impl<Stop, false>(emit, witness_pruning, canonical_order, first_schema, end_schema);
        symmetry_pruner().compute(m_e);
        return generate_impl<Stop, true>(emit, witness_pruning, canonical_order, first_schema, end_schema);
    }

    /// This thread's symmetry pruning state (created on first use).
    [[nodiscard]] detail::SymmetryPruner& symmetry_pruner();

private:
    template<bool Stop, bool Masked, class Emit>
    bool generate_impl(Emit& emit, bool witness_pruning, bool canonical_order, u32 first_schema, u32 end_schema)
    {
        m_e.clear_stop();
        m_cd.num = m_e.numeric();
        const u32 wi = witness_pruning ? 0 : 1;
        const u32 end = std::min(end_schema, static_cast<u32>(m_schemas.size()));
        for (u32 s = first_schema; s < end; ++s)
        {
            if (m_only && !m_only[s])
                continue;
            SchemaExec& se = m_schemas[s];
            [[maybe_unused]] const u64* mask = nullptr;
            if constexpr (Masked)
            {
                if (se.plan->arity > 0 && !se.pre[wi].plan->never)
                    mask = m_sym->masks(s);
            }
            if (!canonical_order || se.ordered[wi])
            {
                auto f = [&]() -> bool
                {
                    if (!collect(se))
                        return true;  // numeric effects break the applicability rules
                    return emit(s, m_e.bind(), delta(se));
                };
                if constexpr (Masked)
                    m_e.run_masked<Stop>(se.pre[wi], f, mask);
                else
                    m_e.run<Stop>(se.pre[wi], f);
                if constexpr (Stop)
                    if (m_e.stopped())
                        return false;
                continue;
            }
            // canonical order: buffer the bindings of this schema, sort them, then collect effects and emit
            const u32 arity = se.plan->arity;
            m_rows.clear();
            auto buffer = [&]() -> bool
            {
                const ObjectId* b = m_e.bind();
                for (u32 i = 0; i < arity; ++i)
                    m_rows.push_back(b[i].v);
                return true;
            };
            if constexpr (Masked)
                m_e.run_masked<false>(se.pre[wi], buffer, mask);
            else
                m_e.run<false>(se.pre[wi], buffer);
            const usize n = m_rows.size() / arity;
            if (n == 0)
                continue;
            ObjectId* bind = m_e.bind();
            if (arity * m_pack_bits <= 64)
            {
                m_keys.resize(n);
                for (usize r = 0; r < n; ++r)
                {
                    u64 k = 0;
                    for (u32 i = 0; i < arity; ++i)
                        k = (k << m_pack_bits) | m_rows[r * arity + i];
                    m_keys[r] = k;
                }
                if (n > 1)
                    std::sort(m_keys.begin(), m_keys.end());
                const u64 mask = m_pack_bits == 64 ? ~u64{0} : (u64{1} << m_pack_bits) - 1;
                for (usize r = 0; r < n; ++r)
                {
                    u64 k = m_keys[r];
                    for (u32 i = arity; i-- > 0;)
                    {
                        bind[i] = ObjectId{static_cast<u32>(k & mask)};
                        k >>= m_pack_bits;
                    }
                    if (!collect(se))
                        continue;
                    if (!emit(s, static_cast<const ObjectId*>(bind), delta(se)))
                        if constexpr (Stop)
                            return false;
                }
            }
            else
            {
                m_perm.resize(n);
                for (usize r = 0; r < n; ++r)
                    m_perm[r] = static_cast<u32>(r);
                const u32* rows = m_rows.data();
                std::sort(m_perm.begin(), m_perm.end(),
                          [&](u32 a, u32 b)
                          {
                              return std::lexicographical_compare(rows + static_cast<usize>(a) * arity, rows + static_cast<usize>(a + 1) * arity,
                                                                  rows + static_cast<usize>(b) * arity, rows + static_cast<usize>(b + 1) * arity);
                          });
                for (usize r = 0; r < n; ++r)
                {
                    for (u32 i = 0; i < arity; ++i)
                        bind[i] = ObjectId{rows[static_cast<usize>(m_perm[r]) * arity + i]};
                    if (!collect(se))
                        continue;
                    if (!emit(s, static_cast<const ObjectId*>(bind), delta(se)))
                        if constexpr (Stop)
                            return false;
                }
            }
        }
        return true;
    }

public:
    /// Whether the prepared state has an applicable action (witness pruning is used regardless of the setting; symmetry
    /// pruning is not applied).
    [[nodiscard]] bool any();

    /// Effects of the binding currently in engine().bind() for `schema` (preconditions are not checked). Numeric tasks:
    /// `ok` is false when the numeric effects break the applicability rules (the delta is then incomplete).
    Delta effects(u32 schema, bool* ok = nullptr)
    {
        const bool valid = collect(m_schemas[schema]);
        if (ok)
            *ok = valid;
        m_cd.num = m_e.numeric();
        return delta(m_schemas[schema]);
    }

private:
    struct SchemaExec
    {
        const plan::Schema* plan = nullptr;
        detail::ExecMatcher pre[2];  // witness pruning on / off
        std::vector<detail::ExecMatcher> ces;
        bool ordered[2] = {false, false};  // emits in canonical order without sorting
        bool numeric = false;              // numeric or total-cost effects (collect_numeric)
    };

    /// The Delta of the action collected last for schema `se` (valid until the next collect). A schema without numeric
    /// and total-cost effects keeps the parent's numeric words and has no writes: its collect() touches no numeric
    /// scratch and its Delta is m_cd, whose numeric part is set once per state (classical tasks pay nothing).
    [[nodiscard]] const Delta& delta(const SchemaExec& se) noexcept
    {
        if (se.numeric) [[unlikely]]
        {
            m_nd = Delta{m_adds, m_dels, m_dnum, m_nnum, {m_writes.data(), m_writes.size()}, {m_aux.data(), m_aux.size()}};
            return m_nd;
        }
        m_cd.add = {m_adds.data(), m_adds.size()};
        m_cd.del = {m_dels.data(), m_dels.size()};
        return m_cd;
    }

    /// Effects of the bound action: false if its numeric effects break the applicability rules.
    bool collect(SchemaExec& se)
    {
        if (se.numeric) [[unlikely]]
            return collect_numeric(se);
        collect_literals(se);
        return true;
    }
    bool collect_numeric(SchemaExec& se);
    bool numeric_effects(const std::vector<plan::NumEffect>& es, bool has_aux, const plan::AuxEffect& aux,
                         std::vector<NumericWrite>& writes, std::vector<AuxWrite>& auxes);
    void ce_literals(const plan::CondEffect& ce)
    {
        for (const plan::Pattern& p : ce.dels)
            if (const u32 s = m_e.find(p); s != AtomIndex::k_empty)
                m_dels.push_back(SlotId{s});
        for (const plan::Pattern& p : ce.adds)
            m_adds.push_back(SlotId{m_e.intern(p)});
    }

    /// Calls f() for every firing binding of conditional effect `ce` (bound in m_e.bind()): the matcher, or the true
    /// atoms of its driver predicate (plan::CondEffect::driver).
    template<class F>
    void run_ce(const plan::CondEffect& ce, detail::ExecMatcher& xm, F& f)
    {
        if (ce.driver == plan::CondEffect::k_no_driver)
        {
            m_e.run<false>(xm, f);
            return;
        }
        const plan::Matcher& m = ce.cond;
        ObjectId* bind = m_e.bind();
        const formalism::Term* terms = ce.driver_terms.data();
        const u32 n = static_cast<u32>(ce.driver_terms.size()), OW = m_e.ow();
        const AtomIndex& atoms = m_e.atoms();
        for (u32 s : m_e.true_atoms(ce.driver))
        {
            const u32* args = atoms.record(AtomKind::Fluent, s) + 1;
            for (u32 i = 0; i < n; ++i)
                if (terms[i] >= static_cast<formalism::Term>(m.first_free))
                    bind[terms[i]] = ObjectId{args[i]};
            bool ok = true;
            for (u32 i = 0; i < n && ok; ++i)  // objects, prebound parameters and repeated forall parameters
                ok = (terms[i] >= 0 ? bind[terms[i]].v : formalism::term_object(terms[i]).v) == args[i];
            for (u32 v = m.first_free; v < m.total && ok; ++v)
            {
                const u32 o = bind[v].v;
                ok = (m.dom0[static_cast<usize>(v) * OW + (o >> 6)] >> (o & 63)) & 1;
            }
            for (usize i = 0; i < ce.lits.size() && ok; ++i)
                ok = m_e.holds(ce.lits[i]);
            for (usize i = 0; i < m.npre.size() && ok; ++i)
                ok = m_e.holds(m.npre[i]);
            for (usize i = 0; i < m.nchecks.size() && ok; ++i)
                ok = m_e.holds(m.nchecks[i]);
            if (ok)
                f();
        }
    }

    void collect_literals(SchemaExec& se)
    {
        m_adds.clear();
        m_dels.clear();
        const plan::Schema& ps = *se.plan;
        for (const plan::Pattern& p : ps.dels)
            if (const u32 s = m_e.find(p); s != AtomIndex::k_empty)
                m_dels.push_back(SlotId{s});
        for (const plan::Pattern& p : ps.adds)
            m_adds.push_back(SlotId{m_e.intern(p)});
        for (usize i = 0; i < ps.ces.size(); ++i)
        {
            const plan::CondEffect& ce = ps.ces[i];
            auto f = [&]() -> bool
            {
                for (const plan::Pattern& p : ce.dels)
                    if (const u32 s = m_e.find(p); s != AtomIndex::k_empty)
                        m_dels.push_back(SlotId{s});
                for (const plan::Pattern& p : ce.adds)
                    m_adds.push_back(SlotId{m_e.intern(p)});
                return true;
            };
            run_ce(ce, se.ces[i], f);
        }
    }
    bool bind_label(const ActionLabel& a);
    bool applicable_bound(u32 schema) const;

    detail::Engine& m_e;
    AxiomEvaluator& m_ax;
    std::vector<SchemaExec> m_schemas;
    std::vector<u32> m_arity;
    LineVector<SlotId> m_adds, m_dels;  // per-state scratch (see LineAllocator)
    Delta m_cd, m_nd;                   // delta(): classical schemas, numeric schemas
    LineVector<u32> m_rows, m_perm;
    LineVector<u64> m_keys;
    // numerics (per-state scratch)
    const u64* m_dnum = nullptr;  // the delta's numeric words: the parent's, or m_num_next
    u32 m_nnum = 0;
    LineVector<u64> m_num_next;
    LineVector<u8> m_fam;  // per slot: the effect family recorded for the current binding
    LineVector<u32> m_fam_touched;
    u8 m_aux_fam = 0;
    std::vector<NumericWrite> m_writes, m_tw;
    std::vector<AuxWrite> m_aux, m_ta;
    std::vector<f64> m_vals;  // per slot: the successor value being built
    LineVector<u32> m_val_touched;
    LineVector<u8> m_val_mark;
    u32 m_pack_bits = 1;
    const u8* m_only = nullptr;  // set_schema_filter
    std::unique_ptr<detail::SymmetryPruner> m_sym;  // symmetry_pruner()
    bool m_witness = true;
    bool m_canonical = true;
};

/// Builds the successor words of `cur` (n words) under delta d (delete, then add) into out, grown as needed.
/// Returns the trimmed length. `out` is any vector of u64 (a LineVector for per-thread hot loops).
template<class Vec>
inline u32 apply_delta(const u64* cur, u32 n, const Delta& d, Vec& out)
{
    u32 wn = n;
    MYMYR_NOVECTOR
    for (SlotId a : d.add)
    {
        const u32 w = bits::word_of(a.v) + 1;
        wn = w > wn ? w : wn;
    }
    if (out.size() < wn) [[unlikely]]
        out.resize(wn + 8, 0);
    u64* nx = out.data();
    u32 i = 0;
    MYMYR_NOVECTOR
    for (; i < n; ++i)  // states are a few words: a loop beats a memcpy call
        nx[i] = cur[i];
    MYMYR_NOVECTOR
    for (; i < wn; ++i)
        nx[i] = 0;
    for (SlotId x : d.del)
        if (bits::word_of(x.v) < wn)
            bits::reset(nx, x.v);
    for (SlotId x : d.add)
        bits::set(nx, x.v);
    MYMYR_NOVECTOR
    while (wn > 0 && nx[wn - 1] == 0)
        --wn;
    return wn;
}
}  // namespace mymyr
