#include "mymyr/successor/successors.hpp"

#include "mymyr/task/task.hpp"

#include <cmath>

namespace mymyr
{
Successors::Successors(detail::Engine& engine, AxiomEvaluator& axioms) : m_e(engine), m_ax(axioms)
{
    const plan::Compiled& C = engine.compiled();
    m_schemas.resize(C.schemas.size());
    m_arity.resize(C.schemas.size());
    for (usize s = 0; s < C.schemas.size(); ++s)
    {
        const plan::Schema& ps = C.schemas[s];
        SchemaExec& se = m_schemas[s];
        se.plan = &ps;
        se.numeric = ps.numeric();
        m_arity[s] = ps.arity;
        for (u32 w = 0; w < 2; ++w)
        {
            engine.instantiate(se.pre[w], ps.pre[w]);
            se.ordered[w] = ps.arity <= 1 || ps.pre[w].binds_in_order;
        }
        se.ces.resize(ps.ces.size());
        for (usize i = 0; i < ps.ces.size(); ++i)
            engine.instantiate(se.ces[i], ps.ces[i].cond);
    }
    const u32 n = std::max<u32>(2, C.num_objects);
    m_pack_bits = static_cast<u32>(std::bit_width(n - 1));
    const plan::Numeric& N = C.num;
    m_nnum = N.words;
    m_cd.nnum = N.words;
    if (N.slots)
    {
        m_num_next.assign(N.words, 0);
        m_fam.assign(N.slots, 0);
        m_vals.assign(N.slots, 0);
        m_val_mark.assign(N.slots, 0);
    }
}

detail::SymmetryPruner& Successors::symmetry_pruner()
{
    if (!m_sym)
        m_sym = std::make_unique<detail::SymmetryPruner>(m_e.task(), m_e.ow());
    return *m_sym;
}

// The applicability rules for the numeric and total-cost effects of one (conditional) effect that fires, checked in
// mimir's order (ActionSatisficingBindingGenerator::is_valid_binding): fluent effects, then the total-cost effect; each
// records its family on its target before its value is checked. The effects are appended to `writes` / `auxes`.
bool Successors::numeric_effects(const std::vector<plan::NumEffect>& es, bool has_aux, const plan::AuxEffect& aux,
                                 std::vector<NumericWrite>& writes, std::vector<AuxWrite>& auxes)
{
    const plan::Numeric& N = m_e.numerics();
    for (const plan::NumEffect& e : es)
    {
        u32 slot = e.slot;
        if (e.lifted)
        {
            const plan::FunctionTable& F = N.tables[e.table];
            const formalism::Term* t = N.terms.data() + e.terms;
            const ObjectId* bind = m_e.bind();
            u64 k = 0;
            for (u32 j = 0; j < e.arity; ++j)
                k += F.rs[static_cast<usize>(j) * N.num_objects + (t[j] >= 0 ? bind[t[j]].v : formalism::term_object(t[j]).v)];
            slot = F.slot_of(k);
        }
        if (slot == plan::FunctionTable::k_none)
            return false;  // the target has no value (mimir allows `assign` to it; see task/numeric.hpp)
        const u8 f = plan::effect_family(e.op);
        if (!plan::compatible_family(m_fam[slot], f))
            return false;
        if (!m_fam[slot])
            m_fam_touched.push_back(slot);
        m_fam[slot] = f;
        const f64 v = m_e.eval(e.expr);
        if (!plan::defined_effect(e.op, v))
            return false;
        writes.push_back({slot, e.op, v});
    }
    if (has_aux)
    {
        const u8 f = plan::effect_family(aux.op);
        if (!plan::compatible_family(m_aux_fam, f))
            return false;
        m_aux_fam = f;
        const f64 v = m_e.eval(aux.expr);
        if (!plan::defined_effect(aux.op, v))
            return false;
        auxes.push_back({aux.op, v});
    }
    return true;
}

bool Successors::collect_numeric(SchemaExec& se)
{
    const plan::Schema& ps = *se.plan;
    m_adds.clear();
    m_dels.clear();
    m_writes.clear();
    m_aux.clear();
    m_aux_fam = 0;
    for (const plan::Pattern& p : ps.dels)
        if (const u32 s = m_e.find(p); s != AtomIndex::k_empty)
            m_dels.push_back(SlotId{s});
    for (const plan::Pattern& p : ps.adds)
        m_adds.push_back(SlotId{m_e.intern(p)});
    auto commit = [&]
    {
        m_writes.insert(m_writes.end(), m_tw.begin(), m_tw.end());
        m_aux.insert(m_aux.end(), m_ta.begin(), m_ta.end());
    };
    bool ok = true;
    for (const plan::NumRef& r : ps.num_order)
    {
        if (!r.conditional)
        {
            // unconditional: a failure makes the action inapplicable, so the writes go straight to the delta
            const plan::NumGroup& g = ps.uncond_num[r.index];
            if (!numeric_effects(g.neffs, g.has_aux, g.aux, m_writes, m_aux))
            {
                ok = false;
                break;
            }
            continue;
        }
        m_tw.clear();
        m_ta.clear();
        const plan::CondEffect& ce = ps.ces[r.index];
        detail::ExecMatcher& xm = se.ces[r.index];
        if (!ce.extras)
        {
            // only an effect that fires executes: an effect that does not fire records no family, so it cannot
            // conflict with one that does
            bool fires = false;
            auto f = [&]() -> bool
            {
                fires = true;
                return true;
            };
            m_e.run<false>(xm, f);
            if (!fires)
                continue;
            if (!numeric_effects(ce.neffs, ce.has_aux, ce.aux, m_tw, m_ta))
            {
                ok = false;
                break;
            }
            commit();
            ce_literals(ce);
            continue;
        }
        // forall parameters: every firing instance, checked as it fires
        auto f = [&]() -> bool
        {
            if (!ok)
                return true;
            m_tw.clear();
            m_ta.clear();
            if (!numeric_effects(ce.neffs, ce.has_aux, ce.aux, m_tw, m_ta))
            {
                ok = false;
                return true;
            }
            commit();
            ce_literals(ce);
            return true;
        };
        m_e.run<false>(xm, f);
        if (!ok)
            break;
    }
    for (u32 s : m_fam_touched)
        m_fam[s] = 0;
    m_fam_touched.clear();
    if (!ok)
        return false;
    for (usize i = 0; i < ps.ces.size(); ++i)
    {
        const plan::CondEffect& ce = ps.ces[i];
        if (ce.numeric)
            continue;
        auto f = [&]() -> bool
        {
            ce_literals(ce);
            return true;
        };
        run_ce(ce, se.ces[i], f);
    }
    // the successor's numeric words: the writes applied in order to the parent's values, stored canonically
    if (m_writes.empty())
    {
        m_dnum = m_e.numeric();
        return true;
    }
    const plan::Numeric& N = m_e.numerics();
    const u64* parent = m_e.numeric();
    for (const NumericWrite& w : m_writes)
    {
        if (!m_val_mark[w.slot])
        {
            m_val_mark[w.slot] = 1;
            m_val_touched.push_back(w.slot);
            m_vals[w.slot] = plan::load(N, parent, w.slot);
        }
        m_vals[w.slot] = apply_assign(w.op, m_vals[w.slot], w.value);
    }
    u64* out = m_num_next.data();
    MYMYR_NOVECTOR
    for (u32 i = 0; i < m_nnum; ++i)
        out[i] = parent[i];
    for (u32 s : m_val_touched)
    {
        m_val_mark[s] = 0;
        plan::store(N, out, s, m_vals[s]);
    }
    m_val_touched.clear();
    m_dnum = out;
    return true;
}

bool Successors::any()
{
    m_e.clear_stop();
    for (SchemaExec& se : m_schemas)
    {
        if (se.numeric)
        {
            auto f = [&]() -> bool { return !collect(se); };  // stop at the first binding that is applicable
            m_e.run<true>(se.pre[0], f);
        }
        else
        {
            auto f = []() -> bool { return false; };
            m_e.run<true>(se.pre[0], f);
        }
        if (m_e.stopped())
        {
            m_e.clear_stop();
            return true;
        }
    }
    return false;
}

bool Successors::bind_label(const ActionLabel& a)
{
    if (a.schema.v >= m_schemas.size() || a.binding.size() != m_arity[a.schema.v])
        return false;
    const u32 n = m_e.compiled().num_objects;
    ObjectId* b = m_e.bind();
    for (usize i = 0; i < a.binding.size(); ++i)
    {
        if (a.binding[i].v >= n)
            return false;
        b[i] = a.binding[i];
    }
    return true;
}

bool Successors::applicable_bound(u32 schema) const
{
    const plan::Schema& ps = *m_schemas[schema].plan;
    for (const plan::Check& c : ps.pre_lits)
        if (!m_e.holds(c))
            return false;
    for (const plan::NumCheck& c : ps.pre_nums)
        if (!m_e.holds(c))
            return false;
    return true;
}

bool Successors::is_applicable(StateView s, const ActionLabel& a)
{
    prepare(s);  // the axioms use the binding array: bind the label afterwards
    if (!bind_label(a) || !applicable_bound(a.schema.v))
        return false;
    return !m_schemas[a.schema.v].numeric || collect(m_schemas[a.schema.v]);
}

Delta Successors::apply_with_delta(StateView s, const ActionLabel& a, StateBuilder& out)
{
    prepare(s);  // the axioms use the binding array: bind the label afterwards
    if (!bind_label(a))
        throw std::invalid_argument("mymyr: apply: malformed action label");
    if (!applicable_bound(a.schema.v) || !collect(m_schemas[a.schema.v]))
        throw std::invalid_argument("mymyr: apply: the action is not applicable in the state");
    m_cd.num = m_e.numeric();
    const Delta d = delta(m_schemas[a.schema.v]);
    out.assign(s);
    out.apply(m_dels, m_adds);
    if (m_nnum)
        out.numeric().assign(d.num, d.num + m_nnum);
    return d;
}

void Successors::apply(StateView s, const ActionLabel& a, StateBuilder& out) { (void) apply_with_delta(s, a, out); }
}  // namespace mymyr
