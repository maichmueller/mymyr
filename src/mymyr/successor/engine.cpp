#include "mymyr/successor/detail/engine.hpp"

#include "mymyr/task/task.hpp"

#include <algorithm>

namespace mymyr::detail
{
Engine::Engine(const Task& task)
    : m_task(&task),
      m_c(&task.compiled()),
      m_atoms(&task.atoms()),
      m_slot_of(task.atoms().slot_table()),
      m_total(task.compiled().layout.total),
      m_vars(task.compiled().pattern_vars.data()),
      m_numc(&task.compiled().num),
      m_ow(task.compiled().ow)
{
    const plan::Compiled& C = *m_c;
    m_view.assign(static_cast<usize>(C.view.rows) * m_ow, 0);
    m_mark.assign(C.view.rows, 0);
    m_touched.assign(std::max<u32>(1, C.view.rows), 0);
    m_bind.assign(std::max<u32>(1, C.max_bind), ObjectId{0});
    m_dnw = bits::words_for(m_atoms->max_derived_slots());
    m_derived.assign(std::max<u32>(1, m_dnw), 0);
    m_drivers = !C.drivers.empty();
    m_driver_of.assign(C.kinds.size(), ~u32{0});
    m_true.resize(C.drivers.size());
    for (u32 d = 0; d < C.drivers.size(); ++d)
        m_driver_of[C.drivers[d]] = d;
}

void Engine::instantiate(ExecMatcher& x, const plan::Matcher& m)
{
    const u32 OW = m_ow;
    auto resolve = [&](plan::TableRef r) -> const u64*
    {
        if (r.is_view())
            return m_view.data() + static_cast<usize>(r.value()) * OW;
        return m_c->static_words.data() + r.value();
    };
    auto flip = [](bool neg) { return neg ? ~u64{0} : u64{0}; };
    x.plan = &m;
    x.unary.clear();
    for (const plan::Unary& u : m.unary)
        x.unary.push_back({resolve(u.ptr), flip(u.neg)});
    x.rows.clear();
    for (const plan::Row& r : m.rows)
        x.rows.push_back({resolve(r.base), r.src, flip(r.neg)});
    x.fc_out.clear();
    for (const plan::Edge& e : m.fc_out)
        x.fc_out.push_back({e.to, resolve(e.base), flip(e.neg)});
    x.fc_pre.clear();
    for (const plan::Row& r : m.fc_pre)
        x.fc_pre.push_back({resolve(r.base), r.src, flip(r.neg)});
    const usize steps = m.steps.size();
    x.cand.assign(std::max<usize>(1, steps) * OW, 0);
    x.dom1.assign(std::max<usize>(1, m.total) * OW, 0);
    x.fc_doms.assign(m.use_fc ? (steps + 1) * m.total * OW : 0, 0);
    x.bound.assign(std::max<u32>(1, m.total), 0);
    // the flat step table (every vector above is final: the pointers stay valid)
    x.steps.clear();
    for (usize d = 0; d < steps; ++d)
    {
        const plan::Step& st = m.steps[d];
        ExecStep e;
        e.param = st.param;
        e.rb = x.rows.data() + st.row_begin;
        e.re = x.rows.data() + st.row_end;
        e.cb = m.step_checks.data() + st.check_begin;
        e.ce = m.step_checks.data() + st.check_end;
        e.ncb = m.step_nchecks.data() + st.ncheck_begin;
        e.nce = m.step_nchecks.data() + st.ncheck_end;
        e.ub = x.unary.data() + m.unary_begin[st.param];
        e.ue = x.unary.data() + m.unary_begin[st.param + 1];
        e.dom0 = m.dom0.data() + static_cast<usize>(st.param) * OW;
        e.dom1 = x.dom1.data() + static_cast<usize>(st.param) * OW;
        e.cand = x.cand.data() + d * OW;
        x.steps.push_back(e);
    }
    x.nsteps = static_cast<u32>(steps);
    x.first_exist = m.first_exist;
    x.checks = m.checks.data();
    x.nchecks = m.nchecks.data();
}
}  // namespace mymyr::detail
