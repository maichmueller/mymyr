// The device's choice between its two matchers (rl/task_arrays.hpp device_search_costs): a model of the work of
// cuda/src/lifted_device.cuh's run_fixed and run_fc, replayed on the CPU's engine over sample states. The replays
// follow the device loops step by step (candidate order, witness jumps, the compact domain stack of forward checking)
// and add the model's weights for what each step reads and writes; the searches' results are not used.

#include "mymyr/rl/task_arrays.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace mymyr::rl
{
namespace
{
// weights of the model (units of one word read or written in registers / L1): a view row is an index load and a
// dependent row load, a literal check a pattern key, a slot lookup and a bit test
constexpr u64 k_w_cand = 2;   // take the next candidate of a level
constexpr u64 k_w_check = 6;  // one literal check
constexpr u64 k_w_row = 3;    // resolve a view or static row (plus its OW words)
constexpr u64 k_w_emit = 4;   // hand a binding to the emitter
/// A state's replay of one matcher stops counting past this much work (a search the device would not finish soon; the
/// suites' largest take some 5000 per state): the probe stays within milliseconds per matcher.
constexpr u64 k_cap = u64{1} << 17;

struct Replay
{
    detail::Engine& e;
    u32 OW;

    /// The device's domain(): dom0 and the per-state unary constraints of `param` into out; false if empty.
    bool domain(const detail::ExecMatcher& x, u32 param, u64* out, u64& c) const
    {
        const plan::Matcher& m = *x.plan;
        const u64* d0 = m.dom0.data() + u64{param} * OW;
        const detail::ExecUnary* ub = x.unary.data() + m.unary_begin[param];
        const detail::ExecUnary* ue = x.unary.data() + m.unary_begin[param + 1];
        c += OW * (1 + static_cast<u64>(ue - ub) * k_w_row);
        u64 any = 0;
        for (u32 w = 0; w < OW; ++w)
        {
            u64 a = d0[w];
            for (const detail::ExecUnary* u = ub; u != ue; ++u)
                a &= u->ptr[w] ^ u->flip;
            out[w] = a;
            any |= a;
        }
        return any != 0;
    }

    bool prologue(const detail::ExecMatcher& x, u64& c) const
    {
        const plan::Matcher& m = *x.plan;
        c += 1;
        if (m.never)
            return false;
        for (const plan::Check& ck : m.pre_checks)
        {
            c += k_w_check;
            if (!e.holds(ck))
                return false;
        }
        return true;
    }

    /// run_fixed's work.
    u64 fixed(detail::ExecMatcher& x) const
    {
        u64 c = 0;
        if (!prologue(x, c))
            return c;
        const u32 nd = x.nsteps;
        if (nd == 0)
            return c + k_w_emit;
        const plan::Matcher& m = *x.plan;
        std::vector<u64> dom1(u64{nd} * OW), cd(u64{nd} * OW);
        for (u32 d = 0; d < nd; ++d)
            if (!domain(x, m.steps[d].param, dom1.data() + u64{d} * OW, c))
                return c;
        ObjectId* bind = e.bind();
        auto fill = [&](u32 d)
        {
            const detail::ExecStep& st = x.steps[d];
            c += OW + static_cast<u64>(st.re - st.rb) * (k_w_row + OW);
            for (u32 w = 0; w < OW; ++w)
            {
                u64 a = dom1[u64{d} * OW + w];
                for (const detail::ExecRow* r = st.rb; r != st.re; ++r)
                    a &= r->base[static_cast<usize>(bind[r->src].v) * OW + w] ^ r->flip;
                cd[u64{d} * OW + w] = a;
            }
        };
        const int fe = static_cast<int>(m.first_exist);
        const int last = static_cast<int>(nd) - 1;
        int d = 0;
        bool wit = fe == 0;
        fill(0);
        while (d >= 0 && c < k_cap)
        {
            u64* cw = cd.data() + static_cast<u64>(d) * OW;
            u32 w = 0;
            while (w < OW && cw[w] == 0)
                ++w;
            c += 1 + w;
            if (w == OW)
            {
                --d;
                if (d < fe)
                    wit = false;
                continue;
            }
            const u32 o = w * 64 + static_cast<u32>(bits::ctz64(cw[w]));
            cw[w] &= cw[w] - 1;
            c += k_w_cand;
            const detail::ExecStep& st = x.steps[static_cast<u32>(d)];
            bind[st.param] = ObjectId{o};
            bool ok = true;
            for (const u32* k = st.cb; k != st.ce && ok; ++k)
            {
                c += k_w_check;
                ok = e.holds(x.checks[*k]);
            }
            if (!ok)
                continue;
            if (d == last)
            {
                c += k_w_emit;
                if (wit)
                {
                    d = fe - 1;
                    wit = false;
                }
                continue;
            }
            ++d;
            if (d == fe)
                wit = true;
            fill(static_cast<u32>(d));
        }
        return c;
    }

    /// run_fc's work (a level copies the domains of the parameters not bound above it).
    u64 fc(detail::ExecMatcher& x) const
    {
        u64 c = 0;
        if (!prologue(x, c))
            return c;
        const plan::Matcher& m = *x.plan;
        const u32 nd = x.nsteps;
        if (nd == 0)
            return c + k_w_emit;
        const u32 total = m.total;
        std::vector<u64> D(u64{nd + 1} * total * OW, 0);
        auto dom = [&](u32 l, u32 v) { return D.data() + (u64{l} * total + v) * OW; };
        for (u32 d = 0; d < nd; ++d)
            if (!domain(x, m.steps[d].param, dom(0, m.steps[d].param), c))
                return c;
        ObjectId* bind = e.bind();
        for (usize i = 0; i < x.fc_pre.size(); ++i)
        {
            const detail::ExecRow& r = x.fc_pre[i];
            u64* dt = dom(0, m.fc_pre_to[i]);
            c += k_w_row + OW;
            u64 any = 0;
            for (u32 w = 0; w < OW; ++w)
                any |= (dt[w] &= r.base[static_cast<usize>(bind[r.src].v) * OW + w] ^ r.flip);
            if (!any)
                return c;
        }
        const u32 nfree = static_cast<u32>(m.free_params.size());
        u64 bound = 0;
        std::vector<u32> pick(nd);
        std::vector<u64> rem(u64{nd} * OW);
        bool witness = false;
        int wlev = -1;
        auto enter = [&](u32 l)
        {
            bool any_relevant = false;
            for (u32 i = 0; i < nfree && !any_relevant; ++i)
                any_relevant = !((bound >> m.free_params[i]) & 1) && m.relevant[m.free_params[i]];
            if (!any_relevant && !witness)
            {
                witness = true;
                wlev = static_cast<int>(l);
            }
            c += nfree;
            u32 best = 0, best_cnt = ~0u;
            for (u32 v : m.free_params)
            {
                if (((bound >> v) & 1) || (any_relevant && !m.relevant[v]))
                    continue;
                c += OW;
                u32 cnt = 0;
                for (u32 w = 0; w < OW; ++w)
                    cnt += static_cast<u32>(bits::popcount64(dom(l, v)[w]));
                if (cnt < best_cnt)
                    best = v, best_cnt = cnt;
            }
            pick[l] = best;
            c += OW;
            std::copy_n(dom(l, best), OW, rem.data() + u64{l} * OW);
            bound |= u64{1} << best;
        };
        int level = 0;
        enter(0);
        while (level >= 0 && c < k_cap)
        {
            const u32 p = pick[static_cast<u32>(level)];
            u64* rw = rem.data() + static_cast<u64>(level) * OW;
            u32 w = 0;
            while (w < OW && rw[w] == 0)
                ++w;
            c += 1 + w;
            if (w == OW)
            {
                bound &= ~(u64{1} << p);
                --level;
                if (level < wlev)
                {
                    witness = false;
                    wlev = -1;
                }
                continue;
            }
            const u32 o = w * 64 + static_cast<u32>(bits::ctz64(rw[w]));
            rw[w] &= rw[w] - 1;
            c += k_w_cand;
            bind[p] = ObjectId{o};
            bool ok = true;
            for (u32 k = m.fc_checks_begin[p]; k < m.fc_checks_begin[p + 1] && ok; ++k)
            {
                const u32 ck = m.fc_checks[k];
                bool all = true;
                c += m.check_vars_begin[ck + 1] - m.check_vars_begin[ck];
                for (u32 j = m.check_vars_begin[ck]; j < m.check_vars_begin[ck + 1]; ++j)
                    all = all && ((bound >> m.check_vars[j]) & 1);
                if (all)
                {
                    c += k_w_check;
                    ok = e.holds(m.checks[ck]);
                }
            }
            if (!ok)
                continue;
            const u32 l1 = static_cast<u32>(level) + 1;
            c += total;
            for (u32 v = 0; v < total; ++v)
                if (!((bound >> v) & 1))
                {
                    c += 2 * OW;  // a local-memory copy: read and write
                    std::copy_n(dom(l1 - 1, v), OW, dom(l1, v));
                }
            for (u32 k = m.fc_out_begin[p]; k < m.fc_out_begin[p + 1] && ok; ++k)
            {
                const detail::ExecEdge& ed = x.fc_out[k];
                c += 1;
                if ((bound >> ed.to) & 1)
                    continue;
                c += k_w_row + OW;
                u64 any = 0;
                u64* dt = dom(l1, ed.to);
                for (u32 i = 0; i < OW; ++i)
                    any |= (dt[i] &= ed.base[static_cast<usize>(o) * OW + i] ^ ed.flip);
                ok = any != 0;
            }
            if (!ok)
                continue;
            if (l1 == nfree)
            {
                c += k_w_emit;
                if (witness)
                {
                    for (int l = level; l >= wlev; --l)
                        bound &= ~(u64{1} << pick[static_cast<u32>(l)]);
                    level = wlev - 1;
                    witness = false;
                    wlev = -1;
                }
                continue;
            }
            ++level;
            enter(static_cast<u32>(level));
        }
        return c;
    }
};

/// Whether the device may run the matcher at all (its deep kernels bind at most k_deep_matcher_depth parameters).
bool probed(const plan::Matcher& m)
{
    return m.total <= dev::k_deep_matcher_depth && m.steps.size() <= dev::k_deep_matcher_depth && m.first_free == 0 &&
           m.npre.empty() && m.nchecks.empty();
}
}  // namespace

DeviceSearchCosts device_search_costs(const Task& task)
{
    const plan::Compiled& C = task.compiled();
    DeviceSearchCosts out;
    out.schemas.assign(C.schemas.size(), {});
    for (const plan::Stratum& st : C.strata)
        out.axioms.resize(out.axioms.size() + st.axioms.size());
    if (task.numeric_slots() > 0)
        return out;  // no device kernels for numeric tasks
    // a workspace of its own: the caller's (Task::workspace) may hold a prepared state
    const auto ws = std::make_unique<Workspace>(task);
    Successors& succ = ws->successors();
    detail::Engine& e = succ.engine();
    // sample states: the initial state and, in frozen mode, its successors in canonical order (under lazy slots the
    // generation, and the axioms, would intern atoms: the probe must not change the task)
    const bool frozen = task.atoms().mode() == AtomMode::Frozen;
    std::vector<State> states{task.initial_state()};
    if (frozen)
    {
        std::vector<Action> acts = succ.applicable_actions(states[0]);
        for (usize i = 0; i < acts.size() && states.size() < k_probe_states; ++i)
            states.push_back(succ.apply(states[0], acts[i].label()));
    }
    out.states = static_cast<u32>(states.size());
    std::vector<detail::ExecMatcher> xs;
    std::vector<SearchCost*> into;
    for (usize s = 0; s < C.schemas.size(); ++s)
        for (u32 wi = 0; wi < 2; ++wi)
            if (probed(C.schemas[s].pre[wi]))
            {
                xs.emplace_back();
                e.instantiate(xs.back(), C.schemas[s].pre[wi]);
                into.push_back(&out.schemas[s][wi]);
            }
    usize ai = 0;
    for (const plan::Stratum& st : C.strata)
        for (const plan::Axiom& a : st.axioms)
        {
            if (probed(a.body))
            {
                xs.emplace_back();
                e.instantiate(xs.back(), a.body);
                into.push_back(&out.axioms[ai]);
            }
            ++ai;
        }
    const Replay r{e, C.ow};
    for (const State& s : states)
    {
        if (frozen)
            succ.prepare(s);  // the view and the derived atoms
        else
        {
            e.set_state(s.data(), s.size_words());
            e.build_view();
        }
        for (usize i = 0; i < xs.size(); ++i)
        {
            into[i]->fixed += r.fixed(xs[i]);
            into[i]->fc += r.fc(xs[i]);
        }
    }
    return out;
}
}  // namespace mymyr::rl
