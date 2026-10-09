// Grounding of the delete relaxation (see relaxed_task.hpp): mimir's LiftedGrounder (delete-free fixpoint, then the
// statically applicable unrelaxed ground actions and axioms) and RelaxedPlanningGraph construction (one operator per
// ground action and conditional effect instance), over mymyr's lifted successor generator.

#include "mymyr/heuristics/relaxed_task.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace mymyr::heuristics
{
using namespace formalism;

namespace
{
using Clock = std::chrono::steady_clock;

constexpr u32 k_neg_bit = u32{1} << 31;  // raw literal entries: atom id | k_neg_bit for negative ones

/// Interning table of ground atoms (pred, args) -> dense id; only used while grounding.
class AtomTable
{
public:
    AtomTable(const TaskData& t, u32 max_arity) : m_t(t), m_stride(1 + std::max<u32>(1, max_arity)), m_slots(1024, 0), m_mask(1023) {}

    [[nodiscard]] u32 size() const noexcept { return m_count; }
    [[nodiscard]] const u32* record(u32 id) const noexcept { return m_rec.data() + static_cast<usize>(id) * m_stride; }
    [[nodiscard]] u32 pred(u32 id) const noexcept { return record(id)[0]; }

    u32 intern(u32 pred, const u32* args)
    {
        const u32 ar = m_t.predicates[pred].arity;
        const u64 h = hash_of(pred, args, ar);
        for (u64 j = h & m_mask;; j = (j + 1) & m_mask)
        {
            const u32 v = m_slots[j];
            if (v == 0)
            {
                const u32 id = m_count++;
                m_slots[j] = id + 1;
                const usize off = m_rec.size();
                m_rec.resize(off + m_stride, 0);
                m_rec[off] = pred;
                for (u32 i = 0; i < ar; ++i)
                    m_rec[off + 1 + i] = args[i];
                if (static_cast<u64>(m_count) * 10 > m_slots.size() * 7)
                    grow();
                return id;
            }
            if (same(v - 1, pred, args, ar))
                return v - 1;
        }
    }
    [[nodiscard]] u32 find(u32 pred, const u32* args) const
    {
        const u32 ar = m_t.predicates[pred].arity;
        for (u64 j = hash_of(pred, args, ar) & m_mask;; j = (j + 1) & m_mask)
        {
            const u32 v = m_slots[j];
            if (v == 0)
                return RelaxedTask::k_none;
            if (same(v - 1, pred, args, ar))
                return v - 1;
        }
    }

private:
    static u64 hash_of(u32 pred, const u32* args, u32 ar)
    {
        u64 h = hash::mix64(pred + 0x9E3779B97F4A7C15ULL);
        for (u32 i = 0; i < ar; ++i)
            h = hash::mix64(h ^ (args[i] + 0x632BE59BD9B4E019ULL));
        return h;
    }
    [[nodiscard]] bool same(u32 id, u32 pred, const u32* args, u32 ar) const
    {
        const u32* r = record(id);
        if (r[0] != pred)
            return false;
        for (u32 i = 0; i < ar; ++i)
            if (r[1 + i] != args[i])
                return false;
        return true;
    }
    void grow()
    {
        std::vector<u32> ns(m_slots.size() * 2, 0);
        const u64 mask = ns.size() - 1;
        for (u32 id = 0; id < m_count; ++id)
        {
            const u32* r = record(id);
            u64 j = hash_of(r[0], r + 1, m_t.predicates[r[0]].arity) & mask;
            while (ns[j])
                j = (j + 1) & mask;
            ns[j] = id + 1;
        }
        m_slots.swap(ns);
        m_mask = mask;
    }

    const TaskData& m_t;
    u32 m_stride;
    std::vector<u32> m_rec;
    std::vector<u32> m_slots;
    u64 m_mask;
    u32 m_count = 0;
};

/// The delete-free exploration task: negative literals and deletes dropped, derived predicates fluent, each axiom an
/// action adding its head (mimir's DeleteRelaxTranslator + delete-free axiom evaluation, as a plain task).
TaskData make_explore_data(const TaskData& T)
{
    TaskData R;
    R.domain_name = T.domain_name;
    R.problem_name = T.problem_name;
    R.requirements = T.requirements;
    R.chars = T.chars;
    R.type_ids = T.type_ids;
    R.object_ids = T.object_ids;
    R.terms = T.terms;
    R.params = T.params;
    R.types = T.types;
    R.objects = T.objects;
    R.predicates = T.predicates;
    for (Predicate& p : R.predicates)
        if (p.kind == PredKind::Derived)
            p.kind = PredKind::Fluent;
    R.static_init = T.static_init;
    R.fluent_init = T.fluent_init;
    auto positives = [&](Range r)
    {
        Range out{static_cast<u32>(R.literals.size()), 0};
        for (const Literal& l : TaskData::slice(T.literals, r))
            if (l.positive)
            {
                R.literals.push_back(l);
                ++out.count;
            }
        return out;
    };
    for (const Schema& s : T.schemas)
    {
        Schema ns;
        ns.name = s.name;
        ns.original_arity = s.original_arity;
        ns.params = s.params;
        ns.precondition.literals = positives(s.precondition.literals);
        std::vector<ConditionalEffect> ces;
        for (const ConditionalEffect& ce : T.effects_of(s))
        {
            ConditionalEffect n;
            n.extra_params = ce.extra_params;
            n.condition.literals = positives(ce.condition.literals);
            n.effects = positives(ce.effects);
            ces.push_back(n);
        }
        ns.effects = {static_cast<u32>(R.conditional_effects.size()), static_cast<u32>(ces.size())};
        R.conditional_effects.insert(R.conditional_effects.end(), ces.begin(), ces.end());
        R.schemas.push_back(ns);
    }
    const Str axiom_name = R.intern_string("axiom");
    for (const Axiom& x : T.axioms)
    {
        Schema ns;
        ns.name = axiom_name;
        ns.original_arity = x.params.count;
        ns.params = x.params;
        ns.precondition.literals = positives(x.body.literals);
        ConditionalEffect n;
        n.effects = {static_cast<u32>(R.literals.size()), 1};
        R.literals.push_back(x.head);
        ns.effects = {static_cast<u32>(R.conditional_effects.size()), 1};
        R.conditional_effects.push_back(n);
        R.schemas.push_back(ns);
    }
    return R;
}
}  // namespace

class Grounder
{
public:
    Grounder(const Task& task, const GroundingBudget& budget, RelaxedTask& out, const RelaxedTaskOptions& options = {})
        : T(task.data()), m_task(task), m_budget(budget), m_out(out), m_atoms(task.data(), max_arity(task.data())), m_n(T.num_objects()),
          m_options(options)
    {
    }

    bool run()
    {
        m_t0 = Clock::now();
        std::shared_ptr<const Task> explore;
        try
        {
            TaskOptions eo;
            eo.atoms = TaskOptions::Atoms::Lazy;
            eo.pilot_expansions = 0;
            eo.matching = m_task.options().matching;
            eo.fc_auto_free_params = m_task.options().fc_auto_free_params;
            explore = Task::create(make_explore_data(T), eo);
        }
        catch (const std::exception& e)
        {
            return fail(std::string("delete-free task: ") + e.what());
        }
        const WorkspaceLease lease = explore->workspace();
        Successors& succ = lease->successors();
        const AtomIndex& xatoms = explore->atoms();

        // 1. delete-free fixpoint in naive rounds (witness pruning is exact for the reached atoms)
        const State& s0 = explore->initial_state();
        std::vector<u64> S(s0.data(), s0.data() + s0.size_words()), N;
        bool stop = false;
        for (;;)
        {
            ++m_out.m_stats.rounds;
            N = S;
            bool grew = false;
            succ.prepare({S.data(), static_cast<u32>(S.size()), nullptr, 0});
            succ.generate<true>(
                [&](u32, const ObjectId*, const Delta& d) -> bool
                {
                    if (!tick())
                    {
                        stop = true;
                        return false;
                    }
                    for (SlotId a : d.add)
                    {
                        if (bits::word_of(a.v) >= N.size())
                            N.resize(bits::word_of(a.v) + 1, 0);
                        if (!bits::test(N.data(), static_cast<u32>(N.size()), a.v))
                        {
                            bits::set(N.data(), a.v);
                            grew = true;
                        }
                    }
                    return true;
                },
                true, false);
            if (stop)
                return fail(m_reason);
            if (!grew)
                break;
            S.swap(N);
        }

        // R: the reached atoms get the first atom ids
        bits::for_each(S.data(), static_cast<u32>(S.size()),
                       [&](u64 slot)
                       {
                           const u32* rec = xatoms.record(AtomKind::Fluent, static_cast<u32>(slot));
                           m_atoms.intern(rec[0], rec + 1);
                       });
        m_num_R = m_atoms.size();
        m_out.m_stats.reached_atoms = m_num_R;

        // 2 + 3. ground actions, ground axioms and their operators, from every relaxed-applicable binding
        m_costs_ok = true;
        try
        {
            m_costs = std::make_unique<ActionCosts>(m_task);
            m_costs_ok = m_costs->state_independent();
        }
        catch (const std::invalid_argument&)
        {
            m_costs_ok = false;
        }
        const u32 nschemas = static_cast<u32>(T.schemas.size());
        succ.prepare({S.data(), static_cast<u32>(S.size()), nullptr, 0});
        succ.generate<true>(
            [&](u32 s, const ObjectId* b, const Delta&) -> bool
            {
                if (!tick())
                {
                    stop = true;
                    return false;
                }
                if (s < nschemas)
                    ground_action(s, b);
                else
                    ground_axiom(s - nschemas, b);
                if (m_raw_npos.size() > m_budget.max_operators)
                {
                    stop = true;
                    m_reason = "more than " + std::to_string(m_budget.max_operators) + " relaxed operators";
                    return false;
                }
                return true;
            },
            false, false);
        if (stop)
            return fail(m_reason.empty() ? "budget" : m_reason);

        collect_goal();
        finish();
        m_out.m_stats.seconds = std::chrono::duration<double>(Clock::now() - m_t0).count();
        return true;
    }

    [[nodiscard]] const std::string& reason() const noexcept { return m_reason; }

private:
    static u32 max_arity(const TaskData& t)
    {
        u32 a = 0;
        for (const Predicate& p : t.predicates)
            a = std::max(a, p.arity);
        return a;
    }

    bool fail(const std::string& why)
    {
        m_reason = why;
        m_out.m_stats.over_budget = true;
        m_out.m_stats.reason = why;
        m_out.m_stats.seconds = std::chrono::duration<double>(Clock::now() - m_t0).count();
        return false;
    }

    /// Counts one enumerated binding; false once the budget is spent.
    bool tick()
    {
        ++m_out.m_stats.bindings;
        if (m_out.m_stats.bindings > m_budget.max_bindings)
        {
            m_reason = "more than " + std::to_string(m_budget.max_bindings) + " relaxed bindings";
            return false;
        }
        if ((m_out.m_stats.bindings & 4095) == 0 &&
            std::chrono::duration<double>(Clock::now() - m_t0).count() > m_budget.max_seconds)
        {
            m_reason = "grounding took more than " + std::to_string(m_budget.max_seconds) + " s";
            return false;
        }
        return true;
    }

    [[nodiscard]] PredKind kind(u32 p) const { return T.predicates[p].kind; }

    /// Object of term t under binding b.
    static u32 value(Term t, const ObjectId* b) { return is_object(t) ? term_object(t).v : b[term_parameter(t)].v; }

    bool static_holds(const Literal& l, const ObjectId* b) const
    {
        const plan::StaticRelation& R = m_task.compiled().statics[l.pred.v];
        const auto terms = T.terms_of(l);
        u64 k = 0;
        for (u32 i = 0; i < R.arity; ++i)
            k += R.position_table(i, m_n)[value(terms[i], b)];
        return R.contains(k);
    }

    u32 atom_of(const Literal& l, const ObjectId* b)
    {
        u32 args[64];
        const auto terms = T.terms_of(l);
        for (usize i = 0; i < terms.size(); ++i)
            args[i] = value(terms[i], b);
        return m_atoms.intern(l.pred.v, args);
    }

    /// As in mimir's is_statically_applicable: static literals hold, no fluent/derived atom required both true and false.
    /// Appends the fluent/derived entries to `entries` (negated derived atoms left out: they always cost 0).
    bool condition_entries(std::span<const Literal> lits, const ObjectId* b, std::vector<u32>& entries, bool check)
    {
        const usize first = entries.size();
        m_negd.clear();
        for (const Literal& l : lits)
        {
            const PredKind k = kind(l.pred.v);
            if (k == PredKind::Static)
            {
                if (check && static_holds(l, b) != l.positive)
                    return false;
                continue;
            }
            const u32 a = atom_of(l, b);
            if (k == PredKind::Derived && !l.positive)
            {
                m_negd.push_back(a);
                continue;
            }
            const u32 e = l.positive ? a : (a | k_neg_bit);
            // A condition is a set of literals in mimir: loki's normalized conditions (and with them mymyr's TaskData,
            // which reproduces mimir's export) repeat nullary literals, which mimir keeps once per condition.
            if (std::find(entries.begin() + static_cast<std::ptrdiff_t>(first), entries.end(), e) == entries.end())
                entries.push_back(e);
        }
        if (check)
        {
            for (usize i = first; i < entries.size(); ++i)
                if (entries[i] & k_neg_bit)
                    for (usize j = first; j < entries.size(); ++j)
                        if (entries[j] == (entries[i] & ~k_neg_bit))
                            return false;
            for (u32 a : m_negd)
                for (usize j = first; j < entries.size(); ++j)
                    if (entries[j] == a)
                        return false;
        }
        return true;
    }

    /// Candidate objects of the quantified parameters of conditional effect `ce` of schema `s`: mimir's
    /// StaticConsistencyGraph::compute_vertices over the effect condition's static literals, per parameter alone.
    const std::vector<std::vector<u32>>& candidates(u32 s, u32 cei, const ConditionalEffect& ce)
    {
        const u64 key = (static_cast<u64>(s) << 32) | cei;
        if (auto it = m_cands.find(key); it != m_cands.end())
            return it->second;
        const u32 arity = T.schemas[s].arity();
        std::vector<std::vector<u32>> out(ce.extra_params.count);
        for (u32 j = 0; j < ce.extra_params.count; ++j)
        {
            const u32 param = arity + j;
            for (u32 o = 0; o < m_n; ++o)
            {
                bool ok = true;
                for (const Literal& l : T.literals_of(ce.condition))
                {
                    if (kind(l.pred.v) != PredKind::Static)
                        continue;
                    const u32 ar = T.predicates[l.pred.v].arity;
                    if (ar < 1 || (!l.positive && ar != 1))
                        continue;
                    const auto terms = T.terms_of(l);
                    for (u32 i = 0; i < ar && ok; ++i)
                    {
                        u32 v;
                        if (is_object(terms[i]))
                            v = term_object(terms[i]).v;
                        else if (term_parameter(terms[i]) == param)
                            v = o;
                        else
                            continue;
                        const bool t = projection(l.pred.v, i)[v];
                        if (l.positive ? !t : t)
                            ok = false;
                    }
                    if (!ok)
                        break;
                }
                if (ok)
                    out[j].push_back(o);
            }
        }
        return m_cands.emplace(key, std::move(out)).first->second;
    }

    /// Objects occurring at position i of static predicate p's initial atoms.
    const std::vector<u8>& projection(u32 p, u32 i)
    {
        const u64 key = (static_cast<u64>(p) << 32) | i;
        if (auto it = m_proj.find(key); it != m_proj.end())
            return it->second;
        std::vector<u8> v(m_n, 0);
        for (const GroundAtom& a : T.static_init)
            if (a.pred.v == p)
                v[T.objects_of(a)[i].v] = 1;
        return m_proj.emplace(key, std::move(v)).first->second;
    }

    void push_op(u32 ga, const std::vector<u32>& pre, const std::vector<u32>& eff, bool unconditional)
    {
        u32 npos = 0;
        for (u32 e : pre)
            npos += (e & k_neg_bit) == 0;
        m_raw_pre.insert(m_raw_pre.end(), pre.begin(), pre.end());
        m_raw_pre_begin.push_back(static_cast<u32>(m_raw_pre.size()));
        m_raw_eff.insert(m_raw_eff.end(), eff.begin(), eff.end());
        m_raw_eff_begin.push_back(static_cast<u32>(m_raw_eff.size()));
        m_raw_npos.push_back(npos);
        m_raw_ga.push_back(ga);
        m_raw_uncond.push_back(unconditional ? 1 : 0);
    }

    void ground_action(u32 s, const ObjectId* b)
    {
        const Schema& sc = T.schemas[s];
        m_apre.clear();
        if (!condition_entries(T.literals_of(sc.precondition), b, m_apre, true))
            return;
        const u32 ga = static_cast<u32>(m_out.m_ga_schema.size());
        m_out.m_ga_schema.push_back(s);
        m_out.m_ga_bind.insert(m_out.m_ga_bind.end(), b, b + sc.arity());
        m_out.m_ga_bind_begin.push_back(static_cast<u32>(m_out.m_ga_bind.size()));
        u32 cost = 1;
        if (m_costs_ok && !m_costs->unit())
        {
            try
            {
                const f64 c = m_costs->cost(s, b);
                if (c == std::floor(c) && c < 2147483648.0)
                    cost = static_cast<u32>(c);
                else
                    m_costs_ok = false;
            }
            catch (const std::domain_error&)
            {
                m_costs_ok = false;
            }
        }
        m_out.m_ga_cost.push_back(cost);
        ++m_out.m_stats.ground_actions;

        const auto ces = T.effects_of(sc);
        for (u32 k = 0; k < ces.size(); ++k)
        {
            const ConditionalEffect& ce = ces[k];
            if (ce.effects.count == 0)
                continue;
            const u32 extra = ce.extra_params.count;
            if (extra == 0)
            {
                ce_instance(ga, ce, b);
                continue;
            }
            const auto& cand = candidates(s, k, ce);
            bool empty = false;
            for (const auto& c : cand)
                empty |= c.empty();
            if (empty)
                continue;
            m_full.assign(b, b + sc.arity());
            m_full.resize(sc.arity() + extra);
            std::vector<u32> idx(extra, 0);
            for (bool more = true; more;)
            {
                for (u32 j = 0; j < extra; ++j)
                    m_full[sc.arity() + j] = ObjectId{cand[j][idx[j]]};
                ce_instance(ga, ce, m_full.data());
                // odometer over the candidate lists, last parameter fastest
                more = false;
                for (u32 j = extra; j-- > 0;)
                {
                    if (++idx[j] < cand[j].size())
                    {
                        more = true;
                        break;
                    }
                    idx[j] = 0;
                }
            }
        }
    }

    void ce_instance(u32 ga, const ConditionalEffect& ce, const ObjectId* b)
    {
        m_pre = m_apre;
        condition_entries(T.literals_of(ce.condition), b, m_pre, false);  // static ce literals ignored, as in mimir's RPG
        m_eff.clear();
        for (const Literal& l : TaskData::slice(T.literals, ce.effects))
        {
            const u32 a = atom_of(l, b);
            m_eff.push_back(l.positive ? a : (a | k_neg_bit));
        }
        push_op(ga, m_pre, m_eff, T.literals_of(ce.condition).empty());
    }

    void ground_axiom(u32 x, const ObjectId* b)
    {
        const Axiom& ax = T.axioms[x];
        m_pre.clear();
        if (!condition_entries(T.literals_of(ax.body), b, m_pre, true))
            return;
        m_eff.assign(1, atom_of(ax.head, b));
        push_op(RelaxedTask::k_none, m_pre, m_eff, true);
        ++m_out.m_stats.ground_axioms;
    }

    void collect_goal()
    {
        const ObjectId* none = nullptr;
        for (const Literal& l : T.literals_of(T.goal))
        {
            const PredKind k = kind(l.pred.v);
            if (k == PredKind::Static || (k == PredKind::Derived && !l.positive))
                continue;
            const u32 a = atom_of(l, none);
            const u32 e = l.positive ? a : (a | k_neg_bit);
            if (std::find(m_goal_raw.begin(), m_goal_raw.end(), e) == m_goal_raw.end())
                m_goal_raw.push_back(e);
        }
    }

    void finish()
    {
        RelaxedTask& R = m_out;
        const u32 A = m_atoms.size();
        const u32 nops = static_cast<u32>(m_raw_npos.size());
        // "false" propositions: fluent atoms of R used negatively by an operator or the goal
        std::vector<u32> neg_of(A, RelaxedTask::k_none);
        u32 P = A;
        auto need = [&](u32 e)
        {
            const u32 a = e & ~k_neg_bit;
            if ((e & k_neg_bit) && a < m_num_R && neg_of[a] == RelaxedTask::k_none)
                neg_of[a] = P++;
        };
        for (u32 e : m_raw_pre)
            need(e);
        for (u32 e : m_goal_raw)
            need(e);
        R.m_prop_negative.assign(P, 0);
        for (u32 p = A; p < P; ++p)
            R.m_prop_negative[p] = 1;

        // atoms some operator can add (or R): positive preconditions outside can never hold
        std::vector<u8> addable(A, 0);
        for (u32 a = 0; a < m_num_R; ++a)
            addable[a] = 1;
        for (u32 e : m_raw_eff)
            if (!(e & k_neg_bit))
                addable[e] = 1;

        for (u32 o = 0; o < nops; ++o)
        {
            bool viable = true;
            const usize pb = m_raw_pre_begin[o], pe = m_raw_pre_begin[o + 1];
            for (usize i = pb; i < pe && viable; ++i)
                if (!(m_raw_pre[i] & k_neg_bit) && !addable[m_raw_pre[i]])
                    viable = false;
            if (!viable && !m_options.keep_unreachable_operators)
                continue;
            const usize eff0 = R.m_eff.size();
            for (usize i = m_raw_eff_begin[o]; i < m_raw_eff_begin[o + 1]; ++i)
            {
                const u32 e = m_raw_eff[i];
                if (!(e & k_neg_bit))
                    R.m_eff.push_back(e);
                else if (neg_of[e & ~k_neg_bit] != RelaxedTask::k_none)
                    R.m_eff.push_back(neg_of[e & ~k_neg_bit]);
            }
            if (R.m_eff.size() == eff0)
                continue;
            R.m_eff_begin.push_back(static_cast<u32>(R.m_eff.size()));
            for (usize i = m_raw_eff_begin[o]; i < m_raw_eff_begin[o + 1]; ++i)
            {
                const u32 e = m_raw_eff[i];
                if ((e & k_neg_bit) && std::find(R.m_del.begin() + R.m_del_begin.back(), R.m_del.end(), e & ~k_neg_bit) == R.m_del.end())
                    R.m_del.push_back(e & ~k_neg_bit);
            }
            R.m_del_begin.push_back(static_cast<u32>(R.m_del.size()));
            R.m_uncond.push_back(m_raw_uncond[o]);
            u32 npos = 0;
            for (usize i = pb; i < pe; ++i)
            {
                const u32 e = m_raw_pre[i];
                if (!(e & k_neg_bit))
                {
                    R.m_pre.push_back(e);
                    ++npos;
                }
                else if ((e & ~k_neg_bit) < m_num_R)
                    R.m_pre.push_back(neg_of[e & ~k_neg_bit]);
                // a negated atom outside R never holds in a covered state: always satisfied, left out
            }
            R.m_pre_begin.push_back(static_cast<u32>(R.m_pre.size()));
            R.m_npos.push_back(npos);
            R.m_ga.push_back(m_raw_ga[o]);
        }
        const u32 O = static_cast<u32>(R.m_npos.size());
        R.m_stats.operators = O;
        R.m_stats.propositions = P;
        // prop -> operators (with multiplicity)
        R.m_pre_of_begin.assign(P + 1, 0);
        for (u32 p : R.m_pre)
            ++R.m_pre_of_begin[p + 1];
        for (u32 p = 0; p < P; ++p)
            R.m_pre_of_begin[p + 1] += R.m_pre_of_begin[p];
        R.m_pre_of.resize(R.m_pre.size());
        {
            std::vector<u32> fill(R.m_pre_of_begin.begin(), R.m_pre_of_begin.end() - 1);
            for (u32 o = 0; o < O; ++o)
                for (u32 i = R.m_pre_begin[o]; i < R.m_pre_begin[o + 1]; ++i)
                    R.m_pre_of[fill[R.m_pre[i]]++] = o;
        }
        for (u32 o = 0; o < O; ++o)
            if (R.m_npos[o] == 0)
                R.m_zero_ops.push_back(o);

        // goal
        for (u32 e : m_goal_raw)
        {
            const u32 a = e & ~k_neg_bit;
            if (!(e & k_neg_bit))
            {
                R.m_goal.push_back(a);
                if (!addable[a])
                    R.m_goal_unreachable = true;
            }
            else if (a < m_num_R)
                R.m_goal.push_back(neg_of[a]);
        }

        // task fluent and derived atoms of R by canonical id
        const CanonicalLayout& L = m_task.atoms().layout();
        R.m_prop_atom.assign(P, ~CanonicalAtom{0});
        for (u32 a = 0; a < A; ++a)
        {
            const u32* rec = m_atoms.record(a);
            if (kind(rec[0]) == PredKind::Static || L.offset[rec[0]] == CanonicalLayout::k_none)
                continue;
            const CanonicalAtom c = L.encode(rec[0], rec + 1);
            if (c >= (kind(rec[0]) == PredKind::Fluent ? L.fluent_count : L.total))
                continue;
            R.m_prop_atom[a] = c;
            if (neg_of[a] != RelaxedTask::k_none)
                R.m_prop_atom[neg_of[a]] = c;
        }
        u64 cap = 64;
        while (cap < 2 * static_cast<u64>(m_num_R) + 2)
            cap *= 2;
        R.m_cid_keys.assign(cap, 0);
        R.m_cid_pos.assign(cap, RelaxedTask::k_none);
        R.m_cid_neg.assign(cap, RelaxedTask::k_none);
        R.m_cid_mask = cap - 1;
        for (u32 a = 0; a < m_num_R; ++a)
        {
            const u32* rec = m_atoms.record(a);
            if (kind(rec[0]) == PredKind::Static)
                continue;
            const CanonicalAtom c = L.encode(rec[0], rec + 1);
            if (c >= (kind(rec[0]) == PredKind::Fluent ? L.fluent_count : L.total))
                continue;
            for (u64 j = hash::mix64(c) & R.m_cid_mask;; j = (j + 1) & R.m_cid_mask)
            {
                if (R.m_cid_keys[j] == 0)
                {
                    R.m_cid_keys[j] = c + 1;
                    R.m_cid_pos[j] = a;
                    R.m_cid_neg[j] = neg_of[a];
                    break;
                }
            }
        }

        // ground actions: lookup table and real costs
        R.m_real_ok = m_costs_ok;
        const u32 G = static_cast<u32>(R.m_ga_schema.size());
        cap = 64;
        while (cap < 2 * static_cast<u64>(G) + 2)
            cap *= 2;
        R.m_ga_slots.assign(cap, 0);
        R.m_ga_mask = cap - 1;
        for (u32 g = 0; g < G; ++g)
        {
            const ActionLabel l = R.ground_action_label(g);
            u64 h = hash::mix64(l.schema.v + 0x51ED27A1ULL);
            for (ObjectId o : l.binding)
                h = hash::mix64(h ^ (o.v + 0x2545F4914F6CDD1DULL));
            for (u64 j = h & R.m_ga_mask;; j = (j + 1) & R.m_ga_mask)
                if (R.m_ga_slots[j] == 0)
                {
                    R.m_ga_slots[j] = g + 1;
                    break;
                }
        }
    }

    const TaskData& T;
    const Task& m_task;
    GroundingBudget m_budget;
    RelaxedTask& m_out;
    AtomTable m_atoms;
    u32 m_n;
    u32 m_num_R = 0;
    Clock::time_point m_t0;
    std::string m_reason;
    std::unique_ptr<ActionCosts> m_costs;
    bool m_costs_ok = false;
    std::map<u64, std::vector<std::vector<u32>>> m_cands;
    std::map<u64, std::vector<u8>> m_proj;
    // raw operators
    std::vector<u32> m_raw_pre_begin{0}, m_raw_pre, m_raw_eff_begin{0}, m_raw_eff, m_raw_npos, m_raw_ga;
    std::vector<u8> m_raw_uncond;
    std::vector<u32> m_goal_raw;
    // scratch
    std::vector<u32> m_apre, m_pre, m_eff, m_negd;
    std::vector<ObjectId> m_full;
    RelaxedTaskOptions m_options;
};

std::shared_ptr<const RelaxedTask> RelaxedTask::build(const Task& task, const GroundingBudget& budget, GroundingStats* stats)
{
    return build(task, budget, stats, RelaxedTaskOptions{});
}

std::shared_ptr<const RelaxedTask> RelaxedTask::build(const Task& task, const GroundingBudget& budget, GroundingStats* stats,
                                                      const RelaxedTaskOptions& options)
{
    auto out = std::make_shared<RelaxedTask>(Private{}, task);
    out->m_ga_bind_begin.push_back(0);
    Grounder g(task, budget, *out, options);
    const bool ok = g.run();
    if (stats)
        *stats = out->m_stats;
    if (!ok)
        return nullptr;
    return out;
}

std::pair<u32, u32> RelaxedTask::fluent_props(CanonicalAtom cid) const noexcept
{
    for (u64 j = hash::mix64(cid) & m_cid_mask;; j = (j + 1) & m_cid_mask)
    {
        const u64 k = m_cid_keys[j];
        if (k == 0)
            return {k_none, k_none};
        if (k == cid + 1)
            return {m_cid_pos[j], m_cid_neg[j]};
    }
}

ActionLabel RelaxedTask::ground_action_label(u32 ga) const noexcept
{
    const u32 b = m_ga_bind_begin[ga], e = m_ga_bind_begin[ga + 1];
    return {SchemaId{m_ga_schema[ga]}, {m_ga_bind.data() + b, e - b}};
}

u32 RelaxedTask::find_ground_action(const ActionLabel& a) const noexcept
{
    u64 h = hash::mix64(a.schema.v + 0x51ED27A1ULL);
    for (ObjectId o : a.binding)
        h = hash::mix64(h ^ (o.v + 0x2545F4914F6CDD1DULL));
    for (u64 j = h & m_ga_mask;; j = (j + 1) & m_ga_mask)
    {
        const u32 v = m_ga_slots[j];
        if (v == 0)
            return k_none;
        const ActionLabel l = ground_action_label(v - 1);
        if (l.schema == a.schema && l.binding.size() == a.binding.size() &&
            std::equal(l.binding.begin(), l.binding.end(), a.binding.begin()))
            return v - 1;
    }
}
}  // namespace mymyr::heuristics
