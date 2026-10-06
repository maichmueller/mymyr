// Relaxed reachability (reachability/relaxed_reachability.hpp): rule compilation from the task and the semi-naive
// fixpoint over reach::Plan matchers.

#include "program.hpp"

#include "mymyr/task/task.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <stdexcept>

namespace mymyr::reachability
{
using namespace formalism;
using detail::KeyPattern;
using detail::Program;
using detail::Rule;
using detail::TableData;
using reach::k_none;
using reach::Lit;
using reach::LitType;

namespace
{
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

KeyPattern key_pattern(const CanonicalLayout& L, u32 pred, const std::vector<i32>& terms)
{
    KeyPattern k;
    k.pred = pred;
    k.terms = terms;
    if (L.offset[pred] == CanonicalLayout::k_none || L.size[pred] == 0)
    {
        k.never = true;
        return k;
    }
    k.base = L.offset[pred];
    for (u32 i = 0; i < terms.size(); ++i)
    {
        const u64* rs = L.position_table(pred, i);
        if (reach::is_var(terms[i]))
            k.vars.push_back({static_cast<u32>(terms[i]), rs});
        else
        {
            const u64 x = rs[reach::term_obj(terms[i])];
            if (x >= CanonicalLayout::k_outside)
                k.never = true;
            k.base += x;
        }
    }
    return k;
}

std::vector<i32> terms_of(const TaskData& T, const Literal& l)
{
    const auto t = T.terms_of(l);
    return {t.begin(), t.end()};
}

/// Adds the literals of one condition to a rule body (as in mimir's RuleBuilder reading).
void add_condition(const TaskData& T, const reach::StaticTables& st, const Options& o, std::span<const Literal> lits, Rule& r)
{
    for (const Literal& l : lits)
    {
        const PredKind k = T.predicates[l.pred.v].kind;
        Lit x;
        x.pred = l.pred.v;
        x.positive = l.positive;
        x.terms = terms_of(T, l);
        if (k == PredKind::Static)
        {
            if (!l.positive && !o.enforce_negative_static)
                continue;
            x.type = st.is_equality(l.pred.v) ? LitType::Eq : LitType::Static;
            r.body.push_back(std::move(x));
            continue;
        }
        if (!l.positive)
            continue;  // the delete relaxation
        x.type = LitType::Rel;
        x.slot = static_cast<u32>(r.rel.size());
        r.rel.push_back(key_pattern(st.compiled().layout, x.pred, x.terms));
        r.body.push_back(std::move(x));
    }
}

std::vector<u32> types_of(const TaskData& T, const Parameter& p)
{
    std::vector<u32> v;
    for (TypeId t : TaskData::slice(T.type_ids, p.types))
        v.push_back(t.v);
    return v;
}

void compile_rule(reach::StaticTables& st, Rule& r, const std::vector<std::vector<u32>>& declared)
{
    const u32 V = r.num_vars;
    // variables mentioned by the head or the body range over their declared type when no positive literal binds them
    std::vector<u8> mentioned(V, 0), head(V, 0);
    for (i32 t : r.head.terms)
        if (reach::is_var(t))
            mentioned[t] = head[t] = 1;
    for (const Lit& l : r.body)
        for (i32 t : l.terms)
            if (reach::is_var(t))
                mentioned[t] = 1;
    reach::CompileSpec spec;
    spec.num_vars = V;
    spec.relevant = head;
    spec.head = head;
    spec.var_types.resize(V);
    for (u32 v = 0; v < V; ++v)
        if (mentioned[v] && v < declared.size())
            spec.var_types[v] = declared[v];
    const u32 S = static_cast<u32>(r.rel.size());
    // Slot S is the head, negated and read from Known (every atom reached so far, the current round's included): a
    // binding whose head is already known is cut by the rows as soon as the head's variables are bound.
    spec.slot_mode.assign(S + 1, 1);
    spec.slot_mode[S] = 2;
    spec.prebound.assign(V, 0);
    std::vector<Lit> with_head = r.body;
    if (!r.head.never)
    {
        Lit h;
        h.type = LitType::Rel;
        h.positive = false;
        h.pred = r.head.pred;
        h.slot = S;
        h.terms = r.head.terms;
        with_head.push_back(std::move(h));
    }
    r.plans.push_back(reach::compile(st, with_head, spec));
    for (u32 a = 0; a < S; ++a)
    {
        reach::CompileSpec sa = spec;
        for (u32 s = 0; s < S; ++s)
            sa.slot_mode[s] = s < a ? 0 : 1;
        std::vector<Lit> body;
        for (const Lit& l : with_head)
        {
            if (l.type == LitType::Rel && l.positive && l.slot == a)
            {
                for (i32 t : l.terms)
                    if (reach::is_var(t))
                        sa.prebound[t] = 1;
                continue;  // the anchor: unified with the new atom
            }
            body.push_back(l);
        }
        // the anchor literal's own consistency (repeated variables, constants) is checked by the unification
        r.plans.push_back(reach::compile(st, body, sa));
    }
    // the seed plan (Rule::seed_slot): the relation literal of largest arity, if at least 3
    u32 best = k_none, best_arity = 2;
    for (const Lit& l : r.body)
        if (l.type == LitType::Rel && l.terms.size() > best_arity)
            best = l.slot, best_arity = static_cast<u32>(l.terms.size());
    if (best != k_none)
    {
        reach::CompileSpec sa = spec;
        std::vector<Lit> body;
        for (const Lit& l : with_head)
        {
            if (l.type == LitType::Rel && l.positive && l.slot == best)
            {
                for (i32 t : l.terms)
                    if (reach::is_var(t))
                        sa.prebound[t] = 1;
                continue;
            }
            body.push_back(l);
        }
        r.plans.push_back(reach::compile(st, body, sa));
        r.seed_slot = best;
    }
}

/// The anchor of rule r at relation slot s (see Anchor; id k_none: no dedup).
detail::Anchor make_anchor(const Rule& r, u32 ri, u32 s, u32 id)
{
    detail::Anchor an;
    an.rule = ri;
    an.slot = s;
    for (u32 i = 0; i < r.body.size(); ++i)
        if (r.body[i].type == LitType::Rel && r.body[i].slot == s)
            an.lit = i;
    if (id == k_none)
        return an;
    an.id = id;
    std::vector<u8> elsewhere(r.num_vars, 0);
    for (i32 t : r.head.terms)
        if (reach::is_var(t))
            elsewhere[t] = 1;
    for (u32 i = 0; i < r.body.size(); ++i)
    {
        const Lit& l = r.body[i];
        if (i == an.lit)
            continue;
        // a static or equality literal over one variable and constants (a type, say) narrows that variable's domain
        // (reach::compile folds it into dom0, which first_run tests)
        const bool domain_only = (l.type == LitType::Static || l.type == LitType::Eq) &&
                                 std::count_if(l.terms.begin(), l.terms.end(), reach::is_var) == 1 &&
                                 (l.terms.size() == 1 || (l.terms.size() == 2 && l.terms[0] != l.terms[1]));
        if (domain_only)
            continue;
        for (i32 t : l.terms)
            if (reach::is_var(t))
                elsewhere[t] = 1;
    }
    for (i32 t : r.body[an.lit].terms)
    {
        if (!reach::is_var(t))
            continue;
        auto& list = elsewhere[t] ? an.key_vars : an.free_vars;
        if (std::find(list.begin(), list.end(), static_cast<u32>(t)) == list.end())
            list.push_back(static_cast<u32>(t));
    }
    an.dedup = !an.free_vars.empty();
    return an;
}

std::shared_ptr<Program> build_program(const Task& task, const Options& o, Statistics& stats)
{
    static std::atomic<u64> next_id{1};
    auto p = std::make_shared<Program>();
    p->id = next_id.fetch_add(1, std::memory_order_relaxed);
    p->task = &task;
    p->options = o;
    p->st = std::make_unique<reach::StaticTables>(task);
    const TaskData& T = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    p->total = L.total;
    p->fluent_count = L.fluent_count;
    p->anchors.resize(T.predicates.size());
    p->seed_anchors.resize(T.predicates.size());
    for (u32 q = 0; q < T.predicates.size(); ++q)
        if (T.predicates[q].kind != PredKind::Static && L.offset[q] != CanonicalLayout::k_none && L.size[q] > 0)
            p->relations.push_back(q);

    auto declared_of = [&](std::initializer_list<Range> ranges)
    {
        std::vector<std::vector<u32>> d;
        for (Range r : ranges)
            for (const Parameter& par : TaskData::slice(T.params, r))
                d.push_back(types_of(T, par));
        return d;
    };
    for (u32 si = 0; si < T.schemas.size(); ++si)
    {
        const Schema& s = T.schemas[si];
        for (const ConditionalEffect& ce : T.effects_of(s))
        {
            for (const Literal& e : TaskData::slice(T.literals, ce.effects))
            {
                if (!e.positive)
                    continue;
                ++stats.rules;
                Rule r;
                r.name = std::string(T.str(s.name));
                r.schema = si;
                r.num_vars = s.arity() + ce.extra_params.count;
                r.head = key_pattern(L, e.pred.v, terms_of(T, e));
                add_condition(T, *p->st, o, T.literals_of(s.precondition), r);
                add_condition(T, *p->st, o, T.literals_of(ce.condition), r);
                compile_rule(*p->st, r, declared_of({s.params, ce.extra_params}));
                if (r.head.never || r.plans[0].never)
                {
                    ++stats.dropped_rules;
                    continue;
                }
                p->rules.push_back(std::move(r));
            }
        }
    }
    for (u32 xi = 0; xi < T.axioms.size(); ++xi)
    {
        const Axiom& x = T.axioms[xi];
        if (!x.head.positive)
            continue;
        ++stats.rules;
        Rule r;
        r.name = "axiom:" + std::string(T.str(T.predicates[x.head.pred.v].name));
        r.axiom = xi;
        r.num_vars = x.params.count;
        r.head = key_pattern(L, x.head.pred.v, terms_of(T, x.head));
        add_condition(T, *p->st, o, T.literals_of(x.body), r);
        compile_rule(*p->st, r, declared_of({x.params}));
        if (r.head.never || r.plans[0].never)
        {
            ++stats.dropped_rules;
            continue;
        }
        p->rules.push_back(std::move(r));
    }
    p->obj_bits = static_cast<u32>(std::bit_width(std::max<u32>(T.num_objects(), 2) - 1));
    u32 anchor_ids = 0;
    for (u32 ri = 0; ri < p->rules.size(); ++ri)
    {
        const Rule& r = p->rules[ri];
        stats.plans += r.plans.size();
        if (r.rel.empty())
            p->seed_rules.push_back(ri);
        for (u32 s = 0; s < r.rel.size(); ++s)
            if (!r.plans[1 + s].never && !r.rel[s].never)
                p->anchors[r.rel[s].pred].push_back(make_anchor(r, ri, s, anchor_ids++));
        if (r.seed_slot != k_none && !r.plans.back().never && !r.rel[r.seed_slot].never)
            p->seed_anchors[r.rel[r.seed_slot].pred].push_back(make_anchor(r, ri, r.seed_slot, k_none));
    }
    // the dedup keys: obj_bits per key variable, then the anchor id in the low id_bits
    p->id_bits = static_cast<u32>(std::bit_width(std::max<u32>(anchor_ids, 1)));
    for (auto& per_pred : p->anchors)
        for (detail::Anchor& an : per_pred)
            if (an.dedup && p->id_bits + an.key_vars.size() * p->obj_bits > 63)
                an.dedup = false;
    for (const GroundAtom& a : T.fluent_init)
    {
        std::vector<u32> args;
        for (ObjectId ob : T.objects_of(a))
            args.push_back(ob.v);
        const u64 c = L.offset[a.pred.v] == CanonicalLayout::k_none ? L.total : L.encode(a.pred.v, args.data());
        if (c < L.total)
            p->initial.push_back(c);
    }
    for (const Literal& l : T.literals_of(T.goal))
    {
        const PredKind k = T.predicates[l.pred.v].kind;
        std::vector<u32> args;
        for (Term t : T.terms_of(l))
            args.push_back(term_object(t).v);
        if (k == PredKind::Static)
        {
            const bool holds = p->st->is_equality(l.pred.v) ? args[0] == args[1] : p->st->contains(l.pred.v, args.data());
            if (holds != l.positive)
                p->static_goal = false;
            continue;
        }
        if (!l.positive)
            continue;
        u64 c = L.offset[l.pred.v] == CanonicalLayout::k_none || L.size[l.pred.v] == 0 ? L.total : L.encode(l.pred.v, args.data());
        c = c < L.total ? c : ~u64{0};
        if (std::find(p->goal.begin(), p->goal.end(), c) == p->goal.end())
            p->goal.push_back(c);
    }
    p->goal_bits.assign(bits::words_for(std::max<u64>(L.total, 1)), 0);
    for (u64 g : p->goal)
        if (g != ~u64{0})
            bits::set(p->goal_bits.data(), g);
    return p;
}

// ------------------------------------------------------------------------------------------------ the fixpoint
class Fixpoint
{
public:
    /// Scratch for fixpoints of one program; reset() prepares a run (sparse clears: restricted fixpoints and goal
    /// queries reuse it, see cached()).
    explicit Fixpoint(const Program& p) : P(p), C(p.task->compiled()), L(p.task->atoms().layout())
    {
        m_known.init(C, p.total);
        m_old.init(C, p.total);
        m_args.resize(std::max<u32>(1, L.max_arity));
    }

    void reset(std::span<const CanonicalAtom> forbid, bool stop_at_goal, bool witnesses)
    {
        if (m_used)
        {
            m_known.clear();
            m_old.clear();
        }
        m_used = true;
        m_stop_at_goal = stop_at_goal;
        m_halt = false;
        m_goal_left = 0;
        m_goal_impossible = false;
        m_bindings = 0;
        m_pending.clear();
        m_pending_args.clear();
        m_delta.clear();
        m_delta_args.clear();
        // a few forbidden atoms are compared directly; more go into a bitset
        if (m_forbidden_bits)
            for (u64 c : m_forbidden_list)
                m_forbidden[c >> 6] = 0;
        m_forbidden_list.clear();
        for (CanonicalAtom c : forbid)
            if (c < P.total)
                m_forbidden_list.push_back(c);
        m_forbidden_bits = m_forbidden_list.size() > k_forbidden_list;
        if (m_forbidden_bits)
        {
            if (m_forbidden.empty())
                m_forbidden.assign(bits::words_for(std::max<u64>(P.total, 1)), 0);
            for (u64 c : m_forbidden_list)
                bits::set(m_forbidden.data(), c);
        }
        m_collect = !stop_at_goal;
        const usize npred = P.task->data().predicates.size();
        if (!m_out)
        {
            m_out = std::make_unique<TableData>();
            m_out->view.init(C, P.total);
            m_out->atoms.resize(npred);
            m_out->tuples.resize(npred);
        }
        else
        {
            // a kept table (goal queries only: nothing collected, no witnesses)
            m_out->view.clear();
            m_out->fluent = m_out->derived = 0;
            m_out->goal = false;
            m_out->rounds = 0;
        }
        m_out->witnesses = witnesses;
        for (u64 g : P.goal)
        {
            if (g == ~u64{0})
                m_goal_impossible = true;
            else if (!forbidden(g))
                ++m_goal_left;
            else
                m_goal_impossible = true;
        }
    }

    /// The table of the last run (the scratch allocates a new one for the next).
    std::unique_ptr<TableData> take() { return std::move(m_out); }
    [[nodiscard]] bool goal() const { return m_out->goal; }

    void run(u64* bindings)
    {
        TableData& out = *m_out;
        // round 0: the initial atoms, then every rule over them
        for (u64 c : P.initial)
        {
            if (known(c) || forbidden(c))
                continue;
            const u32 pred = L.decode(c, m_args.data());
            reach(c, pred, m_args.data(), k_none, nullptr, 0);
        }
        auto goal_done = [&]() { return P.static_goal && !m_goal_impossible && m_goal_left == 0; };
        u32 rounds = 0;
        for (bool first = true;; first = false)
        {
            if (m_pending.empty() && !first)
                break;
            ++rounds;
            // the delta: last round's atoms, now part of New
            m_delta.swap(m_pending);
            m_delta_args.swap(m_pending_args);
            m_pending.clear();
            m_pending_args.clear();
            for (const Pending& a : m_delta)
                out.view.add(C, a.key, a.pred, L.arity[a.pred], m_delta_args.data() + a.args);
            if (m_stop_at_goal && goal_done())
                break;
            if (first)
            {
                for (u32 ri = 0; ri < P.rules.size() && !m_halt; ++ri)
                    if (P.rules[ri].seed_slot == k_none)
                        run_plan(ri, 0);
                for (const Pending& a : m_delta)
                {
                    if (m_halt)
                        break;
                    const u32* args = m_delta_args.data() + a.args;
                    for (const detail::Anchor& an : P.seed_anchors[a.pred])
                    {
                        const Rule& r = P.rules[an.rule];
                        if (!unify(r.body[an.lit], args))
                            continue;
                        if (!run_plan(an.rule, static_cast<u32>(r.plans.size() - 1)))
                            break;
                    }
                }
            }
            else
            {
                m_seen.clear();
                for (const Pending& a : m_delta)
                {
                    const u32* args = m_delta_args.data() + a.args;
                    for (const detail::Anchor& an : P.anchors[a.pred])
                    {
                        if (!unify(P.rules[an.rule].body[an.lit], args))
                            continue;
                        if (an.dedup && !first_run(an))
                            continue;
                        if (!run_plan(an.rule, 1 + an.slot))
                            break;
                    }
                    if (m_halt)
                        break;
                }
            }
            if (m_halt)
                break;
            for (const Pending& a : m_delta)
                m_old.add(C, a.key, a.pred, L.arity[a.pred], m_delta_args.data() + a.args);
        }
        // atoms reached in the last (goal-stopped) round are reached too
        for (const Pending& a : m_pending)
            out.view.add(C, a.key, a.pred, L.arity[a.pred], m_pending_args.data() + a.args);
        out.rounds = rounds;
        out.goal = goal_done();

        if (bindings)
            *bindings = m_bindings;
    }

private:
    struct Pending
    {
        u64 key;
        u32 pred;
        u32 args;
    };
    struct Ctx
    {
        Fixpoint& f;
        const Rule& rule;
        [[nodiscard]] const reach::RowStore& rows(u32 mode) const { return view(mode).rows; }
        [[nodiscard]] const reach::View& view(u32 mode) const { return mode == 0 ? f.m_old : mode == 1 ? f.m_out->view : f.m_known; }
        bool custom(u32, const u32*) { return true; }
        bool head_ok(const u32* b)
        {
            const u64 k = rule.head.key(b);
            return k < f.P.total && !f.known(k) && !f.forbidden(k);
        }
        bool emit(const u32* b)
        {
            ++f.m_bindings;
            const u64 k = rule.head.key(b);
            if (k >= f.P.total || f.known(k) || f.forbidden(k))
                return true;
            for (u32 i = 0; i < rule.head.terms.size(); ++i)
            {
                const i32 t = rule.head.terms[i];
                f.m_args[i] = reach::is_var(t) ? b[t] : reach::term_obj(t);
            }
            f.reach(k, rule.head.pred, f.m_args.data(), f.m_rule_index, b, rule.num_vars);
            return !f.m_halt;
        }
    };

    [[nodiscard]] bool known(u64 c) const noexcept { return (m_known.bits[c >> 6] >> (c & 63)) & 1; }
    static constexpr usize k_forbidden_list = 8;
    [[nodiscard]] bool forbidden(u64 c) const noexcept
    {
        if (m_forbidden_bits)
            return (m_forbidden[c >> 6] >> (c & 63)) & 1;
        for (u64 f : m_forbidden_list)
            if (f == c)
                return true;
        return false;
    }

    void reach(u64 c, u32 pred, const u32* args, u32 rule, const u32* bind, u32 nvars)
    {
        const u32 ar = L.arity[pred];
        m_known.add(C, c, pred, ar, args);
        m_pending.push_back({c, pred, static_cast<u32>(m_pending_args.size())});
        m_pending_args.insert(m_pending_args.end(), args, args + ar);
        TableData& out = *m_out;
        if (m_collect)
        {
            out.atoms[pred].push_back(c);
            out.tuples[pred].insert(out.tuples[pred].end(), args, args + ar);
        }
        (c < P.fluent_count ? out.fluent : out.derived) += 1;
        if (out.witnesses)
        {
            out.position.insert(c, static_cast<u32>(out.wit_rule.size()));
            out.wit_rule.push_back(rule);
            if (bind)
                out.wit_bind.insert(out.wit_bind.end(), bind, bind + nvars);
            out.wit_begin.push_back(out.wit_bind.size());
        }
        if (m_goal_left > 0 && ((P.goal_bits[c >> 6] >> (c & 63)) & 1))
        {
            --m_goal_left;
            if (m_stop_at_goal && m_goal_left == 0 && P.static_goal && !m_goal_impossible)
                m_halt = true;
        }
    }

    /// Whether this round has not yet run the anchored plan for the anchor variables' values (Anchor::dedup), after
    /// testing the anchor-only variables against their domain (the plan would stop there).
    bool first_run(const detail::Anchor& an)
    {
        const u32* b = m_exec.bind();
        const reach::Plan& pl = P.rules[an.rule].plans[1 + an.slot];
        for (u32 v : an.free_vars)
            if (!bits::test(pl.dom0.data() + static_cast<usize>(v) * pl.ow, pl.ow, b[v]))
                return false;
        u64 key = 0;
        for (u32 v : an.key_vars)
            key = (key << P.obj_bits) | b[v];
        key = (key << P.id_bits) | an.id;
        if (m_seen.find(key))
            return false;
        m_seen.insert(key, 1);
        return true;
    }

    /// Unifies the anchor literal with an atom's arguments: binds its variables into the executor.
    bool unify(const Lit& anchor, const u32* args)
    {
        const Lit* l = &anchor;
        u32* b = m_exec_bind();
        for (u32 i = 0; i < l->terms.size(); ++i)
        {
            const i32 t = l->terms[i];
            if (!reach::is_var(t))
            {
                if (reach::term_obj(t) != args[i])
                    return false;
                continue;
            }
            bool seen = false;
            for (u32 j = 0; j < i; ++j)
                if (l->terms[j] == t)
                {
                    seen = true;
                    if (b[t] != args[i])
                        return false;
                    break;
                }
            if (!seen)
                b[t] = args[i];
        }
        return true;
    }
    u32* m_exec_bind()
    {
        m_exec.reserve(m_max_vars(), 1, 1);
        return m_exec.bind();
    }
    u32 m_max_vars()
    {
        if (!m_maxv)
            for (const Rule& r : P.rules)
                m_maxv = std::max(m_maxv, r.num_vars);
        return std::max<u32>(m_maxv, 1);
    }

    bool run_plan(u32 ri, u32 plan)
    {
        m_rule_index = ri;
        Ctx ctx{*this, P.rules[ri]};
        m_exec.run(P.rules[ri].plans[plan], ctx);
        return !m_halt;
    }

    const Program& P;
    const plan::Compiled& C;
    const CanonicalLayout& L;
    bool m_stop_at_goal = false;
    std::unique_ptr<TableData> m_out;
    reach::View m_known;  // every atom reached so far (Old: before the last round; New: up to the last round)
    std::vector<u64> m_forbidden, m_forbidden_list;
    bool m_collect = true;  // the atoms and tuples per predicate (a goal-only query needs neither)
    bool m_forbidden_bits = false;
    bool m_used = false;
    reach::View m_old;
    reach::Executor m_exec;
    reach::KeyMap<u8> m_seen;  // this round's dedup keys (Anchor::dedup)
    std::vector<Pending> m_pending, m_delta;
    std::vector<u32> m_pending_args, m_delta_args, m_args;
    u64 m_goal_left = 0;
    bool m_goal_impossible = false;
    bool m_halt = false;
    u32 m_rule_index = 0;
    u32 m_maxv = 0;

    u64 m_bindings = 0;
};
}  // namespace

namespace detail
{
namespace
{
/// This thread's fixpoint scratch for the program (the last program used on this thread).
Fixpoint& cached(const Program& p)
{
    thread_local std::unique_ptr<Fixpoint> f;
    thread_local u64 id = 0;
    if (!f || id != p.id)
    {
        f = std::make_unique<Fixpoint>(p);
        id = p.id;
    }
    return *f;
}
}  // namespace

std::unique_ptr<TableData> run_fixpoint(const Program& p, std::span<const CanonicalAtom> forbidden, bool stop_at_goal, bool record_witnesses,
                                        u64* bindings)
{
    Fixpoint& f = cached(p);
    f.reset(forbidden, stop_at_goal, record_witnesses);
    f.run(bindings);
    return f.take();
}

bool run_goal_query(const Program& p, std::span<const CanonicalAtom> forbidden)
{
    Fixpoint& f = cached(p);
    f.reset(forbidden, true, false);
    f.run(nullptr);
    return f.goal();
}

struct WitnessMemo
{
    std::vector<u64> forbidden;  // sorted
    std::vector<u8> state;       // per position: 0 unknown, 1 on the stack, 2 avoids, 3 does not
};
}  // namespace detail

// ================================================================================================ Table
Table::Table(std::shared_ptr<const detail::Program> program, std::unique_ptr<detail::TableData> data)
    : m_program(std::move(program)), m_data(std::move(data))
{
}
Table::Table(Table&&) noexcept = default;
Table& Table::operator=(Table&&) noexcept = default;
Table::~Table() = default;

bool Table::is_reachable(CanonicalAtom atom) const { return atom < m_program->total && m_data->view.test(atom); }

bool Table::is_reachable(PredicateId predicate, std::span<const ObjectId> objects) const
{
    const CanonicalLayout& L = m_program->task->atoms().layout();
    if (predicate.v >= L.offset.size() || L.offset[predicate.v] == CanonicalLayout::k_none || objects.size() != L.arity[predicate.v])
        return false;
    u64 c = L.offset[predicate.v];
    for (u32 i = 0; i < objects.size(); ++i)
        c += L.position_table(predicate.v, i)[objects[i].v];
    return is_reachable(c);
}

std::span<const CanonicalAtom> Table::atoms(PredicateId predicate) const
{
    return predicate.v < m_data->atoms.size() ? std::span<const CanonicalAtom>(m_data->atoms[predicate.v]) : std::span<const CanonicalAtom>();
}
std::span<const u32> Table::tuples(PredicateId predicate) const
{
    return predicate.v < m_data->tuples.size() ? std::span<const u32>(m_data->tuples[predicate.v]) : std::span<const u32>();
}
u64 Table::num_atoms() const { return m_data->fluent + m_data->derived; }
u64 Table::num_fluent_atoms() const { return m_data->fluent; }
u64 Table::num_derived_atoms() const { return m_data->derived; }
bool Table::goal_reachable() const { return m_data->goal; }
u32 Table::rounds() const { return m_data->rounds; }
bool Table::has_witnesses() const { return m_data->witnesses; }

WitnessQuery Table::witness_query(std::span<const CanonicalAtom> forbidden) const
{
    if (!m_data->witnesses)
        throw std::logic_error("mymyr: this reachability table recorded no witnesses");
    return WitnessQuery(*this, forbidden);
}

std::vector<std::vector<ObjectId>> Table::project(const ConjunctiveQuery& q) const
{
    const Program& P = *m_program;
    reach::StaticTables& st = *P.st;
    const TaskData& T = P.task->data();
    auto check_term = [&](QueryTerm t)
    {
        if (t >= 0 && static_cast<u32>(t) >= q.num_variables)
            throw std::invalid_argument("mymyr: conjunctive query term uses variable " + std::to_string(t) + " but the query has " +
                                        std::to_string(q.num_variables) + " variables");
        if (t < 0 && reach::term_obj(t) >= T.num_objects())
            throw std::invalid_argument("mymyr: conjunctive query term names an unknown object");
    };
    std::vector<Lit> lits;
    u32 slots = 0;
    for (const QueryLiteral& ql : q.literals)
    {
        for (QueryTerm t : ql.terms)
            check_term(t);
        if (ql.predicate.v >= T.predicates.size() || ql.terms.size() != T.predicates[ql.predicate.v].arity)
            throw std::invalid_argument("mymyr: conjunctive query literal with a wrong predicate or arity");
        const PredKind k = T.predicates[ql.predicate.v].kind;
        Lit l;
        l.pred = ql.predicate.v;
        l.positive = ql.positive;
        l.terms = ql.terms;
        if (k == PredKind::Static)
        {
            if (!ql.positive && !P.options.enforce_negative_static)
                continue;
            l.type = st.is_equality(l.pred) ? LitType::Eq : LitType::Static;
        }
        else
        {
            if (!ql.positive)
                continue;
            l.type = LitType::Rel;
            l.slot = slots++;
        }
        lits.push_back(std::move(l));
    }
    for (int pass = 0; pass < 2; ++pass)
        for (const auto& [a, b] : pass == 0 ? q.equalities : q.disequalities)
        {
            check_term(a);
            check_term(b);
            Lit l;
            l.type = LitType::Eq;
            l.positive = pass == 0;
            l.terms = {a, b};
            lits.push_back(std::move(l));
        }
    const u32 V = q.num_variables, OW = st.ow();
    std::vector<std::vector<u64>> found(V, std::vector<u64>(OW, 0));
    struct Ctx
    {
        const TableData& d;
        std::vector<std::vector<u64>>& found;
        u32 V;
        [[nodiscard]] const reach::RowStore& rows(u32) const { return d.view.rows; }
        [[nodiscard]] const reach::View& view(u32) const { return d.view; }
        bool custom(u32, const u32*) { return true; }
        bool head_ok(const u32*) { return true; }
        bool emit(const u32* b)
        {
            for (u32 u = 0; u < V; ++u)
                bits::set(found[u].data(), b[u]);
            return true;
        }
    };
    Ctx ctx{*m_data, found, V};
    reach::Executor ex;
    for (u32 v = 0; v < V; ++v)
    {
        reach::CompileSpec spec;
        spec.num_vars = V;
        spec.relevant.assign(V, 0);
        spec.relevant[v] = 1;
        spec.slot_mode.assign(slots, 1);
        reach::Plan plan = reach::compile(st, lits, spec);
        if (plan.never)
            return std::vector<std::vector<ObjectId>>(V);
        // values already seen in a solution need no second search
        u64* d = plan.dom0.data() + static_cast<usize>(v) * OW;
        for (u32 w = 0; w < OW; ++w)
            d[w] &= ~found[v][w];
        ex.run(plan, ctx);
    }
    std::vector<std::vector<ObjectId>> out(V);
    for (u32 v = 0; v < V; ++v)
        bits::for_each(found[v].data(), OW, [&](u64 o) { out[v].push_back(ObjectId{static_cast<u32>(o)}); });
    return out;
}

// ================================================================================================ WitnessQuery
WitnessQuery::WitnessQuery(const Table& table, std::span<const CanonicalAtom> forbidden)
    : m_table(&table), m_memo(std::make_unique<detail::WitnessMemo>())
{
    m_memo->forbidden.assign(forbidden.begin(), forbidden.end());
    std::sort(m_memo->forbidden.begin(), m_memo->forbidden.end());
    m_memo->state.assign(table.data().wit_rule.size(), 0);
}
WitnessQuery::WitnessQuery(WitnessQuery&&) noexcept = default;
WitnessQuery& WitnessQuery::operator=(WitnessQuery&&) noexcept = default;
WitnessQuery::~WitnessQuery() = default;

u64 WitnessQuery::memoized() const
{
    return static_cast<u64>(std::count_if(m_memo->state.begin(), m_memo->state.end(), [](u8 s) { return s >= 2; }));
}

WitnessVerdict WitnessQuery::avoids(CanonicalAtom atom) const
{
    const TableData& d = m_table->data();
    const Program& P = *m_table->program();
    detail::WitnessMemo& m = *m_memo;
    const u32* start = d.position.find(atom);
    if (!start)
        return WitnessVerdict::Unknown;
    auto is_forbidden = [&](u64 c) { return std::binary_search(m.forbidden.begin(), m.forbidden.end(), c); };
    // iterative post-order over the recorded derivation tree
    std::vector<std::pair<u32, u32>> stack;  // (position, next body literal)
    auto enter = [&](u32 pos, u64 key) -> bool  // false: decided now
    {
        if (m.state[pos] >= 2)
            return false;
        if (m.state[pos] == 1)
            return false;  // a cycle cannot happen (bodies precede heads); treat as not certified below
        if (is_forbidden(key))
        {
            m.state[pos] = 3;
            return false;
        }
        m.state[pos] = 1;
        stack.emplace_back(pos, 0);
        return true;
    };
    if (!enter(*start, atom))
        return m.state[*start] == 2 ? WitnessVerdict::ReachableWithout : WitnessVerdict::Unknown;
    while (!stack.empty())
    {
        auto& [pos, next] = stack.back();
        const u32 rule = d.wit_rule[pos];
        if (rule == k_none)  // an initial atom
        {
            m.state[pos] = 2;
            stack.pop_back();
            continue;
        }
        const Rule& r = P.rules[rule];
        const u32* b = d.wit_bind.data() + (pos == 0 ? 0 : d.wit_begin[pos - 1]);
        bool descended = false, failed = false;
        while (next < r.rel.size())
        {
            const u64 key = r.rel[next].key(b);
            ++next;
            const u32* bp = d.position.find(key);
            if (!bp)
            {
                failed = true;
                break;
            }
            const u8 s = m.state[*bp];
            if (s == 2)
                continue;
            if (s == 3 || s == 1)
            {
                failed = true;
                break;
            }
            if (enter(*bp, key))
            {
                descended = true;
                break;
            }
            if (m.state[*bp] != 2)
            {
                failed = true;
                break;
            }
        }
        if (descended)
            continue;
        m.state[pos] = failed ? 3 : 2;
        stack.pop_back();
        if (failed)
        {
            // every ancestor on the stack fails as well
            while (!stack.empty())
            {
                m.state[stack.back().first] = 3;
                stack.pop_back();
            }
        }
    }
    return m.state[*start] == 2 ? WitnessVerdict::ReachableWithout : WitnessVerdict::Unknown;
}

// ================================================================================================ RelaxedReachability
RelaxedReachability::RelaxedReachability(Private, const Task& task, const Options& options) : m_options(options)
{
    const auto t0 = Clock::now();
    m_program = build_program(task, options, m_stats);
    m_stats.compile_ms = ms_since(t0);
    const auto t1 = Clock::now();
    auto data = detail::run_fixpoint(*m_program, {}, false, options.record_witnesses, &m_stats.bindings);
    m_stats.fixpoint_ms = ms_since(t1);
    m_stats.rounds = data->rounds;
    m_stats.fluent_atoms = data->fluent;
    m_stats.derived_atoms = data->derived;
    m_table = std::make_unique<Table>(m_program, std::move(data));
}

RelaxedReachability::~RelaxedReachability() = default;

std::shared_ptr<const RelaxedReachability> RelaxedReachability::create(const Task& task, const Options& options)
{
    return std::make_shared<const RelaxedReachability>(Private{}, task, options);
}

const Task& RelaxedReachability::task() const noexcept { return *m_program->task; }

Table RelaxedReachability::restricted(std::span<const CanonicalAtom> forbidden) const
{
    return Table(m_program, detail::run_fixpoint(*m_program, forbidden, false, false, nullptr));
}

bool RelaxedReachability::goal_reachable_without(std::span<const CanonicalAtom> forbidden) const
{
    return detail::run_goal_query(*m_program, forbidden);
}
}  // namespace mymyr::reachability
