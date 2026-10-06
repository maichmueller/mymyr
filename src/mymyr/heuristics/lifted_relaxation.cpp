// The lifted h_max / h_add / h_FF evaluation (lifted_relaxation.hpp).

#include "lifted_relaxation.hpp"

#include "../reachability/join.hpp"
#include "../reachability/keymap.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>

namespace mymyr::heuristics::detail
{
using namespace formalism;
using reach::Lit;
using reach::LitType;

namespace
{
constexpr u32 k_inf = ~u32{0};
constexpr u32 k_none = ~u32{0};

[[nodiscard]] inline u32 sat_add(u32 a, u32 b) noexcept
{
    const u64 s = static_cast<u64>(a) + b;
    return s >= k_inf ? k_inf - 1 : static_cast<u32>(s);
}

/// The canonical id of an atom pattern under a binding: base + sum of rs[bind[var]] (>= total: outside the space).
struct KeyPat
{
    u32 pred = 0;
    u64 base = 0;
    bool never = false;
    std::vector<reach::KeyVar> vars;
    std::vector<i32> terms;

    [[nodiscard]] u64 key(const u32* b, u64 total) const noexcept
    {
        if (never)
            return total;
        u64 k = base;
        for (const reach::KeyVar& v : vars)
            k += v.rs[b[v.var]];
        return k;
    }
};

KeyPat key_pattern(const CanonicalLayout& L, u32 pred, std::span<const Term> terms)
{
    KeyPat k;
    k.pred = pred;
    k.terms.assign(terms.begin(), terms.end());
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

/// One relaxed operator: a (schema, conditional effect) pair or an axiom.
struct Op
{
    u32 schema = 0;  // original schema, or axiom index
    bool axiom = false;
    u32 arity = 0;   // parameters of the original action (the ground action of a relaxed-plan step)
    u32 num_vars = 0;
    std::vector<KeyPat> pos, neg, adds, dels;  // pos: by relation-literal slot
    std::vector<Lit> body;                     // relation literals (slot = index into pos) and static literals
    std::vector<reach::Plan> plans;            // [0] unanchored, [1 + slot] anchored at slot, [1 + |pos|] the seed plan
    u32 seed_slot = k_none;
    bool head_filter = false;  // one add, no delete: a binding whose add is final is cut (the plans carry it)
};

struct Anchor
{
    u32 op = 0, slot = 0;
};

/// Monotone bucket queue of (cost, entry), popped a whole bucket at a time: small costs by index, larger ones in a map.
class Buckets
{
public:
    static constexpr u32 k_direct = 4096;

    void clear()
    {
        for (u32 c : m_used)
            m_b[c].clear();
        m_used.clear();
        m_far.clear();
        m_cur = 0;
    }
    void push(u32 cost, u64 x)
    {
        if (cost < k_direct)
        {
            if (m_b.size() <= cost)
                m_b.resize(cost + 1);
            if (m_b[cost].empty())
                m_used.push_back(cost);
            m_b[cost].push_back(x);
            m_cur = std::min(m_cur, cost);
        }
        else
            m_far[cost].push_back(x);
    }
    /// Moves the entries of the smallest non-empty bucket into `out`; false when empty.
    bool pop(u32& cost, std::vector<u64>& out)
    {
        out.clear();
        for (; m_cur < m_b.size(); ++m_cur)
            if (!m_b[m_cur].empty())
            {
                cost = m_cur;
                out.swap(m_b[m_cur]);
                return true;
            }
        if (m_far.empty())
            return false;
        const auto it = m_far.begin();
        cost = it->first;
        out.swap(it->second);
        m_far.erase(it);
        return true;
    }

private:
    std::vector<std::vector<u64>> m_b;
    std::vector<u32> m_used;
    std::map<u32, std::vector<u64>> m_far;
    u32 m_cur = 0;
};

constexpr u64 k_del_flag = u64{1} << 63;
}  // namespace

struct LiftedRelaxation::Impl
{
    Impl(const Task& task, Kind kind, bool real)
        : T(task), D(task.data()), C(task.compiled()), L(task.atoms().layout()), st(task), kind(kind), real(real)
    {
        total = L.total;
        build();
        if (real)
            costs = std::make_unique<ActionCosts>(task);
        const u64 words = bits::words_for(std::max<u64>(total, 1));
        old_view.init(C, total);
        new_view.init(C, total);
        final_view.init(C, total);
        in_s.assign(words, 0);
        args.resize(std::max<u32>(1, L.max_arity));
        u32 mv = 1;
        for (const Op& op : ops)
            mv = std::max(mv, op.num_vars);
        exec.reserve(mv, 1, 1);
        bind_scratch.resize(mv);
    }

    // ------------------------------------------------------------------------------------ compilation
    void build();
    void compile(Op& op, const std::vector<std::vector<u32>>& declared, const std::vector<std::vector<u64>>& masks);

    // ------------------------------------------------------------------------------------ evaluation
    Value evaluate(StateView s, const std::vector<std::vector<GoalLit>>& goals);

    /// The per-evaluation index of canonical atom c (interned; its scratch reset on first touch in this evaluation).
    u32 touch(u64 c)
    {
        const usize before = dense.size();
        u32& ix = dense.insert(c, static_cast<u32>(before));
        if (dense.size() != before)
        {
            canon.push_back(c);
            stamp.push_back(0);
            cost.push_back(k_inf);
            supp.push_back(k_none);
            dcost.push_back(k_inf);
            dsupp.push_back(k_none);
            flags.push_back(0);
            wait.push_back(k_none);
        }
        const u32 x = ix;
        if (stamp[x] != gen)
        {
            stamp[x] = gen;
            cost[x] = k_inf;
            supp[x] = k_none;
            dcost[x] = k_inf;
            dsupp[x] = k_none;
            flags[x] = 0;
            wait[x] = k_none;
        }
        return x;
    }
    /// The per-evaluation index of c if it was touched in this evaluation, else k_none.
    [[nodiscard]] u32 find(u64 c) const
    {
        const u32* ix = dense.find(c);
        return ix && stamp[*ix] == gen ? *ix : k_none;
    }
    [[nodiscard]] bool state_has(u64 c) const noexcept { return c < total && ((in_s[c >> 6] >> (c & 63)) & 1); }

    static constexpr u8 k_settled = 1, k_dsettled = 2, k_final = 4;

    struct Ctx
    {
        Impl& f;
        u32 op;
        [[nodiscard]] const reach::RowStore& rows(u32 mode) const { return view(mode).rows; }
        [[nodiscard]] const reach::View& view(u32 mode) const { return mode == 0 ? f.old_view : mode == 1 ? f.new_view : f.final_view; }
        bool custom(u32, const u32*) { return true; }
        bool head_ok(const u32* b)
        {
            const u64 k = f.ops[op].adds[0].key(b, f.total);
            return k < f.total && !f.final_view.test(k);
        }
        bool emit(const u32* b) { return f.emit(op, b); }
    };

    bool emit(u32 oi, const u32* b);
    /// Applies a binding whose positive preconditions are settled: false if a negated precondition parks it.
    void fire(u32 oi, const u32* b, u32 acc_pos, u32 last_supp, u32 last_cost, u32 parked);
    void set_cost(u32 x, u32 val, u32 sup);
    void set_dcost(u32 x, u32 val, u32 sup);
    u32 record(u32 oi, const u32* b)
    {
        const u32 r = static_cast<u32>(rec_op.size());
        rec_op.push_back(oi);
        rec_begin.push_back(static_cast<u32>(rec_bind.size()));
        rec_bind.insert(rec_bind.end(), b, b + ops[oi].num_vars);
        return r;
    }
    [[nodiscard]] u32 op_cost(u32 oi, const u32* b)
    {
        const Op& op = ops[oi];
        if (op.axiom)
            return 0;
        if (!real)
            return 1;
        obj_scratch.resize(op.num_vars);
        for (u32 i = 0; i < op.num_vars; ++i)
            obj_scratch[i] = ObjectId{b[i]};
        return static_cast<u32>(costs->cost(op.schema, obj_scratch.data()));
    }
    bool unify(const Op& op, u32 slot, const u32* a)
    {
        const KeyPat& l = op.pos[slot];
        u32* b = exec.bind();
        for (u32 i = 0; i < l.terms.size(); ++i)
        {
            const i32 t = l.terms[i];
            if (!reach::is_var(t))
            {
                if (reach::term_obj(t) != a[i])
                    return false;
                continue;
            }
            bool seen = false;
            for (u32 j = 0; j < i; ++j)
                if (l.terms[j] == t)
                {
                    seen = true;
                    if (b[t] != a[i])
                        return false;
                    break;
                }
            if (!seen)
                b[t] = a[i];
        }
        return true;
    }
    void run(u32 oi, u32 plan)
    {
        Ctx ctx{*this, oi};
        exec.run(ops[oi].plans[plan], ctx);
    }

    const Task& T;
    const TaskData& D;
    const plan::Compiled& C;
    const CanonicalLayout& L;
    reach::StaticTables st;
    Kind kind;
    bool real;
    u64 total = 0;
    std::unique_ptr<ActionCosts> costs;
    std::vector<Op> ops;
    std::vector<std::vector<Anchor>> anchors, seed_anchors;  // per predicate
    std::vector<u32> seed_ops;                               // operators without relation literals

    // per evaluation
    const std::function<bool()>* interrupt = nullptr;
    u64 ticks = 0;
    u32 gen = 0;
    u32 bucket = 0;
    reach::KeyMap<u32> dense;  // canonical id -> index (persistent; the scratch below is stamped per evaluation)
    std::vector<u64> canon;
    std::vector<u32> stamp, cost, supp, dcost, dsupp, wait;
    std::vector<u8> flags;
    reach::View old_view, new_view, final_view;
    std::vector<u64> in_s, state_keys;
    Buckets queue;
    std::vector<u64> batch;
    struct Delta
    {
        u64 key;
        u32 pred;
        u32 args;
    };
    std::vector<Delta> delta;
    std::vector<u32> delta_args, args, bind_scratch;
    std::vector<ObjectId> obj_scratch;
    reach::Executor exec;
    // parked bindings: operator, binding, positive accumulation; linked per deletion proposition
    std::vector<u32> park_op, park_acc, park_next, park_begin, park_bind, park_last_supp, park_last_cost;
    std::vector<u32> woken;
    // h_FF records
    std::vector<u32> rec_op, rec_begin, rec_bind;
    // slot cache: task fluent slot -> canonical id
    std::vector<u64> slot_canon;
};

void LiftedRelaxation::Impl::build()
{
    const u32 n = D.num_objects();
    const u32 ow = st.ow();
    anchors.assign(D.predicates.size(), {});
    seed_anchors.assign(D.predicates.size(), {});
    const auto kind_of = [&](u32 p) { return D.predicates[p].kind; };
    auto same_terms = [&](const KeyPat& k, const Literal& l)
    {
        const auto t = D.terms_of(l);
        return k.pred == l.pred.v && std::equal(k.terms.begin(), k.terms.end(), t.begin(), t.end());
    };
    // one condition's literals as a set (loki repeats nullary literals; mimir keeps them once)
    auto add_unique = [&](std::vector<KeyPat>& v, usize first, const Literal& l)
    {
        for (usize i = first; i < v.size(); ++i)
            if (same_terms(v[i], l))
                return false;
        v.push_back(key_pattern(L, l.pred.v, D.terms_of(l)));
        return true;
    };
    auto relaxed_pre = [&](std::span<const Literal> lits, bool keep_static, Op& op)
    {
        const usize pos0 = op.pos.size(), neg0 = op.neg.size();
        for (const Literal& l : lits)
        {
            const PredKind k = kind_of(l.pred.v);
            if (k == PredKind::Static)
            {
                if (keep_static)
                {
                    Lit x;
                    x.type = st.is_equality(l.pred.v) ? LitType::Eq : LitType::Static;
                    x.positive = l.positive;
                    x.pred = l.pred.v;
                    const auto t = D.terms_of(l);
                    x.terms.assign(t.begin(), t.end());
                    op.body.push_back(std::move(x));
                }
                continue;
            }
            if (l.positive)
            {
                if (add_unique(op.pos, pos0, l))
                {
                    Lit x;
                    x.type = LitType::Rel;
                    x.pred = l.pred.v;
                    x.slot = static_cast<u32>(op.pos.size() - 1);
                    x.terms = op.pos.back().terms;
                    op.body.push_back(std::move(x));
                }
            }
            else if (k == PredKind::Fluent)
                add_unique(op.neg, neg0, l);
        }
    };
    auto declared_of = [&](std::initializer_list<Range> ranges)
    {
        std::vector<std::vector<u32>> d;
        for (Range r : ranges)
            for (const Parameter& par : TaskData::slice(D.params, r))
            {
                std::vector<u32> v;
                for (TypeId t : TaskData::slice(D.type_ids, par.types))
                    v.push_back(t.v);
                d.push_back(std::move(v));
            }
        return d;
    };

    std::map<std::pair<u32, u32>, std::vector<u8>> proj;
    auto projection = [&](u32 p, u32 i) -> const std::vector<u8>&
    {
        auto it = proj.find({p, i});
        if (it != proj.end())
            return it->second;
        std::vector<u8> v(n, 0);
        for (const GroundAtom& a : D.static_init)
            if (a.pred.v == p)
                v[D.objects_of(a)[i].v] = 1;
        return proj.emplace(std::make_pair(p, i), std::move(v)).first->second;
    };

    for (u32 si = 0; si < D.schemas.size(); ++si)
    {
        const Schema& sc = D.schemas[si];
        const auto ces = D.effects_of(sc);
        for (u32 k = 0; k < ces.size(); ++k)
        {
            const ConditionalEffect& ce = ces[k];
            if (ce.effects.count == 0)
                continue;
            Op op;
            op.schema = si;
            op.arity = sc.arity();
            op.num_vars = sc.arity() + ce.extra_params.count;
            relaxed_pre(D.literals_of(sc.precondition), true, op);
            relaxed_pre(D.literals_of(ce.condition), false, op);  // static effect-condition literals ignored, as in mimir
            std::vector<std::vector<u64>> masks(op.num_vars);
            // quantified parameters: mimir's single-parameter static consistency candidates
            for (u32 j = 0; j < ce.extra_params.count; ++j)
            {
                const u32 param = sc.arity() + j;
                std::vector<u64> mask(ow, 0);
                for (u32 o = 0; o < n; ++o)
                {
                    bool ok = true;
                    for (const Literal& l : D.literals_of(ce.condition))
                    {
                        if (kind_of(l.pred.v) != PredKind::Static)
                            continue;
                        const u32 ar = D.predicates[l.pred.v].arity;
                        if (ar < 1 || (!l.positive && ar != 1))
                            continue;
                        const auto terms = D.terms_of(l);
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
                        bits::set(mask.data(), o);
                }
                masks[param] = std::move(mask);
            }
            for (const Literal& l : TaskData::slice(D.literals, ce.effects))
                (l.positive ? op.adds : op.dels).push_back(key_pattern(L, l.pred.v, D.terms_of(l)));
            compile(op, declared_of({sc.params, ce.extra_params}), masks);
            if (!op.plans[0].never)
                ops.push_back(std::move(op));
        }
    }
    for (u32 xi = 0; xi < D.axioms.size(); ++xi)
    {
        const Axiom& x = D.axioms[xi];
        Op op;
        op.schema = xi;
        op.axiom = true;
        op.arity = x.params.count;
        op.num_vars = x.params.count;
        relaxed_pre(D.literals_of(x.body), true, op);
        op.adds.push_back(key_pattern(L, x.head.pred.v, D.terms_of(x.head)));
        compile(op, declared_of({x.params}), {});
        if (!op.plans[0].never)
            ops.push_back(std::move(op));
    }
    for (u32 oi = 0; oi < ops.size(); ++oi)
    {
        const Op& op = ops[oi];
        if (op.pos.empty())
            seed_ops.push_back(oi);
        for (u32 s = 0; s < op.pos.size(); ++s)
            if (!op.plans[1 + s].never && !op.pos[s].never)
                anchors[op.pos[s].pred].push_back({oi, s});
        if (op.seed_slot != k_none && !op.plans.back().never && !op.pos[op.seed_slot].never)
            seed_anchors[op.pos[op.seed_slot].pred].push_back({oi, op.seed_slot});
    }
}

void LiftedRelaxation::Impl::compile(Op& op, const std::vector<std::vector<u32>>& declared, const std::vector<std::vector<u64>>& masks)
{
    const u32 V = op.num_vars;
    const u32 S = static_cast<u32>(op.pos.size());
    // Relevant variables (enumerated exhaustively; the others get one completion per relevant binding): those of the
    // effects and of the negated preconditions; for h_add also those of the positive preconditions (the sum depends on
    // the completion; under h_max every completion of one run costs the same); with real costs all (the cost may read
    // any parameter).
    std::vector<u8> relevant(V, 0);
    auto mark = [&](const std::vector<KeyPat>& ks)
    {
        for (const KeyPat& k : ks)
            for (i32 t : k.terms)
                if (reach::is_var(t))
                    relevant[t] = 1;
    };
    mark(op.adds);
    mark(op.dels);
    mark(op.neg);
    if (kind == Kind::Add)
        mark(op.pos);
    if (real && !op.axiom)
        std::fill(relevant.begin(), relevant.end(), 1);
    op.head_filter = op.adds.size() == 1 && op.dels.empty() && !op.adds[0].never;

    reach::CompileSpec spec;
    spec.num_vars = V;
    spec.relevant = relevant;
    spec.var_types = declared;  // every parameter ranges over its declared type (the grounding's)
    spec.slot_mode.assign(S + 1, 1);
    spec.slot_mode[S] = 2;
    spec.prebound.assign(V, 0);
    std::vector<Lit> body = op.body;
    if (op.head_filter)
    {
        std::vector<u8> head(V, 0);
        for (i32 t : op.adds[0].terms)
            if (reach::is_var(t))
                head[t] = 1;
        spec.head = head;
        Lit h;
        h.type = LitType::Rel;
        h.positive = false;
        h.pred = op.adds[0].pred;
        h.slot = S;
        h.terms = op.adds[0].terms;
        body.push_back(std::move(h));
    }
    auto finish = [&](reach::Plan p)
    {
        // the quantified parameters' candidates
        for (u32 v = 0; v < masks.size() && v < V; ++v)
            if (!masks[v].empty())
                for (u32 w = 0; w < p.ow; ++w)
                    p.dom0[static_cast<usize>(v) * p.ow + w] &= masks[v][w];
        for (u32 v = 0; v < V; ++v)
            if (!bits::any(p.dom0.data() + static_cast<usize>(v) * p.ow, p.ow))
                p.never = true;
        return p;
    };
    op.plans.push_back(finish(reach::compile(st, body, spec)));
    for (u32 a = 0; a < S; ++a)
    {
        reach::CompileSpec sa = spec;
        for (u32 s = 0; s < S; ++s)
            sa.slot_mode[s] = s < a ? 0 : 1;
        std::vector<Lit> rest;
        for (const Lit& l : body)
        {
            if (l.type == LitType::Rel && l.positive && l.slot == a)
            {
                for (i32 t : l.terms)
                    if (reach::is_var(t))
                        sa.prebound[t] = 1;
                continue;  // the anchor: unified with the new atom
            }
            rest.push_back(l);
        }
        op.plans.push_back(finish(reach::compile(st, rest, sa)));
    }
    u32 best = k_none, best_arity = 2;
    for (const Lit& l : op.body)
        if (l.type == LitType::Rel && l.terms.size() > best_arity)
            best = l.slot, best_arity = static_cast<u32>(l.terms.size());
    if (best != k_none)
    {
        reach::CompileSpec sa = spec;
        std::vector<Lit> rest;
        for (const Lit& l : body)
        {
            if (l.type == LitType::Rel && l.positive && l.slot == best)
            {
                for (i32 t : l.terms)
                    if (reach::is_var(t))
                        sa.prebound[t] = 1;
                continue;
            }
            rest.push_back(l);
        }
        op.plans.push_back(finish(reach::compile(st, rest, sa)));
        op.seed_slot = best;
    }
}

void LiftedRelaxation::Impl::set_cost(u32 x, u32 val, u32 sup)
{
    if (val >= cost[x])
        return;
    cost[x] = val;
    supp[x] = sup;
    queue.push(val, x);
    if (val == bucket && !(flags[x] & k_final))
    {
        // final: no binding of this bucket or later can undercut it (the head filter reads this)
        flags[x] |= k_final;
        const u64 c = canon[x];
        const u32 pred = L.decode(c, args.data());
        final_view.add(C, c, pred, L.arity[pred], args.data());
    }
}

void LiftedRelaxation::Impl::set_dcost(u32 x, u32 val, u32 sup)
{
    if (val >= dcost[x])
        return;
    dcost[x] = val;
    dsupp[x] = sup;
    queue.push(val, x | k_del_flag);
}

void LiftedRelaxation::Impl::fire(u32 oi, const u32* b, u32 acc, u32 last_supp, u32 last_cost, u32 parked)
{
    const Op& op = ops[oi];
    const bool add = kind == Kind::Add && !op.axiom;
    const bool ff = kind == Kind::FF;
    for (const KeyPat& k : op.neg)
    {
        const u64 c = k.key(b, total);
        if (!state_has(c))
            continue;  // false in s: costs 0
        const u32 x = find(c);
        if (!(flags[x] & k_dsettled))
        {
            // park until the deletion cost of x settles (the re-check list)
            u32 p = parked;
            if (p == k_none)
            {
                p = static_cast<u32>(park_op.size());
                park_op.push_back(oi);
                park_acc.push_back(acc);
                park_last_supp.push_back(last_supp);
                park_last_cost.push_back(last_cost);
                park_begin.push_back(static_cast<u32>(park_bind.size()));
                park_bind.insert(park_bind.end(), b, b + op.num_vars);
                park_next.push_back(k_none);
            }
            park_next[p] = wait[x];
            wait[x] = p;
            return;
        }
        const u32 v = dcost[x];
        acc = add ? sat_add(acc, v) : std::max(acc, v);
        if (ff && op.axiom && (last_cost == k_inf || v >= last_cost))
        {
            last_cost = v;
            last_supp = dsupp[x];
        }
    }
    const u32 val = sat_add(acc, op_cost(oi, b));
    u32 sup = k_none;
    bool have = false;
    auto supporter = [&]()
    {
        if (!have && ff)
        {
            have = true;
            sup = op.axiom ? last_supp : record(oi, b);
        }
        return sup;
    };
    for (const KeyPat& k : op.adds)
    {
        const u64 c = k.key(b, total);
        if (c >= total)
            continue;
        const u32 x = touch(c);
        if (val < cost[x])
            set_cost(x, val, supporter());
    }
    for (const KeyPat& k : op.dels)
    {
        const u64 c = k.key(b, total);
        if (!state_has(c))
            continue;
        const u32 x = find(c);
        if (val < dcost[x])
            set_dcost(x, val, supporter());
    }
}

bool LiftedRelaxation::Impl::emit(u32 oi, const u32* b)
{
    if ((++ticks & 1023) == 0 && interrupt && *interrupt && (*interrupt)())
        throw Interrupted();
    const Op& op = ops[oi];
    const bool add = kind == Kind::Add && !op.axiom;
    const bool ff_axiom = kind == Kind::FF && op.axiom;
    u32 acc = 0, last_supp = k_none, last_cost = k_inf;
    if (add || ff_axiom)
    {
        for (const KeyPat& k : op.pos)
        {
            const u32 x = find(k.key(b, total));
            const u32 v = x == k_none ? 0 : cost[x];
            acc = add ? sat_add(acc, v) : std::max(acc, v);
            if (ff_axiom && (last_cost == k_inf || v >= last_cost))
            {
                last_cost = v;
                last_supp = supp[x];
            }
        }
    }
    else if (!op.pos.empty())
        acc = bucket;  // h_max: the anchor settled at this bucket, every other precondition no later
    fire(oi, b, acc, last_supp, last_cost, k_none);
    return true;
}

Value LiftedRelaxation::Impl::evaluate(StateView s, const std::vector<std::vector<GoalLit>>& goals)
{
    ++gen;
    if (gen == 0)
    {
        std::fill(stamp.begin(), stamp.end(), 0);
        gen = 1;
    }
    for (u64 c : state_keys)
        in_s[c >> 6] = 0;
    state_keys.clear();
    old_view.clear();
    new_view.clear();
    final_view.clear();
    queue.clear();
    park_op.clear();
    park_acc.clear();
    park_next.clear();
    park_begin.clear();
    park_bind.clear();
    park_last_supp.clear();
    park_last_cost.clear();
    rec_op.clear();
    rec_begin.clear();
    rec_bind.clear();
    bucket = 0;

    // the state's atoms: cost 0
    const AtomIndex& TA = T.atoms();
    bits::for_each(s.w, s.nw,
                   [&](u64 slot)
                   {
                       if (slot >= slot_canon.size())
                           slot_canon.resize(slot + 1, ~u64{0});
                       u64& c = slot_canon[slot];
                       if (c == ~u64{0})
                           c = TA.canonical(AtomKind::Fluent, static_cast<u32>(slot));
                       if (c >= total)
                           return;
                       in_s[c >> 6] |= u64{1} << (c & 63);
                       state_keys.push_back(c);
                       const u32 x = touch(c);
                       cost[x] = 0;
                       queue.push(0, x);
                   });

    // goal literals: positive atoms and the deletion propositions of negated atoms true in s
    std::vector<u32> targets;
    std::vector<std::vector<std::pair<u32, bool>>> goal_refs(goals.size());  // (index or k_none, negated)
    std::vector<u8> goal_dead(goals.size(), 0);
    for (usize gi = 0; gi < goals.size(); ++gi)
        for (const GoalLit& g : goals[gi])
        {
            const u64 c = L.offset[g.pred] == CanonicalLayout::k_none ? total : L.encode(g.pred, g.args.data());
            if (g.positive)
            {
                if (c >= total)
                {
                    goal_dead[gi] = 1;
                    continue;
                }
                const u32 x = touch(c);
                goal_refs[gi].push_back({x, false});
                targets.push_back(x);
            }
            else
            {
                if (!state_has(c))
                    continue;  // false in s: satisfied at cost 0
                const u32 x = find(c);
                goal_refs[gi].push_back({x, true});
                targets.push_back(x | 0x80000000u);
            }
        }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    u64 left = targets.size();
    auto settled_target = [&](u32 x, bool del)
    {
        const u32 key = del ? (x | 0x80000000u) : x;
        if (std::binary_search(targets.begin(), targets.end(), key))
            --left;
    };

    // operators without relation literals fire once, from nothing
    for (u32 oi : seed_ops)
    {
        const Op& op = ops[oi];
        for (u32 v = 0; v < op.num_vars; ++v)
            exec.bind()[v] = 0;
        run(oi, 0);
    }

    bool first = true;
    u32 c;
    while (left > 0 && queue.pop(c, batch))
    {
        bucket = c;
        delta.clear();
        delta_args.clear();
        woken.clear();
        for (u64 e : batch)
        {
            const u32 x = static_cast<u32>(e);
            if (e & k_del_flag)
            {
                if ((flags[x] & k_dsettled) || dcost[x] != c)
                    continue;
                flags[x] |= k_dsettled;
                settled_target(x, true);
                for (u32 p = wait[x]; p != k_none; p = park_next[p])
                    woken.push_back(p);
                wait[x] = k_none;
                continue;
            }
            if ((flags[x] & k_settled) || cost[x] != c)
                continue;
            flags[x] |= k_settled;
            settled_target(x, false);
            const u64 key = canon[x];
            const u32 pred = L.decode(key, args.data());
            const u32 ar = L.arity[pred];
            delta.push_back({key, pred, static_cast<u32>(delta_args.size())});
            delta_args.insert(delta_args.end(), args.data(), args.data() + ar);
            new_view.add(C, key, pred, ar, args.data());
            if (!(flags[x] & k_final))
            {
                flags[x] |= k_final;
                final_view.add(C, key, pred, ar, args.data());
            }
        }
        if (left == 0)
            break;
        if (first)
        {
            first = false;
            for (u32 oi = 0; oi < ops.size(); ++oi)
                if (!ops[oi].pos.empty() && ops[oi].seed_slot == k_none)
                    run(oi, 0);
            for (const Delta& a : delta)
                for (const Anchor& an : seed_anchors[a.pred])
                {
                    const Op& op = ops[an.op];
                    if (unify(op, an.slot, delta_args.data() + a.args))
                        run(an.op, static_cast<u32>(op.plans.size() - 1));
                }
        }
        else
            for (const Delta& a : delta)
                for (const Anchor& an : anchors[a.pred])
                    if (unify(ops[an.op], an.slot, delta_args.data() + a.args))
                        run(an.op, 1 + an.slot);
        // parked bindings whose deletion proposition settled: their positive preconditions are unchanged
        for (u32 p : woken)
            fire(park_op[p], park_bind.data() + park_begin[p], park_acc[p], park_last_supp[p], park_last_cost[p], p);
        for (const Delta& a : delta)
            old_view.add(C, a.key, a.pred, L.arity[a.pred], delta_args.data() + a.args);
    }

    // goals
    const bool add = kind == Kind::Add, ff = kind == Kind::FF;
    u64 best = k_inf;
    for (usize gi = 0; gi < goals.size(); ++gi)
    {
        if (goal_dead[gi])
            continue;
        u64 h = 0;
        bool dead = false;
        for (const auto& [x, neg] : goal_refs[gi])
        {
            const u32 v = neg ? ((flags[x] & k_dsettled) ? dcost[x] : k_inf) : ((flags[x] & k_settled) ? cost[x] : k_inf);
            if (v == k_inf)
            {
                dead = true;
                break;
            }
            h = add ? h + v : std::max<u64>(h, v);
        }
        if (dead)
            continue;
        if (ff)
        {
            std::unordered_set<u64> seen;
            std::unordered_set<std::string> actions;
            std::vector<u64> stack;
            u64 hc = 0;
            for (const auto& [x, neg] : goal_refs[gi])
                stack.push_back((static_cast<u64>(neg) << 32) | x);
            while (!stack.empty())
            {
                const u64 e = stack.back();
                stack.pop_back();
                if (!seen.insert(e).second)
                    continue;
                const u32 x = static_cast<u32>(e);
                const bool neg = (e >> 32) != 0;
                const u32 r = neg ? dsupp[x] : supp[x];
                if (r == k_none)
                    continue;
                const Op& op = ops[rec_op[r]];
                const u32* b = rec_bind.data() + rec_begin[r];
                std::string key(reinterpret_cast<const char*>(&op.schema), sizeof(u32));
                key.append(reinterpret_cast<const char*>(b), op.arity * sizeof(u32));
                if (actions.insert(key).second)
                    hc += op_cost(rec_op[r], b);
                for (const KeyPat& k : op.pos)
                    if (const u32 y = find(k.key(b, total)); y != k_none)
                        stack.push_back(y);
                for (const KeyPat& k : op.neg)
                {
                    const u64 cc = k.key(b, total);
                    if (state_has(cc))
                        stack.push_back((u64{1} << 32) | find(cc));
                }
            }
            h = hc;
        }
        best = std::min(best, h);
    }
    return best >= k_inf ? k_dead_end : static_cast<Value>(best);
}

LiftedRelaxation::LiftedRelaxation(const Task& task, Kind kind, bool real) : m(std::make_unique<Impl>(task, kind, real)) {}
LiftedRelaxation::~LiftedRelaxation() = default;

Value LiftedRelaxation::evaluate(StateView s, const std::vector<std::vector<GoalLit>>& goals)
{
    m->interrupt = interrupt;
    return m->evaluate(s, goals);
}
}  // namespace mymyr::heuristics::detail
