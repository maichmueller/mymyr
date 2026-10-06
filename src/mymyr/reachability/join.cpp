// Compilation of conjunctive matchers (join.hpp) and the static tables they read.

#include "join.hpp"

#include "mymyr/task/task.hpp"

#include <algorithm>
#include <bit>

namespace mymyr::reach
{
using namespace formalism;

StaticTables::StaticTables(const Task& task) : m_task(&task), m_c(&task.compiled())
{
    const TaskData& T = task.data();
    m_n = T.num_objects();
    m_ow = std::max<u32>(1, bits::words_for(m_n));
    m_atoms.resize(T.predicates.size());
    for (const GroundAtom& a : T.static_init)
        for (ObjectId o : T.objects_of(a))
            m_atoms[a.pred.v].push_back(o.v);
    for (u32 p = 0; p < T.predicates.size(); ++p)
        if (T.predicates[p].kind == PredKind::Static && T.predicates[p].arity == 2 && T.str(T.predicates[p].name) == "=")
            m_eq = p;
}

bool StaticTables::contains(u32 pred, const u32* args) const
{
    const plan::StaticRelation& R = m_c->statics[pred];
    u64 k = 0;
    for (u32 i = 0; i < R.arity; ++i)
        k += R.rs[static_cast<usize>(i) * m_n + args[i]];
    return R.contains(k);
}

bool StaticTables::agrees(const u32* atom, const std::vector<i32>& terms, u32 skip1, u32 skip2) const
{
    for (u32 j = 0; j < terms.size(); ++j)
    {
        if (j == skip1 || j == skip2)
            continue;
        if (!is_var(terms[j]) && atom[j] != term_obj(terms[j]))
            return false;
    }
    // repeated variables at the other positions must agree with each other
    for (u32 j = 0; j < terms.size(); ++j)
        for (u32 k = j + 1; k < terms.size(); ++k)
            if (is_var(terms[j]) && terms[j] == terms[k] && atom[j] != atom[k])
                return false;
    return true;
}

namespace
{
std::vector<i64> cache_key(i64 kind, u32 pred, u32 a, u32 b, const std::vector<i32>& terms)
{
    std::vector<i64> k{kind, pred, a, b};
    // constants and the repetition pattern of the variables (variable identities are irrelevant)
    for (u32 j = 0; j < terms.size(); ++j)
    {
        if (!is_var(terms[j]))
        {
            k.push_back(terms[j]);
            continue;
        }
        i64 first = j;
        for (u32 i = 0; i < j; ++i)
            if (terms[i] == terms[j])
            {
                first = i;
                break;
            }
        k.push_back(1'000'000 + first);
    }
    return k;
}
}  // namespace

const u64* StaticTables::projection(u32 pred, u32 i, const std::vector<i32>& terms)
{
    std::lock_guard lock(m_mutex);
    auto key = cache_key(0, pred, i, 0, terms);
    if (auto it = m_cache.find(key); it != m_cache.end())
        return it->second->data();
    auto v = std::make_unique<std::vector<u64>>(m_ow, 0);
    const u32 ar = static_cast<u32>(terms.size());
    const auto& atoms = m_atoms[pred];
    for (usize a = 0; a + ar <= atoms.size() && ar > 0; a += ar)
        if (agrees(atoms.data() + a, terms, i, k_none))
            bits::set(v->data(), atoms[a + i]);
    return m_cache.emplace(std::move(key), std::move(v)).first->second->data();
}

const u64* StaticTables::rows(u32 pred, u32 from, u32 to, const std::vector<i32>& terms)
{
    std::lock_guard lock(m_mutex);
    auto key = cache_key(1, pred, from, to, terms);
    if (auto it = m_cache.find(key); it != m_cache.end())
        return it->second->data();
    // the n rows, a zero word, the rows' word ranges (row_ranges())
    const usize table = static_cast<usize>(m_n) * m_ow;
    auto v = std::make_unique<std::vector<u64>>(table + 1 + m_n, 0);
    const u32 ar = static_cast<u32>(terms.size());
    const auto& atoms = m_atoms[pred];
    for (usize a = 0; a + ar <= atoms.size() && ar > 0; a += ar)
        if (agrees(atoms.data() + a, terms, from, to))
            bits::set(v->data() + static_cast<usize>(atoms[a + from]) * m_ow, atoms[a + to]);
    for (u32 o = 0; o < m_n; ++o)
        (*v)[table + 1 + o] = word_range(v->data() + static_cast<usize>(o) * m_ow, m_ow);
    return m_cache.emplace(std::move(key), std::move(v)).first->second->data();
}

const u64* StaticTables::objects_of_types(const std::vector<u32>& types)
{
    std::lock_guard lock(m_mutex);
    std::vector<i64> key{2};
    for (u32 t : types)
        key.push_back(t);
    if (auto it = m_cache.find(key); it != m_cache.end())
        return it->second->data();
    const TaskData& T = m_task->data();
    auto v = std::make_unique<std::vector<u64>>(m_ow, 0);
    // ancestors (reflexive) of every type
    const u32 nt = static_cast<u32>(T.types.size());
    std::vector<std::vector<u8>> anc(nt, std::vector<u8>(nt, 0));
    for (u32 t = 0; t < nt; ++t)
    {
        std::vector<u32> stack{t};
        while (!stack.empty())
        {
            const u32 x = stack.back();
            stack.pop_back();
            if (anc[t][x])
                continue;
            anc[t][x] = 1;
            for (TypeId b : TaskData::slice(T.type_ids, T.types[x].bases))
                stack.push_back(b.v);
        }
    }
    for (u32 o = 0; o < m_n; ++o)
    {
        bool ok = types.empty();
        for (TypeId ot : TaskData::slice(T.type_ids, T.objects[o].types))
            for (u32 t : types)
                ok = ok || (ot.v < nt && t < nt && anc[ot.v][t]);
        if (ok)
            bits::set(v->data(), o);
    }
    return m_cache.emplace(std::move(key), std::move(v)).first->second->data();
}

// ================================================================================================ compile
namespace
{
struct Binary
{
    u32 a = 0, b = 0;
    RowRef ab;  // constrains b given a
    RowRef ba;  // constrains a given b
};

struct PendingCheck
{
    CheckLit c;
    std::vector<u32> vars;
};

std::vector<u32> distinct_vars(const std::vector<i32>& terms)
{
    std::vector<u32> v;
    for (i32 t : terms)
        if (is_var(t) && std::find(v.begin(), v.end(), static_cast<u32>(t)) == v.end())
            v.push_back(static_cast<u32>(t));
    return v;
}
}  // namespace

Plan compile(StaticTables& st, const std::vector<Lit>& lits, const CompileSpec& spec)
{
    const plan::Compiled& C = st.compiled();
    const CanonicalLayout& L = C.layout;
    const u32 V = spec.num_vars, n = st.n(), OW = st.ow();
    Plan p;
    p.statics = C.statics.data();
    p.num_vars = V;
    p.ow = OW;
    p.slot_mode = spec.slot_mode;
    p.dom0.assign(static_cast<usize>(V) * OW, 0);
    auto dom = [&](u32 v) { return p.dom0.data() + static_cast<usize>(v) * OW; };
    for (u32 v = 0; v < V; ++v)
        for (u32 o = 0; o < n; ++o)
            bits::set(dom(v), o);
    auto and_dom = [&](u32 v, const u64* s, bool neg)
    {
        u64* d = dom(v);
        for (u32 w = 0; w < OW; ++w)
            d[w] &= neg ? ~s[w] : s[w];
        if (neg && n % 64)
            d[OW - 1] &= (u64{1} << (n % 64)) - 1;
    };
    auto single = [&](u32 v, u32 o, bool keep_only)
    {
        u64* d = dom(v);
        if (keep_only)
        {
            const bool had = bits::test(d, OW, o);
            std::fill(d, d + OW, 0);
            if (had)
                bits::set(d, o);
        }
        else
            bits::reset(d, o);
    };
    // a binary static row constraint: the table's row of the source value, with its word range
    auto static_row = [&](bool neg, u32 src, const u64* table, u32 var)
    {
        RowRef r{RowRef::Kind::Static, neg, 0, 0, src, table, var};
        r.rng = st.row_ranges(table);
        return r;
    };
    const std::vector<u8> prebound = spec.prebound.empty() ? std::vector<u8>(V, 0) : spec.prebound;
    const std::vector<u8> relevant = spec.relevant.empty() ? std::vector<u8>(V, 1) : spec.relevant;

    std::vector<std::vector<RowRef>> unary(V);
    std::vector<Binary> binaries;
    std::vector<PendingCheck> checks;
    std::vector<u8> pos_bound(V, 0);  // bound by a positive relation or static literal
    std::vector<std::pair<u32, u32>> eq_edges;

    auto rel_key = [&](CheckLit& c, u32 pred, const std::vector<i32>& terms)
    {
        if (L.offset[pred] == CanonicalLayout::k_none || L.size[pred] == 0)
        {
            c.never = true;
            return;
        }
        c.base = L.offset[pred];
        c.var_begin = static_cast<u32>(p.key_vars.size());
        for (u32 i = 0; i < terms.size(); ++i)
        {
            const u64* rs = L.position_table(pred, i);
            if (is_var(terms[i]))
            {
                p.key_vars.push_back({static_cast<u32>(terms[i]), rs});
                ++c.var_count;
            }
            else
            {
                const u64 x = rs[term_obj(terms[i])];
                if (x >= CanonicalLayout::k_outside)
                    c.never = true;
                c.base += x;
            }
        }
    };
    auto static_key = [&](CheckLit& c, u32 pred, const std::vector<i32>& terms)
    {
        const plan::StaticRelation& R = C.statics[pred];
        c.var_begin = static_cast<u32>(p.key_vars.size());
        for (u32 i = 0; i < terms.size(); ++i)
        {
            const u64* rs = R.position_table(i, n);
            if (is_var(terms[i]))
            {
                p.key_vars.push_back({static_cast<u32>(terms[i]), rs});
                ++c.var_count;
            }
            else
            {
                const u64 x = rs[term_obj(terms[i])];
                if (x >= plan::StaticRelation::k_outside)
                    c.never = true;
                c.base += x;
            }
        }
    };

    for (const Lit& l : lits)
    {
        const std::vector<u32> vars = distinct_vars(l.terms);
        const bool identity_eq = l.type == LitType::Eq || (l.type == LitType::Static && spec.equality_by_identity && st.is_equality(l.pred));
        if (identity_eq)
        {
            const i32 a = l.terms[0], b = l.terms[1];
            if (!is_var(a) && !is_var(b))
            {
                if ((a == b) != l.positive)
                    p.never = true;
                continue;
            }
            if (is_var(a) && is_var(b))
            {
                if (a == b)
                {
                    if (!l.positive)
                        p.never = true;
                    continue;
                }
                Binary bin;
                bin.a = static_cast<u32>(a);
                bin.b = static_cast<u32>(b);
                bin.ab = {RowRef::Kind::Same, !l.positive, 0, 0, bin.a, nullptr, bin.b};
                bin.ba = {RowRef::Kind::Same, !l.positive, 0, 0, bin.b, nullptr, bin.a};
                binaries.push_back(bin);
                if (l.positive)
                    eq_edges.emplace_back(bin.a, bin.b);
                continue;
            }
            const u32 v = static_cast<u32>(is_var(a) ? a : b);
            const u32 o = term_obj(is_var(a) ? b : a);
            single(v, o, l.positive);
            continue;
        }
        if (l.type == LitType::Custom)
        {
            PendingCheck pc;
            pc.c.type = LitType::Custom;
            pc.c.slot = l.slot;
            pc.vars = vars;
            checks.push_back(std::move(pc));
            continue;
        }
        const u32 ar = static_cast<u32>(l.terms.size());
        if (l.type == LitType::Static)
        {
            if (vars.empty())
            {
                std::vector<u32> args;
                for (i32 t : l.terms)
                    args.push_back(term_obj(t));
                if (st.contains(l.pred, args.data()) != l.positive)
                    p.never = true;
                continue;
            }
            if (l.positive)
                for (u32 v : vars)
                    pos_bound[v] = 1;
            if (ar == 1)
            {
                and_dom(vars[0], st.projection(l.pred, 0, l.terms), !l.positive);
                continue;
            }
            if (ar == 2 && l.terms[0] != l.terms[1])
            {
                const bool v0 = is_var(l.terms[0]), v1 = is_var(l.terms[1]);
                if (v0 && v1)
                {
                    Binary bin;
                    bin.a = static_cast<u32>(l.terms[0]);
                    bin.b = static_cast<u32>(l.terms[1]);
                    bin.ab = static_row(!l.positive, bin.a, st.rows(l.pred, 0, 1, l.terms), bin.b);
                    bin.ba = static_row(!l.positive, bin.b, st.rows(l.pred, 1, 0, l.terms), bin.a);
                    binaries.push_back(bin);
                    continue;
                }
                const u32 fpos = v0 ? 0 : 1, cpos = 1 - fpos;
                const u32 v = static_cast<u32>(l.terms[fpos]);
                const u32 c = term_obj(l.terms[cpos]);
                and_dom(v, st.rows(l.pred, cpos, fpos, l.terms) + static_cast<usize>(c) * OW, !l.positive);
                continue;
            }
            // arity >= 3 or repeated variables: projections (positive only; sound over-approximations) + a check
            if (l.positive)
            {
                for (u32 i = 0; i < ar; ++i)
                    if (is_var(l.terms[i]))
                        and_dom(static_cast<u32>(l.terms[i]), st.projection(l.pred, i, l.terms), false);
                for (u32 i = 0; i < ar; ++i)
                    for (u32 j = 0; j < ar; ++j)
                        if (i != j && is_var(l.terms[i]) && is_var(l.terms[j]) && l.terms[i] != l.terms[j] &&
                            std::find(l.terms.begin(), l.terms.begin() + i, l.terms[i]) == l.terms.begin() + i &&
                            std::find(l.terms.begin(), l.terms.begin() + j, l.terms[j]) == l.terms.begin() + j && i < j)
                        {
                            Binary bin;
                            bin.a = static_cast<u32>(l.terms[i]);
                            bin.b = static_cast<u32>(l.terms[j]);
                            bin.ab = static_row(false, bin.a, st.rows(l.pred, i, j, l.terms), bin.b);
                            bin.ba = static_row(false, bin.b, st.rows(l.pred, j, i, l.terms), bin.a);
                            binaries.push_back(bin);
                        }
            }
            PendingCheck pc;
            pc.c.type = LitType::Static;
            pc.c.positive = l.positive;
            pc.c.pred = l.pred;
            static_key(pc.c, l.pred, l.terms);
            pc.vars = vars;
            checks.push_back(std::move(pc));
            continue;
        }
        // relation literal
        if (l.positive)
            for (u32 v : vars)
                pos_bound[v] = 1;
        if (ar == 1 && !vars.empty())
        {
            unary[vars[0]].push_back({RowRef::Kind::Rel, !l.positive, l.slot, C.view.unary_row[l.pred], k_none, nullptr, vars[0]});
            continue;
        }
        if (ar == 2 && !vars.empty() && l.terms[0] != l.terms[1])
        {
            const bool v0 = is_var(l.terms[0]), v1 = is_var(l.terms[1]);
            if (v0 && v1)
            {
                Binary bin;
                bin.a = static_cast<u32>(l.terms[0]);
                bin.b = static_cast<u32>(l.terms[1]);
                bin.ab = {RowRef::Kind::Rel, !l.positive, l.slot, C.view.fwd_row[l.pred], bin.a, nullptr, bin.b};
                bin.ba = {RowRef::Kind::Rel, !l.positive, l.slot, C.view.bwd_row[l.pred], bin.b, nullptr, bin.a};
                binaries.push_back(bin);
                continue;
            }
            const u32 fpos = v0 ? 0 : 1;
            const u32 v = static_cast<u32>(l.terms[fpos]);
            const u32 c = term_obj(l.terms[1 - fpos]);
            const u32 row = fpos == 0 ? C.view.bwd_row[l.pred] + c : C.view.fwd_row[l.pred] + c;
            unary[v].push_back({RowRef::Kind::Rel, !l.positive, l.slot, row, k_none, nullptr, v});
            continue;
        }
        PendingCheck pc;
        pc.c.type = LitType::Rel;
        pc.c.positive = l.positive;
        pc.c.slot = l.slot;
        pc.c.pred = l.pred;
        rel_key(pc.c, l.pred, l.terms);
        pc.vars = vars;
        checks.push_back(std::move(pc));
    }

    // positive equalities propagate "bound by a positive literal"
    for (bool changed = true; changed;)
    {
        changed = false;
        for (auto [a, b] : eq_edges)
            if (pos_bound[a] != pos_bound[b])
            {
                pos_bound[a] = pos_bound[b] = 1;
                changed = true;
            }
    }
    // a variable no positive literal binds ranges over its declared type, as in mimir's type-domain relation
    for (u32 v = 0; v < V; ++v)
        if (!pos_bound[v] && !prebound[v] && v < spec.var_types.size())
            and_dom(v, st.objects_of_types(spec.var_types[v]), false);
    for (u32 v = 0; v < V; ++v)
        if (!prebound[v] && !bits::any(dom(v), OW))
            p.never = true;
    p.dom_rng.resize(V);
    for (u32 v = 0; v < V; ++v)
        p.dom_rng[v] = word_range(dom(v), OW);

    // ---- binding order: relevant (enumerated) variables first, then connected ones, then more constraints, then fewer
    // candidates (the witness variables come last: one completion each)
    std::vector<u8> placed = prebound;
    std::vector<u32> order;
    // a relevant variable enumerated without a link to the variables before it, although later ones constrain it: arc
    // consistency can narrow it through those (the enumeration would otherwise try every value and fail in the witness)
    bool blind = false;
    for (;;)
    {
        int best = -1;
        i64 best_score = 0, best_links = 0;
        for (u32 v = 0; v < V; ++v)
        {
            if (placed[v])
                continue;
            i64 links = 0;
            for (const Binary& b : binaries)
                if ((b.a == v && placed[b.b]) || (b.b == v && placed[b.a]))
                    ++links;
            const i64 pop = static_cast<i64>(bits::count(dom(v), OW));
            const i64 score = (relevant[v] ? i64{1} << 50 : 0) + (links > 0 ? i64{1} << 45 : 0) + links * (i64{1} << 36) +
                              static_cast<i64>(unary[v].size()) * (i64{1} << 30) - pop;
            if (best < 0 || score > best_score)
                best = static_cast<int>(v), best_score = score, best_links = links;
        }
        if (best < 0)
            break;
        if (relevant[best] && best_links == 0)
            for (const Binary& b : binaries)
                if ((b.a == static_cast<u32>(best) && !placed[b.b]) || (b.b == static_cast<u32>(best) && !placed[b.a]))
                {
                    const u32 other = b.a == static_cast<u32>(best) ? b.b : b.a;
                    if (!relevant[other])
                        blind = true;
                }
        placed[best] = 1;
        order.push_back(static_cast<u32>(best));
    }
    std::vector<i32> pos_of(V, -1);
    for (u32 d = 0; d < order.size(); ++d)
        pos_of[order[d]] = static_cast<i32>(d);
    for (u32 v = 0; v < V; ++v)
        if (prebound[v])
            p.prebound.push_back(v);

    std::vector<std::vector<RowRef>> step_rows(order.size());
    std::vector<std::vector<u32>> step_checks(order.size());
    for (u32 v = 0; v < V; ++v)
        for (const RowRef& r : unary[v])
        {
            if (prebound[v])
                p.pre_rows.push_back(r);
            else
                step_rows[pos_of[v]].push_back(r);
        }
    for (const Binary& b : binaries)
    {
        if (prebound[b.a] && prebound[b.b])
        {
            p.pre_rows.push_back(b.ab);  // row selected by a, tested at b
            continue;
        }
        if (pos_of[b.a] < pos_of[b.b])  // a earlier (or prebound)
            step_rows[pos_of[b.b]].push_back(b.ab);
        else
            step_rows[pos_of[b.a]].push_back(b.ba);
    }
    for (PendingCheck& pc : checks)
    {
        i32 last = -1;
        for (u32 v : pc.vars)
            last = std::max(last, pos_of[v]);
        const u32 ci = static_cast<u32>(p.checks.size());
        p.checks.push_back(pc.c);
        if (last < 0)
            p.pre_checks.push_back(ci);
        else
            step_checks[last].push_back(ci);
    }
    for (u32 d = 0; d < order.size(); ++d)
    {
        Step s;
        s.var = order[d];
        s.row_begin = static_cast<u32>(p.rows.size());
        p.rows.insert(p.rows.end(), step_rows[d].begin(), step_rows[d].end());
        s.row_end = static_cast<u32>(p.rows.size());
        s.check_begin = static_cast<u32>(p.step_checks.size());
        p.step_checks.insert(p.step_checks.end(), step_checks[d].begin(), step_checks[d].end());
        s.check_end = static_cast<u32>(p.step_checks.size());
        p.steps.push_back(s);
    }
    // forward check (see Plan::fc): steps after the first whose rows read prebound variables (or none) only; worthwhile
    // when at least one of those rows depends on a prebound variable or on a relation (a unary row of a view)
    for (u32 d = 1; d < order.size(); ++d)
    {
        Plan::FcGroup g;
        g.var = order[d];
        g.begin = static_cast<u32>(p.fc_rows.size());
        bool useful = false;
        for (const RowRef& r : step_rows[d])
            if (r.src == k_none || prebound[r.src])
            {
                p.fc_rows.push_back(r);
                useful |= r.src != k_none || r.kind == RowRef::Kind::Rel;
            }
        g.end = static_cast<u32>(p.fc_rows.size());
        if (useful && g.end > g.begin)
            p.fc.push_back(g);
        else
            p.fc_rows.resize(g.begin);
    }
    // arc consistency for plans with many free variables (see Plan::ac)
    if (order.size() >= k_ac_min_steps || blind)
    {
        p.ac = true;
        for (u32 v = 0; v < V; ++v)
            if (!prebound[v])
                for (const RowRef& r : unary[v])
                    p.ac_unary.push_back(r);
        for (const Binary& b : binaries)
        {
            p.ac_arcs.push_back(b.ab);
            p.ac_arcs.push_back(b.ba);
        }
    }
    p.first_exist = static_cast<u32>(order.size());
    while (p.first_exist > 0 && !relevant[order[p.first_exist - 1]])
        --p.first_exist;
    if (!spec.head.empty())
    {
        i32 last = -1;
        for (u32 v = 0; v < V; ++v)
            if (spec.head[v])
                last = std::max(last, pos_of[v]);
        p.head_depth = static_cast<u32>(last + 1);
        if (p.head_depth > p.first_exist)
            p.head_depth = k_none;  // a head variable is witness-only: no prefix check
    }
    return p;
}
}  // namespace mymyr::reach
