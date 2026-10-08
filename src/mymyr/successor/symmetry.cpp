// Symmetry pruning: the object graph of a state, its colour refinement classes and the kept objects per action
// parameter (successor/symmetry.hpp, successor/detail/symmetry.hpp).

#include "mymyr/successor/detail/symmetry.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/successor/detail/engine.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <numeric>

namespace mymyr::detail
{
namespace
{
using formalism::PredKind;
using formalism::Term;
}  // namespace

SymmetryPruner::SymmetryPruner(const Task& task, u32 ow) : m_n(task.data().num_objects()), m_ow(ow)
{
    const formalism::TaskData& d = task.data();
    const u32 n = m_n;
    const usize P = d.predicates.size();
    m_axioms = task.has_axioms();

    // Colour codes of the structure vertices (datasets/object_graph.hpp): per predicate of arity a != 1, max(a, 1) atom
    // codes (position j, or the atom itself for a = 0), then 2 * max(a, 1) goal literal codes (position, polarity).
    m_arity.resize(P);
    m_pred_code.resize(P);
    for (usize p = 0; p < P; ++p)
    {
        const u32 a = d.predicates[p].arity;
        m_arity[p] = a;
        m_pred_code[p] = m_num_codes;
        if (a != 1)
            m_num_codes += 3 * std::max<u32>(a, 1);
    }
    auto atom_code = [&](u32 p, u32 pos) { return m_pred_code[p] + pos; };
    auto literal_code = [&](u32 p, u32 pos, bool positive) { return m_pred_code[p] + std::max<u32>(m_arity[p], 1) + 2 * pos + (positive ? 1 : 0); };
    // a structure of `arity` vertices (one for arity 0) joined to its objects and along its positions
    auto add_static = [&](u32 arity, const u32* objects, auto&& code)
    {
        const u32 first = n + static_cast<u32>(m_static_code.size());
        for (u32 pos = 0; pos < std::max<u32>(arity, 1); ++pos)
        {
            m_static_code.push_back(code(pos));
            if (arity == 0)
                break;
            m_static_edges.emplace_back(first + pos, objects[pos]);
            if (pos > 0)
                m_static_edges.emplace_back(first + pos - 1, first + pos);
        }
    };

    // static atoms: unary ones colour their object
    std::vector<std::vector<u32>> key(n);  // per object: sorted static unary predicates, a separator, sorted goal (p, s)
    std::vector<u32> objs;
    // per static predicate and position: the objects at that position of some static atom (mimir's vertex assignment
    // sets, for the static domains below)
    std::vector<std::vector<std::vector<u8>>> at(P);
    for (const formalism::GroundAtom& a : d.static_init)
    {
        const auto o = d.objects_of(a);
        const u32 p = a.pred.v;
        objs.assign(o.size(), 0);
        for (usize j = 0; j < o.size(); ++j)
            objs[j] = o[j].v;
        if (at[p].empty())
            at[p].assign(m_arity[p], std::vector<u8>(n, 0));
        for (usize j = 0; j < o.size(); ++j)
            at[p][j][objs[j]] = 1;
        if (o.size() == 1)
            key[objs[0]].push_back(p);
        else
            add_static(static_cast<u32>(o.size()), objs.data(), [&](u32 pos) { return atom_code(p, pos); });
    }
    for (std::vector<u32>& k : key)
    {
        std::sort(k.begin(), k.end());
        k.push_back(~u32{0});
    }
    // goal literals (numeric goal constraints are not part of the graph)
    std::vector<std::vector<std::pair<u32, u32>>> goal_unary(n);
    for (const formalism::Literal& l : d.literals_of(d.goal))
    {
        const auto terms = d.terms_of(l);
        const u32 p = l.pred.v;
        if (terms.size() == 1)
        {
            goal_unary[formalism::term_object(terms[0]).v].emplace_back(p, l.positive ? 1u : 0u);
            continue;
        }
        objs.assign(terms.size(), 0);
        for (usize j = 0; j < terms.size(); ++j)
            objs[j] = formalism::term_object(terms[j]).v;
        add_static(static_cast<u32>(terms.size()), objs.data(), [&](u32 pos) { return literal_code(p, pos, l.positive); });
    }
    for (u32 o = 0; o < n; ++o)
    {
        std::sort(goal_unary[o].begin(), goal_unary[o].end());
        for (auto [p, s] : goal_unary[o])
        {
            key[o].push_back(p);
            key[o].push_back(s);
        }
    }
    // the constant part of the object colours, ranked
    std::vector<u32> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return key[a] < key[b]; });
    m_const_class.assign(n, 0);
    for (u32 i = 0, c = 0; i < n; ++i)
    {
        if (i > 0 && key[order[i - 1]] < key[order[i]])
            ++c;
        m_const_class[order[i]] = c;
    }

    // static domains of the schema parameters: as mimir's StaticConsistencyGraph vertices, object o is in the domain of
    // parameter i iff every static precondition literal of arity >= 1 (negative ones only if unary) holds at each of
    // its positions whose term is i (bound to o) or an object: a positive literal needs some static atom with that
    // object at that position, a negative unary one needs none
    m_param_begin.push_back(0);
    u32 max_arity = 0;
    for (const formalism::Schema& s : d.schemas)
    {
        const u32 k = s.arity();
        max_arity = std::max(max_arity, k);
        const auto lits = d.literals_of(s.precondition);
        for (u32 i = 0; i < k; ++i)
        {
            m_dom_begin.push_back(static_cast<u32>(m_dom.size()));
            for (u32 o = 0; o < n; ++o)
            {
                bool ok = true;
                for (const formalism::Literal& l : lits)
                {
                    const u32 p = l.pred.v;
                    if (d.predicates[p].kind != PredKind::Static || m_arity[p] == 0 || (!l.positive && m_arity[p] != 1))
                        continue;
                    const auto terms = d.terms_of(l);
                    for (usize j = 0; j < terms.size() && ok; ++j)
                    {
                        const Term t = terms[j];
                        u32 obj;
                        if (formalism::is_object(t))
                            obj = formalism::term_object(t).v;
                        else if (static_cast<u32>(t) == i)
                            obj = o;
                        else
                            continue;
                        const bool holds = !at[p].empty() && at[p][j][obj];
                        ok = holds == l.positive;
                    }
                    if (!ok)
                        break;
                }
                if (ok)
                    m_dom.push_back(o);
            }
        }
        m_param_begin.push_back(static_cast<u32>(m_dom_begin.size()));
    }
    m_dom_begin.push_back(static_cast<u32>(m_dom.size()));
    m_masks.assign(static_cast<usize>(max_arity) * m_ow, 0);
    m_unary_begin.assign(n + 1, 0);
}

std::span<const u32> SymmetryPruner::domain(u32 schema, u32 param) const noexcept
{
    const u32 x = m_param_begin[schema] + param;
    return {m_dom.data() + m_dom_begin[x], m_dom_begin[x + 1] - m_dom_begin[x]};
}

void SymmetryPruner::build_graph(const Engine& e)
{
    const u32 n = m_n;
    const AtomIndex& atoms = e.atoms();
    m_unary.clear();
    m_code.assign(m_static_code.begin(), m_static_code.end());  // structure vertices n, n + 1, ...
    m_edges.assign(m_static_edges.begin(), m_static_edges.end());
    auto add = [&](AtomKind kind, u64 slot)
    {
        const u32* r = atoms.record(kind, static_cast<u32>(slot));
        const u32 p = r[0], a = m_arity[p];
        if (a == 1)
        {
            m_unary.emplace_back(r[1], p);
            return;
        }
        const u32 first = n + static_cast<u32>(m_code.size());
        for (u32 pos = 0; pos < std::max<u32>(a, 1); ++pos)
        {
            m_code.push_back(m_pred_code[p] + pos);
            if (a == 0)
                break;
            m_edges.emplace_back(first + pos, r[1 + pos]);
            if (pos > 0)
                m_edges.emplace_back(first + pos - 1, first + pos);
        }
    };
    bits::for_each(e.state(), e.state_words(), [&](u64 slot) { add(AtomKind::Fluent, slot); });
    if (m_axioms)
        bits::for_each(e.derived(), e.derived_words(), [&](u64 slot) { add(AtomKind::Derived, slot); });

    const u32 V = n + static_cast<u32>(m_code.size());
    if (m_color.size() < V)
    {
        m_color.resize(V);
        m_next.resize(V);
        m_order.resize(V);
        m_offsets.resize(V + 1);
        m_count.resize(V, 0);
        m_used.resize(V, 0);
        m_stamp.resize(V, 0);
    }
    m_V = V;

    // object colours: (constant class, sorted unary fluent and derived predicates), ranked
    std::sort(m_unary.begin(), m_unary.end());
    std::fill(m_unary_begin.begin(), m_unary_begin.end(), 0u);
    for (auto [o, p] : m_unary)
        ++m_unary_begin[o + 1];
    for (u32 o = 0; o < n; ++o)
        m_unary_begin[o + 1] += m_unary_begin[o];
    auto less = [&](u32 a, u32 b)
    {
        if (m_const_class[a] != m_const_class[b])
            return m_const_class[a] < m_const_class[b];
        return std::lexicographical_compare(m_unary.begin() + m_unary_begin[a], m_unary.begin() + m_unary_begin[a + 1],
                                            m_unary.begin() + m_unary_begin[b], m_unary.begin() + m_unary_begin[b + 1],
                                            [](const auto& x, const auto& y) { return x.second < y.second; });
    };
    std::iota(m_order.begin(), m_order.begin() + n, 0u);
    std::sort(m_order.begin(), m_order.begin() + n, less);
    u32 c = 0;
    for (u32 i = 0; i < n; ++i)
    {
        if (i > 0 && less(m_order[i - 1], m_order[i]))
            ++c;
        m_color[m_order[i]] = c;
    }
    const u32 base = n == 0 ? 0 : c + 1;
    for (u32 v = n; v < V; ++v)
        m_color[v] = base + m_code[v - n];

    // adjacency (both directions)
    std::fill(m_offsets.begin(), m_offsets.begin() + V + 1, 0u);
    for (auto [a, b] : m_edges)
    {
        ++m_offsets[a + 1];
        ++m_offsets[b + 1];
    }
    for (u32 v = 0; v < V; ++v)
        m_offsets[v + 1] += m_offsets[v];
    const u32 E2 = m_offsets[V];
    if (m_adj.size() < E2)
    {
        m_adj.resize(E2);
        m_sig.resize(E2);
    }
    m_fill.assign(m_offsets.begin(), m_offsets.begin() + V);
    for (auto [a, b] : m_edges)
    {
        m_adj[m_fill[a]++] = b;
        m_adj[m_fill[b]++] = a;
    }
}

void SymmetryPruner::refine()
{
    // Colour refinement to the stable colouring: a vertex's next colour is the rank of (colour, sorted neighbour
    // colours); stop when a round splits no class (the classes then equal the previous round's).
    const u32 V = m_V;
    u32* color = m_color.data();
    u32* next = m_next.data();
    u32* order = m_order.data();
    const u32* off = m_offsets.data();
    u32* sig = m_sig.data();
    // dense initial colours and their number
    std::iota(order, order + V, 0u);
    std::sort(order, order + V, [&](u32 a, u32 b) { return color[a] < color[b]; });
    u32 classes = 0;
    for (u32 i = 0; i < V; ++i)
    {
        if (i > 0 && color[order[i - 1]] != color[order[i]])
            ++classes;
        next[order[i]] = classes;
    }
    classes = V == 0 ? 0 : classes + 1;
    std::swap(color, next);
    auto less = [&](u32 a, u32 b)
    {
        if (color[a] != color[b])
            return color[a] < color[b];
        return std::lexicographical_compare(sig + off[a], sig + off[a + 1], sig + off[b], sig + off[b + 1]);
    };
    while (classes < V)
    {
        for (u32 v = 0; v < V; ++v)
        {
            for (u32 j = off[v]; j < off[v + 1]; ++j)
                sig[j] = color[m_adj[j]];
            std::sort(sig + off[v], sig + off[v + 1]);
        }
        std::iota(order, order + V, 0u);
        std::sort(order, order + V, less);
        u32 c = 0;
        for (u32 i = 0; i < V; ++i)
        {
            if (i > 0 && less(order[i - 1], order[i]))
                ++c;
            next[order[i]] = c;
        }
        std::swap(color, next);
        if (c + 1 == classes)
            break;
        classes = c + 1;
    }
    if (color != m_color.data())
        std::copy(color, color + V, m_color.data());
}

void SymmetryPruner::compute(const Engine& e)
{
    build_graph(e);
    refine();
}

const u64* SymmetryPruner::masks(u32 schema)
{
    const u32 first = m_param_begin[schema], arity = m_param_begin[schema + 1] - first, OW = m_ow;
    std::fill(m_masks.begin(), m_masks.begin() + static_cast<usize>(arity) * OW, u64{0});
    if (m_tick > ~u32{0} - 2 * arity - 2)
    {
        std::fill(m_stamp.begin(), m_stamp.end(), 0u);
        m_tick = 0;
    }
    const u32* color = m_color.data();
    // n(C): the parameters whose static domain meets class C
    for (u32 i = 0; i < arity; ++i)
    {
        ++m_tick;
        for (u32 o : domain(schema, i))
        {
            const u32 c = color[o];
            if (m_stamp[c] != m_tick)
            {
                m_stamp[c] = m_tick;
                ++m_count[c];
            }
        }
    }
    // parameter i keeps the n(C) smallest objects of class C in its domain
    for (u32 i = 0; i < arity; ++i)
    {
        ++m_tick;
        u64* mask = m_masks.data() + static_cast<usize>(i) * OW;
        for (u32 o : domain(schema, i))
        {
            const u32 c = color[o];
            if (m_stamp[c] != m_tick)
            {
                m_stamp[c] = m_tick;
                m_used[c] = 0;
            }
            if (m_used[c] < m_count[c])
            {
                ++m_used[c];
                bits::set(mask, o);
            }
        }
    }
    for (u32 i = 0; i < arity; ++i)
        for (u32 o : domain(schema, i))
            m_count[color[o]] = 0;
    return m_masks.data();
}
}  // namespace mymyr::detail
