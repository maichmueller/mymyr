// Object graphs: a port of mimir's src/datasets/object_graph.cpp (see datasets/object_graph.hpp).

#include "mymyr/datasets/object_graph.hpp"

#include "mymyr/core/hash.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <numeric>

namespace mymyr::datasets
{
namespace
{
using formalism::PredKind;

/// A ground atom or goal literal with a vertex structure (arity 0 or >= 2).
struct Fixed
{
    u32 pred = 0;
    bool literal = false;
    bool positive = true;
    std::vector<u32> args;
};
}  // namespace

u64 ObjectGraph::color_key(u32 c) const noexcept
{
    const std::span<const u32> s = palette(c);
    u64 h = hash::mix64(0x243f6a8885a308d3ULL ^ s.size());
    for (u32 x : s)
        h = hash::combine(h, x);
    return h;
}

struct ObjectGraphBuilder::Impl
{
    const Task& task;
    const formalism::TaskData& d;
    u32 n = 0;
    std::vector<std::vector<u32>> static_unary;                  // per object: predicates of its static unary atoms
    std::vector<std::vector<std::pair<u32, u32>>> goal_unary;    // per object: (predicate, polarity) of unary goal literals
    std::vector<Fixed> static_atoms;                             // static atoms of arity 0 and >= 2
    std::vector<Fixed> goal_literals;                            // goal literals of arity 0 and >= 2 (static, fluent, derived)
    // scratch
    std::vector<std::vector<u32>> unary;  // per object: unary atom predicates of the current state
    std::vector<u32> seq, seq_off, order;
    std::vector<std::pair<u32, u32>> edges;

    explicit Impl(const Task& t) : task(t), d(t.data()), n(t.data().num_objects())
    {
        static_unary.resize(n);
        goal_unary.resize(n);
        unary.resize(n);
        for (const formalism::GroundAtom& a : d.static_init)
        {
            const auto objs = d.objects_of(a);
            if (objs.size() == 1)
                static_unary[objs[0].v].push_back(a.pred.v);
            else
            {
                Fixed f{a.pred.v, false, true, {}};
                for (ObjectId o : objs)
                    f.args.push_back(o.v);
                static_atoms.push_back(std::move(f));
            }
        }
        for (PredKind kind : {PredKind::Static, PredKind::Fluent, PredKind::Derived})
            for (const formalism::Literal& l : d.literals_of(d.goal))
            {
                if (d.predicate(l.pred).kind != kind)
                    continue;
                const auto terms = d.terms_of(l);
                if (terms.size() == 1)
                    goal_unary[formalism::term_object(terms[0]).v].emplace_back(l.pred.v, l.positive ? 1u : 0u);
                else
                {
                    Fixed f{l.pred.v, true, l.positive, {}};
                    for (formalism::Term t : terms)
                        f.args.push_back(formalism::term_object(t).v);
                    goal_literals.push_back(std::move(f));
                }
            }
    }

    // colour sequences (object_graph.hpp): one per vertex, flat
    u32 begin_vertex()
    {
        seq_off.push_back(static_cast<u32>(seq.size()));
        return static_cast<u32>(seq_off.size() - 1);
    }
    void add_structure(u32 pred, const u32* args, u32 arity, bool literal, bool positive)
    {
        if (arity == 0)
        {
            begin_vertex();
            if (literal)
                seq.insert(seq.end(), {3u, pred, positive ? 1u : 0u});
            else
                seq.insert(seq.end(), {1u, pred});
            return;
        }
        for (u32 pos = 0; pos < arity; ++pos)
        {
            const u32 v = begin_vertex();
            if (literal)
                seq.insert(seq.end(), {4u, pred, pos, positive ? 1u : 0u});
            else
                seq.insert(seq.end(), {2u, pred, pos});
            edges.emplace_back(v, args[pos]);
            if (pos > 0)
                edges.emplace_back(v - 1, v);
        }
    }

    void build(StateView s, ObjectGraph& g)
    {
        const AtomIndex& atoms = task.atoms();
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
        const bool derived = task.has_axioms();
        if (derived)
            succ.prepare(s);
        seq.clear();
        seq_off.clear();
        edges.clear();
        for (u32 o = 0; o < n; ++o)
            unary[o] = static_unary[o];
        // true fluent / derived atoms: unary ones colour their object, the others become vertices (after the objects)
        std::vector<Fixed> fluent, der;
        auto collect = [&](AtomKind kind, u32 slot, std::vector<Fixed>& out)
        {
            const u32* r = atoms.record(kind, slot);
            const u32 arity = d.predicates[r[0]].arity;
            if (arity == 1)
                unary[r[1]].push_back(r[0]);
            else
                out.push_back(Fixed{r[0], false, true, std::vector<u32>(r + 1, r + 1 + arity)});
        };
        bits::for_each(s.w, s.nw, [&](u64 slot) { collect(AtomKind::Fluent, static_cast<u32>(slot), fluent); });
        if (derived)
        {
            const detail::Engine& e = succ.engine();
            bits::for_each(e.derived(), e.derived_words(), [&](u64 slot) { collect(AtomKind::Derived, static_cast<u32>(slot), der); });
        }
        // 1. objects
        for (u32 o = 0; o < n; ++o)
        {
            std::vector<u32>& a = unary[o];
            std::sort(a.begin(), a.end());
            std::vector<std::pair<u32, u32>> lits = goal_unary[o];
            std::sort(lits.begin(), lits.end());
            begin_vertex();
            seq.push_back(0);
            seq.push_back(static_cast<u32>(a.size()));
            seq.insert(seq.end(), a.begin(), a.end());
            seq.push_back(static_cast<u32>(lits.size()));
            for (auto [p, pol] : lits)
                seq.insert(seq.end(), {p, pol});
        }
        // 2. atoms: static, fluent, derived; 3. goal literals
        for (const std::vector<Fixed>* list : {&static_atoms, &fluent, &der, &goal_literals})
            for (const Fixed& f : *list)
                add_structure(f.pred, f.args.data(), static_cast<u32>(f.args.size()), f.literal, f.positive);
        const u32 V = static_cast<u32>(seq_off.size());
        seq_off.push_back(static_cast<u32>(seq.size()));
        // palette: distinct sequences in lexicographic order
        order.resize(V);
        std::iota(order.begin(), order.end(), 0u);
        auto less = [&](u32 a, u32 b)
        {
            return std::lexicographical_compare(seq.begin() + seq_off[a], seq.begin() + seq_off[a + 1], seq.begin() + seq_off[b],
                                                seq.begin() + seq_off[b + 1]);
        };
        std::stable_sort(order.begin(), order.end(), less);
        g.num_objects = n;
        g.color.assign(V, 0);
        g.palette_offsets.assign(1, 0);
        g.palette_values.clear();
        for (u32 i = 0; i < V; ++i)
        {
            const u32 v = order[i];
            if (i == 0 || less(order[i - 1], v))
            {
                g.palette_values.insert(g.palette_values.end(), seq.begin() + seq_off[v], seq.begin() + seq_off[v + 1]);
                g.palette_offsets.push_back(static_cast<u32>(g.palette_values.size()));
            }
            g.color[v] = static_cast<u32>(g.palette_offsets.size() - 2);
        }
        // adjacency (both directions, sorted per vertex)
        g.offsets.assign(V + 1, 0);
        for (auto [a, b] : edges)
        {
            ++g.offsets[a + 1];
            ++g.offsets[b + 1];
        }
        for (u32 v = 0; v < V; ++v)
            g.offsets[v + 1] += g.offsets[v];
        g.neighbors.assign(g.offsets[V], 0);
        std::vector<u64> fill(g.offsets.begin(), g.offsets.end() - 1);
        for (auto [a, b] : edges)
        {
            g.neighbors[fill[a]++] = b;
            g.neighbors[fill[b]++] = a;
        }
        for (u32 v = 0; v < V; ++v)
            std::sort(g.neighbors.begin() + static_cast<i64>(g.offsets[v]), g.neighbors.begin() + static_cast<i64>(g.offsets[v + 1]));
    }
};

ObjectGraphBuilder::ObjectGraphBuilder(const Task& task) : m(std::make_unique<Impl>(task)) {}
ObjectGraphBuilder::~ObjectGraphBuilder() = default;

ObjectGraph ObjectGraphBuilder::build(StateView s)
{
    ObjectGraph g;
    m->build(s, g);
    return g;
}

void ObjectGraphBuilder::build(StateView s, ObjectGraph& out) { m->build(s, out); }

ObjectGraph object_graph(const Task& task, StateView s)
{
    ObjectGraphBuilder b(task);
    return b.build(s);
}
}  // namespace mymyr::datasets
