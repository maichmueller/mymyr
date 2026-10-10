// A*, eager and lazy (best_first.hpp): ports of mimir 0.16.3 astar_eager / astar_lazy onto the SoA nodes, shared
// stores and bucket or heap open lists (best_first_detail.hpp).

#include "best_first_detail.hpp"

#include <limits>

namespace mymyr::search
{
namespace
{
using namespace bf;
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

/// f - g0 of a node (the open-list key; computed the same way on push and pop, so stale entries are recognized).
[[nodiscard]] inline f64 fval(const Nodes& nodes, u32 id, f64 g0) { return (nodes.g[id] - g0) + nodes.h[id]; }

template<class S, class Q>
struct AStarEager
{
    void operator()(Context& c) const
    {
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Q open, goals;  // goal states wait in their own list: popped first among equal f (mimir's (f, GOAL) key)
        Transitions tr;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes() + goals.bytes()); };

        const StateView s0 = c.start.view();
        const u32 root = store.insert(s0).first;
        nodes.add(k_none, k_none, nullptr, 0, c.g0);
        if (c.unsolvable())
        {
            c.begin_search();
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        if (c.is_goal(s0))
        {
            c.begin_search();
            c.solved(nodes, root, s0, false);
            return finish();
        }
        const f64 h0 = c.evaluate(s0);
        r.initial_h = h0;
        c.begin_search();
        if (h0 == k_inf)
        {
            ++r.dead_ends;
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        nodes.h[root] = h0;
        nodes.set(root, Nodes::Evaluated);
        nodes.set_status(root, Nodes::Open);
        open.push(Q::key(fval(nodes, root, c.g0)), Q::key(h0), store.make_ref(root, s0));

        auto push = [&](u32 id, StateView v)
        {
            const u64 f = Q::key(fval(nodes, id, c.g0));
            if (nodes.has(id, Nodes::Goal))
                goals.push(f, 0, store.make_ref(id, v));
            else
                open.push(f, Q::key(nodes.h[id]), store.make_ref(id, v));
        };
        const u32 max_depth = c.o.control.budget.max_depth;
        for (;;)
        {
            if (!goals.empty() && (open.empty() || goals.top_key() <= open.top_key()))
            {
                u64 k;
                const u32 ref = goals.pop(k);
                const u32 id = store.id_of(ref);
                if (nodes.status(id) != Nodes::Open || Q::key(fval(nodes, id, c.g0)) != k)
                {
                    store.release(ref);  // reached more cheaply later: a newer entry exists
                    continue;
                }
                store.load(ref, cur);
                store.release(ref);
                c.solved(nodes, id, cur.view(), false);
                break;
            }
            if (open.empty())
            {
                r.status = SearchStatus::Exhausted;
                break;
            }
            if (!c.keep_going())
                break;
            u64 k;
            const u32 ref = open.pop(k);
            const u32 id = store.id_of(ref);
            if (nodes.status(id) != Nodes::Open || Q::key(fval(nodes, id, c.g0)) != k)
            {
                store.release(ref);  // closed, or reopened with a smaller f
                continue;
            }
            store.load(ref, cur);
            store.release(ref);
            nodes.set_status(id, Nodes::Closed);
            if (nodes.track_depth && nodes.depth[id] >= max_depth)
            {
                ++r.stats.pruned;
                continue;
            }
            ++r.stats.expanded;
            if (c.obs)
                c.obs->on_expand(id, cur.view());
            if (!expand(c, store, nodes, id, cur, next, tr, false))
                break;  // r.status is set
            if (c.batched)
                c.evaluate_fresh(tr, false, true);
            for (const Transitions::T& t : tr.t)
            {
                const u32 cid = t.child;
                if (c.obs)
                    c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                if (t.fresh)
                {
                    const StateView v = tr.view(t);
                    if (c.batched ? c.fresh_goal[tr.index(t)] != 0 : c.is_goal(v))
                        nodes.set(cid, Nodes::Goal);
                    const f64 hv = c.batched ? c.fresh_h[tr.index(t)] : c.evaluate(v);
                    if (hv == k_inf)
                    {
                        nodes.set_status(cid, Nodes::DeadEnd);
                        ++r.dead_ends;
                        ++r.stats.pruned;
                        continue;
                    }
                    nodes.h[cid] = hv;
                    nodes.set(cid, Nodes::Evaluated);
                    nodes.set_status(cid, Nodes::Open);
                    push(cid, v);
                    continue;
                }
                const f64 gc = t.g;
                const u8 st = nodes.status(cid);
                if (gc < nodes.g[cid] && (st == Nodes::Open || (st == Nodes::Closed && c.o.reopen)))
                {
                    r.reopened += st == Nodes::Closed;
                    nodes.relink(cid, id, t.schema, tr.binding(t), c.succ.arity(t.schema), gc);
                    nodes.set_status(cid, Nodes::Open);
                    push(cid, tr.view(t));
                }
            }
        }
        finish();
    }
};

template<class S, class Q>
struct AStarLazy
{
    void operator()(Context& c) const
    {
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        const bool requeue = c.o.lazy_requeue;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Alternating<Q> open(c.o.preferred_weight ? c.o.preferred_weight : 1, c.o.standard_weight);
        Q goals;  // requeue: evaluated goal states, accepted once no open entry has a smaller key
        Transitions tr;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes() + goals.bytes()); };

        const StateView s0 = c.start.view();
        const u32 root = store.insert(s0).first;
        nodes.add(k_none, k_none, nullptr, 0, c.g0);
        if (c.unsolvable())
        {
            c.begin_search();
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        if (c.is_goal(s0))
        {
            c.begin_search();
            c.solved(nodes, root, s0, false);
            return finish();
        }
        const f64 h0 = c.evaluate(s0);
        r.initial_h = h0;
        c.begin_search();
        if (h0 == k_inf)
        {
            ++r.dead_ends;
            r.status = SearchStatus::Unsolvable;
            return finish();
        }
        nodes.h[root] = h0;
        nodes.set(root, Nodes::Evaluated);
        nodes.set_status(root, Nodes::Open);
        open.list(0).push(Q::key(fval(nodes, root, c.g0)), Q::key(h0), store.make_ref(root, s0));

        const bool use_preferred = c.o.preferred_operators && c.h->provides_preferred();
        const u32 max_depth = c.o.control.budget.max_depth;
        u32 last_evaluated = root;
        for (;;)
        {
            if (!goals.empty() && (open.empty() || goals.top_key() <= open.min_key()))
            {
                u64 k;
                const u32 ref = goals.pop(k);
                const u32 id = store.id_of(ref);
                if (nodes.status(id) != Nodes::Open || Q::key(fval(nodes, id, c.g0)) != k)
                {
                    store.release(ref);
                    continue;
                }
                store.load(ref, cur);
                store.release(ref);
                c.solved(nodes, id, cur.view(), false);
                break;
            }
            if (open.empty())
            {
                r.status = SearchStatus::Exhausted;
                break;
            }
            if (!c.keep_going())
                break;
            u64 k;
            u32 from;
            const u32 ref = open.pop(k, from);
            const u32 id = store.id_of(ref);
            const u8 st = nodes.status(id);
            if (st == Nodes::Closed || st == Nodes::DeadEnd)
            {
                store.release(ref);
                continue;
            }
            store.load(ref, cur);
            const StateView sv = cur.view();
            if (!nodes.has(id, Nodes::Evaluated))
            {
                const f64 hv = c.evaluate(sv);
                last_evaluated = id;
                nodes.h[id] = hv;
                nodes.set(id, Nodes::Evaluated);
                if (hv == k_inf)
                {
                    store.release(ref);
                    nodes.set_status(id, Nodes::DeadEnd);
                    ++r.dead_ends;
                    ++r.stats.pruned;
                    continue;
                }
            }
            const f64 hp = nodes.h[id];
            if (requeue)
            {
                const u64 fk = Q::key(fval(nodes, id, c.g0));
                if (fk > k)
                {
                    open.list(from).push(fk, Q::key(hp), ref);  // the key was a lower bound: wait for the true f
                    continue;
                }
                if (nodes.has(id, Nodes::Goal))
                {
                    goals.push(fk, 0, ref);
                    continue;
                }
            }
            store.release(ref);
            if (!requeue && nodes.has(id, Nodes::Goal))
            {
                c.solved(nodes, id, sv, false);
                break;
            }
            nodes.set_status(id, Nodes::Closed);
            if (nodes.track_depth && nodes.depth[id] >= max_depth)
            {
                ++r.stats.pruned;
                continue;
            }
            ++r.stats.expanded;
            if (c.obs)
                c.obs->on_expand(id, sv);
            // the preferred operators are those of the heuristic's last evaluation: evaluate again when this state's
            // h was cached (a re-queued state)
            if (use_preferred && last_evaluated != id)
            {
                (void)c.evaluate(sv);
                last_evaluated = id;
            }
            const bool pref = use_preferred;
            if (!expand(c, store, nodes, id, cur, next, tr, pref))
                break;  // r.status is set
            for (const Transitions::T& t : tr.t)
            {
                const u32 cid = t.child;
                if (c.obs)
                    c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                const f64 gc = t.g;
                if (!t.fresh)
                {
                    const u8 cs = nodes.status(cid);
                    if (!(gc < nodes.g[cid] && (cs == Nodes::Open || (cs == Nodes::Closed && c.o.reopen))))
                        continue;
                    r.reopened += cs == Nodes::Closed;
                    nodes.relink(cid, id, t.schema, tr.binding(t), c.succ.arity(t.schema), gc);
                }
                else if (c.is_goal(tr.view(t)))
                    nodes.set(cid, Nodes::Goal);
                nodes.set_status(cid, Nodes::Open);
                // lower bound on f(child) for consistent h: h(child) >= h(parent) - cost
                const f64 key = (gc - c.g0) + (requeue ? std::max(0.0, hp - (gc - nodes.g[id])) : hp);
                const StateView v = tr.view(t);
                open.list(pref && t.preferred ? 0 : 1).push(Q::key(key), Q::key(hp), store.make_ref(cid, v));
            }
        }
        finish();
    }

};
}  // namespace

BestFirstResult astar_eager(const Task& task, const BestFirstOptions& options)
{
    return bf::run<AStarEager>(task, options, "astar_eager");
}

BestFirstResult astar_lazy(const Task& task, const BestFirstOptions& options)
{
    return bf::run<AStarLazy>(task, options, "astar_lazy");
}
}  // namespace mymyr::search
