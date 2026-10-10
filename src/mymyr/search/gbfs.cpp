// Greedy best-first search, eager and lazy (best_first.hpp): ports of mimir 0.16.3 gbfs_eager / gbfs_lazy onto the
// SoA nodes, shared stores and bucket or heap open lists (best_first_detail.hpp).

#include "best_first_detail.hpp"

#include <limits>

namespace mymyr::search
{
namespace
{
using namespace bf;
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

/// The start of every GBFS: the root node, the static goal, the start goal test and h(start). False when the
/// search is over already (the status is set).
template<class S>
bool start_search(Context& c, S& store, Nodes& nodes, u32& root, f64& h0)
{
    const StateView s0 = c.start.view();
    root = store.insert(s0).first;
    nodes.add(k_none, k_none, nullptr, 0, c.g0);
    if (c.unsolvable())
    {
        c.begin_search();
        c.r.status = SearchStatus::Unsolvable;
        return false;
    }
    if (c.is_goal(s0))
    {
        c.begin_search();
        c.solved(nodes, root, s0, true);
        return false;
    }
    h0 = c.evaluate(s0);
    c.r.initial_h = h0;
    c.begin_search();
    if (h0 == k_inf)
    {
        ++c.r.dead_ends;
        c.r.status = SearchStatus::Unsolvable;
        return false;
    }
    nodes.h[root] = h0;
    nodes.set(root, Nodes::Evaluated);
    nodes.set_status(root, Nodes::Open);
    return true;
}

template<class S, class Q>
struct GbfsEager
{
    void operator()(Context& c) const
    {
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Q open;
        Transitions tr;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes()); };
        u32 root = 0;
        f64 h0 = 0;
        if (!start_search(c, store, nodes, root, h0))
            return finish();
        const StateView s0 = c.start.view();
        open.push(Q::key(h0), Q::key(0), store.make_ref(root, s0));
        const u32 max_depth = c.o.control.budget.max_depth;
        for (;;)
        {
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
            if (nodes.status(id) != Nodes::Open)
            {
                store.release(ref);
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
                c.evaluate_fresh(tr, true, false);
            bool done = false;
            for (const Transitions::T& t : tr.t)
            {
                const u32 cid = t.child;
                if (c.obs)
                    c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                if (!t.fresh)
                    continue;  // no reopening
                const StateView v = tr.view(t);
                nodes.set_status(cid, Nodes::Open);
                if (c.batched ? c.fresh_goal[tr.index(t)] != 0 : c.is_goal(v))  // early goal test
                {
                    nodes.set(cid, Nodes::Goal);
                    c.solved(nodes, cid, v, true);
                    if (c.obs)  // the transitions after the goal's were generated too
                        for (const Transitions::T* u = &t + 1; u != tr.t.data() + tr.t.size(); ++u)
                            c.obs->on_generate(id, c.action(u->schema, tr.binding(*u)), u->child, tr.view(*u),
                                               u->fresh);
                    done = true;
                    break;
                }
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
                open.push(Q::key(hv), Q::key(nodes.g[cid] - c.g0), store.make_ref(cid, v));
            }
            if (done)
                break;
        }
        finish();
    }
};

template<class S, class Q>
struct GbfsLazy
{
    void operator()(Context& c) const
    {
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        // mimir's weights {.., 64, 1}: 64 pops from the preferred list per pop from the standard one
        Alternating<Q> open(c.o.preferred_weight ? c.o.preferred_weight : 64, c.o.standard_weight);
        Transitions tr;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes()); };
        u32 root = 0;
        f64 h0 = 0;
        if (!start_search(c, store, nodes, root, h0))
            return finish();
        const StateView s0 = c.start.view();
        open.list(1).push(Q::key(h0), Q::key(0), store.make_ref(root, s0));
        const bool use_preferred = c.o.preferred_operators && c.h->provides_preferred();
        const u32 max_depth = c.o.control.budget.max_depth;
        u32 last_evaluated = root;
        for (;;)
        {
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
            if (nodes.status(id) != Nodes::Open)
            {
                store.release(ref);
                continue;
            }
            store.load(ref, cur);
            store.release(ref);
            const StateView sv = cur.view();
            if (!nodes.has(id, Nodes::Evaluated) || (use_preferred && last_evaluated != id))
            {
                nodes.h[id] = c.evaluate(sv);
                nodes.set(id, Nodes::Evaluated);
                last_evaluated = id;
            }
            const f64 hp = nodes.h[id];
            if (hp == k_inf)
            {
                nodes.set_status(id, Nodes::DeadEnd);
                ++r.dead_ends;
                ++r.stats.pruned;
                continue;
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
            if (!expand(c, store, nodes, id, cur, next, tr, use_preferred))
                break;  // r.status is set
            bool done = false;
            for (const Transitions::T& t : tr.t)
            {
                const u32 cid = t.child;
                if (c.obs)
                    c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                if (!t.fresh)
                    continue;
                const StateView v = tr.view(t);
                nodes.set_status(cid, Nodes::Open);
                if (c.is_goal(v))
                {
                    nodes.set(cid, Nodes::Goal);
                    c.solved(nodes, cid, v, true);
                    if (c.obs)  // the transitions after the goal's were generated too
                        for (const Transitions::T* u = &t + 1; u != tr.t.data() + tr.t.size(); ++u)
                            c.obs->on_generate(id, c.action(u->schema, tr.binding(*u)), u->child, tr.view(*u),
                                               u->fresh);
                    done = true;
                    break;
                }
                open.list(use_preferred && t.preferred ? 0 : 1)
                    .push(Q::key(hp), Q::key(nodes.g[cid] - c.g0), store.make_ref(cid, v));
            }
            if (done)
                break;
        }
        finish();
    }
};
}  // namespace

BestFirstResult gbfs_eager(const Task& task, const BestFirstOptions& options)
{
    return bf::run<GbfsEager>(task, options, "gbfs_eager");
}

BestFirstResult gbfs_lazy(const Task& task, const BestFirstOptions& options)
{
    return bf::run<GbfsLazy>(task, options, "gbfs_lazy");
}
}  // namespace mymyr::search
