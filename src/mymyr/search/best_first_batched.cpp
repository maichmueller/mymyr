// Eager A* and eager GBFS with batched expansions (BestFirstOptions::expansion_batch, best_first.hpp): each round pops
// and expands up to expansion_batch open states, then goal-tests and evaluates the new successors of all of them in one
// heuristic call and handles the transitions expansion by expansion, as the one-expansion searches handle one.

#include "best_first_batched.hpp"

#include "best_first_detail.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::search::bf
{
namespace
{
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

[[nodiscard]] inline f64 fval(const Nodes& nodes, u32 id, f64 g0, f64 weight)
{
    return (nodes.g[id] - g0) + weight * nodes.h[id];
}

/// The expansions of one round and the goal flags and h values of their new successors.
struct Round
{
    std::vector<u32> ids;            // the expanded nodes, in pop order
    std::vector<Transitions> tr;     // their transitions
    std::vector<usize> first;        // the flat index of each expansion's first transition
    std::vector<u8> goal;            // per flat transition: a goal state (fresh transitions only)
    std::vector<f64> h;              // per flat transition: h (fresh transitions that were evaluated)
    std::vector<StateView> views;
    std::vector<usize> where;
    std::vector<f64> values;

    [[nodiscard]] usize at(usize j, const Transitions::T& t) const { return first[j] + tr[j].index(t); }

    /// The transitions of the next expansion (buffers are kept across rounds).
    [[nodiscard]] Transitions& next()
    {
        if (tr.size() == ids.size())
            tr.emplace_back();
        return tr[ids.size()];
    }

    /// Goal-tests the fresh transitions of the round's expansions in order and evaluates them in one call: GBFS
    /// (`stop_at_goal`) those before the first goal state, A* all of them (goal states too) but those a cost bound
    /// prunes before evaluation.
    void evaluate(Context& c, bool stop_at_goal)
    {
        const usize n = ids.size();
        first.assign(n + 1, 0);
        for (usize j = 0; j < n; ++j)
            first[j + 1] = first[j] + tr[j].t.size();
        goal.assign(first[n], 0);
        h.assign(first[n], 0);
        views.clear();
        where.clear();
        bool stop = false;
        for (usize j = 0; j < n && !stop; ++j)
            for (const Transitions::T& t : tr[j].t)
            {
                if (!t.fresh)
                    continue;
                const StateView v = tr[j].view(t);
                const usize i = at(j, t);
                goal[i] = c.is_goal(v);
                if (goal[i] && stop_at_goal)
                {
                    stop = true;
                    break;
                }
                if (c.bounded && c.beyond_bound(t.g, 0))
                    continue;
                views.push_back(v);
                where.push_back(i);
            }
        if (views.empty())
            return;
        values.resize(views.size());
        c.evaluate(views, values);
        for (usize k = 0; k < views.size(); ++k)
            h[where[k]] = values[k];
    }
};

/// The start of batched A*: the static goal, the cost bound, the start goal test and h(start). False when the search
/// is over already (the status is set).
bool astar_start(Context& c, Nodes& nodes, u32 root, StateView s0)
{
    BestFirstResult& r = c.r;
    if (c.unsolvable())
    {
        c.begin_search();
        r.status = SearchStatus::Unsolvable;
        return false;
    }
    if (c.beyond_bound(c.g0, 0))
    {
        c.begin_search();
        r.status = SearchStatus::Exhausted;
        return false;
    }
    if (c.is_goal(s0))
    {
        c.begin_search();
        c.solved(nodes, root, s0, false);
        return false;
    }
    const f64 h0 = c.evaluate(s0);
    r.initial_h = h0;
    c.begin_search();
    if (h0 == k_inf)
    {
        ++r.dead_ends;
        r.status = SearchStatus::Unsolvable;
        return false;
    }
    nodes.h[root] = h0;
    nodes.set(root, Nodes::Evaluated);
    if (c.beyond_bound(c.g0, h0))
    {
        r.status = SearchStatus::Exhausted;
        return false;
    }
    nodes.set_status(root, Nodes::Open);
    return true;
}

void check_batched(const Context& c)
{
    if (c.path_dependent)
        throw std::invalid_argument("mymyr " + c.r.algorithm + ": expansion_batch > 1 needs a heuristic that is not "
                                    "path-dependent");
}

template<class S, class Q>
struct AStarEagerBatched
{
    void operator()(Context& c) const
    {
        check_batched(c);
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Q open, goals;
        Round round;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes() + goals.bytes()); };

        const StateView s0 = c.start.view();
        const u32 root = store.insert(s0).first;
        nodes.add(k_none, k_none, nullptr, 0, c.g0);
        if (!astar_start(c, nodes, root, s0))
            return finish();
        open.push(Q::key(fval(nodes, root, c.g0, c.weight)), Q::key(nodes.h[root]), store.make_ref(root, s0));

        auto push = [&](u32 id, StateView v)
        {
            const u64 f = Q::key(fval(nodes, id, c.g0, c.weight));
            if (nodes.has(id, Nodes::Goal))
                goals.push(f, 0, store.make_ref(id, v));
            else
                open.push(f, Q::key(nodes.h[id]), store.make_ref(id, v));
        };
        f64 last_f = -k_inf;
        auto layer = [&](u32 id)
        {
            if (const f64 f = fval(nodes, id, c.g0, c.weight); f > last_f)
            {
                last_f = f;
                r.expanded_until_last_jump = r.stats.expanded;
            }
        };
        auto dead_end = [&](u32 cid)
        {
            nodes.set_status(cid, Nodes::DeadEnd);
            ++r.dead_ends;
            ++r.stats.pruned;
        };
        const u32 max_depth = c.o.control.budget.max_depth;
        const usize k = c.o.expansion_batch;
        for (;;)
        {
            // pops: a goal state is accepted only before the round's first expansion, so that it is checked against
            // the open list with the successors of every earlier round pushed
            round.ids.clear();
            bool stop = false;
            while (round.ids.size() < k)
            {
                if (!goals.empty() && (open.empty() || goals.top_key() <= open.top_key()))
                {
                    if (!round.ids.empty())
                        break;
                    u64 key;
                    const u32 ref = goals.pop(key);
                    const u32 id = store.id_of(ref);
                    if (nodes.status(id) != Nodes::Open || Q::key(fval(nodes, id, c.g0, c.weight)) != key)
                    {
                        store.release(ref);
                        continue;
                    }
                    store.load(ref, cur);
                    store.release(ref);
                    layer(id);
                    c.solved(nodes, id, cur.view(), false);
                    stop = true;
                    break;
                }
                if (open.empty())
                {
                    if (round.ids.empty())
                    {
                        r.status = SearchStatus::Exhausted;
                        stop = true;
                    }
                    break;
                }
                if (!c.keep_going())
                {
                    stop = true;
                    break;
                }
                u64 key;
                const u32 ref = open.pop(key);
                const u32 id = store.id_of(ref);
                if (nodes.status(id) != Nodes::Open || Q::key(fval(nodes, id, c.g0, c.weight)) != key)
                {
                    store.release(ref);
                    continue;
                }
                store.load(ref, cur);
                store.release(ref);
                nodes.set_status(id, Nodes::Closed);
                layer(id);
                if (nodes.track_depth && nodes.depth[id] >= max_depth)
                {
                    ++r.stats.pruned;
                    continue;
                }
                ++r.stats.expanded;
                if (c.obs)
                    c.obs->on_expand(id, cur.view());
                if (!expand(c, store, nodes, id, cur, next, round.next(), false))
                {
                    stop = true;  // r.status is set
                    break;
                }
                round.ids.push_back(id);
            }
            if (stop)
                break;
            round.evaluate(c, false);
            for (usize j = 0; j < round.ids.size(); ++j)
            {
                const u32 id = round.ids[j];
                const Transitions& tr = round.tr[j];
                for (const Transitions::T& t : tr.t)
                {
                    const u32 cid = t.child;
                    if (c.obs)
                        c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                    if (t.fresh)
                    {
                        const usize i = round.at(j, t);
                        if (round.goal[i])
                            nodes.set(cid, Nodes::Goal);
                        if (c.bounded && c.beyond_bound(t.g, 0))
                        {
                            ++r.stats.pruned;  // stays New: a cheaper path may reach it again
                            continue;
                        }
                        const f64 hv = round.h[i];
                        if (hv == k_inf)
                        {
                            dead_end(cid);
                            continue;
                        }
                        nodes.h[cid] = hv;
                        nodes.set(cid, Nodes::Evaluated);
                        if (c.bounded && c.beyond_bound(t.g, hv))
                        {
                            ++r.stats.pruned;
                            continue;
                        }
                        nodes.set_status(cid, Nodes::Open);
                        push(cid, tr.view(t));
                        continue;
                    }
                    const f64 gc = t.g;
                    const u8 st = nodes.status(cid);
                    if (st == Nodes::New)  // pruned by the cost bound when reached before
                    {
                        if (!(gc < nodes.g[cid]))
                            continue;
                        nodes.relink(cid, id, t.schema, tr.binding(t), c.succ.arity(t.schema), gc);
                        if (c.beyond_bound(gc, 0))
                        {
                            ++r.stats.pruned;
                            continue;
                        }
                        if (!nodes.has(cid, Nodes::Evaluated))
                        {
                            const f64 hv = c.evaluate(tr.view(t));
                            if (hv == k_inf)
                            {
                                dead_end(cid);
                                continue;
                            }
                            nodes.h[cid] = hv;
                            nodes.set(cid, Nodes::Evaluated);
                        }
                        if (c.beyond_bound(gc, nodes.h[cid]))
                        {
                            ++r.stats.pruned;
                            continue;
                        }
                        nodes.set_status(cid, Nodes::Open);
                        push(cid, tr.view(t));
                        continue;
                    }
                    if (gc < nodes.g[cid] && (st == Nodes::Open || (st == Nodes::Closed && c.o.reopen)))
                    {
                        r.reopened += st == Nodes::Closed;
                        nodes.relink(cid, id, t.schema, tr.binding(t), c.succ.arity(t.schema), gc);
                        nodes.set_status(cid, Nodes::Open);
                        push(cid, tr.view(t));
                    }
                }
            }
        }
        finish();
    }
};

template<class S, class Q>
struct GbfsEagerBatched
{
    void operator()(Context& c) const
    {
        check_batched(c);
        BestFirstResult& r = c.r;
        r.store = S::k_name;
        r.queue = Q::k_name;
        S store(c.task);
        Nodes nodes;
        nodes.track_depth = c.o.control.budget.max_depth != std::numeric_limits<u32>::max();
        Q open;
        Round round;
        Cur cur;
        std::vector<u64> next;
        auto finish = [&] { c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes()); };

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
            c.solved(nodes, root, s0, true);
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
        open.push(Q::key(h0), Q::key(0), store.make_ref(root, s0));
        const u32 max_depth = c.o.control.budget.max_depth;
        const usize k = c.o.expansion_batch;
        for (;;)
        {
            round.ids.clear();
            bool stop = false;
            while (round.ids.size() < k)
            {
                if (open.empty())
                {
                    if (round.ids.empty())
                    {
                        r.status = SearchStatus::Exhausted;
                        stop = true;
                    }
                    break;
                }
                if (!c.keep_going())
                {
                    stop = true;
                    break;
                }
                u64 key;
                const u32 ref = open.pop(key);
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
                if (!expand(c, store, nodes, id, cur, next, round.next(), false))
                {
                    stop = true;
                    break;
                }
                round.ids.push_back(id);
            }
            if (stop)
                break;
            round.evaluate(c, true);
            bool done = false;
            for (usize j = 0; j < round.ids.size(); ++j)
            {
                const u32 id = round.ids[j];
                const Transitions& tr = round.tr[j];
                for (const Transitions::T& t : tr.t)
                {
                    const u32 cid = t.child;
                    if (c.obs)
                        c.obs->on_generate(id, c.action(t.schema, tr.binding(t)), cid, tr.view(t), t.fresh);
                    if (done || !t.fresh)
                        continue;  // no reopening; after the goal only the observer's events remain
                    const StateView v = tr.view(t);
                    nodes.set_status(cid, Nodes::Open);
                    const usize i = round.at(j, t);
                    if (round.goal[i])  // early goal test
                    {
                        nodes.set(cid, Nodes::Goal);
                        c.solved(nodes, cid, v, true);
                        done = true;
                        if (!c.obs)
                            break;
                        continue;
                    }
                    const f64 hv = round.h[i];
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
                if (done && !c.obs)
                    break;
            }
            if (done)
                break;
        }
        finish();
    }
};
}  // namespace

BestFirstResult astar_eager_batched(const Task& task, const BestFirstOptions& options)
{
    return run<AStarEagerBatched>(task, options, "astar_eager", true);
}

BestFirstResult gbfs_eager_batched(const Task& task, const BestFirstOptions& options)
{
    return run<GbfsEagerBatched>(task, options, "gbfs_eager");
}
}  // namespace mymyr::search::bf
