#include "mymyr/search/astar_iw.hpp"

#include "astar_iw_novelty.hpp"
#include "best_first_detail.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace mymyr::search
{
namespace
{
using namespace bf;
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

struct NonUnitCost : std::invalid_argument
{
    NonUnitCost() : std::invalid_argument("mymyr: AStarIW requires every generated action to have unit cost") {}
};

struct Entry
{
    f64 f, h, g;
    u32 id, ref;
    bool root_successor;
    auto key() const { return std::tuple{f, h, g, id}; }
    bool operator>(const Entry& other) const { return key() > other.key(); }
};

class Queue : public std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>
{
public:
    [[nodiscard]] u64 bytes() const noexcept { return this->c.capacity() * sizeof(Entry); }
};

f64 evaluate(Context& c, StateView s)
{
    const f64 h = c.evaluate(s);
    if (std::isnan(h) || h == -k_inf)
        throw std::invalid_argument("mymyr: AStarIW heuristic must return a finite value or positive infinity");
    return h;
}

template<class Store>
void run(Context& c, const AStarIwOptions& o, AStarIwResult& r)
{
    Store store(c.task);
    r.store = Store::k_name;
    r.queue = "heap";
    Nodes nodes;
    nodes.track_depth = o.control.budget.max_depth != std::numeric_limits<u32>::max();
    Queue open;
    Cur cur;
    std::vector<u64> next;
    Transitions tr;
    std::optional<detail::MinimumGBackend> novelty;
    auto finish = [&]
    {
        r.novelty.table_bytes = novelty ? novelty->bytes() : 0;
        c.finish(store.size(), store.bytes() + nodes.bytes() + open.bytes() + r.novelty.table_bytes);
    };
    const StateView s0 = c.start.view();
    const u32 root = store.insert(s0).first;
    nodes.add(k_none, k_none, nullptr, 0, c.g0);
    c.begin_search();
    if (c.unsolvable())
    {
        r.status = SearchStatus::Unsolvable;
        return finish();
    }
    if (c.is_goal(s0))
    {
        c.solved(nodes, root, s0, false);
        return finish();
    }
    const f64 h0 = evaluate(c, s0);
    r.initial_h = h0;
    if (h0 == k_inf)
    {
        ++r.dead_ends;
        r.status = SearchStatus::Unsolvable;
        return finish();
    }
    novelty.emplace(c.task, o);
    novelty->initialize(s0, c.g0);
    nodes.set_status(root, Nodes::Open);
    nodes.h[root] = h0;
    open.push({c.g0 + o.weight * h0, h0, c.g0, root, store.make_ref(root, s0), false});
    while (!open.empty())
    {
        if (!c.keep_going())
            break;
        const Entry e = open.top();
        open.pop();
        if (e.g != nodes.g[e.id] || nodes.status(e.id) == Nodes::Closed || nodes.status(e.id) == Nodes::DeadEnd)
        {
            store.release(e.ref);
            ++r.novelty.stale_g;
            continue;
        }
        store.load(e.ref, cur);
        store.release(e.ref);
        const bool goal = c.is_goal(cur.view());
        const bool exempt = e.root_successor && o.allow_non_novel_root_goal;
        if (exempt && goal)
        {
            c.solved(nodes, e.id, cur.view(), false);
            break;
        }
        if (e.id != root && (exempt || novelty->may_have_stale_novelty()))
        {
            ++r.novelty.pop_tests;
            if (!novelty->test_at_g(cur.view(), e.g))
            {
                nodes.set_status(e.id, Nodes::Closed);
                ++r.novelty.stale;
                ++r.stats.pruned;
                continue;
            }
        }
        if (goal)
        {
            c.solved(nodes, e.id, cur.view(), false);
            break;
        }
        nodes.set_status(e.id, Nodes::Closed);
        if (nodes.track_depth && nodes.depth[e.id] >= o.control.budget.max_depth)
        {
            ++r.stats.pruned;
            continue;
        }
        ++r.stats.expanded;
        if (c.obs)
            c.obs->on_expand(e.id, cur.view());
        // Heuristic evaluation may reuse the successor engine, so buffer each transition before evaluating any.
        tr.clear();
        c.succ.prepare(cur.view());
        bool full = false;
        c.generate([&](u32 schema, const ObjectId* binding, const Delta& delta) -> bool
        {
            const f64 g = c.g_next(e.g, delta);
            if (g - e.g != 1)
                throw NonUnitCost();
            ++r.stats.generated;
            const u32 nw = store.successor(cur, delta, next, c.succ);
            const StateView child{next.data(), nw};
            if (c.blocked(next.data(), nw, nullptr))
            {
                ++r.stats.pruned;
                if (c.obs)
                {
                    const Action a = c.action(schema, binding);
                    c.obs->on_prune(e.id, a, child);
                    c.obs->on_transition(e.id, a, ~u64{0}, child, TransitionOutcome::Pruned);
                }
                return true;
            }
            const auto [id, fresh] = store.insert_successor(e.id, cur, next.data(), nw, delta);
            if (fresh)
                nodes.add(k_none, k_none, nullptr, 0, k_inf);
            const u32 trimmed = bits::trimmed_size(next.data(), nw);
            tr.t.push_back({schema, id, tr.bind.size(), tr.words.size(), trimmed, g, fresh, false});
            tr.bind.insert(tr.bind.end(), binding, binding + c.succ.arity(schema));
            tr.words.insert(tr.words.end(), next.data(), next.data() + trimmed);
            if (fresh && store.size() >= c.max_states)
            {
                full = true;
                return false;
            }
            return true;
        });
        for (const Transitions::T& t : tr.t)
        {
            const StateView child = tr.view(t);
            const Action action = c.obs ? c.action(t.schema, tr.binding(t)) : Action{};
            auto outcome = [&](TransitionOutcome kind)
            {
                if (c.obs)
                    c.obs->on_transition(e.id, action, t.child, child, kind);
            };
            if (c.obs)
                c.obs->on_generate(e.id, action, t.child, child, t.fresh);
            if (full && &t == &tr.t.back())
            {
                outcome(TransitionOutcome::Pruned);
                break;
            }
            if (!(t.g < nodes.g[t.child]))
            {
                outcome(TransitionOutcome::Duplicate);
                continue;
            }
            const bool was_closed = nodes.status(t.child) == Nodes::Closed;
            nodes.relink(t.child, e.id, t.schema, tr.binding(t), c.succ.arity(t.schema), t.g);
            const bool root_successor = e.id == root;
            const bool novelty_exempt = root_successor && o.allow_non_novel_root_goal;
            if (o.probe_novelty_before_heuristic && !novelty_exempt)
            {
                ++r.novelty.probes;
                if (!novelty->would_improve(cur.view(), child, t.g))
                {
                    nodes.set_status(t.child, Nodes::Closed);
                    ++r.novelty.rejected;
                    ++r.stats.pruned;
                    outcome(TransitionOutcome::Pruned);
                    continue;
                }
            }
            const f64 h = evaluate(c, child);
            if (h == k_inf)
            {
                nodes.set_status(t.child, Nodes::DeadEnd);
                ++r.dead_ends;
                ++r.stats.pruned;
                outcome(TransitionOutcome::DeadEnd);
                continue;
            }
            ++r.novelty.updates;
            if (!novelty->test_and_update(cur.view(), child, t.g) && !novelty_exempt)
            {
                nodes.set_status(t.child, Nodes::Closed);
                ++r.novelty.rejected;
                ++r.stats.pruned;
                outcome(TransitionOutcome::Pruned);
                continue;
            }
            r.reopened += was_closed;
            nodes.h[t.child] = h;
            nodes.set_status(t.child, Nodes::Open);
            open.push({t.g + o.weight * h, h, t.g, t.child, store.make_ref(t.child, child), root_successor});
            outcome(was_closed ? TransitionOutcome::Reopened : TransitionOutcome::Opened);
        }
        if (full)
        {
            r.status = SearchStatus::OutOfStates;
            break;
        }
    }
    finish();
}
}  // namespace

AStarIwResult astar_iw(const Task& task, const AStarIwOptions& o)
{
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: AStarIW does not support tasks with numeric fluents (IW and SIW do)");
    const heuristics::ActionCosts costs(task);
    if (!costs.unit())
    {
        for (u32 s = 0; s < task.data().schemas.size(); ++s)
            if (costs.constant(s) && costs.cost(s, nullptr) != 1)
                throw std::invalid_argument("mymyr: AStarIW requires every action to have unit cost");
    }
    AStarIwResult r;
    r.algorithm = "astar_iw";
    auto failed = [&](std::string message)
    {
        r.status = SearchStatus::Failed;
        r.message = "mymyr: " + message;
        return r;
    };
    if (o.width < 1 || o.width > novelty::k_max_arity || (o.features != AStarIwFeatures::Classical && o.width > 3))
        return failed("AStarIW width must be in 1..5 for classical features or 1..3 for abstracted features");
    if (!std::isfinite(o.weight) || o.weight < 0)
        return failed("AStarIW weight must be finite and nonnegative");
    if (o.features != AStarIwFeatures::Classical && o.features != AStarIwFeatures::Abstracted && o.features != AStarIwFeatures::BaseAbstracted)
        return failed("unknown AStarIW feature mode");
    BestFirstOptions opts;
    opts.control = o.control;
    opts.heuristic = o.heuristic;
    opts.evaluator = o.evaluator;
    opts.store = o.store;
    opts.start = o.start;
    opts.witness_pruning = o.witness_pruning;
    opts.canonical_order = o.canonical_order;
    std::unique_ptr<Context> c;
    try
    {
        detail::make_coordinates(task, o.landmarks);  // validate even when the initial state is a goal
        c = std::make_unique<Context>(task, opts, r, "astar_iw");
        if (!c->ok())
            return r;
        switch (store_kind(*c))
        {
            case BestFirstOptions::Store::Chunked: run<ChunkedAdapter>(*c, o, r); break;
            case BestFirstOptions::Store::Compact: run<CompactAdapter>(*c, o, r); break;
            default: run<FlatAdapter>(*c, o, r); break;
        }
    }
    catch (const heuristics::Interrupted&)
    {
        r.status = c ? c->interrupted_status() : SearchStatus::OutOfTime;
        if (c)
            c->finish(r.stats.states, r.store_bytes);
    }
    catch (const NonUnitCost&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        r.status = SearchStatus::Failed;
        r.message = e.what();
        r.plan.clear();
        r.goal_state.reset();
    }
    r.fluent_slots = task.atoms().fluent_slots();
    return r;
}

AStarIwResult astar_iw(const Task& task, heuristics::Heuristic& heuristic, const AStarIwOptions& options)
{
    AStarIwOptions o = options;
    o.evaluator = &heuristic;
    return astar_iw(task, o);
}
}  // namespace mymyr::search
