// Enforced hill-climbing (ehc.hpp): breadth-first lookaheads over one shared store, the context of the best-first
// searches (best_first_detail.hpp) and eager GBFS as the fallback.

#include "mymyr/search/ehc.hpp"

#include "best_first_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mymyr::search
{
namespace
{
using namespace bf;
constexpr f64 k_inf = std::numeric_limits<f64>::infinity();

/// A state entered by the current lookahead: the path to it from the lookahead's start (parent visit and action).
struct Visit
{
    u32 parent;  // visit index (k_none: the lookahead's start)
    u32 id;      // state id in the store
    u32 ref;     // the store's open entry while the visit is queued
    u32 schema;
    u64 boff;    // binding in Lookahead::bind
    f64 g;       // metric value
    u32 depth;   // actions below the lookahead's start
};

/// What hill-climbing leaves to the caller.
struct Outcome
{
    bool failed = false;  // a lookahead found no better state (status not set)
    u64 states = 0;
    u64 bytes = 0;
};

/// The helpful-actions mode for the heuristic (Auto resolved). Throws std::invalid_argument for First and Only with
/// a heuristic without preferred operators.
EhcOptions::Helpful resolve_helpful(EhcOptions::Helpful mode, const heuristics::Heuristic& h)
{
    const bool has = h.provides_preferred();
    switch (mode)
    {
        case EhcOptions::Helpful::Auto: return has ? EhcOptions::Helpful::First : EhcOptions::Helpful::Off;
        case EhcOptions::Helpful::Off: return mode;
        case EhcOptions::Helpful::Only:
        case EhcOptions::Helpful::First:
            if (!has)
                throw std::invalid_argument(std::string("mymyr ehc: helpful actions need a heuristic with preferred operators (ff, "
                                                        "set_additive, lmcount, or a heuristic that provides them); '") +
                                            heuristics::to_string(h.kind()) + "' has none");
            return mode;
    }
    throw std::invalid_argument("mymyr ehc: unknown helpful_actions mode");
}

template<class S>
class HillClimbing
{
public:
    HillClimbing(Context& c, EhcOptions::Helpful helpful, EhcStatistics& stats)
        : m_c(c), m_r(c.r), m_store(c.task), m_helpful(helpful), m_st(stats), m_sw(c.status_words)
    {
    }

    Outcome run()
    {
        Outcome out;
        const StateView s0 = m_c.start.view();
        m_root = m_store.insert(s0).first;
        m_root_cur.w.assign(s0.w, s0.w + s0.nw);
        m_root_cur.num.assign(s0.num, s0.num + s0.nnum);
        m_root_cur.n = s0.nw;
        m_root_g = m_c.g0;
        m_root_status.assign(m_sw, 0);
        if (m_c.path_dependent)
            m_c.h->initial_status(s0, m_root_status);
        if (m_c.unsolvable())
        {
            m_c.begin_search();
            m_r.status = SearchStatus::Unsolvable;
            return done(out);
        }
        if (m_c.is_goal(s0))
        {
            m_c.begin_search();
            m_c.solved(std::vector<Action>{}, m_c.g0, s0, true);
            return done(out);
        }
        m_root_h = evaluate(s0, m_root_status);
        m_r.initial_h = m_root_h;
        m_c.begin_search();
        if (m_root_h == k_inf)
        {
            ++m_r.dead_ends;
            m_r.status = SearchStatus::Unsolvable;
            return done(out);
        }
        bool helpful = m_helpful != EhcOptions::Helpful::Off;
        for (;;)
        {
            const Step step = lookahead(helpful);
            if (step == Step::Stop)
                return done(out);
            if (step == Step::Improved)
            {
                helpful = m_helpful != EhcOptions::Helpful::Off;
                continue;
            }
            if (helpful && m_helpful == EhcOptions::Helpful::First)
            {
                ++m_st.retries;
                helpful = false;
                continue;
            }
            m_st.failed = true;
            out.failed = true;
            return done(out);
        }
    }

private:
    enum class Step : u8
    {
        Improved,  // committed to a better state
        Failed,    // no better state below the current one
        Stop,      // solved, or stopped by a budget (the status is set)
    };

    Outcome& done(Outcome& out)
    {
        out.states = m_store.size();
        out.bytes = m_store.bytes() + m_visits.capacity() * sizeof(Visit) + m_bind.capacity() * sizeof(ObjectId) +
                    m_status.capacity() * 8 + m_queue.capacity() * 4 + m_seen.capacity() * 4;
        m_st.expanded = m_r.stats.expanded;
        m_st.evaluations = m_r.evaluations;
        return out;
    }

    f64 evaluate(StateView s, std::span<const u64> status)
    {
        if (!m_c.path_dependent)
            return m_c.evaluate(s);
        ++m_r.evaluations;
        return m_c.h->evaluate(s, status);
    }

    [[nodiscard]] std::span<u64> status_of(u32 v) noexcept { return {m_status.data() + static_cast<usize>(v) * m_sw, m_sw}; }

    /// The actions of the plan so far, then those from the lookahead's start to visit v, then (schema, b).
    [[nodiscard]] std::vector<Action> plan_to(u32 v, u32 schema = k_none, const ObjectId* b = nullptr) const
    {
        std::vector<Action> out = m_plan;
        const usize at = out.size();
        for (u32 x = v; m_visits[x].parent != k_none; x = m_visits[x].parent)
        {
            const Visit& vx = m_visits[x];
            const ObjectId* xb = m_bind.data() + vx.boff;
            out.emplace_back(SchemaId{vx.schema}, std::vector<ObjectId>(xb, xb + m_c.succ.arity(vx.schema)));
        }
        std::reverse(out.begin() + static_cast<std::ptrdiff_t>(at), out.end());
        if (schema != k_none)
            out.emplace_back(SchemaId{schema}, std::vector<ObjectId>(b, b + m_c.succ.arity(schema)));
        return out;
    }

    u32 add_visit(u32 parent, u32 id, u32 schema, const ObjectId* b, f64 g, u32 depth)
    {
        const u32 v = static_cast<u32>(m_visits.size());
        m_visits.push_back({parent, id, k_none, schema, m_bind.size(), g, depth});
        if (b)
            m_bind.insert(m_bind.end(), b, b + m_c.succ.arity(schema));
        m_status.resize(m_status.size() + m_sw, 0);
        return v;
    }

    void mark(u32 id)
    {
        if (id >= m_seen.size())
            m_seen.resize(std::max<usize>(id + 1, m_seen.size() * 2), 0);
        m_seen[id] = m_epoch;
    }
    [[nodiscard]] bool seen(u32 id) const noexcept { return id < m_seen.size() && m_seen[id] == m_epoch; }

    void release_queue()
    {
        for (usize i = m_head; i < m_queue.size(); ++i)
            m_store.release(m_visits[m_queue[i]].ref);
        m_queue.clear();
        m_head = 0;
    }

    /// One breadth-first lookahead from the current state (m_root).
    Step lookahead(bool helpful)
    {
        ++m_st.lookaheads;
        ++m_epoch;
        const SearchStatistics before = m_r.stats;
        const Clock::time_point t0 = Clock::now();
        m_visits.clear();
        m_bind.clear();
        m_status.clear();
        const u32 root = add_visit(k_none, m_root, k_none, nullptr, m_root_g, 0);
        std::copy(m_root_status.begin(), m_root_status.end(), status_of(root).begin());
        mark(m_root);
        m_visits[root].ref = m_store.make_ref(m_root, m_root_cur.view());
        m_queue.push_back(root);
        // the preferred operators of the heuristic's last evaluation are the start's (evaluated last, or the start
        // of the search) except on a retry, which takes every action
        Step step = Step::Failed;
        const u32 max_depth = m_c.o.control.budget.max_depth;
        while (m_head < m_queue.size())
        {
            if (!m_c.keep_going())
            {
                step = Step::Stop;
                break;
            }
            const u32 v = m_queue[m_head++];
            m_store.load(m_visits[v].ref, m_cur);
            m_store.release(m_visits[v].ref);
            const StateView sv = m_cur.view();
            if (v != root)
            {
                const f64 hv = evaluate(sv, status_of(v));
                if (hv == k_inf)
                {
                    ++m_r.dead_ends;
                    ++m_r.stats.pruned;
                    continue;
                }
                if (hv < m_root_h)
                {
                    commit(v, hv);
                    step = Step::Improved;
                    break;
                }
            }
            if (m_plan.size() + m_visits[v].depth >= max_depth)
            {
                ++m_r.stats.pruned;
                continue;
            }
            ++m_r.stats.expanded;
            if (m_c.obs)
                m_c.obs->on_expand(m_visits[v].id, sv);
            if (!expand(v, helpful))
            {
                step = Step::Stop;  // solved, or a budget (the status is set)
                break;
            }
        }
        release_queue();
        if (m_c.obs)
        {
            SearchStatistics pass;
            pass.expanded = m_r.stats.expanded - before.expanded;
            pass.generated = m_r.stats.generated - before.generated;
            pass.states = m_visits.size();
            pass.pruned = m_r.stats.pruned - before.pruned;
            pass.seconds = seconds_since(t0);
            m_c.obs->on_pass(static_cast<u32>(m_st.lookaheads - 1), pass);
        }
        return step;
    }

    /// Commits to visit v (state in m_cur) with value hv: its path joins the plan and it becomes the current state.
    void commit(u32 v, f64 hv)
    {
        const Visit& x = m_visits[v];
        m_plan = plan_to(v);
        m_st.max_depth = std::max<u64>(m_st.max_depth, x.depth);
        ++m_st.improvements;
        m_root = x.id;
        m_root_g = x.g;
        m_root_h = hv;
        m_root_cur = m_cur;
        const std::span<u64> st = status_of(v);
        std::copy(st.begin(), st.end(), m_root_status.begin());
    }

    /// Generates the transitions of visit v (state in m_cur) and enters its successors that are new to this lookahead.
    /// False when the search ends: a goal state (solved), or a budget, the time or the cancellation (status set).
    bool expand(u32 v, bool helpful)
    {
        Transitions& tr = m_tr;
        tr.clear();
        tr.nn = m_c.nn;
        if (m_store.size() >= m_c.max_states)
        {
            m_r.status = SearchStatus::OutOfStates;
            return false;
        }
        const u32 id = m_visits[v].id;
        const f64 gp = m_visits[v].g;
        m_c.succ.prepare(m_cur.view());
        bool stop = false;
        m_c.generate(
            [&](u32 s, const ObjectId* b, const Delta& d) -> bool
            {
                if (!m_c.keep_generating())
                {
                    stop = true;
                    return false;
                }
                const u32 arity = m_c.succ.arity(s);
                if (helpful && !m_c.h->preferred(ActionLabel{SchemaId{s}, std::span<const ObjectId>(b, arity)}))
                    return true;  // not a helpful action: not generated
                ++m_r.stats.generated;
                const u32 nn = m_store.successor(m_cur, d, m_next, m_c.succ);
                if (m_c.blocked(m_next.data(), nn, d.num))
                {
                    ++m_r.stats.pruned;
                    if (m_c.obs)
                        m_c.obs->on_prune(id, m_c.action(s, b), {m_next.data(), nn, d.num, m_c.nn});
                    return true;
                }
                const f64 g = m_c.g_next(gp, d);
                const auto [cid, fresh] = m_store.insert_successor(id, m_cur, m_next.data(), nn, d);
                const bool is_new = !seen(cid);
                if (is_new)
                    mark(cid);
                Transitions::T t{s, cid, tr.bind.size(), tr.words.size(), 0, g, is_new, false};
                tr.bind.insert(tr.bind.end(), b, b + arity);
                if (is_new || m_c.obs)
                {
                    t.nw = bits::trimmed_size(m_next.data(), nn);
                    tr.words.insert(tr.words.end(), m_next.data(), m_next.data() + t.nw);
                    if (m_c.nn)
                        tr.words.insert(tr.words.end(), d.num, d.num + m_c.nn);
                }
                tr.t.push_back(t);
                if (fresh && m_store.size() >= m_c.max_states)
                {
                    m_r.status = SearchStatus::OutOfStates;
                    stop = true;
                    return false;
                }
                return true;
            });
        if (stop)
            return false;
        const StateView pv = m_cur.view();
        const u32 depth = m_visits[v].depth + 1;
        for (usize i = 0; i < tr.t.size(); ++i)
        {
            const Transitions::T& t = tr.t[i];
            const ObjectId* b = tr.binding(t);
            if (m_c.obs)
                m_c.obs->on_generate(id, m_c.action(t.schema, b), t.child, tr.view(t), t.fresh);
            if (!t.fresh)
                continue;
            const StateView cv = tr.view(t);
            if (m_c.is_goal(cv))
            {
                m_c.solved(plan_to(v, t.schema, b), t.g, cv, true);
                if (m_c.obs)  // the transitions after the goal's were generated too
                    for (usize j = i + 1; j < tr.t.size(); ++j)
                        m_c.obs->on_generate(id, m_c.action(tr.t[j].schema, tr.binding(tr.t[j])), tr.t[j].child,
                                             tr.view(tr.t[j]), tr.t[j].fresh);
                return false;
            }
            const u32 w = add_visit(v, t.child, t.schema, b, t.g, depth);
            if (m_c.path_dependent)
                m_c.h->progress(status_of(v), pv, ActionLabel{SchemaId{t.schema}, std::span<const ObjectId>(b, m_c.succ.arity(t.schema))},
                                cv, status_of(w));
            m_visits[w].ref = m_store.make_ref(t.child, cv);
            m_queue.push_back(w);
        }
        return !stop;
    }

    Context& m_c;
    BestFirstResult& m_r;
    S m_store;
    EhcOptions::Helpful m_helpful;
    EhcStatistics& m_st;
    u32 m_sw;  // status words of a path-dependent heuristic
    // the current state (the start of the next lookahead) and the plan to it
    u32 m_root = 0;
    Cur m_root_cur;
    f64 m_root_g = 0, m_root_h = 0;
    std::vector<u64> m_root_status;
    std::vector<Action> m_plan;
    // the current lookahead
    u32 m_epoch = 0;
    std::vector<u32> m_seen;  // per state id: the lookahead that entered it last
    std::vector<Visit> m_visits;
    std::vector<ObjectId> m_bind;
    std::vector<u64> m_status;  // m_sw words per visit
    std::vector<u32> m_queue;
    usize m_head = 0;
    Transitions m_tr;
    Cur m_cur;
    std::vector<u64> m_next;
};

template<class S>
Outcome climb(Context& c, EhcOptions::Helpful helpful, EhcStatistics& stats)
{
    c.r.store = S::k_name;
    HillClimbing<S> h(c, helpful, stats);
    return h.run();
}
}  // namespace

EhcResult ehc(const Task& task, const EhcOptions& o)
{
    EhcResult r;
    r.algorithm = "ehc";
    BestFirstOptions opts;
    opts.control = o.control;
    opts.heuristic = o.heuristic;
    opts.evaluator = o.evaluator;
    opts.store = o.store;
    opts.start = o.start;
    opts.witness_pruning = o.witness_pruning;
    opts.canonical_order = o.canonical_order;
    const Clock::time_point t0 = Clock::now();
    if (o.fallback != EhcOptions::Fallback::Gbfs && o.fallback != EhcOptions::Fallback::None)
        throw std::invalid_argument("mymyr ehc: unknown fallback");
    if (o.helpful_actions != EhcOptions::Helpful::Auto && o.helpful_actions != EhcOptions::Helpful::Off &&
        o.helpful_actions != EhcOptions::Helpful::Only && o.helpful_actions != EhcOptions::Helpful::First)
        throw std::invalid_argument("mymyr ehc: unknown helpful_actions mode");
    std::unique_ptr<Context> c;
    try
    {
        c = std::make_unique<Context>(task, opts, r, "ehc");
        if (!c->ok())
            return r;
        const EhcOptions::Helpful helpful = resolve_helpful(o.helpful_actions, *c->h);
        r.queue = "fifo";
        Outcome out;
        switch (store_kind(*c))
        {
            case BestFirstOptions::Store::Chunked: out = climb<ChunkedAdapter>(*c, helpful, r.ehc); break;
            case BestFirstOptions::Store::Compact: out = climb<CompactAdapter>(*c, helpful, r.ehc); break;
            default: out = climb<FlatAdapter>(*c, helpful, r.ehc); break;
        }
        if (!out.failed || o.fallback == EhcOptions::Fallback::None)
        {
            if (out.failed)
                r.status = SearchStatus::Exhausted;  // hill-climbing failed
            c->finish(out.states, out.bytes);
            r.fluent_slots = task.atoms().fluent_slots();
            return r;
        }
        // greedy best-first search from the start state with the same heuristic and the rest of the budget
        r.ehc.fallback = true;
        BestFirstOptions g = opts;
        g.evaluator = c->h;
        g.queue = use_buckets(*c) ? BestFirstOptions::Queue::Bucket : BestFirstOptions::Queue::Heap;
        std::unique_ptr<InnerObserver> inner;
        if (c->obs)
            inner = std::make_unique<InnerObserver>(*c->obs);
        g.control.observer = inner.get();
        Budget& b = g.control.budget;
        if (std::isfinite(b.max_seconds))
            b.max_seconds = std::max(0.0, b.max_seconds - seconds_since(t0));
        if (b.max_expanded != std::numeric_limits<u64>::max())
            b.max_expanded -= std::min(b.max_expanded, r.stats.expanded);
        BestFirstResult f = gbfs_eager(task, g);
        r.status = f.status;
        r.plan = std::move(f.plan);
        r.cost = f.cost;
        r.goal_state = std::move(f.goal_state);
        r.message = std::move(f.message);
        r.stats.expanded += f.stats.expanded;
        r.stats.generated += f.stats.generated;
        r.stats.pruned += f.stats.pruned;
        r.evaluations += f.evaluations;
        r.dead_ends += f.dead_ends;
        r.store = f.store;
        r.queue = f.queue;
        c->finish(out.states + f.stats.states, f.store_bytes);
    }
    catch (const heuristics::Interrupted&)
    {
        r.status = c->interrupted_status();
        c->finish(r.stats.states, r.store_bytes);
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
}  // namespace mymyr::search
