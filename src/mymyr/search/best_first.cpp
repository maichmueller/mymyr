// Shared machinery of the best-first searches (best_first_detail.hpp): the search context and the store and queue
// selection. The searches are in astar.cpp, gbfs.cpp and beam.cpp.

#include "best_first_detail.hpp"

#include <cmath>
#include <limits>

namespace mymyr::search::bf
{
Context::Context(const Task& t, const BestFirstOptions& opt, BestFirstResult& res, const char* algorithm)
    : task(t), o(opt), r(res), succ(t.workspace().successors()), nn(t.numeric_words()), m_blocked(1, 16, t.numeric_words())
{
    m_t0 = Clock::now();
    m_search_t0 = m_t0;
    r.algorithm = algorithm;
    const Budget& b = o.control.budget;
    m_timed = std::isfinite(b.max_seconds);
    if (m_timed)
        m_deadline = m_t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(0.0, b.max_seconds)));
    obs = o.control.observer;
    witness = o.witness_pruning;
    canonical = o.canonical_order;
    max_states = b.max_states;
    stop_on_states = max_states != std::numeric_limits<u64>::max();
    start = o.start ? *o.start : task.initial_state();
    if (o.control.goal.kind == GoalSpec::Kind::Custom && !o.control.goal.test)
        throw std::invalid_argument("mymyr best-first search: GoalSpec::Custom without a test");
    try
    {
        m_costs = std::make_unique<heuristics::ActionCosts>(task);
    }
    catch (const std::invalid_argument& e)
    {
        r.status = SearchStatus::Failed;
        r.message = e.what();
        m_ok = false;
        return;
    }
    costs = m_costs.get();
    g0 = costs->initial(start.view());
    if (o.evaluator)
        h = o.evaluator;
    else
    {
        heuristics::Options ho = o.heuristic;
        if (m_timed)  // the grounding stops within the search's time budget (then the lifted fallback takes over)
            ho.budget.max_seconds = std::min(ho.budget.max_seconds, std::max(0.0, b.max_seconds));
        m_owned = heuristics::make_heuristic(task, ho);
        h = m_owned.get();
    }
    h->set_interrupt([this] { return (m_timed && Clock::now() >= m_deadline) || o.control.cancel.requested(); });
    batched = h->batched();
    for (const State& s : o.control.blocked_states)
    {
        if (s.numeric_words() != nn)
            throw std::invalid_argument("mymyr best-first search: a blocked state has no numeric values of this task");
        m_blocked.insert(s.view());
    }
    m_has_blocked = !o.control.blocked_states.empty();
}

Context::~Context()
{
    if (h)
        h->set_interrupt({});
}

f64 Context::evaluate(StateView s)
{
    ++r.evaluations;
    if (o.control.goal.kind == GoalSpec::Kind::AnyOf)
        return h->evaluate(s, o.control.goal.goals);
    return h->evaluate(s);
}

void Context::evaluate(std::span<const StateView> states, std::span<f64> out)
{
    r.evaluations += states.size();
    if (o.control.goal.kind == GoalSpec::Kind::AnyOf)
    {
        for (usize i = 0; i < states.size(); ++i)
            out[i] = h->evaluate(states[i], o.control.goal.goals);
        return;
    }
    h->evaluate_batch(states, out);
}

void Context::evaluate_fresh(const Transitions& tr, bool stop_at_goal, bool goals_too)
{
    m_batch.clear();
    m_batch_index.clear();
    fresh_goal.assign(tr.t.size(), 0);
    fresh_h.assign(tr.t.size(), 0);
    for (usize i = 0; i < tr.t.size(); ++i)
    {
        const Transitions::T& t = tr.t[i];
        if (!t.fresh)
            continue;
        const StateView v = tr.view(t);
        const bool goal = is_goal(v);
        fresh_goal[i] = goal;
        if (!goal || goals_too)
        {
            m_batch.push_back(v);
            m_batch_index.push_back(static_cast<u32>(i));
        }
        if (goal && stop_at_goal)
            break;
    }
    if (m_batch.empty())
        return;
    m_batch_h.resize(m_batch.size());
    evaluate(m_batch, m_batch_h);
    for (usize j = 0; j < m_batch.size(); ++j)
        fresh_h[m_batch_index[j]] = m_batch_h[j];
}

bool Context::is_goal(StateView s)
{
    const GoalSpec& g = o.control.goal;
    switch (g.kind)
    {
        case GoalSpec::Kind::Task: return succ.is_goal(s);
        case GoalSpec::Kind::AnyOf:
            for (const GoalSpec::AtomGoal& a : g.goals)
            {
                bool holds = true;
                for (SlotId p : a.positive)
                    holds = holds && s.contains(p);
                for (SlotId p : a.negative)
                    holds = holds && !s.contains(p);
                if (holds)
                    return true;
            }
            return false;
        case GoalSpec::Kind::Custom: return g.test(s);
    }
    return false;
}

bool Context::unsolvable() const
{
    return o.control.goal.kind == GoalSpec::Kind::Task && task.compiled().goal.unsatisfiable;
}

void Context::begin_search()
{
    m_search_t0 = Clock::now();
    r.setup_seconds = std::chrono::duration<double>(m_search_t0 - m_t0).count();
    m_next_progress = std::max<u64>(1, o.control.progress_interval);
    if (obs)
        obs->on_start(start.view());
}

bool Context::keep_going()
{
    const Budget& b = o.control.budget;
    if (r.stats.expanded >= b.max_expanded)
    {
        r.status = SearchStatus::OutOfStates;
        return false;
    }
    if ((m_tick++ & 7) == 0)
    {
        if (m_timed && Clock::now() >= m_deadline)
        {
            r.status = SearchStatus::OutOfTime;
            return false;
        }
        if (o.control.cancel.requested())
        {
            r.status = SearchStatus::Cancelled;
            return false;
        }
    }
    if (obs && r.stats.expanded >= m_next_progress)
    {
        m_next_progress = r.stats.expanded + std::max<u64>(1, o.control.progress_interval);
        r.stats.seconds = seconds_since(m_search_t0);
        if (!obs->on_progress(r.stats))
        {
            r.status = SearchStatus::Cancelled;
            return false;
        }
    }
    return true;
}

void Context::solved(const Nodes& nodes, u32 id, StateView goal, bool cheapest)
{
    r.status = SearchStatus::Solved;
    r.plan = nodes.plan(id, succ);
    r.cost = nodes.g[id];
    r.goal_state = State(goal);
    if (!cheapest || costs->unit())
        return;
    // As in mimir's plan extraction: between two consecutive states of the path the action that gives the successor
    // the lowest metric value is taken.
    r.cost = heuristics::plan_metric(succ, *costs, start, g0, r.plan);
}

void Context::finish(u64 states, u64 bytes)
{
    r.stats.states = states;
    r.stats.seconds = seconds_since(m_search_t0);
    r.store_bytes = bytes;
    if (h)
        r.heuristic = h->stats();
    if (r.status != SearchStatus::Solved)
    {
        r.plan.clear();
        r.cost = 0;
        r.goal_state.reset();
    }
    if (obs)
    {
        if (r.status == SearchStatus::Solved)
            obs->on_solution(r.plan, r.cost);
        obs->on_end(r.status, r.stats);
    }
}

bool use_buckets(const Context& c)
{
    switch (c.o.queue)
    {
        case BestFirstOptions::Queue::Bucket: return true;
        case BestFirstOptions::Queue::Heap: return false;
        case BestFirstOptions::Queue::Auto: break;
    }
    // g - g0 and h must be small integers: integral constant action costs of at most 64 (the heuristics then produce
    // integers too: unit costs count actions, real costs sum the same integral costs). A caller's evaluator may not.
    const heuristics::ActionCosts& k = *c.costs;
    return c.o.evaluator == nullptr && k.integral() && k.max_constant() >= 0 && k.max_constant() <= 64;
}

BestFirstOptions::Store store_kind(const Context& c)
{
    if (c.o.store != BestFirstOptions::Store::Auto)
        return c.o.store;
    // Flat for W <= 8 words, Chunked above (the width estimate: frozen W, else the pilot's W_lazy)
    const Task& t = c.task;
    const u32 w = t.atoms().mode() == AtomMode::Frozen ? t.words() : std::max(t.words(), t.info().pilot_words);
    return w <= 8 ? BestFirstOptions::Store::Flat : BestFirstOptions::Store::Chunked;
}
}  // namespace mymyr::search::bf
