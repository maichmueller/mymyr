// Anytime search (anytime.hpp): restarting weighted A* over the best-first searches, with one heuristic for every
// iteration.

#include "mymyr/search/anytime.hpp"

#include "best_first_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace mymyr::search
{
namespace
{
using bf::Clock;
using bf::seconds_since;

void check(const AnytimeOptions& o)
{
    if (o.weights.empty())
        throw std::invalid_argument("mymyr anytime_astar: AnytimeOptions::weights is empty");
    for (usize i = 0; i < o.weights.size(); ++i)
    {
        const double w = o.weights[i];
        if (!(std::isfinite(w) && w >= 1))
            throw std::invalid_argument("mymyr anytime_astar: every weight must be finite and at least 1 (got " + std::to_string(w) + ")");
        if (i > 0 && w > o.weights[i - 1])
            throw std::invalid_argument("mymyr anytime_astar: the weights must not increase");
        if (o.queue == BestFirstOptions::Queue::Bucket && w != std::floor(w))
            throw std::invalid_argument("mymyr anytime_astar: a bucket queue needs integral weights (use Queue::Heap or Queue::Auto)");
    }
    if (o.weight != 1)
        throw std::invalid_argument("mymyr anytime_astar: BestFirstOptions::weight is not used (the schedule is AnytimeOptions::weights)");
    if (std::isnan(o.cost_bound))
        throw std::invalid_argument("mymyr anytime_astar: BestFirstOptions::cost_bound is NaN");
}
}  // namespace

AnytimeResult anytime_astar(const Task& task, const AnytimeOptions& o)
{
    AnytimeResult r;
    r.algorithm = o.lazy ? "anytime_astar_lazy" : "anytime_astar";
    const Clock::time_point t0 = Clock::now();
    SearchObserver* const obs = o.control.observer;
    check(o);
    try
    {
        const Budget& budget = o.control.budget;
        const bool timed = std::isfinite(budget.max_seconds);
        // the heuristic of every iteration: the caller's, or one built here within the time budget
        std::unique_ptr<heuristics::Heuristic> owned;
        heuristics::Heuristic* h = o.evaluator;
        if (!h)
        {
            heuristics::Options ho = o.heuristic;
            if (timed)
                ho.budget.max_seconds = std::min(ho.budget.max_seconds, std::max(0.0, budget.max_seconds));
            owned = heuristics::make_heuristic(task, ho);
            h = owned.get();
        }
        const heuristics::ActionCosts costs(task);
        const bool admissible = h->admissible() && o.control.goal.kind != GoalSpec::Kind::Custom;
        // Queue::Auto as A* decides it for its own heuristic: buckets for small integral costs and an integral weight
        const bool small_integral =
            o.evaluator == nullptr && costs.integral() && costs.max_constant() >= 0 && costs.max_constant() <= 64;
        const State start = o.start ? *o.start : task.initial_state();
        r.setup_seconds = seconds_since(t0);
        const Clock::time_point search_t0 = Clock::now();
        if (obs)
            obs->on_start(start.view());
        std::unique_ptr<bf::InnerObserver> inner;
        if (obs)
            inner = std::make_unique<bf::InnerObserver>(*obs);

        f64 incumbent = o.cost_bound;
        bool found = false;
        SearchStatus last = SearchStatus::Exhausted;
        for (u32 k = 0;; ++k)
        {
            const f64 w = o.weights[std::min<usize>(k, o.weights.size() - 1)];
            BestFirstOptions it = o;
            it.weight = w;
            it.cost_bound = incumbent;
            it.evaluator = h;
            if (o.queue == BestFirstOptions::Queue::Auto)
                it.queue = small_integral && w == std::floor(w) ? BestFirstOptions::Queue::Bucket : BestFirstOptions::Queue::Heap;
            it.control.observer = inner.get();
            Budget& b = it.control.budget;
            if (timed)
                b.max_seconds = std::max(0.0, budget.max_seconds - seconds_since(t0));
            if (budget.max_expanded != std::numeric_limits<u64>::max())
                b.max_expanded = budget.max_expanded - std::min(budget.max_expanded, r.stats.expanded);
            BestFirstResult x = o.lazy ? astar_lazy(task, it) : astar_eager(task, it);

            r.stats.expanded += x.stats.expanded;
            r.stats.generated += x.stats.generated;
            r.stats.states += x.stats.states;
            r.stats.pruned += x.stats.pruned;
            r.evaluations += x.evaluations;
            r.dead_ends += x.dead_ends;
            r.reopened += x.reopened;
            r.store_bytes = std::max(r.store_bytes, x.store_bytes);
            r.stats.memory_bytes = r.store_bytes;
            if (k == 0)
            {
                r.initial_h = x.initial_h;
                r.store = x.store;
                r.queue = x.queue;
            }
            AnytimeIteration rec;
            rec.weight = w;
            rec.cost_bound = incumbent;
            rec.status = x.status;
            rec.stats = x.stats;
            rec.evaluations = x.evaluations;
            rec.cost = x.status == SearchStatus::Solved ? x.cost : 0;
            r.iterations.push_back(rec);
            if (obs)
                obs->on_pass(k, x.stats);
            last = x.status;
            if (x.status != SearchStatus::Solved)
            {
                if (x.status == SearchStatus::Exhausted)
                    r.optimal = found && o.reopen && budget.max_depth == std::numeric_limits<u32>::max();
                if (x.status == SearchStatus::Failed)
                    r.message = std::move(x.message);
                break;
            }
            found = true;
            incumbent = x.cost;
            r.plan = std::move(x.plan);
            r.cost = x.cost;
            r.goal_state = std::move(x.goal_state);
            r.solutions.push_back({x.cost, seconds_since(search_t0), w, k, r.stats.expanded, r.plan.size()});
            if (obs)
                obs->on_solution(r.plan, r.cost);
            if (w == 1 && admissible && !o.lazy && o.reopen)
            {
                r.optimal = true;  // A* with an admissible heuristic
                break;
            }
        }
        r.status = found ? SearchStatus::Solved : last;
        r.stats.seconds = seconds_since(search_t0);
        r.heuristic = h->stats();
        if (obs)
            obs->on_end(r.status, r.stats);
    }
    catch (const std::exception& e)
    {
        r.status = SearchStatus::Failed;
        r.message = e.what();
        r.plan.clear();
        r.cost = 0;
        r.goal_state.reset();
    }
    r.fluent_slots = task.atoms().fluent_slots();
    return r;
}
}  // namespace mymyr::search
