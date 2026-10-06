// Parallel IW rollouts (search/parallel_rollouts.hpp): a port of mimir's iw::find_rollouts_parallel
// (src/search/algorithms/iw/parallel_rollouts.cpp) on the IW family engine.

#include "mymyr/search/parallel_rollouts.hpp"

#include "novelty_brfs.hpp"

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace mymyr::search
{
namespace
{
std::vector<CanonicalAtom> canonical_of(const Task& task, AtomKind kind, const u64* w, u32 n)
{
    std::vector<CanonicalAtom> out;
    bits::for_each(w, n, [&](u64 s) { out.push_back(task.atoms().canonical(kind, static_cast<u32>(s))); });
    std::sort(out.begin(), out.end());
    return out;
}

/// The reached fluent slots of a rollout ranked by canonical id, to turn sets of them into ascending canonical ids in
/// O(set bits + slots / 64) (no sort per set). Only reached slots are looked up: every set converted here (a created
/// state, a co-occurrence row) holds reached slots only.
class ReachedOrder
{
public:
    ReachedOrder(const Task& task, std::span<const u64> reached)
    {
        std::vector<std::pair<CanonicalAtom, u32>> by;  // (canonical id, slot): unique ids, a total order
        bits::for_each(reached.data(), static_cast<u32>(reached.size()),
                       [&](u64 s) { by.emplace_back(task.atoms().canonical(AtomKind::Fluent, static_cast<u32>(s)), static_cast<u32>(s)); });
        std::sort(by.begin(), by.end());
        m_rank.assign(reached.size() * 64, ~u32{0});
        for (u32 i = 0; i < by.size(); ++i)
        {
            m_rank[by[i].second] = i;
            m_cid.push_back(by[i].first);
            m_slot.push_back(by[i].second);
        }
        m_scratch.resize(bits::words_for(static_cast<u32>(by.size())));
    }
    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_cid.size()); }
    [[nodiscard]] u32 slot(u32 rank) const noexcept { return m_slot[rank]; }
    [[nodiscard]] CanonicalAtom cid(u32 rank) const noexcept { return m_cid[rank]; }
    std::vector<CanonicalAtom> convert(const u64* w, u32 n)
    {
        std::fill(m_scratch.begin(), m_scratch.end(), 0);
        u32 count = 0;
        bits::for_each(w, n,
                       [&](u64 s)
                       {
                           bits::set(m_scratch.data(), m_rank[s]);
                           ++count;
                       });
        std::vector<CanonicalAtom> out;
        out.reserve(count);
        bits::for_each(m_scratch.data(), static_cast<u32>(m_scratch.size()), [&](u64 r) { out.push_back(m_cid[r]); });
        return out;
    }

private:
    std::vector<u32> m_rank;  // per slot below 64 * reached words: its rank, or ~0 (not reached)
    std::vector<CanonicalAtom> m_cid;
    std::vector<u32> m_slot;
    std::vector<u64> m_scratch;
};

/// The states mimir's plan extraction creates in the rollout's repository (search_space.hpp
/// extract_total_ordered_plan: every successor of every non-goal state of the plan, looking for the cheapest action to
/// the next one), recorded after the search as mimir's tracking records them. They are new only where the search
/// did not enumerate a plan state's successors completely (a layer truncated by max_next_layer_states).
void record_plan_extraction(detail::Env& env, detail::StateTracker& tr, const State& start, const std::vector<Action>& plan)
{
    Successors& succ = env.succ;
    State s = start;
    std::vector<Action> acts;
    std::vector<u32> order;
    StateBuilder b;
    for (const Action& step : plan)
    {
        acts.clear();
        succ.prepare(s);
        succ.generate<false>(
            [&](u32 schema, const ObjectId* binding, const Delta&)
            {
                acts.emplace_back(SchemaId{schema}, std::vector<ObjectId>(binding, binding + succ.arity(schema)));
                return true;
            },
            env.witness, env.canonical);
        order.clear();
        if (env.successor_order && *env.successor_order)
            (*env.successor_order)(s, acts, order);
        std::vector<u8> seen(acts.size(), 0);
        auto record = [&](u32 i)
        {
            if (i >= acts.size() || seen[i])
                return;
            seen[i] = 1;
            b.clear();
            succ.apply(s, acts[i].label(), b);
            tr.record(b.words().data(), static_cast<u32>(b.words().size()));
        };
        for (u32 i : order)
            record(i);
        for (u32 i = 0; i < acts.size(); ++i)
            record(i);
        if (tr.pending())
            tr.flush_derived(succ);
        s = succ.apply(s, step.label());
    }
}

/// Converts a rollout's tracked states to the portable result.
void collect(const Task& task, Successors& succ, const detail::StateTracker& tr, bool landing, bool co, RolloutResult& r)
{
    r.num_states = tr.num_states();
    ReachedOrder order(task, tr.reached());
    r.reached_fluent_atoms.reserve(order.size());
    for (u32 i = 0; i < order.size(); ++i)
        r.reached_fluent_atoms.push_back(order.cid(i));
    r.reached_derived_atoms = canonical_of(task, AtomKind::Derived, tr.reached_derived().data(), static_cast<u32>(tr.reached_derived().size()));
    if (landing)
    {
        std::vector<std::pair<CanonicalAtom, u32>> firsts;  // (atom, state id)
        const std::span<const u32> fa = tr.first_achievers();
        for (u32 s = 0; s < fa.size(); ++s)
            if (fa[s] != detail::k_no_node)
                firsts.emplace_back(task.atoms().canonical(AtomKind::Fluent, s), fa[s]);
        std::sort(firsts.begin(), firsts.end());
        std::unordered_map<u32, u32> index;  // state id -> landing index
        for (const auto& [atom, sid] : firsts)
        {
            const auto [it, fresh] = index.emplace(sid, static_cast<u32>(r.landing_states.size()));
            if (fresh)
            {
                const StateView v = tr.store()[StateId{sid}];
                LandingState ls;
                ls.state = State(v.w, v.nw);
                ls.atoms = order.convert(v.w, v.nw);
                ls.direct_dead_end = !succ.any_applicable(ls.state);
                r.landing_states.push_back(std::move(ls));
            }
            r.landing_state_by_atom.emplace_back(atom, it->second);
        }
    }
    if (co)
    {
        const auto& rows = tr.co_occurrence();
        for (u32 i = 0; i < order.size(); ++i)  // ascending canonical ids; a row is non-empty iff its atom is reached
        {
            const u32 s = order.slot(i);
            if (s < rows.size() && !rows[s].empty())
                r.co_occurrence.emplace_back(order.cid(i), order.convert(rows[s].data(), static_cast<u32>(rows[s].size())));
        }
    }
}
}  // namespace

ParallelRolloutsResult find_rollouts_parallel(const Task& task, const ParallelRolloutOptions& o)
{
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: parallel rollouts do not support tasks with numeric fluents (IW and SIW do)");
    ParallelRolloutsResult out;
    const usize K = o.seeds.size();
    out.rollouts.resize(K);
    if (K == 0)
        return out;
    auto fail_all = [&](std::string m)
    {
        for (RolloutResult& r : out.rollouts)
        {
            r.search.status = SearchStatus::Failed;
            r.search.message = m;
        }
        out.message = std::move(m);
        return std::move(out);
    };
    const SearchControl& control = o.iw.control;
    if (control.coordination)
        return fail_all("find_rollouts_parallel: SearchControl::coordination is not supported with randomized layer orders");
    if (o.iw.max_arity > novelty::k_max_arity)
        return fail_all("IW arity " + std::to_string(o.iw.max_arity) + " exceeds the maximum " + std::to_string(novelty::k_max_arity));
    if (o.max_next_layer_states == 0)
        return fail_all("max_next_layer_states must be positive");

    // per-worker hooks, created on the calling thread
    SearchObserver* const root = control.observer;
    std::vector<std::shared_ptr<SearchObserver>> worker_obs(K);
    std::vector<GoalSpec> worker_goals;
    bool safe = true;
    if (root)
        for (usize k = 0; k < K && safe; ++k)
        {
            worker_obs[k] = root->make_worker(static_cast<u32>(k));
            if (!worker_obs[k])
            {
                safe = false;
                out.message = "the observer is not parallel-safe (make_worker returned null): ran on the calling thread";
            }
        }
    if (safe && control.goal.kind == GoalSpec::Kind::Custom)
    {
        worker_goals.resize(K);
        for (usize k = 0; k < K && safe; ++k)
        {
            std::function<bool(StateView)> f;
            if (control.goal.make_worker)
                f = control.goal.make_worker(static_cast<u32>(k));
            if (!f)
            {
                safe = false;
                out.message = "the custom goal test is not parallel-safe (GoalSpec::make_worker): ran on the calling thread";
                break;
            }
            worker_goals[k].kind = GoalSpec::Kind::Custom;
            worker_goals[k].test = std::move(f);
        }
    }
    if (!safe)
    {
        worker_obs.assign(K, nullptr);
        worker_goals.clear();
    }
    u32 T = o.num_threads == 0 ? std::max<u32>(1, std::thread::hardware_concurrency()) : o.num_threads;
    T = static_cast<u32>(std::min<usize>(T, K));
    if (!safe)
        T = 1;
    out.threads_used = T;

    const State start = o.iw.start ? *o.iw.start : task.initial_state();
    const double secs = control.budget.max_seconds;
    const bool timed = secs < 1e15;
    const auto deadline = timed ? detail::Clock::now() + std::chrono::duration_cast<detail::Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)))
                                : detail::Clock::time_point{};
    if (root)
        root->on_start(start);

    auto run_one = [&](usize k)
    {
        Successors& succ = task.workspace().successors();
        const GoalSpec& spec = worker_goals.empty() ? control.goal : worker_goals[k];
        const detail::GoalTest goal = detail::GoalTest::from_spec(task, spec);
        const detail::BlockedSet blocked(control.blocked_states);
        detail::Env env(task, succ, goal, blocked, control);
        env.obs = worker_obs[k] ? worker_obs[k].get() : root;
        env.root = nullptr;  // lifecycle events are replayed on the calling thread
        env.coord = nullptr;
        env.witness = o.iw.witness_pruning;
        env.canonical = o.iw.canonical_order;
        env.successor_order = &o.iw.successor_order;
        env.timed = timed;
        env.deadline = deadline;
        detail::LayerOrderer layers(task, LayerOrdering{.kind = LayerOrdering::Kind::Randomized,
                                                        .seed = o.seeds[k],
                                                        .max_next_layer_states = o.max_next_layer_states,
                                                        .prefer_more_satisfied_goals = true});
        env.layers = &layers;
        detail::StateTracker tracker(task, o.report_landing_states, o.report_co_occurrence);
        env.tracker = &tracker;
        RolloutResult& r = out.rollouts[k];
        r.search = detail::classic_ladder(env, start, o.iw.max_arity, o.iw.optimize_iw1, o.iw.width_zero, o.iw.tables);
        if (r.search.status == SearchStatus::Solved)
            record_plan_extraction(env, tracker, start, r.search.plan);
        detail::finish_result(env, start, r.search);
        collect(task, succ, tracker, o.report_landing_states, o.report_co_occurrence, r);
    };

    std::atomic<usize> next{0};
    std::vector<std::exception_ptr> errors(K);
    auto work = [&](u32)
    {
        for (;;)
        {
            const usize k = next.fetch_add(1, std::memory_order_relaxed);
            if (k >= K)
                return;
            try
            {
                run_one(k);
            }
            catch (...)
            {
                errors[k] = std::current_exception();
            }
        }
    };
    if (T <= 1)
        work(0);
    else
    {
        ThreadPool pool(T);
        pool.run(work);
    }
    for (const std::exception_ptr& e : errors)
        if (e)
            std::rethrow_exception(e);

    if (root)
    {
        SearchStatistics total;
        SearchStatus status = SearchStatus::Exhausted;
        bool solved = false;
        for (const RolloutResult& r : out.rollouts)
        {
            for (const IwPassStatistics& p : r.search.passes)
            {
                root->on_pass(p.arity, p.statistics());
                detail::add_pass(total, p);
            }
            if (r.search.status == SearchStatus::Solved)
            {
                root->on_solution(r.search.plan, r.search.cost);
                solved = true;
            }
            else if (status == SearchStatus::Exhausted && r.search.status != SearchStatus::Exhausted)
                status = r.search.status;
        }
        root->on_end(solved ? SearchStatus::Solved : status, total);
    }
    return out;
}

std::vector<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>> intersect_co_occurrence(std::span<const RolloutResult> results)
{
    if (results.empty())
        return {};
    std::vector<std::pair<CanonicalAtom, std::vector<CanonicalAtom>>> out = results.front().co_occurrence;
    std::vector<CanonicalAtom> tmp;
    for (usize k = 1; k < results.size(); ++k)
    {
        const auto& rows = results[k].co_occurrence;
        for (auto& [atom, row] : out)
        {
            const auto it = std::lower_bound(rows.begin(), rows.end(), atom, [](const auto& e, CanonicalAtom a) { return e.first < a; });
            if (it == rows.end() || it->first != atom)
            {
                row.clear();  // this rollout never reached the atom
                continue;
            }
            tmp.clear();
            std::set_intersection(row.begin(), row.end(), it->second.begin(), it->second.end(), std::back_inserter(tmp));
            row.swap(tmp);
        }
    }
    return out;
}

MergedLandingStates merge_landing_states(std::span<const RolloutResult> results)
{
    MergedLandingStates m;
    std::unordered_map<State, u32> index;
    m.by_rollout.resize(results.size());
    for (usize k = 0; k < results.size(); ++k)
        for (const LandingState& ls : results[k].landing_states)
        {
            const auto [it, fresh] = index.emplace(ls.state, static_cast<u32>(m.states.size()));
            if (fresh)
                m.states.push_back(ls.state);
            m.by_rollout[k].push_back(it->second);
        }
    return m;
}
}  // namespace mymyr::search
