// State spaces: a parallel generator over the concurrent store, with mimir's StateSpace semantics
// (datasets/state_space.hpp).
//
// Generation is the layer-synchronous BrFS of search/brfs.cpp (deterministic ids: atomic-min discoverer keys plus a
// per-layer counting sort), with witness pruning off and canonical successor order, and every generated successor
// recorded as a transition in per-thread buffers (target handle, label, cost). The post-processing turns them into the
// forward CSR in id order, the reverse CSR (rows sorted by forward edge index), the goal distances and the flags.

#include "mymyr/datasets/state_space.hpp"

#include "mymyr/core/team.hpp"
#include "mymyr/core/threads.hpp"
#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/state/concurrent_store.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <exception>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace mymyr::datasets
{
const char* to_string(StateSpaceStatus s) noexcept
{
    switch (s)
    {
        case StateSpaceStatus::Ok: return "ok";
        case StateSpaceStatus::OutOfStates: return "out_of_states";
        case StateSpaceStatus::Timeout: return "timeout";
        case StateSpaceStatus::Unsolvable: return "unsolvable";
    }
    return "unknown";
}

// ------------------------------------------------------------------------------------------------- StateSpace
struct StateSpace::Index
{
    std::vector<u32> slots;  // open addressing over state ids (+1; 0 = empty)
    u64 mask = 0;
};

StateSpace::~StateSpace() = default;

namespace
{
u64 row_hash(StateView s) { return hash::state(s.w, s.nw, s.num, s.nnum); }
bool rows_equal(StateView a, StateView b)
{
    return bits::equal(a.w, a.nw, b.w, b.nw) && a.nnum == b.nnum && (a.nnum == 0 || std::memcmp(a.num, b.num, a.nnum * sizeof(u64)) == 0);
}
}  // namespace

i64 StateSpace::find(StateView s) const
{
    m_index_once.call([&] {
        auto idx = std::make_shared<Index>();
        u64 cap = 16;
        while (cap < static_cast<u64>(m_n) * 2)
            cap *= 2;
        idx->slots.assign(cap, 0);
        idx->mask = cap - 1;
        for (u32 id = 0; id < m_n; ++id)
        {
            u64 j = row_hash(state(id)) & idx->mask;
            while (idx->slots[j])
                j = (j + 1) & idx->mask;
            idx->slots[j] = id + 1;
        }
        m_index = std::move(idx);
    });
    if (s.nnum != m_numeric_words)
        return -1;
    for (u64 j = row_hash(s) & m_index->mask;; j = (j + 1) & m_index->mask)
    {
        const u32 v = m_index->slots[j];
        if (v == 0)
            return -1;
        if (rows_equal(state(v - 1), s))
            return v - 1;
    }
}

u32 StateSpace::source(u64 edge) const
{
    if (edge >= m_targets.size())
        throw std::out_of_range("mymyr: transition index out of range");
    const auto it = std::upper_bound(m_offsets.begin(), m_offsets.end(), edge);
    return static_cast<u32>(it - m_offsets.begin() - 1);
}

Action StateSpace::label(u64 edge) const
{
    if (!m_has_labels)
        throw std::logic_error("mymyr: this state space was generated without labels (StateSpaceOptions::labels)");
    if (edge >= m_targets.size())
        throw std::out_of_range("mymyr: transition index out of range");
    const u32 s = m_schemas[edge];
    const u32 arity = m_task->data().schemas[s].arity();
    std::vector<ObjectId> b(arity);
    for (u32 i = 0; i < arity; ++i)
        b[i] = ObjectId{m_bindings[edge * m_label_width + i]};
    return Action(SchemaId{s}, std::move(b));
}

std::vector<u32> StateSpace::goal_states() const
{
    std::vector<u32> out;
    out.reserve(m_num_goal);
    for (u32 s = 0; s < m_n; ++s)
        if (m_goal[s])
            out.push_back(s);
    return out;
}

std::vector<u32> StateSpace::unsolvable_states() const
{
    std::vector<u32> out;
    out.reserve(m_num_unsolvable);
    for (u32 s = 0; s < m_n; ++s)
        if (m_unsolvable[s])
            out.push_back(s);
    return out;
}

u64 StateSpace::bytes() const noexcept
{
    auto b = [](const auto& v) { return static_cast<u64>(v.capacity() * sizeof(v[0])); };
    return b(m_states) + b(m_offsets) + b(m_targets) + b(m_schemas) + b(m_bindings) + b(m_costs) + b(m_boffsets) + b(m_bsources) +
           b(m_bedges) + b(m_unit) + b(m_cost) + b(m_goal) + b(m_unsolvable) + b(m_alive);
}

// ------------------------------------------------------------------------------------------------- post-processing
/// Fills the backward CSR, the goal distances and the flags of a space whose states, forward CSR, labels, costs and
/// goal flags are set. Returns false if remove_if_unsolvable discards it.
class StateSpaceBuilder
{
public:
    static bool finish(StateSpace& S, Team& team, bool remove_if_unsolvable)
    {
        const u32 T = team.size();
        const u64 N = S.m_n;
        const u64 E = S.m_targets.size();
        std::vector<u64> partial(T + 1, 0);
        // reverse CSR: in-degrees, prefix sum, scatter (edge, source), then sort each row by edge (deterministic)
        S.m_boffsets.assign(N + 1, 0);
        {
            std::unique_ptr<std::atomic<u32>[]> indeg(new std::atomic<u32>[N + 1]);
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, T);
                    for (u64 i = a; i < b; ++i)
                        indeg[i].store(0, std::memory_order_relaxed);
                });
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(E, t, T);
                    for (u64 e = a; e < b; ++e)
                        indeg[S.m_targets[e]].fetch_add(1, std::memory_order_relaxed);
                });
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, T);
                    for (u64 i = a; i < b; ++i)
                        S.m_boffsets[i + 1] = indeg[i].load(std::memory_order_relaxed);
                });
            prefix(team, S.m_boffsets, N, partial);
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, T);
                    for (u64 i = a; i < b; ++i)
                        indeg[i].store(0, std::memory_order_relaxed);
                });
            S.m_bsources.assign(E, 0);
            S.m_bedges.assign(E, 0);
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, T);
                    for (u64 s = a; s < b; ++s)
                        for (u64 e = S.m_offsets[s]; e < S.m_offsets[s + 1]; ++e)
                        {
                            const u32 d = S.m_targets[e];
                            const u64 pos = S.m_boffsets[d] + indeg[d].fetch_add(1, std::memory_order_relaxed);
                            S.m_bedges[pos] = static_cast<u32>(e);
                        }
                });
            team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, T);
                    for (u64 v = a; v < b; ++v)
                    {
                        auto first = S.m_bedges.begin() + static_cast<i64>(S.m_boffsets[v]);
                        auto last = S.m_bedges.begin() + static_cast<i64>(S.m_boffsets[v + 1]);
                        std::sort(first, last);  // edge indices are distinct: no ties
                    }
                });
            // the sources of the sorted rows
            fill_sources(S, team);
        }
        // unit goal distances: level-synchronous backward BFS from every goal state
        S.m_unit.assign(N, k_unsolvable_distance);
        std::vector<u32> frontier;
        for (u32 i = 0; i < N; ++i)
            if (S.m_goal[i])
            {
                S.m_unit[i] = 0;
                frontier.push_back(i);
            }
        if (remove_if_unsolvable && frontier.empty())
            return false;
        {
            std::vector<std::vector<u32>> next(T);
            auto* D = reinterpret_cast<std::atomic<i32>*>(S.m_unit.data());
            static_assert(sizeof(std::atomic<i32>) == sizeof(i32));
            i32 d = 0;
            while (!frontier.empty())
            {
                ++d;
                team.run(
                    [&](u32 t)
                    {
                        next[t].clear();
                        auto [a, b] = Team::slice(frontier.size(), t, T);
                        for (u64 i = a; i < b; ++i)
                        {
                            const u32 v = frontier[i];
                            for (u64 r = S.m_boffsets[v]; r < S.m_boffsets[v + 1]; ++r)
                            {
                                const u32 u = S.m_bsources[r];
                                i32 e = k_unsolvable_distance;
                                if (D[u].load(std::memory_order_relaxed) == k_unsolvable_distance &&
                                    D[u].compare_exchange_strong(e, d, std::memory_order_relaxed))
                                    next[t].push_back(u);
                            }
                        }
                    });
                frontier.clear();
                for (auto& x : next)
                    frontier.insert(frontier.end(), x.begin(), x.end());
            }
        }
        if (remove_if_unsolvable && S.m_unit[0] == k_unsolvable_distance)
            return false;
        // cost goal distances: the unit distances for unit costs, else Dijkstra over the reverse CSR
        S.m_cost.assign(N, std::numeric_limits<f64>::infinity());
        if (S.m_costs.empty())
        {
            for (u64 i = 0; i < N; ++i)
                if (S.m_unit[i] != k_unsolvable_distance)
                    S.m_cost[i] = static_cast<f64>(S.m_unit[i]);
        }
        else
            dijkstra(S);
        // flags
        S.m_unsolvable.assign(N, 0);
        S.m_alive.assign(N, 0);
        S.m_num_goal = S.m_num_unsolvable = 0;
        S.m_max_unit = -1;
        for (u64 i = 0; i < N; ++i)
        {
            const bool g = S.m_goal[i] != 0, u = S.m_unit[i] == k_unsolvable_distance;
            S.m_unsolvable[i] = u;
            S.m_alive[i] = !g && !u;
            S.m_num_goal += g;
            S.m_num_unsolvable += u;
            if (!u)
                S.m_max_unit = std::max(S.m_max_unit, S.m_unit[i]);
        }
        return true;
    }

    /// Parallel inclusive scan of v[1..n] (v[0] = 0); returns the total.
    static u64 prefix(Team& team, std::vector<u64>& v, u64 n, std::vector<u64>& partial)
    {
        const u32 T = team.size();
        team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(n, t, T);
                u64 s = 0;
                for (u64 i = a; i < b; ++i)
                    s += v[i + 1];
                partial[t] = s;
            });
        u64 acc = 0;
        for (u32 t = 0; t < T; ++t)
        {
            const u64 s = partial[t];
            partial[t] = acc;
            acc += s;
        }
        team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(n, t, T);
                u64 s = partial[t];
                for (u64 i = a; i < b; ++i)
                {
                    s += v[i + 1];
                    v[i + 1] = s;
                }
            });
        return acc;
    }

private:
    static void fill_sources(StateSpace& S, Team& team)
    {
        // source of forward edge e: the row containing it (rows are in id order)
        const u32 T = team.size();
        const u64 N = S.m_n;
        std::vector<u32> src_of(S.m_targets.size());
        team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(N, t, T);
                for (u64 s = a; s < b; ++s)
                    for (u64 e = S.m_offsets[s]; e < S.m_offsets[s + 1]; ++e)
                        src_of[e] = static_cast<u32>(s);
            });
        const u64 E = S.m_targets.size();
        team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(E, t, T);
                for (u64 r = a; r < b; ++r)
                    S.m_bsources[r] = src_of[S.m_bedges[r]];
            });
    }

    static void dijkstra(StateSpace& S)
    {
        // backward from every goal state: d[source] = min over edges (source -> target) of d[target] + cost (as in
        // mimir's BGL call on the backward graph). With non-negative costs the result is the minimum over all edges,
        // independent of the order of ties.
        for (f64 c : S.m_costs)
            if (!(c >= 0))
                throw std::domain_error("mymyr: the state space has a negative or NaN transition cost; goal distances by Dijkstra "
                                        "need non-negative costs");
        using Item = std::pair<f64, u32>;
        std::priority_queue<Item, std::vector<Item>, std::greater<>> open;
        std::vector<u8> closed(S.m_n, 0);
        for (u32 i = 0; i < S.m_n; ++i)
            if (S.m_goal[i])
            {
                S.m_cost[i] = 0;
                open.emplace(0.0, i);
            }
        while (!open.empty())
        {
            const auto [d, v] = open.top();
            open.pop();
            if (closed[v])
                continue;
            closed[v] = 1;
            for (u64 r = S.m_boffsets[v]; r < S.m_boffsets[v + 1]; ++r)
            {
                const u32 u = S.m_bsources[r];
                const f64 nd = d + S.m_costs[S.m_bedges[r]];
                if (nd < S.m_cost[u])
                {
                    S.m_cost[u] = nd;
                    open.emplace(nd, u);
                }
            }
        }
    }

public:
    // fields for the generators
    static void set_basic(StateSpace& S, TaskPtr task, u32 n, u32 words, u32 nn, bool labels, u32 width, bool symmetric)
    {
        S.m_task = std::move(task);
        S.m_n = n;
        S.m_words = words;
        S.m_numeric_words = nn;
        S.m_has_labels = labels;
        S.m_label_width = width;
        S.m_symmetry_reduced = symmetric;
    }
    static void set_certificate(StateSpace& S, CertificateKind kind, u32 k, const KfwlLimits& limits)
    {
        S.m_certificate = kind;
        S.m_fwl_k = k;
        S.m_fwl_limits = limits;
    }
    static std::vector<u64>& states(StateSpace& S) { return S.m_states; }
    static std::vector<u64>& offsets(StateSpace& S) { return S.m_offsets; }
    static std::vector<u32>& targets(StateSpace& S) { return S.m_targets; }
    static std::vector<u32>& schemas(StateSpace& S) { return S.m_schemas; }
    static std::vector<u32>& bindings(StateSpace& S) { return S.m_bindings; }
    static std::vector<f64>& costs(StateSpace& S) { return S.m_costs; }
    static std::vector<u8>& goal(StateSpace& S) { return S.m_goal; }
    static void set_stats(StateSpace& S, u32 threads, u32 layers, f64 search_s, f64 post_s)
    {
        S.m_threads = threads;
        S.m_layers = layers;
        S.m_search_s = search_s;
        S.m_post_s = post_s;
    }
};

std::shared_ptr<const StateSpace> StateSpace::create(StateSpaceArrays&& a)
{
    if (!a.task)
        throw std::invalid_argument("mymyr: StateSpace::create needs a task");
    const u64 N = a.num_states, E = a.forward_targets.size(), RW = u64{a.words} + a.numeric_words;
    auto need = [](bool ok, const char* what)
    {
        if (!ok)
            throw std::invalid_argument(std::string("mymyr: StateSpace::create: ") + what);
    };
    need(a.state_words.size() == N * RW, "state_words is not [num_states, words + numeric_words]");
    need(a.forward_offsets.size() == N + 1 && a.forward_offsets.front() == 0 && a.forward_offsets.back() == E,
         "forward_offsets is not a CSR over the transitions");
    need(std::is_sorted(a.forward_offsets.begin(), a.forward_offsets.end()), "forward_offsets is not ascending");
    need(a.backward_offsets.size() == N + 1 && a.backward_offsets.front() == 0 && a.backward_offsets.back() == E &&
             std::is_sorted(a.backward_offsets.begin(), a.backward_offsets.end()),
         "backward_offsets is not a CSR over the transitions");
    need(a.backward_sources.size() == E && a.backward_edges.size() == E, "the backward arrays are not [num_transitions]");
    need(!a.labels || (a.label_schemas.size() == E && a.label_bindings.size() == E * a.label_width),
         "the labels are not [num_transitions] and [num_transitions, label_width]");
    need(a.labels || (a.label_schemas.empty() && a.label_bindings.empty()), "labels without has_labels");
    need(a.costs.empty() || a.costs.size() == E, "costs is not [num_transitions]");
    need(a.unit_goal_distances.size() == N && a.cost_goal_distances.size() == N && a.goal_flags.size() == N &&
             a.unsolvable_flags.size() == N && a.alive_flags.size() == N,
         "the distances and flags are not [num_states]");
    auto below = [](const std::vector<u32>& v, u64 n)  // every element < n (a max reduction: vectorized)
    {
        u32 m = 0;
        for (const u32 x : v)
            m = std::max(m, x);
        return v.empty() || m < n;
    };
    need(below(a.forward_targets, N), "a transition target is not a state");
    need(below(a.backward_sources, N), "a transition source is not a state");
    need(below(a.backward_edges, E), "a backward edge is not a transition");
    auto space = std::make_shared<StateSpace>();
    StateSpace& S = *space;
    StateSpaceBuilder::set_basic(S, std::move(a.task), a.num_states, a.words, a.numeric_words, a.labels,
                                 a.labels ? a.label_width : 0, false);
    S.m_states = std::move(a.state_words);
    S.m_offsets = std::move(a.forward_offsets);
    S.m_targets = std::move(a.forward_targets);
    S.m_schemas = std::move(a.label_schemas);
    S.m_bindings = std::move(a.label_bindings);
    S.m_costs = std::move(a.costs);
    S.m_boffsets = std::move(a.backward_offsets);
    S.m_bsources = std::move(a.backward_sources);
    S.m_bedges = std::move(a.backward_edges);
    S.m_unit = std::move(a.unit_goal_distances);
    S.m_cost = std::move(a.cost_goal_distances);
    S.m_goal = std::move(a.goal_flags);
    S.m_unsolvable = std::move(a.unsolvable_flags);
    S.m_alive = std::move(a.alive_flags);
    for (u64 i = 0; i < N; ++i)
    {
        S.m_num_goal += S.m_goal[i] != 0;
        S.m_num_unsolvable += S.m_unsolvable[i] != 0;
        if (S.m_unit[i] != k_unsolvable_distance)
            S.m_max_unit = std::max(S.m_max_unit, S.m_unit[i]);
    }
    StateSpaceBuilder::set_stats(S, a.threads, a.layers, a.search_seconds, a.post_seconds);
    return space;
}

namespace
{
using Clock = std::chrono::steady_clock;
f64 seconds_since(Clock::time_point t0) { return std::chrono::duration<f64>(Clock::now() - t0).count(); }

u32 max_arity(const Task& task)
{
    u32 k = 0;
    for (const auto& s : task.data().schemas)
        k = std::max(k, s.arity());
    return k;
}

/// Drops the cost array when every cost is exactly 1 (total-cost tasks whose actions all cost 1).
void drop_unit_costs(std::vector<f64>& costs)
{
    for (f64 c : costs)
        if (c != 1.0)
            return;
    std::vector<f64>().swap(costs);
}

// ------------------------------------------------------------------------------------------------- layered generator
struct alignas(64) Worker
{
    WorkspaceLease lease;
    Successors* succ = nullptr;
    LineVector<u64> next;
    std::vector<u32> src_id, src_cnt;  // per expanded source: id and number of transitions
    std::vector<u32> tdst;             // per transition: target handle
    std::vector<u32> tsch, targs;      // labels: schema, width binding objects
    std::vector<f64> tcost;            // costs (tasks with non-unit costs)
    u64 generated = 0;
};

enum class Stop : u8
{
    None,
    OutOfStates,
    Timeout,
};

class LayeredGenerator
{
public:
    LayeredGenerator(TaskPtr task, const StateSpaceOptions& o, u32 T)
        : m_taskp(std::move(task)), m_task(*m_taskp), m_o(o), m_T(T), m_team(T), m_ws(T), m_costs(m_task),
          m_width(max_arity(m_task))
    {
        m_timed = std::isfinite(o.max_seconds);
        m_limit = std::max<u64>(o.max_states, 2);  // mimir: fails iff the space has max(M, 2) states or more
        m_store.emplace(T, m_task.numeric_words());
        m_team.run(
            [&](u32 t)
            {
                m_ws[t].lease = m_task.workspace();
                m_ws[t].succ = &m_ws[t].lease->successors();
            });
        m_partial.assign(m_T + 1, 0);
        m_pend_lo.assign(m_T, 0);
        m_pend_hi.assign(m_T, 0);
        m_unit = m_costs.unit();
        m_metric = m_costs.kind() == heuristics::ActionCosts::Kind::StateMetric;
    }

    StateSpaceResult run()
    {
        StateSpaceResult r;
        const auto t0 = Clock::now();
        m_t0 = t0;
        if (m_task.compiled().goal.unsatisfiable)
        {
            r.status = StateSpaceStatus::Unsolvable;  // mimir's BrFS: a statically false goal ends it at once
            r.seconds = seconds_since(t0);
            return r;
        }
        const State& s0 = m_task.initial_state();
        m_loc.push_back(m_store->insert(0, s0.data(), s0.size_words(), ~u64{0}, s0.numeric().data()).first);
        m_store->begin_layer();
        m_layer_lo = 0;
        m_layer_hi = 1;
        m_goal.assign(1, 0);
        u32 depth = 0;
        while (m_layer_lo < m_layer_hi)
        {
            m_depth = depth++;
            const u64 L = m_layer_hi - m_layer_lo;
            m_chunk = std::clamp<u64>(L / (static_cast<u64>(m_T) * 32), 1, 256);
            m_chunk_ctr.store(0, std::memory_order_relaxed);
            std::fill(m_pend_lo.begin(), m_pend_lo.end(), 0);
            std::fill(m_pend_hi.begin(), m_pend_hi.end(), 0);
            for (;;)
            {
                m_team.run([&](u32 t) { work(t); });
                if (m_stop.load(std::memory_order_relaxed) != 0 || !m_resize.load(std::memory_order_relaxed))
                    break;
                m_store->rehash(m_team);
                m_resize.store(false, std::memory_order_relaxed);
            }
            if (const u8 s = m_stop.load(std::memory_order_relaxed))
            {
                r.status = s == static_cast<u8>(Stop::Timeout) ? StateSpaceStatus::Timeout : StateSpaceStatus::OutOfStates;
                r.states = std::max<u64>(m_layer_hi, m_store->approx_size());
                r.seconds = seconds_since(t0);
                return r;
            }
            finalize_layer();
            m_goal.resize(m_layer_hi, 0);
            if (m_layer_hi >= m_limit)
            {
                r.status = StateSpaceStatus::OutOfStates;
                r.states = m_layer_hi;
                r.seconds = seconds_since(t0);
                return r;
            }
            if (m_layer_hi >= (u64{1} << 31))
                throw std::length_error("mymyr: state spaces are limited to 2^31 - 1 states");
        }
        const f64 search_s = seconds_since(t0);
        const auto t1 = Clock::now();
        auto space = std::make_shared<StateSpace>();
        assemble(*space);
        if (!StateSpaceBuilder::finish(*space, m_team, m_o.remove_if_unsolvable))
        {
            r.status = StateSpaceStatus::Unsolvable;
            r.states = space->num_states();
            r.seconds = seconds_since(t0);
            return r;
        }
        StateSpaceBuilder::set_stats(*space, m_T, depth, search_s, seconds_since(t1));
        r.states = space->num_states();
        r.space = std::move(space);
        r.seconds = seconds_since(t0);
        return r;
    }

private:
    void work(u32 t)
    {
        u64 lo = m_pend_lo[t], hi = m_pend_hi[t];
        for (;;)
        {
            if (lo >= hi)
            {
                if (m_resize.load(std::memory_order_relaxed) || m_stop.load(std::memory_order_relaxed))
                    break;
                if (m_store->approx_size() >= m_limit)  // approx_size() <= size(): never a false stop
                {
                    m_stop.store(static_cast<u8>(Stop::OutOfStates), std::memory_order_relaxed);
                    break;
                }
                if (m_timed && seconds_since(m_t0) > m_o.max_seconds)
                {
                    m_stop.store(static_cast<u8>(Stop::Timeout), std::memory_order_relaxed);
                    break;
                }
                const u64 c = m_chunk_ctr.fetch_add(1, std::memory_order_relaxed);
                lo = m_layer_lo + c * m_chunk;
                if (lo >= m_layer_hi)
                    break;
                hi = std::min(m_layer_hi, lo + m_chunk);
            }
            expand(t, lo++);
            if (m_store->wants_rehash())
                m_resize.store(true, std::memory_order_relaxed);
            if (m_resize.load(std::memory_order_relaxed))
                break;  // stop after the current state; the rest of [lo, hi) resumes after the rehash
        }
        m_pend_lo[t] = lo;
        m_pend_hi[t] = hi;
    }

    void expand(u32 t, u64 id)
    {
        Worker& w = m_ws[t];
        Successors& succ = *w.succ;
        const StateView rec = m_store->record(m_loc[id]);  // records never move
        succ.prepare(rec);
        m_goal[id] = succ.goal_holds();
        w.src_id.push_back(static_cast<u32>(id));
        // the parent's metric value in mimir's cost formula (datasets/state_space.hpp)
        const f64 g = m_metric ? m_costs.initial(rec) : static_cast<f64>(m_depth);
        const bool labels = m_o.labels;
        const u32 K = m_width;
        u32 k = 0;
        succ.generate<false>(
            [&](u32 schema, const ObjectId* b, const Delta& d)
            {
                const u32 kk = k++;
                ++w.generated;
                const u32 nn = apply_delta(rec.w, rec.nw, d, w.next);
                w.tdst.push_back(m_store->insert(t, w.next.data(), nn, (id << 24) | kk, d.num).first);
                if (labels)
                {
                    w.tsch.push_back(schema);
                    const u32 arity = succ.arity(schema);
                    for (u32 i = 0; i < K; ++i)
                        w.targs.push_back(i < arity ? b[i].v : ~u32{0});
                }
                if (!m_unit)
                    w.tcost.push_back(m_costs.next(g, d) - g);
                return true;
            },
            false, true);
        w.src_cnt.push_back(k);
        if (k >= (u32{1} << 24))
            throw std::length_error("mymyr: more than 2^24 successors of one state");
    }

    void finalize_layer()
    {
        // deterministic ids: counting sort of the new states by discoverer (parent id, successor index)
        u64 total = 0;
        const u32 T = m_T;
        ConcurrentStateStore& store = *m_store;
        const u64 L = m_layer_hi - m_layer_lo;
        if (m_cursor_cap < L + 1)
        {
            m_cursor_cap = std::max<u64>(L + 1, m_cursor_cap * 2);
            m_cursor.reset(new std::atomic<u32>[m_cursor_cap]);
            m_offv.resize(m_cursor_cap + 1);
        }
        m_team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(L, t, T);
                for (u64 i = a; i < b; ++i)
                    m_cursor[i].store(0, std::memory_order_relaxed);
            });
        m_team.run(
            [&](u32 t)
            {
                for (u32 li = store.layer_start(t); li < store.local_size(t); ++li)
                    m_cursor[(store.key(store.make_handle(t, li)) >> 24) - m_layer_lo].fetch_add(1, std::memory_order_relaxed);
            });
        m_team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(L, t, T);
                u64 s = 0;
                for (u64 i = a; i < b; ++i)
                    s += m_cursor[i].load(std::memory_order_relaxed);
                m_partial[t] = s;
            });
        for (u32 t = 0; t < T; ++t)
        {
            const u64 s = m_partial[t];
            m_partial[t] = total;
            total += s;
        }
        if (m_tmp.size() < total)
            m_tmp.resize(total + total / 2);
        m_loc.resize(m_layer_hi + total);
        m_team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(L, t, T);
                u64 run = m_partial[t];
                for (u64 i = a; i < b; ++i)
                {
                    const u32 c = m_cursor[i].load(std::memory_order_relaxed);
                    m_offv[i] = static_cast<u32>(run);
                    m_cursor[i].store(static_cast<u32>(run), std::memory_order_relaxed);
                    run += c;
                }
                if (t == T - 1)
                    m_offv[L] = static_cast<u32>(run);
            });
        m_team.run(
            [&](u32 t)
            {
                for (u32 li = store.layer_start(t); li < store.local_size(t); ++li)
                {
                    const ConcurrentStateStore::Handle h = store.make_handle(t, li);
                    const u64 key = store.key(h);
                    const u32 pos = m_cursor[(key >> 24) - m_layer_lo].fetch_add(1, std::memory_order_relaxed);
                    m_tmp[pos] = ((key & 0xFFFFFFULL) << 32) | h;
                }
            });
        m_team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(L, t, T);
                for (u64 i = a; i < b; ++i)
                {
                    u64* lo = m_tmp.data() + m_offv[i];
                    u64* hi = m_tmp.data() + m_offv[i + 1];
                    for (u64* x = lo + 1; x < hi; ++x)  // insertion sort: segments are tiny, keys distinct
                    {
                        const u64 v = *x;
                        u64* y = x;
                        while (y > lo && *(y - 1) > v)
                        {
                            *y = *(y - 1);
                            --y;
                        }
                        *y = v;
                    }
                    for (u64* x = lo; x < hi; ++x)
                        m_loc[m_layer_hi + static_cast<u64>(x - m_tmp.data())] = static_cast<u32>(*x);
                }
            });
        store.begin_layer();
        m_layer_lo = m_layer_hi;
        m_layer_hi += total;
    }

    /// States, forward CSR, labels, costs and goal flags into `S`; frees the store and the per-thread buffers.
    void assemble(StateSpace& S)
    {
        const u32 T = m_T;
        const u64 N = m_layer_hi;
        ConcurrentStateStore& store = *m_store;
        const u32 W = std::max<u32>(1, m_task.words());
        const u32 NN = m_task.numeric_words();
        const u32 RW = W + NN;
        StateSpaceBuilder::set_basic(S, m_taskp, static_cast<u32>(N), W, NN, m_o.labels, m_o.labels ? m_width : 0, false);
        // handle -> id
        // handle decoding outlives the store: thread = h >> lb, local index = h & mask (lb = 32 for one thread)
        const u32 lb = T == 1 ? 32 : static_cast<u32>(std::countr_zero(store.make_handle(1, 0)));
        const u32 lmask = lb == 32 ? ~u32{0} : (u32{1} << lb) - 1;
        std::vector<std::vector<u32>> gid(T);
        m_team.run([&](u32 t) { gid[t].assign(store.local_size(t), ~u32{0}); });
        std::vector<u64>& states = StateSpaceBuilder::states(S);
        states.assign(N * RW, 0);
        m_team.run(
            [&](u32 t)
            {
                auto [a, b] = Team::slice(N, t, T);
                for (u64 id = a; id < b; ++id)
                {
                    const u32 h = m_loc[id];
                    gid[store.thread_of(h)][store.local_of(h)] = static_cast<u32>(id);
                    const StateView v = store.record(h);
                    u64* row = states.data() + id * RW;
                    std::copy(v.w, v.w + v.nw, row);
                    if (NN)
                        std::copy(v.num, v.num + NN, row + W);
                }
            });
        std::vector<u32>().swap(m_loc);
        m_store.reset();  // the records are copied
        // forward CSR rows
        std::vector<u64>& offsets = StateSpaceBuilder::offsets(S);
        offsets.assign(N + 1, 0);
        m_team.run(
            [&](u32 t)
            {
                const Worker& w = m_ws[t];
                for (usize i = 0; i < w.src_id.size(); ++i)
                    offsets[w.src_id[i] + 1] = w.src_cnt[i];
            });
        const u64 E = StateSpaceBuilder::prefix(m_team, offsets, N, m_partial);
        if (E >= (u64{1} << 32))
            throw std::length_error("mymyr: state spaces are limited to 2^32 - 1 transitions");
        std::vector<u32>& targets = StateSpaceBuilder::targets(S);
        std::vector<u32>& schemas = StateSpaceBuilder::schemas(S);
        std::vector<u32>& bindings = StateSpaceBuilder::bindings(S);
        std::vector<f64>& costs = StateSpaceBuilder::costs(S);
        const u32 K = m_width;
        targets.assign(E, 0);
        if (m_o.labels)
        {
            schemas.assign(E, 0);
            bindings.assign(E * K, 0);
        }
        if (!m_unit)
            costs.assign(E, 0);
        m_team.run(
            [&](u32 t)
            {
                Worker& w = m_ws[t];
                u64 p = 0;
                for (usize i = 0; i < w.src_id.size(); ++i)
                {
                    const u64 o = offsets[w.src_id[i]];
                    for (u32 k = 0; k < w.src_cnt[i]; ++k, ++p)
                    {
                        const u32 h = w.tdst[p];
                        targets[o + k] = gid[lb == 32 ? 0 : h >> lb][h & lmask];
                        if (m_o.labels)
                        {
                            schemas[o + k] = w.tsch[p];
                            std::copy(w.targs.begin() + static_cast<i64>(p * K), w.targs.begin() + static_cast<i64>((p + 1) * K),
                                      bindings.begin() + static_cast<i64>((o + k) * K));
                        }
                        if (!m_unit)
                            costs[o + k] = w.tcost[p];
                    }
                }
                std::vector<u32>().swap(w.tdst);
                std::vector<u32>().swap(w.tsch);
                std::vector<u32>().swap(w.targs);
                std::vector<f64>().swap(w.tcost);
                std::vector<u32>().swap(w.src_id);
                std::vector<u32>().swap(w.src_cnt);
            });
        if (!m_unit)
            drop_unit_costs(costs);
        StateSpaceBuilder::goal(S).swap(m_goal);
    }

    TaskPtr m_taskp;
    const Task& m_task;
    const StateSpaceOptions& m_o;
    u32 m_T;
    Team m_team;
    std::optional<ConcurrentStateStore> m_store;
    std::vector<Worker> m_ws;
    heuristics::ActionCosts m_costs;
    u32 m_width = 0;
    bool m_unit = true, m_metric = false;
    bool m_timed = true;
    u64 m_limit = 2;
    Clock::time_point m_t0;
    u32 m_depth = 0;
    std::vector<u32> m_loc;  // id -> handle
    std::vector<u8> m_goal;
    u64 m_layer_lo = 0, m_layer_hi = 0;
    std::atomic<u64> m_chunk_ctr{0};
    u64 m_chunk = 1;
    std::atomic<bool> m_resize{false};
    std::atomic<u8> m_stop{0};
    std::vector<u64> m_pend_lo, m_pend_hi, m_partial;
    std::unique_ptr<std::atomic<u32>[]> m_cursor;
    u64 m_cursor_cap = 0;
    std::vector<u32> m_offv;
    std::vector<u64> m_tmp;
};

// ------------------------------------------------------------------------------------------------ sequential generator
/// One thread (threads == 1, the instance pool): a plain breadth-first search over a FlatStateStore. Parents are
/// expanded in id order and successors in canonical order, so the insertion order of the states is the order of
/// their smallest discoverer key (parent id, successor index) within each layer: the same ids as the layered
/// generator, without its per-layer sort, handle maps and per-thread arenas (whose fixed setup cost dominates on the
/// small instances of a dataset).
class SequentialGenerator
{
public:
    SequentialGenerator(TaskPtr task, const StateSpaceOptions& o)
        : m_taskp(std::move(task)), m_task(*m_taskp), m_o(o), m_costs(m_task), m_width(max_arity(m_task))
    {
    }

    StateSpaceResult run()
    {
        StateSpaceResult r;
        const auto t0 = Clock::now();
        if (m_task.compiled().goal.unsatisfiable)
        {
            r.status = StateSpaceStatus::Unsolvable;  // mimir's BrFS: a statically false goal ends it at once
            r.seconds = seconds_since(t0);
            return r;
        }
        const bool timed = std::isfinite(m_o.max_seconds);
        const bool unit = m_costs.unit(), metric = m_costs.kind() == heuristics::ActionCosts::Kind::StateMetric;
        const bool labels = m_o.labels;
        const u32 K = m_width;
        const u32 NN = m_task.numeric_words();
        const u64 limit = std::max<u64>(m_o.max_states, 2);  // mimir: fails iff the space has max(M, 2) states or more
        FlatStateStore store(std::max<u32>(1, m_task.words()), 10, NN);
        const WorkspaceLease lease = m_task.workspace();
        Successors& succ = lease->successors();
        const State& s0 = m_task.initial_state();
        store.insert(s0.data(), s0.size_words(), s0.numeric().data());
        std::vector<u32> depth{0};
        std::vector<u8> goal;
        std::vector<u64> offsets{0};
        std::vector<u32> targets, schemas, bindings;
        std::vector<f64> costs;
        std::vector<u64> cur;  // the parent's words (the store's arena moves when it grows)
        LineVector<u64> next;
        for (u32 id = 0; id < store.size(); ++id)
        {
            if (timed && (id & 255) == 0 && seconds_since(t0) > m_o.max_seconds)
            {
                r.status = StateSpaceStatus::Timeout;
                r.states = store.size();
                r.seconds = seconds_since(t0);
                return r;
            }
            const StateView sv = store[StateId{id}];
            cur.assign(sv.w, sv.w + static_cast<usize>(sv.nw) + NN);
            const StateView rec{cur.data(), sv.nw, NN ? cur.data() + sv.nw : nullptr, NN};
            succ.prepare(rec);
            goal.push_back(succ.goal_holds());
            const f64 g = metric ? m_costs.initial(rec) : static_cast<f64>(depth[id]);
            const u32 d1 = depth[id] + 1;
            bool full = false;
            succ.generate<false>(
                [&](u32 schema, const ObjectId* b, const Delta& d)
                {
                    const u32 nn = apply_delta(rec.w, rec.nw, d, next);
                    const auto [t, fresh] = store.insert(next.data(), nn, d.num);
                    if (fresh)
                    {
                        depth.push_back(d1);
                        if (store.size() >= limit)
                            full = true;
                    }
                    targets.push_back(t.v);
                    if (labels)
                    {
                        schemas.push_back(schema);
                        const u32 arity = succ.arity(schema);
                        for (u32 i = 0; i < K; ++i)
                            bindings.push_back(i < arity ? b[i].v : ~u32{0});
                    }
                    if (!unit)
                        costs.push_back(m_costs.next(g, d) - g);
                    return true;
                },
                false, true);
            if (full)
            {
                r.status = StateSpaceStatus::OutOfStates;
                r.states = store.size();
                r.seconds = seconds_since(t0);
                return r;
            }
            offsets.push_back(targets.size());
            if (targets.size() >= (u64{1} << 32))
                throw std::length_error("mymyr: state spaces are limited to 2^32 - 1 transitions");
            if (store.size() >= (u64{1} << 31))
                throw std::length_error("mymyr: state spaces are limited to 2^31 - 1 states");
        }
        const f64 search_s = seconds_since(t0);
        const auto t1 = Clock::now();
        auto space = std::make_shared<StateSpace>();
        StateSpace& S = *space;
        const u32 N = store.size();
        const u32 W = std::max<u32>(1, m_task.words());
        const u32 RW = W + NN;
        StateSpaceBuilder::set_basic(S, m_taskp, N, W, NN, labels, labels ? K : 0, false);
        std::vector<u64>& rows = StateSpaceBuilder::states(S);
        if (store.stride() == W)
            rows.assign(store.words(StateId{0}), store.words(StateId{0}) + static_cast<usize>(N) * RW);
        else
        {
            // lazy slots: the store's stride may exceed the final width (it grows by doubling); records are zero
            // beyond their trimmed size, which never exceeds the task's width
            rows.assign(static_cast<usize>(N) * RW, 0);
            const u32 cw = std::min(W, store.stride());
            for (u32 i = 0; i < N; ++i)
            {
                const u64* src = store.words(StateId{i});
                u64* row = rows.data() + static_cast<usize>(i) * RW;
                std::copy(src, src + cw, row);
                if (NN)
                    std::copy(src + store.stride(), src + store.stride() + NN, row + W);
            }
        }
        StateSpaceBuilder::offsets(S).swap(offsets);
        StateSpaceBuilder::targets(S).swap(targets);
        StateSpaceBuilder::schemas(S).swap(schemas);
        StateSpaceBuilder::bindings(S).swap(bindings);
        if (!unit)
            drop_unit_costs(costs);
        StateSpaceBuilder::costs(S).swap(costs);
        StateSpaceBuilder::goal(S).swap(goal);
        Team team(1);
        if (!StateSpaceBuilder::finish(S, team, m_o.remove_if_unsolvable))
        {
            r.status = StateSpaceStatus::Unsolvable;
            r.states = N;
            r.seconds = seconds_since(t0);
            return r;
        }
        StateSpaceBuilder::set_stats(S, 1, depth.back() + 1, search_s, seconds_since(t1));
        r.states = N;
        r.space = std::move(space);
        r.seconds = seconds_since(t0);
        return r;
    }

private:
    TaskPtr m_taskp;
    const Task& m_task;
    const StateSpaceOptions& m_o;
    heuristics::ActionCosts m_costs;
    u32 m_width = 0;
};

// ------------------------------------------------------------------------------------------------- symmetry pruning
/// State space generation with symmetry pruning (as in mimir's SymmetryReducedProblemGraphEventHandler with
/// SymmetryStatePruning), single-threaded: a breadth-first search that keeps the first state of every certificate
/// class as its representative and expands only representatives. A transition from a representative to a successor
/// becomes an edge to the successor's class; the first transition to each class is kept, but, for a class that the
/// same expansion creates, also one parallel edge, matching mimir's behaviour.
class SymmetricGenerator
{
public:
    SymmetricGenerator(TaskPtr task, const StateSpaceOptions& o) : m_taskp(std::move(task)), m_task(*m_taskp), m_o(o), m_costs(m_task) {}

    StateSpaceResult run()
    {
        StateSpaceResult r;
        const auto t0 = Clock::now();
        if (m_task.compiled().goal.unsatisfiable)
        {
            r.status = StateSpaceStatus::Unsolvable;
            r.seconds = seconds_since(t0);
            return r;
        }
        const bool timed = std::isfinite(m_o.max_seconds);
        const u32 NN = m_task.numeric_words();
        const u32 K = max_arity(m_task);
        const bool unit = m_costs.unit(), metric = m_costs.kind() == heuristics::ActionCosts::Kind::StateMetric;
        ObjectGraphBuilder ogb(m_task);
        ObjectGraph graph;
        auto certificate = [&](StateView s)
        {
            ogb.build(s, graph);
            return m_o.certificate == CertificateKind::KFwl ? kfwl_certificate(graph, m_o.fwl_k, m_o.fwl_limits)
                                                            : color_refinement_certificate(graph);
        };
        std::unordered_map<Certificate, u32, CertificateHash> class_of;
        std::unordered_map<State, u32> seen;  // every generated state -> its class
        std::vector<State> reps;               // the representative of each class
        std::vector<u32> depth;
        std::vector<u8> goal;
        std::vector<u64> offsets{0};
        std::vector<u32> targets, schemas, bindings;
        std::vector<f64> costs;
        const State& s0 = m_task.initial_state();
        class_of.emplace(certificate(s0.view()), 0);
        reps.push_back(s0);
        depth.push_back(0);
        seen.emplace(s0, 0);
        const WorkspaceLease lease = m_task.workspace();
        Successors& succ = lease->successors();
        std::vector<u64> next;
        // the successors of the current representative, collected first: certificates evaluate axioms on this
        // thread's workspace, which must not happen inside the generator's callback
        std::vector<State> succ_states;
        std::vector<u32> succ_schema, succ_binding;
        std::vector<f64> succ_cost;
        std::vector<u32> hit;  // classes reached from the current source
        for (u32 id = 0; id < reps.size(); ++id)
        {
            if (timed && seconds_since(t0) > m_o.max_seconds)
            {
                r.status = StateSpaceStatus::Timeout;
                r.states = seen.size();
                r.seconds = seconds_since(t0);
                return r;
            }
            const State cur = reps[id];
            const StateView sv = cur.view();
            succ.prepare(sv);
            goal.push_back(succ.goal_holds());
            const f64 g = metric ? m_costs.initial(sv) : static_cast<f64>(depth[id]);
            succ_states.clear();
            succ_schema.clear();
            succ_binding.clear();
            succ_cost.clear();
            succ.generate<false>(
                [&](u32 schema, const ObjectId* b, const Delta& d)
                {
                    const u32 nn = apply_delta(sv.w, sv.nw, d, next);
                    succ_states.emplace_back(next.data(), nn, d.num, NN);
                    succ_schema.push_back(schema);
                    const u32 arity = succ.arity(schema);
                    for (u32 i = 0; i < K; ++i)
                        succ_binding.push_back(i < arity ? b[i].v : ~u32{0});
                    if (!unit)
                        succ_cost.push_back(m_costs.next(g, d) - g);
                    return true;
                },
                false, true);
            hit.clear();
            for (usize j = 0; j < succ_states.size(); ++j)
            {
                State& t = succ_states[j];
                u32 c;
                if (auto it = seen.find(t); it != seen.end())
                    c = it->second;
                else
                {
                    const Certificate cert = certificate(t.view());
                    bool fresh = false;
                    if (auto jt = class_of.find(cert); jt != class_of.end())
                        c = jt->second;
                    else
                    {
                        c = static_cast<u32>(reps.size());
                        class_of.emplace(cert, c);
                        reps.push_back(t);
                        depth.push_back(depth[id] + 1);
                        fresh = true;
                    }
                    seen.emplace(std::move(t), c);
                    // mimir's BrFS: after a new (unpruned) state, fail if it has created max_states states or more
                    // (every generated state counts, representative or not)
                    if (fresh && seen.size() >= std::max<u64>(m_o.max_states, 2))
                    {
                        r.status = StateSpaceStatus::OutOfStates;
                        r.states = seen.size();
                        r.seconds = seconds_since(t0);
                        return r;
                    }
                }
                if (std::find(hit.begin(), hit.end(), c) != hit.end())
                    continue;  // no parallel edges between classes
                hit.push_back(c);
                targets.push_back(c);
                if (m_o.labels)
                {
                    schemas.push_back(succ_schema[j]);
                    bindings.insert(bindings.end(), succ_binding.begin() + static_cast<i64>(j * K),
                                    succ_binding.begin() + static_cast<i64>((j + 1) * K));
                }
                if (!unit)
                    costs.push_back(succ_cost[j]);
            }
            offsets.push_back(targets.size());
        }
        const f64 search_s = seconds_since(t0);
        const auto t1 = Clock::now();
        auto space = std::make_shared<StateSpace>();
        StateSpace& S = *space;
        const u32 N = static_cast<u32>(reps.size());
        const u32 W = std::max<u32>(1, m_task.words());
        const u32 RW = W + NN;
        StateSpaceBuilder::set_basic(S, m_taskp, N, W, NN, m_o.labels, m_o.labels ? K : 0, true);
        StateSpaceBuilder::set_certificate(S, m_o.certificate, m_o.fwl_k, m_o.fwl_limits);
        std::vector<u64>& rows = StateSpaceBuilder::states(S);
        rows.assign(static_cast<u64>(N) * RW, 0);
        for (u32 i = 0; i < N; ++i)
        {
            u64* row = rows.data() + static_cast<u64>(i) * RW;
            std::copy(reps[i].data(), reps[i].data() + reps[i].size_words(), row);
            if (NN)
                std::copy(reps[i].numeric().begin(), reps[i].numeric().end(), row + W);
        }
        StateSpaceBuilder::offsets(S).swap(offsets);
        StateSpaceBuilder::targets(S).swap(targets);
        StateSpaceBuilder::schemas(S).swap(schemas);
        StateSpaceBuilder::bindings(S).swap(bindings);
        if (!unit)
            drop_unit_costs(costs);
        StateSpaceBuilder::costs(S).swap(costs);
        StateSpaceBuilder::goal(S).swap(goal);
        Team team(1);
        if (!StateSpaceBuilder::finish(S, team, m_o.remove_if_unsolvable))
        {
            r.status = StateSpaceStatus::Unsolvable;
            r.states = N;
            r.seconds = seconds_since(t0);
            return r;
        }
        u32 layers = 0;
        for (u32 d : depth)
            layers = std::max(layers, d + 1);
        StateSpaceBuilder::set_stats(S, 1, layers, search_s, seconds_since(t1));
        r.states = N;
        r.space = std::move(space);
        r.seconds = seconds_since(t0);
        return r;
    }

private:
    TaskPtr m_taskp;
    const Task& m_task;
    const StateSpaceOptions& m_o;
    heuristics::ActionCosts m_costs;
};
}  // namespace

StateSpaceResult generate_state_space(TaskPtr task, const StateSpaceOptions& options)
{
    if (!task)
        throw std::invalid_argument("mymyr: generate_state_space needs a task");
    if (options.symmetry_pruning)
    {
        if (options.certificate == CertificateKind::KFwl && (options.fwl_k < 2 || options.fwl_k > 4))
            throw std::invalid_argument("mymyr: k-FWL certificates support k = 2, 3 and 4");
        SymmetricGenerator g(std::move(task), options);
        return g.run();
    }
    const u32 T = resolve_threads(options.threads);
    if (T == 1)
    {
        SequentialGenerator g(std::move(task), options);
        return g.run();
    }
    LayeredGenerator g(std::move(task), options, T);
    return g.run();
}

void for_each_state_space(u64 count, const std::function<TaskPtr(u64)>& make_task,
                          const std::function<void(u64, StateSpaceResult&&)>& consume, const StateSpaceOptions& options, u32 threads)
{
    const u64 T = std::min<u64>(resolve_threads(threads), std::max<u64>(count, 1));
    StateSpaceOptions one = options;
    one.threads = 1;
    std::atomic<u64> next{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto worker = [&](u32 = 0)
    {
        for (;;)
        {
            const u64 i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count || failed.load(std::memory_order_relaxed))
                return;
            try
            {
                consume(i, generate_state_space(make_task(i), one));
            }
            catch (...)
            {
                std::lock_guard lock(error_mutex);
                if (!error)
                    error = std::current_exception();
                failed.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    start_threads(pool, static_cast<u32>(T - 1), worker, [&] { failed.store(true, std::memory_order_relaxed); });
    worker();
    for (auto& th : pool)
        th.join();
    if (error)
        std::rethrow_exception(error);
}

std::vector<StateSpaceResult> generate_state_spaces(std::span<const TaskPtr> tasks, const StateSpaceOptions& options, u32 threads)
{
    std::vector<StateSpaceResult> out(tasks.size());
    for_each_state_space(
        tasks.size(), [&](u64 i) { return tasks[i]; }, [&](u64 i, StateSpaceResult&& r) { out[i] = std::move(r); }, options, threads);
    return out;
}
}  // namespace mymyr::datasets
