// Breadth-first search. The single-threaded loop runs over a flat, chunked or compact store; the multi-threaded one
// is a layer-synchronous search with a choice of successor id scheme (dedup "cas", deterministic "det" or per-thread
// "ranges" ids).

#include "mymyr/search/brfs.hpp"

#include "beam_detail.hpp"
#include "layer_order_detail.hpp"

#include "mymyr/core/team.hpp"
#include "mymyr/search/goal.hpp"
#include "mymyr/state/chunked_store.hpp"
#include "mymyr/state/compact_store.hpp"
#include "mymyr/state/concurrent_store.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace mymyr
{
u64 brfs_fingerprint_term(u64 id, u64 canonical_state_hash)
{
    return hash::mix64(id * 0x9E3779B97F4A7C15ULL ^ canonical_state_hash);
}

namespace
{
using Clock = std::chrono::steady_clock;

/// BrfsOptions::goal other than the task's goal, in a state `succ` was prepared on.
[[gnu::noinline]] bool other_goal(const search::GoalSpec& g, Successors& succ, StateView s)
{
    if (g.kind == search::GoalSpec::Kind::Custom)
        return g.test(s);
    for (const search::GoalSpec::AtomGoal& a : g.goals)
        if (search::holds(a, succ, s))
            return true;
    return false;
}

/// Goal test of a state `succ` was prepared on: the task's goal, or BrfsOptions::goal.
inline bool is_goal(const search::GoalSpec& g, Successors& succ, StateView s)
{
    if (g.kind == search::GoalSpec::Kind::Task) [[likely]]
        return succ.goal_holds();
    return other_goal(g, succ, s);
}
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

/// Per-layer counts (BrfsOptions::layer_stats) and the observer's on_pass: open() at the start of a layer, close() at
/// its end.
struct LayerLog
{
    std::vector<std::array<u64, 3>>* out = nullptr;
    search::SearchObserver* obs = nullptr;
    u64 e0 = 0, g0 = 0, end = 0;
    u32 depth = 0;
    Clock::time_point t0{};
    bool open_ = false;

    void open(u64 expanded, u64 generated, u64 stored)
    {
        e0 = expanded;
        g0 = generated;
        end = stored;
        open_ = true;
        if (obs)
            t0 = Clock::now();
    }
    void close(u64 expanded, u64 generated, u64 stored)
    {
        if (!open_)
            return;
        if (out)
            out->push_back({expanded - e0, generated - g0, stored - end});
        if (obs)
            obs->on_pass(depth, {.expanded = expanded - e0, .generated = generated - g0, .states = stored - end,
                                 .seconds = seconds_since(t0)});
        ++depth;
        open_ = false;
    }
};

/// The stop conditions of BrfsOptions (max_seconds, cancel, on_progress) and the lifecycle events of the observer,
/// for a single-threaded search and the calling thread of the multi-threaded one.
class Control
{
public:
    Control(const BrfsOptions& o, search::SearchObserver* obs)
        : m_o(o), m_obs(obs), m_timed(o.max_seconds < std::numeric_limits<double>::infinity()),
          m_next_progress(std::max<u64>(1, o.progress_interval))
    {
        if (m_timed)
            m_deadline = m_t0 + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(std::max(0.0, o.max_seconds)));
    }

    [[nodiscard]] search::SearchObserver* observer() const { return m_obs; }
    [[nodiscard]] Clock::time_point deadline() const { return m_deadline; }
    [[nodiscard]] bool timed() const { return m_timed; }

    void start(StateView initial) const
    {
        if (m_obs)
            m_obs->on_start(initial);
    }

    /// Before an expansion: false once max_seconds, the token or on_progress stops the search.
    bool keep_going(const BrfsResult& r, u64 states)
    {
        if (--m_countdown != 0) [[likely]]
            return true;
        m_countdown = k_check_every;
        if (!check())
            return false;
        if (m_obs && r.expanded >= m_next_progress)
        {
            m_next_progress = r.expanded + std::max<u64>(1, m_o.progress_interval);
            if (!m_obs->on_progress(statistics(r.expanded, r.generated, states)))
            {
                stop(search::SearchStatus::Cancelled);
                return false;
            }
        }
        return true;
    }

    /// The time and the token: false (and the reason recorded) once one of them stops the search.
    bool check()
    {
        if (m_timed && Clock::now() >= m_deadline)
            stop(search::SearchStatus::OutOfTime);
        else if (m_o.cancel.requested())
            stop(search::SearchStatus::Cancelled);
        return !m_stopped;
    }
    void stop(search::SearchStatus why)
    {
        if (!m_stopped)
            m_stopped = why;
    }

    [[nodiscard]] search::SearchStatistics statistics(u64 expanded, u64 generated, u64 states) const
    {
        return {.expanded = expanded, .generated = generated, .states = states, .seconds = seconds_since(m_t0)};
    }

    /// Sets r.status and sends on_solution (when solved) and on_end.
    void finish(BrfsResult& r) const
    {
        if (r.solved)
            r.status = search::SearchStatus::Solved;
        else if (m_stopped)
            r.status = *m_stopped;
        else
            r.status = r.exhausted ? search::SearchStatus::Exhausted : search::SearchStatus::OutOfStates;
        if (!m_obs)
            return;
        if (r.solved)
            m_obs->on_solution(r.plan, static_cast<double>(r.plan.size()));
        m_obs->on_end(r.status, statistics(r.expanded, r.generated, r.states));
    }

private:
    const BrfsOptions& m_o;
    search::SearchObserver* m_obs;
    Clock::time_point m_t0 = Clock::now();
    Clock::time_point m_deadline{};
    bool m_timed;
    u64 m_next_progress;
    static constexpr u32 k_check_every = 8;  // expansions between checks of the time, the token and on_progress
    u32 m_countdown = 1;
    std::optional<search::SearchStatus> m_stopped;
};

/// The action of a successor as the observer gets it.
inline Action action_of(Successors& succ, u32 schema, const ObjectId* binding)
{
    return {SchemaId{schema}, std::vector<ObjectId>(binding, binding + succ.arity(schema))};
}

/// SoA search nodes of the single-threaded searches: parent, schema and binding of each state's first discovery.
struct Nodes
{
    std::vector<u32> parent, schema;
    std::vector<u64> offset;
    std::vector<ObjectId> binding;

    void root()
    {
        parent.push_back(~u32{0});
        schema.push_back(~u32{0});
        offset.push_back(0);
    }
    void push(u32 p, u32 s, const ObjectId* b, u32 arity)
    {
        parent.push_back(p);
        schema.push_back(s);
        offset.push_back(binding.size());
        binding.insert(binding.end(), b, b + arity);
    }
    std::vector<Action> plan(u32 id, const Successors& succ) const
    {
        std::vector<Action> out;
        for (u32 v = id; parent[v] != ~u32{0}; v = parent[v])
        {
            const u32 s = schema[v];
            const ObjectId* b = binding.data() + offset[v];
            out.emplace_back(SchemaId{s}, std::vector<ObjectId>(b, b + const_cast<Successors&>(succ).arity(s)));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
    [[nodiscard]] u64 bytes() const
    {
        return parent.capacity() * 4 + schema.capacity() * 4 + offset.capacity() * 8 + binding.capacity() * 4;
    }
};

// ------------------------------------------------------------------------------------------------- flat
template<bool Ordered, bool Observed>
BrfsResult run_flat(const Task& task, const BrfsOptions& o, Successors& succ, Control& ctl)
{
    BrfsResult r;
    r.store = "flat";
    const bool witness = o.witness_pruning, canonical = o.canonical_order;
    const u32 NN = task.numeric_words();
    FlatStateStore store(std::max<u32>(1, task.words()), 16, NN);
    Nodes nodes;
    const State& s0 = task.initial_state();
    store.insert(s0.view());
    nodes.root();
    std::vector<u64> cur, next;
    search::SearchObserver* const obs = Observed ? ctl.observer() : nullptr;
    ctl.start(s0.view());
    const auto t0 = Clock::now();
    u32 layer_end = 0;
    u32 pos = 0;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr, obs};
    // Ordered layers (BrfsOptions::layers): the layer [layer_begin, layer_end) of ids is expanded in the order `order`
    // (with a beam, only its first beam_width ids; the others stay stored and are never expanded)
    [[maybe_unused]] search::detail::LayerOrderer lo;
    [[maybe_unused]] std::vector<u32> order;
    [[maybe_unused]] u32 layer_begin = 0;
    [[maybe_unused]] bool truncated = false;
    if constexpr (Ordered)
        lo = search::detail::LayerOrderer(task, o.layers);
    for (; pos < store.size() && store.size() < o.max_states; ++pos)
    {
        if (!ctl.keep_going(r, store.size()))
            break;
        if (pos == layer_end)
        {
            log.close(r.expanded, r.generated, store.size());
            if (r.layers == o.max_depth)
                break;
            ++r.layers;
            layer_begin = pos;
            layer_end = store.size();
            log.open(r.expanded, r.generated, layer_end);
            if constexpr (Ordered)
            {
                order.resize(layer_end - layer_begin);
                std::iota(order.begin(), order.end(), layer_begin);
                if (layer_begin > 0)
                    lo.select(order, succ, [&](u32 e) {
                        const u64* w = store.words(StateId{e});
                        return StateView{w, store.stride(), NN ? w + store.stride() : nullptr, NN};
                    });
                truncated = false;
            }
        }
        const u32 id = Ordered ? order[pos - layer_begin] : pos;
        const u32 W = store.stride();
        const u64* rec = store.words(StateId{id});
        cur.assign(rec, rec + store.record_words());  // [bits | numeric]: the arena may move during the expansion
        const u32 n = bits::trimmed_size(cur.data(), W);
        const StateView sv{cur.data(), n, NN ? cur.data() + W : nullptr, NN};
        succ.prepare(sv);
        const bool goal = is_goal(o.goal, succ, sv);
        r.goal_states += goal;
        if (goal && o.stop_at_goal)
        {
            r.solved = true;
            r.plan = nodes.plan(id, succ);
            break;
        }
        ++r.expanded;
        if constexpr (Observed)
            obs->on_expand(id, sv);
        succ.generate<Ordered>(
            [&](u32 s, const ObjectId* b, const Delta& d)
            {
                ++r.generated;
                const u32 nn = apply_delta(cur.data(), n, d, next);
                const auto [child, fresh] = store.insert(next.data(), nn, d.num);
                if constexpr (Observed)
                    obs->on_generate(id, action_of(succ, s, b), child.v, {next.data(), nn, d.num, NN}, fresh);
                if (fresh)
                {
                    nodes.push(id, s, b, succ.arity(s));
                    if constexpr (Ordered)
                        if (lo.limited() && store.size() - layer_end >= lo.limit())
                        {
                            truncated = true;
                            return false;
                        }
                }
                return true;
            },
            witness, canonical);
        if constexpr (Ordered)
            if (truncated || pos + 1 - layer_begin == order.size())
                pos = layer_end - 1;  // a full next layer drops the rest of this one (mimir); the beam's last entry ends it
    }
    log.close(r.expanded, r.generated, store.size());
    r.search_s = seconds_since(t0);
    r.exhausted = !r.solved && pos == store.size();
    r.states = store.size();
    r.words = store.stride();
    r.store_bytes = store.bytes() + nodes.bytes();
    if (o.fingerprint)
        for (u32 i = 0; i < store.size(); ++i)
            r.fingerprint ^= brfs_fingerprint_term(i, task.canonical_hash(store[StateId{i}]));
    ctl.finish(r);
    return r;
}

// ------------------------------------------------------------------------------------------------- chunked
template<bool Ordered, bool Observed>
BrfsResult run_chunked(const Task& task, const BrfsOptions& o, Successors& succ, Control& ctl)
{
    BrfsResult r;
    r.store = "chunked";
    const bool witness = o.witness_pruning, canonical = o.canonical_order;
    const u32 NN = task.numeric_words();
    ChunkedStateStore store(std::max<u32>(1, task.words()), NN);
    Nodes nodes;
    const State& s0 = task.initial_state();
    store.insert(s0.view());
    nodes.root();
    std::vector<u64> cur, next, curnum(NN), scratch;
    search::SearchObserver* const obs = Observed ? ctl.observer() : nullptr;
    ctl.start(s0.view());
    const auto t0 = Clock::now();
    u32 layer_end = 0;
    u32 pos = 0;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr, obs};
    // Ordered layers (BrfsOptions::layers): the layer [layer_begin, layer_end) of ids is expanded in the order `order`
    // (with a beam, only its first beam_width ids; the others stay stored and are never expanded)
    [[maybe_unused]] search::detail::LayerOrderer lo;
    [[maybe_unused]] std::vector<u32> order;
    [[maybe_unused]] u32 layer_begin = 0;
    [[maybe_unused]] bool truncated = false;
    if constexpr (Ordered)
        lo = search::detail::LayerOrderer(task, o.layers);
    for (; pos < store.size() && store.size() < o.max_states; ++pos)
    {
        if (!ctl.keep_going(r, store.size()))
            break;
        if (pos == layer_end)
        {
            log.close(r.expanded, r.generated, store.size());
            if (r.layers == o.max_depth)
                break;
            ++r.layers;
            layer_begin = pos;
            layer_end = store.size();
            log.open(r.expanded, r.generated, layer_end);
            if constexpr (Ordered)
            {
                order.resize(layer_end - layer_begin);
                std::iota(order.begin(), order.end(), layer_begin);
                if (layer_begin > 0)
                    lo.select(order, succ, [&](u32 e) {
                        scratch.resize(store.words());
                        store.decode(StateId{e}, scratch.data(), curnum.data());
                        return StateView{scratch.data(), store.words(), NN ? curnum.data() : nullptr, NN};
                    });
                truncated = false;
            }
        }
        const u32 id = Ordered ? order[pos - layer_begin] : pos;
        u32 W = store.words();
        cur.resize(W);
        next.resize(W);
        store.decode(StateId{id}, cur.data(), curnum.data());
        const u32 n = bits::trimmed_size(cur.data(), W);
        const StateView sv{cur.data(), n, NN ? curnum.data() : nullptr, NN};
        succ.prepare(sv);
        const bool goal = is_goal(o.goal, succ, sv);
        r.goal_states += goal;
        if (goal && o.stop_at_goal)
        {
            r.solved = true;
            r.plan = nodes.plan(id, succ);
            break;
        }
        ++r.expanded;
        if constexpr (Observed)
            obs->on_expand(id, sv);
        succ.generate<Ordered>(
            [&](u32 s, const ObjectId* b, const Delta& d)
            {
                ++r.generated;
                u32 need = 0;
                MYMYR_NOVECTOR
                for (SlotId a : d.add)
                    need = std::max<u32>(need, bits::word_of(a.v) + 1);
                if (need > W)
                {
                    // more atoms were interned (lazy slots): extend every state with zero chunks
                    store.widen(std::max(W * 2, need));
                    W = store.words();
                    cur.resize(W, 0);
                    next.resize(W, 0);
                    succ.engine().set_state(cur.data(), n);  // cur may have moved; the view is unchanged
                }
                std::memcpy(next.data(), cur.data(), W * sizeof(u64));
                for (SlotId x : d.del)
                    if (bits::word_of(x.v) < W)
                        bits::reset(next.data(), x.v);
                for (SlotId x : d.add)
                    bits::set(next.data(), x.v);
                const auto [child, fresh] = store.insert_successor(StateId{id}, cur.data(), next.data(), d);
                if constexpr (Observed)
                    obs->on_generate(id, action_of(succ, s, b), child.v,
                                     {next.data(), bits::trimmed_size(next.data(), W), d.num, NN}, fresh);
                if (fresh)
                {
                    nodes.push(id, s, b, succ.arity(s));
                    if constexpr (Ordered)
                        if (lo.limited() && store.size() - layer_end >= lo.limit())
                        {
                            truncated = true;
                            return false;
                        }
                }
                return true;
            },
            witness, canonical);
        if constexpr (Ordered)
            if (truncated || pos + 1 - layer_begin == order.size())
                pos = layer_end - 1;  // a full next layer drops the rest of this one (mimir); the beam's last entry ends it
    }
    log.close(r.expanded, r.generated, store.size());
    r.search_s = seconds_since(t0);
    r.exhausted = !r.solved && pos == store.size();
    r.states = store.size();
    r.words = store.words();
    r.store_bytes = store.bytes() + nodes.bytes();
    if (o.fingerprint)
    {
        std::vector<u64> w(store.words());
        for (u32 i = 0; i < store.size(); ++i)
        {
            store.decode(StateId{i}, w.data(), curnum.data());
            r.fingerprint ^= brfs_fingerprint_term(i, task.canonical_hash({w.data(), store.words(), NN ? curnum.data() : nullptr, NN}));
        }
    }
    ctl.finish(r);
    return r;
}

// ------------------------------------------------------------------------------------------------- beam layer step
/// A beam over the flat or chunked store with the layer step of beam_detail.hpp: threads > 1, or
/// LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly. Exact (threads > 1): the members generate every transition of a
/// batch of the layer's states and the calling thread stores them in the serial order, so ids, counts and plans are
/// the serial search's. Relaxed: the members score the successors, and only the parts' best ones are stored, in rank
/// order, until beam_width new states are stored (they are the next layer).
template<class Store>
BrfsResult run_beam(const Task& task, const BrfsOptions& o, u32 T, Control& ctl)
{
    constexpr bool flat = std::is_same_v<Store, FlatStateStore>;
    using search::detail::Candidates;
    using search::detail::Expansion;
    using search::detail::LayerOrderer;
    BrfsResult r;
    r.store = flat ? "flat" : "chunked";
    const bool witness = o.witness_pruning, canonical = o.canonical_order;
    const u32 NN = task.numeric_words();
    Store store = [&]
    {
        if constexpr (flat)
            return Store(std::max<u32>(1, task.words()), 16, NN);
        else
            return Store(std::max<u32>(1, task.words()), NN);
    }();
    Nodes nodes;
    const State& s0 = task.initial_state();
    store.insert(s0.view());
    nodes.root();
    ctl.start(s0.view());
    const auto t0 = Clock::now();
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr, nullptr};
    LayerOrderer lo(task, o.layers);
    search::detail::BeamTeam team(task, T);
    r.threads = team.size();
    Successors& succ = team.succ(0);  // the calling thread's
    const bool relaxed = lo.relaxed();
    const bool concurrent_goal = o.goal.kind != search::GoalSpec::Kind::Custom;
    const u32 members = team.size();

    // a stored state's words (store width, zero-padded) and numeric words
    auto decode = [&](u32 id, std::vector<u64>& w, std::vector<u64>& num) -> StateView
    {
        if constexpr (flat)
        {
            const u64* rec = store.words(StateId{id});
            w.assign(rec, rec + store.stride());
            num.assign(rec + store.stride(), rec + store.stride() + NN);
        }
        else
        {
            w.resize(store.words());
            num.resize(NN);
            store.decode(StateId{id}, w.data(), num.data());
        }
        return {w.data(), bits::trimmed_size(w.data(), static_cast<u32>(w.size())), NN ? num.data() : nullptr, NN};
    };

    std::vector<u32> order, keys;
    std::vector<Expansion> exps;
    std::vector<std::vector<u64>> mcur(members), mnum(members), mnext(members);
    std::vector<std::vector<u32>> mids(members);  // chunked: find_successor's scratch
    struct Ref
    {
        u32 member, cand, parent;
    };
    std::vector<LayerOrderer::Ranked> ranked;
    std::vector<Ref> refs;
    u64 layer_transitions = 0;
    std::vector<u64> cur, curnum, next;

    // phase 1, on member t: the expansion of order[i]
    auto expand_one = [&](u32 t, usize i)
    {
        Expansion& x = exps[i];
        x = Expansion{};
        x.member = t;
        Candidates& cs = team.candidates(t);
        x.first = cs.size();
        const u32 id = order[i];
        Successors& sc = team.succ(t);
        const StateView cv = decode(id, mcur[t], mnum[t]);
        sc.prepare(cv);
        if (concurrent_goal)
        {
            x.goal = is_goal(o.goal, sc, cv) ? 1 : 0;
            if (x.goal && o.stop_at_goal)
                return;
        }
        x.expanded = 1;
        std::vector<u64>& nx = mnext[t];
        if constexpr (!flat)
            mids[t].resize(store.chunks_per_state());
        sc.generate<true>(
            [&](u32 s, const ObjectId* b, const Delta& d) -> bool
            {
                const u32 seq = x.transitions++;
                const u32 nn = apply_delta(cv.w, cv.nw, d, nx);
                if (!relaxed)
                {
                    // stored before the batch (the store is only read meanwhile): its insert would find it
                    if constexpr (flat)
                    {
                        if (store.find(StateView{nx.data(), nn, d.num, NN}).valid())
                            return true;
                    }
                    else if (store.find_successor(StateId{id}, cv.w, nx.data(), nn, d, mids[t].data()).valid())
                        return true;
                }
                cs.push(seq, s, b, sc.arity(s), {}, flat ? nullptr : &d, nx.data(), nn, d.num, NN);
                return true;
            },
            witness, canonical);
        x.count = cs.size() - x.first;
        if (relaxed)
            for (u32 j = x.first; j < cs.size(); ++j)
                cs.set_value(j, lo.key(sc, StateView{cs.words(j), cs.nwords(j), cs.num(j, NN), NN}));
    };
    // phase 2, on this thread: the pop of order[i] (as the serial loop); false stops the search
    auto pop_one = [&](usize i) -> bool
    {
        const Expansion& x = exps[i];
        const u32 id = order[i];
        bool goal;
        if (concurrent_goal)
            goal = x.goal != 0;
        else
        {
            const StateView cv = decode(id, cur, curnum);
            succ.prepare(cv);
            goal = is_goal(o.goal, succ, cv);
        }
        r.goal_states += goal;
        if (goal && o.stop_at_goal)
        {
            r.solved = true;
            r.plan = nodes.plan(id, succ);
            return false;
        }
        ++r.expanded;
        return true;
    };
    // exact: the transitions of order[i] stored in their order
    auto merge_exact = [&](usize i)
    {
        const Expansion& x = exps[i];
        const u32 id = order[i];
        const Candidates& cs = team.candidates(x.member);
        if constexpr (flat)
        {
            for (u32 j = x.first; j < x.first + x.count; ++j)
                if (store.insert(cs.words(j), cs.nwords(j), cs.num(j, NN)).second)
                    nodes.push(id, cs.schema(j), cs.binding(j), succ.arity(cs.schema(j)));
        }
        else
        {
            // as run_chunked: only the chunks a delta touches are re-interned
            u32 W = store.words();
            cur.resize(W);
            next.resize(W);
            curnum.resize(NN);
            store.decode(StateId{id}, cur.data(), curnum.data());
            for (u32 j = x.first; j < x.first + x.count; ++j)
            {
                const Delta d = cs.delta(j, NN);
                u32 need = 0;
                for (SlotId a : d.add)
                    need = std::max<u32>(need, bits::word_of(a.v) + 1);
                if (need > W)
                {
                    store.widen(std::max(W * 2, need));
                    W = store.words();
                    cur.resize(W, 0);
                    next.resize(W, 0);
                }
                std::memcpy(next.data(), cur.data(), W * sizeof(u64));
                for (SlotId s : d.del)
                    if (bits::word_of(s.v) < W)
                        bits::reset(next.data(), s.v);
                for (SlotId s : d.add)
                    bits::set(next.data(), s.v);
                if (store.insert_successor(StateId{id}, cur.data(), next.data(), d).second)
                    nodes.push(id, cs.schema(j), cs.binding(j), succ.arity(cs.schema(j)));
            }
        }
        r.generated += x.transitions;
    };
    // relaxed: the transitions of order[i] join the layer's ranking
    auto collect_relaxed = [&](usize i)
    {
        const Expansion& x = exps[i];
        const Candidates& cs = team.candidates(x.member);
        for (u32 j = x.first; j < x.first + x.count; ++j)
        {
            ranked.push_back({cs.value(j), 0, layer_transitions + cs.seq(j), static_cast<u32>(refs.size())});
            refs.push_back({x.member, j, order[i]});
        }
        layer_transitions += x.transitions;
        r.generated += x.transitions;
    };
    // relaxed: the parts' best transitions in rank order, stored until beam_width new states are
    auto select_relaxed = [&]()
    {
        const SplitMix64 ties = lo.draw_ties(layer_transitions);
        if (lo.random_ties())
            for (LayerOrderer::Ranked& e : ranked)
                e.tie = ties.output_at(e.pos);
        search::detail::relaxed_rank(ranked, layer_transitions, members, lo.beam_chunk(), lo.beam_width());
        u32 kept = 0;
        for (const LayerOrderer::Ranked& e : ranked)
        {
            if (kept >= lo.beam_width())
                break;
            const Ref& f = refs[e.entry];
            const Candidates& cs = team.candidates(f.member);
            if (store.insert(cs.words(f.cand), cs.nwords(f.cand), cs.num(f.cand, NN)).second)
            {
                nodes.push(f.parent, cs.schema(f.cand), cs.binding(f.cand), succ.arity(cs.schema(f.cand)));
                ++kept;
            }
        }
        ranked.clear();
        refs.clear();
        layer_transitions = 0;
    };

    u32 layer_begin = 0, layer_end = 0;
    bool exhausted = false;
    for (;;)
    {
        // the serial loop's head at the first state of a layer
        if (layer_end >= store.size())
        {
            exhausted = true;
            break;
        }
        if (store.size() >= o.max_states || !ctl.keep_going(r, store.size()))
            break;
        log.close(r.expanded, r.generated, store.size());
        if (r.layers == o.max_depth)
            break;
        ++r.layers;
        layer_begin = layer_end;
        layer_end = store.size();
        log.open(r.expanded, r.generated, layer_end);
        order.resize(layer_end - layer_begin);
        std::iota(order.begin(), order.end(), layer_begin);
        if (layer_begin > 0 && !relaxed)
        {
            if (lo.scored())
            {
                keys.resize(order.size());
                team.for_each(order.size(), 64,
                              [&](u32 t, usize i)
                              { keys[i] = lo.key(team.succ(t), decode(order[i], mcur[t], mnum[t])); });
                lo.select_keyed(order, keys);
            }
            else
                lo.select(order, succ, [&](u32 e) { return decode(e, cur, curnum); });
        }
        const usize L = order.size();
        exps.resize(L);
        const usize B = relaxed ? L : std::max<usize>(64, usize{16} * members);
        bool stop = false;
        for (usize a = 0; a < L && !stop; a += B)
        {
            const usize bend = std::min(L, a + B);
            team.for_each(bend - a, 1, [&](u32 t, usize i) { expand_one(t, a + i); });
            for (usize i = a; i < bend; ++i)
            {
                if (i > 0 && (store.size() >= o.max_states || !ctl.keep_going(r, store.size())))
                {
                    stop = true;
                    break;
                }
                if (!pop_one(i))
                {
                    stop = true;
                    break;
                }
                if (relaxed)
                    collect_relaxed(i);
                else
                    merge_exact(i);
            }
        }
        if (stop)
            break;
        if (relaxed)
            select_relaxed();
    }
    log.close(r.expanded, r.generated, store.size());
    r.search_s = seconds_since(t0);
    r.exhausted = !r.solved && exhausted;
    r.states = store.size();
    if constexpr (flat)
        r.words = store.stride();
    else
        r.words = store.words();
    r.store_bytes = store.bytes() + nodes.bytes();
    if (o.fingerprint)
    {
        std::vector<u64> w, num;
        for (u32 i = 0; i < store.size(); ++i)
        {
            const StateView v = decode(i, w, num);
            r.fingerprint ^= brfs_fingerprint_term(i, task.canonical_hash({w.data(), static_cast<u32>(w.size()), v.num, NN}));
        }
    }
    ctl.finish(r);
    return r;
}

// ------------------------------------------------------------------------------------------------- compact
template<bool Observed>
BrfsResult run_compact(const Task& task, const BrfsOptions& o, Successors& succ, Control& ctl)
{
    BrfsResult r;
    r.store = "compact";
    const bool witness = o.witness_pruning, canonical = o.canonical_order;
    const AtomIndex& atoms = task.atoms();
    std::vector<Fingerprint128> keys;  // per fluent slot, from its canonical id
    auto ensure_keys = [&](u32 slots)
    {
        while (keys.size() < slots)
        {
            const u64 c = atoms.canonical(AtomKind::Fluent, static_cast<u32>(keys.size()));
            keys.push_back({hash::mix64(2 * c + 1 + 0x9E3779B97F4A7C15ULL), hash::mix64(2 * c + 2 + 0x9E3779B97F4A7C15ULL)});
        }
    };
    const State& s0 = task.initial_state();
    const u32 NN = task.numeric_words();
    // numeric tasks: the stored fingerprint also covers the numeric words (the layer keeps the atom part for updates)
    auto key_of = [&](Fingerprint128 f, const u64* num)
    {
        if (NN)
        {
            f.a ^= hash::words(num, NN, 0x3c6ef372fe94f82bULL);
            f.b ^= hash::words(num, NN, 0xa54ff53a5f1d36f1ULL);
        }
        return f;
    };
    u32 W = std::max<u32>(1, std::max(task.words(), s0.size_words()));
    std::vector<u64> layer, next_layer, cur(W, 0), next(W, 0), tog(W, 0);
    std::vector<u64> layer_num, next_layer_num, curnum(NN);
    std::vector<u32> changed;
    std::vector<Fingerprint128> layer_f, next_f;
    Nodes nodes;
    CompactStateSet closed;
    Fingerprint128 f0;
    ensure_keys(task.atoms().fluent_slots());
    bits::for_each(s0.data(), s0.size_words(), [&](u64 slot) { f0.toggle(keys[slot]); });
    layer.assign(W, 0);
    std::copy(s0.data(), s0.data() + s0.size_words(), layer.begin());
    layer_num.assign(s0.numeric().begin(), s0.numeric().end());
    layer_f.push_back(f0);
    closed.insert(key_of(f0, s0.numeric().data()));
    nodes.root();
    u32 layer_first = 0;
    search::SearchObserver* const obs = Observed ? ctl.observer() : nullptr;
    ctl.start(s0.view());
    const auto t0 = Clock::now();
    bool stopped = false;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr, obs};
    while (!layer_f.empty() && closed.size() < o.max_states && !stopped && r.layers < o.max_depth)
    {
        ++r.layers;
        log.open(r.expanded, r.generated, closed.size());
        next_layer.clear();
        next_layer_num.clear();
        next_f.clear();
        for (usize li = 0; li < layer_f.size() && !stopped; ++li)
        {
            if (!ctl.keep_going(r, closed.size()))
            {
                stopped = true;
                break;
            }
            std::memcpy(cur.data(), layer.data() + li * W, W * sizeof(u64));
            if (NN)
                std::memcpy(curnum.data(), layer_num.data() + li * NN, NN * sizeof(u64));
            const Fingerprint128 cf = layer_f[li];
            const u32 pid = layer_first + static_cast<u32>(li);
            const u32 n = bits::trimmed_size(cur.data(), W);
            const StateView sv{cur.data(), n, NN ? curnum.data() : nullptr, NN};
            succ.prepare(sv);
            const bool goal = is_goal(o.goal, succ, sv);
            r.goal_states += goal;
            if (goal && o.stop_at_goal)
            {
                r.solved = true;
                r.plan = nodes.plan(pid, succ);
                stopped = true;
                break;
            }
            ++r.expanded;
            if constexpr (Observed)
                obs->on_expand(pid, sv);
            succ.generate<false>(
                [&](u32 s, const ObjectId* b, const Delta& d)
                {
                    ++r.generated;
                    u32 need = 0;
                    MYMYR_NOVECTOR
                    for (SlotId a : d.add)
                        need = std::max<u32>(need, bits::word_of(a.v) + 1);
                    if (need > W)
                    {
                        const u32 nw = std::max(W * 2, need);
                        auto rewiden = [&](std::vector<u64>& v, usize rows)
                        {
                            std::vector<u64> nv(rows * nw, 0);
                            for (usize i = 0; i < rows; ++i)
                                std::memcpy(nv.data() + i * nw, v.data() + i * W, W * sizeof(u64));
                            v.swap(nv);
                        };
                        rewiden(layer, layer_f.size());
                        rewiden(next_layer, next_f.size());
                        W = nw;
                        cur.resize(W, 0);
                        next.resize(W, 0);
                        succ.engine().set_state(cur.data(), n);
                    }
                    ensure_keys(atoms.fluent_slots());
                    // delete first, then add; toggle exactly the atoms whose truth value changes
                    std::memcpy(next.data(), cur.data(), W * sizeof(u64));
                    for (SlotId x : d.del)
                        if (bits::word_of(x.v) < W)
                            bits::reset(next.data(), x.v);
                    for (SlotId x : d.add)
                        bits::set(next.data(), x.v);
                    Fingerprint128 f = cf;
                    if (tog.size() < W)
                        tog.resize(W, 0);
                    auto toggle = [&](u32 slot)
                    {
                        // each atom whose truth value changes, once (the lists may repeat slots)
                        if (bits::test(cur.data(), W, slot) != bits::test(next.data(), W, slot) && !bits::test(tog.data(), W, slot))
                        {
                            bits::set(tog.data(), slot);
                            f.toggle(keys[slot]);
                            changed.push_back(slot);
                        }
                    };
                    changed.clear();
                    for (SlotId x : d.del)
                        if (bits::word_of(x.v) < W)
                            toggle(x.v);
                    for (SlotId x : d.add)
                        toggle(x.v);
                    for (u32 slot : changed)
                        bits::reset(tog.data(), slot);
                    const bool fresh = closed.size() < o.max_states && closed.insert(key_of(f, d.num));
                    if constexpr (Observed)  // a duplicate's id is not kept (closed states are fingerprints)
                        obs->on_generate(pid, action_of(succ, s, b), fresh ? closed.size() - 1 : ~u64{0},
                                         {next.data(), bits::trimmed_size(next.data(), W), d.num, NN}, fresh);
                    if (!fresh)
                        return true;
                    next_layer.insert(next_layer.end(), next.begin(), next.end());
                    if (NN)
                        next_layer_num.insert(next_layer_num.end(), d.num, d.num + NN);
                    next_f.push_back(f);
                    nodes.push(pid, s, b, succ.arity(s));
                    return true;
                },
                witness, canonical);
        }
        log.close(r.expanded, r.generated, closed.size());
        layer_first += static_cast<u32>(layer_f.size());
        layer.swap(next_layer);
        layer_num.swap(next_layer_num);
        layer_f.swap(next_f);
    }
    r.search_s = seconds_since(t0);
    r.exhausted = !stopped && layer_f.empty();
    r.states = closed.size();
    r.words = W;
    r.store_bytes = closed.bytes() + nodes.bytes() +
                    (layer.capacity() + next_layer.capacity() + layer_num.capacity() + next_layer_num.capacity()) * sizeof(u64);
    if (o.fingerprint)
        r.fingerprint = 0;  // closed states are not kept: no per-id fingerprint
    ctl.finish(r);
    return r;
}

// ------------------------------------------------------------------------------------------------- parallel
struct alignas(64) ThreadState
{
    Successors* succ = nullptr;
    search::SearchObserver* obs = nullptr;  // the thread's observer (BrfsOptions::observer, make_worker)
    std::vector<u64> next;
    u64 generated = 0, goals = 0, expanded = 0;
    u64 next_progress = 0;
    u32 tick = 0;
};

class ParallelBrfs
{
public:
    /// workers: thread t's observer (empty without an observer; with one thread, the root observer)
    ParallelBrfs(const Task& task, const BrfsOptions& o, u32 threads, Control& ctl,
                 std::span<search::SearchObserver* const> workers)
        : m_task(task), m_o(o), m_ctl(ctl), m_T(threads), m_team(threads), m_store(threads, task.numeric_words()),
          m_ws(threads)
    {
        m_team.run([&](u32 t) { m_ws[t].succ = &task.workspace().successors(); });
        for (u32 t = 0; t < m_T && !workers.empty(); ++t)
        {
            m_ws[t].obs = workers[t];
            m_ws[t].next_progress = std::max<u64>(1, o.progress_interval);
        }
        m_partial.assign(m_T + 1, 0);
        m_pend_lo.assign(m_T, 0);
        m_pend_hi.assign(m_T, 0);
    }

    BrfsResult run()
    {
        BrfsResult r;
        r.store = "concurrent";
        r.threads = m_T;
        const auto t0 = Clock::now();
        const State& s0 = m_task.initial_state();
        m_loc.push_back(m_store.insert(0, s0.data(), s0.size_words(), ~u64{0}, s0.numeric().data()).first);
        m_ctl.start(s0.view());
        m_store.begin_layer();
        m_layer_lo = 0;
        m_layer_hi = 1;
        auto totals = [&]
        {
            std::array<u64, 2> x{0, 0};
            for (const ThreadState& w : m_ws)
                x[0] += w.expanded, x[1] += w.generated;
            return x;
        };
        search::SearchObserver* const obs = m_ctl.observer();
        const bool log_layers = m_o.layer_stats || obs;
        while (m_layer_lo < m_layer_hi && m_layer_hi < m_o.max_states && !m_stop.load(std::memory_order_relaxed) &&
               !m_halt.load(std::memory_order_relaxed) && r.layers < m_o.max_depth && m_ctl.check())
        {
            ++r.layers;
            const std::array<u64, 2> before = log_layers ? totals() : std::array<u64, 2>{0, 0};
            const auto layer_t0 = Clock::now();
            const u64 stored = m_layer_hi;
            const u64 L = m_layer_hi - m_layer_lo;
            m_chunk = std::clamp<u64>(L / (static_cast<u64>(m_T) * 32), 1, 256);
            m_chunk_ctr.store(0, std::memory_order_relaxed);
            std::fill(m_pend_lo.begin(), m_pend_lo.end(), 0);
            std::fill(m_pend_hi.begin(), m_pend_hi.end(), 0);
            for (;;)
            {
                m_team.run([&](u32 t) { work(t); });
                if (!m_resize.load(std::memory_order_relaxed))
                    break;
                m_store.rehash(m_team);
                m_resize.store(false, std::memory_order_relaxed);
            }
            if (m_stop.load(std::memory_order_relaxed) || m_halt.load(std::memory_order_relaxed))
            {
                if (obs)  // the partly expanded last layer (its new states get no ids)
                {
                    const std::array<u64, 2> after = totals();
                    obs->on_pass(r.layers - 1, {.expanded = after[0] - before[0], .generated = after[1] - before[1],
                                                .seconds = seconds_since(layer_t0)});
                }
                break;
            }
            finalize_layer();
            if (log_layers)
            {
                const std::array<u64, 2> after = totals();
                if (m_o.layer_stats)
                    r.layer_counts.push_back({after[0] - before[0], after[1] - before[1], m_layer_hi - stored});
                if (obs)
                    obs->on_pass(r.layers - 1, {.expanded = after[0] - before[0], .generated = after[1] - before[1],
                                                .states = m_layer_hi - stored, .seconds = seconds_since(layer_t0)});
            }
        }
        if (const u8 why = m_halt.load(std::memory_order_relaxed); why != 0)
            m_ctl.stop(static_cast<search::SearchStatus>(why - 1));
        r.search_s = seconds_since(t0);
        r.solved = m_stop.load(std::memory_order_relaxed);
        if (r.solved)
            r.plan = plan_to(m_goal.load(std::memory_order_relaxed));
        r.exhausted = !r.solved && m_layer_lo == m_layer_hi;
        r.states = m_layer_hi;
        for (const ThreadState& w : m_ws)
        {
            r.generated += w.generated;
            r.goal_states += w.goals;
            r.expanded += w.expanded;
        }
        r.store_bytes = m_store.bytes() + m_loc.capacity() * sizeof(u32);
        if (m_o.fingerprint)
        {
            std::vector<u64> fps(m_T, 0);
            const u64 N = m_layer_hi;
            m_team.run(
                [&](u32 t)
                {
                    auto [a, b] = Team::slice(N, t, m_T);
                    u64 x = 0;
                    for (u64 id = a; id < b; ++id)
                        x ^= brfs_fingerprint_term(id, m_task.canonical_hash(m_store.record(m_loc[id])));
                    fps[t] = x;
                });
            for (u64 x : fps)
                r.fingerprint ^= x;
        }
        u32 maxw = 0;
        for (u64 id = m_layer_hi > 64 ? m_layer_hi - 64 : 0; id < m_layer_hi; ++id)
            maxw = std::max(maxw, m_store.record(m_loc[id]).nw);
        r.words = maxw;
        m_ctl.finish(r);
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
                if (m_resize.load(std::memory_order_relaxed) || m_stop.load(std::memory_order_relaxed) ||
                    m_halt.load(std::memory_order_relaxed))
                    break;
                const u64 c = m_chunk_ctr.fetch_add(1, std::memory_order_relaxed);
                lo = m_layer_lo + c * m_chunk;
                if (lo >= m_layer_hi)
                    break;
                hi = std::min(m_layer_hi, lo + m_chunk);
            }
            if (!keep_going(t))
                break;
            expand(t, lo++);
            if (m_store.wants_rehash())
                m_resize.store(true, std::memory_order_relaxed);
            if (m_resize.load(std::memory_order_relaxed))
                break;  // stop after the current state; the rest of [lo, hi) resumes after the rehash
        }
        m_pend_lo[t] = lo;
        m_pend_hi[t] = hi;
    }

    /// The plan to state `goal`, on the calling thread: every state's discoverer key names its parent and the index of
    /// the transition among the parent's successors, which are generated again in the same order.
    std::vector<Action> plan_to(u64 goal)
    {
        Successors& succ = *m_ws[0].succ;  // member 0 is the calling thread
        std::vector<Action> plan;
        for (u64 v = goal; v != 0;)
        {
            const u64 key = m_store.key(m_loc[v]);
            const u64 parent = key >> 24;
            const u32 k = static_cast<u32>(key & 0xFFFFFFULL);
            succ.prepare(m_store.record(m_loc[parent]));
            u32 i = 0;
            succ.generate<false>(
                [&](u32 s, const ObjectId* b, const Delta&)
                {
                    if (i++ != k)
                        return true;
                    plan.push_back(action_of(succ, s, b));
                    return false;
                },
                m_o.witness_pruning, m_o.canonical_order);
            v = parent;
        }
        std::reverse(plan.begin(), plan.end());
        return plan;
    }

    /// Thread t, before an expansion: the time, the token and its observer's on_progress (checked every few
    /// expansions); false once one of them, or another thread, stopped the search.
    bool keep_going(u32 t)
    {
        ThreadState& w = m_ws[t];
        if (m_halt.load(std::memory_order_relaxed))
            return false;
        if ((w.tick++ & 7) == 0)
        {
            if (m_ctl.timed() && Clock::now() >= m_ctl.deadline())
                return halt(search::SearchStatus::OutOfTime);
            if (m_o.cancel.requested())
                return halt(search::SearchStatus::Cancelled);
        }
        if (w.obs && w.expanded >= w.next_progress)
        {
            w.next_progress = w.expanded + std::max<u64>(1, m_o.progress_interval);
            if (!w.obs->on_progress(m_ctl.statistics(w.expanded, w.generated, 0)))
                return halt(search::SearchStatus::Cancelled);
        }
        return true;
    }

    bool halt(search::SearchStatus why)
    {
        u8 none = 0;
        m_halt.compare_exchange_strong(none, static_cast<u8>(static_cast<u8>(why) + 1), std::memory_order_relaxed);
        return false;
    }

    void expand(u32 t, u64 id)
    {
        ThreadState& w = m_ws[t];
        Successors& succ = *w.succ;
        const StateView rec = m_store.record(m_loc[id]);  // records never move
        succ.prepare(rec);
        const bool goal = is_goal(m_o.goal, succ, rec);
        w.goals += goal;
        if (goal && m_o.stop_at_goal)
        {
            u64 g = m_goal.load(std::memory_order_relaxed);
            while (id < g && !m_goal.compare_exchange_weak(g, id, std::memory_order_relaxed))
            {
            }
            m_stop.store(true, std::memory_order_relaxed);
            return;
        }
        ++w.expanded;
        if (w.obs)
            w.obs->on_expand(id, rec);
        u32 k = 0;
        succ.generate<false>(
            [&](u32 s, const ObjectId* b, const Delta& d)
            {
                const u32 kk = k++;
                ++w.generated;
                const u32 nn = apply_delta(rec.w, rec.nw, d, w.next);
                const bool fresh = m_store.insert(t, w.next.data(), nn, (id << 24) | kk, d.num).second;
                if (w.obs)  // ids are assigned when the layer ends
                    w.obs->on_generate(id, action_of(succ, s, b), ~u64{0},
                                       {w.next.data(), nn, d.num, m_task.numeric_words()}, fresh);
                return true;
            },
            m_o.witness_pruning, m_o.canonical_order);
        if (k >= (u32{1} << 24))
            throw std::length_error("mymyr brfs: more than 2^24 successors of one state");
    }

    void finalize_layer()
    {
        u64 total = 0;
        const u32 T = m_T;
        if (!m_o.deterministic_ids)
        {
            // per-thread ranges: ids in (thread, local index) order; independent of nothing
            for (u32 t = 0; t < T; ++t)
            {
                m_partial[t] = total;
                total += m_store.local_size(t) - m_store.layer_start(t);
            }
            m_loc.resize(m_layer_hi + total);
            m_team.run(
                [&](u32 t)
                {
                    u64 base = m_layer_hi + m_partial[t];
                    for (u32 li = m_store.layer_start(t); li < m_store.local_size(t); ++li)
                        m_loc[base++] = m_store.make_handle(t, li);
                });
        }
        else
        {
            // deterministic: counting sort by discoverer (parent id), then by successor index within a parent
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
                    for (u32 li = m_store.layer_start(t); li < m_store.local_size(t); ++li)
                        m_cursor[(m_store.key(m_store.make_handle(t, li)) >> 24) - m_layer_lo].fetch_add(1, std::memory_order_relaxed);
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
                    for (u32 li = m_store.layer_start(t); li < m_store.local_size(t); ++li)
                    {
                        const ConcurrentStateStore::Handle h = m_store.make_handle(t, li);
                        const u64 key = m_store.key(h);
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
                        for (u64* x = lo + 1; x < hi; ++x)  // insertion sort: segments are tiny
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
        }
        m_store.begin_layer();
        m_layer_lo = m_layer_hi;
        m_layer_hi += total;
        if (m_layer_hi >= (u64{1} << 40))
            throw std::length_error("mymyr brfs: more than 2^40 states");
    }

    const Task& m_task;
    const BrfsOptions& m_o;
    Control& m_ctl;
    u32 m_T;
    Team m_team;
    ConcurrentStateStore m_store;
    std::vector<ThreadState> m_ws;
    std::vector<u32> m_loc;  // id -> handle
    u64 m_layer_lo = 0, m_layer_hi = 0;
    std::atomic<u64> m_chunk_ctr{0};
    u64 m_chunk = 1;
    std::atomic<bool> m_resize{false};
    std::atomic<bool> m_stop{false};
    std::atomic<u64> m_goal{~u64{0}};  // stop_at_goal: the smallest id of a goal state expanded
    std::atomic<u8> m_halt{0};  // 0, or 1 + the SearchStatus that stopped the search (time, token, on_progress)
    std::vector<u64> m_pend_lo, m_pend_hi, m_partial;
    std::unique_ptr<std::atomic<u32>[]> m_cursor;
    u64 m_cursor_cap = 0;
    std::vector<u32> m_offv;
    std::vector<u64> m_tmp;
};
}  // namespace

BrfsResult brfs(const Task& task, const BrfsOptions& options)
{
    u32 T = search::detail::resolve_threads(options.threads);
    // a beam runs on the flat or chunked store, with the layer step for threads > 1 or a relaxed selection
    const bool ordered = options.layers.kind != search::LayerOrdering::Kind::Queue;
    const bool beam = ordered && options.layers.beam();
    const bool relaxed = beam && options.layers.beam_novelty == search::LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly;
    if (beam && options.observer)
    {
        if (relaxed)
            throw std::invalid_argument("mymyr brfs: LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly cannot be combined with an observer");
        T = 1;  // an observer runs the beam on the calling thread
    }
    if (ordered && !beam && T > 1)
        throw std::invalid_argument("mymyr brfs: ordered layers without a beam are single-threaded (threads == 1)");
    BrfsOptions::Store store = options.store;
    if (store == BrfsOptions::Store::Auto)
    {
        if (T > 1 && !ordered)
            store = BrfsOptions::Store::Concurrent;
        else
        {
            // Flat for W <= 8 words, Chunked above (the width estimate: frozen W, else the pilot's W_lazy;
            // numeric tasks count the whole row [bits | slots])
            const u32 w = task.atoms().mode() == AtomMode::Frozen ? task.words() : std::max(task.words(), task.info().pilot_words);
            store = w + task.numeric_words() <= 8 ? BrfsOptions::Store::Flat : BrfsOptions::Store::Chunked;
        }
    }
    if (store != BrfsOptions::Store::Concurrent && T > 1 && !beam)
        throw std::invalid_argument("mymyr brfs: the flat, chunked and compact stores are single-threaded (but for a beam)");
    if (options.goal.kind == search::GoalSpec::Kind::Custom && (T > 1 || !options.goal.test))
        throw std::invalid_argument("mymyr brfs: a custom goal test needs threads == 1 and a test function");
    if (std::string e = search::detail::check_layers(options.layers); !e.empty())
        throw std::invalid_argument("mymyr brfs: " + e);
    if (ordered && store != BrfsOptions::Store::Flat && store != BrfsOptions::Store::Chunked)
        throw std::invalid_argument("mymyr brfs: ordered layers need the flat or chunked store");
    BrfsResult r;
    Control ctl(options, options.observer);
    if (beam && (T > 1 || relaxed))
        r = store == BrfsOptions::Store::Flat ? run_beam<FlatStateStore>(task, options, T, ctl)
                                              : run_beam<ChunkedStateStore>(task, options, T, ctl);
    else if (store == BrfsOptions::Store::Concurrent)
    {
        // the make_worker protocol (search/control.hpp): one observer per thread, else one thread
        std::vector<std::shared_ptr<search::SearchObserver>> owned;
        std::vector<search::SearchObserver*> workers;
        if (options.observer && T > 1)
        {
            for (u32 t = 0; t < T; ++t)
            {
                owned.push_back(options.observer->make_worker(t));
                if (!owned.back())
                {
                    T = 1;
                    break;
                }
                workers.push_back(owned.back().get());
            }
        }
        if (options.observer && T == 1)
            workers.assign(1, options.observer);
        ParallelBrfs search(task, options, T, ctl, workers);
        r = search.run();
    }
    else
    {
        Successors& succ = task.workspace().successors();
        switch (store)
        {
            case BrfsOptions::Store::Flat:
                r = ordered ? (options.observer ? run_flat<true, true>(task, options, succ, ctl)
                                                : run_flat<true, false>(task, options, succ, ctl))
                            : (options.observer ? run_flat<false, true>(task, options, succ, ctl)
                                                : run_flat<false, false>(task, options, succ, ctl));
                break;
            case BrfsOptions::Store::Chunked:
                r = ordered ? (options.observer ? run_chunked<true, true>(task, options, succ, ctl)
                                                : run_chunked<true, false>(task, options, succ, ctl))
                            : (options.observer ? run_chunked<false, true>(task, options, succ, ctl)
                                                : run_chunked<false, false>(task, options, succ, ctl));
                break;
            default:
                r = options.observer ? run_compact<true>(task, options, succ, ctl)
                                     : run_compact<false>(task, options, succ, ctl);
                break;
        }
    }
    r.fluent_slots = task.atoms().fluent_slots();
    return r;
}
}  // namespace mymyr
