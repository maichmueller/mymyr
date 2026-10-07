// Breadth-first search. The single-threaded loop runs over a flat, chunked or compact store; the multi-threaded one
// is a layer-synchronous search with a choice of successor id scheme (dedup "cas", deterministic "det" or per-thread
// "ranges" ids).

#include "mymyr/search/brfs.hpp"

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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>

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

/// Per-layer counts (BrfsOptions::layer_stats): open() at the start of a layer, close() at its end.
struct LayerLog
{
    std::vector<std::array<u64, 3>>* out = nullptr;
    u64 e0 = 0, g0 = 0, end = 0;
    bool open_ = false;

    void open(u64 expanded, u64 generated, u64 stored)
    {
        e0 = expanded;
        g0 = generated;
        end = stored;
        open_ = true;
    }
    void close(u64 expanded, u64 generated, u64 stored)
    {
        if (out && open_)
            out->push_back({expanded - e0, generated - g0, stored - end});
        open_ = false;
    }
};

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
template<bool Ordered>
BrfsResult run_flat(const Task& task, const BrfsOptions& o, Successors& succ)
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
    const auto t0 = Clock::now();
    u32 layer_end = 0;
    u32 pos = 0;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr};
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
        succ.generate<Ordered>(
            [&](u32 s, const ObjectId* b, const Delta& d)
            {
                ++r.generated;
                const u32 nn = apply_delta(cur.data(), n, d, next);
                if (store.insert(next.data(), nn, d.num).second)
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
    return r;
}

// ------------------------------------------------------------------------------------------------- chunked
template<bool Ordered>
BrfsResult run_chunked(const Task& task, const BrfsOptions& o, Successors& succ)
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
    const auto t0 = Clock::now();
    u32 layer_end = 0;
    u32 pos = 0;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr};
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
                if (store.insert_successor(StateId{id}, cur.data(), next.data(), d).second)
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
    return r;
}

// ------------------------------------------------------------------------------------------------- compact
BrfsResult run_compact(const Task& task, const BrfsOptions& o, Successors& succ)
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
    const auto t0 = Clock::now();
    bool stopped = false;
    LayerLog log{o.layer_stats ? &r.layer_counts : nullptr};
    while (!layer_f.empty() && closed.size() < o.max_states && !stopped && r.layers < o.max_depth)
    {
        ++r.layers;
        log.open(r.expanded, r.generated, closed.size());
        next_layer.clear();
        next_layer_num.clear();
        next_f.clear();
        for (usize li = 0; li < layer_f.size() && !stopped; ++li)
        {
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
                    if (closed.size() >= o.max_states || !closed.insert(key_of(f, d.num)))
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
    return r;
}

// ------------------------------------------------------------------------------------------------- parallel
struct alignas(64) ThreadState
{
    Successors* succ = nullptr;
    std::vector<u64> next;
    u64 generated = 0, goals = 0, expanded = 0;
};

class ParallelBrfs
{
public:
    ParallelBrfs(const Task& task, const BrfsOptions& o, u32 threads)
        : m_task(task), m_o(o), m_T(threads), m_team(threads), m_store(threads, task.numeric_words()), m_ws(threads)
    {
        m_team.run([&](u32 t) { m_ws[t].succ = &task.workspace().successors(); });
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
        while (m_layer_lo < m_layer_hi && m_layer_hi < m_o.max_states && !m_stop.load(std::memory_order_relaxed) &&
               r.layers < m_o.max_depth)
        {
            ++r.layers;
            const std::array<u64, 2> before = m_o.layer_stats ? totals() : std::array<u64, 2>{0, 0};
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
            if (m_stop.load(std::memory_order_relaxed))
                break;
            finalize_layer();
            if (m_o.layer_stats)
            {
                const std::array<u64, 2> after = totals();
                r.layer_counts.push_back({after[0] - before[0], after[1] - before[1], m_layer_hi - stored});
            }
        }
        r.search_s = seconds_since(t0);
        r.solved = m_stop.load(std::memory_order_relaxed);
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
                const u64 c = m_chunk_ctr.fetch_add(1, std::memory_order_relaxed);
                lo = m_layer_lo + c * m_chunk;
                if (lo >= m_layer_hi)
                    break;
                hi = std::min(m_layer_hi, lo + m_chunk);
            }
            expand(t, lo++);
            if (m_store.wants_rehash())
                m_resize.store(true, std::memory_order_relaxed);
            if (m_resize.load(std::memory_order_relaxed))
                break;  // stop after the current state; the rest of [lo, hi) resumes after the rehash
        }
        m_pend_lo[t] = lo;
        m_pend_hi[t] = hi;
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
            m_stop.store(true, std::memory_order_relaxed);
            return;
        }
        ++w.expanded;
        u32 k = 0;
        succ.generate<false>(
            [&](u32, const ObjectId*, const Delta& d)
            {
                const u32 kk = k++;
                ++w.generated;
                const u32 nn = apply_delta(rec.w, rec.nw, d, w.next);
                m_store.insert(t, w.next.data(), nn, (id << 24) | kk, d.num);
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
    std::vector<u64> m_pend_lo, m_pend_hi, m_partial;
    std::unique_ptr<std::atomic<u32>[]> m_cursor;
    u64 m_cursor_cap = 0;
    std::vector<u32> m_offv;
    std::vector<u64> m_tmp;
};
}  // namespace

BrfsResult brfs(const Task& task, const BrfsOptions& options)
{
    u32 T = options.threads == 0 ? std::max<u32>(1, std::thread::hardware_concurrency()) : options.threads;
    BrfsOptions::Store store = options.store;
    if (store == BrfsOptions::Store::Auto)
    {
        if (T > 1)
            store = BrfsOptions::Store::Concurrent;
        else
        {
            // Flat for W <= 8 words, Chunked above (the width estimate: frozen W, else the pilot's W_lazy;
            // numeric tasks count the whole row [bits | slots])
            const u32 w = task.atoms().mode() == AtomMode::Frozen ? task.words() : std::max(task.words(), task.info().pilot_words);
            store = w + task.numeric_words() <= 8 ? BrfsOptions::Store::Flat : BrfsOptions::Store::Chunked;
        }
    }
    if (store != BrfsOptions::Store::Concurrent && T > 1)
        throw std::invalid_argument("mymyr brfs: the flat, chunked and compact stores are single-threaded");
    if (options.goal.kind == search::GoalSpec::Kind::Custom && (T > 1 || !options.goal.test))
        throw std::invalid_argument("mymyr brfs: a custom goal test needs threads == 1 and a test function");
    if (std::string e = search::detail::check_layers(options.layers); !e.empty())
        throw std::invalid_argument("mymyr brfs: " + e);
    const bool ordered = options.layers.kind != search::LayerOrdering::Kind::Queue;
    if (ordered && store != BrfsOptions::Store::Flat && store != BrfsOptions::Store::Chunked)
        throw std::invalid_argument("mymyr brfs: ordered layers need the flat or chunked store (single-threaded)");
    BrfsResult r;
    if (store == BrfsOptions::Store::Concurrent)
    {
        ParallelBrfs search(task, options, T);
        r = search.run();
    }
    else
    {
        Successors& succ = task.workspace().successors();
        switch (store)
        {
            case BrfsOptions::Store::Flat:
                r = ordered ? run_flat<true>(task, options, succ) : run_flat<false>(task, options, succ);
                break;
            case BrfsOptions::Store::Chunked:
                r = ordered ? run_chunked<true>(task, options, succ) : run_chunked<false>(task, options, succ);
                break;
            default: r = run_compact(task, options, succ); break;
        }
    }
    r.fluent_slots = task.atoms().fluent_slots();
    return r;
}
}  // namespace mymyr
