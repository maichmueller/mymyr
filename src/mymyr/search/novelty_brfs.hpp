#pragma once
// The breadth-first engine of the IW family variants: one BrFS pass with a pluggable novelty pruner, mimir's
// root rules, optional ordered (and randomized, truncated) layers, search coordination and state tracking. A port of
// mimir 0.16.3 brfs::find_solution (src/search/algorithms/brfs.cpp, the queued path, and brfs/ordered_layer.cpp) with
// the conventions of the IW pass (search/iw.cpp, run_pass): goal test on pop, blocked states before the novelty
// test, expanded = popped non-goal states, generated = every transition of an expanded state, generated_in_tree =
// admitted successors.
//
// A pruner decides novelty. Its interface (duck-typed):
//   bool init(const u64* w, u32 n);                        marks the start state; false: it is pruned (mimir's
//                                                          test_prune_initial_state; the pass then ends at once)
//   void begin(const u64* w, u32 n);                       once per expanded state, before its transitions
//   bool test(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d);
//                                                          novelty of parent -> child, marking what is new
//   u64 bytes() const;
// and optionally
//   bool quick_reject(std::span<const u32> add, const Delta& d);  true: certainly not novel (nothing is marked), so
//                                                          the successor need not be materialized
//   bool peek(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta& d);
//                                                          test() without marking (a beam with
//                                                          BeamNovelty::SurvivorsOnly needs it)
// `add` holds the atoms the transition adds (true in the child, false in the parent), each once, in effect order.
//
// Root rules (mimir's pruning strategies): Normal (the novelty test applies at the root too); ArityZero (the
// width-0 pass: every successor of the start state other than itself enters, duplicates once per action as queue
// entries, and every successor below the root is pruned); Continuation (optimized IW(1) and abstracted IW: every
// distinct root successor enters; one that is not novel is goal-tested and counted as expanded but not expanded,
// unless keep_depth_one is set).
//
// Every state admitted to the tree is never novel again for the pruners of this family (all of its features are
// marked when it enters), so, as in the IW pass, no duplicate-detection table is needed: mimir's "is new" test is
// implied. Width-0 duplicates are queue entries that point to the node of their first occurrence; popping a closed
// node is skipped, as mimir skips CLOSED search nodes.
//
// Beam (LayerOrdering::beam_width): candidates enter the tree and the next layer as they are generated; at the
// boundary the layer is ordered and cut, and generated_in_tree counts only the kept entries. With
// BeamNovelty::SurvivorsOnly the candidates are tested with peek() (nothing is marked, so a set of the candidates
// stands in for mimir's "is new" test) and the kept ones are replayed in rank order through test(): one that adds
// nothing beyond the better ranked ones leaves the layer (a Continuation root successor stays, marked as not to be
// expanded unless keep_depth_one). Observers see the transitions as they are generated: a successor reported as
// Opened can be dropped by the beam or the replay and is then never expanded.

#include "iw_detail.hpp"
#include "layer_order_detail.hpp"

#include "mymyr/core/random.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mymyr::search::detail
{
inline constexpr u32 k_no_node = ~u32{0};

using SuccessorOrder = std::function<void(StateView, std::span<const Action>, std::vector<u32>&)>;

/// Records every distinct state a search creates: mimir's state repository with first-achiever and
/// co-occurrence tracking (state_repository.cpp record_first_achievers / record_co_occurrence). Indexed by fluent
/// slots; convert with the task's atom index before comparing across tasks or runs.
class StateTracker
{
public:
    StateTracker(const Task& task, bool first_achievers, bool co_occurrence)
        : m_store(std::max<u32>(task.words(), 1), 10), m_first(first_achievers), m_co(co_occurrence), m_derived(task.has_axioms())
    {
    }

    /// Records a created state; true if it is new.
    bool record(const u64* w, u32 n)
    {
        const auto [id, fresh] = m_store.insert(w, n);
        if (!fresh)
            return false;
        n = bits::trimmed_size(w, n);
        if (m_reached.size() < n)
            m_reached.resize(n, 0);
        bits::for_each(w, n,
                       [&](u64 a)
                       {
                           if (!bits::test(m_reached.data(), static_cast<u32>(m_reached.size()), a))
                           {
                               bits::set(m_reached.data(), a);
                               if (m_first)
                               {
                                   if (m_first_achiever.size() <= a)
                                       m_first_achiever.resize(a + 1, k_no_node);
                                   m_first_achiever[a] = id.v;
                               }
                           }
                           if (m_co)
                           {
                               if (m_rows.size() <= a)
                                   m_rows.resize(a + 1);
                               std::vector<u64>& row = m_rows[a];
                               if (row.size() < n)
                                   row.resize(n, 0);
                               for (u32 i = 0; i < n; ++i)
                                   row[i] |= w[i];
                           }
                       });
        if (m_derived)
            m_pending.push_back(id.v);
        return true;
    }

    /// record(w, n) for a successor of a recorded state, where `add` holds the atoms true in w and false in that
    /// parent (each once). The same result, cheaper: the parent's atoms are reached already and their rows hold the
    /// parent's atoms, so a kept atom's row only gains the added atoms, and only an added atom can be newly reached
    /// or gain a whole row.
    bool record_successor(const u64* w, u32 n, std::span<const u32> add)
    {
        const auto [id, fresh] = m_store.insert(w, n);
        if (!fresh)
            return false;
        n = bits::trimmed_size(w, n);
        if (m_reached.size() < n)
            m_reached.resize(n, 0);
        for (u32 a : add)
            if (!bits::test(m_reached.data(), static_cast<u32>(m_reached.size()), a))
            {
                bits::set(m_reached.data(), a);
                if (m_first)
                {
                    if (m_first_achiever.size() <= a)
                        m_first_achiever.resize(a + 1, k_no_node);
                    m_first_achiever[a] = id.v;
                }
            }
        if (m_co && !add.empty())
        {
            for (u32 a : add)
            {
                std::vector<u64>& row = co_row(a, n);
                for (u32 i = 0; i < n; ++i)
                    row[i] |= w[i];
            }
            bits::for_each(w, n,
                           [&](u64 a)
                           {
                               std::vector<u64>& row = co_row(static_cast<u32>(a), n);
                               for (u32 x : add)
                                   bits::set(row.data(), x);
                           });
        }
        if (m_derived)
            m_pending.push_back(id.v);
        return true;
    }

    /// Evaluates the derived atoms of the states recorded since the last call (mimir evaluates the axioms of every
    /// new state). Must not run inside a successor enumeration of `succ`.
    void flush_derived(Successors& succ)
    {
        for (u32 id : m_pending)
        {
            succ.prepare(m_store[StateId{id}]);
            const mymyr::detail::Engine& e = succ.engine();
            const u32 dn = e.derived_words();
            if (m_reached_derived.size() < dn)
                m_reached_derived.resize(dn, 0);
            for (u32 i = 0; i < dn; ++i)
                m_reached_derived[i] |= e.derived()[i];
        }
        m_pending.clear();
    }
    [[nodiscard]] bool pending() const noexcept { return !m_pending.empty(); }

    [[nodiscard]] u32 num_states() const noexcept { return m_store.size(); }
    [[nodiscard]] const FlatStateStore& store() const noexcept { return m_store; }
    [[nodiscard]] std::span<const u64> reached() const noexcept { return m_reached; }
    [[nodiscard]] std::span<const u64> reached_derived() const noexcept { return m_reached_derived; }
    /// Per fluent slot: the id (in store()) of the first created state holding it, or k_no_node.
    [[nodiscard]] std::span<const u32> first_achievers() const noexcept { return m_first_achiever; }
    /// Per fluent slot: the union of the created states holding it.
    [[nodiscard]] const std::vector<std::vector<u64>>& co_occurrence() const noexcept { return m_rows; }
    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 b = m_store.bytes() + m_reached.capacity() * 8 + m_first_achiever.capacity() * 4;
        for (const auto& r : m_rows)
            b += r.capacity() * 8;
        return b;
    }

private:
    /// The co-occurrence row of atom a, at least n words long.
    std::vector<u64>& co_row(u32 a, u32 n)
    {
        if (m_rows.size() <= a)
            m_rows.resize(a + 1);
        std::vector<u64>& row = m_rows[a];
        if (row.size() < n)
            row.resize(n, 0);
        return row;
    }

    FlatStateStore m_store;
    bool m_first, m_co, m_derived;
    std::vector<u64> m_reached, m_reached_derived;
    std::vector<u32> m_first_achiever;
    std::vector<std::vector<u64>> m_rows;
    std::vector<u32> m_pending;
};

/// Everything the passes of one search share.
struct Env
{
    const Task& task;
    Successors& succ;
    const GoalTest& goal;
    const BlockedSet& blocked;
    const SearchControl& control;
    bool witness = false;
    bool canonical = true;
    SymmetryPruning symmetry = SymmetryPruning::Off;
    const SuccessorOrder* successor_order = nullptr;  // null or empty: generation order
    Clock::time_point deadline{};
    bool timed = false;
    SearchObserver* obs = nullptr;   // hot events (expand, generate, prune, transition, progress): a worker observer
    SearchObserver* root = nullptr;  // lifecycle events (pass, solution, end); null inside parallel workers
    SearchCoordination* coord = nullptr;
    StateTracker* tracker = nullptr;
    LayerOrderer* layers = nullptr;  // ordered layers (mimir's layer_ordering_strategy); null: the queued BrFS

    Env(const Task& t, Successors& s, const GoalTest& g, const BlockedSet& b, const SearchControl& c)
        : task(t), succ(s), goal(g), blocked(b), control(c), obs(c.observer), root(c.observer), coord(c.coordination)
    {
        const double secs = c.budget.max_seconds;
        timed = secs < 1e15;
        if (timed)
            deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)));
    }
    [[nodiscard]] bool out_of_time() const { return timed && Clock::now() >= deadline; }
    [[nodiscard]] bool cancelled() const { return control.cancel.requested() || (coord && coord->is_cancelled()); }
    /// The slow path materializes and reports every transition.
    [[nodiscard]] bool slow() const { return obs || !blocked.empty() || tracker || (successor_order && *successor_order); }
};

enum class RootRule : u8
{
    Normal,
    ArityZero,
    Continuation,
};

struct PassConfig
{
    u32 arity = 0;                  // reported in the pass statistics
    RootRule root = RootRule::Normal;
    bool root_only = false;         // ArityZero: depth-1 states are goal-tested, not expanded (WidthZero::RootOnly)
    bool keep_depth_one = false;    // Continuation: non-novel root successors are expanded too
};

struct PassOut
{
    IwPassStatistics st;
    std::vector<Action> plan;
    std::optional<State> goal_state;
    u32 goal_node = k_no_node;
    u64 table_bytes = 0;
    u64 node_bytes = 0;
};

/// The pruner of the width-0 pass (and a placeholder type): never novel.
struct NullPruner
{
    bool init(const u64*, u32) { return true; }
    void begin(const u64*, u32) {}
    bool test(const u64*, u32, const u64*, u32, std::span<const u32>, const Delta&) { return false; }
    bool peek(const u64*, u32, const u64*, u32, std::span<const u32>, const Delta&) { return false; }
    [[nodiscard]] u64 bytes() const { return 0; }
};

/// The states of one pass: fixed-stride words (re-laid out when lazy slots widen the states) and SoA plan records
/// (the IW pass's tree plus a closed flag per node).
class NodeTree
{
public:
    explicit NodeTree(u32 words) : m_W(std::max<u32>(1, words)) {}

    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_parent.size()); }
    [[nodiscard]] u32 stride() const noexcept { return m_W; }
    [[nodiscard]] const u64* words(u32 id) const noexcept { return m_words.data() + static_cast<usize>(id) * m_W; }
    [[nodiscard]] u32 depth(u32 id) const noexcept { return m_depth[id]; }
    [[nodiscard]] bool skip(u32 id) const noexcept { return (m_flags[id] & 1) != 0; }
    void set_skip(u32 id, bool skip) noexcept { m_flags[id] = static_cast<u8>((m_flags[id] & ~1) | (skip ? 1 : 0)); }
    [[nodiscard]] bool closed(u32 id) const noexcept { return (m_flags[id] & 2) != 0; }
    void close(u32 id) noexcept { m_flags[id] |= 2; }
    [[nodiscard]] u32 parent(u32 id) const noexcept { return m_parent[id]; }

    void push(const u64* w, u32 n, u32 parent, u32 schema, const ObjectId* binding, u32 arity, u32 depth, bool skip)
    {
        if (n > m_W) [[unlikely]]
            widen(n);
        const usize at = m_words.size();
        m_words.resize(at + m_W);
        u64* dst = m_words.data() + at;
        MYMYR_NOVECTOR
        for (u32 i = 0; i < n; ++i)
            dst[i] = w[i];
        MYMYR_NOVECTOR
        for (u32 i = n; i < m_W; ++i)
            dst[i] = 0;
        m_parent.push_back(parent);
        m_schema.push_back(schema);
        m_depth.push_back(depth);
        m_flags.push_back(skip ? 1 : 0);
        m_boff.push_back(m_binding.size());
        if (arity)
            m_binding.insert(m_binding.end(), binding, binding + arity);
    }

    [[nodiscard]] Action action(u32 id, Successors& succ) const
    {
        const u32 s = m_schema[id];
        const ObjectId* b = m_binding.data() + m_boff[id];
        return Action(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s)));
    }
    [[nodiscard]] std::vector<Action> plan(u32 id, Successors& succ) const
    {
        std::vector<Action> out;
        for (u32 v = id; m_parent[v] != k_no_node; v = m_parent[v])
            out.push_back(action(v, succ));
        std::reverse(out.begin(), out.end());
        return out;
    }

    [[nodiscard]] u64 bytes() const noexcept
    {
        return m_words.capacity() * 8 + (m_parent.capacity() + m_schema.capacity() + m_depth.capacity()) * 4 + m_flags.capacity() +
               m_boff.capacity() * 8 + m_binding.capacity() * sizeof(ObjectId);
    }

private:
    void widen(u32 n)
    {
        const u32 nw = std::max(m_W * 2, n);
        const usize rows = m_parent.size();
        std::vector<u64> w(rows * nw, 0);
        for (usize i = 0; i < rows; ++i)
            std::memcpy(w.data() + i * nw, m_words.data() + i * m_W, m_W * sizeof(u64));
        m_words.swap(w);
        m_W = nw;
    }

    u32 m_W;
    std::vector<u64> m_words;
    std::vector<u32> m_parent, m_schema, m_depth;
    std::vector<u8> m_flags;  // 1: skip expansion, 2: closed
    std::vector<u64> m_boff;
    std::vector<ObjectId> m_binding;
};

/// Distinct states of the tree, by content: the successors of the root and, in a beam with
/// BeamNovelty::SurvivorsOnly, every candidate.
class SeenStates
{
public:
    [[nodiscard]] u32 find(const NodeTree& tree, const u64* w, u32 n) const
    {
        const auto range = m_map.equal_range(hash::state_words(w, n));
        for (auto it = range.first; it != range.second; ++it)
            if (bits::equal(tree.words(it->second), tree.stride(), w, n))
                return it->second;
        return k_no_node;
    }
    void insert(const u64* w, u32 n, u32 id) { m_map.emplace(hash::state_words(w, n), id); }

private:
    std::unordered_multimap<u64, u32> m_map;
};

/// Runs process(schema, binding, delta) -> bool over the transitions of the prepared state `cv`: in generation order,
/// or, with a successor order hook, buffered and reordered by it (IwOptions::successor_order semantics).
class Driver
{
public:
    template<class Process>
    void run(Env& env, StateView cv, Process&& process)
    {
        Successors& succ = env.succ;
        if (env.successor_order && *env.successor_order)
        {
            m_actions.clear();
            m_slots.clear();
            m_ranges.clear();
            succ.generate<false>(
                [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                {
                    m_actions.emplace_back(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s)));
                    m_ranges.push_back({static_cast<u32>(m_slots.size()), static_cast<u32>(d.add.size()), static_cast<u32>(d.del.size())});
                    m_slots.insert(m_slots.end(), d.add.begin(), d.add.end());
                    m_slots.insert(m_slots.end(), d.del.begin(), d.del.end());
                    return true;
                },
                env.witness, env.canonical, env.symmetry);
            m_order.clear();
            (*env.successor_order)(cv, m_actions, m_order);
            const u32 m = static_cast<u32>(m_actions.size());
            m_taken.assign(m, 0);
            m_perm.clear();
            for (u32 i : m_order)
                if (i < m && !m_taken[i])
                {
                    m_taken[i] = 1;
                    m_perm.push_back(i);
                }
            for (u32 i = 0; i < m; ++i)
                if (!m_taken[i])
                    m_perm.push_back(i);
            for (u32 i : m_perm)
            {
                const std::array<u32, 3>& r = m_ranges[i];
                const Delta d{{m_slots.data() + r[0], r[1]}, {m_slots.data() + r[0] + r[1], r[2]}};
                if (!process(m_actions[i].schema.v, static_cast<const ObjectId*>(m_actions[i].binding.data()), d))
                    return;
            }
            return;
        }
        succ.generate<true>(process, env.witness, env.canonical, env.symmetry);
    }

private:
    std::vector<Action> m_actions;
    std::vector<SlotId> m_slots;
    std::vector<std::array<u32, 3>> m_ranges;
    std::vector<u32> m_order, m_perm;
    std::vector<u8> m_taken;
};

template<class P>
concept Peeking = requires(P& p, const u64* w, u32 n, std::span<const u32> add, const Delta& d) {
    { p.peek(w, n, w, n, add, d) } -> std::convertible_to<bool>;
};

template<class P>
concept QuickRejecting = requires(P& p, std::span<const u32> add, const Delta& d) {
    { p.quick_reject(add, d) } -> std::convertible_to<bool>;
};

/// One BrFS pass from `root` with `pruner`.
template<class Pruner>
void novelty_pass(Env& env, StateView root, Pruner& pruner, const PassConfig& pc, PassOut& out)
{
    const auto t0 = Clock::now();
    const Task& task = env.task;
    const Budget& budget = env.control.budget;
    SearchObserver* const obs = env.obs;
    Successors& succ = env.succ;
    IwPassStatistics& st = out.st;
    st.arity = pc.arity;
    const bool slow = env.slow();
    const bool ordered = env.layers && env.layers->ordered();
    const bool limit_layer = ordered && env.layers->limited();
    const bool beam = ordered && env.layers->beam();
    const bool survivors = ordered && env.layers->survivors_only();
    if (survivors && !Peeking<Pruner>)
        throw std::invalid_argument("mymyr: BeamNovelty::SurvivorsOnly needs a novelty pruner with a read-only test");

    NodeTree tree(std::max(task.words(), root.nw));
    const u32 rn = bits::trimmed_size(root.w, root.nw);
    tree.push(root.w, rn, k_no_node, k_no_node, nullptr, 0, 0, false);
    if (env.tracker)
    {
        env.tracker->record(root.w, rn);
        env.tracker->flush_derived(succ);
    }
    const bool root_enters = pruner.init(root.w, rn);  // false: mimir's brfs returns FAILED (the ladder goes on)
    SeenStates seen;
    Driver driver;

    std::vector<u32> cur, next;
    if (root_enters)
        cur.push_back(0);
    std::vector<u32>* target = ordered ? &next : &cur;
    usize pos = 0;
    std::vector<u64> words, child;
    std::vector<u32> add;
    bool stop = false, status_set = false, truncated = false;
    auto finish = [&](SearchStatus s)
    {
        st.status = s;
        status_set = true;
    };
    const u64 interval = std::max<u64>(env.control.progress_interval, 1);
    u64 progress_next = interval;
    u32 g = 0;  // the layer being popped (coordination)

    auto action_of = [&](u32 s, const ObjectId* b) { return Action(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s))); };

    // SurvivorsOnly: the kept entries of `next` in rank order through the marking test (mimir's finalize_beam_layer and
    // on_end_beam_replay: marking as it goes is the same as its delta set committed at the end).
    std::vector<u64> rparent, rchild;
    std::vector<u32> radd;
    std::vector<SlotId> rslots;
    auto replay = [&]()
    {
        usize kept = 0;
        for (u32 e : next)
        {
            const u32 p = tree.parent(e);
            const u32 W = tree.stride();
            rparent.assign(tree.words(p), tree.words(p) + W);
            rchild.assign(tree.words(e), tree.words(e) + W);
            const u32 pn = bits::trimmed_size(rparent.data(), W), cn = bits::trimmed_size(rchild.data(), W);
            radd.clear();
            rslots.clear();
            for (u32 i = 0; i < W; ++i)
                for (u64 x = rchild[i] & ~rparent[i]; x; x &= x - 1)
                    radd.push_back(i * 64 + static_cast<u32>(bits::ctz64(x)));
            for (u32 a : radd)
                rslots.push_back(SlotId{a});
            const usize nadd = rslots.size();
            for (u32 i = 0; i < W; ++i)
                for (u64 x = rparent[i] & ~rchild[i]; x; x &= x - 1)
                    rslots.push_back(SlotId{i * 64 + static_cast<u32>(bits::ctz64(x))});
            const Delta d{{rslots.data(), nadd}, {rslots.data() + nadd, rslots.size() - nadd}};
            pruner.begin(rparent.data(), pn);
            const bool novel = !radd.empty() && pruner.test(rparent.data(), pn, rchild.data(), cn, radd, d);
            if (p == 0 && pc.root == RootRule::Continuation)
                tree.set_skip(e, !novel && !pc.keep_depth_one);
            else if (!novel)
            {
                --st.generated_in_tree;
                continue;
            }
            next[kept++] = e;
        }
        next.resize(kept);
    };
    // the novelty of a candidate: read-only with SurvivorsOnly
    auto novelty = [&](const u64* pw, u32 pn, const u64* w, u32 nn, std::span<const u32> a, const Delta& d) -> bool
    {
        if constexpr (Peeking<Pruner>)
            if (survivors)
                return pruner.peek(pw, pn, w, nn, a, d);
        return pruner.test(pw, pn, w, nn, a, d);
    };

    while (!stop)
    {
        if (pos == cur.size())
        {
            if (!ordered || next.empty())
                break;
            st.generated_in_tree -= env.layers->select(next, succ, [&](u32 e) { return StateView{tree.words(e), tree.stride(), nullptr, 0}; });
            if (survivors && pc.root != RootRule::ArityZero)
            {
                replay();
                if (next.empty())
                    break;
            }
            cur.swap(next);
            next.clear();
            pos = 0;
            truncated = false;
        }
        const u32 id = cur[pos++];
        if (env.out_of_time())
        {
            finish(SearchStatus::OutOfTime);
            break;
        }
        if (env.cancelled())
        {
            finish(SearchStatus::Cancelled);
            break;
        }
        if (tree.closed(id))
            continue;
        tree.close(id);
        const u32 depth = tree.depth(id);
        if (depth > g)
        {
            if (env.coord)
            {
                env.coord->publish_completed_depth(g);
                if (env.coord->is_incumbent_certified(env.coord->get_incumbent_length()))
                {
                    finish(SearchStatus::Cancelled);
                    break;
                }
            }
            g = depth;
        }
        const u32 W = tree.stride();
        words.assign(tree.words(id), tree.words(id) + W);  // the tree may move while this state is expanded
        const u32 n = bits::trimmed_size(words.data(), W);
        const u64* cw = words.data();
        const StateView cv{cw, n, nullptr, 0};
        bool prepared = false;
        if (env.goal.needs_view())
        {
            succ.prepare(cv);
            prepared = true;
        }
        if (env.goal.test(succ, cv, prepared))
        {
            finish(SearchStatus::Solved);
            out.plan = tree.plan(id, succ);
            out.goal_state = State(cw, n);
            out.goal_node = id;
            break;
        }
        if (st.expanded >= budget.max_expanded)
        {
            finish(SearchStatus::OutOfStates);
            break;
        }
        ++st.expanded;
        if (env.coord)
            env.coord->add_expansions(1);
        if (obs)
        {
            obs->on_expand(id, cv);
            if (st.expanded >= progress_next)
            {
                progress_next += interval;
                if (!obs->on_progress(st.statistics()))
                {
                    finish(SearchStatus::Cancelled);
                    break;
                }
            }
        }
        if (tree.skip(id) || depth >= budget.max_depth)
        {
            ++st.skipped;
            continue;
        }
        if (!prepared)
            succ.prepare(cv);
        pruner.begin(cw, n);
        const u32 cdepth = depth + 1;
        const bool at_root = id == 0;
        const RootRule rule = at_root ? pc.root : (pc.root == RootRule::ArityZero ? RootRule::ArityZero : RootRule::Normal);

        // admits a successor (a new node, or a width-0 duplicate entry); false stops the enumeration
        auto admit_node = [&](u32 s, const ObjectId* b, const u64* w, u32 nn, bool skip) -> bool
        {
            const u32 nid = tree.size();
            tree.push(w, nn, id, s, b, succ.arity(s), cdepth, skip);
            ++st.generated_in_tree;
            target->push_back(nid);
            if (tree.size() >= budget.max_states)
            {
                finish(SearchStatus::OutOfStates);
                stop = true;
                return false;
            }
            if (limit_layer && next.size() >= env.layers->limit())
            {
                truncated = true;
                return false;
            }
            return true;
        };
        auto reject = [&](u32 s, const ObjectId* b, const u64* w, u32 nn, TransitionOutcome outcome = TransitionOutcome::Pruned) -> bool
        {
            if (obs)
            {
                const Action a = action_of(s, b);
                const StateView c{w, nn, nullptr, 0};
                obs->on_generate(id, a, ~u64{0}, c, false);
                obs->on_prune(id, a, c);
                obs->on_transition(id, a, ~u64{0}, c, outcome);
            }
            return true;
        };
        // on_generate + on_transition of a successor that enters the tree (Opened) or is a width-0 duplicate entry
        auto report = [&](u32 s, const ObjectId* b, const u64* w, u32 nn, u32 child_id, TransitionOutcome outcome)
        {
            if (obs)
            {
                const Action a = action_of(s, b);
                const StateView c{w, nn, nullptr, 0};
                obs->on_generate(id, a, child_id, c, true);
                obs->on_transition(id, a, child_id, c, outcome);
            }
        };
        auto process = [&](u32 s, const ObjectId* b, const Delta& d) -> bool
        {
            ++st.generated;
            if (rule == RootRule::ArityZero && !at_root && !slow)
                return true;  // width 0 below the root: pruned, nothing to report
            add.clear();
            for (SlotId x : d.add)
                if (!bits::test(cw, n, x.v) && std::find(add.begin(), add.end(), x.v) == add.end())
                    add.push_back(x.v);
            if constexpr (QuickRejecting<Pruner>)
                if (!slow && rule == RootRule::Normal && pruner.quick_reject(add, d))
                    return true;
            const u32 nn = apply_delta(cw, n, d, child);
            const u64* w = child.data();
            if (env.tracker)
                env.tracker->record_successor(w, nn, add);  // the parent (expanded) is recorded
            if (!env.blocked.empty() && env.blocked.contains(StateView{w, nn, nullptr, 0}))
            {
                ++st.blocked;  // checked first: a blocked state marks nothing
                return reject(s, b, w, nn);
            }
            if (rule == RootRule::ArityZero && !at_root)
                return reject(s, b, w, nn);
            if (add.empty() && bits::equal(w, nn, cw, n))
                return reject(s, b, w, nn);  // self loop: Pruned, as control.hpp documents
            if (rule == RootRule::ArityZero)
            {
                const u32 dup = seen.find(tree, w, nn);
                if (dup != k_no_node && survivors)
                    return reject(s, b, w, nn, TransitionOutcome::Duplicate);  // mimir's ArityZero beam selection: not new
                if (dup != k_no_node)
                {
                    // mimir admits it again (a queue entry) and skips it when popped: it is closed by then
                    ++st.generated_in_tree;
                    report(s, b, w, nn, dup, TransitionOutcome::Duplicate);
                    target->push_back(dup);
                    if (limit_layer && next.size() >= env.layers->limit())
                    {
                        truncated = true;
                        return false;
                    }
                    return true;
                }
                seen.insert(w, nn, tree.size());
                report(s, b, w, nn, tree.size(), TransitionOutcome::Opened);
                return admit_node(s, b, w, nn, pc.root_only);
            }
            if (rule == RootRule::Continuation)
            {
                if (seen.find(tree, w, nn) != k_no_node)
                    return reject(s, b, w, nn, TransitionOutcome::Duplicate);  // not new
                const bool novel = novelty(cw, n, w, nn, add, d);
                seen.insert(w, nn, tree.size());
                report(s, b, w, nn, tree.size(), TransitionOutcome::Opened);
                return admit_node(s, b, w, nn, !novel && !pc.keep_depth_one);
            }
            if (!novelty(cw, n, w, nn, add, d))
                return reject(s, b, w, nn);
            if (survivors)
            {
                if (seen.find(tree, w, nn) != k_no_node)
                    return reject(s, b, w, nn, TransitionOutcome::Duplicate);  // a candidate before (mimir: not new)
                seen.insert(w, nn, tree.size());
            }
            report(s, b, w, nn, tree.size(), TransitionOutcome::Opened);
            return admit_node(s, b, w, nn, false);
        };
        driver.run(env, cv, process);
        if (env.tracker && env.tracker->pending())
            env.tracker->flush_derived(succ);
        if (truncated)
            pos = cur.size();  // mimir drops the rest of the layer once the next one is full
    }
    if (!status_set)
    {
        st.status = SearchStatus::Exhausted;
        if (env.coord)
            env.coord->invalidate_lower_bound();
    }
    if (beam)
        st.generated_in_tree -= next.size();  // candidates of a next layer that was never selected
    st.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    out.table_bytes = pruner.bytes();
    out.node_bytes = tree.bytes();
}

/// An IW-style ladder over passes: run_pass(k, PassOut&) for k = first..top (after an optional placeholder entry
/// for arity 0). Fills status, plan, goal_state, passes, total, effective_width, message and peaks (as run_ladder).
template<class RunPass>
IwResult novelty_ladder(Env& env, u32 first, u32 top, bool placeholder, RunPass&& run_pass)
{
    IwResult r;
    auto push = [&](const IwPassStatistics& p)
    {
        r.passes.push_back(p);
        add_pass(r.total, p);
        if (env.root)
            env.root->on_pass(p.arity, p.statistics());
    };
    if (placeholder)
    {
        IwPassStatistics p;
        p.arity = 0;
        p.placeholder = true;
        push(p);
    }
    if (env.goal.statically_false())
    {
        IwPassStatistics p;
        p.arity = first;
        p.status = SearchStatus::Unsolvable;
        push(p);
        r.status = SearchStatus::Unsolvable;
        return r;
    }
    for (u32 k = first; k <= top; ++k)
    {
        if (env.out_of_time())
        {
            r.status = SearchStatus::OutOfTime;
            return r;
        }
        if (env.cancelled())
        {
            r.status = SearchStatus::Cancelled;
            return r;
        }
        PassOut po;
        try
        {
            run_pass(k, po);
        }
        catch (const std::length_error& e)
        {
            r.status = SearchStatus::Failed;
            r.message = e.what();
            return r;
        }
        r.peak_table_bytes = std::max(r.peak_table_bytes, po.table_bytes);
        r.peak_node_bytes = std::max(r.peak_node_bytes, po.node_bytes);
        push(po.st);
        if (po.st.status == SearchStatus::Solved)
        {
            r.status = SearchStatus::Solved;
            r.plan = std::move(po.plan);
            r.goal_state = std::move(po.goal_state);
            r.effective_width = k;
            return r;
        }
        if (po.st.status != SearchStatus::Exhausted)
        {
            r.status = po.st.status;
            return r;
        }
    }
    r.status = SearchStatus::Exhausted;
    return r;
}

/// Plan cost, slot count and the solution/end events of a finished search.
inline void finish_result(Env& env, const State& root, IwResult& r)
{
    if (r.status == SearchStatus::Solved)
    {
        const PlanCost pc(env.task);
        r.cost = pc.apply(env.succ, root, r.plan);
        r.cost_exact = pc.exact();
    }
    r.fluent_slots = env.task.atoms().fluent_slots();
    if (env.root)
    {
        if (r.status == SearchStatus::Solved)
            env.root->on_solution(r.plan, r.cost);
        env.root->on_end(r.status, r.total);
    }
}

/// The plain IW(k) pruner (novelty/novelty_table.hpp), for ladders that run on this engine (rollouts, certifier).
class ClassicPruner
{
public:
    ClassicPruner(const Task& task, u32 k, const novelty::TableOptions& options)
        : m_task(task), m_table(k, std::max<u32>(task.atoms().max_fluent_slots(), 1), options)
    {
        m_table.reserve(std::max<u32>(task.atoms().fluent_slots(), 1));
    }
    bool init(const u64* w, u32 n)
    {
        reserve_words(n);
        m_table.mark_state(w, n);
        return true;  // as search::iw(): the start state always enters
    }
    void begin(const u64*, u32) {}
    bool quick_reject(std::span<const u32> add, const Delta&)
    {
        if (add.empty())
            return true;
        if (m_table.arity() != 1)
            return false;
        reserve(add);
        return !m_table.any_new1(add);
    }
    bool test(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta&)
    {
        if (add.empty())
            return false;
        reserve(add);
        return m_table.test<true>(pw, pn, cw, cn, add);
    }
    bool peek(const u64* pw, u32 pn, const u64* cw, u32 cn, std::span<const u32> add, const Delta&)
    {
        if (add.empty())
            return false;
        reserve(add);
        return m_table.test<false>(pw, pn, cw, cn, add);
    }
    [[nodiscard]] u64 bytes() const { return m_table.bytes(); }

private:
    void reserve(std::span<const u32> add)
    {
        u32 mx = 0;
        for (u32 a : add)
            mx = std::max(mx, a);
        if (mx >= m_table.capacity()) [[unlikely]]
            m_table.reserve(std::max<u32>(m_task.atoms().fluent_slots(), mx + 1));
    }
    void reserve_words(u32 n)
    {
        if (n * 64 > m_table.capacity())
            m_table.reserve(std::max<u32>(m_task.atoms().fluent_slots(), n * 64));
    }

    const Task& m_task;
    novelty::NoveltyTable m_table;
};

/// The classic IW ladder on this engine: arity 0 (or the optimized IW(1) placeholder), then IW(1..max_arity).
/// Equal to search::iw() pass for pass (tests/cpp/search/test_iw_variants.cpp); additionally supports ordered
/// layers, coordination and state tracking.
inline IwResult classic_ladder(Env& env, StateView root, u32 max_arity, bool optimize_iw1, WidthZero width_zero, const novelty::TableOptions& tables)
{
    if (max_arity > novelty::k_max_arity)
    {
        IwResult r;
        r.status = SearchStatus::Failed;
        r.message = "IW arity " + std::to_string(max_arity) + " exceeds the maximum " + std::to_string(novelty::k_max_arity);
        return r;
    }
    const bool optimize = optimize_iw1 && max_arity == 1;
    return novelty_ladder(env, optimize ? 1 : 0, max_arity, optimize,
                          [&](u32 k, PassOut& po)
                          {
                              if (k == 0)
                              {
                                  NullPruner p;
                                  novelty_pass(env, root, p,
                                               PassConfig{.arity = 0, .root = RootRule::ArityZero, .root_only = width_zero == WidthZero::RootOnly},
                                               po);
                                  return;
                              }
                              ClassicPruner p(env.task, k, tables);
                              novelty_pass(env, root, p,
                                           PassConfig{.arity = k, .root = optimize && k == 1 ? RootRule::Continuation : RootRule::Normal}, po);
                          });
}
}  // namespace mymyr::search::detail
