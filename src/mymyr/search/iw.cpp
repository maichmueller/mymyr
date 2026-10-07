// IW(k) passes and the IW ladder (search/iw.hpp), with mimir's pass semantics (mimir 0.16.3 brfs::find_solution with
// the ArityZero / ArityK novelty pruning strategies, src/search/algorithms/brfs.cpp and iw/pruning_strategy.cpp):
// goal test on pop, blocked states before the novelty test, the width-0 and optimized-IW(1) root rules.

#include "iw_detail.hpp"
#include "layer_order_detail.hpp"

#include "mymyr/search/goal.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace mymyr::search
{
const char* mimir_status_name(SearchStatus s) noexcept
{
    switch (s)
    {
        case SearchStatus::Solved: return "solved";
        case SearchStatus::Exhausted: return "failed";  // mimir's IW/SIW: every pass exhausted
        case SearchStatus::OutOfStates: return "out_of_states";
        case SearchStatus::OutOfTime: return "out_of_time";
        case SearchStatus::Cancelled: return "canceled";
        case SearchStatus::Failed: return "error";
        case SearchStatus::Unsolvable: return "unsolvable";
    }
    return "?";
}

namespace detail
{
// ------------------------------------------------------------------------------------------------- GoalTest
GoalTest GoalTest::from_spec(const Task& task, const GoalSpec& spec)
{
    GoalTest g;
    switch (spec.kind)
    {
        case GoalSpec::Kind::Task:
        {
            g = counter(task, 0);
            g.m_kind = Kind::Task;
            break;
        }
        case GoalSpec::Kind::AnyOf:
            g.m_kind = Kind::AnyOf;
            g.m_view = std::ranges::any_of(spec.goals, &GoalSpec::AtomGoal::needs_view);
            break;
        case GoalSpec::Kind::Custom:
            if (!spec.test)
                throw std::invalid_argument("GoalSpec::Kind::Custom needs a test function");
            g.m_kind = Kind::Custom;
            break;
    }
    g.m_spec = &spec;
    return g;
}

GoalTest GoalTest::counter(const Task& task, u32 threshold)
{
    GoalTest g;
    g.m_kind = Kind::Counter;
    g.m_threshold = threshold;
    const plan::Goal& goal = task.compiled().goal;
    g.m_static_false = goal.unsatisfiable;
    g.m_view = goal.uses_derived;
    // each distinct literal once (mimir counts the atoms of its goal condition, a set per kind and polarity)
    for (const plan::Check& c : goal.lits)
    {
        const bool dup = std::any_of(g.m_lits.begin(), g.m_lits.end(),
                                     [&](const plan::Check& d)
                                     { return d.pos == c.pos && d.pat.kind == c.pat.kind && d.pat.pred == c.pat.pred && d.pat.base == c.pat.base; });
        if (!dup)
            g.m_lits.push_back(c);
    }
    return g;
}

void GoalTest::point(Successors& succ, StateView s, bool prepared) const
{
    if (prepared)
        return;
    if (m_view)
        succ.prepare(s);
    else
    {
        succ.engine().set_state(s.w, s.nw);
        succ.engine().set_numeric(s.num);
    }
}

bool GoalTest::test(Successors& succ, StateView s, bool prepared) const
{
    switch (m_kind)
    {
        case Kind::Task:
        {
            if (m_static_false)
                return false;
            point(succ, s, prepared);
            for (const plan::Check& c : m_lits)
                if (!succ.engine().holds(c))
                    return false;
            for (const plan::NumCheck& c : succ.engine().compiled().num.goal)
                if (!succ.engine().holds(c))
                    return false;
            return true;
        }
        case Kind::AnyOf:
            if (m_view)
                point(succ, s, prepared);
            for (const GoalSpec::AtomGoal& g : m_spec->goals)
                if (holds(g, succ, s))
                    return true;
            return false;
        case Kind::Custom: return m_spec->test(s);
        case Kind::Counter: return unsatisfied(succ, s, prepared) < m_threshold;
    }
    return false;
}

u32 GoalTest::unsatisfied(Successors& succ, StateView s, bool prepared) const
{
    point(succ, s, prepared);
    u32 c = 0;
    for (const plan::Check& l : m_lits)
        c += succ.engine().holds(l) ? 0 : 1;
    return c;
}

// ------------------------------------------------------------------------------------------------- BlockedSet
BlockedSet::BlockedSet(const std::vector<State>& states) : m_states(&states)
{
    m_index.reserve(states.size());
    for (u32 i = 0; i < states.size(); ++i)
        m_index.emplace_back(states[i].hash(), i);
    std::sort(m_index.begin(), m_index.end());
}

bool BlockedSet::contains(StateView v) const
{
    const u64 h = v.hash();
    auto it = std::lower_bound(m_index.begin(), m_index.end(), std::pair<u64, u32>{h, 0});
    for (; it != m_index.end() && it->first == h; ++it)
        if ((*m_states)[it->second].view() == v)
            return true;
    return false;
}

Context::Context(const Task& t, const IwOptions& o, Successors& s)
    : task(t), options(o), succ(s), blocked(o.control.blocked_states), observer(o.control.observer)
{
    const double secs = o.control.budget.max_seconds;
    timed = secs < 1e15;  // finite
    if (timed)
        deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)));
    error = check_layers(o.layers);
    if (error.empty() && o.layers.kind != LayerOrdering::Kind::Queue)
    {
        if (o.transition_ordering)
            error = "IwOptions::layers cannot be combined with a transition ordering";
        else
            layers = std::make_unique<LayerOrderer>(t, o.layers);
    }
}

Context::~Context() = default;

void add_pass(SearchStatistics& total, const IwPassStatistics& p)
{
    const SearchStatistics s = p.statistics();
    total.expanded += s.expanded;
    total.generated += s.generated;
    total.states += s.states;
    total.pruned += s.pruned;
    total.seconds += s.seconds;
}

namespace
{
constexpr u32 k_none = ~u32{0};

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

/// The states of one IW tree: fixed-stride words (re-laid out when lazy slots widen the states, as in the probe) and
/// SoA plan records.
class Tree
{
public:
    explicit Tree(u32 words, u32 numeric_words = 0) : m_W(std::max<u32>(1, words)), m_NN(numeric_words) {}

    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_parent.size()); }
    /// Fluent words per row; a row is [stride() bits | numeric_words()] (numeric tasks).
    [[nodiscard]] u32 stride() const noexcept { return m_W; }
    [[nodiscard]] u32 row() const noexcept { return m_W + m_NN; }
    [[nodiscard]] const u64* words(u32 id) const noexcept { return m_words.data() + static_cast<usize>(id) * row(); }
    [[nodiscard]] u32 depth(u32 id) const noexcept { return m_depth[id]; }
    [[nodiscard]] bool skip(u32 id) const noexcept { return m_skip[id] != 0; }
    void set_skip(u32 id, bool skip) noexcept { m_skip[id] = skip ? 1 : 0; }
    [[nodiscard]] u32 parent(u32 id) const noexcept { return m_parent[id]; }

    void push(const u64* w, u32 n, const u64* num, u32 parent, u32 schema, const ObjectId* binding, u32 arity, u32 depth,
              bool skip)
    {
        if (n > m_W) [[unlikely]]
            widen(n);
        const usize at = m_words.size();
        m_words.resize(at + row());
        u64* dst = m_words.data() + at;
        MYMYR_NOVECTOR
        for (u32 i = 0; i < n; ++i)
            dst[i] = w[i];
        MYMYR_NOVECTOR
        for (u32 i = n; i < m_W; ++i)
            dst[i] = 0;
        MYMYR_NOVECTOR
        for (u32 i = 0; i < m_NN; ++i)
            dst[m_W + i] = num[i];
        m_parent.push_back(parent);
        m_schema.push_back(schema);
        m_depth.push_back(depth);
        m_skip.push_back(skip ? 1 : 0);
        m_boff.push_back(m_binding.size());
        if (arity)
            m_binding.insert(m_binding.end(), binding, binding + arity);
    }

    [[nodiscard]] std::vector<Action> plan(u32 id, const Successors& succ) const
    {
        std::vector<Action> out;
        for (u32 v = id; m_parent[v] != k_none; v = m_parent[v])
        {
            const u32 s = m_schema[v];
            const ObjectId* b = m_binding.data() + m_boff[v];
            out.emplace_back(SchemaId{s}, std::vector<ObjectId>(b, b + const_cast<Successors&>(succ).arity(s)));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    [[nodiscard]] u64 bytes() const noexcept
    {
        return m_words.capacity() * 8 + (m_parent.capacity() + m_schema.capacity() + m_depth.capacity()) * 4 + m_skip.capacity() +
               m_boff.capacity() * 8 + m_binding.capacity() * sizeof(ObjectId);
    }

private:
    void widen(u32 n)
    {
        const u32 nw = std::max(m_W * 2, n);
        const usize rows = m_parent.size();
        const usize nrow = static_cast<usize>(nw) + m_NN;
        std::vector<u64> w(rows * nrow, 0);
        for (usize i = 0; i < rows; ++i)
        {
            std::memcpy(w.data() + i * nrow, m_words.data() + i * row(), m_W * sizeof(u64));
            if (m_NN)
                std::memcpy(w.data() + i * nrow + nw, m_words.data() + i * row() + m_W, m_NN * sizeof(u64));
        }
        m_words.swap(w);
        m_W = nw;
    }

    u32 m_W;
    u32 m_NN = 0;
    std::vector<u64> m_words;
    std::vector<u32> m_parent, m_schema, m_depth;
    std::vector<u8> m_skip;
    std::vector<u64> m_boff;
    std::vector<ObjectId> m_binding;
};

struct PassSpec
{
    u32 arity = 0;
    bool root_continuation = false;  // mimir's optimized IW(1): admit every distinct root successor
};

struct PassOut
{
    IwPassStatistics st;
    std::vector<Action> plan;
    std::optional<State> goal_state;
    u64 table_bytes = 0;
    u64 node_bytes = 0;
};

/// Distinct states of the tree, by content: the successors of the root (width 0 and the optimized IW(1) root rule
/// deduplicate them) and, in a beam with BeamNovelty::SurvivorsOnly, every candidate (mimir's "is new" test: a
/// candidate the beam dropped has marked nothing, so it could pass the read-only novelty test again).
class SeenStates
{
public:
    /// The id of an earlier state equal to s, or k_none.
    [[nodiscard]] u32 find(const Tree& tree, StateView s) const
    {
        const auto range = m_map.equal_range(s.hash());
        for (auto it = range.first; it != range.second; ++it)
        {
            const u64* r = tree.words(it->second);
            if (StateView{r, tree.stride(), s.nnum ? r + tree.stride() : nullptr, s.nnum} == s)
                return it->second;
        }
        return k_none;
    }
    void insert(StateView s, u32 id) { m_map.emplace(s.hash(), id); }

private:
    std::unordered_multimap<u64, u32> m_map;
};

/// One IW pass. Slow: an observer or blocked states are set (every successor is materialized).
template<bool Slow>
void run_pass(Context& c, StateView root, const GoalTest& goal, const PassSpec& ps, PassOut& out)
{
    const auto t0 = Clock::now();
    const Task& task = c.task;
    const IwOptions& o = c.options;
    const Budget& budget = o.control.budget;
    const CancelToken& cancel = o.control.cancel;
    SearchObserver* const obs = c.observer;
    Successors& succ = c.succ;
    IwPassStatistics& st = out.st;
    st.arity = ps.arity;
    const u32 k = ps.arity;
    const bool witness = o.witness_pruning, canonical = o.canonical_order;
    const bool root_only = o.width_zero == WidthZero::RootOnly;

    const u32 NN = task.numeric_words();
    Tree tree(std::max(task.words(), root.nw), NN);
    tree.push(root.w, bits::trimmed_size(root.w, root.nw), root.num, k_none, k_none, nullptr, 0, 0, false);
    std::optional<novelty::NoveltyTable> table;
    if (k >= 1)
    {
        table.emplace(k, std::max<u32>(task.atoms().max_fluent_slots(), 1), o.tables);
        table->reserve(std::max<u32>(task.atoms().fluent_slots(), 1));
        table->mark_state(root.w, root.nw);
    }
    SeenStates seen;

    std::vector<u64> cur, next;
    std::vector<u32> add;
    bool stop = false;
    bool status_set = false;
    auto finish = [&](SearchStatus s)
    {
        st.status = s;
        status_set = true;
    };
    const u64 interval = std::max<u64>(o.control.progress_interval, 1);
    u64 progress_next = interval;

    // Ordered layers (IwOptions::layers): `layer` holds the entries of the current layer in expansion order, `next`
    // those of the next one in generation order (width-0 duplicates are entries of their first node, as in mimir,
    // and are skipped once their node is closed). Otherwise the tree itself is the queue.
    LayerOrderer* const lo = c.layers.get();
    const bool limit_layer = lo && lo->limited();
    // A beam (LayerOrdering::beam_width): candidates enter the tree and the next layer as they are generated; at the
    // boundary the layer is ordered and cut, generated_in_tree then counts only the kept entries. SurvivorsOnly:
    // the novelty test only reads the table during generation (`seen` stands in for mimir's "is new" test) and the
    // kept entries are replayed through the marking test (replay below).
    const bool beam = lo && lo->beam();
    const bool survivors = lo && lo->survivors_only();
    std::vector<u32> layer{0}, next_layer;
    std::vector<u8> closed;
    usize lpos = 0;
    bool truncated = false;
    // an entry of the next layer; false stops the expansion (the next layer is full)
    auto enter = [&](u32 node) -> bool
    {
        next_layer.push_back(node);
        if (limit_layer && next_layer.size() >= lo->limit())
        {
            truncated = true;
            return false;
        }
        return true;
    };

    // admit a successor into the tree; false stops the expansion (max_states, or a full next layer)
    auto admit = [&](const u64* w, u32 nn, const u64* num, u32 parent, u32 schema, const ObjectId* b, u32 depth, bool skip) -> bool
    {
        tree.push(w, nn, num, parent, schema, b, succ.arity(schema), depth, skip);
        ++st.generated_in_tree;
        if (tree.size() >= budget.max_states)
        {
            finish(SearchStatus::OutOfStates);
            stop = true;
            return false;
        }
        return !lo || enter(tree.size() - 1);
    };
    auto action_of = [&](u32 s, const ObjectId* b) { return Action(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s))); };
    // the add set: effect adds that are false in the parent, each once
    auto compute_add = [&](const Delta& d, u32 n)
    {
        add.clear();
        for (SlotId x : d.add)
            if (!bits::test(cur.data(), n, x.v) && std::find(add.begin(), add.end(), x.v) == add.end())
                add.push_back(x.v);
    };
    auto reserve_for_add = [&]()
    {
        u32 mx = 0;
        for (u32 a : add)
            mx = std::max(mx, a);
        if (mx >= table->capacity()) [[unlikely]]
            table->reserve(std::max<u32>(task.atoms().fluent_slots(), mx + 1));
    };

    // Runs process(schema, binding, delta) -> bool over the transitions of the prepared state: in generation order,
    // or (IwOptions::successor_order, slow path only) buffered and reordered by the hook.
    std::vector<Action> buf_actions;
    std::vector<SlotId> buf_slots;
    std::vector<u64> buf_num;
    std::vector<std::array<u32, 3>> buf_ranges;  // (first slot, adds, dels)
    std::vector<u32> order, perm;
    std::vector<u8> taken;
    auto drive = [&](StateView cv, auto&& process)
    {
        if constexpr (Slow)
        {
            if (o.successor_order)
            {
                buf_actions.clear();
                buf_slots.clear();
                buf_num.clear();
                buf_ranges.clear();
                succ.generate<false>(
                    [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                    {
                        buf_actions.emplace_back(SchemaId{s}, std::vector<ObjectId>(b, b + succ.arity(s)));
                        if (NN)
                            buf_num.insert(buf_num.end(), d.num, d.num + NN);
                        buf_ranges.push_back({static_cast<u32>(buf_slots.size()), static_cast<u32>(d.add.size()), static_cast<u32>(d.del.size())});
                        buf_slots.insert(buf_slots.end(), d.add.begin(), d.add.end());
                        buf_slots.insert(buf_slots.end(), d.del.begin(), d.del.end());
                        return true;
                    },
                    witness, canonical);
                order.clear();
                o.successor_order(cv, buf_actions, order);
                // complete to a permutation: invalid and repeated indices are dropped, missing ones appended
                const u32 m = static_cast<u32>(buf_actions.size());
                taken.assign(m, 0);
                perm.clear();
                for (u32 i : order)
                    if (i < m && !taken[i])
                    {
                        taken[i] = 1;
                        perm.push_back(i);
                    }
                for (u32 i = 0; i < m; ++i)
                    if (!taken[i])
                        perm.push_back(i);
                for (u32 i : perm)
                {
                    const std::array<u32, 3>& r = buf_ranges[i];
                    const Delta d{{buf_slots.data() + r[0], r[1]},
                                  {buf_slots.data() + r[0] + r[1], r[2]},
                                  NN ? buf_num.data() + static_cast<usize>(i) * NN : nullptr,
                                  NN,
                                  {},
                                  {}};
                    if (!process(buf_actions[i].schema.v, static_cast<const ObjectId*>(buf_actions[i].binding.data()), d))
                        return;
                }
                return;
            }
        }
        succ.generate<true>(process, witness, canonical);
    };

    // SurvivorsOnly: the kept entries in rank order, each tested and marked against the table plus the tuples of the
    // better ranked ones (mimir's finalize_beam_layer and on_end_beam_replay: marking as it goes is the same as its
    // delta set committed at the end). One that adds no tuple leaves the layer; a root successor of the optimized
    // IW(1) pass stays and is only marked as not to be expanded.
    std::vector<u64> rparent, rchild;
    auto replay = [&]()
    {
        usize kept = 0;
        for (u32 e : next_layer)
        {
            const u32 p = tree.parent(e);
            const u32 W = tree.stride();
            rparent.assign(tree.words(p), tree.words(p) + W);
            rchild.assign(tree.words(e), tree.words(e) + W);
            const u32 pn = bits::trimmed_size(rparent.data(), W), cn = bits::trimmed_size(rchild.data(), W);
            add.clear();
            for (u32 i = 0; i < cn; ++i)
                for (u64 x = rchild[i] & ~(i < pn ? rparent[i] : u64{0}); x; x &= x - 1)
                    add.push_back(i * 64 + static_cast<u32>(bits::ctz64(x)));
            bool novel = false;
            if (!add.empty())
            {
                reserve_for_add();
                novel = table->test<true>(rparent.data(), pn, rchild.data(), cn, add);
            }
            if (p == 0 && ps.root_continuation)
                tree.set_skip(e, !novel);
            else if (!novel)
            {
                --st.generated_in_tree;
                continue;
            }
            next_layer[kept++] = e;
        }
        next_layer.resize(kept);
    };

    for (u32 step = 0;; ++step)
    {
        u32 id = step;
        if (lo)
        {
            if (lpos == layer.size())
            {
                if (next_layer.empty())
                    break;
                st.generated_in_tree -= lo->select(next_layer, succ, [&](u32 e) {
                    const u64* r = tree.words(e);
                    return StateView{r, tree.stride(), NN ? r + tree.stride() : nullptr, NN};
                });
                if (survivors && k >= 1)
                {
                    replay();
                    if (next_layer.empty())
                        break;
                }
                layer.swap(next_layer);
                next_layer.clear();
                lpos = 0;
                truncated = false;
            }
            id = layer[lpos++];
            closed.resize(tree.size(), 0);
            if (closed[id])
                continue;
            closed[id] = 1;
        }
        else if (id >= tree.size())
            break;
        if (c.out_of_time())
        {
            finish(SearchStatus::OutOfTime);
            break;
        }
        if (cancel.requested())
        {
            finish(SearchStatus::Cancelled);
            break;
        }
        const u32 W = tree.stride();
        cur.assign(tree.words(id), tree.words(id) + tree.row());  // the tree may move while this state is expanded
        const u32 n = bits::trimmed_size(cur.data(), W);
        const StateView cv{cur.data(), n, NN ? cur.data() + W : nullptr, NN};
        bool prepared = false;
        if (goal.needs_view())
        {
            succ.prepare(cv);
            prepared = true;
        }
        if (goal.test(succ, cv, prepared))
        {
            finish(SearchStatus::Solved);
            out.plan = tree.plan(id, succ);
            out.goal_state = State(cv);
            break;
        }
        if (st.expanded >= budget.max_expanded)
        {
            finish(SearchStatus::OutOfStates);
            break;
        }
        ++st.expanded;
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
        const u32 depth = tree.depth(id);
        if (tree.skip(id) || depth >= budget.max_depth)
        {
            ++st.skipped;
            continue;
        }
        if (!prepared)
            succ.prepare(cv);
        const u32 cdepth = depth + 1;

        if (id == 0 && (k == 0 || ps.root_continuation))
        {
            // the root of a width-0 pass (mimir's ArityZero root rule) or of an optimized IW(1) pass
            drive(cv,
                [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                {
                    ++st.generated;
                    const u32 nn = apply_delta(cur.data(), n, d, next);
                    const StateView child{next.data(), nn, d.num, NN};
                    auto reject = [&]()
                    {
                        if constexpr (Slow)
                            if (obs)
                            {
                                const Action a = action_of(s, b);
                                obs->on_generate(id, a, ~u64{0}, child, false);
                                obs->on_prune(id, a, child);
                            }
                        return true;
                    };
                    if constexpr (Slow)
                        if (!c.blocked.empty() && c.blocked.contains(child))
                        {
                            ++st.blocked;
                            return reject();
                        }
                    if (child == cv)
                        return reject();  // self loop
                    const u32 dup = seen.find(tree, child);
                    if (k == 0)
                    {
                        if (dup != k_none && survivors)
                            return reject();  // mimir's ArityZero beam selection: not new
                        if (dup != k_none)
                        {
                            // mimir admits it again (counted in the tree) but pops it as closed: no second node
                            ++st.generated_in_tree;
                            if constexpr (Slow)
                                if (obs)
                                    obs->on_generate(id, action_of(s, b), dup, child, true);
                            return !lo || enter(dup);
                        }
                        const u32 nid = tree.size();
                        seen.insert(child, nid);
                        if constexpr (Slow)
                            if (obs)
                                obs->on_generate(id, action_of(s, b), nid, child, true);
                        return admit(next.data(), nn, d.num, id, s, b, cdepth, root_only);
                    }
                    // optimized IW(1): every new root successor enters; the non-novel ones are not expanded
                    if (dup != k_none)
                        return reject();
                    compute_add(d, n);
                    bool novel = false;
                    if (!add.empty())
                    {
                        reserve_for_add();
                        novel = survivors ? table->test<false>(cur.data(), n, next.data(), nn, add)
                                          : table->test<true>(cur.data(), n, next.data(), nn, add);
                    }
                    const u32 nid = tree.size();
                    seen.insert(child, nid);
                    if constexpr (Slow)
                        if (obs)
                            obs->on_generate(id, action_of(s, b), nid, child, true);
                    return admit(next.data(), nn, d.num, id, s, b, cdepth, !novel);
                });
        }
        else if (k == 0)
        {
            // width 0 below the root: every successor is pruned
            drive(cv,
                [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                {
                    ++st.generated;
                    if constexpr (Slow)
                    {
                        const u32 nn = apply_delta(cur.data(), n, d, next);
                        const StateView child{next.data(), nn, d.num, NN};
                        if (!c.blocked.empty() && c.blocked.contains(child))
                            ++st.blocked;
                        if (obs)
                        {
                            const Action a = action_of(s, b);
                            obs->on_generate(id, a, ~u64{0}, child, false);
                            obs->on_prune(id, a, child);
                        }
                    }
                    return true;
                });
        }
        else
        {
            drive(cv,
                [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                {
                    ++st.generated;
                    compute_add(d, n);
                    if constexpr (!Slow)
                    {
                        if (add.empty())
                            return true;  // no added atom: no new tuple
                        reserve_for_add();
                        u32 nn;
                        if (k == 1)
                        {
                            if (!table->any_new1(add))
                                return true;  // add-effect precheck: the successor is never materialized
                            if (!survivors) [[likely]]
                                table->mark1(add);
                            nn = apply_delta(cur.data(), n, d, next);
                        }
                        else
                        {
                            nn = apply_delta(cur.data(), n, d, next);
                            if (!(survivors ? table->test<false>(cur.data(), n, next.data(), nn, add)
                                            : table->test<true>(cur.data(), n, next.data(), nn, add)))
                                return true;
                        }
                        if (survivors) [[unlikely]]
                        {
                            const StateView child{next.data(), nn, d.num, NN};
                            if (seen.find(tree, child) != k_none)
                                return true;
                            seen.insert(child, tree.size());
                        }
                        return admit(next.data(), nn, d.num, id, s, b, cdepth, false);
                    }
                    else
                    {
                        const u32 nn = apply_delta(cur.data(), n, d, next);
                        const StateView child{next.data(), nn, d.num, NN};
                        bool admitted = false;
                        if (!c.blocked.empty() && c.blocked.contains(child))
                            ++st.blocked;  // checked first: a blocked state marks no tuple
                        else if (!add.empty())
                        {
                            reserve_for_add();
                            admitted = survivors ? table->test<false>(cur.data(), n, next.data(), nn, add)
                                                 : table->test<true>(cur.data(), n, next.data(), nn, add);
                            if (admitted && survivors)
                            {
                                admitted = seen.find(tree, child) == k_none;
                                if (admitted)
                                    seen.insert(child, tree.size());
                            }
                        }
                        if (!admitted)
                        {
                            if (obs)
                            {
                                const Action a = action_of(s, b);
                                obs->on_generate(id, a, ~u64{0}, child, false);
                                obs->on_prune(id, a, child);
                            }
                            return true;
                        }
                        if (obs)
                            obs->on_generate(id, action_of(s, b), tree.size(), child, true);
                        return admit(next.data(), nn, d.num, id, s, b, cdepth, false);
                    }
                });
        }
        if (stop)
            break;
        if (truncated)
            lpos = layer.size();  // mimir drops the rest of the layer once the next one is full
    }
    if (!status_set)
        st.status = SearchStatus::Exhausted;
    if (beam)
        st.generated_in_tree -= next_layer.size();  // candidates of a next layer that was never selected
    st.seconds = seconds_since(t0);
    out.table_bytes = table ? table->bytes() : 0;
    out.node_bytes = tree.bytes();
}
}  // namespace

IwResult run_ladder(Context& c, StateView root, const GoalTest& goal, u32 max_arity, int only_arity)
{
    IwResult r;
    const IwOptions& o = c.options;
    if (!c.error.empty())
    {
        r.status = SearchStatus::Failed;
        r.message = c.error;
        return r;
    }
    const u32 top = only_arity >= 0 ? static_cast<u32>(only_arity) : max_arity;
    if (top > novelty::k_max_arity)
    {
        r.status = SearchStatus::Failed;
        r.message = "IW arity " + std::to_string(top) + " exceeds the maximum " + std::to_string(novelty::k_max_arity);
        return r;
    }
    const bool optimize = only_arity < 0 && o.optimize_iw1 && max_arity == 1;
    const u32 first = only_arity >= 0 ? top : optimize ? 1 : 0;
    auto push = [&](const IwPassStatistics& p)
    {
        r.passes.push_back(p);
        add_pass(r.total, p);
        if (c.observer)
            c.observer->on_pass(p.arity, p.statistics());
    };
    if (optimize)
    {
        IwPassStatistics p;
        p.arity = 0;
        p.placeholder = true;
        push(p);
    }
    if (goal.statically_false())
    {
        // mimir's first pass returns UNSOLVABLE before expanding anything
        IwPassStatistics p;
        p.arity = first;
        p.status = SearchStatus::Unsolvable;
        push(p);
        r.status = SearchStatus::Unsolvable;
        return r;
    }
    const bool slow = c.observer != nullptr || !c.blocked.empty() || static_cast<bool>(o.successor_order);
    for (u32 k = first; k <= top; ++k)
    {
        if (c.out_of_time())
        {
            r.status = SearchStatus::OutOfTime;
            return r;
        }
        if (o.control.cancel.requested())
        {
            r.status = SearchStatus::Cancelled;
            return r;
        }
        PassOut po;
        const PassSpec ps{.arity = k, .root_continuation = optimize && k == 1};
        try
        {
            if (k == 1 && o.transition_ordering)
            {
                OrderedPassOut oo;
                run_ordered_pass(c, root, goal, ps.root_continuation, oo);
                po = {std::move(oo.st), std::move(oo.plan), std::move(oo.goal_state), oo.table_bytes, oo.node_bytes};
            }
            else if (slow)
                run_pass<true>(c, root, goal, ps, po);
            else
                run_pass<false>(c, root, goal, ps, po);
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
}  // namespace detail

namespace
{
IwResult run_iw(const Task& task, const IwOptions& options, int only_arity)
{
    Successors& succ = task.workspace().successors();
    detail::Context c(task, options, succ);
    const State root = options.start ? *options.start : task.initial_state();
    if (c.observer)
        c.observer->on_start(root);
    const detail::GoalTest goal = detail::GoalTest::from_spec(task, options.control.goal);
    IwResult r = detail::run_ladder(c, root, goal, options.max_arity, only_arity);
    if (r.status == SearchStatus::Solved)
    {
        const detail::PlanCost pc(task);
        r.cost = pc.apply(succ, root, r.plan);
        r.cost_exact = pc.exact();
    }
    r.fluent_slots = task.atoms().fluent_slots();
    if (c.observer)
    {
        if (r.status == SearchStatus::Solved)
            c.observer->on_solution(r.plan, r.cost);
        c.observer->on_end(r.status, r.total);
    }
    return r;
}
}  // namespace

IwResult iw(const Task& task, const IwOptions& options) { return run_iw(task, options, -1); }

IwResult iw_pass(const Task& task, u32 arity, const IwOptions& options) { return run_iw(task, options, static_cast<int>(arity)); }
}  // namespace mymyr::search
