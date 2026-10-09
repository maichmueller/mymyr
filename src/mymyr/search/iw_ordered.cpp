// The width-1 IW pass under a layer transition ordering (search/transition_ordering.hpp): mimir 0.16.3's
// brfs::find_solution_with_transition_ordering (src/search/algorithms/brfs/transition_ordered_layer_impl.hpp) with the
// ArityK novelty pruning strategy of width 1 (iw/pruning_strategy.cpp), including the optimized-IW(1) root rule. The
// pop, budget and observer conventions are those of the queued pass in iw.cpp.

#include "iw_detail.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/novelty/novelty_table.hpp"
#include "mymyr/search/transition_ordering.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace mymyr::search::detail
{
namespace
{
constexpr u32 k_none = ~u32{0};

/// The admitted states with their plan records (variable-length words: lazy slots may widen later states).
class Nodes
{
public:
    [[nodiscard]] u32 size() const noexcept { return static_cast<u32>(m_parent.size()); }
    [[nodiscard]] StateView view(u32 id) const noexcept
    {
        return {m_words.data() + m_woff[id], m_wlen[id], nullptr, 0};
    }
    [[nodiscard]] u32 depth(u32 id) const noexcept { return m_depth[id]; }
    [[nodiscard]] bool skip(u32 id) const noexcept { return m_skip[id] != 0; }

    void push(const u64* w, u32 n, u32 parent, u32 schema, std::span<const ObjectId> binding, u32 depth, bool skip)
    {
        m_woff.push_back(m_words.size());
        m_wlen.push_back(n);
        m_words.insert(m_words.end(), w, w + n);
        m_parent.push_back(parent);
        m_schema.push_back(schema);
        m_boff.push_back(m_binding.size());
        m_binding.insert(m_binding.end(), binding.begin(), binding.end());
        m_depth.push_back(depth);
        m_skip.push_back(skip ? 1 : 0);
    }

    [[nodiscard]] std::vector<Action> plan(u32 id, const Successors& succ) const
    {
        std::vector<Action> out;
        for (u32 v = id; m_parent[v] != k_none; v = m_parent[v])
        {
            const ObjectId* b = m_binding.data() + m_boff[v];
            out.emplace_back(SchemaId{m_schema[v]}, std::vector<ObjectId>(b, b + succ.arity(m_schema[v])));
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    [[nodiscard]] u64 bytes() const noexcept
    {
        return m_words.capacity() * 8 + (m_woff.capacity() + m_boff.capacity()) * 8 +
               (m_wlen.capacity() + m_parent.capacity() + m_schema.capacity() + m_depth.capacity()) * 4 + m_skip.capacity() +
               m_binding.capacity() * sizeof(ObjectId);
    }

private:
    std::vector<u64> m_words, m_woff, m_boff;
    std::vector<u32> m_wlen, m_parent, m_schema, m_depth;
    std::vector<u8> m_skip;
    std::vector<ObjectId> m_binding;
};

/// One generated transition of the current layer.
struct Candidate
{
    u32 parent;
    u32 schema;
    u64 boff;     // into binding
    u64 woff;     // child words, into words
    u32 wlen;
    u64 aoff;     // add slots false in the parent, into adds
    u32 alen;
};
}  // namespace

void run_ordered_pass(Context& c, StateView root, const GoalTest& goal, bool root_continuation, OrderedPassOut& out)
{
    const auto t0 = Clock::now();
    const Task& task = c.task;
    const IwOptions& o = c.options;
    const Budget& budget = o.control.budget;
    const CancelToken& cancel = o.control.cancel;
    SearchObserver* const obs = c.observer;
    Successors& succ = c.succ;
    IwPassStatistics& st = out.st;
    st.arity = 1;

    // the ordered pass stores atom words only; numeric states would lose their values
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: IW transition ordering does not support tasks with numeric fluents");

    Nodes nodes;
    const u32 rn = bits::trimmed_size(root.w, root.nw);
    nodes.push(root.w, rn, k_none, k_none, {}, 0, false);
    novelty::NoveltyTable table(1, std::max<u32>(task.atoms().max_fluent_slots(), 1), o.tables);
    table.reserve(std::max<u32>(task.atoms().fluent_slots(), 1));
    table.mark_state(root.w, root.nw);
    std::unordered_multimap<u64, u32> root_seen;  // the optimized-IW(1) root admits each distinct successor once

    bool status_set = false;
    auto finish = [&](SearchStatus s)
    {
        st.status = s;
        status_set = true;
    };
    const u64 interval = std::max<u64>(o.control.progress_interval, 1);
    u64 progress_next = interval;

    std::vector<Candidate> cands;
    std::vector<u64> words, adds_buf;
    std::vector<ObjectId> binding;
    std::vector<u32> adds;
    std::vector<u64> cur, next;
    std::vector<LayerTransition> views;
    std::vector<u32> order, perm;
    std::vector<u8> taken;
    u32 layer_begin = 0, layer_end = 1;
    bool stop = false;

    while (!stop && layer_begin < layer_end)
    {
        cands.clear();
        words.clear();
        adds_buf.clear();
        binding.clear();

        // 1. pop and generate
        for (u32 id = layer_begin; id < layer_end; ++id)
        {
            if (c.out_of_time())
            {
                finish(SearchStatus::OutOfTime);
                stop = true;
                break;
            }
            if (cancel.requested())
            {
                finish(SearchStatus::Cancelled);
                stop = true;
                break;
            }
            const StateView sv = nodes.view(id);
            cur.assign(sv.w, sv.w + sv.nw);
            const u32 n = sv.nw;
            const StateView cv{cur.data(), n, nullptr, 0};
            bool prepared = false;
            if (goal.needs_view())
            {
                succ.prepare(cv);
                prepared = true;
            }
            if (goal.test(succ, cv, prepared))
            {
                finish(SearchStatus::Solved);
                out.plan = nodes.plan(id, succ);
                out.goal_state = State(cur.data(), n);
                stop = true;
                break;
            }
            if (st.expanded >= budget.max_expanded)
            {
                finish(SearchStatus::OutOfStates);
                stop = true;
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
                        stop = true;
                        break;
                    }
                }
            }
            if (nodes.skip(id) || nodes.depth(id) >= budget.max_depth)
            {
                ++st.skipped;
                continue;
            }
            if (!prepared)
                succ.prepare(cv);
            succ.generate<false>(
                [&](u32 s, const ObjectId* b, const Delta& d) -> bool
                {
                    ++st.generated;
                    const u32 nn = apply_delta(cur.data(), n, d, next);
                    Candidate x{id, s, binding.size(), words.size(), nn, adds_buf.size(), 0};
                    binding.insert(binding.end(), b, b + succ.arity(s));
                    words.insert(words.end(), next.data(), next.data() + nn);
                    for (SlotId a : d.add)
                        if (!bits::test(cur.data(), n, a.v) &&
                            std::find(adds_buf.begin() + static_cast<std::ptrdiff_t>(x.aoff), adds_buf.end(), a.v) == adds_buf.end())
                            adds_buf.push_back(a.v);
                    x.alen = static_cast<u32>(adds_buf.size() - x.aoff);
                    cands.push_back(x);
                    return true;
                },
                o.witness_pruning, o.canonical_order);
        }
        if (stop)
            break;

        // 2. order the layer
        const u32 m = static_cast<u32>(cands.size());
        views.clear();
        views.reserve(m);
        for (const Candidate& x : cands)
            views.push_back({nodes.view(x.parent),
                             ActionLabel{SchemaId{x.schema}, std::span<const ObjectId>(binding.data() + x.boff, succ.arity(x.schema))},
                             StateView{words.data() + x.woff, x.wlen, nullptr, 0}});
        order.clear();
        o.transition_ordering->order(task, views, order);
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

        // 3. admit in that order
        const u32 next_begin = nodes.size();
        for (u32 i : perm)
        {
            const Candidate& x = cands[i];
            const StateView pv = nodes.view(x.parent);
            const StateView child{words.data() + x.woff, x.wlen, nullptr, 0};
            const std::span<const ObjectId> b(binding.data() + x.boff, succ.arity(x.schema));
            auto reject = [&]()
            {
                if (obs)
                {
                    const Action a(SchemaId{x.schema}, std::vector<ObjectId>(b.begin(), b.end()));
                    obs->on_generate(x.parent, a, ~u64{0}, child, false);
                    obs->on_prune(x.parent, a, child);
                }
            };
            if (!c.blocked.empty() && c.blocked.contains(child))
            {
                ++st.blocked;  // checked first: a blocked state marks no tuple
                reject();
                continue;
            }
            adds.assign(adds_buf.begin() + static_cast<std::ptrdiff_t>(x.aoff), adds_buf.begin() + static_cast<std::ptrdiff_t>(x.aoff + x.alen));
            u32 mx = 0;
            for (u32 a : adds)
                mx = std::max(mx, a);
            if (!adds.empty() && mx >= table.capacity()) [[unlikely]]
                table.reserve(std::max<u32>(task.atoms().fluent_slots(), mx + 1));
            bool skip = false;
            if (root_continuation && x.parent == 0)
            {
                // mimir's optimized IW(1) root: every new successor other than the root enters, the non-novel ones
                // are not expanded
                if (bits::equal(child.w, child.nw, pv.w, pv.nw))
                {
                    reject();
                    continue;
                }
                const u64 h = hash::state_words(child.w, child.nw);
                bool dup = false;
                for (auto [it, end] = root_seen.equal_range(h); it != end && !dup; ++it)
                {
                    const StateView e = nodes.view(it->second);
                    dup = bits::equal(e.w, e.nw, child.w, child.nw);
                }
                if (dup)
                {
                    reject();
                    continue;
                }
                skip = adds.empty() || !table.test<true>(pv.w, pv.nw, child.w, child.nw, adds);
                root_seen.emplace(h, nodes.size());
            }
            else if (adds.empty() || !table.test<true>(pv.w, pv.nw, child.w, child.nw, adds))
            {
                reject();
                continue;
            }
            if (obs)
                obs->on_generate(x.parent, Action(SchemaId{x.schema}, std::vector<ObjectId>(b.begin(), b.end())), nodes.size(), child, true);
            nodes.push(child.w, child.nw, x.parent, x.schema, b, nodes.depth(x.parent) + 1, skip);
            ++st.generated_in_tree;
            if (nodes.size() >= budget.max_states)
            {
                finish(SearchStatus::OutOfStates);
                stop = true;
                break;
            }
        }
        layer_begin = next_begin;
        layer_end = nodes.size();
    }
    if (!status_set)
        st.status = SearchStatus::Exhausted;
    st.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    out.table_bytes = table.bytes();
    out.node_bytes = nodes.bytes();
}
}  // namespace mymyr::search::detail
