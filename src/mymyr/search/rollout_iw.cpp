// Rollout IW(1) (search/rollout_iw.hpp): a port of mimir's RolloutIWSearch (src/search/algorithms/rollout_iw.cpp)
// and its action orderings (rollout_iw/action_ordering.cpp), with portable randomization.

#include "rollout_detail.hpp"

#include "iw_detail.hpp"

#include "mymyr/core/random.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mymyr::search
{
const char* to_string(ActionOrdering o) noexcept
{
    switch (o)
    {
        case ActionOrdering::InOrder: return "in_order";
        case ActionOrdering::Randomized: return "randomized";
        case ActionOrdering::DirectGoalAchieverFirst: return "direct_goal_achiever_first";
        case ActionOrdering::GoalRegressionRelevance: return "goal_regression_relevance";
        case ActionOrdering::MixedRegressionRandom: return "mixed_regression_random";
    }
    return "?";
}

namespace detail
{
namespace
{
constexpr u32 k_none = ~u32{0};
constexpr u32 k_inf = ~u32{0};

/// Positive fluent goal atoms the goal-directed orderings aim at (see the header).
std::vector<CanonicalAtom> ordering_goal_atoms(const Task& task, const GoalSpec& spec)
{
    std::vector<CanonicalAtom> out;
    if (spec.kind == GoalSpec::Kind::AnyOf)
    {
        for (const GoalSpec::AtomGoal& g : spec.goals)
            for (SlotId s : g.positive)
                out.push_back(task.atoms().canonical(AtomKind::Fluent, s.v));
    }
    else
    {
        const formalism::TaskData& t = task.data();
        const CanonicalLayout& L = task.atoms().layout();
        std::vector<u32> args;
        for (const formalism::Literal& l : t.literals_of(t.goal))
        {
            if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
                continue;
            args.clear();
            for (formalism::Term x : t.terms_of(l))
                args.push_back(formalism::term_object(x).v);
            const CanonicalAtom c = L.encode(l.pred.v, args.data());
            if (c < L.fluent_count)
                out.push_back(c);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

/// Mimir's compute_schema_regression_ranks over the task's schemas.
std::vector<u32> regression_ranks(const Task& task, const std::vector<CanonicalAtom>& goal_atoms)
{
    const formalism::TaskData& t = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    std::vector<u32> pred_rank(t.predicates.size(), k_inf);
    std::vector<u32> args(std::max<u32>(1, L.max_arity));
    for (CanonicalAtom c : goal_atoms)
        pred_rank[L.decode(c, args.data())] = 0;
    auto best_add = [&](const formalism::Schema& s)
    {
        u32 best = k_inf;
        for (const formalism::ConditionalEffect& ce : t.effects_of(s))
            for (const formalism::Literal& l : formalism::TaskData::slice(t.literals, ce.effects))
                if (l.positive)
                    best = std::min(best, pred_rank[l.pred.v]);
        return best;
    };
    for (usize pass = 0; pass <= t.schemas.size(); ++pass)
    {
        bool changed = false;
        for (const formalism::Schema& s : t.schemas)
        {
            const u32 best = best_add(s);
            if (best == k_inf)
                continue;
            for (const formalism::Literal& l : t.literals_of(s.precondition))
                if (t.predicate(l.pred).kind == formalism::PredKind::Fluent && best + 1 < pred_rank[l.pred.v])
                {
                    pred_rank[l.pred.v] = best + 1;
                    changed = true;
                }
        }
        if (!changed)
            break;
    }
    std::vector<u32> out;
    out.reserve(t.schemas.size());
    for (const formalism::Schema& s : t.schemas)
        out.push_back(best_add(s));
    return out;
}

class RolloutSearch
{
public:
    RolloutSearch(const Task& task, const RolloutIwOptions& o, SearchObserver* hot)
        : m_task(task), m_o(o), m_lease(task.workspace()), m_succ(m_lease->successors()), m_goal(GoalTest::from_spec(task, o.control.goal)),
          m_blocked(o.control.blocked_states), m_hot(hot), m_coord(o.control.coordination), m_store(std::max<u32>(task.words(), 1), 10),
          m_rng(o.seed)
    {
        const double secs = o.control.budget.max_seconds;
        m_timed = secs < 1e15;
        if (m_timed)
            m_deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(secs, 0.0)));
        const bool directed = o.ordering == ActionOrdering::DirectGoalAchieverFirst;
        const bool regression = o.ordering == ActionOrdering::GoalRegressionRelevance || o.ordering == ActionOrdering::MixedRegressionRandom;
        if (directed || regression)
        {
            m_goal_atoms = ordering_goal_atoms(task, o.control.goal);
            if (regression)
                m_schema_rank = regression_ranks(task, m_goal_atoms);
            if (directed)
                prepare_directed();
        }
    }

    RolloutIwResult run()
    {
        const auto t0 = Clock::now();
        RolloutIwResult r;
        RolloutIwStatistics& st = m_st;
        const State root = m_o.start ? *m_o.start : m_task.initial_state();
        if (m_goal.statically_false())
        {
            r.status = SearchStatus::Unsolvable;
            r.message = "static goal cannot hold";
            return r;
        }
        add_node(root.data(), root.size_words(), k_none, k_none, 0);
        register_features(0, 0);
        if (m_goal.test(m_succ, root.view(), false))
            m_goal_node = 0;
        while (!m_root_solved && m_goal_node == k_none)
        {
            if (st.rollouts >= m_o.max_rollouts)
            {
                m_interrupt = Interrupt::Rollouts;
                break;
            }
            ++st.rollouts;
            if (!rollout())
                break;
        }
        r.root_solved = m_root_solved;
        if (m_goal_node != k_none)
        {
            r.status = SearchStatus::Solved;
            r.plan = plan(m_goal_node);
            const StateView g = m_store[StateId{m_nodes[m_goal_node].state}];
            r.goal_state = State(g.w, g.nw);
            const PlanCost pc(m_task);
            r.cost = pc.apply(m_succ, root, r.plan);
            r.cost_exact = pc.exact();
        }
        else
            switch (m_interrupt)
            {
                case Interrupt::Time:
                    r.status = SearchStatus::OutOfTime;
                    r.message = "time budget expired";
                    break;
                case Interrupt::Cancelled:
                    r.status = SearchStatus::Cancelled;
                    r.message = "cancelled";
                    break;
                case Interrupt::States:
                    r.status = SearchStatus::OutOfStates;
                    r.message = "state budget expired";
                    break;
                case Interrupt::Rollouts:
                    r.status = SearchStatus::Failed;
                    r.message = "rollout budget expired";
                    break;
                case Interrupt::None:
                    r.status = SearchStatus::Exhausted;
                    r.message = st.depth_bound_prunings + st.incumbent_bound_prunings > 0 ? "width-1 space exhausted under the depth/incumbent bounds"
                                                                                          : "width-1 space exhausted";
                    break;
            }
        st.tree_nodes = m_nodes.size();
        st.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        r.statistics = st;
        r.fluent_slots = m_task.atoms().fluent_slots();
        return r;
    }

private:
    enum class Interrupt : u8
    {
        None,
        Time,
        Cancelled,
        States,
        Rollouts,
    };
    struct Node
    {
        u32 state;
        u32 parent;
        u32 in_pos;  // position in the parent's action list
        u32 depth;
        u32 first = 0, count = 0;  // the actions (pools below), valid once materialized
        u32 unsolved = 0;          // actions whose child is missing or not solved
        u32 cursor = 0;            // static orders: no action before it has an unsolved child
        u32 witness = ~u32{0};     // an atom that last showed the node novel (still_novel)
        bool solved = false;
        bool materialized = false;
    };

    // ------------------------------------------------------------------ budgets
    /// check_time false: only the (atomic) cancel flags, which are cheap enough to test on every rollout step, so
    /// that a shared expansion cap stops every worker at its next step; the clock is read every 64 steps.
    bool should_interrupt(bool check_time = true)
    {
        if (check_time && m_timed && Clock::now() >= m_deadline)
        {
            m_interrupt = Interrupt::Time;
            return true;
        }
        if (m_o.control.cancel.requested() || (m_coord && m_coord->is_cancelled()))
        {
            m_interrupt = Interrupt::Cancelled;
            return true;
        }
        return false;
    }
    [[nodiscard]] u32 incumbent_bound() const
    {
        u32 b = m_o.incumbent_bound;
        if (m_coord)
            b = std::min(b, m_coord->get_incumbent_length());
        return b;
    }

    // ------------------------------------------------------------------ tree
    u32 add_node(const u64* w, u32 n, u32 parent, u32 in_pos, u32 depth)
    {
        const u32 sid = m_store.insert(w, n).first.v;
        m_nodes.push_back(Node{.state = sid, .parent = parent, .in_pos = in_pos, .depth = depth});
        m_st.max_rollout_depth = std::max(m_st.max_rollout_depth, depth);
        return static_cast<u32>(m_nodes.size() - 1);
    }
    [[nodiscard]] StateView view(u32 node) const { return m_store[StateId{m_nodes[node].state}]; }
    [[nodiscard]] ActionLabel label(u32 action) const
    {
        const u32 s = m_schema[action];
        return ActionLabel{SchemaId{s}, {m_binding.data() + m_boff[action], m_succ_arity[s]}};
    }
    std::vector<Action> plan(u32 node) const
    {
        std::vector<Action> out;
        for (u32 v = node; m_nodes[v].parent != k_none; v = m_nodes[v].parent)
        {
            const ActionLabel a = label(m_nodes[m_nodes[v].parent].first + m_nodes[v].in_pos);
            out.emplace_back(a);
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    /// All applicable actions of a node, in (replayed) generation order. False: a budget expired meanwhile (the node
    /// stays unmaterialized).
    bool materialize(u32 node)
    {
        const StateView sv = view(node);
        m_succ.prepare(sv);
        m_buf_schema.clear();
        m_buf_boff.clear();
        m_buf_binding.clear();
        m_buf_goal.clear();
        m_buf_delta.clear();
        m_buf_doff.clear();
        u32 yields = 0;
        bool interrupted = false;
        const bool directed = m_o.ordering == ActionOrdering::DirectGoalAchieverFirst;
        m_succ.generate<true>(
            [&](u32 s, const ObjectId* b, const Delta& d) -> bool
            {
                const u32 ar = m_succ.arity(s);
                m_buf_schema.push_back(s);
                m_buf_boff.push_back(static_cast<u32>(m_buf_binding.size()));
                m_buf_binding.insert(m_buf_binding.end(), b, b + ar);
                // the successor's delta in this state (deletes, then adds), so that generating the child later is
                // apply_delta on the node's words instead of a new engine evaluation of the action
                m_buf_doff.push_back(static_cast<u32>(m_buf_delta.size()));
                m_buf_doff.push_back(static_cast<u32>(d.del.size()));
                m_buf_delta.insert(m_buf_delta.end(), d.del.begin(), d.del.end());
                m_buf_delta.insert(m_buf_delta.end(), d.add.begin(), d.add.end());
                if (s >= m_succ_arity.size())
                    m_succ_arity.resize(s + 1, 0);
                m_succ_arity[s] = ar;
                if (directed)
                    m_buf_goal.push_back(adds_goal(s, b, d) ? 1 : 0);
                if ((++yields & 63) == 0 && should_interrupt())
                {
                    interrupted = true;
                    return false;
                }
                return true;
            },
            false, m_o.canonical_order, m_o.symmetry_pruning);
        if (interrupted)
            return false;
        const u32 m = static_cast<u32>(m_buf_schema.size());
        // the generation order, replayed if a hook is set
        m_perm.resize(m);
        std::iota(m_perm.begin(), m_perm.end(), 0u);
        if (m_o.successor_order)
        {
            m_buf_actions.clear();
            for (u32 i = 0; i < m; ++i)
                m_buf_actions.emplace_back(SchemaId{m_buf_schema[i]}, std::vector<ObjectId>(m_buf_binding.begin() + m_buf_boff[i],
                                                                                            m_buf_binding.begin() + m_buf_boff[i] + m_succ_arity[m_buf_schema[i]]));
            m_order.clear();
            m_o.successor_order(sv, m_buf_actions, m_order);
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
        }
        // static orders are applied once, here: the node's actions are stored in rank order (stable, as mimir's
        // per-visit stable sort of the generation order), so a visit takes the first action without a solved child
        if (directed)
            std::stable_partition(m_perm.begin(), m_perm.end(), [&](u32 i) { return m_buf_goal[i] != 0; });
        else if (m_o.ordering == ActionOrdering::GoalRegressionRelevance)
            std::stable_sort(m_perm.begin(), m_perm.end(), [&](u32 a, u32 b) { return schema_rank_of(m_buf_schema[a]) < schema_rank_of(m_buf_schema[b]); });
        Node& nd = m_nodes[node];
        nd.first = static_cast<u32>(m_schema.size());
        nd.count = m;
        nd.unsolved = m;
        nd.cursor = 0;
        for (u32 k = 0; k < m; ++k)
        {
            const u32 i = m_perm[k];
            const u32 sc = m_buf_schema[i];
            const u32 d0 = m_buf_doff[2 * i], d1 = i + 1 < m ? m_buf_doff[2 * i + 2] : static_cast<u32>(m_buf_delta.size());
            m_doff.push_back(m_delta.size());
            m_dndel.push_back(m_buf_doff[2 * i + 1]);
            m_delta.insert(m_delta.end(), m_buf_delta.begin() + d0, m_buf_delta.begin() + d1);
            m_schema.push_back(sc);
            m_boff.push_back(m_binding.size());
            m_binding.insert(m_binding.end(), m_buf_binding.begin() + m_buf_boff[i], m_buf_binding.begin() + m_buf_boff[i] + m_succ_arity[sc]);
            m_child.push_back(k_none);
            m_act_solved.push_back(0);
            if (m_o.ordering == ActionOrdering::MixedRegressionRandom)
                m_act_rank.push_back(schema_rank_of(sc));
        }
        nd.materialized = true;
        ++m_st.expanded;
        if (m_coord)
            m_coord->add_expansions(1);
        if (m_hot)
        {
            m_hot->on_expand(node, sv);
            const u64 interval = std::max<u64>(m_o.control.progress_interval, 1);
            if (m_st.expanded % interval == 0 && !m_hot->on_progress(m_st.statistics()))
            {
                m_interrupt = Interrupt::Cancelled;
                return false;
            }
        }
        return true;
    }

    /// Marks a node solved and every ancestor whose actions all have solved children (mimir's
    /// propagate_solved_labels), by counting each node's unsolved actions.
    void mark_solved(u32 node)
    {
        if (m_nodes[node].solved)
            return;
        m_nodes[node].solved = true;
        for (u32 v = node;;)
        {
            const u32 parent = m_nodes[v].parent;
            if (parent == k_none)
                break;
            Node& pn = m_nodes[parent];
            m_act_solved[pn.first + m_nodes[v].in_pos] = 1;
            if (--pn.unsolved != 0)
                break;
            pn.solved = true;
            ++m_st.solved_propagations;
            v = parent;
        }
        m_root_solved = m_nodes.front().solved;
    }
    bool register_features(u32 node, u32 depth)
    {
        bool improved = false;
        const StateView sv = view(node);
        bits::for_each(sv.w, sv.nw,
                       [&](u64 a)
                       {
                           if (a >= m_best.size())
                               m_best.resize(a + 1, k_inf);
                           if (depth < m_best[a])
                           {
                               m_best[a] = depth;
                               improved = true;
                               ++m_st.feature_depth_improvements;
                           }
                       });
        return improved;
    }
    /// Whether some atom of the node still has its best depth at the node's depth (case 4). Best depths only
    /// decrease, so an atom that showed it stays a witness until its depth improves: it is tested first.
    bool still_novel(u32 node)
    {
        Node& nd = m_nodes[node];
        const u32 depth = nd.depth;
        auto holds = [&](u64 a) { return a >= m_best.size() || depth <= m_best[a]; };
        if (nd.witness != k_none && holds(nd.witness))
            return true;
        const StateView sv = view(node);
        for (u32 i = 0; i < sv.nw; ++i)
            for (u64 x = sv.w[i]; x; x &= x - 1)
            {
                const u64 a = static_cast<u64>(i) * 64 + static_cast<u64>(bits::ctz64(x));
                if (holds(a))
                {
                    nd.witness = static_cast<u32>(a);
                    return true;
                }
            }
        return false;
    }

    // ------------------------------------------------------------------ orderings
    void prepare_directed()
    {
        const formalism::TaskData& t = m_task.data();
        for (CanonicalAtom c : m_goal_atoms)
        {
            const u32 s = m_task.atoms().intern(c);
            if (s >= m_goal_slot.size())
                m_goal_slot.resize(s + 1, 0);
            m_goal_slot[s] = 1;
        }
        // conditional adds without forall parameters, per schema (inspected whether or not their condition holds)
        m_ce_adds.resize(t.schemas.size());
        for (u32 s = 0; s < t.schemas.size(); ++s)
            for (const formalism::ConditionalEffect& ce : t.effects_of(t.schemas[s]))
            {
                if (ce.extra_params.count != 0)
                    continue;
                for (const formalism::Literal& l : formalism::TaskData::slice(t.literals, ce.effects))
                    if (l.positive)
                        m_ce_adds[s].push_back(&l);
            }
    }
    bool adds_goal(u32 schema, const ObjectId* b, const Delta& d)
    {
        for (SlotId x : d.add)
            if (x.v < m_goal_slot.size() && m_goal_slot[x.v])
                return true;
        const formalism::TaskData& t = m_task.data();
        const CanonicalLayout& L = m_task.atoms().layout();
        for (const formalism::Literal* l : m_ce_adds[schema])
        {
            m_args.clear();
            for (formalism::Term x : t.terms_of(*l))
                m_args.push_back(formalism::is_object(x) ? formalism::term_object(x).v : b[formalism::term_parameter(x)].v);
            const CanonicalAtom c = L.encode(l->pred.v, m_args.data());
            if (c < L.fluent_count && std::binary_search(m_goal_atoms.begin(), m_goal_atoms.end(), c))
                return true;
        }
        return false;
    }
    [[nodiscard]] u32 schema_rank_of(u32 schema) const { return schema < m_schema_rank.size() ? m_schema_rank[schema] : k_inf; }

    /// The position of the action a visit of `node` follows: the first action without a solved child in the node's
    /// order (mimir ranks the actions on every visit: static orders are stored ranked, see materialize; the random
    /// ones shuffle the generation order on every visit, mixed then stable-sorts it by schema rank). k_none: none.
    u32 select(u32 node)
    {
        Node& nd = m_nodes[node];
        const u8* solved = m_act_solved.data() + nd.first;
        if (m_o.ordering != ActionOrdering::Randomized && m_o.ordering != ActionOrdering::MixedRegressionRandom)
        {
            while (nd.cursor < nd.count && solved[nd.cursor])
                ++nd.cursor;
            return nd.cursor < nd.count ? nd.cursor : k_none;
        }
        m_rank.resize(nd.count);
        std::iota(m_rank.begin(), m_rank.end(), 0u);
        m_rng.shuffle(std::span<u32>(m_rank));
        if (m_o.ordering == ActionOrdering::Randomized)
        {
            for (u32 p : m_rank)
                if (!solved[p])
                    return p;
            return k_none;
        }
        // mixed: the stable sort by rank puts first, among the unsolved actions of the smallest rank, the earliest
        const u32* rank = m_act_rank.data() + nd.first;
        u32 best = k_inf;
        for (u32 i = 0; i < nd.count; ++i)
            best = std::min(best, solved[i] ? k_inf : rank[i]);
        for (u32 p : m_rank)
            if (!solved[p] && rank[p] == best)
                return p;
        return k_none;
    }

    // ------------------------------------------------------------------ one rollout
    bool rollout()
    {
        u32 cur = 0;
        for (;;)
        {
            if (should_interrupt((m_steps++ & 63) == 0))
                return false;
            if (!m_nodes[cur].materialized && !materialize(cur))
                return false;
            if (m_nodes[cur].count == 0)
            {
                ++m_st.dead_ends;
                mark_solved(cur);
                return true;
            }
            const u32 pos = select(cur);
            if (pos == k_none)
            {
                mark_solved(cur);
                return true;
            }
            const u32 existing = m_child[m_nodes[cur].first + pos];
            if (existing != k_none)
            {
                if (still_novel(existing))
                {
                    ++m_st.case4;
                    cur = existing;
                    continue;
                }
                ++m_st.case3;
                mark_solved(existing);
                return true;
            }
            // an untried action: generate its successor
            const u32 action = m_nodes[cur].first + pos;
            const ActionLabel a = label(action);
            const StateView pv = view(cur);
            const usize d0 = m_doff[action], d1 = action + 1 < m_doff.size() ? m_doff[action + 1] : m_delta.size();
            const std::span<const SlotId> dels(m_delta.data() + d0, m_dndel[action]);
            const std::span<const SlotId> adds(m_delta.data() + d0 + m_dndel[action], d1 - d0 - m_dndel[action]);
            std::vector<u64>& cw = m_child_words;
            const u32 cn = apply_delta(pv.w, pv.nw, Delta{.add = adds, .del = dels}, cw);
            ++m_st.generated;
            const u32 depth = m_nodes[cur].depth + 1;
            const u32 child = add_node(cw.data(), cn, cur, pos, depth);
            m_child[action] = child;
            const StateView cv = view(child);
            auto report = [&](TransitionOutcome outcome, bool prune)
            {
                if (!m_hot)
                    return;
                const Action act(a);
                m_hot->on_generate(cur, act, child, cv, true);
                m_hot->on_transition(cur, act, child, cv, outcome);
                if (prune)
                    m_hot->on_prune(cur, act, cv);
            };
            if (!m_blocked.empty() && m_blocked.contains(cv))
            {
                ++m_st.blocked;
                report(TransitionOutcome::Pruned, true);
                mark_solved(child);
                return true;
            }
            // the goal first, so that a goal on a budget or bound boundary is still found
            if (m_goal.test(m_succ, cv, false))
            {
                report(TransitionOutcome::Goal, false);
                m_goal_node = child;
                return false;
            }
            if (m_st.generated >= m_o.control.budget.max_states)
            {
                report(TransitionOutcome::Pruned, true);
                m_interrupt = Interrupt::States;
                return false;
            }
            if (depth >= m_o.control.budget.max_depth)
            {
                ++m_st.depth_bound_prunings;
                report(TransitionOutcome::Pruned, true);
                mark_solved(child);
                return true;
            }
            const u32 bound = incumbent_bound();
            if (bound != SearchCoordination::k_no_incumbent && depth + 1 >= bound)
            {
                ++m_st.incumbent_bound_prunings;
                report(TransitionOutcome::Pruned, true);
                mark_solved(child);
                return true;
            }
            if (register_features(child, depth))
            {
                ++m_st.case1;
                report(TransitionOutcome::Opened, false);
                cur = child;
                continue;
            }
            ++m_st.case2;
            report(TransitionOutcome::Pruned, true);
            mark_solved(child);
            return true;
        }
    }

    const Task& m_task;
    const RolloutIwOptions& m_o;
    WorkspaceLease m_lease;
    Successors& m_succ;
    GoalTest m_goal;
    BlockedSet m_blocked;
    SearchObserver* m_hot;
    SearchCoordination* m_coord;
    Clock::time_point m_deadline{};
    bool m_timed = false;
    Interrupt m_interrupt = Interrupt::None;
    RolloutIwStatistics m_st;

    FlatStateStore m_store;
    std::vector<Node> m_nodes;
    std::vector<u32> m_schema, m_child, m_succ_arity;
    std::vector<u64> m_boff;
    std::vector<ObjectId> m_binding;
    std::vector<u64> m_doff;      // per action: its delta's offset in m_delta (deletes, then adds)
    std::vector<u32> m_dndel;     // per action: its delta's delete count
    std::vector<SlotId> m_delta;
    std::vector<u8> m_act_solved;  // per action: its child exists and is solved
    std::vector<u32> m_act_rank;   // per action: its schema's regression rank (mixed ordering)
    std::vector<u32> m_best;
    bool m_root_solved = false;
    u64 m_steps = 0;  // rollout steps (the clock is read every 64)
    u32 m_goal_node = k_none;

    SplitMix64 m_rng;
    std::vector<CanonicalAtom> m_goal_atoms;  // sorted
    std::vector<u8> m_goal_slot;
    std::vector<std::vector<const formalism::Literal*>> m_ce_adds;
    std::vector<u32> m_schema_rank;

    // scratch
    std::vector<Action> m_buf_actions;
    std::vector<u32> m_buf_schema, m_buf_boff;
    std::vector<ObjectId> m_buf_binding;
    std::vector<u8> m_buf_goal;
    std::vector<SlotId> m_buf_delta;
    std::vector<u32> m_buf_doff;  // per generated action: offset in m_buf_delta, delete count
    std::vector<u32> m_order, m_perm, m_rank, m_args;
    std::vector<u8> m_taken;
    std::vector<u64> m_child_words;
};
}  // namespace

RolloutIwResult run_rollout_iw(const Task& task, const RolloutIwOptions& options, SearchObserver* hot, SearchObserver* root)
{
    // children are rebuilt from atom deltas and the orderings read atom goals only (numeric values are not kept)
    if (task.numeric_slots() > 0)
        throw std::invalid_argument("mymyr: rollout IW does not support tasks with numeric fluents (IW and SIW do)");
    const State start = options.start ? *options.start : task.initial_state();
    if (root)
        root->on_start(start);
    RolloutIwResult r = RolloutSearch(task, options, hot).run();
    if (root)
    {
        if (r.status == SearchStatus::Solved)
            root->on_solution(r.plan, r.cost);
        root->on_end(r.status, r.statistics.statistics());
    }
    return r;
}
}  // namespace detail

RolloutIwResult rollout_iw(const Task& task, const RolloutIwOptions& options)
{
    return detail::run_rollout_iw(task, options, options.control.observer, options.control.observer);
}
}  // namespace mymyr::search
