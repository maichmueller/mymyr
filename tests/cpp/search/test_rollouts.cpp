// Rollout IW(1), parallel IW rollouts and the atomic-goal portfolio (search/rollout_iw.hpp, parallel_rollouts.hpp,
// portfolio.hpp), and the per-worker observer / goal hooks (the make_worker protocol, search/control.hpp):
//   - rollout_iw equals an independent oracle of the documented algorithm counter for counter, for every ordering;
//   - budgets, bounds, blocked states and observer events of rollout_iw;
//   - a parallel rollout batch gives identical results at every thread count, and falls back to the calling thread
//     for observers and goals that are not parallel-safe;
//   - the portfolio: serial reproducibility, the certifier alone equals IW(1), certification rules, hooks and budgets.
// Fork parity of the counts is tested in test_iw_variants_golden.cpp, against tests/data/iw_variants/fork_golden.json.

#include "../support/suite.hpp"
#include "mymyr/core/random.hpp"
#include "mymyr/formalism/task_data.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
constexpr u32 k_none = ~u32{0};
constexpr u32 k_inf = ~u32{0};

std::vector<CanonicalAtom> goal_atoms(const Task& task)
{
    const formalism::TaskData& t = task.data();
    const CanonicalLayout& L = task.atoms().layout();
    std::vector<CanonicalAtom> out;
    for (const formalism::Literal& l : t.literals_of(t.goal))
    {
        if (!l.positive || t.predicate(l.pred).kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> args;
        for (formalism::Term x : t.terms_of(l))
            args.push_back(formalism::term_object(x).v);
        const CanonicalAtom c = L.encode(l.pred.v, args.data());
        if (c < L.fluent_count)
            out.push_back(c);
    }
    return out;
}

/// A single positive goal atom (the portfolio's use case).
GoalSpec atom_goal(const Task& task, CanonicalAtom c)
{
    GoalSpec g;
    g.kind = GoalSpec::Kind::AnyOf;
    g.goals.push_back({{SlotId{task.atoms().intern(c)}}, {}});
    return g;
}

/// The fork's schema regression ranks, written independently (a fixpoint over a predicate map).
std::vector<u32> oracle_schema_ranks(const Task& task, const std::set<u32>* goal_preds)
{
    const formalism::TaskData& t = task.data();
    std::map<u32, u32> pr;
    if (goal_preds)
        for (u32 p : *goal_preds)
            pr[p] = 0;
    else
        for (const formalism::Literal& l : t.literals_of(t.goal))
            if (l.positive && t.predicate(l.pred).kind == formalism::PredKind::Fluent)
                pr[l.pred.v] = 0;
    auto rank_of = [&](u32 p) { return pr.contains(p) ? pr[p] : k_inf; };
    auto best_add = [&](const formalism::Schema& s)
    {
        u32 best = k_inf;
        for (const formalism::ConditionalEffect& ce : t.effects_of(s))
            for (const formalism::Literal& l : formalism::TaskData::slice(t.literals, ce.effects))
                if (l.positive)
                    best = std::min(best, rank_of(l.pred.v));
        return best;
    };
    for (bool changed = true; changed;)
    {
        changed = false;
        for (const formalism::Schema& s : t.schemas)
        {
            const u32 best = best_add(s);
            if (best == k_inf)
                continue;
            for (const formalism::Literal& l : t.literals_of(s.precondition))
                if (t.predicate(l.pred).kind == formalism::PredKind::Fluent && best + 1 < rank_of(l.pred.v))
                {
                    pr[l.pred.v] = best + 1;
                    changed = true;
                }
        }
    }
    std::vector<u32> out;
    for (const formalism::Schema& s : t.schemas)
        out.push_back(best_add(s));
    return out;
}

struct ORollout
{
    SearchStatus status = SearchStatus::Exhausted;
    RolloutIwStatistics st;
    u32 plan_length = 0;
    bool root_solved = false;
};

/// core/random.hpp's documented shuffle, draw by draw: for i = n..2 swap a[i - 1] with a[bounded(i)].
void reference_shuffle(SplitMix64& rng, std::span<u32> a)
{
    for (usize i = a.size(); i > 1; --i)
        std::swap(a[i - 1], a[static_cast<usize>(rng.bounded(i))]);
}

/// Rollout IW(1) as documented in search/rollout_iw.hpp: nodes hold whole States, successors come from
/// Successors::for_each_applicable / apply, the best depths live in a std::map. Shares no code with rollout_iw.cpp.
/// DirectGoalAchieverFirst uses the fired adds (equal to the documented rule on tasks without conditional effects).
/// `atom`: an atomic goal (the portfolio's use case) instead of the task's goal.
ORollout oracle_rollout(const Task& task, ActionOrdering ord, u64 seed, u64 max_states, std::optional<CanonicalAtom> atom = {})
{
    Successors& succ = task.workspace().successors();
    const bool w0 = succ.witness_pruning(), c0 = succ.canonical_order();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);
    std::set<u32> goal_slots;
    for (CanonicalAtom c : atom ? std::vector<CanonicalAtom>{*atom} : goal_atoms(task))
        goal_slots.insert(task.atoms().intern(c));
    std::set<u32> goal_preds;
    if (atom)
    {
        std::vector<u32> args(64);
        goal_preds.insert(task.atoms().layout().decode(*atom, args.data()));
    }
    const std::vector<u32> schema_rank = oracle_schema_ranks(task, atom ? &goal_preds : nullptr);
    auto is_goal = [&](const State& s)
    {
        if (!atom)
            return task.is_goal(s);
        return bits::test(s.data(), s.size_words(), *goal_slots.begin());
    };
    struct N
    {
        State s;
        u32 parent, depth;
        bool mat = false, solved = false;
        std::vector<Action> acts;
        std::vector<u8> goal_adder;
        std::vector<u32> child;
    };
    std::vector<N> t;
    auto node = [](State s, u32 parent, u32 depth)
    {
        N n;
        n.s = std::move(s);
        n.parent = parent;
        n.depth = depth;
        return n;
    };
    std::map<u32, u32> best;
    ORollout r;
    SplitMix64 rng(seed);
    auto reg = [&](u32 v)
    {
        bool imp = false;
        for (SlotId x : t[v].s.slots())
        {
            const auto it = best.find(x.v);
            if (it == best.end() || t[v].depth < it->second)
            {
                best[x.v] = t[v].depth;
                imp = true;
                ++r.st.feature_depth_improvements;
            }
        }
        return imp;
    };
    auto all_solved = [&](u32 v)
    {
        if (!t[v].mat)
            return false;
        for (u32 c : t[v].child)
            if (c == k_none || !t[c].solved)
                return false;
        return true;
    };
    auto mark = [&](u32 v)
    {
        t[v].solved = true;
        for (u32 p = t[v].parent; p != k_none && all_solved(p); p = t[p].parent)
        {
            t[p].solved = true;
            ++r.st.solved_propagations;
        }
    };
    auto finish = [&](SearchStatus s)
    {
        r.status = s;
        r.st.tree_nodes = t.size();
        r.root_solved = t[0].solved;
        succ.set_witness_pruning(w0);
        succ.set_canonical_order(c0);
        return r;
    };
    t.push_back(node(task.initial_state(), k_none, 0));
    reg(0);
    if (is_goal(t[0].s))
        return finish(SearchStatus::Solved);
    while (!t[0].solved)
    {
        ++r.st.rollouts;
        u32 cur = 0;
        for (;;)
        {
            if (!t[cur].mat)
            {
                t[cur].mat = true;
                ++r.st.expanded;
                succ.for_each_applicable(t[cur].s,
                                         [&](const ActionLabel& a, const Delta& d)
                                         {
                                             t[cur].acts.emplace_back(a);
                                             bool g = false;
                                             for (SlotId x : d.add)
                                                 g = g || goal_slots.contains(x.v);
                                             t[cur].goal_adder.push_back(g ? 1 : 0);
                                         });
                t[cur].child.assign(t[cur].acts.size(), k_none);
            }
            const usize n = t[cur].acts.size();
            if (n == 0)
            {
                ++r.st.dead_ends;
                mark(cur);
                break;
            }
            std::vector<u32> order(n);
            std::iota(order.begin(), order.end(), 0u);
            auto by = [&](auto score) { std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) { return score(a) < score(b); }); };
            auto srank = [&](u32 i) { return schema_rank[t[cur].acts[i].schema.v]; };
            switch (ord)
            {
                case ActionOrdering::InOrder: break;
                case ActionOrdering::Randomized: reference_shuffle(rng, std::span<u32>(order)); break;
                case ActionOrdering::DirectGoalAchieverFirst: by([&](u32 i) { return t[cur].goal_adder[i] ? 0 : 1; }); break;
                case ActionOrdering::GoalRegressionRelevance: by(srank); break;
                case ActionOrdering::MixedRegressionRandom:
                    reference_shuffle(rng, std::span<u32>(order));
                    by(srank);
                    break;
            }
            u32 pos = k_none;
            for (u32 p : order)
                if (t[cur].child[p] == k_none || !t[t[cur].child[p]].solved)
                {
                    pos = p;
                    break;
                }
            if (pos == k_none)
            {
                mark(cur);
                break;
            }
            if (const u32 ex = t[cur].child[pos]; ex != k_none)
            {
                bool novel = false;
                for (SlotId x : t[ex].s.slots())
                    novel = novel || best.at(x.v) == t[ex].depth;
                if (novel)
                {
                    ++r.st.case4;
                    cur = ex;
                    continue;
                }
                ++r.st.case3;
                mark(ex);
                break;
            }
            State c = succ.apply(t[cur].s, t[cur].acts[pos].label());
            ++r.st.generated;
            const u32 id = static_cast<u32>(t.size());
            const u32 depth = t[cur].depth + 1;
            t.push_back(node(std::move(c), cur, depth));
            t[cur].child[pos] = id;
            r.st.max_rollout_depth = std::max(r.st.max_rollout_depth, depth);
            if (is_goal(t[id].s))
            {
                r.plan_length = depth;
                return finish(SearchStatus::Solved);
            }
            if (r.st.generated >= max_states)
                return finish(SearchStatus::OutOfStates);
            if (reg(id))
            {
                ++r.st.case1;
                cur = id;
                continue;
            }
            ++r.st.case2;
            mark(id);
            break;
        }
    }
    return finish(SearchStatus::Exhausted);
}

bool reaches_goal(const Task& task, std::span<const Action> plan, const std::function<bool(StateView)>& goal = {})
{
    Successors& succ = task.workspace().successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    return goal ? goal(s) : task.is_goal(s);
}

void expect_same_stats(const RolloutIwStatistics& a, const RolloutIwStatistics& b, const std::string& what)
{
    EXPECT_EQ(a.rollouts, b.rollouts) << what;
    EXPECT_EQ(a.generated, b.generated) << what;
    EXPECT_EQ(a.expanded, b.expanded) << what;
    EXPECT_EQ(a.feature_depth_improvements, b.feature_depth_improvements) << what;
    EXPECT_EQ(a.case1, b.case1) << what;
    EXPECT_EQ(a.case2, b.case2) << what;
    EXPECT_EQ(a.case3, b.case3) << what;
    EXPECT_EQ(a.case4, b.case4) << what;
    EXPECT_EQ(a.solved_propagations, b.solved_propagations) << what;
    EXPECT_EQ(a.dead_ends, b.dead_ends) << what;
    EXPECT_EQ(a.max_rollout_depth, b.max_rollout_depth) << what;
    EXPECT_EQ(a.tree_nodes, b.tree_nodes) << what;
}

const ActionOrdering k_orderings[] = {ActionOrdering::InOrder, ActionOrdering::Randomized, ActionOrdering::DirectGoalAchieverFirst,
                                      ActionOrdering::GoalRegressionRelevance, ActionOrdering::MixedRegressionRandom};

bool has_conditional_effects(const Task& task)
{
    const formalism::TaskData& t = task.data();
    for (const formalism::Schema& s : t.schemas)
        for (const formalism::ConditionalEffect& ce : t.effects_of(s))
            if (ce.condition.literals.count != 0 || ce.extra_params.count != 0)
                return true;
    return false;
}

/// Counts every event, per kind and per transition outcome; make_worker hands out counting children.
class Counter : public SearchObserver
{
public:
    explicit Counter(bool parallel_safe = false) : m_safe(parallel_safe) {}
    std::atomic<u64> starts{0}, expands{0}, generates{0}, prunes{0}, passes{0}, solutions{0}, ends{0};
    std::atomic<u64> outcome[6] = {};
    std::vector<std::shared_ptr<Counter>> workers;
    std::vector<u32> worker_ids;

    void on_start(StateView) override { ++starts; }
    void on_expand(u64, StateView) override { ++expands; }
    void on_generate(u64, const Action&, u64, StateView, bool) override { ++generates; }
    void on_prune(u64, const Action&, StateView) override { ++prunes; }
    void on_pass(u32, const SearchStatistics&) override { ++passes; }
    void on_solution(std::span<const Action>, double) override { ++solutions; }
    void on_end(SearchStatus, const SearchStatistics&) override { ++ends; }
    void on_transition(u64, const Action&, u64, StateView, TransitionOutcome o) override { ++outcome[static_cast<u32>(o)]; }
    std::shared_ptr<SearchObserver> make_worker(u32 w) override
    {
        if (!m_safe)
            return nullptr;
        workers.push_back(std::make_shared<Counter>());
        worker_ids.push_back(w);
        return workers.back();
    }
    [[nodiscard]] u64 transitions() const
    {
        u64 s = 0;
        for (const auto& o : outcome)
            s += o.load();
        return s;
    }
    [[nodiscard]] u64 of(TransitionOutcome o) const { return outcome[static_cast<u32>(o)].load(); }

private:
    bool m_safe;
};
}  // namespace

// ================================================================================================ Rollout IW
TEST(RolloutIw, EqualsTheOracle)
{
    u32 solved = 0, exhausted = 0, compared = 0;
    for (const SuiteTask& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        const bool ce = has_conditional_effects(*task);
        // the task's goal, and its first three positive atoms as atomic goals
        std::vector<std::optional<CanonicalAtom>> goals = {std::nullopt};
        for (CanonicalAtom c : goal_atoms(*task))
            if (goals.size() < 4)
                goals.emplace_back(c);
        for (const std::optional<CanonicalAtom>& atom : goals)
            for (ActionOrdering ord : k_orderings)
            {
                if (ord == ActionOrdering::DirectGoalAchieverFirst && ce)
                    continue;  // the oracle inspects fired adds only
                RolloutIwOptions o;
                o.ordering = ord;
                o.seed = 11;
                o.control.budget.max_states = 3000;
                if (atom)
                    o.control.goal = atom_goal(*task, *atom);
                const RolloutIwResult r = rollout_iw(*task, o);
                const ORollout ref = oracle_rollout(*task, ord, 11, 3000, atom);
                const std::string what = t.name + " " + to_string(ord) + (atom ? " atom " + std::to_string(*atom) : "");
                EXPECT_EQ(r.status, ref.status) << what;
                EXPECT_EQ(r.root_solved, ref.root_solved) << what;
                expect_same_stats(r.statistics, ref.st, what);
                if (r.status == SearchStatus::Solved)
                {
                    EXPECT_EQ(r.plan.size(), ref.plan_length) << what;
                    ASSERT_TRUE(r.goal_state.has_value());
                    if (!atom)
                    {
                        EXPECT_TRUE(reaches_goal(*task, r.plan)) << what;
                        EXPECT_TRUE(task->is_goal(*r.goal_state)) << what;
                    }
                    ++solved;
                }
                exhausted += r.status == SearchStatus::Exhausted;
                ++compared;
            }
    }
    std::printf("rollout_iw vs oracle: %u runs, %u solved, %u exhausted\n", compared, solved, exhausted);
    EXPECT_GT(solved, 100u);
}

TEST(RolloutIw, DeterministicPerSeed)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    for (ActionOrdering ord : {ActionOrdering::Randomized, ActionOrdering::MixedRegressionRandom})
    {
        RolloutIwOptions o;
        o.ordering = ord;
        o.control.budget.max_states = 20000;
        o.seed = 5;
        const RolloutIwResult a = rollout_iw(*task, o), b = rollout_iw(*task, o);
        EXPECT_EQ(a.status, b.status);
        EXPECT_EQ(a.plan, b.plan);
        expect_same_stats(a.statistics, b.statistics, to_string(ord));
        bool differs = false;
        for (u64 seed = 6; seed < 12 && !differs; ++seed)
        {
            o.seed = seed;
            const RolloutIwResult c = rollout_iw(*task, o);
            differs = c.statistics.generated != a.statistics.generated || c.plan != a.plan;
        }
        EXPECT_TRUE(differs) << "the seed matters";
    }
}

TEST(RolloutIw, BudgetsAndBounds)
{
    const auto task = Task::from_text_file(task_path("visitall__visitall_x-6_y-3_r-100"));
    {
        RolloutIwOptions o;
        o.max_rollouts = 1;
        const RolloutIwResult r = rollout_iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Failed);
        EXPECT_EQ(r.message, "rollout budget expired");
        EXPECT_EQ(r.statistics.rollouts, 1u);
    }
    {
        RolloutIwOptions o;
        o.control.budget.max_states = 7;
        const RolloutIwResult r = rollout_iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::OutOfStates);
        EXPECT_EQ(r.statistics.generated, 7u);
    }
    {
        RolloutIwOptions o;
        o.control.cancel.request();
        EXPECT_EQ(rollout_iw(*task, o).status, SearchStatus::Cancelled);
        RolloutIwOptions p;
        p.control.budget.max_seconds = 0;
        EXPECT_EQ(rollout_iw(*task, p).status, SearchStatus::OutOfTime);
    }
    {
        // depth bound: nothing deeper than max_depth is generated, and cut nodes are counted
        RolloutIwOptions o;
        o.control.budget.max_depth = 2;
        const RolloutIwResult r = rollout_iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Exhausted);
        EXPECT_TRUE(r.root_solved);
        EXPECT_LE(r.statistics.max_rollout_depth, 2u);
        EXPECT_GT(r.statistics.depth_bound_prunings, 0u);
    }
    {
        // incumbent bound: no plan of length >= the bound is returned
        RolloutIwOptions free;
        free.control.goal = atom_goal(*task, goal_atoms(*task).back());
        free.ordering = ActionOrdering::Randomized;
        free.seed = 2;
        const RolloutIwResult base = rollout_iw(*task, free);
        ASSERT_EQ(base.status, SearchStatus::Solved);
        RolloutIwOptions o = free;
        o.incumbent_bound = static_cast<u32>(base.plan.size());
        const RolloutIwResult r = rollout_iw(*task, o);
        if (r.status == SearchStatus::Solved)
        {
            EXPECT_LT(r.plan.size(), base.plan.size());
        }
        EXPECT_GT(r.statistics.incumbent_bound_prunings, 0u);
        // the same bound through a coordination, which also counts the expansions
        SearchCoordination coord;
        coord.incumbent_length.store(static_cast<u32>(base.plan.size()));
        RolloutIwOptions c = free;
        c.control.coordination = &coord;
        const RolloutIwResult rc = rollout_iw(*task, c);
        EXPECT_EQ(rc.status, r.status);
        expect_same_stats(rc.statistics, r.statistics, "coordination bound");
        EXPECT_EQ(coord.get_total_expansions(), rc.statistics.expanded);
        SearchCoordination cancelled;
        cancelled.request_cancel();
        c.control.coordination = &cancelled;
        EXPECT_EQ(rollout_iw(*task, c).status, SearchStatus::Cancelled);
    }
    {
        // a goal holding in the start state, and a statically false goal
        RolloutIwOptions o;
        o.control.goal.kind = GoalSpec::Kind::Custom;
        o.control.goal.test = [](StateView) { return true; };
        const RolloutIwResult r = rollout_iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Solved);
        EXPECT_TRUE(r.plan.empty());
        EXPECT_EQ(r.statistics.expanded, 0u);
    }
}

TEST(RolloutIw, BlockedStatesAreNeverEntered)
{
    const auto task = Task::from_text_file(task_path("visitall__visitall_x-6_y-3_r-100"));
    RolloutIwOptions o;
    o.control.goal = atom_goal(*task, goal_atoms(*task).back());
    const RolloutIwResult base = rollout_iw(*task, o);
    ASSERT_EQ(base.status, SearchStatus::Solved);
    ASSERT_TRUE(base.goal_state);
    o.control.blocked_states = {*base.goal_state};
    const RolloutIwResult r = rollout_iw(*task, o);
    EXPECT_GT(r.statistics.blocked, 0u);
    if (r.status == SearchStatus::Solved)
    {
        EXPECT_NE(*r.goal_state, *base.goal_state);
    }
}

TEST(RolloutIw, ObserverEvents)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    for (ActionOrdering ord : k_orderings)
    {
        Counter c;
        RolloutIwOptions o;
        o.ordering = ord;
        o.control.observer = &c;
        o.control.budget.max_states = 5000;
        const RolloutIwResult r = rollout_iw(*task, o);
        const RolloutIwStatistics& s = r.statistics;
        EXPECT_EQ(c.starts, 1u);
        EXPECT_EQ(c.ends, 1u);
        EXPECT_EQ(c.solutions, r.status == SearchStatus::Solved ? 1u : 0u);
        EXPECT_EQ(c.expands, s.expanded);
        EXPECT_EQ(c.generates, s.generated);
        EXPECT_EQ(c.transitions(), s.generated);
        EXPECT_EQ(c.of(TransitionOutcome::Opened), s.case1);
        EXPECT_EQ(c.of(TransitionOutcome::Goal), r.status == SearchStatus::Solved ? 1u : 0u);
        EXPECT_EQ(c.prunes, c.of(TransitionOutcome::Pruned));
        // an observer does not change the search
        RolloutIwOptions q = o;
        q.control.observer = nullptr;
        expect_same_stats(rollout_iw(*task, q).statistics, s, to_string(ord));
    }
}

// ================================================================================================ engine events
TEST(IwVariantEvents, EveryTransitionHasOneOutcome)
{
    const auto task = Task::from_text_file(task_path("depot__p02"));
    Counter c;
    LiwOptions lo;
    lo.max_arity = 2;
    lo.control.observer = &c;
    const IwResult r = liw(*task, lo);
    u64 generated = 0, in_tree = 0;
    for (const IwPassStatistics& p : r.passes)
    {
        generated += p.generated;
        in_tree += p.generated_in_tree;
    }
    EXPECT_EQ(c.transitions(), generated);
    EXPECT_EQ(c.generates, generated);
    EXPECT_EQ(c.of(TransitionOutcome::Opened) + c.of(TransitionOutcome::Duplicate) + c.of(TransitionOutcome::Pruned), generated);
    EXPECT_LE(c.of(TransitionOutcome::Opened), in_tree);
    EXPECT_EQ(c.passes, r.passes.size());
    EXPECT_EQ(c.ends, 1u);

    Counter a;
    AbstractedIwOptions ao;
    ao.control.observer = &a;
    const IwResult ra = abstracted_iw(*task, ao);
    ASSERT_EQ(ra.passes.size(), 1u);
    EXPECT_EQ(a.transitions(), ra.passes[0].generated);
    EXPECT_EQ(a.of(TransitionOutcome::Opened), ra.passes[0].generated_in_tree);
}

// ================================================================================================ parallel rollouts
namespace
{
void expect_same_rollout(const RolloutResult& a, const RolloutResult& b, const std::string& what)
{
    EXPECT_EQ(a.search.status, b.search.status) << what;
    EXPECT_EQ(a.search.plan, b.search.plan) << what;
    ASSERT_EQ(a.search.passes.size(), b.search.passes.size()) << what;
    for (usize i = 0; i < a.search.passes.size(); ++i)
    {
        EXPECT_EQ(a.search.passes[i].expanded, b.search.passes[i].expanded) << what;
        EXPECT_EQ(a.search.passes[i].generated, b.search.passes[i].generated) << what;
        EXPECT_EQ(a.search.passes[i].generated_in_tree, b.search.passes[i].generated_in_tree) << what;
    }
    EXPECT_EQ(a.num_states, b.num_states) << what;
    EXPECT_EQ(a.reached_fluent_atoms, b.reached_fluent_atoms) << what;
    EXPECT_EQ(a.reached_derived_atoms, b.reached_derived_atoms) << what;
    ASSERT_EQ(a.landing_states.size(), b.landing_states.size()) << what;
    for (usize i = 0; i < a.landing_states.size(); ++i)
    {
        EXPECT_EQ(a.landing_states[i].atoms, b.landing_states[i].atoms) << what;
        EXPECT_EQ(a.landing_states[i].direct_dead_end, b.landing_states[i].direct_dead_end) << what;
    }
    EXPECT_EQ(a.landing_state_by_atom, b.landing_state_by_atom) << what;
    EXPECT_EQ(a.co_occurrence, b.co_occurrence) << what;
}

ParallelRolloutOptions rollout_batch(u32 n, u32 threads)
{
    ParallelRolloutOptions o;
    o.iw.max_arity = 1;
    o.num_threads = threads;
    o.report_landing_states = true;
    o.report_co_occurrence = true;
    for (u32 i = 0; i < n; ++i)
        o.seeds.push_back(1000 + 17 * i);
    return o;
}
}  // namespace

TEST(ParallelRollouts, SameResultAtEveryThreadCount)
{
    struct Case
    {
        const char* name;
        u32 arity;
        bool atomic;  // the last goal atom instead of the (wide) task goal
    };
    const Case cases[] = {{"gripper__prob05", 1, false},      {"depot__p02", 2, false},           {"depot__p02", 2, true},
                          {"philosophers__p03-phil4", 2, false}, {"miconic-simpleadl__s10-2", 1, true}, {"visitall__visitall_x-6_y-3_r-100", 2, true},
                          {"blocks__probBLOCKS-8-0", 1, true}};
    u32 solved = 0, runs = 0;
    for (const auto& [name, arity, atomic] : cases)
    {
        // a fresh task per thread count: lazy slots are assigned in a different order each time
        const auto t1 = Task::from_text_file(task_path(name));
        const std::vector<CanonicalAtom> goals = goal_atoms(*t1);
        ASSERT_TRUE(!atomic || !goals.empty());
        const CanonicalAtom atom = goals.empty() ? CanonicalAtom{} : goals.back();
        ParallelRolloutOptions o = rollout_batch(8, 1);
        o.max_next_layer_states = arity == 1 ? 64 : ~u32{0};
        o.iw.max_arity = arity;
        if (atomic)
            o.iw.control.goal = atom_goal(*t1, atom);
        const ParallelRolloutsResult ref = find_rollouts_parallel(*t1, o);
        ASSERT_EQ(ref.threads_used, 1u);
        for (u32 threads : {3u, 8u})
        {
            const auto tn = Task::from_text_file(task_path(name));
            o.num_threads = threads;
            if (atomic)
                o.iw.control.goal = atom_goal(*tn, atom);
            const ParallelRolloutsResult r = find_rollouts_parallel(*tn, o);
            EXPECT_EQ(r.threads_used, threads);
            ASSERT_EQ(r.rollouts.size(), ref.rollouts.size());
            const u32 slot = tn->atoms().intern(atom);
            for (usize k = 0; k < r.rollouts.size(); ++k)
            {
                // plans are compared by content: the same actions in both tasks
                expect_same_rollout(r.rollouts[k], ref.rollouts[k], std::string(name) + " rollout " + std::to_string(k) + " threads " + std::to_string(threads));
                if (r.rollouts[k].search.status == SearchStatus::Solved)
                {
                    ++solved;
                    if (atomic)
                        EXPECT_TRUE(reaches_goal(*tn, r.rollouts[k].search.plan, [slot](StateView s) { return bits::test(s.w, s.nw, slot); }));
                    else
                        EXPECT_TRUE(reaches_goal(*tn, r.rollouts[k].search.plan));
                }
                ++runs;
            }
        }
    }
    std::printf("parallel rollouts: %u runs, %u solved\n", runs, solved);
    EXPECT_GE(solved, 24u);
}

TEST(ParallelRollouts, SixtyFourThreads)
{
    // the TSan target: 64 rollouts on 64 threads over one task, with per-worker observers
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    Counter root(true);
    ParallelRolloutOptions o = rollout_batch(64, 64);
    o.iw.control.observer = &root;
    const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
    EXPECT_EQ(r.threads_used, 64u);
    ASSERT_EQ(root.workers.size(), 64u);
    u64 expanded = 0, events = 0;
    for (const RolloutResult& x : r.rollouts)
        expanded += x.search.total.expanded;
    for (const auto& w : root.workers)
        events += w->expands;
    EXPECT_EQ(events, expanded);
    EXPECT_EQ(root.expands, 0u) << "hot events go to the worker observers";
    EXPECT_EQ(root.starts, 1u);
    EXPECT_EQ(root.ends, 1u);
    const auto serial_task = Task::from_text_file(task_path("gripper__prob05"));
    const ParallelRolloutsResult s = find_rollouts_parallel(*serial_task, rollout_batch(64, 1));
    for (usize k = 0; k < r.rollouts.size(); ++k)
        expect_same_rollout(r.rollouts[k], s.rollouts[k], "rollout " + std::to_string(k));

    // IW(2) ladders on a fresh task: 64 threads assign lazy atom slots concurrently
    const auto fresh = Task::from_text_file(task_path("depot__p02"));
    ParallelRolloutOptions o2 = rollout_batch(64, 64);
    o2.iw.max_arity = 2;
    const ParallelRolloutsResult r2 = find_rollouts_parallel(*fresh, o2);
    EXPECT_EQ(r2.threads_used, 64u);
    const auto serial2 = Task::from_text_file(task_path("depot__p02"));
    o2.num_threads = 1;
    const ParallelRolloutsResult s2 = find_rollouts_parallel(*serial2, o2);
    for (usize k = 0; k < r2.rollouts.size(); ++k)
        expect_same_rollout(r2.rollouts[k], s2.rollouts[k], "IW(2) rollout " + std::to_string(k));
}

TEST(ParallelRollouts, RandomizedLayersFollowTheSeed)
{
    const auto task = Task::from_text_file(task_path("visitall__visitall_x-6_y-3_r-100"));
    ParallelRolloutOptions o = rollout_batch(6, 2);
    const CanonicalAtom far = goal_atoms(*task).back();
    const u32 far_slot = task->atoms().intern(far);
    o.iw.control.goal = atom_goal(*task, far);
    const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
    IwOptions io = o.iw;
    const IwResult plain = iw(*task, io);
    ASSERT_EQ(plain.status, SearchStatus::Solved);
    std::set<std::vector<Action>> plans;
    for (const RolloutResult& x : r.rollouts)
    {
        // a width-1 atom: every layer order finds a plan of the same (shortest) length
        ASSERT_EQ(x.search.status, SearchStatus::Solved);
        EXPECT_EQ(x.search.plan.size(), plain.plan.size());
        EXPECT_TRUE(reaches_goal(*task, x.search.plan, [&](StateView s) { return bits::test(s.w, s.nw, far_slot); }));
        plans.insert(x.search.plan);
        // tracker invariants
        EXPECT_GE(x.num_states, x.landing_states.size());
        EXPECT_EQ(x.landing_state_by_atom.size(), x.reached_fluent_atoms.size());
        for (const auto& [atom, idx] : x.landing_state_by_atom)
        {
            ASSERT_LT(idx, x.landing_states.size());
            EXPECT_TRUE(std::binary_search(x.landing_states[idx].atoms.begin(), x.landing_states[idx].atoms.end(), atom));
        }
        for (const auto& [atom, row] : x.co_occurrence)
        {
            EXPECT_TRUE(std::binary_search(row.begin(), row.end(), atom));
            EXPECT_TRUE(std::includes(x.reached_fluent_atoms.begin(), x.reached_fluent_atoms.end(), row.begin(), row.end()));
        }
    }
    EXPECT_GT(plans.size(), 1u) << "different seeds, different layer orders";
    const auto inter = intersect_co_occurrence(r.rollouts);
    for (const auto& [atom, row] : inter)
        for (const RolloutResult& x : r.rollouts)
        {
            const auto it = std::find_if(x.co_occurrence.begin(), x.co_occurrence.end(), [&](const auto& e) { return e.first == atom; });
            if (it == x.co_occurrence.end())
            {
                EXPECT_TRUE(row.empty()) << "an atom this rollout never reached clears the row";
                continue;
            }
            EXPECT_TRUE(std::includes(it->second.begin(), it->second.end(), row.begin(), row.end()));
        }
    const MergedLandingStates m = merge_landing_states(r.rollouts);
    ASSERT_EQ(m.by_rollout.size(), r.rollouts.size());
    for (usize k = 0; k < r.rollouts.size(); ++k)
    {
        ASSERT_EQ(m.by_rollout[k].size(), r.rollouts[k].landing_states.size());
        for (usize i = 0; i < m.by_rollout[k].size(); ++i)
            EXPECT_EQ(m.states[m.by_rollout[k][i]], r.rollouts[k].landing_states[i].state);
    }
    EXPECT_EQ(std::unordered_set<State>(m.states.begin(), m.states.end()).size(), m.states.size());
}

TEST(ParallelRollouts, NotParallelSafeRunsOnTheCallingThread)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    {
        Counter root(false);
        ParallelRolloutOptions o = rollout_batch(4, 4);
        o.iw.control.observer = &root;
        const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
        EXPECT_EQ(r.threads_used, 1u);
        EXPECT_NE(r.message.find("not parallel-safe"), std::string::npos);
        u64 expanded = 0;
        for (const RolloutResult& x : r.rollouts)
            expanded += x.search.total.expanded;
        EXPECT_EQ(root.expands, expanded) << "every event on the root observer";
    }
    const std::thread::id caller = std::this_thread::get_id();
    {
        // a custom goal without make_worker: serial, and called on the calling thread only
        ParallelRolloutOptions o = rollout_batch(4, 4);
        std::atomic<bool> foreign{false};
        o.iw.control.goal.kind = GoalSpec::Kind::Custom;
        o.iw.control.goal.test = [&](StateView s)
        {
            foreign = foreign || std::this_thread::get_id() != caller;
            return task->is_goal(s);
        };
        const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
        EXPECT_EQ(r.threads_used, 1u);
        EXPECT_FALSE(foreign);
        // with make_worker: parallel, and the same results as the task's goal
        std::mutex m;
        std::set<u32> made;
        o.iw.control.goal.make_worker = [&](u32 w) -> std::function<bool(StateView)>
        {
            std::lock_guard lock(m);
            made.insert(w);
            return [&task](StateView s) { return task->is_goal(s); };
        };
        const ParallelRolloutsResult p = find_rollouts_parallel(*task, o);
        EXPECT_EQ(p.threads_used, 4u);
        EXPECT_EQ(made, (std::set<u32>{0, 1, 2, 3}));
        const ParallelRolloutsResult g = find_rollouts_parallel(*task, rollout_batch(4, 4));
        for (usize k = 0; k < 4; ++k)
            expect_same_rollout(p.rollouts[k], g.rollouts[k], "custom goal rollout " + std::to_string(k));
    }
    {
        SearchCoordination coord;
        ParallelRolloutOptions o = rollout_batch(2, 2);
        o.iw.control.coordination = &coord;
        const ParallelRolloutsResult r = find_rollouts_parallel(*task, o);
        EXPECT_EQ(r.rollouts[0].search.status, SearchStatus::Failed);
        EXPECT_FALSE(r.message.empty());
    }
}

// ================================================================================================ portfolio
namespace
{
PortfolioOptions portfolio(u32 workers, u32 threads)
{
    PortfolioOptions o;
    o.num_rollout_workers = workers;
    o.num_threads = threads;
    o.base_seed = 3;
    return o;
}

}  // namespace

TEST(Portfolio, CertifierAloneEqualsIw1)
{
    for (const char* name : {"gripper__prob05", "depot__p02", "visitall__visitall_x-6_y-3_r-100", "philosophers__p03-phil4"})
    {
        const auto task = Task::from_text_file(task_path(name));
        IwOptions io;
        io.max_arity = 1;
        const IwResult ref = iw(*task, io);
        const PortfolioResult r = atomic_goal_portfolio(*task, portfolio(0, 1));
        ASSERT_TRUE(r.certifier_ran);
        EXPECT_EQ(r.certifier_status, ref.status) << name;
        ASSERT_EQ(r.certifier.passes.size(), ref.passes.size()) << name;
        for (usize i = 0; i < ref.passes.size(); ++i)
        {
            EXPECT_EQ(r.certifier.passes[i].expanded, ref.passes[i].expanded) << name;
            EXPECT_EQ(r.certifier.passes[i].generated, ref.passes[i].generated) << name;
        }
        if (ref.status == SearchStatus::Solved)
        {
            EXPECT_EQ(r.status, SearchStatus::Solved) << name;
            EXPECT_EQ(r.plan, ref.plan) << name;
            EXPECT_EQ(r.plan_length, ref.plan.size());
            EXPECT_EQ(r.cost, ref.cost);
            EXPECT_TRUE(r.certified_optimal);
            EXPECT_EQ(r.winning_worker, 0u);
            EXPECT_EQ(r.total_expansions, ref.total.expanded) << name;
        }
        else
            EXPECT_EQ(r.status, SearchStatus::Exhausted) << name;
    }
}

TEST(Portfolio, SerialIsReproducibleAndCertifiesAtomicGoals)
{
    const auto task = Task::from_text_file(task_path("visitall__visitall_x-6_y-3_r-100"));
    u32 by_rollouts = 0;
    for (CanonicalAtom c : goal_atoms(*task))
    {
        PortfolioOptions o = portfolio(4, 1);
        o.control.goal = atom_goal(*task, c);
        const PortfolioResult a = atomic_goal_portfolio(*task, o);
        const PortfolioResult b = atomic_goal_portfolio(*task, o);
        ASSERT_EQ(a.status, SearchStatus::Solved);
        EXPECT_EQ(a.plan, b.plan);
        EXPECT_EQ(a.winning_worker, b.winning_worker);
        EXPECT_EQ(a.rollout_rounds, b.rollout_rounds);
        EXPECT_EQ(a.total_expansions, b.total_expansions);
        const u32 slot = task->atoms().intern(c);
        EXPECT_TRUE(reaches_goal(*task, a.plan, [slot](StateView s) { return bits::test(s.w, s.nw, slot); }));
        // serial: the certifier runs first and, having found the plan, leaves nothing to the rollouts
        EXPECT_EQ(a.winning_worker, 0u);
        EXPECT_TRUE(a.certified_optimal);
        EXPECT_EQ(a.rollout_rounds, std::vector<u32>(4, 0));
        // the plan is shortest (unit costs, blind A*)
        BestFirstOptions bo;
        bo.control.goal = o.control.goal;
        bo.heuristic.kind = heuristics::Kind::Blind;
        const BestFirstResult opt = astar_eager(*task, bo);
        ASSERT_EQ(opt.status, SearchStatus::Solved);
        EXPECT_EQ(a.plan_length, opt.plan.size());
        by_rollouts += a.winning_worker != 0;
    }
    EXPECT_EQ(by_rollouts, 0u);
}

TEST(Portfolio, RolloutWorkersAndCertification)
{
    // a conjunctive goal of width > 1 in parallel: whoever wins, the plan is valid and the certificate honest
    for (const char* name : {"gripper__prob05", "blocks__probBLOCKS-8-0", "miconic__s7-4"})
    {
        const auto task = Task::from_text_file(task_path(name));
        const PortfolioResult r = atomic_goal_portfolio(*task, portfolio(4, 5));
        EXPECT_EQ(r.threads_used, 5u);
        if (r.status != SearchStatus::Solved)
            continue;
        EXPECT_TRUE(reaches_goal(*task, r.plan)) << name;
        EXPECT_EQ(r.plan.size(), r.plan_length);
        if (r.certified_optimal && r.winning_worker != 0)
        {
            EXPECT_GE(r.iw_lower_bound, r.plan_length) << name;
        }
        if (r.winning_worker == 0)
        {
            EXPECT_TRUE(r.certified_optimal);
        }
        EXPECT_GE(r.total_expansions, r.certifier.total.expanded);
    }
    // rollouts alone find a plan the exhausted certifier does not: not certified
    {
        const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
        IwOptions io;
        io.max_arity = 1;
        if (iw(*task, io).status == SearchStatus::Exhausted)
        {
            PortfolioOptions o = portfolio(4, 1);
            o.rollout_orderings = {{ActionOrdering::InOrder, 0}, {ActionOrdering::Randomized, 9}};
            const PortfolioResult r = atomic_goal_portfolio(*task, o);
            EXPECT_EQ(r.certifier_status, SearchStatus::Exhausted);
            if (r.status == SearchStatus::Solved)
            {
                EXPECT_NE(r.winning_worker, 0u);
                EXPECT_FALSE(r.certified_optimal);
                EXPECT_NE(r.message.find("does not apply"), std::string::npos);
                EXPECT_TRUE(reaches_goal(*task, r.plan));
            }
            else
                EXPECT_EQ(r.status, SearchStatus::Exhausted);
        }
    }
}

TEST(Portfolio, HooksBudgetsAndErrors)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    {
        Counter root(true);
        PortfolioOptions o = portfolio(3, 4);
        o.control.observer = &root;
        const PortfolioResult r = atomic_goal_portfolio(*task, o);
        EXPECT_EQ(r.threads_used, 4u);
        EXPECT_EQ(root.worker_ids, (std::vector<u32>{0, 1, 2, 3}));
        EXPECT_EQ(root.starts, 1u);
        EXPECT_EQ(root.ends, 1u);
        EXPECT_EQ(root.expands, 0u);
        EXPECT_EQ(root.workers[0]->expands, r.certifier.total.expanded);
        for (u32 k = 1; k < 4; ++k)
            EXPECT_EQ(root.workers[k]->expands, r.rollout_statistics[k - 1].expanded);
    }
    {
        Counter root(false);
        PortfolioOptions o = portfolio(3, 4);
        o.control.observer = &root;
        const PortfolioResult r = atomic_goal_portfolio(*task, o);
        EXPECT_EQ(r.threads_used, 1u);
        EXPECT_NE(r.message.find("not parallel-safe"), std::string::npos);
        EXPECT_EQ(root.expands, r.certifier.total.expanded + r.rollout_statistics[0].expanded + r.rollout_statistics[1].expanded +
                                    r.rollout_statistics[2].expanded);
    }
    {
        PortfolioOptions o = portfolio(3, 4);
        o.control.budget.max_expanded = 10;
        const PortfolioResult r = atomic_goal_portfolio(*task, o);
        EXPECT_NE(r.status, SearchStatus::Solved);
        EXPECT_GE(r.total_expansions, 10u);
        EXPECT_LE(r.total_expansions, 10u + 4u) << "every worker stops at its next expansion";
    }
    {
        SearchCoordination coord;
        PortfolioOptions o = portfolio(1, 1);
        o.control.coordination = &coord;
        EXPECT_EQ(atomic_goal_portfolio(*task, o).status, SearchStatus::Failed);
    }
    {
        PortfolioOptions o = portfolio(2, 1);
        o.control.cancel.request();
        const PortfolioResult r = atomic_goal_portfolio(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Cancelled);
        EXPECT_FALSE(r.certifier_ran);
    }
}

// ================================================================================================ core/random
TEST(Random, BatchedShuffleEqualsFisherYates)
{
    // the batched shuffle equals the draw-by-draw definition, permutation and generator state, at every size
    for (u64 seed : {0ull, 1ull, 7ull, 0x9e3779b97f4a7c15ull, ~0ull})
        for (usize n = 0; n < 300; n += (n < 140 ? 1 : 7))
        {
            SplitMix64 a(seed + n), b(seed + n);
            std::vector<u32> x(n), y(n);
            std::iota(x.begin(), x.end(), 0u);
            std::iota(y.begin(), y.end(), 0u);
            a.shuffle(std::span<u32>(x));
            reference_shuffle(b, std::span<u32>(y));
            ASSERT_EQ(x, y) << "seed " << seed << " n " << n;
            ASSERT_EQ(a.state(), b.state()) << "seed " << seed << " n " << n;
            a.shuffle(std::span<u32>(x));  // continue from the same state
            reference_shuffle(b, std::span<u32>(y));
            ASSERT_EQ(x, y);
        }
    // pinned values (the fork's parity tool re-implements the same stream)
    SplitMix64 r(42);
    std::vector<u32> v(10);
    std::iota(v.begin(), v.end(), 0u);
    r.shuffle(std::span<u32>(v));
    SplitMix64 ref(42);
    std::vector<u32> w(10);
    std::iota(w.begin(), w.end(), 0u);
    for (usize i = w.size(); i > 1; --i)
    {
        const u64 x = ref.next();
        const u64 j = static_cast<u64>((static_cast<u128>(x) * i) >> 64);  // no rejection for these draws
        std::swap(w[i - 1], w[j]);
    }
    EXPECT_EQ(v, w);
}

TEST(Random, PinnedValues)
{
    // the stream, the bounded draws and the shuffle are specified bit for bit, whatever the standard library.
    // The values come from an independent Python implementation of core/random.hpp's three definitions; the first
    // three outputs for seed 0 are the published splitmix64 reference values.
    SplitMix64 a(0);
    EXPECT_EQ(a.next(), 0xe220a8397b1dcdafULL);
    EXPECT_EQ(a.next(), 0x6e789e6aa1b965f4ULL);
    EXPECT_EQ(a.next(), 0x06c45d188009454fULL);

    SplitMix64 b(0x0123456789abcdefULL);
    std::vector<u64> draws;
    for (int i = 0; i < 5; ++i)
        draws.push_back(b.bounded(1000));
    EXPECT_EQ(draws, (std::vector<u64>{83, 833, 185, 636, 4}));
    EXPECT_EQ(b.state(), 0x1838a60706203a58ULL);

    SplitMix64 c(42);
    std::vector<u32> v(20);
    std::iota(v.begin(), v.end(), 0u);
    c.shuffle(std::span<u32>(v));
    EXPECT_EQ(v, (std::vector<u32>{9, 16, 7, 15, 1, 12, 19, 8, 11, 2, 6, 4, 10, 18, 13, 0, 17, 5, 3, 14}));
    EXPECT_EQ(c.state(), 0xbe1e08c4728735b9ULL);

    SplitMix64 d(7);  // more than one batch of 64 draws
    std::vector<u32> w(200);
    std::iota(w.begin(), w.end(), 0u);
    d.shuffle(std::span<u32>(w));
    EXPECT_EQ(std::vector<u32>(w.begin(), w.begin() + 12), (std::vector<u32>{140, 107, 62, 22, 39, 170, 151, 20, 30, 149, 191, 65}));
    EXPECT_EQ(d.state(), 0xfd1f9f31f2e6745aULL);
}

TEST(Random, XoshiroPinnedValues)
{
    // xoshiro256** seeded by splitmix64 (the dataset samplers' datasets::Rng); the values come from an independent
    // Python implementation. The state after seeding is the first four splitmix64 outputs (see PinnedValues).
    Xoshiro256 a(0);
    EXPECT_EQ(a.next(), 0x99ec5f36cb75f2b4ULL);
    EXPECT_EQ(a.next(), 0xbf6e1f784956452aULL);
    EXPECT_EQ(a.next(), 0x1a5f849d4933e6e0ULL);

    Xoshiro256 b(0x0123456789abcdefULL);
    std::vector<u64> draws;
    for (int i = 0; i < 5; ++i)
        draws.push_back(b.bounded(1000));
    EXPECT_EQ(draws, (std::vector<u64>{635, 23, 384, 107, 871}));

    Xoshiro256 c(7);
    EXPECT_EQ(c.uniform(), 0.7005764821796896);
    EXPECT_EQ(c.uniform(), 0.2787512294737843);

    Xoshiro256 d(3), e(99);  // reseed restarts the stream
    d.next();
    d.reseed(99);
    EXPECT_EQ(d.next(), e.next());
    EXPECT_EQ(d.bounded(17), e.bounded(17));
}
