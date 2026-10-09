// Heuristic best-first searches (search/best_first.hpp) on small suite tasks (unit costs: the text tasks carry no
// total-cost): A* is optimal (the BrFS plan length) with blind and h_max, eager and lazy, on every store and queue, and
// the stores and queues do not change the search (identical counts: the queues share one total order); GBFS and beam
// return valid plans; SearchControl budgets, cancellation, observer events, blocked states and goal specs. Also
// heuristic unit tests on the same tasks (the golden comparison with the fork is mymyr_heuristics_golden_tests).

#include "../support/suite.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
// Small enough for the sanitizer builds: exhaustive state spaces of 40k-100k states (suite.hpp).
const char* const k_tasks[] = {"depot__p02", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "pegsol-08-strips__p22"};

std::shared_ptr<const Task> load(const std::string& name) { return Task::from_text_file(task_path(name)); }

/// Length of a shortest plan (BrFS).
size_t optimal_length(const Task& task)
{
    BrfsOptions o;
    o.stop_at_goal = true;
    const BrfsResult r = brfs(task, o);
    EXPECT_TRUE(r.solved);
    return r.plan.size();
}

/// The plan is applicable from `start` and ends in a state satisfying `goal` (default: the task's goal).
bool valid_plan(const Task& task, const std::vector<Action>& plan, const State& start, State* end = nullptr)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = start;
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    if (end)
        *end = s;
    return end ? true : succ.is_goal(s);
}

BestFirstOptions with_h(heuristics::Kind k)
{
    BestFirstOptions o;
    o.heuristic.kind = k;
    return o;
}

using SearchFn = BestFirstResult (*)(const Task&, const BestFirstOptions&);
struct Algo
{
    const char* name;
    SearchFn run;
};
const Algo k_astar[] = {{"astar_eager", &astar_eager}, {"astar_lazy", &astar_lazy}};
const Algo k_all[] = {{"astar_eager", &astar_eager}, {"astar_lazy", &astar_lazy}, {"gbfs_eager", &gbfs_eager},
                      {"gbfs_lazy", &gbfs_lazy}, {"beam", &beam}};
}  // namespace

TEST(BestFirst, AStarIsOptimalOnEveryStoreAndQueue)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const size_t L = optimal_length(*task);
        for (const Algo& a : k_astar)
        {
            SCOPED_TRACE(a.name);
            u64 first = 0;  // every store and queue sees the same states in the same order
            for (const auto store : {BestFirstOptions::Store::Flat, BestFirstOptions::Store::Chunked, BestFirstOptions::Store::Compact})
                for (const auto queue : {BestFirstOptions::Queue::Bucket, BestFirstOptions::Queue::Heap})
                {
                    BestFirstOptions o = with_h(heuristics::Kind::Max);
                    o.store = store;
                    o.queue = queue;
                    const BestFirstResult r = a.run(*task, o);
                    SCOPED_TRACE(r.store + "/" + r.queue);
                    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                    EXPECT_EQ(r.plan.size(), L);
                    EXPECT_EQ(r.cost, static_cast<double>(L));
                    EXPECT_TRUE(valid_plan(*task, r.plan, task->initial_state()));
                    ASSERT_TRUE(r.goal_state.has_value());
                    EXPECT_TRUE(task->is_goal(*r.goal_state));
                    EXPECT_GT(r.stats.expanded, 0u);
                    EXPECT_GE(r.stats.states, r.stats.expanded);
                    if (first == 0)
                        first = r.stats.expanded;
                    EXPECT_EQ(r.stats.expanded, first);
                }
        }
    }
}

TEST(BestFirst, AStarIsOptimalWithBlindAndHmax)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const size_t L = optimal_length(*task);
        for (const auto k : {heuristics::Kind::Blind, heuristics::Kind::Max})
            for (const Algo& a : k_astar)
                for (const bool requeue : {true, false})
                {
                    if (!requeue && a.run != &astar_lazy)
                        continue;
                    SCOPED_TRACE(std::string(a.name) + "/" + heuristics::to_string(k) + (requeue ? "" : "/fork-keys"));
                    BestFirstOptions o = with_h(k);
                    o.lazy_requeue = requeue;
                    const BestFirstResult r = a.run(*task, o);
                    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                    if (requeue)  // the fork's lazy keys give no guarantee
                    {
                        EXPECT_EQ(r.plan.size(), L);
                    }
                    EXPECT_EQ(r.cost, static_cast<double>(r.plan.size()));
                    EXPECT_TRUE(valid_plan(*task, r.plan, task->initial_state()));
                    if (k == heuristics::Kind::Max)
                    {
                        EXPECT_LE(r.initial_h, static_cast<double>(L));
                    }
                }
    }
}

TEST(BestFirst, GreedyAndBeamReturnValidPlans)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        for (const auto k : {heuristics::Kind::FF, heuristics::Kind::Add, heuristics::Kind::GoalCount})
            for (const Algo& a : k_all)
            {
                if (a.run == &astar_eager || a.run == &astar_lazy)
                    continue;
                SCOPED_TRACE(std::string(a.name) + "/" + heuristics::to_string(k));
                BestFirstOptions o = with_h(k);
                o.beam_width = 64;
                const BestFirstResult r = a.run(*task, o);
                if (a.run == &beam && r.status == SearchStatus::Exhausted)
                    continue;  // beam search is incomplete
                ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
                EXPECT_EQ(r.cost, static_cast<double>(r.plan.size()));
                EXPECT_TRUE(valid_plan(*task, r.plan, task->initial_state()));
                // deterministic: a second run is identical
                const BestFirstResult r2 = a.run(*task, o);
                EXPECT_EQ(r2.plan, r.plan);
                EXPECT_EQ(r2.stats.expanded, r.stats.expanded);
                EXPECT_EQ(r2.stats.generated, r.stats.generated);
            }
    }
}

TEST(BestFirst, Budgets)
{
    const auto task = load("depot__p02");
    for (const Algo& a : k_all)
    {
        SCOPED_TRACE(a.name);
        {
            BestFirstOptions o = with_h(heuristics::Kind::Blind);
            o.control.budget.max_expanded = 10;
            const BestFirstResult r = a.run(*task, o);
            EXPECT_EQ(r.status, SearchStatus::OutOfStates);
            EXPECT_LE(r.stats.expanded, 10u);
            EXPECT_TRUE(r.plan.empty());
        }
        {
            BestFirstOptions o = with_h(heuristics::Kind::Blind);
            o.control.budget.max_states = 50;
            const BestFirstResult r = a.run(*task, o);
            EXPECT_EQ(r.status, SearchStatus::OutOfStates);
            EXPECT_EQ(r.stats.states, 51u);  // the state that exceeded the budget is stored
        }
        {
            BestFirstOptions o = with_h(heuristics::Kind::Blind);
            o.control.budget.max_seconds = 0;
            EXPECT_EQ(a.run(*task, o).status, SearchStatus::OutOfTime);
        }
        {
            BestFirstOptions o = with_h(heuristics::Kind::Blind);
            o.control.cancel.request();
            EXPECT_EQ(a.run(*task, o).status, SearchStatus::Cancelled);
        }
        {
            // depth budget below the optimal plan length: A* and GBFS exhaust the bounded space
            BestFirstOptions o = with_h(heuristics::Kind::Blind);
            o.control.budget.max_depth = 2;
            const BestFirstResult r = a.run(*task, o);
            EXPECT_EQ(r.status, SearchStatus::Exhausted);
            EXPECT_GT(r.stats.pruned, 0u);
        }
    }
}

namespace
{
class Recorder : public SearchObserver
{
public:
    u64 starts = 0, expands = 0, generates = 0, fresh = 0, prunes = 0, solutions = 0, ends = 0, progress = 0;
    u64 stop_after = ~u64{0};
    SearchStatus end_status = SearchStatus::Failed;
    double solution_cost = -1;
    void on_start(StateView) override { ++starts; }
    void on_expand(u64, StateView) override { ++expands; }
    void on_generate(u64, const Action&, u64, StateView child, bool is_new) override
    {
        ++generates;
        fresh += is_new;
        EXPECT_GT(child.nw, 0u);
    }
    void on_prune(u64, const Action&, StateView) override { ++prunes; }
    void on_solution(std::span<const Action>, double cost) override
    {
        ++solutions;
        solution_cost = cost;
    }
    bool on_progress(const SearchStatistics&) override { return ++progress < stop_after; }
    void on_end(SearchStatus s, const SearchStatistics&) override
    {
        ++ends;
        end_status = s;
    }
};
}  // namespace

TEST(BestFirst, ObserverEvents)
{
    const auto task = load("depot__p02");
    for (const Algo& a : k_all)
    {
        SCOPED_TRACE(a.name);
        Recorder rec;
        BestFirstOptions o = with_h(heuristics::Kind::FF);
        o.control.observer = &rec;
        o.control.progress_interval = 16;
        const BestFirstResult r = a.run(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved);
        EXPECT_EQ(rec.starts, 1u);
        EXPECT_EQ(rec.expands, r.stats.expanded);
        // GBFS and beam stop at a goal found while generating: the transitions after it are not reported
        if (a.run == &astar_eager || a.run == &astar_lazy)
        {
            EXPECT_EQ(rec.generates, r.stats.generated);  // no blocked states
            EXPECT_EQ(rec.fresh + 1, r.stats.states);
        }
        else
        {
            EXPECT_LE(rec.generates, r.stats.generated);
            EXPECT_LE(rec.fresh + 1, r.stats.states);
            EXPECT_GT(rec.fresh, 0u);
        }
        EXPECT_EQ(rec.prunes, 0u);
        EXPECT_EQ(rec.solutions, 1u);
        EXPECT_EQ(rec.solution_cost, r.cost);
        EXPECT_EQ(rec.ends, 1u);
        EXPECT_EQ(rec.end_status, SearchStatus::Solved);

        // on_progress returning false cancels
        Recorder stop;
        stop.stop_after = 1;
        BestFirstOptions o2 = with_h(heuristics::Kind::Blind);
        o2.control.observer = &stop;
        o2.control.progress_interval = 4;
        const BestFirstResult r2 = a.run(*task, o2);
        EXPECT_EQ(r2.status, SearchStatus::Cancelled);
        EXPECT_EQ(stop.end_status, SearchStatus::Cancelled);
    }
}

TEST(BestFirst, BlockedStatesAreNeverEntered)
{
    const auto task = load("depot__p02");
    const BestFirstResult base = astar_eager(*task, with_h(heuristics::Kind::Max));
    ASSERT_EQ(base.status, SearchStatus::Solved);
    ASSERT_GE(base.plan.size(), 2u);
    // block the state after the first action of the optimal plan (and the start state, which stays exempt)
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s1 = succ.apply(task->initial_state(), base.plan[0].label());
    for (const Algo& a : k_all)
    {
        SCOPED_TRACE(a.name);
        Recorder rec;
        BestFirstOptions o = with_h(heuristics::Kind::Max);
        o.control.blocked_states = {s1, task->initial_state()};
        o.control.observer = &rec;
        const BestFirstResult r = a.run(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        EXPECT_TRUE(valid_plan(*task, r.plan, task->initial_state()));
        State s = task->initial_state();
        for (const Action& x : r.plan)
        {
            s = succ.apply(s, x.label());
            EXPECT_FALSE(s == s1);
        }
        EXPECT_GT(rec.prunes, 0u);
        EXPECT_GE(r.stats.pruned, rec.prunes);
        if (a.run == &astar_eager || a.run == &astar_lazy)
        {
            EXPECT_GE(r.cost, base.cost);
        }
    }
}

TEST(BestFirst, GoalSpecs)
{
    const auto task = load("depot__p02");
    const BestFirstResult base = astar_eager(*task, with_h(heuristics::Kind::Blind));
    ASSERT_EQ(base.status, SearchStatus::Solved);
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    // AnyOf: reach the fluent atoms of the state two steps into the optimal plan, or an unreachable combination
    State s2 = task->initial_state();
    for (size_t i = 0; i < 2; ++i)
        s2 = succ.apply(s2, base.plan[i].label());
    GoalSpec::AtomGoal g2;
    for (u32 slot = 0; slot < s2.size_words() * 64; ++slot)
        if (s2.view().contains(SlotId{slot}))
            g2.positive.push_back(SlotId{slot});
    GoalSpec::AtomGoal never = g2;
    never.negative.push_back(g2.positive.front());  // p and not p
    for (const Algo& a : k_all)
    {
        SCOPED_TRACE(a.name);
        const bool astar = a.run == &astar_eager || a.run == &astar_lazy;
        BestFirstOptions o = with_h(astar ? heuristics::Kind::Max : heuristics::Kind::GoalCount);
        o.control.goal.kind = GoalSpec::Kind::AnyOf;
        o.control.goal.goals = {never, g2};
        const BestFirstResult r = a.run(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
        State end;
        ASSERT_TRUE(valid_plan(*task, r.plan, task->initial_state(), &end));
        for (SlotId p : g2.positive)
            EXPECT_TRUE(end.view().contains(p));
        if (astar)
        {
            EXPECT_LE(r.plan.size(), 2u);
        }

        // Custom: a predicate over states (here: the state two steps in, compared by content)
        BestFirstOptions c = with_h(heuristics::Kind::Blind);
        c.control.goal.kind = GoalSpec::Kind::Custom;
        c.control.goal.test = [&](StateView s) { return s == s2.view(); };
        const BestFirstResult rc = a.run(*task, c);
        ASSERT_EQ(rc.status, SearchStatus::Solved) << rc.message;
        ASSERT_TRUE(valid_plan(*task, rc.plan, task->initial_state(), &end));
        EXPECT_TRUE(end == s2);
    }
}

TEST(BestFirst, StartStateOption)
{
    const auto task = load("depot__p02");
    const BestFirstResult base = astar_eager(*task, with_h(heuristics::Kind::Max));
    ASSERT_EQ(base.status, SearchStatus::Solved);
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s1 = succ.apply(task->initial_state(), base.plan[0].label());
    BestFirstOptions o = with_h(heuristics::Kind::Max);
    o.start = s1;
    const BestFirstResult r = astar_eager(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    EXPECT_EQ(r.plan.size() + 1, base.plan.size());
    EXPECT_TRUE(valid_plan(*task, r.plan, s1));
}

namespace
{
/// A caller's evaluator: h_max through a library heuristic, one state at a time or in batches.
heuristics::Options max_options()
{
    heuristics::Options o;
    o.kind = heuristics::Kind::Max;
    return o;
}

class CountingHeuristic final : public heuristics::Heuristic
{
public:
    CountingHeuristic(const Task& task, bool batched) : m_h(heuristics::make_heuristic(task, max_options())), m_batched(batched) {}
    [[nodiscard]] heuristics::Kind kind() const noexcept override { return heuristics::Kind::Custom; }
    heuristics::Value evaluate(StateView s) override
    {
        ++singles;
        return m_h->evaluate(s);
    }
    heuristics::Value evaluate(StateView s, std::span<const GoalSpec::AtomGoal> goals) override { return m_h->evaluate(s, goals); }
    void evaluate_batch(std::span<const StateView> states, std::span<heuristics::Value> out) override
    {
        ++batches;
        batched_states += states.size();
        for (usize i = 0; i < states.size(); ++i)
            out[i] = m_h->evaluate(states[i]);
    }
    [[nodiscard]] bool batched() const noexcept override { return m_batched; }

    u64 singles = 0, batches = 0, batched_states = 0;

private:
    std::unique_ptr<heuristics::Heuristic> m_h;
    bool m_batched;
};
}  // namespace

TEST(BestFirst, BatchedEvaluatorsSearchLikeSingleOnes)
{
    // the same expansions, plans and costs; a batch per expansion (A*, GBFS) or per layer (beam)
    using Fn = BestFirstResult (*)(const Task&, const BestFirstOptions&);
    const std::pair<const char*, Fn> searches[] = {{"astar_eager", &astar_eager}, {"gbfs_eager", &gbfs_eager}, {"beam", &beam}};
    for (const char* name : {"depot__p02", "pegsol-08-strips__p22"})
    {
        const auto task = load(name);
        for (const auto& [algo, fn] : searches)
        {
            CountingHeuristic one(*task, false), many(*task, true);
            BestFirstOptions o;
            o.beam_width = 50;
            o.evaluator = &one;
            const BestFirstResult a = fn(*task, o);
            o.evaluator = &many;
            const BestFirstResult b = fn(*task, o);
            ASSERT_EQ(a.status, SearchStatus::Solved) << name << " " << algo;
            ASSERT_EQ(b.status, SearchStatus::Solved) << name << " " << algo;
            EXPECT_EQ(a.plan, b.plan) << name << " " << algo;
            EXPECT_EQ(a.cost, b.cost) << name << " " << algo;
            EXPECT_EQ(a.stats.expanded, b.stats.expanded) << name << " " << algo;
            EXPECT_EQ(one.batches, 0u);
            EXPECT_GT(many.batches, 0u) << name << " " << algo;
            EXPECT_EQ(many.singles, 1u) << name << " " << algo;  // the start state
            EXPECT_EQ(many.batched_states + 1, b.evaluations) << name << " " << algo;
            if (std::string(algo) == "beam")
                EXPECT_LE(many.batches, b.layers) << name;
            else
                EXPECT_LE(many.batches, b.stats.expanded) << name << " " << algo;
        }
    }
}

// ------------------------------------------------------------------------------------------------ heuristics
TEST(Heuristics, BasicPropertiesAlongAnOptimalPlan)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const BestFirstResult base = astar_eager(*task, with_h(heuristics::Kind::Max));
        ASSERT_EQ(base.status, SearchStatus::Solved);
        std::map<heuristics::Kind, std::unique_ptr<heuristics::Heuristic>> grounded, lifted;
        for (const auto k : {heuristics::Kind::Blind, heuristics::Kind::GoalCount, heuristics::Kind::Max,
                             heuristics::Kind::Add, heuristics::Kind::FF})
        {
            heuristics::Options o;
            o.kind = k;
            o.evaluation = heuristics::Evaluation::Grounded;
            grounded[k] = heuristics::make_heuristic(*task, o);
            o.evaluation = heuristics::Evaluation::Lifted;
            lifted[k] = heuristics::make_heuristic(*task, o);
        }
        const WorkspaceLease lease = task->workspace();
        Successors& succ = lease->successors();
        State s = task->initial_state();
        const size_t L = base.plan.size();
        for (size_t i = 0; i <= L; ++i)
        {
            const double remaining = static_cast<double>(L - i);
            const double hmax = grounded[heuristics::Kind::Max]->evaluate(s);
            const double hadd = grounded[heuristics::Kind::Add]->evaluate(s);
            const double hff = grounded[heuristics::Kind::FF]->evaluate(s);
            const double gc = grounded[heuristics::Kind::GoalCount]->evaluate(s);
            EXPECT_EQ(grounded[heuristics::Kind::Blind]->evaluate(s), 0.0);
            EXPECT_LE(hmax, remaining);  // admissible
            EXPECT_GE(hadd, hmax);
            EXPECT_TRUE(std::isfinite(hff));
            EXPECT_EQ(gc == 0, i == L);
            if (i == L)
            {
                EXPECT_EQ(hmax, 0.0);
                EXPECT_EQ(hff, 0.0);
            }
            // The fork's relaxation treats every negative derived literal as true at cost 0 (RelaxedPlanningGraph
            // initializes all negative derived propositions), so with derived predicates (philosophers, openstacks)
            // h can be 0 outside the goal and a relaxed plan need not contain an applicable action.
            const bool derived = std::string(name) == "philosophers__p03-phil4" || std::string(name) == "openstacks-opt08-adl__p03";
            EXPECT_EQ(lifted[heuristics::Kind::Max]->evaluate(s), hmax);
            EXPECT_EQ(lifted[heuristics::Kind::Add]->evaluate(s), hadd);
            // h_FF's relaxed plan: its size is h_FF (unit costs); its applicable actions are the preferred operators
            auto& ff = *grounded[heuristics::Kind::FF];
            ASSERT_TRUE(ff.provides_preferred());
            EXPECT_EQ(static_cast<double>(ff.relaxed_plan().size()), hff);
            u32 preferred = 0;
            for (const Action& a : succ.applicable_actions(s))
                preferred += ff.preferred(a.label());
            if (i < L && !derived)
            {
                EXPECT_GT(hmax, 0.0);  // goal-aware
                EXPECT_GT(preferred, 0u);
            }
            if (i < L)
                s = succ.apply(s, base.plan[i].label());
        }
        EXPECT_GT(grounded[heuristics::Kind::Max]->stats().grounded, 0u);
        EXPECT_EQ(lifted[heuristics::Kind::Max]->stats().grounded, 0u);
    }
}

TEST(Heuristics, AnyOfGoalsTakeTheCheapest)
{
    const auto task = load("depot__p02");
    const State s0 = task->initial_state();
    // one goal that holds in s0 (h = 0) and one that does not
    GoalSpec::AtomGoal holds, other;
    for (u32 slot = 0; slot < s0.size_words() * 64; ++slot)
        if (s0.view().contains(SlotId{slot}))
        {
            holds.positive.push_back(SlotId{slot});
            other.negative.push_back(SlotId{slot});
        }
    for (const auto k : {heuristics::Kind::GoalCount, heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
    {
        SCOPED_TRACE(heuristics::to_string(k));
        heuristics::Options o;
        o.kind = k;
        auto h = heuristics::make_heuristic(*task, o);
        const std::vector<GoalSpec::AtomGoal> both = {other, holds};
        EXPECT_EQ(h->evaluate(s0, both), 0.0);
        const std::vector<GoalSpec::AtomGoal> just_other = {other};
        EXPECT_GT(h->evaluate(s0, just_other), 0.0);
    }
    // the grounded-only kinds: a negative literal over an atom of s0 that depot never uses negatively has no
    // proposition in the grounding (refused); positive atoms of a later state do
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const BestFirstResult plan = astar_eager(*task, with_h(heuristics::Kind::Max));
    ASSERT_GE(plan.plan.size(), 3u);
    State s3 = s0;
    for (int i = 0; i < 3; ++i)
        s3 = succ.apply(s3, plan.plan[i].label());
    GoalSpec::AtomGoal later;
    for (u32 slot = 0; slot < s3.size_words() * 64; ++slot)
        if (s3.view().contains(SlotId{slot}) && !s0.view().contains(SlotId{slot}))
            later.positive.push_back(SlotId{slot});
    ASSERT_FALSE(later.positive.empty());
    for (const auto k : {heuristics::Kind::SetAdditive, heuristics::Kind::H2})
    {
        SCOPED_TRACE(heuristics::to_string(k));
        heuristics::Options o;
        o.kind = k;
        auto h = heuristics::make_heuristic(*task, o);
        const std::vector<GoalSpec::AtomGoal> both = {later, holds};
        EXPECT_EQ(h->evaluate(s0, both), 0.0);
        const std::vector<GoalSpec::AtomGoal> just_later = {later};
        const double v = h->evaluate(s0, just_later);
        EXPECT_GT(v, 0.0);
        if (k == heuristics::Kind::H2)
        {
            EXPECT_LE(v, 3.0);  // admissible: s3 is 3 steps away
        }
        EXPECT_EQ(h->evaluate(s3, just_later), 0.0);
        const std::vector<GoalSpec::AtomGoal> just_other = {other};
        EXPECT_THROW((void)h->evaluate(s0, just_other), std::runtime_error);
    }
}

TEST(Heuristics, InterruptStopsTheLiftedFallback)
{
    const auto task = load("freecell__p02");
    heuristics::Options o;
    o.kind = heuristics::Kind::FF;
    o.evaluation = heuristics::Evaluation::Lifted;
    auto h = heuristics::make_heuristic(*task, o);
    const double v = h->evaluate(task->initial_state());
    EXPECT_TRUE(std::isfinite(v));
    h->set_interrupt([] { return true; });
    EXPECT_THROW((void)h->evaluate(task->initial_state()), heuristics::Interrupted);
    h->set_interrupt({});
    EXPECT_EQ(h->evaluate(task->initial_state()), v);  // usable again

    // a search whose lifted evaluation runs past its deadline ends OutOfTime (not Failed)
    BestFirstOptions so;
    so.heuristic = o;
    so.control.budget.max_seconds = 1e-4;
    const BestFirstResult r = gbfs_eager(*task, so);
    EXPECT_EQ(r.status, SearchStatus::OutOfTime) << to_string(r.status) << " " << r.message;
}
