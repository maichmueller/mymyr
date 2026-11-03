// Searches for learned evaluators: batched expansions of eager A* and eager GBFS (BestFirstOptions::expansion_batch)
// and scored layer orderings (LayerOrdering::Kind::Scored). A* with batched expansions stays optimal with an
// admissible heuristic and hands the evaluator larger batches; a scorer that computes GoalCount's key ranks exactly as
// GoalCount does, in every beam mode, with random ties and with threads.

#include "../support/suite.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/layer_ordering.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
std::shared_ptr<const Task> load(const std::string& name) { return Task::from_text_file(task_path(name)); }

size_t optimal_length(const Task& task)
{
    BrfsOptions o;
    o.stop_at_goal = true;
    const BrfsResult r = brfs(task, o);
    EXPECT_TRUE(r.solved);
    return r.plan.size();
}

bool valid_plan(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    return succ.is_goal(s);
}

/// A batched evaluator over a library heuristic: records the batch sizes.
class BatchRecorder final : public heuristics::Heuristic
{
public:
    BatchRecorder(const Task& task, heuristics::Kind kind)
    {
        heuristics::Options o;
        o.kind = kind;
        m_h = heuristics::make_heuristic(task, o);
    }
    [[nodiscard]] heuristics::Kind kind() const noexcept override { return heuristics::Kind::Custom; }
    heuristics::Value evaluate(StateView s) override { return m_h->evaluate(s); }
    heuristics::Value evaluate(StateView s, std::span<const GoalSpec::AtomGoal> goals) override { return m_h->evaluate(s, goals); }
    void evaluate_batch(std::span<const StateView> states, std::span<heuristics::Value> out) override
    {
        ++batches;
        states_in_batches += states.size();
        largest = std::max<u64>(largest, states.size());
        for (usize i = 0; i < states.size(); ++i)
            out[i] = m_h->evaluate(states[i]);
    }
    [[nodiscard]] bool batched() const noexcept override { return true; }

    u64 batches = 0, states_in_batches = 0, largest = 0;

private:
    std::unique_ptr<heuristics::Heuristic> m_h;
};

/// Unsatisfied goal literals (fluent goals; each literal once): GoalCount's key with prefer_more_satisfied_goals.
f64 unsatisfied(const Task& task, StateView s)
{
    const formalism::TaskData& t = task.data();
    std::set<std::pair<std::vector<u32>, bool>> lits;
    for (const formalism::Literal& l : t.literals_of(t.goal))
    {
        if (t.predicates[l.pred.v].kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> key{l.pred.v};
        for (formalism::Term x : t.terms_of(l))
            key.push_back(formalism::term_object(x).v);
        lits.insert({key, l.positive});
    }
    u32 c = 0;
    for (const auto& [key, pos] : lits)
    {
        std::vector<ObjectId> args;
        for (usize i = 1; i < key.size(); ++i)
            args.push_back(ObjectId{key[i]});
        const SlotId slot = task.find_atom(PredicateId{key[0]}, args);
        c += s.contains(slot) != pos;
    }
    return c;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ batched expansions

TEST(BatchedExpansions, AStarStaysOptimalWithLargerBatches)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4", "pegsol-08-strips__p22"})
    {
        const auto task = load(name);
        const size_t best = optimal_length(*task);
        u64 one_largest = 0;
        for (const u32 k : {1u, 4u, 32u})
            for (const auto queue : {BestFirstOptions::Queue::Bucket, BestFirstOptions::Queue::Heap})
            {
                const std::string what = std::string(name) + " k=" + std::to_string(k);
                BatchRecorder h(*task, heuristics::Kind::Max);
                BestFirstOptions o;
                o.evaluator = &h;
                o.queue = queue;
                o.expansion_batch = k;
                const BestFirstResult r = astar_eager(*task, o);
                ASSERT_EQ(r.status, SearchStatus::Solved) << what;
                EXPECT_EQ(r.plan.size(), best) << what;
                EXPECT_EQ(r.cost, static_cast<double>(best)) << what;
                EXPECT_TRUE(valid_plan(*task, r.plan)) << what;
                EXPECT_EQ(h.states_in_batches + 1, r.evaluations) << what;  // + the start state
                if (k == 1)
                    one_largest = std::max(one_largest, h.largest);
                else
                    EXPECT_GT(h.largest, one_largest) << what;
            }
    }
}

TEST(BatchedExpansions, MatchesTheOneExpansionSearchForOne)
{
    const auto task = load("depot__p02");
    BestFirstOptions o;
    o.heuristic.kind = heuristics::Kind::Max;
    const BestFirstResult a = astar_eager(*task, o);
    const BestFirstResult g = gbfs_eager(*task, o);
    o.expansion_batch = 1;
    EXPECT_EQ(astar_eager(*task, o).plan, a.plan);
    EXPECT_EQ(gbfs_eager(*task, o).stats.expanded, g.stats.expanded);
}

TEST(BatchedExpansions, GbfsAndWeightedAStarSolve)
{
    for (const char* name : {"depot__p02", "openstacks-opt08-adl__p03", "pegsol-08-strips__p22"})
    {
        const auto task = load(name);
        const size_t best = optimal_length(*task);
        for (const u32 k : {2u, 16u})
        {
            BestFirstOptions o;
            o.heuristic.kind = heuristics::Kind::FF;
            o.expansion_batch = k;
            const BestFirstResult g = gbfs_eager(*task, o);
            ASSERT_EQ(g.status, SearchStatus::Solved) << name << " k=" << k;
            EXPECT_TRUE(valid_plan(*task, g.plan)) << name;
            o.heuristic.kind = heuristics::Kind::Max;
            o.weight = 2;
            o.queue = BestFirstOptions::Queue::Heap;
            const BestFirstResult w = astar_eager(*task, o);
            ASSERT_EQ(w.status, SearchStatus::Solved) << name << " k=" << k;
            EXPECT_LE(w.cost, 2.0 * static_cast<double>(best)) << name;
        }
    }
}

TEST(BatchedExpansions, BudgetsAndRefusals)
{
    const auto task = load("pegsol-08-strips__p22");
    BestFirstOptions o;
    o.heuristic.kind = heuristics::Kind::Blind;
    o.expansion_batch = 8;
    o.control.budget.max_expanded = 37;
    for (auto fn : {&astar_eager, &gbfs_eager})
    {
        const BestFirstResult r = fn(*task, o);
        EXPECT_EQ(r.status, SearchStatus::OutOfStates);
        EXPECT_EQ(r.stats.expanded, 37u);
    }
    o.control.budget.max_expanded = std::numeric_limits<u64>::max();
    EXPECT_THROW((void)astar_lazy(*task, o), std::invalid_argument);
    EXPECT_THROW((void)gbfs_lazy(*task, o), std::invalid_argument);
    EXPECT_THROW((void)beam(*task, o), std::invalid_argument);
    o.expansion_batch = 0;
    EXPECT_THROW((void)astar_eager(*task, o), std::invalid_argument);
    o.expansion_batch = 4;
    o.heuristic.kind = heuristics::Kind::LmCount;
    const BestFirstResult p = gbfs_eager(*task, o);
    EXPECT_EQ(p.status, SearchStatus::Failed);
    EXPECT_NE(p.message.find("path-dependent"), std::string::npos) << p.message;
}

// ------------------------------------------------------------------------------------------------ scored layers

TEST(ScoredLayers, RankLikeGoalCountInEveryBeamMode)
{
    using Novelty = LayerOrdering::BeamNovelty;
    for (const char* name : {"blocks__probBLOCKS-8-0", "depot__p02", "gripper__prob05"})
    {
        const auto task = load(name);
        u64 calls = 0;
        auto scorer = [&](std::span<const StateView> states, std::span<f64> out)
        {
            ++calls;
            for (usize i = 0; i < states.size(); ++i)
                out[i] = unsatisfied(*task, states[i]);
        };
        for (const Novelty mode : {Novelty::AllTested, Novelty::SurvivorsOnly})
            for (const bool ties : {false, true})
                for (const u32 threads : {1u, 4u})
                {
                    const std::string what = std::string(name) + " mode=" + std::to_string(static_cast<int>(mode)) +
                                             " ties=" + std::to_string(ties) + " threads=" + std::to_string(threads);
                    IwOptions o;
                    o.max_width = 1;
                    o.threads = threads;
                    o.layers.kind = LayerOrdering::Kind::GoalCount;
                    o.layers.beam_width = 8;
                    o.layers.beam_novelty = mode;
                    o.layers.randomize_ties = ties;
                    o.layers.seed = 7;
                    const IwResult a = iw(*task, o);
                    o.layers.kind = LayerOrdering::Kind::Scored;
                    o.layers.scorer = scorer;
                    calls = 0;
                    const IwResult b = iw(*task, o);
                    EXPECT_EQ(a.status, b.status) << what;
                    EXPECT_EQ(a.plan, b.plan) << what;
                    EXPECT_EQ(a.stats.expanded, b.stats.expanded) << what;
                    EXPECT_EQ(a.stats.generated, b.stats.generated) << what;
                    EXPECT_GT(calls, 0u) << what;

                    BrfsOptions bo;
                    bo.stop_at_goal = true;
                    bo.threads = threads;
                    bo.layers = o.layers;
                    bo.layers.kind = LayerOrdering::Kind::GoalCount;
                    bo.layers.scorer = nullptr;
                    const BrfsResult c = brfs(*task, bo);
                    bo.layers.kind = LayerOrdering::Kind::Scored;
                    bo.layers.scorer = scorer;
                    const BrfsResult d = brfs(*task, bo);
                    EXPECT_EQ(c.solved, d.solved) << what;
                    EXPECT_EQ(c.plan, d.plan) << what;
                    EXPECT_EQ(c.expanded, d.expanded) << what;
                }
    }
}

TEST(ScoredLayers, LowerScoresFirstWithNegativesAndInfinities)
{
    // the reversed score (more unsatisfied literals first) ranks like GoalCount with prefer_more_satisfied_goals off;
    // shifting by -inf-free negatives and mapping the worst to +inf keeps the order
    const auto task = load("blocks__probBLOCKS-8-0");
    IwOptions o;
    o.max_width = 1;
    o.layers.kind = LayerOrdering::Kind::GoalCount;
    o.layers.prefer_more_satisfied_goals = false;
    o.layers.beam_width = 4;
    const IwResult a = iw(*task, o);
    const f64 most = 64;
    o.layers.kind = LayerOrdering::Kind::Scored;
    o.layers.prefer_more_satisfied_goals = true;
    o.layers.scorer = [&](std::span<const StateView> states, std::span<f64> out)
    {
        for (usize i = 0; i < states.size(); ++i)
        {
            const f64 u = unsatisfied(*task, states[i]);
            out[i] = u == 0 ? std::numeric_limits<f64>::infinity() : -1e6 * u + most;
        }
    };
    const IwResult b = iw(*task, o);
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.plan, b.plan);
    EXPECT_EQ(a.stats.expanded, b.stats.expanded);
}

TEST(ScoredLayers, Refusals)
{
    LayerOrdering lo;
    lo.kind = LayerOrdering::Kind::Scored;
    EXPECT_NE(check_layers(lo), "");  // no scorer
    lo.scorer = [](std::span<const StateView>, std::span<f64> out) { std::ranges::fill(out, 0.0); };
    EXPECT_EQ(check_layers(lo), "");
    lo.randomize_ties = true;
    EXPECT_EQ(check_layers(lo), "");
    lo.beam_width = 4;
    lo.beam_novelty = LayerOrdering::BeamNovelty::RelaxedSurvivorsOnly;
    EXPECT_NE(check_layers(lo), "");
    LayerOrdering other;
    other.kind = LayerOrdering::Kind::InOrder;
    other.scorer = lo.scorer;
    EXPECT_NE(check_layers(other), "");  // a scorer for a kind without scores

    const auto task = load("gripper__prob05");
    IwOptions o;
    o.max_width = 1;
    o.layers.kind = LayerOrdering::Kind::Scored;
    o.layers.scorer = [](std::span<const StateView>, std::span<f64> out) { std::ranges::fill(out, std::nan("")); };
    EXPECT_THROW((void)iw(*task, o), std::invalid_argument);
}
