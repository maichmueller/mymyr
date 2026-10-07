#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/novelty/minimum_g_table.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>

using namespace mymyr;
using namespace mymyr::search;
namespace
{
constexpr std::array k_modes{AStarIwFeatures::Classical, AStarIwFeatures::Abstracted, AStarIwFeatures::BaseAbstracted};

std::shared_ptr<const Task> gap(bool negative_goal = false)
{
    std::istringstream in(std::string("O 0\nP 4\nF 0 seed\nF 0 a\nF 0 b\nF 0 goal\nSI 0\nFI 1\n0\n") +
        (negative_goal ? "G 1\n0 0\n" : "G 1\n3 1\n") +
        "A 4\nadd-a 0\nL 1\n0 1\nE 1\n0\nL 0\nF 1\n1 1\n"
        "add-b 0\nL 1\n0 1\nE 1\n0\nL 0\nF 1\n2 1\n"
        "finish 0\nL 2\n1 1\n2 1\nE 1\n0\nL 0\nF 1\n3 1\n"
        "clear-seed 0\nL 1\n0 1\nE 1\n0\nL 0\nF 1\n0 0\n");
    return Task::create(formalism::read_task_text(in));
}

std::shared_ptr<const landmarks::FactLandmarkGraph> graph(const Task& task)
{
    return std::make_shared<const landmarks::FactLandmarkGraph>(landmarks::FactLandmarkGraph::create(
        {task.atoms().layout().encode(1, nullptr), task.atoms().layout().encode(2, nullptr), task.atoms().layout().encode(3, nullptr)}));
}

void same(const AStarIwResult& a, const AStarIwResult& b)
{
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.cost, b.cost);
    EXPECT_EQ(a.plan, b.plan);
    EXPECT_EQ(a.stats.expanded, b.stats.expanded);
    EXPECT_EQ(a.stats.generated, b.stats.generated);
}

State state(std::initializer_list<u32> atoms)
{
    std::vector<u64> words;
    for (u32 a : atoms)
    {
        words.resize(std::max<usize>(words.size(), bits::words_for(a + 1)), 0);
        bits::set(words.data(), a);
    }
    return State(StateView{words.data(), static_cast<u32>(words.size())});
}

struct Callback : heuristics::Heuristic
{
    std::function<double(StateView)> fn;
    explicit Callback(std::function<double(StateView)> f) : fn(std::move(f)) {}
    heuristics::Kind kind() const noexcept override { return heuristics::Kind::Custom; }
    double evaluate(StateView s) override { return fn(s); }
    double evaluate(StateView s, std::span<const GoalSpec::AtomGoal>) override { return fn(s); }
};
}  // namespace

TEST(AStarIw, ClassicalSolves)
{
    const auto task = gap();
    AStarIwOptions o;
    o.width = 2;
    const auto r = astar_iw(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.cost, 3);
    EXPECT_EQ(r.plan.size(), 3u);
    EXPECT_GT(r.stats.expanded, 0u);
    EXPECT_GT(r.stats.generated, 0u);
    State s = task->initial_state();
    for (const Action& a : r.plan)
        s = task->workspace().successors().apply(s, a.label());
    EXPECT_TRUE(task->is_goal(s));
}

TEST(AStarIw, AbstractedModesSolve)
{
    const auto task = gap();
    for (auto mode : {AStarIwFeatures::Abstracted, AStarIwFeatures::BaseAbstracted})
    {
        AStarIwOptions o;
        o.width = 2;
        o.features = mode;
        o.weight = mode == AStarIwFeatures::BaseAbstracted ? 2 : 1;
        const auto r = astar_iw(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    }
}

TEST(AStarIw, ValidatesOptions)
{
    const auto task = gap();
    for (auto mode : k_modes)
        for (u32 width : {0u, mode == AStarIwFeatures::Classical ? 6u : 4u})
        {
            AStarIwOptions o;
            o.features = mode;
            o.width = width;
            auto r = astar_iw(*task, o);
            EXPECT_EQ(r.status, SearchStatus::Failed);
            EXPECT_FALSE(r.message.empty());
        }
    for (double weight : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        AStarIwOptions o;
        o.weight = weight;
        EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Failed);
    }
}

TEST(AStarIw, LandmarkNoveltyClosesWidthOneGapInEveryMode)
{
    const auto task = gap();
    for (auto mode : k_modes)
    {
        AStarIwOptions o;
        o.features = mode;
        EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Exhausted);
        o.landmarks.graph = graph(*task);
        EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Solved);
    }
}

TEST(AStarIw, LandmarkNoveltySolvesAtWidthOne)
{
    const auto task = gap();
    AStarIwOptions o;
    o.landmarks.graph = graph(*task);
    const auto r = astar_iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Solved);
    EXPECT_EQ(r.plan.size(), 3u);
}

TEST(AStarIw, EmptyLandmarksMatchUnrestrictedSearch)
{
    const auto task = gap();
    for (auto mode : k_modes)
    {
        AStarIwOptions o;
        o.width = 2;
        o.features = mode;
        const auto plain = astar_iw(*task, o);
        o.landmarks.graph = std::make_shared<const landmarks::FactLandmarkGraph>(landmarks::FactLandmarkGraph::create({}));
        same(plain, astar_iw(*task, o));
    }
}

TEST(AStarIw, MatchesIwWidthBoundary)
{
    const auto task = gap();
    for (u32 width : {1u, 2u})
    {
        AStarIwOptions o;
        o.width = width;
        IwOptions iw_opts;
        iw_opts.max_arity = width;
        const auto r = astar_iw(*task, o);
        EXPECT_EQ(r.status == SearchStatus::Solved, iw(*task, iw_opts).status == SearchStatus::Solved);
        EXPECT_EQ(r.status == SearchStatus::Solved, width == 2);
        if (width == 1)
        {
            EXPECT_GT(r.novelty.rejected, 0u);
        }
    }
}

TEST(AStarIw, RootGoalOnlyException)
{
    const auto task = gap(true);
    AStarIwOptions o;
    const auto r = astar_iw(*task, o);
    EXPECT_EQ(r.status, SearchStatus::Solved);
    EXPECT_EQ(r.plan.size(), 1u);
    o.allow_non_novel_root_goal = false;
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Exhausted);
}

TEST(AStarIw, ForwardsBudgets)
{
    const auto task = gap();
    AStarIwOptions o;
    o.width = 2;
    o.control.budget.max_states = 1;
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::OutOfStates);
    o.control.budget = {};
    o.control.budget.max_seconds = 0;
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::OutOfTime);
    o.control.budget = {};
    o.control.budget.max_expanded = 1;
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::OutOfStates);
    o.control.budget = {};
    o.control.budget.max_depth = 1;
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Exhausted);
}

TEST(MinimumGNovelty, LowersTupleDepths)
{
    novelty::MinimumGNoveltyTable table(2);
    const State root = state({0}), child = state({0, 1});
    EXPECT_TRUE(table.test_and_update(root.view(), 0));
    EXPECT_TRUE(table.test_at_g(root.view(), 0));
    EXPECT_TRUE(table.test_and_update(root.view(), child.view(), 1));
    EXPECT_TRUE(table.test_at_g(child.view(), 1));
    EXPECT_FALSE(table.test_and_update(root.view(), child.view(), 1));
    EXPECT_TRUE(table.test_and_update(root.view(), child.view(), 0));
    EXPECT_FALSE(table.test_at_g(child.view(), 1));
}

TEST(MinimumGNovelty, TracksLoweringOfExistingLabels)
{
    novelty::MinimumGNoveltyTable table(2);
    const State root = state({0}), child = state({0, 1});
    table.test_and_update(root.view(), 0);
    EXPECT_FALSE(table.has_lowered_existing_label());
    table.test_and_update(root.view(), child.view(), 2);
    EXPECT_FALSE(table.has_lowered_existing_label());
    table.test_and_update(root.view(), child.view(), 1);
    EXPECT_TRUE(table.has_lowered_existing_label());
}

TEST(AStarIw, ProbeOptionDoesNotChangeTheSearch)
{
    const auto task = Task::from_text_file(test::task_path("blocks__probBLOCKS-8-0"));
    for (auto mode : k_modes)
        for (u32 width : {1u, 2u})
        {
            AStarIwOptions o;
            o.features = mode;
            o.width = width;
            const auto probed = astar_iw(*task, o);
            o.probe_novelty_before_heuristic = false;
            const auto unprobed = astar_iw(*task, o);
            same(probed, unprobed);
            EXPECT_LE(probed.evaluations, unprobed.evaluations);
        }
}

TEST(MinimumGNovelty, ProbeAgreesAndDoesNotWrite)
{
    novelty::MinimumGNoveltyTable table(2);
    const State root = state({0}), child = state({0, 1});
    table.test_and_update(root.view(), 0);
    EXPECT_TRUE(table.would_improve(root.view(), child.view(), 2));
    EXPECT_FALSE(table.test_at_g(child.view(), 2));
    EXPECT_TRUE(table.would_improve(root.view(), child.view(), 2));
    EXPECT_TRUE(table.test_and_update(root.view(), child.view(), 2));
    EXPECT_FALSE(table.would_improve(root.view(), child.view(), 2));
    EXPECT_TRUE(table.would_improve(root.view(), child.view(), 1));
}

TEST(MinimumGNovelty, RetainsLabelsAcrossSeventyThousandDistinctCosts)
{
    novelty::MinimumGNoveltyTable table(2);
    const State s = state({0, 1});
    for (u32 g = 70000; g > 0; --g)
        ASSERT_TRUE(table.test_and_update(s.view(), g));
    EXPECT_TRUE(table.test_at_g(s.view(), 1));
    EXPECT_FALSE(table.test_at_g(s.view(), 2));
    EXPECT_FALSE(table.test_at_g(s.view(), 70000));
    EXPECT_FALSE(table.test_and_update(s.view(), 1));
}

TEST(AStarIw, PreserveLandmarkAtomsIsIgnoredWhenInapplicable)
{
    const auto task = gap();
    for (auto mode : k_modes)
    {
        AStarIwOptions o;
        o.width = 2;
        o.features = mode;
        if (mode == AStarIwFeatures::Classical)
            o.landmarks.graph = graph(*task);
        const auto preserved = astar_iw(*task, o);
        o.preserve_landmark_atoms = false;
        same(preserved, astar_iw(*task, o));
    }
}

TEST(AStarIw, PreserveLandmarkAtomsSolvesInEveryAbstractedMode)
{
    const auto task = gap();
    for (auto mode : {AStarIwFeatures::Abstracted, AStarIwFeatures::BaseAbstracted})
        for (bool preserve : {false, true})
        {
            AStarIwOptions o;
            o.features = mode;
            o.landmarks.graph = graph(*task);
            o.preserve_landmark_atoms = preserve;
            EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Solved);
        }
}

TEST(AStarIw, StoresHaveIdenticalSearches)
{
    const auto task = gap();
    AStarIwOptions o;
    o.width = 2;
    const auto baseline = astar_iw(*task, o);
    for (auto store : {BestFirstOptions::Store::Flat, BestFirstOptions::Store::Chunked, BestFirstOptions::Store::Compact})
    {
        o.store = store;
        same(baseline, astar_iw(*task, o));
    }
}

TEST(AStarIw, ConcurrentSearchesShareOneTask)
{
    const auto task = gap();
    std::array<AStarIwResult, 8> results;
    std::vector<std::thread> workers;
    for (usize i = 0; i < results.size(); ++i)
        workers.emplace_back([&, i]
        {
            AStarIwOptions o;
            o.width = 2;
            o.features = k_modes[i % k_modes.size()];
            o.landmarks.graph = graph(*task);
            results[i] = astar_iw(*task, o);
        });
    for (auto& worker : workers)
        worker.join();
    for (const auto& r : results)
    {
        EXPECT_EQ(r.status, SearchStatus::Solved) << r.message;
        EXPECT_EQ(r.cost, 3);
    }
}

TEST(AStarIw, InitialGoalAndDeadEnd)
{
    const auto task = gap();
    AStarIwOptions o;
    o.start = state({3});
    Callback inf([](StateView) { return std::numeric_limits<double>::infinity(); });
    EXPECT_EQ(astar_iw(*task, inf, o).status, SearchStatus::Solved);
    o.start.reset();
    const auto r = astar_iw(*task, inf, o);
    EXPECT_EQ(r.status, SearchStatus::Unsolvable);
    EXPECT_EQ(r.stats.expanded, 0u);
    Callback nan([](StateView) { return std::numeric_limits<double>::quiet_NaN(); });
    EXPECT_EQ(astar_iw(*task, nan, o).status, SearchStatus::Failed);
}

TEST(MinimumGNovelty, WidthsCoordinatesAndEmptyStates)
{
    const State empty = state({}), one = state({0}), five = state({0, 1, 2, 3, 4});
    for (u32 width = 1; width <= novelty::k_max_arity; ++width)
    {
        novelty::MinimumGNoveltyTable table(width);
        EXPECT_EQ(table.test_and_update(empty.view(), 0), width == 1);
        EXPECT_EQ(table.test_and_update(one.view(), 1), width <= 2);
        EXPECT_TRUE(table.test_and_update(five.view(), 2, 3));
        EXPECT_TRUE(table.test_at_g(five.view(), 2, 3));
        EXPECT_FALSE(table.test_at_g(five.view(), 2, 4));
        EXPECT_TRUE(table.would_improve(one.view(), five.view(), 2, 4));
    }
    EXPECT_THROW(novelty::MinimumGNoveltyTable(0), std::invalid_argument);
    EXPECT_THROW(novelty::MinimumGNoveltyTable(6), std::invalid_argument);
}

TEST(AStarIw, DiscardsStatesWhoseTupleLabelsWereLowered)
{
    std::istringstream in(
        "O 0\nP 7\nF 0 root\nF 0 a\nF 0 b\nF 0 q\nF 0 t\nF 0 z\nF 0 goal\n"
        "SI 0\nFI 1\n0\nG 1\n6 1\nA 5\n"
        "long 0\nL 1\n0 1\nE 1\n0\nL 0\nF 2\n0 0\n1 1\n"
        "short 0\nL 1\n0 1\nE 1\n0\nL 0\nF 2\n0 0\n3 1\n"
        "next 0\nL 1\n1 1\nE 1\n0\nL 0\nF 2\n1 0\n2 1\n"
        "target 0\nL 1\n2 1\nE 1\n0\nL 0\nF 2\n2 0\n4 1\n"
        "lower 0\nL 1\n3 1\nE 1\n0\nL 0\nF 3\n3 0\n4 1\n5 1\n");
    const auto task = Task::create(formalism::read_task_text(in));
    const SlotId q = task->find_atom(PredicateId{3}, {}), t = task->find_atom(PredicateId{4}, {});
    for (auto mode : k_modes)
    {
        Callback h([&](StateView s) { return s.contains(q) ? 10 : s.contains(t) ? 20 : 0; });
        AStarIwOptions o;
        o.features = mode;
        const auto r = astar_iw(*task, h, o);
        EXPECT_EQ(r.status, SearchStatus::Exhausted);
        EXPECT_EQ(r.novelty.stale, 1u);
        EXPECT_GT(r.novelty.pop_tests, 0u);
        EXPECT_EQ(r.stats.expanded, 5u);
    }
}

TEST(AStarIw, CancellationBlockedStatesAndObserverEvents)
{
    const auto task = gap();
    AStarIwOptions o;
    o.width = 2;
    o.control.cancel.request();
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Cancelled);
    o.control = {};
    struct Observer : SearchObserver
    {
        u32 starts = 0, transitions = 0, solutions = 0, ends = 0;
        void on_start(StateView) override { ++starts; }
        void on_transition(u64, const Action&, u64, StateView, TransitionOutcome) override { ++transitions; }
        void on_solution(std::span<const Action>, double) override { ++solutions; }
        void on_end(SearchStatus, const SearchStatistics&) override { ++ends; }
    } observer;
    o.control.observer = &observer;
    const auto r = astar_iw(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    EXPECT_EQ(observer.starts, 1u);
    EXPECT_EQ(observer.solutions, 1u);
    EXPECT_EQ(observer.ends, 1u);
    EXPECT_EQ(observer.transitions, r.stats.generated);
    o.control.observer = nullptr;
    o.control.blocked_states.push_back(*r.goal_state);
    EXPECT_EQ(astar_iw(*task, o).status, SearchStatus::Exhausted);
}
