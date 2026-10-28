// Datasets: state spaces on small instances, ids and arrays independent of
// the thread count, the array invariants, the options (max_states, remove_if_unsolvable, statically false goals,
// timeouts), transition costs and numeric tasks, the instance pool, generalized state spaces and samplers.

#include "support.hpp"
#include "../support/benchmarks.hpp"
#include "../support/branching.hpp"
#include "../support/suite.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
using namespace mymyr::test::ss;

// ----------------------------------------------------------------------------------------------- tasks
TaskPtr small_task(const std::string& dir, const std::string& problem, TaskOptions to = {})
{
    return dense_task(test::small_dir(), dir, problem, to);
}

TaskPtr pddl_task(const char* domain, const char* problem)
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Dense;
    return Task::create(*frontend::Domain::from_string(domain, "d.pddl")->instantiate_string(problem, "p.pddl"), to);
}

// ----------------------------------------------------------------------------------------------- determinism
TEST(StateSpace, ArraysIndependentOfThreadCount)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4"})
    {
        for (auto atoms : {TaskOptions::Atoms::Dense, TaskOptions::Atoms::Lazy})
        {
            TaskOptions to;
            to.atoms = atoms;
            const auto task = Task::from_text_file(test::task_path(name), to);
            const StateSpacePtr ref = space_of(task, ss_options(1));
            ASSERT_TRUE(ref);
            const bool frozen = atoms == TaskOptions::Atoms::Dense;
            const u64 h = arrays_hash(*ref, frozen);  // lazy slots: the state words depend on the interning order
            for (u32 T : {2u, 3u, 8u})
            {
                const StateSpacePtr s = space_of(task, ss_options(T));
                ASSERT_TRUE(s);
                EXPECT_EQ(arrays_hash(*s, frozen), h) << name << " T=" << T;
            }
            if (frozen)
                check_invariants(*ref);
        }
    }
}

TEST(StateSpace, ConcurrentStoreGrowsInsideAnExpansion)
{
    // one expansion (modes 1) or the threads' concurrent expansions (modes 8) insert more states than the concurrent
    // store's initial table holds: the threads pause inside them for the rehash, and the arrays equal one thread's
    struct Case
    {
        u32 modes, n;
    };
    for (const Case c : {Case{1, 65}, Case{8, 35}})
    {
        const TaskPtr task =
            pddl_task(test::branching_domain().c_str(), test::branching_problem(c.modes, c.n).c_str());
        const StateSpacePtr ref = space_of(task, ss_options(1));
        ASSERT_TRUE(ref);
        ASSERT_EQ(ref->num_states(), test::branching_states(c.modes, c.n));
        EXPECT_EQ(ref->num_transitions(), ref->num_states() - 1);
        const u64 h = arrays_hash(*ref, true);
        for (u32 T : {2u, 8u})
        {
            const StateSpacePtr s = space_of(task, ss_options(T));
            ASSERT_TRUE(s);
            EXPECT_EQ(arrays_hash(*s, true), h) << "modes " << c.modes << " T=" << T;
        }
    }
}

TEST(StateSpace, CountsMatchTheBrfsSuite)
{
    for (const auto& t : test::suite())
    {
        if (t.states > 50000)
            continue;
        const auto task = Task::from_text_file(test::task_path(t.name));
        const StateSpacePtr s = space_of(task, ss_options(2));
        ASSERT_TRUE(s);
        EXPECT_EQ(s->num_states(), t.states) << t.name;
        EXPECT_EQ(s->num_goal_states(), t.goal_states) << t.name;
    }
}

// ----------------------------------------------------------------------------------------------- options
TEST(StateSpace, MaxStatesFailsIffTheSpaceReachesIt)
{
    // the state that fills the store ends the generation: OutOfStates with `states` the limit, on every thread count
    const TaskPtr task = small_task("gripper", "test_problem4.pddl");  // 256 states
    if (!task)
        GTEST_SKIP();
    for (u32 T : {1u, 4u})
    {
        StateSpaceOptions o = ss_options(T);
        for (const u64 limit : {u64{0}, u64{1}, u64{2}, u64{10}, u64{255}, u64{256}})
        {
            o.max_states = limit;
            const StateSpaceResult r = generate_state_space(task, o);
            EXPECT_EQ(r.status, search::SearchStatus::OutOfStates) << T << " " << limit;
            EXPECT_EQ(r.states, std::max<u64>(limit, 1)) << T << " " << limit;
        }
        o.max_states = 257;
        EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::Exhausted) << T;
    }
    // a single-state space fits a limit of 2 or more; the initial state alone fills one of 1 (or 0)
    const TaskPtr one = small_task("deadend", "test_problem.pddl");
    ASSERT_TRUE(one);
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o = ss_options(T);
        o.remove_if_unsolvable = false;
        o.max_states = 2;
        const StateSpaceResult r = generate_state_space(one, o);
        ASSERT_EQ(r.status, search::SearchStatus::Exhausted);
        EXPECT_EQ(r.space->num_states(), 1u);
        for (const u64 limit : {u64{0}, u64{1}})
        {
            o.max_states = limit;
            const StateSpaceResult full = generate_state_space(one, o);
            EXPECT_EQ(full.status, search::SearchStatus::OutOfStates) << T << " " << limit;
            EXPECT_EQ(full.states, 1u);
        }
    }
}

TEST(StateSpace, RemoveIfUnsolvableDropsTheWholeSpace)
{
    const TaskPtr task = small_task("deadend", "test_problem.pddl");  // the initial state is a dead end
    if (!task)
        GTEST_SKIP();
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o;
        o.threads = T;
        EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::Unsolvable);
        o.remove_if_unsolvable = false;
        const StateSpaceResult r = generate_state_space(task, o);
        ASSERT_EQ(r.status, search::SearchStatus::Exhausted);
        EXPECT_EQ(r.space->num_unsolvable_states(), r.space->num_states());
        EXPECT_EQ(r.space->unit_goal_distances()[0], k_unsolvable_distance);
        EXPECT_TRUE(std::isinf(r.space->cost_goal_distances()[0]));
    }
    // a space whose initial state reaches a goal keeps its dead ends
    const TaskPtr spanner = small_task("spanner", "p15-easy.pddl");
    ASSERT_TRUE(spanner);
    const StateSpaceResult r = generate_state_space(spanner, {});
    ASSERT_EQ(r.status, search::SearchStatus::Exhausted);
    EXPECT_GT(r.space->num_unsolvable_states(), 0u);
}

const char* kCountDomain = R"(
(define (domain count)
 (:requirements :strips :typing :action-costs)
 (:types cell)
 (:predicates (at ?c - cell) (next ?a ?b - cell) (never))
 (:functions (total-cost) (step-cost ?a ?b - cell))
 (:action step :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) (step-cost ?a ?b))))
 (:action jump :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) 10)))
)
)";

TEST(StateSpace, StaticallyFalseGoalHasNoSpace)
{
    const TaskPtr task = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 - cell)
 (:init (at c0) (next c0 c1) (= (total-cost) 0) (= (step-cost c0 c1) 1))
 (:goal (and (at c1) (never)))
 (:metric minimize (total-cost)))
)");
    for (bool remove : {false, true})
    {
        StateSpaceOptions o;
        o.remove_if_unsolvable = remove;
        EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::Unsolvable);
        o.threads = 2;
        EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::Unsolvable);
    }
}

TEST(StateSpace, TransitionCostsAndCostDistances)
{
    // c0 -> c1 -> c2 -> c3 (goal), step costs 1, 2, 3, jumps cost 10; c0 -> c3 directly for 4
    const TaskPtr task = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c0) (next c0 c1) (next c1 c2) (next c2 c3) (next c0 c3) (= (total-cost) 0)
        (= (step-cost c0 c1) 1) (= (step-cost c1 c2) 2) (= (step-cost c2 c3) 3) (= (step-cost c0 c3) 7))
 (:goal (at c3))
 (:metric minimize (total-cost)))
)");
    for (u32 T : {1u, 2u})
    {
        const StateSpacePtr S = space_of(task, ss_options(T));
        ASSERT_TRUE(S);
        ASSERT_EQ(S->num_states(), 4u);
        EXPECT_EQ(S->num_transitions(), 8u);
        ASSERT_FALSE(S->unit_costs());
        // state 0 = at c0: its transitions in canonical order (schema, then binding): step c0 c1, step c0 c3,
        // jump c0 c1, jump c0 c3
        std::vector<f64> row0(S->costs().begin(), S->costs().begin() + 4);
        EXPECT_EQ(row0, (std::vector<f64>{1, 7, 10, 10}));
        EXPECT_EQ(S->unit_goal_distances()[0], 1);
        EXPECT_DOUBLE_EQ(S->cost_goal_distances()[0], 6.0);  // 1 + 2 + 3 beats 7 and 10
        check_invariants(*S);
    }
    // without labels the arrays are the same except the labels
    StateSpaceOptions o = ss_options();
    o.labels = false;
    const StateSpacePtr S = space_of(task, o);
    ASSERT_TRUE(S);
    EXPECT_FALSE(S->has_labels());
    EXPECT_TRUE(S->label_schemas().empty());
    EXPECT_THROW((void)S->label(0), std::logic_error);
}

const char* kNumericDomain = R"(
(define (domain counter)
 (:requirements :strips :numeric-fluents)
 (:predicates (done))
 (:functions (x) (fuel))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1) (decrease (fuel) 1)))
 (:action dec :parameters () :precondition (and (> (x) 0)) :effect (and (decrease (x) 1) (increase (fuel) 2)))
 (:action finish :parameters () :precondition (and (= (x) 3)) :effect (and (done)))
)
)";

TEST(StateSpace, NumericTasks)
{
    const TaskPtr task = pddl_task(kNumericDomain, R"(
(define (problem p) (:domain counter)
 (:init (= (x) 0) (= (fuel) 0))
 (:goal (done))
 (:metric minimize (fuel)))
)");
    ASSERT_GT(task->numeric_slots(), 0u);
    StateSpaceOptions o = ss_options();
    o.max_states = 1000;  // fuel is unbounded: the space is infinite
    EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::OutOfStates);
    o.threads = 3;
    EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::OutOfStates);
    // a bounded counter: x in 0..3 with and without done
    const TaskPtr bounded = pddl_task(R"(
(define (domain counter2)
 (:requirements :strips :numeric-fluents :negative-preconditions)
 (:predicates (done))
 (:functions (x))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1)))
 (:action dec :parameters () :precondition (and (> (x) 0)) :effect (and (decrease (x) 1)))
 (:action finish :parameters () :precondition (and (= (x) 3) (not (done))) :effect (and (done)))
)
)",
                                      R"(
(define (problem p) (:domain counter2)
 (:init (= (x) 0))
 (:goal (and (done) (= (x) 0))))
)");
    for (u32 T : {1u, 2u})
    {
        const StateSpacePtr S = space_of(bounded, ss_options(T));
        ASSERT_TRUE(S);
        EXPECT_EQ(S->num_states(), 8u);
        EXPECT_EQ(S->numeric_words(), bounded->numeric_words());
        EXPECT_EQ(S->num_goal_states(), 1u);
        EXPECT_EQ(S->unit_goal_distances()[0], 7);  // up 3, finish, down 3
        check_invariants(*S);
    }
}

TEST(StateSpace, MetricWithoutTotalCostTakesTheMetricDifference)
{
    // the metric (fuel) decreases along inc: negative transition costs, which Dijkstra rejects 
    const TaskPtr task = pddl_task(R"(
(define (domain m)
 (:requirements :strips :numeric-fluents)
 (:predicates (a) (b))
 (:functions (fuel))
 (:action ab :parameters () :precondition (and (a)) :effect (and (not (a)) (b) (increase (fuel) 3)))
)
)",
                                   R"(
(define (problem p) (:domain m)
 (:init (a) (= (fuel) 1))
 (:goal (b))
 (:metric minimize (fuel)))
)");
    const StateSpacePtr S = space_of(task, ss_options());
    ASSERT_TRUE(S);
    ASSERT_EQ(S->num_transitions(), 1u);
    EXPECT_DOUBLE_EQ(S->cost(0), 3.0);
    EXPECT_DOUBLE_EQ(S->cost_goal_distances()[0], 3.0);
    const TaskPtr negative = pddl_task(R"(
(define (domain m2)
 (:requirements :strips :numeric-fluents)
 (:predicates (a) (b))
 (:functions (fuel))
 (:action ab :parameters () :precondition (and (a)) :effect (and (not (a)) (b) (decrease (fuel) 3)))
)
)",
                                       R"(
(define (problem p) (:domain m2)
 (:init (a) (= (fuel) 1))
 (:goal (b))
 (:metric minimize (fuel)))
)");
    EXPECT_THROW((void)generate_state_space(negative, ss_options()), std::domain_error);
}

TEST(StateSpace, TimeoutStops)
{
    const auto task = Task::from_text_file(test::task_path("depot__p02"));
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o = ss_options(T);
        o.max_seconds = 0;
        EXPECT_EQ(generate_state_space(task, o).status, search::SearchStatus::OutOfTime) << T;
    }
}

// ----------------------------------------------------------------------------------------------- pool
TEST(StateSpace, PoolEqualsSingleGenerations)
{
    std::vector<TaskPtr> tasks;
    for (const char* dir : {"gripper", "spanner", "blocks_3", "miconic", "delivery", "visitall"})
        if (TaskPtr t = small_task(dir, "test_problem.pddl"))
            tasks.push_back(t);
    if (tasks.size() < 6)
        GTEST_SKIP();
    const std::vector<StateSpaceResult> rs = generate_state_spaces(tasks, ss_options(), 3);
    ASSERT_EQ(rs.size(), tasks.size());
    for (usize i = 0; i < tasks.size(); ++i)
    {
        ASSERT_TRUE(rs[i].space);
        EXPECT_EQ(rs[i].space->task(), tasks[i]);
        EXPECT_EQ(arrays_hash(*rs[i].space), arrays_hash(*space_of(tasks[i], ss_options())));
    }
    // exceptions stop the pool and reach the caller
    EXPECT_THROW(for_each_state_space(
                     8, [&](u64 i) -> TaskPtr { if (i == 5) throw std::runtime_error("boom"); return tasks[i % tasks.size()]; },
                     [](u64, StateSpaceResult&&) {}, ss_options(), 3),
                 std::runtime_error);
}

// ----------------------------------------------------------------------------------------------- samplers
TEST(Sampler, DeterministicPerSeedAndCorrectSupports)
{
    const TaskPtr task = small_task("spanner", "p15-easy.pddl");
    if (!task)
        GTEST_SKIP();
    const StateSpacePtr S = space_of(task, ss_options());
    ASSERT_TRUE(S);
    StateSpaceSampler a(S, 7), b(S, 7), c(S, 8);
    std::vector<u32> xa(500), xb(500), xc(500);
    a.sample_states(xa);
    b.sample_states(xb);
    c.sample_states(xc);
    EXPECT_EQ(xa, xb);
    EXPECT_NE(xa, xc);
    a.set_seed(7);
    std::vector<u32> again(500);
    a.sample_states(again);
    EXPECT_EQ(again, xa);
    EXPECT_EQ(a.num_states(), S->num_states());
    EXPECT_EQ(a.num_unsolvable_states(), S->num_unsolvable_states());
    EXPECT_EQ(a.num_solvable_states(), S->num_states() - S->num_unsolvable_states());
    EXPECT_EQ(static_cast<i32>(a.max_steps_to_goal()), S->max_goal_distance());
    for (i32 n = 0; n <= S->max_goal_distance(); ++n)
    {
        std::set<u32> hits;
        for (int i = 0; i < 2000; ++i)
        {
            const u32 s = a.sample_state_n_steps_from_goal(n);
            EXPECT_EQ(S->unit_goal_distances()[s], n);
            hits.insert(s);
        }
        const auto support = a.states_n_steps_from_goal(n);
        EXPECT_EQ(hits.size(), std::min<usize>(support.size(), hits.size()));
        EXPECT_TRUE(std::is_sorted(support.begin(), support.end()));
    }
    for (int i = 0; i < 1000; ++i)
        EXPECT_TRUE(S->is_unsolvable(a.sample_unsolvable_state()));
    EXPECT_THROW((void)a.sample_state_n_steps_from_goal(S->max_goal_distance() + 1), std::out_of_range);
    EXPECT_THROW((void)a.sample_state_n_steps_from_goal(-1), std::out_of_range);
    // uniform over all states: every state of a small space shows up
    std::set<u32> all(xa.begin(), xa.end());
    std::vector<u32> many(20000);
    a.sample_states(many);
    all.insert(many.begin(), many.end());
    EXPECT_EQ(all.size(), S->num_states());
}

TEST(Sampler, NoDeadEnds)
{
    const TaskPtr task = small_task("gripper", "p-2-0.pddl");
    if (!task)
        GTEST_SKIP();
    StateSpaceSampler s(space_of(task, {}), 1);
    EXPECT_EQ(s.num_unsolvable_states(), 0u);
    EXPECT_THROW((void)s.sample_unsolvable_state(), std::out_of_range);
}

// ----------------------------------------------------------------------------------------------- generalized
TEST(GeneralizedStateSpace, DisjointUnion)
{
    std::vector<TaskPtr> tasks;
    for (const char* p : {"test_problem4.pddl", "p-2-0.pddl", "test_problem2.pddl", "p-1-0.pddl"})
        if (TaskPtr t = small_task("gripper", p))
            tasks.push_back(t);
    if (tasks.size() < 4)
        GTEST_SKIP();
    // the disjoint union, spaces sorted by size (stable)
    const auto results = generate_state_spaces(tasks, ss_options(), 2);
    const auto spaces = ordered_spaces(results);
    ASSERT_EQ(spaces.size(), 4u);
    for (usize i = 1; i < spaces.size(); ++i)
        EXPECT_LE(spaces[i - 1]->num_states(), spaces[i]->num_states());
    EXPECT_EQ(spaces[1]->task(), tasks[1]);  // p-2-0 and test_problem2 tie at 28 states: input order
    EXPECT_EQ(spaces[2]->task(), tasks[2]);
    const auto G = GeneralizedStateSpace::create(spaces);
    EXPECT_EQ(G->spaces(), spaces);
    u64 V = 0, E = 0, goals = 0, unsolvable = 0;
    for (const auto& s : spaces)
        V += s->num_states(), E += s->num_transitions(), goals += s->num_goal_states(), unsolvable += s->num_unsolvable_states();
    EXPECT_EQ(G->num_vertices(), V);
    EXPECT_EQ(G->num_edges(), E);
    ASSERT_EQ(G->vertex_offsets().size(), 5u);
    ASSERT_EQ(G->edge_offsets().size(), 5u);
    ASSERT_EQ(G->forward_offsets().size(), V + 1);
    ASSERT_EQ(G->forward_targets().size(), E);
    EXPECT_EQ(G->goal_vertices().size(), goals);
    EXPECT_EQ(G->unsolvable_vertices().size(), unsolvable);
    std::vector<u32> initial;
    for (u32 p = 0; p < 4; ++p)
    {
        const StateSpace& S = *spaces[p];
        EXPECT_EQ(G->vertex_offsets()[p + 1] - G->vertex_offsets()[p], S.num_states());
        EXPECT_EQ(G->edge_offsets()[p + 1] - G->edge_offsets()[p], S.num_transitions());
        initial.push_back(G->vertex(p, S.initial_state()));
        for (u32 v = 0; v < S.num_states(); ++v)
        {
            const u32 g = G->vertex(p, v);
            EXPECT_EQ(g, G->vertex_offsets()[p] + v);
            EXPECT_EQ(G->problem_of(g), p);
            EXPECT_EQ(G->goal_flags()[g], S.goal_flags()[v]);
            EXPECT_EQ(G->unsolvable_flags()[g], S.unsolvable_flags()[v]);
            // the edges of vertex g are the transitions of state v, in their order
            ASSERT_EQ(G->forward_offsets()[g + 1] - G->forward_offsets()[g], S.forward_offsets()[v + 1] - S.forward_offsets()[v]);
            for (u64 e = S.forward_offsets()[v]; e < S.forward_offsets()[v + 1]; ++e)
            {
                const u64 ge = G->edge(p, e);
                EXPECT_EQ(ge, G->forward_offsets()[g] + (e - S.forward_offsets()[v]));
                EXPECT_EQ(G->forward_targets()[ge], G->vertex(p, S.forward_targets()[e]));
                EXPECT_EQ(G->source(ge), g);
            }
        }
    }
    EXPECT_EQ(G->initial_vertices(), initial);
    EXPECT_THROW((void)G->vertex(4, 0), std::out_of_range);
    EXPECT_THROW((void)G->vertex(0, spaces[0]->num_states()), std::out_of_range);
    EXPECT_THROW((void)G->edge(0, spaces[0]->num_transitions()), std::out_of_range);
    EXPECT_THROW((void)G->problem_of(G->num_vertices()), std::out_of_range);
    EXPECT_THROW((void)G->source(G->num_edges()), std::out_of_range);
    // no spaces: an empty graph
    const auto empty = GeneralizedStateSpace::create({});
    EXPECT_EQ(empty->num_vertices(), 0u);
    EXPECT_EQ(empty->num_edges(), 0u);
    EXPECT_EQ(empty->forward_offsets().size(), 1u);
    // one domain only
    const TaskPtr other = small_task("blocks_3", "test_problem.pddl");
    ASSERT_TRUE(other);
    EXPECT_THROW((void)GeneralizedStateSpace::create({spaces[0], space_of(other, ss_options())}), std::invalid_argument);
}

// ----------------------------------------------------------------------------------------------- from arrays
void expect_same_space(const StateSpace& a, const StateSpace& b)
{
    auto eq = [](auto x, auto y) { return std::ranges::equal(x, y); };
    EXPECT_EQ(a.task(), b.task());
    EXPECT_EQ(a.num_states(), b.num_states());
    EXPECT_EQ(a.words(), b.words());
    EXPECT_EQ(a.numeric_words(), b.numeric_words());
    EXPECT_EQ(a.has_labels(), b.has_labels());
    EXPECT_EQ(a.label_width(), b.label_width());
    EXPECT_TRUE(eq(a.state_words(), b.state_words()));
    EXPECT_TRUE(eq(a.forward_offsets(), b.forward_offsets()));
    EXPECT_TRUE(eq(a.forward_targets(), b.forward_targets()));
    EXPECT_TRUE(eq(a.label_schemas(), b.label_schemas()));
    EXPECT_TRUE(eq(a.label_bindings(), b.label_bindings()));
    EXPECT_TRUE(eq(a.costs(), b.costs()));
    EXPECT_TRUE(eq(a.backward_offsets(), b.backward_offsets()));
    EXPECT_TRUE(eq(a.backward_sources(), b.backward_sources()));
    EXPECT_TRUE(eq(a.backward_edges(), b.backward_edges()));
    EXPECT_TRUE(eq(a.unit_goal_distances(), b.unit_goal_distances()));
    EXPECT_TRUE(eq(a.cost_goal_distances(), b.cost_goal_distances()));
    EXPECT_TRUE(eq(a.goal_flags(), b.goal_flags()));
    EXPECT_TRUE(eq(a.unsolvable_flags(), b.unsolvable_flags()));
    EXPECT_TRUE(eq(a.alive_flags(), b.alive_flags()));
    EXPECT_EQ(a.num_goal_states(), b.num_goal_states());
    EXPECT_EQ(a.num_unsolvable_states(), b.num_unsolvable_states());
    EXPECT_EQ(a.max_goal_distance(), b.max_goal_distance());
    EXPECT_EQ(a.threads(), b.threads());
    EXPECT_EQ(a.layers(), b.layers());
    EXPECT_EQ(a.search_seconds(), b.search_seconds());
    EXPECT_EQ(a.post_seconds(), b.post_seconds());
}

const char* kCostProblem = R"(
(define (problem p) (:domain count)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c0) (next c0 c1) (next c1 c2) (next c2 c3) (next c0 c3) (= (total-cost) 0)
        (= (step-cost c0 c1) 1) (= (step-cost c1 c2) 2) (= (step-cost c2 c3) 3) (= (step-cost c0 c3) 7))
 (:goal (at c3))
 (:metric minimize (total-cost)))
)";

TEST(StateSpace, CreateFromArraysRoundTrips)
{
    // costs, numeric words, a small-instance task, lazy atom slots
    std::vector<TaskPtr> tasks{pddl_task(kCountDomain, kCostProblem), pddl_task(R"(
(define (domain counter2)
 (:requirements :strips :numeric-fluents :negative-preconditions)
 (:predicates (done))
 (:functions (x))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1)))
 (:action dec :parameters () :precondition (and (> (x) 0)) :effect (and (decrease (x) 1)))
 (:action finish :parameters () :precondition (and (= (x) 3) (not (done))) :effect (and (done)))
)
)",
                                                                                                                 R"(
(define (problem p) (:domain counter2)
 (:init (= (x) 0))
 (:goal (and (done) (= (x) 0))))
)")};
    if (TaskPtr t = small_task("spanner", "p15-easy.pddl"))
        tasks.push_back(t);
    TaskOptions lazy;
    lazy.atoms = TaskOptions::Atoms::Lazy;
    if (TaskPtr t = small_task("gripper", "p-2-0.pddl", lazy))
        tasks.push_back(t);
    for (const TaskPtr& task : tasks)
        for (bool labels : {true, false})
        {
            StateSpaceOptions o = ss_options();
            o.labels = labels;
            const StateSpacePtr S = space_of(task, o);
            ASSERT_TRUE(S);
            const StateSpacePtr R = StateSpace::create(S->to_arrays());
            expect_same_space(*S, *R);
            for (u32 i = 0; i < S->num_states(); i += std::max<u32>(1, S->num_states() / 11))
                EXPECT_EQ(R->find(S->state(i)), i);
            if (labels)
            {
                EXPECT_EQ(R->label(S->num_transitions() - 1), S->label(S->num_transitions() - 1));
            }
        }
}

TEST(StateSpace, CreateRejectsArraysThatAreNotASpaceOfTheTask)
{
    const TaskPtr task = pddl_task(kCountDomain, kCostProblem);
    const StateSpacePtr S = space_of(task, ss_options());
    ASSERT_TRUE(S);
    ASSERT_GE(S->num_transitions(), 2u);
    ASSERT_LT(task->atoms().fluent_slots(), 64u * S->words());
    // another problem of the domain: its initial state differs
    const TaskPtr other = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c1) (next c0 c1) (next c1 c2) (next c2 c3) (next c0 c3) (= (total-cost) 0)
        (= (step-cost c0 c1) 1) (= (step-cost c1 c2) 2) (= (step-cost c2 c3) 3) (= (step-cost c0 c3) 7))
 (:goal (at c3))
 (:metric minimize (total-cost)))
)");
    const u32 N = S->num_states(), RW = S->row_words(), alive = [&] {
        for (u32 i = 0; i < N; ++i)
            if (S->is_alive(i))
                return i;
        return 0u;
    }();
    ASSERT_TRUE(S->is_alive(alive));
    struct Case
    {
        const char* what;  // part of the message
        std::function<void(StateSpaceArrays&)> mutate;
    };
    const Case cases[] = {
        {"no task", [](StateSpaceArrays& a) { a.task = nullptr; }},
        {"initial state", [&](StateSpaceArrays& a) { a.task = other; }},
        {"no states", [](StateSpaceArrays& a) {
             StateSpaceArrays empty;
             empty.task = a.task;
             a = std::move(empty);
         }},
        {"state_words", [](StateSpaceArrays& a) { a.state_words.pop_back(); }},
        {"target is not a state", [&](StateSpaceArrays& a) { a.forward_targets[0] = N; }},
        {"backward CSR", [](StateSpaceArrays& a) { std::swap(a.backward_edges[0], a.backward_edges[1]); }},
        {"backward CSR", [](StateSpaceArrays& a) { a.backward_sources[0] = a.backward_sources[0] == 0 ? 1 : 0; }},
        {"words is not the task's", [&](StateSpaceArrays& a) {
             a.words += 1;
             a.state_words.assign(u64{N} * (RW + 1), 0);
         }},
        {"numeric_words", [&](StateSpaceArrays& a) {
             a.numeric_words = 1;
             a.state_words.assign(u64{N} * (RW + 1), 0);
         }},
        {"atom slot", [&](StateSpaceArrays& a) { a.state_words[u64{N - 1} * RW] |= u64{1} << 63; }},
        {"schema", [](StateSpaceArrays& a) { a.label_schemas[0] = 99; }},
        {"binds an object", [](StateSpaceArrays& a) { a.label_bindings[0] = 99; }},
        {"label_width", [](StateSpaceArrays& a) {
             a.label_width = 1;
             a.label_bindings.resize(a.label_schemas.size());
         }},
        {"cost is NaN", [](StateSpaceArrays& a) { a.costs[0] = std::nan(""); }},
        {"flag is not 0 or 1", [](StateSpaceArrays& a) { a.goal_flags[0] = 2; }},
        {"goal and unsolvable flags", [](StateSpaceArrays& a) { a.goal_flags[0] = 1; }},
        {"goal and unsolvable flags", [&](StateSpaceArrays& a) { a.unsolvable_flags[alive] = 1; }},
        {"cost goal distances", [&](StateSpaceArrays& a) { a.cost_goal_distances[alive] = std::numeric_limits<f64>::infinity(); }},
        {"alive flags", [&](StateSpaceArrays& a) { a.alive_flags[alive] = 0; }},
    };
    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.what);
        StateSpaceArrays a = S->to_arrays();
        c.mutate(a);
        try
        {
            (void)StateSpace::create(std::move(a));
            ADD_FAILURE() << "accepted";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find(c.what), std::string::npos) << e.what();
        }
    }
    // ArraysCheck::Structure checks sizes and ranges only: contents of another task pass, broken ranges do not
    StateSpaceArrays a = S->to_arrays();
    a.task = other;
    a.goal_flags[0] = 1;
    EXPECT_EQ(StateSpace::create(std::move(a), ArraysCheck::Structure)->num_goal_states(), S->num_goal_states() + 1);
    StateSpaceArrays b = S->to_arrays();
    b.forward_targets[0] = N;
    EXPECT_THROW((void)StateSpace::create(std::move(b), ArraysCheck::Structure), std::invalid_argument);
}
}  // namespace
