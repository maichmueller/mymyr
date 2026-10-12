// Tuple graphs: the definition on small tasks and independence of the thread count.

#include "../support/benchmarks.hpp"
#include "support.hpp"
#include "tuple_graph_support.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;
using namespace mymyr::test::tg;

namespace
{
TaskPtr small_task(const std::string& dir, const std::string& problem)
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Dense;
    return test::ss::dense_task(test::small_dir(), dir, problem, to);
}

// ----------------------------------------------------------------------------------------------- definition
// A chain a -> b -> c of three locations visited by one agent, one fact per location: the tuple graph of width 1
// from the start follows the chain.
constexpr const char* kChainDomain = R"((define (domain chain)
  (:requirements :strips)
  (:predicates (at ?l) (link ?a ?b) (visited ?l))
  (:action move :parameters (?a ?b)
    :precondition (and (at ?a) (link ?a ?b))
    :effect (and (at ?b) (not (at ?a)) (visited ?b))))
)";
constexpr const char* kChainProblem = R"((define (problem chain3) (:domain chain)
  (:objects a b c)
  (:init (at a) (visited a) (link a b) (link b c))
  (:goal (visited c)))
)";

TaskPtr chain_task()
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Dense;
    return Task::create(*frontend::Domain::from_string(kChainDomain, "d.pddl")->instantiate_string(kChainProblem, "p.pddl"), to);
}

TEST(TupleGraph, ChainWidthOne)
{
    const TaskPtr task = chain_task();
    const StateSpacePtr S = space_of(task);
    ASSERT_TRUE(S);
    ASSERT_EQ(S->num_states(), 3u);  // {at a}, {at b, visited b}, {at c, visited b, visited c}
    const TupleGraph g = tuple_graph(S, 0, {.width = 1});
    check_invariants(g);
    ASSERT_EQ(g.num_distances(), 3u);
    // distance 0: the empty tuple; 1: (at b) and (visited b) reach state 1, one kept; 2: (at c), (visited c)
    EXPECT_EQ(g.vertices_at(0).size(), 1u);
    EXPECT_TRUE(g.tuple(0).empty());
    ASSERT_EQ(g.vertices_at(1).size(), 1u);
    ASSERT_EQ(g.vertices_at(2).size(), 1u);
    const u32 v1 = g.distance_offsets()[1], v2 = g.distance_offsets()[2];
    // the canonical tuple: "(at b)" < "(visited b)"
    ASSERT_EQ(g.tuple(v1).size(), 1u);
    EXPECT_EQ(task->format(SlotId{g.tuple(v1)[0]}), "(at b)");
    EXPECT_EQ(task->format(SlotId{g.tuple(v2)[0]}), "(at c)");
    EXPECT_EQ(std::vector<u32>(g.successors(0).begin(), g.successors(0).end()), std::vector<u32>{v1});
    EXPECT_EQ(std::vector<u32>(g.predecessors(v2).begin(), g.predecessors(v2).end()), std::vector<u32>{v1});
    // without dominance pruning: every root tuple at distance 0 (empty, (at a), (link a b), ... are static: only
    // fluent atoms count), both tuples at 1 and 2
    const TupleGraph h = tuple_graph(S, 0, {.width = 1, .dominance_pruning = false});
    check_invariants(h);
    EXPECT_EQ(h.vertices_at(0).size(), 1u + S->state(0).count());
    EXPECT_EQ(h.vertices_at(1).size(), 2u);
    EXPECT_EQ(h.vertices_at(2).size(), 2u);
    EXPECT_EQ(h.num_edges(), h.vertices_at(0).size() * 2 + 2 * 2);
    // width 0: the root and its successor
    const TupleGraph z = tuple_graph(S, 0, {.width = 0});
    check_invariants(z);
    EXPECT_EQ(z.num_vertices(), 2u);
    EXPECT_EQ(std::vector<u32>(z.problem_vertices_at(1).begin(), z.problem_vertices_at(1).end()), std::vector<u32>{1});
    // the last state: no successor, one distance
    const TupleGraph last = tuple_graph(S, 2, {.width = 2});
    EXPECT_EQ(last.num_distances(), 1u);
    EXPECT_EQ(last.num_vertices(), 1u);
}

TEST(TupleGraph, RejectsBadArguments)
{
    const StateSpacePtr S = space_of(chain_task());
    ASSERT_TRUE(S);
    EXPECT_THROW((void)tuple_graph(S, 0, {.width = 6}), std::invalid_argument);
    EXPECT_THROW((void)tuple_graph(S, 3, {.width = 1}), std::invalid_argument);
    EXPECT_THROW((void)tuple_graphs(nullptr, {}), std::invalid_argument);
    const TupleGraph g = tuple_graph(S, 0, {.width = 1});
    EXPECT_THROW((void)g.tuple(g.num_vertices()), std::out_of_range);
    EXPECT_THROW((void)g.problem_vertices_at(g.num_distances()), std::out_of_range);
}

// ----------------------------------------------------------------------------------------------- threads
TEST(TupleGraph, IndependentOfTheThreadCount)
{
    struct Case
    {
        const char* dir;
        const char* problem;
    };
#if defined(MYMYR_SANITIZED)
    const Case cases[] = {{"gripper", "p-2-0.pddl"}, {"spanner", "p-1-1-3-1.pddl"}};
#else
    const Case cases[] = {{"gripper", "p-2-0.pddl"}, {"blocks_3", "test_problem2.pddl"}, {"delivery", "test_problem2.pddl"},
                          {"miconic-fulladl", "test_problem.pddl"}};
#endif
    for (const Case& c : cases)
    {
        const TaskPtr task = small_task(c.dir, c.problem);
        if (!task)
            GTEST_SKIP();
        const StateSpacePtr S = space_of(task);
        ASSERT_TRUE(S);
        for (u32 w : {0u, 1u, 2u})
            for (bool pruning : {true, false})
            {
                SCOPED_TRACE(std::string(c.dir) + "/" + c.problem + " width " + std::to_string(w) + (pruning ? "" : " no pruning"));
                const auto ref = tuple_graphs(S, {.width = w, .dominance_pruning = pruning, .threads = 1});
                ASSERT_EQ(ref.size(), S->num_states());
                for (u32 T : {4u, 8u})
                {
                    const auto par = tuple_graphs(S, {.width = w, .dominance_pruning = pruning, .threads = T});
                    ASSERT_EQ(par.size(), ref.size());
                    for (usize v = 0; v < ref.size(); ++v)
                        EXPECT_TRUE(par[v] == ref[v]) << "T=" << T << " vertex " << v;
                }
                for (u32 v = 0; v < S->num_states(); v += std::max<u32>(1, S->num_states() / 7))
                    EXPECT_TRUE(tuple_graph(S, v, {.width = w, .dominance_pruning = pruning}) == ref[v]);
            }
    }
}

// ----------------------------------------------------------------------------------------------- from arrays
TEST(TupleGraph, CreateFromArrays)
{
    const StateSpacePtr S = space_of(chain_task());
    ASSERT_TRUE(S);
    for (u32 w : {0u, 1u, 2u})
        for (bool pruning : {true, false})
            for (const TupleGraph& g : tuple_graphs(S, {.width = w, .dominance_pruning = pruning}))
                EXPECT_TRUE(TupleGraph::create(g.to_arrays()) == g);
    const TupleGraph g = tuple_graph(S, 0, {.width = 1, .dominance_pruning = false});
    ASSERT_EQ(g.num_distances(), 3u);
    ASSERT_GT(g.num_edges(), 1u);
    const u32 slots = S->task()->atoms().fluent_slots();
    struct Case
    {
        const char* what;  // part of the message
        std::function<void(TupleGraphArrays&)> mutate;
    };
    const Case cases[] = {
        {"no state space", [](TupleGraphArrays& a) { a.space = nullptr; }},
        {"root", [](TupleGraphArrays& a) { a.root = 3; }},
        {"width", [](TupleGraphArrays& a) { a.width = 6; }},
        {"distance_offsets", [](TupleGraphArrays& a) { a.distance_offsets[1] = 0; }},
        {"tuple_offsets", [](TupleGraphArrays& a) { a.tuple_offsets.pop_back(); }},
        {"atom slots", [&](TupleGraphArrays& a) { a.tuple_atoms.back() = slots; }},
        {"more atoms than width", [](TupleGraphArrays& a) { a.width = 0; }},
        {"layer 0", [](TupleGraphArrays& a) { a.layer_vertex_ids[0] = 1; }},
        {"not in its vertex's layer", [](TupleGraphArrays& a) { a.problem_vertex_ids.back() = 0; }},
        {"successors", [](TupleGraphArrays& a) { a.successor_ids[0] = 0; }},
        {"predecessors", [](TupleGraphArrays& a) { std::ranges::reverse(a.predecessor_ids); }},
        {"problem_offsets", [](TupleGraphArrays& a) { a.problem_offsets.back() += 1; }},
    };
    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.what);
        TupleGraphArrays a = g.to_arrays();
        c.mutate(a);
        try
        {
            (void)TupleGraph::create(std::move(a));
            ADD_FAILURE() << "accepted";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find(c.what), std::string::npos) << e.what();
        }
    }
}
}  // namespace
