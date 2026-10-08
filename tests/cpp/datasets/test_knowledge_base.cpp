// Knowledge bases: the fork's knowledge base tests (tests/unit/datasets/knowledge_base.cpp: spanner and gripper,
// with and without symmetry reduction, tuple graphs of width 0, 1 and 2), the order of the state spaces and the
// options.

#include "../frontend/golden.hpp"
#include "mymyr/datasets/knowledge_base.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
rl::TaskTablePtr fork_table(const std::string& dir, std::initializer_list<const char*> problems)
{
    const auto root = test::fork_data_dir() / dir;
    std::vector<TaskPtr> tasks;
    const auto domain = frontend::Domain::from_file(root / "domain.pddl");
    for (const char* p : problems)
    {
        if (!std::filesystem::exists(root / p))
            return nullptr;
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        tasks.push_back(Task::create(*domain->instantiate_file(root / p), to));
    }
    return rl::TaskTable::create(std::move(tasks));
}

struct Totals
{
    u64 vertices = 0, edges = 0;
};
Totals totals(const KnowledgeBase& kb)
{
    Totals t;
    for (const auto& graphs : kb.tuple_graphs())
        for (const TupleGraph& g : graphs)
            t.vertices += g.num_vertices(), t.edges += g.num_edges();
    return t;
}

KnowledgeBaseOptions options(bool symmetry, u32 width)
{
    KnowledgeBaseOptions o;
    o.state_space.symmetry_pruning = symmetry;
    o.generalized = true;
    o.tuple_graphs = TupleGraphOptions{.width = width};
    return o;
}

// The spanner is at location 1 in one problem and at location 2 in the other: no symmetry within a problem, but
// between the two.
TEST(KnowledgeBase, SpannerSymmetryBetweenProblems)
{
    if (!std::filesystem::exists(test::fork_data_dir() / "spanner/domain.pddl"))
        GTEST_SKIP();
    const auto table = fork_table("spanner", {"p-1-1-2-1.pddl", "p-1-1-2-1(2).pddl"});
    ASSERT_TRUE(table);
    {
        const auto kb = KnowledgeBase::create(table, options(false, 2));
        const auto& G = *kb->generalized_state_space();
        EXPECT_EQ(G.num_vertices(), 15u);
        EXPECT_EQ(G.num_edges(), 13u);
        EXPECT_EQ(G.initial_vertices().size(), 2u);
        EXPECT_EQ(G.goal_vertices().size(), 2u);
        EXPECT_EQ(G.unsolvable_vertices().size(), 3u);
        ASSERT_EQ(kb->tuple_graphs().size(), 2u);
        EXPECT_EQ(kb->tuple_graphs()[0].size(), 7u);
        EXPECT_EQ(kb->tuple_graphs()[1].size(), 8u);
        EXPECT_EQ(totals(*kb).vertices, 53u);
        EXPECT_EQ(totals(*kb).edges, 38u);
        // ascending by size: the second problem (7 states) first
        EXPECT_EQ(std::vector<u32>(kb->task_indices().begin(), kb->task_indices().end()), (std::vector<u32>{1, 0}));
    }
    {
        const auto kb = KnowledgeBase::create(table, options(true, 2));
        const auto& G = *kb->generalized_state_space();
        EXPECT_TRUE(G.symmetry_reduced());
        EXPECT_EQ(G.num_vertices(), 12u);
        EXPECT_EQ(G.num_edges(), 11u);
        EXPECT_EQ(G.initial_vertices().size(), 2u);
        EXPECT_EQ(G.goal_vertices().size(), 1u);
        EXPECT_EQ(G.unsolvable_vertices().size(), 3u);
        ASSERT_EQ(kb->tuple_graphs().size(), 2u);
        EXPECT_EQ(kb->tuple_graphs()[0].size(), 7u);
        EXPECT_EQ(kb->tuple_graphs()[1].size(), 8u);
        EXPECT_EQ(totals(*kb).vertices, 53u);
        EXPECT_EQ(totals(*kb).edges, 38u);
    }
}

// One and two balls: symmetry within a problem, none between the two.
TEST(KnowledgeBase, GripperSymmetryWithinProblems)
{
    if (!std::filesystem::exists(test::fork_data_dir() / "gripper/domain.pddl"))
        GTEST_SKIP();
    const auto table = fork_table("gripper", {"p-1-0.pddl", "p-2-0.pddl"});
    ASSERT_TRUE(table);
    struct Expect
    {
        bool symmetry;
        u32 width;
        u32 vertices, edges, goals;
        usize graphs0, graphs1;
        u64 tg_vertices, tg_edges;
    };
    for (const Expect& e : {Expect{false, 1, 36, 128, 4, 8, 28, 220, 184}, Expect{true, 1, 18, 52, 4, 6, 12, 76, 70},
                            Expect{false, 0, 36, 128, 4, 8, 28, 128, 92}, Expect{true, 0, 18, 52, 4, 6, 12, 52, 34}})
    {
        SCOPED_TRACE(std::string(e.symmetry ? "symmetry" : "no symmetry") + " width " + std::to_string(e.width));
        const auto kb = KnowledgeBase::create(table, options(e.symmetry, e.width));
        const auto& G = *kb->generalized_state_space();
        EXPECT_EQ(G.num_vertices(), e.vertices);
        EXPECT_EQ(G.num_edges(), e.edges);
        EXPECT_EQ(G.initial_vertices().size(), 2u);
        EXPECT_EQ(G.goal_vertices().size(), e.goals);
        EXPECT_EQ(G.unsolvable_vertices().size(), 0u);
        ASSERT_EQ(kb->tuple_graphs().size(), 2u);
        EXPECT_EQ(kb->tuple_graphs()[0].size(), e.graphs0);
        EXPECT_EQ(kb->tuple_graphs()[1].size(), e.graphs1);
        EXPECT_EQ(totals(*kb).vertices, e.tg_vertices);
        EXPECT_EQ(totals(*kb).edges, e.tg_edges);
    }
}

TEST(KnowledgeBase, PartsAndOrder)
{
    if (!std::filesystem::exists(test::fork_data_dir() / "gripper/domain.pddl"))
        GTEST_SKIP();
    const auto table = fork_table("gripper", {"test_problem4.pddl", "p-2-0.pddl", "p-1-0.pddl"});
    ASSERT_TRUE(table);
    // the defaults: state spaces only, ascending by size
    const auto kb = KnowledgeBase::create(table);
    EXPECT_FALSE(kb->generalized_state_space());
    EXPECT_FALSE(kb->has_tuple_graphs());
    EXPECT_TRUE(kb->tuple_graphs().empty());
    ASSERT_EQ(kb->state_spaces().size(), 3u);
    EXPECT_EQ(std::vector<u32>(kb->task_indices().begin(), kb->task_indices().end()), (std::vector<u32>{2, 1, 0}));
    for (usize i = 0; i < 3; ++i)
        EXPECT_EQ(kb->state_spaces()[i]->task(), table->task(kb->task_indices()[i]));
    // in task order; one thread gives the same tuple graphs as several
    KnowledgeBaseOptions o;
    o.sort_by_size = false;
    o.tuple_graphs = TupleGraphOptions{.width = 1, .dominance_pruning = false};
    o.threads = 1;
    const auto a = KnowledgeBase::create(table, o);
    EXPECT_EQ(std::vector<u32>(a->task_indices().begin(), a->task_indices().end()), (std::vector<u32>{0, 1, 2}));
    o.threads = 4;
    const auto b = KnowledgeBase::create(table, o);
    ASSERT_EQ(a->tuple_graphs().size(), 3u);
    for (usize i = 0; i < 3; ++i)
    {
        EXPECT_EQ(a->state_spaces()[i]->num_states(), b->state_spaces()[i]->num_states());
        ASSERT_EQ(a->tuple_graphs()[i].size(), a->state_spaces()[i]->num_states());
        for (usize v = 0; v < a->tuple_graphs()[i].size(); ++v)
        {
            const TupleGraph &x = a->tuple_graphs()[i][v], &y = b->tuple_graphs()[i][v];
            EXPECT_EQ(x.num_vertices(), y.num_vertices());
            EXPECT_TRUE(std::ranges::equal(x.tuple_atoms(), y.tuple_atoms()));
            EXPECT_TRUE(std::ranges::equal(x.successor_ids(), y.successor_ids()));
            EXPECT_FALSE(x.dominance_pruning());
        }
    }
    // failed generations are skipped
    KnowledgeBaseOptions small;
    small.state_space.max_states = 30;
    const auto c = KnowledgeBase::create(table, small);
    EXPECT_EQ(std::vector<u32>(c->task_indices().begin(), c->task_indices().end()), (std::vector<u32>{2, 1}));
    // errors
    EXPECT_THROW((void)KnowledgeBase::create(nullptr), std::invalid_argument);
    KnowledgeBaseOptions wide;
    wide.tuple_graphs = TupleGraphOptions{.width = 6};
    EXPECT_THROW((void)KnowledgeBase::create(table, wide), std::invalid_argument);
}
}  // namespace
