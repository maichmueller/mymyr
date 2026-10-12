// Knowledge bases: spanner and gripper, tuple graphs of width 0, 1 and 2, the order of the state spaces and the options.

#include "../support/benchmarks.hpp"
#include "mymyr/datasets/knowledge_base.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
rl::TaskTablePtr small_table(const std::string& dir, std::initializer_list<const char*> problems)
{
    const auto root = test::small_dir() / dir;
    std::vector<TaskPtr> tasks;
    const auto domain = frontend::Domain::from_file(root / "domain.pddl");
    for (const char* p : problems)
    {
        if (!std::filesystem::exists(root / p))
            return nullptr;
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Dense;
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

KnowledgeBaseOptions options(u32 width)
{
    KnowledgeBaseOptions o;
    o.generalized = true;
    o.tuple_graphs = TupleGraphOptions{.width = width};
    return o;
}

// The spanner at location 1 in one problem and at location 2 in the other.
TEST(KnowledgeBase, Spanner)
{
    const auto table = small_table("spanner", {"p-1-1-2-1.pddl", "p-1-1-2-1(2).pddl"});
    ASSERT_TRUE(table);
    const auto kb = KnowledgeBase::create(table, options(2));
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
    // ascending by size: the second problem (7 states) first; the generalized state space's problems are the spaces
    EXPECT_EQ(std::vector<u32>(kb->task_indices().begin(), kb->task_indices().end()), (std::vector<u32>{1, 0}));
    EXPECT_EQ(G.spaces(), kb->state_spaces());
}

// One and two balls.
TEST(KnowledgeBase, Gripper)
{
    const auto table = small_table("gripper", {"p-1-0.pddl", "p-2-0.pddl"});
    ASSERT_TRUE(table);
    struct Expect
    {
        u32 width;
        u32 vertices, edges, goals;
        usize graphs0, graphs1;
        u64 tg_vertices, tg_edges;
    };
    for (const Expect& e : {Expect{1, 36, 128, 4, 8, 28, 220, 184}, Expect{0, 36, 128, 4, 8, 28, 128, 92}})
    {
        SCOPED_TRACE("width " + std::to_string(e.width));
        const auto kb = KnowledgeBase::create(table, options(e.width));
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
    const auto table = small_table("gripper", {"test_problem4.pddl", "p-2-0.pddl", "p-1-0.pddl"});
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

TEST(KnowledgeBase, FromParts)
{
    const auto table = small_table("gripper", {"test_problem4.pddl", "p-2-0.pddl", "p-1-0.pddl"});
    ASSERT_TRUE(table);
    const auto kb = KnowledgeBase::create(table, options(1));
    auto parts = [&] {
        return KnowledgeBaseParts{kb->state_spaces(), {kb->task_indices().begin(), kb->task_indices().end()}, kb->tuple_graphs()};
    };
    const auto back = KnowledgeBase::create(table, kb->options(), parts());
    EXPECT_EQ(back->state_spaces(), kb->state_spaces());
    EXPECT_TRUE(std::ranges::equal(back->task_indices(), kb->task_indices()));
    ASSERT_TRUE(back->generalized_state_space());
    EXPECT_TRUE(std::ranges::equal(back->generalized_state_space()->forward_targets(), kb->generalized_state_space()->forward_targets()));
    EXPECT_EQ(back->tuple_graphs(), kb->tuple_graphs());
    // without tuple graphs or the generalized state space
    KnowledgeBaseOptions plain;
    const auto bare = KnowledgeBase::create(table, plain, KnowledgeBaseParts{kb->state_spaces(), parts().task_indices, {}});
    EXPECT_FALSE(bare->generalized_state_space());
    EXPECT_FALSE(bare->has_tuple_graphs());

    struct Case
    {
        const char* what;  // part of the message
        std::function<void(KnowledgeBaseParts&)> mutate;
    };
    const Case cases[] = {
        {"one per state space", [](KnowledgeBaseParts& p) { p.task_indices.pop_back(); }},
        {"out of range or repeated", [](KnowledgeBaseParts& p) { p.task_indices[0] = 3; }},
        {"out of range or repeated", [](KnowledgeBaseParts& p) { p.task_indices[1] = p.task_indices[0]; }},
        {"not of its task", [](KnowledgeBaseParts& p) { std::swap(p.task_indices[0], p.task_indices[1]); }},
        {"knowledge-base order", [](KnowledgeBaseParts& p) {
             std::ranges::reverse(p.state_spaces);
             std::ranges::reverse(p.task_indices);
             std::ranges::reverse(p.tuple_graphs);
         }},
        {"one list per state space", [](KnowledgeBaseParts& p) { p.tuple_graphs.pop_back(); }},
        {"one tuple graph per state", [](KnowledgeBaseParts& p) { p.tuple_graphs[0].pop_back(); }},
        {"not of its state", [](KnowledgeBaseParts& p) { std::swap(p.tuple_graphs[1][0], p.tuple_graphs[1][1]); }},
    };
    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.what);
        KnowledgeBaseParts p = parts();
        c.mutate(p);
        try
        {
            (void)KnowledgeBase::create(table, kb->options(), std::move(p));
            ADD_FAILURE() << "accepted";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find(c.what), std::string::npos) << e.what();
        }
    }
    EXPECT_THROW((void)KnowledgeBase::create(table, plain, parts()), std::invalid_argument);  // tuple graphs without options
    EXPECT_THROW((void)KnowledgeBase::create(nullptr, plain, {}), std::invalid_argument);
}
}  // namespace
