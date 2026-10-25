#pragma once

#include "mymyr/search/iw.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <utility>

namespace mymyr::test
{
struct TreeObserver : search::SearchObserver
{
    struct Edge
    {
        u64 parent, child;
        Action action;
        State state;
        search::TransitionOutcome outcome;
    };
    std::vector<std::vector<Edge>> passes;
    std::vector<Edge> edges;

    void on_transition(u64 parent, const Action& action, u64 child, StateView state, search::TransitionOutcome outcome) override
    {
        edges.push_back({parent, child, action, State(state), outcome});
    }
    void on_pass(u32, const search::SearchStatistics&) override { passes.push_back(std::exchange(edges, {})); }
};

inline void expect_tree_events(const Task& task, const search::IwPassStatistics& pass, const std::vector<TreeObserver::Edge>& events)
{
    ASSERT_TRUE(pass.tree);
    const auto& t = *pass.tree;
    ASSERT_EQ(t.num_transitions(), pass.generated);
    ASSERT_EQ(t.num_transitions(), events.size());
    ASSERT_EQ(t.offsets.size(), t.num_nodes() + 1);
    EXPECT_EQ(t.parent[0], -1);
    EXPECT_EQ(t.offsets.back(), static_cast<i64>(events.size()));
    const usize R = t.words + t.numeric_words;
    usize ti = 0;
    for (usize p = 0; p < t.num_nodes(); ++p)
        for (const auto& e : events)
            if (e.parent == p)
            {
                EXPECT_GE(static_cast<i64>(ti), t.offsets[p]);
                EXPECT_LT(static_cast<i64>(ti), t.offsets[p + 1]);
                EXPECT_EQ(t.t_schema[ti], static_cast<i32>(e.action.schema.v));
                EXPECT_EQ(t.t_child[ti], e.child == ~u64{0} ? -1 : static_cast<i32>(e.child));
                EXPECT_EQ(t.t_outcome[ti], static_cast<u8>(e.outcome));
                const u64* r = t.rows.data() + static_cast<usize>(t.t_row[ti]) * R;
                EXPECT_EQ((StateView{r, t.words, r + t.words, t.numeric_words}), e.state.view());
                for (usize j = 0; j < t.label_width; ++j)
                    EXPECT_EQ(t.t_binding[ti * t.label_width + j],
                              j < e.action.binding.size() ? static_cast<i32>(e.action.binding[j].v) : -1);
                ++ti;
            }
    for (usize i = 0; i < t.row_goal.size(); ++i)
    {
        const u64* r = t.rows.data() + i * R;
        EXPECT_EQ(t.row_goal[i] != 0, task.is_goal({r, t.words, r + t.words, t.numeric_words}));
    }
}

inline void expect_same_tree(const search::SearchTree& a, const search::SearchTree& b)
{
    EXPECT_EQ(a.width, b.width);
    EXPECT_EQ(a.words, b.words);
    EXPECT_EQ(a.numeric_words, b.numeric_words);
    EXPECT_EQ(a.label_width, b.label_width);
    EXPECT_EQ(a.rows, b.rows);
    EXPECT_EQ(a.parent, b.parent);
    EXPECT_EQ(a.depth, b.depth);
    EXPECT_EQ(a.schema, b.schema);
    EXPECT_EQ(a.binding, b.binding);
    EXPECT_EQ(a.node_flags, b.node_flags);
    EXPECT_EQ(a.offsets, b.offsets);
    EXPECT_EQ(a.t_schema, b.t_schema);
    EXPECT_EQ(a.t_binding, b.t_binding);
    EXPECT_EQ(a.t_row, b.t_row);
    EXPECT_EQ(a.t_child, b.t_child);
    EXPECT_EQ(a.t_outcome, b.t_outcome);
    EXPECT_EQ(a.row_goal, b.row_goal);
}
}  // namespace mymyr::test
