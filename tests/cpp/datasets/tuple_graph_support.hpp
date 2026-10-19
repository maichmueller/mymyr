#pragma once
// Test support of the tuple-graph tests: breadth-first distances over a state space, the structural invariants of a
// tuple graph, and a generator of the full space of a task.

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace mymyr::test::tg
{
using namespace mymyr::datasets;

inline StateSpacePtr space_of(TaskPtr task)
{
    StateSpaceOptions o;
    o.remove_if_unsolvable = false;
    StateSpaceResult r = generate_state_space(std::move(task), o);
    EXPECT_EQ(r.status, search::SearchStatus::Exhausted);
    return r.space;
}

/// The breadth-first distances from r over the space's forward graph (-1: unreachable).
inline std::vector<i64> bfs(const StateSpace& S, u32 r)
{
    std::vector<i64> dist(S.num_states(), -1);
    std::vector<u32> layer{r}, next;
    dist[r] = 0;
    for (i64 d = 1; !layer.empty(); ++d)
    {
        next.clear();
        for (u32 p : layer)
            for (u64 e = S.forward_offsets()[p]; e < S.forward_offsets()[p + 1]; ++e)
                if (const u32 c = S.forward_targets()[e]; dist[c] < 0)
                {
                    dist[c] = d;
                    next.push_back(c);
                }
        std::swap(layer, next);
    }
    return dist;
}

/// The structural invariants of a tuple graph.
inline void check_invariants(const TupleGraph& g)
{
    const StateSpace& S = *g.space();
    const auto dist = bfs(S, g.root());
    ASSERT_GE(g.num_distances(), 1u);
    EXPECT_EQ(g.distance_offsets().front(), 0u);
    EXPECT_EQ(g.distance_offsets().back(), g.num_vertices());
    EXPECT_TRUE(g.dominance_pruning() || g.width() == 0 ? g.vertices_at(0).size() == 1 : g.vertices_at(0).size() >= 1);
    u64 edges = 0;
    for (u32 d = 0; d < g.num_distances(); ++d)
    {
        EXPECT_FALSE(g.vertices_at(d).empty()) << d;
        const auto layer = g.problem_vertices_at(d);
        EXPECT_TRUE(std::ranges::is_sorted(layer));
        for (u32 p : layer)
            EXPECT_EQ(dist[p], d);
        for (u32 v : g.vertices_at(d))
        {
            EXPECT_EQ(g.distance(v), d);
            EXPECT_LE(g.tuple(v).size(), g.width());
            EXPECT_TRUE(std::ranges::is_sorted(g.tuple(v)));
            EXPECT_FALSE(g.problem_vertices(v).empty());
            EXPECT_TRUE(std::ranges::is_sorted(g.problem_vertices(v)));
            for (u32 p : g.problem_vertices(v))
            {
                EXPECT_EQ(dist[p], d);
                for (u32 a : g.tuple(v))
                    EXPECT_TRUE(S.state(p).contains(SlotId{a}));
            }
            EXPECT_EQ(d == 0, g.predecessors(v).empty());
            for (u32 u : g.predecessors(v))
                EXPECT_EQ(g.distance(u), d - 1);
            for (u32 w : g.successors(v))
                EXPECT_EQ(g.distance(w), d + 1);
            edges += g.successors(v).size();
        }
    }
    EXPECT_EQ(edges, g.num_edges());
}
}  // namespace mymyr::test::tg
