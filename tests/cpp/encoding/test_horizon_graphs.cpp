// The horizon graphs made from expansions (horizon_graphs.hpp): numbering, merging, bounds, transitions, state spaces.

#include "support.hpp"

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/encoding/horizon_graphs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
using Row = std::vector<u64>;

std::vector<Row> rows_of(const HorizonGraphs& h)
{
    std::vector<Row> out;
    for (u64 n = 0; n < h.num_nodes(); ++n)
        out.emplace_back(h.nodes.begin() + static_cast<std::ptrdiff_t>(n * h.words),
                         h.nodes.begin() + static_cast<std::ptrdiff_t>((n + 1) * h.words));
    return out;
}

/// The successors of one state: (schema, binding of label_width entries, row).
struct Successor
{
    i32 schema;
    std::vector<i32> binding;
    Row row;
};

std::vector<Successor> successors(const rl::TaskTable& table, const Row& state, i32 id)
{
    const u32 words = static_cast<u32>(state.size());
    const u32 width = table.label_width();
    rl::Expansion out;
    out.words = words;
    out.label_width = width;
    out.capacity = 4096;
    std::vector<u64> succ(out.capacity * words);
    std::vector<i32> schema(out.capacity), binding(out.capacity * width), offsets(2);
    for (;;)
    {
        succ.resize(out.capacity * words);
        schema.resize(out.capacity);
        binding.resize(out.capacity * width);
        out.succ = succ.data();
        out.schema = schema.data();
        out.binding = binding.data();
        out.offsets = offsets.data();
        rl::expand(table, {state.data(), 1, words, 0, 0}, &id, out);
        if (!out.overflow())
            break;
        out.capacity = out.total;
    }
    std::vector<Successor> result;
    for (u64 k = 0; k < out.total; ++k)
        result.push_back({schema[k], std::vector<i32>(binding.begin() + static_cast<std::ptrdiff_t>(k * width),
                                                      binding.begin() + static_cast<std::ptrdiff_t>((k + 1) * width)),
                          Row(succ.begin() + static_cast<std::ptrdiff_t>(k * words),
                              succ.begin() + static_cast<std::ptrdiff_t>((k + 1) * words))});
    return result;
}

/// The graph of one root by the definition: breadth first, the first reached state of a row is its node.
struct ReferenceGraph
{
    std::vector<Row> nodes;
    std::vector<std::array<i32, 2>> edges;
    std::vector<i32> schemas;
    std::vector<std::vector<i32>> bindings;
};

ReferenceGraph reference(const rl::TaskTable& table, const Row& root, u32 depth, bool merge)
{
    ReferenceGraph g;
    g.nodes.push_back(root);
    std::map<Row, i32> index{{root, 0}};
    std::vector<i32> level{0};
    for (u32 d = 0; d < depth; ++d)
    {
        std::vector<i32> next;
        for (const i32 node : level)
            for (Successor& s : successors(table, g.nodes[static_cast<usize>(node)], 0))
            {
                i32 to;
                if (merge && index.count(s.row))
                    to = index[s.row];
                else
                {
                    to = static_cast<i32>(g.nodes.size());
                    g.nodes.push_back(s.row);
                    index[s.row] = to;
                    next.push_back(to);
                }
                g.edges.push_back({node, to});
                g.schemas.push_back(s.schema);
                g.bindings.push_back(std::move(s.binding));
            }
        level = std::move(next);
    }
    // the edges are ordered by destination, then schema, then binding; edges that agree keep their order of discovery
    std::vector<usize> order(g.edges.size());
    std::iota(order.begin(), order.end(), usize{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](usize a, usize b)
                     {
                         return std::tie(g.edges[a][1], g.schemas[a], g.bindings[a]) <
                                std::tie(g.edges[b][1], g.schemas[b], g.bindings[b]);
                     });
    ReferenceGraph sorted;
    sorted.nodes = std::move(g.nodes);
    for (const usize e : order)
    {
        sorted.edges.push_back(g.edges[e]);
        sorted.schemas.push_back(g.schemas[e]);
        sorted.bindings.push_back(std::move(g.bindings[e]));
    }
    return sorted;
}

void expect_graph(const HorizonGraphs& h, u64 g, const ReferenceGraph& ref)
{
    const std::vector<Row> rows = rows_of(h);
    ASSERT_EQ(static_cast<usize>(h.node_offsets[g + 1] - h.node_offsets[g]), ref.nodes.size()) << "graph " << g;
    ASSERT_EQ(static_cast<usize>(h.edge_offsets[g + 1] - h.edge_offsets[g]), ref.edges.size()) << "graph " << g;
    for (usize n = 0; n < ref.nodes.size(); ++n)
        EXPECT_EQ(rows[static_cast<usize>(h.node_offsets[g]) + n], ref.nodes[n]) << "graph " << g << " node " << n;
    for (usize e = 0; e < ref.edges.size(); ++e)
    {
        const usize at = static_cast<usize>(h.edge_offsets[g]) + e;
        EXPECT_EQ(h.edge_src[at], ref.edges[e][0]) << "graph " << g << " edge " << e;
        EXPECT_EQ(h.edge_dst[at], ref.edges[e][1]) << "graph " << g << " edge " << e;
        EXPECT_EQ(h.edge_schema[at], ref.schemas[e]) << "graph " << g << " edge " << e;
        EXPECT_EQ(std::vector<i32>(h.edge_binding.begin() + static_cast<std::ptrdiff_t>(at * h.label_width),
                                   h.edge_binding.begin() + static_cast<std::ptrdiff_t>((at + 1) * h.label_width)),
                  ref.bindings[e])
            << "graph " << g << " edge " << e;
    }
}

bool equal(const HorizonGraphs& a, const HorizonGraphs& b)
{
    return a.words == b.words && a.label_width == b.label_width && a.task_ids == b.task_ids && a.nodes == b.nodes &&
           a.node_offsets == b.node_offsets && a.edge_offsets == b.edge_offsets && a.edge_src == b.edge_src &&
           a.edge_dst == b.edge_dst && a.edge_schema == b.edge_schema && a.edge_binding == b.edge_binding &&
           a.truncated == b.truncated;
}

struct Case
{
    const char* domain;
    const char* problem;
};
const Case k_cases[] = {{"blocks", "smedium"}, {"logistics00", "probLOGISTICS-6-1"}};

TEST(HorizonGraphs, FollowTheDefinition)
{
    for (const Case& c : k_cases)
    {
        const auto table = rl::TaskTable::single(test::pddl_task(c.domain, c.problem));
        const test::Rows roots = test::random_walk(*table, 0, 3, 11);
        for (const u32 depth : {0u, 1u, 2u, 3u})
            for (const bool merge : {true, false})
            {
                if (!merge && depth == 3 && std::string(c.domain) == "logistics00")
                    continue;  // a tree this deep is large
                SCOPED_TRACE(std::string(c.domain) + " depth " + std::to_string(depth) + (merge ? " merged" : " tree"));
                HorizonBuild options;
                options.depth = depth;
                options.merge = merge;
                const HorizonGraphs h = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
                ASSERT_EQ(h.graphs(), roots.size());
                for (u64 g = 0; g < roots.size(); ++g)
                    expect_graph(h, g,
                                 reference(*table,
                                           Row(roots.data.begin() + static_cast<std::ptrdiff_t>(g * roots.words),
                                               roots.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * roots.words)),
                                           depth, merge));
            }
    }
}

TEST(HorizonGraphs, ATreeIsWhatTheTestSupportBuilds)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 3, 5);
    const test::Horizon expected = test::horizon_of(*table, roots, 2);
    HorizonBuild options;
    options.depth = 2;
    options.merge = false;
    const HorizonGraphs h = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    EXPECT_EQ(h.nodes, expected.nodes);
    EXPECT_EQ(h.node_offsets, expected.node_offsets);
    EXPECT_EQ(h.edge_offsets, expected.edge_offsets);
    EXPECT_EQ(h.edge_src, expected.edge_src);
    EXPECT_EQ(h.edge_dst, expected.edge_dst);
    EXPECT_EQ(h.edge_schema, expected.edge_schema);
    EXPECT_EQ(h.edge_binding, expected.edge_binding);
}

TEST(HorizonGraphs, MergingKeepsEachStateOnce)
{
    // blocks can undo an action: depth 2 reaches the root again and states by more than one path
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows roots = test::random_walk(*table, 0, 1, 1);
    HorizonBuild options;
    options.depth = 3;
    const HorizonGraphs merged = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    options.merge = false;
    const HorizonGraphs tree = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    EXPECT_LT(merged.num_nodes(), tree.num_nodes());
    EXPECT_LT(merged.num_edges(), tree.num_edges());
    const std::vector<Row> rows = rows_of(merged);
    EXPECT_EQ(std::set<Row>(rows.begin(), rows.end()).size(), rows.size());
    // the edges of a node lead to nodes: every one of them from a node of depth below 3, and some back to the root
    bool back_edge = false, shared = false;
    std::map<i32, int> incoming;
    for (usize e = 0; e < merged.num_edges(); ++e)
    {
        back_edge = back_edge || merged.edge_dst[e] == 0;
        shared = shared || ++incoming[merged.edge_dst[e]] > 1;
    }
    EXPECT_TRUE(back_edge);
    EXPECT_TRUE(shared);
}

TEST(HorizonGraphs, ThePoolDoesNotChangeTheGraphs)
{
    const auto table = rl::TaskTable::single(test::pddl_task("logistics00", "probLOGISTICS-6-1"));
    const test::Rows roots = test::random_walk(*table, 0, 40, 3);
    HorizonBuild options;
    options.depth = 2;
    const HorizonGraphs expected = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    for (const u32 threads : {1u, 2u, 7u})
    {
        ThreadPool pool(threads);
        rl::ExpandScratch scratch;
        EXPECT_TRUE(equal(build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options, pool, scratch), expected))
            << threads << " threads";
    }
}

TEST(HorizonGraphs, MaxNodesRefusesOrTruncates)
{
    const auto table = rl::TaskTable::single(test::pddl_task("logistics00", "probLOGISTICS-6-1"));
    const test::Rows roots = test::random_walk(*table, 0, 3, 3);
    HorizonBuild options;
    options.depth = 2;
    const HorizonGraphs full = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    u32 widest = 0;
    for (u64 g = 0; g < full.graphs(); ++g)
        widest = std::max<u32>(widest, static_cast<u32>(full.node_offsets[g + 1] - full.node_offsets[g]));
    // a limit that fits changes nothing
    options.max_nodes = widest;
    EXPECT_TRUE(equal(build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options), full));
    // one below refuses, naming the graph
    options.max_nodes = widest - 1;
    try
    {
        (void)build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
        FAIL() << "expected a refusal";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("max_nodes"), std::string::npos) << error.what();
    }
    // with truncation: the graphs keep the first max_nodes nodes in order, only edges between kept nodes
    options.truncate = true;
    options.max_nodes = 12;
    const HorizonGraphs cut = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
    for (u64 g = 0; g < cut.graphs(); ++g)
    {
        const u64 nodes = static_cast<u64>(cut.node_offsets[g + 1] - cut.node_offsets[g]);
        const u64 all = static_cast<u64>(full.node_offsets[g + 1] - full.node_offsets[g]);
        EXPECT_EQ(nodes, std::min<u64>(all, 12)) << "graph " << g;
        EXPECT_EQ(cut.truncated[g] != 0, all > 12) << "graph " << g;
        for (u64 n = 0; n < nodes; ++n)
            EXPECT_EQ(rows_of(cut)[static_cast<u64>(cut.node_offsets[g]) + n], rows_of(full)[static_cast<u64>(full.node_offsets[g]) + n]);
        for (i32 e = cut.edge_offsets[g]; e < cut.edge_offsets[g + 1]; ++e)
        {
            EXPECT_LT(static_cast<u64>(cut.edge_src[static_cast<usize>(e)]), nodes);
            EXPECT_LT(static_cast<u64>(cut.edge_dst[static_cast<usize>(e)]), nodes);
        }
    }
    options.max_nodes = 0;
    EXPECT_THROW((void)build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options), std::invalid_argument);
}

TEST(HorizonGraphs, TaskIdsAreChecked)
{
    const auto table = rl::TaskTable::create({test::pddl_task("blocks", "smedium"), test::pddl_task("blocks", "smedium")});
    const test::Rows roots = test::random_walk(*table, 0, 2, 3);
    HorizonBuild options;
    EXPECT_THROW((void)build_horizon_graphs(*table, roots.view(), nullptr, options), std::invalid_argument);
    std::vector<i32> ids = {0, 2};
    EXPECT_THROW((void)build_horizon_graphs(*table, roots.view(), ids.data(), options), std::invalid_argument);
    ids = {1, 0};
    const HorizonGraphs h = build_horizon_graphs(*table, roots.view(), ids.data(), options);
    EXPECT_EQ(h.task_ids, ids);
    const test::Rows none;
    EXPECT_EQ(build_horizon_graphs(*table, {nullptr, 0, roots.words, 0, 0}, nullptr, options).graphs(), 0u);
}

TEST(HorizonGraphs, TransitionsAreTheDepthOneTree)
{
    for (const Case& c : k_cases)
    {
        const auto table = rl::TaskTable::single(test::pddl_task(c.domain, c.problem));
        const test::Rows roots = test::random_walk(*table, 0, 5, 9);
        std::vector<u64> succ;
        std::vector<i32> schema, binding, offsets(roots.size() + 1);
        rl::Expansion out;
        out.words = roots.words;
        out.label_width = table->label_width();
        out.capacity = 8;
        for (;;)
        {
            succ.resize(out.capacity * roots.words);
            schema.resize(out.capacity);
            binding.resize(out.capacity * out.label_width);
            out.succ = succ.data();
            out.schema = schema.data();
            out.binding = binding.data();
            out.offsets = offsets.data();
            rl::expand(*table, roots.view(), roots.task_ids.data(), out);
            if (!out.overflow())
                break;
            out.capacity = out.total;
        }
        HorizonBuild options;
        options.depth = 1;
        options.merge = false;
        const HorizonGraphs tree = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
        const HorizonGraphs transitions = transition_graphs(*table, roots.view(), roots.task_ids.data(), out);
        EXPECT_TRUE(equal(tree, transitions)) << c.domain;
    }
}

TEST(HorizonGraphs, TransitionGraphsCarryDerivedWords)
{
    const auto table = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    ASSERT_GT(table->derived_words(), 0u);
    const test::Rows roots = test::random_walk(*table, 0, 2, 4);
    const u32 d = table->derived_words();
    std::vector<u64> succ(1024 * roots.words), derived(1024 * d), parent_derived(roots.size() * d, 7);
    std::vector<i32> schema(1024), binding(1024 * table->label_width()), offsets(roots.size() + 1);
    rl::Expansion out;
    out.words = roots.words;
    out.label_width = table->label_width();
    out.derived_words = d;
    out.capacity = 1024;
    out.succ = succ.data();
    out.schema = schema.data();
    out.binding = binding.data();
    out.derived = derived.data();
    out.offsets = offsets.data();
    rl::expand(*table, roots.view(), roots.task_ids.data(), out);
    ASSERT_FALSE(out.overflow());
    const HorizonGraphs h = transition_graphs(*table, roots.view(), roots.task_ids.data(), out, parent_derived.data(), d);
    ASSERT_EQ(h.derived.size(), h.num_nodes() * d);
    for (u64 g = 0; g < roots.size(); ++g)
        EXPECT_EQ(h.derived[static_cast<u64>(h.node_offsets[g]) * d], 7u);
    // without the parents' words there are none
    EXPECT_TRUE(transition_graphs(*table, roots.view(), roots.task_ids.data(), out).derived.empty());
    // an overflowed expansion is refused
    out.capacity = 1;
    EXPECT_THROW((void)transition_graphs(*table, roots.view(), roots.task_ids.data(), out), std::invalid_argument);
}

TEST(HorizonGraphs, AStateSpaceGivesTheGraphsOfItsExpansion)
{
    int tested = 0;
    for (const Case& c : k_cases)
    {
        const TaskPtr task = test::pddl_task(c.domain, c.problem);
        datasets::StateSpaceOptions space_options;
        space_options.max_states = 20000;
        space_options.remove_if_unsolvable = false;
        const datasets::StateSpaceResult result = datasets::generate_state_space(task, space_options);
        if (result.status != search::SearchStatus::Exhausted)
            continue;  // the space is too large for the test
        ++tested;
        const datasets::StateSpace& space = *result.space;
        const auto table = rl::TaskTable::single(task);
        SuccessorGraph graph;
        graph.states = space.num_states();
        graph.words = space.words();
        graph.state_words = space.state_words().data();
        graph.row_stride = space.row_words();
        graph.offsets = space.forward_offsets().data();
        graph.targets = space.forward_targets().data();
        graph.schemas = space.label_schemas().data();
        graph.bindings = space.label_bindings().data();
        graph.label_width = space.label_width();
        const u32 last = static_cast<u32>(graph.states - 1);
        const std::vector<u32> ids = {0, std::min<u32>(3, last), last};
        // the same states as rows
        test::Rows roots;
        roots.words = table->words();
        for (const u32 id : ids)
        {
            roots.data.insert(roots.data.end(), graph.state_words + u64{id} * graph.row_stride,
                              graph.state_words + u64{id} * graph.row_stride + graph.words);
            roots.task_ids.push_back(0);
        }
        for (const u32 depth : {0u, 1u, 2u})
            for (const bool merge : {true, false})
            {
                SCOPED_TRACE(std::string(c.domain) + " depth " + std::to_string(depth) + (merge ? " merged" : " tree"));
                HorizonBuild options;
                options.depth = depth;
                options.merge = merge;
                const HorizonGraphs from_space = build_horizon_graphs(graph, ids, 0, options);
                const HorizonGraphs from_rows = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), options);
                EXPECT_TRUE(equal(from_space, from_rows));
            }
    }
    EXPECT_GT(tested, 0);
}

TEST(HorizonGraphs, AStateOutsideTheSpaceIsRefused)
{
    const TaskPtr task = test::pddl_task("blocks", "smedium");
    const auto result = datasets::generate_state_space(task);
    ASSERT_EQ(result.status, search::SearchStatus::Exhausted);
    const datasets::StateSpace& space = *result.space;
    SuccessorGraph graph{space.num_states(),
                         space.words(),
                         space.state_words().data(),
                         space.row_words(),
                         space.forward_offsets().data(),
                         space.forward_targets().data(),
                         space.label_schemas().data(),
                         space.label_bindings().data(),
                         space.label_width()};
    const std::vector<u32> bad = {space.num_states()};
    EXPECT_THROW((void)build_horizon_graphs(graph, bad, 0, HorizonBuild{}), std::invalid_argument);
}
}  // namespace
