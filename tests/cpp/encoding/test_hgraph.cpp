// The HGraph view of a relation-major batch: its node types, its edge types as strided views of the batch's arguments
// (checked by pointer identity and against the tuples of the same input encoded graph-major), for the encodings of
// states, horizons and transitions in both index types, and its refusals.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/hgraph.hpp"
#include "mymyr/encoding/horizon_graphs.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <random>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
std::vector<i64> ints(const rl::ArrayBundle& bundle, const char* name)
{
    const rl::ArrayInfo* info = bundle.find(name);
    std::vector<i64> out;
    if (!info)
    {
        ADD_FAILURE() << "missing array " << name;
        return out;
    }
    for (u64 i = 0; i < info->elements(); ++i)
    {
        const std::byte* at = static_cast<const std::byte*>(bundle.data(*info)) + i * rl::dtype_bytes(info->dtype);
        switch (info->dtype)
        {
            case rl::DType::U8: out.push_back(std::to_integer<u8>(*at)); break;
            case rl::DType::I32:
            {
                i32 v;
                std::memcpy(&v, at, 4);
                out.push_back(v);
                break;
            }
            case rl::DType::I64:
            {
                i64 v;
                std::memcpy(&v, at, 8);
                out.push_back(v);
                break;
            }
            default: ADD_FAILURE() << "unexpected dtype of " << name;
        }
    }
    return out;
}

constexpr auto k_layouts = {Layout::Graph, Layout::Relation};
constexpr auto k_indices = {IndexType::I32, IndexType::I64};

FlatOptions as_layout(FlatOptions options, Layout layout, IndexType index)
{
    options.layout = layout;
    options.index = index;
    return options;
}

/// The view of the relation-major `batch` agrees with the graph-major encoding `reference` of the same input: the
/// counts, the node ptrs, every edge (the entity ids of the tuples, graph after graph), the pointers (row 1 points into
/// the batch's relation_args, the `_symbol_` ptr is its entity_offsets) and the entity roles.
void expect_view_is_batch(const std::shared_ptr<const rl::ArrayBundle>& batch, const rl::ArrayBundle& reference,
                          const std::shared_ptr<const EncodingSchema>& schema)
{
    const HGraphView view = HGraphView::of(batch, schema);
    const u64 R = schema->num_relations();
    const u64 G = static_cast<u64>(batch->scalar("num_graphs"));
    const u64 width = view.wide() ? 8 : 4;
    ASSERT_EQ(view.num_graphs(), G);
    ASSERT_EQ(view.num_node_types(), R + 1);
    ASSERT_EQ(view.symbol_type(), R);
    EXPECT_EQ(view.node_type_name(static_cast<u32>(R)), "_symbol_");
    EXPECT_THROW((void)view.node_type_name(static_cast<u32>(R + 1)), std::out_of_range);
    EXPECT_THROW((void)view.num_nodes(static_cast<u32>(R + 1)), std::out_of_range);

    const auto reference_offsets = ints(reference, "relation_offsets");
    const auto reference_arg_offsets = ints(reference, "relation_arg_offsets");
    const auto reference_args = ints(reference, "relation_args");
    const auto arg_offsets = ints(*batch, "relation_arg_offsets");
    const auto entity_offsets = ints(*batch, "entity_offsets");
    const auto symbol_ptr = view.symbol_ptr();
    ASSERT_EQ(symbol_ptr.size(), G + 1);
    EXPECT_EQ(symbol_ptr.data(), batch->data(*batch->find("entity_offsets")));
    EXPECT_EQ(symbol_ptr.stride(), 1u);
    for (u64 g = 0; g <= G; ++g)
        EXPECT_EQ(symbol_ptr[g], entity_offsets[g]);
    EXPECT_EQ(view.num_nodes(view.symbol_type()), static_cast<u64>(entity_offsets[G]));
    EXPECT_EQ(view.num_entities(), static_cast<u64>(entity_offsets[G]));

    u64 edge_types = 0;
    for (u32 r = 0; r < R; ++r)
    {
        SCOPED_TRACE(schema->relation(r).name);
        EXPECT_EQ(view.node_type_name(r), schema->relation(r).name);
        const u64 arity = schema->relation(r).arity;
        // the tuples of relation r in graph-major order: cell (g, r) is segment g * R + r
        u64 nodes = 0;
        std::vector<i64> ptr{0};
        for (u64 g = 0; g < G; ++g)
        {
            nodes += static_cast<u64>(reference_offsets[g * R + r + 1] - reference_offsets[g * R + r]);
            ptr.push_back(static_cast<i64>(nodes));
        }
        ASSERT_EQ(view.num_nodes(r), nodes);
        std::vector<i64> got(G + 1);
        std::vector<i32> got32(G + 1);
        std::vector<i64> got64(G + 1);
        view.relation_ptr(r, view.wide() ? static_cast<void*>(got64.data()) : static_cast<void*>(got32.data()));
        for (u64 g = 0; g <= G; ++g)
            EXPECT_EQ(view.wide() ? got64[g] : got32[g], ptr[g]);
        for (u64 slot = 0; slot < arity; ++slot)
        {
            const auto id = view.find_edge_type(r, static_cast<u32>(slot));
            ASSERT_TRUE(id.has_value());
            EXPECT_EQ(*id, edge_types + slot);
            EXPECT_EQ(view.edge_type(*id), (HGraphEdgeType{r, static_cast<u32>(slot)}));
            ASSERT_EQ(view.num_edges(*id), nodes);
            const IndexRow row1 = view.edge_index_1(*id);
            ASSERT_EQ(row1.size(), nodes);
            EXPECT_EQ(row1.stride(), arity);
            EXPECT_EQ(row1.wide(), view.wide());
            // a view of the batch's arguments, no copy
            const auto* base = static_cast<const std::byte*>(batch->data(*batch->find("relation_args")));
            EXPECT_EQ(static_cast<const std::byte*>(row1.data()),
                      base + (static_cast<u64>(arg_offsets[r * G]) + slot) * width);
            for (u64 g = 0; g < G; ++g)
                for (i64 j = 0; j < ptr[g + 1] - ptr[g]; ++j)
                    ASSERT_EQ(row1[static_cast<u64>(ptr[g] + j)],
                              reference_args[static_cast<u64>(reference_arg_offsets[g * R + r]) +
                                             static_cast<u64>(j) * arity + slot])
                        << "graph " << g << " tuple " << j << " position " << slot;
            std::vector<i32> arange32(nodes);
            std::vector<i64> arange64(nodes);
            view.edge_index_0(*id, view.wide() ? static_cast<void*>(arange64.data()) : static_cast<void*>(arange32.data()));
            for (u64 i = 0; i < nodes; ++i)
                ASSERT_EQ(view.wide() ? arange64[i] : arange32[i], static_cast<i64>(i));
        }
        EXPECT_FALSE(view.find_edge_type(r, static_cast<u32>(arity)).has_value());
        edge_types += arity;
    }
    EXPECT_EQ(view.num_edge_types(), edge_types);
    EXPECT_THROW((void)view.edge_type(edge_types), std::out_of_range);
    EXPECT_THROW((void)view.num_edges(edge_types), std::out_of_range);
    EXPECT_FALSE(view.find_edge_type(static_cast<u32>(R), 0).has_value());

    // the entity roles: the stored array itself, or the implicit roles of the objects and the auxiliary entity
    std::vector<u8> roles(view.num_entities());
    view.write_entity_roles(roles.data());
    if (const rl::ArrayInfo* stored = batch->find("entity_role"))
    {
        const auto reference_roles = ints(reference, "entity_role");
        EXPECT_EQ(view.stored_entity_roles(), batch->data(*stored));
        for (u64 e = 0; e < roles.size(); ++e)
            ASSERT_EQ(roles[e], reference_roles[e]);
    }
    else
    {
        EXPECT_EQ(view.stored_entity_roles(), nullptr);
        const auto objects = ints(*batch, "num_objects");
        for (u64 g = 0; g < G; ++g)
            for (i64 e = entity_offsets[g]; e < entity_offsets[g + 1]; ++e)
                ASSERT_EQ(roles[static_cast<u64>(e)], e - entity_offsets[g] < objects[g]
                                                          ? static_cast<u8>(EntityRole::Object)
                                                          : static_cast<u8>(EntityRole::Auxiliary));
    }
}

/// Encodes one input in both layouts and both index types with `options` and checks the view of each relation-major
/// batch against the graph-major encoding.
void expect_views(const rl::TaskTablePtr& table, const FlatOptions& options,
                  const std::function<EncodeInput()>& make_input)
{
    for (const IndexType index : k_indices)
    {
        const auto graph_major = FlatEncoder::create(table, as_layout(options, Layout::Graph, index));
        const auto relation_major = FlatEncoder::create(table, as_layout(options, Layout::Relation, index));
        const EncodeInput in = make_input();
        const auto reference = graph_major->encode(in);
        const auto batch = relation_major->encode(in);
        validate(*batch, relation_major->schema());
        expect_view_is_batch(batch, *reference, relation_major->schema_ptr());
    }
}

// ------------------------------------------------------------------------------------------------ inputs

/// An expansion of rows that outlives the call that made it.
struct Expanded
{
    rl::Expansion x;
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
};

std::shared_ptr<Expanded> expand_rows(const rl::TaskTable& table, const test::Rows& rows)
{
    auto e = std::make_shared<Expanded>();
    u32 words = rows.words;
    u64 capacity = rows.size() * 8 + 16;
    for (;;)
    {
        e->succ.assign(capacity * words, 0);
        e->parent.assign(capacity, 0);
        e->schema.assign(capacity, 0);
        e->binding.assign(capacity * table.label_width(), 0);
        e->offsets.assign(rows.size() + 1, 0);
        e->x = {};
        e->x.capacity = capacity;
        e->x.words = words;
        e->x.label_width = table.label_width();
        e->x.succ = e->succ.data();
        e->x.parent = e->parent.data();
        e->x.schema = e->schema.data();
        e->x.binding = e->binding.data();
        e->x.offsets = e->offsets.data();
        rl::expand(table, rows.view(), rows.task_ids.data(), e->x);
        if (!e->x.overflow())
            return e;
        capacity = std::max(capacity * 2, e->x.total);
        words = std::max(words, e->x.words_needed);
    }
}

/// Random subgoal layers and history steps for rows: subsets of the rows' own atoms and of the atoms of other rows.
struct Layers
{
    std::vector<u64> layer_pos, layer_neg, step_pos, step_neg;
    std::vector<i32> step_dt, step_count;
    u32 levels = 0, steps = 0, words = 0;

    Layers(const test::Rows& rows, u32 levels_in, u32 steps_in, u64 seed) : levels(levels_in), steps(steps_in), words(rows.words)
    {
        const u64 n = rows.size();
        layer_pos.assign(n * levels * words, 0);
        layer_neg = layer_pos;
        step_pos.assign(n * steps * words, 0);
        step_neg = step_pos;
        step_dt.assign(n * steps, 0);
        step_count.assign(n, 0);
        std::mt19937_64 rng(seed);
        // the negative literals come from a row of the same instance, whose atom slots are the same
        auto literals = [&](u64 row, u64 other, u64* pos, u64* neg)
        {
            u64 peer = row;
            for (u64 k = 0; k < n; ++k)
                if (rows.task_ids[(other + k) % n] == rows.task_ids[row])
                {
                    peer = (other + k) % n;
                    break;
                }
            for (u32 w = 0; w < words; ++w)
            {
                pos[w] = rows.data[row * words + w] & rng();
                neg[w] = rows.data[peer * words + w] & rng() & ~pos[w];
            }
        };
        for (u64 i = 0; i < n; ++i)
        {
            for (u32 l = 0; l < levels; ++l)
                literals(i, i + 7 + l, &layer_pos[(i * levels + l) * words], &layer_neg[(i * levels + l) * words]);
            step_count[i] = steps ? static_cast<i32>(i % (steps + 1)) : 0;
            for (u32 k = 0; k < steps; ++k)
            {
                literals(i, i + 3 * k + 5, &step_pos[(i * steps + k) * words],
                         &step_neg[(i * steps + k) * words]);
                step_dt[i * steps + k] = static_cast<i32>(k) - static_cast<i32>(steps);
            }
        }
    }

    void apply(EncodeInput& in) const
    {
        if (levels)
        {
            in.subgoals.pos = layer_pos.data();
            in.subgoals.neg = layer_neg.data();
            in.subgoals.stride = u64{levels} * words;
            in.subgoals.layers = levels;
            in.subgoals.words = words;
        }
        if (steps)
        {
            in.history.masks.pos = step_pos.data();
            in.history.masks.neg = step_neg.data();
            in.history.masks.stride = u64{steps} * words;
            in.history.masks.layers = steps;
            in.history.masks.words = words;
            in.history.dt = step_dt.data();
            in.history.count = step_count.data();
        }
    }
};

struct Problem
{
    const char* domain;
    std::vector<const char*> problems;
};

const Problem k_problems[] = {{"blocks", {"smedium", "probBLOCKS-8-0"}},
                              {"features", {"p01", "p02", "p03", "p04"}},
                              {"gripper", {"p01", "p02"}},
                              {"philosophers", {"p03-phil4", "p03-phil4-one-blocked"}}};

rl::TaskTablePtr table_of(const Problem& problem)
{
    std::vector<TaskPtr> tasks;
    for (const char* name : problem.problems)
        tasks.push_back(test::pddl_task(problem.domain, name));
    return rl::TaskTable::create(tasks);
}

/// `n` rows of a walk over each instance of the table, one after the other.
test::Rows rows_of(const rl::TaskTable& table, u64 n, u64 seed)
{
    test::Rows all;
    all.words = table.words();
    for (u32 instance = 0; instance < table.size(); ++instance)
    {
        const test::Rows part = test::random_walk(table, instance, n, seed + instance);
        all.data.insert(all.data.end(), part.data.begin(), part.data.end());
        all.task_ids.insert(all.task_ids.end(), part.task_ids.begin(), part.task_ids.end());
    }
    return all;
}

FlatOptions everything()
{
    FlatOptions options;
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    options.max_goal_level = 2;
    options.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::Action, TargetSource::History};
    options.history = true;
    return options;
}

// ------------------------------------------------------------------------------------------------ states

TEST(HGraphView, EdgesAreTheTuplesOfTheBatchForEveryConfiguration)
{
    struct Config
    {
        const char* name;
        std::function<void(FlatOptions&)> change;
    };
    const GoalDerivations all = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    const std::vector<Config> configs = {
        {"state", [](FlatOptions&) {}},
        {"goals", [&](FlatOptions& o) { o.goal_derivations = all; }},
        {"goal_targets",
         [&](FlatOptions& o)
         {
             o.goal_derivations = all;
             o.targets = {TargetSource::Goal};
         }},
        {"no_nullary_equality",
         [](FlatOptions& o)
         {
             o.nullary = false;
             o.equality = true;
         }},
        {"no_static_no_derived",
         [](FlatOptions& o)
         {
             o.include_static = false;
             o.include_derived = false;
         }},
        {"implicit_entities_no_counts",
         [](FlatOptions& o)
         {
             o.include_actions = false;
             o.emit_counts = false;
             o.entity_arrays = EntityArrays::Auto;
         }},
        {"everything", [](FlatOptions& o) { o = everything(); }},
        {"layers_without_actions",
         [](FlatOptions& o)
         {
             o = everything();
             o.include_actions = false;
             o.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::History};
         }},
    };
    for (const Problem& problem : k_problems)
    {
        const auto table = table_of(problem);
        const test::Rows rows = rows_of(*table, 12, 3);
        const auto actions = expand_rows(*table, rows);
        const Layers layers(rows, 2, 3, 9);
        for (const Config& config : configs)
        {
            SCOPED_TRACE(std::string(problem.domain) + " " + config.name);
            FlatOptions options = test::encoder_options();
            config.change(options);
            expect_views(table, options,
                         [&]
                         {
                             EncodeInput in;
                             in.states = rows.view();
                             in.task_ids = rows.task_ids.data();
                             if (options.include_actions)
                                 in.actions = &actions->x;
                             if (options.max_goal_level || options.history)
                                 layers.apply(in);
                             return in;
                         });
        }
    }
}

TEST(HGraphView, ANullaryRelationsEdgeLeadsToTheAuxiliaryEntity)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 20, 4);
    FlatOptions options = test::encoder_options();
    options.layout = Layout::Relation;
    options.goal_derivations = {GoalDerivation::Plain};
    const auto encoder = FlatEncoder::create(table, options);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto batch = encoder->encode(in);
    const HGraphView view = HGraphView::of(batch, encoder->schema_ptr());
    const auto handempty = encoder->schema().find("handempty");
    ASSERT_TRUE(handempty.has_value());
    ASSERT_EQ(encoder->schema().relation(*handempty).logical_arity, 0u);
    ASSERT_EQ(encoder->schema().relation(*handempty).arity, 1u);
    const auto id = view.find_edge_type(*handempty, 0);
    ASSERT_TRUE(id.has_value());
    const IndexRow row1 = view.edge_index_1(*id);
    ASSERT_GT(row1.size(), 0u);
    const auto entity_offsets = ints(*batch, "entity_offsets");
    const auto num_objects = ints(*batch, "num_objects");
    const auto roles = ints(*batch, "entity_role");
    std::vector<i32> ptr(view.num_graphs() + 1);
    view.relation_ptr(*handempty, ptr.data());
    u64 graphs_with_tuple = 0;
    for (u64 g = 0; g < view.num_graphs(); ++g)
    {
        ASSERT_LE(ptr[g + 1] - ptr[g], 1);
        for (i32 i = ptr[g]; i < ptr[g + 1]; ++i)
        {
            ++graphs_with_tuple;
            const i64 entity = row1[static_cast<u64>(i)];
            EXPECT_EQ(entity, entity_offsets[g] + num_objects[g]);
            EXPECT_EQ(roles[static_cast<u64>(entity)], static_cast<i64>(EntityRole::Auxiliary));
        }
    }
    EXPECT_GT(graphs_with_tuple, 0u);
}

TEST(HGraphView, TheEmptyBatchAndASingleRow)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 1, 2);
    for (const bool actions : {false, true})
    {
        FlatOptions options = everything();
        options.include_actions = actions;
        if (!actions)
            options.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::History};
        for (const u64 n : {u64{0}, u64{1}})
        {
            SCOPED_TRACE("rows " + std::to_string(n) + (actions ? " with actions" : " without actions"));
            test::Rows part = rows;
            part.data.resize(n * rows.words);
            part.task_ids.resize(n);
            const auto expansion = expand_rows(*table, part);
            const Layers layers(part, 2, 3, 1);
            expect_views(table, options,
                         [&]
                         {
                             EncodeInput in;
                             in.states = part.view();
                             in.task_ids = part.task_ids.data();
                             if (actions)
                                 in.actions = &expansion->x;
                             if (n)
                                 layers.apply(in);
                             return in;
                         });
        }
    }
}

// ------------------------------------------------------------------------------------------------ horizons

struct HorizonConfig
{
    HorizonMode mode;
    RootPolicy root;
    bool parent, sibling, cousin;
};

TEST(HGraphView, EdgesAreTheTuplesOfHorizonAndTransitionBatches)
{
    const HorizonConfig configs[] = {
        {HorizonMode::Full, RootPolicy::Include, false, false, false},
        {HorizonMode::Full, RootPolicy::EncodeOnly, true, true, true},
        {HorizonMode::Delta, RootPolicy::Exclude, false, false, false},
        {HorizonMode::Delta, RootPolicy::EncodeOnly, true, false, false},
        {HorizonMode::Action, RootPolicy::Exclude, true, true, false},
    };
    for (const Problem& problem : {k_problems[0], k_problems[2]})
    {
        const auto table = table_of(problem);
        const test::Rows roots = rows_of(*table, 3, 5);
        for (const u32 depth : {1u, 2u})
        {
            HorizonBuild build;
            build.depth = depth;
            const HorizonGraphs graphs = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), build);
            for (const HorizonConfig& c : configs)
            {
                SCOPED_TRACE(std::string(problem.domain) + " depth " + std::to_string(depth) + " " +
                             std::string(to_string(c.mode)) + "/" + std::string(to_string(c.root)));
                FlatOptions options;
                options.horizon.mode = c.mode;
                options.horizon.root = c.root;
                options.horizon.parent_relation = c.parent;
                options.horizon.sibling_relation = c.sibling;
                options.horizon.cousin_relation = c.cousin;
                if (c.mode == HorizonMode::Delta)
                    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::AddedSatisfied,
                                                GoalDerivation::AddedUnsatisfied};
                expect_views(table, options,
                             [&]
                             {
                                 EncodeInput in;
                                 in.horizon = graphs.input();
                                 in.task_ids = graphs.task_ids.data();
                                 return in;
                             });
            }
        }
    }
}

TEST(HGraphView, TransitionsFromAnExpansionAreViewedLikeAnyHorizon)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows parents = test::random_walk(*table, 0, 8, 6);
    const auto expansion = expand_rows(*table, parents);
    const HorizonGraphs graphs =
        transition_graphs(*table, parents.view(), parents.task_ids.data(), expansion->x, nullptr, 0);
    for (const HorizonMode mode : {HorizonMode::Full, HorizonMode::Delta})
    {
        FlatOptions options;
        options.horizon.mode = mode;
        options.horizon.root = RootPolicy::Exclude;
        if (mode == HorizonMode::Delta)
            options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::AddedSatisfied};
        expect_views(table, options,
                     [&]
                     {
                         EncodeInput in;
                         in.horizon = graphs.input();
                         in.task_ids = graphs.task_ids.data();
                         return in;
                     });
    }
}

// ------------------------------------------------------------------------------------------------ lifetime

TEST(HGraphView, TheViewKeepsTheBatchAlive)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 16, 8);
    FlatOptions options = test::encoder_options();
    options.layout = Layout::Relation;
    const auto encoder = FlatEncoder::create(table, options);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    std::shared_ptr<const rl::ArrayBundle> batch = encoder->encode(in);
    const rl::ArrayBundle& copy = *batch;
    (void)copy;
    // the graph-major reference tuples of the first relation with tuples, read before the batch is dropped
    std::vector<i64> expected;
    u32 relation = 0;
    for (; relation < encoder->schema().num_relations(); ++relation)
    {
        const HGraphView probe = HGraphView::of(batch, encoder->schema_ptr());
        if (probe.num_nodes(relation) > 0)
            break;
    }
    ASSERT_LT(relation, encoder->schema().num_relations());
    std::optional<HGraphView> view = HGraphView::of(batch, encoder->schema_ptr());
    const IndexRow before = view->edge_index_1(*view->find_edge_type(relation, 0));
    for (u64 i = 0; i < before.size(); ++i)
        expected.push_back(before[i]);
    batch.reset();
    EXPECT_GT(view->batch().use_count(), 0);
    const IndexRow row = view->edge_index_1(*view->find_edge_type(relation, 0));
    ASSERT_EQ(row.size(), expected.size());
    for (u64 i = 0; i < row.size(); ++i)
        EXPECT_EQ(row[i], expected[i]);
    // a copy of the view is the same view, and the batch outlives all but the last of them
    const HGraphView second = *view;
    view.reset();
    EXPECT_EQ(second.edge_index_1(*second.find_edge_type(relation, 0)).size(), expected.size());
}

// ------------------------------------------------------------------------------------------------ refusals

/// A copy of `bundle` with the scalar `name` set to `value`.
std::shared_ptr<const rl::ArrayBundle> with_scalar(const rl::ArrayBundle& bundle, const char* name, i64 value)
{
    rl::ArrayBundle::Builder out;
    for (const rl::ArrayInfo& info : bundle.arrays())
    {
        const auto* first = static_cast<const std::byte*>(bundle.data(info));
        out.add_bytes(info.name, info.dtype, info.shape,
                      std::vector<std::byte>(first, first + info.elements() * rl::dtype_bytes(info.dtype)), info.words);
    }
    for (const auto& [scalar, old] : bundle.scalars())
        out.scalar(scalar, scalar == name ? value : old);
    return std::make_shared<rl::ArrayBundle>(out.build());
}

template<class Statement>
void expect_refusal(Statement statement, const std::string& mentions)
{
    try
    {
        statement();
        FAIL() << "no exception; expected one that mentions '" << mentions << "'";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("mymyr: encoding: "), std::string::npos) << message;
        EXPECT_NE(message.find(mentions), std::string::npos) << message;
    }
}

TEST(HGraphView, RefusesWhatItCannotView)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const test::Rows rows = test::random_walk(*table, 0, 4, 1);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();

    // a graph-major batch
    {
        FlatOptions options = test::encoder_options();
        const auto encoder = FlatEncoder::create(table, options);
        const auto batch = encoder->encode(in);
        expect_refusal([&] { (void)HGraphView::of(batch, encoder->schema_ptr()); }, "relation-major");
    }
    // predicate virtual nodes
    {
        FlatOptions options = test::encoder_options();
        options.layout = Layout::Relation;
        options.predicate_virtual_nodes = true;
        const auto encoder = FlatEncoder::create(table, options);
        const auto batch = encoder->encode(in);
        expect_refusal([&] { (void)HGraphView::of(batch, encoder->schema_ptr()); }, "predicate virtual nodes");
    }
    FlatOptions options = test::encoder_options();
    options.layout = Layout::Relation;
    const auto encoder = FlatEncoder::create(table, options);
    const auto batch = encoder->encode(in);
    EXPECT_NO_THROW((void)HGraphView::of(batch, encoder->schema_ptr()));
    // shared static storage
    expect_refusal([&] { (void)HGraphView::of(with_scalar(*batch, "statics", 1), encoder->schema_ptr()); },
                   "shared static");
    // a batch of another schema
    {
        FlatOptions other = options;
        other.nullary = false;
        const auto other_encoder = FlatEncoder::create(table, other);
        expect_refusal([&] { (void)HGraphView::of(batch, other_encoder->schema_ptr()); }, "schema");
    }
    expect_refusal([&] { (void)HGraphView::of(nullptr, encoder->schema_ptr()); }, "needs a batch");
}
}  // namespace
