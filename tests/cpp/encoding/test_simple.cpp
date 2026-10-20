// The Simple graph of the relational encoder: the atom-object incidence graph of a batch's tuples. Every batch is
// compared with a reference built from its relation arrays by the definition (entity nodes, then atom nodes in graph-major
// tuple order; an edge and its reverse per tuple position), for states, subgoal layers, history, horizons and rooted
// horizons, both layouts and index types; the Simple-only encoding, single rows, slices, streams, threads, capacities,
// labels across instances and the refusals follow.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/horizon_graphs.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <functional>
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

/// The Simple arrays a batch must have, built from its relation arrays by the definition.
struct Reference
{
    std::vector<i64> node_offsets{0}, node_label, edge_offsets{0}, source, target, edge_label;
    u64 nodes = 0, edges = 0;
};

Reference reference_of(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const u64 R = schema.num_relations();
    const i64 K = schema.simple_max_arity();
    const auto entity_offsets = ints(batch, "entity_offsets");
    const auto role = ints(batch, "entity_role");
    const auto offsets = ints(batch, "relation_offsets");
    const auto arg_offsets = ints(batch, "relation_arg_offsets");
    const auto args = ints(batch, "relation_args");
    const u64 G = entity_offsets.size() - 1;
    const bool relation_major = schema.options().layout == Layout::Relation;
    Reference ref;
    i64 tuple_base = 0;
    std::vector<i64> forward_source, forward_target, forward_label, back_source, back_target, back_label;
    for (u64 g = 0; g < G; ++g)
    {
        const i64 first = entity_offsets[g], entities = entity_offsets[g + 1] - first;
        const i64 node_first = first + tuple_base;
        // the entity nodes
        for (i64 j = 0; j < entities; ++j)
            ref.node_label.push_back(role[static_cast<u64>(first + j)] == 0 && static_cast<u64>(j) < schema.simple_constants()
                                         ? static_cast<i64>(9 + R) + j
                                         : role[static_cast<u64>(first + j)]);
        // the atoms, relation after relation
        forward_source.clear();
        forward_target.clear();
        forward_label.clear();
        back_source.clear();
        back_target.clear();
        back_label.clear();
        i64 atoms = 0;
        for (u64 r = 0; r < R; ++r)
        {
            const u64 s = relation_major ? r * G + g : g * R + r;
            const u64 arity = schema.relation(static_cast<u32>(r)).arity;
            for (i64 t = 0; t < offsets[s + 1] - offsets[s]; ++t, ++atoms)
            {
                const i64 atom = node_first + entities + atoms;
                ref.node_label.push_back(static_cast<i64>(9 + r));
                for (u64 k = 0; k < arity; ++k)
                {
                    const i64 entity = args[static_cast<u64>(arg_offsets[s]) + static_cast<u64>(t) * arity + k] + tuple_base;
                    forward_source.push_back(entity);
                    forward_target.push_back(atom);
                    forward_label.push_back(static_cast<i64>(k));
                    back_source.push_back(atom);
                    back_target.push_back(entity);
                    back_label.push_back(K + static_cast<i64>(k));
                }
            }
        }
        for (const auto* part : {&forward_source, &back_source})
            ref.source.insert(ref.source.end(), part->begin(), part->end());
        for (const auto* part : {&forward_target, &back_target})
            ref.target.insert(ref.target.end(), part->begin(), part->end());
        for (const auto* part : {&forward_label, &back_label})
            ref.edge_label.insert(ref.edge_label.end(), part->begin(), part->end());
        tuple_base += atoms;
        ref.node_offsets.push_back(entity_offsets[g + 1] + tuple_base);
        ref.edge_offsets.push_back(static_cast<i64>(ref.source.size()));
    }
    ref.nodes = ref.node_label.size();
    ref.edges = ref.source.size();
    return ref;
}

/// The Simple arrays of `simple` equal the reference built from the relation arrays of `relations`.
void expect_simple_of(const rl::ArrayBundle& simple, const rl::ArrayBundle& relations, const EncodingSchema& schema)
{
    const Reference ref = reference_of(relations, schema);
    EXPECT_EQ(ints(simple, "node_offsets"), ref.node_offsets);
    EXPECT_EQ(ints(simple, "node_label"), ref.node_label);
    EXPECT_EQ(ints(simple, "edge_offsets"), ref.edge_offsets);
    EXPECT_EQ(ints(simple, "edge_label"), ref.edge_label);
    const auto index = ints(simple, "edge_index");
    ASSERT_EQ(index.size(), 2 * ref.edges);
    EXPECT_EQ(std::vector<i64>(index.begin(), index.begin() + static_cast<std::ptrdiff_t>(ref.edges)), ref.source);
    EXPECT_EQ(std::vector<i64>(index.begin() + static_cast<std::ptrdiff_t>(ref.edges), index.end()), ref.target);
    // V = E + T and the edges are twice the tuple arguments, per graph and in total
    const auto entity_offsets = ints(relations, "entity_offsets");
    const auto arg_offsets = ints(relations, "relation_arg_offsets");
    const u64 G = entity_offsets.size() - 1;
    EXPECT_EQ(ref.nodes, static_cast<u64>(entity_offsets[G] + ints(relations, "relation_offsets").back()));
    EXPECT_EQ(ref.edges, static_cast<u64>(2 * arg_offsets.back()));
    // per graph: its entities and the tuples of its segments are its nodes, twice their arguments its edges
    const auto offsets = ints(relations, "relation_offsets");
    const u64 R = schema.num_relations();
    const bool relation_major = schema.options().layout == Layout::Relation;
    for (u64 g = 0; g < G; ++g)
    {
        i64 tuples = 0, args = 0;
        for (u64 r = 0; r < R; ++r)
        {
            const u64 s = relation_major ? r * G + g : g * R + r;
            tuples += offsets[s + 1] - offsets[s];
            args += arg_offsets[s + 1] - arg_offsets[s];
        }
        EXPECT_EQ(ref.node_offsets[g + 1] - ref.node_offsets[g], entity_offsets[g + 1] - entity_offsets[g] + tuples);
        EXPECT_EQ(ref.edge_offsets[g + 1] - ref.edge_offsets[g], 2 * args);
    }
    EXPECT_GT(ref.nodes, 0u);
}

struct Tables
{
    rl::TaskTablePtr table;
    Tables(const std::string& domain, const std::vector<std::string>& problems)
    {
        std::vector<TaskPtr> tasks;
        for (const std::string& problem : problems)
            tasks.push_back(test::pddl_task(domain, problem));
        table = rl::TaskTable::create(tasks);
    }
};

/// `per_instance` rows of every instance, instance after instance.
test::Rows rows_of(const rl::TaskTable& table, u64 per_instance, u64 seed)
{
    test::Rows out;
    out.words = table.words();
    for (u32 i = 0; i < table.size(); ++i)
    {
        const test::Rows part = test::random_walk(table, i, per_instance, seed + i);
        out.data.insert(out.data.end(), part.data.begin(), part.data.end());
        out.task_ids.insert(out.task_ids.end(), part.task_ids.begin(), part.task_ids.end());
    }
    return out;
}

/// Subgoal layers and history steps drawn from the bits of other rows.
struct Layers
{
    std::vector<u64> layer_pos, layer_neg, step_pos, step_neg;
    std::vector<i32> step_dt, step_count;
    u32 levels = 0, steps = 0, words = 0;

    Layers(const test::Rows& rows, u32 levels_, u32 steps_, u64 seed) : levels(levels_), steps(steps_), words(rows.words)
    {
        const u64 n = rows.size();
        std::mt19937_64 rng(seed);
        layer_pos.assign(n * levels * words, 0);
        layer_neg = layer_pos;
        step_pos.assign(n * steps * words, 0);
        step_neg = step_pos;
        step_dt.assign(n * steps, 0);
        step_count.assign(n, 0);
        // a row of the same instance at distance `other` among the rows of that instance (bits of another instance's
        // atoms are not valid)
        auto partner = [&](u64 row, u64 other)
        {
            u64 left = other % n, at = row;
            while (left > 0 || rows.task_ids[at] != rows.task_ids[row])
            {
                at = (at + 1) % n;
                if (rows.task_ids[at] == rows.task_ids[row])
                    --left;
            }
            return at;
        };
        auto literals = [&](u64 row, u64 other, u64* pos, u64* neg)
        {
            const u64 mate = partner(row, other);
            for (u32 w = 0; w < words; ++w)
            {
                pos[w] = rows.data[mate * words + w] & rng();
                neg[w] = rows.data[partner(row, other + 1) * words + w] & rng() & ~pos[w];
            }
        };
        for (u64 i = 0; i < n; ++i)
        {
            for (u32 l = 0; l < levels; ++l)
                literals(i, 7 + l, &layer_pos[(i * levels + l) * words], &layer_neg[(i * levels + l) * words]);
            step_count[i] = steps ? static_cast<i32>(1 + i % steps) : 0;
            for (u32 k = 0; k < steps; ++k)
            {
                literals(i, 3 * k + 5, &step_pos[(i * steps + k) * words], &step_neg[(i * steps + k) * words]);
                step_dt[i * steps + k] = static_cast<i32>(k) - static_cast<i32>(steps);
            }
        }
    }

    void attach(EncodeInput& in, u64 first = 0) const
    {
        if (levels)
        {
            in.subgoals.pos = layer_pos.data() + first * levels * words;
            in.subgoals.neg = layer_neg.data() + first * levels * words;
            in.subgoals.stride = u64{levels} * words;
            in.subgoals.layers = levels;
            in.subgoals.words = words;
        }
        if (steps)
        {
            in.history.masks.pos = step_pos.data() + first * steps * words;
            in.history.masks.neg = step_neg.data() + first * steps * words;
            in.history.masks.stride = u64{steps} * words;
            in.history.masks.layers = steps;
            in.history.masks.words = words;
            in.history.dt = step_dt.data() + first * steps;
            in.history.count = step_count.data() + first;
        }
    }
};

struct Config
{
    const char* name;
    std::function<void(FlatOptions&)> change;
    u32 levels = 0, steps = 0;
};

std::vector<Config> configs()
{
    const GoalDerivations all = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    return {
        {"state", [](FlatOptions&) {}},
        {"goal", [](FlatOptions& o) { o.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied}; }},
        {"goal_targets",
         [all](FlatOptions& o)
         {
             o.goal_derivations = all;
             o.targets = {TargetSource::Goal};
         }},
        {"no_extras",
         [](FlatOptions& o)
         {
             o.include_static = false;
             o.include_derived = false;
             o.nullary = false;
         }},
        {"equality_actions_targets",
         [all](FlatOptions& o)
         {
             o.equality = true;
             o.goal_derivations = all;
             o.targets = {TargetSource::Goal, TargetSource::Action};
         }},
        {"layers",
         [all](FlatOptions& o)
         {
             o.goal_derivations = all;
             o.max_goal_level = 2;
             o.targets = {TargetSource::Goal, TargetSource::Subgoal};
         },
         2, 0},
        {"history",
         [](FlatOptions& o)
         {
             o.goal_derivations = {GoalDerivation::Plain};
             o.history = true;
             o.history_max_steps = 3;
             o.targets = {TargetSource::Goal, TargetSource::History};
         },
         0, 4},
        {"layers_and_history",
         [all](FlatOptions& o)
         {
             o.goal_derivations = all;
             o.max_goal_level = 1;
             o.history = true;
         },
         1, 2},
    };
}

constexpr auto k_layouts = {Layout::Graph, Layout::Relation};
constexpr auto k_indices = {IndexType::I32, IndexType::I64};

FlatOptions options_of(const Config& config, Layout layout, IndexType index, Outputs outputs)
{
    FlatOptions o = test::encoder_options();
    config.change(o);
    o.layout = layout;
    o.index = index;
    o.outputs = outputs;
    if (!outputs.contains(Output::Relations))
        o.layout = Layout::Graph;
    return o;
}

const Outputs k_both{Output::Relations, Output::Simple};
const Outputs k_simple{Output::Simple};

/// The arrays two bundles share by name are equal; both hold the same arrays when `same_names`.
void expect_arrays_equal(const rl::ArrayBundle& a, const rl::ArrayBundle& b, std::initializer_list<const char*> names)
{
    for (const char* name : names)
    {
        SCOPED_TRACE(name);
        EXPECT_EQ(ints(a, name), ints(b, name));
    }
}

constexpr std::initializer_list<const char*> k_simple_names = {"node_offsets", "node_label", "edge_offsets", "edge_index",
                                                               "edge_label"};

class SimpleStates : public ::testing::TestWithParam<std::tuple<const char*, const char*>>
{
};

std::vector<std::tuple<const char*, const char*>> domains()
{
    return {{"blocks", "smedium"}, {"gripper", "p01"}, {"goals", "conjunctive"}, {"features", "p01"},
            {"philosophers", "p03-phil4"}};
}

TEST_P(SimpleStates, IsTheIncidenceGraphOfTheTuplesForEveryConfiguration)
{
    const auto [domain, problem] = GetParam();
    const Tables tables(domain, {problem});
    const auto& table = tables.table;
    const test::Rows rows = rows_of(*table, 24, 5);
    for (const Config& config : configs())
        for (const Layout layout : k_layouts)
            for (const IndexType index : k_indices)
            {
                SCOPED_TRACE(std::string(config.name) + (layout == Layout::Graph ? " graph " : " relation ") +
                             (index == IndexType::I32 ? "i32" : "i64"));
                const auto encoder = FlatEncoder::create(table, options_of(config, layout, index, k_both));
                const Layers layers(rows, config.levels, config.steps, 3);
                EncodeInput in;
                in.states = rows.view();
                in.task_ids = rows.task_ids.data();
                layers.attach(in);
                const auto batch = encoder->encode(in);
                ASSERT_NO_THROW(validate(*batch, encoder->schema()));
                expect_simple_of(*batch, *batch, encoder->schema());
                // the other layout gives the same graph
                const auto other = FlatEncoder::create(
                    table, options_of(config, layout == Layout::Graph ? Layout::Relation : Layout::Graph, index, k_both));
                expect_arrays_equal(*other->encode(in), *batch, k_simple_names);
            }
}

INSTANTIATE_TEST_SUITE_P(Domains, SimpleStates, ::testing::ValuesIn(domains()));

TEST(Simple, SeveralInstancesAndTheEmptyAndSingleBatch)
{
    const Tables tables("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto& table = tables.table;
    const test::Rows rows = rows_of(*table, 9, 2);
    FlatOptions options = test::encoder_options();
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied};
    options.outputs = k_both;
    for (const IndexType index : k_indices)
    {
        options.index = index;
        const auto encoder = FlatEncoder::create(table, options);
        EncodeInput in;
        in.states = rows.view();
        in.task_ids = rows.task_ids.data();
        const auto batch = encoder->encode(in);
        validate(*batch, encoder->schema());
        expect_simple_of(*batch, *batch, encoder->schema());
        // the empty batch
        EncodeInput none;
        none.states = {rows.data.data(), 0, rows.words, 0, 0};
        const auto empty = encoder->encode(none);
        validate(*empty, encoder->schema());
        EXPECT_EQ(ints(*empty, "node_offsets"), std::vector<i64>{0});
        EXPECT_EQ(ints(*empty, "edge_offsets"), std::vector<i64>{0});
        EXPECT_TRUE(ints(*empty, "node_label").empty());
        EXPECT_TRUE(ints(*empty, "edge_label").empty());
        // single == batch: the graph of row g alone is the slice of the batch with its offsets rebased
        const auto offsets = ints(*batch, "node_offsets");
        const auto edge_offsets = ints(*batch, "edge_offsets");
        const auto labels = ints(*batch, "node_label");
        const auto edge_labels = ints(*batch, "edge_label");
        const auto index_rows = ints(*batch, "edge_index");
        const u64 edges = edge_offsets.back();
        for (u64 g = 0; g < rows.size(); ++g)
        {
            EncodeInput one;
            one.states = {rows.data.data() + g * rows.words, 1, rows.words, 0, 0};
            one.task_ids = rows.task_ids.data() + g;
            const auto single = encoder->encode(one);
            const auto node_first = static_cast<u64>(offsets[g]), node_last = static_cast<u64>(offsets[g + 1]);
            const auto edge_first = static_cast<u64>(edge_offsets[g]), edge_last = static_cast<u64>(edge_offsets[g + 1]);
            EXPECT_EQ(ints(*single, "node_label"),
                      std::vector<i64>(labels.begin() + static_cast<std::ptrdiff_t>(node_first),
                                       labels.begin() + static_cast<std::ptrdiff_t>(node_last)));
            EXPECT_EQ(ints(*single, "edge_label"),
                      std::vector<i64>(edge_labels.begin() + static_cast<std::ptrdiff_t>(edge_first),
                                       edge_labels.begin() + static_cast<std::ptrdiff_t>(edge_last)));
            const auto single_index = ints(*single, "edge_index");
            const u64 count = edge_last - edge_first;
            ASSERT_EQ(single_index.size(), 2 * count);
            for (u64 i = 0; i < count; ++i)
                for (u64 row = 0; row < 2; ++row)
                    ASSERT_EQ(single_index[row * count + i],
                              index_rows[row * edges + edge_first + i] - offsets[g]);
            EXPECT_EQ(ints(*single, "node_offsets"), (std::vector<i64>{0, offsets[g + 1] - offsets[g]}));
            EXPECT_EQ(ints(*single, "edge_offsets"), (std::vector<i64>{0, static_cast<i64>(count)}));
        }
    }
}

TEST(Simple, TheSimpleOnlyEncodingHoldsTheSameGraphWithoutRelationArrays)
{
    const Tables tables("gripper", {"p01", "p02"});
    const auto& table = tables.table;
    const test::Rows rows = rows_of(*table, 20, 7);
    for (const Config& config : configs())
        for (const IndexType index : k_indices)
        {
            SCOPED_TRACE(config.name);
            const auto both = FlatEncoder::create(table, options_of(config, Layout::Graph, index, k_both));
            const auto only = FlatEncoder::create(table, options_of(config, Layout::Graph, index, k_simple));
            const Layers layers(rows, config.levels, config.steps, 11);
            EncodeInput in;
            in.states = rows.view();
            in.task_ids = rows.task_ids.data();
            layers.attach(in);
            const auto a = both->encode(in);
            const auto b = only->encode(in);
            validate(*b, only->schema());
            expect_arrays_equal(*a, *b, k_simple_names);
            expect_arrays_equal(*a, *b, {"task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref"});
            for (const char* name : {"relation_counts", "relation_offsets", "relation_arg_offsets", "relation_args"})
            {
                EXPECT_NE(a->find(name), nullptr) << name;
                EXPECT_EQ(b->find(name), nullptr) << name;
            }
            // the scratch is reused by calls of other sizes
            EncodeInput part = in;
            part.states.rows = 3;
            EncodeScratch scratch;
            ThreadPool pool(1);
            (void)only->encode(in, pool, scratch);
            const auto small = only->encode(part, pool, scratch);
            validate(*small, only->schema());
            const auto big = only->encode(in, pool, scratch);
            expect_arrays_equal(*big, *b, k_simple_names);
        }
}

TEST(Simple, ThreadsGiveTheSameBatchAndTheCapacityFormHoldsItInItsFirstEntries)
{
    const Tables tables("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto& table = tables.table;
    const test::Rows rows = rows_of(*table, 1400, 3);
    for (const Outputs outputs : {k_both, k_simple})
    {
        FlatOptions options = test::encoder_options();
        options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied};
        options.outputs = outputs;
        const auto encoder = FlatEncoder::create(table, options);
        EncodeInput in;
        in.states = rows.view();
        in.task_ids = rows.task_ids.data();
        const auto expected = encoder->encode(in);
        EncodeScratch scratch;
        ThreadPool pool(4);
        const auto threaded = encoder->encode(in, pool, scratch);
        expect_arrays_equal(*threaded, *expected, k_simple_names);
        // capacities
        const auto entities = static_cast<u64>(ints(*expected, "entity_offsets").back());
        const auto nodes = static_cast<u64>(ints(*expected, "node_offsets").back());
        const auto edges = static_cast<u64>(ints(*expected, "edge_offsets").back());
        const u64 args = edges / 2;
        FlatSizes sizes;
        const BatchShape capacities{rows.size(), entities + 17, args + 31, 0, 0};
        const auto padded = encoder->encode_capacity(in, capacities, pool, scratch, sizes);
        ASSERT_FALSE(sizes.overflowed());
        EXPECT_EQ(sizes.entities + sizes.tuples, nodes);
        validate(*padded, encoder->schema());
        const auto padded_labels = ints(*padded, "node_label");
        const auto labels = ints(*expected, "node_label");
        ASSERT_EQ(padded_labels.size(), entities + 17 + args + 31);
        EXPECT_TRUE(std::equal(labels.begin(), labels.end(), padded_labels.begin()));
        EXPECT_TRUE(std::all_of(padded_labels.begin() + static_cast<std::ptrdiff_t>(nodes), padded_labels.end(),
                                [](i64 v) { return v == 0; }));
        const auto padded_index = ints(*padded, "edge_index");
        const auto index = ints(*expected, "edge_index");
        const u64 stride = 2 * (args + 31);
        ASSERT_EQ(padded_index.size(), 2 * stride);
        for (u64 row = 0; row < 2; ++row)
            for (u64 i = 0; i < stride; ++i)
                ASSERT_EQ(padded_index[row * stride + i], i < edges ? index[row * edges + i] : 0);
        EXPECT_EQ(ints(*padded, "node_offsets"), ints(*expected, "node_offsets"));
        // a capacity that is too small leaves the batch all zero
        const BatchShape small{rows.size(), entities, args - 1, 0, 0};
        const auto overflowed = encoder->encode_capacity(in, small, pool, scratch, sizes);
        EXPECT_TRUE(sizes.overflowed());
        for (const char* name : k_simple_names)
        {
            const auto values = ints(*overflowed, name);
            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](i64 v) { return v == 0; })) << name;
        }
    }
}

TEST(Simple, StreamsFlushTheBatchEncodeGives)
{
    const Tables tables("blocks", {"smedium", "probBLOCKS-8-0"});
    const auto& table = tables.table;
    const test::Rows rows = rows_of(*table, 30, 4);
    FlatOptions options = test::encoder_options();
    options.outputs = k_both;
    const auto encoder = FlatEncoder::create(table, options);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto expected = encoder->encode(in);
    ThreadPool pool(2);
    FlatStream stream(encoder);
    for (u64 g = 0; g < rows.size(); g += 7)
    {
        EncodeInput part;
        const u64 count = std::min<u64>(7, rows.size() - g);
        part.states = {rows.data.data() + g * rows.words, count, rows.words, 0, 0};
        part.task_ids = rows.task_ids.data() + g;
        stream.append(part);
    }
    expect_arrays_equal(*stream.flush(pool), *expected, k_simple_names);
    FlatMutableStream mutable_stream(encoder);
    std::vector<i64> indices(rows.size());
    for (u64 g = 0; g < rows.size(); ++g)
        indices[g] = static_cast<i64>(g);
    mutable_stream.append(indices, in);
    const auto flushed = mutable_stream.flush(pool);
    validate(*flushed, encoder->schema());
    expect_arrays_equal(*flushed, *expected, k_simple_names);
}

class SimpleHorizons : public ::testing::TestWithParam<std::tuple<HorizonMode, RootPolicy, bool>>
{
};

TEST_P(SimpleHorizons, AreTheIncidenceGraphOfTheirTuples)
{
    const auto [mode, root, relations] = GetParam();
    const Tables tables("blocks", {"smedium"});
    const auto& table = tables.table;
    const test::Rows roots = test::random_walk(*table, 0, 6, 9);
    const test::Horizon horizon = test::horizon_of(*table, roots, 2, 5);
    for (const Layout layout : k_layouts)
        for (const IndexType index : k_indices)
        {
            FlatOptions options = test::encoder_options();
            options.goal_derivations = mode == HorizonMode::Delta
                                           ? GoalDerivations{GoalDerivation::Plain, GoalDerivation::Satisfied,
                                                             GoalDerivation::AddedSatisfied,
                                                             GoalDerivation::AddedUnsatisfied}
                                           : GoalDerivations{GoalDerivation::Plain, GoalDerivation::Satisfied};
            options.horizon.mode = mode;
            options.horizon.root = root;
            options.horizon.parent_relation = relations;
            options.horizon.sibling_relation = relations;
            options.horizon.cousin_relation = relations;
            options.layout = layout;
            options.index = index;
            options.outputs = k_both;
            const auto encoder = FlatEncoder::create(table, options);
            const auto batch = encoder->encode(horizon.input());
            ASSERT_NO_THROW(validate(*batch, encoder->schema()));
            expect_simple_of(*batch, *batch, encoder->schema());
            options.outputs = k_simple;
            options.layout = Layout::Graph;
            const auto only = FlatEncoder::create(table, options);
            expect_arrays_equal(*only->encode(horizon.input()), *batch, k_simple_names);
        }
}

INSTANTIATE_TEST_SUITE_P(Modes, SimpleHorizons,
                         ::testing::Combine(::testing::Values(HorizonMode::Full, HorizonMode::Delta, HorizonMode::Action),
                                            ::testing::Values(RootPolicy::Include, RootPolicy::Exclude,
                                                              RootPolicy::EncodeOnly),
                                            ::testing::Bool()));

TEST(Simple, TheBuiltGraphsOfTransitionsAndHorizonsAreEncoded)
{
    const Tables tables("gripper", {"p01"});
    const auto& table = tables.table;
    const test::Rows roots = test::random_walk(*table, 0, 5, 1);
    FlatOptions options = test::encoder_options();
    options.horizon.mode = HorizonMode::Full;
    options.horizon.parent_relation = true;
    options.outputs = k_both;
    const auto encoder = FlatEncoder::create(table, options);
    const HorizonGraphs graphs = build_horizon_graphs(*table, roots.view(), roots.task_ids.data(), HorizonBuild{2});
    EncodeInput in;
    in.horizon = graphs.input();
    in.task_ids = roots.task_ids.data();
    const auto batch = encoder->encode(in);
    validate(*batch, encoder->schema());
    expect_simple_of(*batch, *batch, encoder->schema());
}

TEST(Simple, LabelsAreTheSameForEveryInstanceOfADomain)
{
    // the relations of a domain have one id in every instance, so an atom's label does too
    const Tables tables("blocks", {"smedium", "probBLOCKS-8-0"});
    FlatOptions options = test::encoder_options();
    options.outputs = k_simple;
    const auto together = FlatEncoder::create(tables.table, options);
    const auto first = FlatEncoder::create(rl::TaskTable::single(tables.table->task(0)), options);
    const auto second = FlatEncoder::create(rl::TaskTable::single(tables.table->task(1)), options);
    EXPECT_EQ(together->schema().simple_label_names(), first->schema().simple_label_names());
    EXPECT_EQ(together->schema().simple_label_names(), second->schema().simple_label_names());
    EXPECT_EQ(together->schema().fingerprint(), first->schema().fingerprint());
    // the initial states: the labels of the atom nodes name the same relations
    const auto names = together->schema().simple_label_names();
    ASSERT_EQ(names.size(), 9 + together->schema().num_relations());
    EXPECT_EQ(names[0], "object");
    EXPECT_EQ(names[8], "auxiliary");
    for (u32 r = 0; r < together->schema().num_relations(); ++r)
        EXPECT_EQ(names[9 + r], together->schema().relation(r).name);
    const test::Rows rows = rows_of(*tables.table, 4, 1);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto batch = together->encode(in);
    const auto labels = ints(*batch, "node_label");
    EXPECT_TRUE(std::all_of(labels.begin(), labels.end(), [&](i64 v) { return v >= 0 && v < 9 + together->schema().num_relations(); }));
    // the instance alone gives the same labels as its graphs in the shared encoding
    test::Rows one;
    one.words = rows.words;
    one.data.assign(rows.data.begin(), rows.data.begin() + rows.words);
    one.task_ids = {0};
    EncodeInput single;
    single.states = one.view();
    single.task_ids = one.task_ids.data();
    const auto alone = first->encode({.states = one.view()});
    const auto shared = together->encode(single);
    EXPECT_EQ(ints(*alone, "node_label"), ints(*shared, "node_label"));
}

TEST(Simple, ConstantsGetLabelsOfTheirOwnWhenAsked)
{
    const Tables tables("gripper", {"p01", "p02"});
    FlatOptions options = test::encoder_options();
    options.outputs = k_both;
    const auto plain = FlatEncoder::create(tables.table, options);
    options.constant_labels = true;
    const auto labelled = FlatEncoder::create(tables.table, options);
    const u32 constants = labelled->schema().simple_constants();
    ASSERT_GT(constants, 0u);
    EXPECT_EQ(plain->schema().simple_constants(), 0u);
    EXPECT_NE(plain->schema().fingerprint(), labelled->schema().fingerprint());
    const auto names = labelled->schema().simple_label_names();
    ASSERT_EQ(names.size(), 9 + labelled->schema().num_relations() + constants);
    const test::Rows rows = rows_of(*tables.table, 5, 1);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto batch_plain = plain->encode(in);
    const auto batch_labelled = labelled->encode(in);
    validate(*batch_labelled, labelled->schema());
    expect_simple_of(*batch_labelled, *batch_labelled, labelled->schema());
    const auto entity_offsets = ints(*batch_labelled, "entity_offsets");
    const auto offsets = ints(*batch_labelled, "node_offsets");
    const auto a = ints(*batch_plain, "node_label");
    const auto b = ints(*batch_labelled, "node_label");
    ASSERT_EQ(a.size(), b.size());
    const u64 R = labelled->schema().num_relations();
    u64 changed = 0;
    for (u64 g = 0; g + 1 < offsets.size(); ++g)
        for (i64 j = 0; j < entity_offsets[g + 1] - entity_offsets[g]; ++j)
        {
            const auto at = static_cast<u64>(offsets[g] + j);
            if (j < constants)
            {
                EXPECT_EQ(a[at], 0);
                EXPECT_EQ(b[at], static_cast<i64>(9 + R) + j);
                ++changed;
            }
            else
                EXPECT_EQ(a[at], b[at]);
        }
    EXPECT_EQ(changed, rows.size() * constants);
    // constants have the same label in every instance
    EXPECT_EQ(b[static_cast<u64>(offsets[0])], b[static_cast<u64>(offsets[rows.size() - 1])]);
}

TEST(Simple, RefusesWhatItCannotRepresent)
{
    const Tables tables("blocks", {"smedium"});
    auto refused = [&](const std::function<void(FlatOptions&)>& change, const char* fragment)
    {
        FlatOptions options = test::encoder_options();
        change(options);
        try
        {
            (void)FlatEncoder::create(tables.table, options);
            ADD_FAILURE() << "accepted: " << fragment;
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find(fragment), std::string::npos) << error.what();
        }
    };
    refused([](FlatOptions& o) { o.outputs = k_both, o.predicate_virtual_nodes = true; }, "predicate_virtual_nodes");
    refused([](FlatOptions& o) { o.outputs = {}; }, "at least one");
    refused([](FlatOptions& o) { o.constant_labels = true; }, "constant_labels");
    refused([](FlatOptions& o) { o.outputs = k_both, o.entity_arrays = EntityArrays::Auto; }, "entity_arrays");
    refused([](FlatOptions& o) { o.outputs = k_simple, o.layout = Layout::Relation; }, "layout");
    refused([](FlatOptions& o) { o.outputs = k_simple, o.emit_counts = false; }, "emit_counts");
}

TEST(Simple, ValidateRejectsAGraphThatIsNotTheIncidenceGraphOfItsTuples)
{
    const Tables tables("blocks", {"smedium"});
    FlatOptions options = test::encoder_options();
    options.outputs = k_both;
    const auto encoder = FlatEncoder::create(tables.table, options);
    const test::Rows rows = rows_of(*tables.table, 4, 1);
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto batch = encoder->encode(in);
    ASSERT_NO_THROW(validate(*batch, encoder->schema()));
    const u64 nodes = static_cast<u64>(ints(*batch, "node_offsets").back());
    const u64 edges = static_cast<u64>(ints(*batch, "edge_offsets").back());
    ASSERT_GT(edges, 4u);
    auto rejected = [&](const rl::ArrayBundle& changed, const char* what)
    { EXPECT_THROW(validate(changed, encoder->schema()), std::invalid_argument) << what; };
    rejected(test::with_change(*batch, "node_offsets", 1, 1), "node offset");
    rejected(test::with_change(*batch, "edge_offsets", 1, -1), "edge offset");
    rejected(test::with_change(*batch, "node_label", nodes - 1, 1), "atom label");
    rejected(test::with_value(*batch, "node_label", 0, 99), "label out of range");
    rejected(test::with_change(*batch, "edge_label", 0, 1), "edge label");
    rejected(test::with_change(*batch, "edge_label", edges - 1, 1), "reverse edge label");
    rejected(test::with_change(*batch, "edge_index", 0, 1), "edge source");
    rejected(test::with_change(*batch, "edge_index", edges + 1, 1), "edge target");
    rejected(test::with_value(*batch, "edge_index", 2, static_cast<i64>(nodes)), "node out of range");
    rejected(test::with_change(*batch, "relation_args", 0, 1), "arguments that differ from the edges");
}

TEST(Simple, TheSchemaNamesItsVocabulary)
{
    const Tables tables("blocks", {"smedium"});
    FlatOptions options = test::encoder_options();
    options.outputs = k_both;
    const auto schema = EncodingSchema::create(*tables.table, options);
    EXPECT_TRUE(schema->emits_simple());
    EXPECT_TRUE(schema->emits_relations());
    u32 widest = 0;
    for (const Relation& relation : schema->relations())
        widest = std::max(widest, relation.arity);
    EXPECT_EQ(schema->simple_max_arity(), widest);
    EXPECT_EQ(schema->simple_edge_labels(), 2 * widest);
    const auto edge_names = schema->simple_edge_label_names();
    ASSERT_EQ(edge_names.size(), 2 * widest);
    EXPECT_EQ(edge_names.front(), "0");
    EXPECT_EQ(edge_names[widest], "0^-1");
    // the relations alone keep their fingerprint
    EXPECT_NE(schema->fingerprint(), EncodingSchema::create(*tables.table, test::encoder_options())->fingerprint());
}
}  // namespace
