// The action lane and the predicate virtual nodes of the relational encoder.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
struct Problem
{
    const char* domain;
    const char* problem;
};
const std::vector<Problem> k_problems = {{"blocks", "smedium"}, {"gripper", "p01"},       {"logistics00", "probLOGISTICS-6-1"},
                                         {"relay", "p01"},      {"features", "p01"},      {"blocks_flags", "p01"},
                                         {"philosophers", "p03-phil4"}};

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
            case rl::DType::I32: { i32 v; std::memcpy(&v, at, 4); out.push_back(v); break; }
            case rl::DType::I64: { i64 v; std::memcpy(&v, at, 8); out.push_back(v); break; }
            default: ADD_FAILURE() << "unexpected dtype of " << name;
        }
    }
    return out;
}

/// The expansion of every row of a batch, with the arrays it points at.
struct Expanded
{
    rl::Expansion x;
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
};

std::shared_ptr<Expanded> expand_rows(const rl::TaskTable& table, const test::Rows& rows, rl::ExpandOptions options = {})
{
    auto e = std::make_shared<Expanded>();
    u64 capacity = rows.size() * 8 + 16;
    for (;;)
    {
        e->succ.assign(capacity * rows.words, 0);
        e->parent.assign(capacity, 0);
        e->schema.assign(capacity, 0);
        e->binding.assign(capacity * table.label_width(), 0);
        e->offsets.assign(rows.size() + 1, 0);
        e->x = {};
        e->x.capacity = capacity;
        e->x.words = rows.words;
        e->x.label_width = table.label_width();
        e->x.succ = e->succ.data();
        e->x.parent = e->parent.data();
        e->x.schema = e->schema.data();
        e->x.binding = e->binding.data();
        e->x.offsets = e->offsets.data();
        rl::expand(table, rows.view(), rows.task_ids.data(), e->x, options);
        if (!e->x.overflow())
            return e;
        capacity = std::max(capacity * 2, e->x.total);
    }
}

FlatOptions action_options()
{
    FlatOptions options = test::encoder_options();
    options.targets = {TargetSource::Action};
    return options;
}

EncodeInput input_of(const test::Rows& rows, const Expanded* expanded = nullptr)
{
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    in.actions = expanded ? &expanded->x : nullptr;
    return in;
}

using Binding = std::vector<i32>;

/// The distinct ground actions of state g by action relation, each with the first expansion row that has it.
std::vector<std::map<Binding, i32>> expected_actions(const FlatEncoder& encoder, const Expanded& e, u64 g)
{
    const auto& relation_of = encoder.schema().schema_relations();
    std::vector<std::map<Binding, i32>> out(encoder.schema().num_relations());
    const u32 L = e.x.label_width;
    for (i32 row = e.offsets[g]; row < e.offsets[g + 1]; ++row)
    {
        const u32 schema = static_cast<u32>(e.schema[static_cast<usize>(row)]);
        Binding b(e.binding.begin() + static_cast<std::ptrdiff_t>(static_cast<u64>(row) * L),
                  e.binding.begin() + static_cast<std::ptrdiff_t>(static_cast<u64>(row) * L + encoder.table().schema_arities()[schema]));
        out[static_cast<usize>(relation_of[schema])].emplace(std::move(b), row);
    }
    return out;
}

void expect_actions_match(const FlatEncoder& encoder, const test::Rows& rows, const Expanded& e,
                          const rl::ArrayBundle& batch)
{
    const EncodingSchema& schema = encoder.schema();
    const u64 R = schema.num_relations();
    const auto entity_offsets = ints(batch, "entity_offsets");
    const auto role = ints(batch, "entity_role");
    const auto ref = ints(batch, "entity_ref");
    const auto counts = ints(batch, "relation_counts");
    const auto arg_offsets = ints(batch, "relation_arg_offsets");
    const auto args = ints(batch, "relation_args");
    const bool relation_major = schema.options().layout == Layout::Relation;
    const u64 G = rows.size();
    u64 total_actions = 0;
    for (u64 g = 0; g < G; ++g)
    {
        const auto expected = expected_actions(encoder, e, g);
        const i64 base = entity_offsets[g];
        for (u32 r = 0; r < R; ++r)
        {
            if (schema.relation(r).lane != Lane::Action)
                continue;
            const u64 cell = relation_major ? r * G + g : g * R + r;
            ASSERT_EQ(static_cast<usize>(counts[g * R + r]), expected[r].size()) << schema.relation(r).name;
            auto it = expected[r].begin();
            for (i64 t = 0; t < counts[g * R + r]; ++t, ++it)
            {
                const u32 arity = schema.relation(r).arity;
                const i64* tuple = args.data() + arg_offsets[cell] + t * arity;
                ASSERT_EQ(role[static_cast<usize>(tuple[0])], static_cast<i64>(EntityRole::Action));
                // the objects, in ascending order of the binding
                for (u32 k = 0; k + 1 < arity; ++k)
                    ASSERT_EQ(tuple[1 + k] - base, it->first[k]) << schema.relation(r).name;
                // the action entity refers to the first row that has the ground action
                EXPECT_EQ(ref[static_cast<usize>(tuple[0])], it->second);
                ++total_actions;
            }
        }
    }
    const auto targets = ints(batch, "target_entity");
    const auto groups = ints(batch, "target_group");
    EXPECT_EQ(targets.size(), total_actions);
    for (i64 group : groups)
        EXPECT_EQ(group, static_cast<i64>(TargetGroup::Action));
}

TEST(EncoderActions, TuplesAreTheDistinctGroundActionsOfTheExpansionRows)
{
    for (const Problem& p : k_problems)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const test::Rows rows = test::random_walk(*table, 0, 64, 11);
        const auto expanded = expand_rows(*table, rows);
        for (const Layout layout : {Layout::Graph, Layout::Relation})
        {
            FlatOptions options = action_options();
            options.layout = layout;
            const auto encoder = FlatEncoder::create(table, options);
            const auto batch = encoder->encode(input_of(rows, expanded.get()));
            validate(*batch, encoder->schema());
            expect_actions_match(*encoder, rows, *expanded, *batch);
        }
    }
}

TEST(EncoderActions, WitnessActionsHoldTheFullObjectList)
{
    const auto table = rl::TaskTable::single(test::pddl_task("relay", "p01"));
    const auto encoder = FlatEncoder::create(table, action_options());
    const u32 step = *encoder->schema().find("step");
    EXPECT_EQ(encoder->schema().relation(step).logical_arity, 3u);
    EXPECT_EQ(encoder->schema().relation(step).arity, 4u);
    const test::Rows rows = test::random_walk(*table, 0, 32, 2);
    const auto expanded = expand_rows(*table, rows);
    const auto batch = encoder->encode(input_of(rows, expanded.get()));
    validate(*batch, encoder->schema());
    const auto counts = ints(*batch, "relation_counts");
    u64 steps = 0;
    for (u64 g = 0; g < rows.size(); ++g)
        steps += static_cast<u64>(counts[g * encoder->schema().num_relations() + step]);
    EXPECT_GT(steps, 0u);
    // with witness pruning the lane is the pruned set
    rl::ExpandOptions pruned;
    pruned.witness_pruning = true;
    const auto few = expand_rows(*table, rows, pruned);
    const auto few_batch = encoder->encode(input_of(rows, few.get()));
    validate(*few_batch, encoder->schema());
    expect_actions_match(*encoder, rows, *few, *few_batch);
    const auto few_counts = ints(*few_batch, "relation_counts");
    u64 pruned_steps = 0;
    for (u64 g = 0; g < rows.size(); ++g)
        pruned_steps += static_cast<u64>(few_counts[g * encoder->schema().num_relations() + step]);
    EXPECT_LE(pruned_steps, steps);
}

TEST(EncoderActions, NoExpansionMeansNoActionTuples)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto encoder = FlatEncoder::create(table, action_options());
    const test::Rows rows = test::random_walk(*table, 0, 16, 1);
    const auto batch = encoder->encode(input_of(rows));
    validate(*batch, encoder->schema());
    EXPECT_TRUE(ints(*batch, "target_entity").empty());
    const auto role = ints(*batch, "entity_role");
    EXPECT_EQ(std::count(role.begin(), role.end(), static_cast<i64>(EntityRole::Action)), 0);
}

/// A domain whose action has a disjunctive precondition, so that the front end splits it into two schemas of one name.
rl::TaskTablePtr disjunction_table()
{
    const std::filesystem::path directory = std::filesystem::path(::testing::TempDir());
    const std::filesystem::path domain = directory / "disjunction_domain.pddl";
    const std::filesystem::path problem = directory / "disjunction_problem.pddl";
    std::ofstream(domain) << "(define (domain disjunction) (:requirements :strips :typing :disjunctive-preconditions)\n"
                             " (:types item) (:predicates (p ?x - item) (q ?x - item) (done ?x - item))\n"
                             " (:action go :parameters (?x - item) :precondition (or (p ?x) (q ?x))\n"
                             "  :effect (done ?x)))\n";
    std::ofstream(problem) << "(define (problem d) (:domain disjunction) (:objects a b c - item)\n"
                              " (:init (p a) (q a) (p b) (q c)) (:goal (and (done a) (done b) (done c))))\n";
    const auto task = Task::create(*frontend::load_task(domain, problem));
    std::filesystem::remove(domain);
    std::filesystem::remove(problem);
    return rl::TaskTable::single(task);
}

TEST(EncoderActions, SchemasOfOneNameAreMergedAndDuplicatesDropped)
{
    const auto table = disjunction_table();
    ASSERT_GT(table->num_schemas(), 1u);
    ASSERT_EQ(table->schema_names()[0], table->schema_names()[1]);
    FlatOptions options = action_options();
    options.goal_derivations = {};
    const auto encoder = FlatEncoder::create(table, options);
    const u32 go = *encoder->schema().find("go");
    EXPECT_EQ(encoder->schema().schema_relations()[0], static_cast<i32>(go));
    EXPECT_EQ(encoder->schema().schema_relations()[1], static_cast<i32>(go));
    const test::Rows rows = test::random_walk(*table, 0, 1, 1);  // the initial state
    const auto expanded = expand_rows(*table, rows);
    // (go a) is applicable through both disjuncts: more rows than distinct ground actions
    ASSERT_EQ(expanded->offsets[1], 4);
    const auto batch = encoder->encode(input_of(rows, expanded.get()));
    validate(*batch, encoder->schema());
    EXPECT_EQ(ints(*batch, "relation_counts")[go], 3);
    expect_actions_match(*encoder, rows, *expanded, *batch);
    // the entity of (go a) refers to its first row
    const auto ref = ints(*batch, "entity_ref");
    const auto role = ints(*batch, "entity_role");
    std::set<i64> refs;
    for (usize e = 0; e < role.size(); ++e)
        if (role[e] == static_cast<i64>(EntityRole::Action))
            refs.insert(ref[e]);
    EXPECT_EQ(refs.size(), 3u);
    EXPECT_EQ(*refs.begin(), 0);
}

TEST(EncoderActions, ARunThatIsNotInCanonicalOrderIsRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto encoder = FlatEncoder::create(table, action_options());
    const test::Rows rows = test::random_walk(*table, 0, 8, 4);
    auto expanded = expand_rows(*table, rows);
    ASSERT_NO_THROW((void)encoder->encode(input_of(rows, expanded.get())));
    // a state with two rows of one schema: swap their bindings
    u64 g = 0;
    auto has_pair = [&](u64 state)
    {
        for (i32 row = expanded->offsets[state] + 1; row < expanded->offsets[state + 1]; ++row)
            if (expanded->schema[static_cast<usize>(row)] == expanded->schema[static_cast<usize>(row) - 1])
                return true;
        return false;
    };
    while (g < rows.size() && !has_pair(g))
        ++g;
    ASSERT_LT(g, rows.size());
    const u32 L = expanded->x.label_width;
    for (i32 row = expanded->offsets[g] + 1; row < expanded->offsets[g + 1]; ++row)
        if (expanded->schema[static_cast<usize>(row)] == expanded->schema[static_cast<usize>(row) - 1])
        {
            for (u32 k = 0; k < L; ++k)
                std::swap(expanded->binding[static_cast<usize>(row) * L + k], expanded->binding[static_cast<usize>(row - 1) * L + k]);
            break;
        }
    try
    {
        (void)encoder->encode(input_of(rows, expanded.get()));
        FAIL() << "accepted";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("mymyr: encoding: the action rows of state " + std::to_string(g)), std::string::npos)
            << error.what();
        EXPECT_NE(std::string(error.what()).find("canonical order"), std::string::npos) << error.what();
    }
}

TEST(EncoderActions, ForeignLabelsAndMismatchedStatesAreRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    const auto encoder = FlatEncoder::create(table, action_options());
    const test::Rows rows = test::random_walk(*table, 0, 8, 4);
    auto expanded = expand_rows(*table, rows);
    auto refused = [&](const std::string& what)
    {
        try
        {
            (void)encoder->encode(input_of(rows, expanded.get()));
            ADD_FAILURE() << "accepted: " << what;
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find("mymyr: encoding: "), std::string::npos) << error.what();
        }
    };
    const i32 object = expanded->binding[0];
    expanded->binding[0] = static_cast<i32>(table->instance(0).num_objects);
    refused("an object out of range");
    expanded->binding[0] = object;
    expanded->parent[0] = 5;
    refused("another state's row");
    expanded->parent[0] = 0;
    expanded->x.label_width = 0;
    refused("a narrow label width");
    expanded->x.label_width = table->label_width();
    ASSERT_NO_THROW((void)encoder->encode(input_of(rows, expanded.get())));
    FlatOptions without = test::encoder_options();
    without.include_actions = false;
    EXPECT_THROW((void)FlatEncoder::create(table, without)->encode(input_of(rows, expanded.get())), std::invalid_argument);
}

TEST(EncoderActions, LayoutsThreadsAndCapacitiesAgree)
{
    const auto table = rl::TaskTable::create({test::pddl_task("blocks", "smedium"), test::pddl_task("blocks", "probBLOCKS-8-0")});
    test::Rows rows;
    rows.words = table->words();
    for (u32 i = 0; i < table->size(); ++i)
    {
        const test::Rows part = test::random_walk(*table, i, 1500, 3 + i);
        rows.data.insert(rows.data.end(), part.data.begin(), part.data.end());
        rows.task_ids.insert(rows.task_ids.end(), part.task_ids.begin(), part.task_ids.end());
    }
    const auto expanded = expand_rows(*table, rows);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions options = action_options();
        options.layout = layout;
        options.predicate_virtual_nodes = true;
        const auto encoder = FlatEncoder::create(table, options);
        const auto single = encoder->encode(input_of(rows, expanded.get()));
        validate(*single, encoder->schema());
        for (u32 threads : {2u, 5u})
        {
            ThreadPool pool(threads);
            EncodeScratch scratch;
            const auto parallel = encoder->encode(input_of(rows, expanded.get()), pool, scratch);
            ASSERT_EQ(single->bytes(), parallel->bytes());
            EXPECT_EQ(std::memcmp(single->block(), parallel->block(), single->bytes()), 0) << threads;
        }
        FlatSizes sizes;
        ThreadPool pool(3);
        EncodeScratch scratch;
        const FlatCapacities capacities{single->find("entity_role")->elements() + 7, single->find("relation_args")->elements() + 5,
                                        single->find("relation_args")->elements() + 5, single->find("target_entity")->elements() + 3};
        const auto roomy = encoder->encode_capacity(input_of(rows, expanded.get()), capacities, pool, scratch, sizes);
        EXPECT_FALSE(sizes.overflowed());
        EXPECT_EQ(ints(*roomy, "relation_args").size(), capacities.args);
        const auto exact = ints(*single, "relation_args");
        const auto padded = ints(*roomy, "relation_args");
        EXPECT_TRUE(std::equal(exact.begin(), exact.end(), padded.begin()));
        const auto roomy_ref = ints(*roomy, "entity_ref");
        EXPECT_EQ(ints(*single, "entity_ref"),
                  std::vector<i64>(roomy_ref.begin(), roomy_ref.begin() + static_cast<std::ptrdiff_t>(sizes.entities)));
    }
}

TEST(EncoderActions, ASingleStateEqualsItsGraphInABatch)
{
    const auto table = rl::TaskTable::single(test::pddl_task("logistics00", "probLOGISTICS-6-1"));
    const FlatOptions options = action_options();
    const auto encoder = FlatEncoder::create(table, options);
    const test::Rows rows = test::random_walk(*table, 0, 20, 8);
    const auto expanded = expand_rows(*table, rows);
    const auto batch = encoder->encode(input_of(rows, expanded.get()));
    const u64 R = encoder->schema().num_relations();
    const auto batch_counts = ints(*batch, "relation_counts");
    const auto batch_args = ints(*batch, "relation_args");
    const auto batch_arg_offsets = ints(*batch, "relation_arg_offsets");
    const auto batch_entities = ints(*batch, "entity_offsets");
    const auto batch_ref = ints(*batch, "entity_ref");
    for (u64 g = 0; g < rows.size(); ++g)
    {
        test::Rows one;
        one.words = rows.words;
        one.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(g * rows.words),
                        rows.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * rows.words));
        one.task_ids = {0};
        const auto single = expand_rows(*table, one);
        const auto alone = encoder->encode(input_of(one, single.get()));
        const auto counts = ints(*alone, "relation_counts");
        ASSERT_EQ(std::vector<i64>(batch_counts.begin() + static_cast<std::ptrdiff_t>(g * R),
                                   batch_counts.begin() + static_cast<std::ptrdiff_t>((g + 1) * R)),
                  counts);
        const auto args = ints(*alone, "relation_args");
        const std::vector<i64> slice(batch_args.begin() + batch_arg_offsets[g * R], batch_args.begin() + batch_arg_offsets[(g + 1) * R]);
        std::vector<i64> rebased;
        for (i64 a : slice)
            rebased.push_back(a - batch_entities[g]);
        EXPECT_EQ(rebased, args);
        // the batch's references count rows over the whole expansion; the single's over its own
        const auto ref = ints(*alone, "entity_ref");
        const auto alone_role = ints(*alone, "entity_role");
        for (usize e = 0; e < ref.size(); ++e)
        {
            if (alone_role[e] == static_cast<i64>(EntityRole::Action))
            {
                EXPECT_EQ(batch_ref[static_cast<usize>(batch_entities[g]) + e], ref[e] + expanded->offsets[g]);
            }
        }
    }
}

TEST(EncoderVirtualNodes, OneNodePerUsedPredicateInCatalogOrder)
{
    for (const Problem& p : k_problems)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        FlatOptions options = action_options();
        options.predicate_virtual_nodes = true;
        options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
        options.targets = {TargetSource::Goal, TargetSource::Action};
        const auto encoder = FlatEncoder::create(table, options);
        const EncodingSchema& schema = encoder->schema();
        const test::Rows rows = test::random_walk(*table, 0, 64, 21);
        const auto expanded = expand_rows(*table, rows);
        const auto batch = encoder->encode(input_of(rows, expanded.get()));
        validate(*batch, schema);
        const u64 R = schema.num_relations();
        const auto counts = ints(*batch, "relation_counts");
        const auto entity_offsets = ints(*batch, "entity_offsets");
        const auto role = ints(*batch, "entity_role");
        const auto ref = ints(*batch, "entity_ref");
        bool some_unused = false;
        for (u64 g = 0; g < rows.size(); ++g)
        {
            std::set<i64> used;
            for (u32 r = 0; r < R; ++r)
                if (counts[g * R + r] > 0 && schema.virtual_key(r) >= 0)
                    used.insert(schema.virtual_key(r));
            std::vector<i64> nodes;
            for (i64 e = entity_offsets[g]; e < entity_offsets[g + 1]; ++e)
                if (role[static_cast<usize>(e)] == static_cast<i64>(EntityRole::PredicateVirtual))
                    nodes.push_back(ref[static_cast<usize>(e)]);
            EXPECT_EQ(std::vector<i64>(used.begin(), used.end()), nodes) << "graph " << g;
            std::set<i64> all;
            for (u32 r = 0; r < R; ++r)
                if (schema.virtual_key(r) >= 0)
                    all.insert(schema.virtual_key(r));
            some_unused = some_unused || used.size() < all.size();
        }
        EXPECT_TRUE(some_unused) << "every predicate is used in every graph";
    }
}

TEST(EncoderVirtualNodes, TheKeyIsThePredicatesStateRelationWhenThereIsOne)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    FlatOptions options = test::encoder_options();
    options.predicate_virtual_nodes = true;
    options.goal_derivations = {GoalDerivation::Plain};
    const auto schema = EncodingSchema::create(*table, options);
    for (const char* name : {"on", "clear", "handempty"})
    {
        const u32 state = *schema->find(name);
        EXPECT_EQ(schema->virtual_key(state), static_cast<i32>(state));
        EXPECT_EQ(schema->virtual_key(*schema->find(std::string("[+]") + name + "[g]")), static_cast<i32>(state));
    }
}
}  // namespace
