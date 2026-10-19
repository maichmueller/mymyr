#include "../support/suite.hpp"
#include "mymyr/encoding/schema.hpp"

#include <gtest/gtest.h>

#include <utility>

#include <algorithm>
#include <random>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
DomainSignature blocks()
{
    DomainSignature d;
    d.predicates = {{"number", 1, PredicateKind::Static},  {"object", 1, PredicateKind::Static},
                    {"on", 2, PredicateKind::Fluent},      {"ontable", 1, PredicateKind::Fluent},
                    {"clear", 1, PredicateKind::Fluent},   {"handempty", 0, PredicateKind::Fluent},
                    {"holding", 1, PredicateKind::Fluent}, {"above", 2, PredicateKind::Derived}};
    d.schemas = {{"pick-up", 1}, {"put-down", 1}, {"stack", 2}, {"unstack", 2}};
    return d;
}

std::vector<std::string> names(const EncodingSchema& schema)
{
    std::vector<std::string> out;
    for (const Relation& relation : schema.relations())
        out.push_back(relation.name);
    return out;
}

bool has(const EncodingSchema& schema, std::string_view name) { return schema.find(name).has_value(); }

std::vector<SlotRole> roles(const EncodingSchema& schema, std::string_view name)
{
    const auto span = schema.slot_roles(*schema.find(name));
    return {span.begin(), span.end()};
}

using R = SlotRole;

constexpr u64 PIN_DEFAULT = 0x0f88d5dff3ee07cf, PIN_DELTA = 0x5a56537a247d53c5, PIN_TARGETS = 0x31e6ae6fd252e8a1,
                PIN_FULL = 0x8f230d9913e98bf7;

TEST(EncodingSchema, DefaultCatalogIsSortedBytewise)
{
    const auto schema = EncodingSchema::create(blocks());
    const auto all = names(*schema);
    EXPECT_TRUE(std::ranges::is_sorted(all, [](const std::string& a, const std::string& b)
                                       { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                                                             [](char x, char y) { return static_cast<unsigned char>(x) < static_cast<unsigned char>(y); }); }));
    EXPECT_EQ(all.front(), "[+]above[g]");
    EXPECT_EQ(all.back(), "unstack");
    for (const char* name : {"[+]on[g]", "[+]on[g][sat]", "[-]on[hist]", "[+]number[hist]", "on", "number", "above", "stack"})
        EXPECT_TRUE(has(*schema, name)) << name;
    // no goal relations for the top types, no [unsat] and no subgoal layers by default
    for (const char* name : {"[+]number[g]", "[+]object[g]", "[+]on[g][unsat]", "[+]on[sg]"})
        EXPECT_FALSE(has(*schema, name)) << name;
    // nullary predicates are kept: their atoms are tuples over the graph's auxiliary entity
    for (const char* name : {"handempty", "[+]handempty[g]", "[-]handempty[g][sat]", "[+]handempty[hist]"})
        EXPECT_TRUE(has(*schema, name)) << name;
    EXPECT_EQ(roles(*schema, "handempty"), (std::vector<SlotRole>{R::Auxiliary}));
    EXPECT_EQ(roles(*schema, "[+]handempty[hist]"), (std::vector<SlotRole>{R::History, R::Auxiliary}));
    EXPECT_EQ(schema->relation(*schema->find("handempty")).arity, 1u);
    EXPECT_EQ(schema->relation(*schema->find("handempty")).logical_arity, 0u);
    EXPECT_EQ(schema->entity_role_names().size(), 9u);
    EXPECT_EQ(schema->entity_role_names().back(), k_auxiliary_entity_name);
    const Relation& goal_sat = schema->relation(*schema->find("[-]on[g][sat]"));
    EXPECT_EQ(goal_sat.lane, Lane::GoalDerivation);
    EXPECT_EQ(goal_sat.sign, -1);
    EXPECT_EQ(goal_sat.level, std::optional<u8>(0));
    EXPECT_EQ(goal_sat.derivation, std::optional<GoalDerivation>(GoalDerivation::Satisfied));
    EXPECT_EQ(goal_sat.predicate, 2);
    EXPECT_EQ(schema->relation(*schema->find("unstack")).schema, 3);
    EXPECT_EQ(schema->relation(*schema->find("above")).lane, Lane::Derived);
    EXPECT_EQ(schema->relation(*schema->find("number")).lane, Lane::Static);
}

TEST(EncodingSchema, SlotRolesFollowTheTargetsAndVirtualNodes)
{
    FlatOptions options;
    options.targets = {TargetSource::Goal, TargetSource::History, TargetSource::Action};
    options.predicate_virtual_nodes = true;
    options.max_goal_level = 1;
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Unsatisfied};
    const auto schema = EncodingSchema::create(blocks(), options);
    EXPECT_EQ(roles(*schema, "on"), (std::vector<SlotRole>{R::Predicate, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*schema, "[+]on[g]"), (std::vector<SlotRole>{R::GoalTarget, R::Predicate, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*schema, "[+]on[sg]"), (std::vector<SlotRole>{R::Predicate, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*schema, "[+]on[g][unsat]"), (std::vector<SlotRole>{R::Predicate, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*schema, "[-]on[hist]"),
              (std::vector<SlotRole>{R::HistoryTarget, R::History, R::Predicate, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*schema, "stack"), (std::vector<SlotRole>{R::Action, R::Argument, R::Argument}));
    EXPECT_EQ(schema->relation(*schema->find("[+]on[g]")).arity, 4u);
    EXPECT_EQ(schema->relation(*schema->find("[+]on[g]")).logical_arity, 2u);
    const auto offsets = schema->slot_role_offsets();
    ASSERT_EQ(offsets.size(), schema->num_relations() + 1);
    EXPECT_EQ(offsets.back(), schema->all_slot_roles().size());
}

TEST(EncodingSchema, NullaryStaticAndDerivedOptionsSelectRelations)
{
    FlatOptions options;
    options.nullary = false;
    {
        const auto schema = EncodingSchema::create(blocks(), options);
        for (const char* name : {"handempty", "[+]handempty[g]", "[+]handempty[hist]"})
            EXPECT_FALSE(has(*schema, name)) << name;
        EXPECT_EQ(schema->entity_role_names().size(), 9u);
        EXPECT_EQ(schema->slot_role_names().size(), 9u);
        EXPECT_EQ(std::ranges::count(schema->all_slot_roles(), SlotRole::Auxiliary), 0);
        EXPECT_NE(schema->fingerprint(), EncodingSchema::create(blocks())->fingerprint());
    }
    options = {};
    options.include_static = false;
    options.include_derived = false;
    const auto schema = EncodingSchema::create(blocks(), options);
    for (const char* name : {"number", "object", "above"})
        EXPECT_FALSE(has(*schema, name)) << name;
    EXPECT_TRUE(has(*schema, "[+]above[g]"));
    EXPECT_TRUE(has(*schema, "[+]number[hist]"));
    options = {};
    options.include_actions = false;
    EXPECT_FALSE(has(*EncodingSchema::create(blocks(), options), "stack"));
    EXPECT_TRUE(EncodingSchema::create(blocks(), options)->schema_relations().empty());
}

TEST(EncodingSchema, EqualityIsOptional)
{
    DomainSignature d = blocks();
    d.predicates.push_back({"=", 2, PredicateKind::Static});
    const auto without = EncodingSchema::create(d);
    EXPECT_FALSE(has(*without, "="));
    EXPECT_FALSE(has(*without, "[+]=[g]"));
    EXPECT_EQ(without->fingerprint(), EncodingSchema::create(blocks())->fingerprint());
    FlatOptions options;
    options.equality = true;
    const auto with = EncodingSchema::create(d, options);
    EXPECT_TRUE(has(*with, "="));
    EXPECT_TRUE(has(*with, "[+]=[g][sat]"));
    EXPECT_NE(with->fingerprint(), without->fingerprint());
}

TEST(EncodingSchema, SubgoalLevelsAndDerivationsName)
{
    FlatOptions options;
    options.max_goal_level = 3;
    options.goal_derivations = {GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    const auto schema = EncodingSchema::create(blocks(), options);
    EXPECT_TRUE(has(*schema, "[+]on[sssg][unsat]"));
    EXPECT_TRUE(has(*schema, "[-]on[ssg][sat]"));
    EXPECT_FALSE(has(*schema, "[+]on[g]"));
}

TEST(EncodingSchema, HorizonCatalogs)
{
    FlatOptions delta;
    delta.horizon.mode = HorizonMode::Delta;
    delta.horizon.root = RootPolicy::Exclude;
    delta.horizon.parent_relation = true;
    delta.goal_derivations = {GoalDerivation::AddedSatisfied, GoalDerivation::AddedUnsatisfied};
    const auto d = EncodingSchema::create(blocks(), delta);
    EXPECT_EQ(roles(*d, "on"), (std::vector<SlotRole>{R::Argument, R::Argument}));
    EXPECT_EQ(roles(*d, "[+]on"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*d, "[-]on[g][sat-]"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*d, "[+]on[sat+]"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*d, "_parent_"), (std::vector<SlotRole>{R::State, R::State}));
    EXPECT_EQ(roles(*d, "stack"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(d->relation(*d->find("_parent_")).lane, Lane::Parent);
    EXPECT_FALSE(has(*d, "[+]number"));  // static predicates never change

    FlatOptions full;
    full.horizon.mode = HorizonMode::Full;
    full.horizon.root = RootPolicy::Exclude;
    const auto f = EncodingSchema::create(blocks(), full);
    EXPECT_EQ(roles(*f, "on"), (std::vector<SlotRole>{R::Argument, R::Argument}));
    EXPECT_EQ(roles(*f, "on[state]"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*f, "[+]on[g][sat][state]"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    EXPECT_EQ(roles(*f, "[+]on[g]"), (std::vector<SlotRole>{R::Argument, R::Argument}));
    full.horizon.root = RootPolicy::EncodeOnly;
    const auto e = EncodingSchema::create(blocks(), full);
    EXPECT_FALSE(has(*e, "on[state]"));
    EXPECT_EQ(roles(*e, "on"), (std::vector<SlotRole>{R::State, R::Argument, R::Argument}));
    // Include and EncodeOnly differ in the targets only, which the fingerprint tells apart
    full.horizon.root = RootPolicy::Include;
    EXPECT_NE(EncodingSchema::create(blocks(), full)->fingerprint(), e->fingerprint());
    EXPECT_EQ(names(*EncodingSchema::create(blocks(), full)), names(*e));
}

TEST(EncodingSchema, ActionSchemasOfOneNameAndArityShareARelation)
{
    DomainSignature d = blocks();
    d.schemas.clear();
    for (const auto& [name, arity] : {std::pair{"stop", 1u}, std::pair{"go", 2u},
                                     std::pair{"stop", 1u}, std::pair{"stop", 1u}})
    {
        auto& action = d.schemas.emplace_back();
        action.name = name;
        action.arity = arity;
    }
    const auto schema = EncodingSchema::create(d);
    EXPECT_EQ(schema->relation(*schema->find("stop")).schema, 0);
    const auto map = schema->schema_relations();
    ASSERT_EQ(map.size(), 4u);
    EXPECT_EQ(map[0], map[2]);
    EXPECT_EQ(map[0], map[3]);
    EXPECT_NE(map[0], map[1]);
    // a name with several arities: one relation per arity
    d.schemas = {{"stop", 1}, {"go", 2}, {"stop", 3}, {"stop", 1}};
    const auto split = EncodingSchema::create(d);
    EXPECT_FALSE(has(*split, "stop"));
    ASSERT_TRUE(has(*split, "stop#1") && has(*split, "stop#3") && has(*split, "go"));
    EXPECT_EQ(split->relation(*split->find("stop#3")).arity, 4u);
    EXPECT_EQ(split->relation(*split->find("stop#3")).schema, 2);
    const auto split_map = split->schema_relations();
    EXPECT_EQ(split_map[0], split_map[3]);
    EXPECT_EQ(split_map[0], static_cast<i32>(*split->find("stop#1")));
    EXPECT_EQ(split_map[2], static_cast<i32>(*split->find("stop#3")));
}

TEST(EncodingSchema, FingerprintIgnoresDeclarationOrderAndTracksTheOutput)
{
    const DomainSignature d = blocks();
    const u64 reference = EncodingSchema::create(d)->fingerprint();
    std::mt19937 rng(7);
    for (int round = 0; round < 8; ++round)
    {
        DomainSignature shuffled = d;
        std::ranges::shuffle(shuffled.predicates, rng);
        std::ranges::shuffle(shuffled.schemas, rng);
        const auto schema = EncodingSchema::create(shuffled);
        EXPECT_EQ(schema->fingerprint(), reference);
        EXPECT_EQ(names(*schema), names(*EncodingSchema::create(d)));
    }
    EXPECT_EQ(EncodingSchema::create(d)->fingerprint(), reference);
    // pinned: the values are part of the contract (stable across processes and builds)
    EXPECT_EQ(reference, PIN_DEFAULT);
    FlatOptions pin_delta;
    pin_delta.horizon.mode = HorizonMode::Delta;
    pin_delta.goal_derivations = {GoalDerivation::AddedSatisfied, GoalDerivation::Satisfied};
    pin_delta.predicate_virtual_nodes = true;
    EXPECT_EQ(EncodingSchema::create(d, pin_delta)->fingerprint(), PIN_DELTA);
    FlatOptions pin_targets;
    pin_targets.max_goal_level = 2;
    pin_targets.targets = {TargetSource::Goal, TargetSource::Subgoal, TargetSource::History, TargetSource::Action};
    pin_targets.layout = Layout::Relation;
    pin_targets.index = IndexType::I64;
    EXPECT_EQ(EncodingSchema::create(d, pin_targets)->fingerprint(), PIN_TARGETS);
    FlatOptions pin_full;
    pin_full.horizon = {HorizonMode::Full, RootPolicy::Exclude, true};
    EXPECT_EQ(EncodingSchema::create(d, pin_full)->fingerprint(), PIN_FULL);

    DomainSignature renamed = d;
    renamed.predicates[2].name = "over";
    EXPECT_NE(EncodingSchema::create(renamed)->fingerprint(), reference);
    DomainSignature arity = d;
    arity.schemas[0].arity = 2;
    EXPECT_NE(EncodingSchema::create(arity)->fingerprint(), reference);
    DomainSignature kind = d;
    kind.predicates[2].kind = PredicateKind::Derived;
    EXPECT_NE(EncodingSchema::create(kind)->fingerprint(), reference);

    std::vector<FlatOptions> variants;
    auto vary = [&](auto change)
    {
        FlatOptions o;
        change(o);
        variants.push_back(o);
    };
    vary([](FlatOptions& o) { o.goal_derivations = {GoalDerivation::Plain}; });
    vary([](FlatOptions& o) { o.max_goal_level = 1; });
    vary([](FlatOptions& o) { o.include_static = false; });
    vary([](FlatOptions& o) { o.include_derived = false; });
    vary([](FlatOptions& o) { o.nullary = false; });
    vary([](FlatOptions& o) { o.predicate_virtual_nodes = true; });
    vary([](FlatOptions& o) { o.include_actions = false; });
    vary([](FlatOptions& o) { o.targets = {TargetSource::Action}; });
    vary([](FlatOptions& o) { o.targets = {TargetSource::Goal}; });
    vary([](FlatOptions& o) { o.layout = Layout::Relation; });
    vary([](FlatOptions& o) { o.index = IndexType::I64; });
    vary([](FlatOptions& o) { o.horizon.mode = HorizonMode::Full; });
    vary([](FlatOptions& o) { o.horizon.mode = HorizonMode::Action; });
    vary([](FlatOptions& o) { o.horizon = {HorizonMode::Full, RootPolicy::Exclude, false}; });
    vary([](FlatOptions& o) { o.horizon = {HorizonMode::Full, RootPolicy::EncodeOnly, false}; });
    vary([](FlatOptions& o) { o.horizon = {HorizonMode::Full, RootPolicy::Include, true}; });
    std::vector<u64> seen{reference};
    for (const FlatOptions& o : variants)
    {
        const u64 f = EncodingSchema::create(d, o)->fingerprint();
        EXPECT_EQ(std::ranges::count(seen, f), 0);
        seen.push_back(f);
    }
}

TEST(EncodingSchema, InvalidOptionsAndDomainsThrow)
{
    auto fails = [](auto change)
    {
        FlatOptions o;
        change(o);
        EXPECT_THROW((void)EncodingSchema::create(blocks(), o), std::invalid_argument);
    };
    fails([](FlatOptions& o) { o.max_goal_level = 4; });
    fails([](FlatOptions& o) { o.targets = {TargetSource::Subgoal}; });
    fails([](FlatOptions& o) { o.goal_derivations = {GoalDerivation::AddedSatisfied}; });
    fails([](FlatOptions& o)
          {
              o.horizon.mode = HorizonMode::Full;
              o.goal_derivations = {GoalDerivation::AddedUnsatisfied};
          });
    fails([](FlatOptions& o)
          {
              o.horizon.mode = HorizonMode::Full;
              o.targets = {TargetSource::Goal};
          });
    fails([](FlatOptions& o)
          {
              o.horizon.mode = HorizonMode::Action;
              o.include_actions = false;
          });
    fails([](FlatOptions& o) { o.horizon.parent_relation = true; });
    fails([](FlatOptions& o) { o.horizon.root = RootPolicy::Exclude; });
    fails([](FlatOptions& o) { o.horizon.root = RootPolicy::EncodeOnly; });
    fails([](FlatOptions& o)
          {
              o.targets = {TargetSource::Action};
              o.include_actions = false;
          });
    for (const GoalDerivations derivations : {GoalDerivations{GoalDerivation::Satisfied}, GoalDerivations{}})
    {
        fails([&](FlatOptions& o)
              {
                  o.goal_derivations = derivations;
                  o.targets = {TargetSource::Goal};
              });
        fails([&](FlatOptions& o)
              {
                  o.goal_derivations = derivations;
                  o.max_goal_level = 1;
                  o.targets = {TargetSource::Subgoal};
              });
    }
    DomainSignature twice = blocks();
    twice.predicates.push_back({"on", 2, PredicateKind::Fluent});
    EXPECT_THROW((void)EncodingSchema::create(twice), std::invalid_argument);
    DomainSignature collision = blocks();
    collision.schemas.push_back({"on", 1});  // an action named like a predicate
    EXPECT_THROW((void)EncodingSchema::create(collision), std::invalid_argument);
    FlatOptions delta;
    delta.horizon.mode = HorizonMode::Delta;
    delta.goal_derivations = {GoalDerivation::AddedSatisfied};
    EXPECT_NO_THROW((void)EncodingSchema::create(blocks(), delta));
}

TEST(EncodingOptions, NormalizationAndText)
{
    FlatOptions o;
    o.horizon.root = RootPolicy::Exclude;
    EXPECT_THROW((void)normalized(o), std::invalid_argument);
    o.horizon.mode = HorizonMode::Full;
    EXPECT_EQ(normalized(o).horizon.root, RootPolicy::Exclude);
    EXPECT_EQ(normalized(FlatOptions{}).horizon.root, RootPolicy::Include);
    for (const auto text : {"plain", "satisfied", "unsatisfied", "added_satisfied", "added_unsatisfied"})
        EXPECT_EQ(to_string(*parse_goal_derivation(text)), text);
    for (const auto text : {"goal", "subgoal", "action", "history"})
        EXPECT_EQ(to_string(*parse_target_source(text)), text);
    for (const auto text : {"none", "full", "delta", "action"})
        EXPECT_EQ(to_string(*parse_horizon_mode(text)), text);
    for (const auto text : {"include", "exclude", "encode_only"})
        EXPECT_EQ(to_string(*parse_root_policy(text)), text);
    EXPECT_EQ(to_string(*parse_layout("relation")), "relation");
    EXPECT_EQ(to_string(*parse_index_type("int64")), "int64");
    EXPECT_FALSE(parse_layout("row"));
    EXPECT_FALSE(parse_goal_derivation(""));
}

TEST(EncodingSchema, DomainOfATableKeepsHelperPredicatesAndFullActionArities)
{
    const auto task = Task::from_text_file(test::task_path("philosophers__p03-phil4"), {});
    const auto table = rl::TaskTable::single(task);
    const DomainSignature domain = DomainSignature::of(*table);
    EXPECT_EQ(domain.predicates.size(), table->num_domain_predicates());
    const auto it = std::ranges::find(domain.predicates, std::string("axiom_1"), &DomainSignature::Predicate::name);
    ASSERT_NE(it, domain.predicates.end());
    EXPECT_EQ(it->kind, PredicateKind::Derived);
    EXPECT_EQ(it->arity, 3u);
    const auto schema = EncodingSchema::create(*table);
    EXPECT_TRUE(has(*schema, "axiom_0"));
    EXPECT_TRUE(has(*schema, "[+]axiom_1[g]"));
    EXPECT_EQ(schema->fingerprint(), EncodingSchema::create(domain)->fingerprint());
    ASSERT_EQ(domain.schemas.size(), table->num_schemas());
    for (u32 s = 0; s < table->num_schemas(); ++s)
        EXPECT_EQ(domain.schemas[s].arity, table->schema_arities()[s]);
}
}

TEST(EncodingSchema, TheStoredArraysAreOptionsOfTheFingerprint)
{
    const DomainSignature d = blocks();
    FlatOptions options;
    options.include_actions = false;
    const u64 reference = EncodingSchema::create(d, options)->fingerprint();
    EXPECT_TRUE(EncodingSchema::create(d, options)->emits_counts());
    EXPECT_FALSE(EncodingSchema::create(d, options)->implicit_entity_arrays());
    FlatOptions no_counts = options;
    no_counts.emit_counts = false;
    EXPECT_FALSE(EncodingSchema::create(d, no_counts)->emits_counts());
    EXPECT_NE(EncodingSchema::create(d, no_counts)->fingerprint(), reference);
    FlatOptions automatic = options;
    automatic.entity_arrays = EntityArrays::Auto;
    EXPECT_TRUE(EncodingSchema::create(d, automatic)->implicit_entity_arrays());
    EXPECT_NE(EncodingSchema::create(d, automatic)->fingerprint(), reference);
    // Auto that cannot be implicit is the same encoding as Always: same arrays, same fingerprint
    FlatOptions with_actions;
    FlatOptions with_actions_auto;
    with_actions_auto.entity_arrays = EntityArrays::Auto;
    EXPECT_FALSE(EncodingSchema::create(d, with_actions_auto)->implicit_entity_arrays());
    EXPECT_EQ(EncodingSchema::create(d, with_actions_auto)->fingerprint(),
              EncodingSchema::create(d, with_actions)->fingerprint());
    EXPECT_EQ(parse_entity_arrays("auto"), EntityArrays::Auto);
    EXPECT_EQ(parse_entity_arrays("always"), EntityArrays::Always);
    EXPECT_FALSE(parse_entity_arrays("never"));
    EXPECT_EQ(to_string(EntityArrays::Auto), "auto");
}
