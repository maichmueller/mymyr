#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <functional>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
struct Variant
{
    IndexType index = IndexType::I32;
    Layout layout = Layout::Graph;
    bool counts = true;
    bool implicit_entities = false;
};

std::shared_ptr<const EncodingSchema> schema(Variant variant = {})
{
    DomainSignature domain;
    domain.predicates = {{"on", 2, PredicateKind::Fluent}, {"handempty", 0, PredicateKind::Fluent}};
    FlatOptions options;
    options.goal_derivations = {};
    options.include_actions = false;
    options.index = variant.index;
    options.layout = variant.layout;
    options.emit_counts = variant.counts;
    options.entity_arrays = variant.implicit_entities ? EntityArrays::Auto : EntityArrays::Always;
    return EncodingSchema::create(domain, options);
}

/// Two graphs over two objects each. Graph 0 has the auxiliary entity, a `handempty` tuple and two `on` tuples, graph 1
/// one `on` tuple. The tuples are batch-global entity ids per graph and relation; build() derives the offsets.
struct Batch
{
    u32 handempty, on, R;
    std::vector<i32> task_ids{0, 0}, num_objects{2, 2}, entity_offsets{0, 3, 5}, entity_ref{0, 1, -1, 0, 1};
    std::vector<u8> entity_role{0, 0, 8, 0, 0};
    std::vector<std::vector<std::vector<std::vector<i32>>>> tuples;  // [graph][relation][tuple]
    std::vector<i32> counts, offsets, arg_offsets, args;
    i64 fingerprint_xor = 0;       // flips bits of the stored schema_fingerprint
    bool wide_num_objects = false;  // num_objects as int64 instead of int32
    bool flat_counts = false;       // relation_counts with shape [G * R] instead of [G, R]
    Variant variant;                // the schema the batch is built for

    Batch()
    {
        const auto s = schema();
        handempty = *s->find("handempty");
        on = *s->find("on");
        R = s->num_relations();
        tuples.assign(2, std::vector<std::vector<std::vector<i32>>>(R));
        tuples[0][handempty] = {{2}};
        tuples[0][on] = {{0, 1}, {1, 0}};
        tuples[1][on] = {{3, 4}};
    }

    void assemble()
    {
        counts.clear();
        offsets = {0};
        arg_offsets = {0};
        args.clear();
        auto add = [&](const std::vector<std::vector<i32>>& cell)
        {
            for (const auto& tuple : cell)
                args.insert(args.end(), tuple.begin(), tuple.end());
            offsets.push_back(offsets.back() + static_cast<i32>(cell.size()));
            arg_offsets.push_back(static_cast<i32>(args.size()));
        };
        for (const auto& graph : tuples)
            for (const auto& cell : graph)
                counts.push_back(static_cast<i32>(cell.size()));
        if (variant.layout == Layout::Graph)
            for (const auto& graph : tuples)
                for (const auto& cell : graph)
                    add(cell);
        else
            for (u32 r = 0; r < R; ++r)
                for (const auto& graph : tuples)
                    add(graph[r]);
    }

    [[nodiscard]] rl::ArrayBundle bundle() const
    {
        rl::ArrayBundle::Builder b;
        auto size = [](const auto& v) { return static_cast<i64>(v.size()); };
        b.add("task_ids", rl::DType::I32, {2}, task_ids);
        if (wide_num_objects)
            b.add("num_objects", rl::DType::I64, {2}, std::vector<i64>(num_objects.begin(), num_objects.end()));
        else
            b.add("num_objects", rl::DType::I32, {2}, num_objects);
        b.add("entity_offsets", rl::DType::I32, {3}, entity_offsets);
        if (!variant.implicit_entities)
        {
            b.add("entity_role", rl::DType::U8, {size(entity_role)}, entity_role);
            b.add("entity_ref", rl::DType::I32, {size(entity_ref)}, entity_ref);
        }
        if (variant.counts)
            b.add("relation_counts", rl::DType::I32,
                  flat_counts ? std::vector<i64>{2 * static_cast<i64>(R)} : std::vector<i64>{2, static_cast<i64>(R)}, counts);
        b.add("relation_offsets", rl::DType::I32, {size(offsets)}, offsets);
        b.add("relation_arg_offsets", rl::DType::I32, {size(arg_offsets)}, arg_offsets);
        b.add("relation_args", rl::DType::I32, {size(args)}, args);
        b.scalar("version", 1);
        b.scalar("num_graphs", 2);
        b.scalar("num_relations", R);
        b.scalar("layout", variant.layout == Layout::Relation ? 1 : 0);
        b.scalar("index_dtype", 0);
        b.scalar("schema_fingerprint", std::bit_cast<i64>(schema(variant)->fingerprint()) ^ fingerprint_xor);
        b.scalar("statics", 0);
        return b.build();
    }
};

/// `by_tuples` changes the tuples before the offsets are derived, `by_arrays` the arrays after.
void expect_rejected(const char* what, const std::function<void(Batch&)>& by_tuples,
                     const std::function<void(Batch&)>& by_arrays = {}, Variant variant = {})
{
    Batch batch;
    batch.variant = variant;
    by_tuples(batch);
    batch.assemble();
    if (by_arrays)
        by_arrays(batch);
    EXPECT_THROW(validate(batch.bundle(), *schema(variant)), std::invalid_argument) << what;
}

TEST(EncodingValidate, AValidBatchPasses)
{
    Batch batch;
    batch.assemble();
    EXPECT_NO_THROW(validate(batch.bundle(), *schema()));
}

TEST(EncodingValidate, CorruptionsOfTheTuplesAreRejected)
{
    expect_rejected("an argument outside the entities", [](Batch& b) { b.tuples[1][b.on] = {{3, 9}}; });
    expect_rejected("an argument of another graph", [](Batch& b) { b.tuples[1][b.on] = {{0, 1}}; });
    expect_rejected("a duplicate tuple", [](Batch& b) { b.tuples[0][b.on] = {{0, 1}, {0, 1}}; });
    expect_rejected("an unsorted cell", [](Batch& b) { b.tuples[0][b.on] = {{1, 0}, {0, 1}}; });
    expect_rejected("an object at an auxiliary slot", [](Batch& b) { b.tuples[0][b.handempty] = {{0}}; });
    expect_rejected("the auxiliary entity at an argument slot", [](Batch& b) { b.tuples[0][b.on] = {{0, 2}}; });
    expect_rejected("an auxiliary entity that no tuple uses", [](Batch& b) { b.tuples[0][b.handempty].clear(); });
    expect_rejected("an auxiliary tuple in a graph without the entity", [](Batch& b) { b.tuples[1][b.handempty] = {{4}}; });
}

TEST(EncodingValidate, CorruptionsOfTheArraysAreRejected)
{
    const auto none = [](Batch&) {};
    expect_rejected("a missing auxiliary entity", none, [](Batch& b) { b.entity_role[2] = 0; });
    expect_rejected("an auxiliary entity with a reference", none, [](Batch& b) { b.entity_ref[2] = 0; });
    expect_rejected("an object with the wrong reference", none, [](Batch& b) { b.entity_ref[0] = 1; });
    expect_rejected("non-monotone tuple offsets", none, [](Batch& b) { b.offsets[1] = b.offsets.back() + 1; });
    expect_rejected("non-monotone argument offsets", none, [](Batch& b) { b.arg_offsets[2] = b.arg_offsets[1] - 1; });
    expect_rejected("offsets that do not start at 0", none, [](Batch& b) { b.offsets[0] = 1; });
    expect_rejected("an entity offset array that does not start at 0", none, [](Batch& b) { b.entity_offsets[0] = 1; });
    expect_rejected("counts that disagree with the offsets", none, [](Batch& b) { b.counts[b.on] += 1; });
    expect_rejected("argument offsets that disagree with the counts", none,
                    [](Batch& b) { b.arg_offsets[b.on + 1] += 1; b.args.push_back(0); });
    expect_rejected("a second auxiliary entity", none, [](Batch& b) { b.entity_role[4] = 8; b.entity_ref[4] = -1; });
}

TEST(EncodingValidate, ScalarsDtypesAndShapesAreChecked)
{
    Batch batch;
    batch.assemble();
    EXPECT_THROW(validate(batch.bundle(), *schema({IndexType::I64})), std::invalid_argument);
    batch.wide_num_objects = true;
    EXPECT_THROW(validate(batch.bundle(), *schema()), std::invalid_argument);
    batch.wide_num_objects = false;
    batch.flat_counts = true;
    EXPECT_THROW(validate(batch.bundle(), *schema()), std::invalid_argument);
    batch.flat_counts = false;
    EXPECT_NO_THROW(validate(batch.bundle(), *schema()));
    batch.fingerprint_xor = 1;
    EXPECT_THROW(validate(batch.bundle(), *schema()), std::invalid_argument);
    batch.fingerprint_xor = 0;
    rl::ArrayBundle::Builder wrong_version;
    wrong_version.scalar("version", 2);
    EXPECT_THROW(validate(wrong_version.build(), *schema()), std::invalid_argument);
    rl::ArrayBundle::Builder empty;
    empty.scalar("version", 1);
    empty.scalar("num_graphs", 0);
    empty.scalar("num_relations", schema()->num_relations());
    EXPECT_THROW(validate(empty.build(), *schema()), std::invalid_argument);
}

std::vector<Variant> all_variants()
{
    std::vector<Variant> out;
    for (const Layout layout : {Layout::Graph, Layout::Relation})
        for (const bool counts : {true, false})
            for (const bool implicit : {false, true})
                out.push_back({IndexType::I32, layout, counts, implicit});
    return out;
}

TEST(EncodingValidate, EveryLayoutAndOmissionPasses)
{
    for (const Variant& variant : all_variants())
    {
        Batch batch;
        batch.variant = variant;
        batch.assemble();
        EXPECT_NO_THROW(validate(batch.bundle(), *schema(variant)));
        EXPECT_EQ(schema(variant)->implicit_entity_arrays(), variant.implicit_entities);
    }
}

TEST(EncodingValidate, TheLayoutOfTheBatchMustBeTheSchemas)
{
    Batch relation;
    relation.variant.layout = Layout::Relation;
    relation.assemble();
    EXPECT_THROW(validate(relation.bundle(), *schema()), std::invalid_argument);
    Batch graph;
    graph.assemble();
    EXPECT_THROW(validate(graph.bundle(), *schema({IndexType::I32, Layout::Relation})), std::invalid_argument);
    // a graph-major batch under the relation-major scalar
    Batch lying;
    lying.assemble();
    lying.variant.layout = Layout::Relation;
    EXPECT_THROW(validate(lying.bundle(), *schema({IndexType::I32, Layout::Relation})), std::invalid_argument);
}

TEST(EncodingValidate, RelationMajorCorruptionsAreRejected)
{
    const Variant relation{IndexType::I32, Layout::Relation};
    const auto none = [](Batch&) {};
    expect_rejected("an argument of another graph", [](Batch& b) { b.tuples[1][b.on] = {{0, 1}}; }, none, relation);
    expect_rejected("an unsorted cell", [](Batch& b) { b.tuples[0][b.on] = {{1, 0}, {0, 1}}; }, none, relation);
    expect_rejected("counts that disagree with the offsets", none, [](Batch& b) { b.counts[b.on] += 1; }, relation);
    expect_rejected("non-monotone offsets", none, [](Batch& b) { b.offsets[2] = b.offsets.back() + 1; }, relation);
    expect_rejected("graph 0's tuples stored at graph 1's segment",
                    [](Batch& b) { std::swap(b.tuples[0][b.on], b.tuples[1][b.on]); }, none, relation);
}

TEST(EncodingValidate, OmittedArraysMustBeAbsentAndStoredOnesPresent)
{
    const Variant no_counts{IndexType::I32, Layout::Graph, false, false};
    const Variant implicit{IndexType::I32, Layout::Graph, true, true};
    Batch full;
    full.assemble();
    EXPECT_THROW(validate(full.bundle(), *schema(no_counts)), std::invalid_argument);
    EXPECT_THROW(validate(full.bundle(), *schema(implicit)), std::invalid_argument);
    Batch omitted;
    omitted.variant = no_counts;
    omitted.assemble();
    EXPECT_THROW(validate(omitted.bundle(), *schema()), std::invalid_argument);
    Batch without_entities;
    without_entities.variant = implicit;
    without_entities.assemble();
    EXPECT_THROW(validate(without_entities.bundle(), *schema()), std::invalid_argument);
}

TEST(EncodingValidate, ImplicitEntitiesAreCheckedFromTheTuples)
{
    const Variant implicit{IndexType::I32, Layout::Graph, true, true};
    const auto none = [](Batch&) {};
    expect_rejected("an object at an auxiliary slot", [](Batch& b) { b.tuples[0][b.handempty] = {{0}}; }, none, implicit);
    expect_rejected("the auxiliary entity at an argument slot", [](Batch& b) { b.tuples[0][b.on] = {{0, 2}}; }, none, implicit);
    expect_rejected("an auxiliary entity that no tuple uses", [](Batch& b) { b.tuples[0][b.handempty].clear(); }, none, implicit);
    expect_rejected("two entities past the objects", none,
                    [](Batch& b) { b.entity_offsets = {0, 4, 6}; b.entity_ref.assign(6, 0); b.entity_role.assign(6, 0); },
                    implicit);
}
}  // namespace
