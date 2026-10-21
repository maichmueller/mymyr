// The numeric lane of the relational encoder: one relation `[num]f` per fluent function, one tuple per numeric slot of
// the instance, and the value of each tuple in numeric_values (and, with the Simple graph, in node_value). Every batch
// is compared with the tuples and values the definition gives (the slots of the task and rl::numeric_values), for both
// layouts, index types, value dtypes and storages, mixed instances, nullary functions, and every form of the encoder.

#include "support.hpp"

#include "mymyr/encoding/detail/lanes.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <string>

using namespace mymyr;
using namespace mymyr::encoding;
namespace ed = mymyr::encoding::detail;

namespace
{
using StorageMode = TaskOptions::NumericStorageMode;

TaskPtr counters(const std::string& problem, StorageMode mode)
{
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
    TaskOptions options;
    options.numeric_storage = mode;
    return Task::create(*frontend::load_task(dir + "domain.pddl", dir + problem + ".pddl"), options);
}

TaskPtr text_task(const std::string& name)
{
    return Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/" + name + ".txt");
}

struct Table
{
    std::string name;
    rl::TaskTablePtr table;
};

/// Tables with i32 and f64 storage, instances of different sizes, nullary functions, and functions of several arities.
std::vector<Table> tables()
{
    std::vector<Table> out;
    out.push_back({"counters_i32", rl::TaskTable::create({counters("p01", StorageMode::Auto),
                                                           counters("p02", StorageMode::Auto)})});
    out.push_back({"counters_f64", rl::TaskTable::create({counters("p01", StorageMode::F64),
                                                           counters("p03-half", StorageMode::F64),
                                                           counters("p02", StorageMode::F64)})});
    out.push_back({"drone", rl::TaskTable::single(text_task("cs-drone"))});
    out.push_back({"refuel", rl::TaskTable::single(text_task("m-refuel"))});
    out.push_back({"expedition", rl::TaskTable::single(text_task("cs-expedition"))});
    return out;
}

/// `per_instance` rows of each instance, interleaved.
test::Rows mixed_rows(const rl::TaskTable& table, u64 per_instance, u64 seed)
{
    std::vector<test::Rows> sources;
    for (u32 i = 0; i < table.size(); ++i)
        sources.push_back(test::random_walk(table, i, per_instance, seed + i));
    test::Rows mixed;
    mixed.words = table.words();
    mixed.numeric_words = table.numeric_words();
    const u32 width = table.row_words();
    for (u64 k = 0; k < per_instance; ++k)
        for (const test::Rows& source : sources)
        {
            mixed.data.insert(mixed.data.end(), source.data.begin() + static_cast<std::ptrdiff_t>(k * width),
                              source.data.begin() + static_cast<std::ptrdiff_t>((k + 1) * width));
            mixed.task_ids.push_back(source.task_ids[k]);
        }
    return mixed;
}

EncodeInput input_of(const test::Rows& rows)
{
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    return in;
}

/// One row of `rows`.
test::Rows row_of(const test::Rows& rows, u64 g, u32 width)
{
    test::Rows one;
    one.words = rows.words;
    one.numeric_words = rows.numeric_words;
    one.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(g * width),
                    rows.data.begin() + static_cast<std::ptrdiff_t>((g + 1) * width));
    one.task_ids = {rows.task_ids[g]};
    return one;
}

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

/// The entries of a float array as doubles.
std::vector<f64> reals(const rl::ArrayBundle& bundle, const char* name, rl::DType want)
{
    const rl::ArrayInfo* info = bundle.find(name);
    std::vector<f64> out;
    if (!info || info->dtype != want)
    {
        ADD_FAILURE() << "array " << name << " is missing or has another dtype";
        return out;
    }
    for (u64 i = 0; i < info->elements(); ++i)
    {
        const std::byte* at = static_cast<const std::byte*>(bundle.data(*info)) + i * rl::dtype_bytes(info->dtype);
        if (want == rl::DType::F32)
        {
            f32 v;
            std::memcpy(&v, at, 4);
            out.push_back(v);
        }
        else
        {
            f64 v;
            std::memcpy(&v, at, 8);
            out.push_back(v);
        }
    }
    return out;
}

struct Tuple
{
    u32 relation = 0;
    std::vector<i64> args;  // graph-local entity ids
    f64 value = 0;
};
using Graphs = std::vector<std::vector<Tuple>>;

bool same_value(f64 a, f64 b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

void expect_equal(const Graphs& got, const Graphs& want)
{
    ASSERT_EQ(got.size(), want.size());
    for (u64 g = 0; g < got.size(); ++g)
    {
        ASSERT_EQ(got[g].size(), want[g].size()) << "graph " << g;
        for (u64 t = 0; t < got[g].size(); ++t)
        {
            EXPECT_EQ(got[g][t].relation, want[g][t].relation) << "graph " << g << " tuple " << t;
            EXPECT_EQ(got[g][t].args, want[g][t].args) << "graph " << g << " tuple " << t;
            EXPECT_TRUE(same_value(got[g][t].value, want[g][t].value))
                << "graph " << g << " tuple " << t << ": " << got[g][t].value << " vs " << want[g][t].value;
        }
    }
}

rl::DType value_dtype(const FlatOptions& options)
{
    return options.numeric_dtype == NumericDtype::F64 ? rl::DType::F64 : rl::DType::F32;
}

/// The numeric tuples of every graph of an encoded batch, read from the relation arrays and numeric_values by the
/// layout: the tuples of the numeric segments in segment order own the entries of numeric_values in order.
Graphs decode(const rl::ArrayBundle& batch, const EncodingSchema& schema)
{
    const auto entity_offsets = ints(batch, "entity_offsets");
    const auto offsets = ints(batch, "relation_offsets");
    const auto arg_offsets = ints(batch, "relation_arg_offsets");
    const auto args = ints(batch, "relation_args");
    const auto values = reals(batch, "numeric_values", value_dtype(schema.options()));
    const u64 G = entity_offsets.size() - 1;
    const u32 R = static_cast<u32>(schema.num_relations());
    const bool relation_major = schema.options().layout == Layout::Relation;
    Graphs graphs(G);
    u64 next = 0;
    auto segment = [&](u64 g, u32 r)
    {
        const u64 s = relation_major ? r * G + g : g * R + r;
        const u32 arity = schema.relation(r).arity;
        for (i64 t = offsets[s]; t < offsets[s + 1]; ++t)
        {
            Tuple tuple;
            tuple.relation = r;
            for (u32 k = 0; k < arity; ++k)
                tuple.args.push_back(args[static_cast<u64>(arg_offsets[s]) + static_cast<u64>(t - offsets[s]) * arity + k] -
                                     entity_offsets[g]);
            EXPECT_LT(next, values.size());
            tuple.value = next < values.size() ? values[next] : 0;
            ++next;
            graphs[g].push_back(std::move(tuple));
        }
    };
    if (relation_major)
    {
        for (u32 r = schema.first_numeric_relation(); r < R; ++r)
            for (u64 g = 0; g < G; ++g)
                segment(g, r);
    }
    else
        for (u64 g = 0; g < G; ++g)
            for (u32 r = schema.first_numeric_relation(); r < R; ++r)
                segment(g, r);
    EXPECT_EQ(next, values.size());
    return graphs;
}

/// The numeric tuples the definition gives: per graph, per numeric relation, the instance's slots of the function in
/// slot order, over the objects of the slot's term (the auxiliary entity for a nullary function), with the value of the
/// row's slot rounded to the dtype.
Graphs expected(const rl::TaskTable& table, const EncodingSchema& schema, const test::Rows& rows,
                const rl::ArrayBundle& batch)
{
    const u32 slots = rl::max_numeric_slots(*rl::TaskSuite::of(table));
    std::vector<f64> values(rows.size() * slots);
    rl::numeric_values(*rl::TaskSuite::of(table), rows.view(), rows.task_ids.data(), values.data(), slots);
    const auto num_objects = ints(batch, "num_objects");
    Graphs graphs(rows.size());
    const bool wide = schema.options().numeric_dtype == NumericDtype::F64;
    // the tuples of a graph in the order of the decoder: relation by relation
    for (u64 g = 0; g < rows.size(); ++g)
    {
        const plan::Numeric& numeric = table.task(static_cast<u32>(rows.task_ids[g]))->numeric();
        for (u32 r = schema.first_numeric_relation(); r < schema.num_relations(); ++r)
        {
            const Relation& relation = schema.relation(r);
            EXPECT_EQ(relation.lane, Lane::Numeric);
            for (u32 slot = 0; slot < numeric.slots; ++slot)
            {
                if (numeric.slot_function[slot] != static_cast<u32>(relation.function))
                    continue;
                Tuple tuple;
                tuple.relation = r;
                const u32 begin = numeric.slot_args_begin[slot], end = numeric.slot_args_begin[slot + 1];
                for (u32 k = begin; k < end; ++k)
                    tuple.args.push_back(numeric.slot_args[k].v);
                if (begin == end)
                    tuple.args.push_back(num_objects[g]);
                const f64 value = values[g * slots + slot];
                tuple.value = wide ? value : static_cast<f64>(static_cast<f32>(value));
                graphs[g].push_back(std::move(tuple));
            }
        }
    }
    return graphs;
}

FlatOptions options_with(Layout layout, IndexType index, NumericDtype dtype)
{
    FlatOptions options = test::encoder_options();
    options.layout = layout;
    options.index = index;
    options.numeric_dtype = dtype;
    return options;
}

constexpr Layout k_layouts[] = {Layout::Graph, Layout::Relation};
constexpr IndexType k_indices[] = {IndexType::I32, IndexType::I64};
constexpr NumericDtype k_dtypes[] = {NumericDtype::F32, NumericDtype::F64};

void expect_same_bytes(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    ASSERT_EQ(a.bytes(), b.bytes());
    EXPECT_EQ(std::memcmp(a.block(), b.block(), a.bytes()), 0);
    EXPECT_EQ(a.scalars(), b.scalars());
}
}  // namespace

TEST(EncoderNumeric, TheCatalogHasOneRelationPerFluentFunctionAfterAllOthers)
{
    const auto table = rl::TaskTable::create({counters("p01", StorageMode::Auto), counters("p02", StorageMode::Auto)});
    const auto schema = EncodingSchema::create(*table, test::encoder_options());
    ASSERT_TRUE(schema->has_numeric());
    ASSERT_EQ(schema->num_relations() - schema->first_numeric_relation(), 1u);
    const Relation& value = schema->relation(schema->first_numeric_relation());
    EXPECT_EQ(value.name, "[num]value");
    EXPECT_EQ(value.lane, Lane::Numeric);
    EXPECT_EQ(value.arity, 1u);
    EXPECT_EQ(schema->find("[num]value"), schema->first_numeric_relation());
    EXPECT_TRUE(schema->stores_numeric_values());
    EXPECT_FALSE(schema->stores_node_value());
    // total-cost is not part of a state, a static function (step, limit) is not a relation
    for (u32 r = 0; r < schema->first_numeric_relation(); ++r)
        EXPECT_NE(schema->relation(r).lane, Lane::Numeric);

    // a nullary function is a tuple over the auxiliary entity; the relations are sorted by name
    const auto drone = rl::TaskTable::single(text_task("cs-drone"));
    const auto drone_schema = EncodingSchema::create(*drone, test::encoder_options());
    std::vector<std::string> names;
    for (u32 r = drone_schema->first_numeric_relation(); r < drone_schema->num_relations(); ++r)
    {
        names.push_back(drone_schema->relation(r).name);
        EXPECT_EQ(drone_schema->relation(r).arity, std::max(1u, drone_schema->relation(r).logical_arity));
    }
    EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
    EXPECT_GE(names.size(), 4u);

    // the schema of a table without numeric functions has none, and the option leaves them out
    const auto classical = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    EXPECT_FALSE(EncodingSchema::create(*classical, test::encoder_options())->has_numeric());
    FlatOptions without = test::encoder_options();
    without.include_numeric = false;
    const auto left_out = EncodingSchema::create(*table, without);
    EXPECT_FALSE(left_out->has_numeric());
    EXPECT_EQ(left_out->num_relations() + 1, schema->num_relations());
    EXPECT_NE(left_out->fingerprint(), schema->fingerprint());
    // the dtype is part of the schema only where values are stored
    FlatOptions wide = test::encoder_options();
    wide.numeric_dtype = NumericDtype::F64;
    EXPECT_NE(EncodingSchema::create(*table, wide)->fingerprint(), schema->fingerprint());
    wide.include_numeric = without.include_numeric = false;
    EXPECT_EQ(EncodingSchema::create(*classical, wide)->fingerprint(),
              EncodingSchema::create(*classical, test::encoder_options())->fingerprint());
}

TEST(EncoderNumeric, TheTablesUnderTestCoverBothStoragesAndSeveralInstanceSizes)
{
    const auto all = tables();
    EXPECT_EQ(all[0].table->numeric_storage(), NumericStorage::I32);
    EXPECT_EQ(all[1].table->numeric_storage(), NumericStorage::F64);
    EXPECT_NE(all[0].table->task(0)->numeric().slots, all[0].table->task(1)->numeric().slots);
}

TEST(EncoderNumeric, TuplesAndValuesAreThoseOfTheRowsInEveryConfiguration)
{
    for (const Table& t : tables())
    {
        const test::Rows rows = mixed_rows(*t.table, 30, 7);
        for (const Layout layout : k_layouts)
            for (const IndexType index : k_indices)
                for (const NumericDtype dtype : k_dtypes)
                {
                    SCOPED_TRACE(t.name + " layout=" + std::string(to_string(layout)) +
                                 (index == IndexType::I64 ? " int64 " : " int32 ") +
                                 std::string(to_string(dtype)));
                    const auto encoder = FlatEncoder::create(t.table, options_with(layout, index, dtype));
                    const auto batch = encoder->encode(input_of(rows));
                    EXPECT_EQ(batch->scalar("numeric_dtype"), dtype == NumericDtype::F64 ? 1 : 0);
                    EXPECT_NO_THROW(validate(*batch, encoder->schema()));
                    const Graphs got = decode(*batch, encoder->schema());
                    ASSERT_FALSE(got.empty());
                    EXPECT_FALSE(got[0].empty());
                    expect_equal(got, expected(*t.table, encoder->schema(), rows, *batch));
                }
    }
}

TEST(EncoderNumeric, ValuesFollowTheStateAlongAWalk)
{
    // the values change from row to row and instances differ in their slots: not every row has the same values
    const auto table = tables()[1].table;
    const test::Rows rows = mixed_rows(*table, 40, 3);
    const auto encoder = FlatEncoder::create(table, options_with(Layout::Graph, IndexType::I32, NumericDtype::F64));
    const auto graphs = decode(*encoder->encode(input_of(rows)), encoder->schema());
    std::map<std::vector<f64>, int> distinct;
    for (const auto& graph : graphs)
    {
        std::vector<f64> v;
        for (const Tuple& t : graph)
            v.push_back(t.value);
        ++distinct[v];
    }
    EXPECT_GT(distinct.size(), 5u);
    // instance 1 has a half value in the initial state
    bool half = false;
    for (const Tuple& t : graphs[1])
        half = half || t.value == 0.5;
    EXPECT_TRUE(half);
}

TEST(EncoderNumeric, AFunctionTermWithoutAValueEncodesNaN)
{
    // u (nullary) and w(b) have no value until an action assigns one
    const auto domain = frontend::Domain::from_string(
        "(define (domain undef) (:requirements :strips :typing :numeric-fluents) (:types o)"
        " (:predicates (p)) (:functions (u) (w ?x - o))"
        " (:action set-u :parameters () :precondition (and) :effect (assign (u) 1))"
        " (:action set-w :parameters (?x - o) :precondition (and) :effect (assign (w ?x) 4))"
        " (:action raise :parameters (?x - o) :precondition (and) :effect (increase (w ?x) 1)))");
    const auto task = Task::create(*domain->instantiate_string(
        "(define (problem undefined) (:domain undef) (:objects a b - o) (:init (= (w a) 2)) (:goal (and (p))))"));
    const auto table = rl::TaskTable::single(task);
    const test::Rows rows = test::random_walk(*table, 0, 12, 5);
    for (const NumericDtype dtype : k_dtypes)
        for (const Layout layout : k_layouts)
        {
            const auto encoder = FlatEncoder::create(table, options_with(layout, IndexType::I32, dtype));
            const auto batch = encoder->encode(input_of(rows));
            const auto graphs = decode(*batch, encoder->schema());
            expect_equal(graphs, expected(*table, encoder->schema(), rows, *batch));
            u64 undefined = 0, defined = 0;
            for (const auto& graph : graphs)
                for (const Tuple& t : graph)
                    (std::isnan(t.value) ? undefined : defined) += 1;
            EXPECT_GT(undefined, 0u);
            EXPECT_GT(defined, 0u);
            // the initial state: u and w(b) are undefined, w(a) is 2
            u32 nan_in_first = 0;
            for (const Tuple& t : graphs[0])
                nan_in_first += std::isnan(t.value) ? 1 : 0;
            EXPECT_EQ(nan_in_first, 2u);
        }
}

TEST(EncoderNumeric, SingleBatchPoolAndStreamsAgree)
{
    for (const Table& t : tables())
    {
        // more rows than a member takes, so that the pool splits the batch
        const test::Rows rows = mixed_rows(*t.table, 2200 / t.table->size(), 11);
        const u32 width = t.table->row_words();
        for (const Layout layout : k_layouts)
            for (const NumericDtype dtype : k_dtypes)
            {
                SCOPED_TRACE(t.name + " layout=" + std::string(to_string(layout)) + std::string(to_string(dtype)));
                const auto encoder = FlatEncoder::create(t.table, options_with(layout, IndexType::I32, dtype));
                const auto batch = encoder->encode(input_of(rows));
                const Graphs reference = decode(*batch, encoder->schema());
                // the pool
                ThreadPool pool(3);
                EncodeScratch scratch;
                expect_same_bytes(*batch, *encoder->encode(input_of(rows), pool, scratch));
                EXPECT_EQ(scratch.numeric_tuples(), [&] {
                    u64 n = 0;
                    for (const auto& g : reference)
                        n += g.size();
                    return n;
                }());
                // single rows
                for (u64 g = 0; g < rows.size(); g += 97)
                {
                    const test::Rows one = row_of(rows, g, width);
                    const auto single = encoder->encode(input_of(one));
                    const Graphs got = decode(*single, encoder->schema());
                    ASSERT_EQ(got.size(), 1u);
                    expect_equal(got, Graphs{reference[g]});
                }
                // the append-only stream, in pieces
                FlatStream stream(encoder);
                for (u64 g = 0; g < rows.size(); g += 500)
                {
                    const u64 n = std::min<u64>(500, rows.size() - g);
                    test::Rows part;
                    part.words = rows.words;
                    part.numeric_words = rows.numeric_words;
                    part.data.assign(rows.data.begin() + static_cast<std::ptrdiff_t>(g * width),
                                     rows.data.begin() + static_cast<std::ptrdiff_t>((g + n) * width));
                    part.task_ids.assign(rows.task_ids.begin() + static_cast<std::ptrdiff_t>(g),
                                         rows.task_ids.begin() + static_cast<std::ptrdiff_t>(g + n));
                    stream.append(input_of(part));
                }
                expect_same_bytes(*batch, *stream.flush(pool));
                // the mutable stream, stored under the row indices and flushed in index order
                FlatMutableStream mutable_stream(encoder);
                std::vector<i64> indices(rows.size());
                for (u64 g = 0; g < rows.size(); ++g)
                    indices[g] = static_cast<i64>(g);
                mutable_stream.append(indices, input_of(rows));
                expect_same_bytes(*batch, *mutable_stream.flush(pool));
                // the mutable stream after an update: the values are those of the new rows
                const test::Rows replacement = row_of(rows, rows.size() - 1, width);
                const std::vector<i64> first{0};
                mutable_stream.update(first, input_of(replacement));
                const auto changed = decode(*mutable_stream.flush(pool), encoder->schema());
                expect_equal(Graphs{changed[0]}, Graphs{reference[rows.size() - 1]});
            }
    }
}

TEST(EncoderNumeric, CapacitiesPadWithZerosAndOverflowOnTheNumericCount)
{
    const auto table = tables()[0].table;
    const test::Rows rows = mixed_rows(*table, 10, 5);
    for (const Layout layout : k_layouts)
        for (const NumericDtype dtype : k_dtypes)
        {
            const auto encoder = FlatEncoder::create(table, options_with(layout, IndexType::I32, dtype));
            ThreadPool pool(1);
            EncodeScratch scratch;
            const auto exact = encoder->encode(input_of(rows), pool, scratch);
            const u64 numeric = scratch.numeric_tuples();
            ASSERT_GT(numeric, 0u);
            FlatSizes sizes;
            const BatchShape roomy{rows.size(), static_cast<u64>(ints(*exact, "entity_offsets").back()) + 5,
                                   ints(*exact, "relation_args").size() + 5, 0, 0, numeric + 7};
            const auto padded = encoder->encode_capacity(input_of(rows), roomy, pool, scratch, sizes);
            ASSERT_FALSE(sizes.overflowed());
            EXPECT_NO_THROW(validate(*padded, encoder->schema()));
            const auto values = reals(*padded, "numeric_values", value_dtype(encoder->schema().options()));
            const auto want = reals(*exact, "numeric_values", value_dtype(encoder->schema().options()));
            ASSERT_EQ(values.size(), numeric + 7);
            for (u64 i = 0; i < numeric; ++i)
                EXPECT_TRUE(same_value(values[i], want[i]));
            for (u64 i = numeric; i < values.size(); ++i)
                EXPECT_EQ(values[i], 0.0);
            // one numeric value too few: nothing is written
            BatchShape tight = roomy;
            tight.numeric = numeric - 1;
            const auto overflowed = encoder->encode_capacity(input_of(rows), tight, pool, scratch, sizes);
            EXPECT_TRUE(sizes.overflow & k_overflow_capacity);
            for (const rl::ArrayInfo& info : overflowed->arrays())
            {
                const auto* data = static_cast<const u8*>(overflowed->data(info));
                for (u64 i = 0; i < info.elements() * rl::dtype_bytes(info.dtype); ++i)
                    ASSERT_EQ(data[i], 0) << info.name.str();
            }
            // the stream's capacity form
            FlatStream stream(encoder);
            stream.append(input_of(rows));
            (void)stream.flush(pool, tight, sizes);
            EXPECT_TRUE(sizes.overflow & k_overflow_capacity);
            EXPECT_EQ(stream.size(), rows.size());
            (void)stream.flush(pool, roomy, sizes);
            EXPECT_EQ(stream.size(), 0u);
        }
}

TEST(EncoderNumeric, TheNumericOptionLeavesTheValuesOut)
{
    for (const Table& t : tables())
    {
        const test::Rows rows = mixed_rows(*t.table, 8, 2);
        FlatOptions options = test::encoder_options();
        options.include_numeric = false;
        const auto without = FlatEncoder::create(t.table, options);
        const auto with = FlatEncoder::create(t.table, test::encoder_options());
        const auto a = without->encode(input_of(rows));
        EXPECT_EQ(a->find("numeric_values"), nullptr);
        EXPECT_EQ(a->find("node_value"), nullptr);
        EXPECT_NE(with->encode(input_of(rows))->find("numeric_values"), nullptr);
        // the rows still carry the table's numeric words
        test::Rows narrow = rows;
        narrow.numeric_words = 0;
        EXPECT_THROW((void)without->encode(input_of(narrow)), std::invalid_argument);
    }
}

TEST(EncoderNumeric, RowsOfAnotherNumericWidthAreRefused)
{
    const auto table = tables()[0].table;
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    test::Rows rows = mixed_rows(*table, 4, 1);
    EXPECT_NO_THROW((void)encoder->encode(input_of(rows)));
    test::Rows narrow = rows;
    narrow.numeric_words = 0;
    EXPECT_THROW((void)encoder->encode(input_of(narrow)), std::invalid_argument);
    test::Rows wide = rows;
    wide.numeric_words += 1;
    EXPECT_THROW((void)encoder->encode(input_of(wide)), std::invalid_argument);
    FlatStream stream(encoder);
    EXPECT_THROW(stream.append(input_of(narrow)), std::invalid_argument);
    EXPECT_EQ(stream.size(), 0u);
    // a classical table refuses numeric words
    const auto classical = FlatEncoder::create(rl::TaskTable::single(test::pddl_task("blocks", "smedium")),
                                               test::encoder_options());
    test::Rows blocks = test::random_walk(*classical->table_ptr(), 0, 3, 1);
    blocks.numeric_words = 1;
    EXPECT_THROW((void)classical->encode(input_of(blocks)), std::invalid_argument);
}

TEST(EncoderNumeric, HorizonsRefuseNumericDomains)
{
    const auto table = tables()[0].table;
    for (const HorizonMode mode : {HorizonMode::Full, HorizonMode::Delta, HorizonMode::Action})
    {
        FlatOptions options = test::encoder_options();
        options.horizon.mode = mode;
        try
        {
            (void)FlatEncoder::create(table, options);
            ADD_FAILURE() << "a horizon over numeric functions was created";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("horizon encodings do not encode numeric functions"), std::string::npos)
                << e.what();
        }
        // without the numeric relations a horizon over the table is a horizon over its atoms, but the rows still
        // carry numeric words, which a horizon batch does not take
        options.include_numeric = false;
        EXPECT_THROW((void)FlatEncoder::create(table, options), std::invalid_argument);
    }
}

TEST(EncoderNumeric, TheSimpleGraphGivesEveryNumericAtomNodeItsValue)
{
    for (const Table& t : tables())
    {
        const test::Rows rows = mixed_rows(*t.table, 12, 9);
        for (const Layout layout : k_layouts)
        {
            FlatOptions both = options_with(layout, IndexType::I32, NumericDtype::F32);
            both.outputs = {Output::Relations, Output::Simple};
            FlatOptions simple_only = test::encoder_options();
            simple_only.outputs = {Output::Simple};
            simple_only.entity_arrays = EntityArrays::Always;
            for (const FlatOptions& options : {both, simple_only})
            {
                if (options.outputs == Outputs{Output::Simple} && layout == Layout::Relation)
                    continue;
                SCOPED_TRACE(t.name + " layout=" + std::string(to_string(layout)));
                const auto encoder = FlatEncoder::create(t.table, options);
                const auto batch = encoder->encode(input_of(rows));
                EXPECT_NO_THROW(validate(*batch, encoder->schema()));
                ASSERT_NE(batch->find("node_value"), nullptr);
                EXPECT_EQ(batch->find("numeric_values") != nullptr, options.outputs.contains(Output::Relations));
                const auto node_value = reals(*batch, "node_value", rl::DType::F32);
                const auto node_offsets = ints(*batch, "node_offsets");
                ASSERT_EQ(node_value.size(), static_cast<u64>(node_offsets.back()));
                // the nodes of a graph: its entities, then its atoms relation after relation (the tuples of the numeric
                // relations are the last ones)
                const Graphs want = expected(*t.table, encoder->schema(), rows, *batch);
                const auto entity_offsets = ints(*batch, "entity_offsets");
                for (u64 g = 0; g < rows.size(); ++g)
                {
                    const i64 begin = node_offsets[g], end = node_offsets[g + 1];
                    const i64 entities = entity_offsets[g + 1] - entity_offsets[g];
                    const i64 numeric = static_cast<i64>(want[g].size());
                    for (i64 n = begin; n < end - numeric; ++n)
                        EXPECT_EQ(node_value[static_cast<u64>(n)], 0.0f) << "graph " << g << " node " << n - begin;
                    ASSERT_GE(end - begin, entities + numeric);
                    for (i64 k = 0; k < numeric; ++k)
                        EXPECT_TRUE(same_value(node_value[static_cast<u64>(end - numeric + k)], want[g][static_cast<u64>(k)].value))
                            << "graph " << g << " numeric tuple " << k;
                }
            }
        }
    }
}

TEST(EncoderNumeric, TheLaneGroupFormsWriteTheSameValues)
{
    for (const Table& t : tables())
    {
        const test::Rows rows = mixed_rows(*t.table, 9, 4);
        for (const Layout layout : k_layouts)
            for (const NumericDtype dtype : k_dtypes)
            {
                SCOPED_TRACE(t.name + " layout=" + std::string(to_string(layout)) + std::string(to_string(dtype)));
                const auto encoder = FlatEncoder::create(t.table, options_with(layout, IndexType::I32, dtype));
                const auto reference = encoder->encode(input_of(rows));
                const u64 G = rows.size(), R = encoder->schema().num_relations();
                std::map<std::string, std::vector<std::byte>> plain, grouped, expected_arrays;
                for (const auto& a : reference->arrays())
                {
                    const auto* first = static_cast<const std::byte*>(reference->data(a));
                    expected_arrays[a.name.str()] = {first, first + a.elements() * rl::dtype_bytes(a.dtype)};
                }
                for (auto* dst : {&plain, &grouped})
                {
                    *dst = expected_arrays;
                    for (const char* name : {"num_objects", "relation_counts", "relation_args", "entity_role",
                                             "entity_ref", "numeric_values"})
                        std::fill((*dst)[name].begin(), (*dst)[name].end(), std::byte{0xa5});
                    if (layout == Layout::Graph)
                        for (const char* name : {"relation_offsets", "relation_arg_offsets"})
                        {
                            std::fill((*dst)[name].begin(), (*dst)[name].end(), std::byte{0xa5});
                            std::memcpy((*dst)[name].data() + G * R * 4, expected_arrays[name].data() + G * R * 4, 4);
                        }
                }
                auto out_of = [&](std::map<std::string, std::vector<std::byte>>& arrays, u64 numeric_total)
                {
                    auto at = [&](const char* name) { return reinterpret_cast<i32*>(arrays[name].data()); };
                    ed::GraphOut<i32> out{G, at("num_objects"), at("relation_counts"), at("relation_offsets"),
                                          at("relation_arg_offsets"), at("relation_args"),
                                          reinterpret_cast<u8*>(arrays["entity_role"].data()), at("entity_ref")};
                    if (dtype == NumericDtype::F32)
                        out.numeric_f32 = reinterpret_cast<f32*>(arrays["numeric_values"].data());
                    else
                        out.numeric_f64 = reinterpret_cast<f64*>(arrays["numeric_values"].data());
                    const u64 tuples = static_cast<u64>(at("relation_offsets")[G * R]);
                    out.numeric_block = layout == Layout::Relation ? tuples - numeric_total : 0;
                    return out;
                };
                u64 numeric_total = 0;
                for (u64 g = 0; g < G; ++g)
                    numeric_total += ed::numeric_slot_count(
                        ed::instance_view(encoder->context().view(), static_cast<u32>(rows.task_ids[g])));
                const auto offsets = reinterpret_cast<const i32*>(expected_arrays["relation_offsets"].data());
                const auto arg_offsets = reinterpret_cast<const i32*>(expected_arrays["relation_arg_offsets"].data());
                struct Positions
                {
                    const i32* offsets;
                    const i32* arguments;
                    u32 count(u64 s) const { return static_cast<u32>(offsets[s + 1] - offsets[s]); }
                    u64 args(u64 s, u32) const { return static_cast<u64>(arguments[s]); }
                } positions{offsets, arg_offsets};
                const auto plain_out = out_of(plain, numeric_total);
                const auto grouped_out = out_of(grouped, numeric_total);
                const auto entity_offsets = reinterpret_cast<const i32*>(expected_arrays["entity_offsets"].data());
                const u32 row_width = t.table->row_words();
                u64 numeric_base = 0;
                for (u64 g = 0; g < G; ++g)
                {
                    const auto v = ed::instance_view(encoder->context().view(), static_cast<u32>(rows.task_ids[g]));
                    ed::RowInput row{rows.data.data() + g * row_width, nullptr};
                    row.numeric = row.atoms + rows.words;
                    const u32 cells = ed::cell_list(v, row).count;
                    std::vector<u32> a(cells), b(cells), va(v.num_virtual), vb(v.num_virtual);
                    const auto ta = ed::count_graph(v, row, a.data(), va.data());
                    const auto tb = ed::count_graph_group(ed::SoloGroup{}, v, row, b.data(), vb.data());
                    ASSERT_EQ(ta.tuples, tb.tuples);
                    ASSERT_EQ(a, b);
                    const u64 entity = static_cast<u64>(entity_offsets[g]);
                    if (layout == Layout::Graph)
                    {
                        const u64 tuple = static_cast<u64>(offsets[g * R]);
                        const u64 arg = static_cast<u64>(arg_offsets[g * R]);
                        ed::fill_graph<i32>(v, row, g, tuple, arg, entity, ta, plain_out, 0, va.data(), a.data(), 0,
                                            numeric_base);
                        ed::fill_graph_group<i32>(ed::SoloGroup{}, v, row, g, tuple, arg, entity, tb, grouped_out, 0,
                                                  vb.data(), b.data(), numeric_base);
                    }
                    else
                    {
                        ed::fill_graph_positioned<i32>(v, row, g, entity, ta, plain_out, positions, 0, va.data(), a.data());
                        ed::fill_graph_positioned_group<i32>(ed::SoloGroup{}, v, row, g, entity, tb, grouped_out, positions, 0,
                                                             vb.data(), b.data());
                    }
                    numeric_base += ed::numeric_slot_count(v);
                }
                EXPECT_EQ(plain, expected_arrays);
                EXPECT_EQ(grouped, expected_arrays);
            }
    }
}
