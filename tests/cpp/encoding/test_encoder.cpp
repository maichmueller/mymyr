#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/validate.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstring>
#include <limits>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
struct Problem
{
    const char* domain;
    const char* problem;
};
const std::vector<Problem> k_problems = {{"blocks", "smedium"},    {"blocks", "probBLOCKS-8-0"}, {"gripper", "p01"},
                                         {"logistics00", "probLOGISTICS-6-1"}, {"visitall", "test_problem"},
                                         {"philosophers", "p03-phil4"}, {"delivery", "test_problem"},
                                         {"features", "p01"},           {"blocks_flags", "p01"},      {"relay", "p01"}};

std::vector<i64> ints(const rl::ArrayBundle& bundle, const char* name)
{
    const rl::ArrayInfo* info = bundle.find(name);
    EXPECT_NE(info, nullptr) << name;
    std::vector<i64> out;
    if (!info)
        return out;
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

using Tuple = std::vector<i64>;
using Graph = std::vector<std::vector<Tuple>>;  // [relation][tuple]: instance-local entity ids

std::vector<Graph> decode(const rl::ArrayBundle& bundle, const EncodingSchema& schema)
{
    const u64 R = schema.num_relations();
    const auto entity_offsets = ints(bundle, "entity_offsets");
    const auto counts = ints(bundle, "relation_counts");
    const auto arg_offsets = ints(bundle, "relation_arg_offsets");
    const auto args = ints(bundle, "relation_args");
    std::vector<Graph> out(entity_offsets.size() - 1, Graph(R));
    for (u64 g = 0; g + 1 < entity_offsets.size(); ++g)
        for (u64 r = 0; r < R; ++r)
        {
            const u32 arity = schema.relation(static_cast<u32>(r)).arity;
            for (i64 t = 0; t < counts[g * R + r]; ++t)
            {
                Tuple tuple;
                for (u32 k = 0; k < arity; ++k)
                    tuple.push_back(args[arg_offsets[g * R + r] + t * arity + k] - entity_offsets[g]);
                out[g][r].push_back(tuple);
            }
        }
    return out;
}

/// The tuples of a state, from the task's own APIs (no encoder array is read): the atoms of the predicates the schema
/// has a state relation for, sorted, and a nullary atom as [auxiliary].
Graph reference(const EncodingSchema& schema, const rl::TaskTable& table, u32 instance, const u64* row)
{
    const Task& task = *table.task(instance);
    const auto& map = table.instance(instance).pred_map;
    const i64 auxiliary = task.num_objects();
    Graph out(schema.num_relations());
    auto add = [&](u32 instance_pred, const std::vector<i64>& objects)
    {
        const u32 tp = map[instance_pred];
        if (tp >= table.num_domain_predicates())
            return;
        const auto relation = schema.find(table.predicate_names()[tp]);
        if (!relation)
            return;
        out[*relation].push_back(objects.empty() ? Tuple{auxiliary} : objects);
    };
    auto record_objects = [&](const u32* record, u32 pred)
    {
        std::vector<i64> objects;
        for (u32 k = 0; k < task.compiled().arity[pred]; ++k)
            objects.push_back(record[1 + k]);
        return objects;
    };
    for (u32 slot = 0; slot < task.atoms().fluent_slots(); ++slot)
        if ((row[slot >> 6] >> (slot & 63)) & 1)
        {
            const u32* record = task.atoms().record(AtomKind::Fluent, slot);
            add(record[0], record_objects(record, record[0]));
        }
    if (task.atoms().derived_slots() > 0)
    {
        const WorkspaceLease workspace = task.workspace();
        auto& engine = workspace->engine();
        engine.set_state(row, table.instance(instance).words);
        engine.build_view();
        workspace->axioms().evaluate();
        const u64* derived = engine.derived();
        for (u32 slot = 0; slot < task.atoms().derived_slots(); ++slot)
            if ((derived[slot >> 6] >> (slot & 63)) & 1)
            {
                const u32* record = task.atoms().record(AtomKind::Derived, slot);
                add(record[0], record_objects(record, record[0]));
            }
    }
    for (const formalism::GroundAtom& atom : task.data().static_init)
    {
        std::vector<i64> objects;
        for (const auto object : task.data().objects_of(atom))
            objects.push_back(object.v);
        add(atom.pred.v, objects);
    }
    for (auto& list : out)
    {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    return out;
}

std::shared_ptr<const rl::ArrayBundle> encode_checked(const FlatEncoder& encoder, const test::Rows& rows)
{
    EncodeInput in;
    in.states = rows.view();
    in.task_ids = rows.task_ids.data();
    const auto bundle = encoder.encode(in);
    validate(*bundle, encoder.schema());
    return bundle;
}

void expect_equal_bundles(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    ASSERT_EQ(a.arrays().size(), b.arrays().size());
    for (usize i = 0; i < a.arrays().size(); ++i)
    {
        const rl::ArrayInfo& x = a.arrays()[i];
        const rl::ArrayInfo& y = b.arrays()[i];
        ASSERT_EQ(x.name, y.name);
        EXPECT_EQ(x.dtype, y.dtype) << x.name;
        EXPECT_EQ(x.shape, y.shape) << x.name;
        EXPECT_EQ(x.offset, y.offset) << x.name;
        EXPECT_EQ(std::memcmp(a.data(x), b.data(y), x.elements() * rl::dtype_bytes(x.dtype)), 0) << x.name;
    }
    EXPECT_EQ(a.scalars(), b.scalars());
}

void expect_matches_reference(const FlatOptions& options)
{
    for (const Problem& p : k_problems)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const auto encoder = FlatEncoder::create(table, options);
        const test::Rows rows = test::random_walk(*table, 0, 256, 17);
        const auto bundle = encode_checked(*encoder, rows);
        const auto decoded = decode(*bundle, encoder->schema());
        ASSERT_EQ(decoded.size(), rows.size());
        for (u64 g = 0; g < rows.size(); ++g)
        {
            const Graph expected = reference(encoder->schema(), *table, 0, rows.data.data() + g * rows.words);
            ASSERT_EQ(decoded[g].size(), expected.size());
            for (usize r = 0; r < expected.size(); ++r)
                ASSERT_EQ(decoded[g][r], expected[r]) << "graph " << g << " relation " << encoder->schema().relation(static_cast<u32>(r)).name;
        }
    }
}

TEST(EncoderReference, DefaultOptions) { expect_matches_reference(test::encoder_options()); }

TEST(EncoderReference, WithEquality)
{
    FlatOptions options = test::encoder_options();
    options.equality = true;
    expect_matches_reference(options);
}

TEST(EncoderReference, WithoutNullaryStaticOrDerived)
{
    FlatOptions options = test::encoder_options();
    options.nullary = false;
    expect_matches_reference(options);
    options.include_static = false;
    expect_matches_reference(options);
    options.include_derived = false;
    expect_matches_reference(options);
}

TEST(EncoderReference, Int64IndicesGiveTheSameValues)
{
    FlatOptions wide = test::encoder_options();
    wide.index = IndexType::I64;
    const auto table = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    const auto narrow_encoder = FlatEncoder::create(table, test::encoder_options());
    const auto wide_encoder = FlatEncoder::create(table, wide);
    const test::Rows rows = test::random_walk(*table, 0, 64, 3);
    const auto a = encode_checked(*narrow_encoder, rows);
    const auto b = encode_checked(*wide_encoder, rows);
    for (const char* name : {"num_objects", "entity_offsets", "entity_ref", "relation_counts", "relation_offsets",
                             "relation_arg_offsets", "relation_args"})
    {
        EXPECT_EQ(a->find(name)->dtype, rl::DType::I32) << name;
        EXPECT_EQ(b->find(name)->dtype, rl::DType::I64) << name;
        EXPECT_EQ(a->find(name)->shape, b->find(name)->shape) << name;
        EXPECT_EQ(ints(*a, name), ints(*b, name)) << name;
    }
    EXPECT_EQ(a->find("task_ids")->dtype, rl::DType::I32);
    EXPECT_EQ(b->find("task_ids")->dtype, rl::DType::I32);
    EXPECT_EQ(a->find("entity_role")->dtype, rl::DType::U8);
    EXPECT_EQ(a->scalar("index_dtype"), 0);
    EXPECT_EQ(b->scalar("index_dtype"), 1);
    EXPECT_EQ(a->scalar("version"), 1);
    EXPECT_EQ(a->scalar("layout"), 0);
    EXPECT_EQ(a->scalar("statics"), 0);
    EXPECT_EQ(a->scalar("num_graphs"), 64);
    EXPECT_EQ(a->scalar("num_relations"), narrow_encoder->schema().num_relations());
    EXPECT_EQ(a->scalar("schema_fingerprint"), std::bit_cast<i64>(narrow_encoder->schema().fingerprint()));
    EXPECT_NE(a->scalar("schema_fingerprint"), b->scalar("schema_fingerprint"));
}

TEST(EncoderAuxiliary, ExistsExactlyWhenANullaryTupleDoes)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "probBLOCKS-8-0"));
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    const test::Rows rows = test::random_walk(*table, 0, 256, 5);
    const auto bundle = encode_checked(*encoder, rows);
    const auto decoded = decode(*bundle, encoder->schema());
    const auto offsets = ints(*bundle, "entity_offsets");
    const auto role = ints(*bundle, "entity_role");
    const auto ref = ints(*bundle, "entity_ref");
    const u32 handempty = *encoder->schema().find("handempty");
    const i64 objects = table->instance(0).num_objects;
    u32 with = 0, without = 0;
    for (u64 g = 0; g < rows.size(); ++g)
    {
        const i64 entities = offsets[g + 1] - offsets[g];
        if (decoded[g][handempty].empty())
        {
            ++without;
            EXPECT_EQ(entities, objects);
        }
        else
        {
            ++with;
            ASSERT_EQ(entities, objects + 1);
            EXPECT_EQ(decoded[g][handempty], (std::vector<Tuple>{{objects}}));
            EXPECT_EQ(role[offsets[g] + objects], 8);
            EXPECT_EQ(ref[offsets[g] + objects], -1);
        }
    }
    EXPECT_GT(with, 0u);
    EXPECT_GT(without, 0u);
}

TEST(EncoderAuxiliary, NeverExistsWithoutNullaryRelations)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "probBLOCKS-8-0"));
    FlatOptions options = test::encoder_options();
    options.nullary = false;
    const auto encoder = FlatEncoder::create(table, options);
    EXPECT_FALSE(encoder->schema().find("handempty"));
    const test::Rows rows = test::random_walk(*table, 0, 64, 5);
    const auto bundle = encode_checked(*encoder, rows);
    const auto offsets = ints(*bundle, "entity_offsets");
    for (u64 g = 0; g < rows.size(); ++g)
        EXPECT_EQ(offsets[g + 1] - offsets[g], table->instance(0).num_objects);
    for (const i64 value : ints(*bundle, "entity_role"))
        EXPECT_EQ(value, 0);
}

TEST(EncoderAuxiliary, AStaticNullaryAtomAloneGivesTheGraphAnAuxiliaryEntity)
{
    // every graph of a state-free instance still has the static tuples of the instance; the auxiliary entity follows
    // whichever nullary tuple exists, in any lane
    const auto table = rl::TaskTable::single(test::pddl_task("blocks_flags", "p01"));
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    const test::Rows rows = test::random_walk(*table, 0, 64, 9);
    const auto bundle = encode_checked(*encoder, rows);
    const auto decoded = decode(*bundle, encoder->schema());
    const auto offsets = ints(*bundle, "entity_offsets");
    for (u64 g = 0; g < rows.size(); ++g)
    {
        bool nullary = false;
        for (u32 r = 0; r < encoder->schema().num_relations(); ++r)
            nullary = nullary || (encoder->schema().relation(r).logical_arity == 0 && !decoded[g][r].empty());
        EXPECT_EQ(offsets[g + 1] - offsets[g], table->instance(0).num_objects + (nullary ? 1 : 0));
    }
}

TEST(EncoderEquality, OneEqualityTuplePerObjectOnlyOnRequest)
{
    const auto table = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    EXPECT_FALSE(FlatEncoder::create(table, test::encoder_options())->schema().find("="));
    FlatOptions options = test::encoder_options();
    options.equality = true;
    const auto encoder = FlatEncoder::create(table, options);
    const u32 equality = *encoder->schema().find("=");
    const test::Rows rows = test::random_walk(*table, 0, 8, 1);
    const auto decoded = decode(*encode_checked(*encoder, rows), encoder->schema());
    const i64 objects = table->instance(0).num_objects;
    for (const Graph& graph : decoded)
    {
        ASSERT_EQ(static_cast<i64>(graph[equality].size()), objects);
        for (i64 o = 0; o < objects; ++o)
            EXPECT_EQ(graph[equality][o], (Tuple{o, o}));
    }
}

/// The slice [a, b) of a batch equals the batch of the slice, with the offsets rebased.
void expect_slice(const rl::ArrayBundle& whole, const rl::ArrayBundle& slice, u64 a, u64 b, u64 R)
{
    const auto eo = ints(whole, "entity_offsets"), ro = ints(whole, "relation_offsets"), ao = ints(whole, "relation_arg_offsets");
    const auto ceo = ints(slice, "entity_offsets"), cro = ints(slice, "relation_offsets"), cao = ints(slice, "relation_arg_offsets");
    ASSERT_EQ(ceo.size(), b - a + 1);
    for (u64 i = 0; i <= b - a; ++i)
        ASSERT_EQ(ceo[i], eo[a + i] - eo[a]);
    ASSERT_EQ(cro.size(), (b - a) * R + 1);
    for (u64 i = 0; i < cro.size(); ++i)
    {
        ASSERT_EQ(cro[i], ro[a * R + i] - ro[a * R]);
        ASSERT_EQ(cao[i], ao[a * R + i] - ao[a * R]);
    }
    const auto counts = ints(whole, "relation_counts"), ccounts = ints(slice, "relation_counts");
    ASSERT_TRUE(std::equal(ccounts.begin(), ccounts.end(), counts.begin() + static_cast<std::ptrdiff_t>(a * R)));
    const auto args = ints(whole, "relation_args"), cargs = ints(slice, "relation_args");
    ASSERT_EQ(static_cast<i64>(cargs.size()), ao[b * R] - ao[a * R]);
    for (usize i = 0; i < cargs.size(); ++i)
        ASSERT_EQ(cargs[i], args[ao[a * R] + i] - eo[a]);
    for (const char* name : {"entity_role", "entity_ref", "num_objects"})
    {
        const auto x = ints(whole, name), y = ints(slice, name);
        const i64 from = std::string(name) == "num_objects" ? static_cast<i64>(a) : eo[a];
        ASSERT_TRUE(std::equal(y.begin(), y.end(), x.begin() + from)) << name;
    }
}

TEST(EncoderBatches, ASliceOfABatchIsTheEncodingOfTheSlice)
{
    const auto small = test::pddl_task("blocks", "smedium");
    const auto large = test::pddl_task("blocks", "probBLOCKS-8-0");
    const auto table = rl::TaskTable::create({small, large});
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    test::Rows rows = test::random_walk(*table, 0, 40, 1);
    const test::Rows other = test::random_walk(*table, 1, 40, 2);
    // interleave the instances
    test::Rows mixed;
    mixed.words = rows.words;
    const u32 row_words = table->row_words();
    for (u64 i = 0; i < 40; ++i)
        for (const test::Rows* source : std::initializer_list<const test::Rows*>{&rows, &other})
        {
            mixed.data.insert(mixed.data.end(), source->data.begin() + static_cast<std::ptrdiff_t>(i * row_words),
                              source->data.begin() + static_cast<std::ptrdiff_t>((i + 1) * row_words));
            mixed.task_ids.push_back(source->task_ids[i]);
        }
    const auto whole = encode_checked(*encoder, mixed);
    const u64 R = encoder->schema().num_relations();
    std::mt19937_64 rng(4);
    for (int round = 0; round < 30; ++round)
    {
        const u64 a = rng() % 80, b = a + 1 + rng() % (80 - a);
        test::Rows slice;
        slice.words = mixed.words;
        slice.data.assign(mixed.data.begin() + static_cast<std::ptrdiff_t>(a * row_words),
                          mixed.data.begin() + static_cast<std::ptrdiff_t>(b * row_words));
        slice.task_ids.assign(mixed.task_ids.begin() + static_cast<std::ptrdiff_t>(a), mixed.task_ids.begin() + static_cast<std::ptrdiff_t>(b));
        const auto part = encode_checked(*encoder, slice);
        expect_slice(*whole, *part, a, b, R);
        if (HasFatalFailure())
            return;
    }
    // a single row is its graph in the batch, and an encoder of that instance alone gives the same tuples
    const auto single_encoder = FlatEncoder::create(rl::TaskTable::single(large), test::encoder_options());
    ASSERT_EQ(single_encoder->schema().fingerprint(), encoder->schema().fingerprint());
    const auto batch_graphs = decode(*whole, encoder->schema());
    for (u64 g = 1; g < 10; g += 2)
    {
        test::Rows one;
        one.words = single_encoder->table().words();
        one.data.assign(mixed.data.begin() + static_cast<std::ptrdiff_t>(g * row_words),
                        mixed.data.begin() + static_cast<std::ptrdiff_t>(g * row_words + one.words));
        one.task_ids = {0};
        ASSERT_EQ(mixed.task_ids[g], 1);
        EXPECT_EQ(decode(*encode_checked(*single_encoder, one), single_encoder->schema())[0], batch_graphs[g]);
    }
}

TEST(EncoderBatches, PaddingWordsAndRowStridesDoNotChangeTheOutput)
{
    test::dirty_recycled_blocks();
    const auto small = test::pddl_task("blocks", "smedium");
    const auto large = test::pddl_task("blocks", "probBLOCKS-8-0");
    const auto table = rl::TaskTable::create({small, large});
    ASSERT_LT(table->instance(0).words, table->instance(1).words);
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    test::Rows rows = test::random_walk(*table, 0, 30, 1);
    const test::Rows other = test::random_walk(*table, 1, 30, 2);
    rows.data.insert(rows.data.end(), other.data.begin(), other.data.end());
    rows.task_ids.insert(rows.task_ids.end(), other.task_ids.begin(), other.task_ids.end());
    const auto clean = encode_checked(*encoder, rows);

    std::mt19937_64 rng(8);
    test::Rows garbage = rows;
    for (u64 g = 0; g < garbage.size(); ++g)
        for (u32 w = table->instance(static_cast<u32>(garbage.task_ids[g])).words; w < garbage.words; ++w)
            garbage.data[g * garbage.words + w] = rng();
    expect_equal_bundles(*clean, *encode_checked(*encoder, garbage));

    // rows of a wider stride, with garbage between them
    const u64 stride = rows.words + 3;
    std::vector<u64> strided(rows.size() * stride);
    for (u64 g = 0; g < rows.size(); ++g)
        for (u64 w = 0; w < stride; ++w)
            strided[g * stride + w] = w < rows.words ? rows.data[g * rows.words + w] : rng();
    EncodeInput in;
    in.states = {strided.data(), rows.size(), rows.words, stride, 0};
    in.task_ids = rows.task_ids.data();
    expect_equal_bundles(*clean, *encoder->encode(in));
}

TEST(EncoderDerived, ProvidedDerivedWordsGiveTheSameBundleAsEvaluatingThem)
{
    const auto task = test::pddl_task("philosophers", "p03-phil4");
    const auto table = rl::TaskTable::single(task);
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    const test::Rows rows = test::random_walk(*table, 0, 64, 6);
    const auto expected = encode_checked(*encoder, rows);
    const u64 stride = encoder->context().max_derived_words() + 2;
    ASSERT_GT(encoder->context().max_derived_words(), 0u);
    std::vector<u64> derived(rows.size() * stride, 0);
    const u32 words = bits::words_for(task->atoms().derived_slots());
    for (u64 g = 0; g < rows.size(); ++g)
    {
        const WorkspaceLease workspace = task->workspace();
        auto& engine = workspace->engine();
        engine.set_state(rows.data.data() + g * rows.words, table->instance(0).words);
        engine.build_view();
        workspace->axioms().evaluate();
        std::copy_n(engine.derived(), words, derived.begin() + static_cast<std::ptrdiff_t>(g * stride));
    }
    EncodeInput in;
    in.states = rows.view();
    in.derived = derived.data();
    in.derived_stride = stride;
    expect_equal_bundles(*expected, *encoder->encode(in));

    in.derived_stride = encoder->context().max_derived_words() - 1;
    EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    in.derived_stride = stride;
    ASSERT_NE(task->atoms().derived_slots() % 64, 0u);
    derived[words - 1] |= u64{1} << 63;
    EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
}

TEST(EncoderRefusals, InvalidOptions)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "smedium"));
    // an invalid combination is reported as such
    try
    {
        FlatOptions options = test::encoder_options();
        options.targets = {TargetSource::Subgoal};
        (void)FlatEncoder::create(table, options);
        ADD_FAILURE() << "the options were accepted";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("subgoal targets need max_goal_level"), std::string::npos) << error.what();
    }
    // the options an encoder accepts: actions (zero tuples), both index types, nullary and equality either way
    for (const bool actions : {true, false})
        for (const bool nullary : {true, false})
        {
            FlatOptions options = test::encoder_options();
            options.include_actions = actions;
            options.nullary = nullary;
            EXPECT_NO_THROW((void)FlatEncoder::create(table, options));
        }
}

TEST(EncoderRefusals, InvalidInput)
{
    const auto small = test::pddl_task("blocks", "smedium");
    const auto large = test::pddl_task("blocks", "probBLOCKS-8-0");
    const auto table = rl::TaskTable::create({small, large});
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    const test::Rows rows = test::random_walk(*table, 1, 4, 1);
    auto encode = [&](test::Rows r, u32 words, u32 numeric = 0)
    {
        EncodeInput in;
        in.states = {r.data.data(), r.size(), words, r.words, numeric};
        in.task_ids = r.task_ids.data();
        return encoder->encode(in);
    };
    EXPECT_NO_THROW((void)encode(rows, rows.words));
    test::Rows bad_id = rows;
    bad_id.task_ids[2] = 2;
    EXPECT_THROW((void)encode(bad_id, rows.words), std::invalid_argument);
    bad_id.task_ids[2] = -1;
    EXPECT_THROW((void)encode(bad_id, rows.words), std::invalid_argument);
    // a row narrower than its instance
    EXPECT_THROW((void)encode(rows, table->instance(1).words - 1), std::invalid_argument);
    // a bit its instance does not have: slot `fluent_slots` of the small instance
    test::Rows past = test::random_walk(*table, 0, 2, 1);
    const u32 slots = small->atoms().fluent_slots();
    ASSERT_LT(slots, 64u);
    past.data[1 * past.words] |= u64{1} << slots;
    EXPECT_THROW((void)encode(past, past.words), std::invalid_argument);
    // numeric words
    EXPECT_THROW((void)encode(rows, rows.words, 1), std::invalid_argument);
    // the rows of a numeric table carry the table's numeric words
    const auto numeric = rl::TaskTable::single(Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-counters.txt"));
    const auto numeric_encoder = FlatEncoder::create(numeric, test::encoder_options());
    test::Rows numeric_rows = test::random_walk(*numeric, 0, 2, 1);
    EncodeInput numeric_in;
    numeric_in.states = numeric_rows.view();
    numeric_in.task_ids = numeric_rows.task_ids.data();
    EXPECT_NO_THROW((void)numeric_encoder->encode(numeric_in));
    numeric_in.states.numeric_words = 0;
    EXPECT_THROW((void)numeric_encoder->encode(numeric_in), std::invalid_argument);
}

TEST(EncoderOverflow, IndexRangeIsCheckedForBothTypes)
{
    constexpr u64 max32 = std::numeric_limits<i32>::max();
    EXPECT_NO_THROW(encoding::detail::check_index_range(max32, IndexType::I32, "tuples"));
    EXPECT_NO_THROW(encoding::detail::check_index_range(0, IndexType::I32, "tuples"));
    try
    {
        encoding::detail::check_index_range(max32 + 1, IndexType::I32, "tuple arguments");
        ADD_FAILURE() << "no overflow";
    }
    catch (const std::overflow_error& error)
    {
        const std::string text = error.what();
        EXPECT_NE(text.find("mymyr: encoding: the batch needs 2147483648 tuple arguments"), std::string::npos) << text;
        EXPECT_NE(text.find("index_dtype='int64'"), std::string::npos) << text;
    }
    EXPECT_NO_THROW(encoding::detail::check_index_range(max32 + 1, IndexType::I64, "tuples"));
    EXPECT_NO_THROW(encoding::detail::check_index_range(static_cast<u64>(std::numeric_limits<i64>::max()), IndexType::I64, "tuples"));
    EXPECT_THROW(encoding::detail::check_index_range(static_cast<u64>(std::numeric_limits<i64>::max()) + 1, IndexType::I64, "tuples"),
                 std::overflow_error);
}

TEST(EncoderBatches, EncodingTwiceGivesEqualBytesAndAnEmptyBatchIsEmpty)
{
    test::dirty_recycled_blocks();
    const auto table = rl::TaskTable::single(test::pddl_task("logistics00", "probLOGISTICS-6-1"));
    const auto encoder = FlatEncoder::create(table, test::encoder_options());
    const test::Rows rows = test::random_walk(*table, 0, 32, 2);
    const auto a = encode_checked(*encoder, rows);
    const auto b = encode_checked(*encoder, rows);
    expect_equal_bundles(*a, *b);
    ASSERT_EQ(a->bytes(), b->bytes());
    EXPECT_EQ(std::memcmp(a->block(), b->block(), a->bytes()), 0);

    EncodeInput empty;
    empty.states = {nullptr, 0, table->words(), 0, 0};
    const auto none = encoder->encode(empty);
    validate(*none, encoder->schema());
    EXPECT_EQ(ints(*none, "entity_offsets"), (std::vector<i64>{0}));
    EXPECT_EQ(ints(*none, "relation_offsets"), (std::vector<i64>{0}));
    EXPECT_EQ(ints(*none, "relation_arg_offsets"), (std::vector<i64>{0}));
    EXPECT_EQ(none->scalar("num_graphs"), 0);
}

TEST(EncoderBatches, ActionRelationsHaveNoTuplesAndAnEncoderOfATaskWorks)
{
    const auto task = test::pddl_task("gripper", "p01");
    const auto encoder = FlatEncoder::create(task, test::encoder_options());
    EXPECT_EQ(encoder->table().size(), 1u);
    const u32 pick = *encoder->schema().find("pick");
    const test::Rows rows = test::random_walk(encoder->table(), 0, 16, 2);
    const auto decoded = decode(*encode_checked(*encoder, rows), encoder->schema());
    for (const Graph& graph : decoded)
        EXPECT_TRUE(graph[pick].empty());
}

// FNV-1a 64 over the scalars (name, then the value as 8 little-endian bytes), then each array's name, extents (8 bytes
// each, little-endian) and bytes. tests/python/test_encoding.py computes the same digest of the arrays that Python exports and asserts
// the same constants, so what Python exports is what this encoder wrote.
static_assert(std::endian::native == std::endian::little, "the digests hash scalars, extents and arrays in little-endian byte order");

u64 bundle_digest(const rl::ArrayBundle& bundle)
{
    u64 h = 0xCBF29CE484222325ull;
    const auto feed = [&h](const void* data, u64 bytes)
    {
        const auto* p = static_cast<const unsigned char*>(data);
        for (u64 i = 0; i < bytes; ++i)
            h = (h ^ p[i]) * 0x100000001B3ull;
    };
    for (const auto& [name, value] : bundle.scalars())
    {
        feed(name.data(), name.size());
        feed(&value, sizeof(value));
    }
    for (const rl::ArrayInfo& info : bundle.arrays())
    {
        feed(info.name.data(), info.name.size());
        for (const i64 extent : info.shape)
            feed(&extent, sizeof(extent));
        feed(bundle.data(info), info.elements() * rl::dtype_bytes(info.dtype));
    }
    return h;
}

/// `length` rows: the initial state, then each row's first successor.
test::Rows first_successor_chain(const rl::TaskTable& table, u64 length)
{
    test::Rows rows;
    rows.words = table.words();
    std::vector<u64> current(table.instance(0).init.begin(), table.instance(0).init.end());
    const i32 id = 0;
    for (u64 i = 0; i < length; ++i)
    {
        rows.data.insert(rows.data.end(), current.begin(), current.end());
        rows.task_ids.push_back(id);
        rl::Expansion out;
        out.words = rows.words;
        out.capacity = 4096;
        std::vector<u64> succ(out.capacity * rows.words, 0);
        out.succ = succ.data();
        rl::expand(table, {current.data(), 1, rows.words, 0, 0}, &id, out);
        EXPECT_FALSE(out.overflow());
        EXPECT_GT(out.total, 0u);
        current.assign(succ.begin(), succ.begin() + rows.words);
    }
    return rows;
}

TEST(EncoderBytes, TheDigestsOfTextTasksAreStable)
{
    const std::vector<std::pair<std::string, u64>> expected = {{"gripper__prob05", 9666753341531803529ull},
                                                               {"philosophers__p03-phil4", 8295029642410720015ull}};
    for (const auto& [name, digest] : expected)
    {
        const auto task = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/tasks/" + name + ".txt");
        const auto encoder = FlatEncoder::create(task, test::encoder_options());
        const test::Rows rows = first_successor_chain(encoder->table(), 6);
        EXPECT_EQ(bundle_digest(*encode_checked(*encoder, rows)), digest) << name;
    }
}
}  // namespace

TEST(EncoderBatches, SlotMappedStatesUseTheDestinationLayout)
{
    const auto fixed = test::pddl_task("blocks", "probBLOCKS-8-0");
    TaskOptions options;
    options.atoms = TaskOptions::Atoms::Lazy;
    const auto lazy = Task::create(fixed->data(), options);
    const auto twin = lazy->frozen();
    const auto map = lazy->slot_map_to(*twin)->snapshot();
    const State source = lazy->initial_state();
    const State destination = twin->initial_state();
    for (Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions encoding_options;
        encoding_options.layout = layout;
        const auto encoder = FlatEncoder::create(lazy, encoding_options);
        EncodeInput in;
        in.states = {destination.words().data(), 1, destination.size_words(), 0};
        const auto expected = encoder->encode(in);
        in.states = {source.words().data(), 1, source.size_words(), 0};
        in.slot_map = map->data();
        in.slot_map_size = static_cast<u32>(map->size());
        expect_equal_bundles(*expected, *encoder->encode(in));
        in.slot_map_size = 0;
        EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
        std::vector<u32> missing(map->size(), AtomIndex::k_empty);
        in.slot_map = missing.data();
        in.slot_map_size = static_cast<u32>(missing.size());
        EXPECT_THROW((void)encoder->encode(in), std::invalid_argument);
    }
}
