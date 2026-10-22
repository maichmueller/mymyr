// Shared statics of the relational encoder: the static tuples of an instance are written once per batch, and the shared
// form together with the graphs rebuilds the repeated form exactly.

#include "matrix.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"
#include "mymyr/frontend/domain.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <array>
#include <map>
#include <optional>
#include <random>
#include <set>

using namespace mymyr;
using namespace mymyr::encoding;
namespace em = mymyr::test::encoding_matrix;

namespace
{
using Arrays = std::map<std::string, std::vector<i64>>;

std::vector<i64> values_of(const rl::ArrayBundle& bundle, const rl::ArrayInfo& info)
{
    std::vector<i64> out;
    for (u64 i = 0; i < info.elements(); ++i)
    {
        const std::byte* at = static_cast<const std::byte*>(bundle.data(info)) + i * rl::dtype_bytes(info.dtype);
        switch (info.dtype)
        {
            case rl::DType::U8: out.push_back(std::to_integer<u8>(*at)); break;
            case rl::DType::I32: { i32 v; std::memcpy(&v, at, 4); out.push_back(v); break; }
            case rl::DType::I64: { i64 v; std::memcpy(&v, at, 8); out.push_back(v); break; }
            // numeric values compare by their bits
            case rl::DType::F32: { i32 v; std::memcpy(&v, at, 4); out.push_back(v); break; }
            case rl::DType::F64: { i64 v; std::memcpy(&v, at, 8); out.push_back(v); break; }
            default: ADD_FAILURE() << "unexpected dtype of " << info.name;
        }
    }
    return out;
}

Arrays read(const rl::ArrayBundle& bundle)
{
    Arrays out;
    for (const rl::ArrayInfo& info : bundle.arrays())
        out[std::string(info.name)] = values_of(bundle, info);
    return out;
}

void expect_equal(const Arrays& a, const Arrays& b, const std::string& context)
{
    ASSERT_EQ(a.size(), b.size()) << context;
    for (const auto& [name, values] : a)
    {
        const auto it = b.find(name);
        ASSERT_TRUE(it != b.end()) << context << ": " << name;
        EXPECT_EQ(values, it->second) << context << ": " << name;
    }
}

struct Shape
{
    Layout layout;
    IndexType index;
    [[nodiscard]] std::string text() const
    {
        return std::string(to_string(layout)) + "/" + std::string(to_string(index));
    }
};
const Shape k_shapes[] = {{Layout::Graph, IndexType::I32},
                          {Layout::Graph, IndexType::I64},
                          {Layout::Relation, IndexType::I32},
                          {Layout::Relation, IndexType::I64}};

FlatOptions shared_of(FlatOptions options)
{
    options.statics = Statics::Shared;
    return options;
}

/// The arrays of the repeated batch that the shared batch `a` of `schema` stands for: the static cells of every graph
/// from the block of its instance (ids shifted by the graph's first entity, the predicate keys replaced by the graph's
/// virtual nodes), the offsets and counts that follow, and the references of the goal, subgoal and history target
/// entities counted among the tuples of the repeated graph.
Arrays repeated_form(const EncodingSchema& schema, Arrays a)
{
    const u64 G = a.at("task_ids").size();
    const u64 R = schema.num_relations();
    const bool relation_major = schema.options().layout == Layout::Relation;
    const std::span<const u32> statics = schema.static_relations();
    const u64 Rs = statics.size();
    const auto& entity_offsets = a.at("entity_offsets");
    const auto& task_ids = a.at("task_ids");
    const auto& block_ids = a.at("static_task_ids");
    const bool entity_arrays = a.contains("entity_role");
    auto segment = [&](u64 g, u64 r) { return relation_major ? r * G + g : g * R + r; };
    std::map<u32, u64> static_index;
    for (u64 k = 0; k < Rs; ++k)
        static_index[statics[k]] = k;

    std::vector<std::vector<std::vector<i64>>> cells(G, std::vector<std::vector<i64>>(R));
    std::vector<std::vector<u64>> shared_tuples(G, std::vector<u64>(R, 0)), static_tuples(G, std::vector<u64>(R, 0));
    for (u64 g = 0; g < G; ++g)
    {
        const auto block = static_cast<u64>(std::ranges::lower_bound(block_ids, task_ids[g]) - block_ids.begin());
        EXPECT_LT(block, block_ids.size());
        const i64 first = entity_offsets[g], last = entity_offsets[g + 1];
        for (u64 r = 0; r < R; ++r)
        {
            const u32 arity = schema.relation(static_cast<u32>(r)).arity;
            const auto roles = schema.slot_roles(static_cast<u32>(r));
            const auto it = static_index.find(static_cast<u32>(r));
            if (it == static_index.end())
            {
                const u64 s = segment(g, r);
                const i64 begin = a.at("relation_arg_offsets")[s], end = a.at("relation_arg_offsets")[s + 1];
                cells[g][r].assign(a.at("relation_args").begin() + begin, a.at("relation_args").begin() + end);
                shared_tuples[g][r] = static_cast<u64>(end - begin) / arity;
                continue;
            }
            const u64 s = block * Rs + it->second;
            const i64 begin = a.at("static_arg_offsets")[s], end = a.at("static_arg_offsets")[s + 1];
            static_tuples[g][r] = static_cast<u64>(end - begin) / arity;
            for (i64 at = begin; at < end; ++at)
            {
                const i64 id = a.at("static_args")[static_cast<u64>(at)];
                const SlotRole role = roles[static_cast<u64>(at - begin) % arity];
                if (role != SlotRole::Predicate)
                {
                    cells[g][r].push_back(id + first);
                    continue;
                }
                i64 node = first;
                while (node < last && !(a.at("entity_role")[static_cast<u64>(node)] == 1 &&
                                        a.at("entity_ref")[static_cast<u64>(node)] == id))
                    ++node;
                EXPECT_LT(node, last) << "no virtual node for the key " << id;
                cells[g][r].push_back(node);
            }
        }
    }
    // the references of the goal, subgoal and history target entities count the graph's tuples, the static ones
    // included
    if (entity_arrays)
        for (u64 g = 0; g < G; ++g)
            for (i64 e = entity_offsets[g]; e < entity_offsets[g + 1]; ++e)
            {
                const i64 role = a.at("entity_role")[static_cast<u64>(e)];
                if (role != 2 && role != 3 && role != 6)
                    continue;
                i64& ref = a.at("entity_ref")[static_cast<u64>(e)];
                u64 passed = 0, r = 0;
                while (r < R && passed + shared_tuples[g][r] <= static_cast<u64>(ref))
                    passed += shared_tuples[g][r++];
                u64 before = 0;
                for (u64 q = 0; q < r; ++q)
                    before += static_tuples[g][q];
                ref += static_cast<i64>(before);
            }
    std::vector<i64> counts(G * R), arg_offsets, args;
    for (u64 s = 0; s < G * R; ++s)
    {
        const u64 g = relation_major ? s % G : s / R, r = relation_major ? s / G : s % R;
        counts[g * R + r] = static_cast<i64>(cells[g][r].size() / schema.relation(static_cast<u32>(r)).arity);
        arg_offsets.push_back(static_cast<i64>(args.size()));
        args.insert(args.end(), cells[g][r].begin(), cells[g][r].end());
    }
    arg_offsets.push_back(static_cast<i64>(args.size()));
    // the tuple offsets are the prefix sums of the counts in segment order
    std::vector<i64> tuple_offsets(G * R + 1, 0);
    for (u64 s = 0; s < G * R; ++s)
    {
        const u64 g = relation_major ? s % G : s / R, r = relation_major ? s / G : s % R;
        tuple_offsets[s + 1] = tuple_offsets[s] + counts[g * R + r];
    }
    if (a.contains("relation_counts"))
        a["relation_counts"] = counts;
    a["relation_offsets"] = tuple_offsets;
    a["relation_arg_offsets"] = arg_offsets;
    a["relation_args"] = args;
    for (const char* name : {"static_task_ids", "static_offsets", "static_arg_offsets", "static_args", "static_relations"})
        a.erase(name);
    return a;
}

/// Encodes with the repeated and the shared form and compares the rebuilt repeated form with the repeated batch.
void expect_same_as_repeated(const rl::TaskTablePtr& table, const FlatOptions& options, const EncodeInput& in,
                             const std::string& context)
{
    const auto repeated = FlatEncoder::create(table, options);
    const auto shared = FlatEncoder::create(table, shared_of(options));
    ThreadPool pool(1);
    EncodeScratch scratch;
    const auto expected = repeated->encode(in, pool, scratch);
    const auto actual = shared->encode(in, pool, scratch);
    ASSERT_NO_THROW(validate(*actual, shared->schema())) << context;
    EXPECT_EQ(actual->scalar("statics"), 1) << context;
    EXPECT_EQ(expected->scalar("statics"), 0) << context;
    expect_equal(read(*expected), repeated_form(shared->schema(), read(*actual)), context);
}

EncodeInput slice_of(const em::Batch& b, u64 first, u64 last, const em::Inputs& inputs)
{
    EncodeInput in = em::host_input(b, last - first, inputs);
    in.states.data += first * b.words;
    in.task_ids += first;
    if (in.derived)
        in.derived += first * b.derived_words;
    if (in.goal_pos)
    {
        in.goal_pos += first * b.words;
        in.goal_neg += first * b.words;
    }
    if (in.goal_derived_pos)
    {
        in.goal_derived_pos += first * b.derived_words;
        in.goal_derived_neg += first * b.derived_words;
    }
    return in;
}

TEST(SharedStatics, RebuildTheRepeatedFormOverTheCorpus)
{
    for (const em::Corpus& corpus : em::corpus())
    {
        const em::Batch batch = em::make_batch(corpus.table, 180, corpus.walk, 11);
        for (const em::Config& config : em::configs())
            for (const Shape& shape : k_shapes)
            {
                const FlatOptions options = em::with(config.options, shape.layout, shape.index);
                expect_same_as_repeated(corpus.table, options, em::host_input(batch, batch.count, config.inputs),
                                        std::string(corpus.name) + "/" + config.name + "/" + shape.text());
            }
    }
}

TEST(SharedStatics, RebuildTheRepeatedFormWithTheNumericLane)
{
    // the numeric relations are fluent: they stay in the graphs, and numeric_values is that of the repeated form
    u64 static_tables = 0;
    for (const em::Corpus& corpus : em::numeric_corpus())
    {
        const test::Rows rows = em::walk_rows(*corpus.table, 20, 13);
        const EncodeInput in = em::input_of(rows);
        for (const Shape& shape : k_shapes)
            for (const NumericDtype dtype : {NumericDtype::F32, NumericDtype::F64})
            {
                FlatOptions options = em::with(test::encoder_options(), shape.layout, shape.index);
                options.numeric_dtype = dtype;
                const auto schema = FlatEncoder::create(corpus.table, shared_of(options))->schema_ptr();
                ASSERT_TRUE(schema->stores_numeric_values());
                for (const u32 r : schema->static_relations())
                    EXPECT_LT(r, schema->first_numeric_relation()) << corpus.name;
                static_tables += !schema->static_relations().empty();
                expect_same_as_repeated(corpus.table, options, in,
                                        std::string(corpus.name) + "/" + shape.text() + "/" +
                                            std::string(to_string(dtype)));
            }
    }
    EXPECT_GT(static_tables, 0u);
}

TEST(SharedStatics, ASingleRowAndAnEmptyBatchHaveTheirBlocks)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 40, true, 3);
    const auto encoder = FlatEncoder::create(table, shared_of(test::encoder_options()));
    for (const u64 g : {u64{0}, u64{1}, u64{39}})
    {
        const Arrays a = read(*encoder->encode(slice_of(batch, g, g + 1, {})));
        EXPECT_EQ(a.at("static_task_ids"), std::vector<i64>{batch.task_ids[g]});
    }
    const auto empty = encoder->encode(slice_of(batch, 3, 3, {}));
    EXPECT_NO_THROW(validate(*empty, encoder->schema()));
    EXPECT_TRUE(read(*empty).at("static_task_ids").empty());
    EXPECT_EQ(read(*empty).at("static_offsets"), std::vector<i64>{0});
}

TEST(SharedStatics, TheBlocksAreInInstanceOrderWhateverTheRowOrder)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0", "smedium"});
    em::Batch batch = em::make_batch(table, 90, true, 5);
    const auto encoder = FlatEncoder::create(table, shared_of(test::encoder_options()));
    const std::set<i32> present(batch.task_ids.begin(), batch.task_ids.end());
    const Arrays a = read(*encoder->encode(em::host_input(batch, batch.count, {})));
    EXPECT_EQ(a.at("static_task_ids"), std::vector<i64>(present.begin(), present.end()));
    // reversing the rows does not change the blocks
    std::ranges::reverse(batch.task_ids);
    std::vector<u64> reversed(batch.states.size());
    for (u64 g = 0; g < batch.count; ++g)
        std::copy_n(batch.states.begin() + static_cast<std::ptrdiff_t>((batch.count - 1 - g) * batch.words), batch.words,
                    reversed.begin() + static_cast<std::ptrdiff_t>(g * batch.words));
    batch.states = reversed;
    batch.derived.assign(batch.count * batch.derived_words, 0);
    const Arrays b = read(*encoder->encode(em::host_input(batch, batch.count, {})));
    for (const char* name : {"static_task_ids", "static_offsets", "static_arg_offsets", "static_args", "static_relations"})
        EXPECT_EQ(a.at(name), b.at(name)) << name;
}

TEST(SharedStatics, ASliceKeepsTheBlocksItsGraphsNeedUnchanged)
{
    const rl::TaskTablePtr table = em::table_of("features", {"p01", "p02", "p03", "p04"});
    const em::Batch batch = em::make_batch(table, 120, true, 17);
    for (const Shape& shape : k_shapes)
    {
        const auto encoder = FlatEncoder::create(table, shared_of(em::with(test::encoder_options(), shape.layout, shape.index)));
        const Arrays whole = read(*encoder->encode(em::host_input(batch, batch.count, {})));
        const u64 Rs = encoder->schema().static_relations().size();
        auto block = [&](const Arrays& a, i64 id)
        {
            const auto at = std::ranges::lower_bound(a.at("static_task_ids"), id) - a.at("static_task_ids").begin();
            std::vector<std::vector<i64>> out;
            for (u64 k = 0; k < Rs; ++k)
            {
                const u64 s = static_cast<u64>(at) * Rs + k;
                out.emplace_back(a.at("static_args").begin() + a.at("static_arg_offsets")[s],
                                 a.at("static_args").begin() + a.at("static_arg_offsets")[s + 1]);
            }
            return out;
        };
        for (const auto& [first, last] : std::vector<std::pair<u64, u64>>{{0, 1}, {5, 40}, {37, 120}, {0, 120}, {64, 65}})
        {
            const Arrays part = read(*encoder->encode(slice_of(batch, first, last, {})));
            const std::set<i32> present(batch.task_ids.begin() + static_cast<std::ptrdiff_t>(first),
                                        batch.task_ids.begin() + static_cast<std::ptrdiff_t>(last));
            ASSERT_EQ(part.at("static_task_ids"), std::vector<i64>(present.begin(), present.end())) << shape.text();
            for (const i32 id : present)
                EXPECT_EQ(block(part, id), block(whole, id)) << shape.text() << " instance " << id;
            EXPECT_EQ(part.at("static_relations"), whole.at("static_relations"));
        }
    }
}

TEST(SharedStatics, ThePoolSizeDoesNotChangeTheBatch)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 5200, true, 23);
    for (const Shape& shape : k_shapes)
    {
        const auto encoder = FlatEncoder::create(table, shared_of(em::with(test::encoder_options(), shape.layout, shape.index)));
        const EncodeInput in = em::host_input(batch, batch.count, {});
        const Arrays expected = read(*encoder->encode(in));
        EncodeScratch scratch;  // one scratch for all calls: the state of an earlier call must not matter
        for (const u32 threads : {1u, 2u, 3u, 8u})
        {
            ThreadPool pool(threads);
            expect_equal(expected, read(*encoder->encode(in, pool, scratch)), shape.text() + " threads " + std::to_string(threads));
            EncodeInput part = slice_of(batch, 100, 3000, {});
            const Arrays once = read(*encoder->encode(part));
            expect_equal(once, read(*encoder->encode(part, pool, scratch)), shape.text() + " slice");
        }
    }
}

TEST(SharedStatics, TheCapacityFormPadsTheBlocksAndReportsTheirSizes)
{
    const rl::TaskTablePtr table = em::table_of("logistics00", {"probLOGISTICS-6-1"});
    const em::Batch batch = em::make_batch(table, 30, true, 9);
    for (const Shape& shape : k_shapes)
    {
        const FlatOptions options = em::with(test::encoder_options(), shape.layout, shape.index);
        const auto encoder = FlatEncoder::create(table, shared_of(options));
        const EncodeInput in = em::host_input(batch, batch.count, {});
        ThreadPool pool(1);
        EncodeScratch scratch;
        const Arrays exact = read(*encoder->encode(in, pool, scratch));
        ExtraSizes true_extra;
        const FlatSizes true_sizes = encoder->count(in, pool, scratch, &true_extra);
        ASSERT_EQ(true_extra.statics.instances, 1u);
        ASSERT_GT(true_extra.statics.args, 1u);
        // static capacities that are too small overflow, and an overflow leaves every array zero
        const StaticShape one{1, 1};
        ExtraSizes extra;
        FlatSizes sizes;
        const auto probe = encoder->encode_capacity(in, {30, true_sizes.entities, true_sizes.args, 0}, pool, scratch,
                                                    sizes, &one, &extra);
        EXPECT_NE(sizes.overflow & k_overflow_capacity, 0u);
        EXPECT_EQ(extra.statics.args, true_extra.statics.args);
        for (const rl::ArrayInfo& info : probe->arrays())
            for (const i64 v : values_of(*probe, info))
                EXPECT_EQ(v, 0) << info.name;
        const BatchShape caps{35, true_sizes.entities + 7, true_sizes.args + 5, 0};
        const StaticShape static_caps{3, true_extra.statics.args + 9};
        const auto padded = encoder->encode_capacity(in, caps, pool, scratch, sizes, &static_caps, &extra);
        EXPECT_FALSE(sizes.overflowed());
        EXPECT_EQ(extra.statics.instances, true_extra.statics.instances);
        EXPECT_EQ(extra.statics.args, true_extra.statics.args);
        const Arrays a = read(*padded);
        const u64 Rs = encoder->schema().static_relations().size();
        EXPECT_EQ(a.at("static_task_ids").size(), 3u);
        EXPECT_EQ(a.at("static_offsets").size(), 3 * Rs + 1);
        EXPECT_EQ(a.at("static_arg_offsets").size(), 3 * Rs + 1);
        EXPECT_EQ(a.at("static_args").size(), true_extra.statics.args + 9);
        // the first block's entries are the exact ones, the padding is empty blocks and zeros
        EXPECT_TRUE(std::equal(exact.at("static_args").begin(), exact.at("static_args").end(), a.at("static_args").begin()));
        EXPECT_TRUE(std::all_of(a.at("static_args").begin() + static_cast<std::ptrdiff_t>(true_extra.statics.args), a.at("static_args").end(), [](i64 v) { return v == 0; }));
        EXPECT_EQ(a.at("static_task_ids")[0], exact.at("static_task_ids")[0]);
        EXPECT_EQ(a.at("static_task_ids")[1], 0);
        for (u64 i = Rs; i <= 3 * Rs; ++i)
        {
            EXPECT_EQ(a.at("static_offsets")[i], exact.at("static_offsets")[Rs]);
            EXPECT_EQ(a.at("static_arg_offsets")[i], exact.at("static_arg_offsets")[Rs]);
        }
        EXPECT_NO_THROW(validate(*padded, encoder->schema()));
        // capacities one short in either static dimension overflow
        const StaticShape few_args{3, true_extra.statics.args - 1};
        FlatSizes short_sizes;
        (void)encoder->encode_capacity(in, caps, pool, scratch, short_sizes, &few_args, &extra);
        EXPECT_NE(short_sizes.overflow & k_overflow_capacity, 0u);
        EXPECT_EQ(extra.statics.args, true_extra.statics.args);
        // the capacity form needs static capacities exactly when the statics are shared
        EXPECT_THROW((void)encoder->encode_capacity(in, caps, pool, scratch, sizes), std::invalid_argument);
        const auto repeated = FlatEncoder::create(table, options);
        EXPECT_THROW((void)repeated->encode_capacity(in, caps, pool, scratch, sizes, &static_caps, &extra),
                     std::invalid_argument);
    }
}

TEST(SharedStatics, StreamsFlushTheSharedForm)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 60, true, 4);
    const auto encoder = FlatEncoder::create(table, shared_of(test::encoder_options()));
    ThreadPool pool(2);
    EncodeScratch scratch;
    const Arrays expected = read(*encoder->encode(em::host_input(batch, batch.count, {}), pool, scratch));
    FlatStream stream(encoder);
    stream.append(slice_of(batch, 0, 25, {}));
    stream.append(slice_of(batch, 25, 60, {}));
    expect_equal(expected, read(*stream.flush(pool)), "append-only stream");
    FlatMutableStream mutable_stream(encoder);
    std::vector<i64> indices(batch.count);
    for (u64 g = 0; g < batch.count; ++g)
        indices[g] = static_cast<i64>(g);
    mutable_stream.append(indices, em::host_input(batch, batch.count, {}));
    expect_equal(expected, read(*mutable_stream.flush(pool)), "mutable stream");
    // a capacity flush
    const FlatSizes want = encoder->count(em::host_input(batch, batch.count, {}), pool, scratch);
    ExtraSizes want_extra;
    (void)encoder->count(em::host_input(batch, batch.count, {}), pool, scratch, &want_extra);
    FlatSizes sizes;
    ExtraSizes extra;
    const StaticShape static_caps{4, want_extra.statics.args + 3};
    stream.append(em::host_input(batch, batch.count, {}));
    const auto flushed = stream.flush(pool, {70, want.entities + 2, want.args + 2, 0}, sizes, &static_caps, &extra);
    EXPECT_FALSE(sizes.overflowed());
    EXPECT_EQ(extra.statics.instances, 2u);
    EXPECT_EQ(extra.statics.args, want_extra.statics.args);
    EXPECT_NO_THROW(validate(*flushed, encoder->schema()));
    // an overflowing flush keeps the rows
    stream.append(em::host_input(batch, batch.count, {}));
    const StaticShape small{1, 1};
    (void)stream.flush(pool, {70, want.entities, want.args, 0}, sizes, &small, &extra);
    EXPECT_TRUE(sizes.overflowed());
    EXPECT_EQ(stream.size(), batch.count);
}

// ------------------------------------------------------------------------------------------------ actions

struct Expanded
{
    rl::Expansion x;
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
};

std::shared_ptr<Expanded> expand_rows(const rl::TaskTable& table, const test::Rows& rows)
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
        rl::expand(table, rows.view(), rows.task_ids.data(), e->x, {});
        if (!e->x.overflow())
            return e;
        capacity = std::max(capacity * 2, e->x.total);
    }
}

TEST(SharedStatics, ActionsAndTargetsKeepTheirReferences)
{
    for (const auto& [domain, problem] : std::vector<std::pair<const char*, const char*>>{
             {"blocks", "smedium"}, {"gripper", "p01"}, {"logistics00", "probLOGISTICS-6-1"}, {"relay", "p01"}})
    {
        const rl::TaskTablePtr table = rl::TaskTable::single(test::pddl_task(domain, problem));
        const test::Rows rows = test::random_walk(*table, 0, 40, 6);
        const auto expanded = expand_rows(*table, rows);
        EncodeInput in;
        in.states = rows.view();
        in.task_ids = rows.task_ids.data();
        in.actions = &expanded->x;
        for (const bool virtual_nodes : {false, true})
            for (const Shape& shape : k_shapes)
            {
                FlatOptions options = em::with(test::encoder_options(), shape.layout, shape.index);
                options.targets = {TargetSource::Action, TargetSource::Goal};
                options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied};
                options.predicate_virtual_nodes = virtual_nodes;
                expect_same_as_repeated(table, options, in,
                                        std::string(domain) + (virtual_nodes ? "/virtual/" : "/") + shape.text());
            }
    }
}

// ------------------------------------------------------------------------------------------------ nullary statics

/// A domain with a nullary static atom (powered) beside a nullary fluent (lit), and one whose only nullary atom is
/// static (ready): in the shared form the auxiliary entity of a graph can come from the static block alone.
constexpr const char* k_lamp_domain = R"((define (domain lamp) (:requirements :strips)
  (:predicates (powered) (lit) (at ?x) (link ?x ?y))
  (:action go :parameters (?x ?y) :precondition (and (at ?x) (link ?x ?y)) :effect (and (at ?y) (not (at ?x))))
  (:action on :parameters () :precondition (powered) :effect (lit))
  (:action off :parameters () :precondition (lit) :effect (not (lit)))))";
constexpr const char* k_lamp_problem = R"((define (problem p) (:domain lamp) (:objects a b c)
  (:init (powered) (at a) (link a b) (link b c) (link c a))
  (:goal (at c))))";
constexpr const char* k_ready_domain = R"((define (domain ready) (:requirements :strips)
  (:predicates (ready) (at ?x) (link ?x ?y))
  (:action go :parameters (?x ?y) :precondition (and (ready) (at ?x) (link ?x ?y)) :effect (and (at ?y) (not (at ?x))))))";
constexpr const char* k_ready_problem = R"((define (problem p) (:domain ready) (:objects a b c)
  (:init (ready) (at a) (link a b) (link b c))
  (:goal (at c))))";
// the instance without its nullary atom, with another number of objects
constexpr const char* k_unready_problem = R"((define (problem q) (:domain ready) (:objects a b c d)
  (:init (at a) (link a b) (link b c) (link c d))
  (:goal (at d))))";

rl::TaskTablePtr string_table(const char* domain, const std::vector<const char*>& problems)
{
    const auto d = frontend::Domain::from_string(domain);
    std::vector<TaskPtr> tasks;
    for (const char* problem : problems)
        tasks.push_back(Task::create(*d->instantiate_string(problem)));
    return rl::TaskTable::create(std::move(tasks));
}

TEST(SharedStatics, ANullaryStaticAtomGivesEveryGraphOfItsInstanceTheAuxiliaryEntity)
{
    const std::pair<const char*, rl::TaskTablePtr> tables[] = {
        {"lamp", string_table(k_lamp_domain, {k_lamp_problem})},
        {"ready", string_table(k_ready_domain, {k_ready_problem})},
        {"ready_unready", string_table(k_ready_domain, {k_unready_problem, k_ready_problem})}};
    for (const auto& [name, table] : tables)
    {
        const em::Batch batch = em::make_batch(table, 48, true, 17);
        for (const u64 rows : {u64{1}, batch.count})
            for (const em::Config& config : em::configs())
                for (const Shape& shape : k_shapes)
                    expect_same_as_repeated(table, em::with(config.options, shape.layout, shape.index),
                                            em::host_input(batch, rows, config.inputs),
                                            std::string(name) + "/" + std::to_string(rows) + "/" + config.name + "/" +
                                                shape.text());
    }
    // the first row of the single instance holds no fluent nullary atom: the static block alone needs the entity
    const rl::TaskTablePtr table = string_table(k_ready_domain, {k_ready_problem});
    const em::Batch batch = em::make_batch(table, 1, false, 1);
    const auto encoder = FlatEncoder::create(table, shared_of(test::encoder_options()));
    const Arrays a = read(*encoder->encode(em::host_input(batch, 1, {})));
    EXPECT_EQ(a.at("entity_offsets"), (std::vector<i64>{0, 4}));
}

// ------------------------------------------------------------------------------------------------ layers and history

/// Random subgoal layers or history steps for the rows of `batch`: per row and layer a random subset of the instance's
/// fluent and derived slots, as positive and as negative masks.
struct RandomLayers
{
    std::vector<u64> pos, neg, derived_pos, derived_neg;  // [rows, layers, words]
    u32 layers = 0, words = 0, derived_words = 0;
    std::vector<i32> dt, count;  // [rows, layers], [rows]: the history's dt and valid steps

    RandomLayers(const em::Batch& batch, u32 layer_count, u64 seed)
    {
        layers = layer_count;
        words = batch.words;
        derived_words = std::max(1u, batch.derived_words);
        pos.assign(batch.count * layers * words, 0);
        neg = pos;
        derived_pos.assign(batch.count * layers * derived_words, 0);
        derived_neg = derived_pos;
        dt.assign(batch.count * layers, 0);
        std::mt19937_64 rng(seed);
        for (u64 g = 0; g < batch.count; ++g)
        {
            const Task& task = *batch.table->task(static_cast<u32>(batch.task_ids[g]));
            count.push_back(static_cast<i32>(rng() % (layers + 1)));
            for (u32 l = 0; l < layers; ++l)
            {
                dt[g * layers + l] = static_cast<i32>(rng() % 7) - 6;
                if (rng() % 4 == 0)
                    continue;  // an empty layer
                const double density = std::array{0.02, 0.1, 0.3}[rng() % 3];
                auto fill = [&](std::vector<u64>& positive, std::vector<u64>& negative, u32 slots, u32 width)
                {
                    for (u32 slot = 0; slot < slots; ++slot)
                        if (std::uniform_real_distribution<double>(0, 1)(rng) < density)
                            (rng() & 1 ? positive : negative)[(g * layers + l) * width + (slot >> 6)] |=
                                u64{1} << (slot & 63);
                };
                fill(pos, neg, task.atoms().fluent_slots(), words);
                fill(derived_pos, derived_neg, task.atoms().derived_slots(), derived_words);
            }
        }
    }

    [[nodiscard]] LayerInput masks() const
    {
        LayerInput in;
        in.pos = pos.data();
        in.neg = neg.data();
        in.layers = layers;
        in.words = words;
        in.stride = u64{layers} * words;
        in.derived_pos = derived_pos.data();
        in.derived_neg = derived_neg.data();
        in.derived_words = derived_words;
        in.derived_stride = u64{layers} * derived_words;
        return in;
    }
    [[nodiscard]] HistoryInput history() const
    {
        HistoryInput in;
        in.masks = masks();
        in.dt = dt.data();
        in.count = count.data();
        return in;
    }
};

struct LayeredConfig
{
    const char* name;
    u32 levels;
    bool history;
    std::optional<u32> max_steps;
    bool virtual_nodes;
    bool actions;
};

FlatOptions layered_options(const LayeredConfig& config)
{
    FlatOptions options = test::encoder_options();
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    options.max_goal_level = config.levels;
    options.targets = {TargetSource::Goal};
    if (config.levels > 0)
        options.targets.insert(TargetSource::Subgoal);
    options.history = config.history;
    options.history_max_steps = config.max_steps;
    if (config.history)
        options.targets.insert(TargetSource::History);
    options.predicate_virtual_nodes = config.virtual_nodes;
    options.include_actions = config.actions;
    if (config.actions)
        options.targets.insert(TargetSource::Action);
    return options;
}

const LayeredConfig k_layered[] = {{"subgoals", 2, false, std::nullopt, false, false},
                                   {"history", 0, true, std::nullopt, false, false},
                                   {"history_newest_two", 0, true, 2, true, false},
                                   {"subgoals_history_virtual", 3, true, std::nullopt, true, false},
                                   {"subgoals_history_actions_virtual", 2, true, 3, true, true}};

TEST(SharedStatics, SubgoalLayersAndHistoryStepsRebuildTheRepeatedFormOverTheCorpus)
{
    for (const em::Corpus& corpus : em::corpus())
    {
        const em::Batch batch = em::make_batch(corpus.table, 70, corpus.walk, 13);
        const RandomLayers history(batch, 3, 21);
        test::Rows rows;
        rows.words = batch.words;
        rows.data = batch.states;
        rows.task_ids = batch.task_ids;
        // the wide instances have thousands of actions per state
        const auto expanded = std::string(corpus.name) == "wide" ? nullptr : expand_rows(*corpus.table, rows);
        for (const LayeredConfig& config : k_layered)
            for (const Shape& shape : k_shapes)
            {
                if (config.actions && !expanded)
                    continue;
                const RandomLayers subgoals(batch, std::max(config.levels, 1u), 34);
                EncodeInput in = em::host_input(batch, batch.count, {});
                if (config.levels > 0)
                    in.subgoals = subgoals.masks();
                if (config.history)
                    in.history = history.history();
                if (config.actions)
                    in.actions = &expanded->x;
                const FlatOptions options = em::with(layered_options(config), shape.layout, shape.index);
                expect_same_as_repeated(corpus.table, options, in,
                                        std::string(corpus.name) + "/" + config.name + "/" + shape.text());
            }
    }
}

TEST(SharedStatics, SubgoalAndHistoryTargetsAreInTheBatchesTheRebuildCompares)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 70, true, 13);
    const RandomLayers subgoals(batch, 3, 34), history(batch, 3, 21);
    EncodeInput in = em::host_input(batch, batch.count, {});
    in.subgoals = subgoals.masks();
    in.history = history.history();
    const auto encoder = FlatEncoder::create(table, shared_of(layered_options(k_layered[3])));
    const Arrays arrays = read(*encoder->encode(in));
    std::map<i64, u64> per_role;
    for (const i64 role : arrays.at("entity_role"))
        ++per_role[role];
    EXPECT_GT(per_role[3], 0u);
    EXPECT_GT(per_role[5], 0u);
    EXPECT_GT(per_role[6], 0u);
}

TEST(SharedStatics, ACapacityFlushWithHistoryKeepsBothKindsOfBlocks)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 40, true, 5);
    const RandomLayers layers(batch, 3, 8);
    EncodeInput in = em::host_input(batch, batch.count, {});
    in.subgoals = layers.masks();
    in.history = layers.history();
    const auto encoder = FlatEncoder::create(table, shared_of(layered_options(k_layered[3])));
    ThreadPool pool(2);
    EncodeScratch scratch;
    ExtraSizes want_extra;
    const FlatSizes want = encoder->count(in, pool, scratch, &want_extra);
    ASSERT_GT(want.history, 0u);
    const BatchShape capacities{50, want.entities + 3, want.args + 5, want.targets + 2, want.history + 4};
    const StaticShape static_capacities{3, want_extra.statics.args + 2};
    FlatSizes sizes;
    ExtraSizes extra;
    const auto padded = encoder->encode_capacity(in, capacities, pool, scratch, sizes, &static_capacities, &extra);
    EXPECT_FALSE(sizes.overflowed());
    EXPECT_EQ(sizes.history, want.history);
    EXPECT_EQ(extra.statics.instances, 2u);
    EXPECT_EQ(extra.statics.args, want_extra.statics.args);
    EXPECT_NO_THROW(validate(*padded, encoder->schema()));
    // the batch equals the exact one in the first entries
    const Arrays exact = read(*encoder->encode(in, pool, scratch));
    const Arrays got = read(*padded);
    for (const char* name : {"history_entity", "history_dt", "static_args", "static_task_ids"})
        EXPECT_TRUE(std::equal(exact.at(name).begin(), exact.at(name).end(), got.at(name).begin())) << name;
    // a history that does not fit its capacity overflows, and a static block that does not fit does too
    const BatchShape tight{50, want.entities, want.args, want.targets, want.history - 1};
    (void)encoder->encode_capacity(in, tight, pool, scratch, sizes, &static_capacities, &extra);
    EXPECT_TRUE(sizes.overflowed());
    const StaticShape small{1, 1};
    (void)encoder->encode_capacity(in, capacities, pool, scratch, sizes, &small, &extra);
    EXPECT_TRUE(sizes.overflowed());
}

TEST(SharedStatics, AStreamWithSubgoalLayersAndHistoryFlushesTheBatch)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 36, true, 6);
    const RandomLayers layers(batch, 2, 9);
    EncodeInput in = em::host_input(batch, batch.count, {});
    in.subgoals = layers.masks();
    in.history = layers.history();
    const auto encoder = FlatEncoder::create(table, shared_of(layered_options(k_layered[3])));
    ThreadPool pool(1);
    EncodeScratch scratch;
    const Arrays expected = read(*encoder->encode(in, pool, scratch));
    FlatStream stream(encoder);
    stream.append(in);
    expect_equal(expected, read(*stream.flush(pool)), "append-only stream");
    FlatMutableStream mutable_stream(encoder);
    std::vector<i64> indices(batch.count);
    for (u64 g = 0; g < batch.count; ++g)
        indices[g] = static_cast<i64>(g);
    mutable_stream.append(indices, in);
    expect_equal(expected, read(*mutable_stream.flush(pool)), "mutable stream");
}

// ------------------------------------------------------------------------------------------------ validation

/// A copy of `bundle` whose scalar `name` is `value`.
rl::ArrayBundle with_scalar(const rl::ArrayBundle& bundle, const char* name, i64 value)
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
    return out.build();
}

TEST(SharedStatics, ValidateRefusesDamagedBlocks)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium", "probBLOCKS-8-0"});
    const em::Batch batch = em::make_batch(table, 30, true, 8);
    FlatOptions options = test::encoder_options();
    options.predicate_virtual_nodes = true;
    const auto encoder = FlatEncoder::create(table, shared_of(options));
    const auto good = encoder->encode(em::host_input(batch, batch.count, {}));
    ASSERT_NO_THROW(validate(*good, encoder->schema()));
    const Arrays a = read(*good);
    ASSERT_GT(a.at("static_args").size(), 4u);
    EXPECT_THROW(validate(test::with_change(*good, "static_args", 1, 1), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*good, "static_args", 0, 5000), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*good, "static_task_ids", 0, 1), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*good, "static_offsets", 1, 1), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*good, "static_arg_offsets", 1, -1), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(test::with_change(*good, "static_relations", 0, 1), encoder->schema()), std::invalid_argument);
    EXPECT_THROW(validate(with_scalar(*good, "statics", 0), encoder->schema()), std::invalid_argument);
    // a static tuple in a graph's cell, and a shared batch against the schema that repeats
    const auto repeated = FlatEncoder::create(table, options);
    EXPECT_THROW(validate(*good, repeated->schema()), std::invalid_argument);
    EXPECT_THROW(validate(*repeated->encode(em::host_input(batch, batch.count, {})), encoder->schema()), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ options

TEST(SharedStatics, TheOptionIsPartOfTheFingerprintOnlyWhenShared)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium"});
    const FlatOptions options = test::encoder_options();
    const auto repeat = EncodingSchema::create(*table, options);
    const auto shared = EncodingSchema::create(*table, shared_of(options));
    EXPECT_NE(repeat->fingerprint(), shared->fingerprint());
    FlatOptions explicit_repeat = options;
    explicit_repeat.statics = Statics::Repeat;
    EXPECT_EQ(EncodingSchema::create(*table, explicit_repeat)->fingerprint(), repeat->fingerprint());
    EXPECT_FALSE(repeat->shared_statics());
    EXPECT_TRUE(shared->shared_statics());
    EXPECT_EQ(repeat->static_relations().size(), shared->static_relations().size());
    EXPECT_GT(shared->static_relations().size(), 0u);
    for (const u32 id : shared->static_relations())
        EXPECT_EQ(shared->relation(id).lane, Lane::Static);
    EXPECT_EQ(parse_statics("shared"), Statics::Shared);
    EXPECT_EQ(parse_statics("repeat"), Statics::Repeat);
    EXPECT_FALSE(parse_statics("once"));
    EXPECT_EQ(to_string(Statics::Shared), "shared");
}

TEST(SharedStatics, FamiliesThatNeedPerGraphStaticsRefuse)
{
    FlatOptions options = test::encoder_options();
    EXPECT_NO_THROW(require_repeated_statics(options, "Simple"));
    options.statics = Statics::Shared;
    for (const char* family : {"Simple", "HGraph"})
        try
        {
            require_repeated_statics(options, family);
            ADD_FAILURE() << family;
        }
        catch (const std::invalid_argument& e)
        {
            const std::string message = e.what();
            EXPECT_EQ(message.rfind("mymyr: encoding: statics='shared' is not supported by the ", 0), 0u) << message;
            EXPECT_NE(message.find(family), std::string::npos) << message;
        }
}

TEST(SharedStatics, HorizonSchemasRefuseSharedStatics)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium"});
    for (const HorizonMode mode : {HorizonMode::Full, HorizonMode::Delta, HorizonMode::Action})
        for (const RootPolicy root : {RootPolicy::Include, RootPolicy::Exclude, RootPolicy::EncodeOnly})
        {
            FlatOptions options = test::encoder_options();
            options.horizon.mode = mode;
            options.horizon.root = root;
            options.horizon.parent_relation = true;
            EXPECT_NO_THROW((void)EncodingSchema::create(*table, options));
            options.statics = Statics::Shared;
            try
            {
                (void)EncodingSchema::create(*table, options);
                ADD_FAILURE() << to_string(mode) << "/" << to_string(root);
            }
            catch (const std::invalid_argument& e)
            {
                EXPECT_NE(std::string(e.what()).find("statics='shared' cannot be combined with a horizon mode"),
                          std::string::npos)
                    << e.what();
            }
        }
}

TEST(SharedStatics, AnEncoderRefusesBuffersOfTheOtherForm)
{
    const rl::TaskTablePtr table = em::table_of("blocks", {"smedium"});
    const em::Batch batch = em::make_batch(table, 10, true, 2);
    const EncodeInput in = em::host_input(batch, batch.count, {});
    ThreadPool pool(1);
    EncodeScratch scratch;
    const auto repeated = FlatEncoder::create(table, test::encoder_options());
    const FlatSizes sizes = repeated->count(in, pool, scratch);
    std::vector<i32> task_ids(10), numbers(1000);
    FlatBuffers out;
    out.task_ids = task_ids.data();
    out.static_offsets = numbers.data();  // a repeated schema has no static blocks
    EXPECT_THROW(repeated->fill(in, sizes, out, pool, scratch), std::invalid_argument);
}
}  // namespace
