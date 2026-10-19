// The layouts, thread counts, capacity form and omitted arrays of the relational encoder.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
struct Problem
{
    const char* domain;
    const char* problem;
};
const std::vector<Problem> k_problems = {
    {"blocks", "smedium"},
    {"blocks", "probBLOCKS-8-0"},
    {"gripper", "p01"},
    {"logistics00", "probLOGISTICS-6-1"},
    {"philosophers", "p03-phil4"},
    {"features", "p01"},
    {"blocks_flags", "p01"},
    {"relay", "p01"}};

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

void expect_same_bytes(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    ASSERT_EQ(a.bytes(), b.bytes());
    EXPECT_EQ(std::memcmp(a.block(), b.block(), a.bytes()), 0);
    EXPECT_EQ(a.scalars(), b.scalars());
}

/// Rows of a table of the given problems, interleaving the instances.
test::Rows mixed_rows(const rl::TaskTable& table, u64 per_instance, u64 seed)
{
    std::vector<test::Rows> sources;
    for (u32 i = 0; i < table.size(); ++i)
        sources.push_back(test::random_walk(table, i, per_instance, seed + i));
    test::Rows mixed;
    mixed.words = table.words();
    const u32 row_words = table.row_words();
    for (u64 k = 0; k < per_instance; ++k)
        for (const test::Rows& source : sources)
        {
            mixed.data.insert(mixed.data.end(), source.data.begin() + static_cast<std::ptrdiff_t>(k * row_words),
                              source.data.begin() + static_cast<std::ptrdiff_t>((k + 1) * row_words));
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

FlatOptions options_with(Layout layout, IndexType index = IndexType::I32)
{
    FlatOptions options = test::encoder_options();
    options.layout = layout;
    options.index = index;
    return options;
}

rl::TaskTablePtr two_instance_table()
{
    return rl::TaskTable::create({test::pddl_task("blocks", "smedium"), test::pddl_task("blocks", "probBLOCKS-8-0")});
}

TEST(EncoderLayout, RelationMajorIsThePermutationOfGraphMajor)
{
    for (const Problem& p : k_problems)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const test::Rows rows = test::random_walk(*table, 0, 64, 5);
        const auto graph_major = FlatEncoder::create(table, options_with(Layout::Graph))->encode(input_of(rows));
        const auto relation_encoder = FlatEncoder::create(table, options_with(Layout::Relation));
        const auto relation_major = relation_encoder->encode(input_of(rows));
        validate(*relation_major, relation_encoder->schema());
        EXPECT_EQ(relation_major->scalar("layout"), 1);
        EXPECT_EQ(graph_major->scalar("layout"), 0);
        for (const char* name :
             {"task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref", "relation_counts"})
            EXPECT_EQ(ints(*graph_major, name), ints(*relation_major, name)) << name;

        const u64 G = rows.size();
        const u64 R = relation_encoder->schema().num_relations();
        const auto counts = ints(*graph_major, "relation_counts");
        const auto graph_args = ints(*graph_major, "relation_args");
        const auto graph_arg_offsets = ints(*graph_major, "relation_arg_offsets");
        const auto relation_offsets = ints(*relation_major, "relation_offsets");
        const auto relation_arg_offsets = ints(*relation_major, "relation_arg_offsets");
        std::vector<i64> permuted;
        i64 tuples = 0, args = 0;
        for (u64 r = 0; r < R; ++r)
            for (u64 g = 0; g < G; ++g)
            {
                // the cell at relation-major segment r * G + g is the cell at graph-major segment g * R + r
                const u64 from = g * R + r;
                const u64 to = r * G + g;
                permuted.insert(permuted.end(), graph_args.begin() + graph_arg_offsets[from],
                                graph_args.begin() + graph_arg_offsets[from + 1]);
                // the offsets are the exclusive prefix sums of the cells' sizes in segment order
                ASSERT_EQ(relation_offsets[to], tuples);
                ASSERT_EQ(relation_arg_offsets[to], args);
                tuples += counts[from];
                args += graph_arg_offsets[from + 1] - graph_arg_offsets[from];
            }
        EXPECT_EQ(relation_offsets[G * R], tuples);
        EXPECT_EQ(relation_arg_offsets[G * R], args);
        EXPECT_EQ(permuted, ints(*relation_major, "relation_args"));
    }
}

TEST(EncoderLayout, MixedInstancesAndBothIndexTypes)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 40, 3);
    for (const IndexType index : {IndexType::I32, IndexType::I64})
    {
        const auto graph = FlatEncoder::create(table, options_with(Layout::Graph, index));
        const auto relation = FlatEncoder::create(table, options_with(Layout::Relation, index));
        const auto a = graph->encode(input_of(rows));
        const auto b = relation->encode(input_of(rows));
        validate(*a, graph->schema());
        validate(*b, relation->schema());
        EXPECT_NE(graph->schema().fingerprint(), relation->schema().fingerprint());
        const u64 R = graph->schema().num_relations();
        const auto counts = ints(*b, "relation_counts");
        const auto offsets = ints(*b, "relation_offsets");
        for (u64 g = 0; g < rows.size(); ++g)
            for (u64 r = 0; r < R; ++r)
                ASSERT_EQ(offsets[r * rows.size() + g + 1] - offsets[r * rows.size() + g], counts[g * R + r]);
    }
}

TEST(EncoderThreads, EveryThreadCountGivesTheSameBytes)
{
    const auto mixed = two_instance_table();
    const auto derived = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    for (const rl::TaskTablePtr& table : {mixed, derived})
    {
        const u64 per_instance = 3 * k_graphs_per_member / 2 + 40;
        const test::Rows all = mixed_rows(*table, per_instance, 7);
        for (const Layout layout : {Layout::Graph, Layout::Relation})
            for (const IndexType index : {IndexType::I32, IndexType::I64})
            {
                SCOPED_TRACE(std::string(to_string(layout)) + " " + std::string(to_string(index)));
                const auto encoder = FlatEncoder::create(table, options_with(layout, index));
                for (const u64 n :
                     {u64{0}, u64{1}, u64{5}, k_graphs_per_member + 1, 2 * k_graphs_per_member + 7, all.size()})
                {
                    if (n > all.size())
                        continue;
                    test::Rows rows = all;
                    rows.data.resize(n * table->row_words());
                    rows.task_ids.resize(n);
                    const auto reference = encoder->encode(input_of(rows));
                    validate(*reference, encoder->schema());
                    for (const u32 threads : {1u, 2u, 3u, 8u, 64u})
                    {
                        ThreadPool pool(threads);
                        EncodeScratch scratch;
                        const auto got = encoder->encode(input_of(rows), pool, scratch);
                        expect_same_bytes(*reference, *got);
                        // a second call with the same scratch (reused buffers) gives the same bytes
                        expect_same_bytes(*reference, *encoder->encode(input_of(rows), pool, scratch));
                        if (HasFatalFailure())
                            return;
                    }
                }
            }
    }
}

TEST(EncoderThreads, ConcurrentCallsOnOneEncoderAgree)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 100, 9);
    const auto encoder = FlatEncoder::create(table, options_with(Layout::Relation));
    const auto reference = encoder->encode(input_of(rows));
    std::vector<std::shared_ptr<const rl::ArrayBundle>> got(4);
    std::vector<std::thread> callers;
    for (usize i = 0; i < got.size(); ++i)
        callers.emplace_back(
            [&, i]
            {
                ThreadPool pool(2);
                EncodeScratch scratch;
                got[i] = encoder->encode(input_of(rows), pool, scratch);
            });
    for (auto& caller : callers)
        caller.join();
    for (const auto& bundle : got)
        expect_same_bytes(*reference, *bundle);
}

/// A destination array of the capacity form: `capacity` entries of T between canary guards.
template<class T>
struct Guarded
{
    static constexpr usize k_guard = 16;
    static constexpr T k_canary = static_cast<T>(0x5a);
    std::vector<T> storage;
    usize capacity;
    explicit Guarded(usize n) : storage(n + 2 * k_guard, k_canary), capacity(n) {}
    T* data() { return storage.data() + k_guard; }
    [[nodiscard]] bool guards_intact() const
    {
        for (usize i = 0; i < k_guard; ++i)
            if (storage[i] != k_canary || storage[k_guard + capacity + i] != k_canary)
                return false;
        return true;
    }
    [[nodiscard]] bool untouched() const
    {
        return std::all_of(storage.begin(), storage.end(), [](T v) { return v == k_canary; });
    }
    [[nodiscard]] std::vector<i64> values() const
    {
        return {storage.begin() + k_guard, storage.begin() + k_guard + capacity};
    }
};

template<class I>
struct Destination
{
    Guarded<i32> task_ids;
    Guarded<I> num_objects, entity_offsets, relation_counts, relation_offsets, relation_arg_offsets, relation_args,
        entity_ref;
    Guarded<u8> entity_role;
    Destination(u64 G, u64 R, const FlatCapacities& cap)
        : task_ids(G), num_objects(G), entity_offsets(G + 1), relation_counts(G * R), relation_offsets(G * R + 1),
          relation_arg_offsets(G * R + 1), relation_args(cap.args), entity_ref(cap.entities), entity_role(cap.entities)
    {
    }
    [[nodiscard]] FlatBuffers buffers()
    {
        return {task_ids.data(),        num_objects.data(),       entity_offsets.data(),
                entity_role.data(),     entity_ref.data(),        relation_counts.data(),
                relation_offsets.data(), relation_arg_offsets.data(), relation_args.data()};
    }
    [[nodiscard]] bool guards_intact() const
    {
        return task_ids.guards_intact() && num_objects.guards_intact() && entity_offsets.guards_intact() &&
               relation_counts.guards_intact() && relation_offsets.guards_intact() &&
               relation_arg_offsets.guards_intact() &&
               relation_args.guards_intact() && entity_ref.guards_intact() && entity_role.guards_intact();
    }
    [[nodiscard]] bool untouched() const
    {
        return task_ids.untouched() && num_objects.untouched() && entity_offsets.untouched() &&
               relation_counts.untouched() &&
               relation_offsets.untouched() && relation_arg_offsets.untouched() && relation_args.untouched() &&
               entity_ref.untouched() && entity_role.untouched();
    }
};

template<class I>
void run_capacity_cases(const FlatOptions& options)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 30, 11);
    const auto encoder = FlatEncoder::create(table, options);
    const auto expected = encoder->encode(input_of(rows));
    const u64 G = rows.size();
    const u64 R = encoder->schema().num_relations();
    FlatCapacities exact;
    exact.entities = static_cast<u64>(ints(*expected, "entity_offsets").back());
    exact.tuples = static_cast<u64>(ints(*expected, "relation_offsets").back());
    exact.args = static_cast<u64>(ints(*expected, "relation_arg_offsets").back());
    ASSERT_GT(exact.entities, 0u);
    ASSERT_GT(exact.args, 0u);

    ThreadPool pool(3);
    EncodeScratch scratch;
    auto run = [&](const FlatCapacities& cap, bool expect_fit)
    {
        Destination<I> destination(G, R, cap);
        const FlatSizes sizes = encoder->encode_into(input_of(rows), cap, destination.buffers(), pool, scratch);
        // the sizes are true, whatever the capacities
        EXPECT_EQ(sizes.entities, exact.entities);
        EXPECT_EQ(sizes.tuples, exact.tuples);
        EXPECT_EQ(sizes.args, exact.args);
        EXPECT_TRUE(destination.guards_intact());
        if (expect_fit)
        {
            EXPECT_EQ(sizes.overflow, 0u);
            for (const auto& [name, got] : std::vector<std::pair<const char*, std::vector<i64>>>{
                     {"task_ids", destination.task_ids.values()},
                     {"num_objects", destination.num_objects.values()},
                     {"entity_offsets", destination.entity_offsets.values()},
                     {"entity_role", destination.entity_role.values()},
                     {"entity_ref", destination.entity_ref.values()},
                     {"relation_counts", destination.relation_counts.values()},
                     {"relation_offsets", destination.relation_offsets.values()},
                     {"relation_arg_offsets", destination.relation_arg_offsets.values()},
                     {"relation_args", destination.relation_args.values()}})
            {
                // a buffer larger than the batch keeps the canary past the batch
                const std::vector<i64> want = ints(*expected, name);
                const auto past = got.begin() + static_cast<std::ptrdiff_t>(want.size());
                EXPECT_EQ(std::vector<i64>(got.begin(), past), want) << name;
                EXPECT_TRUE(std::all_of(past, got.end(), [](i64 v) { return v == 0x5a; })) << name;
            }
        }
        else
        {
            EXPECT_TRUE(sizes.overflow & k_overflow_capacity);
            EXPECT_FALSE(sizes.overflow & k_overflow_index);
            EXPECT_TRUE(destination.untouched());
        }
    };
    run(exact, true);
    run({exact.entities + 100, exact.tuples + 100, exact.args + 100}, true);
    run({exact.entities - 1, exact.tuples, exact.args}, false);
    run({exact.entities, exact.tuples - 1, exact.args}, false);
    run({exact.entities, exact.tuples, exact.args - 1}, false);
    run({0, 0, 0}, false);
    run({exact.entities, exact.tuples, 0}, false);

    // count() and fill() separately; fill refuses another input
    const FlatSizes sizes = encoder->count(input_of(rows), pool, scratch);
    EXPECT_EQ(sizes.overflow, 0u);
    Destination<I> destination(G, R, exact);
    encoder->fill(input_of(rows), sizes, destination.buffers(), pool, scratch);
    EXPECT_TRUE(destination.guards_intact());
    EXPECT_EQ(destination.relation_args.values(), ints(*expected, "relation_args"));
    test::Rows fewer = rows;
    fewer.data.resize(10 * table->row_words());
    fewer.task_ids.resize(10);
    EXPECT_THROW(encoder->fill(input_of(fewer), sizes, destination.buffers(), pool, scratch), std::invalid_argument);
    FlatSizes overflowing = sizes;
    overflowing.overflow = k_overflow_capacity;
    EXPECT_THROW(encoder->fill(input_of(rows), overflowing, destination.buffers(), pool, scratch),
                 std::invalid_argument);
    // omitted arrays must be null, stored ones not
    FlatBuffers wrong = destination.buffers();
    wrong.relation_counts = nullptr;
    EXPECT_THROW(encoder->fill(input_of(rows), sizes, wrong, pool, scratch), std::invalid_argument);
    wrong = destination.buffers();
    wrong.relation_args = nullptr;
    EXPECT_THROW(encoder->fill(input_of(rows), sizes, wrong, pool, scratch), std::invalid_argument);
}

TEST(EncoderCapacity, FillNeedsAPoolAsLargeAsTheCountUsed)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 2 * k_graphs_per_member, 13);
    const auto encoder = FlatEncoder::create(table, options_with(Layout::Relation));
    ThreadPool pool(4);
    EncodeScratch scratch;
    const FlatSizes sizes = encoder->count(input_of(rows), pool, scratch);
    Destination<i32> destination(rows.size(), encoder->schema().num_relations(),
                                 {sizes.entities, sizes.tuples, sizes.args});
    ThreadPool one(1);
    EXPECT_THROW(encoder->fill(input_of(rows), sizes, destination.buffers(), one, scratch), std::invalid_argument);
    encoder->fill(input_of(rows), sizes, destination.buffers(), pool, scratch);
    EXPECT_EQ(destination.relation_args.values(), ints(*encoder->encode(input_of(rows)), "relation_args"));
}

TEST(EncoderThreads, ASmallBatchRunsOnTheCallingThread)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, k_graphs_per_member / 2, 17);
    ASSERT_LE(rows.size(), k_graphs_per_member);
    const auto encoder = FlatEncoder::create(table, options_with(Layout::Relation));
    const auto expected = encoder->encode(input_of(rows));
    ThreadPool pool(4);
    // keep the pool busy: a call that needs it would wait for the job to end (at its time-out)
    std::atomic<bool> started{false}, encoded{false}, timed_out{false};
    std::thread holder(
        [&]
        {
            pool.run(
                [&](u32 member)
                {
                    if (member != 0)
                        return;
                    started = true;
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (!encoded && std::chrono::steady_clock::now() < deadline)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    timed_out = !encoded;
                });
        });
    while (!started)
        std::this_thread::yield();
    EncodeScratch scratch;
    const auto got = encoder->encode(input_of(rows), pool, scratch);
    encoded = true;
    holder.join();
    EXPECT_FALSE(timed_out);
    expect_same_bytes(*expected, *got);
}

TEST(EncoderCapacity, ExactShortAndZeroCapacitiesWriteNothingPastThem)
{
    test::dirty_recycled_blocks();
    run_capacity_cases<i32>(options_with(Layout::Graph));
    run_capacity_cases<i32>(options_with(Layout::Relation));
    run_capacity_cases<i64>(options_with(Layout::Relation, IndexType::I64));
}

TEST(EncoderCapacity, ABundleAtCapacityHoldsTheBatchThenZeros)
{
    test::dirty_recycled_blocks();
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 20, 6);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        const auto encoder = FlatEncoder::create(table, options_with(layout));
        const auto expected = encoder->encode(input_of(rows));
        const FlatCapacities exact{static_cast<u64>(ints(*expected, "entity_offsets").back()),
                                   static_cast<u64>(ints(*expected, "relation_offsets").back()),
                                   static_cast<u64>(ints(*expected, "relation_arg_offsets").back())};
        ThreadPool pool(2);
        EncodeScratch scratch;
        FlatSizes sizes;
        const auto same = encoder->encode_capacity(input_of(rows), exact, pool, scratch, sizes);
        EXPECT_EQ(sizes.overflow, 0u);
        expect_same_bytes(*expected, *same);
        const FlatCapacities roomy{exact.entities + 7, exact.tuples, exact.args + 5};
        const auto padded = encoder->encode_capacity(input_of(rows), roomy, pool, scratch, sizes);
        EXPECT_EQ(sizes.overflow, 0u);
        validate(*padded, encoder->schema());
        const auto args = ints(*padded, "relation_args");
        ASSERT_EQ(args.size(), roomy.args);
        EXPECT_TRUE(std::equal(args.begin(), args.begin() + static_cast<std::ptrdiff_t>(exact.args),
                               ints(*expected, "relation_args").begin()));
        EXPECT_TRUE(std::all_of(args.begin() + static_cast<std::ptrdiff_t>(exact.args), args.end(),
                                [](i64 v) { return v == 0; }));
        const auto roles = ints(*padded, "entity_role");
        EXPECT_TRUE(std::all_of(roles.begin() + static_cast<std::ptrdiff_t>(exact.entities), roles.end(),
                                [](i64 v) { return v == 0; }));
        // too small: true sizes, every array zero
        const auto short_bundle = encoder->encode_capacity(
            input_of(rows), {exact.entities, exact.tuples, exact.args - 1}, pool, scratch, sizes);
        EXPECT_EQ(sizes.overflow, k_overflow_capacity);
        EXPECT_EQ(sizes.args, exact.args);
        for (const rl::ArrayInfo& info : short_bundle->arrays())
        {
            const auto values = ints(*short_bundle, info.name.str().c_str());
            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](i64 v) { return v == 0; })) << info.name;
        }
    }
}

TEST(EncoderCapacity, AnEmptyBatchFitsAnyCapacity)
{
    const auto table = two_instance_table();
    const auto encoder = FlatEncoder::create(table, options_with(Layout::Relation));
    const u64 R = encoder->schema().num_relations();
    const test::Rows rows;
    ThreadPool pool(2);
    EncodeScratch scratch;
    Destination<i32> destination(0, R, {0, 0, 0});
    const FlatSizes sizes = encoder->encode_into(input_of(rows), {0, 0, 0}, destination.buffers(), pool, scratch);
    EXPECT_EQ(sizes.overflow, 0u);
    EXPECT_EQ(sizes.entities + sizes.tuples + sizes.args, 0u);
    EXPECT_EQ(destination.entity_offsets.values(), std::vector<i64>{0});
    EXPECT_EQ(destination.relation_offsets.values(), std::vector<i64>{0});
    EXPECT_TRUE(destination.guards_intact());
}

TEST(EncoderOmitted, WithoutCountsTheOffsetsStillGiveThem)
{
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 30, 2);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions options = options_with(layout);
        const auto with = FlatEncoder::create(table, options);
        options.emit_counts = false;
        const auto without = FlatEncoder::create(table, options);
        EXPECT_NE(with->schema().fingerprint(), without->schema().fingerprint());
        const auto a = with->encode(input_of(rows));
        const auto b = without->encode(input_of(rows));
        validate(*b, without->schema());
        EXPECT_EQ(b->find("relation_counts"), nullptr);
        ASSERT_NE(a->find("relation_counts"), nullptr);
        for (const char* name : {"task_ids", "num_objects", "entity_offsets", "entity_role", "entity_ref",
                                 "relation_offsets", "relation_arg_offsets", "relation_args"})
            EXPECT_EQ(ints(*a, name), ints(*b, name)) << name;
        const u64 G = rows.size(), R = with->schema().num_relations();
        const auto counts = ints(*a, "relation_counts");
        const auto offsets = ints(*b, "relation_offsets");
        for (u64 g = 0; g < G; ++g)
            for (u64 r = 0; r < R; ++r)
            {
                const u64 s = layout == Layout::Graph ? g * R + r : r * G + g;
                ASSERT_EQ(offsets[s + 1] - offsets[s], counts[g * R + r]);
            }
        // the validator knows what the schema omits
        EXPECT_THROW(validate(*a, without->schema()), std::invalid_argument);
        EXPECT_THROW(validate(*b, with->schema()), std::invalid_argument);
    }
}

TEST(EncoderOmitted, ImplicitEntityArraysAreTheObjectsThenTheAuxiliaryEntity)
{
    test::dirty_recycled_blocks();
    const auto table = two_instance_table();
    const test::Rows rows = mixed_rows(*table, 30, 4);
    FlatOptions options = options_with(Layout::Graph);
    options.include_actions = false;
    const auto always = FlatEncoder::create(table, options);
    EXPECT_FALSE(always->schema().implicit_entity_arrays());
    options.entity_arrays = EntityArrays::Auto;
    const auto automatic = FlatEncoder::create(table, options);
    ASSERT_TRUE(automatic->schema().implicit_entity_arrays());
    EXPECT_NE(always->schema().fingerprint(), automatic->schema().fingerprint());
    const auto explicit_batch = always->encode(input_of(rows));
    const auto implicit_batch = automatic->encode(input_of(rows));
    validate(*implicit_batch, automatic->schema());
    EXPECT_EQ(implicit_batch->find("entity_role"), nullptr);
    EXPECT_EQ(implicit_batch->find("entity_ref"), nullptr);
    for (const char* name : {"task_ids", "num_objects", "entity_offsets", "relation_counts", "relation_offsets",
                             "relation_arg_offsets", "relation_args"})
        EXPECT_EQ(ints(*explicit_batch, name), ints(*implicit_batch, name)) << name;
    // the implicit values, written out, are the stored ones
    const auto offsets = ints(*implicit_batch, "entity_offsets");
    const auto objects = ints(*implicit_batch, "num_objects");
    const auto role = ints(*explicit_batch, "entity_role");
    const auto ref = ints(*explicit_batch, "entity_ref");
    bool auxiliary_seen = false;
    for (usize g = 0; g + 1 < offsets.size(); ++g)
        for (i64 e = offsets[g]; e < offsets[g + 1]; ++e)
        {
            const bool object = e - offsets[g] < objects[g];
            ASSERT_EQ(role[e], object ? 0 : 8);
            ASSERT_EQ(ref[e], object ? e - offsets[g] : -1);
            auxiliary_seen = auxiliary_seen || !object;
        }
    EXPECT_TRUE(auxiliary_seen);
    // the same in the relation-major layout, and with threads
    options.layout = Layout::Relation;
    const auto relation = FlatEncoder::create(table, options);
    ThreadPool pool(4);
    EncodeScratch scratch;
    const auto relation_batch = relation->encode(input_of(rows), pool, scratch);
    validate(*relation_batch, relation->schema());
    EXPECT_EQ(relation_batch->find("entity_role"), nullptr);
}

TEST(EncoderOmitted, TheImplicitRuleFollowsTheOptions)
{
    const auto table = two_instance_table();
    FlatOptions options = test::encoder_options();
    options.entity_arrays = EntityArrays::Auto;
    options.include_actions = false;
    auto implicit = [&](const FlatOptions& o)
    { return FlatEncoder::create(table, o)->schema().implicit_entity_arrays(); };
    EXPECT_TRUE(implicit(options));
    FlatOptions with_actions = options;
    with_actions.include_actions = true;
    EXPECT_FALSE(implicit(with_actions));
    FlatOptions virtual_nodes = options;
    virtual_nodes.predicate_virtual_nodes = true;
    EXPECT_FALSE(implicit(virtual_nodes));
    FlatOptions always = options;
    always.entity_arrays = EntityArrays::Always;
    EXPECT_FALSE(implicit(always));
    // the schema alone knows the rule for options this encoder does not encode
    const auto schema = [&](const FlatOptions& o) { return EncodingSchema::create(*table, o); };
    FlatOptions targets = options;
    targets.goal_derivations = {GoalDerivation::Plain};
    targets.targets = {TargetSource::Goal};
    EXPECT_FALSE(schema(targets)->implicit_entity_arrays());
    FlatOptions nodes = options;
    nodes.predicate_virtual_nodes = true;
    EXPECT_FALSE(schema(nodes)->implicit_entity_arrays());
    FlatOptions horizon = options;
    horizon.horizon.mode = HorizonMode::Full;
    EXPECT_FALSE(schema(horizon)->implicit_entity_arrays());
}

TEST(EncoderOmitted, AnEncodeIntoNeedsNullArraysExactlyWhereTheSchemaOmitsThem)
{
    const auto table = two_instance_table();
    FlatOptions options = options_with(Layout::Relation);
    options.include_actions = false;
    options.emit_counts = false;
    options.entity_arrays = EntityArrays::Auto;
    const auto encoder = FlatEncoder::create(table, options);
    const test::Rows rows = mixed_rows(*table, 20, 8);
    const u64 G = rows.size(), R = encoder->schema().num_relations();
    const auto expected = encoder->encode(input_of(rows));
    const FlatCapacities exact{static_cast<u64>(ints(*expected, "entity_offsets").back()),
                               static_cast<u64>(ints(*expected, "relation_offsets").back()),
                               static_cast<u64>(ints(*expected, "relation_arg_offsets").back())};
    Destination<i32> destination(G, R, exact);
    FlatBuffers buffers = destination.buffers();
    ThreadPool pool(2);
    EncodeScratch scratch;
    EXPECT_THROW((void)encoder->encode_into(input_of(rows), exact, buffers, pool, scratch), std::invalid_argument);
    buffers.relation_counts = nullptr;
    buffers.entity_role = nullptr;
    buffers.entity_ref = nullptr;
    const FlatSizes sizes = encoder->encode_into(input_of(rows), exact, buffers, pool, scratch);
    EXPECT_EQ(sizes.overflow, 0u);
    EXPECT_TRUE(destination.guards_intact());
    EXPECT_EQ(destination.relation_args.values(), ints(*expected, "relation_args"));
    EXPECT_EQ(destination.relation_offsets.values(), ints(*expected, "relation_offsets"));
}
}  // namespace
