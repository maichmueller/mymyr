// The append-only and the mutable stream of the relational encoder: a flush equals one encode of the rows in order.

#include "support.hpp"

#include "mymyr/encoding/encoder.hpp"
#include "mymyr/encoding/stream.hpp"
#include "mymyr/encoding/validate.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <tuple>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
/// Candidate rows of a table with their goals and the expansion of their states.
struct Pool
{
    rl::TaskTablePtr table;
    test::Rows rows;
    std::vector<u64> goal_pos, goal_neg;  // [n, words]
    rl::Expansion x;
    std::vector<i32> schema, binding, offsets;  // the expansion of every row
};

Pool make_pool(rl::TaskTablePtr table, u64 per_instance, u64 seed)
{
    Pool pool;
    pool.table = table;
    pool.rows.words = table->words();
    std::mt19937_64 rng(seed);
    for (u32 i = 0; i < table->size(); ++i)
    {
        const test::Rows part = test::random_walk(*table, i, per_instance, seed + i);
        pool.rows.data.insert(pool.rows.data.end(), part.data.begin(), part.data.end());
        pool.rows.task_ids.insert(pool.rows.task_ids.end(), part.task_ids.begin(), part.task_ids.end());
    }
    const u32 words = table->words();
    pool.goal_pos.assign(pool.rows.size() * words, 0);
    pool.goal_neg.assign(pool.rows.size() * words, 0);
    for (u64 g = 0; g < pool.rows.size(); ++g)
    {
        const u32 slots = table->task(static_cast<u32>(pool.rows.task_ids[g]))->atoms().fluent_slots();
        for (u32 s = 0; s < slots; ++s)
        {
            const u64 draw = rng() % 5;
            if (draw == 0)
                pool.goal_pos[g * words + (s >> 6)] |= u64{1} << (s & 63);
            else if (draw == 1)
                pool.goal_neg[g * words + (s >> 6)] |= u64{1} << (s & 63);
        }
    }
    u64 capacity = pool.rows.size() * 8 + 16;
    for (;;)
    {
        std::vector<u64> succ(capacity * words, 0);
        std::vector<i32> parent(capacity, 0);
        pool.schema.assign(capacity, 0);
        pool.binding.assign(capacity * table->label_width(), 0);
        pool.offsets.assign(pool.rows.size() + 1, 0);
        pool.x = {};
        pool.x.capacity = capacity;
        pool.x.words = words;
        pool.x.label_width = table->label_width();
        pool.x.succ = succ.data();
        pool.x.parent = parent.data();
        pool.x.schema = pool.schema.data();
        pool.x.binding = pool.binding.data();
        pool.x.offsets = pool.offsets.data();
        rl::expand(*table, pool.rows.view(), pool.rows.task_ids.data(), pool.x);
        if (!pool.x.overflow())
            break;
        capacity = std::max(capacity * 2, pool.x.total);
    }
    return pool;
}

/// The rows `picks` of a pool as one batch: rows, goals and an expansion of their own (no parent array).
struct Selected
{
    static inline i32 k_empty = 0;
    test::Rows rows;
    std::vector<u64> goal_pos, goal_neg;
    std::vector<i32> schema, binding, offsets;
    rl::Expansion x;

    Selected(const Pool& pool, std::span<const u64> picks)
    {
        const u32 words = pool.rows.words, L = pool.table->label_width();
        rows.words = words;
        offsets.push_back(0);
        for (const u64 pick : picks)
        {
            const auto row = pool.rows.data.begin() + static_cast<std::ptrdiff_t>(pick * words);
            rows.data.insert(rows.data.end(), row, row + words);
            rows.task_ids.push_back(pool.rows.task_ids[pick]);
            goal_pos.insert(goal_pos.end(), pool.goal_pos.begin() + static_cast<std::ptrdiff_t>(pick * words),
                            pool.goal_pos.begin() + static_cast<std::ptrdiff_t>((pick + 1) * words));
            goal_neg.insert(goal_neg.end(), pool.goal_neg.begin() + static_cast<std::ptrdiff_t>(pick * words),
                            pool.goal_neg.begin() + static_cast<std::ptrdiff_t>((pick + 1) * words));
            const i32 first = pool.offsets[pick], last = pool.offsets[pick + 1];
            schema.insert(schema.end(), pool.schema.begin() + first, pool.schema.begin() + last);
            binding.insert(binding.end(), pool.binding.begin() + static_cast<std::ptrdiff_t>(first * L),
                           pool.binding.begin() + static_cast<std::ptrdiff_t>(last * L));
            offsets.push_back(static_cast<i32>(schema.size()));
        }
        x.capacity = x.total = schema.size();
        x.label_width = L;
        x.schema = schema.empty() ? &k_empty : schema.data();
        x.binding = binding.empty() ? &k_empty : binding.data();
        x.offsets = offsets.data();
    }
    Selected(const Selected&) = delete;
    Selected& operator=(const Selected&) = delete;

    [[nodiscard]] EncodeInput input(bool goals, bool actions) const
    {
        EncodeInput in;
        in.states = rows.view();
        in.task_ids = rows.task_ids.data();
        if (goals)
        {
            in.goal_pos = goal_pos.data();
            in.goal_neg = goal_neg.data();
            in.goal_stride = in.goal_words = rows.words;
        }
        if (actions)
            in.actions = &x;
        return in;
    }
};

void expect_equal_bundles(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    ASSERT_EQ(a.bytes(), b.bytes());
    ASSERT_EQ(a.arrays().size(), b.arrays().size());
    EXPECT_EQ(std::memcmp(a.block(), b.block(), a.bytes()), 0);
    EXPECT_EQ(a.scalars(), b.scalars());
}

struct Flavor
{
    bool goal_relations;  // the schema has goal relations
    bool row_goals;       // the rows give their goals
    bool actions;
    Layout layout;
};

FlatOptions options_of(const Flavor& flavor)
{
    FlatOptions options = test::encoder_options();
    if (flavor.goal_relations)
        options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    options.include_actions = flavor.actions;
    options.targets = {};
    if (flavor.goal_relations)
        options.targets.insert(TargetSource::Goal);
    if (flavor.actions)
        options.targets.insert(TargetSource::Action);
    options.predicate_virtual_nodes = flavor.actions;
    options.layout = flavor.layout;
    return options;
}

std::vector<Flavor> flavors()
{
    std::vector<Flavor> out;
    for (const Layout layout : {Layout::Graph, Layout::Relation})
        for (const bool actions : {false, true})
        {
            out.push_back({false, false, actions, layout});
            out.push_back({true, false, actions, layout});
            out.push_back({true, true, actions, layout});
        }
    return out;
}

std::string name_of(const Flavor& f)
{
    return std::string(f.goal_relations ? (f.row_goals ? "row goals" : "instance goals") : "no goals") +
           (f.actions ? ", actions, " : ", ") + std::string(to_string(f.layout));
}

rl::TaskTablePtr blocks_table()
{
    return rl::TaskTable::create({test::pddl_task("blocks", "smedium"), test::pddl_task("blocks", "probBLOCKS-8-0")});
}
}  // namespace

TEST(FlatStream, AFlushEqualsOneEncodeOfTheAppendedRows)
{
    const Pool pool = make_pool(blocks_table(), 80, 5);
    for (const Flavor& flavor : flavors())
    {
        SCOPED_TRACE(name_of(flavor));
        const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
        FlatStream stream(encoder);
        ThreadPool one(1);
        std::mt19937_64 rng(11);
        for (u32 round = 0; round < 6; ++round)
        {
            std::vector<u64> all;
            const u32 chunks = 1 + static_cast<u32>(rng() % 4);
            for (u32 c = 0; c < chunks; ++c)
            {
                std::vector<u64> picks(rng() % 9 == 0 ? 0 : 1 + rng() % 40);
                for (u64& p : picks)
                    p = rng() % pool.rows.size();
                const Selected chunk(pool, picks);
                stream.append(chunk.input(flavor.row_goals, flavor.actions));
                all.insert(all.end(), picks.begin(), picks.end());
            }
            ASSERT_EQ(stream.size(), all.size());
            const Selected whole(pool, all);
            const auto reference = encoder->encode(whole.input(flavor.row_goals, flavor.actions));
            const auto flushed = stream.flush(one);
            expect_equal_bundles(*reference, *flushed);
            validate(*flushed, encoder->schema());
            EXPECT_EQ(stream.size(), 0u);
        }
    }
}

TEST(FlatStream, AppendedBuffersMayBeReusedAtOnce)
{
    const Pool pool = make_pool(blocks_table(), 40, 3);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    std::vector<u64> all;
    for (u64 start = 0; start < 60; start += 20)
    {
        std::vector<u64> picks;
        for (u64 i = 0; i < 20; ++i)
            picks.push_back(start + i);
        auto chunk = std::make_unique<Selected>(pool, picks);
        stream.append(chunk->input(true, true));
        // overwrite what was appended: the stream holds copies
        std::fill(chunk->rows.data.begin(), chunk->rows.data.end(), ~u64{0});
        std::fill(chunk->goal_pos.begin(), chunk->goal_pos.end(), ~u64{0});
        std::fill(chunk->schema.begin(), chunk->schema.end(), 99);
        all.insert(all.end(), picks.begin(), picks.end());
    }
    ThreadPool one(1);
    const Selected whole(pool, all);
    expect_equal_bundles(*encoder->encode(whole.input(true, true)), *stream.flush(one));
}

TEST(FlatStream, ABatchOfSeveralMembersEqualsTheSingleThreadedEncode)
{
    const Pool pool = make_pool(blocks_table(), 1600, 9);
    const Flavor flavor{true, true, true, Layout::Relation};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    std::vector<u64> all(pool.rows.size());
    for (u64 i = 0; i < all.size(); ++i)
        all[i] = (i * 7919) % all.size();
    for (u64 start = 0; start < all.size(); start += 1000)
    {
        const std::span<const u64> part(all.data() + start, std::min<u64>(1000, all.size() - start));
        stream.append(Selected(pool, part).input(true, true));
    }
    ThreadPool pool3(3);
    const Selected whole(pool, all);
    expect_equal_bundles(*encoder->encode(whole.input(true, true)), *stream.flush(pool3));
}

TEST(FlatStream, AFlushOfNoRowsIsAnEmptyBatch)
{
    const Pool pool = make_pool(blocks_table(), 10, 1);
    for (const Flavor& flavor : flavors())
    {
        const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
        FlatStream stream(encoder);
        ThreadPool one(1);
        const Selected none(pool, {});
        expect_equal_bundles(*encoder->encode(none.input(flavor.row_goals, flavor.actions)), *stream.flush(one));
    }
}

TEST(FlatStream, TheInputsOfAllRowsAgreeUntilTheStreamIsEmptyAgain)
{
    const Pool pool = make_pool(blocks_table(), 10, 2);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    const std::vector<u64> picks = {0, 1, 2};
    const Selected some(pool, picks);
    stream.append(some.input(true, true));
    EXPECT_THROW(stream.append(some.input(false, true)), std::invalid_argument);
    EXPECT_THROW(stream.append(some.input(true, false)), std::invalid_argument);
    EXPECT_EQ(stream.size(), 3u);
    stream.clear();
    stream.append(some.input(false, false));  // other inputs once it is empty
    EXPECT_THROW(stream.append(some.input(true, false)), std::invalid_argument);
    ThreadPool one(1);
    expect_equal_bundles(*encoder->encode(some.input(false, false)), *stream.flush(one));
    stream.append(some.input(true, true));
    EXPECT_EQ(stream.size(), 3u);
}

TEST(FlatStream, InvalidRowsAreRefusedAtTheAppendAndChangeNothing)
{
    const Pool pool = make_pool(blocks_table(), 10, 2);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    const std::vector<u64> picks = {0, 1, 2, 3};
    Selected rows(pool, picks);
    stream.append(rows.input(true, true));
    ASSERT_EQ(stream.size(), 4u);

    Selected bad_id(pool, picks);
    bad_id.rows.task_ids[3] = 7;
    EXPECT_THROW(stream.append(bad_id.input(true, true)), std::invalid_argument);

    Selected bad_order(pool, picks);
    EncodeInput in = bad_order.input(true, true);
    std::vector<i32> parent(bad_order.schema.size(), 0);
    rl::Expansion x = bad_order.x;
    x.parent = parent.data();  // every row says it belongs to state 0
    in.actions = &x;
    ASSERT_LT(bad_order.offsets[1], bad_order.offsets[4]);
    EXPECT_THROW(stream.append(in), std::invalid_argument);

    Selected narrow(pool, picks);
    EncodeInput narrow_in = narrow.input(true, true);
    narrow_in.states.words = 0;
    EXPECT_THROW(stream.append(narrow_in), std::invalid_argument);

    Selected no_goal_neg(pool, picks);
    EncodeInput half = no_goal_neg.input(true, true);
    half.goal_neg = nullptr;
    EXPECT_THROW(stream.append(half), std::invalid_argument);

    EXPECT_EQ(stream.size(), 4u);

    ThreadPool one(1);
    expect_equal_bundles(*encoder->encode(rows.input(true, true)), *stream.flush(one));
}

TEST(FlatStream, AFlushThatThrowsKeepsTheRows)
{
    const Pool pool = make_pool(blocks_table(), 10, 2);
    const Flavor flavor{true, false, false, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    const std::vector<u64> picks = {0, 1, 2, 3};
    Selected rows(pool, picks);
    // a state bit its instance has no atom for: the stream stores it, the encoder refuses it at the flush
    const u32 slots = pool.table->task(0)->atoms().fluent_slots();
    ASSERT_LT(slots, 64 * pool.table->words());
    rows.rows.data[2 * rows.rows.words + (slots >> 6)] |= u64{1} << (slots & 63);
    stream.append(rows.input(false, false));
    ThreadPool one(1);
    EXPECT_THROW(std::ignore = stream.flush(one), std::invalid_argument);
    EXPECT_EQ(stream.size(), 4u);
    EXPECT_THROW(std::ignore = stream.flush(one), std::invalid_argument);
    stream.clear();
    EXPECT_EQ(stream.size(), 0u);
}

TEST(FlatStream, TheCapacityFormDropsTheRowsOnlyWhenTheBatchFits)
{
    test::dirty_recycled_blocks();
    const Pool pool = make_pool(blocks_table(), 30, 4);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    std::vector<u64> all(50);
    for (u64 i = 0; i < all.size(); ++i)
        all[i] = (i * 3) % pool.rows.size();
    const Selected rows(pool, all);
    stream.append(rows.input(true, true));
    ThreadPool one(1);
    FlatSizes sizes;
    const BatchShape small{50, 1, 1, 1};
    std::ignore = stream.flush(one, small, sizes);
    EXPECT_TRUE(sizes.overflowed());
    EXPECT_EQ(stream.size(), 50u);
    FlatSizes true_sizes;
    EncodeScratch scratch;
    const BatchShape roomy{50, sizes.entities + 5, sizes.args + 5, sizes.targets + 5};
    const auto reference = encoder->encode_capacity(rows.input(true, true), roomy, one, scratch, true_sizes);
    const auto fitted = stream.flush(one, roomy, sizes);
    EXPECT_FALSE(sizes.overflowed());
    expect_equal_bundles(*reference, *fitted);
    EXPECT_EQ(stream.size(), 0u);
}

TEST(FlatStream, AStreamNeedsAnEncoder)
{
    EXPECT_THROW(FlatStream(nullptr), std::invalid_argument);
    EXPECT_THROW(FlatMutableStream(nullptr), std::invalid_argument);
}

// ------------------------------------------------------------------------------------------------ the mutable stream

namespace
{
/// The model of a mutable stream: the pick (a row of the pool) under each index.
using Model = std::map<i64, u64>;

std::vector<u64> picks_of(const Model& model)
{
    std::vector<u64> out;
    for (const auto& [index, pick] : model)
        out.push_back(pick);
    return out;
}

std::vector<i64> indices_of(const Model& model)
{
    std::vector<i64> out;
    for (const auto& [index, pick] : model)
        out.push_back(index);
    return out;
}
}  // namespace

TEST(FlatMutableStream, AFlushEqualsOneEncodeOfTheRowsInIndexOrder)
{
    const Pool pool = make_pool(blocks_table(), 120, 6);
    for (const Flavor& flavor : flavors())
    {
        SCOPED_TRACE(name_of(flavor));
        const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
        FlatMutableStream stream(encoder);
        ThreadPool one(1);
        std::mt19937_64 rng(21);
        Model model;
        for (u32 step = 0; step < 160; ++step)
        {
            const u64 op = rng() % 10;
            std::vector<i64> indices;
            std::vector<u64> picks;
            if (op < 4 || model.empty())  // append new indices (negative ones too)
            {
                const u64 count = rng() % 12;
                for (u64 i = 0; i < count; ++i)
                {
                    const i64 index = static_cast<i64>(rng() % 90) - 20;
                    if (model.contains(index) || std::find(indices.begin(), indices.end(), index) != indices.end())
                        continue;
                    indices.push_back(index);
                    picks.push_back(rng() % pool.rows.size());
                }
                const Selected rows(pool, picks);
                stream.append(indices, rows.input(flavor.row_goals, flavor.actions));
                for (usize i = 0; i < indices.size(); ++i)
                    model[indices[i]] = picks[i];
            }
            else if (op < 7)  // update held indices
            {
                std::vector<i64> held = indices_of(model);
                std::shuffle(held.begin(), held.end(), rng);
                held.resize(std::min<usize>(held.size(), rng() % 8));
                for (usize i = 0; i < held.size(); ++i)
                    picks.push_back(rng() % pool.rows.size());
                const Selected rows(pool, picks);
                stream.update(held, rows.input(flavor.row_goals, flavor.actions));
                for (usize i = 0; i < held.size(); ++i)
                    model[held[i]] = picks[i];
            }
            else if (op < 9)  // remove held indices
            {
                std::vector<i64> held = indices_of(model);
                std::shuffle(held.begin(), held.end(), rng);
                held.resize(std::min<usize>(held.size(), rng() % 10));
                stream.remove(held);
                for (const i64 index : held)
                    model.erase(index);
            }
            else  // flush: the rows are kept
            {
                const std::vector<u64> live = picks_of(model);
                const Selected whole(pool, live);
                ASSERT_EQ(stream.indices(), indices_of(model));
                const auto reference = encoder->encode(whole.input(flavor.row_goals, flavor.actions));
                expect_equal_bundles(*reference, *stream.flush(one));
                EXPECT_EQ(stream.size(), model.size());
            }
            ASSERT_EQ(stream.size(), model.size());
        }
        const std::vector<u64> live = picks_of(model);
        const Selected whole(pool, live);
        expect_equal_bundles(*encoder->encode(whole.input(flavor.row_goals, flavor.actions)), *stream.flush(one));
    }
}

TEST(FlatMutableStream, ManyRemovalsAndUpdatesKeepTheActionRowsExact)
{
    const Pool pool = make_pool(blocks_table(), 1500, 8);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatMutableStream stream(encoder);
    ThreadPool one(1);
    std::mt19937_64 rng(5);
    Model model;
    for (u32 round = 0; round < 30; ++round)
    {
        std::vector<i64> indices;
        std::vector<u64> picks;
        for (u32 i = 0; i < 400; ++i)
        {
            const i64 index = static_cast<i64>(rng() % 600);
            if (model.contains(index) || std::find(indices.begin(), indices.end(), index) != indices.end())
                continue;
            indices.push_back(index);
            picks.push_back(rng() % pool.rows.size());
        }
        stream.append(indices, Selected(pool, picks).input(true, true));
        for (usize i = 0; i < indices.size(); ++i)
            model[indices[i]] = picks[i];
        std::vector<i64> held = indices_of(model);
        std::shuffle(held.begin(), held.end(), rng);
        std::vector<i64> updated(held.begin(), held.begin() + static_cast<std::ptrdiff_t>(held.size() / 2));
        std::vector<u64> new_picks;
        for (usize i = 0; i < updated.size(); ++i)
            new_picks.push_back(rng() % pool.rows.size());
        stream.update(updated, Selected(pool, new_picks).input(true, true));
        for (usize i = 0; i < updated.size(); ++i)
            model[updated[i]] = new_picks[i];
        std::vector<i64> removed(held.begin() + static_cast<std::ptrdiff_t>(held.size() / 2),
                                 held.begin() + static_cast<std::ptrdiff_t>(held.size() / 2 + held.size() / 3));
        removed.erase(std::remove_if(removed.begin(), removed.end(), [&](i64 i) { return std::find(updated.begin(), updated.end(), i) != updated.end(); }),
                      removed.end());
        stream.remove(removed);
        for (const i64 index : removed)
            model.erase(index);
    }
    const std::vector<u64> live = picks_of(model);
    const Selected whole(pool, live);
    expect_equal_bundles(*encoder->encode(whole.input(true, true)), *stream.flush(one));
}

TEST(FlatMutableStream, IndexErrorsChangeNothing)
{
    const Pool pool = make_pool(blocks_table(), 10, 2);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatMutableStream stream(encoder);
    const std::vector<u64> picks = {0, 1, 2};
    const Selected rows(pool, picks);
    const std::vector<i64> indices = {5, 3, 9};
    stream.append(indices, rows.input(true, true));
    const std::vector<i64> held_and_new = {3, 4, 6};
    EXPECT_THROW(stream.append(held_and_new, rows.input(true, true)), std::invalid_argument);
    const std::vector<i64> twice = {10, 11, 10};
    EXPECT_THROW(stream.append(twice, rows.input(true, true)), std::invalid_argument);
    const std::vector<i64> two = {10, 11};
    EXPECT_THROW(stream.append(two, rows.input(true, true)), std::invalid_argument);  // 2 indices, 3 rows
    const std::vector<i64> missing = {5, 4, 9};
    EXPECT_THROW(stream.update(missing, rows.input(true, true)), std::out_of_range);
    EXPECT_THROW(stream.remove(missing), std::out_of_range);
    const std::vector<i64> repeated = {5, 9, 5};
    EXPECT_THROW(stream.update(repeated, rows.input(true, true)), std::invalid_argument);
    EXPECT_THROW(stream.remove(repeated), std::invalid_argument);
    EXPECT_THROW(stream.append(std::vector<i64>{20, 21, 22}, rows.input(false, true)), std::invalid_argument);
    EXPECT_EQ(stream.indices(), (std::vector<i64>{3, 5, 9}));
    EXPECT_TRUE(stream.contains(5));
    EXPECT_FALSE(stream.contains(4));
    ThreadPool one(1);
    const std::vector<u64> ordered = {1, 0, 2};  // the rows under 3, 5, 9
    expect_equal_bundles(*encoder->encode(Selected(pool, ordered).input(true, true)), *stream.flush(one));
    stream.remove(indices);
    EXPECT_EQ(stream.size(), 0u);
    stream.append(std::vector<i64>{1, 2, 3}, rows.input(false, false));  // empty again: other inputs
    EXPECT_EQ(stream.size(), 3u);
}

TEST(FlatMutableStream, AscendingAppendsAndAppendsBelowTheLargestIndexKeepTheOrder)
{
    const Pool pool = make_pool(blocks_table(), 100, 12);
    const Flavor flavor{false, false, false, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatMutableStream stream(encoder);
    ThreadPool one(1);
    std::mt19937_64 rng(31);
    Model model;
    const auto check = [&]
    {
        ASSERT_EQ(stream.indices(), indices_of(model));
        expect_equal_bundles(*encoder->encode(Selected(pool, picks_of(model)).input(false, false)), *stream.flush(one));
    };
    const auto add = [&](std::vector<i64> indices)
    {
        std::vector<u64> picks;
        for (usize i = 0; i < indices.size(); ++i)
            picks.push_back(rng() % pool.rows.size());
        stream.append(indices, Selected(pool, picks).input(false, false));
        for (usize i = 0; i < indices.size(); ++i)
            model[indices[i]] = picks[i];
    };
    add({-3, -2, 0, 7});  // ascending into an empty stream
    check();
    add({8, 9, 40});  // ascending above the largest index
    check();
    add({41});
    add({100, 50, 60});  // not ascending: merged into the rows held
    check();
    add({-100, 200});  // below the smallest and above the largest at once
    check();
    stream.remove(std::vector<i64>{7, 41, 200});
    for (const i64 index : {7, 41, 200})
        model.erase(index);
    add({300, 301});  // after a removal the order is rebuilt
    check();
    stream.update(std::vector<i64>{0, 300}, Selected(pool, std::vector<u64>{3, 4}).input(false, false));
    model[0] = 3;
    model[300] = 4;
    check();
    stream.clear();
    model.clear();
    add({5, 6, 7});
    check();
}

TEST(FlatMutableStream, ManyIndicesOfAnyShapeAreFoundAfterTheTableGrowsAndShrinks)
{
    const Pool pool = make_pool(blocks_table(), 20, 13);
    const Flavor flavor{false, false, false, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatMutableStream stream(encoder);
    ThreadPool one(1);
    std::mt19937_64 rng(77);
    Model model;
    // counters, multiples of a power of two, the 64-bit extremes and random values collide in different ways
    std::vector<i64> candidates;
    for (i64 i = 0; i < 2500; ++i)
    {
        candidates.push_back(i - 1000);
        candidates.push_back(i * 4096);
        candidates.push_back(std::numeric_limits<i64>::max() - i);
        candidates.push_back(std::numeric_limits<i64>::min() + i);
        candidates.push_back(static_cast<i64>(rng()));
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const usize total = candidates.size();
    ASSERT_GT(total, 10000u);
    for (usize start = 0; start < total; start += 3000)
    {
        const std::span<const i64> part(candidates.data() + start, std::min<usize>(3000, total - start));
        std::vector<u64> picks;
        for (usize i = 0; i < part.size(); ++i)
            picks.push_back(rng() % pool.rows.size());
        stream.append(part, Selected(pool, picks).input(false, false));
        for (usize i = 0; i < part.size(); ++i)
            model[part[i]] = picks[i];
    }
    ASSERT_EQ(stream.size(), model.size());
    for (const i64 index : candidates)
        ASSERT_TRUE(stream.contains(index)) << index;
    EXPECT_FALSE(stream.contains(1'000'000'007));
    // remove most of them in batches: the table shifts entries back over the holes
    const std::vector<i64> doomed(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(total * 9 / 10));
    for (usize start = 0; start < doomed.size(); start += 1500)
    {
        const std::span<const i64> part(doomed.data() + start, std::min<usize>(1500, doomed.size() - start));
        stream.remove(part);
        for (const i64 index : part)
            model.erase(index);
        for (usize probe = 0; probe < 200; ++probe)
        {
            const i64 index = candidates[rng() % total];
            ASSERT_EQ(stream.contains(index), model.contains(index)) << index;
        }
    }
    ASSERT_EQ(stream.size(), model.size());
    ASSERT_EQ(stream.indices(), indices_of(model));
    expect_equal_bundles(*encoder->encode(Selected(pool, picks_of(model)).input(false, false)), *stream.flush(one));
}

// ------------------------------------------------------------------------------------------------ index type and members

TEST(FlatStream, BatchesOfSeveralMembersWithInt64IndicesEqualTheSingleThreadedEncode)
{
    const Pool pool = make_pool(blocks_table(), 1600, 14);
    for (const Layout layout : {Layout::Graph, Layout::Relation})
    {
        FlatOptions options = options_of({true, true, true, layout});
        options.index = IndexType::I64;
        const auto encoder = FlatEncoder::create(pool.table, options);
        FlatStream stream(encoder);
        FlatMutableStream mutable_stream(encoder);
        std::vector<u64> all(pool.rows.size());
        for (u64 i = 0; i < all.size(); ++i)
            all[i] = (i * 7919) % all.size();
        ASSERT_GT(all.size(), 2 * k_graphs_per_member);
        for (u64 start = 0; start < all.size(); start += 900)
        {
            const std::span<const u64> part(all.data() + start, std::min<u64>(900, all.size() - start));
            const Selected chunk(pool, part);
            stream.append(chunk.input(true, true));
            std::vector<i64> indices;
            for (u64 i = 0; i < part.size(); ++i)
                indices.push_back(-static_cast<i64>(start + i));  // descending
            mutable_stream.append(indices, chunk.input(true, true));
        }
        ThreadPool members(3);
        const auto appended = encoder->encode(Selected(pool, all).input(true, true));
        expect_equal_bundles(*appended, *stream.flush(members));
        const std::vector<u64> reversed(all.rbegin(), all.rend());
        const auto by_index = encoder->encode(Selected(pool, reversed).input(true, true));
        expect_equal_bundles(*by_index, *mutable_stream.flush(members));
        validate(*by_index, encoder->schema());
    }
}

// ------------------------------------------------------------------------------------------------ threads

TEST(FlatStream, AppendsFlushesAndChangesFromSeveralThreadsLoseNoRow)
{
    const Pool pool = make_pool(blocks_table(), 200, 41);
    const Flavor flavor{true, true, true, Layout::Graph};
    const auto encoder = FlatEncoder::create(pool.table, options_of(flavor));
    FlatStream stream(encoder);
    FlatMutableStream mutable_stream(encoder);
    constexpr u32 threads = 4, rounds = 60;
    std::atomic<u64> appended{0}, flushed{0};
    std::mutex held_mutex;
    Model held;  // the rows the mutable stream must end up with: index -> pick
    std::vector<std::thread> workers;
    for (u32 t = 0; t < threads; ++t)
        workers.emplace_back(
            [&, t]
            {
                std::mt19937_64 rng(t + 1);
                ThreadPool one(1);
                Model mine;
                for (u32 r = 0; r < rounds; ++r)
                {
                    std::vector<u64> picks(1 + rng() % 9);
                    for (u64& pick : picks)
                        pick = rng() % pool.rows.size();
                    const Selected rows(pool, picks);
                    stream.append(rows.input(true, true));
                    appended += picks.size();
                    std::vector<i64> indices;
                    for (usize i = 0; i < picks.size(); ++i)
                        indices.push_back(static_cast<i64>(t) * 1'000'000 + static_cast<i64>(r) * 100 + static_cast<i64>(i));
                    mutable_stream.append(indices, rows.input(true, true));
                    for (usize i = 0; i < picks.size(); ++i)
                        mine[indices[i]] = picks[i];
                    if (r % 2)
                    {
                        const std::vector<u64> other(1, rng() % pool.rows.size());
                        mutable_stream.update(std::span<const i64>(indices.data(), 1), Selected(pool, other).input(true, true));
                        mine[indices[0]] = other[0];
                    }
                    if (r % 3 == 0 && indices.size() > 1)
                    {
                        mutable_stream.remove(std::span<const i64>(indices.data() + indices.size() - 1, 1));
                        mine.erase(indices.back());
                    }
                    if (r % 5 == 4)
                    {
                        flushed += static_cast<u64>(stream.flush(one)->scalar("num_graphs"));
                        (void)mutable_stream.flush(one);
                        (void)mutable_stream.indices();
                        (void)mutable_stream.contains(indices[0]);
                        (void)mutable_stream.size();
                    }
                }
                std::lock_guard lock(held_mutex);
                held.insert(mine.begin(), mine.end());
            });
    for (std::thread& worker : workers)
        worker.join();
    ThreadPool one(1);
    flushed += static_cast<u64>(stream.flush(one)->scalar("num_graphs"));
    EXPECT_EQ(appended.load(), flushed.load());
    EXPECT_EQ(stream.size(), 0u);
    ASSERT_EQ(mutable_stream.size(), held.size());
    ASSERT_EQ(mutable_stream.indices(), indices_of(held));
    expect_equal_bundles(*encoder->encode(Selected(pool, picks_of(held)).input(true, true)), *mutable_stream.flush(one));
}

// ------------------------------------------------------------------------------------------------ derived words

namespace
{
/// Successor rows of a walk with their derived words and goals over the fluent and derived slots.
struct DerivedRows
{
    test::Rows rows;
    u32 derived_words = 0;
    std::vector<u64> derived, pos, neg, derived_pos, derived_neg;

    explicit DerivedRows(const rl::TaskTablePtr& table)
    {
        const test::Rows walk = test::random_walk(*table, 0, 60, 3);
        const u32 words = table->words();
        derived_words = table->derived_words();
        rows.words = words;
        u64 capacity = walk.size() * 16;
        std::vector<u64> succ, derived_out;
        std::vector<i32> parent, offsets;
        rl::Expansion out;
        for (;;)
        {
            succ.assign(capacity * words, 0);
            derived_out.assign(capacity * derived_words, 0);
            parent.assign(capacity, 0);
            offsets.assign(walk.size() + 1, 0);
            out = {};
            out.capacity = capacity;
            out.words = words;
            out.derived_words = derived_words;
            out.succ = succ.data();
            out.parent = parent.data();
            out.derived = derived_out.data();
            out.offsets = offsets.data();
            rl::expand(*table, walk.view(), walk.task_ids.data(), out);
            if (!out.overflow())
                break;
            capacity = std::max(capacity * 2, out.total);
        }
        rows.data.assign(succ.begin(), succ.begin() + static_cast<std::ptrdiff_t>(out.total * words));
        derived.assign(derived_out.begin(), derived_out.begin() + static_cast<std::ptrdiff_t>(out.total * derived_words));
        rows.task_ids.assign(out.total, 0);
        std::mt19937_64 rng(4);
        const Task& task = *table->task(0);
        pos.assign(out.total * words, 0);
        neg = pos;
        derived_pos.assign(out.total * derived_words, 0);
        derived_neg = derived_pos;
        for (u64 g = 0; g < out.total; ++g)
        {
            for (u32 s = 0; s < task.atoms().fluent_slots(); ++s)
                if (rng() % 3 == 0)
                    (rng() % 4 == 0 ? pos : neg)[g * words + (s >> 6)] |= u64{1} << (s & 63);
            for (u32 s = 0; s < task.atoms().derived_slots(); ++s)
                if (rng() % 2 == 0)
                    (rng() % 2 ? derived_pos : derived_neg)[g * derived_words + (s >> 6)] |= u64{1} << (s & 63);
        }
    }

    /// The rows `picks` as an input; `keep` holds the gathered arrays.
    [[nodiscard]] EncodeInput input(std::span<const u64> picks, std::vector<std::vector<u64>>& keep,
                                    std::vector<i32>& ids, bool with_derived, bool quad) const
    {
        keep.clear();
        keep.reserve(16);
        const auto gather = [&](const std::vector<u64>& from, u32 width)
        {
            std::vector<u64> to;
            for (const u64 pick : picks)
                to.insert(to.end(), from.begin() + static_cast<std::ptrdiff_t>(pick * width),
                          from.begin() + static_cast<std::ptrdiff_t>((pick + 1) * width));
            keep.push_back(std::move(to));
            return keep.back().data();
        };
        ids.assign(picks.size(), 0);
        EncodeInput in;
        in.states = {gather(rows.data, rows.words), picks.size(), rows.words, 0, 0};
        in.task_ids = ids.data();
        if (with_derived)
        {
            in.derived = gather(derived, derived_words);
            in.derived_stride = derived_words;
        }
        in.goal_pos = gather(pos, rows.words);
        in.goal_neg = gather(neg, rows.words);
        in.goal_stride = in.goal_words = rows.words;
        if (quad)
        {
            in.goal_derived_pos = gather(derived_pos, derived_words);
            in.goal_derived_neg = gather(derived_neg, derived_words);
            in.goal_derived_stride = in.goal_derived_words = derived_words;
        }
        return in;
    }
};
}  // namespace

TEST(FlatStream, DerivedWordsAndDerivedGoalMasksAreStoredWithTheRows)
{
    const auto table = rl::TaskTable::single(test::pddl_task("goals", "conjunctive"));
    ASSERT_GT(table->derived_words(), 0u);
    const DerivedRows data(table);
    ASSERT_GT(data.rows.size(), 40u);
    FlatOptions options = test::encoder_options();
    options.goal_derivations = {GoalDerivation::Plain, GoalDerivation::Satisfied, GoalDerivation::Unsatisfied};
    const auto encoder = FlatEncoder::create(table, options);
    ThreadPool one(1);
    for (const bool with_derived : {false, true})
        for (const bool quad : {false, true})
        {
            std::vector<u64> picks;
            for (u64 i = 0; i < data.rows.size(); ++i)
                picks.push_back((i * 7) % data.rows.size());
            std::vector<std::vector<u64>> keep, keep_part;
            std::vector<i32> ids, ids_part;
            const EncodeInput whole = data.input(picks, keep, ids, with_derived, quad);
            const auto reference = encoder->encode(whole);

            FlatStream stream(encoder);
            for (u64 start = 0; start < picks.size(); start += 17)
            {
                const std::span<const u64> part(picks.data() + start, std::min<u64>(17, picks.size() - start));
                stream.append(data.input(part, keep_part, ids_part, with_derived, quad));
            }
            expect_equal_bundles(*reference, *stream.flush(one));

            FlatMutableStream mutable_stream(encoder);
            std::vector<i64> indices(picks.size());
            for (u64 i = 0; i < picks.size(); ++i)
                indices[i] = static_cast<i64>(picks.size() - i);  // appended in descending index order
            const std::vector<u64> reversed(picks.rbegin(), picks.rend());
            mutable_stream.append(indices, data.input(reversed, keep_part, ids_part, with_derived, quad));
            expect_equal_bundles(*reference, *mutable_stream.flush(one));
        }
}
