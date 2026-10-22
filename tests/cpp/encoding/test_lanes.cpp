#include "mymyr/encoding/detail/lanes.hpp"

#include <gtest/gtest.h>

#include <random>
#include <vector>

using namespace mymyr;
using namespace mymyr::encoding::detail;

namespace
{
bool bit_set(const std::vector<u64>& words, u32 index) { return (words[index >> 6] >> (index & 63)) & 1; }

u32 reference_count(const std::vector<u64>& words, u32 lo, u32 hi)
{
    u32 n = 0;
    for (u32 i = lo; i < hi; ++i)
        n += bit_set(words, i);
    return n;
}

std::vector<u32> iterate(const std::vector<u64>& words, u32 lo, u32 hi)
{
    std::vector<u32> out;
    if (lo >= hi)
        return out;
    for (u32 w = lo >> 6; w <= ((hi - 1) >> 6); ++w)
    {
        u64 bits = range_word(words.data(), w, lo, hi);
        while (bits)
        {
            out.push_back(w * 64 + lowest_bit64(bits));
            bits &= bits - 1;
        }
    }
    return out;
}

TEST(EncodingLanes, PopcountRangeAndSetBitIterationMatchABitLoop)
{
    std::mt19937_64 rng(7);
    std::vector<u64> words(4);
    for (int round = 0; round < 2000; ++round)
    {
        for (u64& w : words)
            w = rng() & rng();
        const u32 lo = static_cast<u32>(rng() % 257), hi = static_cast<u32>(rng() % 257);
        const u32 a = std::min(lo, hi), b = std::max(lo, hi);
        ASSERT_EQ(popcount_range(words.data(), a, b), reference_count(words, a, b)) << a << " " << b;
        std::vector<u32> expected;
        for (u32 i = a; i < b; ++i)
            if (bit_set(words, i))
                expected.push_back(i);
        ASSERT_EQ(iterate(words, a, b), expected) << a << " " << b;
    }
    const u32 edges[] = {0, 1, 63, 64, 65, 127, 128, 191, 192};
    for (u64& w : words)
        w = ~u64{0};
    for (u32 lo : edges)
        for (u32 hi : edges)
        {
            EXPECT_EQ(popcount_range(words.data(), lo, hi), hi > lo ? hi - lo : 0u) << lo << " " << hi;
            EXPECT_EQ(iterate(words, lo, hi).size(), hi > lo ? hi - lo : 0u) << lo << " " << hi;
        }
    std::vector<u64> wide(257);
    for (u64& word : wide)
        word = rng();
    for (const u32 lo : {0u, 1u, 63u, 64u, 65u})
        for (const u32 length : {0u, 1u, 63u, 64u, 65u, 127u, 128u, 256u, 1024u, 8192u})
            EXPECT_EQ(popcount_range(wide.data(), lo, lo + length), reference_count(wide, lo, lo + length));
}

TEST(EncodingLanes, AnyBitFromLooksOnlyAtTheGivenWords)
{
    std::vector<u64> words = {0, 0, ~u64{0}};
    EXPECT_FALSE(any_bit_from(words.data(), 2, 0));
    EXPECT_TRUE(any_bit_from(words.data(), 3, 100));
    words = {u64{1} << 5, 0};
    EXPECT_TRUE(any_bit_from(words.data(), 2, 5));
    EXPECT_FALSE(any_bit_from(words.data(), 2, 6));
    EXPECT_FALSE(any_bit_from(words.data(), 2, 128));
}

/// Two instances over four relations: 0 a binary fluent relation, 1 a unary static one, 2 a nullary fluent one, 3 has no
/// cell. Instance 0 has 3 objects and 67 fluent slots (66 binary atoms, 1 nullary), 2 words; instance 1 has 2 objects
/// and 5 slots (4 binary, 1 nullary), 1 word.
struct Fixture
{
    static constexpr u32 R = 4, A = 2;
    std::vector<u64> inst;
    std::vector<Cell> cells;
    std::vector<i32> slot_args, derived_args, static_args;

    /// Gives instance `i` `count` cells from row `offset` on, for every list (there are no goal and no action cells, so
    /// the lists are the same).
    void set_cells(u64 i, u64 count, u64 offset)
    {
        u64* row = inst.data() + i * k_instance_columns;
        for (const auto& [n, at] :
             {std::pair{k_num_cells, k_cell_offset}, std::pair{k_num_row_cells, k_row_cell_offset},
              std::pair{k_num_plain_cells, k_plain_cell_offset},
              std::pair{k_num_plain_row_cells, k_plain_row_cell_offset}})
        {
            row[n] = count;
            row[at] = offset;
        }
    }

    Fixture()
    {
        auto instance = [&](u64 objects, u64 words, u64 slots, u64 cell_count, u64 slot_offset, u64 static_offset)
        {
            u64 row[k_instance_columns] = {};
            row[k_num_objects] = objects;
            row[k_words] = words;
            row[k_fluent_slots] = slots;
            row[k_slot_args_offset] = slot_offset;
            row[k_static_args_offset] = static_offset;
            inst.insert(inst.end(), std::begin(row), std::end(row));
            set_cells(inst.size() / k_instance_columns - 1, cell_count, cells.size());
        };
        // instance 0
        instance(3, 2, 67, 3, 0, 0);
        cells.push_back({0, static_cast<u32>(CellOp::Fluent), 0, 66, 2, 2});
        cells.push_back({1, static_cast<u32>(CellOp::Static), 0, 2, 1, 1});
        cells.push_back({2, static_cast<u32>(CellOp::Fluent), 66, 67, 1, 0});
        for (u32 s = 0; s < 67; ++s)
        {
            slot_args.push_back(static_cast<i32>(s % 3));
            slot_args.push_back(s < 66 ? static_cast<i32>((s / 3) % 3) : -1);
        }
        static_args = {2, -1, 0, -1};
        // instance 1: its rows follow instance 0's
        instance(2, 1, 5, 2, 67, 2);
        cells.push_back({0, static_cast<u32>(CellOp::Fluent), 0, 4, 2, 2});
        cells.push_back({2, static_cast<u32>(CellOp::Fluent), 4, 5, 1, 0});
        for (u32 s = 0; s < 5; ++s)
        {
            slot_args.push_back(static_cast<i32>(s % 2));
            slot_args.push_back(s < 4 ? static_cast<i32>(1 - s % 2) : -1);
        }
    }

    [[nodiscard]] ContextView view() const
    {
        return {inst.data(), 2, cells.data(), slot_args.data(), derived_args.data(), static_args.data(), A, R};
    }
};

struct Arrays
{
    std::vector<i64> num_objects, counts, offsets, arg_offsets, args, ref;
    std::vector<u8> role;
};

/// The positions of a relation-major batch from its segment arrays, as a fill with lanes reads them.
template<class I>
struct SegmentPositions
{
    const GraphOut<I>& out;

    u64 args(u64 segment, u32 /*relation*/) const { return static_cast<u64>(out.relation_arg_offsets[segment]); }
    u64 numeric_before(u64 segment, u64 block) const
    {
        return static_cast<u64>(out.relation_offsets[segment]) - block;
    }
    u32 count(u64 segment) const
    {
        return static_cast<u32>(out.relation_offsets[segment + 1] - out.relation_offsets[segment]);
    }
};

/// Encodes the rows with the per-graph functions; `group` takes the forms for the lanes of a group, with a group of one
/// lane.
template<class I>
Arrays run(const Fixture& fixture, const std::vector<std::pair<u32, std::vector<u64>>>& rows,
           std::vector<GraphTotals>* totals_out, bool relation_major = false, bool group = false)
{
    const ContextView view = fixture.view();
    const u64 G = rows.size();
    std::vector<GraphTotals> totals;
    std::vector<std::vector<u32>> cell_counts(G);
    u64 entities = 0, tuples = 0, args = 0;
    std::vector<u64> entity_base(G), tuple_base(G), arg_base(G);
    for (u64 g = 0; g < G; ++g)
    {
        const InstanceView v = instance_view(view, rows[g].first);
        const RowInput in{rows[g].second.data(), nullptr};
        cell_counts[g].resize(v.num_cells);
        totals.push_back(group ? count_graph_group<false>(SoloGroup{}, v, in, cell_counts[g].data())
                               : count_graph<false>(v, in, cell_counts[g].data()));
        entity_base[g] = entities;
        tuple_base[g] = tuples;
        arg_base[g] = args;
        entities += totals[g].entities;
        tuples += totals[g].tuples;
        args += totals[g].args;
    }
    std::vector<I> num_objects(G), counts(G * Fixture::R), offsets(G * Fixture::R + 1), arg_offsets(G * Fixture::R + 1),
        arg(args), ref(entities);
    std::vector<u8> role(entities);
    const GraphOut<I> out{G,         num_objects.data(), counts.data(), offsets.data(),
                          arg_offsets.data(), arg.data(),   role.data(),   ref.data()};
    if (relation_major)
    {
        // the scan: the first position of every relation, then the graphs in order
        std::vector<u64> relation_tuples(Fixture::R, 0), run_tuples(Fixture::R, 0), run_args(Fixture::R, 0);
        for (u64 g = 0; g < G; ++g)
        {
            const InstanceView v = instance_view(view, rows[g].first);
            for (u32 i = 0; i < v.num_cells; ++i)
                relation_tuples[v.cells[i].relation] += cell_counts[g][i];
        }
        u64 tuple_position = 0, arg_position = 0;
        for (u32 r = 0; r < Fixture::R; ++r)
        {
            run_tuples[r] = tuple_position;
            run_args[r] = arg_position;
            tuple_position += relation_tuples[r];
            arg_position += relation_tuples[r] * (r == 0 ? 2 : 1);
        }
        for (u64 g = 0; g < G; ++g)
        {
            const InstanceView v = instance_view(view, rows[g].first);
            std::vector<u32> by_relation(Fixture::R, 0);
            for (u32 i = 0; i < v.num_cells; ++i)
                by_relation[v.cells[i].relation] = cell_counts[g][i];
            for (u32 r = 0; r < Fixture::R; ++r)
            {
                offsets[cell_segment(true, g, r, G, Fixture::R)] = static_cast<I>(run_tuples[r]);
                arg_offsets[cell_segment(true, g, r, G, Fixture::R)] = static_cast<I>(run_args[r]);
                run_tuples[r] += by_relation[r];
                run_args[r] += by_relation[r] * (r == 0 ? 2 : 1);
            }
        }
        offsets[G * Fixture::R] = static_cast<I>(tuples);
        arg_offsets[G * Fixture::R] = static_cast<I>(args);
        for (u64 g = 0; g < G; ++g)
        {
            const InstanceView v = instance_view(view, rows[g].first);
            const RowInput in{rows[g].second.data(), nullptr};
            if (group)
                fill_graph_positioned_group<I, SegmentPositions<I>, false>(SoloGroup{}, v, in, g, entity_base[g],
                                                                           totals[g], out, SegmentPositions<I>{out}, 0,
                                                                           nullptr, nullptr);
            else
                fill_graph_relation_major<I, false>(v, in, g, entity_base[g], totals[g], out);
        }
    }
    else
        for (u64 g = 0; g < G; ++g)
        {
            const InstanceView v = instance_view(view, rows[g].first);
            const RowInput in{rows[g].second.data(), nullptr};
            if (group)
                fill_graph_group<I, false>(SoloGroup{}, v, in, g, tuple_base[g], arg_base[g], entity_base[g], totals[g],
                                           out, 0, nullptr, cell_counts[g].data());
            else
                fill_graph<I, false>(v, in, g, tuple_base[g], arg_base[g], entity_base[g], totals[g], out);
        }
    offsets[G * Fixture::R] = static_cast<I>(tuples);
    arg_offsets[G * Fixture::R] = static_cast<I>(args);
    if (totals_out)
        *totals_out = totals;
    auto widen = [](const auto& v) { return std::vector<i64>(v.begin(), v.end()); };
    return {widen(num_objects), widen(counts), widen(offsets), widen(arg_offsets), widen(arg), widen(ref), role};
}

/// The expected arrays of one graph, computed bit by bit from the fixture's definition.
struct Reference
{
    std::vector<std::vector<std::vector<i64>>> cell_tuples;  // [relation][tuple] -> entity ids (local)
    bool auxiliary = false;
};

Reference reference(const Fixture& f, u32 instance, const std::vector<u64>& atoms)
{
    Reference r;
    r.cell_tuples.resize(Fixture::R);
    const u64* row = f.inst.data() + instance * k_instance_columns;
    const u32 objects = static_cast<u32>(row[k_num_objects]);
    for (u64 c = 0; c < row[k_num_cells]; ++c)
    {
        const Cell& cell = f.cells[row[k_cell_offset] + c];
        for (u32 s = cell.a; s < cell.b; ++s)
        {
            std::vector<i64> tuple;
            if (cell.op == static_cast<u32>(CellOp::Static))
            {
                for (u32 k = 0; k < cell.logical_arity; ++k)
                    tuple.push_back(f.static_args[(row[k_static_args_offset] + s) * Fixture::A + k]);
            }
            else
            {
                if (!bit_set(atoms, s))
                    continue;
                if (cell.logical_arity == 0)
                    tuple.push_back(objects);
                for (u32 k = 0; k < cell.logical_arity; ++k)
                    tuple.push_back(f.slot_args[(row[k_slot_args_offset] + s) * Fixture::A + k]);
            }
            if (cell.logical_arity == 0 && cell.op == static_cast<u32>(CellOp::Static))
                tuple.push_back(objects);
            r.cell_tuples[cell.relation].push_back(tuple);
            r.auxiliary = r.auxiliary || cell.logical_arity == 0;
        }
    }
    return r;
}

TEST(EncodingLanes, CountAndFillMatchAReferenceForBothIndexTypes)
{
    const Fixture fixture;
    std::mt19937_64 rng(11);
    for (int round = 0; round < 50; ++round)
    {
        std::vector<std::pair<u32, std::vector<u64>>> rows;
        for (int g = 0; g < 5; ++g)
        {
            const u32 instance = static_cast<u32>(rng() % 2);
            std::vector<u64> atoms(2, 0);
            const u32 slots = instance == 0 ? 67 : 5;
            for (u32 s = 0; s < slots; ++s)
                if (rng() % 3 == 0)
                    atoms[s >> 6] |= u64{1} << (s & 63);
            // the padding word of the narrower instance is garbage that must not be read
            if (instance == 1)
                atoms[1] = rng();
            rows.emplace_back(instance, atoms);
        }
        std::vector<GraphTotals> totals;
        const Arrays wide = run<i64>(fixture, rows, &totals);
        const Arrays narrow = run<i32>(fixture, rows, nullptr);
        EXPECT_EQ(wide.args, narrow.args);
        EXPECT_EQ(wide.offsets, narrow.offsets);
        u64 entity_base = 0, tuple_base = 0, arg_base = 0;
        for (u64 g = 0; g < rows.size(); ++g)
        {
            const u32 instance = rows[g].first;
            const u32 objects = instance == 0 ? 3 : 2;
            const Reference expected = reference(fixture, instance, rows[g].second);
            EXPECT_EQ(totals[g].flags, expected.auxiliary ? k_graph_auxiliary : 0u);
            EXPECT_EQ(totals[g].entities, objects + (expected.auxiliary ? 1u : 0u));
            EXPECT_EQ(wide.num_objects[g], objects);
            u64 tuples = 0, args = 0;
            for (u32 r = 0; r < Fixture::R; ++r)
            {
                const u32 arity = r == 0 ? 2 : 1;
                const auto& list = expected.cell_tuples[r];
                EXPECT_EQ(wide.counts[g * Fixture::R + r], static_cast<i64>(list.size()));
                EXPECT_EQ(wide.offsets[g * Fixture::R + r], static_cast<i64>(tuple_base + tuples));
                EXPECT_EQ(wide.arg_offsets[g * Fixture::R + r], static_cast<i64>(arg_base + args));
                for (u64 t = 0; t < list.size(); ++t)
                    for (u32 k = 0; k < arity; ++k)
                        EXPECT_EQ(wide.args[arg_base + args + t * arity + k], static_cast<i64>(entity_base) + list[t][k]);
                tuples += list.size();
                args += list.size() * arity;
            }
            EXPECT_EQ(totals[g].tuples, tuples);
            EXPECT_EQ(totals[g].args, args);
            for (u32 o = 0; o < objects; ++o)
            {
                EXPECT_EQ(wide.role[entity_base + o], 0);
                EXPECT_EQ(wide.ref[entity_base + o], o);
            }
            if (expected.auxiliary)
            {
                EXPECT_EQ(wide.role[entity_base + objects], 8);
                EXPECT_EQ(wide.ref[entity_base + objects], -1);
            }
            entity_base += totals[g].entities;
            tuple_base += tuples;
            arg_base += args;
        }
        EXPECT_EQ(wide.offsets.back(), static_cast<i64>(tuple_base));
        EXPECT_EQ(wide.arg_offsets.back(), static_cast<i64>(arg_base));
    }
}

std::vector<std::pair<u32, std::vector<u64>>> random_rows(std::mt19937_64& rng, int graphs)
{
    std::vector<std::pair<u32, std::vector<u64>>> rows;
    for (int g = 0; g < graphs; ++g)
    {
        const u32 instance = static_cast<u32>(rng() % 2);
        std::vector<u64> atoms(2, 0);
        const u32 slots = instance == 0 ? 67 : 5;
        for (u32 s = 0; s < slots; ++s)
            if (rng() % 3 == 0)
                atoms[s >> 6] |= u64{1} << (s & 63);
        rows.emplace_back(instance, atoms);
    }
    return rows;
}

TEST(EncodingLanes, RelationMajorIsThePermutationOfGraphMajor)
{
    const Fixture fixture;
    std::mt19937_64 rng(5);
    for (int round = 0; round < 40; ++round)
    {
        const auto rows = random_rows(rng, 1 + static_cast<int>(rng() % 7));
        const u64 G = rows.size();
        const u64 R = Fixture::R;
        const Arrays graph_major = run<i32>(fixture, rows, nullptr, false);
        const Arrays relation_major = run<i32>(fixture, rows, nullptr, true);
        // everything but the cell order is equal
        EXPECT_EQ(graph_major.num_objects, relation_major.num_objects);
        EXPECT_EQ(graph_major.counts, relation_major.counts);
        EXPECT_EQ(graph_major.role, relation_major.role);
        EXPECT_EQ(graph_major.ref, relation_major.ref);
        EXPECT_EQ(graph_major.offsets.back(), relation_major.offsets.back());
        // the cell at relation-major segment r * G + g is the cell at graph-major segment g * R + r
        std::vector<i64> permuted;
        for (u64 r = 0; r < R; ++r)
            for (u64 g = 0; g < G; ++g)
            {
                const u64 from = g * R + r;
                permuted.insert(permuted.end(), graph_major.args.begin() + graph_major.arg_offsets[from],
                                graph_major.args.begin() + graph_major.arg_offsets[from + 1]);
                const u64 to = r * G + g;
                EXPECT_EQ(relation_major.offsets[to + 1] - relation_major.offsets[to], graph_major.counts[from]);
            }
        EXPECT_EQ(permuted, relation_major.args);
        // the offsets are the exclusive prefix sums of the cell sizes in segment order
        i64 tuples = 0, args = 0;
        for (u64 r = 0; r < R; ++r)
            for (u64 g = 0; g < G; ++g)
            {
                EXPECT_EQ(relation_major.offsets[r * G + g], tuples);
                EXPECT_EQ(relation_major.arg_offsets[r * G + g], args);
                tuples += graph_major.counts[g * R + r];
                args += graph_major.counts[g * R + r] * (r == 0 ? 2 : 1);
            }
        EXPECT_EQ(relation_major.offsets.back(), tuples);
        EXPECT_EQ(relation_major.arg_offsets.back(), args);
    }
}

TEST(EncodingLanes, TheFormsForALaneGroupOfOneLaneEqualThePlainForms)
{
    const Fixture fixture;
    std::mt19937_64 rng(21);
    for (int round = 0; round < 40; ++round)
    {
        const auto rows = random_rows(rng, 1 + static_cast<int>(rng() % 7));
        for (const bool relation_major : {false, true})
        {
            std::vector<GraphTotals> plain_totals, group_totals;
            const Arrays plain = run<i32>(fixture, rows, &plain_totals, relation_major, false);
            const Arrays grouped = run<i32>(fixture, rows, &group_totals, relation_major, true);
            ASSERT_EQ(plain_totals.size(), group_totals.size());
            for (u64 g = 0; g < plain_totals.size(); ++g)
            {
                EXPECT_EQ(plain_totals[g].tuples, group_totals[g].tuples);
                EXPECT_EQ(plain_totals[g].args, group_totals[g].args);
                EXPECT_EQ(plain_totals[g].entities, group_totals[g].entities);
                EXPECT_EQ(plain_totals[g].flags, group_totals[g].flags);
            }
            EXPECT_EQ(plain.num_objects, grouped.num_objects);
            EXPECT_EQ(plain.counts, grouped.counts);
            EXPECT_EQ(plain.offsets, grouped.offsets);
            EXPECT_EQ(plain.arg_offsets, grouped.arg_offsets);
            EXPECT_EQ(plain.args, grouped.args);
            EXPECT_EQ(plain.ref, grouped.ref);
            EXPECT_EQ(plain.role, grouped.role);
        }
    }
}

TEST(EncodingLanes, SegmentOfACellFollowsTheLayout)
{
    EXPECT_EQ(cell_segment(false, 3, 2, 10, 7), 3u * 7u + 2u);
    EXPECT_EQ(cell_segment(true, 3, 2, 10, 7), 2u * 10u + 3u);
}

TEST(EncodingLanes, OmittedCountsAndEntityArraysAreNotWritten)
{
    const Fixture fixture;
    const ContextView view = fixture.view();
    std::mt19937_64 rng(9);
    const auto rows = random_rows(rng, 4);
    const u64 G = rows.size();
    const Arrays reference = run<i32>(fixture, rows, nullptr, false);
    std::vector<i32> num_objects(G), offsets(G * Fixture::R + 1), arg_offsets(G * Fixture::R + 1),
        args(reference.args.size());
    GraphOut<i32> out{G,           num_objects.data(), nullptr, offsets.data(), arg_offsets.data(),
                      args.data(), nullptr,            nullptr};
    u64 entity = 0, tuple = 0, arg = 0;
    for (u64 g = 0; g < G; ++g)
    {
        const InstanceView v = instance_view(view, rows[g].first);
        const GraphTotals totals = count_graph<false>(v, {rows[g].second.data(), nullptr}, nullptr);
        fill_graph<i32, false>(v, {rows[g].second.data(), nullptr}, g, tuple, arg, entity, totals, out);
        entity += totals.entities;
        tuple += totals.tuples;
        arg += totals.args;
    }
    offsets[G * Fixture::R] = static_cast<i32>(tuple);
    arg_offsets[G * Fixture::R] = static_cast<i32>(arg);
    EXPECT_EQ(std::vector<i64>(offsets.begin(), offsets.end()), reference.offsets);
    EXPECT_EQ(std::vector<i64>(args.begin(), args.end()), reference.args);
}

TEST(EncodingLanes, AStaticTupleOfANullaryPredicateIsTheAuxiliaryTuple)
{
    Fixture fixture;
    // instance 1 gets a nullary static cell on relation 3
    fixture.cells.push_back({3, static_cast<u32>(CellOp::Static), 0, 1, 1, 0});
    fixture.set_cells(1, 3, fixture.inst[k_instance_columns + k_cell_offset]);
    fixture.static_args.insert(fixture.static_args.end(), {-1, -1});
    const ContextView view = fixture.view();
    const InstanceView v = instance_view(view, 1);
    const std::vector<u64> atoms = {0, 0};
    const GraphTotals totals = count_graph<false>(v, {atoms.data(), nullptr}, nullptr);
    EXPECT_EQ(totals.tuples, 1u);
    EXPECT_EQ(totals.args, 1u);
    EXPECT_EQ(totals.flags, k_graph_auxiliary);
    EXPECT_EQ(totals.entities, 3u);
}

TEST(EncodingLanes, CellCountsAreReportedInCellOrder)
{
    const Fixture fixture;
    const InstanceView v = instance_view(fixture.view(), 0);
    const std::vector<u64> atoms = {0b101, u64{1} << 2};  // slots 0, 2 and 66
    u32 counts[3] = {};
    const GraphTotals totals = count_graph<false>(v, {atoms.data(), nullptr}, counts);
    EXPECT_EQ(counts[0], 2u);
    EXPECT_EQ(counts[1], 2u);
    EXPECT_EQ(counts[2], 1u);
    EXPECT_EQ(totals.tuples, 5u);
}

TEST(EncodingLanes, BitsPastTheFluentSlotsAreFlaggedAndPaddingWordsAreIgnored)
{
    const Fixture fixture;
    const ContextView view = fixture.view();
    // instance 0: 67 slots in 2 words; bit 67 is past them
    std::vector<u64> atoms = {0, u64{1} << 3};
    EXPECT_TRUE(count_graph<false>(instance_view(view, 0), {atoms.data(), nullptr}, nullptr).flags & k_graph_bad_row);
    atoms = {0, u64{1} << 2};
    EXPECT_FALSE(count_graph<false>(instance_view(view, 0), {atoms.data(), nullptr}, nullptr).flags & k_graph_bad_row);
    // instance 1: 5 slots in 1 word; bit 5 is past them, and a second word is padding
    atoms = {u64{1} << 5, 0};
    EXPECT_TRUE(count_graph<false>(instance_view(view, 1), {atoms.data(), nullptr}, nullptr).flags & k_graph_bad_row);
    atoms = {u64{1} << 4, ~u64{0}};
    const GraphTotals totals = count_graph<false>(instance_view(view, 1), {atoms.data(), nullptr}, nullptr);
    EXPECT_FALSE(totals.flags & k_graph_bad_row);
    EXPECT_EQ(totals.tuples, 1u);
}

TEST(EncodingLanes, DerivedBitsPastTheDerivedSlotsAreFlagged)
{
    Fixture fixture;
    fixture.inst[k_derived_words] = 1;
    fixture.inst[k_derived_slots] = 3;
    fixture.cells.insert(fixture.cells.begin() + 3, {3, static_cast<u32>(CellOp::Derived), 0, 3, 1, 1});
    fixture.set_cells(0, 4, 0);
    fixture.set_cells(1, 2, fixture.inst[k_instance_columns + k_cell_offset] + 1);
    fixture.derived_args = {0, -1, 1, -1, 2, -1};
    const ContextView view = fixture.view();
    const std::vector<u64> atoms = {0, 0};
    std::vector<u64> derived = {0b101};
    GraphTotals totals = count_graph<false>(instance_view(view, 0), {atoms.data(), derived.data()}, nullptr);
    EXPECT_EQ(totals.tuples, 2u + 2u);  // two derived atoms and two static tuples
    EXPECT_FALSE(totals.flags & k_graph_bad_row);
    derived = {0b1000};
    totals = count_graph<false>(instance_view(view, 0), {atoms.data(), derived.data()}, nullptr);
    EXPECT_TRUE(totals.flags & k_graph_bad_row);
}
}  // namespace
