// The derived words of rl::expand: one axiom evaluation per successor, equal to a fresh evaluation, with the capacity and
// overflow rules of `succ`, and accepted by the encoder in place of its own evaluation.

#include "support.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/encoding/encoder.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace mymyr;
using namespace mymyr::encoding;

namespace
{
struct Problem
{
    const char* domain;
    const char* problem;
};
const std::vector<Problem> k_derived = {{"philosophers", "p03-phil4"}, {"features", "p01"}, {"features", "p03"}};

/// Rows of `derived` past the capacity: they must stay untouched.
constexpr u64 k_guard_rows = 2;

struct Result
{
    u64 total = 0, capacity = 0;
    u32 words = 0, derived_words = 0;
    std::vector<u64> succ, derived;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
};

Result run(const rl::TaskTable& table, const test::Rows& rows, bool derived, u64 capacity, ThreadPool* pool = nullptr,
           u32 derived_words = ~u32{0})
{
    Result r;
    const u32 L = table.label_width();
    r.words = rows.words;
    r.derived_words = derived_words == ~u32{0} ? table.derived_words() : derived_words;
    r.capacity = capacity;
    r.succ.assign(capacity * rows.words, 0);
    r.parent.assign(capacity, -1);
    r.schema.assign(capacity, -1);
    r.binding.assign(capacity * L, -1);
    r.goal.assign(capacity, 0);
    r.derived.assign((capacity + k_guard_rows) * r.derived_words, 0xDEADBEEFDEADBEEFULL);
    r.offsets.assign(rows.size() + 1, 0);
    rl::Expansion x;
    x.capacity = capacity;
    x.words = rows.words;
    x.label_width = L;
    x.succ = r.succ.data();
    x.parent = r.parent.data();
    x.schema = r.schema.data();
    x.binding = r.binding.data();
    x.goal = r.goal.data();
    x.offsets = r.offsets.data();
    if (derived)
    {
        x.derived = r.derived.data();
        x.derived_words = r.derived_words;
    }
    if (pool)
    {
        rl::ExpandScratch scratch;
        rl::expand(table, rows.view(), rows.task_ids.data(), x, {}, *pool, scratch);
    }
    else
        rl::expand(table, rows.view(), rows.task_ids.data(), x, {});
    r.total = x.total;
    return r;
}

/// The derived words of a state row by a fresh axiom evaluation, as the encoder does it.
std::vector<u64> fresh(const rl::TaskTable& table, u32 instance, const u64* row, u32 width)
{
    const Task& task = *table.task(instance);
    std::vector<u64> out(width, 0);
    const WorkspaceLease ws = task.workspace();
    auto& engine = ws->engine();
    engine.set_state(row, table.instance(instance).words);
    engine.build_view();
    ws->axioms().evaluate();
    std::copy_n(engine.derived(), bits::words_for(task.atoms().derived_slots()), out.begin());
    return out;
}

/// The successor rows of a result (a batch over the table) and their instances.
test::Rows successor_rows(const Result& r, const test::Rows& parents)
{
    test::Rows s;
    s.words = r.words;
    const u64 n = std::min(r.total, r.capacity);
    s.data.assign(r.succ.begin(), r.succ.begin() + static_cast<std::ptrdiff_t>(n * r.words));
    for (u64 j = 0; j < n; ++j)
        s.task_ids.push_back(parents.task_ids[static_cast<usize>(r.parent[j])]);
    return s;
}

void expect_same_arrays(const rl::ArrayBundle& a, const rl::ArrayBundle& b)
{
    ASSERT_EQ(a.arrays().size(), b.arrays().size());
    for (usize i = 0; i < a.arrays().size(); ++i)
    {
        const rl::ArrayInfo& x = a.arrays()[i];
        const rl::ArrayInfo& y = b.arrays()[i];
        ASSERT_EQ(x.name, y.name);
        ASSERT_EQ(x.elements(), y.elements());
        EXPECT_EQ(std::memcmp(a.data(x), b.data(y), x.elements() * rl::dtype_bytes(x.dtype)), 0) << x.name;
    }
}
}  // namespace

TEST(ExpandDerived, WordsEqualAFreshEvaluationPerSuccessor)
{
    for (const Problem& p : k_derived)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        ASSERT_GT(table->derived_words(), 0u);
        const test::Rows rows = test::random_walk(*table, 0, 48, 5);
        const Result r = run(*table, rows, true, 4096);
        ASSERT_LE(r.total, r.capacity);
        for (u64 j = 0; j < r.total; ++j)
        {
            const auto expected = fresh(*table, 0, r.succ.data() + j * r.words, r.derived_words);
            ASSERT_TRUE(std::equal(expected.begin(), expected.end(), r.derived.begin() + static_cast<std::ptrdiff_t>(j * r.derived_words)))
                << "successor " << j;
        }
        // the pool variant gives the same bytes
        ThreadPool pool(4);
        const Result q = run(*table, rows, true, 4096, &pool);
        EXPECT_EQ(q.total, r.total);
        EXPECT_EQ(q.derived, r.derived);
        EXPECT_EQ(q.succ, r.succ);
        EXPECT_EQ(q.goal, r.goal);
    }
}

TEST(ExpandDerived, OffLeavesEveryOtherOutputUnchanged)
{
    for (const Problem& p : k_derived)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const test::Rows rows = test::random_walk(*table, 0, 48, 9);
        const Result on = run(*table, rows, true, 4096);
        const Result off = run(*table, rows, false, 4096);
        EXPECT_EQ(on.total, off.total);
        EXPECT_EQ(on.succ, off.succ);
        EXPECT_EQ(on.parent, off.parent);
        EXPECT_EQ(on.schema, off.schema);
        EXPECT_EQ(on.binding, off.binding);
        EXPECT_EQ(on.goal, off.goal);
        EXPECT_EQ(on.offsets, off.offsets);
        for (u64 w : off.derived)
            ASSERT_EQ(w, 0xDEADBEEFDEADBEEFULL);  // not written
    }
}

TEST(ExpandDerived, CapacityClipsAsForSucc)
{
    const auto table = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    const test::Rows rows = test::random_walk(*table, 0, 32, 3);
    const Result full = run(*table, rows, true, 4096);
    const u64 cap = full.total / 2;
    ASSERT_GT(cap, 0u);
    const Result part = run(*table, rows, true, cap);
    EXPECT_EQ(part.total, full.total);
    const usize kept = cap * full.derived_words;
    EXPECT_TRUE(std::equal(part.derived.begin(), part.derived.begin() + static_cast<std::ptrdiff_t>(kept), full.derived.begin()));
    ThreadPool pool(3);
    const Result pooled = run(*table, rows, true, cap, &pool);
    EXPECT_EQ(pooled.derived, part.derived);
    // rows past the capacity are not written, as for succ
    for (usize i = kept; i < part.derived.size(); ++i)
        ASSERT_EQ(part.derived[i], 0xDEADBEEFDEADBEEFULL);
}

TEST(ExpandDerived, WiderRowsAreZeroPaddedAndNarrowerRowsRefused)
{
    const auto table = rl::TaskTable::single(test::pddl_task("philosophers", "p03-phil4"));
    const test::Rows rows = test::random_walk(*table, 0, 8, 1);
    const u32 dw = table->derived_words();
    const Result tight = run(*table, rows, true, 1024);
    const Result wide = run(*table, rows, true, 1024, nullptr, dw + 2);
    for (u64 j = 0; j < tight.total; ++j)
    {
        for (u32 w = 0; w < dw; ++w)
            ASSERT_EQ(wide.derived[j * (dw + 2) + w], tight.derived[j * dw + w]);
        ASSERT_EQ(wide.derived[j * (dw + 2) + dw], 0u);
        ASSERT_EQ(wide.derived[j * (dw + 2) + dw + 1], 0u);
    }
    rl::Expansion x;
    x.capacity = 4;
    x.words = rows.words;
    std::vector<u64> d(4 * dw + 1), succ(4 * rows.words);
    x.derived = d.data();
    x.derived_words = dw;
    EXPECT_THROW(rl::expand(*table, rows.view(), rows.task_ids.data(), x, {}), std::invalid_argument);  // no succ
    x.succ = succ.data();
    x.derived_words = dw - 1;
    EXPECT_THROW(rl::expand(*table, rows.view(), rows.task_ids.data(), x, {}), std::invalid_argument);  // too narrow
}

TEST(ExpandDerived, TasksWithoutDerivedAtomsHaveZeroWidthWords)
{
    const auto table = rl::TaskTable::single(test::pddl_task("blocks", "probBLOCKS-8-0"));
    EXPECT_EQ(table->derived_words(), 0u);
    const test::Rows rows = test::random_walk(*table, 0, 16, 2);
    const Result r = run(*table, rows, true, 1024);
    EXPECT_EQ(r.derived_words, 0u);
    EXPECT_TRUE(r.derived.empty());
    const Result off = run(*table, rows, false, 1024);
    EXPECT_EQ(r.succ, off.succ);
}

TEST(ExpandDerived, PaddedViewCarriesTheWords)
{
    const auto table = rl::TaskTable::single(test::pddl_task("features", "p01"));
    const test::Rows rows = test::random_walk(*table, 0, 16, 4);
    Result r = run(*table, rows, true, 4096);
    rl::Expansion flat;
    flat.capacity = r.capacity;
    flat.words = r.words;
    flat.derived_words = r.derived_words;
    flat.succ = r.succ.data();
    flat.derived = r.derived.data();
    flat.offsets = r.offsets.data();
    flat.total = r.total;
    const u32 K = 16, dw = r.derived_words;
    std::vector<u64> padded(rows.size() * K * dw, 0xFF);
    std::vector<i32> index(rows.size() * K);
    rl::PaddedExpansion out;
    out.K = K;
    out.index = index.data();
    out.derived_words = dw;
    out.derived = padded.data();
    rl::pad(flat, rows.size(), out);
    for (u64 i = 0; i < rows.size(); ++i)
        for (u32 k = 0; k < K; ++k)
        {
            const i32 j = index[i * K + k];
            for (u32 w = 0; w < dw; ++w)
                ASSERT_EQ(padded[(i * K + k) * dw + w], j < 0 ? 0u : r.derived[static_cast<u64>(j) * dw + w]);
        }
}

TEST(ExpandDerived, EncodingWithTheWordsEqualsEncodingWithInternalEvaluation)
{
    for (const Problem& p : k_derived)
    {
        SCOPED_TRACE(std::string(p.domain) + "/" + p.problem);
        const auto table = rl::TaskTable::single(test::pddl_task(p.domain, p.problem));
        const auto encoder = FlatEncoder::create(table, test::encoder_options());
        const test::Rows rows = test::random_walk(*table, 0, 48, 11);
        const Result r = run(*table, rows, true, 4096);
        const test::Rows succ = successor_rows(r, rows);
        ASSERT_GT(succ.size(), 0u);
        EncodeInput own{succ.view(), succ.task_ids.data(), nullptr, 0};
        EncodeInput given{succ.view(), succ.task_ids.data(), r.derived.data(), r.derived_words};
        ASSERT_GE(r.derived_words, encoder->context().max_derived_words());
        expect_same_arrays(*encoder->encode(own), *encoder->encode(given));
    }
}
