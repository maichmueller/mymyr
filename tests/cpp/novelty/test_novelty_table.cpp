// Novelty tables against a reference set of tuples: random states and transitions, arities 1..5, dense and sparse
// levels, growth (lazy slots), and the switch from dense to sparse, with ranked (64-bit) and packed (128-bit) keys.
// The generator is std::mt19937_64 with plain modulo draws (identical under libstdc++ and libc++).

#include "mymyr/novelty/novelty_table.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

using namespace mymyr;
using namespace mymyr::novelty;

namespace
{
/// The fork's definition, literally: a set of sorted tuples of size 1..k.
struct Reference
{
    u32 k;
    std::set<std::vector<u32>> seen;

    template<class Keep>
    bool visit(const std::vector<u32>& atoms, Keep keep, bool mark)
    {
        bool novel = false;
        std::vector<u32> t;
        const u32 n = static_cast<u32>(atoms.size());
        auto rec = [&](auto& self, u32 from) -> void
        {
            if (!t.empty() && keep(t))
            {
                if (!seen.count(t))
                {
                    novel = true;
                    if (mark)
                        seen.insert(t);
                }
            }
            if (t.size() == k)
                return;
            for (u32 i = from; i < n; ++i)
            {
                t.push_back(atoms[i]);
                self(self, i + 1);
                t.pop_back();
            }
        };
        rec(rec, 0);
        return novel;
    }
    bool mark_state(const std::vector<u32>& atoms)
    {
        return visit(atoms, [](const std::vector<u32>&) { return true; }, true);
    }
    bool transition(const std::vector<u32>& succ, const std::vector<u32>& add, bool mark)
    {
        return visit(
            succ, [&](const std::vector<u32>& t)
            { return std::any_of(t.begin(), t.end(), [&](u32 a) { return std::find(add.begin(), add.end(), a) != add.end(); }); },
            mark);
    }
};

std::vector<u64> words_of(const std::vector<u32>& atoms, u32 nw)
{
    std::vector<u64> w(nw, 0);
    for (u32 a : atoms)
        w[a >> 6] |= u64{1} << (a & 63);
    return w;
}

struct Case
{
    u32 k;
    u64 dense_bytes;
    u32 atoms;           // slots used
    u32 state_size;      // atoms per state
    u32 max_atoms = 0;   // the table's bound (0: atoms); large bounds make levels unranked (128-bit packed keys)
};

class NoveltyRandom : public ::testing::TestWithParam<Case>
{
};

TEST_P(NoveltyRandom, EqualsTheReference)
{
    const Case c = GetParam();
    std::mt19937_64 rng(1234 + c.k * 7 + c.atoms);
    NoveltyTable table(c.k, c.max_atoms ? c.max_atoms : c.atoms, TableOptions{.max_dense_bytes = c.dense_bytes});
    Reference ref{c.k, {}};
    const u32 nw = (c.atoms + 63) / 64;
    // slots appear gradually, as under lazy slots: the table grows (and remaps) as they do
    u32 live = std::min<u32>(c.atoms, 70);
    auto random_state = [&](u32 limit)
    {
        std::vector<u32> s;
        while (s.size() < std::min(c.state_size, limit))
        {
            const u32 a = static_cast<u32>(rng() % limit);
            if (std::find(s.begin(), s.end(), a) == s.end())
                s.push_back(a);
        }
        std::sort(s.begin(), s.end());
        return s;
    };
    std::vector<u32> root = random_state(live);
    table.reserve(live);
    const std::vector<u64> rw = words_of(root, nw);
    EXPECT_EQ(table.mark_state(rw.data(), nw), ref.mark_state(root));
    std::vector<std::vector<u32>> pool{root};
    for (int step = 0; step < 3000; ++step)
    {
        if (live < c.atoms && step % 50 == 49)
            live = std::min(c.atoms, live + 37);
        const std::vector<u32>& parent = pool[rng() % pool.size()];
        // successor: drop a few atoms, add a few (some possibly already true)
        std::vector<u32> succ = parent;
        const u32 dels = static_cast<u32>(rng() % 3);
        for (u32 i = 0; i < dels && !succ.empty(); ++i)
            succ.erase(succ.begin() + static_cast<long>(rng() % succ.size()));
        std::vector<u32> add;
        const u32 adds = 1 + static_cast<u32>(rng() % 3);
        for (u32 i = 0; i < adds; ++i)
        {
            const u32 a = static_cast<u32>(rng() % live);
            if (std::find(succ.begin(), succ.end(), a) == succ.end())
            {
                succ.push_back(a);
                if (std::find(parent.begin(), parent.end(), a) == parent.end())
                    add.push_back(a);
            }
        }
        std::sort(succ.begin(), succ.end());
        table.reserve(live);
        const std::vector<u64> pw = words_of(parent, nw), sw = words_of(succ, nw);
        const bool expect = ref.transition(succ, add, false);
        ASSERT_EQ(table.test<false>(pw.data(), nw, sw.data(), nw, add), expect) << "step " << step << " (read-only)";
        const bool marked_ref = ref.transition(succ, add, true);
        ASSERT_EQ(table.test<true>(pw.data(), nw, sw.data(), nw, add), marked_ref) << "step " << step;
        if (marked_ref && pool.size() < 400)
            pool.push_back(succ);
        // every tuple containing an added atom is now marked
        ASSERT_FALSE(table.test<false>(pw.data(), nw, sw.data(), nw, add)) << "step " << step;
        if (c.k == 1)
        {
            ASSERT_FALSE(table.any_new1(add));
        }
    }
    // the switch to sparse happened where the budget says so
    if (c.k >= 2 && c.dense_bytes == 0)
    {
        EXPECT_FALSE(table.dense(2));
    }
    if (c.k >= 2 && c.dense_bytes > (u64{1} << 30))
    {
        EXPECT_TRUE(table.dense(2));
    }
}

INSTANTIATE_TEST_SUITE_P(Arities, NoveltyRandom,
                         ::testing::Values(Case{1, u64{1} << 40, 200, 12}, Case{2, u64{1} << 40, 200, 12}, Case{2, 0, 200, 12},
                                           Case{2, 4096, 300, 10},  // dense first, sparse after growth
                                           Case{3, u64{1} << 40, 120, 8}, Case{3, 0, 150, 8}, Case{3, 40000, 160, 8},
                                           Case{4, 0, 90, 7}, Case{4, u64{1} << 40, 64, 7}, Case{5, 0, 64, 6},
                                           // unranked levels (C(max_atoms, j) >= 2^64): dense, then packed sparse
                                           Case{4, 1000000, 150, 7, 200000}, Case{5, 2000000, 90, 6, 20000},
                                           Case{5, u64{1} << 40, 64, 6, 20000}),
                         [](const auto& info)
                         {
                             const Case& c = info.param;
                             return "k" + std::to_string(c.k) + "_atoms" + std::to_string(c.atoms) +
                                    (c.dense_bytes == 0 ? "_sparse" : c.dense_bytes > (u64{1} << 30) ? "_dense" : "_switch") +
                                    (c.max_atoms ? "_bound" + std::to_string(c.max_atoms) : "");
                         });

TEST(NoveltyTable, RejectsBadArity)
{
    EXPECT_THROW(NoveltyTable(0, 10), std::invalid_argument);
    EXPECT_THROW(NoveltyTable(k_max_arity + 1, 10), std::invalid_argument);
    EXPECT_THROW(NoveltyTable(5, 1u << 27), std::length_error);  // 5 x 28 bits do not pack into 128
    EXPECT_NO_THROW(NoveltyTable(2, 1u << 27, TableOptions{.max_dense_bytes = 0}));
}

TEST(NoveltyTable, TransitionWithoutAddedAtomsIsNotNovel)
{
    for (u32 k = 1; k <= 3; ++k)
    {
        NoveltyTable t(k, 128);
        const std::vector<u64> s = words_of({1, 5, 70}, 2), u = words_of({1, 70}, 2);
        EXPECT_TRUE(t.mark_state(s.data(), 2));
        EXPECT_FALSE(t.test<true>(s.data(), 2, u.data(), 2, {}));
        EXPECT_FALSE(t.mark_state(s.data(), 2));  // every tuple seen
    }
}

TEST(NoveltyTable, PairNoveltyNeedsANewPair)
{
    NoveltyTable t(2, 256);
    const std::vector<u64> root = words_of({0, 1}, 4);
    t.mark_state(root.data(), 4);
    // add 2 to {0, 1}: singleton 2 is new
    const std::vector<u64> a = words_of({0, 1, 2}, 4);
    const std::vector<u32> add2{2};
    EXPECT_TRUE(t.test<true>(root.data(), 4, a.data(), 4, add2));
    // from {0, 1}: {0, 2} again: every tuple containing 2 is seen
    const std::vector<u64> b = words_of({0, 2}, 4);
    EXPECT_FALSE(t.test<true>(root.data(), 4, b.data(), 4, add2));
    // {1, 3, 200}: 200 is a slot above the first capacity (grows) and new
    t.reserve(201);
    const std::vector<u64> c = words_of({1, 3, 200}, 4);
    const std::vector<u32> add3{3, 200};
    EXPECT_TRUE(t.test<true>(root.data(), 4, c.data(), 4, add3));
    // {0, 3} from {0}: pair {0, 3} is unseen although 3 is not new
    const std::vector<u64> p = words_of({0}, 4), d = words_of({0, 3}, 4);
    const std::vector<u32> add4{3};
    EXPECT_TRUE(t.test<false>(p.data(), 4, d.data(), 4, add4));
    EXPECT_TRUE(t.test<true>(p.data(), 4, d.data(), 4, add4));
    EXPECT_FALSE(t.test<false>(p.data(), 4, d.data(), 4, add4));
}
}  // namespace
