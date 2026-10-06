#include "mymyr/core/bitset.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mymyr;

TEST(Bitset, WordLayoutFollowsTheStateContract)
{
    // slot i lives in word i >> 6, bit i & 63
    std::vector<u64> w(2, 0);
    bits::set(w.data(), 0);
    bits::set(w.data(), 63);
    bits::set(w.data(), 64);
    EXPECT_EQ(w[0], (u64{1} << 63) | 1u);
    EXPECT_EQ(w[1], u64{1});
    EXPECT_TRUE(bits::test(w.data(), 2, 64));
    EXPECT_FALSE(bits::test(w.data(), 2, 65));
    EXPECT_FALSE(bits::test(w.data(), 2, 1000));  // beyond the stored words reads as zero
}

TEST(Bitset, MissingWordsCompareAsZero)
{
    const std::vector<u64> a{5, 0, 0};
    const std::vector<u64> b{5};
    const std::vector<u64> c{5, 1};
    EXPECT_TRUE(bits::equal(a.data(), 3, b.data(), 1));
    EXPECT_TRUE(bits::equal(b.data(), 1, a.data(), 3));
    EXPECT_FALSE(bits::equal(a.data(), 3, c.data(), 2));
    EXPECT_EQ(bits::trimmed_size(a.data(), 3), 1u);
    EXPECT_EQ(bits::trimmed_size(a.data(), 0), 0u);
}

TEST(Bitset, SubsetAndIntersection)
{
    const std::vector<u64> small{0b0101};
    const std::vector<u64> big{0b1101, 7};
    EXPECT_TRUE(bits::subset(small.data(), 1, big.data(), 2));
    EXPECT_FALSE(bits::subset(big.data(), 2, small.data(), 1));
    EXPECT_TRUE(bits::intersects(small.data(), 1, big.data(), 2));
    const std::vector<u64> other{0b0010};
    EXPECT_FALSE(bits::intersects(small.data(), 1, other.data(), 1));
}

TEST(Bitset, ForEachVisitsSetBitsInOrder)
{
    bits::Bitset s;
    for (u64 b : {3u, 64u, 65u, 200u})
        s.set(b);
    std::vector<u64> seen;
    s.for_each([&](u64 b) { seen.push_back(b); });
    EXPECT_EQ(seen, (std::vector<u64>{3, 64, 65, 200}));
    EXPECT_EQ(s.count(), 4u);
    s.reset(64);
    EXPECT_FALSE(s.test(64));
    EXPECT_EQ(s.count(), 3u);
}

TEST(Bitset, OwningEqualityIgnoresCapacity)
{
    bits::Bitset a(10), b(1000);
    a.set(7);
    b.set(7);
    EXPECT_EQ(a, b);
    b.set(999);
    EXPECT_NE(a, b);
}
