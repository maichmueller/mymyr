#include "mymyr/core/hash.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <unordered_set>
#include <vector>

using namespace mymyr;

TEST(Hash, StateHashIgnoresTrailingZeroWords)
{
    const std::vector<u64> a{0xdeadbeef, 42, 0, 0};
    const std::vector<u64> b{0xdeadbeef, 42};
    EXPECT_EQ(hash::state_words(a.data(), 4), hash::state_words(b.data(), 2));
    // but interior zeros matter
    const std::vector<u64> c{0xdeadbeef, 0, 42};
    EXPECT_NE(hash::state_words(b.data(), 2), hash::state_words(c.data(), 3));
}

TEST(Hash, EmptyAndZeroStatesHashEqual)
{
    const std::vector<u64> z{0, 0, 0};
    EXPECT_EQ(hash::state_words(z.data(), 3), hash::state_words(nullptr, 0));
}

TEST(Hash, NoCollisionsOnSingleBitStates)
{
    // every single-bit state over 64 words, and every pair of bits within 4 words, hash distinctly
    std::unordered_set<u64> seen;
    std::vector<u64> w(64, 0);
    for (u32 bit = 0; bit < 64 * 64; ++bit)
    {
        std::fill(w.begin(), w.end(), 0);
        bits::set(w.data(), bit);
        EXPECT_TRUE(seen.insert(hash::state_words(w.data(), 64)).second) << bit;
    }
    seen.clear();
    std::vector<u64> v(4, 0);
    for (u32 i = 0; i < 256; ++i)
        for (u32 j = i + 1; j < 256; ++j)
        {
            std::fill(v.begin(), v.end(), 0);
            bits::set(v.data(), i);
            bits::set(v.data(), j);
            EXPECT_TRUE(seen.insert(hash::state_words(v.data(), 4)).second) << i << "," << j;
        }
}

TEST(Hash, LowBitsAreWellDistributed)
{
    // open addressing uses the low bits: bucket counts of random 3-word states stay near uniform
    std::mt19937_64 rng(7);
    constexpr u32 buckets = 1024, n = 1 << 18;
    std::vector<u32> hist(buckets, 0);
    std::vector<u64> w(3);
    for (u32 i = 0; i < n; ++i)
    {
        for (auto& x : w)
            x = rng() & 0x00ff00ff00ff00ffULL;  // structured inputs
        ++hist[hash::state_words(w.data(), 3) & (buckets - 1)];
    }
    const double mean = double(n) / buckets;
    for (u32 c : hist)
        EXPECT_NEAR(c, mean, 6 * std::sqrt(mean));
}

TEST(Hash, Mix64IsABijectionOnSamples)
{
    std::unordered_set<u64> out;
    for (u64 i = 0; i < 100000; ++i)
        out.insert(hash::mix64(i));
    EXPECT_EQ(out.size(), 100000u);
}
