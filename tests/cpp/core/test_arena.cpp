#include "mymyr/core/arena.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace mymyr;

TEST(Arena, SegmentGeometry)
{
    SegmentedArena<u64> a(2);  // base 4: segments of 4, 8, 16, ... starting at 0, 4, 12, 28, ...
    EXPECT_EQ(a.segment_start(0), 0u);
    EXPECT_EQ(a.segment_start(1), 4u);
    EXPECT_EQ(a.segment_start(2), 12u);
    EXPECT_EQ(a.segment_start(3), 28u);
    for (u64 i = 0; i < 1000; ++i)
    {
        const auto [k, off] = a.locate(i);
        EXPECT_GE(i, a.segment_start(k));
        EXPECT_LT(off, a.segment_size(k));
        EXPECT_EQ(a.segment_start(k) + off, i);
    }
}

TEST(Arena, AddressesAreStableAcrossGrowth)
{
    SegmentedArena<u64> a(3);
    std::vector<const u64*> addr;
    for (u64 i = 0; i < 5000; ++i)
    {
        a.push_back(i * 3);
        addr.push_back(&a[i]);
    }
    for (u64 i = 0; i < 5000; ++i)
    {
        EXPECT_EQ(addr[i], &a[i]);
        EXPECT_EQ(a[i], i * 3);
    }
}

TEST(Arena, ContiguousRecordsNeverStraddleSegments)
{
    SegmentedArena<u64> a(4);  // base 16
    std::vector<std::pair<u64, u64>> recs;
    for (u64 r = 0; r < 500; ++r)
    {
        const u64 n = 1 + (r * 7) % 16;
        const u64 start = a.alloc_contiguous(n);
        const auto [k0, off0] = a.locate(start);
        const auto [k1, off1] = a.locate(start + n - 1);
        EXPECT_EQ(k0, k1);
        for (u64 j = 0; j < n; ++j)
            a[start + j] = r * 100 + j;
        recs.emplace_back(start, n);
    }
    for (u64 r = 0; r < recs.size(); ++r)
    {
        const u64* p = a.ptr(recs[r].first);
        for (u64 j = 0; j < recs[r].second; ++j)
            EXPECT_EQ(p[j], r * 100 + j);  // one contiguous run
    }
    EXPECT_THROW(a.alloc_contiguous(17), std::length_error);
}

TEST(Arena, AppendAndRunsCoverTheTail)
{
    SegmentedArena<u32> a(2);
    std::vector<u32> src(100);
    for (u32 i = 0; i < 100; ++i)
        src[i] = i;
    a.append(src.data(), 100);
    ASSERT_EQ(a.size(), 100u);
    std::vector<u32> copied;
    a.for_each_run(37, 100, [&](u64 lo, const u32* p, u64 n) {
        EXPECT_EQ(p[0], lo);
        copied.insert(copied.end(), p, p + n);
    });
    ASSERT_EQ(copied.size(), 63u);
    for (u32 i = 0; i < 63; ++i)
        EXPECT_EQ(copied[i], 37 + i);
}

TEST(Arena, ConcurrentReadersSeePublishedPrefix)
{
    // one writer appends; readers only read indices published through a release store (the store's CAS table
    // plays this role in production).
    SegmentedArena<u64> a(6);
    std::atomic<u64> published{0};
    constexpr u64 n = 200000;
    std::atomic<bool> bad{false};
    std::thread writer([&] {
        for (u64 i = 0; i < n; ++i)
        {
            a.push_back(i ^ 0x5555);
            published.store(i + 1, std::memory_order_release);
        }
    });
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            u64 seen = 0;
            while (seen < n)
            {
                const u64 p = published.load(std::memory_order_acquire);
                for (u64 i = seen; i < p; ++i)
                    if (a[i] != (i ^ 0x5555))
                        bad = true;
                seen = p;
            }
        });
    writer.join();
    for (auto& r : readers)
        r.join();
    EXPECT_FALSE(bad.load());
}

TEST(Arena, MoveTransfersOwnership)
{
    SegmentedArena<u64> a(4);
    for (u64 i = 0; i < 100; ++i)
        a.push_back(i);
    const u64* p = &a[50];
    SegmentedArena<u64> b(std::move(a));
    EXPECT_EQ(b.size(), 100u);
    EXPECT_EQ(&b[50], p);
    EXPECT_EQ(a.size(), 0u);
}
