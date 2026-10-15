// The recycling allocator of large output blocks (core/block_cache.hpp).

#include "mymyr/core/block_cache.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace mymyr;

TEST(BlockCache, BlocksAreAlignedAndLargeEnough)
{
    for (u64 bytes : {u64{1}, u64{64}, u64{1000}, u64{k_thread_block_max}, u64{k_thread_block_max} + 1, u64{9} << 20})
    {
        const HostBlock block = acquire_host_block(bytes);
        ASSERT_NE(block.get(), nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block.get()) % 64, 0u);
        EXPECT_GE(block.get_deleter().capacity, bytes);
        std::memset(block.get(), 0xab, bytes);
    }
}

TEST(BlockCache, AReleasedLargeBlockIsReused)
{
    trim_block_cache();
    const u64 bytes = u64{12} << 20;
    std::byte* first = nullptr;
    {
        HostBlock block = acquire_host_block(bytes);
        first = block.get();
    }
    EXPECT_GE(block_cache_bytes(), bytes);
    HostBlock again = acquire_host_block(bytes);
    EXPECT_EQ(again.get(), first);
    EXPECT_EQ(block_cache_bytes(), 0u);
}

TEST(BlockCache, ASmallThreadReusesItsOwnBlocks)
{
    trim_block_cache();
    std::byte* first = nullptr;
    {
        HostBlock block = acquire_host_block(100000);
        first = block.get();
    }
    EXPECT_EQ(acquire_host_block(90000).get(), first);
}

TEST(BlockCache, ABlockMuchLargerThanTheRequestIsNotHandedOut)
{
    trim_block_cache();
    const u64 big = u64{64} << 20;
    { HostBlock block = acquire_host_block(big); }
    const u64 cached = block_cache_bytes();
    EXPECT_GE(cached, big);
    HostBlock small = acquire_host_block(u64{8} << 20);
    EXPECT_EQ(block_cache_bytes(), cached);
    EXPECT_LT(small.get_deleter().capacity, big);
}

TEST(BlockCache, TheCacheStaysWithinItsLimit)
{
    trim_block_cache();
    std::vector<HostBlock> held;
    for (int i = 0; i < 12; ++i)
        held.push_back(acquire_host_block(u64{48} << 20));
    held.clear();
    EXPECT_LE(block_cache_bytes(), k_block_cache_limit);
    trim_block_cache();
    EXPECT_EQ(block_cache_bytes(), 0u);
}

TEST(BlockCache, BlocksAreReleasedOnOtherThreadsThanTheirAcquirers)
{
    trim_block_cache();
    constexpr int k_threads = 8;
    constexpr int k_rounds = 400;
    std::atomic<int> failures{0};
    std::mutex mutex;
    std::vector<HostBlock> stash;
    std::vector<std::thread> threads;
    for (int t = 0; t < k_threads; ++t)
        threads.emplace_back(
            [&, t]
            {
                for (int r = 0; r < k_rounds; ++r)
                {
                    const u64 bytes = (r % 3 == 0) ? (u64{6} << 20) : (u64{1} << 18) + static_cast<u64>(r) * 64;
                    HostBlock block = acquire_host_block(bytes);
                    const int tag = (t * 31 + r) & 0x7f;
                    std::memset(block.get(), tag, bytes);
                    if (block.get()[0] != static_cast<std::byte>(tag) || block.get()[bytes - 1] != static_cast<std::byte>(tag))
                        ++failures;
                    HostBlock other;
                    {
                        const std::scoped_lock lock(mutex);
                        stash.push_back(std::move(block));
                        if (stash.size() > 6)
                        {
                            other = std::move(stash.front());
                            stash.erase(stash.begin());
                        }
                    }
                    // `other` is released here, usually on a thread that did not acquire it
                }
            });
    for (auto& th : threads)
        th.join();
    stash.clear();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_LE(block_cache_bytes(), k_block_cache_limit + k_thread_block_max * 4 * (k_threads + 1));
}
