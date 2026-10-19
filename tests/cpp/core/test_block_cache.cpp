// The recycling allocator of large output blocks (core/block_cache.hpp).

#include "mymyr/core/block_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#if defined(__SANITIZE_ADDRESS__)
#define MYMYR_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MYMYR_TEST_ASAN 1
#endif
#endif
#if defined(MYMYR_TEST_ASAN)
#include <sanitizer/asan_interface.h>
#endif

using namespace mymyr;

namespace
{
constexpr u64 k_mib = u64{1} << 20;

#if defined(__linux__)
/// The resident set of this process in bytes.
u64 resident_bytes()
{
    std::FILE* file = std::fopen("/proc/self/statm", "r");
    long pages = 0, resident = 0;
    if (file)
    {
        if (std::fscanf(file, "%ld %ld", &pages, &resident) != 2)
            resident = 0;
        std::fclose(file);
    }
    return static_cast<u64>(resident) * static_cast<u64>(sysconf(_SC_PAGESIZE));
}
#endif
}  // namespace

TEST(BlockCache, BlocksAreAlignedAndLargeEnough)
{
    for (u64 bytes : {u64{0}, u64{1}, u64{64}, u64{1000}, k_cached_block_min, k_cached_block_min + 1,
                      k_thread_block_max, k_thread_block_max + 1, u64{9} << 20})
    {
        const HostBlock block = acquire_host_block(bytes);
        ASSERT_NE(block.get(), nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block.get()) % 64, 0u);
        EXPECT_GE(block.capacity(), bytes);
        std::memset(block.get(), 0xab, bytes);
    }
}

TEST(BlockCache, ABlockOwnsItsPointerAndCapacityTogether)
{
    HostBlock a = acquire_host_block(300000);
    const std::byte* data = a.get();
    const u64 capacity = a.capacity();
    HostBlock b = std::move(a);
    EXPECT_FALSE(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(a.capacity(), 0u);
    EXPECT_EQ(b.get(), data);
    EXPECT_EQ(b.capacity(), capacity);
    HostBlock c;
    c = std::move(b);
    EXPECT_EQ(c.get(), data);
    c.reset();
    EXPECT_FALSE(c);
    EXPECT_EQ(c.capacity(), 0u);
    c.reset();  // an empty block releases nothing
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

TEST(BlockCache, AThreadReusesItsOwnMidSizedBlocks)
{
    trim_block_cache();
    std::byte* first = nullptr;
    {
        HostBlock block = acquire_host_block(300000);
        first = block.get();
    }
    EXPECT_GE(block_cache_bytes(), 300000u);
    EXPECT_EQ(acquire_host_block(280000).get(), first);
}

TEST(BlockCache, SmallBlocksHaveExactSizes)
{
    trim_block_cache();
    for (u64 bytes : {u64{1}, u64{64}, u64{65}, u64{1000}, u64{12288}, u64{100000}, k_cached_block_min})
    {
        const u64 expected = (std::max<u64>(bytes, 64) + 63) / 64 * 64;
        const HostBlock block = acquire_host_block(bytes);
        EXPECT_EQ(block.capacity(), expected) << bytes;
    }
}

TEST(BlockCache, ASmallBlockIsReusedByTheNextRequestOfAboutItsSize)
{
    trim_block_cache();
    std::byte* first = nullptr;
    {
        const HostBlock block = acquire_host_block(12288);
        first = block.get();
    }
    EXPECT_EQ(block_cache_bytes(), 12288u);
    for (const u64 bytes : {u64{12288}, u64{12000}, u64{10000}})  // up to a quarter larger than the request
    {
        HostBlock again = acquire_host_block(bytes);
        EXPECT_EQ(again.get(), first) << bytes;
    }
    EXPECT_EQ(block_cache_bytes(), 12288u);
    HostBlock smaller = acquire_host_block(8000);  // more than a quarter smaller than the cached block
    EXPECT_NE(smaller.get(), first);
    EXPECT_EQ(smaller.capacity(), 8000u);
}

TEST(BlockCache, MidSizedBlocksWasteAtMostAQuarter)
{
    for (u64 bytes = k_cached_block_min + 1; bytes <= k_thread_block_max; bytes += bytes / 7 + 1)
    {
        const HostBlock block = acquire_host_block(bytes);
        EXPECT_GE(block.capacity(), bytes);
        EXPECT_LE(block.capacity(), bytes + bytes / 4) << bytes;
    }
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
    EXPECT_LT(small.capacity(), big);
}

TEST(BlockCache, TheCacheStaysWithinItsLimits)
{
    trim_block_cache();
    std::vector<HostBlock> held;
    for (int i = 0; i < 12; ++i)
        held.push_back(acquire_host_block(u64{48} << 20));
    held.clear();
    EXPECT_LE(block_cache_bytes(), k_block_cache_limit);
    trim_block_cache();
    EXPECT_EQ(block_cache_bytes(), 0u);
    // no block above k_global_block_max is kept
    { HostBlock huge = acquire_host_block(k_global_block_max + k_mib); }
    EXPECT_EQ(block_cache_bytes(), 0u);
    // at most k_global_blocks blocks are kept
    for (usize i = 0; i < k_global_blocks + 3; ++i)
        held.push_back(acquire_host_block(5 * k_mib));
    held.clear();
    EXPECT_LE(block_cache_bytes(), k_global_blocks * 5 * k_mib + k_thread_blocks * k_thread_block_max);
    trim_block_cache();
}

TEST(BlockCache, ABlockThatNoAcquireTakesIsReleasedAfterABoundedNumberOfAcquires)
{
    trim_block_cache();
    { HostBlock block = acquire_host_block(u64{12} << 20); }
    ASSERT_GE(block_cache_bytes(), u64{12} << 20);
    // acquires of other sizes do not take it; each counts towards its idle time
    for (u64 i = 0; i < k_idle_acquires / 2; ++i)
        HostBlock other = acquire_host_block(k_mib);
    EXPECT_GE(block_cache_bytes(), u64{12} << 20);
    for (u64 i = 0; i < k_idle_acquires / 2 + 2; ++i)
        HostBlock other = acquire_host_block(k_mib);
    EXPECT_LT(block_cache_bytes(), u64{12} << 20);
    trim_block_cache();
}

TEST(BlockCache, SmallAcquiresDoNotAgeTheCachedBlocks)
{
    trim_block_cache();
    { HostBlock block = acquire_host_block(u64{12} << 20); }
    for (u64 i = 0; i < 4 * k_idle_acquires; ++i)
        HostBlock other = acquire_host_block(k_cached_block_min);
    EXPECT_GE(block_cache_bytes(), u64{12} << 20);
    trim_block_cache();
}

TEST(BlockCache, ABlockThatIsTakenAgainStartsItsIdleTimeAnew)
{
    trim_block_cache();
    std::byte* first = nullptr;
    {
        HostBlock block = acquire_host_block(u64{12} << 20);
        first = block.get();
    }
    for (u64 round = 0; round < 3 * k_idle_acquires; ++round)
    {
        HostBlock block = acquire_host_block(u64{12} << 20);
        ASSERT_EQ(block.get(), first) << round;
    }
    EXPECT_GE(block_cache_bytes(), u64{12} << 20);
    trim_block_cache();
}

TEST(BlockCache, AnIdleBlockIsReleasedByAnotherThreadsAcquires)
{
    trim_block_cache();
    { HostBlock block = acquire_host_block(u64{12} << 20); }
    // the shared tier belongs to every thread: another thread's acquires age the block
    std::thread other(
        [&]
        {
            for (u64 i = 0; i < k_idle_acquires + 2; ++i)
                HostBlock block = acquire_host_block(k_mib);
        });
    other.join();
    EXPECT_LT(block_cache_bytes(), u64{12} << 20);
    trim_block_cache();
}

TEST(BlockCache, TrimReturnsTheMemoryOfCachedBlocks)
{
    trim_block_cache();
    constexpr u64 k_block = u64{48} << 20;
    constexpr int k_blocks = 4;
    {
        std::vector<HostBlock> held;
        for (int i = 0; i < k_blocks; ++i)
        {
            held.push_back(acquire_host_block(k_block));
            std::memset(held.back().get(), 0x5a, k_block);
        }
    }
    EXPECT_GE(block_cache_bytes(), k_blocks * k_block);
#if defined(__linux__)
    const u64 before = resident_bytes();
#endif
    trim_block_cache();
    EXPECT_EQ(block_cache_bytes(), 0u);
#if defined(__linux__)
    const u64 after = resident_bytes();
    EXPECT_GT(before, after);
    EXPECT_GE(before - after, k_blocks * k_block / 2) << "resident before " << before << ", after " << after;
#endif
}

TEST(BlockCache, ARequestThatOverflowsThrows)
{
    EXPECT_THROW((void)acquire_host_block(~u64{0}), std::bad_alloc);
    EXPECT_THROW((void)acquire_host_block(~u64{0} - 10), std::bad_alloc);
    EXPECT_THROW((void)acquire_host_block((u64{1} << 62) + 1), std::bad_alloc);
    EXPECT_THROW((void)acquire_host_block(~u64{0} - k_mib), std::bad_alloc);
    // the cache still works
    const HostBlock block = acquire_host_block(5 * k_mib);
    EXPECT_GE(block.capacity(), 5 * k_mib);
}

#if defined(MYMYR_TEST_ASAN)
TEST(BlockCache, CachedBlocksArePoisonedUntilTheyAreHandedOutAgain)
{
    for (const u64 bytes : {u64{12288}, u64{300000}, u64{12} << 20})
    {
        trim_block_cache();
        std::byte* data = nullptr;
        u64 capacity = 0;
        {
            HostBlock block = acquire_host_block(bytes);
            data = block.get();
            capacity = block.capacity();
            EXPECT_EQ(__asan_region_is_poisoned(data, capacity), nullptr);
        }
        ASSERT_GE(block_cache_bytes(), bytes);
        EXPECT_NE(__asan_address_is_poisoned(data), 0) << bytes;
        EXPECT_NE(__asan_address_is_poisoned(data + capacity / 2), 0) << bytes;
        EXPECT_NE(__asan_address_is_poisoned(data + capacity - 1), 0) << bytes;
        HostBlock again = acquire_host_block(bytes);
        ASSERT_EQ(again.get(), data);
        EXPECT_EQ(__asan_region_is_poisoned(data, capacity), nullptr) << bytes;
        std::memset(data, 1, capacity);
        again.reset();
        trim_block_cache();
        if (bytes > k_cached_block_min)  // a mapping is gone; a small block is freed memory that the sanitizer tracks
        {
            EXPECT_EQ(__asan_region_is_poisoned(data, capacity), nullptr) << "unmapped memory has no shadow: " << bytes;
        }
    }
}

TEST(BlockCacheDeathTest, ReadingAReleasedBlockIsReported)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    for (const u64 bytes : {u64{12288}, u64{300000}, u64{12} << 20})
    {
        EXPECT_DEATH(
            {
                trim_block_cache();
                volatile std::byte* data = nullptr;
                {
                    HostBlock block = acquire_host_block(bytes);
                    data = block.get();
                    data[0] = std::byte{1};
                }
                volatile std::byte read = data[100];  // the block sits in the cache
                (void)read;
            },
            "AddressSanitizer");
    }
}
#endif

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
    EXPECT_LE(block_cache_bytes(), k_block_cache_limit + k_thread_block_max * k_thread_blocks * (k_threads + 1));
}

TEST(BlockCache, AForkedChildAcquiresLargeBlocksWhileAnotherThreadUsesTheCache)
{
    trim_block_cache();
    std::atomic<bool> stop{false};
    std::thread user(
        [&]
        {
            while (!stop)
            {
                HostBlock a = acquire_host_block(u64{6} << 20);
                HostBlock b = acquire_host_block(u64{9} << 20);
                std::memset(a.get(), 1, 4096);
                std::memset(b.get(), 2, 4096);
            }
        });
    constexpr int k_forks = 24;
    int failed = 0;
    for (int i = 0; i < k_forks; ++i)
    {
        const pid_t child = ::fork();
        ASSERT_GE(child, 0);
        if (child == 0)
        {
            int status = 1;
            ::alarm(30);  // a child that waits on a lock it inherited ends here
            {
                HostBlock block = acquire_host_block(u64{12} << 20);
                std::memset(block.get(), 7, block.capacity());
                HostBlock other = acquire_host_block(u64{7} << 20);
                std::memset(other.get(), 8, other.capacity());
                status = block.get()[block.capacity() - 1] == std::byte{7} ? 0 : 2;
            }
            trim_block_cache();
            ::_exit(status);
        }
        int wait_status = 0;
        ASSERT_EQ(::waitpid(child, &wait_status, 0), child);
        if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0)
            ++failed;
    }
    stop = true;
    user.join();
    EXPECT_EQ(failed, 0) << "of " << k_forks << " forked children";
    trim_block_cache();
}
