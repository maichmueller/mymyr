#include "mymyr/core/per_thread.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <set>
#include <thread>
#include <vector>

using namespace mymyr;

namespace
{
struct Scratch
{
    static inline std::atomic<int> alive{0};
    std::vector<u64> buf = std::vector<u64>(16, 0);
    Scratch() { ++alive; }
    ~Scratch() { --alive; }
};
auto make_scratch = [] { return std::make_unique<Scratch>(); };
}  // namespace

TEST(PerThread, SameThreadGetsSameInstance)
{
    PerThread<Scratch> pt;
    Scratch& a = pt.local(make_scratch);
    Scratch& b = pt.local(make_scratch);
    EXPECT_EQ(&a, &b);
    EXPECT_EQ(pt.created(), 1u);
}

TEST(PerThread, ThreadsGetDistinctInstances)
{
    PerThread<Scratch> pt;
    constexpr int T = 8;
    std::vector<Scratch*> seen(T);
    std::vector<std::thread> ths;
    std::atomic<int> ready{0};
    for (int t = 0; t < T; ++t)
        ths.emplace_back([&, t] {
            seen[t] = &pt.local(make_scratch);
            ++ready;
            while (ready.load() < T)  // keep all threads alive together so none is reused
                std::this_thread::yield();
        });
    for (auto& th : ths)
        th.join();
    EXPECT_EQ(std::set<Scratch*>(seen.begin(), seen.end()).size(), size_t(T));
    EXPECT_EQ(pt.created(), size_t(T));
}

TEST(PerThread, ExitedThreadsReturnInstancesForReuse)
{
    PerThread<Scratch> pt;
    for (int round = 0; round < 20; ++round)
    {
        std::thread th([&] { pt.local(make_scratch).buf[0] += 1; });
        th.join();
    }
    // sequential short-lived threads reuse one released instance instead of creating 20
    EXPECT_EQ(pt.created(), 1u);
}

TEST(PerThread, OwnersDieWithoutLeakingAndCachesSelfPrune)
{
    const int before = Scratch::alive.load();
    for (int i = 0; i < 1000; ++i)
    {
        PerThread<Scratch> pt;  // thousands of short-lived owners on one thread (multi-instance RL)
        pt.local(make_scratch).buf[0] = i;
    }
    EXPECT_EQ(Scratch::alive.load(), before);
    // the cache of this thread holds at most the entries created since the last prune
    PerThread<Scratch> last;
    last.local(make_scratch);
    EXPECT_LE(detail::PerThreadCache::get().entries.size(), 2u);
}

TEST(PerThread, OwnerDestroyedWhileThreadStillCachesIt)
{
    const int before = Scratch::alive.load();
    std::atomic<bool> go{false}, used{false};
    auto pt = std::make_unique<PerThread<Scratch>>();
    std::thread th([&] {
        pt->local(make_scratch);
        used = true;
        while (!go.load())
            std::this_thread::yield();
        // thread exits after the owner died: its cache must not touch freed memory
    });
    while (!used.load())
        std::this_thread::yield();
    pt.reset();
    go = true;
    th.join();
    EXPECT_EQ(Scratch::alive.load(), before);
}
