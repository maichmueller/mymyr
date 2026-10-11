// Once: a throwing initializer reaches its caller and runs again on the next call; a completed one never runs again,
// also when many threads call at once.

#include "mymyr/core/once.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mymyr;

TEST(Once, AThrowingInitializerReachesTheCallerAndRunsAgain)
{
    Once once;
    int runs = 0;
    auto failing = [&] {
        ++runs;
        throw std::invalid_argument("refused");
    };
    EXPECT_THROW(once.call(failing), std::invalid_argument);
    EXPECT_FALSE(once.done());
    EXPECT_THROW(once.call(failing), std::invalid_argument);
    EXPECT_EQ(runs, 2);
    once.call([&] { ++runs; });
    EXPECT_TRUE(once.done());
    once.call(failing);  // done: not run
    EXPECT_EQ(runs, 3);
}

TEST(Once, ConcurrentCallersRunTheInitializerOnceAndSeeItsResult)
{
    for (int round = 0; round < 20; ++round)
    {
        Once once;
        std::atomic<int> runs{0}, failures{0};
        int value = 0;
        std::atomic<int> seen{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t)
            threads.emplace_back([&] {
                try
                {
                    once.call([&] {
                        // the first two attempts fail; a later caller completes it
                        if (runs.fetch_add(1) < 2)
                            throw std::runtime_error("transient");
                        value = 42;
                    });
                    seen += value == 42;
                }
                catch (const std::runtime_error&)
                {
                    ++failures;
                }
            });
        for (std::thread& th : threads)
            th.join();
        EXPECT_EQ(runs.load(), 3);
        EXPECT_EQ(failures.load(), 2);
        EXPECT_EQ(seen.load(), 6);
        EXPECT_TRUE(once.done());
    }
}
