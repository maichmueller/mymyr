// Thread counts and exception-safe thread creation (core/threads.hpp, core/team.hpp, core/thread_pool.hpp). The
// system's refusal to start a thread is simulated by interposing pthread_create in this executable (Linux): with
// fail_from(k), creations k, k + 1, ... from now on fail with EAGAIN.

#include "mymyr/core/team.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/core/threads.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/rl/pool.hpp"
#include "mymyr/rl/task_table.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>

namespace
{
std::atomic<long> g_creations_left{-1};  // -1: never fail
}

extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg)
{
    using Real = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static const Real real = reinterpret_cast<Real>(dlsym(RTLD_NEXT, "pthread_create"));
    long left = g_creations_left.load();
    while (left > 0 && !g_creations_left.compare_exchange_weak(left, left - 1))
    {
    }
    if (left == 0)
        return EAGAIN;
    return real(thread, attr, start, arg);
}

namespace
{
/// Thread creations succeed k more times, then fail until the guard ends.
struct FailFrom
{
    explicit FailFrom(long k) { g_creations_left.store(k); }
    ~FailFrom() { g_creations_left.store(-1); }
    FailFrom(const FailFrom&) = delete;
    FailFrom& operator=(const FailFrom&) = delete;
};
}  // namespace
#endif

using namespace mymyr;

namespace
{
std::string task_path(const std::string& name) { return std::string(MYMYR_TEST_DATA_DIR) + "/tasks/" + name + ".txt"; }

template<class F>
std::string message_of(F&& f)
{
    try
    {
        f();
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    return {};
}
}  // namespace

TEST(Threads, CountsAreResolvedAndBounded)
{
    EXPECT_EQ(resolve_threads(0), hardware_threads());
    EXPECT_EQ(resolve_threads(3), 3u);
    EXPECT_EQ(resolve_threads(max_threads()), max_threads());
    EXPECT_GE(max_threads(), 64u);
    EXPECT_GE(max_threads(), 4 * hardware_threads());
    EXPECT_THROW((void)resolve_threads(max_threads() + 1), std::invalid_argument);
    EXPECT_THROW((void)resolve_threads(u64{1} << 40), std::invalid_argument);
    const std::string m = message_of([] { (void)resolve_threads(1'000'000, "num_threads"); });
    EXPECT_NE(m.find("num_threads=1000000"), std::string::npos) << m;
    EXPECT_NE(m.find(std::to_string(max_threads())), std::string::npos) << m;
    EXPECT_THROW(Team(max_threads() + 1), std::invalid_argument);
    EXPECT_THROW(ThreadPool(max_threads() + 1), std::invalid_argument);
}

TEST(Threads, SearchesAndGeneratorsRefuseTooManyThreads)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    const u32 many = max_threads() + 1;
    EXPECT_THROW((void)brfs(*task, {.threads = many}), std::invalid_argument);
    search::IwOptions iw;
    iw.threads = many;
    EXPECT_THROW((void)search::iw(*task, iw), std::invalid_argument);
    datasets::StateSpaceOptions so;
    so.threads = many;
    EXPECT_THROW((void)datasets::generate_state_space(task, so), std::invalid_argument);
    EXPECT_THROW(rl::CpuEnvPool(rl::TaskTable::single(task), {}, 4, {.threads = many}), std::invalid_argument);
}

#if defined(__linux__)
TEST(Threads, TeamAndPoolStopTheStartedThreadsWhenOneCannotStart)
{
    for (long k : {0L, 1L, 5L})
    {
        SCOPED_TRACE(k);
        {
            const FailFrom fail(k);
            EXPECT_THROW(Team(12), ThreadStartError);
        }
        {
            const FailFrom fail(k);
            EXPECT_THROW(ThreadPool(12), ThreadStartError);
        }
        const std::string m = message_of([&] {
            const FailFrom fail(k);
            ThreadPool p(12);
        });
        EXPECT_NE(m.find("could not start worker thread " + std::to_string(k + 1) + " of 11"), std::string::npos) << m;
    }
    // the process is intact: new teams and pools work
    Team team(4);
    std::atomic<u32> ran{0};
    team.run([&](u32) { ran.fetch_add(1); });
    EXPECT_EQ(ran.load(), 4u);
    ThreadPool pool(4);
    pool.run([&](u32) { ran.fetch_add(1); });
    EXPECT_EQ(ran.load(), 8u);
}

TEST(Threads, ParallelSearchesAndPoolsReportARefusedThread)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    const BrfsResult ref = brfs(*task, {.threads = 4, .stop_at_goal = true});
    for (long k : {0L, 2L})
    {
        SCOPED_TRACE(k);
        {
            const FailFrom fail(k);
            EXPECT_THROW((void)brfs(*task, {.threads = 4, .stop_at_goal = true}), ThreadStartError);
        }
        {
            const FailFrom fail(k);
            datasets::StateSpaceOptions so;
            so.threads = 4;
            EXPECT_THROW((void)datasets::generate_state_space(task, so), ThreadStartError);
        }
        {
            const FailFrom fail(k);
            EXPECT_THROW(rl::CpuEnvPool(rl::TaskTable::single(task), {}, 8, {.threads = 4}), ThreadStartError);
        }
        {
            const FailFrom fail(k);
            const std::vector<TaskPtr> tasks(6, task);
            EXPECT_THROW((void)datasets::generate_state_spaces(tasks, {}, 4), ThreadStartError);
        }
    }
    const BrfsResult again = brfs(*task, {.threads = 4, .stop_at_goal = true});
    EXPECT_EQ(again.plan.size(), ref.plan.size());
}
#endif
