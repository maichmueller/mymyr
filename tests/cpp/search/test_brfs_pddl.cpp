// BrFS on tasks built from PDDL text: the concurrent store grows while the threads are inside expansions.

#include "../support/branching.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
std::shared_ptr<const Task> branching_task(u32 modes, u32 n)
{
    return Task::create(*frontend::Domain::from_string(test::branching_domain(), "d.pddl")
                             ->instantiate_string(test::branching_problem(modes, n), "p.pddl"));
}

bool replays(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return false;
        s = succ.apply(s, a.label());
    }
    return task.is_goal(s);
}
}  // namespace

TEST(BrfsPddl, ConcurrentStoreGrowsInsideAnExpansion)
{
    // the threads stop inserting as soon as the table wants to grow, also in the middle of an expansion: one
    // expansion (modes 1) or the threads' concurrent expansions (modes 8) insert more states than the initial table
    // holds; the results equal the flat store's
    struct Case
    {
        u32 modes, n;
    };
    for (const Case c : {Case{1, 65}, Case{8, 35}})
    {
        const auto task = branching_task(c.modes, c.n);
        const u64 states = test::branching_states(c.modes, c.n);
        BrfsOptions bo;
        bo.stop_at_goal = false;
        bo.fingerprint = true;
        bo.store = BrfsOptions::Store::Flat;
        const BrfsResult ref = brfs(*task, bo);
        ASSERT_EQ(ref.states, states);
        bo.store = BrfsOptions::Store::Concurrent;
        for (const u32 T : {2u, 8u})
        {
            const std::string what = "modes " + std::to_string(c.modes) + " T=" + std::to_string(T);
            bo.threads = T;
            bo.max_states = ~u64{0};
            bo.stop_at_goal = false;
            const BrfsResult r = brfs(*task, bo);
            EXPECT_TRUE(r.exhausted) << what;
            EXPECT_EQ(r.states, states) << what;
            EXPECT_EQ(r.generated, ref.generated) << what;
            EXPECT_EQ(r.fingerprint, ref.fingerprint) << what;
            // a limit inside the wide layer: exactly that many states
            bo.max_states = states - states / 3;
            const BrfsResult bounded = brfs(*task, bo);
            EXPECT_EQ(bounded.status, search::SearchStatus::OutOfStates) << what;
            EXPECT_EQ(bounded.states, bo.max_states) << what;
            bo.max_states = ~u64{0};
            bo.stop_at_goal = true;
            const BrfsResult solved = brfs(*task, bo);
            ASSERT_TRUE(solved.solved) << what;
            EXPECT_EQ(solved.plan.size(), 2u) << what;
            EXPECT_TRUE(replays(*task, solved.plan)) << what;
        }
    }
}

TEST(BrfsPddl, ConcurrentStoreDoesNotGrowAfterTheStateLimit)
{
    // the initial table of 2^18 entries wants to grow once the states counted in batches of 64 per thread exceed 60%
    // of it; with one thread that is exactly `grow` states. A search stopped by max_states keeps its table; one that
    // gets there earlier grows it, which adds the 2^18 entries to memory_bytes.
    // memory_bytes is compared with one thread only: it also counts the arena of every thread that stored a state,
    // allocated at that thread's first state, so with several threads it depends on which threads took the states of
    // the first layers. With 8 threads the same limits only have to stop the search at exactly max_states.
    constexpr u64 cap = u64{1} << 18;
    u64 grow = 64;
    while (grow * 10 <= cap * 6)
        grow += 64;
    const auto task = branching_task(1, 55);
    ASSERT_GT(test::branching_states(1, 55), grow + 3);
    BrfsOptions bo;
    bo.stop_at_goal = false;
    bo.store = BrfsOptions::Store::Concurrent;
    const auto bytes_at = [&](u64 max_states)
    {
        bo.max_states = max_states;
        const BrfsResult r = brfs(*task, bo);
        EXPECT_EQ(r.status, search::SearchStatus::OutOfStates) << "max_states " << max_states;
        EXPECT_EQ(r.states, max_states);
        return r.memory_bytes;
    };
    bo.threads = 1;
    const u64 below = bytes_at(grow - 1);
    EXPECT_EQ(bytes_at(grow), below);
    EXPECT_EQ(bytes_at(grow + 1), below + cap * sizeof(u64));
    bo.threads = 8;
    for (const u64 limit : {grow - 1, grow, grow + 1, grow + 3})
        (void)bytes_at(limit);
}
