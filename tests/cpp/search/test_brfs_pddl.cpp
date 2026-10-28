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
