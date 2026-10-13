// State spaces against the fork's StateSpace on the fork's own test instances: counts, V*, and the content, transition
// and V* fingerprints of the fork's state-space tool.

#include "../../cpp/datasets/support.hpp"
#include "../../cpp/support/suite.hpp"
#include "golden.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;
using namespace mymyr::test::ss;

namespace
{
// ----------------------------------------------------------------------------------------------- fingerprints
// The fork_datasets definitions (FNV-1a 64 over bytes; u64 as 8 little-endian bytes).
u64 fnv_str(const std::string& s)
{
    Fnv f;
    f.str(s);
    return f.h;
}
u64 fnv_list(std::vector<u64> v)
{
    std::sort(v.begin(), v.end());
    Fnv f;
    for (u64 x : v)
        f.u64le(x);
    return f.h;
}

u64 state_hash(const Task& task, StateView s)
{
    std::vector<std::string> atoms = task.format_atoms(s);
    std::sort(atoms.begin(), atoms.end());
    std::string key;
    for (usize i = 0; i < atoms.size(); ++i)
        key += (i ? "\n" : "") + atoms[i];
    if (task.numeric_slots())
        for (f64 v : task.numeric_values(s))
        {
            char b[64];
            std::snprintf(b, sizeof b, "\n=%.17g", v);
            key += b;
        }
    return fnv_str(key);
}

struct Fingerprints
{
    u64 content, transitions, vstar;
};
Fingerprints fingerprints(const StateSpace& S)
{
    const Task& task = *S.task();
    std::vector<u64> hs(S.num_states());
    for (u32 v = 0; v < S.num_states(); ++v)
        hs[v] = state_hash(task, S.state(v));
    std::vector<u64> trans, vst;
    const auto off = S.forward_offsets();
    const auto tgt = S.forward_targets();
    for (u32 v = 0; v < S.num_states(); ++v)
    {
        for (u64 e = off[v]; e < off[v + 1]; ++e)
        {
            Fnv f;
            f.u64le(hs[v]);
            f.u64le(fnv_str(task.format(S.label(e).label())));
            f.u64le(hs[tgt[e]]);
            trans.push_back(f.h);
        }
        Fnv f;
        f.u64le(hs[v]);
        const i32 d = S.unit_goal_distances()[v];
        f.u64le(static_cast<u64>(static_cast<i64>(d == k_unsolvable_distance ? std::numeric_limits<i32>::max() : d)));
        vst.push_back(f.h);
    }
    return {fnv_list(hs), fnv_list(trans), fnv_list(vst)};
}

TaskPtr fork_task(const std::string& dir, const std::string& problem)
{
    return dense_task(test::fork_data_dir(), dir, problem);
}

// ----------------------------------------------------------------------------------------------- fork parity
struct ForkCase
{
    const char* dir;
    const char* problem;
    u32 states;
    u64 transitions;
    u32 goal, unsolvable;
    i32 max_goal_dist;
    u64 fp_content, fp_transitions, fp_vstar;
    f64 cost_sum;  // sum of the finite cost goal distances (the fork's action goal distances)
};

// fork_datasets ss lists/fork_data.txt --max-states 200000 (build/parity/fork_data.jsonl; the fork 0.16.3 configured
// as fork_ss: remove_if_unsolvable = false)
const ForkCase kForkCases[] = {
#include "fork_cases.inc"
};

class ForkParity : public ::testing::TestWithParam<ForkCase>
{
};

TEST_P(ForkParity, MatchesTheFork)
{
    const ForkCase& c = GetParam();
    const TaskPtr task = fork_task(c.dir, c.problem);
    if (!task)
        GTEST_SKIP() << "reference data missing: " << test::fork_data_dir() / c.dir << " (set MYMYR_PARITY_DATA)";
    const StateSpacePtr S = space_of(task, ss_options());
    ASSERT_TRUE(S);
    EXPECT_EQ(S->num_states(), c.states);
    EXPECT_EQ(S->num_transitions(), c.transitions);
    EXPECT_EQ(S->num_goal_states(), c.goal);
    EXPECT_EQ(S->num_unsolvable_states(), c.unsolvable);
    EXPECT_EQ(S->max_goal_distance(), c.max_goal_dist);
    const Fingerprints fp = fingerprints(*S);
    EXPECT_EQ(fp.content, c.fp_content);
    EXPECT_EQ(fp.transitions, c.fp_transitions);
    EXPECT_EQ(fp.vstar, c.fp_vstar);
    f64 cost_sum = 0;
    for (f64 x : S->cost_goal_distances())
        if (std::isfinite(x))
            cost_sum += x;
    EXPECT_DOUBLE_EQ(cost_sum, c.cost_sum);
    check_invariants(*S);
    // the layered generator on several threads: the same arrays
    const StateSpacePtr P = space_of(task, ss_options(3));
    ASSERT_TRUE(P);
    EXPECT_EQ(arrays_hash(*P), arrays_hash(*S));
}

std::string case_name(const ::testing::TestParamInfo<ForkCase>& info)
{
    std::string n = std::string(info.param.dir) + "_" + info.param.problem;
    for (char& ch : n)
        if (!std::isalnum(static_cast<unsigned char>(ch)))
            ch = '_';
    return n;
}
INSTANTIATE_TEST_SUITE_P(ForkData, ForkParity, ::testing::ValuesIn(kForkCases), case_name);

// Numeric tasks (the fork's StateSpace generates them too): tests/data/pddl/counters, fork numbers from
// fork_datasets ss lists/numeric.txt. The state hashes cover the numeric values ("\n=%.17g" per variable).
TEST(StateSpace, NumericTasksMatchTheFork)
{
    struct Case
    {
        const char* problem;
        u32 states;
        u64 transitions;
        u32 goal, unsolvable;
        i32 max_goal_dist;
        u64 fp_content, fp_transitions, fp_vstar;
        f64 cost_sum;
    };
    const Case cases[] = {
        {"p01.pddl", 38, 108, 10, 0, 4, 0x2da00399d901c099ULL, 0xf345107ab284777cULL, 0xb86c738b18e1a06dULL, 59.0},
        {"p02.pddl", 129, 544, 0, 129, -1, 0x37338418a09264c5ULL, 0x0d275494cc07b57cULL, 0x3e585b03385b3749ULL, 0.0},
    };
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
    for (const Case& c : cases)
    {
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Dense;
        const TaskPtr task = Task::create(*frontend::load_task(dir + "domain.pddl", dir + c.problem), to);
        ASSERT_GT(task->numeric_slots(), 0u);
        for (u32 T : {1u, 3u})
        {
            const StateSpacePtr S = space_of(task, ss_options(T));
            ASSERT_TRUE(S);
            EXPECT_EQ(S->num_states(), c.states) << c.problem;
            EXPECT_EQ(S->num_transitions(), c.transitions);
            EXPECT_EQ(S->num_goal_states(), c.goal);
            EXPECT_EQ(S->num_unsolvable_states(), c.unsolvable);
            EXPECT_EQ(S->max_goal_distance(), c.max_goal_dist);
            const Fingerprints fp = fingerprints(*S);
            EXPECT_EQ(fp.content, c.fp_content) << c.problem;
            EXPECT_EQ(fp.transitions, c.fp_transitions);
            EXPECT_EQ(fp.vstar, c.fp_vstar);
            f64 cost_sum = 0;
            for (f64 x : S->cost_goal_distances())
                if (std::isfinite(x))
                    cost_sum += x;
            EXPECT_DOUBLE_EQ(cost_sum, c.cost_sum);
            check_invariants(*S);
        }
    }
}
}  // namespace
