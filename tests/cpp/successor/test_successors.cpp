// Differential test of the lifted successor generator against an exhaustive oracle.
// On seeded random walks over every suite task, for fixed-order and forward-checking matchers and lazy and frozen
// slots, with witness pruning off:
//   - the applicable action labels equal the oracle's, in canonical order;
//   - every successor equals the oracle's successor (conditional effects and axioms included);
//   - goal tests, is_applicable, apply and any_applicable agree.
// With witness pruning on, the emitted actions are applicable and cover every oracle action's effect-relevant
// parameters exactly once.

#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <map>
#include <random>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct Mode
{
    TaskOptions::Matching matching;
    TaskOptions::Atoms atoms;
    const char* name;
};
const Mode k_modes[] = {
    {TaskOptions::Matching::FixedOrder, TaskOptions::Atoms::Lazy, "fixed_lazy"},
    {TaskOptions::Matching::FixedOrder, TaskOptions::Atoms::Frozen, "fixed_frozen"},
    {TaskOptions::Matching::ForwardChecking, TaskOptions::Atoms::Lazy, "fc_lazy"},
    {TaskOptions::Matching::ForwardChecking, TaskOptions::Atoms::Frozen, "fc_frozen"},
};

struct Param
{
    std::string task;
    Mode mode;
};
void PrintTo(const Param& p, std::ostream* os) { *os << p.task << "/" << p.mode.name; }

class OracleWalk : public ::testing::TestWithParam<Param>
{
};

TEST_P(OracleWalk, ApplicableSetsAndSuccessorsMatch)
{
    const Param& p = GetParam();
    TaskOptions opt;
    opt.matching = p.mode.matching;
    opt.atoms = p.mode.atoms;
    const auto task = Task::from_text_file(task_path(p.task), opt);
    Oracle oracle(task->data());
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    succ.set_witness_pruning(false);
    succ.set_canonical_order(true);

    const u32 walks = 2, steps = 20;
    u64 checked_actions = 0;
    for (u32 w = 0; w < walks; ++w)
    {
        std::mt19937_64 rng(1234 + 7919 * w);
        State s = task->initial_state();
        AtomSet os = oracle.initial();
        for (u32 step = 0; step < steps; ++step)
        {
            ASSERT_EQ(atoms_of(*task, s), os) << "walk " << w << " step " << step;
            const AtomSet der = oracle.derive(os);
            ASSERT_EQ(task->is_goal(s), oracle.is_goal(os, der)) << "walk " << w << " step " << step;

            std::vector<Action> expect;
            try
            {
                expect = oracle.applicable(os, der);
            }
            catch (const std::runtime_error&)
            {
                GTEST_SKIP() << "oracle enumeration budget exceeded";
            }
            std::vector<Action> got;
            std::vector<State> succs;
            succ.for_each_applicable(s,
                                     [&](const ActionLabel& a, const Delta& d)
                                     {
                                         got.emplace_back(a);
                                         StateBuilder b(s);
                                         b.apply(d.del, d.add);
                                         succs.push_back(b.build());
                                     });
            ASSERT_TRUE(std::is_sorted(got.begin(), got.end())) << "canonical order violated";
            ASSERT_EQ(got.size(), expect.size()) << "walk " << w << " step " << step;
            ASSERT_EQ(got, expect) << "walk " << w << " step " << step;
            ASSERT_EQ(succ.any_applicable(s), !got.empty());
            for (usize i = 0; i < got.size(); ++i)
            {
                ASSERT_EQ(atoms_of(*task, succs[i]), oracle.successor(os, der, got[i])) << task->format(got[i].label());
                ASSERT_TRUE(succ.is_applicable(s, got[i].label()));
                ASSERT_EQ(succ.apply(s, got[i].label()), succs[i]);
                ++checked_actions;
            }
            if (got.empty())
                break;
            const usize pick = std::uniform_int_distribution<usize>(0, got.size() - 1)(rng);
            os = oracle.successor(os, der, got[pick]);
            s = succs[pick];
        }
    }
    RecordProperty("checked_actions", std::to_string(checked_actions));
}

TEST_P(OracleWalk, WitnessPruningCoversRelevantProjections)
{
    const Param& p = GetParam();
    TaskOptions opt;
    opt.matching = p.mode.matching;
    opt.atoms = p.mode.atoms;
    const auto task = Task::from_text_file(task_path(p.task), opt);
    Oracle oracle(task->data());
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    succ.set_witness_pruning(true);
    succ.set_canonical_order(true);
    std::mt19937_64 rng(99);
    State s = task->initial_state();
    AtomSet os = oracle.initial();
    for (u32 step = 0; step < 15; ++step)
    {
        const AtomSet der = oracle.derive(os);
        std::vector<Action> expect;
        try
        {
            expect = oracle.applicable(os, der);
        }
        catch (const std::runtime_error&)
        {
            GTEST_SKIP() << "oracle enumeration budget exceeded";
        }
        auto project = [&](const Action& a)
        {
            const std::vector<bool> rel = oracle.relevant(a.schema.v);
            std::vector<u32> key{a.schema.v};
            for (usize i = 0; i < a.binding.size(); ++i)
                key.push_back(rel[i] ? a.binding[i].v : ~u32{0});
            return key;
        };
        std::set<std::vector<u32>> want;
        for (const auto& a : expect)
            want.insert(project(a));
        std::multiset<std::vector<u32>> have;
        std::vector<Action> got;
        succ.for_each_applicable(s,
                                 [&](const ActionLabel& a, const Delta&)
                                 {
                                     got.emplace_back(a);
                                     have.insert(project(got.back()));
                                 });
        ASSERT_TRUE(std::is_sorted(got.begin(), got.end()));
        ASSERT_EQ(std::set<std::vector<u32>>(have.begin(), have.end()), want);
        ASSERT_EQ(have.size(), want.size()) << "a relevant projection was emitted twice";
        for (const auto& a : got)
            ASSERT_TRUE(std::binary_search(expect.begin(), expect.end(), a)) << task->format(a.label());
        if (got.empty())
            break;
        const Action& a = got[std::uniform_int_distribution<usize>(0, got.size() - 1)(rng)];
        os = oracle.successor(os, der, a);
        s = succ.apply(s, a.label());
    }
}

std::vector<Param> params()
{
    std::vector<Param> out;
    for (const auto& t : suite())
        for (const auto& m : k_modes)
            out.push_back({t.name, m});
    return out;
}

INSTANTIATE_TEST_SUITE_P(Suite, OracleWalk, ::testing::ValuesIn(params()),
                         [](const ::testing::TestParamInfo<Param>& i)
                         {
                             std::string n = i.param.task + "_" + i.param.mode.name;
                             for (char& c : n)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return n;
                         });

TEST(Successors, CanonicalOrderOffEmitsTheSameMultiset)
{
    for (const auto& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        const WorkspaceLease lease = task->workspace();
        Successors& succ = lease->successors();
        succ.set_witness_pruning(false);
        std::mt19937_64 rng(5);
        State s = task->initial_state();
        for (u32 step = 0; step < 10; ++step)
        {
            succ.set_canonical_order(true);
            const auto a = succ.applicable_actions(s);
            succ.set_canonical_order(false);
            auto b = succ.applicable_actions(s);
            std::sort(b.begin(), b.end());
            ASSERT_EQ(a, b) << t.name;
            if (a.empty())
                break;
            s = succ.apply(s, a[std::uniform_int_distribution<usize>(0, a.size() - 1)(rng)].label());
        }
        succ.set_canonical_order(true);
        succ.set_witness_pruning(true);
    }
}

TEST(Successors, RejectsInapplicableAndMalformedLabels)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s = task->initial_state();
    const auto acts = succ.applicable_actions(s);
    ASSERT_FALSE(acts.empty());
    Action bad = acts.front();
    bad.binding.push_back(ObjectId{0});
    EXPECT_FALSE(succ.is_applicable(s, bad.label()));
    EXPECT_THROW((void) succ.apply(s, bad.label()), std::invalid_argument);
    // some binding of the first schema is not applicable
    const u32 arity = succ.arity(0);
    std::vector<ObjectId> b(arity, ObjectId{0});
    bool found_inapplicable = false;
    for (u32 o = 0; o < task->num_objects() && !found_inapplicable; ++o)
    {
        std::fill(b.begin(), b.end(), ObjectId{o});
        if (!succ.is_applicable(s, {SchemaId{0}, b}))
        {
            found_inapplicable = true;
            EXPECT_THROW((void) succ.apply(s, {SchemaId{0}, b}), std::invalid_argument);
        }
    }
    EXPECT_TRUE(found_inapplicable);
}
}  // namespace
