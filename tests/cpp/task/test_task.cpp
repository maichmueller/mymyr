// Task construction: atom modes, the canonical layout, initial states, goal tests and rejected input.

#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <thread>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
TEST(Task, InitialStateHoldsExactlyTheInitialAtoms)
{
    for (const auto& t : suite())
        for (auto mode : {TaskOptions::Atoms::Lazy, TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Auto})
        {
            TaskOptions opt;
            opt.atoms = mode;
            const auto task = Task::from_text_file(task_path(t.name), opt);
            Oracle oracle(task->data());
            EXPECT_EQ(atoms_of(*task, task->initial_state()), oracle.initial()) << t.name;
            EXPECT_EQ(task->initial_state().count(), oracle.initial().size()) << t.name;
            if (mode == TaskOptions::Atoms::Lazy)
            {
                EXPECT_EQ(task->atoms().mode(), AtomMode::Lazy);
                EXPECT_EQ(task->atoms().fluent_slots(), oracle.initial().size()) << t.name;  // first touch only
            }
            if (mode == TaskOptions::Atoms::Frozen)
            {
                EXPECT_EQ(task->atoms().mode(), AtomMode::Frozen);
                EXPECT_EQ(task->atoms().fluent_slots(), task->info().dense_fluent);
            }
        }
}

TEST(Task, CanonicalLayoutRoundTrips)
{
    for (const auto& t : suite())
    {
        TaskOptions opt;
        opt.atoms = TaskOptions::Atoms::Frozen;
        const auto task = Task::from_text_file(task_path(t.name), opt);
        const CanonicalLayout& L = task->compiled().layout;
        std::vector<u32> args(std::max<u32>(1, L.max_arity));
        const u64 step = std::max<u64>(1, L.total / 5000);  // sample large spaces
        for (u64 c = 0; c < L.total; c += step)
        {
            const u32 p = L.decode(c, args.data());
            ASSERT_EQ(L.encode(p, args.data()), c) << t.name;
            ASSERT_EQ(L.kind_of(c) == AtomKind::Fluent, task->compiled().kinds[p] == formalism::PredKind::Fluent) << t.name;
            // frozen: slot = cid (minus F_fluent for derived atoms), and the record decodes to the same atom
            const AtomKind k = L.kind_of(c);
            const u32 slot = static_cast<u32>(k == AtomKind::Fluent ? c : c - L.fluent_count);
            ASSERT_EQ(task->atoms().find(c), slot);
            const u32* rec = task->atoms().record(k, slot);
            ASSERT_EQ(rec[0], p);
            for (u32 i = 0; i < L.arity[p]; ++i)
                ASSERT_EQ(rec[1 + i], args[i]);
            ASSERT_EQ(task->atoms().canonical(k, slot), c);
        }
    }
}

TEST(Task, FindAtomAndFormat)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    const State s0 = task->initial_state();
    for (SlotId s : s0.slots())
    {
        const u32* rec = task->atoms().record(AtomKind::Fluent, s.v);
        std::vector<ObjectId> args;
        for (u32 i = 0; i < task->compiled().arity[rec[0]]; ++i)
            args.push_back(ObjectId{rec[1 + i]});
        EXPECT_EQ(task->find_atom(PredicateId{rec[0]}, args), s);
    }
    EXPECT_EQ(task->format_atoms(s0).size(), s0.count());
    EXPECT_FALSE(task->find_atom(PredicateId{0}, {}).valid());  // a static predicate (type) has no slot
}

TEST(Task, MakeStateFromAtomsAndValues)
{
    for (const char* path : {"tasks/gripper__prob05.txt", "numeric_tasks/cs-counters.txt"})
        for (auto mode : {TaskOptions::Atoms::Lazy, TaskOptions::Atoms::Frozen})
        {
            TaskOptions opt;
            opt.atoms = mode;
            const auto task = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/" + path, opt);
            const State s0 = task->initial_state();
            std::vector<std::vector<ObjectId>> objects;
            std::vector<PredicateId> preds;
            for (SlotId s : s0.slots())
            {
                const u32* rec = task->atoms().record(AtomKind::Fluent, s.v);
                preds.push_back(PredicateId{rec[0]});
                objects.emplace_back();
                for (u32 i = 0; i < task->compiled().arity[rec[0]]; ++i)
                    objects.back().push_back(ObjectId{rec[1 + i]});
            }
            std::vector<AtomArgs> atoms;
            for (usize i = preds.size(); i-- > 0;)  // reversed: the order does not matter
                atoms.push_back({preds[i], objects[i]});
            const std::vector<f64> values = task->numeric_values(s0.view());
            EXPECT_EQ(task->make_state(atoms, values), s0) << path;
            if (!values.empty())
            {
                std::vector<f64> other = values;
                other[0] += 1;
                const State t = task->make_state(atoms, other);
                EXPECT_NE(t, s0);
                EXPECT_EQ(task->numeric_values(t.view()), other);
                EXPECT_THROW((void)task->make_state(atoms), std::invalid_argument);  // values missing
                other[0] = 0.5;
                EXPECT_THROW((void)task->make_state(atoms, other), std::overflow_error);  // I32 slots
            }
            const std::vector<AtomArgs> statics{{PredicateId{0}, {}}};  // a static predicate (type)
            EXPECT_THROW((void)task->make_state(statics, values), std::invalid_argument);
            if (atoms.empty())
                continue;
            std::vector<AtomArgs> short_atom{atoms.front()};
            if (!short_atom.front().objects.empty())
            {
                short_atom.front().objects = short_atom.front().objects.first(0);
                EXPECT_THROW((void)task->make_state(short_atom, values), std::invalid_argument);
            }
        }
}

TEST(Task, AutoModeFollowsTheDenseWidthRule)
{
    for (const auto& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        const u32 wd = bits::words_for(task->info().dense_fluent);
        if (wd <= 8)
        {
            EXPECT_EQ(task->atoms().mode(), AtomMode::Frozen) << t.name;
        }
        else if (task->info().pilot_words)
        {
            EXPECT_EQ(task->atoms().mode() == AtomMode::Frozen, wd <= 2 * task->info().pilot_words) << t.name;
        }
    }
}

TEST(Task, LiveLeasesHaveDistinctWorkspacesAndReleasedOnesAreReused)
{
    const auto task = Task::from_text_file(task_path("philosophers__p03-phil4"));
    WorkspaceLease a = task->workspace();
    WorkspaceLease b = task->workspace();
    ASSERT_TRUE(a && b);
    EXPECT_NE(&*a, &*b);
    Workspace* const pb = &*b;
    b.release();
    EXPECT_FALSE(b);
    const WorkspaceLease c = task->workspace();
    EXPECT_EQ(&*c, pb);  // this thread's released workspace
    Workspace* other = nullptr;
    std::thread([&] {
        const WorkspaceLease d = task->workspace();
        other = &*d;
    }).join();
    EXPECT_NE(other, &*a);
    EXPECT_NE(other, &*c);
    // leases taken on short-lived threads and released elsewhere are reused, not made anew
    const usize made = task->workspaces_made();
    for (u32 i = 0; i < 50; ++i)
    {
        WorkspaceLease e;
        std::thread([&] { e = task->workspace(); }).join();
        e.release();  // on this thread
    }
    EXPECT_LE(task->workspaces_made(), made + 1);
    // a moved lease keeps its workspace, the source is empty
    WorkspaceLease moved = std::move(a);
    EXPECT_FALSE(a);
    EXPECT_TRUE(moved);
}

TEST(Task, NestedEnumerationsAndGoalTestsInsideAnEnumerationDoNotDisturbIt)
{
    const auto task = Task::from_text_file(task_path("philosophers__p03-phil4"));  // derived goal
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const State s0 = task->initial_state();
    const auto plain = succ.applicable_actions(s0);
    ASSERT_FALSE(plain.empty());
    std::vector<Action> nested;
    succ.for_each_applicable(s0,
                             [&](const ActionLabel& a, const Delta& d)
                             {
                                 StateBuilder b(s0);
                                 b.apply(d.del, d.add);
                                 const State child = b.build();
                                 (void) task->is_goal(child.view());
                                 // a full enumeration of the child's actions in a workspace of its own
                                 const WorkspaceLease inner = task->workspace();
                                 EXPECT_NE(&*inner, &*lease);
                                 for (const Action& x : inner->successors().applicable_actions(child.view()))
                                     (void) inner->successors().apply(child.view(), x.label());
                                 nested.emplace_back(a);
                             });
    EXPECT_EQ(plain, nested);
}
}  // namespace
