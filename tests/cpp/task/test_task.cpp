// Task construction: atom modes, the canonical layout, initial states, goal tests and rejected input.

#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <thread>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
TEST(Task, InitialStateHoldsExactlyTheInitialAtoms)
{
    for (const auto& t : suite())
        for (auto mode : {TaskOptions::Atoms::Lazy, TaskOptions::Atoms::Dense, TaskOptions::Atoms::Reachable,
                          TaskOptions::Atoms::Auto})
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
            if (mode == TaskOptions::Atoms::Dense)
            {
                EXPECT_EQ(task->atoms().mode(), AtomMode::Dense);
                EXPECT_EQ(task->atoms().fluent_slots(), task->info().dense_fluent);
            }
            if (mode == TaskOptions::Atoms::Reachable)
            {
                EXPECT_EQ(task->atoms().mode(), AtomMode::Reachable);
                EXPECT_EQ(task->atoms().fluent_slots(), task->info().reachable_fluent);
                EXPECT_EQ(task->atoms().derived_slots(), task->info().reachable_derived);
            }
        }
}

TEST(Task, CanonicalLayoutRoundTrips)
{
    for (const auto& t : suite())
    {
        TaskOptions opt;
        opt.atoms = TaskOptions::Atoms::Dense;
        const auto task = Task::from_text_file(task_path(t.name), opt);
        const CanonicalLayout& L = task->compiled().layout;
        std::vector<u32> args(std::max<u32>(1, L.max_arity));
        const u64 step = std::max<u64>(1, L.total / 5000);  // sample large spaces
        for (u64 c = 0; c < L.total; c += step)
        {
            const u32 p = L.decode(c, args.data());
            ASSERT_EQ(L.encode(p, args.data()), c) << t.name;
            ASSERT_EQ(L.kind_of(c) == AtomKind::Fluent, task->compiled().kinds[p] == formalism::PredKind::Fluent) << t.name;
            // dense: slot = cid (minus F_fluent for derived atoms), and the record decodes to the same atom
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
        for (auto mode : {TaskOptions::Atoms::Lazy, TaskOptions::Atoms::Dense})
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

TEST(Task, AutoModeIsDenseExactlyWhereItIsNoWider)
{
    for (const auto& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name));
        const TaskInfo& i = task->info();
        ASSERT_TRUE(i.reachability_complete) << t.name;
        const bool no_wider = bits::words_for(i.dense_fluent) <= bits::words_for(i.reachable_fluent) &&
                              bits::words_for(i.dense_derived) <= bits::words_for(i.reachable_derived);
        EXPECT_EQ(task->atoms().mode(), no_wider ? AtomMode::Dense : AtomMode::Reachable) << t.name;
        EXPECT_LE(i.reachable_fluent, i.dense_fluent) << t.name;
        EXPECT_EQ(task->words(), bits::words_for(no_wider ? i.dense_fluent : i.reachable_fluent)) << t.name;
        EXPECT_EQ(task->words(), task->max_words()) << t.name;
    }
}

TaskOptions with_atoms(TaskOptions::Atoms atoms)
{
    TaskOptions o;
    o.atoms = atoms;
    return o;
}

TEST(Task, ReachableSlotsAreTheReachableAtomsInCanonicalOrder)
{
    for (const auto& t : suite())
    {
        SCOPED_TRACE(t.name);
        const auto reach = Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Reachable));
        const auto dense = Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Dense));
        ASSERT_EQ(reach->atoms().mode(), AtomMode::Reachable);
        const AtomIndex& A = reach->atoms();
        for (const AtomKind kind : {AtomKind::Fluent, AtomKind::Derived})
        {
            const u32 n = kind == AtomKind::Fluent ? A.fluent_slots() : A.derived_slots();
            for (u32 s = 0; s < n; ++s)
            {
                EXPECT_EQ(A.find(A.canonical(kind, s)), s);
                if (s > 0)
                {
                    EXPECT_LT(A.canonical(kind, s - 1), A.canonical(kind, s));
                }
            }
        }
        // closed under successors: the search meets no atom without a slot and stores what the dense task stores
        BrfsOptions o;
        o.max_states = 20000;
        o.stop_at_goal = false;
        o.fingerprint = true;
        const BrfsResult a = brfs(*reach, o), b = brfs(*dense, o);
        EXPECT_EQ(a.states, b.states);
        EXPECT_EQ(a.generated, b.generated);
        EXPECT_EQ(a.fingerprint, b.fingerprint);
    }
}

TEST(Task, ReachableSlotsRefuseAtomsNoStateHolds)
{
    u32 tasks = 0;
    for (const auto& t : suite())
    {
        const auto task = Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Reachable));
        const CanonicalLayout& L = task->compiled().layout;
        CanonicalAtom c = 0;
        while (c < L.fluent_count && task->atoms().find(c) != AtomIndex::k_empty)
            ++c;
        if (c == L.fluent_count)
            continue;
        SCOPED_TRACE(t.name);
        ++tasks;
        std::vector<u32> args(std::max<u32>(1, L.max_arity));
        const u32 p = L.decode(c, args.data());
        std::vector<ObjectId> objects;
        for (u32 i = 0; i < L.arity[p]; ++i)
            objects.push_back(ObjectId{args[i]});
        const std::vector<AtomArgs> atoms{{PredicateId{p}, objects}};
        EXPECT_THROW((void)task->make_state(atoms), std::invalid_argument);
        EXPECT_THROW((void)task->atoms().intern(c), std::logic_error);
    }
    EXPECT_GT(tasks, 0u);
}

TEST(Task, ReachabilityBudgetFallsBackToDenseSlots)
{
    const std::string path = task_path("gripper__prob05");
    TaskOptions o;
    o.reachability_budget = 1;
    const auto t = Task::from_text_file(path, o);
    EXPECT_FALSE(t->info().reachability_complete);
    EXPECT_EQ(t->atoms().mode(), AtomMode::Dense);
    o.atoms = TaskOptions::Atoms::Reachable;
    EXPECT_THROW((void)Task::from_text_file(path, o), std::length_error);
    o.reachability_budget = ~u64{0};  // unbounded
    EXPECT_TRUE(Task::from_text_file(path, o)->info().reachability_complete);
}

TEST(Task, EncodingIsAFunctionOfTheTask)
{
    for (const auto& t : suite())
    {
        SCOPED_TRACE(t.name);
        const auto a = Task::from_text_file(task_path(t.name));
        std::vector<std::shared_ptr<const Task>> built(8);
        std::vector<std::thread> threads;
        for (u32 i = 0; i < built.size(); ++i)
            threads.emplace_back([&, i] { built[i] = Task::from_text_file(task_path(t.name)); });
        for (std::thread& th : threads)
            th.join();
        for (const auto& b : built)
        {
            EXPECT_EQ(b->atoms().mode(), a->atoms().mode());
            EXPECT_EQ(b->words(), a->words());
            EXPECT_EQ(b->encoding_fingerprint(), a->encoding_fingerprint());
            ASSERT_EQ(b->atoms().fluent_slots(), a->atoms().fluent_slots());
            ASSERT_EQ(b->atoms().derived_slots(), a->atoms().derived_slots());
            for (u32 s = 0; s < a->atoms().fluent_slots(); ++s)
                ASSERT_EQ(b->atoms().canonical(AtomKind::Fluent, s), a->atoms().canonical(AtomKind::Fluent, s));
            const State sa = a->initial_state(), sb = b->initial_state();
            EXPECT_TRUE(std::equal(sa.data(), sa.data() + sa.size_words(), sb.data(), sb.data() + sb.size_words()));
        }
    }
}

TEST(Task, EncodingFingerprintFollowsTheSlotTable)
{
    u32 differ = 0;
    for (const auto& t : suite())
    {
        SCOPED_TRACE(t.name);
        const auto dense = Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Dense));
        const auto reach = Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Reachable));
        EXPECT_EQ(dense->encoding_fingerprint(),
                  Task::from_text_file(task_path(t.name), with_atoms(TaskOptions::Atoms::Dense))->encoding_fingerprint());
        if (dense->atoms().fluent_slots() == reach->atoms().fluent_slots() &&
            dense->atoms().derived_slots() == reach->atoms().derived_slots())
        {
            // every atom reachable: the same slots, the same encoding
            EXPECT_EQ(dense->encoding_fingerprint(), reach->encoding_fingerprint());
            continue;
        }
        ++differ;
        EXPECT_NE(dense->encoding_fingerprint(), reach->encoding_fingerprint());
    }
    EXPECT_GT(differ, 0u);
    // lazy slots: the fingerprint follows the slots assigned so far
    const std::string path = task_path("blocks__probBLOCKS-8-0");
    const auto a = Task::from_text_file(path, with_atoms(TaskOptions::Atoms::Lazy));
    const auto b = Task::from_text_file(path, with_atoms(TaskOptions::Atoms::Lazy));
    EXPECT_EQ(a->encoding_fingerprint(), b->encoding_fingerprint());
    BrfsOptions o;
    o.max_states = 200;
    (void)brfs(*a, o);
    EXPECT_NE(a->encoding_fingerprint(), b->encoding_fingerprint());
    (void)brfs(*b, o);
    EXPECT_EQ(a->encoding_fingerprint(), b->encoding_fingerprint());
}

TEST(Task, LazySlotsGrowRaceFreeUnderConcurrentSearches)
{
    const std::string path = task_path("blocks__probBLOCKS-8-0");
    const auto shared = Task::from_text_file(path, with_atoms(TaskOptions::Atoms::Lazy));
    BrfsOptions o;
    o.max_states = 3000;
    o.stop_at_goal = false;
    o.fingerprint = true;
    const BrfsResult ref = brfs(*Task::from_text_file(path, with_atoms(TaskOptions::Atoms::Lazy)), o);
    std::vector<BrfsResult> got(8);
    std::vector<std::thread> threads;
    for (u32 i = 0; i < got.size(); ++i)
        threads.emplace_back([&, i] { got[i] = brfs(*shared, o); });
    for (std::thread& th : threads)
        th.join();
    for (const BrfsResult& r : got)
    {
        EXPECT_EQ(r.states, ref.states);
        EXPECT_EQ(r.fingerprint, ref.fingerprint);  // canonical hashes: independent of the slot numbering
    }
    // one slot per atom, and every slot's atom finds it
    const AtomIndex& A = shared->atoms();
    std::set<CanonicalAtom> seen;
    for (u32 s = 0; s < A.fluent_slots(); ++s)
    {
        EXPECT_TRUE(seen.insert(A.canonical(AtomKind::Fluent, s)).second);
        EXPECT_EQ(A.find(A.canonical(AtomKind::Fluent, s)), s);
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
