#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/slot_map.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>
#include <sstream>
#include <thread>

using namespace mymyr;
namespace
{
TaskPtr task(bool fixed, bool fact, TaskOptions::Atoms mode = TaskOptions::Atoms::Dense)
{
    std::istringstream text(std::string("O 2\nP 1\n") + (fixed ? "S" : "F") +
                            " 1 p\nSI " + (fixed && fact ? "1\n0 0\n" : "0\n") +
                            "FI " + (!fixed && fact ? "1\n0 0\n" : "0\n") + "G 0\n" + (fixed ? "A 0\n" : "A 1\nset 1\nL 0\nE 1\n0\nL 0\nF 1\n0 1 0\n"));
    TaskOptions options;
    options.atoms = mode;
    return Task::create(formalism::read_task_text(text), options);
}

TEST(SlotMap, StaticFactsBecomeConstantFluentBits)
{
    const auto source = task(true, true), target = task(false, false);
    const auto map = SlotMap::between(*source, *target);
    const State result = map->translate(source->initial_state());
    EXPECT_EQ(target->format_atoms(result), (std::vector<std::string>{"(p o0)"}));
    EXPECT_EQ(map->translate(State{}), result);
}

TEST(SlotMap, FluentFactsCheckTargetStaticsPerRow)
{
    const auto source = task(false, true), yes = task(true, true), no = task(true, false);
    EXPECT_EQ(SlotMap::between(*source, *yes)->translate(source->initial_state()).count(), 0);
    const auto map = SlotMap::between(*source, *no);
    EXPECT_THROW((void)map->translate(source->initial_state()), std::invalid_argument);
    bool lossless = true;
    EXPECT_EQ(map->translate(source->initial_state(), true, &lossless).count(), 0);
    EXPECT_FALSE(lossless);
    EXPECT_NO_THROW((void)map->translate(State{}));
}

TEST(SlotMap, LazyDestinationInternsConstantFacts)
{
    const auto source = task(true, true), target = task(false, false, TaskOptions::Atoms::Lazy);
    const auto map = SlotMap::between(*source, *target);
    EXPECT_EQ(target->format_atoms(map->translate(source->initial_state())), (std::vector<std::string>{"(p o0)"}));
}

TEST(SlotMap, SnapshotsRemainValidWhileLazySlotsGrow)
{
    const auto source = task(false, true, TaskOptions::Atoms::Lazy), target = task(false, false, TaskOptions::Atoms::Lazy);
    const auto map = SlotMap::between(*source, *target);
    const auto before = map->snapshot();
    const ObjectId object{1};
    const AtomArgs atom{PredicateId{0}, {&object, 1}};
    const State state = source->make_state({&atom, 1});
    std::vector<std::thread> workers;
    for (u32 i = 0; i < 4; ++i)
        workers.emplace_back([&] { EXPECT_EQ(target->format_atoms(map->translate(state)), (std::vector<std::string>{"(p o1)"})); });
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(before->size(), 1);
    EXPECT_EQ(map->snapshot()->size(), 2);
}
}
