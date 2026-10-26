#include "mymyr/formalism/text_format.hpp"
#include "mymyr/task/slot_map.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>

using namespace mymyr;
namespace
{
TaskPtr task(bool fixed, bool fact, TaskOptions::Atoms mode = TaskOptions::Atoms::Dense)
{
    std::istringstream text(std::string("O 2\nP 1\n") + (fixed ? "S" : "F") +
                            " 1 p\nSI " + (fixed && fact ? "1\n0 0\n" : "0\n") +
                            "FI " + (!fixed && fact ? "1\n0 0\n" : "0\n") + "G 0\n" +
                            (fixed ? "A 0\n" : "A 1\nset 1\nL 0\nE 1\n0\nL 0\nF 1\n0 1 0\n"));
    TaskOptions options;
    options.atoms = mode;
    return Task::create(formalism::read_task_text(text), options);
}

// n objects; `link` holds 100 facts per object (static or fluent), `at` is fluent with one initial fact.
TaskPtr linked(u32 n, bool fluent_links, TaskOptions::Atoms mode)
{
    std::string facts;
    for (u32 a = 0; a < n; ++a)
        for (u32 d = 1; d <= 100; ++d)
            facts += "0 " + std::to_string(a) + " " + std::to_string((a + d) % n) + "\n";
    const std::string count = std::to_string(n * 100);
    const std::string statics = fluent_links ? "0\n" : count + "\n" + facts;
    const std::string fluents = fluent_links ? std::to_string(n * 100 + 1) + "\n" + facts : std::string("1\n");
    std::istringstream text("O " + std::to_string(n) + "\nP 2\n" + (fluent_links ? "F" : "S") + " 2 link\nF 1 at\nSI " +
                            statics + "FI " + fluents + "1 0\nG 0\nA 0\n");
    TaskOptions options;
    options.atoms = mode;
    return Task::create(formalism::read_task_text(text), options);
}

f64 seconds_since(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<f64>(std::chrono::steady_clock::now() - start).count();
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

TEST(SlotMap, StaticFactsWithoutAFluentDestinationAreLossy)
{
    const auto source = task(true, true);
    std::istringstream text("O 2\nP 1\nF 1 p\nSI 0\nFI 0\nG 0\nA 0\n");
    const auto target = Task::create(formalism::read_task_text(text));
    const auto map = SlotMap::between(*source, *target);
    ASSERT_EQ(map->missing_constants(), (std::vector<std::string>{"(p o0)"}));
    EXPECT_THROW((void)map->translate(State{}), std::invalid_argument);
    bool lossless = true;
    (void)map->translate(State{}, true, &lossless);
    EXPECT_FALSE(lossless);
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

TEST(SlotMap, EncodingMapsCacheIndependentTasksWithTheSameFingerprint)
{
    const auto source = task(false, true, TaskOptions::Atoms::Lazy), target = task(false, true);
    const auto unrelated = task(false, false), lazy_target = task(false, true, TaskOptions::Atoms::Lazy);
    const auto map = source->slot_map_to(*target);
    ASSERT_TRUE(map);
    EXPECT_EQ(map, source->slot_map_to(*target));
    EXPECT_EQ(map, source->translation_map_to(*target));
    EXPECT_EQ(map->translate(source->initial_state()), target->initial_state());
    (void)source->translation_map_to(*unrelated);
    EXPECT_FALSE(source->slot_map_to(*unrelated));
    EXPECT_FALSE(source->slot_map_to(*lazy_target));
}

TEST(SlotMap, ConcurrentFrozenModesPublishMatchingTasksAndMaps)
{
    const auto source = task(false, true, TaskOptions::Atoms::Lazy);
    std::array<std::array<TaskPtr, 3>, 8> twins;
    std::vector<std::thread> workers;
    for (u32 worker = 0; worker < twins.size(); ++worker)
        workers.emplace_back([&, worker] {
            for (u32 step = 0; step < 3; ++step)
            {
                const u32 mode = (worker + step) % 3;
                const auto twin = source->frozen(static_cast<TaskOptions::Atoms>(mode));
                twins[worker][mode] = twin;
                const auto map = source->slot_map_to(*twin);
                ASSERT_TRUE(map);
                EXPECT_EQ(&map->target(), twin.get());
                EXPECT_EQ(map->translate(source->initial_state()), twin->initial_state());
            }
        });
    for (auto& worker : workers)
        worker.join();
    for (const auto& row : twins)
        EXPECT_EQ(row, twins.front());
}

TEST(SlotMap, SnapshotsAndFingerprintsWaitForConcurrentPublication)
{
    const auto make = [](TaskOptions::Atoms mode) {
        std::istringstream text("O 4096\nP 1\nF 1 p\nSI 0\nFI 0\nG 0\nA 1\nset 1\nL 0\nE 1\n0\nL 0\nF 1\n0 1 0\n");
        TaskOptions options;
        options.atoms = mode;
        return Task::create(formalism::read_task_text(text), options);
    };
    const auto source = make(TaskOptions::Atoms::Lazy), target = make(TaskOptions::Atoms::Dense);
    const auto map = SlotMap::between(*source, *target);
    const auto before = map->snapshot();
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (u32 object = 0; object < 4096; ++object)
            (void)source->atoms().intern(source->atoms().layout().encode(0, &object));
        done.store(true, std::memory_order_release);
    });
    u32 checked = 0;
    do
    {
        const auto slots = map->snapshot();
        for (; checked < slots->size(); ++checked)
        {
            EXPECT_LT((*slots)[checked], target->atoms().fluent_slots());
            if ((*slots)[checked] < target->atoms().fluent_slots())
            {
                EXPECT_EQ(target->format(SlotId{(*slots)[checked]}), source->format(SlotId{checked}));
            }
        }
        (void)source->encoding_fingerprint();
    } while (!done.load(std::memory_order_acquire));
    writer.join();
    EXPECT_TRUE(before->empty());
    EXPECT_EQ(map->snapshot()->size(), 4096);
    const auto serial = make(TaskOptions::Atoms::Lazy);
    for (u32 object = 0; object < 4096; ++object)
        (void)serial->atoms().intern(serial->atoms().layout().encode(0, &object));
    EXPECT_EQ(source->encoding_fingerprint(), serial->encoding_fingerprint());
}


TEST(SlotMap, ManyStaticFactsMapInLinearTime)
{
    // 150,000 static facts: comparing every source fact with every target fact takes several seconds.
    const auto source = linked(1500, false, TaskOptions::Atoms::Dense), target = linked(1500, false, TaskOptions::Atoms::Dense);
    auto start = std::chrono::steady_clock::now();
    const auto map = SlotMap::between(*source, *target);
    EXPECT_EQ(map->translate(source->initial_state()), target->initial_state());
    const auto lazy = linked(1500, false, TaskOptions::Atoms::Lazy);
    const auto twin = lazy->frozen(TaskOptions::Atoms::Dense);
    EXPECT_EQ(lazy->slot_map_to(*twin)->translate(lazy->initial_state()), twin->initial_state());
    EXPECT_LT(seconds_since(start), 3.0);

    // Every source fluent link is checked against the target's static links.
    const auto fluent = linked(400, true, TaskOptions::Atoms::Dense), statics = linked(400, false, TaskOptions::Atoms::Dense);
    start = std::chrono::steady_clock::now();
    const auto checked = SlotMap::between(*fluent, *statics);
    const State state = checked->translate(fluent->initial_state());
    EXPECT_LT(seconds_since(start), 3.0);
    EXPECT_EQ(state, statics->initial_state());
}

TEST(SlotMap, LazyTargetsInternOnlyTranslatedAtoms)
{
    const auto source = linked(400, true, TaskOptions::Atoms::Dense), target = linked(400, true, TaskOptions::Atoms::Lazy);
    const u32 before = target->atoms().fluent_slots();
    const auto map = SlotMap::between(*source, *target);
    (void)map->snapshot();
    EXPECT_EQ(target->atoms().fluent_slots(), before);
    const ObjectId objects[2] = {ObjectId{3}, ObjectId{4}};
    const AtomArgs atom{PredicateId{0}, {objects, 2}};
    const State state = source->make_state({&atom, 1});
    const State translated = map->translate(state);
    EXPECT_LE(target->atoms().fluent_slots(), before + 1);
    EXPECT_EQ(target->format_atoms(translated), source->format_atoms(state));
}

TEST(SlotMap, InvalidSlotsAreReportedBeforeMissingAtoms)
{
    const auto source = task(false, true), target = task(true, false);
    const auto map = SlotMap::between(*source, *target);
    const auto slots = map->snapshot();
    ASSERT_EQ(slots->size(), 2);
    ASSERT_EQ((*slots)[0], SlotMap::missing);
    const u64 word = 0b101;  // slot 0 is missing in the target, slot 2 is outside the source
    std::vector<u64> atoms(target->words() + 1), numeric;
    for (const bool drop : {false, true})
    {
        try
        {
            (void)map->translate_row(*slots, StateView{&word, 1, nullptr, 0}, atoms, numeric, drop, 7);
            ADD_FAILURE() << "an invalid slot must raise";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_STREQ(e.what(), "mymyr: translate: row 7 sets an unknown source slot 2");
        }
    }
}

TEST(SlotMap, CachedMapsLeaveWithTheirTargets)
{
    const auto source = task(false, true);
    std::weak_ptr<SlotMap> first;
    {
        const auto target = task(false, false);
        first = source->translation_map_to(*target);
        EXPECT_EQ(first.lock(), source->translation_map_to(*target));
    }
    for (u32 i = 0; i < 4; ++i)
    {
        const auto target = task(false, false);
        (void)source->translation_map_to(*target)->translate(source->initial_state());
    }
    EXPECT_TRUE(first.expired());
}

}
