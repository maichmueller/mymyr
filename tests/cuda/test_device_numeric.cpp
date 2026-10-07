#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/brfs.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/env.hpp"
#include "mymyr/cuda/rollouts.hpp"
#include "mymyr/cuda/state_space.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "env_harness.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/numeric.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace mymyr;
namespace
{
std::vector<std::string> numeric_tasks()
{
    std::vector<std::string> names;
    for (const auto& p : std::filesystem::directory_iterator(std::string(MYMYR_SOURCE_DIR) + "/tests/data/numeric_tasks"))
        if (p.path().extension() == ".txt") names.push_back(p.path().stem().string());
    std::sort(names.begin(), names.end());
    return names;
}

cuda::ContextPtr context()
{
    cuda::ContextOptions options;
    options.max_bytes = u64{4} << 30;
    return cuda::DeviceContext::create(0, options);
}

TaskPtr task_of(const std::string& name, const TaskOptions& options = {})
{
    return Task::from_text_file(std::string(MYMYR_SOURCE_DIR) + "/tests/data/numeric_tasks/" + name + ".txt", options);
}

std::vector<State> walks(const Task& task, u32 n = 8)
{
    auto& successors = task.workspace().successors();
    std::vector<State> states{task.initial_state()};
    for (u32 i = 1; i < n; ++i)
    {
        const auto actions = successors.applicable_actions(states.back().view());
        states.push_back(actions.empty() ? task.initial_state() : successors.apply(states.back().view(), actions[(i * 7) % actions.size()].label()));
    }
    return states;
}

void compare_expand(const cuda::ContextPtr& ctx, TaskPtr task, const std::vector<State>& states, bool witness, u64 chunk = 0)
{
    const auto table = rl::TaskTable::single(task);
    const u32 W = std::max<u32>(1, task->words()), NN = task->numeric_words(), RW = W + NN;
    const u32 L = std::max<u32>(1, table->label_width());
    std::vector<u64> input(states.size() * RW, 0);
    for (u64 i = 0; i < states.size(); ++i)
    {
        std::copy(states[i].words().begin(), states[i].words().end(), input.begin() + static_cast<std::ptrdiff_t>(i * RW));
        std::copy(states[i].numeric().begin(), states[i].numeric().end(), input.begin() + static_cast<std::ptrdiff_t>(i * RW + W));
    }
    cuda::DeviceBuffer din(ctx, input.size() * 8);
    cuda::check(cudaMemcpy(din.data(), input.data(), input.size() * 8, cudaMemcpyHostToDevice), "numeric input");
    cuda::DeviceExpander expander(ctx, table);
    expander.set_chunk_rows(chunk);
    const rl::ExpandOptions options{true, witness, true};
    const u64 count = expander.count({static_cast<const u64*>(din.data()), states.size(), W, RW, NN}, nullptr, options);
    for (u64 capacity : {count, count ? count / 2 : 0})
    {
        const u64 cap = std::max<u64>(capacity, 1);
        std::vector<u64> expected(cap * RW), actual(cap * RW);
        std::vector<i32> parent(cap), schema(cap), binding(cap * L), offsets(states.size() + 1);
        std::vector<u8> goal(cap);
        rl::Expansion cpu;
        cpu.capacity = capacity; cpu.words = W; cpu.numeric_words = NN; cpu.label_width = L;
        cpu.succ = expected.data(); cpu.parent = parent.data(); cpu.schema = schema.data(); cpu.binding = binding.data();
        cpu.goal = goal.data(); cpu.offsets = offsets.data();
        rl::expand(*table, {input.data(), states.size(), W, RW, NN}, nullptr, cpu, options);
        ASSERT_EQ(count, cpu.total);
        cuda::DeviceBuffer dout(ctx, cap * RW * 8), dp(ctx, cap * 4), ds(ctx, cap * 4), db(ctx, cap * L * 4),
                           dg(ctx, cap), di(ctx, offsets.size() * 4);
        rl::Expansion gpu = cpu;
        gpu.succ = static_cast<u64*>(dout.data()); gpu.parent = static_cast<i32*>(dp.data()); gpu.schema = static_cast<i32*>(ds.data());
        gpu.binding = static_cast<i32*>(db.data()); gpu.goal = static_cast<u8*>(dg.data()); gpu.offsets = static_cast<i32*>(di.data());
        expander.write(gpu);
        EXPECT_EQ(gpu.total, cpu.total); EXPECT_EQ(gpu.words_needed, cpu.words_needed);
        const u64 valid = std::min(count, capacity);
        cuda::check(cudaMemcpy(actual.data(), dout.data(), valid * RW * 8, cudaMemcpyDeviceToHost), "numeric output");
        EXPECT_TRUE(std::equal(expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(valid * RW), actual.begin()));
        auto check = [&](const auto& expected_array, const cuda::DeviceBuffer& src, u64 n)
        {
            using Value = typename std::decay_t<decltype(expected_array)>::value_type;
            std::vector<Value> got(n);
            cuda::check(cudaMemcpy(got.data(), src.data(), n * sizeof(Value), cudaMemcpyDeviceToHost), "numeric labels");
            EXPECT_TRUE(std::equal(got.begin(), got.end(), expected_array.begin()));
        };
        check(parent, dp, valid); check(schema, ds, valid); check(binding, db, valid * L); check(goal, dg, valid); check(offsets, di, offsets.size());
    }
}

class DeviceNumeric : public testing::TestWithParam<std::string> {};

TEST_P(DeviceNumeric, ExpandEqualsCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const auto matching : {TaskOptions::Matching::FixedOrder, TaskOptions::Matching::ForwardChecking})
            for (const auto storage : {TaskOptions::NumericStorageMode::Auto, TaskOptions::NumericStorageMode::F64})
            {
                TaskOptions options; options.atoms = atoms; options.matching = matching; options.numeric_storage = storage;
                const auto task = task_of(GetParam(), options);
                const auto states = walks(*task);
                for (bool witness : {false, true})
                    compare_expand(ctx, task, states, witness, 3);
            }
}

TEST_P(DeviceNumeric, BrfsEqualsCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (const auto storage : {TaskOptions::NumericStorageMode::Auto, TaskOptions::NumericStorageMode::F64})
    {
        TaskOptions to; to.numeric_storage = storage;
        const auto task = task_of(GetParam(), to);
        BrfsOptions cpu; cpu.max_depth = 4; cpu.fingerprint = true;
        const auto expected = brfs(*task, cpu);
        cuda::DeviceBrfsOptions options; options.max_depth = 4; options.fingerprint = true; options.chunk_states = 17;
        const auto actual = cuda::brfs(ctx, task, options).result;
        EXPECT_EQ(actual.states, expected.states); EXPECT_EQ(actual.generated, expected.generated);
        EXPECT_EQ(actual.expanded, expected.expanded); EXPECT_EQ(actual.goal_states, expected.goal_states);
        EXPECT_EQ(actual.fingerprint, expected.fingerprint);
    }
}

TEST_P(DeviceNumeric, IwEqualsCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    const auto task = task_of(GetParam());
    const auto starts = walks(*task, 4);
    for (u32 width : {1u, 2u})
    {
        cuda::MultiIwOptions options; options.max_arity = width; options.budget.max_states = 20000;
        options.chunk_states = 7; options.track_reached = true;
        const auto results = cuda::multi_iw(ctx, task, starts, options);
        search::IwOptions cpu; cpu.max_arity = width; cpu.control.budget.max_states = 20000;
        for (u32 i = 0; i < starts.size(); ++i)
        {
            cpu.start = starts[i];
            const auto expected = search::iw(*task, cpu);
            EXPECT_EQ(results[i].status, expected.status);
            EXPECT_EQ(results[i].plan.size(), expected.plan.size());
            if (expected.status == search::SearchStatus::Solved)
            {
                EXPECT_EQ(results[i].cost, expected.cost);
                EXPECT_EQ(results[i].goal_state, expected.goal_state);
            }
        }
    }
}

void compare_env_and_rollouts(const cuda::ContextPtr& ctx, const TaskPtr& task)
{
    const auto table = rl::TaskTable::single(task);
    for (bool autoreset : {false, true})
        for (bool custom : {false, true})
        {
            if (custom && task->compiled().goal.uses_derived) continue;
            rl::EnvConfig cfg; cfg.autoreset = autoreset; cfg.max_steps = 7; cfg.seed = 17;
            cuda::DeviceEnv device(ctx, table, cfg);
            if (task->numeric_slots())
            {
                EXPECT_FALSE(device.fast());
            }
            rl::HostEnv host(table, cfg);
            const std::vector<i32> ids(4, 0);
            test::HostEnvs h(*table, ids, custom);
            test::DeviceEnvs d(ctx, device, ids, custom);
            host.reset(h.b, nullptr, h.v.count.data());
            device.reset(d.b, nullptr, static_cast<i32*>(d.count.data()));
            EXPECT_EQ(test::to_host<u64>(d.states.data(), h.v.states.size(), d.s), h.v.states);
            for (int step = 0; step < 20; ++step)
            {
                if (step % 2)
                {
                    const auto actions = test::actions(h.v.count, step);
                    test::upload(d.action.data(), actions, d.s);
                    host.step(h.b, h.out, rl::Actions{actions.data(), nullptr});
                    device.step(d.b, d.out, rl::Actions{static_cast<const i64*>(d.action.data()), nullptr});
                }
                else
                {
                    host.step(h.b, h.out);
                    device.step(d.b, d.out);
                }
                test::expect_equal(d.snapshot(), h.v, step);
            }
            EXPECT_NO_THROW(device.check_errors());
        }
    for (u32 next : {3u, ~u32{0}})
    {
        cuda::DeviceRolloutOptions options;
        options.seeds = {17, 1017, 2017, 3017};
        options.iw.max_arity = 2; options.iw.max_next_layer_states = next; options.iw.budget.max_states = 20000;
        search::ParallelRolloutOptions cpu;
        cpu.seeds = options.seeds; cpu.iw.max_arity = 2; cpu.max_next_layer_states = next;
        cpu.iw.control.budget.max_states = 20000; cpu.num_threads = 2;
        search::ParallelRolloutsResult expected;
        if (!task->numeric_slots())
            expected = search::find_rollouts_parallel(*task, cpu);
        else
            for (u64 seed : cpu.seeds)
            {
                auto iw = cpu.iw;
                iw.layers = {.kind = search::LayerOrdering::Kind::Randomized, .seed = seed,
                                  .max_next_layer_states = next};
                search::RolloutResult rollout;
                rollout.search = search::iw(*task, iw);
                expected.rollouts.push_back(std::move(rollout));
            }
        const auto actual = cuda::find_rollouts(ctx, task, options);
        ASSERT_EQ(actual.rollouts.size(), expected.rollouts.size());
        for (u32 i = 0; i < actual.rollouts.size(); ++i)
        {
            const auto& a = actual.rollouts[i]; const auto& b = expected.rollouts[i];
            EXPECT_EQ(a.search.status, b.search.status);
            EXPECT_EQ(a.search.plan.size(), b.search.plan.size());
            if (!task->numeric_slots())
            {
                EXPECT_EQ(a.reached_fluent_atoms, b.reached_fluent_atoms);
            }
            if (b.search.status == search::SearchStatus::Solved)
            {
                EXPECT_EQ(a.search.goal_state, b.search.goal_state);
                EXPECT_EQ(a.search.cost, b.search.cost);
                for (u32 k = 0; k < a.search.plan.size(); ++k)
                {
                    EXPECT_EQ(a.search.plan[k].schema, b.search.plan[k].schema);
                    EXPECT_EQ(a.search.plan[k].binding, b.search.plan[k].binding);
                }
            }
        }
    }
}

TEST_P(DeviceNumeric, EnvAndRolloutsEqualCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (const auto storage : {TaskOptions::NumericStorageMode::Auto, TaskOptions::NumericStorageMode::F64})
    {
        TaskOptions options; options.numeric_storage = storage;
        compare_env_and_rollouts(ctx, task_of(GetParam(), options));
    }
}

INSTANTIATE_TEST_SUITE_P(Tasks, DeviceNumeric, testing::ValuesIn(numeric_tasks()),
    [](const auto& info) { std::string name = info.param; std::replace(name.begin(), name.end(), '-', '_'); return name; });

TEST(DeviceNumericRules, CanonicalValuesAndPrograms)
{
    const auto task = task_of("cs-farmland");
    const auto arrays = rl::device_arrays(*task);
    const auto view = rl::task_view(arrays);
    const State initial = task->initial_state();
    std::vector<u64> row(task->words() + task->numeric_slots());
    cuda::numeric::encode(*task, initial.view(), row.data(), static_cast<u32>(row.size()));
    std::vector<ObjectId> binding(task->compiled().max_bind, ObjectId{0});
    const auto& n = task->compiled().num;
    const auto check = [&](const plan::NumCheck& c)
    {
        for (const auto program : {c.lhs, c.rhs})
        {
            const f64 expected = plan::eval(n, program, initial.numeric().data(), binding.data());
            const f64 actual = rl::dev::numeric_eval(view.numeric, program.begin, program.end, row.data() + task->words(),
                                                    reinterpret_cast<const u32*>(binding.data()), view.num_objects);
            EXPECT_EQ(std::bit_cast<u64>(actual), std::bit_cast<u64>(expected));
        }
    };
    for (const auto& schema : task->compiled().schemas)
        for (const auto& m : schema.pre)
        {
            for (const auto& c : m.npre) check(c);
            for (const auto& c : m.nchecks) check(c);
        }
    EXPECT_EQ(std::bit_cast<u64>(rl::dev::numeric_canonical(view.numeric, -0.0)), u64{0});
    EXPECT_EQ(std::bit_cast<u64>(rl::dev::numeric_canonical(view.numeric, -std::numeric_limits<f64>::quiet_NaN())),
              std::bit_cast<u64>(plan::canonical(n, std::numeric_limits<f64>::quiet_NaN())));
}
const char* kDomain = R"(
(define (domain nd)
 (:requirements :strips :numeric-fluents :conditional-effects :action-costs)
 (:predicates (p) (q))
 (:functions (x) (y) (z) (u) (total-cost))
 (:action inc :parameters () :precondition (and) :effect (and (increase (x) 1)))
 (:action make-p :parameters () :precondition (and) :effect (and (p) (not (q))))
 (:action inc-undefined :parameters () :precondition (and) :effect (and (increase (u) 1)))
 (:action assign-undefined :parameters () :precondition (and) :effect (and (assign (u) 1)))
 (:action assign-and-increase :parameters () :precondition (and) :effect (and (increase (x) 1) (assign (x) 2)))
 (:action two-increases :parameters () :precondition (and) :effect (and (increase (x) 1) (increase (x) 2)))
 (:action increase-decrease :parameters () :precondition (and) :effect (and (increase (x) 1) (decrease (x) 3)))
 (:action two-scales :parameters () :precondition (and) :effect (and (scale-up (y) 2) (scale-up (y) 3)))
 (:action increase-scale :parameters () :precondition (and) :effect (and (increase (y) 1) (scale-up (y) 2)))
 (:action divide-by-zero :parameters () :precondition (and) :effect (and (assign (y) (/ (x) (z)))))
 (:action ce-not-firing :parameters () :precondition (and) :effect (and (increase (x) 1) (when (p) (assign (x) 5))))
 (:action ce-firing :parameters () :precondition (and) :effect (and (increase (x) 1) (when (q) (assign (x) 5))))
 (:action ce-families :parameters () :precondition (and) :effect (and (when (p) (assign (y) 5)) (when (q) (increase (y) 1))))
 (:action ce-families-conflict :parameters () :precondition (and) :effect (and (when (q) (assign (y) 5)) (when (p) (increase (y) 1))))
 (:action fluent-cost :parameters () :precondition (and) :effect (and (increase (y) 1) (increase (total-cost) (x))))
 (:action undefined-cost :parameters () :precondition (and) :effect (and (increase (y) 1) (increase (total-cost) (u))))
 (:action conditional-cost :parameters () :precondition (and)
   :effect (and (increase (y) 1) (increase (total-cost) 1) (when (q) (increase (total-cost) 10)) (when (p) (increase (total-cost) 100))))
 (:action numeric-pre :parameters () :precondition (and (>= (x) 1)) :effect (and (increase (y) 1)))
 (:action undefined-pre :parameters () :precondition (and (>= (u) 0)) :effect (and (increase (y) 1)))
 (:action negative-zero :parameters () :precondition (and) :effect (and (assign (y) (* -1 (z)))))
 (:action nan-value :parameters () :precondition (and) :effect (and (scale-down (z) 0)))
 (:action zero :parameters () :precondition (and) :effect (and (assign (y) 0)))
)
)";

const char* kProblem = R"(
(define (problem np) (:domain nd)
 (:init (q) (= (x) 1) (= (y) 2) (= (z) 0) (= (total-cost) 0))
 (:goal (and (>= (x) 3)))
 (:metric minimize (total-cost))
)
)";

std::shared_ptr<const Task> rules_task()
{
    const auto domain = frontend::Domain::from_string(kDomain, "nd.pddl");
    return Task::create(*domain->instantiate_string(kProblem, "np.pddl"));
}

TEST(DeviceNumericRules, EffectApplicabilityAndCanonicalZeros)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    const auto task = rules_task();
    compare_expand(ctx, task, walks(*task, 16), false, 3);
    compare_expand(ctx, task, walks(*task, 16), true, 3);
}
}  // namespace

TEST(DeviceNumericRules, ExhaustiveCountsOnFiniteTasks)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (const char* name : {"cs-counters", "cs-drone", "cs-farmland", "cs-tpp", "m-refuel-adl",
                             "m-tpp-numeric", "m-woodworking", "m-barman", "m-transport"})
    {
        const auto task = task_of(name);
        BrfsOptions cpu; cpu.fingerprint = true;
        const auto expected = brfs(*task, cpu);
        cuda::DeviceBrfsOptions options; options.fingerprint = true; options.chunk_states = 257;
        const auto actual = cuda::brfs(ctx, task, options).result;
        EXPECT_TRUE(actual.exhausted) << name;
        EXPECT_EQ(actual.states, expected.states) << name;
        EXPECT_EQ(actual.generated, expected.generated) << name;
        EXPECT_EQ(actual.goal_states, expected.goal_states) << name;
        EXPECT_EQ(actual.fingerprint, expected.fingerprint) << name;
    }
}

TEST(DeviceNumericRules, QuantizationAndTolerantComparisons)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (bool tolerant : {false, true})
        for (f64 quantum : {0.0, 0.01})
        {
            TaskOptions options;
            options.numeric_tolerant = tolerant;
            options.numeric_quantum = quantum;
            options.numeric_storage = TaskOptions::NumericStorageMode::F64;
            const auto task = task_of("cs-sailing", options);
            compare_expand(ctx, task, walks(*task, 12), true, 3);
        }
}

class DeviceNumericPddl : public testing::TestWithParam<std::string> {};

TEST_P(DeviceNumericPddl, ExpandBrfsAndIwEqualCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto name = GetParam();
    const char* work_env = std::getenv("MYMYR_WORK_DIR");
    const char* fork_env = std::getenv("MYMYR_FORK_DATA_DIR");
    if (!work_env || !fork_env) GTEST_SKIP() << "set MYMYR_WORK_DIR and MYMYR_FORK_DATA_DIR";
    std::filesystem::path domain, problem;
    if (name.starts_with("cs-"))
    {
        std::string d = name.substr(3);
        if (d == "block-grouping-conj") d = "block-grouping";
        if (d == "delivery-nometric") d = "delivery";
        const auto dir = std::filesystem::path(work_env) / "mimir-cs/Benchmark/numeric" / d;
        domain = dir / "domain.pddl";
        problem = dir / "pfile1.pddl";
        if (name == "cs-block-grouping-conj" || name == "cs-delivery-nometric")
            problem = std::filesystem::path(MYMYR_SOURCE_DIR) / "tests/data/numeric_tasks/pddl" / (name.substr(3) + ".pddl");
    }
    else
    {
        std::string d = name.substr(2);
        if (d == "tpp-numeric") d = "tpp/numeric";
        if (d == "zenotravel-numeric") d = "zenotravel/numeric";
        const auto dir = std::filesystem::path(fork_env) / d;
        domain = dir / "domain.pddl";
        problem = dir / "test_problem.pddl";
    }
    if (!std::filesystem::exists(domain) || !std::filesystem::exists(problem)) GTEST_SKIP() << problem;
    const auto parsed = frontend::Domain::from_file(domain);
    const auto task = Task::create(*parsed->instantiate_file(problem));
    const auto ctx = context();
    const auto states = walks(*task);
    compare_env_and_rollouts(ctx, task);
    compare_expand(ctx, task, states, true, 3);
    BrfsOptions cpu; cpu.max_depth = 4; cpu.fingerprint = true;
    const auto expected = brfs(*task, cpu);
    cuda::DeviceBrfsOptions options; options.max_depth = 4; options.fingerprint = true;
    const auto actual = cuda::brfs(ctx, task, options).result;
    EXPECT_EQ(actual.states, expected.states); EXPECT_EQ(actual.generated, expected.generated);
    EXPECT_EQ(actual.goal_states, expected.goal_states); EXPECT_EQ(actual.fingerprint, expected.fingerprint);
    cuda::MultiIwOptions iw; iw.max_arity = 2; iw.budget.max_states = 20000;
    const auto results = cuda::multi_iw(ctx, task, std::span(states).first(4), iw);
    search::IwOptions reference; reference.max_arity = 2; reference.control.budget.max_states = 20000;
    for (u32 i = 0; i < results.size(); ++i)
    {
        reference.start = states[i];
        const auto want = search::iw(*task, reference);
        EXPECT_EQ(results[i].status, want.status); EXPECT_EQ(results[i].plan.size(), want.plan.size());
        if (want.status == search::SearchStatus::Solved)
        {
            EXPECT_EQ(results[i].cost, want.cost);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Tasks, DeviceNumericPddl, testing::ValuesIn(numeric_tasks()),
    [](const auto& info) { std::string name = info.param; std::replace(name.begin(), name.end(), '-', '_'); return name; });

TEST(DeviceNumericRules, Int32OverflowIsReportedBeforeWriting)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto domain = frontend::Domain::from_string(R"(
(define (domain overflow) (:requirements :strips :numeric-fluents)
 (:predicates (p))
 (:functions (x))
 (:action inc :parameters () :precondition (and) :effect (and (increase (x) 1)))
)
)", "overflow.pddl");
    const auto data = domain->instantiate_string(R"(
(define (problem overflow1) (:domain overflow)
 (:init (= (x) 2147483647)) (:goal (and (>= (x) 2147483648)))
)
)", "overflow1.pddl");
    const auto task = Task::create(*data);
    ASSERT_EQ(task->numeric_storage(), NumericStorage::I32);
    EXPECT_THROW((void)task->workspace().successors().applicable_actions(task->initial_state().view()), std::overflow_error);
    cuda::DeviceBrfsOptions options; options.max_depth = 1;
    EXPECT_THROW((void)cuda::brfs(context(), task, options), std::overflow_error);
    TaskOptions f64; f64.numeric_storage = TaskOptions::NumericStorageMode::F64;
    const auto wide = Task::create(*data, f64);
    EXPECT_EQ(cuda::brfs(context(), wide, options).result.states, 2u);
}

TEST(DeviceNumericRules, FiniteStateSpacesEqualCpu)
{
    if (cuda::device_count() == 0) GTEST_SKIP();
    const auto ctx = context();
    for (const char* name : {"cs-counters", "cs-drone", "cs-farmland", "cs-tpp", "m-refuel-adl",
                             "m-tpp-numeric", "m-woodworking", "m-barman", "m-transport"})
        for (const auto storage : {TaskOptions::NumericStorageMode::Auto, TaskOptions::NumericStorageMode::F64})
        {
            SCOPED_TRACE(name);
            TaskOptions options; options.numeric_storage = storage;
            const auto task = task_of(name, options);
            datasets::StateSpaceOptions cpu; cpu.remove_if_unsolvable = false; cpu.threads = 2;
            const auto expected = datasets::generate_state_space(task, cpu);
            cuda::DeviceStateSpaceOptions gpu; gpu.space = cpu; gpu.chunk_states = 257;
            gpu.output = cuda::StateSpaceOutput::Both;
            const auto actual = cuda::state_space(ctx, task, gpu);
            ASSERT_EQ(actual.status, expected.status);
            ASSERT_TRUE(actual.host && actual.space && expected.space);
            const auto compare = [&](const datasets::StateSpace& a)
            {
                EXPECT_EQ(a.num_states(), expected.space->num_states());
                EXPECT_EQ(a.num_transitions(), expected.space->num_transitions());
                const auto same = [](auto x, auto y) { return std::equal(x.begin(), x.end(), y.begin(), y.end()); };
                EXPECT_TRUE(same(a.state_words(), expected.space->state_words()));
                EXPECT_TRUE(same(a.forward_offsets(), expected.space->forward_offsets()));
                EXPECT_TRUE(same(a.forward_targets(), expected.space->forward_targets()));
                EXPECT_TRUE(same(a.label_schemas(), expected.space->label_schemas()));
                EXPECT_TRUE(same(a.label_bindings(), expected.space->label_bindings()));
                EXPECT_TRUE(same(a.costs(), expected.space->costs()));
                EXPECT_TRUE(same(a.goal_flags(), expected.space->goal_flags()));
                EXPECT_TRUE(same(a.unit_goal_distances(), expected.space->unit_goal_distances()));
                EXPECT_TRUE(same(a.cost_goal_distances(), expected.space->cost_goal_distances()));
            };
            compare(*actual.host); compare(*actual.space->to_host());
            const auto table = cuda::state_spaces(ctx, rl::TaskTable::single(task), gpu);
            ASSERT_EQ(table.results.front().status, expected.status);
            ASSERT_TRUE(table.results.front().host);
            compare(*table.results.front().host);
        }
}
