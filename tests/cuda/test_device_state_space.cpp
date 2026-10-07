// Device state-space tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device): device state spaces
// (cuda/state_space.hpp) against the host's datasets::generate_state_space, array by array (ids, forward and reverse CSR with
// labels and costs, flags, unit and cost goal distances, layers; state words byte for byte under frozen slots,
// canonically under lazy ones):
//   - the lifted suite (tasks up to 10^5 states; MYMYR_TEST_FULL_SUITE=1: all 22) and the fork's test instances
//     (conditional effects, axioms, action costs), frozen and lazy, with tiny chunks too;
//   - transition costs: integral and real state-independent costs (device tables), state-dependent costs (the CPU);
//   - the host's options: max_states, remove_if_unsolvable, statically false goals, labels, max_seconds;
//   - the table instance sets (tests/cpp/rl/table_instance_sets.hpp) generated together over a table equal their single runs and
//     the CPU, for any chunk size, wave size and stream;
//   - generalized state spaces and samplers of device results equal the host's;
//   - IW over tables (cuda::DeviceTableIw, multi_iw and batched_iw1 with task ids) equals the per-instance device runs
//     and search::iw().

#include "../cpp/frontend/golden.hpp"
#include "../cpp/rl/table_instance_sets.hpp"
#include "../cpp/support/suite.hpp"
#include "../cpp/support/wide_costs.hpp"
#include "mymyr/core/bitset.hpp"
#include "mymyr/cuda/generator.hpp"
#include "mymyr/cuda/multi_iw.hpp"
#include "mymyr/cuda/state_space.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/search/iw.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (cuda::device_count() == 0)                                                                                  \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

/// compute-sanitizer runs (MYMYR_TEST_SANITIZER=1) keep to small sizes.
bool sanitizer_size()
{
    const char* e = std::getenv("MYMYR_TEST_SANITIZER");
    return e && std::string(e) == "1";
}

cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{2} << 30;  // the GPU is shared
    return cuda::DeviceContext::create(0, o);
}

std::filesystem::path data(const std::string& rel) { return test::fork_data_dir() / rel; }

TaskPtr fork_task(const std::string& dir, const std::string& problem, TaskOptions::Atoms atoms = TaskOptions::Atoms::Frozen)
{
    const auto d = data(dir + "/domain.pddl"), p = data(dir + "/" + problem);
    if (!std::filesystem::exists(d) || !std::filesystem::exists(p))
        return nullptr;
    TaskOptions to;
    to.atoms = atoms;
    return Task::create(*frontend::load_task(d, p), to);
}

TaskPtr pddl_task(const char* domain, const char* problem, TaskOptions::Atoms atoms = TaskOptions::Atoms::Frozen)
{
    TaskOptions to;
    to.atoms = atoms;
    return Task::create(*frontend::Domain::from_string(domain, "d.pddl")->instantiate_string(problem, "p.pddl"), to);
}

StateSpaceOptions cpu_options(bool remove = true)
{
    StateSpaceOptions o;
    o.threads = 8;
    o.remove_if_unsolvable = remove;
    return o;
}

cuda::DeviceStateSpaceOptions dev_options(const StateSpaceOptions& space, u32 chunk = u32{1} << 20)
{
    cuda::DeviceStateSpaceOptions o;
    o.space = space;
    o.space.threads = 4;
    o.output = cuda::StateSpaceOutput::Both;
    o.chunk_states = chunk;
    return o;
}

template<class T>
bool same(std::span<const T> a, std::span<const T> b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}

/// The host's space `a` and the device's `b`: every array equal (state words byte for byte when `words`, else by canonical
/// state hashes: lazy slots number atoms in the order they are reached).
void expect_equal(const StateSpace& a, const StateSpace& b, bool words, const std::string& what)
{
    ASSERT_EQ(a.num_states(), b.num_states()) << what;
    ASSERT_EQ(a.num_transitions(), b.num_transitions()) << what;
    EXPECT_EQ(a.words(), b.words()) << what;
    EXPECT_EQ(a.row_words(), b.row_words()) << what;
    if (words)
        EXPECT_TRUE(same(a.state_words(), b.state_words())) << what << ": state words";
    else
        for (u32 i = 0; i < a.num_states(); ++i)
            if (a.task()->canonical_hash(a.state(i)) != b.task()->canonical_hash(b.state(i)))
            {
                ADD_FAILURE() << what << ": state " << i << " differs";
                break;
            }
    EXPECT_TRUE(same(a.forward_offsets(), b.forward_offsets())) << what << ": forward offsets";
    EXPECT_TRUE(same(a.forward_targets(), b.forward_targets())) << what << ": forward targets";
    EXPECT_EQ(a.has_labels(), b.has_labels()) << what;
    EXPECT_EQ(a.label_width(), b.label_width()) << what;
    EXPECT_TRUE(same(a.label_schemas(), b.label_schemas())) << what << ": schemas";
    EXPECT_TRUE(same(a.label_bindings(), b.label_bindings())) << what << ": bindings";
    EXPECT_EQ(a.unit_costs(), b.unit_costs()) << what;
    EXPECT_TRUE(same(a.costs(), b.costs())) << what << ": costs";
    EXPECT_TRUE(same(a.backward_offsets(), b.backward_offsets())) << what << ": backward offsets";
    EXPECT_TRUE(same(a.backward_sources(), b.backward_sources())) << what << ": backward sources";
    EXPECT_TRUE(same(a.backward_edges(), b.backward_edges())) << what << ": backward edges";
    EXPECT_TRUE(same(a.unit_goal_distances(), b.unit_goal_distances())) << what << ": unit V*";
    EXPECT_TRUE(same(a.cost_goal_distances(), b.cost_goal_distances())) << what << ": cost V*";
    EXPECT_TRUE(same(a.goal_flags(), b.goal_flags())) << what << ": goal flags";
    EXPECT_TRUE(same(a.unsolvable_flags(), b.unsolvable_flags())) << what << ": unsolvable flags";
    EXPECT_TRUE(same(a.alive_flags(), b.alive_flags())) << what << ": alive flags";
    EXPECT_EQ(a.num_goal_states(), b.num_goal_states()) << what;
    EXPECT_EQ(a.num_unsolvable_states(), b.num_unsolvable_states()) << what;
    EXPECT_EQ(a.max_goal_distance(), b.max_goal_distance()) << what;
    EXPECT_EQ(a.layers(), b.layers()) << what;
}

/// The device run of `task` against the host's result `ref` (same status; equal spaces).
void expect_device_equals(const cuda::ContextPtr& ctx, const TaskPtr& task, const StateSpaceResult& ref,
                          const cuda::DeviceStateSpaceOptions& o, bool words, const std::string& what)
{
    const cuda::DeviceStateSpaceResult r = cuda::state_space(ctx, task, o);
    ASSERT_EQ(r.status, ref.status) << what << ": " << to_string(r.status) << " vs " << to_string(ref.status);
    if (r.status != StateSpaceStatus::Ok)
        return;
    ASSERT_TRUE(r.host && r.space) << what;
    expect_equal(*ref.space, *r.host, words, what);
    // the device arrays are the host ones
    const StateSpacePtr again = r.space->to_host();
    expect_equal(*r.host, *again, true, what + " (to_host)");
    EXPECT_EQ(r.space->num_states(), r.host->num_states());
    EXPECT_EQ(r.space->num_goal_states(), r.host->num_goal_states());
}

// ----------------------------------------------------------------------------------------------- the suite
TEST(DeviceStateSpace, SuiteEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const auto& t : test::suite())
        {
            if (t.states > (sanitizer_size() ? 50'000 : test::suite_state_limit()))
                continue;
            TaskOptions to;
            to.atoms = atoms;
            const bool frozen = atoms == TaskOptions::Atoms::Frozen;
            const std::string what = t.name + (frozen ? " frozen" : " lazy");
            const auto cpu_task = Task::from_text_file(test::task_path(t.name), to);
            const StateSpaceResult ref = generate_state_space(cpu_task, cpu_options(false));
            ASSERT_EQ(ref.status, StateSpaceStatus::Ok) << what;
            EXPECT_EQ(ref.space->num_states(), t.states) << what;
            // a fresh task: lazy slots intern in the device's order
            const auto task = frozen ? cpu_task : Task::from_text_file(test::task_path(t.name), to);
            expect_device_equals(ctx, task, ref, dev_options(cpu_options(false)), frozen, what);
            if (t.states <= 50000u)
            {
                const auto task2 = frozen ? cpu_task : Task::from_text_file(test::task_path(t.name), to);
                expect_device_equals(ctx, task2, ref, dev_options(cpu_options(false), 1000), frozen, what + " chunk 1000");
            }
        }
}

// ----------------------------------------------------------------------------------------------- the fork's instances
struct ForkCase
{
    const char* dir;
    const char* problem;
    u32 states;
    u64 transitions;
    u32 goal, unsolvable;
    i32 max_goal_dist;
    u64 fp_content, fp_transitions, fp_vstar;
    f64 cost_sum;
};
const ForkCase kForkCases[] = {
#include "../cpp/datasets/fork_cases.inc"
};

TEST(DeviceStateSpace, ForkInstancesEqualHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    u32 ran = 0, ce = 0, axioms = 0, costs = 0;
    for (const ForkCase& c : kForkCases)
        for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        {
            const TaskPtr cpu_task = fork_task(c.dir, c.problem, atoms);
            if (!cpu_task)
            {
                ADD_FAILURE() << "missing " << c.dir << "/" << c.problem << " (MYMYR_FORK_DATA_DIR)";
                continue;
            }
            if (!cuda::state_space_unsupported(*cpu_task).empty())
                continue;  // numeric
            const bool frozen = atoms == TaskOptions::Atoms::Frozen;
            for (const bool remove : {false, true})
            {
                const std::string what = std::string(c.dir) + "/" + c.problem + (frozen ? " frozen" : " lazy") +
                                         (remove ? " remove" : "");
                const StateSpaceResult ref = generate_state_space(cpu_task, cpu_options(remove));
                if (ref.status == StateSpaceStatus::Ok)
                {
                    EXPECT_EQ(ref.space->num_states(), c.states) << what;
                }
                for (const u32 chunk : {u32{1} << 20, 7u})
                {
                    const TaskPtr task = frozen ? cpu_task : fork_task(c.dir, c.problem, atoms);
                    expect_device_equals(ctx, task, ref, dev_options(cpu_options(remove), chunk), frozen,
                                         what + " chunk " + std::to_string(chunk));
                }
            }
            ++ran;
            const cuda::SchemaPlacement pl = cuda::ChunkGenerator::place(*cpu_task, false);
            ce += pl.ce_count + pl.host_ce_count > 0;
            axioms += cpu_task->has_axioms();
            costs += !heuristics::ActionCosts(*cpu_task).unit();
        }
    EXPECT_GT(ran, 60u);
    EXPECT_GT(ce, 0u);
    EXPECT_GT(axioms, 0u);
    EXPECT_GT(costs, 0u);
}

// ----------------------------------------------------------------------------------------------- costs
const char* kCountDomain = R"(
(define (domain count)
 (:requirements :strips :typing :action-costs)
 (:types cell)
 (:predicates (at ?c - cell) (next ?a ?b - cell) (never))
 (:functions (total-cost) (step-cost ?a ?b - cell))
 (:action step :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) (step-cost ?a ?b))))
 (:action jump :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) 10)))
)
)";

/// A ring of n cells with steps both ways; costs step-cost (from `cost(i)`), jumps 10.
std::string ring_problem(u32 n, const std::function<std::string(u32)>& cost)
{
    std::string p = "(define (problem p) (:domain count)\n (:objects";
    for (u32 i = 0; i < n; ++i)
        p += " c" + std::to_string(i);
    p += " - cell)\n (:init (at c0) (= (total-cost) 0)";
    for (u32 i = 0; i < n; ++i)
    {
        const u32 j = (i + 1) % n;
        p += " (next c" + std::to_string(i) + " c" + std::to_string(j) + ") (next c" + std::to_string(j) + " c" + std::to_string(i) + ")";
        p += " (= (step-cost c" + std::to_string(i) + " c" + std::to_string(j) + ") " + cost(i) + ")";
        p += " (= (step-cost c" + std::to_string(j) + " c" + std::to_string(i) + ") " + cost(i + n) + ")";
    }
    p += ")\n (:goal (at c" + std::to_string(n / 2) + "))\n (:metric minimize (total-cost)))\n";
    return p;
}

TEST(DeviceStateSpace, TransitionCostsEqualHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    // integral state-independent costs (device tables)
    const std::string integral = ring_problem(9, [](u32 i) { return std::to_string(1 + (i * 7) % 5); });
    // real costs: (g + c) - g is not c, and the cost distances add reals in the order Dijkstra does
    const std::string real = ring_problem(9, [](u32 i) { return "0." + std::to_string(1 + (i * 3) % 9) + "7"; });
    for (const std::string& p : {integral, real})
        for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        {
            const TaskPtr cpu = pddl_task(kCountDomain, p.c_str(), atoms);
            const StateSpaceResult ref = generate_state_space(cpu, cpu_options());
            ASSERT_EQ(ref.status, StateSpaceStatus::Ok);
            ASSERT_FALSE(ref.space->unit_costs());
            const TaskPtr task = atoms == TaskOptions::Atoms::Frozen ? cpu : pddl_task(kCountDomain, p.c_str(), atoms);
            expect_device_equals(ctx, task, ref, dev_options(cpu_options()), atoms == TaskOptions::Atoms::Frozen, "ring costs");
        }
    // state-dependent costs (a conditional total-cost effect): the CPU regenerates the costs
    const char* cond = R"(
(define (domain cond)
 (:requirements :strips :typing :action-costs :conditional-effects)
 (:types cell)
 (:predicates (at ?c - cell) (next ?a ?b - cell) (heavy))
 (:functions (total-cost))
 (:action step :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (when (heavy) (increase (total-cost) 3)) (increase (total-cost) 1)))
 (:action toggle :parameters () :precondition (and) :effect (and (heavy)))
)
)";
    const char* cond_p = R"(
(define (problem p) (:domain cond)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c0) (next c0 c1) (next c1 c2) (next c2 c3) (next c3 c0) (= (total-cost) 0))
 (:goal (at c2))
 (:metric minimize (total-cost)))
)";
    const TaskPtr dyn = pddl_task(cond, cond_p);
    ASSERT_FALSE(heuristics::ActionCosts(*dyn).state_independent());
    const StateSpaceResult ref = generate_state_space(dyn, cpu_options());
    ASSERT_EQ(ref.status, StateSpaceStatus::Ok);
    expect_device_equals(ctx, dyn, ref, dev_options(cpu_options()), true, "conditional costs");
    // The host's example: transitions c0 -> c3 and costs 1, 7, 10, 10 from state 0
    const TaskPtr count = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c0) (next c0 c1) (next c1 c2) (next c2 c3) (next c0 c3) (= (total-cost) 0)
        (= (step-cost c0 c1) 1) (= (step-cost c1 c2) 2) (= (step-cost c2 c3) 3) (= (step-cost c0 c3) 7))
 (:goal (at c3))
 (:metric minimize (total-cost)))
)");
    const cuda::DeviceStateSpaceResult r = cuda::state_space(ctx, count, dev_options(cpu_options()));
    ASSERT_TRUE(r.host);
    EXPECT_EQ(std::vector<f64>(r.host->costs().begin(), r.host->costs().begin() + 4), (std::vector<f64>{1, 7, 10, 10}));
    EXPECT_DOUBLE_EQ(r.host->cost_goal_distances()[0], 6.0);
    // an undefined cost (a missing step-cost): the CPU drops the action, the device refuses
    const TaskPtr undefined = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 - cell)
 (:init (at c0) (next c0 c1) (= (total-cost) 0))
 (:goal (at c1))
 (:metric minimize (total-cost)))
)");
    EXPECT_THROW((void)cuda::state_space(ctx, undefined, dev_options(cpu_options())), std::domain_error);
}

TEST(DeviceStateSpace, CostProgramsEqualHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    // Costs the tables (at most 4 parameters, 2^24 entries) left to the CPU
    for (const std::string& hop : {test::k_wide_integral, test::k_wide_real})
    {
        const std::string dom = test::wide_domain(hop);
        std::vector<TaskPtr> tasks;
        std::vector<StateSpaceResult> cpu;
        for (const u32 n : {30u, 31u})
        {
            const std::string what = "wide " + std::to_string(n) + (hop == test::k_wide_real ? " real" : " integral");
            const TaskPtr task = pddl_task(dom.c_str(), test::wide_problem(n).c_str());
            const heuristics::ActionCosts costs(*task);
            ASSERT_TRUE(costs.state_independent()) << what;
            ASSERT_EQ(costs.cost_parameters(0).size(), 5u) << what;
            cpu.push_back(generate_state_space(task, cpu_options()));
            ASSERT_EQ(cpu.back().status, StateSpaceStatus::Ok) << what;
            ASSERT_FALSE(cpu.back().space->unit_costs()) << what;
            expect_device_equals(ctx, task, cpu.back(), dev_options(cpu_options()), true, what);
            // on the device: no CPU cost regeneration
            cuda::DeviceStateSpaceStats st;
            (void)cuda::state_space(ctx, task, dev_options(cpu_options()), &st);
            EXPECT_EQ(st.host_cost_ms, 0.0) << what;
            tasks.push_back(task);
        }
        // two instances in one table: a program per instance
        const cuda::DeviceStateSpaces all = cuda::state_spaces(ctx, rl::TaskTable::create(tasks), dev_options(cpu_options()));
        ASSERT_EQ(all.results.size(), 2u);
        for (usize i = 0; i < 2; ++i)
        {
            ASSERT_EQ(all.results[i].status, StateSpaceStatus::Ok);
            ASSERT_TRUE(all.results[i].host);
            expect_equal(*cpu[i].space, *all.results[i].host, true, "wide table " + std::to_string(i));
        }
        EXPECT_EQ(all.stats.host_cost_ms, 0.0);
    }
    // a negative cost: the host's error (the post-processing), as with the CPU's costs
    const std::string negative = test::wide_domain(test::k_wide_negative);
    const TaskPtr neg = pddl_task(negative.c_str(), test::wide_problem(30).c_str());
    EXPECT_THROW((void)generate_state_space(neg, cpu_options()), std::domain_error);
    EXPECT_THROW((void)cuda::state_space(ctx, neg, dev_options(cpu_options())), std::domain_error);
}

// ----------------------------------------------------------------------------------------------- options
TEST(DeviceStateSpace, OptionsMatchHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const TaskPtr gripper = fork_task("gripper", "test_problem4.pddl");  // 256 states
    ASSERT_TRUE(gripper);
    for (const u64 m : {u64{256}, u64{257}, u64{10}, u64{0}, u64{100}, u64{171}})
    {
        StateSpaceOptions o = cpu_options();
        o.max_states = m;
        // The host's one-thread generator (the instance pool's) fails after the parent whose successors reached the limit:
        // the device reports its count then, whatever the chunk
        StateSpaceOptions one = o;
        one.threads = 1;
        const StateSpaceResult ref = generate_state_space(gripper, one);
        EXPECT_EQ(generate_state_space(gripper, o).status, ref.status) << m;
        for (const u32 chunk : {u32{1} << 20, u32{5}, u32{1}})
        {
            const cuda::DeviceStateSpaceResult d = cuda::state_space(ctx, gripper, dev_options(o, chunk));
            EXPECT_EQ(d.status, ref.status) << m << " chunk " << chunk;
            EXPECT_EQ(d.states, ref.states) << m << " chunk " << chunk;
        }
    }
    const TaskPtr deadend = fork_task("deadend", "test_problem.pddl");  // the initial state is a dead end
    ASSERT_TRUE(deadend);
    for (const bool remove : {true, false})
    {
        const StateSpaceResult ref = generate_state_space(deadend, cpu_options(remove));
        expect_device_equals(ctx, deadend, ref, dev_options(cpu_options(remove)), true, remove ? "deadend remove" : "deadend");
    }
    const TaskPtr never = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 - cell)
 (:init (at c0) (next c0 c1) (= (total-cost) 0) (= (step-cost c0 c1) 1))
 (:goal (and (at c1) (never)))
 (:metric minimize (total-cost)))
)");
    for (const bool remove : {true, false})
        EXPECT_EQ(cuda::state_space(ctx, never, dev_options(cpu_options(remove))).status, StateSpaceStatus::Unsolvable);
    // without labels
    const TaskPtr spanner = fork_task("spanner", "p15-easy.pddl");
    ASSERT_TRUE(spanner);
    StateSpaceOptions o = cpu_options();
    o.labels = false;
    expect_device_equals(ctx, spanner, generate_state_space(spanner, o), dev_options(o), true, "spanner without labels");
    // host output only, device output only
    cuda::DeviceStateSpaceOptions d = dev_options(cpu_options());
    d.output = cuda::StateSpaceOutput::Host;
    const cuda::DeviceStateSpaceResult h = cuda::state_space(ctx, spanner, d);
    EXPECT_TRUE(h.host && !h.space);
    d.output = cuda::StateSpaceOutput::Device;
    const cuda::DeviceStateSpaceResult v = cuda::state_space(ctx, spanner, d);
    EXPECT_TRUE(!v.host && v.space);
    // max_seconds
    const auto depot = Task::from_text_file(test::task_path("depot__p02"));
    o = cpu_options();
    o.max_seconds = 0;
    EXPECT_EQ(cuda::state_space(ctx, depot, dev_options(o)).status, StateSpaceStatus::Timeout);
    // symmetry pruning requires the CPU generator
    o = cpu_options();
    o.symmetry_pruning = true;
    EXPECT_THROW((void)cuda::state_space(ctx, spanner, dev_options(o)), std::invalid_argument);
    const TaskPtr numeric = pddl_task(R"(
(define (domain counter)
 (:requirements :strips :numeric-fluents)
 (:predicates (done))
 (:functions (x))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1)))
)
)",
                                      R"(
(define (problem p) (:domain counter)
 (:init (= (x) 0))
 (:goal (done)))
)");
    EXPECT_TRUE(cuda::state_space_unsupported(*numeric).empty());
    EXPECT_NO_THROW((void)cuda::state_space(ctx, numeric, dev_options(cpu_options())));
}

// ----------------------------------------------------------------------------------------------- tables
struct Set
{
    std::string name;
    std::vector<TaskPtr> tasks;
    rl::TaskTablePtr table;
};

std::vector<Set> sets(TaskOptions::Atoms atoms)
{
    std::vector<Set> out;
    for (const test::InstanceSet& s : test::table_instance_sets())
    {
        std::vector<TaskPtr> tasks = test::load_set(s, atoms);
        if (tasks.empty())
        {
            ADD_FAILURE() << "missing PDDL of set " << s.name << " (MYMYR_FORK_DATA_DIR)";
            continue;
        }
        auto table = rl::TaskTable::create(tasks);
        out.push_back({s.name, std::move(tasks), std::move(table)});
    }
    return out;
}

/// Each instance of the table's run `rs` against its single run (same options) and the host.
void expect_table_equals(const cuda::ContextPtr& ctx, const Set& set, const std::vector<cuda::DeviceStateSpaceResult>& rs,
                         const std::vector<StateSpaceResult>& cpu, const cuda::DeviceStateSpaceOptions& o, bool frozen,
                         const std::string& what)
{
    ASSERT_EQ(rs.size(), set.tasks.size());
    for (usize i = 0; i < rs.size(); ++i)
    {
        const std::string w = what + " " + set.name + "[" + std::to_string(i) + "]";
        ASSERT_EQ(rs[i].status, cpu[i].status) << w << ": " << to_string(rs[i].status) << " vs " << to_string(cpu[i].status);
        if (rs[i].status == StateSpaceStatus::OutOfStates)
        {
            EXPECT_EQ(rs[i].states, cpu[i].states) << w << ": the count at the failure";
        }
        if (rs[i].status == StateSpaceStatus::Ok)
        {
            ASSERT_TRUE(rs[i].host) << w;
            expect_equal(*cpu[i].space, *rs[i].host, frozen, w + " vs CPU");
        }
        if (frozen)  // lazy: a single run would intern into the same tasks again
        {
            const cuda::DeviceStateSpaceResult one = cuda::state_space(ctx, set.tasks[i], o);
            ASSERT_EQ(one.status, rs[i].status) << w;
            if (one.status == StateSpaceStatus::Ok)
                expect_equal(*one.host, *rs[i].host, true, w + " vs single");
        }
    }
}

TEST(DeviceStateSpaces, SetsTogetherEqualSinglesAndCpu)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    StateSpaceOptions so = cpu_options(false);
    so.max_states = sanitizer_size() ? 4000 : 40000;  // the larger instances fail on both sides (OutOfStates)
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const bool frozen = atoms == TaskOptions::Atoms::Frozen;
        const std::vector<Set> fresh = frozen ? std::vector<Set>{} : sets(atoms);  // lazy: the CPU on tasks of its own
        for (const Set& set : sets(atoms))
        {
            std::vector<StateSpaceResult> cpu;
            const Set* cpu_set = &set;
            for (const Set& f : fresh)
                if (f.name == set.name)
                    cpu_set = &f;
            for (const TaskPtr& t : cpu_set->tasks)
            {
                cpu.push_back(generate_state_space(t, so));
                if (cpu.back().status == StateSpaceStatus::OutOfStates)
                {
                    // the count at the failure: the host's one-thread generator's (the instance pool's)
                    StateSpaceOptions one = so;
                    one.threads = 1;
                    cpu.back() = generate_state_space(t, one);
                }
            }
            u32 ok = 0;
            for (const auto& r : cpu)
                ok += r.status == StateSpaceStatus::Ok;
            EXPECT_GT(ok, 0u) << set.name;
            const cuda::DeviceStateSpaceOptions o = dev_options(so);
            const cuda::DeviceStateSpaces all = cuda::state_spaces(ctx, set.table, o);
            EXPECT_EQ(all.stats.multi, frozen && set.name != "miconic-simpleadl") << set.name;
            expect_table_equals(ctx, set, all.results, cpu, o, frozen, frozen ? "frozen" : "lazy");
            if (!frozen)
                continue;
            // chunk sizes, waves, a stream of its own: the same results
            cuda::DeviceStateSpaceOptions small = o;
            small.chunk_states = 777;
            small.wave_states = sanitizer_size() ? 1000 : 3000;
            expect_table_equals(ctx, set, cuda::state_spaces(ctx, set.table, small).results, cpu, o, true, "waves of 3000");
            cuda::DeviceStateSpaceOptions pairs = o;
            pairs.wave_instances = 2;
            pairs.chunk_states = 4096;
            const cuda::DeviceStateSpaces p = cuda::state_spaces(ctx, set.table, pairs);
            EXPECT_GE(p.stats.waves, 2u);
            expect_table_equals(ctx, set, p.results, cpu, o, true, "waves of 2 instances");
            const cuda::Stream stream;
            cuda::DeviceStateSpaceOptions own = o;
            own.stream = stream.get();
            own.view_bytes = 1 << 16;
            expect_table_equals(ctx, set, cuda::state_spaces(ctx, set.table, own).results, cpu, o, true, "own stream");
        }
    }
}

// ----------------------------------------------------------------------------------------------- generalized, samplers
TEST(DeviceStateSpaces, GeneralizedStateSpaceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    std::vector<TaskPtr> tasks;
    for (const char* p : {"test_problem4.pddl", "p-2-0.pddl", "p-1-0.pddl", "test_problem.pddl"})
        if (TaskPtr t = fork_task("gripper", p))
            tasks.push_back(t);
    ASSERT_EQ(tasks.size(), 4u);
    const std::vector<StateSpaceResult> cpu = generate_state_spaces(tasks, cpu_options(), 2);
    const auto ref = GeneralizedStateSpace::create(ordered_spaces(cpu));
    cuda::DeviceStateSpaceOptions o = dev_options(cpu_options());
    o.output = cuda::StateSpaceOutput::Device;  // generalized_state_space downloads
    const cuda::DeviceStateSpaces dev = cuda::state_spaces(ctx, rl::TaskTable::create(tasks), o);
    const auto g = cuda::generalized_state_space(dev.results);
    ASSERT_EQ(g->spaces().size(), ref->spaces().size());
    for (usize i = 0; i < g->spaces().size(); ++i)
    {
        EXPECT_EQ(g->spaces()[i]->task(), ref->spaces()[i]->task());
        expect_equal(*ref->spaces()[i], *g->spaces()[i], true, "generalized " + std::to_string(i));
    }
    EXPECT_EQ(g->num_vertices(), ref->num_vertices());
    EXPECT_EQ(g->num_edges(), ref->num_edges());
    EXPECT_TRUE(same(g->edge_sources(), ref->edge_sources()));
    EXPECT_TRUE(same(g->edge_targets(), ref->edge_targets()));
    EXPECT_TRUE(same(g->forward_offsets(), ref->forward_offsets()));
    EXPECT_TRUE(same(g->forward_edges(), ref->forward_edges()));
    EXPECT_TRUE(same(g->goal_flags(), ref->goal_flags()));
    EXPECT_TRUE(same(g->unsolvable_flags(), ref->unsolvable_flags()));
    EXPECT_TRUE(same(g->initial_flags(), ref->initial_flags()));
}

TEST(DeviceStateSpace, SamplersOverDeviceDistances)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    const TaskPtr spanner = fork_task("spanner", "p15-easy.pddl");
    ASSERT_TRUE(spanner);
    const cuda::DeviceStateSpaceResult r = cuda::state_space(ctx, spanner, dev_options(cpu_options()));
    ASSERT_TRUE(r.space && r.host);
    std::vector<i32> unit(r.space->num_states());
    cuda::check(cudaMemcpy(unit.data(), r.space->unit_goal_distances(), unit.size() * sizeof(i32), cudaMemcpyDeviceToHost),
                "cudaMemcpy");
    StateSpaceSampler a(r.host, 7), b(std::span<const i32>(unit), 7);
    EXPECT_EQ(a.num_states(), b.num_states());
    EXPECT_EQ(a.num_dead_end_states(), b.num_dead_end_states());
    EXPECT_EQ(a.max_steps_to_goal(), b.max_steps_to_goal());
    for (int k = 0; k < 100; ++k)
    {
        EXPECT_EQ(a.sample_state(), b.sample_state());
        EXPECT_EQ(a.sample_dead_end_state(), b.sample_dead_end_state());
        EXPECT_EQ(a.sample_state_n_steps_from_goal(3), b.sample_state_n_steps_from_goal(3));
    }
    EXPECT_FALSE(b.space());
}
// ----------------------------------------------------------------------------------------------- IW over tables
search::IwOptions iw_options(const cuda::MultiIwOptions& d, const State& start)
{
    search::IwOptions o;
    o.max_arity = d.max_arity;
    o.optimize_iw1 = d.optimize_iw1;
    o.width_zero = d.width_zero;
    o.witness_pruning = d.witness_pruning;
    o.canonical_order = d.canonical_order;
    o.control.budget = d.budget;
    o.start = start;
    return o;
}

void expect_same_iw(const search::IwResult& d, const search::IwResult& c, const std::string& what)
{
    SCOPED_TRACE(what);
    EXPECT_EQ(d.status, c.status);
    EXPECT_EQ(d.effective_width, c.effective_width);
    ASSERT_EQ(d.passes.size(), c.passes.size());
    for (usize k = 0; k < c.passes.size(); ++k)
    {
        EXPECT_EQ(d.passes[k].status, c.passes[k].status) << k;
        EXPECT_EQ(d.passes[k].expanded, c.passes[k].expanded) << k;
        EXPECT_EQ(d.passes[k].generated, c.passes[k].generated) << k;
        EXPECT_EQ(d.passes[k].generated_in_tree, c.passes[k].generated_in_tree) << k;
        EXPECT_EQ(d.passes[k].skipped, c.passes[k].skipped) << k;
    }
    EXPECT_TRUE(d.plan == c.plan) << "plans differ: " << d.plan.size() << " vs " << c.plan.size() << " steps";
    EXPECT_EQ(d.cost, c.cost);
    ASSERT_EQ(d.goal_state.has_value(), c.goal_state.has_value());
    if (c.goal_state)
    {
        EXPECT_TRUE(d.goal_state->view() == c.goal_state->view());
    }
}

TEST(DeviceStateSpaceIw, TablesEqualPerInstanceRunsAndCpu)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = context();
    u32 solved = 0, total = 0;
    for (const Set& set : sets(TaskOptions::Atoms::Frozen))
    {
        const rl::TaskTable& table = *set.table;
        std::vector<u64> rows;
        std::vector<i32> ids32;
        test::table_rows(table, 6, rows, ids32);  // 6 walk states per instance, interleaved
        const u32 W = table.words(), n = static_cast<u32>(ids32.size());
        const std::vector<u32> ids(ids32.begin(), ids32.end());
        std::vector<State> starts;
        for (u32 r = 0; r < n; ++r)
            starts.emplace_back(rows.data() + u64{r} * W, W);
        for (const u32 arity : {1u, 2u})
        {
            if (arity == 2 && set.name != "gripper" && set.name != "miconic")
                continue;
            cuda::MultiIwOptions o;
            o.max_arity = arity;
            o.budget.max_states = sanitizer_size() ? 3000 : 20000;  // the large instances run out on both sides
            const std::string what = set.name + " IW(" + std::to_string(arity) + ")";
            // the table's batch from device rows (on a stream of its own), and from host states
            cuda::DeviceBuffer dev(ctx, rows.size() * sizeof(u64));
            cuda::check(cudaMemcpy(dev.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice), "cudaMemcpy");
            const cuda::Stream stream;
            cuda::DeviceTableIw tiw(ctx, set.table, o);
            const cuda::DeviceStarts ds{static_cast<const u64*>(dev.data()), W, W, n};
            const cuda::MultiIwBatch b = tiw.run(ds, ids, {}, {}, stream.get());
            ASSERT_EQ(b.n, n);
            const std::vector<search::IwResult> hr = cuda::multi_iw(ctx, set.table, ids, starts, o);
            for (u32 r = 0; r < n; ++r)
            {
                const Task& task = *table.task(ids[r]);
                const search::IwResult d = b.result(r, task, starts[r], true);
                const search::IwResult c = search::iw(task, iw_options(o, starts[r]));
                expect_same_iw(d, c, what + " search " + std::to_string(r) + " (instance " + std::to_string(ids[r]) + ")");
                expect_same_iw(hr[r], c, what + " host starts " + std::to_string(r));
                solved += c.status == search::SearchStatus::Solved;
                ++total;
            }
            // the per-instance device runs
            for (u32 i = 0; i < table.size(); ++i)
            {
                std::vector<State> mine;
                std::vector<u32> where;
                for (u32 r = 0; r < n; ++r)
                    if (ids[r] == i)
                    {
                        mine.push_back(starts[r]);
                        where.push_back(r);
                    }
                cuda::DeviceMultiIw one(ctx, table.task(i), o);
                const cuda::MultiIwBatch x = one.run(mine);
                for (u32 j = 0; j < x.n; ++j)
                {
                    const u32 r = where[j];
                    EXPECT_EQ(x.status[j], b.status[r]) << what;
                    EXPECT_EQ(x.plan_length[j], b.plan_length[r]) << what;
                    EXPECT_EQ(x.effective_width[j], b.effective_width[r]) << what;
                    EXPECT_TRUE(x.plan(j, *table.task(i)) == b.plan(r, *table.task(i))) << what;
                }
            }
            // per-search goals: atoms of another walk state of the same instance that the start lacks
            std::vector<search::GoalSpec::AtomGoal> goals(n);
            for (u32 r = 0; r < n; ++r)
            {
                u32 t = r;
                for (u32 q = 1; q < n && t == r; ++q)
                    if (ids[(r + q) % n] == ids[r])
                        t = (r + q) % n;
                bits::for_each(starts[t].data(), starts[t].size_words(), [&](u64 a)
                               {
                                   if (goals[r].positive.size() < 2 && !bits::test(starts[r].data(), starts[r].size_words(), a))
                                       goals[r].positive.push_back(SlotId{static_cast<u32>(a)});
                               });
            }
            const cuda::MultiIwBatch bg = tiw.run(ds, ids, goals);
            for (u32 r = 0; r < n; ++r)
            {
                const Task& task = *table.task(ids[r]);
                search::IwOptions co = iw_options(o, starts[r]);
                co.control.goal.kind = search::GoalSpec::Kind::AnyOf;
                co.control.goal.goals = {goals[r]};
                const search::IwResult c = search::iw(task, co);
                expect_same_iw(bg.result(r, task, starts[r], true), c, what + " goals " + std::to_string(r));
                solved += c.status == search::SearchStatus::Solved;
            }
            // batched IW(1) over the table
            if (arity == 1)
            {
                const cuda::MultiIwBatch bi = cuda::batched_iw1(ctx, set.table, ds, ids, o);
                EXPECT_EQ(bi.status, b.status) << what;
                EXPECT_EQ(bi.plan_labels, b.plan_labels) << what;
                EXPECT_EQ(bi.goal_rows, b.goal_rows) << what;
            }
        }
        // task ids outside the table
        std::vector<u32> bad(ids);
        bad[0] = table.size();
        EXPECT_THROW((void)cuda::multi_iw(ctx, set.table, bad, starts), std::out_of_range);
    }
    EXPECT_GT(solved, 10u);
    EXPECT_GT(total, 100u);
}
}  // namespace
