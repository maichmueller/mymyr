// Device heuristics tests (run on GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device):
//   - heuristics: the device h_max and h_add equal heuristics::make_heuristic's on the states of random walks
//     and of CPU A* / GBFS frontiers of every suite task whose grounding fits the budget, frozen and lazy, unit and
//     real costs; the device h_FF equals DeviceHeuristic::reference (the CPU implementation of the device's supporter
//     rule; the agreement with the CPU's h_FF is printed as "FF_AGREE"); no value depends on the variant, the
//     group size, the grid, the scratch placement or how the states are split into launches; states outside the
//     grounding take the CPU fallback; numeric tasks, other kinds and groundings beyond the budget are refused;
//   - searches: device A* and GBFS at batch 1 are search::astar_eager and gbfs_eager (every statistic, the
//     plan); device A* (h_max, blind) finds the CPU's optimal cost with a valid plan on every suite task the CPU A*
//     solves within the budget ("ASTAR" lines: expansions of both), device GBFS (h_FF, h_add) valid plans
//     ("GBFS"); results do not depend on the chunking, the heuristic's launch configuration or the run; budgets,
//     start states and refusals;
//   - with the PDDL front end: heuristics and device A* on PDDL tasks with real action costs.
// MYMYR_TEST_SANITIZER=1 shrinks the tests for compute-sanitizer.
// MYMYR_TEST_FULL_SUITE=1 runs the gate's sizes.

#include "../cpp/support/device_ref.hpp"
#include "../cpp/support/suite.hpp"
#include "../cpp/support/wide_costs.hpp"
#include "mymyr/cuda/astar.hpp"
#include "mymyr/cuda/gbfs.hpp"
#include "mymyr/cuda/heuristics.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#if defined(MYMYR_DEVICE_HEURISTICS_FRONTEND)
#include "../cpp/frontend/golden.hpp"
#include "mymyr/frontend/domain.hpp"
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
#define SKIP_WITHOUT_GPU()                                                                                              \
    do                                                                                                                  \
    {                                                                                                                   \
        if (cuda::device_count() == 0)                                                                                  \
            GTEST_SKIP() << "no CUDA device";                                                                           \
    } while (0)

cuda::ContextPtr context()
{
    cuda::ContextOptions o;
    o.max_bytes = u64{3} << 30;  // GPU 0 may be shared: cap this process at 4 GB
    return cuda::DeviceContext::create(0, o);
}

TaskPtr load(const std::string& name, bool frozen)
{
    TaskOptions o;
    o.atoms = frozen ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
    return Task::from_text_file(task_path(name), o);
}

bool env_flag(const char* name)
{
    const char* e = std::getenv(name);
    return e && std::string(e) == "1";
}

bool full_suite() { return env_flag("MYMYR_TEST_FULL_SUITE"); }
bool sanitizer_size() { return env_flag("MYMYR_TEST_SANITIZER"); }

std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const SuiteTask& t : suite())
        for (bool frozen : {true, false})
            out.emplace_back(t.name, frozen);
    return out;
}

std::string param_name(const ::testing::TestParamInfo<std::tuple<std::string, bool>>& info)
{
    std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
    std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
    return n;
}

/// The grounding of a suite task, or nullptr beyond the budget (organic-synthesis).
std::shared_ptr<const heuristics::RelaxedTask> grounding(const Task& task)
{
    return heuristics::RelaxedTask::build(task, {}, nullptr);
}

/// Collects the new states a CPU search generates (its frontier and closed states), at most `cap`.
struct Collect final : search::SearchObserver
{
    std::vector<State>* out = nullptr;
    usize cap = 0;
    void on_generate(u64, const Action&, u64, StateView s, bool is_new) override
    {
        if (is_new && out->size() < cap)
            out->emplace_back(s);
    }
};

/// States from real searches: random walks, then what CPU A* (h_max) and GBFS (h_FF) generate within a budget.
std::vector<State> sample_states(const Task& task, u64 seed)
{
    const u32 walks = sanitizer_size() ? 2 : full_suite() ? 16 : 6;
    const u32 steps = sanitizer_size() ? 8 : full_suite() ? 60 : 30;
    const u64 expand = sanitizer_size() ? 10 : full_suite() ? 1500 : 150;
    std::vector<State> out = device_ref_walks(task, walks, steps, seed);
    for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::FF})
    {
        search::BestFirstOptions o;
        o.heuristic.kind = k;
        o.control.budget.max_expanded = expand;
        Collect c;
        c.out = &out;
        c.cap = out.size() + expand * 8;
        o.control.observer = &c;
        (void)(k == heuristics::Kind::Max ? search::astar_eager(task, o) : search::gbfs_eager(task, o));
    }
    return out;
}

/// Rows of `states` on the device, `W` words each (at least the task's width).
struct DeviceRows
{
    cuda::DeviceBuffer buf;
    u32 W = 1;
    u64 n = 0;
    [[nodiscard]] const u64* data() const { return static_cast<const u64*>(buf.data()); }
};

DeviceRows upload(const cuda::ContextPtr& ctx, const Task& task, const std::vector<State>& states, u32 extra_words = 0)
{
    DeviceRows r;
    r.W = std::max<u32>(1, task.words());
    for (const State& s : states)
        r.W = std::max(r.W, s.size_words());
    r.W += extra_words;
    r.n = states.size();
    std::vector<u64> rows(std::max<u64>(1, r.n * r.W), 0);
    for (usize i = 0; i < states.size(); ++i)
        std::copy_n(states[i].data(), states[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * r.W));
    r.buf = cuda::DeviceBuffer(ctx, rows.size() * sizeof(u64));
    cuda::check(cudaMemcpyAsync(r.buf.data(), rows.data(), rows.size() * sizeof(u64), cudaMemcpyHostToDevice, ctx->stream()),
                "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(ctx->stream()), "cudaStreamSynchronize");
    return r;
}

/// u32 values of rows [first, first + n) of `r`.
std::vector<u32> eval_u32(cuda::DeviceHeuristic& h, const DeviceRows& r, u64 first = 0, u64 n = ~u64{0})
{
    n = std::min(n, r.n - first);
    cuda::DeviceBuffer out(h.context(), std::max<u64>(n, 1) * sizeof(u32));
    h.evaluate(r.data() + first * r.W, r.W, r.W, n, static_cast<u32*>(out.data()));
    std::vector<u32> v(n);
    const cudaStream_t s = h.context()->stream();
    cuda::check(cudaMemcpyAsync(v.data(), out.data(), n * sizeof(u32), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    return v;
}

std::vector<f64> eval_f64(cuda::DeviceHeuristic& h, const DeviceRows& r)
{
    cuda::DeviceBuffer out(h.context(), std::max<u64>(r.n, 1) * sizeof(f64));
    h.evaluate(r.data(), r.W, r.W, r.n, static_cast<f64*>(out.data()));  // stream-ordered: not synchronized
    std::vector<f64> v(r.n);
    const cudaStream_t s = h.context()->stream();
    cuda::check(cudaMemcpyAsync(v.data(), out.data(), r.n * sizeof(f64), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync");
    cuda::check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
    return v;
}

/// heuristics::make_heuristic with the grounding R (Evaluation::Auto: its lifted fallback outside the grounding).
std::unique_ptr<heuristics::Heuristic> cpu_heuristic(const Task& task, heuristics::Kind k, heuristics::Costs c,
                                                     std::shared_ptr<const heuristics::RelaxedTask> R)
{
    heuristics::Options o;
    o.kind = k;
    o.costs = c;
    o.relaxed = std::move(R);
    return heuristics::make_heuristic(task, o);
}

u32 as_u32(f64 v) { return v == heuristics::k_dead_end ? cuda::DeviceHeuristic::k_dead_end : static_cast<u32>(v); }

/// Every operator of R an action of one positive cost (h_FF on the device then has no levels phase).
bool uniform_costs(const heuristics::RelaxedTask& R, heuristics::Costs c)
{
    u32 first = 0;
    for (u32 op = 0; op < R.num_ops(); ++op)
    {
        if (R.is_axiom(op))
            return false;
        const u32 v = c == heuristics::Costs::Real ? R.real_cost(R.ground_action(op)) : 1;
        if (v == 0 || (op > 0 && v != first))
            return false;
        first = v;
    }
    return R.num_ops() > 0;
}

const char* kind_name(heuristics::Kind k) { return heuristics::to_string(k); }

// ------------------------------------------------------------------------------------------------ equality

class CudaHeuristicSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(CudaHeuristicSuite, EqualsTheCpu)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const TaskPtr task = load(name, frozen);
    const auto R = grounding(*task);
    if (!R)
    {
        // beyond the budget: refused, as the CPU's Evaluation::Grounded
        cuda::DeviceHeuristicOptions o;
        o.kind = heuristics::Kind::Max;
        EXPECT_THROW((void)cuda::DeviceHeuristic(context(), task, o), std::invalid_argument);
        GTEST_SKIP() << "grounding beyond the budget";
    }
    const std::vector<State> states = sample_states(*task, 0x6b4 + (frozen ? 1 : 0));
    const cuda::ContextPtr ctx = context();
    std::vector<heuristics::Costs> costs{heuristics::Costs::Unit};
    if (!heuristics::ActionCosts(*task).unit() && R->real_costs_available())
        costs.push_back(heuristics::Costs::Real);
    for (heuristics::Costs c : costs)
        for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
        {
            SCOPED_TRACE(std::string(kind_name(k)) + (c == heuristics::Costs::Real ? " real" : " unit"));
            cuda::DeviceHeuristicOptions o;
            o.kind = k;
            o.costs = c;
            o.relaxed = R;
            cuda::DeviceHeuristic d(ctx, task, o);
            // the rows are packed after the heuristic exists: lazy slots of the samples are all assigned by now
            const DeviceRows rows = upload(ctx, *task, states);
            const std::vector<u32> got = eval_u32(d, rows);
            const std::vector<f64> got64 = eval_f64(d, rows);
            auto cpu = cpu_heuristic(*task, k, c, R);
            u64 agree = 0, failures = 0, lower = 0, finite = 0;
            f64 sum_device = 0, sum_cpu = 0;
            for (usize i = 0; i < states.size() && failures < 5; ++i)
            {
                const f64 want = k == heuristics::Kind::FF ? d.reference(states[i].view()) : cpu->evaluate(states[i].view());
                if (got[i] != as_u32(want) || got64[i] != want)
                {
                    ADD_FAILURE() << "state " << i << ": device " << got[i] << " / " << got64[i] << ", expected " << want;
                    ++failures;
                }
                if (k == heuristics::Kind::FF)
                {
                    const f64 v = cpu->evaluate(states[i].view());
                    agree += v == want ? 1 : 0;
                    lower += want < v ? 1 : 0;
                    if (v != heuristics::k_dead_end && want != heuristics::k_dead_end)
                    {
                        ++finite;
                        sum_device += want;
                        sum_cpu += v;
                    }
                }
            }
            EXPECT_EQ(d.stats().evaluations, 2 * states.size());
            EXPECT_EQ(d.stats().supporter_levels, k == heuristics::Kind::FF && !uniform_costs(*R, c));
            if (k == heuristics::Kind::FF)
                std::printf("FF_AGREE %s %s %s equal %llu/%zu device_lower %llu mean_device %.3f mean_cpu %.3f\n",
                            name.c_str(), frozen ? "frozen" : "lazy", c == heuristics::Costs::Real ? "real" : "unit",
                            static_cast<unsigned long long>(agree), states.size(), static_cast<unsigned long long>(lower),
                            finite ? sum_device / static_cast<f64>(finite) : 0.0, finite ? sum_cpu / static_cast<f64>(finite) : 0.0);
            // host states
            if (!sanitizer_size())
            {
                const std::vector<f64> host = d.evaluate(states);
                EXPECT_TRUE(host == got64);
            }
            if (HasFailure())
                return;
        }
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaHeuristicSuite, ::testing::ValuesIn(params()), param_name);

// ------------------------------------------------------------------------------------------------ launch configurations

TEST(CudaHeuristic, DoesNotDependOnTheLaunchConfigurationOrTheBatch)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    std::vector<std::string> names;
    for (const SuiteTask& t : suite())
        names.push_back(t.name);
    if (sanitizer_size())
        names = {"depot__p02", "philosophers__p03-phil4", "freecell__p02"};
    struct Config
    {
        cuda::HeuristicVariant variant;
        int warp;
        u32 threads, max_blocks;
        bool global;
    };
    const std::vector<Config> configs = {
        {cuda::HeuristicVariant::Sweep, 0, 128, 0, false},     {cuda::HeuristicVariant::Frontier, 0, 128, 0, false},
        {cuda::HeuristicVariant::Sweep, 1, 32, 0, false},      {cuda::HeuristicVariant::Frontier, 1, 64, 3, false},
        {cuda::HeuristicVariant::Frontier, 0, 512, 2, true},   {cuda::HeuristicVariant::Sweep, 0, 32, 1, true},
        {cuda::HeuristicVariant::Frontier, 1, 256, 0, true},   {cuda::HeuristicVariant::Sweep, 1, 96, 5, false},
    };
    for (const std::string& name : names)
    {
        SCOPED_TRACE(name);
        const TaskPtr task = load(name, true);
        const auto R = grounding(*task);
        if (!R)
            continue;
        const std::vector<State> states = device_ref_walks(*task, sanitizer_size() ? 2 : 4, sanitizer_size() ? 6 : 25, 99);
        const DeviceRows rows = upload(ctx, *task, states);
        for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
        {
            SCOPED_TRACE(kind_name(k));
            cuda::DeviceHeuristicOptions o;
            o.kind = k;
            o.relaxed = R;
            cuda::DeviceHeuristic base(ctx, task, o);
            const std::vector<u32> want = eval_u32(base, rows);
            // twice (the scratch is reused), and split into launches of 1, 5 and 64 states
            EXPECT_EQ(eval_u32(base, rows), want);
            for (u64 chunk : {u64{1}, u64{5}, u64{64}})
            {
                std::vector<u32> parts;
                for (u64 first = 0; first < rows.n; first += chunk)
                {
                    const std::vector<u32> p = eval_u32(base, rows, first, chunk);
                    parts.insert(parts.end(), p.begin(), p.end());
                }
                EXPECT_EQ(parts, want) << "chunks of " << chunk;
            }
            for (const Config& c : configs)
            {
                cuda::DeviceHeuristicOptions v = o;
                v.variant = c.variant;
                v.warp_groups = c.warp;
                v.threads = c.threads;
                v.max_blocks = c.max_blocks;
                v.force_global = c.global;
                cuda::DeviceHeuristic d(ctx, task, v);
                EXPECT_EQ(eval_u32(d, rows), want)
                    << to_string(c.variant) << " warp " << c.warp << " threads " << c.threads << " blocks " << c.max_blocks
                    << (c.global ? " global" : " shared") << " -> resolved: shared " << d.stats().shared << " blocks "
                    << d.stats().blocks;
                EXPECT_EQ(eval_u32(d, rows), want);
            }
            if (HasFailure())
                return;
        }
    }
}

// ------------------------------------------------------------------------------------------------ fallbacks and refusals

TEST(CudaHeuristic, StatesOutsideTheGroundingTakeTheCpuFallback)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    u32 found = 0;
    for (const SuiteTask& t : suite())
    {
        const TaskPtr task = load(t.name, true);
        const auto R = grounding(*task);
        if (!R)
            continue;
        const AtomIndex& A = task->atoms();
        u32 outside = ~u32{0};
        for (u32 slot = 0; slot < A.fluent_slots() && outside == ~u32{0}; ++slot)
            if (R->fluent_props(A.canonical(AtomKind::Fluent, slot)).first == heuristics::RelaxedTask::k_none)
                outside = slot;
        if (outside == ~u32{0})
            continue;
        SCOPED_TRACE(t.name + " slot " + std::to_string(outside));
        const State s0 = task->initial_state();
        std::vector<u64> w(std::max<u32>(s0.size_words(), bits::words_for(outside + 1)), 0);
        std::copy_n(s0.data(), s0.size_words(), w.begin());
        w[outside / 64] |= u64{1} << (outside % 64);
        const std::vector<State> states{s0, State(w.data(), static_cast<u32>(w.size())), s0};
        const DeviceRows rows = upload(ctx, *task, states);
        for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
        {
            cuda::DeviceHeuristicOptions o;
            o.kind = k;
            o.relaxed = R;
            cuda::DeviceHeuristic d(ctx, task, o);
            auto cpu = cpu_heuristic(*task, k, heuristics::Costs::Unit, R);
            const std::vector<u32> got = eval_u32(d, rows);
            EXPECT_EQ(d.stats().fallbacks, 1u);
            EXPECT_EQ(got[1], as_u32(cpu->evaluate(states[1].view())));
            EXPECT_EQ(got[1], as_u32(d.reference(states[1].view())));
            EXPECT_EQ(got[0], got[2]);
        }
        if (++found == 3 || sanitizer_size())
            break;
    }
    if (found == 0)
        GTEST_SKIP() << "no suite task has an atom outside its grounding";
}

TEST(CudaHeuristic, RefusesNumericTasksOtherKindsAndLargeGroundings)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    const auto numeric = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-counters.txt");
    EXPECT_FALSE(cuda::DeviceHeuristic::unsupported(*numeric, {}).empty());
    EXPECT_THROW((void)cuda::DeviceHeuristic(ctx, numeric), std::invalid_argument);
    const TaskPtr task = load("depot__p02", true);
    for (heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::GoalCount, heuristics::Kind::SetAdditive,
                               heuristics::Kind::H2, heuristics::Kind::Perfect})
    {
        cuda::DeviceHeuristicOptions o;
        o.kind = k;
        const std::string why = cuda::DeviceHeuristic::unsupported(*task, o);
        EXPECT_NE(why.find(std::string("'") + heuristics::to_string(k) + "'"), std::string::npos) << why;
        EXPECT_THROW((void)cuda::DeviceHeuristic(ctx, task, o), std::invalid_argument);
    }
    cuda::DeviceHeuristicOptions small;
    small.budget.max_operators = 10;
    EXPECT_TRUE(cuda::DeviceHeuristic::unsupported(*task, small).empty());  // unsupported() does not ground
    try
    {
        (void)cuda::DeviceHeuristic(ctx, task, small);
        ADD_FAILURE() << "a grounding beyond the budget was accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_EQ(std::string(e.what()).rfind("mymyr: ", 0), 0u) << e.what();
    }
    // a grounding of another task
    const TaskPtr twin = load("depot__p02", true);
    cuda::DeviceHeuristicOptions other;
    other.relaxed = grounding(*twin);
    EXPECT_THROW((void)cuda::DeviceHeuristic(ctx, task, other), std::invalid_argument);
    // a row that sets a slot the task has not assigned
    cuda::DeviceHeuristicOptions add;
    add.kind = heuristics::Kind::Add;
    cuda::DeviceHeuristic d(ctx, task, add);
    const u32 slots = task->atoms().fluent_slots();
    std::vector<u64> w(bits::words_for(slots + 1), 0);
    w[slots / 64] |= u64{1} << (slots % 64);
    const std::vector<State> bad{State(w.data(), static_cast<u32>(w.size()))};
    const DeviceRows rows = upload(ctx, *task, bad);
    EXPECT_THROW((void)eval_u32(d, rows), std::invalid_argument);
    // and the heuristic still works afterwards
    const std::vector<State> good{task->initial_state()};
    EXPECT_EQ(eval_u32(d, upload(ctx, *task, good))[0], as_u32(d.reference(good[0].view())));
}

// ------------------------------------------------------------------------------------------------ A* and GBFS

/// Replays `plan` from `start` on the CPU: every action applicable and the end a goal state. Returns the plan's cost
/// (g0 plus the action costs), or -1 when the plan is not valid.
f64 replay(const Task& task, const State& start, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    const heuristics::ActionCosts costs(task);
    f64 g = costs.initial();
    State s = start;
    for (const Action& a : plan)
    {
        const ActionLabel label = a.label();
        if (!succ.is_applicable(s.view(), label))
            return -1;
        g += costs.unit() ? 1.0 : costs.cost(a.schema.v, a.binding.data());
        s = succ.apply(s.view(), label);
    }
    return succ.is_goal(s.view()) ? g : -1;
}

search::BestFirstOptions cpu_search(heuristics::Kind k, u64 max_expanded)
{
    search::BestFirstOptions o;
    o.heuristic.kind = k;
    o.control.budget.max_expanded = max_expanded;
    return o;
}

cuda::DeviceBestFirstOptions device_search(heuristics::Kind k, u32 batch, u64 max_expanded = ~u64{0})
{
    cuda::DeviceBestFirstOptions o;
    o.search.heuristic.kind = k;
    o.search.control.budget.max_expanded = max_expanded;
    o.batch = batch;
    return o;
}

/// Everything the device search reproduces of the CPU search at batch 1.
void expect_same_search(const search::BestFirstResult& d, const search::BestFirstResult& c, bool evaluations, const std::string& what)
{
    SCOPED_TRACE(what);
    EXPECT_EQ(d.status, c.status);
    EXPECT_EQ(d.stats.expanded, c.stats.expanded);
    EXPECT_EQ(d.stats.generated, c.stats.generated);
    EXPECT_EQ(d.stats.states, c.stats.states);
    EXPECT_EQ(d.stats.pruned, c.stats.pruned);
    EXPECT_EQ(d.reopened, c.reopened);
    EXPECT_EQ(d.dead_ends, c.dead_ends);
    if (evaluations)
    {
        EXPECT_EQ(d.evaluations, c.evaluations);
    }
    EXPECT_EQ(d.initial_h, c.initial_h);
    EXPECT_EQ(d.cost, c.cost);
    EXPECT_TRUE(d.plan == c.plan) << "plans differ: " << d.plan.size() << " vs " << c.plan.size() << " steps";
    ASSERT_EQ(d.goal_state.has_value(), c.goal_state.has_value());
    if (c.goal_state)
    {
        EXPECT_TRUE(d.goal_state->view() == c.goal_state->view());
    }
}

void expect_same_device(const cuda::DeviceBestFirstResult& a, const cuda::DeviceBestFirstResult& b, const std::string& what)
{
    SCOPED_TRACE(what);
    expect_same_search(a.result, b.result, true, what);
    EXPECT_EQ(a.device.steps, b.device.steps);
    EXPECT_EQ(a.device.max_batch, b.device.max_batch);
}

const char* status_name(search::SearchStatus s) { return search::to_string(s); }

TEST(CudaBestFirst, BatchOfOneIsTheCpuSearch)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    std::vector<std::string> names = {"depot__p02",        "philosophers__p03-phil4", "blocks__probBLOCKS-8-0", "gripper__prob05",
                                      "rovers__p02",       "miconic__s7-4",           "caldera-split-opt18-adl__p04",
                                      "openstacks-opt08-adl__p03", "parcprinter-opt11-strips__p03", "visitall__visitall_x-6_y-3_r-100"};
    if (sanitizer_size())
        names = {"depot__p02", "philosophers__p03-phil4"};
    const u64 budget = sanitizer_size() ? 12 : full_suite() ? 3000 : 300;
    for (const std::string& name : names)
        for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::Blind})
        {
            const std::string what = name + " " + kind_name(k);
            const TaskPtr task = load(name, true);
            const search::BestFirstResult c = search::astar_eager(*task, cpu_search(k, budget));
            const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, device_search(k, 1, budget));
            expect_same_search(d.result, c, true, "A* " + what);
            if (k == heuristics::Kind::Blind)
                continue;
            const search::BestFirstResult cg = search::gbfs_eager(*task, cpu_search(k, budget));
            const cuda::DeviceBestFirstResult dg = cuda::gbfs(ctx, task, device_search(k, 1, budget));
            expect_same_search(dg.result, cg, false, "GBFS " + what);
            if (HasFailure())
                return;
        }
}

class CudaBestFirstSuite : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(CudaBestFirstSuite, OptimalCostsAndValidPlans)
{
    SKIP_WITHOUT_GPU();
    const auto& [name, frozen] = GetParam();
    const TaskPtr task = load(name, frozen);
    if (!grounding(*task))
        GTEST_SKIP() << "grounding beyond the budget";
    const cuda::ContextPtr ctx = context();
    const u64 cpu_budget = sanitizer_size() ? 200 : full_suite() ? 2000000 : 20000;
    const u32 batch = sanitizer_size() ? 16 : full_suite() ? 10000 : 1000;
    const State s0 = task->initial_state();
    for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Blind})
    {
        search::BestFirstOptions co = cpu_search(k, cpu_budget);
        co.control.budget.max_seconds = 300;
        const search::BestFirstResult c = search::astar_eager(*task, co);
        std::printf("ASTAR %s %s %s cpu %s expanded %llu cost %g", name.c_str(), frozen ? "frozen" : "lazy", kind_name(k),
                    status_name(c.status), static_cast<unsigned long long>(c.stats.expanded), c.cost);
        if (c.status != search::SearchStatus::Solved)
        {
            std::printf("\n");
            continue;
        }
        const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, device_search(k, batch));
        std::printf(" device %s expanded %llu steps %llu cost %g\n", status_name(d.result.status),
                    static_cast<unsigned long long>(d.result.stats.expanded), static_cast<unsigned long long>(d.device.steps),
                    d.result.cost);
        SCOPED_TRACE(kind_name(k));
        ASSERT_EQ(d.result.status, search::SearchStatus::Solved);
        EXPECT_EQ(d.result.cost, c.cost);
        EXPECT_EQ(replay(*task, s0, d.result.plan), c.cost);
    }
    for (heuristics::Kind k : {heuristics::Kind::FF, heuristics::Kind::Add})
    {
        search::BestFirstOptions co = cpu_search(k, cpu_budget);
        co.control.budget.max_seconds = 300;
        const search::BestFirstResult c = search::gbfs_eager(*task, co);
        const cuda::DeviceBestFirstResult d = cuda::gbfs(ctx, task, device_search(k, batch, sanitizer_size() ? 2000 : ~u64{0}));
        std::printf("GBFS %s %s %s cpu %s expanded %llu cost %g device %s expanded %llu steps %llu cost %g\n", name.c_str(),
                    frozen ? "frozen" : "lazy", kind_name(k), status_name(c.status), static_cast<unsigned long long>(c.stats.expanded),
                    c.cost, status_name(d.result.status), static_cast<unsigned long long>(d.result.stats.expanded),
                    static_cast<unsigned long long>(d.device.steps), d.result.cost);
        SCOPED_TRACE(std::string("gbfs ") + kind_name(k));
        if (c.status == search::SearchStatus::Solved && !sanitizer_size())
        {
            EXPECT_EQ(d.result.status, search::SearchStatus::Solved);
        }
        if (d.result.status == search::SearchStatus::Solved)
        {
            EXPECT_EQ(replay(*task, s0, d.result.plan), d.result.cost);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Suite, CudaBestFirstSuite, ::testing::ValuesIn(params()), param_name);

TEST(CudaBestFirst, DoesNotDependOnTheChunkingOrTheRun)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    std::vector<std::string> names = {"depot__p02", "caldera-split-opt18-adl__p04", "blocks__probBLOCKS-8-0", "snake-opt18-strips__p05"};
    if (sanitizer_size())
        names = {"depot__p02"};
    u64 graph_steps = 0;
    for (const std::string& name : names)
        for (bool greedy : {false, true})
        {
            const std::string what = name + (greedy ? " gbfs" : " astar");
            const TaskPtr task = load(name, false);
            cuda::DeviceBestFirstOptions o = device_search(greedy ? heuristics::Kind::FF : heuristics::Kind::Max, sanitizer_size() ? 8 : 200,
                                                           sanitizer_size() ? 40 : 20000);
            auto run = [&](const cuda::DeviceBestFirstOptions& x) { return greedy ? cuda::gbfs(ctx, task, x) : cuda::astar(ctx, task, x); };
            const cuda::DeviceBestFirstResult ref = run(o);
            expect_same_device(run(o), ref, what + " second run");
            for (u64 chunk : {u64{1}, u64{7}, u64{64}})
            {
                cuda::DeviceBestFirstOptions v = o;
                v.chunk_states = chunk;
                expect_same_device(run(v), ref, what + " chunk " + std::to_string(chunk));
            }
            cuda::DeviceBestFirstOptions v = o;
            v.heuristic.variant = cuda::HeuristicVariant::Frontier;
            v.heuristic.force_global = true;
            expect_same_device(run(v), ref, what + " heuristic launch");
            // Steps in device loops of 1 and 3 steps, and every step launched from the host
            graph_steps += ref.device.graph_steps;
            for (u32 steps : {1u, 3u})
            {
                v = o;
                v.loop_steps = steps;
                expect_same_device(run(v), ref, what + " loops of " + std::to_string(steps) + " steps");
            }
            v = o;
            v.graphs = false;
            const cuda::DeviceBestFirstResult host = run(v);
            expect_same_device(host, ref, what + " host-driven steps");
            EXPECT_EQ(host.device.graph_steps, 0u) << what;
            if (HasFailure())
                return;
        }
    if (!sanitizer_size() && cuda::GraphExec::enabled())
    {
        EXPECT_GT(graph_steps, 0u) << "no step ran in a device loop or graph";
    }
}

TEST(CudaBestFirst, BudgetsStartsAndRefusals)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    const TaskPtr task = load("blocks__probBLOCKS-8-0", true);
    // max_expanded: exact
    {
        const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, device_search(heuristics::Kind::Blind, 100, 250));
        EXPECT_EQ(d.result.status, search::SearchStatus::OutOfStates);
        EXPECT_EQ(d.result.stats.expanded, 250u);
    }
    // max_states, max_seconds, cancellation
    {
        cuda::DeviceBestFirstOptions o = device_search(heuristics::Kind::Blind, 100);
        o.search.control.budget.max_states = 1000;
        const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, o);
        EXPECT_EQ(d.result.status, search::SearchStatus::OutOfStates);
        EXPECT_GT(d.result.stats.states, 1000u);
        o = device_search(heuristics::Kind::Blind, 100);
        o.search.control.budget.max_seconds = 0;
        EXPECT_EQ(cuda::astar(ctx, task, o).result.status, search::SearchStatus::OutOfTime);
        o = device_search(heuristics::Kind::Blind, 100);
        o.search.control.cancel.request();
        EXPECT_EQ(cuda::gbfs(ctx, task, o).result.status, search::SearchStatus::Cancelled);
    }
    // max_depth: pruned at the depth, as the CPU
    {
        search::BestFirstOptions co = cpu_search(heuristics::Kind::Max, 400);
        co.control.budget.max_depth = 3;
        cuda::DeviceBestFirstOptions o = device_search(heuristics::Kind::Max, 1, 400);
        o.search.control.budget.max_depth = 3;
        expect_same_search(cuda::astar(ctx, task, o).result, search::astar_eager(*task, co), true, "max_depth");
    }
    // a start state: the CPU's optimal cost from it
    if (!sanitizer_size())
    {
        const std::vector<State> walk = device_ref_walks(*task, 1, 12, 5);
        search::BestFirstOptions co = cpu_search(heuristics::Kind::Max, ~u64{0});
        co.start = walk.back();
        cuda::DeviceBestFirstOptions o = device_search(heuristics::Kind::Max, 500);
        o.search.start = walk.back();
        const search::BestFirstResult c = search::astar_eager(*task, co);
        const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, o);
        ASSERT_EQ(d.result.status, search::SearchStatus::Solved);
        EXPECT_EQ(d.result.cost, c.cost);
        EXPECT_EQ(replay(*task, walk.back(), d.result.plan), c.cost);
        // the goal as the start: solved without a step
        o.search.start = *d.result.goal_state;
        const cuda::DeviceBestFirstResult z = cuda::astar(ctx, task, o);
        EXPECT_EQ(z.result.status, search::SearchStatus::Solved);
        EXPECT_TRUE(z.result.plan.empty());
    }
    // refusals
    const auto numeric = Task::from_text_file(std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks/cs-counters.txt");
    EXPECT_FALSE(cuda::best_first_unsupported(*numeric, {}).empty());
    EXPECT_THROW((void)cuda::astar(ctx, numeric), std::invalid_argument);
    EXPECT_THROW((void)cuda::gbfs(ctx, numeric), std::invalid_argument);
    cuda::DeviceBestFirstOptions o = device_search(heuristics::Kind::GoalCount, 10);
    EXPECT_THROW((void)cuda::astar(ctx, task, o), std::invalid_argument);
    o = device_search(heuristics::Kind::Max, 0);
    EXPECT_THROW((void)cuda::astar(ctx, task, o), std::invalid_argument);
    o = device_search(heuristics::Kind::Max, 10);
    o.search.control.goal.kind = search::GoalSpec::Kind::Custom;
    o.search.control.goal.test = [](StateView) { return false; };
    EXPECT_THROW((void)cuda::gbfs(ctx, task, o), std::invalid_argument);
    search::SearchObserver observer;
    o = device_search(heuristics::Kind::Max, 10);
    o.search.control.observer = &observer;
    EXPECT_THROW((void)cuda::astar(ctx, task, o), std::invalid_argument);
    o = device_search(heuristics::Kind::Max, 10);
    o.search.heuristic.budget.max_operators = 10;
    EXPECT_THROW((void)cuda::astar(ctx, task, o), std::invalid_argument);
}
#if defined(MYMYR_DEVICE_HEURISTICS_FRONTEND)
// ------------------------------------------------------------------------------------------------ PDDL tasks

/// The A* rows compared with the fork (tests/cpp/search/test_best_first_golden.cpp):
/// the heuristics on sampled states with unit and real costs, device A* at the gate's batch ("DEVICE_VS_CPU" lines).
TEST(DeviceHeuristicsPddl, RealActionCosts)
{
    SKIP_WITHOUT_GPU();
    struct Row
    {
        const char* dir;
        const char* problem;
        heuristics::Kind kind;
        double cost;
    };
    const Row rows[] = {
        {"strips/blocks", "probBLOCKS-8-0", heuristics::Kind::Blind, 18},
        {"strips/driverlog", "p03", heuristics::Kind::Blind, 12},
        {"strips/miconic", "s7-4", heuristics::Kind::Blind, 25},
        {"strips/zenotravel", "p05", heuristics::Kind::Blind, 11},
        {"strips/blocks", "probBLOCKS-8-0", heuristics::Kind::Max, 18},
        {"strips/freecell", "p02", heuristics::Kind::Max, 14},
        {"strips/gripper", "prob05", heuristics::Kind::Max, 35},
        {"strips/miconic", "s7-4", heuristics::Kind::Max, 25},
        {"data/sokoban", "p68", heuristics::Kind::Max, 34},
        {"strips/transport-opt08-strips", "p23", heuristics::Kind::Max, 630},
        {"strips/zenotravel", "p05", heuristics::Kind::Max, 11},
    };
    const cuda::ContextPtr ctx = context();
    const u32 batch = full_suite() ? 10000 : 1000;
    std::vector<std::string> checked;  // tasks whose heuristics were compared
    u64 run = 0;
    for (const Row& row : rows)
    {
        const std::string name = std::string(row.dir) + "/" + row.problem;
        if (sanitizer_size() && (name != "strips/zenotravel/p05" || row.kind != heuristics::Kind::Max))
            continue;  // 12,135 expansions
        const std::filesystem::path base = std::string(row.dir) == "data/sokoban" ? work_dir() / "mimir" / "data" / "sokoban"
                                                                                  : work_dir() / "mimir-cs" / "Benchmark" / row.dir;
        const std::filesystem::path dom = base / "domain.pddl", prob = base / (std::string(row.problem) + ".pddl");
        if (!std::filesystem::exists(dom) || !std::filesystem::exists(prob))
        {
            std::printf("DEVICE_VS_CPU %s missing\n", name.c_str());
            continue;
        }
        SCOPED_TRACE(name + " " + kind_name(row.kind));
        const TaskPtr task = Task::create(*frontend::load_task(dom, prob));
        const auto R = grounding(*task);
        ASSERT_TRUE(R);
        const bool real = !heuristics::ActionCosts(*task).unit() && R->real_costs_available();
        std::vector<heuristics::Costs> costs{heuristics::Costs::Unit};
        if (real)
            costs.push_back(heuristics::Costs::Real);
        if (std::find(checked.begin(), checked.end(), name) == checked.end())
        {
            checked.push_back(name);
            const std::vector<State> states = sample_states(*task, 0x4b + checked.size());
            for (heuristics::Costs c : costs)
                for (heuristics::Kind k : {heuristics::Kind::Max, heuristics::Kind::Add, heuristics::Kind::FF})
                {
                    SCOPED_TRACE(std::string(kind_name(k)) + (c == heuristics::Costs::Real ? " real" : " unit"));
                    cuda::DeviceHeuristicOptions o;
                    o.kind = k;
                    o.costs = c;
                    o.relaxed = R;
                    cuda::DeviceHeuristic d(ctx, task, o);
                    const DeviceRows r = upload(ctx, *task, states);
                    const std::vector<u32> got = eval_u32(d, r);
                    auto cpu = cpu_heuristic(*task, k, c, R);
                    u64 failures = 0, agree = 0;
                    for (usize i = 0; i < states.size() && failures < 5; ++i)
                    {
                        const f64 want = k == heuristics::Kind::FF ? d.reference(states[i].view()) : cpu->evaluate(states[i].view());
                        if (got[i] != as_u32(want))
                        {
                            ADD_FAILURE() << "state " << i << ": device " << got[i] << ", expected " << want;
                            ++failures;
                        }
                        if (k == heuristics::Kind::FF)
                            agree += cpu->evaluate(states[i].view()) == want ? 1 : 0;
                    }
                    if (k == heuristics::Kind::FF)
                        std::printf("FF_AGREE %s pddl %s equal %llu/%zu\n", name.c_str(),
                                    c == heuristics::Costs::Real ? "real" : "unit", static_cast<unsigned long long>(agree),
                                    states.size());
                }
        }
        // A*: unit heuristic costs, and real ones where the task has them (the plan cost is the task's either way)
        for (heuristics::Costs c : costs)
        {
            if (row.kind == heuristics::Kind::Blind && c == heuristics::Costs::Real)
                continue;
            search::BestFirstOptions co = cpu_search(row.kind, ~u64{0});
            co.heuristic.costs = c;
            const search::BestFirstResult cr = search::astar_eager(*task, co);
            ASSERT_EQ(cr.status, search::SearchStatus::Solved) << cr.message;
            EXPECT_EQ(cr.cost, row.cost);
            cuda::DeviceBestFirstOptions o = device_search(row.kind, batch);
            o.search.heuristic.costs = c;
            const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, o);
            std::printf("DEVICE_VS_CPU %s %s %s cpu expanded %llu cost %g device B=%u expanded %llu steps %llu cost %g overhead %.3f\n",
                        name.c_str(), kind_name(row.kind), c == heuristics::Costs::Real ? "real" : "unit",
                        static_cast<unsigned long long>(cr.stats.expanded), cr.cost, batch,
                        static_cast<unsigned long long>(d.result.stats.expanded), static_cast<unsigned long long>(d.device.steps),
                        d.result.cost, static_cast<double>(d.result.stats.expanded) / static_cast<double>(std::max<u64>(1, cr.stats.expanded)));
            ASSERT_EQ(d.result.status, search::SearchStatus::Solved) << d.result.message;
            EXPECT_EQ(d.result.cost, cr.cost);
            EXPECT_EQ(replay(*task, task->initial_state(), d.result.plan), cr.cost);
            ++run;
        }
    }
    if (run == 0)
        GTEST_SKIP() << "no PDDL task found under " << work_dir();
}

/// Costs on the device from cost programs (five cost parameters, a static function of more than 2^24 keys):
/// the CPU's search at batch 1, its optimal cost at larger batches; non-integral costs are refused.
TEST(DeviceHeuristicsPddl, CostProgramsEqualTheCpu)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    const auto make = [](const std::string& hop, u32 n)
    {
        return Task::create(*frontend::Domain::from_string(test::wide_domain(hop), "d.pddl")->instantiate_string(test::wide_problem(n), "p.pddl"));
    };
    for (const u32 n : {30u, 31u})
    {
        const TaskPtr task = make(test::k_wide_integral, n);
        ASSERT_EQ(heuristics::ActionCosts(*task).cost_parameters(0).size(), 5u);
        ASSERT_TRUE(cuda::best_first_unsupported(*task, {}).empty()) << cuda::best_first_unsupported(*task, {});
        for (heuristics::Kind k : {heuristics::Kind::Blind, heuristics::Kind::Max})
        {
            const std::string what = "wide " + std::to_string(n) + " " + kind_name(k);
            const search::BestFirstResult c = search::astar_eager(*task, cpu_search(k, ~u64{0}));
            ASSERT_EQ(c.status, search::SearchStatus::Solved) << what;
            expect_same_search(cuda::astar(ctx, task, device_search(k, 1)).result, c, true, "A* " + what);
            const cuda::DeviceBestFirstResult d = cuda::astar(ctx, task, device_search(k, 64));
            ASSERT_EQ(d.result.status, search::SearchStatus::Solved) << what;
            EXPECT_EQ(d.result.cost, c.cost) << what;
            EXPECT_EQ(replay(*task, task->initial_state(), d.result.plan), c.cost) << what;
            if (k == heuristics::Kind::Max)
                expect_same_search(cuda::gbfs(ctx, task, device_search(k, 1)).result, search::gbfs_eager(*task, cpu_search(k, ~u64{0})),
                                   false, "GBFS " + what);
        }
    }
    // real costs: refused (u32 g values), as before
    const TaskPtr real = make(test::k_wide_real, 30);
    EXPECT_FALSE(cuda::best_first_unsupported(*real, {}).empty());
    EXPECT_THROW((void)cuda::astar(ctx, real, device_search(heuristics::Kind::Blind, 8)), std::invalid_argument);
}
#endif
}  // namespace
