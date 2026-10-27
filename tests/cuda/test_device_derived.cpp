// The derived words of the device expand (GPU 0: CUDA_VISIBLE_DEVICES=0; the tests skip without a device) against
// rl::expand on the host, byte for byte: a single instance, a table whose instances expand apart (axioms), a table of
// the multi-instance kernels (no axioms: zero words), a suite of several domains (runs and gathered rows), the
// capacity clip, and the padded view.

#include "../cpp/encoding/support.hpp"
#include "env_harness.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/suite_expand.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/rl/task_suite.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace mymyr;

namespace
{
struct Batch
{
    std::vector<u64> rows;  // [N, suite words]
    std::vector<i32> ids;
};

struct Flat
{
    std::vector<u64> succ, derived;
    std::vector<i32> parent, offsets;
    std::vector<u8> goal;
    u64 total = 0, generated = 0;
    u32 words_needed = 0;
};

/// Seeded random walks over a suite, `steps` states per instance, rows at the suite's width.
Batch walks(const rl::TaskSuite& suite, u32 steps, u64 seed)
{
    Batch b;
    const u32 W = suite.words();
    std::mt19937_64 rng(seed);
    for (u32 g = 0; g < suite.size(); ++g)
    {
        std::vector<u64> cur(W, 0);
        suite.initial_row(g, cur.data(), W, 0);
        for (u32 k = 0; k < steps; ++k)
        {
            b.rows.insert(b.rows.end(), cur.begin(), cur.end());
            b.ids.push_back(static_cast<i32>(g));
            std::vector<u64> succ(256 * W, 0);
            i32 id = static_cast<i32>(g);
            rl::Expansion x;
            x.capacity = 256;
            x.words = W;
            x.succ = succ.data();
            rl::expand(suite, {cur.data(), 1, W, 0, 0}, &id, x);
            if (x.total == 0 || x.total > 256)
                break;
            const u64 pick = rng() % x.total;
            std::copy_n(succ.begin() + static_cast<std::ptrdiff_t>(pick * W), W, cur.begin());
        }
    }
    // interleave the instances' rows so that the batch is not one run per instance
    const u64 n = b.ids.size();
    std::vector<u64> order(n);
    for (u64 i = 0; i < n; ++i)
        order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    Batch s;
    for (u64 i : order)
    {
        s.rows.insert(s.rows.end(), b.rows.begin() + static_cast<std::ptrdiff_t>(i * W), b.rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * W));
        s.ids.push_back(b.ids[i]);
    }
    return s;
}

Flat host_run(const rl::TaskSuite& suite, const Batch& b, u64 cap, u32 dw)
{
    const u32 W = suite.words(), N = static_cast<u32>(b.ids.size());
    Flat f;
    f.succ.assign(std::max<u64>(cap, 1) * W, 0x5A5A5A5A5A5A5A5Aull);
    f.derived.assign(std::max<u64>(cap, 1) * dw, 0x5A5A5A5A5A5A5A5Aull);
    f.parent.assign(std::max<u64>(cap, 1), 0x5A5A5A5A);
    f.goal.assign(std::max<u64>(cap, 1), 0x5A);
    f.offsets.assign(N + 1, 0);
    rl::Expansion x;
    x.capacity = cap;
    x.words = W;
    x.derived_words = dw;
    x.succ = f.succ.data();
    x.derived = dw ? f.derived.data() : nullptr;
    x.parent = f.parent.data();
    x.goal = f.goal.data();
    x.offsets = f.offsets.data();
    rl::expand(suite, {b.rows.data(), N, W, 0, 0}, b.ids.data(), x);
    f.total = x.total;
    f.generated = x.generated;
    f.words_needed = x.words_needed;
    return f;
}

Flat device_run(const cuda::ContextPtr& ctx, const rl::TaskSuitePtr& suite, const Batch& b, u64 cap, u32 dw,
                cuda::BucketLaunch launch = cuda::BucketLaunch::PerBucket)
{
    cuda::SuiteExpander ex(ctx, suite);
    ex.set_launch(launch);
    const cudaStream_t s = ex.stream();
    const u32 W = suite->words(), N = static_cast<u32>(b.ids.size());
    const u64 c = std::max<u64>(cap, 1);
    cuda::DeviceBuffer rows = test::to_device(ctx, b.rows, s), ids = test::to_device(ctx, b.ids, s);
    cuda::DeviceBuffer succ = test::to_device(ctx, std::vector<u64>(c * W, 0x5A5A5A5A5A5A5A5Aull), s);
    cuda::DeviceBuffer derived = test::to_device(ctx, std::vector<u64>(c * std::max<u32>(dw, 1), 0x5A5A5A5A5A5A5A5Aull), s);
    cuda::DeviceBuffer parent = test::to_device(ctx, std::vector<i32>(c, 0x5A5A5A5A), s);
    cuda::DeviceBuffer goal = test::to_device(ctx, std::vector<u8>(c, 0x5A), s);
    cuda::DeviceBuffer offsets(ctx, (N + 1) * sizeof(i32), s);
    rl::Expansion x;
    x.capacity = cap;
    x.words = W;
    x.derived_words = dw;
    x.succ = static_cast<u64*>(succ.data());
    x.derived = dw ? static_cast<u64*>(derived.data()) : nullptr;
    x.parent = static_cast<i32*>(parent.data());
    x.goal = static_cast<u8*>(goal.data());
    x.offsets = static_cast<i32*>(offsets.data());
    ex.expand({static_cast<const u64*>(rows.data()), N, W, 0, 0}, static_cast<const i32*>(ids.data()), x);
    Flat f;
    f.total = x.total;
    f.generated = x.generated;
    f.words_needed = x.words_needed;
    f.succ = test::to_host<u64>(succ.data(), c * W, s);
    f.derived = test::to_host<u64>(derived.data(), c * std::max<u32>(dw, 1), s);
    f.parent = test::to_host<i32>(parent.data(), c, s);
    f.goal = test::to_host<u8>(goal.data(), c, s);
    f.offsets = test::to_host<i32>(offsets.data(), N + 1, s);
    return f;
}

rl::TaskSuitePtr suite_of(std::initializer_list<std::pair<const char*, const char*>> problems)
{
    std::vector<TaskPtr> tasks;
    for (const auto& [d, p] : problems)
        tasks.push_back(test::pddl_task(d, p));
    return rl::TaskSuite::group(std::move(tasks));
}

void expect_equal(const rl::TaskSuitePtr& suite, u64 cap_divisor, u32 extra_words = 0)
{
    const auto ctx = test::context();
    const Batch b = walks(*suite, 24, 7);
    const u32 dw = suite->derived_words() + extra_words;
    const Flat full = host_run(*suite, b, 1 << 16, dw);
    ASSERT_GT(full.total, 0u);
    const u64 cap = cap_divisor ? std::max<u64>(1, full.total / cap_divisor) : full.total;
    const Flat h = host_run(*suite, b, cap, dw);
    for (auto launch : {cuda::BucketLaunch::PerBucket, cuda::BucketLaunch::Widest})
    {
        const Flat d = device_run(ctx, suite, b, cap, dw, launch);
        ASSERT_EQ(d.total, h.total);
        EXPECT_EQ(d.generated, h.generated);
        EXPECT_EQ(d.words_needed, h.words_needed);
        EXPECT_EQ(d.offsets, h.offsets);
        EXPECT_EQ(d.succ, h.succ);
        EXPECT_EQ(d.parent, h.parent);
        EXPECT_EQ(d.goal, h.goal);
        if (dw)
        {
            EXPECT_EQ(d.derived, h.derived);
        }
    }
}
}  // namespace

TEST(DeviceExpandDerived, SingleInstanceEqualsHost)
{
    SKIP_WITHOUT_GPU();
    for (auto p : {std::pair{"philosophers", "p03-phil4"}, std::pair{"features", "p01"}})
    {
        SCOPED_TRACE(p.second);
        const auto suite = suite_of({p});
        ASSERT_GT(suite->derived_words(), 0u);
        expect_equal(suite, 0);
        expect_equal(suite, 3);       // the capacity clips the rows
        expect_equal(suite, 0, 2);    // wider derived rows are zero padded
    }
}

TEST(DeviceExpandDerived, TableOfInstancesWithAxiomsEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const auto suite = suite_of({{"features", "p01"}, {"features", "p02"}, {"features", "p03"}, {"features", "p04"}});
    ASSERT_EQ(suite->num_domains(), 1u);
    {
        cuda::SuiteExpander ex(test::context(), suite);
        EXPECT_EQ(ex.domain(0).mode(), cuda::DeviceExpander::Mode::PerInstance);
    }
    expect_equal(suite, 0);
    expect_equal(suite, 4);
    expect_equal(suite, 0, 2);  // wider derived rows: each part scatters zero padded
}

TEST(DeviceExpandDerived, TableWithoutAxiomsHasZeroWords)
{
    SKIP_WITHOUT_GPU();
    const auto suite = suite_of({{"blocks", "probBLOCKS-8-0"}, {"blocks", "smedium"}});
    EXPECT_EQ(suite->derived_words(), 0u);
    expect_equal(suite, 0);
    expect_equal(suite, 0, 3);  // a request wider than any instance needs: zeros
}

TEST(DeviceExpandDerived, SuiteOfSeveralDomainsEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const auto suite = suite_of({{"philosophers", "p03-phil4"}, {"blocks", "probBLOCKS-8-0"}, {"features", "p01"},
                                 {"features", "p02"}, {"blocks", "smedium"}});
    ASSERT_GT(suite->num_domains(), 1u);
    expect_equal(suite, 0);
    expect_equal(suite, 5);
    expect_equal(suite, 0, 2);  // wider derived rows than any instance needs
}

TEST(DeviceExpandDerived, PaddedViewEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const auto ctx = test::context();
    const auto suite = suite_of({{"features", "p01"}});
    const Batch b = walks(*suite, 16, 3);
    const u32 W = suite->words(), N = static_cast<u32>(b.ids.size()), dw = suite->derived_words(), K = 16;
    const Flat f = host_run(*suite, b, 1 << 14, dw);
    // host padded
    std::vector<u64> hp(u64{N} * K * dw, 0x77);
    bool host_overflow = false;
    {
        rl::Expansion x;
        x.capacity = 1 << 14;
        x.words = W;
        x.derived_words = dw;
        x.succ = const_cast<u64*>(f.succ.data());
        x.derived = const_cast<u64*>(f.derived.data());
        x.offsets = const_cast<i32*>(f.offsets.data());
        x.total = f.total;
        x.generated = f.generated;
        x.words_needed = f.words_needed;
        rl::PaddedExpansion p;
        p.k = K;
        p.derived_words = dw;
        p.derived = hp.data();
        rl::pad(x, N, p);
        host_overflow = p.overflow;
    }
    cuda::DeviceExpander ex(ctx, suite->table(0));
    const cudaStream_t s = ex.stream();
    cuda::DeviceBuffer succ = test::to_device(ctx, f.succ, s), derived = test::to_device(ctx, f.derived, s),
                       offsets = test::to_device(ctx, f.offsets, s), out = test::to_device(ctx, std::vector<u64>(hp.size(), 0x77), s);
    rl::Expansion x;
    x.capacity = 1 << 14;
    x.words = W;
    x.derived_words = dw;
    x.succ = static_cast<u64*>(succ.data());
    x.derived = static_cast<u64*>(derived.data());
    x.offsets = static_cast<i32*>(offsets.data());
    x.total = f.total;
    x.generated = f.generated;
    x.words_needed = f.words_needed;
    rl::PaddedExpansion p;
    p.k = K;
    p.derived_words = dw;
    p.derived = static_cast<u64*>(out.data());
    ex.pad(x, N, p);
    EXPECT_EQ(p.overflow, host_overflow);
    EXPECT_EQ(test::to_host<u64>(out.data(), hp.size(), s), hp);
}

// ------------------------------------------------------------------------------------------------ wide tasks

namespace
{
/// A chain of n lights (with a few cross links) whose derived predicates bright and glow follow the lit lights: rows
/// past 64 words and object bitsets past 8 words for the larger n, instances past the device kernels' 2048 objects.
/// Goal: light 1 is bright (a goal over derived atoms).
TaskPtr lit_lights(u32 n, u32 colors)
{
    std::string d = "(define (domain lights) (:requirements :strips :typing :derived-predicates) (:types light) "
                    "(:predicates (on ?x - light) (off ?x - light) (linked ?x ?y - light) (bright ?x - light) (glow ?x - light)";
    for (u32 k = 0; k < colors; ++k)
        d += " (c" + std::to_string(k) + " ?x - light)";
    d += ") (:derived (bright ?x - light) (on ?x)) (:derived (glow ?x - light) (bright ?x))"
         " (:action spread :parameters (?x ?y - light) :precondition (and (on ?x) (linked ?x ?y) (off ?y)) :effect "
         "(and (on ?y) (not (off ?y)))) (:action dim :parameters (?x - light) :precondition (on ?x) :effect (and (off ?x) "
         "(not (on ?x))))";
    for (u32 k = 0; k < colors; ++k)
        d += " (:action paint-" + std::to_string(k) + " :parameters (?x - light) :precondition (on ?x) :effect (c" +
             std::to_string(k) + " ?x))";
    d += ")";
    std::string p = "(define (problem p) (:domain lights) (:objects";
    for (u32 i = 0; i < n; ++i)
        p += " l" + std::to_string(i);
    p += " - light) (:init (on l0)";
    for (u32 i = 1; i < n; ++i)
        p += " (off l" + std::to_string(i) + ")";
    for (u32 i = 0; i + 1 < n; ++i)
        p += " (linked l" + std::to_string(i) + " l" + std::to_string(i + 1) + ")";
    for (u32 i = 0; i < n; i += 5)
        p += " (linked l" + std::to_string(i) + " l" + std::to_string((i * 7 + 3) % n) + ")";
    p += ") (:goal (and (bright l1))))";
    TaskOptions o;
    o.atoms = TaskOptions::Atoms::Reachable;
    return Task::create(*frontend::Domain::from_string(d, "lights.pddl")->instantiate_string(p, "p.pddl"), o);
}

rl::TaskSuitePtr lights_suite(std::initializer_list<std::pair<u32, u32>> sizes)
{
    std::vector<TaskPtr> tasks;
    for (const auto& [n, colors] : sizes)
        tasks.push_back(lit_lights(n, colors));
    return rl::TaskSuite::group(std::move(tasks));
}
}  // namespace

TEST(DeviceExpandDerived, RowsPastSixtyFourWordsEqualHost)
{
    SKIP_WITHOUT_GPU();
    for (const auto& sizes : {std::pair{300u, 14u}, std::pair{1100u, 4u}})
    {
        SCOPED_TRACE("lights " + std::to_string(sizes.first));
        const auto suite = lights_suite({sizes});
        ASSERT_GT(suite->words(), 64u);
        ASSERT_GT(suite->derived_words(), 0u);
        expect_equal(suite, 0);
        expect_equal(suite, 3);
        expect_equal(suite, 0, 2);
    }
}

TEST(DeviceExpandDerived, InstancesOnTheHostInADeviceTableEqualHost)
{
    SKIP_WITHOUT_GPU();
    {
        // one instance beyond the device kernels' objects: its rows are expanded on the host, per instance
        const auto suite = lights_suite({{40, 4}, {700, 4}, {2200, 4}, {300, 4}});
        ASSERT_EQ(suite->num_domains(), 1u);
        {
            cuda::SuiteExpander ex(test::context(), suite);
            EXPECT_EQ(ex.domain(0).mode(), cuda::DeviceExpander::Mode::PerInstance);
        }
        expect_equal(suite, 0);
        expect_equal(suite, 4);
        expect_equal(suite, 0, 2);
    }
    {
        const auto suite = lights_suite({{2200, 0}});
        {
            cuda::SuiteExpander ex(test::context(), suite);
            EXPECT_EQ(ex.domain(0).mode(), cuda::DeviceExpander::Mode::Host);
        }
        expect_equal(suite, 0);
        expect_equal(suite, 3);
        expect_equal(suite, 0, 2);
    }
}
