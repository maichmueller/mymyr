// LM-cut and landmark count on small suite tasks (tasks from PDDL with conditional effects, axioms and zero or
// fractional costs are in mymyr_landmark_heuristics_tests):
//   - LM-cut equals a from-scratch evaluation of its definition (lm_cut_reference.hpp) and h_max <= LM-cut <= h* on
//     states of the whole state space, with unit and real costs;
//   - one heuristic per thread over one shared task and grounding gives the single-threaded values (TSan build).

#include "../support/suite.hpp"
#include "h2_reference.hpp"
#include "lm_cut_reference.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/perfect.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;
namespace H = mymyr::heuristics;

namespace
{
// Exhaustive state spaces of 40k-320k states (suite.hpp): philosophers and openstacks have derived predicates,
// openstacks and miconic-simpleadl conditional effects, openstacks and pegsol action costs.
const char* const k_tasks[] = {"depot__p02", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "pegsol-08-strips__p22",
                               "miconic-simpleadl__s10-2"};

std::shared_ptr<const Task> load(const std::string& name) { return Task::from_text_file(task_path(name)); }

datasets::StateSpacePtr space_of(const std::shared_ptr<const Task>& task)
{
    datasets::StateSpaceOptions o;
    o.remove_if_unsolvable = false;
    o.labels = false;
    auto r = datasets::generate_state_space(task, o);
    EXPECT_EQ(r.status, datasets::StateSpaceStatus::Ok);
    return r.space;
}

/// About `n` state ids of the space, evenly spread (the initial state first).
std::vector<u32> sample(const datasets::StateSpace& sp, u32 n)
{
    std::vector<u32> ids;
    const u32 step = std::max<u32>(1, sp.num_states() / n);
    for (u32 i = 0; i < sp.num_states(); i += step)
        ids.push_back(i);
    return ids;
}

std::unique_ptr<H::Heuristic> make(const Task& task, H::Kind k, H::Costs costs, std::shared_ptr<const H::RelaxedTask> relaxed)
{
    H::Options o;
    o.kind = k;
    o.costs = costs;
    o.relaxed = std::move(relaxed);
    return H::make_heuristic(task, o);
}
}  // namespace

TEST(LandmarkHeuristics, KindNamesAndRefusals)
{
    EXPECT_EQ(H::parse_kind("lmcut"), H::Kind::LmCut);
    EXPECT_EQ(H::parse_kind("lm_cut"), H::Kind::LmCut);
    EXPECT_EQ(H::parse_kind("lmcount"), H::Kind::LmCount);
    EXPECT_EQ(H::parse_kind("lm_count"), H::Kind::LmCount);
    EXPECT_EQ(H::parse_kind("landmark_count"), H::Kind::LmCount);
    EXPECT_STREQ(H::to_string(H::Kind::LmCut), "lmcut");
    EXPECT_STREQ(H::to_string(H::Kind::LmCount), "lmcount");

    const auto task = load("depot__p02");
    for (const auto k : {H::Kind::LmCut})  // grounded only
    {
        H::Options o;
        o.kind = k;
        o.evaluation = H::Evaluation::Lifted;
        EXPECT_THROW((void)H::make_heuristic(*task, o), std::invalid_argument);
        o.evaluation = H::Evaluation::Auto;
        o.budget.max_operators = 1;
        EXPECT_THROW((void)H::make_heuristic(*task, o), std::runtime_error);
        o.budget = {};
        const auto h = H::make_heuristic(*task, o);
        EXPECT_EQ(h->kind(), k);
        EXPECT_NE(h->relaxed(), nullptr);
    }
}

TEST(LandmarkHeuristics, LmCutIsItsDefinitionAndBetweenHmaxAndHstar)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto sp = space_of(task);
        ASSERT_TRUE(sp);
        const auto relaxed = H::ground(*task);
        ASSERT_TRUE(relaxed);
        for (const H::Costs costs : {H::Costs::Unit, H::Costs::Real})
        {
            const bool real = costs == H::Costs::Real;
            if (real && !relaxed->real_costs_available())
                continue;
            auto lmcut = make(*task, H::Kind::LmCut, costs, relaxed);
            auto hmax = make(*task, H::Kind::Max, costs, relaxed);
            auto hstar = H::perfect(sp, costs);
            ReferenceLmCut ref(*relaxed, real);
            const std::vector<u32> goal(relaxed->goal().begin(), relaxed->goal().end());
            u32 checked = 0, dead = 0, above_hmax = 0, referenced = 0, exact = 0;
            double ratio = 0;
            u32 ratios = 0;
            for (u32 id : sample(*sp, 400))
            {
                const StateView s = sp->state(id);
                const double vc = lmcut->evaluate(s), vm = hmax->evaluate(s), vs = hstar->evaluate(s);
                EXPECT_LE(vm, vc) << id;
                EXPECT_LE(vc, vs + 1e-9) << id;
                dead += std::isinf(vc);
                above_hmax += vc > vm;
                exact += vc == vs;
                if (std::isfinite(vs) && vs > 0)
                {
                    ratio += vc / vs;
                    ++ratios;
                }
                if (checked++ % 4 == 0)  // the reference is slow
                {
                    const u64 r = relaxed->goal_unreachable() ? ReferenceLmCut::k_inf : ref.evaluate(true_props(*task, *relaxed, s), goal);
                    const double vr = r == ReferenceLmCut::k_inf ? H::k_dead_end : static_cast<double>(r) / (real ? relaxed->cost_scale() : 1.0);
                    EXPECT_EQ(vc, vr) << id;
                    ++referenced;
                }
            }
            std::printf("LMCUT %-28s %s states %u (%u against the reference) | lmcut > h_max %u | = h* %u | mean lmcut/h* %.3f | "
                        "dead ends %u\n",
                        name, real ? "real" : "unit", checked, referenced, above_hmax, exact, ratios ? ratio / ratios : 1.0, dead);
        }
    }
}

TEST(LandmarkHeuristics, OneHeuristicPerThreadOverASharedTask)
{
    const auto task = load("openstacks-opt08-adl__p03");
    const auto sp = space_of(task);
    ASSERT_TRUE(sp);
    const auto relaxed = H::ground(*task);
    const std::vector<u32> ids = sample(*sp, 120);
    const H::Kind kinds[] = {H::Kind::LmCut};
    std::vector<std::vector<double>> expected;
    for (const H::Kind k : kinds)
    {
        auto h = make(*task, k, H::Costs::Auto, relaxed);
        auto& e = expected.emplace_back();
        for (u32 id : ids)
            e.push_back(h->evaluate(sp->state(id)));
    }
    constexpr u32 k_threads = 4;
    std::vector<std::vector<std::vector<double>>> got(k_threads);
    {
        std::vector<std::jthread> pool;
        for (u32 t = 0; t < k_threads; ++t)
            pool.emplace_back(
                [&, t]
                {
                    for (const H::Kind k : kinds)
                    {
                        auto h = make(*task, k, H::Costs::Auto, relaxed);
                        auto& g = got[t].emplace_back();
                        for (usize i = 0; i < ids.size(); ++i)
                            g.push_back(h->evaluate(sp->state(ids[(i + t * 7) % ids.size()])));
                    }
                });
    }
    for (u32 t = 0; t < k_threads; ++t)
        for (usize k = 0; k < std::size(kinds); ++k)
            for (usize i = 0; i < ids.size(); ++i)
                EXPECT_EQ(got[t][k][i], expected[k][(i + t * 7) % ids.size()]) << t << " " << k << " " << i;
}
