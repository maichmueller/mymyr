// h², set-additive and the perfect heuristic on small suite tasks (the comparison with the fork is
// mymyr_heuristics_golden_tests):
//   - h² equals a plain Bellman-Ford evaluation of its definition (written here independently), and
//     h_max <= h² <= h* on states of the whole state space (h* from the perfect heuristic), so h² is infinite only on
//     dead ends;
//   - set-additive: finite iff h_max is, its relaxed plan (the ground actions of the union of the goal's achiever sets)
//     reaches the goal in the delete relaxation and has at most h actions, the preferred operators are its applicable
//     actions;
//   - perfect: the state space's goal distances, errors for states outside the space and for other goals, and A* with
//     it expands exactly the states of one optimal plan;
//   - one heuristic per thread over one shared task and grounding gives the single-threaded values (TSan build).

#include "../support/suite.hpp"
#include "../support/heuristic_task.hpp"
#include "h2_reference.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/perfect.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;
namespace H = mymyr::heuristics;

namespace
{
// Exhaustive state spaces of 40k-100k states (suite.hpp); philosophers and openstacks have derived predicates.
const char* const k_tasks[] = {"depot__p02", "philosophers__p03-phil4", "openstacks-opt08-adl__p03", "pegsol-08-strips__p22"};

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

std::unique_ptr<H::Heuristic> make(const Task& task, H::Kind k, std::shared_ptr<const H::RelaxedTask> relaxed = nullptr)
{
    H::Options o;
    o.kind = k;
    o.relaxed = std::move(relaxed);
    return H::make_heuristic(task, o);
}

/// Whether the ground actions of `plan` (with every axiom) reach the goal from s in the delete relaxation.
bool relaxed_plan_reaches_goal(const Task& task, const H::RelaxedTask& R, StateView s, const std::vector<Action>& plan)
{
    std::set<u32> gas;
    for (const Action& a : plan)
        gas.insert(R.find_ground_action(a.label()));
    std::vector<u8> reached(R.num_props(), 0);
    for (u32 p : test::true_props(task, R, s))
        reached[p] = 1;
    for (bool changed = true; changed;)
    {
        changed = false;
        for (u32 o = 0; o < R.num_ops(); ++o)
        {
            if (!R.is_axiom(o) && !gas.contains(R.ground_action(o)))
                continue;
            bool ok = true;
            for (u32 p : R.pre(o))
                ok = ok && reached[p];
            if (!ok)
                continue;
            for (u32 q : R.eff(o))
                if (!reached[q])
                    reached[q] = changed = true;
        }
    }
    for (u32 g : R.goal())
        if (!reached[g])
            return false;
    return true;
}
}  // namespace

TEST(Heuristics, KindNames)
{
    EXPECT_EQ(H::parse_kind("set_additive"), H::Kind::SetAdditive);
    EXPECT_EQ(H::parse_kind("hsa"), H::Kind::SetAdditive);
    EXPECT_EQ(H::parse_kind("setadd"), H::Kind::SetAdditive);
    EXPECT_EQ(H::parse_kind("h2"), H::Kind::H2);
    EXPECT_STREQ(H::to_string(H::Kind::SetAdditive), "set_additive");
    EXPECT_STREQ(H::to_string(H::Kind::H2), "h2");
    EXPECT_STREQ(H::to_string(H::Kind::Perfect), "perfect");
    EXPECT_THROW((void)H::parse_kind("perfect"), std::invalid_argument);  // needs a state space
    EXPECT_THROW((void)H::parse_kind("h3"), std::invalid_argument);

    const auto task = load("depot__p02");
    H::Options o;
    o.kind = H::Kind::Perfect;
    EXPECT_THROW((void)H::make_heuristic(*task, o), std::invalid_argument);
    for (const auto k : {H::Kind::SetAdditive, H::Kind::H2})  // grounded only
    {
        o.kind = k;
        o.evaluation = H::Evaluation::Lifted;
        EXPECT_THROW((void)H::make_heuristic(*task, o), std::invalid_argument);
        o.evaluation = H::Evaluation::Auto;
        o.budget.max_operators = 1;
        EXPECT_THROW((void)H::make_heuristic(*task, o), std::runtime_error);
        o.budget = {};
        EXPECT_EQ(H::make_heuristic(*task, o)->kind(), k);
    }
}

TEST(Heuristics, H2IsItsDefinitionAndBetweenHmaxAndHstar)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto sp = space_of(task);
        ASSERT_TRUE(sp);
        const auto relaxed = H::ground(*task);
        ASSERT_TRUE(relaxed);
        auto h2 = make(*task, H::Kind::H2, relaxed);
        auto hmax = make(*task, H::Kind::Max, relaxed);
        auto hstar = H::perfect(sp);
        test::ReferenceH2 ref(*relaxed);
        u32 checked = 0, dead = 0, above_hmax = 0, referenced = 0;
        for (u32 id : sample(*sp, 400))
        {
            const StateView s = sp->state(id);
            const double v2 = h2->evaluate(s), vm = hmax->evaluate(s), vs = hstar->evaluate(s);
            EXPECT_LE(vm, v2) << id;
            EXPECT_LE(v2, vs) << id;
            if (std::isinf(v2))
                ++dead;
            above_hmax += v2 > vm;
            if (checked++ % 8 == 0)  // the reference is slow
            {
                const double vr = relaxed->goal_unreachable() ? H::k_dead_end : ref.evaluate(test::true_props(*task, *relaxed, s), relaxed->goal());
                EXPECT_EQ(v2, vr) << id;
                ++referenced;
            }
        }
        std::printf("H2 %-32s states %u (%u against the reference) | h2 > h_max %u | dead ends %u\n", name, checked, referenced,
                    above_hmax, dead);
    }
}

TEST(Heuristics, H2WithConditionalEffectsIsAdmissible)
{
    // miconic-simpleadl: every action has conditional effects (boarding and leaving passengers)
    const auto task = load("miconic-simpleadl__s10-2");
    ASSERT_TRUE(task->compiled().has_conditional_effects);
    const auto relaxed = H::ground(*task);
    auto h2 = make(*task, H::Kind::H2, relaxed);
    auto hmax = make(*task, H::Kind::Max, relaxed);
    test::ReferenceH2 ref(*relaxed);
    // states along an optimal plan: h² <= the remaining plan length
    search::BestFirstOptions so;
    so.heuristic.kind = H::Kind::Max;
    const auto r = search::astar_eager(*task, so);
    ASSERT_EQ(r.status, search::SearchStatus::Solved);
    Successors& succ = task->workspace().successors();
    State s = task->initial_state();
    for (usize i = 0; i <= r.plan.size(); ++i)
    {
        const double v2 = h2->evaluate(s);
        EXPECT_LE(hmax->evaluate(s), v2);
        EXPECT_LE(v2, static_cast<double>(r.plan.size() - i));
        EXPECT_EQ(v2, ref.evaluate(test::true_props(*task, *relaxed, s), relaxed->goal()));
        if (i < r.plan.size())
            s = succ.apply(s, r.plan[i].label());
    }
}

TEST(Heuristics, SetAdditiveIsARelaxedPlan)
{
    for (const char* name : k_tasks)
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto sp = space_of(task);
        ASSERT_TRUE(sp);
        const auto relaxed = H::ground(*task);
        auto hsa = make(*task, H::Kind::SetAdditive, relaxed);
        auto hmax = make(*task, H::Kind::Max, relaxed);
        auto hff = make(*task, H::Kind::FF, relaxed);
        ASSERT_TRUE(hsa->provides_preferred());
        Successors& succ = task->workspace().successors();
        const bool ce = task->compiled().has_conditional_effects;
        u32 checked = 0, above_ff = 0, below_ff = 0;
        for (u32 id : sample(*sp, 300))
        {
            const StateView s = sp->state(id);
            const double vm = hmax->evaluate(s), vf = hff->evaluate(s), v = hsa->evaluate(s);
            ++checked;
            above_ff += v > vf;
            below_ff += v < vf;
            EXPECT_EQ(std::isinf(v), std::isinf(vm)) << id;
            if (std::isinf(v))
            {
                EXPECT_TRUE(hsa->relaxed_plan().empty());
                continue;
            }
            if (!ce)
            {
                EXPECT_GE(v, vm) << id;
            }
            const std::vector<Action> plan = hsa->relaxed_plan();
            EXPECT_LE(static_cast<double>(plan.size()), v) << id;  // a ground action can be several members
            EXPECT_EQ(plan.empty(), v == 0) << id;
            EXPECT_TRUE(relaxed_plan_reaches_goal(*task, *relaxed, s, plan)) << id;
            std::set<u32> in_plan;
            for (const Action& a : plan)
                in_plan.insert(relaxed->find_ground_action(a.label()));
            for (const Action& a : succ.applicable_actions(s))
                EXPECT_EQ(hsa->preferred(a.label()), in_plan.contains(relaxed->find_ground_action(a.label()))) << id;
        }
        std::printf("SETADD %-32s states %u | above h_FF %u, below %u\n", name, checked, above_ff, below_ff);
    }
}

TEST(Heuristics, PerfectIsTheGoalDistance)
{
    const auto task = load("depot__p02");
    const auto sp = space_of(task);
    ASSERT_TRUE(sp);
    auto h = H::perfect(sp);
    EXPECT_EQ(h->kind(), H::Kind::Perfect);
    for (u32 id : sample(*sp, 2000))
    {
        const i32 d = sp->unit_goal_distances()[id];
        EXPECT_EQ(h->evaluate(sp->state(id)), d < 0 ? H::k_dead_end : static_cast<double>(d));
    }
    auto hc = H::perfect(sp, H::Costs::Real);
    EXPECT_EQ(hc->evaluate(sp->state(0)), sp->cost_goal_distances()[0]);
    // batched: a lookup per state
    std::vector<StateView> views = {sp->state(0), sp->state(1)};
    std::vector<double> out(2);
    h->evaluate_batch(views, out);
    EXPECT_EQ(out[0], h->evaluate(sp->state(0)));
    EXPECT_EQ(out[1], h->evaluate(sp->state(1)));
    // other goals and states of another task are refused
    std::vector<search::GoalSpec::AtomGoal> goals(1);
    EXPECT_THROW((void)h->evaluate(sp->state(0), goals), std::invalid_argument);
    const auto other = load("pegsol-08-strips__p22");
    EXPECT_THROW((void)h->evaluate(other->initial_state().view()), std::invalid_argument);
    EXPECT_THROW((void)H::perfect(nullptr), std::invalid_argument);
    datasets::StateSpaceOptions so;
    so.symmetry_pruning = true;
    so.certificate = datasets::CertificateKind::ColorRefinement;  // the cheap one
    so.labels = false;
    const auto reduced = datasets::generate_state_space(task, so);
    ASSERT_TRUE(reduced.space);
    EXPECT_THROW((void)H::perfect(reduced.space), std::invalid_argument);
}

TEST(Heuristics, AStarWithThePerfectHeuristicExpandsOnePlan)
{
    for (const char* name : {"depot__p02", "pegsol-08-strips__p22", "philosophers__p03-phil4"})
    {
        SCOPED_TRACE(name);
        const auto task = load(name);
        const auto sp = space_of(task);
        ASSERT_TRUE(sp);
        auto h = H::perfect(sp);
        search::BestFirstOptions o;
        o.evaluator = h.get();
        const auto r = search::astar_eager(*task, o);
        ASSERT_EQ(r.status, search::SearchStatus::Solved);
        EXPECT_EQ(static_cast<i32>(r.plan.size()), sp->unit_goal_distances()[0]);
        // the states of the plan but the goal, which is returned when popped without being expanded (mimir's A*
        // counts the goal as expanded and does not break ties among equal f by h, so it expands more states)
        EXPECT_EQ(r.stats.expanded, r.plan.size());
    }
}

TEST(Heuristics, OneHeuristicPerThreadOverASharedTask)
{
    const auto task = load("depot__p02");
    const auto sp = space_of(task);
    ASSERT_TRUE(sp);
    const auto relaxed = H::ground(*task);
    const std::vector<u32> ids = sample(*sp, 120);
    const H::Kind kinds[] = {H::Kind::SetAdditive, H::Kind::H2, H::Kind::Perfect};
    auto make_kind = [&](H::Kind k) { return k == H::Kind::Perfect ? H::perfect(sp) : make(*task, k, relaxed); };
    std::vector<std::vector<double>> expected;
    for (const H::Kind k : kinds)
    {
        auto h = make_kind(k);
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
                        auto h = make_kind(k);
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

TEST(Heuristics, H2RefusesPairTablesBeyondThePropositionLimit)
{
    const auto task = initial_proposition_task(8192);
    const auto relaxed = H::ground(*task);
    ASSERT_TRUE(relaxed);
    ASSERT_EQ(relaxed->num_props(), 8192u);
    H::Options o;
    o.kind = H::Kind::H2;
    o.relaxed = relaxed;
    try
    {
        (void)H::make_heuristic(*task, o);
        FAIL() << "accepted a pair table beyond the proposition limit";
    }
    catch (const std::invalid_argument& e)
    {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("8192"), std::string::npos);
        EXPECT_NE(msg.find("8191"), std::string::npos);
    }
}
