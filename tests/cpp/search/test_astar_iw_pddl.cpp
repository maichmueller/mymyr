#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/astar_iw.hpp"
#include "mymyr/task/task.hpp"
#include "../../../src/mymyr/search/abstracted_features.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>

using namespace mymyr;
using namespace mymyr::search;
namespace
{
std::shared_ptr<const Task> cost_task(u32 cost)
{
    const auto domain = frontend::Domain::from_string(
        "(define (domain d) (:requirements :strips :action-costs) (:predicates (ready) (goal)) "
        "(:functions (total-cost) - number) (:action finish :parameters () :precondition (ready) "
        ":effect (and (goal) (increase (total-cost) " + std::to_string(cost) + "))))");
    return Task::create(*domain->instantiate_string(
        "(define (problem p) (:domain d) (:init (ready) (= (total-cost) 7)) (:goal (goal)) "
        "(:metric minimize (total-cost)))"));
}
}  // namespace

TEST(AStarIwPddl, RejectsNonUnitCosts)
{
    EXPECT_THROW((void) astar_iw(*cost_task(2)), std::invalid_argument);
    EXPECT_THROW((void) astar_iw(*cost_task(0)), std::invalid_argument);
}

TEST(AStarIwPddl, UnitTotalCostIncludesInitialValue)
{
    const auto task = cost_task(1);
    const auto r = astar_iw(*task);
    ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
    EXPECT_EQ(r.cost, 8);
    EXPECT_EQ(r.plan.size(), 1u);
}

TEST(AStarIwPddl, RejectsNumericFluents)
{
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d) (:requirements :strips :fluents) (:predicates (goal))
 (:functions (fuel)) (:action go :parameters () :precondition (> (fuel) 0)
 :effect (and (goal) (decrease (fuel) 1))))
)");
    const auto task = Task::create(*domain->instantiate_string(
        "(define (problem p) (:domain d) (:init (= (fuel) 2)) (:goal (goal)))"));
    EXPECT_THROW((void) astar_iw(*task), std::invalid_argument);
}

TEST(AStarIwPddl, GoalAndLandmarkPreservationAddFullIdentity)
{
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d) (:requirements :strips) (:predicates (p ?x ?y))
 (:action add :parameters (?x ?y) :precondition (and) :effect (p ?x ?y)))
)");
    const auto task = Task::create(*domain->instantiate_string(
        "(define (problem p) (:domain d) (:objects a b c) (:init (p a b)) (:goal (p a c)))"));
    const auto pred = task->data().literals_of(task->data().goal).front().pred;
    const std::array<u32, 2> args{0, 2};
    const auto atom = task->atoms().layout().encode(pred.v, args.data());
    const u32 slot = task->atoms().intern(atom);
    for (bool base : {false, true})
    {
        mymyr::search::detail::AbstractedFeatures plain(*task, base, false, nullptr, false);
        mymyr::search::detail::AbstractedFeatures goal(*task, base, true, nullptr, false);
        novelty::LandmarkCoordinates coords({{slot}});
        mymyr::search::detail::AbstractedFeatures landmark(*task, base, false, &coords, true);
        mymyr::search::detail::AbstractedFeatures ignored(*task, base, false, &coords, false);
        EXPECT_EQ(plain.features(slot).size(), 2u);
        EXPECT_EQ(goal.features(slot).size(), 3u);
        EXPECT_EQ(landmark.features(slot).size(), 3u);
        EXPECT_EQ(ignored.features(slot).size(), 2u);
    }
}

TEST(AStarIwPddl, ParameterizedCostsAreCheckedOnGeneration)
{
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d) (:requirements :strips :action-costs) (:predicates (goal ?x))
 (:functions (total-cost) - number (price ?x) - number)
 (:action finish :parameters (?x) :precondition (and)
 :effect (and (goal ?x) (increase (total-cost) (price ?x)))))
)");
    for (u32 cost : {1u, 2u})
    {
        const auto task = Task::create(*domain->instantiate_string(
            "(define (problem p) (:domain d) (:objects a) (:init (= (total-cost) 0) (= (price a) " +
            std::to_string(cost) + ")) (:goal (goal a)) (:metric minimize (total-cost)))"));
        if (cost == 1)
        {
            EXPECT_EQ(astar_iw(*task).status, SearchStatus::Solved);
        }
        else
        {
            EXPECT_THROW((void) astar_iw(*task), std::invalid_argument);
        }
    }
}

TEST(AStarIwPddl, ConditionalUnitCostsAreAccepted)
{
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d) (:requirements :strips :conditional-effects :action-costs)
 (:predicates (ready) (goal)) (:functions (total-cost) - number)
 (:action finish :parameters () :precondition (ready)
 :effect (and (goal) (when (ready) (increase (total-cost) 1)))))
)");
    const auto task = Task::create(*domain->instantiate_string(
        "(define (problem p) (:domain d) (:init (ready) (= (total-cost) 0)) (:goal (goal)))"));
    const auto r = astar_iw(*task);
    EXPECT_EQ(r.status, SearchStatus::Solved);
    EXPECT_EQ(r.cost, 1);
}
