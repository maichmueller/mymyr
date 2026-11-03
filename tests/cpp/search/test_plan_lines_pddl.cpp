// Plan lines (search::parse_plan, search::validate_plan_text) name any applicable binding, also bindings that differ
// only in parameters no effect mentions, which witness pruning enumerates once.

#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/search/validate.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
// work's hubs ?h ?g occur in its precondition only; dock adds an existential hub ?z by normalization, and its own
// hubs occur in no effect either
constexpr const char* kDomain = R"((define (domain binding-hubs)
  (:requirements :strips :typing :existential-preconditions)
  (:types agent tool hub job)
  (:predicates (link ?h - hub ?g - hub) (end ?t - tool ?h - hub) (free ?t - tool) (ready ?a - agent)
               (holding ?a - agent ?t - tool) (open ?x - job) (done ?x - job))
  (:action pick :parameters (?a - agent ?t - tool) :precondition (and (ready ?a) (free ?t))
    :effect (and (holding ?a ?t) (not (ready ?a)) (not (free ?t))))
  (:action work :parameters (?a - agent ?t - tool ?h - hub ?g - hub ?x - job)
    :precondition (and (holding ?a ?t) (end ?t ?h) (end ?t ?g) (link ?h ?g) (open ?x))
    :effect (and (done ?x) (free ?t) (not (open ?x)) (not (holding ?a ?t))))
  (:action dock :parameters (?a - agent ?t - tool ?h - hub ?g - hub ?x - job)
    :precondition (and (holding ?a ?t) (end ?t ?h) (end ?t ?g) (link ?h ?g) (open ?x) (exists (?z - hub) (link ?g ?z)))
    :effect (and (done ?x) (free ?t) (not (open ?x)) (not (holding ?a ?t))))))";

constexpr const char* kProblem = R"((define (problem p) (:domain binding-hubs)
  (:objects a - agent t - tool h1 h2 - hub x - job)
  (:init (ready a) (free t) (end t h1) (end t h2) (link h1 h2) (link h2 h1) (open x)) (:goal (done x))))";

std::shared_ptr<const Task> hubs()
{
    return Task::create(*frontend::Domain::from_string(kDomain, "d.pddl")->instantiate_string(kProblem, "p.pddl"));
}

/// The objects of an action's binding, by name.
std::vector<std::string> names(const Task& task, const Action& a)
{
    std::vector<std::string> out;
    for (const ObjectId o : a.binding)
        out.emplace_back(task.data().str(task.data().objects[o.v].name));
    return out;
}
}  // namespace

TEST(PlanLines, NameBindingsThatWitnessPruningEnumeratesOnce)
{
    const auto task = hubs();
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    ASSERT_TRUE(succ.witness_pruning());
    const State s1 = succ.apply(task->initial_state().view(), parse_plan(*task, "(pick a t)").front().label());
    const auto count_work = [&](bool witness) {
        succ.set_witness_pruning(witness);
        const std::vector<Action> all = succ.applicable_actions(s1.view());
        return std::ranges::count_if(
            all, [&](const Action& a) { return task->data().str(task->data().schemas[a.schema.v].name) == "work"; });
    };
    ASSERT_EQ(count_work(true), 1);  // the case the plan lines must not depend on
    ASSERT_EQ(count_work(false), 2);
    for (const std::string& line : {std::string("(work a t h1 h2 x)"), std::string("(work a t h2 h1 x)")})
    {
        const std::string text = "(pick a t)\n" + line + "\n";
        const std::vector<Action> plan = parse_plan(*task, text);
        ASSERT_EQ(plan.size(), 2u);
        const std::vector<std::string> want{"a", "t", line.substr(10, 2), line.substr(13, 2), "x"};
        EXPECT_EQ(names(*task, plan[1]), want) << line;
        const PlanValidation v = validate_plan_text(*task, text);
        EXPECT_TRUE(v.valid) << line << ": " << v.message;
        EXPECT_EQ(v.plan, plan);
        // a line of the PDDL action's own parameters leaves dock's added hub open: every completion is a candidate
        const std::string docked = "(pick a t)\n(dock" + line.substr(5) + "\n";
        const std::vector<Action> dock = parse_plan(*task, docked);
        ASSERT_EQ(dock.size(), 2u);
        EXPECT_EQ(dock[1].binding.size(), task->data().schemas[dock[1].schema.v].arity());
        EXPECT_GT(dock[1].binding.size(), 5u);
        std::vector<std::string> named = names(*task, dock[1]);
        named.resize(5);
        EXPECT_EQ(named, want);
        EXPECT_TRUE(validate_plan_text(*task, docked).valid) << docked;
    }
}

TEST(PlanLines, ErrorsNameTheLineAndTheArgument)
{
    const auto task = hubs();
    const auto message = [&](const std::string& text) -> std::string {
        try
        {
            (void)parse_plan(*task, text);
        }
        catch (const std::invalid_argument& e)
        {
            return e.what();
        }
        return "no error";
    };
    const std::string unknown = message("(pick a t)\n(work a t h1 h3 x)\n");
    EXPECT_NE(unknown.find("line 2 '(work a t h1 h3 x)': argument 4 'h3' is no object"), std::string::npos) << unknown;
    EXPECT_NE(message("(lift a t)\n").find("no action 'lift'"), std::string::npos);
    EXPECT_NE(message("(pick a)\n").find("pick takes 2 arguments, the line has 1"), std::string::npos);
    EXPECT_NE(message("(work a t h1 h2 x)\n").find("line 1 '(work a t h1 h2 x)': the action is not applicable"),
              std::string::npos);
    EXPECT_NE(message("(pick a t)\n(work a t h1 h1 x)\n").find("line 2"), std::string::npos);
    const PlanValidation v = validate_plan_text(*task, "(pick a t)\n(work a t h1 h1 x)\n");
    EXPECT_EQ(v.error, PlanError::NotApplicable);
    EXPECT_EQ(v.condition, "(link h1 h1)") << v.message;
}
