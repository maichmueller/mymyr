// Action parameters that nothing but their type mentions (formalism::CollapsedParameter): collapsed by default, kept
// with InstantiateOptions::collapse_unused_parameters = false.

#include "mymyr/formalism/fingerprint.hpp"
#include "mymyr/formalism/pddl_writer.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/plan_file.hpp"
#include "mymyr/search/validate.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::search;

namespace
{
// press: ?s and ?r occur nowhere but in their types. tune: ?s occurs in a conditional effect's condition, ?k in a
// numeric effect, ?r in an equality; nothing of tune collapses. flip: an `or` splits it; ?a is unused in the branch
// that tests ?b, ?b in the one that tests ?a.
constexpr const char* kDomain = R"((define (domain lights)
  (:requirements :typing :conditional-effects :equality :disjunctive-preconditions :numeric-fluents)
  (:types switch room knob)
  (:constants main - switch)
  (:predicates (on) (lit) (wired ?s - switch) (hot ?r - room) (bright))
  (:functions (level ?k - knob))
  (:action press :parameters (?s - switch ?r - room) :precondition (on) :effect (lit))
  (:action tune :parameters (?s - switch ?r - room ?k - knob) :precondition (not (= ?r ?r))
    :effect (and (when (wired ?s) (bright)) (increase (level ?k) 1)))
  (:action flip :parameters (?a - room ?b - room) :precondition (or (hot ?a) (hot ?b)) :effect (bright)))))";

constexpr const char* kProblem = R"((define (problem p) (:domain lights)
  (:objects s1 s2 - switch r1 r2 r3 - room k1 - knob)
  (:init (on) (hot r2) (= (level k1) 0)) (:goal (and (lit) (bright)))))";

// no object of type room: press has no action
constexpr const char* kNoRooms = R"((define (problem p) (:domain lights)
  (:objects s1 - switch k1 - knob)
  (:init (on) (= (level k1) 0)) (:goal (lit))))";

std::shared_ptr<const Task> make(const char* problem, bool collapse)
{
    const auto domain = frontend::Domain::from_string(kDomain, "d.pddl");
    return Task::create(*domain->instantiate_string(problem, "p.pddl", {.collapse_unused_parameters = collapse}));
}

u32 schema(const Task& t, std::string_view name, u32 nth = 0)
{
    for (u32 s = 0; s < t.num_schemas(); ++s)
        if (t.data().str(t.data().schemas[s].name) == name && nth-- == 0)
            return s;
    throw std::logic_error("no schema");
}

std::vector<std::string> applicable(const Task& t, const State& s, std::string_view name)
{
    const WorkspaceLease lease = t.workspace();
    Successors& succ = lease->successors();
    succ.set_witness_pruning(false);
    std::vector<std::string> out;
    for (const Action& a : succ.applicable_actions(s.view()))
        if (t.data().str(t.data().schemas[a.schema.v].name) == name)
            out.push_back(t.format(a.label()));
    return out;
}

std::string error_of(const Task& t, const std::string& text)
{
    try
    {
        (void)parse_plan(t, text);
    }
    catch (const std::invalid_argument& e)
    {
        return e.what();
    }
    return "no error";
}
}  // namespace

TEST(UnusedParameters, OnlyParametersThatNothingButTheirTypeMentionsCollapse)
{
    const auto task = make(kProblem, true);
    const formalism::TaskData& D = task->data();
    ASSERT_TRUE(D.unused_parameters_collapsed);
    const formalism::Schema& press = D.schemas[schema(*task, "press")];
    EXPECT_EQ(press.arity(), 0u);
    EXPECT_EQ(press.original_arity, 0u);
    EXPECT_EQ(formalism::written_arity(D, press), 2u);
    const auto collapsed = D.collapsed_of(press);
    ASSERT_EQ(collapsed.size(), 2u);
    EXPECT_EQ(collapsed[0].position, 0u);
    EXPECT_EQ(collapsed[1].position, 1u);
    EXPECT_TRUE(collapsed[0].written && collapsed[1].written);
    // representatives: the first object of the type in object order, the domain's constants first
    EXPECT_EQ(D.str(D.objects[collapsed[0].representative.v].name), "main");
    EXPECT_EQ(D.str(D.objects[collapsed[1].representative.v].name), "r1");
    EXPECT_EQ(D.str(D.params[collapsed[1].param].name), "r");
    EXPECT_EQ(D.collapsed_of(D.schemas[schema(*task, "tune")]).size(), 0u);
    // the branches of flip: ?b collapses in the branch that tests ?a and the other way round
    const formalism::Schema& f0 = D.schemas[schema(*task, "flip", 0)];
    const formalism::Schema& f1 = D.schemas[schema(*task, "flip", 1)];
    ASSERT_EQ(f0.arity(), 1u);
    ASSERT_EQ(f1.arity(), 1u);
    EXPECT_NE(D.collapsed_of(f0)[0].position, D.collapsed_of(f1)[0].position);

    const auto full = make(kProblem, false);
    EXPECT_FALSE(full->data().unused_parameters_collapsed);
    for (const formalism::Schema& s : full->data().schemas)
    {
        EXPECT_EQ(s.collapsed.count, 0u);
        EXPECT_EQ(s.original_arity, s.arity());
    }
}

TEST(UnusedParameters, OneActionStandsForEveryObjectOfTheType)
{
    const auto task = make(kProblem, true);
    const auto full = make(kProblem, false);
    EXPECT_EQ(applicable(*task, task->initial_state(), "press"), std::vector<std::string>{"(press main r1)"});
    EXPECT_EQ(applicable(*full, full->initial_state(), "press").size(), 9u);  // 3 switches, 3 rooms
    // flip: ?a = r2 with any ?b, or ?b = r2 with any ?a; collapsed, each branch has one action
    EXPECT_EQ(applicable(*task, task->initial_state(), "flip"),
              (std::vector<std::string>{"(flip r2 r1)", "(flip r1 r2)"}));
    EXPECT_EQ(applicable(*full, full->initial_state(), "flip").size(), 6u);  // (r2, *) and (*, r2) per branch
    // the successor states are the same
    const auto states = [](const Task& t) {
        const WorkspaceLease lease = t.workspace();
        Successors& succ = lease->successors();
        std::vector<std::string> out;
        for (const Action& a : succ.applicable_actions(t.initial_state().view()))
        {
            const State s = succ.apply(t.initial_state().view(), a.label());
            std::vector<std::string> atoms = t.format_atoms(s.view());
            std::string line;
            for (const std::string& a : atoms)
                line += a;
            out.push_back(line);
        }
        std::ranges::sort(out);
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };
    EXPECT_EQ(states(*task), states(*full));
}

TEST(UnusedParameters, ASchemaWhoseParameterHasNoObjectOfItsTypeHasNoAction)
{
    for (const bool collapse : {true, false})
    {
        const auto task = make(kNoRooms, collapse);
        EXPECT_TRUE(applicable(*task, task->initial_state(), "press").empty()) << collapse;
        if (!collapse)
            continue;
        const formalism::Schema& press = task->data().schemas[schema(*task, "press")];
        EXPECT_FALSE(formalism::has_representatives(task->data(), press));
        const WorkspaceLease lease = task->workspace();
        EXPECT_FALSE(lease->successors().is_applicable(task->initial_state().view(),
                                                       ActionLabel{SchemaId{schema(*task, "press")}, {}}));
        // no room object: a plan line cannot name one
        const PlanValidation v = validate_plan_text(*task, "(press s1 k1)\n");
        EXPECT_EQ(v.error, PlanError::WrongType) << v.message;
    }
}

TEST(UnusedParameters, PlansWriteTheRepresentativeAndReadAnyObjectOfTheType)
{
    const auto task = make(kProblem, true);
    const std::vector<Action> plan = parse_plan(*task, "(press s2 r3)\n(flip r2 r3)\n");
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].binding.size(), 0u);
    EXPECT_EQ(plan, parse_plan(*task, "(press main r1)\n(flip r2 r1)\n"));
    const std::string text = format_plan(*task, plan);
    EXPECT_EQ(text.substr(0, text.find(';')), "(press main r1)\n(flip r2 r1)\n");
    const PlanValidation v = validate_plan_text(*task, "(press s1 r2)\n(flip r3 r2)\n");
    EXPECT_TRUE(v.valid) << v.message;
    EXPECT_TRUE(validate_plan(*task, plan).valid);
    // an object of another type names no action of the schema
    const std::string wrong = error_of(*task, "(press r1 r1)\n");
    EXPECT_NE(wrong.find("line 1 '(press r1 r1)': the argument r1 - switch is not of its parameter's type"),
              std::string::npos)
        << wrong;
    const PlanValidation w = validate_plan_text(*task, "(press s1 k1)\n");
    EXPECT_EQ(w.error, PlanError::WrongType) << w.message;
    EXPECT_EQ(w.condition, "k1 - room");
    EXPECT_NE(error_of(*task, "(press s1)\n").find("press takes 2 arguments, the line has 1"), std::string::npos);
}

TEST(UnusedParameters, TheSettingIsPartOfTheTask)
{
    const auto task = make(kProblem, true);
    const auto full = make(kProblem, false);
    EXPECT_NE(task->fingerprint(), full->fingerprint());
    // a domain without unused parameters: the tasks are equal but for the setting, and still distinguished
    const char* blocks = R"((define (domain b) (:requirements :strips) (:predicates (p ?x))
      (:action a :parameters (?x) :precondition (p ?x) :effect (not (p ?x)))))";
    const auto domain = frontend::Domain::from_string(blocks, "b.pddl");
    const char* problem = "(define (problem q) (:domain b) (:objects o) (:init (p o)) (:goal (and)))";
    EXPECT_NE(formalism::fingerprint(*domain->instantiate_string(problem, "", {.collapse_unused_parameters = true})),
              formalism::fingerprint(*domain->instantiate_string(problem, "", {.collapse_unused_parameters = false})));
    // the text format records it
    for (const auto* t : {task.get(), full.get()})
    {
        const std::string text = formalism::write_task_text(t->data(), true);
        std::istringstream in(text);
        const formalism::TaskData back = formalism::read_task_text(in);
        EXPECT_EQ(back.unused_parameters_collapsed, t->data().unused_parameters_collapsed);
        EXPECT_EQ(formalism::write_task_text(back, true), text);
    }
}

TEST(UnusedParameters, WrittenPddlDeclaresTheCollapsedParameters)
{
    const auto task = make(kProblem, true);
    const formalism::PddlFiles f = formalism::write_pddl(task->data());
    EXPECT_NE(f.domain.find(":parameters (?s - switch ?r - room)"), std::string::npos) << f.domain;
    // read back, it collapses to the same task
    const auto domain = frontend::Domain::from_string(f.domain, "d.pddl");
    const auto again = Task::create(*domain->instantiate_string(f.problems.at(0), "p.pddl"));
    EXPECT_EQ(formalism::write_task_text(again->data()), formalism::write_task_text(task->data()));
}
