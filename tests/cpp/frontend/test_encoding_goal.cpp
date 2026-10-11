// The encoding goal (TaskData::encoding_goal): the goal as written, expanded into signed ground literals. Problems in
// tests/data/pddl/goals differ only in their goal; each test lists the literals that goal expands to.

#include "mymyr/formalism/fingerprint.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::formalism;
namespace fs = std::filesystem;

namespace
{
const fs::path k_pddl = fs::path(MYMYR_SOURCE_DIR) / "tests/data/pddl";

/// "(pred o1 o2)", "(not (pred o1))": objects by name, or by index where a text round trip has renamed them.
std::string text(const TaskData& t, const Literal& l, bool by_index = false)
{
    std::string s = "(" + std::string(t.str(t.predicate(l.pred).name));
    for (Term x : t.terms_of(l))
        s += " " + (by_index ? std::to_string(term_object(x).v) : std::string(t.str(t.objects[term_object(x).v].name)));
    s += ")";
    return l.positive ? s : "(not " + s + ")";
}

std::vector<std::string> goal_text(const TaskData& t, bool by_index = false)
{
    std::vector<std::string> out;
    for (const Literal& l : t.encoding_goal_literals())
        out.push_back(text(t, l, by_index));
    return out;
}

std::set<std::string> literal_set(const TaskData& t, bool by_index = false)
{
    const auto v = goal_text(t, by_index);
    return {v.begin(), v.end()};
}

std::set<std::string> planner_goal(const TaskData& t)
{
    std::set<std::string> out;
    for (const Literal& l : t.literals_of(t.goal))
        out.insert(text(t, l));
    return out;
}

frontend::TaskPtr load(const std::string& domain, const std::string& problem)
{
    return frontend::Domain::from_file(k_pddl / domain / "domain.pddl")->instantiate_file(k_pddl / domain / problem);
}

struct Case
{
    const char* problem;
    std::set<std::string> literals;
    bool complete;
    bool unsatisfiable;
};

const std::vector<Case>& cases()
{
    static const std::vector<Case> v = {
        {"conjunctive", {"(lit a)", "(not (ok t1))", "(at t2 b)", "(done)"}, true, false},
        // objects in order: constants, then the problem's
        {"forall", {"(lit home)", "(lit a)", "(lit b)"}, true, false},
        {"forall_nested", {"(at b1 home)", "(at b1 a)", "(at b1 b)"}, true, false},
        // the constant k0 and the objects of the subtype big belong to thing
        {"forall_subtypes_constants", {"(ok k0)", "(ok t1)", "(ok t2)", "(ok b1)"}, true, false},
        // a variable of type (either big place) ranges over the objects of either type
        {"forall_either", {"(seen home)", "(seen a)", "(seen b)", "(seen b1)"}, true, false},
        {"forall_empty_type", {"(lit a)"}, true, false},
        {"forall_negative", {"(not (ok b1))"}, true, false},
        {"forall_conjunctive_body", {"(ok b1)", "(not (at b1 a))"}, true, false},
        {"imply_true", {"(ok t1)", "(at b1 home)", "(at b1 a)", "(at b1 b)"}, true, false},
        // an antecedent that fails statically makes the implication hold: nothing to require
        {"imply_false", {"(lit a)"}, true, false},
        {"imply_equality", {"(lit a)"}, true, false},
        // the antecedents (exists, forall over thing / big) are decided by static atoms: true, true, false
        {"imply_quantified_antecedent", {"(flag)", "(ok t2)"}, true, false},
        {"imply_fluent", {"(lit a)"}, false, false},
        {"or", {"(lit a)"}, false, false},
        {"exists", {"(lit a)"}, false, false},
        {"forall_exists", {"(lit a)"}, false, false},
        {"negated_forall", {"(lit a)"}, false, false},
        {"static_false", {"(heavy t2)", "(lit a)"}, true, true},
        // equality is a static predicate: (not (= a a)) fails
        {"forall_equality", {"(not (= home a))", "(not (= a a))", "(not (= b a))"}, true, true},
        {"static_true", {"(heavy t1)", "(not (heavy t2))", "(lit a)"}, true, false},
        {"duplicate", {"(lit home)", "(lit a)", "(lit b)", "(not (ok t1))"}, true, false},
    };
    return v;
}
}  // namespace

TEST(EncodingGoal, ExpandsEveryGoalShape)
{
    for (const Case& c : cases())
    {
        SCOPED_TRACE(c.problem);
        const auto t = load("goals", std::string(c.problem) + ".pddl");
        const auto got = goal_text(*t);
        EXPECT_EQ(std::set<std::string>(got.begin(), got.end()), c.literals);
        EXPECT_EQ(got.size(), c.literals.size()) << "duplicates";
        EXPECT_EQ(t->encoding_goal_complete, c.complete);
        EXPECT_EQ(t->encoding_goal_unsatisfiable, c.unsatisfiable);
    }
}

TEST(EncodingGoal, LiteralsAreSortedBySignPredicateAndObjects)
{
    for (const Case& c : cases())
    {
        SCOPED_TRACE(c.problem);
        const auto t = load("goals", std::string(c.problem) + ".pddl");
        const auto lits = t->encoding_goal_literals();
        auto key = [&](const Literal& l)
        {
            std::vector<u32> k{l.positive ? 0u : 1u, l.pred.v};
            for (Term x : t->terms_of(l))
                k.push_back(term_object(x).v);
            return k;
        };
        for (size_t i = 1; i < lits.size(); ++i)
            EXPECT_LT(key(lits[i - 1]), key(lits[i]));
    }
}

TEST(EncodingGoal, ConjunctiveGoalsEqualTheNormalizedGoal)
{
    const std::vector<std::pair<const char*, const char*>> tasks = {
        {"blocks", "probBLOCKS-8-0.pddl"},   {"gripper", "p01.pddl"},        {"gripper", "p02.pddl"},
        {"logistics00", "probLOGISTICS-6-1.pddl"}, {"visitall", "test_problem.pddl"}, {"delivery", "test_problem.pddl"},
        {"philosophers", "p03-phil4.pddl"},  {"counters", "p01.pddl"},
    };
    for (const auto& [domain, problem] : tasks)
    {
        SCOPED_TRACE(std::string(domain) + "/" + problem);
        const auto t = load(domain, problem);
        const auto got = goal_text(*t);
        EXPECT_EQ(std::set<std::string>(got.begin(), got.end()), planner_goal(*t));
        EXPECT_TRUE(t->encoding_goal_complete);
        EXPECT_FALSE(t->encoding_goal_unsatisfiable);
        EXPECT_TRUE(has_default_encoding_goal(*t));
    }
    const auto t = load("goals", "conjunctive.pddl");
    EXPECT_EQ(literal_set(*t), planner_goal(*t));
}

TEST(EncodingGoal, ThePlannersGoalIsUnchanged)
{
    // forall: the planner keeps the derived predicate of the normalized goal
    const auto t = load("goals", "forall.pddl");
    ASSERT_EQ(t->goal.literals.count, 1u);
    EXPECT_EQ(t->predicate(t->literals[t->goal.literals.begin].pred).kind, PredKind::Derived);
    EXPECT_FALSE(has_default_encoding_goal(*t));
    EXPECT_EQ(t->encoding_goal.count, 3u);
}

TEST(EncodingGoal, TaskAccessors)
{
    const auto data = load("goals", "static_false.pddl");
    const auto task = Task::create(*data);
    EXPECT_EQ(task->encoding_goal().size(), 2u);
    EXPECT_TRUE(task->encoding_goal_complete());
    EXPECT_TRUE(task->encoding_goal_unsatisfiable());
    EXPECT_TRUE(task->is_goal(task->initial_state()) == false);
}

TEST(EncodingGoal, NumericConstraintsAreNotExpandable)
{
    const auto d = frontend::Domain::from_string(R"((define (domain n) (:requirements :strips :numeric-fluents)
 (:predicates (p)) (:functions (f))
 (:action inc :parameters () :precondition (and) :effect (and (p) (increase (f) 1)))))");
    const auto t = d->instantiate_string(R"((define (problem q) (:domain n) (:init (= (f) 0))
 (:goal (and (p) (>= (f) 2)))))");
    EXPECT_EQ(goal_text(*t), std::vector<std::string>{"(p)"});
    EXPECT_FALSE(t->encoding_goal_complete);
}

TEST(EncodingGoal, TextWithTheGoalRoundTrips)
{
    for (const Case& c : cases())
    {
        SCOPED_TRACE(c.problem);
        const auto t = load("goals", std::string(c.problem) + ".pddl");
        std::istringstream in(write_task_text(*t, true));
        const TaskData back = read_task_text(in);
        EXPECT_EQ(goal_text(back, true), goal_text(*t, true));
        EXPECT_EQ(back.encoding_goal_complete, t->encoding_goal_complete);
        EXPECT_EQ(back.encoding_goal_unsatisfiable, t->encoding_goal_unsatisfiable);
        // a text round trip keeps the fingerprint (it renames objects, so compare two of them)
        std::istringstream again(write_task_text(back, true));
        EXPECT_EQ(fingerprint(read_task_text(again)), fingerprint(back));
    }
}

TEST(EncodingGoal, TextWithoutTheGoalAsWrittenKeepsTheConjunctivePart)
{
    // the text format has no goal as written: literals over predicates that normalization introduced are left out
    for (const char* problem : {"conjunctive", "forall", "or", "imply_true", "imply_fluent", "static_false", "negated_forall"})
    {
        SCOPED_TRACE(problem);
        const auto t = load("goals", std::string(problem) + ".pddl");
        std::set<std::string> kept;
        bool complete = true;
        for (const Literal& l : t->literals_of(t->goal))
        {
            if (t->str(t->predicate(l.pred).name).starts_with("axiom_"))
                complete = false;
            else
                kept.insert(text(*t, l, true));
        }
        std::istringstream in(write_task_text(*t));
        const TaskData back = read_task_text(in);
        EXPECT_EQ(literal_set(back, true), kept);
        EXPECT_EQ(back.encoding_goal_complete, complete);
        EXPECT_TRUE(has_default_encoding_goal(back));
    }
}

TEST(EncodingGoal, FingerprintCoversTheEncodingGoalOnlyWhenItIsNotTheDefault)
{
    const auto conjunctive = load("goals", "conjunctive.pddl");
    EXPECT_TRUE(has_default_encoding_goal(*conjunctive));
    // the fingerprint of a task with the default encoding goal is that of its text and names alone
    TaskData same = *conjunctive;
    set_encoding_goal_from_normalized(same);
    EXPECT_EQ(fingerprint(same), fingerprint(*conjunctive));

    const auto forall = load("goals", "forall.pddl");
    TaskData stripped = *forall;
    set_encoding_goal_from_normalized(stripped);
    EXPECT_NE(fingerprint(stripped), fingerprint(*forall));
}

TEST(EncodingGoal, IsIndependentOfTheInitPath)
{
    const auto d = frontend::Domain::from_file(k_pddl / "goals/domain.pddl");
    for (const Case& c : cases())
    {
        SCOPED_TRACE(c.problem);
        frontend::InstantiateOptions fast, full;
        fast.fast_init = true;
        full.fast_init = false;
        const auto path = k_pddl / "goals" / (std::string(c.problem) + ".pddl");
        EXPECT_EQ(goal_text(*d->instantiate_file(path, fast)), goal_text(*d->instantiate_file(path, full)));
    }
}
