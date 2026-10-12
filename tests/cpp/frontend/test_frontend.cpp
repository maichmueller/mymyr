// Front end: loki parse + normalize once per Domain, the translation to a TaskData per problem, and the one-pass `:init`
// reader. Fast and full paths token-equal; the output independent of heap addresses; one Domain shared by 16 threads;
// a large problem instantiated fast.

#include "../support/benchmarks.hpp"
#include "../support/text.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <thread>

using namespace mymyr;
using namespace mymyr::formalism;
namespace fs = std::filesystem;

// loki's type order depends on heap addresses; two Domain instances in one process see different addresses, so this
// catches any address dependence the canonical type order misses.
TEST(Frontend, OutputIndependentOfHeapAddresses)
{
    std::vector<std::string> missing;
    const auto suite = test::classic_suite(&missing);
    const auto numeric = test::numeric_suite(&missing);
    std::vector<test::PddlTask> tasks(suite.begin(), suite.end());
    tasks.insert(tasks.end(), numeric.begin(), numeric.end());
    ASSERT_FALSE(tasks.empty());
    for (const auto& gt : tasks)
    {
        SCOPED_TRACE(gt.name);
        std::string first;
        std::vector<std::shared_ptr<const frontend::Domain>> keep;  // keep earlier heaps occupied
        for (int rep = 0; rep < 4; ++rep)
        {
            keep.push_back(frontend::Domain::from_file(gt.domain));
            const std::string text = write_task_text(*keep.back()->instantiate_file(gt.problem));
            if (rep == 0)
                first = text;
            else
                ASSERT_EQ(test::tokens(first), test::tokens(text)) << "rep " << rep;
        }
    }
}

TEST(Frontend, ConcurrentInstantiationSharesOneDomain)
{
    std::vector<std::string> missing;
    auto tasks = test::classic_suite(&missing);
    ASSERT_FALSE(tasks.empty());
    // a typed ADL domain with axioms-free problems and a STRIPS one with action costs
    std::vector<test::PddlTask> picked;
    for (const auto& t : tasks)
        if (t.name == "rovers__p02" || t.name == "openstacks-opt08-adl__p03" || t.name == "philosophers__p03-phil4")
            picked.push_back(t);
    ASSERT_FALSE(picked.empty());
    for (const auto& gt : picked)
    {
        SCOPED_TRACE(gt.name);
        const auto domain = frontend::Domain::from_file(gt.domain);
        const std::string expected = write_task_text(*domain->instantiate_file(gt.problem));
        constexpr int kThreads = 16, kReps = 8;
        std::atomic<int> mismatches{0}, errors{0};
        std::vector<std::thread> threads;
        for (int i = 0; i < kThreads; ++i)
            threads.emplace_back(
                [&, i]
                {
                    try
                    {
                        for (int r = 0; r < kReps; ++r)
                        {
                            frontend::InstantiateOptions opt;
                            opt.fast_init = (i + r) % 2 == 0;
                            if (write_task_text(*domain->instantiate_file(gt.problem, opt)) != expected)
                                ++mismatches;
                        }
                    }
                    catch (...)
                    {
                        ++errors;
                    }
                });
        for (auto& t : threads)
            t.join();
        EXPECT_EQ(errors.load(), 0);
        EXPECT_EQ(mismatches.load(), 0);
    }
}

namespace
{
/// A gripper problem with `balls` balls (a PDDL text of megabytes for a few hundred thousand): a very large `:init`.
std::string large_problem(u32 balls)
{
    std::string text = "(define (problem large) (:domain gripper-strips)\n (:objects left right";
    for (u32 i = 0; i < balls; ++i)
        text += " ball" + std::to_string(i);
    text += ")\n (:init (room rooma) (room roomb) (gripper left) (gripper right) (free left) (free right) (at-robby rooma)";
    for (u32 i = 0; i < balls; ++i)
        text += " (ball ball" + std::to_string(i) + ") (at ball" + std::to_string(i) + " rooma)";
    text += ")\n (:goal (and (at ball0 roomb))))\n";
    return text;
}

constexpr u32 kLargeBalls = 40'000;
}  // namespace

TEST(Frontend, LargeProblemInstantiatesQuickly)
{
    const auto domain = frontend::Domain::from_file(test::test_data_dir() / "pddl/gripper/domain.pddl");
    const std::string text = large_problem(kLargeBalls);
    double best = 1e300;
    frontend::TaskPtr task;
    for (int r = 0; r < 3; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        task = domain->instantiate_string(text);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("[large problem] instantiate (fast_init) best of 3: %.3f s; %zu objects, %zu static + %zu fluent initial atoms\n",
                best, task->objects.size(), task->static_init.size(), task->fluent_init.size());
    EXPECT_EQ(task->objects.size(), size_t{kLargeBalls} + 4);
#if defined(NDEBUG) && !defined(MYMYR_SANITIZED)
    EXPECT_LT(best, 2.0);  // about 0.5 s on an idle core; a quadratic init reader needs minutes
#endif
}

TEST(Frontend, FastInitMatchesFullPathOnALargeProblem)
{
    const auto domain = frontend::Domain::from_file(test::test_data_dir() / "pddl/gripper/domain.pddl");
    const std::string text = large_problem(kLargeBalls / 5);
    frontend::InstantiateOptions full;
    full.fast_init = false;
    EXPECT_EQ(test::tokens(write_task_text(*domain->instantiate_string(text))),
              test::tokens(write_task_text(*domain->instantiate_string(text, "", full))));
}

// ------------------------------------------------------------------------------------------------ small inputs

namespace
{
constexpr const char* kDomain = R"((define (domain d)
  (:requirements :strips :typing :equality :negative-preconditions)
  (:types vehicle place - object truck - vehicle)
  (:constants depot - place)
  (:predicates (at ?v - vehicle ?p - place) (road ?a ?b - place) (ready))
  (:action drive
    :parameters (?t - truck ?a ?b - place)
    :precondition (and (at ?t ?a) (road ?a ?b) (ready) (not (= ?a ?b)))
    :effect (and (at ?t ?b) (not (at ?t ?a)))))
)";

std::string problem(const std::string& init)
{
    return "(define (problem p) (:domain d)\n (:objects t1 t2 - truck x y - place)\n (:init " + init
           + ")\n (:goal (and (at t1 y))))";
}
}  // namespace

TEST(Frontend, StringInputAndDomainData)
{
    const auto domain = frontend::Domain::from_string(kDomain, "d.pddl");
    EXPECT_EQ(domain->name(), "d");
    const TaskData& d = domain->domain_data();
    EXPECT_EQ(d.schemas.size(), 1u);
    EXPECT_TRUE(d.static_init.empty() && d.fluent_init.empty());
    EXPECT_EQ(d.objects.size(), 1u);  // the constant

    const std::string init = "(at t1 x) (road x y) (road y depot) (road x y) (ready)";
    const auto fast = domain->instantiate_string(problem(init), "p.pddl");
    frontend::InstantiateOptions full;
    full.fast_init = false;
    const auto slow = domain->instantiate_string(problem(init), "p.pddl", full);
    EXPECT_EQ(write_task_text(*fast), write_task_text(*slow));
    EXPECT_EQ(fast->objects.size(), 5u);
    EXPECT_EQ(fast->fluent_init.size(), 1u);  // (at t1 x)
    // no :init at all
    const auto empty = domain->instantiate_string("(define (problem p) (:domain d) (:objects a - truck) (:goal (and (ready))))");
    EXPECT_TRUE(empty->fluent_init.empty());
}

// loki lists the types in the order of a hash table of type names, which differs between standard libraries; the
// numbering of the types and the order of the type predicates must follow the names only.
TEST(Frontend, TypeOrderFollowsTheTypeNamesOnly)
{
    const std::vector<std::string> declared = {"truck", "plane", "ship", "crate", "depot", "yard", "pier", "gate", "lift", "belt", "arm", "cart"};
    std::string types;
    for (const auto& n : declared)
        types += n + " ";
    const std::string text = "(define (domain ty) (:requirements :strips :typing) (:types " + types + "- object) (:predicates (p ?x - truck)))";
    const auto domain = frontend::Domain::from_string(text, "ty.pddl");
    const auto task = domain->instantiate_string("(define (problem q) (:domain ty) (:objects a - truck) (:init (p a)) (:goal (p a)))");

    std::vector<std::string> sorted = declared;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::string> want_ids = {"object", "number"};
    want_ids.insert(want_ids.end(), sorted.begin(), sorted.end());
    std::vector<std::string> ids;
    for (const auto& ty : task->types)
        ids.emplace_back(task->str(ty.name));
    EXPECT_EQ(ids, want_ids);

    // the type predicates come first, in the reverse of the visiting order of the types
    std::vector<std::string> want_preds(want_ids.rbegin(), want_ids.rend());
    std::vector<std::string> preds;
    for (size_t i = 0; i < want_preds.size(); ++i)
        preds.emplace_back(task->str(task->predicates[i].name));
    EXPECT_EQ(preds, want_preds);
}

TEST(Frontend, StringInputIsPreprocessedLikeFiles)
{
    // PDDL is case-insensitive and ';' starts a comment: the string entry points must read text exactly as the file
    // entry points do (loki's read_file preprocessing), e.g. a problem naming its domain in upper case.
    std::string upper_domain;
    for (const char c : std::string_view(kDomain))
        upper_domain.push_back(static_cast<char>(c >= 'a' && c <= 'z' ? c - 'a' + 'A' : c));
    const auto lower = frontend::Domain::from_string(kDomain, "d.pddl");
    const auto upper = frontend::Domain::from_string("; a comment\n" + upper_domain, "D.pddl");
    EXPECT_EQ(upper->name(), "d");
    const std::string init = "(AT t1 X)\t(road x Y) ; comment (road y depot)\n (road y depot) (READY)";
    std::string text = problem(init);
    text.replace(text.find("(:domain d)"), 11, "(:domain D)");
    for (const bool fast : {true, false})
    {
        frontend::InstantiateOptions o;
        o.fast_init = fast;
        EXPECT_EQ(write_task_text(*upper->instantiate_string(text, "p.pddl", o)),
                  write_task_text(*lower->instantiate_string(problem("(at t1 x) (road x y) (road y depot) (ready)"), "p.pddl", o)));
    }
}

TEST(Frontend, FastInitReportsErrors)
{
    const auto domain = frontend::Domain::from_string(kDomain, "d.pddl");
    using frontend::PddlError;
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1 nowhere)")), PddlError);       // undefined object
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1)")), PddlError);              // arity
    EXPECT_THROW((void) domain->instantiate_string(problem("(at x t1)")), PddlError);            // types
    EXPECT_THROW((void) domain->instantiate_string(problem("(fly t1 x)")), PddlError);           // predicate
    EXPECT_THROW((void) domain->instantiate_string(problem("(not (road x y))")), PddlError);      // negative
    EXPECT_THROW((void) domain->instantiate_string(problem("(at 5 (road x y))")), PddlError);    // timed
    frontend::InstantiateOptions full;
    full.fast_init = false;
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1 nowhere)"), "", full), PddlError);
}

namespace
{
/// The PddlError that `f` throws.
template <class F>
frontend::PddlError error_of(F&& f)
{
    try
    {
        f();
    }
    catch (const frontend::PddlError& e)
    {
        return e;
    }
    ADD_FAILURE() << "no PddlError";
    return frontend::PddlError("");
}

std::string with(std::string text, const std::string& from, const std::string& to)
{
    text.replace(text.find(from), from.size(), to);
    return text;
}
}  // namespace

TEST(Frontend, PddlErrorsNameTheFileLineAndAction)
{
    // an unsupported construct
    auto e = error_of([] { (void) frontend::Domain::from_string(with(kDomain, "(:action", "(:durative-action"), "d.pddl"); });
    EXPECT_EQ(e.path(), "d.pddl");
    EXPECT_EQ(e.line(), 6u);
    EXPECT_EQ(e.message(), "durative actions are not supported (:durative-action drive)");
    EXPECT_STREQ(e.what(), "d.pddl:6: durative actions are not supported (:durative-action drive)");
    // an undefined predicate inside an action
    e = error_of([] { (void) frontend::Domain::from_string(with(kDomain, "(ready) (not", "(parked ?t) (not"), "d.pddl"); });
    EXPECT_EQ(e.line(), 8u);
    EXPECT_EQ(e.action(), "drive");
    EXPECT_EQ(e.message(), "undefined predicate 'parked'");
    EXPECT_STREQ(e.what(), "d.pddl:8: undefined predicate 'parked' (in action drive)");
    // a requirement mymyr does not support, and one PDDL does not define
    e = error_of([] { (void) frontend::Domain::from_string(with(kDomain, ":equality", ":equality :preferences")); });
    EXPECT_EQ(e.path(), "");
    EXPECT_EQ(e.line(), 2u);
    EXPECT_NE(std::string(e.what()).find("line 2: requirement :preferences is not supported"), std::string::npos)
        << e.what();
    e = error_of([] { (void) frontend::Domain::from_string(with(kDomain, ":equality", ":equality :teleportation")); });
    EXPECT_EQ(e.message(), "unknown requirement :teleportation");
    // the problem's :init (the fast reader) and its goal (loki) report the line in the problem file
    const auto domain = frontend::Domain::from_string(kDomain, "d.pddl");
    e = error_of([&] { (void) domain->instantiate_string(problem("(at t1 x)\n (at t2 nowhere)"), "p.pddl"); });
    EXPECT_EQ(e.path(), "p.pddl");
    EXPECT_EQ(e.line(), 4u);
    EXPECT_EQ(e.message(), "undefined object 'nowhere'");
    e = error_of([&] { (void) domain->instantiate_string(with(problem("(at t1 x)"), "(at t1 y)", "(at t1 z)"), "p.pddl"); });
    EXPECT_EQ(e.line(), 4u);
    EXPECT_EQ(e.message(), "undefined object 'z'");
    // trajectory constraints in a problem are refused, not dropped
    e = error_of([&] {
        (void) domain->instantiate_string(with(problem("(at t1 x)"), "\n (:goal", "\n (:constraints (always (ready)))\n (:goal"), "p.pddl");
    });
    EXPECT_EQ(e.line(), 4u);
    EXPECT_EQ(e.message(), "trajectory constraints are not supported (:constraints)");
    // a file that does not exist is a filesystem error, not a PddlError
    EXPECT_THROW((void) frontend::Domain::from_file("/nonexistent/domain.pddl"), std::filesystem::filesystem_error);
}
