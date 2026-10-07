// Front end: loki parse + normalize once per Domain, the port of the fork's
// ToMimirStructures per problem, and the one-pass `:init` reader.
//
// Golden tests: write_task_text(instantiate(...)) against the fork's exports of the 22-task suite and the 20 numeric
// tasks, with fast_init on and off (golden.hpp explains the comparison); fast and full paths token-equal; the output
// independent of heap addresses; one Domain shared by 16 threads; rovers p30-hard under 0.3 s.

#include "golden.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

namespace
{
/// The task without numeric content, as the plain lifted format (and exporter) sees it.
TaskData without_numerics(TaskData t)
{
    t.functions.clear();
    t.metric.reset();
    t.auxiliary_initial.reset();
    t.static_values.clear();
    t.fluent_values.clear();
    for (auto& ce : t.conditional_effects)
    {
        ce.numeric_effects = {};
        ce.auxiliary.reset();
        ce.condition.constraints = {};
    }
    for (auto& s : t.schemas)
        s.precondition.constraints = {};
    for (auto& x : t.axioms)
        x.body.constraints = {};
    t.goal.constraints = {};
    return t;
}

struct Outcome
{
    int raw_equal = 0;        // token-equal to the file as written by the fork
    int canonical_equal = 0;  // equal modulo the fork's address-dependent orders (golden.hpp)
    int via_argument_order = 0;  // ... after replaying the fork's argument order of generated axiom predicates
    int compared = 0;
    std::vector<std::string> failures;
};

/// Every argument order of the generated axiom predicates of `t` (all permutations of each, combined; capped).
std::vector<frontend::DomainOptions> argument_orders(const TaskData& t)
{
    std::vector<frontend::DomainOptions> out(1);
    for (u32 p = 0; p < t.predicates.size(); ++p)
    {
        if (!test::is_generated_axiom_predicate(t, p))
            continue;
        std::vector<std::string> names;
        for (const auto& q : TaskData::slice(t.params, t.predicates[p].params))
            names.emplace_back(t.str(q.name));
        std::sort(names.begin(), names.end());
        std::vector<frontend::DomainOptions> next;
        do
            for (const auto& o : out)
            {
                next.push_back(o);
                next.back().generated_argument_order[std::string(t.str(t.predicates[p].name))] = names;
            }
        while (std::next_permutation(names.begin(), names.end()) && next.size() < 5000);
        out = std::move(next);
    }
    out.erase(out.begin());  // the default order (sorted names) was compared already
    return out;
}

/// Compares one task against every golden file of it.
void compare(const test::GoldenTask& gt, const TaskData& ours, bool fast, Outcome& out)
{
    const auto type_preds = test::type_predicates(ours);
    auto project = [](const TaskData& golden, const TaskData& t) { return !golden.has_numerics() && !golden.metric ? without_numerics(t) : t; };
    for (const auto& g : gt.goldens)
    {
        const TaskData golden = read_task_text_file(g.string());
        const TaskData mine = project(golden, ours);
        ++out.compared;
        std::ifstream f(g);
        const std::string file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (test::tokens(file) == test::tokens(write_task_text(mine)))
            ++out.raw_equal;
        const auto a = test::tokens(test::canonical_text(golden, type_preds));
        auto d = test::first_difference(a, test::tokens(test::canonical_text(mine, type_preds)));
        if (!d)
        {
            ++out.canonical_equal;
            continue;
        }
        // the fork's (address-dependent) argument order of loki's generated axiom predicates: replay each candidate
        bool matched = false;
        for (const auto& opt : argument_orders(ours))
        {
            frontend::InstantiateOptions io;
            io.fast_init = fast;
            const auto alt = frontend::Domain::from_file(gt.domain, opt)->instantiate_file(gt.problem, io);
            if (!test::first_difference(a, test::tokens(test::canonical_text(project(golden, *alt), type_preds))))
            {
                matched = true;
                break;
            }
        }
        if (matched)
        {
            ++out.canonical_equal;
            ++out.via_argument_order;
        }
        else
            out.failures.push_back(gt.name + " vs " + g.parent_path().filename().string() + "/" + g.filename().string()
                                   + ": " + *d);
    }
}

void run_golden(const std::vector<test::GoldenTask>& tasks, const std::vector<std::string>& missing, const char* label)
{
    for (const auto& m : missing)
        std::printf("[%s] PDDL missing, skipped: %s\n", label, m.c_str());
    ASSERT_FALSE(tasks.empty()) << "no " << label << " PDDL found under " << test::work_dir();
    std::map<bool, Outcome> outcomes;
    int with_goldens = 0;
    for (const auto& gt : tasks)
    {
        SCOPED_TRACE(gt.name);
        const auto domain = frontend::Domain::from_file(gt.domain);
        std::string texts[2];
        for (const bool fast : {true, false})
        {
            frontend::InstantiateOptions opt;
            opt.fast_init = fast;
            const auto task = domain->instantiate_file(gt.problem, opt);
            ASSERT_NO_THROW(validate(*task));
            texts[fast] = write_task_text(*task);
            compare(gt, *task, fast, outcomes[fast]);
        }
        EXPECT_EQ(test::tokens(texts[0]), test::tokens(texts[1])) << gt.name << ": fast and full :init paths differ";
        with_goldens += !gt.goldens.empty();
    }
    for (const bool fast : {true, false})
    {
        const Outcome& o = outcomes[fast];
        std::printf("[%s] fast_init=%d: %zu tasks (%d with goldens), %d comparisons: canonical-equal %d (%d after replaying "
                    "the fork's generated-predicate argument order), raw token-equal %d\n",
                    label, fast, tasks.size(), with_goldens, o.compared, o.canonical_equal, o.via_argument_order, o.raw_equal);
        for (const auto& f : o.failures)
            ADD_FAILURE() << "fast_init=" << fast << " " << f;
    }
    EXPECT_EQ(with_goldens, static_cast<int>(tasks.size())) << "some tasks have no golden export";
}

std::optional<std::pair<fs::path, fs::path>> rovers_p30_hard()
{
    for (const char* sub : {"ipc/rovers-ipc/test", "ipc/rovers/test"})
    {
        const fs::path dir = test::fork_data_dir() / sub;
        if (fs::exists(dir / "p30-hard.pddl") && fs::exists(dir / "domain.pddl"))
            return std::make_pair(dir / "domain.pddl", dir / "p30-hard.pddl");
    }
    return std::nullopt;
}
}  // namespace

TEST(FrontendGolden, SuiteMatchesForkExports)
{
    std::vector<std::string> missing;
    const auto tasks = test::suite_tasks(&missing);
    run_golden(tasks, missing, "suite");
    EXPECT_EQ(tasks.size(), 22u);
}

TEST(FrontendGolden, NumericMatchesForkExports)
{
    std::vector<std::string> missing;
    const auto tasks = test::numeric_tasks(&missing);
    run_golden(tasks, missing, "numeric");
    EXPECT_EQ(tasks.size(), 20u);
}

TEST(FrontendGolden, IpcSampleMatchesForkExports)
{
    std::vector<std::string> missing;
    const auto tasks = test::ipc_tasks(&missing);
    if (tasks.empty() && missing.empty())
        GTEST_SKIP() << "no tests/data/golden/ipc (tests/data/make_golden.sh)";
    run_golden(tasks, missing, "ipc");
    EXPECT_TRUE(missing.empty()) << "IPC PDDL missing under " << test::fork_data_dir() << " (set MYMYR_FORK_DATA)";
}

// loki's type order depends on heap addresses; two Domain instances in one process see different addresses, so this
// catches any address dependence the canonical type order misses.
TEST(Frontend, OutputIndependentOfHeapAddresses)
{
    std::vector<std::string> missing;
    const auto suite = test::suite_tasks(&missing);
    const auto numeric = test::numeric_tasks(&missing);
    std::vector<test::GoldenTask> tasks(suite.begin(), suite.end());
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
    auto tasks = test::suite_tasks(&missing);
    ASSERT_FALSE(tasks.empty());
    // a typed ADL domain with axioms-free problems and a STRIPS one with action costs
    std::vector<test::GoldenTask> picked;
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

TEST(Frontend, RoversP30HardInstantiatesUnder300ms)
{
    const auto files = rovers_p30_hard();
    if (!files)
        GTEST_SKIP() << "rovers p30-hard not found under " << test::fork_data_dir() << " (set MYMYR_FORK_DATA)";
    const auto domain = frontend::Domain::from_file(files->first);
    double best = 1e300;
    frontend::TaskPtr task;
    for (int r = 0; r < 3; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        task = domain->instantiate_file(files->second);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("[rovers p30-hard] instantiate (fast_init) best of 3: %.3f s; %zu objects, %zu static + %zu fluent initial atoms\n",
                best, task->objects.size(), task->static_init.size(), task->fluent_init.size());
#if defined(NDEBUG) && !defined(MYMYR_SANITIZED)
    EXPECT_LT(best, 0.3);
#endif
}

TEST(Frontend, FastInitMatchesFullPathOnRoversP30Hard)
{
    const auto files = rovers_p30_hard();
    if (!files)
        GTEST_SKIP() << "rovers p30-hard not found under " << test::fork_data_dir();
    const auto domain = frontend::Domain::from_file(files->first);
    frontend::InstantiateOptions full;
    full.fast_init = false;
    EXPECT_EQ(test::tokens(write_task_text(*domain->instantiate_file(files->second))),
              test::tokens(write_task_text(*domain->instantiate_file(files->second, full))));
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
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1 nowhere)")), std::exception);       // undefined object
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1)")), std::exception);              // arity
    EXPECT_THROW((void) domain->instantiate_string(problem("(at x t1)")), std::exception);            // types
    EXPECT_THROW((void) domain->instantiate_string(problem("(fly t1 x)")), std::exception);           // predicate
    EXPECT_THROW((void) domain->instantiate_string(problem("(not (road x y))")), std::exception);      // negative
    EXPECT_THROW((void) domain->instantiate_string(problem("(at 5 (road x y))")), std::exception);    // timed
    frontend::InstantiateOptions full;
    full.fast_init = false;
    EXPECT_THROW((void) domain->instantiate_string(problem("(at t1 nowhere)"), "", full), std::exception);
}

// The canonical comparison must forgive exactly the address-dependent orders and nothing else.
TEST(GoldenComparator, ForgivesTypeRunsOnlyAndCatchesRealDifferences)
{
    const fs::path g = fs::path(MYMYR_TEST_DATA_DIR) / "numeric_tasks/suite/rovers__p02.txt";
    ASSERT_TRUE(fs::exists(g));
    const TaskData t = read_task_text_file(g.string());
    // the text format carries no types: rovers' type predicates are the unary statics named after its types
    std::set<std::string> tp;
    for (const char* n : {"object", "number", "rover", "waypoint", "store", "camera", "mode", "lander", "objective"})
        tp.insert(n);
    const auto base = test::tokens(test::canonical_text(t, tp));
    ASSERT_GE(t.static_init.size(), 3u);

    TaskData u = t;  // the type atoms of object 0, (object o0) (lander o0), in the other order: forgiven
    std::swap(u.static_init[0], u.static_init[1]);
    EXPECT_EQ(test::tokens(test::canonical_text(u, tp)), base);

    u = t;  // atoms of different objects swapped: a difference
    std::swap(u.static_init[1], u.static_init[2]);
    EXPECT_NE(test::tokens(test::canonical_text(u, tp)), base);

    u = t;  // a precondition literal's polarity: a difference
    u.literals[t.schemas[0].precondition.literals.begin].positive ^= true;
    EXPECT_NE(test::tokens(test::canonical_text(u, tp)), base);

    u = t;  // an object of a fluent initial atom: a difference
    auto& o = u.object_ids[u.fluent_init[0].objects.begin];
    o = ObjectId{(o.v + 1) % u.num_objects()};
    EXPECT_NE(test::tokens(test::canonical_text(u, tp)), base);

    u = t;  // two schemas' preconditions' order of non-type literals: a difference
    const auto& pre = t.schemas[0].precondition.literals;
    ASSERT_GE(pre.count, 2u);
    std::swap(u.literals[pre.begin + pre.count - 1], u.literals[pre.begin + pre.count - 2]);
    EXPECT_NE(test::tokens(test::canonical_text(u, tp)), base);
}
