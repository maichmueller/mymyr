// Front end against the reference exports: write_task_text(instantiate(...)) of the 22-task suite and the 20 numeric
// tasks and of an IPC sample against the fork's exports, with fast_init on and off (golden.hpp explains the comparison),
// and the comparison itself.

#include "golden.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/frontend/domain.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

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
    if (tasks.empty())
        GTEST_SKIP() << "no " << label << " PDDL found under " << test::work_dir() << " (set MYMYR_PARITY_DATA)";
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
        GTEST_SKIP() << "no tests/parity/data/golden/ipc (tests/parity/tools/make_golden.sh)";
    if (!missing.empty())
        GTEST_SKIP() << "IPC PDDL missing under " << test::fork_data_dir() << " (set MYMYR_PARITY_DATA)";
    run_golden(tasks, missing, "ipc");
}

// The canonical comparison must forgive exactly the address-dependent orders and nothing else.
TEST(GoldenComparator, ForgivesTypeRunsOnlyAndCatchesRealDifferences)
{
    const fs::path g = test::test_data_dir() / "numeric_tasks/suite/rovers__p02.txt";
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
