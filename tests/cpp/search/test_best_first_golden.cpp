// A* plan costs are optimal and equal the fork's (tests/data/expected/*.json, `astar.optimal_cost`, written
// by the fork 0.16.3's astar_eager with the golden file's heuristic: blind with action costs, h_max otherwise) and
// 0.13.60's A* results on the same instances. Every task is loaded from its PDDL.
//
// Per golden task: astar_eager and astar_lazy with the fork's heuristic (unit-cost h, the fork's convention; blind on
// tasks with action costs) and, on tasks with action costs, A* with real-cost h_max (admissible there, unlike the
// unit-cost one when actions cost 0). Every plan is replayed: applicable, reaching the goal, costing what the search
// reports. Env: MYMYR_GOLDEN_DIR (default <source>/tests/data/expected; skipped when missing), MYMYR_GOLDEN_FILTER,
// MYMYR_GOLDEN_ASTAR_MAX_EXPANDED (skip tasks whose fork run expanded more; default 50000 in sanitizer builds).
// Also: a goal with a false static literal gives Unsolvable, and GBFS/beam plan costs take the cheapest action per
// step (the fork's plan extraction).

#include "../frontend/golden.hpp"
#include "../support/json.hpp"

#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;
using namespace mymyr::search;

namespace
{
using SearchFn = BestFirstResult (*)(const Task&, const BestFirstOptions&);

fs::path golden_dir()
{
    if (const char* d = std::getenv("MYMYR_GOLDEN_DIR"); d && *d)
        return d;
    return fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "expected";
}

bool pddl_of(const test::json::Value& src, fs::path& domain, fs::path& problem)
{
    const std::string tag = src["tag"].str;
    fs::path base;
    if (tag.rfind("ipc/", 0) == 0)
        base = test::fork_data_dir() / "ipc" / tag.substr(4) / "test";
    else if (tag == "adl/philosophers")
        base = fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "pddl" / "philosophers";
    else
        base = test::work_dir() / "mimir-cs" / "Benchmark" / tag;
    domain = base / src["domain_file"].str;
    problem = base / src["problem"].str;
    return fs::exists(domain) && fs::exists(problem);
}

u64 max_expanded_limit()
{
    if (const char* e = std::getenv("MYMYR_GOLDEN_ASTAR_MAX_EXPANDED"); e && *e)
        return std::stoull(e);
#if defined(MYMYR_SANITIZED)
    return 50000;
#else
    return ~u64{0};
#endif
}

/// Replays the plan: every action applicable, the goal reached; returns the cost under ActionCosts (-1 if invalid).
double replay_cost(const Task& task, const std::vector<Action>& plan)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    const heuristics::ActionCosts costs(task);
    State s = task.initial_state();
    double g = costs.initial();
    for (const Action& a : plan)
    {
        if (!succ.is_applicable(s, a.label()))
            return -1;
        g += costs.cost(a.schema.v, a.binding.data());
        s = succ.apply(s, a.label());
    }
    return succ.is_goal(s) ? g : -1;
}

double seconds(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

BestFirstOptions astar_options(heuristics::Kind k, heuristics::Costs costs)
{
    BestFirstOptions o;
    o.heuristic.kind = k;
    o.heuristic.costs = costs;
    o.control.budget.max_states = 3'000'000;  // the golden budget
    o.control.budget.max_seconds = 300;
    return o;
}
}  // namespace

TEST(AStarGolden, OptimalCostsEqualTheForks)
{
    const fs::path dir = golden_dir();
    if (!fs::is_directory(dir))
        GTEST_SKIP() << "no golden directory " << dir;
    const char* filter = std::getenv("MYMYR_GOLDEN_FILTER");
    const u64 limit = max_expanded_limit();
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".json")
            files.push_back(e.path());
    std::sort(files.begin(), files.end());
    u32 checked = 0, skipped = 0;
    for (const fs::path& f : files)
    {
        const std::string name = f.stem().string();
        if (filter && *filter && name.find(filter) == std::string::npos)
            continue;
        const test::json::Value doc = test::json::parse_file(f);
        const test::json::Value& a = doc["astar"];
        if (a.is_null() || a["optimal_cost"].is_null())
            continue;
        if (static_cast<u64>(a["expanded"].num) > limit)
        {
            ++skipped;
            continue;
        }
        fs::path dom, prob;
        if (!pddl_of(doc["source"], dom, prob))
        {
            ++skipped;
            continue;
        }
        SCOPED_TRACE(name);
        const auto data = frontend::load_task(dom, prob);
        const auto task = Task::create(*data);
        const double expected = a["optimal_cost"].num;
        const heuristics::Kind k = a["heuristic"].str == "blind" ? heuristics::Kind::Blind : heuristics::Kind::Max;
        const heuristics::ActionCosts costs(*task);
        std::string line;
        char buf[256];
        std::snprintf(buf, sizeof buf, "GOLDEN-ASTAR %-40s %-5s cost %8.0f | fork expanded %8llu |", name.c_str(),
                      a["heuristic"].str.c_str(), expected, static_cast<unsigned long long>(a["expanded"].num));
        line = buf;
        struct Run
        {
            const char* label;
            SearchFn run;
            heuristics::Kind kind;
            heuristics::Costs costs;
        };
        std::vector<Run> runs = {{"eager", &astar_eager, k, heuristics::Costs::Unit}, {"lazy", &astar_lazy, k, heuristics::Costs::Unit}};
        if (!costs.unit())
            runs.push_back({"eager-hmax-real", &astar_eager, heuristics::Kind::Max, heuristics::Costs::Real});
        for (const Run& r : runs)
        {
            SCOPED_TRACE(r.label);
            const auto t0 = std::chrono::steady_clock::now();
            const BestFirstResult res = r.run(*task, astar_options(r.kind, r.costs));
            const double s = seconds(t0);
            ASSERT_EQ(res.status, SearchStatus::Solved) << to_string(res.status) << " " << res.message;
            EXPECT_EQ(res.cost, expected);
            EXPECT_EQ(replay_cost(*task, res.plan), res.cost);
            std::snprintf(buf, sizeof buf, " %s %llu exp %.2fs", r.label, static_cast<unsigned long long>(res.stats.expanded), s);
            line += buf;
        }
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        ++checked;
    }
    std::printf("GOLDEN-ASTAR checked %u tasks, skipped %u (budget or missing PDDL)\n", checked, skipped);
}

TEST(AStarProbe, CostsEqual0_13_60)
{
    // 0.13.60's plan costs on the same instances
    struct Row
    {
        const char* dir;
        const char* problem;
        heuristics::Kind kind;
        double cost;
    };
    const Row rows[] = {
        {"strips/blocks", "probBLOCKS-8-0", heuristics::Kind::Blind, 18},
        {"strips/driverlog", "p03", heuristics::Kind::Blind, 12},
        {"strips/miconic", "s7-4", heuristics::Kind::Blind, 25},
        {"strips/zenotravel", "p05", heuristics::Kind::Blind, 11},
        {"strips/blocks", "probBLOCKS-8-0", heuristics::Kind::Max, 18},
        {"strips/freecell", "p02", heuristics::Kind::Max, 14},
        {"strips/gripper", "prob05", heuristics::Kind::Max, 35},
        {"strips/miconic", "s7-4", heuristics::Kind::Max, 25},
        {"data/sokoban", "p68", heuristics::Kind::Max, 34},
        {"strips/transport-opt08-strips", "p23", heuristics::Kind::Max, 630},
        {"strips/zenotravel", "p05", heuristics::Kind::Max, 11},
    };
    const u64 limit = max_expanded_limit();
    for (const Row& row : rows)
    {
        const fs::path base = std::string(row.dir) == "data/sokoban" ? test::work_dir() / "mimir" / "data" / "sokoban"
                                                                     : test::work_dir() / "mimir-cs" / "Benchmark" / row.dir;
        const fs::path dom = base / "domain.pddl", prob = base / (std::string(row.problem) + ".pddl");
        if (!fs::exists(dom) || !fs::exists(prob))
            continue;
        SCOPED_TRACE(std::string(row.dir) + "/" + row.problem + "/" + heuristics::to_string(row.kind));
        const auto data = frontend::load_task(dom, prob);
        const auto task = Task::create(*data);
        for (const heuristics::Costs hc : {heuristics::Costs::Unit, heuristics::Costs::Real})
        {
            if (row.kind == heuristics::Kind::Blind && hc == heuristics::Costs::Real)
                continue;
            BestFirstOptions o = astar_options(row.kind, hc);
            if (limit != ~u64{0})
                o.control.budget.max_expanded = limit * 4;
            const BestFirstResult r = astar_eager(*task, o);
            if (r.status == SearchStatus::OutOfStates && limit != ~u64{0})
                continue;  // sanitizer builds: over the budget
            ASSERT_EQ(r.status, SearchStatus::Solved) << r.message;
            EXPECT_EQ(r.cost, row.cost);
            EXPECT_EQ(replay_cost(*task, r.plan), r.cost);
            std::printf("PROBE-ASTAR %s/%s %s %s cost %.0f expanded %llu search %.3fs\n", row.dir, row.problem,
                        heuristics::to_string(row.kind), hc == heuristics::Costs::Real ? "real" : "unit", r.cost,
                        static_cast<unsigned long long>(r.stats.expanded), r.stats.seconds);
        }
    }
}

TEST(BestFirstPddl, FalseStaticGoalIsUnsolvable)
{
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d)
  (:requirements :strips)
  (:predicates (adj ?a ?b) (at ?a))
  (:action move :parameters (?a ?b) :precondition (and (at ?a) (adj ?a ?b)) :effect (and (at ?b) (not (at ?a)))))
)");
    const auto data = domain->instantiate_string(R"(
(define (problem p) (:domain d) (:objects a b c)
  (:init (at a) (adj a b) (adj b c))
  (:goal (and (at c) (adj c a))))
)");
    const auto task = Task::create(*data);
    for (SearchFn f : {&astar_eager, &astar_lazy, &gbfs_eager, &gbfs_lazy, &beam})
        EXPECT_EQ(f(*task, {}).status, SearchStatus::Unsolvable);
    // without the false static literal it is solvable in two steps
    const auto ok = Task::create(*domain->instantiate_string(R"(
(define (problem p) (:domain d) (:objects a b c)
  (:init (at a) (adj a b) (adj b c))
  (:goal (at c)))
)"));
    for (SearchFn f : {&astar_eager, &astar_lazy, &gbfs_eager, &gbfs_lazy, &beam})
    {
        const BestFirstResult r = f(*ok, {});
        EXPECT_EQ(r.status, SearchStatus::Solved);
        EXPECT_EQ(r.plan.size(), 2u);
    }
}

TEST(BestFirstPddl, PlanCostTakesTheCheapestActionPerStep)
{
    // Two actions lead from a to b; GBFS generates the expensive one first (canonical order: schema order).
    const auto domain = frontend::Domain::from_string(R"(
(define (domain d)
  (:requirements :strips :action-costs)
  (:predicates (at ?a) (adj ?a ?b))
  (:functions (total-cost) - number)
  (:action slow :parameters (?a ?b) :precondition (and (at ?a) (adj ?a ?b))
    :effect (and (at ?b) (not (at ?a)) (increase (total-cost) 10)))
  (:action fast :parameters (?a ?b) :precondition (and (at ?a) (adj ?a ?b))
    :effect (and (at ?b) (not (at ?a)) (increase (total-cost) 1))))
)");
    const auto task = Task::create(*domain->instantiate_string(R"(
(define (problem p) (:domain d) (:objects a b)
  (:init (at a) (adj a b) (= (total-cost) 0))
  (:goal (at b)) (:metric minimize (total-cost)))
)"));
    for (SearchFn f : {&astar_eager, &astar_lazy, &gbfs_eager, &gbfs_lazy, &beam})
    {
        BestFirstOptions o;
        o.heuristic.kind = heuristics::Kind::Blind;
        const BestFirstResult r = f(*task, o);
        ASSERT_EQ(r.status, SearchStatus::Solved);
        EXPECT_EQ(r.cost, 1.0);
        ASSERT_EQ(r.plan.size(), 1u);
        EXPECT_EQ(task->format(r.plan[0].label()).rfind("(fast", 0), 0u) << task->format(r.plan[0].label());
    }
}
