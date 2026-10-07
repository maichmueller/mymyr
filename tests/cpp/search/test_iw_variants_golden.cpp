// The IW family variants equal the fork 0.16.3 on recorded runs (tests/data/iw_variants/fork_golden.json: rows that
// are equal in canonical successor order and quick, run with the fork_iw_variants arguments, the fork's landmark
// graph and goal atom).
// Per row the variant runs through the public API (abstracted_iw, liw, rollout_iw, find_rollouts_parallel,
// atomic_goal_portfolio) and its counts are compared with the fork's: per-pass counts and status (AIW, LIW), every
// Rollout IW counter, per-rollout passes, states, reached atoms, landing states and co-occurrence pairs (parallel
// rollouts), and the portfolio's result, certificate and per-worker counters. Tasks are loaded from PDDL (skipped when
// missing) or, for the numeric ones, from their text export. Env: MYMYR_IW_VARIANTS_GOLDEN (the file), MYMYR_GOLDEN_FILTER.

#include "../frontend/golden.hpp"
#include "../support/json.hpp"

#include "mymyr/formalism/task_data.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/landmarks/fact_landmark_graph.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;
using namespace mymyr::search;
namespace json = mymyr::test::json;

namespace
{
fs::path golden_file()
{
    if (const char* f = std::getenv("MYMYR_IW_VARIANTS_GOLDEN"); f && *f)
        return f;
    return fs::path(MYMYR_TEST_DATA_DIR) / "iw_variants" / "fork_golden.json";
}

std::shared_ptr<const Task> load_task(const json::Value& src)
{
    const std::string kind = src["kind"].str;
    if (kind == "text")
    {
        const fs::path p = fs::path(MYMYR_SOURCE_DIR) / src["path"].str;
        return fs::exists(p) ? Task::from_text_file(p.string()) : nullptr;
    }
    const std::string r = src["root"].str;
    const fs::path root = r == "fork_data" ? test::fork_data_dir() : r == "source" ? fs::path(MYMYR_SOURCE_DIR) : test::work_dir();
    const fs::path d = root / src["domain"].str, p = root / src["problem"].str;
    if (!fs::exists(d) || !fs::exists(p))
        return nullptr;
    return Task::create(*frontend::load_task(d.string(), p.string()));
}

class Names
{
public:
    explicit Names(const Task& task) : m_task(task)
    {
        const formalism::TaskData& t = task.data();
        for (u32 p = 0; p < t.predicates.size(); ++p)
            m_pred.emplace(std::string(t.str(t.predicates[p].name)), p);
        for (u32 o = 0; o < t.objects.size(); ++o)
            m_obj.emplace(std::string(t.str(t.objects[o].name)), o);
    }
    [[nodiscard]] std::optional<CanonicalAtom> parse(const std::string& text) const
    {
        std::string s = text;
        for (char& c : s)
            if (c == '(' || c == ')')
                c = ' ';
        std::istringstream in(s);
        std::string p, o;
        in >> p;
        const auto it = m_pred.find(p);
        if (it == m_pred.end() || m_task.data().predicate(PredicateId{it->second}).kind != formalism::PredKind::Fluent)
            return std::nullopt;
        std::vector<u32> args;
        while (in >> o)
        {
            const auto jt = m_obj.find(o);
            if (jt == m_obj.end())
                return std::nullopt;
            args.push_back(jt->second);
        }
        const CanonicalAtom c = m_task.atoms().layout().encode(it->second, args.data());
        if (c >= m_task.atoms().layout().fluent_count)
            return std::nullopt;
        return c;
    }

private:
    const Task& m_task;
    std::unordered_map<std::string, u32> m_pred, m_obj;
};

struct Args
{
    std::string algo, ordering = "in_order", seeds, orderings;
    u32 k = 1, workers = 4, threads = 1, max_next_layer_states = ~u32{0};
    u64 seed = 0, max_states = ~u64{0};
    bool base = false, preserve_goal = true, lm_disjunctive = false, lm_all_private = false;
};

Args parse_args(const json::Value& list)
{
    Args a;
    for (usize i = 0; i < list.arr.size(); ++i)
    {
        const std::string& x = list.arr[i].str;
        auto next = [&]() -> const std::string& { return list.arr.at(++i).str; };
        if (x == "--algo")
            a.algo = next();
        else if (x == "--k")
            a.k = static_cast<u32>(std::stoul(next()));
        else if (x == "--base")
            a.base = true;
        else if (x == "--no-preserve-goal")
            a.preserve_goal = false;
        else if (x == "--lm-disjunctive")
            a.lm_disjunctive = true;
        else if (x == "--lm-all-private")
            a.lm_all_private = true;
        else if (x == "--ordering")
            a.ordering = next();
        else if (x == "--seed")
            a.seed = std::stoull(next());
        else if (x == "--seeds")
            a.seeds = next();
        else if (x == "--workers")
            a.workers = static_cast<u32>(std::stoul(next()));
        else if (x == "--threads")
            a.threads = static_cast<u32>(std::stoul(next()));
        else if (x == "--orderings")
            a.orderings = next();
        else if (x == "--max-next-layer-states")
            a.max_next_layer_states = static_cast<u32>(std::stoul(next()));
        else if (x == "--max-states")
            a.max_states = std::stoull(next());
        else if (x == "--goal-atom")
            next();  // resolved in the row's goal_atom
        else
            throw std::runtime_error("unknown argument " + x);
    }
    return a;
}

ActionOrdering ordering_of(const std::string& k)
{
    if (k == "randomized")
        return ActionOrdering::Randomized;
    if (k == "dgaf")
        return ActionOrdering::DirectGoalAchieverFirst;
    if (k == "regression")
        return ActionOrdering::GoalRegressionRelevance;
    if (k == "mixed")
        return ActionOrdering::MixedRegressionRandom;
    return ActionOrdering::InOrder;
}

std::string per_pass(std::span<const IwPassStatistics> passes, std::optional<u32> arity = std::nullopt)
{
    std::string s;
    for (const IwPassStatistics& p : passes)
        s += (s.empty() ? "" : ";") + std::to_string(arity ? *arity : p.arity) + ":" + std::to_string(p.expanded) + "/" + std::to_string(p.generated) +
             "/" + std::to_string(p.generated_in_tree);
    return s;
}

/// The fork's status names, per search kind (bench/cpp/mymyr_iw_variants.cpp).
std::string status_name(SearchStatus s, bool ladder)
{
    if (s == SearchStatus::Exhausted)
        return ladder ? "failed" : "exhausted";
    if (s == SearchStatus::Failed)
        return "failed";
    return mimir_status_name(s);
}

u64 num(const json::Value& v) { return static_cast<u64>(v.num); }

void expect_rollout_stats(const RolloutIwStatistics& s, const json::Value& e, const std::string& what)
{
    EXPECT_EQ(s.rollouts, num(e["rollouts"])) << what;
    EXPECT_EQ(s.generated, num(e["generated"])) << what;
    EXPECT_EQ(s.expanded, num(e["expanded"])) << what;
    EXPECT_EQ(s.feature_depth_improvements, num(e["feature_depth_improvements"])) << what;
    EXPECT_EQ(s.case1, num(e["case1"])) << what;
    EXPECT_EQ(s.case2, num(e["case2"])) << what;
    EXPECT_EQ(s.case3, num(e["case3"])) << what;
    EXPECT_EQ(s.case4, num(e["case4"])) << what;
    EXPECT_EQ(s.solved_propagations, num(e["solved_propagations"])) << what;
    EXPECT_EQ(s.dead_ends, num(e["dead_ends"])) << what;
    EXPECT_EQ(s.depth_bound_prunings, num(e["depth_bound_prunings"])) << what;
    EXPECT_EQ(s.incumbent_bound_prunings, num(e["incumbent_bound_prunings"])) << what;
    EXPECT_EQ(s.max_rollout_depth, num(e["max_rollout_depth"])) << what;
    EXPECT_EQ(s.tree_nodes, num(e["tree_nodes"])) << what;
}

void run_row(const Task& task, const json::Value& row)
{
    const std::string what = row["task"].str + " " + row["config"].str;
    const Args a = parse_args(row["args"]);
    const json::Value& e = row["expect"];
    const Names names(task);
    SearchControl control;
    if (row.has("goal_atom"))
    {
        const auto c = names.parse(row["goal_atom"].str);
        ASSERT_TRUE(c) << what << ": unknown goal atom " << row["goal_atom"].str;
        control.goal.kind = GoalSpec::Kind::AnyOf;
        control.goal.goals.push_back({.positive = {SlotId{task.atoms().intern(*c)}}});
    }
    LandmarkNovelty lm;
    if (row.has("landmarks"))
    {
        std::vector<CanonicalAtom> facts;
        std::vector<std::vector<CanonicalAtom>> sets;
        for (const json::Value& x : row["landmarks"]["facts"].arr)
            if (const auto c = names.parse(x.str))
                facts.push_back(*c);
        for (const json::Value& set : row["landmarks"]["sets"].arr)
        {
            sets.emplace_back();
            for (const json::Value& x : set.arr)
                if (const auto c = names.parse(x.str))
                    sets.back().push_back(*c);
        }
        lm.graph = std::make_shared<const landmarks::FactLandmarkGraph>(landmarks::FactLandmarkGraph::create(std::move(facts), std::move(sets)));
        lm.disjunctive = a.lm_disjunctive;
        lm.all_private = a.lm_all_private;
    }
    if (a.algo == "aiw" || a.algo == "liw")
    {
        IwResult r;
        if (a.algo == "aiw")
        {
            AbstractedIwOptions o;
            o.control = control;
            o.width = a.k;
            o.base_abstracted = a.base;
            o.preserve_goal_atoms = a.preserve_goal;
            o.landmarks = lm;
            r = abstracted_iw(task, o);
            EXPECT_EQ(per_pass(r.passes, a.k), e["per_pass"].str) << what;
        }
        else
        {
            LiwOptions o;
            o.control = control;
            o.max_arity = a.k;
            o.landmarks = lm;
            r = liw(task, o);
            EXPECT_EQ(per_pass(r.passes), e["per_pass"].str) << what;
        }
        EXPECT_EQ(status_name(r.status, a.algo == "liw"), e["status"].str) << what;
        if (r.status == SearchStatus::Solved)
        {
            EXPECT_EQ(r.plan.size(), num(e["plan_len"])) << what;
        }
    }
    else if (a.algo == "rollout_iw")
    {
        RolloutIwOptions o;
        o.control = control;
        o.control.budget.max_states = a.max_states;
        o.ordering = ordering_of(a.ordering);
        o.seed = a.seed;
        const RolloutIwResult r = rollout_iw(task, o);
        EXPECT_EQ(status_name(r.status, false), e["status"].str) << what;
        expect_rollout_stats(r.statistics, e["rollout"], what);
        EXPECT_EQ(r.root_solved, e["root_solved"].b) << what;
        if (r.status == SearchStatus::Solved)
        {
            EXPECT_EQ(r.plan.size(), num(e["plan_len"])) << what;
        }
    }
    else if (a.algo == "rollouts")
    {
        ParallelRolloutOptions o;
        o.iw.control = control;
        o.iw.max_arity = a.k;
        o.num_threads = 4;  // the fork ran them serially: the result does not depend on the thread count
        o.max_next_layer_states = a.max_next_layer_states;
        o.report_landing_states = true;
        o.report_co_occurrence = true;
        std::istringstream in(a.seeds);
        std::string s;
        while (std::getline(in, s, ','))
            o.seeds.push_back(std::stoull(s));
        const ParallelRolloutsResult r = find_rollouts_parallel(task, o);
        ASSERT_EQ(r.rollouts.size(), e["rollouts"].arr.size()) << what;
        for (usize i = 0; i < r.rollouts.size(); ++i)
        {
            const RolloutResult& x = r.rollouts[i];
            const json::Value& y = e["rollouts"].arr[i];
            const std::string w = what + " seed " + std::to_string(num(y["seed"]));
            EXPECT_EQ(status_name(x.search.status, true), y["status"].str) << w;
            EXPECT_EQ(per_pass(x.search.passes), y["per_pass"].str) << w;
            if (x.search.status == SearchStatus::Solved)
            {
                EXPECT_EQ(x.search.plan.size(), num(y["plan_len"])) << w;
            }
            EXPECT_EQ(x.num_states, num(y["num_states"])) << w;
            EXPECT_EQ(x.reached_fluent_atoms.size(), num(y["reached_fluent"])) << w;
            EXPECT_EQ(x.reached_derived_atoms.size(), num(y["reached_derived"])) << w;
            EXPECT_EQ(x.landing_states.size(), num(y["landing_states"])) << w;
            EXPECT_EQ(static_cast<u64>(std::count_if(x.landing_states.begin(), x.landing_states.end(), [](const LandingState& l) { return l.direct_dead_end; })),
                      num(y["landing_dead_ends"]))
                << w;
            u64 pairs = 0;
            for (const auto& [atom, row_atoms] : x.co_occurrence)
                pairs += row_atoms.size();
            EXPECT_EQ(pairs, num(y["co_occurrence_pairs"])) << w;
        }
    }
    else if (a.algo == "portfolio")
    {
        PortfolioOptions o;
        o.control = control;
        o.num_rollout_workers = a.workers;
        o.num_threads = a.threads;
        std::istringstream in(a.orderings);
        std::string s;
        while (std::getline(in, s, ','))
        {
            const auto colon = s.find(':');
            o.rollout_orderings.push_back({ordering_of(s.substr(0, colon)), colon == std::string::npos ? 0 : std::stoull(s.substr(colon + 1))});
        }
        const PortfolioResult r = atomic_goal_portfolio(task, o);
        EXPECT_EQ(status_name(r.status, true), e["status"].str) << what;
        EXPECT_EQ(r.status == SearchStatus::Solved ? r.plan_length : 0u, num(e["plan_length"])) << what;
        EXPECT_EQ(r.certified_optimal, e["certified_optimal"].b) << what;
        EXPECT_EQ(r.winning_worker, num(e["winning_worker"])) << what;
        EXPECT_EQ(r.certifier_ran ? status_name(r.certifier_status, true) : std::string("in_progress"), e["certifier_status"].str) << what;
        EXPECT_EQ(r.iw_completed_depth, num(e["iw_completed_depth"])) << what;
        EXPECT_EQ(r.iw_lower_bound, num(e["iw_lower_bound"])) << what;
        EXPECT_EQ(r.total_expansions, num(e["total_expansions"])) << what;
        EXPECT_EQ(r.certifier.total.expanded, num(e["expanded"])) << what;
        EXPECT_EQ(r.certifier.total.generated, num(e["generated"])) << what;
        ASSERT_EQ(r.rollout_statistics.size(), e["rollout_statistics"].arr.size()) << what;
        for (usize i = 0; i < r.rollout_statistics.size(); ++i)
        {
            expect_rollout_stats(r.rollout_statistics[i], e["rollout_statistics"].arr[i], what + " worker " + std::to_string(i + 1));
            EXPECT_EQ(r.rollout_rounds[i] ? status_name(r.rollout_statuses[i], false) : std::string("in_progress"), e["rollout_statuses"].arr[i].str)
                << what << " worker " << i + 1;
        }
    }
    else
        FAIL() << what << ": unknown algo " << a.algo;
}
}  // namespace

TEST(IwVariantGolden, EqualTheFork)
{
    const fs::path f = golden_file();
    ASSERT_TRUE(fs::exists(f)) << f;
    const json::Value doc = json::parse_file(f.string());
    const char* filter = std::getenv("MYMYR_GOLDEN_FILTER");
    std::map<std::string, std::shared_ptr<const Task>> tasks;
    u32 run = 0, missing = 0;
    for (const json::Value& row : doc["rows"].arr)
    {
        const std::string name = row["task"].str;
        if (filter && *filter && (name + " " + row["config"].str).find(filter) == std::string::npos)
            continue;
        auto it = tasks.find(name);
        if (it == tasks.end())
            it = tasks.emplace(name, load_task(row["source"])).first;
        if (!it->second)
        {
            ++missing;
            continue;
        }
        run_row(*it->second, row);
        ++run;
    }
    std::printf("IW variants golden: %u rows run, %u without their task files\n", run, missing);
    if (run == 0)
        GTEST_SKIP() << "no task files";
}
