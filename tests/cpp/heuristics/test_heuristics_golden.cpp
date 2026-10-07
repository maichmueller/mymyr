// Heuristic values equal the fork's (tests/data/expected/*.json, written by the fork 0.16.3's
// tests/data/fork_golden/export_golden.cpp on the golden-fork branch).
//
// Every golden task is loaded from its PDDL with mymyr's front end; the three seeded random walks of the golden file are
// replayed by the names of the actions taken (checking the fluent atom set of every step by count and hash) and in every
// step blind, goal count, h_max, h_add and h_FF are compared with the fork's values, grounded and (unless
// MYMYR_GOLDEN_NO_LIFTED is set; sanitizer builds: tasks up to MYMYR_GOLDEN_LIFTED_MAX_OPS = 2000 relaxed operators)
// with the lifted fallback. The golden directory is MYMYR_GOLDEN_DIR, else
// <source>/tests/data/expected; the test is skipped when it is missing. MYMYR_GOLDEN_FILTER restricts the tasks to those
// whose name contains the given text.
//
// h_max, h_add, goal count and blind must match everywhere. h_FF may differ: among equally cheap supporters the fork
// takes the first to fire in its Dijkstra, whose order among equal costs is its binary heap over its own atom and
// unary-action numbering. With MYMYR_FORK_PLANS pointing at the fork's relaxed plans along
// the same walks and their supporters, every differing state must be
// explained: each supporter the fork used must be a cheapest (h_max) supporter in mymyr's relaxation, and they must make
// up the fork's plan (ForkPlanCheck). The fork's own h_FF is not stable across fork runs where its atom numbering depends
// on heap addresses (rubiks-cube); a state is then checked against the plan of the run that wrote the plans. Without the
// plans the test only counts h_FF differences. It always checks the invariants of relaxed
// plans: h_FF finite iff h_max is finite, and at least h_max without conditional effects (with them the fork counts a
// ground action once however many of its effect instances a plan uses, so h_FF < h_max happens, 6 golden states).
//
// Set-additive, h² and the perfect heuristic are compared with the fork's values on the same walk states of the BrFS
// suite (tests/data/heuristics/fork_heuristics.json, written by search_fork/run_heuristics.py; each step carries the
// fluent atom count and hash of the fork's state, checked against the golden walk). Perfect must match everywhere (on
// tasks whose state space the fork could build; sanitizer builds: spaces of at most 100,000 states). Set-additive must
// match up to the fork's ties among equally cheap supporters (as h_FF), counted. h² differs from the fork's by design
// (heuristic.hpp): on tasks without negative literals, conditional effects and axioms the fork's value must equal
// mimir's definition evaluated on mymyr's grounding (test::ReferenceH2 with mimir's delete check), which can only be
// lower than mymyr's; elsewhere mymyr's h² must lie between h_max and h* (when known). A* with the perfect heuristic
// finds the fork's optimal cost and, with unit costs, expands exactly the states of its plan but the goal. The fork's
// perfect heuristic is the cost-goal distance: mymyr's with real costs.

#include "../frontend/golden.hpp"
#include "../support/json.hpp"
#include "h2_reference.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/perfect.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;

namespace
{
fs::path golden_dir()
{
    if (const char* d = std::getenv("MYMYR_GOLDEN_DIR"); d && *d)
        return d;
    return fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "expected";
}

/// PDDL of a golden task (export_all.py's resolution: bench.py tags under the C# benchmark set, the fork's IPC data,
/// the patched philosophers domain).
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

std::string hex16(u64 v)
{
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

u64 set_hash(const std::vector<std::string>& items)
{
    u64 sum = 0;
    for (const std::string& s : items)
    {
        u64 h = 0xcbf29ce484222325ULL;
        for (unsigned char c : s)
        {
            h ^= c;
            h *= 0x100000001b3ULL;
        }
        sum += h;
    }
    return sum;
}

double golden_h(const test::json::Value& v) { return v.is_null() ? heuristics::k_dead_end : v.num; }

std::string show(double v) { return std::isinf(v) ? std::string("inf") : std::to_string(static_cast<long long>(v)); }

struct Tally
{
    u64 states = 0, hmax = 0, hadd = 0, hff = 0, gc = 0, blind = 0, goal = 0, ff_invariant = 0;
    u64 ff_checked = 0, ff_explained = 0, ff_higher = 0, ff_lower = 0, ff_fork_varies = 0;
    u64 ff_observed = 0;       // steps where the fork's h_FF varies between processes (golden `hff_observed`)
    u64 ff_observed_hit = 0;   // ... and mymyr's value is one the fork produced
    u64 ff_below_hmax = 0;     // steps where the fork's h_FF is below its h_max (conditional effects)
    u64 lifted_hmax = 0, lifted_hadd = 0, lifted_ff_invariant = 0;
    double ff_absdiff = 0;
    // set-additive, h², perfect against fork_heuristics.json
    u64 sa_states = 0, sa_equal = 0, sa_higher = 0, sa_lower = 0;
    u64 h2_states = 0, h2_equal = 0, h2_higher = 0, h2_comparable = 0, h2_mimir_equal = 0, h2_bad = 0;
    u64 h2_fork_inadmissible = 0;  // the fork's h² above the unit-cost h*
    u64 hstar_states = 0, hstar_diff = 0, misaligned = 0;
};

/// The fork's set-additive, h² and perfect values on the golden walks (search_fork/run_heuristics.py).
const test::json::Value& fork_heuristics()
{
    static const test::json::Value doc = []
    {
        const fs::path f = fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "heuristics" / "fork_heuristics.json";
        return fs::exists(f) ? test::json::parse_file(f.string()) : test::json::Value{};
    }();
    return doc;
}

/// The fork's h_FF of one walk state with its relaxed plan and supporters (fork_heur --mode walks).
struct ForkFF
{
    double hff = 0;
    std::vector<std::string> plan;
    struct Row
    {
        std::string prop, action, effect;  // proposition ("not " / "D" prefixes), achieving ground action, its effect atom
    };
    std::vector<Row> achievers;
};

std::map<std::pair<u64, u64>, ForkFF> read_fork_plans(const fs::path& file)
{
    std::map<std::pair<u64, u64>, ForkFF> out;
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("STEP ", 0) != 0)
            continue;
        const auto v = test::json::parse(std::string_view(line).substr(5));
        ForkFF& f = out[{static_cast<u64>(v["w"].num), static_cast<u64>(v["i"].num)}];
        f.hff = golden_h(v["hff"]);
        for (const auto& a : v["plan"].arr)
            f.plan.push_back(a.str);
        for (const auto& r : v["achievers"].arr)
            f.achievers.push_back({r[0].str, r[1].str, r[2].str});
    }
    return out;
}

/// Checks that the fork's relaxed plan is a best-supporter relaxed plan of mymyr's relaxation, i.e. that an h_FF
/// difference is a tie-break: every supporter the fork used (proposition, ground action, the effect atom of the fork's
/// unary action) is an operator of mymyr's grounding that reaches the proposition at its h_max cost (derived
/// propositions: the forwarded supporter reaches the derived atom's cost), every proposition of positive cost the
/// extraction needs (the goal and the supporters' preconditions) has such a supporter, and the plan is exactly the set of
/// supporting ground actions. Returns an empty string when it holds, else the first violation.
class ForkPlanCheck
{
public:
    ForkPlanCheck(const Task& task, const heuristics::RelaxedTask& R) : T(task), R(R), m_ops_of_ga(R.num_ground_actions())
    {
        for (u32 o = 0; o < R.num_ops(); ++o)
            if (!R.is_axiom(o))
                m_ops_of_ga[R.ground_action(o)].push_back(o);
        for (u32 g = 0; g < R.num_ground_actions(); ++g)
            m_ga.emplace(task.format(R.ground_action_label(g)), g);
        const auto& D = task.data();
        for (u32 p = 0; p < D.predicates.size(); ++p)
            m_pred.emplace(std::string(D.str(D.predicates[p].name)), p);
        for (u32 o = 0; o < D.num_objects(); ++o)
            m_obj.emplace(task.object_name(ObjectId{o}), o);
    }

    std::string check(StateView s, const ForkFF& f)
    {
        costs(s);
        // the fork's supporters by proposition
        std::unordered_map<u32, u32> supp;
        std::set<u32> plan;
        for (const auto& r : f.achievers)
        {
            const u32 x = prop(r.prop), e = prop(r.effect);
            const auto g = m_ga.find(r.action);
            if (x == k_none || e == k_none || g == m_ga.end())
                return "a supporter of the fork is not in mymyr's relaxation: " + r.prop + " <- " + r.action;
            u32 op = k_none;
            for (u32 o : m_ops_of_ga[g->second])
                if (std::find(R.eff(o).begin(), R.eff(o).end(), e) != R.eff(o).end())
                {
                    op = o;
                    if (level(o) + 1 == m_cost[x])
                        break;
                }
            if (op == k_none)
                return "no operator of " + r.action + " adds " + r.effect;
            if (level(op) == k_inf || level(op) + 1 != m_cost[x])
                return r.prop + " <- " + r.action + " is not a cheapest supporter (h_max " + std::to_string(m_cost[x]) + ")";
            supp[x] = op;
            plan.insert(g->second);
        }
        // the extraction: every needed proposition of positive cost has a supporter, and they make up the plan
        std::set<u32> used;
        std::vector<u32> stack(R.goal().begin(), R.goal().end());
        std::vector<u8> seen(R.num_props(), 0);
        while (!stack.empty())
        {
            const u32 x = stack.back();
            stack.pop_back();
            if (seen[x] || m_cost[x] == 0)
                continue;
            seen[x] = 1;
            const auto it = supp.find(x);
            if (it == supp.end())
                return "the fork's extraction has no supporter for a needed proposition";
            used.insert(R.ground_action(it->second));
            for (u32 y : R.pre(it->second))
                stack.push_back(y);
        }
        if (used != plan)
            return "the fork's supporters do not form its plan";
        if (static_cast<double>(plan.size()) != f.hff || plan.size() != f.plan.size())
            return "plan size differs from the fork's h_FF";
        return {};
    }

private:
    static constexpr u32 k_inf = ~u32{0}, k_none = ~u32{0};

    /// A proposition by the fork's name: "(p o1 ...)", "not (p ...)", "D(p ...)".
    u32 prop(std::string_view n) const
    {
        bool neg = false;
        if (n.rfind("not ", 0) == 0)
        {
            neg = true;
            n.remove_prefix(4);
        }
        if (!n.empty() && n[0] == 'D')
            n.remove_prefix(1);
        if (n.size() < 2 || n.front() != '(' || n.back() != ')')
            return k_none;
        n = n.substr(1, n.size() - 2);
        std::vector<std::string> tok;
        for (usize i = 0; i < n.size();)
        {
            const usize j = std::min(n.find(' ', i), n.size());
            tok.emplace_back(n.substr(i, j - i));
            i = j + 1;
        }
        const auto p = m_pred.find(tok[0]);
        if (p == m_pred.end())
            return k_none;
        std::vector<u32> args;
        for (usize i = 1; i < tok.size(); ++i)
        {
            const auto o = m_obj.find(tok[i]);
            if (o == m_obj.end())
                return k_none;
            args.push_back(o->second);
        }
        const CanonicalLayout& L = T.atoms().layout();
        if (!L.has_predicate(p->second) || args.size() != L.arity[p->second])
            return k_none;
        const CanonicalAtom c = L.encode(p->second, args.data());
        if (c >= L.total)
            return k_none;
        const auto [pos, negp] = R.fluent_props(c);
        return neg ? negp : pos;
    }

    [[nodiscard]] u32 level(u32 o) const
    {
        u32 a = 0;
        for (u32 p : R.pre(o))
        {
            if (m_cost[p] == k_inf)
                return k_inf;
            a = std::max(a, m_cost[p]);
        }
        return a;
    }

    /// h_max costs from s (unit action costs), with the fork's initialization.
    void costs(StateView s)
    {
        const u32 P = R.num_props(), O = R.num_ops();
        std::vector<u8> in_s(P, 0);
        bits::for_each(s.w, s.nw,
                       [&](u64 slot)
                       {
                           const auto [pp, np] = R.fluent_props(T.atoms().canonical(AtomKind::Fluent, static_cast<u32>(slot)));
                           if (pp != k_none)
                               in_s[pp] = 1;
                           if (np != k_none)
                               in_s[np] = 2;
                       });
        m_cost.assign(P, k_inf);
        std::vector<u32> cnt(O), acc(O, 0);
        using Entry = std::pair<u32, u32>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<>> q;
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p) ? in_s[p] != 2 : in_s[p] == 1)
            {
                m_cost[p] = 0;
                q.emplace(0, p);
            }
        auto fire = [&](u32 o)
        {
            const u32 val = acc[o] + (R.is_axiom(o) ? 0 : 1);
            for (u32 x : R.eff(o))
                if (val < m_cost[x])
                {
                    m_cost[x] = val;
                    q.emplace(val, x);
                }
        };
        for (u32 o = 0; o < O; ++o)
        {
            cnt[o] = static_cast<u32>(R.pre(o).size());
            if (cnt[o] == 0)
                fire(o);
        }
        while (!q.empty())
        {
            const auto [c, p] = q.top();
            q.pop();
            if (c > m_cost[p])
                continue;
            for (u32 o : R.pre_of(p))
            {
                acc[o] = std::max(acc[o], c);
                if (--cnt[o] == 0)
                    fire(o);
            }
        }
    }

    const Task& T;
    const heuristics::RelaxedTask& R;
    std::vector<std::vector<u32>> m_ops_of_ga;
    std::unordered_map<std::string, u32> m_ga, m_pred, m_obj;
    std::vector<u32> m_cost;
};

bool ce_task(const Task& task) { return task.compiled().has_conditional_effects; }

/// One walk state against the fork's set-additive, h² and perfect values (fork_heuristics.json).
template<class Report>
void compare_fork_heuristics(const test::json::Value& fh, usize wi, usize si, const test::json::Value& st, const State& s,
                             heuristics::Heuristic* hsa, heuristics::Heuristic* hh2, heuristics::Heuristic* hstar,
                             heuristics::Heuristic* hstar_unit,
                             test::ReferenceH2* h2_mimir, bool h2_comparable, const Task& task, const heuristics::RelaxedTask& R,
                             Tally& t, Report report)
{
    const auto& walks = fh["walks"];
    if (walks.is_null() || wi >= walks.arr.size() || si >= walks[wi].arr.size())
        return;
    const auto& fs_ = walks[wi][si];
    if (fs_["atoms"].num != st["fluent_atoms"]["count"].num || fs_["hash"].str != st["fluent_atoms"]["hash"].str)
    {
        ++t.misaligned;
        report("the walk of fork_heuristics.json differs from the golden walk");
        return;
    }
    const auto& h = fh["h"];
    const double hmax_fork = golden_h(st["h"]["hmax"]);
    if (hsa)
    {
        const double v = hsa->evaluate(s), f = golden_h(h["setadd"][wi][si]);
        ++t.sa_states;
        t.sa_equal += v == f;
        t.sa_higher += v > f;
        t.sa_lower += v < f;
        if (std::isinf(v) != std::isinf(f))
            report("set-additive " + show(v) + ", fork " + show(f));
    }
    if (hstar)
    {
        const double vstar = hstar->evaluate(s), f = golden_h(h["perfect"][wi][si]);
        ++t.hstar_states;
        if (vstar != f)
        {
            ++t.hstar_diff;
            report("perfect " + show(vstar) + ", fork " + show(f));
        }
    }
    if (hh2)
    {
        const double v = hh2->evaluate(s), f = golden_h(h["h2"][wi][si]);
        ++t.h2_states;
        t.h2_equal += v == f;
        t.h2_higher += v > f;
        // mymyr's h² is admissible and at least h_max (the fork's h_max equals mymyr's)
        const double vstar = hstar_unit ? hstar_unit->evaluate(s) : heuristics::k_dead_end;
        t.h2_fork_inadmissible += f > vstar;
        if (v < hmax_fork || v > vstar)
        {
            ++t.h2_bad;
            report("h2 " + show(v) + " outside [h_max " + show(hmax_fork) + ", unit h* " + show(vstar) + "]");
        }
        if (h2_comparable)
        {
            ++t.h2_comparable;
            // the fork's delete check only drops pairs mymyr excludes: its value is at most mymyr's
            if (f > v)
            {
                ++t.h2_bad;
                report("h2 " + show(v) + " below the fork's " + show(f));
            }
            if (h2_mimir)
            {
                const double m = h2_mimir->evaluate(test::true_props(task, R, s), R.goal());
                if (m == f)
                    ++t.h2_mimir_equal;
                else
                {
                    ++t.h2_bad;
                    report("mimir's h2 on mymyr's grounding " + show(m) + ", fork " + show(f));
                }
            }
        }
    }
}

void run_task(const fs::path& file, Tally& total)
{
    const test::json::Value doc = test::json::parse_file(file.string());
    const std::string name = doc["task"].str.empty() ? file.stem().string() : doc["task"].str;
    fs::path dom, prob;
    if (!pddl_of(doc["source"], dom, prob))
    {
        std::printf("GOLDEN %-40s skipped: PDDL not found (%s)\n", name.c_str(), prob.string().c_str());
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto data = frontend::load_task(dom, prob);
    const auto task = Task::create(*data);
    Workspace ws(*task);
    Successors& succ = ws.successors();
    succ.set_witness_pruning(false);

    const auto& walks = doc["walks"]["walks"];
    bool want_relaxed = false;
    for (const auto& w : walks.arr)
        for (const auto& st : w["steps"].arr)
            want_relaxed |= st["h"].has("hmax");

    heuristics::GroundingStats gs;
    std::shared_ptr<const heuristics::RelaxedTask> relaxed;
    std::unique_ptr<heuristics::Heuristic> hb, hg, hm, ha, hf, lm, la, lf;
    bool lifted = std::getenv("MYMYR_GOLDEN_NO_LIFTED") == nullptr;
    {
        heuristics::Options o;
        o.kind = heuristics::Kind::Blind;
        hb = heuristics::make_heuristic(*task, o);
        o.kind = heuristics::Kind::GoalCount;
        hg = heuristics::make_heuristic(*task, o);
    }
    std::map<std::pair<u64, u64>, ForkFF> fork_plans;
    std::unique_ptr<ForkPlanCheck> plan_check;
    // the fork's set-additive, h² and perfect values of this task
    const test::json::Value* fh = nullptr;
    if (fork_heuristics().has("tasks") && fork_heuristics()["tasks"].has(name))
        fh = &fork_heuristics()["tasks"][name];
    std::unique_ptr<heuristics::Heuristic> hsa, hh2, hstar, hstar_unit;
    std::unique_ptr<test::ReferenceH2> h2_mimir;  // mimir's h² on mymyr's grounding
    bool h2_comparable = false;
    if (want_relaxed)
    {
        relaxed = heuristics::ground(*task, {}, &gs);
        // The lifted comparison dominates the run time (sokoban-ipc p05: 24 s in Release); sanitizer builds keep it to
        // tasks of at most MYMYR_GOLDEN_LIFTED_MAX_OPS relaxed operators (default 2000 there, unlimited otherwise).
        u64 lifted_max_ops = ~u64{0};
#if defined(MYMYR_SANITIZED)
        lifted_max_ops = 2000;
#endif
        if (const char* e = std::getenv("MYMYR_GOLDEN_LIFTED_MAX_OPS"); e && *e)
            lifted_max_ops = std::stoull(e);
        if (relaxed && relaxed->num_ops() > lifted_max_ops)
            lifted = false;
        heuristics::Options o;
        o.relaxed = relaxed;
        o.kind = heuristics::Kind::Max;
        hm = heuristics::make_heuristic(*task, o);
        o.kind = heuristics::Kind::Add;
        ha = heuristics::make_heuristic(*task, o);
        o.kind = heuristics::Kind::FF;
        hf = heuristics::make_heuristic(*task, o);
        if (lifted)  // the fallback: the same values without grounding
        {
            heuristics::Options l;
            l.evaluation = heuristics::Evaluation::Lifted;
            l.kind = heuristics::Kind::Max;
            lm = heuristics::make_heuristic(*task, l);
            l.kind = heuristics::Kind::Add;
            la = heuristics::make_heuristic(*task, l);
            l.kind = heuristics::Kind::FF;
            lf = heuristics::make_heuristic(*task, l);
        }
        if (fh && relaxed)
        {
            o.kind = heuristics::Kind::SetAdditive;
            if (fh->operator[]("h").has("setadd"))
                hsa = heuristics::make_heuristic(*task, o);
            o.kind = heuristics::Kind::H2;
            if (fh->operator[]("h").has("h2") && relaxed->num_props() <= 4000)
            {
                hh2 = heuristics::make_heuristic(*task, o);
                bool negative = false;
                for (u32 p = 0; p < relaxed->num_props(); ++p)
                    negative = negative || relaxed->negative(p);
                h2_comparable = !ce_task(*task) && !task->has_axioms() && !negative;
                u32 max_props = 1500;  // the reference is a Bellman-Ford over the full pair table
#if defined(MYMYR_SANITIZED)
                max_props = 300;
#endif
                if (h2_comparable && relaxed->num_props() <= max_props)
                    h2_mimir = std::make_unique<test::ReferenceH2>(*relaxed, true);
            }
            u64 max_states = 1'500'000;
#if defined(MYMYR_SANITIZED)
            max_states = 100'000;
#endif
            if (fh->operator[]("h").has("perfect"))
            {
                datasets::StateSpaceOptions so;
                so.remove_if_unsolvable = false;
                so.labels = false;
                so.max_states = max_states;
                auto space = datasets::generate_state_space(task, so);
                if (space.status == datasets::StateSpaceStatus::Ok && space.space)
                {
                    // the fork's perfect heuristic is the cost-goal distance (action costs), as mymyr's with real costs
                    hstar = heuristics::perfect(space.space, heuristics::Costs::Real);
                    hstar_unit = heuristics::perfect(space.space);  // bounds h² (unit costs)
                    if (fh->has("astar_perfect"))
                    {
                        // A* with h*: the fork's optimal cost; with unit costs it expands the states of its plan but
                        // the goal (popped, not expanded)
                        search::BestFirstOptions bo;
                        bo.evaluator = hstar.get();
                        const auto r = search::astar_eager(*task, bo);
                        const auto& fa = fh->operator[]("astar_perfect");
                        EXPECT_EQ(r.status, search::SearchStatus::Solved) << name;
                        EXPECT_EQ(r.cost, fa["plan_cost"].num) << name;
                        if (space.space->unit_costs())
                        {
                            EXPECT_EQ(r.stats.expanded, r.plan.size()) << name;
                        }
                        std::printf("GOLDEN %-40s A* perfect: cost %g, plan length %zu, expanded %llu (fork: cost %g, "
                                    "expanded %g)\n",
                                    name.c_str(), r.cost, r.plan.size(), static_cast<unsigned long long>(r.stats.expanded),
                                    fa["plan_cost"].num, fa["expanded"].num);
                    }
                }
            }
        }
        if (const char* d = std::getenv("MYMYR_FORK_PLANS"); d && *d && fs::exists(fs::path(d) / (file.stem().string() + ".txt")))
        {
            fork_plans = read_fork_plans(fs::path(d) / (file.stem().string() + ".txt"));
            plan_check = std::make_unique<ForkPlanCheck>(*task, *relaxed);
        }
    }

    Tally t;
    int shown = 0;
    auto report = [&](const std::string& what)
    {
        if (shown++ < 8)
            ADD_FAILURE() << name << ": " << what;
    };
    const bool ce = task->compiled().has_conditional_effects;
    for (usize wi = 0; wi < walks.size(); ++wi)
    {
        const auto& steps = walks[wi]["steps"];
        State s = task->initial_state();
        for (usize si = 0; si < steps.size(); ++si)
        {
            const auto& st = steps[si];
            const std::string where = "walk " + std::to_string(wi) + " step " + std::to_string(si);
            std::vector<std::string> atoms = task->format_atoms(s);
            if (atoms.size() != static_cast<usize>(st["fluent_atoms"]["count"].num) ||
                hex16(set_hash(atoms)) != st["fluent_atoms"]["hash"].str)
            {
                report(where + ": fluent atoms differ from the fork's (replay diverged)");
                break;
            }
            ++t.states;
            if (task->is_goal(s) != st["is_goal"].b)
            {
                ++t.goal;
                report(where + ": goal test differs");
            }
            const auto& h = st["h"];
            if (hb->evaluate(s) != golden_h(h["blind"]))
                ++t.blind;
            const double gc = hg->evaluate(s);
            if (gc != golden_h(h["goal_count"]))
            {
                ++t.gc;
                report(where + ": goal count " + show(gc) + ", fork " + show(golden_h(h["goal_count"])));
            }
            if (hm && h.has("hmax"))
            {
                const double vm = hm->evaluate(s), va = ha->evaluate(s), vf = hf->evaluate(s);
                const double fm = golden_h(h["hmax"]), fa = golden_h(h["hadd"]), ff = golden_h(h["hff"]);
                if (vm != fm)
                {
                    ++t.hmax;
                    report(where + ": h_max " + show(vm) + ", fork " + show(fm));
                }
                if (va != fa)
                {
                    ++t.hadd;
                    report(where + ": h_add " + show(va) + ", fork " + show(fa));
                }
                // The fork's h_FF differs between fork processes on some tasks (rubiks-cube: its atom numbering, and
                // with it the heap order among equal costs, depends on heap addresses). There the golden step lists
                // every value seen over repeated fork runs, and any of them is the fork's.
                bool observed_hit = false;
                if (h.has("hff_observed"))
                {
                    ++t.ff_observed;
                    for (const auto& x : h["hff_observed"].arr)
                        observed_hit = observed_hit || golden_h(x) == vf;
                    t.ff_observed_hit += observed_hit;
                }
                if (!std::isinf(ff) && ff < fm)
                {
                    ++t.ff_below_hmax;
                    std::printf("GOLDEN %s %s: fork h_FF %s < fork h_max %s; mymyr h_FF %s h_max %s\n", name.c_str(),
                                where.c_str(), show(ff).c_str(), show(fm).c_str(), show(vf).c_str(), show(vm).c_str());
                }
                if (vf != ff && !observed_hit)
                {
                    ++t.hff;
                    (vf > ff ? t.ff_higher : t.ff_lower) += 1;
                    if (!std::isinf(vf) && !std::isinf(ff))
                        t.ff_absdiff = std::max(t.ff_absdiff, std::abs(vf - ff));
                    const auto it = fork_plans.find({wi, si});
                    if (plan_check && it != fork_plans.end() && !std::isinf(ff))
                    {
                        // explained: the fork's plan is a best-supporter plan of mymyr's relaxation (a tie-break)
                        ++t.ff_checked;
                        if (it->second.hff != ff)
                            ++t.ff_fork_varies;  // this fork run differs from the golden run (see ForkPlanCheck)
                        const std::string why = plan_check->check(s, it->second);
                        if (why.empty())
                            ++t.ff_explained;
                        else
                            report(where + ": h_FF " + show(vf) + ", fork " + show(ff) + " not explained: " + why);
                    }
                }
                if (std::isinf(vf) != std::isinf(vm) || (!std::isinf(vf) && vf < vm && !ce))
                {
                    ++t.ff_invariant;
                    report(where + ": h_FF " + show(vf) + " violates the relaxed-plan invariants (h_max " + show(vm) + ")");
                }
                if (lm)
                {
                    const double wm = lm->evaluate(s), wa = la->evaluate(s), wf = lf->evaluate(s);
                    if (wm != fm)
                    {
                        ++t.lifted_hmax;
                        report(where + ": lifted h_max " + show(wm) + ", fork " + show(fm));
                    }
                    if (wa != fa)
                    {
                        ++t.lifted_hadd;
                        report(where + ": lifted h_add " + show(wa) + ", fork " + show(fa));
                    }
                    if (std::isinf(wf) != std::isinf(wm) || (!std::isinf(wf) && wf < wm && !ce))
                    {
                        ++t.lifted_ff_invariant;
                        report(where + ": lifted h_FF " + show(wf) + " violates the relaxed-plan invariants (h_max " + show(wm) + ")");
                    }
                }
            }
            if (fh && (hsa || hh2 || hstar))
                compare_fork_heuristics(*fh, wi, si, st, s, hsa.get(), hh2.get(), hstar.get(), hstar_unit.get(), h2_mimir.get(),
                                        h2_comparable,
                                        *task, *relaxed, t, [&](const std::string& what) { report(where + ": " + what); });
            const auto& taken = st["taken"];
            if (taken.is_null())
                break;
            bool found = false;
            Action next;
            succ.for_each_applicable(s,
                                     [&](const ActionLabel& a, const Delta&) -> bool
                                     {
                                         if (task->format(a) != taken.str)
                                             return true;
                                         next = Action(a);
                                         found = true;
                                         return false;
                                     });
            if (!found)
            {
                report(where + ": action " + taken.str + " not applicable");
                break;
            }
            s = succ.apply(s, ActionLabel{next.schema, next.binding});
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto u = [](u64 v) { return static_cast<unsigned long long>(v); };
    std::printf("GOLDEN %-40s states %3llu | h_max %llu h_add %llu gc %llu blind %llu goal %llu | h_FF %llu (mine higher %llu, lower %llu, "
                "max |diff| %g; explained %llu/%llu, fork run differs %llu; fork-unstable %llu, mine observed %llu; fork h_FF<h_max %llu) | "
                "lifted h_max %llu h_add %llu | ops %llu | %.2fs\n",
                name.c_str(), u(t.states), u(t.hmax), u(t.hadd), u(t.gc), u(t.blind), u(t.goal), u(t.hff), u(t.ff_higher), u(t.ff_lower),
                t.ff_absdiff, u(t.ff_explained), u(t.ff_checked), u(t.ff_fork_varies), u(t.ff_observed), u(t.ff_observed_hit),
                u(t.ff_below_hmax), u(t.lifted_hmax), u(t.lifted_hadd), u(gs.operators), secs);
    std::fflush(stdout);
    EXPECT_EQ(t.hmax, 0u) << name;
    EXPECT_EQ(t.hadd, 0u) << name;
    EXPECT_EQ(t.gc, 0u) << name;
    EXPECT_EQ(t.blind, 0u) << name;
    EXPECT_EQ(t.goal, 0u) << name;
    EXPECT_EQ(t.ff_invariant, 0u) << name;
    EXPECT_EQ(t.ff_explained, t.ff_checked) << name;
    total.ff_observed += t.ff_observed;
    total.ff_observed_hit += t.ff_observed_hit;
    total.ff_below_hmax += t.ff_below_hmax;
    EXPECT_EQ(t.lifted_hmax, 0u) << name;
    EXPECT_EQ(t.lifted_hadd, 0u) << name;
    EXPECT_EQ(t.lifted_ff_invariant, 0u) << name;
    if (t.sa_states || t.h2_states || t.hstar_states)
        std::printf("GOLDEN %-40s set-additive %llu states: equal %llu, higher %llu, lower %llu | h2 %llu states: equal %llu, "
                    "higher %llu, fork above h* %llu, comparable %llu (mimir's definition reproduces the fork in %llu) | "
                    "perfect %llu states, differ %llu\n",
                    name.c_str(), u(t.sa_states), u(t.sa_equal), u(t.sa_higher), u(t.sa_lower), u(t.h2_states), u(t.h2_equal),
                    u(t.h2_higher), u(t.h2_fork_inadmissible), u(t.h2_comparable), u(t.h2_mimir_equal), u(t.hstar_states), u(t.hstar_diff));
    EXPECT_EQ(t.misaligned, 0u) << name;
    EXPECT_EQ(t.h2_bad, 0u) << name;
    EXPECT_EQ(t.hstar_diff, 0u) << name;
    total.sa_states += t.sa_states;
    total.sa_equal += t.sa_equal;
    total.sa_higher += t.sa_higher;
    total.sa_lower += t.sa_lower;
    total.h2_states += t.h2_states;
    total.h2_equal += t.h2_equal;
    total.h2_higher += t.h2_higher;
    total.h2_fork_inadmissible += t.h2_fork_inadmissible;
    total.h2_comparable += t.h2_comparable;
    total.h2_mimir_equal += t.h2_mimir_equal;
    total.hstar_states += t.hstar_states;
    total.states += t.states;
    total.hmax += t.hmax;
    total.hadd += t.hadd;
    total.gc += t.gc;
    total.hff += t.hff;
    total.ff_checked += t.ff_checked;
    total.ff_explained += t.ff_explained;
    total.ff_fork_varies += t.ff_fork_varies;
    total.ff_higher += t.ff_higher;
    total.ff_lower += t.ff_lower;
    total.lifted_hmax += t.lifted_hmax;
    total.lifted_hadd += t.lifted_hadd;
    total.ff_absdiff = std::max(total.ff_absdiff, t.ff_absdiff);
}
}  // namespace

TEST(HeuristicsGolden, WalkValuesEqualTheForks)
{
    const fs::path dir = golden_dir();
    if (!fs::is_directory(dir))
        GTEST_SKIP() << "no golden data at " << dir << " (set MYMYR_GOLDEN_DIR)";
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".json")
            files.push_back(e.path());
    std::sort(files.begin(), files.end());
    const char* filter = std::getenv("MYMYR_GOLDEN_FILTER");
    Tally total;
    for (const auto& f : files)
    {
        if (filter && *filter && f.stem().string().find(filter) == std::string::npos)
            continue;
        SCOPED_TRACE(f.stem().string());
        run_task(f, total);
    }
    auto u = [](u64 v) { return static_cast<unsigned long long>(v); };
    std::printf("GOLDEN total states %llu | h_max %llu h_add %llu gc %llu | h_FF %llu (mine higher %llu, lower %llu, max |diff| %g; "
                "explained %llu/%llu, fork run differs %llu; fork-unstable %llu, mine observed %llu; fork h_FF<h_max %llu) | "
                "lifted h_max %llu h_add %llu\n",
                u(total.states), u(total.hmax), u(total.hadd), u(total.gc), u(total.hff), u(total.ff_higher), u(total.ff_lower),
                total.ff_absdiff, u(total.ff_explained), u(total.ff_checked), u(total.ff_fork_varies), u(total.ff_observed),
                u(total.ff_observed_hit), u(total.ff_below_hmax), u(total.lifted_hmax), u(total.lifted_hadd));
    std::printf("GOLDEN total set-additive %llu states: equal %llu, higher %llu, lower %llu | h2 %llu states: equal %llu, higher %llu, "
                "fork above h* %llu, comparable %llu (mimir's definition reproduces the fork in %llu) | perfect %llu states\n",
                u(total.sa_states), u(total.sa_equal), u(total.sa_higher), u(total.sa_lower), u(total.h2_states), u(total.h2_equal),
                u(total.h2_higher), u(total.h2_fork_inadmissible),
                u(total.h2_comparable), u(total.h2_mimir_equal), u(total.hstar_states));
}
