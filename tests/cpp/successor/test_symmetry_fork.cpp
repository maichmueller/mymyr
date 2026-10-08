// Symmetry pruning against the fork's WL1 mode (tests/data/symmetry/fork_symmetry.json, written by
// tests/data/fork_golden/search_fork/run_symmetry.py from search_fork --algo symmetry_states) on the tasks of the
// state-space parity suite (tests/cpp/datasets/fork_cases.inc).
//
// Every task is loaded from its PDDL and its states are enumerated breadth-first without pruning; each state is
// matched to the fork's by its fluent atoms and function values (both enumerations must find the same states). Per
// state:
//   - classes: the colour classes of the objects equal those of datasets::color_refinement_certificate on the
//     state's object graph (the reference implementation), and the fork's classes refined by "the object has a
//     neighbour in the object graph" (successor/symmetry.hpp: mimir never refines vertices without neighbours);
//   - selection: the representative rule, evaluated here from the applicable actions, the static domains and the
//     fork's classes, keeps exactly the fork's actions (so the static domains and the rule are the fork's even where
//     the classes differ), and evaluated with mymyr's classes it keeps exactly the actions generate() emits under
//     SymmetryPruning::Wl1;
//   - where the classes equal the fork's, the emitted actions equal the fork's, as sets of action names.
// Then brfs (exhaustive, and stopping at a goal) and A* with the blind heuristic, with pruning on, on the tasks whose
// classes equal the fork's in every state: the status, the plan length (brfs) and cost (A*), and the number of states
// of the exhaustive brfs equal the fork's (none of them depends on the order of the successors). Every plan found is
// replayed and reaches a goal state.

#include "../frontend/golden.hpp"
#include "../support/json.hpp"

#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;

namespace
{
const test::json::Value& fork_doc()
{
    static const test::json::Value doc = []
    {
        const fs::path f = fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "symmetry" / "fork_symmetry.json";
        return fs::exists(f) ? test::json::parse_file(f.string()) : test::json::Value{};
    }();
    return doc;
}

/// Class labels renumbered by first occurrence (equal partitions give equal vectors).
std::vector<u32> canonical(const std::vector<u32>& c)
{
    std::map<u32, u32> id;
    std::vector<u32> out;
    for (u32 x : c)
        out.push_back(id.emplace(x, static_cast<u32>(id.size())).first->second);
    return out;
}

/// The fluent atoms and the defined fluent function values ("(f o1 ... ok)=v", %.17g) of s, sorted and joined.
std::string key_of(const Task& task, const State& s)
{
    std::vector<std::string> items = task.format_atoms(s);
    const std::vector<f64> values = task.numeric_values(s);
    for (u32 slot = 0; slot < values.size(); ++slot)
    {
        if (std::isnan(values[slot]))
            continue;
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.17g", values[slot]);
        items.push_back(task.numeric_name(slot) + "=" + buf);
    }
    std::sort(items.begin(), items.end());
    std::string k;
    for (const std::string& a : items)
        k += a + " ";
    return k;
}

/// The representative rule of successor/symmetry.hpp, evaluated on the applicable `actions` with object classes
/// `cls`: the names of the actions whose parameters are all kept.
std::set<std::string> select(const Task& task, detail::SymmetryPruner& pruner, const std::vector<Action>& actions,
                             const std::vector<u32>& cls)
{
    std::map<u32, std::vector<std::set<u32>>> kept;  // per schema and parameter
    std::set<std::string> out;
    for (const Action& a : actions)
    {
        const u32 schema = a.schema.v, arity = static_cast<u32>(a.binding.size());
        auto [it, fresh] = kept.try_emplace(schema);
        if (fresh)
        {
            std::map<u32, u32> n;  // class -> parameters whose domain meets it
            for (u32 i = 0; i < arity; ++i)
            {
                std::set<u32> met;
                for (u32 o : pruner.domain(schema, i))
                    met.insert(cls[o]);
                for (u32 c : met)
                    ++n[c];
            }
            for (u32 i = 0; i < arity; ++i)
            {
                std::map<u32, u32> used;
                std::set<u32> k;
                for (u32 o : pruner.domain(schema, i))  // ascending object ids
                    if (used[cls[o]]++ < n[cls[o]])
                        k.insert(o);
                it->second.push_back(std::move(k));
            }
        }
        bool all = true;
        for (u32 i = 0; i < arity; ++i)
            all = all && it->second[i].contains(a.binding[i].v);
        if (all)
            out.insert(task.format(ActionLabel{a.schema, a.binding}));
    }
    return out;
}

std::string difference(const std::set<std::string>& mine, const std::set<std::string>& theirs)
{
    std::string diff;
    for (const std::string& a : mine)
        if (!theirs.contains(a))
            diff += " +" + a;
    for (const std::string& a : theirs)
        if (!mine.contains(a))
            diff += " -" + a;
    return diff;
}

/// Replays `plan` from the initial state; whether every action is applicable and the last state is a goal state.
bool plan_reaches_goal(const Task& task, const std::vector<Action>& plan)
{
    Successors& succ = task.workspace().successors();
    State s = task.initial_state();
    for (const Action& a : plan)
    {
        const ActionLabel label{a.schema, a.binding};
        if (!succ.is_applicable(s, label))
            return false;
        s = succ.apply(s, label);
    }
    return succ.is_goal(s);
}

struct Tally
{
    u64 tasks = 0, states = 0, equal_classes = 0, refined_classes = 0, actions_compared = 0, searches = 0;
};
}  // namespace

TEST(SymmetryFork, ActionsAndSearchesMatchTheFork)
{
    const test::json::Value& doc = fork_doc();
    if (!doc.has("tasks"))
        GTEST_SKIP() << "no tests/data/symmetry/fork_symmetry.json";
    Tally total;
    u64 skipped = 0;
    for (const test::json::Value& rec : doc["tasks"].arr)
    {
        const std::string name = rec["task"].str;
        if (rec.has("killed"))
        {
            std::printf("SYMMETRY %-44s fork killed: %s\n", name.c_str(), rec["killed"].str.c_str());
            continue;
        }
        const std::string dir = name.substr(0, name.find('/')), problem = name.substr(name.find('/') + 1);
        const fs::path d = test::fork_data_dir() / dir / "domain.pddl", p = test::fork_data_dir() / dir / problem;
        if (!fs::exists(d) || !fs::exists(p))
        {
            ++skipped;
            continue;
        }
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        const TaskPtr task = Task::create(*frontend::load_task(d, p), to);
        const formalism::TaskData& D = task->data();
        ASSERT_EQ(D.num_objects(), rec["objects"].arr.size()) << name;
        for (u32 o = 0; o < D.num_objects(); ++o)
            ASSERT_EQ(D.str(D.objects[o].name), rec["objects"].arr[o].str) << name << ": object order differs from the fork's";

        // the fork's states by their fluent atoms and function values
        std::map<std::string, usize> fork_state;
        const auto& states = rec["states"].arr;
        for (usize i = 0; i < states.size(); ++i)
        {
            std::vector<std::string> items;
            for (const auto& a : states[i]["atoms"].arr)
                items.push_back(rec["atom_names"].arr[static_cast<usize>(a.num)].str);
            std::sort(items.begin(), items.end());
            std::string k;
            for (const std::string& a : items)
                k += a + " ";
            fork_state.emplace(k, i);
        }

        Successors& succ = task->workspace().successors();
        detail::SymmetryPruner& pruner = succ.symmetry_pruner();
        datasets::ObjectGraphBuilder graphs(*task);
        std::set<std::string> seen;
        std::deque<State> queue{task->initial_state()};
        seen.insert(key_of(*task, queue.front()));
        Tally t;
        int shown = 0;
        auto report = [&](const std::string& what)
        {
            if (shown++ < 6)
                ADD_FAILURE() << name << ": " << what;
        };
        while (!queue.empty())
        {
            const State s = queue.front();
            queue.pop_front();
            const std::string key = key_of(*task, s);
            const auto it = fork_state.find(key);
            if (it == fork_state.end())
            {
                report("state not among the fork's: " + key);
                continue;
            }
            const test::json::Value& fork = states[it->second];
            ++t.states;

            // classes: the reference colour refinement, and the fork's refined by having a neighbour
            succ.prepare(s);
            (void) pruner.compute(succ.engine());
            const auto cls = pruner.object_classes();
            const std::vector<u32> mine(cls.begin(), cls.end());
            const datasets::ObjectGraph g = graphs.build(s.view());
            std::vector<u32> stable;
            (void) datasets::color_refinement_certificate(g, &stable);
            stable.resize(D.num_objects());
            EXPECT_EQ(canonical(mine), canonical(stable)) << name << ": classes differ from the reference colour refinement";
            std::vector<u32> fork_classes, fork_refined;
            for (u32 o = 0; o < D.num_objects(); ++o)
            {
                const u32 c = static_cast<u32>(fork["classes"].arr[o].num);
                fork_classes.push_back(c);
                fork_refined.push_back(2 * c + (g.adjacent(o).empty() ? 0 : 1));
            }
            EXPECT_EQ(canonical(mine), canonical(fork_refined)) << name << ": classes differ from the fork's (refined)";
            const bool equal = canonical(mine) == canonical(fork_classes);
            t.equal_classes += equal;
            t.refined_classes += !equal;

            // selection
            std::set<std::string> theirs;
            for (const auto& a : fork["actions"].arr)
                theirs.insert(rec["action_names"].arr[static_cast<usize>(a.num)].str);
            succ.prepare(s);
            std::vector<Action> all;
            std::vector<State> next;
            LineVector<u64> tmp;
            succ.generate<false>(
                [&](u32 schema, const ObjectId* b, const Delta& delta)
                {
                    all.emplace_back(ActionLabel{SchemaId{schema}, {b, succ.arity(schema)}});
                    const u32 n = apply_delta(s.view().w, s.view().nw, delta, tmp);
                    next.emplace_back(tmp.data(), n, delta.num, delta.nnum);
                    return true;
                },
                false, true);
            std::set<std::string> pruned;
            succ.generate<false>(
                [&](u32 schema, const ObjectId* b, const Delta&)
                {
                    pruned.insert(task->format(ActionLabel{SchemaId{schema}, {b, succ.arity(schema)}}));
                    return true;
                },
                false, true, SymmetryPruning::Wl1);
            if (const std::set<std::string> rule = select(*task, pruner, all, fork_classes); rule != theirs)
                report("state " + key + ": the rule with the fork's classes keeps (+ rule only, - fork only):" + difference(rule, theirs));
            if (const std::set<std::string> rule = select(*task, pruner, all, mine); rule != pruned)
                report("state " + key + ": generate() differs from the rule (+ generate only, - rule only):" + difference(pruned, rule));
            if (equal)
            {
                ++t.actions_compared;
                if (pruned != theirs)
                    report("state " + key + ": pruned actions differ (+ mymyr only, - fork only):" + difference(pruned, theirs));
            }

            for (const State& x : next)
                if (seen.insert(key_of(*task, x)).second)
                    queue.push_back(x);
        }
        EXPECT_EQ(t.states, static_cast<u64>(rec["num_states"].num)) << name << ": reachable states differ";

        // searches with pruning on
        const test::json::Value& fsr = rec["searches"];
        BrfsOptions bo;
        bo.witness_pruning = false;
        bo.symmetry_pruning = SymmetryPruning::Wl1;
        const BrfsResult ex = brfs(*task, bo);
        bo.stop_at_goal = true;
        const BrfsResult br = brfs(*task, bo);
        search::BestFirstOptions ao;
        ao.heuristic.kind = heuristics::Kind::Blind;
        ao.symmetry_pruning = SymmetryPruning::Wl1;
        const search::BestFirstResult ar = search::astar_eager(*task, ao);
        if (br.solved)
        {
            EXPECT_TRUE(plan_reaches_goal(*task, br.plan)) << name << ": brfs plan";
        }
        if (ar.status == search::SearchStatus::Solved)
        {
            EXPECT_TRUE(plan_reaches_goal(*task, ar.plan)) << name << ": astar plan";
        }
        const bool fork_brfs_solved = fsr["brfs"]["status"].str == "solved";
        const bool fork_astar_solved = fsr["astar_blind"]["status"].str == "solved";
        if (t.refined_classes == 0)
        {
            EXPECT_EQ(ex.states, static_cast<u64>(fsr["brfs_exhaustive"]["expanded"].num)) << name << ": exhaustive brfs";
            EXPECT_EQ(br.solved, fork_brfs_solved) << name << ": brfs status";
            if (br.solved && fork_brfs_solved)
            {
                EXPECT_EQ(br.plan.size(), static_cast<usize>(fsr["brfs"]["plan_length"].num)) << name << ": brfs plan length";
            }
            EXPECT_EQ(ar.status == search::SearchStatus::Solved, fork_astar_solved) << name << ": astar status";
            if (ar.status == search::SearchStatus::Solved && fork_astar_solved)
            {
                EXPECT_DOUBLE_EQ(ar.cost, fsr["astar_blind"]["plan_cost"].num) << name << ": astar plan cost";
            }
            t.searches += 3;
        }

        std::printf("SYMMETRY %-44s states %4llu | classes equal %4llu refined %3llu | actions compared %4llu | searches "
                    "%s: brfs %llu states%s, astar %s (fork: %llu, %s, %s)\n",
                    name.c_str(), static_cast<unsigned long long>(t.states), static_cast<unsigned long long>(t.equal_classes),
                    static_cast<unsigned long long>(t.refined_classes), static_cast<unsigned long long>(t.actions_compared),
                    t.refined_classes == 0 ? "compared" : "not compared", static_cast<unsigned long long>(ex.states),
                    br.solved ? (", plan " + std::to_string(br.plan.size())).c_str() : ", unsolved",
                    ar.status == search::SearchStatus::Solved ? std::to_string(ar.cost).c_str() : "unsolved",
                    static_cast<unsigned long long>(fsr["brfs_exhaustive"]["expanded"].num), fsr["brfs"]["status"].str.c_str(),
                    fsr["astar_blind"]["status"].str.c_str());
        ++total.tasks;
        total.states += t.states;
        total.equal_classes += t.equal_classes;
        total.refined_classes += t.refined_classes;
        total.actions_compared += t.actions_compared;
        total.searches += t.searches;
    }
    std::printf("SYMMETRY total: %llu tasks (%llu skipped), %llu states (classes equal to the fork's %llu, refined %llu), "
                "%llu action sets equal to the fork's, %llu searches compared\n",
                static_cast<unsigned long long>(total.tasks), static_cast<unsigned long long>(skipped),
                static_cast<unsigned long long>(total.states), static_cast<unsigned long long>(total.equal_classes),
                static_cast<unsigned long long>(total.refined_classes), static_cast<unsigned long long>(total.actions_compared),
                static_cast<unsigned long long>(total.searches));
    if (total.tasks == 0)
        GTEST_SKIP() << "no task's PDDL found";
}
