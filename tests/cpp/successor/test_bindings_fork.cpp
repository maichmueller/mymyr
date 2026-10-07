// Binding generators against the fork's (tests/data/bindings/fork_bindings.json, written by
// tests/data/fork_golden/search_fork/run_bindings.py from search_fork --algo walk_ground).
//
// Every task is loaded from its PDDL with mymyr's front end and the fork's seeded walks are replayed by the names of
// the actions taken (checking the fluent atom set of every step by count and hash). In every step:
//   - goal: the goal's literals as a condition without variables (ConjunctiveCondition::goal without its numeric
//     constraints, which the fork's condition leaves out) has as many groundings as the fork's;
//   - pre: per schema (matched by name), the schema's precondition as a condition has the fork's groundings: count and
//     set hash of the binding strings "(schema o1 ... ok)";
//   - act: per schema, the schema's bindings (the applicable actions) equal the fork's ActionSatisficingBindingGenerator;
//   - the ground literals of every grounding, summed per kind (static, fluent, derived), equal the fork's.
// The tasks whose PDDL is missing are skipped (reported).

#include "../frontend/golden.hpp"
#include "../support/json.hpp"

#include "mymyr/frontend/domain.hpp"
#include "mymyr/successor/bindings.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace mymyr;

namespace
{
u64 fnv1a64(const std::string& s)
{
    u64 h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s)
    {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string hex16(u64 v)
{
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

/// Groundings in the fork's summary: count, set hash of "(name o1 ... ok)", ground literals per kind.
struct Counts
{
    u64 n = 0, hash = 0, lits[3] = {0, 0, 0};

    [[nodiscard]] std::string str() const
    {
        return "n " + std::to_string(n) + " hash " + hex16(hash) + " literals " + std::to_string(lits[0]) + "/" +
               std::to_string(lits[1]) + "/" + std::to_string(lits[2]);
    }
};

std::string fork_str(const test::json::Value& v)
{
    return "n " + std::to_string(static_cast<u64>(v["n"].num)) + " hash " + v["hash"].str + " literals " +
           std::to_string(static_cast<u64>(v["literals"].arr[0].num)) + "/" +
           std::to_string(static_cast<u64>(v["literals"].arr[1].num)) + "/" +
           std::to_string(static_cast<u64>(v["literals"].arr[2].num));
}

template<class Target>
Counts ground(const Task& task, Workspace& ws, const Target& target, StateView s, const std::string& name)
{
    const formalism::TaskData& D = task.data();
    Counts c;
    for_each_ground_conjunction(task, ws, target, s, {},
                                [&](const GroundConjunction& g)
                                {
                                    std::string str = "(" + name;
                                    for (ObjectId o : g.binding)
                                        str += " " + std::string(D.str(D.objects[o.v].name));
                                    c.hash += fnv1a64(str + ")");
                                    c.lits[0] += g.static_literals.size();
                                    c.lits[1] += g.fluent_literals.size();
                                    c.lits[2] += g.derived_literals.size();
                                    ++c.n;
                                });
    return c;
}

const test::json::Value& fork_doc()
{
    static const test::json::Value doc = []
    {
        const fs::path f = fs::path(MYMYR_SOURCE_DIR) / "tests" / "data" / "bindings" / "fork_bindings.json";
        return fs::exists(f) ? test::json::parse_file(f.string()) : test::json::Value{};
    }();
    return doc;
}

struct Tally
{
    u64 states = 0, goal = 0, goal_diff = 0, pre = 0, pre_diff = 0, act = 0, act_diff = 0, lit_diff = 0;
};
}  // namespace

TEST(Fork, BindingCountsAlongTheWalks)
{
    const test::json::Value& doc = fork_doc();
    if (!doc.has("tasks"))
        GTEST_SKIP() << "no tests/data/bindings/fork_bindings.json";
    std::map<std::string, test::GoldenTask> pddl;
    std::vector<std::string> missing;
    for (auto& t : test::suite_tasks(&missing))
        pddl.emplace(t.name, t);
    for (auto& t : test::numeric_tasks(&missing))
        pddl.emplace(t.name, t);

    Tally total;
    u64 tasks = 0, skipped = 0;
    for (const auto& [name, rec] : doc["tasks"].obj)
    {
        if (rec.has("killed"))
        {
            std::printf("BINDINGS %-36s fork killed: %s\n", name.c_str(), rec["killed"].str.c_str());
            continue;
        }
        const auto it = pddl.find(name);
        if (it == pddl.end())
        {
            std::printf("BINDINGS %-36s skipped: PDDL not found\n", name.c_str());
            ++skipped;
            continue;
        }
        const auto data = frontend::load_task(it->second.domain, it->second.problem);
        const auto task = Task::create(*data);
        const formalism::TaskData& D = task->data();
        Workspace ws(*task);
        Successors& succ = ws.successors();
        succ.set_witness_pruning(false);
        // the fork's schemas by name (the i-th schema of a name, where normalization splits one action into several)
        std::vector<i64> schema_of;
        std::map<std::string, u32> seen;
        for (const auto& n : rec["schemas"].arr)
        {
            i64 k = -1;
            u32 occurrence = seen[n.str]++;
            for (u32 i = 0; i < D.schemas.size() && k < 0; ++i)
                if (D.str(D.schemas[i].name) == n.str && occurrence-- == 0)
                    k = i;
            schema_of.push_back(k);
            EXPECT_GE(k, 0) << name << ": no schema named " << n.str;
        }
        ConjunctiveCondition goal = ConjunctiveCondition::goal(*task);
        goal.constraints.clear();
        goal.exprs.clear();
        goal.expr_terms.clear();
        std::vector<ConjunctiveCondition> pre;
        for (u32 k = 0; k < D.schemas.size(); ++k)
            pre.push_back(ConjunctiveCondition::precondition(*task, SchemaId{k}));

        Tally t;
        int shown = 0;
        auto report = [&](const std::string& what)
        {
            if (shown++ < 6)
                ADD_FAILURE() << name << ": " << what;
        };
        const auto& walks = rec["walks"].arr;
        for (usize wi = 0; wi < walks.size(); ++wi)
        {
            const auto& steps = walks[wi]["steps"].arr;
            State s = task->initial_state();
            for (usize si = 0; si < steps.size(); ++si)
            {
                const auto& st = steps[si];
                const std::string where = "walk " + std::to_string(wi) + " step " + std::to_string(si);
                const std::vector<std::string> atoms = task->format_atoms(s);
                u64 h = 0;
                for (const std::string& a : atoms)
                    h += fnv1a64(a);
                if (atoms.size() != static_cast<usize>(st["atoms"].num) || hex16(h) != st["hash"].str)
                {
                    report(where + ": fluent atoms differ from the fork's (replay diverged)");
                    break;
                }
                ++t.states;
                auto lits_equal = [](const Counts& c, const test::json::Value& f)
                {
                    return c.lits[0] == static_cast<u64>(f["literals"].arr[0].num) &&
                           c.lits[1] == static_cast<u64>(f["literals"].arr[1].num) &&
                           c.lits[2] == static_cast<u64>(f["literals"].arr[2].num);
                };
                auto check = [&](const Counts& c, const test::json::Value& f, u64& cases, u64& diffs, const std::string& what)
                {
                    ++cases;
                    const bool bindings_equal =
                        c.n == static_cast<u64>(f["n"].num) && hex16(c.hash) == f["hash"].str;
                    if (!bindings_equal)
                    {
                        ++diffs;
                        report(where + ": " + what + ": mymyr " + c.str() + ", fork " + fork_str(f));
                    }
                    else if (!lits_equal(c, f))
                    {
                        ++t.lit_diff;
                        report(where + ": " + what + " ground literals: mymyr " + c.str() + ", fork " + fork_str(f));
                    }
                };
                check(ground(*task, ws, goal, s.view(), "goal"), st["goal"], t.goal, t.goal_diff, "goal");
                for (usize i = 0; i < schema_of.size(); ++i)
                {
                    if (schema_of[i] < 0)
                        continue;
                    const u32 k = static_cast<u32>(schema_of[i]);
                    const std::string sname = rec["schemas"].arr[i].str;
                    check(ground(*task, ws, pre[k], s.view(), sname), st["pre"].arr[i], t.pre, t.pre_diff, "precondition of " + sname);
                    check(ground(*task, ws, SchemaId{k}, s.view(), sname), st["act"].arr[i], t.act, t.act_diff, "schema " + sname);
                }
                if (!st.has("action"))
                    break;
                bool found = false;
                Action next;
                succ.for_each_applicable(s,
                                         [&](const ActionLabel& a, const Delta&) -> bool
                                         {
                                             if (task->format(a) != st["action"].str)
                                                 return true;
                                             next = Action(a);
                                             found = true;
                                             return false;
                                         });
                if (!found)
                {
                    report(where + ": action " + st["action"].str + " not applicable");
                    break;
                }
                s = succ.apply(s, ActionLabel{next.schema, next.binding});
            }
        }
        auto u = [](u64 v) { return static_cast<unsigned long long>(v); };
        std::printf("BINDINGS %-36s states %3llu | goal %llu/%llu pre %llu/%llu act %llu/%llu differ | literals differ %llu\n",
                    name.c_str(), u(t.states), u(t.goal_diff), u(t.goal), u(t.pre_diff), u(t.pre), u(t.act_diff), u(t.act),
                    u(t.lit_diff));
        EXPECT_GT(t.states, 0u) << name;
        total.states += t.states;
        total.goal += t.goal;
        total.goal_diff += t.goal_diff;
        total.pre += t.pre;
        total.pre_diff += t.pre_diff;
        total.act += t.act;
        total.act_diff += t.act_diff;
        total.lit_diff += t.lit_diff;
        ++tasks;
    }
    auto u = [](u64 v) { return static_cast<unsigned long long>(v); };
    std::printf("BINDINGS total: %llu tasks (%llu skipped), %llu states, %llu goal + %llu precondition + %llu schema cases; "
                "differ: %llu + %llu + %llu, ground literals %llu\n",
                u(tasks), u(skipped), u(total.states), u(total.goal), u(total.pre), u(total.act), u(total.goal_diff),
                u(total.pre_diff), u(total.act_diff), u(total.lit_diff));
    EXPECT_EQ(total.goal_diff + total.pre_diff + total.act_diff + total.lit_diff, 0u);
    if (tasks == 0)
        GTEST_SKIP() << "no task's PDDL found";
}
