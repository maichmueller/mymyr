// Tuple graphs against the fork's TupleGraphImpl on the state-space suite (tests/parity/data/tuple_graphs/
// fork_tuple_graphs.json, written by tests/parity/tools/fork_golden/search_fork/run_tuple_graphs.py).

#include "../../cpp/datasets/support.hpp"
#include "../../cpp/datasets/tuple_graph_support.hpp"
#include "../../cpp/support/json.hpp"
#include "golden.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;
using namespace mymyr::test::tg;

namespace
{
TaskPtr fork_task(const std::string& dir, const std::string& problem)
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Dense;
    return test::ss::dense_task(test::fork_data_dir(), dir, problem, to);
}

u64 fnv(const std::string& s)
{
    u64 h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s)
    {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string hex(u64 h)
{
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
    return b;
}

/// The fork golden's state key: FNV-1a-64 of the sorted fluent atom strings joined by newlines, then "\n=%.17g" per
/// numeric value.
std::string state_key(const Task& task, StateView s)
{
    std::vector<std::string> atoms = task.format_atoms(s);
    std::sort(atoms.begin(), atoms.end());
    std::string key;
    for (usize i = 0; i < atoms.size(); ++i)
        key += (i ? "\n" : "") + atoms[i];
    if (task.numeric_slots())
        for (f64 v : task.numeric_values(s))
        {
            char b[64];
            std::snprintf(b, sizeof b, "\n=%.17g", v);
            key += b;
        }
    return hex(fnv(key));
}

std::string set_hash(const std::set<std::string>& items)
{
    u64 h = 0;
    for (const auto& s : items)
        h += fnv(s);
    return hex(h);
}

/// The digest of the golden file (run_tuple_graphs.py; tests/parity/tools/fork_golden/README.md, "Tuple graphs").
struct Digest
{
    std::vector<u64> n, m, p;
    std::string v, e, q;
};

Digest digest(const TupleGraph& g, const std::vector<std::string>& keys)
{
    const Task& task = *g.space()->task();
    Digest out;
    std::vector<std::string> id(g.num_vertices());
    std::set<std::string> vs, es, qs;
    for (u32 d = 0; d < g.num_distances(); ++d)
    {
        std::set<std::string> at;
        for (u32 v : g.vertices_at(d))
        {
            std::vector<std::string> names;
            for (u32 a : g.tuple(v))
                names.push_back(task.format(SlotId{a}));
            std::sort(names.begin(), names.end());
            std::string t;
            for (const auto& n : names)
                t += n;
            id[v] = std::to_string(d) + ":" + t;
            std::vector<std::string> pk;
            for (u32 p : g.problem_vertices(v))
                pk.push_back(keys.at(p));
            std::sort(pk.begin(), pk.end());
            std::string ps;
            for (usize i = 0; i < pk.size(); ++i)
                ps += (i ? "," : "") + pk[i];
            vs.insert(id[v] + ":" + ps);
            at.insert(id[v]);
        }
        out.n.push_back(at.size());
        std::set<std::string> layer;
        for (u32 p : g.problem_vertices_at(d))
            layer.insert(std::to_string(d) + ":" + keys.at(p));
        out.p.push_back(layer.size());
        qs.insert(layer.begin(), layer.end());
    }
    for (u32 d = 1; d < g.num_distances(); ++d)
    {
        std::set<std::string> at;
        for (u32 v : g.vertices_at(d))
            for (u32 u : g.predecessors(v))
                at.insert(id[u] + ">" + id[v]);
        out.m.push_back(at.size());
        es.insert(at.begin(), at.end());
    }
    out.v = set_hash(vs);
    out.e = set_hash(es);
    out.q = set_hash(qs);
    return out;
}

std::vector<u64> counts(const test::json::Value& a)
{
    std::vector<u64> out;
    for (const auto& x : a.arr)
        out.push_back(static_cast<u64>(x.num));
    return out;
}

struct SuiteCase
{
    const char* dir;
    const char* problem;
    u32 states;
    u64 transitions;
    u32 goal, unsolvable;
    i32 max_goal_dist;
    u64 fp_content, fp_transitions, fp_vstar;
    f64 cost_sum;
};
const SuiteCase kSuite[] = {
#include "fork_cases.inc"
};

const test::json::Value& golden()
{
    static const test::json::Value doc =
        test::json::parse_file(std::string(MYMYR_PARITY_TEST_DATA_DIR) + "/tuple_graphs/fork_tuple_graphs.json");
    return doc;
}

class ForkTupleGraphs : public ::testing::TestWithParam<SuiteCase>
{
};

TEST_P(ForkTupleGraphs, MatchTheFork)
{
    const SuiteCase& c = GetParam();
    const std::string name = std::string(c.dir) + "/" + c.problem;
    const test::json::Value* rec = nullptr;
    for (const auto& r : golden()["tasks"].arr)
        if (r["task"].str == name)
            rec = &r;
    ASSERT_TRUE(rec) << name << " is not in fork_tuple_graphs.json";
    const TaskPtr task = fork_task(c.dir, c.problem);
    if (!task)
        GTEST_SKIP() << "reference data missing: " << test::fork_data_dir() / c.dir << " (set MYMYR_PARITY_DATA)";
#if defined(MYMYR_SANITIZED)
    if (c.states > 100)
        GTEST_SKIP() << "large space under sanitizers";
#endif
    const StateSpacePtr S = space_of(task);
    ASSERT_TRUE(S);
    ASSERT_EQ(S->num_states(), static_cast<u32>((*rec)["states"].num));
    std::vector<std::string> keys(S->num_states());
    std::unordered_map<std::string, u32> by_key;
    for (u32 v = 0; v < S->num_states(); ++v)
    {
        keys[v] = state_key(*task, S->state(v));
        ASSERT_TRUE(by_key.emplace(keys[v], v).second) << "two states with one key";
    }
    std::vector<u32> roots;
    for (const auto& k : (*rec)["roots"].arr)
    {
        const auto it = by_key.find(k.str);
        ASSERT_NE(it, by_key.end()) << "root " << k.str;
        roots.push_back(it->second);
    }
    u64 compared = 0;
    for (const auto& [config, graphs] : (*rec)["graphs"].obj)
    {
        TupleGraphOptions o;
        o.width = static_cast<u32>(config[1] - '0');
        o.dominance_pruning = config.size() < 4 || config[3] == '1';
        ASSERT_EQ(graphs.size(), roots.size());
        for (usize i = 0; i < roots.size(); ++i)
        {
            SCOPED_TRACE(name + " " + config + " root " + std::to_string(roots[i]));
            const TupleGraph g = tuple_graph(S, roots[i], o);
            check_invariants(g);
            const Digest d = digest(g, keys);
            const auto& want = graphs[i];
            EXPECT_EQ(d.n, counts(want["n"]));
            EXPECT_EQ(d.m, counts(want["m"]));
            EXPECT_EQ(d.p, counts(want["p"]));
            EXPECT_EQ(d.v, want["v"].str);
            EXPECT_EQ(d.e, want["e"].str);
            EXPECT_EQ(d.q, want["q"].str);
            ++compared;
        }
    }
    EXPECT_GT(compared, 0u);
}

std::string case_name(const ::testing::TestParamInfo<SuiteCase>& info)
{
    std::string n = std::string(info.param.dir) + "_" + info.param.problem;
    for (char& ch : n)
        if (!std::isalnum(static_cast<unsigned char>(ch)))
            ch = '_';
    return n;
}
INSTANTIATE_TEST_SUITE_P(ForkData, ForkTupleGraphs, ::testing::ValuesIn(kSuite), case_name);
}  // namespace
