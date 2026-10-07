// Tuple graphs: against the fork's TupleGraphImpl on the state-space suite (tests/data/tuple_graphs/
// fork_tuple_graphs.json, written by tests/data/fork_golden/search_fork/run_tuple_graphs.py), the definition on small
// tasks, independence of the thread count, and symmetry reduction.

#include "../frontend/golden.hpp"
#include "../support/json.hpp"
#include "../support/suite.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
std::filesystem::path data(const std::string& rel) { return test::fork_data_dir() / rel; }

TaskPtr fork_task(const std::string& dir, const std::string& problem)
{
    const auto d = data(dir + "/domain.pddl"), p = data(dir + "/" + problem);
    if (!std::filesystem::exists(d) || !std::filesystem::exists(p))
        return nullptr;
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Frozen;
    return Task::create(*frontend::load_task(d, p), to);
}

StateSpacePtr space_of(TaskPtr task, bool symmetry_pruning = false, u32 threads = 1)
{
    StateSpaceOptions o;
    o.threads = threads;
    o.remove_if_unsolvable = false;
    o.symmetry_pruning = symmetry_pruning;
    StateSpaceResult r = generate_state_space(std::move(task), o);
    EXPECT_EQ(r.status, StateSpaceStatus::Ok);
    return r.space;
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

/// The digest of the golden file (run_tuple_graphs.py, tests/data/tuple_graphs/README.md).
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

/// The breadth-first distances from r over the space's forward graph (-1: unreachable).
std::vector<i64> bfs(const StateSpace& S, u32 r)
{
    std::vector<i64> dist(S.num_states(), -1);
    std::vector<u32> layer{r}, next;
    dist[r] = 0;
    for (i64 d = 1; !layer.empty(); ++d)
    {
        next.clear();
        for (u32 p : layer)
            for (u64 e = S.forward_offsets()[p]; e < S.forward_offsets()[p + 1]; ++e)
                if (const u32 c = S.forward_targets()[e]; dist[c] < 0)
                {
                    dist[c] = d;
                    next.push_back(c);
                }
        std::swap(layer, next);
    }
    return dist;
}

/// The structural invariants of a tuple graph of a space without symmetry reduction.
void check_invariants(const TupleGraph& g)
{
    const StateSpace& S = *g.space();
    const auto dist = bfs(S, g.root());
    ASSERT_GE(g.num_distances(), 1u);
    EXPECT_EQ(g.distance_offsets().front(), 0u);
    EXPECT_EQ(g.distance_offsets().back(), g.num_vertices());
    EXPECT_TRUE(g.dominance_pruning() || g.width() == 0 ? g.vertices_at(0).size() == 1 : g.vertices_at(0).size() >= 1);
    u64 edges = 0;
    for (u32 d = 0; d < g.num_distances(); ++d)
    {
        EXPECT_FALSE(g.vertices_at(d).empty()) << d;
        const auto layer = g.problem_vertices_at(d);
        EXPECT_TRUE(std::ranges::is_sorted(layer));
        for (u32 p : layer)
            EXPECT_EQ(dist[p], d);
        for (u32 v : g.vertices_at(d))
        {
            EXPECT_EQ(g.distance(v), d);
            EXPECT_LE(g.tuple(v).size(), g.width());
            EXPECT_TRUE(std::ranges::is_sorted(g.tuple(v)));
            EXPECT_FALSE(g.problem_vertices(v).empty());
            EXPECT_TRUE(std::ranges::is_sorted(g.problem_vertices(v)));
            for (u32 p : g.problem_vertices(v))
            {
                EXPECT_EQ(dist[p], d);
                for (u32 a : g.tuple(v))
                    EXPECT_TRUE(S.state(p).contains(SlotId{a}));
            }
            EXPECT_EQ(d == 0, g.predecessors(v).empty());
            for (u32 u : g.predecessors(v))
                EXPECT_EQ(g.distance(u), d - 1);
            for (u32 w : g.successors(v))
                EXPECT_EQ(g.distance(w), d + 1);
            edges += g.successors(v).size();
        }
    }
    EXPECT_EQ(edges, g.num_edges());
}

// ----------------------------------------------------------------------------------------------- fork parity
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
        test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/tuple_graphs/fork_tuple_graphs.json");
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
        GTEST_SKIP() << "fork data missing: " << data(c.dir);
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

// ----------------------------------------------------------------------------------------------- definition
// A chain a -> b -> c of three locations visited by one agent, one fact per location: the tuple graph of width 1
// from the start follows the chain.
constexpr const char* kChainDomain = R"((define (domain chain)
  (:requirements :strips)
  (:predicates (at ?l) (link ?a ?b) (visited ?l))
  (:action move :parameters (?a ?b)
    :precondition (and (at ?a) (link ?a ?b))
    :effect (and (at ?b) (not (at ?a)) (visited ?b))))
)";
constexpr const char* kChainProblem = R"((define (problem chain3) (:domain chain)
  (:objects a b c)
  (:init (at a) (visited a) (link a b) (link b c))
  (:goal (visited c)))
)";

TaskPtr chain_task()
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Frozen;
    return Task::create(*frontend::Domain::from_string(kChainDomain, "d.pddl")->instantiate_string(kChainProblem, "p.pddl"), to);
}

TEST(TupleGraph, ChainWidthOne)
{
    const TaskPtr task = chain_task();
    const StateSpacePtr S = space_of(task);
    ASSERT_TRUE(S);
    ASSERT_EQ(S->num_states(), 3u);  // {at a}, {at b, visited b}, {at c, visited b, visited c}
    const TupleGraph g = tuple_graph(S, 0, {.width = 1});
    check_invariants(g);
    ASSERT_EQ(g.num_distances(), 3u);
    // distance 0: the empty tuple; 1: (at b) and (visited b) reach state 1, one kept; 2: (at c), (visited c)
    EXPECT_EQ(g.vertices_at(0).size(), 1u);
    EXPECT_TRUE(g.tuple(0).empty());
    ASSERT_EQ(g.vertices_at(1).size(), 1u);
    ASSERT_EQ(g.vertices_at(2).size(), 1u);
    const u32 v1 = g.distance_offsets()[1], v2 = g.distance_offsets()[2];
    // the canonical tuple: "(at b)" < "(visited b)"
    ASSERT_EQ(g.tuple(v1).size(), 1u);
    EXPECT_EQ(task->format(SlotId{g.tuple(v1)[0]}), "(at b)");
    EXPECT_EQ(task->format(SlotId{g.tuple(v2)[0]}), "(at c)");
    EXPECT_EQ(std::vector<u32>(g.successors(0).begin(), g.successors(0).end()), std::vector<u32>{v1});
    EXPECT_EQ(std::vector<u32>(g.predecessors(v2).begin(), g.predecessors(v2).end()), std::vector<u32>{v1});
    // without dominance pruning: every root tuple at distance 0 (empty, (at a), (link a b), ... are static: only
    // fluent atoms count), both tuples at 1 and 2
    const TupleGraph h = tuple_graph(S, 0, {.width = 1, .dominance_pruning = false});
    check_invariants(h);
    EXPECT_EQ(h.vertices_at(0).size(), 1u + S->state(0).count());
    EXPECT_EQ(h.vertices_at(1).size(), 2u);
    EXPECT_EQ(h.vertices_at(2).size(), 2u);
    EXPECT_EQ(h.num_edges(), h.vertices_at(0).size() * 2 + 2 * 2);
    // width 0: the root and its successor
    const TupleGraph z = tuple_graph(S, 0, {.width = 0});
    check_invariants(z);
    EXPECT_EQ(z.num_vertices(), 2u);
    EXPECT_EQ(std::vector<u32>(z.problem_vertices_at(1).begin(), z.problem_vertices_at(1).end()), std::vector<u32>{1});
    // the last state: no successor, one distance
    const TupleGraph last = tuple_graph(S, 2, {.width = 2});
    EXPECT_EQ(last.num_distances(), 1u);
    EXPECT_EQ(last.num_vertices(), 1u);
}

TEST(TupleGraph, RejectsBadArguments)
{
    const StateSpacePtr S = space_of(chain_task());
    ASSERT_TRUE(S);
    EXPECT_THROW((void)tuple_graph(S, 0, {.width = 6}), std::invalid_argument);
    EXPECT_THROW((void)tuple_graph(S, 3, {.width = 1}), std::invalid_argument);
    EXPECT_THROW((void)tuple_graphs(nullptr, {}), std::invalid_argument);
    const TupleGraph g = tuple_graph(S, 0, {.width = 1});
    EXPECT_THROW((void)g.tuple(g.num_vertices()), std::out_of_range);
    EXPECT_THROW((void)g.problem_vertices_at(g.num_distances()), std::out_of_range);
}

// ----------------------------------------------------------------------------------------------- threads
TEST(TupleGraph, IndependentOfTheThreadCount)
{
    struct Case
    {
        const char* dir;
        const char* problem;
        bool symmetric;
    };
#if defined(MYMYR_SANITIZED)
    const Case cases[] = {{"gripper", "p-2-0.pddl", false}, {"gripper", "p-2-0.pddl", true}, {"spanner", "p-1-1-3-1.pddl", false}};
#else
    const Case cases[] = {{"gripper", "p-2-0.pddl", false},  {"gripper", "test_problem4.pddl", true}, {"blocks_3", "test_problem2.pddl", false},
                          {"delivery", "test_problem2.pddl", false}, {"miconic-fulladl", "test_problem.pddl", false}};
#endif
    for (const Case& c : cases)
    {
        const TaskPtr task = fork_task(c.dir, c.problem);
        if (!task)
            GTEST_SKIP();
        const StateSpacePtr S = space_of(task, c.symmetric);
        ASSERT_TRUE(S);
        for (u32 w : {0u, 1u, 2u})
            for (bool pruning : {true, false})
            {
                SCOPED_TRACE(std::string(c.dir) + "/" + c.problem + " width " + std::to_string(w) + (pruning ? "" : " no pruning") +
                             (c.symmetric ? " symmetric" : ""));
                const auto ref = tuple_graphs(S, {.width = w, .dominance_pruning = pruning, .threads = 1});
                ASSERT_EQ(ref.size(), S->num_states());
                for (u32 T : {4u, 8u})
                {
                    const auto par = tuple_graphs(S, {.width = w, .dominance_pruning = pruning, .threads = T});
                    ASSERT_EQ(par.size(), ref.size());
                    for (usize v = 0; v < ref.size(); ++v)
                        EXPECT_TRUE(par[v] == ref[v]) << "T=" << T << " vertex " << v;
                }
                for (u32 v = 0; v < S->num_states(); v += std::max<u32>(1, S->num_states() / 7))
                    EXPECT_TRUE(tuple_graph(S, v, {.width = w, .dominance_pruning = pruning}) == ref[v]);
            }
    }
}

// ----------------------------------------------------------------------------------------------- symmetry
TEST(TupleGraph, SymmetryReducedSpaces)
{
    // the counts of the fork's knowledge base tests (tests/unit/datasets/knowledge_base.cpp), per problem
    const TaskPtr p1 = fork_task("gripper", "p-1-0.pddl"), p2 = fork_task("gripper", "p-2-0.pddl");
    if (!p1 || !p2)
        GTEST_SKIP();
    auto totals = [](const std::vector<TupleGraph>& gs)
    {
        u64 v = 0, e = 0;
        for (const auto& g : gs)
            v += g.num_vertices(), e += g.num_edges();
        return std::pair{v, e};
    };
    for (bool symmetric : {false, true})
    {
        const StateSpacePtr S1 = space_of(p1, symmetric), S2 = space_of(p2, symmetric);
        ASSERT_TRUE(S1 && S2);
        EXPECT_EQ(S1->num_states() + S2->num_states(), symmetric ? 18u : 36u);
        const auto [v1, e1] = totals(tuple_graphs(S1, {.width = 1}));
        const auto [v2, e2] = totals(tuple_graphs(S2, {.width = 1}));
        EXPECT_EQ(v1 + v2, symmetric ? 76u : 220u);
        EXPECT_EQ(e1 + e2, symmetric ? 70u : 184u);
        const auto [z1, f1] = totals(tuple_graphs(S1, {.width = 0}));
        const auto [z2, f2] = totals(tuple_graphs(S2, {.width = 0}));
        EXPECT_EQ(z1 + z2, symmetric ? 52u : 128u);
        EXPECT_EQ(f1 + f2, symmetric ? 34u : 92u);
    }
}
}  // namespace
