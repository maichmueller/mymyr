// k-FWL certificates (k = 2, 3, 4): the partition of the states of the state-space suite into certificate classes
// against the fork's kfwl::compute_certificate<K> (tests/data/kfwl/fork_kfwl.json, written by
// tests/data/fork_golden/search_fork/run_kfwl.py), Cai-Fürer-Immerman pairs that separate 3-FWL from 4-FWL, isomorphic
// relabellings, the limits, and symmetry-reduced state spaces with k = 4.

#include "../frontend/golden.hpp"
#include "../support/json.hpp"
#include "../support/suite.hpp"
#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
// ----------------------------------------------------------------------------------------------- graphs
/// An undirected vertex-coloured graph (colours default to 0) with a palette of the colours 0 .. max.
ObjectGraph make_graph(u32 n, const std::vector<std::pair<u32, u32>>& edges, std::vector<u32> colors = {})
{
    ObjectGraph g;
    g.num_objects = n;
    g.color = colors.empty() ? std::vector<u32>(n, 0) : colors;
    const u32 k = n ? *std::max_element(g.color.begin(), g.color.end()) + 1 : 0;
    g.palette_offsets.assign(1, 0);
    for (u32 c = 0; c < k; ++c)
    {
        g.palette_values.push_back(c);
        g.palette_offsets.push_back(c + 1);
    }
    std::vector<std::vector<u32>> adj(n);
    for (auto [a, b] : edges)
    {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    g.offsets.assign(1, 0);
    for (auto& a : adj)
    {
        std::sort(a.begin(), a.end());
        g.neighbors.insert(g.neighbors.end(), a.begin(), a.end());
        g.offsets.push_back(g.neighbors.size());
    }
    return g;
}

/// The graph with vertex v renamed perm[v] (an isomorphic copy).
ObjectGraph relabel(const ObjectGraph& g, const std::vector<u32>& perm)
{
    const u32 n = g.num_vertices();
    std::vector<std::pair<u32, u32>> edges;
    std::vector<u32> colors(n);
    for (u32 v = 0; v < n; ++v)
    {
        colors[perm[v]] = g.color[v];
        for (u32 u : g.adjacent(v))
            if (v < u)
                edges.emplace_back(perm[v], perm[u]);
    }
    ObjectGraph h = make_graph(n, edges, colors);
    h.palette_offsets = g.palette_offsets;  // same colour ids, same palette
    h.palette_values = g.palette_values;
    return h;
}

/// The Cai-Fürer-Immerman graph over a base graph (Cai, Fürer, Immerman, "An optimal lower bound on the number of
/// variables for graph identification", Combinatorica 12, 1992), in the form with one vertex pair per base edge: per
/// base vertex v, one vertex m(v, S) for every even subset S of its edges (colour v); per base edge e, vertices e0
/// and e1 (colour |V| + e); m(v, S) is adjacent to e1 for e in S and to e0 for the other edges of v. `twisted` swaps
/// e0 and e1 for the first endpoint of edge 0. The two graphs are not isomorphic; k-FWL tells them apart iff k is at
/// least the treewidth of the base graph (Dawar and Richerby, "The power of counting logics on restricted classes of
/// finite structures", CSL 2007; K_{k+1} has treewidth k).
ObjectGraph cfi(u32 base_vertices, const std::vector<std::pair<u32, u32>>& base_edges, bool twisted)
{
    const u32 E = static_cast<u32>(base_edges.size());
    std::vector<std::vector<u32>> incident(base_vertices);
    for (u32 e = 0; e < E; ++e)
    {
        incident[base_edges[e].first].push_back(e);
        incident[base_edges[e].second].push_back(e);
    }
    std::vector<u32> colors;
    std::vector<std::pair<u32, u32>> edges;
    auto edge_vertex = [&](u32 e, u32 bit) { return 2 * e + bit; };  // the edge pairs come first
    for (u32 e = 0; e < E; ++e)
        colors.insert(colors.end(), {base_vertices + e, base_vertices + e});
    for (u32 v = 0; v < base_vertices; ++v)
    {
        const u32 d = static_cast<u32>(incident[v].size());
        for (u32 S = 0; S < (1u << d); ++S)
        {
            if (std::popcount(S) % 2)
                continue;
            const u32 m = static_cast<u32>(colors.size());
            colors.push_back(v);
            for (u32 i = 0; i < d; ++i)
            {
                const u32 e = incident[v][i];
                u32 bit = (S >> i) & 1;
                if (twisted && e == 0 && v == base_edges[0].first)
                    bit ^= 1;
                edges.emplace_back(m, edge_vertex(e, bit));
            }
        }
    }
    return make_graph(static_cast<u32>(colors.size()), edges, colors);
}

std::vector<std::pair<u32, u32>> complete_graph(u32 n)
{
    std::vector<std::pair<u32, u32>> e;
    for (u32 a = 0; a < n; ++a)
        for (u32 b = a + 1; b < n; ++b)
            e.emplace_back(a, b);
    return e;
}

/// A random graph on n vertices with edge probability p and `colors` vertex colours.
ObjectGraph random_graph(u32 n, double p, u32 colors, u64 seed)
{
    std::mt19937_64 rng(seed);
    std::vector<std::pair<u32, u32>> edges;
    for (u32 a = 0; a < n; ++a)
        for (u32 b = a + 1; b < n; ++b)
            if (std::uniform_real_distribution<double>(0, 1)(rng) < p)
                edges.emplace_back(a, b);
    std::vector<u32> col(n);
    for (u32 v = 0; v < n; ++v)
        col[v] = v < colors ? v : static_cast<u32>(rng() % colors);  // every colour used
    return make_graph(n, edges, col);
}

std::vector<u32> random_permutation(u32 n, u64 seed)
{
    std::vector<u32> p(n);
    std::iota(p.begin(), p.end(), 0u);
    std::mt19937_64 rng(seed);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

// ----------------------------------------------------------------------------------------------- certificates
TEST(Kfwl, CaiFurerImmermanSeparates2From3)
{
    // CFI over K4 (treewidth 3, 28 vertices): 2-FWL does not tell the pair apart, 3-FWL and 4-FWL do
    const ObjectGraph a = cfi(4, complete_graph(4), false), b = cfi(4, complete_graph(4), true);
    ASSERT_EQ(a.num_vertices(), 28u);
    EXPECT_EQ(color_refinement_certificate(a), color_refinement_certificate(b));
    EXPECT_EQ(kfwl_certificate(a, 2), kfwl_certificate(b, 2));
    EXPECT_NE(kfwl_certificate(a, 3), kfwl_certificate(b, 3));
#if !defined(MYMYR_SANITIZED)
    EXPECT_NE(kfwl_certificate(a, 4), kfwl_certificate(b, 4));
#endif
}

TEST(Kfwl, CaiFurerImmermanSeparates3From4)
{
    // CFI over K5 (treewidth 4, 60 vertices): 3-FWL does not tell the pair apart, 4-FWL does
    const ObjectGraph a = cfi(5, complete_graph(5), false), b = cfi(5, complete_graph(5), true);
    ASSERT_EQ(a.num_vertices(), 60u);
    EXPECT_EQ(kfwl_certificate(a, 3), kfwl_certificate(b, 3));
#if !defined(MYMYR_SANITIZED)  // 4-FWL on 60 vertices takes minutes under sanitizers
    EXPECT_NE(kfwl_certificate(a, 4), kfwl_certificate(b, 4));
#endif
}

TEST(Kfwl, IsomorphicRelabellingsAgree)
{
    std::vector<ObjectGraph> graphs = {cfi(4, complete_graph(4), false), cfi(4, complete_graph(4), true)};
    for (u64 seed = 1; seed <= 4; ++seed)
        graphs.push_back(random_graph(14, 0.3, 1 + static_cast<u32>(seed % 3), seed));
    graphs.push_back(make_graph(12, {}, {0, 1, 0, 1, 2, 2, 0, 0, 1, 1, 2, 0}));  // isolated vertices only
    for (const ObjectGraph& g : graphs)
        for (u64 seed = 11; seed <= 13; ++seed)
        {
            const ObjectGraph h = relabel(g, random_permutation(g.num_vertices(), seed));
            for (u32 k : {2u, 3u, 4u})
            {
#if defined(MYMYR_SANITIZED)
                if (k == 4 && g.num_vertices() > 14)
                    continue;
#endif
                EXPECT_EQ(kfwl_certificate(g, k), kfwl_certificate(h, k)) << "k = " << k << ", n = " << g.num_vertices();
            }
        }
}

TEST(Kfwl, HierarchyAndDegenerateGraphs)
{
    const ObjectGraph triangles = make_graph(6, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}});
    const ObjectGraph hexagon = make_graph(6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}});
    EXPECT_NE(kfwl_certificate(triangles, 4), kfwl_certificate(hexagon, 4));
    // the certificate depends on k
    EXPECT_NE(kfwl_certificate(hexagon, 3), kfwl_certificate(hexagon, 4));
    // empty and single-vertex graphs
    const ObjectGraph empty = make_graph(0, {});
    const ObjectGraph one = make_graph(1, {});
    for (u32 k : {2u, 3u, 4u})
    {
        EXPECT_EQ(kfwl_certificate(empty, k), kfwl_certificate(empty, k));
        EXPECT_NE(kfwl_certificate(empty, k), kfwl_certificate(one, k));
    }
    // colours matter: the two ends of a path coloured differently
    EXPECT_NE(kfwl_certificate(make_graph(3, {{0, 1}, {1, 2}}, {0, 1, 0}), 4), kfwl_certificate(make_graph(3, {{0, 1}, {1, 2}}, {1, 0, 0}), 4));
}

TEST(Kfwl, Limits)
{
    const ObjectGraph g = random_graph(12, 0.3, 2, 7);
    EXPECT_THROW((void)kfwl_certificate(g, 1), std::invalid_argument);
    EXPECT_THROW((void)kfwl_certificate(g, 5), std::invalid_argument);
    // the defaults admit n = 64 for k = 4 (64^5 = 2^30 colour tuples per round) and reject n = 65
    const KfwlLimits defaults;
    EXPECT_EQ(defaults.max_tuples, u64{1} << 26);
    EXPECT_EQ(defaults.max_round_work, u64{1} << 30);
    const ObjectGraph big = random_graph(65, 0.1, 3, 8);
    try
    {
        (void)kfwl_certificate(big, 4);
        ADD_FAILURE() << "no length_error for n = 65, k = 4";
    }
    catch (const std::length_error& e)
    {
        const std::string what = e.what();
        EXPECT_NE(what.find("4-FWL"), std::string::npos) << what;
        EXPECT_NE(what.find("n = 65"), std::string::npos) << what;
        EXPECT_NE(what.find("max_round_work"), std::string::npos) << what;
    }
    // explicit limits: the tuple bound, then the work bound
    EXPECT_THROW((void)kfwl_certificate(g, 4, {.max_tuples = 12 * 12 * 12 * 12 - 1}), std::length_error);
    EXPECT_THROW((void)kfwl_certificate(g, 4, {.max_round_work = 12ull * 12 * 12 * 12 * 12 - 1}), std::length_error);
    EXPECT_EQ(kfwl_certificate(g, 4, {.max_tuples = 12 * 12 * 12 * 12, .max_round_work = 12ull * 12 * 12 * 12 * 12}), kfwl_certificate(g, 4));
    try
    {
        (void)kfwl_certificate(g, 3, {.max_tuples = 100});
        ADD_FAILURE() << "no length_error for max_tuples = 100";
    }
    catch (const std::length_error& e)
    {
        const std::string what = e.what();
        EXPECT_NE(what.find("3-FWL"), std::string::npos) << what;
        EXPECT_NE(what.find("n = 12"), std::string::npos) << what;
        EXPECT_NE(what.find("max_tuples"), std::string::npos) << what;
    }
}

// ----------------------------------------------------------------------------------------------- fork parity
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

/// The fork golden's state key (tests/data/fork_golden/README.md, "Tuple graphs").
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
            char x[64];
            std::snprintf(x, sizeof x, "\n=%.17g", v);
            key += x;
        }
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(fnv(key)));
    return b;
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
    static const test::json::Value doc = test::json::parse_file(std::string(MYMYR_TEST_DATA_DIR) + "/kfwl/fork_kfwl.json");
    return doc;
}

const test::json::Value* golden_task(const SuiteCase& c)
{
    const std::string name = std::string(c.dir) + "/" + c.problem;
    for (const auto& r : golden()["tasks"].arr)
        if (r["task"].str == name)
            return &r;
    return nullptr;
}

/// Union-find over the sampled states.
struct Partition
{
    std::vector<u32> parent;
    explicit Partition(usize n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    u32 find(u32 x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    void join(u32 a, u32 b) { parent[find(a)] = find(b); }
    /// Joins the states of equal ids (ids < 0: not in the partition's domain).
    template<class Id>
    void join_equal(const std::vector<Id>& ids, const std::vector<bool>& in)
    {
        std::map<Id, u32> first;
        for (u32 i = 0; i < ids.size(); ++i)
        {
            if (!in[i])
                continue;
            if (auto [it, fresh] = first.emplace(ids[i], i); !fresh)
                join(i, it->second);
        }
    }
    /// The canonical class ids (first occurrence) of the states in `in`, -1 elsewhere.
    std::vector<i64> classes(const std::vector<bool>& in)
    {
        std::map<u32, i64> id;
        std::vector<i64> out(parent.size(), -1);
        for (u32 i = 0; i < parent.size(); ++i)
            if (in[i])
                out[i] = id.emplace(find(i), static_cast<i64>(id.size())).first->second;
        return out;
    }
};

/// Largest object graph certified per k in this build (the golden's caps are 200, 64 and 24).
u32 test_cap(u32 k)
{
#if defined(MYMYR_SANITIZED)
    return k == 2 ? 64 : k == 3 ? 24 : 12;
#else
    return k == 2 ? 200 : k == 3 ? 64 : 24;
#endif
}

class ForkKfwl : public ::testing::TestWithParam<SuiteCase>
{
};

// The fork's k-FWL is not invariant under relabelling for k >= 3 (its tuple subgraphs mismatch colours and edges, see
// datasets/certificates.hpp) and then splits classes of isomorphic states. The expectation is therefore the fork's
// partition with the classes of isomorphic states (by nauty, the fork's canonical forms) joined; where the fork splits
// no isomorphism class, that is the fork's partition itself.
TEST_P(ForkKfwl, PartitionsMatchTheFork)
{
    const SuiteCase& c = GetParam();
    const test::json::Value* rec = golden_task(c);
    ASSERT_TRUE(rec) << c.dir << "/" << c.problem << " is not in fork_kfwl.json";
    const TaskPtr task = fork_task(c.dir, c.problem);
    if (!task)
        GTEST_SKIP() << "fork data missing: " << data(c.dir);
    StateSpaceOptions o;
    o.remove_if_unsolvable = false;
    const StateSpaceResult r = generate_state_space(task, o);
    ASSERT_TRUE(r.space);
    const StateSpace& S = *r.space;
    ASSERT_EQ(S.num_states(), static_cast<u32>((*rec)["states"].num));
    std::unordered_map<std::string, u32> by_key;
    for (u32 v = 0; v < S.num_states(); ++v)
        ASSERT_TRUE(by_key.emplace(state_key(*task, S.state(v)), v).second) << "two states with one key";
    const auto& keys = (*rec)["keys"].arr;
    const usize m = keys.size();
    ObjectGraphBuilder ogb(*task);
    std::vector<ObjectGraph> graphs;
    for (usize i = 0; i < m; ++i)
    {
        const auto it = by_key.find(keys[i].str);
        ASSERT_NE(it, by_key.end()) << "sampled state " << keys[i].str << " is not in mymyr's space";
        graphs.push_back(ogb.build(S.state(it->second)));
        ASSERT_EQ(graphs.back().num_vertices(), static_cast<u32>((*rec)["n"][i].num)) << "object graph size of " << keys[i].str;
    }
    std::vector<i64> nauty(m);
    for (usize i = 0; i < m; ++i)
        nauty[i] = static_cast<i64>((*rec)["nauty"][i].num);
    std::vector<std::vector<bool>> in(5);
    std::vector<std::vector<Certificate>> certs(5);
    for (u32 k : {2u, 3u, 4u})
    {
        const std::string kk = "k" + std::to_string(k);
        ASSERT_TRUE((*rec).has(kk));
        const u32 cap = std::min<u32>(test_cap(k), static_cast<u32>((*rec)[kk + "_max_n"].num));
        std::vector<i64> fork(m);
        in[k].assign(m, false);
        certs[k].resize(m);
        for (usize i = 0; i < m; ++i)
        {
            fork[i] = static_cast<i64>((*rec)[kk][i].num);
            if (graphs[i].num_vertices() > cap)
                continue;
            ASSERT_GE(fork[i], 0) << kk << " missing for a graph of " << graphs[i].num_vertices() << " vertices";
            in[k][i] = true;
            certs[k][i] = kfwl_certificate(graphs[i], k);
        }
        Partition expected(m), mine(m);
        expected.join_equal(fork, in[k]);
        expected.join_equal(nauty, in[k]);
        mine.join_equal(certs[k], in[k]);
        EXPECT_EQ(mine.classes(in[k]), expected.classes(in[k])) << kk << " partition of the sampled states differs";
        // k-FWL never separates isomorphic states
        Partition iso(m);
        iso.join_equal(nauty, in[k]);
        iso.join_equal(certs[k], in[k]);
        EXPECT_EQ(iso.classes(in[k]), mine.classes(in[k])) << kk << " separates isomorphic states";
    }
    // (k+1)-FWL refines k-FWL
    for (u32 k : {2u, 3u})
        for (usize i = 0; i < m; ++i)
            for (usize j = i + 1; j < m; ++j)
            {
                if (in[k + 1][i] && in[k + 1][j] && certs[k + 1][i] == certs[k + 1][j])
                {
                    EXPECT_EQ(certs[k][i], certs[k][j]) << "k = " << k + 1 << " merges what k = " << k << " separates";
                }
            }
}

std::string case_name(const ::testing::TestParamInfo<SuiteCase>& info)
{
    std::string n = std::string(info.param.dir) + "_" + info.param.problem;
    for (char& ch : n)
        if (!std::isalnum(static_cast<unsigned char>(ch)))
            ch = '_';
    return n;
}
INSTANTIATE_TEST_SUITE_P(ForkData, ForkKfwl, ::testing::ValuesIn(kSuite), case_name);

// ----------------------------------------------------------------------------------------------- symmetry reduction
/// Largest object graph of the symmetry-reduced state spaces with 4-FWL in this build.
#if defined(MYMYR_SANITIZED)
constexpr u64 kSymmetricMaxN = 11;
#else
constexpr u64 kSymmetricMaxN = 16;
#endif

/// The suite tasks whose sampled object graphs (the golden's n) have at most kSymmetricMaxN vertices.
std::vector<SuiteCase> symmetric_cases()
{
    std::vector<SuiteCase> out;
    for (const SuiteCase& c : kSuite)
        if (const test::json::Value* rec = golden_task(c))
        {
            double n = 0;
            for (const auto& x : (*rec)["n"].arr)
                n = std::max(n, x.num);
            if (n <= static_cast<double>(kSymmetricMaxN))
                out.push_back(c);
        }
    return out;
}

/// The symmetry-reduced state space with 4-FWL: one state per class, as many as the fork's symmetry-reduced space
/// (nauty canonical forms), and independent of the thread count, on the tasks of symmetric_cases(). The work bound
/// of KfwlLimits enforces the size bound on every reachable object graph.
class ForkSymmetricKfwl4 : public ::testing::TestWithParam<SuiteCase>
{
};

TEST_P(ForkSymmetricKfwl4, ClassCountMatchesTheFork)
{
    const SuiteCase& c = GetParam();
    const test::json::Value* rec = golden_task(c);
    ASSERT_TRUE(rec);
    const TaskPtr task = fork_task(c.dir, c.problem);
    if (!task)
        GTEST_SKIP() << "fork data missing: " << data(c.dir);
    constexpr u64 n_max = kSymmetricMaxN;
    StateSpaceOptions o;
    o.remove_if_unsolvable = false;
    o.symmetry_pruning = true;
    o.certificate = CertificateKind::KFwl;
    o.fwl_k = 4;
    o.fwl_limits.max_round_work = n_max * n_max * n_max * n_max * n_max;
    StateSpaceResult r;
    try
    {
        r = generate_state_space(task, o);
    }
    catch (const std::length_error& e)
    {
        FAIL() << "a reachable object graph above " << n_max << " vertices: " << e.what();
    }
    ASSERT_TRUE(r.space);
    EXPECT_TRUE(r.space->symmetry_reduced());
    EXPECT_EQ(r.space->fwl_k(), 4u);
    EXPECT_EQ(r.space->fwl_limits(), o.fwl_limits);
    ASSERT_FALSE((*rec)["symmetric_states"].is_null());
    EXPECT_EQ(r.space->num_states(), static_cast<u32>((*rec)["symmetric_states"].num));
    for (u32 threads : {4u, 8u})
    {
        StateSpaceOptions ot = o;
        ot.threads = threads;
        const StateSpaceResult rt = generate_state_space(task, ot);
        ASSERT_TRUE(rt.space);
        ASSERT_EQ(rt.space->num_states(), r.space->num_states());
        EXPECT_TRUE(std::ranges::equal(rt.space->forward_offsets(), r.space->forward_offsets()));
        EXPECT_TRUE(std::ranges::equal(rt.space->forward_targets(), r.space->forward_targets()));
        for (u32 v = 0; v < r.space->num_states(); ++v)
            ASSERT_EQ(rt.space->state(v), r.space->state(v));
    }
}
INSTANTIATE_TEST_SUITE_P(ForkData, ForkSymmetricKfwl4, ::testing::ValuesIn(symmetric_cases()), case_name);

/// The instance pool with 4-FWL symmetry pruning: equal spaces on 1, 4 and 8 worker threads (certificates computed
/// concurrently), and the generalized state space over them.
TEST(SymmetricKfwl4, PoolAndGeneralizedStateSpace)
{
    std::vector<TaskPtr> tasks;
    for (const char* p : {"p-1-0.pddl", "p-2-0.pddl", "test_problem.pddl", "test_problem2.pddl"})
        if (TaskPtr t = fork_task("gripper", p))
            tasks.push_back(t);
    if (tasks.size() < 4)
        GTEST_SKIP() << "fork data missing";
    StateSpaceOptions o;
    o.remove_if_unsolvable = false;
    o.symmetry_pruning = true;
    o.fwl_k = 4;
    const auto one = generate_state_spaces(tasks, o, 1);
    for (u32 threads : {4u, 8u})
    {
        const auto many = generate_state_spaces(tasks, o, threads);
        ASSERT_EQ(many.size(), one.size());
        for (usize i = 0; i < one.size(); ++i)
        {
            ASSERT_TRUE(one[i].space && many[i].space);
            ASSERT_EQ(many[i].space->num_states(), one[i].space->num_states());
            EXPECT_TRUE(std::ranges::equal(many[i].space->forward_targets(), one[i].space->forward_targets()));
        }
    }
    EXPECT_EQ(one[0].space->num_states(), 6u);  // the fork's symmetry-reduced p-1-0
    EXPECT_EQ(one[1].space->num_states(), 12u);  // and p-2-0
    const auto G = GeneralizedStateSpace::create(ordered_spaces(one));
    EXPECT_TRUE(G->symmetry_reduced());
    // p-2-0 and test_problem2 are isomorphic: one of them is dropped
    EXPECT_EQ(G->spaces().size(), 3u);
    // k = 5 is rejected
    o.fwl_k = 5;
    EXPECT_THROW((void)generate_state_space(tasks[0], o), std::invalid_argument);
}
}  // namespace
