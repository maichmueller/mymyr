// IW ladder, optimized IW(1), single passes and SIW:
//   - fork parity: per-pass {arity, expanded, generated, generated_in_tree}, status and plan length equal the fork's
//     golden data (tests/data/expected on branch golden-fork, exported by the fork 0.16.3) on the 20 suite tasks whose
//     counts do not depend on the successor order (organic-synthesis and pegsol do, on a replay of the fork's exact order);
//   - an oracle: a literal re-implementation of the fork's pass semantics (BrFS with a duplicate check and the ArityZero
//     / ArityK pruning strategies over a hash set of tuples, successors materialized with Successors::apply) must give
//     the same per-pass statistics in every convention, for k = 0..3, with blocked states and custom goals;
//   - hand-computed counts on a small line task; budgets, cancellation, observer events, goal specs, plans;
//   - SIW against a reference built on the oracle.

#include "../support/suite.hpp"
#include "mymyr/formalism/text_format.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cctype>
#include <functional>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>

using namespace mymyr;
using namespace mymyr::search;
using namespace mymyr::test;

namespace
{
// ------------------------------------------------------------------------------------------------- fork golden data
struct GoldenPass
{
    u32 arity;
    u64 expanded, generated, in_tree;
};
struct GoldenRow
{
    const char* task;
    u32 k;
    SearchStatus status;
    u32 plan_length;
    std::vector<GoldenPass> passes;
};

// tests/data/expected/<task>.json "iw" (fork 0.16.3, lifted KPKC).
const std::vector<GoldenRow>& golden_rows()
{
    static const std::vector<GoldenRow> rows = {
        {"gripper__prob05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 26, 364, 25}}},
        {"gripper__prob05", 2, SearchStatus::Exhausted, 0, {{0, 26, 364, 25}, {1, 26, 364, 25}, {2, 470, 4732, 469}}},
        {"blocks__probBLOCKS-8-0", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 43, 200, 42}}},
        {"blocks__probBLOCKS-8-0", 2, SearchStatus::Exhausted, 0, {{0, 5, 22, 4}, {1, 43, 200, 42}, {2, 1526, 7758, 1525}}},
        {"logistics00__probLOGISTICS-6-1", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 10, 114, 9}}},
        {"logistics00__probLOGISTICS-6-1", 2, SearchStatus::Exhausted, 0, {{0, 10, 114, 9}, {1, 10, 114, 9}, {2, 316, 3225, 315}}},
        {"miconic__s7-4", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 21, 289, 20}}},
        {"miconic__s7-4", 2, SearchStatus::Exhausted, 0, {{0, 15, 203, 14}, {1, 21, 289, 20}, {2, 280, 3845, 279}}},
        {"visitall__visitall_x-6_y-3_r-100", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 18, 54, 17}}},
        {"visitall__visitall_x-6_y-3_r-100", 2, SearchStatus::Exhausted, 0, {{0, 4, 11, 3}, {1, 18, 54, 17}, {2, 267, 805, 266}}},
        {"sokoban-opt08-strips__p14", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 39, 99, 38}}},
        {"sokoban-opt08-strips__p14", 2, SearchStatus::Exhausted, 0, {{0, 3, 6, 2}, {1, 39, 99, 38}, {2, 2059, 5135, 2058}}},
        {"depot__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 10, 92, 9}}},
        {"depot__p02", 2, SearchStatus::Exhausted, 0, {{0, 8, 74, 7}, {1, 10, 92, 9}, {2, 241, 2255, 240}}},
        {"driverlog__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 27, 196, 26}}},
        {"driverlog__p03", 2, SearchStatus::Exhausted, 0, {{0, 8, 54, 7}, {1, 27, 196, 26}, {2, 485, 3666, 484}}},
        {"rovers__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 22, 158, 21}}},
        {"rovers__p02", 2, SearchStatus::Exhausted, 0, {{0, 8, 48, 7}, {1, 22, 158, 21}, {2, 225, 1675, 224}}},
        {"zenotravel__p05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 30, 318, 29}}},
        {"zenotravel__p05", 2, SearchStatus::Exhausted, 0, {{0, 11, 122, 10}, {1, 26, 318, 25}, {2, 587, 8003, 586}}},
        {"transport-opt08-strips__p23", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 27, 155, 26}}},
        {"transport-opt08-strips__p23", 2, SearchStatus::Exhausted, 0, {{0, 7, 39, 6}, {1, 27, 155, 26}, {2, 1137, 6981, 1136}}},
        {"freecell__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 43, 408, 42}}},
        {"freecell__p02", 2, SearchStatus::Solved, 16, {{0, 9, 79, 8}, {1, 43, 408, 42}, {2, 867, 7063, 869}}},
        {"snake-opt18-strips__p05", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 86, 196, 85}}},
        {"snake-opt18-strips__p05", 2, SearchStatus::Exhausted, 0, {{0, 2, 3, 1}, {1, 86, 196, 85}, {2, 2912, 5744, 2911}}},
        {"parcprinter-opt11-strips__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 41, 148, 40}}},
        {"parcprinter-opt11-strips__p03", 2, SearchStatus::Exhausted, 0, {{0, 2, 5, 1}, {1, 41, 148, 40}, {2, 663, 2269, 662}}},
        {"miconic-simpleadl__s10-2", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 29, 580, 28}}},
        {"miconic-simpleadl__s10-2", 2, SearchStatus::Exhausted, 0, {{0, 21, 420, 20}, {1, 29, 580, 28}, {2, 514, 10280, 513}}},
        {"caldera-split-opt18-adl__p04", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 14, 79, 13}}},
        {"caldera-split-opt18-adl__p04", 2, SearchStatus::Exhausted, 0, {{0, 5, 19, 4}, {1, 14, 79, 13}, {2, 69, 330, 68}}},
        {"pathways__p02", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 31, 383, 30}}},
        {"pathways__p02", 2, SearchStatus::Solved, 12, {{0, 13, 156, 12}, {1, 31, 383, 30}, {2, 1096, 13799, 1097}}},
        {"folding-opt23-adl__p01", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 92, 105, 91}}},
        {"folding-opt23-adl__p01", 2, SearchStatus::Exhausted, 0, {{0, 15, 28, 14}, {1, 92, 105, 91}, {2, 127, 140, 126}}},
        {"openstacks-opt08-adl__p03", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 25, 109, 24}}},
        {"openstacks-opt08-adl__p03", 2, SearchStatus::Exhausted, 0, {{0, 2, 9, 1}, {1, 25, 109, 24}, {2, 276, 1285, 275}}},
        {"philosophers__p03-phil4", 1, SearchStatus::Exhausted, 0, {{0, 0, 0, 0}, {1, 25, 76, 24}}},
        {"philosophers__p03-phil4", 2, SearchStatus::Exhausted, 0, {{0, 5, 20, 4}, {1, 25, 76, 24}, {2, 737, 2016, 736}}},
    };
    return rows;
}

IwOptions with_k(u32 k)
{
    IwOptions o;
    o.max_arity = k;
    return o;
}

TaskOptions lazy_atoms()
{
    TaskOptions o;
    o.atoms = TaskOptions::Atoms::Lazy;
    return o;
}

std::string param_name(std::string n)
{
    for (char& c : n)
        if (!std::isalnum(static_cast<unsigned char>(c)))
            c = '_';
    return n;
}

class IwGolden : public ::testing::TestWithParam<GoldenRow>
{
};

TEST_P(IwGolden, PerPassEqualsTheFork)
{
    const GoldenRow& g = GetParam();
    for (auto atoms : {TaskOptions::Atoms::Auto, TaskOptions::Atoms::Lazy})
    {
        TaskOptions to;
        to.atoms = atoms;
        const auto task = Task::from_text_file(task_path(g.task), to);
        IwOptions o;
        o.max_arity = g.k;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, g.status);
        EXPECT_EQ(r.plan.size(), g.plan_length);
        ASSERT_EQ(r.passes.size(), g.passes.size());
        for (usize i = 0; i < g.passes.size(); ++i)
        {
            EXPECT_EQ(r.passes[i].arity, g.passes[i].arity) << "pass " << i;
            EXPECT_EQ(r.passes[i].expanded, g.passes[i].expanded) << "pass " << i;
            EXPECT_EQ(r.passes[i].generated, g.passes[i].generated) << "pass " << i;
            EXPECT_EQ(r.passes[i].generated_in_tree, g.passes[i].in_tree) << "pass " << i;
        }
        if (g.k == 1)
        {
            EXPECT_TRUE(r.passes[0].placeholder);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Fork, IwGolden, ::testing::ValuesIn(golden_rows()),
                         [](const auto& i) { return param_name(std::string(i.param.task) + "_k" + std::to_string(i.param.k)); });

// ------------------------------------------------------------------------------------------------- the oracle
std::vector<u32> atoms_of(const State& s)
{
    std::vector<u32> out;
    for (SlotId x : s.slots())
        out.push_back(x.v);
    return out;
}

/// A tuple of at most 3 atoms (each below 2^21 - 1) as one key: the sorted atoms + 1, 21 bits each.
u64 pack(std::vector<u32> t)
{
    std::sort(t.begin(), t.end());
    u64 key = 0;
    for (u32 x : t)
        key = (key << 21) | (u64{x} + 1);
    return key;
}

/// Calls f(tuple) for every set of at most `size` atoms of `atoms` (the empty set included).
template<class F>
void for_each_subset(const std::vector<u32>& atoms, u32 size, F f)
{
    std::vector<u32> t;
    auto rec = [&](auto& self, usize from) -> void
    {
        f(t);
        if (t.size() == size)
            return;
        for (usize i = from; i < atoms.size(); ++i)
        {
            t.push_back(atoms[i]);
            self(self, i + 1);
            t.pop_back();
        }
    };
    rec(rec, 0);
}

struct RefPass
{
    SearchStatus status = SearchStatus::Exhausted;
    u64 expanded = 0, generated = 0, in_tree = 0, skipped = 0;
    u32 plan_length = 0;
    State goal_state;
    std::vector<State> admitted;  // in admission order (root excluded)
};

struct RefSpec
{
    u32 k = 1;
    bool root_continuation = false;
    WidthZero width_zero = WidthZero::ExpandDepthOne;
    bool witness = false, canonical = true;
    u64 max_expanded = ~u64{0};
    u32 max_depth = ~u32{0};
    std::vector<State> blocked;
};

/// The fork's pass (brfs.cpp handle_surviving_action + ArityZero/ArityK pruning strategies), literally: every
/// successor materialized, a duplicate check on admitted states, blocked states first, novelty over a set of tuples.
RefPass ref_pass(const Task& task, const State& root, const RefSpec& sp, const std::function<bool(const State&)>& goal)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    const bool w0 = succ.witness_pruning(), c0 = succ.canonical_order();
    succ.set_witness_pruning(sp.witness);
    succ.set_canonical_order(sp.canonical);
    RefPass r;
    // the tuples seen: every tuple of 1..k atoms of an admitted state (the root's are marked up front)
    std::unordered_set<u64> seen;
    EXPECT_LE(sp.k, 3u);
    EXPECT_LT(task.atoms().max_fluent_slots(), (1u << 21) - 1);
    if (sp.k >= 1)
        for_each_subset(atoms_of(root), sp.k,
                        [&](const std::vector<u32>& t)
                        {
                            if (!t.empty())
                                seen.insert(pack(t));
                        });
    struct Node
    {
        State s;
        u32 depth;
        bool skip;
    };
    std::vector<Node> q{{root, 0, false}};
    std::unordered_set<State> admitted{root};
    const std::unordered_set<State> blocked(sp.blocked.begin(), sp.blocked.end());
    for (usize i = 0; i < q.size(); ++i)
    {
        const Node nd = q[i];
        if (goal(nd.s))
        {
            r.status = SearchStatus::Solved;
            r.plan_length = nd.depth;
            r.goal_state = nd.s;
            break;
        }
        if (r.expanded >= sp.max_expanded)
        {
            r.status = SearchStatus::OutOfStates;
            break;
        }
        ++r.expanded;
        if (nd.skip || nd.depth >= sp.max_depth)
        {
            ++r.skipped;
            continue;
        }
        const bool is_root = i == 0;
        const std::vector<u32> parent_atoms = atoms_of(nd.s);
        for (const Action& a : succ.applicable_actions(nd.s))
        {
            ++r.generated;
            const State t = succ.apply(nd.s, a.label());
            if (blocked.count(t))
                continue;
            if (sp.k == 0)
            {
                if (!is_root || t == nd.s)
                    continue;
                ++r.in_tree;
                if (!admitted.insert(t).second)
                    continue;  // pushed again by the fork, popped as closed
                r.admitted.push_back(t);
                q.push_back({t, 1, sp.width_zero == WidthZero::RootOnly});
                continue;
            }
            if (t == nd.s || admitted.count(t))
                continue;  // self loop, or not a new search node
            // novel: some tuple of the successor that contains an atom the parent lacks is unseen (tuples without
            // one are seen: the parent was admitted); every such tuple is marked
            bool novel = false;
            const std::vector<u32> succ_atoms = atoms_of(t);
            for (u32 a : succ_atoms)
            {
                if (std::binary_search(parent_atoms.begin(), parent_atoms.end(), a))
                    continue;
                std::vector<u32> others;
                for (u32 x : succ_atoms)
                    if (x != a)
                        others.push_back(x);
                for_each_subset(others, sp.k - 1,
                                [&](std::vector<u32> tu)
                                {
                                    tu.push_back(a);
                                    novel |= seen.insert(pack(std::move(tu))).second;
                                });
            }
            if (is_root && sp.root_continuation)
            {
                ++r.in_tree;
                admitted.insert(t);
                r.admitted.push_back(t);
                q.push_back({t, 1, !novel});
                continue;
            }
            if (!novel)
                continue;
            ++r.in_tree;
            admitted.insert(t);
            r.admitted.push_back(t);
            q.push_back({t, nd.depth + 1, false});
        }
    }
    succ.set_witness_pruning(w0);
    succ.set_canonical_order(c0);
    return r;
}

std::function<bool(const State&)> task_goal(const Task& task)
{
    return [&task](const State& s) { return task.is_goal(s); };
}

void expect_pass(const IwPassStatistics& p, const RefPass& ref, const std::string& what)
{
    EXPECT_EQ(p.status, ref.status) << what;
    EXPECT_EQ(p.expanded, ref.expanded) << what;
    EXPECT_EQ(p.generated, ref.generated) << what;
    EXPECT_EQ(p.generated_in_tree, ref.in_tree) << what;
    EXPECT_EQ(p.skipped, ref.skipped) << what;
}

struct OracleVariant
{
    const char* name;
    u32 max_arity;
    int single_pass;  // >= 0: iw_pass(arity)
    WidthZero width_zero;
    bool witness, canonical;
    TaskOptions::Atoms atoms;
    u64 dense_bytes;
    u64 max_expanded;
};

const OracleVariant k_variants[] = {
    {"ladder2", 2, -1, WidthZero::ExpandDepthOne, false, true, TaskOptions::Atoms::Auto, u64{256} << 20, ~u64{0}},
    {"ladder1_optimized", 1, -1, WidthZero::ExpandDepthOne, false, true, TaskOptions::Atoms::Lazy, u64{256} << 20, ~u64{0}},
    {"ladder2_rootonly_witness_nocanonical", 2, -1, WidthZero::RootOnly, true, false, TaskOptions::Atoms::Lazy, u64{256} << 20, ~u64{0}},
    {"ladder2_frozen_sparse", 2, -1, WidthZero::ExpandDepthOne, false, true, TaskOptions::Atoms::Frozen, 0, ~u64{0}},
    {"pass3", 3, 3, WidthZero::ExpandDepthOne, false, true, TaskOptions::Atoms::Lazy, u64{256} << 20, 1500},
    {"pass3_sparse_witness", 3, 3, WidthZero::ExpandDepthOne, true, true, TaskOptions::Atoms::Auto, 0, 1500},
};

struct OracleParam
{
    std::string task;
    OracleVariant v;
};
void PrintTo(const OracleParam& p, std::ostream* os) { *os << p.task << "/" << p.v.name; }

class IwOracle : public ::testing::TestWithParam<OracleParam>
{
};

TEST_P(IwOracle, EqualsTheForkSemantics)
{
    const OracleParam& p = GetParam();
    TaskOptions to;
    to.atoms = p.v.atoms;
    const auto task = Task::from_text_file(task_path(p.task), to);
    IwOptions o;
    o.max_arity = p.v.max_arity;
    o.width_zero = p.v.width_zero;
    o.witness_pruning = p.v.witness;
    o.canonical_order = p.v.canonical;
    o.tables.max_dense_bytes = p.v.dense_bytes;
    o.control.budget.max_expanded = p.v.max_expanded;
    const IwResult r = p.v.single_pass >= 0 ? iw_pass(*task, static_cast<u32>(p.v.single_pass), o) : iw(*task, o);
    const State root = task->initial_state();
    const bool optimized = p.v.single_pass < 0 && p.v.max_arity == 1;
    u32 first = p.v.single_pass >= 0 ? static_cast<u32>(p.v.single_pass) : optimized ? 1 : 0;
    usize at = optimized ? 1 : 0;
    ASSERT_GE(r.passes.size(), at + 1);
    if (optimized)
    {
        EXPECT_TRUE(r.passes[0].placeholder);
        EXPECT_EQ(r.passes[0].expanded + r.passes[0].generated, 0u);
    }
    for (u32 k = first; at < r.passes.size(); ++k, ++at)
    {
        RefSpec sp;
        sp.k = k;
        sp.root_continuation = optimized && k == 1;
        sp.width_zero = p.v.width_zero;
        sp.witness = p.v.witness;
        sp.canonical = p.v.canonical;
        sp.max_expanded = p.v.max_expanded;
        const RefPass ref = ref_pass(*task, root, sp, task_goal(*task));
        expect_pass(r.passes[at], ref, "arity " + std::to_string(k));
        EXPECT_EQ(r.passes[at].arity, k);
        if (ref.status == SearchStatus::Solved)
        {
            EXPECT_EQ(r.status, SearchStatus::Solved);
            EXPECT_EQ(r.plan.size(), ref.plan_length);
            ASSERT_TRUE(r.goal_state.has_value());
            EXPECT_EQ(*r.goal_state, ref.goal_state);
            EXPECT_EQ(r.effective_width, k);
        }
    }
    if (r.status != SearchStatus::Solved && r.status != SearchStatus::OutOfStates)
    {
        EXPECT_EQ(r.passes.back().arity, p.v.single_pass >= 0 ? static_cast<u32>(p.v.single_pass) : p.v.max_arity);
    }
}

std::vector<OracleParam> oracle_params()
{
    std::vector<OracleParam> out;
    for (const auto& t : suite())
        for (OracleVariant v : k_variants)
        {
            // organic-synthesis has about 260 successors per state and large states: width 3 costs ~1 s per 15
            // expansions in the oracle, so its budget is smaller (the sanitizer builds run this suite too)
            if (v.single_pass == 3 && t.name.starts_with("organic-synthesis"))
                v.max_expanded = 40;
            out.push_back({t.name, v});
        }
    return out;
}

INSTANTIATE_TEST_SUITE_P(Suite, IwOracle, ::testing::ValuesIn(oracle_params()),
                         [](const auto& i) { return param_name(i.param.task + "_" + i.param.v.name); });

// ------------------------------------------------------------------------------------------------- a line task
// o0 - o1 - o2 - o3 - o4; move(x, y): at(x), adj(x, y) -> at(y), not at(x), visited(y). Start at o0, goal visited(o4).
void expect_valid_plan(const Task& task, const State& start, const std::vector<Action>& plan, const std::function<bool(const State&)>& goal);

formalism::TaskData line_task_data(const std::string& goal = "G 1\n2 1 4\n", u32 schemas = 1)
{
    std::string text = "O 5\nP 3\nS 2 adj\nF 1 at\nF 1 visited\n"
                       "SI 8\n0 0 1\n0 1 0\n0 1 2\n0 2 1\n0 2 3\n0 3 2\n0 3 4\n0 4 3\n"
                       "FI 2\n1 0\n2 0\n" +
                       goal + "A " + std::to_string(schemas) + "\n";
    for (u32 s = 0; s < schemas; ++s)
        text += (s == 0 ? "move" : "walk" + std::to_string(s)) + " 2\nL 2\n1 1 0\n0 1 0 1\nE 1\n0\nL 0\nF 3\n1 1 1\n1 0 0\n2 1 1\n";
    std::istringstream in(text);
    return formalism::read_task_text(in);
}

std::shared_ptr<const Task> line_task(const std::string& goal = "G 1\n2 1 4\n") { return Task::create(line_task_data(goal)); }

TEST(IwSemantics, LineTaskByHand)
{
    const auto task = line_task();
    {
        const IwResult r = iw(*task, with_k(2));
        EXPECT_EQ(r.status, SearchStatus::Solved);
        ASSERT_EQ(r.passes.size(), 2u);
        // width 0: the root and its one successor are expanded; that successor's two successors are pruned
        EXPECT_EQ(r.passes[0].expanded, 2u);
        EXPECT_EQ(r.passes[0].generated, 3u);
        EXPECT_EQ(r.passes[0].generated_in_tree, 1u);
        EXPECT_EQ(r.passes[0].status, SearchStatus::Exhausted);
        // width 1: o0 .. o3 expanded (moving back adds only at(x), which was seen), o4 popped as the goal
        EXPECT_EQ(r.passes[1].expanded, 4u);
        EXPECT_EQ(r.passes[1].generated, 7u);
        EXPECT_EQ(r.passes[1].generated_in_tree, 4u);
        EXPECT_EQ(r.plan.size(), 4u);
        EXPECT_EQ(r.effective_width, 1u);
        EXPECT_EQ(r.cost, 4.0);
        EXPECT_EQ(r.total.expanded, 6u);
        EXPECT_EQ(r.total.generated, 10u);
        ASSERT_EQ(r.plan.size(), 4u);
        EXPECT_EQ(task->format(r.plan[3].label()), "(move o3 o4)");
    }
    {
        const IwResult r = iw(*task, with_k(1));  // optimized IW(1)
        ASSERT_EQ(r.passes.size(), 2u);
        EXPECT_TRUE(r.passes[0].placeholder);
        EXPECT_EQ(r.passes[1].expanded, 4u);
        EXPECT_EQ(r.passes[1].generated, 7u);
        EXPECT_EQ(r.passes[1].generated_in_tree, 4u);
        EXPECT_EQ(r.passes[1].skipped, 0u);
    }
    {
        IwOptions o = with_k(1);
        o.optimize_iw1 = false;
        const IwResult r = iw(*task, o);
        ASSERT_EQ(r.passes.size(), 2u);
        EXPECT_FALSE(r.passes[0].placeholder);
        EXPECT_EQ(r.passes[0].expanded, 2u);
    }
    {
        IwOptions o = with_k(0);
        o.width_zero = WidthZero::RootOnly;
        const IwResult r = iw(*task, o);
        ASSERT_EQ(r.passes.size(), 1u);
        EXPECT_EQ(r.status, SearchStatus::Exhausted);
        EXPECT_EQ(r.passes[0].expanded, 2u);  // the root, and its successor: popped, counted, not expanded
        EXPECT_EQ(r.passes[0].skipped, 1u);
        EXPECT_EQ(r.passes[0].generated, 1u);
    }
    {
        // SIW: one subproblem (one goal literal)
        const SiwResult s = siw(*task, with_k(2));
        EXPECT_EQ(s.status, SearchStatus::Solved);
        ASSERT_EQ(s.subproblems.size(), 1u);
        EXPECT_EQ(s.subproblems[0].unsatisfied_at_start, 1u);
        EXPECT_EQ(s.subproblems[0].effective_width, 1u);
        EXPECT_EQ(s.plan.size(), 4u);
        EXPECT_EQ(s.max_effective_width, 1u);
    }
}

TEST(IwSemantics, PlanCostIsTheForks)
{
    // two schemas with the same precondition and effects: move costs 3, walk1 costs 1 (total-cost effects), and the
    // problem's total-cost starts at 0.5. The tree keeps the first admitting action (move, canonical order); the fork
    // takes the cheapest action between two plan states (search_space.hpp extract_total_ordered_plan): walk1.
    formalism::TaskData d = line_task_data("G 1\n2 1 4\n", 2);
    formalism::Function tc;
    tc.name = d.intern_string("total-cost");
    tc.kind = formalism::FuncKind::Auxiliary;
    d.functions.push_back(tc);
    d.auxiliary_initial = 0.5;
    const double costs[] = {3.0, 1.0};
    for (u32 s = 0; s < 2; ++s)
    {
        formalism::Expr e;
        e.op = formalism::ExprOp::Number;
        e.value = costs[s];
        d.exprs.push_back(e);
        formalism::NumericEffect ne;
        ne.op = formalism::AssignOp::Increase;
        ne.func = FunctionId{0};
        ne.expr = static_cast<u32>(d.exprs.size() - 1);
        d.conditional_effects[d.schemas[s].effects.begin].auxiliary = ne;
    }
    const auto task = Task::create(std::move(d));
    const IwResult r = iw(*task, with_k(2));
    ASSERT_EQ(r.status, SearchStatus::Solved);
    ASSERT_EQ(r.plan.size(), 4u);
    EXPECT_DOUBLE_EQ(r.cost, 4.5);
    EXPECT_TRUE(r.cost_exact);
    for (const Action& a : r.plan)
        EXPECT_EQ(a.schema.v, 1u);
    expect_valid_plan(*task, task->initial_state(), r.plan, task_goal(*task));
    const SiwResult s = siw(*task, with_k(2));
    ASSERT_EQ(s.status, SearchStatus::Solved);
    EXPECT_DOUBLE_EQ(s.cost, 4.5);
    // without a total-cost function a step costs 1
    const IwResult u = iw(*line_task(), with_k(2));
    EXPECT_DOUBLE_EQ(u.cost, 4.0);
    // unsolved: 0
    EXPECT_EQ(iw(*task, with_k(0)).cost, 0.0);
}

TEST(IwSemantics, UnsolvableStaticGoal)
{
    const auto task = line_task("G 1\n0 1 0 4\n");  // adj(o0, o4) is false and static
    const IwResult r = iw(*task, with_k(2));
    EXPECT_EQ(r.status, SearchStatus::Unsolvable);
    ASSERT_EQ(r.passes.size(), 1u);
    EXPECT_EQ(r.passes[0].expanded, 0u);
    const IwResult r1 = iw(*task, with_k(1));
    EXPECT_EQ(r1.status, SearchStatus::Unsolvable);
    ASSERT_EQ(r1.passes.size(), 2u);  // the placeholder, then the pass that stopped
    EXPECT_EQ(siw(*task, with_k(2)).status, SearchStatus::Unsolvable);
    EXPECT_STREQ(mimir_status_name(SearchStatus::Unsolvable), "unsolvable");
    EXPECT_STREQ(mimir_status_name(SearchStatus::Exhausted), "failed");
}

TEST(IwSemantics, InitialGoalAndTooLargeArity)
{
    const auto task = line_task("G 1\n2 1 0\n");  // visited(o0) holds initially
    const IwResult r = iw(*task, with_k(2));
    EXPECT_EQ(r.status, SearchStatus::Solved);
    EXPECT_TRUE(r.plan.empty());
    ASSERT_EQ(r.passes.size(), 1u);
    EXPECT_EQ(r.passes[0].expanded, 0u);
    const SiwResult s = siw(*task, with_k(2));
    EXPECT_EQ(s.status, SearchStatus::Solved);
    EXPECT_TRUE(s.subproblems.empty());
    const IwResult bad = iw(*task, with_k(novelty::k_max_arity + 1));
    EXPECT_EQ(bad.status, SearchStatus::Failed);
    EXPECT_FALSE(bad.message.empty());
}

void expect_valid_plan(const Task& task, const State& start, const std::vector<Action>& plan, const std::function<bool(const State&)>& goal)
{
    const WorkspaceLease lease = task.workspace();
    Successors& succ = lease->successors();
    State s = start;
    for (const Action& a : plan)
    {
        ASSERT_TRUE(succ.is_applicable(s, a.label())) << task.format(a.label());
        s = succ.apply(s, a.label());
    }
    EXPECT_TRUE(goal(s));
}

TEST(IwSemantics, PlansReachTheGoal)
{
    for (const char* name : {"freecell__p02", "pathways__p02", "organic-synthesis-opt18-strips__p20"})
    {
        const auto task = Task::from_text_file(task_path(name));
        const IwResult r = iw(*task, with_k(2));
        ASSERT_EQ(r.status, SearchStatus::Solved) << name;
        expect_valid_plan(*task, task->initial_state(), r.plan, task_goal(*task));
        EXPECT_EQ(r.cost, static_cast<double>(r.plan.size()));
    }
}

// ------------------------------------------------------------------------------------------------- control
class Recorder final : public SearchObserver
{
public:
    u64 starts = 0, expands = 0, generates = 0, news = 0, prunes = 0, passes = 0, solutions = 0, ends = 0, progress = 0;
    u64 stop_after = ~u64{0};
    std::vector<State> children;  // admitted children
    SearchStatus end_status = SearchStatus::Failed;
    void on_start(StateView) override { ++starts; }
    void on_expand(u64, StateView) override { ++expands; }
    void on_generate(u64, const Action&, u64 child, StateView s, bool is_new) override
    {
        ++generates;
        if (is_new)
        {
            ++news;
            EXPECT_NE(child, ~u64{0});
            children.emplace_back(s);
        }
        else
            EXPECT_EQ(child, ~u64{0});
    }
    void on_prune(u64, const Action&, StateView) override { ++prunes; }
    void on_pass(u32, const SearchStatistics&) override { ++passes; }
    void on_solution(std::span<const Action>, double) override { ++solutions; }
    bool on_progress(const SearchStatistics&) override { return ++progress < stop_after; }
    void on_end(SearchStatus s, const SearchStatistics&) override
    {
        ++ends;
        end_status = s;
    }
};

TEST(IwControl, ObserverEventsMatchTheStatistics)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "freecell__p02", "philosophers__p03-phil4"})
    {
        const auto task = Task::from_text_file(task_path(name));
        const IwResult plain = iw(*task, with_k(2));
        Recorder rec;
        IwOptions o = with_k(2);
        o.control.observer = &rec;
        const IwResult r = iw(*task, o);
        ASSERT_EQ(r.passes.size(), plain.passes.size()) << name;
        for (usize i = 0; i < r.passes.size(); ++i)
        {
            EXPECT_EQ(r.passes[i].expanded, plain.passes[i].expanded);
            EXPECT_EQ(r.passes[i].generated, plain.passes[i].generated);
            EXPECT_EQ(r.passes[i].generated_in_tree, plain.passes[i].generated_in_tree);
        }
        EXPECT_EQ(rec.starts, 1u);
        EXPECT_EQ(rec.ends, 1u);
        EXPECT_EQ(rec.end_status, r.status);
        EXPECT_EQ(rec.passes, r.passes.size());
        EXPECT_EQ(rec.expands, r.total.expanded);
        EXPECT_EQ(rec.generates, r.total.generated);
        EXPECT_EQ(rec.news, r.total.generated - r.total.pruned);
        EXPECT_EQ(rec.prunes, r.total.pruned);
        EXPECT_EQ(rec.solutions, r.status == SearchStatus::Solved ? 1u : 0u);
    }
}

TEST(IwControl, Budgets)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    {
        IwOptions o = with_k(2);
        o.control.budget.max_states = 100;  // per pass: the root plus 99 admitted states
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::OutOfStates);
        ASSERT_EQ(r.passes.size(), 3u);  // widths 0 and 1 stay below 100 states
        EXPECT_EQ(r.passes[2].generated_in_tree, 99u);
    }
    {
        IwOptions o = with_k(2);
        o.control.budget.max_expanded = 10;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::OutOfStates);
        EXPECT_EQ(r.passes.back().expanded, 10u);
    }
    {
        IwOptions o = with_k(2);
        o.control.budget.max_seconds = 0;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::OutOfTime);
        EXPECT_TRUE(r.passes.empty());
    }
    {
        IwOptions o = with_k(2);
        o.control.budget.max_depth = 1;
        RefSpec sp;
        sp.k = 2;
        sp.max_depth = 1;
        const IwResult r = iw(*task, o);
        const RefPass ref = ref_pass(*task, task->initial_state(), sp, task_goal(*task));
        expect_pass(r.passes[2], ref, "max_depth 1");
        EXPECT_GT(r.passes[2].skipped, 0u);
    }
}

TEST(IwControl, Cancellation)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    {
        IwOptions o = with_k(2);
        o.control.cancel.request();
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Cancelled);
        EXPECT_TRUE(r.passes.empty());
        EXPECT_EQ(siw(*task, o).status, SearchStatus::Cancelled);
    }
    {
        Recorder rec;
        rec.stop_after = 3;
        IwOptions o = with_k(2);
        o.control.observer = &rec;
        o.control.progress_interval = 7;
        const IwResult r = iw(*task, o);
        EXPECT_EQ(r.status, SearchStatus::Cancelled);
        EXPECT_EQ(rec.end_status, SearchStatus::Cancelled);
        // width 0 expands 5 states (no call); width 1 calls on_progress after 7, 14 and 21 of its expansions
        EXPECT_EQ(rec.progress, 3u);
        ASSERT_EQ(r.passes.size(), 2u);
        EXPECT_EQ(r.passes[0].expanded, 5u);
        EXPECT_EQ(r.passes[1].expanded, 21u);
        EXPECT_EQ(r.passes[1].status, SearchStatus::Cancelled);
    }
    {
        // cancelled from another thread while running
        const auto big = Task::from_text_file(task_path("snake-opt18-strips__p05"));
        IwOptions o = with_k(3);
        std::atomic<bool> go{false};
        std::thread t(
            [&]
            {
                while (!go.load())
                    std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                o.control.cancel.request();
            });
        go.store(true);
        const IwResult r = iw(*big, o);
        t.join();
        EXPECT_TRUE(r.status == SearchStatus::Cancelled || r.status == SearchStatus::Exhausted || r.status == SearchStatus::Solved);
    }
}

TEST(IwControl, BlockedStatesAreNeverEntered)
{
    for (const char* name : {"blocks__probBLOCKS-8-0", "gripper__prob05", "freecell__p02"})
    {
        const auto task = Task::from_text_file(task_path(name));
        // block every third state the plain width-2 pass admits (and, for freecell, the goal state)
        RefSpec sp;
        sp.k = 2;
        const RefPass plain = ref_pass(*task, task->initial_state(), sp, task_goal(*task));
        for (usize i = 0; i < plain.admitted.size(); i += 3)
            sp.blocked.push_back(plain.admitted[i]);
        if (plain.status == SearchStatus::Solved)
            sp.blocked.push_back(plain.goal_state);
        Recorder rec;
        IwOptions o = with_k(2);
        o.control.blocked_states = sp.blocked;
        o.control.observer = &rec;
        const IwResult r = iw(*task, o);
        const std::unordered_set<State> blocked(sp.blocked.begin(), sp.blocked.end());
        for (const State& c : rec.children)
            EXPECT_FALSE(blocked.count(c)) << name;
        for (u32 k = 0; k < r.passes.size(); ++k)
        {
            sp.k = k;
            const RefPass ref = ref_pass(*task, task->initial_state(), sp, task_goal(*task));
            expect_pass(r.passes[k], ref, std::string(name) + " arity " + std::to_string(k));
            EXPECT_GT(r.passes[k].blocked, 0u);  // the first state blocked is a successor of the root
        }
        if (r.status == SearchStatus::Solved)
        {
            expect_valid_plan(*task, task->initial_state(), r.plan, task_goal(*task));
            EXPECT_FALSE(blocked.count(*r.goal_state));
        }
        // the optimized IW(1) root rule respects them too
        o.max_arity = 1;
        o.control.observer = nullptr;
        const IwResult r1 = iw(*task, o);
        sp.k = 1;
        sp.root_continuation = true;
        ASSERT_EQ(r1.passes.size(), 2u);
        expect_pass(r1.passes[1], ref_pass(*task, task->initial_state(), sp, task_goal(*task)), std::string(name) + " optimized");
    }
}

TEST(IwControl, GoalSpecs)
{
    const auto task = Task::from_text_file(task_path("blocks__probBLOCKS-8-0"));
    // a state deep in the width-1 tree, and one of its atoms that the start state lacks
    RefSpec sp;
    sp.k = 1;
    const RefPass plain = ref_pass(*task, task->initial_state(), sp, [](const State&) { return false; });
    ASSERT_GT(plain.admitted.size(), 10u);
    const State target = plain.admitted[plain.admitted.size() - 3];
    SlotId atom{};
    for (SlotId x : target.slots())
        if (!task->initial_state().contains(x))
            atom = x;
    ASSERT_NE(atom.v, SlotId{}.v);
    auto has_atom = [atom](const State& s) { return s.contains(atom); };
    // AnyOf: an unreachable atom goal (a slot no state has), or the target atom
    GoalSpec any;
    any.kind = GoalSpec::Kind::AnyOf;
    any.goals.push_back({.positive = {SlotId{task->atoms().max_fluent_slots() + 5}}});
    any.goals.push_back({.positive = {atom}});
    IwOptions o = with_k(2);
    o.control.goal = any;
    const IwResult r = iw(*task, o);
    ASSERT_EQ(r.status, SearchStatus::Solved);
    expect_valid_plan(*task, task->initial_state(), r.plan, has_atom);
    for (u32 k = 0; k < r.passes.size(); ++k)
    {
        sp.k = k;
        expect_pass(r.passes[k], ref_pass(*task, task->initial_state(), sp, has_atom), "anyof arity " + std::to_string(k));
    }
    // Custom: the same test as a function
    GoalSpec custom;
    custom.kind = GoalSpec::Kind::Custom;
    custom.test = [atom](StateView s) { return s.contains(atom); };
    o.control.goal = custom;
    const IwResult c = iw(*task, o);
    ASSERT_EQ(c.status, SearchStatus::Solved);
    EXPECT_EQ(c.plan, r.plan);
    ASSERT_EQ(c.passes.size(), r.passes.size());
    for (usize i = 0; i < c.passes.size(); ++i)
        EXPECT_EQ(c.passes[i].expanded, r.passes[i].expanded);
    // a negative atom goal: the start state's first atom false
    GoalSpec neg;
    neg.kind = GoalSpec::Kind::AnyOf;
    neg.goals.push_back({.negative = {task->initial_state().slots().front()}});
    o.control.goal = neg;
    const IwResult n = iw(*task, o);
    ASSERT_EQ(n.status, SearchStatus::Solved);
    EXPECT_FALSE(n.goal_state->contains(task->initial_state().slots().front()));
}

TEST(IwControl, StartStateAndSinglePass)
{
    const auto task = Task::from_text_file(task_path("gripper__prob05"));
    const WorkspaceLease lease = task->workspace();
    Successors& succ = lease->successors();
    const std::vector<Action> acts = succ.applicable_actions(task->initial_state());
    ASSERT_FALSE(acts.empty());
    const State start = succ.apply(task->initial_state(), acts.back().label());
    IwOptions o = with_k(2);
    o.start = start;
    const IwResult r = iw(*task, o);
    for (u32 k = 0; k <= 2; ++k)
    {
        RefSpec sp;
        sp.k = k;
        expect_pass(r.passes[k], ref_pass(*task, start, sp, task_goal(*task)), "start arity " + std::to_string(k));
        // the ladder's pass k equals the single pass k
        const IwResult one = iw_pass(*task, k, o);
        ASSERT_EQ(one.passes.size(), 1u);
        EXPECT_EQ(one.passes[0].expanded, r.passes[k].expanded);
        EXPECT_EQ(one.passes[0].generated, r.passes[k].generated);
        EXPECT_EQ(one.passes[0].generated_in_tree, r.passes[k].generated_in_tree);
    }
}

TEST(IwControl, ConcurrentSearchesOnOneTask)
{
    const auto task = Task::from_text_file(task_path("zenotravel__p05"), lazy_atoms());
    const IwResult ref = iw(*task, with_k(2));
    std::vector<IwResult> out(8);
    std::vector<std::thread> ts;
    for (usize i = 0; i < out.size(); ++i)
        ts.emplace_back([&, i] { out[i] = iw(*task, with_k(2)); });
    for (auto& t : ts)
        t.join();
    for (const IwResult& r : out)
    {
        ASSERT_EQ(r.passes.size(), ref.passes.size());
        for (usize i = 0; i < r.passes.size(); ++i)
        {
            EXPECT_EQ(r.passes[i].expanded, ref.passes[i].expanded);
            EXPECT_EQ(r.passes[i].generated, ref.passes[i].generated);
            EXPECT_EQ(r.passes[i].generated_in_tree, ref.passes[i].generated_in_tree);
        }
    }
}

// ------------------------------------------------------------------------------------------------- SIW
/// Unsatisfied goal literals of the task (fluent goals only; static literals are folded), each literal once.
u32 unsatisfied(const Task& task, const State& s)
{
    const formalism::TaskData& t = task.data();
    std::set<std::pair<std::vector<u32>, bool>> lits;
    for (const formalism::Literal& l : t.literals_of(t.goal))
    {
        if (t.predicates[l.pred.v].kind != formalism::PredKind::Fluent)
            continue;
        std::vector<u32> key{l.pred.v};
        for (formalism::Term x : t.terms_of(l))
            key.push_back(formalism::term_object(x).v);
        lits.insert({key, l.positive});
    }
    u32 c = 0;
    for (const auto& [key, pos] : lits)
    {
        std::vector<ObjectId> args;
        for (usize i = 1; i < key.size(); ++i)
            args.push_back(ObjectId{key[i]});
        const SlotId slot = task.find_atom(PredicateId{key[0]}, args);
        c += s.contains(slot) != pos;
    }
    return c;
}

TEST(Siw, EqualsAReferenceOnTheOracle)
{
    for (const auto& st : suite())
    {
        const auto task = Task::from_text_file(task_path(st.name));
        if (task->compiled().goal.uses_derived)
            continue;
        for (u32 k : {1u, 2u})
        {
            SiwOptions o = with_k(k);
            o.control.budget.max_expanded = 4000;
            const SiwResult r = siw(*task, o);
            // the reference: IW ladders with the goal counter, from the state each subproblem reached
            State cur = task->initial_state();
            usize sub = 0;
            u32 plan_length = 0;
            SearchStatus status = SearchStatus::Solved;
            while (!task->is_goal(cur))
            {
                const u32 h0 = unsatisfied(*task, cur);
                auto counter = [&](const State& s) { return unsatisfied(*task, s) < h0; };
                ASSERT_LT(sub, r.subproblems.size()) << st.name << " k=" << k;
                const SiwSubproblem& sp = r.subproblems[sub];
                EXPECT_EQ(sp.unsatisfied_at_start, h0);
                const bool optimized = k == 1;
                SearchStatus sub_status = SearchStatus::Exhausted;
                RefPass solved;
                usize at = optimized ? 1 : 0;
                for (u32 w = optimized ? 1 : 0; w <= k; ++w, ++at)
                {
                    RefSpec rs;
                    rs.k = w;
                    rs.root_continuation = optimized;
                    rs.max_expanded = 4000;
                    const RefPass ref = ref_pass(*task, cur, rs, counter);
                    ASSERT_LT(at, sp.passes.size()) << st.name;
                    expect_pass(sp.passes[at], ref, st.name + " k=" + std::to_string(k) + " sub " + std::to_string(sub));
                    if (ref.status != SearchStatus::Exhausted)
                    {
                        sub_status = ref.status;
                        solved = ref;
                        break;
                    }
                }
                EXPECT_EQ(sp.status, sub_status) << st.name;
                if (sub_status != SearchStatus::Solved)
                {
                    status = sub_status;
                    break;
                }
                plan_length += solved.plan_length;
                cur = solved.goal_state;
                ++sub;
            }
            EXPECT_EQ(r.status, status) << st.name << " k=" << k;
            if (status == SearchStatus::Solved)
            {
                EXPECT_EQ(r.subproblems.size(), sub);
                EXPECT_EQ(r.plan.size(), plan_length);
                expect_valid_plan(*task, task->initial_state(), r.plan, task_goal(*task));
            }
        }
    }
}
}  // namespace
