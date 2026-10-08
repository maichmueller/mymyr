// Datasets: state spaces against the fork's StateSpace on the fork's own test instances (counts, V*, and the
// content, transition and V* fingerprints of the fork's own state-space tool), ids and arrays independent of
// the thread count, the array invariants, the options (max_states, remove_if_unsolvable, statically false goals,
// timeouts), transition costs and numeric tasks, the instance pool, generalized state spaces, samplers, object graphs
// and certificates.

#include "../frontend/golden.hpp"
#include "../support/suite.hpp"
#include "mymyr/datasets/certificates.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/object_graph.hpp"
#include "mymyr/datasets/sampler.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace mymyr;
using namespace mymyr::datasets;

namespace
{
// ----------------------------------------------------------------------------------------------- fingerprints
// The fork_datasets definitions (FNV-1a 64 over bytes; u64 as 8 little-endian bytes).
struct Fnv
{
    u64 h = 0xcbf29ce484222325ULL;
    void bytes(const void* p, usize n)
    {
        const auto* b = static_cast<const unsigned char*>(p);
        for (usize i = 0; i < n; ++i)
        {
            h ^= b[i];
            h *= 0x100000001b3ULL;
        }
    }
    void str(const std::string& s) { bytes(s.data(), s.size()); }
    void u64le(u64 x)
    {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i)
            b[i] = static_cast<unsigned char>(x >> (8 * i));
        bytes(b, 8);
    }
    template<class T>
    void span(std::span<const T> s)
    {
        u64le(s.size());
        bytes(s.data(), s.size_bytes());
    }
};
u64 fnv_str(const std::string& s)
{
    Fnv f;
    f.str(s);
    return f.h;
}
u64 fnv_list(std::vector<u64> v)
{
    std::sort(v.begin(), v.end());
    Fnv f;
    for (u64 x : v)
        f.u64le(x);
    return f.h;
}

u64 state_hash(const Task& task, StateView s)
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
    return fnv_str(key);
}

struct Fingerprints
{
    u64 content, transitions, vstar;
};
Fingerprints fingerprints(const StateSpace& S)
{
    const Task& task = *S.task();
    std::vector<u64> hs(S.num_states());
    for (u32 v = 0; v < S.num_states(); ++v)
        hs[v] = state_hash(task, S.state(v));
    std::vector<u64> trans, vst;
    const auto off = S.forward_offsets();
    const auto tgt = S.forward_targets();
    for (u32 v = 0; v < S.num_states(); ++v)
    {
        for (u64 e = off[v]; e < off[v + 1]; ++e)
        {
            Fnv f;
            f.u64le(hs[v]);
            f.u64le(fnv_str(task.format(S.label(e).label())));
            f.u64le(hs[tgt[e]]);
            trans.push_back(f.h);
        }
        Fnv f;
        f.u64le(hs[v]);
        const i32 d = S.unit_goal_distances()[v];
        f.u64le(static_cast<u64>(static_cast<i64>(d == k_unsolvable_distance ? std::numeric_limits<i32>::max() : d)));
        vst.push_back(f.h);
    }
    return {fnv_list(hs), fnv_list(trans), fnv_list(vst)};
}

/// FNV over every array of a space (mymyr_datasets --hash).
u64 arrays_hash(const StateSpace& S, bool with_states = true)
{
    Fnv f;
    f.u64le(S.num_states());
    if (with_states)
    {
        f.u64le(S.row_words());
        f.span(S.state_words());
    }
    f.span(S.forward_offsets());
    f.span(S.forward_targets());
    f.span(S.label_schemas());
    f.span(S.label_bindings());
    f.span(S.costs());
    f.span(S.backward_offsets());
    f.span(S.backward_sources());
    f.span(S.backward_edges());
    f.span(S.unit_goal_distances());
    f.span(S.cost_goal_distances());
    f.span(S.goal_flags());
    f.span(S.unsolvable_flags());
    f.span(S.alive_flags());
    return f.h;
}

// ----------------------------------------------------------------------------------------------- tasks
std::filesystem::path data(const std::string& rel) { return test::fork_data_dir() / rel; }

TaskPtr fork_task(const std::string& dir, const std::string& problem, TaskOptions to = {})
{
    const auto d = data(dir + "/domain.pddl"), p = data(dir + "/" + problem);
    if (!std::filesystem::exists(d) || !std::filesystem::exists(p))
        return nullptr;
    if (to.atoms == TaskOptions::Atoms::Auto)
        to.atoms = TaskOptions::Atoms::Frozen;
    return Task::create(*frontend::load_task(d, p), to);
}

TaskPtr pddl_task(const char* domain, const char* problem)
{
    TaskOptions to;
    to.atoms = TaskOptions::Atoms::Frozen;
    return Task::create(*frontend::Domain::from_string(domain, "d.pddl")->instantiate_string(problem, "p.pddl"), to);
}

StateSpaceOptions fork_ss(u32 threads = 1)
{
    StateSpaceOptions o;
    o.threads = threads;
    o.remove_if_unsolvable = false;
    return o;
}

StateSpacePtr space_of(TaskPtr task, const StateSpaceOptions& o)
{
    StateSpaceResult r = generate_state_space(std::move(task), o);
    EXPECT_EQ(r.status, StateSpaceStatus::Ok);
    return r.space;
}

// ----------------------------------------------------------------------------------------------- invariants
/// Checks the arrays against each other and against an independent search over the task.
void check_invariants(const StateSpace& S)
{
    const u32 N = S.num_states();
    const u64 E = S.num_transitions();
    const auto off = S.forward_offsets();
    const auto tgt = S.forward_targets();
    ASSERT_EQ(off.size(), N + 1u);
    ASSERT_EQ(off[N], E);
    // reverse CSR: exactly the forward edges, grouped by target, ascending edge indices
    const auto boff = S.backward_offsets();
    const auto bsrc = S.backward_sources();
    const auto bedg = S.backward_edges();
    ASSERT_EQ(boff.size(), N + 1u);
    ASSERT_EQ(boff[N], E);
    std::vector<u8> seen(E, 0);
    for (u32 t = 0; t < N; ++t)
        for (u64 r = boff[t]; r < boff[t + 1]; ++r)
        {
            const u32 e = bedg[r];
            ASSERT_LT(e, E);
            EXPECT_EQ(tgt[e], t);
            EXPECT_EQ(S.source(e), bsrc[r]);
            EXPECT_TRUE(e >= off[bsrc[r]] && e < off[bsrc[r] + 1]);
            if (r > boff[t])
            {
                EXPECT_LT(bedg[r - 1], e);
            }
            EXPECT_FALSE(seen[e]);
            seen[e] = 1;
        }
    // goal distances: an independent backward BFS
    std::vector<i32> d(N, k_unsolvable_distance);
    std::deque<u32> q;
    for (u32 v = 0; v < N; ++v)
        if (S.is_goal(v))
        {
            d[v] = 0;
            q.push_back(v);
        }
    std::vector<std::vector<u32>> pred(N);
    for (u32 v = 0; v < N; ++v)
        for (u64 e = off[v]; e < off[v + 1]; ++e)
            pred[tgt[e]].push_back(v);
    while (!q.empty())
    {
        const u32 v = q.front();
        q.pop_front();
        for (u32 u : pred[v])
            if (d[u] == k_unsolvable_distance)
            {
                d[u] = d[v] + 1;
                q.push_back(u);
            }
    }
    i32 max = -1;
    u32 goals = 0, unsolvable = 0;
    for (u32 v = 0; v < N; ++v)
    {
        EXPECT_EQ(S.unit_goal_distances()[v], d[v]) << v;
        EXPECT_EQ(S.is_unsolvable(v), d[v] == k_unsolvable_distance);
        EXPECT_EQ(S.is_alive(v), !S.is_goal(v) && !S.is_unsolvable(v));
        goals += S.is_goal(v);
        unsolvable += S.is_unsolvable(v);
        if (d[v] != k_unsolvable_distance)
            max = std::max(max, d[v]);
        // cost distances satisfy the Bellman equations
        const f64 c = S.cost_goal_distances()[v];
        if (S.is_goal(v))
        {
            EXPECT_EQ(c, 0.0);
        }
        else
        {
            f64 best = std::numeric_limits<f64>::infinity();
            for (u64 e = off[v]; e < off[v + 1]; ++e)
                best = std::min(best, S.cost(e) + S.cost_goal_distances()[tgt[e]]);
            EXPECT_DOUBLE_EQ(c, best) << v;
        }
    }
    EXPECT_EQ(S.num_goal_states(), goals);
    EXPECT_EQ(S.num_unsolvable_states(), unsolvable);
    EXPECT_EQ(S.max_goal_distance(), max);
    // states are distinct and find() maps them back
    for (u32 v = 0; v < N; ++v)
        EXPECT_EQ(S.find(S.state(v)), static_cast<i64>(v));
    // each row is the task's canonical successor sequence of its state (labels and targets)
    if (S.has_labels() && !S.symmetry_reduced())
    {
        const Task& task = *S.task();
        Successors& succ = task.workspace().successors();
        LineVector<u64> next;
        for (u32 v = 0; v < N; v += std::max<u32>(1, N / 200))
        {
            const State cur(S.state(v));
            succ.prepare(cur.view());
            EXPECT_EQ(succ.goal_holds(), S.is_goal(v));
            u64 e = off[v];
            succ.generate<false>(
                [&](u32 schema, const ObjectId* b, const Delta& dl)
                {
                    EXPECT_LT(e, off[v + 1]);
                    const Action a = S.label(e);
                    EXPECT_EQ(a.schema.v, schema);
                    for (usize i = 0; i < a.binding.size(); ++i)
                        EXPECT_EQ(a.binding[i], b[i]);
                    const u32 nn = apply_delta(cur.data(), cur.size_words(), dl, next);
                    const State t(next.data(), nn, dl.num, task.numeric_words());
                    EXPECT_EQ(S.find(t.view()), static_cast<i64>(tgt[e]));
                    ++e;
                    return true;
                },
                false, true);
            EXPECT_EQ(e, off[v + 1]);
        }
    }
}

// ----------------------------------------------------------------------------------------------- fork parity
struct ForkCase
{
    const char* dir;
    const char* problem;
    u32 states;
    u64 transitions;
    u32 goal, unsolvable;
    i32 max_goal_dist;
    u64 fp_content, fp_transitions, fp_vstar;
    f64 cost_sum;  // sum of the finite cost goal distances (the fork's action goal distances)
};

// fork_datasets ss lists/fork_data.txt --max-states 200000 (build/parity/fork_data.jsonl; the fork 0.16.3 configured
// as fork_ss: remove_if_unsolvable = false, no symmetry pruning)
const ForkCase kForkCases[] = {
#include "fork_cases.inc"
};

class ForkParity : public ::testing::TestWithParam<ForkCase>
{
};

TEST_P(ForkParity, MatchesTheFork)
{
    const ForkCase& c = GetParam();
    const TaskPtr task = fork_task(c.dir, c.problem);
    if (!task)
        GTEST_SKIP() << "fork data missing: " << data(c.dir);
    const StateSpacePtr S = space_of(task, fork_ss());
    ASSERT_TRUE(S);
    EXPECT_EQ(S->num_states(), c.states);
    EXPECT_EQ(S->num_transitions(), c.transitions);
    EXPECT_EQ(S->num_goal_states(), c.goal);
    EXPECT_EQ(S->num_unsolvable_states(), c.unsolvable);
    EXPECT_EQ(S->max_goal_distance(), c.max_goal_dist);
    const Fingerprints fp = fingerprints(*S);
    EXPECT_EQ(fp.content, c.fp_content);
    EXPECT_EQ(fp.transitions, c.fp_transitions);
    EXPECT_EQ(fp.vstar, c.fp_vstar);
    f64 cost_sum = 0;
    for (f64 x : S->cost_goal_distances())
        if (std::isfinite(x))
            cost_sum += x;
    EXPECT_DOUBLE_EQ(cost_sum, c.cost_sum);
    check_invariants(*S);
    // the layered generator on several threads: the same arrays
    const StateSpacePtr P = space_of(task, fork_ss(3));
    ASSERT_TRUE(P);
    EXPECT_EQ(arrays_hash(*P), arrays_hash(*S));
}

std::string case_name(const ::testing::TestParamInfo<ForkCase>& info)
{
    std::string n = std::string(info.param.dir) + "_" + info.param.problem;
    for (char& ch : n)
        if (!std::isalnum(static_cast<unsigned char>(ch)))
            ch = '_';
    return n;
}
INSTANTIATE_TEST_SUITE_P(ForkData, ForkParity, ::testing::ValuesIn(kForkCases), case_name);

// Numeric tasks (the fork's StateSpace generates them too): tests/data/pddl/counters, fork numbers from
// fork_datasets ss lists/numeric.txt. The state hashes cover the numeric values ("\n=%.17g" per variable).
TEST(StateSpace, NumericTasksMatchTheFork)
{
    struct Case
    {
        const char* problem;
        u32 states;
        u64 transitions;
        u32 goal, unsolvable;
        i32 max_goal_dist;
        u64 fp_content, fp_transitions, fp_vstar;
        f64 cost_sum;
    };
    const Case cases[] = {
        {"p01.pddl", 38, 108, 10, 0, 4, 0x2da00399d901c099ULL, 0xf345107ab284777cULL, 0xb86c738b18e1a06dULL, 59.0},
        {"p02.pddl", 129, 544, 0, 129, -1, 0x37338418a09264c5ULL, 0x0d275494cc07b57cULL, 0x3e585b03385b3749ULL, 0.0},
    };
    const std::string dir = std::string(MYMYR_TEST_DATA_DIR) + "/pddl/counters/";
    for (const Case& c : cases)
    {
        TaskOptions to;
        to.atoms = TaskOptions::Atoms::Frozen;
        const TaskPtr task = Task::create(*frontend::load_task(dir + "domain.pddl", dir + c.problem), to);
        ASSERT_GT(task->numeric_slots(), 0u);
        for (u32 T : {1u, 3u})
        {
            const StateSpacePtr S = space_of(task, fork_ss(T));
            ASSERT_TRUE(S);
            EXPECT_EQ(S->num_states(), c.states) << c.problem;
            EXPECT_EQ(S->num_transitions(), c.transitions);
            EXPECT_EQ(S->num_goal_states(), c.goal);
            EXPECT_EQ(S->num_unsolvable_states(), c.unsolvable);
            EXPECT_EQ(S->max_goal_distance(), c.max_goal_dist);
            const Fingerprints fp = fingerprints(*S);
            EXPECT_EQ(fp.content, c.fp_content) << c.problem;
            EXPECT_EQ(fp.transitions, c.fp_transitions);
            EXPECT_EQ(fp.vstar, c.fp_vstar);
            f64 cost_sum = 0;
            for (f64 x : S->cost_goal_distances())
                if (std::isfinite(x))
                    cost_sum += x;
            EXPECT_DOUBLE_EQ(cost_sum, c.cost_sum);
            check_invariants(*S);
        }
    }
}

// ----------------------------------------------------------------------------------------------- determinism
TEST(StateSpace, ArraysIndependentOfThreadCount)
{
    for (const char* name : {"depot__p02", "philosophers__p03-phil4"})
    {
        for (auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        {
            TaskOptions to;
            to.atoms = atoms;
            const auto task = Task::from_text_file(test::task_path(name), to);
            const StateSpacePtr ref = space_of(task, fork_ss(1));
            ASSERT_TRUE(ref);
            const bool frozen = atoms == TaskOptions::Atoms::Frozen;
            const u64 h = arrays_hash(*ref, frozen);  // lazy slots: the state words depend on the interning order
            for (u32 T : {2u, 3u, 8u})
            {
                const StateSpacePtr s = space_of(task, fork_ss(T));
                ASSERT_TRUE(s);
                EXPECT_EQ(arrays_hash(*s, frozen), h) << name << " T=" << T;
            }
            if (frozen)
                check_invariants(*ref);
        }
    }
}

TEST(StateSpace, CountsMatchTheBrfsSuite)
{
    for (const auto& t : test::suite())
    {
        if (t.states > 50000)
            continue;
        const auto task = Task::from_text_file(test::task_path(t.name));
        const StateSpacePtr s = space_of(task, fork_ss(2));
        ASSERT_TRUE(s);
        EXPECT_EQ(s->num_states(), t.states) << t.name;
        EXPECT_EQ(s->num_goal_states(), t.goal_states) << t.name;
    }
}

// ----------------------------------------------------------------------------------------------- options
TEST(StateSpace, MaxStatesFailsIffTheSpaceReachesIt)
{
    const TaskPtr task = fork_task("gripper", "test_problem4.pddl");  // 256 states
    if (!task)
        GTEST_SKIP();
    for (u32 T : {1u, 4u})
    {
        StateSpaceOptions o = fork_ss(T);
        o.max_states = 256;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::OutOfStates) << T;
        o.max_states = 257;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::Ok) << T;
        o.max_states = 10;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::OutOfStates) << T;
    }
    // the fork: a single-state space fits every limit (it fails iff N >= max(max_states, 2))
    const TaskPtr one = fork_task("deadend", "test_problem.pddl");
    ASSERT_TRUE(one);
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o = fork_ss(T);
        o.max_states = 0;
        const StateSpaceResult r = generate_state_space(one, o);
        ASSERT_EQ(r.status, StateSpaceStatus::Ok);
        EXPECT_EQ(r.space->num_states(), 1u);
    }
}

TEST(StateSpace, RemoveIfUnsolvableDropsTheWholeSpace)
{
    const TaskPtr task = fork_task("deadend", "test_problem.pddl");  // the initial state is a dead end
    if (!task)
        GTEST_SKIP();
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o;
        o.threads = T;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::Unsolvable);
        o.remove_if_unsolvable = false;
        const StateSpaceResult r = generate_state_space(task, o);
        ASSERT_EQ(r.status, StateSpaceStatus::Ok);
        EXPECT_EQ(r.space->num_unsolvable_states(), r.space->num_states());
        EXPECT_EQ(r.space->unit_goal_distances()[0], k_unsolvable_distance);
        EXPECT_TRUE(std::isinf(r.space->cost_goal_distances()[0]));
    }
    // a space whose initial state reaches a goal keeps its dead ends
    const TaskPtr spanner = fork_task("spanner", "p15-easy.pddl");
    ASSERT_TRUE(spanner);
    const StateSpaceResult r = generate_state_space(spanner, {});
    ASSERT_EQ(r.status, StateSpaceStatus::Ok);
    EXPECT_GT(r.space->num_unsolvable_states(), 0u);
}

const char* kCountDomain = R"(
(define (domain count)
 (:requirements :strips :typing :action-costs)
 (:types cell)
 (:predicates (at ?c - cell) (next ?a ?b - cell) (never))
 (:functions (total-cost) (step-cost ?a ?b - cell))
 (:action step :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) (step-cost ?a ?b))))
 (:action jump :parameters (?a ?b - cell) :precondition (and (at ?a) (next ?a ?b))
   :effect (and (not (at ?a)) (at ?b) (increase (total-cost) 10)))
)
)";

TEST(StateSpace, StaticallyFalseGoalHasNoSpace)
{
    const TaskPtr task = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 - cell)
 (:init (at c0) (next c0 c1) (= (total-cost) 0) (= (step-cost c0 c1) 1))
 (:goal (and (at c1) (never)))
 (:metric minimize (total-cost)))
)");
    for (bool remove : {false, true})
    {
        StateSpaceOptions o;
        o.remove_if_unsolvable = remove;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::Unsolvable);
        o.threads = 2;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::Unsolvable);
    }
}

TEST(StateSpace, TransitionCostsAndCostDistances)
{
    // c0 -> c1 -> c2 -> c3 (goal), step costs 1, 2, 3, jumps cost 10; c0 -> c3 directly for 4
    const TaskPtr task = pddl_task(kCountDomain, R"(
(define (problem p) (:domain count)
 (:objects c0 c1 c2 c3 - cell)
 (:init (at c0) (next c0 c1) (next c1 c2) (next c2 c3) (next c0 c3) (= (total-cost) 0)
        (= (step-cost c0 c1) 1) (= (step-cost c1 c2) 2) (= (step-cost c2 c3) 3) (= (step-cost c0 c3) 7))
 (:goal (at c3))
 (:metric minimize (total-cost)))
)");
    for (u32 T : {1u, 2u})
    {
        const StateSpacePtr S = space_of(task, fork_ss(T));
        ASSERT_TRUE(S);
        ASSERT_EQ(S->num_states(), 4u);
        EXPECT_EQ(S->num_transitions(), 8u);
        ASSERT_FALSE(S->unit_costs());
        // state 0 = at c0: its transitions in canonical order (schema, then binding): step c0 c1, step c0 c3,
        // jump c0 c1, jump c0 c3
        std::vector<f64> row0(S->costs().begin(), S->costs().begin() + 4);
        EXPECT_EQ(row0, (std::vector<f64>{1, 7, 10, 10}));
        EXPECT_EQ(S->unit_goal_distances()[0], 1);
        EXPECT_DOUBLE_EQ(S->cost_goal_distances()[0], 6.0);  // 1 + 2 + 3 beats 7 and 10
        check_invariants(*S);
    }
    // without labels the arrays are the same except the labels
    StateSpaceOptions o = fork_ss();
    o.labels = false;
    const StateSpacePtr S = space_of(task, o);
    ASSERT_TRUE(S);
    EXPECT_FALSE(S->has_labels());
    EXPECT_TRUE(S->label_schemas().empty());
    EXPECT_THROW((void)S->label(0), std::logic_error);
}

const char* kNumericDomain = R"(
(define (domain counter)
 (:requirements :strips :numeric-fluents)
 (:predicates (done))
 (:functions (x) (fuel))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1) (decrease (fuel) 1)))
 (:action dec :parameters () :precondition (and (> (x) 0)) :effect (and (decrease (x) 1) (increase (fuel) 2)))
 (:action finish :parameters () :precondition (and (= (x) 3)) :effect (and (done)))
)
)";

TEST(StateSpace, NumericTasks)
{
    const TaskPtr task = pddl_task(kNumericDomain, R"(
(define (problem p) (:domain counter)
 (:init (= (x) 0) (= (fuel) 0))
 (:goal (done))
 (:metric minimize (fuel)))
)");
    ASSERT_GT(task->numeric_slots(), 0u);
    StateSpaceOptions o = fork_ss();
    o.max_states = 1000;  // fuel is unbounded: the space is infinite
    EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::OutOfStates);
    o.threads = 3;
    EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::OutOfStates);
    // a bounded counter: x in 0..3 with and without done
    const TaskPtr bounded = pddl_task(R"(
(define (domain counter2)
 (:requirements :strips :numeric-fluents :negative-preconditions)
 (:predicates (done))
 (:functions (x))
 (:action inc :parameters () :precondition (and (< (x) 3)) :effect (and (increase (x) 1)))
 (:action dec :parameters () :precondition (and (> (x) 0)) :effect (and (decrease (x) 1)))
 (:action finish :parameters () :precondition (and (= (x) 3) (not (done))) :effect (and (done)))
)
)",
                                      R"(
(define (problem p) (:domain counter2)
 (:init (= (x) 0))
 (:goal (and (done) (= (x) 0))))
)");
    for (u32 T : {1u, 2u})
    {
        const StateSpacePtr S = space_of(bounded, fork_ss(T));
        ASSERT_TRUE(S);
        EXPECT_EQ(S->num_states(), 8u);
        EXPECT_EQ(S->numeric_words(), bounded->numeric_words());
        EXPECT_EQ(S->num_goal_states(), 1u);
        EXPECT_EQ(S->unit_goal_distances()[0], 7);  // up 3, finish, down 3
        check_invariants(*S);
    }
}

TEST(StateSpace, MetricWithoutTotalCostTakesTheMetricDifference)
{
    // the metric (fuel) decreases along inc: negative transition costs, which Dijkstra rejects (as the fork's BGL)
    const TaskPtr task = pddl_task(R"(
(define (domain m)
 (:requirements :strips :numeric-fluents)
 (:predicates (a) (b))
 (:functions (fuel))
 (:action ab :parameters () :precondition (and (a)) :effect (and (not (a)) (b) (increase (fuel) 3)))
)
)",
                                   R"(
(define (problem p) (:domain m)
 (:init (a) (= (fuel) 1))
 (:goal (b))
 (:metric minimize (fuel)))
)");
    const StateSpacePtr S = space_of(task, fork_ss());
    ASSERT_TRUE(S);
    ASSERT_EQ(S->num_transitions(), 1u);
    EXPECT_DOUBLE_EQ(S->cost(0), 3.0);
    EXPECT_DOUBLE_EQ(S->cost_goal_distances()[0], 3.0);
    const TaskPtr negative = pddl_task(R"(
(define (domain m2)
 (:requirements :strips :numeric-fluents)
 (:predicates (a) (b))
 (:functions (fuel))
 (:action ab :parameters () :precondition (and (a)) :effect (and (not (a)) (b) (decrease (fuel) 3)))
)
)",
                                       R"(
(define (problem p) (:domain m2)
 (:init (a) (= (fuel) 1))
 (:goal (b))
 (:metric minimize (fuel)))
)");
    EXPECT_THROW((void)generate_state_space(negative, fork_ss()), std::domain_error);
}

TEST(StateSpace, TimeoutStops)
{
    const auto task = Task::from_text_file(test::task_path("depot__p02"));
    for (u32 T : {1u, 2u})
    {
        StateSpaceOptions o = fork_ss(T);
        o.max_seconds = 0;
        EXPECT_EQ(generate_state_space(task, o).status, StateSpaceStatus::Timeout) << T;
    }
}

// ----------------------------------------------------------------------------------------------- pool
TEST(StateSpace, PoolEqualsSingleGenerations)
{
    std::vector<TaskPtr> tasks;
    for (const char* dir : {"gripper", "spanner", "blocks_3", "miconic", "delivery", "visitall"})
        if (TaskPtr t = fork_task(dir, "test_problem.pddl"))
            tasks.push_back(t);
    if (tasks.size() < 6)
        GTEST_SKIP();
    const std::vector<StateSpaceResult> rs = generate_state_spaces(tasks, fork_ss(), 3);
    ASSERT_EQ(rs.size(), tasks.size());
    for (usize i = 0; i < tasks.size(); ++i)
    {
        ASSERT_TRUE(rs[i].space);
        EXPECT_EQ(rs[i].space->task(), tasks[i]);
        EXPECT_EQ(arrays_hash(*rs[i].space), arrays_hash(*space_of(tasks[i], fork_ss())));
    }
    // exceptions stop the pool and reach the caller
    EXPECT_THROW(for_each_state_space(
                     8, [&](u64 i) -> TaskPtr { if (i == 5) throw std::runtime_error("boom"); return tasks[i % tasks.size()]; },
                     [](u64, StateSpaceResult&&) {}, fork_ss(), 3),
                 std::runtime_error);
}

// ----------------------------------------------------------------------------------------------- samplers
TEST(Sampler, DeterministicPerSeedAndCorrectSupports)
{
    const TaskPtr task = fork_task("spanner", "p15-easy.pddl");
    if (!task)
        GTEST_SKIP();
    const StateSpacePtr S = space_of(task, fork_ss());
    ASSERT_TRUE(S);
    StateSpaceSampler a(S, 7), b(S, 7), c(S, 8);
    std::vector<u32> xa(500), xb(500), xc(500);
    a.sample_states(xa);
    b.sample_states(xb);
    c.sample_states(xc);
    EXPECT_EQ(xa, xb);
    EXPECT_NE(xa, xc);
    a.set_seed(7);
    std::vector<u32> again(500);
    a.sample_states(again);
    EXPECT_EQ(again, xa);
    EXPECT_EQ(a.num_states(), S->num_states());
    EXPECT_EQ(a.num_dead_end_states(), S->num_unsolvable_states());
    EXPECT_EQ(a.num_alive_states(), S->num_states() - S->num_unsolvable_states());
    EXPECT_EQ(static_cast<i32>(a.max_steps_to_goal()), S->max_goal_distance());
    for (i32 n = 0; n <= S->max_goal_distance(); ++n)
    {
        std::set<u32> hits;
        for (int i = 0; i < 2000; ++i)
        {
            const u32 s = a.sample_state_n_steps_from_goal(n);
            EXPECT_EQ(S->unit_goal_distances()[s], n);
            hits.insert(s);
        }
        const auto support = a.states_n_steps_from_goal(n);
        EXPECT_EQ(hits.size(), std::min<usize>(support.size(), hits.size()));
        EXPECT_TRUE(std::is_sorted(support.begin(), support.end()));
    }
    for (int i = 0; i < 1000; ++i)
        EXPECT_TRUE(S->is_unsolvable(a.sample_dead_end_state()));
    EXPECT_THROW((void)a.sample_state_n_steps_from_goal(S->max_goal_distance() + 1), std::out_of_range);
    EXPECT_THROW((void)a.sample_state_n_steps_from_goal(-1), std::out_of_range);
    // uniform over all states: every state of a small space shows up
    std::set<u32> all(xa.begin(), xa.end());
    std::vector<u32> many(20000);
    a.sample_states(many);
    all.insert(many.begin(), many.end());
    EXPECT_EQ(all.size(), S->num_states());
}

TEST(Sampler, NoDeadEnds)
{
    const TaskPtr task = fork_task("gripper", "p-2-0.pddl");
    if (!task)
        GTEST_SKIP();
    StateSpaceSampler s(space_of(task, {}), 1);
    EXPECT_EQ(s.num_dead_end_states(), 0u);
    EXPECT_THROW((void)s.sample_dead_end_state(), std::out_of_range);
}

// ----------------------------------------------------------------------------------------------- object graphs, certificates
ObjectGraph make_graph(u32 n, const std::vector<std::pair<u32, u32>>& edges, std::vector<u32> colors = {})
{
    ObjectGraph g;
    g.num_objects = n;
    g.color = colors.empty() ? std::vector<u32>(n, 0) : colors;
    const u32 k = *std::max_element(g.color.begin(), g.color.end()) + 1;
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

TEST(Certificates, WeisfeilerLemanHierarchy)
{
    // two triangles and a hexagon: 1-WL cannot tell them apart, 2-FWL can
    const ObjectGraph triangles = make_graph(6, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}});
    const ObjectGraph hexagon = make_graph(6, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}});
    EXPECT_EQ(color_refinement_certificate(triangles), color_refinement_certificate(hexagon));
    EXPECT_NE(kfwl_certificate(triangles, 2), kfwl_certificate(hexagon, 2));
    EXPECT_NE(kfwl_certificate(triangles, 3), kfwl_certificate(hexagon, 3));
    // isomorphic relabellings
    const ObjectGraph hexagon2 = make_graph(6, {{3, 0}, {0, 5}, {5, 1}, {1, 4}, {4, 2}, {2, 3}});
    EXPECT_EQ(color_refinement_certificate(hexagon), color_refinement_certificate(hexagon2));
    EXPECT_EQ(kfwl_certificate(hexagon, 2), kfwl_certificate(hexagon2, 2));
    EXPECT_EQ(kfwl_certificate(hexagon, 3), kfwl_certificate(hexagon2, 3));
    // colours matter, and isolated vertices of different colours are told apart (the fork's certificate is not)
    const ObjectGraph path_a = make_graph(3, {{0, 1}, {1, 2}}, {0, 1, 0});
    const ObjectGraph path_b = make_graph(3, {{0, 1}, {1, 2}}, {1, 0, 0});
    EXPECT_NE(color_refinement_certificate(path_a), color_refinement_certificate(path_b));
    EXPECT_NE(kfwl_certificate(path_a, 2), kfwl_certificate(path_b, 2));
    EXPECT_NE(color_refinement_certificate(make_graph(2, {}, {0, 1})), color_refinement_certificate(make_graph(2, {}, {0, 0})));
    std::vector<u32> stable;
    (void)color_refinement_certificate(path_a, &stable);
    ASSERT_EQ(stable.size(), 3u);
    EXPECT_EQ(stable[0], stable[2]);
    EXPECT_NE(stable[0], stable[1]);
    EXPECT_THROW((void)kfwl_certificate(hexagon, 5), std::invalid_argument);
    EXPECT_THROW((void)kfwl_certificate(hexagon, 1), std::invalid_argument);
}

TEST(ObjectGraph, GripperStructureAndClasses)
{
    const TaskPtr task = fork_task("gripper", "p-2-0.pddl");
    if (!task)
        GTEST_SKIP();
    const StateSpacePtr S = space_of(task, fork_ss());
    ASSERT_TRUE(S);
    ObjectGraphBuilder ogb(*task);
    // the fork's test (tests/unit/datasets/object_graph.cpp): 12 isomorphism classes among the 28 states
    std::set<Certificate> cr, fwl;
    for (u32 v = 0; v < S->num_states(); ++v)
    {
        const ObjectGraph g = ogb.build(S->state(v));
        EXPECT_EQ(g.num_objects, task->num_objects());
        EXPECT_EQ(g.offsets.size(), g.num_vertices() + 1u);
        for (u32 x = 0; x < g.num_vertices(); ++x)
            for (u32 y : g.adjacent(x))
            {
                EXPECT_NE(x, y);
                const auto back = g.adjacent(y);
                EXPECT_TRUE(std::binary_search(back.begin(), back.end(), x));
            }
        for (u32 c = 1; c < g.num_colors(); ++c)
            EXPECT_TRUE(std::lexicographical_compare(g.palette(c - 1).begin(), g.palette(c - 1).end(), g.palette(c).begin(),
                                                     g.palette(c).end()));
        cr.insert(color_refinement_certificate(g));
        fwl.insert(kfwl_certificate(g, 2));
    }
    EXPECT_EQ(cr.size(), 12u);
    EXPECT_EQ(fwl.size(), 12u);
}

// ----------------------------------------------------------------------------------------------- symmetry pruning
TEST(SymmetricStateSpace, OneStatePerClass)
{
    const TaskPtr task = fork_task("gripper", "p-2-0.pddl");
    if (!task)
        GTEST_SKIP();
    for (CertificateKind k : {CertificateKind::ColorRefinement, CertificateKind::KFwl})
    {
        StateSpaceOptions o = fork_ss();
        o.symmetry_pruning = true;
        o.certificate = k;
        const StateSpacePtr S = space_of(task, o);
        ASSERT_TRUE(S);
        EXPECT_TRUE(S->symmetry_reduced());
        EXPECT_EQ(S->num_states(), 12u);
        // no parallel edges; every class reached
        for (u32 v = 0; v < S->num_states(); ++v)
        {
            std::set<u32> t(S->forward_targets().begin() + static_cast<i64>(S->forward_offsets()[v]),
                            S->forward_targets().begin() + static_cast<i64>(S->forward_offsets()[v + 1]));
            EXPECT_EQ(t.size(), S->forward_offsets()[v + 1] - S->forward_offsets()[v]);
        }
        check_invariants(*S);
        // goal distances are those of the full space's corresponding states
        const StateSpacePtr F = space_of(task, fork_ss());
        EXPECT_EQ(S->max_goal_distance(), F->max_goal_distance());
    }
}

// ----------------------------------------------------------------------------------------------- generalized
TEST(GeneralizedStateSpace, ConcatenationAndSymmetryReduction)
{
    std::vector<TaskPtr> tasks;
    for (const char* p : {"test_problem4.pddl", "p-2-0.pddl", "test_problem2.pddl", "p-1-0.pddl"})
        if (TaskPtr t = fork_task("gripper", p))
            tasks.push_back(t);
    if (tasks.size() < 4)
        GTEST_SKIP();
    // without symmetry reduction: the disjoint union, spaces sorted by size (stable)
    const auto results = generate_state_spaces(tasks, fork_ss(), 2);
    const auto spaces = ordered_spaces(results);
    ASSERT_EQ(spaces.size(), 4u);
    for (usize i = 1; i < spaces.size(); ++i)
        EXPECT_LE(spaces[i - 1]->num_states(), spaces[i]->num_states());
    EXPECT_EQ(spaces[1]->task(), tasks[1]);  // p-2-0 and test_problem2 tie at 28 states: input order
    EXPECT_EQ(spaces[2]->task(), tasks[2]);
    const auto G = GeneralizedStateSpace::create(spaces);
    EXPECT_FALSE(G->symmetry_reduced());
    u64 V = 0, E = 0, goals = 0;
    for (const auto& s : spaces)
        V += s->num_states(), E += s->num_transitions(), goals += s->num_goal_states();
    EXPECT_EQ(G->num_vertices(), V);
    EXPECT_EQ(G->num_edges(), E);
    EXPECT_EQ(G->initial_vertices().size(), 4u);
    EXPECT_EQ(G->goal_vertices().size(), goals);
    for (u32 p = 0; p < 4; ++p)
        for (u32 v = 0; v < spaces[p]->num_states(); ++v)
        {
            const u32 c = G->vertex_mapping(p)[v];
            EXPECT_EQ(G->vertex_problems()[c], p);
            EXPECT_EQ(G->vertex_problem_vertices()[c], v);
        }
    EXPECT_EQ(G->forward_offsets().back(), G->num_edges());
    // with symmetry reduction: p-2-0 and test_problem2 are isomorphic (same initial state class): one is dropped
    StateSpaceOptions o = fork_ss();
    o.symmetry_pruning = true;
    const auto sym = GeneralizedStateSpace::create(ordered_spaces(generate_state_spaces(tasks, o, 2)));
    EXPECT_TRUE(sym->symmetry_reduced());
    EXPECT_LT(sym->spaces().size(), 4u);
    for (u32 p = 0; p < sym->spaces().size(); ++p)
        for (u32 v = 0; v < sym->spaces()[p]->num_states(); ++v)
        {
            const u32 c = sym->vertex_mapping(p)[v];
            ASSERT_LT(c, sym->num_vertices());
            EXPECT_EQ(sym->goal_flags()[c], sym->spaces()[p]->goal_flags()[v]);
        }
    // one domain only
    const TaskPtr other = fork_task("blocks_3", "test_problem.pddl");
    ASSERT_TRUE(other);
    EXPECT_THROW((void)GeneralizedStateSpace::create({spaces[0], space_of(other, fork_ss())}), std::invalid_argument);
}
}  // namespace
