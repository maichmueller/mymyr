// Numeric fluents and action costs: bytecode values, [bits | slots] states, stores, BrFS counts against the fork,
// plan costs.

#include "mymyr/formalism/text_format.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/search/aiw.hpp"
#include "mymyr/search/best_first.hpp"
#include "mymyr/search/brfs.hpp"
#include "mymyr/search/iw.hpp"
#include "mymyr/search/liw.hpp"
#include "mymyr/search/parallel_rollouts.hpp"
#include "mymyr/search/portfolio.hpp"
#include "mymyr/search/rollout_iw.hpp"
#include "mymyr/search/siw.hpp"
#include "mymyr/state/chunked_store.hpp"
#include "mymyr/state/flat_store.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/numeric.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace mymyr;

namespace
{
std::string numeric_dir() { return std::string(MYMYR_TEST_DATA_DIR) + "/numeric_tasks"; }
std::string numeric_task(const std::string& name) { return numeric_dir() + "/" + name + ".txt"; }
std::string lifted_task(const std::string& name) { return std::string(MYMYR_TEST_DATA_DIR) + "/tasks/" + name + ".txt"; }

struct Expected
{
    std::string name;
    u64 states, generated, goal_states;
};

// Exhaustive BrFS with witness pruning, against the fork's state counts per layer.
const std::vector<Expected>& small_tasks()
{
    static const std::vector<Expected> t = {
        {"cs-counters", 6561, 46656, 126},  {"cs-drone", 335, 1071, 10},       {"cs-farmland", 90900, 360600, 7043},
        {"cs-tpp", 1140, 6528, 46},         {"m-refuel-adl", 343, 1911, 1},    {"m-tpp-numeric", 1140, 6528, 46},
        {"m-woodworking", 20, 70, 4},       {"m-barman", 406, 1298, 28},       {"m-transport", 225, 960, 9},
    };
    return t;
}

plan::Numeric numeric_of(NumericStorage st, f64 quantum = 0, bool tolerant = false)
{
    plan::Numeric N;
    N.storage = st;
    N.quantum = quantum;
    N.tolerant = tolerant;
    return N;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ values

TEST(Numeric, CanonicalBitPatterns)
{
    const plan::Numeric N = numeric_of(NumericStorage::F64);
    EXPECT_EQ(std::bit_cast<u64>(plan::canonical(N, -0.0)), std::bit_cast<u64>(0.0));
    const f64 nan1 = std::numeric_limits<f64>::quiet_NaN();
    const f64 nan2 = -std::numeric_limits<f64>::quiet_NaN();
    const f64 nan3 = std::bit_cast<f64>(u64{0x7ff0000000000123ULL});  // a signalling NaN payload
    EXPECT_EQ(std::bit_cast<u64>(plan::canonical(N, nan1)), std::bit_cast<u64>(plan::canonical(N, nan2)));
    EXPECT_EQ(std::bit_cast<u64>(plan::canonical(N, nan1)), std::bit_cast<u64>(plan::canonical(N, nan3)));
    EXPECT_EQ(plan::canonical(N, 1.25), 1.25);
    EXPECT_EQ(plan::canonical(N, std::numeric_limits<f64>::infinity()), std::numeric_limits<f64>::infinity());

    const plan::Numeric Q = numeric_of(NumericStorage::F64, 0.5);
    EXPECT_EQ(plan::canonical(Q, 1.26), 1.5);
    EXPECT_EQ(plan::canonical(Q, -0.2), 0.0);
    EXPECT_EQ(std::bit_cast<u64>(plan::canonical(Q, -0.2)), std::bit_cast<u64>(0.0));

    const plan::Numeric C = numeric_of(NumericStorage::F64, 0, true);
    EXPECT_EQ(plan::canonical(C, 0.1 + 0.2), 0.3);
    EXPECT_NE(0.1 + 0.2, 0.3);
    EXPECT_TRUE(std::isnan(plan::canonical(C, std::numeric_limits<f64>::infinity())));  // non-finite: undefined
    EXPECT_EQ(plan::canonical(C, 1e7 + 0.1), 1e7 + 0.1);                               // above 1e6: unchanged
}

TEST(Numeric, Int32SlotsPackTwoPerWordAndRejectNonIntegers)
{
    const plan::Numeric N = numeric_of(NumericStorage::I32);
    std::array<u64, 3> w{0, 0, 0};  // three I32 slots take two words; the third stays untouched
    plan::store(N, w.data(), 0, 7);
    plan::store(N, w.data(), 1, -3);
    plan::store(N, w.data(), 2, 2147483647.0);
    EXPECT_EQ(plan::load(N, w.data(), 0), 7);
    EXPECT_EQ(plan::load(N, w.data(), 1), -3);
    EXPECT_EQ(plan::load(N, w.data(), 2), 2147483647.0);
    EXPECT_EQ(w[0] & 0xFFFFFFFFULL, 7u);  // slot 2k in the low half of word k
    EXPECT_EQ(w[1] >> 32, 0u);
    EXPECT_EQ(w[2], 0u);
    plan::store(N, w.data(), 1, -0.0);  // canonical zero
    EXPECT_EQ(w[0] >> 32, 0u);
    EXPECT_THROW(plan::store(N, w.data(), 0, 2147483648.0), std::overflow_error);
    EXPECT_THROW(plan::store(N, w.data(), 0, 0.5), std::overflow_error);
    EXPECT_THROW(plan::store(N, w.data(), 0, std::numeric_limits<f64>::quiet_NaN()), std::overflow_error);
}

TEST(Numeric, ComparisonsWithNaNAreFalse)
{
    using formalism::Comparator;
    const plan::Numeric N = numeric_of(NumericStorage::F64);
    const f64 nan = std::numeric_limits<f64>::quiet_NaN();
    for (Comparator c : {Comparator::Lt, Comparator::Le, Comparator::Eq, Comparator::Ge, Comparator::Gt})
    {
        EXPECT_FALSE(plan::compare(N, c, nan, 1));
        EXPECT_FALSE(plan::compare(N, c, 1, nan));
    }
    EXPECT_TRUE(plan::compare(N, Comparator::Le, 1, 1));
    EXPECT_FALSE(plan::compare(N, Comparator::Eq, 0.1 + 0.2, 0.3));
    const plan::Numeric C = numeric_of(NumericStorage::F64, 0, true);
    EXPECT_TRUE(plan::compare(C, Comparator::Eq, 0.1 + 0.2, 0.3));  // Mimir-C#: 1e-9 tolerance
}

TEST(Numeric, EffectFamilies)
{
    using formalism::AssignOp;
    EXPECT_EQ(plan::effect_family(AssignOp::Assign), 1);
    EXPECT_EQ(plan::effect_family(AssignOp::Increase), plan::effect_family(AssignOp::Decrease));
    EXPECT_EQ(plan::effect_family(AssignOp::ScaleUp), plan::effect_family(AssignOp::ScaleDown));
    EXPECT_NE(plan::effect_family(AssignOp::Increase), plan::effect_family(AssignOp::ScaleUp));
    const u8 add = plan::effect_family(AssignOp::Increase), mul = plan::effect_family(AssignOp::ScaleUp), as = 1;
    EXPECT_TRUE(plan::compatible_family(0, as));
    EXPECT_FALSE(plan::compatible_family(as, as));  // two assignments to one target
    EXPECT_TRUE(plan::compatible_family(add, add));
    EXPECT_TRUE(plan::compatible_family(mul, mul));
    EXPECT_FALSE(plan::compatible_family(add, mul));
    EXPECT_FALSE(plan::compatible_family(add, as));
}

// ------------------------------------------------------------------------------------------------ tasks and states

TEST(Numeric, NumericTasksCompileWithTheirInitialValues)
{
    if (!std::filesystem::exists(numeric_dir()))
        GTEST_SKIP() << "no numeric tasks";
    u32 checked = 0;
    for (const auto& e : std::filesystem::directory_iterator(numeric_dir()))
    {
        if (e.path().extension() != ".txt")
            continue;
        const formalism::TaskData d = formalism::read_task_text_file(e.path().string());
        const auto task = Task::create(d);
        // every fluent function value of the problem is a slot, with its initial value
        std::map<std::string, f64> want;
        for (const formalism::GroundFunctionValue& v : d.fluent_values)
        {
            std::string name = "(" + std::string(d.str(d.functions[v.func.v].name));
            for (ObjectId o : d.slice(d.object_ids, v.objects))
                name += " " + std::string(d.str(d.objects[o.v].name));
            want[name + ")"] = v.value;
        }
        std::map<std::string, f64> got;
        const std::vector<f64> vals = task->numeric_values(task->initial_state().view());
        ASSERT_EQ(vals.size(), task->numeric_slots());
        for (u32 i = 0; i < task->numeric_slots(); ++i)
            got[task->numeric_name(i)] = vals[i];
        EXPECT_EQ(got, want) << e.path();
        EXPECT_EQ(task->numeric_words(), plan::Numeric::words_for(task->numeric_slots(), task->numeric_storage())) << e.path();
        ++checked;
    }
    EXPECT_EQ(checked, 20u);
}

TEST(Numeric, ClassicalTasksHaveNoNumericWords)
{
    const auto task = Task::from_text_file(lifted_task("gripper__prob05"));
    EXPECT_EQ(task->numeric_slots(), 0u);
    EXPECT_EQ(task->numeric_words(), 0u);
    EXPECT_EQ(task->initial_state().numeric_words(), 0u);
    // the hash of a classical state is the hash of its words
    const State& s = task->initial_state();
    EXPECT_EQ(s.hash(), hash::state_words(s.data(), s.size_words()));
}

TEST(Numeric, StatesThatDifferOnlyInTheirValuesAreDistinct)
{
    const auto task = Task::from_text_file(numeric_task("cs-counters"));
    ASSERT_GT(task->numeric_slots(), 0u);
    Successors& succ = task->workspace().successors();
    const State s0 = task->initial_state();
    std::vector<State> next;
    for (const Action& a : succ.applicable_actions(s0.view()))  // not from inside a callback: not reentrant
        next.push_back(succ.apply(s0.view(), a.label()));
    ASSERT_FALSE(next.empty());
    u32 same_bits = 0;
    FlatStateStore flat(std::max<u32>(1, task->words()), 4, task->numeric_words());
    ChunkedStateStore chunked(std::max<u32>(1, task->words()), task->numeric_words());
    flat.insert(s0.view());
    chunked.insert(s0.view());
    for (const State& t : next)
    {
        const bool bits_equal = t.size_words() == s0.size_words() && std::equal(t.data(), t.data() + t.size_words(), s0.data());
        if (!bits_equal)
            continue;
        ++same_bits;
        EXPECT_NE(t, s0);
        EXPECT_NE(t.hash(), s0.hash());
        EXPECT_NE(task->canonical_hash(t.view()), task->canonical_hash(s0.view()));
        EXPECT_TRUE(flat.insert(t.view()).second);
        EXPECT_TRUE(chunked.insert(t.view()).second);
        EXPECT_FALSE(flat.insert(t.view()).second);
        EXPECT_FALSE(chunked.insert(t.view()).second);
        const State copy(t.view());
        EXPECT_EQ(copy, t);
        EXPECT_EQ(copy.hash(), t.hash());
    }
    EXPECT_GT(same_bits, 0u);  // counters: increments change values only
    std::vector<u64> w(chunked.words()), num(task->numeric_words());
    chunked.decode(StateId{1}, w.data(), num.data());
    EXPECT_EQ(State(w.data(), chunked.words(), num.data(), task->numeric_words()), State(flat[StateId{1}]));
}

// ------------------------------------------------------------------------------------------------ BrFS

TEST(Numeric, BrfsCountsEqualTheForkOnEveryStore)
{
    for (const Expected& x : small_tasks())
    {
        const auto task = Task::from_text_file(numeric_task(x.name));
        for (BrfsOptions::Store store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked, BrfsOptions::Store::Compact,
                                         BrfsOptions::Store::Concurrent})
            for (u32 threads : {1u, 3u})
            {
                if (threads > 1 && store != BrfsOptions::Store::Concurrent)
                    continue;
                BrfsOptions o;
                o.store = store;
                o.threads = threads;
                const BrfsResult r = brfs(*task, o);
                EXPECT_TRUE(r.exhausted) << x.name << " " << r.store;
                EXPECT_EQ(r.states, x.states) << x.name << " " << r.store << " T=" << threads;
                EXPECT_EQ(r.generated, x.generated) << x.name << " " << r.store << " T=" << threads;
                EXPECT_EQ(r.goal_states, x.goal_states) << x.name << " " << r.store << " T=" << threads;
            }
    }
}

TEST(Numeric, BrfsIsIndependentOfTheSlotType)
{
    for (const char* name : {"cs-counters", "cs-tpp", "m-refuel-adl"})
    {
        TaskOptions f64;
        f64.numeric_storage = TaskOptions::NumericStorageMode::F64;
        const auto a = Task::from_text_file(numeric_task(name));
        const auto b = Task::from_text_file(numeric_task(name), f64);
        EXPECT_EQ(a->numeric_storage(), NumericStorage::I32) << name;
        EXPECT_EQ(b->numeric_storage(), NumericStorage::F64) << name;
        BrfsOptions o;
        o.fingerprint = true;
        const BrfsResult ra = brfs(*a, o), rb = brfs(*b, o);
        EXPECT_EQ(ra.states, rb.states) << name;
        EXPECT_EQ(ra.generated, rb.generated) << name;
        EXPECT_EQ(ra.goal_states, rb.goal_states) << name;
        // the same states in the same order: the canonical hash covers the values, not their encoding
        EXPECT_EQ(ra.fingerprint, rb.fingerprint) << name;
    }
}

TEST(Numeric, BrfsIdsAreDeterministicAcrossThreadCounts)
{
    const auto task = Task::from_text_file(numeric_task("cs-farmland"));
    BrfsOptions o;
    o.fingerprint = true;
    o.store = BrfsOptions::Store::Flat;
    const BrfsResult one = brfs(*task, o);
    o.store = BrfsOptions::Store::Concurrent;
    for (u32 t : {1u, 2u, 4u})
    {
        o.threads = t;
        const BrfsResult r = brfs(*task, o);
        EXPECT_EQ(r.states, one.states) << t;
        EXPECT_EQ(r.fingerprint, one.fingerprint) << t;
    }
}

TEST(Numeric, DepthCappedLayersEqualTheProbe)
{
    struct Layers
    {
        const char* name;
        std::vector<std::array<u64, 3>> layers;
    };
    const std::vector<Layers> want = {
        {"cs-block-grouping", {{1, 19, 19}, {19, 361, 180}, {180, 3422, 1141}, {1141, 21712, 5476}}},
        {"cs-sailing", {{1, 14, 14}, {14, 196, 97}, {97, 1358, 448}, {448, 6272, 1562}}},  // f64 slots
        {"cs-hydropower", {{1, 2, 2}, {2, 5, 4}, {4, 10, 6}, {6, 15, 9}, {9, 23, 15}, {15, 38, 24}}},  // f64 slots
        {"cs-expedition", {{1, 6, 6}, {6, 26, 13}, {13, 48, 22}, {22, 96, 39}, {39, 148, 54}, {54, 246, 86}}},
        {"cs-ext-plant-watering", {{1, 16, 16}, {16, 256, 96}, {96, 1506, 290}, {290, 4372, 564}}},
        {"m-zenotravel-numeric", {{1, 7, 5}, {5, 27, 12}, {12, 59, 25}, {25, 129, 48}, {48, 262, 91}, {91, 491, 164}}},
        {"cs-delivery", {{1, 20, 20}, {20, 284, 150}, {150, 1664, 648}, {648, 6536, 2091}}},
    };
    for (const Layers& x : want)
    {
        const auto task = Task::from_text_file(numeric_task(x.name));
        for (BrfsOptions::Store store : {BrfsOptions::Store::Flat, BrfsOptions::Store::Chunked, BrfsOptions::Store::Compact,
                                         BrfsOptions::Store::Concurrent})
        {
            BrfsOptions o;
            o.store = store;
            o.threads = store == BrfsOptions::Store::Concurrent ? 2 : 1;
            o.max_depth = static_cast<u32>(x.layers.size());
            o.layer_stats = true;
            const BrfsResult r = brfs(*task, o);
            EXPECT_EQ(r.layer_counts, x.layers) << x.name << " " << r.store;
            EXPECT_EQ(r.layers, x.layers.size()) << x.name << " " << r.store;
            EXPECT_FALSE(r.exhausted) << x.name << " " << r.store;
            u64 states = 1;
            for (const auto& l : x.layers)
                states += l[2];
            EXPECT_EQ(r.states, states) << x.name << " " << r.store;
        }
    }
}

// ------------------------------------------------------------------------------------------------ costs

TEST(ActionCosts, KindsFollowTheFork)
{
    {
        const auto task = Task::from_text_file(lifted_task("gripper__prob05"));
        const heuristics::ActionCosts c(*task);
        EXPECT_TRUE(c.unit());
        EXPECT_TRUE(c.state_independent());
    }
    {
        const auto task = Task::from_text_file(numeric_task("m-transport"));  // (increase (total-cost) (road-length ?l1 ?l2))
        const heuristics::ActionCosts c(*task);
        EXPECT_EQ(c.kind(), heuristics::ActionCosts::Kind::TotalCost);
        EXPECT_TRUE(c.state_independent());
        EXPECT_TRUE(c.integral());
    }
    {
        const auto task = Task::from_text_file(numeric_task("cs-tpp"));  // total-cost over the fluent (bought ...)
        const heuristics::ActionCosts c(*task);
        EXPECT_EQ(c.kind(), heuristics::ActionCosts::Kind::TotalCost);
        EXPECT_FALSE(c.state_independent());
    }
}

TEST(ActionCosts, NumericPlansEqualTheFork)
{
    // IW and SIW on numeric tasks (novelty over atoms only): status, plan length and cost against the fork's;
    // the plan reaches a goal state, numeric goal constraints included, and its cost
    // is at most the replayed metric value (the fork's extraction may pick cheaper actions between the same states).
    struct Case
    {
        const char* name;
        bool siw;
        u32 k;
        bool solved;
        usize plan_len;
        f64 cost;
    };
    const std::vector<Case> cases = {
        {"m-woodworking", false, 1, true, 2, 40},  {"m-transport", false, 2, true, 5, 54},
        {"m-transport", true, 2, true, 7, 154},    {"cs-delivery", true, 2, true, 14, 79},
        {"cs-delivery-nometric", true, 2, true, 14, 14},
        {"cs-counters", false, 2, false, 0, 0},  // the fork's IW(2) crashes here; every state has the same atoms
        {"m-zenotravel-numeric", true, 2, false, 0, 0},
    };
    for (const Case& c : cases)
    {
        const auto task = Task::from_text_file(numeric_task(c.name));
        search::IwOptions o;
        o.max_arity = c.k;
        std::vector<Action> plan;
        f64 cost = 0;
        std::optional<State> goal;
        if (c.siw)
        {
            const search::SiwResult r = search::siw(*task, o);
            ASSERT_EQ(r.status == search::SearchStatus::Solved, c.solved) << c.name;
            EXPECT_TRUE(r.cost_exact);
            plan = r.plan, cost = r.cost, goal = r.goal_state;
        }
        else
        {
            const search::IwResult r = search::iw(*task, o);
            ASSERT_EQ(r.status == search::SearchStatus::Solved, c.solved) << c.name;
            EXPECT_TRUE(r.cost_exact);
            plan = r.plan, cost = r.cost, goal = r.goal_state;
        }
        if (!c.solved)
            continue;
        EXPECT_EQ(plan.size(), c.plan_len) << c.name;
        EXPECT_EQ(cost, c.cost) << c.name;
        Successors& succ = task->workspace().successors();
        State s = task->initial_state();
        const heuristics::ActionCosts costs(*task);
        f64 g = costs.initial(s.view());
        for (const Action& a : plan)
        {
            StateBuilder b;
            const Delta d = succ.apply_with_delta(s.view(), a.label(), b);
            g = costs.next(g, d);
            s = b.build();
        }
        EXPECT_TRUE(task->is_goal(s.view())) << c.name;
        EXPECT_EQ(s, *goal) << c.name;
        if (!c.siw)
        {
            EXPECT_LE(cost, g) << c.name;
        }
    }
}

// ------------------------------------------------------------------------------------------------ rl::expand

TEST(NumericRl, ExpandRowsCarryTheNumericWords)
{
    // rows [bits | slots]: the batched expansion equals Successors on every state of a BrFS prefix, in canonical order
    for (const char* name : {"cs-tpp", "cs-hydropower", "m-transport"})
    {
        const auto task = Task::from_text_file(numeric_task(name));
        const u32 NN = task->numeric_words(), W = std::max<u32>(1, task->words()), RW = W + NN;
        Successors& succ = task->workspace().successors();
        // a batch: the initial state and its successors, twice
        const State s0 = task->initial_state();
        std::vector<State> batch{s0};
        for (const Action& a : succ.applicable_actions(s0))  // not from inside the callback: Successors is not reentrant
            batch.push_back(succ.apply(s0, a.label()));
        const usize n0 = batch.size();
        for (usize i = 0; i < n0; ++i)
            batch.push_back(batch[i]);
        std::vector<u64> rows(batch.size() * RW, 0);
        for (usize i = 0; i < batch.size(); ++i)
        {
            std::copy(batch[i].data(), batch[i].data() + batch[i].size_words(), rows.begin() + static_cast<std::ptrdiff_t>(i * RW));
            std::copy(batch[i].numeric().begin(), batch[i].numeric().end(), rows.begin() + static_cast<std::ptrdiff_t>(i * RW + W));
        }
        const rl::StateBatchView in{rows.data(), batch.size(), W, 0, NN};
        const u64 cap = 4096;
        std::vector<u64> out(cap * RW), out2(cap * RW);
        std::vector<i32> parent(cap), schema(cap), offsets(batch.size() + 1);
        std::vector<u8> goal(cap), goal2(cap);
        rl::Expansion x;
        x.capacity = cap;
        x.words = W;
        x.numeric_words = NN;
        x.succ = out.data();
        x.parent = parent.data();
        x.schema = schema.data();
        x.goal = goal.data();
        x.offsets = offsets.data();
        rl::expand(*rl::TaskTable::single(task), in, nullptr, x);
        ASSERT_FALSE(x.overflow()) << name;
        u64 j = 0;
        for (usize i = 0; i < batch.size(); ++i)
        {
            std::vector<State> want;
            for (const Action& a : succ.applicable_actions(batch[i].view()))
                want.push_back(succ.apply(batch[i].view(), a.label()));
            ASSERT_EQ(static_cast<u64>(offsets[i + 1] - offsets[i]), want.size()) << name;
            for (const State& t : want)
            {
                const u64* r = out.data() + j * RW;
                EXPECT_EQ(State(r, W, NN ? r + W : nullptr, NN), t) << name << " row " << j;
                EXPECT_EQ(goal[j] != 0, task->is_goal(t.view())) << name;
                ++j;
            }
        }
        EXPECT_EQ(j, x.total);
        // the pool variant: bit for bit
        ThreadPool pool(3);
        rl::ExpandScratch scratch;
        rl::Expansion y = x;
        y.succ = out2.data();
        y.goal = goal2.data();
        rl::expand(*rl::TaskTable::single(task), in, nullptr, y, {}, pool, scratch);
        EXPECT_EQ(y.total, x.total);
        EXPECT_TRUE(std::equal(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(x.total * RW), out2.begin())) << name;
        EXPECT_TRUE(std::equal(goal.begin(), goal.begin() + static_cast<std::ptrdiff_t>(x.total), goal2.begin())) << name;
        // is_goal over the batch rows
        std::vector<u8> g(batch.size());
        rl::is_goal(*rl::TaskTable::single(task), in, nullptr, g.data());
        for (usize i = 0; i < batch.size(); ++i)
            EXPECT_EQ(g[i] != 0, task->is_goal(batch[i].view())) << name;
        // padded rows keep the numeric words after the (wider) atom words
        const u32 K = 64, PW = W + 1;
        std::vector<u64> psucc(batch.size() * K * (PW + NN));
        std::vector<u8> mask(batch.size() * K);
        rl::PaddedExpansion pd;
        pd.K = K;
        pd.words = PW;
        pd.numeric_words = NN;
        pd.succ = psucc.data();
        pd.mask = mask.data();
        rl::pad(x, batch.size(), pd);
        for (usize i = 0; i < batch.size(); ++i)
            for (u32 k = 0; k < K; ++k)
            {
                if (!mask[i * K + k])
                    continue;
                const u64* r = psucc.data() + (i * K + k) * (PW + NN);
                const u64* f = out.data() + (static_cast<u64>(offsets[i]) + k) * RW;
                EXPECT_EQ(State(r, PW, NN ? r + PW : nullptr, NN), State(f, W, NN ? f + W : nullptr, NN)) << name;
            }
        // a batch without the numeric words is refused (m-transport has no numeric slots: total-cost only)
        rl::StateBatchView bad = in;
        bad.numeric_words = 0;
        bad.words = RW;
        if (NN > 0)
        {
            EXPECT_THROW(rl::expand(*rl::TaskTable::single(task), bad, nullptr, x), std::invalid_argument);
        }
    }
}

TEST(NumericRl, RandomWalksKeepTheValues)
{
    const auto task = Task::from_text_file(numeric_task("cs-counters"));
    const rl::WalkStats st = rl::random_walks(*task, 2000, 30, 7);
    EXPECT_EQ(st.steps, 2000u);
    EXPECT_GT(st.successors, 2000u);  // counters: values change, so walks do not stall at the initial state
}

TEST(Numeric, IwFamilyVariantsRefuseNumericTasks)
{
    // the IW family variants rebuild states from atom deltas and read atom goals only: they must refuse, not drop the values
    const auto task = Task::from_text_file(numeric_task("cs-counters"));
    EXPECT_THROW((void)search::liw(*task, search::LiwOptions{}), std::invalid_argument);
    EXPECT_THROW((void)search::abstracted_iw(*task, search::AbstractedIwOptions{}), std::invalid_argument);
    EXPECT_THROW((void)search::rollout_iw(*task, search::RolloutIwOptions{}), std::invalid_argument);
    EXPECT_THROW((void)search::find_rollouts_parallel(*task, search::ParallelRolloutOptions{}), std::invalid_argument);
    EXPECT_THROW((void)search::atomic_goal_portfolio(*task, search::PortfolioOptions{}), std::invalid_argument);
}
