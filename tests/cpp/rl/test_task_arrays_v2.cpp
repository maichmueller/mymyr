// rl::device_arrays version 2: a strict superset of version 1, deterministic, structurally valid, and read on the
// host through rl/task_arrays_view.hpp it answers is_applicable / apply / is_goal exactly as the CPU engine does, on
// every task of the lifted suite, with frozen and lazy slots. The CUDA smoke test (tests/cuda) runs the same functions
// on the device against the same reference.

#include "../support/device_ref.hpp"
#include "../support/suite.hpp"
#include "mymyr/rl/task_arrays.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
std::vector<std::tuple<std::string, bool>> params()
{
    std::vector<std::tuple<std::string, bool>> out;
    for (const SuiteTask& t : suite())
        for (bool frozen : {true, false})
            out.emplace_back(t.name, frozen);
    return out;
}

class TaskArraysV2 : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
protected:
    TaskPtr load() const
    {
        TaskOptions o;
        o.atoms = std::get<1>(GetParam()) ? TaskOptions::Atoms::Frozen : TaskOptions::Atoms::Lazy;
        return Task::from_text_file(task_path(std::get<0>(GetParam())), o);
    }
};

bool same_bytes(const rl::ArrayBundle& a, const rl::ArrayInfo& x, const rl::ArrayBundle& b, const rl::ArrayInfo& y)
{
    if (x.dtype != y.dtype || x.shape != y.shape || x.words != y.words)
        return false;
    const u64 n = x.elements() * rl::dtype_bytes(x.dtype);
    return n == 0 || std::memcmp(a.data(x), b.data(y), n) == 0;
}
}  // namespace

TEST_P(TaskArraysV2, IsASupersetOfVersionOneAndDeterministic)
{
    const auto task = load();
    (void)device_ref_walks(*task, 2, 10, 5);  // assign some lazy slots first
    const rl::ArrayBundle v1 = rl::device_arrays(*task, 1);
    const rl::ArrayBundle v2 = rl::device_arrays(*task, 2);
    const rl::ArrayBundle again = rl::device_arrays(*task);
    EXPECT_EQ(v1.scalar("version"), 1);
    EXPECT_EQ(v2.scalar("version"), 2);
    EXPECT_EQ(again.scalar("version"), static_cast<i64>(rl::k_device_arrays_version));
    EXPECT_EQ(v1.find("plan_matcher"), nullptr);
    EXPECT_EQ(v1.scalar("section_plan", -7), -7);
    EXPECT_EQ(v2.scalar("section_core"), 1);
    EXPECT_EQ(v2.scalar("section_plan"), 2);
    for (const rl::ArrayInfo& a : v1.arrays())
    {
        const rl::ArrayInfo* b = v2.find(a.name);
        ASSERT_NE(b, nullptr) << a.name;
        EXPECT_TRUE(same_bytes(v1, a, v2, *b)) << a.name;
    }
    for (const auto& [name, value] : v1.scalars())
    {
        if (name != "version")
        {
            EXPECT_EQ(v2.scalar(name, value + 1), value) << name;
        }
    }
    // the same task exported twice: the same block, byte for byte
    ASSERT_EQ(v2.bytes(), again.bytes());
    EXPECT_EQ(std::memcmp(v2.block(), again.block(), v2.bytes()), 0);
    ASSERT_EQ(v2.arrays().size(), again.arrays().size());
    for (usize i = 0; i < v2.arrays().size(); ++i)
    {
        EXPECT_EQ(v2.arrays()[i].name, again.arrays()[i].name);
        EXPECT_EQ(v2.arrays()[i].offset, again.arrays()[i].offset);
    }
    EXPECT_THROW((void)rl::task_view(v1), std::invalid_argument);
    EXPECT_THROW((void)rl::device_arrays(*task, 3), std::invalid_argument);
}

TEST_P(TaskArraysV2, ValidatesAndMatchesThePlan)
{
    const auto task = load();
    const rl::ArrayBundle b = rl::device_arrays(*task, 2);
    const rl::dev::TaskView v = rl::task_view(b);
    const plan::Compiled& C = task->compiled();
    ASSERT_GT(rl::dev::validate_items(v), 0u);
    for (u64 i = 0; i < rl::dev::validate_items(v); ++i)
        ASSERT_EQ(rl::dev::validate_item(v, i), 0u) << "item " << i;
    EXPECT_EQ(v.num_schemas, C.schemas.size());
    EXPECT_EQ(v.n_pattern_vars, C.pattern_vars.size());
    EXPECT_EQ(v.max_bind, C.max_bind);
    EXPECT_EQ(v.atom_total, C.layout.total);
    usize matchers = 0, ces = 0, axioms = 0;
    for (const plan::Schema& s : C.schemas)
    {
        matchers += 2 + s.ces.size();
        ces += s.ces.size();
    }
    for (const plan::Stratum& st : C.strata)
    {
        matchers += st.axioms.size();
        axioms += st.axioms.size();
    }
    EXPECT_EQ(v.n_matchers, matchers);
    EXPECT_EQ(v.n_cond_effects, ces);
    EXPECT_EQ(v.n_axioms, axioms);
    EXPECT_EQ(v.goal_check_n, C.goal.lits.size());
    // every pattern key equals the engine's for a random binding (pattern vars resolved through plan_rs)
    std::mt19937_64 rng(11);
    std::vector<u32> bind(std::max<u32>(1, v.max_bind));
    for (u32 s = 0; s < v.num_schemas; ++s)
    {
        const u32* sc = v.schema + static_cast<u64>(s) * rl::dev::k_sc_count;
        for (u32 j = 0; j < bind.size(); ++j)
            bind[j] = static_cast<u32>(rng() % std::max<u32>(1, v.num_objects));
        for (u32 i = 0; i < C.schemas[s].pre_lits.size(); ++i)
        {
            const plan::Pattern& pt = C.schemas[s].pre_lits[i].pat;
            u64 expect = pt.base;
            for (u32 k = 0; k < pt.var_count; ++k)
            {
                const plan::PatternVar& pv = C.pattern_vars[pt.var_begin + k];
                expect += pv.rs[bind[pv.param]];
            }
            const u32 p = v.check[(static_cast<u64>(sc[rl::dev::k_sc_pre_lits]) + i) * 2];
            ASSERT_EQ(rl::dev::pattern_key(v, p, bind.data()), expect) << "schema " << s << " literal " << i;
        }
    }
}

TEST_P(TaskArraysV2, HostViewAgreesWithTheEngine)
{
    const auto task = load();
    const std::vector<State> states = device_ref_walks(*task, 3, 25, 17);
    const DeviceRef ref = device_ref(*task, states, 1, 23);
    const rl::ArrayBundle b = rl::device_arrays(*task, 2);  // after the reference: its successors' slots exist
    const rl::dev::TaskView v = rl::task_view(b);
    const u32 W = ref.W;
    u64 applicable = 0, applied = 0, conditional = 0;
    std::vector<u64> out(W);
    for (u64 k = 0; k < ref.k(); ++k)
    {
        const u32 si = ref.label_state[k];
        const u64* s = ref.states.data() + static_cast<u64>(si) * W;
        const u64* d = ref.DW ? ref.derived.data() + static_cast<u64>(si) * ref.DW : nullptr;
        const u32* bind = ref.label_binding.data() + k * ref.L;
        const bool app = rl::dev::is_applicable(v, ref.label_schema[k], bind, s, W, d, ref.DW);
        ASSERT_EQ(app, ref.applicable[k] != 0) << "label " << k << " schema " << ref.label_schema[k];
        if (!app)
            continue;
        ++applicable;
        const u32 st = rl::dev::apply_action(v, ref.label_schema[k], bind, s, W, out.data(), W);
        if (st == rl::dev::k_apply_conditional)
        {
            ++conditional;
            continue;
        }
        ASSERT_EQ(st, rl::dev::k_apply_ok) << "label " << k;
        ASSERT_TRUE(std::equal(out.begin(), out.end(), ref.successor.begin() + static_cast<std::ptrdiff_t>(k * W))) << "label " << k;
        ++applied;
    }
    for (u64 i = 0; i < ref.n(); ++i)
    {
        const u64* d = ref.DW ? ref.derived.data() + i * ref.DW : nullptr;
        ASSERT_EQ(rl::dev::goal_holds(v, ref.states.data() + i * W, W, d, ref.DW), ref.goal[i] != 0) << "state " << i;
    }
    EXPECT_GT(applicable, 0u);
    EXPECT_LT(applicable, ref.k());  // perturbed labels include inapplicable ones
    EXPECT_EQ(applied + conditional, applicable);
    if (!task->compiled().has_conditional_effects)
    {
        EXPECT_EQ(conditional, 0u);
    }
}

TEST_P(TaskArraysV2, DeviceMatcherFlagsFollowTheSearchCosts)
{
    // the device's matcher per plan matcher comes from device_search_costs (deterministic, and under lazy slots
    // it interns no atom); the export's k_mc_device_fc follows it, k_mc_witnesses the plan's witness-only parameters.
    // a matcher deeper than k_max_matcher_depth (the deep kernels) keeps the plan's choice (k_mc_use_fc)
    const auto task = load();
    const u32 fluent = task->atoms().fluent_slots(), derived = task->atoms().derived_slots();
    const rl::DeviceSearchCosts costs = rl::device_search_costs(*task);
    const rl::DeviceSearchCosts again = rl::device_search_costs(*task);
    EXPECT_EQ(task->atoms().fluent_slots(), fluent);
    EXPECT_EQ(task->atoms().derived_slots(), derived);
    const plan::Compiled& C = task->compiled();
    ASSERT_EQ(costs.schemas.size(), C.schemas.size());
    ASSERT_EQ(again.schemas.size(), C.schemas.size());
    EXPECT_GE(costs.states, 1u);
    EXPECT_LE(costs.states, task->atoms().mode() == AtomMode::Frozen ? rl::k_probe_states : 1u);
    const rl::ArrayBundle b = rl::device_arrays(*task, 2);
    const rl::dev::TaskView v = rl::task_view(b);
    auto expect_flags = [&](u32 flags, const plan::Matcher& m, rl::SearchCost c, bool schema, const std::string& what)
    {
        const bool deep = m.total > rl::dev::k_max_matcher_depth || m.steps.size() > rl::dev::k_max_matcher_depth;
        if (deep && schema)
            EXPECT_EQ((flags & rl::dev::k_mc_device_fc) != 0, m.use_fc && C.ow <= rl::dev::k_max_fc_ow) << what;
        else if ((flags & rl::dev::k_mc_device_fc) != 0)
        {
            EXPECT_TRUE(rl::prefers_fc(c) && C.ow <= rl::dev::k_max_fc_ow) << what;
        }
        EXPECT_EQ((flags & rl::dev::k_mc_witnesses) != 0, m.steps.size() >= m.first_exist + 2) << what;
        EXPECT_EQ((flags & rl::dev::k_mc_use_fc) != 0, m.use_fc) << what;
    };
    for (u32 s = 0; s < v.num_schemas; ++s)
        for (u32 wi = 0; wi < 2; ++wi)
        {
            EXPECT_EQ(costs.schemas[s][wi].fixed, again.schemas[s][wi].fixed);
            EXPECT_EQ(costs.schemas[s][wi].fc, again.schemas[s][wi].fc);
            expect_flags(rl::dev::schema_matcher_flags(v, s, wi == 0), C.schemas[s].pre[wi], costs.schemas[s][wi], true,
                         "schema " + std::to_string(s) + " pre " + std::to_string(wi));
            // the conditions of conditional effects run the fixed order
            const u32* sc = v.schema + static_cast<u64>(s) * rl::dev::k_sc_count;
            for (u32 k = 0; k < sc[rl::dev::k_sc_ces_n]; ++k)
            {
                const u32 m = v.cond_effect[(static_cast<u64>(sc[rl::dev::k_sc_ces]) + k) * 5];
                EXPECT_EQ(v.matcher[static_cast<u64>(m) * rl::dev::k_mc_count + rl::dev::k_mc_flags] & rl::dev::k_mc_device_fc, 0u);
            }
        }
    u32 a = 0;
    for (const plan::Stratum& st : C.strata)
        for (const plan::Axiom& x : st.axioms)
        {
            expect_flags(rl::dev::axiom_matcher_flags(v, a), x.body, costs.axioms[a], false, "axiom " + std::to_string(a));
            ++a;
        }
    EXPECT_EQ(a, v.n_axioms);
    if (std::get<0>(GetParam()) == "sokoban-opt08-strips__p14")
    {
        // the fixed order starts at a 35-object parameter (mimir's order heuristic): forward checking starts at the player
        u32 s = 0;
        while (s < v.num_schemas && task->schema_name(SchemaId{s}) != "move")
            ++s;
        ASSERT_LT(s, v.num_schemas);
        EXPECT_TRUE(rl::prefers_fc(costs.schemas[s][1]));
        EXPECT_TRUE(rl::dev::device_fc(rl::dev::schema_matcher_flags(v, s, false), rl::dev::MatchOrder::Free));
    }
    if (std::get<0>(GetParam()) == "gripper__prob05")
    {
        for (u32 s = 0; s < v.num_schemas; ++s)
            EXPECT_FALSE(rl::dev::device_fc(rl::dev::schema_matcher_flags(v, s, true), rl::dev::MatchOrder::Free)) << s;
    }
}

INSTANTIATE_TEST_SUITE_P(Suite, TaskArraysV2, ::testing::ValuesIn(params()),
                         [](const auto& info)
                         {
                             std::string n = std::get<0>(info.param) + (std::get<1>(info.param) ? "_frozen" : "_lazy");
                             std::replace_if(n.begin(), n.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)); }, '_');
                             return n;
                         });
