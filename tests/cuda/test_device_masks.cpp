// Device action mask tests (GPU 0: CUDA_VISIBLE_DEVICES=0; they skip without a device), on the table instance sets
// (tests/cpp/rl/table_instance_sets.hpp) as tables and as a suite of several domains, dense and reachable slots:
//   - cuda::DeviceMasks equals rl::schema_masks, rl::parameter_masks (every depth; prefixes of applicable labels,
//     mutated ones and invalid schemas) and rl::is_applicable (applicable, mutated and invalid labels) byte for byte;
//     depths past an arity, narrow bindings and task ids outside the suite set status bits;
//   - the label-mode device env (cuda::DeviceEnv with rl::ActionSpace::Label) equals rl::HostEnv step by step over
//     walks of applicable, inapplicable and invalid labels, with truncation, autoresets into next_task_ids and per-env
//     goals;
//   - which suites they run, and why not.

#include "env_harness.hpp"
#include "mymyr/cuda/mask_kernels.hpp"
#include "mymyr/cuda/masks.hpp"
#include "mymyr/rl/masks.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct Named
{
    std::string name;
    rl::TaskSuitePtr suite;
};

/// Each set's table as a suite, and a suite of four domains (the sets without conditional effects and axioms).
std::vector<Named> suites(TaskOptions::Atoms atoms)
{
    std::vector<Named> out;
    std::map<std::string, rl::TaskTablePtr> by;
    for (Set& s : sets(atoms))
    {
        by[s.name] = s.table;
        if (s.name != "miconic-simpleadl")
            out.push_back({s.name + atoms_name(atoms), rl::TaskSuite::of(s.table)});
    }
    if (by.size() == table_instance_sets().size())
        out.push_back({std::string("miconic+blocks+gripper+logistics") + atoms_name(atoms),
                       rl::TaskSuite::create({by.at("miconic"), by.at("blocks"), by.at("gripper"), by.at("logistics")})});
    return out;
}

/// Rows of the suite at its width (each domain's table_rows, interleaved by domain) and their global task ids.
void suite_rows(const rl::TaskSuite& suite, u32 per, std::vector<u64>& rows, std::vector<i32>& ids)
{
    const u32 W = suite.words();
    std::vector<std::vector<u64>> drows(suite.num_domains());
    std::vector<std::vector<i32>> dids(suite.num_domains());
    for (u32 d = 0; d < suite.num_domains(); ++d)
        table_rows(*suite.table(d), per, drows[d], dids[d]);
    rows.clear();
    ids.clear();
    for (u64 k = 0;; ++k)
    {
        bool any = false;
        for (u32 d = 0; d < suite.num_domains(); ++d)
        {
            if (k >= dids[d].size())
                continue;
            any = true;
            const u32 w = suite.table(d)->words();
            for (u32 j = 0; j < W; ++j)
                rows.push_back(j < w ? drows[d][k * w + j] : 0);
            ids.push_back(static_cast<i32>(suite.global_id(d, static_cast<u32>(dids[d][k]))));
        }
        if (!any)
            break;
    }
}

/// A label of each row (schema [N], binding [N, L], -1 padding): successor (5 t + 3 i) mod count of the row's state
/// (canonical order), mutated (an object replaced) in every fourth row, an invalid schema in every seventh, schema 0
/// with object 0 where the state has no successors.
void row_labels(const rl::TaskSuite& suite, const std::vector<u64>& rows, const std::vector<i32>& ids, u32 L, int t,
                std::vector<i32>& schema, std::vector<i32>& binding)
{
    const u32 W = suite.words();
    const u64 N = ids.size();
    schema.assign(N, 0);
    binding.assign(N * L, -1);
    for (u64 i = 0; i < N; ++i)
    {
        const Task& task = *suite.task(static_cast<u32>(ids[i]));
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
        const u32 w = task.words();
        succ.prepare(StateView{rows.data() + i * W, w, nullptr, 0});
        std::vector<std::pair<u32, std::vector<u32>>> labels;
        succ.generate<false>(
            [&](u32 s, const ObjectId* b, const Delta&) -> bool
            {
                std::vector<u32> v(succ.arity(s));
                for (u32 k = 0; k < v.size(); ++k)
                    v[k] = b[k].v;
                labels.emplace_back(s, std::move(v));
                return true;
            },
            false, true);
        if (labels.empty())
        {
            binding[i * L] = 0;
            continue;
        }
        const auto& [s, b] = labels[(5 * static_cast<u64>(t) + 3 * i) % labels.size()];
        schema[i] = static_cast<i32>(s);
        for (u32 k = 0; k < b.size(); ++k)
            binding[i * L + k] = static_cast<i32>(b[k]);
        const u32 objects = task.compiled().num_objects;
        if (i % 4 == 1 && !b.empty())
            binding[i * L + (i / 4) % b.size()] = static_cast<i32>(hash::mix64(i + static_cast<u64>(t)) % objects);
        if (i % 7 == 5)
            schema[i] = i % 14 == 5 ? -1 : static_cast<i32>(suite.max_schemas());
    }
}

TEST(DeviceMasks, EqualHostMasks)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    const cudaStream_t s = ctx->stream();
    for (const auto atoms : {TaskOptions::Atoms::Dense, TaskOptions::Atoms::Reachable})
        for (const Named& n : suites(atoms))
        {
            SCOPED_TRACE(n.name);
            const rl::TaskSuite& suite = *n.suite;
            ASSERT_EQ(cuda::DeviceMasks::unsupported(suite), "");
            const cuda::DeviceMasks masks(ctx, n.suite);
            std::vector<u64> rows;
            std::vector<i32> ids;
            suite_rows(suite, 12, rows, ids);
            const u64 N = ids.size();
            const u32 W = suite.words(), S = suite.max_schemas(), O = suite.max_objects() + 3;
            const u32 L = std::max<u32>(1, suite.label_width());
            const rl::StateBatchView view{rows.data(), N, W, W, 0};
            const cuda::DeviceBuffer drows = to_device(ctx, rows, s), dids = to_device(ctx, ids, s);
            cuda::DeviceBuffer out(ctx, N * std::max(S, O), s), views(ctx, N * masks.view_words() * 8, s), st(ctx, 4, s);
            const cuda::DeviceMasks::Batch b{static_cast<const u64*>(drows.data()), N, W, W,
                                             static_cast<const i32*>(dids.data())};
            auto status = [&] { return to_host<u32>(st.data(), 1, s)[0]; };

            std::vector<u8> host(N * S);
            rl::schema_masks(suite, view, ids.data(), host.data());
            masks.schemas(b, static_cast<u8*>(out.data()), static_cast<u64*>(views.data()), static_cast<u32*>(st.data()), s);
            EXPECT_EQ(to_host<u8>(out.data(), N * S, s), host);
            EXPECT_EQ(status(), 0u);
            u64 set = 0;
            for (const u8 x : host)
                set += x;
            EXPECT_GT(set, 0u);

            for (int t = 0; t < 2; ++t)
            {
                std::vector<i32> schema, binding;
                row_labels(suite, rows, ids, L, t, schema, binding);
                const cuda::DeviceBuffer dschema = to_device(ctx, schema, s), dbinding = to_device(ctx, binding, s);
                const cuda::DeviceMasks::Schemas sa{static_cast<const i32*>(dschema.data())};
                for (u32 depth = 0; depth < L; ++depth)
                {
                    SCOPED_TRACE("depth " + std::to_string(depth));
                    std::vector<u8> hobj(N * O);
                    rl::parameter_masks(suite, view, ids.data(), schema.data(), binding.data(), L, depth, hobj.data(), O);
                    masks.objects(b, sa, static_cast<const i32*>(dbinding.data()), L, depth, static_cast<u8*>(out.data()),
                                  O, static_cast<u64*>(views.data()), static_cast<u32*>(st.data()), s);
                    EXPECT_EQ(to_host<u8>(out.data(), N * O, s), hobj);
                    bool past = false;
                    for (u64 i = 0; i < N; ++i)
                    {
                        const auto& plans = suite.task(static_cast<u32>(ids[i]))->compiled().schemas;
                        past = past || (schema[i] >= 0 && static_cast<u32>(schema[i]) < plans.size() &&
                                        depth >= plans[static_cast<u32>(schema[i])].arity);
                    }
                    EXPECT_EQ(status(), past ? cuda::maskk::k_bad_depth : 0u);
                }
                std::vector<u8> happ(N);
                rl::is_applicable(suite, view, ids.data(), schema.data(), binding.data(), L, happ.data());
                masks.applicable(b, sa, static_cast<const i32*>(dbinding.data()), L, static_cast<u8*>(out.data()),
                                 static_cast<u32*>(st.data()), s);
                EXPECT_EQ(to_host<u8>(out.data(), N, s), happ);
                EXPECT_EQ(status(), 0u);
                u64 yes = 0;
                for (const u8 x : happ)
                    yes += x;
                EXPECT_GT(yes, 0u);
                EXPECT_LT(yes, N);
                // one schema for every row (SchemaArg::value)
                const i32 one = schema[0] >= 0 && static_cast<u32>(schema[0]) < S ? schema[0] : 0;
                std::vector<i32> same(N, one);
                rl::is_applicable(suite, view, ids.data(), same.data(), binding.data(), L, happ.data());
                masks.applicable(b, {nullptr, one}, static_cast<const i32*>(dbinding.data()), L,
                                 static_cast<u8*>(out.data()), static_cast<u32*>(st.data()), s);
                EXPECT_EQ(to_host<u8>(out.data(), N, s), happ);
            }

            // the conditions the host functions throw for are status bits
            if (L > 1)
            {
                masks.applicable(b, {nullptr, 0}, static_cast<const i32*>(dids.data()), 0, static_cast<u8*>(out.data()),
                                 static_cast<u32*>(st.data()), s);
                bool wide = false;
                for (u64 i = 0; i < N; ++i)
                    wide = wide || suite.task(static_cast<u32>(ids[i]))->compiled().schemas.at(0).arity > 0;
                EXPECT_EQ(status(), wide ? cuda::maskk::k_narrow_binding : 0u);
            }
            std::vector<i32> bad = ids;
            bad[N / 2] = static_cast<i32>(suite.size());
            const cuda::DeviceBuffer dbad = to_device(ctx, bad, s);
            masks.schemas({b.states, N, W, W, static_cast<const i32*>(dbad.data())}, static_cast<u8*>(out.data()),
                          static_cast<u64*>(views.data()), static_cast<u32*>(st.data()), s);
            EXPECT_EQ(status(), cuda::maskk::k_bad_task_id);
        }
}

TEST(DeviceMasks, Unsupported)
{
    for (Set& s : sets(TaskOptions::Atoms::Dense))
        if (s.name == "miconic-simpleadl")
        {
            const rl::TaskSuitePtr suite = rl::TaskSuite::of(s.table);
            // goal axioms: the masks run on the host; conditional effects alone do not stop them
            EXPECT_NE(cuda::DeviceMasks::unsupported(*suite).find("axioms"), std::string::npos)
                << cuda::DeviceMasks::unsupported(*suite);
            rl::EnvConfig cfg;
            cfg.actions = rl::ActionSpace::Label;
            EXPECT_NE(cuda::DeviceEnv::fast_unsupported(*suite, cfg), "");
            if (cuda::device_count() == 0)
                continue;
            EXPECT_THROW(cuda::DeviceEnv(context(), suite, cfg), std::invalid_argument);
        }
}

TEST(DeviceLabelEnv, StepEqualsHost)
{
    SKIP_WITHOUT_GPU();
    const cuda::ContextPtr ctx = context();
    for (const auto atoms : {TaskOptions::Atoms::Dense, TaskOptions::Atoms::Reachable})
        for (const Named& n : suites(atoms))
            for (const bool goals : {false, true})
            {
                SCOPED_TRACE(n.name + (goals ? " goals" : ""));
                const rl::TaskSuite& suite = *n.suite;
                rl::EnvConfig cfg = config(7);
                cfg.actions = rl::ActionSpace::Label;
                cfg.dead_end_terminal = !goals;
                rl::HostEnv host(n.suite, cfg);
                cuda::DeviceEnv dev(ctx, n.suite, cfg);
                ASSERT_TRUE(dev.fast());
                ASSERT_EQ(dev.capture_unsupported(), "");
                EXPECT_EQ(dev.cache_schemas(), 1u);
                EXPECT_EQ(dev.cache_view_words(), 0u);
                const u32 I = suite.size(), W = suite.words(), L = dev.label_width();
                std::vector<i32> ids(3 * I + 5);
                for (usize i = 0; i < ids.size(); ++i)
                    ids[i] = static_cast<i32>((i * 7) % I);
                const u64 N = ids.size();
                HostEnvs h(suite, ids, goals);
                DeviceEnvs d(ctx, dev, ids, goals);
                host.reset(h.b, nullptr, h.v.count.data());
                dev.reset(d.b, nullptr, static_cast<i32*>(d.count.data()));
                if (goals)
                {
                    custom_goals(h.v, W);
                    upload(d.gpos.data(), h.v.gpos, d.s);
                    upload(d.gneg.data(), h.v.gneg, d.s);
                    host.refresh(h.b, h.v.count.data());
                    dev.refresh(d.b, static_cast<i32*>(d.count.data()));
                }
                ASSERT_EQ(to_host<i32>(d.count.data(), N, d.s), h.v.count);
                ASSERT_EQ(to_host<u8>(d.at_goal.data(), N, d.s), h.v.at_goal);
                for (int t = 0; t < 14; ++t)
                {
                    std::vector<i32> schema, binding, next(N);
                    row_labels(suite, h.v.states, h.v.task_ids, L, t, schema, binding);
                    for (u64 i = 0; i < N; ++i)
                        next[i] = static_cast<i32>((i + static_cast<u64>(t)) % I);
                    const cuda::DeviceBuffer ds = to_device(ctx, schema, d.s), db = to_device(ctx, binding, d.s),
                                             dn = to_device(ctx, next, d.s);
                    const bool with_next = t % 3 == 1;
                    host.step(h.b, h.out, rl::Labels{schema.data(), binding.data(), L}, with_next ? next.data() : nullptr);
                    dev.step(d.b, d.out,
                             rl::Labels{static_cast<const i32*>(ds.data()), static_cast<const i32*>(db.data()), L},
                             with_next ? static_cast<const i32*>(dn.data()) : nullptr);
                    expect_equal(d.snapshot(), h.v, t);
                    if (HasFatalFailure())
                        return;
                }
                dev.check_errors();
                // refresh after writing states: the counts and goal flags of the rows as they are
                const std::vector<u64> rows = h.v.states;
                dev.refresh(d.b, static_cast<i32*>(d.count.data()));
                host.refresh(h.b, h.v.count.data());
                EXPECT_EQ(to_host<i32>(d.count.data(), N, d.s), h.v.count);
                EXPECT_EQ(to_host<u8>(d.at_goal.data(), N, d.s), h.v.at_goal);
                // the index step and the random policy are refused in label mode
                EXPECT_THROW(dev.step(d.b, d.out), std::invalid_argument);
                EXPECT_THROW(dev.random_actions(nullptr, nullptr, 0, 0, 1, nullptr), std::invalid_argument);
            }
}
}  // namespace
