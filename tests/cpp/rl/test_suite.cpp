// Task suite tests (host): instances of several domains in one batch (rl/task_suite.hpp), on the instance sets
// of tests/cpp/rl/table_instance_sets.hpp:
//   - construction: create (one table per domain), group (tasks grouped by domain, first-seen order), the global id
//     mapping, widths, schema offsets, the refusals; a table is the suite of its domain (TaskSuite::of);
//   - the atom metadata of a suite is its domains' tables' in global order, with the domain columns;
//   - rl::expand over a suite equals each domain's table's expand apart, byte for byte, for interleaved and grouped
//     batches, frozen and lazy, flat and through the thread pool;
//   - rl::HostEnv over a suite equals each domain's env apart (the rows' RNG streams by env id), step by step;
//     next_task_ids restart rows in instances of other domains; per-env goals; both dead-end modes;
//   - rl::CpuEnvPool over a suite equals rl::HostEnv over it.

#include "../frontend/golden.hpp"
#include "table_instance_sets.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/pool.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_suite.hpp"
#include "mymyr/rl/task_table.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct Domain
{
    std::string name;
    std::vector<TaskPtr> tasks;
    rl::TaskTablePtr table;
};

/// The table instance sets as one table per domain, by name.
std::map<std::string, Domain> domains(TaskOptions::Atoms atoms)
{
    std::map<std::string, Domain> out;
    for (const InstanceSet& s : table_instance_sets())
    {
        std::vector<TaskPtr> tasks = load_set(s, atoms);
        if (tasks.empty())
        {
            ADD_FAILURE() << "missing PDDL of set " << s.name << " (MYMYR_FORK_DATA_DIR)";
            continue;
        }
        auto table = rl::TaskTable::create(tasks);
        out[s.name] = {s.name, std::move(tasks), std::move(table)};
    }
    return out;
}

/// The suites of the tests: two domains, four, and one with conditional effects and goal axioms (the general paths).
std::vector<std::vector<std::string>> suite_names()
{
    return {{"gripper", "blocks"}, {"miconic", "blocks", "gripper", "logistics"}, {"miconic-simpleadl", "gripper", "blocks"}};
}

rl::TaskSuitePtr suite_of(const std::map<std::string, Domain>& ds, const std::vector<std::string>& names)
{
    std::vector<rl::TaskTablePtr> tables;
    for (const auto& n : names)
        tables.push_back(ds.at(n).table);
    return rl::TaskSuite::create(tables);
}

std::string joined(const std::vector<std::string>& names)
{
    std::string s;
    for (const auto& n : names)
        s += (s.empty() ? "" : "+") + n;
    return s;
}

/// A batch over a suite: `per` rows of every instance of every domain (tests::table_rows of each domain's table, at
/// the suite's width), interleaved across the domains (round robin) or grouped (domain by domain); global ids.
struct Batch
{
    std::vector<u64> rows;
    std::vector<i32> ids;
};

Batch suite_rows(const rl::TaskSuite& S, u32 per, bool interleaved)
{
    const u32 D = S.num_domains(), W = S.words();
    std::vector<std::vector<u64>> rows(D);
    std::vector<std::vector<i32>> ids(D);
    for (u32 d = 0; d < D; ++d)
        table_rows(*S.table(d), per, rows[d], ids[d]);
    Batch b;
    auto put = [&](u32 d, u64 k)
    {
        const u32 Wd = S.table(d)->words();
        std::vector<u64> row(W, 0);
        std::copy_n(rows[d].begin() + static_cast<std::ptrdiff_t>(k * Wd), Wd, row.begin());
        b.rows.insert(b.rows.end(), row.begin(), row.end());
        b.ids.push_back(static_cast<i32>(S.global_id(d, static_cast<u32>(ids[d][k]))));
    };
    if (interleaved)
    {
        std::vector<u64> next(D, 0);
        for (bool any = true; any;)
        {
            any = false;
            for (u32 d = 0; d < D; ++d)
                if (next[d] < ids[d].size())
                {
                    put(d, next[d]++);
                    any = true;
                }
        }
    }
    else
        for (u32 d = 0; d < D; ++d)
            for (u64 k = 0; k < ids[d].size(); ++k)
                put(d, k);
    return b;
}

struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    u64 total = 0;
    u32 words_needed = 0;
};

/// rl::expand over a suite (a table: TaskSuite::of) into exactly sized buffers of label width L.
Flat expand_flat(const rl::TaskSuite& S, const u64* rows, u64 n, u32 W, const i32* ids, u32 L, ThreadPool* pool = nullptr)
{
    Flat f;
    rl::Expansion x;
    x.words = W;
    x.label_width = L;
    f.offsets.resize(n + 1);
    x.offsets = f.offsets.data();
    rl::expand(S, rl::StateBatchView{rows, n, W, 0}, ids, x);
    const u64 cap = std::max<u64>(x.total, 1);
    f.succ.resize(cap * W);
    f.parent.resize(cap);
    f.schema.resize(cap);
    f.binding.resize(cap * L);
    f.goal.resize(cap);
    x.capacity = x.total;
    x.succ = f.succ.data();
    x.parent = f.parent.data();
    x.schema = f.schema.data();
    x.binding = f.binding.data();
    x.goal = f.goal.data();
    if (pool)
    {
        rl::ExpandScratch scratch;
        rl::expand(S, rl::StateBatchView{rows, n, W, 0}, ids, x, {}, *pool, scratch);
    }
    else
        rl::expand(S, rl::StateBatchView{rows, n, W, 0}, ids, x);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

/// The expansion of a suite batch assembled from each domain's table's expansion of its rows apart.
Flat expand_apart(const rl::TaskSuite& S, const Batch& b)
{
    const u32 D = S.num_domains(), W = S.words(), L = std::max<u32>(1, S.label_width());
    const u64 N = b.ids.size();
    std::vector<std::vector<u64>> rows(D);
    std::vector<std::vector<i32>> local(D);
    std::vector<u64> pos(N);
    for (u64 r = 0; r < N; ++r)
    {
        const rl::TaskSuite::Ref ref = S.ref(static_cast<u32>(b.ids[r]));
        pos[r] = local[ref.domain].size();
        local[ref.domain].push_back(static_cast<i32>(ref.local));
        rows[ref.domain].insert(rows[ref.domain].end(), b.rows.begin() + static_cast<std::ptrdiff_t>(r * W),
                                b.rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * W));
    }
    std::vector<Flat> part(D);
    for (u32 d = 0; d < D; ++d)
        part[d] = expand_flat(*rl::TaskSuite::of(S.table(d)), rows[d].data(), local[d].size(), W, local[d].data(), L);
    Flat f;
    f.offsets.push_back(0);
    for (u64 r = 0; r < N; ++r)
    {
        const u32 d = S.domain_of(static_cast<u32>(b.ids[r]));
        const Flat& p = part[d];
        const auto lo = static_cast<u64>(p.offsets[pos[r]]), hi = static_cast<u64>(p.offsets[pos[r] + 1]);
        for (u64 j = lo; j < hi; ++j)
        {
            f.succ.insert(f.succ.end(), p.succ.begin() + static_cast<std::ptrdiff_t>(j * W),
                          p.succ.begin() + static_cast<std::ptrdiff_t>((j + 1) * W));
            f.parent.push_back(static_cast<i32>(r));
            f.schema.push_back(p.schema[j]);
            f.binding.insert(f.binding.end(), p.binding.begin() + static_cast<std::ptrdiff_t>(j * L),
                             p.binding.begin() + static_cast<std::ptrdiff_t>((j + 1) * L));
            f.goal.push_back(p.goal[j]);
        }
        f.offsets.push_back(static_cast<i32>(f.parent.size()));
    }
    f.total = f.parent.size();
    for (u32 d = 0; d < D; ++d)
        f.words_needed = std::max(f.words_needed, part[d].words_needed);
    return f;
}

void expect_flat_equal(const Flat& a, const Flat& b)
{
    ASSERT_EQ(a.total, b.total);
    ASSERT_EQ(a.words_needed, b.words_needed);
    ASSERT_EQ(a.offsets, b.offsets);
    const auto n = static_cast<std::ptrdiff_t>(a.total);
    const auto w = n ? static_cast<std::ptrdiff_t>(a.succ.size() / a.total) : 0;
    const auto l = n ? static_cast<std::ptrdiff_t>(a.binding.size() / std::max<u64>(a.total, 1)) : 0;
    ASSERT_TRUE(std::equal(a.parent.begin(), a.parent.begin() + n, b.parent.begin()));
    ASSERT_TRUE(std::equal(a.schema.begin(), a.schema.begin() + n, b.schema.begin()));
    ASSERT_TRUE(std::equal(a.binding.begin(), a.binding.begin() + n * l, b.binding.begin()));
    ASSERT_TRUE(std::equal(a.succ.begin(), a.succ.begin() + n * w, b.succ.begin()));
    ASSERT_TRUE(std::equal(a.goal.begin(), a.goal.begin() + n, b.goal.begin()));
}

/// Arrays of an env batch (rows of W atom words, labels of width L) and one step's outputs.
struct Envs
{
    u32 W = 0, L = 0;
    std::vector<u64> states, final_states, gpos, gneg, draws;
    std::vector<i32> task_ids, steps, count, schema, binding;
    std::vector<u32> env_ids;
    std::vector<f32> reward;
    std::vector<u8> terminated, truncated, invalid, goal;
    rl::EnvBatch b;
    rl::StepOutputs out;

    Envs(u32 words, u32 label_width, std::vector<i32> ids, bool goals = false) : W(words), L(label_width)
    {
        const u64 n = ids.size();
        states.assign(n * W, 0);
        final_states.assign(n * W, 0);
        draws.assign(n, 0);
        task_ids = std::move(ids);
        steps.assign(n, 0);
        count.assign(n, 0);
        schema.assign(n, 0);
        binding.assign(n * L, 0);
        reward.assign(n, 0);
        terminated.assign(n, 0);
        truncated.assign(n, 0);
        invalid.assign(n, 0);
        goal.assign(n, 0);
        if (goals)
        {
            gpos.assign(n * W, 0);
            gneg.assign(n * W, 0);
        }
        b.states = states.data();
        b.rows = n;
        b.words = W;
        b.task_ids = task_ids.data();
        b.goal_pos = goals ? gpos.data() : nullptr;
        b.goal_neg = goals ? gneg.data() : nullptr;
        b.steps = steps.data();
        b.draws = draws.data();
        out = rl::StepOutputs{reward.data(), terminated.data(), truncated.data(), count.data(), final_states.data(),
                              schema.data(), binding.data(), L, invalid.data(), goal.data()};
    }
    void use_env_ids(std::vector<u32> ids)
    {
        env_ids = std::move(ids);
        b.env_ids = env_ids.data();
    }
};

rl::EnvConfig config()
{
    rl::EnvConfig cfg;
    cfg.seed = 0xD0D0;
    cfg.max_steps = 9;
    cfg.goal_reward = 5.0f;
    cfg.dead_end_reward = -3.0f;
    return cfg;
}

template<class T>
std::vector<T> gather(const std::vector<T>& v, const std::vector<u64>& rows, u64 width)
{
    std::vector<T> out;
    for (u64 r : rows)
        out.insert(out.end(), v.begin() + static_cast<std::ptrdiff_t>(r * width),
                   v.begin() + static_cast<std::ptrdiff_t>((r + 1) * width));
    return out;
}

/// Global ids of a mixed batch: every instance of the suite `per` times, the domains interleaved.
std::vector<i32> mixed_ids(const rl::TaskSuite& S, u32 per)
{
    std::vector<i32> ids;
    const u32 I = S.size();
    for (u32 k = 0; k < per * I; ++k)
        ids.push_back(static_cast<i32>((k * 7 + k / I) % I));
    return ids;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ the suite

TEST(TaskSuite, CreateGroupAndIds)
{
    const auto ds = domains(TaskOptions::Atoms::Frozen);
    for (const auto& names : suite_names())
    {
        SCOPED_TRACE(joined(names));
        const rl::TaskSuitePtr S = suite_of(ds, names);
        const u32 D = static_cast<u32>(names.size());
        ASSERT_EQ(S->num_domains(), D);
        EXPECT_FALSE(S->single_domain());
        EXPECT_STREQ(S->noun(), "suite");
        u32 I = 0, W = 0, L = 0, P = 0, O = 0;
        for (u32 d = 0; d < D; ++d)
        {
            const rl::TaskTable& T = *S->table(d);
            EXPECT_EQ(S->table(d), ds.at(names[d]).table);
            EXPECT_EQ(S->schema_offset(d), P);
            for (u32 i = 0; i < T.size(); ++i)
            {
                // create(): table 0's instances first, then table 1's, ...
                const u32 g = I + i;
                EXPECT_EQ(S->global_id(d, i), g);
                EXPECT_EQ(S->domain_of(g), d);
                EXPECT_EQ(S->local_id(g), i);
                EXPECT_EQ(S->task(g), T.task(i));
                EXPECT_EQ(&S->table_of(g), &T);
                EXPECT_EQ(&S->instance(g), &T.instance(i));
            }
            I += T.size();
            W = std::max(W, T.words());
            L = std::max(L, T.label_width());
            P += T.num_schemas();
            O = std::max(O, T.num_schemas());
            EXPECT_EQ(S->domain_name(d), T.task(0)->data().domain_name);
        }
        EXPECT_EQ(S->size(), I);
        EXPECT_EQ(S->words(), W);
        EXPECT_EQ(S->label_width(), L);
        EXPECT_EQ(S->max_schemas(), O);
        EXPECT_EQ(S->schema_offsets().back(), P);
        EXPECT_NE(S->fingerprint(), S->table(0)->fingerprint());
        EXPECT_EQ(S->fingerprint(), suite_of(ds, names)->fingerprint());
        // task ids
        std::vector<i32> ids{0, static_cast<i32>(I) - 1};
        EXPECT_NO_THROW(S->check_task_ids(ids.data(), ids.size()));
        ids[1] = static_cast<i32>(I);
        EXPECT_THROW(S->check_task_ids(ids.data(), ids.size()), std::invalid_argument);
        ids[1] = -1;
        EXPECT_THROW(S->check_task_ids(ids.data(), ids.size()), std::invalid_argument);
        // group(): the tasks of the domains interleaved keep their order as global ids; domains in first-seen order
        std::vector<TaskPtr> tasks;
        std::vector<u32> dom;
        for (u32 k = 0;; ++k)
        {
            bool any = false;
            for (u32 d = D; d-- > 0;)  // the last domain first
                if (k < S->table(d)->size())
                {
                    tasks.push_back(S->table(d)->task(k));
                    dom.push_back(d);
                    any = true;
                }
            if (!any)
                break;
        }
        const rl::TaskSuitePtr G = rl::TaskSuite::group(tasks);
        ASSERT_EQ(G->num_domains(), D);
        ASSERT_EQ(G->size(), I);
        for (u32 g = 0; g < I; ++g)
        {
            EXPECT_EQ(G->task(g), tasks[g]);
            EXPECT_EQ(G->domain_of(g), D - 1 - dom[g]);  // the last domain was seen first
            EXPECT_EQ(G->table(G->domain_of(g))->task(G->local_id(g)), tasks[g]);
            EXPECT_EQ(G->global_id(G->domain_of(g), G->local_id(g)), g);
        }
    }
    // refusals: two tables of one domain, an empty list, a null table
    const auto& g = ds.at("gripper");
    const auto& b = ds.at("blocks");
    try
    {
        (void)rl::TaskSuite::create({g.table, b.table, rl::TaskTable::single(g.tasks[1])});
        ADD_FAILURE() << "two tables of one domain were accepted";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("mymyr: TaskSuite: tables 0 and 2 are of one domain"), std::string::npos)
            << e.what();
    }
    EXPECT_THROW((void)rl::TaskSuite::create({}), std::invalid_argument);
    EXPECT_THROW((void)rl::TaskSuite::create({g.table, nullptr}), std::invalid_argument);
    EXPECT_THROW((void)rl::TaskSuite::group({}), std::invalid_argument);
    EXPECT_THROW((void)rl::TaskSuite::group({g.tasks[0], nullptr}), std::invalid_argument);
    // a table is the suite of its domain: global ids are its ids; group() of one domain's tasks is one table
    const rl::TaskSuitePtr one = rl::TaskSuite::of(g.table);
    EXPECT_TRUE(one->single_domain());
    EXPECT_STREQ(one->noun(), "table");
    EXPECT_EQ(one->table(0), g.table);
    EXPECT_EQ(one->fingerprint(), g.table->fingerprint());
    EXPECT_EQ(one->words(), g.table->words());
    EXPECT_EQ(one->max_schemas(), g.table->num_schemas());
    for (u32 i = 0; i < g.table->size(); ++i)
    {
        EXPECT_EQ(one->domain_of(i), 0u);
        EXPECT_EQ(one->local_id(i), i);
        EXPECT_EQ(one->global_id(0, i), i);
    }
    const rl::TaskSuitePtr grouped = rl::TaskSuite::group(g.tasks);
    EXPECT_TRUE(grouped->single_domain());
    EXPECT_EQ(grouped->size(), g.table->size());
}

TEST(TaskSuite, AtomMetadataIsTheDomainsInGlobalOrder)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto ds = domains(atoms);
        for (const auto& names : suite_names())
        {
            SCOPED_TRACE(joined(names) + (atoms == TaskOptions::Atoms::Frozen ? " frozen" : " lazy"));
            // a suite with a domain's instances in another order than its table's (group(): global ids interleave)
            std::vector<TaskPtr> tasks;
            for (u32 k = 0; k < 6; ++k)
                for (const auto& n : names)
                    if (k < ds.at(n).tasks.size())
                        tasks.push_back(ds.at(n).tasks[k]);
            const rl::TaskSuitePtr S = rl::TaskSuite::group(tasks);
            const u32 D = S->num_domains(), I = S->size();
            const rl::ArrayBundle m = rl::suite_atom_metadata(*S);
            std::vector<rl::ArrayBundle> per;
            for (u32 d = 0; d < D; ++d)
                per.push_back(rl::table_atom_metadata(*S->table(d)));
            auto ints = [](const rl::ArrayBundle& b, const std::string& k)
            {
                const rl::ArrayInfo* a = b.find(k);
                EXPECT_NE(a, nullptr) << k;
                std::vector<i64> v;
                if (!a)
                    return v;
                const void* p = b.data(*a);
                for (u64 i = 0; i < a->elements(); ++i)
                    switch (a->dtype)
                    {
                        case rl::DType::I32: v.push_back(static_cast<const i32*>(p)[i]); break;
                        case rl::DType::U32: v.push_back(static_cast<const u32*>(p)[i]); break;
                        case rl::DType::I64: v.push_back(static_cast<const i64*>(p)[i]); break;
                        case rl::DType::U8: v.push_back(static_cast<const u8*>(p)[i]); break;
                        case rl::DType::I8: v.push_back(static_cast<const i8*>(p)[i]); break;
                        default: ADD_FAILURE() << k << ": not an integer array"; return v;
                    }
                return v;
            };
            const auto domain = ints(m, "domain"), local = ints(m, "local_id"), aoff = ints(m, "atom_offsets"),
                       apred = ints(m, "atom_pred"), poff = ints(m, "pred_offsets"), soff = ints(m, "schema_offsets"),
                       ooff = ints(m, "object_offsets"), nobj = ints(m, "num_objects");
            ASSERT_EQ(domain.size(), I);
            ASSERT_EQ(poff.size(), D + 1);
            ASSERT_EQ(soff.size(), D + 1);
            EXPECT_EQ(m.scalar("num_domains"), D);
            EXPECT_EQ(m.scalar("num_instances"), I);
            i64 objects = 0;
            for (u32 g = 0; g < I; ++g)
            {
                const rl::TaskSuite::Ref r = S->ref(g);
                ASSERT_EQ(domain[g], r.domain);
                ASSERT_EQ(local[g], r.local);
                const rl::ArrayBundle& t = per[r.domain];
                const auto taoff = ints(t, "atom_offsets"), tapred = ints(t, "atom_pred"), tnobj = ints(t, "num_objects");
                // the instance's atoms: its table's rows, predicate ids the domain table's
                ASSERT_EQ(aoff[g + 1] - aoff[g], taoff[r.local + 1] - taoff[r.local]);
                for (i64 j = 0; j < aoff[g + 1] - aoff[g]; ++j)
                    ASSERT_EQ(apred[static_cast<u64>(aoff[g] + j)], tapred[static_cast<u64>(taoff[r.local] + j)]);
                ASSERT_EQ(nobj[g], tnobj[r.local]);
                ASSERT_EQ(ooff[g], objects);
                objects += nobj[g];
            }
            for (u32 d = 0; d < D; ++d)
            {
                EXPECT_EQ(poff[d + 1] - poff[d], per[d].scalar("num_predicates"));
                EXPECT_EQ(soff[d + 1] - soff[d], S->table(d)->num_schemas());
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ expand

TEST(TaskSuiteExpand, SuiteEqualsTheDomainsApart)
{
    ThreadPool pool(4);
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto ds = domains(atoms);
        for (const auto& names : suite_names())
            for (const bool interleaved : {true, false})
            {
                SCOPED_TRACE(joined(names) + (atoms == TaskOptions::Atoms::Frozen ? " frozen" : " lazy") +
                             (interleaved ? " interleaved" : " grouped"));
                const rl::TaskSuitePtr S = suite_of(ds, names);
                const Batch b = suite_rows(*S, 6, interleaved);
                const u32 W = S->words(), L = std::max<u32>(1, S->label_width());
                const Flat apart = expand_apart(*S, b);
                ASSERT_GT(apart.total, 0u);
                expect_flat_equal(expand_flat(*S, b.rows.data(), b.ids.size(), W, b.ids.data(), L), apart);
                expect_flat_equal(expand_flat(*S, b.rows.data(), b.ids.size(), W, b.ids.data(), L, &pool), apart);
                // goal flags of the rows (is_goal) are the domains'
                std::vector<u8> g(b.ids.size()), h(b.ids.size());
                rl::is_goal(*S, rl::StateBatchView{b.rows.data(), b.ids.size(), W, 0}, b.ids.data(), g.data());
                for (u64 r = 0; r < b.ids.size(); ++r)
                {
                    const rl::TaskSuite::Ref ref = S->ref(static_cast<u32>(b.ids[r]));
                    const i32 lid = static_cast<i32>(ref.local);
                    rl::is_goal(*S->table(ref.domain), rl::StateBatchView{b.rows.data() + r * W, 1, W, 0}, &lid, &h[r]);
                }
                EXPECT_EQ(g, h);
                if (::testing::Test::HasFatalFailure())
                    return;
            }
    }
}

TEST(TaskSuiteExpand, TaskIdsAreGlobal)
{
    const auto ds = domains(TaskOptions::Atoms::Frozen);
    const rl::TaskSuitePtr S = suite_of(ds, suite_names()[0]);
    const Batch b = suite_rows(*S, 2, true);
    std::vector<i32> ids = b.ids;
    rl::Expansion x;
    x.words = S->words();
    const rl::StateBatchView in{b.rows.data(), ids.size(), S->words(), 0};
    EXPECT_THROW(rl::expand(*S, in, nullptr, x), std::invalid_argument);  // several instances need task ids
    for (const i32 bad : {static_cast<i32>(S->size()), -1})
    {
        ids[3] = bad;
        try
        {
            rl::expand(*S, in, ids.data(), x);
            ADD_FAILURE() << "task id " << bad << " was accepted";
        }
        catch (const std::invalid_argument& e)
        {
            EXPECT_NE(std::string(e.what()).find("outside the suite's " + std::to_string(S->size()) + " instances"),
                      std::string::npos)
                << e.what();
        }
    }
}

// ------------------------------------------------------------------------------------------------ envs

TEST(TaskSuiteEnv, SuiteEqualsTheDomainEnvsApart)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
    {
        const auto ds = domains(atoms);
        for (const auto& names : suite_names())
            for (const auto dead_end : {rl::DeadEnd::NoSuccessors, rl::DeadEnd::None})
            {
                SCOPED_TRACE(joined(names) + (atoms == TaskOptions::Atoms::Frozen ? " frozen" : " lazy") +
                             (dead_end == rl::DeadEnd::None ? " no dead ends" : ""));
                const rl::TaskSuitePtr S = suite_of(ds, names);
                const u32 D = S->num_domains(), W = S->words(), L = std::max<u32>(1, S->label_width());
                const std::vector<i32> ids = mixed_ids(*S, 3);
                const u64 N = ids.size();
                rl::EnvConfig cfg = config();
                cfg.dead_end = dead_end;
                rl::HostEnv env(S, cfg);
                Envs m(W, L, ids);
                env.reset(m.b, nullptr, m.count.data());
                // each domain's rows apart, on its table's env, with the rows' RNG streams (env ids) and the suite's
                // widths
                std::vector<std::unique_ptr<rl::HostEnv>> one;
                std::vector<Envs> apart;
                std::vector<std::vector<u64>> rows(D);
                for (u64 r = 0; r < N; ++r)
                    rows[S->domain_of(static_cast<u32>(ids[r]))].push_back(r);
                for (u32 d = 0; d < D; ++d)
                {
                    std::vector<i32> local;
                    std::vector<u32> env_ids;
                    for (const u64 r : rows[d])
                    {
                        local.push_back(static_cast<i32>(S->local_id(static_cast<u32>(ids[r]))));
                        env_ids.push_back(static_cast<u32>(r));
                    }
                    one.push_back(std::make_unique<rl::HostEnv>(S->table(d), cfg));
                    apart.emplace_back(W, L, std::move(local));
                    apart.back().use_env_ids(std::move(env_ids));
                    one[d]->reset(apart.back().b, nullptr, apart.back().count.data());
                }
                for (u32 g = 0; g < S->size(); ++g)
                    ASSERT_EQ(env.initial_count(g), one[S->domain_of(g)]->initial_count(S->local_id(g)));
                for (int t = 0; t < 30; ++t)
                {
                    env.step(m.b, m.out);
                    for (u32 d = 0; d < D; ++d)
                    {
                        Envs& e = apart[d];
                        one[d]->step(e.b, e.out);
                        const auto& r = rows[d];
                        ASSERT_EQ(gather(m.states, r, W), e.states) << "step " << t << " domain " << d;
                        ASSERT_EQ(gather(m.final_states, r, W), e.final_states);
                        ASSERT_EQ(gather(m.reward, r, 1), e.reward);
                        ASSERT_EQ(gather(m.terminated, r, 1), e.terminated);
                        ASSERT_EQ(gather(m.truncated, r, 1), e.truncated);
                        ASSERT_EQ(gather(m.count, r, 1), e.count);
                        ASSERT_EQ(gather(m.schema, r, 1), e.schema);
                        ASSERT_EQ(gather(m.binding, r, L), e.binding);
                        ASSERT_EQ(gather(m.goal, r, 1), e.goal);
                        ASSERT_EQ(gather(m.steps, r, 1), e.steps);
                        ASSERT_EQ(gather(m.invalid, r, 1), e.invalid);
                    }
                    ASSERT_EQ(m.task_ids, ids);  // autoreset keeps the instance
                }
            }
    }
}

TEST(TaskSuiteEnv, NextTaskIdsAcrossDomainsAndPerEnvGoals)
{
    const auto ds = domains(TaskOptions::Atoms::Frozen);
    for (const auto& names : suite_names())
    {
        SCOPED_TRACE(joined(names));
        const rl::TaskSuitePtr S = suite_of(ds, names);
        const u32 I = S->size(), W = S->words(), L = std::max<u32>(1, S->label_width());
        std::vector<i32> ids;
        for (u32 k = 0; k < 3 * I; ++k)
            ids.push_back(static_cast<i32>(k % I));
        rl::EnvConfig cfg = config();
        cfg.max_steps = 3;
        rl::HostEnv env(S, cfg);
        // next_task_ids: finished rows restart in an instance of the next domain
        Envs e(W, L, ids);
        env.reset(e.b);
        auto next_of = [&](i32 g)
        {
            const rl::TaskSuite::Ref r = S->ref(static_cast<u32>(g));
            const u32 d = (r.domain + 1) % S->num_domains();
            return static_cast<i32>(S->global_id(d, r.local % S->table(d)->size()));
        };
        std::vector<i32> next(ids.size());
        for (u64 r = 0; r < ids.size(); ++r)
            next[r] = next_of(ids[r]);
        u64 moved = 0;
        for (int t = 0; t < 8; ++t)
        {
            const std::vector<i32> before = e.task_ids;
            env.step(e.b, e.out, {}, next.data());
            for (u64 r = 0; r < ids.size(); ++r)
            {
                const bool done = e.terminated[r] || e.truncated[r];
                ASSERT_EQ(e.task_ids[r], done ? next[r] : before[r]);
                if (!done)
                    continue;
                ++moved;
                // the new instance's initial row at the suite's width (its extra words zero)
                const auto g = static_cast<u32>(next[r]);
                const auto& init = S->instance(g).init;
                const u32 Wd = S->table_of(g).words();
                ASSERT_TRUE(std::equal(init.begin(), init.begin() + Wd, e.states.begin() + static_cast<std::ptrdiff_t>(r * W)));
                for (u32 w = Wd; w < W; ++w)
                    ASSERT_EQ(e.states[r * W + w], 0u);
                ASSERT_EQ(e.count[r], static_cast<i32>(env.initial_count(g)));
                ASSERT_EQ(e.steps[r], 0);
            }
            for (u64 r = 0; r < ids.size(); ++r)
                next[r] = next_of(e.task_ids[r]);
        }
        EXPECT_GT(moved, 0u);
        // per-env goals: equal to the instances' goals they give the same episodes (suites without derived goals)
        bool derived = false;
        for (u32 g = 0; g < I; ++g)
            derived = derived || S->instance(g).goal_derived || S->instance(g).goal_unsatisfiable;
        if (derived)
        {
            Envs g(W, L, ids, true);
            EXPECT_THROW(env.reset(g.b), std::invalid_argument);
            continue;
        }
        Envs a(W, L, ids), g(W, L, ids, true);
        env.reset(a.b);
        env.reset(g.b);
        for (u64 r = 0; r < ids.size(); ++r)
        {
            const auto& in = S->instance(static_cast<u32>(ids[r]));
            const u32 Wd = S->table_of(static_cast<u32>(ids[r])).words();
            ASSERT_TRUE(std::equal(in.goal_pos.begin(), in.goal_pos.begin() + Wd, g.gpos.begin() + static_cast<std::ptrdiff_t>(r * W)));
        }
        for (u64 r = 0; r < ids.size(); ++r)
            next[r] = next_of(ids[r]);
        for (int t = 0; t < 12; ++t)
        {
            env.step(a.b, a.out, {}, next.data());
            env.step(g.b, g.out, {}, next.data());
            ASSERT_EQ(a.states, g.states);
            ASSERT_EQ(a.task_ids, g.task_ids);
            ASSERT_EQ(a.goal, g.goal);
            ASSERT_EQ(a.reward, g.reward);
            // the autoreset into another domain's instance writes that instance's goal masks
            for (u64 r = 0; r < ids.size(); ++r)
            {
                const auto& in = S->instance(static_cast<u32>(g.task_ids[r]));
                const u32 Wd = S->table_of(static_cast<u32>(g.task_ids[r])).words();
                if (g.terminated[r] || g.truncated[r])
                {
                    ASSERT_TRUE(std::equal(in.goal_pos.begin(), in.goal_pos.begin() + Wd,
                                           g.gpos.begin() + static_cast<std::ptrdiff_t>(r * W)));
                }
            }
        }
    }
}

TEST(TaskSuitePool, EqualsTheHostEnv)
{
    const auto ds = domains(TaskOptions::Atoms::Frozen);
    for (const auto& names : suite_names())
    {
        SCOPED_TRACE(joined(names));
        const rl::TaskSuitePtr S = suite_of(ds, names);
        const u32 W = S->words(), L = std::max<u32>(1, S->label_width());
        const std::vector<i32> ids = mixed_ids(*S, 2);
        const u64 N = ids.size();
        const rl::EnvConfig cfg = config();
        rl::HostEnv env(S, cfg);
        Envs ref(W, L, ids);
        env.reset(ref.b, nullptr, ref.count.data());
        rl::CpuEnvPool pool(S, cfg, N, {.threads = 3});
        EXPECT_EQ(pool.suite(), S);
        const rl::PoolBatch r0 = pool.reset(nullptr, 0, ids.data(), nullptr, nullptr);
        ASSERT_EQ(r0.rows, N);
        auto expect_rows = [&](const rl::PoolBatch& p, bool step)
        {
            for (u64 e = 0; e < N; ++e)
            {
                ASSERT_TRUE(std::equal(p.states.begin() + static_cast<std::ptrdiff_t>(e * W),
                                       p.states.begin() + static_cast<std::ptrdiff_t>((e + 1) * W),
                                       ref.states.begin() + static_cast<std::ptrdiff_t>(e * W)));
                ASSERT_EQ(p.task_ids[e], ref.task_ids[e]);
                ASSERT_EQ(p.count[e], ref.count[e]);
                if (!step)
                    continue;
                ASSERT_EQ(p.reward[e], ref.reward[e]);
                ASSERT_EQ(p.terminated[e], ref.terminated[e]);
                ASSERT_EQ(p.truncated[e], ref.truncated[e]);
                ASSERT_EQ(p.schema[e], ref.schema[e]);
                for (u32 k = 0; k < L; ++k)
                    ASSERT_EQ(p.binding[e * p.label_width + k], ref.binding[e * L + k]);
            }
        };
        expect_rows(r0, false);
        std::vector<i32> next(N);
        for (int t = 0; t < 20; ++t)
        {
            for (u64 e = 0; e < N; ++e)
                next[e] = static_cast<i32>((static_cast<u32>(ref.task_ids[e]) + 5 + static_cast<u32>(t)) % S->size());
            const bool curriculum = t % 2 == 1;
            env.step(ref.b, ref.out, {}, curriculum ? next.data() : nullptr);
            const rl::PoolBatch p = pool.step(nullptr, 0, {}, curriculum ? next.data() : nullptr);
            expect_rows(p, true);
            if (::testing::Test::HasFatalFailure())
                return;
        }
    }
}
