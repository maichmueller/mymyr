// Task tables on the host: task tables (rl/task_table.hpp), mixed-instance expand (rl/expand.hpp), environments (rl/env.hpp)
// and the CPU pool (rl/pool.hpp) on the instance sets of tests/cpp/rl/table_instance_sets.hpp (PDDL through the front end):
//   - a table checks that its instances are of one domain; the domain predicates' table ids are the domain's (the
//     same in two tables over different instances), problem-local predicates (goal axioms) get ids of their own per
//     instance; its atom metadata and object offsets round-trip to the instances' atoms and objects by name;
//   - a mixed batch expands (flat and padded, one thread or a pool, whole or split) to exactly the rows, labels and goal
//     flags of the single-instance expansions of its rows;
//   - a mixed HostEnv batch equals single-instance HostEnvs over the same rows and RNG streams, step by step (states,
//     labels, rewards, terminated / truncated, final states, counts, autoreset); next_task_ids and per-env goals;
//   - the CPU pool equals the HostEnv for any thread count, any split of the envs into sends and any send order, with
//     sends of several home ranges from several caller threads at once and recycled batches.

#include "../frontend/golden.hpp"
#include "table_instance_sets.hpp"
#include "mymyr/core/thread_pool.hpp"
#include "mymyr/rl/env.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/pool.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_table.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace mymyr;
using namespace mymyr::test;

namespace
{
struct Flat
{
    std::vector<u64> succ;
    std::vector<i32> parent, schema, binding, offsets;
    std::vector<u8> goal;
    u64 total = 0;
    u32 words_needed = 0;
};

Flat expand_flat(const rl::TaskTable& table, const u64* rows, u64 n, u32 W, const i32* ids, ThreadPool* pool = nullptr)
{
    const u32 NN = table.numeric_words(), RW = W + NN, L = std::max<u32>(1, table.label_width());
    Flat f;
    rl::Expansion x;
    x.words = W;
    x.numeric_words = NN;
    x.label_width = L;
    f.offsets.resize(n + 1);
    x.offsets = f.offsets.data();
    rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, ids, x);
    const u64 cap = std::max<u64>(x.total, 1);
    f.succ.resize(cap * RW);
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
        rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, ids, x, {}, *pool, scratch);
    }
    else
        rl::expand(table, rl::StateBatchView{rows, n, W, 0, NN}, ids, x);
    f.total = x.total;
    f.words_needed = x.words_needed;
    return f;
}

struct Set
{
    std::string name;
    std::vector<TaskPtr> tasks;
    rl::TaskTablePtr table;
};

std::vector<Set> sets(TaskOptions::Atoms atoms)
{
    std::vector<Set> out;
    for (const InstanceSet& s : table_instance_sets())
    {
        std::vector<TaskPtr> tasks = load_set(s, atoms);
        if (tasks.empty())
        {
            ADD_FAILURE() << "missing PDDL of set " << s.name << " (MYMYR_FORK_DATA_DIR)";
            continue;
        }
        auto table = rl::TaskTable::create(tasks);
        out.push_back({s.name, std::move(tasks), std::move(table)});
    }
    return out;
}

u32 bucket(u32 words) { return words <= 2 ? 2 : words <= 4 ? 4 : words <= 16 ? 16 : 64; }
}  // namespace

TEST(Table, SetsSpanSizesAndBuckets)
{
    for (const Set& s : sets(TaskOptions::Atoms::Frozen))
    {
        SCOPED_TRACE(s.name);
        u32 lo = ~0u, hi = 0;
        std::vector<u32> buckets;
        for (const auto& in : s.table->instances())
        {
            lo = std::min(lo, in.num_objects);
            hi = std::max(hi, in.num_objects);
            buckets.push_back(bucket(in.words));
        }
        std::sort(buckets.begin(), buckets.end());
        buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
        EXPECT_GE(s.table->size(), 4u);
        EXPECT_LE(s.table->size(), 8u);
        EXPECT_GE(hi, 3 * lo) << lo << " .. " << hi << " objects";
        EXPECT_GE(buckets.size(), 2u);
        EXPECT_EQ(s.table->max_objects(), hi);
    }
}

TEST(Table, OneDomainIsChecked)
{
    const auto all = table_instance_sets();
    const auto blocks = load_set(all[0], TaskOptions::Atoms::Frozen);
    const auto gripper = load_set(all[1], TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(blocks.empty());
    ASSERT_FALSE(gripper.empty());
    try
    {
        (void)rl::TaskTable::create({blocks[0], blocks[1], gripper[0]});
        FAIL() << "a table of two domains";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("mymyr: TaskTable: instance 2 is not of the domain of instance 0"), std::string::npos)
            << e.what();
    }
    EXPECT_THROW((void)rl::TaskTable::create({}), std::invalid_argument);
    EXPECT_THROW((void)rl::TaskTable::create({blocks[0], nullptr}), std::invalid_argument);
    // a table of one
    const auto one = rl::TaskTable::single(blocks[2]);
    EXPECT_EQ(one->size(), 1u);
    EXPECT_EQ(one->words(), std::max<u32>(1, blocks[2]->max_words()));
    EXPECT_NE(one->fingerprint(), rl::TaskTable::single(blocks[3])->fingerprint());
    EXPECT_EQ(one->fingerprint(), rl::TaskTable::single(blocks[2])->fingerprint());
}

TEST(Table, PredicatesAndMetadataRoundTrip)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const Set& s : sets(atoms))
        {
            SCOPED_TRACE(s.name);
            const rl::TaskTable& T = *s.table;
            // predicates: the domain's first, in every instance's order; then each instance's problem-local ones
            const u32 PD = T.num_domain_predicates(), P = static_cast<u32>(T.predicate_names().size());
            ASSERT_LE(PD, P);
            for (u32 p = 0; p < P; ++p)
                EXPECT_EQ(T.predicate_instances()[p] < 0, p < PD) << p;
            u32 locals = 0;
            std::map<std::string, u32> local_names;
            for (u32 i = 0; i < T.size(); ++i)
            {
                const auto& D = T.task(i)->data();
                u32 d = 0;
                for (u32 p = 0; p < D.predicates.size(); ++p)
                {
                    const u32 t = T.instance(i).pred_map[p];
                    ASSERT_LT(t, P);
                    EXPECT_EQ(T.predicate_names()[t], D.str(D.predicates[p].name));
                    EXPECT_EQ(T.predicate_arities()[t], D.predicates[p].arity);
                    if (t < PD)
                    {
                        EXPECT_EQ(t, d++);
                    }
                    else
                    {
                        EXPECT_EQ(T.predicate_instances()[t], static_cast<i32>(i));
                        EXPECT_EQ(T.predicate_kinds()[t], rl::k_pred_derived);
                        ++locals;
                        ++local_names[T.predicate_names()[t]];
                    }
                }
                EXPECT_EQ(d, PD);
            }
            EXPECT_EQ(PD + locals, P);
            if (s.name == "miconic-simpleadl")
            {
                // the quantified goals (-q) are goal axioms: problem-local, one table predicate per instance even where
                // two problems name theirs alike
                EXPECT_GE(locals, 3u);
                u32 shared = 0;
                for (const auto& [name, n] : local_names)
                    shared = std::max(shared, n);
                EXPECT_GE(shared, 2u) << "two problems' goal axioms of one name";
            }
            else
            {
                EXPECT_EQ(locals, 0u);
            }
            // atom metadata: rows atom_offsets[i] + slot name instance i's atoms (table predicates, local objects)
            const rl::ArrayBundle m = rl::table_atom_metadata(T);
            const auto* off = static_cast<const i64*>(m.data(*m.find("atom_offsets")));
            const auto* pred = static_cast<const i32*>(m.data(*m.find("atom_pred")));
            const rl::ArrayInfo& ai = *m.find("atom_args");
            const auto* args = static_cast<const i32*>(m.data(ai));
            const u32 A = static_cast<u32>(ai.shape[1]);
            const auto* objs = static_cast<const i64*>(m.data(*m.find("object_offsets")));
            EXPECT_EQ(m.scalar("num_instances"), T.size());
            EXPECT_EQ(m.scalar("num_predicates"), P);
            EXPECT_EQ(m.scalar("num_domain_predicates"), PD);
            {
                const auto* owner = static_cast<const i32*>(m.data(*m.find("pred_instance")));
                EXPECT_EQ(std::vector<i32>(owner, owner + P), T.predicate_instances());
            }
            for (u32 i = 0; i < T.size(); ++i)
            {
                const Task& task = *T.task(i);
                EXPECT_EQ(objs[i + 1] - objs[i], task.num_objects());
                ASSERT_GE(off[i + 1] - off[i], 1);
                for (i64 r = off[i]; r < off[i + 1]; ++r)
                {
                    const u32 slot = static_cast<u32>(r - off[i]);
                    const u32* rec = task.atoms().record(AtomKind::Fluent, slot);
                    const auto& P = task.data().predicates[rec[0]];
                    std::string want = "(" + std::string(task.data().str(P.name)), got = "(" + T.predicate_names()[static_cast<u32>(pred[r])];
                    for (u32 k = 0; k < A; ++k)
                    {
                        const i32 o = args[static_cast<u64>(r) * A + k];
                        if (k < P.arity)
                        {
                            ASSERT_GE(o, 0);
                            got += " " + task.object_name(ObjectId{static_cast<u32>(o)});
                            want += " " + task.object_name(ObjectId{rec[1 + k]});
                        }
                        else
                            EXPECT_EQ(o, -1);
                    }
                    ASSERT_EQ(got + ")", want + ")");
                    ASSERT_EQ(task.format(SlotId{slot}), want + ")");
                }
            }
        }
}

TEST(Table, DomainPredicateIdsAreTheDomains)
{
    // a "training" table of the small instances and an "evaluation" table of the large ones (another order, and a table
    // of one): the same table ids for the same domain predicates; the problem-local predicates are each table's own
    for (const InstanceSet& set : table_instance_sets())
    {
        SCOPED_TRACE(set.name);
        const std::vector<TaskPtr> t = load_set(set, TaskOptions::Atoms::Frozen);
        ASSERT_GE(t.size(), 4u);
        const u64 n = t.size();
        const auto train = rl::TaskTable::create({t[0], t[1], t[2]});
        const auto eval = rl::TaskTable::create({t[n - 1], t[n - 2], t[3]});
        const auto one = rl::TaskTable::single(t[n - 1]);
        for (const auto* other : {eval.get(), one.get()})
        {
            ASSERT_EQ(train->num_domain_predicates(), other->num_domain_predicates());
            for (u32 p = 0; p < train->num_domain_predicates(); ++p)
            {
                EXPECT_EQ(train->predicate_names()[p], other->predicate_names()[p]);
                EXPECT_EQ(train->predicate_arities()[p], other->predicate_arities()[p]);
                EXPECT_EQ(train->predicate_kinds()[p], other->predicate_kinds()[p]);
            }
        }
        // an atom of a domain predicate has the same predicate id in both tables' metadata: t[n - 1]'s atoms, instance 0
        // of eval and of one, and t[0]'s, instance 0 of train
        for (const auto& [table, task] :
             {std::pair{train.get(), t[0].get()}, std::pair{eval.get(), t[n - 1].get()}, std::pair{one.get(), t[n - 1].get()}})
        {
            const rl::ArrayBundle m = rl::table_atom_metadata(*table);
            const auto* off = static_cast<const i64*>(m.data(*m.find("atom_offsets")));
            const auto* pred = static_cast<const i32*>(m.data(*m.find("atom_pred")));
            for (i64 r = off[0]; r < off[1]; ++r)
            {
                const u32* rec = task->atoms().record(AtomKind::Fluent, static_cast<u32>(r));
                const auto& D = task->data();
                const std::string name(D.str(D.predicates[rec[0]].name));
                const u32 id = static_cast<u32>(pred[r]);
                ASSERT_EQ(table->predicate_names()[id], name);
                if (id < table->num_domain_predicates())
                {
                    EXPECT_EQ(train->predicate_names()[id], name);
                }
            }
        }
    }
}

TEST(TableExpand, MixedBatchEqualsSingleInstanceExpansions)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const Set& s : sets(atoms))
        {
            SCOPED_TRACE(s.name + (atoms == TaskOptions::Atoms::Frozen ? " frozen" : " lazy"));
            const rl::TaskTable& T = *s.table;
            std::vector<u64> rows;
            std::vector<i32> ids;
            table_rows(T, 6, rows, ids);
            const u32 W = T.words(), RW = W + T.numeric_words(), L = std::max<u32>(1, T.label_width());
            const u64 N = ids.size();
            const Flat mixed = expand_flat(T, rows.data(), N, W, ids.data());
            ASSERT_EQ(mixed.words_needed <= W, true);
            ThreadPool pool(3);
            const Flat pooled = expand_flat(T, rows.data(), N, W, ids.data(), &pool);
            EXPECT_EQ(pooled.succ, mixed.succ);
            EXPECT_EQ(pooled.binding, mixed.binding);
            EXPECT_EQ(pooled.goal, mixed.goal);
            EXPECT_EQ(pooled.offsets, mixed.offsets);
            // split: two halves give the same rows
            const u64 h = N / 2;
            const Flat a = expand_flat(T, rows.data(), h, W, ids.data());
            const Flat b = expand_flat(T, rows.data() + h * RW, N - h, W, ids.data() + h);
            ASSERT_EQ(a.total + b.total, mixed.total);
            EXPECT_TRUE(std::equal(a.succ.begin(), a.succ.begin() + static_cast<std::ptrdiff_t>(a.total * RW), mixed.succ.begin()));
            EXPECT_TRUE(std::equal(b.schema.begin(), b.schema.begin() + static_cast<std::ptrdiff_t>(b.total),
                                   mixed.schema.begin() + static_cast<std::ptrdiff_t>(a.total)));
            // single-instance expansions of the same rows
            for (u32 i = 0; i < T.size(); ++i)
            {
                std::vector<u64> mine;
                std::vector<u64> which;
                for (u64 r = 0; r < N; ++r)
                    if (static_cast<u32>(ids[r]) == i)
                    {
                        mine.insert(mine.end(), rows.begin() + static_cast<std::ptrdiff_t>(r * RW),
                                    rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * RW));
                        which.push_back(r);
                    }
                const auto one = rl::TaskTable::single(s.tasks[i]);
                const Flat f = expand_flat(*one, mine.data(), which.size(), W, nullptr);
                for (u64 k = 0; k < which.size(); ++k)
                {
                    const u64 r = which[k];
                    const i32 c = f.offsets[k + 1] - f.offsets[k];
                    ASSERT_EQ(mixed.offsets[r + 1] - mixed.offsets[r], c) << "row " << r;
                    for (i32 j = 0; j < c; ++j)
                    {
                        const u64 jm = static_cast<u64>(mixed.offsets[r] + j), js = static_cast<u64>(f.offsets[k] + j);
                        ASSERT_TRUE(std::equal(mixed.succ.begin() + static_cast<std::ptrdiff_t>(jm * RW),
                                               mixed.succ.begin() + static_cast<std::ptrdiff_t>((jm + 1) * RW),
                                               f.succ.begin() + static_cast<std::ptrdiff_t>(js * RW)));
                        ASSERT_EQ(mixed.schema[jm], f.schema[js]);
                        ASSERT_EQ(mixed.parent[jm], static_cast<i32>(r));
                        for (u32 q = 0; q < L; ++q)
                            ASSERT_EQ(mixed.binding[jm * L + q], f.binding[js * L + q]);
                        ASSERT_EQ(mixed.goal[jm], f.goal[js]);
                    }
                }
            }
            // the padded view of the mixed batch
            i32 maxc = 0;
            for (u64 r = 0; r < N; ++r)
                maxc = std::max(maxc, mixed.offsets[r + 1] - mixed.offsets[r]);
            const u32 K = static_cast<u32>(std::max(1, maxc));
            std::vector<i32> index(N * K), count(N), schema(N * K), binding(N * K * L);
            std::vector<u8> mask(N * K), goal(N * K);
            std::vector<u64> succ(N * K * RW);
            rl::Expansion x;
            x.capacity = mixed.total;
            x.words = W;
            x.numeric_words = T.numeric_words();
            x.label_width = L;
            x.succ = const_cast<u64*>(mixed.succ.data());
            x.schema = const_cast<i32*>(mixed.schema.data());
            x.binding = const_cast<i32*>(mixed.binding.data());
            x.goal = const_cast<u8*>(mixed.goal.data());
            x.offsets = const_cast<i32*>(mixed.offsets.data());
            rl::PaddedExpansion p{K, index.data(), mask.data(), count.data(), W, succ.data(), schema.data(), L,
                                  binding.data(), goal.data(), false, T.numeric_words()};
            rl::pad(x, N, p);
            EXPECT_FALSE(p.overflow);
            for (u64 r = 0; r < N; ++r)
                for (u32 k = 0; k < K; ++k)
                {
                    const bool valid = static_cast<i32>(k) < count[r];
                    ASSERT_EQ(mask[r * K + k], valid ? 1 : 0);
                    if (valid)
                    {
                        ASSERT_EQ(schema[r * K + k], mixed.schema[static_cast<u64>(mixed.offsets[r]) + k]);
                    }
                }
        }
}

TEST(TableExpand, TaskIdsAreChecked)
{
    const auto s = sets(TaskOptions::Atoms::Frozen);
    ASSERT_FALSE(s.empty());
    const rl::TaskTable& T = *s[0].table;
    std::vector<u64> rows;
    std::vector<i32> ids;
    table_rows(T, 1, rows, ids);
    ids[0] = static_cast<i32>(T.size());
    rl::Expansion x;
    x.words = T.words();
    EXPECT_THROW(rl::expand(T, rl::StateBatchView{rows.data(), ids.size(), T.words(), 0}, ids.data(), x), std::invalid_argument);
    ids[0] = -1;
    EXPECT_THROW(rl::expand(T, rl::StateBatchView{rows.data(), ids.size(), T.words(), 0}, ids.data(), x), std::invalid_argument);
}

namespace
{
/// Arrays of an env batch over a table and one step's outputs.
struct Envs
{
    u32 W = 0, RW = 0, L = 0;
    std::vector<u64> states, final_states, gpos, gneg, draws;
    std::vector<i32> task_ids, steps, count, schema, binding;
    std::vector<u32> env_ids;
    std::vector<f32> reward;
    std::vector<u8> terminated, truncated, invalid, goal;
    rl::EnvBatch b;
    rl::StepOutputs out;

    Envs(const rl::TaskTable& T, std::vector<i32> ids, bool goals = false)
    {
        const u64 n = ids.size();
        W = T.words();
        RW = W + T.numeric_words();
        L = std::max<u32>(1, T.label_width());
        states.assign(n * RW, 0);
        final_states.assign(n * RW, 0);
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
        b.numeric_words = T.numeric_words();
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
    cfg.seed = 0xC0FFEE;
    cfg.max_steps = 9;
    cfg.goal_reward = 5.0f;
    cfg.dead_end_reward = -3.0f;
    return cfg;
}

/// Rows of instance i of a mixed batch: their indices.
std::vector<u64> rows_of(const std::vector<i32>& ids, u32 i)
{
    std::vector<u64> r;
    for (u64 k = 0; k < ids.size(); ++k)
        if (static_cast<u32>(ids[k]) == i)
            r.push_back(k);
    return r;
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
}  // namespace

TEST(TableEnv, MixedBatchEqualsSingleInstanceEnvs)
{
    for (const auto atoms : {TaskOptions::Atoms::Frozen, TaskOptions::Atoms::Lazy})
        for (const Set& s : sets(atoms))
        {
            SCOPED_TRACE(s.name + (atoms == TaskOptions::Atoms::Frozen ? " frozen" : " lazy"));
            const rl::TaskTable& T = *s.table;
            const u32 I = T.size(), per = 5;
            std::vector<i32> ids;
            for (u32 k = 0; k < per * I; ++k)
                ids.push_back(static_cast<i32>((k * 3 + k / I) % I));
            const rl::EnvConfig cfg = config();
            rl::HostEnv mixed(s.table, cfg);
            Envs m(T, ids);
            mixed.reset(m.b);
            Envs pooled(T, ids);
            mixed.reset(pooled.b);
            ThreadPool pool(4);
            std::vector<std::unique_ptr<rl::HostEnv>> one;
            std::vector<Envs> single;
            std::vector<std::vector<u64>> rows(I);
            for (u32 i = 0; i < I; ++i)
            {
                rows[i] = rows_of(ids, i);
                one.push_back(std::make_unique<rl::HostEnv>(rl::TaskTable::single(s.tasks[i]), cfg));
                // rows of the table's width (wider than the instance needs: valid) and the mixed rows' RNG streams
                Envs e(T, std::vector<i32>(rows[i].size(), 0));
                std::vector<u32> env;
                for (u64 r : rows[i])
                    env.push_back(static_cast<u32>(r));
                e.use_env_ids(std::move(env));
                e.b.task_ids = nullptr;
                single.push_back(std::move(e));
                one[i]->reset(single.back().b);
            }
            for (int t = 0; t < 30; ++t)
            {
                mixed.step(m.b, m.out);
                mixed.step(pooled.b, pooled.out, {}, nullptr, &pool);
                ASSERT_EQ(pooled.states, m.states);
                ASSERT_EQ(pooled.reward, m.reward);
                ASSERT_EQ(pooled.binding, m.binding);
                for (u32 i = 0; i < I; ++i)
                {
                    Envs& e = single[i];
                    one[i]->step(e.b, e.out);
                    const auto& r = rows[i];
                    ASSERT_EQ(gather(m.states, r, m.RW), e.states) << "step " << t << " instance " << i;
                    ASSERT_EQ(gather(m.final_states, r, m.RW), e.final_states);
                    ASSERT_EQ(gather(m.reward, r, 1), e.reward);
                    ASSERT_EQ(gather(m.terminated, r, 1), e.terminated);
                    ASSERT_EQ(gather(m.truncated, r, 1), e.truncated);
                    ASSERT_EQ(gather(m.count, r, 1), e.count);
                    ASSERT_EQ(gather(m.schema, r, 1), e.schema);
                    ASSERT_EQ(gather(m.binding, r, m.L), e.binding);
                    ASSERT_EQ(gather(m.goal, r, 1), e.goal);
                    ASSERT_EQ(gather(m.steps, r, 1), e.steps);
                }
                ASSERT_EQ(m.task_ids, ids);  // autoreset keeps the instance
            }
        }
}

TEST(TableEnv, NextTaskIdsAndPerEnvGoals)
{
    for (const Set& s : sets(TaskOptions::Atoms::Frozen))
    {
        SCOPED_TRACE(s.name);
        const rl::TaskTable& T = *s.table;
        const u32 I = T.size();
        std::vector<i32> ids;
        for (u32 k = 0; k < 4 * I; ++k)
            ids.push_back(static_cast<i32>(k % I));
        rl::EnvConfig cfg = config();
        cfg.max_steps = 3;
        rl::HostEnv env(s.table, cfg);
        // next_task_ids: truncated rows restart in the next instance
        Envs e(T, ids);
        env.reset(e.b);
        std::vector<i32> next(ids.size());
        for (u64 r = 0; r < ids.size(); ++r)
            next[r] = static_cast<i32>((static_cast<u32>(ids[r]) + 1) % I);
        for (int t = 0; t < 7; ++t)
        {
            const std::vector<i32> before = e.task_ids;
            env.step(e.b, e.out, {}, next.data());
            for (u64 r = 0; r < ids.size(); ++r)
            {
                const bool done = e.terminated[r] || e.truncated[r];
                ASSERT_EQ(e.task_ids[r], done ? next[r] : before[r]);
                if (done)
                {
                    const auto& init = T.instance(static_cast<u32>(next[r])).init;
                    ASSERT_TRUE(std::equal(init.begin(), init.end(), e.states.begin() + static_cast<std::ptrdiff_t>(r * e.RW)));
                    ASSERT_EQ(e.count[r], static_cast<i32>(env.initial_count(static_cast<u32>(next[r]))));
                    ASSERT_EQ(e.steps[r], 0);
                }
            }
            for (u64 r = 0; r < ids.size(); ++r)
                next[r] = static_cast<i32>((static_cast<u32>(e.task_ids[r]) + 1) % I);
        }
        // per-env goals equal to the instances' goals give the same episodes (instances without derived goals)
        bool derived = false;
        for (const auto& in : T.instances())
            derived = derived || in.goal_derived || in.goal_unsatisfiable;
        if (derived)
        {
            Envs g(T, ids, true);
            EXPECT_THROW(env.reset(g.b), std::invalid_argument);
            continue;
        }
        Envs a(T, ids), g(T, ids, true);
        env.reset(a.b);
        env.reset(g.b);
        for (u64 r = 0; r < ids.size(); ++r)
        {
            const auto& in = T.instance(static_cast<u32>(ids[r]));
            ASSERT_TRUE(std::equal(in.goal_pos.begin(), in.goal_pos.end(), g.gpos.begin() + static_cast<std::ptrdiff_t>(r * g.W)));
        }
        for (int t = 0; t < 12; ++t)
        {
            env.step(a.b, a.out);
            env.step(g.b, g.out);
            ASSERT_EQ(a.states, g.states);
            ASSERT_EQ(a.goal, g.goal);
            ASSERT_EQ(a.reward, g.reward);
        }
        // custom goals: one atom of the first successor of the initial state; the goal flag is the mask test
        Envs c(T, ids, true);
        env.reset(c.b);
        for (u64 r = 0; r < ids.size(); ++r)
        {
            std::fill_n(c.gpos.begin() + static_cast<std::ptrdiff_t>(r * c.W), c.W, u64{0});
            std::fill_n(c.gneg.begin() + static_cast<std::ptrdiff_t>(r * c.W), c.W, u64{0});
        }
        std::vector<i64> first(ids.size(), 0);
        Envs probe(T, ids);
        env.reset(probe.b);
        env.step(probe.b, probe.out, rl::Actions{first.data()});
        for (u64 r = 0; r < ids.size(); ++r)
        {
            // the highest atom of the reached state that the initial state lacks (none: any atom it has)
            const u64* s2 = probe.final_states.data() + r * probe.RW;
            const auto& init = T.instance(static_cast<u32>(ids[r])).init;
            i64 fresh = -1, any = -1;
            for (u32 w = 0; w < c.W; ++w)
                for (u32 k = 0; k < 64; ++k)
                    if ((s2[w] >> k) & 1)
                    {
                        any = static_cast<i64>(w * 64 + k);
                        if (!((init[w] >> k) & 1))
                            fresh = any;
                    }
            const i64 bit = fresh >= 0 ? fresh : any;
            ASSERT_GE(bit, 0);
            c.gpos[r * c.W + static_cast<u64>(bit) / 64] |= u64{1} << (bit % 64);
        }
        const std::vector<u64> custom = c.gpos;
        env.step(c.b, c.out, rl::Actions{first.data()});
        for (u64 r = 0; r < ids.size(); ++r)
        {
            const u64* s2 = c.final_states.data() + r * c.RW;
            bool hit = true;
            for (u32 w = 0; w < c.W; ++w)
                hit = hit && (s2[w] & custom[r * c.W + w]) == custom[r * c.W + w];
            EXPECT_TRUE(hit);
            EXPECT_EQ(c.goal[r], 1);
            EXPECT_EQ(c.terminated[r], 1);
            // the autoreset restored the instance's own goal
            const auto& in = T.instance(static_cast<u32>(c.task_ids[r]));
            EXPECT_TRUE(std::equal(in.goal_pos.begin(), in.goal_pos.end(), c.gpos.begin() + static_cast<std::ptrdiff_t>(r * c.W)));
        }
    }
}

namespace
{
void expect_batch_rows(const rl::PoolBatch& p, u64 pos, const Envs& e, u64 row, bool step)
{
    const u32 RW = e.RW;
    ASSERT_TRUE(std::equal(p.states.begin() + static_cast<std::ptrdiff_t>(pos * RW),
                           p.states.begin() + static_cast<std::ptrdiff_t>((pos + 1) * RW),
                           e.states.begin() + static_cast<std::ptrdiff_t>(row * RW)))
        << "env " << row;
    ASSERT_EQ(p.task_ids[pos], e.task_ids[row]);
    ASSERT_EQ(p.count[pos], e.count[row]);
    ASSERT_EQ(p.steps[pos], e.steps[row]);
    if (!step)
        return;
    ASSERT_EQ(p.reward[pos], e.reward[row]);
    ASSERT_EQ(p.terminated[pos], e.terminated[row]);
    ASSERT_EQ(p.truncated[pos], e.truncated[row]);
    ASSERT_EQ(p.goal[pos], e.goal[row]);
    ASSERT_EQ(p.invalid[pos], e.invalid[row]);
    ASSERT_EQ(p.schema[pos], e.schema[row]);
    ASSERT_TRUE(std::equal(p.final_states.begin() + static_cast<std::ptrdiff_t>(pos * RW),
                           p.final_states.begin() + static_cast<std::ptrdiff_t>((pos + 1) * RW),
                           e.final_states.begin() + static_cast<std::ptrdiff_t>(row * RW)));
    for (u32 k = 0; k < e.L; ++k)
        ASSERT_EQ(p.binding[pos * p.label_width + k], e.binding[row * e.L + k]);
}
}  // namespace

TEST(TablePool, EqualsTheHostEnvForAnyThreadsSplitsAndOrder)
{
    for (const Set& s : sets(TaskOptions::Atoms::Frozen))
    {
        SCOPED_TRACE(s.name);
        const rl::TaskTable& T = *s.table;
        const u32 I = T.size(), N = 6 * I;
        std::vector<i32> ids;
        for (u32 k = 0; k < N; ++k)
            ids.push_back(static_cast<i32>((5 * k + 1) % I));
        const rl::EnvConfig cfg = config();
        rl::HostEnv env(s.table, cfg);
        for (u32 threads : {1u, 3u, 8u})
        {
            SCOPED_TRACE(threads);
            Envs ref(T, ids);
            env.reset(ref.b, nullptr, ref.count.data());
            rl::CpuEnvPool pool(s.table, cfg, N, {.threads = threads});
            const rl::PoolBatch r0 = pool.reset(nullptr, 0, ids.data(), nullptr, nullptr);
            ASSERT_EQ(r0.rows, N);
            for (u64 e = 0; e < N; ++e)
                expect_batch_rows(r0, e, ref, e, false);
            std::vector<u32> even, odd;
            for (u32 e = 0; e < N; ++e)
                (e % 2 ? odd : even).push_back(e);
            for (int t = 0; t < 20; ++t)
            {
                env.step(ref.b, ref.out);
                if (t % 3 == 0)
                {
                    const rl::PoolBatch p = pool.step(nullptr, 0, {}, nullptr);
                    for (u64 e = 0; e < N; ++e)
                        expect_batch_rows(p, e, ref, e, true);
                }
                else
                {
                    // two sends in either order, received as one batch (recv(min_rows)) or by ticket
                    const bool swap = t % 3 == 2;
                    const auto& first = swap ? odd : even;
                    const auto& second = swap ? even : odd;
                    const u64 t1 = pool.send(first.data(), first.size(), {}, nullptr);
                    const u64 t2 = pool.send(second.data(), second.size(), {}, nullptr);
                    EXPECT_EQ(pool.pending(), 2u);
                    if (swap)
                    {
                        const rl::PoolBatch b2 = pool.recv_ticket(t2);
                        const rl::PoolBatch b1 = pool.recv_ticket(t1);
                        for (u64 k = 0; k < b1.rows; ++k)
                            expect_batch_rows(b1, k, ref, b1.env_ids[k], true);
                        for (u64 k = 0; k < b2.rows; ++k)
                            expect_batch_rows(b2, k, ref, b2.env_ids[k], true);
                    }
                    else
                    {
                        const rl::PoolBatch both = pool.recv(N);
                        ASSERT_EQ(both.rows, N);
                        for (u64 k = 0; k < both.rows; ++k)
                            expect_batch_rows(both, k, ref, both.env_ids[k], true);
                    }
                    EXPECT_EQ(pool.pending(), 0u);
                }
            }
            // given actions (every action index 1, invalid where there is no second successor)
            std::vector<i64> act(N, 1);
            env.step(ref.b, ref.out, rl::Actions{act.data()});
            const rl::PoolBatch p = pool.step(nullptr, 0, rl::Actions{act.data()}, nullptr);
            for (u64 e = 0; e < N; ++e)
                expect_batch_rows(p, e, ref, e, true);
            // errors: an env in flight, an env given twice, nothing to receive
            const u32 two[2] = {0, 0};
            EXPECT_THROW((void)pool.send(two, 2, {}, nullptr), std::invalid_argument);
            const u64 tk = pool.send(two, 1, {}, nullptr);
            EXPECT_THROW((void)pool.send(two, 1, {}, nullptr), std::invalid_argument);
            EXPECT_THROW((void)pool.reset(two, 1, nullptr, nullptr, nullptr), std::invalid_argument);
            (void)pool.recv_ticket(tk);
            EXPECT_THROW((void)pool.recv(), std::invalid_argument);
        }
    }
}

TEST(TablePool, ConcurrentCallersHomeRangesAndRecycledBatches)
{
    // sends large enough for several home ranges per send (the workers steal pieces of each other's ranges), from
    // several caller threads at once, with next_task_ids and recycled batches: every env equals the HostEnv
    for (const Set& s : sets(TaskOptions::Atoms::Frozen))
    {
        SCOPED_TRACE(s.name);
        const rl::TaskTable& T = *s.table;
        const u32 I = T.size(), N = 1031, callers = 3, steps = 8;
        std::vector<i32> ids;
        for (u32 k = 0; k < N; ++k)
            ids.push_back(static_cast<i32>((3 * k + 2) % I));
        std::vector<std::vector<i32>> next(steps, std::vector<i32>(N));
        for (u32 t = 0; t < steps; ++t)
            for (u32 k = 0; k < N; ++k)
                next[t][k] = static_cast<i32>((k + t) % I);
        const rl::EnvConfig cfg = config();
        rl::HostEnv env(s.table, cfg);
        Envs ref(T, ids);
        env.reset(ref.b, nullptr, ref.count.data());
        // the reference after each step: states, task ids, counts and rewards
        std::vector<std::vector<u64>> states;
        std::vector<std::vector<i32>> tids, counts;
        std::vector<std::vector<f32>> rewards;
        for (u32 t = 0; t < steps; ++t)
        {
            env.step(ref.b, ref.out, {}, next[t].data());
            states.push_back(ref.states);
            tids.push_back(ref.task_ids);
            counts.push_back(ref.count);
            rewards.push_back(ref.reward);
        }
        const u32 RW = ref.RW;
        auto check = [&](const rl::PoolBatch& p, u32 t, u32 lo) -> std::string
        {
            for (u64 k = 0; k < p.rows; ++k)
            {
                const u64 e = p.env_ids[k];
                const bool same = e == lo + k && p.task_ids[k] == tids[t][e] && p.count[k] == counts[t][e] &&
                                  p.reward[k] == rewards[t][e] &&
                                  std::equal(p.states.begin() + static_cast<std::ptrdiff_t>(k * RW),
                                             p.states.begin() + static_cast<std::ptrdiff_t>((k + 1) * RW),
                                             states[t].begin() + static_cast<std::ptrdiff_t>(e * RW));
                if (!same)
                    return "step " + std::to_string(t) + " env " + std::to_string(e);
            }
            return {};
        };
        // one caller: whole-batch steps (8 workers: 8 home ranges of about 129 rows), batches recycled
        rl::CpuEnvPool pool(s.table, cfg, N, {.threads = 8});
        (void)pool.reset(nullptr, 0, ids.data(), nullptr, nullptr);
        for (u32 t = 0; t < steps; ++t)
        {
            rl::PoolBatch p = pool.step(nullptr, 0, {}, next[t].data());
            ASSERT_EQ(p.rows, N);
            EXPECT_EQ(check(p, t, 0), "");
            pool.recycle(std::move(p));
        }
        // several callers, each with a third of the envs (two ranges per send), each batch recycled
        rl::CpuEnvPool pool2(s.table, cfg, N, {.threads = 8});
        (void)pool2.reset(nullptr, 0, ids.data(), nullptr, nullptr);
        std::vector<std::string> failures(callers);
        std::vector<std::thread> threads;
        for (u32 c = 0; c < callers; ++c)
            threads.emplace_back(
                [&, c]
                {
                    const u32 lo = N * c / callers, hi = N * (c + 1) / callers;
                    std::vector<u32> mine;
                    for (u32 e = lo; e < hi; ++e)
                        mine.push_back(e);
                    for (u32 t = 0; t < steps; ++t)
                    {
                        const u64 tk = pool2.send(mine.data(), mine.size(), {}, next[t].data() + lo);
                        rl::PoolBatch p = pool2.recv_ticket(tk);
                        if (failures[c].empty())
                            failures[c] = check(p, t, lo);
                        pool2.recycle(std::move(p));
                    }
                });
        for (std::thread& th : threads)
            th.join();
        for (u32 c = 0; c < callers; ++c)
            EXPECT_EQ(failures[c], "") << "caller " << c;
        EXPECT_EQ(pool2.pending(), 0u);
    }
}
