#pragma once
// Test support of the state-space tests: hashes over the arrays of a space, the invariants of its arrays, and
// generation helpers.

#include "mymyr/datasets/state_space.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/workspace.hpp"
#include "mymyr/frontend/domain.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace mymyr::test::ss
{
using namespace mymyr::datasets;

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

/// FNV over every array of a space (mymyr_datasets --hash).
inline u64 arrays_hash(const StateSpace& S, bool with_states = true)
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

/// The task of `dir`/`problem` below `root` with dense atoms unless `to` asks otherwise; null if a file is missing.
inline TaskPtr dense_task(const std::filesystem::path& root, const std::string& dir, const std::string& problem, TaskOptions to = {})
{
    const auto d = root / dir / "domain.pddl", p = root / dir / problem;
    if (!std::filesystem::exists(d) || !std::filesystem::exists(p))
        return nullptr;
    if (to.atoms == TaskOptions::Atoms::Auto)
        to.atoms = TaskOptions::Atoms::Dense;
    return Task::create(*frontend::load_task(d, p), to);
}

/// Options with unreachable states kept, on `threads` threads.
inline StateSpaceOptions ss_options(u32 threads = 1)
{
    StateSpaceOptions o;
    o.threads = threads;
    o.remove_if_unsolvable = false;
    return o;
}

inline StateSpacePtr space_of(TaskPtr task, const StateSpaceOptions& o)
{
    StateSpaceResult r = generate_state_space(std::move(task), o);
    EXPECT_EQ(r.status, StateSpaceStatus::Ok);
    return r.space;
}

/// Checks the arrays against each other and against an independent search over the task.
inline void check_invariants(const StateSpace& S)
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
    if (S.has_labels())
    {
        const Task& task = *S.task();
        const WorkspaceLease lease = task.workspace();
        Successors& succ = lease->successors();
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
}  // namespace mymyr::test::ss
