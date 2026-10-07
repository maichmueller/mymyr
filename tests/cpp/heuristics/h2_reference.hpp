#pragma once
// Test support: h² by its definition (heuristics/heuristic.hpp), a plain Bellman-Ford over the full pair table of a
// grounding, to check the Dijkstra of src/mymyr/heuristics/h2.cpp and to reproduce mimir's H2Heuristic.

#include "mymyr/heuristics/heuristic.hpp"
#include "mymyr/heuristics/relaxed_task.hpp"
#include "mymyr/task/task.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <span>
#include <vector>

namespace mymyr::test
{
namespace h2_detail
{
inline constexpr u32 k_inf = ~u32{0};
inline constexpr u32 k_none = ~u32{0};
}  // namespace h2_detail

/// The propositions true in s: its atoms' "true" propositions and the "false" propositions of the other atoms.
inline std::vector<u32> true_props(const Task& task, const heuristics::RelaxedTask& R, StateView s)
{
    std::vector<u8> in(R.num_props(), 0);
    for (u32 slot = 0; slot < s.nw * 64; ++slot)
        if (s.contains(SlotId{slot}))
        {
            const auto [pp, np] = R.fluent_props(task.atoms().canonical(AtomKind::Fluent, slot));
            EXPECT_NE(pp, h2_detail::k_none);
            if (pp != h2_detail::k_none)
                in[pp] = 1;
            if (np != h2_detail::k_none)
                in[np] = 2;
        }
    std::vector<u32> out;
    for (u32 p = 0; p < R.num_props(); ++p)
        if (R.negative(p) ? in[p] != 2 : in[p] == 1)
            out.push_back(p);
    return out;
}

/// h² by its definition, Bellman-Ford over the full pair table until nothing changes (unit costs).
class ReferenceH2
{
public:
    /// mimir_deletes: mimir's H2Heuristic instead, which checks the deletes of an action against the negations of
    /// atoms (every atom persists); the two agree with mimir only on tasks without negative literals, conditional
    /// effects and axioms (its other differences).
    explicit ReferenceH2(const heuristics::RelaxedTask& R, bool mimir_deletes = false) : R(R), P(R.num_props()), O(R.num_ops())
    {
        comp.assign(P, h2_detail::k_none);
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p))
            {
                const u32 t = R.fluent_props(R.prop_atom(p)).first;
                comp[p] = t;
                if (t != h2_detail::k_none)
                    comp[t] = p;
            }
        for (u32 o = 0; o < O; ++o)
        {
            pre.emplace_back(R.pre(o).begin(), R.pre(o).end());
            std::sort(pre.back().begin(), pre.back().end());
            pre.back().erase(std::unique(pre.back().begin(), pre.back().end()), pre.back().end());
            add.emplace_back(R.eff(o).begin(), R.eff(o).end());
        }
        auto falsified = [&](u32 o, std::set<u32>& out)
        {
            for (u32 p : R.del(o))
                out.insert(p);
            for (u32 p : add[o])
                if (comp[p] != h2_detail::k_none)
                    out.insert(comp[p]);
        };
        for (u32 o = 0; o < O; ++o)
        {
            std::set<u32> ex;
            if (!mimir_deletes)
            {
                ex.insert(add[o].begin(), add[o].end());
                falsified(o, ex);
            }
            std::vector<u32> sib;
            for (u32 o2 = 0; o2 < O; ++o2)
                if (o2 != o && !R.is_axiom(o) && !R.is_axiom(o2) && R.ground_action(o2) == R.ground_action(o))
                {
                    sib.push_back(o2);
                    if (R.unconditional(o2) && !mimir_deletes)
                        falsified(o2, ex);
                }
            excl.push_back(std::move(ex));
            siblings.push_back(std::move(sib));
        }
    }

    double evaluate(const std::vector<u32>& truth, std::span<const u32> goal)
    {
        h.assign(static_cast<usize>(P) * P, h2_detail::k_inf);
        for (u32 a : truth)
            for (u32 b : truth)
                h[a * P + b] = 0;
        for (bool changed = true; changed;)
        {
            changed = false;
            auto upd = [&](u32 a, u32 b, u32 v)
            {
                if (v < h[a * P + b])
                {
                    h[a * P + b] = h[b * P + a] = v;
                    changed = true;
                }
            };
            for (u32 o = 0; o < O; ++o)
            {
                const u32 c1 = of(pre[o]);
                if (c1 == h2_detail::k_inf)
                    continue;
                const u32 cost = R.is_axiom(o) ? 0 : 1;
                for (u32 p : add[o])
                    for (u32 q : add[o])
                        if (comp[p] != q)
                            upd(p, q, c1 + cost);
                for (u32 o2 : siblings[o])
                {
                    const u32 c2 = of(pre[o2]);
                    if (c2 == h2_detail::k_inf)
                        continue;
                    for (u32 p : add[o])
                        for (u32 q : add[o2])
                            if (p != q && comp[p] != q)
                                upd(p, q, std::max(c1, c2) + cost);
                }
                for (u32 r = 0; r < P; ++r)
                {
                    if (excl[o].contains(r))
                        continue;
                    u32 c = c1;
                    for (u32 x : pre[o])
                        c = std::max(c, h[x * P + r]);
                    if (pre[o].empty())
                        c = h[r * P + r];
                    if (c == h2_detail::k_inf)
                        continue;
                    for (u32 p : add[o])
                        if (comp[p] != r)
                            upd(p, r, c + cost);
                }
            }
        }
        const u32 v = of(std::vector<u32>(goal.begin(), goal.end()));
        return v == h2_detail::k_inf ? heuristics::k_dead_end : static_cast<double>(v);
    }

private:
    [[nodiscard]] u32 of(const std::vector<u32>& set) const
    {
        u32 v = 0;
        for (u32 a : set)
            for (u32 b : set)
                v = std::max(v, h[a * P + b]);
        return v;
    }

    const heuristics::RelaxedTask& R;
    u32 P, O;
    std::vector<u32> comp, h;
    std::vector<std::vector<u32>> pre, add, siblings;
    std::vector<std::set<u32>> excl;
};

}  // namespace mymyr::test
