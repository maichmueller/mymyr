#pragma once
// Test support: LM-cut written out from its definition over a RelaxedTask, recomputing h_max from scratch in every
// round (Bellman-Ford until nothing changes) instead of updating it incrementally. The rules are the documented ones:
// one operator per ground action and precondition, costs shared per ground action, h at least h_max, an operator's supporter the first of its most expensive preconditions unless its
// supporter kept its cost, the goal zone depth first with the border and goal-zone tie-breaks of zero-cost achievers,
// and the cut from the state's propositions along supporter edges.

#include "mymyr/heuristics/relaxed_task.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace mymyr::test
{
class ReferenceLmCut
{
public:
    static constexpr u64 k_inf = std::numeric_limits<u64>::max();

    /// real: the grounding's real costs, else 1 per action (axioms 0).
    ReferenceLmCut(const heuristics::RelaxedTask& R, bool real) : R(R), P(R.num_props()), O(R.num_ops())
    {
        init = P;
        for (u32 o = 0; o < R.num_ops(); ++o)
        {
            std::vector<u32> p;
            for (u32 q : R.pre(o))
                if (std::find(p.begin(), p.end(), q) == p.end())
                    p.push_back(q);
            if (p.empty())
                p.push_back(init);
            const u32 a = R.is_axiom(o) ? ~u32{0} : R.ground_action(o);
            usize at = pre.size();
            if (a != ~u32{0})  // an action's effect instances with equal preconditions are one operator
                for (usize i = 0; i < pre.size(); ++i)
                    if (ga[i] == a && pre[i] == p)
                        at = i;
            if (at == pre.size())
            {
                pre.push_back(std::move(p));
                eff.emplace_back();
                ga.push_back(a);
            }
            eff[at].insert(eff[at].end(), R.eff(o).begin(), R.eff(o).end());
            std::sort(eff[at].begin(), eff[at].end());
            eff[at].erase(std::unique(eff[at].begin(), eff[at].end()), eff[at].end());
        }
        O = static_cast<u32>(pre.size());
        for (u32 g = 0; g < R.num_ground_actions(); ++g)
            base.push_back(real ? R.real_cost(g) : 1);
    }

    /// LM-cut of the state whose cost-0 propositions are `state` (test::true_props) for the goal `goal`, in cost units;
    /// k_inf for a dead end.
    u64 evaluate(const std::vector<u32>& state, const std::vector<u32>& goal)
    {
        const u32 G = P + 1;  // the goal proposition; the goal operator is O
        std::vector<std::vector<u32>> pr = pre;
        pr.push_back(goal);
        std::vector<std::vector<u32>> ef = eff;
        ef.push_back({G});
        std::vector<u32> g2 = ga;
        g2.push_back(~u32{0});
        std::vector<u64> rem = base;
        auto opcost = [&](u32 o) { return g2[o] == ~u32{0} ? u64{0} : rem[g2[o]]; };
        std::vector<u32> start = state;
        start.push_back(init);
        std::vector<u32> supp(O + 1, ~u32{0});
        std::vector<u64> supp_cost(O + 1, k_inf);
        std::vector<u64> cost;
        auto hmax = [&]
        {
            cost.assign(P + 2, k_inf);
            for (u32 p : start)
                cost[p] = 0;
            for (bool changed = true; changed;)
            {
                changed = false;
                for (u32 o = 0; o <= O; ++o)
                {
                    u64 m = 0;
                    for (u32 q : pr[o])
                        m = std::max(m, cost[q]);
                    if (m == k_inf)
                        continue;
                    for (u32 q : ef[o])
                        if (m + opcost(o) < cost[q])
                        {
                            cost[q] = m + opcost(o);
                            changed = true;
                        }
                }
            }
            for (u32 o = 0; o <= O; ++o)
            {
                u32 first = ~u32{0};
                u64 m = 0;
                for (u32 q : pr[o])
                    if (first == ~u32{0} || cost[q] > m)
                    {
                        first = q;
                        m = cost[q];
                    }
                if (m == k_inf)
                    continue;
                if (supp[o] == ~u32{0} || cost[supp[o]] != supp_cost[o])
                    supp[o] = first;
                supp_cost[o] = m;
            }
        };
        std::vector<std::vector<u32>> ach(P + 2);
        for (u32 o = 0; o <= O; ++o)
            for (u32 q : ef[o])
                ach[q].push_back(o);
        hmax();
        if (cost[G] == k_inf)
            return k_inf;
        const u64 h0 = cost[G];
        u64 h = 0;
        while (cost[G] != 0)
        {
            std::vector<u8> zone(P + 2, 0);  // 1 goal zone, 2 before it
            auto no_free_achiever = [&](u32 q)
            { return std::none_of(ach[q].begin(), ach[q].end(), [&](u32 o) { return opcost(o) == 0; }); };
            auto mark = [&](auto&& self, u32 p) -> void
            {
                if (p == ~u32{0} || zone[p] == 1)
                    return;
                zone[p] = 1;
                for (u32 o : ach[p])
                {
                    if (opcost(o) != 0 || supp[o] == ~u32{0})
                        continue;
                    for (u32 q : pr[o])
                        if (cost[q] == supp_cost[o] && (no_free_achiever(q) || zone[q] == 1))
                        {
                            supp[o] = q;
                            break;
                        }
                    self(self, supp[o]);
                }
            };
            mark(mark, G);
            std::vector<u32> stack, cut;
            std::vector<std::vector<u32>> by_supporter(P + 2);
            for (u32 o = 0; o < O; ++o)
                if (supp[o] != ~u32{0})
                    by_supporter[supp[o]].push_back(o);
            for (u32 p : start)
            {
                zone[p] = 2;
                stack.push_back(p);
            }
            while (!stack.empty())
            {
                const u32 p = stack.back();
                stack.pop_back();
                for (u32 o : by_supporter[p])
                {
                    if (std::any_of(ef[o].begin(), ef[o].end(), [&](u32 q) { return zone[q] == 1; }))
                    {
                        cut.push_back(o);
                        continue;
                    }
                    for (u32 q : ef[o])
                        if (zone[q] != 2)
                        {
                            zone[q] = 2;
                            stack.push_back(q);
                        }
                }
            }
            u64 c = k_inf;
            for (u32 o : cut)
                c = std::min(c, opcost(o));
            if (cut.empty() || c == 0)
                return k_inf - 1;  // never: a cut is not empty and has positive costs
            h += c;
            std::vector<u32> gas;
            for (u32 o : cut)
                gas.push_back(g2[o]);
            std::sort(gas.begin(), gas.end());
            gas.erase(std::unique(gas.begin(), gas.end()), gas.end());
            for (u32 a : gas)
                rem[a] -= c;
            hmax();
        }
        return std::max(h, h0);
    }

private:
    const heuristics::RelaxedTask& R;
    u32 P, O, init;
    std::vector<std::vector<u32>> pre, eff;
    std::vector<u32> ga;
    std::vector<u64> base;
};
}  // namespace mymyr::test
