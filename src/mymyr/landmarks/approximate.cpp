// Approximate fact landmarks (landmarks/approximate.hpp).

#include "mymyr/landmarks/approximate.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace mymyr::landmarks
{
using heuristics::RelaxedTask;

namespace
{
constexpr u32 k_inf = std::numeric_limits<u32>::max();

class Generator
{
public:
    Generator(const RelaxedTask& R, const ApproximateFactLandmarkOptions& o)
        : R(R), T(R.task()), L(R.task().atoms().layout()), O(o), m_args(std::max<u32>(1, L.max_arity))
    {
    }

    FactLandmarkGraph run()
    {
        dijkstra();
        fact_landmarks();
        std::vector<std::vector<CanonicalAtom>> disjunctive;
        if (O.max_disjunctive_landmark_size > 0)
            disjunctive = disjunctive_landmarks();
        return FactLandmarkGraph::create(m_landmarks, std::move(disjunctive), std::move(m_orderings), {}, achiever_index());
    }

private:
    [[nodiscard]] bool positive_fluent(u32 prop) const
    {
        return !R.negative(prop) && R.prop_atom(prop) < L.fluent_count;
    }

    /// Step 1: h_max-style Dijkstra with unit firing costs, first achievers per proposition.
    void dijkstra()
    {
        const u32 P = R.num_props(), NO = R.num_ops();
        m_cost.assign(P, k_inf);
        m_first.assign(P, {});
        std::vector<u32> cnt(NO), acc(NO, 0);
        for (u32 o = 0; o < NO; ++o)
            cnt[o] = R.npos(o);
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p))
                m_cost[p] = 0;
        std::vector<std::vector<u32>> buckets(1);
        auto push = [&](u32 c, u32 p)
        {
            if (buckets.size() <= c)
                buckets.resize(c + 1);
            buckets[c].push_back(p);
        };
        // the initial state: its fluent atoms cost 0, their "false" propositions need an operator deleting them
        const formalism::TaskData& D = T.data();
        std::vector<u32> args;
        for (const formalism::GroundAtom& a : D.fluent_init)
        {
            if (L.offset[a.pred.v] == CanonicalLayout::k_none)
                continue;
            args.clear();
            for (ObjectId ob : D.objects_of(a))
                args.push_back(ob.v);
            const CanonicalAtom c = L.encode(a.pred.v, args.data());
            if (c >= L.fluent_count)
                continue;
            const auto [pp, np] = R.fluent_props(c);
            if (pp != RelaxedTask::k_none && m_cost[pp] != 0)
            {
                m_cost[pp] = 0;
                push(0, pp);
            }
            if (np != RelaxedTask::k_none && m_cost[np] == 0)
            {
                m_cost[np] = k_inf;
                for (u32 o : R.pre_of(np))
                    ++cnt[o];
            }
        }
        auto fire = [&](u32 o)
        {
            const u32 val = acc[o] + 1;
            for (u32 q : R.eff(o))
            {
                if (val < m_cost[q])
                {
                    m_cost[q] = val;
                    m_first[q].assign(1, o);
                    push(val, q);
                }
                else if (val == m_cost[q])
                    m_first[q].push_back(o);
            }
        };
        for (u32 o : R.zero_ops())
            if (cnt[o] == 0)
                fire(o);
        for (u32 c = 0; c < buckets.size(); ++c)
            for (usize i = 0; i < buckets[c].size(); ++i)
            {
                const u32 p = buckets[c][i];
                if (m_cost[p] != c)
                    continue;
                for (u32 o : R.pre_of(p))
                {
                    acc[o] = std::max(acc[o], c);
                    if (--cnt[o] == 0)
                        fire(o);
                }
            }
    }

    /// Positive fluent preconditions of an operator: canonical ids, ascending, distinct.
    const std::vector<CanonicalAtom>& pre_atoms(u32 o)
    {
        if (m_pre_done.empty())
        {
            m_pre_done.assign(R.num_ops(), 0);
            m_pre_cache.resize(R.num_ops());
        }
        std::vector<CanonicalAtom>& v = m_pre_cache[o];
        if (m_pre_done[o])
            return v;
        m_pre_done[o] = 1;
        for (u32 p : R.pre(o))
            if (positive_fluent(p))
                v.push_back(R.prop_atom(p));
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }

    [[nodiscard]] std::span<const u32> first_achievers(CanonicalAtom c) const
    {
        const u32 pp = R.fluent_props(c).first;
        if (pp == RelaxedTask::k_none)
            return {};
        return m_first[pp];
    }

    /// Step 2: goal facts and back-chaining through the shared preconditions of the first achievers.
    void fact_landmarks()
    {
        std::deque<CanonicalAtom> work;
        auto add = [&](CanonicalAtom c)
        {
            if (m_is_landmark.insert(c).second)
            {
                m_landmarks.push_back(c);
                work.push_back(c);
            }
        };
        if (O.include_positive_goal_facts)
        {
            const formalism::TaskData& D = T.data();
            std::vector<u32> args;
            for (const formalism::Literal& l : D.literals_of(D.goal))
            {
                if (!l.positive || D.predicates[l.pred.v].kind != formalism::PredKind::Fluent || L.offset[l.pred.v] == CanonicalLayout::k_none)
                    continue;
                args.clear();
                for (formalism::Term t : D.terms_of(l))
                    args.push_back(formalism::term_object(t).v);
                const CanonicalAtom c = L.encode(l.pred.v, args.data());
                if (c < L.fluent_count)
                    add(c);
            }
        }
        while (!work.empty())
        {
            const CanonicalAtom lm = work.front();
            work.pop_front();
            const auto first = first_achievers(lm);
            if (first.empty())
                continue;  // initially true (or unreachable): no predecessors
            std::vector<CanonicalAtom> shared = pre_atoms(first[0]);
            for (usize i = 1; i < first.size() && !shared.empty(); ++i)
            {
                const auto& p = pre_atoms(first[i]);
                std::vector<CanonicalAtom> x;
                std::set_intersection(shared.begin(), shared.end(), p.begin(), p.end(), std::back_inserter(x));
                shared = std::move(x);
            }
            for (CanonicalAtom p : shared)
            {
                if (O.compute_greedy_necessary_orderings)
                    m_orderings.emplace_back(p, lm);
                add(p);
            }
        }
    }

    [[nodiscard]] u32 predicate_of(CanonicalAtom c)
    {
        return L.decode(c, m_args.data());
    }

    /// Step 3: disjunctive landmarks by predicate over the first achievers, expanded layer by layer.
    std::vector<std::vector<CanonicalAtom>> disjunctive_landmarks()
    {
        std::vector<std::vector<CanonicalAtom>> out;
        std::set<CanonicalAtom> expanded(m_landmarks.begin(), m_landmarks.end());
        std::vector<CanonicalAtom> frontier = m_landmarks;
        for (usize depth = 0; !frontier.empty(); ++depth)
        {
            if (O.max_disjunctive_landmark_depth > 0 && depth >= O.max_disjunctive_landmark_depth)
                break;
            std::vector<CanonicalAtom> next;
            for (CanonicalAtom a : frontier)
            {
                const auto first = first_achievers(a);
                if (first.empty())
                    continue;
                // per achiever: predicate -> its positive fluent preconditions over that predicate
                std::vector<std::map<u32, std::vector<CanonicalAtom>>> by_achiever;
                by_achiever.reserve(first.size());
                for (u32 o : first)
                {
                    auto& g = by_achiever.emplace_back();
                    for (CanonicalAtom p : pre_atoms(o))
                        g[predicate_of(p)].push_back(p);
                }
                for (const auto& [pred, atoms0] : by_achiever.front())
                {
                    (void)atoms0;
                    std::vector<CanonicalAtom> members;
                    bool everywhere = true;
                    for (const auto& g : by_achiever)
                    {
                        const auto it = g.find(pred);
                        if (it == g.end())
                        {
                            everywhere = false;
                            break;
                        }
                        members.insert(members.end(), it->second.begin(), it->second.end());
                    }
                    if (!everywhere)
                        continue;
                    std::sort(members.begin(), members.end());
                    members.erase(std::unique(members.begin(), members.end()), members.end());
                    if (members.size() > O.max_disjunctive_landmark_size)
                        continue;
                    for (CanonicalAtom m : members)
                        if (expanded.insert(m).second)
                            next.push_back(m);
                    out.push_back(std::move(members));
                }
            }
            frontier = std::move(next);
        }
        return out;  // FactLandmarkGraph::create sorts, dedups and drops the sets holding a fact landmark
    }

    /// Step 4: the achiever index over the ground actions of the grounding.
    AchieverIndex achiever_index()
    {
        AchieverIndex ix;
        const u32 G = R.num_ground_actions(), NO = R.num_ops();
        // ground actions in canonical order
        std::vector<u32> order(G);
        for (u32 g = 0; g < G; ++g)
            order[g] = g;
        std::sort(order.begin(), order.end(),
                  [&](u32 a, u32 b) { return AchieverIndex::less(R.ground_action_label(a), R.ground_action_label(b)); });
        std::vector<u32> pos(G);
        ix.actions.reserve(G);
        for (u32 i = 0; i < G; ++i)
        {
            pos[order[i]] = i;
            ix.actions.emplace_back(R.ground_action_label(order[i]));
        }
        // achievers: (atom, action position) over every operator adding a fluent atom
        std::vector<std::pair<CanonicalAtom, u32>> ach, first;
        for (u32 o = 0; o < NO; ++o)
        {
            if (R.is_axiom(o))
                continue;
            for (u32 q : R.eff(o))
                if (positive_fluent(q))
                    ach.emplace_back(R.prop_atom(q), pos[R.ground_action(o)]);
        }
        std::sort(ach.begin(), ach.end());
        ach.erase(std::unique(ach.begin(), ach.end()), ach.end());
        for (u32 q = 0; q < R.num_props(); ++q)
            if (positive_fluent(q))
                for (u32 o : m_first[q])
                    if (!R.is_axiom(o))
                        first.emplace_back(R.prop_atom(q), pos[R.ground_action(o)]);
        std::sort(first.begin(), first.end());
        first.erase(std::unique(first.begin(), first.end()), first.end());
        for (const auto& [c, a] : ach)
            if (ix.atoms.empty() || ix.atoms.back() != c)
                ix.atoms.push_back(c);
        const usize NA = ix.atoms.size();
        ix.achievers_begin.assign(NA + 1, 0);
        ix.first_achievers_begin.assign(NA + 1, 0);
        {
            usize i = 0;
            for (usize k = 0; k < NA; ++k)
            {
                while (i < ach.size() && ach[i].first == ix.atoms[k])
                    ix.achievers.push_back(ach[i++].second);
                ix.achievers_begin[k + 1] = static_cast<u32>(ix.achievers.size());
            }
            i = 0;
            for (usize k = 0; k < NA; ++k)
            {
                while (i < first.size() && first[i].first < ix.atoms[k])
                    ++i;  // cannot happen: a first achiever is an achiever
                while (i < first.size() && first[i].first == ix.atoms[k])
                    ix.first_achievers.push_back(first[i++].second);
                ix.first_achievers_begin[k + 1] = static_cast<u32>(ix.first_achievers.size());
            }
        }
        // per action: the landmarks it achieves, first-achieves and uniquely achieves
        std::vector<std::vector<CanonicalAtom>> achieved(G), first_achieved(G), uniquely(G);
        std::vector<CanonicalAtom> lms = m_landmarks;
        std::sort(lms.begin(), lms.end());
        for (CanonicalAtom c : lms)
        {
            const auto a = ix.achievers_of(c);
            for (u32 x : a)
                achieved[x].push_back(c);
            for (u32 x : ix.first_achievers_of(c))
                first_achieved[x].push_back(c);
            if (a.size() == 1)
                uniquely[a[0]].push_back(c);
        }
        auto flatten = [&](const std::vector<std::vector<CanonicalAtom>>& v, std::vector<u32>& begin, std::vector<CanonicalAtom>& flat)
        {
            begin.assign(1, 0);
            for (const auto& x : v)
            {
                flat.insert(flat.end(), x.begin(), x.end());
                begin.push_back(static_cast<u32>(flat.size()));
            }
        };
        flatten(achieved, ix.achieved_begin, ix.achieved);
        flatten(first_achieved, ix.first_achieved_begin, ix.first_achieved);
        flatten(uniquely, ix.uniquely_achieved_begin, ix.uniquely_achieved);
        return ix;
    }

    const RelaxedTask& R;
    const Task& T;
    const CanonicalLayout& L;
    const ApproximateFactLandmarkOptions& O;
    std::vector<u32> m_args;
    std::vector<u32> m_cost;
    std::vector<std::vector<u32>> m_first;  // per proposition: operators of minimal firing cost
    std::vector<std::vector<CanonicalAtom>> m_pre_cache;
    std::vector<u8> m_pre_done;
    std::vector<CanonicalAtom> m_landmarks;
    std::set<CanonicalAtom> m_is_landmark;
    std::vector<std::pair<CanonicalAtom, CanonicalAtom>> m_orderings;
};
}  // namespace

FactLandmarkGraph approximate_fact_landmarks(const heuristics::RelaxedTask& relaxed, const ApproximateFactLandmarkOptions& options)
{
    return Generator(relaxed, options).run();
}

FactLandmarkGraph approximate_fact_landmarks(const Task& task, const ApproximateFactLandmarkOptions& options)
{
    heuristics::GroundingStats stats;
    heuristics::RelaxedTaskOptions ro;
    ro.keep_unreachable_operators = true;
    const auto R = heuristics::RelaxedTask::build(task, options.budget, &stats, ro);
    if (!R)
        throw std::runtime_error("mymyr: approximate fact landmarks need a grounding: " + stats.reason);
    return approximate_fact_landmarks(*R, options);
}
}  // namespace mymyr::landmarks
