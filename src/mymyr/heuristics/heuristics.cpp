// Heuristic evaluators: blind, goal count, and h_max / h_add / h_FF over mimir's delete relaxation, grounded
// (relaxed_task.hpp: GroundedEval extended with negative propositions, axiom rules and relaxed-plan extraction) or
// lifted (lifted_relaxation.hpp: a cost-bucketed semi-naive fixpoint over the relaxed reachability matchers).

#include "mymyr/heuristics/heuristic.hpp"

#include "lifted_relaxation.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace mymyr::heuristics
{
using namespace formalism;

const char* to_string(Kind k) noexcept
{
    switch (k)
    {
        case Kind::Blind: return "blind";
        case Kind::GoalCount: return "goal_count";
        case Kind::Max: return "max";
        case Kind::Add: return "add";
        case Kind::FF: return "ff";
        case Kind::Custom: return "custom";
    }
    return "?";
}

Kind parse_kind(std::string_view n)
{
    if (n == "blind")
        return Kind::Blind;
    if (n == "goal_count" || n == "gc" || n == "goalcount")
        return Kind::GoalCount;
    if (n == "max" || n == "hmax" || n == "h_max")
        return Kind::Max;
    if (n == "add" || n == "hadd" || n == "h_add")
        return Kind::Add;
    if (n == "ff" || n == "hff" || n == "h_ff")
        return Kind::FF;
    throw std::invalid_argument("mymyr: unknown heuristic '" + std::string(n) + "'");
}

std::shared_ptr<const RelaxedTask> ground(const Task& task, const GroundingBudget& budget, GroundingStats* stats)
{
    return RelaxedTask::build(task, budget, stats);
}

namespace
{
constexpr u32 k_inf = ~u32{0};
constexpr u32 k_none = ~u32{0};
constexpr u32 k_unknown = ~u32{0} - 1;

[[nodiscard]] inline u32 sat_add(u32 a, u32 b) noexcept
{
    const u64 s = static_cast<u64>(a) + b;
    return s >= k_inf ? k_inf - 1 : static_cast<u32>(s);
}
[[nodiscard]] inline Value to_value(u64 h) noexcept { return h >= k_inf ? k_dead_end : static_cast<Value>(h); }

/// Monotone priority queue of the relaxed exploration: buckets for small keys, a binary heap beyond.
class RelaxedQueue
{
public:
    static constexpr u32 k_buckets = 1024;

    RelaxedQueue() : m_b(k_buckets) {}
    void clear()
    {
        if (m_size)
            for (u32 i = m_cur; i <= m_top; ++i)
                m_b[i].clear();
        m_cur = 0;
        m_top = 0;
        m_size = 0;
        m_heap.clear();
    }
    void push(u32 key, u32 x)
    {
        if (key < k_buckets)
        {
            m_b[key].push_back(x);
            m_top = std::max(m_top, key);
            m_cur = std::min(m_cur, key);
            ++m_size;
        }
        else
        {
            m_heap.push_back((static_cast<u64>(key) << 32) | x);
            std::push_heap(m_heap.begin(), m_heap.end(), std::greater<u64>());
        }
    }
    bool pop(u32& key, u32& x)
    {
        if (m_size)
        {
            while (m_b[m_cur].empty())
                ++m_cur;
            x = m_b[m_cur].back();
            m_b[m_cur].pop_back();
            key = m_cur;
            --m_size;
            return true;
        }
        if (m_heap.empty())
            return false;
        std::pop_heap(m_heap.begin(), m_heap.end(), std::greater<u64>());
        key = static_cast<u32>(m_heap.back() >> 32);
        x = static_cast<u32>(m_heap.back());
        m_heap.pop_back();
        return true;
    }

private:
    std::vector<std::vector<u32>> m_b;
    u32 m_cur = 0, m_top = 0;
    u64 m_size = 0;
    std::vector<u64> m_heap;
};

// ================================================================================================ blind
class BlindHeuristic final : public Heuristic
{
public:
    [[nodiscard]] Kind kind() const noexcept override { return Kind::Blind; }
    Value evaluate(StateView) override
    {
        ++m_stats.evaluations;
        return 0;
    }
    Value evaluate(StateView, std::span<const search::GoalSpec::AtomGoal>) override
    {
        ++m_stats.evaluations;
        return 0;
    }
};

// ================================================================================================ goal count
class GoalCountHeuristic final : public Heuristic
{
public:
    explicit GoalCountHeuristic(const Task& task) : m_task(task)
    {
        if (task.compiled().goal.uses_derived)
            m_ws = std::make_unique<Workspace>(task);
    }
    [[nodiscard]] Kind kind() const noexcept override { return Kind::GoalCount; }

    Value evaluate(StateView s) override
    {
        ++m_stats.evaluations;
        const plan::Goal& g = m_task.compiled().goal;
        u32 open = 0;
        if (m_ws)
        {
            Successors& succ = m_ws->successors();
            succ.prepare(s);
            for (const plan::Check& c : g.lits)
                open += !succ.engine().holds(c);
            return open;
        }
        const AtomIndex& atoms = m_task.atoms();
        for (const plan::Check& c : g.lits)  // ground literals: the key is the pattern's base
            open += bits::test(s.w, s.nw, atoms.find(c.pat.base)) != c.pos;
        return open;
    }

    Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> goals) override
    {
        ++m_stats.evaluations;
        if (goals.empty())
            return 0;
        u32 best = k_inf;
        for (const auto& g : goals)
        {
            u32 open = 0;
            for (SlotId p : g.positive)
                open += !s.contains(p);
            for (SlotId p : g.negative)
                open += s.contains(p);
            best = std::min(best, open);
        }
        return best;
    }

private:
    const Task& m_task;
    std::unique_ptr<Workspace> m_ws;
};

using detail::GoalLit;
using detail::LiftedRelaxation;

// ================================================================================================ h_max / h_add / h_FF
class RelaxationHeuristic final : public Heuristic
{
public:
    RelaxationHeuristic(const Task& task, const Options& o)
        : m_task(task), m_kind(o.kind), m_real(o.costs == Costs::Real), m_grounded_only(o.evaluation == Evaluation::Grounded)
    {
        if (m_real)
        {
            const ActionCosts costs(task);
            if (!costs.state_independent())
                throw std::invalid_argument("mymyr: real-cost heuristics need action costs that do not depend on the state");
            if (!costs.integral())
                throw std::invalid_argument("mymyr: real-cost heuristics need integral action costs");
        }
        if (o.evaluation != Evaluation::Lifted)
        {
            const auto t0 = std::chrono::steady_clock::now();
            m_R = o.relaxed;
            if (!m_R)
            {
                GroundingStats st;
                m_R = RelaxedTask::build(task, o.budget, &st);
                if (!m_R && m_grounded_only)
                    throw std::runtime_error("mymyr: grounding the relaxation exceeded the budget: " + st.reason);
            }
            m_stats.grounding_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        if (m_R && m_real && !m_R->real_costs_available())
            throw std::invalid_argument("mymyr: real-cost heuristics need non-negative integer action costs below 2^31");
        if (m_R)
            init_grounded();
    }

    [[nodiscard]] Kind kind() const noexcept override { return m_kind; }
    [[nodiscard]] std::shared_ptr<const RelaxedTask> relaxed() const override { return m_R; }
    [[nodiscard]] bool provides_preferred() const noexcept override { return m_kind == Kind::FF && m_R != nullptr; }

    Value evaluate(StateView s) override
    {
        ++m_stats.evaluations;
        m_plan_valid = false;
        Value v;
        if (m_R && convert(s))
        {
            ++m_stats.grounded;
            v = m_R->goal_unreachable() ? k_dead_end : single(m_R->goal());
        }
        else
        {
            ++m_stats.lifted;
            v = lifted().evaluate(s, task_goal());
        }
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> goals) override
    {
        ++m_stats.evaluations;
        m_plan_valid = false;
        if (goals.empty())
            return 0;
        Value v;
        if (m_R && convert(s) && goal_props(s, goals))
        {
            ++m_stats.grounded;
            v = multi();
        }
        else
        {
            ++m_stats.lifted;
            std::vector<std::vector<GoalLit>> gl;
            const AtomIndex& A = m_task.atoms();
            for (const auto& g : goals)
            {
                auto& lits = gl.emplace_back();
                for (int pol = 0; pol < 2; ++pol)
                    for (SlotId p : pol ? g.negative : g.positive)
                    {
                        const u32* rec = A.record(AtomKind::Fluent, p.v);
                        lits.push_back({rec[0], std::vector<u32>(rec + 1, rec + 1 + m_task.compiled().arity[rec[0]]), pol == 0});
                    }
            }
            v = lifted().evaluate(s, gl);
        }
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    [[nodiscard]] bool preferred(const ActionLabel& a) const override
    {
        if (!m_plan_valid)
            return false;
        const u32 ga = m_R->find_ground_action(a);
        return ga != RelaxedTask::k_none && m_ga_mark[ga] == m_plan_stamp;
    }

    [[nodiscard]] std::vector<Action> relaxed_plan() const override
    {
        std::vector<Action> out;
        if (m_plan_valid)
            for (u32 ga : m_plan)
                out.emplace_back(m_R->ground_action_label(ga));
        return out;
    }

private:
    // ------------------------------------------------------------------------------------ grounded evaluation
    void init_grounded()
    {
        const RelaxedTask& R = *m_R;
        const u32 P = R.num_props(), O = R.num_ops();
        m_cost_init.assign(P, k_inf);
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p))
                m_cost_init[p] = 0;
        m_cost.resize(P);
        m_supp.assign(P, k_none);
        m_gmark.assign(P, 0);
        m_pmark.assign(P, 0);
        m_cnt.resize(O);
        m_acc.resize(O);
        m_last.assign(O, k_none);
        m_npos.resize(O);
        m_opcost.resize(O);
        m_axiom.resize(O);
        for (u32 o = 0; o < O; ++o)
        {
            m_npos[o] = R.npos(o);
            m_axiom[o] = R.is_axiom(o);
            m_opcost[o] = m_axiom[o] ? 0 : (m_real ? R.real_cost(R.ground_action(o)) : 1);
        }
        m_ga_mark.assign(R.num_ground_actions(), 0);
    }

    /// Maps the state's atoms to propositions; false if an atom lies outside the grounded relaxation.
    bool convert(StateView s)
    {
        const AtomIndex& A = m_task.atoms();
        m_state_pos.clear();
        m_state_neg.clear();
        bool ok = true;
        bits::for_each(s.w, s.nw,
                       [&](u64 slot)
                       {
                           if (slot >= m_slot_pos.size())
                           {
                               m_slot_pos.resize(slot + 1, k_unknown);
                               m_slot_neg.resize(slot + 1, k_unknown);
                           }
                           if (m_slot_pos[slot] == k_unknown)
                           {
                               const auto [pp, np] = m_R->fluent_props(A.canonical(AtomKind::Fluent, static_cast<u32>(slot)));
                               m_slot_pos[slot] = pp;
                               m_slot_neg[slot] = np;
                           }
                           const u32 p = m_slot_pos[slot];
                           if (p == k_none)
                           {
                               ok = false;
                               return;
                           }
                           m_state_pos.push_back(p);
                           if (m_slot_neg[slot] != k_none)
                               m_state_neg.push_back(m_slot_neg[slot]);
                       });
        return ok;
    }

    /// Generalized Dijkstra from the converted state until every proposition of `targets` is settled. A proposition of
    /// cost 0 is settled from the start (costs never drop below 0).
    void explore(std::span<const u32> targets)
    {
        const RelaxedTask& R = *m_R;
        const bool add = m_kind == Kind::Add, ff = m_kind == Kind::FF;
        const u32 P = R.num_props(), O = R.num_ops();
        std::memcpy(m_cost.data(), m_cost_init.data(), P * sizeof(u32));
        std::memcpy(m_cnt.data(), m_npos.data(), O * sizeof(u32));
        std::memset(m_acc.data(), 0, O * sizeof(u32));
        if (ff)
            std::fill(m_supp.begin(), m_supp.end(), k_none);
        m_q.clear();
        for (u32 p : m_state_pos)
        {
            m_cost[p] = 0;
            m_q.push(0, p);
        }
        for (u32 np : m_state_neg)
        {
            m_cost[np] = k_inf;
            for (u32 o : R.pre_of(np))
                ++m_cnt[o];
        }
        auto fire = [&](u32 o, u32 base, u32 sup)
        {
            const u32 val = sat_add(base, m_opcost[o]);
            for (u32 q : R.eff(o))
                if (val < m_cost[q])
                {
                    m_cost[q] = val;
                    if (ff)
                        m_supp[q] = sup;
                    m_q.push(val, q);
                }
        };
        for (u32 o : R.zero_ops())
            if (m_cnt[o] == 0)
                fire(o, 0, m_axiom[o] ? k_none : o);
        ++m_stamp;
        u32 left = 0;
        for (u32 g : targets)
            if (m_cost[g] != 0 && m_gmark[g] != m_stamp)
            {
                m_gmark[g] = m_stamp;
                ++left;
            }
        u32 c, p;
        while (left && m_q.pop(c, p))
        {
            if (c != m_cost[p])
                continue;
            if (m_gmark[p] == m_stamp)
            {
                m_gmark[p] = 0;
                --left;
            }
            for (u32 o : R.pre_of(p))
            {
                if (add && !m_axiom[o])
                    m_acc[o] = sat_add(m_acc[o], c);
                else if (c > m_acc[o])
                    m_acc[o] = c;
                if (ff && m_axiom[o])
                    m_last[o] = m_supp[p];
                if (--m_cnt[o] == 0)
                    fire(o, m_acc[o], m_axiom[o] ? (ff ? m_last[o] : k_none) : o);
            }
        }
    }

    /// h of one goal from the explored costs; h_FF extracts the relaxed plan and marks its ground actions.
    u64 fold(std::span<const u32> goal)
    {
        u64 h = 0;
        for (u32 g : goal)
        {
            const u32 c = m_cost[g];
            if (c == k_inf)
                return k_inf;
            h = m_kind == Kind::Add ? h + c : std::max<u64>(h, c);
        }
        if (m_kind != Kind::FF)
            return h;
        const RelaxedTask& R = *m_R;
        ++m_pstamp;
        m_plan.clear();
        m_plan_stamp = m_pstamp;
        u64 hc = 0;
        m_stack.assign(goal.begin(), goal.end());
        while (!m_stack.empty())
        {
            const u32 x = m_stack.back();
            m_stack.pop_back();
            if (m_pmark[x] == m_pstamp)
                continue;
            m_pmark[x] = m_pstamp;
            const u32 o = m_supp[x];
            if (o == k_none)
                continue;
            for (u32 q : R.pre(o))
                m_stack.push_back(q);
            const u32 ga = R.ground_action(o);
            if (m_ga_mark[ga] != m_pstamp)
            {
                m_ga_mark[ga] = m_pstamp;
                m_plan.push_back(ga);
                hc += m_real ? R.real_cost(ga) : 1;
            }
        }
        m_plan_valid = true;
        return hc;
    }

    Value single(std::span<const u32> goal)
    {
        explore(goal);
        const u64 h = fold(goal);
        if (h >= k_inf)
            m_plan_valid = false;
        return to_value(h);
    }

    /// The propositions of each atom goal; false if the grounding lacks a "false" proposition that one needs.
    bool goal_props(StateView s, std::span<const search::GoalSpec::AtomGoal> goals)
    {
        const AtomIndex& A = m_task.atoms();
        m_goal_props.clear();
        m_goal_begin.assign(1, 0);
        m_goal_dead.clear();
        for (const auto& g : goals)
        {
            bool dead = false;
            for (SlotId p : g.positive)
            {
                const u32 pp = m_R->fluent_props(A.canonical(AtomKind::Fluent, p.v)).first;
                if (pp == k_none)
                    dead = true;  // outside the relaxation: never reachable
                else
                    m_goal_props.push_back(pp);
            }
            for (SlotId p : g.negative)
            {
                if (!s.contains(p))
                    continue;  // false in s: satisfied at cost 0
                const u32 np = m_R->fluent_props(A.canonical(AtomKind::Fluent, p.v)).second;
                if (np == k_none)
                    return false;
                m_goal_props.push_back(np);
            }
            m_goal_begin.push_back(static_cast<u32>(m_goal_props.size()));
            m_goal_dead.push_back(dead);
        }
        m_union = m_goal_props;
        std::sort(m_union.begin(), m_union.end());
        m_union.erase(std::unique(m_union.begin(), m_union.end()), m_union.end());
        return true;
    }

    Value multi()
    {
        explore(m_union);
        u64 best = k_inf;
        u32 best_g = k_none;
        const auto goal = [&](u32 gi)
        { return std::span<const u32>(m_goal_props.data() + m_goal_begin[gi], m_goal_begin[gi + 1] - m_goal_begin[gi]); };
        for (u32 gi = 0; gi + 1 < m_goal_begin.size(); ++gi)
        {
            if (m_goal_dead[gi])
                continue;
            const u64 h = fold(goal(gi));
            if (h < best)
            {
                best = h;
                best_g = gi;
            }
        }
        m_plan_valid = false;
        if (m_kind == Kind::FF && best_g != k_none && best < k_inf)
            (void) fold(goal(best_g));  // leave the best goal's relaxed plan marked
        return to_value(best);
    }

    // ------------------------------------------------------------------------------------ lifted fallback
    LiftedRelaxation& lifted()
    {
        if (m_grounded_only)
            throw std::runtime_error("mymyr: the state lies outside the grounded relaxation (Evaluation::Grounded)");
        if (!m_lifted)
            m_lifted = std::make_unique<LiftedRelaxation>(m_task, m_kind, m_real);
        m_lifted->interrupt = &m_interrupt;
        return *m_lifted;
    }
    const std::vector<std::vector<GoalLit>>& task_goal()
    {
        if (m_task_goal.empty())
        {
            const TaskData& T = m_task.data();
            auto& lits = m_task_goal.emplace_back();
            for (const Literal& l : T.literals_of(T.goal))
            {
                const PredKind k = T.predicates[l.pred.v].kind;
                if (k == PredKind::Static || (k == PredKind::Derived && !l.positive))
                    continue;
                std::vector<u32> args;
                for (Term t : T.terms_of(l))
                    args.push_back(term_object(t).v);
                lits.push_back({l.pred.v, std::move(args), l.positive});
            }
        }
        return m_task_goal;
    }

    const Task& m_task;
    Kind m_kind;
    bool m_real;
    bool m_grounded_only;
    std::shared_ptr<const RelaxedTask> m_R;
    // grounded scratch
    std::vector<u32> m_cost_init, m_cost, m_supp, m_gmark, m_pmark, m_cnt, m_acc, m_last, m_npos, m_opcost;
    std::vector<u8> m_axiom;
    std::vector<u32> m_slot_pos, m_slot_neg, m_state_pos, m_state_neg;
    std::vector<u32> m_stack, m_plan, m_ga_mark;
    std::vector<u32> m_goal_props, m_goal_begin, m_union;
    std::vector<u8> m_goal_dead;
    u32 m_stamp = 0, m_pstamp = 0, m_plan_stamp = 0;
    bool m_plan_valid = false;
    RelaxedQueue m_q;
    // lifted fallback
    std::unique_ptr<LiftedRelaxation> m_lifted;
    std::vector<std::vector<GoalLit>> m_task_goal;
};
}  // namespace

std::unique_ptr<Heuristic> make_heuristic(const Task& task, const Options& options)
{
    switch (options.kind)
    {
        case Kind::Blind: return std::make_unique<BlindHeuristic>();
        case Kind::GoalCount: return std::make_unique<GoalCountHeuristic>(task);
        case Kind::Custom:
            throw std::invalid_argument("mymyr: Kind::Custom names a caller's Heuristic (BestFirstOptions::evaluator); make_heuristic "
                                        "builds the library's kinds only");
        default: return std::make_unique<RelaxationHeuristic>(task, options);
    }
}
}  // namespace mymyr::heuristics
