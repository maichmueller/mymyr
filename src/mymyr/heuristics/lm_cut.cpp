// LM-cut (Helmert and Domshlak 2009) over the grounded relaxation (relaxed_task.hpp), following Fast Downward's
// implementation (lm_cut_landmarks.cc):
//   1. h_max from the state with each operator's h_max supporter: the last of its most expensive preconditions in the
//      grounding's order (the action's precondition, then the effect's condition, each in the schema's order). Operators without preconditions get an artificial one that holds in every state; the goal
//      is an artificial zero-cost operator whose preconditions are the goal's propositions.
//   2. While h_max of the goal is positive:
//      a. the goal zone: the goal and, recursively, the supporters of the zero-cost achievers of goal-zone
//         propositions. Before recursing, a zero-cost achiever's supporter moves to an equally expensive precondition
//         that has no zero-cost achiever, else to one that is in the goal zone already (Fast Downward's border and
//         goal-zone detection, both on by default there);
//      b. the cut: from the state's propositions along supporter edges without entering the goal zone; an operator
//         reached this way that adds a goal-zone proposition joins the cut (and its other effects are not followed);
//      c. h grows by the cheapest cost in the cut, which is subtracted from the remaining cost of every ground action
//         with an operator in the cut (once per action);
//      d. h_max is updated incrementally: the operators of the cheaper actions propagate their new costs, and an
//         operator whose supporter became cheaper takes the last of its most expensive preconditions again.
// Operators: one per ground action and distinct precondition (the action's precondition with the condition of an
// effect instance; instances without a condition of their own form one operator, as in a STRIPS task; an instance
// whose static condition fails never fires and has none), and one per ground axiom. Costs are shared per ground action: the operators of one action all cost its remaining cost, and a cut
// lowers it once. Each cut, mapped to its actions, is an action landmark of the task (every plan's actions with the
// effects that fire form a relaxed plan, which must contain an operator of the cut), and the cut costs never exceed
// what the actions have left, so h is a cost partitioning of action landmarks: admissible with conditional effects.
// Lowering an action lowers all of its operators, also those outside the cut, so with conditional effects the sum of
// the cuts can fall below h_max; h is the larger of the two (both are admissible). Without conditional effects every
// action is one operator and h is the sum of the cuts, which is at least h_max. Axioms are zero-cost operators: a
// zero-cost operator's supporter lies in the goal zone whenever one of its effects does, so no cut holds one, and the
// derived atoms of a state are reachable in the relaxation from its fluent atoms. "False" propositions (negative
// preconditions) hold where their atom is false and are otherwise added by the operators deleting it, which
// over-approximates every plan as well. Numeric conditions are ignored.
// Grounded only: a state outside the grounding raises std::runtime_error.

#include "landmark_heuristics.hpp"
#include "relaxation_detail.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::heuristics::detail
{
namespace
{
class LmCutHeuristic final : public Heuristic
{
public:
    LmCutHeuristic(const Task& task, const Options& o)
        : m_task(task), m_real(resolve_costs(task, o.costs) == Costs::Real), m_admissible(objective_costs(task, o.costs))
    {
        if (o.evaluation == Evaluation::Lifted)
            throw std::invalid_argument("mymyr: LM-cut has no lifted evaluation (use Evaluation::Auto or Grounded)");
        if (m_real)
            m_scale = ActionCosts(task).relaxed_scale();
        const auto t0 = std::chrono::steady_clock::now();
        m_R = o.relaxed;
        if (!m_R)
        {
            GroundingStats st;
            m_R = RelaxedTask::build(task, o.budget, &st);
            if (!m_R)
                throw std::runtime_error("mymyr: grounding the relaxation exceeded the budget, and LM-cut has no lifted evaluation: " +
                                         st.reason);
        }
        m_stats.grounding_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (m_real && !m_R->real_costs_available())
            throw std::invalid_argument("mymyr: real-cost heuristics need defined, non-negative action costs below 2^31 units of "
                                        "the cost scale (heuristics::ActionCosts::relaxed_cost)");
        build();
    }

    [[nodiscard]] Kind kind() const noexcept override { return Kind::LmCut; }
    [[nodiscard]] bool admissible() const noexcept override { return m_admissible; }
    [[nodiscard]] std::shared_ptr<const RelaxedTask> relaxed() const override { return m_R; }

    Value evaluate(StateView s) override
    {
        ++m_stats.evaluations;
        convert(s);
        u64 h = k_inf;
        if (!m_R->goal_unreachable())
        {
            set_goal(m_R->goal());
            h = run();
        }
        const Value v = to_value(h, m_scale);
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    /// The minimum of LM-cut over the goals.
    Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> goals) override
    {
        ++m_stats.evaluations;
        if (goals.empty())
            return 0;
        convert(s);
        const AtomIndex& A = m_task.atoms();
        u64 best = k_inf;
        std::vector<u32> props;
        for (const auto& g : goals)
        {
            props.clear();
            bool dead = false;
            for (SlotId p : g.positive)
            {
                const u32 pp = m_R->fluent_props(A.canonical(AtomKind::Fluent, p.v)).first;
                if (pp == k_none)
                    dead = true;  // outside the relaxation: never reachable
                else
                    props.push_back(pp);
            }
            for (SlotId p : g.negative)
            {
                if (!s.contains(p))
                    continue;  // false in s: satisfied
                const u32 np = m_R->fluent_props(A.canonical(AtomKind::Fluent, p.v)).second;
                if (np == k_none)
                    throw std::runtime_error("mymyr: a negative goal literal lies outside the grounded relaxation (no operator "
                                             "or goal of the task uses its atom negatively), and LM-cut has no lifted evaluation");
                props.push_back(np);
            }
            if (dead)
                continue;
            std::sort(props.begin(), props.end());
            props.erase(std::unique(props.begin(), props.end()), props.end());
            set_goal(props);
            best = std::min(best, run());
        }
        const Value v = to_value(best, m_scale);
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

private:
    enum : u8
    {
        Reached = 0,
        GoalZone = 1,
        BeforeGoalZone = 2,
    };

    // ------------------------------------------------------------------------------------ the operators
    void build()
    {
        const RelaxedTask& R = *m_R;
        m_P = R.num_props();
        m_init = m_P;      // the artificial precondition of operators without one
        m_G = m_P + 1;     // the artificial goal proposition
        const u32 NP = m_P + 2;
        // One operator per ground action and precondition list: the effect instances of an action whose conditions
        // add nothing to its precondition are one STRIPS operator.
        std::vector<std::vector<u32>> pres, effs;
        std::map<std::pair<u32, std::vector<u32>>, u32> merged;
        std::vector<u32> pre, eff;
        for (u32 o = 0; o < R.num_ops(); ++o)
        {
            if (!R.static_condition_holds(o))
                continue;  // never fires
            pre.assign(R.pre(o).begin(), R.pre(o).end());
            // first occurrence order: the supporter is the last of the most expensive preconditions in this order
            u32 k = 0;
            for (u32 i = 0; i < pre.size(); ++i)
                if (std::find(pre.begin(), pre.begin() + k, pre[i]) == pre.begin() + k)
                    pre[k++] = pre[i];
            pre.resize(k);
            if (pre.empty())
                pre.push_back(m_init);
            const u32 ga = R.is_axiom(o) ? k_none : R.ground_action(o);
            u32 at = static_cast<u32>(pres.size());
            if (ga != k_none)
                at = merged.try_emplace({ga, pre}, at).first->second;
            if (at == pres.size())
            {
                pres.push_back(pre);
                effs.emplace_back();
                m_op_ga.push_back(ga);
            }
            effs[at].insert(effs[at].end(), R.eff(o).begin(), R.eff(o).end());
        }
        const u32 O = static_cast<u32>(pres.size());
        m_O = O;
        m_GO = O;  // the goal operator
        m_op_ga.push_back(k_none);
        m_pre_begin.assign(1, 0);
        m_eff_begin.assign(1, 0);
        for (u32 o = 0; o < O; ++o)
        {
            eff = std::move(effs[o]);
            std::sort(eff.begin(), eff.end());
            eff.erase(std::unique(eff.begin(), eff.end()), eff.end());
            m_pre.insert(m_pre.end(), pres[o].begin(), pres[o].end());
            m_pre_begin.push_back(static_cast<u32>(m_pre.size()));
            m_eff.insert(m_eff.end(), eff.begin(), eff.end());
            m_eff_begin.push_back(static_cast<u32>(m_eff.size()));
        }
        // proposition -> operators having it as a precondition / an effect
        csr(NP, m_pre_begin, m_pre, m_pre_of_begin, m_pre_of);
        csr(NP, m_eff_begin, m_eff, m_ach_begin, m_ach);
        // ground action -> its operators
        const u32 G = R.num_ground_actions();
        m_ga_cost.resize(G);
        for (u32 ga = 0; ga < G; ++ga)
            m_ga_cost[ga] = m_real ? R.real_cost(ga) : 1;
        m_ga_ops_begin.assign(G + 1, 0);
        for (u32 o = 0; o < O; ++o)
            if (m_op_ga[o] != k_none)
                ++m_ga_ops_begin[m_op_ga[o] + 1];
        for (u32 ga = 0; ga < G; ++ga)
            m_ga_ops_begin[ga + 1] += m_ga_ops_begin[ga];
        m_ga_ops.resize(m_ga_ops_begin[G]);
        {
            std::vector<u32> fill(m_ga_ops_begin.begin(), m_ga_ops_begin.end() - 1);
            for (u32 o = 0; o < O; ++o)
                if (m_op_ga[o] != k_none)
                    m_ga_ops[fill[m_op_ga[o]]++] = o;
        }
        m_rem = m_ga_cost;
        m_ga_stamp.assign(G, 0);
        for (u32 p = 0; p < m_P; ++p)
            if (R.negative(p))
                m_false_props.push_back(p);
        m_cost.assign(NP, k_inf);
        m_zone.assign(NP, Reached);
        m_negmark.assign(NP, 0);
        m_in_goal.assign(NP, 0);
        m_unsat_init.resize(O + 1);
        for (u32 o = 0; o < O; ++o)
            m_unsat_init[o] = m_pre_begin[o + 1] - m_pre_begin[o];
        m_unsat.resize(O + 1);
        m_supp.assign(O + 1, k_none);
        m_supp_cost.assign(O + 1, k_inf);
    }

    /// Inverts the operator lists `begin`/`items` (operator -> propositions) into proposition -> operators.
    static void csr(u32 NP, const std::vector<u32>& begin, const std::vector<u32>& items, std::vector<u32>& out_begin,
                    std::vector<u32>& out)
    {
        out_begin.assign(NP + 1, 0);
        for (u32 x : items)
            ++out_begin[x + 1];
        for (u32 p = 0; p < NP; ++p)
            out_begin[p + 1] += out_begin[p];
        out.resize(items.size());
        std::vector<u32> fill(out_begin.begin(), out_begin.end() - 1);
        for (u32 o = 0; o + 1 < begin.size(); ++o)
            for (u32 i = begin[o]; i < begin[o + 1]; ++i)
                out[fill[items[i]]++] = o;
    }

    [[nodiscard]] std::span<const u32> pre(u32 o) const noexcept
    {
        if (o == m_GO)
            return m_goal;
        return {m_pre.data() + m_pre_begin[o], m_pre_begin[o + 1] - m_pre_begin[o]};
    }
    [[nodiscard]] std::span<const u32> eff(u32 o) const noexcept
    {
        return {m_eff.data() + m_eff_begin[o], m_eff_begin[o + 1] - m_eff_begin[o]};
    }
    [[nodiscard]] std::span<const u32> pre_of(u32 p) const noexcept
    {
        return {m_pre_of.data() + m_pre_of_begin[p], m_pre_of_begin[p + 1] - m_pre_of_begin[p]};
    }
    /// Achievers of p (the goal proposition: the goal operator only).
    [[nodiscard]] std::span<const u32> achievers(u32 p) const noexcept
    {
        if (p == m_G)
            return {&m_GO, 1};
        return {m_ach.data() + m_ach_begin[p], m_ach_begin[p + 1] - m_ach_begin[p]};
    }
    [[nodiscard]] u32 op_cost(u32 o) const noexcept { return m_op_ga[o] == k_none ? 0 : m_rem[m_op_ga[o]]; }

    // ------------------------------------------------------------------------------------ evaluation
    void convert(StateView s)
    {
        if (!m_props.convert(m_task, *m_R, s))
            throw std::runtime_error("mymyr: the state lies outside the grounded relaxation, and LM-cut has no lifted evaluation");
        ++m_stats.grounded;
    }

    void set_goal(std::span<const u32> goal)
    {
        for (u32 g : m_goal)
            m_in_goal[g] = 0;
        m_goal.assign(goal.begin(), goal.end());
        for (u32 g : m_goal)
            m_in_goal[g] = 1;
    }

    /// The propositions of the converted state that hold at cost 0 (with the artificial precondition), in order.
    template<class F>
    void for_each_state_prop(F&& f)
    {
        f(m_init);
        for (u32 p : m_props.pos)
            f(p);
        if (++m_negstamp == 0)
        {
            std::fill(m_negmark.begin(), m_negmark.end(), 0);
            m_negstamp = 1;
        }
        for (u32 np : m_props.neg)
            m_negmark[np] = m_negstamp;
        for (u32 np : m_false_props)
            if (m_negmark[np] != m_negstamp)
                f(np);
    }

    void enqueue(u32 p, u32 c)
    {
        if (c < m_cost[p])
        {
            m_cost[p] = c;
            m_q.push(c, p);
        }
    }

    /// The last of the most expensive preconditions of an enabled operator.
    void choose_supporter(u32 o)
    {
        u32 best = k_none, bc = 0;
        for (u32 q : pre(o))
            if (best == k_none || m_cost[q] >= bc)
            {
                best = q;
                bc = m_cost[q];
            }
        m_supp[o] = best;
        m_supp_cost[o] = bc;
    }

    void poll()
    {
        if ((++m_pops & 4095) == 0 && m_interrupt && m_interrupt())
        {
            m_q.clear();
            throw Interrupted();
        }
    }

    /// Step 1: h_max with supporters from the converted state.
    void first_exploration()
    {
        const u32 NP = m_P + 2;
        std::fill_n(m_cost.data(), NP, k_inf);
        std::copy_n(m_unsat_init.data(), m_O, m_unsat.data());
        m_unsat[m_GO] = static_cast<u32>(m_goal.size());
        std::fill(m_supp.begin(), m_supp.end(), k_none);
        m_q.clear();
        for_each_state_prop([&](u32 p) { enqueue(p, 0); });
        if (m_goal.empty())
        {
            m_supp[m_GO] = m_init;
            m_supp_cost[m_GO] = 0;
            m_cost[m_G] = 0;
        }
        u32 c, p;
        while (m_q.pop(c, p))
        {
            if (c != m_cost[p])
                continue;
            poll();
            for (u32 o : pre_of(p))
                if (--m_unsat[o] == 0)
                {
                    choose_supporter(o);
                    const u32 t = sat_add(m_supp_cost[o], op_cost(o));
                    for (u32 q : eff(o))
                        enqueue(q, t);
                }
            if (m_in_goal[p] && --m_unsat[m_GO] == 0)
            {
                choose_supporter(m_GO);
                m_cost[m_G] = m_supp_cost[m_GO];
            }
        }
    }

    [[nodiscard]] bool no_zero_cost_achiever(u32 p) const
    {
        for (u32 o : achievers(p))
            if (op_cost(o) == 0)
                return false;
        return true;
    }

    /// Fast Downward's border and goal-zone detection for a zero-cost achiever before its supporter joins the goal zone.
    void break_supporter_ties(u32 o)
    {
        for (u32 q : pre(o))
        {
            if (m_cost[q] != m_supp_cost[o])
                continue;
            if (no_zero_cost_achiever(q) || m_zone[q] == GoalZone)
            {
                m_supp[o] = q;
                return;
            }
        }
    }

    /// Step 2a, depth first in the order of the recursive definition.
    void mark_goal_zone()
    {
        m_frames.clear();
        auto enter = [&](u32 p)
        {
            if (p == k_none || m_zone[p] == GoalZone)
                return;
            m_zone[p] = GoalZone;
            m_touched.push_back(p);
            m_frames.push_back({p, 0});
        };
        enter(m_G);
        while (!m_frames.empty())
        {
            GoalFrame& f = m_frames.back();
            const auto ach = achievers(f.p);
            if (f.next == ach.size())
            {
                m_frames.pop_back();
                continue;
            }
            const u32 o = ach[f.next++];
            if (op_cost(o) != 0 || m_supp[o] == k_none)
                continue;  // not free, or never enabled
            break_supporter_ties(o);
            enter(m_supp[o]);  // may reallocate m_frames: f is not used after this
        }
    }

    /// Step 2b: the cut into m_cut.
    void second_exploration()
    {
        m_cut.clear();
        m_stack.clear();
        for_each_state_prop(
            [&](u32 p)
            {
                m_zone[p] = BeforeGoalZone;
                m_touched.push_back(p);
                m_stack.push_back(p);
            });
        while (!m_stack.empty())
        {
            const u32 p = m_stack.back();
            m_stack.pop_back();
            for (u32 o : pre_of(p))
            {
                if (m_supp[o] != p)
                    continue;
                bool cut = false;
                for (u32 q : eff(o))
                    if (m_zone[q] == GoalZone)
                    {
                        cut = true;
                        break;
                    }
                if (cut)
                {
                    m_cut.push_back(o);
                    continue;
                }
                for (u32 q : eff(o))
                    if (m_zone[q] != BeforeGoalZone)
                    {
                        m_zone[q] = BeforeGoalZone;
                        m_touched.push_back(q);
                        m_stack.push_back(q);
                    }
            }
        }
    }

    /// Step 2d: propagates the lowered costs of the actions in m_lowered.
    void incremental_exploration()
    {
        m_q.clear();
        for (u32 ga : m_lowered)
            for (u32 i = m_ga_ops_begin[ga]; i < m_ga_ops_begin[ga + 1]; ++i)
            {
                const u32 o = m_ga_ops[i];
                if (m_supp[o] == k_none)
                    continue;
                const u32 t = sat_add(m_supp_cost[o], op_cost(o));
                for (u32 q : eff(o))
                    enqueue(q, t);
            }
        u32 c, p;
        while (m_q.pop(c, p))
        {
            if (c != m_cost[p])
                continue;
            poll();
            for (u32 o : pre_of(p))
            {
                if (m_supp[o] != p || m_supp_cost[o] <= c)
                    continue;
                const u32 old = m_supp_cost[o];
                choose_supporter(o);
                if (m_supp_cost[o] != old)
                {
                    const u32 t = sat_add(m_supp_cost[o], op_cost(o));
                    for (u32 q : eff(o))
                        enqueue(q, t);
                }
            }
            if (m_in_goal[p] && m_supp[m_GO] == p && m_supp_cost[m_GO] > c)
            {
                choose_supporter(m_GO);
                m_cost[m_G] = m_supp_cost[m_GO];
            }
        }
    }

    /// LM-cut of the converted state for m_goal, in cost units (k_inf: a dead end).
    u64 run()
    {
        // the remaining costs of the actions the previous evaluation lowered
        for (u32 ga : m_all_lowered)
            m_rem[ga] = m_ga_cost[ga];
        m_all_lowered.clear();
        first_exploration();
        if (m_cost[m_G] == k_inf)
            return k_inf;
        const u64 hmax = m_cost[m_G];
        u64 h = 0;
        while (m_cost[m_G] != 0)
        {
            mark_goal_zone();
            second_exploration();
            u32 cut_cost = k_inf;
            for (u32 o : m_cut)
                cut_cost = std::min(cut_cost, op_cost(o));
            if (m_cut.empty() || cut_cost == 0 || cut_cost == k_inf)
                throw std::logic_error("mymyr: LM-cut found an empty or free cut");
            h += cut_cost;
            if (++m_round == 0)
            {
                std::fill(m_ga_stamp.begin(), m_ga_stamp.end(), 0);
                m_round = 1;
            }
            m_lowered.clear();
            for (u32 o : m_cut)
            {
                const u32 ga = m_op_ga[o];
                if (m_ga_stamp[ga] == m_round)
                    continue;
                m_ga_stamp[ga] = m_round;
                if (m_rem[ga] == m_ga_cost[ga])
                    m_all_lowered.push_back(ga);
                m_rem[ga] -= cut_cost;
                m_lowered.push_back(ga);
            }
            for (u32 p : m_touched)
                m_zone[p] = Reached;
            m_touched.clear();
            incremental_exploration();
        }
        return std::min<u64>(std::max(h, hmax), k_inf - 1);
    }

    const Task& m_task;
    bool m_real;
    bool m_admissible;
    f64 m_scale = 1;  // real costs: the unit of the integer costs is 1 / m_scale
    std::shared_ptr<const RelaxedTask> m_R;
    // the operators: those of the grounding, then the goal operator (index m_GO = m_O)
    u32 m_P = 0, m_O = 0, m_init = 0, m_G = 0, m_GO = 0;
    std::vector<u32> m_pre_begin, m_pre, m_eff_begin, m_eff, m_pre_of_begin, m_pre_of, m_ach_begin, m_ach;
    std::vector<u32> m_op_ga, m_ga_cost, m_ga_ops_begin, m_ga_ops, m_false_props, m_unsat_init;
    // per evaluation
    StateProps m_props;
    std::vector<u32> m_goal, m_cost, m_unsat, m_supp, m_supp_cost, m_rem, m_ga_stamp, m_negmark;
    std::vector<u8> m_zone, m_in_goal;
    std::vector<u32> m_cut, m_stack, m_touched, m_lowered, m_all_lowered;
    struct GoalFrame  // the goal-zone recursion: a proposition and its next achiever
    {
        u32 p, next;
    };
    std::vector<GoalFrame> m_frames;
    u32 m_negstamp = 0, m_round = 0;
    u64 m_pops = 0;
    RelaxedQueue m_q;
};
}  // namespace

std::unique_ptr<Heuristic> make_lm_cut(const Task& task, const Options& options)
{
    return std::make_unique<LmCutHeuristic>(task, options);
}
}  // namespace mymyr::heuristics::detail
