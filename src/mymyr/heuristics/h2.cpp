// h² (Haslum and Geffner 2000) over the grounded relaxation of relaxed_task.hpp (heuristic.hpp has the definition).
//
// The fixpoint is a generalized Dijkstra over pairs {p, q} of propositions (a single p is the pair {p, p}), i.e. h_max
// over the pair task Π² (Haslum 2009), with Π²'s actions left implicit. For an operator o with the distinct
// preconditions pre(o), adds add(o) and the propositions it makes false del(o):
//   - base(o): every pair within pre(o) is settled. It reaches the pairs within add(o) at cost(base) + c(o), and the
//     pairs add(o) x add(o') for the operators o' of the same ground action whose base came before (two effects of one
//     action: both conditions hold, an underestimate of the cost of their union, so the bound stays admissible);
//   - persist(o, r): base(o) and the pairs {x, r} for every x of pre(o) ({r, r} when pre(o) is empty) are settled, and
//     r is not made false by o or by an effect of o's action without a condition. It reaches the pairs {p, r} for p
//     in add(o).
// Because the pops come in increasing cost, an implicit action fires at the cost of the pop that completes it plus c(o).
// The base counts its unsettled pairs; persist(o, r) is checked when one of its pairs settles (and for every r when
// base(o) completes). A pair {p, not p} is never reached.
//
// Cost per evaluation: the pair table (P(P+1)/2 costs and settled bits) is reset, every settled pair visits the
// operators of its two propositions, and every operator scans the P persistence candidates once.

#include "h2.hpp"

#include "relaxation_detail.hpp"

#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::heuristics::detail
{
namespace
{
/// Index of the pair {x, y} with x <= y in the triangular table.
[[nodiscard]] inline u32 tri(u32 x, u32 y) noexcept { return static_cast<u32>(static_cast<u64>(y) * (y + 1) / 2 + x); }
[[nodiscard]] inline u32 pair_of(u32 a, u32 b) noexcept { return a <= b ? tri(a, b) : tri(b, a); }
/// The pair {x, y} (x <= y) of a triangular index.
inline void untri(u32 idx, u32& x, u32& y) noexcept
{
    u64 t = static_cast<u64>((std::sqrt(8.0 * idx + 1.0) - 1.0) / 2.0);
    while (t * (t + 1) / 2 > idx)
        --t;
    while ((t + 1) * (t + 2) / 2 <= idx)
        ++t;
    y = static_cast<u32>(t);
    x = static_cast<u32>(idx - t * (t + 1) / 2);
}

/// CSR lists of u32.
struct Lists
{
    std::vector<u32> begin{0}, items;
    [[nodiscard]] std::span<const u32> operator[](u32 i) const noexcept
    {
        return {items.data() + begin[i], begin[i + 1] - begin[i]};
    }
    void close() { begin.push_back(static_cast<u32>(items.size())); }
};

class H2Heuristic final : public Heuristic
{
public:
    H2Heuristic(const Task& task, const Options& o) : m_task(task), m_real(o.costs == Costs::Real)
    {
        if (o.evaluation == Evaluation::Lifted)
            throw std::invalid_argument("mymyr: h2 has no lifted evaluation (use Evaluation::Auto or Grounded)");
        if (m_real)
        {
            const ActionCosts costs(task);
            if (!costs.state_independent())
                throw std::invalid_argument("mymyr: real-cost heuristics need action costs that do not depend on the state");
            if (!costs.integral())
                throw std::invalid_argument("mymyr: real-cost heuristics need integral action costs");
        }
        const auto t0 = std::chrono::steady_clock::now();
        m_R = o.relaxed;
        if (!m_R)
        {
            GroundingStats st;
            m_R = RelaxedTask::build(task, o.budget, &st);
            if (!m_R)
                throw std::runtime_error("mymyr: grounding the relaxation exceeded the budget, and h2 has no lifted evaluation: " +
                                         st.reason);
        }
        m_stats.grounding_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (m_real && !m_R->real_costs_available())
            throw std::invalid_argument("mymyr: real-cost heuristics need non-negative integer action costs below 2^31");
        if (m_R->num_props() > k_h2_max_props)
            throw std::invalid_argument("mymyr: h2 supports at most " + std::to_string(k_h2_max_props) +
                                        " propositions (its table holds every pair); the grounding has " +
                                        std::to_string(m_R->num_props()));
        init();
    }

    [[nodiscard]] Kind kind() const noexcept override { return Kind::H2; }
    [[nodiscard]] std::shared_ptr<const RelaxedTask> relaxed() const override { return m_R; }

    Value evaluate(StateView s) override
    {
        ++m_stats.evaluations;
        convert(s);
        Value v = k_dead_end;
        if (!m_R->goal_unreachable())
        {
            m_goal_props.assign(m_R->goal().begin(), m_R->goal().end());
            m_goal_begin.assign({0, static_cast<u32>(m_goal_props.size())});
            m_goal_dead.assign(1, 0);
            v = run();
        }
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    Value evaluate(StateView s, std::span<const search::GoalSpec::AtomGoal> goals) override
    {
        ++m_stats.evaluations;
        if (goals.empty())
            return 0;
        convert(s);
        const AtomIndex& A = m_task.atoms();
        m_goal_props.clear();
        m_goal_begin.assign(1, 0);
        m_goal_dead.clear();
        for (const auto& g : goals)
        {
            const usize first = m_goal_props.size();
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
                const u32 np = m_R->fluent_props(A.canonical(AtomKind::Fluent, p.v)).second;
                if (np != k_none)
                    m_goal_props.push_back(np);
                else if (s.contains(p))
                    throw std::runtime_error("mymyr: a negative goal literal lies outside the grounded relaxation (no operator "
                                             "or goal of the task uses its atom negatively), and h2 has no lifted evaluation");
                // else: it holds and no operator can make its atom true (pairs with it count as reached): a lower bound
            }
            std::sort(m_goal_props.begin() + static_cast<std::ptrdiff_t>(first), m_goal_props.end());
            m_goal_props.erase(std::unique(m_goal_props.begin() + static_cast<std::ptrdiff_t>(first), m_goal_props.end()),
                               m_goal_props.end());
            m_goal_begin.push_back(static_cast<u32>(m_goal_props.size()));
            m_goal_dead.push_back(dead ? 1 : 0);
        }
        const Value v = run();
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

private:
    void convert(StateView s)
    {
        if (!m_props.convert(m_task, *m_R, s))
            throw std::runtime_error("mymyr: the state lies outside the grounded relaxation, and h2 has no lifted evaluation");
        ++m_stats.grounded;
    }

    void init()
    {
        const RelaxedTask& R = *m_R;
        const u32 P = R.num_props(), O = R.num_ops();
        m_P = P;
        // complement of a proposition: "p" <-> "not p" (k_none when the other one has no proposition)
        m_comp.assign(P, k_none);
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p))
            {
                const u32 t = R.fluent_props(R.prop_atom(p)).first;
                m_comp[p] = t;
                if (t != k_none)
                    m_comp[t] = p;
            }
        // per operator: distinct preconditions and adds, the propositions persistence excludes, the base counter
        std::vector<u32> tmp;
        m_base_init.resize(O);
        m_opcost.resize(O);
        for (u32 o = 0; o < O; ++o)
        {
            tmp.assign(R.pre(o).begin(), R.pre(o).end());
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            m_pre.items.insert(m_pre.items.end(), tmp.begin(), tmp.end());
            m_pre.close();
            const u64 k = tmp.size();
            m_base_init[o] = static_cast<u32>(k * (k + 1) / 2);
            if (k == 0)
                m_empty_ops.push_back(o);
            tmp.assign(R.eff(o).begin(), R.eff(o).end());
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            m_add.items.insert(m_add.items.end(), tmp.begin(), tmp.end());
            m_add.close();
            m_opcost[o] = R.is_axiom(o) ? 0 : (m_real ? R.real_cost(R.ground_action(o)) : 1);
        }
        // operators per ground action
        const u32 G = R.num_ground_actions();
        std::vector<u32> count(G + 1, 0);
        for (u32 o = 0; o < O; ++o)
            if (!R.is_axiom(o))
                ++count[R.ground_action(o) + 1];
        for (u32 g = 0; g < G; ++g)
            count[g + 1] += count[g];
        m_ga_ops.begin.assign(count.begin(), count.end());
        m_ga_ops.items.resize(count[G]);
        for (u32 o = 0; o < O; ++o)
            if (!R.is_axiom(o))
                m_ga_ops.items[count[R.ground_action(o)]++] = o;
        // what an operator makes false: its deleted atoms and the negations of its added atoms
        auto falsified = [&](u32 o, std::vector<u32>& out)
        {
            for (u32 p : R.del(o))
                out.push_back(p);
            for (u32 p : m_add[o])
                if (m_comp[p] != k_none)
                    out.push_back(m_comp[p]);
        };
        for (u32 o = 0; o < O; ++o)
        {
            tmp.assign(m_add[o].begin(), m_add[o].end());
            falsified(o, tmp);
            if (!R.is_axiom(o))
                for (u32 o2 : m_ga_ops[R.ground_action(o)])
                    if (o2 != o && R.unconditional(o2))
                        falsified(o2, tmp);
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            m_excl.items.insert(m_excl.items.end(), tmp.begin(), tmp.end());
            m_excl.close();
        }
        // operators per precondition (distinct)
        count.assign(P + 1, 0);
        for (u32 p : m_pre.items)
            ++count[p + 1];
        for (u32 p = 0; p < P; ++p)
            count[p + 1] += count[p];
        m_pre_of.begin.assign(count.begin(), count.end());
        m_pre_of.items.resize(count[P]);
        for (u32 o = 0; o < O; ++o)
            for (u32 p : m_pre[o])
                m_pre_of.items[count[p]++] = o;

        const u64 pairs = static_cast<u64>(P) * (P + 1) / 2;
        m_cost.resize(pairs);
        m_settled.resize((pairs + 63) / 64);
        m_target.assign((pairs + 63) / 64, 0);
        m_base.resize(O);
        m_fired.assign(O, 0);
        m_negmark.assign(P, 0);
    }

    [[nodiscard]] bool settled(u32 i) const noexcept { return (m_settled[i >> 6] >> (i & 63)) & 1; }

    void update(u32 i, u32 v)
    {
        if (v < m_cost[i])
        {
            m_cost[i] = v;
            m_q.push(v, i);
        }
    }

    [[nodiscard]] bool excluded(u32 o, u32 r) const noexcept
    {
        const auto ex = m_excl[o];
        return std::binary_search(ex.begin(), ex.end(), r);
    }

    /// Whether the pairs {x, r} for the preconditions x of o ({r, r} for an operator without preconditions) are settled.
    [[nodiscard]] bool persist_ready(u32 o, u32 r) const noexcept
    {
        const auto pre = m_pre[o];
        if (pre.empty())
            return settled(tri(r, r));
        for (u32 x : pre)
            if (!settled(pair_of(x, r)))
                return false;
        return true;
    }

    void fire_persist(u32 o, u32 r, u32 c)
    {
        if (excluded(o, r))
            return;
        const u32 v = sat_add(c, m_opcost[o]);
        for (u32 p : m_add[o])
            if (m_comp[p] != r)
                update(pair_of(p, r), v);
    }

    void fire_base(u32 o, u32 c)
    {
        const u32 v = sat_add(c, m_opcost[o]);
        const auto add = m_add[o];
        for (usize i = 0; i < add.size(); ++i)
        {
            update(tri(add[i], add[i]), v);
            for (usize j = i + 1; j < add.size(); ++j)
                if (m_comp[add[i]] != add[j])
                    update(pair_of(add[i], add[j]), v);
        }
        if (!m_R->is_axiom(o))
        {
            for (u32 o2 : m_ga_ops[m_R->ground_action(o)])
                if (o2 != o && m_fired[o2] == m_stamp)
                    for (u32 p : add)
                        for (u32 q : m_add[o2])
                            if (p != q && m_comp[p] != q)
                                update(pair_of(p, q), v);
        }
        m_fired[o] = m_stamp;
        for (u32 r = 0; r < m_P; ++r)
            if (persist_ready(o, r))
                fire_persist(o, r, c);
    }

    /// The operators waiting for the settled pair {x, y} (x <= y) at cost c.
    void process(u32 x, u32 y, u32 c)
    {
        for (u32 o : m_pre_of[x])
        {
            const auto pre = m_pre[o];
            if ((x == y || std::binary_search(pre.begin(), pre.end(), y)) && --m_base[o] == 0)
            {
                fire_base(o, c);  // also every persist(o, r) that is ready
                continue;
            }
            if (m_base[o] == 0 && persist_ready(o, y))
                fire_persist(o, y, c);
        }
        if (x != y)
        {
            for (u32 o : m_pre_of[y])
                if (m_base[o] == 0 && persist_ready(o, x))
                    fire_persist(o, x, c);
        }
        else
            for (u32 o : m_empty_ops)
                fire_persist(o, x, c);
    }

    Value run()
    {
        const RelaxedTask& R = *m_R;
        const u32 P = m_P;
        std::fill(m_cost.begin(), m_cost.end(), k_inf);
        std::fill(m_settled.begin(), m_settled.end(), u64{0});
        std::copy(m_base_init.begin(), m_base_init.end(), m_base.begin());
        ++m_stamp;
        m_q.clear();
        // the pairs of the state's literals cost 0
        m_true.assign(m_props.pos.begin(), m_props.pos.end());
        for (u32 np : m_props.neg)
            m_negmark[np] = m_stamp;
        for (u32 p = 0; p < P; ++p)
            if (R.negative(p) && m_negmark[p] != m_stamp)
                m_true.push_back(p);
        for (usize i = 0; i < m_true.size(); ++i)
            for (usize j = i; j < m_true.size(); ++j)
                update(pair_of(m_true[i], m_true[j]), 0);
        // the pairs of the goals
        m_targets.clear();
        for (u32 gi = 0; gi + 1 < m_goal_begin.size(); ++gi)
        {
            if (m_goal_dead[gi])
                continue;
            for (u32 i = m_goal_begin[gi]; i < m_goal_begin[gi + 1]; ++i)
                for (u32 j = i; j < m_goal_begin[gi + 1]; ++j)
                {
                    const u32 t = pair_of(m_goal_props[i], m_goal_props[j]);
                    if (!((m_target[t >> 6] >> (t & 63)) & 1))
                    {
                        m_target[t >> 6] |= u64{1} << (t & 63);
                        m_targets.push_back(t);
                    }
                }
        }
        u64 left = m_targets.size();
        for (u32 o : m_empty_ops)
            if (left)
                fire_base(o, 0);
        u32 c, i;
        u64 pops = 0;
        while (left && m_q.pop(c, i))
        {
            if (c != m_cost[i] || settled(i))
                continue;
            m_settled[i >> 6] |= u64{1} << (i & 63);
            if ((m_target[i >> 6] >> (i & 63)) & 1)
                if (--left == 0)
                    break;
            if ((++pops & 4095) == 0 && m_interrupt && m_interrupt())
            {
                clear_targets();
                throw Interrupted();
            }
            u32 x, y;
            untri(i, x, y);
            process(x, y, c);
        }
        clear_targets();
        u64 best = k_inf;
        for (u32 gi = 0; gi + 1 < m_goal_begin.size(); ++gi)
        {
            if (m_goal_dead[gi])
                continue;
            u64 h = 0;
            for (u32 a = m_goal_begin[gi]; a < m_goal_begin[gi + 1] && h < k_inf; ++a)
                for (u32 b = a; b < m_goal_begin[gi + 1]; ++b)
                    h = std::max<u64>(h, m_cost[pair_of(m_goal_props[a], m_goal_props[b])]);
            best = std::min(best, h);
        }
        return to_value(best);
    }

    void clear_targets()
    {
        for (u32 t : m_targets)
            m_target[t >> 6] &= ~(u64{1} << (t & 63));
    }

    const Task& m_task;
    bool m_real;
    std::shared_ptr<const RelaxedTask> m_R;
    u32 m_P = 0;
    // the operators
    Lists m_pre, m_add, m_excl, m_pre_of, m_ga_ops;
    std::vector<u32> m_comp, m_base_init, m_opcost, m_empty_ops;
    // per evaluation
    StateProps m_props;
    std::vector<u32> m_cost, m_base, m_fired, m_negmark, m_true, m_targets;
    std::vector<u64> m_settled, m_target;
    std::vector<u32> m_goal_props, m_goal_begin;
    std::vector<u8> m_goal_dead;
    u32 m_stamp = 0;
    RelaxedQueue m_q;
};
}  // namespace

std::unique_ptr<Heuristic> make_h2(const Task& task, const Options& options) { return std::make_unique<H2Heuristic>(task, options); }
}  // namespace mymyr::heuristics::detail
