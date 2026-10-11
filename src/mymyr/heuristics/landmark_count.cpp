// Landmark count (Richter and Westphal 2010, LAMA's lmcount) with Fast Downward's landmark progression (Büchner,
// Keller, Eriksson and Helmert 2023).
//
// Landmarks: approximate_fact_landmarks over the heuristic's grounding (landmarks/approximate.hpp) with sound first
// achievers (FirstAchievers::PossiblyBefore), derived atoms (include_derived_atoms), greedy-necessary orderings and
// disjunctive landmarks of at most k_disjunctive_size atoms drawn from the fact landmarks: the fact landmarks in
// discovery order, then the disjunctive ones in the graph's order. A disjunctive landmark holds when one of its atoms
// holds; a derived atom holds when the state's closure under the axioms has it.
//
// Status (one bit per landmark): the landmarks reached on the path, i.e. true in some state of it. A transition adds
// those true in the successor, and a state reached on several paths keeps the landmarks every path reached, as Fast
// Downward intersects its past landmarks. The landmarks are those of the task's initial state, so a path starts there:
// the status of the initial state is the landmarks true in it. A path may start in another state s as well (a search's
// start, evaluate(s) without a status): its status is then what every path from the initial state to s reaches, as
// far as s shows it (the landmarks true in s or in the initial state, and those ordered greedy-necessarily before a
// landmark true in s but not initially), and a flag bit after the landmarks marks it as inferred.
// Fast Downward also stores the future landmarks and unites them over the paths; with goal and greedy-necessary
// orderings only (no reasonable orderings), its future landmarks are those computed here from the reached ones.
//
// Value of a state s with reached set R: the needed landmarks are those not in R (they must still be reached) and those
// in R that are false in s and either a goal atom or ordered greedy-necessarily before a landmark not in R (they must
// hold again; for an inferred status, not those that have no achiever and are needed only for an ordering, as the
// path to s may have reached the later landmark). Unit costs count them; real costs add, per needed landmark, the
// cheapest cost of its first achievers
// (not reached yet) or of all its achievers (needed again), as Fast Downward's landmark_sum. A landmark with a derived
// atom costs the cheapest action (some action must change the state for an axiom to derive it; Fast Downward's). A
// needed landmark without such an achiever makes s a dead end (+inf): sound for these landmarks and orderings (a
// landmark without first achievers that is not true initially can never be reached).
//
// Preferred operators: an applicable action is preferred if one of its effects that fires in the evaluated state adds
// an atom of a needed landmark that is false there (Fast Downward's); a condition over a derived atom counts as
// firing. Grounded only: a state outside the grounding raises std::runtime_error.

#include "landmark_heuristics.hpp"
#include "relaxation_detail.hpp"

#include "mymyr/core/bitset.hpp"
#include "mymyr/heuristics/action_costs.hpp"
#include "mymyr/landmarks/approximate.hpp"
#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"
#include "mymyr/task/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace mymyr::heuristics::detail
{
namespace
{
/// The largest disjunctive landmark (as Fast Downward's lm_rhw, which keeps those of fewer than five atoms).
constexpr usize k_disjunctive_size = 4;

class LandmarkCountHeuristic final : public Heuristic
{
public:
    LandmarkCountHeuristic(const Task& task, const Options& o) : m_task(task), m_real(resolve_costs(task, o.costs) == Costs::Real)
    {
        if (o.evaluation == Evaluation::Lifted)
            throw std::invalid_argument("mymyr: landmark count has no lifted evaluation (use Evaluation::Auto or Grounded)");
        if (m_real)
            m_scale = ActionCosts(task).relaxed_scale();
        const auto t0 = std::chrono::steady_clock::now();
        m_R = o.relaxed;
        if (!m_R)
        {
            GroundingStats st;
            m_R = RelaxedTask::build(task, o.budget, &st);
            if (!m_R)
                throw std::runtime_error("mymyr: grounding the relaxation exceeded the budget, and landmark count has no lifted "
                                         "evaluation: " +
                                         st.reason);
        }
        if (m_real && !m_R->real_costs_available())
            throw std::invalid_argument("mymyr: real-cost heuristics need defined, non-negative action costs below 2^31 units of "
                                        "the cost scale (heuristics::ActionCosts::relaxed_cost)");
        build();
        m_stats.grounding_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    [[nodiscard]] Kind kind() const noexcept override { return Kind::LmCount; }
    [[nodiscard]] std::shared_ptr<const RelaxedTask> relaxed() const override { return m_R; }
    [[nodiscard]] bool path_dependent() const noexcept override { return true; }
    [[nodiscard]] u32 status_words() const noexcept override { return m_words; }
    [[nodiscard]] bool provides_preferred() const noexcept override { return true; }

    void initial_status(StateView s, std::span<u64> out) override
    {
        std::fill(out.begin(), out.end(), u64{0});
        add_true(s, out, m_scan);
        if (s == m_s0.view())
            return;
        // reached on every path from the initial state: the landmarks true there, and transitively those ordered
        // before a landmark that became true on the way (it was false initially)
        const u32 n = static_cast<u32>(m_lms.size());
        m_work.clear();
        for (u32 i = 0; i < n; ++i)
            if (bits::test(out.data(), m_words, i) && !bits::test(m_init.data(), m_words, i))
                m_work.push_back(i);
        for (usize w = 0; w < m_words; ++w)
            out[w] |= m_init[w];
        while (!m_work.empty())
        {
            const u32 i = m_work.back();
            m_work.pop_back();
            for (u32 k = m_pred_begin[i]; k < m_pred_begin[i + 1]; ++k)
                if (const u32 p = m_pred[k]; !bits::test(out.data(), m_words, p))
                {
                    bits::set(out.data(), p);
                    if (!bits::test(m_init.data(), m_words, p))
                        m_work.push_back(p);
                }
        }
        bits::set(out.data(), n);  // inferred
    }

    /// The landmarks both statuses reached; inferred if either is.
    bool merge(std::span<u64> into, std::span<const u64> other) const override
    {
        const u32 flag = static_cast<u32>(m_lms.size());
        const bool inferred = bits::test(into.data(), m_words, flag) || bits::test(other.data(), m_words, flag);
        bool changed = false;
        for (usize w = 0; w < into.size(); ++w)
        {
            u64 x = into[w] & other[w];
            if (inferred && w == flag / 64)
                x |= u64{1} << (flag % 64);
            changed |= x != into[w];
            into[w] = x;
        }
        return changed;
    }

    void progress(std::span<const u64> parent_status, StateView /*parent*/, const ActionLabel& /*action*/, StateView child,
                  std::span<u64> out) override
    {
        std::copy(parent_status.begin(), parent_status.end(), out.begin());
        add_true(child, out, m_scan);
    }

    Value evaluate(StateView s) override
    {
        m_tmp.assign(m_words, 0);
        initial_status(s, m_tmp);
        return evaluate(s, m_tmp);
    }

    Value evaluate(StateView s, std::span<const u64> status) override
    {
        ++m_stats.evaluations;
        if (status.size() != m_words)
            throw std::invalid_argument("mymyr: a landmark count status has " + std::to_string(status.size()) + " words, not " +
                                        std::to_string(m_words));
        convert(s, m_eval);
        m_plan_valid = false;
        u64 h = 0;
        if (m_R->goal_unreachable())
            h = k_inf;
        const u32 n = static_cast<u32>(m_lms.size());
        m_reached.assign(status.begin(), status.end());
        const bool inferred = bits::test(status.data(), m_words, n);
        m_true.assign(m_words, 0);
        for (u32 i = 0; i < n && h < k_inf; ++i)
            if (holds(i, m_eval))
            {
                bits::set(m_true.data(), i);
                bits::set(m_reached.data(), i);
            }
        m_needed.assign(m_words, 0);
        for (u32 i = 0; i < n && h < k_inf; ++i)
        {
            u32 c;
            if (!bits::test(m_reached.data(), m_words, i))
                c = m_lms[i].first_cost;
            else if (!bits::test(m_true.data(), m_words, i) && needed_again(i))
            {
                c = m_lms[i].cost;
                if (c == k_inf && inferred && !m_lms[i].goal)
                    continue;  // the path to s may have reached the landmark ordered after it
            }
            else
                continue;
            bits::set(m_needed.data(), i);
            h = c == k_inf ? k_inf : h + c;
        }
        m_plan_valid = h < k_inf;
        const Value v = to_value(std::min<u64>(h, k_inf), m_scale);
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    /// Landmark count estimates the task's goal: its landmarks are the task goal's.
    Value evaluate(StateView, std::span<const search::GoalSpec::AtomGoal>) override
    {
        throw std::invalid_argument("mymyr: landmark count estimates the task's goal only (its landmarks are the goal's); "
                                    "GoalSpec::AnyOf is not supported");
    }

    [[nodiscard]] bool preferred(const ActionLabel& a) const override
    {
        if (!m_plan_valid)
            return false;
        const u32 ga = m_R->find_ground_action(a);
        if (ga == RelaxedTask::k_none)
            return false;
        for (u32 k = m_ga_adds_begin[ga]; k < m_ga_adds_begin[ga + 1]; ++k)
        {
            const auto [op, lm] = m_ga_adds[k];
            if (bits::test(m_needed.data(), m_words, lm) && !bits::test(m_true.data(), m_words, lm) && fires(op))
                return true;
        }
        return false;
    }

private:
    struct Landmark
    {
        std::vector<u32> props;  // "true" propositions of its atoms (a fact landmark has one)
        bool goal = false;       // a fact landmark that is a goal atom
        u32 first_cost = k_inf;  // cheapest first achiever (in cost units; k_inf: none)
        u32 cost = k_inf;        // cheapest achiever
    };

    /// Per-state scratch: the state's propositions, marked.
    struct Scan
    {
        StateProps props;
        std::vector<u32> derived;  // the "true" propositions of the derived landmark atoms that hold
        std::vector<u32> mark;
        u32 stamp = 0;
    };

    void build()
    {
        const RelaxedTask& R = *m_R;
        landmarks::ApproximateFactLandmarkOptions lo;
        lo.first_achievers = landmarks::FirstAchievers::PossiblyBefore;
        lo.max_disjunctive_landmark_size = k_disjunctive_size;
        lo.max_disjunctive_landmark_depth = 1;
        lo.include_derived_atoms = true;
        const landmarks::FactLandmarkGraph g = landmarks::approximate_fact_landmarks(R, lo);
        const landmarks::AchieverIndex& ix = g.achievers();
        std::vector<u8> goal(R.num_props(), 0);
        for (u32 p : R.goal())
            goal[p] = 1;
        // the landmarks and their atoms' propositions
        std::vector<CanonicalAtom> facts(g.landmarks().begin(), g.landmarks().end());
        for (CanonicalAtom c : facts)
        {
            Landmark& l = m_lms.emplace_back();
            const u32 pp = R.fluent_props(c).first;
            if (pp != k_none)
            {
                l.props.push_back(pp);
                l.goal = goal[pp] != 0;
            }
        }
        for (const auto& members : g.disjunctive())
        {
            Landmark& l = m_lms.emplace_back();
            for (CanonicalAtom c : members)
                if (const u32 pp = R.fluent_props(c).first; pp != k_none)
                    l.props.push_back(pp);
        }
        const u32 n = static_cast<u32>(m_lms.size());
        m_words = (n + 1 + 63) / 64;  // and the flag of an inferred status
        // greedy-necessary orderings: per landmark the landmarks it must hold right before
        {
            std::vector<std::vector<u32>> succ(n);
            for (const auto& [before, after] : g.orderings())
            {
                const auto bi = std::find(facts.begin(), facts.end(), before) - facts.begin();
                const auto ai = std::find(facts.begin(), facts.end(), after) - facts.begin();
                if (bi < static_cast<std::ptrdiff_t>(facts.size()) && ai < static_cast<std::ptrdiff_t>(facts.size()))
                    succ[static_cast<usize>(bi)].push_back(static_cast<u32>(ai));
            }
            std::vector<std::vector<u32>> pred(n);
            m_succ_begin.assign(1, 0);
            for (u32 i = 0; i < n; ++i)
            {
                m_succ.insert(m_succ.end(), succ[i].begin(), succ[i].end());
                m_succ_begin.push_back(static_cast<u32>(m_succ.size()));
                for (u32 j : succ[i])
                    pred[j].push_back(i);
            }
            m_pred_begin.assign(1, 0);
            for (const auto& v : pred)
            {
                m_pred.insert(m_pred.end(), v.begin(), v.end());
                m_pred_begin.push_back(static_cast<u32>(m_pred.size()));
            }
        }
        // proposition -> landmarks holding it
        {
            std::vector<std::vector<u32>> of(R.num_props());
            for (u32 i = 0; i < n; ++i)
                for (u32 p : m_lms[i].props)
                    of[p].push_back(i);
            m_prop_lms_begin.assign(1, 0);
            for (const auto& v : of)
            {
                m_prop_lms.insert(m_prop_lms.end(), v.begin(), v.end());
                m_prop_lms_begin.push_back(static_cast<u32>(m_prop_lms.size()));
            }
        }
        // the derived landmark atoms, which the axioms achieve: the cheapest action
        const u64 fluents = m_task.atoms().layout().fluent_count;
        auto unit_or_real = [&](u32 ga) { return m_real ? R.real_cost(ga) : u32{1}; };
        u32 cheapest = k_inf;
        for (u32 ga = 0; ga < R.num_ground_actions(); ++ga)
            cheapest = std::min(cheapest, unit_or_real(ga));
        std::vector<u8> derived_prop(R.num_props(), 0);
        for (u32 i = 0; i < n; ++i)
            for (u32 p : m_lms[i].props)
                if (R.prop_atom(p) >= fluents && !derived_prop[p])
                {
                    derived_prop[p] = 1;
                    m_derived.emplace_back(p, R.prop_atom(p));
                }
        // achievers: every operator adding an atom of the landmark; first achievers from the graph (fact landmarks)
        std::vector<std::vector<std::pair<u32, u32>>> adds(R.num_ground_actions());
        for (u32 o = 0; o < R.num_ops(); ++o)
        {
            if (R.is_axiom(o) || !R.static_condition_holds(o))
                continue;
            const u32 ga = R.ground_action(o);
            for (u32 q : R.eff(o))
                for (u32 k = m_prop_lms_begin[q]; k < m_prop_lms_begin[q + 1]; ++k)
                {
                    const u32 i = m_prop_lms[k];
                    m_lms[i].cost = std::min(m_lms[i].cost, unit_or_real(ga));
                    adds[ga].emplace_back(o, i);
                }
        }
        for (u32 i = 0; i < n; ++i)
        {
            Landmark& l = m_lms[i];
            for (u32 p : l.props)
                if (derived_prop[p])
                    l.cost = std::min(l.cost, cheapest);
            if (i >= facts.size() || (!l.props.empty() && derived_prop[l.props[0]]))
            {
                l.first_cost = l.cost;  // disjunctive or derived: every achiever
                continue;
            }
            for (u32 a : ix.first_achievers_of(facts[i]))
            {
                const u32 ga = R.find_ground_action(ix.actions[a].label());
                if (ga != RelaxedTask::k_none)
                    l.first_cost = std::min(l.first_cost, unit_or_real(ga));
            }
        }
        m_ga_adds_begin.assign(1, 0);
        for (auto& v : adds)
        {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            m_ga_adds.insert(m_ga_adds.end(), v.begin(), v.end());
            m_ga_adds_begin.push_back(static_cast<u32>(m_ga_adds.size()));
        }
        m_scan.mark.assign(R.num_props(), 0);
        m_eval.mark.assign(R.num_props(), 0);
        m_s0 = m_task.initial_state();
        m_init.assign(m_words, 0);
        add_true(m_s0.view(), m_init, m_scan);
    }

    /// Converts s into `scan` and marks its "true" propositions.
    void convert(StateView s, Scan& scan)
    {
        if (!scan.props.convert(m_task, *m_R, s))
            throw std::runtime_error("mymyr: the state lies outside the grounded relaxation, and landmark count has no lifted "
                                     "evaluation");
        if (++scan.stamp == 0)
        {
            std::fill(scan.mark.begin(), scan.mark.end(), 0);
            scan.stamp = 1;
        }
        for (u32 p : scan.props.pos)
            scan.mark[p] = scan.stamp;
        scan.derived.clear();
        if (!m_derived.empty())
        {
            const WorkspaceLease lease = m_task.workspace();
            Successors& succ = lease->successors();
            succ.prepare(s);
            const mymyr::detail::Engine& e = succ.engine();
            for (const auto& [p, c] : m_derived)
            {
                const u32 slot = m_task.atoms().find(c);
                if (slot != AtomIndex::k_empty && bits::test(e.derived(), e.derived_words(), slot))
                {
                    scan.mark[p] = scan.stamp;
                    scan.derived.push_back(p);
                }
            }
        }
    }

    [[nodiscard]] bool holds(u32 i, const Scan& scan) const
    {
        for (u32 p : m_lms[i].props)
            if (scan.mark[p] == scan.stamp)
                return true;
        return false;
    }

    /// Sets the bits of the landmarks true in s.
    void add_true(StateView s, std::span<u64> out, Scan& scan)
    {
        convert(s, scan);
        auto set = [&](u32 p)
        {
            for (u32 k = m_prop_lms_begin[p]; k < m_prop_lms_begin[p + 1]; ++k)
                out[m_prop_lms[k] >> 6] |= u64{1} << (m_prop_lms[k] & 63);
        };
        for (u32 p : scan.props.pos)
            set(p);
        for (u32 p : scan.derived)
            set(p);
    }

    /// A reached landmark false in the state: a goal, or ordered greedy-necessarily before an unreached landmark.
    [[nodiscard]] bool needed_again(u32 i) const
    {
        if (m_lms[i].goal)
            return true;
        for (u32 k = m_succ_begin[i]; k < m_succ_begin[i + 1]; ++k)
            if (!bits::test(m_reached.data(), m_words, m_succ[k]))
                return true;
        return false;
    }

    /// Whether operator o's effect fires in the evaluated state (its action being applicable): its "true" fluent
    /// preconditions hold and its "false" ones are not negated by the state; derived ones count as holding.
    [[nodiscard]] bool fires(u32 o) const
    {
        if (m_R->unconditional(o))
            return true;
        const u64 fluents = m_task.atoms().layout().fluent_count;
        for (u32 p : m_R->pre(o))
        {
            if (m_R->prop_atom(p) >= fluents)
                continue;  // derived
            if (m_R->negative(p))
            {
                if (std::find(m_eval.props.neg.begin(), m_eval.props.neg.end(), p) != m_eval.props.neg.end())
                    return false;
            }
            else if (m_eval.mark[p] != m_eval.stamp)
                return false;
        }
        return true;
    }

    const Task& m_task;
    bool m_real;
    f64 m_scale = 1;  // real costs: the unit of the integer costs is 1 / m_scale
    std::shared_ptr<const RelaxedTask> m_R;
    std::vector<Landmark> m_lms;
    u32 m_words = 0;
    std::vector<u32> m_succ_begin, m_succ, m_pred_begin, m_pred, m_prop_lms_begin, m_prop_lms, m_ga_adds_begin;
    State m_s0;               // the task's initial state
    std::vector<u64> m_init;  // the landmarks true in it
    std::vector<std::pair<u32, u32>> m_ga_adds;  // per ground action: (operator, landmark) for each landmark atom it adds
    std::vector<std::pair<u32, CanonicalAtom>> m_derived;  // the derived landmark atoms: ("true" proposition, atom)
    // scratch: transitions (m_scan) and the last evaluation (m_eval, the bit sets below)
    Scan m_scan, m_eval;
    std::vector<u64> m_tmp, m_reached, m_true, m_needed;
    std::vector<u32> m_work;
    bool m_plan_valid = false;
};
}  // namespace

std::unique_ptr<Heuristic> make_landmark_count(const Task& task, const Options& options)
{
    return std::make_unique<LandmarkCountHeuristic>(task, options);
}
}  // namespace mymyr::heuristics::detail
