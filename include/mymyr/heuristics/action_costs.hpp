#pragma once
// Action costs and metric values: the objective every search, state space and heuristic shares. The kind follows the
// problem's metric:
//   - `minimize (total-cost)`, or no metric in a domain that declares the auxiliary function total-cost (as in mimir;
//     Kind::TotalCost): the metric value of a state starts at the problem's initial total-cost (0 when the problem sets
//     none) and every transition applies, in order, the total-cost effects of the (conditional) effects that fire,
//     whatever their operator (increase, decrease, assign, scale-up, scale-down) and expression (numbers, static and
//     fluent functions of the parameters and objects) -- matching mimir's `apply_action_effects`;
//   - `minimize <expression>` over the state's numeric fluents, static functions and numbers (Kind::StateMetric): the
//     expression evaluated on the state's values (mimir's `compute_state_metric_value`); total-cost effects do not
//     count, also when the domain declares total-cost;
//   - no metric and no total-cost (Kind::Unit): every action costs 1 and g starts at 0.
// The constructor throws std::invalid_argument, naming the metric, for `maximize` (every search minimizes; a
// maximization is not turned into a minimization) and for a metric that combines total-cost with other terms. A
// state whose metric value is undefined (an undefined fluent, a division by zero) makes initial() and next() throw
// std::domain_error naming the metric: the objective has no value there.
// The successor generator evaluates the total-cost effects of each applicable action on the parent state (mimir's
// applicability rules apply to them: an undefined value or an effect-family conflict makes the action inapplicable)
// and hands them over in Delta::aux; next(g, delta) turns them into the successor's metric value.
//
// State-independent costs: when every total-cost effect of a schema is an unconditional `increase` by an expression
// over numbers and static functions, cost(schema, binding) gives the action's cost without a state (the heuristics' real
// costs, bucket queues). state_independent() says whether this holds for every schema.
//
// Relaxation costs: the relaxation heuristics (h_max, h_add, h_FF, set-additive, h², on the CPU and the device) work
// with integer costs. relaxed_cost(schema, binding) is the action's cost in units of 1 / relaxed_scale(), where
// relaxed_scale() = 10^k for the smallest k <= 6 such that every state-independent cost is a multiple of 10^-k (so
// 0.25 is 25 at scale 100, exactly); costs that are not are rounded down at scale 10^6 (lower bounds). An action whose
// cost depends on the state, and every action under a state metric, costs 0 there: a lower bound of its cost when the
// costs are non-negative. Unit costs are 1 at scale 1.
//
// Thread-safety: immutable after construction; every member may be called from any thread.

#include "mymyr/core/ids.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/state/state.hpp"
#include "mymyr/successor/action.hpp"
#include "mymyr/task/numeric.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mymyr
{
class Task;
class Successors;
}

namespace mymyr::heuristics
{
class ActionCosts
{
public:
    enum class Kind : u8
    {
        Unit,
        TotalCost,
        StateMetric,
    };

    explicit ActionCosts(const Task& task);

    [[nodiscard]] Kind kind() const noexcept { return m_kind; }
    /// No total-cost and no metric: every action costs 1.
    [[nodiscard]] bool unit() const noexcept { return m_kind == Kind::Unit; }
    /// Every action's cost is a function of its binding only (see above); the StateMetric kind never is.
    [[nodiscard]] bool state_independent() const noexcept { return m_independent; }
    /// Every state-independent cost is a non-negative integer (bounded searches may then use bucket queues): no
    /// division, integral numbers and static function values. False when a cost depends on the state.
    [[nodiscard]] bool integral() const noexcept { return m_integral; }
    /// Largest cost when every schema's cost is a constant (unit costs: 1), else a negative value.
    [[nodiscard]] f64 max_constant() const noexcept { return m_max_constant; }
    /// g of the initial state (TotalCost, Unit); for StateMetric use initial(state).
    [[nodiscard]] f64 initial() const noexcept { return m_initial; }
    /// mimir's metric value of the start state s. Throws std::domain_error naming the metric when it is undefined in s.
    [[nodiscard]] f64 initial(StateView s) const;
    /// mimir's metric value of the successor under delta d of a state with metric value g. Throws std::domain_error
    /// naming the metric when it is undefined in the successor.
    [[nodiscard]] f64 next(f64 g, const Delta& d) const
    {
        switch (m_kind)
        {
            case Kind::Unit: return g + 1;
            case Kind::TotalCost:
                for (const AuxWrite& a : d.aux)
                    g = apply_assign(a.op, g, a.value);
                return g;
            case Kind::StateMetric:
            {
                const f64 v = plan::eval(*m_num, m_num->metric, d.num, nullptr);
                if (std::isnan(v))
                    undefined_metric("a reached state");
                return v;
            }
        }
        return g + 1;
    }
    /// Whether schema s costs the same under every binding (state-independent schemas only).
    [[nodiscard]] bool constant(u32 schema) const noexcept { return m_schemas[schema].kind == Cost::Const; }
    /// The state-independent cost of action (schema, binding). Throws std::domain_error for a negative, NaN or
    /// undefined (missing static function value) cost, std::logic_error for a state-dependent schema.
    [[nodiscard]] f64 cost(u32 schema, const ObjectId* binding) const
    {
        const SchemaCost& c = m_schemas[schema];
        if (c.kind == Cost::Const)
            return c.value;
        if (c.kind == Cost::Dynamic)
            throw std::logic_error("mymyr: the cost of this action depends on the state (use ActionCosts::next)");
        return evaluate_checked(c.expr, binding);
    }
    /// Parameters of schema s its cost depends on (empty for constant costs).
    [[nodiscard]] const std::vector<u32>& cost_parameters(u32 schema) const noexcept { return m_schemas[schema].params; }

    /// relaxed_cost() of an action whose state-independent cost is undefined (a missing static function value or a
    /// division by zero): its total-cost effect is undefined, so it is never applicable.
    static constexpr u64 k_undefined = ~u64{0};
    /// The integer cost of action (schema, binding) for the relaxation heuristics, in units of 1 / relaxed_scale() (see
    /// above); k_undefined for an action that is never applicable. Throws std::domain_error for a negative cost.
    [[nodiscard]] u64 relaxed_cost(u32 schema, const ObjectId* binding) const;
    /// 10^k: the unit of relaxed_cost() is 1 / relaxed_scale().
    [[nodiscard]] f64 relaxed_scale() const noexcept { return m_scale; }
    /// Whether every relaxed_cost() is the exact cost times relaxed_scale() (else a lower bound).
    [[nodiscard]] bool relaxed_exact() const noexcept { return m_exact; }
    /// The problem's metric as text, e.g. "(:metric minimize (total-cost))"; empty without a metric.
    [[nodiscard]] const std::string& metric_text() const noexcept { return m_metric_text; }
    /// Throws the std::domain_error of a metric value that is undefined (NaN) in `where` (e.g. "a reached state"):
    /// for evaluations of the metric outside next() and initial() (the device's).
    [[noreturn]] void undefined_metric(const char* where) const;

private:
    enum class Cost : u8
    {
        Const,
        Expr,
        Dynamic,  // conditional, not an increase, over fluent functions, or several effects: next() only
    };
    struct SchemaCost
    {
        Cost kind = Cost::Const;
        f64 value = 1;
        u32 expr = 0;
        std::vector<u32> params;
    };

    [[nodiscard]] u32 decimal_places(u32 expr) const;
    [[nodiscard]] f64 evaluate(u32 expr, const ObjectId* binding) const;
    [[nodiscard]] f64 evaluate_checked(u32 expr, const ObjectId* binding) const;
    [[nodiscard]] bool function_key(u32 func, const ObjectId* args, u32 arity, u64& key) const noexcept;

    const Task* m_task;
    const plan::Numeric* m_num;
    Kind m_kind = Kind::Unit;
    bool m_independent = true;
    bool m_integral = true;
    f64 m_max_constant = 1;
    f64 m_initial = 0;
    f64 m_scale = 1;
    bool m_exact = true;
    std::string m_metric_text;
    std::vector<u32> m_func_places;  // per static function: decimal places of its values
    std::vector<SchemaCost> m_schemas;
    // static function values: key = mixed radix over (function, arguments)
    u32 m_num_objects = 0;
    std::vector<u64> m_func_base;  // per function: first key
    std::unordered_map<u64, f64> m_values;
};

/// mimir's plan extraction (matching `extract_total_ordered_plan`): replays `plan` from `start`, whose metric
/// value is g0. At each step, among the applicable actions whose successor equals the plan's next state, the one that
/// gives the successor the lowest metric value is taken (the plan's own action unless another is strictly lower; the
/// first such in canonical order). Rewrites the plan's actions accordingly and returns the final metric value, which
/// is the plan's cost. Throws std::invalid_argument if an action of the plan is not applicable.
f64 plan_metric(Successors& succ, const ActionCosts& costs, const State& start, f64 g0, std::vector<Action>& plan);
}  // namespace mymyr::heuristics
