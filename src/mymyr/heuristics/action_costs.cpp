#include "mymyr/heuristics/action_costs.hpp"

#include "mymyr/successor/successors.hpp"
#include "mymyr/task/task.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace mymyr::heuristics
{
using namespace formalism;

namespace
{
constexpr u32 k_inexact = 7;  // more decimal places than the relaxation costs keep (10^6)

/// The smallest k <= 6 such that v is a multiple of 10^-k (up to the precision of a double), else k_inexact.
u32 places_of(f64 v)
{
    for (u32 k = 0; k < k_inexact; ++k)
    {
        const f64 x = v * std::pow(10.0, k);
        if (std::isfinite(x) && std::fabs(x - std::nearbyint(x)) <= 1e-9 * std::max(1.0, std::fabs(x)))
            return k;
    }
    return k_inexact;
}

std::string number_text(f64 v)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

/// A ground expression of the task as PDDL text.
std::string expr_text(const TaskData& T, u32 e)
{
    const Expr& x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return number_text(x.value);
        case ExprOp::Function:
        {
            std::string out = "(" + std::string(T.str(T.functions[x.func.v].name));
            for (Term t : TaskData::slice(T.terms, x.terms))
                out += " " + (is_object(t) && term_object(t).v < T.objects.size() ? std::string(T.str(T.objects[term_object(t).v].name))
                                                                                  : "?" + std::to_string(t));
            return out + ")";
        }
        case ExprOp::Add: return "(+ " + expr_text(T, x.a) + " " + expr_text(T, x.b) + ")";
        case ExprOp::Sub: return "(- " + expr_text(T, x.a) + " " + expr_text(T, x.b) + ")";
        case ExprOp::Mul: return "(* " + expr_text(T, x.a) + " " + expr_text(T, x.b) + ")";
        case ExprOp::Div: return "(/ " + expr_text(T, x.a) + " " + expr_text(T, x.b) + ")";
        case ExprOp::Neg: return "(- " + expr_text(T, x.a) + ")";
    }
    return "?";
}

/// Whether expression e reads the auxiliary function total-cost.
bool reads_total_cost(const TaskData& T, u32 e)
{
    const Expr& x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return false;
        case ExprOp::Function: return T.functions[x.func.v].kind == FuncKind::Auxiliary;
        case ExprOp::Neg: return reads_total_cost(T, x.a);
        default: return reads_total_cost(T, x.a) || reads_total_cost(T, x.b);
    }
}
}  // namespace

ActionCosts::ActionCosts(const Task& task) : m_task(&task), m_num(&task.numeric())
{
    const TaskData& T = task.data();
    m_num_objects = T.num_objects();
    const plan::Numeric& N = task.numeric();
    m_schemas.assign(T.schemas.size(), SchemaCost{});
    bool total_cost = N.has_aux;  // no metric: total-cost if the domain declares it (mimir)
    if (T.metric)
    {
        m_metric_text = std::string("(:metric ") + (T.metric->minimize ? "minimize " : "maximize ") + expr_text(T, T.metric->expr) + ")";
        if (!T.metric->minimize)
            throw std::invalid_argument("mymyr: the metric " + m_metric_text +
                                        " is not supported: maximization is not supported (every search minimizes)");
        const Expr& x = T.exprs[T.metric->expr];
        total_cost = x.op == ExprOp::Function && T.functions[x.func.v].kind == FuncKind::Auxiliary;
        if (!total_cost && reads_total_cost(T, T.metric->expr))
            throw std::invalid_argument("mymyr: the metric " + m_metric_text +
                                        " is not supported: total-cost can only be the whole metric, (:metric minimize "
                                        "(total-cost)); other metrics are expressions over the numeric fluents");
    }
    if (!total_cost)
    {
        if (T.metric)
        {
            // g is the metric evaluated on the state (mimir's state metric)
            m_kind = Kind::StateMetric;
            m_independent = false;
            m_integral = false;
            m_max_constant = -1;
            for (SchemaCost& c : m_schemas)
                c.kind = Cost::Dynamic;
        }
        return;  // Unit: every action costs 1, g starts at 0
    }

    m_kind = Kind::TotalCost;
    m_initial = N.aux_initial;

    // static function values
    const u64 n = std::max<u32>(1, m_num_objects);
    m_func_base.assign(T.functions.size() + 1, 0);
    for (usize f = 0; f < T.functions.size(); ++f)
    {
        u64 size = 1;
        for (u32 i = 0; i < T.functions[f].arity; ++i)
        {
            if (size > (u64{1} << 62) / n)
                throw std::invalid_argument("mymyr: static function '" + std::string(T.str(T.functions[f].name)) +
                                            "' has too many argument combinations for the cost table");
            size *= n;
        }
        m_func_base[f + 1] = m_func_base[f] + size;
        if (m_func_base[f + 1] < m_func_base[f] || m_func_base[f + 1] > (u64{1} << 62))
            throw std::invalid_argument("mymyr: static function key space overflows");
    }
    bool values_integral = true;
    for (const GroundFunctionValue& v : T.static_values)
    {
        u64 key = 0;
        const auto objs = TaskData::slice(T.object_ids, v.objects);
        if (!function_key(v.func.v, objs.data(), static_cast<u32>(objs.size()), key))
            continue;
        m_values[key] = v.value;
        values_integral &= v.value == std::floor(v.value);
    }
    m_func_places.assign(T.functions.size(), 0);
    for (const GroundFunctionValue& v : T.static_values)
        m_func_places[v.func.v] = std::max(m_func_places[v.func.v], places_of(v.value));

    m_max_constant = 0;
    for (usize s = 0; s < T.schemas.size(); ++s)
    {
        const Schema& sc = T.schemas[s];
        SchemaCost& c = m_schemas[s];
        c.kind = Cost::Const;
        c.value = 0;
        bool found = false;
        auto dynamic = [&]
        {
            c.kind = Cost::Dynamic;
            c.params.clear();
            m_independent = false;
            m_integral = false;
            m_max_constant = -1;
        };
        for (const ConditionalEffect& ce : T.effects_of(sc))
        {
            if (!ce.auxiliary || c.kind == Cost::Dynamic)
                continue;
            if (ce.extra_params.count != 0 || ce.condition.literals.count != 0 || ce.condition.constraints.count != 0 ||
                ce.auxiliary->op != AssignOp::Increase || found)
            {
                dynamic();  // conditional, not an increase, or several: the successor generator evaluates them
                continue;
            }
            found = true;
            c.expr = ce.auxiliary->expr;
            // classify the expression: constant (numbers only), depending on static functions, or on fluents
            bool depends = false, integral = true, fluent = false;
            std::function<void(u32)> scan = [&](u32 e)
            {
                const Expr& x = T.exprs[e];
                switch (x.op)
                {
                    case ExprOp::Number: integral &= x.value == std::floor(x.value); return;
                    case ExprOp::Function:
                    {
                        const Function& f = T.functions[x.func.v];
                        if (f.kind != FuncKind::Static)
                        {
                            fluent = true;
                            return;
                        }
                        depends = true;
                        integral &= values_integral;
                        for (Term t : TaskData::slice(T.terms, x.terms))
                            if (!is_object(t) && std::find(c.params.begin(), c.params.end(), term_parameter(t)) == c.params.end())
                                c.params.push_back(term_parameter(t));
                        return;
                    }
                    case ExprOp::Neg: scan(x.a); return;
                    case ExprOp::Div: integral = false; [[fallthrough]];
                    default:
                        scan(x.a);
                        scan(x.b);
                }
            };
            scan(c.expr);
            if (fluent)
            {
                dynamic();
                continue;
            }
            std::sort(c.params.begin(), c.params.end());
            m_integral &= integral;
            if (depends)
            {
                c.kind = Cost::Expr;
                m_max_constant = -1;
            }
            else
            {
                c.value = evaluate(c.expr, nullptr);
                if (!(c.value >= 0))
                    throw std::invalid_argument(std::string(std::isnan(c.value) ? "mymyr: undefined action cost (a division by zero) in schema '"
                                                                                : "mymyr: negative action cost in schema '") +
                                                std::string(T.str(sc.name)) + "'");
                if (m_max_constant >= 0)
                    m_max_constant = std::max(m_max_constant, c.value);
            }
        }
    }
    u32 places = 0;
    for (const SchemaCost& c : m_schemas)
        if (c.kind == Cost::Const)
            places = std::max(places, places_of(c.value));
        else if (c.kind == Cost::Expr)
            places = std::max(places, decimal_places(c.expr));
    m_exact = places < k_inexact;
    m_scale = std::pow(10.0, std::min(places, k_inexact - 1));
}

u32 ActionCosts::decimal_places(u32 e) const
{
    const TaskData& T = m_task->data();
    const Expr& x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return places_of(x.value);
        case ExprOp::Function: return x.func.v < m_func_places.size() ? m_func_places[x.func.v] : k_inexact;
        case ExprOp::Neg: return decimal_places(x.a);
        case ExprOp::Add:
        case ExprOp::Sub: return std::max(decimal_places(x.a), decimal_places(x.b));
        case ExprOp::Mul: return std::min(k_inexact, decimal_places(x.a) + decimal_places(x.b));
        case ExprOp::Div: return k_inexact;
    }
    return k_inexact;
}

u64 ActionCosts::relaxed_cost(u32 schema, const ObjectId* binding) const
{
    if (m_kind == Kind::Unit)
        return 1;
    const SchemaCost& c = m_schemas[schema];
    if (c.kind == Cost::Dynamic)
        return 0;
    const f64 v = c.kind == Cost::Const ? c.value : evaluate(c.expr, binding);
    if (std::isnan(v))
        return k_undefined;
    if (v < 0)
        throw std::domain_error("mymyr: negative action cost");
    const f64 n = m_exact ? std::nearbyint(v * m_scale) : std::floor(v * m_scale);
    return n >= 9223372036854775808.0 ? u64{1} << 63 : static_cast<u64>(n);
}

void ActionCosts::undefined_metric(const char* where) const
{
    throw std::domain_error("mymyr: the metric " + m_metric_text + " is undefined in " + where +
                            " (an undefined function value or a division by zero)");
}

f64 plan_metric(Successors& succ, const ActionCosts& costs, const State& start, f64 g0, std::vector<Action>& plan)
{
    if (costs.unit())
        return g0 + static_cast<f64>(plan.size());
    f64 g = g0;
    State s = start;
    StateBuilder b;
    std::vector<u64> tmp;
    for (Action& a : plan)
    {
        const Delta da = succ.apply_with_delta(s, a.label(), b);
        f64 best = costs.next(g, da);
        const State to = b.build();
        const u32 nnum = to.numeric_words();
        succ.prepare(s);
        succ.generate<false>(
            [&](u32 schema, const ObjectId* bind, const Delta& d)
            {
                const u32 nn = apply_delta(s.data(), s.size_words(), d, tmp);
                if (!bits::equal(tmp.data(), nn, to.data(), to.size_words()))
                    return true;
                if (nnum && std::memcmp(d.num, to.numeric().data(), nnum * sizeof(u64)) != 0)
                    return true;
                const f64 c = costs.next(g, d);
                if (c < best)
                {
                    best = c;
                    a = Action(SchemaId{schema}, std::vector<ObjectId>(bind, bind + succ.arity(schema)));
                }
                return true;
            },
            false, true);
        g = best;
        s = to;
    }
    return g;
}

f64 ActionCosts::initial(StateView s) const
{
    if (m_kind != Kind::StateMetric)
        return m_initial;
    const f64 v = plan::eval(*m_num, m_num->metric, s.num, nullptr);
    if (std::isnan(v))
        undefined_metric("the start state");
    return v;
}

bool ActionCosts::function_key(u32 func, const ObjectId* args, u32 arity, u64& key) const noexcept
{
    if (func + 1 >= m_func_base.size())
        return false;
    u64 k = 0, mul = 1;
    for (u32 i = 0; i < arity; ++i)
    {
        if (args[i].v >= m_num_objects)
            return false;
        k += mul * args[i].v;
        mul *= std::max<u32>(1, m_num_objects);
    }
    key = m_func_base[func] + k;
    return true;
}

f64 ActionCosts::evaluate(u32 e, const ObjectId* binding) const
{
    const TaskData& T = m_task->data();
    const Expr& x = T.exprs[e];
    switch (x.op)
    {
        case ExprOp::Number: return x.value;
        case ExprOp::Function:
        {
            ObjectId args[64];
            const auto terms = TaskData::slice(T.terms, x.terms);
            if (terms.size() > 64)
                return std::numeric_limits<f64>::quiet_NaN();
            for (usize i = 0; i < terms.size(); ++i)
                args[i] = is_object(terms[i]) ? term_object(terms[i]) : binding[term_parameter(terms[i])];
            u64 key = 0;
            if (!function_key(x.func.v, args, static_cast<u32>(terms.size()), key))
                return std::numeric_limits<f64>::quiet_NaN();
            const auto it = m_values.find(key);
            return it == m_values.end() ? std::numeric_limits<f64>::quiet_NaN() : it->second;
        }
        case ExprOp::Add: return evaluate(x.a, binding) + evaluate(x.b, binding);
        case ExprOp::Sub: return evaluate(x.a, binding) - evaluate(x.b, binding);
        case ExprOp::Mul: return evaluate(x.a, binding) * evaluate(x.b, binding);
        case ExprOp::Div:
        {
            const f64 a = evaluate(x.a, binding), b = evaluate(x.b, binding);
            return b == 0 ? std::numeric_limits<f64>::quiet_NaN() : a / b;  // undefined, as in numeric expressions
        }
        case ExprOp::Neg: return -evaluate(x.a, binding);
    }
    return std::numeric_limits<f64>::quiet_NaN();
}

f64 ActionCosts::evaluate_checked(u32 expr, const ObjectId* binding) const
{
    const f64 v = evaluate(expr, binding);
    if (!(v >= 0))  // NaN (undefined static function value) or negative
        throw std::domain_error(std::isnan(v) ? "mymyr: action cost undefined (a static function value is missing or a "
                                                "division by zero)"
                                              : "mymyr: negative action cost");
    return v;
}
}  // namespace mymyr::heuristics
