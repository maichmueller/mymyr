// The encoding goal: the problem's goal as written, expanded into signed ground literals (TaskData::encoding_goal).
// The expansion works on loki's parsed problem, before goal normalization turns quantified and disjunctive goals into
// derived predicates.

#include "translate.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace mymyr::frontend::detail
{
namespace
{
using namespace formalism;

/// A goal may cost at most this many quantifier bindings (counted up front); a conjunct that would exceed what is left
/// is left out and the goal is marked incomplete.
constexpr u64 k_max_bindings = u64{1} << 24;

u64 saturating_add(u64 a, u64 b) { return std::min(k_max_bindings + 1, a + b); }

u64 saturating_mul(u64 a, u64 b)
{
    if (a == 0 || b == 0)
        return 0;
    return a > (k_max_bindings + 1) / b ? k_max_bindings + 1 : std::min(k_max_bindings + 1, a * b);
}

class Expander
{
public:
    Expander(const DomainState& ds, TaskData& t) : m_ds(ds), m_t(t)
    {
        for (u32 o = 0; o < t.objects.size(); ++o)
            m_object.emplace(std::string(t.str(t.objects[o].name)), o);
        for (u32 p = 0; p < t.predicates.size(); ++p)
            m_pred.emplace(std::string(t.str(t.predicates[p].name)), p);
        m_type_pred.assign(t.types.size(), ~0u);
        for (u32 p = 0; p < ds.pred_type.size(); ++p)
            if (ds.pred_type[p] < t.types.size())
                m_type_pred[ds.pred_type[p]] = p;
    }

    void run(const loki::Problem& problem)
    {
        if (problem->get_goal_condition())
            expand(*problem->get_goal_condition(), true);
        set_encoding_goal(m_t, std::move(m_out), m_complete);
    }

private:
    using Value = std::optional<bool>;  // nullopt: not decided by static atoms and equality

    u32 object_of(const loki::Term& term) const
    {
        if (const auto* v = std::get_if<loki::Variable>(&term->get_object_or_variable()))
            return m_binding.at(*v);
        return m_object.at(std::get<loki::Object>(term->get_object_or_variable())->get_name());
    }

    std::vector<ObjectId> objects_of(const loki::Atom& atom) const
    {
        std::vector<ObjectId> out;
        for (const auto& term : atom->get_terms())
            out.push_back(ObjectId{object_of(term)});
        return out;
    }

    bool static_holds(u32 pred, std::span<const ObjectId> objects)
    {
        auto it = m_static.find(pred);
        if (it == m_static.end())
        {
            absl::flat_hash_set<std::vector<u32>> atoms;
            for (const GroundAtom& a : m_t.static_init)
                if (a.pred.v == pred)
                {
                    std::vector<u32> key;
                    for (ObjectId o : m_t.objects_of(a))
                        key.push_back(o.v);
                    atoms.insert(std::move(key));
                }
            it = m_static.emplace(pred, std::move(atoms)).first;
        }
        std::vector<u32> key;
        for (ObjectId o : objects)
            key.push_back(o.v);
        return it->second.contains(key);
    }

    /// Whether the object has the type: it has the type's atom, which loki gives an object for each type it belongs to.
    bool has_type(u32 object, const loki::Type& type)
    {
        const u32 ty = m_ds.type_by_name.at(type->get_name());
        if (m_t.str(m_t.types[ty].name) == "object")
            return true;
        const ObjectId o{object};
        return m_type_pred[ty] != ~0u && static_holds(m_type_pred[ty], std::span<const ObjectId>(&o, 1));
    }

    /// The objects a quantified parameter ranges over, in object order: those of any of its types. Computed once per
    /// parameter.
    const std::vector<u32>& domain_of(const loki::Parameter& parameter)
    {
        auto it = m_domain.find(parameter);
        if (it != m_domain.end())
            return it->second;
        std::vector<u32> out;
        for (u32 o = 0; o < m_t.objects.size(); ++o)
        {
            bool any = parameter->get_bases().empty();
            for (const auto& type : parameter->get_bases())
                any = any || has_type(o, type);
            if (any)
                out.push_back(o);
        }
        return m_domain.emplace(parameter, std::move(out)).first->second;
    }

    /// The number of bindings of the parameters: the product of their domain sizes, saturated.
    u64 bindings_of(const loki::ParameterList& parameters)
    {
        u64 n = 1;
        for (const auto& p : parameters)
            n = saturating_mul(n, domain_of(p).size());
        return n;
    }

    /// The number of quantifier bindings expanding or evaluating the condition visits, at most: a quantifier costs its
    /// bindings times the cost of its body. Saturated above the budget.
    u64 cost(const loki::Condition& c)
    {
        return std::visit(
            [&](const auto& x) -> u64
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::ConditionNot>)
                    return cost(x->get_condition());
                else if constexpr (std::is_same_v<T, loki::ConditionAnd> || std::is_same_v<T, loki::ConditionOr>)
                {
                    u64 n = 0;
                    for (const auto& part : x->get_conditions())
                        n = saturating_add(n, cost(part));
                    return n;
                }
                else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                    return saturating_add(cost(x->get_left_condition()), cost(x->get_right_condition()));
                else if constexpr (std::is_same_v<T, loki::ConditionExists> || std::is_same_v<T, loki::ConditionForall>)
                    return saturating_mul(bindings_of(x->get_parameters()), std::max<u64>(1, cost(x->get_condition())));
                else
                    return 0;
            },
            c->get_condition());
    }

    /// Calls `body` with the parameters bound to every tuple of objects of their types, in lexicographic order, until
    /// it returns false.
    template<class F>
    bool bind_all(const loki::ParameterList& parameters, size_t position, F& body)
    {
        if (position == parameters.size())
            return body();
        const loki::Variable var = parameters[position]->get_variable();
        const auto outer = m_binding.find(var);
        const std::optional<u32> saved = outer == m_binding.end() ? std::nullopt : std::optional<u32>(outer->second);
        bool more = true;
        for (u32 o : domain_of(parameters[position]))
        {
            m_binding[var] = o;
            if (!(more = bind_all(parameters, position + 1, body)))
                break;
        }
        if (saved)
            m_binding[var] = *saved;
        else
            m_binding.erase(var);
        return more;
    }

    /// The truth value of a condition from static atoms and equality alone.
    Value evaluate(const loki::Condition& c)
    {
        return std::visit(
            [&](const auto& x) -> Value
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::ConditionLiteral>)
                {
                    const auto& literal = x->get_literal();
                    const auto objects = objects_of(literal->get_atom());
                    const std::string& name = literal->get_atom()->get_predicate()->get_name();
                    if (name == "=")
                        return (objects[0] == objects[1]) == literal->get_polarity();
                    const auto it = m_pred.find(name);
                    if (it == m_pred.end() || m_t.predicates[it->second].kind != PredKind::Static)
                        return std::nullopt;
                    return static_holds(it->second, objects) == literal->get_polarity();
                }
                else if constexpr (std::is_same_v<T, loki::ConditionNot>)
                {
                    const Value v = evaluate(x->get_condition());
                    return v ? Value(!*v) : std::nullopt;
                }
                else if constexpr (std::is_same_v<T, loki::ConditionAnd>)
                    return connective(x->get_conditions(), false);
                else if constexpr (std::is_same_v<T, loki::ConditionOr>)
                    return connective(x->get_conditions(), true);
                else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                {
                    const Value left = evaluate(x->get_left_condition());
                    const Value right = evaluate(x->get_right_condition());
                    if ((left && !*left) || (right && *right))
                        return true;
                    return left && right ? Value(false) : std::nullopt;
                }
                else if constexpr (std::is_same_v<T, loki::ConditionExists> || std::is_same_v<T, loki::ConditionForall>)
                {
                    constexpr bool any = std::is_same_v<T, loki::ConditionExists>;  // else: all
                    bool decided = false, undecided = false;
                    auto visit = [&]
                    {
                        const Value v = evaluate(x->get_condition());
                        if (!v)
                            undecided = true;
                        else if (*v == any)
                            decided = true;
                        return !decided;
                    };
                    bind_all(x->get_parameters(), 0, visit);
                    if (decided)
                        return any;
                    return undecided ? std::nullopt : Value(!any);
                }
                else
                    return std::nullopt;  // numeric constraints
            },
            c->get_condition());
    }

    /// The value of a disjunction (`any`) or conjunction of conditions.
    Value connective(const loki::ConditionList& parts, bool any)
    {
        bool undecided = false;
        for (const auto& part : parts)
        {
            const Value v = evaluate(part);
            if (!v)
                undecided = true;
            else if (*v == any)
                return any;
        }
        return undecided ? std::nullopt : Value(!any);
    }

    /// Charges a quantified conjunct against the budget; false (nothing charged) when it does not fit. Conjuncts
    /// inside an expanded one are already covered by its charge.
    bool charge(const loki::Condition& c)
    {
        if (m_depth > 0)
            return true;
        const u64 n = cost(c);
        if (n > m_budget)
            return false;
        m_budget -= n;
        return true;
    }

    /// Adds the literals that the condition (negated unless `positive`) requires; a part that cannot be expanded
    /// clears `m_complete`. Negation is pushed inward: not-or is a conjunction of negations, not-exists a forall of
    /// the negation, not-imply with a static antecedent that holds the negated consequent; not-and and not-forall
    /// cannot be expanded.
    void expand(const loki::Condition& c, bool positive)
    {
        std::visit(
            [&](const auto& x)
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::ConditionLiteral>)
                {
                    const auto& literal = x->get_literal();
                    const auto it = m_pred.find(literal->get_atom()->get_predicate()->get_name());
                    if (it == m_pred.end())
                        m_complete = false;
                    else
                        m_out.push_back({PredicateId{it->second}, literal->get_polarity() == positive, objects_of(literal->get_atom())});
                }
                else if constexpr (std::is_same_v<T, loki::ConditionNot>)
                    expand(x->get_condition(), !positive);
                else if constexpr (std::is_same_v<T, loki::ConditionAnd> || std::is_same_v<T, loki::ConditionOr>)
                {
                    constexpr bool is_and = std::is_same_v<T, loki::ConditionAnd>;
                    if (positive != is_and)  // or, not and
                        m_complete = false;
                    else
                        for (const auto& part : x->get_conditions())
                            expand(part, positive);
                }
                else if constexpr (std::is_same_v<T, loki::ConditionForall> || std::is_same_v<T, loki::ConditionExists>)
                {
                    constexpr bool is_forall = std::is_same_v<T, loki::ConditionForall>;
                    if (positive != is_forall || !charge(c))  // exists, not forall
                    {
                        m_complete = false;
                        return;
                    }
                    auto visit = [&]
                    {
                        expand(x->get_condition(), positive);
                        return true;
                    };
                    ++m_depth;
                    bind_all(x->get_parameters(), 0, visit);
                    --m_depth;
                }
                else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                {
                    if (!charge(c))
                    {
                        m_complete = false;
                        return;
                    }
                    ++m_depth;
                    const Value antecedent = evaluate(x->get_left_condition());
                    if (!antecedent || (!positive && !*antecedent))
                        m_complete = false;
                    else if (*antecedent)
                        expand(x->get_right_condition(), positive);
                    --m_depth;
                }
                else
                    m_complete = false;  // numeric constraints
            },
            c->get_condition());
    }

    const DomainState& m_ds;
    TaskData& m_t;
    absl::flat_hash_map<std::string, u32> m_object, m_pred;
    std::vector<u32> m_type_pred;
    absl::flat_hash_map<u32, absl::flat_hash_set<std::vector<u32>>> m_static;  // static atoms by predicate, on demand
    absl::flat_hash_map<loki::Variable, u32> m_binding;
    absl::flat_hash_map<loki::Parameter, std::vector<u32>> m_domain;
    std::vector<GroundLiteralData> m_out;
    bool m_complete = true;
    u64 m_budget = k_max_bindings;
    u32 m_depth = 0;  // 0 outside every quantified conjunct
};
}  // namespace

void record_encoding_goal(const DomainState& ds, const loki::Problem& parsed, TaskData& t)
{
    Expander(ds, t).run(parsed);
}
}  // namespace mymyr::frontend::detail
