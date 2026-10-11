// The encoding goal: the problem's goal as written, expanded into signed ground literals (TaskData::encoding_goal).
// The expansion works on loki's parsed problem, before goal normalization turns quantified and disjunctive goals into
// derived predicates.

#include "translate.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace mymyr::frontend::detail
{
namespace
{
using namespace formalism;

/// Expansion stops, and the goal is marked incomplete, after this many quantifier bindings.
constexpr u64 k_max_bindings = u64{1} << 24;

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

    /// The objects a quantified parameter ranges over, in object order: those of any of its types.
    std::vector<u32> domain_of(const loki::Parameter& parameter)
    {
        std::vector<u32> out;
        for (u32 o = 0; o < m_t.objects.size(); ++o)
        {
            bool any = parameter->get_bases().empty();
            for (const auto& type : parameter->get_bases())
                any = any || has_type(o, type);
            if (any)
                out.push_back(o);
        }
        return out;
    }

    /// Calls `body` with the parameters bound to every tuple of objects of their types, in lexicographic order;
    /// false once the binding budget is spent.
    template<class F>
    bool bind_all(const loki::ParameterList& parameters, size_t position, F& body)
    {
        if (position == parameters.size())
        {
            if (m_bindings++ >= k_max_bindings)
                return false;
            body();
            return true;
        }
        const loki::Variable var = parameters[position]->get_variable();
        const auto outer = m_binding.find(var);
        const std::optional<u32> saved = outer == m_binding.end() ? std::nullopt : std::optional<u32>(outer->second);
        bool ok = true;
        for (u32 o : domain_of(parameters[position]))
        {
            m_binding[var] = o;
            if (!(ok = bind_all(parameters, position + 1, body)))
                break;
        }
        if (saved)
            m_binding[var] = *saved;
        else
            m_binding.erase(var);
        return ok;
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
                    };
                    const bool finished = bind_all(x->get_parameters(), 0, visit);
                    if (decided)
                        return any;
                    return finished && !undecided ? Value(!any) : std::nullopt;
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

    /// Adds the literals that the condition (negated unless `positive`) requires; a part that cannot be expanded
    /// clears `m_complete`.
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
                else if constexpr (std::is_same_v<T, loki::ConditionAnd>)
                {
                    if (!positive)
                        m_complete = false;
                    else
                        for (const auto& part : x->get_conditions())
                            expand(part, true);
                }
                else if constexpr (std::is_same_v<T, loki::ConditionForall>)
                {
                    auto visit = [&] { expand(x->get_condition(), true); };
                    if (!positive || !bind_all(x->get_parameters(), 0, visit))
                        m_complete = false;
                }
                else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                {
                    const Value antecedent = positive ? evaluate(x->get_left_condition()) : std::nullopt;
                    if (!antecedent)
                        m_complete = false;
                    else if (*antecedent)
                        expand(x->get_right_condition(), true);
                }
                else
                    m_complete = false;  // or, exists, numeric constraints
            },
            c->get_condition());
    }

    const DomainState& m_ds;
    TaskData& m_t;
    absl::flat_hash_map<std::string, u32> m_object, m_pred;
    std::vector<u32> m_type_pred;
    absl::flat_hash_map<u32, absl::flat_hash_set<std::vector<u32>>> m_static;  // static atoms by predicate, on demand
    absl::flat_hash_map<loki::Variable, u32> m_binding;
    std::vector<GroundLiteralData> m_out;
    bool m_complete = true;
    u64 m_bindings = 0;
};
}  // namespace

void record_encoding_goal(const DomainState& ds, const loki::Problem& parsed, TaskData& t)
{
    Expander(ds, t).run(parsed);
}
}  // namespace mymyr::frontend::detail
