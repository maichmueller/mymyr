#include "normalize.hpp"

#include <loki/details/pddl/action.hpp>
#include <loki/details/pddl/atom.hpp>
#include <loki/details/pddl/axiom.hpp>
#include <loki/details/pddl/conditions.hpp>
#include <loki/details/pddl/domain_builder.hpp>
#include <loki/details/pddl/effects.hpp>
#include <loki/details/pddl/function.hpp>
#include <loki/details/pddl/function_expressions.hpp>
#include <loki/details/pddl/function_skeleton.hpp>
#include <loki/details/pddl/literal.hpp>
#include <loki/details/pddl/object.hpp>
#include <loki/details/pddl/parameter.hpp>
#include <loki/details/pddl/predicate.hpp>
#include <loki/details/pddl/repositories.hpp>
#include <loki/details/pddl/requirements.hpp>
#include <loki/details/pddl/term.hpp>
#include <loki/details/pddl/type.hpp>
#include <loki/details/pddl/variable.hpp>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace mymyr::frontend::detail
{
namespace
{
using VarList = std::vector<loki::Variable>;

void add_unique(VarList& out, loki::Variable v)
{
    if (std::find(out.begin(), out.end(), v) == out.end())
        out.push_back(v);
}

void term_vars(const loki::TermList& ts, const VarList& bound, VarList& out)
{
    for (const auto& t : ts)
        if (const auto* v = std::get_if<loki::Variable>(&t->get_object_or_variable()))
            if (std::find(bound.begin(), bound.end(), *v) == bound.end())
                add_unique(out, *v);
}

void expression_vars(const loki::FunctionExpression& e, const VarList& bound, VarList& out)
{
    std::visit(
        [&](auto&& x)
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, loki::FunctionExpressionBinaryOperator>)
            {
                expression_vars(x->get_left_function_expression(), bound, out);
                expression_vars(x->get_right_function_expression(), bound, out);
            }
            else if constexpr (std::is_same_v<T, loki::FunctionExpressionMultiOperator>)
            {
                for (const auto& y : x->get_function_expressions())
                    expression_vars(y, bound, out);
            }
            else if constexpr (std::is_same_v<T, loki::FunctionExpressionMinus>)
                expression_vars(x->get_function_expression(), bound, out);
            else if constexpr (std::is_same_v<T, loki::FunctionExpressionFunction>)
                term_vars(x->get_function()->get_terms(), bound, out);
        },
        e->get_function_expression());
}

/// The free variables of a condition, in the order of their first occurrence.
void free_vars(const loki::Condition& c, VarList& bound, VarList& out)
{
    std::visit(
        [&](auto&& x)
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, loki::ConditionLiteral>)
                term_vars(x->get_literal()->get_atom()->get_terms(), bound, out);
            else if constexpr (std::is_same_v<T, loki::ConditionNumericConstraint>)
            {
                expression_vars(x->get_left_function_expression(), bound, out);
                expression_vars(x->get_right_function_expression(), bound, out);
            }
            else if constexpr (std::is_same_v<T, loki::ConditionAnd> || std::is_same_v<T, loki::ConditionOr>)
            {
                for (const auto& part : x->get_conditions())
                    free_vars(part, bound, out);
            }
            else if constexpr (std::is_same_v<T, loki::ConditionNot>)
                free_vars(x->get_condition(), bound, out);
            else if constexpr (std::is_same_v<T, loki::ConditionImply>)
            {
                free_vars(x->get_left_condition(), bound, out);
                free_vars(x->get_right_condition(), bound, out);
            }
            else  // exists, forall
            {
                const size_t n = bound.size();
                for (const auto& p : x->get_parameters())
                    bound.push_back(p->get_variable());
                free_vars(x->get_condition(), bound, out);
                bound.resize(n);
            }
        },
        c->get_condition());
}

VarList free_vars(const loki::Condition& c)
{
    VarList bound, out;
    free_vars(c, bound, out);
    return out;
}

/// Whether the negation normal form of `c` (at `positive` polarity) has a universal quantifier with a free variable in
/// `effect_vars`: loki's RemoveUniversalQuantifiers finds no parameter for such a variable.
bool has_unscoped_universal(const loki::Condition& c, bool positive, const VarList& effect_vars)
{
    return std::visit(
        [&](auto&& x) -> bool
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, loki::ConditionLiteral> || std::is_same_v<T, loki::ConditionNumericConstraint>)
                return false;
            else if constexpr (std::is_same_v<T, loki::ConditionAnd> || std::is_same_v<T, loki::ConditionOr>)
                return std::any_of(x->get_conditions().begin(), x->get_conditions().end(),
                                   [&](const loki::Condition& part) { return has_unscoped_universal(part, positive, effect_vars); });
            else if constexpr (std::is_same_v<T, loki::ConditionNot>)
                return has_unscoped_universal(x->get_condition(), !positive, effect_vars);
            else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                return has_unscoped_universal(x->get_left_condition(), !positive, effect_vars)
                       || has_unscoped_universal(x->get_right_condition(), positive, effect_vars);
            else
            {
                const bool universal = std::is_same_v<T, loki::ConditionForall> == positive;
                if (universal)
                    for (const auto& v : free_vars(c))
                        if (std::find(effect_vars.begin(), effect_vars.end(), v) != effect_vars.end())
                            return true;
                return has_unscoped_universal(x->get_condition(), positive, effect_vars);
            }
        },
        c->get_condition());
}

/// Whether some conditional effect of `e` needs lifting: inside a forall effect, its condition has a universal
/// quantifier over a variable of an enclosing forall effect.
bool needs_lifting(const loki::Effect& e, VarList& effect_vars)
{
    return std::visit(
        [&](auto&& x) -> bool
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, loki::EffectAnd> || std::is_same_v<T, loki::EffectCompositeOneof>)
                return std::any_of(x->get_effects().begin(), x->get_effects().end(),
                                   [&](const loki::Effect& part) { return needs_lifting(part, effect_vars); });
            else if constexpr (std::is_same_v<T, loki::EffectCompositeProbabilistic>)
                return std::any_of(x->get_effect_distribution().begin(), x->get_effect_distribution().end(),
                                   [&](const auto& part) { return needs_lifting(part.second, effect_vars); });
            else if constexpr (std::is_same_v<T, loki::EffectCompositeForall>)
            {
                const size_t n = effect_vars.size();
                for (const auto& p : x->get_parameters())
                    effect_vars.push_back(p->get_variable());
                const bool out = needs_lifting(x->get_effect(), effect_vars);
                effect_vars.resize(n);
                return out;
            }
            else if constexpr (std::is_same_v<T, loki::EffectCompositeWhen>)
                return (!effect_vars.empty() && has_unscoped_universal(x->get_condition(), true, effect_vars))
                       || needs_lifting(x->get_effect(), effect_vars);
            else
                return false;
        },
        e->get_effect());
}

/// Copies a domain into the repositories of a DomainBuilder, lifting the conditions of the conditional effects that
/// need it (needs_lifting) into derived predicates.
class Lifter
{
public:
    Lifter(loki::DomainBuilder& b, std::unordered_set<std::string> names) : m_b(b), r(b.get_repositories()), m_names(std::move(names)) {}

    loki::Type type(const loki::Type& t) { return r.get_or_create_type(t->get_name(), types(t->get_bases())); }
    loki::TypeList types(const loki::TypeList& ts)
    {
        loki::TypeList out;
        for (const auto& t : ts)
            out.push_back(type(t));
        return out;
    }
    loki::Object object(const loki::Object& o) { return r.get_or_create_object(o->get_name(), types(o->get_bases())); }
    loki::Variable var(const loki::Variable& v) { return r.get_or_create_variable(v->get_name()); }
    loki::Term term(const loki::Term& t)
    {
        if (const auto* o = std::get_if<loki::Object>(&t->get_object_or_variable()))
            return r.get_or_create_term(object(*o));
        return r.get_or_create_term(var(std::get<loki::Variable>(t->get_object_or_variable())));
    }
    loki::TermList terms(const loki::TermList& ts)
    {
        loki::TermList out;
        for (const auto& t : ts)
            out.push_back(term(t));
        return out;
    }
    loki::Parameter param(const loki::Parameter& p) { return r.get_or_create_parameter(var(p->get_variable()), types(p->get_bases())); }
    loki::ParameterList params(const loki::ParameterList& ps)
    {
        loki::ParameterList out;
        for (const auto& p : ps)
            out.push_back(param(p));
        return out;
    }
    loki::Predicate predicate(const loki::Predicate& p) { return r.get_or_create_predicate(p->get_name(), params(p->get_parameters())); }
    loki::Literal literal(const loki::Literal& l)
    {
        const auto& a = l->get_atom();
        return r.get_or_create_literal(l->get_polarity(), r.get_or_create_atom(predicate(a->get_predicate()), terms(a->get_terms())));
    }
    loki::FunctionSkeleton skeleton(const loki::FunctionSkeleton& f)
    {
        return r.get_or_create_function_skeleton(f->get_name(), params(f->get_parameters()), type(f->get_type()));
    }
    loki::Function function(const loki::Function& f) { return r.get_or_create_function(skeleton(f->get_function_skeleton()), terms(f->get_terms())); }

    loki::FunctionExpression expression(const loki::FunctionExpression& e)
    {
        return std::visit(
            [&](auto&& x) -> loki::FunctionExpression
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::FunctionExpressionNumber>)
                    return r.get_or_create_function_expression(r.get_or_create_function_expression_number(x->get_number()));
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionBinaryOperator>)
                {
                    const auto a = expression(x->get_left_function_expression());
                    const auto b = expression(x->get_right_function_expression());
                    return r.get_or_create_function_expression(r.get_or_create_function_expression_binary_operator(x->get_binary_operator(), a, b));
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMultiOperator>)
                {
                    loki::FunctionExpressionList xs;
                    for (const auto& y : x->get_function_expressions())
                        xs.push_back(expression(y));
                    return r.get_or_create_function_expression(r.get_or_create_function_expression_multi_operator(x->get_multi_operator(), xs));
                }
                else if constexpr (std::is_same_v<T, loki::FunctionExpressionMinus>)
                    return r.get_or_create_function_expression(r.get_or_create_function_expression_minus(expression(x->get_function_expression())));
                else
                    return r.get_or_create_function_expression(r.get_or_create_function_expression_function(function(x->get_function())));
            },
            e->get_function_expression());
    }

    loki::Condition condition(const loki::Condition& c)
    {
        return std::visit(
            [&](auto&& x) -> loki::Condition
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::ConditionLiteral>)
                    return r.get_or_create_condition(r.get_or_create_condition_literal(literal(x->get_literal())));
                else if constexpr (std::is_same_v<T, loki::ConditionAnd>)
                    return r.get_or_create_condition(r.get_or_create_condition_and(conditions(x->get_conditions())));
                else if constexpr (std::is_same_v<T, loki::ConditionOr>)
                    return r.get_or_create_condition(r.get_or_create_condition_or(conditions(x->get_conditions())));
                else if constexpr (std::is_same_v<T, loki::ConditionNot>)
                    return r.get_or_create_condition(r.get_or_create_condition_not(condition(x->get_condition())));
                else if constexpr (std::is_same_v<T, loki::ConditionImply>)
                {
                    const auto a = condition(x->get_left_condition());
                    const auto b = condition(x->get_right_condition());
                    return r.get_or_create_condition(r.get_or_create_condition_imply(a, b));
                }
                else if constexpr (std::is_same_v<T, loki::ConditionExists>)
                {
                    const auto ps = params(x->get_parameters());
                    return r.get_or_create_condition(r.get_or_create_condition_exists(ps, condition(x->get_condition())));
                }
                else if constexpr (std::is_same_v<T, loki::ConditionForall>)
                {
                    const auto ps = params(x->get_parameters());
                    return r.get_or_create_condition(r.get_or_create_condition_forall(ps, condition(x->get_condition())));
                }
                else
                {
                    const auto a = expression(x->get_left_function_expression());
                    const auto b = expression(x->get_right_function_expression());
                    return r.get_or_create_condition(r.get_or_create_condition_numeric_constraint(x->get_binary_comparator(), a, b));
                }
            },
            c->get_condition());
    }
    loki::ConditionList conditions(const loki::ConditionList& cs)
    {
        loki::ConditionList out;
        for (const auto& c : cs)
            out.push_back(condition(c));
        return out;
    }

    /// The effect, with the conditions that need lifting replaced by helper literals. `scope`: the parameters of the
    /// action and of the enclosing forall effects, innermost last.
    loki::Effect effect(const loki::Effect& e, loki::ParameterList& scope, size_t num_action_params)
    {
        return std::visit(
            [&](auto&& x) -> loki::Effect
            {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, loki::EffectLiteral>)
                    return r.get_or_create_effect(r.get_or_create_effect_literal(literal(x->get_literal())));
                else if constexpr (std::is_same_v<T, loki::EffectAnd> || std::is_same_v<T, loki::EffectCompositeOneof>)
                {
                    loki::EffectList parts;
                    for (const auto& part : x->get_effects())
                        parts.push_back(effect(part, scope, num_action_params));
                    if constexpr (std::is_same_v<T, loki::EffectAnd>)
                        return r.get_or_create_effect(r.get_or_create_effect_and(parts));
                    else
                        return r.get_or_create_effect(r.get_or_create_effect_composite_oneof(parts));
                }
                else if constexpr (std::is_same_v<T, loki::EffectNumeric>)
                {
                    const auto f = function(x->get_function());
                    return r.get_or_create_effect(r.get_or_create_effect_numeric(x->get_assign_operator(), f, expression(x->get_function_expression())));
                }
                else if constexpr (std::is_same_v<T, loki::EffectCompositeForall>)
                {
                    const auto ps = params(x->get_parameters());
                    const size_t n = scope.size();
                    scope.insert(scope.end(), x->get_parameters().begin(), x->get_parameters().end());
                    const auto inner = effect(x->get_effect(), scope, num_action_params);
                    scope.resize(n);
                    return r.get_or_create_effect(r.get_or_create_effect_composite_forall(ps, inner));
                }
                else if constexpr (std::is_same_v<T, loki::EffectCompositeWhen>)
                {
                    VarList effect_vars;
                    for (size_t i = num_action_params; i < scope.size(); ++i)
                        effect_vars.push_back(scope[i]->get_variable());
                    const loki::Condition c = !effect_vars.empty() && has_unscoped_universal(x->get_condition(), true, effect_vars)
                                                  ? lift(x->get_condition(), scope)
                                                  : condition(x->get_condition());
                    return r.get_or_create_effect(r.get_or_create_effect_composite_when(c, effect(x->get_effect(), scope, num_action_params)));
                }
                else
                {
                    loki::EffectDistribution parts;
                    for (const auto& [p, part] : x->get_effect_distribution())
                        parts.emplace_back(p, effect(part, scope, num_action_params));
                    return r.get_or_create_effect(r.get_or_create_effect_composite_probabilistic(parts));
                }
            },
            e->get_effect());
    }

    /// A derived predicate `axiom_<k>` over the free variables of `c` (in the order of their first occurrence, typed
    /// as the parameter of `scope` that binds them, the innermost first) whose rule is `c`; returns its literal.
    loki::Condition lift(const loki::Condition& c, const loki::ParameterList& scope)
    {
        loki::ParameterList ps;
        loki::TermList ts;
        for (const auto& v : free_vars(c))
        {
            auto it = std::find_if(scope.rbegin(), scope.rend(), [&](const loki::Parameter& p) { return p->get_variable() == v; });
            const loki::TypeList bases = it != scope.rend() ? types((*it)->get_bases()) : loki::TypeList{};
            ps.push_back(r.get_or_create_parameter(var(v), bases));
            ts.push_back(r.get_or_create_term(var(v)));
        }
        std::string name;
        do
            name = "axiom_" + std::to_string(m_next++);
        while (m_names.contains(name));
        m_names.insert(name);
        const auto pred = r.get_or_create_predicate(name, ps);
        const auto head = r.get_or_create_literal(true, r.get_or_create_atom(pred, ts));
        m_b.get_predicates().push_back(pred);
        m_b.get_axioms().push_back(r.get_or_create_axiom(ps, head, condition(c)));
        return r.get_or_create_condition(r.get_or_create_condition_literal(head));
    }

    loki::Action action(const loki::Action& a)
    {
        const auto ps = params(a->get_parameters());
        std::optional<loki::Condition> c;
        if (a->get_condition())
            c = condition(*a->get_condition());
        std::optional<loki::Effect> e;
        if (a->get_effect())
        {
            loki::ParameterList scope = a->get_parameters();
            e = effect(*a->get_effect(), scope, scope.size());
        }
        return r.get_or_create_action(a->get_name(), a->get_original_arity(), ps, c, e);
    }

    loki::Axiom axiom(const loki::Axiom& x)
    {
        const auto ps = params(x->get_parameters());
        const auto head = literal(x->get_literal());
        return r.get_or_create_axiom(ps, head, condition(x->get_condition()));
    }

private:
    loki::DomainBuilder& m_b;
    loki::Repositories& r;
    std::unordered_set<std::string> m_names;  // predicate names in use
    size_t m_next = 0;                        // as loki's create_unique_axiom_name
};
}  // namespace

loki::DomainTranslationResult translate_loki_domain(const loki::Domain& parsed)
{
    const bool lifting = std::any_of(parsed->get_actions().begin(), parsed->get_actions().end(),
                                     [](const loki::Action& a)
                                     {
                                         VarList effect_vars;
                                         return a->get_effect() && needs_lifting(*a->get_effect(), effect_vars);
                                     });
    if (!lifting)
        return loki::translate(parsed);

    // the domain in a builder's repositories: the parsed one's in their order (each list is sorted by index), then the
    // helper predicates and their rules in the order of the effects
    loki::DomainBuilder b;
    std::unordered_set<std::string> names;
    for (const auto& p : parsed->get_predicates())
        names.insert(p->get_name());
    Lifter copy(b, std::move(names));
    auto& r = b.get_repositories();
    b.get_name() = parsed->get_name();
    b.get_filepath() = parsed->get_filepath();
    loki::RequirementEnumSet requirements = parsed->get_requirements()->get_requirements();
    requirements.insert(loki::RequirementEnum::DERIVED_PREDICATES);
    b.get_requirements() = r.get_or_create_requirements(std::move(requirements));
    for (const auto& t : parsed->get_types())
        b.get_types().push_back(copy.type(t));
    for (const auto& o : parsed->get_constants())
        b.get_constants().push_back(copy.object(o));
    for (const auto& p : parsed->get_predicates())
        b.get_predicates().push_back(copy.predicate(p));
    for (const auto& f : parsed->get_function_skeletons())
        b.get_function_skeletons().push_back(copy.skeleton(f));
    for (const auto& l : parsed->get_static_initial_literals())
        b.get_static_initial_literals().push_back(copy.literal(l));
    for (const auto& x : parsed->get_axioms())
        b.get_axioms().push_back(copy.axiom(x));
    for (const auto& a : parsed->get_actions())
        b.get_actions().push_back(copy.action(a));
    const loki::Domain lifted = b.get_result();
    return loki::DomainTranslationResult(parsed, loki::translate(lifted).get_translated_domain());
}
}  // namespace mymyr::frontend::detail
