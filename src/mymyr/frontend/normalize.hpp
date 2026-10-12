#pragma once
// Conditions of the ADL fragment in mymyr's conjunctive form.
//
// loki's translation (loki::translate) compiles every condition into conjunctions of literals and numeric constraints
// plus derived predicates, the recipe of Fast Downward's translator:
//   - negation normal form (`imply` and `not` over compounds are pushed down to literals and comparisons; a negated
//     comparison is the complementary one);
//   - each universal quantifier (forall), also one that negation normal form makes of `not exists`, becomes the
//     negated literal of a derived predicate `axiom_<k>` over the quantifier's free variables, whose rule is the
//     existential quantification of the negated body (so a comparison with an undefined side, false under negation
//     as well, does not falsify a universal quantifier);
//   - disjunctions are distributed to the top and split: an action or a derived predicate rule into one copy per
//     disjunct, a conditional effect into one effect per disjunct; a goal part that is not a literal becomes a nullary
//     derived predicate `axiom_<k>` first;
//   - existential quantifiers move into the parameters of their action or rule (and turn a conditional effect into a
//     `forall` effect).
// The axioms are stratified by the compiled rules (task/compile.cpp), helpers included: a helper of a universal
// quantifier over a derived predicate lies in a stratum above it. A derived predicate whose rule depends on itself
// through a universal quantifier depends on itself through negation after the compilation and is refused as not
// stratifiable.
//
// Three parts complete it:
//   - translate_loki_domain (here): loki keeps no scope for the variables of `forall` effects, so it cannot compile a
//     universal quantifier over such a variable inside an effect condition. The condition of such a conditional
//     effect becomes a derived predicate `axiom_<k>` over its free variables first, whose rule loki compiles like any
//     other rule body;
//   - conjuncts (translate.cpp): loki's problem translation adds the type literals of a generated derived
//     predicate's parameters in its last pass, as a conjunction around the rule body, so a body that is a conjunction
//     arrives nested; mymyr reads nested conjunctions flattened;
//   - effect_conjuncts (translate.cpp): loki merges the conditions of nested conditional effects after it moved the
//     existential quantifiers out of them, so an existential condition can arrive inside the conjunction of a
//     conditional effect; its variables become parameters of the conditional effect.
// A domain without such effect conditions goes to loki as parsed.

#include <loki/details/pddl/domain.hpp>
#include <loki/details/pddl/translator.hpp>

namespace mymyr::frontend::detail
{
/// loki::translate(domain), after lifting the effect conditions loki cannot scope into derived predicates (see
/// above). The result's original domain is `parsed`, so problems parsed against it translate with it.
loki::DomainTranslationResult translate_loki_domain(const loki::Domain& parsed);
}  // namespace mymyr::frontend::detail
