#pragma once
// Stratified axiom evaluation.
//
// Derived atoms have their own slot space and are never stored in states: for each state the evaluator clears the
// engine's derived bitset and closes it under the axioms, stratum by stratum. New derived atoms are pushed into the
// view immediately, so a non-recursive stratum needs one round; a recursive one runs naive rounds to the fixpoint.
// Evaluation always starts from scratch; it does not reuse a parent state's derived atoms.

#include "mymyr/successor/detail/engine.hpp"
#include "mymyr/task/plan.hpp"

#include <vector>

namespace mymyr
{
class AxiomEvaluator
{
public:
    explicit AxiomEvaluator(detail::Engine& engine);
    AxiomEvaluator(const AxiomEvaluator&) = delete;
    AxiomEvaluator& operator=(const AxiomEvaluator&) = delete;

    [[nodiscard]] bool empty() const noexcept { return m_strata.empty(); }

    /// Closes the engine's current state under the axioms. The engine's view must already hold the state's fluent
    /// atoms (Engine::build_view); afterwards it also holds the derived atoms, and Engine::derived() their bitset.
    void evaluate();

    /// Rounds run so far (diagnostics).
    [[nodiscard]] u64 rounds() const noexcept { return m_rounds; }

private:
    struct Exec
    {
        const plan::Axiom* axiom;
        detail::ExecMatcher body;
    };
    struct Stratum
    {
        std::vector<Exec> axioms;
        bool recursive;
    };

    detail::Engine& m_engine;
    std::vector<Stratum> m_strata;
    u64 m_rounds = 0;
};
}  // namespace mymyr
