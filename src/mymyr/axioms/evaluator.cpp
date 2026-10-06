#include "mymyr/axioms/evaluator.hpp"

namespace mymyr
{
AxiomEvaluator::AxiomEvaluator(detail::Engine& engine) : m_engine(engine)
{
    const plan::Compiled& C = engine.compiled();
    m_strata.resize(C.strata.size());
    for (usize si = 0; si < C.strata.size(); ++si)
    {
        const plan::Stratum& st = C.strata[si];
        m_strata[si].recursive = st.recursive;
        m_strata[si].axioms.resize(st.axioms.size());
        for (usize i = 0; i < st.axioms.size(); ++i)
        {
            Exec& e = m_strata[si].axioms[i];
            e.axiom = &st.axioms[i];
            engine.instantiate(e.body, st.axioms[i].body);
        }
    }
}

void AxiomEvaluator::evaluate()
{
    detail::Engine& E = m_engine;
    E.clear_derived();
    for (Stratum& st : m_strata)
        for (bool changed = true; changed;)
        {
            // New derived atoms are pushed into the view incrementally; non-recursive strata need one round.
            changed = false;
            ++m_rounds;
            for (Exec& x : st.axioms)
            {
                const plan::Pattern& head = x.axiom->head;
                auto emit = [&]
                {
                    if (E.add_derived(E.intern(head)))
                        changed = true;
                    return true;
                };
                E.run<false>(x.body, emit);
            }
            if (!st.recursive)
                break;
        }
}
}  // namespace mymyr
