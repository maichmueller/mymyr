#pragma once
// Workspace: one thread's mutable scratch for one task: the matching engine (per-state view, binding array, resolved
// matchers), the axiom evaluator and the successor generator. A task
// creates one per thread on first use (Task::workspace()) and frees them with itself; users never construct one.

#include "mymyr/axioms/evaluator.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/successor/detail/engine.hpp"
#include "mymyr/successor/successors.hpp"

namespace mymyr
{
class Task;

class alignas(k_cache_line) Workspace  // whole cache lines: its scalars are written per state (see LineAllocator)
{
public:
    explicit Workspace(const Task& task) : m_engine(task), m_axioms(m_engine), m_successors(m_engine, m_axioms) {}
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    [[nodiscard]] detail::Engine& engine() noexcept { return m_engine; }
    [[nodiscard]] AxiomEvaluator& axioms() noexcept { return m_axioms; }
    [[nodiscard]] Successors& successors() noexcept { return m_successors; }

private:
    detail::Engine m_engine;
    AxiomEvaluator m_axioms;
    Successors m_successors;
};
}  // namespace mymyr
