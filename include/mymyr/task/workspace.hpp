#pragma once
// Workspace: mutable scratch for one task: the matching engine (per-state view, binding array, resolved matchers), the
// axiom evaluator, the successor generator and, made on first use, the scratch of the binding generators
// (successor/bindings.hpp). A task lends them out (Task::workspace(): one per live lease) and frees them with itself;
// users never construct one.

#include "mymyr/axioms/evaluator.hpp"
#include "mymyr/core/memory.hpp"
#include "mymyr/successor/detail/engine.hpp"
#include "mymyr/successor/successors.hpp"

#include <memory>
#include <vector>

namespace mymyr
{
class Task;
namespace detail
{
class BindingScratch;
}

class alignas(k_cache_line) Workspace  // whole cache lines: its scalars are written per state (see LineAllocator)
{
public:
    explicit Workspace(const Task& task);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    [[nodiscard]] detail::Engine& engine() noexcept { return m_engine; }
    [[nodiscard]] AxiomEvaluator& axioms() noexcept { return m_axioms; }
    [[nodiscard]] Successors& successors() noexcept { return m_successors; }
    /// Scratch of the binding generators (successor/bindings.hpp), made on first use.
    [[nodiscard]] detail::BindingScratch& bindings();

private:
    detail::Engine m_engine;
    AxiomEvaluator m_axioms;
    Successors m_successors;
    std::unique_ptr<detail::BindingScratch> m_bindings;
};

namespace detail
{
/// One thread's released workspaces of one task, for its next leases (Task::workspace()).
struct WorkspacePool
{
    std::vector<std::unique_ptr<Workspace>> idle;
};
}  // namespace detail
}  // namespace mymyr
