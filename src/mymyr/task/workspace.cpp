#include "mymyr/task/workspace.hpp"

#include "../successor/binding_scratch.hpp"

namespace mymyr
{
Workspace::Workspace(const Task& task) : m_engine(task), m_axioms(m_engine), m_successors(m_engine, m_axioms) {}

Workspace::~Workspace() = default;

detail::BindingScratch& Workspace::bindings()
{
    if (!m_bindings)
        m_bindings = std::make_unique<detail::BindingScratch>(m_engine.task());
    return *m_bindings;
}
}  // namespace mymyr
