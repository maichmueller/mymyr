#pragma once
// Knowledge bases, as mimir's KnowledgeBaseImpl: what is known about a set of tasks of one domain, for learning across
// instances of different sizes. A knowledge base holds the tasks (a TaskTable), the state space of every task whose
// generation succeeded, optionally the generalized state space over them, and optionally the tuple graphs of every
// vertex of every space.
//
//   auto kb = mymyr::datasets::KnowledgeBase::create(table, {.generalized = true, .tuple_graphs = TupleGraphOptions{.width = 2}});
//   kb->state_spaces(); kb->generalized_state_space(); kb->tuple_graphs()[i][v];
//
// Construction (mimir's KnowledgeBaseImpl::create): the state spaces of the tasks (the instance pool, one task per
// thread), the failed generations skipped; sorted ascending by size if sort_by_size (stable: ties keep the task order;
// mimir's sort_ascending_by_num_states, default on as in mimir); with `generalized`, the generalized state space over
// them, whose kept spaces (with symmetry reduction, problems isomorphic to an earlier one are dropped) become the
// knowledge base's spaces; with `tuple_graphs`, the tuple graphs of every vertex of every kept space (tuple graphs over
// symmetry-reduced spaces use symmetry reduction, as in mimir).

#include "mymyr/core/types.hpp"
#include "mymyr/datasets/generalized_state_space.hpp"
#include "mymyr/datasets/state_space.hpp"
#include "mymyr/datasets/tuple_graph.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mymyr::datasets
{
struct KnowledgeBaseOptions
{
    /// Options of every state space generation (threads is ignored: each task's space is generated on one thread).
    StateSpaceOptions state_space;
    bool sort_by_size = true;                        // order the spaces ascending by their number of states
    bool generalized = false;                        // build the generalized state space
    std::optional<TupleGraphOptions> tuple_graphs;   // build tuple graphs with these options (their threads ignored)
    u32 threads = 0;                                 // worker threads of every step (0: hardware concurrency)
};

class KnowledgeBase
{
public:
    /// Builds the knowledge base of the tasks of `tasks`. Throws std::invalid_argument for a null table or a tuple
    /// graph width above novelty::k_max_arity.
    [[nodiscard]] static std::shared_ptr<const KnowledgeBase> create(rl::TaskTablePtr tasks, const KnowledgeBaseOptions& options = {});

    [[nodiscard]] const rl::TaskTablePtr& tasks() const noexcept { return m_tasks; }
    [[nodiscard]] const KnowledgeBaseOptions& options() const noexcept { return m_options; }
    /// The state spaces, in knowledge-base order (see above).
    [[nodiscard]] const std::vector<StateSpacePtr>& state_spaces() const noexcept { return m_spaces; }
    /// Per state space: the index of its task in tasks().
    [[nodiscard]] std::span<const u32> task_indices() const noexcept { return m_task_indices; }
    /// The generalized state space over state_spaces() (problem p = state space p), or null if not requested.
    [[nodiscard]] const GeneralizedStateSpacePtr& generalized_state_space() const noexcept { return m_gss; }
    /// Whether tuple graphs were built; then tuple_graphs()[i][v] is the tuple graph of vertex v of state space i.
    [[nodiscard]] bool has_tuple_graphs() const noexcept { return m_options.tuple_graphs.has_value(); }
    [[nodiscard]] const std::vector<std::vector<TupleGraph>>& tuple_graphs() const noexcept { return m_tuple_graphs; }

    KnowledgeBase() = default;

private:
    rl::TaskTablePtr m_tasks;
    KnowledgeBaseOptions m_options;
    std::vector<StateSpacePtr> m_spaces;
    std::vector<u32> m_task_indices;
    GeneralizedStateSpacePtr m_gss;
    std::vector<std::vector<TupleGraph>> m_tuple_graphs;
};
using KnowledgeBasePtr = std::shared_ptr<const KnowledgeBase>;
}  // namespace mymyr::datasets
