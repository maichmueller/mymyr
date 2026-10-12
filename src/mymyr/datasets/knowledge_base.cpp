#include "mymyr/datasets/knowledge_base.hpp"

#include "mymyr/core/threads.hpp"
#include "mymyr/novelty/novelty_table.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace mymyr::datasets
{
std::shared_ptr<const KnowledgeBase> KnowledgeBase::create(rl::TaskTablePtr tasks, const KnowledgeBaseOptions& options)
{
    if (!tasks)
        throw std::invalid_argument("mymyr: KnowledgeBase: no tasks");
    if (options.tuple_graphs && options.tuple_graphs->width > novelty::k_max_arity)
        throw std::invalid_argument("mymyr: tuple graph width must be in 0.." + std::to_string(novelty::k_max_arity) + ", got " +
                                    std::to_string(options.tuple_graphs->width));
    const u32 threads = resolve_threads(options.threads);
    auto kb = std::make_shared<KnowledgeBase>();
    kb->m_tasks = tasks;
    kb->m_options = options;

    // the state spaces (failures skipped), with their task indices, sorted stably by size
    std::vector<TaskPtr> ts;
    for (u32 i = 0; i < tasks->size(); ++i)
        ts.push_back(tasks->task(i));
    StateSpaceOptions so = options.state_space;
    so.threads = 1;
    std::vector<StateSpaceResult> results = generate_state_spaces(ts, so, threads);
    std::vector<std::pair<StateSpacePtr, u32>> spaces;
    for (u32 i = 0; i < results.size(); ++i)
        if (results[i].space)
            spaces.emplace_back(std::move(results[i].space), i);
    if (options.sort_by_size)
        std::stable_sort(spaces.begin(), spaces.end(),
                         [](const auto& a, const auto& b) { return a.first->num_states() < b.first->num_states(); });
    for (auto& [space, index] : spaces)
    {
        kb->m_spaces.push_back(space);
        kb->m_task_indices.push_back(index);
    }

    if (options.generalized)
        kb->m_gss = GeneralizedStateSpace::create(kb->m_spaces);

    if (options.tuple_graphs)
    {
        TupleGraphOptions to = *options.tuple_graphs;
        to.threads = threads;
        for (const StateSpacePtr& s : kb->m_spaces)
            kb->m_tuple_graphs.push_back(datasets::tuple_graphs(s, to));
    }
    return kb;
}

std::shared_ptr<const KnowledgeBase> KnowledgeBase::create(rl::TaskTablePtr tasks, const KnowledgeBaseOptions& options,
                                                           KnowledgeBaseParts&& parts)
{
    auto need = [](bool ok, const char* what)
    {
        if (!ok)
            throw std::invalid_argument(std::string("mymyr: invalid knowledge base: ") + what);
    };
    need(tasks != nullptr, "no tasks");
    need(!options.tuple_graphs || options.tuple_graphs->width <= novelty::k_max_arity, "tuple graph width above the largest tuple size");
    const usize n = parts.state_spaces.size();
    need(parts.task_indices.size() == n, "the task indices are not one per state space");
    std::vector<char> seen(tasks->size(), 0);
    for (usize i = 0; i < n; ++i)
    {
        const u32 t = parts.task_indices[i];
        need(t < tasks->size() && !seen[t], "a task index is out of range or repeated");
        seen[t] = 1;
        const StateSpacePtr& s = parts.state_spaces[i];
        need(s != nullptr && s->task() == tasks->task(t), "a state space is not of its task");
        if (i > 0)
        {
            const StateSpace& prev = *parts.state_spaces[i - 1];
            const bool in_order = parts.task_indices[i - 1] < t;
            need(options.sort_by_size ? prev.num_states() < s->num_states() || (prev.num_states() == s->num_states() && in_order) : in_order,
                 "the state spaces are not in knowledge-base order");
        }
    }
    if (options.tuple_graphs)
    {
        need(parts.tuple_graphs.size() == n, "the tuple graphs are not one list per state space");
        for (usize i = 0; i < n; ++i)
        {
            const auto& graphs = parts.tuple_graphs[i];
            need(graphs.size() == parts.state_spaces[i]->num_states(), "a state space does not have one tuple graph per state");
            for (u32 v = 0; v < graphs.size(); ++v)
                need(graphs[v].space() == parts.state_spaces[i] && graphs[v].root() == v &&
                         graphs[v].width() == options.tuple_graphs->width &&
                         graphs[v].dominance_pruning() == options.tuple_graphs->dominance_pruning,
                     "a tuple graph is not of its state, or has other options");
        }
    }
    else
        need(parts.tuple_graphs.empty(), "tuple graphs without tuple graph options");

    auto kb = std::make_shared<KnowledgeBase>();
    kb->m_tasks = std::move(tasks);
    kb->m_options = options;
    kb->m_spaces = std::move(parts.state_spaces);
    kb->m_task_indices = std::move(parts.task_indices);
    kb->m_tuple_graphs = std::move(parts.tuple_graphs);
    if (options.generalized)
        kb->m_gss = GeneralizedStateSpace::create(kb->m_spaces);
    return kb;
}
}  // namespace mymyr::datasets
