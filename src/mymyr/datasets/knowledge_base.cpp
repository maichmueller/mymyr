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

    // the generalized state space: its kept spaces become the knowledge base's
    if (options.generalized)
    {
        kb->m_gss = GeneralizedStateSpace::create(kb->m_spaces);
        std::vector<u32> indices;
        for (const StateSpacePtr& s : kb->m_gss->spaces())
        {
            const auto it = std::find_if(spaces.begin(), spaces.end(), [&](const auto& p) { return p.first == s; });
            indices.push_back(it->second);
        }
        kb->m_spaces = kb->m_gss->spaces();
        kb->m_task_indices = std::move(indices);
    }

    if (options.tuple_graphs)
    {
        TupleGraphOptions to = *options.tuple_graphs;
        to.threads = threads;
        for (const StateSpacePtr& s : kb->m_spaces)
            kb->m_tuple_graphs.push_back(datasets::tuple_graphs(s, to));
    }
    return kb;
}
}  // namespace mymyr::datasets
