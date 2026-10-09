// Generalized state spaces (datasets/generalized_state_space.hpp).

#include "mymyr/datasets/generalized_state_space.hpp"

#include <algorithm>
#include <stdexcept>

namespace mymyr::datasets
{
std::vector<StateSpacePtr> ordered_spaces(std::span<const StateSpaceResult> results, bool sort_ascending)
{
    std::vector<StateSpacePtr> spaces;
    for (const StateSpaceResult& r : results)
        if (r.space)
            spaces.push_back(r.space);
    if (sort_ascending)
        std::stable_sort(spaces.begin(), spaces.end(),
                         [](const StateSpacePtr& a, const StateSpacePtr& b) { return a->num_states() < b->num_states(); });
    return spaces;
}

namespace
{
std::vector<u32> flagged(std::span<const u8> flags)
{
    std::vector<u32> out;
    for (usize i = 0; i < flags.size(); ++i)
        if (flags[i])
            out.push_back(static_cast<u32>(i));
    return out;
}
}  // namespace

std::vector<u32> GeneralizedStateSpace::initial_vertices() const { return flagged(m_initial); }
std::vector<u32> GeneralizedStateSpace::goal_vertices() const { return flagged(m_goal); }
std::vector<u32> GeneralizedStateSpace::unsolvable_vertices() const { return flagged(m_unsolvable); }

u32 GeneralizedStateSpace::vertex(u32 problem, u32 state) const
{
    if (problem >= m_spaces.size() || state >= m_spaces[problem]->num_states())
        throw std::out_of_range("mymyr: problem or state index out of range");
    return m_voffsets[problem] + state;
}

u64 GeneralizedStateSpace::edge(u32 problem, u64 edge) const
{
    if (problem >= m_spaces.size() || edge >= m_spaces[problem]->num_transitions())
        throw std::out_of_range("mymyr: problem or transition index out of range");
    return m_eoffsets[problem] + edge;
}

u32 GeneralizedStateSpace::problem_of(u32 vertex) const
{
    if (vertex >= num_vertices())
        throw std::out_of_range("mymyr: vertex index out of range");
    const auto it = std::upper_bound(m_voffsets.begin(), m_voffsets.end(), vertex);
    return static_cast<u32>(it - m_voffsets.begin() - 1);
}

u32 GeneralizedStateSpace::source(u64 edge) const
{
    if (edge >= m_targets.size())
        throw std::out_of_range("mymyr: edge index out of range");
    const auto it = std::upper_bound(m_foffsets.begin(), m_foffsets.end(), edge);
    return static_cast<u32>(it - m_foffsets.begin() - 1);
}

std::shared_ptr<const GeneralizedStateSpace> GeneralizedStateSpace::create(std::vector<StateSpacePtr> spaces)
{
    for (const StateSpacePtr& s : spaces)
        if (!s)
            throw std::invalid_argument("mymyr: a generalized state space needs non-null state spaces");
    for (usize i = 1; i < spaces.size(); ++i)
        if (spaces[i]->task()->data().domain_name != spaces[0]->task()->data().domain_name)
            throw std::invalid_argument("mymyr: the state spaces of a generalized state space must share one domain ('" +
                                        spaces[0]->task()->data().domain_name + "' and '" +
                                        spaces[i]->task()->data().domain_name + "')");
    u64 V = 0, E = 0;
    for (const StateSpacePtr& s : spaces)
        V += s->num_states(), E += s->num_transitions();
    if (V > 0xffffffffULL)
        throw std::length_error("mymyr: a generalized state space supports fewer than 2^32 vertices");
    auto G = std::make_shared<GeneralizedStateSpace>();
    G->m_voffsets.reserve(spaces.size() + 1);
    G->m_eoffsets.reserve(spaces.size() + 1);
    G->m_foffsets.reserve(V + 1);
    G->m_targets.reserve(E);
    G->m_initial.reserve(V);
    G->m_goal.reserve(V);
    G->m_unsolvable.reserve(V);
    for (const StateSpacePtr& sp : spaces)
    {
        const StateSpace& S = *sp;
        const u32 voff = G->m_voffsets.back();
        const u64 eoff = G->m_eoffsets.back();
        const auto off = S.forward_offsets();
        for (u32 v = 0; v < S.num_states(); ++v)
        {
            G->m_foffsets.push_back(eoff + off[v + 1]);
            G->m_initial.push_back(v == S.initial_state());
        }
        for (u32 t : S.forward_targets())
            G->m_targets.push_back(voff + t);
        G->m_goal.insert(G->m_goal.end(), S.goal_flags().begin(), S.goal_flags().end());
        G->m_unsolvable.insert(G->m_unsolvable.end(), S.unsolvable_flags().begin(), S.unsolvable_flags().end());
        G->m_voffsets.push_back(voff + S.num_states());
        G->m_eoffsets.push_back(eoff + S.num_transitions());
    }
    G->m_spaces = std::move(spaces);
    return G;
}
}  // namespace mymyr::datasets
