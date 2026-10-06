// State space samplers (datasets/sampler.hpp).

#include "mymyr/datasets/sampler.hpp"

#include <stdexcept>

namespace mymyr::datasets
{
StateSpaceSampler::StateSpaceSampler(StateSpacePtr space, u64 seed) : m_space(std::move(space)), m_rng(seed)
{
    if (!m_space)
        throw std::invalid_argument("mymyr: StateSpaceSampler needs a state space");
    index(m_space->unit_goal_distances());
}

StateSpaceSampler::StateSpaceSampler(std::span<const i32> unit_goal_distances, u64 seed) : m_rng(seed)
{
    for (const i32 d : unit_goal_distances)
        if (d < k_unsolvable_distance)
            throw std::invalid_argument("mymyr: StateSpaceSampler: a unit goal distance below -1");
    index(unit_goal_distances);
}

void StateSpaceSampler::index(std::span<const i32> unit)
{
    if (unit.size() > 0xFFFFFFFFull)
        throw std::length_error("mymyr: StateSpaceSampler: more than 2^32 - 1 states");
    const auto N = static_cast<u32>(unit.size());
    m_n = N;
    i32 max = -1;
    for (u32 s = 0; s < N; ++s)
        if (unit[s] != k_unsolvable_distance)
            max = std::max(max, unit[s]);
    m_max_steps = max < 0 ? 0 : static_cast<u32>(max);
    m_by_steps_offsets.assign(static_cast<usize>(max + 2), 0);
    for (u32 s = 0; s < N; ++s)
    {
        if (unit[s] == k_unsolvable_distance)
            m_dead_ends.push_back(s);
        else
            ++m_by_steps_offsets[static_cast<usize>(unit[s]) + 1];
    }
    for (usize i = 1; i < m_by_steps_offsets.size(); ++i)
        m_by_steps_offsets[i] += m_by_steps_offsets[i - 1];
    m_by_steps.resize(N - m_dead_ends.size());
    std::vector<u64> fill(m_by_steps_offsets.begin(), m_by_steps_offsets.end());
    for (u32 s = 0; s < N; ++s)
        if (unit[s] != k_unsolvable_distance)
            m_by_steps[fill[static_cast<usize>(unit[s])]++] = s;
}

std::span<const u32> StateSpaceSampler::states_n_steps_from_goal(i32 n) const noexcept
{
    if (n < 0 || static_cast<usize>(n) + 1 >= m_by_steps_offsets.size())
        return {};
    const u64 a = m_by_steps_offsets[static_cast<usize>(n)], b = m_by_steps_offsets[static_cast<usize>(n) + 1];
    return {m_by_steps.data() + a, static_cast<usize>(b - a)};
}

u32 StateSpaceSampler::sample_state()
{
    if (num_states() == 0)
        throw std::out_of_range("mymyr: cannot sample a state from an empty state space");
    return static_cast<u32>(m_rng.bounded(num_states()));
}

u32 StateSpaceSampler::sample_state_n_steps_from_goal(i32 n)
{
    const std::span<const u32> s = states_n_steps_from_goal(n);
    if (s.empty())
        throw std::out_of_range("mymyr: no state has the requested number of steps to a goal");
    return s[m_rng.bounded(s.size())];
}

u32 StateSpaceSampler::sample_dead_end_state()
{
    if (m_dead_ends.empty())
        throw std::out_of_range("mymyr: the state space has no dead-end states");
    return m_dead_ends[m_rng.bounded(m_dead_ends.size())];
}

void StateSpaceSampler::sample_states(std::span<u32> out)
{
    for (u32& x : out)
        x = sample_state();
}

void StateSpaceSampler::sample_states_n_steps_from_goal(i32 n, std::span<u32> out)
{
    for (u32& x : out)
        x = sample_state_n_steps_from_goal(n);
}

void StateSpaceSampler::sample_dead_end_states(std::span<u32> out)
{
    for (u32& x : out)
        x = sample_dead_end_state();
}
}  // namespace mymyr::datasets
