#pragma once
// State space samplers over a StateSpace, matching mimir's sampler semantics:
//   - sample_state: uniform over all states;
//   - sample_state_n_steps_from_goal(n): uniform over the states with unit goal distance n (dead ends never);
//   - sample_dead_end_state: uniform over the unsolvable states;
//   - the counts: states, dead ends, "alive" states (all states that are not dead ends, goal states included), and
//     the largest unit goal distance of a non-dead-end state.
// mimir draws with std::mt19937 and std::uniform_int_distribution, seeded from std::random_device unless set_seed is
// called, so its samples are neither reproducible nor portable. mymyr draws with datasets::Rng, core/random.hpp's
// Xoshiro256 (xoshiro256**, Lemire's bounded integers): the same seed gives the same samples on every platform and
// standard library. Samples are state ids of the space.
//
// A sampler needs the unit goal distances only: it is built from a StateSpace or from the distances of a space held
// elsewhere (the device state spaces of cuda/state_space.hpp download just those).
//
// Thread-safety: a sampler holds its RNG; use one per thread (or copy it). The space is shared and immutable.

#include "mymyr/core/random.hpp"
#include "mymyr/core/types.hpp"
#include "mymyr/datasets/state_space.hpp"

#include <span>
#include <vector>

namespace mymyr::datasets
{
/// The samplers' generator: a Xoshiro256 instance, never a <random> distribution, so every draw is reproducible.
using Rng = Xoshiro256;

class StateSpaceSampler
{
public:
    explicit StateSpaceSampler(StateSpacePtr space, u64 seed = 0);
    /// A sampler over a space given by its unit goal distances (StateSpace::unit_goal_distances(), one per state id).
    explicit StateSpaceSampler(std::span<const i32> unit_goal_distances, u64 seed = 0);

    /// The space (null for a sampler built from distances).
    [[nodiscard]] const StateSpacePtr& space() const noexcept { return m_space; }
    void set_seed(u64 seed) noexcept { m_rng.reseed(seed); }

    [[nodiscard]] u32 sample_state();
    /// Throws std::out_of_range if no state has unit goal distance n.
    [[nodiscard]] u32 sample_state_n_steps_from_goal(i32 n);
    /// Throws std::out_of_range if the space has no dead end.
    [[nodiscard]] u32 sample_dead_end_state();
    /// Bulk versions: out.size() samples each.
    void sample_states(std::span<u32> out);
    void sample_states_n_steps_from_goal(i32 n, std::span<u32> out);
    void sample_dead_end_states(std::span<u32> out);

    [[nodiscard]] u32 num_states() const noexcept { return m_n; }
    [[nodiscard]] u32 num_dead_end_states() const noexcept { return static_cast<u32>(m_dead_ends.size()); }
    [[nodiscard]] u32 num_alive_states() const noexcept { return num_states() - num_dead_end_states(); }
    [[nodiscard]] u32 max_steps_to_goal() const noexcept { return m_max_steps; }
    /// States with unit goal distance n (ascending ids; empty if none).
    [[nodiscard]] std::span<const u32> states_n_steps_from_goal(i32 n) const noexcept;
    [[nodiscard]] std::span<const u32> dead_end_states() const noexcept { return m_dead_ends; }

private:
    void index(std::span<const i32> unit);

    StateSpacePtr m_space;
    Rng m_rng;
    u32 m_n = 0;
    std::vector<u64> m_by_steps_offsets;  // CSR over unit distance 0..max
    std::vector<u32> m_by_steps;
    std::vector<u32> m_dead_ends;
    u32 m_max_steps = 0;
};
}  // namespace mymyr::datasets
