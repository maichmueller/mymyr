// The perfect heuristic (heuristics/perfect.hpp): a goal distance lookup in a state space.

#include "mymyr/heuristics/perfect.hpp"

#include <stdexcept>
#include <utility>

namespace mymyr::heuristics
{
namespace
{
class PerfectHeuristic final : public Heuristic
{
public:
    PerfectHeuristic(datasets::StateSpacePtr space, Costs costs) : m_space(std::move(space)), m_real(costs == Costs::Real) {}

    [[nodiscard]] Kind kind() const noexcept override { return Kind::Perfect; }

    Value evaluate(StateView s) override
    {
        ++m_stats.evaluations;
        const i64 id = m_space->find(s);
        if (id < 0)
            throw std::invalid_argument("mymyr: the perfect heuristic got a state outside its state space");
        const Value v = distance(static_cast<u32>(id));
        m_stats.dead_ends += v == k_dead_end;
        return v;
    }

    Value evaluate(StateView, std::span<const search::GoalSpec::AtomGoal>) override
    {
        throw std::invalid_argument("mymyr: the perfect heuristic knows the distances to the task's goal only (its state space's), "
                                    "not to other goals");
    }

private:
    [[nodiscard]] Value distance(u32 id) const noexcept
    {
        if (m_real)
            return m_space->cost_goal_distances()[id];
        const i32 d = m_space->unit_goal_distances()[id];
        return d == datasets::k_unsolvable_distance ? k_dead_end : static_cast<Value>(d);
    }

    datasets::StateSpacePtr m_space;
    bool m_real;
};
}  // namespace

std::unique_ptr<Heuristic> perfect(datasets::StateSpacePtr space, Costs costs)
{
    if (!space)
        throw std::invalid_argument("mymyr: the perfect heuristic needs a state space (got none)");
    if (space->symmetry_reduced())
        throw std::invalid_argument("mymyr: the perfect heuristic needs a state space without symmetry pruning");
    return std::make_unique<PerfectHeuristic>(std::move(space), costs);
}
}  // namespace mymyr::heuristics
