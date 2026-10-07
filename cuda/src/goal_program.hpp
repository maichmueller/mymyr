#pragma once

#include "mymyr/cuda/goal_kernels.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/search/control.hpp"
#include "mymyr/task/task.hpp"

#include <span>

namespace mymyr::cuda::detail
{
class GoalPrograms
{
public:
    GoalPrograms(const ContextPtr& ctx, const Task& task, std::span<const search::GoalSpec::AtomGoal> goals,
                 cudaStream_t stream);
    [[nodiscard]] const goal::View& view() const noexcept { return m_view; }

private:
    goal::View m_view;
    DeviceBuffer m_positive, m_negative, m_derived_positive, m_derived_negative, m_offsets, m_code, m_consts, m_checks;
};
}  // namespace mymyr::cuda::detail
