#pragma once
// Internal: LM-cut (lm_cut.cpp) and landmark count (landmark_count.cpp); make_heuristic builds them for Kind::LmCut and
// Kind::LmCount.

#include "mymyr/heuristics/heuristic.hpp"

#include <memory>

namespace mymyr::heuristics::detail
{
[[nodiscard]] std::unique_ptr<Heuristic> make_lm_cut(const Task& task, const Options& options);
[[nodiscard]] std::unique_ptr<Heuristic> make_landmark_count(const Task& task, const Options& options);
}  // namespace mymyr::heuristics::detail
