#pragma once
// Internal: h² over the grounded relaxation (h2.cpp); make_heuristic builds it for Kind::H2.

#include "mymyr/heuristics/heuristic.hpp"

#include <memory>

namespace mymyr::heuristics::detail
{
/// The most propositions h² accepts: its pair table has P(P+1)/2 entries per heuristic object.
inline constexpr u32 k_h2_max_props = 8191;

[[nodiscard]] std::unique_ptr<Heuristic> make_h2(const Task& task, const Options& options);
}  // namespace mymyr::heuristics::detail
