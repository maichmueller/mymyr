// Landmark count (Richter and Westphal 2010).

#include "landmark_heuristics.hpp"

#include <stdexcept>

namespace mymyr::heuristics::detail
{
std::unique_ptr<Heuristic> make_landmark_count(const Task&, const Options&)
{
    throw std::invalid_argument("mymyr: landmark count is not built yet");
}
}  // namespace mymyr::heuristics::detail
