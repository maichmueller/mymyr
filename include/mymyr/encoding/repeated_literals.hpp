#pragma once

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// Where a literal listed by several goal levels is encoded. Every keeps it at each listing level; Lowest and
/// Highest keep it once, at the lowest or highest listing level respectively. The goal is level 0.
enum class RepeatedLiterals : u8
{
    Every,
    Lowest,
    Highest,
};

inline constexpr RepeatedLiterals k_default_repeated_literals = RepeatedLiterals::Every;
}
