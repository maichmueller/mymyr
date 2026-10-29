#pragma once

#include "mymyr/core/types.hpp"

namespace mymyr::rl
{
/// Both masks must leave every bit past the instance's fluent slots zero, including row padding.
[[nodiscard]] MYMYR_HD bool goal_masks_fit(const u64* pos, const u64* neg, u32 words, u32 slots)
{
    u32 w = slots >> 6;
    if ((slots & 63) && w < words)
    {
        if ((pos[w] | neg[w]) >> (slots & 63))
            return false;
        ++w;
    }
    for (; w < words; ++w)
        if (pos[w] | neg[w])
            return false;
    return true;
}
}  // namespace mymyr::rl
