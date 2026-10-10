#pragma once
// Checked size arithmetic for allocations, copies and views: a byte count or element count that does not fit in 64
// bits throws std::length_error naming what was sized, instead of wrapping to a small allocation.

#include "mymyr/core/types.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>

namespace mymyr
{
/// The product of `factors`; std::length_error naming `what` if it exceeds 2^64 - 1.
[[nodiscard]] inline u64 checked_mul(std::initializer_list<u64> factors, const char* what)
{
    u64 r = 1;
    for (const u64 f : factors)
        if (__builtin_mul_overflow(r, f, &r))
            throw std::length_error(std::string("mymyr: ") + what + ": the size exceeds 2^64 - 1 bytes");
    return r;
}

/// a + b; std::length_error naming `what` if it exceeds 2^64 - 1.
[[nodiscard]] inline u64 checked_add(u64 a, u64 b, const char* what)
{
    u64 r = 0;
    if (__builtin_add_overflow(a, b, &r))
        throw std::length_error(std::string("mymyr: ") + what + ": the size exceeds 2^64 - 1 bytes");
    return r;
}
}  // namespace mymyr
