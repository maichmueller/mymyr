#pragma once
// The portable RNG of core/random.hpp (SplitMix64, Lemire's bounded draw, Fisher-Yates from the back) as POD
// host/device functions for the device rollouts (cuda/rollouts.hpp). Bit for bit the same stream as
// mymyr::SplitMix64 (tests/cuda/test_device_iw.cpp pins it against core/random.hpp), so a device rollout seeded like a CPU
// rollout of search::find_rollouts_parallel shuffles its layers exactly as the CPU does.
//
// SplitMix64 is counter-based: the k-th output of a stream started at state s is mix64(s + k * gamma), so a draw
// depends only on (seed, position in the rollout's own stream), never on the launch configuration or on other
// rollouts. Each rollout owns one stream (its seed), consumed in the CPU's order: one shuffle per completed layer,
// across the passes of its ladder.
//
// Device-code subset: included by .cu files compiled by nvcc as C++20.

#include "mymyr/core/types.hpp"

namespace mymyr::cuda::device_rng
{
inline constexpr u64 k_gamma = 0x9e3779b97f4a7c15ULL;

/// The splitmix64 finalizer (hash::mix64).
[[nodiscard]] MYMYR_HD u64 mix64(u64 x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/// SplitMix64::next(): advances `state` and returns its output.
[[nodiscard]] MYMYR_HD u64 next(u64& state)
{
    state += k_gamma;
    return mix64(state);
}

[[nodiscard]] MYMYR_HD u64 mul_hi(u64 a, u64 b)
{
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#else
    return static_cast<u64>((static_cast<u128>(a) * b) >> 64);
#endif
}

/// SplitMix64::bounded(n): a uniform value in [0, n), n > 0 (Lemire's multiply-shift with rejection).
[[nodiscard]] MYMYR_HD u64 bounded(u64& state, u64 n)
{
    u64 x = next(state);
    u64 low = x * n;
    if (low < n)
    {
        const u64 threshold = (0 - n) % n;  // 2^64 mod n
        while (low < threshold)
        {
            x = next(state);
            low = x * n;
        }
    }
    return mul_hi(x, n);
}

/// SplitMix64::shuffle: Fisher-Yates from the back (for i = n..2: swap a[i - 1] with a[bounded(i)]).
MYMYR_HD void shuffle(u64& state, u32* a, u64 n)
{
    for (u64 i = n; i > 1; --i)
    {
        const u64 j = bounded(state, i);
        const u32 t = a[i - 1];
        a[i - 1] = a[j];
        a[j] = t;
    }
}
}  // namespace mymyr::cuda::device_rng
