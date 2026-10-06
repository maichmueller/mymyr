#pragma once
// The counter-based RNG of the RL module: Philox-4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as
// easy as 1, 2, 3", SC 2011), the generator
// of cuRAND's Philox and of torch's CUDA generator. Every draw is a pure function of (key, counter), so a result does
// not depend on the batch size, the batch composition, the launch configuration, the thread or the device:
//
//   key     = the seed (two 32-bit words, low first);
//   counter = (draw lo, draw hi, env, purpose): env is the environment's id (EnvBatch::first_env + row), draw its
//             running draw counter (EnvBatch::draws; one per step), purpose names what the draw decides.
//
// Host and device run the same code (MYMYR_HD; a POD header of the device-code subset): the host reference
// is the device result. No <random> engine or distribution is involved. Known-answer tests against Random123's
// vectors and cuRAND's curand_Philox4x32_10 are in tests/cpp/rl/test_env.cpp and tests/cuda/test_device_env.cpp.
//
// A uniform index below n is the high word of the 64-bit draw times n (Lemire's multiply-shift without the rejection
// step): P(j) differs from 1/n by less than n / 2^64, and the choice needs no loop.

#include "mymyr/core/types.hpp"

namespace mymyr::rl::rng
{
/// What a draw decides (the counter's last word).
enum Purpose : u32
{
    k_successor = 0,  // the successor a random-policy step takes
    k_her = 1,        // the source step and the goal atoms of a hindsight relabel (rl/her.hpp)
};

/// The four 32-bit outputs of one Philox block.
struct Block4
{
    u32 v[4];
};

namespace detail
{
inline constexpr u32 k_m0 = 0xD2511F53u, k_m1 = 0xCD9E8D57u;  // round multipliers
inline constexpr u32 k_w0 = 0x9E3779B9u, k_w1 = 0xBB67AE85u;  // Weyl key increments

MYMYR_HD u32 mulhi32(u32 a, u32 b) { return static_cast<u32>((static_cast<u64>(a) * b) >> 32); }
}  // namespace detail

/// Philox-4x32 with 10 rounds: the block of counter `c` under key `k`.
MYMYR_HD Block4 philox4x32_10(Block4 c, u32 k0, u32 k1)
{
    for (int r = 0; r < 10; ++r)
    {
        const u32 hi0 = detail::mulhi32(detail::k_m0, c.v[0]), lo0 = detail::k_m0 * c.v[0];
        const u32 hi1 = detail::mulhi32(detail::k_m1, c.v[2]), lo1 = detail::k_m1 * c.v[2];
        c = Block4{{hi1 ^ c.v[1] ^ k0, lo1, hi0 ^ c.v[3] ^ k1, lo0}};
        k0 += detail::k_w0;
        k1 += detail::k_w1;
    }
    return c;
}

/// The Philox block of (seed; draw, env, purpose).
MYMYR_HD Block4 block(u64 seed, u64 env, u64 draw, u32 purpose)
{
    return philox4x32_10(Block4{{static_cast<u32>(draw), static_cast<u32>(draw >> 32), static_cast<u32>(env), purpose}},
                         static_cast<u32>(seed), static_cast<u32>(seed >> 32));
}

/// 64 random bits of (seed; draw, env, purpose): the block's first two words, low first.
MYMYR_HD u64 bits64(u64 seed, u64 env, u64 draw, u32 purpose)
{
    const Block4 b = block(seed, env, draw, purpose);
    return static_cast<u64>(b.v[0]) | (static_cast<u64>(b.v[1]) << 32);
}

/// high64(x * n): an index in [0, n) for n >= 1 (32 x 32-bit partial products, identical on host and device).
MYMYR_HD u32 below(u64 x, u32 n)
{
    const u64 hi = (x >> 32) * n, lo = (x & 0xFFFFFFFFull) * n;
    return static_cast<u32>((hi + (lo >> 32)) >> 32);
}

/// The successor a random policy takes in environment `env` at its draw `draw`, among `count` >= 1 successors in
/// canonical order.
MYMYR_HD u32 successor_index(u64 seed, u64 env, u64 draw, u32 count)
{
    return below(bits64(seed, env, draw, k_successor), count);
}
}  // namespace mymyr::rl::rng
