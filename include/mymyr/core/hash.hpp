#pragma once
// Hashing of word spans. The same functions run on the host and the device (the GPU dedup table must agree with
// the CPU store bit for bit), and the state hash covers only the *trimmed* words so that states of different
// stored widths that compare equal (bits::equal) also hash equal.

#include "mymyr/core/bitset.hpp"
#include "mymyr/core/types.hpp"

namespace mymyr::hash
{
/// splitmix64 finalizer: a bijective 64-bit mix.
[[nodiscard]] MYMYR_HD_HOT constexpr u64 mix64(u64 x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/// 64x64 -> 128 multiply folded to 64 bits (the wyhash/rapidhash primitive).
[[nodiscard]] MYMYR_HD_HOT u64 mulfold(u64 a, u64 b)
{
#if defined(__CUDA_ARCH__)
    return (a * b) ^ __umul64hi(a, b);
#else
    const u128 r = static_cast<u128>(a) * b;
    return static_cast<u64>(r) ^ static_cast<u64>(r >> 64);
#endif
}

inline constexpr u64 k_seed = 0x2d358dccaa6c78a5ULL;
inline constexpr u64 k_p1 = 0x8bb84b93962eacc9ULL;
inline constexpr u64 k_p2 = 0x4b33a62ed433d4a3ULL;

/// Hash of n words, sensitive to order and to the (trimmed) length.
[[nodiscard]] MYMYR_HD_HOT u64 words(const u64* w, u32 n, u64 seed = k_seed)
{
    u64 h = seed ^ (static_cast<u64>(n) * k_p1);
    u32 i = 0;
    for (; i + 1 < n; i += 2)
        h = mulfold(h ^ w[i] ^ k_p1, w[i + 1] ^ k_p2);
    if (i < n)
        h = mulfold(h ^ w[i] ^ k_p1, k_p2);
    return mix64(h);
}

/// State hash: words() over the trimmed length (trailing zero words ignored).
[[nodiscard]] MYMYR_HD_HOT u64 state_words(const u64* w, u32 n, u64 seed = k_seed)
{
    return words(w, bits::trimmed_size(w, n), seed);
}

/// Hash of a state row [bits | numeric words]: the trimmed bits seeded by the numeric words. Equal to
/// state_words(w, nw) when there are no numeric words.
[[nodiscard]] MYMYR_HD_HOT u64 state(const u64* w, u32 nw, const u64* num, u32 nnum)
{
    return nnum ? state_words(w, nw, words(num, nnum)) : state_words(w, nw);
}

/// Combine a hash with another value (boost-style, but with a strong mix).
[[nodiscard]] MYMYR_HD constexpr u64 combine(u64 h, u64 v) { return mix64(h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2))); }
}  // namespace mymyr::hash
