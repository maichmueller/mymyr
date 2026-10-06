#pragma once
// Portable random number generators for everything that must be reproducible across platforms, standard libraries and
// thread counts: randomized rollouts, parallel IW rollouts and randomized layer orders (SplitMix64), and the dataset
// samplers (Xoshiro256).
//
// std::mt19937_64 is portable, but std::uniform_int_distribution and std::shuffle are not: their algorithms are
// implementation-defined, so libstdc++, libc++ and MSVC produce different permutations from the same seed. Everything
// here is specified bit for bit:
//   - SplitMix64 (Steele, Lea, Flood 2014): state += 0x9e3779b97f4a7c15, output = mix64(state) (the splitmix64
//     finalizer, hash::mix64);
//   - Xoshiro256 (Blackman, Vigna 2018): xoshiro256** over four words seeded with the first four outputs of
//     SplitMix64(seed); output = rotl(s1 * 5, 7) * 9;
//   - bounded(n), for both: Lemire's multiply-shift with rejection ("Fast random integer generation in an interval",
//     2019): draw x, m = x * n (128-bit); reject while low64(m) < (2^64 - n) mod n; return high64(m). Unbiased;
//   - SplitMix64::shuffle: Fisher-Yates from the back, for i = n-1 down to 1 swap a[i] with a[bounded(i + 1)];
//   - Xoshiro256::uniform: the top 53 bits of a draw times 2^-53, in [0, 1).
// mimir's parity tool re-implements SplitMix64's three functions, so its randomized runs replay mymyr's.

#include "mymyr/core/hash.hpp"
#include "mymyr/core/types.hpp"

#include <span>
#include <utility>

namespace mymyr
{
/// Lemire's unbiased draw in [0, n) from the 64-bit outputs of `g` (SplitMix64, Xoshiro256); n must be positive.
template<class Generator>
[[nodiscard]] u64 bounded_draw(Generator& g, u64 n) noexcept
{
    u128 m = static_cast<u128>(g.next()) * n;
    u64 low = static_cast<u64>(m);
    if (low < n) [[unlikely]]
    {
        const u64 threshold = (0 - n) % n;  // 2^64 mod n
        while (low < threshold)
        {
            m = static_cast<u128>(g.next()) * n;
            low = static_cast<u64>(m);
        }
    }
    return static_cast<u64>(m >> 64);
}

class SplitMix64
{
public:
    explicit constexpr SplitMix64(u64 seed = 0) noexcept : m_state(seed) {}

    [[nodiscard]] constexpr u64 state() const noexcept { return m_state; }

    /// The next 64-bit output.
    constexpr u64 next() noexcept
    {
        m_state += 0x9e3779b97f4a7c15ULL;
        return hash::mix64(m_state);
    }

    /// A uniform value in [0, n); n must be positive.
    u64 bounded(u64 n) noexcept { return bounded_draw(*this, n); }

    /// Fisher-Yates from the back. The draws are computed in batches (independent of each other, so the loop
    /// vectorizes), with the same results as for i = n..2: swap(a[i - 1], a[bounded(i)]): a batch in which some draw
    /// might be rejected (low64 < i, probability about i / 2^64) is redone draw by draw with bounded().
    template<class T>
    void shuffle(std::span<T> a) noexcept
    {
        using std::swap;
        usize i = a.size();
        if (i >= (usize{1} << 32))
        {
            for (; i > 1; --i)
                swap(a[i - 1], a[static_cast<usize>(bounded(i))]);
            return;
        }
        u32 js[k_batch];
        while (i > 1)
        {
            const usize m = i - 1 < k_batch ? i - 1 : k_batch;  // steps i, i - 1, ..., i - m + 1
            draw_batch(i, m, js);
            for (usize k = 0; k < m; ++k, --i)
                swap(a[i - 1], a[js[k]]);
        }
    }

private:
    static constexpr u64 k_gamma = 0x9e3779b97f4a7c15ULL;
    static constexpr usize k_batch = 64;

    /// out[k] = bounded(i - k) for k < m, as drawn one after the other (i < 2^32), advancing the state alike.
    void draw_batch(usize i, usize m, u32* out) noexcept
    {
        const u64 s0 = m_state;
        u64 maybe_rejected = 0;
        for (usize k = 0; k < m; ++k)
        {
            const u64 x = hash::mix64(s0 + (k + 1) * k_gamma);
            const u64 n = static_cast<u32>(i - k);  // i < 2^32: the products below are 32 x 32 bits
            const u64 hi = (x >> 32) * n, lo = (x & 0xffffffffULL) * n;
            out[k] = static_cast<u32>((hi + (lo >> 32)) >> 32);           // high64(x * n)
            maybe_rejected |= static_cast<u64>((hi << 32) + lo < n);  // low64(x * n) < n
        }
        if (maybe_rejected != 0) [[unlikely]]
        {
            for (usize k = 0; k < m; ++k)  // m_state is still s0
                out[k] = static_cast<u32>(bounded(i - k));
            return;
        }
        m_state = s0 + m * k_gamma;
    }

    u64 m_state;
};

/// xoshiro256** (Blackman and Vigna, "Scrambled linear pseudorandom number generators", 2018): 256 bits of state, a
/// period of 2^256 - 1, seeded from SplitMix64. The dataset samplers draw with it (datasets::Rng).
class Xoshiro256
{
public:
    explicit Xoshiro256(u64 seed = 0) noexcept { reseed(seed); }

    /// The state becomes the first four outputs of SplitMix64(seed), which are never all zero.
    void reseed(u64 seed) noexcept
    {
        SplitMix64 init(seed);
        for (u64& w : m_s)
            w = init.next();
    }

    /// The next 64-bit output.
    u64 next() noexcept
    {
        const u64 result = rotl(m_s[1] * 5, 7) * 9;
        const u64 t = m_s[1] << 17;
        m_s[2] ^= m_s[0];
        m_s[3] ^= m_s[1];
        m_s[1] ^= m_s[2];
        m_s[0] ^= m_s[3];
        m_s[2] ^= t;
        m_s[3] = rotl(m_s[3], 45);
        return result;
    }

    /// A uniform value in [0, n); n must be positive.
    u64 bounded(u64 n) noexcept { return bounded_draw(*this, n); }

    /// A uniform value in [0, 1) with 53 random bits.
    f64 uniform() noexcept { return static_cast<f64>(next() >> 11) * 0x1.0p-53; }

private:
    static constexpr u64 rotl(u64 x, int k) noexcept { return (x << k) | (x >> (64 - k)); }

    u64 m_s[4] = {0, 0, 0, 0};
};
}  // namespace mymyr
