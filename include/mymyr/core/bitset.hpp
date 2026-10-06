#pragma once
// Bit operations over u64 word spans: the contract that state words follow throughout mymyr.
//   - little-endian u64 words; slot i lives in word i >> 6, bit i & 63;
//   - a set may be stored with fewer words than another: missing words compare as zero (lazy slots let the word
//     count grow as new atoms are seen), so equality and hashing run over the *trimmed* length.
// Everything here is MYMYR_HD so the CUDA executor uses the same definitions.

#include "mymyr/core/types.hpp"

#include <algorithm>
#include <bit>
#include <span>
#include <vector>

namespace mymyr::bits
{
[[nodiscard]] MYMYR_HD constexpr u32 words_for(u64 nbits) { return static_cast<u32>((nbits + 63) >> 6); }
[[nodiscard]] MYMYR_HD constexpr u32 word_of(u64 bit) { return static_cast<u32>(bit >> 6); }
[[nodiscard]] MYMYR_HD constexpr u64 mask_of(u64 bit) { return u64{1} << (bit & 63); }

[[nodiscard]] MYMYR_HD int popcount64(u64 x)
{
#if defined(__CUDA_ARCH__)
    return __popcll(x);
#else
    return std::popcount(x);
#endif
}

[[nodiscard]] MYMYR_HD int ctz64(u64 x)  // x != 0
{
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(x)) - 1;
#else
    return std::countr_zero(x);
#endif
}

/// Bit test that treats words beyond `n` as zero.
[[nodiscard]] MYMYR_HD bool test(const u64* w, u32 n, u64 bit)
{
    const u32 i = word_of(bit);
    return i < n && (w[i] & mask_of(bit)) != 0;
}
MYMYR_HD void set(u64* w, u64 bit) { w[word_of(bit)] |= mask_of(bit); }
MYMYR_HD void reset(u64* w, u64 bit) { w[word_of(bit)] &= ~mask_of(bit); }

/// Number of words after dropping trailing zero words.
[[nodiscard]] MYMYR_HD u32 trimmed_size(const u64* w, u32 n)
{
    MYMYR_NOVECTOR
    while (n > 0 && w[n - 1] == 0)
        --n;
    return n;
}

[[nodiscard]] MYMYR_HD u64 count(const u64* w, u32 n)
{
    u64 c = 0;
    for (u32 i = 0; i < n; ++i)
        c += static_cast<u64>(popcount64(w[i]));
    return c;
}

[[nodiscard]] MYMYR_HD bool any(const u64* w, u32 n)
{
    MYMYR_NOVECTOR
    for (u32 i = 0; i < n; ++i)
        if (w[i])
            return true;
    return false;
}

/// a == b with missing words treated as zero.
[[nodiscard]] MYMYR_HD bool equal(const u64* a, u32 na, const u64* b, u32 nb)
{
    const u32 m = na < nb ? na : nb;
    MYMYR_NOVECTOR
    for (u32 i = 0; i < m; ++i)
        if (a[i] != b[i])
            return false;
    MYMYR_NOVECTOR
    for (u32 i = m; i < na; ++i)
        if (a[i])
            return false;
    MYMYR_NOVECTOR
    for (u32 i = m; i < nb; ++i)
        if (b[i])
            return false;
    return true;
}

/// (a & ~b) == 0, i.e. a is a subset of b, missing words zero.
[[nodiscard]] MYMYR_HD bool subset(const u64* a, u32 na, const u64* b, u32 nb)
{
    MYMYR_NOVECTOR
    for (u32 i = 0; i < na; ++i)
        if (a[i] & ~(i < nb ? b[i] : u64{0}))
            return false;
    return true;
}

/// (a & b) != 0, missing words zero.
[[nodiscard]] MYMYR_HD bool intersects(const u64* a, u32 na, const u64* b, u32 nb)
{
    const u32 m = na < nb ? na : nb;
    MYMYR_NOVECTOR
    for (u32 i = 0; i < m; ++i)
        if (a[i] & b[i])
            return true;
    return false;
}

/// Calls f(bit) for every set bit in increasing order.
template<class F>
MYMYR_HD void for_each(const u64* w, u32 n, F&& f)
{
    for (u32 i = 0; i < n; ++i)
    {
        u64 x = w[i];
        while (x)
        {
            f(static_cast<u64>(i) * 64 + static_cast<u64>(ctz64(x)));
            x &= x - 1;
        }
    }
}

/// Owning, growable bitset for host-side code (compile-time structures, tests). Hot paths use raw word spans.
class Bitset
{
public:
    Bitset() = default;
    explicit Bitset(u64 nbits) : m_words(words_for(nbits), 0) {}

    [[nodiscard]] bool test(u64 bit) const { return bits::test(m_words.data(), size_words(), bit); }
    void set(u64 bit)
    {
        grow_to(bit + 1);
        bits::set(m_words.data(), bit);
    }
    void reset(u64 bit)
    {
        if (word_of(bit) < m_words.size())
            bits::reset(m_words.data(), bit);
    }
    void clear() { std::fill(m_words.begin(), m_words.end(), u64{0}); }
    void grow_to(u64 nbits)
    {
        if (words_for(nbits) > m_words.size())
            m_words.resize(words_for(nbits), 0);
    }

    [[nodiscard]] u64 count() const { return bits::count(m_words.data(), size_words()); }
    [[nodiscard]] u32 size_words() const { return static_cast<u32>(m_words.size()); }
    [[nodiscard]] std::span<const u64> words() const { return m_words; }
    [[nodiscard]] std::span<u64> words() { return m_words; }

    template<class F>
    void for_each(F&& f) const
    {
        bits::for_each(m_words.data(), size_words(), static_cast<F&&>(f));
    }

    friend bool operator==(const Bitset& a, const Bitset& b)
    {
        return bits::equal(a.m_words.data(), a.size_words(), b.m_words.data(), b.size_words());
    }

private:
    std::vector<u64> m_words;
};
}  // namespace mymyr::bits
