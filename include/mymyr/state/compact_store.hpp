#pragma once
// Compact state set: closed states kept only as 128-bit fingerprints (an opt-in alternative to a full state store):
//   - ZobristKeys: two independent 64-bit keys per fluent atom, derived from the atom's *canonical* id, so a
//     fingerprint does not depend on the lazy slot numbering. A state's fingerprint is the xor of the keys of its true
//     atoms; a successor's fingerprint is updated from its delta in O(|delta|);
//   - CompactStateSet: open addressing over (u64, u64) pairs.
// Identity by fingerprint is probabilistic: two distinct states collide with probability about n^2 / 2^129.

#include "mymyr/core/hash.hpp"
#include "mymyr/core/types.hpp"

#include <utility>
#include <vector>

namespace mymyr
{
struct Fingerprint128
{
    u64 a = 0, b = 0;
    friend bool operator==(Fingerprint128, Fingerprint128) = default;
    void toggle(const Fingerprint128& k) noexcept
    {
        a ^= k.a;
        b ^= k.b;
    }
};

class CompactStateSet
{
public:
    CompactStateSet() : m_a(1u << 16, 0), m_b(1u << 16, 0), m_mask((1u << 16) - 1) {}

    /// Inserts a fingerprint; returns true if it is new.
    bool insert(Fingerprint128 f)
    {
        if (f.a == 0 && f.b == 0)
            f.b = 1;  // (0, 0) marks an empty entry
        if ((m_count + 1) * 10 > m_a.size() * 7)
            grow();
        for (u64 j = hash::mix64(f.a) & m_mask;; j = (j + 1) & m_mask)
        {
            if (!(m_a[j] | m_b[j]))
            {
                m_a[j] = f.a;
                m_b[j] = f.b;
                ++m_count;
                return true;
            }
            if (m_a[j] == f.a && m_b[j] == f.b)
                return false;
        }
    }
    /// Whether the fingerprint was inserted.
    [[nodiscard]] bool contains(Fingerprint128 f) const noexcept
    {
        if (f.a == 0 && f.b == 0)
            f.b = 1;
        for (u64 j = hash::mix64(f.a) & m_mask;; j = (j + 1) & m_mask)
        {
            if (!(m_a[j] | m_b[j]))
                return false;
            if (m_a[j] == f.a && m_b[j] == f.b)
                return true;
        }
    }
    [[nodiscard]] u64 size() const noexcept { return m_count; }
    [[nodiscard]] u64 bytes() const noexcept { return m_a.size() * 2 * sizeof(u64); }

private:
    void grow()
    {
        std::vector<u64> na(m_a.size() * 2, 0), nb(m_a.size() * 2, 0);
        const u64 m = na.size() - 1;
        for (usize i = 0; i < m_a.size(); ++i)
            if (m_a[i] | m_b[i])
            {
                u64 j = hash::mix64(m_a[i]) & m;
                while (na[j] | nb[j])
                    j = (j + 1) & m;
                na[j] = m_a[i];
                nb[j] = m_b[i];
            }
        m_a.swap(na);
        m_b.swap(nb);
        m_mask = m;
    }

    std::vector<u64> m_a, m_b;
    u64 m_mask;
    u64 m_count = 0;
};
}  // namespace mymyr
