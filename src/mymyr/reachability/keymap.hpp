#pragma once
// Internal: an open-addressing map from 64-bit keys (canonical atom ids) to small values.

#include "mymyr/core/hash.hpp"
#include "mymyr/core/types.hpp"

#include <vector>

namespace mymyr::reach
{
template<class V>
class KeyMap
{
public:
    static constexpr u64 k_empty = ~u64{0};

    KeyMap() { m_keys.assign(16, k_empty), m_vals.resize(16); }

    [[nodiscard]] usize size() const noexcept { return m_size; }
    void clear()
    {
        if (m_size)
        {
            std::fill(m_keys.begin(), m_keys.end(), k_empty);
            m_size = 0;
        }
    }
    void reserve(usize n)
    {
        usize cap = m_keys.size();
        while (cap * 7 < n * 10 + 10)
            cap *= 2;
        if (cap != m_keys.size())
            rehash(cap);
    }
    /// Pointer to the value of key, or nullptr.
    [[nodiscard]] const V* find(u64 key) const
    {
        const u64 mask = m_keys.size() - 1;
        for (u64 j = hash::mix64(key) & mask;; j = (j + 1) & mask)
        {
            if (m_keys[j] == key)
                return &m_vals[j];
            if (m_keys[j] == k_empty)
                return nullptr;
        }
    }
    [[nodiscard]] V* find(u64 key) { return const_cast<V*>(static_cast<const KeyMap*>(this)->find(key)); }
    /// Inserts (key, value) if absent; returns the stored value.
    V& insert(u64 key, const V& value)
    {
        if ((m_size + 1) * 10 > m_keys.size() * 7)
            rehash(m_keys.size() * 2);
        const u64 mask = m_keys.size() - 1;
        for (u64 j = hash::mix64(key) & mask;; j = (j + 1) & mask)
        {
            if (m_keys[j] == key)
                return m_vals[j];
            if (m_keys[j] == k_empty)
            {
                m_keys[j] = key;
                m_vals[j] = value;
                ++m_size;
                return m_vals[j];
            }
        }
    }

private:
    void rehash(usize cap)
    {
        std::vector<u64> keys(cap, k_empty);
        std::vector<V> vals(cap);
        const u64 mask = cap - 1;
        for (usize i = 0; i < m_keys.size(); ++i)
        {
            if (m_keys[i] == k_empty)
                continue;
            u64 j = hash::mix64(m_keys[i]) & mask;
            while (keys[j] != k_empty)
                j = (j + 1) & mask;
            keys[j] = m_keys[i];
            vals[j] = m_vals[i];
        }
        m_keys.swap(keys);
        m_vals.swap(vals);
    }

    std::vector<u64> m_keys;
    std::vector<V> m_vals;
    usize m_size = 0;
};
}  // namespace mymyr::reach
