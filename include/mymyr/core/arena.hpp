#pragma once
// SegmentedArena: an append-only array with stable element addresses.
//
// Segment k holds `base << k` elements and starts at index base * (2^k - 1), so the segment directory is a fixed
// array of at most 64 pointers that never reallocates. Indexing is two shifts and a count-leading-zeros. Because
// nothing ever moves:
//   - other threads may read any index that was published to them through a release/acquire pair (the concurrent
//     store's CAS table does this), while one writer keeps appending;
//   - host<->device sync copies only the tail, segment by segment (for_each_run);
//   - dropping the arena releases everything at once.
//
// `alloc_contiguous(n)` returns the start of n elements that do not straddle a segment boundary (skipping the rest
// of the current segment if needed). State records use it, so a record is always one contiguous word run.

#include "mymyr/core/memory.hpp"
#include "mymyr/core/types.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mymyr
{
template<class T>
class SegmentedArena
{
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "arenas hold POD data (shared with the device)");

public:
    static constexpr u32 k_max_segments = 48;

    explicit SegmentedArena(u32 base_log = 12, MemoryResource* mr = host_memory()) : m_base_log(base_log), m_mr(mr)
    {
        if (base_log > 30)
            throw std::invalid_argument("SegmentedArena: base_log must be <= 30");
    }
    SegmentedArena(const SegmentedArena&) = delete;
    SegmentedArena& operator=(const SegmentedArena&) = delete;
    SegmentedArena(SegmentedArena&& o) noexcept { swap(o); }
    SegmentedArena& operator=(SegmentedArena&& o) noexcept
    {
        if (this != &o)
        {
            release();
            swap(o);
        }
        return *this;
    }
    ~SegmentedArena() { release(); }

    [[nodiscard]] u64 size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] u64 base() const noexcept { return u64{1} << m_base_log; }
    [[nodiscard]] MemoryResource* resource() const noexcept { return m_mr; }
    /// Largest n accepted by alloc_contiguous.
    [[nodiscard]] u64 max_contiguous() const noexcept { return base(); }

    [[nodiscard]] T& operator[](u64 i) noexcept { return *ptr(i); }
    [[nodiscard]] const T& operator[](u64 i) const noexcept { return *ptr(i); }

    [[nodiscard]] T* ptr(u64 i) noexcept
    {
        const auto [k, off] = locate(i);
        return m_seg[k] + off;
    }
    [[nodiscard]] const T* ptr(u64 i) const noexcept
    {
        const auto [k, off] = locate(i);
        return m_seg[k] + off;
    }

    u64 push_back(const T& v)
    {
        const u64 i = alloc_contiguous(1);
        *ptr(i) = v;
        return i;
    }

    /// Reserve n contiguous elements (uninitialized) and return the index of the first one.
    u64 alloc_contiguous(u64 n)
    {
        if (n == 0)
            return m_size;
        if (n > base())
            throw std::length_error("SegmentedArena: contiguous run larger than the base segment");
        auto [k, off] = locate(m_size);
        if (off + n > segment_size(k))
        {  // skip the tail of segment k
            m_size = segment_start(k + 1);
            k += 1;
            off = 0;
        }
        ensure_segment(k);
        const u64 start = m_size;
        m_size += n;
        return start;
    }

    /// Append n elements (may span segments).
    u64 append(const T* src, u64 n)
    {
        const u64 start = m_size;
        while (n > 0)
        {
            auto [k, off] = locate(m_size);
            ensure_segment(k);
            const u64 take = std::min<u64>(n, segment_size(k) - off);
            std::memcpy(m_seg[k] + off, src, take * sizeof(T));
            m_size += take;
            src += take;
            n -= take;
        }
        return start;
    }

    /// Calls f(index, pointer, count) for every contiguous run covering [lo, hi). Used for tail sync.
    template<class F>
    void for_each_run(u64 lo, u64 hi, F&& f) const
    {
        while (lo < hi)
        {
            const auto [k, off] = locate(lo);
            const u64 take = std::min<u64>(hi - lo, segment_size(k) - off);
            f(lo, m_seg[k] + off, take);
            lo += take;
        }
    }

    void clear() noexcept { m_size = 0; }  // keeps the segments for reuse

    [[nodiscard]] u64 segment_size(u32 k) const noexcept { return base() << k; }
    [[nodiscard]] u64 segment_start(u32 k) const noexcept { return ((u64{1} << k) - 1) << m_base_log; }

    [[nodiscard]] std::pair<u32, u64> locate(u64 i) const noexcept
    {
        const u64 j = (i >> m_base_log) + 1;
        const u32 k = static_cast<u32>(63 - std::countl_zero(j));
        return {k, i - segment_start(k)};
    }

private:
    void ensure_segment(u32 k)
    {
        if (k >= k_max_segments)
            throw std::length_error("SegmentedArena: capacity exhausted");
        if (!m_seg[k])
            m_seg[k] = static_cast<T*>(m_mr->allocate(segment_size(k) * sizeof(T), alignof(T) < 64 ? 64 : alignof(T)));
    }
    void release() noexcept
    {
        for (u32 k = 0; k < k_max_segments; ++k)
            if (m_seg[k])
            {
                m_mr->deallocate(m_seg[k], segment_size(k) * sizeof(T), alignof(T) < 64 ? 64 : alignof(T));
                m_seg[k] = nullptr;
            }
        m_size = 0;
    }
    void swap(SegmentedArena& o) noexcept
    {
        std::swap(m_seg, o.m_seg);
        std::swap(m_size, o.m_size);
        std::swap(m_base_log, o.m_base_log);
        std::swap(m_mr, o.m_mr);
    }

    std::array<T*, k_max_segments> m_seg{};
    u64 m_size = 0;
    u32 m_base_log = 12;
    MemoryResource* m_mr = host_memory();
};
}  // namespace mymyr
