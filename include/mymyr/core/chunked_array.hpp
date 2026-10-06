#pragma once
// ChunkedArray<T>: a fixed-capacity array whose blocks are allocated on first write and published through atomic
// block pointers. Used for append-only arrays such as the two-level atom index's `cid_of[slot]` / `atom_pred[slot]` /
// `atom_args`.
//   - The block directory is sized once from the capacity, so it never reallocates and a reader never sees a moving
//     element, while any number of writers allocate blocks concurrently (CAS on the block pointer).
//   - Element i of a published block is read with one relaxed load of the block pointer plus the element load. The
//     happens-before edge for the element itself is the caller's (the atom index publishes a slot with a release store
//     after writing its record).
//   - `stride` > 1 stores fixed-size records of `stride` elements: record r occupies [r * stride, (r + 1) * stride) and
//     never straddles a block.

#include "mymyr/core/types.hpp"

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace mymyr
{
template<class T, u32 BlockLog = 12>
class ChunkedArray
{
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);

public:
    static constexpr u64 k_block_records = u64{1} << BlockLog;

    ChunkedArray() = default;
    ChunkedArray(u64 capacity_records, u32 stride) { reset(capacity_records, stride); }
    ChunkedArray(const ChunkedArray&) = delete;
    ChunkedArray& operator=(const ChunkedArray&) = delete;
    ~ChunkedArray() { release(); }

    /// Drops everything and resizes the directory. Not thread-safe.
    void reset(u64 capacity_records, u32 stride)
    {
        release();
        m_stride = stride == 0 ? 1 : stride;
        m_capacity = capacity_records;
        m_nblocks = (capacity_records >> BlockLog) + 1;
        m_blocks = std::make_unique<std::atomic<T*>[]>(m_nblocks);
        for (u64 b = 0; b < m_nblocks; ++b)
            m_blocks[b].store(nullptr, std::memory_order_relaxed);
    }

    [[nodiscard]] u64 capacity() const noexcept { return m_capacity; }
    [[nodiscard]] u32 stride() const noexcept { return m_stride; }

    /// Pointer to record r, allocating (zero-filled) its block if needed. Thread-safe.
    T* ensure(u64 r)
    {
        if (r >= m_capacity)
            throw std::length_error("ChunkedArray: capacity exceeded");
        const u64 b = r >> BlockLog;
        T* p = m_blocks[b].load(std::memory_order_acquire);
        if (!p)
        {
            T* fresh = new T[k_block_records * m_stride];
            std::memset(static_cast<void*>(fresh), 0, sizeof(T) * k_block_records * m_stride);
            if (m_blocks[b].compare_exchange_strong(p, fresh, std::memory_order_acq_rel))
                p = fresh;
            else
                delete[] fresh;
        }
        return p + (r & (k_block_records - 1)) * m_stride;
    }

    /// Record r of an allocated block.
    [[nodiscard]] const T* get(u64 r) const noexcept
    {
        return m_blocks[r >> BlockLog].load(std::memory_order_relaxed) + (r & (k_block_records - 1)) * m_stride;
    }

    [[nodiscard]] u64 bytes() const noexcept
    {
        u64 total = m_nblocks * sizeof(void*);
        for (u64 b = 0; b < m_nblocks; ++b)
            if (m_blocks[b].load(std::memory_order_relaxed))
                total += k_block_records * m_stride * sizeof(T);
        return total;
    }

private:
    void release() noexcept
    {
        for (u64 b = 0; b < m_nblocks; ++b)
            delete[] m_blocks[b].load(std::memory_order_relaxed);
        m_blocks.reset();
        m_nblocks = 0;
    }

    std::unique_ptr<std::atomic<T*>[]> m_blocks;
    u64 m_nblocks = 0;
    u64 m_capacity = 0;
    u32 m_stride = 1;
};
}  // namespace mymyr
