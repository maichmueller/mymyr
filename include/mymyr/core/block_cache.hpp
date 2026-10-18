#pragma once
// Recycling allocator for the output blocks of batch calls (encodings, expansions, table batches). Fresh memory
// costs a kernel page fault per 4 KiB page, and a block of tens of megabytes is served by the system allocator with
// fresh pages every time and unmapped on release, which costs several times the work of filling it. Released blocks
// are therefore kept and handed out again with their pages still mapped.
//
// Blocks are cached in two tiers by size: a few blocks of at most k_thread_block_max bytes per thread (no locking),
// and a few larger blocks, up to k_block_cache_limit bytes in all, shared by all threads under a mutex (a block over
// k_global_block_max bytes is never kept). A block returns to the cache of the thread that releases it, which need not
// be the thread that acquired it. Blocks of at most k_cached_block_min bytes come from the system allocator and have
// the requested size rounded up to a cache line (a recycled one may be up to a quarter larger); larger ones are
// anonymous memory mappings, so dropping one gives its memory back to the operating system.
//
// Memory is released in three ways: a shared block that no acquire took within the k_idle_acquires acquires of blocks
// above k_cached_block_min bytes that followed its release is unmapped; the least recently released blocks make room
// for newer ones; and trim_block_cache() drops everything the calling thread can reach.

#include "mymyr/core/types.hpp"

#include <cstddef>
#include <utility>

namespace mymyr
{
inline constexpr u64 k_cached_block_min = u64{128} << 10;
inline constexpr u64 k_thread_block_max = u64{4} << 20;
inline constexpr usize k_thread_blocks = 4;
inline constexpr u64 k_block_cache_limit = u64{512} << 20;
inline constexpr u64 k_global_block_max = k_block_cache_limit / 2;
inline constexpr usize k_global_blocks = 8;
inline constexpr u64 k_idle_acquires = 256;

/// A host block aligned to a cache line; its contents are unspecified. Move-only; releasing it (by destruction or
/// reset()) hands it back to the cache from the calling thread.
class HostBlock
{
public:
    HostBlock() noexcept = default;
    HostBlock(const HostBlock&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock(HostBlock&& other) noexcept
        : m_data(std::exchange(other.m_data, nullptr)), m_capacity(std::exchange(other.m_capacity, 0))
    {
    }
    HostBlock& operator=(HostBlock&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_data = std::exchange(other.m_data, nullptr);
            m_capacity = std::exchange(other.m_capacity, 0);
        }
        return *this;
    }
    ~HostBlock() { reset(); }

    [[nodiscard]] std::byte* get() const noexcept { return m_data; }
    /// The usable bytes, at least those that were requested.
    [[nodiscard]] u64 capacity() const noexcept { return m_capacity; }
    explicit operator bool() const noexcept { return m_data != nullptr; }
    /// Releases the block; the object is empty afterwards.
    void reset() noexcept;

private:
    friend HostBlock acquire_host_block(u64 bytes);
    HostBlock(std::byte* data, u64 capacity) noexcept : m_data(data), m_capacity(capacity) {}

    std::byte* m_data = nullptr;
    u64 m_capacity = 0;
};

/// A block of at least `bytes` bytes (capacity() says how many). Throws std::bad_alloc when the memory cannot be
/// obtained, which includes every request above 2^62 bytes.
[[nodiscard]] HostBlock acquire_host_block(u64 bytes);

/// Frees the blocks the calling thread's tier and the shared tier hold. The private tiers of other threads are
/// freed when those threads end, and are bounded by k_thread_blocks blocks of k_thread_block_max bytes each.
void trim_block_cache() noexcept;

/// Bytes currently held by the calling thread's tier and the shared tier.
[[nodiscard]] u64 block_cache_bytes() noexcept;
}  // namespace mymyr
