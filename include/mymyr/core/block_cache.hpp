#pragma once
// Recycling allocator for the output blocks of batch calls (encodings, expansions, table batches). Fresh memory
// costs a kernel page fault per 4 KiB page, and a block of tens of megabytes is served by the system allocator with
// fresh pages every time and unmapped on release, which costs several times the work of filling it. Released blocks
// are therefore kept and handed out again with their pages still mapped, in two tiers: a few blocks of at most
// k_thread_block_max bytes per thread (no locking), and a few larger blocks, up to k_block_cache_limit bytes in all,
// shared by all threads under a mutex. A block returns to the cache from whichever thread releases it last.

#include "mymyr/core/types.hpp"

#include <cstddef>
#include <memory>

namespace mymyr
{
inline constexpr u64 k_thread_block_max = u64{4} << 20;
inline constexpr u64 k_block_cache_limit = u64{512} << 20;

struct BlockDeleter
{
    u64 capacity = 0;
    void operator()(std::byte* block) const noexcept;
};

/// A host block aligned to a cache line; its contents are unspecified.
using HostBlock = std::unique_ptr<std::byte[], BlockDeleter>;

/// A block of at least `bytes` bytes (the deleter's capacity says how many); releasing it returns it to the cache.
[[nodiscard]] HostBlock acquire_host_block(u64 bytes);

/// Frees every cached block (the cache refills as large blocks are released).
void trim_block_cache() noexcept;

/// Bytes currently held by the cache.
[[nodiscard]] u64 block_cache_bytes() noexcept;
}  // namespace mymyr
