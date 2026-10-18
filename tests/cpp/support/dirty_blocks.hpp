#pragma once
// Recycled memory for tests: output blocks come back from the block cache and the system allocator with whatever their
// last user left in them, so a test that expects zeros, padding or untouched entries would pass by luck on fresh
// (zero) pages. dirty_recycled_blocks() leaves non-zero bytes in the blocks that the next acquires hand out.

#include "mymyr/core/block_cache.hpp"

#include <cstring>
#include <vector>

namespace mymyr::test
{
/// Acquires blocks of every size up to `largest` bytes (on a grid of a quarter step per doubling), fills them with a
/// non-zero byte and releases them: the next acquires get recycled, non-zero memory, from the block cache or from the
/// system allocator's free chunks.
inline void dirty_recycled_blocks(u64 largest = u64{1} << 20)
{
    constexpr int k_fill = 0xa5;
    std::vector<HostBlock> held;
    for (u64 bytes = 64; bytes <= largest; bytes += bytes / 4 + 64)
    {
        HostBlock block = acquire_host_block(bytes);
        std::memset(block.get(), k_fill, block.capacity());
        held.push_back(std::move(block));
    }
    // a small block that stays allocated above the others keeps the allocator from returning the freed chunks to the
    // system as a trimmed heap top
    static const HostBlock guard = acquire_host_block(64);
    (void)guard;
    held.clear();
}
}  // namespace mymyr::test
