#include "mymyr/core/block_cache.hpp"

#include "mymyr/core/memory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <mutex>
#include <new>

namespace mymyr
{
namespace
{
constexpr u64 k_round = u64{1} << 20;  // large requests are rounded up so that similar sizes share blocks

struct Entry
{
    std::byte* block = nullptr;
    u64 capacity = 0;
    u64 stamp = 0;
};

std::byte* allocate_system(u64 capacity)
{
    return static_cast<std::byte*>(::operator new(capacity, std::align_val_t{k_cache_line}));
}

void free_system(std::byte* block) noexcept { ::operator delete(block, std::align_val_t{k_cache_line}); }

/// Blocks of at most k_thread_block_max bytes released on this thread, reused by its next acquire without locking.
struct ThreadTier
{
    static constexpr usize k_keep = 4;
    std::array<Entry, k_keep> slots{};
    u64 clock = 0;

    ~ThreadTier()
    {
        gone() = true;
        for (Entry& e : slots)
            if (e.block)
                free_system(e.block);
    }
    /// Trivially destructible, so readable while and after the thread's tier is destroyed.
    static bool& gone() noexcept
    {
        thread_local bool flag = false;
        return flag;
    }
    static ThreadTier* get() noexcept
    {
        if (gone())
            return nullptr;
        thread_local ThreadTier tier;
        return &tier;
    }

    std::byte* take(u64 need, u64& capacity) noexcept
    {
        Entry* best = nullptr;  // the smallest cached block that fits and is at most 4x the request
        for (Entry& e : slots)
            if (e.block && e.capacity >= need && e.capacity / 4 <= need && (!best || e.capacity < best->capacity))
                best = &e;
        if (!best)
            return nullptr;
        std::byte* block = best->block;
        capacity = best->capacity;
        *best = Entry{};
        return block;
    }
    void put(std::byte* block, u64 capacity) noexcept
    {
        Entry* victim = &slots[0];
        for (Entry& e : slots)
            if (!e.block)
            {
                victim = &e;
                break;
            }
            else if (e.stamp < victim->stamp)
                victim = &e;
        std::byte* evicted = victim->block;
        *victim = Entry{block, capacity, ++clock};
        if (evicted)
            free_system(evicted);
    }
};

/// Larger blocks, shared by all threads. Never destroyed: a block released during static destruction must still find
/// a live cache.
struct GlobalTier
{
    static constexpr usize k_slots = 8;
    std::mutex mutex;
    std::array<Entry, k_slots> slots{};
    u64 bytes = 0;
    u64 clock = 0;

    static GlobalTier& get() noexcept
    {
        static GlobalTier* const instance = new GlobalTier;
        return *instance;
    }
};
}  // namespace

HostBlock acquire_host_block(u64 bytes)
{
    bytes = std::max<u64>(bytes, k_cache_line);
    if (bytes <= k_thread_block_max)
    {
        if (ThreadTier* tier = ThreadTier::get())
        {
            u64 capacity = 0;
            if (std::byte* hit = tier->take(bytes, capacity))
                return HostBlock(hit, BlockDeleter{capacity});
        }
        const u64 capacity = std::bit_ceil(bytes);  // powers of two, so that varying sizes reuse each other's blocks
        return HostBlock(allocate_system(capacity), BlockDeleter{capacity});
    }
    bytes = (bytes + k_round - 1) / k_round * k_round;
    {
        GlobalTier& g = GlobalTier::get();
        const std::scoped_lock lock(g.mutex);
        Entry* best = nullptr;  // the smallest cached block that fits without wasting more than the request itself
        for (Entry& e : g.slots)
            if (e.block && e.capacity >= bytes && e.capacity <= 2 * bytes && (!best || e.capacity < best->capacity))
                best = &e;
        if (best)
        {
            const Entry hit = *best;
            *best = Entry{};
            g.bytes -= hit.capacity;
            return HostBlock(hit.block, BlockDeleter{hit.capacity});
        }
    }
    return HostBlock(allocate_system(bytes), BlockDeleter{bytes});
}

void BlockDeleter::operator()(std::byte* block) const noexcept
{
    if (!block)
        return;
    if (capacity <= k_thread_block_max)
    {
        if (ThreadTier* tier = ThreadTier::get())
        {
            tier->put(block, capacity);
            return;
        }
        free_system(block);
        return;
    }
    if (capacity > k_block_cache_limit / 2)
    {
        free_system(block);
        return;
    }
    std::array<std::byte*, GlobalTier::k_slots + 1> evicted{};
    usize n_evicted = 0;
    bool kept = false;
    {
        GlobalTier& g = GlobalTier::get();
        const std::scoped_lock lock(g.mutex);
        // make room by dropping the least recently released blocks
        for (;;)
        {
            Entry* free_slot = nullptr;
            Entry* oldest = nullptr;
            for (Entry& e : g.slots)
            {
                if (!e.block)
                    free_slot = &e;
                else if (!oldest || e.stamp < oldest->stamp)
                    oldest = &e;
            }
            if (free_slot && g.bytes + capacity <= k_block_cache_limit)
            {
                *free_slot = Entry{block, capacity, ++g.clock};
                g.bytes += capacity;
                kept = true;
                break;
            }
            if (!oldest)
                break;
            evicted[n_evicted++] = oldest->block;
            g.bytes -= oldest->capacity;
            *oldest = Entry{};
        }
    }
    if (!kept)
        free_system(block);
    for (usize i = 0; i < n_evicted; ++i)
        free_system(evicted[i]);
}

void trim_block_cache() noexcept
{
    if (ThreadTier* tier = ThreadTier::get())
        for (Entry& e : tier->slots)
        {
            if (e.block)
                free_system(e.block);
            e = Entry{};
        }
    std::array<std::byte*, GlobalTier::k_slots> dropped{};
    {
        GlobalTier& g = GlobalTier::get();
        const std::scoped_lock lock(g.mutex);
        for (usize i = 0; i < GlobalTier::k_slots; ++i)
        {
            dropped[i] = g.slots[i].block;
            g.slots[i] = Entry{};
        }
        g.bytes = 0;
    }
    for (std::byte* b : dropped)
        if (b)
            free_system(b);
}

u64 block_cache_bytes() noexcept
{
    u64 total = 0;
    if (ThreadTier* tier = ThreadTier::get())
        for (const Entry& e : tier->slots)
            total += e.capacity;
    GlobalTier& g = GlobalTier::get();
    const std::scoped_lock lock(g.mutex);
    return total + g.bytes;
}
}  // namespace mymyr
