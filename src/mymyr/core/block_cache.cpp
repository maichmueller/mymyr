#include "mymyr/core/block_cache.hpp"

#include "mymyr/core/memory.hpp"

#include <pthread.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>

#if defined(__SANITIZE_ADDRESS__)
#define MYMYR_BLOCK_CACHE_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MYMYR_BLOCK_CACHE_ASAN 1
#endif
#endif
#if defined(MYMYR_BLOCK_CACHE_ASAN)
#include <sanitizer/asan_interface.h>
#endif

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif

namespace mymyr
{
namespace
{
constexpr u64 k_max_request = u64{1} << 62;  // larger requests cannot be served (and would overflow the rounding)
[[maybe_unused]] constexpr u64 k_huge_page = u64{2} << 20;
constexpr u64 k_round = u64{1} << 20;        // large requests are rounded up so that similar sizes share blocks

constexpr u64 round_up(u64 n, u64 multiple) noexcept { return (n + multiple - 1) / multiple * multiple; }

/// The capacity of a block of `bytes` bytes (k_cached_block_min < bytes <= k_thread_block_max) on a grid of four steps
/// per doubling, so that similar sizes share blocks and no block exceeds its request by more than a quarter.
constexpr u64 grid_capacity(u64 bytes) noexcept { return round_up(bytes, std::bit_floor(bytes - 1) / 4); }

/// Cached memory is unreadable while it sits in a tier, so that sanitized builds report a use after release.
void poison([[maybe_unused]] std::byte* block, [[maybe_unused]] u64 capacity) noexcept
{
#if defined(MYMYR_BLOCK_CACHE_ASAN)
    ASAN_POISON_MEMORY_REGION(block, capacity);
#endif
}

void unpoison([[maybe_unused]] std::byte* block, [[maybe_unused]] u64 capacity) noexcept
{
#if defined(MYMYR_BLOCK_CACHE_ASAN)
    ASAN_UNPOISON_MEMORY_REGION(block, capacity);
#endif
}

/// Page-aligned anonymous memory, so that unmapping returns it to the operating system.
std::byte* map_block(u64 capacity)
{
    void* block = ::mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (block == MAP_FAILED)
        throw std::bad_alloc();
#if defined(MADV_HUGEPAGE)
    // a fresh block of at least a huge page takes a page fault per 2 MiB instead of per 4 KiB, which fills it several
    // times faster; a hint, so a system without huge pages ignores it
    if (capacity >= k_huge_page)
        ::madvise(block, capacity, MADV_HUGEPAGE);
#endif
    return static_cast<std::byte*>(block);
}

/// A cache-line aligned block of the system allocator. A plain allocation is aligned by hand, because an aligned
/// allocation costs several times as much for blocks of a few kilobytes.
std::byte* allocate_small(u64 capacity)
{
    auto* raw = static_cast<std::byte*>(::operator new(capacity + k_cache_line));
    std::byte* aligned = raw + k_cache_line - reinterpret_cast<std::uintptr_t>(raw) % k_cache_line;
    aligned[-1] = static_cast<std::byte>(aligned - raw);  // the distance back to the allocation, 1 to 64
    return aligned;
}

void free_small(std::byte* block) noexcept { ::operator delete(block - static_cast<unsigned>(block[-1])); }

/// Gives a block back to the system, whichever way it was obtained.
void free_block(std::byte* block, u64 capacity) noexcept
{
    unpoison(block, capacity);  // the shadow of the range must not outlive the block
    if (capacity <= k_cached_block_min)
        free_small(block);
    else
        ::munmap(block, capacity);
}

struct Entry
{
    std::byte* block = nullptr;
    u64 capacity = 0;
    u64 stamp = 0;     // release order: the larger, the more recent
    u64 released = 0;  // the shared tier's acquire count at the release
};

/// Blocks of at most k_thread_block_max bytes released on this thread, reused by its next acquire without locking.
struct ThreadTier
{
    std::array<Entry, k_thread_blocks> slots{};
    u64 clock = 0;

    ~ThreadTier()
    {
        gone() = true;
        for (Entry& e : slots)
            if (e.block)
                free_block(e.block, e.capacity);
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

    /// The smallest cached block of at least `need` and at most need + `slack` bytes, the most recently released among
    /// equals (its lines are the likeliest to still be cached).
    std::byte* take(u64 need, u64 slack, u64& capacity) noexcept
    {
        Entry* best = nullptr;
        for (Entry& e : slots)
            if (e.block && e.capacity >= need && e.capacity - need <= slack &&
                (!best || e.capacity < best->capacity || (e.capacity == best->capacity && e.stamp > best->stamp)))
                best = &e;
        if (!best)
            return nullptr;
        std::byte* block = best->block;
        capacity = best->capacity;
        *best = Entry{};
        unpoison(block, capacity);
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
        const Entry evicted = *victim;
        poison(block, capacity);
        *victim = Entry{block, capacity, ++clock, 0};
        if (evicted.block)
            free_block(evicted.block, evicted.capacity);
    }
};

/// Larger blocks, shared by all threads. Never destroyed: a block released during static destruction must still find
/// a live cache.
struct GlobalTier
{
    static constexpr u64 k_never = std::numeric_limits<u64>::max();

    std::mutex mutex;
    std::array<Entry, k_global_blocks> slots{};
    u64 bytes = 0;
    u64 clock = 0;
    /// The number of acquires of blocks above k_cached_block_min bytes so far (by every thread).
    std::atomic<u64> acquires{0};
    /// The acquire count from which some cached block is idle (k_never when none is cached): acquires compare against
    /// it without taking the mutex.
    std::atomic<u64> expiry{k_never};

    GlobalTier()
    {
        // a fork must not copy the mutex while another thread holds it
        ::pthread_atfork(
            [] { get().mutex.lock(); }, [] { get().mutex.unlock(); }, [] { get().mutex.unlock(); });
    }

    static GlobalTier& get() noexcept
    {
        static GlobalTier* const instance = new GlobalTier;
        return *instance;
    }

    /// Requires the mutex.
    void refresh_expiry() noexcept
    {
        u64 oldest = k_never;
        for (const Entry& e : slots)
            if (e.block)
                oldest = std::min(oldest, e.released);
        expiry.store(oldest == k_never ? k_never : oldest + k_idle_acquires + 1, std::memory_order_relaxed);
    }

    /// Unmaps the blocks that no acquire took during the k_idle_acquires acquires before `now`.
    void drop_idle(u64 now) noexcept
    {
        std::array<Entry, k_global_blocks> dropped{};
        {
            const std::scoped_lock lock(mutex);
            for (usize i = 0; i < slots.size(); ++i)
                if (slots[i].block && now - slots[i].released > k_idle_acquires)
                {
                    dropped[i] = slots[i];
                    bytes -= slots[i].capacity;
                    slots[i] = Entry{};
                }
            refresh_expiry();
        }
        for (const Entry& e : dropped)
            if (e.block)
                free_block(e.block, e.capacity);
    }
};

/// The tier exists, and so does its fork handler, before main() and before any thread another library starts.
[[maybe_unused]] const bool k_global_tier_ready = (GlobalTier::get(), true);
}  // namespace

HostBlock acquire_host_block(u64 bytes)
{
    if (bytes > k_max_request)
        throw std::bad_alloc();
    bytes = std::max<u64>(bytes, k_cache_line);
    if (bytes <= k_cached_block_min)
    {
        // exact sizes: a recycled block may be up to a quarter larger than the request
        const u64 need = round_up(bytes, k_cache_line);
        if (ThreadTier* tier = ThreadTier::get())
        {
            u64 capacity = 0;
            if (std::byte* hit = tier->take(need, need / 4, capacity))
                return HostBlock(hit, capacity);
        }
        return HostBlock(allocate_small(need), need);
    }
    GlobalTier& g = GlobalTier::get();
    const u64 now = g.acquires.fetch_add(1, std::memory_order_relaxed) + 1;
    if (now >= g.expiry.load(std::memory_order_relaxed))
        g.drop_idle(now);
    if (bytes <= k_thread_block_max)
    {
        if (ThreadTier* tier = ThreadTier::get())
        {
            u64 capacity = 0;
            if (std::byte* hit = tier->take(bytes, 3 * bytes, capacity))  // at most four times the request
                return HostBlock(hit, capacity);
        }
        const u64 capacity = grid_capacity(bytes);
        return HostBlock(map_block(capacity), capacity);
    }
    bytes = round_up(bytes, k_round);
    Entry hit;
    {
        const std::scoped_lock lock(g.mutex);
        Entry* best = nullptr;  // the smallest cached block that fits without wasting more than the request itself
        for (Entry& e : g.slots)
            if (e.block && e.capacity >= bytes && e.capacity <= 2 * bytes &&
                (!best || e.capacity < best->capacity || (e.capacity == best->capacity && e.stamp > best->stamp)))
                best = &e;
        if (best)
        {
            hit = *best;
            *best = Entry{};
            g.bytes -= hit.capacity;
            g.refresh_expiry();
        }
    }
    if (hit.block)
    {
        unpoison(hit.block, hit.capacity);
        return HostBlock(hit.block, hit.capacity);
    }
    return HostBlock(map_block(bytes), bytes);
}

void HostBlock::reset() noexcept
{
    std::byte* const block = std::exchange(m_data, nullptr);
    const u64 capacity = std::exchange(m_capacity, 0);
    if (!block)
        return;
    if (capacity <= k_thread_block_max)
    {
        if (ThreadTier* tier = ThreadTier::get())
        {
            tier->put(block, capacity);
            return;
        }
        free_block(block, capacity);
        return;
    }
    if (capacity > k_global_block_max)
    {
        free_block(block, capacity);
        return;
    }
    std::array<Entry, k_global_blocks + 1> evicted{};
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
                poison(block, capacity);
                *free_slot = Entry{block, capacity, ++g.clock, g.acquires.load(std::memory_order_relaxed)};
                g.bytes += capacity;
                kept = true;
                break;
            }
            if (!oldest)
                break;
            evicted[n_evicted++] = *oldest;
            g.bytes -= oldest->capacity;
            *oldest = Entry{};
        }
        g.refresh_expiry();
    }
    if (!kept)
        free_block(block, capacity);
    for (usize i = 0; i < n_evicted; ++i)
        free_block(evicted[i].block, evicted[i].capacity);
}

void trim_block_cache() noexcept
{
    if (ThreadTier* tier = ThreadTier::get())
        for (Entry& e : tier->slots)
        {
            if (e.block)
                free_block(e.block, e.capacity);
            e = Entry{};
        }
    std::array<Entry, k_global_blocks> dropped{};
    {
        GlobalTier& g = GlobalTier::get();
        const std::scoped_lock lock(g.mutex);
        dropped = g.slots;
        g.slots = {};
        g.bytes = 0;
        g.refresh_expiry();
    }
    for (const Entry& e : dropped)
        if (e.block)
            free_block(e.block, e.capacity);
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
