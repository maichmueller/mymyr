#pragma once
// Memory resources: every arena allocates through one of these, so the same container code can live in host,
// pinned, device or managed memory. Only the host resource exists in the CPU library; the CUDA plugin
// (libmymyr_cuda) provides the others. Allocation is never on a hot path (arenas grow by whole segments), so a
// virtual interface is fine here.

#include "mymyr/core/types.hpp"

#include <iterator>
#include <new>
#include <vector>

namespace mymyr
{
enum class MemKind : u8
{
    Host,
    Pinned,
    Device,
    Managed,
};

class MemoryResource
{
public:
    virtual ~MemoryResource() = default;
    [[nodiscard]] virtual void* allocate(usize bytes, usize align) = 0;
    virtual void deallocate(void* p, usize bytes, usize align) noexcept = 0;
    [[nodiscard]] virtual MemKind kind() const noexcept = 0;
};

/// Stateless aligned host allocator (the analogue of std::pmr::new_delete_resource()).
[[nodiscard]] MemoryResource* host_memory() noexcept;

inline constexpr usize k_cache_line = 64;

/// Allocator for per-thread scratch written on hot paths: every buffer starts on a cache line and owns whole lines,
/// so one thread's writes never invalidate a line another thread reads. Plain malloc does not guarantee this: glibc's
/// per-thread cache hands a thread small chunks that other threads freed, from the middle of their arenas, which can
/// cause false sharing between threads under concurrent workloads on a shared task.
template<class T>
struct LineAllocator
{
    using value_type = T;
    static constexpr usize k_align = alignof(T) > k_cache_line ? alignof(T) : k_cache_line;

    LineAllocator() noexcept = default;
    template<class U>
    LineAllocator(const LineAllocator<U>&) noexcept
    {
    }

    [[nodiscard]] T* allocate(usize n)
    {
        return static_cast<T*>(::operator new(round(n), std::align_val_t{k_align}));
    }
    void deallocate(T* p, usize n) noexcept { ::operator delete(p, round(n), std::align_val_t{k_align}); }

    template<class U>
    friend bool operator==(const LineAllocator&, const LineAllocator<U>&) noexcept
    {
        return true;
    }

private:
    static usize round(usize n) noexcept { return (n * sizeof(T) + k_align - 1) / k_align * k_align; }
};

/// A vector of per-thread hot scratch (see LineAllocator).
template<class T>
using LineVector = std::vector<T, LineAllocator<T>>;

/// Hints that the whole huge pages inside [p, p + bytes) be backed by transparent huge pages, where the system offers
/// them (Linux): filling fresh memory then takes a page fault per 2 MiB instead of per 4 KiB. Contents are unchanged;
/// ranges shorter than a huge page and systems without the hint are left alone.
void advise_huge_pages(void* p, usize bytes) noexcept;

/// Makes room for at least `n` elements of `v`, growing its capacity at least twofold. A new buffer of at least a huge
/// page is advised for huge pages (advise_huge_pages) before the elements move in, so neither the move nor the
/// later appends fault per 4 KiB page. For vectors that grow to many megabytes while being filled.
template<class T>
void reserve_huge(std::vector<T>& v, usize n)
{
    if (n <= v.capacity())
        return;
    std::vector<T> fresh;
    fresh.reserve(n > v.capacity() * 2 ? n : v.capacity() * 2);
    advise_huge_pages(fresh.data(), fresh.capacity() * sizeof(T));
    fresh.insert(fresh.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    v.swap(fresh);
}

/// Process-wide unique, never reused 64-bit ids for tasks, stores and registries. This atomic counter is the only
/// process-global state in mymyr: everything else is owned by the object that uses it, with no global repositories
/// or pools.
[[nodiscard]] u64 next_uid() noexcept;
}  // namespace mymyr
