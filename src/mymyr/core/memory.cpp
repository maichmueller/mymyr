#include "mymyr/core/memory.hpp"

#include <atomic>
#include <cstdint>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace mymyr
{
namespace
{
class HostMemory final : public MemoryResource
{
public:
    void* allocate(usize bytes, usize align) override
    {
        return ::operator new(bytes == 0 ? 1 : bytes, std::align_val_t{align < k_cache_line ? k_cache_line : align});
    }
    void deallocate(void* p, usize, usize align) noexcept override
    {
        ::operator delete(p, std::align_val_t{align < k_cache_line ? k_cache_line : align});
    }
    MemKind kind() const noexcept override { return MemKind::Host; }
};
}  // namespace

MemoryResource* host_memory() noexcept
{
    static HostMemory resource;
    return &resource;
}

void advise_huge_pages([[maybe_unused]] void* p, [[maybe_unused]] usize bytes) noexcept
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr std::uintptr_t page = 4096, huge = std::uintptr_t{2} << 20;
    const auto begin = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t first = (begin + page - 1) & ~(page - 1), last = (begin + bytes) & ~(page - 1);
    if (p && last >= first + huge)
        ::madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE);  // a hint: failure changes nothing
#endif
}

u64 next_uid() noexcept
{
    static std::atomic<u64> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace mymyr
