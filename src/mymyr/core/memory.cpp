#include "mymyr/core/memory.hpp"

#include <atomic>
#include <new>

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

u64 next_uid() noexcept
{
    static std::atomic<u64> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace mymyr
