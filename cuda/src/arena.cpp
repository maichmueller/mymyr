// DeviceArena: append-only device records with tail syncs to a pinned mirror (include/mymyr/cuda/arena.hpp).

#include "mymyr/cuda/arena.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mymyr::cuda
{
DeviceArena::DeviceArena(ContextPtr ctx, u32 record_bytes, u64 capacity, bool host_mirror)
    : m_ctx(std::move(ctx)), m_rb(record_bytes)
{
    if (!m_ctx || record_bytes == 0)
        throw std::invalid_argument("mymyr: DeviceArena: null context or zero record size");
    DeviceGuard g(m_ctx->device());
    m_capacity = std::max<u64>(capacity, 1);
    m_gen = std::make_shared<DeviceBuffer>(m_ctx, m_capacity * m_rb, m_ctx->stream());
    if (host_mirror)
        m_mirror = std::make_shared<PinnedBuffer>(m_capacity * m_rb);
}

DeviceArena::~DeviceArena()
{
    try
    {
        wait();  // the mirror must outlive the copies into it
    }
    catch (...)
    {
    }
}

Event DeviceArena::take_event()
{
    if (m_spare.empty())
        return Event{};
    Event e = std::move(m_spare.back());
    m_spare.pop_back();
    return e;
}

void DeviceArena::reserve(u64 more)
{
    if (m_device_size + more <= m_capacity)
        return;
    DeviceGuard g(m_ctx->device());
    wait();  // pending copies read the old generation and write the old mirror
    const u64 cap = std::max(m_device_size + more, 2 * m_capacity);
    auto gen = std::make_shared<DeviceBuffer>(m_ctx, cap * m_rb, m_ctx->stream());
    if (m_device_size)
        check(cudaMemcpyAsync(gen->data(), m_gen->data(), m_device_size * m_rb, cudaMemcpyDeviceToDevice, m_ctx->stream()),
              "cudaMemcpyAsync (arena growth)");
    m_gen = std::move(gen);  // the old generation is freed after the copy (its stream), unless an export holds it
    if (m_mirror)
    {
        auto mirror = std::make_shared<PinnedBuffer>(cap * m_rb);
        if (m_host_size)
            std::memcpy(mirror->data(), m_mirror->data(), m_host_size * m_rb);
        m_mirror = std::move(mirror);
    }
    m_capacity = cap;
}

std::byte* DeviceArena::tail(u64 n)
{
    reserve(n);
    return device_data() + m_device_size * m_rb;
}

void DeviceArena::commit(u64 n)
{
    if (m_device_size + n > m_capacity)
        throw std::out_of_range("mymyr: DeviceArena::commit beyond the reserved tail");
    m_device_size += n;
}

void DeviceArena::append_from_host(const void* src, u64 n)
{
    if (n == 0)
        return;
    std::byte* dst = tail(n);
    DeviceGuard g(m_ctx->device());
    check(cudaMemcpyAsync(dst, src, n * m_rb, cudaMemcpyHostToDevice, m_ctx->stream()), "cudaMemcpyAsync (arena append)");
    // a pageable source is staged before the call returns; a pinned one is read asynchronously: order the caller
    // after it so `src` may be released on return in both cases
    check(cudaStreamSynchronize(m_ctx->stream()), "cudaStreamSynchronize");
    commit(n);
}

void DeviceArena::append_from_device(const void* src, u64 n, cudaStream_t producer)
{
    if (n == 0)
        return;
    std::byte* dst = tail(n);
    DeviceGuard g(m_ctx->device());
    stream_wait(m_ctx->stream(), producer);
    check(cudaMemcpyAsync(dst, src, n * m_rb, cudaMemcpyDeviceToDevice, m_ctx->stream()), "cudaMemcpyAsync (arena append)");
    commit(n);
}

void DeviceArena::sync_to_host(cudaStream_t after)
{
    if (!m_mirror)
        throw std::logic_error("mymyr: DeviceArena::sync_to_host: the arena has no host mirror");
    const u64 lo = m_issued, hi = m_device_size;
    if (lo == hi)
        return;
    DeviceGuard g(m_ctx->device());
    const cudaStream_t copy = m_ctx->copy_stream();
    m_after.record(after ? after : m_ctx->stream());
    m_after.wait_on(copy);  // the wait captures the event now: m_after may be re-recorded afterwards
    check(cudaMemcpyAsync(static_cast<std::byte*>(m_mirror->data()) + lo * m_rb, device_data() + lo * m_rb, (hi - lo) * m_rb,
                          cudaMemcpyDeviceToHost, copy),
          "cudaMemcpyAsync (tail sync)");
    Event done = take_event();
    done.record(copy);
    m_gen->record_stream(copy);
    m_pending.push_back({std::move(done), hi});
    m_issued = hi;
}

bool DeviceArena::poll()
{
    while (!m_pending.empty() && m_pending.front().done.done())
    {
        m_host_size = m_pending.front().upto;
        m_spare.push_back(std::move(m_pending.front().done));
        m_pending.pop_front();
    }
    return m_pending.empty();
}

void DeviceArena::wait()
{
    if (m_pending.empty())
        return;
    m_pending.back().done.synchronize();  // the copy stream is in order: the last one landing means all did
    poll();
}
}  // namespace mymyr::cuda
