#pragma once
// DeviceArena: an append-only array of fixed-size records on the device with a pinned host mirror synchronized by
// tail copies, with most of the copy latency hidden behind the next layer's work.
//
//   - Watermarks: device_size() records exist on the device (the writer's count; kernels append at tail() and the
//     driver commits), host_size() records are valid in the mirror. sync_to_host() copies only the tail
//     [synced_upto(), device_size()) on the context's copy stream, after the work enqueued on the writer's stream, and
//     returns at once; poll() / wait() advance host_size() when the copy has landed. Several syncs may be in flight
//     (the copy stream keeps them in order).
//   - One writer: the records are written on one stream (the context's compute stream unless the caller orders its
//     own stream first). The arena object is not thread-safe.
//   - Growth is a batch-boundary event: reserve() allocates a new device generation (stream-ordered copy of the
//     records) and, when the mirror is full, a mirror of at least twice its size (after waiting for the syncs in
//     flight). Exports hold the generation they point into (generation()), so a grown arena never invalidates an
//     exported slice. Stable addresses without copies (VMM) are a later refinement.

#include "mymyr/cuda/runtime.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace mymyr::cuda
{
class DeviceArena
{
public:
    /// `host_mirror` false: no pinned mirror (sync_to_host() throws; growth then copies only on the device, which
    /// matters for large arenas: pinning memory costs milliseconds per 100 MB). Throws std::length_error when
    /// capacity x record_bytes exceeds 2^64 - 1 (and the context's error when the allocation fails).
    DeviceArena(ContextPtr ctx, u32 record_bytes, u64 capacity = 1024, bool host_mirror = true);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] u32 record_bytes() const noexcept { return m_rb; }
    [[nodiscard]] u64 capacity() const noexcept { return m_capacity; }
    [[nodiscard]] u64 device_size() const noexcept { return m_device_size; }
    /// Records valid in the mirror (advanced by poll() and wait()).
    [[nodiscard]] u64 host_size() const noexcept { return m_host_size; }
    /// End of the last sync issued (host_size() once every sync in flight has landed).
    [[nodiscard]] u64 synced_upto() const noexcept { return m_issued; }
    [[nodiscard]] usize syncs_in_flight() const noexcept { return m_pending.size(); }
    /// Device generations allocated by reserve() since construction, and mirrors allocated by it.
    [[nodiscard]] u64 growths() const noexcept { return m_growths; }
    [[nodiscard]] u64 mirror_growths() const noexcept { return m_mirror_growths; }
    /// The writer's stream.
    [[nodiscard]] cudaStream_t stream() const noexcept { return m_ctx->stream(); }

    /// The current device generation: records [0, device_size()). Its address changes when the arena grows.
    [[nodiscard]] std::byte* device_data() const noexcept { return static_cast<std::byte*>(m_gen->data()); }
    [[nodiscard]] const std::shared_ptr<DeviceBuffer>& generation() const noexcept { return m_gen; }
    /// The pinned mirror: records [0, host_size()) are valid (null without a mirror).
    [[nodiscard]] const std::byte* host_data() const noexcept
    {
        return m_mirror ? static_cast<const std::byte*>(m_mirror->data()) : nullptr;
    }
    [[nodiscard]] bool has_mirror() const noexcept { return m_mirror != nullptr; }
    /// The mirror generation (exports of host_data() hold it: growth allocates a new mirror).
    [[nodiscard]] const std::shared_ptr<PinnedBuffer>& mirror() const noexcept { return m_mirror; }

    /// Makes room for `more` records past device_size() (a new generation when full; capacity at least doubles, but
    /// stays at or below `most` records when that holds device_size() + more). The new generation is allocated while
    /// the old one is still held. The mirror grows with it to at least twice its own capacity, so that it is
    /// reallocated a logarithmic number of times however small the device steps are. Throws std::length_error, leaving the arena unchanged, when the records or their
    /// bytes exceed 2^64 - 1 (and the context's error, leaving it unchanged, when the allocation fails).
    void reserve(u64 more, u64 most = ~u64{0});
    /// The bytes reserve(more, most) allocates (0: the records fit).
    [[nodiscard]] u64 growth_bytes(u64 more, u64 most = ~u64{0}) const;
    /// Pointer to records [device_size(), device_size() + n) for a kernel on stream() to fill; then commit(n).
    [[nodiscard]] std::byte* tail(u64 n);
    void commit(u64 n);
    /// Drops the records [n, device_size()) (no-op when n >= device_size()); waits for the syncs in flight first.
    void truncate(u64 n);
    /// Appends n records from host memory (H2D on stream(); `src` may be released when the call returns).
    void append_from_host(const void* src, u64 n);
    /// Appends n records from device memory written on `producer` (D2D on stream(), after producer's work).
    void append_from_device(const void* src, u64 n, cudaStream_t producer);

    /// Starts copying the tail to the mirror on the copy stream after the work enqueued so far on `after` (nullptr:
    /// stream()). Returns immediately; a no-op when there is no new tail.
    void sync_to_host(cudaStream_t after = nullptr);
    /// Advances host_size() over the syncs that have landed; true when none is in flight.
    bool poll();
    /// Blocks until every sync in flight has landed.
    void wait();

private:
    struct Pending
    {
        Event done;
        u64 upto;
    };
    Event take_event();
    [[nodiscard]] u64 grown_capacity(u64 more, u64 most) const;

    ContextPtr m_ctx;
    u32 m_rb;
    u64 m_capacity = 0;
    u64 m_mirror_capacity = 0;  // records the mirror holds (at least m_capacity)
    u64 m_growths = 0;
    u64 m_mirror_growths = 0;
    u64 m_device_size = 0;
    u64 m_host_size = 0;
    u64 m_issued = 0;
    std::shared_ptr<DeviceBuffer> m_gen;
    std::shared_ptr<PinnedBuffer> m_mirror;
    Event m_after;
    std::deque<Pending> m_pending;
    std::vector<Event> m_spare;
};
}  // namespace mymyr::cuda
