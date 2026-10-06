#pragma once
// DeviceTask: a task's version-2 export (rl::device_arrays) uploaded once to the device. The export is one
// 64-byte aligned block of offset-addressed arrays, so the upload is
// one copy and the device view (rl::dev::TaskView) is the host view rebased onto the device block.
//
// Shareable across streams: the upload runs on the context's stream and records an event; acquire(s) makes stream s
// wait for it and records s as a user of the block (the block is freed only after the work enqueued on every user
// stream), so any number of streams and threads may launch kernels over one DeviceTask. The task is immutable on the
// device after the upload. Owned by the user (or by a Python object): no global cache of uploads.

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/task_arrays.hpp"

#include <memory>

namespace mymyr::cuda
{
class DeviceTask
{
    struct Private
    {
    };

public:
    /// Uploads `bundle` (a device_arrays export of version 2 or later) through `ctx`.
    static std::shared_ptr<DeviceTask> upload(ContextPtr ctx, std::shared_ptr<const rl::ArrayBundle> bundle);
    /// Exports the task (device_arrays, the latest version) and uploads it.
    static std::shared_ptr<DeviceTask> upload(ContextPtr ctx, const Task& task);

    DeviceTask(Private, ContextPtr ctx, std::shared_ptr<const rl::ArrayBundle> bundle);
    DeviceTask(const DeviceTask&) = delete;
    DeviceTask& operator=(const DeviceTask&) = delete;

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_block.context(); }
    [[nodiscard]] const rl::ArrayBundle& bundle() const noexcept { return *m_bundle; }
    [[nodiscard]] const std::shared_ptr<const rl::ArrayBundle>& bundle_ptr() const noexcept { return m_bundle; }
    [[nodiscard]] const std::byte* device_block() const noexcept { return static_cast<const std::byte*>(m_block.data()); }
    [[nodiscard]] u64 bytes() const noexcept { return m_bundle->bytes(); }
    [[nodiscard]] u32 version() const noexcept { return static_cast<u32>(m_bundle->scalar("version", 0)); }

    /// The device view (pointers into the device block). Use it on a stream that acquire() was called for.
    [[nodiscard]] const rl::dev::TaskView& view() const noexcept { return m_view; }
    /// Makes `s` wait for the upload and records it as a user of the block; returns view(). Thread-safe.
    const rl::dev::TaskView& acquire(cudaStream_t s) const;
    /// The uses on `s` end with `last` (just recorded on s by the user): the block's free waits for it instead of
    /// touching s (DeviceBuffer::retire_stream), so the user may destroy s.
    void release(cudaStream_t s, std::shared_ptr<const LastUse> last) const;
    /// Records `s` as a user of the block without waiting for the upload (the caller already ordered s after it).
    void record_stream(cudaStream_t s) const { m_block.record_stream(s); }
    /// Keeps the device block alive (DLPack exports of its arrays).
    [[nodiscard]] const DeviceBuffer& block() const noexcept { return m_block; }

    /// Copies the device block back into `dst` (bytes() bytes) and waits: the round-trip check.
    void download(void* dst) const;
    /// Wall time of the upload call, and the H2D copy measured by events on the device (milliseconds).
    [[nodiscard]] double upload_seconds() const noexcept { return m_upload_s; }
    [[nodiscard]] float upload_device_ms() const;

private:
    std::shared_ptr<const rl::ArrayBundle> m_bundle;
    mutable DeviceBuffer m_block;
    rl::dev::TaskView m_view;
    Event m_ready;
    Event m_start{true}, m_end{true};
    double m_upload_s = 0;
};

using DeviceTaskPtr = std::shared_ptr<DeviceTask>;
}  // namespace mymyr::cuda
