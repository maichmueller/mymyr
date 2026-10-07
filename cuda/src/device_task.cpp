// DeviceTask: the one-time upload of a task's version-2 export (include/mymyr/cuda/device_task.hpp).

#include "mymyr/cuda/device_task.hpp"

#include <chrono>
#include <stdexcept>

namespace mymyr::cuda
{
std::shared_ptr<DeviceTask> DeviceTask::upload(ContextPtr ctx, std::shared_ptr<const rl::ArrayBundle> bundle)
{
    if (!ctx || !bundle)
        throw std::invalid_argument("mymyr: DeviceTask::upload: null context or bundle");
    DeviceGuard g(ctx->device());  // the events below belong to the context's device
    return std::make_shared<DeviceTask>(Private{}, std::move(ctx), std::move(bundle));
}

std::shared_ptr<DeviceTask> DeviceTask::upload(ContextPtr ctx, const Task& task)
{
    return upload(std::move(ctx), std::make_shared<const rl::ArrayBundle>(rl::device_arrays(task, rl::k_device_arrays_version)));
}

DeviceTask::DeviceTask(Private, ContextPtr ctx, std::shared_ptr<const rl::ArrayBundle> bundle)
    : m_bundle(std::move(bundle)), m_block(ctx, m_bundle->bytes(), ctx->stream())
{
    const auto t0 = std::chrono::steady_clock::now();
    m_view = rl::task_view(*m_bundle, m_block.data());  // validates the version before anything is copied
    const cudaStream_t s = m_block.stream();
    m_start.record(s);
    check(cudaMemcpyAsync(m_block.data(), m_bundle->block(), m_bundle->bytes(), cudaMemcpyHostToDevice, s),
          "cudaMemcpyAsync (task upload)");
    m_end.record(s);
    m_ready.record(s);
    m_upload_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const rl::dev::TaskView& DeviceTask::acquire(cudaStream_t s) const
{
    if (s != m_block.stream())
        m_ready.wait_on(s);
    m_block.record_stream(s);
    return m_view;
}

void DeviceTask::release(cudaStream_t s, std::shared_ptr<const LastUse> last) const { m_block.retire_stream(s, std::move(last)); }

void DeviceTask::download(void* dst) const
{
    DeviceGuard g(context()->device());
    const cudaStream_t s = m_block.stream();
    check(cudaMemcpyAsync(dst, m_block.data(), m_bundle->bytes(), cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync (download)");
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

float DeviceTask::upload_device_ms() const
{
    m_end.synchronize();
    return m_end.elapsed_ms(m_start);
}
}  // namespace mymyr::cuda
