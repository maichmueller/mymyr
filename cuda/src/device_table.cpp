// DeviceTaskTable: a task table's instances uploaded in one block (include/mymyr/cuda/device_table.hpp).

#include "mymyr/cuda/device_table.hpp"

#include "mymyr/cuda/generator.hpp"

#include <cstring>
#include <stdexcept>

namespace mymyr::cuda
{
std::shared_ptr<DeviceTaskTable> DeviceTaskTable::upload(ContextPtr ctx, rl::TaskTablePtr table)
{
    if (!ctx || !table)
        throw std::invalid_argument("mymyr: DeviceTaskTable: null context or table");
    if (const std::string why = unsupported(*table); !why.empty())
        throw std::invalid_argument("mymyr: the CUDA backend cannot run " + why);
    DeviceGuard g(ctx->device());
    return std::make_shared<DeviceTaskTable>(Private{}, std::move(ctx), std::move(table));
}

std::string DeviceTaskTable::unsupported(const rl::TaskTable& table)
{
    if (table.numeric())
        return "numeric task tables: device multi-instance kernels require atom-only states";
    for (u32 i = 0; i < table.size(); ++i)
        if (const std::string why = ChunkGenerator::unsupported(*table.task(i)); !why.empty())
            return "instance " + std::to_string(i) + " of this table: " + why;
    return {};
}

DeviceTaskTable::DeviceTaskTable(Private, ContextPtr ctx, rl::TaskTablePtr table)
    : m_ctx(std::move(ctx)), m_table(std::move(table))
{
    upload_all();
}

void DeviceTaskTable::upload_all()
{
    const u32 I = m_table->size();
    m_bundles.clear();
    m_exported_fluent.assign(I, 0);
    m_exported_derived.assign(I, 0);
    std::vector<u64> at(I);
    u64 bytes = 0;
    for (u32 i = 0; i < I; ++i)
    {
        auto b = std::make_shared<const rl::ArrayBundle>(rl::device_arrays(*m_table->task(i), rl::k_device_arrays_version));
        m_exported_fluent[i] = static_cast<u32>(b->scalar("plan_fluent_slots", 0));
        m_exported_derived[i] = static_cast<u32>(b->scalar("plan_derived_slots", 0));
        at[i] = bytes;
        bytes += (b->bytes() + 63) & ~u64{63};
        m_bundles.push_back(std::move(b));
    }
    m_views_offset = bytes;
    bytes += u64{I} * sizeof(rl::dev::TaskView);
    const cudaStream_t s = m_ctx->stream();
    DeviceBuffer block(m_ctx, bytes, s);  // the previous block (if any) is freed after its users' work
    auto* base = static_cast<std::byte*>(block.data());
    m_views.clear();
    for (u32 i = 0; i < I; ++i)
        m_views.push_back(rl::task_view(*m_bundles[i], base + at[i]));
    // one staging copy of the whole block (pinned, so the copy is asynchronous; freed after it)
    PinnedBuffer staging(bytes);
    auto* h = static_cast<std::byte*>(staging.data());
    for (u32 i = 0; i < I; ++i)
        std::memcpy(h + at[i], m_bundles[i]->block(), m_bundles[i]->bytes());
    std::memcpy(h + m_views_offset, m_views.data(), u64{I} * sizeof(rl::dev::TaskView));
    check(cudaMemcpyAsync(block.data(), h, bytes, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync (task table upload)");
    m_ready.record(s);
    check(cudaStreamSynchronize(s), "cudaStreamSynchronize");  // the staging buffer goes
    m_block = std::move(block);
    ++m_uploads;
}

const rl::dev::TaskView* DeviceTaskTable::device_views() const noexcept
{
    return reinterpret_cast<const rl::dev::TaskView*>(static_cast<const std::byte*>(m_block.data()) + m_views_offset);
}

void DeviceTaskTable::acquire(cudaStream_t s) const
{
    if (s != m_block.stream())
        m_ready.wait_on(s);
    m_block.record_stream(s);
}

bool DeviceTaskTable::refresh()
{
    bool grew = false;
    for (u32 i = 0; i < m_table->size() && !grew; ++i)
    {
        const AtomIndex& a = m_table->task(i)->atoms();
        grew = a.fluent_slots() != m_exported_fluent[i] || a.derived_slots() != m_exported_derived[i];
    }
    if (!grew)
        return false;
    DeviceGuard g(m_ctx->device());
    upload_all();
    return true;
}
}  // namespace mymyr::cuda
