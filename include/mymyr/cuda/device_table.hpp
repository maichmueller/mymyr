#pragma once
// DeviceTaskTable: a task table (rl/task_table.hpp) on the device: the version-2 exports (rl::device_arrays)
// of all its instances in one 64-byte aligned block, followed by the array of their device views (rl::dev::TaskView[I],
// each instance's view rebased onto its part of the block), uploaded with one copy. The multi-instance kernel launches
// (cuda/lifted.hpp, "several instances") read views[inst[row]].
//
// Shareable across streams like DeviceTask: the upload runs on the context's stream and records an event; acquire(s)
// makes stream s wait for it and records s as a user of the block. Immutable on the device after an upload; refresh()
// re-exports and uploads a new block when an instance's lazy atom slots grew (the old block is freed after the work of
// its user streams).

#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/task_arrays.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mymyr::cuda
{
/// How the successor kernels of a multi-instance batch (write, pick) are launched over the row-width buckets: the
/// kernels hold a successor in u64 y[WB] registers, WB in {2, 4, 16, 64}, the smallest that
/// holds the instance's state words. Widest: one launch per object-bitset width group at the group's widest bucket;
/// PerBucket: one launch per (group, bucket), each over the rows of its bucket. The results do not depend on it.
/// Defaults: DeviceEnv uses Widest (fewer launches; its picks are within a few percent either way), DeviceExpander
/// uses PerBucket (its write kernel runs faster per bucket).
enum class BucketLaunch : u8
{
    Widest,
    PerBucket,
};

class DeviceTaskTable
{
    struct Private
    {
    };

public:
    /// Exports every instance and uploads the table. Throws std::invalid_argument when the device cannot run some
    /// instance (unsupported()).
    static std::shared_ptr<DeviceTaskTable> upload(ContextPtr ctx, rl::TaskTablePtr table);
    /// Empty if the device kernels run every instance, else `instance i: <why>` for the first one they do not (numeric
    /// fluents, more than lifted::k_max_ow object words, states wider than lifted::k_max_words words).
    [[nodiscard]] static std::string unsupported(const rl::TaskTable& table);

    DeviceTaskTable(Private, ContextPtr ctx, rl::TaskTablePtr table);
    DeviceTaskTable(const DeviceTaskTable&) = delete;
    DeviceTaskTable& operator=(const DeviceTaskTable&) = delete;

    [[nodiscard]] const ContextPtr& context() const noexcept { return m_ctx; }
    [[nodiscard]] const rl::TaskTablePtr& table() const noexcept { return m_table; }
    [[nodiscard]] u32 size() const noexcept { return m_table->size(); }
    /// Instance i's device view (pointers into the device block), on the host: the uniform kernel parameter of
    /// single-instance launches.
    [[nodiscard]] const rl::dev::TaskView& view(u32 i) const { return m_views.at(i); }
    /// The views on the device [I] (multi-instance launches).
    [[nodiscard]] const rl::dev::TaskView* device_views() const noexcept;
    [[nodiscard]] const rl::ArrayBundle& bundle(u32 i) const { return *m_bundles.at(i); }
    [[nodiscard]] u64 bytes() const noexcept { return m_block.size(); }
    /// Makes `s` wait for the latest upload and records it as a user of the block. Thread-safe.
    void acquire(cudaStream_t s) const;
    /// Lazy slots: re-exports and uploads the table if an instance's slots grew since the last upload; true if it did.
    /// The views change: re-read them (and acquire the new block) after a refresh.
    bool refresh();
    [[nodiscard]] u32 uploads() const noexcept { return m_uploads; }

private:
    void upload_all();

    ContextPtr m_ctx;
    rl::TaskTablePtr m_table;
    std::vector<std::shared_ptr<const rl::ArrayBundle>> m_bundles;
    std::vector<u32> m_exported_fluent, m_exported_derived;
    std::vector<rl::dev::TaskView> m_views;
    mutable DeviceBuffer m_block;
    u64 m_views_offset = 0;
    Event m_ready;
    u32 m_uploads = 0;
};

using DeviceTaskTablePtr = std::shared_ptr<DeviceTaskTable>;
}  // namespace mymyr::cuda
