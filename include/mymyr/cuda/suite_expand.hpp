#pragma once
// The device expand over a task suite: rl::expand on the device for batches
// whose rows are instances of several domains (rl/task_suite.hpp), driving one DeviceExpander per domain
// (cuda/expand.hpp) through its public interface. Byte-equal to rl::expand on the suite (and so to each row's domain
// table apart): rows per parent in batch order, canonical order within a parent, labels with the domain's schema
// ids and instance-local objects, parents, CSR offsets, goal flags, capacity clipping and width rules.
//
//   cuda::SuiteExpander x(ctx, suite);
//   x.expand({d_states, N, suite->words(), 0}, d_task_ids, out);   // global task ids (device int32)
//
// A suite of one domain (a table: TaskSuite::of) is its table's DeviceExpander: every call forwards, nothing is added.
// Over several domains, count() classifies the rows on the device (cuda/src/suite.cu: each row's domain and local id,
// the rows per domain, whether each domain's rows are one run; the rows are validated as rl::expand does: task ids,
// bits past the instance's assigned slots) and reads that summary back (one synchronization, as DeviceExpander's own):
//   - one run per domain (the rows of each domain are consecutive in the batch, in any domain order): each domain's
//     expander expands its run in place and writes straight into the caller's outputs at the run's first successor
//     (parents and offsets shifted to batch positions);
//   - otherwise each domain's rows are gathered into consecutive scratch rows (a stable sort by domain,
//     lifted::launch_order: batch order within a domain) and counted apart; the domains' CSR offsets
//     (DeviceExpander::offsets) give the batch its offsets before any domain writes, and each domain writes its rows
//     straight at their batch positions at the suite's widths (DeviceExpander::write with a RowMap; zero words and -1
//     labels past the domain's): a Multi domain's successor kernel puts each row at its batch row, a Single or
//     PerInstance domain expands into scratch and scatters it.
// Each domain's expander runs its own mode on its rows (Single, Multi or PerInstance; cuda/expand.hpp).
//
// Destination-passing as DeviceExpander; one SuiteExpander serves one thread at a time.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/expand.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_suite.hpp"

#include <memory>

namespace mymyr::cuda
{
class SuiteExpander
{
public:
    /// Throws std::invalid_argument for suites the device cannot run (a domain's DeviceTaskTable::unsupported, rows
    /// wider than the kernels take).
    SuiteExpander(ContextPtr ctx, rl::TaskSuitePtr suite, cudaStream_t stream = nullptr);
    ~SuiteExpander();
    SuiteExpander(const SuiteExpander&) = delete;
    SuiteExpander& operator=(const SuiteExpander&) = delete;

    /// Empty if the device runs every domain of the suite, else why not ("domain d: ...").
    [[nodiscard]] static std::string unsupported(const rl::TaskSuite& suite);

    [[nodiscard]] const ContextPtr& context() const noexcept;
    [[nodiscard]] const rl::TaskSuitePtr& suite() const noexcept;
    /// The domain's expander (created with the suite).
    [[nodiscard]] DeviceExpander& domain(u32 d) const;
    /// The stream of the following calls (null: the context's stream).
    void set_stream(cudaStream_t s);
    [[nodiscard]] cudaStream_t stream() const noexcept;
    /// DeviceExpander::set_chunk_rows / set_launch of every domain's expander.
    void set_chunk_rows(u64 rows) noexcept;
    void set_launch(BucketLaunch launch) noexcept;

    /// DeviceExpander::count over the suite (task_ids [rows] device, global ids; null only for a table of one).
    u64 count(rl::StateBatchView in, const i32* task_ids, const rl::ExpandOptions& options = {});
    /// DeviceExpander::write (count() first; may be called again with other destinations).
    void write(rl::Expansion& out);
    /// count() then write().
    void expand(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, const rl::ExpandOptions& options = {});
    /// rl::pad on the device (DeviceExpander::pad).
    void pad(const rl::Expansion& flat, u64 rows, rl::PaddedExpansion& out);

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::cuda
