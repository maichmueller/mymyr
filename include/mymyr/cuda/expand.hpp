#pragma once
// The device expand: rl::expand on the device, for
// state batches over a task table that live in device memory, writing into device destinations.
//
//   mymyr::cuda::DeviceExpander x(ctx, table);
//   rl::Expansion out{...device pointers, capacity, words, label_width...};
//   x.expand({d_states, N, W, 0}, d_task_ids, out);   // out.total, out.words_needed as rl::expand sets them
//   x.pad(out, N, padded);                             // rl::pad on the device
//
// The results are byte-equal to rl::expand on the same table: the same canonical order (per state by schema, then
// lexicographically by binding), the same rows, labels (binding -1 past the arity; instance-local object ids), parents,
// CSR offsets, goal flags, capacity clipping (rows past the capacity are counted, not written; their widths still
// count) and width rules (a successor wider than out.words is zeroed and out.words_needed reports the width that fits
// all). Lazy slots: atoms the device meets without a slot are interned per chunk in canonical-id order (rl::expand
// interns them in generation order), so a fresh task may number them differently than the CPU would; on the same task
// state the bytes agree.
//
// A batch over a table of several instances (task_ids [rows], device, required) runs in one of two modes (mode()):
//   - Multi: one pass of the multi-instance kernels over the mixed batch (each thread resolves its row's instance;
//     launches per object-bitset width group, concurrently on the expander's auxiliary streams, and row-width bucket,
//     set_launch; rows grouped by instance), when every
//     instance runs completely on the multi-instance kernels (no numeric fluents, CPU-fallback schema, conditional effects or axioms; frozen
//     slots: detail::multi_unsupported);
//   - PerInstance: otherwise each instance's rows are gathered and expanded apart by the single-instance path (below),
//     and the results are scattered back into batch order (one synchronization per instance in count() and write()).
// A table of one instance (Single) runs the single-instance path: the task's kernels, conditional effects and axioms on
// the device, goals over derived predicates too; schemas the kernels cannot run need the parents on the host
// (the batch is copied down first: ChunkGenerator's per-schema CPU fallback), and so do axioms whose body matchers are
// beyond the kernels.
//
// Destination-passing: every array of `out` is device memory the caller owns (null: not written); nothing of the
// caller's is allocated. The work runs on one stream (set_stream; null: the context's stream) in stream order after
// the work already enqueued there, which must have produced the input states and task ids. count() synchronizes once
// (the total and the validation), write() once more (words_needed; lazy slots may add one round per chunk), pad() once
// (overflow).
//
// One DeviceExpander serves one thread at a time; its scratch is reused across calls.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>

namespace mymyr::cuda
{
class DeviceExpander
{
public:
    enum class Mode : u8
    {
        Single,       // a table of one instance
        Multi,        // one pass of the multi-instance kernels
        PerInstance,  // each instance's rows apart
    };

    /// Throws std::invalid_argument for tasks beyond ChunkGenerator::unsupported limits. Numeric rows use each
    /// instance's CPU encoding, padded to the table's numeric width at the public boundary.
    DeviceExpander(ContextPtr ctx, rl::TaskTablePtr table, cudaStream_t stream = nullptr);
    ~DeviceExpander();
    DeviceExpander(const DeviceExpander&) = delete;
    DeviceExpander& operator=(const DeviceExpander&) = delete;

    [[nodiscard]] const ContextPtr& context() const noexcept;
    [[nodiscard]] const rl::TaskTablePtr& table() const noexcept;
    [[nodiscard]] Mode mode() const noexcept;
    /// The stream of the following calls (null: the context's stream).
    void set_stream(cudaStream_t s);
    [[nodiscard]] cudaStream_t stream() const noexcept;
    /// Parents per chunk at most for the following count() (0: automatic, the views of a chunk within 256 MB). The
    /// bytes do not depend on it; tests use tiny chunks.
    void set_chunk_rows(u64 rows) noexcept;
    /// How Multi mode launches the write kernel over the row-width buckets (default PerBucket: the write kernel of rows
    /// narrower than their group's widest bucket runs faster at their own bucket). The bytes do
    /// not depend on it.
    void set_launch(BucketLaunch launch) noexcept;

    /// Validates the batch (device rows; task_ids [rows] device, null only for a table of one instance) and counts its
    /// successors: returns the total. Throws std::invalid_argument like rl::expand for malformed batches, task ids
    /// outside the table or rows with unassigned atom slots (validate), std::length_error beyond 2^31 - 1 successors.
    u64 count(rl::StateBatchView in, const i32* task_ids, const rl::ExpandOptions& options = {});
    /// Writes the expansion of the counted batch (count() first; the batch and its task ids must still hold the same
    /// values) into `out` and sets out.total and out.words_needed. May be called again with other destinations.
    void write(rl::Expansion& out);
    /// count() then write().
    void expand(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, const rl::ExpandOptions& options = {});

    /// The counted batch as a part of a larger batch (cuda::SuiteExpander's domains): row q of the counted batch
    /// is row rows[q] of the larger one, whose CSR offsets are batch_offsets (device arrays; [counted rows] and [larger
    /// rows + 1]).
    struct RowMap
    {
        const u32* rows = nullptr;
        const i32* batch_offsets = nullptr;
    };
    /// The counted batch's CSR offsets [rows + 1] (device) as write() sets them, without the expansion (count() first):
    /// the parts' counts give the larger batch its offsets before any part writes.
    void offsets(i32* out);
    /// write() into the larger batch's outputs `out` (its capacity, widths; out.offsets is not written): successor k of
    /// counted row q goes to row map.batch_offsets[map.rows[q]] + k, its parent label is map.rows[q]. Multi mode puts
    /// the rows there itself (the successor kernel's destination map: no copy of the expansion); Single and PerInstance
    /// write into scratch and scatter it.
    void write(rl::Expansion& out, const RowMap& map);

    /// rl::pad on the device: the padded [rows, K] view of a flat device expansion (it must carry offsets).
    void pad(const rl::Expansion& flat, u64 rows, rl::PaddedExpansion& out);

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::cuda
