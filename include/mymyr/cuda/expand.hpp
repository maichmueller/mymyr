#pragma once
// The device expand: rl::expand on the device, for
// state batches over a task table that live in device memory, writing into device destinations.
//
//   mymyr::cuda::DeviceExpander x(ctx, table);
//   rl::Expansion out{...device pointers, capacity, words, label_width...};
//   x.expand({d_states, N, W, 0}, d_task_ids, out);   // out.total, generated, words_needed as rl::expand does
//   x.pad(out, N, padded);                             // rl::pad on the device
//
// The results are byte-equal to rl::expand on the same table: the same canonical order (per state by schema, then
// lexicographically by binding), the same rows, labels (binding -1 past the arity; instance-local object ids), parents,
// CSR offsets, goal flags, capacity clipping (rows past the capacity are counted, not written; their widths still
// count) and width rules (a successor wider than out.words is zeroed and out.words_needed reports the width that fits
// all).
//
// A batch over a table of several instances (task_ids [rows], device, required) runs in one of two modes (mode()):
//   - Multi: one pass of the multi-instance kernels over the mixed batch (each thread resolves its row's instance;
//     launches per object-bitset width group, concurrently on the expander's auxiliary streams, and row-width bucket,
//     set_launch; rows grouped by instance), when every
//     instance runs completely on the multi-instance kernels (no numeric fluents, CPU-fallback schema, conditional effects or axioms:
//     detail::multi_unsupported);
//   - PerInstance: otherwise each instance's rows are gathered and expanded apart by the single-instance path (below),
//     and the results are scattered back into batch order (one synchronization per instance in count() and write()).
// A table of one instance (Single) runs the single-instance path: the task's kernels, conditional effects and axioms on
// the device, goals over derived predicates too; schemas the kernels cannot run need the parents on the host
// (the batch is copied down first: ChunkGenerator's per-schema CPU fallback), and so do axioms whose body matchers are
// beyond the kernels. An instance the kernels cannot run at all (ChunkGenerator::unsupported: more than 2048 objects,
// numeric states wider than 64 words) is expanded on the host (Host; in a table of several instances, a part of
// PerInstance): its rows are copied down, rl::expand runs on the host's threads and the results are copied up (count()
// and write() synchronize). So are a numeric instance's batches whose rows (the batch's atom words and the numeric
// slots) are wider than 64 words.
//
// Optional succ_hash/noop destinations digest the actual successor while it is built, including when succ is null.
// Hashes equal State::hash(), with public packed numeric words; noop compares fluent and numeric bits. With
// drop_noops, a stable compaction keeps effective successors, or all successors of an all-no-op parent. The index
// destination carries the kept rows' positions in each parent's full expansion. generated counts before compaction,
// total and offsets after it; capacity bounds generated. Capturable calls retain compaction scratch across replays.
//
// Derived words (rl::Expansion::derived): every mode fills them as rl::expand does. A Single task evaluates the
// successors' axioms on the device (on the CPU for axioms beyond the kernels); a Multi table has no axioms, so its
// words are zero; PerInstance parts are scattered at the caller's width. A mapped write in Multi mode does not write
// out.derived: the caller zeroes those rows first.
//
// Destination-passing: every array of `out` is device memory the caller owns (null: not written); nothing of the
// caller's is allocated. The work runs on one stream (set_stream; null: the context's stream) in stream order after
// the work already enqueued there, which must have produced the input states and task ids. count() synchronizes once
// (the total and the validation), write() once more (words_needed; lazy slots may add one round per chunk), pad() once
// (overflow). expand_async() synchronizes nothing: into destinations of a fixed capacity, it leaves the total, the
// overflow and the validation on the device (Single and Multi batches of one chunk whose work stays on the device).
//
// One DeviceExpander serves one thread at a time; its scratch is reused across calls.

#include "mymyr/cuda/device_table.hpp"
#include "mymyr/cuda/runtime.hpp"
#include "mymyr/rl/expand.hpp"
#include "mymyr/rl/task_table.hpp"

#include <memory>
#include <string>

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
        Host,         // a table of one instance the device kernels cannot run: rl::expand on the host's threads
    };

    /// Numeric rows use each instance's CPU encoding, padded to the table's numeric width at the public boundary.
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
    /// values) into `out` and sets out.total, out.generated and out.words_needed as rl::expand does (generated equals
    /// the total without drop_noops). May be called again with other destinations.
    void write(rl::Expansion& out);
    /// count() then write().
    void expand(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, const rl::ExpandOptions& options = {});
    /// After a write() or expand_async() with drop_noops: the device CSR offsets [rows + 1] of the successors before
    /// dropping (offsets[i + 1] - offsets[i] counts state i's generated successors), on this expander's stream and
    /// valid until its next call; null after a call without drop_noops.
    [[nodiscard]] const i32* generated_offsets() const noexcept;

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
    /// write into scratch and scatter it (their derived rows are scattered zero-padded to out.derived_words). Multi mode
    /// does not write out.derived (its instances have no axioms): the caller zeroes those rows first, as SuiteExpander does.
    void write(rl::Expansion& out, const RowMap& map);

    /// Why expand_async() of a batch of `rows` rows would need the host (empty: it does not): an instance the kernels
    /// cannot run, instances expanded apart (PerInstance), numeric rows, CPU-fallback schemas or axioms on the CPU, or
    /// more rows than one chunk of views holds.
    [[nodiscard]] std::string async_unsupported(u64 rows, const rl::ExpandOptions& options = {}) const;
    /// The expansion of a batch into destinations of a fixed capacity, without waiting for the device: count() and
    /// write() in stream order, with what they read back kept on the device. out.offsets is required and gets the true
    /// counts while the total is at most 2^31 - 1 (offsets[rows] is the total, even past the capacity); above that,
    /// the i32 offsets are unusable and only status bit 0 reports the overflow. Rows past the capacity are not written.
    /// *status (a device u32, required) gets bit 0 when the successors exceed the capacity, bit 1 when the batch is
    /// invalid (a task id outside the table, a row with bits its instance has no slot for, a successor wider than
    /// out.words); an invalid batch writes no successor or label. Once the running total saturates at UINT32_MAX,
    /// successor widths past it are not counted: bit 1's width check is exact only while the batch fits; bit 0 is
    /// always exact. Successor rows past the total are zero, and goal flags and derived words cover every row below
    /// the capacity (past the total: those of zero rows). out.total, out.generated and out.words_needed are not set,
    /// and a later write() needs a count() first. With rows_dev (device u32), only min(*rows_dev, in.rows) rows are
    /// present; padding contributes neither successors nor validation errors, and its offsets equal the live total.
    /// Needs async_unsupported() empty (std::invalid_argument otherwise).
    void expand_async(rl::StateBatchView in, const i32* task_ids, rl::Expansion& out, u32* status,
                      const rl::ExpandOptions& options = {}, const u32* rows_dev = nullptr);

    /// rl::pad on the device: the padded [rows, K] view of a flat device expansion (it must carry offsets).
    void pad(const rl::Expansion& flat, u64 rows, rl::PaddedExpansion& out);

    struct Impl;

private:
    std::unique_ptr<Impl> m;
};
}  // namespace mymyr::cuda
