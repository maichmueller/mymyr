#pragma once
// Device kernels of the RL helpers (rl/novelty.hpp, rl/prefix.hpp, rl/her.hpp): destination-passing launchers
// (caller-owned device buffers, one stream, no allocation, no synchronization) running the element functions of
// the host loops, so both give the same bytes. Masks are zeroed on the stream first, then scattered (writes of 1 in any
// order); novelty rows and relabels are independent threads.
//
// Device-code subset: included by .cu files compiled by nvcc as C++23; POD, raw pointers and cudaStream_t.

#include "mymyr/rl/her.hpp"
#include "mymyr/core/hash.hpp"
#include "mymyr/rl/match.hpp"
#include "mymyr/rl/novelty.hpp"
#include "mymyr/rl/prefix.hpp"
#include "mymyr/rl/visited.hpp"

#include <cuda_runtime_api.h>

namespace mymyr::cuda
{
/// Device destinations for a flat expansion. Widths include the numeric tail; null arrays are skipped.
struct ExpansionRows
{
    u64 capacity = 0;
    u32 words = 0, label_width = 0, derived_words = 0;
    u64* succ = nullptr;
    i32* parent = nullptr;
    i32* schema = nullptr;
    i32* binding = nullptr;
    u8* goal = nullptr;
    u64* derived = nullptr;
    u64* hash = nullptr;
    u8* noop = nullptr;
    i32* index = nullptr;
    i32* offsets = nullptr;
};
/// Writes a generated count on the stream without a pageable host transfer.
cudaError_t launch_expansion_generated(u32 value, u32* out, cudaStream_t s);
/// Native state hashes, with the numeric block following `words` atom words in each row.
cudaError_t launch_hash_rows(const u64* states, u64 rows, u64 stride, u32 words, u32 numeric_words,
                             u64* out, cudaStream_t s);
/// Kept counts per parent (all-no-op parents keep all rows); counts[rows] = 0 for an exclusive scan. When the
/// pre-drop expansion overflows its capacity (or `status` is nonzero), every parent keeps all of its rows.
cudaError_t launch_noop_counts(ExpansionRows src, u64 rows, u32* counts, cudaStream_t s, const u32* status = nullptr);
/// Stable scatter into scanned offsets, including canonical indices and cleared padding. Writes nothing when
/// `status` has bit 1 (an invalid batch).
cudaError_t launch_noop_scatter(ExpansionRows src, ExpansionRows dst, u64 rows, cudaStream_t s,
                                const u32* status = nullptr);
/// Canonical positions without dropping, and zero digest padding beyond the live total. Writes nothing when
/// `status` has bit 1 (an invalid batch).
cudaError_t launch_expansion_indices(ExpansionRows out, u64 rows, cudaStream_t s, const u32* status = nullptr);
/// counts[i] = offsets[i + 1] - offsets[i] for i < rows.
cudaError_t launch_segment_counts(const i32* offsets, u64 rows, i32* counts, cudaStream_t s);
/// rl::visited_mask on device arrays (one thread per successor): out[j] = whether succ_hash[j] is in the visited table
/// (table [table_rows, capacity], count [table_rows]) of row parent[j], or of rows[parent[j]] with `rows` ([parents]
/// entries; without it parents = table_rows). A parent or mapped row of -1 is padding (false); one outside [-1,
/// parents) or [-1, table_rows) reads as not visited.
cudaError_t launch_visited_mask(const u64* table, const i32* count, u64 table_rows, u32 capacity, const u64* succ_hash,
                                const i32* parent, const i32* rows, u64 parents, u64 size, u8* out, cudaStream_t s);
/// rl::target_visited on device arrays, one thread per target: out [x.targets].
cudaError_t launch_target_visited(const rl::TargetRows<i32>& x, const u64* table, const i32* count, u64 table_rows,
                                  u32 capacity, u8* out, cudaStream_t s);
cudaError_t launch_target_visited(const rl::TargetRows<i64>& x, const u64* table, const i32* count, u64 table_rows,
                                  u32 capacity, u8* out, cudaStream_t s);
/// Lifted goal matching on caller-owned arrays; no allocation or host synchronization.
cudaError_t launch_match_first(const rl::MatchTemplates& templates, const rl::MatchBatch& batch, cudaStream_t s);

/// Lifted hindsight relabelling with fixed caller-owned scratch; no allocation or host synchronization.
cudaError_t launch_her_relabel_lifted(const rl::MatchTemplates& templates, const rl::HerLiftedBatch& batch,
                                      const rl::HerLiftedConfig& config, cudaStream_t s);

/// rl::novelty_update on device arrays: seen [rows, words] (in place), states [rows, state_stride], reward [rows].
cudaError_t launch_novelty_update(u64* seen, const u64* states, u64 rows, u32 words, u32 state_stride, i32* reward,
                                  cudaStream_t s);
/// rl::prefix_masks on device arrays (x, q point to device memory; the structs themselves are passed by value).
cudaError_t launch_prefix_masks(const rl::LabelRows& x, const rl::PrefixQuery& q, u8* mask, cudaStream_t s);
/// rl::schema_masks on device arrays.
cudaError_t launch_schema_masks(const rl::LabelRows& x, u64 rows, u32 num_schemas, u8* mask, cudaStream_t s);
/// rl::her_relabel on device arrays (one thread per relabel).
cudaError_t launch_her_relabel(const rl::HerBatch& b, const rl::HerConfig& c, cudaStream_t s);
}  // namespace mymyr::cuda
