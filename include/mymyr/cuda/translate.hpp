#pragma once

#include "mymyr/core/types.hpp"
#include <cuda_runtime_api.h>

namespace mymyr::cuda
{
/// Offsets into concatenated fluent maps, numeric maps, constant masks and initial numeric words.
struct TranslateInstance
{
    u32 slots_begin = 0, slots = 0, numeric_begin = 0, numeric_slots = 0;
    u32 constant_begin = 0, initial_begin = 0, target_slots = 0, target_numeric_slots = 0;
    u32 missing_constants = 0;
    u32 source_words = 0, target_words = 0, source_i32 = 0, target_i32 = 0;
    f64 quantum = 0;
    bool tolerant = false;
};

/// Device pointers; instance i translates into instance i. Numeric maps use source slot -> target slot.
struct TranslateView
{
    const TranslateInstance* instances = nullptr;
    const u32* slots = nullptr;
    const u32* numeric = nullptr;
    const u64* constants = nullptr;
    const u64* initial = nullptr;
    u32 count = 0;
};

/// Name-based row translation without allocation or host reads. status is int32 [3]: flags over all rows, then the
/// first error as the host reports it: flags 1 missing atom/value, 2 invalid source slot or task id, 4 unrepresentable
/// numeric value. With flag 2 the error is the first row with an invalid task id (-1) or, when every task id is valid,
/// the first row with an invalid source slot (that slot); otherwise the first row with a missing or unrepresentable
/// value and its first such slot, where numeric slots follow the instance's fluent slots, followed by missing constant
/// facts. No error: [0, -1, -1]. lossless is bool [rows].
/// All buffers outlive the stream work and every graph replay that uses them. Source and target rows must not overlap.
cudaError_t translate_rows(TranslateView map, const u64* source, u64 rows, u64 stride, u32 source_words,
                            const i32* task_ids, u64* target, u32 target_words, u32 target_row_words,
                            u8* lossless, i32* status, cudaStream_t stream);
}
