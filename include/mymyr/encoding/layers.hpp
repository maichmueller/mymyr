#pragma once

// The per-row subgoal layers and history steps of an encoding call, as the host encoder (encoder.hpp) and the device
// encoder (cuda/encoding.hpp, with device pointers) take them. Plain structs, so that device code can use them.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding
{
/// Masks in layers, one set of layers per row: the subgoal layers of the rows, or (in HistoryInput) their history
/// steps. A row's layer l is the pair of masks pos[row * stride + l * words ...] and neg[...] over the fluent slots,
/// optionally the same over the derived slots (derived_pos, derived_neg, `derived_words` wide, `derived_stride` per
/// row). The masks are `words` wide, at least as wide as the words (derived words) of every instance in the batch; a
/// set bit of a slot an instance does not have is an error. A stride of 0 is one set of layers that every row shares.
struct LayerInput
{
    const u64* pos = nullptr;
    const u64* neg = nullptr;
    u64 stride = 0;
    u32 layers = 0;
    u32 words = 0;
    const u64* derived_pos = nullptr;
    const u64* derived_neg = nullptr;
    u64 derived_stride = 0;
    u32 derived_words = 0;
};

/// The history of each row: steps of literals, oldest first. `masks` has one set of layers per row, as many as the
/// longest history (`masks.layers`), and a nonzero stride of at least layers * words (derived_stride likewise): rows
/// never share. `dt` [rows, masks.layers] is the dt of each step and `count` [rows] the valid steps of each row, a
/// prefix of its layers (null: all of them). Only the newest FlatOptions::history_max_steps of a row's valid steps are
/// encoded (all of them when it is not set).
struct HistoryInput
{
    LayerInput masks;
    const i32* dt = nullptr;
    const i32* count = nullptr;
};
}  // namespace mymyr::encoding
