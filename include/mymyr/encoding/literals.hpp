#pragma once

// Literals of the instances of an EncodingContext, given as table domain predicate ids (rl::TaskTable::predicate_names)
// and instance-local object ids: their resolution to slots (detail::resolve_literal, one literal after the other) and
// the goal masks they make, per row or per row and layer, in the form the encoders take as row goals
// (EncodeInput::goal_pos ...), subgoal layers and history steps (LayerInput).

#include "mymyr/encoding/context.hpp"
#include "mymyr/encoding/detail/literals.hpp"

#include <span>
#include <string>
#include <vector>

namespace mymyr::encoding
{
using detail::LiteralKind;

/// What literal_masks does with a literal of fixed truth: one over a static predicate (StaticTrue, StaticFalse) or one
/// whose atom has no slot (Unreachable).
enum class FixedLiterals : u8
{
    Decide,  // a literal that holds is left out; one that fails marks its row (layer) unsatisfiable
    Raise,   // std::invalid_argument
};

/// N literals as arrays: literal i is the atom predicates[i](objects[i * width], ...), the predicate's arity in objects
/// followed by -1 up to `width`.
struct LiteralArrays
{
    std::span<const i32> predicates;  // [N]
    std::span<const i32> objects;     // [N, width]
    u32 width = 0;
};

/// Resolves every literal: literal i in instance instances[i] (empty: every literal in instance 0) into kinds[i] and
/// slots[i] (the fluent or derived slot, -1 for the other kinds). Invalid input gives the kind Invalid; throws
/// std::invalid_argument only for arrays of the wrong sizes.
void resolve_literals(const EncodingContext& context, const LiteralArrays& literals, std::span<const i32> instances,
                      std::span<LiteralKind> kinds, std::span<i32> slots);

/// Where each literal of literal_masks goes: its sign, its row and its layer, and the instance of each row.
struct LiteralPlacement
{
    std::span<const u8> positive;    // [N]: nonzero for a positive literal
    std::span<const i32> rows;       // [N], in [0, num_rows)
    u32 num_rows = 0;
    std::span<const i32> layers;     // [N], in [0, num_layers); empty: every literal in layer 0
    u32 num_layers = 1;
    std::span<const i32> instances;  // [num_rows]; empty: every row of instance 0 (a table of one instance)
};

/// The masks of rows of literals, num_layers per row (1 for goals): pos and neg [rows, layers, words] over the fluent
/// slots, derived_pos and derived_neg [rows, layers, derived_words] over the derived slots, and unsatisfiable [rows,
/// layers]: 1 where FixedLiterals::Decide left out a literal that fails. words and derived_words are the table's.
struct LiteralMasks
{
    u32 rows = 0, layers = 0, words = 0, derived_words = 0;
    std::vector<u64> pos, neg, derived_pos, derived_neg;
    std::vector<u8> unsatisfiable;
};

/// The masks of `literals` placed by `placement`: each literal sets the bit of its slot in the positive or negative
/// mask of its row and layer; a literal of fixed truth is handled as `fixed` says. Throws std::invalid_argument for an
/// invalid literal, row, layer or instance, and (FixedLiterals::Raise) for a literal of fixed truth, naming the literal.
[[nodiscard]] LiteralMasks literal_masks(const EncodingContext& context, const LiteralArrays& literals,
                                         const LiteralPlacement& placement, FixedLiterals fixed);

/// The literal `predicate(objects...)` of `instance` written with the names of its predicate and objects, as
/// `(on a b)` or `(not (on a b))`; ids out of range are written as numbers.
[[nodiscard]] std::string literal_text(const EncodingContext& context, i64 instance, i64 predicate,
                                       std::span<const i32> objects, bool positive);
}  // namespace mymyr::encoding
