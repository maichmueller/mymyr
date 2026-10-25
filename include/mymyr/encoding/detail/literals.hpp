#pragma once

// Literal resolution, written once for the host and for device kernels: a ground atom of one instance, given as a table
// domain predicate id (rl::TaskTable::predicate_names) and instance-local object ids, resolves to what decides its
// truth in the instance's states. The section of the context that this reads (ContextView::literal_ranges) holds, for
// every instance and domain predicate, the predicate's arity and kind and where its atoms are: its fluent slots, its
// derived slots and its static tuples, whatever the options leave out of the encoding.
//
// Kinds:
//   - Fluent / Derived: the atom has a fluent (derived) slot; `slot` is its bit in the state words (derived words);
//   - StaticTrue / StaticFalse: the atom is over a static predicate and holds (does not hold) in every state;
//   - Unreachable: the atom is over a fluent or derived predicate but has no slot, so it holds in no state;
//   - Invalid: the instance, the predicate, an object or the padding is out of range.
// The lookup is a binary search over the predicate's slot rows (or static tuples) of the instance, which are sorted
// lexicographically by their objects under every frozen atom layout, so it costs O(arity * log(atoms of the predicate)).
//
// Part of the device-code subset, as lanes.hpp: no standard containers, exceptions, allocation or lambdas.

#include "mymyr/core/types.hpp"
#include "mymyr/encoding/detail/lanes.hpp"

namespace mymyr::encoding::detail
{
/// What decides the truth of a ground atom in an instance's states. The values are stable within a build.
enum class LiteralKind : u8
{
    Fluent = 0,
    Derived = 1,
    StaticTrue = 2,
    StaticFalse = 3,
    Unreachable = 4,
    Invalid = 5,
};

/// Columns of one row of ContextView::literal_ranges (u32 [instances, predicates, k_literal_columns]). The slot ranges
/// are the instance's slots, the static rows count from the instance's first row of ContextView::static_args.
enum LiteralColumn : u32
{
    k_literal_fluent_begin,
    k_literal_fluent_end,
    k_literal_derived_begin,
    k_literal_derived_end,
    k_literal_static_begin,
    k_literal_static_end,
    k_literal_arity,
    k_literal_predicate_kind,  // k_literal_fluent_predicate, ...
    k_literal_columns,
};

/// The kind of a predicate in an instance (column k_literal_predicate_kind); `absent` for a domain predicate that the
/// instance does not have.
inline constexpr u32 k_literal_fluent_predicate = 0;
inline constexpr u32 k_literal_derived_predicate = 1;
inline constexpr u32 k_literal_static_predicate = 2;
inline constexpr u32 k_literal_absent_predicate = 3;

/// A resolved literal: its kind and, for Fluent and Derived, its slot (0 otherwise).
struct LiteralSlot
{
    LiteralKind kind;
    u32 slot;
};

/// The first row among the rows [begin, end) of `rows` (`stride` entries each, the first `arity` compared) that is not
/// lexicographically less than `objects`, and whether it equals `objects`.
MYMYR_HD bool find_literal_row(const i32* rows, u32 stride, u32 begin, u32 end, const i32* objects, u32 arity, u32& at)
{
    u32 lo = begin, hi = end;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2;
        const i32* row = rows + static_cast<u64>(mid) * stride;
        bool less = false;
        for (u32 k = 0; k < arity; ++k)
            if (row[k] != objects[k])
            {
                less = row[k] < objects[k];
                break;
            }
        // a row equal in its first `arity` objects is not less
        if (less)
            lo = mid + 1;
        else
            hi = mid;
    }
    at = lo;
    if (lo == end)
        return false;
    const i32* row = rows + static_cast<u64>(lo) * stride;
    for (u32 k = 0; k < arity; ++k)
        if (row[k] != objects[k])
            return false;
    return true;
}

/// Resolves the atom `predicate(objects...)` of `instance`. `objects` has `width` entries: the predicate's arity in
/// objects of the instance, then -1 up to `width` (Invalid if the arity exceeds `width`).
MYMYR_HD LiteralSlot resolve_literal(const ContextView& c, i64 instance, i64 predicate, const i32* objects, u32 width)
{
    const LiteralSlot invalid{LiteralKind::Invalid, 0};
    if (instance < 0 || instance >= c.num_instances || predicate < 0 || predicate >= c.num_predicates ||
        !c.literal_ranges)
        return invalid;
    const u64* inst = c.inst + static_cast<u64>(instance) * k_instance_columns;
    const u32* r = c.literal_ranges + (static_cast<u64>(instance) * c.num_predicates + static_cast<u64>(predicate)) *
                                          k_literal_columns;
    const u32 arity = r[k_literal_arity], kind = r[k_literal_predicate_kind];
    if (kind == k_literal_absent_predicate || arity > width)
        return invalid;
    const u64 num_objects = inst[k_num_objects];
    for (u32 k = 0; k < width; ++k)
        if (k < arity ? (objects[k] < 0 || static_cast<u64>(objects[k]) >= num_objects) : objects[k] != -1)
            return invalid;
    u32 at = 0;
    if (kind == k_literal_static_predicate)
    {
        const i32* rows = c.static_args + inst[k_static_args_offset] * c.max_arity;
        const bool found =
            find_literal_row(rows, c.max_arity, r[k_literal_static_begin], r[k_literal_static_end], objects, arity, at);
        return {found ? LiteralKind::StaticTrue : LiteralKind::StaticFalse, 0};
    }
    const bool derived = kind == k_literal_derived_predicate;
    const i32* rows = derived ? c.derived_args + inst[k_derived_args_offset] * c.max_arity
                              : c.slot_args + inst[k_slot_args_offset] * c.max_arity;
    const u32 begin = r[derived ? k_literal_derived_begin : k_literal_fluent_begin];
    const u32 end = r[derived ? k_literal_derived_end : k_literal_fluent_end];
    if (!find_literal_row(rows, c.max_arity, begin, end, objects, arity, at))
        return {LiteralKind::Unreachable, 0};
    return {derived ? LiteralKind::Derived : LiteralKind::Fluent, at};
}

/// Whether a literal of fixed truth (StaticTrue, StaticFalse, Unreachable) holds with sign `positive`.
MYMYR_HD bool fixed_literal_holds(LiteralKind kind, bool positive)
{
    return (kind == LiteralKind::StaticTrue) == positive;
}
}  // namespace mymyr::encoding::detail
