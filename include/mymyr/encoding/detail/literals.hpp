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
// LiteralKind, the literal section's columns, find_literal_row and resolve_literal_row are in lanes.hpp, which the
// constant literals of goals use too.

/// Resolves the atom `predicate(objects...)` of `instance`. `objects` has `width` entries: the predicate's arity in
/// objects of the instance, then -1 up to `width` (Invalid if the arity exceeds `width`).
template<class Objects>
MYMYR_HD LiteralSlot resolve_literal(const ContextView& c, i64 instance, i64 predicate, const Objects& objects, u32 width)
{
    if (instance < 0 || instance >= c.num_instances || predicate < 0 || predicate >= c.num_predicates ||
        !c.literal_ranges)
        return {LiteralKind::Invalid, 0};
    const u64* inst = c.inst + static_cast<u64>(instance) * k_instance_columns;
    const u32* r = c.literal_ranges + (static_cast<u64>(instance) * c.num_predicates + static_cast<u64>(predicate)) *
                                          k_literal_columns;
    return resolve_literal_row(r, inst[k_num_objects], c.static_args + inst[k_static_args_offset] * c.max_arity,
                               c.slot_args + inst[k_slot_args_offset] * c.max_arity,
                               c.derived_args + inst[k_derived_args_offset] * c.max_arity, c.max_arity, objects, width);
}
}  // namespace mymyr::encoding::detail
