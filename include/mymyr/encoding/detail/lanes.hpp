#pragma once

// The per-graph semantics of the relational encoding, written once for the host and for device kernels: which tuples a
// graph has, in which order, over which entities, and whether it has an auxiliary entity. A graph is one state row of
// one instance; the host encoder (encoder.hpp) only loops over graphs, scans their sizes, allocates and checks, and a
// device kernel calls the same functions, so the encodings of the two agree bit for bit.
//
// The functions work on POD views of an EncodingContext (context.hpp): typed pointers into one block of fixed-width
// arrays, which is either the host block or its uploaded copy. This header is part of the device-code subset: it
// includes only core/types.hpp, uses no standard containers, exceptions, allocation, virtual calls or lambdas, and
// compiles as C++20 under nvcc.
//
// The tuples of a graph, relation after relation (a relation without a tuple has count 0):
//   - Fluent cell: one tuple per set bit of the state words in the cell's slot range, in ascending slot order, which
//     is the lexicographic order of the atoms' objects;
//   - Derived cell: the same over the derived words;
//   - Static cell: the instance's static tuples of the predicate, sorted and without duplicates;
//   - a nullary predicate's atom is the unary tuple [auxiliary entity].
// Entities of a graph: the instance's objects (local index = object id), then one auxiliary entity (local index
// num_objects) iff some nullary tuple exists. Entity ids written to the arrays are batch-global: the graph's entity
// base plus the local index.
//
// The rules exist once, in per-cell and per-entity functions: cell_count and accumulate_cell / finish_totals (sizes),
// fill_entities and fill_cell (contents). count_graph and fill_graph are loops over them, and a kernel that spreads a
// graph over several threads calls the same functions. A cell's count is per cell: an instance's cells are sorted by
// relation id, at most one per relation, and per-cell counts are indexed by the cell, not by the relation.

#include "mymyr/core/types.hpp"

namespace mymyr::encoding::detail
{
/// The role bytes of the entities a graph has (EntityRole::Object and EntityRole::Auxiliary in schema.hpp).
inline constexpr u8 k_role_object = 0;
inline constexpr u8 k_role_auxiliary = 8;

/// What a cell's tuples come from. The values are stable within a build: the device copy of a context uses them.
enum class CellOp : u32
{
    Fluent = 0,
    Derived = 1,
    Static = 2,
};

/// One relation of one instance that can have tuples. An instance's cells are sorted by relation id.
struct Cell
{
    u32 relation;       // relation id
    u32 op;             // CellOp
    u32 a, b;           // Fluent / Derived: the slot range [a, b); Static: the rows [a, b) of the instance's static tuples
    u32 arity;          // encoded arity: the logical arity, or 1 for a nullary predicate
    u32 logical_arity;  // the predicate's arity
};
static_assert(sizeof(Cell) == 24);

/// Columns of the context's per-instance table `inst` (u64 [instances, k_instance_columns]).
enum InstanceColumn : u32
{
    k_num_objects,
    k_words,                // atom words of the instance's states
    k_fluent_slots,         // atom slots: the state bits at or beyond it must be clear
    k_derived_words,
    k_derived_slots,
    k_num_cells,
    k_cell_offset,          // first row of the instance's cells in `cells`
    k_slot_args_offset,     // first row of the instance's fluent slots in `slot_args`
    k_derived_args_offset,  // the same for derived slots in `derived_args`
    k_static_args_offset,   // the same for static tuples in `static_args`
    k_encoding_fingerprint, // Task::encoding_fingerprint(): states of equal tasks and equal fingerprints decode alike
    k_instance_columns,
};

/// The arrays of an EncodingContext (the host block or a copy of it).
struct ContextView
{
    const u64* inst;  // [num_instances, k_instance_columns]
    u32 num_instances;
    const Cell* cells;
    const i32* slot_args;     // [fluent slots of all instances, max_arity]: instance-local object ids, -1 padding
    const i32* derived_args;  // [derived slots of all instances, max_arity]
    const i32* static_args;   // [static tuples of all instances, max_arity]
    u32 max_arity;            // at least 1
    u32 num_relations;
};

/// One instance of a context, resolved to pointers and counts.
struct InstanceView
{
    u32 num_objects, words, fluent_slots, derived_words, derived_slots, num_cells;
    const Cell* cells;
    const i32* slot_args;
    const i32* derived_args;
    const i32* static_args;
    u32 max_arity, num_relations;
};

MYMYR_HD InstanceView instance_view(const ContextView& c, u32 instance)
{
    const u64* row = c.inst + static_cast<u64>(instance) * k_instance_columns;
    InstanceView v;
    v.num_objects = static_cast<u32>(row[k_num_objects]);
    v.words = static_cast<u32>(row[k_words]);
    v.fluent_slots = static_cast<u32>(row[k_fluent_slots]);
    v.derived_words = static_cast<u32>(row[k_derived_words]);
    v.derived_slots = static_cast<u32>(row[k_derived_slots]);
    v.num_cells = static_cast<u32>(row[k_num_cells]);
    v.cells = c.cells + row[k_cell_offset];
    v.slot_args = c.slot_args + row[k_slot_args_offset] * c.max_arity;
    v.derived_args = c.derived_args + row[k_derived_args_offset] * c.max_arity;
    v.static_args = c.static_args + row[k_static_args_offset] * c.max_arity;
    v.max_arity = c.max_arity;
    v.num_relations = c.num_relations;
    return v;
}

/// The words a graph's tuples come from. `derived` may be null for an instance without derived cells.
struct RowInput
{
    const u64* atoms;
    const u64* derived;
};

/// What a graph needs: its tuple, argument and entity counts, and flags.
struct GraphTotals
{
    u64 tuples;
    u64 args;
    u32 entities;
    u32 flags;
};
inline constexpr u32 k_graph_auxiliary = 1u;  // the graph has an auxiliary entity
inline constexpr u32 k_graph_bad_row = 2u;    // a bit is set at or beyond the instance's fluent (or derived) slots

MYMYR_HD u32 popcount64(u64 x)
{
#if defined(__CUDA_ARCH__)
    return static_cast<u32>(__popcll(x));
#else
    return static_cast<u32>(__builtin_popcountll(x));
#endif
}

/// Index of the lowest set bit of x (x != 0).
MYMYR_HD u32 lowest_bit64(u64 x)
{
#if defined(__CUDA_ARCH__)
    return static_cast<u32>(__ffsll(static_cast<long long>(x))) - 1u;
#else
    return static_cast<u32>(__builtin_ctzll(x));
#endif
}

/// Word `w` of the bit range [lo, hi) (lo < hi) of `words`, with the bits outside the range cleared.
MYMYR_HD u64 range_word(const u64* words, u32 w, u32 lo, u32 hi)
{
    u64 x = words[w];
    if (w == (lo >> 6))
        x &= ~u64{0} << (lo & 63u);
    if (w == ((hi - 1u) >> 6))
        x &= ~u64{0} >> (63u - ((hi - 1u) & 63u));
    return x;
}

/// The number of set bits of `words` in [lo, hi). Reads only the words that hold the range.
MYMYR_HD u32 popcount_range(const u64* words, u32 lo, u32 hi)
{
    if (lo >= hi)
        return 0;
    u32 n = 0;
    for (u32 w = lo >> 6; w <= ((hi - 1u) >> 6); ++w)
        n += popcount64(range_word(words, w, lo, hi));
    return n;
}

/// Whether any bit at or beyond `first` is set in the first `count` words.
MYMYR_HD bool any_bit_from(const u64* words, u32 count, u32 first)
{
    u32 w = first >> 6;
    if (w >= count)
        return false;
    if ((first & 63u) != 0)
    {
        if ((words[w] >> (first & 63u)) != 0)
            return true;
        ++w;
    }
    for (; w < count; ++w)
        if (words[w] != 0)
            return true;
    return false;
}

/// Whether a state row sets a bit the instance does not have: at or beyond its fluent slots (within its words).
MYMYR_HD bool has_foreign_atom_bits(const InstanceView& v, const u64* atoms)
{
    return any_bit_from(atoms, v.words, v.fluent_slots);
}

/// The same for the derived words of a row (at or beyond the instance's derived slots).
MYMYR_HD bool has_foreign_derived_bits(const InstanceView& v, const u64* derived)
{
    return any_bit_from(derived, v.derived_words, v.derived_slots);
}

/// The tuples of one cell in this graph.
MYMYR_HD u32 cell_count(const Cell& c, const RowInput& in)
{
    if (c.op == static_cast<u32>(CellOp::Fluent))
        return popcount_range(in.atoms, c.a, c.b);
    if (c.op == static_cast<u32>(CellOp::Derived))
        return in.derived ? popcount_range(in.derived, c.a, c.b) : 0u;
    return c.b - c.a;
}

/// Adds a cell with `n` tuples to the totals of its graph: tuples, arguments and the auxiliary flag (a nullary cell
/// with a tuple). The entity count and the bad-row flag are set by finish_totals.
MYMYR_HD void accumulate_cell(GraphTotals& t, const Cell& c, u32 n)
{
    t.tuples += n;
    t.args += static_cast<u64>(n) * c.arity;
    if (c.logical_arity == 0 && n > 0)
        t.flags |= k_graph_auxiliary;
}

/// Completes the totals once every cell of the graph is accumulated: the entities (the objects, plus the auxiliary
/// entity if there is one) and the bad-row flag.
MYMYR_HD void finish_totals(GraphTotals& t, const InstanceView& v, const RowInput& in)
{
    t.entities = v.num_objects + ((t.flags & k_graph_auxiliary) ? 1u : 0u);
    if (has_foreign_atom_bits(v, in.atoms) || (in.derived && has_foreign_derived_bits(v, in.derived)))
        t.flags |= k_graph_bad_row;
}

/// The sizes of one graph. With `cell_counts` non-null it also writes the counts of the instance's cells: one entry
/// per cell, in cell order (num_cells entries, not one per relation; the relation of entry i is cells[i].relation).
/// Atom words past the instance's `words` are never read.
MYMYR_HD GraphTotals count_graph(const InstanceView& v, const RowInput& in, u32* cell_counts)
{
    GraphTotals t;
    t.tuples = 0;
    t.args = 0;
    t.entities = 0;
    t.flags = 0;
    const u32 num_cells = v.num_cells;
    const Cell* cells = v.cells;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const u32 n = cell_count(cells[i], in);
        if (cell_counts)
            cell_counts[i] = n;
        accumulate_cell(t, cells[i], n);
    }
    finish_totals(t, v, in);
    return t;
}

/// Where graphs write (graph-major layout): pointers to the batch arrays.
template<class I>
struct GraphOut
{
    I* num_objects;            // [G]
    I* relation_counts;        // [G * R]
    I* relation_offsets;       // [G * R + 1]
    I* relation_arg_offsets;   // [G * R + 1]
    I* relation_args;
    u8* entity_role;           // [E]
    I* entity_ref;             // [E]
};

/// Writes num_objects[g] and the entities of graph `g` from `entity_base` on: the objects, then the auxiliary entity
/// if `auxiliary` (totals.flags & k_graph_auxiliary).
template<class I>
MYMYR_HD void fill_entities(const InstanceView& v, bool auxiliary, u64 g, u64 entity_base, const GraphOut<I>& out)
{
    const u32 num_objects = v.num_objects;
    u8* role = out.entity_role + entity_base;
    I* ref = out.entity_ref + entity_base;
    out.num_objects[g] = static_cast<I>(num_objects);
    for (u32 o = 0; o < num_objects; ++o)
    {
        role[o] = k_role_object;
        ref[o] = static_cast<I>(o);
    }
    if (auxiliary)
    {
        role[num_objects] = k_role_auxiliary;
        ref[num_objects] = static_cast<I>(-1);
    }
}

/// Writes the entity ids of one tuple of `arity` >= 1 objects, `row` holding their instance-local ids, to `dst` and
/// returns the position after it.
template<class I>
MYMYR_HD I* emit_tuple(I* dst, const i32* row, u32 arity, u64 entity_base)
{
    switch (arity)
    {
        case 1:
            dst[0] = static_cast<I>(entity_base + static_cast<u64>(row[0]));
            return dst + 1;
        case 2:
            dst[0] = static_cast<I>(entity_base + static_cast<u64>(row[0]));
            dst[1] = static_cast<I>(entity_base + static_cast<u64>(row[1]));
            return dst + 2;
        case 3:
            dst[0] = static_cast<I>(entity_base + static_cast<u64>(row[0]));
            dst[1] = static_cast<I>(entity_base + static_cast<u64>(row[1]));
            dst[2] = static_cast<I>(entity_base + static_cast<u64>(row[2]));
            return dst + 3;
        default:
            for (u32 k = 0; k < arity; ++k)
                *dst++ = static_cast<I>(entity_base + static_cast<u64>(row[k]));
            return dst;
    }
}

/// Writes the arguments of the tuples of cell `c` in this graph to `dst`, in tuple order, and returns the number of
/// tuples (the cell's count). Object ids become entity ids by adding `entity_base`; the tuple of a nullary cell is
/// the entity `auxiliary`. Writes count * c.arity entries; the caller places `dst`.
template<class I>
MYMYR_HD u32 fill_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary, I* dst)
{
    const u64 A = v.max_arity;
    const u32 arity = c.logical_arity;
    u32 n = 0;
    if (c.op == static_cast<u32>(CellOp::Static))
    {
        n = c.b - c.a;
        if (arity == 0)
        {
            for (u32 s = 0; s < n; ++s)
                *dst++ = static_cast<I>(auxiliary);
            return n;
        }
        const i32* row = v.static_args + static_cast<u64>(c.a) * A;
        for (u32 s = 0; s < n; ++s, row += A)
            dst = emit_tuple<I>(dst, row, arity, entity_base);
        return n;
    }
    const bool fluent = c.op == static_cast<u32>(CellOp::Fluent);
    const u64* words = fluent ? in.atoms : in.derived;
    const i32* table = fluent ? v.slot_args : v.derived_args;
    if (!words || c.a >= c.b)
        return 0;
    for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
    {
        u64 bits = range_word(words, w, c.a, c.b);
        while (bits)
        {
            const u64 s = static_cast<u64>(w) * 64u + lowest_bit64(bits);
            bits &= bits - 1u;
            ++n;
            if (arity == 0)
                *dst++ = static_cast<I>(auxiliary);
            else
                dst = emit_tuple<I>(dst, table + s * A, arity, entity_base);
        }
    }
    return n;
}

/// Writes graph `g`: num_objects[g]; its entities from `entity_base` on (the objects, then the auxiliary entity if
/// totals say the graph has one); and, for its segments g*R .. g*R+R-1, the counts and the offsets, which start at
/// `tuple_base` and `arg_base`, and the tuples' arguments. `totals` is count_graph's result for the same graph. The end
/// entries of the offset arrays are the caller's. A loop over fill_entities and fill_cell: the cells of an instance
/// are sorted by relation, and the relations between them are empty.
template<class I>
MYMYR_HD void fill_graph(const InstanceView& v, const RowInput& in, u64 g, u64 tuple_base, u64 arg_base,
                         u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out)
{
    fill_entities<I>(v, (totals.flags & k_graph_auxiliary) != 0, g, entity_base, out);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const u32 num_cells = v.num_cells;
    const Cell* cells = v.cells;
    I* counts = out.relation_counts + g * R;
    I* offsets = out.relation_offsets + g * R;
    I* arg_offsets = out.relation_arg_offsets + g * R;
    I* args_out = out.relation_args;
    u64 tuples = tuple_base;
    u64 args = arg_base;
    u32 r = 0;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const Cell c = cells[i];
        for (; r < c.relation; ++r)
        {
            offsets[r] = static_cast<I>(tuples);
            arg_offsets[r] = static_cast<I>(args);
            counts[r] = static_cast<I>(0);
        }
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        const u32 n = fill_cell<I>(v, c, in, entity_base, auxiliary, args_out + args);
        counts[r] = static_cast<I>(n);
        tuples += n;
        args += static_cast<u64>(n) * c.arity;
        ++r;
    }
    for (; r < R; ++r)
    {
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        counts[r] = static_cast<I>(0);
    }
}
}  // namespace mymyr::encoding::detail
