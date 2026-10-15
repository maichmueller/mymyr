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
//   - a nullary predicate's atom is the unary tuple [auxiliary entity];
//   - Goal cell, plain: one tuple per goal literal of the cell's predicate and sign, which is each set bit of the goal's
//     positive (negative) mask in the cell's slot range, merged in lexicographic order with the cell's constant part,
//     the literals no state can change (see below). With goal targets the tuple starts with its goal entity;
//   - Goal cell, satisfied / unsatisfied: the tuples of the plain cell that hold / do not hold in the row: with the
//     state words x (the derived words d for a derived predicate), a positive literal holds iff its bit is set in x and a
//     negative one iff it is clear, so the tuples are `gpos & x` and `gneg & ~x` (satisfied), `gpos & ~x` and `gneg & x`
//     (unsatisfied), plus the constant literals that hold / do not hold.
// The goal of a graph is its instance's goal (masks over the instance's fluent and derived slots, and the constant
// part: the literals of static predicates and of atoms without a slot, whose truth is fixed) or, when the row brings its
// own masks (RowInput::goal_pos), exactly those masks with no constant part.
// Entities of a graph: the instance's objects (local index = object id), then one auxiliary entity (local index
// num_objects) iff some nullary tuple exists, then one goal entity per tuple of a plain goal cell with goal targets, in
// the order the plain goal cells are filled (relation, then tuple). Entity ids written to the arrays are batch-global:
// the graph's entity base plus the local index. A goal entity's `entity_ref` is the batch-global id of its tuple (its
// position in the concatenation of all cells in layout order).
//
// The rules exist once, in per-cell and per-entity functions: cell_count and accumulate_cell / finish_totals (sizes),
// fill_entities and fill_cell (contents). count_graph and fill_graph are loops over them, and a kernel that spreads a
// graph over several threads calls the same functions. A cell's count is per cell: an instance's cells are sorted by
// relation id, at most one per relation, and per-cell counts are indexed by the cell, not by the relation.
//
// Layouts. The cell (g, r) of G graphs and R relations is segment g * R + r in the graph-major layout and r * G + g in
// the relation-major one (cell_segment). Graph-major positions are a prefix over the graph's own cells and need nothing
// but the graph's base (fill_graph). Relation-major positions come from an exclusive scan over all G * R segments in
// segment order (the cells' counts, as count_graph reports them), written to relation_offsets and relation_arg_offsets
// before the fill; fill_graph_relation_major reads its cells' positions from the two arrays.

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

/// The variants of a goal cell: the plain goal literals (with the goal entity in front of each tuple for PlainTarget) and
/// the literals that hold / do not hold in the row.
enum class GoalMode : u32
{
    Plain = 0,
    PlainTarget = 1,
    Satisfied = 2,
    Unsatisfied = 3,
};

/// Goal cells have an op of at least k_goal_op_base: the base, plus 8 for a derived predicate (else a fluent or static
/// one), plus 4 for the negative literals, plus the GoalMode.
inline constexpr u32 k_goal_op_base = 16;

MYMYR_HD constexpr u32 goal_op(bool derived, bool negative, GoalMode mode)
{
    return k_goal_op_base + (derived ? 8u : 0u) + (negative ? 4u : 0u) + static_cast<u32>(mode);
}
MYMYR_HD constexpr bool is_goal_op(u32 op) { return op >= k_goal_op_base; }
MYMYR_HD constexpr bool goal_op_derived(u32 op) { return ((op - k_goal_op_base) & 8u) != 0; }
MYMYR_HD constexpr bool goal_op_negative(u32 op) { return ((op - k_goal_op_base) & 4u) != 0; }
MYMYR_HD constexpr GoalMode goal_op_mode(u32 op) { return static_cast<GoalMode>((op - k_goal_op_base) & 3u); }

/// One relation of one instance that can have tuples. An instance's cells are sorted by relation id.
struct Cell
{
    u32 relation;       // relation id
    u32 op;             // CellOp
    u32 a, b;           // Fluent / Derived: the slot range [a, b); Static: the rows [a, b) of the instance's static tuples;
                        // a goal cell: the slot range of its predicate's atoms in the goal masks (empty for a static one)
    u32 arity;          // encoded arity: the logical arity, or 1 for a nullary predicate, plus the target slots
    u32 logical_arity;  // the predicate's arity
    u32 c = 0, d = 0;   // a goal cell: the rows [c, d) of the instance's constant goal tuples that belong to the cell
};
static_assert(sizeof(Cell) == 32);

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
    k_goal_offset,          // first word of the instance's goal masks in `goal_words`
    k_goal_args_offset,     // first row of the instance's constant goal tuples in `goal_args`
    k_goal_complete,        // 1 iff the instance's goal masks and constant part hold its whole goal
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
    const u64* goal_words = nullptr;  // per instance: positive and negative goal masks over the fluent slots (`words` words
                                      // each), then over the derived slots (`derived_words` words each)
    const i32* goal_args = nullptr;   // [constant goal tuples of all instances, max_arity]: the cells' constant part
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
    const u64* goal_pos;          // the instance's goal: masks over the fluent slots
    const u64* goal_neg;
    const u64* goal_derived_pos;  // and over the derived slots
    const u64* goal_derived_neg;
    const i32* goal_args;         // the constant goal tuples
    u32 goal_complete;
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
    const u64* goals = c.goal_words + row[k_goal_offset];
    v.goal_pos = goals;
    v.goal_neg = goals + v.words;
    v.goal_derived_pos = goals + 2u * v.words;
    v.goal_derived_neg = goals + 2u * v.words + v.derived_words;
    v.goal_args = c.goal_args + row[k_goal_args_offset] * c.max_arity;
    v.goal_complete = static_cast<u32>(row[k_goal_complete]);
    return v;
}

/// The words a graph's tuples come from. `derived` may be null for an instance without derived cells. `goal_pos` and
/// `goal_neg` (null: the instance's goal) are the row's own goal, masks over the fluent slots that replace the
/// instance's goal and its constant part; `goal_derived_pos` and `goal_derived_neg` are the same over the derived slots
/// (null: no derived goal literal). The goal masks are at least as wide as the instance's words (derived words).
struct RowInput
{
    const u64* atoms;
    const u64* derived;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
};

/// The goal a graph encodes: the row's masks, or the instance's with its constant part.
struct GoalWords
{
    const u64* pos;
    const u64* neg;
    const u64* derived_pos;
    const u64* derived_neg;
    bool constants;  // the instance's goal: its constant part counts
};

MYMYR_HD GoalWords goal_words(const InstanceView& v, const RowInput& in)
{
    if (in.goal_pos)
        return {in.goal_pos, in.goal_neg, in.goal_derived_pos, in.goal_derived_neg, false};
    return {v.goal_pos, v.goal_neg, v.goal_derived_pos, v.goal_derived_neg, true};
}

/// Whether the graph's goal is its whole goal: always for a row's own masks, else the instance's flag.
MYMYR_HD u32 goal_is_complete(const InstanceView& v, const RowInput& in)
{
    return in.goal_pos ? 1u : v.goal_complete;
}

/// What a graph needs: its tuple, argument and entity counts, and flags.
struct GraphTotals
{
    u64 tuples;
    u64 args;
    u32 entities;
    u32 flags;
    u32 targets;  // the goal entities: the tuples of the plain goal cells with a goal target
};
inline constexpr u32 k_graph_auxiliary = 1u;  // the graph has an auxiliary entity
inline constexpr u32 k_graph_bad_row = 2u;    // a bit is set at or beyond the instance's fluent (or derived) slots
inline constexpr u32 k_graph_bad_goal = 4u;   // the same in the row's own goal masks

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

/// Whether the masks of the row's own goal set a bit the instance does not have (at or beyond its fluent or derived
/// slots, within its words).
MYMYR_HD bool has_foreign_goal_bits(const InstanceView& v, const RowInput& in)
{
    if (!in.goal_pos)
        return false;
    if (any_bit_from(in.goal_pos, v.words, v.fluent_slots) || any_bit_from(in.goal_neg, v.words, v.fluent_slots))
        return true;
    return (in.goal_derived_pos && any_bit_from(in.goal_derived_pos, v.derived_words, v.derived_slots)) ||
           (in.goal_derived_neg && any_bit_from(in.goal_derived_neg, v.derived_words, v.derived_slots));
}

/// Word `w` of the goal tuples a goal cell takes from its slot range: the cell's mask word (positive or negative literals,
/// fluent or derived), for the satisfied and unsatisfied variants masked by the state word: a positive literal holds iff
/// its bit is set, a negative one iff it is clear.
MYMYR_HD u64 goal_word(const Cell& c, const GoalWords& g, const RowInput& in, u32 w)
{
    const bool derived = goal_op_derived(c.op);
    const bool negative = goal_op_negative(c.op);
    const GoalMode mode = goal_op_mode(c.op);
    const u64* mask = derived ? (negative ? g.derived_neg : g.derived_pos) : (negative ? g.neg : g.pos);
    if (!mask)
        return 0;
    const u64 m = range_word(mask, w, c.a, c.b);
    if (mode == GoalMode::Plain || mode == GoalMode::PlainTarget)
        return m;
    const u64* x = derived ? in.derived : in.atoms;
    const u64 xw = x ? x[w] : u64{0};
    const bool wants_set = (mode == GoalMode::Satisfied) != negative;
    return m & (wants_set ? xw : ~xw);
}

/// The tuples of a goal cell in this graph: those its masks give plus the cell's constant part (the instance's goal only).
MYMYR_HD u32 goal_cell_count(const Cell& c, const GoalWords& g, const RowInput& in)
{
    u32 n = g.constants ? c.d - c.c : 0u;
    if (c.a < c.b)
        for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
            n += popcount64(goal_word(c, g, in, w));
    return n;
}

/// The tuples of one cell in this graph.
MYMYR_HD u32 cell_count(const InstanceView& v, const Cell& c, const RowInput& in)
{
    if (c.op == static_cast<u32>(CellOp::Fluent))
        return popcount_range(in.atoms, c.a, c.b);
    if (c.op == static_cast<u32>(CellOp::Derived))
        return in.derived ? popcount_range(in.derived, c.a, c.b) : 0u;
    if (is_goal_op(c.op))
        return goal_cell_count(c, goal_words(v, in), in);
    return c.b - c.a;
}

/// Adds a cell with `n` tuples to the totals of its graph: tuples, arguments, the goal entities (the tuples of a plain
/// goal cell with a goal target) and the auxiliary flag (a nullary cell with a tuple, goal cells included). The entity count and the bad-row flag are set by finish_totals.
MYMYR_HD void accumulate_cell(GraphTotals& t, const Cell& c, u32 n)
{
    t.tuples += n;
    t.args += static_cast<u64>(n) * c.arity;
    if (c.logical_arity == 0 && n > 0)
        t.flags |= k_graph_auxiliary;
    if (is_goal_op(c.op) && goal_op_mode(c.op) == GoalMode::PlainTarget)
        t.targets += n;
}

/// Completes the totals once every cell of the graph is accumulated: the entities (the objects, the auxiliary entity if
/// there is one, the goal entities) and the bad-row flags.
MYMYR_HD void finish_totals(GraphTotals& t, const InstanceView& v, const RowInput& in)
{
    t.entities = v.num_objects + ((t.flags & k_graph_auxiliary) ? 1u : 0u) + t.targets;
    if (has_foreign_atom_bits(v, in.atoms) || (in.derived && has_foreign_derived_bits(v, in.derived)))
        t.flags |= k_graph_bad_row;
    if (has_foreign_goal_bits(v, in))
        t.flags |= k_graph_bad_goal;
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
    t.targets = 0;
    const u32 num_cells = v.num_cells;
    const Cell* cells = v.cells;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const u32 n = cell_count(v, cells[i], in);
        if (cell_counts)
            cell_counts[i] = n;
        accumulate_cell(t, cells[i], n);
    }
    finish_totals(t, v, in);
    return t;
}

/// Where graphs write: pointers to the batch arrays.
template<class I>
struct GraphOut
{
    u64 graphs;                // G
    I* num_objects;            // [G]
    I* relation_counts;        // [G, R] indexed (g, r) in both layouts; null: not stored
    I* relation_offsets;       // [G * R + 1] in segment order
    I* relation_arg_offsets;   // [G * R + 1] in segment order
    I* relation_args;
    u8* entity_role;           // [E]; null: not stored (together with entity_ref)
    I* entity_ref;             // [E]
    u8* goal_complete = nullptr;  // [G]; null: not stored
    I* target_offsets = nullptr;  // [G + 1]; null: no target arrays (they are stored together)
    I* target_entity = nullptr;   // [T] the goal entities of the graphs, in entity order
    u8* target_group = nullptr;   // [T] the group code of each target
};

/// The segment of cell (g, r) among the G * R cells of a batch with R relations: g * R + r graph-major, r * G + g
/// relation-major.
MYMYR_HD u64 cell_segment(bool relation_major, u64 g, u32 r, u64 G, u32 R)
{
    return relation_major ? static_cast<u64>(r) * G + g : g * R + r;
}

/// The group code of goal targets (TargetGroup::Goal in schema.hpp) and the role of goal entities (EntityRole::Goal).
inline constexpr u8 k_target_group_goal = 0;
inline constexpr u8 k_role_goal = 2;

/// Writes num_objects[g], goal_complete[g] and target_offsets[g] (when stored; `target_base` is the batch position of
/// the graph's first target) and, unless the entity arrays are not stored, the objects of graph `g` from `entity_base`
/// on, then the auxiliary entity if `auxiliary` (totals.flags & k_graph_auxiliary). The goal entities that follow are
/// written with their tuples (fill_cell_at).
template<class I>
MYMYR_HD void fill_entities(const InstanceView& v, const RowInput& in, bool auxiliary, u64 g, u64 entity_base,
                            u64 target_base, const GraphOut<I>& out)
{
    const u32 num_objects = v.num_objects;
    out.num_objects[g] = static_cast<I>(num_objects);
    if (out.goal_complete)
        out.goal_complete[g] = static_cast<u8>(goal_is_complete(v, in));
    if (out.target_offsets)
        out.target_offsets[g] = static_cast<I>(target_base);
    if (!out.entity_role)
        return;
    u8* role = out.entity_role + entity_base;
    I* ref = out.entity_ref + entity_base;
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

/// Whether the tuple `a` is lexicographically before `b` (`arity` object ids each).
MYMYR_HD bool tuple_before(const i32* a, const i32* b, u32 arity)
{
    for (u32 k = 0; k < arity; ++k)
        if (a[k] != b[k])
            return a[k] < b[k];
    return false;
}

/// Writes one tuple of a goal cell to `dst` (the goal entity `target_id` first when `target`; the tuple of a nullary
/// predicate is the entity `auxiliary`) and returns the position after it.
template<class I>
MYMYR_HD I* emit_goal_tuple(I* dst, const i32* row, u32 arity, u64 entity_base, u64 auxiliary, bool target,
                            u64 target_id)
{
    if (target)
        *dst++ = static_cast<I>(target_id);
    if (arity == 0)
    {
        *dst++ = static_cast<I>(auxiliary);
        return dst;
    }
    return emit_tuple<I>(dst, row, arity, entity_base);
}

/// Writes the arguments of the tuples of goal cell `c` in this graph to `dst`, in lexicographic order of their objects,
/// and returns the number of tuples: the cell's mask tuples (ascending slots) merged with its constant rows (the
/// instance's goal only). With goal targets (GoalMode::PlainTarget) each tuple starts with its goal entity, the ids
/// `first_target`, `first_target + 1`, ... in tuple order. Writes count * c.arity entries.
template<class I>
MYMYR_HD u32 fill_goal_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                            u64 first_target, I* dst)
{
    const GoalWords g = goal_words(v, in);
    const u64 A = v.max_arity;
    const u32 arity = c.logical_arity;
    const bool target = goal_op_mode(c.op) == GoalMode::PlainTarget;
    const i32* table = goal_op_derived(c.op) ? v.derived_args : v.slot_args;
    const i32* constant = v.goal_args + static_cast<u64>(c.c) * A;
    u32 constants_left = g.constants ? c.d - c.c : 0u;
    u32 n = 0;
    if (c.a < c.b)
        for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
        {
            u64 bits = goal_word(c, g, in, w);
            while (bits)
            {
                const u64 s = static_cast<u64>(w) * 64u + lowest_bit64(bits);
                bits &= bits - 1u;
                const i32* row = table + s * A;
                while (constants_left && tuple_before(constant, row, arity))
                {
                    dst = emit_goal_tuple<I>(dst, constant, arity, entity_base, auxiliary, target, first_target + n++);
                    constant += A;
                    --constants_left;
                }
                dst = emit_goal_tuple<I>(dst, row, arity, entity_base, auxiliary, target, first_target + n++);
            }
        }
    for (; constants_left; --constants_left, constant += A)
        dst = emit_goal_tuple<I>(dst, constant, arity, entity_base, auxiliary, target, first_target + n++);
    return n;
}

/// Where the goal entities of a graph go: the id of the first and the number written so far (the plain goal cells are
/// filled in relation order, so the running count is the position among the graph's goal entities), and where the
/// graph's targets start in the target arrays.
struct GoalCursor
{
    u64 first_entity;
    u64 written;
    u64 target_base;
};

/// Writes one cell of a graph: its tuples' arguments to `dst` (like fill_cell, and fill_goal_cell for a goal cell) and,
/// for a plain goal cell with goal targets, the entities of its tuples and their target entries. `tuple_id` is the
/// batch-global position of the cell's first tuple in layout order (the `entity_ref` of its first goal entity). Returns
/// the number of tuples.
template<class I>
MYMYR_HD u32 fill_cell_at(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                          u64 tuple_id, GoalCursor& goals, const GraphOut<I>& out, I* dst)
{
    if (!is_goal_op(c.op))
        return fill_cell<I>(v, c, in, entity_base, auxiliary, dst);
    const u32 n = fill_goal_cell<I>(v, c, in, entity_base, auxiliary, goals.first_entity + goals.written, dst);
    if (goal_op_mode(c.op) == GoalMode::PlainTarget)
    {
        for (u32 k = 0; k < n; ++k)
        {
            const u64 entity = goals.first_entity + goals.written + k;
            if (out.entity_role)
            {
                out.entity_role[entity] = k_role_goal;
                out.entity_ref[entity] = static_cast<I>(tuple_id + k);
            }
            if (out.target_entity)
            {
                out.target_entity[goals.target_base + goals.written + k] = static_cast<I>(entity);
                out.target_group[goals.target_base + goals.written + k] = k_target_group_goal;
            }
        }
        goals.written += n;
    }
    return n;
}

/// Writes graph `g` of a graph-major batch: num_objects[g]; its entities from `entity_base` on (the objects, then the
/// auxiliary entity if totals say the graph has one); and, for its segments g*R .. g*R+R-1, the counts (unless
/// out.relation_counts is null) and the offsets, which start at `tuple_base` and `arg_base`, and the tuples' arguments.
/// `totals` is count_graph's result for the same graph; `target_base` is where the graph's targets start in the target
/// arrays. The end entries of the offset arrays are the caller's. A loop over fill_entities and fill_cell_at: the cells
/// of an instance are sorted by relation, and the relations between them are empty.
template<class I>
MYMYR_HD void fill_graph(const InstanceView& v, const RowInput& in, u64 g, u64 tuple_base, u64 arg_base,
                         u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0)
{
    const bool has_auxiliary = (totals.flags & k_graph_auxiliary) != 0;
    fill_entities<I>(v, in, has_auxiliary, g, entity_base, target_base, out);
    const u64 auxiliary = entity_base + v.num_objects;
    GoalCursor goals{auxiliary + (has_auxiliary ? 1u : 0u), 0, target_base};
    const u32 R = v.num_relations;
    const u32 num_cells = v.num_cells;
    const Cell* cells = v.cells;
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
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
            if (counts)
                counts[r] = static_cast<I>(0);
        }
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        const u32 n = fill_cell_at<I>(v, c, in, entity_base, auxiliary, tuples, goals, out, args_out + args);
        if (counts)
            counts[r] = static_cast<I>(n);
        tuples += n;
        args += static_cast<u64>(n) * c.arity;
        ++r;
    }
    for (; r < R; ++r)
    {
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        if (counts)
            counts[r] = static_cast<I>(0);
    }
}

/// Writes graph `g` of a relation-major batch: like fill_graph, except that a cell's position is read from the scanned
/// segment arrays (out.relation_offsets and out.relation_arg_offsets at the cell's segment, scanned
/// beforehand) and not computed from a local prefix. The counts, if stored, are [G, R] indexed (g, r).
template<class I>
MYMYR_HD void fill_graph_relation_major(const InstanceView& v, const RowInput& in, u64 g, u64 entity_base,
                                        const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0)
{
    const bool has_auxiliary = (totals.flags & k_graph_auxiliary) != 0;
    fill_entities<I>(v, in, has_auxiliary, g, entity_base, target_base, out);
    const u64 auxiliary = entity_base + v.num_objects;
    GoalCursor goals{auxiliary + (has_auxiliary ? 1u : 0u), 0, target_base};
    const u32 R = v.num_relations;
    const u32 num_cells = v.num_cells;
    const Cell* cells = v.cells;
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    u32 r = 0;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const Cell c = cells[i];
        if (counts)
            for (; r < c.relation; ++r)
                counts[r] = static_cast<I>(0);
        const u64 segment = cell_segment(true, g, c.relation, out.graphs, R);
        const u32 n = fill_cell_at<I>(v, c, in, entity_base, auxiliary, static_cast<u64>(out.relation_offsets[segment]),
                                      goals, out, out.relation_args + out.relation_arg_offsets[segment]);
        if (counts)
        {
            counts[c.relation] = static_cast<I>(n);
            r = c.relation + 1;
        }
    }
    if (counts)
        for (; r < R; ++r)
            counts[r] = static_cast<I>(0);
}
}  // namespace mymyr::encoding::detail
