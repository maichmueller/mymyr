#pragma once

// The per-graph semantics of the relational encoding, written once for the host and for device kernels: which tuples a
// graph has, in which order, over which entities, and whether it has an auxiliary entity. A graph is one state row of
// one instance; the host encoder (encoder.hpp) only loops over graphs, scans their sizes, allocates and checks, and a
// device kernel calls the same functions, so the encodings of the two agree bit for bit.
//
// The functions work on POD views of an EncodingContext (context.hpp): typed pointers into one block of fixed-width
// arrays, which is either the host block or its uploaded copy. This header is part of the device-code subset: it
// includes only core/types.hpp, uses no standard containers, exceptions, allocation, virtual calls or lambdas, and
// compiles as C++23 under nvcc.
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
// own masks (RowInput::goal_pos), exactly those masks with no constant part. An instance has two lists of cells for the
// two cases (cell_list): the goal cells of the instance's goal are only those that can hold a tuple, each narrowed to
// the slots between the first and the last literal of its mask, and the goal cells of row goals are those of every
// predicate that has slots, over the predicate's whole slot range.
//   - Action cell: one tuple (action entity, objects...) per distinct ground action of the cell's action relation in the
//     graph's expansion rows (RowInput::actions), in lexicographic order of the objects. A relation is the one of
//     several action schemas when the front end split an action over a disjunction: their rows are merged, and a ground
//     action that more than one schema produces is a single tuple. A graph without expansion rows has no action tuple.
// Every predicate relation (state, goal, goal derivation) of a context with predicate virtual nodes has one more slot,
// after its leading slots and before its arguments: the virtual entity of the relation's predicate.
// Entities of a graph, in this order (local index; ids written to the arrays are batch-global: the graph's entity base
// plus the local index). Every `entity_ref` is graph-local or intrinsic to the catalog, so it is the same in both
// layouts and in every slice of a batch:
//   - the instance's objects (local index = object id); `entity_ref` is the object id;
//   - one auxiliary entity (local index num_objects) iff some nullary tuple exists; `entity_ref` is -1;
//   - one goal entity per tuple of a plain goal cell with goal targets, in the order the plain goal cells are filled
//     (relation, then tuple); its `entity_ref` is the index of its tuple among the graph's tuples in graph-major order
//     (the tuples of the graph's cells in relation order, the cell's first tuple at the sum of the counts of the cells
//     before it);
//   - one action entity per tuple of an action cell, in fill order (relation, then tuple); its `entity_ref` is the row
//     of the graph's first expansion row that has the ground action, counted from the graph's first row;
//   - one virtual entity per predicate that has a tuple in the graph, in the order of the predicates' keys (the id of the
//     predicate's plain state relation, else of its first relation); its `entity_ref` is that key. Predicates without a
//     tuple have no entity, so the entities of a graph depend on its own tuples only.
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
/// The role bytes of the entities a graph has (EntityRole in schema.hpp).
inline constexpr u8 k_role_object = 0;
inline constexpr u8 k_role_predicate_virtual = 1;
inline constexpr u8 k_role_action = 4;
inline constexpr u8 k_role_auxiliary = 8;

/// The most action schemas that share one action relation (the merge of their rows keeps one cursor per schema).
inline constexpr u32 k_max_action_runs = 32;
/// The virtual rank of a predicate the graph does not use.
inline constexpr u32 k_no_virtual = 0xFFFFFFFFu;

/// What a cell's tuples come from. The values are stable within a build: the device copy of a context uses them.
enum class CellOp : u32
{
    Fluent = 0,
    Derived = 1,
    Static = 2,
    Action = 3,        // the ground actions of the schemas a..b of ContextView::action_schemas
    ActionTarget = 4,  // the same; the action entities are also listed as targets
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
MYMYR_HD constexpr bool is_action_op(u32 op)
{
    return op == static_cast<u32>(CellOp::Action) || op == static_cast<u32>(CellOp::ActionTarget);
}

/// One relation of one instance that can have tuples. An instance's cells are sorted by relation id.
struct Cell
{
    u32 relation;       // relation id
    u32 op;             // CellOp
    u32 a, b;           // Fluent / Derived: the slot range [a, b); Static: the rows [a, b) of the instance's static tuples;
                        // a goal cell: the slots of its predicate's atoms that the goal masks can set (empty for a
                        // static predicate);
                        // an action cell: the entries [a, b) of the context's action schema list
    u32 arity;          // encoded arity: the logical arity, or 1 for a nullary predicate, plus the slots of the other roles
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
    k_num_cells,            // the cells of a graph with the instance's goal
    k_cell_offset,          // first row of those cells in `cells`
    k_num_row_cells,        // the cells of a graph with goals of its own
    k_row_cell_offset,      // first row of those cells in `cells`
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
    const u32* action_schemas = nullptr;  // the schemas of every action relation, ascending: the entries of an action cell
    const u32* schema_arity = nullptr;    // [num_schemas] the objects of each action schema's ground actions
    u32 num_schemas = 0;
    // Predicate virtual nodes (null: the context has none): the virtual index of each relation's predicate, or
    // k_no_virtual for a relation without a predicate, and the key relation of each virtual index. Virtual indices
    // ascend with their keys.
    const u32* relation_virtual = nullptr;  // [num_relations]
    const u32* virtual_relation = nullptr;  // [num_virtual]
    u32 num_virtual = 0;
};

/// One instance of a context, resolved to pointers and counts.
struct InstanceView
{
    u32 num_objects, words, fluent_slots, derived_words, derived_slots;
    u32 num_cells;      // the cells of a graph with the instance's goal, sorted by relation
    u32 num_row_cells;  // the cells of a graph with goals of its own, sorted by relation
    const Cell* cells;
    const Cell* row_cells;
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
    const u32* action_schemas;
    const u32* schema_arity;
    u32 num_schemas;
    const u32* relation_virtual;
    const u32* virtual_relation;
    u32 num_virtual;
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
    v.num_row_cells = static_cast<u32>(row[k_num_row_cells]);
    v.cells = c.cells + row[k_cell_offset];
    v.row_cells = c.cells + row[k_row_cell_offset];
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
    v.action_schemas = c.action_schemas;
    v.schema_arity = c.schema_arity;
    v.num_schemas = c.num_schemas;
    v.relation_virtual = c.relation_virtual;
    v.virtual_relation = c.virtual_relation;
    v.num_virtual = c.num_virtual;
    return v;
}

/// The expansion rows of one graph's state: the labels of its successors, in canonical order (by schema, then
/// lexicographically by binding). `schema`, `binding` (rows of `label_width` objects, -1 past the schema's arity) and
/// `parent` (the batch index of the expanded state; null: not checked) point at the state's first row, and `state` is
/// the graph's index in the batch.
struct ActionRows
{
    const i32* schema = nullptr;
    const i32* binding = nullptr;
    const i32* parent = nullptr;
    u32 label_width = 0;
    u32 count = 0;
    u64 state = 0;
    const u32* starts = nullptr;  // [num_schemas + 1]: the first row with a schema of at least s (null: searched for)
};

/// The most schemas whose run starts one graph indexes (a context with more searches the rows instead).
inline constexpr u32 k_action_run_table = 64;

/// The words a graph's tuples come from. `derived` may be null for an instance without derived cells. `goal_pos` and
/// `goal_neg` (null: the instance's goal) are the row's own goal, masks over the fluent slots that replace the
/// instance's goal and its constant part; `goal_derived_pos` and `goal_derived_neg` are the same over the derived slots
/// (null: no derived goal literal). The goal masks are at least as wide as the instance's words (derived words).
/// `actions` are the graph's expansion rows (none: no action tuples).
struct RowInput
{
    const u64* atoms;
    const u64* derived;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    ActionRows actions{};
};

/// The cells of a graph: those of the instance's goal, or, when the row has goal masks of its own, those of row goals.
struct CellList
{
    const Cell* cells;
    u32 count;
};

MYMYR_HD CellList cell_list(const InstanceView& v, const RowInput& in)
{
    if (in.goal_pos)
        return {v.row_cells, v.num_row_cells};
    return {v.cells, v.num_cells};
}

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
    u32 targets;        // the entries of the graph's target arrays: its goal entities, and its action entities for action
                        // cells whose op is ActionTarget
    u32 goal_entities;  // the tuples of the plain goal cells with a goal target
    u32 actions;        // the tuples of the action cells
    u32 virtuals;       // the predicates that have a tuple (with predicate virtual nodes)
};
inline constexpr u32 k_graph_auxiliary = 1u;  // the graph has an auxiliary entity
inline constexpr u32 k_graph_bad_row = 2u;    // a bit is set at or beyond the instance's fluent (or derived) slots
inline constexpr u32 k_graph_bad_goal = 4u;   // the same in the row's own goal masks
inline constexpr u32 k_graph_bad_action_order = 8u;   // the expansion rows are not strictly ascending in (schema, binding)
inline constexpr u32 k_graph_bad_action_label = 16u;  // a row has a schema or object the instance does not have, or
                                                      // belongs to another state

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

/// Whether the tuple `a` is lexicographically before `b` (`arity` object ids each).
MYMYR_HD bool tuple_before(const i32* a, const i32* b, u32 arity)
{
    for (u32 k = 0; k < arity; ++k)
        if (a[k] != b[k])
            return a[k] < b[k];
    return false;
}

/// Whether the expansion rows of a graph are valid labels of the instance and strictly ascending in (schema, binding):
/// the flags k_graph_bad_action_label and k_graph_bad_action_order of the first violation, else 0. With `starts`
/// (v.num_schemas + 1 words) it also records, on the way, the first row of each schema (the rows' run starts, meaningful
/// when the result is 0).
MYMYR_HD u32 check_actions(const InstanceView& v, const ActionRows& a, u32* starts = nullptr)
{
    const u32 S = v.num_schemas;
    const u32 objects = v.num_objects;
    u32 filled = 0, before = 0;
    const i32* previous = nullptr;
    u32 foreign = 0;
    for (u32 i = 0; i < a.count; ++i)
        foreign |= static_cast<u32>(a.schema[i]) >= S;
    if (a.parent)
        for (u32 i = 0; i < a.count; ++i)
            foreign |= static_cast<u64>(a.parent[i]) != a.state;
    if (foreign)
        return k_graph_bad_action_label;
    for (u32 i = 0; i < a.count; ++i)
    {
        const u32 s = static_cast<u32>(a.schema[i]);
        const u32 arity = v.schema_arity[s];
        const i32* row = a.binding + static_cast<u64>(i) * a.label_width;
        u32 outside = 0;
        switch (arity)
        {
        case 3:
            outside |= static_cast<u32>(row[2]) >= objects;
            [[fallthrough]];
        case 2:
            outside |= static_cast<u32>(row[1]) >= objects;
            [[fallthrough]];
        case 1:
            outside |= static_cast<u32>(row[0]) >= objects;
            [[fallthrough]];
        case 0:
            break;
        default:
            for (u32 k = 0; k < arity; ++k)
                outside |= static_cast<u32>(row[k]) >= objects;
        }
        if (outside)
            return k_graph_bad_action_label;
        if (i > 0 && (s < before || (s == before && !tuple_before(previous, row, arity))))
            return k_graph_bad_action_order;
        if (starts)
            for (; filled <= s; ++filled)
                starts[filled] = i;
        before = s;
        previous = row;
    }
    if (starts)
        for (; filled <= S; ++filled)
            starts[filled] = a.count;
    return 0;
}

/// The cursors of the merge of one action relation's runs: the rows of the graph with each schema of the relation form
/// one run (the rows are sorted by schema, so a run is found by binary search), and each run is ascending in the
/// binding. next() yields the distinct ground actions in ascending order of the binding, each as the row of the first
/// run (the lowest schema) that has it.
struct ActionRuns
{
    const ActionRows* rows;
    u32 arity;
    u32 runs;
    u32 pos[k_max_action_runs];
    u32 end[k_max_action_runs];
};

/// The first row of `a` with a schema of at least `schema` (rows are sorted by schema).
MYMYR_HD u32 schema_lower_bound(const ActionRows& a, u32 schema)
{
    if (a.starts)
        return a.starts[schema];
    u32 lo = 0, hi = a.count;
    while (lo < hi)
    {
        const u32 mid = lo + (hi - lo) / 2u;
        if (static_cast<u32>(a.schema[mid]) < schema)
            lo = mid + 1u;
        else
            hi = mid;
    }
    return lo;
}

/// Fills `starts` (v.num_schemas + 1 words, at most k_action_run_table schemas) with the first row of each schema in the
/// graph's rows (valid, sorted by schema), in one pass over the rows.
MYMYR_HD void build_action_starts(const InstanceView& v, const ActionRows& a, u32* starts)
{
    const u32 S = v.num_schemas;
    u32 filled = 0;
    for (u32 i = 0; i < a.count; ++i)
        for (const u32 s = static_cast<u32>(a.schema[i]); filled <= s; ++filled)
            starts[filled] = i;
    for (; filled <= S; ++filled)
        starts[filled] = a.count;
}

/// `in`, or `local` (a copy of it) with the run starts of its expansion rows in `starts` (k_action_run_table + 1 words)
/// when it has rows and few enough schemas to index them. The rows must be valid (check_actions).
MYMYR_HD const RowInput& index_actions(const InstanceView& v, const RowInput& in, RowInput& local, u32* starts)
{
    if (!in.actions.count || v.num_schemas > k_action_run_table || in.actions.starts)
        return in;
    local = in;
    build_action_starts(v, in.actions, starts);
    local.actions.starts = starts;
    return local;
}

/// The runs of the action cell `c` in the rows `a` (of rows sorted by schema).
MYMYR_HD void action_runs_begin(ActionRuns& r, const Cell& c, const u32* action_schemas, const ActionRows& a)
{
    r.rows = &a;
    r.arity = c.logical_arity;
    r.runs = c.b - c.a;
    for (u32 j = 0; j < r.runs; ++j)
    {
        const u32 schema = action_schemas[c.a + j];
        r.pos[j] = schema_lower_bound(a, schema);
        r.end[j] = schema_lower_bound(a, schema + 1u);
    }
}

/// The next distinct ground action: sets `row` to its row in the graph's rows (the first run's, if several runs have it)
/// and advances every run past it. False once every run is exhausted.
MYMYR_HD bool action_runs_next(ActionRuns& r, u32& row)
{
    const ActionRows& a = *r.rows;
    u32 best = r.runs;
    for (u32 j = 0; j < r.runs; ++j)
    {
        if (r.pos[j] == r.end[j])
            continue;
        if (best == r.runs ||
            tuple_before(a.binding + static_cast<u64>(r.pos[j]) * a.label_width,
                         a.binding + static_cast<u64>(r.pos[best]) * a.label_width, r.arity))
            best = j;
    }
    if (best == r.runs)
        return false;
    row = r.pos[best];
    const i32* chosen = a.binding + static_cast<u64>(row) * a.label_width;
    for (u32 j = best; j < r.runs; ++j)
    {
        if (r.pos[j] == r.end[j])
            continue;
        const i32* head = a.binding + static_cast<u64>(r.pos[j]) * a.label_width;
        if (j == best || (!tuple_before(chosen, head, r.arity) && !tuple_before(head, chosen, r.arity)))
            ++r.pos[j];
    }
    return true;
}

/// The distinct ground actions of an action cell in the graph.
MYMYR_HD u32 action_cell_count(const InstanceView& v, const Cell& c, const ActionRows& a)
{
    if (a.count == 0)
        return 0;
    if (c.b - c.a == 1u)
    {
        const u32 schema = v.action_schemas[c.a];
        return schema_lower_bound(a, schema + 1u) - schema_lower_bound(a, schema);
    }
    ActionRuns runs;
    action_runs_begin(runs, c, v.action_schemas, a);
    u32 n = 0, row = 0;
    while (action_runs_next(runs, row))
        ++n;
    return n;
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
    if (is_action_op(c.op))
        return action_cell_count(v, c, in.actions);
    return c.b - c.a;
}


/// Adds a cell with `n` tuples to the totals of its graph: tuples, arguments, the goal entities (the tuples of a plain
/// goal cell with a goal target), the action entities (the tuples of an action cell) and the auxiliary flag (a nullary
/// cell with a tuple, goal cells included; an action relation has no auxiliary slot, its action entity is its only
/// entity). The entity count, the used predicates and the bad-row flags are set by count_graph and finish_totals.
MYMYR_HD void accumulate_cell(GraphTotals& t, const Cell& c, u32 n)
{
    t.tuples += n;
    t.args += static_cast<u64>(n) * c.arity;
    if (is_action_op(c.op))
    {
        t.actions += n;
        if (c.op == static_cast<u32>(CellOp::ActionTarget))
            t.targets += n;
        return;
    }
    if (c.logical_arity == 0 && n > 0)
        t.flags |= k_graph_auxiliary;
    if (is_goal_op(c.op) && goal_op_mode(c.op) == GoalMode::PlainTarget)
    {
        t.goal_entities += n;
        t.targets += n;
    }
}

/// Completes the totals once every cell of the graph is accumulated: the entities (the objects, the auxiliary entity if
/// there is one, the goal, action and virtual entities) and the bad-row flags.
MYMYR_HD void finish_totals(GraphTotals& t, const InstanceView& v, const RowInput& in)
{
    t.entities = v.num_objects + ((t.flags & k_graph_auxiliary) ? 1u : 0u) + t.goal_entities + t.actions + t.virtuals;
    if (has_foreign_atom_bits(v, in.atoms) || (in.derived && has_foreign_derived_bits(v, in.derived)))
        t.flags |= k_graph_bad_row;
    if (has_foreign_goal_bits(v, in))
        t.flags |= k_graph_bad_goal;
}

/// Whether the context has predicate virtual nodes.
MYMYR_HD bool has_virtual_nodes(const InstanceView& v) { return v.relation_virtual != nullptr; }

/// The sizes of one graph. With `cell_counts` non-null it also writes the counts of the instance's cells: one entry
/// per cell, in the order of cell_list(v, given) (one entry per cell, not one per relation; the relation of entry i is
/// the relation of the list's cell i).
/// Atom words past the instance's `words` are never read. With predicate virtual nodes `virtual_scratch` must hold
/// v.num_virtual words (it is overwritten); the totals then count the predicates that have a tuple. The expansion rows
/// of the graph are checked: a violation sets k_graph_bad_action_order or k_graph_bad_action_label, and the counts of
/// the action cells are then meaningless.
MYMYR_HD GraphTotals count_graph(const InstanceView& v, const RowInput& given, u32* cell_counts,
                                 u32* virtual_scratch = nullptr)
{
    u32 starts[k_action_run_table + 1];
    RowInput indexed{nullptr, nullptr};
    const RowInput* chosen = &given;
    u32 action_flags = 0;
    if (given.actions.count)
    {
        const bool index = v.num_schemas <= k_action_run_table && !given.actions.starts;
        action_flags = check_actions(v, given.actions, index ? starts : nullptr);
        if (index && !action_flags)
        {
            indexed = given;
            indexed.actions.starts = starts;
            chosen = &indexed;
        }
    }
    const RowInput& in = *chosen;
    GraphTotals t;
    t.tuples = 0;
    t.args = 0;
    t.entities = 0;
    t.flags = 0;
    t.targets = 0;
    t.goal_entities = 0;
    t.actions = 0;
    t.virtuals = 0;
    const bool virtual_nodes = has_virtual_nodes(v) && virtual_scratch;
    if (virtual_nodes)
        for (u32 p = 0; p < v.num_virtual; ++p)
            virtual_scratch[p] = 0;
    t.flags |= action_flags;
    const CellList list = cell_list(v, in);
    const u32 num_cells = list.count;
    const Cell* cells = list.cells;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const u32 n = cell_count(v, cells[i], in);
        if (cell_counts)
            cell_counts[i] = n;
        accumulate_cell(t, cells[i], n);
        if (virtual_nodes && n)
        {
            const u32 p = v.relation_virtual[cells[i].relation];
            if (p != k_no_virtual)
                virtual_scratch[p] = 1;
        }
    }
    if (virtual_nodes)
        for (u32 p = 0; p < v.num_virtual; ++p)
            t.virtuals += virtual_scratch[p];
    finish_totals(t, v, in);
    return t;
}

/// The virtual rank of every predicate in this graph: `rank[p]` for virtual index p is the number of used predicates
/// below it if p has a tuple in the graph, else k_no_virtual. `cell_counts` are the counts of count_graph (null: they are
/// recounted). Returns the number of used predicates. `rank` holds v.num_virtual words.
MYMYR_HD u32 virtual_ranks(const InstanceView& v, const RowInput& in, const u32* cell_counts, u32* rank)
{
    const u32 P = v.num_virtual;
    for (u32 p = 0; p < P; ++p)
        rank[p] = 0;
    const CellList list = cell_list(v, in);
    for (u32 i = 0; i < list.count; ++i)
    {
        const u32 p = v.relation_virtual[list.cells[i].relation];
        if (p == k_no_virtual || rank[p])
            continue;
        if (cell_counts ? cell_counts[i] != 0 : cell_count(v, list.cells[i], in) != 0)
            rank[p] = 1;
    }
    u32 used = 0;
    for (u32 p = 0; p < P; ++p)
    {
        const u32 flag = rank[p];
        rank[p] = flag ? used : k_no_virtual;
        used += flag;
    }
    return used;
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
    I* target_entity = nullptr;   // [T] the target entities of the graphs (goal, then action), in entity order
    u8* target_group = nullptr;   // [T] the group code of each target
};

/// The segment of cell (g, r) among the G * R cells of a batch with R relations: g * R + r graph-major, r * G + g
/// relation-major.
MYMYR_HD u64 cell_segment(bool relation_major, u64 g, u32 r, u64 G, u32 R)
{
    return relation_major ? static_cast<u64>(r) * G + g : g * R + r;
}

/// The group codes of targets (TargetGroup in schema.hpp) and the role of goal entities (EntityRole::Goal).
inline constexpr u8 k_target_group_goal = 0;
inline constexpr u8 k_target_group_action = 2;
inline constexpr u8 k_role_goal = 2;

/// The first entity of each block of a graph past its objects and auxiliary entity (batch-global ids): the goal
/// entities, the action entities and the virtual entities, in this order.
struct EntityBlocks
{
    u64 goal;
    u64 action;
    u64 virtual_nodes;
};

MYMYR_HD EntityBlocks entity_blocks(const InstanceView& v, const GraphTotals& totals, u64 entity_base)
{
    EntityBlocks blocks;
    blocks.goal = entity_base + v.num_objects + ((totals.flags & k_graph_auxiliary) ? 1u : 0u);
    blocks.action = blocks.goal + totals.goal_entities;
    blocks.virtual_nodes = blocks.action + totals.actions;
    return blocks;
}

/// Writes num_objects[g], goal_complete[g] and target_offsets[g] (when stored; `target_base` is the batch position of
/// the graph's first target) and, unless the entity arrays are not stored, the objects of graph `g` from `entity_base`
/// on, then the auxiliary entity if `auxiliary` (totals.flags & k_graph_auxiliary). The goal and action entities that
/// follow are written with their tuples (fill_cell_at), the virtual entities by fill_virtual_entities.
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

/// Writes the virtual entities of a graph: one per predicate with a rank, at `blocks.virtual_nodes` + its rank, with the
/// predicate's key relation as reference. `rank` is the result of virtual_ranks.
template<class I>
MYMYR_HD void fill_virtual_entities(const InstanceView& v, const EntityBlocks& blocks, const u32* rank,
                                    const GraphOut<I>& out)
{
    if (!out.entity_role)
        return;
    for (u32 p = 0; p < v.num_virtual; ++p)
        if (rank[p] != k_no_virtual)
        {
            const u64 entity = blocks.virtual_nodes + rank[p];
            out.entity_role[entity] = k_role_predicate_virtual;
            out.entity_ref[entity] = static_cast<I>(v.virtual_relation[p]);
        }
}

/// The "no predicate slot" value of the `pred` argument of the tuple writers.
inline constexpr u64 k_no_entity = ~u64{0};

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

/// Writes one tuple of a state cell to `dst`: the predicate entity `pred` (unless k_no_entity), then the objects (the
/// entity `auxiliary` for a nullary predicate); returns the position after it.
template<class I>
MYMYR_HD I* emit_state_tuple(I* dst, const i32* row, u32 arity, u64 entity_base, u64 auxiliary, u64 pred)
{
    if (pred != k_no_entity)
        *dst++ = static_cast<I>(pred);
    if (arity == 0)
    {
        *dst++ = static_cast<I>(auxiliary);
        return dst;
    }
    return emit_tuple<I>(dst, row, arity, entity_base);
}

/// Writes the arguments of the tuples of cell `c` in this graph to `dst`, in tuple order, and returns the number of
/// tuples (the cell's count). Object ids become entity ids by adding `entity_base`; the tuple of a nullary cell is
/// the entity `auxiliary`; each tuple starts with the entity `pred` unless it is k_no_entity. Writes count * c.arity
/// entries; the caller places `dst`.
template<class I>
MYMYR_HD u32 fill_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary, u64 pred,
                       I* dst)
{
    const u64 A = v.max_arity;
    const u32 arity = c.logical_arity;
    u32 n = 0;
    if (c.op == static_cast<u32>(CellOp::Static))
    {
        n = c.b - c.a;
        const i32* row = v.static_args + static_cast<u64>(c.a) * A;
        for (u32 s = 0; s < n; ++s, row += A)
            dst = emit_state_tuple<I>(dst, row, arity, entity_base, auxiliary, pred);
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
            dst = emit_state_tuple<I>(dst, table + s * A, arity, entity_base, auxiliary, pred);
        }
    }
    return n;
}

/// Writes one tuple of a goal cell to `dst` (the goal entity `target_id` first when `target`, then the predicate entity
/// `pred` unless it is k_no_entity; the tuple of a nullary predicate is the entity `auxiliary`) and returns the position
/// after it.
template<class I>
MYMYR_HD I* emit_goal_tuple(I* dst, const i32* row, u32 arity, u64 entity_base, u64 auxiliary, bool target,
                            u64 target_id, u64 pred)
{
    if (target)
        *dst++ = static_cast<I>(target_id);
    return emit_state_tuple<I>(dst, row, arity, entity_base, auxiliary, pred);
}

/// Writes the arguments of the tuples of goal cell `c` in this graph to `dst`, in lexicographic order of their objects,
/// and returns the number of tuples: the cell's mask tuples (ascending slots) merged with its constant rows (the
/// instance's goal only). With goal targets (GoalMode::PlainTarget) each tuple starts with its goal entity, the ids
/// `first_target`, `first_target + 1`, ... in tuple order. A predicate entity `pred` (unless k_no_entity) follows it.
/// Writes count * c.arity entries.
template<class I>
MYMYR_HD u32 fill_goal_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                            u64 first_target, u64 pred, I* dst)
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
                    dst = emit_goal_tuple<I>(dst, constant, arity, entity_base, auxiliary, target, first_target + n++,
                                             pred);
                    constant += A;
                    --constants_left;
                }
                dst = emit_goal_tuple<I>(dst, row, arity, entity_base, auxiliary, target, first_target + n++, pred);
            }
        }
    for (; constants_left; --constants_left, constant += A)
        dst = emit_goal_tuple<I>(dst, constant, arity, entity_base, auxiliary, target, first_target + n++, pred);
    return n;
}

/// Where the goal and action entities of a graph go: the blocks, the number of each written so far (the cells are filled
/// in relation order, so the running count is the position among the graph's entities of that kind), where the graph's
/// targets start in the target arrays, the graph's goal entities (the action targets follow them), and the virtual rank
/// of each predicate (null without virtual nodes).
///
/// A fill that writes the cells of one graph on several lanes (one lane per cell, or per run of cells) gives each lane
/// its own cursor and seeds it from the counts of count_graph (cell_counts, in cell order): `goals_written` is the sum
/// of the counts of the plain goal cells with goal targets before the lane's first cell, `actions_written` the sum of
/// the counts of the action cells before it, and the `graph_tuple` of fill_cell_at the sum of the counts of all cells
/// before it. The sums are exclusive prefix sums over the graph's cells in cell order, so the lanes need no
/// communication.
struct FillCursor
{
    EntityBlocks blocks;
    u64 goals_written;
    u64 actions_written;
    u64 target_base;
    u64 goal_entities;
    const u32* rank;
};

/// The predicate entity of the tuples of relation `relation` in this graph: k_no_entity if the relation has no predicate
/// slot.
MYMYR_HD u64 predicate_entity(const InstanceView& v, const FillCursor& cursor, u32 relation)
{
    if (!cursor.rank)
        return k_no_entity;
    const u32 p = v.relation_virtual[relation];
    if (p == k_no_virtual || cursor.rank[p] == k_no_virtual)
        return k_no_entity;
    return cursor.blocks.virtual_nodes + cursor.rank[p];
}

/// Writes the entity of one ground action: its role, its reference (`row`: the first row of the graph's expansion rows
/// that has it, counted from the graph's first row) and, for an action target, its entry in the target arrays.
template<class I>
MYMYR_HD void write_action_entity(const GraphOut<I>& out, const FillCursor& cursor, bool target, u64 entity, u64 row)
{
    if (out.entity_role)
    {
        out.entity_role[entity] = k_role_action;
        out.entity_ref[entity] = static_cast<I>(row);
    }
    if (target && out.target_entity)
    {
        const u64 at = cursor.target_base + cursor.goal_entities + (entity - cursor.blocks.action);
        out.target_entity[at] = static_cast<I>(entity);
        out.target_group[at] = k_target_group_action;
    }
}

/// Writes the tuples of action cell `c` in this graph to `dst`: for each distinct ground action, in ascending order of the
/// binding, its entity (the next action entity of the graph) and its objects; writes the action entities too. Returns
/// the number of tuples. Writes count * c.arity entries.
template<class I>
MYMYR_HD u32 fill_action_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                              FillCursor& cursor, const GraphOut<I>& out, I* dst)
{
    const ActionRows& a = in.actions;
    if (a.count == 0)
        return 0;
    const u32 arity = c.logical_arity;
    const bool target = c.op == static_cast<u32>(CellOp::ActionTarget);
    u32 n = 0;
    if (c.b - c.a == 1u)
    {
        const u32 schema = v.action_schemas[c.a];
        const u32 begin = schema_lower_bound(a, schema);
        const u32 end = schema_lower_bound(a, schema + 1u);
        n = end - begin;
        const u64 first = cursor.blocks.action + cursor.actions_written;
        u8* role = out.entity_role ? out.entity_role + first : nullptr;
        I* ref = out.entity_role ? out.entity_ref + first : nullptr;
        I* target_entity = nullptr;
        u8* target_group = nullptr;
        if (target && out.target_entity)
        {
            const u64 at = cursor.target_base + cursor.goal_entities + cursor.actions_written;
            target_entity = out.target_entity + at;
            target_group = out.target_group + at;
        }
        const i32* binding = a.binding + static_cast<u64>(begin) * a.label_width;
        for (u32 k = 0; k < n; ++k, binding += a.label_width)
        {
            const u64 entity = first + k;
            *dst++ = static_cast<I>(entity);
            if (arity)
                dst = emit_tuple<I>(dst, binding, arity, entity_base);
            if (role)
            {
                role[k] = k_role_action;
                ref[k] = static_cast<I>(begin + k);
            }
            if (target_entity)
            {
                target_entity[k] = static_cast<I>(entity);
                target_group[k] = k_target_group_action;
            }
        }
    }
    else
    {
        ActionRuns runs;
        action_runs_begin(runs, c, v.action_schemas, a);
        u32 row = 0;
        while (action_runs_next(runs, row))
        {
            const u64 entity = cursor.blocks.action + cursor.actions_written + n;
            *dst++ = static_cast<I>(entity);
            if (arity)
                dst = emit_tuple<I>(dst, a.binding + static_cast<u64>(row) * a.label_width, arity, entity_base);
            write_action_entity<I>(out, cursor, target, entity, row);
            ++n;
        }
    }
    cursor.actions_written += n;
    return n;
}

/// Writes one cell of a graph: its tuples' arguments to `dst` (like fill_cell, fill_goal_cell for a goal cell and
/// fill_action_cell for an action cell) and, for a plain goal cell with goal targets, the entities of its tuples and
/// their target entries. `graph_tuple` is the index of the cell's first tuple among the graph's tuples in graph-major
/// order, the sum of the counts of the graph's cells before it (the `entity_ref` of its first goal entity), whatever
/// the layout. Returns the number of tuples.
template<class I>
MYMYR_HD u32 fill_cell_at(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                          u64 graph_tuple, FillCursor& cursor, const GraphOut<I>& out, I* dst)
{
    if (is_action_op(c.op))
        return fill_action_cell<I>(v, c, in, entity_base, cursor, out, dst);
    const u64 pred = predicate_entity(v, cursor, c.relation);
    if (!is_goal_op(c.op))
        return fill_cell<I>(v, c, in, entity_base, auxiliary, pred, dst);
    const u32 n = fill_goal_cell<I>(v, c, in, entity_base, auxiliary, cursor.blocks.goal + cursor.goals_written, pred, dst);
    if (goal_op_mode(c.op) == GoalMode::PlainTarget)
    {
        for (u32 k = 0; k < n; ++k)
        {
            const u64 entity = cursor.blocks.goal + cursor.goals_written + k;
            if (out.entity_role)
            {
                out.entity_role[entity] = k_role_goal;
                out.entity_ref[entity] = static_cast<I>(graph_tuple + k);
            }
            if (out.target_entity)
            {
                out.target_entity[cursor.target_base + cursor.goals_written + k] = static_cast<I>(entity);
                out.target_group[cursor.target_base + cursor.goals_written + k] = k_target_group_goal;
            }
        }
        cursor.goals_written += n;
    }
    return n;
}

/// The start of a graph's fill: its entities (the objects, the auxiliary entity, and the virtual entities, which need the
/// ranks of the predicates, computed into `scratch` from `cell_counts` (null: recounted)) and the cursor of its goal and
/// action entities. `scratch` must hold v.num_virtual words with predicate virtual nodes and may be null otherwise.
template<class I>
MYMYR_HD FillCursor begin_graph(const InstanceView& v, const RowInput& in, u64 g, u64 entity_base,
                                const GraphTotals& totals, const GraphOut<I>& out, u64 target_base,
                                const u32* cell_counts, u32* scratch)
{
    const bool auxiliary = (totals.flags & k_graph_auxiliary) != 0;
    fill_entities<I>(v, in, auxiliary, g, entity_base, target_base, out);
    FillCursor cursor{entity_blocks(v, totals, entity_base), 0, 0, target_base, totals.goal_entities, nullptr};
    if (has_virtual_nodes(v) && scratch)
    {
        virtual_ranks(v, in, cell_counts, scratch);
        cursor.rank = scratch;
        fill_virtual_entities<I>(v, cursor.blocks, scratch, out);
    }
    return cursor;
}

/// Writes graph `g` of a graph-major batch: num_objects[g]; its entities from `entity_base` on (the objects, then the
/// auxiliary entity if totals say the graph has one, the goal and action entities with their tuples, the virtual
/// entities); and, for its segments g*R .. g*R+R-1, the counts (unless out.relation_counts is null) and the offsets,
/// which start at `tuple_base` and `arg_base`, and the tuples' arguments. `totals` is count_graph's result for the same
/// graph; `target_base` is where the graph's targets start in the target arrays; `virtual_scratch` is as in count_graph.
/// The end entries of the offset arrays are the caller's. A loop over begin_graph and fill_cell_at: the cells of an
/// instance are sorted by relation, and the relations between them are empty.
template<class I>
MYMYR_HD void fill_graph(const InstanceView& v, const RowInput& given, u64 g, u64 tuple_base, u64 arg_base,
                         u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0,
                         u32* virtual_scratch = nullptr)
{
    u32 starts[k_action_run_table + 1];
    RowInput indexed{nullptr, nullptr};
    const RowInput& in = index_actions(v, given, indexed, starts);
    FillCursor cursor = begin_graph<I>(v, in, g, entity_base, totals, out, target_base, nullptr, virtual_scratch);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list(v, in);
    const u32 num_cells = list.count;
    const Cell* cells = list.cells;
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
        const u32 n =
            fill_cell_at<I>(v, c, in, entity_base, auxiliary, tuples - tuple_base, cursor, out, args_out + args);
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
/// beforehand) and not computed from a local prefix. The `entity_ref` of a goal entity is the same as in fill_graph.
/// The counts, if stored, are [G, R] indexed (g, r). `cell_counts` are the graph's cell counts from count_graph (null:
/// they are recounted when virtual nodes need them).
template<class I>
MYMYR_HD void fill_graph_relation_major(const InstanceView& v, const RowInput& given, u64 g, u64 entity_base,
                                        const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0,
                                        u32* virtual_scratch = nullptr, const u32* cell_counts = nullptr)
{
    u32 starts[k_action_run_table + 1];
    RowInput indexed{nullptr, nullptr};
    const RowInput& in = index_actions(v, given, indexed, starts);
    FillCursor cursor =
        begin_graph<I>(v, in, g, entity_base, totals, out, target_base, cell_counts, virtual_scratch);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list(v, in);
    const u32 num_cells = list.count;
    const Cell* cells = list.cells;
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    u64 graph_tuple = 0;  // the tuples of the graph's cells before this one: the cells are filled in cell order
    u32 r = 0;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const Cell c = cells[i];
        if (counts)
            for (; r < c.relation; ++r)
                counts[r] = static_cast<I>(0);
        const u64 segment = cell_segment(true, g, c.relation, out.graphs, R);
        const u32 n = fill_cell_at<I>(v, c, in, entity_base, auxiliary, graph_tuple, cursor, out,
                                      out.relation_args + out.relation_arg_offsets[segment]);
        graph_tuple += n;
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
