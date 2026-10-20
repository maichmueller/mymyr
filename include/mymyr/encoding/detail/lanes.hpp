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
//   - Subgoal cells (goal relations of a level above 0) have the cells of the goal's three variants, with the layer's
//     masks in place of the goal's: layer l of the row's subgoals (RowInput::layers) fills level l + 1, minus every
//     literal of the same sign at a lower level, the goal included, so that a literal appears once, at its lowest
//     level. They have no constant part;
//   - History cell `[+]p[hist]` / `[-]p[hist]`: for each used step of the row's history in order (the newest steps,
//     oldest first), the step's positive (negative) literals of the cell's predicate in ascending slot order; a tuple
//     is [history target]? + [history step] + [predicate]? + objects. With history targets each tuple has its own
//     target entity.
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
//   - one subgoal entity per tuple of a plain subgoal cell with subgoal targets, numbered and referring like the goal
//     entities;
//   - one action entity per tuple of an action cell, in fill order (relation, then tuple); its `entity_ref` is the row
//     of the graph's first expansion row that has the ground action, counted from the graph's first row;
//   - one history entity per used step, in step order; its `entity_ref` is the step's index among the row's valid
//     steps (so the input step of the entity is history[g][entity_ref]);
//   - one history-target entity per tuple of a history cell with history targets, numbered like the goal entities
//     (relation, then tuple), referring to its tuple like them;
//   - one virtual entity per predicate that has a tuple in the graph, in the order of the predicates' names; its
//     `entity_ref` is the predicate's key (the id of its plain state relation, else of its first relation). Predicates
//     without a tuple have no entity, so the entities of a graph depend on its own tuples only.
//
// The rules exist once, in per-cell and per-entity functions: cell_count and accumulate_cell / finish_totals (sizes),
// fill_entities and fill_cell (contents). count_graph and fill_graph are loops over them, and a kernel that spreads a
// graph over several threads calls the same functions. A cell's count is per cell: an instance's cells are sorted by
// relation id, at most one per relation, and per-cell counts are indexed by the cell, not by the relation.
//
// The functions that handle subgoal layers and history steps take a `Layered` template argument (default true). False
// is for a batch whose rows have neither (RowInput::layers null): that handling is compiled out, and the results are
// those of Layered true.
//
// Layouts. The cell (g, r) of G graphs and R relations is segment g * R + r in the graph-major layout and r * G + g in
// the relation-major one (cell_segment). Graph-major positions are a prefix over the graph's own cells and need nothing
// but the graph's base (fill_graph). Relation-major positions come from an exclusive scan over all G * R segments in
// segment order (the cells' counts, as count_graph reports them), written to relation_offsets and relation_arg_offsets
// before the fill; fill_graph_relation_major reads its cells' positions from the two arrays. A kernel that does not
// keep those arrays gives fill_graph_positioned a position policy instead (args(segment, relation) and count(segment)).
//
// Lane groups. A graph can be served by a group of lanes (threads of one warp, or a single thread) that split its cells
// and its entities between them: the group forms (count_graph_group_into, fill_graph_group,
// fill_graph_positioned_group) take an object with the lane's rank and size, a sum, a bit-or and an exclusive scan over
// the group, and a barrier. They compute the same sizes and write the same bytes as the plain forms, for any group
// size; SoloGroup is the group of one lane.

#include <cassert>

#include "mymyr/core/types.hpp"

#if defined(__CUDACC__)
#define MYMYR_NOINLINE __host__ __device__ __noinline__
#else
#define MYMYR_NOINLINE __attribute__((noinline))
#endif

namespace mymyr::encoding::detail
{
/// The role bytes of the entities a graph has (EntityRole in schema.hpp).
inline constexpr u8 k_role_object = 0;
inline constexpr u8 k_role_predicate_virtual = 1;
inline constexpr u8 k_role_subgoal = 3;
inline constexpr u8 k_role_action = 4;
inline constexpr u8 k_role_history = 5;
inline constexpr u8 k_role_history_target = 6;
inline constexpr u8 k_role_auxiliary = 8;

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

/// Goal cells have an op of at least k_goal_op_base: the base, plus 16 per goal level (0: the goal, 1 to 3: the subgoal
/// layers), plus 8 for a derived predicate (else a fluent or static one), plus 4 for the negative literals, plus the
/// GoalMode.
inline constexpr u32 k_goal_op_base = 16;

MYMYR_HD constexpr u32 goal_op(bool derived, bool negative, GoalMode mode, u32 level = 0)
{
    return k_goal_op_base + (level << 4) + (derived ? 8u : 0u) + (negative ? 4u : 0u) + static_cast<u32>(mode);
}
MYMYR_HD constexpr bool is_goal_op(u32 op) { return op >= k_goal_op_base; }
MYMYR_HD constexpr bool goal_op_derived(u32 op) { return ((op - k_goal_op_base) & 8u) != 0; }
MYMYR_HD constexpr bool goal_op_negative(u32 op) { return ((op - k_goal_op_base) & 4u) != 0; }
MYMYR_HD constexpr GoalMode goal_op_mode(u32 op) { return static_cast<GoalMode>((op - k_goal_op_base) & 3u); }
MYMYR_HD constexpr u32 goal_op_level(u32 op) { return ((op - k_goal_op_base) >> 4) & 3u; }

/// History cells have an op in [k_history_op_base, k_goal_op_base): the base, plus 1 for a derived predicate, plus 2
/// for the negative literals, plus 4 when each tuple has a history target.
inline constexpr u32 k_history_op_base = 8;

MYMYR_HD constexpr u32 history_op(bool derived, bool negative, bool target)
{
    return k_history_op_base + (derived ? 1u : 0u) + (negative ? 2u : 0u) + (target ? 4u : 0u);
}
MYMYR_HD constexpr bool is_history_op(u32 op) { return op >= k_history_op_base && op < k_goal_op_base; }
MYMYR_HD constexpr bool history_op_derived(u32 op) { return ((op - k_history_op_base) & 1u) != 0; }
MYMYR_HD constexpr bool history_op_negative(u32 op) { return ((op - k_history_op_base) & 2u) != 0; }
MYMYR_HD constexpr bool history_op_target(u32 op) { return ((op - k_history_op_base) & 4u) != 0; }
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
    k_num_plain_cells,      // the same two lists without the action cells (for a graph without expansion rows)
    k_plain_cell_offset,
    k_num_plain_row_cells,
    k_plain_row_cell_offset,
    k_slot_args_offset,     // first row of the instance's fluent slots in `slot_args`
    k_derived_args_offset,  // the same for derived slots in `derived_args`
    k_static_args_offset,   // the same for static tuples in `static_args`
    k_encoding_fingerprint, // Task::encoding_fingerprint(): states of equal tasks and equal fingerprints decode alike
    k_goal_offset,          // first word of the instance's goal masks in `goal_words`
    k_goal_args_offset,     // first row of the instance's constant goal tuples in `goal_args`
    k_goal_complete,        // 1 iff the instance's goal masks and constant part hold its whole goal
    k_needs_derived,        // which derived words its graphs read (k_needs_* bits)
    k_ext_lists,            // 12 pairs (cells, first row in `cells`): the lists of a graph with subgoal layers, history
                            // steps, or both, for each of the four lists above (see cell_list)
    k_instance_columns = k_ext_lists + 24,
};

/// The bits of the column k_needs_derived: the instance's state cells list derived atoms, the literals of the instance's
/// own goal include derived ones that a satisfied or unsatisfied relation lists, and a goal of the row over derived
/// slots (RowInput::goal_derived_pos) is listed by those relations.
inline constexpr u32 k_needs_state_derived = 1u;
inline constexpr u32 k_needs_goal_derived = 2u;
inline constexpr u32 k_needs_row_goal_derived = 4u;

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
    // ascend with the names of their predicates.
    const u32* relation_virtual = nullptr;  // [num_relations]
    const u32* virtual_relation = nullptr;  // [num_virtual]
    u32 num_virtual = 0;
    const u32* relation_arity = nullptr;  // [num_relations] the encoded arity of each relation
};

/// One instance of a context, resolved to pointers and counts.
struct InstanceView
{
    u32 num_objects, words, fluent_slots, derived_words, derived_slots;
    // The cells of a graph, sorted by relation: with the instance's goal or with goals of its own (row_cells), and with
    // the action cells (a graph that has expansion rows) or without (plain_cells, plain_row_cells).
    u32 num_cells, num_row_cells, num_plain_cells, num_plain_row_cells;
    const Cell *cells, *row_cells, *plain_cells, *plain_row_cells;
    const Cell* cell_base;   // the context's cells: the rows of the extended lists are relative to it
    const u64* ext_lists;    // 12 pairs (count, first row) of the lists of a graph with subgoal layers and / or history
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
    u32 needs_derived;
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
    v.num_plain_cells = static_cast<u32>(row[k_num_plain_cells]);
    v.num_plain_row_cells = static_cast<u32>(row[k_num_plain_row_cells]);
    v.cells = c.cells + row[k_cell_offset];
    v.row_cells = c.cells + row[k_row_cell_offset];
    v.plain_cells = c.cells + row[k_plain_cell_offset];
    v.plain_row_cells = c.cells + row[k_plain_row_cell_offset];
    v.cell_base = c.cells;
    v.ext_lists = row + k_ext_lists;
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
    v.needs_derived = static_cast<u32>(row[k_needs_derived]);
    v.action_schemas = c.action_schemas;
    v.schema_arity = c.schema_arity;
    v.num_schemas = c.num_schemas;
    v.relation_virtual = c.relation_virtual;
    v.virtual_relation = c.virtual_relation;
    v.num_virtual = c.num_virtual;
    return v;
}

/// Whether the graphs of the instance read the derived atoms of the state: it has a derived state cell, or a derived
/// literal in its own goal that a satisfied or unsatisfied goal relation lists. With the goals of the rows (`row_goals`)
/// the instance's own goal does not count, and only a goal over derived slots (`row_derived_goals`) reads them, in the
/// satisfied and unsatisfied relations of the instance's derived predicates. Subgoal layers over derived slots
/// (`layer_derived_goals`) read them in the same relations of the subgoal levels.
MYMYR_HD bool reads_derived(const InstanceView& v, bool row_goals, bool row_derived_goals,
                            bool layer_derived_goals = false)
{
    const u32 flags = v.needs_derived;
    return (flags & k_needs_state_derived) != 0 ||
           (row_goals ? (row_derived_goals && (flags & k_needs_row_goal_derived) != 0)
                      : (flags & k_needs_goal_derived) != 0) ||
           (layer_derived_goals && (flags & k_needs_row_goal_derived) != 0);
}

/// Whether a row can be read for its instance. Width checks happen before any state or goal word is read.
enum class RowValidity : u8 { Valid, BadTask, NarrowState, NarrowGoal };

MYMYR_HD RowValidity row_validity(const ContextView& context, i32 instance, u32 state_words,
                                  bool row_goals, u32 goal_words, bool derived_goals, u32 goal_derived_words)
{
    if (instance < 0 || static_cast<u32>(instance) >= context.num_instances)
        return RowValidity::BadTask;
    const InstanceView v = instance_view(context, static_cast<u32>(instance));
    if (state_words < v.words)
        return RowValidity::NarrowState;
    if ((row_goals && goal_words < v.words) || (derived_goals && goal_derived_words < v.derived_words))
        return RowValidity::NarrowGoal;
    return RowValidity::Valid;
}

/// Conservative work of one instance without expansion rows, including either instance or row goals.
struct GraphWidth
{
    u64 words = 0, cells = 0, args = 0;
};
MYMYR_HD GraphWidth graph_width(const InstanceView& v)
{
    GraphWidth out;
    out.words = v.words > v.derived_words ? v.words : v.derived_words;
    out.cells = v.num_plain_cells > v.num_plain_row_cells ? v.num_plain_cells : v.num_plain_row_cells;
    for (u32 own = 0; own < 2; ++own)
    {
        const Cell* cells = own ? v.plain_row_cells : v.plain_cells;
        const u32 count = own ? v.num_plain_row_cells : v.num_plain_cells;
        u64 args = 0;
        for (u32 i = 0; i < count; ++i)
        {
            const Cell& c = cells[i];
            args += (u64{c.b - c.a} + (is_goal_op(c.op) ? c.d - c.c : 0u)) * c.arity;
        }
        out.args = args > out.args ? args : out.args;
    }
    return out;
}

/// The expansion rows of one graph's state: the labels of its successors, in canonical order (by schema, then
/// lexicographically by binding). `schema`, `binding` (rows of `label_width` objects, -1 past the schema's arity) and
/// `parent` (the batch index of the expanded state; null: not checked) point at the state's first row, and `state` is
/// the graph's index in the batch. A graph with rows (count > 0) also has two scratch arrays of the caller's: `starts`
/// (v.num_schemas + 1 words), which count_graph fills with the first row of each schema and fill_graph reads, so it
/// must survive between the two for the graph; and `cursors` (v.num_schemas words), the cursors of the merge of the
/// runs of an action relation, which only the call that uses them needs.
struct ActionRows
{
    const i32* schema = nullptr;
    const i32* binding = nullptr;
    const i32* parent = nullptr;
    u32 label_width = 0;
    u32 count = 0;
    u64 state = 0;
    u32* starts = nullptr;
    u32* cursors = nullptr;
};

/// Masks in layers, for one row: `layers` masks of `words` words each, one after the other (layer l starts at
/// `pos + l * words`), and the same over the derived slots (`derived_words` each; null: none). The subgoals of a row
/// are layers, and its history steps are, oldest first.
struct MaskLayers
{
    const u64* pos;
    const u64* neg;
    const u64* derived_pos;
    const u64* derived_neg;
    u32 layers;
    u32 words;
    u32 derived_words;
};

/// The history steps a row uses: `masks.layers` of them, the newest of its valid steps (`first` is the index of the
/// first used one among the valid steps, so that the masks of used step j are those of step first + j), and the `dt` of
/// each.
struct HistoryRows
{
    MaskLayers masks;
    const i32* dt;
    u32 first;
};

/// The masks of the subgoal levels of a row (effective_layers), at op - k_goal_op_base for the op of a goal cell of the
/// level: the literals its cell takes (the plain mask of the positive or negative literals over the fluent or derived
/// slots of the level, or the part of it that the state satisfies or does not), null where there is none.
struct LevelPlanes
{
    const u64* plane[64];
};

/// The subgoal layers and history steps of a row; a row without either has no RowLayers. `levels` are the subgoal
/// levels (effective_layers); `subgoals` are the layers as given, which only the checks of the row read.
struct RowLayers
{
    MaskLayers subgoals;
    HistoryRows history;
    LevelPlanes levels;
};

/// The words a graph's tuples come from. `derived` may be null for an instance without derived cells. `goal_pos` and
/// `goal_neg` (null: the instance's goal) are the row's own goal, masks over the fluent slots that replace the
/// instance's goal and its constant part; `goal_derived_pos` and `goal_derived_neg` are the same over the derived slots
/// (null: no derived goal literal). The goal masks are at least as wide as the instance's words (derived words).
/// `actions` are the graph's expansion rows (none: no action tuples). `layers` (null: none) are the row's subgoal
/// layers and history steps.
struct RowInput
{
    const u64* atoms;
    const u64* derived;
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    ActionRows actions{};
    const RowLayers* layers = nullptr;
};

/// The cells of a graph: those of the instance's goal, or, when the row has goals of its own, those of row goals; the
/// action cells only for a graph that has expansion rows.
struct CellList
{
    const Cell* cells;
    u32 count;
};

MYMYR_HD CellList cell_list(const InstanceView& v, bool row_goals, bool actions)
{
    if (actions)
        return row_goals ? CellList{v.row_cells, v.num_row_cells} : CellList{v.cells, v.num_cells};
    return row_goals ? CellList{v.plain_row_cells, v.num_plain_row_cells}
                     : CellList{v.plain_cells, v.num_plain_cells};
}

/// The same for a graph with subgoal layers and / or history steps: the list also holds their cells.
MYMYR_HD CellList cell_list(const InstanceView& v, bool row_goals, bool actions, bool subgoals, bool history)
{
    if (!subgoals && !history)
        return cell_list(v, row_goals, actions);
    const u32 extension = (subgoals ? 1u : 0u) | (history ? 2u : 0u);  // 1 to 3
    const u32 base = (row_goals ? 1u : 0u) + (actions ? 0u : 2u);  // the index of the list without layers
    const u64* entry = v.ext_lists + 2u * ((extension - 1u) * 4u + base);
    return CellList{v.cell_base + entry[1], static_cast<u32>(entry[0])};
}

template<bool Layered = true>
MYMYR_HD CellList cell_list(const InstanceView& v, const RowInput& in)
{
    if (Layered && in.layers)
        return cell_list(v, in.goal_pos != nullptr, in.actions.count != 0, in.layers->subgoals.layers != 0,
                         in.layers->history.masks.layers != 0);
    return cell_list(v, in.goal_pos != nullptr, in.actions.count != 0);
}

/// The precondition of the forms for the lanes of a group: the row has no subgoal layers and no history steps (they are
/// counted and filled by count_graph and fill_graph).
MYMYR_HD void expect_no_layers([[maybe_unused]] const RowInput& in) { assert(in.layers == nullptr); }

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
    u32 subgoal_entities;  // the tuples of the plain subgoal cells with a subgoal target
    u32 history_steps;     // the history steps the row uses: one entity each
    u32 history_targets;   // the tuples of the history cells with a history target
};
inline constexpr u32 k_graph_auxiliary = 1u;  // the graph has an auxiliary entity
inline constexpr u32 k_graph_bad_row = 2u;    // a bit is set at or beyond the instance's fluent (or derived) slots
inline constexpr u32 k_graph_bad_goal = 4u;   // the same in the row's own goal masks
inline constexpr u32 k_graph_bad_action_order = 8u;   // the expansion rows are not strictly ascending in (schema, binding)
inline constexpr u32 k_graph_bad_action_label = 16u;  // a row has a schema or object the instance does not have, or
                                                      // belongs to another state
inline constexpr u32 k_graph_bad_layers = 32u;   // a subgoal layer sets a bit at or beyond the instance's slots
inline constexpr u32 k_graph_bad_history = 64u;  // the same in a history step

MYMYR_HD u32 popcount64(u64 x)
{
#if defined(__CUDA_ARCH__)
    return static_cast<u32>(__popcll(x));
#elif defined(__POPCNT__)
    return static_cast<u32>(__builtin_popcountll(x));
#else
    // without a popcount instruction the builtin is a library call; the bit-parallel sum is inline
    x -= (x >> 1) & 0x5555555555555555ull;
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return static_cast<u32>((x * 0x0101010101010101ull) >> 56);
#endif
}

/// The number of bits of x (0 for 0).
MYMYR_HD u32 bit_length(u32 x)
{
    if (x == 0)
        return 0;
#if defined(__CUDA_ARCH__)
    return 32u - static_cast<u32>(__clz(static_cast<int>(x)));
#else
    return 32u - static_cast<u32>(__builtin_clz(x));
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

/// A group of lanes that work on one graph together: a warp, or a part of one, on a device. Its lanes execute the same
/// control flow on the same graph. The members are the operations the group functions below need:
///   lane(), size()        the lane's index in [0, size()) and the lanes in the group;
///   sync()                a barrier of the lanes, with the memory ordering of the writes before it;
///   sum(x)                the sum of x over the lanes, for u32 and u64, returned to every lane;
///   bit_or(x)             the bitwise or of the u32 x over the lanes, returned to every lane;
///   exclusive_scan(x, total)  the sum of x over the lanes before this one; `total` is the sum over all lanes.
/// SoloGroup is the group of one lane, which makes the group functions plain sequential code.
struct SoloGroup
{
    MYMYR_HD u32 lane() const { return 0; }
    MYMYR_HD u32 size() const { return 1; }
    MYMYR_HD void sync() const {}
    MYMYR_HD u32 sum(u32 x) const { return x; }
    MYMYR_HD u64 sum(u64 x) const { return x; }
    MYMYR_HD u32 bit_or(u32 x) const { return x; }
    MYMYR_HD u32 exclusive_scan(u32 x, u32& total) const
    {
        total = x;
        return 0;
    }
};

/// any_bit_from over the lanes of a group: each lane reads its share of the words and the result is the same on every
/// lane.
template<class Group>
MYMYR_HD bool any_bit_from_group(const Group& grp, const u64* words, u32 count, u32 first)
{
    u32 found = 0;
    u32 w = first >> 6;
    if (w < count && (first & 63u) != 0)
    {
        found = static_cast<u32>(grp.lane() == 0 && (words[w] >> (first & 63u)) != 0);
        ++w;
    }
    for (w += grp.lane(); w < count; w += grp.size())
        found |= static_cast<u32>(words[w] != 0);
    return grp.bit_or(found) != 0;
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

/// Whether a mask of `layers` sets a bit the instance does not have, in any layer.
MYMYR_HD bool has_foreign_layer_bits(const InstanceView& v, const MaskLayers& layers)
{
    for (u32 l = 0; l < layers.layers; ++l)
    {
        const u64 at = static_cast<u64>(l) * layers.words;
        if (layers.pos && any_bit_from(layers.pos + at, v.words, v.fluent_slots))
            return true;
        if (layers.neg && any_bit_from(layers.neg + at, v.words, v.fluent_slots))
            return true;
        const u64 derived_at = static_cast<u64>(l) * layers.derived_words;
        if (layers.derived_pos && any_bit_from(layers.derived_pos + derived_at, v.derived_words, v.derived_slots))
            return true;
        if (layers.derived_neg && any_bit_from(layers.derived_neg + derived_at, v.derived_words, v.derived_slots))
            return true;
    }
    return false;
}

/// has_foreign_goal_bits over the lanes of a group.
template<class Group>
MYMYR_HD bool has_foreign_goal_bits_group(const Group& grp, const InstanceView& v, const RowInput& in)
{
    if (!in.goal_pos)
        return false;
    u32 found = static_cast<u32>(any_bit_from_group(grp, in.goal_pos, v.words, v.fluent_slots));
    found |= static_cast<u32>(any_bit_from_group(grp, in.goal_neg, v.words, v.fluent_slots));
    if (in.goal_derived_pos)
        found |= static_cast<u32>(any_bit_from_group(grp, in.goal_derived_pos, v.derived_words, v.derived_slots));
    if (in.goal_derived_neg)
        found |= static_cast<u32>(any_bit_from_group(grp, in.goal_derived_neg, v.derived_words, v.derived_slots));
    return found != 0;
}

/// Whether the tuple `a` is lexicographically before `b` (`arity` object ids each).
MYMYR_HD bool tuple_before(const i32* a, const i32* b, u32 arity)
{
    for (u32 k = 0; k < arity; ++k)
        if (a[k] != b[k])
            return a[k] < b[k];
    return false;
}

/// The rows of one schema (`n` rows of `label_width` entries from `row`, each starting with `K` objects): sets
/// `outside` if an object is not below `objects` and `unordered` if the rows are not strictly ascending. The rows are
/// compared as keys of `bits` bits per object.
template<u32 K>
MYMYR_HD void check_run_packed(const i32* row, u32 label_width, u32 n, u32 objects, u32 bits, u32& outside,
                               u32& unordered)
{
    u32 largest = 0;
    u64 previous = 0;
    for (u32 i = 0; i < n; ++i, row += label_width)
    {
        u64 key = 0;
        for (u32 k = 0; k < K; ++k)
        {
            const u32 x = static_cast<u32>(row[k]);
            largest = x > largest ? x : largest;
            key = (key << bits) | x;
        }
        unordered |= static_cast<u32>(key <= previous) & static_cast<u32>(i > 0);
        previous = key;
    }
    outside |= static_cast<u32>(K > 0 && largest >= objects);
}

/// The same for rows of `arity` objects too wide for a key.
MYMYR_HD void check_run_wide(const i32* row, u32 label_width, u32 n, u32 arity, u32 objects, u32& outside,
                             u32& unordered)
{
    const i32* previous = nullptr;
    for (u32 i = 0; i < n; ++i, row += label_width)
    {
        for (u32 k = 0; k < arity; ++k)
            outside |= static_cast<u32>(static_cast<u32>(row[k]) >= objects);
        if (i > 0 && !tuple_before(previous, row, arity))
            unordered = 1;
        previous = row;
    }
}

/// Why the rows are not valid: the label flag for a schema out of range or a parent that is not the graph's state, else
/// the order flag (the schemas descend). The slow path of check_actions.
MYMYR_HD u32 action_row_fault(const InstanceView& v, const ActionRows& a)
{
    for (u32 i = 0; i < a.count; ++i)
        if (static_cast<u32>(a.schema[i]) >= v.num_schemas ||
            (a.parent && static_cast<u64>(static_cast<i64>(a.parent[i])) != a.state))
            return k_graph_bad_action_label;
    return k_graph_bad_action_order;
}

/// Whether the expansion rows of a graph are valid labels of the instance and strictly ascending in (schema, binding):
/// the flags k_graph_bad_action_label and k_graph_bad_action_order, else 0. A row is a valid label if its schema is one
/// of the context's, its parent (if given) is the graph's state, and the first `arity` entries of its binding are
/// objects of the instance. It writes a.starts (the first row of each schema, in v.num_schemas + 1 words): one pass
/// over the rows finds the run of each schema in turn, which consumes every row exactly if the schemas ascend and are
/// in range and the parents are right, and the bindings of each run are then checked with its arity. After a violation
/// the starts are all 0, so that the counts of the action cells are 0.
MYMYR_HD u32 check_actions(const InstanceView& v, const ActionRows& a)
{
    const u32 S = v.num_schemas;
    const u32 count = a.count;
    u32* starts = a.starts;
    const i32* schema = a.schema;
    const i32* parent = a.parent;
    const bool state_fits = a.state <= 0x7FFFFFFFu;
    const i32 state = static_cast<i32>(a.state);
    const u32 L = a.label_width;
    const u32 objects = v.num_objects;
    const u32 bits = bit_length(objects ? objects - 1u : 0u);
    u32 outside = 0, unordered = 0;
    // the runs of the rows, one per schema that has rows: `next` is the first schema whose start is not set yet
    u32 row = 0, next = 0;
    while (row < count)
    {
        const u32 s = static_cast<u32>(schema[row]);
        if (!state_fits || s < next || s >= S || (parent && parent[row] != state))
        {
            for (u32 t = 0; t <= S; ++t)
                starts[t] = 0;
            return !state_fits ? k_graph_bad_action_label : action_row_fault(v, a);
        }
        for (; next <= s; ++next)
            starts[next] = row;
        const u32 begin = row;
        if (parent)
        {
            do
                ++row;
            while (row < count && static_cast<u32>(schema[row]) == s && parent[row] == state);
        }
        else
        {
            do
                ++row;
            while (row < count && static_cast<u32>(schema[row]) == s);
        }
        const u32 n = row - begin;
        const u32 arity = v.schema_arity[s];
        const i32* rows = a.binding + static_cast<u64>(begin) * L;
        if (arity > L)
        {
            outside = 1;
            continue;
        }
        if (u64{arity} * bits > 64u)
        {
            check_run_wide(rows, L, n, arity, objects, outside, unordered);
            continue;
        }
        switch (arity)
        {
        case 0: check_run_packed<0>(rows, L, n, objects, bits, outside, unordered); break;
        case 1: check_run_packed<1>(rows, L, n, objects, bits, outside, unordered); break;
        case 2: check_run_packed<2>(rows, L, n, objects, bits, outside, unordered); break;
        case 3: check_run_packed<3>(rows, L, n, objects, bits, outside, unordered); break;
        case 4: check_run_packed<4>(rows, L, n, objects, bits, outside, unordered); break;
        default: check_run_wide(rows, L, n, arity, objects, outside, unordered);
        }
    }
    for (; next <= S; ++next)
        starts[next] = count;
    return outside ? k_graph_bad_action_label : unordered ? k_graph_bad_action_order : 0u;
}

/// The cursors of the merge of one action relation's runs: the rows of the graph with each schema of the relation form
/// one run (the rows are sorted by schema, so a run is the rows between the schema's start and the next schema's), and
/// each run is ascending in the binding. next() yields the distinct ground actions in ascending order of the binding,
/// each as the row of the first run (the lowest schema) that has it.
struct ActionRuns
{
    const ActionRows* rows;
    const u32* schemas;  // the relation's schemas, ascending
    u32* pos;            // the next row of each run: the rows' `cursors`
    u32 arity;
    u32 runs;
};

/// The runs of the action cell `c` in the rows `a` (of rows sorted by schema, with their starts).
MYMYR_HD void action_runs_begin(ActionRuns& r, const Cell& c, const u32* action_schemas, const ActionRows& a)
{
    r.rows = &a;
    r.schemas = action_schemas + c.a;
    r.pos = a.cursors;
    r.arity = c.logical_arity;
    r.runs = c.b - c.a;
    for (u32 j = 0; j < r.runs; ++j)
        r.pos[j] = a.starts[r.schemas[j]];
}

/// The next distinct ground action: sets `row` to its row in the graph's rows (the first run's, if several runs have it)
/// and advances every run past it. False once every run is exhausted.
MYMYR_HD bool action_runs_next(ActionRuns& r, u32& row)
{
    const ActionRows& a = *r.rows;
    const u32* starts = a.starts;
    u32 best = r.runs;
    for (u32 j = 0; j < r.runs; ++j)
    {
        if (r.pos[j] == starts[r.schemas[j] + 1u])
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
        if (r.pos[j] == starts[r.schemas[j] + 1u])
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
        return a.starts[schema + 1u] - a.starts[schema];
    }
    ActionRuns runs;
    action_runs_begin(runs, c, v.action_schemas, a);
    u32 n = 0, row = 0;
    while (action_runs_next(runs, row))
        ++n;
    return n;
}

/// The 64-bit words `effective_layers` needs for the subgoal layers `raw` of an instance.
MYMYR_HD u64 effective_layer_words(const InstanceView& v, const MaskLayers& raw)
{
    return u64{raw.layers} * 3u * (2u * v.words + (raw.derived_pos ? 2u * v.derived_words : 0u));
}

/// The levels of a row's subgoal layers: level l (layer l - 1) holds the literals of that layer that neither the goal
/// nor a layer below it has with the same sign, over the instance's words (derived words). The goal is the row's own or
/// the instance's (goal_words). For each level and sign, the plain mask, the part of it whose literals the state
/// (`atoms`, `derived`: null is all clear) satisfies and the part it does not, written to `storage`
/// (effective_layer_words). A mask without a literal, and one the layers lack, is null.
MYMYR_HD LevelPlanes effective_layers(const InstanceView& v, const GoalWords& goal, const MaskLayers& raw,
                                      const u64* atoms, const u64* derived, u64* storage)
{
    LevelPlanes out{};
    const auto fill = [&](bool derived_slots, const u64* layer_pos, const u64* layer_neg, u32 layer_words, u32 words,
                          const u64* goal_pos, const u64* goal_neg, const u64* state)
    {
        for (u32 l = 0; l < raw.layers; ++l)
            for (u32 sign = 0; sign < 2; ++sign)
            {
                const u64* layer = sign ? layer_neg : layer_pos;
                const u64* lowest = sign ? goal_neg : goal_pos;
                u64* plain = storage;
                u64* satisfied = storage + words;
                u64* unsatisfied = storage + 2u * u64{words};
                storage += 3u * u64{words};
                u64 any = 0, any_satisfied = 0, any_unsatisfied = 0;
                for (u32 w = 0; w < words; ++w)
                {
                    u64 lower = lowest ? lowest[w] : u64{0};
                    for (u32 k = 0; k < l; ++k)
                        lower |= layer[u64{k} * layer_words + w];
                    const u64 m = layer[u64{l} * layer_words + w] & ~lower;
                    const u64 x = state ? state[w] : u64{0};
                    // a positive literal holds iff its bit is set, a negative one iff it is clear
                    plain[w] = m;
                    satisfied[w] = m & (sign ? ~x : x);
                    unsatisfied[w] = m & (sign ? x : ~x);
                    any |= m;
                    any_satisfied |= satisfied[w];
                    any_unsatisfied |= unsatisfied[w];
                }
                const u32 kind = ((l + 1u) << 4) | (derived_slots ? 8u : 0u) | (sign ? 4u : 0u);
                out.plane[kind | static_cast<u32>(GoalMode::Plain)] = any ? plain : nullptr;
                out.plane[kind | static_cast<u32>(GoalMode::PlainTarget)] = any ? plain : nullptr;
                out.plane[kind | static_cast<u32>(GoalMode::Satisfied)] = any_satisfied ? satisfied : nullptr;
                out.plane[kind | static_cast<u32>(GoalMode::Unsatisfied)] = any_unsatisfied ? unsatisfied : nullptr;
            }
    };
    if (raw.pos && raw.neg)
        fill(false, raw.pos, raw.neg, raw.words, v.words, goal.pos, goal.neg, atoms);
    if (raw.derived_pos && raw.derived_neg)
        fill(true, raw.derived_pos, raw.derived_neg, raw.derived_words, v.derived_words, goal.derived_pos,
             goal.derived_neg, derived);
    return out;
}

/// The mask a goal cell takes its literals from: the positive or negative literals over the fluent or derived slots of
/// the goal, or, for a cell of a subgoal level (1 to 3), of that level of the row's subgoal layers (effective_layers);
/// null if there is none (no tuples).
template<bool Layered = true>
MYMYR_HD_HOT const u64* goal_cell_mask(const Cell& c, const GoalWords& g, const RowInput& in)
{
    const u32 kind = c.op - k_goal_op_base;
    if (Layered && kind >= 16u)
        return in.layers ? in.layers->levels.plane[kind & 63u] : nullptr;
    return (kind & 8u) ? ((kind & 4u) ? g.derived_neg : g.derived_pos) : ((kind & 4u) ? g.neg : g.pos);
}

/// The words of the goal tuples a goal cell takes from its slot range, resolved once per cell: its mask
/// (goal_cell_mask; null: no tuples), which the satisfied and unsatisfied variants cut down by the state words: a
/// positive literal holds iff its bit is set, a negative one iff it is clear.
struct GoalCellWords
{
    const u64* mask;
    const u64* state;  // the state words of a satisfied or unsatisfied variant (null: all clear)
    u64 first;         // the bits of the first word that are in the range, and of the last word
    u64 last;
    u64 flip;          // the satisfied and unsatisfied variants take the state word xor `flip`; the others all ones
    u64 keep;          // all ones for a satisfied or unsatisfied variant, else zero
    u32 first_word, last_word;

    /// Word `w` of the range, w from first_word to last_word.
    MYMYR_HD_HOT u64 word(u32 w) const
    {
        u64 m = mask[w];
        if (w == first_word)
            m &= first;
        if (w == last_word)
            m &= last;
        if (keep)
            m &= (state ? state[w] : u64{0}) ^ flip;
        return m;
    }
};

template<bool Layered = true>
MYMYR_HD_HOT GoalCellWords goal_cell_words(const Cell& c, const GoalWords& g, const RowInput& in)
{
    const u32 kind = c.op - k_goal_op_base;
    // the masks of a subgoal level are cut down by the state already
    const bool plain = (kind & 3u) < 2u || (Layered && kind >= 16u);
    const bool wants_set = ((kind & 3u) == static_cast<u32>(GoalMode::Satisfied)) != ((kind & 4u) != 0);
    GoalCellWords out{c.a < c.b ? goal_cell_mask<Layered>(c, g, in) : nullptr,
                      (kind & 8u) ? in.derived : in.atoms,
                      ~u64{0} << (c.a & 63u),
                      ~u64{0} >> (63u - ((c.b - 1u) & 63u)),
                      wants_set ? u64{0} : ~u64{0},
                      plain ? u64{0} : ~u64{0},
                      c.a >> 6,
                      (c.b - 1u) >> 6};
    return out;
}

/// The tuples of a goal cell in this graph: those its masks give plus the cell's constant part (the instance's goal only).
template<bool Layered = true>
MYMYR_HD u32 goal_cell_count(const Cell& c, const GoalWords& g, const RowInput& in)
{
    if (Layered && c.op >= k_goal_op_base + 16u)  // a subgoal level has no constant part
    {
        const u64* mask = in.layers ? in.layers->levels.plane[(c.op - k_goal_op_base) & 63u] : nullptr;
        if (!mask || c.a >= c.b)
            return 0;
        const u32 last = (c.b - 1u) >> 6;
        const u64 head = ~u64{0} << (c.a & 63u), tail = ~u64{0} >> (63u - ((c.b - 1u) & 63u));
        if ((c.a >> 6) == last)
            return popcount64(mask[last] & head & tail);
        u32 n = popcount64(mask[c.a >> 6] & head) + popcount64(mask[last] & tail);
        for (u32 w = (c.a >> 6) + 1u; w < last; ++w)
            n += popcount64(mask[w]);
        return n;
    }
    u32 n = g.constants ? c.d - c.c : 0u;
    const GoalCellWords words = goal_cell_words<Layered>(c, g, in);
    if (words.mask)
        for (u32 w = c.a >> 6, last = (c.b - 1u) >> 6; w <= last; ++w)
            if (const u64 m = words.word(w))
                n += popcount64(m);
    return n;
}

/// The masks a history cell takes literals from, resolved once per cell: for each used step j of the row, the positive
/// or negative literals over the fluent or derived slots in the cell's slot range [a, b), the words first_word to
/// last_word of `base + j * stride` cut to the range. `base` is null if the row has no such masks, or the range is
/// empty.
struct HistoryCellWords
{
    const u64* base;
    u32 stride;
    u32 steps;
    u32 first_word, last_word;
    u64 head, tail;

    /// Word `w` of step `j`.
    MYMYR_HD_HOT u64 word(u32 j, u32 w) const
    {
        u64 m = base[static_cast<u64>(j) * stride + w];
        if (w == first_word)
            m &= head;
        if (w == last_word)
            m &= tail;
        return m;
    }
};

MYMYR_HD_HOT HistoryCellWords history_cell_words(const Cell& c, const RowInput& in)
{
    HistoryCellWords out{nullptr, 0, 0, c.a >> 6, (c.b - 1u) >> 6, ~u64{0} << (c.a & 63u),
                         ~u64{0} >> (63u - ((c.b - 1u) & 63u))};
    if (!in.layers || c.a >= c.b)
        return out;
    const MaskLayers& m = in.layers->history.masks;
    const bool derived = history_op_derived(c.op);
    const bool negative = history_op_negative(c.op);
    out.base = derived ? (negative ? m.derived_neg : m.derived_pos) : (negative ? m.neg : m.pos);
    out.stride = derived ? m.derived_words : m.words;
    out.steps = m.layers;
    return out;
}

/// The tuples of a history cell in this graph: the literals of its predicate and sign in the slots [a, b) of the row's
/// used steps, summed over the steps.
MYMYR_HD u32 history_cell_count(const Cell& c, const RowInput& in)
{
    const HistoryCellWords words = history_cell_words(c, in);
    if (!words.base)
        return 0;
    u32 n = 0;
    for (u32 j = 0; j < words.steps; ++j)
        for (u32 w = words.first_word; w <= words.last_word; ++w)
            n += popcount64(words.word(j, w));
    return n;
}

/// The words of a fluent or derived cell from which on the lanes of a group count the cell together.
inline constexpr u32 k_shared_count_words = 4;

/// Whether the lanes of a group count the cell together (a fluent or derived cell over many words) and not the one lane
/// that owns the cell.
MYMYR_HD bool counted_by_group(const Cell& c, const RowInput& in)
{
    const bool words = c.op == static_cast<u32>(CellOp::Fluent) || (c.op == static_cast<u32>(CellOp::Derived) && in.derived);
    return words && c.a < c.b && ((c.b - 1u) >> 6) - (c.a >> 6) + 1u >= k_shared_count_words;
}

/// The tuples of one cell in this graph. `Layered` false: the row has no subgoal layers or history steps.
template<bool Layered = true>
MYMYR_HD_HOT u32 cell_count(const InstanceView& v, const Cell& c, const RowInput& in)
{
    if (c.op == static_cast<u32>(CellOp::Fluent))
        return popcount_range(in.atoms, c.a, c.b);
    if (c.op == static_cast<u32>(CellOp::Derived))
        return in.derived ? popcount_range(in.derived, c.a, c.b) : 0u;
    if (c.op == static_cast<u32>(CellOp::Static))
        return c.b - c.a;
    if (is_goal_op(c.op))
        return goal_cell_count<Layered>(c, goal_words(v, in), in);
    if (Layered && is_history_op(c.op))
        return history_cell_count(c, in);
    return action_cell_count(v, c, in.actions);
}


/// Adds a cell with `n` tuples to the totals of its graph: tuples, arguments, the goal entities (the tuples of a plain
/// goal cell with a goal target; the subgoal entities likewise), the action entities (the tuples of an action cell) and
/// the auxiliary flag (a nullary cell with a tuple, goal cells included; an action relation has no auxiliary slot, its
/// action entity is its only entity). The entity count, the used predicates and the bad-row flags are set by
/// count_graph and finish_totals.
template<bool Layered = true>
MYMYR_HD void accumulate_cell(GraphTotals& t, const Cell& c, u32 n)
{
    t.tuples += n;
    t.args += static_cast<u64>(n) * c.arity;
    if (c.op <= static_cast<u32>(CellOp::Static))
    {
        if (c.logical_arity == 0 && n > 0)
            t.flags |= k_graph_auxiliary;
        return;
    }
    if (is_action_op(c.op))
    {
        t.actions += n;
        if (c.op == static_cast<u32>(CellOp::ActionTarget))
            t.targets += n;
        return;
    }
    if (c.logical_arity == 0 && n > 0)
        t.flags |= k_graph_auxiliary;
    if (is_goal_op(c.op))
    {
        if (goal_op_mode(c.op) == GoalMode::PlainTarget)
        {
            if (!Layered || goal_op_level(c.op) == 0)
                t.goal_entities += n;
            else
                t.subgoal_entities += n;
            t.targets += n;
        }
    }
    else if (Layered && history_op_target(c.op))
    {
        t.history_targets += n;
        t.targets += n;
    }
}

/// The totals of a graph with no cell accumulated yet.
MYMYR_HD GraphTotals empty_totals()
{
    GraphTotals t;
    t.tuples = 0;
    t.args = 0;
    t.entities = 0;
    t.flags = 0;
    t.targets = 0;
    t.goal_entities = 0;
    t.actions = 0;
    t.virtuals = 0;
    t.subgoal_entities = 0;
    t.history_steps = 0;
    t.history_targets = 0;
    return t;
}

/// The entities of a graph: the objects, the auxiliary entity if there is one, the goal, subgoal, action, history
/// step, history target and virtual entities. `Layered` false: the graph has no subgoal layers or history steps.
template<bool Layered = true>
MYMYR_HD u32 entity_count(const GraphTotals& t, const InstanceView& v)
{
    u32 n = v.num_objects + ((t.flags & k_graph_auxiliary) ? 1u : 0u) + t.goal_entities + t.actions + t.virtuals;
    if (Layered)
        n += t.subgoal_entities + t.history_steps + t.history_targets;
    return n;
}

/// Completes the totals once every cell of the graph is accumulated: the entities and the bad-row flags.
template<bool Layered = true>
MYMYR_HD void finish_totals(GraphTotals& t, const InstanceView& v, const RowInput& in)
{
    t.entities = entity_count<Layered>(t, v);
    if (has_foreign_atom_bits(v, in.atoms) || (in.derived && has_foreign_derived_bits(v, in.derived)))
        t.flags |= k_graph_bad_row;
    if (has_foreign_goal_bits(v, in))
        t.flags |= k_graph_bad_goal;
    if (Layered && in.layers)
    {
        if (has_foreign_layer_bits(v, in.layers->subgoals))
            t.flags |= k_graph_bad_layers;
        if (has_foreign_layer_bits(v, in.layers->history.masks))
            t.flags |= k_graph_bad_history;
    }
}

/// Whether the context has predicate virtual nodes.
MYMYR_HD bool has_virtual_nodes(const InstanceView& v) { return v.relation_virtual != nullptr; }

/// The sizes of one graph. With `cell_counts` non-null it also writes the counts of the instance's cells: one entry
/// per cell, in the order of cell_list(v, given) (one entry per cell, not one per relation; the relation of entry i is
/// the relation of the list's cell i).
/// Atom words past the instance's `words` are never read. With predicate virtual nodes `virtual_scratch` must hold
/// v.num_virtual words (it is overwritten); the totals then count the predicates that have a tuple. The expansion rows
/// of the graph are checked (check_actions, which writes the rows' `starts`): a violation sets k_graph_bad_action_order
/// or k_graph_bad_action_label, and the counts of the action cells are then meaningless.
template<bool Layered = true>
MYMYR_HD GraphTotals count_graph(const InstanceView& v, const RowInput& in, u32* cell_counts,
                                 u32* virtual_scratch = nullptr)
{
    GraphTotals t = empty_totals();
    if (Layered && in.layers)
        t.history_steps = in.layers->history.masks.layers;
    const bool virtual_nodes = has_virtual_nodes(v) && virtual_scratch;
    if (virtual_nodes)
        for (u32 p = 0; p < v.num_virtual; ++p)
            virtual_scratch[p] = 0;
    t.flags |= in.actions.count ? check_actions(v, in.actions) : 0u;
    const CellList list = cell_list<Layered>(v, in);
    const u32 num_cells = list.count;
    const Cell* cells = list.cells;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const Cell& c = cells[i];
        const u32 n = cell_count<Layered>(v, c, in);
        if (cell_counts)
            cell_counts[i] = n;
        if (n == 0)
            continue;
        accumulate_cell<Layered>(t, c, n);
        if (virtual_nodes)
        {
            const u32 p = v.relation_virtual[c.relation];
            if (p != k_no_virtual)
                virtual_scratch[p] = 1;
        }
    }
    if (virtual_nodes)
        for (u32 p = 0; p < v.num_virtual; ++p)
            t.virtuals += virtual_scratch[p];
    finish_totals<Layered>(t, v, in);
    return t;
}

/// count_graph with the cells shared among the lanes of a group: lane l counts the cells l, l + size(), ... and the
/// partial totals are summed, so every lane returns the totals count_graph returns. The count of cell i is reported to
/// `sink(i, cell, n)` by the lane that counted it. The graph's expansion rows are not read (a graph with expansion
/// rows is counted by count_graph). With predicate virtual nodes `virtual_scratch` (v.num_virtual words, shared by the
/// lanes) is overwritten. The group may be a part of a warp; all of its lanes call this function together. The row has
/// no subgoal layers or history steps (expect_no_layers).
template<class Group, class Sink>
MYMYR_HD GraphTotals count_graph_group_into(const Group& grp, const InstanceView& v, const RowInput& in,
                                            const Sink& sink, u32* virtual_scratch)
{
    expect_no_layers(in);
    const u32 lane = grp.lane(), lanes = grp.size();
    GraphTotals t = empty_totals();
    const bool virtual_nodes = has_virtual_nodes(v) && virtual_scratch;
    if (virtual_nodes)
    {
        grp.sync();
        for (u32 p = lane; p < v.num_virtual; p += lanes)
            virtual_scratch[p] = 0;
        grp.sync();
    }
    const CellList list = cell_list<false>(v, in);
    for (u32 i = 0; i < list.count; ++i)
    {
        const Cell& c = list.cells[i];
        u32 n = 0;
        bool mine = i % lanes == lane;
        if (counted_by_group(c, in))
        {
            // the lanes share the cell's words; lane 0 reports it
            const u64* words = c.op == static_cast<u32>(CellOp::Fluent) ? in.atoms : in.derived;
            u32 own = 0;
            for (u32 w = (c.a >> 6) + lane; w <= ((c.b - 1u) >> 6); w += lanes)
                own += popcount64(range_word(words, w, c.a, c.b));
            n = grp.sum(own);
            mine = lane == 0;
        }
        else if (mine)
            n = cell_count(v, c, in);
        if (!mine)
            continue;
        sink(i, c, n);
        accumulate_cell(t, c, n);
        if (virtual_nodes && n)
        {
            const u32 p = v.relation_virtual[c.relation];
            if (p != k_no_virtual)
                virtual_scratch[p] = 1;
        }
    }
    if (virtual_nodes)
    {
        grp.sync();
        for (u32 p = lane; p < v.num_virtual; p += lanes)
            t.virtuals += virtual_scratch[p];
    }
    t.tuples = grp.sum(t.tuples);
    t.args = grp.sum(t.args);
    t.targets = grp.sum(t.targets);
    t.goal_entities = grp.sum(t.goal_entities);
    t.actions = grp.sum(t.actions);
    t.virtuals = grp.sum(t.virtuals);
    t.flags = grp.bit_or(t.flags);
    t.entities = entity_count(t, v);
    if (any_bit_from_group(grp, in.atoms, v.words, v.fluent_slots) ||
        (in.derived && any_bit_from_group(grp, in.derived, v.derived_words, v.derived_slots)))
        t.flags |= k_graph_bad_row;
    if (has_foreign_goal_bits_group(grp, v, in))
        t.flags |= k_graph_bad_goal;
    return t;
}

/// count_graph_group writing the count of cell i to cell_counts[i] (not null).
template<class Group>
MYMYR_HD GraphTotals count_graph_group(const Group& grp, const InstanceView& v, const RowInput& in, u32* cell_counts,
                                       u32* virtual_scratch = nullptr)
{
    return count_graph_group_into(
        grp, v, in, [cell_counts](u32 i, const Cell&, u32 n) { cell_counts[i] = n; }, virtual_scratch);
}

/// The totals of a graph from the counts of its cells, `count(i, cell)` for the cell i of cell_list(v, in), without
/// counting the cells again. The goal entities, targets and the auxiliary flag are set; `entities` and `virtuals` are
/// not (a fill needs neither).
template<class CountOf>
MYMYR_HD GraphTotals totals_from_counts(const InstanceView& v, const RowInput& in, const CountOf& count)
{
    expect_no_layers(in);
    GraphTotals t = empty_totals();
    const CellList list = cell_list<false>(v, in);
    for (u32 i = 0; i < list.count; ++i)
        accumulate_cell(t, list.cells[i], count(i, list.cells[i]));
    return t;
}

/// The virtual rank of every predicate in this graph: `rank[p]` for virtual index p is the number of used predicates
/// below it if p has a tuple in the graph, else k_no_virtual. `cell_counts` are the counts of count_graph (null: they are
/// recounted). Returns the number of used predicates. `rank` holds v.num_virtual words.
template<bool Layered = true>
MYMYR_HD u32 virtual_ranks(const InstanceView& v, const RowInput& in, const u32* cell_counts, u32* rank)
{
    const u32 P = v.num_virtual;
    for (u32 p = 0; p < P; ++p)
        rank[p] = 0;
    const CellList list = cell_list<Layered>(v, in);
    for (u32 i = 0; i < list.count; ++i)
    {
        const u32 p = v.relation_virtual[list.cells[i].relation];
        if (p == k_no_virtual || rank[p])
            continue;
        if (cell_counts ? cell_counts[i] != 0 : cell_count<Layered>(v, list.cells[i], in) != 0)
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
    I* target_entity = nullptr;   // [T] the graphs' target entities (goal, subgoal, action, history), in entity order
    u8* target_group = nullptr;   // [T] the group code of each target
    I* history_offsets = nullptr;  // [G + 1]; null: no history arrays (they are stored together)
    I* history_entity = nullptr;   // [S] the history step entities of the graphs, in step order
    i32* history_dt = nullptr;     // [S] the dt of each step
};

/// The segment of cell (g, r) among the G * R cells of a batch with R relations: g * R + r graph-major, r * G + g
/// relation-major.
MYMYR_HD u64 cell_segment(bool relation_major, u64 g, u32 r, u64 G, u32 R)
{
    return relation_major ? static_cast<u64>(r) * G + g : g * R + r;
}

/// The group codes of targets (TargetGroup in schema.hpp) and the role of goal entities (EntityRole::Goal).
inline constexpr u8 k_target_group_goal = 0;
inline constexpr u8 k_target_group_subgoal = 1;
inline constexpr u8 k_target_group_action = 2;
inline constexpr u8 k_target_group_history = 3;
inline constexpr u8 k_role_goal = 2;

/// The first entity of each block of a graph past its objects and auxiliary entity (batch-global ids): the goal
/// entities, the subgoal entities, the action entities, the history step entities, the history target entities and the
/// virtual entities, in this order.
struct EntityBlocks
{
    u64 goal;
    u64 subgoal;
    u64 action;
    u64 history_step;
    u64 history_target;
    u64 virtual_nodes;
};

template<bool Layered = true>
MYMYR_HD EntityBlocks entity_blocks(const InstanceView& v, const GraphTotals& totals, u64 entity_base)
{
    EntityBlocks blocks;
    blocks.goal = entity_base + v.num_objects + ((totals.flags & k_graph_auxiliary) ? 1u : 0u);
    blocks.subgoal = blocks.goal + totals.goal_entities;
    blocks.action = blocks.subgoal + (Layered ? totals.subgoal_entities : 0u);
    blocks.history_step = blocks.action + totals.actions;
    blocks.history_target = blocks.history_step + (Layered ? totals.history_steps : 0u);
    blocks.virtual_nodes = blocks.history_target + (Layered ? totals.history_targets : 0u);
    return blocks;
}

/// Writes num_objects[g], goal_complete[g] and target_offsets[g] (when stored; `target_base` is the batch position of
/// the graph's first target).
template<class I>
MYMYR_HD void fill_graph_header(const InstanceView& v, const RowInput& in, u64 g, u64 target_base,
                                const GraphOut<I>& out)
{
    out.num_objects[g] = static_cast<I>(v.num_objects);
    if (out.goal_complete)
        out.goal_complete[g] = static_cast<u8>(goal_is_complete(v, in));
    if (out.target_offsets)
        out.target_offsets[g] = static_cast<I>(target_base);
}

/// Writes the entities [begin, end) of the graph's objects (role and reference) from `entity_base` on, unless the entity
/// arrays are not stored.
template<class I>
MYMYR_HD void fill_objects(u32 begin, u32 end, u64 entity_base, const GraphOut<I>& out)
{
    if (!out.entity_role)
        return;
    u8* role = out.entity_role + entity_base;
    I* ref = out.entity_ref + entity_base;
    for (u32 o = begin; o < end; ++o)
    {
        role[o] = k_role_object;
        ref[o] = static_cast<I>(o);
    }
}

/// Writes the auxiliary entity, which follows the `num_objects` objects of the graph from `entity_base` on.
template<class I>
MYMYR_HD void fill_auxiliary_entity(u32 num_objects, u64 entity_base, const GraphOut<I>& out)
{
    if (!out.entity_role)
        return;
    out.entity_role[entity_base + num_objects] = k_role_auxiliary;
    out.entity_ref[entity_base + num_objects] = static_cast<I>(-1);
}

/// The graph's header (fill_graph_header), its objects, then the auxiliary entity if `auxiliary`
/// (totals.flags & k_graph_auxiliary). The goal and action entities that follow are written with their tuples
/// (fill_cell_at), the virtual entities by fill_virtual_entities.
template<class I>
MYMYR_HD void fill_entities(const InstanceView& v, const RowInput& in, bool auxiliary, u64 g, u64 entity_base,
                            u64 target_base, const GraphOut<I>& out)
{
    fill_graph_header<I>(v, in, g, target_base, out);
    fill_objects<I>(0, v.num_objects, entity_base, out);
    if (auxiliary)
        fill_auxiliary_entity<I>(v.num_objects, entity_base, out);
}

/// Writes history_offsets[g] (`history_base` is the batch position of the graph's first history step) and, for each
/// used step j of the graph, its entity at `blocks.history_step` + j (role and reference unless the entity arrays are
/// not stored) and its entry of history_entity and history_dt. Does nothing without history arrays.
template<class I>
MYMYR_HD void fill_history_entities(const RowInput& in, const EntityBlocks& blocks, u64 g, u64 history_base,
                                    const GraphOut<I>& out)
{
    if (!out.history_offsets)
        return;
    out.history_offsets[g] = static_cast<I>(history_base);
    if (!in.layers)
        return;
    const HistoryRows& h = in.layers->history;
    for (u32 j = 0; j < h.masks.layers; ++j)
    {
        const u64 entity = blocks.history_step + j;
        out.history_entity[history_base + j] = static_cast<I>(entity);
        out.history_dt[history_base + j] = h.dt[j];
        if (out.entity_role)
        {
            out.entity_role[entity] = k_role_history;
            out.entity_ref[entity] = static_cast<I>(h.first + j);
        }
    }
}

/// Writes the virtual entity of predicate `p` if it has a rank: at `blocks.virtual_nodes` + its rank, with the
/// predicate's key relation as reference. `rank` is the result of virtual_ranks.
template<class I>
MYMYR_HD void fill_virtual_entity(const InstanceView& v, const EntityBlocks& blocks, const u32* rank, u32 p,
                                  const GraphOut<I>& out)
{
    if (!out.entity_role || rank[p] == k_no_virtual)
        return;
    const u64 entity = blocks.virtual_nodes + rank[p];
    out.entity_role[entity] = k_role_predicate_virtual;
    out.entity_ref[entity] = static_cast<I>(v.virtual_relation[p]);
}

/// Writes the virtual entities of a graph: one per predicate with a rank (fill_virtual_entity).
template<class I>
MYMYR_HD void fill_virtual_entities(const InstanceView& v, const EntityBlocks& blocks, const u32* rank,
                                    const GraphOut<I>& out)
{
    for (u32 p = 0; p < v.num_virtual; ++p)
        fill_virtual_entity<I>(v, blocks, rank, p, out);
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

/// The arity of a state cell that fill_cell_of instantiates for: 0 to 4 objects, or any other (k_any_arity, the arity
/// read from the cell).
inline constexpr u32 k_any_arity = ~u32{0};

/// Writes the tuples of a state cell of K objects (K == k_any_arity: `arity` objects): the tuple of a row of instance-local
/// object ids is, optionally behind the predicate entity `pred`, the entity ids of its objects (the entity `auxiliary`
/// for a nullary predicate). The arity is a template argument so that the loops over the tuples have no per-tuple
/// dispatch.
template<class I, u32 K, bool Pred>
struct StateTupleWriter
{
    u64 entity_base;
    u64 auxiliary;
    u64 pred;
    u32 arity;  // the cell's objects, read when K == k_any_arity

    /// Writes one tuple at `dst` and returns the position after it.
    MYMYR_HD I* operator()(I* dst, const i32* row) const
    {
        if constexpr (Pred)
            *dst++ = static_cast<I>(pred);
        if constexpr (K == 0)
            *dst++ = static_cast<I>(auxiliary);
        else if constexpr (K == k_any_arity)
        {
            if (arity == 0)
                *dst++ = static_cast<I>(auxiliary);
            else
                for (u32 k = 0; k < arity; ++k)
                    *dst++ = static_cast<I>(entity_base + static_cast<u64>(row[k]));
        }
        else
            for (u32 k = 0; k < K; ++k)
                dst[k] = static_cast<I>(entity_base + static_cast<u64>(row[k]));
        if constexpr (K != 0 && K != k_any_arity)
            dst += K;
        return dst;
    }
};

/// Writes the tuples of the bits `bits` of word `w` of a fluent or derived cell, in ascending slot order, from `dst` on;
/// `table` holds the objects of each slot (rows of `stride` entries). Returns the position after them.
template<class I, u32 K, bool Pred>
MYMYR_HD I* emit_word_tuples(const StateTupleWriter<I, K, Pred>& write, const i32* table, u64 stride, u32 w, u64 bits,
                             I* dst)
{
    while (bits)
    {
        const u64 s = static_cast<u64>(w) * 64u + lowest_bit64(bits);
        bits &= bits - 1u;
        dst = write(dst, table + s * stride);
    }
    return dst;
}

/// Writes the arguments of the tuples of cell `c` in this graph to `dst` for a state cell of K objects (K ==
/// k_any_arity: of c.logical_arity objects); with Pred each tuple starts with the entity `pred`. Returns the number of
/// tuples.
template<class I, u32 K, bool Pred>
MYMYR_HD u32 fill_cell_of(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                          u64 pred, I* dst)
{
    const u64 A = v.max_arity;
    const StateTupleWriter<I, K, Pred> write{entity_base, auxiliary, pred, K == k_any_arity ? c.logical_arity : K};
    u32 n = 0;
    if (c.op == static_cast<u32>(CellOp::Static))
    {
        n = c.b - c.a;
        const i32* row = v.static_args + static_cast<u64>(c.a) * A;
        for (u32 s = 0; s < n; ++s, row += A)
            dst = write(dst, row);
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
            dst = write(dst, table + s * A);
        }
    }
    return n;
}

template<class I, bool Pred>
MYMYR_HD u32 fill_cell_dispatch(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                                u64 auxiliary, u64 pred, I* dst)
{
    switch (c.logical_arity)
    {
    case 0: return fill_cell_of<I, 0, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    case 1: return fill_cell_of<I, 1, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    case 2: return fill_cell_of<I, 2, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    case 3: return fill_cell_of<I, 3, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    case 4: return fill_cell_of<I, 4, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    default: return fill_cell_of<I, k_any_arity, Pred>(v, c, in, entity_base, auxiliary, pred, dst);
    }
}

/// Writes the arguments of the tuples of cell `c` in this graph to `dst`, in tuple order, and returns the number of
/// tuples (the cell's count). Object ids become entity ids by adding `entity_base`; the tuple of a nullary cell is
/// the entity `auxiliary`; each tuple starts with the entity `pred` unless it is k_no_entity. Writes count * c.arity
/// entries; the caller places `dst`.
template<class I>
MYMYR_HD u32 fill_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                       u64 pred, I* dst)
{
    if (pred == k_no_entity)
        return fill_cell_dispatch<I, false>(v, c, in, entity_base, auxiliary, pred, dst);
    return fill_cell_dispatch<I, true>(v, c, in, entity_base, auxiliary, pred, dst);
}

/// Writes the arguments of the tuples of goal cell `c` in this graph to `dst` for a goal cell of K objects (K ==
/// k_any_arity: of c.logical_arity objects), with goal targets when Target, and returns the number of tuples: the
/// cell's mask tuples (ascending slots) merged with its constant rows (the instance's goal only), in lexicographic
/// order of their objects. With goal targets each tuple starts with its goal entity, the ids `first_target`,
/// `first_target + 1`, ... in tuple order. A predicate entity `pred` (unless k_no_entity) follows it; the tuple of a
/// nullary predicate is the entity `auxiliary`. Writes count * c.arity entries.
template<class I, u32 K, bool Target, bool Layered = true>
MYMYR_HD u32 fill_goal_cell_of(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                               u64 auxiliary, u64 first_target, u64 pred, I* dst)
{
    const GoalWords g = goal_words(v, in);
    const u64 A = v.max_arity;
    const u32 arity = K == k_any_arity ? c.logical_arity : K;
    const i32* table = goal_op_derived(c.op) ? v.derived_args : v.slot_args;
    const i32* constant = v.goal_args + static_cast<u64>(c.c) * A;
    u32 constants_left = g.constants ? c.d - c.c : 0u;
    u32 n = 0;
    auto emit = [&](const i32* row)
    {
        if constexpr (Target)
            *dst++ = static_cast<I>(first_target + n);
        ++n;
        if (pred != k_no_entity)
            *dst++ = static_cast<I>(pred);
        if constexpr (K == 0)
            *dst++ = static_cast<I>(auxiliary);
        else if constexpr (K == k_any_arity)
        {
            if (arity == 0)
                *dst++ = static_cast<I>(auxiliary);
            else
                for (u32 k = 0; k < arity; ++k)
                    *dst++ = static_cast<I>(entity_base + static_cast<u64>(row[k]));
        }
        else
        {
            for (u32 k = 0; k < K; ++k)
                dst[k] = static_cast<I>(entity_base + static_cast<u64>(row[k]));
            dst += K;
        }
    };
    if (const GoalCellWords words = goal_cell_words<Layered>(c, g, in); words.mask)
        for (u32 w = c.a >> 6; w <= ((c.b - 1u) >> 6); ++w)
        {
            u64 bits = words.word(w);
            while (bits)
            {
                const u64 s = static_cast<u64>(w) * 64u + lowest_bit64(bits);
                bits &= bits - 1u;
                const i32* row = table + s * A;
                while (constants_left && tuple_before(constant, row, arity))
                {
                    emit(constant);
                    constant += A;
                    --constants_left;
                }
                emit(row);
            }
        }
    for (; constants_left; --constants_left, constant += A)
        emit(constant);
    return n;
}

template<class I, bool Target, bool Layered = true>
MYMYR_HD u32 fill_goal_cell_dispatch(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                                     u64 auxiliary, u64 first_target, u64 pred, I* dst)
{
    switch (c.logical_arity)
    {
    case 0: return fill_goal_cell_of<I, 0, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    case 1: return fill_goal_cell_of<I, 1, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    case 2: return fill_goal_cell_of<I, 2, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    case 3: return fill_goal_cell_of<I, 3, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    case 4: return fill_goal_cell_of<I, 4, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    default:
        return fill_goal_cell_of<I, k_any_arity, Target, Layered>(v, c, in, entity_base, auxiliary, first_target, pred,
                                                                  dst);
    }
}

/// Writes the arguments of the tuples of goal cell `c` in this graph to `dst`, in lexicographic order of their objects,
/// and returns the number of tuples: the cell's mask tuples (ascending slots) merged with its constant rows (the
/// instance's goal only). With goal targets (GoalMode::PlainTarget) each tuple starts with its goal entity, the ids
/// `first_target`, `first_target + 1`, ... in tuple order. A predicate entity `pred` (unless k_no_entity) follows it.
/// Writes count * c.arity entries.
template<class I, bool Layered = true>
MYMYR_HD u32 fill_goal_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                            u64 first_target, u64 pred, I* dst)
{
    if (goal_op_mode(c.op) == GoalMode::PlainTarget)
        return fill_goal_cell_dispatch<I, true, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
    return fill_goal_cell_dispatch<I, false, Layered>(v, c, in, entity_base, auxiliary, first_target, pred, dst);
}

/// Where the goal, subgoal, action and history target entities of a graph go: the blocks, the number of each written so
/// far (the cells are filled in relation order, so the running count is the position among the graph's entities of that
/// kind), where the graph's targets start in the target arrays, the graph's goal and subgoal entities (the action
/// targets follow them, then the history targets at `history_target_at`), and the virtual rank of each predicate (null
/// without virtual nodes).
///
/// A fill that writes the cells of one graph on several lanes (one lane per cell, or per run of cells) gives each lane
/// its own cursor and seeds it from the counts of count_graph (cell_counts, in cell order): `goals_written` is the sum
/// of the counts of the plain goal cells with goal targets before the lane's first cell, `subgoals_written`,
/// `actions_written` and `history_written` the same sums over the plain subgoal cells with subgoal targets, the action
/// cells and the history cells with history targets before it, and the `graph_tuple` of fill_cell_at the sum of the
/// counts of all cells before it. The sums are exclusive prefix sums over the graph's cells in cell order, so the lanes
/// need no communication.
struct FillCursor
{
    EntityBlocks blocks;
    u64 goals_written;
    u64 actions_written;
    u64 target_base;
    u64 goal_entities;
    const u32* rank;
    u64 subgoals_written = 0;
    u64 history_written = 0;
    u64 subgoal_entities = 0;
    u64 history_target_at = 0;  // where the history targets start in the target arrays
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

/// Writes the rows [begin, begin + n) of one action schema as the tuples of an action cell: for each row, its action
/// entity (`first` + the row's position) and the first K objects of its binding (rows of `label_width` entries) as
/// entity ids, the entity's role and reference (its row), and with Targets its target entry.
template<class I, u32 K, bool Targets>
MYMYR_HD void emit_action_rows(I* dst, const i32* binding, u32 label_width, u32 n, u32 begin, u64 first,
                               u64 entity_base, u8* role, I* ref, I* target_entity, u8* target_group)
{
    for (u32 k = 0; k < n; ++k, binding += label_width, dst += 1 + K)
    {
        dst[0] = static_cast<I>(first + k);
        for (u32 j = 0; j < K; ++j)
            dst[1 + j] = static_cast<I>(entity_base + static_cast<u64>(binding[j]));
        role[k] = k_role_action;
        ref[k] = static_cast<I>(begin + k);
        if constexpr (Targets)
        {
            target_entity[k] = static_cast<I>(first + k);
            target_group[k] = k_target_group_action;
        }
    }
}

template<class I, bool Targets>
MYMYR_HD void emit_action_rows_of(u32 arity, I* dst, const i32* binding, u32 label_width, u32 n, u32 begin, u64 first,
                                  u64 entity_base, u8* role, I* ref, I* target_entity, u8* target_group)
{
    switch (arity)
    {
    case 0:
        emit_action_rows<I, 0, Targets>(dst, binding, label_width, n, begin, first, entity_base, role, ref,
                                      target_entity, target_group);
        break;
    case 1:
        emit_action_rows<I, 1, Targets>(dst, binding, label_width, n, begin, first, entity_base, role, ref,
                                      target_entity, target_group);
        break;
    case 2:
        emit_action_rows<I, 2, Targets>(dst, binding, label_width, n, begin, first, entity_base, role, ref,
                                      target_entity, target_group);
        break;
    case 3:
        emit_action_rows<I, 3, Targets>(dst, binding, label_width, n, begin, first, entity_base, role, ref,
                                      target_entity, target_group);
        break;
    case 4:
        emit_action_rows<I, 4, Targets>(dst, binding, label_width, n, begin, first, entity_base, role, ref,
                                      target_entity, target_group);
        break;
    default:
        for (u32 k = 0; k < n; ++k, binding += label_width)
        {
            *dst++ = static_cast<I>(first + k);
            dst = emit_tuple<I>(dst, binding, arity, entity_base);
            role[k] = k_role_action;
            ref[k] = static_cast<I>(begin + k);
            if constexpr (Targets)
            {
                target_entity[k] = static_cast<I>(first + k);
                target_group[k] = k_target_group_action;
            }
        }
    }
}

/// Writes the tuples of action cell `c` in this graph to `dst`: for each distinct ground action, in ascending order of the
/// binding, its entity (the next action entity of the graph) and its objects; writes the action entities (role and
/// reference) and their target entries too. Returns the number of tuples. Writes count * c.arity entries.
template<class I>
MYMYR_HD u32 fill_action_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                              FillCursor& cursor, const GraphOut<I>& out, I* dst)
{
    const ActionRows& a = in.actions;
    if (a.count == 0)
        return 0;
    const u32 arity = c.logical_arity;
    const bool target = c.op == static_cast<u32>(CellOp::ActionTarget);
    const u64 first = cursor.blocks.action + cursor.actions_written;
    u32 n = 0;
    if (c.b - c.a == 1u && out.entity_role && (!target || out.target_entity))
    {
        // the cell's rows are the rows of its one schema: consecutive, in the order of the entities
        const u32 schema = v.action_schemas[c.a];
        const u32 begin = a.starts[schema];
        n = a.starts[schema + 1u] - begin;
        const i32* binding = a.binding + static_cast<u64>(begin) * a.label_width;
        u8* role = out.entity_role + first;
        I* ref = out.entity_ref + first;
        if (target)
        {
            const u64 at = cursor.target_base + cursor.goal_entities + cursor.subgoal_entities + cursor.actions_written;
            emit_action_rows_of<I, true>(arity, dst, binding, a.label_width, n, begin, first, entity_base, role, ref,
                                         out.target_entity + at, out.target_group + at);
        }
        else
            emit_action_rows_of<I, false>(arity, dst, binding, a.label_width, n, begin, first, entity_base, role, ref,
                                          nullptr, nullptr);
    }
    else
    {
        ActionRuns runs;
        action_runs_begin(runs, c, v.action_schemas, a);
        u32 row = 0;
        while (action_runs_next(runs, row))
        {
            const u64 entity = first + n;
            *dst++ = static_cast<I>(entity);
            if (arity)
                dst = emit_tuple<I>(dst, a.binding + static_cast<u64>(row) * a.label_width, arity, entity_base);
            if (out.entity_role)
            {
                out.entity_role[entity] = k_role_action;
                out.entity_ref[entity] = static_cast<I>(row);
            }
            if (target && out.target_entity)
            {
                const u64 at =
                    cursor.target_base + cursor.goal_entities + cursor.subgoal_entities + cursor.actions_written + n;
                out.target_entity[at] = static_cast<I>(entity);
                out.target_group[at] = k_target_group_action;
            }
            ++n;
        }
    }
    cursor.actions_written += n;
    return n;
}

/// Writes the tuples of history cell `c` in this graph to `dst`: for each used step in order, the literals of the
/// cell's predicate and sign in the step's mask, in ascending slot order, as [history target]? + [history step] +
/// [predicate]? + objects (the entity `auxiliary` for a nullary predicate). With a history target the tuples start with
/// the entities `first_target`, `first_target + 1`, ...; step j is the entity `first_step + j`. Returns the number of
/// tuples. Writes count * c.arity entries.
template<class I>
MYMYR_HD u32 fill_history_cell(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                               u64 first_target, u64 first_step, u64 pred, I* dst)
{
    const HistoryCellWords words = history_cell_words(c, in);
    if (!words.base)
        return 0;
    const bool target = history_op_target(c.op);
    const u64 A = v.max_arity;
    const u32 arity = c.logical_arity;
    const i32* table = history_op_derived(c.op) ? v.derived_args : v.slot_args;
    u32 n = 0;
    for (u32 j = 0; j < words.steps; ++j)
        for (u32 w = words.first_word; w <= words.last_word; ++w)
        {
            u64 bits = words.word(j, w);
            while (bits)
            {
                const u64 slot = static_cast<u64>(w) * 64u + lowest_bit64(bits);
                bits &= bits - 1u;
                if (target)
                    *dst++ = static_cast<I>(first_target + n);
                *dst++ = static_cast<I>(first_step + j);
                if (pred != k_no_entity)
                    *dst++ = static_cast<I>(pred);
                if (arity == 0)
                    *dst++ = static_cast<I>(auxiliary);
                else
                    dst = emit_tuple<I>(dst, table + slot * A, arity, entity_base);
                ++n;
            }
        }
    return n;
}

/// Writes a history cell of a graph and, with history targets, the entities of its tuples and their target entries (see
/// fill_cell_at). Kept out of line: the cells of the other lanes are the hot ones.
template<class I>
MYMYR_NOINLINE u32 fill_history_cell_at(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base,
                                        u64 auxiliary, u64 graph_tuple, u64 pred, FillCursor& cursor,
                                        const GraphOut<I>& out, I* dst)
{
    const u64 first = cursor.blocks.history_target + cursor.history_written;
    const u32 n = fill_history_cell<I>(v, c, in, entity_base, auxiliary, first, cursor.blocks.history_step, pred, dst);
    if (history_op_target(c.op))
    {
        for (u32 k = 0; k < n; ++k)
        {
            const u64 entity = first + k;
            if (out.entity_role)
            {
                out.entity_role[entity] = k_role_history_target;
                out.entity_ref[entity] = static_cast<I>(graph_tuple + k);
            }
            if (out.target_entity)
            {
                out.target_entity[cursor.history_target_at + cursor.history_written + k] = static_cast<I>(entity);
                out.target_group[cursor.history_target_at + cursor.history_written + k] = k_target_group_history;
            }
        }
        cursor.history_written += n;
    }
    return n;
}

/// Writes one cell of a graph: its tuples' arguments to `dst` (like fill_cell, fill_goal_cell for a goal cell,
/// fill_history_cell for a history cell and fill_action_cell for an action cell) and, for a plain goal or subgoal cell
/// with targets and a history cell with history targets, the entities of its tuples and their target entries.
/// `graph_tuple` is the index of the cell's first tuple among the graph's tuples in graph-major order, the sum of the
/// counts of the graph's cells before it (the `entity_ref` of its first target entity), whatever the layout. Returns
/// the number of tuples.
template<class I, bool Layered = true>
MYMYR_HD u32 fill_cell_at(const InstanceView& v, const Cell& c, const RowInput& in, u64 entity_base, u64 auxiliary,
                          u64 graph_tuple, FillCursor& cursor, const GraphOut<I>& out, I* dst)
{
    if (is_action_op(c.op))
        return fill_action_cell<I>(v, c, in, entity_base, cursor, out, dst);
    const u64 pred = predicate_entity(v, cursor, c.relation);
    if constexpr (Layered)
    {
        if (c.op < k_history_op_base)
            return fill_cell<I>(v, c, in, entity_base, auxiliary, pred, dst);
        if (!is_goal_op(c.op))
            return fill_history_cell_at<I>(v, c, in, entity_base, auxiliary, graph_tuple, pred, cursor, out, dst);
    }
    else if (!is_goal_op(c.op))
        return fill_cell<I>(v, c, in, entity_base, auxiliary, pred, dst);
    const bool subgoal = Layered && goal_op_level(c.op) != 0;
    const u64 first =
        subgoal ? cursor.blocks.subgoal + cursor.subgoals_written : cursor.blocks.goal + cursor.goals_written;
    const u32 n = fill_goal_cell<I, Layered>(v, c, in, entity_base, auxiliary, first, pred, dst);
    if (goal_op_mode(c.op) == GoalMode::PlainTarget)
    {
        const u64 written = subgoal ? cursor.subgoals_written : cursor.goals_written;
        const u64 at = subgoal ? cursor.target_base + cursor.goal_entities + written : cursor.target_base + written;
        for (u32 k = 0; k < n; ++k)
        {
            const u64 entity = first + k;
            if (out.entity_role)
            {
                out.entity_role[entity] = subgoal ? k_role_subgoal : k_role_goal;
                out.entity_ref[entity] = static_cast<I>(graph_tuple + k);
            }
            if (out.target_entity)
            {
                out.target_entity[at + k] = static_cast<I>(entity);
                out.target_group[at + k] = subgoal ? k_target_group_subgoal : k_target_group_goal;
            }
        }
        if (subgoal)
            cursor.subgoals_written += n;
        else
            cursor.goals_written += n;
    }
    return n;
}

/// The start of a graph's fill: its entities (the objects, the auxiliary entity, and the virtual entities, which need the
/// ranks of the predicates, computed into `scratch` from `cell_counts` (null: recounted)) and the cursor of its goal and
/// action entities. `scratch` must hold v.num_virtual words with predicate virtual nodes and may be null otherwise.
template<class I, bool Layered = true>
MYMYR_HD FillCursor begin_graph(const InstanceView& v, const RowInput& in, u64 g, u64 entity_base,
                                const GraphTotals& totals, const GraphOut<I>& out, u64 target_base,
                                const u32* cell_counts, u32* scratch, u64 history_base = 0)
{
    const bool auxiliary = (totals.flags & k_graph_auxiliary) != 0;
    fill_entities<I>(v, in, auxiliary, g, entity_base, target_base, out);
    FillCursor cursor{entity_blocks<Layered>(v, totals, entity_base), 0, 0, target_base, totals.goal_entities, nullptr};
    if constexpr (Layered)
    {
        cursor.subgoal_entities = totals.subgoal_entities;
        cursor.history_target_at = target_base + (totals.targets - totals.history_targets);
        fill_history_entities<I>(in, cursor.blocks, g, history_base, out);
    }
    else if (out.history_offsets)
        out.history_offsets[g] = static_cast<I>(history_base);
    if (has_virtual_nodes(v) && scratch)
    {
        virtual_ranks<Layered>(v, in, cell_counts, scratch);
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
template<class I, bool Layered = true>
MYMYR_HD void fill_graph(const InstanceView& v, const RowInput& given, u64 g, u64 tuple_base, u64 arg_base,
                         u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0,
                         u32* virtual_scratch = nullptr, const u32* cell_counts = nullptr, u64 history_base = 0)
{
    const RowInput& in = given;
    FillCursor cursor = begin_graph<I, Layered>(v, in, g, entity_base, totals, out, target_base, cell_counts,
                                                virtual_scratch, history_base);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list<Layered>(v, in);
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
        const Cell& c = cells[i];
        for (; r < c.relation; ++r)
        {
            offsets[r] = static_cast<I>(tuples);
            arg_offsets[r] = static_cast<I>(args);
            if (counts)
                counts[r] = static_cast<I>(0);
        }
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        const u32 n = is_action_op(c.op)
                          ? fill_action_cell<I>(v, c, in, entity_base, cursor, out, args_out + args)
                      : Layered && cell_counts && cell_counts[i] == 0
                          ? 0u
                          : fill_cell_at<I, Layered>(v, c, in, entity_base, auxiliary, tuples - tuple_base, cursor,
                                                     out, args_out + args);
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

/// Where the arguments of a relation-major cell go, read from the batch's scanned segment arrays: the segment's entry of
/// out.relation_arg_offsets.
template<class I>
struct ArrayArgPositions
{
    const GraphOut<I>& out;

    MYMYR_HD u64 args(u64 segment, u32 /*relation*/) const { return static_cast<u64>(out.relation_arg_offsets[segment]); }
};

/// Writes graph `g` of a relation-major batch: like fill_graph, except that a cell's position is not computed from a
/// local prefix but given by `positions.args(segment, relation)`, the first argument of the cell's segment (see
/// ArrayArgPositions). The `entity_ref` of a goal entity is the same as in fill_graph. The counts, if stored, are [G, R]
/// indexed (g, r). `cell_counts` are the graph's cell counts from count_graph (null: they are recounted when virtual
/// nodes need them).
template<class I, class Positions, bool Layered = true>
MYMYR_HD void fill_graph_positioned(const InstanceView& v, const RowInput& in, u64 g, u64 entity_base,
                                    const GraphTotals& totals, const GraphOut<I>& out, const Positions& positions,
                                    u64 target_base, u32* virtual_scratch, const u32* cell_counts,
                                    u64 history_base = 0)
{
    FillCursor cursor = begin_graph<I, Layered>(v, in, g, entity_base, totals, out, target_base, cell_counts,
                                                virtual_scratch, history_base);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list<Layered>(v, in);
    const u32 num_cells = list.count;
    const Cell* cells = list.cells;
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    u64 graph_tuple = 0;  // the tuples of the graph's cells before this one: the cells are filled in cell order
    u32 r = 0;
    for (u32 i = 0; i < num_cells; ++i)
    {
        const Cell& c = cells[i];
        if (counts)
            for (; r < c.relation; ++r)
                counts[r] = static_cast<I>(0);
        const u64 segment = cell_segment(true, g, c.relation, out.graphs, R);
        I* dst = out.relation_args + positions.args(segment, c.relation);
        const u32 n = is_action_op(c.op)
                          ? fill_action_cell<I>(v, c, in, entity_base, cursor, out, dst)
                      : Layered && cell_counts && cell_counts[i] == 0
                          ? 0u
                          : fill_cell_at<I, Layered>(v, c, in, entity_base, auxiliary, graph_tuple, cursor, out, dst);
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

/// Writes graph `g` of a relation-major batch with the cells' positions read from the scanned segment arrays
/// (out.relation_offsets and out.relation_arg_offsets at the cell's segment, scanned beforehand).
template<class I, bool Layered = true>
MYMYR_HD void fill_graph_relation_major(const InstanceView& v, const RowInput& given, u64 g, u64 entity_base,
                                        const GraphTotals& totals, const GraphOut<I>& out, u64 target_base = 0,
                                        u32* virtual_scratch = nullptr, const u32* cell_counts = nullptr,
                                        u64 history_base = 0)
{
    fill_graph_positioned<I, ArrayArgPositions<I>, Layered>(v, given, g, entity_base, totals, out,
                                                            ArrayArgPositions<I>{out}, target_base, virtual_scratch,
                                                            cell_counts, history_base);
}

// ------------------------------------------------------------------------------------------------ lanes of a group

/// The cursor of a fill past a cell of `n` tuples: the goal entities and action entities the cell wrote. A fill that
/// writes a cell on one lane only advances every lane's cursor with this.
MYMYR_HD void advance_cursor(FillCursor& cursor, const Cell& c, u32 n)
{
    if (is_action_op(c.op))
        cursor.actions_written += n;
    else if (is_goal_op(c.op))
    {
        if (goal_op_mode(c.op) == GoalMode::PlainTarget)
            (goal_op_level(c.op) != 0 ? cursor.subgoals_written : cursor.goals_written) += n;
    }
    else if (is_history_op(c.op) && history_op_target(c.op))
        cursor.history_written += n;
}

/// begin_graph for the lanes of a group: the lanes share the header, the objects and the virtual entities, and the
/// ranks of the virtual predicates (virtual_ranks into `scratch`, which the lanes share) are computed by lane 0. The
/// row has no subgoal layers or history steps (expect_no_layers).
template<class I, class Group>
MYMYR_HD FillCursor begin_graph_group(const Group& grp, const InstanceView& v, const RowInput& in, u64 g,
                                      u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out,
                                      u64 target_base, const u32* cell_counts, u32* scratch)
{
    const u32 lane = grp.lane(), lanes = grp.size();
    if (lane == 0)
        fill_graph_header<I>(v, in, g, target_base, out);
    for (u32 o = lane; o < v.num_objects; o += lanes)
        fill_objects<I>(o, o + 1, entity_base, out);
    if (lane == 0 && (totals.flags & k_graph_auxiliary))
        fill_auxiliary_entity<I>(v.num_objects, entity_base, out);
    expect_no_layers(in);
    FillCursor cursor{entity_blocks<false>(v, totals, entity_base), 0, 0, target_base, totals.goal_entities, nullptr};
    if (has_virtual_nodes(v) && scratch)
    {
        grp.sync();
        if (lane == 0)
            virtual_ranks<false>(v, in, cell_counts, scratch);
        grp.sync();
        cursor.rank = scratch;
        for (u32 p = lane; p < v.num_virtual; p += lanes)
            fill_virtual_entity<I>(v, cursor.blocks, scratch, p, out);
    }
    return cursor;
}

/// The tuples of a fluent, derived or static cell of K objects (K == k_any_arity: c.logical_arity) written by the lanes of
/// a group: the same tuples at the same places as fill_cell_of writes them, from `dst` on. The lanes share the words of
/// the cell's slot range (or its static rows), and a scan over the lanes' tuple counts places each lane's tuples.
template<class I, u32 K, bool Pred, class Group>
MYMYR_HD void fill_state_cell_group_of(const Group& grp, const InstanceView& v, const Cell& c, const RowInput& in,
                                       u64 entity_base, u64 auxiliary, u64 pred, I* dst)
{
    const u64 A = v.max_arity;
    const u32 lane = grp.lane(), lanes = grp.size();
    const StateTupleWriter<I, K, Pred> write{entity_base, auxiliary, pred, K == k_any_arity ? c.logical_arity : K};
    if (c.op == static_cast<u32>(CellOp::Static))
    {
        const i32* rows = v.static_args + static_cast<u64>(c.a) * A;
        for (u32 s = lane; s < c.b - c.a; s += lanes)
            write(dst + static_cast<u64>(s) * c.arity, rows + static_cast<u64>(s) * A);
        return;
    }
    const bool fluent = c.op == static_cast<u32>(CellOp::Fluent);
    const u64* words = fluent ? in.atoms : in.derived;
    const i32* table = fluent ? v.slot_args : v.derived_args;
    if (!words || c.a >= c.b)
        return;
    const u32 last = (c.b - 1u) >> 6;
    u64 written = 0;  // the cell's tuples before this round of words
    for (u32 first = c.a >> 6; first <= last; first += lanes)
    {
        const u32 w = first + lane;
        const u64 bits = w <= last ? range_word(words, w, c.a, c.b) : u64{0};
        u32 total = 0;
        const u32 before = grp.exclusive_scan(popcount64(bits), total);
        emit_word_tuples<I, K, Pred>(write, table, A, w, bits, dst + (written + before) * c.arity);
        written += total;
    }
}

template<class I, bool Pred, class Group>
MYMYR_HD void fill_state_cell_group_dispatch(const Group& grp, const InstanceView& v, const Cell& c, const RowInput& in,
                                             u64 entity_base, u64 auxiliary, u64 pred, I* dst)
{
    switch (c.logical_arity)
    {
    case 0: fill_state_cell_group_of<I, 0, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst); break;
    case 1: fill_state_cell_group_of<I, 1, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst); break;
    case 2: fill_state_cell_group_of<I, 2, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst); break;
    case 3: fill_state_cell_group_of<I, 3, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst); break;
    case 4: fill_state_cell_group_of<I, 4, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst); break;
    default: fill_state_cell_group_of<I, k_any_arity, Pred>(grp, v, c, in, entity_base, auxiliary, pred, dst);
    }
}

/// fill_cell_at for the lanes of a group, for a cell of `n` tuples (its count): a fluent, derived or static cell is shared
/// among the lanes; any other cell (goal and action cells, whose tuples depend on one another) is written by lane 0
/// alone. Every lane's cursor advances past the cell.
template<class I, class Group>
MYMYR_HD void fill_cell_group(const Group& grp, const InstanceView& v, const Cell& c, const RowInput& in,
                              u64 entity_base, u64 auxiliary, u64 graph_tuple, FillCursor& cursor,
                              const GraphOut<I>& out, I* dst, u32 n)
{
    if (c.op == static_cast<u32>(CellOp::Fluent) || c.op == static_cast<u32>(CellOp::Derived) ||
        c.op == static_cast<u32>(CellOp::Static))
    {
        const u64 pred = predicate_entity(v, cursor, c.relation);
        if (pred == k_no_entity)
            fill_state_cell_group_dispatch<I, false>(grp, v, c, in, entity_base, auxiliary, pred, dst);
        else
            fill_state_cell_group_dispatch<I, true>(grp, v, c, in, entity_base, auxiliary, pred, dst);
        return;
    }
    if (grp.lane() == 0)
    {
        FillCursor mine = cursor;
        fill_cell_at<I>(v, c, in, entity_base, auxiliary, graph_tuple, mine, out, dst);
    }
    advance_cursor(cursor, c, n);
}

/// Writes the offsets (and counts, when `counts` is not null) of the relations [from, to), which have no cell and so no
/// tuple, at the position `tuples` / `args`; the lanes share the relations.
template<class I, class Group>
MYMYR_HD void fill_empty_relations(const Group& grp, I* offsets, I* arg_offsets, I* counts, u32 from, u32 to, u64 tuples,
                                   u64 args)
{
    for (u32 r = from + grp.lane(); r < to; r += grp.size())
    {
        offsets[r] = static_cast<I>(tuples);
        arg_offsets[r] = static_cast<I>(args);
        if (counts)
            counts[r] = static_cast<I>(0);
    }
}

/// fill_graph for the lanes of a group: the same arrays with the same values. `cell_counts` are the graph's counts, as
/// count_graph or count_graph_group wrote them, and are required: the cells are written one after the other, each by
/// the whole group, from the positions the counts give. `virtual_scratch` is shared by the lanes (v.num_virtual
/// words, or null without predicate virtual nodes). A graph with expansion rows is filled by fill_graph.
template<class I, class Group>
MYMYR_HD void fill_graph_group(const Group& grp, const InstanceView& v, const RowInput& in, u64 g, u64 tuple_base,
                               u64 arg_base, u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out,
                               u64 target_base, u32* virtual_scratch, const u32* cell_counts)
{
    FillCursor cursor =
        begin_graph_group<I>(grp, v, in, g, entity_base, totals, out, target_base, cell_counts, virtual_scratch);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list<false>(v, in);
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    I* offsets = out.relation_offsets + g * R;
    I* arg_offsets = out.relation_arg_offsets + g * R;
    u64 tuples = tuple_base;
    u64 args = arg_base;
    u32 r = 0;
    for (u32 i = 0; i < list.count; ++i)
    {
        const Cell& c = list.cells[i];
        const u32 n = cell_counts[i];
        fill_empty_relations<I>(grp, offsets, arg_offsets, counts, r, c.relation, tuples, args);
        if (grp.lane() == 0)
        {
            offsets[c.relation] = static_cast<I>(tuples);
            arg_offsets[c.relation] = static_cast<I>(args);
            if (counts)
                counts[c.relation] = static_cast<I>(n);
        }
        fill_cell_group<I>(grp, v, c, in, entity_base, auxiliary, tuples - tuple_base, cursor, out,
                           out.relation_args + args, n);
        tuples += n;
        args += static_cast<u64>(n) * c.arity;
        r = c.relation + 1;
    }
    fill_empty_relations<I>(grp, offsets, arg_offsets, counts, r, R, tuples, args);
}

/// fill_graph_positioned for the lanes of a group, with the counts of the graph's cells from `positions.count(segment)`
/// (a cell with its tuples is written by the whole group, one cell after the other).
template<class I, class Positions, class Group>
MYMYR_HD void fill_graph_positioned_group(const Group& grp, const InstanceView& v, const RowInput& in, u64 g,
                                          u64 entity_base, const GraphTotals& totals, const GraphOut<I>& out,
                                          const Positions& positions, u64 target_base, u32* virtual_scratch,
                                          const u32* cell_counts)
{
    FillCursor cursor =
        begin_graph_group<I>(grp, v, in, g, entity_base, totals, out, target_base, cell_counts, virtual_scratch);
    const u64 auxiliary = entity_base + v.num_objects;
    const u32 R = v.num_relations;
    const CellList list = cell_list<false>(v, in);
    I* counts = out.relation_counts ? out.relation_counts + g * R : nullptr;
    u64 graph_tuple = 0;
    u32 r = 0;
    for (u32 i = 0; i < list.count; ++i)
    {
        const Cell& c = list.cells[i];
        const u64 segment = cell_segment(true, g, c.relation, out.graphs, R);
        const u32 n = positions.count(segment);
        if (counts)
        {
            for (u32 q = r + grp.lane(); q < c.relation; q += grp.size())
                counts[q] = static_cast<I>(0);
            if (grp.lane() == 0)
                counts[c.relation] = static_cast<I>(n);
        }
        fill_cell_group<I>(grp, v, c, in, entity_base, auxiliary, graph_tuple, cursor, out,
                           out.relation_args + positions.args(segment, c.relation), n);
        graph_tuple += n;
        r = c.relation + 1;
    }
    if (counts)
        for (u32 q = r + grp.lane(); q < R; q += grp.size())
            counts[q] = static_cast<I>(0);
}

// ------------------------------------------------------------------------------------------------------ Simple
//
// The homogeneous atom-object incidence graph of a graph's tuples. Its nodes are the graph's entities, in entity order,
// then one node per tuple (an atom), in the order of the graph's tuples in the graph-major layout (the cells' tuples in
// relation order, whatever the layout of the relation arrays). Node j of graph g is the batch-global node
// node_offsets[g] + j, where node_offsets[g] is the graph's entity base plus the tuples of the graphs before it, so the
// node of the batch-global entity e of graph g is e + (the tuples before g).
//
// The label of an entity node is its role (0 to 8), or, for an object that is one of the first `constants` objects (the
// domain's constants), 9 + R + its id; the label of an atom node is 9 + the id of its relation. For every tuple t (atom
// node a) and every position k of its relation, the entity e at that position gives two edges: e -> a with label k and
// a -> e with label K + k, K the largest encoded arity of a relation. A graph's edges are all its forward edges in
// (tuple, position) order, then all its reverse edges in the same order: the edges of graph g start at 2 * (the
// arguments before g), and the argument i of the graph (counted in graph-major tuple order) has the forward edge
// 2 * (the arguments before g) + i and the reverse edge 2 * (the arguments before g) + (the graph's arguments) + i.
//
// The emitters read the arguments that the relation fill has written, so they follow it graph by graph with the cells'
// counts of the graph.

/// The roles that label an entity node (EntityRole): atom labels follow them.
inline constexpr u32 k_simple_roles = 9;

/// The vocabulary of the Simple graph.
struct SimpleParams
{
    u32 relations;  // R: the relations of the schema
    u32 max_arity;  // K: the largest encoded arity of a relation
    u32 constants;  // the objects that get a label of their own (constant_labels), else 0
};

/// Where the Simple graph is written: pointers to the batch arrays.
template<class I>
struct SimpleOut
{
    I* node_offsets;  // [G + 1]
    I* edge_offsets;  // [G + 1]
    i32* node_label;  // [V]
    I* edge_source;   // [E]: row 0 of edge_index
    I* edge_target;   // [E]: row 1 of edge_index
    i32* edge_label;  // [E]
};

/// The place and the size of one graph in the batch.
struct SimpleGraph
{
    u64 entity_base;  // the graph's first entity (batch-global)
    u64 entities;
    u64 tuple_base;   // the tuples of the graphs before it
    u64 tuples;
    u64 arg_base;     // the tuple arguments of the graphs before it
    u64 args;
};

/// Writes the node labels of the graph's entities. `entity_role` is the batch's entity_role.
template<class I>
MYMYR_HD void simple_fill_entities(const SimpleParams& params, const SimpleOut<I>& out, const u8* entity_role,
                                   const SimpleGraph& graph)
{
    i32* label = out.node_label + graph.entity_base + graph.tuple_base;
    const u8* role = entity_role + graph.entity_base;
    const i32 constants_at = static_cast<i32>(k_simple_roles + params.relations);
    for (u64 j = 0; j < graph.entities; ++j)
        label[j] = role[j] == k_role_object && j < params.constants ? constants_at + static_cast<i32>(j)
                                                                    : static_cast<i32>(role[j]);
}

/// Writes the atom nodes and the edges of the `n` tuples of one cell: `args` are the cell's arguments (batch-global
/// entity ids, `arity` per tuple), `graph_tuple` is the index of the cell's first tuple among the graph's tuples and
/// `graph_arg` the index of its first argument among the graph's arguments (both in graph-major order).
template<class I>
MYMYR_HD void simple_emit_cell(const SimpleParams& params, const SimpleOut<I>& out, const SimpleGraph& graph,
                               u32 relation, u32 arity, u32 n, const I* args, u64 graph_tuple, u64 graph_arg)
{
    const i32 atom_label = static_cast<i32>(k_simple_roles + relation);
    const i32 back = static_cast<i32>(params.max_arity);
    const u64 shift = graph.tuple_base;
    I* const forward_source = out.edge_source + 2 * graph.arg_base + graph_arg;
    I* const forward_target = out.edge_target + 2 * graph.arg_base + graph_arg;
    i32* const forward_label = out.edge_label + 2 * graph.arg_base + graph_arg;
    I* const back_source = forward_source + graph.args;
    I* const back_target = forward_target + graph.args;
    i32* const back_label = forward_label + graph.args;
    const u64 first_atom = graph.entity_base + graph.tuple_base + graph.entities + graph_tuple;
    for (u32 k = 0; k < n; ++k)
    {
        const I atom = static_cast<I>(first_atom + k);
        out.node_label[first_atom + k] = atom_label;
        const u64 at = static_cast<u64>(k) * arity;
        for (u32 slot = 0; slot < arity; ++slot)
        {
            const I entity = static_cast<I>(static_cast<u64>(args[at + slot]) + shift);
            forward_source[at + slot] = entity;
            forward_target[at + slot] = atom;
            forward_label[at + slot] = static_cast<i32>(slot);
            back_source[at + slot] = atom;
            back_target[at + slot] = entity;
            back_label[at + slot] = back + static_cast<i32>(slot);
        }
    }
}

/// Writes graph `g` of the Simple graph after its relation tuples are written: its entry of node_offsets and
/// edge_offsets, the labels of its entities and its atoms, and its edges. `cells` are the graph's cells and
/// `cell_counts` their tuple counts (the counts of count_graph, one per cell); `out` is the destination of the relation
/// arrays (entity_role, relation_args and, relation-major, relation_arg_offsets). The end entries of the two offset
/// arrays are the caller's.
template<class I>
MYMYR_HD void simple_fill_graph(const SimpleParams& params, const SimpleOut<I>& simple, const GraphOut<I>& out,
                                const CellList& cells, const u32* cell_counts, bool relation_major, u64 g,
                                const SimpleGraph& graph)
{
    simple.node_offsets[g] = static_cast<I>(graph.entity_base + graph.tuple_base);
    simple.edge_offsets[g] = static_cast<I>(2 * graph.arg_base);
    simple_fill_entities<I>(params, simple, out.entity_role, graph);
    u64 graph_tuple = 0, graph_arg = 0;
    for (u32 i = 0; i < cells.count; ++i)
    {
        const u32 n = cell_counts[i];
        if (n == 0)
            continue;
        const Cell& c = cells.cells[i];
        const u64 at = relation_major
                           ? static_cast<u64>(out.relation_arg_offsets[cell_segment(true, g, c.relation, out.graphs,
                                                                                    params.relations)])
                           : graph.arg_base + graph_arg;
        simple_emit_cell<I>(params, simple, graph, c.relation, c.arity, n, out.relation_args + at, graph_tuple,
                            graph_arg);
        graph_tuple += n;
        graph_arg += static_cast<u64>(n) * c.arity;
    }
}
}  // namespace mymyr::encoding::detail
