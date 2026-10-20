#pragma once

// The relational encoder: state rows of a task table to one array bundle of graphs, one graph per row. A graph has the
// instance's objects as entities (plus one auxiliary entity iff some nullary tuple exists, one goal entity per goal
// literal with goal targets, one subgoal entity per subgoal literal with subgoal targets, one history entity per
// history step and one history-target entity per history literal with history targets) and, for every relation of the
// schema, the tuples of the row's atoms: fluent atoms from the state, derived atoms, the instance's static atoms, the
// goal literals of the instance (or of the row) in the plain variant and as the literals that hold or do not hold in
// the row, the literals of the row's subgoal layers likewise, and the literals of the row's history steps. Every
// per-graph decision lives in detail/lanes.hpp; this file only loops, scans, allocates and checks.
//
// The bundle holds (G graphs, R = schema().num_relations(), I the index type of the options: int32 or int64; every
// entity id is batch-global, the graph's first entity is entity_offsets[g]):
//   task_ids             i32 [G]       the instance of each row
//   num_objects          I   [G]       the instance's objects (the auxiliary entity is not counted)
//   entity_offsets       I   [G + 1]   graph g's entities are [entity_offsets[g], entity_offsets[g + 1])
//   entity_role          u8  [E]       the EntityRole: 0 object, 1 predicate virtual node, 2 goal, 3 subgoal, 4 action,
//                                      5 history step, 6 history target, 8 auxiliary (omitted: see below)
//   entity_ref           I   [E]       object: its instance-local id; goal, subgoal and history target: the index of
//                                      its tuple among the graph's tuples in graph-major order (the tuples of the
//                                      graph's cells in relation order); action: the row of its ground action in the
//                                      graph's expansion rows (counted from the graph's first row); history step: its
//                                      index among the row's valid history steps; predicate virtual node: the id of
//                                      the predicate's key relation; auxiliary: -1 (omitted: see below)
//   relation_counts      I   [G, R]    the tuples of cell (g, r), indexed (g, r) in both layouts (omitted: see below)
//   relation_offsets     I   [G * R + 1]      cell (g, r) is segment s: its tuples are [offsets[s], offsets[s + 1])
//   relation_arg_offsets I   [G * R + 1]      the same for the argument array
//   relation_args        I   [A]       the entities of the tuples, tuple after tuple, Relation::arity per tuple
//   goal_complete        u8  [G]       1 iff the graph's goal literals are its whole goal (present iff the schema has goal
//                                      relations)
//   target_offsets       I   [G + 1]   graph g's targets are [target_offsets[g], target_offsets[g + 1]) (present iff the
//                                      options name target sources)
//   target_entity        I   [T]       the entity of each target, in entity order
//   target_group         u8  [T]       the group of each target (TargetGroup: 0 goal, 1 subgoal, 2 action, 3 history)
//   history_offsets      I   [G + 1]   graph g's history steps are [history_offsets[g], history_offsets[g + 1])
//                                      (present iff the options have history)
//   history_entity       I   [S]       the entity of each history step, graph after graph in step order
//   history_dt           i32 [S]       the dt of each history step
// and the scalars version, num_graphs, num_relations, layout (0: graph-major, 1: relation-major), index_dtype (0:
// int32, 1: int64), schema_fingerprint (the schema's fingerprint as an i64 bit pattern) and statics (0: the static
// tuples are repeated in every graph).
//
// Layouts. The segment of cell (g, r) is s = g * R + r in the graph-major layout and s = r * G + g in the
// relation-major one: the relation-major layout holds the tuples of a relation contiguously across the graphs. The two
// layouts are a permutation of each other: task_ids, num_objects, entity_offsets, entity_role, entity_ref,
// relation_counts, the target arrays, goal_complete and every cell's tuples (their order and their entity ids) are
// equal; relation_offsets and relation_arg_offsets are the exclusive prefix sums of the cells' tuple and argument
// counts in segment order, and relation_args is the concatenation of the cells' arguments in segment order. So the cell
// at relation-major segment r * G + g is the cell at graph-major segment g * R + r.
//
// Entity references. Every entity_ref is graph-local or intrinsic to the catalog, so it does not depend on the layout
// or on the graph's place in the batch. The tuple of a goal (subgoal, history target) entity with reference k in graph
// g is, in the graph-major layout, tuple relation_offsets[g * R] + k; in the relation-major layout subtract the counts
// of the graph's cells (relation_counts[g, 0], relation_counts[g, 1], ...) from k until the rest j falls into a cell
// (g, r), and the tuple is relation_offsets[r * G + g] + j. The expansion row of an action entity with reference k is
// row offsets[g] + k of the expansion given as `actions`. The history step with reference k of graph g is the k-th valid
// step of the row's history input (counted before FlatOptions::history_max_steps drops the oldest steps): its dt is
// history.dt[g * history.masks.layers + k] and its masks are at index k of the row's history masks.
//
// Omitted arrays. With FlatOptions::emit_counts = false a batch has no relation_counts: the tuples of segment s are
// relation_offsets[s + 1] - relation_offsets[s]. When EncodingSchema::implicit_entity_arrays() holds, a batch has no
// entity_role and entity_ref; the implicit values are documented there. Everything else is unchanged.
//
// The tuples of a cell are the atoms of its predicate in the order of the instance's atom slots, which is the
// lexicographic order of their objects; a nullary atom is the unary tuple [auxiliary entity]. The tuples of a goal cell
// are the goal literals of its predicate and sign (plain), or those that hold or do not hold in the row, in the same
// order; with goal targets a plain goal tuple starts with its goal entity. The goal entities follow the objects and the
// auxiliary entity in the order the plain goal cells are filled: relation order, then tuple order, so the entity of the
// k-th plain goal tuple of a graph has local index num_objects + has_auxiliary + k. A graph contains nothing that its
// row and instance do not have. Encoding a slice of a batch gives the slice of the batch's encoding, with the
// offsets rebased (in the graph-major layout; the cells of a relation-major slice are the same cells, in the order of
// their own segments); no entity_ref needs a rebase.
//
// Threads. The calls that take a ThreadPool split the graphs into one contiguous slice per member (ThreadPool::slice)
// and run two parallel passes, count and fill, between which the members' sizes are scanned serially. Every position is
// a sum over the graphs before it in batch order, so the output does not depend on the pool size.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/encoding/buffers.hpp"
#include "mymyr/encoding/context.hpp"
#include "mymyr/encoding/options.hpp"
#include "mymyr/rl/expand.hpp"

#include <memory>

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

/// The graphs of a horizon batch: G graphs, each a root state and the nodes below it, as rows of one state batch, and
/// the edges between the nodes. The rows of graph g are [node_offsets[g], node_offsets[g + 1]) of `nodes`; the first of
/// them is the graph's root, so every graph has at least one node, and node k of graph g is row node_offsets[g] + k.
/// The edges of graph g are the entries [edge_offsets[g], edge_offsets[g + 1]) of the edge arrays: edge e leads from node
/// edge_src[e] to node edge_dst[e] (graph-local node indices) by the ground action of schema edge_schema[e] with the
/// objects edge_binding[e * label_width ...] (the first arity-of-the-schema entries; the others are not read). The
/// edges are given as they are: a node may have several incoming edges, or none. rl::expand's rows have this form
/// (schema, binding, label_width).
/// `node_offsets` and `edge_offsets` start at 0. `edge_schema` and `edge_binding` are needed only for a schema with
/// action relations (`label_width` at least the table's), `edge_src` and `edge_dst` only for one with action or parent
/// relations; the other edge arrays may be null then. A schema without either does not read the edges at all.
struct HorizonInput
{
    rl::StateBatchView nodes;           // M node rows (numeric_words must be 0); the stride is honoured
    u64 graphs = 0;                     // G
    const i32* node_offsets = nullptr;  // [G + 1] strictly ascending from 0, the last at most nodes.rows
    const i32* edge_offsets = nullptr;  // [G + 1] ascending from 0
    const i32* edge_src = nullptr;      // [E]
    const i32* edge_dst = nullptr;      // [E]
    const i32* edge_schema = nullptr;   // [E]
    const i32* edge_binding = nullptr;  // [E, label_width]
    u32 label_width = 0;
};

/// The inputs of one batch. With a horizon schema (EncodingSchema::has_horizon) the graphs are those of `horizon` and
/// `states` is empty; `task_ids` and the goals and subgoal layers are then per graph and `derived` per node row (M
/// rows). Without one, `horizon` is empty.
struct EncodeInput
{
    rl::StateBatchView states;      // N rows (numeric_words must be 0); the stride is honoured
    const i32* task_ids = nullptr;  // [N] the instance of each row; null: instance 0
    const u64* derived = nullptr;   // [N, derived_stride] the derived words of each row; null: evaluated here
    u64 derived_stride = 0;         // words from one row of `derived` to the next (at least max_derived_words())
    // The goal of each row, replacing its instance's goal (and its constant part): masks over the fluent slots, as the
    // goal masks of rl (rl::goal_masks) and, optionally, over the derived slots, `goal_pos` and `goal_neg` both or
    // neither, likewise the derived pair. A row of a mask array is `*_words` wide, at least as wide as the words (derived
    // words) of every instance in the batch; `*_stride` is the distance between consecutive rows, 0 for one row that
    // every graph shares. Bits of slots an instance does not have are an error. The goal of such a row is complete.
    const u64* goal_pos = nullptr;
    const u64* goal_neg = nullptr;
    u64 goal_stride = 0;
    u32 goal_words = 0;
    const u64* goal_derived_pos = nullptr;
    const u64* goal_derived_neg = nullptr;
    u64 goal_derived_stride = 0;
    u32 goal_derived_words = 0;
    // The successors of each row, as rl::expand returns them (null: no action tuples). The graph of row g has one action
    // entity per distinct ground action among the expansion rows [offsets[g], offsets[g + 1]), which must be the rows of
    // the same state g: canonical order (by schema, then lexicographically by binding), the labels of the row's instance,
    // and `parent` (when written) naming g. The expansion must not have overflowed, and needs `offsets`, `schema` and
    // `binding`; `label_width` at least the table's. The schema must have action relations. The graph only reads the
    // labels: the successor states, goal flags and derived words of the expansion are not used.
    const rl::Expansion* actions = nullptr;
    // The subgoal layers of each row: layer l fills the subgoal relations of level l + 1 (at most max_goal_level
    // layers). Each level holds the literals of its layer that no lower level has, the goal's included: a literal is
    // encoded once, at its lowest level. Needs subgoal relations in the schema.
    LayerInput subgoals{};
    // The history steps of each row. Needs the history lane in the options.
    HistoryInput history{};
    // The graphs of a horizon schema.
    HorizonInput horizon{};
};

/// The graphs a pool member should have at least: waking a member costs about as much as encoding several hundred
/// graphs, so a batch is split over members only as far as each gets this many, and a batch of at most this many
/// graphs is encoded on the calling thread whatever the size of the pool.
inline constexpr u64 k_graphs_per_member = 1024;

/// Working memory of encoder calls: per-member graph sizes, derived words and tallies. It is reused across calls (grown
/// as needed). One scratch serves one call at a time, and fill() needs the scratch of the count() before it.
class EncodeScratch
{
public:
    EncodeScratch();
    ~EncodeScratch();
    EncodeScratch(const EncodeScratch&) = delete;
    EncodeScratch& operator=(const EncodeScratch&) = delete;

    /// The scratch of the calling thread, kept for the life of the thread: the one for calls that do not own a scratch
    /// (it is warm once the thread has encoded). It serves one call at a time, like any scratch.
    [[nodiscard]] static EncodeScratch& of_this_thread();

    struct Impl;
    [[nodiscard]] Impl& impl() noexcept { return *m_impl; }

private:
    std::unique_ptr<Impl> m_impl;
};

class FlatEncoder
{
public:
    /// Throws std::invalid_argument ("mymyr: encoding: ...") for invalid options.
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(rl::TaskTablePtr table, const FlatOptions& options);
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(TaskPtr task, const FlatOptions& options);

    [[nodiscard]] const rl::TaskTable& table() const noexcept { return m_context->table(); }
    [[nodiscard]] const rl::TaskTablePtr& table_ptr() const noexcept { return m_context->table_ptr(); }
    [[nodiscard]] const EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const std::shared_ptr<const EncodingSchema>& schema_ptr() const noexcept { return m_schema; }
    [[nodiscard]] const EncodingContext& context() const noexcept { return *m_context; }
    [[nodiscard]] const std::shared_ptr<const EncodingContext>& context_ptr() const noexcept { return m_context; }

    /// Encodes the rows on the calling thread into a new bundle. Const and reentrant. Throws std::invalid_argument for
    /// invalid input (a task id out of range, a row narrower than its instance, a state or goal bit its instance does
    /// not have, a goal mask narrower than its instance, goal masks without their pair, subgoal layers or history steps
    /// that the schema has no relations for, that are narrower than their instance or set a bit it does not have, a
    /// history count beyond its row's steps, numeric words, expansion rows that are not the canonical rows of their
    /// state) and
    /// std::overflow_error when a count exceeds the index type.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in) const;
    /// The same on the members of `pool` (graphs are split over at most ceil(graphs / k_graphs_per_member) of them, on
    /// the calling thread alone for a small batch). The bundle is bitwise equal for every pool size. `scratch` is not
    /// shared with a concurrent call.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in, ThreadPool& pool,
                                                                EncodeScratch& scratch) const;

    /// The capacity form into a new bundle whose variable-size arrays (the entity arrays, relation_args, the target
    /// entity and group arrays and the history entity and dt arrays) have the capacities' sizes. The arrays hold the
    /// batch in their first `sizes.entities` / `sizes.args` / ... entries and are zero beyond; if `sizes.overflow` is
    /// set, every array is zero (no valid encoding). `sizes` is set to the true sizes.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode_capacity(const EncodeInput& in,
                                                                         const BatchShape& capacities,
                                                                         ThreadPool& pool, EncodeScratch& scratch,
                                                                         FlatSizes& sizes) const;

    /// Throws std::invalid_argument for the subgoal layers and history steps of `in` that the input alone shows
    /// invalid: given for a schema without subgoal or history relations, more layers than levels, a missing pair of
    /// masks or dt, a row stride below the layers. The rows' words, bits and history counts are checked by count().
    void check_layer_inputs(const EncodeInput& in) const;

    /// The capacity form, step one: checks the input (the exceptions of encode, except std::overflow_error), counts and
    /// scans the batch, and returns its true sizes; `overflow` has k_overflow_index set when a size or count does not
    /// fit the index type. `scratch` keeps the counts for fill().
    [[nodiscard]] FlatSizes count(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const;
    /// Step two: writes the batch counted by the last count() of `scratch` (`in` and `sizes` must be that call's) into
    /// `out`. Requires sizes.overflow == 0 and the buffers as described at FlatBuffers; throws std::invalid_argument
    /// otherwise, and when `scratch` does not hold the count of `in`.
    void fill(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
              EncodeScratch& scratch) const;
    /// Both steps against capacities: counts, and fills `out` iff the sizes fit the index type and `capacities`. The
    /// returned sizes are true; if they overflow, no array of `out` is written. Entries of `out` past the sizes are
    /// left as they were. Reuses the buffers of `scratch`.
    FlatSizes encode_into(const EncodeInput& in, const BatchShape& capacities, const FlatBuffers& out,
                          ThreadPool& pool, EncodeScratch& scratch) const;

private:
    FlatEncoder() = default;
    /// An uninitialized bundle with the pointers to its arrays (null for an array the schema omits).
    struct Allocation
    {
        std::shared_ptr<rl::ArrayBundle> bundle;
        FlatBuffers buffers;
    };
    /// An uninitialized bundle of `graphs` graphs with room for `entities` entities, `args` tuple arguments, `targets`
    /// targets and `history` history steps.
    [[nodiscard]] Allocation allocate(u64 graphs, u64 entities, u64 args, u64 targets, u64 history) const;
    /// count() and fill() for a horizon schema (horizon.cpp).
    [[nodiscard]] FlatSizes count_horizon(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const;
    void fill_horizon(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
                      EncodeScratch& scratch) const;
    /// The number of graphs of a batch: the rows of `in.states`, or the graphs of `in.horizon`.
    [[nodiscard]] u64 graphs_of(const EncodeInput& in) const noexcept;

    std::shared_ptr<const EncodingSchema> m_schema;
    std::shared_ptr<const EncodingContext> m_context;
};

namespace detail
{
/// Throws std::overflow_error unless `value` fits the index type; `what` names the counted thing.
void check_index_range(u64 value, IndexType index, const char* what);
}
}
