#pragma once

// The relational encoder: state rows of a task table to one array bundle of graphs, one graph per row. A graph has the
// instance's objects as entities (plus one auxiliary entity iff some nullary tuple exists, and one goal entity per goal
// literal with goal targets) and, for every relation of the schema, the tuples of the row's atoms: fluent atoms from the
// state, derived atoms, the instance's static atoms, and the goal literals of the instance (or of the row) in the plain
// variant and as the literals that hold or do not hold in the row. Every per-graph decision lives in
// detail/lanes.hpp; this file only loops, scans, allocates and checks.
//
// The bundle holds (G graphs, R = schema().num_relations(), I the index type of the options: int32 or int64; every
// entity id is batch-global, the graph's first entity is entity_offsets[g]):
//   task_ids             i32 [G]       the instance of each row
//   num_objects          I   [G]       the instance's objects (the auxiliary entity is not counted)
//   entity_offsets       I   [G + 1]   graph g's entities are [entity_offsets[g], entity_offsets[g + 1])
//   entity_role          u8  [E]       the EntityRole: 0 object, 2 goal, 8 auxiliary (omitted: see below)
//   entity_ref           I   [E]       object: its instance-local id; goal: the batch id of its tuple (the position of
//                                      the tuple in the concatenation of all cells in layout order); auxiliary: -1
//                                      (omitted: see below)
//   relation_counts      I   [G, R]    the tuples of cell (g, r), indexed (g, r) in both layouts (omitted: see below)
//   relation_offsets     I   [G * R + 1]      cell (g, r) is segment s: its tuples are [offsets[s], offsets[s + 1])
//   relation_arg_offsets I   [G * R + 1]      the same for the argument array
//   relation_args        I   [A]       the entities of the tuples, tuple after tuple, Relation::arity per tuple
//   goal_complete        u8  [G]       1 iff the graph's goal literals are its whole goal (present iff the schema has goal
//                                      relations)
//   target_offsets       I   [G + 1]   graph g's targets are [target_offsets[g], target_offsets[g + 1]) (present iff the
//                                      options name target sources)
//   target_entity        I   [T]       the entity of each target, in entity order
//   target_group         u8  [T]       the group of each target (TargetGroup: 0 goal)
// and the scalars version, num_graphs, num_relations, layout (0: graph-major, 1: relation-major), index_dtype (0:
// int32, 1: int64), schema_fingerprint (the schema's fingerprint as an i64 bit pattern) and statics (0: the static
// tuples are repeated in every graph).
//
// Layouts. The segment of cell (g, r) is s = g * R + r in the graph-major layout and s = r * G + g in the
// relation-major one: the relation-major layout holds the tuples of a relation contiguously across the graphs. The two
// layouts are a permutation of each other: task_ids, num_objects, entity_offsets, entity_role, relation_counts, the
// target arrays, goal_complete and every cell's tuples (their order and their entity ids) are equal, and so is
// entity_ref except the references of goal entities, which index the tuples in layout order; relation_offsets and relation_arg_offsets are
// the exclusive prefix sums of the cells' tuple and argument counts in segment order, and relation_args is the
// concatenation of the cells' arguments in segment order. So the cell at relation-major segment r * G + g is the cell
// at graph-major segment g * R + r.
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
// their own segments).
//
// Threads. The calls that take a ThreadPool split the graphs into one contiguous slice per member (ThreadPool::slice)
// and run two parallel passes, count and fill, between which the members' sizes are scanned serially. Every position is
// a sum over the graphs before it in batch order, so the output does not depend on the pool size.

#include "mymyr/core/thread_pool.hpp"
#include "mymyr/encoding/context.hpp"
#include "mymyr/encoding/options.hpp"
#include "mymyr/rl/expand.hpp"

#include <memory>

namespace mymyr::encoding
{
/// The inputs of one batch.
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
};

/// The capacities of the variable-size arrays of a batch, for the capacity form: how many entities (the entries of the
/// entity arrays), tuples, tuple arguments (the entries of relation_args) and targets (the entries of target_entity and
/// target_group) the caller's buffers hold. The arrays
/// whose size follows from the number of graphs and relations are always large enough by contract (see FlatBuffers).
struct FlatCapacities
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
};

/// Flags of FlatSizes::overflow.
inline constexpr u32 k_overflow_capacity = 1;  // some size exceeds its capacity
inline constexpr u32 k_overflow_index = 2;     // some size or count does not fit the index type

/// The true sizes of a batch (not clipped to any capacity) and whether the batch can be written. A POD, so that a
/// device call can return it in the same form.
struct FlatSizes
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u64 targets = 0;
    u32 overflow = 0;  // k_overflow_* flags; 0: the batch fits
    [[nodiscard]] bool overflowed() const noexcept { return overflow != 0; }
};

/// The destination arrays of a batch of G graphs and R relations, typed by the index type of the options (int32 or
/// int64). The caller owns them. Arrays of size G, G + 1, G * R or G * R + 1 must have that size; relation_args,
/// entity_role, entity_ref, target_entity and target_group must hold at least the sizes of the batch (FlatSizes).
/// entity_role and entity_ref must be null iff the schema has implicit entity arrays, relation_counts iff it does not
/// store counts, goal_complete iff it has no goal relations, the three target arrays iff it has no targets. Nothing is
/// written past the sizes: the entries of the variable-size arrays beyond them keep their old values.
struct FlatBuffers
{
    i32* task_ids = nullptr;
    void* num_objects = nullptr;
    void* entity_offsets = nullptr;
    u8* entity_role = nullptr;
    void* entity_ref = nullptr;
    void* relation_counts = nullptr;
    void* relation_offsets = nullptr;
    void* relation_arg_offsets = nullptr;
    void* relation_args = nullptr;
    u8* goal_complete = nullptr;
    void* target_offsets = nullptr;
    void* target_entity = nullptr;
    u8* target_group = nullptr;
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

    struct Impl;
    [[nodiscard]] Impl& impl() noexcept { return *m_impl; }

private:
    std::unique_ptr<Impl> m_impl;
};

class FlatEncoder
{
public:
    /// Throws std::invalid_argument ("mymyr: encoding: ...") for invalid options and for options this encoder does not
    /// encode: subgoal levels, subgoal, action and history targets, predicate virtual nodes and horizons.
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
    /// not have, a goal mask narrower than its instance, goal masks without their pair, numeric words) and
    /// std::overflow_error when a count exceeds the index type.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in) const;
    /// The same on the members of `pool` (graphs are split over at most ceil(graphs / k_graphs_per_member) of them, on
    /// the calling thread alone for a small batch). The bundle is bitwise equal for every pool size. `scratch` is not
    /// shared with a concurrent call.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in, ThreadPool& pool,
                                                                EncodeScratch& scratch) const;

    /// The capacity form into a new bundle whose variable-size arrays (the entity arrays, relation_args and the target
    /// entity and group arrays) have the capacities' sizes. The arrays hold the batch in their first `sizes.entities` / `sizes.args` entries and are zero
    /// beyond; if `sizes.overflow` is set, every array is zero (no valid encoding). `sizes` is set to the true sizes.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode_capacity(const EncodeInput& in,
                                                                         const FlatCapacities& capacities,
                                                                         ThreadPool& pool, EncodeScratch& scratch,
                                                                         FlatSizes& sizes) const;

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
    FlatSizes encode_into(const EncodeInput& in, const FlatCapacities& capacities, const FlatBuffers& out,
                          ThreadPool& pool, EncodeScratch& scratch) const;

private:
    FlatEncoder() = default;
    /// An uninitialized bundle of `graphs` graphs with room for `entities` entities, `args` tuple arguments and `targets`
    /// targets.
    [[nodiscard]] rl::ArrayBundle allocate(u64 graphs, u64 entities, u64 args, u64 targets) const;

    std::shared_ptr<const EncodingSchema> m_schema;
    std::shared_ptr<const EncodingContext> m_context;
};

namespace detail
{
/// Throws std::overflow_error unless `value` fits the index type; `what` names the counted thing.
void check_index_range(u64 value, IndexType index, const char* what);
}
}
