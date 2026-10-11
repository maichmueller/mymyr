#pragma once

// The relational encoder: state rows of a task table to one array bundle of graphs, one graph per row. A graph has the
// instance's objects as entities (plus one auxiliary entity iff some nullary tuple exists) and, for every relation of
// the schema, the tuples of the row's atoms: fluent atoms from the state, derived atoms, and the instance's static
// atoms. Every per-graph decision lives in detail/lanes.hpp; this file only loops, scans, allocates and checks.
//
// The bundle holds (G graphs, R = schema().num_relations(), I the index type of the options: int32 or int64; every
// entity id is batch-global, the graph's first entity is entity_offsets[g]):
//   task_ids             i32 [G]       the instance of each row
//   num_objects          I   [G]       the instance's objects (the auxiliary entity is not counted)
//   entity_offsets       I   [G + 1]   graph g's entities are [entity_offsets[g], entity_offsets[g + 1])
//   entity_role          u8  [E]       the EntityRole: 0 object, 8 auxiliary (omitted: see below)
//   entity_ref           I   [E]       object: its instance-local id; auxiliary: -1 (omitted: see below)
//   relation_counts      I   [G, R]    the tuples of cell (g, r), indexed (g, r) in both layouts (omitted: see below)
//   relation_offsets     I   [G * R + 1]      cell (g, r) is segment s: its tuples are [offsets[s], offsets[s + 1])
//   relation_arg_offsets I   [G * R + 1]      the same for the argument array
//   relation_args        I   [A]       the entities of the tuples, tuple after tuple, Relation::arity per tuple
// and the scalars version, num_graphs, num_relations, layout (0: graph-major, 1: relation-major), index_dtype (0: int32,
// 1: int64), schema_fingerprint (the schema's fingerprint as an i64 bit pattern) and statics (0: the static tuples are
// repeated in every graph).
//
// Layouts. The segment of cell (g, r) is s = g * R + r in the graph-major layout and s = r * G + g in the relation-major
// one: the relation-major layout holds the tuples of a relation contiguously across the graphs. The two layouts are a
// permutation of each other: task_ids, num_objects, entity_offsets, the entity arrays, relation_counts and every cell's
// tuples (their order and their entity ids) are equal; relation_offsets and relation_arg_offsets are the exclusive prefix
// sums of the cells' tuple and argument counts in segment order, and relation_args is the concatenation of the cells'
// arguments in segment order. So the cell at relation-major segment r * G + g is the cell at graph-major segment g * R + r.
//
// Omitted arrays. With FlatOptions::emit_counts = false a batch has no relation_counts: the tuples of segment s are
// relation_offsets[s + 1] - relation_offsets[s]. When EncodingSchema::implicit_entity_arrays() holds, a batch has no
// entity_role and entity_ref; the implicit values are documented there. Everything else is unchanged.
//
// The tuples of a cell are the atoms of its predicate in the order of the instance's atom slots, which is the
// lexicographic order of their objects; a nullary atom is the unary tuple [auxiliary entity]. A graph contains nothing
// that its row and instance do not have. Encoding a slice of a batch gives the slice of the batch's encoding, with the
// offsets rebased (in the graph-major layout; the cells of a relation-major slice are the same cells, in the order of
// their own segments).
//
// Threads. The calls that take a ThreadPool split the graphs into one contiguous slice per member (ThreadPool::slice)
// and run two parallel passes, count and fill, between which the members' sizes are scanned serially. Every position
// is a sum over the graphs before it in batch order, so the output does not depend on the pool size.

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
};

/// The capacities of the variable-size arrays of a batch, for the capacity form: how many entities (the entries of the
/// entity arrays), tuples and tuple arguments (the entries of relation_args) the caller's buffers hold. The arrays whose
/// size follows from the number of graphs and relations are always large enough by contract (see FlatBuffers).
struct FlatCapacities
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
};

/// Flags of FlatSizes::overflow.
inline constexpr u32 k_overflow_capacity = 1;  // some size exceeds its capacity
inline constexpr u32 k_overflow_index = 2;     // some size or count does not fit the index type

/// The true sizes of a batch (not clipped to any capacity) and whether the batch can be written. A POD, so that a device
/// call can return it in the same form.
struct FlatSizes
{
    u64 entities = 0;
    u64 tuples = 0;
    u64 args = 0;
    u32 overflow = 0;  // k_overflow_* flags; 0: the batch fits
    [[nodiscard]] bool overflowed() const noexcept { return overflow != 0; }
};

/// The destination arrays of a batch of G graphs and R relations, typed by the index type of the options (int32 or
/// int64). The caller owns them. Arrays of size G, G + 1, G * R or G * R + 1 must have that size; relation_args,
/// entity_role and entity_ref must hold at least the capacities. entity_role and entity_ref must be null iff the schema
/// has implicit entity arrays, relation_counts iff it does not store counts. Nothing is written past these sizes.
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
};

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
    /// encode: goal derivations, subgoal levels, targets, predicate virtual nodes, horizons and the relation layout.
    /// A default-constructed FlatOptions lists goal derivations and is therefore refused: pass `goal_derivations = {}`.
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(rl::TaskTablePtr table, const FlatOptions& options);
    [[nodiscard]] static std::shared_ptr<const FlatEncoder> create(TaskPtr task, const FlatOptions& options);

    [[nodiscard]] const rl::TaskTable& table() const noexcept { return m_context->table(); }
    [[nodiscard]] const rl::TaskTablePtr& table_ptr() const noexcept { return m_context->table_ptr(); }
    [[nodiscard]] const EncodingSchema& schema() const noexcept { return *m_schema; }
    [[nodiscard]] const std::shared_ptr<const EncodingSchema>& schema_ptr() const noexcept { return m_schema; }
    [[nodiscard]] const EncodingContext& context() const noexcept { return *m_context; }

    /// Encodes the rows on the calling thread into a new bundle. Const and reentrant. Throws std::invalid_argument for
    /// invalid input (a task id out of range, a row narrower than its instance, a state bit its instance does not have,
    /// numeric words) and std::overflow_error when a count exceeds the index type.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in) const;
    /// The same on every member of `pool` (graphs are split over the members). The bundle is bitwise equal for every
    /// pool size. `scratch` is not shared with a concurrent call.
    [[nodiscard]] std::shared_ptr<const rl::ArrayBundle> encode(const EncodeInput& in, ThreadPool& pool,
                                                                EncodeScratch& scratch) const;

    /// The capacity form, step one: checks the input (the exceptions of encode, except std::overflow_error), counts and
    /// scans the batch, and returns its true sizes; `overflow` has k_overflow_index set when a size or count does not fit
    /// the index type. `scratch` keeps the counts for fill().
    [[nodiscard]] FlatSizes count(const EncodeInput& in, ThreadPool& pool, EncodeScratch& scratch) const;
    /// Step two: writes the batch counted by the last count() of `scratch` (`in` and `sizes` must be that call's) into
    /// `out`. Requires sizes.overflow == 0 and the buffers as described at FlatBuffers; throws std::invalid_argument
    /// otherwise, and when `scratch` does not hold the count of `in`.
    void fill(const EncodeInput& in, const FlatSizes& sizes, const FlatBuffers& out, ThreadPool& pool,
              EncodeScratch& scratch) const;
    /// Both steps against capacities: counts, and fills `out` iff the sizes fit the index type and `capacities`. The
    /// returned sizes are true; if they overflow, no array of `out` is written. Reuses the buffers of `scratch`.
    FlatSizes encode_into(const EncodeInput& in, const FlatCapacities& capacities, const FlatBuffers& out, ThreadPool& pool,
                          EncodeScratch& scratch) const;

private:
    FlatEncoder() = default;

    std::shared_ptr<const EncodingSchema> m_schema;
    std::shared_ptr<const EncodingContext> m_context;
};

namespace detail
{
/// Throws std::overflow_error unless `value` fits the index type; `what` names the counted thing.
void check_index_range(u64 value, IndexType index, const char* what);
}
}
