#pragma once
// Static capacity bounds of an encoding: sizes that no batch of a given number of rows over a table exceeds, computed
// from the table and the schema alone, without encoding a state.
//
//   const auto bounds = encoding::capacity_bounds(*encoder->context_ptr(), {.graphs = 64});
//   // bounds.entities, bounds.args, ... : the capacities of encode; bounds.relation_tuples[r]: the tuples of relation r
//
// The bounds of one graph are the largest over the table's instances and over the cell lists a graph can use (the
// instance's goal or goals of its own row), and those of `graphs` rows are `graphs` times them. Per relation:
//   - a fluent or derived state relation: the atom slots of its predicate (an upper bound: a state rarely holds every
//     atom it could);
//   - a static relation: the instance's static tuples (exact), none in a graph under shared statics;
//   - a numeric relation: the instance's numeric slots (exact);
//   - a goal relation of any level and derivation: the atom slots of its predicate, plus, with the instance's goal, the
//     goal's literals of fixed truth (an upper bound);
//   - a history relation: the steps times the atom slots of its predicate (an upper bound);
//   - an action relation (with `actions`): the ground actions of its schemas that the delete relaxation reaches and
//     whose static preconditions hold (heuristics::RelaxedTask): every action a reachable state has is one of them (an
//     upper bound). The grounding may exceed its budget; the action bounds are then unknown (k_unbounded).
// The scalars follow from them as the encoder counts: entities are the objects, the auxiliary entity, one per goal,
// subgoal and action target tuple and per action tuple, one per history step and history target, and the predicate
// virtual nodes; args are each relation's tuples times its arity; targets the goal, subgoal, action and history target
// tuples. `successors` bounds the expansion rows of one state (every relaxed-reachable ground action of every
// schema), times `graphs`. Rows that bring constant literals of their own (encoding::ConstantInput) are not covered.

#include "mymyr/core/types.hpp"
#include "mymyr/encoding/context.hpp"

#include <string>
#include <vector>

namespace mymyr::encoding
{
/// A bound that is not known.
inline constexpr u64 k_unbounded = ~u64{0};

struct CapacityBoundsOptions
{
    /// The rows of a batch (at least 1).
    u64 graphs = 1;
    /// Whether the rows bring expansion rows (the action relations, their entities and targets, `successors`).
    bool actions = true;
    /// With history: the most steps one row has (0: history_max_steps, which a schema that encodes every step must set).
    u64 history_steps = 0;
};

struct CapacityBounds
{
    u64 graphs = 0, entities = 0, args = 0, targets = 0, history = 0, numeric = 0;
    /// Shared statics: the static blocks of `graphs` rows (at most one per instance) and their arguments.
    u64 static_instances = 0, static_args = 0;
    /// The expansion rows of `graphs` rows (0 without actions).
    u64 successors = 0;
    /// Per relation id, the tuples of `graphs` rows.
    std::vector<u64> relation_tuples;
    /// Per relation id, 1 if some batch of `graphs` rows has exactly relation_tuples[r] tuples (the static and numeric
    /// relations), 0 for an upper bound.
    std::vector<u8> relation_exact;
    /// Why the action bounds are unknown (empty when they are known or not asked for).
    std::string unbounded_reason;
};

/// The bounds of batches of the rows of `context`'s table under its schema. Throws std::invalid_argument for a schema
/// with a horizon (its graphs have no bound of their own), for options.graphs of 0, and for a schema with history that
/// encodes every step when options.history_steps is 0.
[[nodiscard]] CapacityBounds capacity_bounds(const EncodingContext& context, const CapacityBoundsOptions& options);
}  // namespace mymyr::encoding
