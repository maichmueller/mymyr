#pragma once

// A checker of the array contract of an encoded batch (encoder.hpp), for tests and debugging: it is linear in the batch
// and not meant for hot paths.

#include "mymyr/encoding/schema.hpp"
#include "mymyr/rl/task_arrays.hpp"

namespace mymyr::encoding
{
/// Throws std::invalid_argument ("mymyr: encoding: ...") naming the first violation of the contract of a batch encoded
/// for `schema`:
/// - array names, dtypes and shapes, and the scalars; entity_role, entity_ref and relation_args may be longer than the
///   batch needs (a capacity bundle), with zeros past it; relation_counts is present iff the schema stores counts, and
///   entity_role and entity_ref are present iff the schema does not have implicit entity arrays;
/// - offsets: start at 0, never decrease, end at the sizes of the arrays they index; the differences of relation_offsets
///   (cell (g, r) is segment g * R + r, or r * G + g in a relation-major batch) equal relation_counts, those of
///   relation_arg_offsets the counts times the relations' arities;
/// - every argument lies in the entity range of its graph, and the entity at an argument position has the role of the
///   position's slot role (an object for an argument slot, the auxiliary entity for an auxiliary slot, ...);
/// - each graph's objects come first and have their own index as entity_ref; a graph has at most one auxiliary entity, at
///   local index num_objects, and has it iff one of its tuples uses an auxiliary slot;
/// - no tuple occurs twice in a cell, and the tuples of a cell ascend strictly in lexicographic order;
/// - goal_complete is present iff the schema has goal relations, with values 0 and 1; the target arrays are present iff
///   the schema has targets, and target_offsets is a valid offset array;
/// - the entities of a graph past its objects are, in this order: the auxiliary entity; the goal entities, one per tuple of
///   the plain goal relations with a goal target, numbered in the order of those relations and tuples, each the first
///   argument of its tuple and referring to the tuple by its index among the graph's tuples in graph-major order (the
///   cells' tuples in relation order); the action entities, one per tuple of an action relation, numbered the same way,
///   each the first argument of its tuple and referring to a row of the graph's expansion (a non-negative entity_ref);
///   the predicate virtual entities, one per predicate that has a tuple in the graph and none for the others, ascending
///   in the name of the predicate (EncodingSchema::virtual_rank; the entity_ref is the predicate's key,
///   EncodingSchema::virtual_key), each the predicate slot of exactly the tuples of its predicate's relations;
/// - the graph's targets are its goal entities in group 0, then, with action targets, its action entities in group 2.
void validate(const rl::ArrayBundle& batch, const EncodingSchema& schema);
}
