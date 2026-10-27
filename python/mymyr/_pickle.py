"""Pickle reconstructors. The native __reduce__ methods name these Python functions, which pickle
can store by reference (native nanobind functions cannot be pickled).

- A Task pickles as its source (the PDDL texts, or the normalized text format) plus its options, content
  fingerprint and encoding (Task.encoding_fingerprint, and under lazy atom slots the slots' canonical ids in slot
  order); loading rebuilds it, gives lazy slots the pickled numbering and checks both fingerprints.
- A State pickles as its task, the fingerprint and its atoms: raw words under fixed slots (atoms 'auto', 'reachable'
  or 'dense': the restored task encodes them alike), the canonical ids of its atoms under lazy slots.
- An Action pickles as its task, the fingerprint and its label (schema, binding).
- A formula (Variable, Atom, Literal, GroundAtom, GroundLiteral, ConjunctiveCondition, GroundCondition) made by a Task
  pickles as that task, the fingerprint and its content in ids; loading checks it against the task.
- A StateSpace pickles as its task, the task's fingerprints, its lazy atom slots (if any), its generation statistics and
  its arrays (rows included); loading validates the arrays against the task (mymyr.datasets.StateSpace.from_arrays).
- A TupleGraph pickles as its state space, root, options and arrays; a GeneralizedStateSpace as its state spaces; a
  KnowledgeBase as its tasks, options, state spaces, task indices and packed tuple graphs. Nothing is generated again.
- A SearchTree pickles as its task, fingerprint and flat arrays, with the task's lazy slot numbering preserved.
- A search result pickles as its task, the fingerprint and its fields: the plan as action labels (schema, objects),
  states as States, statistics and statuses as numbers; loading checks the labels and atoms against the task.
- A mymyr.encoding encoder (FlatRelationEncoder, SimpleEncoder, FlatHorizonEncoder, FlatRootedHorizonEncoder,
  FlatTransitionEncoder, FlatTransitionEffectsEncoder) pickles as its class, its table, its options and its schema's
  fingerprint; loading builds the encoder again and checks the fingerprint.
- A FactLandmarkGraph pickles as its task, the task's content fingerprint and its arrays (FactLandmarkGraph.to_arrays:
  canonical atom ids, the same under lazy atom slots); loading checks the fingerprint and validates the arrays
  (FactLandmarkGraph.from_arrays). The achiever index travels as computed; nothing is grounded again.
- A mymyr.search.Heuristic pickles as its specification (task, kind, costs, evaluation; a perfect heuristic: its state
  space and costs) and is built again when loaded.
- A mymyr.rl.GoalTemplates pickles as its source, its generator options (None for templates from conditions) and its
  packed arrays; loading validates the arrays against the source (nothing is generated again).
"""

from mymyr import _core


def _restore_task(kind, a, b, a_path, b_path, options, fingerprint, encoding):
    return _core._restore_task(kind, a, b, a_path, b_path, options, fingerprint, encoding)


def _restore_frozen_task(source, atoms):
    return source.frozen(atoms)


def _restore_state(task, fingerprint, kind, payload):
    return _core._restore_state(task, fingerprint, kind, payload)


def _restore_action(task, fingerprint, schema, binding):
    return _core._restore_action(task, fingerprint, schema, binding)


def _restore_formula(task, fingerprint, kind, payload):
    return _core._restore_formula(task, fingerprint, kind, payload)


def _restore_handle(task):
    return _core._restore_handle(task)


def _restore_state_space(task, fingerprint, encoding_fingerprint, slots, stats, arrays):
    return _core._datasets._restore_state_space(task, fingerprint, encoding_fingerprint, slots, stats, arrays)


def _restore_tuple_graph(space, root, width, dominance_pruning, arrays):
    return _core._datasets._restore_tuple_graph(space, root, width, dominance_pruning, arrays)


def _restore_knowledge_base(tasks, options, spaces, task_indices, tuple_graphs):
    return _core._datasets._restore_knowledge_base(tasks, options, spaces, task_indices, tuple_graphs)


def _restore_generation_result(status, space, states, seconds):
    return _core._datasets._restore_generation_result(status, space, states, seconds)


def _restore_search_result(kind, task, fingerprint, fields):
    return _core._search._restore_search_result(kind, task, fingerprint, fields)


def _restore_landmark_graph(task, fingerprint, arrays):
    return _core._search._restore_landmark_graph(task, fingerprint, arrays)


def _restore_heuristic(task, kind, costs, evaluation):
    return _core._search.Heuristic(task, kind, costs=costs, evaluation=evaluation)


def _restore_perfect_heuristic(space, costs):
    return _core._search.Heuristic.perfect(space, costs=costs)


_HORIZON_OPTIONS = ("horizon_mode", "root_policy", "parent_relation", "sibling_relation", "cousin_relation")
_HISTORY_OPTIONS = ("history", "history_max_steps")
# the options that each encoder class does not take (its constructor has none of them or fixes them)
_NOT_TAKEN = {
    "FlatRelationEncoder": _HORIZON_OPTIONS,
    "SimpleEncoder": (*_HORIZON_OPTIONS, "outputs"),
    "FlatHorizonEncoder": _HISTORY_OPTIONS,
    "FlatRootedHorizonEncoder": _HISTORY_OPTIONS,
    "FlatTransitionEncoder": (*_HISTORY_OPTIONS, "horizon_mode", "sibling_relation", "cousin_relation"),
    "FlatTransitionEffectsEncoder": (*_HISTORY_OPTIONS, "horizon_mode", "sibling_relation", "cousin_relation"),
}


def _restore_flat_encoder(name, table, options, fingerprint):
    skipped = _NOT_TAKEN[name]
    encoder = getattr(_core._encoding, name)(table, **{k: v for k, v in options.items() if k not in skipped})
    if encoder.schema_fingerprint != fingerprint:
        raise ValueError(
            f"mymyr: encoding: the pickled encoder's schema fingerprint is {fingerprint:#018x}, the schema rebuilt from "
            f"its table and options has {encoder.schema_fingerprint:#018x} (the catalog or its rules differ)"
        )
    return encoder


def _restore_goal_templates(source, options, state):
    return _core._rl.GoalTemplates._restore(source, options, state)
