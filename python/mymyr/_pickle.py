"""Pickle reconstructors. The native __reduce__ methods name these Python functions, which pickle
can store by reference (native nanobind functions cannot be pickled).

- A Task pickles as its source (the PDDL texts, or the normalized text format) plus its options and content
  fingerprint; loading rebuilds it and checks the fingerprint.
- A State pickles as its task, the fingerprint and its atoms: raw words under frozen slots (slot = canonical id), the
  canonical ids of its atoms under lazy slots (lazy slot numbers depend on the order of first touch).
- An Action pickles as its task, the fingerprint and its label (schema, binding).
- A formula (Variable, Atom, Literal, GroundAtom, GroundLiteral, ConjunctiveCondition, GroundCondition) made by a Task
  pickles as that task, the fingerprint and its content in ids; loading checks it against the task.
"""

from mymyr import _core


def _restore_task(kind, a, b, a_path, b_path, options, fingerprint):
    return _core._restore_task(kind, a, b, a_path, b_path, options, fingerprint)


def _restore_state(task, fingerprint, kind, payload):
    return _core._restore_state(task, fingerprint, kind, payload)


def _restore_action(task, fingerprint, schema, binding):
    return _core._restore_action(task, fingerprint, schema, binding)


def _restore_formula(task, fingerprint, kind, payload):
    return _core._restore_formula(task, fingerprint, kind, payload)


def _restore_handle(task):
    return _core._restore_handle(task)
