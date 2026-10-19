"""Masks of lists of literals, the form in which an encoder takes the subgoals and the history of its rows.

A literal is anything ``Task.literal`` takes: a GroundLiteral or GroundAtom, ``'(on a b)'``, ``'(not (on a b))'`` or
``('on', 'a', 'b')``. It becomes a bit of the positive or negative mask of its row (and layer or step) at the slot of
its atom: the fluent slot of a fluent atom, the derived slot of a derived atom. A literal that has no slot raises
ValueError: an atom of a static predicate (its truth is the instance's, so a state cannot change it), a fluent or
derived atom that the task gave no slot, and a literal with variables.
"""

from collections.abc import Sequence

import numpy as np

from mymyr.encoding._names import _tasks_of


def _tasks(encoder, task_ids, rows):
    tasks = _tasks_of(encoder)
    if task_ids is None:
        if len(tasks) > 1:
            raise ValueError("mymyr: encoding: task_ids are required for a table of several instances")
        return [tasks[0]] * rows
    ids = np.asarray(task_ids)
    if ids.shape != (rows,):
        raise ValueError(f"mymyr: encoding: task_ids has shape {ids.shape}, expected ({rows},)")
    if rows and (ids.min() < 0 or ids.max() >= len(tasks)):
        raise ValueError(f"mymyr: encoding: task_ids must be in [0, {len(tasks)})")
    return [tasks[int(i)] for i in ids]


def _slot(task, literal):
    """``(positive, derived, slot)`` of a literal of ``task``."""
    ground = task.literal(literal)
    atom = getattr(ground, "atom", None)
    if atom is None or not hasattr(atom, "kind"):
        raise ValueError(f"mymyr: encoding: the literal {literal!r} has variables: literals must be ground")
    if atom.kind == "static":
        raise ValueError(f"mymyr: encoding: the literal {literal!r} is over a static predicate: it has no slot")
    slot = atom.slot if atom.kind == "fluent" else atom.derived_slot
    if slot is None:
        raise ValueError(f"mymyr: encoding: the literal {literal!r} has no {atom.kind} slot in its task")
    return ground.positive, atom.kind == "derived", slot


def _fill(masks, index, task, literals):
    """Sets the bits of ``literals`` in the layer ``index`` of the four mask arrays of one row."""
    pos, neg, derived_pos, derived_neg = masks
    for literal in literals:
        positive, derived, slot = _slot(task, literal)
        target = (derived_pos if positive else derived_neg) if derived else (pos if positive else neg)
        target[index, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)


def _masks(rows, layer_lists, widths):
    """The pos / neg (and derived) mask arrays [rows, layers, words] of ``layer_lists`` (per row, per layer, literals),
    padded to the longest row."""
    words, derived_words = widths
    layers = max((len(row) for row in layer_lists), default=0)
    shape = (rows, layers)
    pos = np.zeros((*shape, words), dtype=np.uint64)
    neg = np.zeros_like(pos)
    derived_pos = np.zeros((*shape, derived_words), dtype=np.uint64)
    derived_neg = np.zeros_like(derived_pos)
    return layers, (pos, neg, derived_pos, derived_neg)


def _widths(encoder):
    return encoder.table.words, encoder.table.derived_words


def _check_sequence(value, what):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"mymyr: encoding: {what} must be a sequence")


def subgoal_masks(encoder, subgoals, task_ids=None):
    """The ``subgoals`` argument of ``encoder.encode`` for the rows ``subgoals``: per row, a sequence of layers, each a
    sequence of literals (layer ``l`` fills the relations of level ``l + 1``). Returns ``(pos, neg)``, or
    ``(pos, neg, derived_pos, derived_neg)`` for a table with derived predicates: uint64 arrays of shape
    ``[rows, layers, words]`` (``[rows, layers, derived_words]``) with as many layers as the longest row has. Raises
    ValueError for a literal without a slot (see the module) and for task ids outside the table."""
    _check_sequence(subgoals, "subgoals")
    tasks = _tasks(encoder, task_ids, len(subgoals))
    layer_lists = []
    for row in subgoals:
        _check_sequence(row, "the subgoal layers of a row")
        layer_lists.append(row)
    layers, masks = _masks(len(subgoals), layer_lists, _widths(encoder))
    for r, row in enumerate(layer_lists):
        for layer, literals in enumerate(row):
            _check_sequence(literals, "a subgoal layer")
            _fill(tuple(m[r] for m in masks), layer, tasks[r], literals)
    return masks if encoder.table.derived_words else masks[:2]


def history_masks(encoder, history, task_ids=None):
    """The ``history`` argument of ``encoder.encode`` for the rows ``history``: per row, its steps oldest first, each a
    pair ``(dt, literals)``. Returns ``(pos, neg, dt, count)``, or ``(pos, neg, dt, count, derived_pos, derived_neg)``
    for a table with derived predicates: masks ``[rows, steps, words]`` (``[rows, steps, derived_words]``) of as many
    steps as the longest history, ``dt`` int32 ``[rows, steps]`` and ``count`` int32 ``[rows]`` the steps of each row.
    Raises ValueError for a literal without a slot (see the module) and for task ids outside the table."""
    _check_sequence(history, "history")
    tasks = _tasks(encoder, task_ids, len(history))
    for row in history:
        _check_sequence(row, "the history of a row")
    layers, masks = _masks(len(history), history, _widths(encoder))
    dt = np.zeros((len(history), layers), dtype=np.int32)
    count = np.fromiter((len(row) for row in history), dtype=np.int32, count=len(history))
    for r, row in enumerate(history):
        for step, item in enumerate(row):
            if isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError("mymyr: encoding: a history step is a pair (dt, literals)")
            delta, literals = item
            _check_sequence(literals, "the literals of a history step")
            dt[r, step] = delta
            _fill(tuple(m[r] for m in masks), step, tasks[r], literals)
    pos, neg, derived_pos, derived_neg = masks
    if encoder.table.derived_words:
        return pos, neg, dt, count, derived_pos, derived_neg
    return pos, neg, dt, count
