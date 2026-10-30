"""Literals of the instances of a table and the masks they make: the goals of rows (``goal_masks``), their subgoal layers
(``subgoal_masks``) and history steps (``history_masks``) from lists of literals, the same from arrays of predicate and
object ids (``literal_masks``), and the lookup of each literal (``literal_slots``).

A literal is anything ``Task.literal`` takes: a GroundLiteral or GroundAtom, ``'(on a b)'``, ``'(not (on a b))'`` or
``('on', 'a', 'b')``, read in the task of its row's instance (names are looked up by name, so a literal written for
another parse of the same problem works too). Problem-local derived predicates have no encoding relation and raise
ValueError even when their atoms have derived slots. In the array forms it is a table predicate id (an index into
``table.predicate_names`` below ``table.num_domain_predicates``) and the instance-local ids of its objects, padded with
-1. Its atom resolves to one of the kinds of ``LiteralKind``: a fluent or derived slot, whose bit the literal sets in
the positive or negative mask of its row (and layer or step); an atom of a static predicate, which holds in every state
or in none (``STATIC_TRUE``, ``STATIC_FALSE``); or an atom of a fluent or derived predicate without a slot, which holds
in no state (``UNREACHABLE``). The last three are literals of fixed truth: no state changes them, and the masks have no
bit for them. ``fixed`` says what to do with them: ``'decide'`` leaves out one that holds and marks its row (or layer)
unsatisfiable when it fails, ``'raise'`` raises ValueError, and ``'encode'`` keeps it as a constant literal of its row
(or layer) in the masks' ``constants`` (and still marks a row whose literal fails unsatisfiable): the encoding lists it
in its goal (subgoal) relations as the instance's goal lists its own constant literals, so a row goal is as expressive
as an instance goal. A literal with variables or an id out of range raises ValueError, an unknown name KeyError (as
``Task.literal``).

Constant literals are int32 rows ``[N, K, 3 + A]`` (``A`` the widest predicate's arity): per row up to ``K`` literals,
each its layer (0 in a goal), its table predicate id, 1 for a negative literal (else 0) and its objects padded with -1;
unused rows are -1. A row's literals are strictly ascending, as the helpers make them.

The source of every helper is an encoder (its table), a ``mymyr.rl.TaskTable`` or a Task: a table or task uses an
encoding context of the default options, made once per table or task (the slots do not depend on the options).
"""

from collections.abc import Sequence

import numpy as np

from mymyr._core._encoding import LiteralKind, _literal_masks, _literal_slots, _named_literal_masks


class _Masks(tuple):
    def __new__(cls, pos, neg, derived_pos, derived_neg, unsatisfiable, status=None, constants=None):
        masks = (pos, neg) if derived_pos is None else (pos, neg, derived_pos, derived_neg)
        result = super().__new__(cls, masks)
        result._unsatisfiable = unsatisfiable.view(np.bool_) if isinstance(unsatisfiable, np.ndarray) else unsatisfiable
        result._status = status
        result._constants = constants
        return result

    @property
    def pos(self):
        return self[0]

    @property
    def neg(self):
        return self[1]

    @property
    def derived_pos(self):
        return self[2] if len(self) == 4 else None

    @property
    def derived_neg(self):
        return self[3] if len(self) == 4 else None

    @property
    def unsatisfiable(self):
        return self._unsatisfiable

    @property
    def status(self):
        """CUDA int64 scalar: value 4 flags invalid input; None for host masks."""
        return self._status

    @property
    def constants(self):
        """int32 ``[N, K, 3 + A]``: the constant literals of each row (see the module); None without."""
        return self._constants

    def __getnewargs__(self):
        return self.pos, self.neg, self.derived_pos, self.derived_neg, self.unsatisfiable, self.status, self.constants

    def tree_flatten(self):
        extra = tuple(x for x in (self.status, self.constants) if x is not None)
        return (*self, self.unsatisfiable, *extra), (self.status is not None, self.constants is not None, len(self))

    @classmethod
    def tree_unflatten(cls, aux, children):
        has_status, has_constants, masks = aux
        children = list(children)
        constants = children.pop() if has_constants else None
        status = children.pop() if has_status else None
        pos, neg, *derived = children[:masks]
        return cls(pos, neg, *(derived or (None, None)), children[masks], status, constants)

    def __repr__(self):
        derived = "" if self.derived_pos is None else f", derived {self.derived_pos.shape}"
        constants = "" if self.constants is None else f", constants {tuple(self.constants.shape)}"
        if self.status is not None:
            return f"{type(self).__name__}(pos {self.pos.shape}{derived}{constants}, status {self.status.shape})"
        return (f"{type(self).__name__}(pos {self.pos.shape}{derived}{constants}, "
                f"{int(self.unsatisfiable.sum())} unsatisfiable)")


class GoalMasks(_Masks):
    """The goals of N rows as masks, what ``encode(goals=...)`` takes: ``pos`` and ``neg`` uint64 ``[N, W]`` over the
    fluent slots, ``derived_pos`` and ``derived_neg`` uint64 ``[N, D]`` over the derived slots (None for a table
    without derived words), and ``unsatisfiable`` bool ``[N]``: the rows whose goal has a literal of fixed truth that
    fails (``fixed='decide'``). It is a tuple ``(pos, neg)`` or ``(pos, neg, derived_pos, derived_neg)`` with read-only
    attributes. Importing ``mymyr.encoding.jax`` registers it as a JAX pytree, including ``unsatisfiable``.

    CUDA ``literal_masks`` returns int64 torch words (or uint32 halves) and a ``status`` int64 scalar: value 4 flags
    invalid entries, which were skipped. Host masks have ``status=None``.

    ``constants`` (int32 ``[N, K, 3 + A]``, None without) are the constant literals of the rows (``fixed='encode'``,
    ``instance_constants``): the goal relations list them as the instance's goal lists its own literals of fixed truth.
    Without them a literal of fixed truth is in no relation of the row. ``goal_complete`` of a row goal is 1."""


class LayerMasks(_Masks):
    """The subgoal layers of N rows as masks, what ``encode(subgoals=...)`` takes: ``pos`` and ``neg`` uint64
    ``[N, L, W]``, ``derived_pos`` and ``derived_neg`` ``[N, L, D]`` (None for a table without derived words), and
    ``unsatisfiable`` bool ``[N, L]``: the layers with a literal of fixed truth that fails (``fixed='decide'``). It is
    a tuple ``(pos, neg)`` or ``(pos, neg, derived_pos, derived_neg)`` with read-only attributes. Importing
    ``mymyr.encoding.jax`` registers it as a JAX pytree, including ``unsatisfiable``. CUDA masks carry torch words
    and a ``status`` scalar, as in ``GoalMasks``. ``constants`` are the constant literals of the layers (column 0 the
    layer), as in ``GoalMasks``."""


def _check_sequence(value, what):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"mymyr: encoding: {what} must be a sequence")


def _ints(value, name, ndim):
    array = np.asarray(value)
    if array.size == 0:
        array = array.astype(np.int32)
    if array.dtype.kind not in "iu":
        raise TypeError(f"mymyr: encoding: {name} must hold integers")
    if array.ndim != ndim:
        raise ValueError(f"mymyr: encoding: {name} must have {ndim} dimension{'s' if ndim > 1 else ''}, "
                         f"not shape {array.shape}")
    if array.size and (array.min() < np.iinfo(np.int32).min or array.max() > np.iinfo(np.int32).max):
        raise ValueError(f"mymyr: encoding: {name} has a value that int32 cannot hold")
    return np.ascontiguousarray(array, dtype=np.int32)


def _task_ids(task_ids):
    return None if task_ids is None else _ints(task_ids, "task_ids", 1)


def _rows(rows, what, layered):
    _check_sequence(rows, what)
    out = []
    for row in rows:
        _check_sequence(row, f"the {'layers' if layered else 'literals'} of a row")
        if layered:
            for layer in row:
                _check_sequence(layer, "a layer of literals")
            out.append([list(layer) for layer in row])
        else:
            out.append(list(row))
    return out


def _fixed(fixed):
    if fixed not in ("decide", "raise", "encode"):
        raise ValueError(f"mymyr: encoding: fixed must be 'decide', 'raise' or 'encode', not {fixed!r}")
    return fixed


def _masks(cls, arrays):
    pos, neg, derived_pos, derived_neg, unsatisfiable, constants = arrays
    return cls(pos, neg, derived_pos, derived_neg, unsatisfiable, constants=constants)


def goal_masks(source, goals, task_ids=None, *, fixed="decide", instance_constants=False):
    """The goals of the rows ``goals`` (per row, a sequence of literals) as ``GoalMasks``, for ``encode(goals=...)``.
    ``task_ids`` [N] is the instance of each row (required for a table of several instances). ``fixed``: ``'decide'``
    leaves out a literal of fixed truth that holds and sets ``unsatisfiable`` of its row for one that fails, ``'raise'``
    raises ValueError for any, ``'encode'`` keeps them as constant literals (see the module). ``instance_constants``
    adds the constant part of each row's instance goal (its literals of fixed truth) to the row's constant literals, so
    that a row goal of the instance's own literals encodes as the instance goal does."""
    rows = _rows(goals, "goals", False)
    return _masks(GoalMasks, _named_literal_masks(source, rows, _task_ids(task_ids), False, _fixed(fixed),
                                                  bool(instance_constants)))


def subgoal_masks(source, subgoals, task_ids=None, *, fixed="raise"):
    """The ``subgoals`` argument of ``encode`` for the rows ``subgoals`` (per row, a sequence of layers, each a sequence
    of literals; layer ``l`` fills the relations of level ``l + 1``) as ``LayerMasks``, with as many layers as the
    longest row has. ``fixed`` as in ``goal_masks``; ``'raise'`` by default, since a subgoal that no state can change
    is most likely a mistake. Raises ValueError for task ids outside the table."""
    rows = _rows(subgoals, "subgoals", True)
    return _masks(LayerMasks, _named_literal_masks(source, rows, _task_ids(task_ids), True, _fixed(fixed)))


def history_masks(source, history, task_ids=None):
    """The ``history`` argument of ``encode`` for the rows ``history``: per row, its steps oldest first, each a pair
    ``(dt, literals)``. Returns ``(pos, neg, dt, count)``, or ``(pos, neg, dt, count, derived_pos, derived_neg)``
    for a table with derived predicates: masks ``[rows, steps, words]`` (``[rows, steps, derived_words]``) of as many
    steps as the longest history, ``dt`` int32 ``[rows, steps]`` and ``count`` int32 ``[rows]`` the steps of each row.
    A step says which literals were observed, so a literal of fixed truth raises ValueError (see the module), as do
    task ids outside the table."""
    _check_sequence(history, "history")
    for row in history:
        _check_sequence(row, "the history of a row")
    steps = max((len(row) for row in history), default=0)
    dt = np.zeros((len(history), steps), dtype=np.int32)
    count = np.fromiter((len(row) for row in history), dtype=np.int32, count=len(history))
    rows = []
    for r, row in enumerate(history):
        layers = []
        for step, item in enumerate(row):
            if isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError("mymyr: encoding: a history step is a pair (dt, literals)")
            delta, literals = item
            _check_sequence(literals, "the literals of a history step")
            dt[r, step] = delta
            layers.append(list(literals))
        rows.append(layers)
    pos, neg, derived_pos, derived_neg, _, _ = _named_literal_masks(source, rows, _task_ids(task_ids), True, "raise")
    if derived_pos is not None:
        return pos, neg, dt, count, derived_pos, derived_neg
    return pos, neg, dt, count


def _literal_arrays(predicates, objects):
    predicates = _ints(predicates, "predicates", 1)
    objects = np.asarray(objects)
    if objects.size == 0 and objects.ndim < 2:
        objects = objects.reshape(len(predicates), 0)
    objects = _ints(objects, "objects", 2)
    if objects.shape[0] != predicates.shape[0]:
        raise ValueError(f"mymyr: encoding: objects has {objects.shape[0]} rows, predicates {predicates.shape[0]}")
    return predicates, objects


def literal_slots(source, predicates, objects, task_ids=None):
    """The lookup of N literals given as arrays: ``predicates`` [N] table predicate ids, ``objects`` [N, k]
    instance-local object ids, each row the predicate's arity in objects followed by -1, and ``task_ids`` [N] the
    instance of each literal (required for a table of several instances). Returns ``(kind, slot)``: ``kind`` uint8
    [N] a ``LiteralKind`` for each literal (``INVALID`` for ids out of range, which raise nowhere here) and ``slot``
    int32 [N] its fluent or derived slot (-1 for the other kinds). With CUDA torch tensors, source must be an
    encoder and the result is ``(kind, slot, status)``: status is an int64 scalar with value 4 for invalid input.
    Arrays must be contiguous int32 on one device. Warm on the capture stream before CUDA graph capture and keep
    the encoder, inputs and outputs alive through the last replay; no host synchronization occurs."""
    if getattr(predicates, "is_cuda", False):
        from ._device_literals import slots
        return slots(source, predicates, objects, task_ids)
    predicates, objects = _literal_arrays(predicates, objects)
    return _literal_slots(source, predicates, objects, _task_ids(task_ids))


def literal_masks(source, predicates, objects, positive, rows, num_rows, task_ids=None, *, layers=None,
                  num_layers=None, fixed="decide", instance_constants=False, word_dtype=None):
    """The masks of N literals given as arrays (``predicates`` and ``objects`` as in ``literal_slots``): literal ``i``
    has the sign ``positive[i]`` and goes to the row ``rows[i]`` of ``num_rows``, whose instance is ``task_ids[row]``
    (``task_ids`` [num_rows], required for a table of several instances). Returns ``GoalMasks``; with ``layers`` [N]
    (the layer of each literal, of ``num_layers``, by default one more than the largest) ``LayerMasks`` for
    ``encode(subgoals=...)``. ``fixed`` and ``instance_constants`` (goals only) as in ``goal_masks``. Raises ValueError
    for an invalid literal, row or layer.

    CUDA torch arrays require an encoder, contiguous int32 ids, bool signs, explicit ``num_layers`` with layers,
    ``fixed='decide'`` and no instance constants (constant literals are made on the host). Results are torch tensors,
    int64 words (or ``word_dtype=torch.uint32`` for low/high halves), bool unsatisfiable, and ``status`` an int64
    scalar (value 4 for invalid input, which is skipped).
    Warm on the capture stream and retain the encoder and buffers through the last replay. No host read occurs."""
    if getattr(predicates, "is_cuda", False):
        import torch
        from ._device_literals import masks
        if _fixed(fixed) != "decide" or instance_constants:
            raise ValueError("mymyr: encoding: CUDA literal masks require fixed='decide' and no instance constants")
        if (layers is None) != (num_layers is None):
            raise ValueError("mymyr: encoding: CUDA layers require explicit num_layers, and num_layers requires layers")
        if word_dtype not in (None, torch.int64, torch.uint32):
            raise ValueError("mymyr: encoding: word_dtype must be torch.int64 or torch.uint32")
        pos, neg, dp, dn, unsatisfiable, status = masks(
            source, predicates, objects, positive, rows, num_rows, task_ids, layers,
            1 if num_layers is None else num_layers)
        if word_dtype == torch.uint32:
            pos, neg, dp, dn = (a.view(torch.uint32) for a in (pos, neg, dp, dn))
        return (GoalMasks if layers is None else LayerMasks)(
            pos, neg, dp if dp.shape[-1] else None, dn if dn.shape[-1] else None, unsatisfiable, status)
    if word_dtype is not None:
        raise ValueError("mymyr: encoding: word_dtype is available for CUDA tensors")
    predicates, objects = _literal_arrays(predicates, objects)
    count = predicates.shape[0]
    positive = np.asarray(positive)
    if positive.shape != (count,):
        raise ValueError(f"mymyr: encoding: positive has shape {positive.shape}, expected ({count},)")
    positive = np.ascontiguousarray(positive != 0, dtype=np.uint8)
    rows = _ints(rows, "rows", 1)
    if rows.shape != (count,):
        raise ValueError(f"mymyr: encoding: rows has shape {rows.shape}, expected ({count},)")
    if layers is not None:
        layers = _ints(layers, "layers", 1)
        if layers.shape != (count,):
            raise ValueError(f"mymyr: encoding: layers has shape {layers.shape}, expected ({count},)")
        if num_layers is None:
            num_layers = int(layers.max()) + 1 if count else 0
    elif num_layers is not None:
        raise ValueError("mymyr: encoding: num_layers goes with layers")
    if num_rows < 0 or (num_layers is not None and num_layers < 0):
        raise ValueError("mymyr: encoding: num_rows and num_layers must not be negative")
    if instance_constants and layers is not None:
        raise ValueError("mymyr: encoding: instance_constants go with goals, not with layers")
    arrays = _literal_masks(source, predicates, objects, positive, rows, int(num_rows), _task_ids(task_ids), layers,
                            int(num_layers or 0), _fixed(fixed), bool(instance_constants))
    return _masks(GoalMasks if layers is None else LayerMasks, arrays)
