"""Action masks as torch custom ops: ``mymyr::applicable_schemas``, ``mymyr::applicable_objects`` and
``mymyr::is_applicable``, the mymyr.rl functions of the same names on a table handle (:func:`register`).

Each op returns ``(mask, status)``: the function's mask (bool [N, S], [N, O] or [N]) and an int32 status scalar. On a
CUDA device the ops run on torch's current stream, wait for nothing and allocate only their outputs and scratch (with
torch's allocator), so they trace under ``torch.compile`` and capture into CUDA graphs (warm them once outside the
capture: the first call uploads the table). Inputs the host functions reject set status bits instead of raising; the
mask rows they concern are unspecified:

- 1: a task id outside the table's instances;
- 2: a row sets atom slots its instance does not have;
- 4: the rows have fewer atom words than some row's instance;
- 8: ``depth`` is not below a selected schema's arity (``applicable_objects``);
- 16: the binding has fewer columns than a selected schema's arity (``is_applicable``).

The device kernels run tables without numeric fluents and axioms whose schemas have at most 16 parameters; over other
tables the ops copy the rows to the host and run the host functions there (they wait, raise their errors, and cannot
be captured). On the CPU the ops are the host functions (status 0).

Tensors, on the states' device and contiguous: ``states`` int64 [N, W] (the atom words, then the table's numeric
words), ``task_ids`` int32 [N] (None over a table of one instance), ``schema`` int32 [N], ``prefix`` int32
[N, >= depth] (None at depth 0) and ``binding`` int32 [N, L], with instance-local objects.
"""

from typing import Optional

import torch
from torch import Tensor

import mymyr.rl
from ._ops import _op, _stream, lookup

__all__ = ["applicable_objects", "applicable_schemas", "is_applicable"]

_SCHEMAS, _OBJECTS, _APPLICABLE = 0, 1, 2


def _sizes(table: int) -> tuple[int, int]:
    """(schema columns, object columns) of a handle's table."""
    obj = lookup(table)
    t = obj if isinstance(obj, (mymyr.rl.TaskSuite, mymyr.rl.TaskTable)) else mymyr.rl.as_table(obj)
    return t.num_schemas, t.max_objects


def _ptr(t: Optional[Tensor], name: str, ndim: int, n: int, device: torch.device) -> int:
    if t is None:
        return 0
    if t.dtype != torch.int32 or t.device != device or t.ndim != ndim or t.shape[0] != n or not t.is_contiguous():
        shape = f"[{n}]" if ndim == 1 else f"[{n}, L]"
        raise ValueError(f"mymyr: '{name}' must be a contiguous int32 tensor {shape} on {device}")
    return t.data_ptr()


def _masks(kind: int, table: int, states: Tensor, task_ids: Optional[Tensor], schema: Optional[Tensor],
           labels: Optional[Tensor], depth: int, num_objects: Optional[int]) -> tuple[Tensor, Tensor]:
    obj = lookup(table)
    dev = states.device
    if dev.type != "cuda":
        if kind == _SCHEMAS:
            mask = mymyr.rl.applicable_schemas(obj, states, task_ids)
        elif kind == _OBJECTS:
            mask = mymyr.rl.applicable_objects(obj, states, schema, labels, depth, task_ids, num_objects=num_objects)
        else:
            mask = mymyr.rl.is_applicable(obj, states, schema, labels, task_ids)
        return mask, torch.zeros((), dtype=torch.int32)
    from mymyr._core import _cuda

    if states.dtype != torch.int64 or states.ndim != 2 or not states.is_contiguous():
        raise ValueError("mymyr: 'states' must be a contiguous int64 tensor [N, W]")
    n = states.shape[0]
    ids = _ptr(task_ids, "task_ids", 1, n, dev)
    sch = _ptr(schema, "schema", 1, n, dev)
    lab = _ptr(labels, "prefix" if kind == _OBJECTS else "binding", 2, n, dev)
    view_words, _, S, O = _cuda._masks_info(obj, dev.index)
    O = O if num_objects is None else num_objects
    shape = (n, S) if kind == _SCHEMAS else (n, O) if kind == _OBJECTS else (n,)
    out = torch.empty(shape, dtype=torch.bool, device=dev)
    status = torch.empty((), dtype=torch.int32, device=dev)
    views = None
    if kind != _APPLICABLE and view_words:
        views = torch.empty((n, view_words), dtype=torch.int64, device=dev)
    _cuda._masks_ptrs(obj, kind, dev.index, _stream(states), n, states.shape[1], states.data_ptr(), ids, sch, -1, lab,
                      0 if labels is None else labels.shape[1], depth, out.data_ptr(), O if kind == _OBJECTS else 0,
                      0 if views is None else views.data_ptr(), status.data_ptr())
    return out, status


@_op("mymyr::applicable_schemas", mutates_args=())
def applicable_schemas(table: int, states: Tensor, task_ids: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """(bool [N, S], status): ``mymyr.rl.applicable_schemas`` (see the module docstring)."""
    return _masks(_SCHEMAS, table, states, task_ids, None, None, 0, None)


@applicable_schemas.register_fake
def _(table: int, states: Tensor, task_ids: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    S, _ = _sizes(table)
    return states.new_empty((states.shape[0], S), dtype=torch.bool), states.new_empty((), dtype=torch.int32)


@_op("mymyr::applicable_objects", mutates_args=())
def applicable_objects(table: int, states: Tensor, schema: Tensor, prefix: Optional[Tensor], depth: int,
                       task_ids: Optional[Tensor] = None, num_objects: Optional[int] = None) -> tuple[Tensor, Tensor]:
    """(bool [N, O], status): ``mymyr.rl.applicable_objects`` (O: num_objects, default the table's maximum object
    count; see the module docstring)."""
    return _masks(_OBJECTS, table, states, task_ids, schema, prefix, depth, num_objects)


@applicable_objects.register_fake
def _(table: int, states: Tensor, schema: Tensor, prefix: Optional[Tensor], depth: int,
      task_ids: Optional[Tensor] = None, num_objects: Optional[int] = None) -> tuple[Tensor, Tensor]:
    O = _sizes(table)[1] if num_objects is None else num_objects
    return states.new_empty((states.shape[0], O), dtype=torch.bool), states.new_empty((), dtype=torch.int32)


@_op("mymyr::is_applicable", mutates_args=())
def is_applicable(table: int, states: Tensor, schema: Tensor, binding: Tensor,
                  task_ids: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """(bool [N], status): ``mymyr.rl.is_applicable`` (see the module docstring)."""
    return _masks(_APPLICABLE, table, states, task_ids, schema, binding, 0, None)


@is_applicable.register_fake
def _(table: int, states: Tensor, schema: Tensor, binding: Tensor,
      task_ids: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    return states.new_empty((states.shape[0],), dtype=torch.bool), states.new_empty((), dtype=torch.int32)
