"""The torch custom ops of mymyr.rl.torch (torch.library).

Tables, tasks and environments enter an op as an int handle (:func:`register`): a custom op's arguments are tensors
and scalars only. Every op runs on the input's device, on torch's current stream there (host arrays: on the CPU), and
has a fake implementation, so it traces under ``torch.compile`` (no graph break) and under ``torch.export``.

CUDA graphs: ``reset``, ``refresh``, ``step`` and ``random_actions`` of a device environment on the fast path
synchronize nothing and allocate nothing (the environment's scratch is sized for its batch), so they can be captured
(``torch.cuda.graph``, ``torch.compile(mode="reduce-overhead")``) and replayed; the tensors they mutate must keep their
addresses (BatchedEnv marks its tensors static). The general path synchronizes on the successor count: its calls go
through ``refresh_sync`` and ``step_sync`` (the same ops, tagged ``cudagraph_unsafe``: torch.compile leaves them out of
its CUDA graphs), and so do ``expand`` and ``expand_flat``.

The environment state over a table: ``states`` [N, row_words] int64, ``task_ids`` [N] int32 (the rows' instances;
None over a table of one), ``steps``, ``draws``, the fast path's cache (``counts``, ``views``), optional per-env goal
masks ``goal_pos`` / ``goal_neg`` [N, words] int64 (the step's goal test is then the mask test) and ``at_goal`` [N]
bool, whether each row's state is a goal (reset, refresh and step keep it; a step absorbs those rows without testing
the states again; required on a device).

- ``mymyr::reset(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, at_goal, mask,
  keep_goals)``: resets rows (all, or those with mask[i]) in place into their instances (task_ids, set by the caller
  first).
- ``mymyr::refresh(env, states, task_ids, counts, views, goal_pos, goal_neg, count, at_goal)``: recomputes the cache,
  counts and at_goal flags of states (task ids, goal masks) the caller wrote.
- ``mymyr::step(env, states, task_ids, steps, draws, counts, views, count, at_goal, goal_pos, goal_neg, action,
  next_task_ids, first_env, final, action_schema=None, action_binding=None)``: one step of every environment in place (autoresetting rows restart in
  next_task_ids[i] where given); returns (reward, terminated, truncated, goal, invalid, schema, binding, final_states).
- ``mymyr::refresh_sync`` / ``mymyr::step_sync``: ``refresh`` / ``step`` of environments whose calls synchronize (the
  device's general path), tagged ``cudagraph_unsafe``.
- ``mymyr::random_actions(env, draws, count, first_env, max_actions)``: the random policy's next choices [N] int64
  (the counter-based RNG at the draw counters; no step).
- ``mymyr::expand(table, states, task_ids, k, witness_pruning)``: the padded expansion [N, k] with the same fields as expand_flat.
- ``mymyr::expand_flat(table, states, task_ids, capacity, witness_pruning)``: the flat expansion (succ, parent, schema, binding, goal, offsets,
  derived, succ_hash, noop, index, status). ``capacity=None`` sizes the outputs by the data: the row count M is an unbacked symbolic
  size under ``torch.compile``; a capacity gives static shapes (rows past it are dropped, rows past the total are
  padding: succ 0, parent / schema / binding -1, goal False; offsets keep the true counts).
- ``mymyr::bounded_actions(encoder, states, task_ids, entities, successors=None, drop_noops=False)``: the label-only
  expansion of a ``register``-ed encoder's task whose rows the action entities of an encoding with the entity
  capacity ``entities`` stand for, with successor digests and no-op flags: (labels, binding, offsets, succ_hash, noop,
  status), the rows of ``labels`` parent, schema and, with ``drop_noops``, index. Static shapes, zero past the kept
  rows; ``successors`` bounds the successors before ``drop_noops`` drops the no-op rows. It waits for nothing, so it
  captures (no ``cudagraph_unsafe`` tag).
- ``mymyr::encode_flat(encoder, states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg,
  subgoal_pos, subgoal_neg, subgoal_derived_pos, subgoal_derived_neg, history_pos, history_neg, history_dt,
  history_count, history_derived_pos, history_derived_neg, graphs, entities, args, targets, history, numeric=0,
  static_instances=0, static_args=0)``: the encoding of device rows by a ``FlatRelationEncoder`` (``register``
  handle) with the capacities ``graphs``, ``entities``, ``args``, ``targets``, ``history``, ``numeric``,
  ``static_instances`` and ``static_args`` (0 for an absent lane): a list of 29 tensors, the arrays of a batch in the
  order of ``DeviceFlatBatch.array_names`` (an array the schema does not store is an empty tensor), ``static_sizes``
  int64 [3] and ``sizes`` int64 [6] ([7] with the numeric value count appended). It waits for nothing and allocates
  only its outputs and scratch, on torch's current stream, so it captures in a CUDA graph (warm it once on the capture
  stream first; no scratch reservation is needed) and traces under ``torch.compile``. The expansion rows and horizon
  views follow as keyword tensors: ``action_schema``, ``action_binding``, ``action_offsets``, ``action_parent``,
  ``node_offsets``, ``edge_offsets``, ``edge_src``, ``edge_dst``, ``edge_schema``, ``edge_binding`` and
  ``action_succ``.
- ``mymyr::visited_mask(table, count, succ_hash, parent, rows)``: membership of successor hashes in visited tables
  (bool [M]); ``mymyr::clear_visited(env, ...)``: restarts the visited tables of rows.
- ``mymyr::target_visited(target, role, ref, entity_offsets, live, row_offsets, row_index, succ_hash, status, table,
  count, env_rows)``: whether each action target of an encoded batch reaches a visited successor (bool [T]; one
  thread per target, the observations' ``action_visited``).
- ``mymyr::literal_slots`` / ``mymyr::literal_masks``: CUDA lookup of table predicate ids and instance-local objects
  on a registered encoder; kinds and slots, or goal/subgoal masks, with a device status scalar for invalid input.
The expansion ops take ``derived``, ``rows``, ``hashes``, ``noops`` and ``drop_noops`` flags and return the same
eleven fields: succ, parent, schema, binding, goal, offsets, derived, succ_hash, noop, index, status. Absent parts
are empty tensors; a capacity keeps counts and status on the device.
"""

import itertools
import threading
from typing import Any, Optional

import torch
from torch import Tensor

import mymyr.rl
from mymyr.encoding import _device
from mymyr.rl._ops import tensor_view

__all__ = [
    "bounded_actions",
    "encode_flat",
    "literal_masks",
    "literal_slots",
    "match_first",
    "expand",
    "expand_flat",
    "lookup",
    "random_actions",
    "refresh",
    "refresh_sync",
    "register",
    "reset",
    "step",
    "step_sync",
    "unregister",
]

_unsafe = (torch.Tag.cudagraph_unsafe,)


def _op(name: str, **kwargs: Any) -> Any:
    """torch.library.custom_op that keeps the function's docstring on the op."""

    def wrap(fn: Any) -> Any:
        op = torch.library.custom_op(name, **kwargs)(fn)
        op.__doc__ = fn.__doc__
        return op

    return wrap

_lock = threading.Lock()
_objects: dict[int, Any] = {}
_by_id: dict[int, int] = {}
_ids = itertools.count(1)


def register(obj: Any) -> int:
    """The op handle of a task source, host or uploaded templates, an encoder or a native Env (the same handle for
    the same object). The
    registry holds a reference until :func:`unregister`."""
    with _lock:
        h = _by_id.get(id(obj))
        if h is not None and _objects.get(h) is obj:
            return h
        h = next(_ids)
        _objects[h] = obj
        _by_id[id(obj)] = h
        return h


def unregister(handle: int) -> None:
    """Drops the registry's reference (ops with this handle fail afterwards)."""
    with _lock:
        obj = _objects.pop(handle, None)
        _info.pop(handle, None)
        if obj is not None and _by_id.get(id(obj)) == handle:
            del _by_id[id(obj)]


def _register_owned(obj: Any) -> int:
    """An independent handle whose owner may unregister it without affecting other registrations of ``obj``."""
    with _lock:
        h = next(_ids)
        _objects[h] = obj
        return h


def lookup(handle: int) -> Any:
    """The object registered under ``handle`` (:func:`register`). Raises ValueError for an unknown handle."""
    if isinstance(handle, torch.SymInt):
        # A handle passed to a compiled function becomes symbolic once it varies: specialize on it (one guard and
        # one recompilation per handle) since every handle names a different object.
        handle = int(handle)
    try:
        return _objects[handle]
    except KeyError:
        raise ValueError(f"mymyr: no object registered under handle {handle} (see mymyr.rl.torch.register)") from None


def _stream(t: Tensor | torch.device) -> Optional[int]:
    device = t if isinstance(t, torch.device) else t.device
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None


# ------------------------------------------------------------------------------------------------ environments
# The ops check the tensors here (dtype, shape, device, contiguity) and pass their data pointers to the native Env
# (Env._step_ptrs, ...), which runs on torch's current stream: a DLPack import costs about 10 us per torch tensor, a
# check about 1 us. The *_impl functions are the eager entry points of BatchedEnv (no dispatcher round trip).

_info: dict[int, tuple] = {}  # handle -> (env, (device, row_words, words, S, V, L))


def _env_info(env: int) -> tuple[Any, tuple]:
    e = lookup(env)
    info = _info.get(env)
    if info is None or info[0] is not e:
        dev = torch.device("cpu") if e.device is None else torch.device("cuda", e.device)
        info = (e, (dev, e.row_words, e.words, e.cache_schemas, e.cache_view_words, e.label_width))
        _info[env] = info
    return info


def _ptr(t: Optional[Tensor], name: str, dtype: torch.dtype, shape: tuple, device: torch.device) -> int:
    if t is None:
        return 0
    if t.dtype != dtype or t.device != device or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(
            f"mymyr: '{name}' must be a contiguous {dtype} tensor of shape {list(shape)} on {device}, not a "
            f"{t.dtype} tensor of shape {list(t.shape)} on {t.device}"
        )
    return t.data_ptr()


def _batch_ptrs(
    info: tuple,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Optional[Tensor],
    draws: Optional[Tensor],
    counts: Optional[Tensor],
    views: Optional[Tensor],
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
) -> list[int]:
    """Pointers of states, task_ids, steps, draws, counts, views, goal_pos, goal_neg."""
    dev, rw, w, S, V, _ = info
    n = states.shape[0]
    return [
        _ptr(states, "states", torch.int64, (n, rw), dev),
        _ptr(task_ids, "task_ids", torch.int32, (n,), dev),
        _ptr(steps, "steps", torch.int32, (n,), dev),
        _ptr(draws, "draws", torch.int64, (n,), dev),
        _ptr(counts, "counts", torch.int32, (n, S), dev),
        _ptr(views, "views", torch.int64, (n, V), dev),
        _ptr(goal_pos, "goal_pos", torch.int64, (n, w), dev),
        _ptr(goal_neg, "goal_neg", torch.int64, (n, w), dev),
    ]


def _visited_ptrs(info: tuple, states: Tensor, visited: Optional[Tensor], visited_count: Optional[Tensor],
                  visited_status: Optional[Tensor]) -> tuple[list[int], int]:
    if visited is None:
        if visited_count is not None or visited_status is not None:
            raise ValueError("mymyr: visited tracking needs table, count and status arrays")
        return [0, 0, 0], 0
    if visited.ndim != 2:
        raise ValueError("mymyr: visited must have shape [N, H]")
    n, H = states.shape[0], visited.shape[1]
    if H < 2 or H > 1 << 20 or H & (H - 1):
        raise ValueError("mymyr: visited capacity must be a power of two in [2, 2^20]")
    if visited_count is None or visited_status is None:
        raise ValueError("mymyr: visited tracking needs table, count and status arrays")
    return [
        _ptr(visited, "visited", torch.int64, (n, H), info[0]),
        _ptr(visited_count, "visited_count", torch.int32, (n,), info[0]),
        _ptr(visited_status, "visited_status", torch.int32, (1,), info[0]),
    ], H


def reset_impl(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Tensor,
    counts: Tensor,
    views: Tensor,
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    count: Tensor,
    at_goal: Optional[Tensor],
    mask: Optional[Tensor],
    keep_goals: bool,
    visited: Optional[Tensor] = None,
    visited_count: Optional[Tensor] = None,
    visited_status: Optional[Tensor] = None,
) -> None:
    """The body of ``mymyr::reset``."""
    e, info = _env_info(env)
    n = states.shape[0]
    p = _batch_ptrs(info, states, task_ids, steps, None, counts, views, goal_pos, goal_neg)
    del p[3]  # draws
    p += [
        _ptr(count, "count", torch.int32, (n,), info[0]),
        _ptr(mask, "mask", torch.bool, (n,), info[0]),
        _ptr(at_goal, "at_goal", torch.bool, (n,), info[0]),
    ]
    vp, H = _visited_ptrs(info, states, visited, visited_count, visited_status)
    e._reset_ptrs(n, p + vp, keep_goals, _stream(states), H)


def refresh_impl(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    counts: Tensor,
    views: Tensor,
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    count: Tensor,
    at_goal: Optional[Tensor],
) -> None:
    """The body of ``mymyr::refresh``."""
    e, info = _env_info(env)
    n = states.shape[0]
    p = _batch_ptrs(info, states, task_ids, None, None, counts, views, goal_pos, goal_neg)
    p = [
        p[0], p[1], p[4], p[5], p[6], p[7], _ptr(count, "count", torch.int32, (n,), info[0]),
        _ptr(at_goal, "at_goal", torch.bool, (n,), info[0]),
    ]  # fmt: skip
    e._refresh_ptrs(n, p, _stream(states))


def _step_outputs(label_width: int, states: Tensor, final: bool) -> tuple[Tensor, ...]:
    n = states.shape[0]
    b = torch.empty(n, dtype=torch.bool, device=states.device)
    return (
        states.new_empty(n, dtype=torch.float32),  # reward
        b,  # terminated
        torch.empty_like(b),  # truncated
        torch.empty_like(b),  # goal
        torch.empty_like(b),  # invalid
        states.new_empty(n, dtype=torch.int32),  # schema
        states.new_empty((n, label_width), dtype=torch.int32),  # binding
        states.new_empty((n if final else 0, states.shape[1])),  # final_states
    )


def step_impl(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Tensor,
    draws: Tensor,
    counts: Tensor,
    views: Tensor,
    count: Tensor,
    at_goal: Optional[Tensor],
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
    action_schema: Optional[Tensor] = None,
    action_binding: Optional[Tensor] = None,
    visited: Optional[Tensor] = None,
    visited_count: Optional[Tensor] = None,
    visited_status: Optional[Tensor] = None,
    revisited: Optional[Tensor] = None,
    out: Optional[tuple[Tensor, ...]] = None,
) -> tuple[Tensor, ...]:
    """The body of ``mymyr::step``; ``out``: the 8 outputs' destinations (contiguous, of _step_outputs' shapes and
    dtypes; CapturedRollout's slices of its stacked outputs), else new tensors."""
    e, info = _env_info(env)
    dev, L = info[0], info[5]
    n = states.shape[0]
    if (action_schema is None) != (action_binding is None):
        raise ValueError("mymyr: label actions require both action_schema and action_binding")
    stride = 0
    if action_binding is not None:
        if action_binding.ndim != 2:
            raise ValueError("mymyr: action_binding must have shape [N, L]")
        stride = action_binding.shape[1]
    if out is None:
        out = _step_outputs(L, states, final)
    else:
        shapes = [(n,)] * 6 + [(n, L), (n if final else 0, states.shape[1])]
        dtypes = [torch.float32] + [torch.bool] * 4 + [torch.int32] * 2 + [states.dtype]
        for x, shape, dtype in zip(out, shapes, dtypes, strict=True):
            if tuple(x.shape) != shape or x.dtype != dtype or x.device != states.device or not x.is_contiguous():
                raise ValueError("mymyr: a step's output destinations must match its outputs")
    reward, terminated, truncated, goal, invalid, schema, binding, final_states = out
    p = _batch_ptrs(info, states, task_ids, steps, draws, counts, views, goal_pos, goal_neg)
    p += [
        _ptr(action, "action", torch.int64, (n,), dev),
        _ptr(next_task_ids, "next_task_ids", torch.int32, (n,), dev),
        reward.data_ptr(),
        terminated.data_ptr(),
        truncated.data_ptr(),
        _ptr(count, "count", torch.int32, (n,), dev),
        final_states.data_ptr() if final else 0,
        schema.data_ptr(),
        binding.data_ptr(),
        invalid.data_ptr(),
        goal.data_ptr(),
        _ptr(at_goal, "at_goal", torch.bool, (n,), dev),
        _ptr(action_schema, "action_schema", torch.int32, (n,), dev),
        _ptr(action_binding, "action_binding", torch.int32, (n, stride), dev),
    ]
    vp, H = _visited_ptrs(info, states, visited, visited_count, visited_status)
    rp = _ptr(revisited, "revisited", torch.bool, (n,), dev)
    e._step_ptrs(n, p + vp + [rp], L, first_env, _stream(states), stride, H)
    return out


def visited_mask_impl(table: Tensor, count: Tensor, succ_hash: Tensor, parent: Tensor,
                      rows: Optional[Tensor] = None) -> Tensor:
    """The eager body of ``mymyr::visited_mask`` (no dispatcher: an eager call of a custom op imports
    torch._dynamo)."""
    from mymyr._core import _rl_ops

    out = torch.empty(succ_hash.shape, dtype=torch.bool, device=table.device)
    v = tensor_view
    _rl_ops.visited_mask(v(table), v(count), v(succ_hash), v(parent), rows=v(rows), out=v(out), stream=_stream(table))
    return out


@_op("mymyr::visited_mask", mutates_args=())
def visited_mask(table: Tensor, count: Tensor, succ_hash: Tensor, parent: Tensor,
                 rows: Optional[Tensor] = None) -> Tensor:
    """Membership of successor hashes in visited tables (CPU or CUDA tensors; CUDA on torch's current stream, without
    a host synchronization); -1 parent or mapped row denotes padding."""
    return visited_mask_impl(table, count, succ_hash, parent, rows)


@visited_mask.register_fake
def _(table, count, succ_hash, parent, rows=None):
    return succ_hash.new_empty(succ_hash.shape, dtype=torch.bool)


def target_visited_impl(target: Tensor, role: Tensor, ref: Tensor, entity_offsets: Tensor, live: Optional[Tensor],
                        row_offsets: Tensor, row_index: Optional[Tensor], succ_hash: Tensor, status: Optional[Tensor],
                        table: Tensor, count: Tensor, env_rows: Optional[Tensor]) -> Tensor:
    """The eager body of ``mymyr::target_visited``."""
    from mymyr._core import _rl_ops

    out = torch.empty(target.shape, dtype=torch.bool, device=table.device)
    v = tensor_view
    _rl_ops.target_visited(v(target), v(role), v(ref), v(entity_offsets), v(row_offsets), v(succ_hash), v(table),
                           v(count), live=v(live), row_index=v(row_index), status=v(status), env_rows=v(env_rows),
                           out=v(out), stream=_stream(table))
    return out


@_op("mymyr::target_visited", mutates_args=())
def target_visited(target: Tensor, role: Tensor, ref: Tensor, entity_offsets: Tensor, live: Optional[Tensor],
                   row_offsets: Tensor, row_index: Optional[Tensor], succ_hash: Tensor, status: Optional[Tensor],
                   table: Tensor, count: Tensor, env_rows: Optional[Tensor]) -> Tensor:
    """Whether each target of an encoded batch (``target`` [T]: target_entity; ``live`` int64 [1]: the live targets,
    None: all) is an action entity (``role``: entity_role) whose successor row is in its environment's visited table:
    bool [T]. The graph of an entity comes from ``entity_offsets``, its row from ``entity_ref`` (``ref``), the graph's
    rows ``[row_offsets[g], row_offsets[g + 1])`` and, with the drop of no-op rows, ``row_index``; ``env_rows`` [G]
    maps graphs to table rows (None: graph g is row g). A nonzero ``status`` [1] makes every target false. CPU or CUDA
    tensors (one thread per target on torch's current stream, without a host synchronization)."""
    return target_visited_impl(target, role, ref, entity_offsets, live, row_offsets, row_index, succ_hash, status,
                               table, count, env_rows)


@target_visited.register_fake
def _(target, role, ref, entity_offsets, live, row_offsets, row_index, succ_hash, status, table, count, env_rows):
    return target.new_empty(target.shape, dtype=torch.bool)


def clear_visited_impl(env: int, states: Tensor, task_ids: Optional[Tensor], visited: Tensor,
                       visited_count: Tensor, visited_status: Tensor, mask: Optional[Tensor]) -> None:
    """The eager body of ``mymyr::clear_visited``."""
    e, info = _env_info(env)
    n = states.shape[0]
    p = _batch_ptrs(info, states, task_ids, None, None, None, None, None, None)[:2]
    vp, H = _visited_ptrs(info, states, visited, visited_count, visited_status)
    e._clear_visited_ptrs(n, p + vp + [_ptr(mask, "mask", torch.bool, (n,), info[0])], _stream(states), H)


@_op("mymyr::clear_visited", mutates_args=("visited", "visited_count"))
def clear_visited(env: int, states: Tensor, task_ids: Optional[Tensor], visited: Tensor,
                  visited_count: Tensor, visited_status: Tensor, mask: Optional[Tensor]) -> None:
    """``mymyr::clear_visited``: the visited tables of the rows with ``mask[i]`` (None: all) hold their current states
    only; the overflow status is kept."""
    clear_visited_impl(env, states, task_ids, visited, visited_count, visited_status, mask)


@clear_visited.register_fake
def _(env, states, task_ids, visited, visited_count, visited_status, mask):
    return None


def random_actions_impl(env: int, draws: Tensor, count: Tensor, first_env: int, max_actions: Optional[int]) -> Tensor:
    """The body of ``mymyr::random_actions``."""
    e, info = _env_info(env)
    dev = info[0]
    n = draws.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=draws.device)
    p = [_ptr(draws, "draws", torch.int64, (n,), dev), _ptr(count, "count", torch.int32, (n,), dev), out.data_ptr()]
    e._random_actions_ptrs(n, p, first_env, max_actions, _stream(draws))
    return out


@_op(
    "mymyr::reset", mutates_args=("states", "steps", "counts", "views", "goal_pos", "goal_neg", "count", "at_goal",
                  "visited", "visited_count", "visited_status")
)
def reset(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Tensor,
    counts: Tensor,
    views: Tensor,
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    count: Tensor,
    at_goal: Optional[Tensor],
    mask: Optional[Tensor],
    keep_goals: bool,
    visited: Optional[Tensor] = None,
    visited_count: Optional[Tensor] = None,
    visited_status: Optional[Tensor] = None,
) -> None:
    """``mymyr::reset``: resets the rows (all, or those with ``mask[i]``) in place into their instances ``task_ids``
    (set by the caller first): steps 0, the instance's goal masks unless ``keep_goals``, the cache, ``count`` and
    ``at_goal``. ``env`` is a :func:`register` handle; the tensors are the environment state of the module docstring."""
    reset_impl(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, at_goal, mask, keep_goals,
               visited, visited_count, visited_status)


@reset.register_fake
def _(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, at_goal, mask, keep_goals,
      visited=None, visited_count=None, visited_status=None):
    return None


@_op("mymyr::refresh", mutates_args=("counts", "views", "count", "at_goal"))
def refresh(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    counts: Tensor,
    views: Tensor,
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    count: Tensor,
    at_goal: Optional[Tensor],
) -> None:
    """``mymyr::refresh``: recomputes the cache, ``count`` and ``at_goal`` of states (task ids, goal masks) the
    caller wrote. ``env`` is a :func:`register` handle."""
    refresh_impl(env, states, task_ids, counts, views, goal_pos, goal_neg, count, at_goal)


@refresh.register_fake
def _(env, states, task_ids, counts, views, goal_pos, goal_neg, count, at_goal):
    return None


@_op("mymyr::refresh_sync", mutates_args=("counts", "views", "count", "at_goal"), tags=_unsafe)
def refresh_sync(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    counts: Tensor,
    views: Tensor,
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    count: Tensor,
    at_goal: Optional[Tensor],
) -> None:
    """``mymyr::refresh_sync``: :func:`refresh` for environments whose calls synchronize (the device's general path);
    tagged ``cudagraph_unsafe``."""
    refresh_impl(env, states, task_ids, counts, views, goal_pos, goal_neg, count, at_goal)


@refresh_sync.register_fake
def _(env, states, task_ids, counts, views, goal_pos, goal_neg, count, at_goal):
    return None


@_op(
    "mymyr::step",
    mutates_args=(
        "states", "task_ids", "steps", "draws", "counts", "views", "count", "at_goal", "goal_pos", "goal_neg",
        "visited", "visited_count", "visited_status", "revisited"
    ),
)
def step(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Tensor,
    draws: Tensor,
    counts: Tensor,
    views: Tensor,
    count: Tensor,
    at_goal: Optional[Tensor],
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
    action_schema: Optional[Tensor] = None,
    action_binding: Optional[Tensor] = None,
    visited: Optional[Tensor] = None,
    visited_count: Optional[Tensor] = None,
    visited_status: Optional[Tensor] = None,
    revisited: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``mymyr::step``: one step of every environment in place, ``action`` [N] int64 (None: the random policy);
    label mode instead takes ``action_schema`` int32 [N] and ``action_binding`` int32 [N, L] with action=None.
    Autoresetting rows restart in ``next_task_ids[i]`` where given. Returns (reward, terminated, truncated, goal,
    invalid, schema, binding, final_states); ``final_states`` is [N, row_words], the reached states before the
    autoreset, with ``final`` and [0, row_words] without."""
    return step_impl(
        env, states, task_ids, steps, draws, counts, views, count, at_goal, goal_pos, goal_neg, action, next_task_ids,
        first_env, final, action_schema, action_binding, visited, visited_count, visited_status, revisited,
    )  # fmt: skip


@step.register_fake
def _(
    env, states, task_ids, steps, draws, counts, views, count, at_goal, goal_pos, goal_neg, action, next_task_ids,
    first_env, final, action_schema=None, action_binding=None, visited=None, visited_count=None,
    visited_status=None, revisited=None,
):  # fmt: skip
    return _step_outputs(lookup(env).label_width, states, final)


@_op(
    "mymyr::step_sync",
    mutates_args=(
        "states", "task_ids", "steps", "draws", "counts", "views", "count", "at_goal", "goal_pos", "goal_neg",
        "visited", "visited_count", "visited_status", "revisited"
    ),
    tags=_unsafe,
)
def step_sync(
    env: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    steps: Tensor,
    draws: Tensor,
    counts: Tensor,
    views: Tensor,
    count: Tensor,
    at_goal: Optional[Tensor],
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
    action_schema: Optional[Tensor] = None,
    action_binding: Optional[Tensor] = None,
    visited: Optional[Tensor] = None,
    visited_count: Optional[Tensor] = None,
    visited_status: Optional[Tensor] = None,
    revisited: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``mymyr::step_sync``: :func:`step` for environments whose calls synchronize (the device's general path);
    tagged ``cudagraph_unsafe``."""
    return step_impl(
        env, states, task_ids, steps, draws, counts, views, count, at_goal, goal_pos, goal_neg, action, next_task_ids,
        first_env, final, action_schema, action_binding, visited, visited_count, visited_status, revisited,
    )  # fmt: skip


@step_sync.register_fake
def _(
    env, states, task_ids, steps, draws, counts, views, count, at_goal, goal_pos, goal_neg, action, next_task_ids,
    first_env, final, action_schema=None, action_binding=None, visited=None, visited_count=None,
    visited_status=None, revisited=None,
):  # fmt: skip
    return _step_outputs(lookup(env).label_width, states, final)


@_op("mymyr::random_actions", mutates_args=())
def random_actions(env: int, draws: Tensor, count: Tensor, first_env: int, max_actions: Optional[int]) -> Tensor:
    """``mymyr::random_actions``: the random policy's next choices [N] int64 at the draw counters (no step);
    ``max_actions`` (at least 1) caps the choice at that many actions, None does not."""
    return random_actions_impl(env, draws, count, first_env, max_actions)


@random_actions.register_fake
def _(env, draws, count, first_env, max_actions):
    return draws.new_empty(draws.shape[0], dtype=torch.int64)


# ------------------------------------------------------------------------------------------------ expansion


def _expansion_outputs(t: Any, states: Tensor, capacity: Any, derived: bool, rows: bool,
                       hashes: bool, noops: bool, drop_noops: bool) -> tuple[Tensor, ...]:
    n, rw = states.shape
    return (
        states.new_zeros((capacity, rw) if rows else (0, 0)),
        states.new_full((capacity,), -1, dtype=torch.int32),
        states.new_full((capacity,), -1, dtype=torch.int32),
        states.new_full((capacity, t.label_width), -1, dtype=torch.int32),
        states.new_zeros((capacity,) if rows else (0,), dtype=torch.bool),
        states.new_empty(n + 1, dtype=torch.int32),
        states.new_zeros((capacity, t.derived_words) if derived else (0, 0)),
        states.new_zeros((capacity,) if hashes else (0,)),
        states.new_zeros((capacity,) if noops or drop_noops else (0,), dtype=torch.bool),
        states.new_full((capacity,) if drop_noops else (0,), -1, dtype=torch.int32),
        states.new_zeros(1, dtype=torch.int32),
    )


@_op("mymyr::expand_flat", mutates_args=(), tags=_unsafe)
def expand_flat(
    table: int, states: Tensor, task_ids: Optional[Tensor], capacity: Optional[int] = None,
    witness_pruning: bool = False, derived: bool = False, rows: bool = True,
    hashes: bool = False, noops: bool = False, drop_noops: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Flat CSR expansion: succ, parent, schema, binding, goal, offsets, derived, succ_hash, noop, index, status.

    Absent fields are empty tensors. Capacity bounds the count before dropping; status bit 0 reports overflow,
    bit 1 invalid input or insufficient row width. A CUDA capacity call has no host wait and captures after warm-up.
    Label-only calls (rows=False) build requested digests without storing successor rows.
    """
    t = lookup(table)
    if derived and not rows:
        raise ValueError("mymyr: derived outputs require successor rows")
    dynamic_labels = capacity is None and not rows
    if dynamic_labels:
        offsets = states.new_empty(states.shape[0] + 1, dtype=torch.int32)
        counted = mymyr.rl.expand_into(t, states, task_ids, offsets=offsets,
                                      witness_pruning=witness_pruning, stream=_stream(states))
        capacity = counted["generated"]
    if capacity is None:
        x = mymyr.rl.expand(t, states, task_ids, words=states.shape[1] - t.numeric_words, goal=rows,
                            derived=derived, hashes=hashes, noops=noops, drop_noops=drop_noops,
                            witness_pruning=witness_pruning, stream=_stream(states))
        empty = states.new_empty(0)
        return (x.succ if rows else states.new_empty((0, 0)), x.parent, x.schema, x.binding,
                x.goal if rows else empty.bool(), x.offsets, x.derived if derived else states.new_empty((0, 0)),
                x.succ_hash if hashes else empty, x.noop if noops or drop_noops else empty.bool(),
                x.index if drop_noops else empty.int(), states.new_tensor([int(x.overflow)], dtype=torch.int32))
    out = _expansion_outputs(t, states, capacity, derived, rows, hashes, noops, drop_noops)
    succ, parent, schema, binding, goal, offsets, der, digest, noop, index, status = out
    result = mymyr.rl.expand_into(
        t, states, task_ids, succ=succ if rows else None, parent=parent, schema=schema, binding=binding,
        goal=goal if rows else None, offsets=offsets, derived=der if derived else None,
        succ_hash=digest if hashes else None, noop=noop if noops or drop_noops else None,
        index=index if drop_noops else None, drop_noops=drop_noops,
        status=status if states.is_cuda and not dynamic_labels else None, stream=_stream(states), witness_pruning=witness_pruning,
    )
    if not states.is_cuda or dynamic_labels:
        status.fill_(int(result["overflow"]) | (2 if result["words_needed"] > states.shape[1] - t.numeric_words else 0))
    elif rows:
        # the device's capturable form evaluates goal and axioms on the zero rows past the total: pad them as the
        # host does, without a host read of the total
        live = torch.arange(capacity, device=states.device) < offsets[-1]
        goal.logical_and_(live)
        if derived:
            der.masked_fill_(~live[:, None], 0)
    if dynamic_labels:
        count = result["total"]
        return tuple(x if i in (5, 10) else x[:count] for i, x in enumerate(out))
    return out


@expand_flat.register_fake
def _(table, states, task_ids, capacity=None, witness_pruning=False, derived=False, rows=True,
      hashes=False, noops=False, drop_noops=False):
    m = torch.library.get_ctx().new_dynamic_size() if capacity is None else capacity
    return _expansion_outputs(lookup(table), states, m, derived, rows, hashes, noops, drop_noops)


def _pad_expansion(flat: tuple[Tensor, ...], states: Tensor, k: int, rows: bool, derived: bool,
                   hashes: bool, noops: bool, drop_noops: bool) -> tuple[Tensor, ...]:
    succ, parent, schema, binding, goal, offsets, der, digest, noop, index, status = flat
    n = states.shape[0]
    pos = offsets[:-1, None].to(torch.int64) + torch.arange(k, device=states.device)
    counts = offsets[1:] - offsets[:-1]
    valid = (torch.arange(k, device=states.device) < counts[:, None]) & (pos < schema.shape[0])
    at = pos.clamp(0, max(schema.shape[0] - 1, 0))
    # A sentinel makes gathering safe even for capacity zero.
    def gather(x, fill=0):
        extra = x.new_full((1, *x.shape[1:]), fill)
        padded = torch.cat((x, extra), dim=0)
        selected = padded[torch.where(valid, at, x.shape[0])]
        mask = valid.reshape((*valid.shape, *((1,) * (x.ndim - 1))))
        return torch.where(mask, selected, torch.full_like(selected, fill))
    status = status | torch.any(counts > k).to(torch.int32).reshape(1)
    return (gather(succ) if rows else succ, gather(parent, -1), gather(schema, -1), gather(binding, -1),
            gather(goal) if rows else goal, offsets, gather(der) if derived else der,
            gather(digest) if hashes else digest, gather(noop) if noops or drop_noops else noop,
            gather(index, -1) if drop_noops else index, status)


@_op("mymyr::expand", mutates_args=(), tags=_unsafe)
def expand(
    table: int, states: Tensor, task_ids: Optional[Tensor], k: int, witness_pruning: bool = False,
    derived: bool = False, rows: bool = True, hashes: bool = False, noops: bool = False,
    drop_noops: bool = False, capacity: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Padded [N, k] expansion with the same eleven fields as expand_flat; offsets remain CSR counts.

    Capacity bounds successors before dropping and makes CUDA calls capturable after warm-up. Parent -1 marks
    padding; hashes/noop pad with zero/False, and canonical indices with -1. Status also reports counts above k.
    """
    if k <= 0:
        raise ValueError("mymyr: expand needs k > 0")
    flat = expand_flat(table, states, task_ids, capacity, witness_pruning, derived, rows, hashes, noops, drop_noops)
    return _pad_expansion(flat, states, k, rows, derived, hashes, noops, drop_noops)


@expand.register_fake
def _(table, states, task_ids, k, witness_pruning=False, derived=False, rows=True,
      hashes=False, noops=False, drop_noops=False, capacity=None):
    n, rw = states.shape
    t = lookup(table)
    return (
        states.new_empty((n, k, rw) if rows else (0, 0)),
        states.new_empty((n, k), dtype=torch.int32),
        states.new_empty((n, k), dtype=torch.int32),
        states.new_empty((n, k, t.label_width), dtype=torch.int32),
        states.new_empty((n, k) if rows else (0,), dtype=torch.bool),
        states.new_empty(n + 1, dtype=torch.int32),
        states.new_empty((n, k, t.derived_words) if derived else (0, 0)),
        states.new_empty((n, k) if hashes else (0,)),
        states.new_empty((n, k) if noops or drop_noops else (0,), dtype=torch.bool),
        states.new_empty((n, k) if drop_noops else (0,), dtype=torch.int32),
        states.new_empty(1, dtype=torch.int32),
    )


# ------------------------------------------------------------------------------------------------ encoding


@_op("mymyr::bounded_actions", mutates_args=())
def bounded_actions(
    encoder: int, states: Tensor, task_ids: Optional[Tensor], entities: int, successors: Optional[int] = None,
    drop_noops: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The expansion rows that the action entities of an encoding of ``states`` [N, W] (device rows) by the
    :func:`register`-ed ``FlatRelationEncoder`` ``encoder`` stand for, labels only: ``labels`` int32 [2, M] (parent
    and schema; [3, M] with index as the third row with ``drop_noops``), binding [M, label_width], offsets [N + 1],
    succ_hash (int64 [M]), noop (bool [M]) and the int32 status [1].

    The row arrays have the rows that ``entities`` action entities can stand for (the entity capacity times the
    largest number of schemas that share an action relation) and are zero past the kept rows (unlike
    ``expand_flat``'s padding, so that the rows past them index nothing out of range). ``successors`` bounds
    the successors before ``drop_noops`` drops the no-op rows (default: the rows of the arrays). Status bit 0 reports
    more successors than that or more kept rows than the arrays have, bit 1 an invalid row; pass the arrays and the
    status to ``encode_flat`` (``action_*``), whose batch then reports the overflow. Nothing waits for the device and
    the op allocates only its outputs and scratch that a captured call keeps, so it captures into CUDA graphs (warm it
    on the capture stream first)."""
    from mymyr.encoding._device_actions import bounded_actions as bounded

    x = bounded(lookup(encoder), states, task_ids, entities, successors=successors, hashes=True, noops=True,
                drop_noops=drop_noops)
    return x.labels, x.binding, x.offsets, x.succ_hash, x.noop, x.status


@bounded_actions.register_fake
def _(encoder, states, task_ids, entities, successors=None, drop_noops=False):
    from mymyr.encoding._device_actions import action_capacity, table_of

    enc = lookup(encoder)
    rows = action_capacity(enc, entities)
    i32 = dict(dtype=torch.int32)
    return (states.new_empty((3 if drop_noops else 2, rows), **i32),
            states.new_empty((rows, table_of(enc).label_width), **i32), states.new_empty(states.shape[0] + 1, **i32),
            states.new_empty(rows, dtype=torch.int64), states.new_empty(rows, dtype=torch.bool),
            states.new_empty(1, **i32))


def _encode_flat_outputs(
    encoder: int, states: Tensor, graphs: int, entities: int, args: int, targets: int, history: int, numeric: int,
    static_instances: int, static_args: int,
) -> list[Tensor]:
    layout = _device.layout_of(lookup(int(encoder)))
    n = states.shape[0] if states.ndim == 2 else 1
    if n > graphs:
        raise ValueError("mymyr: encoding: the batch has more graphs than the output capacity")
    specs = layout.specs(torch, n, entities, args, targets, history, numeric=numeric,
                         static_instances=static_instances, static_args=static_args)
    return [torch.empty(specs[name][0], dtype=specs[name][1], device=states.device) for name in _device._OUTPUTS]


@_op("mymyr::encode_flat", mutates_args=())
def encode_flat(
    encoder: int,
    states: Tensor,
    task_ids: Optional[Tensor],
    derived: Optional[Tensor],
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    goal_derived_pos: Optional[Tensor],
    goal_derived_neg: Optional[Tensor],
    subgoal_pos: Optional[Tensor],
    subgoal_neg: Optional[Tensor],
    subgoal_derived_pos: Optional[Tensor],
    subgoal_derived_neg: Optional[Tensor],
    history_pos: Optional[Tensor],
    history_neg: Optional[Tensor],
    history_dt: Optional[Tensor],
    history_count: Optional[Tensor],
    history_derived_pos: Optional[Tensor],
    history_derived_neg: Optional[Tensor],
    graphs: int,
    entities: int,
    args: int,
    targets: int,
    history: int,
    numeric: int = 0,
    static_instances: int = 0,
    static_args: int = 0,
    action_schema: Optional[Tensor] = None,
    action_binding: Optional[Tensor] = None,
    action_offsets: Optional[Tensor] = None,
    action_parent: Optional[Tensor] = None,
    node_offsets: Optional[Tensor] = None,
    edge_offsets: Optional[Tensor] = None,
    edge_src: Optional[Tensor] = None,
    edge_dst: Optional[Tensor] = None,
    edge_schema: Optional[Tensor] = None,
    edge_binding: Optional[Tensor] = None,
    action_succ: Optional[Tensor] = None,
    expand_actions: bool = False,
    action_index: Optional[Tensor] = None,
    action_status: Optional[Tensor] = None,
) -> list[Tensor]:
    """``mymyr::encode_flat``: ``FlatRelationEncoder.encode`` of the device rows ``states`` [N, W] (int64 words; the
    other arguments as in ``encode``: ``task_ids`` int32 [N], ``derived`` [N, D], the goal masks [N or 1, W] and
    [N or 1, D], the subgoal layers [N or 1, L, W] (and [N or 1, L, D]), the history masks [N, H, W] (and [N, H, D]),
    ``history_dt`` int32 [N, H] and ``history_count`` int32 [N] or None) with the capacities ``graphs``, ``entities``,
    ``args``, ``targets``, ``history`` and ``numeric`` (0 for an absent lane); shared statics also take
    ``static_instances`` and ``static_args``, and ``static_sizes`` holds their device counts [instances, tuples, args].
    ``encoder`` is a :func:`register` handle of the encoder. Returns the 29 arrays of ``DeviceFlatBatch.array_names``
    (arrays that the schema does not
    store are empty) with ``sizes`` last; ``DeviceFlatBatch.from_arrays`` wraps them. A batch exceeding ``graphs``
    raises before launch. Nothing waits for the device, and a batch that exceeds its array capacities is all zero with
    the overflow flags in ``sizes``. Warm it on the capture stream to acquire its context before capturing a CUDA
    graph, and keep the encoder registered and input owners alive while the graph can replay. Scratch belongs to
    torch's allocator and needs no reservation. Horizon schemas take node/edge arrays, or parent ``states`` with
    ``action_succ`` and expansion labels to build depth-one transitions; ``derived`` is per node for a horizon schema.
    ``expand_actions=True`` expands observation actions into buffers bounded by the entity capacity and the schema
    multiplicity, with device overflow status; it requires asynchronous expansion. It takes no explicit action or
    horizon arrays. Explicit expansion rows (``action_schema``, ``action_binding``, ``action_offsets``, optionally
    ``action_parent``) may carry ``action_index``, each row's position in its state's full canonical successor order
    (``mymyr::expand_flat``'s ``index`` with ``drop_noops``: the ``entity_ref`` of its action entity), and
    ``action_status``, the expansion's int32 or int64 device status word [1] (bit 0 overflow, bit 1 invalid input),
    which the batch's overflow flags then report (capacity, invalid row): such a batch is all zero. The arguments after ``numeric`` are passed by
    keyword."""
    goals = (goal_pos, goal_neg, goal_derived_pos, goal_derived_neg)
    given = [g for g in goals if g is not None]
    if len(given) not in (0, 2, 4) or any(goals[k] is None for k in range(len(given))):
        raise ValueError("mymyr: encoding: goals are (goal_pos, goal_neg) or all four masks")
    enc = lookup(encoder)
    caps = _device._capacities(
        {"entities": entities, "graphs": graphs, "args": args, "targets": targets, "history": history, "numeric": numeric,
         "static_instances": static_instances, "static_args": static_args},
        _device.layout_of(enc),
    )
    subgoals = _layer_tuple(
        "subgoals", (subgoal_pos, subgoal_neg), (subgoal_derived_pos, subgoal_derived_neg), ()
    )
    steps = _layer_tuple(
        "history", (history_pos, history_neg), (history_derived_pos, history_derived_neg), (history_dt, history_count)
    )
    actions = None
    if expand_actions:
        if any(a is not None for a in (action_schema, action_binding, action_offsets, action_parent, action_succ,
                                      action_index, action_status, node_offsets, edge_offsets, edge_src, edge_dst,
                                      edge_schema, edge_binding)):
            raise ValueError("mymyr: encoding: expand_actions takes no explicit action or horizon arrays")
        from mymyr.encoding._device_actions import bounded_actions
        actions = bounded_actions(enc, states, task_ids, entities)
    if any(a is not None for a in (action_schema, action_binding, action_offsets, action_parent, action_succ,
                                   action_index, action_status)):
        if any(a is None for a in (action_schema, action_binding, action_offsets)):
            raise ValueError("mymyr: encoding: actions need schema, binding and offsets")
        from types import SimpleNamespace
        actions = SimpleNamespace(schema=action_schema, binding=action_binding, offsets=action_offsets,
                                  parent=action_parent, succ=action_succ, index=action_index, status=action_status)
    horizon = None
    if node_offsets is not None:
        if steps is not None or actions is not None:
            raise ValueError("mymyr: encoding: horizons take edge labels and no history or expansion rows")
        from mymyr.encoding import HorizonView
        horizon = HorizonView(states, node_offsets, edge_offsets, edge_src, edge_dst, edge_schema,
                              edge_binding, task_ids)
    elif enc.options["horizon_mode"] is not None:
        if actions is None or action_succ is None or steps is not None:
            raise ValueError("mymyr: encoding: a horizon op needs node offsets and edges, or successor rows and "
                             "expansion labels")
        from mymyr.encoding._device_horizon import build_horizon
        horizon = build_horizon(enc, states, task_ids, actions, merge=False)
        actions = None
    arrays = _device.encode_arrays(
        enc, states, task_ids, derived, tuple(given) or None, caps, subgoals=subgoals, history=steps,
        actions=actions, horizon=horizon
    )
    return [arrays[name] for name in _device.DeviceFlatBatch.array_names]


def _layer_tuple(name: str, masks: tuple, derived: tuple, rest: tuple) -> Optional[tuple]:
    """The ``subgoals`` or ``history`` argument of ``encode`` from the operands of the op: the masks, then for the
    history (dt, count) and the derived masks (None when the pair is absent)."""
    if all(m is None for m in (*masks, *derived, *rest)):
        return None
    if (masks[0] is None) != (masks[1] is None) or (derived[0] is None) != (derived[1] is None):
        raise ValueError(f"mymyr: encoding: the {name} masks come in pairs (positive and negative)")
    if masks[0] is None:
        raise ValueError(f"mymyr: encoding: the {name} need their masks over the fluent slots")
    if name == "history":
        return (*masks, *rest, *(derived if derived[0] is not None else ()))
    return (*masks, *(derived if derived[0] is not None else ()))


@encode_flat.register_fake
def _(
    encoder, states, task_ids, derived, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg, subgoal_pos, subgoal_neg,
    subgoal_derived_pos, subgoal_derived_neg, history_pos, history_neg, history_dt, history_count,
    history_derived_pos, history_derived_neg, graphs, entities, args, targets, history, numeric=0,
    static_instances=0, static_args=0,
    action_schema=None, action_binding=None, action_offsets=None, action_parent=None,
    node_offsets=None, edge_offsets=None, edge_src=None, edge_dst=None, edge_schema=None, edge_binding=None,
    action_succ=None,
    expand_actions=False,
    action_index=None,
    action_status=None,
):  # fmt: skip
    if node_offsets is not None:
        states = states[:node_offsets.shape[0] - 1]
    return _encode_flat_outputs(encoder, states, graphs, entities, args, targets, history, numeric, static_instances,
                                static_args)


@torch.library.custom_op("mymyr::literal_slots", mutates_args=())
def literal_slots(encoder: int, predicates: Tensor, objects: Tensor,
                  task_ids: Optional[Tensor] = None) -> list[Tensor]:
    """CUDA literal lookup on a registered encoder: kinds uint8 [N], slots int32 [N], status int64 scalar.

    Predicate ids are table domain predicate ids; objects [N, k] are instance-local with -1 padding. task_ids is
    int32 [N] or None over one instance. Status value 4 reports invalid literals. Warm on the capture stream and
    retain the registered encoder and buffers through the last replay; no host synchronization occurs.
    """
    from mymyr.encoding._device_literals import slots
    return list(slots(lookup(encoder), predicates, objects, task_ids))


@literal_slots.register_fake
def _(encoder, predicates, objects, task_ids=None):
    return [predicates.new_empty(predicates.shape, dtype=torch.uint8),
            predicates.new_empty(predicates.shape, dtype=torch.int32),
            predicates.new_empty((), dtype=torch.int64)]


@torch.library.custom_op("mymyr::literal_masks", mutates_args=())
def literal_masks(encoder: int, predicates: Tensor, objects: Tensor, positive: Tensor, rows: Tensor,
                  num_rows: int, task_ids: Optional[Tensor] = None, layers: Optional[Tensor] = None,
                  num_layers: int = 1) -> list[Tensor]:
    """CUDA literal masks: pos, neg, derived_pos, derived_neg, unsatisfiable, status.

    The four masks are int64 [num_rows, W/D], or [num_rows, num_layers, W/D] with layers. Derived masks have width
    zero without derived slots. Signs are bool [N]; ids, rows and layers are contiguous int32 on the same device.
    task_ids is per output row. Fixed false literals set unsatisfiable (bool); invalid entries are skipped and set
    status (int64 scalar) to 4. Outputs have fixed shapes and allocate on torch's current stream. Warm on the capture
    stream and retain the registered encoder and buffers through the last replay.
    """
    from mymyr.encoding._device_literals import masks
    return masks(lookup(encoder), predicates, objects, positive, rows, num_rows, task_ids, layers, num_layers)


@literal_masks.register_fake
def _(encoder, predicates, objects, positive, rows, num_rows, task_ids=None, layers=None, num_layers=1):
    table = mymyr.rl.as_table(lookup(int(encoder)).table)
    shape = (num_rows,) if layers is None else (num_rows, num_layers)
    return [*(predicates.new_empty((*shape, words), dtype=torch.int64)
              for words in (table.words, table.words, table.derived_words, table.derived_words)),
            predicates.new_empty(shape, dtype=torch.bool), predicates.new_empty((), dtype=torch.int64)]


@torch.library.custom_op("mymyr::match_first", mutates_args=())
def match_first(templates: int, rows: Tensor, task_ids: Optional[Tensor] = None, first: bool = False,
                derived: Optional[Tensor] = None, capacity: Optional[int] = None,
                max_steps: Optional[int] = 100_000_000) -> list[Tensor]:
    """Lifted matching on registered host or uploaded templates; found, binding, template, count, status.

    Shapes and canonical binding order are those of mymyr.rl.match_first. Upload before capture and retain the
    registered templates through the last replay. Output capacity is static; no host read of device counts occurs.
    """
    from mymyr.rl._match import _match
    return list(_match(rows, task_ids, lookup(templates), first, derived, capacity, max_steps=max_steps))


@match_first.register_fake
def _(templates, rows, task_ids=None, first=False, derived=None, capacity=None, max_steps=100_000_000):
    t = lookup(int(templates))
    width = (1 if first else t.capacity) if capacity is None else capacity
    shape = (rows.shape[0],) if first else (rows.shape[0], width)
    return [rows.new_empty(shape, dtype=torch.bool),
            rows.new_empty((*shape, t.max_vars), dtype=torch.int32),
            rows.new_empty(shape, dtype=torch.int32), rows.new_empty((rows.shape[0],), dtype=torch.int32),
            rows.new_empty((), dtype=torch.int64)]
