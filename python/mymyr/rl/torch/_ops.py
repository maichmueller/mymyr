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
None over a table of one), ``steps``, ``draws``, the fast path's cache (``counts``, ``views``) and optional per-env goal
masks ``goal_pos`` / ``goal_neg`` [N, words] int64 (the step's goal test is then the mask test).

- ``mymyr::reset(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask, keep_goals)``: resets
  rows (all, or those with mask[i]) in place into their instances (task_ids, set by the caller first).
- ``mymyr::refresh(env, states, task_ids, counts, views, count)``: recomputes the cache and counts of states the
  caller wrote.
- ``mymyr::step(env, states, task_ids, steps, draws, counts, views, count, goal_pos, goal_neg, action,
  next_task_ids, first_env, final)``: one step of every environment in place (autoresetting rows restart in
  next_task_ids[i] where given); returns (reward, terminated, truncated, goal, invalid, schema, binding, final_states).
- ``mymyr::refresh_sync`` / ``mymyr::step_sync``: ``refresh`` / ``step`` of environments whose calls synchronize (the
  device's general path), tagged ``cudagraph_unsafe``.
- ``mymyr::random_actions(env, draws, count, first_env, max_actions)``: the random policy's next choices [N] int64
  (the counter-based RNG at the draw counters; no step).
- ``mymyr::expand(table, states, task_ids, K, witness)``: the padded expansion [N, K] (succ, schema, binding, goal,
  mask, count).
- ``mymyr::expand_flat(table, states, task_ids, capacity, witness)``: the flat expansion (succ, parent, schema,
  binding, goal, offsets). ``capacity=None`` sizes the outputs by the data: the row count M is an unbacked symbolic
  size under ``torch.compile``; a capacity gives static shapes (rows past it are dropped, rows past the total are
  padding: succ 0, parent / schema / binding -1, goal False; offsets keep the true counts).
"""

import itertools
import threading
from typing import Any, Optional

import torch
from torch import Tensor

import mymyr.rl

__all__ = [
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

_lock = threading.Lock()
_objects: dict[int, Any] = {}
_by_id: dict[int, int] = {}
_ids = itertools.count(1)


def register(obj: Any) -> int:
    """The op handle of a TaskTable, Task, TaskHandle or native Env (the same handle for the same object). The
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


def lookup(handle: int) -> Any:
    try:
        return _objects[handle]
    except KeyError:
        raise ValueError(f"mymyr: no object registered under handle {handle} (see mymyr.rl.torch.register)") from None


def _stream(t: Tensor) -> Optional[int]:
    return torch.cuda.current_stream(t.device).cuda_stream if t.device.type == "cuda" else None


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
    mask: Optional[Tensor],
    keep_goals: bool,
) -> None:
    """The body of ``mymyr::reset``."""
    e, info = _env_info(env)
    n = states.shape[0]
    p = _batch_ptrs(info, states, task_ids, steps, None, counts, views, goal_pos, goal_neg)
    del p[3]  # draws
    p += [_ptr(count, "count", torch.int32, (n,), info[0]), _ptr(mask, "mask", torch.bool, (n,), info[0])]
    e._reset_ptrs(n, p, keep_goals, _stream(states))


def refresh_impl(env: int, states: Tensor, task_ids: Optional[Tensor], counts: Tensor, views: Tensor, count: Tensor) -> None:
    """The body of ``mymyr::refresh``."""
    e, info = _env_info(env)
    n = states.shape[0]
    p = _batch_ptrs(info, states, task_ids, None, None, counts, views, None, None)
    p = [p[0], p[1], p[4], p[5], _ptr(count, "count", torch.int32, (n,), info[0])]
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
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
    out: Optional[tuple[Tensor, ...]] = None,
) -> tuple[Tensor, ...]:
    """The body of ``mymyr::step``; ``out``: the 8 outputs' destinations (contiguous, of _step_outputs' shapes and
    dtypes; CapturedRollout's slices of its stacked outputs), else new tensors."""
    e, info = _env_info(env)
    dev, L = info[0], info[5]
    n = states.shape[0]
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
    ]
    e._step_ptrs(n, p, L, first_env, _stream(states))
    return out


def random_actions_impl(env: int, draws: Tensor, count: Tensor, first_env: int, max_actions: int) -> Tensor:
    """The body of ``mymyr::random_actions``."""
    e, info = _env_info(env)
    dev = info[0]
    n = draws.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=draws.device)
    p = [_ptr(draws, "draws", torch.int64, (n,), dev), _ptr(count, "count", torch.int32, (n,), dev), out.data_ptr()]
    e._random_actions_ptrs(n, p, first_env, max_actions, _stream(draws))
    return out


@torch.library.custom_op(
    "mymyr::reset", mutates_args=("states", "steps", "counts", "views", "goal_pos", "goal_neg", "count")
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
    mask: Optional[Tensor],
    keep_goals: bool,
) -> None:
    reset_impl(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask, keep_goals)


@reset.register_fake
def _(env, states, task_ids, steps, counts, views, goal_pos, goal_neg, count, mask, keep_goals):
    return None


@torch.library.custom_op("mymyr::refresh", mutates_args=("counts", "views", "count"))
def refresh(env: int, states: Tensor, task_ids: Optional[Tensor], counts: Tensor, views: Tensor, count: Tensor) -> None:
    refresh_impl(env, states, task_ids, counts, views, count)


@refresh.register_fake
def _(env, states, task_ids, counts, views, count):
    return None


@torch.library.custom_op("mymyr::refresh_sync", mutates_args=("counts", "views", "count"), tags=_unsafe)
def refresh_sync(env: int, states: Tensor, task_ids: Optional[Tensor], counts: Tensor, views: Tensor, count: Tensor) -> None:
    refresh_impl(env, states, task_ids, counts, views, count)


@refresh_sync.register_fake
def _(env, states, task_ids, counts, views, count):
    return None


@torch.library.custom_op(
    "mymyr::step",
    mutates_args=("states", "task_ids", "steps", "draws", "counts", "views", "count", "goal_pos", "goal_neg"),
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
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return step_impl(
        env, states, task_ids, steps, draws, counts, views, count, goal_pos, goal_neg, action, next_task_ids,
        first_env, final,
    )  # fmt: skip


@step.register_fake
def _(env, states, task_ids, steps, draws, counts, views, count, goal_pos, goal_neg, action, next_task_ids, first_env, final):
    return _step_outputs(lookup(env).label_width, states, final)


@torch.library.custom_op(
    "mymyr::step_sync",
    mutates_args=("states", "task_ids", "steps", "draws", "counts", "views", "count", "goal_pos", "goal_neg"),
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
    goal_pos: Optional[Tensor],
    goal_neg: Optional[Tensor],
    action: Optional[Tensor],
    next_task_ids: Optional[Tensor],
    first_env: int,
    final: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return step_impl(
        env, states, task_ids, steps, draws, counts, views, count, goal_pos, goal_neg, action, next_task_ids,
        first_env, final,
    )  # fmt: skip


@step_sync.register_fake
def _(env, states, task_ids, steps, draws, counts, views, count, goal_pos, goal_neg, action, next_task_ids, first_env, final):
    return _step_outputs(lookup(env).label_width, states, final)


@torch.library.custom_op("mymyr::random_actions", mutates_args=())
def random_actions(env: int, draws: Tensor, count: Tensor, first_env: int, max_actions: int) -> Tensor:
    return random_actions_impl(env, draws, count, first_env, max_actions)


@random_actions.register_fake
def _(env, draws, count, first_env, max_actions):
    return draws.new_empty(draws.shape[0], dtype=torch.int64)


# ------------------------------------------------------------------------------------------------ expansion


@torch.library.custom_op("mymyr::expand", mutates_args=(), tags=_unsafe)
def expand(
    table: int, states: Tensor, task_ids: Optional[Tensor], K: int, witness: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    t = lookup(table)
    if K <= 0:
        raise ValueError("mymyr: expand needs K > 0 (the padded width is a static shape)")
    words = states.shape[1] - t.numeric_words
    x = mymyr.rl.expand(t, states, task_ids, words=words, K=K, goal=True, witness=witness, stream=_stream(states))
    p = x.padded
    return p.succ, p.schema, p.binding, p.goal, p.mask, p.count


@expand.register_fake
def _(table, states, task_ids, K, witness=False):
    t = lookup(table)
    n, rw = states.shape
    return (
        states.new_empty((n, K, rw)),
        states.new_empty((n, K), dtype=torch.int32),
        states.new_empty((n, K, t.label_width), dtype=torch.int32),
        states.new_empty((n, K), dtype=torch.bool),
        states.new_empty((n, K), dtype=torch.bool),
        states.new_empty(n, dtype=torch.int32),
    )


@torch.library.custom_op("mymyr::expand_flat", mutates_args=(), tags=_unsafe)
def expand_flat(
    table: int, states: Tensor, task_ids: Optional[Tensor], capacity: Optional[int] = None, witness: bool = False
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    t = lookup(table)
    n, rw = states.shape
    words = rw - t.numeric_words
    if capacity is None:
        x = mymyr.rl.expand(t, states, task_ids, words=words, goal=True, witness=witness, stream=_stream(states))
        return x.succ, x.parent, x.schema, x.binding, x.goal, x.offsets
    out = _flat_outputs(t, states, capacity)
    succ, parent, schema, binding, goal, offsets = out
    mymyr.rl.expand_into(
        t, states, task_ids, succ=succ, parent=parent, schema=schema, binding=binding, goal=goal, offsets=offsets,
        witness=witness, stream=_stream(states),
    )  # fmt: skip
    return out


def _flat_outputs(t: Any, states: Tensor, m: Any) -> tuple[Tensor, ...]:
    """The flat outputs; rows past the total stay padding (succ 0, parent / schema / binding -1, goal False)."""
    n, rw = states.shape
    return (
        states.new_zeros((m, rw)),
        states.new_full((m,), -1, dtype=torch.int32),
        states.new_full((m,), -1, dtype=torch.int32),
        states.new_full((m, t.label_width), -1, dtype=torch.int32),
        states.new_zeros(m, dtype=torch.bool),
        states.new_empty(n + 1, dtype=torch.int32),
    )


@expand_flat.register_fake
def _(table, states, task_ids, capacity=None, witness=False):
    m = torch.library.get_ctx().new_dynamic_size() if capacity is None else capacity
    return _flat_outputs(lookup(table), states, m)
