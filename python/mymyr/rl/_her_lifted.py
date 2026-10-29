"""Lifted hindsight segments over contiguous trajectory windows."""

import operator
from typing import Any, NamedTuple

import numpy as np

from mymyr._core import _rl_ops
from ._match import DeviceGoalTemplates
from ._ops import _arrays, _framework, tensor_view


class HerLiftedBatch(NamedTuple):
    """Padded hindsight segments on the window's device.

    segment int32, reward float32 and achieved bool are [T,N]; segment is a slot or -1 outside retained segments.
    seg_start/seg_end (inclusive), seg_valid and seg_template (global packed id) are [N,S]; seg_binding is [N,S,V].
    Fluent goal masks are [N,S,W], derived masks [N,S,DW] (zero width without derived inputs), all 64-bit words.
    Padding has metadata/binding -1, masks zero, valid False. count [N] counts retained segments. status int64 scalar
    has bits 0 segment overflow, 1 device variable limit, 2 invalid instance, 3 matcher search budget exhausted.
    """

    segment: Any
    reward: Any
    achieved: Any
    seg_start: Any
    seg_end: Any
    seg_valid: Any
    seg_template: Any
    seg_binding: Any
    goal_pos: Any
    goal_neg: Any
    goal_pos_derived: Any
    goal_neg_derived: Any
    count: Any
    status: Any


def _limit(value, name, default):
    if value is None:
        return default
    value = operator.index(value)
    if not 0 <= value < 2**63:
        raise ValueError(f"mymyr: her_relabel_lifted {name} must be None or in [0, 2**63)")
    return value


def _relabel(states, done, task_ids, templates, prev_states, max_segments, max_generated=None,
             step_reward=0.0, goal_reward=1.0, words=None, derived=None, prev_derived=None, status=None,
             max_steps=100_000_000, pool=None):
    fw = _framework(states)
    if fw not in ("numpy", "torch"):
        raise TypeError("mymyr: her_relabel_lifted supports NumPy and torch arrays")
    _arrays(done=done, task_ids=task_ids, prev_states=prev_states, derived=derived, prev_derived=prev_derived, status=status)
    if states.ndim != 3:
        raise ValueError("mymyr: her_relabel_lifted states must have shape [T,N,RW]")
    if prev_states is None or done is None:
        raise ValueError("mymyr: her_relabel_lifted prev_states and done are required")
    T, N, _ = states.shape
    S = operator.index(max_segments)
    if not 0 <= S < 2**31:
        raise ValueError("mymyr: her_relabel_lifted max_segments must be in [0, 2**31)")
    generated = _limit(max_generated, "max_generated", 2**64 - 1)
    budget = _limit(max_steps, "max_steps", 2**64 - 1)
    host = templates.host if isinstance(templates, DeviceGoalTemplates) else templates
    W = host.source.words if words is None else operator.index(words)
    if W < 0 or W >= 2**31:
        raise ValueError("mymyr: her_relabel_lifted words must be in [0, 2**31)")
    D = 0 if derived is None else derived.shape[-1] // (2 if derived.dtype.itemsize == 4 else 1)
    stream, block = None, None
    if fw == "torch":
        import torch

        def empty(shape, dtype):
            return torch.empty(shape, dtype=getattr(torch, dtype), device=states.device)

        if states.is_cuda:
            if not isinstance(templates, DeviceGoalTemplates) or templates.device != states.device:
                raise ValueError("mymyr: upload templates to the states' device with templates.to(device)")
            current = torch.cuda.current_stream(states.device)
            if current.cuda_stream not in templates._streams:
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("mymyr: warm relabelling on the capture stream before capture")
                current.wait_event(templates._ready)
                templates._streams.add(current.cuda_stream)
            templates._block.record_stream(current)
            block, stream = templates._block, current.cuda_stream
            for x in (states, done, task_ids, prev_states, derived, prev_derived, status):
                if x is not None:
                    x.record_stream(current)
        elif isinstance(templates, DeviceGoalTemplates):
            raise ValueError("mymyr: CPU states require host templates")
    else:
        if isinstance(templates, DeviceGoalTemplates):
            raise ValueError("mymyr: NumPy states require host templates")

        def empty(shape, dtype):
            return np.empty(shape, dtype=dtype)

    outputs = HerLiftedBatch(
        empty((T, N), "int32"), empty((T, N), "float32"), empty((T, N), "bool"),
        empty((N, S), "int32"), empty((N, S), "int32"), empty((N, S), "bool"), empty((N, S), "int32"),
        empty((N, S, host.max_vars), "int32"),
        *(empty((N, S, w), "uint64" if fw == "numpy" else "int64") for w in (W, W, D, D)), empty((N,), "int32"),
        empty((), "int64") if status is None else status,
    )
    scratch = (empty((T * N,), "int64"), empty((T * N,), "int64"), empty((T * N,), "int32"),
               empty((T * N,), "bool"), empty((T * N, host.max_vars), "int32"), empty((T * N,), "int32"),
               empty((T * N,), "int32"), empty((N, 2 * (W + D)), "int64"))
    v = tensor_view
    _rl_ops.her_relabel_lifted(v(states), v(done), v(task_ids), host, v(block), v(prev_states),
                              tuple(v(x) for x in outputs), tuple(v(x) for x in scratch), derived=v(derived),
                              prev_derived=v(prev_derived), max_generated=generated, max_steps=budget,
                              step_reward=float(step_reward), goal_reward=float(goal_reward), stream=stream, pool=pool)
    return outputs


def her_relabel_lifted(states: Any, done: Any, task_ids: Any, templates: Any, *, prev_states: Any,
                       max_segments: int, max_generated: int | None = None, step_reward: float = 0.0,
                       goal_reward: float = 1.0, words: int | None = None, derived: Any = None,
                       prev_derived: Any = None, status: Any = None, max_steps: int | None = 100_000_000,
                       pool: Any = None) -> HerLiftedBatch:
    """Relabel reached states [T,N,RW] using lifted templates and starting states prev_states of the same shape.

    done bool [T,N] splits each column into episodes, including partial episodes at window edges. task_ids int32
    [N] or [T,N] (None for a single instance) identifies the transition's instance and stays constant per episode.
    Each episode is scanned backwards: ground the first matching template in a reached state, then stop before
    the first starting state satisfying that goal or repeating a scanned successor. Emit the nonempty suffix and
    continue before the excluded step. Templates carry distinctness and canonical binding order.

    S=max_segments fixes per-environment capacity; retain the first S generated segments and report overflow in
    status bit 0. max_generated caps generation globally in environment, episode, then backward scan order, including
    segments lost to capacity overflow. count counts retained segments. max_generated=0 retains none. The last
    transition has reward goal_reward and achieved=True; others inside have step_reward; outside rewards are zero.
    words defaults to the template source's fluent width and must cover its slots. derived and prev_derived are
    required together when templates use derived literals. Goals drop satisfied static literals.

    max_steps bounds matcher work per reached row and template (None: unlimited). Exhaustion sets status bit 3 and
    skips that template; later templates may match. Outputs then describe only completed searches. Status is cleared
    each call. NumPy and torch CPU/CUDA give identical results. Upload templates and warm the capture stream before
    CUDA graph capture; all scratch sizes are fixed by the input shapes. No device result is read on the host.
    A host pool may partition matching and scans. JAX is not supported.
    """
    return _relabel(states, done, task_ids, templates, prev_states, max_segments, max_generated,
                    step_reward, goal_reward, words, derived, prev_derived, status, max_steps, pool)
