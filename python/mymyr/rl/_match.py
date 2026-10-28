"""Packed lifted queries on host rows and on a torch CUDA stream."""

import operator
from typing import Any, NamedTuple

import numpy as np

from mymyr._core import _rl_ops
from ._ops import _framework


class MatchResult(NamedTuple):
    """Padded matches: found bool [N, C], binding int32 [N, C, V], template int32 [N, C], count int32 [N],
    status int64 scalar. With first=True, the C axis is removed (C=1). Bindings are indexed by variable id;
    unused entries and failed template ids are -1. Count includes matches beyond capacity. Status bits are
    0 truncation, 1 device variable limit, 2 invalid instance, 3 search budget exhausted. Outputs stay on the
    input device.
    """

    found: Any
    binding: Any
    template: Any
    count: Any
    status: Any


class DeviceGoalTemplates:
    """An immutable packed template table and literal index on one torch CUDA device.

    Upload before capture, warm matching on the capture stream, and retain this object through the last replay.
    The first eager match on each stream waits on the upload event; later calls reuse that acquisition.
    Matching records the packed tensor's use on every stream.
    """

    def __init__(self, host, device):
        import torch

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("mymyr: GoalTemplates.to requires a CUDA device")
        if host.max_vars > 64:
            raise ValueError("mymyr: device templates support at most 64 variables")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("mymyr: upload templates before CUDA graph capture")
            self.host = host
            self.device = torch.device("cuda", torch.cuda.current_device())
            self._block = torch.tensor(np.array(host._block(), copy=True), dtype=torch.uint8, device=self.device)
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))
        self._streams = set()
        self.capacity = host.capacity
        self.max_vars = host.max_vars
        self.num_instances = host.num_instances
        self.needs_derived = host.needs_derived
        self.source = host.source

    @property
    def nbytes(self):
        return self._block.numel()

    def to(self, device):
        import torch

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return self if device == self.device else self.host.to(device)


def upload(host, stream=None):
    import torch

    if stream is None:
        stream = torch.cuda.current_stream()
    elif not isinstance(stream, torch.cuda.Stream):
        handle = operator.index(getattr(stream, "cuda_stream", stream))
        stream = torch.cuda.ExternalStream(handle, device=torch.cuda.current_device())
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        return DeviceGoalTemplates(host, stream.device)


def _match(rows, task_ids, templates, first, derived, capacity, status=None, pool=None, max_steps=100_000_000):
    if max_steps is None:
        max_steps = 2**64 - 1
    else:
        max_steps = operator.index(max_steps)
        if not 0 <= max_steps < 2**63:
            raise ValueError("mymyr: match_first max_steps must be None or in [0, 2**63)")
    fw = _framework(rows)
    if fw not in ("numpy", "torch"):
        raise TypeError("mymyr: match_first supports NumPy and torch arrays")
    if rows.ndim != 2:
        raise ValueError("mymyr: match_first rows must have shape [N, RW]")
    host = templates.host if isinstance(templates, DeviceGoalTemplates) else templates
    capacity = (1 if first else host.capacity) if capacity is None else operator.index(capacity)
    if capacity < 0 or capacity >= 2**31 or (first and capacity != 1):
        raise ValueError("mymyr: match_first capacity must be in [0, 2**31), or 1 with first=True")
    shape = (rows.shape[0], capacity)
    block, stream = None, None
    if fw == "torch":
        import torch

        def empty(shape, dtype):
            return torch.empty(shape, dtype=getattr(torch, dtype), device=rows.device)

        if rows.is_cuda:
            if not isinstance(templates, DeviceGoalTemplates) or templates.device != rows.device:
                raise ValueError("mymyr: upload templates to the rows' device with templates.to(device)")
            stream_obj = torch.cuda.current_stream(rows.device)
            key = stream_obj.cuda_stream
            if key not in templates._streams:
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("mymyr: warm matching on the capture stream before capture")
                stream_obj.wait_event(templates._ready)
                templates._streams.add(key)
            templates._block.record_stream(stream_obj)
            block, stream = templates._block, stream_obj.cuda_stream
            for tensor in (rows, task_ids, derived, status):
                if tensor is not None:
                    tensor.record_stream(stream_obj)
        elif isinstance(templates, DeviceGoalTemplates):
            raise ValueError("mymyr: CPU rows require host templates")
    else:
        if isinstance(templates, DeviceGoalTemplates):
            raise ValueError("mymyr: NumPy rows require host templates")

        def empty(shape, dtype):
            return np.empty(shape, dtype=dtype)

    found = empty(shape, "bool")
    binding = empty((*shape, host.max_vars), "int32")
    template = empty(shape, "int32")
    count = empty((rows.shape[0],), "int32")
    if status is None:
        status = empty((), "int64")
    _rl_ops.match_first(rows, task_ids, host, block, found, binding, template, count, status,
                        first=first, derived=derived, stream=stream, pool=pool, max_steps=max_steps)
    if first:
        found, binding, template = found[:, 0], binding[:, 0], template[:, 0]
    return MatchResult(found, binding, template, count, status)


def match_first(rows: Any, task_ids: Any, templates: Any, *, first: bool = False, derived: Any = None,
                status: Any = None, capacity: int | None = None, pool: Any = None,
                max_steps: int | None = 100_000_000) -> MatchResult:
    """The first canonical binding of each instance's lifted templates in each state row.

    Templates carry distinctness (True by default; also 'equality' and False). The first binding equals
    Task.bindings(condition, state, limit=1): compiler binding order, objects ascending. Static literals use the
    instance's facts, fluent literals the row, derived literals the supplied derived words (required when used),
    negative literals absence. Inputs use contiguous arrays; task_ids is int32 [N] (None for a single instance).

    With first=True, stop at the first matching template and return found/template [N], binding [N,V]. Otherwise
    slot j holds instance-local template j, padded to capacity (default templates.capacity); returned template ids
    are global packed ids. Count includes truncated successes; status is cleared each call. No device counts are
    read on the host. CUDA uses uploaded templates and captures after warm-up. No JAX form is provided.

    Search is exponential in the worst case. max_steps bounds the work per row and template in steps, each a
    32-object candidate tile scan over the template's literals and their tuples or a binding trial (None: no bound).
    Exhaustion sets status bit 3 (value 8) and leaves that template unmatched; other templates are still searched,
    so counts and first=True then describe only completed searches. Retry with a larger budget to resolve them.
    """
    return _match(rows, task_ids, templates, first, derived, capacity, status, pool, max_steps)
