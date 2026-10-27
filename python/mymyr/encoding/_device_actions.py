"""Bounded observation-action expansions in the encoder's stream order."""

from __future__ import annotations

import threading
import weakref
from typing import Any

from mymyr.encoding import _device as d

_widths: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def action_states(encoder: Any, states: Any, ids: Any) -> Any:
    """Clear words beyond each instance's atom width before the expansion validates its padded input."""
    from mymyr import rl

    torch = d._torch()
    table, device = rl.as_table(encoder.table), states.device
    if table.numeric or not len(table):
        return states
    stream = torch.cuda.current_stream(device)
    capturing = torch.cuda.is_current_stream_capturing()
    with _lock:
        devices = _widths.setdefault(encoder, {})
        if device not in devices:
            if capturing:
                raise d._error("warm action observations on the capture stream before CUDA graph capture")
            words = torch.tensor(table.instance_words, dtype=torch.int64, device=device)
            ready = torch.cuda.Event()
            ready.record(stream)
            devices[device] = words, ready, set()
        words, ready, streams = devices[device]
        if capturing:
            if stream.cuda_stream not in streams:
                raise d._error("warm action observations on the capture stream before CUDA graph capture")
        else:
            stream.wait_event(ready)
            streams.add(stream.cuda_stream)
    if ids is None:
        local = torch.zeros(states.shape[0], dtype=torch.int64, device=device)
    else:
        local = torch.where((ids >= 0) & (ids < len(table)), ids, 0).to(torch.int64)
    return torch.where(torch.arange(states.shape[1], device=device) < words[local, None], states, 0)


def bounded_actions(encoder: Any, states: Any, ids: Any, capacity: int, *, live: Any = None, hashes: bool = False,
                    noops: bool = False, drop_noops: bool = False) -> Any:
    """Expand labels into caller-sized device arrays, retaining the backend's overflow and invalid-input flags.

    ``hashes``, ``noops`` and ``drop_noops`` add ``succ_hash``, ``noop`` and ``index`` (None when absent), digested
    while each successor is built, without storing successor rows.
    """
    from collections import Counter
    from types import SimpleNamespace
    from mymyr import rl
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    table, device = rl.as_table(encoder.table), states.device
    states = action_states(encoder, states, ids)
    # Split schemas can yield several expansion rows for one encoded action entity.
    multiplicity = max(Counter(encoder.schema_relations).values(), default=1)
    capacity = min(0x7fffffff, capacity * multiplicity)
    opts = dict(dtype=torch.int32, device=device)
    x = SimpleNamespace(succ=None,
                        schema=torch.zeros(capacity, **opts), binding=torch.zeros((capacity, table.label_width), **opts),
                        parent=torch.zeros(capacity, **opts), offsets=torch.zeros(states.shape[0] + 1, **opts),
                        status=torch.zeros(1, **opts),
                        succ_hash=torch.zeros(capacity, dtype=torch.int64, device=device) if hashes else None,
                        noop=torch.zeros(capacity, dtype=torch.bool, device=device) if noops or drop_noops else None,
                        index=torch.full((capacity,), -1, **opts) if drop_noops else None)
    if len(table):
        from mymyr._core import _cuda

        _cuda._check_async_expansion(table, states.shape[0], device.index, _stream(device))
        result = rl.expand_into(table, states, ids, schema=x.schema, binding=x.binding, parent=x.parent,
                                offsets=x.offsets, status=x.status, stream=_stream(device),
                                succ_hash=x.succ_hash, noop=x.noop, index=x.index, drop_noops=drop_noops,
                                row_count=None if live is None else live.clamp(0, states.shape[0]).to(torch.int32))
        x._workspace = result.get("_workspace")
        if live is not None:
            total = x.offsets.gather(0, live)
            x.offsets = torch.where(torch.arange(states.shape[0] + 1, device=device) <= live, x.offsets, total)
            used = torch.arange(capacity, device=device) < total
            for name in ("schema", "binding", "parent"):
                array = getattr(x, name)
                mask = used if array.ndim == 1 else used[:, None]
                setattr(x, name, torch.where(mask, array, 0))
    return x
