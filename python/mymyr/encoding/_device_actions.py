"""Bounded observation-action expansions in the encoder's stream order."""

from __future__ import annotations

import threading
import weakref
from typing import Any

from mymyr.encoding import _device as d

_widths: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_tables: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def table_of(encoder: Any) -> Any:
    """The TaskTable of the encoder, one object per encoder: the table of an encoder over a single Task is made once,
    so that the device expansions of every call (and every captured replay) share its device context."""
    from mymyr import rl

    with _lock:
        table = _tables.get(encoder)
        if table is None:
            table = _tables[encoder] = rl.as_table(encoder.table)
    return table


def action_states(encoder: Any, states: Any, ids: Any) -> Any:
    """Clear words beyond each instance's atom width before the expansion validates its padded input."""
    torch = d._torch()
    table, device = table_of(encoder), states.device
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


def bounded_actions(encoder: Any, states: Any, ids: Any, capacity: int, *, successors: int | None = None,
                    live: Any = None, hashes: bool = False, noops: bool = False, drop_noops: bool = False) -> Any:
    """Expand labels into caller-sized device arrays, retaining the backend's overflow and invalid-input flags.

    ``capacity`` is the entity capacity of the encoding: the arrays have ``action_capacity(encoder, capacity)`` rows,
    as many as its action entities can stand for. ``successors`` bounds the successors before ``drop_noops`` drops the
    no-op rows (default: the rows of the arrays), so that the arrays bound the kept rows only. Status bit 0 reports
    more successors than ``successors`` or more kept rows than the arrays have, bit 1 an invalid row. ``hashes``,
    ``noops`` and ``drop_noops`` add ``succ_hash``, ``noop`` and ``index`` (None when absent), digested while each
    successor is built, without storing successor rows. Every array is zero past the kept rows (``offsets[N]``; with
    ``live``, the rows of the first ``live`` states). ``labels`` is the block that ``parent``, ``schema`` and
    ``index`` are rows of (without ``live``).
    """
    from types import SimpleNamespace
    from mymyr import rl
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    table, device = table_of(encoder), states.device
    states = action_states(encoder, states, ids)
    rows = action_capacity(encoder, capacity)
    work = expansion_rows(encoder, capacity, successors, drop_noops)
    width = max(rows, work)
    opts = dict(dtype=torch.int32, device=device)
    # parent, schema and index in one block, so that one operation zeroes the padding the drop's compaction writes
    # past the kept rows (-1; their kept values are not negative)
    labels = torch.zeros((3 if drop_noops else 2, width), **opts)
    x = SimpleNamespace(succ=None, parent=labels[0], schema=labels[1], labels=labels,
                        binding=torch.zeros((width, table.label_width), **opts),
                        offsets=torch.zeros(states.shape[0] + 1, **opts), status=torch.zeros(1, **opts),
                        succ_hash=torch.zeros(width, dtype=torch.int64, device=device) if hashes else None,
                        noop=torch.zeros(width, dtype=torch.bool, device=device) if noops or drop_noops else None,
                        index=labels[2] if drop_noops else None)
    names = ("schema", "binding", "parent", "succ_hash", "noop", "index")
    if len(table):
        from mymyr._core import _cuda

        _cuda._check_async_expansion(table, states.shape[0], device.index, _stream(device))
        into = {name: None if getattr(x, name) is None else getattr(x, name)[:work] for name in names}
        result = rl.expand_into(table, states, ids, offsets=x.offsets, status=x.status, stream=_stream(device),
                                drop_noops=drop_noops, **into,
                                row_count=None if live is None else live.clamp(0, states.shape[0]).to(torch.int32))
        x._workspace = result.get("_workspace")
        total = x.offsets[-1:] if live is None else x.offsets.gather(0, live)
        if work > rows:
            x.status = x.status | (total > rows).to(torch.int32)
        if live is not None:
            # the rows of the states past `live` are not the observation's
            x.offsets = torch.where(torch.arange(states.shape[0] + 1, device=device) <= live, x.offsets, total)
            used = torch.arange(rows, device=device) < total
            for name in names:
                array = getattr(x, name)
                if array is not None:
                    mask = used if array.ndim == 1 else used[:, None]
                    setattr(x, name, torch.where(mask, array[:rows], False if array.dtype == torch.bool else 0))
        elif drop_noops:
            # the expansion writes the rows below its total only; the compaction pads the rows past the kept ones
            # (labels -1, digests and flags 0)
            labels.clamp_(min=0)
            x.binding = torch.where((torch.arange(rows, device=device) < total)[:, None], x.binding[:rows], 0)
    if width > rows:
        # a contiguous block of the rows, as wide as the arrays (the working rows past them are scratch)
        labels = labels[:, :rows].contiguous()
        if live is None:
            x.parent, x.schema = labels[0], labels[1]
            x.index = labels[2] if drop_noops else None
    for name in names:
        if getattr(x, name) is not None and getattr(x, name).shape[0] != rows:
            setattr(x, name, getattr(x, name)[:rows])
    x.labels = labels
    return x


def action_capacity(encoder: Any, entities: int) -> int:
    """The expansion rows that ``entities`` action entities of ``encoder`` can stand for: split schemas can yield
    several expansion rows for one encoded action entity."""
    from collections import Counter

    multiplicity = max(Counter(encoder.schema_relations).values(), default=1)
    return min(0x7fffffff, entities * multiplicity)


def expansion_rows(encoder: Any, entities: int, successors: int | None, drop_noops: bool) -> int:
    """The successors a bounded expansion has room for before it drops no-op rows: ``successors`` (default: the
    ``action_capacity`` of the entity capacity; at most 2**31 - 1, the rows an expansion indexes); without the drop
    the kept rows are all successors, so the action capacity bounds them as well. With the drop a device holds that
    many working rows twice (the labels, digests and flags, and the compaction's copy of them): about
    ``38 + 8 * label_width`` bytes a successor."""
    rows = action_capacity(encoder, entities)
    if successors is None:
        return rows
    if isinstance(successors, bool) or not isinstance(successors, int) or not 0 <= successors <= 0x7fffffff:
        raise d._error(f"the successors capacity must be an int in [0, 2**31 - 1], not {successors!r}")
    return successors if drop_noops else min(rows, successors)


def bound_status(encoder: Any, expansion: Any, entities: int, successors: int | None, drop_noops: bool) -> int:
    """The status word bit 0 that ``bounded_actions`` reports for the host expansion ``expansion`` of the same rows:
    more successors than ``expansion_rows`` or more kept rows than ``action_capacity``."""
    work = expansion_rows(encoder, entities, successors, drop_noops)
    return int(int(expansion.generated) > work or int(expansion.total) > action_capacity(encoder, entities))


def overflowed(batch: Any) -> Any:
    """``batch`` (a host FlatBatch encoded with capacities) as a batch whose bounded expansion overflowed: all arrays
    and sizes zero, ``capacity_overflow`` set, as a device reports it (its encoding does not run on such rows)."""
    import numpy as np

    from mymyr._core._encoding import FlatBatch

    arrays = {name: np.zeros_like(value) if isinstance(value, np.ndarray) else value
              for name, value in batch.as_dict("numpy").items()}
    sizes = {name: value if isinstance(value, bool) else 0 for name, value in batch.sizes.items()}
    sizes.update(overflow=True, capacity_overflow=True)
    module = type(batch.task_ids).__module__.split(".")[0]
    framework = {"torch": "torch", "jax": "jax", "jaxlib": "jax"}.get(module, "numpy")
    return FlatBatch.from_arrays(batch.schema, arrays, sizes=sizes, framework=framework)
