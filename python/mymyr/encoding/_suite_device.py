"""Stable device grouping for the domain-scoped batches of a task suite."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from typing import Any

from mymyr.encoding import _device as d
from mymyr.encoding._device_actions import action_states, bounded_actions

_metadata: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def _catalog(encoder: Any, device: Any) -> tuple:
    torch = d._torch()
    stream = torch.cuda.current_stream(device)
    capturing = torch.cuda.is_current_stream_capturing()
    with _lock:
        devices = _metadata.setdefault(encoder, {})
        if device not in devices:
            if capturing:
                raise d._error("warm the suite encoder before CUDA graph capture to upload its domain catalog")
            domain = torch.tensor(encoder.suite.domain_of, dtype=torch.int64, device=device)
            local = torch.tensor(encoder.suite.local_ids, dtype=torch.int32, device=device)
            ready = torch.cuda.Event()
            ready.record(stream)
            devices[device] = (domain, local, ready, set())
        domain, local, ready, streams = devices[device]
        if capturing:
            if stream.cuda_stream not in streams:
                raise d._error("warm the suite encoder on the capture stream before CUDA graph capture")
        else:
            stream.wait_event(ready)
            streams.add(stream.cuda_stream)
        return domain, local


def _input(value: Any, idx: Any, count: int, keep: list) -> Any:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(_input(v, idx, count, keep) for v in value)
    torch = d._torch()
    value = d._cuda_tensor(torch, value, "per-row input", keep)
    if value.shape[0] == 1 and count != 1:
        return value
    return value[idx]


def _states(states: Any, idx: Any, table: Any, suite: Any) -> Any:
    rows = states[idx]
    if table.numeric_words:
        return d._torch().cat((rows[:, :table.words],
                               rows[:, suite.words:suite.words + table.numeric_words]), dim=1)
    return rows[:, :table.words]


def encode(encoder: Any, states: Any, task_ids: Any, *, derived: Any, goals: Any, actions: Any,
           subgoals: Any, history: Any, framework: Any, threads: Any, pool: Any, capacities: Any,
           drop_noops: bool = False) -> Any:
    from mymyr import rl
    from mymyr.encoding._suite import SuiteBatch
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    if pool is not None or framework not in (None, "torch", "pytorch"):
        raise d._error("device suites need torch output and no host thread pool")
    for part_encoder in encoder.encoders:
        if part_encoder.options["horizon_mode"] is not None:
            raise d._error(f"device suites encode relational schemas; {type(part_encoder).__name__} takes host rows "
                           "or a HorizonView per domain")
    if drop_noops and actions is not True:
        raise d._error("drop_noops needs actions=True")
    if capacities is not None and isinstance(actions, Mapping) and actions:
        raise d._error("device suite capacity calls do not accept given per-domain expansions; use actions=True")
    if task_ids is None:
        raise d._error("a suite needs task_ids, one per row")
    keep: list = []
    states = d._words(torch, d._cuda_tensor(torch, states, "states", keep), "states")
    ids = d._cuda_tensor(torch, task_ids, "task_ids", keep)
    n, device = states.shape[0], states.device
    if ids.dtype != torch.int32 or ids.shape != (n,) or ids.device != device:
        raise d._error("suite task_ids must be int32 [N] on the states' device")
    suite = encoder.suite
    if n and not len(suite):
        raise d._error("a nonempty batch needs instances in its suite")
    catalog, local_ids = _catalog(encoder, device)
    valid = (ids >= 0) & (ids < len(suite))
    safe = torch.where(valid, ids, 0).to(torch.int64)
    domain = catalog[safe] if len(suite) else torch.empty(n, dtype=torch.int64, device=device)
    local = torch.where(valid, local_ids[safe], -1) if len(suite) else ids
    one_hot = (domain[:, None] == torch.arange(suite.num_domains, device=device)).to(torch.int64)
    within = one_hot.cumsum(0)
    counts = one_hot.sum(0)
    index = (within - 1).gather(1, domain[:, None]).reshape(-1) if suite.num_domains else domain
    starts = counts.cumsum(0) - counts
    order = torch.empty(n, dtype=torch.int64, device=device)
    row = torch.arange(n, device=device)
    order.scatter_(0, starts[domain] + index, row)
    given = actions if isinstance(actions, Mapping) else {}
    if set(given) - set(range(suite.num_domains)):
        raise d._error("actions name a domain outside the suite")
    if capacities is None:
        if torch.cuda.is_current_stream_capturing():
            raise d._error("device suite encoding under CUDA graph capture needs capacities")
        # Only a multi-domain batch's counts cross the device boundary; the permutation and row data stay resident.
        sizes = [n] if suite.num_domains == 1 else counts.cpu().tolist()
    else:
        missing = set(range(suite.num_domains)) - set(capacities)
        if missing:
            raise d._error(f"capacities has no entry for domain {min(missing)}; device capacity calls need every domain")
        sizes = None
    batches, rows, expansions = {}, {}, {}
    offset = 0
    for k, (table, part_encoder) in enumerate(zip(suite.tables, encoder.encoders, strict=True)):
        if sizes is not None:
            count = sizes[k]
            if not count:
                continue
            idx = order[offset:offset + count]
            offset += count
            live = None
            caps = None
        else:
            caps = d._capacities(dict(capacities[k]), d.layout_of(part_encoder))
            if caps[0] < n:
                raise d._error("each domain's graph capacity must cover all input rows")
            live = counts[k:k + 1].clone()
            idx = order[(starts[k] + row).clamp(max=n - 1)] if n else order
        rows[k] = idx if live is None else torch.where(row < live, idx, -1)
        part = _states(states, idx, table, suite)
        part_ids = local[idx].contiguous()
        if live is not None:
            mask = row < live
            part = torch.where(mask[:, None], part, 0)
            part_ids = torch.where(mask, part_ids, 0)
        exp = given.get(k)
        if exp is None and actions is True:
            if live is None:
                from mymyr._core import _cuda

                _cuda._check_async_expansion(table, part.shape[0], device.index, _stream(device))
                exp = rl.expand(table, action_states(part_encoder, part, part_ids), part_ids,
                                framework="torch", stream=_stream(device), drop_noops=drop_noops)
            else:
                exp = bounded_actions(part_encoder, part, part_ids, caps[1], live=live, drop_noops=drop_noops)
        if exp is not None:
            expansions[k] = exp
        inputs = [_input(v, idx, n, keep) for v in (derived, goals, subgoals, history)]
        if live is None:
            kwargs = dict(derived=inputs[0], goals=inputs[1], actions=exp, subgoals=inputs[2], history=inputs[3],
                          framework=framework)
            batches[k] = part_encoder.encode(part, part_ids, threads=threads,
                                             **{name: value for name, value in kwargs.items() if value is not None})
        else:
            checked = d._Rows(torch, part, part_ids, inputs[0], inputs[1], inputs[2], inputs[3])
            if exp is not None:
                from mymyr.encoding._device_horizon import bind_actions
                bind_actions(part_encoder, checked, exp)
            arrays = d._encode_rows(part_encoder, checked, caps, keep, graph_count=live)
            batches[k] = d.DeviceFlatBatch(part_encoder, arrays, exact=False)
            batches[k]._device_graphs = live
    d._retain(torch, [encoder, *keep, *expansions.values()], device)
    return SuiteBatch(batches, domain.to(torch.int32), index.to(torch.int32), rows, ids, expansions)
