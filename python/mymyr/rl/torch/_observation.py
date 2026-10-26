"""The encoded observations of a BatchedEnv: its rows through FlatRelationEncoders, on the environments' device."""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from torch import Tensor

from mymyr import rl
from mymyr._core import _rl
from mymyr._core._encoding import FlatRelationEncoder
from mymyr.encoding import _device

__all__ = ["SuiteObservation"]


class SuiteObservation(NamedTuple):
    """The encoding of the N rows of environments over a TaskSuite of several domains: one batch per domain, since
    each domain has a relation catalog of its own.

    - ``batches[d]``: the batch of domain ``d``'s rows, in row order (graph k is the k-th row of domain d); its
      ``task_ids`` are the domain's instance numbers (``suite.local_ids``). On a device every batch has the graph
      capacity N: ``device_graphs`` counts the domain's rows (``num_graphs`` reads it), the per-graph arrays are zero
      past them. On the CPU a batch has the domain's rows exactly (possibly none).
    - ``domain``: int32 [N], the domain of each row.
    - ``index``: int32 [N], the graph of each row in its domain's batch: row i is graph ``index[i]`` of
      ``batches[domain[i]]``.
    """

    batches: tuple
    domain: Tensor
    index: Tensor

    @property
    def nbytes_by_array(self) -> dict[str, int]:
        """Payload bytes of domain batches and row mappings, including capacity and device counters; never waits.
        Names are ``batches.<domain>.<array>``, ``domain`` and ``index``. Aliases are counted by name."""
        out = {f"batches.{d}.{name}": size for d, batch in enumerate(self.batches)
               for name, size in batch.nbytes_by_array.items()}
        out.update(domain=int(self.domain.nbytes), index=int(self.index.nbytes))
        return out

    @property
    def nbytes(self) -> int:
        """The sum of ``nbytes_by_array``; excludes allocator overhead, inputs, schema and scratch."""
        return sum(self.nbytes_by_array.values())


def _encoders(observation: Any) -> list:
    """The encoders of ``observation``: one encoder, a sequence of encoders (one per domain) or an object with an
    ``encoders`` sequence (a suite encoder)."""
    if hasattr(observation, "encoders"):
        observation = observation.encoders
    if isinstance(observation, (list, tuple)):
        return list(observation)
    return [observation]


def _capacities(value: Any, domains: int, rows: int) -> Optional[list[dict]]:
    """Per encoder the capacities dict with ``graphs`` (default: the rows), or None (exact encodings)."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = [value] * domains
    elif not isinstance(value, Sequence) or isinstance(value, str) or len(value) != domains:
        raise ValueError(f"mymyr: obs_capacities must be a dict or a sequence of {domains} dicts (one per domain)")
    caps = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("mymyr: obs_capacities must be a dict or a sequence of dicts, one per domain")
        caps.append({"graphs": rows, **item})
        if caps[-1]["graphs"] < rows:
            raise ValueError(f"mymyr: obs_capacities['graphs'] must be at least the number of environments ({rows})")
    return caps


class Observer:
    """Encodes rows of a BatchedEnv (states, task ids and goal masks of its layout) with the environment's encoders.

    Over a table (or a suite of one domain) the rows are one batch. Over a suite of several domains a device groups the
    rows by domain with torch operations into buffers of its own (fixed addresses, so the grouping captures into CUDA
    graphs) and encodes each domain's rows as the live rows of a device ring: the batch has the graph capacity N and a
    device count of graphs, and nothing waits for the device. On the CPU the rows are grouped on the host.
    """

    def __init__(self, env: Any, observation: Any, capacities: Any, actions: bool) -> None:
        table = env.core.table
        n = env.num_envs
        self.device = env.device
        self.suite = table if isinstance(table, _rl.TaskSuite) and table.num_domains > 1 else None
        tables = list(self.suite.tables) if self.suite is not None else [_table_of(table)]
        encoders = _encoders(observation)
        if len(encoders) != len(tables):
            raise ValueError(f"mymyr: observation needs {len(tables)} encoders, one per domain of the environments' "
                             f"{'suite' if self.suite is not None else 'table'}, got {len(encoders)}")
        for d, (encoder, t) in enumerate(zip(encoders, tables, strict=True)):
            if type(encoder) is not FlatRelationEncoder:
                raise TypeError(f"mymyr: observation must be a FlatRelationEncoder (or one per domain), not "
                                f"{type(encoder).__name__}")
            if encoder.table_fingerprint != t.fingerprint:
                where = f"domain {d} of the suite" if self.suite is not None else "the environments' table"
                raise ValueError(f"mymyr: observation: encoder {d} encodes another table than {where} (encode the "
                                 "table the environments run on)")
        self.encoders = encoders
        self.tables = tables
        self.capacities = _capacities(capacities, len(encoders), n)
        self.actions = bool(actions)
        if self.actions and self.capacities is not None and self.device.type == "cuda":
            self._warm_rows = (env.states, env.task_ids, (env.goal_pos, env.goal_neg) if env.goals else None)
        self.goals = [_goal_lanes(e) for e in encoders]
        if self.actions:
            for e in encoders:
                if not e.options["include_actions"]:
                    raise ValueError("mymyr: obs_actions needs an encoder with action relations (include_actions=True)")
        if self.device.type == "cuda":
            for k, e in enumerate(encoders):
                layout = _device.layout_of(e)  # refuses schemas the device does not encode
                if self.capacities is not None:
                    _device._capacities(self.capacities[k], layout)
            if self.capacities is not None and self.suite is None:
                from mymyr.rl.torch import _ops

                # what a traced encoding needs, made eagerly: torch.compile traces the op and the batch only
                self.handles = [_ops._register_owned(e) for e in encoders]
                self._templates = [_device.DeviceFlatBatch._template(e) for e in encoders]
                # graphs, entities, args, targets, history and numeric: the op's capacity arguments, in its order
                self._op_capacities = [_device._capacities(c, _device.layout_of(e))
                                       for c, e in zip(self.capacities, encoders, strict=True)]
            if self.suite is not None and self.capacities is not None:
                self._suite_buffers(env)
        if self.suite is not None:
            domain_of = np.asarray(self.suite.domain_of, np.int32)
            local = np.asarray(self.suite.local_ids, np.int32)
            self._domain_of = torch.as_tensor(domain_of, device=self.device)
            self._local_ids = torch.as_tensor(local, device=self.device)
            self._domains = torch.arange(len(encoders), dtype=torch.int64, device=self.device)

    def warm(self, rows: int) -> None:
        """Acquires the device contexts of the encoders on torch's current stream (a capture stream, before the
        capture: a context is uploaded outside any capture)."""
        if self.device.type != "cuda":
            return
        from mymyr.rl.torch._ops import _stream

        stream = _stream(self.device)
        for e in self.encoders:
            _device._device_encoder(e, self.device.index).scratch_bytes(stream, rows, False)
        if self.actions and self.capacities is not None:
            self.encode(*self._warm_rows)

    # ------------------------------------------------------------------------------------------ the device rings

    def _suite_buffers(self, env: Any) -> None:
        from mymyr.rl.torch._ops import _stream

        n, d = env.num_envs, self.device
        native = _device._native()
        for e in self.encoders:
            native.check_stream_schema(e)
        self._stream = _stream(d)
        self._natives = [_device._device_encoder(e, d.index) for e in self.encoders]
        for native_d in self._natives:
            native_d.scratch_bytes(self._stream, n, False)  # acquires the device context outside any capture
        self._rows_states = torch.zeros((n, env.row_words), dtype=torch.int64, device=d)
        self._rows_ids = torch.zeros(n, dtype=torch.int32, device=d)
        self._rows_goals = None
        if env.goals:
            self._rows_goals = (torch.zeros((n, env.words), dtype=torch.int64, device=d),
                                torch.zeros((n, env.words), dtype=torch.int64, device=d))
        self._status = torch.zeros((len(self.encoders), 3), dtype=torch.int64, device=d)
        self._ring = [
            _device._Rows(torch, self._rows_states, self._rows_ids, None,
                          self._rows_goals if self.goals[k] else None)
            for k in range(len(self.encoders))
        ]

    # ------------------------------------------------------------------------------------------ encoding

    def encode(self, states: Tensor, task_ids: Tensor, goals: Optional[tuple]) -> Any:
        """The encoding of the rows: a batch over a table, a :class:`SuiteObservation` over a suite of several
        domains."""
        if self.suite is None:
            return self._encode(0, states, task_ids, goals)
        if self.device.type == "cuda" and self.capacities is not None:
            return self._encode_suite_device(states, task_ids, goals)
        return self._encode_suite_split(states, task_ids, goals)

    def _encode(self, k: int, states: Tensor, task_ids: Tensor, goals: Optional[tuple]) -> Any:
        encoder = self.encoders[k]
        goals = goals if self.goals[k] else None
        caps = None if self.capacities is None else self.capacities[k]
        if self.device.type == "cuda":
            if torch.compiler.is_compiling():
                return self._encode_op(k, states, task_ids, goals, caps)
            actions = None
            if self.actions:
                from mymyr.rl.torch._ops import _stream

                if caps is None:
                    actions = rl.expand(self.tables[k], states, task_ids, framework="torch", stream=_stream(states))
                else:
                    from mymyr.encoding._device_actions import bounded_actions

                    actions = bounded_actions(encoder, states, task_ids, caps["entities"])
            arrays = _device.encode_arrays(encoder, states, task_ids, None, goals,
                                           None if caps is None else _device._capacities(caps, _device.layout_of(encoder)),
                                           actions=actions)
            return _device.DeviceFlatBatch(encoder, arrays, exact=caps is None)
        actions = None
        if self.actions:
            actions = rl.expand(self.tables[k], states, task_ids)
        return encoder.encode(states, task_ids, goals=goals, actions=actions, capacities=caps)

    def _encode_op(self, k: int, states: Tensor, task_ids: Tensor, goals: Optional[tuple], caps: Optional[dict]) -> Any:
        if caps is None:
            raise ValueError("mymyr: observations under torch.compile need obs_capacities (static shapes)")
        gp, gn = (None, None) if goals is None else goals
        outs = torch.ops.mymyr.encode_flat(
            self.handles[k], states, task_ids, None, gp, gn, None, None, None, None, None, None, None, None, None,
            None, None, None, *self._op_capacities[k],
            expand_actions=self.actions,
        )  # fmt: skip
        return _device.DeviceFlatBatch._of(self._templates[k], outs, self._op_capacities[k][1])

    def _suite_ids(self, task_ids: Tensor) -> tuple[Tensor, Tensor]:
        ids = task_ids.to(torch.int64)
        valid = (ids >= 0) & (ids < self._domain_of.numel())
        safe = torch.where(valid, ids, 0)
        domain = self._domain_of[safe].to(torch.int64)
        # Invalid rows reach the encoder with an invalid local id, so its validation reports them.
        local = torch.where(valid, self._local_ids[safe], -1)
        return domain, local

    def _group(self, domain: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Each row's index among its domain's rows, its position in the grouped rows (domain by
        domain, in row order within one) and the rows of each domain [D] (int64 tensors on the device)."""
        one_hot = (domain.unsqueeze(1) == self._domains).to(torch.int64)  # no host check of the values
        within = one_hot.cumsum(0)
        counts = within[-1] if within.shape[0] else one_hot.sum(0)
        index = (within - 1).gather(1, domain.unsqueeze(1)).squeeze(1)
        starts = counts.cumsum(0) - counts
        return index, starts[domain] + index, counts

    def _encode_suite_device(self, states: Tensor, task_ids: Tensor, goals: Optional[tuple]) -> SuiteObservation:
        if torch.compiler.is_compiling():
            raise ValueError("mymyr: the observations of a suite of several domains are not traced by torch.compile: "
                             "capture the steps with BatchedEnv.capture or torch.cuda.graph")
        domain, local = self._suite_ids(task_ids)
        index, position, counts = self._group(domain)
        self._rows_states.index_copy_(0, position, states)
        self._rows_ids.index_copy_(0, position, local)
        if self._rows_goals is not None and goals is not None:
            self._rows_goals[0].index_copy_(0, position, goals[0])
            self._rows_goals[1].index_copy_(0, position, goals[1])
        starts = counts.cumsum(0) - counts
        self._status.copy_(torch.stack((starts, counts, torch.zeros_like(counts)), dim=1))
        n = states.shape[0]
        from mymyr.rl.torch._ops import _stream

        stream = _stream(self.device)
        batches = []
        for k, encoder in enumerate(self.encoders):
            caps = _device._capacities(self.capacities[k], _device.layout_of(encoder))
            if self.actions:
                from mymyr.encoding._device_actions import bounded_actions
                from mymyr.encoding._device_horizon import bind_actions

                live = counts[k:k + 1].clone()
                row = torch.arange(n, device=self.device)
                idx = (starts[k] + row).clamp(max=n - 1)
                mask = row < live
                table = self.tables[k]
                states_k = torch.where(mask[:, None], self._rows_states[idx, :table.words], 0)
                ids_k = torch.where(mask, self._rows_ids[idx], 0)
                goals_k = None if goals is None or not self.goals[k] else tuple(g[idx] for g in self._rows_goals)
                actions = bounded_actions(encoder, states_k, ids_k, caps[1], live=live)
                checked = _device._Rows(torch, states_k, ids_k, None, goals_k)
                bind_actions(encoder, checked, actions)
                arrays = _device._encode_rows(encoder, checked, caps, [actions], graph_count=live)
                batch = _device.DeviceFlatBatch(encoder, arrays, exact=False)
                batch._device_graphs = live
                batches.append(batch)
                continue
            batch, scratch = _device._flush_rows(torch, self._natives[k], stream, self._ring[k], self._status[k], encoder,
                                                 n, caps, self.device, False, zero_unused=True)
            batch._keepalive = (scratch,)
            batches.append(batch)
        return SuiteObservation(tuple(batches), domain.to(torch.int32), index.to(torch.int32))

    def _encode_suite_split(self, states: Tensor, task_ids: Tensor, goals: Optional[tuple]) -> SuiteObservation:
        """Groups the rows on the host (a CPU environment, or exact encodings on a device, which wait anyway)."""
        domain, local = self._suite_ids(task_ids)
        one_hot = (domain.unsqueeze(1) == self._domains).to(torch.int64)  # no host check of the values
        index = (one_hot.cumsum(0) - 1).gather(1, domain.unsqueeze(1)).squeeze(1)
        host_domain = domain.cpu()
        batches = []
        for k in range(len(self.encoders)):
            rows = torch.nonzero(host_domain == k).reshape(-1).to(states.device)
            g = None if goals is None else (goals[0][rows], goals[1][rows])
            batches.append(self._encode(k, states[rows], local[rows].contiguous(), g))
        return SuiteObservation(tuple(batches), domain.to(torch.int32), index.to(torch.int32))

    def close(self) -> None:
        from mymyr.rl.torch import _ops

        for h in getattr(self, "handles", ()):
            _ops.unregister(h)
        self.handles = []


def _table_of(table: Any) -> Any:
    if isinstance(table, _rl.TaskSuite):
        return table.tables[0]
    return rl.as_table(table)


def _goal_lanes(encoder: Any) -> bool:
    return any(lane in _device._GOAL_LANES for lane in encoder.relation_lanes)
