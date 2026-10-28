"""Encoding the rows of a task suite: one encoder per domain, one batch per domain."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from mymyr._core import _encoding as _core
from mymyr._core._rl import TaskSuite, TaskTable
from mymyr._suite import _Split, _take

if TYPE_CHECKING:
    from mymyr._typing import SupportsDLPack
    from mymyr.encoding import FlatBatch
    from mymyr.rl import Expansion, ThreadPool

__all__ = ["SuiteBatch", "SuiteEncoder"]


class SuiteBatch(Mapping):
    """The encoding of a mixed batch of rows over a suite: a mapping from a domain's index to the batch of the rows
    of that domain (domains without rows are absent), plus where each row went.

    - ``batches[d]`` (also ``suite_batch[d]``): the ``FlatBatch`` of domain ``d``'s rows, in the order they had in the
      call. Its ``task_ids`` are the domain's instance numbers (``suite.local_ids``).
    - ``domain``: int32 ``[N]``, the domain of each row of the call.
    - ``index``: int32 ``[N]``, the index of each row in its domain's batch: row ``i`` is graph ``index[i]`` of
      ``batches[domain[i]]``.
    - ``rows[d]``: int64, the rows of the call that domain ``d``'s batch holds, in graph order (the inverse of
      ``index``): graph ``k`` of ``batches[d]`` encodes row ``rows[d][k]``.
    - ``task_ids``: int32 ``[N]``, the rows' global instance numbers.
    - ``expansions[d]``: the expansion of domain ``d``'s rows when the call expanded them (``actions=True``), else the
      expansions that were given; empty without actions. CUDA capacity calls return device labels, offsets, parents
      and a status word, with ``succ=None``. Graph ``k`` of ``batches[d]`` is state ``k`` of
      ``expansions[d]``, so ``mymyr.encoding.action_rows(batches[d], expansions[d])`` maps its action entities back.

    CUDA results keep all arrays on the device. Capacity calls include every domain, including those without rows;
    each batch has graph capacity N and ``device_graphs`` counts its live prefix. ``rows[d]`` has N entries with -1
    past that prefix. ``graph`` and batch scalar accessors read device values and wait; tensor access does not.
    Invalid global task ids are grouped with instance 0's domain and set that batch's ``invalid_input`` flag;
    host calls raise instead.
    """

    def __init__(
        self,
        batches: dict[int, FlatBatch],
        domain: Any,
        index: Any,
        rows: dict[int, Any],
        task_ids: Any,
        expansions: dict[int, Any],
    ) -> None:
        self.batches = batches
        self.domain = domain
        self.index = index
        self.rows = rows
        self.task_ids = task_ids
        self.expansions = expansions

    def __getitem__(self, domain: int) -> FlatBatch:
        return self.batches[domain]

    def __iter__(self) -> Iterator[int]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def nbytes_by_array(self) -> dict[str, int]:
        """Payload bytes of the encoded batches and row mappings, including capacity and device counters.
        Names are ``batches.<domain>.<array>``, ``rows.<domain>``, ``domain``, ``index`` and ``task_ids``.
        Counts each named array once, without deduplicating aliases; excludes input expansions. Never waits."""
        out = {f"batches.{d}.{name}": size for d, batch in self.batches.items()
               for name, size in batch.nbytes_by_array.items()}
        out.update({f"rows.{d}": int(array.nbytes) for d, array in self.rows.items()})
        out.update({name: int(getattr(self, name).nbytes) for name in ("domain", "index", "task_ids")})
        return out

    @property
    def nbytes(self) -> int:
        """The sum of ``nbytes_by_array``; excludes allocator overhead, inputs, schema and scratch."""
        return sum(self.nbytes_by_array.values())

    @property
    def num_rows(self) -> int:
        """The rows of the call."""
        return len(self.domain)

    def graph(self, row: int) -> tuple[int, int]:
        """``(domain, graph)`` of a row of the call: the batch holding it and its graph there."""
        return int(self.domain[row]), int(self.index[row])

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {b.num_graphs}" for d, b in self.batches.items())
        return f"SuiteBatch({self.num_rows} rows; graphs per domain {{{sizes}}})"


def _shared(array: Any, rows: int) -> bool:
    """Whether a per-row input holds one row that every row shares."""
    shape = getattr(array, "shape", None)
    return shape is not None and len(shape) > 0 and shape[0] == 1 and rows != 1


def _take_input(value: Any, idx: np.ndarray, rows: int) -> Any:
    """The rows `idx` of a per-row input: an array, or a tuple of arrays (goal masks, subgoal layers, history)."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(_take_input(item, idx, rows) for item in value)
    return value if _shared(value, rows) else _take(value, idx, None)


class SuiteEncoder:
    """The encoders of a task suite: one encoder per domain, all under the same options, and the encoding of a batch of
    rows of any instances of any domain.

    Relation ids are domain-scoped (a domain's schema holds that domain's relations), so rows of different domains
    have no common catalog and are not merged into one batch: ``encode`` groups the rows by domain, encodes each group
    with its domain's encoder and returns the batches in a ``SuiteBatch``, which also says where every row went.

    ``SuiteEncoder(suite, encoder=FlatRelationEncoder, **options)`` builds ``encoder(table, **options)`` for the table
    of each domain (``encoder`` is any encoder class that takes a table and options; a ``TaskTable`` is a suite of one
    domain). ``encoders[d]`` is domain ``d``'s encoder and ``schemas[d]`` its ``EncodingSchema``. The encoder is
    reentrant like its parts. CUDA rows and task ids are grouped on their device and produce torch batches.
    """

    def __init__(self, suite: TaskSuite | TaskTable, encoder: type = _core.FlatRelationEncoder, **options: Any) -> None:
        if isinstance(suite, TaskTable):
            suite = TaskSuite([suite])
        if not isinstance(suite, TaskSuite):
            raise TypeError(f"mymyr: encoding: SuiteEncoder takes a TaskSuite or a TaskTable, not {type(suite).__name__}")
        self.suite = suite
        self.encoders = tuple(encoder(table, **options) for table in suite.tables)

    @property
    def schemas(self) -> tuple[Any, ...]:
        """The ``EncodingSchema`` of each domain."""
        return tuple(e.schema for e in self.encoders)

    def encode(
        self,
        states: SupportsDLPack | Sequence[Any],
        task_ids: SupportsDLPack | Sequence[int],
        *,
        derived: SupportsDLPack | None = None,
        goals: Any = None,
        actions: bool | Mapping[int, Expansion] = False,
        subgoals: Any = None,
        history: Any = None,
        framework: str | None = None,
        threads: int = 1,
        pool: ThreadPool | None = None,
        capacities: Mapping[int, Mapping[str, int]] | None = None,
        drop_noops: bool = False,
    ) -> SuiteBatch:
        """Encodes the rows ``states`` (an array ``[N, W]`` of state words, or a list of States) of the instances
        ``task_ids`` (global numbers of the suite) into one batch per domain.

        ``derived``, ``goals``, ``subgoals`` and ``history`` are the per-row inputs of ``FlatRelationEncoder.encode``,
        over the rows of the call (an input of one row, ``[1, ...]``, is shared by all rows); each domain's rows get
        their share. ``actions=True`` expands every domain's rows (``rl.expand`` over the domain's table) and encodes
        their actions; a mapping from a domain to an ``rl.Expansion`` of that domain's rows (in the order of the call,
        over the domain's table with the domain's instance numbers) is used instead where it names a domain.
        ``drop_noops=True`` (with ``actions=True``) expands with ``rl.expand(..., drop_noops=True)``: the actions whose
        successor equals the state are left out unless every successor of the state is one, and the action entities
        keep their canonical successor index as ``entity_ref``.
        ``framework``, ``threads`` and ``pool`` are as in ``encode``; ``capacities`` maps a domain to its capacities,
        and every domain with rows needs an entry.

        On CUDA, exact calls over several domains copy only the per-domain counts to the host to size the groups;
        one domain uses the known input row count. Each domain's exact expansion and encoding also waits for its
        output sizes. Capacity calls need an entry for every domain with
        ``graphs >= N`` and keep all counts and arrays on the device. They allocate at capacity, zero unused output
        tails and report overflow in ``device_sizes``. ``actions=True`` bounds expansions by each domain's entity
        capacity and returns its device label arrays in ``expansions``. Automatic device action expansion requires
        an asynchronous backend, including for exact calls, so the grouping never falls back to downloading rows.
        Given per-domain expansions are accepted only by exact device calls; capacity calls require ``actions=True``
        or no actions. Capacity expansions store labels only (``succ=None``), and preserve the backend's raw status,
        including overflow caused by staged rows past the domain's live prefix.
        Warm a capacity call on the capture stream before CUDA graph capture; keep the encoder, inputs and results
        alive through the final replay. Relations and actions use each domain's local task ids.
        """
        if any(hasattr(value, "__cuda_array_interface__") or
               hasattr(value, "__dlpack_device__") and value.__dlpack_device__()[0] == 2
               for value in (states, task_ids)):
            from mymyr.encoding._suite_device import encode

            return encode(self, states, task_ids, derived=derived, goals=goals, actions=actions, subgoals=subgoals,
                          history=history, framework=framework, threads=threads, pool=pool, capacities=capacities,
                          drop_noops=drop_noops)
        if drop_noops and actions is not True:
            raise ValueError("mymyr: encoding: drop_noops needs actions=True")
        suite = self.suite
        if isinstance(states, (list, tuple)):
            rows = len(states)
            states = np.fromiter(states, object, count=rows)  # so that a domain's States are taken by one index
        else:
            rows = int(states.shape[0])
        split = _Split(suite, task_ids, rows)
        global_ids = split.ids.astype(np.int32)
        domain = np.asarray(suite.domain_of, np.int32)[split.ids]
        index = np.zeros(rows, np.int32)
        batches: dict[int, FlatBatch] = {}
        row_ids: dict[int, np.ndarray] = {}
        expansions: dict[int, Expansion] = {}
        given = actions if isinstance(actions, Mapping) else {}
        if isinstance(actions, Mapping):
            unknown = sorted(set(actions) - set(range(suite.num_domains)))
            if unknown:
                raise ValueError(f"mymyr: encoding: actions name domain {unknown[0]}, the suite has {suite.num_domains}")
        for d, idx, local in split.parts:
            table, encoder = suite.tables[d], self.encoders[d]
            index[idx] = np.arange(len(idx), dtype=np.int32)
            row_ids[d] = idx
            part = _states_of(states, idx, table.words)
            exp = given.get(d)
            if exp is None and actions is True:
                from mymyr import rl

                exp = rl.expand(table, part, local.astype(np.int32), drop_noops=drop_noops)
            if exp is not None:
                expansions[d] = exp
            caps = None
            if capacities is not None:
                if d not in capacities:
                    raise ValueError(f"mymyr: encoding: capacities has no entry for domain {d}, which has rows")
                caps = dict(capacities[d])
            inputs = {
                "derived": _take_input(derived, idx, rows),
                "goals": _take_input(goals, idx, rows),
                "actions": exp,
                "subgoals": _take_input(subgoals, idx, rows),
                "history": _take_input(history, idx, rows),
                "framework": framework,
                "capacities": caps,
            }  # the inputs a call does not give are left out: not every encoder takes all of them
            batches[d] = encoder.encode(
                part,
                local.astype(np.int32),
                threads=threads,
                pool=pool,
                **{name: value for name, value in inputs.items() if value is not None},
            )
        return SuiteBatch(batches, domain, index, row_ids, global_ids, expansions)


def _states_of(states: Any, idx: np.ndarray, words: int) -> Any:
    """The rows `idx` of the states (a list of the States for an object array of them), limited to the `words` atom
    words of a domain's table."""
    if isinstance(states, np.ndarray) and states.dtype == object:
        return states[idx].tolist()
    return _take(states, idx, words)

