"""State rows as a training dataset, encoded when a batch is loaded."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from operator import attrgetter, itemgetter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mymyr._core._datasets import KnowledgeBase, StateSpace
    from mymyr._typing import SupportsDLPack
    from mymyr.encoding import FlatBatch, SuiteBatch, SuiteEncoder
    from mymyr.rl import Expansion, ThreadPool

__all__ = ["EncodedRows", "EncodingCollate", "StateRow", "StateRows"]

_DEFAULT_LABELS = ("unit_goal_distances",)


@dataclass(frozen=True)
class StateRow:
    """One item of a ``StateRows`` dataset: a state's row of atom words ``[W]`` (uint64), the instance it belongs to,
    its labels (name to value) and, for a dataset with goals, the goal masks ``(positive, negative)`` of the row."""

    row: np.ndarray
    task_id: int
    labels: dict[str, Any] = field(default_factory=dict)
    goal: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True)
class EncodedRows:
    """The encoded form of a batch of dataset rows, as ``EncodingCollate`` makes it:

    - ``graphs``: the ``FlatBatch`` of the rows, one graph per row in the order of the call (a ``SuiteBatch`` for a
      ``SuiteEncoder``: one batch per domain);
    - ``task_ids``: int32 ``[N]``, the rows' instances;
    - ``labels``: the stacked labels, name to array ``[N, ...]``;
    - ``actions``: the ``rl.Expansion`` the action relations were encoded from (``None`` without ``actions=True``);
      for a ``SuiteEncoder`` the expansions are in ``graphs.expansions``.
    """

    graphs: FlatBatch | SuiteBatch
    task_ids: np.ndarray
    labels: dict[str, np.ndarray]
    actions: Expansion | None = None


def _words(values: Any, what: str) -> np.ndarray:
    """`values` as a C-contiguous uint64 array [N, W] (int64 words are reinterpreted)."""
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype not in (np.uint64, np.int64):
        raise ValueError(f"mymyr: datasets: StateRows takes {what} as a 64-bit word array [N, W]")
    return np.ascontiguousarray(array).view(np.uint64)


class StateRows:
    """A map-style dataset of state rows: ``dataset[i]`` is a ``StateRow`` and ``len(dataset)`` the number of rows, so it
    works with ``torch.utils.data.DataLoader`` (and anything else that takes ``__len__`` and ``__getitem__``). A
    batch is fetched whole: ``dataset.__getitems__(indices)`` is ``dataset.take(indices)``, which a ``DataLoader``
    calls once per batch, so its ``collate_fn`` receives a ``StateRows``.

    A dataset stores what the state spaces hold: the rows of atom words and the instance of each row, never an encoding.
    The encoding happens when a batch is loaded: ``EncodingCollate(encoder)`` is the ``collate_fn`` that turns the
    items of a batch into an ``EncodedRows``, so an encoder (and its options) can change without touching the data.

    ``StateRows(rows, task_ids=None, *, labels=None, goals=None)``: ``rows`` a ``uint64`` array ``[N, W]`` of state
    words of one table's instances (rows of instances with fewer words are padded with zeros), ``task_ids`` the
    instance of each row (int32 ``[N]``; ``None``: all rows are instance 0), ``labels`` a mapping from a name to an
    array whose first dimension is ``N``, ``goals`` the goal masks ``(positive, negative)`` of every row, each
    ``[N, W]`` (``None``: each row's goal is its instance's).
    """

    def __init__(
        self,
        rows: SupportsDLPack,
        task_ids: SupportsDLPack | Sequence[int] | None = None,
        *,
        labels: Mapping[str, SupportsDLPack] | None = None,
        goals: tuple[SupportsDLPack, SupportsDLPack] | None = None,
    ) -> None:
        self.rows = _words(rows, "rows")
        n = len(self.rows)
        self.task_ids = (
            np.zeros(n, np.int32) if task_ids is None else np.ascontiguousarray(np.asarray(task_ids, dtype=np.int32))
        )
        if self.task_ids.shape != (n,):
            raise ValueError(f"mymyr: datasets: {self.task_ids.shape} task ids for {n} rows")
        self.labels = {name: np.asarray(values) for name, values in (labels or {}).items()}
        for name, values in self.labels.items():
            if len(values) != n:
                raise ValueError(f"mymyr: datasets: label '{name}' has {len(values)} entries for {n} rows")
        self.goals = None
        if goals is not None:
            positive, negative = (_words(g, "goals") for g in goals)
            if positive.shape != negative.shape or len(positive) != n:
                raise ValueError("mymyr: datasets: goals must be two arrays [N, W] of goal masks, one row per state row")
            self.goals = (positive, negative)

    @classmethod
    def from_state_space(
        cls,
        space: StateSpace,
        ids: SupportsDLPack | Sequence[int] | None = None,
        *,
        task_id: int = 0,
        labels: Sequence[str] = _DEFAULT_LABELS,
    ) -> StateRows:
        """The rows of the states ``ids`` of a state space (``None``: all, in id order), all of instance ``task_id``,
        with the arrays of ``space.arrays()`` named by ``labels`` (per-state arrays such as ``unit_goal_distances``,
        ``cost_goal_distances``, ``goal``, ``unsolvable``) as labels. A space of a task with lazy atom slots has no rows
        and is refused."""
        arrays = space.arrays()
        if "state_words" not in arrays:
            raise ValueError("mymyr: datasets: the state space has no rows (its task has lazy atom slots)")
        picked = np.arange(len(arrays["state_words"])) if ids is None else np.asarray(ids, dtype=np.int64)
        rows = np.asarray(arrays["state_words"])[picked]
        return cls(
            rows,
            np.full(len(picked), task_id, np.int32),
            labels={name: np.asarray(arrays[name])[picked] for name in labels},
        )

    @classmethod
    def from_knowledge_base(cls, kb: KnowledgeBase, *, labels: Sequence[str] = _DEFAULT_LABELS) -> StateRows:
        """The rows of every state of every state space of a knowledge base, space after space, each of the instance
        the base's ``task_indices`` give (its index in ``kb.tasks``). The rows are as wide as the widest space's. Encode
        them with an encoder over ``kb.tasks``."""
        parts = [
            cls.from_state_space(space, task_id=int(index), labels=labels)
            for space, index in zip(kb.state_spaces, kb.task_indices)
        ]
        if not parts:
            raise ValueError("mymyr: datasets: the knowledge base has no state spaces")
        width = max(part.rows.shape[1] for part in parts)
        rows = np.concatenate([np.pad(part.rows, ((0, 0), (0, width - part.rows.shape[1]))) for part in parts])
        return cls(
            rows,
            np.concatenate([part.task_ids for part in parts]),
            labels={name: np.concatenate([part.labels[name] for part in parts]) for name in labels},
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> StateRow:
        n = len(self.rows)
        if not isinstance(i, (int, np.integer)):
            raise TypeError("mymyr: datasets: a StateRows is indexed by an integer (take(indices) selects several rows)")
        if not -n <= i < n:
            raise IndexError(f"mymyr: datasets: row {i} of a dataset of {n} rows")
        i %= n
        goal = None if self.goals is None else (self.goals[0][i], self.goals[1][i])
        return StateRow(self.rows[i], int(self.task_ids[i]), {name: v[i] for name, v in self.labels.items()}, goal)

    def take(self, indices: SupportsDLPack | Sequence[int]) -> StateRows:
        """The dataset of the rows ``indices`` (any order, repeats allowed), as a copy: a batch loaded whole instead
        of as ``len(indices)`` single items."""
        picked = np.asarray(indices, dtype=np.int64)
        return StateRows(
            self.rows[picked],
            self.task_ids[picked],
            labels={name: v[picked] for name, v in self.labels.items()},
            goals=None if self.goals is None else (self.goals[0][picked], self.goals[1][picked]),
        )

    def __getitems__(self, indices: Sequence[int]) -> StateRows:
        """``take(indices)``: ``torch.utils.data.DataLoader`` fetches each batch of its sampler with one call of this
        instead of one ``dataset[i]`` per row, and hands the result to its ``collate_fn`` as it is."""
        return self.take(indices)

    def __repr__(self) -> str:
        return f"StateRows({len(self)} rows of {self.rows.shape[1]} words, labels {sorted(self.labels)})"


_ROW, _TASK_ID, _LABELS, _GOAL = attrgetter("row"), attrgetter("task_id"), attrgetter("labels"), attrgetter("goal")
_FIRST, _SECOND = itemgetter(0), itemgetter(1)


def _pack(
    items: Sequence[StateRow],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """The rows, task ids, labels and goals of a list of items, stacked. The fields are gathered by ``map`` and stacked
    by ``np.array``, which iterate in C: the cost per item is that of reading its fields, with no Python code per item."""
    n = len(items)
    if n == 0:
        raise ValueError("mymyr: datasets: cannot encode an empty batch of rows")
    rows = np.array(list(map(_ROW, items)), np.uint64)
    task_ids = np.fromiter(map(_TASK_ID, items), np.int32, count=n)
    label_maps = list(map(_LABELS, items))
    labels = {name: np.array(list(map(itemgetter(name), label_maps))) for name in label_maps[0]}
    goal_pairs = list(map(_GOAL, items))
    without = goal_pairs.count(None)
    if 0 < without < n:
        raise ValueError("mymyr: datasets: some rows of the batch have a goal and some do not")
    goals = None
    if without == 0:
        goals = (np.array(list(map(_FIRST, goal_pairs)), np.uint64), np.array(list(map(_SECOND, goal_pairs)), np.uint64))
    return rows, task_ids, labels, goals


class EncodingCollate:
    """The ``collate_fn`` of a data loader that encodes the loaded rows: ``EncodingCollate(encoder)(items)`` takes the
    items of one batch (a ``StateRows``, as a data loader fetches it, or a list of ``StateRow``) and returns their
    ``EncodedRows``.

    ``encoder`` is a ``FlatRelationEncoder`` over the table whose instances the rows' ``task_id`` number, or a
    ``SuiteEncoder`` when the rows are of the instances of a suite (global ids; the graphs are then one batch per domain).
    ``actions=True`` expands the rows' states and encodes their actions (the expansion is in the result). ``framework``,
    ``threads`` and ``pool`` are as in ``encoder.encode``. Rows with their own goal (a dataset made with ``goals``) are
    encoded with it. The collate function is a plain callable object: it pickles with its encoder, so it works with
    data loader worker processes.
    """

    def __init__(
        self,
        encoder: Any,
        *,
        actions: bool = False,
        framework: str | None = None,
        threads: int = 1,
        pool: ThreadPool | None = None,
    ) -> None:
        self.encoder = encoder
        self.actions = actions
        self.framework = framework
        self.threads = threads
        self.pool = pool

    def __call__(self, items: StateRows | Sequence[StateRow]) -> EncodedRows:
        from mymyr import rl
        from mymyr.encoding import SuiteEncoder

        if isinstance(items, StateRows):
            rows, task_ids, labels, goals = items.rows, items.task_ids, items.labels, items.goals
        else:
            rows, task_ids, labels, goals = _pack(items)
        expansion = None
        extra: dict[str, Any] = {}
        if self.actions:
            if isinstance(self.encoder, SuiteEncoder):
                extra["actions"] = True
            else:
                expansion = rl.expand(self.encoder.table, rows, task_ids)
                extra["actions"] = expansion
        graphs = self.encoder.encode(
            rows, task_ids, goals=goals, framework=self.framework, threads=self.threads, pool=self.pool, **extra
        )
        return EncodedRows(graphs, task_ids, labels, expansion)
