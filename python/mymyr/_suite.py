"""Task suites through the one-domain entry points of the CUDA backend: device IW
(``mymyr.cuda.multi_iw`` / ``batched_iw1``) and device state spaces (``mymyr.cuda.state_spaces`` /
``generate_state_spaces``) run each domain's searches or instances on its table, the domains one after the other, and
return the results in the suite's order (global task ids). A one-domain suite is its table (no split).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from mymyr._core._rl import TaskSuite

__all__ = ["SuiteIwBatch", "batched_iw1", "generate_state_spaces", "multi_iw", "state_spaces"]


def _host_ints(ids: Any) -> np.ndarray:
    """Task ids as host int64 (sequences, NumPy, torch and JAX arrays on any device)."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    return np.asarray(ids).astype(np.int64, copy=False).reshape(-1)


def _take(rows: Any, idx: np.ndarray, cols: int | None) -> Any:
    """rows[idx] (and its first `cols` 64-bit words) in the rows' framework and on their device."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(rows, torch.Tensor):
        out = rows[torch.as_tensor(idx, device=rows.device)]
    else:
        jax = sys.modules.get("jax")
        if jax is not None and isinstance(rows, jax.Array):
            import jax.numpy as jnp

            out = rows[jnp.asarray(idx)]
        elif isinstance(rows, np.ndarray):
            out = rows[idx]
        else:
            raise TypeError("mymyr: rows over a TaskSuite must be NumPy, torch or JAX arrays (or States)")
    if cols is not None and out.ndim == 2:
        out = out[:, : cols * (2 if out.dtype.itemsize == 4 else 1)]
    return out


class _Split:
    """A batch's rows grouped by domain (stable: batch order within a domain)."""

    def __init__(self, suite: TaskSuite, task_ids: Any, rows: int) -> None:
        if task_ids is None:
            raise ValueError(f"mymyr: a suite of {len(suite)} instances needs task_ids (one per search)")
        ids = _host_ints(task_ids)
        if len(ids) != rows:
            raise ValueError(f"mymyr: {len(ids)} task ids for {rows} searches")
        bad = np.flatnonzero((ids < 0) | (ids >= len(suite)))
        if len(bad):
            raise ValueError(
                f"mymyr: task id {int(ids[bad[0]])} of search {int(bad[0])} is outside the suite's {len(suite)} instances"
            )
        domain = np.asarray(suite.domain_of, np.int64)[ids]
        local = np.asarray(suite.local_ids, np.int32)[ids]
        self.ids = ids
        self.parts = []  # (domain, batch rows, local ids)
        for d in range(suite.num_domains):
            idx = np.flatnonzero(domain == d)
            if len(idx):
                self.parts.append((d, idx, local[idx]))


def _starts_of(starts: Any, idx: np.ndarray, cols: int | None) -> Any:
    if isinstance(starts, (list, tuple)):
        return [starts[i] for i in idx]
    return _take(starts, idx, cols)


def _rows(starts: Any) -> int:
    if isinstance(starts, (list, tuple)):
        return len(starts)
    return int(starts.shape[0]) if getattr(starts, "ndim", 1) > 1 else 1


class SuiteIwBatch:
    """The results of device IW searches over a TaskSuite (mymyr.cuda.multi_iw / batched_iw1): search i ran on global
    instance ``task_ids[i]`` (of domain ``domains[i]``), with what an IwBatch reports per search. Each domain's searches
    ran as one IwBatch over its table (``parts``), the domains one after the other."""

    def __init__(self, suite: TaskSuite, split: _Split, parts: list[Any]) -> None:
        self.suite = suite
        self.parts = parts
        n = len(split.ids)
        self._where = [(0, 0)] * n
        for k, (_, idx, _) in enumerate(split.parts):
            for j, i in enumerate(idx):
                self._where[int(i)] = (k, j)
        self._ids = [int(v) for v in split.ids]

    def __len__(self) -> int:
        return len(self._ids)

    def _at(self, i: int) -> tuple[Any, int]:
        if i < 0:
            i += len(self._ids)
        if not 0 <= i < len(self._ids):
            raise IndexError("mymyr: search index out of range")
        k, j = self._where[i]
        return self.parts[k], j

    def _gather(self, name: str) -> list[Any]:
        cols = [getattr(p, name) for p in self.parts]
        return [cols[k][j] for k, j in self._where]

    @property
    def status(self) -> list[Any]:
        return self._gather("status")

    @property
    def solved(self) -> list[bool]:
        return self._gather("solved")

    @property
    def plan_length(self) -> list[int]:
        return self._gather("plan_length")

    @property
    def effective_width(self) -> list[int]:
        return self._gather("effective_width")

    @property
    def expanded(self) -> list[int]:
        return self._gather("expanded")

    @property
    def generated(self) -> list[int]:
        return self._gather("generated")

    @property
    def task_ids(self) -> list[int]:
        """Per search: its global instance."""
        return list(self._ids)

    @property
    def domains(self) -> list[int]:
        """Per search: its instance's domain."""
        dom = self.suite.domain_of
        return [dom[g] for g in self._ids]

    @property
    def stats(self) -> dict[str, Any]:
        """The parts' device statistics summed (device_bytes: the most of a part)."""
        out: dict[str, Any] = {}
        for p in self.parts:
            for key, v in p.stats.items():
                out[key] = max(out.get(key, 0), v) if key == "device_bytes" else out.get(key, 0) + v
        return out

    def plan(self, i: int) -> list[Any]:
        p, j = self._at(i)
        return p.plan(j)

    def goal_state(self, i: int) -> Any:
        p, j = self._at(i)
        return p.goal_state(j)

    def passes(self, i: int) -> list[Any]:
        p, j = self._at(i)
        return p.passes(j)

    def cost(self, i: int) -> float:
        p, j = self._at(i)
        return p.cost(j)

    def reached_slots(self, i: int) -> list[int]:
        p, j = self._at(i)
        return p.reached_slots(j)

    def reached_atoms(self, i: int) -> list[int]:
        p, j = self._at(i)
        return p.reached_atoms(j)

    def __repr__(self) -> str:
        return f"SuiteIwBatch(searches={len(self)}, solved={sum(self.solved)}, domains={len(self.parts)})"


def _one_domain(table: Any) -> Any:
    """A one-domain suite's table (its global ids are the table's), else the argument."""
    if isinstance(table, TaskSuite) and table.num_domains == 1:
        return table.tables[0]
    return table


def _searches(run: Callable[..., Any], task: Any, starts: Any, task_ids: Any, goals: Any, device_rows: bool,
              kwargs: dict[str, Any]) -> Any:
    task = _one_domain(task)
    if not isinstance(task, TaskSuite):
        return run(task, starts, task_ids=task_ids, goals=goals, **kwargs)
    split = _Split(task, task_ids, _rows(starts))
    if goals is not None and len(goals) != len(split.ids):
        raise ValueError(f"mymyr: {len(goals)} goals for {len(split.ids)} searches (pass None, or one per search)")
    parts = []
    for d, idx, local in split.parts:
        table = task.tables[d]
        sub_goals = None if goals is None else [goals[i] for i in idx]
        sub = _starts_of(starts, idx, table.words if device_rows else None)
        parts.append(run(table, sub, task_ids=local, goals=sub_goals, **kwargs))
    return SuiteIwBatch(task, split, parts)


def multi_iw(run: Callable[..., Any], task: Any, starts: Any, *, task_ids: Any = None, goals: Any = None,
             **kwargs: Any) -> Any:
    """mymyr.cuda.multi_iw over a TaskSuite: search i on global instance task_ids[i] (a SuiteIwBatch)."""
    return _searches(run, task, starts, task_ids, goals, False, kwargs)


def batched_iw1(run: Callable[..., Any], task: Any, starts: Any, *, task_ids: Any = None, goals: Any = None,
                **kwargs: Any) -> Any:
    """mymyr.cuda.batched_iw1 over a TaskSuite (device starts: rows of the suite's width, cut to each domain's)."""
    return _searches(run, task, starts, task_ids, goals, True, kwargs)


def _spaces(run: Callable[..., Any], table: Any, kwargs: dict[str, Any]) -> list[Any]:
    table = _one_domain(table)
    if not isinstance(table, TaskSuite):
        return run(table, **kwargs)
    out: list[Any] = [None] * len(table)
    for d, t in enumerate(table.tables):
        for k, r in enumerate(run(t, **kwargs)):
            out[table.global_id(d, k)] = r
    return out


def state_spaces(run: Callable[..., Any], table: Any, **kwargs: Any) -> list[Any]:
    """mymyr.cuda.state_spaces over a TaskSuite: space i is global instance i's (its domain: suite.domain_of[i]); each
    domain's instances in one device pipeline, the domains one after the other."""
    return _spaces(run, table, kwargs)


def generate_state_spaces(run: Callable[..., Any], table: Any, **kwargs: Any) -> list[Any]:
    """mymyr.cuda.generate_state_spaces over a TaskSuite (results in global order)."""
    return _spaces(run, table, kwargs)
