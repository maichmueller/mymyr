"""Plans replayed into state rows and action labels, for behaviour cloning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mymyr._core import Action, Task, TaskHandle
    from mymyr._core._rl import Expansion, TaskSuite, TaskTable

__all__ = ["PlanRows", "plan_rows"]

ActionLike = Any  # an Action, a label (schema, binding) or a line of IPC plan text


@dataclass(frozen=True)
class PlanRows:
    """The states a plan visits and the action it takes in each, as ``datasets.plan_rows`` replays it. ``L`` is the
    number of steps; the rows have one more entry than the steps with ``goal_state=True`` (the state the plan reaches).

    - ``rows``: uint64 ``[L(+1), W]``, the states in the source's row layout (``task.encode`` for a Task, the table's
      layout of instance ``task_id`` otherwise); row ``i`` is the state step ``i`` is taken in;
    - ``task_id``: the instance of the rows (0 for a Task source);
    - ``schema``: int32 ``[L]`` and ``binding``: int32 ``[L, B]`` (``B`` the source's ``label_width``, -1 padded): each
      step's action label, as ``rl.expand`` lists it;
    - ``index``: int32 ``[L]``, each step's position in the canonical successor order of its state: the environment
      action that takes the step, and ``rl.expand(..., drop_noops=True).index`` of its row when that row is kept (in
      a padded expansion, ``pad().canonical``);
    - ``noop``: bool ``[L]``, the step's successor equals its state (a no-op, dropped by ``drop_noops`` unless every
      action of the state is one);
    - ``num_actions``: int32 ``[L(+1)]``, the number of successors of each row's state (the length of the order
      ``index`` points into);
    - ``equivalent``: int32 ``[L, K]`` (-1 padded), per step the canonical indices of the actions the caller declared
      equivalent to the plan's, the plan's own index first; ``None`` without ``equivalent=``;
    - ``goal_reached``: the state after the last step satisfies the goal;
    - ``source``: the Task (a TaskHandle's Task), TaskTable or TaskSuite the plan was replayed on.
    """

    rows: np.ndarray
    task_id: int
    schema: np.ndarray
    binding: np.ndarray
    index: np.ndarray
    noop: np.ndarray
    num_actions: np.ndarray
    goal_reached: bool
    source: Any
    equivalent: np.ndarray | None = None

    @property
    def num_steps(self) -> int:
        """``L``, the number of actions of the plan."""
        return len(self.index)

    @property
    def goal_state(self) -> bool:
        """Whether ``rows`` ends with the state the plan reaches (``plan_rows(..., goal_state=True)``)."""
        return len(self.rows) > len(self.index)

    def equivalent_actions(self, width: int | None = None) -> np.ndarray:
        """The multi-hot form of ``equivalent``: bool ``[L(+1), width]``, entry ``[i, j]`` set when the action at
        canonical position ``j`` of row ``i``'s state is the plan's or one declared equivalent to it (a goal-state row
        has none). ``width`` defaults to the largest ``num_actions``."""
        if self.equivalent is None:
            raise ValueError("mymyr: datasets: the plan rows have no equivalent actions (plan_rows(..., equivalent=))")
        width = int(self.num_actions.max(initial=0)) if width is None else int(width)
        if width < int(self.num_actions.max(initial=0)):
            raise ValueError(
                f"mymyr: datasets: width {width} is smaller than a state's {self.num_actions.max()} actions")
        hot = np.zeros((len(self.rows), width), dtype=bool)
        steps, slots = np.nonzero(self.equivalent >= 0)
        hot[steps, self.equivalent[steps, slots]] = True
        return hot


def _source(source: Any, task_id: int) -> tuple[Task | TaskHandle, list[int] | None]:
    """The task of the plan and the ``task_ids`` argument of ``rl`` calls on the source."""
    from mymyr._core import Task, TaskHandle
    from mymyr._core._rl import TaskSuite, TaskTable

    if isinstance(source, (Task, TaskHandle)):
        if task_id != 0:
            raise ValueError(
                f"mymyr: datasets: a Task source is instance 0 (task_id={task_id}); pass a TaskTable or TaskSuite for "
                "rows of several instances")
        task, task_ids = source, None
    elif isinstance(source, (TaskTable, TaskSuite)):
        if not 0 <= task_id < len(source):
            raise IndexError(f"mymyr: datasets: task_id {task_id} of a source of {len(source)} instances")
        task, task_ids = source.tasks[task_id], [task_id]
    else:
        raise TypeError("mymyr: datasets: plan_rows takes a Task, TaskHandle, TaskTable or TaskSuite")
    if task.options["atoms"] == "lazy":
        raise ValueError(
            "mymyr: datasets: the task has lazy atom slots, whose row layout grows while states are expanded; replay "
            "the plan on task.frozen() (fixed slots) or build the task with atoms 'auto', 'reachable' or 'dense'")
    return task, task_ids


def same_source(a: Any, b: Any) -> bool:
    """Whether two plan sources are one: the same Task, or the same TaskTable or TaskSuite object."""
    from mymyr._core import Task

    def key(source: Any) -> Any:
        return ("task", source.uid) if isinstance(source, Task) else ("table", id(source))

    return key(a) == key(b)


def _plan_actions(task: Task | TaskHandle, plan: str | Sequence[Action]) -> list[Action]:
    from mymyr import search
    from mymyr._core import Action

    if isinstance(plan, str):
        return list(search.parse_plan(task, plan))
    actions = list(plan)
    for step, action in enumerate(actions, 1):
        if not isinstance(action, Action):
            raise TypeError(f"mymyr: datasets: plan step {step} is a {type(action).__name__}, not an Action "
                            "(a plan is IPC plan text or a list of Actions)")
        if action.task.uid != task.uid:
            raise ValueError(f"mymyr: datasets: plan step {step} ({action}) is an action of another task "
                             f"({action.task}), not of {task}")
    return actions


def _rows_of(exp: Expansion, schema: int, binding: Sequence[int]) -> np.ndarray:
    """The rows of an expansion whose label is ``(schema, binding)`` (a binding -1 padded or of the schema's arity)."""
    labels = np.asarray(exp.binding)
    width = labels.shape[1]
    b = np.asarray(binding, dtype=np.int64)
    if b.ndim != 1 or len(b) > width:
        return np.empty(0, np.int64)
    match = (np.asarray(exp.schema) == schema) & np.all(labels[:, : len(b)] == b, axis=1)
    if len(b) < width:
        match &= np.all(labels[:, len(b):] == -1, axis=1)
    return np.flatnonzero(match)


class _Step:
    """The expansions of one state: the canonical one that locates actions and successors and, under witness
    pruning, the pruned one whose order ``index`` refers to."""

    def __init__(self, source: Any, row: np.ndarray, task_ids: list[int] | None, witness_pruning: bool) -> None:
        from mymyr import rl

        self.full = rl.expand(source, row, task_ids, noops=True)
        self.pruned = rl.expand(source, row, task_ids, witness_pruning=True) if witness_pruning else None

    @property
    def num_actions(self) -> int:
        return int((self.full if self.pruned is None else self.pruned).total)

    def locate(self, schema: int, binding: Sequence[int]) -> tuple[int, int]:
        """The row of the label in the canonical expansion and its position in the order ``index`` refers to; ``(-1,
        -1)`` when the state has no such applicable action. Under witness pruning an action without a row of its own
        is at its witness: the first row of the same schema with the same successor."""
        rows = _rows_of(self.full, schema, binding)
        if len(rows) == 0:
            return -1, -1
        row = int(rows[0])
        if self.pruned is None:
            return row, row
        own = _rows_of(self.pruned, schema, binding)
        if len(own):
            return row, int(own[0])
        succ = np.asarray(self.full.succ)[row]
        same = np.flatnonzero((np.asarray(self.pruned.schema) == schema)
                              & np.all(np.asarray(self.pruned.succ) == succ, axis=1))
        return row, int(same[0]) if len(same) else -1


def _label(task: Task | TaskHandle, entry: ActionLike, state_row: np.ndarray, task_ids: list[int] | None,
           source: Any) -> tuple[int, Sequence[int]]:
    """The label ``(schema, binding)`` of an equivalent-action entry."""
    from mymyr import rl, search
    from mymyr._core import Action

    if isinstance(entry, Action):
        if entry.task.uid != task.uid:
            raise ValueError(f"an action of another task ({entry.task})")
        return int(entry.schema), tuple(entry.binding)
    if isinstance(entry, str):
        state = rl.decode(source, state_row, task_ids)[0]
        actions = search.parse_plan(task, entry, start=state)
        if len(actions) != 1:
            raise ValueError(f"{len(actions)} actions in '{entry}' (one plan line names one action)")
        return int(actions[0].schema), tuple(actions[0].binding)
    try:
        schema, binding = entry
        return int(schema), tuple(int(o) for o in binding)
    except (TypeError, ValueError):
        raise TypeError(f"a {type(entry).__name__}; an equivalent action is an Action, a label (schema, binding) or "
                        "a plan line") from None


def plan_rows(
    source: Task | TaskHandle | TaskTable | TaskSuite,
    plan: str | Sequence[Action],
    *,
    task_id: int = 0,
    goal_state: bool = False,
    witness_pruning: bool = False,
    equivalent: Sequence[Sequence[ActionLike]] | None = None,
) -> PlanRows:
    """The rows of the states a plan visits and its action labels, for behaviour cloning (see ``PlanRows``).

    ``source`` is a Task (or TaskHandle), or a TaskTable or TaskSuite with the plan's instance ``task_id``; ``plan``
    is IPC plan text (read by ``search.parse_plan``: of the applicable actions a line matches, the first in canonical
    order) or a list of the task's Actions. Each step is located in the host expansion of its state, ``rl.expand(source,
    row, ...)`` with ``witness_pruning``, so ``index`` is the action an environment with the same option takes; under
    witness pruning a step without a row of its own is at its witness (its schema and successor), and its label stays
    the plan's. ``goal_state=True`` appends the state the plan reaches. ``equivalent``, one list per step, names the
    actions the caller regards as equivalent to the step's (each an Action, a label ``(schema, binding)`` or a plan
    line, as from per-step symmetry orbits); they become ``equivalent`` canonical indices, without duplicates.

    Raises ValueError, naming the step, when an action is not applicable in its state or belongs to another task, when
    an equivalent action is not applicable, and for a task with lazy atom slots. Cost: one host expansion per step
    (two with witness pruning).
    """
    from mymyr import rl
    from mymyr._core import TaskHandle

    if isinstance(source, TaskHandle):
        source = source.task  # the replay is the same through the shared Task
    task, task_ids = _source(source, task_id)
    actions = _plan_actions(task, plan)
    if equivalent is not None and len(equivalent) != len(actions):
        raise ValueError(f"mymyr: datasets: {len(equivalent)} lists of equivalent actions for {len(actions)} steps")
    if task_ids is None:
        row = np.ascontiguousarray(np.asarray(task.encode([task.initial_state]), dtype=np.uint64))
    else:
        row = np.ascontiguousarray(np.asarray(source.initial_states(task_ids), dtype=np.uint64))
    width = int(source.label_width)
    n = len(actions)
    rows = [row[0]]
    schema = np.empty(n, np.int32)
    binding = np.full((n, width), -1, np.int32)
    index = np.empty(n, np.int32)
    noop = np.empty(n, bool)
    num_actions = np.empty(n + 1, np.int32)
    picked: list[list[int]] = []
    for i, action in enumerate(actions):
        step = _Step(source, row, task_ids, witness_pruning)
        label = (int(action.schema), tuple(action.binding))
        at, position = step.locate(*label)
        if position < 0:
            raise ValueError(f"mymyr: datasets: plan step {i + 1} ({action}) is not applicable in its state")
        schema[i] = label[0]
        binding[i, : len(label[1])] = label[1]
        index[i] = position
        noop[i] = bool(np.asarray(step.full.noop)[at])
        num_actions[i] = step.num_actions
        if equivalent is not None:
            same = [position]
            for entry in equivalent[i]:
                try:
                    other = _label(task, entry, row, task_ids, source)
                except (TypeError, ValueError) as e:
                    raise type(e)(f"mymyr: datasets: equivalent action {entry!r} of plan step {i + 1}: {e}") from None
                found = step.locate(*other)[1]
                if found < 0:
                    raise ValueError(f"mymyr: datasets: equivalent action {entry!r} of plan step {i + 1} is not "
                                     "applicable in the step's state")
                if found not in same:
                    same.append(found)
            picked.append(same)
        # a copy, so that no row keeps its expansion alive
        row = np.array(np.asarray(step.full.succ)[at : at + 1], dtype=np.uint64)
        rows.append(row[0])
    goal_reached = bool(np.asarray(rl.is_goal(source, row, task_ids))[0])
    if goal_state:
        num_actions[n] = rl.expand(source, row, task_ids, witness_pruning=witness_pruning).total
    else:
        rows.pop()
        num_actions = num_actions[:n]
    stacked = np.stack(rows) if rows else np.empty((0, row.shape[1]), np.uint64)
    eq = None
    if equivalent is not None:
        eq = np.full((n, max((len(p) for p in picked), default=1)), -1, np.int32)
        for k, p in enumerate(picked):
            eq[k, : len(p)] = p
    return PlanRows(stacked, task_id, schema, binding, index, noop, num_actions, goal_reached, source, eq)
