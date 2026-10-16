"""Env: one planning environment as a gymnasium.Env."""

from typing import Any, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from mymyr._core import Action, State, Task
from mymyr._core._rl import CpuEnvPool

from ._core import (
    Draws,
    GroundActionSpace,
    Problems,
    SymbolicObserver,
    SymbolicSpace,
    check_action_mode,
    check_seed,
    env_config,
    ground_index,
    make_observer,
    render_text,
)

__all__ = ["Env"]

_NO_CAP = 2**31 - 1


class Env(gymnasium.Env):
    """One planning environment over a task, a table of instances or a suite (a gymnasium.Env).

    Problems: ``table`` is a Task (or TaskHandle), a ``mymyr.rl.TaskTable`` or a ``mymyr.rl.TaskSuite``; or
    ``domain=`` and ``problems=`` (a problem file, a directory of them, a glob or a list) build the table
    ``TaskTable.from_pddl(domain, problems, **task_options)``. Each episode starts in an instance drawn by ``sampler``
    (``"auto"`` / ``"uniform"``: uniform, an ``rl.InstanceSampler``: weighted, ``None``: the previous instance,
    instance 0 at first), in the instance of ``fix_problem_index(i)`` (pddlgym's fixed problem), or in
    ``reset(options={"task_id": i})``'s.

    Observations (``observation=``):

    - ``"symbolic"`` (default): a :class:`SymbolicObservation`, the state's true fluent atoms as a frozenset of
      mymyr GroundAtoms (``derived=True`` adds the derived atoms), and, as options, the instance's objects
      (``objects=True``, default), goal (``goal=True``, default), static atoms (``static=True``) and numeric values
      (``numeric=``, default: whether the task has numeric fluents); space: :class:`SymbolicSpace`.
    - ``"graph"``: the state's relational graph from ``mymyr.encoding.FlatRelationEncoder`` (its arrays for one graph as
      a dict; ``graph_options`` are the encoder's options, by default ``goal_derivations=()``,
      ``include_actions=False``, ``emit_counts=False``, ``entity_arrays="auto"``); space: a ``Dict`` of ``Box`` for the
      per-graph arrays and of ``Sequence(Discrete, stack=True)`` for the variable ones. Classical tasks of one domain.
    - ``"bits"``: the fluent atom slots of the state as int8 0/1 ``MultiBinary(n)`` (n: the most slots of an
      instance); numeric tasks add the values (``Dict(atoms=..., numeric=Box)``).

    Actions (``action=``):

    - ``"index"`` (default): ``Discrete(max_actions)``, an index into the state's applicable actions in canonical
      order (schema, then binding; :meth:`applicable_actions` lists them). ``info["action_mask"]`` (int8, the mask
      ``Discrete.sample(mask=...)`` takes) and :meth:`action_masks` (bool) mark the valid indices. ``max_actions``
      defaults to the largest :func:`mymyr.rl.successor_bounds` of the instances, so every applicable action of
      every reachable state has an index; a smaller value puts a state's actions past it out of reach
      (``info["num_actions"]`` counts them all).
    - ``"ground"``: the ground action itself, a mymyr Action of the current instance (or ``'(stack a b)'`` /
      ``(schema, objects)``, as ``Task.action`` takes); space: :class:`GroundActionSpace` (pddlgym's literal action
      space: ``sample()`` draws an applicable action).

    An action that is not applicable (or outside the space) does not move: ``info["invalid"]`` is set and the reward
    is ``step_reward``. In a state without applicable actions no action moves, and ``invalid`` stays unset. Rewards,
    terminations and truncations are those of ``mymyr.rl.CpuEnvPool`` (the RL contract, with the same keywords):
    ``reward = step_reward``, plus ``goal_reward`` on reaching a goal, plus ``dead_end_reward`` on reaching (or being
    stuck in) a dead end; ``terminated`` at a goal or a terminal dead end; ``truncated`` after ``max_steps`` steps of
    an episode (None: never). Goals and terminal dead ends are absorbing: a step there does not move, pays nothing and
    stays terminated. A reset into a goal state (a solved instance) reports ``info["is_goal"]``, and its first step is
    such an absorbing step.

    Seeding: ``reset(seed=s)`` seeds ``np_random`` and the instance draws: episode k after it starts in the sampler's
    draw for (s, ``env_id``, k), as env ``env_id`` of a ``CpuEnvPool(seed=s)`` or a torch ``BatchedEnv(seed=s)``,
    whose construction starts episode 0. The transitions themselves are deterministic, so a seed fixes the
    trajectory of a sequence of actions in every process.

    ``info``: ``task_id`` (the instance), ``num_actions`` (applicable actions of the observed state), ``is_goal``
    (the observed state is a goal), ``steps`` (the episode's steps), ``action_mask`` (index actions), ``invalid``
    (steps), ``terminated`` (resets: the initial state is terminal, a goal or a terminal dead end, so the first step
    is absorbing). ``render_mode="ansi"``: :meth:`render` returns the state's atoms and the goal as text.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        table: Any = None,
        *,
        domain: Any = None,
        problems: Any = None,
        task_options: Optional[dict] = None,
        observation: str = "symbolic",
        action: str = "index",
        objects: bool = True,
        goal: bool = True,
        static: bool = False,
        derived: bool = False,
        numeric: Optional[bool] = None,
        graph_options: Optional[dict] = None,
        max_actions: Optional[int] = None,
        sampler: Any = "auto",
        env_id: int = 0,
        max_steps: int | None = None,
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
        dead_end: str = "no_successors",
        dead_end_reward: float = 0.0,
        dead_end_terminal: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"mymyr: render_mode must be None or 'ansi', not {render_mode!r}")
        self._problems = Problems(table, domain, problems, task_options)
        self._observation_mode = observation
        self._observer = make_observer(
            self._problems, observation,
            dict(objects=objects, goal=goal, static=static, derived=derived, numeric=numeric), graph_options,
        )  # fmt: skip
        self._action_mode = check_action_mode(action)
        if self._action_mode == "index":
            self.max_actions = self._problems.max_actions() if max_actions is None else int(max_actions)
            if self.max_actions < 1:
                raise ValueError("mymyr: max_actions must be at least 1")
        elif max_actions is not None:
            raise ValueError("mymyr: max_actions applies to the index action mode")
        else:
            self.max_actions = None
        self.render_mode = render_mode
        self._env_id = check_seed(env_id)
        self._draws = Draws(self._problems, sampler, np.array([self._env_id], np.uint64))
        words = self._problems.words or None
        self._pool: Optional[CpuEnvPool] = CpuEnvPool(
            self._problems.table, 1, threads=1, autoreset=False, sampler=None, words=words,
            **env_config(max_steps, step_reward, goal_reward, dead_end, dead_end_reward, dead_end_terminal),
        )  # fmt: skip
        if isinstance(self._observer, SymbolicObserver):
            self.observation_space = SymbolicSpace(self._observer)
        elif observation == "graph":
            self.observation_space = self._observer.space(1)
        else:
            self.observation_space = self._observer.space()
        if self._action_mode == "index":
            self.action_space = spaces.Discrete(self.max_actions)
        else:
            self.action_space = GroundActionSpace(self._problems, self._current_actions)
        self._rows: Optional[np.ndarray] = None
        self._task_id = 0
        self._count = 0
        self._steps = 0
        self._at_goal = False
        self._state: Optional[State] = None
        self._actions: Optional[list[Action]] = None

    # ------------------------------------------------------------------------------------------ properties

    @property
    def table(self) -> Any:
        """The Task, TaskTable or TaskSuite of the instances."""
        return self._problems.table

    @property
    def num_instances(self) -> int:
        """The number of instances an episode can start in (task ids 0 .. num_instances - 1)."""
        return len(self._problems)

    @property
    def task_id(self) -> int:
        """The current instance."""
        return self._task_id

    @property
    def task(self) -> Task:
        """The current instance's Task."""
        return self._problems.tasks[self._task_id]

    @property
    def state(self) -> State:
        """The current state (a mymyr State of the current instance's Task)."""
        self._require_reset()
        if self._state is None:
            self._state = self._problems.decode(self._rows, np.array([self._task_id], np.int32))[0]
        return self._state

    @property
    def env_id(self) -> int:
        """This environment's index in the instance draws (env ``env_id`` of a ``CpuEnvPool`` with the same seed)."""
        return self._env_id

    def applicable_actions(self) -> list[Action]:
        """The applicable actions of the current state in canonical order: index i of the index action mode is
        entry i."""
        return self._current_actions()[1]

    def action_masks(self) -> np.ndarray:
        """The valid indices of the index action mode as bool [max_actions] (the method maskable policies call)."""
        self._require_reset()
        if self.max_actions is None:
            raise ValueError("mymyr: action masks belong to the index action mode")
        return np.arange(self.max_actions) < self._count

    def fix_problem_index(self, task_id: Optional[int]) -> None:
        """Starts every following episode in instance ``task_id`` (None: draw the instances again)."""
        self._draws.fixed = None if task_id is None else self._problems.check_task_id(task_id)

    # ------------------------------------------------------------------------------------------ gymnasium API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Any, dict]:
        """Starts an episode: in ``options["task_id"]`` if given, else in the next drawn instance (see the class)."""
        super().reset(seed=seed)
        task_id = None
        if options:
            unknown = set(options) - {"task_id"}
            if unknown:
                raise ValueError(f"mymyr: unknown reset options {sorted(unknown)} (the option is 'task_id')")
            if options.get("task_id") is not None:
                task_id = self._problems.check_task_id(options["task_id"])
        envs = np.zeros(1, np.int64)
        if seed is not None:
            self._draws.seed(envs, check_seed(seed))
        elif not self._draws.seeded:
            self._draws.seed(envs, int(self.np_random.integers(0, 2**63)))
        ids = self._draws.peek(envs)
        if task_id is not None:
            ids[0] = task_id
        self._draws.start(envs, ids)
        batch = self._open_pool().reset(task_ids=ids)
        terminal = bool(batch.terminated[0])
        self._capture(batch)
        info = self._info()
        info["terminated"] = terminal
        return self._observe(), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Takes ``action`` (an index, or a ground action under ``action="ground"``) and returns (observation,
        reward, terminated, truncated, info). An action that is not applicable does not move and sets
        ``info["invalid"]``. Raises RuntimeError before the first reset."""
        self._require_reset()
        index = self._index(action)
        batch = self._open_pool().step(actions=np.array([index], np.int64))
        reward = float(batch.reward[0])
        terminated = bool(batch.terminated[0])
        truncated = bool(batch.truncated[0])
        invalid = bool(batch.invalid[0])
        self._capture(batch)
        info = self._info()
        info["invalid"] = invalid
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """The state's atoms and the goal as text under ``render_mode="ansi"``, else None."""
        if self.render_mode != "ansi":
            return None
        self._require_reset()
        return render_text(self._problems, self._task_id, self.state, self._steps, self._count)

    def close(self) -> None:
        """Releases the environment's pool; reset and step raise RuntimeError afterwards."""
        self._pool = None
        super().close()

    # ------------------------------------------------------------------------------------------ internals

    def _open_pool(self) -> CpuEnvPool:
        if self._pool is None:
            raise RuntimeError("mymyr: the environment is closed")
        return self._pool

    def _require_reset(self) -> None:
        if self._rows is None:
            raise RuntimeError("mymyr: call reset() before step()")

    def _capture(self, batch: Any) -> None:
        self._rows = np.array(batch.states)
        self._task_id = int(batch.task_ids[0])
        self._count = int(batch.count[0])
        self._steps = int(batch.steps[0])
        self._at_goal = bool(batch.at_goal[0])
        self._state = None
        self._actions = None

    def _current_actions(self) -> tuple[int, list[Action]]:
        self._require_reset()
        if self._actions is None:
            self._actions = self.task.applicable_actions(self.state)
        return self._task_id, self._actions

    def _index(self, action: Any) -> int:
        if self._action_mode == "ground":
            return ground_index(self._problems, self._task_id, self._current_actions()[1], action, _NO_CAP)
        if isinstance(action, np.ndarray) and action.shape == ():
            action = action.item()
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"mymyr: an index action must be an int, not {type(action).__name__}")
        index = int(action)
        return index if 0 <= index < self.max_actions else -1

    def _observe(self) -> Any:
        if isinstance(self._observer, SymbolicObserver):
            return self._observer.observe(self.state, self._task_id)
        ids = np.array([self._task_id], np.int32)
        out = self._observer.observe(self._rows, ids)
        if self._observation_mode == "graph":
            return out
        if isinstance(out, dict):
            return {key: value[0] for key, value in out.items()}
        return out[0]

    def _info(self) -> dict:
        info = {"task_id": self._task_id, "num_actions": self._count, "is_goal": self._at_goal, "steps": self._steps}
        if self.max_actions is not None:
            info["action_mask"] = (np.arange(self.max_actions) < self._count).astype(np.int8)
        return info
