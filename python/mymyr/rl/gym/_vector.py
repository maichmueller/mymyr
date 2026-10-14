"""VectorEnv: N planning environments as a gymnasium.vector.VectorEnv, stepped natively by one CpuEnvPool."""

from typing import Any, Optional, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from mymyr._core import Action, State
from mymyr._core._rl import CpuEnvPool

from ._core import (
    Draws,
    GroundActionSpace,
    Problems,
    SymbolicObserver,
    SymbolicSpace,
    as_int_ids,
    check_action_mode,
    check_seed,
    env_config,
    ground_index,
    make_observer,
    render_text,
)

__all__ = ["VectorEnv"]

_NO_CAP = 2**31 - 1
_MODES = {
    "next_step": AutoresetMode.NEXT_STEP,
    "same_step": AutoresetMode.SAME_STEP,
    "disabled": AutoresetMode.DISABLED,
}


class VectorEnv(gymnasium.vector.VectorEnv):
    """``num_envs`` planning environments as one gymnasium.vector.VectorEnv, stepped by one native
    ``mymyr.rl.CpuEnvPool`` (``threads`` worker threads; 0: the hardware's) instead of N Python environments.

    The problems, observation and action modes and keywords are those of :class:`Env`, batched:

    - ``"symbolic"``: a tuple of N SymbolicObservations (``observation_space``: ``batch_space`` of the
      SymbolicSpace, a Tuple);
    - ``"bits"``: int8 [N, n] (and float64 [N, values] for numeric tasks);
    - ``"graph"``: one FlatBatch of the encoder with the N graphs (entity ids global over the batch,
      ``entity_offsets`` [N + 1], ``relation_offsets`` [N * R + 1]): the batched graph a message-passing model takes;
    - actions ``"index"``: int [N] (``MultiDiscrete``), ``info["action_mask"]`` int8 [N, max_actions]; ``"ground"``:
      a sequence of N Actions (each env's GroundActionSpace in ``action_space``, a Tuple, samples its own state's).

    Env i is env ``first_env + i`` of the RL contract's RNG and instance streams: ``reset(seed=s)`` gives every env
    the seed s (a list gives one seed per env), so the instances equal those of ``CpuEnvPool(seed=s)``'s envs and of
    single :class:`Env` with ``env_id = first_env + i`` reset with s.

    Autoreset (``autoreset_mode``, ``metadata["autoreset_mode"]``), as Gymnasium defines it:

    - ``"next_step"`` (default, Gymnasium's default): a step that ends an episode returns its final observation; the
      env's next step ignores its action and returns the reset observation with reward 0 and no termination;
    - ``"same_step"``: the native autoreset (one pool call per step): an ending step returns the reset observation,
      and ``info["final_obs"]`` / ``info["final_info"]`` (with the masks ``_final_obs`` / ``_final_info``) the final
      ones;
    - ``"disabled"``: finished envs stay in their terminal states (absorbing, per the contract) until
      ``reset(options={"reset_mask": mask})``.

    ``reset(options=...)``: ``"task_ids"`` (the instances of the reset envs) and ``"reset_mask"`` (bool [N]: reset only
    these envs; the others keep their episodes). :meth:`set_next_task_ids` names the instances of the next
    autoresets (a curriculum); :meth:`fix_problem_index` fixes the instance of every episode. ``info``: per-env arrays
    ``task_id``, ``num_actions``, ``is_goal``, ``steps``, ``action_mask`` (index actions), ``invalid`` (steps) and
    ``terminated`` (resets: the reset envs' initial states are terminal), each with its ``_key`` mask.
    ``max_episode_steps`` (what ``gymnasium.make_vec`` passes for a registered time limit) truncates natively, like
    ``max_steps`` (the smaller one when both are given).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4, "autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        table: Any = None,
        num_envs: int = 1,
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
        first_env: int = 0,
        autoreset_mode: Union[str, AutoresetMode] = "next_step",
        threads: int = 0,
        max_steps: int = 0,
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
        dead_end: str = "no_successors",
        dead_end_reward: float = 0.0,
        dead_end_terminal: bool = True,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"mymyr: render_mode must be None or 'ansi', not {render_mode!r}")
        if max_episode_steps is not None:  # gymnasium.make_vec's time limit: the native truncation
            max_steps = int(max_episode_steps) if max_steps == 0 else min(int(max_steps), int(max_episode_steps))
        if isinstance(num_envs, (bool, np.bool_)) or not isinstance(num_envs, (int, np.integer)) or num_envs < 1:
            raise ValueError("mymyr: num_envs must be a positive int")
        self.num_envs = int(num_envs)
        if isinstance(autoreset_mode, AutoresetMode):
            mode = autoreset_mode
        elif autoreset_mode in _MODES:
            mode = _MODES[autoreset_mode]
        else:
            raise ValueError(f"mymyr: autoreset_mode must be one of {tuple(_MODES)} or an AutoresetMode")
        self.metadata = {**type(self).metadata, "autoreset_mode": mode}
        self._mode = mode
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
        first = check_seed(first_env)
        n = self.num_envs
        self._draws = Draws(self._problems, sampler, np.arange(first, first + n, dtype=np.uint64))
        words = self._problems.words or None
        self._pool: Optional[CpuEnvPool] = CpuEnvPool(
            self._problems.table, n, threads=threads, autoreset=mode == AutoresetMode.SAME_STEP, sampler=None,
            words=words,
            **env_config(max_steps, step_reward, goal_reward, dead_end, dead_end_reward, dead_end_terminal),
        )  # fmt: skip

        if isinstance(self._observer, SymbolicObserver):
            self.single_observation_space = SymbolicSpace(self._observer)
            self.observation_space = batch_space(self.single_observation_space, n)
        elif observation == "graph":
            self.single_observation_space = self._observer.space(1)
            self.observation_space = self._observer.space(n)
        else:
            self.single_observation_space = self._observer.space()
            self.observation_space = self._observer.space(n)
        if self._action_mode == "index":
            self.single_action_space = spaces.Discrete(self.max_actions)
            self.action_space = batch_space(self.single_action_space, n)
        else:
            self.single_action_space = GroundActionSpace(self._problems, lambda: self._current_actions(0))
            self.action_space = spaces.Tuple(
                [GroundActionSpace(self._problems, (lambda i: lambda: self._current_actions(i))(i)) for i in range(n)]
            )

        rw = self._problems.row_words
        self._rows = np.zeros((n, rw), np.uint64)
        self._task_ids = np.zeros(n, np.int32)
        self._count = np.zeros(n, np.int32)
        self._steps = np.zeros(n, np.int32)
        self._at_goal = np.zeros(n, np.bool_)
        self._autoreset = np.zeros(n, np.bool_)
        self._next_task_ids: Optional[np.ndarray] = None
        self._states: list[Optional[State]] = [None] * n
        self._actions: list[Optional[list[Action]]] = [None] * n
        self._started = False

    # ------------------------------------------------------------------------------------------ properties

    @property
    def table(self) -> Any:
        return self._problems.table

    @property
    def num_instances(self) -> int:
        return len(self._problems)

    @property
    def task_ids(self) -> np.ndarray:
        """The envs' current instances, int32 [N] (a copy)."""
        return self._task_ids.copy()

    def states(self) -> list[State]:
        """The envs' current states (mymyr States of their instances' Tasks)."""
        self._require_reset()
        return [self._state(i) for i in range(self.num_envs)]

    def applicable_actions(self, env: int) -> list[Action]:
        """The applicable actions of env ``env``'s state in canonical order (its index actions)."""
        return self._current_actions(int(env))[1]

    def action_masks(self) -> np.ndarray:
        """bool [N, max_actions]: the valid index actions."""
        self._require_reset()
        if self.max_actions is None:
            raise ValueError("mymyr: action masks belong to the index action mode")
        return np.arange(self.max_actions)[None, :] < self._count[:, None]

    def fix_problem_index(self, task_id: Optional[int]) -> None:
        """Starts every following episode of every env in instance ``task_id`` (None: draw them again)."""
        self._draws.fixed = None if task_id is None else self._problems.check_task_id(task_id)

    def set_next_task_ids(self, task_ids: Optional[Any]) -> None:
        """The instances [N] the envs' next autoresets start in (a curriculum; used once, then drawn again)."""
        self._next_task_ids = (
            None if task_ids is None else as_int_ids(task_ids, self.num_envs, self._problems, "next task ids")
        )

    # ------------------------------------------------------------------------------------------ gymnasium API

    def reset(self, *, seed: Any = None, options: Optional[dict] = None) -> tuple[Any, dict]:
        n = self.num_envs
        mask = np.ones(n, np.bool_)
        task_ids = None
        if options:
            unknown = set(options) - {"task_ids", "reset_mask"}
            if unknown:
                raise ValueError(f"mymyr: unknown reset options {sorted(unknown)} (the options are 'task_ids' and "
                                 "'reset_mask')")
            if options.get("reset_mask") is not None:
                mask = np.asarray(options["reset_mask"])
                if mask.shape != (n,) or mask.dtype != np.bool_:
                    raise ValueError(f"mymyr: reset_mask must be bool [{n}]")
                if not self._started and not mask.all():
                    raise ValueError("mymyr: the first reset resets every env")
            if options.get("task_ids") is not None:
                task_ids = as_int_ids(options["task_ids"], n, self._problems, "task_ids")
        envs = np.flatnonzero(mask)
        if isinstance(seed, (list, tuple, np.ndarray)):
            if len(seed) != n:
                raise ValueError(f"mymyr: a seed list needs {n} seeds")
            super().reset(seed=None)
            self._draws.seed(envs, np.array([check_seed(s) for s in seed], dtype=object)[envs])
        elif seed is not None:
            super().reset(seed=check_seed(seed))
            self._draws.seed(envs, int(seed))
        elif not self._draws.seeded:
            self._draws.seed(np.arange(n), int(self.np_random.integers(0, 2**63)))
        terminal = np.zeros(n, np.bool_)
        if len(envs):
            ids = self._draws.peek(envs)
            if task_ids is not None:
                ids = task_ids[envs]
            self._draws.start(envs, ids)
            batch = self._open_pool().reset(env_ids=envs.astype(np.int32), task_ids=ids)
            terminal[envs] = batch.terminated
            self._capture(envs, batch)
            self._autoreset[envs] = False
        self._started = True
        infos = self._infos()
        infos["terminated"], infos["_terminated"] = terminal, mask.copy()
        return self._observe(), infos

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
        self._require_reset()
        n = self.num_envs
        index = self._indices(actions)
        pool = self._open_pool()
        rewards = np.zeros(n, np.float64)
        terminated = np.zeros(n, np.bool_)
        truncated = np.zeros(n, np.bool_)
        invalid = np.zeros(n, np.bool_)
        final = None
        if self._mode == AutoresetMode.SAME_STEP:
            everyone = np.arange(n)
            nxt = self._draws.peek(everyone) if self._next_task_ids is None else self._next_task_ids
            self._next_task_ids = None
            old_ids = self._task_ids.copy()
            batch = pool.step(actions=index, next_task_ids=nxt)
            rewards[:] = batch.reward
            terminated[:] = batch.terminated
            truncated[:] = batch.truncated
            invalid[:] = batch.invalid
            done = np.flatnonzero(terminated | truncated)
            if len(done):
                self._draws.start(done, nxt[done])
                final = (done, np.array(batch.final_states)[done], old_ids[done], np.array(batch.goal)[done])
            self._capture(everyone, batch)
        else:
            resetting = self._autoreset if self._mode == AutoresetMode.NEXT_STEP else np.zeros(n, np.bool_)
            active = np.flatnonzero(~resetting)
            if len(active):
                batch = pool.step(env_ids=active.astype(np.int32), actions=index[active])
                rewards[active] = batch.reward
                terminated[active] = batch.terminated
                truncated[active] = batch.truncated
                invalid[active] = batch.invalid
                self._capture(active, batch)
            restart = np.flatnonzero(resetting)
            if len(restart):
                ids = self._draws.peek(restart)
                if self._next_task_ids is not None:
                    ids = self._next_task_ids[restart]
                self._next_task_ids = None
                self._draws.start(restart, ids)
                self._capture(restart, pool.reset(env_ids=restart.astype(np.int32), task_ids=ids))
            if self._mode == AutoresetMode.NEXT_STEP:
                self._autoreset = terminated | truncated
        infos = self._infos()
        infos["invalid"] = invalid
        infos["_invalid"] = np.ones(n, np.bool_)
        if final is not None:
            self._final_infos(infos, *final)
        return self._observe(), rewards, terminated, truncated, infos

    def render(self) -> Optional[tuple[str, ...]]:
        if self.render_mode != "ansi":
            return None
        self._require_reset()
        return tuple(
            render_text(self._problems, int(self._task_ids[i]), self._state(i), int(self._steps[i]), int(self._count[i]))
            for i in range(self.num_envs)
        )

    def close_extras(self, **kwargs: Any) -> None:
        self._pool = None

    # ------------------------------------------------------------------------------------------ internals

    def _open_pool(self) -> CpuEnvPool:
        if self._pool is None:
            raise RuntimeError("mymyr: the environment is closed")
        return self._pool

    def _require_reset(self) -> None:
        if not self._started:
            raise RuntimeError("mymyr: call reset() before step()")

    def _capture(self, envs: np.ndarray, batch: Any) -> None:
        self._rows[envs] = batch.states
        self._task_ids[envs] = batch.task_ids
        self._count[envs] = batch.count
        self._steps[envs] = batch.steps
        self._at_goal[envs] = batch.at_goal
        for e in envs.tolist():
            self._states[e] = None
            self._actions[e] = None

    def _state(self, i: int) -> State:
        state = self._states[i]
        if state is None:
            state = self._problems.decode(self._rows[i : i + 1], self._task_ids[i : i + 1])[0]
            self._states[i] = state
        return state

    def _decode_all(self) -> list[State]:
        missing = [i for i, s in enumerate(self._states) if s is None]
        if missing:
            where = np.asarray(missing)
            for i, state in zip(missing, self._problems.decode(self._rows[where], self._task_ids[where])):
                self._states[i] = state
        return self._states  # type: ignore[return-value]

    def _current_actions(self, i: int) -> tuple[int, list[Action]]:
        self._require_reset()
        if not 0 <= i < self.num_envs:
            raise IndexError(f"mymyr: env {i} is outside the {self.num_envs} envs")
        actions = self._actions[i]
        t = int(self._task_ids[i])
        if actions is None:
            actions = self._problems.tasks[t].applicable_actions(self._state(i))
            self._actions[i] = actions
        return t, actions

    def _indices(self, actions: Any) -> np.ndarray:
        n = self.num_envs
        if self._action_mode == "ground":
            if isinstance(actions, (str, Action)) or len(actions) != n:
                raise ValueError(f"mymyr: ground actions must be a sequence of {n} actions")
            out = np.empty(n, np.int64)
            for i, action in enumerate(actions):
                t, applicable = self._current_actions(i)
                out[i] = ground_index(self._problems, t, applicable, action, _NO_CAP)
            return out
        index = np.asarray(actions)
        if index.shape != (n,) or not np.issubdtype(index.dtype, np.integer):
            raise ValueError(f"mymyr: index actions must be {n} ints")
        index = index.astype(np.int64)
        return np.where((index >= 0) & (index < self.max_actions), index, -1)

    def _observe(self) -> Any:
        if isinstance(self._observer, SymbolicObserver):
            states = self._decode_all()
            return tuple(self._observer.observe(states[i], int(self._task_ids[i])) for i in range(self.num_envs))
        return self._observer.observe(self._rows, self._task_ids)

    def _observe_rows(self, rows: np.ndarray, task_ids: np.ndarray) -> list:
        if isinstance(self._observer, SymbolicObserver):
            states = self._problems.decode(rows, task_ids)
            return [self._observer.observe(s, int(t)) for s, t in zip(states, task_ids)]
        if self._observation_mode == "graph":
            return [self._observer.observe(rows[i : i + 1], task_ids[i : i + 1]) for i in range(len(rows))]
        out = self._observer.observe(rows, task_ids)
        if isinstance(out, dict):
            return [{key: value[i] for key, value in out.items()} for i in range(len(rows))]
        return list(out)

    def _infos(self) -> dict:
        n = self.num_envs
        everyone = np.ones(n, np.bool_)
        infos = {
            "task_id": self._task_ids.copy(), "_task_id": everyone,
            "num_actions": self._count.copy(), "_num_actions": everyone.copy(),
            "is_goal": self._at_goal.copy(), "_is_goal": everyone.copy(),
            "steps": self._steps.copy(), "_steps": everyone.copy(),
        }  # fmt: skip
        if self.max_actions is not None:
            infos["action_mask"] = (np.arange(self.max_actions)[None, :] < self._count[:, None]).astype(np.int8)
            infos["_action_mask"] = everyone.copy()
        return infos

    def _final_infos(self, infos: dict, done: np.ndarray, rows: np.ndarray, task_ids: np.ndarray,
                     goal: np.ndarray) -> None:
        n = self.num_envs
        mask = np.zeros(n, np.bool_)
        mask[done] = True
        final_obs = np.full(n, None, dtype=object)
        for i, obs in zip(done.tolist(), self._observe_rows(rows, task_ids)):
            final_obs[i] = obs
        final_task = np.zeros(n, np.int32)
        final_task[done] = task_ids
        final_goal = np.zeros(n, np.bool_)
        final_goal[done] = goal
        infos["final_obs"], infos["_final_obs"] = final_obs, mask
        infos["final_info"] = {"task_id": final_task, "_task_id": mask.copy(), "is_goal": final_goal,
                               "_is_goal": mask.copy()}
        infos["_final_info"] = mask.copy()
