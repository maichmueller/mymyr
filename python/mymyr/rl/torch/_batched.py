"""BatchedEnv: N planning environments whose state lives in torch tensors, stepped through the custom ops."""

from typing import Any, Callable, NamedTuple, Optional, Union

import torch
from torch import Tensor

from mymyr._core import _rl, _rl_torch
from mymyr.rl._sampling import _instance_sampler

from . import _ops
from ._graph import CapturedRollout

__all__ = ["BatchedEnv", "Observation", "StepResult"]


class Observation(NamedTuple):
    """Owned policy inputs by default; :meth:`BatchedEnv.observation` can explicitly borrow live tensors."""

    states: Tensor
    task_ids: Tensor
    count: Tensor
    goal_pos: Tensor
    goal_neg: Tensor
    at_goal: Tensor  # bool: the state is a goal (a solved reset, a goal mask that holds): the next step absorbs it


class StepResult(NamedTuple):
    """The outputs of one :meth:`BatchedEnv.step`, including a stable next observation.

    ``final_states`` precedes autoreset; ``states`` / ``task_ids`` / goal masks describe the next policy observation.
    Empty goal masks have shape [N, 0]. The native environment's public tensors remain borrowed live buffers.
    """

    reward: Tensor  # float32
    terminated: Tensor  # bool: goal reached, or a dead end (dead_end_terminal)
    truncated: Tensor  # bool: max_steps reached (not terminated)
    goal: Tensor  # bool: the reached state is a goal
    invalid: Tensor  # bool: the given action was outside [0, count)
    schema: Tensor  # int32: the label of the step's action (-1: no move)
    binding: Tensor  # [N, label_width] int32 objects of the row's instance (-1 padding)
    final_states: Tensor  # [N, row_words] int64 with final=True (the reached states before the autoreset), else [0, *]
    count: Tensor  # int32: successors of the state after the step (a copy)
    states: Tensor
    task_ids: Tensor
    goal_pos: Tensor
    goal_neg: Tensor
    at_goal: Tensor  # bool: the next observation's state is a goal (an autoreset into a solved initial state)


def _mark_static(t: Tensor) -> None:
    """What ``torch._dynamo.mark_static_address(t)`` does. Calling that would import torch._dynamo, and with it triton,
    which enables the GIL of a free-threaded interpreter, for every environment constructed."""
    t._dynamo_static_input_type = "unguarded"


def _device_index(device: torch.device) -> Optional[int]:
    if device.type == "cpu":
        return None
    if device.type != "cuda":
        raise ValueError(f"mymyr: environments run on the CPU or a CUDA device, not {device}")
    return device.index if device.index is not None else torch.cuda.current_device()


def _task_ids(core: Any, ids: Any, mask: Optional[Tensor], n: int, device: torch.device, name: str) -> Tensor:
    """``ids`` (one id for every row, or [n]) as [n] int32 on ``device``. Host values (ints, sequences, NumPy arrays,
    CPU tensors) are checked here: ValueError for an id outside the table's (suite's) instances in a row the call uses
    (``mask``: the masked rows). Device ids are checked on the device: a device environment raises ids outside them at
    :meth:`BatchedEnv.check_errors` and at the general path's next step or refresh (which synchronize)."""
    instances = core.num_instances
    t = torch.as_tensor(ids)
    if t.dim() == 0:
        t = t.expand(n)  # one id for every row
    if t.device.type == "cpu":
        flat = t.reshape(n).to(torch.int64)
        bad = (flat < 0) | (flat >= instances)
        if mask is not None:
            bad &= mask.reshape(n).to(device="cpu", dtype=torch.bool)
        if bool(bad.any()):
            r = int(bad.nonzero()[0])
            table = core.table
            noun = "suite" if isinstance(table, _rl.TaskSuite) and table.num_domains > 1 else "table"
            raise ValueError(
                f"mymyr: {name}: task id {int(flat[r])} of row {r} is outside the {noun}'s {instances} instances"
            )
    return t.reshape(n).to(device=device, dtype=torch.int32).contiguous()


def _where(mask: Optional[Tensor], new: Tensor, old: Tensor) -> Tensor:
    if mask is None:
        return new
    m = mask.reshape((-1,) + (1,) * (old.dim() - 1))
    return torch.where(m, new, old)


class BatchedEnv:
    """N environments over a TaskSuite or a TaskTable (or a Task), stepped in lockstep (see mymyr.rl.torch).

    The environments' state is held in tensors on ``device``: ``states`` [N, row_words] int64 (rows over a
    table are the table's words wide, over a suite the widest domain's), ``task_ids`` [N] int32 (each env's instance;
    global ids over a suite), ``steps`` [N] int32 (the episode's steps), ``draws`` [N] int64 (the RNG draw counters),
    ``count`` [N] int32 (successors of the current state), ``at_goal`` [N] bool (the current state is a goal: reset,
    step and refresh keep it, a step absorbs those rows), with ``goals=True`` the per-env goal masks ``goal_pos`` /
    ``goal_neg`` [N, words] int64 (the step's goal test is then the mask test; reset sets the instances' goals unless
    given), and, on a device's fast path, the cache ``counts`` / ``views``. The methods run the custom ops
    ``mymyr::reset`` / ``mymyr::step`` / ``mymyr::refresh`` on these tensors in place (on torch's current stream), so
    they trace under ``torch.compile``. Writing ``states``, ``task_ids`` or the goal masks directly needs
    :meth:`refresh`.

    ``sampler="auto"`` samples instances uniformly unless constructor ``task_ids`` pin the assignment. ``sampler=None``
    keeps instances, ``"uniform"`` always samples, and ``rl.InstanceSampler(I, weights)`` specifies a weighted
    curriculum. Sampling uses independent per-env reset counters, so partial resets and batch splits agree.
    ``autoreset=True`` restarts finished environments within the step in sampled instances or explicit
    ``next_task_ids[i]`` (``final=True`` returns the reached states); TorchRL's convention is ``autoreset=False`` plus
    ``reset(mask)``. ``step(action=None)`` is the random policy: a uniform successor from the counter-based RNG (seed,
    env id ``first_env + i``, draw counter), the same on the CPU and on any device and batch split.

    CUDA graphs: on a device's fast path (``capturable``) reset, refresh, step and random_actions synchronize nothing
    and allocate nothing (the environment's scratch is sized for N at construction), and the state tensors are marked
    static for torch.compile (they keep their addresses; assign into them in place). So loops of
    steps capture into CUDA graphs: :meth:`capture` (``torch.cuda.CUDAGraph``), ``torch.cuda.graph`` around the calls,
    or ``torch.compile(mode="reduce-overhead")`` of a function that steps. On the general path the compiled calls are
    the ``cudagraph_unsafe`` ops ``mymyr::step_sync`` / ``refresh_sync``, which torch.compile leaves out of its graphs.
    """

    def __init__(
        self,
        table: Any,
        num_envs: int,
        *,
        task_ids: Optional[Any] = None,
        sampler: Any = "auto",
        device: Union[str, torch.device] = "cpu",
        goals: bool = False,
        seed: int = 0,
        max_steps: int | None = None,
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
        dead_end: str = "no_successors",
        dead_end_reward: float = 0.0,
        dead_end_terminal: bool = True,
        autoreset: bool = True,
        witness_pruning: bool = False,
        path: str = "auto",
        threads: int = 1,
        ctx: Any = None,
        first_env: int = 0,
    ) -> None:
        self.device = torch.device(device)
        index = _device_index(self.device)
        if index is not None:
            self.device = torch.device("cuda", index)
        self.core = _rl_torch.Env(
            table, device=index, ctx=ctx, seed=seed, max_steps=max_steps, step_reward=step_reward,
            goal_reward=goal_reward, dead_end=dead_end, dead_end_reward=dead_end_reward,
            dead_end_terminal=dead_end_terminal, autoreset=autoreset, witness_pruning=witness_pruning, path=path,
            threads=threads,
        )  # fmt: skip
        self.handle = _ops.register(self.core)
        self.num_envs = int(num_envs)
        self.first_env = int(first_env)
        self.multi = self.core.num_instances > 1
        self.sampler = _instance_sampler(sampler, self.core.num_instances, default_uniform=task_ids is None)
        self._autoreset = bool(autoreset)
        n, d = self.num_envs, self.device
        self.states = torch.zeros((n, self.core.row_words), dtype=torch.int64, device=d)
        self.task_ids = torch.zeros(n, dtype=torch.int32, device=d)
        self._reset_counts = torch.zeros(n, dtype=torch.int64, device=d)
        self._sampling_seed = torch.tensor([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF], device=d)
        self._sampling_env_ids = torch.arange(self.first_env, self.first_env + n, dtype=torch.int64, device=d)
        self._sampling_arrays = None if self.sampler is None else self.sampler._arrays(self.task_ids)
        self.steps = torch.zeros(n, dtype=torch.int32, device=d)
        self.draws = torch.zeros(n, dtype=torch.int64, device=d)
        self.counts = torch.zeros((n, self.core.cache_schemas), dtype=torch.int32, device=d)
        self.views = torch.zeros((n, self.core.cache_view_words), dtype=torch.int64, device=d)
        self.count = torch.zeros(n, dtype=torch.int32, device=d)
        self.at_goal = torch.zeros(n, dtype=torch.bool, device=d)
        self.goal_pos: Optional[Tensor] = None
        self.goal_neg: Optional[Tensor] = None
        if goals:
            self.goal_pos = torch.zeros((n, self.core.words), dtype=torch.int64, device=d)
            self.goal_neg = torch.zeros((n, self.core.words), dtype=torch.int64, device=d)
        # CUDA graphs: the scratch of N rows, and the tensors the ops mutate at fixed addresses
        self.capturable = index is not None and not self.core.capture_unsupported
        if index is not None:
            self.core.reserve(n)
            for t in self._tensors():
                _mark_static(t)
        initial_ids = None if task_ids is None else _task_ids(self.core, task_ids, None, n, d, "task_ids")
        self.reset(task_ids=initial_ids)

    @property
    def table(self) -> Any:
        """The Task, TaskTable or TaskSuite of the instances."""
        return self.core.table

    @property
    def num_instances(self) -> int:
        """The number of instances an episode can start in (task ids 0 .. num_instances - 1)."""
        return self.core.num_instances

    @property
    def row_words(self) -> int:
        """Words per state row: the atom words, then the numeric words."""
        return self.core.row_words

    @property
    def words(self) -> int:
        """Atom words per state row (the table's words, at least 1)."""
        return self.core.words

    @property
    def label_width(self) -> int:
        """Columns of a step's binding output (the largest arity, at least 1)."""
        return self.core.label_width

    @property
    def goals(self) -> bool:
        """Whether the environments carry per-env goal masks (``goals=True``)."""
        return self.goal_pos is not None

    def _ids(self) -> Optional[Tensor]:
        return self.task_ids if self.multi else None

    def observation(self, *, copy: bool = True) -> Observation:
        """Policy inputs that survive later steps/resets by default. ``copy=False`` borrows the live buffers."""
        keep = (lambda tensor: tensor.clone()) if copy else (lambda tensor: tensor)
        empty = self.states.new_empty((self.num_envs, 0))
        return Observation(
            keep(self.states), keep(self.task_ids), keep(self.count),
            keep(self.goal_pos) if self.goals else empty, keep(self.goal_neg) if self.goals else empty,
            keep(self.at_goal),
        )

    def is_goal(self) -> Tensor:
        """Whether each environment's current state is a goal (the goal masks' test with goals=True): [N] bool on
        ``device``, tested on torch's current stream without stepping (``at_goal`` holds the same flags when the
        tensors were not written since the last reset, step or refresh)."""
        out = torch.empty(self.num_envs, dtype=torch.bool, device=self.device)
        self.core.goal(self.states, self._ids(), goal_pos=self.goal_pos, goal_neg=self.goal_neg, out=out,
                       stream=_ops._stream(self.states))
        return out

    def _tensors(self) -> list[Tensor]:
        t = [self.states, self.task_ids, self.steps, self.draws, self.counts, self.views, self.count, self.at_goal,
             self._reset_counts, self._sampling_seed, self._sampling_env_ids]
        return t + ([self.goal_pos, self.goal_neg] if self.goals else [])

    def _sample_task_ids(self) -> Tensor:
        return self.sampler.sample(
            self._sampling_seed, self._sampling_env_ids, self._reset_counts, _arrays=self._sampling_arrays
        )

    def _advance_resets(self, mask: Optional[Tensor] = None) -> None:
        if self.sampler is not None:
            increment = 1 if mask is None else mask.to(torch.int64)
            self._reset_counts.copy_((self._reset_counts + increment) & 0xFFFFFFFF)

    def reset(
        self,
        mask: Optional[Tensor] = None,
        *,
        task_ids: Optional[Tensor] = None,
        goal_pos: Optional[Tensor] = None,
        goal_neg: Optional[Tensor] = None,
    ) -> Tensor:
        """Resets all environments, or those with mask[i] ([N] bool), to sampled instances (their current instances
        with sampler=None); returns ``count``. ``task_ids`` [N] overrides sampling for reset rows; ``goal_pos`` /
        ``goal_neg`` [N, words] (goals=True) set the reset rows' goals (default: their instances' goals). ``at_goal``
        reports the reset states that are goals already (solved initial states, given goals that hold)."""
        if mask is not None:
            mask = mask.reshape(self.num_envs).to(device=self.device, dtype=torch.bool).contiguous()
        if task_ids is None and self.sampler is not None:
            new = self._sample_task_ids()
            self.task_ids.copy_(_where(mask, new, self.task_ids))
        elif task_ids is not None:
            new = _task_ids(self.core, task_ids, mask, self.num_envs, self.device, "reset")
            self.task_ids.copy_(_where(mask, new, self.task_ids))
        keep = goal_pos is not None or goal_neg is not None
        if keep:
            if not self.goals or goal_pos is None or goal_neg is None:
                raise ValueError("mymyr: per-env goals need goals=True and both goal_pos and goal_neg")
            self.goal_pos.copy_(_where(mask, goal_pos.to(self.goal_pos), self.goal_pos))
            self.goal_neg.copy_(_where(mask, goal_neg.to(self.goal_neg), self.goal_neg))
        args = (
            self.handle, self.states, self._ids(), self.steps, self.counts, self.views, self.goal_pos, self.goal_neg,
            self.count, self.at_goal, mask, keep,
        )  # fmt: skip
        if torch.compiler.is_compiling():
            torch.ops.mymyr.reset(*args)
        else:
            _ops.reset_impl(*args)  # eager: without the dispatcher's round trip
        self._advance_resets(mask)
        return self.count

    def refresh(self) -> Tensor:
        """Recomputes the cache, ``count`` and ``at_goal`` after the caller wrote ``states``, ``task_ids`` or the goal
        masks."""
        args = (
            self.handle, self.states, self._ids(), self.counts, self.views, self.goal_pos, self.goal_neg, self.count,
            self.at_goal,
        )  # fmt: skip
        if torch.compiler.is_compiling():
            (torch.ops.mymyr.refresh if self.capturable else torch.ops.mymyr.refresh_sync)(*args)
        else:
            _ops.refresh_impl(*args)
        return self.count

    def step(
        self, action: Optional[Tensor] = None, *, next_task_ids: Optional[Tensor] = None, final: bool = True,
        copy: bool = True,
    ) -> StepResult:
        """One step of every environment; ``action`` [N] int64 indexes the canonical successor order (None: the
        random policy); ``next_task_ids`` [N] overrides sampling for autoresetting environments (None: sample,
        or keep their instances with sampler=None). ``final=True`` retains the reached state before reset;
        ``copy=True`` owns the next observation. ``copy=False`` explicitly borrows live observation buffers."""
        args = self._step_args(action, next_task_ids, final)
        if torch.compiler.is_compiling():
            r = (torch.ops.mymyr.step if self.capturable else torch.ops.mymyr.step_sync)(*args)
        else:
            r = _ops.step_impl(*args)
        if self._autoreset:
            self._advance_resets(r[1] | r[2])
        observation = self.observation(copy=copy)
        return StepResult(*r, observation.count, observation.states, observation.task_ids,
                          observation.goal_pos, observation.goal_neg, observation.at_goal)

    def _step_args(self, action: Optional[Tensor], next_task_ids: Optional[Tensor], final: bool) -> tuple:
        if action is not None:
            action = action.reshape(self.num_envs).to(device=self.device, dtype=torch.int64).contiguous()
        sampled = next_task_ids is None and self.sampler is not None and self._autoreset
        if sampled:
            next_task_ids = self._sample_task_ids()
        if next_task_ids is not None:
            if not sampled:
                next_task_ids = _task_ids(self.core, next_task_ids, None, self.num_envs, self.device, "next_task_ids")
            if not self.multi:
                next_task_ids = None  # a table of one: every instance is 0
        return (
            self.handle, self.states, self._ids(), self.steps, self.draws, self.counts, self.views, self.count,
            self.at_goal, self.goal_pos, self.goal_neg, action, next_task_ids, self.first_env, final,
        )  # fmt: skip

    def _step_into(
        self, out: StepResult, action: Optional[Tensor], next_task_ids: Optional[Tensor], final: bool
    ) -> None:
        """step() writing its outputs into ``out``'s tensors (eager only; CapturedRollout's slices of its stacked
        outputs)."""
        _ops.step_impl(*self._step_args(action, next_task_ids, final), out=tuple(out)[:8])
        if self._autoreset:
            self._advance_resets(out.terminated | out.truncated)
        out.count.copy_(self.count)
        out.at_goal.copy_(self.at_goal)
        out.states.copy_(self.states)
        out.task_ids.copy_(self.task_ids)
        if self.goals:
            out.goal_pos.copy_(self.goal_pos)
            out.goal_neg.copy_(self.goal_neg)

    def random_actions(self, max_actions: Optional[int] = None) -> Tensor:
        """The random policy's next choices without stepping ([N] int64 on ``device``): the successors ``step(None)``
        would take now (the counter-based RNG at the current draw counters), limited to the first ``max_actions``
        successors (None: no limit); 0 where there is none. ``step(random_actions())`` equals ``step(None)`` when no
        count exceeds max_actions. No host synchronization; the custom op ``mymyr::random_actions`` under
        torch.compile."""
        if max_actions is not None and int(max_actions) < 1:
            raise ValueError("mymyr: max_actions must be None or positive")
        args = (self.handle, self.draws, self.count, self.first_env, None if max_actions is None else int(max_actions))
        if torch.compiler.is_compiling():
            return torch.ops.mymyr.random_actions(*args)
        return _ops.random_actions_impl(*args)

    def set_seed(self, seed: int) -> None:
        """New action and sampling RNG keys; action draw and instance reset counters restart at 0. On a device,
        both in stream order on torch's current stream:
        steps enqueued before draw with the old key, steps and graph replays after with the new one."""
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self.core.set_seed(int(seed), stream=stream)
        self.draws.zero_()
        self._sampling_seed.copy_(torch.tensor(
            [int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF], device=self.device
        ))
        self._reset_counts.zero_()

    def capture(
        self,
        steps: int,
        policy: Optional[Callable[["BatchedEnv"], Tensor]] = None,
        *,
        next_task_ids: Optional[Tensor] = None,
        final: bool = False,
        pool: Any = None,
    ) -> "CapturedRollout":
        """``steps`` steps (actions from ``policy(self)``, or the random policy) captured into one CUDA graph: a
        :class:`CapturedRollout`, whose ``replay()`` runs them with one launch and returns their stacked outputs. Needs
        ``capturable`` (a device environment on the fast path)."""
        return CapturedRollout(self, steps, policy, next_task_ids=next_task_ids, final=final, pool=pool)

    def set_launch(self, launch: str) -> None:
        """'widest' or 'per_bucket': how a device environment launches over a table's row-width buckets."""
        self.core.set_launch(launch)

    def check_errors(self) -> None:
        """Waits for the device work and raises if a fast-path step met states written without :meth:`refresh`, or
        met task ids outside the table (device ``task_ids`` / ``next_task_ids``; host values are checked when given;
        the general path's step and refresh raise such ids themselves). A no-op on the CPU."""
        self.core.check_errors()

    def close(self) -> None:
        """Releases the op handle (the tensors stay); also done when the BatchedEnv is collected."""
        _ops.unregister(self.handle)

    def __del__(self) -> None:
        try:
            _ops.unregister(self.handle)
        except Exception:  # interpreter shutdown, or a failed __init__
            pass

    def __repr__(self) -> str:
        return f"BatchedEnv(num_envs={self.num_envs}, device={self.device}, {self.core!r})"
