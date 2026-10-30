"""BatchedEnv: N planning environments whose state lives in torch tensors, stepped through the custom ops."""

from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import torch
from torch import Tensor

from mymyr._core import _rl, _rl_torch
from mymyr.rl._sampling import _instance_sampler

from . import _ops
from ._graph import CapturedRollout
from ._observation import ActionRows, Observer, SuiteObservation

__all__ = ["ActionRows", "BatchedEnv", "Observation", "StepResult", "SuiteObservation"]


class Observation(NamedTuple):
    """Owned policy inputs by default; :meth:`BatchedEnv.observation` can explicitly borrow live tensors. ``encoding``
    is the encoded observation of an environment with an ``observation`` encoder (None without one)."""

    states: Tensor
    task_ids: Tensor
    count: Tensor
    goal_pos: Tensor
    goal_neg: Tensor
    at_goal: Tensor  # bool: the state is a goal (a solved reset, a goal mask that holds): the next step absorbs it
    encoding: Any = None  # the encoding of the rows (see BatchedEnv's observation)
    actions: Any = None  # the expansion rows of the encoding's action entities (ActionRows, with obs_actions)
    action_visited: Any = None  # bool [targets]: the target's successor was visited (track_visited, obs_actions)


class StepResult(NamedTuple):
    """The outputs of one :meth:`BatchedEnv.step`, including a stable next observation.

    ``final_states`` precedes autoreset; ``states`` / ``task_ids`` / goal masks describe the next policy observation.
    Empty goal masks have shape [N, 0]. The native environment's public tensors remain borrowed live buffers.

    With an ``observation`` encoder, ``obs`` is the encoding of the next observation (the rows after the step and the
    autoreset) and ``final_obs`` the encoding of the reached states before the autoreset, with the instances and goal
    masks they were reached in (``final=True``; the same batch as ``obs`` without autoreset; None with
    ``final=False``). Both are None without an encoder. Their arrays are new tensors of every step.
    """

    reward: Tensor  # float32
    terminated: Tensor  # bool: goal reached, or a dead end (dead_end_terminal)
    truncated: Tensor  # bool: max_steps reached (not terminated)
    goal: Tensor  # bool: the reached state is a goal
    invalid: Tensor  # bool: an out-of-range index or inapplicable label in a nonterminal, non-stuck row
    schema: Tensor  # int32: the label of the step's action (-1: no move)
    binding: Tensor  # [N, label_width] int32 objects of the row's instance (-1 padding)
    final_states: Tensor  # [N, row_words] int64 with final=True (the reached states before the autoreset), else [0, *]
    count: Tensor  # int32: successors of the state after the step (Label: 0/1; a copy)
    states: Tensor
    task_ids: Tensor
    goal_pos: Tensor
    goal_neg: Tensor
    at_goal: Tensor  # bool: the next observation's state is a goal (an autoreset into a solved initial state)
    obs: Any = None  # the encoding of the next observation
    final_obs: Any = None  # the encoding of the reached states before the autoreset (final=True)
    revisited: Optional[Tensor] = None  # bool [N]: reached state was visited before this step
    actions: Any = None  # the expansion rows of obs's action entities (ActionRows, with obs_actions)
    action_visited: Any = None  # bool [targets] of obs: the target's successor was visited (track_visited, obs_actions)


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

    ``actions="label"`` takes ``step((schema, binding))`` with int32 tensors [N] and [N, L] on the environments'
    device, where L is at least the largest schema arity. It validates and applies the label directly, without
    enumerating successors. ``count`` is then 0/1 (whether an applicable action exists); ``step(None)`` and
    ``random_actions`` are unavailable. Inapplicable labels do not move and set ``invalid`` in a nonterminal state with
    applicable actions. Padding past a schema's arity is ignored. Labels use each row's domain's schemas and instance's
    objects. On a device, label steps make no host round trip and can be captured (``capture`` with a policy returning
    the labels); they need tables without numeric fluents, axioms and conditional effects whose schemas have at most 16
    parameters (``fast_unsupported(table, actions="label")`` says why not).

    ``track_visited=H`` keeps hash tables ``visited`` int64 [N, H], ``visited_count`` int32 [N], and
    ``visited_status`` int32 [1] (sticky bit 0: overflow) on ``device``. H is a power of two in [2, 2^20]; True selects
    the next power of two at least 2 * (max_steps + 1), requiring bounded episodes. Reset inserts the initial state;
    step reports ``revisited`` before inserting the reached state and before any autoreset clears its table.
    Equal state hashes share one entry. Above 3/4 occupancy new keys are skipped, so membership can miss revisits.
    ``clear_visited(mask)`` clears selected histories and inserts their current states; refresh keeps histories.
    The table and counts are borrowed read-only views; status can be acknowledged with ``visited_status.zero_()``.
    A device keeps the tables with the same layout and values as the CPU, without host synchronization, and captures
    them with the steps. With ``obs_actions`` and an encoder that lists targets, ``Observation.action_visited`` and
    ``StepResult.action_visited`` say for each target of the encoding (bool [targets], aligned with
    ``target_entity``; a tuple per domain batch over a suite) whether it is an action entity whose successor is in its
    environment's table, the table after the step and its autoreset (a no-op successor is always visited).

    Encoded observations: ``observation`` is a ``mymyr.encoding.FlatRelationEncoder`` of the environments' table (over a
    suite of several domains: one encoder per domain, in the suite's order). Every step then encodes the next
    observation (``StepResult.obs``) and, with ``final=True``, the reached states before the autoreset
    (``StepResult.final_obs``), from the environment tensors after the step and on torch's current stream; the goal
    relations hold the per-env goal masks with ``goals=True``. On a device the encoding runs the device encoder; on the
    CPU the host encoder. ``obs_capacities`` (``dict(entities=, args=)``, plus ``targets=`` for target sources and
    ``graphs=``, by default N; over a suite one dict for every domain or one per domain) gives the arrays fixed sizes:
    the true sizes are in the batch's ``device_sizes`` (``sizes``), and a batch that does not fit is all zero with its
    overflow flag set. Without capacities each encoding is exact, which on a device waits for the batch's sizes once
    (and cannot be captured). ``obs_actions=True`` adds the action relations of the observed states (their expansion;
    with capacities the device expansion is bounded, keeps its status on the device and captures with the step;
    tables requiring host expansion are rejected); by default observations hold no action tuples. With ``obs_actions``
    the observation's expansion rows come along as ``Observation.actions`` / ``StepResult.actions``
    (:class:`ActionRows`: labels, successor hashes and no-op flags, without a second expansion), and
    ``obs_drop_noops=True`` leaves out the actions whose successor equals the state (unless every successor of the
    state is one): their action entities are absent, the others keep their canonical successor index as
    ``entity_ref``, so ``mymyr.encoding.action_indices`` still gives the actions the environment takes. With
    capacities the entity capacity bounds the kept action rows (``entities`` times the largest number of schemas that
    share an action relation), and ``obs_capacities['successors']`` the successors before the drop (default: the same
    rows; with ``obs_drop_noops`` set it to the most successors the observed states have); more of either is an
    overflow, reported alike on the CPU and on a device. Over a suite an observation is a :class:`SuiteObservation`:
    one batch per domain and each row's index in its batch.

    CUDA graphs: on a device's fast path (``capturable``) reset, refresh, step and random_actions synchronize nothing
    and allocate nothing (the environment's scratch is sized for N at construction), and the state tensors are marked
    static for torch.compile (they keep their addresses; assign into them in place). So loops of
    steps capture into CUDA graphs: :meth:`capture` (``torch.cuda.CUDAGraph``), ``torch.cuda.graph`` around the calls,
    or ``torch.compile(mode="reduce-overhead")`` of a function that steps; the encoding of observations with
    ``obs_capacities`` waits for nothing, so it is captured with the steps. On the general path the compiled calls are
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
        observation: Any = None,
        obs_capacities: Any = None,
        obs_actions: bool = False,
        obs_drop_noops: bool = False,
        actions: Literal["index", "label"] = "index",
        track_visited: bool | int = False,
    ) -> None:
        self.device = torch.device(device)
        if actions not in ("index", "label"):
            raise ValueError("mymyr: actions must be 'index' or 'label'")
        self.visited_capacity = _rl._visited_capacity(track_visited, max_steps)
        self.actions = actions
        index = _device_index(self.device)
        if index is not None:
            self.device = torch.device("cuda", index)
        self.core = _rl_torch.Env(
            table, device=index, ctx=ctx, seed=seed, max_steps=max_steps, step_reward=step_reward,
            goal_reward=goal_reward, dead_end=dead_end, dead_end_reward=dead_end_reward,
            dead_end_terminal=dead_end_terminal, autoreset=autoreset, witness_pruning=witness_pruning, path=path,
            threads=threads, actions=actions,
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
        H = self.visited_capacity
        self.visited = torch.zeros((n, H), dtype=torch.int64, device=d) if H else None
        self.visited_count = torch.zeros(n, dtype=torch.int32, device=d) if H else None
        self.visited_status = torch.zeros(1, dtype=torch.int32, device=d) if H else None
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
        if observation is None and (obs_capacities is not None or obs_actions or obs_drop_noops):
            raise ValueError("mymyr: obs_capacities, obs_actions and obs_drop_noops need an observation encoder")
        self._observer = None if observation is None else Observer(self, observation, obs_capacities, obs_actions,
                                                                    obs_drop_noops)
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

    @property
    def observation_encoders(self) -> Optional[list]:
        """The observation encoders (one per domain over a suite of several domains), or None."""
        return None if self._observer is None else list(self._observer.encoders)

    def observation(self, *, copy: bool = True) -> Observation:
        """Policy inputs that survive later steps/resets by default. ``copy=False`` borrows the live buffers. With an
        observation encoder, ``encoding`` is the encoding of the current rows (new arrays either way)."""
        keep = (lambda tensor: tensor.clone()) if copy else (lambda tensor: tensor)
        empty = self.states.new_empty((self.num_envs, 0))
        encoding, actions = self._observe()
        return Observation(
            keep(self.states), keep(self.task_ids), keep(self.count),
            keep(self.goal_pos) if self.goals else empty, keep(self.goal_neg) if self.goals else empty,
            keep(self.at_goal), encoding, actions, self._action_visited(encoding, actions),
        )

    def _action_visited(self, encoding: Any, actions: Any) -> Any:
        """The observation's ``action_visited`` against the current tables (None without visited tables, action rows
        or targets)."""
        if not self.visited_capacity or actions is None or not self._observer.has_targets:
            return None
        return self._observer.action_visited(encoding, actions, self.visited, self.visited_count)

    def _goal_masks(self) -> Optional[tuple[Tensor, Tensor]]:
        return (self.goal_pos, self.goal_neg) if self.goals else None

    def encode_observation(self) -> Any:
        """The encoding of the current rows (states, task ids and goal masks) with the observation encoder; None without
        one. On torch's current stream, like a step's ``obs``."""
        return self._observe()[0]

    def _observe(self) -> tuple[Any, Any]:
        """The encoding of the current rows and the expansion rows of its action entities (None, None without an
        observation encoder)."""
        if self._observer is None:
            return None, None
        return self._observer.encode(self.states, self.task_ids, self._goal_masks())

    def _before_step(self, final: bool) -> Optional[tuple]:
        """What the final observation needs of the rows before an autoresetting step: their task ids and goals."""
        if self._observer is None or not final or not self._autoreset:
            return None
        goals = self._goal_masks()
        return self.task_ids.clone(), None if goals is None else (goals[0].clone(), goals[1].clone())

    def _final_obs(self, before: Optional[tuple], final_states: Tensor, obs: Any, final: bool) -> Any:
        if self._observer is None or not final:
            return None
        if before is None:
            return obs  # without autoreset the reached states are the next observation
        return self._observer.encode(final_states, before[0], before[1])[0]
    def encode_transitions(self, encoder: Any, *, actions: Any = None, capacities: Any = None) -> Any:
        """Encode the current rows and successors with a transition or transition-effects encoder on this device.

        ``actions`` may supply an existing expansion; otherwise successors are expanded on the current stream.
        ``capacities`` has the encoder's usual meaning. The returned arrays own their storage.
        """
        from mymyr import encoding
        from mymyr.rl import as_table

        if not isinstance(encoder, (encoding.FlatTransitionEncoder, encoding.FlatTransitionEffectsEncoder)):
            raise ValueError("mymyr: encoding: encode_transitions needs a transition encoder")
        if as_table(encoder.table).fingerprint != as_table(self.core.table).fingerprint:
            raise ValueError("mymyr: encoding: the transition encoder belongs to another table")
        return encoder.encode(self.states, self.task_ids, goals=self._goal_masks(), actions=actions,
                              framework="torch", capacities=capacities)

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
        return t + ([self.goal_pos, self.goal_neg] if self.goals else []) + (
            [self.visited, self.visited_count, self.visited_status] if self.visited_capacity else []
        )

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
        ids = self.task_ids
        if task_ids is None and self.sampler is not None:
            ids = _where(mask, self._sample_task_ids(), ids)
        elif task_ids is not None:
            new = _task_ids(self.core, task_ids, mask, self.num_envs, self.device, "reset")
            ids = _where(mask, new, ids)
        keep = goal_pos is not None or goal_neg is not None
        pos, neg = self.goal_pos, self.goal_neg
        if keep:
            if not self.goals or goal_pos is None or goal_neg is None:
                raise ValueError("mymyr: per-env goals need goals=True and both goal_pos and goal_neg")
            pos = _where(mask, goal_pos.to(pos).expand_as(pos), pos).contiguous()
            neg = _where(mask, goal_neg.to(neg).expand_as(neg), neg).contiguous()
        # Host reset validates every selected mask against the proposed ids before mutating any output.
        args = (
            self.handle, self.states, ids if self.multi else None, self.steps, self.counts, self.views, pos, neg,
            self.count, self.at_goal, mask, keep, self.visited, self.visited_count, self.visited_status,
        )  # fmt: skip
        _ops.reset(*args)
        self.task_ids.copy_(ids)
        if keep:
            self.goal_pos.copy_(pos)
            self.goal_neg.copy_(neg)
        self._advance_resets(mask)
        return self.count

    def clear_visited(self, mask: Optional[Tensor] = None) -> None:
        """Clear all tables, or rows selected by bool mask [N], and insert their current states.

        Use after writing states directly; refresh() only updates the action and goal caches. Overflow status is
        sticky across resets and clears; visited_status.zero_() explicitly acknowledges it.
        """
        if not self.visited_capacity:
            return
        if mask is not None:
            mask = mask.reshape(self.num_envs).to(device=self.device, dtype=torch.bool).contiguous()
        args = (self.handle, self.states, self._ids(), self.visited, self.visited_count, self.visited_status, mask)
        _ops.clear_visited(*args)

    def refresh(self) -> Tensor:
        """Recomputes the cache, ``count`` and ``at_goal`` after the caller wrote ``states``, ``task_ids`` or the goal
        masks. Goals stay in the written instance's slot layout; bits outside its fluent slots raise ValueError on
        the host or set a device error reported by :meth:`check_errors`."""
        args = (
            self.handle, self.states, self._ids(), self.counts, self.views, self.goal_pos, self.goal_neg, self.count,
            self.at_goal,
        )  # fmt: skip
        (_ops.refresh if self.capturable else _ops.refresh_sync)(*args)
        return self.count

    def step(
        self, action: Tensor | tuple[Tensor, Tensor] | None = None, *, next_task_ids: Optional[Tensor] = None, final: bool = True,
        copy: bool = True,
    ) -> StepResult:
        """One step of every environment; ``action`` [N] int64 indexes the canonical successor order (None: the
        random policy). With ``actions="label"``, action is a tuple (int32 [N] schema, int32 [N, L] binding),
        checked and applied directly; None is refused and count is 0/1. ``next_task_ids`` [N] overrides sampling for autoresetting environments (None: sample,
        or keep their instances with sampler=None). ``final=True`` retains the reached state before reset;
        ``copy=True`` owns the next observation. ``copy=False`` explicitly borrows live observation buffers."""
        args = self._step_args(action, next_task_ids, final)
        before = self._before_step(final)
        r = (_ops.step if self.capturable else _ops.step_sync)(*args)
        if self._autoreset:
            self._advance_resets(r[1] | r[2])
        observation = self.observation(copy=copy)
        final_obs = self._final_obs(before, r[7], observation.encoding, final)
        return StepResult(*r, observation.count, observation.states, observation.task_ids,
                          observation.goal_pos, observation.goal_neg, observation.at_goal, observation.encoding,
                          final_obs, args[-1], observation.actions, observation.action_visited)

    def _step_args(self, action: Tensor | tuple[Tensor, Tensor] | None, next_task_ids: Optional[Tensor], final: bool,
                   revisited: Optional[Tensor] = None) -> tuple:
        if revisited is None:
            revisited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        schema, binding = None, None
        if self.actions == "label":
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("mymyr: label mode requires action=(schema, binding); the random policy is unavailable")
            schema, binding = action
            if not isinstance(schema, Tensor) or not isinstance(binding, Tensor):
                raise ValueError("mymyr: label actions require schema and binding tensors")
            schema, binding = schema.contiguous(), binding.contiguous()
            action = None
        elif isinstance(action, tuple):
            raise ValueError("mymyr: label actions require actions='label'")
        elif action is not None:
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
            self.at_goal, self.goal_pos, self.goal_neg, action, next_task_ids, self.first_env, final, schema, binding,
            self.visited, self.visited_count, self.visited_status, revisited,
        )  # fmt: skip

    def _step_into(
        self, out: StepResult, action: Tensor | tuple[Tensor, Tensor] | None, next_task_ids: Optional[Tensor], final: bool
    ) -> tuple[Any, Any, Any, Any]:
        """step() writing its outputs into ``out``'s tensors (eager only; CapturedRollout's slices of its stacked
        outputs); returns the step's ``obs``, ``final_obs``, ``actions`` and ``action_visited``."""
        before = self._before_step(final)
        _ops.step_out(*tuple(out)[:8], *self._step_args(action, next_task_ids, final, out.revisited))
        if self._autoreset:
            self._advance_resets(out.terminated | out.truncated)
        out.count.copy_(self.count)
        out.at_goal.copy_(self.at_goal)
        out.states.copy_(self.states)
        out.task_ids.copy_(self.task_ids)
        if self.goals:
            out.goal_pos.copy_(self.goal_pos)
            out.goal_neg.copy_(self.goal_neg)
        obs, actions = self._observe()
        return obs, self._final_obs(before, out.final_states, obs, final), actions, self._action_visited(obs, actions)

    def random_actions(self, max_actions: Optional[int] = None) -> Tensor:
        """The random policy's next choices without stepping ([N] int64 on ``device``): the successors ``step(None)``
        would take now (the counter-based RNG at the current draw counters), limited to the first ``max_actions``
        successors (None: no limit); 0 where there is none. ``step(random_actions())`` equals ``step(None)`` when no
        count exceeds max_actions. No host synchronization; the custom op ``mymyr::random_actions`` under
        torch.compile."""
        if self.actions == "label":
            raise ValueError("mymyr: label mode has no random policy")
        if max_actions is not None and int(max_actions) < 1:
            raise ValueError("mymyr: max_actions must be None or positive")
        args = (self.handle, self.draws, self.count, self.first_env, None if max_actions is None else int(max_actions))
        return _ops.random_actions(*args)

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
        the general path's step and refresh raise such ids themselves), or goal masks outside the instance's fluent
        slots. A no-op on the CPU."""
        self.core.check_errors()

    def close(self) -> None:
        """Releases the op handles (the tensors stay); also done when the BatchedEnv is collected."""
        _ops.unregister(self.handle)
        if getattr(self, "_observer", None) is not None:
            self._observer.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # interpreter shutdown, or a failed __init__
            pass

    def __repr__(self) -> str:
        return f"BatchedEnv(num_envs={self.num_envs}, device={self.device}, {self.core!r})"
