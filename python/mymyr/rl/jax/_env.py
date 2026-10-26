"""The functional JAX planning environment."""

from typing import Any, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from mymyr._core import _rl, _rl_jax
from mymyr.rl._sampling import _instance_sampler

from . import _ffi

__all__ = ["Env", "EnvState", "Observation", "PaddedExpansion", "FlatExpansion", "SuiteObservation", "key_seed"]


class EnvState(NamedTuple):
    """The state of N environments (a pytree of arrays; the leading dimension is the environment).

    Fields:
        states: the state words as uint32 halves (a u64 row of ``row_words`` words is ``2 * row_words`` uint32, low
            word first).
        task_id: each environment's instance (int32; 0 over a table of one).
        steps: the steps of the current episode.
        draws: the counter-based RNG's draw counters (uint32 [N, 2]: low, high).
        env_id: each environment's RNG stream.
        count: the successors of the current state.
        counts, views: the device fast path's cache of the current state (uint32 [N, C] and [N, 2V]; [N, 0] on the
            CPU and on the general path): opaque, kept current by reset, step and refresh.
        goal_pos, goal_neg: the per-env goal masks (uint32 [N, 2 words] with goals=True, else [N, 0]).
        reset_counts, sampling_seed: the instance sampler's per-environment counters ([N] uint32) and keys
            ([N, 2] uint32).
        at_goal: whether the current state is a goal ([N] bool; by the goal masks with goals=True, else the
            instance's goal). Reset, step and refresh keep it, and a step leaves those environments where they are
            (absorbing).

    Writing ``states`` or the goal masks needs :meth:`Env.refresh`."""

    states: Any
    task_id: Any
    steps: Any
    draws: Any
    env_id: Any
    count: Any
    counts: Any
    views: Any
    goal_pos: Any
    goal_neg: Any
    reset_counts: Any
    sampling_seed: Any
    at_goal: Any


class Observation(NamedTuple):
    """What a policy sees: the state words (uint32 [N, 2 row_words]), the successor count [N] (the action mask of
    successor slots is ``arange(K) < count``), the instance [N], the per-env goal masks ([N, 2 words], or [N, 0]
    without goals=True) and whether the state is a goal already (bool [N]: a reset into a solved state, or a goal mask
    that holds; the next step absorbs it). With an ``observation`` encoder, ``encoding`` is the encoding of the rows (a
    ``mymyr.encoding.jax.JaxFlatBatch``); None without one."""

    state: Any
    count: Any
    task_id: Any
    goal_pos: Any
    goal_neg: Any
    at_goal: Any
    encoding: Any = None


class SuiteObservation(NamedTuple):
    """The encoding of the N rows of environments over a TaskSuite of several domains: one batch per domain, since
    each domain has a relation catalog of its own (a pytree).

    - ``batches[d]``: the ``JaxFlatBatch`` of domain ``d``'s rows, in row order (graph k is the k-th row of domain d);
      its ``task_ids`` are the domain's instance numbers (``suite.local_ids``). With capacities every batch has an entry
      for each of the N rows in its per-graph arrays, zero past the domain's rows, whose number is its
      ``device_graphs``; an exact batch has the domain's rows only.
    - ``domain``: int32 [N], the domain of each row.
    - ``index``: int32 [N], the graph of each row in its domain's batch: row i is graph ``index[i]`` of
      ``batches[domain[i]]``.
    """

    batches: tuple
    domain: Any
    index: Any

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


class PaddedExpansion(NamedTuple):
    """The padded [N, K] expansion in canonical order (rl.expand's padded view): succ [N, K, 2 row_words], schema
    [N, K] (-1 padding), binding [N, K, L] (-1 padding), goal [N, K], mask [N, K], count [N] (true counts; count > K
    is an overflow)."""

    succ: Any
    schema: Any
    binding: Any
    goal: Any
    mask: Any
    count: Any
    derived: Any = None  # [N, K, 2 D] uint32 derived words of the successors (expand(..., derived=True))
    succ_hash: Any = None  # [N, K, 2] uint32 pairs
    noop: Any = None  # [N, K] bool
    index: Any = None  # [N, K] canonical successor positions, -1 padding
    generated: Any = None  # scalar pre-drop count


class FlatExpansion(NamedTuple):
    """The flat expansion with capacity M (rl.expand's CSR): succ [M, 2 row_words], parent, schema [M], binding
    [M, L], goal [M], offsets [N + 1] (true counts: offsets[N] > M is an overflow). Rows past the total are padding
    (succ 0, parent / schema / binding -1, goal False)."""

    succ: Any
    parent: Any
    schema: Any
    binding: Any
    goal: Any
    offsets: Any
    derived: Any = None  # [M, 2 D] uint32 derived words of the successors (expand_flat(..., derived=True))
    succ_hash: Any = None  # [M, 2] uint32 pairs
    noop: Any = None  # [M] bool
    index: Any = None  # [M] canonical successor positions, -1 padding
    generated: Any = None  # scalar pre-drop count


def key_seed(key: Any) -> Any:
    """The Philox seed of a JAX key as uint32 [..., 2] (low, high): the key's first two data words (high, low), so
    ``jax.random.key(s)`` and ``jax.random.PRNGKey(s)`` give seed s (as ``rl::HostEnv``'s EnvConfig::seed). An int is
    taken as the seed itself."""
    if isinstance(key, (int, np.integer)):
        s = int(key) & 0xFFFFFFFFFFFFFFFF
        return jnp.asarray([s & 0xFFFFFFFF, s >> 32], dtype=jnp.uint32)
    key = jnp.asarray(key) if not isinstance(key, jax.Array) else key
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        data = jax.random.key_data(key)
    elif key.ndim == 0 and jnp.issubdtype(key.dtype, jnp.integer):
        # an integer seed (a traced int under jit): the seed itself
        lo = key.astype(jnp.uint32)
        hi = (key >> 32).astype(jnp.uint32) if key.dtype.itemsize == 8 else jnp.zeros((), jnp.uint32)
        return jnp.stack([lo, hi])
    else:
        data = key  # raw key data (uint32 [..., 2])
    if data.ndim == 0 or data.shape[-1] < 2:
        raise TypeError("mymyr: key must be a JAX key, raw key data [..., 2] or an integer seed")
    data = data.astype(jnp.uint32)
    return jnp.stack([data[..., 1], data[..., 0]], axis=-1)


def _jax_device(device: Any, numeric: bool) -> jax.Device:
    if device is None:
        if numeric:
            return jax.devices("cpu")[0]
        return jax.devices()[0]
    if isinstance(device, jax.Device):
        return device
    if isinstance(device, int):
        return jax.devices("gpu")[device]
    if isinstance(device, str):
        kind, _, index = device.partition(":")
        kind = {"cuda": "gpu"}.get(kind, kind)
        return jax.devices(kind)[int(index) if index else 0]
    raise TypeError(f"mymyr: device must be None, a jax.Device, 'cpu', 'cuda[:i]' or a CUDA index, not {device!r}")


def _numeric(table: Any) -> bool:
    numeric_words = getattr(table, "numeric_words", 0)
    return bool(numeric_words)


class Env:
    """N planning environments over a TaskSuite or a TaskTable (or a Task: its table of one) as pure JAX functions
    (gymnax / Jumanji style)::

        env = mymyr.rl.jax.Env(table, max_steps=200, goal_reward=1.0)
        state, obs = env.reset(key, task_ids)                                             # int32 [N] instances
        state, obs, reward, terminated, truncated, info = env.step(key, state)            # the random policy
        state, obs, reward, terminated, truncated, info = env.step(key, state, action)    # int32 [N] successor slots
        state, obs, *_ = env.step(key, state, action, next_task_ids=ids)                  # curriculum at autoreset

    ``reset`` / ``step`` / ``refresh`` / ``reset_where`` / ``expand`` are pure functions of their arguments (the
    environment's native side is configuration only), so they run under ``jax.jit``, ``jax.vmap`` and ``lax.scan``.
    They are XLA FFI calls (mymyr._core._rl_jax): on a CUDA device the step runs ``cuda::DeviceEnv`` on XLA's stream
    (the fast path: no host synchronization), on the CPU ``rl::HostEnv``; both give the same results bit for bit.

    A step follows the host environments' (CpuEnvPool's) semantics: the action indexes the current state's successors in
    canonical order (outside [0, count): no move, ``info["invalid"]``), the reward is step_reward (+ goal_reward
    at a goal, + dead_end_reward at a dead end), terminated = goal or dead end (``dead_end_terminal``), truncated = not
    terminated and the episode reached max_steps. With ``autoreset`` (SAME_STEP) a finished environment restarts from
    the initial state of its instance (or of ``next_task_ids[i]``) within the step and ``info["final_state"]`` holds the
    state it reached (for bootstrapping truncated episodes); ``autoreset=False`` leaves resets to :meth:`reset_where`.
    With ``goals=True`` each environment carries goal masks (the goal test is then the mask test; reset sets its
    instance's goal unless goals are given).

    Randomness: ``action=None`` is the random policy, a uniform successor from Philox-4x32-10 of (seed, env id, draw
    counter) with seed = :func:`key_seed` (key): an environment's trajectory depends on the key, its
    ``env_id`` and its draw counter only, not on the batch, the device or ``vmap``. Passing the same key to every step
    reproduces ``rl::HostEnv`` / ``cuda::DeviceEnv`` with that seed; per-step keys are fine too.

    ``device``: None (JAX's default device; the CPU for numeric tables), a ``jax.Device``, ``"cpu"``, ``"cuda[:i]"`` or
    a CUDA index; the arrays of a jitted rollout must live there. Numeric tables use the general expansion path on a CUDA device. ``path`` chooses the device path ('auto': the fast path where the table allows it; 'general': the
    device expansion, which synchronizes on the successor count every step). ``labels=False`` skips the binding labels
    and ``final_state=False`` the final states (both are written every step otherwise).

    Encoded observations: ``observation`` is a ``mymyr.encoding.FlatRelationEncoder`` of the environments' table (over
    a suite of several domains: one encoder per domain, in the suite's order; an observation is then a
    :class:`SuiteObservation`, one batch per domain and each row's index in its batch). Every
    observation (of reset, reset_where and step) then carries the encoding of its rows (``Observation.encoding``,
    through ``mymyr.encoding.jax.encode``: the device encoder on CUDA, the host encoder on the CPU, with equal arrays),
    and a step with ``final_state=True`` the encoding of the reached states before the autoreset, with the instances
    and goal masks they were reached in (``info["final_obs"]``; the observation's encoding without autoreset). The goal
    relations hold the per-env goal masks with ``goals=True``. ``obs_capacities`` (``dict(entities=, args=)``, plus
    ``targets=`` for target sources and ``graphs=``, by default the number of environments) gives the arrays static
    shapes, which ``jax.jit`` and ``lax.scan`` need; the true sizes are in ``encoding.device_sizes``, and a batch that
    does not fit is all zero with its overflow flag set; over a suite one dict serves every domain, or a sequence gives
    one per domain. Without capacities the encodings are exact (eager calls only). The JAX encoding holds no action
    relations.
    """

    def __init__(
        self,
        table: Any,
        *,
        device: Any = None,
        goals: bool = False,
        sampler: Any = "uniform",
        max_steps: int | None = None,
        step_reward: float = 0.0,
        goal_reward: float = 1.0,
        dead_end: str = "no_successors",
        dead_end_reward: float = 0.0,
        dead_end_terminal: bool = True,
        autoreset: bool = True,
        canonical_order: bool = True,
        witness_pruning: bool = False,
        path: str = "auto",
        labels: bool = True,
        final_state: bool = True,
        ctx: Any = None,
        observation: Any = None,
        obs_capacities: Optional[dict] = None,
        obs_actions: bool = False,
    ) -> None:
        numeric = _numeric(table)
        self.device: jax.Device = _jax_device(device, numeric)
        on_gpu = self.device.platform in ("gpu", "cuda")
        index = getattr(self.device, "local_hardware_id", None)
        if index is None:
            index = self.device.id
        self.native = _rl_jax.Env(
            table,
            device=index if on_gpu else None,
            ctx=ctx,
            max_steps=max_steps,
            step_reward=step_reward,
            goal_reward=goal_reward,
            dead_end=dead_end,
            dead_end_reward=dead_end_reward,
            dead_end_terminal=dead_end_terminal,
            autoreset=autoreset,
            canonical_order=canonical_order,
            witness_pruning=witness_pruning,
            path=path,
        )
        _ffi.register()
        self.table = table
        self.handle: int = self.native.handle
        self.num_instances: int = self.native.num_instances
        self.sampler = _instance_sampler(sampler, self.num_instances)
        self.autoreset = bool(autoreset)
        self.words: int = self.native.words
        self.row_words: int = self.native.row_words
        self.label_width: int = self.native.label_width
        self.derived_words: int = self.native.derived_words
        self.cache_schemas: int = self.native.cache_schemas
        self.cache_view_words: int = self.native.cache_view_words
        self.initial_counts: list[int] = list(self.native.initial_counts)
        self.goals = bool(goals)
        self.labels = labels
        self.final_state = final_state
        rw2 = 2 * self.row_words
        self._initial_states = np.asarray(self.native.initial_states(), dtype=np.uint32).reshape(self.num_instances, rw2)
        self.observation_encoders, self.obs_capacities, self._obs_goals = _observation(
            table, observation, obs_capacities, obs_actions
        )
        if self.observation_encoders is not None and len(self.observation_encoders) > 1:
            self._domain_of = np.asarray(table.domain_of, np.int32)
            self._local_ids = np.asarray(table.local_ids, np.int32)

    def __repr__(self) -> str:
        return f"mymyr.rl.jax.Env({self.native!r}, device={self.device})"

    @property
    def fast(self) -> bool:
        """Whether a device environment takes the fast path (the per-environment cache)."""
        return self.native.fast

    @property
    def initial_states(self) -> np.ndarray:
        """The instances' initial states as uint32 [I, 2 row_words]."""
        return self._initial_states

    def _dev(self):
        return jax.default_device(self.device)

    def _goal_cols(self) -> int:
        return 2 * self.words if self.goals else 0

    def _ids(self, task_ids: Any, n: Optional[int] = None) -> Any:
        if task_ids is None:
            if self.num_instances > 1:
                raise ValueError(
                    f"mymyr: a batch over a table of {self.num_instances} instances needs task ids"
                )
            return jnp.zeros((n,), jnp.int32)
        return self._checked(task_ids, "task_ids")

    def _checked(self, ids: Any, name: str, mask: Any = None) -> Any:
        """``ids`` as int32. Host values (ints, sequences, NumPy arrays) are checked here: ValueError for an id outside
        the table's (suite's) instances in a row the call uses (``mask``: the masked rows, when it is concrete). JAX
        arrays are checked by the native environment: on the CPU in the call, on a device by :meth:`check_errors` and
        the general path's next step or refresh (which synchronize)."""
        if not isinstance(ids, jax.Array):
            flat = np.asarray(ids, np.int64).reshape(-1)
            bad = (flat < 0) | (flat >= self.num_instances)
            if isinstance(mask, jax.core.Tracer):
                bad[:] = False  # a traced mask: the native environment checks the rows it uses
            elif mask is not None:
                bad &= np.broadcast_to(np.asarray(mask, np.bool_).reshape(-1), bad.shape)
            if bad.any():
                r = int(np.flatnonzero(bad)[0])
                multi = isinstance(self.table, _rl.TaskSuite) and self.table.num_domains > 1
                raise ValueError(
                    f"mymyr: {name}: task id {int(flat[r])} of row {r} is outside the "
                    f"{'suite' if multi else 'table'}'s {self.num_instances} instances"
                )
        return jnp.asarray(ids).astype(jnp.int32)

    def observation(self, state: EnvState) -> Observation:
        """The observation of ``state`` (with its encoding, given an observation encoder)."""
        return Observation(state.states, state.count, state.task_id, state.goal_pos, state.goal_neg, state.at_goal,
                           self.encode(state.states, state.task_id, state.goal_pos, state.goal_neg))

    def encode(self, states: Any, task_ids: Any, goal_pos: Any = None, goal_neg: Any = None) -> Any:
        """The encoding of rows (uint32 [N, 2 row_words]) of the instances ``task_ids`` with the observation encoders,
        the goal relations from the goal masks with goals=True: a ``JaxFlatBatch`` over a table, a
        :class:`SuiteObservation` over a suite of several domains; None without an encoder."""
        encoders = self.observation_encoders
        if encoders is None:
            return None
        from mymyr.encoding import jax as encoding_jax

        with self._dev():
            if len(encoders) == 1:
                return encoding_jax.encode(encoders[0], states, task_ids, goals=self._obs_goal_pair(0, goal_pos, goal_neg),
                                           capacities=self._obs_caps(0, states.shape[0]))
            return self._encode_suite(encoding_jax, states, task_ids, goal_pos, goal_neg)

    def _obs_goal_pair(self, d: int, goal_pos: Any, goal_neg: Any) -> Optional[tuple]:
        return (goal_pos, goal_neg) if self.goals and self._obs_goals[d] else None

    def _obs_caps(self, d: int, rows: int) -> Optional[dict]:
        caps = self.obs_capacities
        return None if caps is None else {"graphs": rows, **caps[d]}

    def _encode_suite(self, encoding_jax: Any, states: Any, task_ids: Any, goal_pos: Any, goal_neg: Any) -> Any:
        """One batch per domain: the rows of each domain moved to the front of rows of their own (in row order),
        encoded as the live rows of a batch of N."""
        D = len(self.observation_encoders)
        n = states.shape[0]
        ids = jnp.asarray(task_ids).astype(jnp.int32)
        domain = jnp.asarray(self._domain_of)[ids]
        local = jnp.asarray(self._local_ids)[ids]
        within = jnp.cumsum(jax.nn.one_hot(domain, D, dtype=jnp.int32), axis=0)
        index = jnp.take_along_axis(within, domain[:, None], axis=1)[:, 0] - 1
        batches = []
        if self.obs_capacities is None:  # exact batches: the rows of each domain, grouped on the host
            host = np.asarray(domain)
            for d in range(D):
                rows = jnp.asarray(np.flatnonzero(host == d))
                goals = self._obs_goal_pair(d, None if goal_pos is None else goal_pos[rows],
                                            None if goal_neg is None else goal_neg[rows])
                batches.append(encoding_jax.encode(self.observation_encoders[d], states[rows], local[rows], goals=goals))
            return SuiteObservation(tuple(batches), domain, index.astype(jnp.int32))
        counts = within[-1] if n else jnp.zeros((D,), jnp.int32)

        def front(x: Any, at: Any) -> Any:
            return jnp.zeros((n + 1, *x.shape[1:]), x.dtype).at[at].set(x)[:n]

        for d in range(D):
            at = jnp.where(domain == d, index, n)  # the rows of other domains go to a dropped row
            goals = self._obs_goal_pair(d, goal_pos, goal_neg)
            if goals is not None:
                goals = (front(goals[0], at), front(goals[1], at))
            batches.append(encoding_jax.encode(self.observation_encoders[d], front(states, at), front(local, at),
                                               goals=goals, capacities=self._obs_caps(d, n), live=counts[d]))
        return SuiteObservation(tuple(batches), domain, index.astype(jnp.int32))

    # ---------------------------------------------------------------------------------------------- env functions

    def reset(
        self,
        key: Any = None,
        task_ids: Union[int, Any] = 1,
        *,
        first_env: int = 0,
        goal_pos: Optional[Any] = None,
        goal_neg: Optional[Any] = None,
    ) -> tuple[EnvState, Observation]:
        """Environments at initial states, with env ids ``first_env + i`` and zero action draw counters. ``task_ids``:
        int32 [N] explicit instances, or an int N (sample N instances with ``key``; None means seed 0). ``sampler=None``
        disables sampling and an int N then starts instance 0. ``goal_pos`` / ``goal_neg`` [N, 2 words] (goals=True)
        set the goals (default: the instances'). The key and counters are retained in the functional state."""
        if isinstance(task_ids, (int, np.integer)):
            n = int(task_ids)
            env_id = jnp.arange(first_env, first_env + n, dtype=jnp.uint32)
            seed = jnp.broadcast_to(key_seed(0 if key is None else key), (n, 2))
            ids = jnp.zeros((n,), jnp.int32) if self.sampler is None else self.sampler.sample(
                seed, env_id, jnp.zeros(n, jnp.uint32)
            )
        else:
            ids = self._ids(task_ids)
        n = ids.shape[0]
        with self._dev():
            ids = jnp.asarray(ids)
            states, steps, count, counts, views, gpos, gneg, at_goal = _ffi.env_init(
                self.handle, ids, 2 * self.row_words, self.cache_schemas, 2 * self.cache_view_words, self._goal_cols()
            )
            draws = jnp.zeros((n, 2), jnp.uint32)
            env_id = jnp.arange(first_env, first_env + n, dtype=jnp.uint32)
            sampling_seed = jnp.broadcast_to(key_seed(0 if key is None else key), (n, 2))
            reset_counts = jnp.ones(n, jnp.uint32)
        if goal_pos is not None or goal_neg is not None:
            if not self.goals or goal_pos is None or goal_neg is None:
                raise ValueError("mymyr: per-env goals need goals=True and both goal_pos and goal_neg")
            gpos = jnp.asarray(goal_pos, jnp.uint32).reshape(gpos.shape)
            gneg = jnp.asarray(goal_neg, jnp.uint32).reshape(gneg.shape)
            at_goal = _masks_hold(states, gpos, gneg)
        state = EnvState(
            states, ids, steps, draws, env_id, count, counts, views, gpos, gneg, reset_counts, sampling_seed, at_goal
        )
        if not isinstance(states, jax.core.Tracer):
            # committed to the environment's device: jitted steps of this state run there, whatever the default device
            state = jax.device_put(state, self.device)
        return state, self.observation(state)

    def reset_where(
        self,
        state: EnvState,
        mask: Any,
        task_ids: Optional[Any] = None,
        *,
        key: Any = None,
        goal_pos: Optional[Any] = None,
        goal_neg: Optional[Any] = None,
    ) -> tuple[EnvState, Observation]:
        """Resets the environments with ``mask[i]`` (bool [N]) to sampled instances (steps 0; action draw counters
        and env ids kept). ``task_ids`` [N] overrides sampling; sampler=None keeps the instances. ``key`` reseeds only
        reset rows, restarting their sampling counters. Otherwise the retained sampling keys are used, independent
        of action keys. ``goal_pos`` / ``goal_neg`` [N, 2 words] set their goals (default: their instances')."""
        m = jnp.asarray(mask, jnp.bool_)
        sampling_seed = state.sampling_seed if key is None else jnp.broadcast_to(key_seed(key), state.sampling_seed.shape)
        reset_counts = state.reset_counts if key is None else jnp.where(m, jnp.uint32(0), state.reset_counts)
        if task_ids is None and self.sampler is not None:
            task_ids = self.sampler.sample(sampling_seed, state.env_id, reset_counts)
        ids = state.task_id if task_ids is None else jnp.where(m, self._checked(task_ids, "task_ids", mask), state.task_id)
        keep = goal_pos is not None or goal_neg is not None
        gpos, gneg = state.goal_pos, state.goal_neg
        if keep:
            if not self.goals or goal_pos is None or goal_neg is None:
                raise ValueError("mymyr: per-env goals need goals=True and both goal_pos and goal_neg")
            gpos = jnp.where(m[:, None], jnp.asarray(goal_pos, jnp.uint32), gpos)
            gneg = jnp.where(m[:, None], jnp.asarray(goal_neg, jnp.uint32), gneg)
        with self._dev():
            states, steps, count, counts, views, gpos, gneg, at_goal = _ffi.env_reset(
                self.handle, state.states, ids, state.steps, state.count, state.counts, state.views, gpos, gneg,
                state.at_goal, m, keep,
            )  # fmt: skip
        s = state._replace(
            states=states, task_id=ids, steps=steps, count=count, counts=counts, views=views, goal_pos=gpos, goal_neg=gneg,
            at_goal=at_goal,
            reset_counts=reset_counts + m.astype(jnp.uint32),
            sampling_seed=jnp.where(m[:, None], sampling_seed, state.sampling_seed),
        )
        return s, self.observation(s)

    def refresh(self, state: EnvState) -> EnvState:
        """Recomputes the successor counts, the device cache and ``at_goal`` of states, task ids or goal masks the
        caller wrote into ``state``."""
        with self._dev():
            count, counts, views, at_goal = _ffi.env_refresh(
                self.handle, state.states, state.task_id, state.counts, state.views, state.goal_pos, state.goal_neg
            )
        return state._replace(count=count, counts=counts, views=views, at_goal=at_goal)

    def step(
        self, key: Any, state: EnvState, action: Optional[Any] = None, *, next_task_ids: Optional[Any] = None
    ) -> tuple[EnvState, Observation, Any, Any, Any, dict[str, Any]]:
        """One step of every environment: ``(state, obs, reward, terminated, truncated, info)``.

        ``action``: int32 [N] indices into the canonical successor order (None: the random policy of ``key``).
        ``next_task_ids``: int32 [N], overrides sampled autoreset instances (None: sample, or keep with sampler=None).
        ``info``: ``final_state`` (the reached state before the autoreset; with final_state=True), ``goal`` (the
        reached state is a goal), ``invalid`` (the action was outside [0, count)), ``schema`` [N] and ``binding``
        [N, L] (the action's label, objects of the env's instance before the autoreset; -1 for no move; binding with
        labels=True)."""
        n = state.states.shape[:-1]
        with self._dev():
            seed = jnp.broadcast_to(key_seed(key), (*n, 2))
            act = jnp.zeros((0,), jnp.int32) if action is None else jnp.asarray(action).astype(jnp.int32)
            if next_task_ids is None and self.sampler is not None and self.autoreset:
                next_task_ids = self.sampler.sample(state.sampling_seed, state.env_id, state.reset_counts)
            nxt = (
                jnp.zeros((0,), jnp.int32) if next_task_ids is None else self._checked(next_task_ids, "next_task_ids")
            )
            outs = _ffi.env_step(
                self.handle,
                state.states,
                state.task_id,
                state.steps,
                state.draws,
                state.count,
                state.counts,
                state.views,
                state.goal_pos,
                state.goal_neg,
                state.at_goal,
                state.env_id,
                seed,
                act,
                nxt,
                self.label_width if self.labels else 0,
                self.final_state,
            )
        (states, task_id, steps, draws, count, counts, views, gpos, gneg, at_goal, reward, terminated, truncated, goal,
         invalid, schema, binding, final) = outs  # fmt: skip
        reset_counts = state.reset_counts
        if self.autoreset:
            reset_counts = reset_counts + (terminated | truncated).astype(jnp.uint32)
        s = state._replace(
            states=states, task_id=task_id, steps=steps, draws=draws, count=count, counts=counts, views=views,
            goal_pos=gpos, goal_neg=gneg, reset_counts=reset_counts, at_goal=at_goal,
        )
        info: dict[str, Any] = {"goal": goal, "invalid": invalid, "schema": schema}
        if self.labels:
            info["binding"] = binding
        if self.final_state:
            info["final_state"] = final
        if not self.native.autoreset:
            info["no_autoreset"] = ()
        obs = self.observation(s)
        if self.final_state and self.observation_encoders is not None:
            # the reached states in the instances and goals before the step (the observation without autoreset)
            info["final_obs"] = (self.encode(final, state.task_id, state.goal_pos, state.goal_neg)
                                 if self.native.autoreset else obs.encoding)
        return s, obs, reward, terminated, truncated, info

    def random_actions(self, key: Any, state: EnvState, max_actions: Optional[int] = None) -> Any:
        """The random policy's next choices without stepping (int32 [N]): the successor slots ``step(key, state)``
        would take (Philox of (key_seed(key), env_id, draws), in jnp), limited to the first
        ``max_actions`` successors (None: no limit); 0 where there is no successor. ``step(key, state,
        random_actions(key, state))`` equals ``step(key, state)`` when no count exceeds max_actions."""
        from . import rng

        n = state.count.shape
        seed = jnp.broadcast_to(key_seed(key), (*n, 2))
        count = state.count.astype(jnp.int32)
        if max_actions is not None:
            if int(max_actions) < 1:
                raise ValueError("mymyr: max_actions must be None or positive")
            count = jnp.minimum(count, jnp.int32(max_actions))
        lo, hi = rng.bits64(seed, state.env_id, state.draws, rng.PURPOSE_SUCCESSOR)
        pick = rng.below(lo, hi, jnp.maximum(count, 1).astype(jnp.uint32)).astype(jnp.int32)
        return jnp.where(count > 0, pick, 0)

    # ---------------------------------------------------------------------------------------------- expansions

    def expand(self, states: Any, k: int, task_ids: Optional[Any] = None, derived: bool = False, *, hashes: bool = False, noops: bool = False, drop_noops: bool = False) -> PaddedExpansion:
        """The padded expansion [N, K] of uint32 state rows [N, 2 row_words] of instances ``task_ids`` [N] (None: a
        table of one) in canonical order (count > K overflows). On a CUDA device it synchronizes on the successor count
        (it is not the environment step). ``derived=True`` adds the successors' derived words (``table.derived_words``
        words, as uint32 pairs), the result's ``derived`` field."""
        with self._dev():
            ids = self._ids(task_ids, states.shape[0])
            dw = self.derived_words if derived else None
            out = _ffi.expand(self.handle, states, ids, int(k), self.label_width, dw,
                              hashes=hashes, noops=noops, drop_noops=drop_noops)
            if hashes or noops or drop_noops:
                return PaddedExpansion(*out[:6], out[6] if derived else None, out[7] if hashes else None,
                                       out[8] if noops or drop_noops else None, out[9] if drop_noops else None, out[10][0])
            return PaddedExpansion(*out, generated=jnp.sum(out[5]))

    def expand_flat(
        self, states: Any, capacity: int, task_ids: Optional[Any] = None, derived: bool = False, *,
        hashes: bool = False, noops: bool = False, drop_noops: bool = False
    ) -> FlatExpansion:
        """The flat expansion of states [N, 2 row_words] of instances ``task_ids`` into ``capacity`` rows (see
        :class:`FlatExpansion`)."""
        with self._dev():
            ids = self._ids(task_ids, states.shape[0])
            dw = self.derived_words if derived else None
            out = _ffi.expand_flat(self.handle, states, ids, int(capacity), self.label_width, dw,
                                   hashes=hashes, noops=noops, drop_noops=drop_noops)
            if hashes or noops or drop_noops:
                return FlatExpansion(*out[:6], out[6] if derived else None, out[7] if hashes else None,
                                     out[8] if noops or drop_noops else None, out[9] if drop_noops else None, out[10][0])
            return FlatExpansion(*out, generated=out[5][-1])

    def set_launch(self, launch: str) -> None:
        """'widest' or 'per_bucket': how the device calls launch over a table's row-width buckets."""
        self.native.set_launch(launch)

    def check_errors(self) -> None:
        """Waits for this environment's device work and raises if a fast-path step met a stale cache (states written
        without :meth:`refresh`) or task ids outside the table."""
        self.native.check_errors()


def _observation(table: Any, observation: Any, capacities: Any, actions: bool) -> tuple[Any, Any, Any]:
    """The checked observation encoders (one per domain), their capacities and whether their schemas have goal
    relations."""
    if observation is None:
        if capacities is not None or actions:
            raise ValueError("mymyr: obs_capacities and obs_actions need an observation encoder")
        return None, None, None
    from collections.abc import Mapping, Sequence

    from mymyr import rl
    from mymyr._core._encoding import FlatRelationEncoder
    from mymyr.encoding import _device

    suite = isinstance(table, _rl.TaskSuite) and table.num_domains > 1
    tables = list(table.tables) if suite else [table.tables[0] if isinstance(table, _rl.TaskSuite) else rl.as_table(table)]
    if hasattr(observation, "encoders"):
        observation = observation.encoders
    encoders = list(observation) if isinstance(observation, (list, tuple)) else [observation]
    if len(encoders) != len(tables):
        raise ValueError(f"mymyr: observation needs {len(tables)} encoders, one per domain of the environments' "
                         f"{'suite' if suite else 'table'}, got {len(encoders)}")
    for d, (encoder, t) in enumerate(zip(encoders, tables, strict=True)):
        if type(encoder) is not FlatRelationEncoder:
            raise TypeError(f"mymyr: observation must be a FlatRelationEncoder (or one per domain), not "
                            f"{type(encoder).__name__}")
        if encoder.table_fingerprint != t.fingerprint:
            where = f"domain {d} of the suite" if suite else "the environments' table"
            raise ValueError(f"mymyr: observation: encoder {d} encodes another table than {where} (encode the table "
                             "the environments run on)")
    if actions:
        raise ValueError("mymyr: obs_actions: the JAX encoding holds no action relations; leave obs_actions off")
    caps = None
    if capacities is not None:
        if isinstance(capacities, Mapping):
            caps = [dict(capacities)] * len(encoders)
        elif isinstance(capacities, Sequence) and not isinstance(capacities, str) and len(capacities) == len(encoders) \
                and all(isinstance(c, Mapping) for c in capacities):
            caps = [dict(c) for c in capacities]
        else:
            raise ValueError(f"mymyr: obs_capacities must be a dict or a sequence of {len(encoders)} dicts (one per "
                             "domain)")
    goals = [any(lane in _device._GOAL_LANES for lane in e.relation_lanes) for e in encoders]
    return encoders, caps, goals


def _masks_hold(states: Any, goal_pos: Any, goal_neg: Any) -> Any:
    """Whether each state row (uint32 halves) holds its goal masks ([N, 2 words]: the rows' first 2 words columns)."""
    w = goal_pos.shape[-1]
    head = states[..., :w]
    return jnp.all((head & goal_pos) == goal_pos, axis=-1) & jnp.all((head & goal_neg) == 0, axis=-1)


def as_words(x: Union[np.ndarray, Any]) -> np.ndarray:
    """uint32 halves [..., 2W] as NumPy uint64 [..., W] (host copy)."""
    a = np.ascontiguousarray(np.asarray(x), dtype=np.uint32)
    return a.view(np.uint64)
