"""PlanningEnv: the TorchRL environment over a :class:`~mymyr.rl.torch.BatchedEnv` (needs torchrl and tensordict)."""

from typing import Any, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from ._batched import BatchedEnv

__all__ = ["PlanningEnv"]


class PlanningEnv(EnvBase):
    """N planning environments over a TaskSuite or a TaskTable (or a Task) as a TorchRL environment (batch_size [N]).

    Observation: ``state`` [N, row_words] int64 (the state words), ``task_id`` [N] int64 (the env's instance) and
    ``count`` [N] int64 (its successors); with ``max_actions=K`` also ``action_mask`` [N, K] bool (successor k
    exists); with ``goals=True`` also ``goal_pos`` / ``goal_neg`` [N, words] int64 (the env's goal as word masks: its
    instance's goal after a reset, or the one given; the step's goal test is then the mask test). Action: ``action``
    [N] int64, an index into the state's successors in canonical order (schema, then binding), Categorical(K) with
    ``max_actions=K`` (successors past K are out of reach), otherwise unbounded (an index outside [0, count) does not
    move). Reward [N, 1] float32: step_reward, plus goal_reward on reaching a goal, plus dead_end_reward on reaching
    (or being stuck in) a dead end. ``terminated`` = goal or dead end (``dead_end_terminal``), ``truncated`` =
    max_steps reached (0: never), ``done`` = either. Finished environments are reset by TorchRL (``_reset`` masks); the
    reset tensordict may carry ``task_id`` [N] (curriculum: the instances the reset rows restart in) and, with goals,
    ``goal_pos`` / ``goal_neg``. The environments never autoreset themselves.

    ``device="cpu"`` runs the host environments (any table, numeric ones included; ``threads`` splits a batch),
    ``"cuda[:i]"`` the device ones, including numeric tables, with identical trajectories for the same actions. Random actions
    (:meth:`rand_action`, used by ``rollout`` without a policy) are the environments' counter-based random policy (the
    seed of ``set_seed``, env id, draw counter; rl/rng.hpp) through the custom op ``mymyr::random_actions``: uniform over
    the current successors (the first ``max_actions`` of them), so a rollout without a policy equals
    ``BatchedEnv.step(None)`` and the host, device and JAX environments for the same seed, on any device and batch
    split. A TaskSuite, a TaskTable and a Task pickle, so ``ParallelEnv`` and collectors can build PlanningEnvs in
    subprocesses. Over a suite, ``task_id`` is global (``mymyr.rl.task_domains`` gives the rows' domains).

    CUDA graphs: on a device environment whose steps can be captured (``batched.capturable``: the fast path), a
    step replays a CUDA graph of the environment step and the observation (captured at the first step; ``graph=False``:
    eager steps), then copies its outputs into the step's TensorDict: the same tensors, with a fraction of the host's
    launch work (TorchRL's own per-step work stays). The graph reads the environments' tensors in place, so resets,
    ``set_seed`` and writes into ``batched``'s tensors between steps apply as with eager steps. Steps traced by
    torch.compile, or captured into an enclosing CUDA graph, are eager.
    """

    def __init__(
        self,
        table: Any,
        num_envs: int,
        *,
        task_ids: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
        max_actions: Optional[int] = None,
        goals: bool = False,
        seed: int = 0,
        max_steps: int = 0,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        dead_end: str = "no_successors",
        dead_end_reward: float = 0.0,
        dead_end_terminal: bool = True,
        witness: bool = False,
        path: str = "auto",
        threads: int = 1,
        ctx: Any = None,
        first_env: int = 0,
        graph: bool = True,
    ) -> None:
        env = BatchedEnv(
            table, num_envs, task_ids=task_ids, device=device, goals=goals, seed=seed, max_steps=max_steps,
            step_reward=step_reward, goal_reward=goal_reward, dead_end=dead_end, dead_end_reward=dead_end_reward,
            dead_end_terminal=dead_end_terminal, autoreset=False, witness=witness, path=path, threads=threads, ctx=ctx,
            first_env=first_env,
        )  # fmt: skip
        super().__init__(device=env.device, batch_size=torch.Size([num_envs]))
        self._env = env
        self._K = None if max_actions is None else int(max_actions)
        self._graph_on = bool(graph) and env.capturable
        self._graph: Any = None  # the step's torch.cuda.CUDAGraph, captured at the first step
        self._static: Optional[dict] = None
        n, d = int(num_envs), env.device
        if self._K is not None and self._K <= 0:
            raise ValueError("mymyr: max_actions must be positive")
        obs = {
            "state": Unbounded(shape=(n, env.row_words), dtype=torch.int64, device=d),
            "task_id": Unbounded(shape=(n,), dtype=torch.int64, device=d),
            "count": Unbounded(shape=(n,), dtype=torch.int64, device=d),
        }
        if self._K is not None:
            obs["action_mask"] = Categorical(n=2, shape=(n, self._K), dtype=torch.bool, device=d)
        if env.goals:
            obs["goal_pos"] = Unbounded(shape=(n, env.words), dtype=torch.int64, device=d)
            obs["goal_neg"] = Unbounded(shape=(n, env.words), dtype=torch.int64, device=d)
        self.observation_spec = Composite(obs, shape=(n,), device=d)
        if self._K is not None:
            self.action_spec = Categorical(n=self._K, shape=(n,), dtype=torch.int64, device=d)
        else:
            self.action_spec = Unbounded(shape=(n,), dtype=torch.int64, device=d)
        self.reward_spec = Unbounded(shape=(n, 1), dtype=torch.float32, device=d)
        flag = Categorical(n=2, shape=(n, 1), dtype=torch.bool, device=d)
        self.done_spec = Composite(done=flag.clone(), terminated=flag.clone(), truncated=flag.clone(), shape=(n,), device=d)

    @property
    def batched(self) -> BatchedEnv:
        """The underlying :class:`BatchedEnv` (its tensors are the environments' state)."""
        return self._env

    @property
    def graph(self) -> bool:
        """Whether steps replay a CUDA graph (``graph=True`` on a device environment whose steps can be captured)."""
        return self._graph_on

    def _observation(self, copy: bool = True) -> dict:
        e = self._env
        keep = (lambda x: x.clone()) if copy else (lambda x: x)
        obs = {"state": keep(e.states), "task_id": e.task_ids.to(torch.int64), "count": e.count.to(torch.int64)}
        if self._K is not None:
            obs["action_mask"] = torch.arange(self._K, device=e.device).unsqueeze(0) < e.count.unsqueeze(1)
        if e.goals:
            obs["goal_pos"], obs["goal_neg"] = keep(e.goal_pos), keep(e.goal_neg)
        return obs

    def _outputs(self, action: Any, copy: bool = True) -> dict:
        """The step's outputs: the observation after stepping with ``action``, reward and flags."""
        r = self._env.step(action)
        out = self._observation(copy)
        out.update(
            reward=r.reward.unsqueeze(-1),
            done=(r.terminated | r.truncated).unsqueeze(-1),
            terminated=r.terminated.unsqueeze(-1),
            truncated=r.truncated.unsqueeze(-1),
        )
        return out

    def _capture(self) -> None:
        """Captures the step (an action buffer in, the outputs as the graph's tensors: the environments' tensors
        themselves where the observation is one) into a CUDA graph; the capture runs nothing."""
        e = self._env
        self._action = torch.zeros(e.num_envs, dtype=torch.int64, device=e.device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static = self._outputs(self._action, copy=False)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs: Any) -> TensorDictBase:
        mask = task_ids = goal_pos = goal_neg = None
        if tensordict is not None:
            keys = tensordict.keys()
            if "_reset" in keys:
                mask = tensordict.get("_reset").reshape(self._env.num_envs)
            if "task_id" in keys:
                task_ids = tensordict.get("task_id")
            if self._env.goals and "goal_pos" in keys and "goal_neg" in keys:
                goal_pos, goal_neg = tensordict.get("goal_pos"), tensordict.get("goal_neg")
        self._env.reset(mask, task_ids=task_ids, goal_pos=goal_pos, goal_neg=goal_neg)
        out = self._observation()
        z = torch.zeros((self._env.num_envs, 1), dtype=torch.bool, device=self._env.device)
        out.update(done=z, terminated=z.clone(), truncated=z.clone())
        return TensorDict(out, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get("action")
        if self._graph_on and not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing():
            if self._graph is None:
                self._capture()
            self._action.copy_(action.reshape(self._env.num_envs))
            self._graph.replay()
            out = {k: v.clone() for k, v in self._static.items()}
        else:
            out = self._outputs(action)
        return TensorDict(out, batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: Optional[int]) -> None:
        s = 0 if seed is None else int(seed)
        self._env.set_seed(s)

    def rand_action(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        """A uniform successor index of each environment's current state (0 where there is none), from the counter-based
        RNG at the environments' draw counters (the choice their random-policy step would make; the custom op
        ``mymyr::random_actions``, so it traces under torch.compile)."""
        action = self._env.random_actions(self._K or 0)
        if tensordict is None:
            return TensorDict({"action": action}, batch_size=self.batch_size, device=self.device)
        tensordict.set("action", action)
        return tensordict

    def close(self, *, raise_if_closed: bool = True) -> None:
        self._graph = self._static = None
        self._env.close()
        super().close(raise_if_closed=raise_if_closed)
