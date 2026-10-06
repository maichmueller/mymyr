"""CapturedRollout: K steps of a BatchedEnv (and its policy) captured into one CUDA graph."""

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._batched import BatchedEnv

__all__ = ["CapturedRollout", "Rollout"]


class Rollout(NamedTuple):
    """The outputs of a captured rollout's K steps, stacked over the steps ([K, N, ...] on the environments' device;
    the same tensors after every replay, overwritten by the next one)."""

    action: Optional[Tensor]  # int64 [K, N]: the policy's actions (None: the random policy)
    reward: Tensor  # float32 [K, N]
    terminated: Tensor  # bool [K, N]
    truncated: Tensor  # bool [K, N]
    goal: Tensor  # bool [K, N]
    invalid: Tensor  # bool [K, N]
    schema: Tensor  # int32 [K, N]
    binding: Tensor  # int32 [K, N, label_width]
    final_states: Tensor  # int64 [K, N, row_words] with final=True, else [K, 0, row_words]
    count: Tensor  # int32 [K, N]: successors after each step


class CapturedRollout:
    """``steps`` steps of a device :class:`BatchedEnv` captured into one CUDA graph (``torch.cuda.CUDAGraph``);
    :meth:`replay` runs them again with a single launch, continuing from the environments' current state.

    Step k of the graph is ``env.step(policy(env), next_task_ids=next_task_ids, final=final)`` (``policy=None``: the
    random policy, ``env.step(None)``). A replay equals these calls made eagerly, byte for byte (states, task ids,
    steps, draw counters, the cache, goals and every output), from whatever the environments' tensors hold when it
    runs: write into ``env.states`` (then :meth:`BatchedEnv.refresh`, eagerly), reset rows or change the seed between
    replays as with eager steps. ``policy(env) -> action`` [N] int64 must be capturable: torch operations on the
    device without host synchronization (a network over ``env.states``, a sampler over ``env.random_actions()``); it
    runs twice on a side stream before the capture to warm up (lazy initializations such as cuBLAS handles), without
    stepping. ``next_task_ids`` [N] int32 is read at every replay: write the next instances into it between replays.

    Needs a device environment whose calls can be captured (``env.capturable``: the fast path). The graph keeps
    pointers into the environments' tensors and its scratch: it holds the BatchedEnv, whose tensors must not be
    replaced (assign into them in place). ``pool`` shares a memory pool with other graphs (``torch.cuda.graph``).
    """

    def __init__(
        self,
        env: "BatchedEnv",
        steps: int,
        policy: Optional[Callable[["BatchedEnv"], Tensor]] = None,
        *,
        next_task_ids: Optional[Tensor] = None,
        final: bool = False,
        pool: Any = None,
    ) -> None:
        if env.device.type != "cuda":
            raise ValueError("mymyr: CUDA graphs capture device environments; this BatchedEnv runs on the CPU")
        why = env.core.capture_unsupported
        if why:
            raise ValueError(f"mymyr: this BatchedEnv's steps cannot be captured into a CUDA graph: {why}")
        if steps <= 0:
            raise ValueError("mymyr: a captured rollout needs steps > 0")
        if next_task_ids is not None:
            n = env.num_envs
            if next_task_ids.dtype != torch.int32 or next_task_ids.device != env.device or tuple(next_task_ids.shape) != (n,):
                raise ValueError(f"mymyr: next_task_ids must be an int32 tensor of shape [{n}] on {env.device}")
        self.env = env
        self.steps = int(steps)
        self.policy = policy
        self.next_task_ids = next_task_ids
        self.final = bool(final)
        dev = env.device
        if policy is not None:
            side = torch.cuda.Stream(dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    policy(env)
            torch.cuda.current_stream(dev).wait_stream(side)
        from ._batched import StepResult

        self.graph = torch.cuda.CUDAGraph()
        K, n = self.steps, env.num_envs
        with torch.cuda.graph(self.graph, pool=pool):
            # the steps write into slices of the stacked outputs (no stacking copies)
            b = torch.empty((K, n), dtype=torch.bool, device=dev)
            rows = n if final else 0
            out = StepResult(
                torch.empty((K, n), dtype=torch.float32, device=dev), b, torch.empty_like(b), torch.empty_like(b),
                torch.empty_like(b), torch.empty((K, n), dtype=torch.int32, device=dev),
                torch.empty((K, n, env.label_width), dtype=torch.int32, device=dev),
                torch.empty((K, rows, env.row_words), dtype=env.states.dtype, device=dev),
                torch.empty((K, n), dtype=torch.int32, device=dev),
            )  # fmt: skip
            actions = None if policy is None else torch.empty((K, n), dtype=torch.int64, device=dev)
            for k in range(K):
                a = None
                if actions is not None:
                    a = actions[k]
                    a.copy_(policy(env).reshape(n))
                env._step_into(StepResult(*(x[k] for x in out)), a, next_task_ids, final)
            self._result = Rollout(actions, *out)

    @property
    def result(self) -> Rollout:
        """The outputs of the last replay (the capture itself ran nothing)."""
        return self._result

    def replay(self) -> Rollout:
        """Runs the K captured steps on torch's current stream; returns :attr:`result` (the same tensors every time)."""
        self.graph.replay()
        return self._result

    def __repr__(self) -> str:
        return f"CapturedRollout(steps={self.steps}, policy={self.policy is not None}, env={self.env!r})"

