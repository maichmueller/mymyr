"""PyTorch environments and ops over mymyr task tables and suites.

Three layers, all on the CPU or a CUDA device with the same results bit for bit, over a ``mymyr.rl.TaskTable`` (many
instances of one domain; each env has its instance, ``task_ids``), a ``mymyr.rl.TaskSuite`` (instances of several
domains; global task ids, so one batch mixes domains and an autoreset may change an env's domain) or a Task (its table
of one):

- :class:`PlanningEnv`: a TorchRL ``EnvBase`` of N environments (needs ``torchrl`` and ``tensordict``)::

      env = mymyr.rl.torch.PlanningEnv(table, 4096, task_ids=ids, device="cuda", max_steps=200, goal_reward=1.0,
                                       max_actions=64)
      td = env.rollout(100, break_when_any_done=False)   # state, task_id, count, action_mask, action, reward, ...

- :class:`BatchedEnv`: the same environments without TorchRL: state tensors on the device plus ``reset(mask,
  task_ids=...)``, ``step(action, next_task_ids=...)`` (``action=None``: the random policy of the counter-based RNG)
  and ``refresh()``. The device step makes no host round trip.

- torch custom ops (``torch.library``; :mod:`mymyr.rl.torch._ops`): ``torch.ops.mymyr.reset / refresh / step /
  random_actions`` on an environment handle, and ``torch.ops.mymyr.expand`` (padded [N, K]) / ``expand_flat`` (CSR;
  ``capacity=None``: the row count is an unbacked symbolic size under ``torch.compile``) on a table or suite handle
  (:func:`register`). All have fake implementations: they trace under ``torch.compile`` without graph breaks
  (``fullgraph=True`` works for loops of ``step``; the TorchRL env's own Python and TensorDict code is not meant to be
  compiled).

CUDA graphs: on a device's fast path the ops synchronize nothing and allocate nothing, so steps capture into CUDA
graphs and replay byte for byte like the eager calls: :class:`CapturedRollout` (``BatchedEnv.capture(K, policy)``: K
steps and the policy in one graph, one launch per replay), ``torch.cuda.graph`` around BatchedEnv calls, or
``torch.compile(mode="reduce-overhead")`` of a function that steps a BatchedEnv (its tensors are marked static).
PlanningEnv replays a captured graph of its step (the env step and the observation) at every TorchRL step.

Semantics of a step (the host reference rl::HostEnv, include/mymyr/rl/env.hpp): the action indexes the current state's
successors in canonical order; the environment moves there (an index outside [0, count) does not move and is
flagged ``invalid``; a state without successors is stuck: no move, terminated). reward = step_reward (+ goal_reward if
the reached state is a goal, + dead_end_reward if it is a non-goal state without successors); terminated = goal or
dead end (``dead_end="no_successors"``, terminal unless ``dead_end_terminal=False``; ``"none"``: no dead ends);
truncated = not terminated and max_steps > 0 and the episode reached max_steps steps. With autoreset (BatchedEnv's
default) a finished environment restarts from the initial state of its instance (or of ``next_task_ids[i]``) within
the step (``final=True`` returns the reached states); PlanningEnv leaves resets to TorchRL. With ``goals=True`` each
env carries goal masks (``goal_pos`` / ``goal_neg``) and the goal test is the mask test (relabelled or multi-goal
envs).

Random draws (:mod:`mymyr.rl.torch.rng`): Philox-4x32-10 counters (seed, env id, draw counter), so an environment's
trajectory does not depend on the batch, the thread count, the device or the launch configuration.

Device environments support numeric tables through the general path. The fast device path (no host
synchronization at all) needs frozen atom slots, no axioms and device-complete schemas in every instance
(``fast_unsupported(table)`` says why not); other tables take the general path (a device expansion per step, which
synchronizes on the successor count). Over a suite the device groups each step's rows by domain and runs every domain
on its table's path, on a stream of its own (captured graphs included).
"""

from mymyr._core._rl_torch import Env, fast_unsupported

from . import rng
from ._batched import BatchedEnv, StepResult
from ._graph import CapturedRollout, Rollout
from ._ops import (
    expand,
    expand_flat,
    lookup,
    random_actions,
    refresh,
    refresh_sync,
    register,
    reset,
    step,
    step_sync,
    unregister,
)

__all__ = [
    "BatchedEnv",
    "CapturedRollout",
    "Env",
    "PlanningEnv",
    "Rollout",
    "StepResult",
    "expand",
    "expand_flat",
    "fast_unsupported",
    "lookup",
    "random_actions",
    "refresh",
    "refresh_sync",
    "register",
    "reset",
    "rng",
    "step",
    "step_sync",
    "unregister",
]


def __getattr__(name: str):
    if name == "PlanningEnv":  # imports torchrl on first use
        from ._planning import PlanningEnv

        return PlanningEnv
    raise AttributeError(f"module 'mymyr.rl.torch' has no attribute {name!r}")
