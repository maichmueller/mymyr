# Reinforcement learning

`mymyr.rl.expand(table, states)` batches action expansion. It returns successor rows and CSR offsets, along with
parent ids and action labels (`schema` and `binding`). NumPy, CPU torch and CPU JAX arrays use the CPU path; the output
uses the input array's framework and word encoding. Inputs are read in place and outputs are views over a mymyr-owned
buffer.

```python
from pathlib import Path

import mymyr
from mymyr import rl

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
states = [task.initial_state]
expansion = rl.expand(task, states, goal=True)

print(expansion.succ.shape, expansion.schema, expansion.binding, expansion.offsets)
pool = rl.CpuEnvPool(task, num_envs=4, seed=0, max_steps=20)
pool.reset()
step = pool.step()
print(step.states.shape, step.reward, step.terminated)
```

`TaskTable` groups instances of one domain and `TaskSuite` groups tables from different domains. Pass `task_ids` with a
batch over a table or suite. The `CpuEnvPool` provides asynchronous CPU environments with `send`/`recv`; `step` is a
convenience for enqueueing and waiting for one batch. The environment action is an index into each state's canonical
successor order.

States travel as rows of atom words followed by numeric words, in the encoding of
[State encodings](parsing-and-tasks.md#state-encodings). Tables refuse tasks with lazy atom slots and use one numeric
storage for all instances. A batch's rows are as wide as their widest instance (`words=` fixes the width); each row
costs its own instance's width, and the words past it are zero padding. `rl.numeric_values` decodes numeric words.

## JAX and PyTorch

The optional `mymyr.rl.jax.Env` is a functional environment whose `reset` and `step` methods work under
`jax.jit`, `jax.vmap` and `jax.lax.scan`. JAX device arrays can use the CUDA path. `mymyr.rl.torch.BatchedEnv` provides
batched tensor environments; `mymyr.rl.torch.PlanningEnv` adapts them to TorchRL and needs `torchrl` and `tensordict`.

Goal states and terminal dead ends are absorbing until reset. Stepping them preserves the state and episode step
counter, returns zero reward and a terminal flag, and reports no action label (`schema=-1`). Reaching a goal or dead
end from a nonterminal state still receives the configured transition reward once. Draw counters advance once per
native step, including absorbing steps.

Every environment reports the reset states that are goals already: solved initial states, and states that satisfy
the goals given to `reset(goal_pos=..., goal_neg=...)`. `CpuEnvPool.reset()` sets `goal` and `terminated`; TorchRL
resets set `terminated` (also for terminal dead ends); torch `BatchedEnv` and JAX `Env` keep a per-environment
`at_goal` flag, which reset, step and `refresh()` write and observations and step results carry. A step reads the
flag instead of testing the current state again, so the next step absorbs those rows. Writing states, instances or
goal masks directly needs `refresh()`.
The mctx adapter reads the terminal test of the root and of each node from `at_goal` and the successor count, without
a step; it preserves the entire embedding at absorbing nodes and sets reward, discount and value to zero.
Zero-successor tree nodes are terminal even when the environment disables dead-end termination.
Both environments use successor indices as discrete actions. JAX, PyTorch and TorchRL are optional dependencies.
A padded action space of K slots (`PlanningEnv(max_actions=K)`, padded expansions) holds a state's first K successors
in canonical order: later successors are out of reach even when only they lead to a goal (a padded expansion reports
such states by `count > K`).

## Transition ownership and replay

Environment and HER defaults use a sparse reach-goal objective: `step_reward=0`, `goal_reward=1`,
`dead_end_reward=0`. Successful episodes therefore beat early failures instead of paying a negative cost for each
additional step toward success. Discounting below one favours earlier success; undiscounted sparse returns do not
distinguish successful plan lengths. For unit-cost shaping, configure `step_reward=-1` explicitly and choose goal
bonuses/failure penalties consistently with the episode horizon and discount. These rewards do not automatically
optimize a PDDL action-cost metric.

Torch `BatchedEnv.observation()` snapshots states, task ids, counts and per-environment goals. `step()` returns those
next-observation tensors in `StepResult`, along with `final_states`, the reached states before autoreset. These
snapshots survive subsequent steps and resets. The public environment tensors are live buffers for native calls and
CUDA graphs: use `observation(copy=False)` or `step(copy=False)` only when intentionally borrowing them, and consume
or copy them before the next mutation. Captured rollout results remain borrowed graph outputs overwritten on replay.

JAX `transition_item` records the source observation's instance and goals and uses the reached state, not the
autoreset observation, as its next state. Autoreset replay requires `Env(final_state=True)` (the default); the helper
raises rather than recording an unrelated reset row when final states are disabled. An environment without autoreset
may disable final states because its next observation is the reached state. Goal-conditioned observations propagate
their source masks automatically; initialize replay with `transition_example(env, goals=True)` for those items.

HER helpers recompute `achieved` and `reward` from the transitions' reached states. Under a relabelled goal the episode
ends at its first state that satisfies the goal, so `valid` marks the relabels whose transition is still part of that
episode: no reached state of the episode's earlier steps in the window satisfies the goal. With `prev_states` (the
states the steps started from: the observations) the episode's first state, its reset state or the state before the
window, is tested too; without them a goal that already held there goes undetected. Drop the relabels with `valid`
false.

The Python 3.14t CI lane installs `ci/rl-requirements.txt` before building, verifies that the framework imports and
XLA FFI support are present, and runs the ordinary suite with those adapters enabled. The pinned file records the
CPU framework bundle used for validation; those packages remain optional runtime dependencies for users.

## Instance sampling and curricula

Without constructor `task_ids`, the torch environments and `CpuEnvPool` sample uniformly across a table or suite at
each reset.
Explicit constructor ids retain a fixed assignment (`sampler="auto"`); `sampler="uniform"` enables sampling even with
an explicit initial assignment. JAX `Env.reset(key, N)` samples N instances uniformly; `reset(key, ids)` overrides
the initial draw. In either framework, `sampler=None` disables sampling and
`sampler=rl.InstanceSampler(num_instances, weights=[...])` selects a weighted distribution. Weights must be finite,
nonnegative and not all zero; zero-weight instances are excluded.

Instance draws use the seed, environment id (`first_env + i`) and a per-environment uint32 reset counter, independent
of action selection. Masked resets advance only reset rows, so splitting a batch or resetting another row cannot
change a stream. Torch `set_seed(seed)` restarts both action and sampling counters. JAX keeps sampling keys and
counters in `EnvState`; `reset_where(state, mask, key=new_key)` reseeds only the reset rows. Its sampler contains no
mutable host RNG and works with `jit`, `vmap` and `scan`.

External curricula override a draw with `reset(task_ids=ids)` or `step(..., next_task_ids=ids)`. A TorchRL reset
TensorDict uses `next_task_id`, not the observation's `task_id`: collectors therefore do not accidentally pin each
new episode to the preceding one. Likewise custom reset goals use `next_goal_pos` / `next_goal_neg` or explicit
`reset(goal_pos=..., goal_neg=...)` kwargs; observed goal masks do not override a new instance's default goals.
`CpuEnvPool(sampler=...)` takes the same samplers (`"auto"` and `"uniform"` sample uniformly, `None` starts every
env at instance 0 and keeps its instance) and draws what a torch `BatchedEnv` with the same seed and sampler draws: the
pool's env id is its index, and each reset and each autoresetting step of an env advances that env's counter, whichever
send or thread steps it. `reset(task_ids=ids)` and `send(..., next_task_ids=ids)` override the draws.

## CUDA

On CUDA, array exchange uses DLPack without copying and follows the producer/consumer stream protocol. Host and device
environments and expansion support numeric fluents, preconditions, effects and goals, preserving the CPU I32/F64 row
encoding. Numeric tables and suites run per-instance kernels. Numeric environments use the general path, which
synchronizes; the environment fast path and CUDA graph capture require tasks that path can run. mymyr supplies
expansion, environment and interoperability primitives and [native classical encodings](encoding.md), not replay buffers.

The following JAX rollout uses CUDA and the JAX/CUDA package in the optional CUDA environment:

```python skip-if-no-cuda
from pathlib import Path

import jax
import mymyr
from mymyr.rl.jax import Env

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
env = Env(task, device="cuda", max_steps=4)
key = jax.random.key(0)
state, observation = env.reset(key, 4)
state, observation, reward, terminated, truncated, info = jax.jit(env.step)(key, state)
print(state.states.shape, reward.shape)
env.check_errors()
```

The torch environment can also run a batch of discrete successor choices directly on CUDA:

```python skip-if-no-cuda
from pathlib import Path

import mymyr
import torch
from mymyr.rl.torch import BatchedEnv

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl", atoms="dense")
env = BatchedEnv(task, 4, device=torch.device("cuda", 0), max_steps=4, seed=0)
step = env.step(final=True)
print(env.states.shape, step.reward.shape)
```

## A learning smoke example

Run `python examples/rl_learning.py` for a NumPy REINFORCE successor scorer using native goal-relation counts and
action-schema readouts. It trains on three chain instances with 3–5 objects, then evaluates on held-out 8- and
12-object instances against a random policy. The report includes success, solved plan length, instance coverage,
truncations, elapsed time and a counted subset of live array storage (not process peak memory).
The scorer is intentionally small: it pools goal counts and scores action schemas, not object-level relational
reasoning. It exercises the path from collection through native encoding to a policy update, not general planning ability.
