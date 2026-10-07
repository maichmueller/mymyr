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

## JAX and PyTorch

The optional `mymyr.rl.jax.Env` is a functional environment whose `reset` and `step` methods work under
`jax.jit`, `jax.vmap` and `jax.lax.scan`. JAX device arrays can use the CUDA path. `mymyr.rl.torch.BatchedEnv` provides
batched tensor environments; `mymyr.rl.torch.PlanningEnv` adapts them to TorchRL and needs `torchrl` and `tensordict`.
Both environments use successor indices as discrete actions. JAX, PyTorch and TorchRL are optional dependencies.

On CUDA, array exchange uses DLPack without copying and follows the producer/consumer stream protocol. CPU tasks with
numeric fluents are supported by the host RL APIs; device environments and device expansion currently require
classical tasks. mymyr supplies expansion, environment and interoperability primitives, not encoders or replay buffers.

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
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl", atoms="frozen")
env = BatchedEnv(task, 4, device=torch.device("cuda", 0), max_steps=4, seed=0)
step = env.step(final=True)
print(env.states.shape, step.reward.shape)
```
