# Reinforcement learning

Options follow the [API conventions](conventions.md): `threads=` (default 1, `0`: one per hardware thread), `seed=`,
`max_steps=` (`None`: no truncation).

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
costs its own instance's width, and the words past it are zero padding. `rl.numeric_values` decodes numeric words,
and `rl.decode(table, rows, task_ids)` turns rows back into States (`Task.decode` reads rows in the task's own
layout, with only its own numeric words last).

## Derived words

Derived atoms are never stored in states. `rl.expand(..., derived=True)` also returns, for every successor, the bitset of
its derived atoms (`expansion.derived`, `[M, D]` words in the encoding of the states, bit `d` of the row for derived slot
`d`), from one axiom evaluation of the successor row; the same evaluation yields the goal flag. `D` is
`table.derived_words` (`suite.derived_words`: the widest domain; `task.derived_words` for one task), and 0 for tasks
without derived atoms, whose arrays are zero-width. An instance with fewer derived words has zeros past its own. The
words are exactly what the [relational encoder](encoding.md) evaluates itself, so they can replace its evaluation.
Capacity and overflow follow `succ`: rows past the capacity are counted, not written; `derived=False` leaves every other
output unchanged and costs nothing. `expand_into` takes a `derived=` destination (it needs `succ=`), `exp.pad(k)` /
`k=` carry `padded.derived` `[N, k, D]` with zero padding, and device inputs fill the words on the device (every device
expansion path, including tables of instances with axioms and suites). The torch ops `expand_derived` and
`expand_flat_derived` return the words as a seventh result, and `Env.expand(..., derived=True)` /
`Env.expand_flat(..., derived=True)` of `mymyr.rl.jax` fill the `derived` field (uint32 pairs).

```python
from pathlib import Path

import mymyr
from mymyr import rl

data = Path("tests/data/pddl/features")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
expansion = rl.expand(task, [task.initial_state], derived=True)
print(expansion.derived.shape)
```

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

## Encoding the pool's states

The [relational encoder](encoding.md) reads the arrays of a `CpuEnvPool` batch where they are. A pool created with
`goals=True` carries the goal of every environment as masks (`goal_pos`, `goal_neg`), and the expansion of the batch's
states, which `rl.expand` makes, is what the policy's actions index. One call encodes all of it:

```python
import numpy as np
import mymyr
from mymyr import encoding, rl

task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
table = rl.as_table(task)
encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
pool = rl.CpuEnvPool(task, num_envs=4, seed=0, max_steps=20, goals=True)
step = pool.reset()

expansion = rl.expand(table, step.states, step.task_ids)
batch = encoder.encode(step.states, step.task_ids, goals=(step.goal_pos, step.goal_neg), actions=expansion)
assert batch.num_graphs == 4

# the same call on copies of everything gives the same batch
copies = encoder.encode(
    np.array(step.states), np.array(step.task_ids), goals=(np.array(step.goal_pos), np.array(step.goal_neg)),
    actions=rl.expand(table, np.array(step.states), np.array(step.task_ids)),
)
assert all(np.array_equal(batch.as_dict()[k], copies.as_dict()[k]) for k in batch.as_dict())
```

Nothing is copied on the way in: the states, task ids and goal masks are read in place, and the encoder takes the
action rows (schemas and bindings) from the expansion instead of computing the applicable actions again. A table of
several instances needs the batch's `task_ids`, which the pool provides. The batch owns its memory, so it stays valid
when the pool moves on.

### Action entities

With action relations (`include_actions=True`, the default) and `actions=`, every distinct action of a state is an
entity of its graph, listed as a target when the encoder has `target_sources=("action",)`. A policy that scores the
action targets chooses entities, and `mymyr.encoding` maps a chosen entity back to the expansion:

- `encoding.action_rows(batch, expansion, entities)` is the row of `expansion` of each chosen entity
  (`expansion.offsets[g] + entity_ref` for the entity of graph `g`), so `expansion.succ[row]` is the successor the
  action leads to and `expansion.action(row)` the action;
- `encoding.action_indices(batch, entities)` is each entity's position among its own state's successors, which is what
  `CpuEnvPool.step` (and every environment of `mymyr.rl`) takes as the action;
- without `entities` both cover every action entity of the batch, in entity order
  (`encoding.action_entities(batch)` lists their ids).

An expansion lists an action once per action schema that produces it (a disjunctive precondition splits a schema), and
an entity stands for the first of those rows. The small loop below encodes the pool's states, lets a random policy
pick one action entity per graph, and steps the pool with it; the state each environment reaches is the successor in
the entity's expansion row.

```python
rng = np.random.default_rng(0)
total_reward = np.zeros(4)
for _ in range(25):
    expansion = rl.expand(table, step.states, step.task_ids)
    batch = encoder.encode(step.states, step.task_ids, goals=(step.goal_pos, step.goal_neg), actions=expansion)

    targets = np.asarray(batch.target_entity)[np.asarray(batch.target_group) == 2]   # the action targets (group 2)
    owner = np.searchsorted(np.asarray(batch.entity_offsets), targets, side="right") - 1
    chosen = np.array([rng.choice(targets[owner == g]) for g in range(batch.num_graphs)])

    rows = encoding.action_rows(batch, expansion, chosen)
    step = pool.step(step.env_ids, encoding.action_indices(batch, chosen))
    reached = np.asarray(step.final_states)[:, : expansion.succ.shape[1]]
    assert np.array_equal(reached, np.asarray(expansion.succ)[rows])
    total_reward += np.asarray(step.reward)
print(total_reward)
```

### Reusing a batch's memory

`encode(..., capacities=...)` gives every array the same size from call to call, which `jax.jit` and a fixed-shape
model need. A call takes new memory for its batch by default. Passing an earlier batch of the same capacities
as `into=` (the same schema and table, the same number of rows) writes the encoding into that batch's arrays instead,
and returns the batch. Every array of the batch is rewritten as a fresh encode would make it: the Simple graph, the
numeric values and, with `statics="shared"`, the static blocks of the call's instances, which change with the instance
set of each call:

```python
capacities = dict(graphs=4, entities=600, args=6000, targets=300)
expansion = rl.expand(table, step.states, step.task_ids)
reused = encoder.encode(step.states, step.task_ids, actions=expansion, capacities=capacities)
exported = reused.relation_args                     # a view of the batch's memory
before = np.array(exported)

step = pool.step(step.env_ids, np.zeros(4, np.int32))
expansion = rl.expand(table, step.states, step.task_ids)
encoder.encode(step.states, step.task_ids, actions=expansion, capacities=capacities, into=reused)
assert not np.array_equal(exported, before)         # the exported view shows the new batch
```

Reuse is opt-in because it aliases: every array and every tensor exported from the batch before the call (NumPy views,
`as_dict("torch")`, `as_dict("jax")`) shares the memory that the call overwrites, and nothing waits for readers or
copies on write. Use it when the previous contents are no longer read (a model that has consumed the tensors, a rollout
step that is finished), and copy what must survive. The call raises `ValueError` before writing if the batch does not
fit (another schema or table, another number of rows, other capacities, or a batch encoded without capacities). A
batch that does not fit its capacities becomes all zero, as it does without `into=`, and `batch.sizes` says so.

### Tensors and arrays

The batch's arrays export without a copy to PyTorch and to JAX on the CPU; the tensors share the memory of the batch
and keep it alive:

```python needs-torch
import torch

tensors = batch.as_dict("torch")
assert isinstance(tensors["relation_args"], torch.Tensor)
assert tensors["relation_args"].data_ptr() == batch.relation_args.__array_interface__["data"][0]
```

```python needs-jax
import jax

arrays = batch.as_dict("jax")
assert isinstance(arrays["relation_args"], jax.Array)
assert arrays["relation_args"].unsafe_buffer_pointer() == batch.relation_args.__array_interface__["data"][0]
```

### Several domains

Relation ids are scoped to a domain, so the rows of a suite of several domains have no common catalog. A
`encoding.SuiteEncoder` holds one encoder per domain and encodes a mixed batch as a dictionary of batches, one per
domain, together with each row's index in its batch:

```python
suite = rl.TaskSuite.group([task, mymyr.Task.from_text("tests/data/tasks/blocks__probBLOCKS-8-0.txt")])
suite_encoder = encoding.SuiteEncoder(suite, target_sources=("action",))
initial = [np.pad(np.asarray(t.encode([t.initial_state])), ((0, 0), (0, suite.words - t.words))) for t in suite.tasks]
rows = np.concatenate([initial[1], initial[0], initial[1]])      # a row of the second domain, the first, the second
result = suite_encoder.encode(rows, np.array([1, 0, 1], np.int32), actions=True)

assert set(result) == {0, 1}                                   # the batches by domain; a domain without rows has none
assert list(result.domain) == [1, 0, 1] and list(result.index) == [0, 0, 1]
assert result[1].num_graphs == 2 and result[0].num_graphs == 1
domain, graph = result.graph(2)                                # row 2 is graph 1 of the batch of domain 1
assert (domain, graph) == (1, 1) and result.rows[1][graph] == 2
```

`result.domain[i]` and `result.index[i]` are the domain and the graph of row `i` of the call, `result.rows[d]` the rows
of the call in domain `d`'s batch, and `result.expansions[d]` the expansion that domain's action entities index
(`encoding.action_rows(result[d], result.expansions[d])`). The inputs of `encode` (`goals`, `derived`, `subgoals`,
`history`) are given for all rows of the call and split by domain. A `CpuEnvPool` over a suite takes this path with
`suite_encoder.encode(step.states, step.task_ids, goals=(step.goal_pos, step.goal_neg), actions=True)`.

## Gymnasium

`mymyr.rl.gym` provides `Env`, a `gymnasium.Env` over a task, a `TaskTable` or a `TaskSuite`, and `VectorEnv`, a
`gymnasium.vector.VectorEnv` that steps N environments in one native `CpuEnvPool`. Gymnasium (1.1 or later) is an
optional dependency, installed by the extra `mymyr[gym]`. `import mymyr` does not import it, and importing
`mymyr.rl.gym` without it raises an ImportError that names the extra.

```python needs-gymnasium
from pathlib import Path

import gymnasium as gym
from mymyr.rl import gym as mg

data = Path("tests/data/pddl/gripper")
env = gym.make("mymyr/PDDL-v0", domain=data / "domain.pddl", problems=data, max_steps=50, render_mode="ansi")
env.action_space.seed(0)
obs, info = env.reset(seed=0)
print(env.render())
while True:
    action = env.action_space.sample(mask=info["action_mask"])  # one of the applicable actions
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(sorted(map(str, obs.atoms)), reward, terminated, info["is_goal"])
```

`examples/rl_gym.py` runs a greedy agent on symbolic observations and collects a rollout batch from a `VectorEnv` in
the layout that PPO-style trainers take.

### Observations

`observation=` selects one of three observation modes:

| mode | observation | space |
| --- | --- | --- |
| `"symbolic"` (default) | `SymbolicObservation(atoms, objects, goal, static, numeric)` | `SymbolicSpace` |
| `"graph"` | the relational graph of `mymyr.encoding.FlatRelationEncoder`: a dict of its arrays | `Dict` of `Box` and `Sequence` |
| `"bits"` | the state's fluent atom slots as int8 0/1 | `MultiBinary(n)`; numeric tasks: `Dict(atoms=..., numeric=Box)` |

A symbolic observation holds the state's true fluent atoms as a frozenset of mymyr `GroundAtom`s (`derived=True` adds
the derived atoms). The other fields are optional: the instance's object names (`objects=True`, the default), its
`GroundCondition` goal (`goal=True`, the default), its static atoms, types included (`static=True`), and the numeric
fluent values as a tuple of floats (`numeric=`, by default on for numeric tasks). A field that is turned off is
`None`. Observations are immutable and hashable, and they pickle together with their task. Atoms compare equal within
one `Task` object, so an observation that crossed a process boundary (as in Gymnasium's `AsyncVectorEnv`) belongs to a
restored copy of the task; compare such observations by the atoms' text. `SymbolicSpace` is a custom space because no
standard Gymnasium space holds sets of atoms. `Text` would reduce the atoms to strings, and `Graph` holds binary
edges, not atoms of any arity. Its `contains` checks the fields and that the atoms are fluent atoms. Its `sample`
returns the initial observation of a random instance.

The graph mode builds observations with the native encoder (with `goal_derivations=()`, `include_actions=False`,
`emit_counts=False` and `entity_arrays="auto"` by default; `graph_options=` passes other encoder options), so the
arrays are the ones that [native encoding](encoding.md) documents. The space is a `Dict` of `Box` for the arrays of
fixed length and of `Sequence(..., stack=True)` for the arrays whose length depends on the state, such as
`relation_args`. A Gymnasium `Graph` space has nodes and binary edges, while the encoder's relations have any arity. It
supports the classical tasks of one domain. The bits mode gives a fixed-size vector for standard networks; slot `i` is
atom slot `i` of the current instance, and the instances of a table number their atoms independently.

### Actions

`action=` selects how an action is given:

- `"index"` (the default): `Discrete(max_actions)`, an index into the state's applicable actions in their canonical
  order (by schema, then by binding; `env.unwrapped.applicable_actions()` lists them). `info["action_mask"]` (int8,
  the mask `Discrete.sample(mask=...)` takes) and `action_masks()` (bool, the method maskable policy-gradient
  implementations call) mark the valid indices. `max_actions` defaults to the largest `rl.successor_bounds(table)`:
  the ground actions of an instance's delete relaxation, which bound the applicable actions of every reachable state, so
  every applicable action has an index. A smaller `max_actions` truncates the action list as a padded expansion does.
- `"ground"`: the ground action itself, as pddlgym's literal actions: a mymyr `Action` of the current instance, or
  its text `"(pick ball1 rooma left)"` or `("pick", ["ball1", "rooma", "left"])`. `GroundActionSpace.sample()` draws
  an applicable action, and `contains` accepts the actions of the current instance.

The index mode is the default because standard RL code works with a `Discrete` space directly: it needs no custom
wrapper, the same network output layer serves every state, and an index that the mask rules out is still a
well-defined step. An index past the state's actions, or an action that is not applicable, does not move. It sets
`info["invalid"]`, pays `step_reward` and counts as a step. A state without applicable actions (`num_actions == 0`) is
a dead end, which `dead_end=` and `dead_end_terminal=` configure as in `CpuEnvPool`. A step there does not move
whatever the action, and it does not set `info["invalid"]`.

### Contract, problems and seeding

Transitions are `CpuEnvPool` steps with the same keywords (`max_steps`, `step_reward`, `goal_reward`, `dead_end`,
`dead_end_reward`, `dead_end_terminal`), and the defaults are the same sparse reach-goal rewards. `terminated` holds at
goals and terminal dead ends, and `truncated` holds after `max_steps` steps of an episode (`max_steps=None`, the
default, never truncates). Goals and terminal dead ends are absorbing: a further step does not move, pays nothing and
stays terminated. A reset into a goal (a solved instance) reports `info["is_goal"]` and `info["terminated"]`, and its
first step is absorbing. `info` also carries `task_id`, `num_actions`, `steps` and `action_mask`.

The problems are a task, a table or a suite, or `domain=` and `problems=` (a file, a directory, a glob or a list), which
build `TaskTable.from_pddl(domain, problems)`. Each reset draws the instance with `sampler=`: uniform (`"auto"`,
`"uniform"`), weighted (`rl.InstanceSampler(n, weights=[...])`), or `None`, which keeps the instance. Like pddlgym,
`fix_problem_index(i)` fixes the instance of every later episode (`None` releases it), and
`reset(options={"task_id": i})` picks the instance of one episode.

`reset(seed=s)` seeds `np_random` and the instance draws, which are counter-based as in the pool: episode `k` after the
seed starts in the sampler's draw for `(s, env_id, k)`. Transitions are deterministic, so the same seed and the same
actions give the same trajectory in every process. As in Gymnasium, `action_space.sample()` uses the space's own
generator (`action_space.seed(s)`).

### Vector environment

`VectorEnv(table, num_envs, threads=1, ...)` takes the keywords of `Env` and steps all environments in one native call.
It does not step N Python environments. Its observations are batched: a tuple of N symbolic observations, an int8
`[N, n]` array, or one encoder batch of the N graphs (entity ids global over the batch, the input of a message-passing
model). Index actions are an int array `[N]` with `info["action_mask"]` `[N, max_actions]`. Env `i` is environment
`first_env + i` of the RL contract's streams, so after `reset(seed=s)` its episodes start in the instances of the
episodes of `CpuEnvPool(seed=s)` and of the torch `BatchedEnv(seed=s)`, whose construction starts episode 0; a list
of N seeds seeds the environments one by one.

`autoreset_mode=` follows Gymnasium's definitions. In `"next_step"`, the default, a step that ends an episode returns
its final observation, and the next step resets that environment, ignores its action and returns reward 0. In
`"same_step"`, the native autoreset, an ending step returns the next episode's first observation, with the final ones
in `info["final_obs"]` and `info["final_info"]`. In `"disabled"`, finished environments stay absorbing until
`reset(options={"reset_mask": mask})`. For curricula, `set_next_task_ids(ids)` chooses the instance of each
environment's next autoreset, in whichever step it comes (-1 keeps an environment's draw), and
`reset(options={"task_ids": ids})` chooses the instances of a reset.

### Registration

Importing `mymyr.rl.gym` registers `mymyr/PDDL-v0`: `gymnasium.make("mymyr/PDDL-v0", domain=..., problems=...,
**keywords)` builds an `Env`, and `gymnasium.make_vec` builds a `VectorEnv`. `mg.register_pddl(domain, problems,
**keywords)` registers one domain's problems as `mymyr/<name>-v0`, named after the domain file or its directory.
PDDL environments are defined by files, not by a fixed catalog, so registration takes the paths as keywords instead of
shipping a list of ids.

### Differences to pddlgym

- The Gymnasium API: `reset` returns `(observation, info)`, and `step` returns `terminated` and `truncated`; pddlgym
  never truncates.
- pddlgym pays 1 at the goal and continues stepping after it. Here the rewards are configurable, goals and terminal
  dead ends are absorbing, dead ends can end episodes, and `max_steps` truncates.
- Observations hold mymyr values: `GroundAtom`s, the object names and a `GroundCondition` goal (negative literals and
  numeric constraints included). Static atoms are a separate, optional field rather than part of the state's atoms.
  Derived atoms (axioms) and numeric fluents are supported.
- Actions are indices with a mask by default. The ground mode takes the domain's actions as they are, where pddlgym
  needs action predicates in the domain unless `operators_as_actions=True`. pddlgym's action spaces leave out the
  ground actions that bind one object to several parameters, while mymyr keeps them as PDDL defines them.
- Instance draws are counter-based per environment id and seed, weighted or uniform, and can be overridden for
  curricula. A `VectorEnv` steps N environments natively.
- `render` is text only (`render_mode="ansi"`); pddlgym's domain-specific image renderers have no counterpart.

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

Run `python examples/rl_learning.py` for a NumPy REINFORCE successor scorer. It scores each applicable action by an
action-schema one-hot and by the agent's position in the native relational encoding of the successor state. It trains on three chain instances with 3–5 objects, then evaluates on held-out 8- and
12-object instances against a random policy. The report includes success, solved plan length, instance coverage,
truncations, elapsed time and a counted subset of live array storage (not process peak memory).
The scorer is intentionally small: the position is the object index of the successor's `at` tuple, which equals the
chain position because the example declares its cells in chain order. It is not object-level relational reasoning. It exercises the path from collection through native encoding to a policy update, not general planning ability.
