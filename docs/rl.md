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

## Action masks without an expansion

`rl.applicable_schemas(table, states, task_ids=None, *, pool=None)` returns bool `[N, S]`, where
`S` is the maximum schema count across the table's domains. A schema is true when it has an applicable full binding.
`rl.applicable_objects(table, states, schema, prefix, depth, task_ids=None, *, num_objects=None, pool=None)`
returns bool `[N, O]`: object `o` is true exactly when `(schema, prefix[:depth], o, ...)` has an applicable completion.
`O` defaults to the maximum object count of the table; `num_objects` can increase it with false padding.
`rl.is_applicable(table, states, schema, binding, task_ids=None, *, pool=None)` returns bool `[N]` for full labels.

```python
schemas = rl.applicable_schemas(task, states)
objects = rl.applicable_objects(task, states, schema=0, prefix=None, depth=0)
# Choose objects, extend an int32 prefix [N, depth], then ask for the next parameter.
```

Schemas are domain-local indices, passed as an integer for all rows or int32 `[N]`. Objects are instance-local
indices. Prefixes are int32 `[N, >= depth]` (`None` at depth zero); bindings are int32 `[N, >= arity]`, with
ignored padding past each schema's arity. Invalid schemas (including `-1` padding), objects and bindings produce
false rows. The scalar depth must be below the arity of every valid selected schema, or the call raises
`ValueError`. A binding width below any valid selected schema's arity also raises `ValueError`.
Zero-parameter schemas participate in `applicable_schemas` and
`is_applicable` with binding shape `[N, 0]`.

These functions take a Task, TaskTable or TaskSuite and the same CPU state inputs as `expand`: NumPy, torch or JAX
arrays, a State, or a list of States. Outputs use the state array's framework (States produce NumPy arrays).
Calls release the GIL, accept `pool=ThreadPool(T)` and support concurrent callers. They are eager CPU operations;
CUDA inputs raise `ValueError` with “runs on the CPU; pass host rows”.

The masks equal `schema_masks(rl.expand(...))` and `prefix_masks(rl.expand(...), ...)` with witness pruning off.
Full-label checks equal membership in that expansion's labels, including derived and numeric preconditions and
numeric effect applicability. Matchers compile lazily per schema and parameter; each candidate needs one witness
for its remaining parameters. Numeric effects may require several witnesses before a candidate is accepted. Each
workspace caches its most recently prepared row, reusing its view and axiom closure for consecutive equal rows.
This supports parameter-by-parameter decoding with one-row calls.
The witness search can backtrack exponentially in the number of remaining parameters; it avoids storing or
visiting the full applicable set when a witness can be found early.

### Stepping by label

`mymyr.rl.torch.BatchedEnv(table, N, actions="label")` accepts
`env.step((schema, binding))`: `schema` is int32 `[N]` and `binding` is int32 `[N, L]`, with at least the suite's
largest schema arity. Schema ids belong to each row's domain and object ids to its instance; columns past an
action's arity are ignored. The result's `schema` and `binding` report the applied action, with `-1` padding and
all `-1` for a row that did not move. An inapplicable or malformed label in a nonterminal state with applicable
actions leaves the state unchanged, sets `invalid`, and earns `step_reward`.

Label mode checks and applies the chosen action directly. Its `count` is int32 0/1: whether the observation has
any applicable action, found with witness pruning. This also applies to reset, refresh, and `core.initial_counts`.
Dead ends, absorbing goals, rewards, truncation, goal masks, `next_task_ids`, same-step autoreset, and
`final_states` have the same semantics as `actions="index"` (the default). Labels include every applicable binding,
regardless of `witness_pruning` and `canonical_order`. `step(None)` and `random_actions()` raise `ValueError` in
label mode, since a uniform random action requires enumerating the action space.

Label stepping runs on the host and torch CPU, including `torch.compile`. Requesting `actions="label"` with a
CUDA device raises `ValueError`. The C++ API selects `EnvConfig::actions = rl::ActionSpace::Label` and passes
`rl::Labels{schema, binding, stride}` to `HostEnv::step`; `stride` is the binding row width in int32 elements.
Index mode uses `rl::ActionSpace::Index` and `rl::Actions`.

## Derived words

Derived atoms are never stored in states. `rl.expand(..., derived=True)` also returns, for every successor, the bitset of
its derived atoms (`expansion.derived`, `[M, D]` words in the encoding of the states, bit `d` of the row for derived slot
`d`), from one axiom evaluation of the successor row; the same evaluation yields the goal flag. `D` is
`table.derived_words` (`suite.derived_words`: the widest domain; `task.derived_words` for one task), and 0 for tasks
without derived atoms, whose arrays are zero-width. An instance with fewer derived words has zeros past its own. The
words are exactly what the [relational encoder](encoding.md) evaluates itself, so they can replace its evaluation.
On the device, `expand_into(..., status=, stream=)` (`status` an int32 device array `[1]`) does not wait for the device
and needs `stream=`, the stream that reads its outputs (for instance `torch.cuda.current_stream()`): the expansion runs
on it, so the outputs are ready in that stream's order, and without `stream` the call raises `ValueError`. The total
stays in `offsets[N]` (true counts), and `status[0]` gets bit 0 when the successors exceed the capacity and bit 1 when a
row is invalid (a task id outside the table, bits its instance has no slot for, a successor wider than `succ`; nothing
is written then); successor rows past the total are zero. The result holds only `capacity`, unless the table's
expansion needs the host (instances expanded apart, CPU-fallback schemas or axioms, numeric rows, suites of several
domains, batches beyond one chunk of views), which waits as without `status` and returns the usual keys (`total`
among them): then rows past the total keep what the destination held, and an invalid row raises `ValueError` instead
of setting bit 1.
The optional `row_count=` is a contiguous int32 or uint32 CUDA array `[1]`, read as an unsigned count clamped to
`N`. Only that prefix is expanded or validated; offsets past it repeat the live total. This needs `status=` and an
asynchronous device table: host expansion is rejected when a device live count is supplied.
Capacity and overflow follow `succ`: rows past the capacity are counted, not written; `derived=False` leaves every other
output unchanged and costs nothing. `expand_into` takes a `derived=` destination (it needs `succ=`), `exp.pad(k)` /
`k=` carry `padded.derived` `[N, k, D]` with zero padding, and device inputs fill the words on the device (every device
expansion path, including tables of instances with axioms and suites). The torch ops `expand` and
`expand_flat` return the words as their seventh result with `derived=True`, and `Env.expand(..., derived=True)` /
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

## Successor hashes and no-op actions

On host and CUDA arrays, `rl.expand(..., hashes=True, noops=True)` returns `succ_hash` (uint64; torch uses int64 bit patterns)
and `noop` (bool), one per successor. The hash is the native state hash: trimmed atom words seeded by the numeric
words. For instances with the same numeric width it equals
`mymyr.hash_rows(exp.succ, numeric_words=exp.numeric_words)`. For mixed numeric widths, use each instance's numeric
prefix, omitting its padding, when hashing rows separately. `mymyr.hash_rows(states)` over a list of States hashes
each with its own numeric words. It excludes derived atoms and differs from Python's task-aware `hash(state)`. No-op equality includes the numeric words bit for bit: changing the sign of
zero or a NaN payload is a change. Derived words do not enter equality.

`drop_noops=True` removes no-op successors separately for each parent, keeping all successors when every one is a
no-op. Parents with no successors stay empty. Kept rows retain their order, and `exp.index` gives each row's position
in its parent's full successor order, so that it can be used as an environment action. With `canonical_order=False`
or witness pruning, it refers to that expansion's own order. Filtering also returns `noop` even without `noops=True`.

```python
import numpy as np
import mymyr
from mymyr import rl

task = mymyr.Task.from_pddl("tests/data/pddl/blocks/domain.pddl",
                           "tests/data/pddl/blocks/probBLOCKS-8-0.pddl")
full = rl.expand(task, [task.initial_state], hashes=True, noops=True)
assert np.array_equal(full.succ_hash, mymyr.hash_rows(full.succ, numeric_words=full.numeric_words))
kept = rl.expand(task, [task.initial_state], hashes=True, drop_noops=True)
assert np.array_equal(kept.succ, full.succ[full.offsets[kept.parent] + kept.index])
assert kept.generated == full.total
padded = kept.pad()
assert np.array_equal(padded.succ_hash[padded.mask], kept.succ_hash)
assert np.array_equal(padded.canonical[padded.mask], kept.index)
```

`generated` counts successors before filtering; `total` and `offsets` count kept successors. A fixed `capacity` bounds
`generated`, so filtering cannot hide an overflow (`generated > capacity`). On that overflow with filtering, `total`
and `offsets` are the counts before filtering (`offsets[N] == total == generated`) on every platform, and the rows
and per-successor outputs are unspecified: size `capacity` from `generated` or `offsets[N]`.
`expand_into` accepts `succ_hash=`, `noop=` and `index=` destinations; `drop_noops=True` requires `index=` when it
writes any per-successor array. With `offsets=` alone the capacity is 0, so the offsets are the counts before
filtering. Digests work with `succ=None`, without
storing successor rows; they are exact for every stored row, also one too wide for `words` (`words_needed`), and the
width bounds stored rows only, so a label-only call never reports a width overflow. A single `Task` may use lazy atom slots: expansion
uses its live slot layout, counts new atoms as changes, and grows output rows automatically unless `words=` fixes
the width. Tables require fixed atom slots. Every requested array is filtered alike, including goal
flags and derived words. The padded view adds zero-padded hashes and no-op flags and, when filtering, `canonical`:
each kept successor's position in its parent's full order, -1 padding. Its `index` is always the flat row.
Padding preserves the flat expansion's capacity and atom-word width overflow, even when `K` fits every parent's count.

JAX `Env.expand` and `Env.expand_flat` accept `hashes`, `noops` and `drop_noops` under `jit` and `vmap`; hashes
use uint32 pairs (low word first), the padded result names the positions `canonical` and the flat one `index`, and
`generated` is a scalar pre-drop count. Flat capacity bounds the pre-drop count there too.
Absent digests are `None` and cost no digest work.

On CUDA arrays, `rl.expand` and `rl.expand_into` compute the same hashes, no-op flags and filtered rows bit for bit,
also without a `succ=` destination (the label-only form), and `mymyr.hash_rows(rows, numeric_words=..., stream=...)`
hashes device rows. With `status=` and `stream=` the filtering stays on the stream and captures in CUDA graphs; an
overflowing filtered call reports the offsets before filtering, as on the host, so `offsets[N]` sizes the next
capacity. JAX
`Env.expand` and `Env.expand_flat` take the same flags on a CUDA device. The torch ops `mymyr::expand` and
`mymyr::expand_flat` take `derived`, `rows`, `hashes`, `noops` and `drop_noops` and return eleven tensors `(succ,
parent, schema, binding, goal, offsets, derived, succ_hash, noop, index, status)`, empty where absent; the padded
op's `index` slot holds the canonical positions. `rows=False` selects label-only output, `capacity=` keeps counts and
status on the device, and both trace with `torch.compile(fullgraph=True)`. See [CUDA](cuda.md#driver).

## JAX and PyTorch

The optional `mymyr.rl.jax.Env` is a functional environment whose `reset` and `step` methods work under
`jax.jit`, `jax.vmap` and `jax.lax.scan`. JAX device arrays can use the CUDA path. `mymyr.rl.torch.BatchedEnv` provides
batched tensor environments; `mymyr.rl.torch.PlanningEnv` adapts them to TorchRL and needs `torchrl` and `tensordict`.

On the CPU, `mymyr.rl.jax.Env(..., track_visited=H)` carries per-episode tables in `state.visited` (uint32 `[N, 2H]`,
low/high halves of 64-bit hashes), `state.visited_count` (int32 `[N]`) and `state.visited_status` (int32 `[N]`, sticky
bit 0 once the row's table overflowed; every state array leads with the rows, so states split and join by rows). `H` and
`track_visited=True` follow the capacity rules in [Revisit avoidance](#revisit-avoidance). `info["revisited"]` tests the
reached state before insertion and autoreset. Reset clears selected histories and inserts initial states;
`env.clear_visited(state, mask=None)` returns a state with selected histories cleared and current states inserted,
retaining overflow status. `refresh(state)` preserves histories. Disabled tracking uses `[N, 0]` table/status arrays and
reports false revisit flags. CUDA tracking raises a clear construction error because device visit tables are
unavailable. `mymyr.rl.jax.visited_mask` (also dispatched by `rl.visited_mask`) accepts these uint32 halves, or (u)int64
hashes with JAX x64 enabled (64-bit inputs raise `TypeError` without it; `x.view(np.uint32)` gives the halves), and
equals the NumPy membership operation under jit without host reads. The successor hashes in that layout are
`env.expand_flat(state.states, capacity, hashes=True).succ_hash`, and `flat.parent` their parents.

```python needs-jax
import jax
import jax.numpy as jnp
from mymyr.rl.jax import Env

jax_visit_data = Path("tests/data/pddl/blocks")
jax_visit_task = mymyr.Task.from_pddl(jax_visit_data / "domain.pddl", jax_visit_data / "probBLOCKS-8-0.pddl")
jax_visits = Env(jax_visit_task, device="cpu", max_steps=20, track_visited=True)
jax_state, _ = jax_visits.reset(0, 4)
jax_next, _, _, _, _, jax_info = jax.jit(jax_visits.step)(0, jax_state, jnp.full(4, -1, jnp.int32))
assert jax_info["revisited"].all()
jax_state = jax.jit(jax_visits.clear_visited)(jax_next)
assert (jax_state.visited_count == 1).all()
jax_flat = jax_visits.expand_flat(jax_state.states, 64, hashes=True)
jax_seen = mymyr.rl.jax.visited_mask(jax_state.visited, jax_state.visited_count, jax_flat.succ_hash, jax_flat.parent)
assert jax_seen.shape == jax_flat.parent.shape
```

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
next-observation tensors in `StepResult`, along with `final_states`, the reached states before autoreset. With an
observation encoder (see [Encoded observations](#encoded-observations)), `Observation.encoding` and
`StepResult.obs` / `final_obs` hold their encodings, in new arrays of every call; without one these fields are None,
so iterate the fields by name rather than treating either tuple as all tensors. These snapshots survive subsequent
steps and resets. The public environment tensors are live buffers for native calls and
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

## Revisit avoidance

`CpuEnvPool(..., track_visited=H)` and torch CPU `BatchedEnv(..., track_visited=H)` keep one fixed-capacity hash table
per environment. `H` is a power of two from 2 to 2²⁰. With `track_visited=True`, set `max_steps`: the capacity is the
next power of two at least `2 * (max_steps + 1)`, up to 2²⁰. Tracking is disabled by default.
Disabled tracking reports all-false `revisited` flags, including the `[K, N]` tensor in captured torch rollouts.
CUDA environments reject enabled tracking because device visit tables are unavailable.

Reset inserts the initial state. Each step reports `revisited`, whether the reached state was present before that
step, then inserts it. This includes invalid actions that stay in place. Absorbing rows without autoreset report
false and insert nothing. An autoreset first records the reached state's membership, then clears its table and inserts
the new initial state. Tables are independent across environment rows, including rows from different domains.

The keys are the state content hashes, numeric words included: the successor hashes `succ_hash` of `rl.expand(...,
hashes=True)`, equal to `mymyr.hash_rows(rows, numeric_words=k)` with the rows' numeric width `k` (`hash_rows` without
`numeric_words` keys classical rows only). Equal 64-bit hashes count as one entry; zero shares the reserved key
`0x9E3779B97F4A7C15`. An insertion above ¾ occupancy is skipped, saturates the row's count to `H`, and sets bit 0 of the
sticky overflow status. Stored entries remain searchable: overflow can miss a revisit but cannot invent one, apart from
hash collisions. The torch tensors `env.visited` (int64 `[N, H]`), `env.visited_count` (int32 `[N]`) and
`env.visited_status` (int32 `[1]`) expose the live storage; treat the table and counts as read-only. The pool exposes
the same status as an integer `pool.visited_status`. Resets and clears leave status set; torch callers can acknowledge
it with `env.visited_status.zero_()`.

`refresh()` keeps visit history. After writing state rows directly, use `env.clear_visited(mask=None)` to clear all
or selected rows and insert their current states. The pool's `clear_visited(env_ids=None)` accepts selected idle
environments, like `reset`. Table storage uses `8 * N * H` bytes plus counts and status (8 MiB for N = H = 1024).

`rl.visited_mask(table, count, succ_hash, parent, rows=None)` returns a boolean per successor. `parent` is int32;
`-1` denotes padding. With a subset expansion, `rows` maps its parent ids to environment rows (mapped `-1` is also
padding). NumPy and torch CPU inputs are supported, including `mymyr::visited_mask` under `torch.compile`.

For a soft policy, suppress visited successors' scores. The example takes the successor hashes of the expansion:

```python needs-torch
import torch
from mymyr.rl.torch import BatchedEnv

visit_task = mymyr.Task.from_text("tests/data/tasks/gripper__prob05.txt")
visit_env = BatchedEnv(visit_task, 4, max_steps=20, track_visited=True, autoreset=False)
visit_exp = rl.expand(visit_task, visit_env.states, framework="torch", hashes=True)
action_visited = rl.visited_mask(visit_env.visited, visit_env.visited_count, visit_exp.succ_hash, visit_exp.parent)
q = torch.zeros(len(action_visited))
soft_scores = q.masked_fill(action_visited, -1e6)
assert soft_scores.shape == action_visited.shape
```

For hard avoidance, take the highest-ranked unvisited action in each parent's canonical successor order. A row with
none is trapped; the caller decides how to end that rollout:

```python needs-torch
chosen = torch.full((4,), -1, dtype=torch.int64)
trapped = torch.zeros(4, dtype=torch.bool)
for i in range(4):
    lo, hi = int(visit_exp.offsets[i]), int(visit_exp.offsets[i + 1])
    available = (~action_visited[lo:hi]).nonzero().flatten()
    trapped[i] = len(available) == 0
    if not trapped[i]:
        chosen[i] = available[q[lo:hi][available].argmax()]
assert torch.all((chosen >= 0) | trapped)
```

Cycle detection can use the reached-state flag directly, without expanding successors. Compute any successor masks
for the next policy observation after the step: its table already includes the reached state, or its new initial
state following autoreset.

```python needs-torch
visit_result = visit_env.step()  # random policy
cycle = visit_result.revisited & ~visit_result.goal
assert cycle.shape == (4,)
visit_env.clear_visited()
assert torch.all(visit_env.visited_count == 1)
visit_env.close()
```

## Lifted hindsight relabelling

Lifted hindsight relabelling uses parts of an instance's goal, with the objects replaced by variables, as goals for
the states a trajectory reaches. `rl.lifted_goal_templates(source)` makes these goal templates for every instance of a
Task, TaskTable or TaskSuite. An instance's templates come from its goal literals, `task.goal_condition.literals`
(both polarities and all predicate kinds; numeric constraints are skipped):

1. Two literals are joined when their atoms share an object. The components of this graph are ordered by their
   smallest literal index.
2. Each component lists its first `max_subsets_per_size` connected subsets of each size up to `max_size`, in
   lexicographic order of their literal indices.
3. For c = 1 up to `max_size` components, up to `max_component_samples` uniform draws of c components are taken, and
   each draw contributes the first `max_products` combinations of one subset per drawn component. The draws use a
   counter-based random generator keyed by `seed` and the instance's index in the source.
4. Subgoals with more than `max_size` literals are dropped. The first `max_subgoals_per_size` of each size are kept,
   largest first.
5. Each subgoal is lifted. Every object, constants included, becomes a variable `?x<i>`, numbered by first appearance.

With `distinct=True` (the default) distinct variables must bind distinct objects, and the condition carries one
disequality per pair of variables. `distinct=False` lets variables share an object, and `distinct="equality"` behaves
like `True` when the domain declares `:equality` and like `False` otherwise. The templates of an instance depend only
on its task, its index in the source and the options, not on the other instances.

```python
from pathlib import Path
import pickle

import mymyr
from mymyr import rl

data = Path("tests/data/pddl/gripper")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p02.pddl")
templates = rl.lifted_goal_templates(task, seed=0)
for literals, condition in zip(templates.goal_literals(0), templates.conditions(0)):
    print(literals, condition)
# (0, 1) (?x0 ?x1 ?x2) (and (at ?x0 ?x1) (at ?x2 ?x1) (!= ?x0 ?x1) (!= ?x0 ?x2) (!= ?x1 ?x2))
# (0,) (?x0 ?x1) (and (at ?x0 ?x1) (!= ?x0 ?x1))
# (1,) (?x0 ?x1) (and (at ?x0 ?x1) (!= ?x0 ?x1))

largest = templates.conditions(0)[0]
assert largest.binding_order(task) == (0, 1, 2)
binding = next(iter(task.bindings(largest, task.initial_state, limit=1)))
assert [obj.name for obj in binding] == ["ball1", "rooma", "ball2"]
assert str(rl.lifted_goal_templates(task, distinct=False).conditions(0)[0]) == "(?x0 ?x1 ?x2) (and (at ?x0 ?x1) (at ?x2 ?x1))"

arrays = templates.arrays()
assert arrays["instance_offsets"].tolist() == [0, 3]
assert arrays["template_size"].tolist() == [2, 1, 1]
assert arrays["order"].tolist() == [[0, 1, 2], [0, 1, -1], [0, 1, -1]]
assert arrays["terms"].tolist() == [[0, 1], [2, 1], [0, 1], [0, 1]]

fixed = rl.GoalTemplates.from_conditions(task, [[largest]])
assert fixed.options is None and fixed.goal_literals(0) == [()]
restored = pickle.loads(pickle.dumps(templates))
assert [str(c) for c in restored.conditions(0)] == [str(c) for c in templates.conditions(0)]
```

`templates.conditions(i)` returns instance i's templates as `ConjunctiveCondition`s, and `templates.goal_literals(i)`
returns the goal literal indices each template came from. `condition.binding_order(task)` lists the free variables in
the order `task.bindings` binds them: the bindings come in lexicographic order over those variables by object index,
so a state's first binding is the smallest in that order. `templates.arrays(framework=None)` is the packed form, with
the templates of instance i at `[instance_offsets[i], instance_offsets[i + 1])`. Each template has a `template_size`,
`num_vars`, `distinct` flag, `literal_offsets` and an `order` row (its binding order, padded with -1). Each literal has
a `predicate` (the domain table's predicate id), `polarity`, `kind` (0 static, 1 fluent, 2 derived), `arity`, a `terms`
row (variable indices, padded with -2**31) and `goal_literal`.

`GoalTemplates.from_conditions(source, conditions)` packs your own templates, one list per instance in match order.
Each condition needs literals, untyped variables and no numeric constraints. It carries either no equalities or one
disequality per pair of variables, as `GroundCondition.lift(True)` makes them. Terms may be objects. Templates pickle
with their source, options and arrays; nothing is generated again. Like tables, templates refuse tasks with lazy
atom slots (load with `atoms="auto"`, `"reachable"` or `"dense"`).

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
states, which `rl.expand` makes, is what the policy's actions index. Goals of the caller's own, such as a manager's
subgoals decoded as literals, become such masks with `encoding.goal_masks(table, literals, task_ids)`
([Goals](encoding.md#goals)). One call encodes all of it:

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
  (`encoding.action_entities(batch)` lists their ids);
- `encoding.action_entities_of(batch, indices, graphs=None)` is the inverse: the entity of each canonical successor
  index (`[N]` or `[N, K]`, row `i` of graph `graphs[i]`, by default graph `i`), -1 where the graph has none. On a
  device batch it is a torch computation on the device that waits for nothing.

Encoded from an expansion that dropped its no-op actions (`rl.expand(..., drop_noops=True)`), the batch has no entity
for them, and every other entity keeps its position in the full canonical order as `entity_ref` (the expansion's
`index`): `action_indices` gives the same environment actions as with the full expansion, and `action_rows` finds the
entity's row by its `index`.

The targets of behaviour cloning are positions of this kind: `datasets.plan_rows` replays a plan into its states and,
per step, the canonical successor index of the plan's action, the entity whose `action_indices` equals it
([Plans for behaviour cloning](datasets.md#plans-for-behaviour-cloning)).

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
domain, together with each row's index in its batch. CUDA rows and global task ids are grouped on the device and
produce device batches. Exact calls over several domains download only their row counts; one domain uses the known
input row count. Each domain then sizes its outputs as a flat encode does. Capacities for every domain (`graphs >= N`) avoid all count downloads and include domains with zero
rows; `device_graphs` counts each batch's live prefix and `rows[d]` is padded with -1. This example uses host rows:

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

## Encoded observations

Both environments can hand the policy the [relational encoding](encoding.md) of what it observes instead of the raw
state words: pass `observation=` a `mymyr.encoding.FlatRelationEncoder` of the environments' table. Every step then
encodes the rows after the step and the autoreset, on the environments' device: the device encoder on CUDA, the host
encoder on the CPU, with the same arrays bit for bit as `encoder.encode` of those rows on the host.

- **What is encoded.** The next observation's states, instances and, with `goals=True`, the per-environment goal
  masks as the goal relations (`goals=(goal_pos, goal_neg)` of `encode`). The reached states before the autoreset are
  encoded too, with the instances and goal masks they were reached in: torch `StepResult.final_obs` (with
  `final=True`) and JAX `info["final_obs"]` (with `final_state=True`), for bootstrapping truncated episodes. Without
  autoreset the final observation is the observation itself.
- **Capacities.** `obs_capacities=dict(entities=, args=)` (plus `targets=` for target sources; `graphs=` defaults to
  the number of environments) gives every array a fixed size, so nothing waits for the device and the encoding is
  captured with the step (`CapturedRollout`, `torch.cuda.graph`, `torch.compile`, `jax.jit`, `lax.scan`). The true
  sizes are in the batch's `device_sizes` (`sizes` reads them on the host); a batch that does not fit is all zero with
  its overflow flag set. Without capacities every encoding is exact: on a device it waits for the batch's sizes once,
  and it cannot be captured or jitted. `obs.nbytes` and `obs.nbytes_by_array` report the allocated output payload,
  including unused capacity and device counters, without waiting. The same API covers `SuiteObservation`, including
  its domain/index mappings. See
  [observation bytes](encoding.md#observation-bytes) and [device status](encoding.md#capacities-status-and-synchronization).
- **Actions** are off by default: an observation holds no action tuples even when the schema has action relations.
  `BatchedEnv(..., obs_actions=True)` adds them from the expansion of the observed states: on the CPU, and on a device
  with or without `obs_capacities`. Exact device expansion waits for its total, and the encoding waits for its
  sizes. Capacity observations allocate bounded action buffers and keep the total and status on the device; they
  wait for nothing and capture with the step. The buffers have the rows that the entity capacity's action entities
  can stand for (`entities` times the largest number of schemas that share an action relation): they bound the kept
  rows. `obs_capacities["successors"]` bounds the successors before `obs_drop_noops` drops the no-op rows (default:
  the same rows), so with the drop set it to the most successors an observed state batch has; it is at most
  2**31 - 1, and with the drop a device holds that many working rows of about `38 + 8 * label_width` bytes each.
  More kept rows or more successors is an overflow, and the CPU environment reports it as a device does, for the same
  capacities: the capacity overflow flag, every output array and the batch's sizes zero (the encoding does not run).
  This requires a table whose expansion runs asynchronously on the device; tables needing host expansion are rejected
  before expansion. Insufficient encoding capacity sets the batch's capacity overflow flag and zeroes every output
  array, with the true sizes. Successful capacity calls zero unused output tails. These buffers store labels only;
  successor state rows are not allocated.
  A captured graph owns an exclusive expansion workspace per table; eager calls and other captures use their own.
  The workspace stays alive until the graph is destroyed and its last replay has completed, also for a manual
  `torch.cuda.CUDAGraph`. Callers may drop every result without releasing the workspace. Two instantiations of one
  capture share its workspace, so do not launch them concurrently.
  Warm on the capture stream with the same shapes and options. All expansions over one table within a graph must
  keep them fixed; unwarmed calls and changes to these shapes or options are refused during capture.
  The [encoding buffer requirements](encoding.md#streams-and-cuda-graphs) also apply to expansion inputs and
  caller-provided destinations: keep them alive at their captured addresses until the final replay completes, and
  synchronize the replay stream before freeing or reusing them.
  External CUDA streams must remain valid while arrays produced on them are retained. The JAX environments do not
  encode actions.
- **Action rows and no-op actions.** With `obs_actions=True` the observation's expansion rows come along as
  `Observation.actions` and `StepResult.actions` (`mymyr.rl.torch.ActionRows`; over a suite of several domains a tuple,
  one per domain batch; `Rollout.obs_actions` for a captured rollout): `parent`, `schema`, `binding`, `offsets`, the
  successor hash `succ_hash` (int64 bit patterns of `State.hash()`) and `noop` of every row, `index` and the device
  `status`. `encoding.action_rows(obs, actions, entities)` gives the row of a chosen entity, so a policy reads the
  label (Label mode) or the hash of the state an action reaches without a second expansion. `obs_drop_noops=True`
  leaves out the actions whose successor equals the state, unless every successor of the state is one: their entities
  are absent, `actions.index` is each kept row's canonical successor index, and `encoding.action_indices` of an entity
  is still the action that `step` takes. It costs one compaction pass of the label-only expansion and captures like
  it; under `torch.compile` the observation is traced as `mymyr::bounded_actions` followed by `mymyr::encode_flat`
  with those rows, neither tagged `cudagraph_unsafe`, so `mode="reduce-overhead"` keeps the step in CUDA graphs.
  With `obs_capacities` the row arrays have the capacity's rows on the CPU and on a device alike, every array zero
  past `offsets[-1]` (so that `parent` there still names a state); `offsets` has N + 1 entries, over a suite the
  total repeated past a domain batch's graphs. Without capacities the row arrays have exactly the rows. After an
  overflow the rows and `offsets` are unspecified.
- **Suites.** The domains of a `TaskSuite` have relation catalogs of their own, so over a suite of several domains
  `observation=` takes one encoder per domain and an observation is a `SuiteObservation`: `batches[d]` is the batch of
  domain `d`'s rows in row order (with the domain's instance numbers as task ids), `domain[i]` the domain of row `i`
  and `index[i]` its graph in that batch. With capacities a domain's batch has an entry for every environment in its
  per-graph arrays and counts its rows on the device (`device_graphs`; `num_graphs` reads it).
  Multi-domain torch observations capture with `BatchedEnv.capture` or `torch.cuda.graph`; their domain dispatch
  is not traced by `torch.compile`. Grouping, status and the live-row bound on action expansion follow the
  [device-suite contract](encoding.md#several-domains-and-resident-datasets).

A torch environment on the CPU, whose observations are the host encoder's batches:

```python needs-torch
from pathlib import Path

import mymyr
from mymyr import encoding, rl
from mymyr.rl.torch import BatchedEnv

data = Path("tests/data/pddl/blocks")
tasks = [mymyr.Task.from_pddl(data / "domain.pddl", data / p, atoms="dense") for p in ("probBLOCKS-8-0.pddl", "smedium.pddl")]
table = rl.TaskTable(tasks)
encoder = encoding.FlatRelationEncoder(table)
env = BatchedEnv(table, 8, seed=0, max_steps=6, observation=encoder, obs_capacities=dict(entities=4096, args=16384))
step = env.step()
print(step.obs.num_graphs, step.obs.sizes["tuples"], step.final_obs.sizes["overflow"])
```

On CUDA the same environment encodes on the device, and a captured rollout returns the observation of each of its
steps; `encoding.collate` concatenates them into one batch of all its rows:

```python skip-if-no-cuda needs-torch
import torch

env = BatchedEnv(table, 1024, device=torch.device("cuda", 0), seed=0, max_steps=6, observation=encoder,
                 obs_capacities=dict(entities=1 << 16, args=1 << 18))
rollout = env.capture(4)
result = rollout.replay()
batch = encoding.collate(result.obs)
print(len(result.obs), batch.num_graphs, result.obs[0].device)
assert result.obs[0].nbytes == sum(result.obs[0].nbytes_by_array.values())
```

Without no-op actions, a policy that picks an action entity steps to the successor whose hash the observation names:

```python needs-torch
import numpy as np
import torch

floortile = Path("tests/data/ipc/floortile-ipc/train")
tiles = mymyr.Task.from_pddl(floortile / "domain.pddl", floortile / "p01.pddl", atoms="dense")
tile_encoder = encoding.FlatRelationEncoder(tiles, target_sources=("action",))
tile_env = BatchedEnv(tiles, 4, seed=1, observation=tile_encoder, obs_actions=True, obs_drop_noops=True)
observation = tile_env.observation()
batch, rows = observation.encoding, observation.actions
entities = encoding.action_entities(batch)
graphs = np.searchsorted(np.asarray(batch.entity_offsets), entities, side="right") - 1
chosen = entities[np.searchsorted(graphs, np.arange(4))]       # the first action entity of each graph
assert not rows.noop[encoding.action_rows(batch, rows, chosen)].any()
reached = rows.succ_hash[encoding.action_rows(batch, rows, chosen)]
step = tile_env.step(torch.as_tensor(encoding.action_indices(batch, chosen)))
assert torch.equal(mymyr.hash_rows(step.final_states), reached)
```

Action observations use the same capacity form, including the reached states before an autoreset:

```python skip-if-no-cuda needs-torch
encoder_actions = encoding.FlatRelationEncoder(table, target_sources=("action",))
action_env = BatchedEnv(table, 8, device=torch.device("cuda", 0), seed=3, max_steps=4,
                        observation=encoder_actions, obs_actions=True,
                        obs_capacities=dict(entities=4096, args=16384, targets=4096))
action_rollout = action_env.capture(3, final=True)
action_result = action_rollout.replay()
torch.cuda.synchronize()
assert len(action_result.obs) == len(action_result.final_obs) == 3
assert all(not batch.sizes["overflow"] for batch in (*action_result.obs, *action_result.final_obs))
assert all(batch.nbytes == sum(batch.nbytes_by_array.values()) for batch in action_result.obs)
```

The JAX environment carries the encoding in its observations, under `jax.jit` and `lax.scan`:

```python needs-jax
from pathlib import Path

import jax
import mymyr
from mymyr import encoding, rl
from mymyr.rl.jax import Env

data = Path("tests/data/pddl/blocks")
tasks = [mymyr.Task.from_pddl(data / "domain.pddl", data / p, atoms="dense") for p in ("probBLOCKS-8-0.pddl", "smedium.pddl")]
table = rl.TaskTable(tasks)
encoder = encoding.FlatRelationEncoder(table)
env = Env(table, device="cpu", max_steps=6, observation=encoder, obs_capacities=dict(entities=4096, args=16384))
state, observation = env.reset(jax.random.key(0), 8)
state, observation, reward, terminated, truncated, info = jax.jit(env.step)(jax.random.key(1), state)
print(observation.encoding.num_graphs, int(observation.encoding.overflow))
assert observation.encoding.nbytes == sum(observation.encoding.nbytes_by_array.values())
```

## A learning smoke example

Run `python examples/rl_learning.py` for a NumPy REINFORCE successor scorer. It scores each applicable action by an
action-schema one-hot and by the agent's position in the native relational encoding of the successor state. It trains on three chain instances with 3–5 objects, then evaluates on held-out 8- and
12-object instances against a random policy. The report includes success, solved plan length, instance coverage,
truncations, elapsed time and a counted subset of live array storage (not process peak memory).
The scorer is intentionally small: the position is the object index of the successor's `at` tuple, which equals the
chain position because the example declares its cells in chain order. It is not object-level relational reasoning. It exercises the path from collection through native encoding to a policy update, not general planning ability.

`torch.BatchedEnv.encode_transitions(encoder, *, actions=None, capacities=None)` encodes its current rows and
successors with a `FlatTransitionEncoder` or `FlatTransitionEffectsEncoder` on the environment's device. Device
expansion labels and depth-one horizons are encoded into torch arrays on the current stream; with `capacities` the call
does not wait for the device (see [the encoding](encoding.md)).

An environment's device observation rows can be passed directly to `encoder.encode(obs.states, obs.task_ids)`.
The table may mix row widths; each instance reads only its own words. `statics="shared"` keeps one static block per
instance in the resulting CUDA batch. The same observation tensors can back `datasets.StateRows` for encoding on load.

States and rows from another parse must be translated before use. `source.translate(states_or_rows, target)` matches
predicate, function and object names and returns target-owned States or target-layout rows. For a table, use
`source_table.translate(rows, task_ids, target_table)`; instance i maps into instance i. `mymyr.SlotMap` reuses the
mapping across batches. `missing='drop'` also returns a per-row lossless mask; the default raises on a missing true
atom or numeric value. CUDA arrays support `status=` and graph capture; see
[Moving states between parses](parsing-and-tasks.md#state-encodings) for the status layout and buffer lifetimes.
