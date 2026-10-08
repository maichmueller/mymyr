# CUDA

The optional CUDA backend builds with CUDA Toolkit 13.0 and needs an NVIDIA GPU at runtime. Python wheels
contain the CPU backend only. For a source build, set `MYMYR_CUDA=ON` and `CMAKE_CUDA_COMPILER` during the CMake
configure or Python package build.

Device operations include task upload and smoke kernels, BrFS, multi-IW and rollouts; batched grounded h-max, h-add,
h-FF, h² and set-additive; A* and GBFS; state-space generation; and the device path for `mymyr.rl.expand`. Device arrays can exchange
data with JAX and torch through DLPack. `mymyr.cuda.Context` owns a stream-ordered memory pool; set `max_bytes=` to
cap it. Arrays and device tasks keep their context alive.

CUDA upload, expansion, BrFS, multi-IW, rollouts, state spaces, RL environments and A*/GBFS support numeric fluent
values, preconditions, effects, goals and metric costs. Public state rows preserve CPU I32/F64 encoding; internal
numeric rows hold one canonical double per fluent slot. TaskTable/TaskSuite numeric batches use per-instance kernels
with the shared atom/numeric row boundary. Classical tasks retain their atom-only layout and device kernels.

The device heuristic kinds are `"max"`, `"add"`, `"ff"`, `"h2"` and `"set_additive"`; `"blind"` is also accepted by device A* and GBFS.
Relaxation ignores numeric values and constraints as on the CPU. Real heuristic costs must be state-independent,
non-negative integers below 2^31. For tasks with numeric fluent slots, unit relaxation supports fractional/fluent
actual search costs. Cost-only classical searches require integral, state-independent costs. Numeric A*/GBFS use
a host double-priority heap for CPU eager ordering, with device successor, cost and heuristic evaluation.

Multi-IW, batched IW(1) and rollouts accept `(positive_slots, negative_slots)` tuples or `GroundCondition` goals with
fluent and derived literals and numeric constraints. Multi-IW/batched IW lists select one conjunction per start;
rollouts take one conjunction shared by their seeds. Derived goals use the device axiom closure or the CPU axiom
fallback. Numeric goal expressions use at most 64 stack values. Callable goals and any-of alternatives within one
search require CPU search; device A*/GBFS use the task goal. See [Formula values](formulas.md) for goal construction.
The bindings reject goals simplified as impossible because of static literals or unreachable positive atoms.

Numeric searches and environments do not capture CUDA graphs, and numeric environments use the general path rather
than the environment fast path, because these paths synchronize for counts, lazy atoms and overflow checks. Backend
limits remain 64 internal state words and 512 objects; unsupported schema/axiom plans use the documented CPU fallback.

This BrFS example runs the repository's classical blocks fixture on device 0:

```python skip-if-no-cuda
from pathlib import Path

import mymyr
from mymyr import cuda

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
result = cuda.brfs(task, stop_at_goal=True)
print(result.status, result.solved, len(result.plan), result.stats)
```

Set `CUDA_VISIBLE_DEVICES=0` to select the device exposed to the process. `cuda.available()` reports whether the build
can see a CUDA device. `mymyr.cuda` cannot be imported from a CPU-only build.

`"max"`, `"add"` and `"h2"` equal the CPU heuristics, including dead ends. `"ff"` and `"set_additive"` choose
equally cheap supporters by the lowest BFS level through tight h-max achievers, then the smallest operator id.
`Heuristic.reference(state)` implements that rule on the CPU. Set-additive keeps axiom supporters and unions all
their preconditions' achiever sets; each action contributes one `(operator, proposition)` member, weighted by its
cost, and axioms contribute no member. An action supporting several propositions can contribute several members.
The CPU heuristic picks supporters by its queue order, so its value can differ when a tie decides the result.
On the 21 fitting heuristic-suite tasks, four seeded walks of up to 15 steps in each frozen/lazy configuration
produce 2630 unit-cost samples: 968 (36.8%) differ from the CPU set-additive value, with zero mismatches against
`reference()`. The remaining organic-synthesis task exceeds the grounding budget.

h² accepts at most **8191 propositions**, matching the CPU limit. Every resident state group has two triangular
global cost tables plus proposition and operator scratch: `4 * P * (P + 1) + 4 * P + 4 * O` bytes, rounded up to
16 bytes (at least 16), for `P` propositions and `O` operators. The global scratch budget defaults to **512 MiB**;
fewer blocks evaluate the batch when necessary, reusing their tables between rows. This bounds quadratic scratch
independently of batch size and leaves memory for uploaded tasks and search states. If one state cannot fit the
budget, or the proposition count exceeds 8191, construction raises `ValueError` naming the size and limit.
Set `max_scratch_bytes=` on `cuda.Heuristic` to choose a smaller budget; C++ searches use
`DeviceBestFirstOptions::heuristic.max_scratch_bytes`. h² always uses a block per state and global Jacobi sweeps.
States outside the grounding raise an error for h² and set-additive, as on the CPU, because they have no lifted
evaluation.

```python skip-if-no-cuda
from pathlib import Path

import mymyr
from mymyr import cuda

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
ctx = cuda.Context(0, max_bytes=1 << 30)
for kind in ("h2", "set_additive"):
    heuristic = cuda.Heuristic(task, kind, ctx=ctx, max_scratch_bytes=64 << 20)
    values = heuristic.evaluate([task.initial_state])
    assert values[0] == heuristic.reference(task.initial_state)
result = cuda.astar(task, heuristic="h2", ctx=ctx, batch=64)
assert result.solved
state = task.initial_state
for action in result.plan:
    state = task.apply(state, action)
assert task.is_goal(state)
ctx.synchronize()
```
