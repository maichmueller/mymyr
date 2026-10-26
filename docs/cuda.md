# CUDA

The optional CUDA backend needs an NVIDIA GPU at runtime. Python wheels contain the CPU backend only. For a source
build, set `MYMYR_CUDA=ON` and `CMAKE_CUDA_COMPILER` during the CMake configure or Python package build. The device
searches and generators take the options and return the result fields of the CPU ones
([API conventions](conventions.md)); a device result adds `counters`, a dict of device counts and times.

## Building

The device code is C++23, compiled by nvcc with the project's GCC 16 as its host compiler (the host code stays C++26).
The configure step refuses an older setup with a message:

- CUDA Toolkit 13.4 or newer (nvcc 13.4 is the first to accept GCC 16 and `-std=c++23` in both compilation passes),
  with the CUDA runtime and CCCL headers;
- CMake 4.4 or newer (the first to map CUDA C++23 to nvcc's flag; `pip install "cmake>=4.4"` provides it).

nvcc's host compiler is the C++ compiler when that is GCC; `CMAKE_CUDA_HOST_COMPILER` or the `CUDAHOSTCXX` environment
variable choose another. Its standard library must be the one the C++ side uses, because both link into one binary.
`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (SASS for the visible GPU). For a build that runs on several GPU
generations, list real architectures, and a virtual one only for the newest, e.g.
`-DCMAKE_CUDA_ARCHITECTURES="80-real;86-real;89-real;90-real;100-real;120"`: `120` adds SASS and PTX for sm_120.

The test that checks the device Philox generator against cuRAND's is built when the toolkit provides cuRAND's headers
(`libcurand-dev-13-4` on Debian and Ubuntu); without them it is not built, and the library does not need them.

## Driver

Device code built with a CUDA 13.x toolkit runs on NVIDIA drivers 580 or newer through CUDA minor-version
compatibility, as SASS for the GPU's own architecture. PTX is compiled by the driver at load time and needs a driver
that knows the toolkit's PTX version (newer than the minimum for SASS), so build SASS for every GPU generation you
run on and keep PTX only as the forward-compatible fallback for GPUs newer than the build. A GPU that has neither
matching SASS nor loadable PTX fails at the first device call.

Device operations include task upload and smoke kernels, BrFS, multi-IW and rollouts; batched grounded h-max, h-add,
h-FF, h² and set-additive; A* and GBFS; state-space generation; and the device path for `mymyr.rl.expand`. Device arrays can exchange
data with JAX and torch through DLPack. `mymyr.cuda.Context` owns a stream-ordered memory pool; set `max_bytes=` to
cap it. Arrays and device tasks keep their context alive.

CUDA upload, expansion, BrFS, multi-IW, rollouts, state spaces, RL environments and A*/GBFS support numeric fluent
values, preconditions, effects, goals and metric costs. Public state rows preserve CPU I32/F64 encoding; internal
numeric rows hold one canonical double per fluent slot. TaskTable/TaskSuite numeric batches use per-instance kernels
with the shared atom/numeric row boundary. Classical tasks retain their atom-only layout and device kernels.

The device heuristic kinds are `"max"`, `"add"`, `"ff"`, `"h2"` and `"set_additive"`; `"blind"` is also accepted by device A* and GBFS.
Relaxation ignores numeric values and constraints as on the CPU, and the heuristic costs are the CPU heuristics'
(`costs="auto"` by default: the task's objective); real costs must stay below 2^31 units of their cost scale. Numeric
searches support fractional and state-dependent action costs. Cost-only classical searches require integral,
state-independent costs. Numeric A*/GBFS use
a host double-priority heap for CPU eager ordering, with device successor, cost and heuristic evaluation. Weighted A*
(`BestFirstOptions::weight` other than 1), a cost bound, enforced hill-climbing and the anytime search run on the CPU
only: the device searches refuse a weight or a bound with `std::invalid_argument`.

Multi-IW, batched IW(1) and rollouts accept `(positive_slots, negative_slots)` tuples or `GroundCondition` goals with
fluent and derived literals and numeric constraints. Multi-IW/batched IW lists select one conjunction per start;
rollouts take one conjunction shared by their seeds. Derived goals use the device axiom closure or the CPU axiom
fallback. Numeric goal expressions use at most 64 stack values. Callable goals and any-of alternatives within one
search require CPU search; device A*/GBFS use the task goal. See [Formula values](formulas.md) for goal construction.
The bindings reject goals simplified as impossible because of static literals or unreachable positive atoms.

Numeric searches and environments do not capture CUDA graphs, and numeric environments use the general path rather
than the environment fast path, because these paths synchronize for counts, lazy atoms and overflow checks.
Expansion and environments run states of any width on the device, and tasks of up to 2048 objects (object bitsets of
up to 32 words). Classical BrFS, IW, rollouts, A*/GBFS and state spaces accept rows of any width that fits memory.
Arenas keep power-of-two widths through 64 words and round wider rows to 64-word chunks; successors are built
one chunk at a time. Numeric kernels hold whole rows and take states of at most 64 words; a numeric search raises
`ValueError` when its states reach more (also when a lazy task's rows grow there during the search).
When a state's successors exceed what the context can still allocate, the device BrFS deduplicates and stores them
a window at a time (the `windows` counter), with the ids, counts and plans of the CPU BrFS; device IW, rollouts and
A*/GBFS expand a state's successors at once and raise `ValueError` naming the successor count, the bytes they need and
what the context has left (raise the Context's `max_bytes`, or use the CPU search). The searches size their buffers by
what the context can still allocate: the BrFS starts its arenas and state table within it, reserves the arenas for a
windowed state's new states once and sizes each window by every buffer it allocates, and a chunk whose buffers do not
fit runs with fewer parents. When the search's own buffers leave too little, or an allocation fails all the same, the
`ValueError` names what holds the context's memory.
IW(1) uses one novelty bit per slot per search. Wide IW(2) tables pack the symmetric slot pairs and diagonal
into roughly `slots * (slots + 1) / 16` bytes per search. Groups shrink to `table_bytes` (an option of `multi_iw`,
`batched_iw1` and `rollouts`); a pass whose table cannot fit even one search ends the searches that reach it with
status `FAILED`, and the batch's `message` gives its required byte size and the budget; the searches solved at a lower
width keep their results. Wide exact novelty uses
full 32-bit atom slots and stable radix sorting, preserving CPU candidate order. A `DeviceExpander` expands an instance the
kernels cannot run (more than 2048 objects, numeric rows wider than 64 words) on the host: its rows are copied to the
host, expanded on the host's threads and the successors copied back, which synchronizes. In a table or suite only
those instances take the host and the others stay on the device, so the environments' general path serves every
table and suite; the fast path and CUDA graph capture need every instance on the device. Unsupported schema/axiom
plans use the documented CPU fallback.
The derived predicates `axiom_<k>` compiled from quantified and disjunctive conditions (see
[Parsing and tasks](parsing-and-tasks.md)) are axioms like any other: the device evaluates their rules, unless a rule
has a numeric comparison or more parameters than the device kernels take; then the task's axioms run on the CPU
(`counters["device_axioms"] == 0` of a device BrFS result) and the results are the same.
The device evaluates numeric conditions and effects only in tasks with numeric state values: a task whose numeric
conditions or effects read only static functions and functions that never get a value (no initial value, no `assign`
effect) is refused with `ValueError`; run it on the CPU.

This BrFS example runs the repository's classical blocks fixture on device 0:

```python skip-if-no-cuda
from pathlib import Path

import mymyr
from mymyr import cuda

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
result = cuda.brfs(task)
print(result.status, result.solved, len(result.plan), result.cost, result.stats)
print(result.counters)  # per-phase device times and counters
```

Set `CUDA_VISIBLE_DEVICES=0` to select the device exposed to the process. `cuda.available()` reports whether the build
can see a CUDA device. `mymyr.cuda` cannot be imported from a CPU-only build.

`"max"`, `"add"` and `"h2"` equal the CPU heuristics, including dead ends. `"ff"` and `"set_additive"` choose
equally cheap supporters by the lowest BFS level through tight h-max achievers, then the smallest operator id.
`Heuristic.reference(state)` implements that rule on the CPU. Set-additive keeps axiom supporters and unions all
their preconditions' achiever sets; each action contributes one `(operator, proposition)` member, weighted by its
cost, and axioms contribute no member. An action supporting several propositions can contribute several members.
The CPU heuristic picks supporters by its queue order, so its value can differ when a tie decides the result.
On the 21 fitting heuristic-suite tasks, four seeded walks of up to 15 steps in each fixed-slot and lazy configuration
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

## Encoded observations and resident rows

[Device encoding](encoding.md#encoding-on-the-device) reads CUDA state rows directly and returns torch arrays:
relations, Simple graphs, numeric values, actions, horizons and shared static blocks, subject to the
[support table](encoding.md#what-the-device-encodes). Exact calls download output counts. Capacity calls retain
counts and status on the device and support CUDA graph capture after warm-up. `batch.nbytes` and
`batch.nbytes_by_array` report capacity payload bytes from host metadata without synchronizing.

Choose the path by where rows arrive:

- Already batched CUDA rows: `encoder.encode(rows, task_ids, capacities=...)` or the
  [torch custom op](encoding.md#the-torch-op) inside `torch.compile`.
- Rows accumulated over time: a [device row ring](encoding.md#device-row-streams-and-collation); prepare a
  `DeviceCollator` for repeated or captured collation.
- Several domains or resident training data: [`SuiteEncoder` and `EncodingCollate`](datasets.md#training-data-encoded-on-load)
  group global task ids on the device, returning a batch and local ids per domain. Capacity calls keep the grouping
  resident, including empty domains.
- Policy observations: [`BatchedEnv`](rl.md#encoded-observations) encodes states after autoreset and, on request,
  final states before reset. Capacity action observations can be captured with the step when expansion is asynchronous.
- Jitted JAX code: the [XLA FFI encoder](encoding.md#encoding-inside-jax) uses XLA's stream and fixed capacity shapes;
  its supported forms are narrower than the torch path.

Torch operations use the current stream. Order cross-stream producers and consumers with events or `wait_stream`.
Create rings, collators and encoder contexts before capture; keep the required owners through completion of the last
replay. See the [capture contract](encoding.md#streams-and-cuda-graphs) for lifetimes and operations allowed during
capture, and [encoded observations](rl.md#encoded-observations) for action expansion requirements.
