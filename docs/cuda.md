# CUDA

The optional CUDA backend builds with CUDA Toolkit 13.0 and needs an NVIDIA GPU at runtime. Python wheels
contain the CPU backend only. For a source build, set `MYMYR_CUDA=ON` and `CMAKE_CUDA_COMPILER` during the CMake
configure or Python package build.

Device operations include task upload and smoke kernels, BrFS, multi-IW and rollouts; batched grounded h-max, h-add
and h-FF; A* and GBFS; state-space generation; and the device path for `mymyr.rl.expand`. Device arrays can exchange
data with JAX and torch through DLPack. `mymyr.cuda.Context` owns a stream-ordered memory pool; set `max_bytes=` to
cap it. Arrays and device tasks keep their context alive.

CUDA upload, expansion, BrFS, multi-IW, rollouts, state spaces, RL environments and A*/GBFS support numeric fluent
values, preconditions, effects, goals and metric costs. Public state rows preserve CPU I32/F64 encoding; internal
numeric rows hold one canonical double per fluent slot. TaskTable/TaskSuite numeric batches use per-instance kernels
with the shared atom/numeric row boundary. Classical tasks retain their atom-only layout and device kernels.

The device heuristic kinds are `"max"`, `"add"` and `"ff"`; `"blind"` is also accepted by device A* and GBFS.
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
