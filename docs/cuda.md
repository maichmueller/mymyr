# CUDA

The optional CUDA backend builds with CUDA Toolkit 13.0 and needs an NVIDIA GPU at runtime. Python wheels
contain the CPU backend only. For a source build, set `MYMYR_CUDA=ON` and `CMAKE_CUDA_COMPILER` during the CMake
configure or Python package build.

Device operations include task upload and smoke kernels, BrFS, multi-IW and rollouts; batched grounded h-max, h-add
and h-FF; A* and GBFS; state-space generation; and the device path for `mymyr.rl.expand`. Device arrays can exchange
data with JAX and torch through DLPack. `mymyr.cuda.Context` owns a stream-ordered memory pool; set `max_bytes=` to
cap it. Arrays and device tasks keep their context alive.

At this revision, CUDA entry points accept classical tasks only. Numeric fluent values, numeric preconditions/effects
and metric costs are not available on the device. CPU search, datasets and RL paths have numeric-task support where
their API permits it. The device heuristic kinds are `"max"`, `"add"` and `"ff"`; `"blind"` is also accepted by device
A* and GBFS.

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
