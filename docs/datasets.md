# Datasets

`mymyr.datasets.state_space(task)` enumerates reachable states and every applicable action transition. A
`StateSpace` exposes states, transitions, initial/goal flags, goal distances and array views. State ids follow
breadth-first discovery order with canonical successor order. `None` means generation stopped or the initial state
cannot reach a goal while `remove_if_unsolvable=True` (the default).

```python
from pathlib import Path

import mymyr
from mymyr import datasets

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
space = datasets.state_space(task, remove_if_unsolvable=False)
arrays = space.arrays()

graph = datasets.object_graph(space.state(space.initial_state_id))
print(space.num_states, space.num_transitions, arrays["goal"].sum())
print(graph.color_refinement_certificate())
print(graph.kfwl_certificate(2))
```

Use `max_states=` and `max_seconds=` to bound generation. `threads=` controls CPU generation workers. Set
`symmetry_pruning=True` to retain one state per object-graph certificate class; choose `certificate="kfwl"` or
`"color_refinement"` and `k=2` or `3`. Symmetry pruning is a CPU option.

`datasets.generate(task)` returns a `GenerationResult` with status and counts when you need to distinguish a completed
generation from a limit. `generate_many(tasks)` processes a task collection. `generalized_state_space(tasks)` builds
each task's space, skips failed generations, sorts the spaces by size and constructs their combined class graph; the
tasks must come from one domain. `StateSpaceSampler(space, seed=...)` samples states and goal-distance layers
deterministically for a seed. An `ObjectGraph` exposes its arrays and color-refinement / k-FWL certificates; these
are structural fingerprints used by symmetry pruning.

For classical tasks, `state_space(task, device=0)` dispatches to `mymyr.cuda` and returns a
`DeviceStateSpace`; `to_host()` copies the same indexed space back. Device state-space generation does not support
symmetry pruning.
