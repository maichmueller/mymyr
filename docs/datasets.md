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

`state_space(task, device=0)` dispatches to `mymyr.cuda` for classical and numeric tasks and returns a
`DeviceStateSpace`; `to_host()` copies the same indexed space back, including CPU-encoded numeric values, costs
and goal distances. Device state-space generation does not support symmetry pruning.

## Task sets, knowledge bases and tuple graphs

A task set is many problems of one domain: a `mymyr.rl.TaskTable`, the same type the RL entry points take.
`TaskTable.from_pddl(domain, problems)` parses the domain once and instantiates the problems in parallel
(`threads=`, default all cores). `problems` is a list of files (kept in its order), a directory (its `*.pddl` files
sorted by path), a glob pattern (its matches sorted by path; `**` spans directories) or one file. Domain files in a
directory or among a pattern's matches are skipped. Instance `i` is the task `Task(domain.instantiate(files[i]))`
with the same options. The table can be indexed (`table[i]`), pickled (it pickles its tasks), and passed wherever
tasks of one domain are taken: `generalized_state_space`, `KnowledgeBase`, `TaskSuite([table, ...])`.

A `KnowledgeBase` holds what is known about a task set, as mimir's knowledge base does: the state space of every
task whose generation succeeded (failures are skipped; `max_states` and `max_seconds` bound each one), sorted by size
unless `sort_by_size=False`; the generalized state space over them when `generalized=True`; and the tuple graph of
every vertex of every space when `width=` is given. With `symmetry_pruning=True` the spaces are symmetry reduced and
problems isomorphic to an earlier one are dropped. `task_indices[i]` is the task of state space `i`. Every step runs
on `threads=` threads and gives the same result at every thread count. A pickled knowledge base stores its tasks and
arguments; unpickling builds it again.

```python
from mymyr import datasets
from mymyr.rl import TaskTable

tasks = TaskTable.from_pddl("tests/data/pddl/gripper/domain.pddl", "tests/data/pddl/gripper", atoms="frozen")
kb = datasets.KnowledgeBase(tasks, generalized=True, width=1)
print([s.num_states for s in kb.state_spaces], kb.task_indices)   # [8, 28] [0, 1]
classes = kb.generalized_state_space
print(classes.num_vertices, classes.num_edges)                    # 36 128

g = kb.tuple_graph(1, 0)              # state space 1, its initial state
for d in range(g.num_distances):
    for v in g.vertices_at(d):
        print(d, [str(a) for a in g.atoms(v)], g.problem_vertices(v), g.successors(v))
```

A tuple graph (Lipovetzky and Geffner, 2012) of a root state describes which tuples of at most `width` atoms are first
reached at each breadth-first distance from the root. Each vertex is such a tuple, together with its problem vertices:
the states at that distance in which the tuple is new. An edge `u -> t` means every problem vertex of `u` has a
successor among those of `t`. `width=0` gives the root and one vertex per successor state. Dominance pruning (on by
default) keeps only the tuples whose problem vertices are minimal; among tuples with the same problem vertices it
keeps the smallest. Over a symmetry-reduced space the problem vertices are class vertices.

`datasets.tuple_graphs(space, width=...)` builds the tuple graph of every vertex in parallel, and
`datasets.tuple_graph(space, vertex, width=...)` builds one. Each search can reach the whole space and enumerates
up to `width` atoms of every state it reaches. For large spaces, build the graphs of the vertices you need rather
than all of them. `arrays()` exports the CSR arrays without copying.

```python
space = datasets.state_space(tasks[1])
graphs = datasets.tuple_graphs(space, width=2)
assert graphs[5] == datasets.tuple_graph(space, 5, width=2)
a = graphs[0].arrays()
print(len(graphs), a["tuple_offsets"][:4], a["successors"][:4])
```
