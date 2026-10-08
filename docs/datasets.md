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
`symmetry_pruning=True` to retain one state per object-graph certificate class; choose `certificate="kfwl"` (the
default) or `"color_refinement"`, and `k=2` (the default), `3` or `4` (see "Object graphs and certificates" below).
Symmetry pruning is a CPU option.

`datasets.generate(task)` returns a `GenerationResult` with status and counts when you need to distinguish a completed
generation from a limit. `generate_many(tasks)` processes a task collection. `generalized_state_space(tasks)` builds
each task's space, skips failed generations, sorts the spaces by size and constructs their combined class graph; the
tasks must come from one domain. `StateSpaceSampler(space, seed=...)` samples states and goal-distance layers
deterministically for a seed.

`state_space(task, device=0)` dispatches to `mymyr.cuda` for classical and numeric tasks and returns a
`DeviceStateSpace`; `to_host()` copies the same indexed space back, including CPU-encoded numeric values, costs
and goal distances. Device state-space generation does not support symmetry pruning.

## Object graphs and certificates

`datasets.object_graph(state)` builds the vertex-coloured graph of a state that symmetry pruning compares (as in
mimir: a vertex per object, coloured by its unary atoms and goal literals, and a path of vertices per atom and goal
literal of other arity). Its certificates are 128-bit integers, equal for isomorphic graphs:

- `color_refinement_certificate()`: colour refinement (1-WL), the cheapest and weakest.
- `kfwl_certificate(k)`: the k-dimensional folklore Weisfeiler-Leman algorithm for `k` = 2, 3 or 4. Every k-tuple of
  vertices starts from its ordered isomorphism type (the colours of its vertices, which of them are equal and which
  are adjacent) and is refined by the colours of the tuples that replace one of its positions, until no class splits.
  A larger k tells more graphs apart: the Cai-Fürer-Immerman pair over K4 (28 vertices) separates 2-FWL from 3-FWL,
  and the pair over K5 (60 vertices) separates 3-FWL from 4-FWL. Weisfeiler-Leman is not complete, so non-isomorphic
  graphs can still share a certificate; mimir's symmetry pruning uses nauty's canonical forms instead.

k-FWL on a graph of n vertices holds n^k tuples of 28 bytes and hashes n^(k+1) colour k-tuples per round, usually for
a handful of rounds, on one core. Two limits bound it, and a larger graph raises `ValueError` naming n, k and the
limit before any work:

| limit | default | largest n for k = 2, 3, 4 |
|---|---|---|
| `max_tuples` (n^k, memory) | 2^26 tuples (1.75 GiB) | 8192, 406, 90 |
| `max_round_work` (n^(k+1), time) | 2^30 (seconds per round) | 1024, 181, 64 |

Pass larger values to `kfwl_certificate(k, max_tuples=..., max_round_work=...)` when the memory and time are
acceptable. A 4-FWL certificate takes about 0.1 s at n = 20, 2 s at n = 40 and 10 to 40 s at n = 64, so k = 4 suits
symmetry pruning over object graphs of a few dozen vertices; symmetry-reduced state spaces use the default limits
(`StateSpaceOptions::fwl_limits` in C++).

```python
import mymyr
from mymyr import datasets

gripper = mymyr.Task.from_pddl("tests/data/pddl/gripper/domain.pddl", "tests/data/pddl/gripper/p02.pddl")
graph = datasets.object_graph(gripper.initial_state)
print(graph.num_vertices, graph.kfwl_certificate(4) != graph.kfwl_certificate(3))   # 14 True
try:
    graph.kfwl_certificate(4, max_round_work=10**5)
except ValueError as error:
    print(error)                                                    # ... n = 14 ... max_round_work ...
sym = datasets.state_space(gripper, symmetry_pruning=True, k=4)
print(sym.num_states)                                               # 12 classes of 28 states
```

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

tasks = TaskTable.from_pddl("tests/data/pddl/gripper/domain.pddl", "tests/data/pddl/gripper/p*.pddl", atoms="frozen")
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
