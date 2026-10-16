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
print(space.num_states, space.num_transitions, arrays["goal"].sum())
```

Use `max_states=` and `max_seconds=` to bound generation. `threads=` controls CPU generation workers (default 1, `0`:
one per hardware thread); the options follow the [API conventions](conventions.md).

`arrays()["state_words"]` holds the states as rows in the task's encoding, the same for every thread count and run
([State encodings](parsing-and-tasks.md#state-encodings)). A task with lazy atom slots has no such rows, so its arrays
leave `state_words` out.

`datasets.generate(task)` returns a `GenerationResult` when you need to distinguish a completed generation from a
limit: its `status` is a `search.Status` (`EXHAUSTED` for a complete space, `OUT_OF_STATES` or `OUT_OF_TIME` for a
budget, `UNSOLVABLE` when an unsolvable space was dropped), `stats` a `search.Statistics` (the stored states and the
time) and `space` the space or None.

```python
from mymyr import search

result = datasets.generate(task, max_states=100)
print(result.status, result.stats.states, result.space is None)
assert datasets.generate(task).status == search.Status.EXHAUSTED
```

A state is a goal state, an unsolvable state (no goal is reachable from it), or alive (neither): `is_goal`,
`is_unsolvable` and `is_alive` and the `goal`, `unsolvable` and `alive` flags of `arrays()` say which.

`generate_many(tasks)` processes a task collection. `StateSpaceSampler(space, seed=...)` samples states, goal-distance
layers (`sample_state_n_steps_from_goal`) and unsolvable states (`sample_unsolvable_state`) deterministically for a
seed; `num_solvable_states` counts the states that are not unsolvable, goal states included.

`generalized_state_space(tasks)` builds each task's space, skips failed generations, sorts the spaces by size and
joins them into one graph, their disjoint union; the tasks must come from one domain. State `s` of space `p` is vertex
`gss.vertex(p, s)` (`vertex_offsets[p] + s`), transition `e` of space `p` is edge `gss.edge(p, e)`, and
`gss.problem_of(v)` gives `(p, s)` back. `arrays()` holds the offsets, the forward CSR (`forward_offsets`,
`forward_targets`) and the initial, goal and unsolvable flags.

`state_space(task, device=0)` dispatches to `mymyr.cuda` for classical and numeric tasks and returns a
`DeviceStateSpace`; `to_host()` copies the same indexed space back, including CPU-encoded numeric values, costs
and goal distances.

## Task sets, knowledge bases and tuple graphs

A task set is many problems of one domain: a `mymyr.rl.TaskTable`, the same type the RL entry points take.
`TaskTable.from_pddl(domain, problems)` parses the domain once and instantiates the problems in parallel
(`threads=`, default 1). `problems` is a list of files (kept in its order), a directory (its `*.pddl` files
sorted by path), a glob pattern (its matches sorted by path; `**` spans directories) or one file. Domain files in a
directory or among a pattern's matches are skipped. Instance `i` is the task `Task(domain.instantiate(files[i]))`
with the same options. The table can be indexed (`table[i]`), pickled (it pickles its tasks), and passed wherever
tasks of one domain are taken: `generalized_state_space`, `KnowledgeBase`, `TaskSuite([table, ...])`.

A `KnowledgeBase` holds what is known about a task set: the state space of every
task whose generation succeeded (failures are skipped; `max_states` and `max_seconds` bound each one), sorted by size
unless `sort_by_size=False`; the generalized state space over them when `generalized=True`; and the tuple graph of
every vertex of every space when `width=` is given. `task_indices[i]` is the task of state space `i`. Every step runs
on `threads=` threads and gives the same result at every thread count. A knowledge base pickles and saves with what
it computed ([Saving and pickling](#saving-and-pickling)): loading generates nothing again.

```python
from mymyr import datasets
from mymyr.rl import TaskTable

tasks = TaskTable.from_pddl("tests/data/pddl/gripper/domain.pddl", "tests/data/pddl/gripper/p*.pddl", atoms="dense")
kb = datasets.KnowledgeBase(tasks, generalized=True, width=1)
print([s.num_states for s in kb.state_spaces], kb.task_indices)   # [8, 28] [0, 1]
gss = kb.generalized_state_space
print(gss.num_vertices, gss.num_edges, gss.vertex(1, 0))          # 36 128 8

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
keeps the smallest.

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

## Saving and pickling

State spaces, generalized state spaces, tuple graphs and knowledge bases pickle with what they computed, so they
travel to worker processes (`multiprocessing`, `concurrent.futures.ProcessPoolExecutor`, also with the spawn start
method) and come back equal: the same arrays bit for bit, states, labels, goal distances, sampler draws and tuple
graphs. A pickle holds the task (as its PDDL or normalized text and options, like a pickled `Task`) and the arrays;
unpickling validates the arrays and builds no state space, generalized state space or tuple graph again, apart from
the generalized state space's joined arrays, which take one linear pass. Objects pickled together share their task
and state space objects. Search results pickle as their task and fields: the plan as action labels, states as
`State`s, statistics and statuses as numbers. A `search.Heuristic` pickles as its specification (task, kind, costs,
evaluation; a perfect heuristic: its state space) and is built again when unpickled.

`StateSpace.from_arrays(task, arrays)` rebuilds a state space from the dict `arrays()` gives (NumPy arrays or DLPack
arrays on the CPU) and validates it: names, dtypes and shapes, the CSR ranges, the backward CSR against the forward
one, the flags against the goal distances, and that the arrays belong to `task` (its row layout and atom slots, its
initial state first, labels of its schemas and objects). `TupleGraph.from_arrays(space, root, arrays, width=...)`
does the same for a tuple graph. Invalid arrays raise `ValueError` or `TypeError` with a `mymyr: ...` message.

`datasets.save(obj, path)` writes a `StateSpace`, `GeneralizedStateSpace`, `KnowledgeBase`, `TupleGraph` or a list of
tuple graphs of one state space to a file, and `datasets.load(path)` reads it back. `load` builds the tasks from the
file; pass `task=` (or `tasks=`, one per task of the file in its order, or a `TaskTable`) to attach the dataset to tasks
you already have. A given task must be the file's task: another task (content fingerprint) or another encoding of
it (for example another `atoms=` or `numeric_storage=`, compared through the encoding fingerprint) raises
`ValueError`.

```python
import pickle
import tempfile
from pathlib import Path

import numpy as np

from mymyr import datasets

kb = datasets.KnowledgeBase(tasks, generalized=True, width=1)
copy = pickle.loads(pickle.dumps(kb))
assert copy.task_indices == kb.task_indices
assert np.array_equal(copy.tuple_graph(1, 0).arrays()["successors"], kb.tuple_graph(1, 0).arrays()["successors"])

space = kb.state_spaces[1]
rebuilt = datasets.StateSpace.from_arrays(space.task, space.arrays())
assert np.array_equal(rebuilt.arrays()["forward_targets"], space.arrays()["forward_targets"])

with tempfile.TemporaryDirectory() as d:
    datasets.save(kb, Path(d) / "gripper.mymyr")
    loaded = datasets.load(Path(d) / "gripper.mymyr", tasks=tasks)
    assert loaded.tasks[0] is tasks[0] and loaded.state_spaces[1].num_states == space.num_states
```

A task with lazy atom slots (`atoms="lazy"`) numbers its atoms in the order it first meets them, so its state rows
mean something only together with that numbering. Its datasets round-trip exactly through pickles and files, which
carry the slot table: unpickling and loading without `task=` give a task with the same numbering, and a given lazy
task takes the saved numbering when the slots it has assigned so far agree with it (a task that numbered its atoms
in another order is refused, and keeps its slots). `from_arrays` refuses such a task, as its `arrays()` hold no rows.

### File format

A dataset file is a zip archive, also readable with `numpy.load` (an `.npz` file); the file name is free. It holds:

- `header.json`: the format name `"mymyr.datasets"`, the format `version` (1), the `mymyr` version that wrote it, the
  `kind` (`state_space`, `generalized_state_space`, `knowledge_base`, `tuple_graph` or `tuple_graphs`), the tasks (their
  source, PDDL paths, `Task` options, content and encoding fingerprints as 16 hex digits), the state spaces (their task
  and generation statistics), the tuple graph groups, and the knowledge base's options and task indices.
- the tasks' PDDL (one member per distinct domain text) or normalized text, and for a lazy task its slot table as
  `tasks/<i>/atom_slots.npy`;
- every array as an `.npy` member: `state_spaces/<i>/<name>.npy` with the names of `StateSpace.arrays()` (rows
  always included), `tuple_graphs/<j>/<name>.npy` with each group of tuple graphs packed (`meta`, `sizes` and the
  arrays of `TupleGraph.arrays()` concatenated).

A reader loads every version up to its own, refuses a newer version or an unknown kind, and ignores sections and
fields it does not know, so later versions can add objects (encoding schemas and encoders, for example) without
breaking older files.

Loading a file reads data only: JSON, text and arrays read with `allow_pickle=False`, then validated as `from_arrays`
does, so a damaged or hostile file raises `ValueError` rather than running code. Pickles are the opposite: like every
pickle, `pickle.loads` can run arbitrary code, so unpickle only data you trust and share datasets as files.
