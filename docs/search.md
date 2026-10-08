# Search

`mymyr.search` provides layer-based searches, best-first searches and an optional heuristic beam. Results expose a
status and a plan; best-first results also expose the path cost and goal state.

```python
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")

result = search.brfs(task, stop_at_goal=True, layer_order="goal_count", beam_width=8)
print(result.status, result.solved, len(result.plan))

best_first = search.astar(task, heuristic="max")
print(best_first.status, best_first.cost, len(best_first.plan))
```

## Layer-based search

`brfs` is breadth-first search. `iw`, `iw_pass` and `siw` use novelty pruning; `liw` restricts novelty with fact
landmarks. `abstracted_iw` and `projective_iw` use abstracted features. `rollout_iw` samples a rollout. IW width is
selected with `max_arity` (or `width` on the abstracted variants). `astar_iw` is weighted A* with minimum-g novelty
pruning; it requires a classical task with unit action costs and does not guarantee an optimal plan.

Layer orderings are `"queue"`, `"in_order"`, `"reverse"`, `"randomized"` and `"goal_count"`. `seed=` fixes randomized
orders. `max_next_layer_states=` caps a generated layer. Ordered IW-family and BrFS searches can use `beam_width=`
(not with `max_next_layer_states=`) to keep the best states of each next layer; `beam_novelty="all_tested"` marks every
novelty-tested successor, while `"survivors_only"` commits novelty only for retained states. The latter is not
supported by LIW, and the two modes are equivalent for BrFS. `randomize_ties=True` uses `seed=` for equal scores. A
layer beam is different from `search.beam`, which is heuristic best-first search with a fixed queue width.

`astar` and `gbfs` accept `lazy=True` for lazy successor scoring, `costs="unit"` or `"real"`, and a heuristic. The
best-first `beam` also takes a heuristic and `width=`. IW and SIW support numeric tasks; the other best-first
algorithms also handle numeric conditions, effects and metric costs. AStarIW does not accept numeric tasks.

## Goals, budgets and callbacks

Common search options include `start=`, `goal=`, `blocked_states=`, `max_states=`, `max_expanded=`, `max_depth=` and
`max_seconds=`. On CPU, `goal=` accepts the task goal (`None`), a callable `state -> bool`, one
`mymyr.formalism.GroundCondition`, or a sequence of goals. Each goal in that sequence is a `GroundCondition` or a
sequence of ground atoms and literals; any one that holds ends the search. See [Formula values](formulas.md) for
construction examples. CUDA multi-IW, batched IW(1) and rollouts accept `GroundCondition` goals with fluent/derived
literals and numeric constraints; each batched goal selects one conjunction per search. Callable goals and any-of
alternatives within one device search are not supported; device A*/GBFS use the task goal.
`search.CancelToken` supports cancellation from another thread.

`brfs` takes `max_states=`, `max_seconds=` and `cancel=`, and its result has a `status` like the other searches.
With `threads > 1` it searches layer-synchronously and still returns a shortest plan with `stop_at_goal=True`.

## Symmetry pruning

`symmetry_pruning="wl1"` restricts the actions a search expands to representatives of symmetric objects, as mimir's
WL1 mode of its lifted successor generator. In every expanded state the objects are partitioned into the colour
classes of colour refinement (1-WL) on the state's object graph: the state's atoms, the static atoms and the goal.
For an action schema, let n(C) be the number of its parameters whose static domain meets class C; each parameter then
keeps only the n(C) first objects of each class C in its domain, and an action is expanded only if all its parameters
are bound to kept objects. Objects that are alone in their class are never pruned.

The option is accepted by `brfs`, `astar`, `gbfs`, `beam`, `astar_iw`, the IW family (`iw`, `iw_pass`, `siw`, `liw`,
`abstracted_iw`, `projective_iw`, `rollout_iw`), `find_rollouts_parallel` and `atomic_goal_portfolio`, and by
`Task.applicable_actions`, `iter_applicable_actions`, `successors` and `successor_states` (and the same `State`
methods), which return the actions a search with pruning would expand. The default is `"off"`. The device searches in
`mymyr.cuda` do not prune.

Pruning gives up completeness and optimality. Colour classes are coarser than the true symmetries, and even objects
that are truly interchangeable are pruned per parameter, so the pruned actions need not have a symmetric counterpart
among the kept ones: a search may report a solvable task as unsolvable, and `astar` may return a costlier plan than the
optimum. A plan found with pruning on is still valid. Pruning pays off on tasks with many interchangeable objects
(balls in gripper, packages in logistics, passengers in miconic); each expansion then costs one colour refinement of
the state's object graph.

```python
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/logistics00")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probLOGISTICS-6-1.pddl")

state = task.initial_state
print(len(task.applicable_actions(state)), len(task.applicable_actions(state, symmetry_pruning="wl1")))

full = search.brfs(task, stop_at_goal=True)
pruned = search.brfs(task, stop_at_goal=True, symmetry_pruning="wl1")
print(full.states, pruned.states, pruned.solved)

state = task.initial_state
for action in pruned.plan:
    assert task.is_applicable(state, action)
    state = state.apply(action)
assert task.is_goal(state)
```

## Observers

`search.Observer` is the base class of search observers: subclass it, override the events you need and pass an instance
as `observer=` to `brfs`, `iw`, `siw`, `astar`, `gbfs`, `beam`, `astar_iw` or the IW family variants. The events are
`on_start(state)`, `on_expand(id, state)`, `on_generate(parent, action, child, state, is_new)`,
`on_prune(parent, action, state)`, `on_pass(arity, stats)` (after an IW pass, or a BrFS layer with its depth),
`on_solution(plan, cost)`, `on_progress(stats)` (every `progress_interval` expansions; returning False stops the search
with status `CANCELLED`) and `on_end(status, stats)`; the IW family variants also send
`on_transition(parent, action, child, state, outcome)`. A search never calls an event the subclass does not override,
and an observer that overrides none is not installed. `on_expand` and `on_generate` are called once per expanded and
generated state, so their counts equal the result's statistics.

```python
from pathlib import Path

import mymyr
from mymyr import search


class Counter(search.Observer):
    def __init__(self):
        self.expanded = self.generated = 0

    def on_expand(self, id, state):
        self.expanded += 1

    def on_generate(self, parent, action, child, state, is_new):
        self.generated += 1


data = Path("tests/data/pddl/logistics00")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probLOGISTICS-6-1.pddl")
counter = Counter()
result = search.astar(task, heuristic="max", observer=counter)
assert (counter.expanded, counter.generated) == (result.stats.expanded, result.stats.generated)
```

The first exception an event raises stops the search and is raised when it returns. The search runs with the Python
thread state detached, and the events run with it attached on the thread that called the search. The parallel searches
(`brfs` with `threads > 1`, `find_rollouts_parallel`, `atomic_goal_portfolio`) first call `make_worker(k)` once per
worker; worker `k` then sends its `on_expand`, `on_generate`, `on_prune`, `on_transition` and `on_progress` events to
the observer it got, from its own thread, while `on_start`, `on_pass`, `on_solution` and `on_end` stay on the root
observer and the calling thread. On free-threaded Python the worker observers run at the same time, so each keeps its
own counts. Without `make_worker` (or when it returns None) these searches run on the calling thread alone. A goal
callable may provide `make_worker(k)` the same way.

```python
import threading


class Worker(search.Observer):
    def __init__(self):
        self.expanded = 0
        self.threads = set()

    def on_expand(self, id, state):
        self.expanded += 1
        self.threads.add(threading.get_ident())


class PerThread(search.Observer):
    def __init__(self):
        self.workers = []

    def make_worker(self, k):
        self.workers.append(Worker())
        return self.workers[-1]


observer = PerThread()
result = search.brfs(task, threads=4, stop_at_goal=True, observer=observer)
assert sum(w.expanded for w in observer.workers) == result.expanded
print(len(result.plan), "steps;", len(set().union(*(w.threads for w in observer.workers))), "threads")
```
