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
construction examples. CUDA searches accept `GroundCondition` goals containing fluent literals only; they reject
derived literals and numeric constraints. `search.CancelToken` supports cancellation from another thread.

An `observer` may implement `on_start`, `on_expand`, `on_generate`, `on_prune`, `on_pass`, `on_solution`,
`on_progress` and `on_end`; IW-family algorithms also call `on_transition`. A false return from `on_progress` stops
the search. Callback exceptions cancel the search and are raised when it returns. The C++ search runs with the Python
thread state detached; callback methods run with it attached. Independent calls can search one shared task at once.
Parallel rollout and portfolio workers use per-worker callbacks only when the observer/goal provides `make_worker(k)`.
