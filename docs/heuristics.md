# Heuristics

Create a heuristic with `mymyr.search.Heuristic(task, kind)`. Built-in kinds are `"blind"`, `"goal_count"`, `"max"`,
`"add"`, `"ff"`, `"h2"` and `"set_additive"`; set-additive also accepts `"hsa"` and `"setadd"`. Call the object on
a state for its value. Positive infinity means the heuristic proves the state is a dead end.

```python
from pathlib import Path

import mymyr
from mymyr import datasets, search

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
state = task.initial_state

for kind in ("max", "add", "ff", "h2", "set_additive"):
    heuristic = search.Heuristic(task, kind)
    print(kind, heuristic(state))

space = datasets.state_space(task, remove_if_unsolvable=False)
perfect = search.Heuristic.perfect(space)
result = search.astar(task, heuristic=perfect)
print(perfect(state), result.cost)
```

`max`, `add`, `ff`, `h2` and `set_additive` are delete-relaxation heuristics. They ignore numeric conditions and
effects. Set-additive sums the costs of its union of supporting actions.

Costs follow the task's objective by default (`costs="auto"`): every action costs 1 in a task without action costs and
metric, and its action cost otherwise. `costs="unit"` and `costs="real"` override this. Real costs are exact for zero
and fractional costs with up to 6 decimal places (the relaxation counts them in units of 10^-k) and rounded down
beyond; an action whose cost depends on the state, and every action under a metric over numeric fluents, costs 0 in
the relaxation, and an action whose cost is undefined is never applicable and left out. So `max` and `h2` never
overestimate the cost of reaching a goal, and A* with them finds an optimal plan. With `costs="unit"` on a task whose
actions cost less than 1 they can overestimate, and A* need not find an optimal plan. The h² and set-additive
implementations use grounded evaluation; they do not support `evaluation="lifted"`.

The perfect heuristic stores goal distances from a complete `mymyr.datasets.StateSpace`. Use
`search.Heuristic.perfect(space)` (the task's objective; `costs="unit"` or `"real"` to override) to reuse a space, or
pass `heuristic="perfect"` to A* or GBFS to have search generate it within the supplied state and time budgets. A state
outside the space is rejected.
