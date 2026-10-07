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
perfect = search.Heuristic.perfect(space, costs="real")
result = search.astar(task, heuristic=perfect, costs="real")
print(perfect(state), result.cost)
```

`max`, `add`, `ff`, `h2` and `set_additive` are delete-relaxation heuristics. They ignore numeric conditions and
effects; select `costs="unit"` or `costs="real"` for action-cost handling. Set-additive sums the costs of its union of
supporting actions. The h² and set-additive implementations use grounded evaluation; they do not support
`evaluation="lifted"`.

The perfect heuristic stores goal distances from a complete `mymyr.datasets.StateSpace`. Use
`search.Heuristic.perfect(space, costs="unit" | "real")` to reuse a space, or pass `heuristic="perfect"` to A* or
GBFS to have search generate it within the supplied state and time budgets. A state outside the space is rejected.
