"""Compare relaxation estimates with the perfect distance from a generated state space."""

from pathlib import Path

import mymyr
from mymyr import datasets, search

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "p01.pddl")
state = task.initial_state
space = datasets.state_space(task, remove_if_unsolvable=False)

for kind in ("max", "add", "ff", "h2", "set_additive"):
    print(kind, search.Heuristic(task, kind, costs="real")(state))

perfect = search.Heuristic.perfect(space, costs="real")
result = search.astar(task, heuristic=perfect, costs="real")
print("perfect", perfect(state), "plan cost", result.cost)
