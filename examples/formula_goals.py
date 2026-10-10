"""Construct ground goals and pass an any-of goal list to CPU search."""

from pathlib import Path

import mymyr
from mymyr import search

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/blocks"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "probBLOCKS-8-0.pddl")

holding_d = task.atom("holding", "d")
goal = task.ground_condition([holding_d])
alternative = task.ground_condition([task.literal("holding", "a")])

result = search.brfs(task, goal=[goal, alternative])
if not result.solved:
    raise RuntimeError(f"search did not reach either goal: {result.status}")
end_state = task.initial_state
for action in result.plan:
    end_state = task.apply(end_state, action)
assert goal.holds(end_state) or alternative.holds(end_state)
print(search.format_plan(task, result.plan))
