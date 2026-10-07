"""Inspect a lifted precondition and lazily ground bindings that hold in a state."""

from pathlib import Path

import mymyr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "p01.pddl")
state = task.initial_state

condition = task.precondition("inc")
variable = condition.variables[0]
bindings = task.bindings(condition, state, partial={variable: "a"}, limit=4)
print(condition)
print([tuple(obj.name for obj in binding) for binding in bindings])

for binding, static, fluent, derived in task.ground_conjunctions(condition, state, limit=1):
    print(binding, static, fluent, derived)
