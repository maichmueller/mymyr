# Parsing and tasks

`Task.from_pddl(domain, problem)` parses PDDL when mymyr is built with the loki front end (enabled by default).
`Task.from_text(path)` reads a normalized task file. `mymyr.formalism.read_task_text(path)` exposes the normalized
formalism without compiling a search task.

A `Task` owns the compiled planning data. Its `initial_state`, `State` values and `Action` labels are immutable and
hashable. A state contains true fluent atoms and, for numeric tasks, numeric values. `task.state(atoms, values=...)`
creates another state value; numeric tasks require values. Derived atoms come from axioms and can be read with
`state.derived_atoms()`.

`task.formalism` gives read-only views of types, objects, predicates, functions, action schemas, axioms, initial facts
and goals. These views keep the task alive. Action schemas expose their parameters, preconditions and conditional
effects. A ground `Action` is a schema plus its full object binding; use `task.apply(state, action)` to produce its
successor.

```python
from pathlib import Path

import mymyr

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
state = task.initial_state

print(task.formalism.schemas[0].name, task.formalism.schemas[0].parameters)
print(task.formalism.goal)
print(state.numeric_values())

action = task.applicable_actions(state, schema="inc")[0]
next_state = task.apply(state, action)
print(action, next_state.numeric_values())
```

## Binding generators

`task.precondition(schema)` and `task.goal_condition` return lifted conjunction views. `task.bindings(target, state)` lazily
enumerates parameter bindings whose conditions hold in that state. `target` can be a schema name/index or one of those
condition views. `partial=` accepts a mapping from parameter index/name to object, or a sequence with `None` for
unbound parameters; `limit=` caps the number of yielded bindings. `task.ground_conjunctions` yields the binding and its
ground static, fluent and derived literals.

```python
condition = task.precondition("inc")
variable = condition.variables[0]
bindings = task.bindings(condition, state, partial={variable: "a"}, limit=4)
print([tuple(obj.name for obj in binding) for binding in bindings])

for binding, static, fluent, derived in task.ground_conjunctions(condition, state, limit=1):
    print(binding, static, fluent, derived)
```

Use `task.local()` to make a per-thread handle for repeated calls in a hot loop. The handle shares the immutable task
and keeps its own workspace. The task's values and formalism views can be pickled with their owner.
