# Formula values

`mymyr.formalism` exposes formula value classes alongside its read-only task views. `formalism.CLASSES` lists every
class in that module. The formula classes are `Variable`, `Atom`, `Literal`, `GroundAtom`, `GroundLiteral`,
`ConjunctiveCondition` and `GroundCondition`; task-made values are immutable, hashable and tied to the task that made
them.

`task.atom(...)` makes an `Atom` when a term is a variable such as `"?x"`, and a `GroundAtom` when every term is an
object. `task.literal(...)` adds positive or negative polarity. `task.condition(parameters, literals, ...)` constructs a
lifted conjunction with optional equalities and numeric constraints. `task.ground_condition(literals, constraints=...)`
makes a ground conjunction. `task.precondition(schema)` returns a `ConjunctiveCondition`; `task.goal_condition` is a
`GroundCondition`.

```python
from pathlib import Path
import pickle

import mymyr
from mymyr import formalism, search

data = Path("tests/data/pddl/blocks")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "probBLOCKS-8-0.pddl")
state = task.initial_state

# Lifted values use variables. A variable name starts with '?'.
lifted_atom = task.atom("on", "?x", "?y")
negative_literal = task.literal(lifted_atom, positive=False)
condition = task.condition(
    ["?x", "?y"],
    [task.literal("holding", "?x"), task.literal("clear", "?y")],
)
assert isinstance(lifted_atom, formalism.Atom)
assert isinstance(negative_literal, formalism.Literal)
assert isinstance(condition, formalism.ConjunctiveCondition)
assert isinstance(task.precondition("stack"), formalism.ConjunctiveCondition)
assert isinstance(task.goal_condition, formalism.GroundCondition)

# With only object terms, atom() and literal() return ground values.
on_a_g = task.atom("on", "a", "g")
not_holding_d = task.literal("holding", "d", positive=False)
ground = task.ground_condition([on_a_g, task.literal("clear", "d")])
assert isinstance(on_a_g, formalism.GroundAtom)
assert isinstance(not_holding_d, formalism.GroundLiteral)
assert state.holds(on_a_g) and state.holds(ground)
assert ground.holds(state)

# Lift replaces each distinct object with a variable. ground(state) returns the matching conditions.
lifted = ground.lift()
matches = lifted.ground(state)
assert ground in matches
assert all(match.holds(state) for match in matches)

# Numeric constraints use PDDL comparison syntax. This lifted condition is the precondition of "inc".
counters = Path("tests/data/pddl/counters")
numeric_task = mymyr.Task.from_pddl(counters / "domain.pddl", counters / "p01.pddl")
numeric_precondition = numeric_task.condition(
    ["?c"], [], constraints=["(<= (+ (value ?c) (step ?c)) (limit))"]
)
numeric_goal = numeric_task.ground_condition(constraints=["(>= (value a) 1)"])
assert str(numeric_precondition.numeric_constraints[0]) == "(<= (+ (value ?c) (step ?c)) (limit))"
assert not numeric_goal.holds(numeric_task.initial_state)

# CPU search accepts one GroundCondition or an any-of sequence of goals.
goal = task.ground_condition([task.atom("holding", "d")])
other_goal = task.ground_condition([task.atom("holding", "a")])
result = search.brfs(task, goal=[goal, other_goal])
end_state = state
for action in result.plan:
    end_state = task.apply(end_state, action)
assert result.solved and any(candidate.holds(end_state) for candidate in (goal, other_goal))

# Pickle formula values together with the Task that created them.
task2, goal2, state2 = pickle.loads(pickle.dumps((task, goal, state)))
assert goal2.holds(state2) == goal.holds(state)
```

`GroundAtom`, `GroundLiteral` and `GroundCondition` provide `holds(state)`. `State.holds(formula)` accepts the same
ground values, so either `goal.holds(state)` or `state.holds(goal)` checks the condition. Static literals use the
task's static facts, fluent literals use the state, derived literals use the task's axioms, and numeric constraints
use the state's function values.

`GroundCondition.lift(add_inequalities=False)` gives variables names `x0`, `x1`, and so on in first-appearance order.
Set `add_inequalities=True` when the generated variables must stand for distinct objects.
`ConjunctiveCondition.ground(state, partial=..., limit=...)` enumerates ground conditions for bindings that hold in
that state; it uses the same binding order and partial-binding format as `task.bindings`.

On CPU, search `goal=` accepts `None` (the task's goal), a callable `state -> bool`, a `GroundCondition`, or a
sequence whose entries are each a `GroundCondition` or a sequence of ground atoms/literals. Any one goal in that
sequence counts. A flat sequence of atoms is interpreted as multiple goals, so group conjunctions in a nested sequence
or make each one a `GroundCondition`. CUDA search also accepts `(positive_slots, negative_slots)` tuples;
CUDA multi-IW, batched IW(1) and rollouts also accept `GroundCondition` goals with derived literals and numeric
constraints. The goal list of a device batch supplies one conjunction per search; any-of alternatives within one
search and callable goals require CPU search. Numeric goal programs need at most 64 stack values; unsupported axiom
plans use the CPU fallback. Device A*/GBFS use the task's goal.

Pickle task-made formulas together with their `Task`, for example `pickle.dumps((task, task.goal_condition))`. Values
read directly from `task.formalism` do not carry a `Task` owner and cannot be pickled on their own.
