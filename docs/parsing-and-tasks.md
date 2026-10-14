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

## Semantics

The semantics of the supported PDDL fragment are as follows; the notes at the end of this section list the points where an implementation could reasonably differ.

- **Types.** A parameter, a quantified variable or a `forall` effect variable of type `(either a b)` ranges over the
  objects of `a`, of `b` and of their subtypes. An object or constant declared `(either a b)` belongs to `a`, to `b`
  and to their supertypes.
- **Conditions.** Goals, preconditions, effect conditions (also inside `forall` effects and nested conditional
  effects) and derived predicate bodies may use every condition of the ADL fragment: `and`, `or`, `not` over any
  condition, `imply`, typed `exists` and `forall` in any nesting, equality `(= ?x ?y)`, constants and numeric
  comparisons. They are compiled into conjunctions of literals and comparisons plus derived predicates, as in Fast
  Downward's translator. Negations are pushed down to literals and comparisons; a negated comparison is the
  complementary one. A universal quantifier becomes the negated literal of a derived predicate `axiom_<k>` that holds
  when some binding satisfies the negated body. Disjunctions split an action or a derived predicate rule into one copy
  per disjunct, and a conditional effect into one effect per disjunct. An existential precondition adds its variables
  to the action's parameters, so the actions list one instance per witness. A goal part that is not a literal or a
  comparison becomes a nullary derived predicate `axiom_<k>`. So does the condition of an effect inside a `forall`
  effect that quantifies universally over the effect's variable, as a derived predicate over its free variables. The
  helper predicates `axiom_<k>` appear in `task.formalism` and among a state's derived atoms like any other derived
  predicate. Two consequences: a comparison with an undefined side does not falsify a universal quantifier (it is
  false negated, too), and a derived predicate that depends on itself through a universal quantifier is refused as not
  stratifiable (`ValueError`).
- **Conditional effects.** The effects whose conditions hold in the state an action is applied to fire, and every
  effect reads the values of that state. A numeric effect that fires conflicts with an earlier one of the same action
  on the same ground function: an `assign` with any other effect, an additive effect (`increase`, `decrease`) with a
  multiplicative one (`scale-up`, `scale-down`). A conflict makes the action inapplicable; effects that do not fire
  take no part in one. `total-cost` effects follow the same rules.
- **Numeric values.** A ground function without a value in `:init` is undefined, and so is an expression that reads
  an undefined value or divides by zero. A comparison with an undefined side is false, in preconditions, goals, axiom
  bodies and effect conditions alike. An action is inapplicable when an effect that fires has an undefined value,
  scales down by zero, or increases, decreases or scales an undefined value; `assign` gives an undefined function a
  value. `state.numeric_values()` shows undefined values as NaN. Every ground function an `assign` effect can target
  has a numeric slot from the start (so every state of a task has the same size); a task whose `assign` effects can
  target more than 65536 ground functions without an initial value is refused with `ValueError`.
- **Metrics.** With `(:metric minimize (total-cost))`, or without a metric in a domain that declares `total-cost`,
  the cost of a plan is the sum of the `total-cost` effects of its actions. With `(:metric minimize <expression>)` over
  numeric fluents, the cost of a plan is the expression's value in the state it reaches, and `total-cost` effects do
  not count. Without a metric and `total-cost`, every action costs 1. A `maximize` metric, and a metric that combines
  `total-cost` with other terms, are refused, naming the metric: best-first searches return `FAILED` with that message,
  the other searches, state spaces and heuristics with real costs raise `ValueError`. A metric that is undefined in the
  start state or in a reached state stops the search or state space with an error naming it.

```python
import mymyr
from mymyr import search

fuel = mymyr.Domain.from_string("""
(define (domain fuel) (:requirements :strips :numeric-fluents)
  (:predicates (at-goal))
  (:functions (fuel) (used))
  (:action fill :parameters () :precondition (and) :effect (and (assign (fuel) 2)))
  (:action drive :parameters () :precondition (>= (fuel) 1)
    :effect (and (at-goal) (decrease (fuel) 1) (increase (used) 1))))""")
fuel_problem = "(define (problem p) (:domain fuel) (:init (= (used) 0)) (:goal (at-goal)) (:metric {} (used)))"

fuel_task = mymyr.Task(fuel.instantiate_string(fuel_problem.format("minimize")))
assert fuel_task.numeric_names == ["(fuel)", "(used)"]  # fuel is undefined until `fill` assigns it
result = search.astar(fuel_task)
assert [str(a) for a in result.plan] == ["(fill)", "(drive)"] and result.cost == 1

refused = search.astar(mymyr.Task(fuel.instantiate_string(fuel_problem.format("maximize"))))
assert refused.status == search.Status.FAILED and "maximize" in refused.message
```

Notes:

- a variable of type `(either a b)` ranges over the objects of `a` and of `b`;
- goals with a quantifier or a disjunction over a conjunction (such as `(forall (?x - t) (imply (s ?x) (p ?x)))`),
  effect conditions with a universal quantifier over a `forall` effect's variable and conditional effects nested in a
  conditional effect with an existential condition load;
- a conditional effect that does not fire records no effect family, so it cannot make its action inapplicable;
- in a numeric constraint without parameters, a unary minus over an expression with fluents is kept;
- a division by zero is undefined also between constants, and a `scale-down` by zero makes its action inapplicable;
- `maximize` metrics and metrics that combine `total-cost` with other terms are refused rather than minimized, and an
  undefined metric value is an error;
- the transition costs of a state space are the exact sums of the actions' `total-cost` increases (no cost is computed as
  a difference of depths, which would round fractional costs), so its goal distances equal plan costs;
- heuristics use the task's action costs by default (`costs="auto"`, see the heuristics guide), whereas the unit-cost heuristics count
  every action 1.

## Errors

PDDL that mymyr cannot read raises `mymyr.PddlError`, a `ValueError`. Its message is one line that names the file, the
line, the action the error is in, and what is wrong: a syntax error, an undefined predicate, type, object or function,
a wrong number of arguments, a requirement used but not declared, or a construct mymyr does not support. Unsupported
constructs are durative actions, processes and events, preferences, trajectory constraints (`:constraints`), object
fluents, timed initial literals, non-deterministic (`oneof`) and probabilistic effects, and unknown requirement flags.
The parts are also attributes: `message`, `path` (None for PDDL given as a string), `line` and `action` (None when
unknown). A file that cannot be opened raises `FileNotFoundError` (an `OSError`).

```python
import mymyr

durative = """(define (domain d)
  (:requirements :strips :durative-actions)
  (:predicates (ready))
  (:durative-action go
    :parameters ()
    :duration (= ?duration 1)
    :condition (at start (ready))
    :effect (at end (not (ready)))))
"""
try:
    mymyr.Domain.from_string(durative)
except mymyr.PddlError as e:
    print(e)  # line 4: durative actions are not supported (:durative-action go)
    assert e.line == 4 and "durative actions" in e.message

blocks = mymyr.Domain.from_file("tests/data/pddl/blocks/domain.pddl")
try:
    blocks.instantiate_string("""(define (problem p) (:domain blocks)
  (:objects a b)
  (:init (clear a) (ontable a) (handempty))
  (:goal (on a c)))""")
except mymyr.PddlError as e:
    print(e)  # line 4: undefined object 'c'
    assert e.line == 4 and e.message == "undefined object 'c'"
```

`Domain.formalism` gives the normalized domain alone (types, constants, predicates, functions, schemas and axioms),
without instantiating a problem.

## State encodings

A state is stored as a row of 64-bit words: `task.words` atom words, one bit per fluent atom slot, and for numeric
tasks `task.numeric_words` words of numeric values after them. `task.encode(states)` returns these rows and
`task.decode(rows)` turns them back into states; `mymyr.rl` and `mymyr.datasets` exchange states in the same layout.
Derived atoms have slots too, but rows do not store them: they follow from the fluent atoms.

`Task.from_pddl`, `Task.from_text`, `Task(normalized)` and `rl.TaskTable.from_pddl` take the same keyword options,
and `task.options` returns them:

| Option | Values | Effect |
| --- | --- | --- |
| `atoms` | `'auto'` (default), `'reachable'`, `'dense'`, `'lazy'` | how atoms get their slots (below) |
| `reachability_budget` | steps, default `2**25`; `0` is unbounded | work bound of the analysis behind `'auto'` and `'reachable'` |
| `numeric_storage` | `'auto'` (default), `'i32'`, `'f64'` | the type of a numeric slot |
| `numeric_quantum` | `0` (default) or `q > 0` | stored values snap to multiples of `q` |
| `numeric_tolerant` | `False` (default) | tolerant numeric semantics: values below 1e6 snap to a 1e-9 grid, comparisons use a 1e-9 tolerance |
| `matching`, `fc_free_params` | `'auto'`, `'fixed'`, `'fc'`; `4` | how bindings are matched (no effect on the encoding) |

**Atom slots.** The *typed-dense* space of a task has one atom per predicate and object tuple whose objects can occur
at those positions: objects of the types that the effects and axioms adding the predicate allow there, plus constants
and the objects of initial atoms. The *relaxed-reachable* atoms are those that a delete relaxation of the task reaches from
the initial state, where actions keep every atom they add and ignore negative and numeric conditions. Every reachable
state holds only relaxed-reachable atoms.

- `'reachable'` gives one slot to each relaxed-reachable atom, in canonical order (by predicate, then by object
  tuple). It raises `ValueError` when the analysis needs more than `reachability_budget` steps.
- `'dense'` gives every atom of the typed-dense space a slot, at its canonical id. It needs no analysis.
- `'auto'` runs the analysis and keeps the dense layout when it is no wider in words, else the reachable one. When the
  analysis passes its budget, `'auto'` uses dense slots.
- `'lazy'` assigns a slot to an atom the first time a state or search meets it.

Under `'reachable'` and `'dense'` the slots of one predicate form a single range, ascending in the lexicographic
order of the atoms' object indices, and the ranges follow the predicate indices. Fluent and derived atoms number
their slots separately. `task.atom_metadata()` gives the ranges: predicate `p`'s fluent slots are
`pred_slot_offsets[p]` to `pred_slot_offsets[p + 1] - 1`, and its derived slots are given by
`derived_pred_slot_offsets` in the same way. Lazy slots have no such order.

Pick the mode that matches how you use the states:

- Keep `'auto'` for learning, datasets and anything else that stores or exchanges rows.
- Use `'lazy'` for searches on large tasks. Its states hold only the atoms the search met, which on large tasks can
  be a few dozen words instead of thousands.

The budget counts the analysis' join steps, not seconds, so whether it suffices depends only on the task. The default
takes a few seconds and about a gigabyte of memory at most. `task.info` reports the mode, the typed-dense and
reachable atom counts (`dense_fluent`, `reachable_fluent`, and their derived counterparts), `reachability_steps` and
`reachability_complete`. `task.num_atoms` is the number of fluent slots, `task.words` is `ceil(num_atoms / 64)`, and
`task.atom_mode` names the mode in use. For example, in gripper with 100 balls, 404 of the 10 816 typed-dense atoms are
reachable, so `'auto'` stores a state in 7 words instead of 170.

**Determinism.** Except under `'lazy'`, the encoding is a function of the task and its options. The same PDDL and
options give the same slots, words and rows in every run, thread count and process, and after pickling.
`task.encoding_fingerprint` hashes the encoding: the slot of every fluent and derived atom, plus the numeric slots and
their storage. Two tasks with equal fingerprints read and write rows alike. A task pickle carries the fingerprint, and
loading it raises `ValueError` if the rebuilt task encodes states differently.

**Lazy slots.** Under `'lazy'`, the numbering depends on the order in which searches meet atoms, including their
threads and earlier searches on the same task object. `task.words` grows as slots are assigned, up to
`task.max_words`. A row from such a task means nothing outside the task object, so the library refuses raw rows for
it:

- `decode`, `task.state(words)` and `rl.expand` raise `ValueError`;
- `rl.TaskTable` refuses the task;
- `StateSpace.arrays()` leaves out `state_words`.

States, searches, datasets and pickles still work. A task pickle carries the slots assigned so far, and the loaded task
numbers its atoms as the pickled one did.

**Numeric storage.** `'auto'` stores values as `'i32'` (two per word) when the task is statically integral and no
quantum is set. A task is statically integral when its initial values, constants and static values in fluent effects
are integers and its effects neither divide nor scale down. Otherwise values are stored as `'f64'`, one double per
word. Values are computed in double precision either way. An `'i32'` task raises `OverflowError` when a value leaves
the int32 range. `task.numeric_storage`, `task.numeric_slots` and `task.numeric_names` describe the slots.
`numeric_quantum` and `numeric_tolerant` change the state space, so neither is ever applied implicitly.

**Tables.** All instances of an `rl.TaskTable` share one numeric storage. `TaskTable.from_pddl` stores every instance
as `'f64'` when any of them needs it under `'auto'`. Building a table from tasks with different storages raises
`ValueError`, and `rl.TaskTable` refuses lazy tasks. Each instance keeps its own `words`: `table.words` is the widest
instance, and `table.instance_words` lists the widths. A batch's rows are as wide as their widest instance, or as wide
as requested (`words=`). Words past a row's own instance width are zero padding: the CPU paths never read them, and
the CUDA paths reject a row whose padding is not zero. Expansion, environment steps, goal tests and decoding cost
each row its own width. `rl.numeric_values(table, states, task_ids)` decodes the numeric words of mixed rows to
float64. The CUDA expansion and environments take tasks of up to 2048 objects; wider instances run on the host
([CUDA](cuda.md)).

**Encoding goal.** The planner works on the normalized goal, in which `forall`, `or` and `imply` have become derived
predicates (`axiom_<k>`). `task.encoding_goal` is the goal as written, expanded into signed ground literals, for
encodings of the goal. A tuple of `GroundLiteral`, positive literals first, then by predicate and objects, without
duplicates:

- conjunctions are flattened;
- `forall (?v - t) φ` is expanded over the objects of type `t` (constants included, in object order; a variable of
  type `(either a b)` ranges over the objects of either type);
- `imply ψ φ` is expanded to `φ` when `ψ` is decided by static atoms and equality alone and holds, and contributes
  nothing when `ψ` fails;
- a literal over a static predicate (equality included) is kept, whether it holds in the initial atoms or not; when it
  fails, `task.encoding_goal_unsatisfiable` is true, because no state is a goal;
- negation is pushed inward: `not or` is a conjunction of negations, `not exists` a `forall` of the negation,
  `not imply ψ φ` with a static `ψ` that holds is `not φ`, and `not not φ` is `φ`;
- a disjunction is decided by static atoms and equality where they can: `or`, `not and` (the disjunction of the
  negated parts), `exists` and `not forall` (a disjunct per binding of the variables) and `imply ψ φ` whose `ψ` they
  do not decide (`not ψ` or `φ`). When a disjunct holds, the disjunction contributes nothing; when all disjuncts but
  one fail, that one is expanded like any conjunct (its static literals kept); when all fail, the last one is
  expanded, and its failing static literals make the goal unsatisfiable. So
  `(forall (?x - t) (or (not (s ?x)) (p ?x)))` with a static `s` is expanded like
  `(forall (?x - t) (imply (s ?x) (p ?x)))`, and `(exists (?x - t) (and (s ?x) (p ?x)))` with a single object `o`
  that has `s` is `(s o)` and `(p o)`.

A conjunct that cannot be expanded (a disjunction with two or more disjuncts that static atoms and equality do not
decide, a numeric constraint) is left out, and `task.encoding_goal_complete` is false. For a task whose goal is a
conjunction of literals, the encoding goal is the goal itself and `encoding_goal_complete` is true. The planner's
goal, `task.goal_condition` and the search are unaffected. `Task.from_text` has no goal as written: the encoding goal
is the planner's goal without the `axiom_<k>` literals, complete if there were none, unless the text carries it
(`NormalizedTask.to_text(encoding_goal=True)`). The bindings of the quantifiers of a goal are counted up front (the
product of the sizes of their domains): a conjunct whose quantifiers would take the goal past 2^24 bindings in total
is left out whole, and the goal is incomplete. A task built without a front end (a `TaskData` with no recorded
encoding goal) gets the text derivation described above when the `Task` is created.

The text derivation recognizes the derived predicates that goal normalization introduced by their name (`axiom_<k>`),
so a task read from text whose goal uses a user predicate of that name has it left out of the encoding goal (and the
goal incomplete). Tasks from PDDL are not affected: they expand the goal as written.

## Binding generators

`task.precondition(schema)` returns a `ConjunctiveCondition`; `task.goal_condition` is a `GroundCondition`.
`task.bindings(target, state)` lazily enumerates parameter bindings whose conditions hold in that state. `target` can
be a schema name/index or a `ConjunctiveCondition`. `partial=` accepts a mapping from parameter index/name to object,
or a sequence with `None` for unbound parameters; `limit=` caps the number of yielded bindings.
`task.ground_conjunctions` yields the binding and its ground static, fluent and derived literals.

See [Formula values](formulas.md) for creating atoms, literals and conditions, checking them against states, and using
ground conditions as search goals.

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
