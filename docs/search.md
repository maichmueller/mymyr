# Search

`mymyr.search` provides layer-based searches, best-first searches (A*, weighted A*, GBFS and a heuristic beam),
enforced hill-climbing and an anytime search, and a plan validator. Every search returns a result with the same
fields, `status`, `solved`, `plan`, `cost`, `goal_state`, `stats` (a `search.Statistics`) and `message`, beside fields
of its own; its options follow the [API conventions](conventions.md) (`threads=`, `seed=`, the budgets, `width=` and
`max_width=`).

```python
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/counters")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")

result = search.brfs(task, layer_order="goal_count", beam_width=8)
print(result.status, result.solved, len(result.plan), result.cost, result.stats.expanded)

best_first = search.astar(task, heuristic="max")
print(best_first.status, best_first.cost, len(best_first.plan), best_first.stats)
```

## Layer-based search

`brfs` is breadth-first search. `iw`, `iw_pass` and `siw` use novelty pruning; `liw` restricts novelty with fact
landmarks. `abstracted_iw` and `projective_iw` use abstracted features. `rollout_iw` samples a rollout. The largest
novelty width of a ladder of IW passes is `max_width` (`iw`, `siw`, `liw`, the rollouts); a single pass takes `width`
(`iw_pass`, `abstracted_iw`, `astar_iw`). `astar_iw` is weighted A* with minimum-g novelty
pruning; it requires a classical task with unit action costs and does not guarantee an optimal plan.

Layer orderings are `"queue"`, `"in_order"`, `"reverse"`, `"randomized"` and `"goal_count"`. `seed=` fixes randomized
orders. `max_next_layer_states=` caps a generated layer. Ordered IW-family and BrFS searches can use `beam_width=`
(not with `max_next_layer_states=`) to keep the best states of each next layer; `beam_novelty="all_tested"` marks every
novelty-tested successor, while `"survivors_only"` commits novelty only for retained states. The latter is not
supported by LIW, and the two modes are equivalent for BrFS. `randomize_ties=True` uses `seed=` for equal scores. Every
search that takes these options raises `ValueError` for a combination it does not support (a `beam_width=` or a
`max_next_layer_states=` with `"queue"`, the default, both together, a zero width or limit, `randomize_ties=True`
without `"goal_count"`) before it starts. A layer beam is different from `search.beam`, which is heuristic best-first
search with a fixed queue width.

LIW requires `beam_novelty="all_tested"`. The IW family raises `ValueError` for `threads > 1` without a beam,
and a relaxed survivor beam rejects an observer, blocked states, a successor order or `tree="transitions"`.
Native C++ entry points throw `std::invalid_argument` for these layer and thread option errors.

`threads=` runs a layer beam's step (successor generation, scoring and read-only novelty tests) on that many threads
(`0`: one per hardware thread) for `brfs` and the IW family. With `"all_tested"` and `"survivors_only"` the plan and
every count equal the single-threaded search's, ties and seeds included; an observer or blocked states keep the
step on the calling thread. `threads > 1` without a beam is refused, except by the plain `brfs`, which has its own
parallel layers.

`beam_novelty="relaxed_survivors_only"` (with `layer_order="goal_count"`; not supported by LIW) is the relaxed
survivors-only beam: a layer's transitions, in generation order, are split into `min(threads, ceil(n / beam_chunk))`
contiguous parts (`beam_chunk=1024` by default). Each part keeps its best `beam_width` successors by the read-only
novelty test; the kept ones of all parts enter in rank order, skipping states seen before, until `beam_width` entered;
then the survivors-only replay commits novelty. Since the parts follow the thread count, so can the result: it is
deterministic for a given thread count and seed, and does not depend on the thread count when a layer has at most
`beam_chunk` transitions.

```python
exact = [search.iw(task, max_width=2, layer_order="goal_count", beam_width=4, threads=t) for t in (1, 4)]
print(exact[0].status == exact[1].status, exact[0].passes[-1].expanded == exact[1].passes[-1].expanded)

relaxed = search.brfs(task, layer_order="goal_count", beam_width=4,
                      beam_novelty="relaxed_survivors_only", beam_chunk=64, threads=4)
print(relaxed.status, len(relaxed.plan))
```

`astar` and `gbfs` accept `lazy=True` for lazy successor scoring, a heuristic, and its costs: `costs="auto"` (the
default: the task's objective), `"unit"` or `"real"` (see the heuristics guide). The
best-first `beam` also takes a heuristic and `beam_width=`. IW and SIW support numeric tasks; the other best-first
algorithms also handle numeric conditions, effects and metric costs. AStarIW does not accept numeric tasks.

## Weighted A*, enforced hill-climbing and the anytime search

```python
weighted = search.astar(task, heuristic="ff", weight=2)
climbed = search.ehc(task)                       # h_FF, helpful actions first, GBFS when hill-climbing fails
print(weighted.status, weighted.cost, climbed.status, climbed.ehc.lookaheads)

anytime = search.anytime_astar(task, heuristic="ff", max_seconds=10)
print([(s.cost, s.weight) for s in anytime.solutions], anytime.optimal)
```

### Weighted A*

`astar(task, weight=w)` is A* with f = g + w·h in place of g + h, eager or lazy (`lazy=True`); it is the same search as
A*, and `weight=1` (the default) is A* with the same expansions. The weight is finite and at least 1; it need not be an
integer (`queue="auto"` then uses a heap; `queue="bucket"` refuses a fractional weight). Ties are broken as in A*:
lower h first, then first in, first out, and goal states first among equal f, so the search is deterministic.

The plan costs at most w times the optimal cost C\* when the heuristic is admissible (`blind`, `max`, `h2`, `lmcut`,
`perfect`, with `costs="auto"`), the goal is the task's (or a list of alternatives; not a callable) and action costs
are non-negative:

- eager weighted A* with reopening (`reopen=True`, the default) for every admissible heuristic;
- eager weighted A* without reopening for a consistent heuristic (`blind`, `max`): a state is then expanded at most
  once and the bound still holds;
- lazy weighted A* with `lazy_requeue=True` (the default) for a consistent heuristic.

Real and unit costs, zero-cost actions and numeric metrics behave as in A*. A path-dependent heuristic (`lmcount`)
gets the status of each node as in A*: a state reached on several paths keeps the merge of their statuses, and eager
search evaluates a state again when a cheaper path reaches it.

`cost_bound=c` restricts the search to plans cheaper than `c`: a state whose g is at least `c` is not opened, and with
an admissible heuristic of the task's goal neither is a state whose g + h is. `EXHAUSTED` then means that the search
space holds no plan cheaper than `c`. The bound needs non-negative action costs: a transition that lowers g makes the
search fail (`FAILED`). The weight and the bound are A*'s: `gbfs` and `beam` do not take them.

### Enforced hill-climbing

`ehc(task)` is enforced hill-climbing (Hoffmann and Nebel 2001). From the current state, the start state first, a
breadth-first lookahead looks for a state with a strictly smaller h; the search commits to the first one it finds,
appends the path to it to the plan and continues from there. The lookahead evaluates a state when it pops it: a state
with h = ∞ is a dead end and is dropped, a state with a smaller h ends the lookahead, any other state is expanded. A
goal state ends the search as soon as it is generated. Within one lookahead every state is entered once, by the first
path that reaches it; a later lookahead may enter it again. The lookahead counts actions (it is breadth-first, also
with action costs), while g and the plan cost are the task's metric.

| Option | Default | Meaning |
| --- | --- | --- |
| `heuristic` | `"ff"` | any heuristic of `astar`, also one written in Python |
| `helpful_actions` | `"auto"` | `"first"`: expand with the heuristic's preferred operators only, and repeat a lookahead that finds no better state with every action; `"only"`: preferred operators only; `"off"`: every action; `"auto"`: `"first"` when the heuristic gives preferred operators, `"off"` otherwise |
| `fallback` | `"gbfs"` | when hill-climbing fails, eager greedy best-first search from the start state with the same heuristic and the rest of the budget; `"none"`: return `EXHAUSTED` (hill-climbing failed, which does not prove the task unsolvable) |

The preferred operators (FF's helpful actions) are the applicable actions of h_FF's relaxed plan (and of
set-additive's), evaluated grounded; the landmark count's are the actions that add an atom of a landmark the state
still needs; a heuristic written in Python gives them with `preferred_actions()`. `"first"` and `"only"` with a heuristic that
has none return `FAILED` with a message.

Hill-climbing is incomplete: it fails in a dead end it commits to, and with helpful actions also on a plateau whose
exits are not helpful. With the default fallback the search is complete on finite state spaces (GBFS is). `result.ehc`
counts the lookaheads, the improvements, the retries with every action, the longest lookahead, and tells whether
hill-climbing failed and the fallback ran; `stats`, `evaluations` and `dead_ends` count both phases.

A path-dependent heuristic (`lmcount`) gets, for every state of a lookahead, the status progressed along the path the
lookahead took from the current state, without merging other paths: the committed state's status is the status of
the plan up to it.

### Anytime search

`anytime_astar(task)` is restarting weighted A* (Richter, Thayer and Ruml 2010; the iterated searches of LAMA).
Iteration k runs weighted A* from the start state with the weight `weights[min(k, len(weights) - 1)]` and the cost
bound of the best plan found so far, so every plan it finds is cheaper than the one before. Every iteration starts
afresh; the heuristic is built once and shared.

| Option | Default | Meaning |
| --- | --- | --- |
| `weights` | `(5, 3, 2, 1)` | the weight schedule: non-empty, finite, at least 1 and non-increasing; the last weight repeats |
| `heuristic` | `"ff"` | any heuristic of `astar` |
| `lazy` | `False` | lazy weighted A* iterations (with `lazy_requeue`, `preferred_operators` and the list weights of `astar`) |
| `cost_bound` | `None` | the bound of the first iteration |

The other options are those of `astar` and apply to every iteration. `max_seconds` and `max_expanded` bound the whole
search, `max_states` and `max_depth` each iteration. The search ends when

- an iteration exhausts its search space: no plan is cheaper than the best one. With `reopen=True` (the default) and
  no `max_depth` this proves the plan optimal (`result.optimal`);
- an eager iteration with weight 1 and an admissible heuristic of the task's goal finds a plan with reopening: A*'s
  plan is optimal (`result.optimal`);
- a budget, the cancellation or an observer stops an iteration.

The status is `SOLVED` when any iteration found a plan, also when a budget stopped the search later; `plan` and `cost`
are the best plan's. `result.solutions` lists every improved plan in the order found with its cost, time, weight and
iteration, and `result.iterations` every iteration with its weight, bound, status and statistics. An observer gets
`on_solution(plan, cost)` for every improved plan as it is found and `on_pass(iteration, stats)` after every iteration,
besides the events of the iterations themselves (with each iteration's own state ids).

A cost bound prunes by g, and with an admissible heuristic also by g + h, so an iteration that cannot improve ends
when its open list runs dry; with an inadmissible heuristic such as h_FF the last iteration expands every state
cheaper than the best plan before it proves the plan optimal, and a time budget usually ends the search first.

### Differences from Fast Downward and FF

- Weighted A*: Fast Downward's `eager_wastar(..., w=...)` orders by f = g + w·h, then h, first in first out, as
  mymyr does, and reopens closed states by default, but its weight is an integer. It tests the goal when it expands a
  state; mymyr recognizes a goal state when it is generated and returns it when it is the minimum, goal states first
  among equal f, so on the last f-layer it can stop earlier. Fast Downward searches the task its translator produces:
  facts and operators that cannot matter are gone, so states that differ only in them are one state there (see the
  LM-cut differences above), and the expansions differ also with the same heuristic values.
- Enforced hill-climbing: Fast Downward's `ehc(h, preferred=[h])` orders the lookahead by g, so with action costs it
  is a uniform-cost lookahead, not a breadth-first one. It keeps one search space for the whole run: a state seen in an
  earlier lookahead is never entered again, while every mymyr lookahead starts afresh. It tests the goal when it
  commits to a state, mymyr when it generates one. Its `preferred_usage=PRUNE_BY_PREFERRED` (the default) is
  `helpful_actions="only"`; `RANK_PREFERRED_FIRST` expands every action and orders the preferred successors first, which
  mymyr does not offer. When hill-climbing fails Fast Downward stops (`fallback="none"`).
- FF's helpful actions are the applicable actions that add a goal of the relaxed plan's first layer, also those the
  relaxed plan did not choose; mymyr's are the actions of the relaxed plan itself, as Fast Downward's preferred
  operators of h_FF. FF restricts the lookahead to helpful actions and, when hill-climbing fails, starts a complete
  best-first search from the initial state; `helpful_actions="only"` with the default fallback is the closest mode,
  and the default `"first"` retries a failed lookahead with every action before it gives up.
- Anytime search: LAMA runs Fast Downward's `iterated` search over lazy GBFS and lazy weighted A* with the weights 5,
  3, 2, 1, repeating the last, with h_FF and the landmark count together, preferred operators and action costs plus
  one in the heuristics. mymyr's anytime search is the restarting weighted A* core with one heuristic: eager by
  default, `lazy=True, lazy_requeue=False` keys successors as Fast Downward's lazy search does. Fast Downward prunes a
  successor whose g reaches the bound; mymyr also prunes by g + h with an admissible heuristic. A Fast Downward
  iteration that finds no plan passes on to the next weight; in mymyr an exhausted iteration ends the search, since no
  cheaper plan exists.

## LM-cut and the landmark count

`"lmcut"` (also `"lm_cut"`) and `"lmcount"` (also `"lm_count"`, `"landmark_count"`) are heuristic kinds of
`search.Heuristic`, of every search's `heuristic=` and of the CLI's `--heuristic`.

```python
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/gripper")
task = mymyr.Task.from_pddl(data / "domain.pddl", data / "p01.pddl")
s = task.initial_state

print(search.Heuristic(task, "max")(s), "<=", search.Heuristic(task, "lmcut")(s))
optimal = search.astar(task, heuristic="lmcut")
print(optimal.cost, optimal.stats.expanded, optimal.expanded_until_last_jump)

lmcount = search.Heuristic(task, "lmcount")
status = lmcount.initial_status(s)
a = task.applicable_actions(s)[0]
t = task.apply(s, a)
status = lmcount.progress(status, s, a, t)  # the landmarks reached on the path s, t
print(lmcount(t, status), lmcount(t))       # along the path, and with t as its start
print(search.gbfs(task, heuristic="lmcount", lazy=True).status)
```

Both evaluate the grounded delete relaxation that `max`, `add` and `ff` use (`share=` reuses one grounding). They have
no lifted evaluation: `evaluation="lifted"` raises `ValueError`, and a state outside the grounding (only a state that
the initial state cannot reach) raises `RuntimeError`. Like the other relaxation heuristics they ignore numeric
conditions and effects. Negative preconditions count through a proposition per atom used negatively, true when the
atom is false. The CUDA heuristics and device searches refuse both kinds.

### LM-cut

LM-cut (Helmert and Domshlak 2009) is admissible, `h_max <= LM-cut <= h*`, so A* with it finds optimal plans. It
computes h_max with a supporter per operator (a most expensive precondition), marks the goal zone (the propositions
that reach the goal through zero-cost operators and their supporters), takes the cut (the operators reachable from the
initial state without entering the goal zone that have an effect in it), adds the cut's cheapest cost to the value and
subtracts it from the cut's operators, and repeats until the goal costs 0.

- Ties: an operator's supporter is the last of its most expensive preconditions in the grounding's order (the
  action's precondition, then the condition of the effect, each in the order of the schema). While the goal zone is
  marked, a zero-cost achiever's supporter moves to an equally expensive precondition without zero-cost
  achievers, else to one in the goal zone (Fast Downward's border and goal-zone detection).
- Costs: unit costs, or the task's action costs on the relaxation's integer scale (`costs="real"`): zero and
  fractional costs are exact up to 6 decimal places.
- Conditional effects: the relaxation has an operator per effect, but an action's cost is charged once. A cut lowers
  each of its actions once, however many of the action's effects it holds: every plan applies some action of the cut,
  so the cuts form a cost partitioning over actions. Effects of one action with the same conditions are one operator,
  and an effect whose static condition fails is left out. Lowering an action lowers its other effects too, so the cuts
  can sum to less than h_max; the value is then h_max.
- Derived predicates: the axioms are operators of cost 0, which no cut contains.
- A search goal of several alternatives (`goal=[...]`): the smallest value over them.

### Landmark count

The landmark count of LAMA (Richter and Westphal 2010), with Fast Downward's landmark progression. The landmarks are
`approximate_fact_landmarks` with sound first achievers (`first_achievers="possibly_before"`), derived atoms
(`include_derived_atoms=True`), greedy-necessary orderings and disjunctive landmarks of up to four atoms, one level
deep. Every plan makes each of them true at some point.

The value of a state depends on the path to it, through the landmarks the path reached (true in one of its states).
The needed landmarks are those not reached, and the reached ones that are false in the state and either a goal atom
or ordered greedy-necessarily before a landmark not reached. With unit costs the value is their number. With real
costs it adds, per needed landmark, the cheapest cost of its first achievers (not reached yet) or of all its achievers
(needed again); a landmark with a derived atom costs the cheapest action. A needed landmark without such an achiever
makes the state a dead end. The landmark count is not admissible (one action can achieve several landmarks).

The best-first searches (`astar`, `gbfs`, `beam`, `astar_iw`) keep the reached landmarks per node: a transition adds
those true in the successor, and a state reached on several paths keeps the landmarks all of them reached. The eager
searches evaluate a new state after generating every successor of its parent, so its status holds the other paths
from that parent too; the lazy searches evaluate when they pop it. Eager A* evaluates a state again when a cheaper path
reaches it. `initial_status`, `progress`, `merge` and `h(state, status)` give the same steps to a search written in
Python.

The landmarks are those of the task's initial state, where the paths start. A search started in another state
(`start=`), and a `Heuristic` called on a state alone, infer the start's status from the state: the landmarks true in
it or in the initial state, and those ordered greedy-necessarily before a landmark true in it but not initially, since
every path from the initial state reached them. Such a status (and every status merged with it) is marked inferred, and
a landmark that it needs again only for an ordering and that no action achieves does not make the state a dead end: the
path to the state may have reached the later landmark.

Preferred operators (the lazy searches' preferred list, `preferred_actions(state, status)`): the applicable actions
with an effect that fires in the state and adds an atom of a needed landmark that is false there.

A search goal of several alternatives fails (`Status.FAILED`): the landmarks are those of the task's goal. There is no
admissible variant (cost partitioning over the landmarks).

### Differences from Fast Downward

Fast Downward (`lmcut()`, and `landmark_sum(lm_factory=lm_rhw())` for the landmark count) works on the SAS+ task its
translator produces; mymyr works on the PDDL atoms. This accounts for most of the differences:

- State space: the translator drops facts and operators that cannot matter for the goal, and states that differ only
  in them are one state for Fast Downward. A* can then expand far fewer states there, also with the same heuristic
  values (and even without a heuristic).
- Language: Fast Downward's LM-cut refuses conditional effects and axioms that its translator does not simplify away.
  mymyr's LM-cut supports both: conditional effects through action-level cost sharing (each cut lowers an action
  once), derived predicates as free operators. Effects whose static condition fails never fire and are left out, as
  the translator does; the other relaxation heuristics (`max`, `add`, `ff`) keep them.
- LM-cut ties: Fast Downward takes the first of an operator's most expensive preconditions in the order of its SAS+
  variables, mymyr the last in the order of the grounding. On ties the cuts can differ, and so can the value (both are
  admissible) and the number of states A* expands before its last f-layer, in either direction.
- Negated atoms: Fast Downward has a fact for an atom being false, so a negative goal (snake: `(not (ispoint ?p))`) is
  a landmark there. mymyr's landmarks are positive atoms; a negative precondition costs 0 in LM-cut when the atom is
  false and is otherwise achieved by an action deleting it.
- Landmark generation: Fast Downward's RHW factory also follows the domain transition graphs of the SAS+ variables
  (lookahead landmarks) and finds reasonable orderings. On some tasks it finds many more landmarks (rubiks cube: about
  twice as many, which lets its greedy search solve tasks that mymyr's landmark count alone does not). mymyr finds
  derived atoms as landmarks too.
- Landmark progression: Fast Downward also stores the landmarks still needed per state (joined over the paths into a
  state) and requires a landmark again through its reasonable orderings. mymyr computes the needed landmarks from the
  reached ones and the state; for goals and greedy-necessary orderings, mymyr's only orderings, both give the same
  landmarks.
- States off the path: Fast Downward assumes its heuristic sees states reached from the initial state; mymyr infers the
  status of another start state (above).

### Landmark graphs

A `FactLandmarkGraph` (from `approximate_fact_landmarks`, `lifted_fact_landmarks`, or given atoms) names its atoms by
canonical id, a property of the task's content: the same in every process and under lazy atom slots. It pickles as
its task, the task's content fingerprint and its arrays; loading checks the fingerprint and validates the arrays, and
nothing is computed again. `to_arrays()` gives the same arrays (NumPy, documented on the method) and
`FactLandmarkGraph.from_arrays(task, arrays)` reads them back, validated against the task, for a cache of your own:

```python
import pickle

g = search.approximate_fact_landmarks(task, first_achievers="possibly_before")
blob = pickle.dumps(g)                    # with the task; or np.savez(path, **g.to_arrays())
g = pickle.loads(blob)
g.achievers(g.landmarks[0])               # the achiever index travels with the graph
```

The arrays include `fingerprint`, a `uint64[1]` array containing the task's content fingerprint. Loading into a task
with a different fingerprint raises `ValueError`, even when its objects and predicates match. Arrays without this
entry are accepted with structural validation only; their task identity cannot be checked.

The achiever index dominates the size: a row of 4 (W + 1) bytes per ground action of the grounding (W: the widest
schema) plus the achiever lists.

A graph made from given atoms has no achiever index unless `achiever_index=True`: then the task is grounded as
`approximate_fact_landmarks` grounds it. Both accept `max_operators=5_000_000`, `max_bindings=100_000_000`,
`max_seconds=60.0` (`None`: unlimited), and `max_ground_actions=None` (unlimited; bounds the ground actions the index
lists); exceeding any limit raises `RuntimeError`. The index is computed for the given landmarks, with first
achievers by `first_achievers=` (`"hmax"` or `"possibly_before"`):

```python
subgoals = search.FactLandmarkGraph(task, ["(at ball1 roomb)", "(free left)"], achiever_index=True)
subgoals.unique_achiever("(at ball1 roomb)")
```

First achievers are computed from the task's initial state, including for hand-made subgoal sets taken from later
states. The constructor accepts fluent and derived atoms in landmarks, disjunctive sets and orderings. Derived atoms
have no action achievers because axioms derive them; the graph's `achieved` / `unachieved` queries see fluent atoms
only.

On the landmarks of `approximate_fact_landmarks` with the same `first_achievers`, the rebuilt index equals the
generator's, with one exception: with `"possibly_before"` the generator lists first achievers only for the atoms it
expanded, the rebuild for every fact landmark and disjunctive member, so `first_achievers()` of an atom that is not a
fact landmark can differ. To keep a computed index exactly, pickle the graph (or keep its arrays).

## Goals, budgets and callbacks

Common search options include `start=`, `goal=`, `blocked_states=`, `max_states=`, `max_expanded=`, `max_depth=` and
`max_seconds=`. `max_states=` counts the states a search stores, the start state included (IW: the nodes of a
pass): the search stops with `OUT_OF_STATES` at the new state that fills them, inside its parent's expansion, so it
never stores more ([API conventions](conventions.md)); the deadline and the cancellation are checked inside
expansions as well (device BrFS and the numeric device A* and GBFS stop at the same state as the CPU's; the other
GPU searches check `max_states`, the deadline and the cancellation between chunks, and stop once a chunk's new states
reach `max_states`). On CPU, `goal=` accepts the task goal (`None`), a callable `state -> bool`, one
`mymyr.formalism.GroundCondition`, or a sequence of goals. Each goal in that sequence is a `GroundCondition` or a
sequence of ground atoms and literals; any one that holds ends the search. See [Formula values](formulas.md) for
construction examples. CUDA multi-IW, batched IW(1) and rollouts accept `GroundCondition` goals with fluent/derived
literals and numeric constraints; each batched goal selects one conjunction per search. Callable goals and any-of
alternatives within one device search are not supported; device A*/GBFS use the task goal.
`search.CancelToken` supports cancellation from another thread. A search that does not solve the task returns an
empty `plan`; an unsolved `siw` keeps the subplans of the subproblems it solved in `partial_plan`.

`brfs` takes the common options too. It stops at the first goal state it expands and returns a shortest plan
(`SOLVED`); with `threads > 1` it searches layer-synchronously and still returns a shortest plan. `max_depth=d`
expands the states of depth below `d` and stores those of depth `d`. `stop_at_goal=False` goes on through the whole
reachable space (for counts such as `num_goal_states`); a goal state reached on the way still makes it `SOLVED` with
that plan, and `exhausted` tells whether every state was expanded.

## Negative action costs

A* and its variants (weighted A* with or without a cost bound, lazy A*, the anytime search) need non-negative action
costs: with a negative one, reopening a state can lower its g value without end (a cycle of negative cost is cheaper
on every lap). They check every transition and stop with `FAILED` at the first one that lowers the metric value,
naming both values in `message`. Greedy best-first search, the heuristic beam, enforced hill-climbing and the
layer-based searches close each state once, so they terminate on such tasks; the plan they return is valid, and its
`cost` is the metric value it reaches, which may be negative. The device A* and GBFS need non-negative costs: both
stop with `FAILED` and a `message` at the first action of negative cost they generate.

```python
domain = """(define (domain d) (:requirements :strips :negative-preconditions :action-costs)
  (:predicates (a) (goal)) (:functions (total-cost))
  (:action refund :parameters () :precondition (not (a)) :effect (and (a) (decrease (total-cost) 1)))
  (:action finish :parameters () :precondition (a) :effect (and (goal) (increase (total-cost) 3))))"""
problem = "(define (problem p) (:domain d) (:init (= (total-cost) 0)) (:goal (goal)) (:metric minimize (total-cost)))"
negative = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
refused = search.astar(negative, heuristic="blind")
print(refused.status, refused.message)  # failed: A* needs non-negative action costs, ...
print(search.gbfs(negative, heuristic="blind").cost)  # 2.0
```

## Validating plans

`search.validate_plan(task, plan)` replays a plan and returns a `PlanValidation`. The plan is IPC plan text (a `str`,
as `format_plan` writes it and the CLI reads it), the path of a plan file (an `os.PathLike`), or a list of `Action`s
as the searches return them; `start=` replays it from another state. Each action must name an action of the task
with as many arguments as its parameters, objects of the parameters' types, and be applicable in the state the ones
before it reached (its precondition holds, derived atoms through the axioms, its effects are defined); its effects
are then applied, conditional and numeric ones included. The goal must hold at the end. The result is truthy when
the plan is valid and carries the plan's `cost` under the task's metric (as the searches report it), the reached
`state` and the parsed `plan`; otherwise `error` (a `search.PlanError`) says why, `step` is the 0-based index of the
first failing action (`len(plan)` when the goal does not hold at the end), `line` its line in plan text, `action`
and `condition` the action and the unsatisfied condition, and `message` all of it in one line.

```python
result = search.gbfs(task)
check = search.validate_plan(task, result.plan)
assert check and check.cost == result.cost

text = search.format_plan(task, result.plan)
broken = search.validate_plan(task, "\n".join(text.splitlines()[1:]))  # drop the first action
print(broken.error, broken.step, broken.message)
```

`mymyr validate DOMAIN PROBLEM PLAN` runs the same check from the [command line](cli.md#validating-a-plan).

## Search trees

Pass `tree="nodes"` or `tree="transitions"` to `iw`, `iw_pass`, `siw`, `liw`, `abstracted_iw`,
`projective_iw` or `find_rollouts_parallel`. The default is `None`; booleans are not accepted. Each recorded
`IwPass.tree` is a `search.SearchTree`. The optimized IW(1) placeholder pass has `tree is None`.
For SIW the passes are in `result.subproblems[i].passes`; for parallel rollouts they are in
`result[i].passes`. Trees use the pass's node ids, which are also the observer ids.

```python
result = search.iw(task, max_width=2, tree="transitions", max_states=10000)
tree = result.passes[-1].tree
print(tree.num_nodes, tree.num_transitions)
print(tree.parent, tree.depth, tree.row_goal)
for node in range(tree.num_nodes):
    begin, end = tree.offsets[node:node + 2]
    successors = tree.rows[tree.t_row[begin:end]]
```

All array attributes are read-only, zero-copy NumPy views. Each view retains the native storage after the result
or tree is deleted. `tree.task` owns the task that interprets the labels and rows. The scalar attributes are
`width`, `words`, `numeric_words`, `label_width`, `num_nodes` and `num_transitions`.

| Arrays | Shape and meaning |
| --- | --- |
| `rows` | uint64 `[R, words + numeric_words]`: distinct state rows, with numeric words last |
| `parent`, `depth`, `schema` | int32 `[N]`: parent node, depth and incoming action schema; root parent/schema are -1 |
| `binding` | int32 `[N, label_width]`: incoming full action bindings, padded with -1; root binding is all -1 |
| `node_flags` | uint8 `[N]`: bit 0 enumerated successors, bit 1 goal, bit 2 skipped expansion, bit 3 dropped by layer selection, bit 4 repeated queue entry |
| `offsets` | int64 `[N + 1]`: transition CSR offsets; empty for node recording |
| `t_schema`, `t_binding` | int32 `[T]`, `[T, label_width]`: transition action labels with -1 padding |
| `t_row`, `t_child` | int32 `[T]`: successor row and admitted node id (-1 when rejected) |
| `t_outcome` | uint8 `[T]`: numeric `TransitionOutcome` values as [observers](#observers) get them: `OPENED`, `DUPLICATE`, `BLOCKED`, `SELF_LOOP` or `PRUNED` |
| `row_goal` | uint8 `[R]`: the pass's goal test for every row |

Node `i` occupies row `i`. In transition recording, additional distinct successors follow the nodes in
first-generation order. Transitions of each node occupy `offsets[i]:offsets[i + 1]` in generation order.
A width-0 duplicate queue entry refers to the existing node and sets its repeated-entry flag; it does not add
a second row or node. Thus `generated_in_tree + 1` equals the node count for ordinary untruncated passes without
duplicate entries, while beams retain admitted candidates even when selection later drops them.

`tree.states()` returns all rows as `State` objects. `tree.labels()` returns incoming node `Action` objects,
with `None` for the root. Following `parent` and these labels reconstructs the path to any node. These convenience
methods allocate Python objects; array access does not. Trees, passes and results support pickling with the task,
its fingerprint, its lazy slot numbering and the arrays. Lazy rows are zero-padded to the task's word width at the
end of the search; the task's slot numbering interprets them.

Node recording keeps the fast novelty path and moves the retained node storage into the result. Transition
recording materializes every successor and interns distinct rows in native code. Its transition arrays require
`T * (17 + 4 * label_width)` bytes, in addition to distinct rows at `R * 8 * (words + numeric_words)` bytes and
node metadata. Numeric rows retain at least one fluent word, matching the task's numeric row encoding. Existing state and expansion budgets bound exploration; transitions and distinct rejected rows
can still greatly outnumber admitted nodes, so size those budgets for the task's branching factor.

Goal flags always use the search's goal: task, alternatives, a custom callable, or the current SIW subproblem's
goal. A callable is also invoked once for each recorded row whose goal was not tested by the search. Errors in
these calls propagate like other callback errors. Skipped nodes are counted expansions without successor generation;
goal nodes are not expansions. Budget, time and cancellation stops retain each pass's partial tree.

Parallel rollouts record independently on their workers and return trees in seed order, identical at every thread
count for a fixed seed on tasks with fixed atom slots. Array wrappers are created on the calling thread after the
workers join. A recorded IW beam runs its serial step even if `threads > 1` is requested. Transition recording has
the same restrictions as observers for `beam_novelty="relaxed_survivors_only"`.

## Observers

For harvesting IW training data, [search trees](#search-trees) retain native arrays without per-event Python callbacks.

`search.Observer` is the base class of search observers: subclass it, override the events you need and pass an instance
as `observer=` to `brfs`, `iw`, `siw`, `astar`, `gbfs`, `beam`, `ehc`, `anytime_astar`, `astar_iw` or the IW family
variants. The events are `on_start(state)`, `on_expand(id, state)`, `on_generate(parent, action, child, state, is_new)`,
`on_prune(parent, action, state)`, `on_pass(width, stats)` (after an IW pass, a BrFS layer with its depth, an EHC
lookahead or an anytime iteration with its index),
`on_solution(plan, cost)`, `on_progress(stats)` (every `progress_interval` expansions; returning False stops the search
with status `CANCELLED`) and `on_end(status, stats)`. A search never calls an event the subclass does not override,
and an observer that overrides none is not installed. `on_expand` and `on_generate` are called once per expanded and
generated state, so their counts equal the result's statistics.

`iw`, `iw_pass`, `siw`, the IW family variants, `astar_iw`, `rollout_iw`, `find_rollouts_parallel` and
`atomic_goal_portfolio` also send `on_transition(parent, action, child, state, outcome)` for every generated
transition, after its `on_generate` and `on_prune`; `child` is None when the successor got no id. The outcome is the
most specific `TransitionOutcome`, the same in all of them:

| Outcome | The successor |
| --- | --- |
| `OPENED` | entered the search as a new node or open state |
| `REOPENED` | was reached on a cheaper path and re-entered the open list (`astar_iw`) |
| `DUPLICATE` | is already known and nothing changed: a width-0 duplicate queue entry, a state already in the tree or among the next layer's candidates, a stored state not reached on a cheaper path |
| `BLOCKED` | is one of `blocked_states` (checked first) |
| `SELF_LOOP` | equals the state it was generated from |
| `DEAD_END` | has an infinite heuristic value (`astar_iw`) |
| `GOAL` | is a goal found on generation (`rollout_iw`) |
| `PRUNED` | was rejected for any other reason: not novel, beyond a depth or incumbent bound, or cut by a budget |

The other searches (`brfs`, `astar`, `gbfs`, `beam`, `ehc`, `anytime_astar`) do not send `on_transition`.

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
result = search.brfs(task, threads=4, observer=observer)
assert sum(w.expanded for w in observer.workers) == result.stats.expanded
print(len(result.plan), "steps;", len(set().union(*(w.threads for w in observer.workers))), "threads")
```

## Encoding the states of a search

The states a search meets, expanded or still on its frontier, encode into relational graphs for a learned heuristic or
policy ([native encoding](encoding.md)). The encoder takes the rows of state words that a search and a state space
already hold, so nothing is converted on the way.

**Frontier states.** A search observer sees every generated state; collecting them and encoding the batch once encodes
the frontier with one call:

```python
import mymyr
import numpy as np
from mymyr import datasets, encoding, search


class Frontier(search.Observer):
    def __init__(self):
        self.states = []

    def on_generate(self, parent, action, child, state, is_new):
        if is_new:
            self.states.append(state)


puzzle = mymyr.Task.from_text("tests/data/tasks/depot__p02.txt")
frontier = Frontier()
search.astar(puzzle, heuristic="max", observer=frontier)
encoder = encoding.FlatRelationEncoder(puzzle)
batch = encoder.encode(frontier.states[:256])                       # the states of the frontier, one graph each
assert batch.num_graphs == min(256, len(frontier.states))
```

**State spaces.** `StateSpace.arrays()` holds the rows of the states (`state_words`, one row per state id) and the
edges of the transition graph (`forward_offsets`, `forward_targets` and the action labels `label_schemas`,
`label_bindings`). Rows encode directly, and rows and edges together are the nodes and edges of a *horizon*: a state
and the states within a few actions of it, as one graph, so that a model sees a state and its candidate successors
together. `encoding.horizon_of_space(space, ids, depth)` builds the horizons of the states `ids` (merged by state, in the
order they are reached) from the space's arrays, and the horizon encoder encodes them:

```python
space = datasets.generate(puzzle, max_states=100_000, remove_if_unsolvable=False).space
arrays = space.arrays()
ids = [0, 5, 17]

# the rows of states encode directly
rows = arrays["state_words"][ids]
states = encoding.FlatRelationEncoder(puzzle).encode(rows)

# the neighbourhood of state 17 from the arrays: its row, its successors' rows, and one edge per transition
lo, hi = int(arrays["forward_offsets"][17]), int(arrays["forward_offsets"][18])
view = encoding.HorizonView(
    nodes=np.concatenate([arrays["state_words"][17:18], arrays["state_words"][arrays["forward_targets"][lo:hi]]]),
    node_offsets=np.array([0, hi - lo + 1], np.int32),
    edge_offsets=np.array([0, hi - lo], np.int32),
    edge_src=np.zeros(hi - lo, np.int32),
    edge_dst=np.arange(1, hi - lo + 1, dtype=np.int32),
    edge_schema=arrays["label_schemas"][lo:hi].astype(np.int32),
    edge_binding=arrays["label_bindings"][lo:hi].astype(np.int32),
)
horizon_encoder = encoding.FlatHorizonEncoder(puzzle, horizon_mode="full", parent_relation=True)
by_hand = horizon_encoder.encode(view)

# the same horizon from the space, and from expanding the state: all three give the same graph
from_space = horizon_encoder.encode(encoding.horizon_of_space(space, [17], depth=1, merge=False))
expanded = horizon_encoder.encode(encoding.horizon(puzzle, arrays["state_words"][17:18], depth=1, merge=False))
for other in (from_space, expanded):
    assert np.array_equal(by_hand.relation_args, other.relation_args)
    assert np.array_equal(by_hand.entity_offsets, other.entity_offsets)
```

A horizon's graph lists its edges in a canonical order (by destination node, then action schema and objects), so a view
that comes from a space, from an expansion or from a search's own bookkeeping gives the same batch whatever order its
edges are in.
