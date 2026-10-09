# Fork golden data

`tests/data/expected/<domain>__<problem>.json` holds expectations exported from the mimir fork
(`maichmueller/mimir` v0.16.3, commit 8033459) for the mymyr parity tests. They are
regenerated only on purpose.

| file | role |
|---|---|
| `export_golden.cpp` | the exporter: one task per process, one `PHASE <name> <json>` line per finished phase |
| `CMakeLists.txt`, `build.sh` | build it against the fork install (`$WORK/install-fork`, made by `build_fork.sh`) |
| `export_all.py` | the driver: every suite task, one process per task, with budgets; assembles the JSON files |
| `check_expected.py` | the validator: structure, hashes, walk replay |

## Tasks

- the 22-task BrFS suite: `export_all.py`'s `BFS_INSTANCES`;
- the IW suite: `tests/data/tasks_iw/*.txt`, resolved against `MYMYR_WORK`'s fork clone (and, for the IPC-2023 tagged
  tasks, `MYMYR_IPC`).

Tasks are deduplicated by PDDL file (organic-synthesis-opt18-strips/p20 is in both). The file name is
`<last tag component>__<problem stem>`, which equals the name of the text export in
`tests/data/tasks/` for the BrFS suite. The `sets` field lists the suites a task belongs to.

## Fork configuration

Everything runs in the fork itself, built out of source (`build_fork.sh`):

- successor generation is lifted KPKC with symmetry pruning off;
- every search phase parses the task afresh (its own `Problem`, repositories and `SearchContext`), so its numbers equal
  those of a separate process and do not depend on earlier phases;
- IW is `iw::find_solution` with the fork's defaults (for `max_arity` 1 the width-0 pass is a placeholder of zeros);
- h_max, h_add and h_FF are the fork's grounded heuristics over a `LiftedGrounder`, as in the fork's `planner_astar`
  (unit action costs, the fork's convention); blind is the fork's `BlindHeuristic` (always 0);
- goal-count is not a fork heuristic: it is the number of fluent and derived goal literals that do not hold, which is
  the complement of the fork's `GoalCountLayerOrderingStrategy` score.

## Format (`"format": "mymyr-fork-golden/1"`)

```
{
 "format", "task", "sets", "source": {tag, problem, domain_file}, "fork": {version, commit, successor_generator},
 "names":  {objects, predicates, introduced_derived_predicates, schemas, duplicate_schema_names, num_axioms,
            num_static_init_atoms, num_fluent_init_atoms, num_goal_literals, has_metric},
 "brfs":   {status, states, expanded, generated, num_states_in_repository, budget},
 "iw":     [{k, status, plan_length, plan_cost, expanded, generated,
             passes: [{arity, expanded, generated, generated_in_tree}], budget}, ...],     // k = 1, 2
 "walks":  {num_steps, seed_base, full_limit, heuristics, heur_runs, h_unstable_steps,
            walks: [{seed, dead_end, duplicate_action_strings, steps}]},
 "astar":  {status, optimal_cost, plan_length, plan_cost, heuristic, expanded, budget},
 "killed": {phase: reason},                     // phases stopped by the driver (timeout, rss, error)
 "walks_items_dropped": {applicable, atoms},    // item lists removed to keep the file under 2 MB
 "meta":   {...}                                // timings, peak RSS; not expectations
}
```

- **names.** `objects` in the fork's order (`get_problem_and_domain_objects`); `predicates` as `[kind, arity, name]` with
  kind S, F or D: static predicates sorted by name, then fluent and derived ones in the order the fork's text exporter
  writes them; `schemas` as `{name, arity, original_arity}` in the domain's order. Object and schema indices equal the
  ids of the text exports (`O` section, `A` order), so tests that start from the text tasks can map object ids to the
  names used below. Map predicates by name (the text export's `P` lines carry names): the fork orders the static
  (type) predicates by heap address, so their order, and with it the predicate ids of a text export, changes from run
  to run (fluent and derived predicates and schemas are stable). `introduced_derived_predicates` lists the derived predicates that
  normalization adds (loki's `axiom_<k>`: derived predicates whose name does not occur in the domain file). `arity`
  counts all normalized parameters; `original_arity` the PDDL action's own.
- **brfs.** Exhaustive `brfs::find_solution` with `stop_if_goal = false`. `states` is the number of expanded states when
  the status is `exhausted`, else null (the budget stopped it; `expanded` is then a lower bound).
- **iw.** Per pass (index = arity) the fork's `brfs::Statistics`: `expanded`, `generated` (every generated
  transition), `generated_in_tree` (successors admitted to the queue). `expanded`/`generated` at the top are the sums.
  `status` is the fork's (`solved`, `failed`, `unsolvable`, `out_of_time`, ...); plan length and cost only when solved.
- **walks.** Each step describes one state: `fluent_atoms`, `derived_atoms` (all derived atoms, including those of
  introduced predicates) and `applicable` (the applicable ground actions) are sets; `is_goal` is the fork's goal test; `h` holds `blind`, `goal_count` and (after the heur phase) `hmax`, `hadd`, `hff`; `taken` is
  the action applied to reach the next step (null on the last step). A walk has `num_steps + 1` steps unless it hits
  a dead end (`dead_end`: no applicable action).
- **sets.** `{"count": n, "hash": "<16 hex>", "items": [...]}`. `items` is present for atoms always and for actions when
  `n <= full_limit` (5,000), unless the size budget dropped it (`walks_items_dropped`, largest sets first, applicable
  actions before atoms). `count` and `hash` are always present.
- **astar.** `astar_eager::find_solution`, blind heuristic when the task has a metric (action costs), h_max otherwise
  (both admissible). `optimal_cost` only when solved within the budget; it is the expectation, while `expanded` and
  `plan_length` depend on tie-breaking and are informative.
- **h values.** Numbers; JSON `null` means infinity (a relaxed dead end). h is evaluated in `heur_runs` independent
  fork processes (5; 50 when any value varies). h_max and h_add never varied, but h_FF breaks ties among achievers in
  an order that depends on heap addresses: where the runs disagree, the step's `h` carries `hff_observed` (the sorted
  distinct values; `hff` is the first run's), and `h_unstable_steps` counts such steps per heuristic. This happened only
  on rubiks-cube-opt23/sat23 p04 (68 and 72 of 78 steps), and the observed list is a sample, not the full range:
  earlier export runs produced 8 of 293 values outside it. Match h_FF exactly only where no `_observed` list is
  present; elsewhere only bounds (h_max <= h_FF) are expectations.

## String conventions

- atom: `(pred o1 ... ok)`, nullary `(pred)`; names as the fork prints them (lower case as parsed);
- atom of an introduced derived predicate (`names.introduced_derived_predicates`, e.g. `axiom_1`): `(axiom_1 ...)`
  with the objects **sorted** by byte order. The fork orders the arguments of the predicates it introduces by heap
  address (airport-adl, openstacks-opt08-adl and philosophers changed from run to run), so only the multiset of
  arguments is an expectation;
- ground action: `(schema o1 ... on)` over **all** normalized parameters (the schema's `arity`, not its
  `original_arity`); the schema name is the fork's action name;
- every list is sorted by byte order (C++ `std::string` `<`, Python `sorted` on these ASCII strings);
- set hash: `sum over items of FNV-1a-64(utf8(item))` mod 2^64, written as 16 lower-case hex digits (FNV offset
  0xcbf29ce484222325, prime 0x100000001b3). The sum makes it independent of order; it is a multiset hash.

## RNG

Walk w (0-based) uses seed `seed_base + w` (seed_base 1, three walks). The generator is splitmix64:

```
x += 0x9e3779b97f4a7c15; z = x;
z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; z = (z ^ (z >> 27)) * 0x94d049bb133111eb; return z ^ (z >> 31);
```

At every step the next action is `sorted_applicable[next() % count]` (sorted as above). The choice depends only on the
set of applicable actions, never on a generator's order, so an implementation with the same successor function
reproduces the walk. `duplicate_action_strings` flags a walk where two applicable ground actions print identically
(then the choice among them is ambiguous); check_expected.py replays the choices.

## Budgets (defaults of export_golden and export_all.py)

| phase | budget |
|---|---|
| brfs | 3,000,000 states, 300 s |
| iw1, iw2 | 300 s each (the fork's `max_time_in_ms`) |
| astar | 3,000,000 states, 300 s |
| walks | 3 walks of 25 steps; action lists above 5,000 actions keep only count and hash |
| heur | 5 independent processes per task, 50 when a value varies (each 900 s) |
| process | 1,800 s per task, then 900 s for each resumed process; 4 GB RSS; files below 2 MB |

When the driver kills a process (time, memory or a fork exception) it records the phase in `killed` and resumes the
remaining phases in a new process (heur is skipped when walks was killed; A* uses blind when heur was killed).
Sets, hashes and walk choices are expressed in names and do not depend on any fork order. IW counts of completed
passes were identical across repeated fork runs where checked (the address-dependent orders touch only the static
predicates and the argument order of introduced derived predicates); counts of a pass stopped by `max_time_in_ms`
depend on the machine and are not expectations.

## Regeneration

Every path comes from `MYMYR_WORK` (and, for IPC-tagged IW-suite tasks, `MYMYR_IPC`); see `build_fork.sh` for the
rest of its variables.

```
tests/data/fork_golden/build_fork.sh              # fork 0.16.3 -> $WORK/install-fork (once)
tests/data/fork_golden/build.sh                   # -> $WORK/build-fork-golden/export_golden
MYMYR_WORK=$WORK python3 tests/data/fork_golden/export_all.py --jobs 6
python3 tests/data/fork_golden/check_expected.py  # structure + sanity checks
```

`export_all.py --only <substr>` regenerates a subset; `--list` shows the tasks and their PDDL files.

Numeric tasks are not covered: the fork's IW(2) crashes (SIGSEGV) on counter tasks, its grounded heuristics ignore
numeric conditions and effects, and a walk state would need its numeric values; `tests/data/numeric_tasks` and
`tests/cpp/task/test_numeric.cpp` check numerics against the fork separately, without this format. The fork's A* and
GBFS results on the numeric tasks (`tests/data/numeric_tasks/fork_best_first.json`, checked by
`tests/cpp/search/test_best_first_numeric.cpp`) come from `search_fork/`: `run_numeric.py --build` builds the
driver against the fork install and runs it on the PDDL files of `make_golden.sh`'s numeric tasks.

The same driver gives the fork's IW(2) and BrFS under its layer ordering strategies (in order, reverse, goal count
with more or fewer satisfied goal literals first, with and without `max_next_layer_states`):
`search_fork/run_layer_orders.py` runs it on the BrFS suite tasks and writes
`tests/data/layer_orders/fork_layer_orders.json`, checked by `tests/cpp/search/test_layer_orders_fork.cpp`.
Randomized orderings are not compared (the fork shuffles with `std::mt19937_64`, mymyr with SplitMix64). The fork
enumerates a state's actions in the order of its clique search (fewest candidates first) where mymyr uses (schema,
binding) order; on the tasks where the two differ, reversed and truncated layers diverge, and the test compares only
what does not depend on that order there.

Its `--beam W --beam-mode all_tested|survivors_only` options run the fork's beam over the goal-count layers:
`search_fork/run_beam.py` runs IW(2) and BrFS with widths 1, 4 and 32 and both novelty modes on the BrFS suite tasks
(without philosophers, pegsol and organic-synthesis) and writes `tests/data/beam/fork_beam.json`, checked by
`tests/cpp/search/test_beam_fork.cpp`. A beam keeps ties in generation order, so it depends on the action order
everywhere: the test skips the tasks where the fork's order differs from mymyr's.

The beam mode `relaxed` (with `--threads N` and `--chunk C`) is the fork's relaxed SurvivorsOnly beam;
`--tie-seed S` randomizes equal-score ties and `--iw1-knobs precheck|atom_first|incremental` sets the fork's IW(1)
options. By default the fork selects the relaxed beam per expanded state (a layer can then exceed the beam width);
with the atom-first knobs its IW(1) pass selects per layer, as mymyr does. `search_fork/run_beam_relaxed.py` records
that pass of IW(2) on 2, 4 and 8 threads, with the exact SurvivorsOnly runs with and without the knobs as a control,
and writes `tests/data/beam/fork_beam_relaxed.json`, also checked by `tests/cpp/search/test_beam_fork.cpp`.

Its `--algo walk_ground` mode records the fork's binding generators along two seeded walks of 15 steps: per step the
groundings (count, set hash of the binding strings, ground literals per kind) of the goal literals as a
`ConjunctiveCondition`, and per action schema of its precondition (`ConjunctiveConditionSatisficingBindingGenerator`)
and of the action (`ActionSatisficingBindingGenerator`). `search_fork/run_bindings.py` runs it on the BrFS suite (without
organic-synthesis) and the numeric tasks and writes `tests/data/bindings/fork_bindings.json`, checked by
`tests/cpp/successor/test_bindings_fork.cpp`. The walks carry the actions taken, so numeric tasks are covered too.

Fork behaviour worth knowing when matching h values: on philosophers (derived goals `blocked`) the fork's h_max,
h_add and h_FF are 0 in non-goal states (`blocked` is defined with universal quantifiers, which normalization turns
into negated introduced derived atoms, and the relaxation ignores negative conditions); on
miconic-simpleadl (conditional effects) h_FF is 1 where h_max is 2 in 6 states (check_expected.py warns).

AStarIW results are generated by `search_fork/run_astar_iw.py` into `tests/data/astar_iw/fork_astar_iw.json`:
classical widths 1 and 2, and abstracted width 1, each with blind and h_max. `search_fork` also accepts
`--algo astar_iw --h blind|hmax --width K --features classical|abstracted|base_abstracted`.
The JSON retains cost refusals and resource failures; the comparison excludes instances with known successor-order
differences, because generation order determines tuple ownership and state ids.

## Tuple graphs

`search_fork/run_tuple_graphs.py` runs `search_fork --algo tuple_graphs` on the state-space suite
(`tests/cpp/datasets/fork_cases.inc`) and writes `tests/data/tuple_graphs/fork_tuple_graphs.json`
(`"format": "mymyr-fork-tuple-graphs/1"`), which `tests/cpp/datasets/test_tuple_graphs.cpp` and
`tests/python/test_tuple_graphs.py` compare with mymyr's tuple graphs:

    env MYMYR_FORK_DATA=<fork>/data python3 tests/data/fork_golden/search_fork/run_tuple_graphs.py --build-dir <dir> --jobs 6

Per task: `states` (the space with `remove_if_unsolvable = false`), `sample_step` and `roots`
(every `sample_step`-th vertex, `sample_step = ceil(states / 64)`, as state keys), and `graphs`, keyed `w0`, `w1p1`,
`w1p0`, `w2p1`, `w2p0` (width, dominance pruning), each a list with one digest per root:

- `n`, `m`, `p`: per distance the number of vertices, of edges into that distance (from distance 1), and of problem
  vertices.
- `v`, `e`, `q`: set hashes (the sum mod 2^64 of the FNV-1a-64 of the items, 16 hex digits) of the vertices
  (`"<d>:<atoms>:<keys>"`: the distance, the tuple's sorted atom strings concatenated, the sorted keys of its problem
  vertices joined by commas), of the edges (`"<u>><t>"` with `u` and `t` the `"<d>:<atoms>"` parts) and of the problem
  vertices per distance (`"<d>:<key>"`).

A state key is the FNV-1a-64 (16 hex digits) of the state's sorted fluent atom strings (`(pred o1 o2)`) joined by
newlines, followed by `"\n=%.17g"` per numeric value. The driver normalizes what the fork leaves unspecified or gets
wrong: the vertex and edge lists become sets (`dropped_duplicates` counts the duplicates: width-0 vertices of
parallel transitions), the root is a problem vertex at distance 0 of width 0, trailing empty distances are removed,
and under dominance pruning the tuple kept for a set of problem vertices is the canonical smallest of its class (fewer
atoms, then the sorted atom strings in lexicographic order; the classes come from the fork's own novelty table, and
the driver checks that the fork's tuple belongs to it). On deadend the fork crashes at width 2 (states without
fluent atoms); such tasks record `fork_failed_widths: [2]` and have width 0 and 1 only.
