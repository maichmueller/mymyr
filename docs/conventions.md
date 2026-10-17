# API conventions

Every module (`mymyr.search`, `mymyr.datasets`, `mymyr.rl`, `mymyr.encoding`, `mymyr.cuda`) and the CLI use the same
option names, defaults and result shape. A function documents only where it departs from these rules.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `threads=` | worker threads; `0` is one per hardware thread | `1` |
| `seed=` | the seed of one run's random choices | `0` |
| `seeds=` | a sequence of seeds, one run per seed | required |
| `max_states=` | states a search stores, the start state included | `None` (no limit) |
| `max_expanded=` | states a search expands | `None` |
| `max_depth=` | the length of the paths a search explores, in actions | `None` |
| `max_seconds=` | wall time of the whole call, in seconds | `None` |
| `max_<noun>=` | any other limit (`max_width=`, `max_next_layer_states=`, `max_steps=`, ...) | `None` or a stated value |
| `width=` | the novelty width of one pass (`iw_pass`, `astar_iw`, `abstracted_iw`, tuple graphs) | per function |
| `max_width=` | the largest novelty width of an IW ladder (`iw`, `siw`, rollouts) | per function |
| `beam_width=` | how many states a beam keeps (a layer beam, `search.beam`) | per function |
| `start=` | the state a search, a replay or a space starts from | the initial state |
| `cancel=` | a `search.CancelToken`, checked inside expansions | none |
| `observer=` | a `search.Observer` receiving the search's events | none |

The thread count never changes a result unless the function says so (the relaxed survivors-only beam does). Seeds are
integers in `[0, 2**64)`. `max_states` and `max_expanded` end a search with `OUT_OF_STATES`, `max_seconds` with
`OUT_OF_TIME`; `max_depth` leaves deeper states unexpanded, so a search that runs out of shallower ones ends with
`EXHAUSTED`. A budget stops a search when it is reached, exactly: the new state that makes the store hold
`max_states` states stops the search inside its parent's expansion, so a stopped search holds exactly
`max(max_states, 1)` states (the start state always counts), and `max_expanded` stops it before the expansion after
that many, on every store, thread count and device. State-space generation follows the same rule: it fails with
`OUT_OF_STATES` when the space has `max(max_states, 1)` states or more, and reports that limit as its `stats.states`.

`None` is the only "no limit": a limit of `0` is a limit of zero (`reachability_budget=0` gives the analysis no steps,
`max_disjunctive_landmark_depth=0` expands no layer), and a limit that cannot be zero refuses it with `ValueError`.
Boolean options name the feature they switch on (`canonical_order=`, `witness_pruning=`, `stop_at_goal=`).

## Names

Names are lower case (`k=`, `PaddedExpansion.k`), and a string option value has exactly one spelling: heuristics are
`"blind"`, `"goal_count"`, `"max"`, `"add"`, `"ff"`, `"set_additive"`, `"h2"`, `"lmcut"` and `"lmcount"`, action
orderings the names of `search.ActionOrdering` (`"direct_goal_achiever_first"`, ...); other spellings raise
`ValueError`. The C++ API uses the same names: `threads`, `seed`, `max_width` and `width` in its options, `stats` (a
`search::SearchStatistics`) in its results, and `search::SearchStatus` as its one status enum.

## Results

Every search result, on the CPU and on the device, has the same fields:

| Field | Value |
| --- | --- |
| `status` | a `search.Status` |
| `solved` | `status == Status.SOLVED` |
| `plan` | the list of `Action`s to the goal (empty without a plan) |
| `cost` | the plan's cost under the task's metric (`0.0` without a plan) |
| `goal_state` | the `State` the plan reaches (`None` without a plan) |
| `stats` | a `search.Statistics` |
| `message` | why a search failed or stopped early (empty when it ran to its end) |

`search.Statistics` has `expanded`, `generated`, `states` (stored), `pruned`, `seconds` (wall time) and
`memory_bytes` (the peak bytes of the search's state store, `0` where a search does not measure it). A search adds its
own fields beside these (`passes` for IW, `rollout` for Rollout IW, `num_goal_states` for BrFS, ...); device results add
`counters`, a dict of device counters and times, as do heuristics (`Heuristic.counters`: their evaluations). Counts
are integers, times are seconds as floats unless the name says otherwise (`*_ms`).

`search.Status` is the one status enum, also used by `datasets` (state-space generation):

| Status | Meaning |
| --- | --- |
| `SOLVED` | a goal state was reached; `plan` leads to it |
| `EXHAUSTED` | every reachable state was expanded (a generated space is complete) |
| `OUT_OF_STATES` | `max_states`, `max_expanded` or another work budget (`max_rollouts`) was hit |
| `OUT_OF_TIME` | `max_seconds` was hit |
| `CANCELLED` | the `CancelToken` was set or an observer stopped the search |
| `FAILED` | the search cannot run on this task; `message` says why |
| `UNSOLVABLE` | the goal can never hold (a static goal literal is false), or a space without goal states was dropped |
