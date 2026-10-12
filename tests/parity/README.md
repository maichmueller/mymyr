# Parity tests

mymyr is derived from [mimir](https://github.com/maichmueller/mimir) (a fork of
[simon-stahlberg/mimir](https://github.com/simon-stahlberg/mimir), v0.16.3), and mifrost's relational encoders, and
this directory is the only place where the code base refers to them: the tests here compare mymyr with results recorded
from those implementations. Everything outside `tests/parity/` and `bench/parity/` is independent of them and needs
none of their data (`ci/check_no_mimir.py` enforces that).

Parity tests carry the ctest label `parity` and the pytest marker `parity`. They run by default, and skip cleanly when
the reference data they need is absent. Select them or leave them out with `ctest -L parity` / `ctest -LE parity` and
`pytest -m parity` / `pytest -m "not parity"`.

## Layout

| path | content |
|---|---|
| `data/` | the recorded reference results (JSON, text exports), one directory per area |
| `cpp/` | the C++ parity tests (`test_<area>_parity.cpp`), the shared golden comparison (`golden.*`) and `export_task.cpp` |
| `common/` | helpers shared with `bench/parity/` (the reference's status names) |
| `python/` | the Python parity tests (`test_*_parity.py`) and `parity_support.py` |
| `tools/` | the scripts that regenerate the data (see below) |
| `fetch_data.sh` | fetches the reference data the tests read (see below) |
| `CMakeLists.txt` | the parity targets, included by `tests/cpp/CMakeLists.txt` |

`bench/parity/` holds the drivers that run the same configurations on both sides (`mymyr_iw`, `mymyr_brfs`,
`mymyr_heur`, `mymyr_landmarks`, `mymyr_iw_variants`, `mymyr_datasets`) and the reference-side programs (`fork_iw_trace`,
`fork_iw_variants`).

## What is compared

| area | recorded in | compared |
|---|---|---|
| front end | `data/golden/` | the text export of tasks (suite, numeric, IPC) token for token: ids, list orders, names |
| BrFS suite | `data/expected/` | states, expanded and generated counts of the 22-task suite |
| IW family | `data/expected/`, `data/iw_variants/` | per-pass counts, statuses, plan lengths and costs of IW, SIW, LIW, AIW, rollouts and the portfolio (24 configurations), name hashes of landing states and co-occurrence |
| best-first | `data/expected/`, `data/numeric_tasks/` | A*, GBFS, beam on the suite and the numeric tasks: status, cost, expansions |
| heuristics | `data/heuristics/`, `data/expected/` | h values along random walks (blind, goal count, h_max, h_add, h_FF, set-additive) |
| bindings | `data/bindings/` | applicable actions and bindings along walks |
| datasets | `data/golden/`, `data/tuple_graphs/` | state spaces, generalized state spaces, samplers' counts, tuple graphs (digests), knowledge bases |
| layer orders, beam, A*-IW | `data/layer_orders/`, `data/beam/`, `data/astar_iw/` | statistics under each layer ordering and beam mode |
| encodings | `data/encoding/` | relation catalogs of the encoding schema and the packed observations of the fixtures |

Python parity tests: `test_search_parity.py` (search results), `test_search_iw_variants_parity.py`,
`test_formulas_parity.py`, `test_datasets_parity.py`, `test_tuple_graphs_parity.py`, `test_frontend_parity.py`,
`test_encoding_parity.py`, `test_encoding_fixtures_parity.py`, `test_encoding_schema_parity.py`.

## Environment variables

| variable | meaning |
|---|---|
| `MYMYR_PARITY_DATA` | the reference data the tests read, with the layout below; tests that need it skip when it is unset. The CMake cache variable `MYMYR_PARITY_DATA_DIR` gives the default for the C++ tests. |
| `MYMYR_FORK_PLANS` | (C++ best-first parity) a directory of reference plans to compare plan costs with |
| `MYMYR_GOLDEN_DIR`, `MYMYR_GOLDEN_FILTER` | (C++ front-end parity) another golden directory, and a name filter |
| `MYMYR_IW_VARIANTS_GOLDEN` | (IW variants) another recorded-runs file |
| `MYMYR_WORK` | (the regeneration tools) the work directory with the reference checkouts and builds; default `<repo>/.work` |
| `MYMYR_FORK_DATA` | (`tools/make_golden.sh`) the fork's `data/` directory with `ipc/` |

Layout of `MYMYR_PARITY_DATA`:

```
mimir-fork/data/        the fork's benchmark data, with ipc/ (the IPC-2023 tasks; vendored in tests/data/ipc)
mimir-cs/Benchmark/     upstream benchmarks (classical and numeric)
mimir/data/             upstream benchmarks (main branch)
```

`tests/parity/fetch_data.sh DEST` assembles this layout from sparse single-commit checkouts of pinned revisions (the
shas are in the script) and copies `ipc/` from `tests/data/ipc`; point `MYMYR_PARITY_DATA` at `DEST`. CI does this in
`.github/actions/test-data`.

Local build: configure with `-DMYMYR_PARITY_DATA_DIR=<dir>` or export `MYMYR_PARITY_DATA` before running ctest.

## Regenerating the data

The recorded data is regenerated only on purpose, with the reference implementation built from source:

1. `tools/fork_golden/build_fork.sh` builds the reference (v0.16.3, commit 8033459) into `$MYMYR_WORK`, and
   `tools/fork_golden/build.sh` builds the exporters against it;
2. `tools/fork_golden/export_all.py` writes `data/expected/*.json` (BrFS, IW, walks, A*), checked by
   `tools/fork_golden/check_expected.py`; the other families come from the scripts in `tools/fork_golden/` (see its
   README for the format of each file);
3. `tools/make_golden.sh` writes the text exports of `data/golden/` (suite, numeric tasks, IPC sample);
4. `tools/check_fork_parity.py` compares mymyr's text export of every IPC-2023 task with the reference's;
5. `tools/export_mifrost_encoding_fixture.py` writes the encoding fixtures of `data/encoding/`.

## Known differences

The tests compare modulo these differences, which are intended; each one is a place where mymyr deliberately differs from
the reference, or where the reference's own output is not reproducible.

Front end and tasks
- Orders that depend on heap addresses in the reference (type literals of typed domains, the generated axiom parameters,
  grouped conditional effects of three or more groups, parameters appended by normalization) are made deterministic in
  mymyr (hierarchy order, variable-name order, creation order). `DomainOptions::generated_argument_order` reproduces one
  reference run; the golden comparison is modulo that freedom.
- The reference's lifted exporter always writes an axiom section; mymyr writes it only when needed.
- A variable of type `(either a b)` ranges over the objects of both types. Goals with a quantifier or a disjunction over
  a conjunction, universal effect conditions over a `forall` variable and nested conditional effects load in mymyr
  (the reference refuses them). A unary minus over a non-constant operand is kept in nullary numeric constraints.
- A conditional effect that does not fire records no effect family (the reference records it whether or not it fires),
  so it cannot make its action inapplicable. An `assign` that gives a function a value gets its slot from the start.
- The reference compares raw bit patterns of numeric values (-0 and +0 differ); mymyr canonicalizes them.
- `numeric_tolerant` and `WidthZero::RootOnly` follow the C# implementation of the reference instead.

Search
- IW: the width-0 default, optimized IW(1), blocked states and the statistics per pass are the reference's. SIW forwards
  blocked states and the observer to every pass (the reference's SIW has neither). The reference's beam is a BrFS/IW
  with layer orderings; mymyr's `beam` is the layer-synchronous best-first beam, the BrFS variant belongs to the IW family.
- Randomized orders use SplitMix64 (portable); the reference uses `std::mt19937_64` with `std::shuffle`. The parity
  tools re-implement SplitMix64's three functions so that randomized runs replay mymyr's.
- `DirectGoalAchieverFirst` inspects the adds of effects that fire, and of conditional effects without `forall`
  parameters; the reference inspects every conditional effect whether or not its condition holds.
- Best-first: a reopened goal state stays a goal and h is cached per state. `lazy_requeue=false` is the reference's
  lazy search (children keyed g(child) + h(parent)). A mymyr plan's cost can differ from the reference's rollout plan only
  when two actions with different costs have the same successor.
- Tie-breaking among equally cheap supporters in h_FF follows mymyr's operator order, the reference's follows a binary
  heap over its atom numbering; h_FF can differ where such ties exist. Goal count is not a reference heuristic (it is
  the complement of the reference's goal-count layer-ordering score). The reference's H2 ignores axioms and effect
  conditions and differs in its delete check; mymyr's h2 handles them.
- Heuristics use the task's action costs by default (`costs="auto"`), the reference's count every action 1.
- The portfolio and the parallel rollouts report the reference's records (landing states, co-occurrence, the states its
  plan extraction creates); the number of states differs only where a truncated layer cut a plan state's enumeration short.
- Widths up to 5 are supported (the reference's maximum arity is 6, so `find_solution` accepts at most 5).

Datasets
- State-space transition costs are the exact sums of the actions' `total-cost` increases (the reference computes
  `(depth + c) - depth`, which rounds fractional costs, and with a metric subtracts the parent's depth from the
  successor's metric value, which makes costs depend on the search order). Goal distances therefore equal plan costs.
- `max_seconds` bounds the generation (the reference declares `timeout_ms` but does not use it).
- Samplers use xoshiro256** (the reference uses `std::mt19937` seeded from `std::random_device`); counts match.
- Generalized state spaces sort ties stably (the reference's `std::sort` leaves ties unspecified; both agree for up to
  16 spaces).
- Tuple graphs: the tuple kept for a set of problem vertices is the smallest in the canonical order (the reference keeps
  the first in an unspecified hash-set order); the vertices, edges and problem vertices are the same. Deviations
  from the reference, which are bugs there: it adds one distance-1 vertex per transition (parallel transitions to one
  state give equal vertices) and leaves the root out of the problem vertices at distance 0.

Encodings
- The relation catalogs are compared name for name, arity for arity and slot role for slot role; the equality
  relations are kept by the reference and left out by default in mymyr (`equality=True` restores them), nullary
  relations appear in every horizon catalog of the reference, static relations are kept (empty) with
  `include_static=False`, and delta relations of static predicates are listed (they can never hold a tuple); the
  reference's `support_literals` option is not ported. The tests list each difference.
- Helper predicates for quantified conditions order their parameters by name in mymyr and by the reference's own
  traversal there; the fixtures map them.
