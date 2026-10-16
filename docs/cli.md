# Command line

Installing the package installs a `mymyr` command (also available as `python -m mymyr`). It plans a PDDL task with one
of the CPU searches and writes the plan in the IPC plan format, validates a plan, writes the normalized task, or
summarizes a domain and problem. Its options follow the [API conventions](conventions.md) (`--threads`, `--max-states`,
`--max-seconds`, `--width`).

```bash
mymyr plan tests/data/pddl/logistics00/domain.pddl tests/data/pddl/logistics00/probLOGISTICS-6-1.pddl
mymyr plan domain.pddl problem.pddl --search gbfs --heuristic ff -o plan.txt -v
mymyr plan domain.pddl problem.pddl --search anytime --weights 5,3,2,1 --max-seconds 60
mymyr validate domain.pddl problem.pddl plan.txt
mymyr write domain.pddl p01.pddl p02.pddl --format pddl -o normalized/
mymyr info domain.pddl problem.pddl
mymyr --version
```

## Planning

`mymyr plan DOMAIN PROBLEM` prints the plan to stdout, one action per line followed by a `; cost = ...` line, or writes
it to the file given with `-o/--plan-file`. `-v` adds the search's statistics on stderr: expanded, generated, stored
and pruned states, search and parse time, the memory of the state store, plan length and cost.

| Option | Meaning |
| --- | --- |
| `--search` | `astar` (default), `gbfs`, `ehc`, `anytime`, `brfs`, `iw`, `siw` or `astar_iw` |
| `--heuristic` | `blind`, `max`, `add`, `ff`, `h2`, `set_additive`, `lmcut`, `lmcount` or `perfect`, for `astar`, `gbfs`, `ehc`, `anytime` and `astar_iw` (default: `max` for A* and AStarIW, `ff` for the others) |
| `--weight` | the weight of `astar` (weighted A*, f = g + w·h; default 1) |
| `--weights` | the weight schedule of `anytime`, for example `5,3,2,1` (the default) |
| `--width` | the largest novelty width of `iw` and `siw` (default 2), the width of `astar_iw` (default 1) |
| `--threads` | threads of `brfs` (default 1, `0`: one per hardware thread); the other searches run on one thread |
| `--atoms` | the task's atom slots: `lazy` (default), `auto`, `reachable` or `dense` ([State encodings](parsing-and-tasks.md#state-encodings)) |
| `--max-seconds`, `--max-states` | budgets |
| `--debug` | show the traceback of an error |

A* finds an optimal plan with an admissible heuristic: `blind`, `max`, `h2`, `lmcut` or `perfect`; with `--weight w` the
plan costs at most w times the optimum. It, GBFS, EHC and the anytime search use the task's action costs. EHC (enforced
hill-climbing) climbs with h_FF's helpful actions and runs GBFS when hill-climbing fails. The anytime search
(restarting weighted A*) finds cheaper and cheaper plans with the weights of `--weights` and writes the best one when
it proves it optimal, exhausts its search space or a budget stops it; `-v` lists each plan's cost and time. BrFS finds a shortest plan. IW, SIW and AStarIW prune by novelty and can miss plans of a task whose width
is larger than the one given. Numeric tasks run in every search but AStarIW, which refuses them with an error.

The exit code tells what happened:

| Code | Meaning |
| --- | --- |
| 0 | a plan was found (by `anytime` also when a budget stopped it after a plan) |
| 1 | the task is unsolvable, or the search exhausted its space without a plan |
| 2 | invalid options, an unreadable file, PDDL mymyr cannot read, or another error |
| 3 | `--max-seconds` or `--max-states` stopped the search |

Errors are one line on stderr, for example
`mymyr: error: domain.pddl:5: durative actions are not supported (:durative-action move)`; see
[Parsing and tasks](parsing-and-tasks.md#errors) for the messages of the front end.

```python
import subprocess
import sys
from pathlib import Path

import mymyr
from mymyr import search

data = Path("tests/data/pddl/logistics00")
domain, problem = data / "domain.pddl", data / "probLOGISTICS-6-1.pddl"
run = subprocess.run([sys.executable, "-m", "mymyr", "plan", domain, problem, "-v"], capture_output=True, text=True)
assert run.returncode == 0
print(run.stdout.splitlines()[-1])  # ; cost = 14 (unit cost)

# the plan reads back with the plan parser
task = mymyr.Task.from_pddl(domain, problem)
plan = search.parse_plan(task, run.stdout)
assert len(plan) == 14

counters = Path("tests/data/pddl/counters")
unsolvable = subprocess.run([sys.executable, "-m", "mymyr", "plan", counters / "domain.pddl", counters / "p02.pddl"],
                            capture_output=True, text=True)
assert unsolvable.returncode == 1
```

## Validating a plan

`mymyr validate DOMAIN PROBLEM PLAN` replays an IPC plan file (one `(action object ...)` per line, `;` comments, names
without regard to case) on the task: each action must name the task's objects with the declared types and be
applicable, its effects are applied (conditional and numeric ones, derived predicates through the axioms), the goal
must hold at the end, and the plan's cost is computed under the task's metric. A valid plan prints
`valid plan: N actions, cost C` and exits with 0; an invalid one exits with 1 and names the first failing action (its
0-based index in the plan and its line in the file) and the reason on stderr:

```text
mymyr: invalid plan: plan[3] (stack c a) (line 4): the precondition (holding c) does not hold
mymyr: invalid plan: plan[0] (fly a b) (line 1): the task has no action 'fly'
mymyr: invalid plan: the goal does not hold after the plan's 17 actions: (on d f) is false
```

An unreadable file or PDDL mymyr cannot read exits with 2. In Python the same check is `search.validate_plan(task,
plan)` ([Search](search.md#validating-plans)).

## Writing the normalized task

`mymyr write DOMAIN PROBLEM...` writes tasks as mymyr normalizes them: `--format text` (the default) as mymyr's task
text (`Task.to_text()`, read back with `Task.from_text`), `--format pddl` as PDDL: one domain file shared by the
problems and a problem file per problem (`mymyr.write_pddl`). Each file reads back as the same task. With
`-o/--output DIR` the files go into DIR (`<problem>.txt`, or `domain.pddl` and `<problem>.pddl`); without it one
problem's text, or its domain followed by its problem, goes to stdout.

```python
import subprocess
import sys
import tempfile
from pathlib import Path

import mymyr

blocks = Path("tests/data/pddl/blocks")
with tempfile.TemporaryDirectory() as out:
    subprocess.run([sys.executable, "-m", "mymyr", "write", blocks / "domain.pddl", blocks / "probBLOCKS-8-0.pddl",
                    "--format", "pddl", "-o", out], check=True)
    again = mymyr.Task.from_pddl(Path(out) / "domain.pddl", Path(out) / "probBLOCKS-8-0.pddl")
    original = mymyr.Task.from_pddl(blocks / "domain.pddl", blocks / "probBLOCKS-8-0.pddl")
    assert again.to_text() == original.to_text()
```

## Summaries

`mymyr info DOMAIN [PROBLEM]` prints what mymyr reads from the files: the requirements, types, predicates by kind,
functions, whether the task is numeric, the action schemas after normalization (an action whose precondition has an
`or` becomes several schemas of its name), conditional effects and axioms, and with a problem the objects, initial atoms
and function values, goal and metric.

```python
import subprocess
import sys

counters = "tests/data/pddl/counters"
info = subprocess.run([sys.executable, "-m", "mymyr", "info", f"{counters}/domain.pddl", f"{counters}/p01.pddl"],
                      capture_output=True, text=True, check=True)
print(info.stdout)
assert "numeric: yes" in info.stdout
```

The same commands are available in Python as `mymyr.cli.main(["plan", domain, problem, ...])`, which returns the exit
code.
