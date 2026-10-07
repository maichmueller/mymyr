# Command line

Installing the package installs a `mymyr` command (also available as `python -m mymyr`). It plans a PDDL task with one
of the CPU searches and writes the plan in the IPC plan format, or summarizes a domain and problem.

```bash
mymyr plan tests/data/pddl/logistics00/domain.pddl tests/data/pddl/logistics00/probLOGISTICS-6-1.pddl
mymyr plan domain.pddl problem.pddl --search gbfs --heuristic ff -o plan.txt -v
mymyr info domain.pddl problem.pddl
mymyr --version
```

## Planning

`mymyr plan DOMAIN PROBLEM` prints the plan to stdout, one action per line followed by a `; cost = ...` line, or writes
it to the file given with `-o/--plan-file`. `-v` adds statistics on stderr: expanded and generated states, stored
states, search and parse time, plan length and cost.

| Option | Meaning |
| --- | --- |
| `--search` | `astar` (default), `gbfs`, `brfs`, `iw`, `siw` or `astar_iw` |
| `--heuristic` | `blind`, `max`, `add`, `ff`, `h2`, `set_additive` or `perfect`, for `astar`, `gbfs` and `astar_iw` (default: `max` for A*, `ff` for GBFS) |
| `--width` | the largest novelty arity of `iw` and `siw` (default 2), the width of `astar_iw` (default 1) |
| `--threads` | threads of `brfs`; the other searches run on one thread |
| `--max-seconds`, `--max-states` | budgets |
| `--debug` | show the traceback of an error |

A* finds an optimal plan with an admissible heuristic: `blind`, `max`, `h2` or `perfect`. It and GBFS use the task's
action costs. BrFS finds a shortest plan. IW, SIW and AStarIW prune by novelty and can miss plans of a task whose width
is larger than the one given. Numeric tasks run in every search but AStarIW, which refuses them with an error.

The exit code tells what happened:

| Code | Meaning |
| --- | --- |
| 0 | a plan was found |
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
