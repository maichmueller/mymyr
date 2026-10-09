"""Parse the shipped numeric counters task and find a metric-cost plan."""

from pathlib import Path

import mymyr
from mymyr import search

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "p01.pddl")

result = search.astar(task, heuristic="max")
if result.status != search.Status.SOLVED:
    raise RuntimeError(f"search failed: {result.status}")
print(search.format_plan(task, result.plan))
