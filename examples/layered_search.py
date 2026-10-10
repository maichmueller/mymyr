"""Use goal-count ordering and a novelty-restricted layer beam in BrFS."""

from pathlib import Path

import mymyr
from mymyr import search

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "p01.pddl")

result = search.brfs(
    task,
    layer_order="goal_count",
    beam_width=8,
    beam_novelty="survivors_only",
    seed=0,
)
print(result.status, result.solved, len(result.plan), result.states)
