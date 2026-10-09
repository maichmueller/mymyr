"""Run layer-synchronous BrFS on a classical task on device 0."""

from pathlib import Path

import mymyr
from mymyr import cuda

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/blocks"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "probBLOCKS-8-0.pddl", atoms="frozen")
result = cuda.brfs(task)
print(result.status, result.solved, len(result.plan), result.stats)
