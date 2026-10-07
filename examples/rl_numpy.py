"""Expand a batch and take a few steps in the asynchronous CPU environment pool."""

from pathlib import Path

import mymyr
from mymyr import rl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/counters"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "p01.pddl")

expansion = rl.expand(task, [task.initial_state], goal=True)
print("successors", expansion.offsets.tolist(), expansion.schema.tolist())

pool = rl.CpuEnvPool(task, num_envs=4, seed=0, max_steps=8)
batch = pool.reset()
for _ in range(3):
    batch = pool.step()
print(batch.states.shape, batch.reward.tolist(), batch.terminated.tolist())
