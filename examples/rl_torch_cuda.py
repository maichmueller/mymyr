"""Run a batched PyTorch planning environment on CUDA."""

from pathlib import Path

import mymyr
import torch
from mymyr.rl.torch import BatchedEnv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/blocks"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "probBLOCKS-8-0.pddl", atoms="frozen")
env = BatchedEnv(task, 4, device=torch.device("cuda", 0), max_steps=4, seed=0)
step = env.step(final=True)
print(env.states.shape, step.reward.shape)
