"""Run a jitted JAX environment step on CUDA."""

from pathlib import Path

import jax
import mymyr
from mymyr.rl.jax import Env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/blocks"
task = mymyr.Task.from_pddl(DATA / "domain.pddl", DATA / "probBLOCKS-8-0.pddl", atoms="frozen")
env = Env(task, device="cuda", max_steps=4)
key = jax.random.key(0)
state, observation = env.reset(key, 4)
state, observation, reward, terminated, truncated, info = jax.jit(env.step)(key, state)
print(state.states.shape, reward.shape)
env.check_errors()
