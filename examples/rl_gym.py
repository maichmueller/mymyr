"""Gymnasium environments over a PDDL domain: a greedy agent on symbolic observations, then a vector env that collects
a rollout batch in the layout policy-gradient trainers such as PPO take (observations, actions, action masks,
rewards, episode ends)."""

from pathlib import Path

import gymnasium as gym
import numpy as np

from mymyr import rl
from mymyr.rl import gym as mg  # registers mymyr/PDDL-v0

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/pddl/gripper"
# the domain's problems, parsed once for both environments (domain= and problems= would parse them per environment)
table = rl.TaskTable.from_pddl(DATA / "domain.pddl", DATA)

# One environment: the observation is a frozenset of ground atoms with the instance's objects and goal; an action is
# an index into the state's applicable actions, info["action_mask"] marks the valid ones.
env = gym.make("mymyr/PDDL-v0", table=table, max_steps=40, step_reward=-1.0, render_mode="ansi")
rng = np.random.default_rng(0)
obs, info = env.reset(seed=0)
print(env.render())

# Greedy: the successor that satisfies the most goal literals, ties broken at random. The successors come in the
# order of the action indices.
solved = episodes = 0
while episodes < 6:
    successors = env.unwrapped.task.successor_states(env.unwrapped.state)
    scores = np.array([sum(s.holds(lit) for lit in obs.goal.literals) for s in successors] or [0])
    best = np.flatnonzero(scores == scores.max())
    obs, reward, terminated, truncated, info = env.step(int(rng.choice(best)))
    if terminated or truncated:
        solved += info["is_goal"]
        episodes += 1
        obs, info = env.reset()
print(f"greedy: {solved} of {episodes} episodes reach the goal")

# A vector env: 8 environments stepped by one native pool, observations as 0/1 atom vectors of a fixed size, a
# random policy over the masked actions in place of a learned one. With same-step autoreset every step is a
# transition of the policy: an ending step returns the next episode's first observation.
envs = mg.VectorEnv(table, num_envs=8, observation="bits", max_steps=40, step_reward=-1.0, autoreset_mode="same_step")
T, N = 32, envs.num_envs
batch = dict(obs=np.zeros((T, *envs.observation_space.shape), np.int8), actions=np.zeros((T, N), np.int64),
             masks=np.zeros((T, N, envs.single_action_space.n), bool), rewards=np.zeros((T, N)),
             dones=np.zeros((T, N), bool))
obs, info = envs.reset(seed=1)
for t in range(T):
    mask = info["action_mask"].astype(bool)
    logits = np.where(mask, rng.normal(size=mask.shape), -np.inf)
    actions = np.where(mask.any(axis=1), logits.argmax(axis=1), 0)
    batch["obs"][t], batch["actions"][t], batch["masks"][t] = obs, actions, mask
    obs, reward, terminated, truncated, info = envs.step(actions)
    batch["rewards"][t], batch["dones"][t] = reward, terminated | truncated
returns = np.zeros((T, N))
running = np.zeros(N)
for t in reversed(range(T)):
    running = batch["rewards"][t] + 0.99 * running * ~batch["dones"][t]
    returns[t] = running
print({key: value.shape for key, value in batch.items()}, "mean return", round(float(returns.mean()), 3))
envs.close()
