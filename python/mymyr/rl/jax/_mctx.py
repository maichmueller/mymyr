"""The mctx adapter: batched MCTS whose actions are successor slots.

The action space is the K successor slots of a node's state (the index into its canonical successor order);
slots at or past the state's successor count are invalid (masked at the root, a -1e9 prior inside the tree). A
simulation is one environment step with the given action (the device fast path: no host synchronization), so the
embedding is the environment state (:class:`EnvState`). The environment must not reset or truncate inside the tree:
use ``autoreset=False`` and ``max_steps=0``. The dynamics are exact (the planning task itself), the value and the
prior come from the user (``value_fn(params, obs) -> [B]``, ``prior_fn(params, obs) -> [B, K]`` logits; uniform
without a prior_fn)::

    env = mymyr.rl.jax.Env(table, autoreset=False)
    state, _ = env.reset(None, task_ids)
    root, invalid = mctx_root(env, state, value_fn, params, num_actions=K)
    out = mctx.gumbel_muzero_policy(params, key, root, mctx_recurrent_fn(env, value_fn, num_actions=K),
                                    num_simulations=32, invalid_actions=invalid)

A step from a goal or dead-end state is terminal (discount 0); an invalid slot chosen inside the tree does not move,
gets ``invalid_reward`` and discount 0. Successors past slot K are not searched (count > K truncates the action space).
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from ._env import Env, EnvState, Observation

__all__ = ["mctx_recurrent_fn", "mctx_root"]

NEG = -1e9


def _check(env: Env) -> None:
    if env.native.autoreset or env.native.max_steps:
        raise ValueError(
            "mymyr: the mctx adapter needs an environment without autoreset and truncation (Env(..., autoreset=False, "
            "max_steps=0)): a search must not leave the tree's states"
        )


def _logits(prior_fn: Optional[Callable], params: Any, obs: Observation, k: int) -> Any:
    n = obs.count.shape
    logits = jnp.zeros((*n, k), jnp.float32) if prior_fn is None else prior_fn(params, obs).astype(jnp.float32)
    return jnp.where(jnp.arange(k) < obs.count[..., None], logits, NEG)


def mctx_root(
    env: Env,
    state: EnvState,
    value_fn: Callable[[Any, Observation], Any],
    params: Any = None,
    prior_fn: Optional[Callable[[Any, Observation], Any]] = None,
    *,
    num_actions: int,
) -> tuple[Any, Any]:
    """``(mctx.RootFnOutput, invalid_actions [B, K])`` for the environments of ``state`` (their current states)."""
    import mctx

    _check(env)
    obs = env.observation(state)
    k = int(num_actions)
    root = mctx.RootFnOutput(
        prior_logits=_logits(prior_fn, params, obs, k),
        value=jnp.asarray(value_fn(params, obs), jnp.float32),
        embedding=state,
    )
    return root, jnp.arange(k) >= state.count[..., None]


def mctx_recurrent_fn(
    env: Env,
    value_fn: Callable[[Any, Observation], Any],
    prior_fn: Optional[Callable[[Any, Observation], Any]] = None,
    *,
    num_actions: int,
    discount: float = 1.0,
    invalid_reward: float = -1.0,
) -> Callable[[Any, Any, Any, EnvState], tuple[Any, EnvState]]:
    """The ``recurrent_fn(params, rng_key, action, embedding)`` of mctx: one environment step with ``action`` (the
    successor slot) from the node's state; reward = the environment's reward (``invalid_reward`` for an invalid slot),
    discount = 0 after a terminal step or an invalid slot, else ``discount``; value and prior logits of the child from
    ``value_fn`` / ``prior_fn`` (0 value after a terminal step)."""
    import mctx

    _check(env)
    k = int(num_actions)

    def recurrent_fn(params: Any, rng_key: Any, action: Any, embedding: EnvState) -> tuple[Any, EnvState]:
        del rng_key  # the dynamics are deterministic: the action is given
        state, obs, reward, terminated, truncated, info = env.step(0, embedding, action)
        invalid = info["invalid"]
        value = jnp.asarray(value_fn(params, obs), jnp.float32)
        out = mctx.RecurrentFnOutput(
            reward=jnp.where(invalid, jnp.float32(invalid_reward), reward),
            discount=jnp.where(terminated | invalid, jnp.float32(0.0), jnp.float32(discount)),
            prior_logits=_logits(prior_fn, params, obs, k),
            value=jnp.where(terminated, jnp.float32(0.0), value),
        )
        return out, state

    return recurrent_fn


def goal_count_value(table: Any) -> Callable[[Any, Observation], Any]:
    """A parameter-free value function: minus the unsatisfied goal atoms of the observed states (goal-count shaping)
    under each environment's goal (its per-env goal masks with goals=True, else its instance's goal of
    ``table``, a TaskSuite, a TaskTable or a Task); for tests and baselines."""
    from mymyr import rl

    gpos, gneg = rl.goal_masks(table, framework="jax")  # uint32 [I, 2W]
    pos, neg = jnp.asarray(gpos), jnp.asarray(gneg)

    def value(params: Any, obs: Observation) -> Any:
        del params
        if obs.goal_pos.shape[-1]:
            p, n = obs.goal_pos, obs.goal_neg
        else:
            p, n = pos[obs.task_id], neg[obs.task_id]
        s = obs.state[..., : p.shape[-1]]
        missing = jax.lax.population_count(p & ~s).sum(-1) + jax.lax.population_count(n & s).sum(-1)
        return -missing.astype(jnp.float32)

    return value
