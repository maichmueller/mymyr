"""JAX planning environments over mymyr tasks. Needs JAX (jaxlib's XLA FFI headers at build time).

- :class:`Env`: N environments over a :class:`~mymyr.rl.TaskTable`, a :class:`~mymyr.rl.TaskSuite` (several domains,
  global task ids) or a Task (its table of one) as pure functions over a pytree state (:class:`EnvState`): ``reset(key, task_ids)`` (or ``reset(key, N)``: N environments of instance 0),
  ``step(key, state, action, next_task_ids=None)`` with SAME_STEP autoreset (into ``next_task_ids`` where given) and
  ``info["final_state"]``, ``reset_where``, ``refresh``,
  ``expand`` (padded) and ``expand_flat``. They are XLA FFI calls (CPU: ``rl::HostEnv``, CUDA: ``cuda::DeviceEnv`` on
  XLA's stream), so they run inside ``jax.jit``, ``jax.vmap`` and ``lax.scan``; JAX keys seed the counter-based RNG
  (:func:`key_seed`), and a jitted rollout equals the native environments for the same seeds::

      env = mymyr.rl.jax.Env(table, max_steps=100, goal_reward=1.0)      # rl.TaskTable([t0, t1, ...])
      state, obs = env.reset(key, task_ids)                                # int32 [16384]: each env's instance

      def body(state, _):
          state, obs, reward, terminated, truncated, info = env.step(key, state)   # the random policy
          return state, reward

      state, rewards = jax.lax.scan(body, state, None, length=200)

- mctx (:func:`mctx_recurrent_fn`, :func:`mctx_root`): batched MCTS over successor slots (action = successor index
  in canonical order, invalid slots masked) with a user value function.
- flashbax (:func:`transition_item`, :func:`transition_spec`): the pytree of one transition (state words, labels,
  rewards, masks) that flashbax buffers store from the environment's outputs without conversion. mymyr ships no
  replay buffer.
- :mod:`mymyr.rl.jax.rng`: the counter-based Philox in jnp; :func:`novelty_update` (width-1 novelty rewards) and
  :func:`prefix_masks` (factored action masks from the binding table) as jitted jnp functions equal to the native
  ``mymyr.rl`` versions.

Numeric tasks use the general device environment path.
"""

from ._env import Env, EnvState, FlatExpansion, Observation, PaddedExpansion, as_words, key_seed
from ._ffi import register, registered
from ._flashbax import transition_example, transition_item, transition_spec
from ._mctx import goal_count_value, mctx_recurrent_fn, mctx_root
from ._ops import HerBatch, her_relabel, novelty_update, prefix_masks, schema_masks
from . import rng

__all__ = [
    "Env",
    "EnvState",
    "FlatExpansion",
    "HerBatch",
    "Observation",
    "PaddedExpansion",
    "as_words",
    "goal_count_value",
    "her_relabel",
    "key_seed",
    "mctx_recurrent_fn",
    "mctx_root",
    "novelty_update",
    "prefix_masks",
    "register",
    "registered",
    "rng",
    "schema_masks",
    "transition_example",
    "transition_item",
    "transition_spec",
]
