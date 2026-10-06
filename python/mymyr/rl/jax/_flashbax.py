"""flashbax item specs (mymyr ships no replay buffer).

A transition is a flat dict of arrays in the environment's own formats (labels, not slots, so stored
transitions survive re-expansion)::

    state       uint32 [2 RW]   the state the action was taken in (state words as uint32 halves)
    task_id     int32           its instance (the label's objects are that instance's)
    action      int32           the successor slot (index into the canonical order)
    schema      int32           the action's label: schema (-1: no move)
    binding     int32 [L]       and objects (-1 padding; with labels=True)
    reward      float32
    terminated  bool
    truncated   bool
    next_state  uint32 [2 RW]   the state the step reached (info["final_state"]: before the autoreset)
    goal        bool            the reached state is a goal state
    goal_pos / goal_neg  uint32 [2W]  (goals=True) per-transition goal masks, for goal-conditioned buffers and HER

:func:`transition_spec` gives the shapes and dtypes (a dict of ``jax.ShapeDtypeStruct``: flashbax's item buffers take
an example item, ``buffer.init(transition_example(env))``), :func:`transition_item` the batch [N] of one step, ready for
``buffer.add(buffer_state, item)`` of a buffer made with ``add_batches=True``::

    buf = flashbax.make_item_buffer(max_length=1 << 20, min_length=1024, sample_batch_size=256, add_batches=True)
    bstate = buf.init(transition_example(env))
    state2, obs2, reward, terminated, truncated, info = env.step(key, state, action)
    bstate = buf.add(bstate, transition_item(obs, action, (state2, obs2, reward, terminated, truncated, info)))
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from ._env import Env, Observation

__all__ = ["transition_spec", "transition_example", "transition_item"]


def transition_spec(env: Env, *, labels: bool = True, goals: bool = False) -> dict[str, jax.ShapeDtypeStruct]:
    """The shapes and dtypes of one transition of ``env`` (see the module docstring)."""
    rw2 = 2 * env.row_words
    spec: dict[str, jax.ShapeDtypeStruct] = {
        "state": jax.ShapeDtypeStruct((rw2,), jnp.uint32),
        "task_id": jax.ShapeDtypeStruct((), jnp.int32),
        "action": jax.ShapeDtypeStruct((), jnp.int32),
        "schema": jax.ShapeDtypeStruct((), jnp.int32),
        "reward": jax.ShapeDtypeStruct((), jnp.float32),
        "terminated": jax.ShapeDtypeStruct((), jnp.bool_),
        "truncated": jax.ShapeDtypeStruct((), jnp.bool_),
        "next_state": jax.ShapeDtypeStruct((rw2,), jnp.uint32),
        "goal": jax.ShapeDtypeStruct((), jnp.bool_),
    }
    if labels:
        spec["binding"] = jax.ShapeDtypeStruct((env.label_width,), jnp.int32)
    if goals:
        w2 = 2 * env.words
        spec["goal_pos"] = jax.ShapeDtypeStruct((w2,), jnp.uint32)
        spec["goal_neg"] = jax.ShapeDtypeStruct((w2,), jnp.uint32)
    return spec


def transition_example(env: Env, *, labels: bool = True, goals: bool = False) -> dict[str, Any]:
    """A zero transition of :func:`transition_spec` (the example item of ``buffer.init``)."""
    return {k: jnp.zeros(s.shape, s.dtype) for k, s in transition_spec(env, labels=labels, goals=goals).items()}


def transition_item(
    obs: Observation,
    action: Any,
    step: tuple,
    *,
    labels: bool = True,
    goal_pos: Optional[Any] = None,
    goal_neg: Optional[Any] = None,
) -> dict[str, Any]:
    """The transitions [N] of one step: ``obs`` is the observation the action was taken in, ``action`` the int32 [N]
    successor slots (None: -1; the random policy's slots are ``env.random_actions(key, state)`` before the step),
    ``step`` the tuple ``env.step`` returned. The next state is ``info["final_state"]`` (the reached state, before the
    autoreset) when the environment reports it, else the new observation's state (equal without autoreset).
    ``goal_pos`` / ``goal_neg`` ([W2] or [N, W2]) add goal masks (the spec with goals=True; an environment with
    per-env goals has them in ``obs.goal_pos`` / ``obs.goal_neg``)."""
    _, obs2, reward, terminated, truncated, info = step
    n = obs.count.shape[0]
    act = jnp.full((n,), -1, jnp.int32) if action is None else jnp.asarray(action, jnp.int32)
    item: dict[str, Any] = {
        "state": obs.state,
        "task_id": obs.task_id,
        "action": act,
        "schema": info["schema"],
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "next_state": info.get("final_state", obs2.state),
        "goal": info["goal"],
    }
    if labels:
        if "binding" not in info:
            raise ValueError("mymyr: the environment reports no bindings (Env(..., labels=False)); pass labels=False")
        item["binding"] = info["binding"]
    if (goal_pos is None) != (goal_neg is None):
        raise ValueError("mymyr: pass both goal_pos and goal_neg, or neither")
    if goal_pos is not None:
        gp = jnp.asarray(goal_pos, jnp.uint32)
        gn = jnp.asarray(goal_neg, jnp.uint32)
        item["goal_pos"] = jnp.broadcast_to(gp, (n, gp.shape[-1]))
        item["goal_neg"] = jnp.broadcast_to(gn, (n, gn.shape[-1]))
    return item
