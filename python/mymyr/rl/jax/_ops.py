"""Prefix masks, novelty rewards and HER relabels as jitted jnp functions.

Each equals its native counterpart in :mod:`mymyr.rl` (``rl::prefix_masks``, ``rl::novelty_update``,
``rl::her_relabel``; include/mymyr/rl/ops.hpp) bit for bit; they are pure functions of JAX arrays (state words as
uint32 halves), so they run inside ``jax.jit``, ``jax.vmap`` and ``lax.scan`` on any JAX device.
"""

import functools
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp

from mymyr.rl._ops import HerBatch

from . import rng
from ._env import FlatExpansion, PaddedExpansion, key_seed

__all__ = ["HerBatch", "her_relabel", "novelty_update", "prefix_masks", "schema_masks"]

_STRATEGIES = {"future": 0, "final": 1, "episode": 2}


# ------------------------------------------------------------------------------------------------ prefix masks


def _labels(exp: Union[FlatExpansion, PaddedExpansion], n: int) -> tuple[Any, Any, Any]:
    """(row [R], schema [R], binding [R, L]) of the expansion's successors; row -1 for padding."""
    if isinstance(exp, PaddedExpansion):
        k = exp.schema.shape[1]
        row = jnp.where(exp.mask, jnp.arange(n, dtype=jnp.int32)[:, None], -1).reshape(-1)
        return row, exp.schema.reshape(-1), exp.binding.reshape(n * k, -1)
    return exp.parent, exp.schema, exp.binding


def prefix_masks(
    exp: Union[FlatExpansion, PaddedExpansion], schema: Any, prefix: Any, depth: int, num_objects: int
) -> Any:
    """The factored action mask of parameter ``depth``: bool [N, num_objects], object o set for
    state i iff one of its successors is labelled (schema[i], prefix[i, :depth], o, ...). ``exp`` is the expansion of
    the N states (flat or padded, from :meth:`Env.expand_flat` / :meth:`Env.expand`); ``schema`` int32 [N] the chosen
    schemas, ``prefix`` int32 [N, P >= depth] the chosen objects of the first ``depth`` parameters (the rest is not
    read). A state whose prefix no successor extends has an empty mask. See :func:`schema_masks` for the first
    decision."""
    schema = jnp.asarray(schema, jnp.int32)
    n = schema.shape[0]
    d = int(depth)
    o = int(num_objects)
    row, sch, binding = _labels(exp, n)
    valid = (row >= 0) & (row < n)
    r = jnp.where(valid, row, 0)
    match = valid & (sch == schema[r])
    if d > 0:
        prefix = jnp.asarray(prefix, jnp.int32)
        match &= jnp.all(binding[:, :d] == prefix[r, :d], axis=1)
    obj = binding[:, d] if d < binding.shape[1] else jnp.full(row.shape, -1, jnp.int32)
    match &= (obj >= 0) & (obj < o)
    return _scatter_or(n, o, jnp.where(match, r, 0), jnp.where(match, obj, 0), match)


def schema_masks(exp: Union[FlatExpansion, PaddedExpansion], num_states: int, num_schemas: int) -> Any:
    """The first factored decision: bool [N, num_schemas], schema s set for state i iff some successor of i is labelled
    with s."""
    n = int(num_states)
    s = int(num_schemas)
    row, sch, _ = _labels(exp, n)
    ok = (row >= 0) & (row < n) & (sch >= 0) & (sch < s)
    return _scatter_or(n, s, jnp.where(ok, row, 0), jnp.where(ok, sch, 0), ok)


def _scatter_or(n: int, m: int, rows: Any, cols: Any, flags: Any) -> Any:
    """bool [n, m] with (rows[j], cols[j]) set iff some flags[j]; every index is in range (a scatter-max of the
    flags: entries without a flag write 0 at (0, 0)), so no index is ever dropped or clamped."""
    mask = jnp.zeros((n, m), jnp.uint8)
    if n == 0 or m == 0 or rows.shape[0] == 0:
        return mask.astype(jnp.bool_)
    return mask.at[rows, cols].max(flags.astype(jnp.uint8), mode="promise_in_bounds").astype(jnp.bool_)


# ------------------------------------------------------------------------------------------------ novelty


def novelty_update(seen: Any, states: Any, width: int = 1) -> tuple[Any, Any]:
    """Width-1 novelty rewards: ``(r_int, seen')`` with r_int int32 [N] = the atoms of states [N, *]
    not yet in the per-environment table seen [N, S] (uint32 halves of the atom words; the first S columns of the
    states are read), and seen' = seen | states. r_int > 0 is the verdict of a width-1 novelty table
    (``NoveltyTable::mark_state``) fed each environment's states in order."""
    if int(width) != 1:
        raise ValueError("mymyr: novelty_update supports width 1 (per-environment atom tables) only")
    seen = jnp.asarray(seen, jnp.uint32)
    s = jnp.asarray(states, jnp.uint32)[..., : seen.shape[-1]]
    new = s & ~seen
    return jax.lax.population_count(new).sum(-1).astype(jnp.int32), seen | s


# ------------------------------------------------------------------------------------------------ HER


def _episode_bounds(done: Any) -> tuple[Any, Any]:
    """(start [T, N], end [T, N]) of the episode of each step: the step after the previous done (0), and the next done
    step at or after t (T - 1 when the episode continues past the window)."""
    t_len = done.shape[0]
    t = jnp.arange(t_len, dtype=jnp.int32)[:, None]
    prev = jnp.where(done, t + 1, 0)
    start = jax.lax.cummax(jnp.concatenate([jnp.zeros_like(prev[:1]), prev[:-1]], axis=0), axis=0)
    nxt = jnp.where(done, t, t_len - 1)
    end = jax.lax.cummin(nxt, axis=0, reverse=True)
    return start, end


def _kth_bit(c: Any, r: Any) -> tuple[Any, Any]:
    """(word, bit) of the r-th set bit (ascending) of the uint32 words c [..., W2]; r < popcount(c)."""
    pc = jax.lax.population_count(c).astype(jnp.int32)
    cum = jnp.cumsum(pc, axis=-1)
    w = jnp.sum(cum <= r[..., None], axis=-1).astype(jnp.int32)
    w = jnp.minimum(w, c.shape[-1] - 1)
    before = jnp.take_along_axis(cum - pc, w[..., None], axis=-1)[..., 0]
    x = jnp.take_along_axis(c, w[..., None], axis=-1)[..., 0]
    rr = r - before
    bits = (x[..., None] >> jnp.arange(32, dtype=jnp.uint32)) & 1
    cb = jnp.cumsum(bits.astype(jnp.int32), axis=-1)
    b = jnp.sum(cb <= rr[..., None], axis=-1).astype(jnp.int32)
    return w, b


def her_relabel(
    states: Any,
    done: Any,
    key: Any,
    *,
    strategy: str = "future",
    k: int = 4,
    subset: Optional[int] = None,
    goal_atoms: Optional[Any] = None,
    env_ids: Optional[Any] = None,
    first_env: int = 0,
    step_reward: float = -1.0,
    goal_reward: float = 0.0,
    words: Optional[int] = None,
) -> HerBatch:
    """Hindsight relabels with atom-set goals (Andrychowicz et al. 2017), equal to
    ``mymyr.rl.her_relabel``: ``states`` uint32 [T, N, W2] are the states the steps reached (``info["final_state"]``),
    ``done`` bool [T, N] ends an episode after step t (terminated or truncated). For every transition (t, i) and
    j < k a source step t' of the same episode is drawn: ``"future"`` uniform in [t, end], ``"episode"`` uniform in
    [start, end], ``"final"`` the episode's last step in the window; the goal is the atoms of states[t', i] (restricted
    to ``goal_atoms`` [W2], or [N, W2]: env i's own, if given), or ``subset`` of them drawn without replacement. Draws are Philox blocks of
    (seed = key_seed(key); draw, env id, purpose rng.PURPOSE_HER) with draw = (t k + j)(1 + subset) + d, d = 0 for the
    step and d = 1.. for the atoms (env id: env_ids[i], else first_env + i), so relabels do not depend on the batch.
    Episodes that continue past the window draw from the window's steps only. ``words``: the goals cover the first
    W words of a row (2W uint32 halves; default: the whole row)."""
    if strategy not in _STRATEGIES:
        raise ValueError("mymyr: strategy must be 'future', 'final' or 'episode'")
    states = jnp.asarray(states, jnp.uint32)
    if words is not None:
        states = states[..., : 2 * int(words)]
    done = jnp.asarray(done, jnp.bool_)
    t_len, n, w2 = states.shape
    kk = int(k)
    m = 0 if subset is None else int(subset)
    if m < 0 or kk < 1:
        raise ValueError("mymyr: k must be >= 1 and subset >= 0")
    if t_len * kk * (1 + m) >= 1 << 32:
        raise ValueError("mymyr: T * k * (1 + subset) must stay below 2^32 draws")
    seed = key_seed(key)
    ids = (
        jnp.arange(first_env, first_env + n, dtype=jnp.uint32)
        if env_ids is None
        else jnp.asarray(env_ids, jnp.uint32)
    )
    atoms = None if goal_atoms is None else jnp.asarray(goal_atoms, jnp.uint32)
    return _her(states, done, seed, ids, atoms, _STRATEGIES[strategy], kk, m, float(step_reward), float(goal_reward))


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def _her(states, done, seed, ids, atoms, strategy, kk, m, step_reward, goal_reward):
    t_len, n, w2 = states.shape
    start, end = _episode_bounds(done)
    t = jnp.arange(t_len, dtype=jnp.uint32)[:, None, None]
    j = jnp.arange(kk, dtype=jnp.uint32)[None, None, :]
    base = (t * jnp.uint32(kk) + j) * jnp.uint32(1 + m)  # [T, 1, k]
    env = ids[None, :, None]

    def draw(d):
        c = base + jnp.uint32(d)
        c = jnp.broadcast_to(c, (t_len, n, kk))
        lo, hi = rng.bits64(seed, env, jnp.stack([c, jnp.zeros_like(c)], axis=-1), rng.PURPOSE_HER)
        return lo, hi

    s0 = start[:, :, None]
    e0 = end[:, :, None]
    tt = jnp.arange(t_len, dtype=jnp.int32)[:, None, None]
    lo_t = {0: tt, 1: e0, 2: s0}[strategy]
    lo_t = jnp.broadcast_to(lo_t, (t_len, n, kk))
    span = (jnp.broadcast_to(e0, (t_len, n, kk)) - lo_t + 1).astype(jnp.uint32)
    blo, bhi = draw(0)
    source = lo_t + rng.below(blo, bhi, span).astype(jnp.int32)

    cand = states[source, jnp.arange(n)[None, :, None]]  # [T, N, k, W2]
    if atoms is not None:
        cand = cand & (atoms[None, :, None, :] if atoms.ndim == 2 else atoms)
    if m == 0:
        goal = cand
    else:

        def pick(d, carry):
            c, g = carry
            total = jax.lax.population_count(c).astype(jnp.int32).sum(-1)
            lo, hi = draw(d)
            r = rng.below(lo, hi, jnp.maximum(total, 1).astype(jnp.uint32)).astype(jnp.int32)
            w, b = _kth_bit(c, r)
            bit = (jnp.uint32(1) << b.astype(jnp.uint32))[..., None] * (jnp.arange(w2) == w[..., None]).astype(jnp.uint32)
            bit = jnp.where((total > 0)[..., None], bit, jnp.uint32(0))
            return c & ~bit, g | bit

        _, goal = jax.lax.fori_loop(1, m + 1, pick, (cand, jnp.zeros_like(cand)))
    reached = states[:, :, None, :]
    achieved = jnp.all((reached & goal) == goal, axis=-1)
    reward = jnp.float32(step_reward) + jnp.where(achieved, jnp.float32(goal_reward), jnp.float32(0.0))
    return HerBatch(goal, source, achieved, reward)
