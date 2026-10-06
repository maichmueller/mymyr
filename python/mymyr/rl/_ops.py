"""Novelty rewards, prefix masks and hindsight relabels in the inputs' framework.

NumPy arrays and torch tensors (CPU or CUDA) run the native helpers (mymyr._core._rl_ops: rl/novelty.hpp,
rl/prefix.hpp, rl/her.hpp; CUDA tensors on torch's current stream, without a host synchronization); JAX arrays run the
jnp versions of :mod:`mymyr.rl.jax` (pure functions, jit-able). All of them give the same values.
"""

import sys
import types
from typing import Any, NamedTuple, Optional, Union

import numpy as np

from mymyr._core import _rl_ops

__all__ = ["HerBatch", "her_relabel", "novelty_update", "prefix_masks", "schema_masks"]


class HerBatch(NamedTuple):
    """k relabels of every transition [T, N]: ``goal`` [T, N, k, W] (the atoms that must hold, as state words),
    ``source`` int32 [T, N, k] (the step whose reached state the goal came from), ``achieved`` bool [T, N, k] (the
    transition's reached state satisfies the goal: terminated under it), ``reward`` float32 [T, N, k] (step_reward,
    plus goal_reward where achieved)."""

    goal: Any
    source: Any
    achieved: Any
    reward: Any


def _framework(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return "numpy"
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return "torch"
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(x, jax.Array):
        return "jax"
    raise TypeError(f"mymyr: expected a NumPy array, a torch tensor or a JAX array, not {type(x).__name__}")


_NP = {"int32": np.int32, "bool": np.bool_, "float32": np.float32, "uint64": np.uint64, "uint32": np.uint32}


def _zeros(fw: str, like: Any, shape: tuple, dtype: str) -> Any:
    if fw == "torch":
        import torch

        tdt = {"int32": torch.int32, "bool": torch.bool, "float32": torch.float32, "uint64": torch.int64,
               "uint32": torch.int32}[dtype]
        return torch.zeros(shape, dtype=tdt, device=like.device)
    return np.zeros(shape, _NP[dtype])


def _stream(fw: str, like: Any) -> Optional[int]:
    if fw == "torch" and like.is_cuda:
        import torch

        return torch.cuda.current_stream(like.device).cuda_stream
    return None


def _word_dtype(x: Any) -> str:
    """'uint64' for 64-bit state words, 'uint32' for 32-bit halves."""
    return "uint64" if x.dtype.itemsize == 8 else "uint32"


def novelty_update(seen: Any, states: Any, width: int = 1) -> tuple[Any, Any]:
    """Width-1 novelty rewards: ``(r_int, seen)`` with r_int int32 [N] = the atoms of each state not
    yet in its environment's table ``seen`` [N, S] (state words; the first S words of states [N, >= S] are read), and
    seen |= states. NumPy / torch: ``seen`` is updated in place (and returned); JAX: a new table is returned. r_int > 0
    is the verdict of a width-1 novelty table fed each environment's states in order."""
    if int(width) != 1:
        raise ValueError("mymyr: novelty_update supports width 1 (per-environment atom tables) only")
    fw = _framework(seen)
    if fw == "jax":
        from mymyr.rl.jax._ops import novelty_update as jnp_update

        return jnp_update(seen, states)
    reward = _zeros(fw, seen, (seen.shape[0],), "int32")
    _rl_ops.novelty_update(seen, states, reward, stream=_stream(fw, seen))
    return reward, seen


def _labels(exp: Any) -> tuple[Any, Any, Any, int]:
    """(parent [R], schema [R], binding [R, L], N) of a flat or padded expansion (padding: parent -1)."""
    if hasattr(exp, "parent"):
        n = getattr(exp, "num_states", None)
        if n is None:
            n = exp.offsets.shape[0] - 1
        return exp.parent, exp.schema, exp.binding, int(n)
    mask = exp.mask
    n, k = mask.shape[0], mask.shape[1]
    fw = _framework(mask)
    if fw == "torch":
        import torch

        rows = torch.arange(n, dtype=torch.int32, device=mask.device)[:, None].expand(n, k)
        parent = torch.where(mask, rows, torch.full_like(rows, -1)).reshape(-1).contiguous()
    elif fw == "jax":
        import jax.numpy as jnp

        parent = jnp.where(mask, jnp.arange(n, dtype=jnp.int32)[:, None], -1).reshape(-1)
    else:
        parent = np.where(mask, np.arange(n, dtype=np.int32)[:, None], -1).reshape(-1).astype(np.int32)
    return parent, exp.schema.reshape(n * k), exp.binding.reshape(n * k, exp.binding.shape[-1]), n


def prefix_masks(
    exp: Any, schema: Any, prefix: Optional[Any], depth: int, *, num_objects: Optional[int] = None
) -> Any:
    """The factored action mask of parameter ``depth``: bool [N, num_objects], object o set for
    state i iff one of its successors is labelled (schema[i], prefix[i, :depth], o, ...). ``exp`` is the expansion of
    the N states (rl.expand's Expansion or padded view, a mymyr.cuda.DeviceExpansion, or mymyr.rl.jax's
    FlatExpansion / PaddedExpansion); ``schema`` int32 [N] the chosen schemas; ``prefix`` int32 [N, >= depth] the
    chosen objects of the first ``depth`` parameters (None for depth 0). The mask is in the framework and on the device
    of the expansion. ``num_objects`` defaults to the expansion's task (``exp.num_objects``)."""
    parent, sch, binding, n = _labels(exp)
    if num_objects is None:
        num_objects = getattr(exp, "num_objects", None)
        if num_objects is None:
            raise ValueError("mymyr: pass num_objects (task.num_objects): this expansion does not know its task")
    fw = _framework(sch)
    d = int(depth)
    if fw == "jax":
        from mymyr.rl.jax._ops import prefix_masks as jnp_masks

        view = types.SimpleNamespace(parent=parent, schema=sch, binding=binding)
        return jnp_masks(view, schema, prefix, d, int(num_objects))
    mask = _zeros(fw, sch, (int(schema.shape[0]), int(num_objects)), "bool")
    _rl_ops.prefix_masks(parent, sch, binding, schema, prefix if d > 0 else None, d, mask, stream=_stream(fw, sch))
    return mask


def schema_masks(exp: Any, *, num_schemas: Optional[int] = None) -> Any:
    """The first factored decision: bool [N, num_schemas], schema s set for state i iff a successor of i is labelled
    with s (``num_schemas`` defaults to the expansion's task)."""
    parent, sch, _, n = _labels(exp)
    if num_schemas is None:
        num_schemas = getattr(exp, "num_schemas", None)
        if num_schemas is None:
            raise ValueError("mymyr: pass num_schemas (task.num_schemas): this expansion does not know its task")
    fw = _framework(sch)
    if fw == "jax":
        from mymyr.rl.jax._ops import schema_masks as jnp_masks

        view = types.SimpleNamespace(parent=parent, schema=sch, binding=None)
        return jnp_masks(view, n, int(num_schemas))
    mask = _zeros(fw, sch, (n, int(num_schemas)), "bool")
    _rl_ops.schema_masks(parent, sch, mask, stream=_stream(fw, sch))
    return mask


def her_relabel(
    states: Any,
    done: Any,
    *,
    strategy: str = "future",
    k: int = 4,
    subset: Optional[int] = None,
    seed: Union[int, Any] = 0,
    words: Optional[int] = None,
    goal_atoms: Optional[Any] = None,
    env_ids: Optional[Any] = None,
    first_env: int = 0,
    step_reward: float = -1.0,
    goal_reward: float = 0.0,
) -> HerBatch:
    """Hindsight relabels with atom-set goals (Andrychowicz et al. 2017) of a trajectory window:
    ``states`` [T, N, RW] the states the steps reached (``final_states`` / ``info["final_state"]``: before an autoreset),
    ``done`` bool [T, N] (the step ended its episode: terminated or truncated).

    For every transition (t, i) and j < k a source step t' of the same episode is drawn (``"future"``: uniform in
    [t, end]; ``"episode"``: uniform in [start, end]; ``"final"``: the episode's last step in the window), and the goal is
    the atoms of states[t', i] (restricted to ``goal_atoms`` [W]), or ``subset`` of them drawn without replacement. The
    draws are the counter-based RNG's (rl/her.hpp: seed, env id = env_ids[i] or first_env + i, purpose k_her), so a
    relabel does not depend on the batch or the device. ``words`` limits the goals to the first W words of a row (the
    atom words of a numeric task's rows). NumPy / torch (CPU or CUDA) run the native helper; JAX arrays the jnp
    version (``seed`` may then be a JAX key)."""
    fw = _framework(states)
    kk = int(k)
    m = 0 if subset is None else int(subset)
    if kk < 1 or m < 0:
        raise ValueError("mymyr: k must be >= 1 and subset >= 0")
    if fw == "jax":
        from mymyr.rl.jax._ops import her_relabel as jnp_relabel

        return jnp_relabel(
            states, done, seed, strategy=strategy, k=kk, subset=subset, goal_atoms=goal_atoms, env_ids=env_ids,
            first_env=first_env, step_reward=step_reward, goal_reward=goal_reward, words=words,
        )
    t_len, n, cols = (int(x) for x in states.shape)
    half = states.dtype.itemsize == 4
    w = (cols // 2 if half else cols) if words is None else int(words)
    goal = _zeros(fw, states, (t_len, n, kk, 2 * w if half else w), _word_dtype(states))
    source = _zeros(fw, states, (t_len, n, kk), "int32")
    achieved = _zeros(fw, states, (t_len, n, kk), "bool")
    reward = _zeros(fw, states, (t_len, n, kk), "float32")
    _rl_ops.her_relabel(
        states, done, goal, source, achieved, reward, strategy=strategy, subset=m, seed=int(seed),
        first_env=int(first_env), env_ids=env_ids, goal_atoms=goal_atoms, step_reward=float(step_reward),
        goal_reward=float(goal_reward), stream=_stream(fw, states),
    )
    return HerBatch(goal, source, achieved, reward)
