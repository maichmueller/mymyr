"""The XLA FFI targets of mymyr.rl.jax: registration and the typed ``jax.ffi.ffi_call``
wrappers.

Every call carries the environment's registry handle as the int64 attribute ``handle`` (FFI attributes are scalars; the
table and the configuration stay native), and the rows' instances as an int32 operand ``task_ids``. The environment state is updated in place: its arrays are both operands and
results (``input_output_aliases``), so a ``lax.scan`` carries them without copies. Batching (``jax.vmap``) broadcasts
the unbatched operands and flattens the leading dimensions into more environments (``vmap_method="broadcast_all"``);
the flat expansion loops instead (its capacity is per call).

Command buffers: on CUDA, the environment targets are compatible with XLA's command buffers (the FFI trait
``kCmdBufferCompatible``: XLA traces them into CUDA graphs, so a jitted ``lax.scan`` of steps runs as graphs), and
their initialize stage sizes the device environment's scratch for the call's rows before XLA runs or traces them. The
general path's refresh and step synchronize on the successor count: states without the cache columns call the targets
``mymyr_env_refresh_sync`` / ``mymyr_env_step_sync`` (the same handlers without the trait; on the CPU the same as the
others), which XLA runs outside command buffers.
"""

import threading
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mymyr._core import _rl_jax

__all__ = ["register", "registered", "TARGETS"]

TARGETS = (
    "mymyr_env_init",
    "mymyr_env_reset",
    "mymyr_env_refresh",
    "mymyr_env_refresh_sync",
    "mymyr_env_step",
    "mymyr_env_step_sync",
    "mymyr_expand",
    "mymyr_expand_flat",
)

_lock = threading.Lock()
_done = False


def register() -> None:
    """Registers the FFI targets with JAX (CPU, and CUDA in a CUDA build); idempotent and thread-safe."""
    global _done
    with _lock:
        if _done:
            return
        targets = _rl_jax.ffi_targets()
        if not targets:
            raise RuntimeError(
                "mymyr: this mymyr was built without the XLA FFI headers (jaxlib was not found when it was built); "
                "rebuild it in an environment with jaxlib to use mymyr.rl.jax"
            )
        platforms = {p.lower() for p in (d.platform for d in jax.devices())} | {"cpu"}
        for name, platform, stages, cmd_buffer in targets:
            if platform.lower() == "cuda" and not ({"gpu", "cuda"} & platforms):
                continue  # JAX without a CUDA backend: the CPU targets only
            jax.ffi.register_ffi_target(name, stages, platform=platform, **_traits(cmd_buffer))
        _done = True


def registered() -> bool:
    return _done


def _traits(cmd_buffer: bool) -> dict:
    """The registration's traits: XLA traces a command-buffer compatible target's calls into CUDA graphs (the handler
    declares kCmdBufferCompatible; JAX's registration needs it again)."""
    if not cmd_buffer:
        return {}
    from jax._src.lib import xla_client

    return {"traits": xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE}


def _u32(shape: tuple) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.uint32)


def _i32(shape: tuple) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.int32)


def _f32(shape: tuple) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def _bool(shape: tuple) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.bool_)


def env_init(handle: int, task_ids, rw2: int, c: int, v2: int, g2: int) -> tuple[Any, ...]:
    """states [n, rw2], steps, count [n], counts [n, c], views [n, v2], goal_pos / goal_neg [n, g2]: every row at the
    initial state of its instance task_ids[i]."""
    n = tuple(task_ids.shape)
    call = jax.ffi.ffi_call(
        "mymyr_env_init",
        (_u32((*n, rw2)), _i32(n), _i32(n), _u32((*n, c)), _u32((*n, v2)), _u32((*n, g2)), _u32((*n, g2))),
        vmap_method="broadcast_all",
    )
    return call(task_ids, handle=np.int64(handle))


def env_reset(handle: int, states, task_ids, steps, count, counts, views, goal_pos, goal_neg, mask, keep_goals: bool):
    """states, steps, count, counts, views, goal_pos, goal_neg after resetting the rows with mask[i]."""
    call = jax.ffi.ffi_call(
        "mymyr_env_reset",
        tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (states, steps, count, counts, views, goal_pos, goal_neg)),
        input_output_aliases={0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
        vmap_method="broadcast_all",
    )
    return call(
        states, task_ids, steps, count, counts, views, goal_pos, goal_neg, mask, handle=np.int64(handle),
        keep_goals=bool(keep_goals),
    )  # fmt: skip


def _fast(counts) -> bool:
    """Whether a state carries the fast path's cache (its calls are command-buffer compatible)."""
    return counts.shape[-1] > 0


def env_refresh(handle: int, states, task_ids, counts, views) -> tuple[Any, ...]:
    n = states.shape[:-1]
    call = jax.ffi.ffi_call(
        "mymyr_env_refresh" if _fast(counts) else "mymyr_env_refresh_sync",
        (_i32(n), jax.ShapeDtypeStruct(counts.shape, counts.dtype), jax.ShapeDtypeStruct(views.shape, views.dtype)),
        input_output_aliases={2: 1, 3: 2},
        vmap_method="broadcast_all",
    )
    return call(states, task_ids, counts, views, handle=np.int64(handle))


def env_step(
    handle: int, states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg, env_id, seed, action,
    next_task_ids, label_width: int, final: bool,
):  # fmt: skip
    """The 17 results of mymyr_env_step (see rl_jax_bindings.cpp)."""
    n = states.shape[:-1]
    same = tuple(
        jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg)
    )
    outs = same + (
        _f32(n),
        _bool(n),
        _bool(n),
        _bool(n),
        _bool(n),
        _i32(n),
        _i32((*n, label_width) if label_width else (0, 1)),
        _u32((*n, states.shape[-1]) if final else (0, states.shape[-1])),
    )
    call = jax.ffi.ffi_call(
        "mymyr_env_step" if _fast(counts) else "mymyr_env_step_sync",
        outs,
        input_output_aliases={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8},
        vmap_method="broadcast_all",
    )
    return call(
        states, task_ids, steps, draws, count, counts, views, goal_pos, goal_neg, env_id, seed, action, next_task_ids,
        handle=np.int64(handle),
    )  # fmt: skip


def expand(handle: int, states, task_ids, k: int, label_width: int) -> tuple[Any, ...]:
    n = states.shape[:-1]
    call = jax.ffi.ffi_call(
        "mymyr_expand",
        (
            _u32((*n, k, states.shape[-1])),
            _i32((*n, k)),
            _i32((*n, k, label_width)),
            _bool((*n, k)),
            _bool((*n, k)),
            _i32(n),
        ),
        vmap_method="broadcast_all",
    )
    return call(states, task_ids, handle=np.int64(handle))


def expand_flat(handle: int, states, task_ids, capacity: int, label_width: int) -> tuple[Any, ...]:
    n = states.shape[0]
    call = jax.ffi.ffi_call(
        "mymyr_expand_flat",
        (
            _u32((capacity, states.shape[-1])),
            _i32((capacity,)),
            _i32((capacity,)),
            _i32((capacity, label_width)),
            _bool((capacity,)),
            _i32((n + 1,)),
        ),
        vmap_method="sequential",
    )
    return call(states, task_ids, handle=np.int64(handle))
