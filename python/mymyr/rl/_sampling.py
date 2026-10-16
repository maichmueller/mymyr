"""Pure, counter-based instance sampling in NumPy, PyTorch and JAX."""

import sys
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["InstanceSampler"]


class InstanceSampler:
    """A uniform or weighted distribution over ``num_instances`` instances.

    ``sample(seed, env_ids, reset_counts)`` returns int32 ids in the arrays' framework and on their device. A draw
    depends only on the 64-bit seed, the environment id and its uint32 reset counter, not on batch order or the
    action RNG. Zero-weight instances are never selected. The distribution is immutable after construction;
    dynamic curricula override the next instance ids in an environment's reset or step.
    """

    def __init__(self, num_instances: int, weights: Optional[Sequence[float]] = None) -> None:
        self._num_instances = int(num_instances)
        if self.num_instances < 1 or self.num_instances > 0x7FFFFFFF:
            raise ValueError("mymyr: instance sampling needs between 1 and 2**31 - 1 instances")
        self._weights: Optional[tuple[float, ...]] = None
        self._indices = np.empty(0, np.int32)
        self._thresholds = np.empty(0, np.uint32)
        if weights is not None:
            values = np.asarray(weights, np.float64)
            if values.shape != (self.num_instances,):
                raise ValueError("mymyr: sampling weights need one value per instance")
            if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
                raise ValueError("mymyr: sampling weights must be finite, nonnegative and have a positive sum")
            self._weights = tuple(float(value) for value in values)
            self._indices = np.flatnonzero(values > 0).astype(np.int32)
            positive = values[self._indices] / values.max()
            cumulative = np.cumsum(positive / positive.sum())[:-1]
            self._thresholds = np.minimum(np.floor(cumulative * 2**32), 0xFFFFFFFF).astype(np.uint32)

    @property
    def num_instances(self) -> int:
        """The number of instances it draws from (ids 0 .. num_instances - 1)."""
        return self._num_instances

    @property
    def weights(self) -> Optional[tuple[float, ...]]:
        """The weight of each instance, or None for the uniform distribution."""
        return self._weights

    def _arrays(self, like: Any) -> tuple[Any, Any]:
        framework = _framework(like)
        if framework == "torch":
            import torch

            return (
                torch.as_tensor(self._thresholds.astype(np.int64), device=like.device),
                torch.as_tensor(self._indices, device=like.device),
            )
        if framework == "jax":
            import jax.numpy as jnp

            return jnp.asarray(self._thresholds), jnp.asarray(self._indices)
        return self._thresholds.astype(np.uint64), self._indices

    def sample(
        self, seed: Any, env_ids: Any, reset_counts: Any, *, _arrays: Optional[tuple[Any, Any]] = None
    ) -> Any:
        """Draw ids for the broadcast environment ids and reset counters; ``seed`` is an integer or [..., 2]
        uint32 halves (low, high). The counters are caller-owned: increment only the environments being reset.
        Pure array operations, including under ``torch.compile``, ``jax.jit``, ``vmap`` and ``scan``."""
        framework = _framework(env_ids)
        if isinstance(seed, (int, np.integer)):
            value = int(seed) & 0xFFFFFFFFFFFFFFFF
            seed = [value & 0xFFFFFFFF, value >> 32]
        if framework == "torch":
            import torch

            seed = torch.as_tensor(seed, dtype=torch.int64, device=env_ids.device)
            streams = env_ids.to(torch.int64)
            counters = torch.as_tensor(reset_counts, dtype=torch.int64, device=env_ids.device)
        elif framework == "jax":
            import jax.numpy as jnp

            seed = jnp.asarray(seed, jnp.uint32)
            streams = jnp.asarray(env_ids, jnp.uint32)
            counters = jnp.asarray(reset_counts, jnp.uint32)
        else:
            seed = np.asarray(seed, np.uint64)
            streams = np.asarray(env_ids, np.uint64)
            counters = np.asarray(reset_counts, np.uint64)
        value = _mix(seed[..., 0] ^ _mix(seed[..., 1] ^ 0x53414D50))
        purpose = np.uint32(0x9E3779B9) if framework == "jax" else 0x9E3779B9
        bits = _mix(value ^ _mix(streams) ^ _mix(counters ^ purpose))
        if self.weights is None:
            if framework == "jax":
                from .jax.rng import mulhilo32

                return mulhilo32(bits, self.num_instances)[0].astype("int32")
            result = ((bits * self.num_instances) >> 32) & 0xFFFFFFFF
        else:
            thresholds, indices = self._arrays(env_ids) if _arrays is None else _arrays
            result = indices[(bits[..., None] >= thresholds).sum(-1)]
        return result.to(dtype=torch.int32) if framework == "torch" else result.astype("int32")


def _mix(value: Any) -> Any:
    if _framework(value) == "jax":
        mask, first, second = np.uint32(0xFFFFFFFF), np.uint32(0x85EBCA6B), np.uint32(0xC2B2AE35)
    else:
        mask, first, second = 0xFFFFFFFF, 0x85EBCA6B, 0xC2B2AE35
    value = value & mask
    value = ((value ^ (value >> 16)) * first) & mask
    value = ((value ^ (value >> 13)) * second) & mask
    return (value ^ (value >> 16)) & mask


def _framework(value: Any) -> str:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return "torch"
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(value, (jax.Array, jax.core.Tracer)):
        return "jax"
    return "numpy"


def _instance_sampler(value: Any, num_instances: int, default_uniform: bool = True) -> Optional[InstanceSampler]:
    if isinstance(value, str):
        if value not in ("auto", "uniform"):
            raise ValueError("mymyr: sampler must be 'auto', 'uniform', an InstanceSampler or None")
        value = InstanceSampler(num_instances) if value == "uniform" or default_uniform else None
    if value is not None:
        if not isinstance(value, InstanceSampler):
            raise TypeError("mymyr: sampler must be 'auto', 'uniform', an InstanceSampler or None")
        if value.num_instances != num_instances:
            raise ValueError("mymyr: sampler instance count does not match the environment's table or suite")
    return None if num_instances == 1 else value
