"""The counter-based RNG of the environments (rl/rng.hpp): Philox-4x32-10, the same on the host and the device.

A draw is a pure function of (seed, env id, draw counter, purpose): the successor of environment ``first_env + i`` at
its draw counter ``d`` is ``successor_indices(seed, first_env + i, d, count)``. No state is shared between
environments, so trajectories do not depend on the batch an environment runs in, the thread count, or the device's
launch configuration. These helpers compute draws on the host (tests, replays); the environments draw on their own
device.
"""

from typing import Union

import numpy as np
import torch

from mymyr._core import _rl_torch

__all__ = ["philox", "successor_indices"]

ArrayLike = Union[torch.Tensor, np.ndarray, int]


def _numpy(x: ArrayLike) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _host(a: np.ndarray, dtype: np.dtype, n: int) -> np.ndarray:
    if a.ndim == 0:
        a = np.full(n, a)
    return np.ascontiguousarray(a.astype(dtype, copy=False))


def philox(counters: ArrayLike, seed: int) -> torch.Tensor:
    """Philox-4x32-10 blocks: counters [n, 4] uint32 under the key (seed low, seed high) -> [n, 4] (as int64 values
    in [0, 2^32)), equal to cuRAND's ``curand_Philox4x32_10``."""
    c = counters.detach().cpu().numpy() if isinstance(counters, torch.Tensor) else np.asarray(counters)
    c = np.ascontiguousarray(c.astype(np.uint32)).reshape(-1, 4)
    return torch.from_numpy(np.asarray(_rl_torch.philox(c, int(seed))).astype(np.int64))


def successor_indices(seed: int, envs: ArrayLike, draws: ArrayLike, counts: ArrayLike) -> torch.Tensor:
    """The random policy's choice for env ids ``envs``, draw counters ``draws`` and successor counts ``counts``
    (broadcast from scalars): a uniform index in [0, count), -1 where count is 0. Returns int64 [n] on the CPU."""
    arrays = [_numpy(x) for x in (envs, draws, counts)]
    n = max(a.size for a in arrays)
    e = _host(arrays[0], np.int64, n)
    d = _host(arrays[1], np.int64, n)
    c = _host(arrays[2], np.int32, n)
    return torch.from_numpy(np.asarray(_rl_torch.successor_indices(int(seed), e, d, c)))
