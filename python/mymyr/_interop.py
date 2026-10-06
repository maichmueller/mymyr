"""Framework glue for mymyr's zero-copy exports.

mymyr owns the memory of its outputs and exports it as ``mymyr.DLArray`` objects: our own ``__dlpack__`` /
``__dlpack_device__`` (DLPack 1.x versioned or legacy capsules, the offset folded into the pointer). The consumer
framework wraps them without a copy. The native code calls :func:`from_dlpack` for torch and JAX outputs; NumPy views
are made natively.
"""

import sys

import numpy as np


def from_dlpack(x, framework):
    """Wrap a DLPack producer as an array of ``framework`` ('numpy', 'torch', 'jax' or 'dlpack') without copying."""
    if framework == "torch":
        torch = sys.modules.get("torch")
        if torch is None:
            import torch
        return torch.from_dlpack(x)
    if framework == "jax":
        dlpack = sys.modules.get("jax.dlpack")
        if dlpack is None:
            import jax.dlpack as dlpack
        return dlpack.from_dlpack(x)
    if framework == "numpy":
        return np.from_dlpack(x)
    return x


def framework_of(x):
    """'torch', 'jax' or 'numpy' (the default) for an array object."""
    module = type(x).__module__
    if module.startswith("torch"):
        return "torch"
    if module.startswith(("jax", "jaxlib")):
        return "jax"
    return "numpy"
