"""The CUDA backend: device contexts, device task uploads, device arenas, DLPack for
device memory, the device BrFS, the device path of ``rl.expand`` / ``rl.expand_into``, device IW, heuristics and
searches, and device state spaces.

Only in CUDA builds of mymyr (``-C cmake.define.MYMYR_CUDA=ON``); importing this module otherwise raises ImportError.

- :class:`Context`: a user-owned device context (device, its own stream-ordered memory pool capped by ``max_bytes``,
  a compute stream and a copy stream). No global state: device tasks and arrays keep their context alive.
- :class:`DeviceTask`: ``task.device_arrays()`` (version 2) uploaded once; shareable across streams and threads.
  ``arrays(framework)`` exposes the uploaded arrays zero-copy; ``download()`` is the round-trip check;
  ``validate``, ``applicable``, ``apply`` and ``goal`` are smoke-test kernels.
- :class:`StateArena`: an append-only device array of state rows with a pinned host mirror kept in sync by tail
  copies on the copy stream.
- :func:`to_memory`: a copy of a CPU array in device, managed or pinned-host memory.
- :func:`brfs`: the layer-synchronous device BrFS. Its ids equal the CPU BrFS's deterministic ids
  (``mymyr.search.brfs`` at any thread count; ``fingerprint=True`` hashes them), whatever the chunk size; the result's
  ``state_words()`` / ``nodes()`` are the state space on the device, ``stats`` the per-phase device times.
- ``mymyr.rl.expand`` / ``mymyr.rl.expand_into`` take CUDA device arrays (torch CUDA tensors, JAX GPU arrays,
  DLArrays): the expansion runs on the device, byte-equal to the CPU's, and the outputs (a :class:`DeviceExpansion`,
  or the caller's device destinations) are device arrays in the input's framework. ``stream=`` picks the stream (None:
  the context's), ``ctx=`` the context (None: the task's default context on the input's device). Host inputs take the
  CPU path unchanged.
- :func:`multi_iw`, :func:`rollouts`, :func:`batched_iw1`: many IW searches at once on the device. Search i of
  ``multi_iw`` equals ``mymyr.search.iw`` from ``starts[i]`` (status, plan, goal state, per-pass counts) whatever the
  group and chunk sizes (exact batch novelty); ``rollouts`` are the CPU's randomized rollouts of the same seeds,
  reached atoms included; ``batched_iw1`` runs optimized IW(1) from host states or from CUDA word arrays read in
  place. All return an :class:`IwBatch`.
  Goals may be ground conjunctions of fluent/derived literals and numeric constraints, one per start; rollouts share
  one conjunction across their seeds. Callable goals and any-of alternatives within one search require CPU search.
- :class:`Heuristic`, :func:`astar`, :func:`gbfs`: batched grounded h_max, h_add, h_FF, h² and set-additive on the device
  (``Heuristic.evaluate``: host states give NumPy float64 values, CUDA word arrays uint32 device values in the input's
  framework, ``Heuristic.DEAD_END`` for dead ends; h_max, h_add and h² equal ``mymyr.search.Heuristic``'s, h_FF and set-additive break
  supporter ties by BFS level and operator id, ``Heuristic.reference`` is that rule on the CPU), and A* / GBFS with
  batched expansion (a :class:`DeviceSearchResult`: with 'max', 'h2' or 'blind' the plan cost of ``mymyr.search.astar``).
- :func:`state_space`, :func:`state_spaces`, :func:`generate_state_space`, :func:`generate_state_spaces`:
  ``mymyr.datasets``' state spaces generated on the device, for one task or for every instance of a
  ``mymyr.rl.TaskTable`` in one pipeline (waves bound the device memory). A :class:`DeviceStateSpace` holds the
  arrays of ``mymyr.datasets.StateSpace`` (the same ids, transitions, reverse CSR, goal distances, flags) in device
  memory: ``arrays(framework)`` exports them zero-copy, ``to_host()`` is the ``StateSpace``, and
  ``mymyr.datasets.StateSpaceSampler`` samples it. ``mymyr.datasets.state_space(task, device=0)`` dispatches here.
  ``multi_iw`` and ``batched_iw1`` take a TaskTable with ``task_ids`` (search i on instance ``task_ids[i]``).
- Over a ``mymyr.rl.TaskSuite`` (several domains, global task ids) ``multi_iw`` and ``batched_iw1`` run each
  domain's searches on its table and return a :class:`SuiteIwBatch` (an IwBatch per domain, indexed in search order),
  and ``state_spaces`` / ``generate_state_spaces`` each domain's instances in one pipeline, in global order.

Arrays cross the boundary through DLPack without copies, in both directions, with the array API's stream semantics:
device outputs are produced on the operation's stream (``stream=None``: the context's stream; an int is a
``cudaStream_t``, 1 / 2 the legacy / per-thread default stream; ``torch.cuda.Stream`` objects are accepted), and a
consumer's ``__dlpack__(stream=s)`` makes ``s`` wait for them; device inputs are imported with
``__dlpack__(stream=<the operation's stream>)`` and read in place.

The other host operations (``rl.is_goal``, ``rl.goal_test``, ...) stay on the CPU: they read pinned-host and managed
CUDA memory in place and reject device memory with a TypeError.
"""

import functools
from typing import Any

from mymyr import _suite
from mymyr._core import Task, TaskHandle
from mymyr._core._rl import TaskSuite, TaskTable

try:
    # CPU builds have no mymyr._core._cuda, nor its stub
    from mymyr._core import _cuda as _impl  # type: ignore[attr-defined, unused-ignore]
except ImportError as e:  # pragma: no cover - depends on the build
    raise ImportError(
        "mymyr.cuda: this mymyr was built without the CUDA backend (rebuild with -C cmake.define.MYMYR_CUDA=ON)"
    ) from e

Context = _impl.Context
DeviceTask = _impl.DeviceTask
StateArena = _impl.StateArena
BrfsResult = _impl.BrfsResult
DeviceExpansion = _impl.DeviceExpansion
DevicePaddedExpansion = _impl.DevicePaddedExpansion
brfs = _impl.brfs
to_memory = _impl.to_memory
available = _impl.available
device_count = _impl.device_count
IwBatch = _impl.IwBatch
SuiteIwBatch = _suite.SuiteIwBatch
rollouts = _impl.rollouts
Heuristic = _impl.Heuristic
DeviceSearchResult = _impl.DeviceSearchResult
astar = _impl.astar
gbfs = _impl.gbfs
DeviceStateSpace = _impl.DeviceStateSpace
DeviceGenerationResult = _impl.DeviceGenerationResult
generate_state_space = _impl.generate_state_space
state_space = _impl.state_space

Tasks = TaskSuite | TaskTable | Task | TaskHandle


# Over a TaskSuite the four below split by domain (mymyr._suite); a table or a task goes straight to the binding,
# whose signature and docstring they carry.
@functools.wraps(_impl.multi_iw)
def multi_iw(task: Tasks, starts: Any, **kwargs: Any) -> IwBatch | SuiteIwBatch:
    return _suite.multi_iw(_impl.multi_iw, task, starts, **kwargs)


@functools.wraps(_impl.batched_iw1)
def batched_iw1(task: Tasks, starts: Any, **kwargs: Any) -> IwBatch | SuiteIwBatch:
    return _suite.batched_iw1(_impl.batched_iw1, task, starts, **kwargs)


@functools.wraps(_impl.state_spaces)
def state_spaces(table: Tasks, **kwargs: Any) -> list[DeviceStateSpace | None]:
    return _suite.state_spaces(_impl.state_spaces, table, **kwargs)


@functools.wraps(_impl.generate_state_spaces)
def generate_state_spaces(table: Tasks, **kwargs: Any) -> list[DeviceGenerationResult]:
    return _suite.generate_state_spaces(_impl.generate_state_spaces, table, **kwargs)


__all__ = [
    "BrfsResult",
    "Context",
    "DeviceExpansion",
    "DeviceGenerationResult",
    "DevicePaddedExpansion",
    "DeviceSearchResult",
    "DeviceStateSpace",
    "DeviceTask",
    "Heuristic",
    "IwBatch",
    "StateArena",
    "SuiteIwBatch",
    "astar",
    "available",
    "batched_iw1",
    "brfs",
    "device_count",
    "gbfs",
    "generate_state_space",
    "generate_state_spaces",
    "multi_iw",
    "rollouts",
    "state_space",
    "state_spaces",
    "to_memory",
]
