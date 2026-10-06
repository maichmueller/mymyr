"""Batched RL primitives.

Tables: a :class:`TaskTable` holds many instances of one domain; every entry point takes a table plus the
instance of each row (``task_ids``, int32 [N]). A ``Task`` is accepted wherever a table is (its table of one; task_ids
may then be None)::

    table = mymyr.rl.TaskTable([task_a, task_b, task_c])   # same schemas and domain predicates; immutable, picklable
    exp = mymyr.rl.expand(table, states, task_ids)         # states [N, table.words]; row i is of instance task_ids[i]

Suites: a :class:`TaskSuite` holds one table per domain, and every entry point
that takes a table takes a suite (expand, expand_into, is_goal, the torch and JAX environments and ops, CpuEnvPool,
``mymyr.datasets.state_spaces``, ``mymyr.cuda.multi_iw`` / ``batched_iw1``). Task ids are global (int32 indices into
the suite), so one batch mixes domains; each domain's rows run on its table's path (on the device: grouped by domain,
each group on its own stream)::

    suite = mymyr.rl.TaskSuite([blocks_table, gripper_table])   # or TaskSuite.group(tasks): one table per domain
    exp = mymyr.rl.expand(suite, states, task_ids)               # states [N, suite.words]; any instance of any domain
    dom = mymyr.rl.task_domains(suite, task_ids)                 # [N] the rows' domains (per-domain policy heads)

Labels keep the domain's schema ids (``suite.schema_offsets`` flattens them) and instance-local objects; the atom
metadata keeps each domain's predicate ids (``suite.atom_metadata()``: ``domain`` [I], ``pred_offsets`` [D + 1]). A
one-domain table is unchanged by all this: it runs the same code at the same speed.

The primitive is :func:`expand`: a batch of states ``[N, W]`` in, the flat CSR expansion out::

    exp = mymyr.rl.expand(task, states)          # states: NumPy / torch / JAX CPU array, a State, or a list of States
    exp.succ      # [M, W] successor words          (same framework and word encoding as the input)
    exp.parent    # [M] int32 index of the expanded state (its instance: task_ids[parent])
    exp.schema    # [M] int32 label schema           } the action label: schema, then the schema's
    exp.binding   # [M, L] int32 label objects       } full parameter list bound to objects (-1 padding; objects of
                  #                                    the row's instance: object_offsets globalizes them)
    exp.offsets   # [N + 1] int32: state i's successors are rows offsets[i]:offsets[i+1]
    exp.goal      # [M] bool, with expand(..., goal=True)
    exp.pad(K)    # the padded [N, K] view: succ [N, K, W], schema, binding, index, mask, count (count > K = overflow)

Per state, successors come in canonical order (schema, then binding); witness pruning is off, so every applicable
ground action is one row. Outputs are sized automatically (``capacity=`` fixes the size: rows past it are counted in
``total`` and ``overflow`` is set). ``pool=ThreadPool(T)`` splits one batch over T threads; any number of Python threads
may call ``expand`` on one shared table at once.

State words: little-endian u64, atom slot i at word i >> 6, bit i & 63. NumPy uses uint64 [N, W], torch int64
[N, W], JAX uint32 [N, 2W] (the same bytes); inputs may use any of them, outputs follow the input. Inputs are read
in place (zero-copy); outputs are views of one mymyr-owned block per call (zero-copy DLPack for torch and JAX). Rows
over a table are ``table.words`` wide (the widest instance's; a smaller instance leaves its extra words zero).

Numeric tasks (``task.numeric_slots > 0``): a row is ``[W + NN]``, the atom words followed by the table's
``NN = table.numeric_words`` numeric words (``task.numeric_storage``: two int32 or one float64 per word, slots in
``task.numeric_names`` order; ``State.numeric_words`` / ``State.numeric_values()``). ``expand``, ``expand_into``,
``is_goal``, ``task.encode`` and ``task.decode`` read and write such rows; ``words=`` and ``Expansion.words`` count atom
words, ``Expansion.numeric_words`` the rest. The mask tests (``goal_test``, ``goal_count``) see atom words only: pass
``states[:, :W]`` (numeric goal constraints need ``is_goal``). Classical tasks are unchanged (NN = 0).

Device inputs (CUDA builds): ``expand`` and ``expand_into`` also take CUDA device arrays (torch CUDA tensors, JAX
GPU arrays, mymyr DLArrays; task_ids then an int32 device array). The expansion then runs on the device, byte-equal to
the CPU's, and its outputs (a ``mymyr.cuda.DeviceExpansion``, or the caller's device destinations) stay on the device in
the input's framework. ``stream=`` names the CUDA stream (None: the context's), ``ctx=`` the ``mymyr.cuda.Context``
(None: the table's default context on the input's device); inputs are imported and outputs exported with the DLPack
stream semantics of ``mymyr.cuda``. Host inputs take the CPU path unchanged (``stream`` and ``ctx`` are then ignored);
the other functions here stay on the CPU. Numeric tables are not supported on the device (ValueError): expand their
rows from host memory.

Environments: :mod:`mymyr.rl.torch` (needs torch) has batched planning environments on the CPU or a CUDA device
(``BatchedEnv``, the TorchRL ``PlanningEnv``), torch custom ops for the step and the expansion, and the counter-based
RNG of the random policy; :mod:`mymyr.rl.jax` the functional JAX environment; :class:`CpuEnvPool` EnvPool-style
asynchronous environments on the CPU (send / recv from any number of Python threads).

RL helpers: :func:`prefix_masks` / :func:`schema_masks` (factored action masks from an expansion's labels),
:func:`novelty_update` (width-1 novelty rewards over per-environment atom tables) and :func:`her_relabel` (hindsight
relabels with atom-set goals, counter-based RNG), in the inputs' framework: NumPy and torch (CPU and CUDA) run the
native helpers, JAX arrays the jnp versions of :mod:`mymyr.rl.jax`.

For encoders and environments (mymyr ships no encoders; mifrost does): ``table.atom_metadata()`` (the instances'
atoms with ``atom_offsets``, table predicate ids), :func:`object_offsets` (PyG ``ptr``), :func:`goal_masks`,
``table.initial_states()``, ``task.device_arrays()``, and :func:`goal_test`, :func:`goal_count`,
:func:`is_goal` here. There is no replay buffer: feed these arrays to TorchRL / flashbax buffers.
"""

import sys
from typing import Any, Optional, Union

import numpy as np

from mymyr._core import Task, TaskHandle
from mymyr._core._rl import (
    CpuEnvPool,
    Expansion,
    PaddedExpansion,
    PoolBatch,
    TaskSuite,
    TaskTable,
    ThreadPool,
    expand,
    expand_into,
    goal_count,
    goal_test,
    is_goal,
    random_walks,
)
from mymyr.rl._ops import HerBatch, her_relabel, novelty_update, prefix_masks, schema_masks

__all__ = [
    "CpuEnvPool",
    "Expansion",
    "HerBatch",
    "PaddedExpansion",
    "PoolBatch",
    "TaskSuite",
    "TaskTable",
    "ThreadPool",
    "as_table",
    "expand",
    "expand_into",
    "goal_count",
    "goal_masks",
    "goal_test",
    "her_relabel",
    "is_goal",
    "novelty_update",
    "object_offsets",
    "prefix_masks",
    "random_walks",
    "schema_masks",
    "task_domains",
]

TableLike = Union[TaskSuite, TaskTable, Task, TaskHandle]


def as_table(table: Union[TaskTable, Task, TaskHandle]) -> TaskTable:
    """A TaskTable: ``table`` itself, or the table of one instance of a Task / TaskHandle (a new TaskTable; the entry
    points take a Task directly, through its cached table)."""
    if isinstance(table, TaskTable):
        return table
    if isinstance(table, (Task, TaskHandle)):
        return TaskTable([table])
    raise TypeError(f"mymyr: expected a TaskTable, a Task or a TaskHandle, not {type(table).__name__}")


def _num_objects(table: TableLike) -> list[int]:
    if isinstance(table, (TaskSuite, TaskTable)):
        return table.num_objects
    if isinstance(table, (Task, TaskHandle)):
        task = table.task if isinstance(table, TaskHandle) else table
        return [int(task.num_objects)]
    raise TypeError(f"mymyr: expected a TaskSuite, a TaskTable, a Task or a TaskHandle, not {type(table).__name__}")


def _gather(fw: str, values: Any, ids: Any, like: Any) -> Any:
    """values[ids] in the framework of ``like`` (values: a NumPy array)."""
    if fw == "torch":
        import torch

        return torch.as_tensor(values, device=like.device)[ids.to(torch.int64)]
    if fw == "jax":
        import jax.numpy as jnp

        return jnp.asarray(values)[ids]
    return values[np.asarray(ids, dtype=np.int64)]


def _framework(x: Any) -> str:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return "torch"
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(x, jax.Array):
        return "jax"
    return "numpy"


def object_offsets(table: TableLike, task_ids: Any) -> Any:
    """The object offsets of a batch (PyG ``ptr``): int32 [N + 1], the exclusive cumsum of the
    rows' object counts ``num_objects[task_ids]``, in the framework (and on the device) of ``task_ids``. A label's
    objects become global node ids by ``binding + object_offsets[parent]``. Pure framework code: it traces under
    ``jax.jit`` and ``torch.compile``."""
    counts = np.asarray(_num_objects(table), dtype=np.int32)
    fw = _framework(task_ids)
    per_row = _gather(fw, counts, task_ids, task_ids)
    if fw == "torch":
        import torch

        out = torch.zeros(per_row.shape[0] + 1, dtype=torch.int32, device=per_row.device)
        out[1:] = torch.cumsum(per_row, 0)
        return out
    if fw == "jax":
        import jax.numpy as jnp

        return jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(per_row, dtype=jnp.int32)])
    return np.concatenate([np.zeros(1, np.int32), np.cumsum(per_row, dtype=np.int32)])


def task_domains(table: TableLike, task_ids: Any) -> Any:
    """The domain of each row: int32 [N], ``domain_of[task_ids]`` of a suite (all 0 over a table or a Task), in the
    framework (and on the device) of ``task_ids``; for per-domain encoder weights and policy heads. Pure framework
    code: it traces under ``jax.jit`` and ``torch.compile``."""
    if isinstance(table, TaskSuite):
        domains = np.asarray(table.domain_of, dtype=np.int32)
    else:
        domains = np.zeros(len(_num_objects(table)), dtype=np.int32)
    return _gather(_framework(task_ids), domains, task_ids, task_ids)


def goal_masks(table: TableLike, task_ids: Optional[Any] = None, framework: Optional[str] = None) -> tuple[Any, Any]:
    """The goals as state-word masks ``(gpos, gneg)``.

    With ``task_ids=None``: the table's (suite's) read-only ``[I, W]`` masks (instance i's goal in row i; a Task:
    ``[1, W]``). With ``task_ids`` [N]: fresh per-env ``[N, W]`` copies (row i: the goal of instance task_ids[i]) that
    may be edited (per-env goals, e.g. relabelled ones), in the framework of ``task_ids`` unless ``framework`` names
    one. Raises ValueError when an instance's goal has derived literals (a mask test cannot decide it)."""
    t = table if isinstance(table, (TaskSuite, TaskTable)) else as_table(table)
    if task_ids is None:
        return t.goal_masks(framework=framework)
    fw = framework or _framework(task_ids)
    gpos, gneg = t.goal_masks(framework=fw)
    if fw == "torch":
        import torch

        ids = task_ids if isinstance(task_ids, torch.Tensor) else torch.as_tensor(np.asarray(task_ids))
        idx = ids.to(torch.int64)
        return gpos.to(ids.device)[idx].contiguous(), gneg.to(ids.device)[idx].contiguous()
    if fw == "jax":
        import jax.numpy as jnp

        ids = jnp.asarray(task_ids)
        return gpos[ids], gneg[ids]
    idx = np.asarray(task_ids, dtype=np.int64)
    return gpos[idx], gneg[idx]
