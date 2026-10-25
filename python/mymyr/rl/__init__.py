"""Batched RL primitives.

Tables: a :class:`TaskTable` holds many instances of one domain; every entry point takes a table plus the
instance of each row (``task_ids``, int32 [N]). A ``Task`` is accepted wherever a table is (its table of one; task_ids
may then be None)::

    table = mymyr.rl.TaskTable([task_a, task_b, task_c])   # same schemas and domain predicates; immutable, picklable
    exp = mymyr.rl.expand(table, states, task_ids)         # states [N, W]; row i is of instance task_ids[i]

Suites: a :class:`TaskSuite` holds one table per domain, and every entry point
that takes a table takes a suite (expand, expand_into, is_goal, the torch and JAX environments and ops, CpuEnvPool,
``mymyr.datasets.state_spaces``, ``mymyr.cuda.multi_iw`` / ``batched_iw1``). Task ids are global (int32 indices into
the suite), so one batch mixes domains; each domain's rows run on its table's path (on the device: grouped by domain,
each group on its own stream)::

    suite = mymyr.rl.TaskSuite([blocks_table, gripper_table])   # or TaskSuite.group(tasks): one table per domain
    exp = mymyr.rl.expand(suite, states, task_ids)               # states [N, W]; any instance of any domain
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
    exp.derived   # [M, D] derived-atom words of the successors, with expand(..., derived=True) (D = table.derived_words)
    exp.pad(k)    # the padded [N, k] view: succ [N, k, W], schema, binding, index, mask, count (count > k = overflow);
                  #   exp.pad() sizes k to the next power of two at or above the largest count; expand(..., k=k)

Per state, successors come in canonical order (schema, then binding); witness pruning is off, so every applicable
ground action is one row. Outputs are sized automatically (``capacity=`` fixes the size, at most 2**31 - 1 rows: rows
past it are counted in ``total`` and ``overflow`` is set; a batch holds at most 2**31 - 1 states and successors).
``pool=ThreadPool(T)`` splits one batch over T threads; any number of Python threads may call ``expand`` on one shared
table at once.

Action masks without an expansion: :func:`applicable_schemas` returns bool [N, S], and
:func:`applicable_objects` returns bool [N, O] for the next object given a schema and a parameter prefix.
A bit is true iff an applicable full binding extends it, including numeric effect applicability rules.
:func:`is_applicable` checks full labels in bool [N]. These functions accept CPU state arrays or States,
return the input framework's bool arrays, release the GIL and split rows with ``pool=ThreadPool(T)``.
They search for witnesses without constructing successor rows; CUDA inputs raise ValueError.

State words: little-endian u64, atom slot i at word i >> 6, bit i & 63. NumPy uses uint64 [N, W], torch int64
[N, W], JAX uint32 [N, 2W] (the same bytes); inputs may use any of them, outputs follow the input. Inputs are read
in place (zero-copy); outputs are views of one mymyr-owned block per call (zero-copy DLPack for torch and JAX).

Row width: a batch's rows have W atom words, at least the widest of its rows' instances' words
(``table.instance_words``) and at most ``table.words`` (the widest instance's). A row's words past its own instance's
are padding: never read, and written zero, so the cost of a row follows its own instance's width, not the table's.
Outputs are as wide as the widest instance among the batch's rows: ``expand`` successors (``words=`` fixes the
width), ``table.initial_states(task_ids)``, and each ``CpuEnvPool`` batch (``CpuEnvPool(words=)`` fixes it); with
every instance (``initial_states()``) that is ``table.words``. The slot layout of a row is the task's encoding, a
function of the task and its options (``Task.from_pddl``: atoms, numeric_storage); tables refuse tasks with lazy
atom slots.

Numeric tasks (``task.numeric_slots > 0``): a row is ``[W + NN]``, the atom words followed by the table's
``NN = table.numeric_words`` numeric words (``task.numeric_storage``: two int32 or one float64 per word, slots in
``task.numeric_names`` order; ``State.numeric_words`` / ``State.numeric_values()``). ``expand``, ``expand_into``,
``is_goal``, ``task.encode`` and ``task.decode`` read and write such rows; ``words=`` and ``Expansion.words`` count atom
words, ``Expansion.numeric_words`` the rest. Every numeric instance of a table stores its values alike
(``table.numeric_storage``), and :func:`numeric_values` decodes the numeric words of rows into float64 values. The mask tests (``goal_test``, ``goal_count``) see atom words only: pass
``states[:, :W]`` (numeric goal constraints need ``is_goal``). Classical tasks are unchanged (NN = 0).

Device inputs (CUDA builds): ``expand`` and ``expand_into`` also take CUDA device arrays (torch CUDA tensors, JAX
GPU arrays, mymyr DLArrays; task_ids then an int32 device array). The expansion then runs on the device, byte-equal to
the CPU's, and its outputs (a ``mymyr.cuda.DeviceExpansion``, or the caller's device destinations) stay on the device in
the input's framework. ``stream=`` names the CUDA stream (None: the context's), ``ctx=`` the ``mymyr.cuda.Context``
(None: the table's default context on the input's device); inputs are imported and outputs exported with the DLPack
stream semantics of ``mymyr.cuda``. Host inputs take the CPU path unchanged (``stream`` and ``ctx`` are then ignored);
the other functions here stay on the CPU. ``expand_into(..., status=s, stream=)`` (s an int32 device array [1]) does
not wait for the device, so it needs ``stream=``, the stream that reads its outputs: the total stays in
``offsets[N]``, ``s[0]`` gets bit 0 for more successors than the capacity and bit 1 for an invalid row (nothing is
written then), and the result holds only ``capacity`` (tables whose expansion needs the host wait and return the usual
keys; rows past the total then keep their contents, and an invalid row raises). Numeric tables and suites preserve each instance's CPU encoding on the
device; their expansion uses per-instance kernels and gathers/scatters rows at the shared atom/numeric boundary.
``row_count=`` optionally bounds this asynchronous expansion to a live prefix: a contiguous int32 or uint32
CUDA array [1], read as an unsigned count and clamped to N. Padding contributes no successors or validation
errors; offsets past the prefix repeat its total. It requires ``status=`` and an asynchronous device table.
A captured graph owns an exclusive expansion workspace per table; eager calls and other captures use their own.
The workspace stays alive until the graph is destroyed and its last replay has completed, also for a manual
``torch.cuda.CUDAGraph``. Callers may drop every result without releasing the workspace. Two instantiations of one
capture share its workspace, so do not launch them concurrently. Warm ``expand_into`` on the capture stream with
identical shapes, destinations present and options; all expansions over one table within a graph must keep them
fixed. Unwarmed calls and changes to these shapes or options are refused during capture. Inputs and caller-provided
destinations must stay alive at their captured addresses until the final replay completes: synchronize the replay
stream before freeing or reusing them. External CUDA streams must remain valid while arrays produced on them are
retained.

Environments: :mod:`mymyr.rl.torch` (needs torch) has batched planning environments on the CPU or a CUDA device
(``BatchedEnv``, the TorchRL ``PlanningEnv``), torch custom ops for the step and the expansion, and the counter-based
RNG of the random policy; :mod:`mymyr.rl.jax` the functional JAX environment; :class:`CpuEnvPool` EnvPool-style
asynchronous environments on the CPU (send / recv from any number of Python threads).

RL helpers: :func:`prefix_masks` / :func:`schema_masks` (factored action masks from an expansion's labels),
:func:`novelty_update` (width-1 novelty rewards over per-environment atom tables) and :func:`her_relabel` (hindsight
relabels with atom-set goals, counter-based RNG), in the inputs' framework: NumPy and torch (CPU and CUDA) run the
native helpers, JAX arrays the jnp versions of :mod:`mymyr.rl.jax`.

For custom encoders and environments (see also :mod:`mymyr.encoding`): ``table.atom_metadata()`` (the instances'
atoms with ``atom_offsets``, table predicate ids), :func:`object_offsets` (PyG ``ptr``), :func:`goal_masks`,
``table.initial_states()``, ``task.device_arrays()``, :func:`decode` (table rows to States),
:func:`successor_bounds` (a bound of every reachable state's successor count, e.g. for a fixed action count), and
:func:`goal_test`, :func:`goal_count`, :func:`is_goal` here. There is no replay buffer: feed these arrays to TorchRL /
flashbax buffers. :mod:`mymyr.rl.gym` (needs gymnasium) has Gymnasium environments over these primitives.
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
    applicable_objects,
    applicable_schemas,
    is_applicable,
    expand,
    expand_into,
    goal_count,
    goal_test,
    is_goal,
    numeric_values,
    random_walks,
    successor_bounds,
)
from mymyr.rl._ops import HerBatch, her_relabel, novelty_update, prefix_masks, schema_masks
from mymyr.rl._sampling import InstanceSampler

__all__ = [
    "CpuEnvPool",
    "Expansion",
    "HerBatch",
    "InstanceSampler",
    "PaddedExpansion",
    "PoolBatch",
    "TaskSuite",
    "TaskTable",
    "ThreadPool",
    "applicable_objects",
    "applicable_schemas",
    "is_applicable",
    "as_table",
    "decode",
    "expand",
    "expand_into",
    "goal_count",
    "goal_masks",
    "goal_test",
    "her_relabel",
    "is_goal",
    "novelty_update",
    "numeric_values",
    "object_offsets",
    "prefix_masks",
    "random_walks",
    "schema_masks",
    "successor_bounds",
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


def decode(table: TableLike, rows: Any, task_ids: Optional[Any] = None) -> list:
    """The States of rows of a table or suite: row i of instance ``task_ids[i]`` (None: instance 0 of a table of
    one) in the table's layout, ``[W + NN]``: W atom words (at least the instance's) and the table's ``NN =
    table.numeric_words`` numeric words, of which an instance uses its first ``task.numeric_words``. ``Task.decode``
    reads a task's own rows (its numeric words last), so a row of a numeric table whose instances have different
    numeric word counts decodes here. ``rows``: a NumPy-convertible uint64 (or int64) array [N, W + NN]; a Task or
    TaskHandle decodes its own rows."""
    if isinstance(table, (Task, TaskHandle)):
        return (table.task if isinstance(table, TaskHandle) else table).decode(rows)
    if not isinstance(table, (TaskSuite, TaskTable)):
        raise TypeError(f"mymyr: expected a TaskSuite, a TaskTable or a Task, not {type(table).__name__}")
    words = np.asarray(rows)
    if words.dtype == np.int64:
        words = words.view(np.uint64)
    if words.ndim != 2 or words.dtype != np.uint64:
        raise ValueError("mymyr: rows must be a uint64 (or int64) array [N, W + NN]")
    if task_ids is None:
        if len(table) != 1:
            raise ValueError("mymyr: task_ids are needed for a table of several instances")
        ids = np.zeros(len(words), np.int64)
    else:
        ids = np.asarray(task_ids, np.int64)
        if ids.shape != (len(words),):
            raise ValueError(f"mymyr: task_ids must have one entry per row ({len(words)})")
        if ((ids < 0) | (ids >= len(table))).any():
            raise ValueError(f"mymyr: a task id is outside the {len(table)} instances")
    numeric = int(table.numeric_words)
    width = words.shape[1] - numeric  # the batch's atom words: a batch is as wide as its widest instance
    tasks = table.tasks
    out: list = [None] * len(words)
    for t in np.unique(ids).tolist():
        where = np.flatnonzero(ids == t)
        task = tasks[t]
        if width < task.words:
            raise ValueError(f"mymyr: rows of {words.shape[1]} words are narrower than instance {t}'s "
                             f"{task.words} atom words plus the table's {numeric} numeric words")
        part = words[where]
        if numeric:
            part = np.concatenate([part[:, : task.words], part[:, width : width + task.numeric_words]], axis=1)
        else:
            part = part[:, :width]
        for i, state in zip(where.tolist(), task.decode(np.ascontiguousarray(part))):
            out[i] = state
    return out


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
