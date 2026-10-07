"""State spaces, generalized state spaces, knowledge bases, tuple graphs, samplers, object graphs and certificates
(datasets/*.hpp in the C++ core), on the CPU or a CUDA device, for classical and numeric tasks.

    task = mymyr.Task.from_pddl("domain.pddl", "p01.pddl", atoms="frozen")
    space = mymyr.datasets.state_space(task, threads=8)         # None if the generation failed
    a = space.arrays(framework="torch")                          # zero-copy: state_words, forward/backward CSR,
    a["forward_targets"], a["unit_goal_distances"], a["goal"]    # labels, costs, goal distances, flags
    space.state(i), space.transitions(i), space.find(state)

    results = mymyr.datasets.generate_many(tasks, threads=32)    # the instance pool: one task per thread
    gss = mymyr.datasets.GeneralizedStateSpace(mymyr.datasets.sorted_by_size([r.space for r in results]))

    sampler = mymyr.datasets.StateSpaceSampler(space, seed=0)    # deterministic for a seed on every platform
    ids = sampler.sample_states(1024)

    g = mymyr.datasets.object_graph(space.state(i))              # as in mimir's object graph (not an encoder)
    g.color_refinement_certificate(), g.kfwl_certificate(2)

    tasks = mymyr.rl.TaskTable.from_pddl("domain.pddl", "problems/")   # a task set: one domain, many problems
    kb = mymyr.datasets.KnowledgeBase(tasks, generalized=True, width=2) # spaces, class graph, tuple graphs
    kb.state_spaces, kb.generalized_state_space, kb.tuple_graph(0, v)
    graphs = mymyr.datasets.tuple_graphs(space, width=1)         # the tuple graph of every vertex (CPU)

On the device (CUDA builds; mymyr.cuda): ``device=`` a device ordinal or a mymyr.cuda.Context.

    space = mymyr.datasets.state_space(task, device=0)           # a mymyr.cuda.DeviceStateSpace
    a = space.arrays(framework="torch")                          # the same arrays, in device memory (zero-copy)
    host = space.to_host()                                       # the StateSpace above
    spaces = mymyr.datasets.state_spaces(table, device=0)        # every instance of a TaskTable, one pipeline
    spaces = mymyr.datasets.state_spaces(suite, device=0)        # a TaskSuite: one pipeline per domain, global order
    sampler = mymyr.datasets.StateSpaceSampler(spaces[0], seed=0)

Semantics follow mimir's StateSpace: every applicable action is a transition, the goal distances
are computed backward from the goal states, remove_if_unsolvable (default True, as in mimir) drops the whole space when
the initial state cannot reach a goal, and a statically false goal gives no space at all. State ids are in
breadth-first discovery order with canonical successor order, independent of the thread count, and the device gives
the same ids and arrays. Every generation runs with the thread state detached.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

from mymyr._core import Task, TaskHandle
from mymyr._core._datasets import (
    GeneralizedStateSpace,
    GenerationResult,
    KnowledgeBase,
    ObjectGraph,
    ObjectGraphBuilder,
    StateSpace,
    StateSpaceSampler,
    Status,
    TupleGraph,
    generate,
    generate_many,
    object_graph,
    sorted_by_size,
    tuple_graph,
    tuple_graphs,
)
from mymyr._core._datasets import state_space as _cpu_state_space
from mymyr._core._rl import TaskSuite, TaskTable

if TYPE_CHECKING:
    from mymyr.cuda import Context, DeviceStateSpace

Certificate = Literal["kfwl", "color_refinement"]


def _device_args(device: int | Context) -> dict[str, Context | int]:
    if isinstance(device, bool) or not isinstance(device, int):
        return {"ctx": device}
    return {"device": device}


def _cuda_module():
    try:
        from mymyr import cuda
    except ImportError as e:
        raise ValueError("mymyr: device= needs a CUDA build of mymyr (this one has no mymyr.cuda)") from e
    return cuda


def _no_symmetry(symmetry_pruning: bool) -> None:
    if symmetry_pruning:
        raise ValueError("mymyr: symmetry pruning runs on the CPU only (device=None)")


Tasks = TaskSuite | TaskTable | Sequence[Task | TaskHandle]


def _table(tasks: Tasks) -> TaskSuite | TaskTable:
    """A table or suite of the tasks (a sequence: TaskSuite.group, global id i = tasks[i]; one domain: its table)."""
    return tasks if isinstance(tasks, (TaskSuite, TaskTable)) else TaskSuite.group(list(tasks))


def _tasks(tasks: Tasks) -> list[Task | TaskHandle]:
    return list(tasks.tasks) if isinstance(tasks, (TaskSuite, TaskTable)) else list(tasks)


@overload
def state_space(
    task: Task | TaskHandle,
    *,
    device: None = None,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
) -> StateSpace | None: ...
@overload
def state_space(
    task: Task | TaskHandle,
    *,
    device: int | Context,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
) -> DeviceStateSpace | None: ...
def state_space(
    task: Task | TaskHandle,
    *,
    device: int | Context | None = None,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
) -> StateSpace | DeviceStateSpace | None:
    """The state space of a task, or None if the generation failed (as in mimir's StateSpace.create).

    device None: on the CPU (a StateSpace; threads: generation threads, default 1). Otherwise a device ordinal or a
    mymyr.cuda.Context: on that device (a mymyr.cuda.DeviceStateSpace with the same ids and arrays, in device memory;
    threads: host threads for the host-side work, default all cores; no symmetry pruning).

    Options: max_states (fail when the space has max(max_states, 2) states or more, as in mimir), max_seconds,
    remove_if_unsolvable (no space when the initial state cannot reach a goal), symmetry_pruning (one state per
    certificate class of its object graph; CPU, single-threaded), certificate ('kfwl' or the cheaper but weaker
    'color_refinement') and k (2 or 3) for symmetry pruning, labels (keep (schema, binding) per transition).
    """
    if device is None:
        return _cpu_state_space(
            task,
            threads=1 if threads is None else threads,
            max_states=max_states,
            max_seconds=max_seconds,
            remove_if_unsolvable=remove_if_unsolvable,
            symmetry_pruning=symmetry_pruning,
            certificate=certificate,
            k=k,
            labels=labels,
        )
    _no_symmetry(symmetry_pruning)
    return _cuda_module().state_space(
        task,
        **_device_args(device),
        threads=0 if threads is None else threads,
        max_states=max_states,
        max_seconds=max_seconds,
        remove_if_unsolvable=remove_if_unsolvable,
        labels=labels,
    )


@overload
def state_spaces(
    tasks: Tasks,
    *,
    device: None = None,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
    wave_states: int | None = None,
    wave_instances: int | None = None,
) -> list[StateSpace | None]: ...
@overload
def state_spaces(
    tasks: Tasks,
    *,
    device: int | Context,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
    wave_states: int | None = None,
    wave_instances: int | None = None,
) -> list[DeviceStateSpace | None]: ...
def state_spaces(
    tasks: Tasks,
    *,
    device: int | Context | None = None,
    threads: int | None = None,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
    wave_states: int | None = None,
    wave_instances: int | None = None,
) -> list[StateSpace | None] | list[DeviceStateSpace | None]:
    """The state spaces of many tasks (a TaskTable's or TaskSuite's instances, or a sequence of tasks), in input
    (global) order; None where the generation failed. Space i equals state_space(tasks[i], ...); over a suite its domain
    is suite.domain_of[i] and its labels carry that domain's schema ids.

    device None: the CPU instance pool (generate_many: one task per worker, threads workers, default all cores).
    Otherwise each domain's instances run in one device pipeline (mymyr.cuda.state_spaces; a sequence of tasks becomes
    TaskSuite.group(tasks)), the domains one after the other, admitted in waves of fewer than wave_states states
    (default 2^24) and at most wave_instances instances (default unlimited), which bound the device memory. Options as
    state_space().
    """
    if device is None:
        results = generate_many(
            _tasks(tasks),
            threads=0 if threads is None else threads,
            max_states=max_states,
            max_seconds=max_seconds,
            remove_if_unsolvable=remove_if_unsolvable,
            symmetry_pruning=symmetry_pruning,
            certificate=certificate,
            k=k,
            labels=labels,
        )
        return [r.space for r in results]
    _no_symmetry(symmetry_pruning)
    return _cuda_module().state_spaces(
        _table(tasks),
        **_device_args(device),
        threads=0 if threads is None else threads,
        max_states=max_states,
        max_seconds=max_seconds,
        remove_if_unsolvable=remove_if_unsolvable,
        labels=labels,
        wave_states=wave_states,
        wave_instances=wave_instances,
    )


def generalized_state_space(
    tasks: Tasks,
    *,
    device: int | Context | None = None,
    threads: int | None = None,
    sort_by_size: bool = True,
    max_states: int | None = None,
    max_seconds: float | None = None,
    remove_if_unsolvable: bool = True,
    symmetry_pruning: bool = False,
    certificate: Certificate = "kfwl",
    k: int = 2,
    labels: bool = True,
) -> GeneralizedStateSpace:
    """The generalized state space of tasks of one domain, as mimir builds it: each task's state space, failures
    skipped, sorted ascending by size (stable), then the class graph. device None: the CPU instance pool (threads
    workers, default all cores); symmetry_pruning=True gives the symmetry-reduced class graph. Otherwise the spaces
    come from one device pipeline (mymyr.cuda.generate_state_spaces with host output); the class graph is the same.
    A TaskSuite of several domains raises ValueError (one generalized space per domain: pass suite.tables[d])."""
    if isinstance(tasks, TaskSuite) and tasks.num_domains > 1:
        raise ValueError(
            f"mymyr: a generalized state space is of one domain; the suite has {tasks.num_domains} (pass suite.tables[d])"
        )
    if device is None:
        spaces = [
            s
            for s in state_spaces(
                tasks,
                threads=threads,
                max_states=max_states,
                max_seconds=max_seconds,
                remove_if_unsolvable=remove_if_unsolvable,
                symmetry_pruning=symmetry_pruning,
                certificate=certificate,
                k=k,
                labels=labels,
            )
            if s is not None
        ]
    else:
        _no_symmetry(symmetry_pruning)
        results = _cuda_module().generate_state_spaces(
            _table(tasks),
            **_device_args(device),
            output="host",
            threads=0 if threads is None else threads,
            max_states=max_states,
            max_seconds=max_seconds,
            remove_if_unsolvable=remove_if_unsolvable,
            labels=labels,
        )
        spaces = [r.host for r in results if r.host is not None]
    if sort_by_size:
        spaces = sorted_by_size(spaces)
    return GeneralizedStateSpace(spaces)


__all__ = [
    "Certificate",
    "GeneralizedStateSpace",
    "GenerationResult",
    "KnowledgeBase",
    "ObjectGraph",
    "ObjectGraphBuilder",
    "StateSpace",
    "StateSpaceSampler",
    "Status",
    "TupleGraph",
    "generalized_state_space",
    "generate",
    "generate_many",
    "object_graph",
    "sorted_by_size",
    "state_space",
    "state_spaces",
    "tuple_graph",
    "tuple_graphs",
]
