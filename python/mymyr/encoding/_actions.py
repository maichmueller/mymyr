"""Mapping the action entities of an encoded batch back to the rows of the expansion they were encoded from."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mymyr._typing import DeviceActionRows, SupportsDLPack
    from mymyr.encoding import DeviceFlatBatch, FlatBatch
    from mymyr.rl import Expansion

__all__ = ["action_entities", "action_entities_of", "action_indices", "action_rows"]

_ACTION_ROLE = 4


def _ints(values: object) -> np.ndarray:
    """`values` as a flat int64 host array: an array that speaks DLPack (NumPy, torch and JAX arrays on the CPU,
    mymyr.DLArray) is read in place, anything else (a sequence of ints) through NumPy. CUDA arrays are copied to the
    host before reading their values."""
    if hasattr(values, "__dlpack_device__") and values.__dlpack_device__()[0] == 2:
        import torch
        values = torch.from_dlpack(values).cpu().numpy()
    if not isinstance(values, np.ndarray) and hasattr(values, "__dlpack__"):
        values = np.from_dlpack(values)
    return np.asarray(values).astype(np.int64, copy=False).reshape(-1)


def _columns(batch: FlatBatch | DeviceFlatBatch) -> Mapping[str, np.ndarray]:
    """The arrays of `batch` as host NumPy views, copying a device batch to the host."""
    return batch.as_dict("numpy")


def action_entities(batch: FlatBatch | DeviceFlatBatch) -> np.ndarray:
    """The batch-global ids of the action entities of ``batch``, int64, in entity order (graph after graph).

    A batch has action entities when its schema has action relations and the call was given ``actions``; a batch with
    implicit entity arrays (``batch.implicit_entities``) has none. The entities that a model chooses among are the same
    ids as the action targets of the batch (``target_group`` 2) when the schema's ``target_sources`` lists
    ``"action"``. The batch may export to any framework.
    """
    if batch.implicit_entities:
        return np.empty(0, np.int64)
    return np.flatnonzero(_columns(batch)["entity_role"] == _ACTION_ROLE)


def _graphs_of(columns: Mapping[str, np.ndarray], entities: np.ndarray) -> np.ndarray:
    offsets = _ints(columns["entity_offsets"])
    if len(entities) and (entities.min() < 0 or entities.max() >= offsets[-1]):
        bad = int(entities[(entities < 0) | (entities >= offsets[-1])][0])
        raise ValueError(f"mymyr: encoding: entity {bad} is outside the batch's {int(offsets[-1])} entities")
    return np.searchsorted(offsets, entities, side="right") - 1


def _chosen(batch: FlatBatch | DeviceFlatBatch, columns: Mapping[str, np.ndarray], entities: SupportsDLPack | None) -> np.ndarray:
    """The checked action entities `entities` (all of the batch for None) of a batch with explicit entity arrays."""
    roles = columns["entity_role"]
    if entities is None:
        return np.flatnonzero(roles == _ACTION_ROLE)
    chosen = _ints(entities)
    _graphs_of(columns, chosen)  # the range check
    wrong = chosen[roles[chosen] != _ACTION_ROLE]
    if len(wrong):
        raise ValueError(
            f"mymyr: encoding: entity {int(wrong[0])} is not an action entity "
            f"(its role is {batch.schema.entity_role_names[int(roles[wrong[0]])]})"
        )
    return chosen


def _implicit(entities: SupportsDLPack | None) -> np.ndarray:
    """The mapping of a batch with implicit entity arrays, which has no action entities: empty, unless entities were
    named."""
    if entities is None:
        return np.empty(0, np.int64)
    raise ValueError("mymyr: encoding: the batch has no action entities")


def action_indices(batch: FlatBatch | DeviceFlatBatch, entities: SupportsDLPack | None = None) -> np.ndarray:
    """The position of each action entity among the successors of its own state: its ``entity_ref``, int64.

    This is the action an environment takes (``CpuEnvPool.send`` and the environments of ``mymyr.rl`` take an index into
    the canonical successor order of the state). It is the position in the full canonical order also for a batch
    encoded from an expansion that dropped no-op successors (``rl.expand(..., drop_noops=True)``): a kept action has the
    index it has in the full expansion. ``entities`` are batch-global entity ids (an array of any framework, or a
    sequence of ints; CUDA arrays are copied to the host), for instance ``batch.target_entity`` entries of the action
    targets a policy chose; ``None`` means every action entity of the batch, which is none for a batch with implicit
    entity arrays. Raises ``ValueError`` for an id outside the batch or one that is not an action entity.
    """
    if batch.implicit_entities:
        return _implicit(entities)
    columns = _columns(batch)
    chosen = _chosen(batch, columns, entities)
    return columns["entity_ref"][chosen].astype(np.int64)


def action_rows(batch: FlatBatch | DeviceFlatBatch, actions: Expansion | DeviceActionRows, entities: SupportsDLPack | None = None) -> np.ndarray:
    """The row of ``actions`` of each action entity, int64. ``actions`` is the expansion the batch was encoded with, so
    ``actions.succ[row]`` is the successor of the chosen action, ``actions.action(row)`` the action, ``actions.schema[row]``
    and ``actions.binding[row]`` its label and ``actions.succ_hash[row]`` its successor's hash.

    The row of an entity of graph ``g`` is ``actions.offsets[g] + entity_ref``; for an expansion with an ``index``
    (``rl.expand(..., drop_noops=True)``) it is the row of graph ``g`` whose ``index`` is the entity's ``entity_ref``,
    found by a binary search over the rows, which are sorted by state and index. ``entities`` are batch-global entity
    ids, as in ``action_indices`` (``None``: every action entity of the batch, in entity order). An expansion lists an
    action once per schema that produces it, and an entity stands for its first row, so ``action_rows`` returns that
    row for the entity. Raises ``ValueError`` for an id outside the batch, for one that is not an action entity, for an
    expansion with fewer states than the batch has graphs, and for an entity whose action the expansion does not have.
    """
    offsets = _ints(actions.offsets)
    if len(offsets) - 1 < batch.num_graphs:
        raise ValueError(
            f"mymyr: encoding: the expansion has {len(offsets) - 1} states, the batch {batch.num_graphs} graphs"
        )
    if batch.implicit_entities:
        return _implicit(entities)
    columns = _columns(batch)
    chosen = _chosen(batch, columns, entities)
    graphs = _graphs_of(columns, chosen).astype(np.int64)
    refs = columns["entity_ref"][chosen].astype(np.int64)
    index = getattr(actions, "index", None)
    if index is None:
        return offsets[graphs] + refs
    total = int(offsets[-1])
    index = _ints(index)[:total]
    key = (np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets)) << 32) | index
    want = (graphs << 32) | refs
    rows = np.minimum(np.searchsorted(key, want), max(total - 1, 0))
    missing = key[rows] != want if total else np.ones(len(want), bool)
    if missing.any():
        bad = int(np.flatnonzero(missing)[0])
        raise ValueError(f"mymyr: encoding: the expansion has no row for action entity {int(chosen[bad])} "
                         f"(graph {int(graphs[bad])}, canonical index {int(refs[bad])})")
    return rows.astype(np.int64)


def action_entities_of(batch: FlatBatch | DeviceFlatBatch, indices: Any, graphs: Any = None) -> Any:
    """The action entity of each canonical successor index: the inverse of ``action_indices``.

    ``indices`` are positions in the canonical successor order of a graph's state (the actions an environment takes, a
    plan step, a set of equivalent actions), ``[N]`` or ``[N, K]``; row ``i`` belongs to graph ``i``, or to graph
    ``graphs[i]`` when ``graphs`` ([N]) is given. The result has the shape of ``indices``: the batch-global id of the
    action entity whose ``entity_ref`` is the index in that graph, or -1 where the graph has none (an index out of
    range or negative, an action whose no-op successor the expansion dropped, a graph without expansion rows, a batch
    with implicit entity arrays).

    On the host the result is an int64 NumPy array (``indices`` and ``graphs`` of any framework). For a
    ``DeviceFlatBatch`` they are torch tensors on the batch's device and so is the result, computed by torch operations
    that do not wait for the device: a policy can call it inside a captured step.
    """
    if hasattr(batch, "device_sizes"):
        return _device_entities_of(batch, indices, graphs)
    if not hasattr(indices, "__dlpack__"):
        indices = np.asarray(indices)
    shape = tuple(np.from_dlpack(indices).shape) if not hasattr(indices, "shape") else tuple(indices.shape)
    if len(shape) not in (1, 2):
        raise ValueError("mymyr: encoding: indices must be [N] or [N, K]")
    values = _ints(indices)
    rows = shape[0]
    owner = np.arange(rows, dtype=np.int64) if graphs is None else _ints(graphs)
    if len(owner) != rows:
        raise ValueError(f"mymyr: encoding: graphs has {len(owner)} entries, indices {rows} rows")
    if batch.implicit_entities:
        return np.full(shape, -1, np.int64)
    columns = _columns(batch)
    offsets = _ints(columns["entity_offsets"])
    roles = np.asarray(columns["entity_role"])
    entities = np.arange(len(roles), dtype=np.int64)
    keys = np.where(roles == _ACTION_ROLE,
                    ((np.searchsorted(offsets, entities, side="right") - 1) << 32) | _ints(columns["entity_ref"]),
                    np.iinfo(np.int64).max)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    graph = np.repeat(owner, values.size // rows if rows else 0)
    valid = (values >= 0) & (values <= 0x7FFFFFFF) & (graph >= 0) & (graph < len(offsets) - 1)
    want = np.where(valid, (graph << 32) | np.clip(values, 0, 0x7FFFFFFF), -1)
    if not len(order):
        return np.full(shape, -1, np.int64)
    at = np.minimum(np.searchsorted(sorted_keys, want), len(order) - 1)
    return np.where(valid & (sorted_keys[at] == want), order[at], -1).reshape(shape).astype(np.int64)


def _device_entities_of(batch: DeviceFlatBatch, indices: Any, graphs: Any) -> Any:
    import torch

    device = batch.device
    if not isinstance(indices, torch.Tensor) or indices.device != device or indices.ndim not in (1, 2):
        raise ValueError(f"mymyr: encoding: indices must be a torch tensor [N] or [N, K] on {device}")
    rows = indices.shape[0]
    if graphs is None:
        owner = torch.arange(rows, dtype=torch.int64, device=device)
    else:
        if not isinstance(graphs, torch.Tensor) or graphs.device != device or tuple(graphs.shape) != (rows,):
            raise ValueError(f"mymyr: encoding: graphs must be a torch tensor [{rows}] on {device}")
        owner = graphs.to(torch.int64)
    if batch.implicit_entities or not batch.entity_role.numel():
        return torch.full(indices.shape, -1, dtype=torch.int64, device=device)
    offsets = batch.entity_offsets.to(torch.int64)
    roles = batch.entity_role
    entities = torch.arange(roles.shape[0], dtype=torch.int64, device=device)
    # the entities past the batch's (the tails of capacity arrays) have role 0: never an action's
    graph_of = torch.searchsorted(offsets, entities, right=True) - 1
    keys = torch.where(roles == _ACTION_ROLE, (graph_of << 32) | batch.entity_ref.to(torch.int64),
                       torch.iinfo(torch.int64).max)
    sorted_keys, order = torch.sort(keys, stable=True)
    values = indices.to(torch.int64)
    graph = owner if values.ndim == 1 else owner[:, None].expand_as(values)
    valid = (values >= 0) & (values <= 0x7FFFFFFF) & (graph >= 0) & (graph < offsets.shape[0] - 1)
    want = torch.where(valid, (graph << 32) | values.clamp(0, 0x7FFFFFFF), -1)
    at = torch.searchsorted(sorted_keys, want.reshape(-1)).clamp(max=sorted_keys.numel() - 1).reshape(want.shape)
    return torch.where(valid & (sorted_keys[at] == want), order[at], -1)
