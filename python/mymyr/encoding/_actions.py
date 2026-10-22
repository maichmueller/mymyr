"""Mapping the action entities of an encoded batch back to the rows of the expansion they were encoded from."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mymyr._typing import DeviceActionRows, SupportsDLPack
    from mymyr.encoding import DeviceFlatBatch, FlatBatch
    from mymyr.rl import Expansion

__all__ = ["action_entities", "action_indices", "action_rows"]

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
    the canonical successor order of the state). ``entities`` are batch-global entity ids (an array of any framework,
    or a sequence of ints; CUDA arrays are copied to the host), for instance ``batch.target_entity`` entries of the action targets a policy chose;
    ``None`` means every action entity of the batch, which is none for a batch with implicit entity arrays. Raises
    ``ValueError`` for an id outside the batch or one that is not an action entity.
    """
    if batch.implicit_entities:
        return _implicit(entities)
    columns = _columns(batch)
    chosen = _chosen(batch, columns, entities)
    return columns["entity_ref"][chosen].astype(np.int64)


def action_rows(batch: FlatBatch | DeviceFlatBatch, actions: Expansion | DeviceActionRows, entities: SupportsDLPack | None = None) -> np.ndarray:
    """The row of ``actions`` of each action entity, int64: ``actions.offsets[g] + entity_ref`` for the entity of graph
    ``g``. ``actions`` is the expansion the batch was encoded with, so ``actions.succ[row]`` is the successor of the
    chosen action, ``actions.action(row)`` the action and ``actions.schema[row]`` and ``actions.binding[row]`` its label.

    ``entities`` are batch-global entity ids, as in ``action_indices`` (``None``: every action entity of the batch, in
    entity order). An expansion lists an action once per schema that produces it, and an entity stands for its first
    row, so ``action_rows`` returns that row for the entity. Raises ``ValueError`` for an id outside the batch, for one
    that is not an action entity, and for an expansion with fewer states than the batch has graphs.
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
    return offsets[_graphs_of(columns, chosen)] + columns["entity_ref"][chosen]
