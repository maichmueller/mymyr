"""Mapping the action entities of an encoded batch back to the rows of the expansion they were encoded from."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mymyr._typing import SupportsDLPack
    from mymyr.encoding import FlatBatch
    from mymyr.rl import Expansion

__all__ = ["action_entities", "action_indices", "action_rows"]

_ACTION_ROLE = 4


def _ints(values: object) -> np.ndarray:
    return np.asarray(values).astype(np.int64, copy=False).reshape(-1)


def action_entities(batch: FlatBatch) -> np.ndarray:
    """The batch-global ids of the action entities of ``batch``, int64, in entity order (graph after graph).

    A batch has action entities when its schema has action relations and the call was given ``actions``. The entities
    that a model chooses among are the same ids as the action targets of the batch (``target_group`` 2) when the
    schema's ``target_sources`` lists ``"action"``.
    """
    if batch.implicit_entities:
        return np.empty(0, np.int64)
    return np.flatnonzero(np.asarray(batch.entity_role) == _ACTION_ROLE)


def _graphs_of(batch: FlatBatch, entities: np.ndarray) -> np.ndarray:
    offsets = _ints(batch.entity_offsets)
    if len(entities) and (entities.min() < 0 or entities.max() >= offsets[-1]):
        bad = int(entities[(entities < 0) | (entities >= offsets[-1])][0])
        raise ValueError(f"mymyr: encoding: entity {bad} is outside the batch's {int(offsets[-1])} entities")
    return np.searchsorted(offsets, entities, side="right") - 1


def _checked(batch: FlatBatch, entities: SupportsDLPack | None) -> np.ndarray:
    if batch.implicit_entities:
        if entities is None:
            return np.empty(0, np.int64)
        raise ValueError("mymyr: encoding: the batch has no action entities")
    chosen = action_entities(batch) if entities is None else _ints(entities)
    if entities is not None:
        _graphs_of(batch, chosen)  # the range check
        all_roles = np.asarray(batch.entity_role)
        wrong = chosen[all_roles[chosen] != _ACTION_ROLE]
        if len(wrong):
            raise ValueError(
                f"mymyr: encoding: entity {int(wrong[0])} is not an action entity "
                f"(its role is {batch.schema.entity_role_names[int(all_roles[wrong[0]])]})"
            )
    return chosen


def action_indices(batch: FlatBatch, entities: SupportsDLPack | None = None) -> np.ndarray:
    """The position of each action entity among the successors of its own state: its ``entity_ref``, int64.

    This is the action an environment takes (``CpuEnvPool.send`` and the environments of ``mymyr.rl`` take an index into
    the canonical successor order of the state). ``entities`` are batch-global entity ids, for instance
    ``batch.target_entity`` entries of the action targets a policy chose; ``None`` means every action entity of the
    batch. Raises ``ValueError`` for an id outside the batch or one that is not an action entity.
    """
    chosen = _checked(batch, entities)
    if batch.implicit_entities:
        return chosen
    return _ints(batch.entity_ref)[chosen]


def action_rows(batch: FlatBatch, actions: Expansion, entities: SupportsDLPack | None = None) -> np.ndarray:
    """The row of ``actions`` of each action entity, int64: ``actions.offsets[g] + entity_ref`` for the entity of graph
    ``g``. ``actions`` is the expansion the batch was encoded with, so ``actions.succ[row]`` is the successor of the
    chosen action, ``actions.action(row)`` the action and ``actions.schema[row]`` and ``actions.binding[row]`` its label.

    ``entities`` are batch-global entity ids (``None``: every action entity of the batch, in entity order). An
    expansion lists an action once per schema that produces it, and an entity stands for its first row, so
    ``action_rows`` returns that row for the entity. Raises ``ValueError`` for an id outside the batch, for one that is
    not an action entity, and for an expansion with fewer states than the batch has graphs.
    """
    chosen = _checked(batch, entities)
    offsets = _ints(actions.offsets)
    if len(offsets) - 1 < batch.num_graphs:
        raise ValueError(
            f"mymyr: encoding: the expansion has {len(offsets) - 1} states, the batch {batch.num_graphs} graphs"
        )
    if batch.implicit_entities:
        return chosen
    graphs = _graphs_of(batch, chosen)
    return offsets[graphs] + _ints(batch.entity_ref)[chosen]
