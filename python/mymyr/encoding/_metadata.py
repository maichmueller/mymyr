"""Per-graph and per-entity metadata computed in the batch's array framework."""

from __future__ import annotations

from typing import Any

import numpy as np

from mymyr.encoding._suite import SuiteBatch

__all__ = ["entity_counts", "tuple_counts", "entity_graph", "entity_roles", "object_mask", "object_entities"]


def _namespace(array: Any) -> Any:
    if type(array).__module__.split(".")[0] == "torch":
        import torch
        return torch
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()
    return np


def _array(batch: Any, name: str) -> Any:
    array = getattr(batch, name)
    if _namespace(array) is np and not isinstance(array, np.ndarray):
        return np.from_dlpack(array)
    return array


def _arange(reference: Any, size: int) -> Any:
    xp = _namespace(reference)
    device = getattr(reference, "device", None)
    placement = {} if device is None else {"device": device}
    return xp.arange(size, dtype=reference.dtype, **placement)


def _cast(array: Any, dtype: Any) -> Any:
    return array.to(dtype=dtype) if _namespace(array).__name__ == "torch" else array.astype(dtype)


def _graphs(batch: Any, offsets: Any) -> Any:
    live = getattr(batch, "device_graphs", None)
    return offsets.shape[0] - 1 if live is None else live.reshape(())


def _end(batch: Any, offsets: Any) -> Any:
    graphs = _graphs(batch, offsets)
    return offsets[graphs] if isinstance(graphs, int) else _namespace(offsets).take(offsets, graphs)


def _entities(batch: Any, offsets: Any) -> Any:
    if not batch.implicit_entities:
        extent = batch.entity_role.shape[0]
    elif getattr(batch, "_entity_capacity", None) is not None:
        extent = batch._entity_capacity
    elif hasattr(batch, "_entity_capacity"):
        raise ValueError("mymyr: encoding: implicit per-entity metadata needs entity_capacity when wrapping arrays")
    else:
        # A host batch retains its offsets on the host even when it exports to another framework.
        extent = int(batch.as_dict("numpy")["entity_offsets"][-1])
    return _arange(offsets, extent)


def entity_counts(batch: Any) -> Any:
    """Entities per graph, index dtype ``[G]``; zero past the live graphs of a capacity batch.

    Results use the batch's NumPy, torch or JAX framework. A SuiteBatch gives a dict of results by domain.
    This function performs no device-to-host reads and supports JAX JIT and CUDA graph capture.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: entity_counts(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    counts = offsets[1:] - offsets[:-1]
    xp = _namespace(offsets)
    return xp.where(_arange(offsets, counts.shape[0]) < _graphs(batch, offsets), counts, 0)


def tuple_counts(batch: Any) -> Any:
    """Tuples per graph, index dtype ``[G]``, in either layout, also with ``emit_counts=False``.

    Counts the tuples stored in the graph arrays; shared static blocks are separate. Zero past live graphs.
    Supports JAX JIT and CUDA graph capture without device-to-host reads. A SuiteBatch gives a dict by domain.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: tuple_counts(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    xp = _namespace(offsets)
    graphs = _arange(offsets, offsets.shape[0] - 1)
    live = _graphs(batch, offsets)
    if hasattr(batch, "relation_counts"):
        counts = _array(batch, "relation_counts").sum(axis=1, dtype=offsets.dtype)
    elif hasattr(batch, "relation_offsets"):
        relations = _arange(offsets, batch.num_relations)
        cells = (graphs[:, None] + live * relations[None, :] if batch.schema.options["layout"] == "relation"
                 else graphs[:, None] * batch.num_relations + relations[None, :])
        tuples = _array(batch, "relation_offsets")
        counts = (tuples[1:] - tuples[:-1])[cells].sum(axis=1, dtype=offsets.dtype)
    else:
        # Simple graphs have one node per entity and one per tuple.
        nodes = _array(batch, "node_offsets")
        counts = nodes[1:] - nodes[:-1] - entity_counts(batch)
    return xp.where(graphs < live, counts, 0)


def entity_graph(batch: Any) -> Any:
    """The graph of each entity, index dtype ``[E]``, suitable for pooling.

    E includes entity capacity where retained by the batch. Unused entries map to the last live graph, G - 1
    (or -1 with no live graphs). Supports JAX JIT and CUDA graph capture without device-to-host reads.
    A SuiteBatch gives a dict by domain, with graph ids local to each domain's batch.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: entity_graph(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    xp = _namespace(offsets)
    ids = _entities(batch, offsets)
    live = _graphs(batch, offsets)
    # Repeated end offsets make empty graphs and unused graph capacity skip correctly in the search.
    boundaries = xp.where(_arange(offsets, offsets.shape[0]) <= live, offsets, _end(batch, offsets))
    found = xp.searchsorted(boundaries, ids, side="right") - 1
    return _cast(xp.where(found < live, found, live - 1), offsets.dtype)


def entity_roles(batch: Any) -> Any:
    """Roles of the entities, uint8 ``[E]``, including implicit object (0) and auxiliary (8) roles.

    Unused entity capacity has role 0. Supports JAX JIT and CUDA graph capture without device-to-host reads.
    A SuiteBatch gives a dict by domain. Role names are in ``batch.schema.entity_role_names``.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: entity_roles(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    xp = _namespace(offsets)
    ids = _entities(batch, offsets)
    if batch.implicit_entities:
        if offsets.shape[0] == 1:
            roles = ids * 0
        else:
            graphs = entity_graph(batch)
            roles = xp.where(ids - offsets[graphs] < _array(batch, "num_objects")[graphs], 0, 8)
    else:
        roles = _array(batch, "entity_role")
    return _cast(xp.where(ids < _end(batch, offsets), roles, 0), xp.uint8)


def object_mask(batch: Any) -> Any:
    """Whether each entity is an object, bool ``[E]``; False past the live entities.

    Supports JAX JIT and CUDA graph capture without device-to-host reads. A SuiteBatch gives a dict by domain.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: object_mask(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    return (entity_roles(batch) == 0) & (_entities(batch, offsets) < _end(batch, offsets))


def object_entities(batch: Any) -> Any:
    """Batch-global object entity ids, index dtype ``[sum(num_objects)]``, graph after graph.

    The result has a data-dependent length: it cannot run under JAX JIT or CUDA graph capture. On CUDA, selecting
    the objects synchronizes to allocate the result. Use ``object_mask`` for a fixed-shape alternative.
    A SuiteBatch gives a dict by domain, with entity ids local to each domain's batch.
    """
    if isinstance(batch, SuiteBatch):
        return {domain: object_entities(part) for domain, part in batch.items()}
    offsets = _array(batch, "entity_offsets")
    return _entities(batch, offsets)[object_mask(batch)]
