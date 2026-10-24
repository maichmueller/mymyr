"""The HGraph view of a relation-major batch, on the host and on the device."""

from __future__ import annotations

import sys
from typing import Any, Optional, overload

from mymyr._core._encoding import FlatBatch, HGraphView
from mymyr.encoding._device import _SIZE_FIELDS, DeviceFlatBatch

__all__ = ["DeviceHGraphView", "hgraph"]

SYMBOL = "_symbol_"
_OVERFLOW_WORD = _SIZE_FIELDS.index("overflow")


def _error(message: str) -> ValueError:
    return ValueError(f"mymyr: encoding: {message}")


class DeviceHGraphView:
    """A relation-major ``DeviceFlatBatch`` as one heterogeneous graph: what ``HGraphView`` is for a host batch, with
    torch tensors on the batch's device. The node types, edge types, numbering and the meaning of every array are those
    of ``HGraphView``, and so are the arrays: the device view of a batch equals the host view of the same rows.

    Nothing is copied and nothing is remapped: the ``_symbol_`` ids are the batch's entity ids, so row 1 of the edge
    type (relation, k, '_symbol_') is a strided view of the batch's ``relation_args`` (stride: the relation's arity),
    ``ptr('_symbol_')`` is the batch's ``entity_offsets`` and ``entity_role`` the batch's ``entity_role``. Writing into
    them writes into the batch. The tuple numbers (row 0, an arange), the ptr of a relation and the roles of a batch
    whose entity arrays are implicit are new tensors, made on request on the device.

    Making the view reads the graph count, sizes and offsets at the relations' boundaries on the host in one transfer,
    so it waits for the device exactly once and must be done outside CUDA graph capture. This transfer and its
    metadata gathering run on torch's current stream for the batch's device; that stream must wait for the batch's
    producer before the view is made.
    After that no method waits (except for ``framework='numpy'``): the new tensors are made by torch operations on the
    current stream for the batch's device, in stream order. The view keeps the counts it read, so a batch whose arrays
    are rewritten in place (a CUDA graph replay) needs a new view. The view keeps the batch alive.
    ``mymyr.encoding.hgraph(batch)`` makes one; it refuses what the host view refuses."""

    __slots__ = ("_arity", "_arrays", "_batch", "_edge_types", "_entities", "_first_arg",
                 "_first_tuple", "_graphs", "_names", "_relation", "_schema", "_tuples")

    def __init__(self, batch: DeviceFlatBatch):
        if not isinstance(batch, DeviceFlatBatch):
            raise TypeError("mymyr: encoding: a DeviceHGraphView is of a DeviceFlatBatch")
        schema = batch.schema
        options = schema.options
        if options["layout"] != "relation":
            raise _error("the HGraph view needs a relation-major batch (layout='relation'), because it views the "
                         "tuples of a relation as one strided run of arguments; this batch is graph-major")
        if options["predicate_virtual_nodes"]:
            raise _error("the HGraph view does not support predicate virtual nodes: every node type already is a "
                         "predicate; encode without predicate_virtual_nodes")
        if options["statics"] == "shared":
            raise _error("statics='shared' is not supported by the HGraph encoding: its graphs hold the static tuples "
                         "among their own tuples")
        torch = _torch()
        with torch.cuda.device(batch.device):
            if torch.cuda.is_current_stream_capturing():
                raise _error("create a DeviceHGraphView outside capture")
        sizes = batch.device_sizes
        arrays, relations = batch._arrays.copy(), schema.num_relations
        graphs = batch.device_graphs
        if graphs is None:
            graphs = torch.full((1,), batch.num_graphs, dtype=torch.int64, device=batch.device)
        # Stream offsets are packed by the live graph count, which stays on the device until the metadata is gathered.
        at = torch.arange(relations + 1, device=batch.device) * graphs
        parts = [arrays["relation_offsets"][at], arrays["relation_arg_offsets"][at], arrays["entity_offsets"][graphs],
                 graphs]
        if sizes is not None:  # the overflow flags, read by their position in the sizes
            parts.append(sizes[_OVERFLOW_WORD : _OVERFLOW_WORD + 1])
        marks = torch.cat([p.to(torch.int64) for p in parts]).tolist()
        if sizes is not None and marks[-1]:
            raise _error("the batch overflowed its capacities, so it holds no encoding to view")
        graphs = marks[2 * relations + 3]
        arrays["entity_offsets"] = arrays["entity_offsets"][: graphs + 1]
        arrays["num_objects"] = arrays["num_objects"][:graphs]
        self._batch = batch
        self._schema = schema
        self._arrays = arrays
        self._graphs = graphs
        self._first_tuple = marks[: relations + 1]
        self._first_arg = marks[relations + 1 : 2 * relations + 1]
        self._entities = marks[2 * relations + 2]
        self._names = list(schema.relation_names)
        self._arity = list(schema.relation_arities)
        self._relation = {name: r for r, name in enumerate(self._names)}
        self._edge_types = [(name, str(k), SYMBOL)
                            for name, arity in zip(self._names, self._arity, strict=True) for k in range(arity)]
        self._tuples = [self._first_tuple[r + 1] - self._first_tuple[r] for r in range(relations)]

    # ---------------------------------------------------------------------------------------------- the types

    @property
    def batch(self) -> DeviceFlatBatch:
        """The DeviceFlatBatch that the view is of."""
        return self._batch

    @property
    def device(self) -> Any:
        """The ``torch.device`` of the batch and of every tensor of the view."""
        return self._batch.device

    @property
    def num_graphs(self) -> int:
        """The number of graphs of the batch."""
        return self._graphs

    @property
    def node_types(self) -> list[str]:
        """The node types: the names of the schema's relations, in relation order, then `_symbol_`."""
        return [*self._names, SYMBOL]

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        """The edge types (relation, position, '_symbol_'), for every relation in relation order and every position of
        its tuples (the slot number as a str)."""
        return list(self._edge_types)

    @property
    def reverse_edge_types(self) -> list[tuple[str, str, str]]:
        """The reverse of each edge type, in the same order: ('_symbol_', position, relation)."""
        return [(SYMBOL, k, name) for name, k, _ in self._edge_types]

    def _node_type(self, value: Any) -> Optional[int]:
        """The relation of a node type, or None for `_symbol_`."""
        if not isinstance(value, str):
            raise TypeError("mymyr: encoding: a node type is a name (str)")
        if value == SYMBOL:
            return None
        relation = self._relation.get(value)
        if relation is None:
            raise KeyError(f"mymyr: encoding: no node type {value}")
        return relation

    def _edge_type(self, value: Any) -> tuple[int, int, bool]:
        """(relation, position, reverse) of an edge type."""
        if not (isinstance(value, tuple) and len(value) == 3 and all(isinstance(x, str) for x in value)):
            raise TypeError("mymyr: encoding: an edge type is a tuple (relation, position, '_symbol_') of three str, "
                            "or the reverse ('_symbol_', position, relation)")
        first, position, last = value
        reverse = first == SYMBOL
        relation = self._relation.get(last if reverse else first) if reverse != (last == SYMBOL) else None
        slot = int(position) if position.isascii() and position.isdigit() and str(int(position)) == position else None
        if relation is None or slot is None or slot >= self._arity[relation]:
            raise KeyError(f"mymyr: encoding: no edge type ({first}, {position}, {last})")
        return relation, slot, reverse

    # ---------------------------------------------------------------------------------------------- the arrays

    def num_nodes(self, node_type: str) -> int:
        """The number of nodes of a node type in the whole batch: the tuples of the relation, or the entities."""
        relation = self._node_type(node_type)
        return self._entities if relation is None else self._tuples[relation]

    def ptr(self, node_type: str, *, framework: Optional[str] = None) -> Any:
        """The first node of each graph in a node type, and the end: [G + 1] entries, graph g's nodes of the type are
        [ptr[g], ptr[g + 1]). For `_symbol_` it is the batch's entity_offsets (no copy); for a relation the tuple
        offsets of its graphs counted from the relation's first tuple (a new tensor)."""
        relation = self._node_type(node_type)
        offsets = self._arrays["entity_offsets"]
        if relation is not None:
            first = relation * self._graphs
            cells = self._arrays["relation_offsets"][first : first + self._graphs + 1]
            offsets = cells - cells[:1]
        return _export(offsets, framework)

    def num_edges(self, edge_type: tuple[str, str, str]) -> int:
        """The number of edges of an edge type in the whole batch: the tuples of its relation."""
        return self._tuples[self._edge_type(edge_type)[0]]

    def _entity_row(self, relation: int, slot: int) -> Any:
        """The entity at position ``slot`` of every tuple of ``relation``: a strided view of relation_args."""
        args, arity, count = self._arrays["relation_args"], self._arity[relation], self._tuples[relation]
        # An explicit shape also describes an empty relation whose slot lies past the stored arguments.
        return args.as_strided((count,), (arity,), args.storage_offset() + self._first_arg[relation] + slot)

    def _arange(self, relation: int) -> Any:
        args = self._arrays["relation_args"]
        return _torch().arange(self._tuples[relation], dtype=args.dtype, device=args.device)

    def edge_index_0(self, edge_type: tuple[str, str, str], *, framework: Optional[str] = None) -> Any:
        """Row 0 of an edge type, the source of each edge: for (relation, k, '_symbol_') the tuple numbers 0 .. n - 1
        (a new tensor), for the reverse type the entity ids (a view of the batch)."""
        relation, slot, reverse = self._edge_type(edge_type)
        return _export(self._entity_row(relation, slot) if reverse else self._arange(relation), framework)

    def edge_index_1(self, edge_type: tuple[str, str, str], *, framework: Optional[str] = None) -> Any:
        """Row 1 of an edge type, the destination of each edge: for (relation, k, '_symbol_') the entity at position k
        of each tuple, a strided view of the batch's relation_args (no copy; its stride is the relation's arity); for
        the reverse type the tuple numbers (a new tensor)."""
        relation, slot, reverse = self._edge_type(edge_type)
        return _export(self._arange(relation) if reverse else self._entity_row(relation, slot), framework)

    def edge_index(self, edge_type: tuple[str, str, str], *, framework: Optional[str] = None) -> tuple[Any, Any]:
        """The pair (edge_index_0, edge_index_1) of an edge type."""
        relation, slot, reverse = self._edge_type(edge_type)
        tuples = _export(self._arange(relation), framework)
        entities = _export(self._entity_row(relation, slot), framework)
        return (entities, tuples) if reverse else (tuples, entities)

    @property
    def entity_role(self) -> Any:
        """The feature of the `_symbol_` nodes: the role of each entity [E] as uint8, an index into
        schema.entity_role_names. The batch's entity_role tensor (no copy), or made on the device from num_objects when
        the batch's entity arrays are implicit (the objects of a graph, then its auxiliary entity)."""
        if not self._schema.implicit_entities:
            return self._arrays["entity_role"][: self._entities]
        torch = _torch()
        offsets, objects = self._arrays["entity_offsets"], self._arrays["num_objects"]
        entity = torch.arange(self._entities, dtype=offsets.dtype, device=offsets.device)
        graph = torch.searchsorted(offsets, entity, right=True) - 1
        names = self._schema.entity_role_names
        role = torch.where(entity - offsets[graph] < objects[graph], names.index("object"), names.index("auxiliary"))
        return role.to(torch.uint8)


def _torch() -> Any:
    import torch

    return torch


def _export(tensor: Any, framework: Optional[str]) -> Any:
    if framework is None or framework in ("torch", "pytorch", "dlpack", "array_api"):
        return tensor
    if framework in ("numpy", "np"):
        return tensor.cpu().numpy()
    raise _error(f"a device HGraph view exports to 'torch', 'dlpack' or 'numpy', not {framework!r}")


def _is_jax_batch(batch: Any) -> bool:
    jax_module = sys.modules.get("mymyr.encoding.jax")
    return jax_module is not None and isinstance(batch, jax_module.JaxFlatBatch)


@overload
def hgraph(batch: FlatBatch) -> HGraphView: ...
@overload
def hgraph(batch: DeviceFlatBatch) -> DeviceHGraphView: ...
def hgraph(batch: FlatBatch | DeviceFlatBatch) -> HGraphView | DeviceHGraphView:
    """The HGraph view of ``batch``: its tuples as a heterogeneous graph, without copying them (see ``HGraphView``).

    ``batch`` is a FlatBatch of the single-state, horizon, rooted-horizon or transition encoders, or a DeviceFlatBatch,
    relation-major (``layout='relation'``), encoded without ``predicate_virtual_nodes`` and with the static tuples in
    every graph. A host batch gives an ``HGraphView``, a device batch a ``DeviceHGraphView`` of torch tensors on its
    device, equal to the host view of the same rows. The view keeps the batch alive. Raises ValueError for a batch the
    view cannot be made of: one that is graph-major, has predicate virtual nodes or shared statics, or overflowed its
    capacities, or for a device view created during CUDA graph capture; TypeError for a JAX batch, whose arrays have no
    strided views.
    """
    if isinstance(batch, DeviceFlatBatch):
        return DeviceHGraphView(batch)
    if _is_jax_batch(batch):
        raise TypeError("mymyr: encoding: the HGraph view does not view a JaxFlatBatch: its edge rows are strided "
                        "views of the batch's relation_args, and a JAX array has no views (a strided slice is a new "
                        "array); view a host batch or a DeviceFlatBatch")
    if isinstance(batch, FlatBatch) and batch.sizes is not None and batch.sizes["overflow"]:
        raise ValueError("mymyr: encoding: the batch overflowed its capacities, so it holds no encoding to view")
    return HGraphView(batch)
