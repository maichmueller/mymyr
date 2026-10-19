"""Names of the entities and tuples of an encoded graph, built from the arrays of a batch and the task.

A batch stores no names. The naming rules, per entity role:

- an object is named as in its task (``entity_ref`` is its index in ``task.objects``);
- the auxiliary entity is ``auxiliary``;
- a goal entity is named after the plain goal tuple it is the target of: ``[+](on c b)[g]``, ``[-](clear a)[g]``; a
  nullary literal is ``[+](handempty)[g]``;
- an action entity is named after its action tuple: ``(unstack a b)``, the schema name followed by the objects;
- a predicate virtual node is ``predicate:<name>`` after the predicate of its key relation (``entity_ref``).
"""

import re
from dataclasses import dataclass

import numpy as np

from mymyr._core._encoding import FlatBatch
from mymyr.encoding._device import DeviceFlatBatch, _Host

_GOAL_RELATION = re.compile(r"^\[([+-])\](.+)\[g\]$")
_DECORATION = re.compile(r"^(?:\[[+-]\])?(.+?)(?:\[[a-z+\-]+\])*$")

_OBJECT, _PREDICATE_VIRTUAL, _AUXILIARY = 0, 1, 8
_ARGUMENT_SLOT, _ACTION_SLOT, _GOAL_TARGET_SLOT = 0, 3, 4


@dataclass(frozen=True)
class GraphDescription:
    """One graph of a batch by names: ``entities`` lists ``(role, name)`` in entity order (the position is the
    graph-local entity id), ``relations`` maps each relation that has tuples to its tuples as tuples of entity names, in
    catalog order and in the order of the encoding. ``str(description)`` prints it."""

    graph: int
    task_id: int
    entities: tuple[tuple[str, str], ...]
    relations: dict[str, tuple[tuple[str, ...], ...]]

    def __str__(self) -> str:
        lines = [f"graph {self.graph} (instance {self.task_id}): {len(self.entities)} entities"]
        width = len(str(max(len(self.entities) - 1, 0)))
        lines += [f"  {i:>{width}}  {role:<17} {name}" for i, (role, name) in enumerate(self.entities)]
        lines.append(f"{len(self.relations)} relations with tuples")
        for relation, tuples in self.relations.items():
            lines += [f"  {relation}({', '.join(t)})" for t in tuples]
        return "\n".join(lines)


def _tasks_of(encoder):
    table = encoder.table
    tasks = getattr(table, "tasks", None)
    return list(tasks) if tasks is not None else [table]


def _host(batch, graph):
    """A device graph as host arrays (waits for that graph), or a host batch and its graph unchanged."""
    if isinstance(batch, DeviceFlatBatch):
        return _Host(batch, int(graph)), 0
    return batch, graph


def _check(encoder, batch, graph):
    if not isinstance(batch, (FlatBatch, _Host)):
        raise TypeError(
            f"mymyr: encoding: batch must be a FlatBatch or a DeviceFlatBatch, not {type(batch).__name__}"
        )
    if batch.schema.schema_fingerprint != encoder.schema_fingerprint:
        raise ValueError("mymyr: encoding: the batch was encoded with another schema than this encoder's")
    sizes = batch.sizes
    if sizes is not None and sizes["overflow"]:
        raise ValueError("mymyr: encoding: the batch overflowed its capacities and holds no graphs")
    graph = int(graph)
    if not 0 <= graph < batch.num_graphs:
        raise IndexError(f"mymyr: encoding: graph {graph} is out of range for a batch of {batch.num_graphs} graphs")
    # tables over one domain share the schema, so the batch's instance must be one of this encoder's table
    tasks = _tasks_of(encoder)
    task_id = int(np.asarray(batch.task_ids)[graph])
    if not 0 <= task_id < len(tasks) or int(np.asarray(batch.num_objects)[graph]) != len(tasks[task_id].objects):
        raise ValueError(
            f"mymyr: encoding: graph {graph} of the batch is not an instance of this encoder's table (instance "
            f"{task_id} has {int(np.asarray(batch.num_objects)[graph])} objects in the batch)"
        )
    return graph


def _base_predicate(relation):
    return _DECORATION.match(relation).group(1)


class _Graph:
    """The arrays of one graph of a batch, as NumPy."""

    def __init__(self, encoder, batch, graph):
        schema = batch.schema
        self.names = schema.relation_names
        self.arities = schema.relation_arities
        self.slot_roles = schema.relation_slot_roles
        self.entity_role_names = schema.entity_role_names
        self.graph = graph
        self.task_id = int(np.asarray(batch.task_ids)[graph])
        self.arg_offsets = np.asarray(batch.relation_arg_offsets)
        self.args = np.asarray(batch.relation_args)
        self.by_relation = encoder.options["layout"] == "relation"
        self.relations = batch.num_relations
        self.graphs = batch.num_graphs
        entity_offsets = np.asarray(batch.entity_offsets)
        self.begin, self.end = int(entity_offsets[graph]), int(entity_offsets[graph + 1])
        if batch.implicit_entities:
            objects = int(np.asarray(batch.num_objects)[graph])
            count = self.end - self.begin
            self.role = np.full(count, _AUXILIARY, dtype=np.int64)
            self.role[:objects] = _OBJECT
            self.ref = np.full(count, -1, dtype=np.int64)
            self.ref[:objects] = np.arange(objects)
        else:
            self.role = np.asarray(batch.entity_role)[self.begin : self.end].astype(np.int64)
            self.ref = np.asarray(batch.entity_ref)[self.begin : self.end].astype(np.int64)

    def segment(self, relation):
        return relation * self.graphs + self.graph if self.by_relation else self.graph * self.relations + relation

    def tuples(self, relation):
        """The tuples of a relation in this graph as an array [count, arity] of graph-local entity ids."""
        segment = self.segment(relation)
        first, last = int(self.arg_offsets[segment]), int(self.arg_offsets[segment + 1])
        arity = self.arities[relation]
        return (self.args[first:last] - self.begin).reshape(-1, arity)


def _argument_names(graph, relation, local, objects):
    """The names of the argument entities of a tuple (the slots of the argument role)."""
    row = graph.tuples(relation)[local]
    roles = graph.slot_roles[relation]
    return [objects[int(graph.ref[int(e)])] for e, role in zip(row, roles) if role == _ARGUMENT_SLOT]


def _names(encoder, batch, graph):
    g = _Graph(encoder, batch, graph)
    objects = _tasks_of(encoder)[g.task_id].objects
    names = [None] * len(g.role)
    for i, (role, ref) in enumerate(zip(g.role.tolist(), g.ref.tolist())):
        if role == _OBJECT:
            names[i] = objects[ref]
        elif role == _AUXILIARY:
            names[i] = g.entity_role_names[_AUXILIARY]
        elif role == _PREDICATE_VIRTUAL:
            names[i] = f"predicate:{_base_predicate(g.names[ref])}"
    # goal and action entities are the target of the tuple they stand for
    for relation, name in enumerate(g.names):
        roles = g.slot_roles[relation]
        if _GOAL_TARGET_SLOT in roles:
            position = roles.index(_GOAL_TARGET_SLOT)
            sign, predicate = _GOAL_RELATION.match(name).groups()
            for local, row in enumerate(g.tuples(relation)):
                arguments = _argument_names(g, relation, local, objects)
                names[int(row[position])] = f"[{sign}]({' '.join([predicate] + arguments)})[g]"
        elif _ACTION_SLOT in roles:
            position = roles.index(_ACTION_SLOT)
            schema = name.partition("#")[0]
            for local, row in enumerate(g.tuples(relation)):
                names[int(row[position])] = f"({' '.join([schema] + _argument_names(g, relation, local, objects))})"
    unnamed = [i for i, name in enumerate(names) if name is None]
    if unnamed:
        raise ValueError(f"mymyr: encoding: no naming rule for the entities {unnamed} of graph {graph} (roles "
                         f"{sorted({g.entity_role_names[int(g.role[i])] for i in unnamed})})")
    return g, names


def entity_names(encoder, batch, graph):
    """The names of the entities of graph ``graph`` of ``batch`` in entity order (see the module's naming rules)."""
    batch, graph = _host(batch, graph)
    graph = _check(encoder, batch, graph)
    return _names(encoder, batch, graph)[1]


def describe(encoder, batch, graph):
    """A GraphDescription of graph ``graph`` of ``batch``: its entities and its relations' tuples by name."""
    original_graph = int(graph)
    batch, graph = _host(batch, graph)
    graph = _check(encoder, batch, graph)
    g, names = _names(encoder, batch, graph)
    entities = tuple((g.entity_role_names[int(role)], name) for role, name in zip(g.role.tolist(), names))
    relations = {}
    for relation, name in enumerate(g.names):
        rows = g.tuples(relation)
        if len(rows):
            relations[name] = tuple(tuple(names[int(e)] for e in row) for row in rows)
    return GraphDescription(original_graph, g.task_id, entities, relations)
