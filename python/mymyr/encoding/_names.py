"""Names of the entities and tuples of an encoded graph, built from the arrays of a batch and the task.

A batch stores no names. The naming rules, per entity role:

- an object is named as in its task (``entity_ref`` is its index in ``task.objects``);
- the auxiliary entity is ``auxiliary``;
- a goal entity is named after the plain goal tuple it is the target of: ``[+](on c b)[g]``, ``[-](clear a)[g]``; a
  nullary literal is ``[+](handempty)[g]``; a subgoal entity likewise with the level of its relation,
  ``[+](on c b)[sg]``, ``[ssg]``, ``[sssg]``;
- an action entity is named after its action tuple: ``(unstack a b)``, the schema name followed by the objects;
- a history step is ``history:<dt>#<i>``, with the dt of the step and ``i`` its position among the graph's history
  steps, counted from 0; a history target is ``history:<dt>#<i>:[+](on c b)``, after the step and the literal of the
  history tuple it is the target of;
- a predicate virtual node is ``predicate:<name>`` after the predicate of its key relation (``entity_ref``).
"""

import re
from dataclasses import dataclass

import numpy as np

from mymyr._core._encoding import FlatBatch

_TARGET_RELATION = re.compile(r"^\[([+-])\](.+?)(\[g\]|\[sg\]|\[ssg\]|\[sssg\]|\[hist\])$")
_DECORATION = re.compile(r"^(?:\[[+-]\])?(.+?)(?:\[[a-z+\-]+\])*$")

_OBJECT, _PREDICATE_VIRTUAL, _AUXILIARY = 0, 1, 8
_ARGUMENT_SLOT, _ACTION_SLOT, _GOAL_TARGET_SLOT = 0, 3, 4
_SUBGOAL_TARGET_SLOT, _HISTORY_TARGET_SLOT, _HISTORY_SLOT = 5, 6, 7


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


def _check(encoder, batch, graph):
    if not isinstance(batch, FlatBatch):
        raise TypeError(f"mymyr: encoding: batch must be a FlatBatch, not {type(batch).__name__}")
    if batch.schema.schema_fingerprint != encoder.schema_fingerprint:
        raise ValueError("mymyr: encoding: the batch was encoded with another schema than this encoder's")
    sizes = batch.sizes
    if sizes is not None and sizes["overflow"]:
        raise ValueError("mymyr: encoding: the batch overflowed its capacities and holds no graphs")
    graph = int(graph)
    if not 0 <= graph < batch.num_graphs:
        raise IndexError(f"mymyr: encoding: graph {graph} is out of range for a batch of {batch.num_graphs} graphs")
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
        self.history_dt = self.history_entities = None
        if batch.schema.options["history"]:
            offsets = np.asarray(batch.history_offsets)
            first, last = int(offsets[graph]), int(offsets[graph + 1])
            self.history_entities = np.asarray(batch.history_entity)[first:last].astype(np.int64)
            self.history_dt = np.asarray(batch.history_dt)[first:last].astype(np.int64)
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
    if g.history_entities is not None:
        for step, (entity, dt) in enumerate(zip(g.history_entities.tolist(), g.history_dt.tolist())):
            names[entity - g.begin] = f"history:{dt}#{step}"
    # goal, subgoal, history target and action entities are the target of the tuple they stand for
    for relation, name in enumerate(g.names):
        roles = g.slot_roles[relation]
        target = next((r for r in (_GOAL_TARGET_SLOT, _SUBGOAL_TARGET_SLOT, _HISTORY_TARGET_SLOT) if r in roles), None)
        if target is not None:
            position = roles.index(target)
            sign, predicate, suffix = _TARGET_RELATION.match(name).groups()
            for local, row in enumerate(g.tuples(relation)):
                literal = f"[{sign}]({' '.join([predicate] + _argument_names(g, relation, local, objects))})"
                if target == _HISTORY_TARGET_SLOT:
                    names[int(row[position])] = f"{names[int(row[roles.index(_HISTORY_SLOT)])]}:{literal}"
                else:
                    names[int(row[position])] = f"{literal}{suffix}"
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
    graph = _check(encoder, batch, graph)
    return _names(encoder, batch, graph)[1]


def describe(encoder, batch, graph):
    """A GraphDescription of graph ``graph`` of ``batch``: its entities and its relations' tuples by name."""
    graph = _check(encoder, batch, graph)
    g, names = _names(encoder, batch, graph)
    entities = tuple((g.entity_role_names[int(role)], name) for role, name in zip(g.role.tolist(), names))
    relations = {}
    for relation, name in enumerate(g.names):
        rows = g.tuples(relation)
        if len(rows):
            relations[name] = tuple(tuple(names[int(e)] for e in row) for row in rows)
    return GraphDescription(graph, g.task_id, entities, relations)
