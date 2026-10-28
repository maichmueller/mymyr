"""Helpers of the parity tests of the HGraph view against mifrost's recorded HGraph encodings.

A record of the HGraph configs (C9A to C9C) lists the nodes of each node type by name (an atom node is the atom written
as `(on a b)`, a goal literal `[+](on c b)[g]`, the nodes of `_symbol_` are the entities) and the edges of each edge type
as pairs of node names. The view of the native batch (``mymyr.encoding.hgraph``) is turned into the same shape: the
atom nodes are named by their tuples' entities, which the view's own edges give, so that a stride or an offset of the
view that is off by one changes a name. Both become a ``Graph`` of ``encoding_parity``: its catalog the edge types as
``source|position|destination`` (both directions, sorted), its entities the nodes by node type, its tuples the edges.

- ``HGRAPH_CONFIGS``: the native options of each config.
- ``HGRAPH_REGISTRY``: the differences the native view has from the reference on purpose, each with its reason, its
  scope and the cases that need it; ``compare_hgraph`` reconciles only inside the scope of an entry and reports every
  other mismatch.
"""

import functools
import re
from dataclasses import dataclass, replace

import numpy as np

from encoding_parity import (
    AUXILIARY, Context, Difference, Graph, base_predicate, compare_exactly, fixture_cases, fixture_graph, load, native_graph,
    read, record_context, replay,
)

SYMBOL = "_symbol_"
# The auxiliary entity in the reference (its node of the nullary predicates)
REFERENCE_AUXILIARY = "![nullary_symbol]!"
TARGET = re.compile(r"^target:\d+\|")
ATOM_NAME = re.compile(r"^(\[[+-]\])?(.+?)((?:\[[^\[\]]+\])*)$")

# Whether the config compares the action relations (the reference was given the actions of its record).
HGRAPH_CONFIGS = {
    "C9A": dict(layout="relation", nullary="omit", include_actions=False),
    "C9B": dict(layout="relation", nullary="auxiliary", include_actions=False),
    "C9C": dict(layout="relation", nullary="omit", include_actions=True, target_sources=("action",)),
}
# The lanes of the native catalog that the configs have relations of (no subgoal layers, no history)
HGRAPH_LANES = ("fluent", "static", "derived", "goal", "goal_derivation", "action")


def is_hgraph(config):
    return config in HGRAPH_CONFIGS


def hgraph_cases():
    """Every (fixture path, config id) of the HGraph configs, in file and config order."""
    return [(path, config) for path, config in fixture_cases() if is_hgraph(config)]


# ------------------------------------------------------------------------------------------------ graphs
def _edge_key(name):
    source, position, destination = name.split("|")
    return source.encode(), position.encode(), destination.encode()


def reference_graph(fixture, record):
    """The reference's HGraph of a record as a ``Graph``: node names without the index of a target (`target:0|` in
    front of the name of an entity that was a target), edge types sorted."""
    graph = fixture_graph(fixture, record)
    entities = [(kind, TARGET.sub("", node)) for kind, node in graph.entities]
    tuples = {edge: [tuple(TARGET.sub("", node) for node in pair) for pair in pairs]
              for edge, pairs in graph.tuples.items()}
    return Graph(sorted(graph.catalog, key=lambda c: _edge_key(c[0])), entities, tuples)


def atom_name(relation, arguments):
    """The name of the atom node of a tuple: the relation's name with its predicate or action schema applied to the
    arguments, `[+](on c b)[g]` for `[+]on[g]`."""
    sign, base, decoration = ATOM_NAME.match(relation).groups()
    return f"{sign or ''}({' '.join([base, *arguments])}){decoration}"


def native_hgraph(encoder, batch, row, task, view):
    """Graph ``row`` of the HGraph ``view`` of ``batch`` in the shape of ``reference_graph``. The `_symbol_` nodes are
    named as ``native_graph`` names the entities, the atom nodes by their tuples' entities (the view's edges)."""
    names = [name for _, name in native_graph(encoder, batch, row, [task], HGRAPH_LANES).entities]
    first = int(np.asarray(view.ptr(SYMBOL))[row])
    entities = []
    edges = {}
    catalog = []
    relation_roles = encoder.relation_slot_roles
    for relation, lane in zip(encoder.relation_names, encoder.relation_lanes):
        if lane not in HGRAPH_LANES:
            continue
        r = encoder.relation_id(relation)
        begin, end = (int(x) for x in np.asarray(view.ptr(relation))[row : row + 2])
        columns = [
            np.asarray(view.edge_index_1((relation, str(k), SYMBOL)))[begin:end] - first
            for k in range(len(relation_roles[r]))
        ]
        arguments = [k for k, role in enumerate(relation_roles[r]) if role in (0,)]  # argument slots
        atoms = [atom_name(relation, [names[int(columns[k][i])] for k in arguments]) for i in range(end - begin)]
        entities += [(relation, atom) for atom in atoms]
        for k, column in enumerate(columns):
            forward, backward = f"{relation}|{k}|{SYMBOL}", f"{SYMBOL}|{k}|{relation}"
            catalog += [forward, backward]
            if atoms:
                edges[forward] = [(atom, names[int(e)]) for atom, e in zip(atoms, column)]
                edges[backward] = [(names[int(e)], atom) for atom, e in zip(atoms, column)]
    entities += [(SYMBOL, name) for name in names]
    entities.sort(key=lambda e: e[0].encode())  # by node type, stable within it
    return Graph([(n, 2) for n in sorted(catalog, key=_edge_key)], entities, edges)


# ------------------------------------------------------------------------------------------------ registry
def _sorted_atoms_and_edges(expected, actual, context):
    """Every node type but `_symbol_` as a sorted list of nodes, and every edge type as a sorted list of edges."""

    def order(graph):
        symbols = [e for e in graph.entities if e[0] == SYMBOL]
        atoms = sorted((e for e in graph.entities if e[0] != SYMBOL), key=lambda e: (e[0].encode(), e[1]))
        entities = sorted(atoms + symbols, key=lambda e: e[0].encode())
        return replace(graph, entities=entities, tuples={n: sorted(ts) for n, ts in graph.tuples.items()})

    return order(expected), order(actual)


def _sorted_entities_of(wanted):
    """The `_symbol_` nodes for which ``wanted(name, context)`` holds, in name order on both sides."""

    def reconcile(expected, actual, context):
        def order(graph):
            positions = [i for i, (kind, name) in enumerate(graph.entities) if kind == SYMBOL and wanted(name, context)]
            entities = list(graph.entities)
            for i, name in zip(positions, sorted(graph.entities[i][1] for i in positions)):
                entities[i] = (SYMBOL, name)
            return replace(graph, entities=entities)

        return order(expected), order(actual)

    return reconcile


def _nullary_symbol(expected, actual, context):
    """The reference's node for the nullary atoms is `![nullary_symbol]!`; the native one is the graph's auxiliary
    entity."""

    def name(node):
        return AUXILIARY if node == REFERENCE_AUXILIARY else node

    entities = [(kind, name(node)) for kind, node in expected.entities]
    tuples = {edge: [tuple(name(node) for node in pair) for pair in pairs] for edge, pairs in expected.tuples.items()}
    return replace(expected, entities=entities, tuples=tuples), actual


HGRAPH_REGISTRY = (
    Difference(
        "nullary_symbol",
        "The relations of nullary predicates have the edge type (relation, 0, `_symbol_`) to the graph's auxiliary "
        "entity, a `_symbol_` node that exists exactly when the graph has a nullary tuple and is named by its role. "
        "mifrost adds a node named `![nullary_symbol]!` for them (and cannot encode a nullary goal literal).",
        "the `_symbol_` node named `![nullary_symbol]!` and the edges over it: renamed to the auxiliary entity",
        (("f01", "C9B"), ("f02", "C9B")),
        _nullary_symbol,
    ),
    Difference(
        "atom_and_edge_order",
        "mifrost lists the atoms of a node type and the edges of an edge type in its backend's order; the native "
        "view's atoms are the tuples in ascending order of their entity ids and its edges follow them.",
        "the order of the atom nodes within each node type other than `_symbol_` and of the edges within each edge type: "
        "compared as sorted lists",
        (("f01", "C9A"), ("f02", "C9A"), ("f09", "C9A")),
        _sorted_atoms_and_edges,
    ),
    Difference(
        "object_order",
        "The objects are `_symbol_` nodes in the task's order in the native view (the entity ids of the batch) and in "
        "name order in mifrost.",
        "the order of the `_symbol_` nodes that are objects of the task: compared in name order",
        (("f01", "C9A"), ("f02", "C9A")),
        _sorted_entities_of(lambda name, context: name in context.task.objects),
    ),
    Difference(
        "action_entity_order",
        "Action entities are numbered in the canonical order of the expansion rows (action schema, then binding); "
        "mifrost numbers them in the order of the actions it was given.",
        "the `_symbol_` nodes that are action entities (neither an object nor the auxiliary entity): compared in name "
        "order",
        (("f02", "C9C"),),
        _sorted_entities_of(lambda name, context: name not in context.task.objects and name != AUXILIARY),
    ),
)
HGRAPH_NAMES = tuple(d.name for d in HGRAPH_REGISTRY)


def compare_hgraph(expected, actual, context, allowed=HGRAPH_NAMES):
    """Every mismatch of the native HGraph ``actual`` against the reference's ``expected`` once the allowed differences
    (names from ``HGRAPH_REGISTRY``; all by default, none for an exact comparison) are reconciled."""
    for difference in HGRAPH_REGISTRY:
        if difference.name in allowed:
            expected, actual = difference.reconcile(expected, actual, context)
    return compare_exactly(expected, actual)


def needed_hgraph_differences(expected, actual, context):
    """The names of the allowed differences without which the comparison finds other mismatches."""
    with_all = set(compare_hgraph(expected, actual, context))
    return {name for name in HGRAPH_NAMES
            if set(compare_hgraph(expected, actual, context, [n for n in HGRAPH_NAMES if n != name])) != with_all}


# ------------------------------------------------------------------------------------------------ encoding
def encode_hgraph_records(task, config, rows):
    """The encoder and the batch of the rows (the states of the records of ``config``), all in one batch."""
    from mymyr import encoding, rl

    options = HGRAPH_CONFIGS[config]
    encoder = encoding.FlatRelationEncoder(task, **options)
    actions = rl.expand(rl.as_table(task), rows) if options.get("include_actions") else None
    return encoder, encoder.encode(rows, actions=actions)


@dataclass
class HGraphCase:
    """The comparison of one (fixture, config): ``mismatches`` by state id, and the allowed differences it needed."""

    mismatches: dict
    needed: set


@functools.cache
def run_hgraph_case(path, config):
    """Encode the recorded states of ``config`` natively and compare the HGraph view of each against the record."""
    from mymyr import encoding

    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, batch = encode_hgraph_records(task, config, rows)
    view = encoding.hgraph(batch)
    result = HGraphCase({}, set())
    for row, record in enumerate(records):
        context = record_context(fixture, task, states[record["state"]], config, record)
        expected = reference_graph(fixture, record)
        actual = native_hgraph(encoder, batch, row, task, view)
        result.mismatches[record["state"]] = compare_hgraph(expected, actual, context)
        result.needed |= needed_hgraph_differences(expected, actual, context)
    return result
