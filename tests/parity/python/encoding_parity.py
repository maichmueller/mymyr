"""Helpers of the parity tests of the native encoder against mifrost's recorded encodings.

- The fixtures (``tests/parity/data/encoding/mifrost``, written by ``tests/parity/tools/export_mifrost_encoding_fixture.py``): loading,
  and the replay of the recorded action paths in mymyr.
- A canonical form of a graph: entities as ``(role, name)`` in id order, tuples as lists of entity names, the catalog
  as an ordered list of ``(relation, arity)``. ``fixture_graph`` builds it from a mifrost record, ``native_graph`` from
  a batch of the native encoder.
- ``REGISTRY``: the differences that the native encoder has from mifrost on purpose. Each entry says why, which
  fixture cases exercise it and where it applies, and rewrites a pair of graphs only inside that scope. A mismatch
  that no entry covers is a failure, and an entry that no case needs is reported as stale.
- ``compare``: every mismatch of a graph against its expectation, not only the first.
- ``SUPPORTED``, ``PENDING_CONFIGS`` and ``PENDING_SOURCES``: which configurations and which relation sources of the
  fixtures the native encoder is compared on, and for the others which lane turns them on.
- ``random_walk_rows``: seeded, distinct states for property tests.
"""

import functools
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pytest

import mymyr
from parity_support import DATA, ROOT

FIXTURES = sorted((DATA / "encoding/mifrost").glob("f*.json"))
HELPER = re.compile(r"axiom_\d+$")
DECORATION = re.compile(r"^(?:\[[+-]\])?(.+?)(?:\[[a-z+\-]+\])*$")
FLAT_ENCODERS = ("FlatRelationEncoder", "FlatHorizonEncoder", "FlatRootedHorizonEncoder", "FlatTransitionEncoder",
                 "FlatTransitionEffectsEncoder")
AUXILIARY = "<auxiliary>"
# The config recorded with nullary relations kept: its catalog lists the nullary relations of the domain.
NULLARY_CONFIG = "C2"


# ------------------------------------------------------------------------------------------------ fixtures
def read(path):
    return json.loads(path.read_text())


def load(path):
    """The fixture and its task, loaded in mymyr; skips without the PDDL front end."""
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    fixture = read(path)
    task = mymyr.Task.from_pddl(ROOT / fixture["task"]["domain"], ROOT / fixture["task"]["problem"])
    return fixture, task


def replay(task, path):
    """The state reached from the initial state by the action names of ``path``."""
    state = task.initial_state
    for name in path:
        state = task.apply(state, name)
    return state


def atoms(values):
    return sorted(str(a) for a in values)


def state_path(fixture, state_id):
    return next(s["path"] for s in fixture["states"] if s["id"] == state_id)


def helper_column_order(task, fixture, predicate):
    """For the helper predicate ``axiom_<k>``: the positions of mymyr's arguments that give the argument order of the
    fixture's (pymimir's) predicate. ``None`` if the fixture records no such predicate, or one without parameters.

    mymyr orders the parameters of a helper predicate by name, pymimir by its own traversal; the parameters carry the
    same names, so the arguments are mapped by name."""
    recorded = fixture["task"]["helper_predicates"].get(predicate)
    if not recorded:
        return None
    own = next(p for p in task.formalism.predicates if p.name == predicate)
    names = [parameter.name for parameter in own.parameters]
    assert sorted(names) == sorted(recorded), predicate
    return [names.index(name) for name in recorded]


def helper_arguments_in_fixture_order(task, fixture, atom):
    """The atom ``(axiom_<k> ...)`` with its arguments in the fixture's argument order. ``None`` if the predicate is a
    helper of the problem as written that the fixture's task does not have."""
    predicate, *arguments = atom.strip("()").split()
    if predicate not in fixture["task"]["helper_predicates"]:
        return None
    order = helper_column_order(task, fixture, predicate)
    if order is not None:
        arguments = [arguments[i] for i in order]
    return "(" + " ".join([predicate] + arguments) + ")"


def derived_in_fixture_order(task, fixture, state):
    output = []
    for atom in atoms(state.derived_atoms()):
        if HELPER.match(atom.strip("()").split()[0]):
            atom = helper_arguments_in_fixture_order(task, fixture, atom)
            if atom is None:
                continue
        output.append(atom)
    return sorted(output)


def base_predicate(relation):
    """The predicate or schema a catalog relation is named after (decorations like ``[+]`` and ``[g]`` removed)."""
    return DECORATION.match(relation).group(1)


def flat_catalogs(fixture):
    for config in fixture["configs"].values():
        if config["encoder"] in FLAT_ENCODERS:
            yield config, fixture["catalogs"][config["catalog"]]


def hgraph_catalogs(fixture):
    for config in fixture["configs"].values():
        if config["encoder"] == "HGraphEncoder":
            yield config, fixture["catalogs"][config["catalog"]]


def fixture_cases():
    """Every (fixture path, config id) of the fixtures, in file and config order."""
    return [(path, config) for path in FIXTURES for config in read(path)["configs"]]


# ------------------------------------------------------------------------------------------------ graphs
@dataclass
class Graph:
    """One encoded graph by names. ``catalog``: ordered ``(relation, arity)`` with the arity of the atoms (a nullary
    relation has arity 0). ``entities``: ``(role, name)`` by entity id. ``tuples``: relation to its tuples of entity
    names, in the order of the encoding; relations without tuples are absent. ``unordered``: the relations whose argument
    rows are not strictly ascending in entity ids (the native encoding's canonical order); empty for a fixture."""

    catalog: list
    entities: list
    tuples: dict
    unordered: tuple = ()


@dataclass(frozen=True)
class Mismatch:
    """One difference between two graphs: ``kind`` is ``catalog``, ``entity``, ``tuple`` or ``order``, ``where`` the relation (or
    ``None``) and ``detail`` says what is missing, unexpected or misplaced."""

    kind: str
    where: str | None
    detail: str

    def __str__(self):
        return f"{self.kind}{'' if self.where is None else ' ' + self.where}: {self.detail}"


@dataclass(frozen=True)
class Context:
    """What the allowed differences may look at: the fixture, its task, the state record of the graph, the config and
    the goal literals the record was encoded with (``None``: the task's goal)."""

    fixture: dict
    task: object
    state: dict
    config: str
    goals: tuple | None = None


def fixture_graph(fixture, record, sources=None):
    """The graph of a fixture record. ``sources`` restricts the catalog to relations of those sources (``None``: all).
    An HGraph record becomes a graph whose entities are its nodes and whose relations are its edge types."""
    config = fixture["configs"][record["config"]]
    catalog = fixture["catalogs"][config["catalog"]]
    outputs = record["outputs"]
    assert "error" not in outputs, outputs["error"]
    if config["encoder"] == "HGraphEncoder":
        hgraph = outputs["hgraph"]
        names = [f"{e['src']}|{e['rel']}|{e['dst']}" for e in catalog["edge_types"]]
        entities = [(kind, name) for kind in catalog["node_types"] for name in hgraph["nodes"].get(kind, [])]
        tuples = {edge: [tuple(pair) for pair in pairs] for edge, pairs in hgraph["edges"].items()}
        assert set(tuples) <= set(names)
        return Graph([(n, 2) for n in names], entities, tuples)
    wanted = [(n, a) for n, a, s in zip(catalog["names"], catalog["logical_arities"], catalog["sources"])
              if sources is None or s in sources]
    cursors = Counter()
    entities = []
    for role, count in outputs["entities"]["runs"]:
        names = outputs["entities"]["by_role"][role]
        entities += [(role, names[cursors[role] + i]) for i in range(count)]
        cursors[role] += count
    kept = {n for n, _ in wanted}
    tuples = {}
    for name, relation in outputs["relations"].items():
        assert name in catalog["names"], name
        if name in kept:
            tuples[name] = [tuple(t) for t in fixture["tuples"][relation["emit"]]]
    return Graph(wanted, entities, tuples)


def entity_name(role, ref, task):
    """The name of an entity of the native encoding, from its role and reference. Objects are named as in the task and
    the auxiliary entity ``AUXILIARY``; the goal entities are named by ``native_graph`` from their tuples."""
    if role == "object":
        return task.objects[ref]
    if role == "auxiliary":
        return AUXILIARY
    raise NotImplementedError(f"no naming rule for the entity role {role!r}")


GOAL_RELATION = re.compile(r"^\[([+-])\](.+)\[g\]$")


def native_graph(encoder, batch, row, tasks, lanes=None):
    """Graph ``row`` of a batch of ``encoder``, restricted to the relations of the given lanes (``None``: all).
    ``tasks`` are the tasks of the batch's table, indexed by task id. A goal entity is named
    ``[+](predicate arguments)[g]`` after the plain goal tuple it stands for (the entity's reference is the tuple's
    index in the batch)."""
    task = tasks[int(batch.task_ids[row])]
    role_names = batch.schema.entity_role_names
    begin, end = (int(x) for x in batch.entity_offsets[row : row + 2])
    roles = [role_names[int(role)] for role in batch.entity_role[begin:end]]
    refs = [int(ref) for ref in batch.entity_ref[begin:end]]
    names = encoder.relation_names
    arities = encoder.relation_arities
    relations = batch.num_relations
    graphs = batch.num_graphs
    by_relation = encoder.options["layout"] == "relation"

    def segment(relation):
        return relation * graphs + row if by_relation else row * relations + relation

    entities = [None if role == "goal" else (role, entity_name(role, ref, task)) for role, ref in zip(roles, refs)]
    rows = {}
    for relation in range(relations):
        count = int(batch.relation_counts[row, relation])
        if not count:
            continue
        cell = segment(relation)
        first, last = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        assert last - first == count * arities[relation]
        arguments = (batch.relation_args[first:last] - begin).reshape(count, arities[relation])
        assert ((0 <= arguments) & (arguments < end - begin)).all(), f"{names[relation]}: argument outside the graph"
        rows[relation] = [tuple(int(x) for x in key) for key in arguments]
        goal = GOAL_RELATION.match(names[relation])
        if goal and encoder.relation_lanes[relation] == "goal" and "goal" in roles:
            identifiers = int(batch.relation_offsets[cell]) + np.arange(count)
            for identifier, key in zip(identifiers, rows[relation]):
                position = next(i for i, (role, ref) in enumerate(zip(roles, refs))
                                if role == "goal" and ref == int(identifier))
                assert key[0] == position, f"{names[relation]}: the first argument is not the goal entity"
                arguments_ = [entities[x][1] for x in key[1:] if roles[x] != "auxiliary"]
                entities[position] = ("goal", f"[{goal.group(1)}]({' '.join([goal.group(2)] + arguments_)})[g]")
    assert all(e is not None for e in entities), "a goal entity belongs to no goal tuple"
    selected = [i for i, lane in enumerate(encoder.relation_lanes) if lanes is None or lane in lanes]
    tuples = {}
    unordered = []
    for relation in selected:
        if relation not in rows:
            continue
        keys = rows[relation]
        if any(a >= b for a, b in zip(keys, keys[1:])):
            unordered.append(names[relation])
        tuples[names[relation]] = [tuple(entities[x][1] for x in key) for key in keys]
    catalog = [(names[i], encoder.relation_logical_arities[i]) for i in selected]
    return Graph(catalog, entities, tuples, tuple(unordered))


# ------------------------------------------------------------------------------------------------ comparison
def _differences(kind, where, expected, actual):
    """Mismatches between two lists: the elements missing from or unexpected in ``actual`` and, if the multisets are
    equal, the positions that hold different elements."""
    missing = Counter(expected) - Counter(actual)
    unexpected = Counter(actual) - Counter(expected)
    found = [Mismatch(kind, where, f"missing {item!r}" + (f" x{n}" if n > 1 else "")) for item, n in missing.items()]
    found += [Mismatch(kind, where, f"unexpected {item!r}" + (f" x{n}" if n > 1 else ""))
              for item, n in unexpected.items()]
    if not found and expected != actual:
        found = [Mismatch(kind, where, f"position {i}: expected {e!r}, got {a!r}")
                 for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    return found


def compare_exactly(expected, actual):
    """Every mismatch of ``actual`` against ``expected``: the catalog and the entities as ordered lists, the tuples of
    every relation as ordered lists."""
    found = _differences("catalog", None, expected.catalog, actual.catalog)
    found += [Mismatch("order", name, "tuples are not strictly ascending in entity ids") for name in actual.unordered]
    found += _differences("entity", None, expected.entities, actual.entities)
    for relation in sorted(set(expected.tuples) | set(actual.tuples), key=lambda n: n.encode()):
        found += _differences("tuple", relation, expected.tuples.get(relation, []), actual.tuples.get(relation, []))
    return found


@dataclass(frozen=True)
class Difference:
    """An allowed difference of the native encoder from mifrost.

    ``name``: its identifier. ``reason``: why the native encoder differs. ``scope``: where it applies, the narrowest
    predicate that does. ``cases``: (fixture file prefix, config) pairs that need it; each must. ``reconcile``: maps
    the pair (mifrost's graph, native graph) to the pair that is compared exactly, touching only its scope."""

    name: str
    reason: str
    scope: str
    cases: tuple
    reconcile: Callable


def _map_tuples(graph, function):
    return replace(graph, tuples={name: function(name, ts) for name, ts in graph.tuples.items()})


def _sorted_tuples(expected, actual, context):
    def order(_, ts):
        return sorted(ts)

    return _map_tuples(expected, order), _map_tuples(actual, order)


def _objects(graph):
    return [name for role, name in graph.entities if role == "object"]


def _sorted_objects(expected, actual, context):
    """Applies only when the native objects are in task order and the reference's are in name order."""
    if _objects(actual) != list(context.task.objects) or _objects(expected) != sorted(_objects(expected)):
        return expected, actual

    def order(graph):
        positions = [i for i, (role, _) in enumerate(graph.entities) if role == "object"]
        entities = list(graph.entities)
        for i, name in zip(positions, sorted(graph.entities[i][1] for i in positions)):
            entities[i] = ("object", name)
        return replace(graph, entities=entities)

    return order(expected), order(actual)


def _without_equality(expected, actual, context):
    if SUPPORTED[context.config].get("equality", False):
        return expected, actual
    kept = {name: ts for name, ts in expected.tuples.items() if base_predicate(name) != "="}
    catalog = [c for c in expected.catalog if base_predicate(c[0]) != "="]
    return replace(expected, catalog=catalog, tuples=kept), actual


def _holding_nullary_atoms(context):
    state = context.state
    atoms_ = list(state["fluent"]) + list(state["derived"]) + list(context.fixture["static_atoms"])
    return {a.strip("()") for a in atoms_ if " " not in a.strip("()")}


def parse_literal(text):
    """(positive, predicate, argument names) of a ground literal ``(p a b)`` or ``(not (p a b))``."""
    text = text.strip()
    positive = not text.startswith("(not ")
    inner = text[5:-1] if not positive else text
    predicate, *arguments = inner.strip("()").split()
    return positive, predicate, tuple(arguments)


def _present_nullary_goal_relations(context, names):
    """The nullary goal relations (among ``names``) that have their tuple: from the goal literals the record was
    encoded with, else from the same state's record of the config that keeps nullary relations."""
    goal_names = [n for n in names if GOAL_RELATION.match(n.split("[sat]")[0].split("[unsat]")[0])]
    if context.goals is None:
        record = next(r for r in context.fixture["encodings"]
                      if r["config"] == NULLARY_CONFIG and r["state"] == context.state["id"])
        relations = record["outputs"]["relations"]
        return {n for n in goal_names if n in relations and context.fixture["tuples"][relations[n]["emit"]]}
    holding = _holding_nullary_atoms(context)
    present = set()
    for literal in context.goals:
        positive, predicate, arguments = parse_literal(literal)
        if arguments:
            continue
        sign = "+" if positive else "-"
        holds = predicate in holding
        for name in (f"[{sign}]{predicate}[g]", f"[{sign}]{predicate}[g][sat]" if holds == positive else
                     f"[{sign}]{predicate}[g][unsat]"):
            if name in goal_names:
                present.add(name)
    return present


def kind_of(relation):
    """The decoration of a relation name after its predicate: ``[g]``, ``[g][sat]``, ``[g][unsat]`` or none."""
    for suffix in ("[g][sat]", "[g][unsat]", "[g]"):
        if relation.endswith(suffix):
            return suffix
    return ""


def _nullary_over_auxiliary(expected, actual, context):
    """The reference's catalog omits nullary relations or lists them without tuples. The native catalog has every
    nullary relation of the domain (the reference lists them in its config with ``ignore_zero_arity_relations`` off),
    and each nullary atom that is true, and each nullary goal literal, is the unary tuple over the auxiliary entity
    (after the goal entity, when the goal relations have targets), which the graph has exactly when there is one.
    The expectation is rebuilt that way from the atoms of the recorded state and the goal's nullary tuples of the
    config that keeps them; a nullary relation that the native catalog lacks, or has in addition, stays a mismatch."""
    fixture = context.fixture
    catalog = fixture["catalogs"][fixture["configs"][NULLARY_CONFIG]["catalog"]]
    own = fixture["catalogs"][fixture["configs"][context.config]["catalog"]]["names"]
    kinds = {kind_of(name) for name in own}
    nullary = [(name, 0) for name, arity, source in zip(catalog["names"], catalog["logical_arities"], catalog["sources"])
               if arity == 0 and source in COMPARED_SOURCES and kind_of(name) in kinds]
    names = {name for name, _ in nullary}
    holding = _holding_nullary_atoms(context)
    expected_catalog = sorted([c for c in expected.catalog if c[1] != 0] + nullary, key=lambda c: c[0].encode())
    tuples = {name: ts for name, ts in expected.tuples.items() if name not in names}
    goal_present = _present_nullary_goal_relations(context, names)
    present = [name for name, _ in nullary if name in holding or name in goal_present]
    targets = bool(SUPPORTED[context.config].get("target_sources"))
    entities = list(expected.entities)
    if present:
        last = max((i for i, (role, _) in enumerate(entities) if role == "object"), default=-1)
        entities.insert(last + 1, ("auxiliary", AUXILIARY))
    for name in present:
        goal = GOAL_RELATION.match(name)
        if targets and goal and name in goal_present:
            entity = f"[{goal.group(1)}]({goal.group(2)})[g]"
            last = max(i for i, (role, _) in enumerate(entities) if role in ("object", "auxiliary", "goal"))
            entities.insert(last + 1, ("goal", entity))
            tuples[name] = [(entity, AUXILIARY)]
        else:
            tuples[name] = [(AUXILIARY,)]
    return replace(expected, catalog=expected_catalog, entities=entities, tuples=tuples), actual


def _without_virtual_nodes(expected, actual, context):
    """The reference's graph has the predicate virtual nodes and the action entities its config asks for. The native
    encoder is compared with both off: their entities, and the predicate slot of each tuple that carries one, are
    projected away from the reference."""
    options = SUPPORTED[context.config]
    if options.get("predicate_virtual_nodes", True) or options.get("include_actions", True):
        return expected, actual
    fixture = context.fixture
    catalog = fixture["catalogs"][fixture["configs"][context.config]["catalog"]]
    predicate = catalog["slot_role_names"].index("predicate_slot")
    slots = dict(zip(catalog["names"], catalog["slots"]))

    def project(name, ts):
        keep = [i for i, role in enumerate(slots[name]) if role != predicate]
        return ts if len(keep) == len(slots[name]) else [tuple(t[i] for i in keep) for t in ts]

    entities = [e for e in expected.entities if e[0] not in ("predicate_virtual", "action")]
    return replace(_map_tuples(expected, project), entities=entities), actual


def _sorted_goal_entities(expected, actual, context):
    """Goal entities are numbered in canonical order (relation, then tuple) natively and in the order of the goal
    literals in mifrost: compared in name order."""
    def order(graph):
        positions = [i for i, (role, _) in enumerate(graph.entities) if role == "goal"]
        entities = list(graph.entities)
        for i, name in zip(positions, sorted(graph.entities[i][1] for i in positions)):
            entities[i] = ("goal", name)
        return replace(graph, entities=entities)

    return order(expected), order(actual)


def _helper_argument_order(expected, actual, context):
    def permute(name, ts):
        order = helper_column_order(context.task, context.fixture, name) if HELPER.match(name) else None
        return ts if order is None else [tuple(t[i] for i in order) for t in ts]

    return expected, _map_tuples(actual, permute)


REGISTRY = (
    Difference(
        "virtual_nodes_and_actions_off",
        "The reference's config asks for predicate virtual nodes and action targets, which the native encoder is "
        "compared without; the entities of both and the predicate slot of the tuples are projected out of the "
        "reference.",
        "the entities of roles predicate_virtual and action, and the predicate slot columns, only while the native "
        "options turn both off",
        (("f03", "C3"),),
        _without_virtual_nodes,
    ),
    Difference(
        "object_order",
        "Object ids follow the order of the task (constants first, then the objects as declared); mifrost sorts them "
        "by name.",
        "the entities of role object: compared in name order",
        (("f04", "C1"), ("f01", "C1")),
        _sorted_objects,
    ),
    Difference(
        "equality_absent",
        "The equality predicate `=` has no relations (state or goal) in the native encoder unless it is asked for; "
        "mifrost always has them.",
        "the relations of the predicate `=` (state and goal), and only while the native catalog lacks them",
        (("f05", "C1"),),
        _without_equality,
    ),
    Difference(
        "nullary_over_auxiliary",
        "Nullary atoms are unary tuples over one auxiliary entity that the graph has exactly when such an atom is "
        "true; mifrost drops nullary relations or lists them without an entity.",
        "the relations of arity 0 and the entity of role auxiliary",
        (("f10", "C1"), ("f08", "C1")),
        _nullary_over_auxiliary,
    ),
    Difference(
        "goal_entity_order",
        "Goal entities are numbered in the canonical order of the goal tuples (relation, then tuple); mifrost numbers "
        "them in the order of the goal literals.",
        "the entities of role goal: compared in name order",
        (("f03", "C3"),),
        _sorted_goal_entities,
    ),
    Difference(
        "helper_argument_order",
        "The arguments of a helper predicate `axiom_<k>` are ordered by parameter name in mymyr and by pymimir's "
        "traversal in mifrost's input; the fixture records pymimir's parameter names and the native arguments are "
        "mapped onto them.",
        "the relations named `axiom_<k>` that the fixture records, by column permutation only",
        (("f05", "C1"),),
        _helper_argument_order,
    ),
    Difference(
        "tuple_order",
        "mifrost emits the tuples of a relation in its backend's order; the native encoder's are strictly ascending "
        "in entity ids, which ``native_graph`` checks.",
        "the order of the reference's tuples within each relation: compared as sorted lists",
        (("f01", "C1"), ("f05", "C1")),
        _sorted_tuples,
    ),
)
NAMES = tuple(d.name for d in REGISTRY)


def compare(expected, actual, context, allowed=NAMES):
    """Every mismatch of ``actual`` against ``expected`` once the allowed differences (names from ``REGISTRY``; all by
    default, none for an exact comparison) are reconciled."""
    for difference in REGISTRY:
        if difference.name in allowed:
            expected, actual = difference.reconcile(expected, actual, context)
    return compare_exactly(expected, actual)


def needed_differences(expected, actual, context):
    """The names of the allowed differences without which the comparison finds other mismatches."""
    with_all = set(compare(expected, actual, context))
    return {name for name in NAMES
            if set(compare(expected, actual, context, [n for n in NAMES if n != name])) != with_all}


# ------------------------------------------------------------------------------------------------ configurations
# The native options of each compared config; lanes the encoder does not have yet are switched off.
SUPPORTED = {
    "C1": dict(include_actions=False),
    "C2": dict(goal_derivations=("plain", "satisfied", "unsatisfied"), nullary=True, layout="relation",
               include_actions=False),
    "C3": dict(target_sources=("goal",), include_actions=False, predicate_virtual_nodes=False),
}
# Relation sources of the fixtures' catalogs (inside a compared config) that the comparison leaves out, with the lane
# that turns them on.
PENDING_SOURCES = {
    "action": "actions",
    "history": "history",
}
COMPARED_SOURCES = ("state", "goal", "goal_derivation")
COMPARED_LANES = ("fluent", "static", "derived", "goal", "goal_derivation")
# Configs that are not compared yet, with the lanes they need.
PENDING_CONFIGS = {
    "C3N": "goals, actions, targets, virtual nodes",
    "C3D": "goals, actions, targets, virtual nodes",
    "C3S": "goals, actions, subgoals, targets, virtual nodes",
    "C3H": "goals, actions, history, targets, virtual nodes",
    "C4_L1": "goals, subgoals",
    "C4_L1N": "goals, subgoals",
    "C4_L2": "goals, subgoals",
    "C4_L2N": "goals, subgoals",
    "C5A": "goals, history, targets",
    "C5AN": "goals, history, targets",
    "C5B": "goals, history",
    "C5BN": "goals, history",
    "C7A": "goals, actions, horizon",
    "C7AN": "goals, actions, horizon",
    "C7B": "goals, horizon",
    "C7C": "goals, actions, horizon",
    "C7D": "goals, actions, horizon",
    "C7DN": "goals, actions, horizon",
    "C7E": "goals, actions, horizon",
    "C7EN": "goals, actions, horizon",
    "C7F": "goals, actions, horizon",
    "C7FN": "goals, actions, horizon",
    "C7G": "goals, actions, horizon",
    "C7H": "goals, actions, horizon",
    "C7I": "goals, horizon",
    "C7J": "goals, actions, horizon",
    "C7K": "goals, actions, horizon",
    "C7S": "goals, subgoals, horizon",
    "C7T": "goals, actions, subgoals, horizon",
    "C8A": "goals, actions, transition",
    "C8AN": "goals, actions, transition",
    "C8B": "goals, actions, transition",
    "C8BN": "goals, actions, transition",
    "C8S": "goals, actions, subgoals, transition",
    "C9A": "goals, HGraph",
    "C9B": "goals, HGraph",
    "C9C": "goals, actions, HGraph",
}


def native_encoder(task, config):
    from mymyr import encoding

    return encoding.FlatRelationEncoder(task, **SUPPORTED[config])


@dataclass
class CaseResult:
    """The comparison of one (fixture, config): ``mismatches`` by state id, and the allowed differences it needed."""

    mismatches: dict
    needed: set


def goal_masks(task, literals):
    """(goal_pos, goal_neg) masks ``[1, W]`` over the fluent atom slots of the ground literals ``literals``."""
    from mymyr import rl

    metadata = rl.as_table(task).atom_metadata()
    predicates = task.formalism.predicates
    slots = {}
    for slot, predicate in enumerate(metadata["atom_pred"]):
        arguments = tuple(task.objects[int(a)] for a in metadata["atom_args"][slot] if a >= 0)
        slots[(predicates[int(predicate)].name, arguments)] = slot
    words = int(metadata["words"][0])
    pos = np.zeros((1, words), np.uint64)
    neg = np.zeros((1, words), np.uint64)
    for literal in literals:
        positive, predicate, arguments = parse_literal(literal)
        slot = slots[(predicate, arguments)]
        (pos if positive else neg)[0, slot // 64] |= np.uint64(1) << np.uint64(slot % 64)
    return pos, neg


def encode_records(task, config, rows, records):
    """The native encoder of ``config`` and, per record, its (batch, row): the records without goal literals in one
    batch, each record with goal literals alone, with its literals as the goal."""
    encoder = native_encoder(task, config)
    placed = [None] * len(records)
    plain = [i for i, r in enumerate(records) if "goals" not in r["inputs"]]
    if plain:
        batch = encoder.encode([rows[i] for i in plain])
        for k, i in enumerate(plain):
            placed[i] = (batch, k)
    for i, record in enumerate(records):
        if "goals" in record["inputs"]:
            placed[i] = (encoder.encode([rows[i]], goals=goal_masks(task, record["inputs"]["goals"])), 0)
    return encoder, placed


@functools.cache
def run_case(path, config):
    """Encode the recorded states of ``config`` natively and compare each graph against mifrost's record."""
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, placed = encode_records(task, config, rows, records)
    result = CaseResult({}, set())
    for record, (batch, row) in zip(records, placed):
        goals = tuple(record["inputs"]["goals"]) if "goals" in record["inputs"] else None
        context = Context(fixture, task, states[record["state"]], config, goals)
        expected = fixture_graph(fixture, record, COMPARED_SOURCES)
        actual = native_graph(encoder, batch, row, [task], COMPARED_LANES)
        result.mismatches[record["state"]] = compare(expected, actual, context)
        result.needed |= needed_differences(expected, actual, context)
    return result


# ------------------------------------------------------------------------------------------------ generators
def random_walk_rows(task, rows, seed=0, max_steps=12):
    """``rows`` distinct states of ``task``: the initial state, then the states of seeded random walks from it.

    A walk takes a uniformly chosen successor until it has ``max_steps`` steps or no successor; the states it meets
    are collected in order of discovery. Raises ``ValueError`` if 200 walks in a row find no new state."""
    rng = random.Random(seed)
    found = {task.initial_state: None}
    idle = 0
    while len(found) < rows:
        before = len(found)
        state = task.initial_state
        for _ in range(max_steps):
            successors = task.successor_states(state)
            if not successors:
                break
            state = rng.choice(successors)
            found.setdefault(state, None)
            if len(found) == rows:
                break
        idle = 0 if len(found) > before else idle + 1
        if idle > 200:
            raise ValueError(f"random walks of {max_steps} steps found {len(found)} of {rows} distinct states")
    return list(found)


def random_walk_batch(tasks, rows, seed=0, max_steps=12):
    """(states, task ids) of ``rows`` distinct rows per task, shuffled together; deterministic in ``seed``."""
    rng = random.Random(seed)
    batch = [(index, state) for index, task in enumerate(tasks)
             for state in random_walk_rows(task, rows, rng.randrange(2**31), max_steps)]
    rng.shuffle(batch)
    return [state for _, state in batch], [index for index, _ in batch]
