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
  fixtures the native encoder is compared on, and for the others which lane turns them on. The horizon configs (C7) are
  encoded by FlatHorizonEncoder or FlatRootedHorizonEncoder from the dags of their records (``horizon_dag``,
  ``horizon_view``); the transition configs (C8) by FlatTransitionEncoder or FlatTransitionEffectsEncoder from the
  record's state, whose graph holds every successor of the state (the registry removes the successors a record does not
  list).
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
STATE_SOURCES = ("state", "goal", "goal_derivation")


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
    the goal literals (``None``: the task's goal), the actions (``None``: none), the subgoal layers and the history
    steps (``None``: none) the record was encoded with, and ``dropped``: the subgoal and history literals that have no
    slot in the task and so are not given to the native encoder."""

    fixture: dict
    task: object
    state: dict
    config: str
    goals: tuple | None = None
    actions: tuple | None = None
    layers: tuple | None = None
    history: tuple | None = None
    dropped: tuple = ()
    dag: dict | None = None


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


def entity_name(role, ref, task, relation_names):
    """The name of an entity of the native encoding, from its role and reference. Objects are named as in the task,
    the auxiliary entity ``AUXILIARY`` and a predicate virtual node ``predicate:<name>`` after its key relation; the
    goal and action entities are named by ``native_graph`` from their tuples."""
    if role == "object":
        return task.objects[ref]
    if role == "auxiliary":
        return AUXILIARY
    if role == "state":
        return f"target:{ref}"
    if role == "predicate_virtual":
        return f"predicate:{base_predicate(relation_names[ref])}"
    raise NotImplementedError(f"no naming rule for the entity role {role!r}")


GOAL_RELATION = re.compile(r"^\[([+-])\](.+)\[g\]$")
SUBGOAL_RELATION = re.compile(r"^\[([+-])\](.+)\[(s{1,3}g)\]$")
HISTORY_RELATION = re.compile(r"^\[([+-])\](.+)\[hist\]$")
TARGET_ROLES = ("goal", "subgoal", "action", "history_target")


def native_graph(encoder, batch, row, tasks, lanes=None):
    """Graph ``row`` of a batch of ``encoder``, restricted to the relations of the given lanes (``None``: all).
    ``tasks`` are the tasks of the batch's table, indexed by task id. A goal entity is named
    ``[+](predicate arguments)[g]`` after the plain goal tuple it stands for (the entity's reference is the tuple's
    index among the graph's tuples in graph-major order), a subgoal entity ``[+](predicate arguments)[sg]`` (``[ssg]``,
    ``[sssg]``) likewise, and a history target ``history:<dt>#<i>:[+](predicate arguments)`` after its step and the
    history tuple it stands for. A history step is ``history:<dt>#<i>``, with ``i`` its position among the graph's
    steps. An action entity is named ``(action arguments)`` after the action tuples it leads."""
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

    steps = {}
    if encoder.options["history"]:
        first, last = (int(x) for x in batch.history_offsets[row : row + 2])
        for k, (entity, dt) in enumerate(zip(batch.history_entity[first:last], batch.history_dt[first:last])):
            steps[int(entity) - begin] = f"history:{int(dt)}#{k}"
    entities = [None if role in TARGET_ROLES else
                ("history", steps[i]) if role == "history" else (role, entity_name(role, ref, task, names))
                for i, (role, ref) in enumerate(zip(roles, refs))]
    rows = {}
    graph_tuples = 0
    for relation in range(relations):
        count = int(batch.relation_counts[row, relation])
        if not count:
            continue
        cell = segment(relation)
        first_tuple, graph_tuples = graph_tuples, graph_tuples + count
        first, last = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        assert last - first == count * arities[relation]
        arguments = (batch.relation_args[first:last] - begin).reshape(count, arities[relation])
        assert ((0 <= arguments) & (arguments < end - begin)).all(), f"{names[relation]}: argument outside the graph"
        rows[relation] = [tuple(int(x) for x in key) for key in arguments]
        for pattern, role in ((GOAL_RELATION, "goal"), (SUBGOAL_RELATION, "subgoal")):
            target = pattern.match(names[relation])
            if target and encoder.relation_lanes[relation] == "goal" and role in roles:
                suffix = "g" if role == "goal" else target.group(3)
                for identifier, key in zip(first_tuple + np.arange(count), rows[relation]):
                    position = next(i for i, (r, ref) in enumerate(zip(roles, refs))
                                    if r == role and ref == int(identifier))
                    assert key[0] == position, f"{names[relation]}: the first argument is not the {role} entity"
                    arguments_ = [entities[x][1] for x in key[1:] if roles[x] not in ("auxiliary", "predicate_virtual")]
                    sign, predicate = target.group(1), target.group(2)
                    entities[position] = (role, f"[{sign}]({' '.join([predicate] + arguments_)})[{suffix}]")
        history = HISTORY_RELATION.match(names[relation])
        if history and encoder.relation_lanes[relation] == "history":
            targeted = "history_target" in roles
            for identifier, key in zip(first_tuple + np.arange(count), rows[relation]):
                step = key[1 if targeted else 0]
                assert roles[step] == "history", f"{names[relation]}: the step argument is not a history entity"
                arguments_ = [entities[x][1] for x in key[2 if targeted else 1 :]
                              if roles[x] not in ("auxiliary", "predicate_virtual")]
                literal = f"[{history.group(1)}]({' '.join([history.group(2)] + arguments_)})"
                if targeted:
                    position = next(i for i, (r, ref) in enumerate(zip(roles, refs))
                                    if r == "history_target" and ref == int(identifier))
                    assert key[0] == position, f"{names[relation]}: the first argument is not the history target"
                    entities[position] = ("history_target", f"{entities[step][1]}:{literal}")
        if encoder.relation_lanes[relation] == "action" and "action" in roles:
            for key in rows[relation]:
                assert roles[key[0]] == "action", f"{names[relation]}: the first argument is not the action entity"
                entities[key[0]] = ("action", f"({' '.join([names[relation]] + [entities[x][1] for x in key[1:]])})")
    assert all(e is not None for e in entities), "a goal, subgoal, history target or action entity belongs to no tuple"
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
    entities = [e for e in expected.entities if e != ("predicate_virtual", "predicate:=")]
    return replace(expected, catalog=catalog, entities=entities, tuples=kept), actual


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


def over_auxiliary(config):
    """Whether the native encoder of ``config`` encodes the nullary atoms over the auxiliary entity."""
    return SUPPORTED[config].get("nullary", "auxiliary") == "auxiliary"


def _nullary_over_auxiliary(expected, actual, context):
    """The reference's catalog omits nullary relations or lists them without tuples. The native catalog has every
    nullary relation of the domain (the reference lists them in its config with ``ignore_zero_arity_relations`` off),
    and each nullary atom that is true, and each nullary goal literal, is the unary tuple over the auxiliary entity
    (after the goal entity, when the goal relations have targets), which the graph has exactly when there is one.
    The expectation is rebuilt that way from the atoms of the recorded state and the goal's nullary tuples of the
    config that keeps them; a nullary relation that the native catalog lacks, or has in addition, stays a mismatch."""
    if not over_auxiliary(context.config) or context.dag is not None:
        return expected, actual
    fixture = context.fixture
    catalog = fixture["catalogs"][fixture["configs"][NULLARY_CONFIG]["catalog"]]
    own_catalog = fixture["catalogs"][fixture["configs"][context.config]["catalog"]]
    kinds = {kind_of(name) for name in own_catalog["names"]}
    nullary = [(name, 0) for name, arity, source in zip(catalog["names"], catalog["logical_arities"], catalog["sources"])
               if arity == 0 and source in STATE_SOURCES and kind_of(name) in kinds]
    names = {name for name, _ in nullary}
    holding = _holding_nullary_atoms(context)
    expected_catalog = sorted([c for c in expected.catalog if c[0] not in names] + nullary, key=lambda c: c[0].encode())
    tuples = {name: ts for name, ts in expected.tuples.items() if name not in names}
    goal_present = _present_nullary_goal_relations(context, names)
    present = [name for name, _ in nullary if name in holding or name in goal_present]
    targets = "goal" in SUPPORTED[context.config].get("target_sources", ())
    virtual = SUPPORTED[context.config].get("predicate_virtual_nodes", False)
    entities = list(expected.entities)
    if present:
        last = max((i for i, (role, _) in enumerate(entities) if role == "object"), default=-1)
        entities.insert(last + 1, ("auxiliary", AUXILIARY))
    for name in present:
        goal = GOAL_RELATION.match(name)
        if targets and goal and name in goal_present:
            entity = f"[{goal.group(1)}]({goal.group(2)})[g]"
            if ("goal", entity) not in entities:
                last = max(i for i, (role, _) in enumerate(entities) if role in ("object", "auxiliary", "goal"))
                entities.insert(last + 1, ("goal", entity))
            tuples[name] = [(entity, AUXILIARY)]
        else:
            tuples[name] = [(AUXILIARY,)]
        if virtual:
            node = ("predicate_virtual", f"predicate:{base_predicate(name)}")
            if node not in entities:
                entities.append(node)
            slot = node[1]
            tuples[name] = [t[:-1] + (slot, t[-1]) for t in tuples[name]]
    return replace(expected, catalog=expected_catalog, entities=entities, tuples=tuples), actual


HORIZON_RELATION = re.compile(r"^(?:\[([+-])\])?([^\[\]]+)((?:\[[a-z+\-]+\])*)$")


def _node_atoms(context):
    """The names of the nullary atoms that hold in each node of the record's dag (static ones included)."""
    task = context.task
    static = {a.strip("()") for a in atoms(task.formalism.static_init) if " " not in a.strip("()")}
    nodes = []
    for path in context.dag["nodes"]:
        state = replay(task, path)
        held = {a.strip("()") for a in atoms(state.atoms()) + atoms(state.derived_atoms()) if " " not in a.strip("()")}
        nodes.append(held | static)
    return nodes


def horizon_nullary_tuples(name, context, nodes):
    """The tuples of the relation ``name`` of a nullary predicate in a horizon graph, from the atoms of the nodes of
    the record's dag: a tuple is ``(target:k, auxiliary)``, with no state for a root that the policy excludes."""
    options = SUPPORTED[context.config]
    mode, root = options["horizon_mode"], options.get("root_policy", "include")
    all_nodes = mode == "full" and root != "exclude"
    lead = [(f"target:{k}", AUXILIARY) for k in range(len(nodes))]
    if root == "exclude":
        lead[0] = (AUXILIARY,)
    sign, predicate, decoration = HORIZON_RELATION.match(name).groups()
    marks = re.findall(r"\[([a-z+\-]+)\]", decoration)
    levels = {"g": 0, "sg": 1, "ssg": 2, "sssg": 3}
    level = next((levels[m] for m in marks if m in levels), None)
    rest = [m for m in marks if m not in levels]
    holds = [predicate in held for held in nodes]
    added = [h and not holds[0] for h in holds]
    deleted = [holds[0] and not h for h in holds]
    below = range(1, len(nodes))
    if level is None:
        if rest and rest != ["state"]:
            return []
        if sign:
            return [lead[k] for k in below if (added if sign == "+" else deleted)[k]]
        if rest == ["state"]:
            return [lead[k] for k in below if holds[k]]
        if rest:
            return []
        return [lead[k] for k in (range(len(nodes)) if all_nodes else [0]) if holds[k]]
    positive = sign == "+"
    literals = [goal_literals(context)] + [set(map(parse_literal, layer)) for layer in context.layers or ()]
    if level >= len(literals) or (positive, predicate, ()) not in literals[level]:
        return []
    if any(kind == "subgoal" and where == level and parse_literal(text) == (positive, predicate, ())
           for kind, where, text in context.dropped):
        return []
    if not rest:
        return [lead[0]]
    satisfied = [h == positive for h in holds]
    if rest[0] in ("sat+", "sat-"):
        becomes = (added if positive else deleted) if rest[0] == "sat+" else (deleted if positive else added)
        return [lead[k] for k in below if becomes[k]]
    wanted = rest[0] == "sat"
    if "state" in rest:
        return [lead[k] for k in below if satisfied[k] == wanted]
    return [lead[k] for k in (range(len(nodes)) if all_nodes else [0]) if satisfied[k] == wanted]


def _nullary_horizon(expected, actual, context):
    """mifrost's horizon graphs have no tuples in the relations of nullary predicates (their catalog lists them); the
    native encoder fills them as it does for a state: the atom of a node is the tuple (node, auxiliary entity), and the
    graph has the auxiliary entity exactly when such a tuple exists. The expectation fills them from the atoms of the
    record's nodes by the semantics of each relation (facts, delta, goal, satisfaction at the nodes or relative to the
    root); a nullary relation that the native catalog lacks, or has in addition, stays a mismatch."""
    if context.dag is None or not over_auxiliary(context.config):
        return expected, actual
    nodes = _node_atoms(context)
    nullary = {p.name for p in context.task.formalism.predicates if p.arity == 0}
    tuples = dict(expected.tuples)
    for name, count in expected.catalog:
        if count == 0 and HORIZON_RELATION.match(name).group(2) in nullary:
            found = horizon_nullary_tuples(name, context, nodes)
            if found:
                tuples[name] = found
            else:
                tuples.pop(name, None)
    entities = list(expected.entities)
    if any(AUXILIARY in t for ts in tuples.values() for t in ts) and ("auxiliary", AUXILIARY) not in entities:
        last = max((i for i, (role, _) in enumerate(entities) if role == "object"), default=-1)
        entities.insert(last + 1, ("auxiliary", AUXILIARY))
    return replace(expected, entities=entities, tuples=tuples), actual


ROOT_ENTITY = ("state", "_root_state_")


def _root_state_entity(expected, actual, context):
    """mifrost names the root that a policy excludes by an entity `_root_state_` that no tuple mentions: the entity is
    dropped from its graph. Applies only when no tuple of the reference has it as an argument."""
    if ROOT_ENTITY not in expected.entities or any(ROOT_ENTITY[1] in t for ts in expected.tuples.values() for t in ts):
        return expected, actual
    return replace(expected, entities=[e for e in expected.entities if e != ROOT_ENTITY]), actual


def _repeated_literal_horizon(expected, actual, context):
    """A literal that the goal and a layer, or two layers, list is in mifrost's horizon graph at the highest of its
    levels, once for each level it is listed in, in each relation family of that level (goal, satisfaction at the
    nodes, added satisfaction); the native encoder has it once at every representable listing level. The expectation
    distributes the equal tuples to those levels only when their count matches the number of listing levels."""
    if context.dag is None or not context.layers:
        return expected, actual
    levels = {}
    for literal in goal_literals(context):
        levels.setdefault(literal, set()).add(0)
    for level, layer in enumerate(context.layers, 1):
        for text in layer:
            levels.setdefault(parse_literal(text), set()).add(level)
    tuples = {name: list(ts) for name, ts in expected.tuples.items()}
    for (positive, predicate, arguments), listed in sorted(levels.items(), key=repr):
        if len(listed) < 2 or not arguments:
            continue
        sign = "+" if positive else "-"
        high = f"[{_level_suffix(max(listed))}]"
        dropped = {level for kind, level, literal in context.dropped
                   if kind == "subgoal" and parse_literal(literal) == (positive, predicate, arguments)}
        for name in [n for n in tuples if n.startswith(f"[{sign}]{predicate}{high}")]:
            moved = [t for t in tuples[name] if t[len(t) - len(arguments) :] == arguments]
            by_tuple = Counter(moved)
            if not moved or any(count != len(listed) for count in by_tuple.values()):
                continue
            tuples[name] = [t for t in tuples[name] if t not in by_tuple]
            for level in sorted(listed - dropped):
                target = f"[{sign}]{predicate}[{_level_suffix(level)}]" + name[len(f"[{sign}]{predicate}{high}") :]
                tuples.setdefault(target, [])
                tuples[target] += [t for t in by_tuple if t not in tuples[target]]
    return replace(expected, tuples={n: ts for n, ts in tuples.items() if ts}), actual


def _static_names(context):
    return {a.strip("()") for a in atoms(context.task.formalism.static_init)}


def _delta_static_relations(expected, actual, context):
    """The delta mode compares a node with its root, and static atoms never change: the native catalog has no delta
    relation (`[+]p`, `[-]p`, `[+]p[sat+]`, ...) for a static predicate. mifrost lists them, always empty. The
    expectation drops the relations of static predicates that the native catalog lacks and the reference leaves empty."""
    if context.dag is None or SUPPORTED[context.config]["horizon_mode"] != "delta":
        return expected, actual
    static = {p.name for p in context.task.formalism.predicates if p.kind == "static"}
    own = {name for name, _ in actual.catalog}
    dropped = {name for name, _ in expected.catalog if name not in own and not expected.tuples.get(name)
               and HORIZON_RELATION.match(name).group(2) in static}
    return replace(expected, catalog=[c for c in expected.catalog if c[0] not in dropped]), actual


def _never_filled_relations(expected, actual, context):
    """The catalogs of mifrost's horizons list relations that no graph has tuples in, and the native catalog leaves them
    out: the delta mode's literal-level support relations (``[+]p[sat]``, ``[-]p[unsat]``, ``[+]p[sat+]``, ``[-]p[sat-]``,
    without a goal level; the native encoder has no support literals) and the state-anchored relation of a static
    predicate in the split catalog (``fixed[state]``; the static atoms are the root's). The expectation drops these
    entries of the catalog while the reference has no tuple in them and the native catalog lacks them."""
    if context.dag is None:
        return expected, actual
    static = {p.name for p in context.task.formalism.predicates if p.kind == "static"}
    own = {name for name, _ in actual.catalog}
    levels = {"g", "sg", "ssg", "sssg"}

    def never_filled(name):
        sign, predicate, decoration = HORIZON_RELATION.match(name).groups()
        marks = re.findall(r"\[([a-z+\-]+)\]", decoration)
        if any(m in levels for m in marks):
            return False
        return (sign is not None and any(m in ("sat", "unsat", "sat+", "sat-") for m in marks)) or (
            marks == ["state"] and predicate in static)

    dropped = {name for name, _ in expected.catalog
               if name not in own and not expected.tuples.get(name) and never_filled(name)}
    return replace(expected, catalog=[c for c in expected.catalog if c[0] not in dropped]), actual


def _static_literal_at_nodes(expected, actual, context):
    """mifrost evaluates the goal at the nodes below the root without the static atoms: a static atom does not hold
    there, so a positive static literal that holds is not satisfied at them, and a negative one is satisfied. The
    native encoder evaluates every node with the instance's static atoms. The native graph is rewritten to mifrost's
    reading, for the literals over static predicates whose atom holds and the nodes below the root only: a native
    satisfaction tuple of a positive literal moves to the unsatisfied relation (or is dropped when the catalog has none),
    one of a negative literal's unsatisfied relation moves to the satisfied one."""
    if context.dag is None:
        return expected, actual
    static = _static_names(context)
    kinds = {p.name: p.kind for p in context.task.formalism.predicates}
    levels = {}
    for literal in goal_literals(context):
        levels.setdefault(literal, set()).add(0)
    for level, layer in enumerate(context.layers or (), 1):
        for text in layer:
            levels.setdefault(parse_literal(text), set()).add(level)
    tuples = {name: list(ts) for name, ts in actual.tuples.items()}
    own = {name for name, _ in actual.catalog}
    for (positive, predicate, arguments), listed in sorted(levels.items(), key=repr):
        if kinds.get(predicate) != "static" or " ".join((predicate, *arguments)) not in static:
            continue
        family = f"[{'+' if positive else '-'}]{predicate}[{_level_suffix(min(listed))}]"
        for anchor in ("", "[state]"):
            source, target = (f"{family}[sat]{anchor}", f"{family}[unsat]{anchor}")
            if not positive:
                source, target = target, source
            moved = [t for t in tuples.get(source, []) if t[len(t) - len(arguments) :] == arguments
                     and t[0].startswith("target:") and t[0] != "target:0"]
            if not moved:
                continue
            tuples[source] = [t for t in tuples[source] if t not in moved]
            if target in own:
                tuples.setdefault(target, []).extend(moved)
    return expected, replace(actual, tuples={n: ts for n, ts in tuples.items() if ts})


def _action_subset(expected, actual, context):
    """The reference was given the actions of its record; the native encoder is given every applicable action. When
    the record lists fewer, the native action entities that it does not list, and the tuples over them, are projected
    away, but only those the task says are applicable in the record's state (computed here, independently of the
    native encoder): any other native action entity stays and is a mismatch."""
    if context.actions is None:
        return expected, actual
    given = set(context.actions)
    state = replay(context.task, context.state["path"])
    applicable = {str(a) for a in context.task.applicable_actions(state)}
    dropped = {name for role, name in actual.entities if role == "action" and name not in given and name in applicable}
    if not dropped:
        return expected, actual
    entities = [e for e in actual.entities if e[1] not in dropped or e[0] != "action"]
    tuples = {n: [t for t in ts if not dropped & set(t)] for n, ts in actual.tuples.items()}
    return expected, replace(actual, entities=entities, tuples=tuples)


def _sorted_role(role):
    def reconcile(expected, actual, context):
        def order(graph):
            positions = [i for i, (r, _) in enumerate(graph.entities) if r == role]
            entities = list(graph.entities)
            for i, name in zip(positions, sorted(graph.entities[i][1] for i in positions)):
                entities[i] = (role, name)
            return replace(graph, entities=entities)

        return order(expected), order(actual)

    return reconcile


def _helper_argument_order(expected, actual, context):
    slots = int(bool(SUPPORTED[context.config].get("predicate_virtual_nodes", False)))

    def permute(name, ts):
        order = helper_column_order(context.task, context.fixture, name) if HELPER.match(name) else None
        return ts if order is None else [t[:slots] + tuple(t[slots + i] for i in order) for t in ts]

    return expected, _map_tuples(actual, permute)


LAYER_RELATION = re.compile(r"^\[([+-])\](.+?)\[(s{1,3}g|hist)\](?:\[(?:sat|unsat)\])?$")


def _nullary_layers_over_auxiliary(expected, actual, context):
    """The reference's history and subgoal relations over nullary predicates: omitted from its catalog by default,
    and with nullary relations kept, tuples without an argument for the auxiliary entity. The native encoder lists them
    (the history ones, in its default; the subgoal ones, with the relations of a config that keeps them) and each
    nullary literal of a layer or step is a tuple over the auxiliary entity, which the graph has exactly when there is
    one. The expectation lists the history relations that its catalog omits, from the config that keeps nullary
    relations, and ends each tuple of a nullary subgoal or history relation with the auxiliary entity."""
    if not over_auxiliary(context.config) or context.dag is not None:
        return expected, actual
    fixture = context.fixture
    kept = fixture["catalogs"][fixture["configs"][NULLARY_CONFIG]["catalog"]]
    catalog = list(expected.catalog)
    listed = {name for name, _ in catalog}
    catalog += [(name, 0) for name, arity, source in zip(kept["names"], kept["logical_arities"], kept["sources"])
                if arity == 0 and source == "history" and name not in listed]
    arity = dict(catalog)
    layered = {name for name in arity if arity[name] == 0 and LAYER_RELATION.match(name)}
    tuples = {name: ([t + (AUXILIARY,) for t in ts] if name in layered else ts) for name, ts in expected.tuples.items()}
    entities = list(expected.entities)
    if any(tuples.get(name) for name in layered) and ("auxiliary", AUXILIARY) not in entities:
        last = max((i for i, (role, _) in enumerate(entities) if role == "object"), default=-1)
        entities.insert(last + 1, ("auxiliary", AUXILIARY))
    catalog.sort(key=lambda c: c[0].encode())
    return replace(expected, catalog=catalog, entities=entities, tuples=tuples), actual


def goal_literals(context):
    if context.goals is not None:
        return set(map(parse_literal, context.goals))
    return {(x.positive, x.predicate.name, tuple(context.task.objects[o.index] for o in x.objects))
            for x in context.task.encoding_goal}


def literal_text(positive, predicate, arguments):
    return f"[{'+' if positive else '-'}]({' '.join((predicate, *arguments))})"


def _level_suffix(level):
    return "g" if level == 0 else "s" * level + "g"


def _repeated_literal(expected, actual, context):
    """A literal that is in two levels (the goal and a layer, or two layers) is in mifrost's graph at the highest of
    them, once for each level it is listed in (as one subgoal entity with equal tuples, and in the satisfaction
    relations of that level) and nowhere else. The native encoder has it once at every listing level that its masks
    can hold, and in the goal when listed there. The expectation distributes the entity and tuples to those levels
    only when the reference's equal tuple count matches the number of listing levels."""
    if not context.layers:
        return expected, actual
    levels = {}
    for literal in goal_literals(context):
        levels.setdefault(literal, set()).add(0)
    for level, layer in enumerate(context.layers, 1):
        for text in layer:
            levels.setdefault(parse_literal(text), set()).add(level)
    entities = list(expected.entities)
    tuples = {name: list(ts) for name, ts in expected.tuples.items()}
    for (positive, predicate, arguments), listed in sorted(levels.items(), key=repr):
        if len(listed) < 2:
            continue
        sign = "+" if positive else "-"
        text = literal_text(positive, predicate, arguments)
        high = f"[{_level_suffix(max(listed))}]"
        entity = f"{text}{high}"
        relation = f"[{sign}]{predicate}{high}"
        copies = [t for t in tuples.get(relation, []) if t[0] == entity]
        if ("subgoal", entity) not in entities or len(copies) != len(listed) or len(set(copies)) != 1:
            continue
        derived = {}
        for kind in ("[sat]", "[unsat]"):
            found = [t for t in tuples.get(relation + kind, []) if t[len(t) - len(arguments) :] == arguments]
            if found:
                if len(found) != len(listed) or len(set(found)) != 1:
                    break
                derived[kind] = found[0]
        else:
            tuples[relation] = [t for t in tuples[relation] if t[0] != entity]
            for kind, row in derived.items():
                tuples[relation + kind] = [t for t in tuples[relation + kind] if t != row]
            entities.remove(("subgoal", entity))
            dropped = {level for kind, level, literal in context.dropped
                       if kind == "subgoal" and parse_literal(literal) == (positive, predicate, arguments)}
            for level in sorted(listed - dropped):
                suffix = f"[{_level_suffix(level)}]"
                target = f"{text}{suffix}"
                target_relation = f"[{sign}]{predicate}{suffix}"
                moved = tuples.setdefault(target_relation, [])
                if (target,) + copies[0][1:] not in moved:
                    moved.append((target,) + copies[0][1:])
                for kind, row in derived.items():
                    if row not in tuples.setdefault(target_relation + kind, []):
                        tuples[target_relation + kind].append(row)
                role = "goal" if level == 0 else "subgoal"
                if (role, target) not in entities:
                    last = max(i for i, (r, _) in enumerate(entities) if r in ("object", "auxiliary", "goal", role))
                    entities.insert(last + 1, (role, target))
    return replace(expected, entities=entities, tuples={n: ts for n, ts in tuples.items() if ts}), actual


def _literal_without_slot(expected, actual, context):
    """mifrost takes a literal over a static predicate, or over an atom that no state can make true, in a subgoal layer
    or a history step. The native encoder takes masks over the atoms that have slots, so such a literal cannot be
    given to it (the harness leaves it out, and ``Context.dropped`` lists it). The expectation drops what mifrost made
    of exactly those literals: their entity, their tuples (the argument rows of their relations when the graph has no
    target entities) and their satisfaction tuples. A literal that is also a goal literal is in the goal of the native
    graph; ``repeated_literal`` places its reference copy in the goal and excludes its unrepresentable layer copies."""
    if not context.dropped or context.dag is not None:
        return expected, actual
    goal = goal_literals(context)
    graphs = {"expected": (list(expected.entities), {n: list(ts) for n, ts in expected.tuples.items()}),
              "actual": (list(actual.entities), {n: list(ts) for n, ts in actual.tuples.items()})}

    def drop(which, role, entity, relation):
        entities, tuples = graphs[which]
        if (role, entity) in entities:
            entities.remove((role, entity))
            tuples[relation] = [t for t in tuples.get(relation, []) if t[0] != entity]

    def drop_satisfaction(which, names, arguments):
        tuples = graphs[which][1]
        for name in names:
            tuples[name] = [t for t in tuples.get(name, []) if t[len(t) - len(arguments) :] != arguments]

    for kind, where, literal in context.dropped:
        positive, predicate, arguments = parse_literal(literal)
        sign = "+" if positive else "-"
        text = literal_text(positive, predicate, arguments)
        if kind == "subgoal":
            if (positive, predicate, arguments) in goal:
                continue  # repeated_literal keeps the goal copy and excludes the unrepresentable layer copies
            suffix = _level_suffix(where)
            relation = f"[{sign}]{predicate}[{suffix}]"
            satisfaction = [relation + "[sat]", relation + "[unsat]"]
            drop("expected", "subgoal", f"{text}[{suffix}]", relation)
            drop_satisfaction("expected", satisfaction, arguments)
        else:
            relation, entity = f"[{sign}]{predicate}[hist]", f"{where}:{text}"
            entities, tuples = graphs["expected"]
            if ("history_target", entity) in entities:
                entities.remove(("history_target", entity))
                tuples[relation] = [t for t in tuples.get(relation, []) if t[0] != entity]
            else:
                tuples[relation] = [t for t in tuples.get(relation, [])
                                    if not (t[0] == where and t[len(t) - len(arguments) :] == arguments)]
    (e_entities, e_tuples), (a_entities, a_tuples) = graphs["expected"], graphs["actual"]
    return (replace(expected, entities=e_entities, tuples={n: ts for n, ts in e_tuples.items() if ts}),
            replace(actual, entities=a_entities, tuples={n: ts for n, ts in a_tuples.items() if ts}))


def recorded_successor(context):
    """The node of the native transition graph that is the successor a record lists: ``1 + `` the index of the row of
    the state's expansion that has the record's action and leads to the record's successor state."""
    from mymyr import rl

    task = context.task
    state_path, successor_path = context.dag["nodes"]
    action = context.dag["edges"][0][2]
    rows = np.ascontiguousarray(task.encode([replay(task, state_path), replay(task, successor_path)]))
    expansion = rl.expand(rl.as_table(task), rows[:1])
    succ = np.asarray(expansion.succ)
    row = next(r for r in range(int(expansion.offsets[1]))
               if str(expansion.action(r)) == action and (succ[r] == rows[1]).all())
    return row + 1


def _unrecorded_successors(expected, actual, context):
    """A record of a transition config lists one successor of its state; the native graph holds all of them. The native
    graph is cut down to the state's relations and the recorded successor, the other successors' state entities and
    the tuples over them removed, the recorded one renumbered ``target:1``, and the auxiliary entity dropped when no
    tuple has it any more."""
    if not is_transition(context.config):
        return expected, actual
    keep = f"target:{recorded_successor(context)}"
    others = {name for role, name in actual.entities if role == "state" and name != keep}
    entities = [("state", "target:1") if e == ("state", keep) else e
                for e in actual.entities if not (e[0] == "state" and e[1] in others)]
    tuples = {}
    for name, ts in actual.tuples.items():
        kept = [tuple("target:1" if x == keep else x for x in t) for t in ts if not others & set(t)]
        if kept:
            tuples[name] = kept
    if not any(AUXILIARY in t for ts in tuples.values() for t in ts):
        entities = [e for e in entities if e[0] != "auxiliary"]
    return expected, replace(actual, entities=entities, tuples=tuples)


REGISTRY = (
    Difference(
        "unrecorded_successors",
        "A transition record of the reference lists one successor of its state; the native transition encoders encode "
        "all of them in the graph of the state.",
        "the state entities of the successors the record does not list and the tuples over them, and the auxiliary "
        "entity when no tuple is left over it, in the graph of a transition config",
        (("f01", "C8A"), ("f01", "C8AN"), ("f01", "C8B"), ("f01", "C8BN"), ("f02", "C8A"), ("f03", "C8B"),
         ("f05", "C8B"), ("f08", "C8A"), ("f08", "C8AN"), ("f08", "C8B"), ("f08", "C8BN"), ("f09", "C8A"),
         ("f10", "C8AN"), ("f10", "C8B"), ("f10", "C8BN"), ("f11", "C8A"), ("f12", "C8A"), ("f13", "C8AN")),
        _unrecorded_successors,
    ),
    Difference(
        "horizon_nullary_over_auxiliary",
        "mifrost's horizon graphs have no tuples in the relations of nullary predicates (its catalog lists them); the "
        "native encoder fills them as it does for a state graph: the atom of a node is the tuple of its state entity and "
        "the graph's auxiliary entity, which the graph has exactly when such a tuple exists.",
        "the relations of arity 0 of a horizon graph and its auxiliary entity",
        (("f10", "C7AN"), ("f01", "C7A")),
        _nullary_horizon,
    ),
    Difference(
        "root_state_entity",
        "mifrost has an entity `_root_state_` for the root that a horizon's root policy excludes, which none of its "
        "tuples mentions; the native graph has no entity for it, and the root's relations are the plain ones.",
        "the entity `_root_state_` of a horizon graph, only when no tuple of the reference mentions it",
        (("f01", "C7B"),),
        _root_state_entity,
    ),
    Difference(
        "repeated_literal_horizon",
        "With repeated_literals='every' (the default), a literal that several goal levels list is encoded once "
        "at each listing level; "
        "mifrost's horizon graph has it once for each listing level, all at the highest. Under 'highest', "
        "placement agrees but native cells keep one tuple instead of duplicate tuples.",
        "the tuples of the highest level of a literal that is listed in several levels, in every relation of that "
        "level, when there are as many equal tuples as levels",
        (("f01", "C7T"), ("f01", "C7S")),
        _repeated_literal_horizon,
    ),
    Difference(
        "delta_static_relations",
        "A delta horizon compares a node with its root and static atoms never change: the native catalog has no delta "
        "relation for a static predicate. mifrost lists them (`[+]p`, `[-]p`, `[sat+]`, `[sat-]`), and they are empty.",
        "the catalog entries of static predicates that the native catalog lacks, only in the delta mode and only when "
        "the reference has no tuple in them",
        (("f08", "C7B"), ("f09", "C7C")),
        _delta_static_relations,
    ),
    Difference(
        "never_filled_horizon_relations",
        "mifrost's horizon catalogs list relations that no graph has a tuple in: the delta mode's relations of the "
        "literals themselves (`[+]p[sat]`, `[-]p[unsat]`, `[+]p[sat+]`, `[-]p[sat-]`, which have no goal level) and, in the "
        "split catalog, the relation of the nodes of a static predicate (`fixed[state]`). The native catalog lists a "
        "relation only if a graph can have a tuple in it: it has no support literals, and static atoms are the root's.",
        "the catalog entries of those relations, only when the reference has no tuple in them and the native catalog "
        "lacks them",
        (("f01", "C7I"), ("f08", "C7G")),
        _never_filled_relations,
    ),
    Difference(
        "static_literal_at_nodes",
        "mifrost evaluates the goal at the nodes below the root without the instance's static atoms (a static atom does "
        "not hold there: a positive static literal that holds is unsatisfied, a negative one satisfied); the native "
        "encoder evaluates every node with the static atoms.",
        "the satisfaction tuples of the literals over static predicates whose atom holds, at the nodes below the root "
        "of a horizon graph",
        (("f08", "C7A"), ("f08", "C7G")),
        _static_literal_at_nodes,
    ),
    Difference(
        "action_subset",
        "A record's reference encoding was given the actions listed in its inputs, which can be fewer than the "
        "applicable ones; the native encoder is given every applicable action.",
        "the action entities that the record does not list, and the tuples over them, only when the record lists "
        "fewer actions than the state has and the task says the entity's action is applicable in the state",
        (("f03", "C3"), ("f04", "C3"), ("f06", "C3"), ("f07", "C3")),
        _action_subset,
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
        "nullary_layers_over_auxiliary",
        "Nullary literals of subgoal layers and history steps are tuples over the auxiliary entity, which the graph "
        "has exactly when there is one; mifrost's tuples have no such argument, and by default its catalog omits "
        "the history relations of nullary predicates.",
        "the history and subgoal relations of arity 0 and the entity of role auxiliary",
        (("f01", "C1"), ("f08", "C4_L1N"), ("f10", "C5BN")),
        _nullary_layers_over_auxiliary,
    ),
    Difference(
        "literal_without_slot",
        "A literal over a static predicate, or over an atom that no state can make true, has no slot, so the "
        "native subgoal and history inputs cannot hold it; mifrost encodes it.",
        "the entities and tuples that mifrost made of the literals the harness left out of the native inputs, except "
        "those of a literal that is also a goal literal",
        (("f08", "C3S"), ("f08", "C5A"), ("f08", "C5BN")),
        _literal_without_slot,
    ),
    Difference(
        "repeated_literal",
        "With repeated_literals='every' (the default), a literal that several goal levels list is encoded once "
        "at each representable listing level; "
        "mifrost encodes it once for each level it is listed in, all at the highest of them, and leaves it out of "
        "the other levels. Under repeated_literals='highest', placement agrees but native cells keep one tuple "
        "instead of duplicate tuples.",
        "the subgoal entity and the equal tuples of the highest level of a literal that is listed in several levels, "
        "and its satisfaction tuples",
        (("f01", "C4_L1"), ("f01", "C4_L2"), ("f08", "C4_L1"), ("f08", "C4_L1N"), ("f08", "C4_L2"), ("f08", "C4_L2N"),
         ("f09", "C4_L2"), ("f13", "C4_L1N")),
        _repeated_literal,
    ),
    Difference(
        "goal_entity_order",
        "Goal entities are numbered in the canonical order of the goal tuples (relation, then tuple); mifrost numbers "
        "them in the order of the goal literals.",
        "the entities of role goal: compared in name order",
        (("f03", "C3"),),
        _sorted_role("goal"),
    ),
    Difference(
        "subgoal_entity_order",
        "Subgoal entities are numbered in the canonical order of the subgoal tuples (relation, then tuple); mifrost "
        "numbers them in the order of the layers' literals.",
        "the entities of role subgoal: compared in name order",
        (("f01", "C4_L2"), ("f09", "C4_L2")),
        _sorted_role("subgoal"),
    ),
    Difference(
        "history_target_entity_order",
        "History targets are numbered in the order of the history tuples (relation, then step, then slot); mifrost "
        "numbers them in the order of the steps and their literals.",
        "the entities of role history_target: compared in name order",
        (("f01", "C5A"), ("f08", "C3H")),
        _sorted_role("history_target"),
    ),
    Difference(
        "action_entity_order",
        "Action entities are numbered in the canonical order of the expansion rows (action schema, then binding); "
        "mifrost numbers them in the order of the actions it was given.",
        "the entities of role action: compared in name order",
        (("f02", "C3"), ("f03", "C3")),
        _sorted_role("action"),
    ),
    Difference(
        "predicate_virtual_order",
        "Predicate virtual nodes are numbered by the name of the predicate among the used predicates; "
        "mifrost numbers them in the order of their first use.",
        "the entities of role predicate_virtual: compared in name order",
        (("f01", "C3"),),
        _sorted_role("predicate_virtual"),
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
# The native options of each compared config. A config without actions in its records has no action lane. The configs
# with ``ignore_zero_arity_relations`` off in the reference keep the nullary relations; the others compare against the
# catalog without them.
LAYERS = dict(goal_derivations=("plain", "satisfied", "unsatisfied"), target_sources=("goal", "subgoal"),
              include_actions=False)
SUPPORTED = {
    "C1": dict(include_actions=False),
    "C2": dict(goal_derivations=("plain", "satisfied", "unsatisfied"), nullary="auxiliary", layout="relation",
               include_actions=False),
    "C3": dict(target_sources=("action", "goal"), predicate_virtual_nodes=True),
    "C3N": dict(target_sources=("action", "goal"), predicate_virtual_nodes=True, nullary="auxiliary"),
    "C3D": dict(target_sources=("action", "goal"), predicate_virtual_nodes=True, nullary="auxiliary",
                goal_derivations=("plain", "satisfied", "unsatisfied")),
    "C3S": dict(target_sources=("action", "goal", "subgoal"), max_goal_level=2, predicate_virtual_nodes=True,
                nullary="auxiliary"),
    "C3H": dict(target_sources=("action", "goal", "history"), history=True, predicate_virtual_nodes=True,
                nullary="auxiliary"),
    "C4_L1": dict(LAYERS, max_goal_level=1, nullary="omit"),
    "C4_L1N": dict(LAYERS, max_goal_level=1, nullary="auxiliary"),
    "C4_L2": dict(LAYERS, max_goal_level=2, nullary="omit"),
    "C4_L2N": dict(LAYERS, max_goal_level=2, nullary="auxiliary"),
    "C5A": dict(target_sources=("history",), history=True, include_actions=False, nullary="omit"),
    "C5AN": dict(target_sources=("history",), history=True, include_actions=False, nullary="auxiliary"),
    "C5B": dict(history=True, include_actions=False, nullary="omit"),
    "C5BN": dict(history=True, include_actions=False, nullary="auxiliary"),
}
# Native configs compared against the records of another config: every config that keeps the nullary relations, with
# the nullary atoms as tuples of their leading slots alone, as the reference encodes them with
# ``ignore_zero_arity_relations`` off (no auxiliary entity).
VARIANTS = {f"{config}Z": config for config in ("C2", "C3N", "C3D", "C3S", "C3H", "C4_L1N", "C4_L2N", "C5AN", "C5BN")}
for _variant, _config in VARIANTS.items():
    SUPPORTED[_variant] = dict(SUPPORTED[_config], nullary="arity0")
# The horizon configs (mifrost's FlatHorizonEncoder). Every one is compared with the nullary relations kept: mifrost's
# catalog lists them whatever its ignore_zero_arity_relations says, and leaves them empty.
ADDED = ("added_satisfied", "added_unsatisfied")
for _config, _options in {
    "C7A": dict(horizon_mode="full"),
    "C7AN": dict(horizon_mode="full"),
    "C7B": dict(horizon_mode="delta", root_policy="exclude", goal_derivations=ADDED, include_actions=False),
    "C7C": dict(horizon_mode="delta", root_policy="encode_only", goal_derivations=ADDED, parent_relation=True),
    "C7D": dict(horizon_mode="action", root_policy="exclude"),
    "C7DN": dict(horizon_mode="action", root_policy="exclude"),
    "C7G": dict(horizon_mode="full", root_policy="exclude"),
    "C7H": dict(horizon_mode="full", root_policy="encode_only"),
    "C7I": dict(horizon_mode="delta", goal_derivations=ADDED, include_actions=False),
    "C7J": dict(horizon_mode="action"),
    "C7K": dict(horizon_mode="action", root_policy="encode_only"),
    "C7S": dict(horizon_mode="delta", root_policy="exclude", goal_derivations=ADDED, include_actions=False,
                max_goal_level=2),
    "C7T": dict(horizon_mode="full", max_goal_level=1),
    "C7E": dict(horizon_mode="full", root_policy="encode_only", parent_relation=True),
    "C7EN": dict(horizon_mode="full", root_policy="encode_only", parent_relation=True),
    "C7F": dict(horizon_mode="full", root_policy="encode_only", parent_relation=True, sibling_relation=True,
                cousin_relation=True),
    "C7FN": dict(horizon_mode="full", root_policy="encode_only", parent_relation=True, sibling_relation=True,
                 cousin_relation=True),
    # the transition configs: the horizon of depth one of a state (their encoders fix the mode)
    "C8A": dict(horizon_mode="full", root_policy="exclude", include_actions=False),
    "C8AN": dict(horizon_mode="full", root_policy="exclude", include_actions=False),
    "C8B": dict(horizon_mode="delta", root_policy="exclude", include_actions=False),
    "C8BN": dict(horizon_mode="delta", root_policy="exclude", include_actions=False),
}.items():
    SUPPORTED[_config] = _options
# The encoder class of the configs that FlatHorizonEncoder does not encode.
ENCODERS = {
    "C7E": "FlatRootedHorizonEncoder", "C7EN": "FlatRootedHorizonEncoder", "C7F": "FlatRootedHorizonEncoder",
    "C7FN": "FlatRootedHorizonEncoder", "C8A": "FlatTransitionEncoder", "C8AN": "FlatTransitionEncoder",
    "C8B": "FlatTransitionEffectsEncoder", "C8BN": "FlatTransitionEffectsEncoder",
}
TRANSITION_CONFIGS = ("C8A", "C8AN", "C8B", "C8BN")
# Relation sources of the fixtures' catalogs (inside a compared config) that the comparison leaves out, with the lane
# that turns them on.
PENDING_SOURCES = {
    "action": "actions (of the configs whose native encoder has no action relations)",
}
COMPARED_SOURCES = ("state", "goal", "goal_derivation", "history", "goal_satisfaction", "parent", "sibling", "cousin")
COMPARED_LANES = ("fluent", "static", "derived", "goal", "goal_derivation", "history", "goal_satisfaction", "parent",
                  "sibling", "cousin")


def sources_of(config):
    """The relation sources compared for ``config``: the action relations too while its native encoder has them."""
    return COMPARED_SOURCES + (("action",) if SUPPORTED[config].get("include_actions", True) else ())


def lanes_of(config):
    """The lanes compared for ``config``: the action lane too while its native encoder has it."""
    return COMPARED_LANES + (("action",) if SUPPORTED[config].get("include_actions", True) else ())


def is_horizon(config):
    """Whether ``config`` is encoded as a horizon (by one of the horizon or transition encoders)."""
    return "horizon_mode" in SUPPORTED[config]


def is_transition(config):
    """Whether ``config`` is encoded from the state alone, as its transitions."""
    return config in TRANSITION_CONFIGS


# Configs that are not compared yet, with the lanes they need. (The HGraph configs, C9A to C9C, are compared by
# encoding_hgraph_parity.)
PENDING_CONFIGS = {
    "C8S": "goals, actions, subgoals, transition",
}


def recorded_config(config):
    """The config of the fixture records that the native config ``config`` is compared against."""
    return VARIANTS.get(config, config)


def native_encoder(task, config, **options):
    from mymyr import encoding

    name = ENCODERS.get(config, "FlatHorizonEncoder" if is_horizon(config) else "FlatRelationEncoder")
    effective = {**SUPPORTED[config], **options}
    if is_transition(config):  # the class fixes the mode
        del effective["horizon_mode"]
    return getattr(encoding, name)(task, **effective)


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


def split_by_slot(encoder, literals):
    """``(kept, dropped)``: the literals that the encoder's masks can hold, and those the task gives no slot."""
    from mymyr import encoding

    kept, dropped = [], []
    for literal in literals:
        try:
            encoding.subgoal_masks(encoder, [[[literal]]])
        except ValueError as error:
            if "slot" not in str(error):
                raise
            dropped.append(literal)
        else:
            kept.append(literal)
    return kept, dropped


def layer_inputs(encoder, inputs):
    """The ``subgoals`` and ``history`` arguments of the native encoder for the layers and steps of a record's inputs
    (literals without a slot left out), and the literals left out."""
    from mymyr import encoding

    arguments, dropped = {}, []
    if "subgoal_layers" in inputs:
        layers = []
        for level, layer in enumerate(inputs["subgoal_layers"], 1):
            kept, lost = split_by_slot(encoder, layer)
            layers.append(kept)
            dropped += [("subgoal", level, literal) for literal in lost]
        arguments["subgoals"] = encoding.subgoal_masks(encoder, [layers])
    if "history_subgoals" in inputs:
        steps = []
        given = inputs["history_subgoals"]
        first = max(len(given) - inputs["history_max_steps"], 0) if inputs.get("history_max_steps") else 0
        for position, (dt, literals) in enumerate(given):
            kept, lost = split_by_slot(encoder, literals)
            steps.append((dt, kept))
            if position >= first:
                dropped += [("history", f"history:{dt}#{position - first}", literal) for literal in lost]
        arguments["history"] = encoding.history_masks(encoder, [steps])
    return arguments, tuple(dropped)


@dataclass
class Placed:
    """One record's native graph: the encoder, the batch and row it is in, and the literals left out of its inputs."""

    encoder: object
    batch: object
    row: int
    dropped: tuple = ()


def record_context(fixture, task, state, config, record, dropped=()):
    """The ``Context`` of one fixture record."""
    inputs = record["inputs"]

    def given(key):
        return tuple(inputs[key]) if key in inputs else None

    steps = inputs.get("history_subgoals")
    history = None if steps is None else tuple((dt, tuple(literals)) for dt, literals in steps)
    layers = inputs.get("subgoal_layers")
    layers = None if layers is None else tuple(tuple(layer) for layer in layers)
    dag = inputs.get("dag")
    if "successor" in inputs:
        successor = inputs["successor"]
        dag = {"nodes": [state["path"], successor["path"]], "edges": [[0, 1, successor["action"]]]}
    return Context(fixture, task, state, config, given("goals"), given("actions"), layers, history, dropped, dag)


def horizon_dag(task, table, dag):
    """The node rows and the edges (source, target, schema, binding) of a recorded dag: the nodes are the states of its
    action paths, and each edge is the row of the parent's expansion that has the edge's action and leads to the child."""
    from mymyr import rl

    rows = np.ascontiguousarray(task.encode([replay(task, path) for path in dag["nodes"]]))
    expansions = {}
    edges = []
    for parent, child, name in dag["edges"]:
        if parent not in expansions:
            expansions[parent] = rl.expand(table, rows[parent : parent + 1])
        expansion = expansions[parent]
        succ = np.asarray(expansion.succ)
        row = next(r for r in range(int(expansion.offsets[1]))
                   if str(expansion.action(r)) == name and (succ[r] == rows[child]).all())
        edges.append((parent, child, int(expansion.schema[row]), np.asarray(expansion.binding[row])))
    return rows, edges


def horizon_view(table, dags):
    """A HorizonView of the dags (``(rows, edges)`` of ``horizon_dag``), one graph each."""
    from mymyr import encoding

    flat = [edge for _, edges in dags for edge in edges]
    return encoding.HorizonView(
        nodes=np.concatenate([rows for rows, _ in dags]),
        node_offsets=np.cumsum([0] + [len(rows) for rows, _ in dags]).astype(np.int32),
        edge_offsets=np.cumsum([0] + [len(edges) for _, edges in dags]).astype(np.int32),
        edge_src=np.array([e[0] for e in flat], np.int32), edge_dst=np.array([e[1] for e in flat], np.int32),
        edge_schema=np.array([e[2] for e in flat], np.int32),
        edge_binding=np.array([e[3] for e in flat], np.int32).reshape(len(flat), table.label_width),
        task_ids=np.zeros(len(dags), np.int32))


def encode_horizon_records(task, config, records):
    """The ``Placed`` graph of each horizon record: the records without subgoal layers in one batch, each other record
    alone with its layers as the subgoals of its graph."""
    from mymyr import rl

    table = rl.as_table(task)
    dags = [horizon_dag(task, table, r["inputs"]["dag"]) for r in records]
    placed = [None] * len(records)
    plain = [i for i, r in enumerate(records) if "subgoal_layers" not in r["inputs"]]
    if plain:
        encoder = native_encoder(task, config)
        batch = encoder.encode(horizon_view(table, [dags[i] for i in plain]))
        for k, i in enumerate(plain):
            placed[i] = Placed(encoder, batch, k)
    for i, record in enumerate(records):
        if placed[i] is None:
            encoder = native_encoder(task, config)
            arguments, dropped = layer_inputs(encoder, record["inputs"])
            placed[i] = Placed(encoder, encoder.encode(horizon_view(table, [dags[i]]), **arguments), 0, dropped)
    return placed


def encode_transition_records(task, config, rows, records):
    """The ``Placed`` graph of each transition record: the graph of its state, which holds all of the state's
    successors (the records of a config are encoded in one batch)."""
    encoder = native_encoder(task, config)
    batch = encoder.encode(list(rows))
    return [Placed(encoder, batch, i) for i in range(len(records))]


def encode_records(task, config, rows, records):
    """The ``Placed`` graph of each record: the records without goal literals, subgoal layers or history in one batch,
    each other record alone, with its literals as the goal, its layers and its history (and, for a record with a
    ``history_max_steps``, an encoder that keeps that many steps). The encoder is given the expansion of its states for
    the action lane."""
    from mymyr import rl

    if is_transition(config):
        return encode_transition_records(task, config, rows, records)
    if is_horizon(config):
        return encode_horizon_records(task, config, records)
    table = rl.as_table(task)
    actions = SUPPORTED[config].get("include_actions", True)
    alone = ("goals", "subgoal_layers", "history_subgoals")

    def encode(encoder, states, **options):
        return encoder.encode(states, actions=rl.expand(table, states) if actions else None, **options)

    placed = [None] * len(records)
    plain = [i for i, r in enumerate(records) if not any(key in r["inputs"] for key in alone)]
    if plain:
        encoder = native_encoder(task, config)
        batch = encode(encoder, [rows[i] for i in plain])
        for k, i in enumerate(plain):
            placed[i] = Placed(encoder, batch, k)
    for i, record in enumerate(records):
        inputs = record["inputs"]
        if placed[i] is None:
            extra = {"history_max_steps": inputs["history_max_steps"]} if "history_max_steps" in inputs else {}
            encoder = native_encoder(task, config, **extra)
            arguments, dropped = layer_inputs(encoder, inputs)
            if "goals" in inputs:
                arguments["goals"] = goal_masks(task, inputs["goals"])
            placed[i] = Placed(encoder, encode(encoder, [rows[i]], **arguments), 0, dropped)
    return placed


@functools.cache
def run_case(path, config):
    """Encode the recorded states of ``config`` natively and compare each graph against mifrost's record."""
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == recorded_config(config)]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    result = CaseResult({}, set())
    for record, placed in zip(records, encode_records(task, config, rows, records)):
        context = record_context(fixture, task, states[record["state"]], config, record, placed.dropped)
        expected = fixture_graph(fixture, record, sources_of(config))
        actual = native_graph(placed.encoder, placed.batch, placed.row, [task], lanes_of(config))
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
