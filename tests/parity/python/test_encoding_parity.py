"""Parity of the native relational encoder with mifrost's recorded encodings (tests/parity/data/encoding/mifrost).

Every fixture x config is compared by names (``encoding_parity``): the configs the encoder supports must have no
mismatch beyond the registered differences, the others are skipped with the lane that turns them on. The harness is
tested on its own (a fixture against itself, mutated copies, the narrowness of the registered differences), and the
native encoder is tested for what parity cannot show: a graph does not depend on the rows around it."""

import itertools
import re

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from encoding_parity import (
    COMPARED_LANES, COMPARED_SOURCES, encode_records, FIXTURES, NAMES, PENDING_CONFIGS, PENDING_SOURCES, REGISTRY, SUPPORTED, AUXILIARY,
    HELPER, Context, horizon_dag, horizon_view, native_graph, base_predicate, Graph, Mismatch, compare, compare_exactly, fixture_cases, fixture_graph, load, native_encoder,
    goal_literals, literal_text, needed_differences,
    lanes_of, native_graph, parse_literal, random_walk_batch, random_walk_rows, read, record_context, replay, run_case,
    sources_of,
)

IDS = [p.stem[:3] for p in FIXTURES]
CONFIGS = sorted({config for _, config in fixture_cases()})


def case_params():
    """pytest parameters of the (fixture, config) cases of the compared configs."""
    return [pytest.param(path, config, id=f"{path.stem[:3]}-{config}")
            for path, config in fixture_cases() if config in SUPPORTED]


def prefix_path(prefix):
    return next(p for p in FIXTURES if p.stem.startswith(prefix))


# ------------------------------------------------------------------------------------------------ the tables
def test_every_config_is_compared_or_pending_and_the_tables_name_existing_configs():
    configs = set(CONFIGS)
    assert set(SUPPORTED) <= configs and set(PENDING_CONFIGS) <= configs
    assert not set(SUPPORTED) & set(PENDING_CONFIGS)
    assert set(SUPPORTED) | set(PENDING_CONFIGS) == configs
    assert all(PENDING_CONFIGS.values())


def test_pending_sources_are_the_other_sources_of_the_compared_configs():
    found = set()
    for path in FIXTURES:
        fixture = read(path)
        for config in SUPPORTED:
            if config not in fixture["configs"]:
                continue
            catalog = fixture["catalogs"][fixture["configs"][config]["catalog"]]
            found |= set(catalog["sources"]) - set(sources_of(config))
    assert found == set(PENDING_SOURCES)
    assert set(COMPARED_SOURCES) & set(PENDING_SOURCES) == set()
    assert COMPARED_LANES == ("fluent", "static", "derived", "goal", "goal_derivation", "history", "goal_satisfaction", "parent")
    assert [c for c in SUPPORTED if "action" in lanes_of(c)] == [c for c in SUPPORTED if "action" in sources_of(c)]
    assert all(("action" in lanes_of(c)) == SUPPORTED[c].get("include_actions", True) for c in SUPPORTED)


# ------------------------------------------------------------------------------------------------ parity
@pytest.mark.parametrize("path, config", case_params())
def test_native_graphs_equal_mifrost_up_to_the_registered_differences(path, config):
    fixture = read(path)
    expected_states = [r["state"] for r in fixture["encodings"] if r["config"] == config]
    result = run_case(path, config)
    assert list(result.mismatches) == expected_states
    report = [f"state {state}: {m}" for state, found in result.mismatches.items() for m in found]
    assert not report, "\n" + "\n".join(report)


# ------------------------------------------------------------------------------------------------ the registry
def test_the_registry_is_documented_and_unique():
    assert len(set(NAMES)) == len(NAMES)
    for difference in REGISTRY:
        assert difference.reason.strip() and difference.scope.strip() and difference.cases, difference.name
        assert all(isinstance(config, str) and config in SUPPORTED for _, config in difference.cases)


@pytest.mark.parametrize("difference", REGISTRY, ids=NAMES)
def test_the_cases_of_a_registered_difference_need_it(difference):
    for prefix, config in difference.cases:
        assert difference.name in run_case(prefix_path(prefix), config).needed, (prefix, config)


def test_no_registered_difference_is_stale():
    needed = set()
    for path, config in itertools.product(FIXTURES, sorted(SUPPORTED)):
        needed |= run_case(path, config).needed
    assert needed == set(NAMES)


def test_without_the_registry_the_exact_comparison_finds_the_differences():
    fixture, task = load(prefix_path("f05"))
    records = [r for r in fixture["encodings"] if r["config"] == "C1"]
    states = {s["id"]: s for s in fixture["states"]}
    encoder = native_encoder(task, "C1")
    batch = encoder.encode([replay(task, states[r["state"]]["path"]) for r in records])
    expected = fixture_graph(fixture, records[0], COMPARED_SOURCES)
    actual = native_graph(encoder, batch, 0, [task], COMPARED_LANES)
    context = Context(fixture, task, states[records[0]["state"]], "C1")
    assert compare(expected, actual, context) == []
    kinds = {m.kind for m in compare(expected, actual, context, allowed=())}
    assert kinds == {"catalog", "entity", "tuple"}
    for difference in REGISTRY:
        if any(config == "C1" for _, config in difference.cases):
            others = [n for n in NAMES if n != difference.name]
            assert compare(expected, actual, context, others), difference.name


# ------------------------------------------------------------------------------------------------ the harness itself
def record_of(fixture, config="C1", state="s2"):
    return next(r for r in fixture["encodings"] if r["config"] == config and r["state"] == state)


def tuples_of(graph):
    """(relation, index, tuple) of every tuple of a graph, relations in name order."""
    return [(name, i, t) for name, ts in sorted(graph.tuples.items()) for i, t in enumerate(ts)]


def altered(graph, tuple_):
    """Another tuple of the same arity: the reversed one, or for a symmetric one the first entity replaced."""
    if tuple_[::-1] != tuple_:
        return tuple_[::-1]
    other = next(name for role, name in graph.entities if role == "object" and name != tuple_[0])
    return (other,) + tuple_[1:]


def copy(graph):
    return Graph(list(graph.catalog), list(graph.entities), {n: list(ts) for n, ts in graph.tuples.items()})


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_a_fixture_compared_with_itself_has_no_mismatch(path):
    fixture = read(path)
    for record in fixture["encodings"]:
        graph = fixture_graph(fixture, record)
        assert graph.entities and graph.catalog
        assert compare_exactly(graph, copy(graph)) == []
        assert compare(graph, copy(graph), None, allowed=()) == []


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_a_mutated_copy_of_a_fixture_fails_with_the_right_mismatch(path):
    fixture = read(path)
    graph = fixture_graph(fixture, record_of(fixture))
    name, index, tuple_ = tuples_of(graph)[0]

    dropped = copy(graph)
    del dropped.tuples[name][index]
    assert compare_exactly(graph, dropped) == [Mismatch("tuple", name, f"missing {tuple_!r}")]
    assert compare_exactly(dropped, graph) == [Mismatch("tuple", name, f"unexpected {tuple_!r}")]

    other = altered(graph, tuple_)
    changed = copy(graph)
    changed.tuples[name][index] = other
    found = compare_exactly(graph, changed)
    assert {m.kind for m in found} == {"tuple"} and {m.where for m in found} == {name}
    assert Mismatch("tuple", name, f"missing {tuple_!r}") in found
    assert Mismatch("tuple", name, f"unexpected {other!r}") in found

    role, victim = graph.entities[0]
    renamed = copy(graph)
    renamed.entities[0] = (role, victim + "_renamed")
    found = compare_exactly(graph, renamed)
    assert found == [Mismatch("entity", None, f"missing {(role, victim)!r}"),
                     Mismatch("entity", None, f"unexpected {(role, victim + '_renamed')!r}")]

    swapped_entities = copy(graph)
    swapped_entities.entities[0], swapped_entities.entities[1] = graph.entities[1], graph.entities[0]
    found = compare_exactly(graph, swapped_entities)
    assert found and {m.kind for m in found} == {"entity"}

    shuffled = copy(graph)
    shuffled.catalog[0], shuffled.catalog[1] = graph.catalog[1], graph.catalog[0]
    assert {m.kind for m in compare_exactly(graph, shuffled)} == {"catalog"}

    for relation in sorted(graph.tuples)[:2]:
        emptied = copy(graph)
        del emptied.tuples[relation]
        found = compare_exactly(graph, emptied)
        assert len(found) == len(graph.tuples[relation]) and {m.where for m in found} == {relation}


def test_every_mismatch_of_a_graph_is_reported():
    fixture = read(prefix_path("f03"))
    graph = fixture_graph(fixture, record_of(fixture))
    broken = copy(graph)
    names = sorted(broken.tuples)[:3]
    for name in names:
        broken.tuples[name].pop()
    broken.entities.pop()
    found = compare_exactly(graph, broken)
    assert {m.where for m in found if m.kind == "tuple"} == set(names)
    assert sum(m.kind == "entity" for m in found) == 1


# ------------------------------------------------------------------------------------------------ narrowness
def native_case(path, state="s2"):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == "C1"]
    states = {s["id"]: s for s in fixture["states"]}
    encoder = native_encoder(task, "C1")
    batch = encoder.encode([replay(task, states[r["state"]]["path"]) for r in records])
    row = [r["state"] for r in records].index(state)
    context = Context(fixture, task, states[state], "C1")
    expected = fixture_graph(fixture, records[row], COMPARED_SOURCES)
    return expected, native_graph(encoder, batch, row, [task], COMPARED_LANES), context


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_registered_differences_do_not_hide_mutations_of_the_native_graph(path):
    expected, actual, context = native_case(path)
    assert compare(expected, actual, context) == []
    name, index, tuple_ = tuples_of(actual)[0]

    dropped = copy(actual)
    del dropped.tuples[name][index]
    found = compare(expected, dropped, context)
    assert found and {m.kind for m in found} == {"tuple"} and {m.where for m in found} == {name}

    changed = copy(actual)
    changed.tuples[name][index] = altered(actual, tuple_)
    found = compare(expected, changed, context)
    assert found and {m.kind for m in found} == {"tuple"} and {m.where for m in found} == {name}

    extra = copy(actual)
    extra.tuples[name].append(tuple_)
    assert [m.detail for m in compare(expected, extra, context)] == [f"unexpected {tuple_!r}"]

    role, victim = actual.entities[0]
    renamed = copy(actual)
    renamed.entities[0] = (role, victim + "_renamed")
    assert {m.kind for m in compare(expected, renamed, context)} == {"entity"}

    moved = copy(actual)
    moved.catalog[0], moved.catalog[1] = actual.catalog[1], actual.catalog[0]
    assert {m.kind for m in compare(expected, moved, context)} == {"catalog"}

    lost = copy(actual)
    victim_relation = next(n for n, a in lost.catalog if a > 0 and n in lost.tuples and n != "=")
    lost.catalog = [c for c in lost.catalog if c[0] != victim_relation]
    del lost.tuples[victim_relation]
    assert {m.kind for m in compare(expected, lost, context)} == {"catalog", "tuple"}


def goal_case(config, state="s1", prefix="f03"):
    fixture, task = load(prefix_path(prefix))
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    index = [r["state"] for r in records].index(state)
    placed = encode_records(task, config, rows, records)[index]
    context = record_context(fixture, task, states[state], config, records[index], placed.dropped)
    expected = fixture_graph(fixture, records[index], sources_of(config))
    return expected, native_graph(placed.encoder, placed.batch, placed.row, [task], lanes_of(config)), context


@pytest.mark.parametrize("config", ["C1", "C2", "C3"])
def test_goal_tuples_and_goal_entities_are_compared(config):
    expected, actual, context = goal_case(config)
    assert compare(expected, actual, context) == []
    goal_relations = [n for n in actual.tuples if n.endswith(("[g]", "[sat]", "[unsat]"))]
    assert any(n.endswith("[g]") for n in goal_relations) and any(n.endswith("[g][sat]") for n in goal_relations)
    assert (config == "C2") == any(n.endswith("[unsat]") for n in goal_relations)
    for name in goal_relations:
        dropped = copy(actual)
        dropped.tuples[name] = dropped.tuples[name][1:]
        assert {(m.kind, m.where) for m in compare(expected, dropped, context)} == {("tuple", name)}
    goals = [e for e in actual.entities if e[0] == "goal"]
    assert bool(goals) == (config == "C3")
    for i, (role, name) in enumerate(actual.entities):
        if role == "goal":
            renamed = copy(actual)
            renamed.entities[i] = (role, name + "_renamed")
            assert {m.kind for m in compare(expected, renamed, context)} >= {"entity"}
            break

def test_action_tuples_action_entities_and_virtual_nodes_are_compared():
    expected, actual, context = goal_case("C3")
    assert compare(expected, actual, context) == []
    roles = {role for role, _ in actual.entities}
    assert {"action", "predicate_virtual"} <= roles
    listed = {name for role, name in expected.entities if role == "action"}
    action_relations = [n for n, ts in actual.tuples.items() if any(t[0] in listed for t in ts)]
    assert action_relations
    for name in action_relations:
        victim = next(t for t in actual.tuples[name] if t[0] in listed)
        dropped = copy(actual)
        dropped.tuples[name] = [t for t in dropped.tuples[name] if t != victim]
        assert ("tuple", name) in {(m.kind, m.where) for m in compare(expected, dropped, context)}
    for role in ("action", "predicate_virtual"):
        i = next(i for i, (r, n) in enumerate(actual.entities) if r == role and (role != "action" or n in listed))
        renamed = copy(actual)
        renamed.entities[i] = (role, actual.entities[i][1] + "_renamed")
        assert {m.kind for m in compare(expected, renamed, context)} >= {"entity"}
        lost = copy(actual)
        del lost.entities[i]
        assert {m.kind for m in compare(expected, lost, context)} >= {"entity"}
    extra = copy(actual)
    extra.entities.append(("predicate_virtual", "predicate:unused"))
    assert {m.kind for m in compare(expected, extra, context)} == {"entity"}


def levelled(graph):
    """The relations of a graph that hold subgoal tuples (``[sg]``, ``[ssg]``, ``[sssg]``, with their satisfaction)."""
    return [n for n in graph.tuples if re.search(r"\[s{1,3}g\]", n)]


def test_subgoal_tuples_and_subgoal_entities_are_compared():
    expected, actual, context = goal_case("C4_L2", "s0", "f01")
    assert compare(expected, actual, context) == []
    relations = levelled(actual)
    assert any(n.endswith("[sg]") for n in relations) and any(n.endswith("[ssg]") for n in relations)
    assert any(n.endswith("[sat]") or n.endswith("[unsat]") for n in relations)
    for name in relations:
        dropped = copy(actual)
        dropped.tuples[name] = dropped.tuples[name][1:]
        assert {(m.kind, m.where) for m in compare(expected, dropped, context)} == {("tuple", name)}
        extra = copy(actual)
        extra.tuples[name] = extra.tuples[name] + extra.tuples[name][:1]
        assert [(m.kind, m.where) for m in compare(expected, extra, context)] == [("tuple", name)]
    subgoals = [i for i, (role, _) in enumerate(actual.entities) if role == "subgoal"]
    assert subgoals
    for i in subgoals:
        role, name = actual.entities[i]
        moved = copy(actual)
        moved.entities[i] = (role, name.replace("[sg]", "[ssg]") if name.endswith("[sg]") else name.replace("[ssg]", "[sg]"))
        assert {m.kind for m in compare(expected, moved, context)} == {"entity"}
        lost = copy(actual)
        del lost.entities[i]
        assert {m.kind for m in compare(expected, lost, context)} == {"entity"}


def test_the_numbering_of_subgoal_entities_is_excused_but_their_names_are_not():
    expected, actual, context = goal_case("C4_L2", "s0", "f01")
    first, second = [i for i, (role, _) in enumerate(actual.entities) if role == "subgoal"][:2]
    swapped = copy(actual)
    swapped.entities[first], swapped.entities[second] = actual.entities[second], actual.entities[first]
    assert compare(expected, swapped, context) == []
    assert {m.kind for m in compare(expected, swapped, context, [n for n in NAMES if n != "subgoal_entity_order"])} == {"entity"}


def test_history_tuples_history_steps_and_history_targets_are_compared():
    expected, actual, context = goal_case("C5A", "s2", "f01")
    assert compare(expected, actual, context) == []
    relations = [n for n in actual.tuples if n.endswith("[hist]")]
    assert relations and {role for role, _ in actual.entities} >= {"history", "history_target"}
    for name in relations:
        dropped = copy(actual)
        dropped.tuples[name] = dropped.tuples[name][1:]
        assert {(m.kind, m.where) for m in compare(expected, dropped, context)} == {("tuple", name)}
    for role in ("history", "history_target"):
        i = next(i for i, (r, _) in enumerate(actual.entities) if r == role)
        renamed = copy(actual)
        renamed.entities[i] = (role, actual.entities[i][1] + "_renamed")
        assert {m.kind for m in compare(expected, renamed, context)} >= {"entity"}
        lost = copy(actual)
        del lost.entities[i]
        assert {m.kind for m in compare(expected, lost, context)} >= {"entity"}
    steps = [i for i, (role, _) in enumerate(actual.entities) if role == "history"]
    swapped = copy(actual)
    swapped.entities[steps[0]], swapped.entities[steps[1]] = actual.entities[steps[1]], actual.entities[steps[0]]
    assert {m.kind for m in compare(expected, swapped, context)} == {"entity"}


def test_the_numbering_of_history_targets_is_excused_but_their_names_are_not():
    expected, actual, context = goal_case("C5A", "s2", "f01")
    first, second = [i for i, (role, _) in enumerate(actual.entities) if role == "history_target"][:2]
    swapped = copy(actual)
    swapped.entities[first], swapped.entities[second] = actual.entities[second], actual.entities[first]
    assert compare(expected, swapped, context) == []
    unexcused = [n for n in NAMES if n != "history_target_entity_order"]
    assert {m.kind for m in compare(expected, swapped, context, unexcused)} == {"entity"}


def test_the_history_of_a_record_keeps_only_its_newest_steps():
    fixture, task = load(prefix_path("f08"))
    records = [r for r in fixture["encodings"] if r["config"] == "C5B"]
    assert all(r["inputs"]["history_max_steps"] == 2 and len(r["inputs"]["history_subgoals"]) == 3 for r in records)
    expected, actual, context = goal_case("C5B", "s0", "f08")
    assert compare(expected, actual, context) == []
    steps = [name for role, name in actual.entities if role == "history"]
    assert steps == [f"history:{dt}#{i}" for i, (dt, _) in enumerate(context.history[-2:])]
    older = copy(actual)
    older.entities = [e for e in actual.entities if e[0] != "history"] + [("history", "history:-3#0")]
    assert {m.kind for m in compare(expected, older, context)} >= {"entity"}


def test_a_goal_literal_that_a_layer_lists_is_expected_once_in_the_goal():
    expected, actual, context = goal_case("C4_L1", "s2", "f01")
    assert "repeated_literal" in needed_differences(expected, actual, context)
    assert compare(expected, actual, context) == []
    layered = {parse_literal(literal) for layer in context.layers for literal in layer}
    shared = sorted(goal_literals(context) & layered, key=repr)
    assert shared
    entity = literal_text(*shared[0]) + "[g]"
    assert ("goal", entity) in actual.entities
    lost = copy(actual)
    lost.entities.remove(("goal", entity))
    assert {m.kind for m in compare(expected, lost, context)} == {"entity"}
    relation = next(n for n, ts in actual.tuples.items() if n.endswith("[g]") and any(t[0] == entity for t in ts))
    twice = copy(actual)
    twice.tuples[relation] = actual.tuples[relation] + [t for t in actual.tuples[relation] if t[0] == entity]
    assert [(m.kind, m.where) for m in compare(expected, twice, context)] == [("tuple", relation)]
    level = copy(actual)
    level.entities.append(("subgoal", entity.replace("[g]", "[sg]")))
    assert {m.kind for m in compare(expected, level, context)} == {"entity"}


def test_a_literal_in_two_layers_is_expected_once_at_the_lower_level():
    expected, actual, context = goal_case("C4_L2", "s3", "f09")
    assert "repeated_literal" in needed_differences(expected, actual, context)
    assert compare(expected, actual, context) == []
    shared = sorted({parse_literal(l) for l in context.layers[0]} & {parse_literal(l) for l in context.layers[1]},
                    key=repr)
    assert shared and not set(shared) & goal_literals(context)
    name = literal_text(*shared[0])
    assert ("subgoal", f"{name}[sg]") in actual.entities and ("subgoal", f"{name}[ssg]") not in actual.entities
    higher = copy(actual)
    higher.entities[higher.entities.index(("subgoal", f"{name}[sg]"))] = ("subgoal", f"{name}[ssg]")
    assert {m.kind for m in compare(expected, higher, context)} == {"entity"}


def test_a_literal_without_a_slot_is_left_out_of_the_native_inputs_and_only_that_literal():
    fixture, task = load(prefix_path("f08"))
    found = 0
    for state in ("s0", "s1", "s2", "s3", "s4"):
        expected, actual, context = goal_case("C5A", state, "f08")
        assert compare(expected, actual, context) == []
        for kind, where, text in context.dropped:
            found += 1
            atom = task.literal(text).atom
            assert atom.kind == "static" or (atom.slot if atom.kind == "fluent" else atom.derived_slot) is None
            assert kind == "history" and where.startswith("history:")
        if context.dropped:
            assert "literal_without_slot" in needed_differences(expected, actual, context)
            extra = copy(actual)
            relation = next(n for n, ts in actual.tuples.items() if ts and n.endswith("[hist]"))
            extra.tuples[relation] = actual.tuples[relation] + actual.tuples[relation][:1]
            assert [(m.kind, m.where) for m in compare(expected, extra, context)] == [("tuple", relation)]
    assert found


def test_a_nullary_literal_of_a_layer_is_a_tuple_over_the_auxiliary_entity():
    exercised = 0
    for prefix, config in (("f01", "C5BN"), ("f08", "C4_L2N"), ("f10", "C5BN"), ("f10", "C4_L1N")):
        for state in ("s0", "s1", "s2", "s3", "s4"):
            expected, actual, context = goal_case(config, state, prefix)
            assert compare(expected, actual, context) == []
            nullary = [n for n, a in actual.catalog if a == 0 and re.search(r"\[(hist|s{1,3}g)\]$", n)
                       and actual.tuples.get(n)]
            for name in nullary:
                exercised += 1
                assert all(t[-1] == AUXILIARY for t in actual.tuples[name])
                other = copy(actual)
                other.tuples[name] = [t[:-1] + (actual.entities[0][1],) for t in actual.tuples[name]]
                assert {(m.kind, m.where) for m in compare(expected, other, context)} == {("tuple", name)}
                lost = copy(actual)
                del lost.tuples[name]
                assert {(m.kind, m.where) for m in compare(expected, lost, context)} == {("tuple", name)}
    assert exercised >= 4


def test_a_record_that_lists_fewer_actions_excuses_only_the_unlisted_ones():
    expected, actual, context = goal_case("C3")
    listed = {name for role, name in expected.entities if role == "action"}
    assert listed < {name for role, name in actual.entities if role == "action"}
    kept = {name for role, name in actual.entities if role == "action" and name in listed}
    assert kept == listed
    name = next(n for n, ts in actual.tuples.items() if any(t[0] in listed for t in ts))
    victim = next(t for t in actual.tuples[name] if t[0] in listed)
    dropped = copy(actual)
    dropped.tuples[name] = [t for t in dropped.tuples[name] if t != victim]
    assert ("tuple", name) in {(m.kind, m.where) for m in compare(expected, dropped, context)}


def test_a_native_action_that_is_not_applicable_is_a_mismatch_even_when_the_record_lists_fewer_actions():
    expected, actual, context = goal_case("C3")
    assert compare(expected, actual, context) == []
    fake = "(bogus x y)"
    entity = copy(actual)
    entity.entities.append(("action", fake))
    assert {m.kind for m in compare(expected, entity, context)} == {"entity"}
    name = next(n for n, ts in actual.tuples.items() if any(t[0] in context.actions for t in ts))
    with_tuple = copy(entity)
    with_tuple.tuples[name] = with_tuple.tuples[name] + [(fake,) + actual.tuples[name][0][1:]]
    assert {m.kind for m in compare(expected, with_tuple, context)} >= {"entity", "tuple"}
    # an applicable action that the record does not list is excused, but the same entity under a name that is not an
    # applicable action is not
    unlisted = next(i for i, (r, n) in enumerate(actual.entities) if r == "action" and n not in context.actions)
    renamed = copy(actual)
    renamed.entities[unlisted] = ("action", fake)
    assert {m.kind for m in compare(expected, renamed, context)} >= {"entity"}


def test_the_goal_given_with_a_record_is_the_goal_of_its_graph():
    expected, actual, context = goal_case("C3", "s1")
    assert context.goals is not None
    named = {name for role, name in actual.entities if role == "goal"}
    assert named == {f"[{'+' if p else '-'}]({n} {' '.join(a)})[g]" for p, n, a in map(parse_literal, context.goals)}
    expected, actual, context = goal_case("C3", "s0")
    assert context.goals is None


def nullary_cases():
    for path in FIXTURES:
        if not hasattr(mymyr, "Domain"):
            pytest.skip("built without the PDDL front end")
        fixture = read(path)
        for state in (s["id"] for s in fixture["states"]):
            yield path, state


def test_the_auxiliary_entity_and_its_tuples_are_not_hidden():
    exercised = 0
    for path, state in nullary_cases():
        expected, actual, context = native_case(path, state)
        nullary = [n for n, a in actual.catalog if a == 0 and actual.tuples.get(n)]
        if not nullary:
            continue
        exercised += 1
        without_tuple = copy(actual)
        del without_tuple.tuples[nullary[0]]
        found = compare(expected, without_tuple, context)
        assert [(m.kind, m.where) for m in found] == [("tuple", nullary[0])]
        without_entity = copy(actual)
        without_entity.entities = [e for e in actual.entities if e[0] != "auxiliary"]
        assert {m.kind for m in compare(expected, without_entity, context)} == {"entity"}
        other = copy(actual)
        other.tuples[nullary[0]] = [(actual.entities[0][1],)]
        assert {(m.kind, m.where) for m in compare(expected, other, context)} == {("tuple", nullary[0])}
        assert any(role == "auxiliary" and name == AUXILIARY for role, name in actual.entities)
        for name in nullary:
            lost = copy(actual)
            lost.catalog = [c for c in actual.catalog if c[0] != name]
            del lost.tuples[name]
            assert {m.kind for m in compare(expected, lost, context)} == {"catalog", "tuple"}
    assert exercised >= 10


@pytest.mark.parametrize("path", [prefix_path("f10"), prefix_path("f08")], ids=["f10", "f08"])
def test_a_native_catalog_without_nullary_relations_is_not_accepted(path):
    expected, actual, context = native_case(path)
    nullary = {n for n, a in actual.catalog if a == 0}
    assert nullary
    bare = copy(actual)
    bare.catalog = [c for c in actual.catalog if c[1] != 0]
    bare.tuples = {n: ts for n, ts in actual.tuples.items() if n not in nullary}
    bare.entities = [e for e in actual.entities if e[0] != "auxiliary"]
    found = compare(expected, bare, context)
    assert {m.kind for m in found} >= {"catalog"} and any(m.detail.startswith("missing ('") for m in found)


class Rewritten:
    """A batch whose ``relation_args`` are replaced; every other attribute is the batch's."""

    def __init__(self, batch, relation_args):
        self._batch = batch
        self.relation_args = relation_args

    def __getattr__(self, name):
        return getattr(self._batch, name)


def reversed_tuples(encoder, batch, row):
    """The relation_args of ``batch`` with the tuples of every cell of ``row`` in reverse order."""
    arguments = np.array(batch.relation_args)
    relations = batch.num_relations
    for relation, arity in enumerate(encoder.relation_arities):
        cell = row * relations + relation
        first, last = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        arguments[first:last] = arguments[first:last].reshape(-1, arity)[::-1].reshape(-1)
    return arguments


def test_the_native_tuple_order_is_checked_not_excused():
    fixture, task = load(prefix_path("f03"))
    encoder = native_encoder(task, "C1")
    states = [replay(task, s["path"]) for s in fixture["states"]]
    batch = encoder.encode(states)
    row = len(states) - 1
    assert native_graph(encoder, batch, row, [task], COMPARED_LANES).unordered == ()
    rewritten = Rewritten(batch, reversed_tuples(encoder, batch, row))
    backwards = native_graph(encoder, rewritten, row, [task], COMPARED_LANES)
    assert backwards.unordered
    record = [r for r in fixture["encodings"] if r["config"] == "C1"][row]
    context = Context(fixture, task, fixture["states"][row], "C1")
    found = compare(fixture_graph(fixture, record, COMPARED_SOURCES), backwards, context)
    assert {m.kind for m in found} == {"order"} and {m.where for m in found} == set(backwards.unordered)


def test_arguments_outside_the_graph_are_refused():
    fixture, task = load(prefix_path("f03"))
    encoder = native_encoder(task, "C1")
    batch = encoder.encode([replay(task, s["path"]) for s in fixture["states"]])
    arguments = np.array(batch.relation_args) - int(batch.entity_offsets[1])
    with pytest.raises(AssertionError, match="outside the graph"):
        native_graph(encoder, Rewritten(batch, arguments), 1, [task], COMPARED_LANES)


def test_the_object_order_is_excused_only_for_the_task_order():
    expected, actual, context = native_case(prefix_path("f04"))
    objects = [n for r, n in actual.entities if r == "object"]
    assert objects == list(context.task.objects)
    remap = dict(zip(objects, objects[::-1]))
    backwards = copy(actual)
    backwards.entities = [(r, remap.get(n, n)) if r == "object" else (r, n) for r, n in actual.entities]
    backwards.tuples = {rel: [tuple(remap.get(n, n) for n in t) for t in ts] for rel, ts in actual.tuples.items()}
    assert {m.kind for m in compare(expected, backwards, context)} == {"entity", "tuple"}


def test_the_argument_order_of_helper_predicates_is_mapped_not_ignored():
    expected, actual, context = native_case(prefix_path("f05"))
    helpers = [(n, i, t) for n, i, t in tuples_of(actual) if HELPER.match(n) and t[0] != t[1]]
    assert helpers
    name, index, tuple_ = helpers[0]
    swapped = copy(actual)
    swapped.tuples[name][index] = tuple_[::-1]
    found = compare(expected, swapped, context)
    assert found and {m.where for m in found} == {name}
    # the pymimir order is not the native order: without the mapping the relation differs
    unmapped = [n for n in NAMES if n != "helper_argument_order"]
    assert {m.where for m in compare(expected, actual, context, unmapped)} == {n for n in actual.tuples if HELPER.match(n)}


def test_equality_is_excused_only_while_the_config_does_not_ask_for_it(monkeypatch):
    expected, actual, context = native_case(prefix_path("f05"))
    assert ("=", 2) in expected.catalog and all(n != "=" for n, _ in actual.catalog)
    assert compare(expected, actual, context) == []
    monkeypatch.setitem(SUPPORTED, "C1", {**SUPPORTED["C1"], "equality": True})
    assert {(m.kind, m.where) for m in compare(expected, actual, context)} >= {("catalog", None), ("tuple", "=")}
    with_equality = copy(actual)
    equality = [c for c in expected.catalog if base_predicate(c[0]) == "="]
    assert ("=", 2) in equality and len(equality) > 1
    with_equality.catalog = sorted(actual.catalog + equality, key=lambda c: c[0].encode())
    with_equality.tuples.update({n: list(ts) for n, ts in expected.tuples.items() if base_predicate(n) == "="})
    assert compare(expected, with_equality, context) == []


# ------------------------------------------------------------------------------------------------ native properties
@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_single_batch_and_slices_agree_on_the_fixture_states_and_walks(path):
    fixture, task = load(path)
    states = [replay(task, s["path"]) for s in fixture["states"]]
    states += [s for s in random_walk_rows(task, 7, seed=5) if s not in states]
    encoder = native_encoder(task, "C1")
    batch = encoder.encode(states)
    whole = [native_graph(encoder, batch, g, [task]) for g in range(len(states))]
    for g, state in enumerate(states):
        single = encoder.encode([state])
        assert native_graph(encoder, single, 0, [task]) == whole[g]
        assert np.array_equal(single.relation_counts[0], batch.relation_counts[g])
    count = len(states)
    for begin, end in [(0, 1), (1, count), (2, count - 3), (count // 2, count), (0, count)]:
        part = encoder.encode(states[begin:end])
        assert part.num_graphs == end - begin
        assert part.num_objects.tolist() == batch.num_objects[begin:end].tolist()
        assert np.array_equal(part.relation_counts, batch.relation_counts[begin:end])
        assert np.array_equal(part.entity_role, batch.entity_role[batch.entity_offsets[begin]:batch.entity_offsets[end]])
        for g in range(begin, end):
            assert native_graph(encoder, part, g - begin, [task]) == whole[g]
    backwards = encoder.encode(states[::-1])
    for g in range(count):
        assert native_graph(encoder, backwards, count - 1 - g, [task]) == whole[g]


def domain_groups():
    groups = {}
    for path in FIXTURES:
        groups.setdefault(read(path)["task"]["domain"], []).append(path)
    return [pytest.param(paths, id=paths[0].stem[:3]) for paths in groups.values() if len(paths) > 1]


@pytest.mark.parametrize("paths", domain_groups())
def test_a_table_of_instances_gives_each_row_its_single_task_graph(paths):
    tasks = [load(path)[1] for path in paths]
    states, ids = random_walk_batch(tasks, 6, seed=11)
    table = rl.TaskTable(tasks)
    encoder = encoding.FlatRelationEncoder(table, **SUPPORTED["C1"])
    batch = encoder.encode(states, task_ids=np.array(ids, dtype=np.int32))
    singles = [native_encoder(task, "C1") for task in tasks]
    assert len({task_id for task_id in ids}) == len(tasks)
    for row, (state, task_id) in enumerate(zip(states, ids)):
        alone = singles[task_id]
        graph = native_graph(encoder, batch, row, tasks)
        reference = native_graph(alone, alone.encode([state]), 0, [tasks[task_id]])
        assert graph.entities == reference.entities
        assert graph.tuples == reference.tuples


# ------------------------------------------------------------------------------------------------ the generator
def test_random_walk_rows_are_distinct_seeded_and_deterministic():
    _, task = load(prefix_path("f03"))
    first = random_walk_rows(task, 30, seed=3)
    assert first[0] == task.initial_state and len(first) == 30 and len(set(first)) == 30
    assert random_walk_rows(task, 30, seed=3) == first
    assert random_walk_rows(task, 30, seed=4) != first
    assert random_walk_rows(task, 10, seed=3) == first[:10]


def test_random_walk_rows_raise_when_the_space_is_too_small():
    _, task = load(prefix_path("f08"))
    with pytest.raises(ValueError, match="distinct states"):
        random_walk_rows(task, 10**6, seed=0, max_steps=3)


def test_random_walk_batches_are_deterministic_and_mix_the_tasks():
    tasks = [load(prefix_path(p))[1] for p in ("f01", "f02")]
    first = random_walk_batch(tasks, 8, seed=2)
    assert first == random_walk_batch(tasks, 8, seed=2) and first != random_walk_batch(tasks, 8, seed=3)
    states, ids = first
    assert len(states) == len(ids) == 16 and sorted(set(ids)) == [0, 1]
    assert ids != sorted(ids)
    for task_id, task in enumerate(tasks):
        mine = [s for s, i in zip(states, ids) if i == task_id]
        assert len(mine) == 8 and len(set(mine)) == 8


@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_the_fixture_states_are_distinct_graphs(path):
    fixture, task = load(path)
    encoder = native_encoder(task, "C1")
    states = [replay(task, s["path"]) for s in fixture["states"]]
    batch = encoder.encode(states)
    assert batch.num_graphs == len(states)
    rendered = {repr(native_graph(encoder, batch, g, [task])) for g in range(batch.num_graphs)}
    assert len(rendered) >= 3


# ------------------------------------------------------------------------------------------------ horizons
def horizon_case(config, prefix, state):
    expected, actual, context = goal_case(config, state=state, prefix=prefix)
    assert compare(expected, actual, context) == []
    return expected, actual, context


def encoding_view(task, record):
    table = rl.as_table(task)
    return horizon_view(table, [horizon_dag(task, table, record["inputs"]["dag"])])


def test_the_horizon_configs_are_compared_on_every_required_fixture():
    required = {"f01", "f02", "f03", "f05", "f07", "f08", "f09", "f10"}
    seen = {}
    for path, config in fixture_cases():
        if config.startswith("C7") and config in SUPPORTED:
            seen.setdefault(path.stem[:3], set()).add(config)
    assert required <= set(seen)
    assert seen["f01"] == {"C7A", "C7AN", "C7B", "C7C", "C7D", "C7DN", "C7G", "C7H", "C7I", "C7J", "C7K", "C7S", "C7T"}
    assert {c for c in PENDING_CONFIGS if c.startswith("C7")} == {"C7E", "C7EN", "C7F", "C7FN"}


def test_a_horizon_graph_has_state_entities_named_by_their_node():
    expected, actual, _ = horizon_case("C7A", "f01", "s2")
    states = [name for role, name in actual.entities if role == "state"]
    assert states == [f"target:{k}" for k in range(5)]
    assert all(("state", name) in expected.entities for name in states)


def test_the_excluded_root_has_no_entity_and_the_excusal_is_narrow():
    expected, actual, context = horizon_case("C7B", "f01", "s0")
    assert ("state", "_root_state_") in expected.entities
    assert ("state", "_root_state_") not in actual.entities and ("state", "target:0") not in actual.entities
    # a native graph that has the entity is a mismatch, and so is an entity that a tuple of the reference mentions
    with_entity = copy(actual)
    with_entity.entities.insert(3, ("state", "_root_state_"))
    assert {m.kind for m in compare(expected, with_entity, context)} == {"entity"}
    used = copy(expected)
    name = next(n for n, ts in used.tuples.items() if ts and n.startswith("[+]"))
    used.tuples[name] = [("_root_state_",) + used.tuples[name][0][1:]] + used.tuples[name][1:]
    found = compare(used, actual, context)
    assert any(m.kind == "entity" and "_root_state_" in m.detail for m in found)
    other = copy(actual)
    other.entities = [e for e in actual.entities if e != ("state", "target:1")]
    assert any(m.kind == "entity" for m in compare(expected, other, context))
    assert [m.detail for m in compare(expected, actual, context, allowed=()) if m.kind == "entity" and "state" in m.detail] \
        == ["missing ('state', '_root_state_')"]


@pytest.mark.parametrize("config, prefix, state", [("C7AN", "f10", "s2"), ("C7C", "f10", "s0")])
def test_horizon_nullary_tuples_are_checked_over_the_auxiliary_entity(config, prefix, state):
    expected, actual, context = horizon_case(config, prefix, state)
    nullary = [n for n, a in actual.catalog if a == 0 and actual.tuples.get(n) and not n.startswith("_")]
    assert nullary and ("auxiliary", AUXILIARY) in actual.entities
    for name in nullary:
        assert all(t[-1] == AUXILIARY for t in actual.tuples[name]), name
        lost = copy(actual)
        lost.tuples[name] = lost.tuples[name][1:]
        found = compare(expected, lost, context)
        assert found and {m.kind for m in found} == {"tuple"} and {m.where for m in found} == {name}
        extra = copy(actual)
        extra.tuples[name] = extra.tuples[name] + [(f"target:{len(actual.entities)}", AUXILIARY)]
        assert {m.kind for m in compare(expected, extra, context)} == {"tuple"}
    gone = copy(actual)
    gone.entities = [e for e in actual.entities if e[0] != "auxiliary"]
    assert {m.kind for m in compare(expected, gone, context)} == {"entity"}
    bare = copy(actual)
    bare.catalog = [c for c in actual.catalog if c[1] != 0 or c[0].startswith("_")]
    bare.tuples = {n: ts for n, ts in actual.tuples.items() if n not in nullary}
    bare.entities = gone.entities
    assert "catalog" in {m.kind for m in compare(expected, bare, context)}


def test_a_horizon_encoder_without_nullary_relations_has_none_and_no_auxiliary_entity():
    fixture, task = load(prefix_path("f10"))
    record = next(r for r in fixture["encodings"] if r["config"] == "C7AN")
    encoder = native_encoder(task, "C7AN", nullary=False)
    batch = encoder.encode(encoding_view(task, record))
    predicates = {p.name for p in task.formalism.predicates if p.arity == 0}
    assert predicates and not predicates & {base_predicate(n) for n in encoder.relation_names}
    assert "auxiliary" not in {role for role, _ in native_graph(encoder, batch, 0, [task], COMPARED_LANES).entities}


def test_the_static_goal_literal_at_the_nodes_is_excused_only_for_static_atoms():
    expected, actual, context = horizon_case("C7A", "f08", "s0")
    assert ("target:1", "a") in actual.tuples["[+]fixed[g][sat]"]
    assert ("target:1", "a") not in expected.tuples["[+]fixed[g][sat]"]
    lost = copy(actual)
    lost.tuples["[+]fixed[g][sat]"] = [t for t in actual.tuples["[+]fixed[g][sat]"] if t != ("target:0", "a")]
    assert [m.where for m in compare(expected, lost, context)] == ["[+]fixed[g][sat]"]
    fluent = next(n for n in actual.tuples if n.endswith("[g][sat]") and "fixed" not in n)
    mutated = copy(actual)
    mutated.tuples[fluent] = mutated.tuples[fluent][1:]
    assert {m.where for m in compare(expected, mutated, context)} == {fluent}
    assert "static_literal_at_nodes" in needed_differences(expected, actual, context)


def test_the_delta_relations_of_static_predicates_are_excused_only_while_empty():
    expected, actual, context = horizon_case("C7B", "f08", "s0")
    static = [n for n, _ in expected.catalog if "fixed" in n and n not in {c[0] for c in actual.catalog}]
    assert static
    filled = copy(expected)
    filled.tuples = {**expected.tuples, static[0]: [("target:1", "a")]}
    assert {m.kind for m in compare(filled, actual, context)} >= {"catalog", "tuple"}
    assert "delta_static_relations" in needed_differences(expected, actual, context)


def test_the_relations_no_graph_fills_are_excused_only_while_empty_and_missing():
    # the delta catalog: literal-level support relations
    expected, actual, context = horizon_case("C7I", "f01", "s0")
    own = {n for n, _ in actual.catalog}
    support = [n for n, a in expected.catalog if a == 2 and n not in own
               and re.search(r"^\[[+-]\]on\[(sat|unsat|sat\+|sat-)\]$", n)]
    assert support
    assert "never_filled_horizon_relations" in needed_differences(expected, actual, context)
    filled = copy(expected)
    filled.tuples = {**expected.tuples, support[0]: [("target:1", "a", "b")]}
    assert {m.kind for m in compare(filled, actual, context)} >= {"catalog", "tuple"}
    # the split catalog: the relation of the nodes of a static predicate
    expected, actual, context = horizon_case("C7G", "f08", "s0")
    own = {n for n, _ in actual.catalog}
    anchored = [n for n, _ in expected.catalog if n == "fixed[state]"]
    assert anchored and anchored[0] not in own
    assert "never_filled_horizon_relations" in needed_differences(expected, actual, context)


def test_a_literal_in_two_levels_is_expected_once_at_the_lowest_level_of_a_horizon():
    expected, actual, context = horizon_case("C7T", "f01", "s2")
    assert ("target:0", "c", "b") in actual.tuples["[+]on[g]"]
    assert "[+]on[sg]" not in actual.tuples
    assert "repeated_literal_horizon" in needed_differences(expected, actual, context)
    lost = copy(actual)
    lost.tuples["[+]on[g]"] = [t for t in actual.tuples["[+]on[g]"] if t != ("target:0", "c", "b")]
    assert any(m.where == "[+]on[g]" for m in compare(expected, lost, context))
    moved = copy(actual)
    moved.tuples["[+]on[sg]"] = [("target:0", "c", "b")]
    assert any(m.kind == "tuple" for m in compare(expected, moved, context))


def test_the_edge_arrays_give_the_action_and_parent_tuples_whatever_the_order_and_multiplicity_of_the_edges():
    fixture, task = load(prefix_path("f01"))
    record = next(r for r in fixture["encodings"] if r["config"] == "C7C" and r["state"] == "s2")
    view = encoding_view(task, record)
    encoder = native_encoder(task, "C7C")
    graph = native_graph(encoder, encoder.encode(view), 0, [task], COMPARED_LANES)
    assert graph.tuples["_parent_"] == [("target:0", "target:1"), ("target:0", "target:2"), ("target:1", "target:3"),
                                        ("target:2", "target:4")]
    assert graph.unordered == ()

    def with_edges(order):
        return type(view)(
            view.nodes, view.node_offsets, np.array([0, len(order)], np.int32), np.asarray(view.edge_src)[order],
            np.asarray(view.edge_dst)[order], np.asarray(view.edge_schema)[order], np.asarray(view.edge_binding)[order],
            view.task_ids)

    # the edges reordered, and one of them twice, give the same graph
    for order in (np.array([3, 1, 0, 2]), np.array([2, 0, 3, 1, 0, 3])):
        again = native_graph(encoder, encoder.encode(with_edges(order)), 0, [task], COMPARED_LANES)
        assert again == graph
        assert again.unordered == ()


@pytest.mark.parametrize("path, config", [c for c in case_params() if c.values[1].startswith("C7")])
def test_the_candidate_nodes_are_the_targets_of_the_horizon_graph(path, config):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    for record, placed in zip(records, encode_records(task, config, None, records)):
        graph = native_graph(placed.encoder, placed.batch, placed.row, [task], lanes_of(config))
        batch, row = placed.batch, placed.row
        first, last = (int(x) for x in batch.target_offsets[row : row + 2])
        base = int(batch.entity_offsets[row])
        names = [graph.entities[int(e) - base][1] for e in batch.target_entity[first:last]]
        groups = {placed.encoder.target_group_names[int(g)] for g in batch.target_group[first:last]}
        recorded = record["outputs"]["targets"]
        assert names == recorded["names"], (record["state"], names)
        assert groups <= set(recorded["groups"]) | set()
