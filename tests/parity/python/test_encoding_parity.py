"""Parity of the native relational encoder with mifrost's recorded encodings (tests/parity/data/encoding/mifrost).

Every fixture x config is compared by names (``encoding_parity``): the configs the encoder supports must have no
mismatch beyond the registered differences, the others are skipped with the lane that turns them on. The harness is
tested on its own (a fixture against itself, mutated copies, the narrowness of the registered differences), and the
native encoder is tested for what parity cannot show: a graph does not depend on the rows around it."""

import itertools

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from encoding_parity import (
    COMPARED_LANES, COMPARED_SOURCES, FIXTURES, NAMES, PENDING_CONFIGS, PENDING_SOURCES, REGISTRY, SUPPORTED, AUXILIARY,
    HELPER, Context, Graph, Mismatch, compare, compare_exactly, fixture_cases, fixture_graph, load, native_encoder,
    native_graph, random_walk_batch, random_walk_rows, read, replay, run_case,
)

IDS = [p.stem[:3] for p in FIXTURES]
CONFIGS = sorted({config for _, config in fixture_cases()})


def case_params(selected):
    """pytest parameters for the (fixture, config) cases: compared ones run, the others skip with their lane."""
    params = []
    for path, config in fixture_cases():
        marks = []
        if config not in SUPPORTED:
            marks.append(pytest.mark.skip(reason=f"pending: {PENDING_CONFIGS.get(config, 'unlisted')}"))
        params.append(pytest.param(path, config, id=f"{path.stem[:3]}-{config}", marks=marks))
    return [p for p in params if selected(p.values[1])]


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
            catalog = fixture["catalogs"][fixture["configs"][config]["catalog"]]
            found |= set(catalog["sources"]) - set(COMPARED_SOURCES)
    assert found == set(PENDING_SOURCES)
    assert set(COMPARED_SOURCES) & set(PENDING_SOURCES) == set()
    assert COMPARED_LANES == ("fluent", "static", "derived")


@pytest.mark.parametrize("config", sorted(SUPPORTED))
@pytest.mark.parametrize(
    "source", [pytest.param(s, marks=pytest.mark.skip(reason=f"pending: {lane}")) for s, lane in PENDING_SOURCES.items()]
)
def test_relations_of_pending_sources_in_a_compared_config(config, source):
    pytest.fail("the relations of this source are compared once its lane exists")


# ------------------------------------------------------------------------------------------------ parity
@pytest.mark.parametrize("path, config", case_params(lambda config: True))
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
    for name in NAMES:
        assert compare(expected, actual, context, [n for n in NAMES if n != name]), name


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
    assert exercised >= 10


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


def test_equality_is_compared_when_the_native_catalog_has_it():
    expected, actual, context = native_case(prefix_path("f05"))
    assert ("=", 2) in expected.catalog and all(n != "=" for n, _ in actual.catalog)
    with_equality = copy(actual)
    with_equality.catalog = sorted(actual.catalog + [("=", 2)], key=lambda c: c[0].encode())
    found = compare(expected, with_equality, context)
    assert {(m.kind, m.where) for m in found} == {("tuple", "=")}
    with_equality.tuples["="] = list(expected.tuples["="])
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
