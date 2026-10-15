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
    COMPARED_LANES, COMPARED_SOURCES, encode_records, FIXTURES, NAMES, PENDING_CONFIGS, PENDING_SOURCES, REGISTRY, SUPPORTED, AUXILIARY,
    HELPER, Context, base_predicate, Graph, Mismatch, compare, compare_exactly, fixture_cases, fixture_graph, load, native_encoder,
    native_graph, parse_literal, random_walk_batch, random_walk_rows, read, replay, run_case,
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
            found |= set(catalog["sources"]) - set(COMPARED_SOURCES)
    assert found == set(PENDING_SOURCES)
    assert set(COMPARED_SOURCES) & set(PENDING_SOURCES) == set()
    assert COMPARED_LANES == ("fluent", "static", "derived", "goal", "goal_derivation")


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


def goal_case(config, state="s1"):
    fixture, task = load(prefix_path("f03"))
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, placed = encode_records(task, config, rows, records)
    index = [r["state"] for r in records].index(state)
    batch, row = placed[index]
    goals = records[index]["inputs"].get("goals")
    context = Context(fixture, task, states[state], config, None if goals is None else tuple(goals))
    expected = fixture_graph(fixture, records[index], COMPARED_SOURCES)
    return expected, native_graph(encoder, batch, row, [task], COMPARED_LANES), context


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
