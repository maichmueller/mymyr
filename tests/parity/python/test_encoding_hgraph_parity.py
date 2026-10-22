"""Parity of the HGraph view of the native encoder with mifrost's recorded HGraph encodings (C9A to C9C).

Each record is compared by names (``encoding_hgraph_parity``): the nodes of every node type, the edges of every edge type
in both directions, and the edge types themselves. The native graph is built from the view alone. The differences the
view has on purpose are the entries of ``HGRAPH_REGISTRY``; the harness is tested on its own as the flat one is."""

import numpy as np
import pytest

from encoding_hgraph_parity import (
    HGRAPH_CONFIGS, HGRAPH_NAMES, HGRAPH_REGISTRY, SYMBOL, compare_hgraph, encode_hgraph_records, hgraph_cases,
    is_hgraph, native_hgraph, needed_hgraph_differences, reference_graph, run_hgraph_case,
)
from encoding_parity import AUXILIARY, FIXTURES, Graph, fixture_cases, load, read, record_context, replay


def case_params():
    return [pytest.param(path, config, id=f"{path.stem[:3]}-{config}") for path, config in hgraph_cases()]


def prefix_path(prefix):
    return next(p for p in FIXTURES if p.stem.startswith(prefix))


# ------------------------------------------------------------------------------------------------ the tables
def test_the_hgraph_configs_are_the_c9_configs_of_the_fixtures():
    found = {config for _, config in fixture_cases() if config.startswith("C9")}
    assert found == set(HGRAPH_CONFIGS)
    seen = {}
    for path, config in hgraph_cases():
        seen.setdefault(path.stem[:3], set()).add(config)
    assert seen == {
        "f01": {"C9A", "C9B", "C9C"}, "f02": {"C9A", "C9B", "C9C"}, "f08": {"C9A", "C9C"}, "f09": {"C9A", "C9C"},
        "f10": {"C9A"}, "f11": {"C9A"}, "f12": {"C9A"}, "f13": {"C9A"},
    }


def test_the_nullary_hgraph_config_is_recorded_only_where_the_reference_can_encode_it():
    # the reference cannot encode a nullary goal literal with nullary predicates on, so C9B has no fixture that has one
    for path, config in hgraph_cases():
        if config == "C9B":
            fixture = read(path)
            assert not [r for r in fixture["encodings"] if r["config"] == "C9B" and "error" in r["outputs"]]
    assert {path.stem[:3] for path, config in hgraph_cases() if config == "C9B"} == {"f01", "f02"}


# ------------------------------------------------------------------------------------------------ parity
@pytest.mark.parametrize("path, config", case_params())
def test_native_hgraphs_equal_mifrost_up_to_the_registered_differences(path, config):
    fixture = read(path)
    expected_states = [r["state"] for r in fixture["encodings"] if r["config"] == config]
    result = run_hgraph_case(path, config)
    assert list(result.mismatches) == expected_states
    report = [f"state {state}: {m}" for state, found in result.mismatches.items() for m in found]
    assert not report, "\n" + "\n".join(report)


@pytest.mark.parametrize("path, config", case_params())
def test_the_reverse_edge_types_hold_the_edges_of_the_forward_ones_the_other_way_round(path, config):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    from mymyr import encoding

    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, batch = encode_hgraph_records(task, config, rows)
    view = encoding.hgraph(batch)
    for row, record in enumerate(records):
        expected = reference_graph(fixture, record)
        actual = native_hgraph(encoder, batch, row, task, view)
        # the reverse edge types hold the edges of the forward ones the other way round, in the reference as here
        for graph in (expected, actual):
            for name, pairs in graph.tuples.items():
                source, position, destination = name.split("|")
                if source == SYMBOL:
                    assert sorted(pairs) == sorted((b, a) for a, b in graph.tuples[f"{destination}|{position}|{source}"])
        assert {n for n, _ in actual.catalog} == {n for n, _ in expected.catalog}


# ------------------------------------------------------------------------------------------------ the entity feature
@pytest.mark.parametrize("path, config", case_params())
def test_the_entity_role_feature_says_object_auxiliary_or_action(path, config):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    from mymyr import encoding

    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, batch = encode_hgraph_records(task, config, rows)
    view = encoding.hgraph(batch)
    roles = np.asarray(view.entity_role)
    pointer = np.asarray(view.ptr(SYMBOL))
    role_names = batch.schema.entity_role_names
    for row, record in enumerate(records):
        expected = reference_graph(fixture, record)
        names = [n for kind, n in native_hgraph(encoder, batch, row, task, view).entities if kind == SYMBOL]
        own = [role_names[int(r)] for r in roles[pointer[row] : pointer[row + 1]]]
        symbols = [n for kind, n in expected.entities if kind == SYMBOL]
        assert len(own) == len(symbols) == len(names)
        objects = set(task.objects)
        for name, role in zip(names, own):
            expectation = "object" if name in objects else "auxiliary" if name == AUXILIARY else "action"
            assert role == expectation, (record["state"], name)


# ------------------------------------------------------------------------------------------------ the registry
def test_the_registry_is_documented_and_unique():
    assert len(set(HGRAPH_NAMES)) == len(HGRAPH_NAMES)
    for difference in HGRAPH_REGISTRY:
        assert difference.reason.strip() and difference.scope.strip() and difference.cases, difference.name
        assert all(is_hgraph(config) for _, config in difference.cases)


@pytest.mark.parametrize("difference", HGRAPH_REGISTRY, ids=HGRAPH_NAMES)
def test_the_cases_of_a_registered_difference_need_it(difference):
    for prefix, config in difference.cases:
        assert difference.name in run_hgraph_case(prefix_path(prefix), config).needed, (prefix, config)


def test_no_registered_difference_is_stale():
    needed = set()
    for path, config in hgraph_cases():
        needed |= run_hgraph_case(path, config).needed
    assert needed == set(HGRAPH_NAMES)


def test_without_the_registry_the_exact_comparison_finds_the_differences():
    path = prefix_path("f02")
    fixture, task = load(path)
    states = {s["id"]: s for s in fixture["states"]}
    from mymyr import encoding

    for config in ("C9A", "C9B", "C9C"):
        records = [r for r in fixture["encodings"] if r["config"] == config]
        rows = [replay(task, states[r["state"]]["path"]) for r in records]
        encoder, batch = encode_hgraph_records(task, config, rows)
        view = encoding.hgraph(batch)
        expected = reference_graph(fixture, records[0])
        actual = native_hgraph(encoder, batch, 0, task, view)
        context = record_context(fixture, task, states[records[0]["state"]], config, records[0])
        assert compare_hgraph(expected, actual, context) == []
        assert compare_hgraph(expected, actual, context, allowed=())
        needed = needed_hgraph_differences(expected, actual, context)
        assert needed
        for name in needed:
            assert compare_hgraph(expected, actual, context, [n for n in HGRAPH_NAMES if n != name]), name


# ------------------------------------------------------------------------------------------------ the harness itself
def native_and_reference(config="C9C", prefix="f01", state="s3"):
    path = prefix_path(prefix)
    fixture, task = load(path)
    states = {s["id"]: s for s in fixture["states"]}
    records = [r for r in fixture["encodings"] if r["config"] == config]
    from mymyr import encoding

    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, batch = encode_hgraph_records(task, config, rows)
    view = encoding.hgraph(batch)
    index = next(i for i, r in enumerate(records) if r["state"] == state)
    context = record_context(fixture, task, states[state], config, records[index])
    return reference_graph(fixture, records[index]), native_hgraph(encoder, batch, index, task, view), context


def copy(graph):
    return Graph(list(graph.catalog), list(graph.entities), {n: list(ts) for n, ts in graph.tuples.items()})


def test_a_graph_compared_with_itself_has_no_mismatch():
    expected, actual, context = native_and_reference()
    assert compare_hgraph(expected, copy(expected), context, allowed=()) == []
    assert compare_hgraph(actual, copy(actual), context, allowed=()) == []


def test_a_changed_edge_a_missing_node_and_a_missing_edge_type_are_found():
    expected, actual, context = native_and_reference()
    assert compare_hgraph(expected, actual, context) == []
    edge = next(n for n, ts in actual.tuples.items() if len(ts) > 1 and n.split("|")[0] != SYMBOL)
    changed = copy(actual)
    changed.tuples[edge][0] = (changed.tuples[edge][0][0], changed.tuples[edge][1][1])
    assert {m.kind for m in compare_hgraph(expected, changed, context)} == {"tuple"}
    dropped = copy(actual)
    dropped.entities.remove(next(e for e in dropped.entities if e[0] == SYMBOL))
    assert "entity" in {m.kind for m in compare_hgraph(expected, dropped, context)}
    narrower = copy(actual)
    narrower.catalog.pop()
    assert "catalog" in {m.kind for m in compare_hgraph(expected, narrower, context)}


def test_a_registered_difference_does_not_hide_a_change_outside_its_scope():
    expected, actual, context = native_and_reference(config="C9B", prefix="f01", state="s0")
    assert compare_hgraph(expected, actual, context) == []
    # the auxiliary entity's edges are renamed only for the reference's own name for it
    changed = copy(actual)
    edge = next(n for n in changed.tuples if n.startswith("handempty|"))
    changed.tuples[edge] = [("(handempty)", "a")]
    assert compare_hgraph(expected, changed, context)
    # sorted lists do not make a different edge equal
    pairs = copy(actual)
    edge = next(n for n, ts in pairs.tuples.items() if len(ts) > 1 and n.split("|")[0] == SYMBOL)
    pairs.tuples[edge] = pairs.tuples[edge][:-1]
    assert compare_hgraph(expected, pairs, context)
