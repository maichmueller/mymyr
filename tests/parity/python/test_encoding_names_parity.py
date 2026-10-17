"""The names of the entities and tuples that the encoder reports (``entity_names``, ``describe``) against mifrost's
recorded encodings: the graph rebuilt from ``describe`` is compared with the fixture's through the registered
differences, exactly as the harness's own naming of a native graph is, and the two namings of the native graph agree."""

import pytest

from encoding_parity import (
    AUXILIARY, Context, SUPPORTED, Graph, compare, encode_records, fixture_cases, fixture_graph, lanes_of,
    load, native_graph, read, replay, sources_of,
)


def described_graph(encoder, batch, row, lanes):
    """Graph ``row`` of a batch as ``describe`` names it, restricted to the relations of the given lanes."""
    description = encoder.describe(batch, row)

    def name(entity):
        return AUXILIARY if entity == "auxiliary" else entity

    selected = [i for i, lane in enumerate(encoder.relation_lanes) if lane in lanes]
    names = encoder.relation_names
    tuples = {names[i]: [tuple(name(e) for e in t) for t in description.relations[names[i]]]
              for i in selected if names[i] in description.relations}
    catalog = [(names[i], encoder.relation_logical_arities[i]) for i in selected]
    return Graph(catalog, [(role, name(entity)) for role, entity in description.entities], tuples)


def cases():
    return [pytest.param(path, config, id=f"{path.stem[:3]}-{config}") for path, config in fixture_cases() if config in SUPPORTED]


@pytest.mark.parametrize("path, config", cases())
def test_the_described_graph_equals_mifrost_up_to_the_registered_differences(path, config):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    encoder, placed = encode_records(task, config, rows, records)
    lanes = lanes_of(config)
    for record, (batch, row) in zip(records, placed):
        goals = tuple(record["inputs"]["goals"]) if "goals" in record["inputs"] else None
        actions = tuple(record["inputs"]["actions"]) if "actions" in record["inputs"] else None
        context = Context(fixture, task, states[record["state"]], config, goals, actions)
        actual = described_graph(encoder, batch, row, lanes)
        assert compare(fixture_graph(fixture, record, sources_of(config)), actual, context) == [], (record["state"], config)
        # the harness names the same graph by its own rules
        harness = native_graph(encoder, batch, row, [task], lanes)
        assert actual.entities == harness.entities
        assert actual.tuples == harness.tuples
        assert actual.catalog == harness.catalog
        names = encoder.entity_names(batch, row)
        assert [(role, AUXILIARY if n == "auxiliary" else n) for (role, _), n in zip(actual.entities, names)] == actual.entities


def test_every_named_role_of_the_fixtures_is_covered():
    roles = set()
    for path, config in fixture_cases():
        if config not in SUPPORTED:
            continue
        fixture = read(path)
        for record in fixture["encodings"]:
            if record["config"] == config and "entities" in record["outputs"]:
                roles |= {role for role, names in record["outputs"]["entities"]["by_role"].items() if names}
    assert {"object", "goal", "action", "predicate_virtual"} <= roles
