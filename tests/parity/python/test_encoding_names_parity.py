"""The names of the entities and tuples that the encoder reports (``entity_names``, ``describe``) against mifrost's
recorded encodings: the graph rebuilt from ``describe`` is compared with the fixture's through the registered
differences, exactly as the harness's own naming of a native graph is, and the two namings of the native graph agree."""

import pytest

from encoding_parity import (
    AUXILIARY, SUPPORTED, Graph, compare, encode_records, fixture_cases, fixture_graph, lanes_of, load,
    native_graph_in_native_order, read, record_context, reorder, replay, sources_of, to_reference,
)


def described_graph(encoder, batch, row, lanes, task):
    """Graph ``row`` of a batch as ``describe`` names it, restricted to the relations of the given lanes, in the native
    order, with the reference's helper names."""
    description = encoder.describe(batch, row)

    def name(entity):
        return AUXILIARY if entity == "auxiliary" else to_reference(entity, task)

    selected = [i for i, lane in enumerate(encoder.relation_lanes) if lane in lanes]
    names = encoder.relation_names
    tuples = {to_reference(names[i], task): [tuple(name(e) for e in t) for t in description.relations[names[i]]]
              for i in selected if names[i] in description.relations}
    catalog = [(to_reference(names[i], task), encoder.relation_logical_arities[i]) for i in selected]
    return Graph(catalog, [(role, name(entity)) for role, entity in description.entities], tuples)


def cases():
    return [pytest.param(path, config, id=f"{path.stem[:3]}-{config}") for path, config in fixture_cases() if config in SUPPORTED]


@pytest.mark.parametrize("path, config", cases())
def test_the_described_graph_equals_mifrost_up_to_the_registered_differences(path, config):
    fixture, task = load(path)
    records = [r for r in fixture["encodings"] if r["config"] == config]
    states = {s["id"]: s for s in fixture["states"]}
    rows = [replay(task, states[r["state"]]["path"]) for r in records]
    lanes = lanes_of(config)
    for record, placed in zip(records, encode_records(task, config, rows, records)):
        encoder, batch, row = placed.encoder, placed.batch, placed.row
        context = record_context(fixture, task, states[record["state"]], config, record, placed.dropped)
        actual = described_graph(encoder, batch, row, lanes, task)
        # the harness names the same graph by its own rules
        harness, catalog_order, entity_order = native_graph_in_native_order(encoder, batch, row, [task], lanes)
        assert actual.entities == harness.entities
        assert actual.tuples == harness.tuples
        assert actual.catalog == harness.catalog
        names = encoder.entity_names(batch, row)
        assert [(role, AUXILIARY if n == "auxiliary" else to_reference(n, task))
                for (role, _), n in zip(actual.entities, names)] == actual.entities
        expected = fixture_graph(fixture, record, sources_of(config))
        assert compare(expected, reorder(actual, catalog_order, entity_order), context) == [], (record["state"], config)


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
