"""Parity: the native encoder's state, static and derived lanes against the recorded reference encodings
(tests/parity/data/encoding/mifrost)."""

import pytest

from mymyr import encoding

from test_encoding_fixtures_parity import FIXTURES, HELPER, load, replay


def encoder_of(task, **options):
    return encoding.FlatRelationEncoder(task, **{"goal_derivations": (), **options})


def lane_relations(encoder, lanes=("fluent", "static", "derived")):
    return [i for i, lane in enumerate(encoder.relation_lanes) if lane in lanes]


def unpack(batch, graph):
    """{relation name: tuples of instance-local entity ids} of one graph, for the relations that have tuples."""
    output = {}
    relations = batch.num_relations
    names = batch.schema.relation_names
    arities = batch.schema.relation_arities
    base = int(batch.entity_offsets[graph])
    for relation, name in enumerate(names):
        count = int(batch.relation_counts[graph, relation])
        if not count:
            continue
        cell = graph * relations + relation
        begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        assert end - begin == count * arities[relation]
        assert batch.relation_offsets[cell + 1] - batch.relation_offsets[cell] == count
        arguments = batch.relation_args[begin:end] - base
        output[name] = [tuple(int(x) for x in row) for row in arguments.reshape(count, arities[relation])]
    return output


def reference(task, state, include_static=True, include_derived=True, nullary=True):
    """The tuples of a state by relation name, from its atoms; a nullary atom is the tuple over the auxiliary entity."""
    atoms = list(state.atoms())
    if include_derived:
        atoms += list(state.derived_atoms())
    if include_static:
        atoms += list(task.formalism.static_init)
    auxiliary = len(task.objects)
    output = {}
    for atom in atoms:
        name = atom.predicate.name
        if name == "=":
            continue
        arguments = tuple(obj.index for obj in atom.objects)
        if not arguments:
            if not nullary:
                continue
            arguments = (auxiliary,)
        output.setdefault(name, set()).add(arguments)
    return {name: sorted(values) for name, values in output.items()}


def mifrost_tuples(fixture, relations, name, helper):
    entry = relations.get(name)
    if entry is None:
        return []
    tuples = [tuple(t) for t in fixture["tuples"][entry["emit"]]]
    return sorted(tuple(sorted(t)) if helper else t for t in tuples)


@pytest.fixture(params=FIXTURES, ids=[p.stem[:3] for p in FIXTURES])
def parity_case(request):
    return load(request.param)


def test_c1_state_static_and_derived_lanes_equal_mifrost(parity_case):
    """mifrost's default flat encoding (config C1) against the state, static and derived lanes of the native one.

    Catalogs are compared as ordered lists and tuples as name multisets. The differences are the allowed ones: tuple
    order (the multisets are compared), object ids (names are compared), nullary relations (mifrost drops them, mymyr
    has one auxiliary entity), the equality predicate (absent) and the argument order of the helper predicates
    `axiom_<k>` (compared sorted)."""
    fixture, task = parity_case
    config = fixture["configs"]["C1"]
    assert config["encoder"] == "FlatRelationEncoder" and not config["options"]
    catalog = fixture["catalogs"][config["catalog"]]
    source_of = dict(zip(catalog["names"], catalog["sources"]))
    mifrost = [
        (name, arity)
        for name, arity, source in zip(catalog["names"], catalog["arities"], catalog["sources"])
        if source == "state" and name != "="  # the equality predicate is absent
    ]
    encoder = encoder_of(task, include_actions=False)
    state_ids = lane_relations(encoder)
    native = [(encoder.relation_names[i], encoder.relation_logical_arities[i]) for i in state_ids]
    nullary = {name for name, arity in native if arity == 0}
    # mifrost lists a nullary predicate with arity 0 or not at all
    assert [(n, a) for n, a in mifrost if a != 0] == [(n, a) for n, a in native if a != 0]
    assert {n for n, a in mifrost if a == 0} <= nullary

    records = [r for r in fixture["encodings"] if r["config"] == "C1"]
    paths = {s["id"]: s for s in fixture["states"]}
    states = [replay(task, paths[r["state"]]["path"]) for r in records]
    batch = encoder.encode(states)
    names = task.objects
    static_nullary = {a.strip("()") for a in fixture["static_atoms"] if " " not in a.strip("()")}
    for graph, (record, state) in enumerate(zip(records, states)):
        outputs = record["outputs"]
        got = unpack(batch, graph)
        objects = len(names)
        for name, arity in native:
            if arity == 0:
                continue
            helper = bool(HELPER.match(name))
            tuples = sorted(
                tuple(sorted(names[i] for i in t)) if helper else tuple(names[i] for i in t) for t in got.get(name, [])
            )
            assert tuples == mifrost_tuples(fixture, outputs["relations"], name, helper), (record["state"], name)
        recorded = {n for n in outputs["relations"] if source_of[n] == "state"}
        assert recorded <= {n for n, _ in mifrost} | {"="}, recorded
        assert sorted(outputs["entities"]["by_role"]["object"]) == sorted(names)
        # the nullary relations hold the auxiliary entity exactly for the nullary atoms that hold
        fixture_state = paths[record["state"]]
        holding = {a.strip("()") for a in fixture_state["fluent"] + fixture_state["derived"]
                   if " " not in a.strip("()")} | static_nullary
        for name in nullary:
            expected = [(objects,)] if name in holding else []
            assert got.get(name, []) == expected, (record["state"], name)
        has_auxiliary = bool(holding & nullary)
        begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
        assert end - begin == objects + has_auxiliary


def test_c1_fixture_states_are_distinct_graphs(parity_case):
    fixture, task = parity_case
    encoder = encoder_of(task, include_actions=False)
    states = [replay(task, s["path"]) for s in fixture["states"]]
    batch = encoder.encode(states)
    assert batch.num_graphs == len(states)
    rendered = {repr(sorted(unpack(batch, g).items())) for g in range(batch.num_graphs)}
    assert len(rendered) >= 3
