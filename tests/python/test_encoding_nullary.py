"""The encodings of the nullary predicates (``nullary='auxiliary'``, ``'arity0'``, ``'omit'``): the catalog under each
choice, and the arity-0 encoding as the auxiliary one without the auxiliary entity, over state, goal, subgoal and
history relations with and without targets, predicate virtual nodes, both layouts, horizons, the Simple graph, the
HGraph view, a nullary numeric function, collation, streams and the validator."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from test_encoding_simple import expect_simple_of

AUXILIARY_ROLE = 8
AUXILIARY_SLOT = 8
GOALS = ROOT / "tests/data/pddl/goals"
IPC = ROOT / "tests/data/ipc"
ALL = ("plain", "satisfied", "unsatisfied")


def pddl(domain, problem):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task.from_pddl(domain, problem)


def blocksworld():
    return pddl(IPC / "blocksworld-ipc/test/domain.pddl", IPC / "blocksworld-ipc/test/p01-easy.pddl")


def ferry():
    return pddl(IPC / "ferry-ipc/test/domain.pddl", IPC / "ferry-ipc/test/p01-easy.pddl")


def conjunctive():
    # (flag) is a fluent nullary predicate, (done) a derived one that the goal holds
    return pddl(GOALS / "domain.pddl", GOALS / "conjunctive.pddl")


# the task and its nullary predicate
TASKS = {"blocksworld": (blocksworld, "arm-empty"), "ferry": (ferry, "empty-ferry"), "conjunctive": (conjunctive, "flag")}


def nullary_literals(task):
    """Literals of the task's nullary predicates that have a slot, both signs."""
    names = [p.name for p in task.formalism.predicates if p.arity == 0 and p.kind != "static"]
    return [text for name in names for text in (f"({name})", f"(not ({name}))")]


def layers_and_history(task, states, seed):
    """Subgoal layers and histories of the states: their nullary literals and some atoms of the state."""
    rng = np.random.default_rng(seed)
    nullary = nullary_literals(task)
    layers, history = [], []
    for state in states:
        atoms = [f"({' '.join([a.predicate.name, *(o.name for o in a.objects)])})" for a in state.atoms()]
        pick = lambda k: [nullary[i] for i in rng.choice(len(nullary), size=min(k, len(nullary)), replace=False)]
        layer = sorted(set(pick(1) + atoms[:2]) - {f"(not {x})" for x in atoms[:2]})
        layers.append([[x for x in layer if not (x.startswith("(not ") and x[5:-1] in layer)]])
        history.append([(-1, pick(1)), (0, pick(1) + atoms[2:3])])
    return layers, history


OPTION_SETS = {
    "state": {},
    "goal_targets_virtual": dict(goal_derivations=ALL, target_sources=("goal",), predicate_virtual_nodes=True),
    "layers_history_targets": dict(goal_derivations=ALL, max_goal_level=1, history=True,
                                   target_sources=("goal", "subgoal", "history")),
    "layers_history_virtual": dict(max_goal_level=1, history=True, predicate_virtual_nodes=True),
}


def encode(task, states, nullary, layout, options):
    encoder = encoding.FlatRelationEncoder(task, nullary=nullary, layout=layout, **options)
    extra = {}
    if options.get("max_goal_level") or options.get("history"):
        layers, history = layers_and_history(task, states, seed=len(states))
        if options.get("max_goal_level"):
            extra["subgoals"] = encoding.subgoal_masks(encoder, layers)
        if options.get("history"):
            extra["history"] = encoding.history_masks(encoder, history)
    return encoder, encoder.encode(states, **extra)


def without_auxiliary(description):
    """A graph description with the auxiliary entity taken out of its entities and tuples."""
    entities = tuple(e for e in description.entities if e[0] != "auxiliary")
    relations = {name: tuple(tuple(x for x in t if x != "auxiliary") for t in ts)
                 for name, ts in description.relations.items()}
    return entities, relations


def expect_valid(batch):
    """The batch passes the validator of its schema."""
    again = encoding.FlatBatch.from_arrays(batch.schema, batch.as_dict())
    assert again.num_graphs == batch.num_graphs


# ------------------------------------------------------------------------------------------------ the option
def test_the_option_is_a_choice_of_three():
    task = text_task("blocks__probBLOCKS-8-0")
    assert encoding.EncodingSchema(task).options["nullary"] == "auxiliary"
    for value in ("auxiliary", "arity0", "omit"):
        assert encoding.EncodingSchema(task, nullary=value).options["nullary"] == value
    with pytest.raises(TypeError, match="nullary must be a string"):
        encoding.EncodingSchema(task, nullary=True)
    with pytest.raises(ValueError, match="nullary must be one of 'auxiliary', 'arity0', 'omit'"):
        encoding.EncodingSchema(task, nullary="none")


@pytest.mark.parametrize("name", ["blocksworld", "ferry"])
def test_the_catalog_under_each_choice(name):
    make, predicate = TASKS[name]
    task = make()
    common = dict(goal_derivations=ALL, history=True, target_sources=("goal", "history"))
    schemas = {v: encoding.EncodingSchema(task, nullary=v, **common) for v in ("auxiliary", "arity0", "omit")}
    roles = {v: dict(zip(s.relation_names, s.relation_slot_roles)) for v, s in schemas.items()}
    arities = {v: dict(zip(s.relation_names, s.relation_arities)) for v, s in schemas.items()}
    # auxiliary: the auxiliary slot after the leading slots; arity0: the leading slots only; omit: no relation
    assert roles["auxiliary"][predicate] == [AUXILIARY_SLOT] and roles["arity0"][predicate] == []
    assert arities["auxiliary"][predicate] == 1 and arities["arity0"][predicate] == 0
    assert roles["auxiliary"][f"[+]{predicate}[g]"] == [4, AUXILIARY_SLOT] and roles["arity0"][f"[+]{predicate}[g]"] == [4]
    assert roles["arity0"][f"[+]{predicate}[g][sat]"] == []
    assert roles["arity0"][f"[-]{predicate}[hist]"] == [6, 7]
    assert not [n for n in schemas["omit"].relation_names if predicate in n.replace("[", " ").replace("]", " ").split()]
    # the catalogs agree but for the slots of the nullary relations
    assert schemas["auxiliary"].relation_names == schemas["arity0"].relation_names
    logical = dict(zip(schemas["arity0"].relation_names, schemas["arity0"].relation_logical_arities))
    for relation, slots in roles["auxiliary"].items():
        expected = [r for r in slots if r != AUXILIARY_SLOT] if logical[relation] == 0 else slots
        assert roles["arity0"][relation] == expected, relation
    assert AUXILIARY_SLOT not in {r for slots in roles["arity0"].values() for r in slots}
    assert len({s.schema_fingerprint for s in schemas.values()}) == 3
    virtual = encoding.EncodingSchema(task, nullary="arity0", predicate_virtual_nodes=True)
    assert dict(zip(virtual.relation_names, virtual.relation_slot_roles))[predicate] == [1]
    # a schema record keeps the choice
    record = encoding.schema_to_dict(schemas["arity0"])
    assert record["options"]["nullary"] == "arity0"
    assert encoding.schema_from_dict(record, task).schema_fingerprint == schemas["arity0"].schema_fingerprint


# ------------------------------------------------------------------------------------------------ the encoding
@pytest.mark.parametrize("name", list(TASKS))
@pytest.mark.parametrize("options", list(OPTION_SETS))
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_arity0_is_the_auxiliary_encoding_without_the_auxiliary_entity(name, options, layout):
    make, predicate = TASKS[name]
    task = make()
    states = walk(task, steps=6, seed=3, walks=2)
    if name == "conjunctive":  # a state with (flag)
        states.append(task.apply(states[-1], "(raise)"))
    options = OPTION_SETS[options]
    aux_encoder, aux = encode(task, states, "auxiliary", layout, options)
    encoder, batch = encode(task, states, "arity0", layout, options)
    expect_valid(batch)
    assert AUXILIARY_ROLE not in np.asarray(batch.entity_role).tolist()
    roles = np.asarray(aux.entity_role)
    with_auxiliary = [int((roles[aux.entity_offsets[g] : aux.entity_offsets[g + 1]] == AUXILIARY_ROLE).sum())
                      for g in range(aux.num_graphs)]
    assert any(with_auxiliary)
    assert np.array_equal(np.diff(batch.entity_offsets), np.diff(aux.entity_offsets) - with_auxiliary)
    assert np.array_equal(batch.relation_counts, aux.relation_counts)
    assert np.array_equal(batch.goal_complete, aux.goal_complete)
    for g in range(batch.num_graphs):
        assert without_auxiliary(aux_encoder.describe(aux, g)) == without_auxiliary(encoder.describe(batch, g))
        assert "auxiliary" not in encoder.entity_names(batch, g)
    # a state tuple of the predicate is empty, or holds the virtual node of its predicate
    relation = encoder.relation_id(predicate)
    holds = [any(a.predicate.name == predicate for a in s.atoms()) for s in states]
    assert batch.relation_counts[:, relation].tolist() == [int(h) for h in holds] and any(holds)
    width = 1 if options.get("predicate_virtual_nodes") else 0
    for g, h in enumerate(holds):
        if h:
            expected = (f"predicate:{predicate}",) if width else ()
            assert encoder.describe(batch, g).relations[predicate] == (expected,)


def test_the_goal_stays_complete_and_its_nullary_literals_are_target_tuples():
    task = conjunctive()  # the goal holds the derived nullary literal (done)
    states = walk(task, steps=4, seed=5)
    states.append(task.apply(states[-1], "(raise)"))
    batches = {v: encoding.FlatRelationEncoder(task, nullary=v, target_sources=("goal",)).encode(states)
               for v in ("auxiliary", "arity0", "omit")}
    assert batches["arity0"].goal_complete.tolist() == batches["auxiliary"].goal_complete.tolist() == [1] * len(states)
    assert batches["omit"].goal_complete.tolist() == [0] * len(states)
    encoder = encoding.FlatRelationEncoder(task, nullary="arity0", target_sources=("goal",))
    description = encoder.describe(batches["arity0"], 0)
    assert description.relations["[+]done[g]"] == (("[+](done)[g]",),)


@pytest.mark.parametrize("mode, policy", [("full", "include"), ("full", "exclude"), ("delta", "exclude"),
                                          ("action", "encode_only")])
def test_horizons_are_the_auxiliary_ones_without_the_auxiliary_entity(mode, policy):
    task = blocksworld()
    table = rl.as_table(task)
    roots = walk(task, steps=3, seed=7, walks=1)[:3]
    view = encoding.horizon(table, np.asarray(task.encode(roots)), 2)
    common = dict(horizon_mode=mode, root_policy=policy, parent_relation=True)
    aux_encoder = encoding.FlatHorizonEncoder(task, nullary="auxiliary", **common)
    encoder = encoding.FlatHorizonEncoder(task, nullary="arity0", **common)
    aux, batch = aux_encoder.encode(view), encoder.encode(view)
    expect_valid(batch)
    assert AUXILIARY_ROLE not in np.asarray(batch.entity_role).tolist()
    assert AUXILIARY_ROLE in np.asarray(aux.entity_role).tolist()
    for g in range(batch.num_graphs):
        assert without_auxiliary(aux_encoder.describe(aux, g)) == without_auxiliary(encoder.describe(batch, g))
    roles = dict(zip(encoder.relation_names, encoder.relation_slot_roles))
    assert roles["arm-empty"] == ([] if policy == "exclude" else [2])


# ------------------------------------------------------------------------------------------------ other views
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_the_simple_graph_has_an_atom_node_without_edges(layout):
    task = blocksworld()
    states = walk(task, steps=6, seed=11)
    encoder = encoding.FlatRelationEncoder(task, nullary="arity0", layout=layout, outputs=("relations", "simple"))
    batch = encoder.encode(states)
    expect_simple_of(batch)
    relation = encoder.relation_id("arm-empty")
    label = len(batch.schema.entity_role_names) + relation
    atoms = np.flatnonzero(np.asarray(batch.node_label) == label)
    assert len(atoms) == int(batch.relation_counts[:, relation].sum()) > 0
    edges = np.asarray(batch.edge_index)
    assert not np.isin(atoms, edges).any()
    only = encoding.SimpleEncoder(task, nullary="arity0").encode(states)
    assert np.array_equal(np.asarray(only.node_label), np.asarray(batch.node_label))
    assert np.array_equal(np.asarray(only.edge_index), edges)


@pytest.mark.parametrize("outputs", [("relations", "simple"), ("simple",)])
def test_simple_capacities_hold_more_tuples_than_arguments(outputs):
    # unary atoms and the nullary (ready): a batch has more tuples than arguments, so the nodes are bounded by one
    # tuple per graph and relation without argument slots
    features = ROOT / "tests/data/pddl/features"
    tasks = [pddl(features / "domain.pddl", features / f"{p}.pddl") for p in ("p01", "p02")]
    table = rl.TaskTable(tasks)
    parts = [np.asarray(task.encode(walk(task, steps=5, seed=3))) for task in tasks]
    rows = np.zeros((sum(len(p) for p in parts), table.words), np.uint64)
    ids = np.repeat(np.arange(len(tasks), dtype=np.int64), [len(p) for p in parts])
    for k, part in enumerate(parts):
        rows[ids == k, :part.shape[1]] = part
    encoder = encoding.FlatRelationEncoder(table, nullary="arity0", goal_derivations=ALL, outputs=outputs)
    argumentless = list(encoder.relation_arities).count(0)
    exact = encoder.encode(rows, ids)
    graphs, entities = exact.num_graphs, int(exact.entity_offsets[-1])
    nodes, args = int(exact.node_offsets[-1]), int(exact.edge_offsets[-1]) // 2
    assert argumentless > 0 and nodes - entities > args
    for extra in (0, 3):
        caps = {"graphs": graphs, "entities": entities + extra, "args": args + extra}
        padded = encoder.encode(rows, ids, capacities=caps)
        assert not padded.sizes["overflow"] and padded.sizes["entities"] + padded.sizes["tuples"] == nodes
        label = np.asarray(padded.node_label)
        assert label.shape == (entities + args + 2 * extra + graphs * argumentless,)
        assert np.array_equal(label[:nodes], np.asarray(exact.node_label)) and not label[nodes:].any()
        assert np.array_equal(np.asarray(padded.edge_offsets), np.asarray(exact.edge_offsets))
        again = encoder.encode(rows, ids, capacities=caps, into=padded)
        assert np.array_equal(np.asarray(again.node_label), label)


def test_the_hgraph_view_has_a_node_type_without_edge_types():
    task = blocksworld()
    states = walk(task, steps=6, seed=13)
    encoder = encoding.FlatRelationEncoder(task, nullary="arity0", layout="relation", goal_derivations=ALL)
    batch = encoder.encode(states)
    view = encoding.hgraph(batch)
    assert "arm-empty" in view.node_types
    assert not [t for t in view.edge_types if t[0] == "arm-empty" or t[2] == "arm-empty"]
    assert ("[+]arm-empty[g]", "0", "_symbol_") not in view.edge_types  # the goal relation has no slot either
    relation = encoder.relation_id("arm-empty")
    counts = np.asarray(batch.relation_counts)[:, relation]
    assert np.array_equal(np.asarray(view.ptr("arm-empty")), np.concatenate([[0], np.cumsum(counts)]))
    assert view.num_nodes("arm-empty") == int(counts.sum()) > 0
    arities = dict(zip(encoder.relation_names, encoder.relation_arities))
    assert view.edge_types == [(n, str(k), "_symbol_") for n, a in arities.items() for k in range(a)]


def test_a_nullary_numeric_function_has_a_tuple_without_arguments():
    drone = mymyr.Task.from_text(ROOT / "tests/data/numeric_tasks/cs-drone.txt")
    states = [drone.initial_state] + list(drone.successor_states(drone.initial_state))[:3]
    batches, schemas = {}, {}
    for value in ("auxiliary", "arity0", "omit"):
        encoder = encoding.FlatRelationEncoder(drone, nullary=value, numeric_dtype="float64")
        batches[value], schemas[value] = encoder.encode(states), encoder
    relation = schemas["arity0"].relation_id("[num]battery-level")
    roles = {v: dict(zip(e.relation_names, e.relation_slot_roles))["[num]battery-level"] for v, e in schemas.items()}
    # the function's tuple has no argument under arity0; omit leaves out predicates only
    assert roles == {"auxiliary": [AUXILIARY_SLOT], "arity0": [], "omit": [AUXILIARY_SLOT]}
    batch = batches["arity0"]
    expect_valid(batch)
    assert batch.relation_counts[:, relation].tolist() == [1] * len(states)
    assert AUXILIARY_ROLE not in np.asarray(batch.entity_role).tolist()
    assert np.array_equal(batch.numeric_values, batches["auxiliary"].numeric_values, equal_nan=True)


# ------------------------------------------------------------------------------------------------ batches
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_collation_and_streams_equal_one_encoding(layout):
    task = ferry()
    rows = np.asarray(task.encode(walk(task, steps=9, seed=17, walks=2)))
    encoder = encoding.FlatRelationEncoder(task, nullary="arity0", layout=layout, target_sources=("goal",),
                                           goal_derivations=ALL)
    expected = encoder.encode(rows)
    slices = [encoder.encode(rows[:4]), encoder.encode(rows[:0]), encoder.encode(rows[4:])]
    stream = encoder.stream()
    stream.append(rows[:5])
    stream.append(rows[5:])
    for actual in (encoding.collate(slices), stream.flush()):
        for key, value in expected.as_dict().items():
            assert np.array_equal(value, actual.as_dict()[key]), key


def test_the_validator_tells_the_choices_apart():
    task = blocksworld()
    states = walk(task, steps=4, seed=19)
    batches = {v: encoding.FlatRelationEncoder(task, nullary=v).encode(states) for v in ("auxiliary", "arity0")}
    for value, batch in batches.items():
        expect_valid(batch)
        other = batches["arity0" if value == "auxiliary" else "auxiliary"]
        with pytest.raises(ValueError):
            encoding.FlatBatch.from_arrays(other.schema, batch.as_dict())
