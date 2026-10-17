"""The action lane and the predicate virtual nodes of the relational encoder through the Python API: action tuples and
action entities from an expansion, action targets and their link back to the expansion rows, the rules of the
expansion argument, and the virtual node of each used predicate."""

import random

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task, walk

GOAL_ROLE = 2
ACTION_ROLE = 4
VIRTUAL_ROLE = 1
ACTION_GROUP = 2
TASKS = ["gripper__prob05", "blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1", "philosophers__p03-phil4"]
OPTIONS = {"target_sources": ("action", "goal"), "predicate_virtual_nodes": True}


def expand(task, states):
    return rl.expand(rl.as_table(task), states)


def graph_cells(batch, graph):
    """{relation name: tuples of the graph's entity ids (local to the graph)}."""
    names = batch.schema.relation_names
    arities = batch.schema.relation_arities
    relations = batch.num_relations
    graphs = batch.num_graphs
    relation_major = batch.schema.options["layout"] == "relation"
    base = int(batch.entity_offsets[graph])
    out = {}
    for relation, name in enumerate(names):
        count = int(batch.relation_counts[graph, relation])
        if not count:
            continue
        cell = relation * graphs + graph if relation_major else graph * relations + relation
        begin, end = (int(x) for x in batch.relation_arg_offsets[cell : cell + 2])
        arguments = (batch.relation_args[begin:end] - base).reshape(count, arities[relation])
        out[name] = [tuple(int(x) for x in row) for row in arguments]
    return out


def action_entities(batch, graph):
    """(local id, row of the graph's slice of the expansion) of each action entity of a graph."""
    begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
    return [(i, int(batch.entity_ref[begin + i])) for i in range(end - begin) if batch.entity_role[begin + i] == ACTION_ROLE]


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", TASKS)
def test_action_tuples_and_entities_follow_the_expansion(name, layout):
    task = text_task(name)
    states = walk(task, steps=6, walks=2)
    expansion = expand(task, states)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, **OPTIONS)
    batch = encoder.encode(states, actions=expansion)
    offsets = expansion.offsets
    lanes = encoder.relation_lanes
    for graph in range(len(states)):
        rows = list(range(int(offsets[graph + 1]) - int(offsets[graph])))
        entities = action_entities(batch, graph)
        assert sorted(row for _, row in entities) == rows
        ids = [i for i, _ in entities]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        cells = graph_cells(batch, graph)
        names = {n for n, lane in zip(encoder.relation_names, lanes) if lane == "action"}
        expected = {}
        for (entity, row) in entities:
            action = expansion.action(int(offsets[graph]) + row)
            expected.setdefault(action.name, []).append((entity, *action.label[1]))
        got = {n: t for n, t in cells.items() if n in names}
        assert {n: sorted(t) for n, t in got.items()} == {n: sorted(t) for n, t in expected.items()}
        begin, end = (int(x) for x in batch.target_offsets[graph : graph + 2])
        base = int(batch.entity_offsets[graph])
        targets = [(int(batch.target_entity[t]) - base, int(batch.target_group[t])) for t in range(begin, end)]
        assert sorted(e for e, g in targets if g == ACTION_GROUP) == ids


def test_without_an_expansion_there_are_no_action_tuples_or_entities():
    task = text_task("gripper__prob05")
    states = walk(task, steps=4, walks=1)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    batch = encoder.encode(states)
    assert not (np.asarray(batch.entity_role) == ACTION_ROLE).any()
    assert not any(n in graph_cells(batch, 0) for n, lane in zip(encoder.relation_names, encoder.relation_lanes)
                   if lane == "action")


def test_a_single_state_equals_its_slice_of_a_batch():
    task = text_task("logistics00__probLOGISTICS-6-1")
    states = walk(task, steps=8, walks=2)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    batch = encoder.encode(states, actions=expand(task, states))
    for graph, state in enumerate(states):
        single = encoder.encode([state], actions=expand(task, [state]))
        assert graph_cells(single, 0) == graph_cells(batch, graph)
        begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
        assert np.array_equal(single.entity_role, batch.entity_role[begin:end])
        assert np.array_equal(single.entity_ref, batch.entity_ref[begin:end])


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_threads_and_layouts_give_the_same_action_graphs(threads, layout):
    task = text_task("blocks__probBLOCKS-8-0")
    states = walk(task, steps=10, walks=3)
    expansion = expand(task, states)
    reference = encoding.FlatRelationEncoder(task, **OPTIONS).encode(states, actions=expansion)
    batch = encoding.FlatRelationEncoder(task, layout=layout, **OPTIONS).encode(states, actions=expansion, threads=threads)
    for graph in range(len(states)):
        assert graph_cells(batch, graph) == graph_cells(reference, graph)
    assert np.array_equal(batch.entity_ref, reference.entity_ref)
    assert np.array_equal(batch.entity_role, reference.entity_role)


@pytest.mark.parametrize("name", ["gripper__prob05", "blocks__probBLOCKS-8-0", "logistics00__probLOGISTICS-6-1"])
def test_picking_action_entities_walks_the_task(name):
    """The reference of an action entity is its expansion row relative to its state's rows (`offsets[graph] + ref` is
    the row of the expansion): following the entities a policy picks reproduces Task.apply."""
    task = text_task(name)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    rng = random.Random(7)
    states = walk(task, steps=6, walks=3)
    expansion = expand(task, states)
    batch = encoder.encode(states, actions=expansion)
    checked = 0
    for graph, state in enumerate(states):
        count = int(expansion.offsets[graph + 1]) - int(expansion.offsets[graph])
        begin, end = (int(x) for x in batch.target_offsets[graph : graph + 2])
        picks = [int(batch.target_entity[t]) for t in range(begin, end) if batch.target_group[t] == ACTION_GROUP]
        assert len(picks) == count
        if not picks:
            continue
        entity = rng.choice(picks)
        row = int(expansion.offsets[graph]) + int(batch.entity_ref[entity])
        assert batch.entity_role[entity] == ACTION_ROLE
        action = expansion.action(row)
        successor = task.decode(expansion.succ[row : row + 1])[0]
        assert task.apply(state, action) == successor
        checked += 1
    assert checked > 1


# ------------------------------------------------------------------------------------------------ errors
def test_the_expansion_argument_is_validated():
    task = text_task("gripper__prob05")
    state = task.initial_state
    other = text_task("blocks__probBLOCKS-8-0")
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    with pytest.raises(TypeError, match="Expansion"):
        encoder.encode([state], actions=object())
    with pytest.raises(ValueError, match="^mymyr: encoding: actions expand 1 states, the batch has 2 rows"):
        encoder.encode([state, state], actions=expand(task, [state]))
    with pytest.raises(ValueError, match="mymyr: encoding"):
        encoder.encode([state], actions=expand(other, [other.initial_state]))


def test_actions_need_action_relations():
    task = text_task("gripper__prob05")
    state = task.initial_state
    encoder = encoding.FlatRelationEncoder(task, include_actions=False)
    with pytest.raises(ValueError, match="^mymyr: encoding: .*action"):
        encoder.encode([state], actions=expand(task, [state]))


def test_an_overflowed_expansion_is_refused():
    task = text_task("gripper__prob05")
    state = task.initial_state
    short = rl.expand(rl.as_table(task), [state], capacity=2)
    assert short.overflow
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    with pytest.raises(ValueError, match="^mymyr: encoding: .*overflow"):
        encoder.encode([state], actions=short)


# ------------------------------------------------------------------------------------------------ virtual nodes
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", TASKS)
def test_each_used_predicate_has_one_virtual_node_in_name_order(name, layout):
    task = text_task(name)
    states = walk(task, steps=8, walks=2)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, **OPTIONS)
    batch = encoder.encode(states, actions=expand(task, states))
    schema = encoder.schema
    for graph in range(len(states)):
        begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
        roles = np.asarray(batch.entity_role[begin:end])
        refs = np.asarray(batch.entity_ref[begin:end])
        virtual = [int(r) for r, role in zip(refs, roles) if role == VIRTUAL_ROLE]
        names = [schema.relation_names[ref] for ref in virtual]
        assert names == sorted(set(names))
        first = int(np.argmax(roles == VIRTUAL_ROLE)) if virtual else None
        cells = graph_cells(batch, graph)
        used = set()
        for relation, tuples in cells.items():
            slots = schema.relation_slot_roles[schema.relation_id(relation)]
            predicate_slot = schema.slot_role_names.index("predicate_slot")
            for column, role in enumerate(slots):
                if role == predicate_slot:
                    used |= {t[column] for t in tuples}
        if virtual:
            assert sorted(used) == [first + i for i in range(len(virtual))]
        else:
            assert not used


def test_virtual_nodes_are_options_of_the_schema():
    task = text_task("gripper__prob05")
    off = encoding.EncodingSchema(task, target_sources=("action", "goal"))
    on = encoding.EncodingSchema(task, **OPTIONS)
    assert "predicate_virtual" in on.entity_role_names
    assert off.schema_fingerprint != on.schema_fingerprint
