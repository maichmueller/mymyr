"""The action lane and the predicate virtual nodes of the relational encoder through the Python API: action tuples and
action entities from an expansion, action targets and their link back to the expansion rows, the rules of the
expansion argument, and the virtual node of each used predicate."""

import random

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import BLOCKS, ROOT, text_task, walk

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
    states = states * (2500 // len(states) + 1)  # more graphs than one thread encodes alone (1024)
    assert len(states) > 1024
    expansion = expand(task, states)
    reference = encoding.FlatRelationEncoder(task, **OPTIONS).encode(states, actions=expansion)
    batch = encoding.FlatRelationEncoder(task, layout=layout, **OPTIONS).encode(states, actions=expansion, threads=threads)
    for graph in range(0, len(states), 7):
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


def test_the_expansion_must_be_over_the_encoders_own_task():
    """An expansion over another problem of the same domain has the same schemas but other objects: it is refused,
    while one over an equal task loaded again is accepted."""
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the loki front end")
    task = mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl")
    smaller = mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "smedium.pddl")
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    state = task.initial_state
    with pytest.raises(ValueError, match="^mymyr: encoding: .*another task than instance 0"):
        encoder.encode([state], actions=expand(smaller, [smaller.initial_state]))
    again = mymyr.Task.from_pddl(BLOCKS / "domain.pddl", BLOCKS / "probBLOCKS-8-0.pddl")
    batch = encoder.encode([state], actions=expand(again, [again.initial_state]))
    reference = encoder.encode([state], actions=expand(task, [state]))
    assert np.array_equal(batch.relation_args, reference.relation_args)


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


# ------------------------------------------------------------------------------------------------ dropped no-ops
NOOP_DOMAINS = ["floortile", "rovers", "satellite"]  # domains whose walks reach actions that change nothing
SWITCHES = """(define (domain switches)
 (:requirements :strips :negative-preconditions)
 (:predicates (live) (p))
 (:action idle-a :parameters (?x) :precondition (live) :effect (and))
 (:action idle-b :parameters (?x) :precondition (live) :effect (and))
 (:action stop :parameters (?x) :precondition (and (live) (not (p))) :effect (not (live)))
 (:action set :parameters (?x) :precondition (and (live) (not (p))) :effect (p)))"""
SWITCHES_PROBLEM = "(define (problem switches) (:domain switches) (:objects a) (:init (live)) (:goal (p)))"


def ipc_task(domain):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/ipc" / f"{domain}-ipc" / "train"
    return mymyr.Task.from_pddl(base / "domain.pddl", base / "p01.pddl")


def switches_task():
    """A task whose state after `set` has only no-op successors (the fallback keeps them all)."""
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return mymyr.Task(mymyr.Domain.from_string(SWITCHES).instantiate_string(SWITCHES_PROBLEM))


def noop_rows(task):
    """A walk's rows and the full and the no-op-dropping expansion of them."""
    table = rl.as_table(task)
    rows = np.asarray(task.encode(walk(task, steps=12, walks=2, seed=9)))
    return (table, rows, rl.expand(table, rows, noops=True, hashes=True),
            rl.expand(table, rows, noops=True, hashes=True, drop_noops=True))


def entity_graphs(batch, entities):
    return np.searchsorted(np.asarray(batch.entity_offsets), entities, side="right") - 1


def check_dropped(batch, full_batch, full, kept):
    """The action entities of `batch` (encoded from `kept`) are those of `full_batch` (encoded from `full`) without the
    no-op actions of states that have another successor, with the same canonical indices and labels."""
    entities = encoding.action_entities(batch)
    graphs = entity_graphs(batch, entities)
    indices = encoding.action_indices(batch)
    rows = encoding.action_rows(batch, kept)
    np.testing.assert_array_equal(kept.index[rows], indices)
    np.testing.assert_array_equal(kept.parent[rows], graphs)
    fallback = np.array([e > b and full.noop[b:e].all() for b, e in zip(full.offsets[:-1], full.offsets[1:])])
    assert not kept.noop[rows][~fallback[graphs]].any()
    full_rows = full.offsets[graphs] + indices
    np.testing.assert_array_equal(full.schema[full_rows], kept.schema[rows])
    np.testing.assert_array_equal(full.binding[full_rows], kept.binding[rows])
    if kept.succ_hash is not None:
        np.testing.assert_array_equal(full.succ_hash[full_rows], kept.succ_hash[rows])
    # the full batch's entities of the same (graph, index) are the kept actions plus the dropped no-ops
    full_entities = encoding.action_entities(full_batch)
    full_keys = set(zip(entity_graphs(full_batch, full_entities).tolist(), encoding.action_indices(full_batch).tolist()))
    kept_keys = set(zip(graphs.tolist(), indices.tolist()))
    dropped = {(int(full.parent[r]), int(r - full.offsets[full.parent[r]])) for r in np.flatnonzero(full.noop)
               if not fallback[full.parent[r]]}
    assert kept_keys == full_keys - dropped
    return entities, graphs, indices


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("domain", NOOP_DOMAINS)
def test_dropped_noops_keep_the_canonical_index_of_every_action_entity(domain, layout):
    task = ipc_task(domain)
    table, rows, full, kept = noop_rows(task)
    assert full.noop.any() and kept.total < full.total
    encoder = encoding.FlatRelationEncoder(table, layout=layout, **OPTIONS)
    full_batch = encoder.encode(rows, actions=full)
    batch = encoder.encode(rows, actions=kept)
    entities, graphs, indices = check_dropped(batch, full_batch, full, kept)
    # action_entities_of inverts action_indices, and has no entity for a dropped or absent index
    np.testing.assert_array_equal(encoding.action_entities_of(batch, indices[:, None], graphs)[:, 0], entities)
    np.testing.assert_array_equal(encoding.action_entities_of(batch, indices, graphs), entities)
    counts = np.diff(full.offsets)
    every = np.arange(counts.max() + 1)[None, :].repeat(len(rows), 0)
    table_of = encoding.action_entities_of(batch, every)
    table_full = encoding.action_entities_of(full_batch, every)
    for g in range(len(rows)):
        b = int(full.offsets[g])
        for j in range(counts.max() + 1):
            present = j < counts[g] and (not full.noop[b + j] or full.noop[b:b + counts[g]].all())
            assert (table_of[g, j] >= 0) == present
            assert (table_full[g, j] >= 0) == (j < counts[g])
            if present:
                assert encoding.action_indices(batch, [table_of[g, j]])[0] == j
    assert (encoding.action_entities_of(batch, [-1, 1 << 40]) == -1).all()
    # pools, slices of the batch and streams give the same entities
    pooled = encoder.encode(rows, actions=kept, threads=3)
    np.testing.assert_array_equal(pooled.entity_ref, batch.entity_ref)
    stream = encoder.stream()
    for part in (slice(0, 5), slice(5, len(rows))):
        stream.append(rows[part], actions=rl.expand(table, rows[part], drop_noops=True))
    np.testing.assert_array_equal(stream.flush().entity_ref, batch.entity_ref)


def test_states_whose_successors_are_all_noops_keep_them():
    task = switches_task()
    table = rl.as_table(task)
    initial = np.asarray(task.encode([task.initial_state]))
    rows = np.concatenate([initial, rl.expand(table, initial).succ])  # the initial state and its four successors
    full, kept = rl.expand(table, rows, noops=True, hashes=True), rl.expand(table, rows, hashes=True, drop_noops=True)
    counts = np.diff(full.offsets)
    every_noop = [g for g in range(len(rows)) if counts[g] and full.noop[full.offsets[g]:full.offsets[g + 1]].all()]
    assert counts[0] == 4 and full.noop[:4].sum() == 2 and every_noop
    np.testing.assert_array_equal(np.diff(kept.offsets)[every_noop], counts[every_noop])
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    batch = encoder.encode(rows, actions=kept)
    check_dropped(batch, encoder.encode(rows, actions=full), full, kept)
    found = encoding.action_entities_of(batch, np.arange(4)[None, :].repeat(len(rows), 0)) >= 0
    np.testing.assert_array_equal(found[0], ~full.noop[:4])
    for g in every_noop:
        np.testing.assert_array_equal(found[g], np.arange(4) < counts[g])


def test_canonical_indices_must_ascend():
    task = ipc_task("floortile")
    table, rows, _, kept = noop_rows(task)
    index = np.asarray(kept.index)
    if not index.flags.writeable:
        pytest.skip("the expansion's arrays are read-only")
    first = int(np.flatnonzero(np.diff(kept.offsets) > 1)[0])
    b = int(kept.offsets[first])
    index[b], index[b + 1] = index[b + 1], index[b]
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    with pytest.raises(ValueError, match="^mymyr: encoding: .*canonical"):
        encoder.encode(rows, actions=kept)


def test_collate_drops_noop_actions():
    from mymyr import datasets

    task = ipc_task("rovers")
    table, rows, full, kept = noop_rows(task)
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    items = datasets.StateRows(rows, np.zeros(len(rows), np.int32))
    out = datasets.EncodingCollate(encoder, actions=True, drop_noops=True)(items)
    np.testing.assert_array_equal(out.actions.index, kept.index)
    check_dropped(out.graphs, encoder.encode(rows, actions=full), full, out.actions)
    with pytest.raises(ValueError, match="drop_noops needs actions=True"):
        datasets.EncodingCollate(encoder, drop_noops=True)


def test_suite_encoder_drops_noop_actions():
    rovers, satellite = ipc_task("rovers"), ipc_task("satellite")
    suite = rl.TaskSuite([rl.as_table(rovers), rl.as_table(satellite)])
    states = [*walk(rovers, steps=6, walks=1, seed=3), *walk(satellite, steps=6, walks=1, seed=3)]
    ids = np.array([0] * 7 + [1] * 7, np.int32)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    out = encoder.encode(states, ids, actions=True, drop_noops=True)
    full = encoder.encode(states, ids, actions=True)
    for d in range(2):
        kept = out.expansions[d]
        assert kept.index is not None
        check_dropped(out.batches[d], full.batches[d], rl.expand(suite.tables[d], states[7 * d:7 * d + 7], noops=True, hashes=True),
                      kept)
    with pytest.raises(ValueError, match="drop_noops needs actions=True"):
        encoder.encode(states, ids, drop_noops=True)
