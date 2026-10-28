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


def test_rows_of_a_dropped_batch_need_the_index():
    """Without its index the rows of a no-op-dropping expansion are not found by position: an entity whose canonical
    index lies past its state's kept rows is refused."""
    from types import SimpleNamespace

    table, rows, full, kept = noop_rows(ipc_task("floortile"))
    batch = encoding.FlatRelationEncoder(table, **OPTIONS).encode(rows, actions=kept)
    assert kept.total < full.total
    with pytest.raises(ValueError, match="no row for action entity .* needs that expansion's index"):
        encoding.action_rows(batch, SimpleNamespace(offsets=kept.offsets))


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


# ------------------------------------------------------------------------------------------------ devices and observations
def torch_or_skip():
    return pytest.importorskip("torch")


def cuda_or_skip():
    from test_rl_table import gpu_device

    return gpu_device()


def noop_table(domain):
    """The table of a no-op domain's first instance with fixed atom slots, as environments need it."""
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/ipc" / f"{domain}-ipc" / "train"
    return rl.TaskTable([mymyr.Task.from_pddl(base / "domain.pddl", base / "p01.pddl", atoms="dense")])


def host_rows(states, task_ids):
    return np.ascontiguousarray(states.cpu().numpy()), np.ascontiguousarray(task_ids.cpu().numpy())


def check_observation(encoder, table, states, task_ids, batch, actions, drop):
    """An observation (batch and ActionRows) against the host encoding of its rows' host expansion."""
    from test_rl_torch_obs import assert_same

    rows, ids = host_rows(states, task_ids)
    want = rl.expand(table, rows, ids, hashes=True, noops=True, drop_noops=drop)
    assert_same(batch, encoder.encode(rows, ids, actions=want))
    total = int(want.total)
    assert not actions.status.any()
    np.testing.assert_array_equal(actions.offsets.cpu().numpy(), want.offsets)
    for name in ("parent", "schema", "binding", "noop"):
        np.testing.assert_array_equal(getattr(actions, name)[:total].cpu().numpy(), getattr(want, name), err_msg=name)
    np.testing.assert_array_equal(actions.succ_hash[:total].cpu().numpy().view(np.uint64), want.succ_hash)
    assert (actions.index is None) == (not drop)
    if drop:
        np.testing.assert_array_equal(actions.index[:total].cpu().numpy(), want.index)
    return want


def step_by_entities(env, batch, actions):
    """Steps with the first action entity of each graph: the reached state is the successor its row names."""
    torch = torch_or_skip()
    entities = encoding.action_entities(batch)
    graphs = np.searchsorted(np.asarray(batch.as_dict("numpy")["entity_offsets"]), entities, side="right") - 1
    chosen = entities[np.searchsorted(graphs, np.arange(env.num_envs))]
    rows = encoding.action_rows(batch, actions, chosen)
    reached = actions.succ_hash[torch.as_tensor(rows, device=env.device)]
    step = env.step(torch.as_tensor(encoding.action_indices(batch, chosen), device=env.device))
    assert not step.invalid.any()
    assert torch.equal(mymyr.hash_rows(step.final_states), reached)
    return step


@pytest.mark.parametrize("capacity", [False, True])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_batched_env_observations_drop_noop_actions(dev, capacity):
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    device = torch.device("cpu") if dev == "cpu" else cuda_or_skip()
    table = noop_table("floortile")
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    caps = dict(entities=1 << 12, args=1 << 14, targets=1 << 12) if capacity else None
    env = rt.BatchedEnv(table, 6, device=device, seed=5, max_steps=8, observation=encoder, obs_actions=True,
                        obs_drop_noops=True, obs_capacities=caps)
    observation = env.observation()
    dropped = 0
    for _ in range(4):
        want = check_observation(encoder, table, observation.states, observation.task_ids, observation.encoding,
                                 observation.actions, True)
        dropped += int(want.generated - want.total)
        step = step_by_entities(env, observation.encoding, observation.actions)
        observation = env.observation()
        check_observation(encoder, table, step.states, step.task_ids, step.obs, step.actions, True)
    assert dropped
    with pytest.raises(ValueError, match="obs_drop_noops needs obs_actions"):
        rt.BatchedEnv(table, 2, device=device, observation=encoder, obs_drop_noops=True)
    env.close()


@pytest.mark.parametrize("drop", [False, True])
def test_captured_observations_drop_noop_actions(drop):
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    device = cuda_or_skip()
    table = noop_table("rovers")
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    env = rt.BatchedEnv(table, 8, device=device, seed=2, max_steps=5, observation=encoder, obs_actions=True,
                        obs_drop_noops=drop, obs_capacities=dict(entities=1 << 12, args=1 << 14, targets=1 << 12))
    rollout = env.capture(3, final=True)
    for _ in range(2):
        out = rollout.replay()
        torch.cuda.synchronize(device)
        assert len(out.obs_actions) == 3
        for k in range(3):
            check_observation(encoder, table, out.states[k], out.task_ids[k], out.obs[k], out.obs_actions[k], drop)
    env.close()


@pytest.mark.needs_gil
@pytest.mark.parametrize("successors", [None, 1 << 16])
@pytest.mark.parametrize("drop", [False, True])
def test_traced_observations_drop_noop_actions(drop, successors):
    """A compiled step with action rows equals the host, and its graph has no node that keeps it out of CUDA graphs
    (inductor's check on the traced forward graph, which needs no triton)."""
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    device = cuda_or_skip()
    from torch._dynamo.backends.common import aot_autograd
    from torch._inductor.utils import get_first_incompatible_cudagraph_node

    assert torch.Tag.cudagraph_unsafe not in torch.ops.mymyr.bounded_actions.default.tags
    table = noop_table("floortile")
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    caps = dict(entities=1 << 12, args=1 << 14, targets=1 << 12, successors=successors)
    env = rt.BatchedEnv(table, 6, device=device, seed=7, max_steps=6, observation=encoder, obs_actions=True,
                        obs_drop_noops=drop, obs_capacities={k: v for k, v in caps.items() if v is not None})
    graphs = []

    def forward(gm, inputs):
        graphs.append(gm)
        return gm

    torch._dynamo.reset()
    try:
        step = torch.compile(lambda: env.step(final=False), fullgraph=True,
                             backend=aot_autograd(fw_compiler=forward))
        for _ in range(3):
            out = step()
            check_observation(encoder, table, out.states, out.task_ids, out.obs, out.actions, drop)
        assert graphs
        assert all(get_first_incompatible_cudagraph_node(g) is None for g in graphs)
    finally:
        torch.cuda.synchronize(device)
        env.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("domain", NOOP_DOMAINS)
def test_device_encodings_of_dropped_noops_equal_the_host(domain, layout):
    torch = torch_or_skip()
    device = cuda_or_skip()
    from test_rl_torch_obs import assert_same
    from mymyr.encoding._device_actions import bounded_actions

    table, rows, full, kept = noop_rows(ipc_task(domain))
    encoder = encoding.FlatRelationEncoder(table, layout=layout, **OPTIONS)
    host = encoder.encode(rows, actions=kept)
    states = torch.as_tensor(rows.view(np.int64), device=device)
    exact = rl.expand(table, states, drop_noops=True, framework="torch")
    batch = encoder.encode(states, actions=exact)
    assert_same(batch, host)
    caps = dict(graphs=len(rows), entities=1 << 12, args=1 << 14, targets=1 << 12)
    bounded = bounded_actions(encoder, states, None, caps["entities"], hashes=True, drop_noops=True)
    assert_same(encoder.encode(states, actions=bounded, capacities=caps), host)
    from mymyr.rl.torch import _ops

    handle = _ops.register(table)
    try:
        flat = torch.ops.mymyr.expand_flat(handle, states, None, int(full.total), rows=False, hashes=True,
                                           drop_noops=True)
        assert_same(encoder.encode(states, actions=flat, capacities=caps), host)
    finally:
        _ops.unregister(handle)
    # the inverse on the device, against the host's
    counts = np.diff(full.offsets)
    every = np.arange(counts.max() + 1)[None, :].repeat(len(rows), 0)
    want = encoding.action_entities_of(host, every)
    got = encoding.action_entities_of(batch, torch.as_tensor(every, device=device))
    np.testing.assert_array_equal(got.cpu().numpy(), want)
    graphs = torch.arange(len(rows), device=device).flip(0)
    np.testing.assert_array_equal(encoding.action_entities_of(batch, torch.as_tensor(every[::-1].copy(), device=device),
                                                              graphs).cpu().numpy(), want[::-1])
    # rows out of canonical order are refused as on the host
    from types import SimpleNamespace

    index = exact.index.clone()
    first = int(np.flatnonzero(np.diff(kept.offsets) > 1)[0])
    b = int(kept.offsets[first])
    index[b], index[b + 1] = exact.index[b + 1], exact.index[b]
    swapped = SimpleNamespace(succ=None, schema=exact.schema, binding=exact.binding, parent=exact.parent,
                              offsets=exact.offsets, index=index)
    with pytest.raises(ValueError, match=f"^mymyr: encoding: the action rows of state {first} are not in canonical"):
        encoder.encode(states, actions=swapped)
    # a row that is no label of its instance is the cause first, as on the host
    schema = exact.schema.clone()
    schema[b] = 1000
    swapped.schema = schema
    with pytest.raises(ValueError, match="^mymyr: encoding: a row is invalid"):
        encoder.encode(states, actions=swapped)
    torch.cuda.synchronize(device)


def test_device_collate_and_suites_drop_noop_actions():
    torch = torch_or_skip()
    device = cuda_or_skip()
    from mymyr import datasets
    from test_rl_torch_obs import assert_same

    table, rows, full, kept = noop_rows(ipc_task("satellite"))
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    states = torch.as_tensor(rows.view(np.int64), device=device)
    ids = torch.zeros(len(rows), dtype=torch.int32, device=device)
    out = datasets.EncodingCollate(encoder, actions=True, drop_noops=True)(datasets.StateRows(states, ids))
    assert_same(out.graphs, encoder.encode(rows, actions=kept))
    suite = rl.TaskSuite([table, noop_table("rovers")])
    suite_encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    words = torch.zeros((len(rows), suite.words), dtype=torch.int64, device=device)
    words[:, :rows.shape[1]] = states
    caps = {d: dict(graphs=len(rows), entities=1 << 12, args=1 << 14, targets=1 << 12) for d in range(2)}
    for capacities in (None, caps):
        batch = suite_encoder.encode(words, ids, actions=True, drop_noops=True, capacities=capacities)
        assert_same(batch.batches[0], encoder.encode(rows, actions=kept))
    torch.cuda.synchronize(device)


def test_device_suite_observations_find_their_action_entities():
    """action_entities_of on the domain batches of a device suite observation (graph capacity N, zero offsets past
    the domain's live graphs) equals the host's on the same batches."""
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    device = cuda_or_skip()
    n = 64
    suite = rl.TaskSuite([noop_table("rovers"), noop_table("floortile")])
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    env = rt.BatchedEnv(suite, n, device=device, seed=4, observation=encoder, obs_actions=True, obs_drop_noops=True,
                        obs_capacities=dict(entities=1 << 14, args=1 << 16, targets=1 << 14))
    every = np.arange(64)[None, :].repeat(n, 0)
    found = 0
    for _ in range(3):
        observation = env.observation()
        torch.cuda.synchronize(device)
        for batch in observation.encoding.batches:
            graphs = batch.num_graphs
            assert 0 < graphs < n
            want = encoding.action_entities_of(encoding.FlatBatch.from_arrays(batch.schema, batch.as_dict("numpy")),
                                               every[:graphs])
            got = encoding.action_entities_of(batch, torch.as_tensor(every, device=device)).cpu().numpy()
            np.testing.assert_array_equal(got[:graphs], want)
            assert (got[graphs:] == -1).all()
            found += int((want >= 0).sum())
        env.step(torch.zeros(n, dtype=torch.int64, device=device))
    assert found
    env.close()


IDLES = """(define (domain idles) (:requirements :strips :typing :negative-preconditions)
 (:types obj)
 (:predicates (at ?x - obj) (done ?x - obj))
 (:action mark :parameters (?x - obj) :precondition (not (done ?x)) :effect (done ?x))
 (:action idle :parameters (?x ?y ?z - obj) :precondition (at ?x) :effect (at ?x)))"""
IDLES_PROBLEM = """(define (problem idles) (:domain idles) (:objects o1 o2 o3 o4 o5 o6 - obj)
 (:init (at o1) (at o2) (at o3) (at o4) (at o5) (at o6)) (:goal (and (done o1) (done o2))))"""


def idles_table():
    """A table whose initial state has 216 no-op successors (idle) and 6 effective ones (mark)."""
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return rl.TaskTable([mymyr.Task(mymyr.Domain.from_string(IDLES).instantiate_string(IDLES_PROBLEM))])


def observe_idles(where, caps, drop):
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    table = idles_table()
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
    env = rt.BatchedEnv(table, 4, device=where, seed=1, observation=encoder, obs_actions=True, obs_drop_noops=drop,
                        obs_capacities=caps)
    observation = env.observation()
    if where.type == "cuda":
        torch.cuda.synchronize(where)
    env.close()
    return observation


IDLES_CAPS = dict(entities=80, args=1024, targets=256)  # 80 action rows: the 24 kept ones fit, the 888 others not


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_capacities_bound_the_kept_rows(dev, drop):
    """The entity capacity bounds the kept action rows, ``successors`` the successors before the drop: environments
    and suite encoders report the overflow (flags, status, zero sizes) on the CPU as on a device."""
    torch = torch_or_skip()
    device = torch.device("cpu") if dev == "cpu" else cuda_or_skip()
    table = idles_table()
    task = table.tasks[0]
    rows = np.repeat(np.asarray(task.encode([task.initial_state])), 4, axis=0)
    full = rl.expand(table, rows, noops=True)
    generated, kept = int(full.total), int((~full.noop).sum())
    assert (generated, kept) == (888, 24)
    cases = [(IDLES_CAPS, True), (dict(IDLES_CAPS, successors=generated), not drop),
             (dict(IDLES_CAPS, successors=generated - 1), True)]
    suite_encoder = encoding.SuiteEncoder(rl.TaskSuite([table]), target_sources=("action",))
    ids = np.zeros(4, np.int32)
    for caps, overflow in cases:
        observation = observe_idles(device, caps, drop)
        sizes = observation.encoding.sizes
        assert (sizes["overflow"], sizes["capacity_overflow"]) == (overflow, overflow), (caps, sizes)
        assert observation.actions.status.tolist() == [int(overflow)]
        assert observation.actions.parent.shape == (80,) and observation.actions.offsets.shape == (5,)
        roles = np.asarray(observation.encoding.as_dict("numpy")["entity_role"])
        if overflow:
            assert sizes["entities"] == sizes["args"] == 0 and not roles.any()
        else:
            assert int((roles == ACTION_ROLE).sum()) == kept and sizes["entities"] > kept
        suite_caps = {0: dict(caps, graphs=4)}
        states, task_ids = (rows, ids) if dev == "cpu" else (torch.as_tensor(rows.view(np.int64), device=device),
                                                              torch.as_tensor(ids, device=device))
        batch = suite_encoder.encode(states, task_ids, actions=True, drop_noops=drop, capacities=suite_caps).batches[0]
        assert batch.sizes["capacity_overflow"] == overflow
    with pytest.raises(ValueError, match="successors capacity must be an int in"):
        observe_idles(device, dict(IDLES_CAPS, successors=1 << 31), drop)


@pytest.mark.parametrize("drop", [False, True])
def test_cpu_and_device_observations_of_bounded_rows_agree(drop):
    torch = torch_or_skip()
    device = cuda_or_skip()
    for caps in (IDLES_CAPS, dict(IDLES_CAPS, successors=888), dict(IDLES_CAPS, successors=887)):
        host, dev = observe_idles(torch.device("cpu"), caps, drop), observe_idles(device, caps, drop)
        assert host.encoding.sizes == dev.encoding.sizes
        assert torch.equal(host.actions.status, dev.actions.status.cpu())
        if not host.encoding.sizes["overflow"]:
            for name, value in host.actions._asdict().items():
                assert torch.equal(value, getattr(dev.actions, name).cpu()), name


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("kind", ["table", "suite"])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_action_rows_have_the_capacity_rows_zero_past_the_kept_ones(dev, kind, drop):
    torch = torch_or_skip()
    rt = pytest.importorskip("mymyr.rl.torch")
    device = torch.device("cpu") if dev == "cpu" else cuda_or_skip()
    from mymyr.encoding._device_actions import action_capacity

    tables = [noop_table("floortile")] + ([noop_table("rovers")] if kind == "suite" else [])
    source = tables[0] if kind == "table" else rl.TaskSuite(tables)
    encoders = [encoding.FlatRelationEncoder(t, **OPTIONS) for t in tables]
    env = rt.BatchedEnv(source, 6, device=device, seed=3, observation=encoders if kind == "suite" else encoders[0],
                        obs_actions=True, obs_drop_noops=drop,
                        obs_capacities=dict(entities=1 << 12, args=1 << 14, targets=1 << 12))
    observation = env.observation()
    states, task_ids = host_rows(observation.states, observation.task_ids)
    if kind == "table":
        parts, domain = [observation.actions], np.zeros(6, np.int64)
    else:
        parts, domain = observation.actions, observation.encoding.domain.cpu().numpy()
        task_ids = np.asarray(source.local_ids, np.int32)[task_ids]
    for d, actions in enumerate(parts):
        mine = domain == d
        want = rl.expand(tables[d], states[mine][:, :tables[d].words], task_ids[mine], hashes=True, noops=True,
                         drop_noops=drop)
        total, rows = int(want.total), action_capacity(encoders[d], 1 << 12)
        assert actions.offsets.shape == (7,)
        np.testing.assert_array_equal(actions.offsets.cpu().numpy(),
                                      np.pad(want.offsets, (0, 6 - int(mine.sum())), mode="edge"))
        assert (actions.index is not None) == drop
        for name in ("parent", "schema", "binding", "succ_hash", "noop", "index"):
            value = getattr(actions, name)
            if value is None:
                continue
            assert value.shape[0] == rows and value.device == device, name
            assert not value[total:].any(), name
            got = value[:total].cpu().numpy()
            expected = np.asarray(getattr(want, name))
            np.testing.assert_array_equal(got.view(np.uint64) if name == "succ_hash" else got, expected, err_msg=name)
    env.close()


@pytest.mark.parametrize("drop", [False, True])
def test_bounded_actions_outputs_match_their_fakes(drop):
    """torch.library.opcheck of the expansion and encoding ops an observation traces, below, at and above the rows
    of the arrays: the fake outputs have the real ones' shapes, strides and dtypes, and nothing aliases."""
    torch = torch_or_skip()
    device = cuda_or_skip()
    from mymyr.rl.torch import _ops

    table = noop_table("floortile")
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    task = table.tasks[0]
    states = torch.as_tensor(np.repeat(np.asarray(task.encode([task.initial_state])), 4, axis=0).view(np.int64),
                             device=device)
    handle, table_handle = _ops.register(encoder), _ops.register(table)
    try:
        for successors in (None, 8, 16, 64):  # 16 rows: the action rows of 16 entities
            args = (handle, states, None, 16, successors, drop)
            torch.library.opcheck(torch.ops.mymyr.bounded_actions.default, args)
            labels, binding, offsets, succ_hash, noop, status = torch.ops.mymyr.bounded_actions(*args)
            assert labels.shape == (3 if drop else 2, 16) and labels.is_contiguous()
            given = dict(action_schema=labels[1], action_binding=binding, action_offsets=offsets,
                         action_parent=labels[0], action_index=labels[2] if drop else None, action_status=status)
            torch.library.opcheck(torch.ops.mymyr.encode_flat.default,
                                  (handle, states, *[None] * 16, 4, 256, 2048, 256, 0), given)
        torch.library.opcheck(torch.ops.mymyr.expand_flat.default, (table_handle, states, None, 64),
                              dict(rows=False, hashes=True, noops=True, drop_noops=drop))
        torch.cuda.synchronize(device)
    finally:
        _ops.unregister(handle)
        _ops.unregister(table_handle)
