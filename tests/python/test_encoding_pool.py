"""The encoder on the CPU environment pool's batches: the arrays of a pool batch (states, task ids, goal masks) and the
expansion of its states go to `encode` as they are, the chosen action entities map back to expansion rows and to the
actions an environment takes, and a batch can be re-encoded into the memory of an earlier one (`into=`)."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import text_task, walk
from test_rl_table import sets  # noqa: F401  (a fixture: the table instances of several domains)

OPTIONS = dict(target_sources=("action",), include_actions=True)
SCALARS = {"version", "num_graphs", "num_relations", "layout", "index_dtype", "schema_fingerprint", "statics"}


def arrays_equal(a, b):
    first, second = a.as_dict(), b.as_dict()
    assert first.keys() == second.keys()
    for key in first:
        assert np.array_equal(first[key], second[key]), key


def pool_of(tasks, num_envs=6, seed=3, **options):
    return rl.CpuEnvPool(tasks, num_envs=num_envs, seed=seed, max_steps=12, goals=True, **options)


def encode_pool_batch(encoder, batch, table):
    """The pool path: the pool's arrays and the expansion of its states, handed over as they are."""
    expansion = rl.expand(table, batch.states, batch.task_ids)
    return expansion, encoder.encode(
        batch.states, batch.task_ids, goals=(batch.goal_pos, batch.goal_neg), actions=expansion
    )


def encode_copies(encoder, batch, table):
    """The same call on copies of everything: rows, task ids, goal masks, and an expansion made from the copies."""
    states, task_ids = np.array(batch.states), np.array(batch.task_ids)
    goals = (np.array(batch.goal_pos), np.array(batch.goal_neg))
    expansion = rl.expand(table, states, task_ids)
    return expansion, encoder.encode(states, task_ids, goals=goals, actions=expansion)


def check_rounds(encoder, tasks, table, rounds=8, seed=0):
    """Steps a pool with the chosen action entities of its encoded states. Per round: the batch of the pool path equals
    the batch of the copies, every action entity maps to its expansion row, and the successor an environment reaches
    with the entity's index is the successor of that row."""
    pool = pool_of(tasks)
    rng = np.random.default_rng(seed)
    batch = pool.reset()
    chosen_total = 0
    for _ in range(rounds):
        expansion, encoded = encode_pool_batch(encoder, batch, table)
        _, copied = encode_copies(encoder, batch, table)
        arrays_equal(encoded, copied)

        # every action entity is the first row of its action among its state's rows
        entities = encoding.action_entities(encoded)
        rows = encoding.action_rows(encoded, expansion)
        assert len(entities) == len(rows) and (np.diff(entities) > 0).all()
        graphs = np.searchsorted(np.asarray(encoded.entity_offsets), entities, side="right") - 1
        for g in range(0, encoded.num_graphs, 2):
            names = encoder.entity_names(encoded, g)
            begin = int(encoded.entity_offsets[g])
            for entity, row in zip(entities[graphs == g], rows[graphs == g]):
                assert expansion.offsets[g] <= row < expansion.offsets[g + 1]
                assert names[entity - begin] == str(expansion.action(int(row)))

        # a policy picks one action entity per graph among the action targets
        targets = np.asarray(encoded.target_entity)[np.asarray(encoded.target_group) == 2]
        picked = []
        for g in range(encoded.num_graphs):
            own = targets[np.searchsorted(targets, np.asarray(encoded.entity_offsets)[g]):][
                : int((graphs == g).sum())
            ]
            if len(own):
                picked.append(int(rng.choice(own)))
        assert len(picked) == encoded.num_graphs  # gripper states always have an action
        picked = np.array(picked)
        indices = encoding.action_indices(encoded, picked)
        picked_rows = encoding.action_rows(encoded, expansion, picked)
        assert np.array_equal(picked_rows, np.asarray(expansion.offsets)[:-1] + indices)

        stepped = pool.step(batch.env_ids, indices)
        assert not np.asarray(stepped.invalid).any()
        reached, successors = np.asarray(stepped.final_states), np.asarray(expansion.succ)[picked_rows]
        width = successors.shape[1]  # the pool's rows may be wider than the batch's: the words past it are padding
        assert np.array_equal(reached[:, :width], successors) and not reached[:, width:].any()
        chosen_total += len(picked)
        batch = stepped
    assert chosen_total == rounds * pool.num_envs


@pytest.mark.parametrize("name", ["gripper__prob05", "blocks__probBLOCKS-8-0"])
def test_pool_batches_of_one_instance_encode_like_copies_and_map_actions_back(name):
    task = text_task(name)
    table = rl.as_table(task)
    check_rounds(encoding.FlatRelationEncoder(task, **OPTIONS), task, table)


def test_pool_batches_of_several_instances_encode_like_copies_and_map_actions_back(sets):  # noqa: F811
    tasks = rl.TaskTable(sets["gripper"])
    check_rounds(encoding.FlatRelationEncoder(tasks, **OPTIONS), tasks, tasks, seed=1)


def test_pool_batches_of_several_instances_in_the_relation_layout(sets):  # noqa: F811
    tasks = rl.TaskTable(sets["blocks"])
    check_rounds(
        encoding.FlatRelationEncoder(tasks, layout="relation", index_dtype="int64", **OPTIONS), tasks, tasks, seed=2
    )


def test_a_pool_without_goals_encodes_the_goals_of_the_instances():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    batch = rl.CpuEnvPool(task, num_envs=3, seed=0, max_steps=5).reset()
    assert batch.goal_pos is None
    expansion = rl.expand(table, batch.states, batch.task_ids)
    encoded = encoder.encode(batch.states, batch.task_ids, actions=expansion)
    copied = encoder.encode(np.array(batch.states), np.array(batch.task_ids), actions=expansion)
    arrays_equal(encoded, copied)


def test_pool_arrays_are_read_in_place():
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    batch = pool_of(task).reset()
    for array in (batch.states, batch.task_ids, batch.goal_pos, batch.goal_neg):
        assert not array.flags.owndata and not array.flags.writeable
    expansion, encoded = encode_pool_batch(encoder, batch, rl.as_table(task))
    # the encoded batch owns its memory: it stays valid and unchanged after the pool moves on
    before = {k: np.array(v) for k, v in encoded.as_dict().items()}
    del batch
    for key, value in encoded.as_dict().items():
        assert np.array_equal(value, before[key]), key


def test_action_entities_of_a_batch_without_actions():
    task = text_task("gripper__prob05")
    batch = encoding.FlatRelationEncoder(task, **OPTIONS).encode([task.initial_state])
    assert len(encoding.action_entities(batch)) == 0
    assert len(encoding.action_rows(batch, rl.expand(task, [task.initial_state]))) == 0
    plain = encoding.FlatRelationEncoder(task, include_actions=False, entity_arrays="auto").encode([task.initial_state])
    assert plain.implicit_entities and len(encoding.action_entities(plain)) == 0
    with pytest.raises(ValueError, match="no action entities"):
        encoding.action_indices(plain, [0])
    with pytest.raises(ValueError, match="no action entities"):
        encoding.action_rows(plain, rl.expand(task, [task.initial_state]), [0])


@pytest.mark.parametrize(
    "options", [dict(include_actions=False, entity_arrays="auto"), {}], ids=["implicit-entities", "default"]
)
def test_the_action_mappings_of_a_batch_without_actions_are_empty(options):
    task = text_task("gripper__prob05")
    states = walk(task, steps=3, walks=1)
    expansion = rl.expand(rl.as_table(task), states)
    batch = encoding.FlatRelationEncoder(task, **options).encode(states)
    assert batch.implicit_entities == ("entity_arrays" in options)
    for mapped in (
        encoding.action_entities(batch),
        encoding.action_indices(batch),
        encoding.action_rows(batch, expansion),
    ):
        assert mapped.dtype == np.int64 and mapped.shape == (0,)


@pytest.mark.parametrize("framework", ["numpy", "dlpack", "torch", "jax"])
def test_the_action_mappings_read_a_batch_of_any_framework(framework):
    if framework in ("torch", "jax"):
        pytest.importorskip(framework)
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    states = walk(task, steps=3, walks=2)
    expansion = rl.expand(rl.as_table(task), states)
    reference = encoder.encode(states, actions=expansion)
    batch = encoder.encode(states, actions=expansion, framework=framework)
    assert type(batch.entity_role) is not np.ndarray or framework == "numpy"
    entities = encoding.action_entities(reference)
    assert len(entities) > len(states)
    assert np.array_equal(encoding.action_entities(batch), entities)
    assert np.array_equal(encoding.action_indices(batch), encoding.action_indices(reference))
    assert np.array_equal(encoding.action_rows(batch, expansion), encoding.action_rows(reference, expansion))
    # the chosen entities in the batch's own framework: its action targets as it exports them
    chosen = np.asarray(reference.target_entity)
    assert np.array_equal(encoding.action_indices(batch, batch.target_entity), encoding.action_indices(reference, chosen))
    assert np.array_equal(
        encoding.action_rows(batch, expansion, batch.target_entity), encoding.action_rows(reference, expansion, chosen)
    )


def test_action_rows_refuse_what_is_not_an_action_entity_of_the_batch():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    states = [task.initial_state, task.successor_states(task.initial_state)[0]]
    expansion = rl.expand(table, states)
    batch = encoder.encode(states, actions=expansion)
    with pytest.raises(ValueError, match="is not an action entity"):
        encoding.action_rows(batch, expansion, [0])  # an object
    with pytest.raises(ValueError, match="outside the batch"):
        encoding.action_rows(batch, expansion, [int(batch.entity_offsets[-1])])
    with pytest.raises(ValueError, match="outside the batch"):
        encoding.action_indices(batch, [-1])
    short = rl.expand(table, states[:1])
    with pytest.raises(ValueError, match="1 states, the batch 2 graphs"):
        encoding.action_rows(batch, short)


def test_an_entity_stands_for_the_first_row_of_its_action():
    # a disjunctive precondition splits an action in several schemas; each binding is listed once, at its first row
    task = text_task("miconic-simpleadl__s10-2")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    state = task.initial_state
    expansion = rl.expand(table, [state])
    batch = encoder.encode([state], actions=expansion)
    rows = encoding.action_rows(batch, expansion)
    seen = {}
    for row in range(int(expansion.offsets[1])):
        seen.setdefault(str(expansion.action(row)), row)
    assert sorted(rows) == sorted(seen.values())


# ------------------------------------------------------------------------------------------------ buffer reuse


CAPACITIES = dict(graphs=6, entities=400, args=4000, targets=200)


def test_encoding_into_a_batch_overwrites_its_arrays_and_equals_a_fresh_encoding():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    pool = pool_of(task, num_envs=4)
    batch = pool.reset()
    expansion = rl.expand(table, batch.states, batch.task_ids)
    reused = encoder.encode(batch.states, batch.task_ids, actions=expansion, capacities=CAPACITIES)
    exported = reused.relation_args  # a view of the batch's memory
    snapshot = np.array(exported)
    rng = np.random.default_rng(0)
    for _ in range(5):
        indices = rng.integers(0, np.asarray(expansion.counts)).astype(np.int32)
        batch = pool.step(batch.env_ids, indices)
        expansion = rl.expand(table, batch.states, batch.task_ids)
        fresh = encoder.encode(batch.states, batch.task_ids, actions=expansion, capacities=CAPACITIES)
        again = encoder.encode(batch.states, batch.task_ids, actions=expansion, capacities=CAPACITIES, into=reused)
        assert again is reused
        arrays_equal(reused, fresh)
        assert reused.sizes == fresh.sizes
        assert reused.relation_args.shape == fresh.relation_args.shape
    # the aliasing: a view exported before the call shows the new contents
    assert not np.array_equal(np.array(exported), snapshot)
    assert np.array_equal(exported, fresh.relation_args)


def test_reuse_zeroes_what_a_smaller_batch_does_not_fill():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    caps = dict(graphs=2, entities=300, args=3000, targets=150)
    by_size = sorted(walk(task, steps=25, walks=3), key=lambda s: len(task.successor_states(s)))
    first, second = [by_size[-1]] * 2, [by_size[0]] * 2
    expansions = [rl.expand(table, states) for states in (first, second)]
    batches = [encoder.encode(s, actions=e, capacities=caps) for s, e in zip((first, second), expansions)]
    assert batches[1].sizes["entities"] < batches[0].sizes["entities"]
    assert batches[1].sizes["args"] < batches[0].sizes["args"]
    reused = encoder.encode(first, actions=expansions[0], capacities=caps)
    encoder.encode(second, actions=expansions[1], capacities=caps, into=reused)
    arrays_equal(reused, batches[1])
    assert reused.sizes == batches[1].sizes


def test_reuse_zeroes_a_batch_that_does_not_fit_and_recovers_when_the_next_call_fits():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    by_size = sorted(walk(task, steps=25, walks=3), key=lambda s: len(task.successor_states(s)))
    small, big = [by_size[0]] * 2, [by_size[-1]] * 2
    expansions = {"small": rl.expand(table, small), "big": rl.expand(table, big)}
    roomy = dict(graphs=2, entities=400, args=4000, targets=200)
    entities = [encoder.encode(s, actions=expansions[k], capacities=roomy).sizes["entities"]
                for k, s in (("small", small), ("big", big))]
    assert entities[0] < entities[1]
    capacities = dict(roomy, entities=(entities[0] + entities[1]) // 2)  # holds the small batch, not the big one

    def zero(batch):
        return all(not np.any(v) for k, v in batch.as_dict().items() if k not in SCALARS)

    reused = encoder.encode(big, actions=expansions["big"], capacities=capacities)
    assert reused.sizes["capacity_overflow"] and zero(reused)
    encoder.encode(small, actions=expansions["small"], capacities=capacities, into=reused)
    fresh = encoder.encode(small, actions=expansions["small"], capacities=capacities)
    assert not reused.sizes["overflow"] and not zero(reused)
    arrays_equal(reused, fresh)
    assert reused.sizes == fresh.sizes
    # and the other way: the batch that fit becomes all zero when the next call does not
    encoder.encode(big, actions=expansions["big"], capacities=capacities, into=reused)
    assert reused.sizes["capacity_overflow"] and zero(reused)


def test_encoding_into_a_batch_checks_that_it_fits_the_call():
    task = text_task("gripper__prob05")
    table = rl.as_table(task)
    encoder = encoding.FlatRelationEncoder(task, **OPTIONS)
    states = [task.initial_state] * 2
    expansion = rl.expand(table, states)
    caps = dict(graphs=2, entities=300, args=3000, targets=150)
    batch = encoder.encode(states, actions=expansion, capacities=caps)
    exact = encoder.encode(states, actions=expansion)
    with pytest.raises(ValueError, match="into needs capacities"):
        encoder.encode(states, actions=expansion, into=batch)
    with pytest.raises(ValueError, match="encoded with capacities"):
        encoder.encode(states, actions=expansion, capacities=caps, into=exact)
    with pytest.raises(ValueError, match="'relation_args', the capacities ask for 3001"):
        encoder.encode(states, actions=expansion, capacities=dict(caps, args=3001), into=batch)
    three = [task.initial_state] * 3
    with pytest.raises(ValueError, match="has 2 graphs, the call 3"):
        encoder.encode(three, actions=rl.expand(table, three), capacities=dict(caps, graphs=3), into=batch)
    other = encoding.FlatRelationEncoder(task, layout="relation", **OPTIONS)
    with pytest.raises(ValueError, match="another schema"):
        other.encode(states, actions=expansion, capacities=caps, into=batch)
    with pytest.raises(TypeError):
        encoder.encode(states, actions=expansion, capacities=caps, into=object())
    # the batch is untouched by the refused calls
    arrays_equal(batch, encoder.encode(states, actions=expansion, capacities=caps))
