"""Encoding the rows of a task suite: one encoder per domain, and a mixed batch encoded as a dictionary of batches per
domain together with each row's index in its batch."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import python_lines, text_task, walk
from test_encoding_pool import arrays_equal
from test_rl_table import sets  # noqa: F401  (a fixture: the table instances of several domains)
from test_rl_task_suite import SUITES, suite_batch, suite_of  # noqa: F401

OPTIONS = dict(target_sources=("action", "goal"), predicate_virtual_nodes=True)


def encode_domain_alone(suite, d, states, ids, **extra):
    """What encoding the rows of domain `d` with that domain's encoder alone gives, rows in the order of the call."""
    table = suite.tables[d]
    rows = np.flatnonzero(np.asarray(suite.domain_of)[ids] == d)
    local = np.asarray(suite.local_ids, np.int32)[ids[rows]]
    own = np.ascontiguousarray(states[rows][:, : table.words])
    encoder = encoding.FlatRelationEncoder(table, **OPTIONS)
    return rows, local, own, encoder


@pytest.mark.parametrize("key", ["gripper+blocks", "four"])
@pytest.mark.parametrize("grouped", [False, True])
def test_a_mixed_batch_is_one_batch_per_domain_equal_to_encoding_each_domain_alone(sets, key, grouped):  # noqa: F811
    suite = suite_of(sets, key)
    states, ids = suite_batch(suite, grouped=grouped)
    suite_encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    result = suite_encoder.encode(states, ids, actions=True)

    assert set(result) == set(np.asarray(suite.domain_of)[ids].tolist())
    assert result.num_rows == len(ids) and len(result) == len(result.batches)
    assert np.array_equal(result.task_ids, ids)
    for d in result:
        rows, local, own, encoder = encode_domain_alone(suite, d, states, ids)
        expansion = rl.expand(suite.tables[d], own, local)
        alone = encoder.encode(own, local, actions=expansion)
        arrays_equal(result[d], alone)
        assert np.array_equal(result.rows[d], rows)
        assert np.array_equal(np.asarray(result[d].task_ids), local)
        arrays_equal_expansions(result.expansions[d], expansion)
        assert result[d].schema.schema_fingerprint == suite_encoder.schemas[d].schema_fingerprint

    # each row's index in its batch: row i is graph index[i] of its domain's batch
    assert np.array_equal(result.domain, np.asarray(suite.domain_of)[ids])
    for i in range(result.num_rows):
        d, g = result.graph(i)
        assert d == result.domain[i] and result.rows[d][g] == i
        assert int(result[d].task_ids[g]) == int(suite.local_ids[int(ids[i])])
    counts = np.bincount(result.domain, minlength=suite.num_domains)
    for d in result:
        assert result[d].num_graphs == counts[d]
        assert sorted(result.index[result.domain == d]) == list(range(counts[d]))


def arrays_equal_expansions(a, b):
    for name in ("succ", "schema", "binding", "offsets", "parent"):
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


def test_row_indices_map_a_policys_choice_back_to_the_suite_row(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    result = encoder.encode(states, ids, actions=True)
    for d in result:
        batch, expansion = result[d], result.expansions[d]
        rows = encoding.action_rows(batch, expansion)
        graphs = np.searchsorted(np.asarray(batch.entity_offsets), encoding.action_entities(batch), side="right") - 1
        for g, row in zip(graphs.tolist(), rows.tolist()):
            suite_row = result.rows[d][g]
            # the action is a row of the expansion of the suite row's state
            assert expansion.offsets[g] <= row < expansion.offsets[g + 1]
            assert int(np.asarray(expansion.parent)[row]) == g
            assert result.domain[suite_row] == d


def test_inputs_per_row_are_split_by_domain(sets):  # noqa: F811
    suite = suite_of(sets, "four")
    states, ids = suite_batch(suite)
    goal_pos, goal_neg = (np.zeros((len(ids), suite.words), np.uint64) for _ in range(2))
    for i, g in enumerate(ids):  # one fluent goal literal per row: the first slot
        goal_pos[i, 0] = 1
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    result = encoder.encode(states, ids, goals=(goal_pos, goal_neg), actions=False)
    shared = encoder.encode(states, ids, goals=(goal_pos[:1], goal_neg[:1]))
    for d in result:
        rows, local, own, domain_encoder = encode_domain_alone(suite, d, states, ids)
        alone = domain_encoder.encode(own, local, goals=(goal_pos[rows], goal_neg[rows]))
        arrays_equal(result[d], alone)
        arrays_equal(shared[d], domain_encoder.encode(own, local, goals=(goal_pos[:1], goal_neg[:1])))
    assert not result.expansions


def test_expansions_that_are_given_are_used(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    made = encoder.encode(states, ids, actions=True)
    given = encoder.encode(states, ids, actions=dict(made.expansions))
    for d in made:
        arrays_equal(made[d], given[d])
    only_first = encoder.encode(states, ids, actions={0: made.expansions[0]})
    assert set(only_first.expansions) == {0}
    assert only_first[0].as_dict()["relation_args"].shape == made[0].as_dict()["relation_args"].shape
    with pytest.raises(ValueError, match="actions name domain 5"):
        encoder.encode(states, ids, actions={5: made.expansions[0]})


def test_a_suite_of_one_domain_is_its_table(sets):  # noqa: F811
    table = rl.TaskTable(sets["gripper"])
    states = np.asarray(table.initial_states())
    ids = np.arange(len(table), dtype=np.int32)
    suite_result = encoding.SuiteEncoder(rl.TaskSuite([table]), **OPTIONS).encode(states, ids)
    alone = encoding.FlatRelationEncoder(table, **OPTIONS).encode(states, ids)
    assert list(suite_result) == [0] and np.array_equal(suite_result.index, np.arange(len(ids)))
    arrays_equal(suite_result[0], alone)
    from_table = encoding.SuiteEncoder(table, **OPTIONS)
    arrays_equal(from_table.encode(states, ids)[0], alone)


def test_a_list_of_states_is_grouped_by_domain_without_python_code_per_state(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = [], []
    for g in range(len(suite)):
        walked = walk(suite[g], steps=4, seed=g, walks=1)
        states += walked
        ids += [g] * len(walked)
    order = np.random.default_rng(0).permutation(len(ids))
    states, ids = [states[i] for i in order], np.asarray(ids, np.int32)[order]
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    result = encoder.encode(states, ids)
    local = np.asarray(suite.local_ids, np.int32)
    for d in result:
        own = [states[i] for i in result.rows[d]]
        arrays_equal(result[d], encoder.encoders[d].encode(own, local[ids[result.rows[d]]]))
    many, many_ids = states * 8, np.tile(ids, 8)
    assert python_lines("encoding/_suite.py", lambda: encoder.encode(states, ids)) == python_lines(
        "encoding/_suite.py", lambda: encoder.encode(many, many_ids)
    )


def test_capacities_are_per_domain(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    exact = encoder.encode(states, ids, actions=True)
    capacities = {d: dict(graphs=exact[d].num_graphs, entities=20000, args=200000, targets=10000) for d in exact}
    fixed = encoder.encode(states, ids, actions=True, capacities=capacities)
    for d in exact:
        assert fixed[d].relation_args.shape == (200000,) and not fixed[d].sizes["overflow"]
        assert np.array_equal(fixed[d].relation_args[: exact[d].relation_args.shape[0]], exact[d].relation_args)
    with pytest.raises(ValueError, match="no entry for domain 1"):
        encoder.encode(states, ids, capacities={0: capacities[0]})


def test_suite_inputs_are_checked(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite)
    with pytest.raises(ValueError, match="task id"):
        encoder.encode(states, ids + 1000)
    with pytest.raises(ValueError, match="needs task_ids"):
        encoder.encode(states, None)
    with pytest.raises(TypeError, match="TaskSuite or a TaskTable"):
        encoding.SuiteEncoder([1, 2])



def test_horizon_encoders_work_per_domain(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite, encoding.FlatTransitionEncoder)
    result = encoder.encode(states, ids)
    for d in result:
        rows = result.rows[d]
        own = np.ascontiguousarray(states[rows][:, : suite.tables[d].words])
        local = np.asarray(suite.local_ids, np.int32)[ids[rows]]
        arrays_equal(result[d], encoding.FlatTransitionEncoder(suite.tables[d]).encode(own, local))


def test_a_suite_encoder_pickles_with_its_suite(sets):  # noqa: F811
    import pickle

    suite = suite_of(sets, "gripper+blocks")
    states, ids = suite_batch(suite)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    loaded = pickle.loads(pickle.dumps(encoder))
    first, second = encoder.encode(states, ids), loaded.encode(states, ids)
    assert list(first) == list(second)
    for d in first:
        arrays_equal(first[d], second[d])


def test_a_pool_over_a_suite_is_encoded_per_domain_and_stepped_with_the_mapped_actions(sets):  # noqa: F811
    suite = suite_of(sets, "gripper+blocks")
    pool = rl.CpuEnvPool(suite, num_envs=8, seed=0, max_steps=10, goals=True)
    encoder = encoding.SuiteEncoder(suite, **OPTIONS)
    step = pool.reset()
    for _ in range(5):
        goals = (step.goal_pos, step.goal_neg)
        result = encoder.encode(step.states, step.task_ids, goals=goals, actions=True)
        assert sorted(np.concatenate([result.rows[d] for d in result])) == list(range(8))
        indices = np.zeros(8, np.int32)
        rows_chosen = {}
        for d in result:
            batch, expansion = result[d], result.expansions[d]
            alone_rows, local, own, domain_encoder = encode_domain_alone(suite, d, np.asarray(step.states), np.asarray(step.task_ids))
            alone = domain_encoder.encode(
                own, local, goals=(np.asarray(step.goal_pos)[alone_rows], np.asarray(step.goal_neg)[alone_rows]),
                actions=rl.expand(suite.tables[d], own, local))
            arrays_equal(batch, alone)
            # the last action entity of every graph is the action each of its environments takes
            entities = encoding.action_entities(batch)
            owner = np.searchsorted(np.asarray(batch.entity_offsets), entities, side="right") - 1
            last = np.array([entities[owner == g][-1] for g in range(batch.num_graphs)])
            indices[result.rows[d]] = encoding.action_indices(batch, last)
            rows_chosen[d] = encoding.action_rows(batch, expansion, last)
        stepped = pool.step(step.env_ids, indices)
        assert not np.asarray(stepped.invalid).any()
        for d in result:
            successors = np.asarray(result.expansions[d].succ)[rows_chosen[d]]
            reached = np.asarray(stepped.final_states)[result.rows[d]][:, : successors.shape[1]]
            assert np.array_equal(reached, successors)
        step = stepped
