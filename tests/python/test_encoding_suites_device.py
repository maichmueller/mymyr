"""Resident suite batches: stable domain grouping, live counts, capacities and graph replay."""

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task
from test_encoding_device_actions import gpu, tensor
from test_rl_table import sets  # noqa: F401
from test_rl_task_suite import suite_batch, suite_of
from test_rl_torch_obs import assert_same

torch = pytest.importorskip("torch")


def capacities(encoder, n):
    return {k: dict(graphs=n, entities=4000, args=16000, targets=4000, static_instances=len(encoder.suite.tables[k]),
                    static_args=16000) if e.options["statics"] == "shared" else
            dict(graphs=n, entities=4000, args=16000, targets=4000) for k, e in enumerate(encoder.encoders)}


def compare(actual, expected, n):
    np.testing.assert_array_equal(actual.domain.cpu(), expected.domain)
    np.testing.assert_array_equal(actual.index.cpu(), expected.index)
    np.testing.assert_array_equal(actual.task_ids.cpu(), expected.task_ids)
    assert actual.num_rows == n
    for k, batch in actual.items():
        if k in expected:
            assert_same(batch, expected[k])
            np.testing.assert_array_equal(actual.rows[k][:batch.num_graphs].cpu(), expected.rows[k])
        else:
            assert batch.num_graphs == 0
            assert not batch.sizes["overflow"]
        assert torch.all(actual.rows[k][batch.num_graphs:] == -1)


@pytest.mark.parametrize("domains", [1, 2])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("actions", [False, True])
@pytest.mark.parametrize("capacity", [False, True])
@pytest.mark.parametrize("selection", ["all", "one_domain", "empty"])
def test_device_suite_batches_equal_cpu(sets, domains, layout, actions, capacity, selection):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks") if domains == 2 else rl.TaskSuite([rl.TaskTable(sets["gripper"])])
    states, ids = suite_batch(suite, per=1)
    if selection == "one_domain":
        mask = np.asarray(suite.domain_of)[ids] == 0
        states, ids = states[mask], ids[mask]
    elif selection == "empty":
        states, ids = states[:0], ids[:0]
    encoder = encoding.SuiteEncoder(suite, layout=layout, target_sources=("action", "goal"))
    expected = encoder.encode(states, ids, actions=actions)
    caps = capacities(encoder, len(ids)) if capacity else None
    actual = encoder.encode(tensor(states, device), tensor(ids, device), actions=actions, capacities=caps)
    assert set(actual) == (set(range(domains)) if capacity else set(expected))
    compare(actual, expected, len(ids))


@pytest.mark.parametrize("actions", [False, True])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_capacity_suite_capture_replays_changed_domain_counts_without_host_downloads(sets, actions, layout, monkeypatch):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite, layout=layout, index_dtype="int64", statics="shared")
    states, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    words, task_ids = tensor(states, device), tensor(ids, device)
    caps = capacities(encoder, len(ids))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        encoder.encode(words, task_ids, actions=actions, capacities=caps)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()

    def no_host(*args, **kwargs):
        raise AssertionError("capacity suite encoding must not download device values")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "tolist"):
            patch.setattr(torch.Tensor, name, no_host)
        with torch.cuda.graph(graph, stream=stream):
            actual = encoder.encode(words, task_ids, actions=actions, capacities=caps)
    for selection in (ids, np.zeros_like(ids), np.full_like(ids, len(suite) - 1), ids[::-1].copy()):
        host_words = states[selection].copy()
        words.copy_(tensor(host_words, device))
        task_ids.copy_(tensor(selection, device))
        graph.replay()
        torch.cuda.synchronize(device)
        expected = encoder.encode(host_words, selection, actions=actions)
        compare(actual, expected, len(ids))


@pytest.mark.parametrize("capacity", [False, True])
@pytest.mark.parametrize("actions", [False, True])
def test_a_domain_without_relations_has_a_device_batch(capacity, actions):
    device = gpu()
    import mymyr
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("(define (domain empty) (:requirements :strips) (:predicates))")
    task = mymyr.Task(domain.instantiate_string("(define (problem p) (:domain empty) (:init) (:goal (and)))"))
    suite = rl.TaskSuite([rl.as_table(task), rl.as_table(text_task("gripper__prob05"))])
    encoder = encoding.SuiteEncoder(suite)
    rows, ids = np.asarray(suite.initial_states()), np.arange(2, dtype=np.int32)
    actual = encoder.encode(tensor(rows, device), tensor(ids, device), actions=actions,
                            capacities=capacities(encoder, 2) if capacity else None)
    compare(actual, encoder.encode(rows, ids, actions=actions), 2)


def test_capacity_suite_returns_before_device_work_and_reports_overflow(sets):  # noqa: F811
    from test_encoding_device_transitions import returns_before_the_device
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite)
    rows, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    words, task_ids = tensor(rows, device), tensor(ids, device)
    caps = capacities(encoder, len(ids))
    call = lambda: encoder.encode(words, task_ids, actions=True, capacities=caps)
    call()
    early, actual = returns_before_the_device(call)
    assert early
    compare(actual, encoder.encode(rows, ids, actions=True), len(ids))
    short = {k: {**v, "entities": 0} for k, v in caps.items()}
    failed = encoder.encode(words, task_ids, actions=True, capacities=short)
    for batch in failed.values():
        assert batch.sizes["capacity_overflow"] and not batch.sizes["invalid_input"]
        assert all(not x.any() for x in batch.as_dict("numpy").values() if isinstance(x, np.ndarray))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("actions", [False, True])
def test_suite_splits_shared_and_per_row_masks_and_layers(sets, layout, actions):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite, layout=layout, max_goal_level=1, history=True,
                                   target_sources=("action", "goal", "subgoal", "history"))
    states, ids = suite_batch(suite, per=1)
    n, w = states.shape
    pos, neg = np.zeros((n, w), np.uint64), np.zeros((n, w), np.uint64)
    pos[:, 0] = 1
    layer = (pos[:, None, :].copy(), neg[:, None, :].copy())
    history = (*layer, np.full((n, 1), -2, np.int32), np.ones(n, np.int32))
    extra = dict(goals=(pos[:1], neg[:1]), subgoals=layer, history=history)
    expected = encoder.encode(states, ids, actions=actions, **extra)
    caps = {k: {**v, "history": n} for k, v in capacities(encoder, n).items()}
    def upload(value):
        return tuple(upload(x) for x in value) if isinstance(value, tuple) else tensor(value, device)
    actual = encoder.encode(tensor(states, device), tensor(ids, device), actions=actions, capacities=caps,
                            **{k: upload(v) for k, v in extra.items()})
    compare(actual, expected, n)


def test_split_schema_action_expansion_fits_exact_entity_capacities():
    from test_encoding_device_actions import split_task, caps
    device = gpu()
    suite = rl.TaskSuite([rl.as_table(split_task())])
    encoder = encoding.SuiteEncoder(suite, target_sources=("action",))
    rows, ids = np.asarray(suite.initial_states()), np.zeros(1, np.int32)
    expected = encoder.encode(rows, ids, actions=True)
    actual = encoder.encode(tensor(rows, device), tensor(ids, device), actions=True, capacities={0: caps(expected[0])})
    compare(actual, expected, 1)


@pytest.mark.parametrize("invalid", [-1, 2**31 - 1])
@pytest.mark.parametrize("capacity", [False, True])
def test_device_suite_validates_ids_with_bounded_gathers(sets, invalid, capacity):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite)
    rows = tensor(np.asarray(suite.initial_states()), device)
    ids = torch.arange(len(suite), dtype=torch.int32, device=device)
    ids[0] = invalid
    if capacity:
        result = encoder.encode(rows, ids, capacities=capacities(encoder, len(suite)))
        assert result[0].sizes["invalid_input"]
    else:
        with pytest.raises(ValueError, match="invalid"):
            encoder.encode(rows, ids)


@pytest.mark.parametrize("capacity", [False, True])
def test_automatic_suite_actions_refuse_expansion_that_downloads_rows(sets, capacity):  # noqa: F811
    device = gpu()
    suite = rl.TaskSuite([rl.TaskTable(sets["miconic-simpleadl"])])
    encoder = encoding.SuiteEncoder(suite)
    states = tensor(np.asarray(suite.initial_states()), device)
    ids = torch.arange(len(suite), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="asynchronous"):
        encoder.encode(states, ids, actions=True, capacities=capacities(encoder, len(suite)) if capacity else None)


@pytest.mark.parametrize("actions", [False, True])
def test_one_domain_exact_calls_download_only_the_flat_encoders_output_sizes(sets, actions, monkeypatch):  # noqa: F811
    device = gpu()
    table = rl.TaskTable(sets["gripper"])
    encoder = encoding.SuiteEncoder(table)
    rows, ids = np.asarray(table.initial_states()), np.arange(len(table), dtype=np.int32)
    words, task_ids = tensor(rows, device), tensor(ids, device)
    expected = encoder.encode(rows, ids, actions=actions)
    encoder.encode(words, task_ids, actions=actions)

    def no_host(*args, **kwargs):
        raise AssertionError("a one-domain suite needs no device counts on the host for grouping")

    with monkeypatch.context() as patch:
        for name in ("cpu", "numpy", "item", "tolist"):
            patch.setattr(torch.Tensor, name, no_host)
        actual = encoder.encode(words, task_ids, actions=actions)
    compare(actual, expected, len(ids))
