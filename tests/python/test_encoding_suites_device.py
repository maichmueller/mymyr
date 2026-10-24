"""Resident suite batches: stable domain grouping, live counts, capacities and graph replay."""

import gc
import os
import pathlib
import time

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task
from test_encoding_device_actions import gpu, tensor, toggle_tasks
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
        host = expected[k].as_dict() if k in expected else batch._encoder.encode(
            np.empty((0, batch._encoder.table.words + batch._encoder.table.numeric_words), np.uint64),
            np.empty(0, np.int32)).as_dict()
        for name, array in batch._arrays.items():
            if array is None or name not in batch.array_names or name == "sizes":
                continue
            raw = array.cpu().numpy()
            padded = np.zeros_like(raw)
            if name in host:
                value = np.asarray(host[name])
                if name in ("static_offsets", "static_arg_offsets") and value.size:
                    padded.fill(value[-1])
                padded[tuple(slice(0, size) for size in value.shape)] = value
            np.testing.assert_array_equal(raw, padded, err_msg=name)


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


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("live", [0, 3, 6])
def test_simple_layers_and_history_fill_raw_capacity_tensors(sets, layout, index, live, monkeypatch):  # noqa: F811
    from mymyr.encoding import _device

    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite, layout=layout, index_dtype=index, outputs=("relations", "simple"),
                                   max_goal_level=1, history=True, target_sources=("goal", "subgoal", "history"))
    states, ids = suite_batch(suite, per=1)
    states, ids = states[:live], ids[:live]
    n, w = states.shape
    pos, neg = np.zeros((n, 1, w), np.uint64), np.zeros((n, 1, w), np.uint64)
    pos[:, 0, 0] = 1
    layers = (pos, neg)
    history = (*layers, np.full((n, 1), -2, np.int32), np.ones(n, np.int32))
    expected = encoder.encode(states, ids, subgoals=layers, history=history)
    # Freed nonzero blocks make missing writes visible when output allocations reuse the cache.
    poison = [torch.full((size,), -1, dtype=torch.int32, device=device) for size in (4, 8, 16, 64, 4096, 16384)]
    torch.cuda.synchronize(device)
    del poison
    allocate = _device._outputs

    def poisoned_outputs(*args, **kwargs):
        arrays = allocate(*args, **kwargs)
        for array in arrays.values():
            array.fill_(1)
        return arrays

    monkeypatch.setattr(_device, "_outputs", poisoned_outputs)
    caps = {k: {**v, "history": n} for k, v in capacities(encoder, n).items()}
    actual = encoder.encode(tensor(states, device), tensor(ids, device), subgoals=tuple(tensor(x, device) for x in layers),
                            history=tuple(tensor(x, device) for x in history), capacities=caps)
    compare(actual, expected, n)


@pytest.mark.parametrize("encoder_type", [encoding.FlatHorizonEncoder, encoding.FlatTransitionEncoder,
                                         encoding.FlatTransitionEffectsEncoder])
@pytest.mark.parametrize("capacity", [False, True])
def test_device_suites_refuse_horizon_encoders_up_front(sets, encoder_type, capacity):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    options = {"horizon_mode": "full"} if encoder_type is encoding.FlatHorizonEncoder else {}
    encoder = encoding.SuiteEncoder(suite, encoder_type, **options)
    states, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    with pytest.raises(ValueError, match=encoder_type.__name__ + ".*host rows.*HorizonView"):
        encoder.encode(tensor(states, device), tensor(ids, device),
                       capacities=capacities(encoder, len(ids)) if capacity else None)


def test_capacity_suites_refuse_given_per_domain_expansions(sets):  # noqa: F811
    device = gpu()
    suite = suite_of(sets, "gripper+blocks")
    encoder = encoding.SuiteEncoder(suite)
    states, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    given = encoder.encode(states, ids, actions=True).expansions
    with pytest.raises(ValueError, match="capacity calls do not accept given per-domain expansions"):
        encoder.encode(tensor(states, device), tensor(ids, device), actions=given, capacities=capacities(encoder, len(ids)))


def test_bounded_labels_preserve_raw_overflow_and_allocate_no_successors(monkeypatch):
    from mymyr.encoding._device_actions import bounded_actions

    device = gpu()
    task = text_task("gripper__prob05")
    encoder = encoding.FlatRelationEncoder(task)
    rows = tensor(np.asarray(task.encode([task.initial_state] * 2)), device)
    ids = torch.zeros(2, dtype=torch.int32, device=device)
    live = torch.ones(1, dtype=torch.int64, device=device)

    def expand(table, states, task_ids, **outputs):
        assert "succ" not in outputs
        outputs["status"].fill_(1)
        outputs["offsets"].copy_(torch.tensor([0, 1, 100], dtype=torch.int32, device=device))
        return {}

    monkeypatch.setattr(rl, "expand_into", expand)
    result = bounded_actions(encoder, rows, ids, 4, live=live)
    assert result.succ is None
    assert result.status.item() == 1
    assert result.offsets.tolist() == [0, 1, 1]



def test_captured_expansions_have_distinct_workspaces_and_refuse_workspace_growth():
    from mymyr.encoding._device_actions import bounded_actions

    device = gpu()
    table = rl.as_table(text_task("gripper__prob05"))
    encoder = encoding.FlatRelationEncoder(table)
    rows = tensor(np.asarray(table.initial_states()), device)
    ids = torch.zeros(1, dtype=torch.int32, device=device)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    graphs, results = [], []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            bounded_actions(encoder, rows, ids, 100)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = bounded_actions(encoder, rows, ids, 100)
        assert result._workspace is not None
        graphs.append(graph)
        results.append(result)
    assert results[0]._workspace is not results[1]._workspace
    for stream, graph in zip(streams, graphs, strict=True):
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        stream.synchronize()
    for graph, result in zip(graphs, results, strict=True):
        expected = rl.expand(table, np.asarray(table.initial_states()))
        total = int(np.asarray(expected.offsets)[-1])
        np.testing.assert_array_equal(result.schema[:total].cpu().numpy(), np.asarray(expected.schema))
        np.testing.assert_array_equal(result.binding[:total].cpu().numpy(), np.asarray(expected.binding))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        bounded_actions(encoder, rows, ids, 100)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = bounded_actions(encoder, rows, ids, 100)
        with pytest.raises(ValueError, match="same stream, shapes and options"):
            bounded_actions(encoder, rows, ids, 200)
    graph.replay()
    torch.cuda.synchronize(device)


@pytest.mark.parametrize("capacity", [20, 21, 40, 60])
@pytest.mark.parametrize("captured", [False, True])
def test_action_expansion_ignores_padding(sets, capacity, captured):  # noqa: F811
    import mymyr

    device = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("PDDL front end unavailable")
    domain = mymyr.Domain.from_string("""(define (domain toggle) (:requirements :strips)
      (:predicates (link ?x ?y) (on ?x) (lit))
      (:action flip :parameters (?x ?y) :precondition (link ?x ?y) :effect (on ?x))
      (:action light :parameters () :precondition (and) :effect (lit)))""")
    task = mymyr.Task(domain.instantiate_string("""(define (problem p) (:domain toggle) (:objects o0 o1 o2 o3 o4)
      (:init (on o0) """ + " ".join(f"(link o{i} o{j})" for i in range(5) for j in range(5) if i != j)
      + ") (:goal (lit)))"))
    suite = rl.TaskSuite([rl.as_table(task), rl.TaskTable(sets["gripper"])])
    ids = np.arange(5, dtype=np.int32)
    states = np.asarray(suite.initial_states())[ids].copy()
    encoder = encoding.SuiteEncoder(suite, target_sources=("action",))
    expected = encoder.encode(states, ids, actions=True)
    words, task_ids = tensor(states, device), tensor(ids, device)
    caps = capacities(encoder, len(ids))
    # The action buffer has 21 live labels; encoding itself has a separate entity budget.
    caps[0]["entities"] = capacity
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        actual = encoder.encode(words, task_ids, actions=True, capacities=caps)
    stream.synchronize()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = encoder.encode(words, task_ids, actions=True, capacities=caps)
        graph.replay()
    x, host = actual.expansions[0], expected.expansions[0]
    assert int(np.asarray(host.offsets)[-1]) == 21
    assert x.status.item() == int(capacity < 21)
    np.testing.assert_array_equal(x.offsets.cpu(), [0, 21, 21, 21, 21, 21])
    if capacity >= 21:
        for name in ("schema", "binding", "parent"):
            raw = getattr(x, name).cpu().numpy()
            np.testing.assert_array_equal(raw[:21], np.asarray(getattr(host, name)))
            assert not raw[21:].any()
    if capacity >= len(expected[0].entity_ref):
        compare(actual, expected, len(ids))
    else:
        assert actual[0].sizes["capacity_overflow"]
        assert_same(actual[1], expected[1])


def _wait_workspace_release(reference):
    import gc
    import time
    from mymyr._core import _cuda

    for _ in range(500):
        gc.collect()
        if _cuda._expansion_workspace_released(reference):
            assert not _cuda._expansion_workspace_alive(reference)
            return
        time.sleep(0.01)
    assert _cuda._expansion_workspace_released(reference)


def test_graph_retains_expansion_after_all_results_are_dropped():
    device = gpu()
    import gc
    from mymyr._core import _cuda
    from mymyr.encoding._device_actions import bounded_actions

    table = rl.as_table(text_task("gripper__prob05"))
    encoder = encoding.FlatRelationEncoder(table)
    rows = tensor(np.asarray(table.initial_states()), device)
    ids = torch.zeros(1, dtype=torch.int32, device=device)
    expected = rl.expand(table, np.asarray(table.initial_states()))
    saved_schema = torch.empty(100, dtype=torch.int32, device=device)
    saved_binding = torch.empty((100, table.label_width), dtype=torch.int32, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        bounded_actions(encoder, rows, ids, 100)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = bounded_actions(encoder, rows, ids, 100)
        saved_schema.copy_(result.schema)
        saved_binding.copy_(result.binding)
    reference = result._workspace
    assert _cuda._expansion_workspace_owners(reference) == 1
    del result
    gc.collect()
    # Check ownership before launching anything that reads the captured workspace.
    assert _cuda._expansion_workspace_alive(reference)
    for _ in range(3):
        graph.replay()
        total = len(expected.schema)
        np.testing.assert_array_equal(saved_schema[:total].cpu(), np.asarray(expected.schema))
        np.testing.assert_array_equal(saved_binding[:total].cpu(), np.asarray(expected.binding))
        assert not saved_schema[total:].any()
        assert not saved_binding[total:].any()
    graph.reset()
    _wait_workspace_release(reference)


@pytest.mark.parametrize("drop_results_first", [False, True])
def test_workspace_survives_graph_destruction_during_side_stream_replay(drop_results_first):
    device = gpu()
    import gc
    from mymyr import cuda
    from mymyr._core import _cuda

    table = rl.as_table(text_task("gripper__prob05"))
    ctx = cuda.Context(0, max_bytes=64 << 20)
    states = tensor(np.asarray(table.initial_states()), device)
    ids = torch.zeros(1, dtype=torch.int32, device=device)
    expected = rl.expand(table, np.asarray(table.initial_states()))
    arrays = dict(schema=torch.zeros(100, dtype=torch.int32, device=device),
                  binding=torch.zeros((100, table.label_width), dtype=torch.int32, device=device),
                  offsets=torch.zeros(2, dtype=torch.int32, device=device),
                  status=torch.zeros(1, dtype=torch.int32, device=device))
    capture_stream = torch.cuda.Stream(device=device)
    replay_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        rl.expand_into(table, states, ids, **arrays, stream=capture_stream, ctx=ctx)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        result = rl.expand_into(table, states, ids, **arrays, stream=capture_stream, ctx=ctx)
    reference = result["_workspace"]
    assert _cuda._expansion_workspace_owners(reference) == 1
    # This independent guard makes a failed ownership assertion safe under asynchronous replay.
    safety_owner = _cuda._expansion_workspace_guard(reference)
    if drop_results_first:
        del result
        gc.collect()
        assert _cuda._expansion_workspace_alive(reference)
    replay_stream.wait_stream(torch.cuda.current_stream(device))
    done = torch.cuda.Event()
    with torch.cuda.stream(replay_stream):
        torch.cuda._sleep(200_000_000)
        graph.replay()
        done.record()
    try:
        assert not done.query()
        if not drop_results_first:
            del result
        graph.reset()
        gc.collect()
        if not done.query():
            assert _cuda._expansion_workspace_owners(reference) == 2
    finally:
        done.synchronize()
    del safety_owner
    total = len(expected.schema)
    np.testing.assert_array_equal(arrays["schema"][:total].cpu(), np.asarray(expected.schema))
    np.testing.assert_array_equal(arrays["binding"][:total].cpu(), np.asarray(expected.binding))
    np.testing.assert_array_equal(arrays["offsets"].cpu(), np.asarray(expected.offsets))
    assert arrays["status"].item() == 0
    _wait_workspace_release(reference)
    ctx.synchronize()


@pytest.mark.parametrize("release_during_capture", [False, True])
def test_repeated_capture_destruction_releases_workspace_memory(release_during_capture):
    device = gpu()
    import time
    from mymyr import cuda
    from mymyr._core import _cuda

    ctx = cuda.Context(0, max_bytes=64 << 20)
    table = rl.as_table(text_task("gripper__prob05"))
    states = tensor(np.asarray(table.initial_states()), device)
    arrays = dict(schema=torch.zeros(100, dtype=torch.int32, device=device),
                  offsets=torch.zeros(2, dtype=torch.int32, device=device),
                  status=torch.zeros(1, dtype=torch.int32, device=device))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    used, torch_used = [], []
    marker = torch.zeros(1, dtype=torch.int32, device=device)
    for cycle in range(12):
        with torch.cuda.stream(stream):
            rl.expand_into(table, states, **arrays, stream=stream, ctx=ctx)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = rl.expand_into(table, states, **arrays, stream=stream, ctx=ctx)
        reference = result["_workspace"]
        safety_owner = _cuda._expansion_workspace_guard(reference)
        assert _cuda._expansion_workspace_owners(reference) == 2
        graph.replay()
        torch.cuda.synchronize(device)
        graph.reset()
        assert _cuda._expansion_workspace_alive(reference)
        del result
        if release_during_capture:
            following = torch.cuda.CUDAGraph()
            with torch.cuda.graph(following, stream=stream):
                del safety_owner
                _wait_workspace_release(reference)
                marker.add_(1)
            following.replay()
            torch.cuda.synchronize(device)
            assert marker.item() == cycle + 1
            following.reset()
            ctx.synchronize()
            value = ctx.usage()["used"]
        else:
            del safety_owner
            _wait_workspace_release(reference)
            for _ in range(500):
                ctx.synchronize()
                value = ctx.usage()["used"]
                if value == 0:
                    break
                time.sleep(0.01)
        used.append(value)
        torch_used.append(torch.cuda.memory_allocated(device))
    assert used == [0] * len(used)
    assert torch_used == [torch_used[0]] * len(torch_used)


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("entities", [30, 4000])
def test_suite_multi_instance_toggle_padding(sets, captured, entities):  # noqa: F811
    """A suite domain of several instances (multi-instance kernels) whose zero state has applicable actions."""
    device = gpu()
    toggle = rl.TaskTable(toggle_tasks([5, 3]))
    suite = rl.TaskSuite([toggle, rl.TaskTable(sets["gripper"])])
    ids = np.asarray([2, 0, 3, 1, 4, 2], np.int32)
    states = np.asarray(suite.initial_states())[ids].copy()
    encoder = encoding.SuiteEncoder(suite, target_sources=("action",))
    expected = encoder.encode(states, ids, actions=True)
    live_total = int(np.asarray(expected.expansions[0].offsets)[-1])
    assert live_total == 21 + 7
    words, task_ids = tensor(states, device), tensor(ids, device)
    caps = capacities(encoder, len(ids))
    caps[0]["entities"] = entities
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        actual = encoder.encode(words, task_ids, actions=True, capacities=caps)
    stream.synchronize()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = encoder.encode(words, task_ids, actions=True, capacities=caps)
        graph.replay()
    torch.cuda.synchronize(device)
    x = actual.expansions[0]
    assert x.status.item() == 0
    if entities >= len(expected[0].entity_ref):
        compare(actual, expected, len(ids))


def _gripper_setup(device):
    from mymyr import cuda
    table = rl.as_table(text_task("gripper__prob05"))
    ctx = cuda.Context(0, max_bytes=64 << 20)
    states = tensor(np.asarray(table.initial_states()), device)
    ids = torch.zeros(1, dtype=torch.int32, device=device)
    arrays = dict(schema=torch.zeros(100, dtype=torch.int32, device=device),
                  binding=torch.zeros((100, table.label_width), dtype=torch.int32, device=device),
                  offsets=torch.zeros(2, dtype=torch.int32, device=device),
                  status=torch.zeros(1, dtype=torch.int32, device=device))
    return table, ctx, states, ids, arrays


@pytest.mark.parametrize("how", ["reset", "del"])
def test_queued_replays_keep_workspace_until_last_completes(how):
    """Three replays queued behind a sleep on another stream; graph destroyed and results dropped at once."""
    device = gpu()
    from mymyr._core import _cuda
    table, ctx, states, ids, arrays = _gripper_setup(device)
    cap_stream, replay_stream = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    cap_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(cap_stream):
        rl.expand_into(table, states, ids, **arrays, stream=cap_stream, ctx=ctx)
    cap_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=cap_stream):
        result = rl.expand_into(table, states, ids, **arrays, stream=cap_stream, ctx=ctx)
    ref = result["_workspace"]
    # Keep scratch alive even if a graph-ownership assertion fails during replay.
    safety = _cuda._expansion_workspace_guard(ref)
    replay_stream.wait_stream(torch.cuda.current_stream(device))
    events = []
    with torch.cuda.stream(replay_stream):
        torch.cuda._sleep(300_000_000)
        for _ in range(3):
            graph.replay()
            e = torch.cuda.Event()
            e.record()
            events.append(e)
    try:
        assert not events[0].query()
        del result
        if how == "reset":
            graph.reset()
        else:
            del graph
        gc.collect()
        pending = not events[-1].query()
        owners = _cuda._expansion_workspace_owners(ref)
        if pending:
            assert owners == 2, owners
    finally:
        events[-1].synchronize()
    # after completion, the graph's reference goes away (asynchronously)
    for _ in range(300):
        if _cuda._expansion_workspace_owners(ref) == 1:
            break
        time.sleep(0.01)
    assert _cuda._expansion_workspace_owners(ref) == 1
    del safety
    _wait_workspace_release(ref)
    ctx.synchronize()
    assert ctx.usage()["used"] == 0 or _poll_used_zero(ctx)


def _poll_used_zero(ctx):
    for _ in range(300):
        ctx.synchronize()
        if ctx.usage()["used"] == 0:
            return True
        time.sleep(0.01)
    return False


def _threads():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return -1


def test_many_capture_destroy_cycles_no_growth():
    """60 cycles: pool usage back to zero, one host worker, no thread or RSS growth."""
    device = gpu()
    from mymyr._core import _cuda

    table, ctx, states, ids, arrays = _gripper_setup(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    threads, rss = [], []
    for cycle in range(60):
        with torch.cuda.stream(stream):
            rl.expand_into(table, states, ids, **arrays, stream=stream, ctx=ctx)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = rl.expand_into(table, states, ids, **arrays, stream=stream, ctx=ctx)
        ref = result["_workspace"]
        safety = _cuda._expansion_workspace_guard(ref)
        assert _cuda._expansion_workspace_owners(ref) == 2
        graph.replay()
        done = torch.cuda.Event()
        done.record()
        del result
        del graph
        done.synchronize()
        del safety
        _wait_workspace_release(ref)
        assert _poll_used_zero(ctx)
        if cycle in (5, 59):
            threads.append(_threads())
            with open("/proc/self/statm") as f:
                rss.append(int(f.read().split()[1]) * 4096)
    assert threads[0] == threads[1], threads
    assert rss[1] - rss[0] < 32 << 20, rss


EXIT_SCRIPT = r"""
import sys, numpy as np, torch
sys.path.insert(0, {tests!r})
from mymyr import rl
from conftest import text_task
mode = {mode!r}
device = torch.device("cuda", 0)
table = rl.as_table(text_task("gripper__prob05"))
def t(a):
    a = np.asarray(a)
    return torch.from_numpy(a.view(np.int64).copy() if a.dtype == np.uint64 else a.copy()).to(device)
states = t(table.initial_states()); ids = torch.zeros(1, dtype=torch.int32, device=device)
arrays = dict(schema=torch.zeros(100, dtype=torch.int32, device=device),
              offsets=torch.zeros(2, dtype=torch.int32, device=device),
              status=torch.zeros(1, dtype=torch.int32, device=device))
s = torch.cuda.Stream(device=device)
s.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.stream(s):
    rl.expand_into(table, states, ids, **arrays, stream=s)
s.synchronize()
graphs = []
for _ in range(4):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        r = rl.expand_into(table, states, ids, **arrays, stream=s)
    with torch.cuda.stream(s):
        rl.expand_into(table, states, ids, **arrays, stream=s)
    graphs.append((g, r))
s.synchronize()
for g, _ in graphs:
    g.replay()
if mode == "drop":
    graphs.clear()
elif mode == "pending":
    torch.cuda._sleep(50_000_000)
    for g, _ in graphs:
        g.replay()
    graphs.clear()
print("ok", flush=True)
"""


@pytest.mark.parametrize("mode", ["keep", "drop", "pending"])
def test_interpreter_exit_with_graph_owned_workspaces(mode):
    gpu()
    from python_children import run_code

    script = EXIT_SCRIPT.format(tests=str(pathlib.Path(__file__).parent), mode=mode)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    bad = []
    for i in range(8):
        p = run_code(script, env=env, timeout=None, capture_output=True, check=False)
        if p.returncode != 0 or "ok" not in p.stdout or p.stderr.strip():
            bad.append((i, p.returncode, p.stderr[-400:]))
    assert not bad, bad
