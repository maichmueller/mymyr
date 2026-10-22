"""Numeric device batches equal host batches byte for byte, including compact scalar storage and replayed rows."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT
from test_encoding_numeric import Case, MIX_DOMAIN, MIX_PROBLEMS, build_cases


@pytest.fixture(scope="module")
def numeric_cases():
    out = {case.name: case for case in build_cases()}
    if hasattr(mymyr, "Domain"):
        domain = mymyr.Domain.from_string(MIX_DOMAIN)
        tasks = [mymyr.Task(domain.instantiate_string(problem)) for problem in MIX_PROBLEMS]
        out["mixed"] = Case("mixed", rl.TaskTable(tasks))
    return out


def cuda():
    torch = pytest.importorskip("torch")
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("CUDA backend required")
    if not mc.available() or not torch.cuda.is_available():
        pytest.skip("CUDA backend and a visible GPU required")
    return torch


def rows_of(case):
    table = case.table
    out = np.zeros((len(case.states), table.words + table.numeric_words), np.uint64)
    for i, state in enumerate(case.states):
        out[i, :state.num_words] = state.words
        out[i, table.words:table.words + len(state.numeric_words)] = state.numeric_words
    return out


def tensor(torch, array):
    array = np.asarray(array)
    if array.dtype == np.uint64:
        array = array.view(np.int64)
    return torch.from_numpy(np.array(array, order="C")).cuda()


def equal(host, device, capacity=False):
    expected, got = host.as_dict(), device.as_dict("numpy")
    assert expected.keys() == got.keys()
    for name in expected:
        a, b = np.asarray(expected[name]), np.asarray(got[name])
        if capacity and b.ndim and a.ndim:
            b = b[:len(a)]
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


def capacities(batch, extra=0):
    return dict(graphs=batch.num_graphs + extra, entities=int(batch.entity_offsets[-1]) + extra,
                args=len(batch.relation_args) + extra, targets=len(batch.target_entity) if hasattr(batch, "target_entity") else 0,
                numeric=len(batch.numeric_values) + extra if hasattr(batch, "numeric_values") else 0)


@pytest.mark.parametrize("name", ["counters_i32", "counters_f64", "cs-drone", "m-refuel", "cs-expedition", "mixed"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_device_numeric_matches_host(numeric_cases, name, layout, index, dtype):
    torch = cuda()
    if name not in numeric_cases:
        pytest.skip("PDDL front end required")
    case = numeric_cases[name]
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    rows, ids = rows_of(case), case.ids
    host = encoder.encode(rows, ids)
    device_rows, device_ids = tensor(torch, rows), tensor(torch, ids)
    equal(host, encoder.encode(device_rows, device_ids))
    caps = capacities(host, 3)
    padded = encoder.encode(device_rows, device_ids, capacities=caps)
    assert padded.sizes["numeric"] == len(host.numeric_values) and not padded.sizes["overflow"]
    equal(host, padded, capacity=True)
    assert torch.count_nonzero(padded.numeric_values[len(host.numeric_values):]).item() == 0
    for at in (0, len(rows) - 1):
        equal(encoder.encode(rows[at:at + 1], ids[at:at + 1]),
              encoder.encode(device_rows[at], device_ids[at:at + 1]))
    overflow = encoder.encode(device_rows, device_ids, capacities={**caps, "numeric": len(host.numeric_values) - 1})
    assert overflow.sizes["capacity_overflow"]
    for array in overflow._arrays.values():
        assert torch.count_nonzero(array).item() == 0
    equal(encoder.encode(rows[:0], ids[:0]), encoder.encode(device_rows[:0], device_ids[:0]))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numeric_stream_collation_and_capture(numeric_cases, layout, index, dtype):
    torch = cuda()
    if "mixed" not in numeric_cases:
        pytest.skip("PDDL front end required")
    case = numeric_cases["mixed"]
    rows, ids = rows_of(case), case.ids
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    expected = encoder.encode(rows, ids)
    device_rows, device_ids = tensor(torch, rows), tensor(torch, ids)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cap = capacities(expected, 5)
        ring_capacity = len(rows) + 2
        cap["graphs"] = ring_capacity
        ring = encoder.device_stream(ring_capacity)
        for begin, end in ((0, 3), (3, len(rows))):
            ring.append(device_rows[begin:end], device_ids[begin:end])
        flushed = ring.flush(capacities=cap)
        equal(expected, flushed)
        assert len(ring) == 0
        # The next append starts away from zero and wraps around the ring.
        ring.append(device_rows, device_ids)
        equal(expected, ring.flush(capacities=cap))
        parts = [encoder.encode(device_rows[:3], device_ids[:3]), encoder.encode(device_rows[3:3], device_ids[3:3]),
                 encoder.encode(device_rows[3:], device_ids[3:])]
        empty_host = encoder.encode(rows[:0], ids[:0])
        equal(empty_host, encoding.DeviceCollator([parts[1], parts[1]]).encode())
        equal(empty_host, ring.flush(capacities=cap))
        prepared = encoding.DeviceCollator(parts)
        equal(expected, prepared.encode())
        equal(expected, prepared.encode(capacities=cap), capacity=True)
        overflow = prepared.encode(capacities={**cap, "numeric": len(expected.numeric_values) - 1})
        assert overflow.sizes["capacity_overflow"]
        for array in overflow._arrays.values():
            assert torch.count_nonzero(array).item() == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ring.append(device_rows, device_ids)
            captured = ring.flush(capacities=cap)
            collated = prepared.encode(capacities=cap)
        for _ in range(2):
            graph.replay()
            equal(expected, captured)
            equal(expected, collated, capacity=True)
        with pytest.raises(ValueError, match="numeric_dtype|schemas"):
            encoding.DeviceCollator([parts[0], encoding.FlatRelationEncoder(
                case.table, layout=layout, index_dtype=index, numeric_dtype="float64" if dtype == "float32" else "float32",
                goal_derivations=()).encode(device_rows[3:], device_ids[3:])])


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numeric_rounding_undefined_values_and_torch_op(numeric_cases, layout, index, dtype):
    torch = cuda()
    from mymyr.rl.torch import _ops as rt
    case = numeric_cases["counters_f64"]
    table = case.table
    ids = np.zeros(5, np.int32)
    rows = np.array(table.initial_states(ids), copy=True)
    values = np.array([1 + 2**-24, 1 + 3 * 2**-24, -(1 + 2**-24), 2**-149,
                       2**-150, -0.0, float("nan"), 16777217.0, -16777217.0, 0.0], np.float64)
    for g in range(len(rows)):
        rows[g, table.words:] = np.resize(values[2 * g:2 * g + 2].view(np.uint64), table.numeric_words)
    encoder = encoding.FlatRelationEncoder(table, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    expected = encoder.encode(rows, ids)
    assert np.isnan(expected.numeric_values).any()
    assert np.signbit(np.asarray(expected.numeric_values)[np.asarray(expected.numeric_values) == 0]).any()
    device_rows, device_ids = tensor(torch, rows), tensor(torch, ids)
    equal(expected, encoder.encode(device_rows, device_ids))
    cap = capacities(expected)
    handle = rt.register(encoder)
    try:
        def encode_op(r):
            arrays = torch.ops.mymyr.encode_flat(handle, r, device_ids, *([None] * 15), cap["graphs"],
                                                 cap["entities"], cap["args"], cap["targets"], 0, cap["numeric"])
            return encoding.DeviceFlatBatch.from_arrays(encoder, arrays)
        equal(expected, encode_op(device_rows))
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake = encode_op(device_rows)
            assert fake.numeric_values.shape == expected.numeric_values.shape
        # Acquire on the capture stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            encoder.encode(device_rows, device_ids, capacities=cap)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = encode_op(device_rows)
            graph.replay()
            equal(expected, captured)
    finally:
        rt.unregister(handle)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_numeric_rows_with_the_lane_disabled(numeric_cases, layout, index):
    torch = cuda()
    case = numeric_cases["cs-drone"]
    rows = rows_of(case)
    encoder = encoding.FlatRelationEncoder(case.table, include_numeric=False, layout=layout, index_dtype=index,
                                           goal_derivations=())
    equal(encoder.encode(rows), encoder.encode(tensor(torch, rows)))


def test_numeric_layout_and_simple_refusals_without_a_gpu():
    from mymyr.encoding._device import layout_of, _capacities
    task = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    encoder = encoding.FlatRelationEncoder(task)
    layout = layout_of(encoder)
    assert layout.numeric and layout.numeric_relations == 1
    with pytest.raises(ValueError, match="numeric"):
        _capacities(dict(graphs=1, entities=8, args=8), layout)
    with pytest.raises(ValueError, match="Simple"):
        layout_of(encoding.SimpleEncoder(task))
    from mymyr._core import _encoding_jax
    with pytest.raises(ValueError, match="JAX FFI has no numeric output buffer"):
        _encoding_jax.register(encoder)


@pytest.mark.parametrize("name", ["cs-counters", "cs-hydropower"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_numeric_environment_rows_on_the_general_device_path(name, layout, index):
    torch = cuda()
    from mymyr.rl import torch as rt
    task = mymyr.Task.from_text(str(ROOT / f"tests/data/numeric_tasks/{name}.txt"))
    env = rt.BatchedEnv(task, 4, device="cuda", seed=17, max_steps=7)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, index_dtype=index, numeric_dtype="float64",
                                           goal_derivations=())
    for _ in range(3):
        env.step()
        host = encoder.encode(env.states.cpu().numpy())
        equal(host, encoder.encode(env.states))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numeric_values_with_subgoal_history_and_virtual_nodes(numeric_cases, layout, index, dtype):
    torch = cuda()
    if "mixed" not in numeric_cases:
        pytest.skip("PDDL front end required")
    case = numeric_cases["mixed"]
    rows, ids = rows_of(case), case.ids
    pos = rows[:, None, :case.table.words].copy()
    neg = np.zeros_like(pos)
    history = (pos, neg, np.full((len(rows), 1), -1, np.int32), np.ones(len(rows), np.int32))
    subgoals = (pos, neg)
    encoder = encoding.FlatRelationEncoder(case.table, layout=layout, index_dtype=index, numeric_dtype=dtype,
        max_goal_level=1, history=True, predicate_virtual_nodes=True,
        goal_derivations=("plain", "satisfied"), target_sources=("goal", "subgoal", "history"))
    expected = encoder.encode(rows, ids, subgoals=subgoals, history=history)
    equal(expected, encoder.encode(tensor(torch, rows), tensor(torch, ids),
        subgoals=tuple(tensor(torch, a) for a in subgoals), history=tuple(tensor(torch, a) for a in history)))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_undefined_nullary_and_ground_functions_on_device(layout, index, dtype):
    torch = cuda()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("PDDL front end required")
    domain = mymyr.Domain.from_string(
        "(define (domain undef) (:requirements :strips :typing :numeric-fluents) (:types o) (:predicates (p))"
        " (:functions (u) (w ?x - o))"
        " (:action set-u :parameters () :precondition (and) :effect (assign (u) 1))"
        " (:action set-w :parameters (?x - o) :precondition (and) :effect (assign (w ?x) 4)))")
    task = mymyr.Task(domain.instantiate_string(
        "(define (problem undefined) (:domain undef) (:objects a b - o) (:init (= (w a) 2)) (:goal (and (p))))"))
    rows = np.asarray(task.encode([task.initial_state]))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    expected = encoder.encode(rows)
    assert np.isnan(expected.numeric_values).sum() == 2
    equal(expected, encoder.encode(tensor(torch, rows)))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_packed_int32_values_round_identically_on_device(numeric_cases, layout, index, dtype):
    torch = cuda()
    if "counters_i32" not in numeric_cases:
        pytest.skip("PDDL front end required")
    table = numeric_cases["counters_i32"].table
    ids = np.array([1, 0, 1], np.int32)
    rows = np.array(table.initial_states(ids), copy=True)
    limits = np.array([16777217, -16777217, 2147483647, -2147483648], np.int32)
    values = np.stack([np.resize(np.roll(limits, -g), 2 * table.numeric_words) for g in range(len(rows))])
    rows[:, table.words:] = values.view(np.uint64)
    encoder = encoding.FlatRelationEncoder(table, layout=layout, index_dtype=index, numeric_dtype=dtype,
                                           goal_derivations=())
    equal(encoder.encode(rows, ids), encoder.encode(tensor(torch, rows), tensor(torch, ids)))
