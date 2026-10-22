"""Shared static blocks, mixed-width observations, and device-resident training rows."""

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from mymyr.datasets import EncodingCollate, StateRows
from conftest import ROOT
from test_encoding_cuda import cases, device_encode, exact_capacities, gpu, host_encode, same_batch, to_device
from test_encoding_statics import OPTION_SETS

torch = pytest.importorskip("torch")


def capacities(batch):
    result = exact_capacities(batch)
    result.update(static_instances=len(batch.static_task_ids), static_args=len(batch.static_args))
    return result


@pytest.mark.parametrize("name", ["gripper", "blocks_table", "philosophers"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("options", OPTION_SETS.values(), ids=OPTION_SETS)
def test_shared_static_device_forms_equal_cpu(cases, name, layout, index, options):
    device = gpu()
    options = {**options, "target_sources": tuple(s for s in options.get("target_sources", ()) if s != "action")}
    encoder = encoding.FlatRelationEncoder(cases[name].source, statics="shared", layout=layout,
                                            index_dtype=index, **options)
    host = host_encode(encoder, cases[name])
    same_batch(host, device_encode(encoder, cases[name], device))
    caps = capacities(host)
    batch = device_encode(encoder, cases[name], device, capacities=caps)
    same_batch(host, batch)
    assert batch.sizes["static_instances"] == len(host.static_task_ids)
    caps["static_args"] = max(0, caps["static_args"] - 1)
    if len(host.static_args):
        short = device_encode(encoder, cases[name], device, capacities=caps)
        assert short.sizes["capacity_overflow"]
        assert all(not v.any() for v in short.as_dict("numpy").values() if isinstance(v, np.ndarray))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_shared_stream_collation_and_block_agreement(cases, layout, index):
    device = gpu()
    case = cases["blocks_table"]
    encoder = encoding.FlatRelationEncoder(case.source, statics="shared", layout=layout, index_dtype=index)
    rows, ids = to_device(case.rows, device), to_device(case.ids, device)
    host = host_encode(encoder, case)
    caps = capacities(host)
    stream = encoder.device_stream(case.n, device=device)
    stream.append(rows[:3], ids[:3])
    stream.append(rows[3:], ids[3:])
    same_batch(host, stream.flush(capacities=caps))
    pieces = [encoder.encode(rows[:3], ids[:3]), encoder.encode(rows[3:], ids[3:])]
    same_batch(host, encoding.collate(pieces))
    collator = encoding.DeviceCollator(pieces)
    same_batch(host, collator.encode(capacities=caps))
    # Alter a valid local object id while retaining the static block's shape.
    changed = encoder.encode(rows[3:], ids[3:])
    common = set(pieces[0].static_task_ids.tolist()) & set(changed.static_task_ids.tolist())
    assert common
    task = min(common)
    block = changed.static_task_ids.tolist().index(task)
    count = len(changed.static_relations)
    lo, hi = changed.static_arg_offsets[block * count:(block + 1) * count + 1][[0, -1]].tolist()
    assert hi > lo
    changed.static_args[lo] = (changed.static_args[lo] + 1) % int(case.source.tasks[task].num_objects)
    rejected = encoding.DeviceCollator([pieces[0], changed]).encode(capacities=caps)
    assert rejected.sizes["invalid_input"]
    with pytest.raises(ValueError, match="overflowed batch"):
        encoding.DeviceCollator([pieces[0], changed]).encode()


@pytest.fixture(scope="module")
def wide_table():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_file(ROOT / "tests/data/ipc/blocksworld-ipc/test/domain.pddl")
    tasks = []
    for n in (5, 50, 107):
        objects = " ".join(f"b{i}" for i in range(n))
        init = " ".join(f"(clear b{i}) (on-table b{i})" for i in range(n))
        tasks.append(mymyr.Task(domain.instantiate_string(
            f"(define (problem p{n}) (:domain blocksworld) (:objects {objects}) "
            f"(:init (arm-empty) {init}) (:goal (on b0 b1)))"
        ), atoms="dense"))
    table = rl.TaskTable(tasks)
    assert [t.words for t in tasks] == [1, 42, 184]
    return table


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("statics", ["repeat", "shared"])
def test_mixed_width_env_observations_and_dataset_rows_stay_on_device(wide_table, layout, index, statics, monkeypatch):
    device = gpu()
    from mymyr.rl.torch import BatchedEnv

    encoder = encoding.FlatRelationEncoder(wide_table, layout=layout, index_dtype=index, statics=statics)
    ids = torch.tensor([2, 0, 1, 0, 2, 1], dtype=torch.int32, device=device)
    env = BatchedEnv(wide_table, len(ids), task_ids=ids, sampler=None, device=device)
    try:
        for _ in range(2):
            obs = env.observation()
            for row, instance in enumerate((2, 0, 1, 0, 2, 1)):
                obs.states[row, (1, 42, 184)[instance]:] = -1
            host = encoder.encode(obs.states.cpu().numpy(), obs.task_ids.cpu().numpy())
            same_batch(host, encoder.encode(obs.states, obs.task_ids))
            dataset = StateRows(obs.states, obs.task_ids, labels={"value": ids})
            assert dataset.rows.data_ptr() == obs.states.data_ptr()
            assert dataset.task_ids.data_ptr() == obs.task_ids.data_ptr()
            # Encoding the stored device rows never calls a host tensor export.
            def no_host(*args, **kwargs):
                raise AssertionError("device rows must remain on the device")
            with monkeypatch.context() as patch:
                patch.setattr(torch.Tensor, "cpu", no_host)
                patch.setattr(torch.Tensor, "numpy", no_host)
                loaded = EncodingCollate(encoder)(dataset)
                selected = EncodingCollate(encoder)(dataset.take([3, 0, 4]))
            same_batch(host, loaded.graphs)
            same_batch(encoder.encode(obs.states[[3, 0, 4]].cpu().numpy(), obs.task_ids[[3, 0, 4]].cpu().numpy()),
                       selected.graphs)
            env.step()
    finally:
        env.close()


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.needs_gil
def test_shared_static_torch_op_fake_compile_and_capture(cases, layout, index):
    device = gpu()
    from mymyr.rl import torch as rt
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = cases["blocks_table"]
    encoder = encoding.FlatRelationEncoder(case.source, statics="shared", layout=layout, index_dtype=index)
    expected = host_encode(encoder, case)
    caps = capacities(expected)
    rows, ids = to_device(case.rows, device), to_device(case.ids, device)
    handle = rt.register(encoder)

    def call(rows, ids):
        return torch.ops.mymyr.encode_flat(handle, rows, ids, *([None] * 15), caps["graphs"], caps["entities"],
                                           caps["args"], caps["targets"], 0,
                                           caps["static_instances"], caps["static_args"])
    try:
        eager = call(rows, ids)
        same_batch(expected, encoding.DeviceFlatBatch.from_arrays(encoder, eager))
        with FakeTensorMode() as mode:
            fake = call(mode.from_tensor(rows), mode.from_tensor(ids))
        assert [(x.shape, x.dtype) for x in fake] == [(x.shape, x.dtype) for x in eager]
        compiled = torch.compile(call, backend="eager", fullgraph=True)
        same_batch(expected, encoding.DeviceFlatBatch.from_arrays(encoder, compiled(rows, ids)))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call(rows, ids)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call(rows, ids)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        same_batch(expected, encoding.DeviceFlatBatch.from_arrays(encoder, captured))
    finally:
        rt.unregister(handle)


@pytest.mark.parametrize("name", ["lamp", "ready"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("virtual", [False, True])
def test_nullary_static_auxiliary_entity_on_device(name, layout, index, virtual):
    device = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    from test_encoding_statics import NULLARY_STATIC_TASKS
    from conftest import walk

    domain, problem = NULLARY_STATIC_TASKS[name]
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    rows = np.asarray(task.encode(walk(task, steps=3, walks=1)))
    encoder = encoding.FlatRelationEncoder(task, statics="shared", layout=layout, index_dtype=index,
                                            predicate_virtual_nodes=virtual)
    host = encoder.encode(rows)
    actual = encoder.encode(to_device(rows, device))
    same_batch(host, actual)
    assert encoder.describe(actual, 0) == encoder.describe(host, 0)
    assert encoder.entity_names(actual, 0) == encoder.entity_names(host, 0)
    same_batch(host, encoder.encode(to_device(rows, device), capacities=capacities(host)))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("statics", ["repeat", "shared"])
def test_device_state_space_rows_encode_in_place(layout, statics, monkeypatch):
    device = gpu()
    from mymyr import datasets
    from conftest import text_task

    task = text_task("gripper__prob05", atoms="dense")
    space = datasets.state_space(task, device=0, remove_if_unsolvable=False)
    dataset = StateRows.from_state_space(space)
    encoder = encoding.FlatRelationEncoder(task, layout=layout, statics=statics)
    expected = encoder.encode(dataset.rows.cpu().numpy())
    pointers = (dataset.rows.data_ptr(), dataset.task_ids.data_ptr())
    with monkeypatch.context() as patch:
        def no_host(*args, **kwargs):
            raise AssertionError("device datasets must not download their stored rows")
        patch.setattr(torch.Tensor, "cpu", no_host)
        patch.setattr(torch.Tensor, "numpy", no_host)
        encoded = EncodingCollate(encoder)(dataset)
        subset = dataset.take([3, 0, 3])
        items = EncodingCollate(encoder)([subset[i] for i in range(len(subset))])
    assert pointers == (dataset.rows.data_ptr(), dataset.task_ids.data_ptr())
    assert encoded.task_ids is dataset.task_ids and encoded.labels is dataset.labels
    assert all(value.device == device for value in encoded.labels.values())
    same_batch(expected, encoded.graphs)
    same_batch(encoder.encode(dataset.rows[[3, 0, 3]].cpu().numpy()), items.graphs)


@pytest.mark.parametrize("statics", ["repeat", "shared"])
def test_device_data_loader_encodes_stored_goals_and_labels(cases, statics, monkeypatch):
    device = gpu()
    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, statics=statics, goal_derivations=("plain", "satisfied"))
    rows = to_device(case.rows, device)
    ids = torch.zeros(len(rows), dtype=torch.int32, device=device)
    goals = (rows.clone(), torch.zeros_like(rows))
    dataset = StateRows(rows, ids, goals=goals, labels={"value": torch.arange(len(rows), device=device)})
    from torch.utils.data import DataLoader

    with monkeypatch.context() as patch:
        def no_host(*args, **kwargs):
            raise AssertionError("loading device rows must not copy them to the host")
        patch.setattr(torch.Tensor, "cpu", no_host)
        patch.setattr(torch.Tensor, "numpy", no_host)
        batches = list(DataLoader(dataset, batch_size=5, num_workers=0, collate_fn=EncodingCollate(encoder)))
    for start, batch in zip(range(0, len(rows), 5), batches, strict=True):
        selection = slice(start, start + 5)
        expected = encoder.encode(case.rows[selection], goals=(case.rows[selection], np.zeros_like(case.rows[selection])))
        same_batch(expected, batch.graphs)
        assert batch.labels["value"].tolist() == list(range(start, min(start + 5, len(rows))))


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_shared_statics_with_layers_history_and_empty_batches(cases, layout, index):
    device = gpu()
    from test_encoding_cuda import layered_rows

    case = cases["gripper"]
    encoder = encoding.FlatRelationEncoder(case.source, statics="shared", layout=layout, index_dtype=index,
                                            max_goal_level=3, history=True, predicate_virtual_nodes=True,
                                            target_sources=("goal", "subgoal", "history"))
    subgoals, history = layered_rows(case)
    host = host_encode(encoder, case, subgoals=subgoals, history=history)
    same_batch(host, device_encode(encoder, case, device, subgoals=subgoals, history=history))
    same_batch(host, device_encode(encoder, case, device, subgoals=subgoals, history=history, capacities=capacities(host)))
    rows = np.empty((0, case.rows.shape[1]), np.uint64)
    empty = encoder.encode(rows)
    same_batch(empty, encoder.encode(to_device(rows, device)))
    same_batch(empty, encoder.encode(to_device(rows, device), capacities=capacities(empty)))
