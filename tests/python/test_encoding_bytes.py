"""Output byte reports use array metadata for every encoding and capacity form."""

from dataclasses import fields

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task

FORMS = ("relations", "coo", "implicit", "shared", "simple", "both", "actions", "layers",
         "numeric", "horizon", "horizon_delta", "horizon_action", "rooted", "transition", "effects")
CONFIGS = [(form, layout) for form in FORMS for layout in ("graph", "relation")
           if form != "simple" or layout == "graph"]


def cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mymyr.cuda")
    if not torch.cuda.is_available():
        pytest.skip("a visible CUDA device is required")
    return torch


def upload(array):
    torch = cuda()
    array = np.asarray(array)
    if array.dtype == np.uint64:
        array = array.view(np.int64)
    return torch.as_tensor(array.copy(), device="cuda:0")


def prepared(form, layout, index):
    task = (mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
            if form == "numeric" else text_task("gripper__prob05"))
    rows = np.asarray(task.encode([task.initial_state] * 2))
    options = dict(layout=layout, index_dtype=index)
    kwargs = {}
    cls = encoding.FlatRelationEncoder
    if form == "coo":
        options["emit_counts"] = False
    elif form == "implicit":
        options.update(include_actions=False, goal_derivations=(), entity_arrays="auto")
    elif form == "shared":
        options["statics"] = "shared"
    elif form == "simple":
        cls = encoding.SimpleEncoder
    elif form == "both":
        options["outputs"] = ("relations", "simple")
    elif form == "actions":
        options["target_sources"] = ("action",)
        expansion = rl.expand(task, rows)
        kwargs["actions"] = expansion
    elif form == "layers":
        options.update(max_goal_level=1, history=True, target_sources=("goal", "subgoal", "history"))
        masks = rows[:, None, :].copy()
        kwargs.update(subgoals=(masks, np.zeros_like(masks)),
                      history=(masks, np.zeros_like(masks), np.ones((2, 1), np.int32)))
    elif form in ("horizon", "horizon_delta", "horizon_action", "rooted"):
        cls = encoding.FlatRootedHorizonEncoder if form == "rooted" else encoding.FlatHorizonEncoder
        options["horizon_mode"] = form.removeprefix("horizon_") if "_" in form else "full"
        rows = encoding.horizon(task, rows, depth=1)
    elif form in ("transition", "effects"):
        cls = encoding.FlatTransitionEncoder if form == "transition" else encoding.FlatTransitionEffectsEncoder
    return cls(task, **options), rows, kwargs


def capacities(batch):
    arrays = batch.as_dict()
    return dict(graphs=batch.num_graphs + 1, entities=int(batch.entity_offsets[-1]) + 3,
                args=(len(batch.relation_args) if "relation_args" in arrays else int(batch.edge_offsets[-1]) // 2) + 7,
                **({"targets": len(batch.target_entity) + 3} if "target_entity" in arrays else {}),
                **({"history": len(batch.history_dt) + 3} if "history_dt" in arrays else {}),
                **({"numeric": len(batch.numeric_values) + 3} if "numeric_values" in arrays else {}),
                **({"static_instances": len(batch.static_task_ids) + 1, "static_args": len(batch.static_args) + 7}
                   if "static_args" in arrays else {}))


def device_inputs(rows, kwargs):
    if isinstance(rows, encoding.HorizonView):
        rows = encoding.HorizonView(**{f.name: None if getattr(rows, f.name) is None else upload(getattr(rows, f.name))
                                       for f in fields(rows)})
    else:
        rows = upload(rows)
    return rows, {name: tuple(upload(a) for a in ((value.schema, value.binding, value.offsets)
                                               if name == "actions" else value))
                  for name, value in kwargs.items()}


def expected_bytes(batch):
    if isinstance(batch, encoding.DeviceFlatBatch):
        arrays = dict(batch._arrays)
        for name in ("device_sizes", "device_static_sizes", "device_graphs"):
            value = getattr(batch, name)
            if value is not None:
                arrays[name] = value
    else:
        arrays = {name: value for name, value in batch.as_dict().items() if hasattr(value, "shape")}
    return {name: int(array.nbytes) for name, array in arrays.items()}


def check(batch):
    expected = expected_bytes(batch)
    assert batch.nbytes_by_array == expected
    assert batch.nbytes == sum(expected.values())
    # The caller receives a fresh mapping, so changing it cannot corrupt later reports.
    batch.nbytes_by_array.clear()
    assert batch.nbytes_by_array == expected
    return expected


@pytest.mark.parametrize("form,layout", CONFIGS)
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("device", [False, True], ids=["cpu", "cuda"])
def test_all_output_byte_reports(form, layout, index, device):
    if device:
        cuda()
    encoder, rows, kwargs = prepared(form, layout, index)
    exact = encoder.encode(rows, **kwargs)
    capacity = capacities(exact)
    if device:
        rows, kwargs = device_inputs(rows, kwargs)
    for caps in (None, capacity):
        batch = encoder.encode(rows, capacities=caps, **kwargs)
        check(batch)
        if layout == "relation" and form not in ("shared", "simple", "both"):
            view = encoding.hgraph(batch)
            assert view.nbytes == batch.nbytes
            assert view.nbytes_by_array == batch.nbytes_by_array


@pytest.mark.parametrize("outputs", [("simple",), ("relations", "simple")])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_numeric_simple_byte_reports(outputs, dtype):
    encoder, rows, _ = prepared("numeric", "graph", "int64")
    encoder = encoding.FlatRelationEncoder(encoder.table, outputs=outputs, numeric_dtype=dtype)
    batch = encoder.encode(rows)
    check(batch)
    check(encoder.encode(rows, capacities=capacities(batch)))


@pytest.mark.parametrize("device", [False, True], ids=["cpu", "cuda"])
def test_empty_and_overflow_bytes_count_capacity(device):
    if device:
        cuda()
    encoder, rows, kwargs = prepared("relations", "graph", "int32")
    if device:
        rows, kwargs = device_inputs(rows, kwargs)
    check(encoder.encode(rows[:0]))
    caps = dict(graphs=2, entities=1, args=1)
    batch = encoder.encode(rows, capacities=caps)
    assert batch.sizes["overflow"]
    check(batch)


def test_capacity_byte_reports_do_not_read_device_values(monkeypatch):
    torch = cuda()
    encoder, rows, _ = prepared("shared", "graph", "int32")
    capacity = capacities(encoder.encode(rows))
    words = upload(rows)
    ring_encoder, _, _ = prepared("relations", "graph", "int32")
    ring = ring_encoder.device_stream(4, device=0)
    ring.append(words)
    live = ring.flush(capacities=dict(graphs=4, entities=128, args=1024))
    batch = encoder.encode(words, capacities=capacity)
    expected = [check(batch), check(live)]
    def forbidden(*args, **kwargs):
        raise AssertionError("a byte report must use host metadata only")
    for name in ("cpu", "numpy", "item", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    for batch, sizes in zip((batch, live), expected, strict=True):
        assert batch.nbytes_by_array == sizes
        assert batch.nbytes == sum(sizes.values())
    monkeypatch.undo()
    # Capture refuses synchronizing operations as well as host exports.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        encoder.encode(words, capacities=capacity)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = encoder.encode(words, capacities=capacity)
        size = captured.nbytes
    graph.replay()
    stream.synchronize()
    assert size == sum(captured.nbytes_by_array.values())


@pytest.mark.parametrize("device", [False, True], ids=["cpu", "cuda"])
def test_suite_reports_include_row_mappings(device):
    if device:
        cuda()
    suite = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")])
    encoder = encoding.SuiteEncoder(suite)
    rows, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    if device:
        rows, ids = upload(rows), upload(ids)
    caps = {d: dict(graphs=2, entities=128, args=1024) for d in range(suite.num_domains)}
    for capacity in (None, caps):
        batch = encoder.encode(rows, ids, capacities=capacity)
        arrays = {f"batches.{d}.{name}": size for d, part in batch.items() for name, size in check(part).items()}
        arrays.update({f"rows.{d}": int(value.nbytes) for d, value in batch.rows.items()})
        arrays.update({name: int(getattr(batch, name).nbytes) for name in ("domain", "index", "task_ids")})
        assert batch.nbytes_by_array == arrays
        assert batch.nbytes == sum(arrays.values())


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
def test_jax_byte_reports_include_live_counts_and_stacked_axes(platform):
    if platform == "cuda":
        cuda()
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import mymyr.encoding.jax as ej
    if not hasattr(mymyr._core, "_encoding_jax"):
        pytest.skip("XLA FFI headers are required")
    devices = jax.devices(platform)
    encoder, rows, _ = prepared("relations", "graph", "int32")
    capacity = capacities(encoder.encode(rows))
    with jax.default_device(devices[0]):
        words = jnp.asarray(rows.view(np.uint32))
        for caps in (None, capacity):
            batch = ej.encode(encoder, words, capacities=caps)
            arrays = {name: int(value.nbytes) for name, value in batch._arrays.items()}
            if batch.device_sizes is not None:
                arrays["device_sizes"] = int(batch.device_sizes.nbytes)
            assert batch.nbytes_by_array == arrays
            assert batch.nbytes == sum(arrays.values())
        @jax.jit
        def run(words, live):
            batch = ej.encode(encoder, words, capacities=capacity, live=live)
            return batch, jnp.asarray(batch.nbytes, jnp.int32)
        batch, size = run(words, jnp.asarray(1, jnp.int32))
        assert int(size) == batch.nbytes
        assert batch.nbytes_by_array["device_graphs"] == 4
        stacked = jax.tree.map(lambda a: jnp.stack([a, a]), batch)
        assert stacked.nbytes == 2 * batch.nbytes


@pytest.mark.parametrize("device", [False, True], ids=["cpu", "cuda"])
def test_encoded_dataset_and_environment_byte_reports(device):
    torch = cuda() if device else pytest.importorskip("torch")
    from mymyr import datasets
    from mymyr.rl.torch import BatchedEnv
    suite = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")])
    encoder = encoding.SuiteEncoder(suite)
    rows, ids = np.asarray(suite.initial_states()), np.arange(len(suite), dtype=np.int32)
    labels = np.arange(len(suite), dtype=np.float32)
    if device:
        rows, ids, labels = upload(rows), upload(ids), upload(labels)
    dataset = datasets.StateRows(rows, ids, labels={"score": labels})
    capacities = {d: dict(graphs=2, entities=128, args=1024) for d in range(suite.num_domains)}
    loaded = datasets.EncodingCollate(encoder, capacities=capacities)(dataset)
    expected = {f"graphs.{name}": size for name, size in loaded.graphs.nbytes_by_array.items()}
    expected.update(task_ids=int(loaded.task_ids.nbytes), **{"labels.score": int(loaded.labels["score"].nbytes)})
    assert loaded.nbytes_by_array == expected
    assert loaded.nbytes == sum(expected.values())
    env = BatchedEnv(suite, 2, device=torch.device("cuda:0" if device else "cpu"),
                     observation=encoder, obs_capacities=dict(entities=128, args=1024))
    batch = env.encode_observation()
    expected = {f"batches.{d}.{name}": size for d, part in enumerate(batch.batches)
                for name, size in part.nbytes_by_array.items()}
    expected.update(domain=int(batch.domain.nbytes), index=int(batch.index.nbytes))
    assert batch.nbytes_by_array == expected
    assert batch.nbytes == sum(expected.values())
    env.close()


@pytest.mark.parametrize("platform", ["cpu", "cuda"])
def test_jax_suite_observation_byte_reports(platform):
    if platform == "cuda":
        cuda()
    jax = pytest.importorskip("jax")
    from mymyr.rl.jax import Env
    suite = rl.TaskSuite.group([text_task("gripper__prob05"), text_task("blocks__probBLOCKS-8-0")])
    encoder = encoding.SuiteEncoder(suite)
    env = Env(suite, device=platform, observation=encoder, obs_capacities=dict(entities=128, args=1024))
    _, observation = jax.jit(lambda key: env.reset(key, 2))(jax.random.key(0))
    batch = observation.encoding
    expected = {f"batches.{d}.{name}": size for d, part in enumerate(batch.batches)
                for name, size in part.nbytes_by_array.items()}
    expected.update(domain=int(batch.domain.nbytes), index=int(batch.index.nbytes))
    assert batch.nbytes_by_array == expected
    assert batch.nbytes == sum(expected.values())
