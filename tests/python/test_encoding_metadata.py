"""Batch metadata across layouts, omitted arrays, frameworks and live capacity prefixes."""

import numpy as np
import pytest

from mymyr import encoding, rl
from conftest import text_task, walk

HELPERS = (encoding.entity_counts, encoding.tuple_counts, encoding.entity_graph, encoding.entity_roles,
           encoding.object_mask, encoding.object_entities)
FIXED = HELPERS[:-1]


@pytest.fixture(scope="module")
def case():
    task = text_task("gripper__prob05")
    rows = np.ascontiguousarray(task.encode(walk(task, steps=2, walks=1)))
    return task, rows


def encoder_of(task, layout, counts=True, implicit=False, index="int32"):
    return encoding.FlatRelationEncoder(task, layout=layout, emit_counts=counts, index_dtype=index,
                                       include_actions=False, entity_arrays="auto" if implicit else "always",
                                       goal_derivations=() if implicit else ("plain",),
                                       target_sources=() if implicit else ("goal",))


def capacities(batch, graphs=None):
    return dict(graphs=batch.num_graphs if graphs is None else graphs,
                entities=int(batch.entity_offsets[-1]) + 7, args=int(batch.relation_arg_offsets[-1]) + 11,
                targets=int(batch.target_offsets[-1]) + 3 if hasattr(batch, "target_offsets") else 0)


def expected(batch, *, graphs=None, entities=None):
    """Reference metadata from the explicit host arrays, padding to the requested capacities."""
    g = batch.num_graphs
    graphs = g if graphs is None else graphs
    live_entities = int(batch.entity_offsets[-1])
    entities = live_entities if entities is None else entities
    index = batch.entity_offsets.dtype
    sizes = np.zeros(graphs, index)
    tuples = np.zeros(graphs, index)
    pooling = np.full(entities, g - 1, index)
    roles = np.zeros(entities, np.uint8)
    mask = np.zeros(entities, bool)
    objects = []
    for graph in range(g):
        lo, hi = map(int, batch.entity_offsets[graph:graph + 2])
        n = int(batch.num_objects[graph])
        sizes[graph] = hi - lo
        tuples[graph] = batch.relation_counts[graph].sum()
        pooling[lo:hi] = graph
        roles[lo:hi] = batch.entity_role[lo:hi]
        mask[lo:lo + n] = True
        objects.extend(range(lo, lo + n))
    return sizes, tuples, pooling, roles, mask, np.asarray(objects, index)


def assert_metadata(values, reference):
    for helper, value, wanted in zip(HELPERS, values, reference, strict=True):
        if type(value).__module__.split(".")[0] == "torch":
            value = value.cpu().numpy()
        value = np.asarray(value)
        assert value.dtype == wanted.dtype, helper.__name__
        np.testing.assert_array_equal(value, wanted, err_msg=helper.__name__)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("counts", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_numpy_metadata(case, layout, counts, implicit, index):
    task, rows = case
    encoder = encoder_of(task, layout, counts, implicit, index)
    explicit = encoder_of(task, layout, index=index)
    if implicit:
        explicit = encoding.FlatRelationEncoder(task, layout=layout, include_actions=False, goal_derivations=(),
                                                index_dtype=index)
    for selected in (rows[:0], rows[:1], rows):
        batch = encoder.encode(selected)
        reference = explicit.encode(selected)
        keys, sizes = set(batch.as_dict()), batch.nbytes_by_array
        values = [helper(batch) for helper in HELPERS]
        assert all(isinstance(value, np.ndarray) for value in values)
        assert_metadata(values, expected(reference))
        assert set(batch.as_dict()) == keys and batch.nbytes_by_array == sizes
        if implicit:
            assert batch.implicit_entities and "entity_role" not in keys
        assert ("relation_counts" in keys) is counts


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("implicit", [False, True])
def test_host_capacity_tails_and_overflow(case, layout, implicit):
    task, rows = case
    encoder = encoder_of(task, layout, False, implicit)
    explicit = encoder_of(task, layout) if not implicit else encoding.FlatRelationEncoder(
        task, layout=layout, include_actions=False, goal_derivations=())
    reference = explicit.encode(rows)
    caps = capacities(reference, graphs=len(rows) + 2)
    batch = encoder.encode(rows, capacities=caps)
    extent = int(reference.entity_offsets[-1]) if implicit else caps["entities"]
    assert_metadata([helper(batch) for helper in HELPERS], expected(reference, entities=extent))
    overflow = encoder.encode(rows, capacities=dict(caps, args=0))
    assert overflow.sizes["overflow"]
    extent = 0 if implicit else caps["entities"]
    wanted = (np.zeros(len(rows), np.int32), np.zeros(len(rows), np.int32),
              np.full(extent, len(rows) - 1, np.int32), np.zeros(extent, np.uint8),
              np.zeros(extent, bool), np.empty(0, np.int32))
    assert_metadata([helper(overflow) for helper in HELPERS], wanted)


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_host_streams_and_suites(case, layout):
    task, rows = case
    encoder = encoder_of(task, layout, counts=False, implicit=True)
    stream = encoder.stream()
    stream.append(rows[:1])
    stream.append(rows[1:])
    flushed = stream.flush()
    direct = encoder.encode(rows)
    for helper in HELPERS:
        np.testing.assert_array_equal(helper(flushed), helper(direct))
    other = text_task("blocks__probBLOCKS-8-0")
    suite = rl.TaskSuite([rl.TaskTable([task]), rl.TaskTable([other])])
    padded = np.zeros((2, suite.words), np.uint64)
    for i, item in enumerate((task, other)):
        own = np.asarray(item.encode([item.initial_state]))
        padded[i, :own.shape[1]] = own[0]
    result = encoding.SuiteEncoder(suite, layout=layout, include_actions=False, emit_counts=False,
                                   entity_arrays="auto", goal_derivations=()).encode(padded, np.array([0, 1], np.int32))
    keys, sizes = set(result), result.nbytes_by_array
    for helper in HELPERS:
        actual = helper(result)
        assert set(actual) == keys
        for domain in result:
            np.testing.assert_array_equal(actual[domain], helper(result[domain]))
    assert result.nbytes_by_array == sizes


@pytest.mark.parametrize("framework", ["torch", "jax", "dlpack"])
@pytest.mark.parametrize("implicit", [False, True])
def test_host_batch_uses_its_export_framework(case, framework, implicit):
    if framework != "dlpack":
        pytest.importorskip(framework)
    task, rows = case
    encoder = encoder_of(task, "relation", counts=False, implicit=implicit)
    reference = encoder.encode(rows)
    batch = encoder.encode(rows, framework=framework)
    for helper in HELPERS:
        value = helper(batch)
        if framework == "torch":
            import torch
            assert isinstance(value, torch.Tensor) and value.device.type == "cpu"
            value = value.numpy()
        elif framework == "jax":
            import jax
            assert isinstance(value, jax.Array)
        else:
            assert isinstance(value, np.ndarray)
        np.testing.assert_array_equal(value, helper(reference))


def gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    try:
        import mymyr.cuda as mc
    except ImportError:
        pytest.skip("mymyr was built without the CUDA backend")
    if not mc.available():
        pytest.skip("mymyr sees no CUDA device")
    return torch


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_device_suite_metadata(case, layout):
    torch = gpu()
    task, _ = case
    other = text_task("blocks__probBLOCKS-8-0")
    suite = rl.TaskSuite([rl.TaskTable([task]), rl.TaskTable([other])])
    encoder = encoding.SuiteEncoder(suite, layout=layout, include_actions=False, emit_counts=False,
                                   entity_arrays="auto", goal_derivations=())
    reference_encoder = encoding.SuiteEncoder(suite, layout=layout, include_actions=False, goal_derivations=())
    rows = np.asarray(suite.initial_states())
    ids = np.arange(2, dtype=np.int32)
    caps = {d: capacities(reference_encoder.encoders[d].encode([item.initial_state] * 2), graphs=2)
            for d, item in enumerate((task, other))}
    for chosen in (np.array([0, 0]), np.array([1, 0])):
        selected = np.ascontiguousarray(rows[chosen])
        task_ids = ids[chosen]
        reference = reference_encoder.encode(selected, task_ids)
        result = encoder.encode(torch.from_numpy(selected.view(np.int64)).to("cuda:0"),
                                torch.from_numpy(task_ids).to("cuda:0"), capacities=caps)
        values = [helper(result) for helper in HELPERS]
        for domain in result:
            if domain in reference:
                wanted = expected(reference[domain], graphs=2, entities=caps[domain]["entities"])
            else:
                empty = reference_encoder.encoders[domain].encode(rows[:0])
                wanted = expected(empty, graphs=2, entities=caps[domain]["entities"])
            assert_metadata([value[domain] for value in values], wanted)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("counts", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_torch_device_metadata(case, layout, counts, implicit, index):
    torch = gpu()
    torch.cuda.set_per_process_memory_fraction(0.1, 0)
    task, rows = case
    encoder = encoder_of(task, layout, counts, implicit, index)
    host_encoder = encoder_of(task, layout, index=index) if not implicit else encoding.FlatRelationEncoder(
        task, layout=layout, include_actions=False, goal_derivations=(), index_dtype=index)
    device_rows = torch.from_numpy(rows.view(np.int64)).to("cuda:0")
    for selected in (device_rows[:0], device_rows):
        reference = host_encoder.encode(rows[:len(selected)])
        for caps in (None, capacities(reference)):
            batch = encoder.encode(selected, capacities=caps)
            keys, sizes = set(batch.as_dict()), batch.nbytes_by_array
            values = [helper(batch) for helper in HELPERS]
            assert all(isinstance(value, torch.Tensor) and value.device == selected.device for value in values)
            extent = int(reference.entity_offsets[-1]) if caps is None else caps["entities"]
            assert_metadata(values, expected(reference, entities=extent))
            assert set(batch.as_dict()) == keys and batch.nbytes_by_array == sizes


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("implicit", [False, True])
def test_device_stream_metadata_capture(case, layout, implicit):
    torch = gpu()
    task, rows = case
    encoder = encoder_of(task, layout, counts=False, implicit=implicit)
    reference_encoder = encoder_of(task, layout) if not implicit else encoding.FlatRelationEncoder(
        task, layout=layout, include_actions=False, goal_derivations=())
    caps = capacities(reference_encoder.encode(rows), graphs=4)
    stream = torch.cuda.Stream(0)
    device_rows = torch.from_numpy(rows.view(np.int64)).to("cuda:0")
    stream.wait_stream(torch.cuda.current_stream(0))
    with torch.cuda.stream(stream):
        encoder.warm(device=0)
        ring = encoder.device_stream(4, device=0)
        ring.append(device_rows[:1])
        warm = ring.flush(capacities=caps)
        for helper in FIXED:
            helper(warm)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            batch = ring.flush(capacities=caps)
            values = tuple(helper(batch) for helper in FIXED)
        for live in (0, 1, len(rows), 0):
            if live:
                ring.append(device_rows[:live])
            capture.replay()
            stream.synchronize()
            reference = reference_encoder.encode(rows[:live])
            assert_metadata((*values, encoding.object_entities(batch)),
                            expected(reference, graphs=4, entities=caps["entities"]))
    stream.synchronize()


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("counts", [False, True])
@pytest.mark.parametrize("implicit", [False, True])
def test_jax_metadata_under_jit(case, layout, counts, implicit):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from mymyr._core import _rl_jax
    if not _rl_jax.has_ffi():
        pytest.skip("mymyr was built without the XLA FFI headers")
    from mymyr.encoding import jax as ej
    task, rows = case
    encoder = encoder_of(task, layout, counts, implicit)
    reference_encoder = encoder_of(task, layout) if not implicit else encoding.FlatRelationEncoder(
        task, layout=layout, include_actions=False, goal_derivations=())
    caps = capacities(reference_encoder.encode(rows))
    words = jnp.asarray(rows.view(np.uint32))

    @jax.jit
    def encode(words, live):
        batch = ej.encode(encoder, words, capacities=caps, live=live)
        return batch, tuple(helper(batch) for helper in FIXED)

    inspect = jax.jit(lambda b: tuple(helper(b) for helper in FIXED))
    try:
        for live in (0, 1, len(rows)):
            batch, values = encode(words, jnp.int32(live))
            assert all(isinstance(value, jax.Array) for value in values)
            # Passing the batch through another JIT checks that its static entity extent survives the pytree.
            again = inspect(batch)
            reference = reference_encoder.encode(rows[:live])
            wanted = expected(reference, graphs=len(rows), entities=caps["entities"])
            assert_metadata((*values, encoding.object_entities(batch)), wanted)
            assert_metadata((*again, encoding.object_entities(batch)), wanted)
            sizes = batch.nbytes_by_array
            for helper in FIXED:
                helper(batch)
            assert batch.nbytes_by_array == sizes
        eager = ej.encode(encoder, words)
        assert_metadata([helper(eager) for helper in HELPERS], expected(reference_encoder.encode(rows)))
    finally:
        ej.unregister(encoder)
        jax.clear_caches()
