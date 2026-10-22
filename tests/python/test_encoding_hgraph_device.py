"""The HGraph view of a relation-major device batch (``encoding.hgraph`` of a DeviceFlatBatch): it equals the host view
of the same rows in every field, its edge rows, `_symbol_` ptr and entity roles are the batch's own device memory
(checked by pointer identity and by the allocator's counters), its methods never wait for the device, it keeps the batch
alive and refuses what the host view refuses.

The tests run only when a GPU is made visible explicitly (conftest.py hides GPUs by default), e.g.

    env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        .venv-rl/bin/python -m pytest tests/python/test_encoding_hgraph_device.py
"""

import gc
import warnings

import numpy as np
import pytest

from mymyr import encoding
from test_encoding_cuda import (  # noqa: F401 (cases is a fixture)
    cases,
    device_encode,
    exact_capacities,
    gpu,
    host_encode,
    layered_kwargs,
    sliced,
    to_device,
)

torch = pytest.importorskip("torch")

SYMBOL = "_symbol_"
ALL_GOALS = ("plain", "satisfied", "unsatisfied")
# the HGraph configs the device encodes (C9A, C9B; C9C needs action relations, which the device does not encode) and
# the lanes that the device encodes besides them
OPTION_SETS = {
    "C9A": {"nullary": False, "include_actions": False},
    "C9B": {"nullary": True, "include_actions": False},
    "goal_targets": {"goal_derivations": ALL_GOALS, "target_sources": ("goal",)},
    "implicit_entities": {"include_actions": False, "entity_arrays": "auto", "goal_derivations": (), "emit_counts": False},
    "no_static_equality": {"goal_derivations": ("plain",), "include_static": False, "equality": True},
}
LAYERED_OPTION_SETS = {
    "levels_targets": {"max_goal_level": 3, "goal_derivations": ALL_GOALS, "target_sources": ("goal", "subgoal")},
    "history_targets": {"history": True, "history_max_steps": 2, "target_sources": ("history",)},
}


def same_array(host, device):
    """A host array of the host view and a tensor of the device view: the same dtype, shape and values."""
    assert isinstance(device, torch.Tensor) and device.is_cuda
    a, b = np.asarray(host), device.cpu().numpy()
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(a, b)


def expect_same_view(host, device):
    """The device view has the types, counts and arrays of the host view, every field."""
    assert device.num_graphs == host.num_graphs
    assert device.node_types == host.node_types
    assert device.edge_types == host.edge_types
    assert device.reverse_edge_types == host.reverse_edge_types
    for node_type in host.node_types:
        assert device.num_nodes(node_type) == host.num_nodes(node_type), node_type
        same_array(host.ptr(node_type), device.ptr(node_type))
    for edge_type in host.edge_types + host.reverse_edge_types:
        assert device.num_edges(edge_type) == host.num_edges(edge_type), edge_type
        same_array(host.edge_index_0(edge_type), device.edge_index_0(edge_type))
        same_array(host.edge_index_1(edge_type), device.edge_index_1(edge_type))
        for a, b in zip(host.edge_index(edge_type), device.edge_index(edge_type), strict=True):
            same_array(a, b)
    same_array(host.entity_role, device.entity_role)


def expect_zero_copy(view, batch):
    """The edge rows of the view point into the batch's relation_args with the relation's arity as stride, the
    `_symbol_` ptr is the batch's entity_offsets and the stored roles are the batch's entity_role; handing them out
    allocates no device memory, and no method of the view waits for the device."""
    arrays = batch.as_dict("torch")
    graphs, args = batch.num_graphs, arrays["relation_args"]
    schema = batch.schema
    rows = []
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    for r, (name, arity) in enumerate(zip(schema.relation_names, schema.relation_arities, strict=True)):
        first = int(arrays["relation_arg_offsets"][r * graphs])
        count = view.num_nodes(name)
        for k in range(arity):
            row = view.edge_index_1((name, str(k), SYMBOL))
            back = view.edge_index_0((SYMBOL, str(k), name))
            rows += [row, back]
            assert row.shape == (count,) and row.stride() == (arity,) and row.dtype == args.dtype
            assert back.data_ptr() == row.data_ptr() and back.stride() == row.stride()
            if count:
                assert row.untyped_storage().data_ptr() == args.untyped_storage().data_ptr()
                assert row.data_ptr() == args.data_ptr() + (first + k) * args.element_size()
    symbol = view.ptr(SYMBOL)
    assert symbol.data_ptr() == arrays["entity_offsets"].data_ptr() and symbol.shape == (graphs + 1,)
    if not batch.implicit_entities and view.num_nodes(SYMBOL):
        assert view.entity_role.data_ptr() == batch.entity_role.data_ptr()
    assert torch.cuda.memory_allocated() == allocated  # nothing was copied
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch calls the sync debug mode a prototype
        torch.cuda.set_sync_debug_mode("error")
        try:
            for edge_type in view.edge_types + view.reverse_edge_types:
                view.edge_index(edge_type)
            for node_type in view.node_types:
                view.ptr(node_type)
            view.entity_role
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return rows


def encoders_of(case, **options):
    return encoding.FlatRelationEncoder(case.source, layout="relation", **options)


# ------------------------------------------------------------------------------------------------ device == host


@pytest.mark.parametrize("options", OPTION_SETS.values(), ids=OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("n", [None, 0, 1], ids=["batch", "empty", "single"])
def test_the_device_view_equals_the_host_view(cases, index, options, n):
    dev = gpu()
    for case in cases.values():
        case = case if n is None else sliced(case, n)
        encoder = encoders_of(case, index_dtype=index, **options)
        host, device = host_encode(encoder, case), device_encode(encoder, case, dev)
        view = encoding.hgraph(device)
        assert type(view) is encoding.DeviceHGraphView and view.batch is device and view.device == dev
        expect_same_view(encoding.hgraph(host), view)
        expect_zero_copy(view, device)


@pytest.mark.parametrize("options", LAYERED_OPTION_SETS.values(), ids=LAYERED_OPTION_SETS.keys())
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_the_device_view_of_subgoal_layers_and_history_equals_the_host_view(cases, index, options):
    dev = gpu()
    for case in cases.values():
        encoder = encoders_of(case, index_dtype=index, **options)
        kwargs = layered_kwargs(case, options)
        host = host_encode(encoder, case, **kwargs)
        device = device_encode(encoder, case, dev, **kwargs)
        assert device.num_graphs and int(host.relation_arg_offsets[-1])
        view = encoding.hgraph(device)
        expect_same_view(encoding.hgraph(host), view)
        expect_zero_copy(view, device)


def test_the_tables_of_several_instances_mix_their_graphs_in_one_view(cases):
    if "blocks_table" not in cases:
        pytest.skip("built without the loki front end")
    dev = gpu()
    for name in ("blocks_table", "philosophers_table"):
        case = cases[name]
        assert len(set(case.ids.tolist())) > 1
        for index in ("int32", "int64"):
            encoder = encoders_of(case, index_dtype=index, **OPTION_SETS["C9B"])
            device = device_encode(encoder, case, dev)
            view = encoding.hgraph(device)
            expect_same_view(encoding.hgraph(host_encode(encoder, case)), view)
            # the entities of the graphs of both instances are the `_symbol_` nodes, numbered by the batch
            objects = device.num_objects.cpu().numpy()
            if name == "blocks_table":
                assert len(set(objects.tolist())) > 1  # instances of different sizes
            assert view.num_nodes(SYMBOL) == int(device.entity_offsets[-1]) >= int(objects.sum())


# ------------------------------------------------------------------------------------------------ batch forms


@pytest.mark.parametrize("index", ["int32", "int64"])
def test_the_capacity_form_is_viewed_up_to_its_sizes(cases, index):
    dev = gpu()
    case = cases["philosophers"]
    encoder = encoders_of(case, index_dtype=index, **OPTION_SETS["goal_targets"])
    host = host_encode(encoder, case)
    device = device_encode(encoder, case, dev, capacities=exact_capacities(host, factor=2))
    assert device.relation_args.shape[0] > int(host.relation_arg_offsets[-1])
    view = encoding.hgraph(device)
    expect_same_view(encoding.hgraph(host), view)
    expect_zero_copy(view, device)
    short = device_encode(encoder, case, dev, capacities={**exact_capacities(host), "args": 1})
    assert short.sizes["overflow"]
    with pytest.raises(ValueError, match="^mymyr: encoding: the batch overflowed its capacities"):
        encoding.hgraph(short)


@pytest.mark.parametrize("index", ["int32", "int64"])
def test_stream_and_collated_batches_are_viewed_like_one_encoding(cases, index):
    dev = gpu()
    case = cases["philosophers_table"] if "philosophers_table" in cases else cases["philosophers"]
    n = min(33, case.n)
    case = sliced(case, n)
    encoder = encoders_of(case, index_dtype=index, **OPTION_SETS["goal_targets"])
    expected = encoding.hgraph(host_encode(encoder, case))
    rows = to_device(case.rows, dev)
    ids = None if case.ids is None else to_device(case.ids, dev)
    pieces = [(0, 3), (3, 3), (3, 11), (11, n)]
    collated = encoding.collate([encoder.encode(rows[a:b], None if ids is None else ids[a:b]) for a, b in pieces])
    view = encoding.hgraph(collated)
    expect_same_view(expected, view)
    expect_zero_copy(view, collated)
    ring = encoder.device_stream(2 * n)
    ring.append(rows[:11], None if ids is None else ids[:11])
    ring.append(rows[11:], None if ids is None else ids[11:])
    flushed = ring.flush(capacities=exact_capacities(host_encode(encoder, case), factor=2) | {"graphs": 2 * n})
    assert flushed.task_ids.shape[0] == 2 * n
    view = encoding.hgraph(flushed)
    assert view.num_graphs == n
    expect_same_view(expected, view)
    expect_zero_copy(view, flushed)


# ------------------------------------------------------------------------------------------------ lifetime


def test_the_view_keeps_the_batch_alive(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoders_of(case, include_actions=False)
    batch = device_encode(encoder, case, dev)
    view = encoding.hgraph(batch)
    expected = {t: view.edge_index_1(t).cpu().numpy() for t in view.edge_types}
    roles = view.entity_role.cpu().numpy()
    row = view.edge_index_1(view.edge_types[0])
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    # the batch is reachable only through the view, which holds it and its arrays
    filler = torch.full((1 << 20,), -1, dtype=torch.int64, device=dev)
    for t, values in expected.items():
        assert np.array_equal(view.edge_index_1(t).cpu().numpy(), values)
    assert np.array_equal(view.entity_role.cpu().numpy(), roles)
    assert view.batch.num_graphs == case.n
    del view, filler
    gc.collect()
    assert np.array_equal(row.cpu().numpy(), expected[next(iter(expected))])  # the rows outlive the view


# ------------------------------------------------------------------------------------------------ the API and refusals


def test_frameworks_name_types_and_reject_what_the_host_view_rejects(cases):
    dev = gpu()
    case = cases["gripper"]
    encoder = encoders_of(case, include_actions=False)
    batch = device_encode(encoder, case, dev)
    view = encoding.hgraph(batch)
    host = encoding.hgraph(host_encode(encoder, case))
    edge_type = view.edge_types[0]
    for framework in ("torch", "pytorch", "dlpack", "array_api"):
        assert view.edge_index_1(edge_type, framework=framework).data_ptr() == view.edge_index_1(edge_type).data_ptr()
    for framework in ("numpy", "np"):
        array = view.edge_index_1(edge_type, framework=framework)
        assert isinstance(array, np.ndarray) and np.array_equal(array, host.edge_index_1(edge_type))
        assert np.array_equal(view.ptr(edge_type[0], framework=framework), host.ptr(edge_type[0]))
    with pytest.raises(ValueError, match="exports to 'torch', 'dlpack' or 'numpy', not 'jax'"):
        view.edge_index(edge_type, framework="jax")
    with pytest.raises(KeyError, match="no node type nothing"):
        view.num_nodes("nothing")
    with pytest.raises(TypeError, match="a node type is a name"):
        view.ptr(3)
    first = edge_type[0]
    arity = encoder.relation_arities[encoder.relation_id(first)]
    for bad in ((first, str(arity), SYMBOL), (first, "-1", SYMBOL), (first, "x", SYMBOL), (first, "0", first),
                (SYMBOL, "0", SYMBOL), ("nothing", "0", SYMBOL), (first, "00", SYMBOL), (first, "", SYMBOL),
                (first, "١", SYMBOL)):
        with pytest.raises(KeyError, match="no edge type"):
            view.edge_index(bad)
        with pytest.raises(KeyError, match="no edge type"):
            host.edge_index(bad)
    for bad in (first, (first, "0"), (first, 0, SYMBOL), [first, "0", SYMBOL], None):
        with pytest.raises(TypeError, match="an edge type is a tuple"):
            view.edge_index(bad)


def test_the_device_view_refuses_what_the_host_view_refuses(cases):
    dev = gpu()
    case = cases["gripper"]
    graph_major = encoding.FlatRelationEncoder(case.source, include_actions=False)
    with pytest.raises(ValueError, match="^mymyr: encoding: the HGraph view needs a relation-major batch.*graph-major"):
        encoding.hgraph(device_encode(graph_major, case, dev))
    virtual = encoders_of(case, include_actions=False, predicate_virtual_nodes=True)
    with pytest.raises(ValueError, match="^mymyr: encoding: the HGraph view does not support predicate virtual nodes"):
        encoding.hgraph(device_encode(virtual, case, dev))
    # the device does not encode shared statics; a batch claiming the schema of such an encoder is refused all the same
    batch = device_encode(encoders_of(case, include_actions=False), case, dev)
    batch._schema = encoders_of(case, include_actions=False, statics="shared").schema
    with pytest.raises(ValueError, match="^mymyr: encoding: statics='shared' is not supported by the HGraph encoding"):
        encoding.hgraph(batch)
    with pytest.raises(TypeError, match="^mymyr: encoding: a DeviceHGraphView is of a DeviceFlatBatch"):
        encoding.DeviceHGraphView(host_encode(encoders_of(case, include_actions=False), case))
