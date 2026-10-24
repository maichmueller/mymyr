"""Deep CUDA horizon arrays match host batches, including graph structure, targets and invalid-input flags."""

from dataclasses import replace

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT, text_task, walk
from test_encoding_device_actions import caps, gpu, same, tensor, upload

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def mixed():
    gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(base / "domain.pddl", base / f"{name}.pddl")
             for name in ("smedium", "probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    ids = np.array([1, 0, 1], np.int32)
    rows = np.zeros((len(ids), table.words), np.uint64)
    for g, i in enumerate(ids):
        state = walk(tasks[i], steps=g + 1, walks=1)[-1]
        row = np.asarray(tasks[i].encode([state]))
        rows[g, :row.shape[1]] = row[0]
    return table, {depth: encoding.horizon(table, rows, depth=depth, task_ids=ids) for depth in (2, 3)}


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("root", ["include", "exclude", "encode_only"])
def test_deep_mixed_horizons_all_modes_roots_layouts_and_widths(mixed, depth, layout, index, mode, root):
    table, views = mixed
    goals = ("plain", "satisfied", "added_satisfied", "added_unsatisfied") if mode == "delta" else (
        "plain", "satisfied", "unsatisfied")
    encoder = encoding.FlatHorizonEncoder(table, horizon_mode=mode, root_policy=root, layout=layout,
        index_dtype=index, parent_relation=True, sibling_relation=True, cousin_relation=True,
        predicate_virtual_nodes=True, goal_derivations=goals)
    host = encoder.encode(views[depth])
    view = upload(views[depth], gpu())
    same(host, encoder.encode(view))
    same(host, encoder.encode(view, capacities=caps(host)))
    small = {**caps(host), "args": 0}
    batch = encoder.encode(view, capacities=small)
    assert batch.sizes["capacity_overflow"] and not batch.sizes["invalid_input"]
    assert all(torch.count_nonzero(a).item() == 0 for a in batch._arrays.values())


ADDED = ("added_satisfied", "added_unsatisfied")
CORPUS_OPTIONS = [
    dict(horizon_mode="full"),
    dict(horizon_mode="delta", root_policy="exclude", goal_derivations=ADDED, include_actions=False),
    dict(horizon_mode="delta", root_policy="encode_only", goal_derivations=ADDED, parent_relation=True),
    dict(horizon_mode="action", root_policy="exclude"),
    dict(horizon_mode="full", root_policy="exclude"),
    dict(horizon_mode="full", root_policy="encode_only"),
    dict(horizon_mode="delta", goal_derivations=ADDED, include_actions=False),
    dict(horizon_mode="action"),
    dict(horizon_mode="action", root_policy="encode_only"),
    dict(horizon_mode="delta", root_policy="exclude", goal_derivations=ADDED, include_actions=False, max_goal_level=2),
    dict(horizon_mode="full", max_goal_level=1),
    dict(horizon_mode="full", root_policy="encode_only", parent_relation=True),
    dict(horizon_mode="full", root_policy="encode_only", parent_relation=True, sibling_relation=True, cousin_relation=True),
]


@pytest.mark.parametrize("options", CORPUS_OPTIONS)
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "philosophers__p03-phil4"])
def test_deep_horizon_corpus_configurations(options, depth, layout, name):
    device = gpu()
    task = text_task(name)
    rows = np.asarray(task.encode(walk(task, steps=1, walks=1)))
    view = encoding.horizon(task, rows, depth=depth)
    levels = options.get("max_goal_level", 0)
    sub = None if not levels else (np.repeat(rows[:, None, :], levels, axis=1),
                                   np.zeros((len(rows), levels, rows.shape[1]), np.uint64))
    encoder = encoding.FlatHorizonEncoder(task, layout=layout, index_dtype="int64", **options)
    expected = encoder.encode(view, subgoals=sub)
    same(expected, encoder.encode(upload(view, device), subgoals=None if sub is None else tuple(tensor(x, device) for x in sub)))


@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_deep_supplied_derived_words_and_per_graph_goals(mode, layout):
    from test_encoding import derived_words

    device = gpu()
    task = text_task("philosophers__p03-phil4")
    rows = np.asarray(task.encode(walk(task, steps=2, walks=1)))
    view = encoding.horizon(task, rows, depth=3)
    derived = derived_words(task, task.decode(view.nodes))
    empty = np.zeros_like(rows)
    layers = np.repeat(rows[:, None, :], 2, axis=1)
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode=mode, root_policy="exclude", layout=layout,
        max_goal_level=2, goal_derivations=ADDED if mode == "delta" else ("plain", "satisfied", "unsatisfied"),
        parent_relation=True, sibling_relation=True, cousin_relation=True)
    expected = encoder.encode(view, derived=derived, goals=(rows, empty), subgoals=(layers, np.zeros_like(layers)))
    inputs = upload(view, device)
    kwargs = dict(goals=(tensor(rows, device), tensor(empty, device)),
                  subgoals=(tensor(layers, device), tensor(np.zeros_like(layers), device)))
    same(expected, encoder.encode(inputs, **kwargs))
    same(expected, encoder.encode(inputs, derived=tensor(derived, device), capacities=caps(expected), **kwargs))


@pytest.mark.parametrize("root", ["include", "exclude", "encode_only"])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_deep_dag_cycles_parallel_edges_and_nonempty_cousins(root, layout):
    device = gpu()
    task = text_task("blocks__probBLOCKS-8-0")
    rows = np.asarray(task.encode([task.initial_state] * 7))
    expansion = rl.expand(task, rows[:1])
    src = np.array([0, 0, 1, 2, 1, 4, 5, 6, 3, 2, 1], np.int32)
    dst = np.array([1, 2, 3, 4, 5, 6, 3, 0, 3, 5, 3], np.int32)
    count = len(src)
    schema = np.repeat(np.asarray(expansion.schema)[:1], count)
    binding = np.repeat(np.asarray(expansion.binding)[:1], count, axis=0)
    view = encoding.HorizonView(rows, np.array([0, 7], np.int32), np.array([0, count], np.int32),
                                src, dst, schema, binding)
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode="action", root_policy=root, layout=layout,
        parent_relation=True, sibling_relation=True, cousin_relation=True)
    expected = encoder.encode(view)
    cousin = encoder.relation_names.index("_cousin_")
    if root != "exclude":
        assert expected.relation_counts[0, cousin] > 0
    same(expected, encoder.encode(upload(view, device)))
    order = np.arange(count - 1, -1, -1).repeat(2)
    shuffled = replace(view, edge_offsets=np.array([0, len(order)], np.int32), edge_src=src[order],
                       edge_dst=dst[order], edge_schema=schema[order], edge_binding=binding[order])
    same(expected, encoder.encode(upload(shuffled, device)))


@pytest.mark.parametrize("field", ["node_start", "node_end", "node_empty", "edge_start", "edge_end", "edge_reverse",
                                   "src_negative", "src_large", "dst_negative", "dst_large", "schema", "binding", "task_ids"])
def test_deep_view_invalid_offsets_and_indices_are_flagged_before_fill(mixed, field):
    table, views = mixed
    device = gpu()
    view = upload(views[2], device)
    encoder = encoding.FlatHorizonEncoder(table, horizon_mode="full", parent_relation=True, cousin_relation=True)
    capacity = caps(encoder.encode(views[2]))
    if field.startswith("node_"):
        x = view.node_offsets.clone()
        x[0 if field == "node_start" else 1] = 1 if field == "node_start" else (
            view.nodes.shape[0] + 1 if field == "node_end" else 0)
        bad = replace(view, node_offsets=x)
    elif field.startswith("edge_"):
        x = view.edge_offsets.clone()
        x[0 if field == "edge_start" else 1] = 1 if field == "edge_start" else (
            view.edge_src.numel() + 1 if field == "edge_end" else -1)
        bad = replace(view, edge_offsets=x)
    else:
        name = "edge_" + field.split("_")[0] if field.startswith(("src_", "dst_")) else (
            "edge_" + field if field != "task_ids" else field)
        x = getattr(view, name).clone()
        x.reshape(-1)[0] = -1 if field.endswith("negative") else 0x7fffffff
        bad = replace(view, **{name: x})
    result = encoder.encode(bad, capacities=capacity)
    assert result.sizes["invalid_input"]
    assert all(torch.count_nonzero(a).item() == 0 for a in result._arrays.values())
    with pytest.raises(ValueError, match="invalid"):
        encoder.encode(bad)


@pytest.fixture(scope="module")
def space():
    gpu()
    cuda = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    base = ROOT / "tests/data/benchmarks/small/blocks_3"
    task = mymyr.Task.from_pddl(base / "domain.pddl", base / "test_problem.pddl", atoms="dense")
    context = cuda.Context(0, max_bytes=256 << 20)
    return cuda.state_space(task, ctx=context, labels=True, remove_if_unsolvable=False, chunk_states=4)


@pytest.mark.parametrize("depth", [0, 2, 3])
@pytest.mark.parametrize("roots", [None, [0, 1, 0], []])
def test_device_state_space_traversal_matches_host_arrays_and_encoding(space, depth, roots):
    expected = encoding.horizon_of_space(space.to_host(), roots, depth=depth)
    actual = encoding.horizon_of_space(space, roots, depth=depth)
    for name in ("node_offsets", "edge_offsets"):
        np.testing.assert_array_equal(getattr(actual, name).cpu().numpy(), getattr(expected, name))
    nodes, edges = int(expected.node_offsets[-1]), int(expected.edge_offsets[-1])
    for name in ("nodes", "edge_src", "edge_dst", "edge_schema", "edge_binding"):
        end = nodes if name == "nodes" else edges
        result = getattr(actual, name)[:end].cpu().numpy()
        reference = np.asarray(getattr(expected, name))
        if name == "nodes":
            result = result.view(np.uint64)
        np.testing.assert_array_equal(result, reference, err_msg=name)
    encoder = encoding.FlatRootedHorizonEncoder(space.task, sibling_relation=True, cousin_relation=True)
    same(encoder.encode(expected), encoder.encode(actual))


def test_device_space_node_limits_and_invalid_roots(space):
    expected = encoding.horizon_of_space(space.to_host(), [0, 1], depth=3, max_nodes=2, truncate=True)
    actual = encoding.horizon_of_space(space, tensor(np.array([0, 1], np.int32), gpu()), depth=3,
                                        max_nodes=2, truncate=True)
    np.testing.assert_array_equal(actual.truncated.cpu().numpy(), expected.truncated)
    encoder = encoding.FlatHorizonEncoder(space.task, horizon_mode="full", parent_relation=True)
    same(encoder.encode(expected), encoder.encode(actual))
    for roots, kwargs in (([-1], {}), ([0], dict(max_nodes=1)), ([space.num_states], {})):
        view = encoding.horizon_of_space(space, tensor(np.array(roots, np.int32), gpu()), depth=3, **kwargs)
        with pytest.raises(ValueError, match="invalid"):
            encoder.encode(view)
    with pytest.raises(ValueError, match="merge=True"):
        encoding.horizon_of_space(space, [0], depth=2, merge=False)


def test_device_spaces_without_labels_or_with_numeric_rows_are_refused(space):
    import mymyr.cuda as cuda

    context = cuda.Context(0, max_bytes=256 << 20)
    numeric = cuda.state_space(mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt")), ctx=context,
                               labels=True, remove_if_unsolvable=False)
    with pytest.raises(ValueError, match="^mymyr: encoding: horizon encodings do not encode numeric state rows"):
        encoding.horizon_of_space(numeric, [0], depth=2)
    unlabelled = cuda.state_space(space.task, ctx=context, labels=False, remove_if_unsolvable=False)
    with pytest.raises(ValueError, match="needs transition labels"):
        encoding.horizon_of_space(unlabelled, [0], depth=2)


def test_device_space_builder_releases_completed_calls(space):
    import gc
    import weakref

    view = encoding.horizon_of_space(space, [0], depth=3)
    nodes = weakref.ref(view.nodes)
    del view
    torch.cuda.synchronize()
    gc.collect()
    following = encoding.horizon_of_space(space, [0], depth=3)
    gc.collect()
    assert nodes() is None
    assert following.nodes.is_cuda


@pytest.mark.needs_gil
@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_deep_torch_op_compiles_and_replays_with_cousins(layout, index):
    import mymyr.rl.torch as rt

    device = gpu()
    task = text_task("blocks__probBLOCKS-8-0")
    rows = np.asarray(task.encode([task.initial_state]))
    view = encoding.horizon(task, rows, depth=3, merge=False)
    inputs = upload(view, device)
    encoder = encoding.FlatRootedHorizonEncoder(task, layout=layout, index_dtype=index,
                                               sibling_relation=True, cousin_relation=True)
    expected = encoder.encode(view)
    capacity = caps(expected)
    graphs = expected.num_graphs
    handle = rt.register(encoder)

    def call():
        return torch.ops.mymyr.encode_flat(handle, inputs.nodes, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            graphs, capacity["entities"], capacity["args"], capacity["targets"], 0,
            node_offsets=inputs.node_offsets, edge_offsets=inputs.edge_offsets, edge_src=inputs.edge_src,
            edge_dst=inputs.edge_dst, edge_schema=inputs.edge_schema, edge_binding=inputs.edge_binding)

    try:
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, call()))
        compiled = torch.compile(call, fullgraph=True, backend="aot_eager")
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, compiled()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call()
        graph.replay()
        graph.replay()
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, output))
    finally:
        rt.unregister(handle)
