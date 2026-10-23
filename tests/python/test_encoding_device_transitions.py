"""Device transition encodes with capacities, the depth-one builder, and expansions captured into CUDA graphs."""

from types import SimpleNamespace

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from mymyr.encoding import _device_horizon
from conftest import ROOT, text_task, walk
from test_encoding_device_actions import caps, gpu, same, tensor

torch = pytest.importorskip("torch")

# about 100 ms of device time at the clocks of current GPUs: far longer than the host side of an encode
SLEEP_CYCLES = 200_000_000


def blocks_table():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    p = ROOT / "tests/data/pddl/blocks"
    tasks = [mymyr.Task.from_pddl(p / "domain.pddl", p / f"{name}.pddl") for name in ("smedium", "probBLOCKS-8-0")]
    table = rl.TaskTable(tasks)
    rows = np.zeros((4, table.words), np.uint64)
    ids = np.array([0, 1, 1, 0], np.int32)
    for g, i in enumerate(ids):
        raw = np.asarray(tasks[i].encode([tasks[i].initial_state]))
        rows[g, :raw.shape[1]] = raw[0]
    return table, rows, ids


def returns_before_the_device(call):
    """Runs call() behind a long device sleep: whether it returned before the sleep ended, and its result."""
    torch.cuda.synchronize()
    torch.cuda._sleep(SLEEP_CYCLES)
    sentinel = torch.cuda.Event()
    sentinel.record()
    result = call()
    early = not sentinel.query()
    torch.cuda.synchronize()
    return early, result


@pytest.mark.parametrize("entry", ["transitions", "effects", "environment"])
def test_capacity_transition_encodes_return_before_the_device(entry):
    device = gpu()
    task = text_task("gripper__prob05")
    if entry == "environment":
        import mymyr.rl.torch as rt
        env = rt.BatchedEnv(task, 6, device=device)
        encoder = encoding.FlatTransitionEffectsEncoder(task, parent_relation=True)
        expected = encoder.encode(env.states.cpu(), env.task_ids.cpu())
        capacity = caps(expected)
        call = lambda: env.encode_transitions(encoder, capacities=capacity)
    else:
        cls = encoding.FlatTransitionEncoder if entry == "transitions" else encoding.FlatTransitionEffectsEncoder
        rows = np.asarray(task.encode(walk(task, steps=5, walks=1)))
        words = tensor(rows, device)
        encoder = cls(task, parent_relation=True)
        expected = encoder.encode(rows)
        capacity = caps(expected)
        call = lambda: encoder.encode(words, capacities=capacity)
    for _ in range(2):  # the expander, the encoder's context and torch's allocator
        call()
    early, batch = returns_before_the_device(call)
    same(expected, batch)
    assert batch.sizes["overflow"] is False
    assert early, "the encode waited for the device"


@pytest.mark.parametrize("cls", [encoding.FlatTransitionEncoder, encoding.FlatTransitionEffectsEncoder])
def test_capacity_transitions_of_mixed_instances_and_derived_words_equal_the_host(cls):
    device = gpu()
    table, rows, ids = blocks_table()
    encoder = cls(table, parent_relation=True)
    expected = encoder.encode(rows, ids)
    same(expected, encoder.encode(tensor(rows, device), tensor(ids, device), capacities=caps(expected)))
    from test_encoding import derived_words
    task = text_task("philosophers__p03-phil4")
    states = walk(task, steps=2, walks=1)
    rows = np.asarray(task.encode(states))
    derived = derived_words(task, states)
    encoder = cls(task)
    expected = encoder.encode(rows, derived=derived, actions=rl.expand(task, rows, derived=True))
    actual = encoder.encode(tensor(rows, device), derived=tensor(derived, device), capacities=caps(expected))
    same(expected, actual)


def test_an_expansion_past_the_entity_capacity_reports_a_capacity_overflow():
    device = gpu()
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode(walk(task, steps=3, walks=1)))
    successors = int(np.asarray(rl.expand(task, rows).offsets)[-1])
    encoder = encoding.FlatTransitionEffectsEncoder(task)
    expected = encoder.encode(rows)
    capacity = dict(caps(expected), entities=successors - 1)
    batch = encoder.encode(tensor(rows, device), capacities=capacity)
    sizes = batch.sizes
    assert sizes["overflow"] and sizes["capacity_overflow"]
    assert not sizes["invalid_input"] and not sizes["index_overflow"]
    assert not batch.entity_offsets.any()


def test_expand_into_with_status_returns_before_the_device_and_flags_overflow_and_invalid_rows():
    device = gpu()
    table, rows, ids = blocks_table()
    host = rl.expand(table, rows, ids)
    total = int(np.asarray(host.offsets)[-1])
    words, device_ids = tensor(rows, device), tensor(ids, device)

    def into(capacity, task_ids=device_ids):
        out = dict(succ=torch.zeros((capacity, rows.shape[1]), dtype=torch.int64, device=device),
                   parent=torch.full((capacity,), -1, dtype=torch.int32, device=device),
                   schema=torch.full((capacity,), -1, dtype=torch.int32, device=device),
                   binding=torch.full((capacity, table.label_width), -1, dtype=torch.int32, device=device),
                   offsets=torch.empty(len(rows) + 1, dtype=torch.int32, device=device),
                   status=torch.full((1,), -1, dtype=torch.int32, device=device))
        result = rl.expand_into(table, words, task_ids, **out, stream=torch.cuda.current_stream(device))
        return result, out

    into(total)
    early, (result, out) = returns_before_the_device(lambda: into(total))
    assert early and "total" not in result
    assert out["status"].item() == 0
    np.testing.assert_array_equal(out["offsets"].cpu().numpy(), np.asarray(host.offsets))
    np.testing.assert_array_equal(out["succ"].cpu().numpy().view(np.uint64), np.asarray(host.succ))
    for name in ("parent", "schema", "binding"):
        np.testing.assert_array_equal(out[name].cpu().numpy(), np.asarray(getattr(host, name)), err_msg=name)
    result, out = into(total - 1)
    torch.cuda.synchronize()
    assert out["status"].item() == 1
    np.testing.assert_array_equal(out["offsets"].cpu().numpy(), np.asarray(host.offsets))
    bad = device_ids.clone()
    bad[2] = 7
    result, out = into(total, bad)
    assert out["status"].item() == 2
    assert not out["succ"].any() and (out["schema"] == -1).all()
    # nothing waits for the outputs: the stream that reads them is named
    status = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="an expansion with 'status' does not wait for the device: pass stream="):
        rl.expand_into(table, words, device_ids, succ=out["succ"], offsets=out["offsets"], status=status)


@pytest.mark.parametrize("kind", ["actions", "transitions"])
def test_an_expansion_warmed_on_the_capture_stream_captures_and_replays(kind):
    device = gpu()
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode(walk(task, steps=2, walks=1)))
    words = tensor(rows, device)
    expansion = rl.expand(task, words, framework="torch")  # on the default stream
    if kind == "actions":
        encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    else:
        encoder = encoding.FlatTransitionEffectsEncoder(task, parent_relation=True)
    expected = encoder.encode(rows, actions=rl.expand(task, rows))
    capacity = caps(expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        encoder.encode(words, actions=expansion, capacities=capacity)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        batch = encoder.encode(words, actions=expansion, capacities=capacity)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        same(expected, batch)


# a refused encode launches nothing, so torch warns that the captured graph is empty
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_capture_refuses_cold_expansions_and_automatic_successors():
    device = gpu()
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode([task.initial_state]))
    words = tensor(rows, device)
    encoder = encoding.FlatTransitionEffectsEncoder(task)
    expected = encoder.encode(rows)
    capacity = caps(expected)
    cold = rl.expand(task, words, framework="torch")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        encoder.encode(words, capacities=capacity)
    stream.synchronize()
    for actions, message in [(cold, "before the CUDA graph capture"), (None, "outside CUDA graph capture")]:
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(ValueError, match=message):
            with torch.cuda.graph(graph, stream=stream):
                encoder.encode(words, actions=actions, capacities=capacity)
        del graph


def reference_build(parents, children, offsets, merge):
    """The host builder's depth-one view of explicit successor rows: first-seen nodes, edges stably by destination."""
    nodes, node_offsets, dst = [], [0], np.zeros(len(children), np.int64)
    order = np.arange(len(children))
    for g in range(len(parents)):
        first, last = offsets[g], offsets[g + 1]
        seen = [tuple(parents[g])]
        for e in range(first, last):
            row = tuple(children[e])
            if merge and row in seen:
                dst[e] = seen.index(row)
            else:
                dst[e] = len(seen)
                seen.append(row)
        order[first:last] = first + np.argsort(dst[first:last], kind="stable")
        nodes.extend(seen)
        node_offsets.append(len(nodes))
    return np.array(nodes, np.uint64), np.array(node_offsets), dst[order], order


@pytest.mark.parametrize("merge", [False, True])
def test_builder_merges_repeated_and_root_equal_successors_like_the_host(merge):
    device = gpu()
    task = text_task("gripper__prob05")
    encoder = encoding.FlatTransitionEffectsEncoder(task)
    rng = np.random.default_rng(7)
    words = 3
    fanouts = [0, 1, 5, 300, 64, 0, 17]
    parents = rng.integers(0, 4, (len(fanouts), words)).astype(np.uint64)
    offsets = np.concatenate([[0], np.cumsum(fanouts)]).astype(np.int32)
    children = rng.integers(0, 4, (offsets[-1], words)).astype(np.uint64)
    parent = np.repeat(np.arange(len(fanouts)), fanouts).astype(np.int32)
    for e in range(0, len(children), 9):  # successors equal to their root
        children[e] = parents[parent[e]]
    schema = rng.integers(0, 3, len(children)).astype(np.int32)
    binding = rng.integers(-1, 6, (len(children), 4)).astype(np.int32)
    actions = SimpleNamespace(succ=tensor(children, device), schema=tensor(schema, device),
                              binding=tensor(binding, device), offsets=tensor(offsets, device),
                              parent=tensor(parent, device))
    view = _device_horizon.build_horizon(encoder, tensor(parents, device), None, actions, merge=merge)
    nodes, node_offsets, dst, order = reference_build(parents, children, offsets, merge)
    np.testing.assert_array_equal(view.node_offsets.cpu().numpy(), node_offsets)
    np.testing.assert_array_equal(view.nodes[:node_offsets[-1]].cpu().numpy().view(np.uint64), nodes)
    np.testing.assert_array_equal(view.edge_offsets.cpu().numpy(), offsets)
    np.testing.assert_array_equal(view.edge_dst.cpu().numpy(), dst)
    np.testing.assert_array_equal(view.edge_src.cpu().numpy(), np.zeros(len(children)))
    np.testing.assert_array_equal(view.edge_schema.cpu().numpy(), schema[order])
    np.testing.assert_array_equal(view.edge_binding.cpu().numpy(), binding[order])
    bad = offsets.copy()
    bad[4] = bad[5] + 1  # graph 3 ends before it starts
    actions.offsets = tensor(bad, device)
    view = _device_horizon.build_horizon(encoder, tensor(parents, device), None, actions, merge=merge)
    np.testing.assert_array_equal(view.node_offsets.cpu().numpy()[:4], node_offsets[:4])
    assert (view.node_offsets.cpu().numpy()[4:] == -1).all()


@pytest.mark.parametrize("merge", [False, True])
def test_builder_keeps_the_graphs_before_offsets_that_fall_back(merge):
    device = gpu()
    task = text_task("gripper__prob05")
    encoder = encoding.FlatTransitionEffectsEncoder(task)
    rng = np.random.default_rng(3)
    parents = rng.integers(0, 1 << 40, (8, 2)).astype(np.uint64)
    children = rng.integers(0, 1 << 40, (8, 2)).astype(np.uint64)
    children[3] = children[2]
    offsets = np.array([0, 2, 4, 6, 1, 1, 1, 1, 8], np.int32)  # graph 3 ends before it starts, the rest fall back
    parent = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    schema = rng.integers(0, 3, 8).astype(np.int32)
    binding = rng.integers(-1, 6, (8, 4)).astype(np.int32)
    actions = SimpleNamespace(succ=tensor(children, device), schema=tensor(schema, device),
                              binding=tensor(binding, device), offsets=tensor(offsets, device),
                              parent=tensor(parent, device))
    view = _device_horizon.build_horizon(encoder, tensor(parents, device), None, actions, merge=merge)
    nodes, node_offsets, dst, order = reference_build(parents[:3], children[:6], offsets[:4], merge)
    np.testing.assert_array_equal(view.node_offsets.cpu().numpy()[:4], node_offsets)
    assert (view.node_offsets.cpu().numpy()[4:] == -1).all()
    np.testing.assert_array_equal(view.nodes[:node_offsets[-1]].cpu().numpy().view(np.uint64), nodes)
    np.testing.assert_array_equal(view.edge_dst[:6].cpu().numpy(), dst)
    np.testing.assert_array_equal(view.edge_schema[:6].cpu().numpy(), schema[:6][order])
    np.testing.assert_array_equal(view.edge_binding[:6].cpu().numpy(), binding[:6][order])


@pytest.mark.parametrize("fanout", [32, 256])
def test_builder_of_wide_fanouts_equals_the_host(fanout):
    device = gpu()
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain flags) (:requirements :strips :typing)
      (:types item) (:predicates (set ?x - item))
      (:action flip :parameters (?x - item) :precondition (and) :effect (set ?x)))""")
    objects = " ".join(f"o{k}" for k in range(fanout))
    task = mymyr.Task(domain.instantiate_string(
        f"(define (problem p) (:domain flags) (:objects {objects} - item) (:init) (:goal (and)))"))
    rows = np.asarray(task.encode([task.initial_state] * 16))
    words = tensor(rows, device)
    for merge in (False, True):
        view = encoding.horizon(task, words, depth=1, merge=merge)
        host = encoding.horizon(task, rows, depth=1, merge=merge)
        for name in ("node_offsets", "edge_offsets", "edge_src", "edge_dst", "edge_schema", "edge_binding"):
            np.testing.assert_array_equal(getattr(view, name).cpu().numpy(), np.asarray(getattr(host, name)), err_msg=name)
        count = int(view.node_offsets[-1].item())
        np.testing.assert_array_equal(view.nodes[:count].cpu().numpy().view(np.uint64), np.asarray(host.nodes))
