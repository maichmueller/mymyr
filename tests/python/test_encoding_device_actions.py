"""Canonical action labels and depth-one device horizons equal their host encodings."""

from types import SimpleNamespace

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl
from conftest import text_task, walk

torch = pytest.importorskip("torch")


def gpu():
    if not torch.cuda.is_available():
        pytest.skip("no visible CUDA device")
    pytest.importorskip("mymyr.cuda")
    return torch.device("cuda", 0)


def tensor(a, device):
    a = np.asarray(a)
    if a.dtype == np.uint64:
        a = a.view(np.int64)
    return torch.from_numpy(a.copy()).to(device)


def same(host, device):
    expected, actual = host.as_dict(), device.as_dict("numpy")
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def caps(batch):
    return dict(graphs=batch.num_graphs, entities=len(batch.entity_ref), args=len(batch.relation_args),
                targets=len(batch.target_entity) if "target_entity" in batch.as_dict() else 0)


def upload(view, device):
    fields = ("nodes", "node_offsets", "edge_offsets", "edge_src", "edge_dst", "edge_schema", "edge_binding", "task_ids")
    return encoding.HorizonView(**{k: None if getattr(view, k) is None else tensor(getattr(view, k), device)
                                   for k in fields})


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("count", [1, 5])
@pytest.mark.parametrize("name", ["blocks__probBLOCKS-8-0", "philosophers__p03-phil4"])
def test_expansion_action_entities_targets_and_capacity_match_host(layout, index, count, name):
    device = gpu()
    task = text_task(name)
    states = walk(task, steps=5, walks=1)[:count]
    rows = np.asarray(task.encode(states))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, index_dtype=index,
        target_sources=("goal", "action"), predicate_virtual_nodes=True,
        goal_derivations=("plain", "satisfied", "unsatisfied"))
    expected = encoder.encode(rows, actions=rl.expand(task, rows))
    words = tensor(rows, device)
    actions = rl.expand(task, words, framework="torch")
    same(expected, encoder.encode(words, actions=actions))
    same(expected, encoder.encode(words, actions=actions, capacities=caps(expected)))


def split_task():
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    domain = mymyr.Domain.from_string("""(define (domain split) (:requirements :strips :typing :disjunctive-preconditions)
      (:types item) (:predicates (p ?x - item) (q ?x - item) (done ?x - item) (flag))
      (:action go :parameters (?x - item) :precondition (or (p ?x) (q ?x)) :effect (done ?x))
      (:action toggle :parameters () :precondition (and) :effect (flag)))""")
    return mymyr.Task(domain.instantiate_string("""(define (problem p) (:domain split) (:objects a b c - item)
      (:init (p a) (q a) (p b) (q c)) (:goal (done a)))"""))


@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_split_schema_merge_is_sorted_and_keeps_the_lowest_schema_row(layout):
    device = gpu()
    task = split_task()
    rows = np.asarray(task.encode([task.initial_state] * 3))
    encoder = encoding.FlatRelationEncoder(task, layout=layout, target_sources=("action",))
    actions = rl.expand(task, rows)
    expected = encoder.encode(rows, actions=actions)
    actual = encoder.encode(tensor(rows, device), actions=rl.expand(task, tensor(rows, device), framework="torch"))
    same(expected, actual)
    go = encoder.relation_names.index("go")
    assert np.all(np.asarray(expected.relation_counts)[:, go] == 3)


@pytest.mark.parametrize("layout", ["graph", "relation"])
@pytest.mark.parametrize("index", ["int32", "int64"])
@pytest.mark.parametrize("mode", ["full", "delta", "action"])
@pytest.mark.parametrize("root", ["include", "exclude", "encode_only"])
def test_depth_one_horizon_modes_roots_kin_and_goals_match_host(layout, index, mode, root):
    device = gpu()
    task = text_task("philosophers__p03-phil4")
    rows = np.asarray(task.encode(walk(task, steps=2, walks=1)))
    view = encoding.horizon(task, rows, depth=1)
    goals = ("plain", "satisfied", "added_satisfied", "added_unsatisfied") if mode == "delta" else ("plain", "satisfied", "unsatisfied")
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode=mode, root_policy=root, layout=layout, index_dtype=index,
        parent_relation=True, sibling_relation=True, cousin_relation=True, predicate_virtual_nodes=True,
        goal_derivations=goals)
    expected = encoder.encode(view)
    uploaded = upload(view, device)
    same(expected, encoder.encode(uploaded))
    same(expected, encoder.encode(uploaded, capacities=caps(expected)))


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("layout", ["graph", "relation"])
def test_device_builder_and_transition_encoders_match_host(merge, layout):
    device = gpu()
    task = text_task("blocks__probBLOCKS-8-0")
    rows = np.asarray(task.encode(walk(task, steps=3, walks=1)))
    words = tensor(rows, device)
    view = encoding.horizon(task, words, depth=1, merge=merge)
    host = encoding.horizon(task, rows, depth=1, merge=merge)
    for name in ("node_offsets", "edge_offsets", "edge_src", "edge_dst", "edge_schema", "edge_binding"):
        np.testing.assert_array_equal(getattr(view, name).cpu().numpy(), np.asarray(getattr(host, name)))
    count = int(view.node_offsets[-1].item())
    np.testing.assert_array_equal(view.nodes[:count].cpu().numpy().view(np.uint64), np.asarray(host.nodes))
    for cls in (encoding.FlatTransitionEncoder, encoding.FlatTransitionEffectsEncoder):
        encoder = cls(task, layout=layout, parent_relation=True)
        same(encoder.encode(rows), encoder.encode(words))


@pytest.mark.parametrize("kind", ["actions", "horizons", "transitions"])
def test_mixed_instance_batches_match_host(kind):
    device = gpu()
    from conftest import ROOT
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
    words, device_ids = tensor(rows, device), tensor(ids, device)
    if kind == "actions":
        encoder = encoding.FlatRelationEncoder(table, target_sources=("action",))
        same(encoder.encode(rows, ids, actions=rl.expand(table, rows, ids)),
             encoder.encode(words, device_ids, actions=rl.expand(table, words, device_ids, framework="torch")))
    elif kind == "horizons":
        encoder = encoding.FlatHorizonEncoder(table, horizon_mode="delta", parent_relation=True)
        view = encoding.horizon(table, rows, task_ids=ids)
        same(encoder.encode(view), encoder.encode(upload(view, device)))
    else:
        encoder = encoding.FlatTransitionEffectsEncoder(table)
        same(encoder.encode(rows, ids), encoder.encode(words, device_ids))


def test_self_loop_has_no_delta_and_deeper_device_horizons_are_refused():
    device = gpu()
    task = text_task("blocks__probBLOCKS-8-0")
    row = np.asarray(task.encode([task.initial_state]))
    x = rl.expand(task, row)
    view = encoding.HorizonView(np.concatenate((row, row)), np.array([0, 2], np.int32), np.array([0, 1], np.int32),
        np.array([0], np.int32), np.array([1], np.int32), np.asarray(x.schema)[:1], np.asarray(x.binding)[:1])
    encoder = encoding.FlatHorizonEncoder(task, horizon_mode="delta", root_policy="exclude", goal_derivations=())
    expected = encoder.encode(view)
    same(expected, encoder.encode(upload(view, device)))
    assert all(expected.relation_counts[0, i] == 0 for i, lane in enumerate(encoder.relation_lanes) if lane == "delta")
    with pytest.raises(ValueError, match="horizons of depth 2 or more"):
        encoding.horizon(task, tensor(row, device), depth=2)
    deeper = encoding.horizon(task, row, depth=2, merge=False)
    with pytest.raises(ValueError, match="horizons of depth 2 or more"):
        encoder.encode(upload(deeper, device))


@pytest.mark.needs_gil
def test_torch_op_actions_and_horizons_compile_and_capture():
    device = gpu()
    import mymyr.rl.torch as rt
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode(walk(task, steps=2, walks=1)))
    words = tensor(rows, device)
    encoder = encoding.FlatRelationEncoder(task, target_sources=("action",))
    x = rl.expand(task, words, framework="torch")
    expected = encoder.encode(rows, actions=rl.expand(task, rows))
    handle = rt.register(encoder)
    capacity = caps(expected)
    def call():
        return torch.ops.mymyr.encode_flat(handle, words, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            len(rows), capacity["entities"], capacity["args"], capacity["targets"], 0,
            x.schema, x.binding, x.offsets, x.parent)
    try:
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, call()))
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, torch.compile(call, fullgraph=True)()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            arrays = call()
        graph.replay()
        same(expected, encoding.DeviceFlatBatch.from_arrays(encoder, arrays))
    finally:
        rt.unregister(handle)


def test_environment_transition_encoding_equals_host():
    device = gpu()
    import mymyr.rl.torch as rt
    task = text_task("gripper__prob05")
    env = rt.BatchedEnv(task, 3, device=device)
    encoder = encoding.FlatTransitionEffectsEncoder(task)
    expected = encoder.encode(env.states.cpu(), env.task_ids.cpu())
    same(expected, env.encode_transitions(encoder))


def test_jax_ffi_refuses_action_rows_and_horizons_with_the_operand_reason():
    pytest.importorskip("jax")
    from mymyr.encoding import jax as ej
    task = text_task("gripper__prob05")
    rows = np.asarray(task.encode([task.initial_state]))
    with pytest.raises(ValueError, match="operands carry no expansion labels or offsets"):
        ej.encode(encoding.FlatRelationEncoder(task), rows, actions=object())
    with pytest.raises(ValueError, match="operands carry no node offsets or edges"):
        ej.encode(encoding.FlatTransitionEncoder(task), rows)
